#!/usr/bin/env python3
"""Walk-forward backtest harness for Express Entry CRS cut-off predictors.

Standard library only (Python 3.10+; written for 3.14, no numpy/scipy).
Data: IRCC's official rounds JSON (ee_rounds_123_en.json), treated as
untrusted input and only ever read.  Modeller's guide: README.md (same folder).

Command line (always run with -I):
  python3 -I harness.py --check                        data report, type map, sanity numbers
  python3 -I harness.py --selftest                     helper + leakage self-tests
  python3 -I harness.py approach_x.py [more.py ...]    the three default windows
  python3 -I harness.py approach_x.py --start 2026-01-01 --end 2026-12-31 [--types cec,pnp]
  options: --data PATH  --strict  --rows  --out-dir DIR  --no-write

Approach module interface:
  NAME = "optional label"
  def predict(ctx, draw_type, n) -> {"point": x, "low": lo (optional), "high": hi (optional)}

Python API (import harness):
  load_rounds(path=None)                 -> list of round dicts, ascending draw number
  canonical_type(name)                   -> one of TYPES
  build_ctx(rounds, target)              -> ctx for predicting `target` (no leakage)
  latest_ctx(path=None, target_date=None)-> ctx for the next draw after the newest round
  evaluate(predict_fn, start, end, types=None, rounds=None, path=None, strict=False)
  cumulative_above(dist, score)          -> candidates with CRS >= score (uniform within band)
  score_for_count(dist, count, integer=True) -> cut-off when the top `count` are invited
  same_type_history(ctx, draw_type, k=None)  -> last k same-type rounds, oldest first
  days_between(a, b)                     -> (b - a) in calendar days
  extras: density, remove_top, remove_above, dist_total, BANDS, BAND_KEYS, BAND_RANGE
"""

import argparse
import datetime as _dt
import importlib.util
import json
import math
import os
import re
import statistics
import sys
import traceback

HARNESS_VERSION = "1.0"
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.environ.get("EE_ROUNDS_JSON") or os.path.normpath(
    os.path.join(HERE, "..", "ircc", "ee_rounds_123_en.json"))

MIN_SCORE, MAX_SCORE = 0, 1200

# Finest disjoint CRS bands published by IRCC, highest first: (key, low, high, json field).
# dd3 (451-500) and dd9 (401-450) are subtotals; dd18 is the pool total.
BANDS = (
    ("601-1200", 601, 1200, "dd1"),
    ("501-600", 501, 600, "dd2"),
    ("491-500", 491, 500, "dd4"),
    ("481-490", 481, 490, "dd5"),
    ("471-480", 471, 480, "dd6"),
    ("461-470", 461, 470, "dd7"),
    ("451-460", 451, 460, "dd8"),
    ("441-450", 441, 450, "dd10"),
    ("431-440", 431, 440, "dd11"),
    ("421-430", 421, 430, "dd12"),
    ("411-420", 411, 420, "dd13"),
    ("401-410", 401, 410, "dd14"),
    ("351-400", 351, 400, "dd15"),
    ("301-350", 301, 350, "dd16"),
    ("0-300", 0, 300, "dd17"),
)
BAND_KEYS = tuple(b[0] for b in BANDS)
BAND_RANGE = {b[0]: (b[1], b[2]) for b in BANDS}
_SUBTOTALS = (
    ("dd3", "451-500", ("dd4", "dd5", "dd6", "dd7", "dd8")),
    ("dd9", "401-450", ("dd10", "dd11", "dd12", "dd13", "dd14")),
)

TYPES = ("pnp", "cec", "french", "healthcare", "trades", "stem", "education", "transport",
         "agriculture", "senior_managers", "physicians", "military", "general", "fsw", "fst",
         "other")
MAJOR_TYPES = ("cec", "pnp", "french", "healthcare", "trades")

DEFAULT_WINDOWS = (
    {"name": "tune_2025", "start": "2025-01-01", "end": "2025-12-31", "types": None},
    {"name": "holdout_2026", "start": "2026-01-01", "end": "2026-10-16", "types": None},
    {"name": "general_2023_2024", "start": "2023-01-01", "end": "2024-12-31",
     "types": ["general"]},
)

# --------------------------------------------------------------------------------------
# Draw types
# --------------------------------------------------------------------------------------
# Ordered: specific rules first ("Federal Skilled Trades" must not become "trades";
# "Physicians/Senior managers with Canadian Work Experience" must not become "cec").
_TYPE_RULES = (
    ("physicians", re.compile(r"physician")),
    ("senior_managers", re.compile(r"senior\s+manager")),
    ("military", re.compile(r"militar")),
    ("french", re.compile(r"french|francophone")),
    ("healthcare", re.compile(r"health")),
    ("fst", re.compile(r"federal\s+skilled\s+trades?\b")),
    ("fsw", re.compile(r"federal\s+skilled\s+workers?\b")),
    ("trades", re.compile(r"\btrades?\b")),
    ("stem", re.compile(r"\bstem\b")),
    ("education", re.compile(r"educat")),
    ("transport", re.compile(r"transport")),
    ("agriculture", re.compile(r"agri")),
    ("cec", re.compile(r"canadian\s+experience\s+class")),
    ("pnp", re.compile(r"provincial\s+nominee")),
    ("general", re.compile(r"no\s+program\s+specified|^general\b|\ball\s+programs?\b")),
)


def canonical_type(name):
    """Map an IRCC drawName to one of TYPES ('other' if nothing matches)."""
    s = re.sub(r"\s+", " ", str(name or "")).strip().lower()
    for t, rx in _TYPE_RULES:
        if rx.search(s):
            return t
    return "other"


def category_version(name):
    """'French language proficiency (Version 1)' -> 'v1'; '... 2026-Version 2' -> '2026-v2'."""
    m = re.search(r"(?:(\d{4})\s*-\s*)?version\s*(\d+)", str(name or ""), re.I)
    if not m:
        return None
    return ("%s-v%s" % (m.group(1), m.group(2))) if m.group(1) else ("v%s" % m.group(2))


# --------------------------------------------------------------------------------------
# Parsing helpers (robust to commas, blanks, ordinals, missing years/time zones, AM/PM)
# --------------------------------------------------------------------------------------
_INT_JUNK = re.compile(r"[\s,\u00a0\u2009\u202f]")


def parse_int(value):
    """'21,638' -> 21638; ints pass through; blanks/odd values -> None."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    t = _INT_JUNK.sub("", str(value))
    return int(t) if re.fullmatch(r"\d+", t) else None


_MONTHS = {}
for _i, _m in enumerate(("january", "february", "march", "april", "may", "june", "july",
                         "august", "september", "october", "november", "december"), 1):
    _MONTHS[_m] = _i
    _MONTHS[_m[:3]] = _i
_MONTHS["sept"] = 9
_DATE_RE = re.compile(r"([A-Za-z]+)\.?\s*(\d{1,2})(?:st|nd|rd|th)?\b\s*[,.]?\s*(\d{4})?")
_ISO_RE = re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})")
_TIME_RE = re.compile(r"(\d{1,2}):(\d{2})(?::(\d{2}))?(?:\s*([AaPp])\.?\s*[Mm]\.?)?")
_TZ_RE = re.compile(r"\b(UTC|GMT|Z)\b")
_CANON_DATE = re.compile(r"[A-Z][a-z]+ \d{1,2}, \d{4}")
_CANON_DT = re.compile(r"[A-Z][a-z]+ \d{1,2}, \d{4} at \d{1,2}:\d{2}:\d{2} UTC")


def _parse_date_part(s, ref_date=None):
    """Find the calendar date in an IRCC string. Returns (date or None, notes)."""
    notes = []
    for m in _DATE_RE.finditer(s):
        mon = _MONTHS.get(m.group(1).lower())
        if not mon:
            continue
        day = int(m.group(2))
        if m.group(3):
            year = int(m.group(3))
        elif ref_date is not None:
            year = ref_date.year
            try:
                if _dt.date(year, mon, day) > ref_date:
                    year -= 1
            except ValueError:
                pass
            notes.append("year missing, inferred %d from draw date" % year)
        else:
            return None, ["year missing and no reference date"]
        try:
            return _dt.date(year, mon, day), notes
        except ValueError:
            return None, ["impossible date %d-%02d-%02d" % (year, mon, day)]
    m = _ISO_RE.search(s)
    if m:
        try:
            return _dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3))), notes
        except ValueError:
            pass
    return None, ["no date found"]


def parse_ircc_date(s, ref_date=None):
    """'October 12, 2026' (and sloppy variants) -> (date or None, notes)."""
    if s is None or not str(s).strip():
        return None, ["blank"]
    s = str(s)
    d, notes = _parse_date_part(s, ref_date)
    if d is not None and not _CANON_DATE.fullmatch(s):
        notes.insert(0, "irregular format")
    return d, notes


def parse_ircc_datetime(s, ref_date=None):
    """'October 06, 2026 at 21:31:22 UTC' (and the ~50 sloppy variants in the file)
    -> (timezone-aware UTC datetime or None, notes)."""
    if s is None or not str(s).strip():
        return None, ["blank"]
    s = str(s)
    d, notes = _parse_date_part(s, ref_date)
    if d is None:
        return None, notes
    times = list(_TIME_RE.finditer(s))
    if not times:
        notes.append("no time, used 00:00:00")
        hh = mm = ss = 0
    else:
        t = times[-1]
        hh, mm, ss = int(t.group(1)), int(t.group(2)), int(t.group(3) or 0)
        ampm = (t.group(4) or "").lower()
        if ampm:
            if 1 <= hh <= 12:
                if ampm == "p" and hh < 12:
                    hh += 12
                elif ampm == "a" and hh == 12:
                    hh = 0
            else:
                notes.append("24-hour time with AM/PM marker, marker ignored")
        if hh > 23 or mm > 59 or ss > 59:
            return None, notes + ["impossible time"]
    if not _TZ_RE.search(s):
        notes.append("no time zone, assumed UTC")
    if not _CANON_DT.fullmatch(s):
        notes.insert(0, "irregular format")
    return _dt.datetime(d.year, d.month, d.day, hh, mm, ss, tzinfo=_dt.timezone.utc), notes


def _as_date(x):
    if x is None:
        return None
    if isinstance(x, _dt.datetime):
        return x.date()
    if isinstance(x, _dt.date):
        return x
    return _dt.date.fromisoformat(str(x).strip()[:10])


def days_between(a, b):
    """Calendar days from a to b (b - a; positive when b is later). Accepts date,
    datetime (date part is used) or 'YYYY-MM-DD' strings."""
    return (_as_date(b) - _as_date(a)).days


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def _parse_dist(raw, issues):
    vals = {"dd%d" % i: parse_int(raw.get("dd%d" % i)) for i in range(1, 19)}
    bad = [k for k, v in vals.items() if v is None]
    if bad:
        issues.append("[invalid] distribution fields not integers: " + ", ".join(
            "%s=%r" % (k, raw.get(k)) for k in bad))
    fine = {key: vals[f] for key, _lo, _hi, f in BANDS}
    if any(v is None for v in fine.values()):
        return None, vals["dd18"]
    if all(v == 0 for v in fine.values()):
        if vals["dd18"]:
            issues.append("[invalid] all bands 0 but dd18=%s" % vals["dd18"])
        issues.append("[missing] no distribution published (all dd fields are 0)")
        return None, vals["dd18"]
    for sub, label, parts in _SUBTOTALS:
        if vals[sub] is not None and all(vals[p] is not None for p in parts):
            s = sum(vals[p] for p in parts)
            if s != vals[sub]:
                issues.append("[invalid] %s (%s subtotal) = %s but %s..%s sum to %s (diff %+d); "
                              "harness uses the fine bands" % (sub, label, format(vals[sub], ","),
                                                             parts[0], parts[-1], format(s, ","),
                                                             vals[sub] - s))
    tot = sum(fine.values())
    if vals["dd18"] is not None and vals["dd18"] != tot:
        issues.append("[invalid] dd18 total = %s but the disjoint bands sum to %s (diff %+d)"
                      % (format(vals["dd18"], ","), format(tot, ","), vals["dd18"] - tot))
    return fine, vals["dd18"]


def _parse_round(raw):
    issues = []
    label = str(raw.get("drawNumber", "")).strip()
    m = re.match(r"(\d+)\s*([A-Za-z]*)$", label)
    number = int(m.group(1)) if m else None
    suffix = m.group(2).lower() if m else ""
    if number is None:
        issues.append("[invalid] drawNumber %r is not a number" % label)
    elif suffix:
        issues.append("[note] drawNumber %r carries a suffix; number=%d" % (label, number))

    draw_date = None
    try:
        draw_date = _dt.date.fromisoformat(str(raw.get("drawDate", "")).strip())
    except ValueError:
        draw_date, notes = parse_ircc_date(raw.get("drawDateFull"))
        issues.append("[repair] drawDate %r unusable; drawDateFull -> %s %s"
                      % (raw.get("drawDate"), draw_date, notes))
    if draw_date is None:
        issues.append("[invalid] no usable draw date")

    draw_dt, notes = parse_ircc_datetime(raw.get("drawDateTime"), draw_date)
    if notes:
        tag = "[invalid]" if draw_dt is None else "[repair]"
        issues.append("%s drawDateTime %r -> %s (%s)" % (
            tag, raw.get("drawDateTime"), draw_dt.isoformat() if draw_dt else None, "; ".join(notes)))
    if draw_dt and draw_date and draw_dt.date() != draw_date:
        gap = (draw_dt.date() - draw_date).days
        if abs(gap) <= 1:  # evening Ottawa time = next day UTC, or next-day publication
            issues.append("[note] drawDateTime %s is %+d day from drawDate %s"
                          % (draw_dt.isoformat(), gap, draw_date))
        else:  # typo (e.g. #357 'July 12' for a July 22 draw): keep the time, trust drawDate
            fixed = draw_dt.replace(year=draw_date.year, month=draw_date.month,
                                    day=draw_date.day)
            issues.append("[repair] drawDateTime %r is %+d days from drawDate %s; using "
                          "drawDate with its time -> %s" % (raw.get("drawDateTime"), gap,
                                                            draw_date, fixed.isoformat()))
            draw_dt = fixed

    tie = None
    if str(raw.get("drawCutOff") or "").strip():
        tie, notes = parse_ircc_datetime(raw.get("drawCutOff"), draw_date)
        if notes:
            tag = "[invalid]" if tie is None else "[repair]"
            issues.append("%s drawCutOff %r -> %s (%s)" % (
                tag, raw.get("drawCutOff"), tie.isoformat() if tie else None, "; ".join(notes)))
        if tie and draw_dt and tie > draw_dt:
            issues.append("[invalid] tie-break %s is after the draw time %s"
                          % (tie.isoformat(), draw_dt.isoformat()))
    else:
        issues.append("[missing] drawCutOff (tie-break) blank")

    size = parse_int(raw.get("drawSize"))
    crs = parse_int(raw.get("drawCRS"))
    if size is None:
        issues.append("[invalid] drawSize %r" % raw.get("drawSize"))
    if crs is None:
        issues.append("[invalid] drawCRS %r" % raw.get("drawCRS"))
    elif not MIN_SCORE <= crs <= MAX_SCORE:
        issues.append("[invalid] drawCRS %d outside 0..1200" % crs)

    dist, reported_total = _parse_dist(raw, issues)
    as_on = None
    as_on_raw = raw.get("drawDistributionAsOn")
    if dist is not None:
        as_on, notes = parse_ircc_date(as_on_raw, draw_date)
        if as_on is None:
            issues.append("[invalid] drawDistributionAsOn %r unparseable (%s); distribution "
                          "dropped" % (as_on_raw, "; ".join(notes)))
            dist = None
        else:
            if notes:
                issues.append("[repair] drawDistributionAsOn %r -> %s (%s)"
                              % (as_on_raw, as_on, "; ".join(notes)))
            if draw_date and as_on > draw_date:
                issues.append("[invalid] distribution as-on %s is after the draw date %s"
                              % (as_on, draw_date))
    elif str(as_on_raw or "").strip() and draw_date:
        d, _n = parse_ircc_date(as_on_raw, draw_date)
        if d and d > draw_date:
            issues.append("[note] as-on %s after draw date %s (no distribution, ignored)"
                          % (d, draw_date))

    name = re.sub(r"\s+", " ", str(raw.get("drawName") or "")).strip()
    return {
        "number": number,
        "label": label,
        "suffix": suffix,
        "seq": None,
        "draw_date": draw_date,
        "draw_datetime": draw_dt,
        "name": name,
        "type": canonical_type(name),
        "version": category_version(name),
        "programs": re.sub(r"\s+", " ", str(raw.get("drawText2") or "")).strip(),
        "size": size,
        "crs": crs,
        "tie_break": tie,
        "dist_as_on": as_on,
        "dist": dist,
        "dist_total": sum(dist.values()) if dist else None,
        "dist_reported_total": reported_total if dist else None,
        "issues": issues,
    }


def load_rounds(path=None):
    """Parse IRCC's rounds JSON into a list of round dicts sorted by draw number ascending
    (91a before 91b). Every anomaly is recorded in round['issues'] (see data_report())."""
    path = os.path.abspath(path or DEFAULT_DATA)
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    raw_rounds = data.get("rounds", []) if isinstance(data, dict) else data
    if isinstance(raw_rounds, dict):
        raw_rounds = list(raw_rounds.values())
    rounds = [_parse_round(r) for r in raw_rounds if isinstance(r, dict)]
    far = _dt.datetime(9999, 1, 1, tzinfo=_dt.timezone.utc)
    rounds.sort(key=lambda r: (r["number"] if r["number"] is not None else 10 ** 9,
                               r["suffix"], r["draw_datetime"] or far))
    for i, r in enumerate(rounds):
        r["seq"] = i
    return rounds


# --------------------------------------------------------------------------------------
# Distribution helpers
# --------------------------------------------------------------------------------------
def dist_total(x):
    """Total candidates in a band dict or a density list."""
    return float(sum(x)) if isinstance(x, list) else sum(x.get(k, 0) or 0 for k in BAND_KEYS)


def density(dist):
    """Band dict -> list of 1201 floats, expected candidates at each integer score 0..1200
    (each band spread uniformly over its integer scores, the same assumption as
    cumulative_above). All helpers below accept either representation."""
    out = [0.0] * (MAX_SCORE + 1)
    for key, lo, hi, _f in BANDS:
        c = dist.get(key, 0) or 0
        if c:
            per = c / (hi - lo + 1)
            for s in range(lo, hi + 1):
                out[s] += per
    return out


def cumulative_above(dist, score):
    """Candidates with CRS >= score. `dist` is a band dict (keys from BAND_KEYS) or a
    density list. Within a band, counts are spread uniformly over its integer scores
    (501-600 holds 100 scores, 601-1200 holds 600, 0-300 holds 301). Non-integer scores
    interpolate linearly, so the function is continuous and non-increasing."""
    if isinstance(dist, list):
        if score <= 0:
            return float(sum(dist))
        if score >= len(dist):
            return 0.0
        i = int(math.ceil(score))
        total = float(sum(dist[i:]))
        if i != score:  # fractional part of the score below i
            total += dist[i - 1] * (i - score)
        return total
    total = 0.0
    for key, lo, hi, _f in BANDS:
        c = dist.get(key, 0) or 0
        if c <= 0:
            continue
        frac = (hi + 1 - score) / (hi - lo + 1)
        if frac >= 1:
            total += c
        elif frac > 0:
            total += c * frac
    return total


def score_for_count(dist, count, integer=True):
    """Cut-off when the top `count` candidates of `dist` are invited: the largest integer
    score s with cumulative_above(dist, s) >= count. Returns 1200 for count <= 0 and 0 when
    count exceeds the whole pool. integer=False returns the real-valued s where the
    piecewise-linear cumulative_above equals count (floor of it == integer answer)."""
    if count <= 0:
        return MAX_SCORE
    if cumulative_above(dist, MIN_SCORE) < count:
        return MIN_SCORE
    lo, hi = MIN_SCORE, MAX_SCORE
    while lo < hi:  # invariant: cumulative_above(dist, lo) >= count
        mid = (lo + hi + 1) // 2
        if cumulative_above(dist, mid) >= count:
            lo = mid
        else:
            hi = mid - 1
    if integer:
        return lo
    a, b = cumulative_above(dist, lo), cumulative_above(dist, lo + 1)
    if a <= b:
        return float(lo)
    return lo + min(1.0, max(0.0, (a - count) / (a - b)))


def remove_top(dist, count):
    """Density list after removing the `count` highest-scoring candidates (e.g. a PNP or
    all-program draw). Accepts a band dict or density list; always returns a new list."""
    d = list(dist) if isinstance(dist, list) else density(dist)
    rem = float(count)
    for s in range(MAX_SCORE, MIN_SCORE - 1, -1):
        if rem <= 0:
            break
        take = min(d[s], rem)
        d[s] -= take
        rem -= take
    return d


def remove_above(dist, score, count):
    """Density list after removing `count` candidates spread proportionally over all
    scores >= `score` (e.g. a CEC or category draw that invited only part of the people
    above its cut-off). Returns a new list."""
    d = list(dist) if isinstance(dist, list) else density(dist)
    start = max(MIN_SCORE, int(math.ceil(score)))
    avail = sum(d[start:])
    if avail <= 0:
        return d
    f = min(1.0, count / avail)
    for s in range(start, MAX_SCORE + 1):
        d[s] *= (1.0 - f)
    return d


# --------------------------------------------------------------------------------------
# Prediction context (walk-forward, leakage-free)
# --------------------------------------------------------------------------------------
def _dist_key(d):
    return tuple(d.get(k) for k in BAND_KEYS)


def _copy_round(r):
    c = dict(r)
    if r["dist"] is not None:
        c["dist"] = dict(r["dist"])
    c["issues"] = list(r["issues"])
    return c


def find_snapshot(history):
    """The newest published pool distribution among `history` rounds.

    IRCC attaches one PRE-draw snapshot to every draw of a cycle (the 601+ count is about
    the size of the cycle's PNP draw), so several rounds report the same numbers. The
    snapshot is the one with the latest as-on date (ties -> most recent round; in this
    data that is always the most recent round with a distribution). 'reported_by' lists
    every history round reporting identical numbers; 'dist_as_on' is the earliest as-on
    date they state (#386 says Dec 15, 2025, #387/#388 say Dec 14 for the same numbers)."""
    valid = [h for h in history if h["dist"] is not None and h["dist_as_on"] is not None]
    if not valid:
        return None
    s = max(valid, key=lambda h: (h["dist_as_on"], h["seq"]))
    key = _dist_key(s["dist"])
    reporters = [h for h in valid if _dist_key(h["dist"]) == key]
    as_ons = sorted({h["dist_as_on"] for h in reporters})
    return {
        "dist": dict(s["dist"]),
        "dist_as_on": as_ons[0],
        "total": sum(s["dist"].values()),
        "round": s["number"],
        "reported_by": [h["number"] for h in reporters],
        "as_on_dates_stated": as_ons,
        "_reporter_seqs": {h["seq"] for h in reporters},
    }


def _same_day_reflected(h, snap):
    """Heuristic for a draw dated ON the snapshot's as-on date that itself reported an older
    snapshot: was it already removed from the snapshot? Compare the count above its cut-off
    (the 601+ band for PNP-style cut-offs) in its own pre-draw snapshot vs this one."""
    own, cur = h["dist"], snap["dist"]
    if not own or h["crs"] is None:
        return False
    if h["crs"] >= 601:
        a, b = own["601-1200"], cur["601-1200"]
    else:
        a, b = cumulative_above(own, h["crs"]), cumulative_above(cur, h["crs"])
    if a <= 0:
        return False
    return (a - b) >= 0.5 * min(h["size"] or 0, a)


def invited_since(history, snap):
    """History draws NOT yet reflected in the snapshot, i.e. whose invitees are still in it.
    Rule: drawn after the as-on date -> included ('after_as_on'); drawn before -> excluded;
    drawn on the as-on date -> included if the draw reported this very snapshot
    ('reported_this_snapshot', e.g. 2023 rounds whose as-on equals the draw date), else
    decided by _same_day_reflected (3 cases, all 2024: #296 included; #322, #325 excluded).
    Returns (included, excluded_same_day)."""
    if snap is None:
        return [], []
    A = snap["dist_as_on"]
    reps = snap["_reporter_seqs"]
    inc, exc = [], []
    for h in history:
        d = h["draw_date"]
        if d is None or d < A:
            continue
        if d > A:
            basis = "after_as_on"
        elif h["seq"] in reps:
            basis = "reported_this_snapshot"
        elif h["dist"] is not None and h["dist_as_on"] is not None and h["dist_as_on"] < A:
            if _same_day_reflected(h, snap):
                exc.append(_brief(h, "same_day_reflected"))
                continue
            basis = "same_day_not_reflected"
        else:
            basis = "same_day_assumed"
        inc.append(_brief(h, basis))
    return inc, exc


def _brief(h, basis):
    return {"number": h["number"], "label": h["label"], "type": h["type"], "name": h["name"],
            "size": h["size"], "crs": h["crs"], "date": h["draw_date"],
            "draw_date": h["draw_date"], "basis": basis}


def make_ctx(history, target_date):
    """Build a ctx from an already-filtered history list (rounds strictly before the target)."""
    target_date = _as_date(target_date)
    snap = find_snapshot(history)
    inc, exc = invited_since(history, snap)
    if snap is not None:
        snap = {k: v for k, v in snap.items() if not k.startswith("_")}
    last = history[-1] if history else None
    return {
        "history": history,
        "snapshot": snap,
        "invited_since_snapshot": inc,
        "same_day_reflected": exc,
        "target_date": target_date,
        "days_since_snapshot": (target_date - snap["dist_as_on"]).days if snap else None,
        "last_draw_date": last["draw_date"] if last else None,
    }


def build_ctx(rounds, target):
    """ctx for predicting `target` (a round dict from load_rounds). history = copies of all
    rounds with drawNumber < target's number; nothing from the target or later leaks."""
    hist = [_copy_round(r) for r in rounds
            if r["number"] is not None and r["number"] < target["number"]]
    return make_ctx(hist, target["draw_date"])


def latest_ctx(path=None, target_date=None, rounds=None):
    """ctx for the NEXT draw after the newest round in the file (history = all rounds).
    target_date defaults to max(today, newest draw date + 1 day)."""
    rounds = rounds if rounds is not None else load_rounds(path)
    hist = [_copy_round(r) for r in rounds]
    if target_date is None:
        target_date = max(_dt.date.today(), hist[-1]["draw_date"] + _dt.timedelta(days=1))
    return make_ctx(hist, target_date)


def same_type_history(ctx, draw_type, k=None):
    """The last k history rounds of `draw_type` (a type string or a collection of them),
    oldest first / newest last ([-1] is the most recent). k=None returns all of them."""
    want = {draw_type} if isinstance(draw_type, str) else set(draw_type)
    out = [h for h in ctx["history"] if h["type"] in want]
    return out if k is None else out[-k:] if k > 0 else []


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------
def _num(v, field):
    if v is None:
        return None
    if isinstance(v, bool):
        raise TypeError("%s must be a number, got %r" % (field, v))
    if not isinstance(v, (int, float)):
        try:
            v = float(v)
        except (TypeError, ValueError):
            raise TypeError("%s must be a number, got %r" % (field, v))
    if not math.isfinite(v):
        raise ValueError("%s is not finite: %r" % (field, v))
    return v


def normalize_prediction(out):
    """Accept {"point", "low"?, "high"?, "info"?} or a bare number. Returns
    (point, low, high, info, warnings)."""
    warnings = []
    if isinstance(out, dict):
        point = _num(out.get("point"), "point")
        low = _num(out.get("low"), "low")
        high = _num(out.get("high"), "high")
        info = out.get("info")
    elif out is None:
        raise ValueError("predict returned None")
    else:
        point, low, high, info = _num(out, "point"), None, None, None
    if point is None:
        raise ValueError("prediction has no 'point'")
    if (low is None) != (high is None):
        warnings.append("only one of low/high given; interval ignored")
        low = high = None
    if low is not None and low > high:
        warnings.append("low > high; swapped")
        low, high = high, low
    if low is not None and not (low <= point <= high):
        warnings.append("point outside [low, high]")
    return point, low, high, info, warnings


def evaluate(predict_fn, start, end, types=None, rounds=None, path=None, strict=False):
    """Walk forward over target draws dated in [start, end] (inclusive), optionally only
    those whose canonical type is in `types`. For each target T: ctx = build_ctx(rounds, T);
    out = predict_fn(ctx, T.type, T.size). Exceptions are recorded per row (and counted as
    n_failed) unless strict=True. Returns {"start","end","types","rows","metrics"}."""
    rounds = rounds if rounds is not None else load_rounds(path)
    start, end = _as_date(start), _as_date(end)
    tset = None if types is None else ({types} if isinstance(types, str) else set(types))
    rows = []
    for T in rounds:
        d = T["draw_date"]
        if d is None or d < start or d > end:
            continue
        if tset is not None and T["type"] not in tset:
            continue
        row = {"number": T["number"], "label": T["label"], "date": d.isoformat(),
               "type": T["type"], "name": T["name"], "size": T["size"], "actual": T["crs"],
               "point": None, "low": None, "high": None, "err": None, "abs_err": None,
               "error": None, "warnings": []}
        if T["crs"] is None or T["size"] is None or T["number"] is None:
            row["error"] = "target not scorable (missing size, cut-off or draw number)"
            row["skipped"] = True
            rows.append(row)
            continue
        ctx = build_ctx(rounds, T)
        try:
            point, low, high, info, warns = normalize_prediction(
                predict_fn(ctx, T["type"], T["size"]))
        except Exception as e:  # noqa: BLE001 - recorded per row
            if strict:
                raise
            row["error"] = "%s: %s" % (type(e).__name__, e)
            rows.append(row)
            continue
        row.update(point=point, low=low, high=high, err=point - T["crs"],
                   abs_err=abs(point - T["crs"]), warnings=warns)
        if info is not None:
            row["info"] = info
        rows.append(row)
    return {"start": start.isoformat(), "end": end.isoformat(),
            "types": sorted(tset) if tset else None, "rows": rows,
            "metrics": compute_metrics(rows)}


def _summ(rows):
    scored = [r for r in rows if not r.get("skipped")]
    ok = [r for r in scored if r["point"] is not None]
    m = {"n": len(ok), "n_failed": len(scored) - len(ok)}
    if not ok:
        return m
    e = [r["point"] - r["actual"] for r in ok]
    a = [abs(x) for x in e]
    m.update(mae=statistics.fmean(a), medae=statistics.median(a),
             rmse=math.sqrt(statistics.fmean([x * x for x in e])), bias=statistics.fmean(e),
             max_abs=max(a), within5=statistics.fmean([1.0 if x <= 5 else 0.0 for x in a]))
    iv = [r for r in ok if r["low"] is not None and r["high"] is not None]
    m["n_interval"] = len(iv)
    if iv:
        m["coverage"] = statistics.fmean(
            [1.0 if r["low"] <= r["actual"] <= r["high"] else 0.0 for r in iv])
        m["mean_width"] = statistics.fmean([r["high"] - r["low"] for r in iv])
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}


def compute_metrics(rows):
    """Per-type metrics plus 'MAJOR' (cec, pnp, french, healthcare, trades) and 'ALL'.
    n, mae, medae, rmse, bias (mean of predicted - actual), max_abs, within5 (share with
    |error| <= 5), n_failed; with intervals: n_interval, coverage (actual in [low, high]),
    mean_width (high - low)."""
    out = {}
    for t in TYPES:
        sel = [r for r in rows if r["type"] == t]
        if sel:
            out[t] = _summ(sel)
    out["MAJOR"] = _summ([r for r in rows if r["type"] in MAJOR_TYPES])
    out["ALL"] = _summ(rows)
    return out


def format_metrics(metrics):
    head = "%-16s %4s %6s %6s %6s %7s %5s %5s %6s %6s %4s" % (
        "type", "n", "MAE", "MedAE", "RMSE", "bias", "max", "<=5", "cover", "width", "fail")
    lines = [head, "-" * len(head)]
    for key, m in metrics.items():
        if key == "MAJOR":
            lines.append("-" * len(head))
        if not m.get("n"):
            lines.append("%-16s %4d %s %4d" % (key, 0, " " * 56, m.get("n_failed", 0)))
            continue
        cov = ("%5.0f%%" % (100 * m["coverage"])) if "coverage" in m else "     -"
        wid = ("%6.1f" % m["mean_width"]) if "mean_width" in m else "     -"
        lines.append("%-16s %4d %6.1f %6.1f %6.1f %+7.1f %5.0f %4.0f%% %6s %6s %4d" % (
            key, m["n"], m["mae"], m["medae"], m["rmse"], m["bias"], m["max_abs"],
            100 * m["within5"], cov, wid, m["n_failed"]))
    return "\n".join(lines)


def format_rows(rows):
    lines = ["%5s %10s %-16s %6s %6s %8s %8s %8s %7s" % (
        "draw", "date", "type", "size", "actual", "point", "low", "high", "err")]
    for r in rows:
        f = lambda v: "-" if v is None else ("%.1f" % v)  # noqa: E731
        lines.append("%5s %10s %-16s %6s %6s %8s %8s %8s %7s%s" % (
            r["label"], r["date"], r["type"], r["size"], r["actual"], f(r["point"]),
            f(r["low"]), f(r["high"]), "-" if r["err"] is None else "%+.1f" % r["err"],
            ("   ! " + r["error"]) if r.get("error") else ""))
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Data report
# --------------------------------------------------------------------------------------
def type_map(rounds):
    """[(name, type, count, first_date, last_date)], grouped by type then count."""
    agg = {}
    for r in rounds:
        a = agg.setdefault(r["name"], [r["type"], 0, r["draw_date"], r["draw_date"]])
        a[1] += 1
        a[2] = min(a[2], r["draw_date"])
        a[3] = max(a[3], r["draw_date"])
    order = {t: i for i, t in enumerate(TYPES)}
    return sorted(((n, a[0], a[1], a[2], a[3]) for n, a in agg.items()),
                  key=lambda x: (order.get(x[1], 99), -x[2], x[0]))


def _ranges(nums):
    out = []
    for n in sorted(set(nums)):
        if out and n == out[-1][1] + 1:
            out[-1][1] = n
        else:
            out.append([n, n])
    return ", ".join(("#%d" % a) if a == b else ("#%d-#%d" % (a, b)) for a, b in out)


def same_day_cases(rounds):
    """Every (snapshot, draw dated on its as-on date that reported an older snapshot)."""
    cases, seen = [], set()
    for i in range(len(rounds)):
        r = rounds[i]
        if r["dist"] is None:
            continue
        key = _dist_key(r["dist"])
        if key in seen:
            continue
        seen.add(key)
        hist = rounds[:i + 1]
        snap = find_snapshot(hist)
        for h in hist:
            if (h["draw_date"] == snap["dist_as_on"] and h["seq"] not in snap["_reporter_seqs"]
                    and h["dist"] is not None and h["dist_as_on"] < snap["dist_as_on"]):
                cases.append((snap, h, _same_day_reflected(h, snap)))
    return cases


def data_report(rounds):
    lines = []
    tm = type_map(rounds)
    lines.append("== drawName -> canonical type (%d distinct names, %d rounds) ==" % (
        len(tm), len(rounds)))
    lines.append("%-62s %-16s %5s  %s" % ("drawName", "type", "count", "first .. last"))
    for name, t, c, a, b in tm:
        lines.append("%-62s %-16s %5d  %s .. %s" % (name[:62], t, c, a, b))
    counts = {}
    for r in rounds:
        counts[r["type"]] = counts.get(r["type"], 0) + 1
    lines.append("type totals: " + ", ".join("%s=%d" % (t, counts[t]) for t in TYPES if t in counts))
    others = [r["label"] for r in rounds if r["type"] == "other"]
    lines.append("rounds typed 'other': %s" % (others or "none"))

    lines.append("")
    lines.append("== band validation (dd3 = dd4..dd8, dd9 = dd10..dd14, dd18 = sum of disjoint bands) ==")
    with_dist = [r for r in rounds if r["dist"] is not None]
    bad = [r for r in with_dist if any(i.startswith("[invalid] dd") for i in r["issues"])]
    lines.append("rounds with a distribution: %d (first #%s %s); failures: %d" % (
        len(with_dist), with_dist[0]["label"], with_dist[0]["draw_date"], len(bad)))
    for r in bad:
        for i in r["issues"]:
            if i.startswith("[invalid] dd"):
                lines.append("  #%s %s: %s" % (r["label"], r["draw_date"], i))
    nodist = [r["number"] for r in rounds if r["dist"] is None]
    lines.append("no distribution published (all dd = 0): %d rounds: %s" % (
        len(nodist), _ranges(nodist)))
    notie = [r["number"] for r in rounds if r["tie_break"] is None]
    lines.append("no tie-break (blank drawCutOff): %d rounds: %s" % (len(notie), _ranges(notie)))

    lines.append("")
    lines.append("== rounds parsed with repairs or problems (excluding the two lists above) ==")
    n_rep = 0
    for r in rounds:
        iss = [i for i in r["issues"] if not i.startswith("[missing]")
               and not i.startswith("[invalid] dd")]
        if iss:
            n_rep += 1
            for i in iss:
                lines.append("  #%-4s %s  %s" % (r["label"], r["draw_date"], i))
    lines.append("rounds listed: %d; rounds with an unusable size/cut-off/date: %d" % (
        n_rep, sum(1 for r in rounds if r["size"] is None or r["crs"] is None
                   or r["draw_date"] is None)))

    lines.append("")
    lines.append("== snapshot notes ==")
    for snap, h, refl in same_day_cases(rounds):
        lines.append("  snapshot as-on %s (reported by %s, 601+=%s): draw #%s %s %s on that day "
                     "reported %s (601+=%s) -> %s" % (
                         snap["dist_as_on"], ",".join("#%d" % n for n in snap["reported_by"]),
                         format(snap["dist"]["601-1200"], ","), h["label"], h["type"],
                         format(h["size"], ","), h["dist_as_on"],
                         format(h["dist"]["601-1200"], ","),
                         "already reflected (excluded)" if refl else "NOT reflected (included)"))
    seen = {}
    for r in with_dist:
        seen.setdefault(_dist_key(r["dist"]), []).append(r)
    for grp in seen.values():
        if len({g["dist_as_on"] for g in grp}) > 1:
            lines.append("  identical numbers, different as-on dates: " + ", ".join(
                "#%s says %s" % (g["label"], g["dist_as_on"]) for g in grp))
    stale = [r for r in with_dist if (r["draw_date"] - r["dist_as_on"]).days >= 8]
    lines.append("  stale snapshots (draw date - as-on >= 8 days): " + ", ".join(
        "#%s %s (%dd)" % (r["label"], r["draw_date"], (r["draw_date"] - r["dist_as_on"]).days)
        for r in stale))
    return "\n".join(lines)


def sanity_report(rounds):
    ctx = latest_ctx(rounds=rounds)
    snap = ctx["snapshot"]
    d = snap["dist"]
    lines = ["== latest snapshot sanity ==",
             "snapshot as on %s, reported by %s, total %s; target_date %s" % (
                 snap["dist_as_on"], ", ".join("#%d" % n for n in snap["reported_by"]),
                 format(snap["total"], ","), ctx["target_date"])]
    lines.append("bands: " + ", ".join("%s: %s" % (k, format(d[k], ",")) for k in BAND_KEYS))
    lines.append("cumulative_above: " + ", ".join(
        "%d -> %s" % (s, format(round(cumulative_above(d, s), 1), ",.1f"))
        for s in (400, 450, 480, 500, 510, 520, 530, 600)))
    lines.append("score_for_count:  " + ", ".join(
        "%s -> %d (real %.2f)" % (format(c, ","), score_for_count(d, c),
                                 score_for_count(d, c, integer=False))
        for c in (1000, 2000, 5000)))
    lines.append("invited_since_snapshot: " + "; ".join(
        "#%s %s %s %s invited @%s (%s)" % (i["label"], i["date"], i["type"],
                                          format(i["size"], ","), i["crs"], i["basis"])
        for i in ctx["invited_since_snapshot"]))
    return "\n".join(lines), ctx


# --------------------------------------------------------------------------------------
# Self-test (helpers + leakage)
# --------------------------------------------------------------------------------------
def selftest(rounds):
    fails = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)

    snap = latest_ctx(rounds=rounds)["snapshot"]
    d = snap["dist"]
    tot = dist_total(d)
    dens = density(d)
    check(abs(cumulative_above(d, 0) - tot) < 1e-6, "cum(0) != total")
    check(cumulative_above(d, 1201) == 0, "cum(1201) != 0")
    check(abs(cumulative_above(d, 601) - d["601-1200"]) < 1e-9, "cum(601) != 601+ band")
    check(abs(cumulative_above(d, 501) - d["601-1200"] - d["501-600"]) < 1e-9, "cum(501)")
    prev = None
    for s in range(0, 1202):
        c = cumulative_above(d, s)
        check(abs(c - cumulative_above(dens, s)) < 1e-6, "dict/density mismatch at %d" % s)
        if prev is not None:
            check(c <= prev + 1e-9, "cum not non-increasing at %d" % s)
        prev = c
    for x in (510.25, 500.5, 449.9, 0.5, 1200.5, -3, 1205):
        check(abs(cumulative_above(d, x) - cumulative_above(dens, x)) < 1e-6, "frac %s" % x)
    for count in (1, 10, 749, 750, 1000, 2000, 5000, 20000, 100000, tot - 1, tot):
        s = score_for_count(d, count)
        check(cumulative_above(d, s) >= count - 1e-9, "sfc(%s): cum(s) < count" % count)
        check(s == MAX_SCORE or cumulative_above(d, s + 1) < count, "sfc(%s) not largest" % count)
        sf = score_for_count(d, count, integer=False)
        check(int(math.floor(sf)) == s, "float sfc floor mismatch %s" % count)
    check(score_for_count(d, tot + 1) == 0, "count > pool should give 0")
    check(score_for_count(d, 0) == MAX_SCORE, "count 0 should give 1200")
    r1 = remove_top(d, 749)
    check(abs(cumulative_above(r1, 601)) < 1e-6, "remove_top(749) should empty 601+")
    r2 = remove_above(d, 500, 1000)
    check(abs(dist_total(r2) - (tot - 1000)) < 1e-6, "remove_above total")

    # Ordering and type map.
    check(all(a["number"] <= b["number"] for a, b in zip(rounds, rounds[1:])), "order")
    check(all(a["draw_date"] <= b["draw_date"] for a, b in zip(rounds, rounds[1:])), "dates")
    check(not [r for r in rounds if r["type"] == "other"], "rounds typed other")

    # Leakage: scramble the target and everything after it; ctx must not change.
    targets = [r for r in rounds if r["draw_date"] >= _dt.date(2023, 1, 1)]
    for T in targets[::3] + targets[-3:]:
        a = json.dumps(_ctx_public(build_ctx(rounds, T)), default=str, sort_keys=True)
        poisoned = []
        for r in rounds:
            if r["number"] >= T["number"]:
                r = dict(r, crs=-999, size=-1, tie_break=None,
                         dist={k: 7 for k in BAND_KEYS}, dist_as_on=_dt.date(2000, 1, 1))
            poisoned.append(r)
        Tp = next(r for r in poisoned if r["seq"] == T["seq"])
        b = json.dumps(_ctx_public(build_ctx(poisoned, Tp)), default=str, sort_keys=True)
        check(a == b, "leakage: ctx for #%s depends on the target or later rounds" % T["label"])
        ctx = build_ctx(rounds, T)
        check(all(h["number"] < T["number"] for h in ctx["history"]), "history has future")
        check(ctx["snapshot"] is not None, "no snapshot for #%s" % T["label"])
    # latest_ctx expectations for this data file (only meaningful for the 2026-10-16 file).
    lc = latest_ctx(rounds=rounds)
    if rounds[-1]["label"] == "450":
        check(lc["snapshot"]["dist_as_on"] == _dt.date(2026, 10, 12), "latest as-on")
        check([i["number"] for i in lc["invited_since_snapshot"]] == [448, 449, 450],
              "latest invited_since")
        check(lc["snapshot"]["total"] == 229456, "latest total")
    return fails


def _ctx_public(ctx):
    return {k: v for k, v in ctx.items()}


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def _json_default(o):
    if isinstance(o, (_dt.date, _dt.datetime)):
        return o.isoformat()
    if isinstance(o, set):
        return sorted(o)
    return str(o)


def load_approach(path):
    """Import an approach file by path. Its folder is APPENDED to sys.path (so sibling
    helper modules import, but can never shadow the standard library)."""
    path = os.path.abspath(path)
    folder = os.path.dirname(path)
    if folder not in sys.path:
        sys.path.append(folder)
    stem = os.path.splitext(os.path.basename(path))[0]
    modname = "approach_mod_" + re.sub(r"\W", "_", stem)
    spec = importlib.util.spec_from_file_location(modname, path)
    if spec is None or spec.loader is None:
        raise SystemExit("cannot load %s" % path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    if not callable(getattr(mod, "predict", None)):
        raise SystemExit("%s has no predict(ctx, draw_type, n)" % path)
    return mod


def _windows_from_args(args, rounds):
    user_types = [t.strip() for t in args.types.split(",")] if args.types else None
    if user_types:
        unknown = [t for t in user_types if t not in TYPES]
        if unknown:
            raise SystemExit("unknown type(s) %s; valid: %s" % (unknown, ", ".join(TYPES)))
    if args.start or args.end:
        start = args.start or rounds[0]["draw_date"].isoformat()
        end = args.end or rounds[-1]["draw_date"].isoformat()
        return [{"name": "custom_%s_%s" % (start, end), "start": start, "end": end,
                 "types": user_types}]
    names = [n.strip() for n in args.window.split(",")] if args.window else None
    if names:
        unknown = [n for n in names if n not in [w["name"] for w in DEFAULT_WINDOWS]]
        if unknown:
            raise SystemExit("unknown window(s) %s; valid: %s" % (
                unknown, ", ".join(w["name"] for w in DEFAULT_WINDOWS)))
    wins = []
    for w in DEFAULT_WINDOWS:
        if names and w["name"] not in names:
            continue
        w = dict(w)
        if user_types:
            w["types"] = [t for t in (w["types"] or user_types) if t in user_types]
            if not w["types"]:
                continue
        wins.append(w)
    return wins


def _latest_predictions(mod, rounds, args):
    """Predict the next draw (latest_ctx) for each requested type and size."""
    sizes = [int(x) for x in re.split(r"[,\s]+", args.latest.strip()) if x]
    types = [t.strip() for t in args.types.split(",")] if args.types else list(MAJOR_TYPES)
    probe = latest_ctx(rounds=rounds)
    print("\n[latest] next draw after #%s %s; snapshot as on %s; target_date %s" % (
        rounds[-1]["label"], rounds[-1]["draw_date"], probe["snapshot"]["dist_as_on"],
        probe["target_date"]))
    print("%-16s %7s %8s %8s %8s" % ("type", "n", "point", "low", "high"))
    out = {"target_date": probe["target_date"], "snapshot_as_on": probe["snapshot"]["dist_as_on"],
           "predictions": []}
    for t in types:
        for n in sizes:
            rec = {"type": t, "n": n}
            try:
                p, lo, hi, info, warns = normalize_prediction(
                    mod.predict(latest_ctx(rounds=rounds), t, n))  # fresh ctx per call
                rec.update(point=p, low=lo, high=hi, warnings=warns)
                f = lambda v: "-" if v is None else "%.1f" % v  # noqa: E731
                print("%-16s %7d %8s %8s %8s" % (t, n, f(p), f(lo), f(hi)))
            except Exception as e:  # noqa: BLE001
                if args.strict:
                    raise
                rec["error"] = "%s: %s" % (type(e).__name__, e)
                print("%-16s %7d   ! %s" % (t, n, rec["error"]))
            out["predictions"].append(rec)
    return out


def run_cli(argv=None):
    sys.modules.setdefault("harness", sys.modules[__name__])
    sys.dont_write_bytecode = True  # no __pycache__ next to approach files
    ap = argparse.ArgumentParser(description="Walk-forward backtest for CRS cut-off predictors")
    ap.add_argument("approaches", nargs="*", help="approach .py files exposing predict()")
    ap.add_argument("--start", help="first target draw date (YYYY-MM-DD)")
    ap.add_argument("--end", help="last target draw date (YYYY-MM-DD)")
    ap.add_argument("--types", help="comma-separated canonical types to evaluate")
    ap.add_argument("--window", help="run only these default windows, e.g. tune_2025 "
                                     "(tune_2025, holdout_2026, general_2023_2024)")
    ap.add_argument("--latest", help="also predict the next draw for these sizes, e.g. "
                                     "1000,2000,5000 (types: --types or the major types)")
    ap.add_argument("--data", help="rounds JSON (default %s)" % DEFAULT_DATA)
    ap.add_argument("--check", action="store_true", help="print the data report and sanity numbers")
    ap.add_argument("--selftest", action="store_true", help="run helper and leakage tests")
    ap.add_argument("--strict", action="store_true", help="re-raise exceptions from predict()")
    ap.add_argument("--rows", action="store_true", help="also print per-draw rows")
    ap.add_argument("--out-dir", help="where to write <approach>.results.json")
    ap.add_argument("--no-write", action="store_true", help="do not write results JSON")
    args = ap.parse_args(argv)

    data_path = os.path.abspath(args.data or DEFAULT_DATA)
    rounds = load_rounds(data_path)
    if not (args.approaches or args.check or args.selftest):
        ap.print_help()
        return 2
    if args.check:
        print("data: %s (%d rounds, #%s %s .. #%s %s)" % (
            data_path, len(rounds), rounds[0]["label"], rounds[0]["draw_date"],
            rounds[-1]["label"], rounds[-1]["draw_date"]))
        print(data_report(rounds))
        print()
        print(sanity_report(rounds)[0])
    if args.selftest:
        fails = selftest(rounds)
        print("selftest: %s" % ("PASS" if not fails else "FAIL"))
        for f in fails:
            print("  - " + f)
        if fails:
            return 1
    if not args.approaches:
        return 0

    windows = _windows_from_args(args, rounds)
    summary = []
    for apath in args.approaches:
        mod = load_approach(apath)
        name = getattr(mod, "NAME", None) or os.path.splitext(os.path.basename(apath))[0]
        result = {"approach": name, "approach_file": os.path.abspath(apath),
                  "harness_version": HARNESS_VERSION, "data_file": data_path,
                  "data_rounds": len(rounds),
                  "data_newest": {"number": rounds[-1]["label"],
                                  "date": rounds[-1]["draw_date"]},
                  "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                  "windows": {}}
        print("\n### %s  (%s)" % (name, os.path.basename(apath)))
        for w in windows:
            try:
                res = evaluate(mod.predict, w["start"], w["end"], w["types"], rounds=rounds,
                               strict=args.strict)
            except Exception:
                traceback.print_exc()
                return 1
            result["windows"][w["name"]] = res
            print("\n[%s] %s .. %s%s" % (w["name"], w["start"], w["end"],
                                        (" types=" + ",".join(w["types"])) if w["types"] else ""))
            print(format_metrics(res["metrics"]))
            failed = [r for r in res["rows"] if r.get("error")]
            for r in failed[:5]:
                print("  ! #%s %s %s: %s" % (r["label"], r["date"], r["type"], r["error"]))
            if len(failed) > 5:
                print("  ! ... %d more failures (see results JSON)" % (len(failed) - 5))
            if args.rows:
                print(format_rows(res["rows"]))
            summary.append((name, w["name"], res["metrics"]))
        if args.latest:
            result["latest"] = _latest_predictions(mod, rounds, args)
        if not args.no_write:
            out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(
                os.path.abspath(apath))
            out = os.path.join(out_dir, os.path.splitext(os.path.basename(apath))[0]
                               + ".results.json")
            with open(out, "w", encoding="utf-8") as fh:
                json.dump(result, fh, indent=1, default=_json_default)
            print("\nwrote %s" % out)
    if len(args.approaches) > 1:
        print("\n### comparison (MAE / coverage)")
        print("%-44s %-20s %8s %8s %8s %8s" % ("approach", "window", "MAJOR", "ALL", "cov MAJ",
                                               "cov ALL"))
        for name, wname, m in summary:
            def g(key, field):
                v = m.get(key, {}).get(field)
                return "-" if v is None else ("%.0f%%" % (100 * v) if field == "coverage"
                                              else "%.2f" % v)
            print("%-44s %-20s %8s %8s %8s %8s" % (name[:44], wname, g("MAJOR", "mae"),
                                                   g("ALL", "mae"), g("MAJOR", "coverage"),
                                                   g("ALL", "coverage")))
    return 0


if __name__ == "__main__":
    sys.exit(run_cli())
