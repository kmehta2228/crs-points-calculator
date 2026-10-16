#!/usr/bin/env python3
"""Build the website's "next draw prediction" data file from IRCC's Express Entry rounds JSON.

Standard library only, Python 3.9+; run with python3 -I.

  python3 -I .github/scripts/build_predictions.py ROUNDS_JSON OUT_JSON [--previous PUBLISHED_JSON]

ROUNDS_JSON is IRCC's raw ee_rounds_123_en.json. update_draws.py downloads it and runs this
script with OUT_JSON and --previous both set to site/data/draws.json. OUT_JSON is written
atomically and only when its content changes. The output depends only on the input bytes and
this code (no clock, no randomness), so re-running on unchanged IRCC data gives a byte-identical
file. When $GITHUB_OUTPUT is set, changed=true|false and latest=<draw number> are appended to it.
Exit codes: 0 = written or unchanged; 1 = refused (input failed a hard check; nothing written).

METHOD (chosen by walk-forward backtests over 2023-2026; see the research summary)
  For each draw type and N, three independently built, walk-forward-tested models predict the
  cut-off at the expected date of that type's next draw:
    1. stock-and-flow queue      crs_engine/approach_stock_and_flow.py
    2. shape-aware pool walk     crs_engine/approach_pool_walk_shaped.py
    3. per-type ridge regression crs_engine/approach_regression.py
  Published point / low / high = the MEDIAN of the three members' points / lows / highs (each
  member's low-high is its own ~80% range from its own walk-forward errors). The crs_engine
  files are verbatim copies of the backtested approach files (their sha256 go into the output):
  never edit them without re-running the backtest (python3 -I crs_engine/harness.py ...).
  Upcoming cycle: the PNP draw and the CEC draw that open a cycle are simulated (placeholder
  rounds at the predicted cut-off) before predicting the draws that follow them in that cycle.
  Nov 3, 2026 rule change: members are kept rule-agnostic (the regression's own reform term is
  switched off) and the shift is added once, explicitly, from NOV3_DELTA (predict_final()).
"""
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import re
import statistics
import sys
import tempfile

sys.dont_write_bytecode = True       # never leave __pycache__ folders in the repository
HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.join(HERE, "crs_engine")
# crs_engine/harness.py (parser, snapshot logic, pool helpers) is loaded by path and registered
# as "harness", the name the approach files import, so no other module called harness on
# sys.path can stand in for it.
_HARNESS_SPEC = importlib.util.spec_from_file_location(
    "harness", os.path.join(ENGINE, "harness.py"))
harness = importlib.util.module_from_spec(_HARNESS_SPEC)
sys.modules["harness"] = harness
_HARNESS_SPEC.loader.exec_module(harness)

SCHEMA_VERSION = 1
MODEL_VERSION = "2026-10-16.1"
SOURCE_URL = "https://www.canada.ca/content/dam/ircc/documents/json/ee_rounds_123_en.json"
ROUND_URL = ("https://www.canada.ca/en/immigration-refugees-citizenship/corporate/mandate/"
             "policies-operational-instructions-agreements/ministerial-instructions/"
             "express-entry-rounds/invitations.html?q=")
ENGINE_FILES = ("harness", "approach_stock_and_flow", "approach_pool_walk_shaped",
                "approach_regression")


def _load_engine(name):
    path = os.path.join(ENGINE, name + ".py")
    spec = importlib.util.spec_from_file_location("crs_engine_" + name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


SF = _load_engine("approach_stock_and_flow")
SH = _load_engine("approach_pool_walk_shaped")
RG = _load_engine("approach_regression")
MEMBERS = (("stock_and_flow", SF), ("pool_walk_shaped", SH), ("regression", RG))

# --------------------------------------------------------------------------------------------
# Static facts and assumptions (change only with a code update; each has a source)
# --------------------------------------------------------------------------------------------
CHANGE_DATE = dt.date(2026, 11, 3)    # CRS "Labour Market Integration" factor takes effect
PAUSE_MAX_DAYS = 14                   # "could take up to two weeks ... we'll pause invitation rounds"
CHANGE_SOURCES = [
    "https://www.canada.ca/en/immigration-refugees-citizenship/services/immigrate-canada/express-entry/changes.html",
    "https://www.canada.ca/en/immigration-refugees-citizenship/news/2026/10/canada-refocusing-express-entry-on-top-talent-that-drives-economic-growth.html",
    "https://www.canada.ca/en/immigration-refugees-citizenship/services/immigrate-canada/express-entry/check-score/high-wage.html",
]
# Shift of the cut-off caused by the rule change ALONE (no extra pause arrivals: the members
# already account for time since the last draw), by type and N: (N, low, mid, high) points.
# Sources: model/nov3/simulate.py "rule change only" rows (CEC, French) and rule-change rows at
# unchanged timing (healthcare, trades); senior managers / physicians whole-curve shifts (every
# eligible candidate has 12 months of Canadian work in a listed NOC); PNP from the scenario
# note (+0..+15: the cut-off is the lowest nominee, who rarely gains); all-program from
# model/final/nov3_general_delta.py --after-pnp (projected post-change pool bands, shaped
# density, nominees already invited). Healthcare and trades at 5,000 were lowered by 1-2 points
# so every column is non-increasing in N. Interpolated in ln N, flat outside the table.
NOV3_DELTA = {
    "cec": ((1000, 32, 39, 45), (2000, 28, 35, 40), (2500, 27, 34, 38), (3000, 26, 34, 38),
            (5000, 19, 30, 36)),
    "french": ((1000, 3, 6, 12), (2000, 2, 5, 10), (2500, 2, 5, 9), (3000, 2, 4, 7),
               (5000, 1, 2, 5)),
    "healthcare": ((1000, 8, 16, 25), (2000, 4, 9, 17), (2500, 3, 7, 14), (3000, 3, 6, 12),
                   (3500, 2, 5, 10), (5000, 2, 5, 10)),
    "trades": ((1000, 6, 14, 24), (2000, 3, 8, 17), (2500, 2, 6, 13), (3000, 2, 5, 11),
               (3500, 1, 4, 8), (5000, 1, 4, 8)),
    "senior_managers": ((250, 50, 62, 66),),
    "physicians": ((250, 63, 70, 70),),
    "pnp": ((500, 0, 5, 15),),
    "general": ((1000, 20, 30, 37), (1500, 20, 31, 39), (2000, 19, 30, 40), (3000, 17, 29, 40),
                (5000, 15, 26, 37), (8000, 12, 22, 33), (10000, 11, 20, 31)),
}
NOV3_PROXY = {"transport": "trades"}          # no scenario run for these: borrow a similar type
NOV3_DEFAULT = "trades"                       # any other category
NOV3_MARGIN = 0.20                            # extra +-20% of the mid shift on the range
# How much of the shift to add once real post-change draws of the type exist (k = how many):
# (weight on the point and low bound, weight on the high bound). After one post-change draw the
# regression member has re-learnt most of the jump while the queue member has not, so the point
# keeps a third of the shift and the upper bound all of it; after two, only the upper bound is
# widened; from three on, nothing is added. All-program draws: k = 0 until a rescored pool
# snapshot exists, then 2 (1-2 rescored snapshots) and 3 (3 or more).
NOV3_FADE = {0: (1.0, 1.0), 1: (1.0 / 3.0, 1.0), 2: (0.0, 0.5)}
NOV3_RECAL_DRAWS = 3

# 2026-2028 Immigration Levels Plan and other official facts shown in the quota panel.
# Update by hand when the 2027-2029 Levels Plan is tabled (due in fall 2026) and each January,
# and bump MODEL_VERSION whenever these facts or the method change.
QUOTA_FACTS = {
    "checked": "2026-10-16",
    "plan": "2026-2028 Immigration Levels Plan",
    "plan_url": "https://www.canada.ca/en/immigration-refugees-citizenship/corporate/mandate/corporate-initiatives/levels/supplementary-immigration-levels-2026-2028.html",
    "targets": [
        {"id": "fhs", "label": "Federal High Skilled admissions (CEC, FSW, FST)", "year": 2026,
         "target": 109000, "low": 85000, "high": 120000},
        {"id": "fhs", "label": "Federal High Skilled admissions (CEC, FSW, FST)", "year": 2027,
         "target": 111000, "low": 86000, "high": 122000, "notional": True},
        {"id": "pnp", "label": "Provincial Nominee Program admissions (all streams)", "year": 2026,
         "target": 91500, "low": 82000, "high": 105000},
        {"id": "total", "label": "All new permanent residents", "year": 2026,
         "target": 380000, "low": 350000, "high": 420000},
        {"id": "francophone", "label": "French-speaking permanent residents outside Quebec",
         "year": 2026, "target": 30267, "share": 0.09},
    ],
    "progress": [
        {"label": "Federal High Skilled admissions, Jan-Jul 2026 (IRCC open data, rounded)",
         "value": 47005, "of": 109000, "as_of": "2026-07-31",
         "url": "https://www.ircc.canada.ca/opendata-donneesouvertes/data/ODP-PR-PT_IMMCAT.csv"},
        {"label": "French-speaking admissions outside Quebec, Jan-Jul 2026 (IRCC open data, rounded)",
         "value": 15615, "of": 30267, "as_of": "2026-07-31",
         "url": "https://www.ircc.canada.ca/opendata-donneesouvertes/data/ODP-PR-FRE_SP_IMMCAT.csv"},
        {"label": "People waiting for a CEC or FSW decision (IRCC processing-times data)",
         "value": 112200, "as_of": "2026-10-09",
         "url": "https://www.canada.ca/content/dam/ircc/documents/json/flpt-en.json"},
    ],
    "statements": [
        {"text": "IRCC only invites the number of clients to apply needed to meet admissions targets in the Levels Plan.",
         "url": "https://www.canada.ca/en/immigration-refugees-citizenship/corporate/transparency/committees/cimm-dec-04-2025/permanent-resident-targets.html"},
        {"text": "The total number of invitations to apply will be within the targets of the 2026-2028 Immigration Levels Plan. The additional categories do not increase the number of invitations within that target.",
         "url": "https://www.canada.ca/en/immigration-refugees-citizenship/news/2026/02/attracting-the-worlds-best-talent-to-fill-canadas-labour-gaps-and-build-our-economy.html"},
    ],
    "notes": [
        "The 2027-2029 Levels Plan is due in fall 2026 and may change the 2027 numbers.",
        "Statistics Canada does not publish Express Entry draws or pool data; IRCC is the only source.",
    ],
}

# Day of each draw relative to the cycle's first (PNP) draw, from the 2026 pattern. The
# hypothetical all-program draw takes the CEC slot (it would replace the CEC draw).
CYCLE_OFFSET = {"pnp": 0, "general": 1, "cec": 1, "french": 2, "physicians": 2}
CYCLE_OFFSET_DEFAULT = 3
HOLIDAY_STOP = ((12, 22), (1, 3))     # never a draw after Dec 23; first draw Jan 5-10

# UI draw types: plain-English labels, slider step and the published N grid.
TYPE_INFO = {
    "cec": {"label": "Canadian Experience Class", "short": "CEC", "step": 250,
            "grid": (500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000, 3500, 4000, 5000,
                     6000, 7000, 8000, 10000),
            "note": "For candidates with at least 1 year of skilled work experience in Canada. "
                    "IRCC held a CEC draw in almost every two-week cycle of 2026."},
    "french": {"label": "French-language proficiency", "short": "French", "step": 250,
               "grid": (500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000,
                        7000, 8000, 10000),
               "note": "For candidates with strong French (at least NCLC 7 in all four abilities). "
                       "The cut-off is higher when French draws are further apart."},
    "healthcare": {"label": "Healthcare and social services", "short": "Healthcare", "step": 250,
                   "grid": (250, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 8000),
                   "note": "For candidates with 12 months of recent work in an eligible healthcare or "
                           "social-services occupation. Draws have come every 2 to 4 months."},
    "trades": {"label": "Trades", "short": "Trades", "step": 250,
               "grid": (250, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000),
               "note": "For candidates with 12 months of recent work in an eligible trade. "
                       "Few trades draws so far, about 6 months apart, so treat this as a rough guide."},
    "pnp": {"label": "Provincial Nominee Program", "short": "PNP", "step": 10,
            "grid": (100, 200, 300, 400, 500, 600, 700, 800, 1000, 1200, 1500, 2000),
            "note": "Only for candidates with a provincial nomination (+600 points). IRCC invites almost "
                    "every nominee in the pool, so this cut-off is just the lowest nominee's score; "
                    "with a nomination you are very likely to be invited in the next PNP draw."},
    "senior_managers": {"label": "Senior managers with Canadian work experience",
                        "short": "Senior managers", "step": 10,
                        "grid": (50, 100, 150, 200, 250, 300, 400, 500, 750, 1000),
                        "note": "A small category started in 2026. Very few draws, so treat this as a rough guide."},
    "physicians": {"label": "Physicians with Canadian work experience", "short": "Physicians",
                   "step": 10, "grid": (50, 100, 150, 200, 250, 300, 400, 500, 750),
                   "note": "A small category started in 2026. Eligibility is narrow, so cut-offs are far "
                           "below other draws. Very few draws: a rough guide only."},
    "transport": {"label": "Transport occupations", "short": "Transport", "step": 10,
                  "grid": (100, 200, 300, 500, 750, 1000, 1500, 2000),
                  "note": "Rare draws (one in 2026), so treat this as a rough guide."},
    "stem": {"label": "STEM occupations", "short": "STEM", "step": 250,
             "grid": (500, 1000, 1500, 2000, 3000, 4000, 5000, 6000),
             "note": "Rare draws, so treat this as a rough guide."},
    "education": {"label": "Education occupations", "short": "Education", "step": 250,
                  "grid": (250, 500, 1000, 1500, 2000, 2500, 3000, 4000),
                  "note": "Rare draws, so treat this as a rough guide."},
    "agriculture": {"label": "Agriculture and agri-food occupations", "short": "Agriculture",
                    "step": 10, "grid": (100, 200, 300, 500, 750, 1000),
                    "note": "Rare draws, so treat this as a rough guide."},
    "fsw": {"label": "Federal Skilled Worker Program", "short": "FSW", "step": 250,
            "grid": (500, 1000, 2000, 3000, 4000, 5000), "note": "Rare draws, so treat this as a rough guide."},
    "fst": {"label": "Federal Skilled Trades Program", "short": "FST", "step": 50,
            "grid": (100, 250, 500, 1000, 1500, 2000), "note": "Rare draws, so treat this as a rough guide."},
    "general": {"label": "All programs (hypothetical)", "short": "All programs", "step": 250,
                "grid": (1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 7000, 8000, 10000),
                "note": None},   # written from the data (date of the last all-program draw)
}
UI_ORDER = ("cec", "french", "healthcare", "trades", "pnp", "senior_managers", "physicians",
            "transport", "stem", "education", "agriculture", "fsw", "fst", "general")
ACTIVE_DAYS = 365          # a type appears in the UI if drawn within this many days ...
MIN_TYPICAL_SIZE = 100     # ... and its typical draw is at least this big (drops "military": 4)
BACKTEST_DAYS = 365        # rolling walk-forward accuracy window
BT_MIN_ROWS = 6            # rare types: extend their backtest back until this many draws ...
BT_EARLIEST = dt.date(2023, 1, 1)   # ... but not before 2023 (pool data starts in 2022)
INTERVAL_LEVEL = 0.8
TYPE_PLAIN = {"pnp": "Provincial Nominee Program", "cec": "Canadian Experience Class",
              "french": "French-language proficiency", "healthcare": "Healthcare and social services",
              "trades": "Trades", "stem": "STEM", "education": "Education", "transport": "Transport",
              "agriculture": "Agriculture and agri-food", "senior_managers": "Senior managers",
              "physicians": "Physicians", "military": "Skilled military recruits",
              "general": "All programs", "fsw": "Federal Skilled Worker", "fst": "Federal Skilled Trades",
              "other": "Other"}

# --------------------------------------------------------------------------------------------
# Two explicit adaptations of the members (no engine file is edited)
# --------------------------------------------------------------------------------------------
# 1) The regression ships with its own Nov-3 reform dummy. Switch it off: the rule change is
#    added once, for all three members alike, in predict_final() (no double counting).
RG.REFORM_DATE = dt.date(9999, 12, 31)
# 2) The shaped pool walk estimates daily inflow per band from the newest snapshot and one about
#    70 days older. Across the Nov-3 rescoring that difference is not inflow (about +12,000 in
#    501-600 overnight), so no inflow is estimated from a pair that straddles CHANGE_DATE.
_SH_INFLOW = SH._inflow


def _inflow_guarded(history, snap, cfg=SH.CFG):
    snaps = SH._snapshots(history)
    t_l = snap["dist_as_on"]
    latest = [s for s in snaps if s[0] <= t_l]
    if len(latest) >= 2:
        older = [s for s in latest[:-1] if (t_l - s[0]).days >= 21]
        if older:
            o = min(older, key=lambda s: abs((t_l - s[0]).days - cfg["inflow_window"]))
            if o[0] < CHANGE_DATE <= latest[-1][0]:
                return None
    return _SH_INFLOW(history, snap, cfg)


SH._inflow = _inflow_guarded


# --------------------------------------------------------------------------------------------
# Input validation (hard checks from the data audit: refuse = write nothing, keep last good file)
# --------------------------------------------------------------------------------------------
class Refuse(Exception):
    """The input failed a hard check: publish nothing."""


_INT_RX = re.compile(r"\d{1,3}(?:,\d{3})+|\d+")


def _int(v):
    t = v.strip() if isinstance(v, str) else None
    if t is None or not _INT_RX.fullmatch(t):
        return None
    return int(t.replace(",", ""))


def _num_key(label):
    return (int(re.match(r"\d+", label).group(0)), label)


def validate_raw(raw_bytes, previous=None):
    """Hard checks on IRCC's raw rounds JSON. Returns (data, warnings) or raises Refuse."""
    warnings = []
    if not 100_000 <= len(raw_bytes) <= 20_000_000:
        raise Refuse("unexpected size %d bytes" % len(raw_bytes))
    try:
        data = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Refuse("not UTF-8 JSON: %s" % exc) from exc
    rounds = data.get("rounds") if isinstance(data, dict) else None
    if not isinstance(rounds, list) or len(rounds) < 400:
        raise Refuse("missing or short 'rounds' list")
    seen, newest = set(), None
    for r in rounds:
        if not isinstance(r, dict):
            raise Refuse("a round is not an object")
        n = str(r.get("drawNumber", "")).strip()
        if not re.fullmatch(r"\d+[a-z]?", n):
            raise Refuse("bad drawNumber %r" % n)
        if n in seen:
            raise Refuse("duplicate drawNumber %s" % n)
        seen.add(n)
        try:
            d = dt.date.fromisoformat(str(r.get("drawDate", "")).strip())
        except ValueError as exc:
            raise Refuse("#%s: bad drawDate %r" % (n, r.get("drawDate"))) from exc
        if d < dt.date(2015, 1, 1):
            raise Refuse("#%s: drawDate %s out of range" % (n, d))
        size, crs = _int(r.get("drawSize")), _int(r.get("drawCRS"))
        if size is None or not 1 <= size <= 50_000:
            raise Refuse("#%s: bad drawSize %r" % (n, r.get("drawSize")))
        if crs is None or not 0 <= crs <= 1200:
            raise Refuse("#%s: bad drawCRS %r" % (n, r.get("drawCRS")))
        name = " ".join(str(r.get("drawName", "")).split())
        if not name or "<" in name:
            raise Refuse("#%s: bad drawName %r" % (n, name))
        newest = max(newest, _num_key(n)) if newest else _num_key(n)
    if previous:
        prov = previous.get("provenance", {})
        if prov.get("rounds") and len(rounds) < prov["rounds"]:
            raise Refuse("round count fell from %s to %d" % (prov["rounds"], len(rounds)))
        pl = (prov.get("latest_draw") or {}).get("number")
        if pl and _num_key(pl) > newest:
            raise Refuse("newest draw went backwards: %s -> %s" % (pl, newest[1]))
    return data, warnings


def parse_rounds(data):
    """IRCC raw rounds -> harness round dicts (ascending draw number), same as load_rounds()."""
    rounds = [harness._parse_round(r) for r in data["rounds"] if isinstance(r, dict)]
    far = dt.datetime(9999, 1, 1, tzinfo=dt.timezone.utc)
    rounds.sort(key=lambda r: (r["number"] if r["number"] is not None else 10 ** 9,
                               r["suffix"], r["draw_datetime"] or far))
    for i, r in enumerate(rounds):
        r["seq"] = i
    return rounds


# --------------------------------------------------------------------------------------------
# Ensemble prediction
# --------------------------------------------------------------------------------------------
def nov3_delta(t, n):
    """Rule-change shift (low, mid, high) for type t and N (ln-N interpolation, flat outside)."""
    table = NOV3_DELTA.get(t) or NOV3_DELTA[NOV3_PROXY.get(t, NOV3_DEFAULT)]
    if len(table) == 1 or n <= table[0][0]:
        return tuple(table[0][1:])
    if n >= table[-1][0]:
        return tuple(table[-1][1:])
    for a, b in zip(table, table[1:]):
        if a[0] <= n <= b[0]:
            f = (math.log(n) - math.log(a[0])) / (math.log(b[0]) - math.log(a[0]))
            return tuple(a[i] + f * (b[i] - a[i]) for i in (1, 2, 3))
    return tuple(table[-1][1:])


def post_change_count(history, t):
    """k for NOV3_FADE: post-change draws of type t (all-program: rescored pool snapshots)."""
    if t == "general":
        snaps = {h["dist_as_on"] for h in history
                 if h["dist"] is not None and h["dist_as_on"] and h["dist_as_on"] >= CHANGE_DATE}
        return 0 if not snaps else (2 if len(snaps) < 3 else 3)
    return sum(1 for h in history if h["type"] == t and h["draw_date"] >= CHANGE_DATE)


def rule_change_weights(history, t, target_date):
    """(weight on point and low bound, weight on high bound) of the Nov-3 shift still to add."""
    if target_date < CHANGE_DATE:
        return 0.0, 0.0
    return NOV3_FADE.get(post_change_count(history, t), (0.0, 0.0))


def ensemble_predict(ctx, t, n):
    """Median of the three members' point / low / high (before the rule-change layer)."""
    pts, los, his, members = [], [], [], {}
    for name, mod in MEMBERS:
        try:
            p, lo, hi, _info, _w = harness.normalize_prediction(mod.predict(ctx, t, n))
        except Exception as exc:  # noqa: BLE001  a failing member is dropped, never fatal
            members[name] = {"error": "%s: %s" % (type(exc).__name__, exc)}
            continue
        if lo is None:
            lo = hi = p
        pts.append(p)
        los.append(min(lo, p))
        his.append(max(hi, p))
        members[name] = (p, lo, hi)
    if not pts:
        raise RuntimeError("all ensemble members failed for %s n=%s" % (t, n))
    point = statistics.median(pts)
    return point, min(statistics.median(los), point), max(statistics.median(his), point), members


def predict_final(ctx, t, n, apply_change=True):
    """Published prediction for type t, N invitations, at ctx['target_date']."""
    point, low, high, members = ensemble_predict(ctx, t, n)
    wp, wh = rule_change_weights(ctx["history"], t, ctx["target_date"]) if apply_change \
        else (0.0, 0.0)
    if wh > 0:
        d_lo, d_mid, d_hi = nov3_delta(t, n)
        margin = NOV3_MARGIN * d_mid * wh
        point += wp * d_mid
        low += wp * d_lo - margin
        high += wh * d_hi + margin
    clamp = lambda v: max(0.0, min(1200.0, v))  # noqa: E731
    return clamp(point), clamp(low), clamp(high), wp, members


def predict_grid(history, target_date, t, grid, apply_change=True):
    """[[n, point, low, high]] over the grid, forced non-increasing in N (a safety net only:
    every member is monotone by construction)."""
    ctx = harness.make_ctx(history, target_date)
    rows, weight, failures = [], 0.0, set()
    for n in grid:
        p, lo, hi, w, members = predict_final(ctx, t, n, apply_change)
        weight = w
        failures.update(k for k, v in members.items() if isinstance(v, dict))
        rows.append([n, p, lo, hi])
    for i in range(1, len(rows)):
        for j in (1, 2, 3):
            rows[i][j] = min(rows[i][j], rows[i - 1][j])
    out = [[n, round(p, 1), round(min(lo, p), 1), round(max(hi, p), 1)] for n, p, lo, hi in rows]
    return out, weight, sorted(failures)


def grid_value(grid, n):
    """Linear interpolation in ln N (the rule the website uses)."""
    if n <= grid[0][0]:
        return tuple(grid[0][1:])
    for a, b in zip(grid, grid[1:]):
        if a[0] <= n <= b[0]:
            f = (math.log(n) - math.log(a[0])) / (math.log(b[0]) - math.log(a[0]))
            return tuple(a[i] + f * (b[i] - a[i]) for i in (1, 2, 3))
    return tuple(grid[-1][1:])


# --------------------------------------------------------------------------------------------
# Calendar: expected date of each type's next draw, and the cycle simulation
# --------------------------------------------------------------------------------------------
def _monday(d):
    return d - dt.timedelta(days=d.weekday())


def _skip_holiday(start):
    (m1, d1), (m2, d2) = HOLIDAY_STOP
    if (start.month, start.day) >= (m1, d1) or (start.month, start.day) <= (m2, d2):
        year = start.year + 1 if start.month == 12 else start.year
        first = dt.date(year, 1, 4)
        return first + dt.timedelta(days=(7 - first.weekday()) % 7)
    return start


def _offset(t):
    return dt.timedelta(days=CYCLE_OFFSET.get(t, CYCLE_OFFSET_DEFAULT))


def cycle_calendar(rounds):
    """Cycle cadence from the data: in 2025-2026 every cycle opens with a PNP draw."""
    starts = sorted({r["draw_date"] for r in rounds if r["type"] == "pnp"})
    last_start = starts[-1]
    gaps = [(b - a).days for a, b in zip(starts[-7:], starts[-6:]) if (b - a).days > 0]
    gap = min(max(int(statistics.median(gaps)) if gaps else 14, 7), 28)
    latest = max(r["draw_date"] for r in rounds)
    post_regime = latest >= CHANGE_DATE
    nxt = _skip_holiday(_monday(last_start + dt.timedelta(days=gap)))
    if post_regime:   # the pause is over: the first post-change draw is known
        after = min(r["draw_date"] for r in rounds if r["draw_date"] >= CHANGE_DATE)
        window = (after, after)
    else:
        after = _monday(last_start)
        while after < CHANGE_DATE + dt.timedelta(days=PAUSE_MAX_DAYS - 1):
            after += dt.timedelta(days=gap)
        after = _skip_holiday(after)
        window = (CHANGE_DATE + dt.timedelta(days=PAUSE_MAX_DAYS - 1), after + dt.timedelta(days=4))
    return {"last_start": last_start, "gap": gap, "latest": latest, "post_regime": post_regime,
            "next_start": nxt, "after_pause_start": after, "after_pause_window": window}


def _in_pause(d):
    return CHANGE_DATE <= d < CHANGE_DATE + dt.timedelta(days=PAUSE_MAX_DAYS - 1)


def next_cycle_slot(rounds, t, cal):
    """(expected date, cycle start) of type t's next draw if held in the next available cycle."""
    drawn = any(r["type"] == t and r["draw_date"] >= cal["last_start"] for r in rounds)
    if not drawn and cal["last_start"] + _offset(t) > cal["latest"]:
        return cal["last_start"] + _offset(t), cal["last_start"]     # cycle still running
    start = cal["next_start"]
    if not cal["post_regime"] and _in_pause(start + _offset(t)):
        start = cal["after_pause_start"]                           # no draws during the pause
    return start + _offset(t), start


def _placeholder(history, t, date, size, crs):
    """A placeholder round for a draw expected earlier in the same cycle (never published)."""
    last = history[-1]
    names = {"pnp": "Provincial Nominee Program", "cec": "Canadian Experience Class"}
    return {"number": last["number"] + 1, "label": "%d*" % (last["number"] + 1), "suffix": "",
            "seq": last["seq"] + 1, "draw_date": date,
            "draw_datetime": dt.datetime(date.year, date.month, date.day, 11, 0, 0,
                                         tzinfo=dt.timezone.utc),
            "name": names.get(t, t), "type": t, "version": None, "programs": "",
            "size": int(size), "crs": int(round(crs)), "tie_break": None, "dist_as_on": None,
            "dist": None, "dist_total": None, "dist_reported_total": None,
            "issues": ["placeholder"]}


class CycleSim:
    """One upcoming cycle. The cycle's PNP draw (day 0) and CEC draw (day 1) are added as
    placeholder rounds at their predicted cut-offs before predicting any later draw of the same
    cycle (so pool-based members remove those invitees, and queue members see the CEC draw).
    Sizes: CEC at its default N; PNP at its default N scaled by the days since the previous PNP
    draw (nominees accumulate between PNP draws), between 1x and 3x. Openers dated on or after
    `until` are not added (no draws during the Nov-3 pause)."""

    OPENERS = (("pnp", 0), ("cec", 1))

    def __init__(self, history, start, default_n, apply_change=True, until=None):
        self.base = list(history)
        self.start = start
        self.default_n = default_n
        self.apply_change = apply_change
        self.until = until
        self.cache = {}
        self.added = []

    def _opener(self, opener, off, hist):
        if opener not in self.cache:
            d = self.start + dt.timedelta(days=off)
            ph = None
            if self.until is None or d < self.until:
                n = self.default_n[opener]
                if opener == "pnp":
                    prev = [r["draw_date"] for r in hist if r["type"] == "pnp"]
                    days = (d - prev[-1]).days if prev else 14
                    n = int(round(n * min(3.0, max(1.0, days / 14.0))))
                p = predict_final(harness.make_ctx(hist, d), opener, n, self.apply_change)[0]
                ph = _placeholder(hist, opener, d, n, p)
                self.added.append({"type": opener, "date": d.isoformat(), "size": n,
                                   "crs": ph["crs"]})
            self.cache[opener] = ph
        return self.cache[opener]

    def history_for(self, t=None):
        """A new list: the history as it will stand just before type t's slot in this cycle
        (t=None: after both openers)."""
        off_t = CYCLE_OFFSET.get(t, CYCLE_OFFSET_DEFAULT) if t else 99
        hist = list(self.base)
        for opener, off in self.OPENERS:
            if opener == t or off_t <= off or opener not in self.default_n:
                continue
            if t == "general" and opener == "cec":
                continue            # the hypothetical all-program draw replaces the CEC draw
            if any(r["type"] == opener and r["draw_date"] >= self.start for r in self.base):
                continue            # already held in this cycle
            ph = self._opener(opener, off, hist)
            if ph is not None:
                hist.append(ph)
        return hist


# --------------------------------------------------------------------------------------------
# Backtest (rolling, walk-forward, exactly the published method)
# --------------------------------------------------------------------------------------------
def _backtest_row(rounds, T):
    hist = [r for r in rounds if r["number"] < T["number"]]
    ctx = harness.make_ctx(hist, T["draw_date"])
    p, lo, hi, _w, _m = predict_final(ctx, T["type"], T["size"])
    prev = [h for h in hist if h["type"] == T["type"]]
    return {"number": T["label"], "date": T["draw_date"].isoformat(), "type": T["type"],
            "size": T["size"], "actual": T["crs"], "point": round(p, 1),
            "low": round(lo, 1), "high": round(hi, 1),
            "baseline_last": prev[-1]["crs"] if prev else None}


def run_backtest(rounds, days, ui_types):
    """Every draw of the last `days` days is predicted from the rounds before it, except the
    first-ever draw of a type (the site never predicts a type with no history). Types with fewer
    than BT_MIN_ROWS such draws are extended back in time (not before BT_EARLIEST)."""
    latest = max(r["draw_date"] for r in rounds)
    start = latest - dt.timedelta(days=days - 1)
    seen, first = set(), set()
    for r in rounds:
        if r["type"] not in seen:
            seen.add(r["type"])
            first.add(r["seq"])
    rows, excluded = {}, 0
    for T in rounds:
        if T["draw_date"] < start:
            continue
        if T["seq"] in first:
            excluded += 1
            continue
        rows[T["seq"]] = _backtest_row(rounds, T)
    window = sorted(rows)
    for t in ui_types:
        have = sum(1 for s in window if rounds[s]["type"] == t)
        older = [r for r in rounds if r["type"] == t and BT_EARLIEST <= r["draw_date"] < start
                 and r["seq"] not in first]
        for T in reversed(older):
            if have >= BT_MIN_ROWS:
                break
            rows[T["seq"]] = _backtest_row(rounds, T)
            have += 1
    return start, latest, [rows[s] for s in sorted(rows)], [rows[s] for s in window], excluded


def _metrics(rows):
    if not rows:
        return None
    err = [r["point"] - r["actual"] for r in rows]
    ae = [abs(e) for e in err]
    base = [abs(r["baseline_last"] - r["actual"]) for r in rows if r["baseline_last"] is not None]
    return {"n": len(rows), "from": rows[0]["date"], "to": rows[-1]["date"],
            "mae": round(statistics.fmean(ae), 1),
            "median_abs_error": round(statistics.median(ae), 1),
            "bias": round(statistics.fmean(err), 1) + 0.0, "max_abs_error": round(max(ae), 1),
            "within_5": round(statistics.fmean([1.0 if a <= 5 else 0.0 for a in ae]), 2),
            "coverage": round(statistics.fmean(
                [1.0 if r["low"] <= r["actual"] <= r["high"] else 0.0 for r in rows]), 2),
            "mean_width": round(statistics.fmean([r["high"] - r["low"] for r in rows]), 1),
            "baseline_last_mae": round(statistics.fmean(base), 1) if base else None}


def _confidence(m, hypothetical=False):
    if hypothetical or not m:
        return "low"
    if m["n"] >= 8 and m["mae"] <= 5 and m["coverage"] >= 0.7:
        return "high"
    if m["n"] >= 4 and m["mae"] <= 15:
        return "medium"
    return "low"


_DOWN = {"high": "medium", "medium": "low", "low": "low"}


# --------------------------------------------------------------------------------------------
# Data summaries
# --------------------------------------------------------------------------------------------
def _round_out(r):
    return {"number": r["label"], "date": r["draw_date"].isoformat(), "type": r["type"],
            "type_label": TYPE_PLAIN.get(r["type"], r["type"]), "name": r["name"],
            "size": r["size"], "crs": r["crs"]}


def _nice(n):
    return int(round(n / 250.0) * 250) if n >= 1000 else int(round(n / 10.0) * 10)


def pool_section(ctx):
    snap = ctx["snapshot"]
    bands = [{"band": key, "lo": lo, "hi": hi, "count": snap["dist"][key]}
             for key, lo, hi, _f in harness.BANDS]
    dens = SH.shaped_density(snap["dist"])
    lo_s, hi_s = 300, 700
    acc, tail = sum(dens[hi_s + 1:]), []
    for s in range(hi_s, lo_s - 1, -1):
        acc += dens[s]
        tail.append(int(round(acc)))
    inv = [{"number": i["label"], "date": i["date"].isoformat(), "type": i["type"],
            "size": i["size"], "crs": i["crs"]} for i in ctx["invited_since_snapshot"]]
    return {"as_of": snap["dist_as_on"].isoformat(),
            "reported_by": [str(x) for x in snap["reported_by"]],
            "total": snap["total"], "bands": bands,
            "invited_since": inv,
            "invited_since_total": sum(i["size"] for i in ctx["invited_since_snapshot"]),
            "at_or_above": {"from": lo_s, "to": hi_s, "values": tail[::-1],
                            "method": "IRCC's bands; 501-600 and 351-400 spread with an exponential "
                                      "shape, 601+ with IRCC's nominee score shape, other bands evenly"}}


def quota_section(rounds, cal):
    latest = cal["latest"]
    year = latest.year
    md = (latest.month, latest.day)

    def upto(y):
        sel = [r for r in rounds if r["draw_date"].year == y and
               (r["draw_date"].month, r["draw_date"].day) <= md]
        return {"year": y, "invited": sum(r["size"] for r in sel), "draws": len(sel)}

    by_type = {}
    for r in rounds:
        if r["draw_date"].year == year:
            b = by_type.setdefault(r["type"], {"type": r["type"], "label": TYPE_PLAIN.get(r["type"], r["type"]),
                                               "invited": 0, "draws": 0})
            b["invited"] += r["size"]
            b["draws"] += 1
    full = [{"year": y, "invited": sum(r["size"] for r in rounds if r["draw_date"].year == y),
             "draws": sum(1 for r in rounds if r["draw_date"].year == y)}
            for y in (year - 3, year - 2, year - 1)]
    starts = sorted({r["draw_date"] for r in rounds if r["type"] == "pnp" and r["draw_date"].year >= year - 1})
    cyc = [sum(r["size"] for r in rounds if a <= r["draw_date"] < b)
           for a, b in zip(starts, starts[1:] + [latest + dt.timedelta(days=1)])]
    last8 = cyc[-8:]
    per_cycle = statistics.fmean(last8) if last8 else 0.0
    # remaining cycles this year: regular two-week slots, none in the pause, none from Dec 22;
    # a slot starting Dec 15-21 counts 30%, the others 90% (IRCC sometimes skips a cycle)
    p_sum, slots, s = 0.0, [], cal["next_start"]
    while s.year == year and (s.month, s.day) < HOLIDAY_STOP[0]:
        if cal["post_regime"] or not _in_pause(s):
            p = 0.3 if (s.month == 12 and s.day >= 15) else 0.9
            slots.append({"start": s.isoformat(), "probability": p})
            p_sum += p
        s += dt.timedelta(days=cal["gap"])
    rnd = lambda v: int(round(v / 1000.0) * 1000)  # noqa: E731
    return {
        "year": year, "through": latest.isoformat(),
        "invited_ytd": upto(year)["invited"], "draws_ytd": upto(year)["draws"],
        "invited_ytd_by_type": sorted(by_type.values(), key=lambda b: (-b["invited"], b["type"])),
        "same_period_previous_years": [upto(y) for y in (year - 3, year - 2, year - 1)],
        "full_previous_years": full,
        "cycles": {"typical_days": cal["gap"],
                   "count_this_year": sum(1 for d in starts if d.year == year),
                   "avg_invited_last_8": int(round(per_cycle)), "last_8": last8},
        "rest_of_year": {"expected_cycles": round(p_sum, 1), "slots": slots,
                         "invited_estimate": rnd(p_sum * per_cycle),
                         "invited_low": rnd(max(0.0, p_sum - 0.75) * per_cycle),
                         "invited_high": rnd((p_sum + 0.75) * per_cycle),
                         "method": "remaining two-week cycles (none during the Nov 3 pause) x average "
                                   "invitations in the last 8 cycles; rough: IRCC does not announce sizes"},
        "facts": QUOTA_FACTS,
    }


# --------------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------------
def _same_type(rounds, t):
    return [r for r in rounds if r["type"] == t]


def _typical_gap(same, latest):
    recent = [r for r in same if (latest - r["draw_date"]).days <= 730][-6:]
    gaps = [(b["draw_date"] - a["draw_date"]).days for a, b in zip(recent, recent[1:])]
    return int(statistics.median(gaps)) if gaps else None


def build(raw_bytes, previous=None):
    data, warnings = validate_raw(raw_bytes, previous)
    rounds = parse_rounds(data)
    for r in rounds:
        if r["type"] == "other":
            warnings.append("#%s: unknown draw type %r (not modelled)" % (r["label"], r["name"]))
    cal = cycle_calendar(rounds)
    base_ctx = harness.make_ctx(rounds, cal["latest"] + dt.timedelta(days=1))
    if base_ctx["snapshot"] is None:
        raise Refuse("no usable pool distribution in the data")
    horizon = cal["latest"] - dt.timedelta(days=ACTIVE_DAYS - 1)

    # 1. which types to show, their default N and expected next-cycle slot
    ui_types, default_n, slot = [], {}, {}
    for t in UI_ORDER:
        recent = [r for r in _same_type(rounds, t) if r["draw_date"] >= horizon]
        if t != "general" and not (recent and statistics.median(
                r["size"] for r in recent[-3:]) >= MIN_TYPICAL_SIZE):
            continue
        ui_types.append(t)
        default_n[t] = _nice(statistics.median(r["size"] for r in recent[-3:])) if recent else 2000
        slot[t] = next_cycle_slot(rounds, t, cal)

    # 2. walk-forward backtest (sets the confidence labels)
    bt_from, bt_to, bt_rows, bt_window, bt_excluded = run_backtest(rounds, BACKTEST_DAYS, ui_types)
    bt_by_type = {}
    for r in bt_rows:
        bt_by_type.setdefault(r["type"], []).append(r)

    # 3. scenarios. "current": each type's next slot (one simulated cycle per cycle start).
    #    "after_nov3" (only while no post-change draw exists): the first cycle after the pause,
    #    assuming the pre-change cycle before it (if any) happens as expected.
    pre = not cal["post_regime"]
    sims = {}
    for t in ui_types:
        d, s = slot[t]
        if s not in sims and (not pre or d < CHANGE_DATE):
            sims[s] = CycleSim(rounds, s, default_n, True, until=CHANGE_DATE if pre else None)
    after_sim, base_added = None, []
    if pre:
        base_after = rounds
        for s in sorted({s for d, s in slot.values() if d < CHANGE_DATE}):
            sim = CycleSim(base_after, s, default_n, True, until=CHANGE_DATE)
            base_after = sim.history_for(None)
            base_added += sim.added
        after_sim = CycleSim(base_after, cal["after_pause_start"], default_n, True)
    next_start = min(s for _d, s in slot.values())

    types_out = []
    for t in ui_types:
        info = TYPE_INFO.get(t, {"label": TYPE_PLAIN.get(t, t), "short": t, "step": 10,
                                 "grid": (100, 250, 500, 1000, 2000), "note": None})
        same = _same_type(rounds, t)
        recent = [r for r in same if r["draw_date"] >= horizon]
        grid = sorted(set(info["grid"]) | {default_n[t]})
        bt = _metrics(bt_by_type.get(t, []))
        hypothetical = t == "general" and not recent
        conf = _confidence(bt, hypothetical)
        date, start = slot[t]
        cur_applies = date < CHANGE_DATE
        scen, order = {}, []
        if pre:
            if cur_applies:
                g, _w, fails = predict_grid(sims[start].history_for(t), date, t, grid)
                scen["current"] = {
                    "label": "Current rules", "rules": "before_nov3", "applies": True,
                    "expected_date": date.isoformat(),
                    "window": [start.isoformat(), (start + dt.timedelta(days=4)).isoformat()],
                    "confidence": conf, "shift_weight": 0.0, "grid": g, "member_failures": fails}
            adate = cal["after_pause_start"] + _offset(t)
            ah = after_sim.history_for(t)
            if not cur_applies:   # no draw expected before Nov 3: old rules for comparison only
                g, _w, fails = predict_grid(ah, adate, t, grid, apply_change=False)
                scen["current"] = {
                    "label": "Old rules (comparison only)", "rules": "before_nov3", "applies": False,
                    "expected_date": adate.isoformat(), "window": None,
                    "confidence": conf, "shift_weight": 0.0, "grid": g, "member_failures": fails}
            g2, w2, fails2 = predict_grid(ah, adate, t, grid)
            scen["after_nov3"] = {
                "label": "After the Nov 3 change", "rules": "after_nov3", "applies": True,
                "expected_date": adate.isoformat(),
                "window": [d.isoformat() for d in cal["after_pause_window"]],
                "confidence": _DOWN[conf], "shift_weight": round(w2, 2),
                "grid": g2, "member_failures": fails2}
            order = ["current", "after_nov3"]
            default_scen = "current" if cur_applies else "after_nov3"
        else:
            g, w, fails = predict_grid(sims[start].history_for(t), date, t, grid)
            k_post = post_change_count(rounds, t)
            scen["current"] = {
                "label": "New rules (from Nov 3, 2026)", "rules": "after_nov3", "applies": True,
                "expected_date": date.isoformat(),
                "window": [start.isoformat(), (start + dt.timedelta(days=4)).isoformat()],
                "confidence": conf if (k_post >= NOV3_RECAL_DRAWS or t == "general") else _DOWN[conf],
                "shift_weight": round(w, 2), "post_change_draws": k_post,
                "grid": g, "member_failures": fails}
            order, default_scen = ["current"], "current"
        note = info["note"]
        if t == "general":
            note = ("IRCC has not held an all-program draw since %s. This shows what the cut-off would be "
                    "if, in the next cycle, IRCC replaced its CEC draw with an all-program draw of N (after "
                    "its usual PNP draw). It is a what-if, not a forecast that such a draw will happen."
                    % (same[-1]["draw_date"].isoformat() if same else "2015")) \
                if hypothetical else "All-program draw: every candidate in the pool is eligible."
        types_out.append({
            "id": t, "label": info["label"], "short": info["short"], "note": note,
            "hypothetical": hypothetical, "confidence": conf,
            "default_n": default_n[t], "n_min": grid[0], "n_max": grid[-1], "n_step": info["step"],
            "recent_sizes": [r["size"] for r in recent[-6:]],
            "last_draw": _round_out(same[-1]) if same else None,
            "typical_gap_days": None if hypothetical else _typical_gap(same, cal["latest"]),
            "default_scenario": default_scen, "scenario_order": [k for k in order if k in scen],
            "scenarios": scen, "backtest": bt,
        })

    snap = base_ctx["snapshot"]
    latest_r = rounds[-1]
    assumed = {"current": [a for s in sorted(sims) for a in sims[s].added]}
    if after_sim is not None:
        assumed["after_nov3"] = base_added + after_sim.added
    return {
        "schema": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "provenance": {
            "source": SOURCE_URL,
            "source_note": "IRCC rounds of invitations data. Statistics Canada does not publish "
                           "Express Entry draw or pool data.",
            "input_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "rounds": len(rounds),
            "latest_draw": dict(_round_out(latest_r), url=ROUND_URL + latest_r["label"]),
            "pool_as_of": snap["dist_as_on"].isoformat(),
            "pool_rescored": snap["dist_as_on"] >= CHANGE_DATE,
            "regime": "after_nov3" if cal["post_regime"] else "before_nov3",
        },
        "method": {
            "summary": "Median of three independent models, each tested on past draws: a queue model of "
                       "each draw type's eligible candidates, a pool model built on IRCC's score "
                       "distribution, and a statistical model of past cut-offs.",
            "members": [m for m, _mod in MEMBERS],
            "interval_level": INTERVAL_LEVEL,
            "interpolate": "linear in ln(N) between grid rows; flat outside n_min..n_max",
            "assumed_draws": assumed,
            "engine_sha256": {},
        },
        "timing": {
            "last_cycle_start": cal["last_start"].isoformat(), "typical_cycle_days": cal["gap"],
            "next_cycle_start": next_start.isoformat(),
            "change": {"date": CHANGE_DATE.isoformat(), "pause_max_days": PAUSE_MAX_DAYS,
                       "status": "in_effect" if cal["post_regime"] else "upcoming",
                       "first_cycle_after_pause": cal["after_pause_start"].isoformat(),
                       "after_pause_window": [d.isoformat() for d in cal["after_pause_window"]],
                       "sources": CHANGE_SOURCES},
        },
        "pool": pool_section(base_ctx),
        "types": types_out,
        "recent_draws": [_round_out(r) for r in reversed(rounds) if r["draw_date"] >= horizon],
        "quota": quota_section(rounds, cal),
        "backtest": {"from": bt_from.isoformat(), "to": bt_to.isoformat(),
                     "excluded_first_draws": bt_excluded,
                     "all": _metrics(bt_window),
                     "by_type": {t: _metrics(v) for t, v in sorted(bt_by_type.items())},
                     "rows": [[r["number"], r["date"], r["type"], r["size"], r["actual"],
                               r["point"], r["low"], r["high"]] for r in bt_rows]},
        "warnings": warnings,
    }


def _engine_hashes():
    out = {}
    for name in ENGINE_FILES:
        with open(os.path.join(ENGINE, name + ".py"), "rb") as fh:
            out[name] = hashlib.sha256(fh.read()).hexdigest()
    return out


def render(result):
    result["method"]["engine_sha256"] = _engine_hashes()
    return json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("rounds_json")
    ap.add_argument("out_json")
    ap.add_argument("--previous", help="the currently published file (rollback guard)")
    a = ap.parse_args(argv)
    with open(a.rounds_json, "rb") as fh:
        raw = fh.read()
    previous = None
    if a.previous and os.path.exists(a.previous):
        try:
            with open(a.previous, encoding="utf-8") as fh:
                previous = json.load(fh)
        except (OSError, json.JSONDecodeError):
            previous = None
    try:
        result = build(raw, previous)
    except Refuse as exc:
        print("::error::Refusing to publish prediction data: %s" % exc)
        return 1
    text = render(result)
    old = None
    if os.path.exists(a.out_json):
        with open(a.out_json, encoding="utf-8") as fh:
            old = fh.read()
    changed = text != old
    if changed:
        d = os.path.dirname(os.path.abspath(a.out_json))
        os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.chmod(tmp, 0o644)
        os.replace(tmp, a.out_json)
    for w in result["warnings"]:
        print("::warning::%s" % w)
    gh = os.environ.get("GITHUB_OUTPUT")
    if gh:
        with open(gh, "a", encoding="utf-8") as fh:
            fh.write("changed=%s\nlatest=%s\n" % ("true" if changed else "false",
                                                  result["provenance"]["latest_draw"]["number"]))
    print("latest draw #%s, %d types, %d bytes, changed=%s" % (
        result["provenance"]["latest_draw"]["number"], len(result["types"]), len(text), changed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
