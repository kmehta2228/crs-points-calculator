"""Shape-aware pool walk (stdlib only).

Mechanism: a draw of type T invites the top N T-eligible candidates, so its cut-off is the
score c where the T-eligible count at or above c reaches N.  Three pieces:

1. Within-band shape (the main fix over a uniform pool walk).  Wide bands are not uniform:
   - 501-600: exponential decay n(s) = d0 * r^(s-501), with d0 continuous with the 491-500
     band's average density and r solved so the band total is matched exactly.
   - 351-400: the same idea mirrored downward from the 401-410 density.
   - 601-1200: fixed nominee shape from IRCC open data (ITAs by CRS score, 2022-2024).
   - 10-point bands: uniform (as the harness).
2. Pool at the target date = shaped snapshot - invitations since the snapshot (PNP from 601+,
   all-program from the top, others proportionally above their cut-off) + per-band net inflow
   x days since the snapshot (inflow estimated from earlier snapshot pairs in history).
3. Eligibility for type T: e(s) = f_T(s) * D_T(s) * pool(s) with
   f_T(s) = min(1, exp(a_T + b_T (s - s_ref)))  (score-dependent eligible share), and a
   refill clock D_T(s) = 1 - exp(-age(s)/tau_T), where age(s) is the days since score s was
   last cleared by a T (or all-program) draw with cut-off <= s.  (a_T, b_T, tau_T) are fitted
   on the type's own history (N, cut-off, pool then) by recency-weighted least squares in
   score space with weak priors.  The clock is what makes a bigger N lower the cut-off only
   modestly (dense un-cleared backlog below the last cut-off) while a smaller N raises it
   faster (only the refill above the last cut-off).
   General (all-program) draws: f = 1, no clock: a pure shaped pool walk.
   PNP and physicians: invited pools are tiny and fully invited, so the cut-off is the lowest
   eligible score: a min-of-N tail model c = alpha - theta ln N (theta shrunk to a prior).

Intervals: walk-forward residuals of this same model on the type's previous draws (each
computed only from data before that draw, cached), scaled by a factor tuned on 2025.

Hyperparameters (CFG) were tuned on the 2025 window only: half_life 45 d, k_fit 8, b0 -0.02,
int_mult 0.85, k_resid 12.  Shape constants (kappa = 1, exponential continuity) were not
tuned; the 601+ nominee shape comes from IRCC open data for 2022-2024 and the PNP slope prior
(theta0 = 35) from 2021-2024 draws.  Nominees (601+) are excluded for CEC/category draws
(each cycle's PNP draw invites them first) and included for all-program draws.
A type never drawn before falls back to the median of the last 5 occupation-category
cut-offs with a +-30 interval.  Driver for next-draw numbers and what-if grids:
approach_pool_walk_shaped_nextdraw.py.
"""
import math
import statistics

try:
    import harness
except ImportError:  # imported outside the harness CLI under -I
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import harness

NAME = "pool_walk_shaped (exp-shaped bands + refill-clock eligibility + PNP tail)"

CFG = {
    "kappa_501": 1.0,        # d(501) = kappa * mean density of 491-500
    "kappa_400": 1.0,        # d(400) = kappa * mean density of 401-410
    "inflow_window": 70,     # days of snapshot history used to estimate inflow
    "inflow_scale": 1.0,     # multiplier on estimated inflow
    "half_life": 45.0,       # days, recency weight for the eligibility fit
    "k_fit": 8,              # same-type draws used in the fit
    "version_weight": 0.5,   # weight multiplier for draws of an older category version
    "tau_grid": (2.0, 5.0, 10.0, 15.0, 21.0, 30.0, 45.0, 60.0, 90.0, 135.0, 200.0, 365.0, 1e6),
    "b_grid": (-0.08, -0.06, -0.045, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.035, 0.05),
    "tau0": 45.0,            # prior centre for tau
    "prior_tau": 2.0,        # prior strength on ln(tau/tau0) (score^2 units per unit^2)
    "prior_b": 4.0,          # prior strength on ((b - b0) / 0.02)
    "b0": -0.02,             # prior centre for b (categories: share rises at lower scores)
    "carry": 0.0,            # share of the last walk-forward residual carried into the point
    "nominee_cap": 0.9,      # general draws: expected 601+ count capped at this share of n
    "cross_cec": 0.0,        # categories: share of eligibles also cleared by CEC draws
    "cross_tau": 30.0,       # refill time (days) of that CEC-cleared share
    # PNP / physicians min-of-N tail model
    "theta0": 35.0,
    "theta_prior_n": 12.0,   # prior weight (in pseudo-draws) on theta0
    "pnp_half_life": 120.0,
    "k_pnp": 30,
    # intervals
    "k_resid": 12,
    "min_resid": 4,
    "resid_half_life": 365.0,
    "int_q": 0.8,
    "int_mult": 0.85,
    "int_floor": 3.0,
    "int_cap": 60.0,
    "int_cap_tail": 90.0,
    "int_default": 30.0,
    "extrap_mult": 0.5,
    "k_range": 8,            # recent same-type draws defining the "observed N" range
}

PROGRAM_TYPES = ("pnp", "cec", "general", "fsw", "fst")
TAIL_TYPES = ("pnp", "physicians")
CLEAR_ALL = ("general",)        # all-program draws clear every type
X_MAX = 700                     # eligibility sums use scores up to here (601+ lumped)

# Nominee / 601+ shape from IRCC open data "Express Entry ITAs by CRS score" 2022-2024
# (PNP + CEC + FSW invitations with 601+): shares of the 601-1200 band per 100 points.
NOMINEE_SHAPE = ((601, 700, 0.0141), (701, 800, 0.0042), (801, 900, 0.0476),
                 (901, 1000, 0.4927), (1001, 1100, 0.4390), (1101, 1200, 0.0025))

_CACHE = {}


# --------------------------------------------------------------------------------------
# 1. within-band shape
# --------------------------------------------------------------------------------------
def _geom_ratio(target, n):
    """r > 0 with (1 - r^n)/(1 - r) = target (sum of n geometric terms, first term 1)."""
    if target <= 1.0:
        return 1e-9
    if abs(target - n) < 1e-9:
        return 1.0

    def S(r):
        return n if abs(r - 1.0) < 1e-12 else (1.0 - r ** n) / (1.0 - r)
    if target < n:
        lo, hi = 1e-9, 1.0
    else:
        lo, hi = 1.0, 1.05
        while S(hi) < target:
            hi = 1.0 + (hi - 1.0) * 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if S(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _exp_band(d, total, lo, hi, d_edge, downward=False):
    """Fill d[lo..hi] with a geometric profile summing to `total`, whose value at the edge
    next to the reference band is d_edge (edge = lo, or hi if downward)."""
    n = hi - lo + 1
    if total <= 0:
        for s in range(lo, hi + 1):
            d[s] = 0.0
        return
    if d_edge <= 0:
        per = total / n
        for s in range(lo, hi + 1):
            d[s] = per
        return
    r = _geom_ratio(total / d_edge, n)
    w = [r ** k for k in range(n)]
    sw = sum(w)
    for k in range(n):
        s = (hi - k) if downward else (lo + k)
        d[s] = total * w[k] / sw


def shaped_density(dist, cfg=CFG):
    """Band dict -> 1201 per-score counts with exponential wide bands (see module doc)."""
    key = ("shape", tuple(dist[k] for k in harness.BAND_KEYS), cfg["kappa_501"], cfg["kappa_400"])
    hit = _CACHE.get(key)
    if hit is not None:
        return list(hit)
    d = harness.density(dist)
    # 501-600, continuous with 491-500
    _exp_band(d, dist["501-600"], 501, 600, cfg["kappa_501"] * dist["491-500"] / 10.0)
    # 351-400, continuous (downward) with 401-410
    _exp_band(d, dist["351-400"], 351, 400, cfg["kappa_400"] * dist["401-410"] / 10.0,
              downward=True)
    # 601-1200 nominee shape
    tot = dist["601-1200"]
    for s in range(601, 1201):
        d[s] = 0.0
    for lo, hi, share in NOMINEE_SHAPE:
        per = tot * share / (hi - lo + 1)
        for s in range(lo, hi + 1):
            d[s] = per
    _CACHE[key] = tuple(d)
    return d


def _band_of(s):
    for k, lo, hi, _f in harness.BANDS:
        if lo <= s <= hi:
            return k
    return None


def _shape_weights(dist_shaped, band):
    lo, hi = harness.BAND_RANGE[band]
    seg = dist_shaped[lo:hi + 1]
    t = sum(seg)
    if t <= 0:
        return [1.0 / (hi - lo + 1)] * (hi - lo + 1)
    return [v / t for v in seg]


# --------------------------------------------------------------------------------------
# 2. pool at the target date
# --------------------------------------------------------------------------------------
def _remove_draw(d, typ, size, crs):
    """Remove one draw's invitees from a density list in place."""
    if size is None or size <= 0:
        return
    if typ == "pnp" or (crs is not None and crs >= 601):
        rem = float(size)
        for s in range(1200, 600, -1):
            if rem <= 0:
                break
            t = min(d[s], rem)
            d[s] -= t
            rem -= t
        return
    if typ == "general":
        rem = float(size)
        for s in range(1200, -1, -1):
            if rem <= 0:
                break
            t = min(d[s], rem)
            d[s] -= t
            rem -= t
        return
    start = max(0, int(math.ceil(crs)))
    hi = 600  # program/category draws: nominees (601+) are invited by PNP draws first
    avail = sum(d[start:hi + 1])
    if avail <= 0:
        return
    f = min(1.0, size / avail)
    for s in range(start, hi + 1):
        d[s] *= (1.0 - f)


def _snapshots(history):
    """Distinct snapshots in history: list of (as_on, dist, first_index) by as-on date."""
    out, seen = [], {}
    for i, h in enumerate(history):
        if h["dist"] is None or h["dist_as_on"] is None:
            continue
        k = tuple(h["dist"][b] for b in harness.BAND_KEYS)
        if k in seen:
            j = seen[k]
            if h["dist_as_on"] < out[j][0]:
                out[j] = (h["dist_as_on"], out[j][1], out[j][2])
            continue
        seen[k] = len(out)
        out.append((h["dist_as_on"], h["dist"], i))
    out.sort(key=lambda x: (x[0], x[2]))
    return out


def _inflow(history, snap, cfg=CFG):
    """Per-band net inflow per day, from the latest snapshot and one ~inflow_window days
    older (both in history), adding back the invitations made in between."""
    snaps = _snapshots(history)
    if len(snaps) < 2:
        return None
    t_l = snap["dist_as_on"]
    latest = [s for s in snaps if s[0] <= t_l]
    if len(latest) < 2:
        return None
    L = latest[-1]
    older = [s for s in latest[:-1] if (t_l - s[0]).days >= 21]
    if not older:
        return None
    O = min(older, key=lambda s: abs((t_l - s[0]).days - cfg["inflow_window"]))
    days = (L[0] - O[0]).days
    if days <= 0:
        return None
    sO = shaped_density(O[1], cfg)
    keyO = tuple(O[1][b] for b in harness.BAND_KEYS)
    keyL = tuple(L[1][b] for b in harness.BAND_KEYS)
    removed = {b: 0.0 for b in harness.BAND_KEYS}
    for h in history:
        d = h["draw_date"]
        if d is None or h["size"] is None or h["crs"] is None:
            continue
        rep = h["dist"] is not None and tuple(h["dist"][b] for b in harness.BAND_KEYS)
        after_O = d > O[0] or (d == O[0] and rep == keyO)
        before_L = d < L[0] and rep != keyL
        if not (after_O and before_L):
            continue
        typ, n, c = h["type"], h["size"], h["crs"]
        if typ == "pnp" or c >= 601:
            removed["601-1200"] += n
            continue
        hi = 1200 if typ == "general" else 600
        start = max(0, int(c))
        tot = sum(sO[start:hi + 1])
        if tot <= 0:
            continue
        for k, lo, bhi, _f in harness.BANDS:
            a, b = max(lo, start), min(bhi, hi)
            if a <= b:
                removed[k] += n * sum(sO[a:b + 1]) / tot
    return {k: (L[1][k] - O[1][k] + removed[k]) / days for k in harness.BAND_KEYS}


def _add_inflow(d, rates, days, shape_ref, cfg=CFG):
    if not rates or days <= 0 or cfg["inflow_scale"] <= 0:
        return
    for k, lo, hi, _f in harness.BANDS:
        add = rates[k] * days * cfg["inflow_scale"]
        if add == 0:
            continue
        if k in ("501-600", "351-400", "601-1200"):
            w = _shape_weights(shape_ref, k)
        else:
            w = [1.0 / (hi - lo + 1)] * (hi - lo + 1)
        for i, s in enumerate(range(lo, hi + 1)):
            d[s] = max(0.0, d[s] + add * w[i])


def pool_at(ctx, cfg=CFG):
    """Shaped pool expected at ctx['target_date'] (density list).  Chronological: inflow
    accrues day by day and each draw since the snapshot removes its invitees from the pool
    as it stood on its own date (so an all-program draw also takes nominees who arrived
    after the snapshot)."""
    snap = ctx["snapshot"]
    shape_ref = shaped_density(snap["dist"], cfg)
    d = list(shape_ref)
    rates = _inflow(ctx["history"], snap, cfg) if (ctx["days_since_snapshot"] or 0) > 0 else None
    t = snap["dist_as_on"]
    for inv in sorted(ctx["invited_since_snapshot"], key=lambda x: x["date"]):
        gap = (inv["date"] - t).days
        if gap > 0:
            _add_inflow(d, rates, gap, shape_ref, cfg)
            t = inv["date"]
        _remove_draw(d, inv["type"], inv["size"], inv["crs"])
    gap = (ctx["target_date"] - t).days
    if gap > 0:
        _add_inflow(d, rates, gap, shape_ref, cfg)
    return d


# --------------------------------------------------------------------------------------
# history states (cached by a fingerprint of everything before the draw)
# --------------------------------------------------------------------------------------
def _prefix_fps(history):
    fps, h = [], 0
    for r in history:
        dk = tuple(r["dist"][b] for b in harness.BAND_KEYS) if r["dist"] else None
        h = hash((h, r["label"], r["type"], r["size"], r["crs"], r["draw_date"],
                  r["dist_as_on"], dk))
        fps.append(h)
    return fps  # fps[i] covers history[:i+1]


def _fp_before(fps, i):
    return fps[i - 1] if i > 0 else 0


def _ages(history_before, types, date):
    """age[s] = days since score s was last cleared by a draw of `types` (cut-off <= s);
    None when never cleared."""
    ev = [(h["crs"], h["draw_date"]) for h in history_before
          if h["type"] in types and h["crs"] is not None and h["draw_date"] is not None]
    ev.sort()
    ages = [None] * (harness.MAX_SCORE + 1)
    best, j = None, 0
    for s in range(harness.MAX_SCORE + 1):
        while j < len(ev) and ev[j][0] <= s:
            if best is None or ev[j][1] > best:
                best = ev[j][1]
            j += 1
        if best is not None:
            ages[s] = (date - best).days
    return ages


def _state(history, i, fps, cfg=CFG):
    """Pool and clearing ages at history[i] (built only from history[:i])."""
    key = ("state", _fp_before(fps, i), history[i]["label"], cfg["kappa_501"],
           cfg["kappa_400"], cfg["inflow_window"], cfg["inflow_scale"])
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    r = history[i]
    sub = history[:i]
    c = harness.make_ctx(sub, r["draw_date"])
    st = None
    if c["snapshot"] is not None:
        st = {"pool": pool_at(c, cfg), "date": r["draw_date"]}
    _CACHE[key] = st
    return st


def _clear_types(T):
    return (T,) + CLEAR_ALL


def _ages_cached(history, i, fps, T, date):
    key = ("ages", _fp_before(fps, i), T, date)
    hit = _CACHE.get(key)
    if hit is None:
        hit = _ages(history[:i], _clear_types(T), date)
        _CACHE[key] = hit
    return hit


def _cross_factor(history_before, T, date, cfg=CFG):
    """Occupation/language categories: CEC draws also invite the CEC-eligible members of the
    category above the CEC cut-off, so a share phi of the category is depleted there and
    refills with time constant cross_tau.  Returns a per-score multiplier (or None)."""
    phi = cfg["cross_cec"]
    if phi <= 0 or T in PROGRAM_TYPES:
        return None
    ages = _ages(history_before, ("cec",), date)
    tau = cfg["cross_tau"]
    return [1.0 - phi * (1.0 - _clock(a, tau)) for a in ages]


def _cross_cached(history, i, fps, T, date, cfg=CFG):
    key = ("cross", _fp_before(fps, i), T, date, cfg["cross_cec"], cfg["cross_tau"])
    if key not in _CACHE:
        _CACHE[key] = _cross_factor(history[:i], T, date, cfg)
    return _CACHE[key]


def _apply(pool, factor):
    """Multiply by an optional per-score factor; nominees (601+) are zeroed because each
    cycle's PNP draw invites them before the CEC/category draws of that cycle."""
    out = list(pool) if factor is None else [p * f for p, f in zip(pool, factor)]
    for s in range(601, len(out)):
        out[s] = 0.0
    return out


# --------------------------------------------------------------------------------------
# 3. eligibility fit and solve
# --------------------------------------------------------------------------------------
def _clock(age, tau):
    if age is None or tau >= 1e5:
        return 1.0
    if age <= 0:
        return 1e-3
    return 1.0 - math.exp(-age / tau)


def _base(pool, ages, tau, lo):
    """D(s) * pool(s) for s in [lo, X_MAX], 601+ lumped into the 601..X_MAX range."""
    out = []
    for s in range(lo, X_MAX + 1):
        out.append(_clock(ages[s], tau) * pool[s])
    top = sum(pool[X_MAX + 1:])
    if top > 0 and X_MAX >= lo:
        out[-1] += top * _clock(ages[X_MAX], tau)
    return out


def _cum(vals, lo, b, s_ref, c):
    """Sum_{s >= c} exp(b (s - s_ref)) vals[s - lo] for real c (linear within a score)."""
    if c < lo:
        c = lo
    i = int(math.ceil(c))
    tot = 0.0
    for s in range(i, lo + len(vals)):
        tot += math.exp(b * (s - s_ref)) * vals[s - lo]
    if i != c and i - 1 >= lo:
        tot += math.exp(b * (i - 1 - s_ref)) * vals[i - 1 - lo] * (i - c)
    return tot


def _fit(rows, cfg=CFG):
    """rows: dicts with N, c (integer cut-off), w, pool, ages.  Returns (a, b, tau, s_ref, J)."""
    from operator import mul
    s_ref = rows[-1]["c"]
    W = sum(r["w"] for r in rows)
    lnN = [math.log(r["N"]) for r in rows]
    ebs = {b: [math.exp(b * (s - s_ref)) for s in range(X_MAX + 1)] for b in cfg["b_grid"]}
    best = None
    for tau in cfg["tau_grid"]:
        bases = []
        for r in rows:
            c = int(r["c"])
            lo = max(0, c - 2)
            bases.append((c, lo, _base(r["pool"], r["ages"], tau, lo)))
        for b in cfg["b_grid"]:
            eb = ebs[b]
            lnS, k = [], []
            ok = True
            for (c, lo, v) in bases:
                if c + 2 > X_MAX or c - 2 < 0:
                    ok = False
                    break
                sp = sum(map(mul, eb[c + 2:X_MAX + 1], v[c + 2 - lo:]))
                s0 = sp + eb[c] * v[c - lo] + eb[c + 1] * v[c + 1 - lo]
                sm = s0 + eb[c - 2] * v[c - 2 - lo] + eb[c - 1] * v[c - 1 - lo]
                if s0 <= 0 or sp <= 0:
                    ok = False
                    break
                lnS.append(math.log(s0))
                k.append(max(1e-3, (math.log(sm) - math.log(sp)) / 4.0))
            if not ok:
                continue
            a = sum(r["w"] * (ln - l) for r, ln, l in zip(rows, lnN, lnS)) / W
            J = sum(r["w"] * ((ln - a - l) / kk) ** 2
                    for r, ln, l, kk in zip(rows, lnN, lnS, k)) / W
            J += cfg["prior_tau"] * math.log(min(tau, 3650.0) / cfg["tau0"]) ** 2
            J += cfg["prior_b"] * ((b - cfg["b0"]) / 0.02) ** 2
            if best is None or J < best[0]:
                best = (J, a, b, tau)
    if best is None:
        return None
    return best[1], best[2], best[3], s_ref, best[0]


def _solve(pool, ages, a, b, tau, s_ref, n):
    e = [0.0] * (harness.MAX_SCORE + 1)
    for s in range(harness.MAX_SCORE + 1):
        if pool[s] <= 0:
            continue
        f = math.exp(min(0.0, a + b * (s - s_ref)))
        e[s] = f * _clock(ages[s], tau) * pool[s]
    return harness.score_for_count(e, n, integer=False), e


# --------------------------------------------------------------------------------------
# core predictor (no interval)
# --------------------------------------------------------------------------------------
def _same_type_idx(history, T):
    return [i for i, h in enumerate(history) if h["type"] == T and h["crs"] is not None
            and h["size"]]


def _tail_predict(history, T, n, date, cfg=CFG):
    idx = _same_type_idx(history, T)[-cfg["k_pnp"]:]
    if not idx:
        return None
    xs, ys, ws = [], [], []
    for i in idx:
        h = history[i]
        xs.append(math.log(h["size"]))
        ys.append(h["crs"])
        ws.append(0.5 ** ((date - h["draw_date"]).days / cfg["pnp_half_life"]))
    W = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / W
    my = sum(w * y for w, y in zip(ws, ys)) / W
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    # shrink slope towards -theta0 (theta_prior_n pseudo-draws with the data's mean spread)
    neff = W * W / sum(w * w for w in ws)
    var_x = sxx / W if W > 0 else 0.0
    pw = cfg["theta_prior_n"] * max(var_x, 0.05) * W / max(neff, 1.0)
    slope = (sxy - pw * cfg["theta0"]) / (sxx + pw)
    slope = min(0.0, slope)
    point = my + slope * (math.log(n) - mx)
    return point, {"model": "tail", "slope": round(slope, 2), "level": round(my, 1),
                   "draws": len(idx)}


def _general_predict(ctx, n, cfg=CFG):
    """All candidates eligible: pure shaped pool walk.  The expected nominee count (601+)
    is capped at nominee_cap * n: IRCC has always sized all-program draws above the number
    of nominees waiting (no all-program cut-off above 600 since 2015), so the chosen n
    itself says the 601+ group is smaller than n; without the cap an over-estimated nominee
    inflow would throw the cut-off into the 900s."""
    pool = pool_at(ctx, cfg)
    m = sum(pool[601:])
    cap = cfg["nominee_cap"] * n
    capped = m > cap
    if capped and m > 0:
        k = cap / m
        for s in range(601, harness.MAX_SCORE + 1):
            pool[s] *= k
    return harness.score_for_count(pool, n, integer=False), {
        "model": "pool_all", "nominees_est": round(m), "nominee_cap_applied": capped}


def _fallback(history, T, n):
    """A type never drawn before: median of the last 5 occupation-category cut-offs
    (French excluded: a language category sits far lower)."""
    cats = [h for h in history if h["type"] not in PROGRAM_TYPES + ("french",)
            and h["crs"] is not None]
    src = cats[-5:] or history[-5:]
    return statistics.median([h["crs"] for h in src]), {"model": "fallback_category_median"}


def predict_core(ctx, T, n, cfg=CFG, fps=None):
    history = ctx["history"]
    date = ctx["target_date"]
    if T in TAIL_TYPES:
        out = _tail_predict(history, T, n, date, cfg)
        return out if out else _fallback(history, T, n)
    if ctx["snapshot"] is None:
        h = harness.same_type_history(ctx, T, 1)
        return (h[-1]["crs"], {"model": "no_snapshot_last"}) if h else _fallback(history, T, n)
    if T == "general":
        return _general_predict(ctx, n, cfg)
    fps = fps or _prefix_fps(history)
    idx = [i for i in _same_type_idx(history, T)][-cfg["k_fit"]:]
    rows = []
    latest_version = history[idx[-1]]["version"] if idx else None
    for i in idx:
        st = _state(history, i, fps, cfg)
        if st is None:
            continue
        h = history[i]
        w = 0.5 ** ((date - h["draw_date"]).days / cfg["half_life"])
        if h["version"] != latest_version:
            w *= cfg["version_weight"]
        rows.append({"N": h["size"], "c": h["crs"], "w": w,
                     "pool": _apply(st["pool"], _cross_cached(history, i, fps, T, h["draw_date"], cfg)),
                     "ages": _ages_cached(history, i, fps, T, h["draw_date"])})
    if not rows:
        return _fallback(history, T, n)
    fit = _fit(rows, cfg)
    if fit is None:
        return _fallback(history, T, n)
    a, b, tau, s_ref, _J = fit
    pool = _apply(pool_at(ctx, cfg), _cross_factor(history, T, date, cfg))
    ages = _ages(history, _clear_types(T), date)
    point, e = _solve(pool, ages, a, b, tau, s_ref, n)
    exhausted = sum(e) < n
    ns = [r["N"] for r in rows[-cfg["k_range"]:]]
    n_lo, n_hi = min(ns), max(ns)
    extra = 0.0
    if n < n_lo or n > n_hi:
        p_edge, _e2 = _solve(pool, ages, a, b, tau, s_ref, min(max(n, n_lo), n_hi))
        extra = abs(point - p_edge)
    return point, {"model": "pool_walk", "a": round(a, 3), "b": b, "tau": tau,
                   "fit_draws": len(rows), "n_range": [n_lo, n_hi], "extrap": round(extra, 1),
                   "eligible_est": round(sum(e)), "pool_exhausted": exhausted}


# --------------------------------------------------------------------------------------
# walk-forward residuals for intervals (each previous draw predicted from its own past)
# --------------------------------------------------------------------------------------
def _wf_point(history, i, fps, T, cfg=CFG):
    key = ("wf", _fp_before(fps, i), history[i]["label"], T, history[i]["size"],
           tuple(sorted((k, v) for k, v in cfg.items() if not isinstance(v, tuple))))
    if key in _CACHE:
        return _CACHE[key]
    h = history[i]
    sub = history[:i]
    c = harness.make_ctx(sub, h["draw_date"])
    try:
        p, _info = predict_core(c, T, h["size"], cfg, fps=fps[:i] if i > 0 else [])
    except Exception:  # noqa: BLE001
        p = None
    _CACHE[key] = p
    return p


def _wquantile(vals, ws, q):
    pairs = sorted(zip(vals, ws))
    tot = sum(ws)
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= q * tot - 1e-12:
            return v
    return pairs[-1][0]


def predict(ctx, draw_type, n, cfg=None):
    cfg = cfg or CFG
    history = ctx["history"]
    date = ctx["target_date"]
    fps = _prefix_fps(history)
    point, info = predict_core(ctx, draw_type, n, cfg, fps=fps)
    # walk-forward residuals of the same type (each from data before that draw)
    idx = _same_type_idx(history, draw_type)[-cfg["k_resid"]:]
    res, ws = [], []
    for i in idx:
        p = _wf_point(history, i, fps, draw_type, cfg)
        if p is not None:
            res.append(history[i]["crs"] - p)
            ws.append(0.5 ** ((date - history[i]["draw_date"]).days / cfg["resid_half_life"]))
    if res and cfg["carry"]:
        point += cfg["carry"] * res[-1]
    cap = cfg["int_cap_tail"] if draw_type in TAIL_TYPES else cfg["int_cap"]
    if len(res) >= cfg["min_resid"]:
        q = _wquantile([abs(x) for x in res], ws, cfg["int_q"])
        hw = min(cap, max(cfg["int_floor"], cfg["int_mult"] * q))
    else:
        hw = cfg["int_default"]
    hw += cfg["extrap_mult"] * info.get("extrap", 0.0)
    info = dict(info, n_resid=len(res), half_width=round(hw, 1))
    lo_, hi_ = harness.MIN_SCORE, harness.MAX_SCORE
    point = min(hi_, max(lo_, point))
    return {"point": point, "low": max(lo_, point - hw), "high": min(hi_, point + hw),
            "info": info}
