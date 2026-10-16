"""Statistical model per draw type (stdlib only).

For each draw type T, regress the cut-off of past T draws on
    lnN    = ln(invitations)
    lngap  = ln(days since the previous T draw), clamped to [5, 180]
    cprev  = the previous T draw's cut-off (an AR(1) term: depletion / refill state)
with exponential recency weights (half-life H_slope) and ridge shrinkage of every slope
toward a prior (lambda_k = (sigma_T / tau_k)^2, sigma_T = robust scale of successive
same-type cut-off changes).  The intercept ("level") is re-estimated with a shorter
half-life (H_level) from the adjusted values y - b.x, i.e. partial pooling toward the
type's recent level.  The ln N slope is constrained to be <= 0, so the predicted
cut-off is non-increasing in N.

Sparse types:  >= MIN_ROWS usable rows -> own regression;  1..MIN_ROWS-1 same-type draws ->
the type's last level moved by the prior slopes ("anchored pooled slopes");  never drawn
before -> median of the last 3 category draws (with a wide interval).

General (all-program) draws: the regression above while all-program draws are recent; when
the type has been dormant for a long time (none since 2024-04) the prediction is blended
toward a pool-based estimate: top-N of the latest published distribution after removing
the draws since it (501-600 band spread exponentially, scale TAU_BAND, instead of uniform),
plus the mean residual of that estimate on past all-program draws.

Interval (~80%): walk-forward residuals of this same model on the type's previous draws
(each re-predicted from the history before it), recency-weighted 10%/90% quantiles,
shrunk toward a normal band from sigma_T for short histories, times INFLATE, plus an
extrapolation term for N outside the recently observed range (upper side widens for
small N, lower side for large N, so both bounds stay non-increasing in N).

Nov 3, 2026 CRS reform: a 4th regressor post = 1{draw date >= 2026-11-03} with a ridge
prior per type (mean = rule-change-only shift from the team's Nov-3 scenario simulation,
SD = its low..high spread; CEC +35+-12, French +4+-5, healthcare/trades +6+-6, senior
managers +60+-15, physicians +68+-12, PNP 0+-20, general +15+-15, others +5+-15).  All
historical rows have post = 0, so it changes nothing in any backtest; for a post-reform
target its uncertainty is added to the interval in quadrature, and once real post-reform
draws exist the regression updates the shift from them (ridge posterior).

Next-draw (harness latest_ctx) convention: when the target date is <= 4 days after the
type's last draw (never happens between two real same-type draws since 2019; the minimum
is 5 days), the draw is assumed to fall in the next cycle and the gap is set to the
median of the type's last 5 gaps.  For irregular types (healthcare, trades, senior
managers, physicians) call latest_ctx(target_date=expected next draw date) instead: the
gap term is strong (about +20 points per doubling of the gap for categories).

Results (harness, this file as is): tune_2025 MAJOR MAE 14.2 (baseline 27.9), coverage 80%
(width 48.7); holdout_2026 MAJOR MAE 12.6 (baselines 16.2 / 16.7), ALL 17.9 (21.4 / 21.7),
coverage 70% (width 36.3); general_2023_2024 MAE 8.5 (12.1 / 11.5), coverage 68%.
Exploration and tuning scripts: regression_work/ (tune.py, robust.py, latest2.py).
"""
import datetime as _dt
import math
import statistics

try:
    import harness
except ImportError:  # imported outside the harness CLI under -I
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import harness

NAME = "regression_per_type (weighted ridge: ln N, ln gap, prev cut-off; WF-residual intervals)"

PROGRAM_TYPES = ("pnp", "cec", "general", "fsw", "fst")
FEATS = ("lnN", "lngap", "cprev", "post")

# ---------------------------------------------------------------------------------------
# Hyperparameters (chosen on the 2025 window only; see regression_work/ for the grids)
# ---------------------------------------------------------------------------------------
BASE = {
    "prior": {"lnN": -20.0, "lngap": 20.0, "cprev": 0.5},
    "tau": {"lnN": 3.0, "lngap": 3.0, "cprev": 0.1},
    "H_slope": 730.0,       # days, recency half-life for the slopes
    "H_level": 90.0,        # days, half-life for the level (None -> same as H_slope)
    "max_age": 1100,        # days, ignore older same-type draws
    "min_rows": 4,
}
TYPE_CFG = {
    "pnp": {"prior": {"lnN": -40.0, "lngap": 25.0, "cprev": 0.0},
            "tau": {"lnN": 3.0, "lngap": 3.0, "cprev": 0.1},
            "H_slope": 730.0, "H_level": None},
    "cec": {"prior": {"lnN": -10.0, "lngap": 5.0, "cprev": 0.3},
            "tau": {"lnN": 1.0, "lngap": 1.0, "cprev": 0.1},
            "H_slope": 365.0, "H_level": 14.0},
}
# Nov 3, 2026 CRS reform ("Labour Market Integration" factor).  A 0/1 regressor `post`
# (draw on/after REFORM_DATE) with a ridge prior per type: prior mean = rule-change-only
# shift from the team's Nov-3 scenario simulation (model/nov3/out/nov3_scenarios.csv, mid
# case), prior SD = roughly its low..high spread.  Every historical row has post = 0, so it
# cannot affect any backtest; after real post-reform draws exist the data update the shift.
REFORM_DATE = _dt.date(2026, 11, 3)
REFORM_PRIOR = {"cec": (35.0, 12.0), "french": (4.0, 5.0), "healthcare": (6.0, 6.0),
                "trades": (6.0, 6.0), "senior_managers": (60.0, 15.0),
                "physicians": (68.0, 12.0), "pnp": (0.0, 20.0), "general": (15.0, 15.0)}
REFORM_DEFAULT = (5.0, 15.0)
GAP_LO, GAP_HI = 5, 180
SIGMA_FLOOR = 3.0
SIGMA_DEFAULT = 25.0        # sigma for a type with < 3 draws in the last 2 years
FIRST_DRAW_SIGMA = 45.0     # interval scale for a never-drawn type

# Interval settings
K_RESID = 12                # walk-forward residuals used (most recent same-type draws)
RESID_MAX_AGE = 1100
H_RESID = 180.0             # recency half-life for residual quantiles
Q_LO, Q_HI = 0.10, 0.90
Z80 = 1.2816
N0_SHRINK = 2.0             # pseudo-count pulling empirical quantiles toward +-Z80*sigma
INFLATE = 0.9               # calibrated on 2025 (raw quantiles over-covered: 84%)
BIAS_BETA = 0.0             # share of the recent median walk-forward residual added to the point
BIAS_MIN = 6
EXTRAP_REL = 0.3            # minimum relative uncertainty of the ln N slope when extrapolating
EXTRAP_K = 1.0              # widening per unit ln N outside the observed range (x |slope| SE)

# General (all-program) pool-based component
TAU_BAND = 20.0             # exponential scale of the 501-600 band (points)
GEN_TS_FULL = 30            # days since last all-program draw with full weight on regression
GEN_TS_DECAY = 90           # e-folding (days) of the regression weight beyond that

_CACHE = {}


# ---------------------------------------------------------------------------------------
# small linear algebra
# ---------------------------------------------------------------------------------------
def _solve(A, b):
    n = len(b)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-12:
            return None
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c and M[r][c] != 0.0:
                f = M[r][c] / M[c][c]
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] for i in range(n)]


def _inv(A):
    n = len(A)
    cols = []
    for j in range(n):
        e = [1.0 if i == j else 0.0 for i in range(n)]
        x = _solve(A, e)
        if x is None:
            return None
        cols.append(x)
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def _wquantile(values, weights, q):
    """Weighted quantile, midpoint rule with linear interpolation between sorted values."""
    pairs = sorted(zip(values, weights))
    tot = sum(w for _v, w in pairs)
    if not pairs or tot <= 0:
        return None
    mids, acc = [], 0.0
    for _v, w in pairs:
        mids.append((acc + w / 2.0) / tot)
        acc += w
    if q <= mids[0]:
        return pairs[0][0]
    if q >= mids[-1]:
        return pairs[-1][0]
    for i in range(1, len(pairs)):
        if mids[i] >= q:
            span = mids[i] - mids[i - 1]
            f = (q - mids[i - 1]) / span if span > 0 else 1.0
            return pairs[i - 1][0] + f * (pairs[i][0] - pairs[i - 1][0])
    return pairs[-1][0]


# ---------------------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------------------
def _cfg(t):
    c = dict(BASE)
    c.update(TYPE_CFG.get(t, {}))
    mu, sd = REFORM_PRIOR.get(t, REFORM_DEFAULT)
    c["prior"] = dict(c["prior"], post=mu)
    c["tau"] = dict(c["tau"], post=sd)
    return c


def _post(d):
    """1.0 for a draw dated on/after the Nov 3, 2026 CRS reform, else 0.0."""
    if isinstance(d, _dt.datetime):
        d = d.date()
    elif not isinstance(d, _dt.date):
        d = _dt.date.fromisoformat(str(d)[:10])
    return 1.0 if d >= REFORM_DATE else 0.0


def _clamp_gap(g):
    return math.log(min(max(g, GAP_LO), GAP_HI))


def _same(history, t):
    return [h for h in history if h["type"] == t]


def _target_gap(same, target_date):
    """Days since the type's last draw; <= 4 days means the 'next draw' query of latest_ctx:
    assume the next cycle (median of the last 5 same-type gaps)."""
    g = harness.days_between(same[-1]["draw_date"], target_date)
    if g <= 4:
        last = same[-6:]
        gaps = [harness.days_between(a["draw_date"], b["draw_date"])
                for a, b in zip(last, last[1:])]
        gaps = [x for x in gaps if x > 0]
        g = statistics.median(gaps) if gaps else 14
        g = max(g, 7)
    return g


def _rows(same, target_date, max_age):
    """Training rows (x, y, age_days, N) from same-type draws that have a predecessor."""
    rows = []
    for prev, cur in zip(same, same[1:]):
        age = harness.days_between(cur["draw_date"], target_date)
        if age > max_age or age < 0:
            continue
        gap = harness.days_between(prev["draw_date"], cur["draw_date"])
        x = (math.log(cur["size"]), _clamp_gap(gap), float(prev["crs"]), _post(cur["draw_date"]))
        rows.append((x, float(cur["crs"]), age, cur["size"]))
    return rows


def _sigma(same, target_date):
    ys = [h["crs"] for h in same if harness.days_between(h["draw_date"], target_date) <= 730]
    if len(ys) < 3:
        return None
    d = [abs(b - a) for a, b in zip(ys, ys[1:])][-12:]
    return max(SIGMA_FLOOR, 1.4826 * statistics.median(d) / math.sqrt(2.0))


# ---------------------------------------------------------------------------------------
# regression fit
# ---------------------------------------------------------------------------------------
def _fit(rows, cfg, sigma):
    k = len(FEATS)
    ws = [0.5 ** (r[2] / cfg["H_slope"]) for r in rows]
    W = sum(ws)
    xbar = [sum(w * r[0][i] for r, w in zip(rows, ws)) / W for i in range(k)]
    ybar = sum(w * r[1] for r, w in zip(rows, ws)) / W
    A = [[0.0] * k for _ in range(k)]
    bb = [0.0] * k
    for r, w in zip(rows, ws):
        xc = [r[0][i] - xbar[i] for i in range(k)]
        for i in range(k):
            bb[i] += w * xc[i] * (r[1] - ybar)
            for j in range(k):
                A[i][j] += w * xc[i] * xc[j]
    lam = [(sigma / cfg["tau"][f]) ** 2 for f in FEATS]
    for i, f in enumerate(FEATS):
        A[i][i] += lam[i]
        bb[i] += lam[i] * cfg["prior"][f]
    b = _solve(A, bb)
    if b is None:
        b = [cfg["prior"][f] for f in FEATS]
    b[0] = min(b[0], 0.0)                    # monotone in N
    # level: recency-weighted mean of adjusted values (shorter half-life)
    adj = [r[1] - sum(bi * xi for bi, xi in zip(b, r[0])) for r in rows]
    if cfg.get("H_level"):
        vs = [0.5 ** (r[2] / cfg["H_level"]) for r in rows]
    else:
        vs = ws
    level = sum(a * v for a, v in zip(adj, vs)) / sum(vs)
    # slope SE for lnN (posterior-ish): sigma^2 * inv(A)[0][0]
    Ainv = _inv(A)
    se_N = math.sqrt(max(0.0, sigma * sigma * Ainv[0][0])) if Ainv else abs(b[0]) * 0.3
    se_post = math.sqrt(max(0.0, sigma * sigma * Ainv[3][3])) if Ainv else cfg["tau"]["post"]
    # weighted N range of recent data (for extrapolation widening)
    lnNs = sorted((r[0][0], w) for r, w in zip(rows, ws))
    lo_lnN = _wquantile([v for v, _ in lnNs], [w for _, w in lnNs], 0.05)
    hi_lnN = _wquantile([v for v, _ in lnNs], [w for _, w in lnNs], 0.95)
    return {"b": b, "level": level, "se_N": se_N, "se_post": se_post,
            "lnN_lo": lo_lnN, "lnN_hi": hi_lnN}


def _category_fallback(history):
    cats = [h for h in history if h["type"] not in PROGRAM_TYPES]
    pool = (cats or history)[-3:]
    return float(statistics.median([h["crs"] for h in pool]))


# ---------------------------------------------------------------------------------------
# point prediction from history only (used for the target and for walk-forward residuals)
# ---------------------------------------------------------------------------------------
def _point_ts(history, t, target_date, n):
    """Regression point prediction. Returns (point, info dict)."""
    cfg = _cfg(t)
    same = _same(history, t)
    if not same:
        return _category_fallback(history), {"mode": "first_of_type", "sigma": FIRST_DRAW_SIGMA}
    sigma = _sigma(same, target_date) or SIGMA_DEFAULT
    gap = _target_gap(same, target_date)
    post_t = _post(target_date)
    xt = (math.log(n), _clamp_gap(gap), float(same[-1]["crs"]), post_t)
    rows = _rows(same, target_date, cfg["max_age"])
    if len(rows) >= cfg["min_rows"]:
        fit = _fit(rows, cfg, sigma)
        b = fit["b"]
        point = fit["level"] + sum(bi * xi for bi, xi in zip(b, xt))
        return point, {"mode": "regression", "b": [round(v, 3) for v in b],
                       "level": round(fit["level"], 2), "sigma": round(sigma, 2),
                       "gap": gap, "rows": len(rows), "se_N": fit["se_N"],
                       "lnN_lo": fit["lnN_lo"], "lnN_hi": fit["lnN_hi"],
                       "post": post_t, "se_post": fit["se_post"] * post_t}
    # sparse: anchored pooled slopes
    b = [cfg["prior"][f] for f in FEATS]
    if rows:
        recent = rows[-3:]
        level = statistics.fmean(r[1] - sum(bi * xi for bi, xi in zip(b, r[0])) for r in recent)
        point = level + sum(bi * xi for bi, xi in zip(b, xt))
        lnNs = [r[0][0] for r in rows]
    else:  # a single previous draw: move by the N slope (and the reform shift) only
        last = same[-1]
        point = (last["crs"] + b[0] * (math.log(n) - math.log(last["size"]))
                 + b[3] * (post_t - _post(last["draw_date"])))
        lnNs = [math.log(last["size"])]
    return point, {"mode": "sparse", "b": b, "sigma": round(sigma, 2), "gap": gap,
                   "rows": len(rows), "se_N": 0.5 * abs(b[0]) + 5.0,
                   "lnN_lo": min(lnNs), "lnN_hi": max(lnNs),
                   "post": post_t, "se_post": cfg["tau"]["post"] * post_t}


# ---- general: pool-based component -------------------------------------------------------
def _shaped_density(dist):
    d = harness.density(dist)
    c = dist.get("501-600", 0) or 0
    if c and TAU_BAND:
        ws = [math.exp(-(s - 501) / TAU_BAND) for s in range(501, 601)]
        W = sum(ws)
        for s, w in zip(range(501, 601), ws):
            d[s] = c * w / W
    return d


def _pool_cutoff(history, target_date, n, memo_key=None):
    """Top-n cut-off of the latest snapshot in `history` after removing the draws since it."""
    if memo_key is not None and memo_key in _CACHE:
        return _CACHE[memo_key]
    ctx = harness.make_ctx(history, target_date)
    snap = ctx["snapshot"]
    if snap is None:
        val = None
    else:
        d = _shaped_density(snap["dist"])
        for i in ctx["invited_since_snapshot"]:
            if i["type"] in ("pnp", "general", "fsw"):
                d = harness.remove_top(d, i["size"])
            else:
                d = harness.remove_above(d, i["crs"], i["size"])
        val = harness.score_for_count(d, n, integer=False)
    if memo_key is not None:
        _CACHE[memo_key] = val
    return val


def _prefix_fps(history):
    """Running fingerprints of history prefixes (cache keys that change if any earlier round changes)."""
    fps = [0]
    h = 0
    for r in history:
        h = hash((h, r["label"], r["crs"], r["size"], r["draw_date"], r["dist_as_on"],
                  r["dist_total"]))
        fps.append(h)
    return fps


def _point_general(history, target_date, n, fps):
    """Blend of regression and pool-based estimate for all-program draws."""
    ts, info = _point_ts(history, "general", target_date, n)
    same = _same(history, "general")
    if not same:
        return ts, info
    dormant = harness.days_between(same[-1]["draw_date"], target_date)
    w_ts = 1.0 if dormant <= GEN_TS_FULL else math.exp(-(dormant - GEN_TS_FULL) / GEN_TS_DECAY)
    if w_ts > 0.999:
        info["w_ts"] = 1.0
        return ts, info
    # bias of the pool estimate on past general draws (long memory)
    idx = {id(r): i for i, r in enumerate(history)}
    res, ws = [], []
    for r in same[-30:]:
        i = idx[id(r)]
        key = ("pool", fps[i], r["size"])
        x = _pool_cutoff(history[:i], r["draw_date"], r["size"], key)
        if x is None:
            continue
        res.append(r["crs"] - x)
        ws.append(0.5 ** (harness.days_between(r["draw_date"], same[-1]["draw_date"]) / 365.0))
    xnow = _pool_cutoff(history, target_date, n, ("pool", fps[len(history)], n, target_date))
    if xnow is None or not res:
        return ts, info
    bias = sum(a * w for a, w in zip(res, ws)) / sum(ws)
    pool_pt = xnow + bias
    snap = harness.find_snapshot(history)
    if _post(target_date) and snap is not None and snap["dist_as_on"] < REFORM_DATE:
        pool_pt += REFORM_PRIOR.get("general", REFORM_DEFAULT)[0]
    point = w_ts * ts + (1 - w_ts) * pool_pt
    info.update({"w_ts": round(w_ts, 3), "pool_raw": round(xnow, 1), "pool_bias": round(bias, 1),
                 "ts_point": round(ts, 1)})
    return point, info


def _point(history, t, target_date, n, fps):
    if t == "general":
        return _point_general(history, target_date, n, fps)
    return _point_ts(history, t, target_date, n)


# ---------------------------------------------------------------------------------------
# walk-forward residuals and interval
# ---------------------------------------------------------------------------------------
def _wf_residuals(history, t, target_date, fps, k=None):
    """Walk-forward residuals of this model for the last k draws of type t in `history`:
    each draw j is re-predicted from history[:j] only. Returns [(err, age_days, sigma_at_j)]."""
    same_idx = [i for i, h in enumerate(history) if h["type"] == t]
    out = []
    for i in same_idx[-(k or K_RESID):]:
        r = history[i]
        age = harness.days_between(r["draw_date"], target_date)
        if age > RESID_MAX_AGE:
            continue
        key = ("wf", t, fps[i], r["size"], r["draw_date"])
        if key in _CACHE:
            p, sig = _CACHE[key]
        else:
            p, inf = _point(history[:i], t, r["draw_date"], r["size"], fps)
            sig = FIRST_DRAW_SIGMA if inf.get("mode") == "first_of_type" else (
                inf.get("sigma") or SIGMA_DEFAULT)
            _CACHE[key] = (p, sig)
        out.append((r["crs"] - p, age, sig))
    return out


def _pooled_z(history, target_date, fps):
    """Recency-weighted 10%/90% quantiles of standardized walk-forward residuals pooled over
    every draw type seen in the last RESID_MAX_AGE days (the shrinkage target)."""
    key = ("pooledz", fps[len(history)], target_date)
    if key in _CACHE:
        return _CACHE[key]
    types = sorted({h["type"] for h in history
                    if harness.days_between(h["draw_date"], target_date) <= RESID_MAX_AGE})
    zs, ws = [], []
    for t in types:
        for e, age, sig in _wf_residuals(history, t, target_date, fps):
            zs.append(e / max(sig, SIGMA_FLOOR))
            ws.append(0.5 ** (age / H_RESID))
    if len(zs) >= 10:
        val = (_wquantile(zs, ws, Q_LO), _wquantile(zs, ws, Q_HI))
    else:
        val = (-Z80, Z80)
    _CACHE[key] = val
    return val


def _interval(point, info, resid, n, zq):
    sigma = info.get("sigma") or SIGMA_DEFAULT
    if info.get("mode") == "first_of_type":
        sigma = FIRST_DRAW_SIGMA
    q_lo_n, q_hi_n = zq[0] * sigma, zq[1] * sigma
    if resid:
        vals = [e for e, _a, _s in resid]
        ws = [0.5 ** (a / H_RESID) for _e, a, _s in resid]
        neff = sum(ws) ** 2 / sum(w * w for w in ws)
        ql = _wquantile(vals, ws, Q_LO)
        qh = _wquantile(vals, ws, Q_HI)
        lam = neff / (neff + N0_SHRINK)
        q_lo = lam * ql + (1 - lam) * q_lo_n
        q_hi = lam * qh + (1 - lam) * q_hi_n
    else:
        q_lo, q_hi = q_lo_n, q_hi_n
    q_lo = min(q_lo, -1.0) * INFLATE
    q_hi = max(q_hi, 1.0) * INFLATE
    sp = info.get("se_post") or 0.0
    if sp > 0:  # reform shift not yet learned from data: add its uncertainty in quadrature
        q_lo = -math.sqrt(q_lo * q_lo + (Z80 * sp) ** 2)
        q_hi = math.sqrt(q_hi * q_hi + (Z80 * sp) ** 2)
    # extrapolation widening (monotone-safe: upper for small N, lower for large N)
    lnN = math.log(n)
    b_n = abs((info.get("b") or [0.0])[0])
    se = max(info.get("se_N") or 0.0, EXTRAP_REL * b_n)   # prior-dominated slopes: >= 30% doubt
    lo_r, hi_r = info.get("lnN_lo"), info.get("lnN_hi")
    up = dn = 0.0
    if lo_r is not None and lnN < lo_r:
        up = EXTRAP_K * se * (lo_r - lnN)
    if hi_r is not None and lnN > hi_r:
        dn = EXTRAP_K * se * (lnN - hi_r)
    return point + q_lo - dn, point + q_hi + up


# ---------------------------------------------------------------------------------------
def predict(ctx, draw_type, n):
    history = ctx["history"]
    t = draw_type
    target_date = ctx["target_date"]
    fps = _prefix_fps(history)
    point, info = _point(history, t, target_date, n, fps)
    resid = _wf_residuals(history, t, target_date, fps)
    if BIAS_BETA and len(resid) >= BIAS_MIN:
        vals = [e for e, _a, _s in resid]
        ws = [0.5 ** (a / H_RESID) for _e, a, _s in resid]
        shift = BIAS_BETA * _wquantile(vals, ws, 0.5)
        point += shift
        resid = [(e - shift, a, s_) for e, a, s_ in resid]
        info["bias_shift"] = round(shift, 2)
    zq = _pooled_z(history, target_date, fps)
    lo, hi = _interval(point, info, resid, n, zq)
    lo = min(lo, point)
    hi = max(hi, point)
    info = {k: v for k, v in info.items() if k not in ("se_N", "lnN_lo", "lnN_hi", "se_post")}
    info["n_resid"] = len(resid)
    info["z80"] = [round(zq[0], 3), round(zq[1], 3)]
    return {"point": point, "low": lo, "high": hi, "info": info}
