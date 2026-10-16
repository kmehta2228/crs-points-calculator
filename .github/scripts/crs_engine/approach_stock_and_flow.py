"""Stock-and-flow CRS cut-off model ("every draw type is a queue").

Standard library only. Interface: predict(ctx, draw_type, n) -> {"point", "low", "high", "info"}.

IDEA
----
IRCC invites the top-n *eligible* candidates. Right after a draw of type k with cut-off c, no
type-k-eligible candidate with a score above c is left in the pool (and at exactly c only the
profiles created after the published tie-break time are left). So the eligible stock is a
queue: it is emptied from the top by every same-type draw and refilled by a daily inflow. The
next cut-off is the score at which the refilled stock reaches n.

Four mechanisms, all fed only by ctx (rounds before the target + the latest published snapshot):

1. CEC and category draws (french, healthcare, trades, stem, education, transport, ...):
   an eligible-stock queue per type on an integer score grid.
     * inflow per day per score: lam_k * g_k(s), with g_k a logistic profile (flat below s_mid,
       exponential tail of scale h above it); lam_k is learned online (multiplicative update
       from every same-type draw: log(n / modelled stock above the actual cut-off));
     * attrition: stock decays with time constant tau_k (expiry, withdrawal, nominations,
       score drift);
     * assimilation: after each real same-type draw the stock above its cut-off is set to zero
       and the cut-off score keeps the share of profiles created after the tie-break time
       (tie-break age / 365);
     * cross-type removals: an all-program draw clears everyone above its cut-off; any other
       category/CEC draw removes a share theta * n_j / pool(>= c_j) of the stock above c_j
       (pool counts from that round's published distribution).
   The "what if n" curve is the inverse of the modelled cumulative stock, so it is
   non-increasing in n by construction.
2. PNP draws: nominees get +600, IRCC invites (almost) all of them, so the cut-off is the
   LOWEST nominee score: an order statistic. c = level - beta * ln(n), level = trimmed mean of
   (c_k + beta ln n_k) over the last 20 PNP draws.
3. All-program ("general") draws: every candidate is eligible, so the published snapshot IS the
   eligible stock. An all-candidate queue is re-anchored to every snapshot (band totals) and
   supplies the within-band shape (the 501-600 band is very skewed and its top was emptied by
   the last all-program draw); invitations and inflow since the snapshot are applied.
4. Types never drawn before: pooled category prior with a wide range.

UNCERTAINTY
-----------
Walk-forward residuals of the same model on earlier draws of the same type, expressed as
log-stock errors r = ln(n / modelled stock above the actual cut-off), give an 80% band
[q10, q90]. The interval is (cut-off at n*exp(-q10), cut-off at n*exp(-q90)), which keeps low
and high non-increasing in n and makes them wide where the stock is thin. It is widened for n
outside the recently observed sizes, for stale types (long gap) and for types with few draws.

All hyperparameters were tuned on 2025 targets only (the all-program parameters on 2022
all-program draws, because there were none in 2025; its tail shape h=15 was also checked against
IRCC's EE Year-End Report 2024, Table 4: pool split 501-550 vs 551-600 on Dec 1, 2022/2023/2024).

NOTES (backtest with harness.py, this exact file, frozen before the holdout was looked at)
-----
* MAE MAJOR: tune_2025 13.0 (baselines 27.9 / 27.95), holdout_2026 12.1 (16.2 / 16.7);
  general_2023_2024 10.6 (12.1 / 11.5). 80%-interval coverage MAJOR 89% (2025), 81% (2026).
* Ablations on 2025 (MAJOR MAE 13.0): no memory below the last cut-off 18.8; frozen inflow level
  18.9; no attrition 15.0; PNP without the ln(n) term 15.2 (PNP 30.4 vs 25.1). The tie-break
  timestamp and the cross-type removals are neutral (+-0.05) and are kept only because they are
  the physically correct bookkeeping, not because they help.
* Forecast date: the cut-off depends on how long the queue has been refilling, so production
  code should pass the EXPECTED next draw date of the type as ctx['target_date']
  (harness.latest_ctx(target_date=...)). If target_date is < 5 days after the previous
  same-type draw, effective_date() assumes previous draw + median recent gap.
* All-program what-if with n below the estimated 601+ stock returns a placeholder in 601..800;
  physically the cut-off would sit among nominees (~900+). Irrelevant for n >= ~1,000.
* Scores change on 2026-11-03 (high-wage bonus +20/+40/+70, job-offer points 50-100): the
  queues' inflow profiles are in pre-change score units and need 1-3 post-change draws to
  re-learn the level; expect the first post-change CEC/category cut-offs to be under-predicted.
"""
import copy
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

NAME = "stock_and_flow (per-type eligible-stock queues + PNP order statistic + snapshot-anchored all-program queue)"

# --------------------------------------------------------------------------------------
# Hyperparameters (tuned on the 2025 window; general on 2022)
# --------------------------------------------------------------------------------------
QUEUE_PARAMS = {
    # s_mid/h: inflow profile; tau: attrition days; kappa: learning rate of the inflow level;
    # theta: cross-type removal intensity; tbA: tie-break age scale (days).
    "cec": dict(s_mid=505.0, h=10.0, tau=45.0, kappa=0.35, theta=1.0, tbA=365.0),
    "french": dict(s_mid=465.0, h=28.0, tau=200.0, kappa=0.8, theta=1.0, tbA=365.0),
    "healthcare": dict(s_mid=500.0, h=12.0, tau=400.0, kappa=0.8, theta=1.0, tbA=365.0),
}
GENERIC_QUEUE = dict(s_mid=None, s_mid_offset=-20.0, h=15.0, tau=365.0, kappa=0.8, theta=1.0,
                     tbA=365.0)
QUEUE_COMMON = dict(rclip=2.0, reset_days=400, start="2022-01-01", lo=0, hi=700)

PNP_PARAMS = dict(beta=27.5, K=20, trim=0.2, since="2022-01-01")
GENERAL_PARAMS = dict(s_mid=520.0, h=15.0, tau=60.0, k_snap=0.5, k_nu=0.5, nu0=40.0, tbA=365.0)

# Interval settings
INT_K = 12             # residuals used (most recent same-type draws)
INT_MIN_OWN = 6        # fewer own residuals -> blend with pooled category residuals
INT_Q = (0.10, 0.90)
INT_INFLATE = 1.0      # multiplies the log-residual quantiles
INT_PAD = 1.0          # score points added on both sides (integer cut-offs, ties)
EXT_W = 0.5            # widening per unit |ln(n / nearest observed size)|
STALE_DAYS = 180       # same-type gap after which the interval is widened
RARE_PAD = 12.0        # extra score points for types with < 8 own residuals (shape is a guess)
PROGRAM_TYPES = ("pnp", "cec", "general", "fsw", "fst")

GRID_LO, GRID_HI = QUEUE_COMMON["lo"], QUEUE_COMMON["hi"]
NG = GRID_HI - GRID_LO + 1


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def _date(x):
    if isinstance(x, _dt.datetime):
        return x.date()
    if isinstance(x, _dt.date):
        return x
    return _dt.date.fromisoformat(str(x)[:10])


def _quantile(v, q):
    """Linear-interpolated quantile of a non-empty list."""
    s = sorted(v)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    i = int(math.floor(pos))
    j = min(i + 1, len(s) - 1)
    return s[i] + (s[j] - s[i]) * (pos - i)


def _tie_rem(r, A):
    """Share of the cut-off score level still in the pool after a draw: profiles created after
    the tie-break time. Approximated as (draw time - tie-break time) / A, clipped to [0, 1]."""
    tb, dtm = r.get("tie_break"), r.get("draw_datetime")
    if tb is None or dtm is None:
        return 0.5
    age = (dtm - tb).total_seconds() / 86400.0
    return max(0.0, min(1.0, age / A))


def _logistic_profile(s_mid, h, lo, hi):
    g = []
    for s in range(lo, hi + 1):
        z = (s - s_mid) / h
        g.append(1.0 / (1.0 + math.exp(z)) if z < 60 else 0.0)
    return g


def _same_type(history, k):
    return [h for h in history if h["type"] == k]


def _median_gap(rounds, k=6):
    gaps = [(b["draw_date"] - a["draw_date"]).days for a, b in zip(rounds, rounds[1:])]
    gaps = [g for g in gaps if g > 0][-k:]
    return statistics.median(gaps) if gaps else 14


def effective_date(ctx, draw_type):
    """The date to forecast for. ctx['target_date'] unless it is fewer than 5 days after the
    previous same-type draw: no two same-type draws were ever that close (minimum gap in the
    data is 5 days), so such a date means "the next draw" (e.g. harness latest_ctx(), which
    uses tomorrow). Then use previous draw + median recent same-type gap."""
    td = _date(ctx["target_date"])
    same = _same_type(ctx["history"], draw_type)
    if not same:
        return td, False
    last = same[-1]["draw_date"]
    if (td - last).days < 5:
        return last + _dt.timedelta(days=int(round(_median_gap(same)))), True
    return td, False


# --------------------------------------------------------------------------------------
# 1. Eligible-stock queue (CEC and categories)
# --------------------------------------------------------------------------------------
class _Queue:
    def __init__(self, P):
        self.P = P
        self.g = _logistic_profile(P["s_mid"], P["h"], GRID_LO, GRID_HI)
        self.S = [0.0] * NG
        self.t = None
        self.lnlam = 0.0
        self.nobs = 0
        self.last_date = None
        self.last_crs = None
        self.resid = []   # (date, rlog, n, crs, pred)

    def advance(self, date):
        if self.t is None:
            self.t = date
            return
        d = (date - self.t).days
        if d <= 0:
            return
        tau = self.P["tau"]
        a = math.exp(-d / tau)
        b = tau * (1.0 - a)
        lam = math.exp(self.lnlam)
        S, g = self.S, self.g
        for i in range(NG):
            S[i] = S[i] * a + lam * g[i] * b
        self.t = date

    def cum(self, x):
        """Stock with score >= x (real x; uniform within each integer score)."""
        if x <= GRID_LO:
            return sum(self.S)
        if x > GRID_HI + 1:
            return 0.0
        i = int(math.floor(x))
        tot = sum(self.S[i - GRID_LO + 1:]) if i < GRID_HI else 0.0
        return tot + (self.S[i - GRID_LO] * ((i + 1) - x) if i <= GRID_HI else 0.0)

    def cutoff(self, n):
        """Largest integer score s with stock(>= s) >= n (GRID_LO if the stock is too small)."""
        acc = 0.0
        S = self.S
        for i in range(NG - 1, -1, -1):
            acc += S[i]
            if acc >= n:
                return GRID_LO + i
        return GRID_LO

    def init_steady(self, date, n, c):
        tau = self.P["tau"]
        G = sum(self.g[max(0, c - GRID_LO):]) if c <= GRID_HI else 1e-9
        lam = max(1e-9, n / max(1e-9, tau * G))
        self.lnlam = math.log(lam)
        self.S = [lam * x * tau for x in self.g]
        self.t = date

    def clear_above(self, c, rem):
        if c > GRID_HI:
            return
        c = max(GRID_LO, c)
        for i in range(c - GRID_LO + 1, NG):
            self.S[i] = 0.0
        self.S[c - GRID_LO] *= rem

    def thin_above(self, c, phi):
        if phi <= 0 or c > GRID_HI:
            return
        f = max(0.0, 1.0 - phi)
        for i in range(max(0, c - GRID_LO), NG):
            self.S[i] *= f


def _queue_params(k, same_hist):
    P = dict(QUEUE_COMMON)
    if k in QUEUE_PARAMS:
        P.update(QUEUE_PARAMS[k])
    else:
        P.update(GENERIC_QUEUE)
        cs = [h["crs"] for h in same_hist[-6:]]
        P["s_mid"] = (statistics.median(cs) if cs else 450.0) + P["s_mid_offset"]
    return P


def _simulate_queue(history, k, P):
    """Online pass over history for type k. Returns the queue (state at the last event) or None
    if the type was never drawn (since P['start'])."""
    start = _date(P["start"])
    q = None
    last_dist = None
    for r in history:
        if r.get("dist"):
            last_dist = r["dist"]
        d = r["draw_date"]
        if d < start:
            continue
        if r["type"] == k:
            gap = (d - q.last_date).days if (q is not None and q.last_date) else None
            if q is None or (gap is not None and gap > P["reset_days"]):
                old = q
                q = _Queue(P)
                q.init_steady(d, r["size"], r["crs"])
                if old is not None:      # keep the residual history across a reset
                    q.resid = old.resid
                q.clear_above(r["crs"], _tie_rem(r, P["tbA"]))
                q.last_date, q.last_crs, q.nobs = d, r["crs"], 1
                continue
            q.advance(d)
            pred = q.cutoff(r["size"])
            ca = q.cum(r["crs"] + 0.5)
            rlog = math.log(r["size"] / max(ca, 1e-6))
            rlog = max(-P["rclip"], min(P["rclip"], rlog))
            q.resid.append((d, rlog, r["size"], r["crs"], pred))
            kap = max(P["kappa"], 1.0 / (q.nobs + 1))
            q.lnlam += kap * rlog
            q.clear_above(r["crs"], _tie_rem(r, P["tbA"]))
            q.last_date, q.last_crs = d, r["crs"]
            q.nobs += 1
        elif q is not None:
            q.advance(d)
            if r["type"] == "general":
                q.clear_above(r["crs"], _tie_rem(r, P["tbA"]))
            elif r["type"] == "pnp":
                pass  # nominees sit at 601+, outside the sub-600 eligible stock
            else:
                dist = r.get("dist") or last_dist
                pa = harness.cumulative_above(dist, r["crs"]) if dist else 0.0
                if pa > 0:
                    q.thin_above(r["crs"], min(1.0, P["theta"] * r["size"] / pa))
    return q


def _category_residual_pool(history, exclude=None, limit=60):
    """Pooled log-stock residuals of all non-program category types (for types with few draws)."""
    out = []
    for k in sorted({h["type"] for h in history}):
        if k in PROGRAM_TYPES or k == exclude:
            continue
        same = _same_type(history, k)
        if len(same) < 2:
            continue
        q = _simulate_queue(history, k, _queue_params(k, same))
        if q is not None:
            out.extend(q.resid)
    out.sort(key=lambda x: x[0])
    return [x[1] for x in out[-limit:]]


def _log_band(own, pooled, k_own_min=INT_MIN_OWN):
    """(q_lo, q_hi) of log-stock residuals; blends in pooled residuals when own ones are few."""
    own = own[-INT_K:]
    if len(own) >= k_own_min:
        v = own
        extra = 0.0
    else:
        v = own + pooled[-(INT_K - len(own)):] if pooled else own
        extra = 0.25 * (k_own_min - len(own)) / k_own_min
    if len(v) < 3:
        return -1.0 - extra, 1.0 + extra
    lo = _quantile(v, INT_Q[0]) * INT_INFLATE - extra
    hi = _quantile(v, INT_Q[1]) * INT_INFLATE + extra
    return min(lo, -0.05), max(hi, 0.05)


def _ext_width(n, sizes):
    if not sizes:
        return 0.5
    lo, hi = min(sizes), max(sizes)
    if n < lo:
        return EXT_W * math.log(lo / n)
    if n > hi:
        return EXT_W * math.log(n / hi)
    return 0.0


def _predict_queue(ctx, k, n, date):
    hist = ctx["history"]
    same = _same_type(hist, k)
    P = _queue_params(k, same)
    q = _simulate_queue(hist, k, P)
    if q is None:
        return None
    gap = (date - q.last_date).days if q.last_date else 0
    q.advance(date)
    point = q.cutoff(n)
    own = [x[1] for x in q.resid]
    pooled = _category_residual_pool(hist, exclude=k) if len(own) < INT_MIN_OWN else []
    qlo, qhi = _log_band(own, pooled)
    sizes = [h["size"] for h in same[-10:]]
    w = _ext_width(n, sizes)
    if gap > STALE_DAYS:
        w += 1.0 * min(2.0, (gap - STALE_DAYS) / 365.0)
    # true stock = model stock * exp(r)  =>  true cut-off(n) = model cut-off(n * exp(-r))
    high = q.cutoff(n * math.exp(-(qhi + w)))
    low = q.cutoff(n * math.exp(-(qlo - w)))
    pad = INT_PAD + RARE_PAD * max(0, 8 - len(own)) / 8.0
    low = min(low, point) - pad
    high = max(high, point) + pad
    return {"point": float(point), "low": float(low), "high": float(high),
            "info": {"model": "queue", "type": k, "date": date.isoformat(), "gap_days": gap,
                     "lam_per_day_above_ref": round(math.exp(q.lnlam), 2),
                     "stock_above_last_cutoff": round(q.cum(q.last_crs + 0.5), 1),
                     "last_cutoff": q.last_crs, "n_resid": len(own),
                     "log_band": [round(qlo, 3), round(qhi, 3)], "ext_w": round(w, 3)}}


# --------------------------------------------------------------------------------------
# 2. PNP: lowest invited nominee (order statistic)
# --------------------------------------------------------------------------------------
def _pnp_fit(pnp_hist, P):
    since = _date(P["since"])
    past = [r for r in pnp_hist if r["draw_date"] >= since][-P["K"]:]
    if not past:
        return None
    v = sorted(r["crs"] + P["beta"] * math.log(r["size"]) for r in past)
    cut = int(len(v) * P["trim"])
    core = v[cut:len(v) - cut] if len(v) - 2 * cut >= 1 else v
    return statistics.fmean(core)


def _predict_pnp(ctx, n):
    P = PNP_PARAMS
    pnp = _same_type(ctx["history"], "pnp")
    level = _pnp_fit(pnp, P)
    if level is None:
        return None
    point = level - P["beta"] * math.log(n)
    # walk-forward residuals (actual - predicted) of the same rule on earlier PNP draws
    res = []
    for i in range(max(1, len(pnp) - 30), len(pnp)):
        lv = _pnp_fit(pnp[:i], P)
        if lv is not None:
            res.append(pnp[i]["crs"] - (lv - P["beta"] * math.log(pnp[i]["size"])))
    if len(res) >= 8:
        dlo, dhi = _quantile(res, INT_Q[0]), _quantile(res, INT_Q[1])
    else:
        dlo, dhi = -45.0, 45.0
    sizes = [r["size"] for r in pnp[-20:]]
    w = _ext_width(n, sizes) * P["beta"]
    low = point + min(dlo, -5.0) - w - INT_PAD
    high = point + max(dhi, 5.0) + w + INT_PAD
    return {"point": point, "low": low, "high": high,
            "info": {"model": "pnp_order_statistic", "level": round(level, 2),
                     "beta": P["beta"], "n_resid": len(res),
                     "resid_q10_q90": [round(dlo, 1), round(dhi, 1)]}}


# --------------------------------------------------------------------------------------
# 3. All-program draws: snapshot-anchored all-candidate queue
# --------------------------------------------------------------------------------------
G_LO, G_HI = 0, 600
G_NG = G_HI - G_LO + 1
_SUB_BANDS = [(key, lo, hi) for key, lo, hi, _f in harness.BANDS if hi <= 600]


class _AllPool:
    def __init__(self, P):
        self.P = P
        self.g = _logistic_profile(P["s_mid"], P["h"], G_LO, G_HI)
        self.Q = [0.0] * G_NG
        self.A = 0.0          # candidates at 601+
        self.t = None
        self.lam = 0.0
        self.nu = P["nu0"]
        self.last_snap = None
        self.resid = []       # (date, rlog, n, crs, pred)
        self.last_general = None

    def advance(self, date):
        if self.t is None or date <= self.t:
            self.t = date if self.t is None else max(self.t, date)
            return
        d = (date - self.t).days
        tau = self.P["tau"]
        a = math.exp(-d / tau)
        b = tau * (1.0 - a)
        for i in range(G_NG):
            self.Q[i] = self.Q[i] * a + self.lam * self.g[i] * b
        self.A += self.nu * d
        self.t = date

    def _band_total(self, lo, hi):
        return sum(self.Q[lo - G_LO:hi - G_LO + 1])

    def _set_band(self, lo, hi, total):
        w = self.g[lo - G_LO:hi - G_LO + 1]
        sw = sum(w) or 1.0
        for j, s in enumerate(range(lo, hi + 1)):
            self.Q[s - G_LO] = total * w[j] / sw

    def init(self, date, dist):
        for key, lo, hi in _SUB_BANDS:
            self._set_band(lo, hi, dist[key])
        self.A = float(dist["601-1200"])
        self.lam = dist["501-600"] / (self.P["tau"] * (sum(self.g[501 - G_LO:]) or 1.0))
        self.t = date
        self.last_snap = date

    def assimilate(self, date, dist):
        self.advance(date)
        top_m = self._band_total(491, 600)
        top_o = dist["501-600"] + dist["491-500"]
        if top_m > 0 and top_o > 0:
            r = max(-1.0, min(1.0, math.log(top_o / top_m)))
            self.lam *= math.exp(self.P["k_snap"] * r)
        if self.last_snap is not None:
            days = max(1, (date - self.last_snap).days)
            self.nu = max(0.0, self.nu + self.P["k_nu"] * (dist["601-1200"] - self.A) / days)
        self.A = float(dist["601-1200"])
        for key, lo, hi in _SUB_BANDS:
            m = self._band_total(lo, hi)
            if m > 1e-9:
                f = dist[key] / m
                for s in range(lo, hi + 1):
                    self.Q[s - G_LO] *= f
            else:
                self._set_band(lo, hi, dist[key])
        self.last_snap = date

    def cum(self, x):
        if x > G_HI + 1:
            return 0.0
        if x > G_HI:
            return self.A
        i = max(G_LO, int(math.floor(x)))
        return self.A + sum(self.Q[i - G_LO + 1:]) + self.Q[i - G_LO] * ((i + 1) - x)

    def cutoff(self, n):
        if n <= self.A:   # the cut-off would sit inside the 601+ band (never happened since 2017)
            return 601 + int(round(200 * (1 - n / max(self.A, 1.0))))
        acc = self.A
        for i in range(G_NG - 1, -1, -1):
            acc += self.Q[i]
            if acc >= n:
                return G_LO + i
        return G_LO

    def clear_general(self, n, c, rem):
        self.A -= min(self.A, n)
        if c <= G_HI:
            for i in range(max(0, c - G_LO + 1), G_NG):
                self.Q[i] = 0.0
            if c >= G_LO:
                self.Q[c - G_LO] *= rem

    def remove_prop(self, n, c):
        if c > G_HI:
            self.A = max(0.0, self.A - n)
            return
        i0 = max(0, c - G_LO)
        avail = sum(self.Q[i0:])
        if avail > 0:
            f = min(1.0, n / avail)
            for i in range(i0, G_NG):
                self.Q[i] *= (1.0 - f)


def _simulate_allpool(history, P, target_types=("general",)):
    """Online pass. Records walk-forward residuals for every round of target_types (predicted
    BEFORE that round's own snapshot is assimilated)."""
    st = None
    seen = set()
    for r in history:
        d = r["draw_date"]
        if st is not None and r["type"] in target_types:
            tmp = copy.deepcopy(st)
            tmp.resid = []
            tmp.advance(d)
            pred = tmp.cutoff(r["size"])
            ca = tmp.cum(r["crs"] + 0.5)
            rl = max(-2.0, min(2.0, math.log(r["size"] / max(ca, 1e-6))))
            st.resid.append((d, rl, r["size"], r["crs"], pred))
        if r.get("dist") and r.get("dist_as_on"):
            key = tuple(r["dist"][kk] for kk in harness.BAND_KEYS)
            if key not in seen:
                seen.add(key)
                A = r["dist_as_on"]
                if st is None:
                    st = _AllPool(P)
                    st.init(A, r["dist"])
                elif A >= st.t:
                    st.assimilate(A, r["dist"])
        if st is None:
            continue
        st.advance(d)
        if r["type"] == "general":
            st.clear_general(r["size"], r["crs"], _tie_rem(r, P["tbA"]))
            st.last_general = r
        elif r["type"] == "pnp":
            st.A = max(0.0, st.A - r["size"])
        else:
            st.remove_prop(r["size"], r["crs"])
    return st


def _predict_general(ctx, n, date, draw_type="general"):
    P = GENERAL_PARAMS
    st = _simulate_allpool(ctx["history"], P, target_types=(draw_type,))
    if st is None:
        return None
    st.advance(date)
    point = st.cutoff(n)
    own = [x[1] for x in st.resid]
    qlo, qhi = _log_band(own, [])
    same = _same_type(ctx["history"], draw_type)
    sizes = [h["size"] for h in same[-10:]]
    w = _ext_width(n, sizes)
    gap = (date - same[-1]["draw_date"]).days if same else 9999
    if gap > STALE_DAYS:
        w += 0.5 * min(3.0, (gap - STALE_DAYS) / 365.0)
    high = st.cutoff(n * math.exp(-(qhi + w)))
    low = st.cutoff(n * math.exp(-(qlo - w)))
    low = min(low, point) - INT_PAD
    high = max(high, point) + INT_PAD
    return {"point": float(point), "low": float(low), "high": float(high),
            "info": {"model": "allpool_snapshot_queue", "date": date.isoformat(),
                     "a601": round(st.A, 1), "nu_per_day": round(st.nu, 2),
                     "snapshot_as_on": str(st.last_snap), "n_resid": len(own),
                     "gap_days": gap, "log_band": [round(qlo, 3), round(qhi, 3)],
                     "ext_w": round(w, 3)}}


# --------------------------------------------------------------------------------------
# 4. Fallback for a type never drawn before
# --------------------------------------------------------------------------------------
def _predict_fallback(ctx, k, n, date):
    cats = [h for h in ctx["history"] if h["type"] not in PROGRAM_TYPES
            and (date - h["draw_date"]).days <= 730]
    if not cats:
        cats = [h for h in ctx["history"] if h["type"] not in ("pnp",)][-10:]
    cs = [h["crs"] for h in cats] or [450]
    point = statistics.median(cs)
    low = min(_quantile(cs, 0.05), point - 40) - 30
    high = max(_quantile(cs, 0.95), point + 40) + 10
    # a mild size effect so the curve still answers "what if n" (non-increasing in n)
    adj = 10.0 * math.log(max(n, 1) / 2000.0)
    return {"point": point - adj, "low": low - adj, "high": high - adj,
            "info": {"model": "fallback_category_prior", "n_cats": len(cs)}}


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------
def predict(ctx, draw_type, n):
    n = max(1.0, float(n))
    date, shifted = effective_date(ctx, draw_type)
    out = None
    if draw_type == "pnp":
        out = _predict_pnp(ctx, n)
    elif draw_type in ("general", "fsw"):
        out = _predict_general(ctx, n, date, draw_type="general")
    else:
        out = _predict_queue(ctx, draw_type, n, date)
    if out is None:
        out = _predict_fallback(ctx, draw_type, n, date)
    out["point"] = max(0.0, min(1200.0, out["point"]))
    out["low"] = max(0.0, min(out["low"], out["point"]))
    out["high"] = min(1200.0, max(out["high"], out["point"]))
    info = out.setdefault("info", {})
    info["forecast_date"] = date.isoformat()
    if shifted:
        info["date_shifted_to_next_regular_draw"] = True
    return out
