"""Ticket 17: inside the mixed loop, who enters `lm`, what it costs, what it rescues.

Layered on ticket 04's instrument (`count_mixed.py`), which already attributes
every NJL internal solve to a site of the mixed engine. This adds what happens
INSIDE each of those solves: `eos.njl.thermodynamics.thermo_from_mu` closes
the internal system through `solve_system(..., tol=1e-13)` with no Jacobian,
i.e. `hybr` from the seed, `lm` from the same seed if that missed the gate,
then `newton_polish` from the better of the two if both missed.

Nothing in `eos/` is edited: `root`, `newton_polish` and `internal_residual`
are looked up through module globals at call time, so they are patched here.

PRICED IN COUNTS. Every `internal_residual` evaluation is attributed to the
rung it was spent in (hybr / lm / polish), including the gate check
`solve_system` makes after each rung. The per-eval cost is set by the pattern
(the BdG quadrature), not by the rung, so evaluation shares are cost shares.

THE COUNTERFACTUAL IS EXACT, not a model of one. The mixed engine hands every
`thermo` call in one solve the SAME memoized seed (`MixedCtx.phase_seed`), so
there is no path dependence inside a call: with methods=('hybr',) the call
runs the identical `hybr` and then polishes from ITS iterate. So for every
call that entered `lm`, the polish is re-run from hybr's (x, err) with the
real `newton_polish`, and whether the bounded call would have converged --
and onto which layout and pressure -- is read off directly. Its evaluations
go to a separate bucket and are excluded from every total.

A call's output reaches the mixed engine only through the adapter's rule:
skip if not ok, keep the max-P candidate as the block, keep the vector of
every candidate that held its layout as the seed. So a rescue that changes
nothing the adapter returns is recorded as such.
"""
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import count_mixed as CM                                        # noqa: E402
import eos.general.solve as G                                   # noqa: E402
import eos.njl.thermodynamics as NT                             # noqa: E402
from eos.general.pairing import realised_pattern                # noqa: E402

#: one dict per thermo_from_mu call, in call order
CALLS = []
_cur = []          # the open thermo_from_mu record (stack; never deeper than 1)
_call_id = [0]     # adapter-level `thermo` call counter


def _phase(name):
    if _cur:
        _cur[-1]["phase"] = name


def install():
    CM.install()                  # ticket 04's counters, incl. its tfm wrapper
    real_root, real_polish = G.root, G.newton_polish
    real_ir, real_tfm = NT.internal_residual, NT.thermo_from_mu

    def ir(*a, **kw):
        if _cur:
            rec = _cur[-1]
            rec["evals"][rec["phase"]] += 1
        return real_ir(*a, **kw)

    def root(fun, x0, *a, method="hybr", **kw):
        if not _cur:
            return real_root(fun, x0, *a, method=method, **kw)
        rec = _cur[-1]
        _phase(method)
        t0 = time.perf_counter()
        sol = real_root(fun, x0, *a, method=method, **kw)
        rec["rungs"].append({"method": method, "nfev": int(sol.nfev),
                             "s": time.perf_counter() - t0, "x": sol.x})
        if method == "hybr":
            rec["fun"] = fun
        return sol

    def polish(residual, x, scales_at, err, *a, **kw):
        if not _cur:
            return real_polish(residual, x, scales_at, err, *a, **kw)
        rec = _cur[-1]
        _phase("polish")
        t0 = time.perf_counter()
        out = real_polish(residual, x, scales_at, err, *a, **kw)
        rec["polish"] = {"err_in": err, "err_out": out[1],
                         "s": time.perf_counter() - t0}
        return out

    def tfm(par, mu_B, mu_C=0.0, mu_S=0.0, T=0.0, pattern="unpaired",
            x0=None, vac=None, return_state=False, backend="reference",
            **extra):
        site, depth = CM.site_and_depth()
        rec = {"site": site, "depth": depth, "call": _call_id[0],
               "pattern": pattern, "cold": x0 is None, "phase": "pre",
               "evals": defaultdict(int), "rungs": [], "polish": None,
               "mu": (mu_B, mu_C, mu_S, T)}
        _cur.append(rec)
        t0 = time.perf_counter()
        try:
            out = real_tfm(par, mu_B, mu_C, mu_S, T, pattern=pattern, x0=x0,
                           vac=vac, return_state=return_state,
                           backend=backend, **extra)
        finally:
            rec["s"] = time.perf_counter() - t0
            _cur.pop()
        st, ok, err = out[0], out[1], out[2]
        rec.update(ok=bool(ok), err=float(err), P=float(st.P),
                   realised=realised_pattern(st.Delta))
        if any(r["method"] == "lm" for r in rec["rungs"]):
            counterfactual(rec, par, backend, vac)
        rec.pop("fun", None)
        for r in rec["rungs"]:
            r.pop("x", None)
        CALLS.append(rec)
        return out

    G.root, G.newton_polish = root, polish
    NT.internal_residual, NT.thermo_from_mu = ir, tfm
    # `count_mixed`'s own tfm wrapper was bound into NT before this one, so
    # its counters still run: ours wraps theirs.


def counterfactual(rec, par, backend, vac):
    """What the same call returns under methods=('hybr',): polish from hybr."""
    fun = rec["fun"]
    x_h = rec["rungs"][0]["x"]
    ones = lambda x: [1.0] * len(x)                             # noqa: E731
    err_h = G.scaled_residual_max(fun(x_h), ones(x_h))
    x_cf, err_cf = G.newton_polish(fun, x_h, ones, err_h)       # real one:
    # G.newton_polish is our wrapper, but _cur is empty here, so it passes
    # straight through and its evaluations land in no bucket.
    mu_B, mu_C, mu_S, T = rec["mu"]
    M, Delta, mu_3, mu_8, Sigma_V = NT._unpack_internal(x_cf, par,
                                                        rec["pattern"])
    st = NT.state_at(par, M, Delta, Sigma_V, mu_B, mu_C, mu_S, mu_3, mu_8, T,
                     vac=vac, pattern=rec["pattern"], backend=backend)
    rec["cf"] = {"err_hybr": float(err_h), "err": float(err_cf),
                 "ok": bool(err_cf <= G.RESIDUAL_TOL),
                 "realised": realised_pattern(st.Delta), "P": float(st.P)}


def wrap_adapter(phase):
    """Number each quark `thermo` call, so candidates group into calls."""
    import dataclasses
    real = phase.thermo

    def numbered(*a, **kw):
        _call_id[0] += 1
        return real(*a, **kw)

    # `seed()` calls the adapter's INNER thermo, not this one, so it needs its
    # own number or its candidates are grouped with the previous call's.
    real_seed = phase.seed

    def numbered_seed(*a, **kw):
        _call_id[0] += 1
        return real_seed(*a, **kw)
    return dataclasses.replace(phase, thermo=numbered, seed=numbered_seed)


_cm_make_phases = CM.make_phases


def make_phases():
    p_H, p_Q = _cm_make_phases()
    return p_H, wrap_adapter(p_Q)


# ---------------------------------------------------------------------------
# reading it
# ---------------------------------------------------------------------------

def outcome(rec):
    methods = [r["method"] for r in rec["rungs"]]
    if "lm" not in methods:
        return "hybr-ok" if rec["polish"] is None else (
            "polish-ok" if rec["ok"] else "fail")
    if rec["polish"] is None:
        return "lm-rescue"
    return "lm+polish-ok" if rec["ok"] else "fail(lm)"


def winners():
    """call id -> (pattern, P) of the candidate the adapter returned."""
    best = {}
    for rec in CALLS:
        if not rec["ok"]:
            continue
        b = best.get(rec["call"])
        if b is None or rec["P"] > b[1]:
            best[rec["call"]] = (rec["pattern"], rec["P"])
    return best


def winners_cf():
    """The same, had every lm-entered call returned its counterfactual."""
    best = {}
    for rec in CALLS:
        ok, P = rec["ok"], rec["P"]
        if "cf" in rec:
            ok, P = rec["cf"]["ok"], rec["cf"]["P"]
        if not ok:
            continue
        b = best.get(rec["call"])
        if b is None or P > b[1]:
            best[rec["call"]] = (rec["pattern"], P)
    return best


def report(title):
    print(f"\n##### ticket 17: {title} -- {len(CALLS)} thermo_from_mu calls, "
          f"{len({r['call'] for r in CALLS})} quark thermo calls", flush=True)
    tot = defaultdict(int)
    for r in CALLS:
        for k, v in r["evals"].items():
            tot[k] += v
    all_ev = sum(tot.values())
    print("  internal_residual evaluations by rung: " + ", ".join(
        f"{k} {v} ({v / max(all_ev, 1):.1%})" for k, v in sorted(tot.items()))
        + f"   total {all_ev}")
    sec = defaultdict(float)
    for r in CALLS:
        for g in r["rungs"]:
            sec[g["method"]] += g["s"]
        if r["polish"]:
            sec["polish"] += r["polish"]["s"]
        sec["all"] += r["s"]
    print("  seconds (NOT quotable under load): " + ", ".join(
        f"{k} {v:.1f}" for k, v in sorted(sec.items())))

    by = defaultdict(lambda: defaultdict(int))
    for r in CALLS:
        by[(r["site"], r["pattern"])][outcome(r)] += 1
        by[(r["site"], r["pattern"])]["lm_evals"] += r["evals"].get("lm", 0)
        by[(r["site"], r["pattern"])]["evals"] += sum(r["evals"].values())
    keys = ["hybr-ok", "lm-rescue", "lm+polish-ok", "fail(lm)", "polish-ok",
            "fail"]
    print(f"  {'site':16s} {'pattern':9s} " + " ".join(f"{k:>12s}" for k in keys)
          + f" {'lm evals':>9s} {'of evals':>9s}")
    for (site, pat), d in sorted(by.items()):
        print(f"  {site:16s} {pat:9s} " + " ".join(f"{d[k]:12d}" for k in keys)
              + f" {d['lm_evals']:9d} {d['lm_evals'] / max(d['evals'], 1):9.1%}")

    entered = [r for r in CALLS if "cf" in r]
    rescued = [r for r in entered if r["ok"]]
    print(f"  entered lm: {len(entered)}   converged after lm "
          f"(lm or polish): {len(rescued)}")
    lost = [r for r in rescued if not r["cf"]["ok"]]
    kept = [r for r in rescued if r["cf"]["ok"]]
    gained = [r for r in entered if not r["ok"] and r["cf"]["ok"]]
    print(f"    ...of which the bounded call ALSO converges: {len(kept)}; "
          f"LOST under the bound: {len(lost)}; failing now but converging "
          f"bounded: {len(gained)}")
    for r in lost:
        print(f"    LOST  {r['site']:16s} d{r['depth']} {r['pattern']:9s} "
              f"call {r['call']:5d} -> realised {r['realised']:9s} "
              f"P {r['P']:.6f}  hybr err {r['cf']['err_hybr']:.2e} "
              f"polish-from-hybr {r['cf']['err']:.2e}  mu {r['mu'][:3]}")
    for r in kept:
        dP = abs(r["cf"]["P"] - r["P"]) / max(abs(r["P"]), 1e-30)
        print(f"    kept  {r['site']:16s} {r['pattern']:9s} realised "
              f"{r['realised']}/{r['cf']['realised']}  |dP|/P {dP:.1e}")
    # Does any adapter output move? The block is the max-P ok candidate; the
    # seed dict is every ok candidate that held its layout.
    w, wcf = winners(), winners_cf()
    moved = []
    for cid in sorted(set(w) | set(wcf)):
        a, b = w.get(cid), wcf.get(cid)
        if a is None or b is None or a[0] != b[0] or \
                abs(a[1] - b[1]) > 1e-8 * max(abs(a[1]), 1.0):
            moved.append((cid, a, b))
    print(f"  quark thermo calls whose RETURNED BLOCK would change under the "
          f"bound: {len(moved)}")
    for cid, a, b in moved[:20]:
        print(f"    call {cid}: {a} -> {b}")
    seeds_moved = [r for r in lost if r["realised"] == r["pattern"]]
    print(f"  lm-rescued candidates that held their layout (would be carried "
          f"as a seed by `seed()`): {len(seeds_moved)}")

    # Is `lm` where the cost is? Per site and retry depth.
    d_ev = defaultdict(lambda: [0, 0])
    for r in CALLS:
        key = (r["site"], min(r["depth"], 1))
        d_ev[key][0] += r["evals"].get("lm", 0)
        d_ev[key][1] += sum(r["evals"].values())
    print("  lm share of evals by (site, retry depth>=1): " + "; ".join(
        f"{s}/d{d}: {a}/{b} = {a / max(b, 1):.1%}"
        for (s, d), (a, b) in sorted(d_ev.items())))


RUNS = {
    "onset": CM.run_onset, "warm": CM.run_warm_sweep, "cold": CM.run_cold_point,
    "locator": CM.run_locator,
}

if __name__ == "__main__":
    install()
    CM.make_phases = make_phases          # ticket 04's runs build ours
    print(f"ticket 17: lm inside the mixed loop\n{CM.stack_line()}")
    print(f"{CM.BENCH_SET}, {CM.BENCH_MODE}, T = {CM.BENCH_T}, eta = "
          f"{CM.BENCH_ETA}, patterns {CM.BENCH_PATTERNS}, backend "
          f"{CM.BENCH_BACKEND!r}", flush=True)
    CM.warmup()
    for name in sys.argv[1:]:
        CALLS.clear()
        _call_id[0] = 0
        RUNS[name]()
        report(name)
