"""PROTOTYPE (wayfinder ticket 13) -- throwaway. Not production code.

WHAT IT ASKS. `eos.njl.solver`'s `attempt` is a four-rung ladder and ticket 05
priced the rungs (B 54.8%, A 22.6%, D 21.0%, C 0.0%). What 05 could NOT say is
which rung rescues the ONE CFL candidate that holds its layout at the onset,
nor what that rescue SPENDS -- and both are what decides whether rung B can be
bounded the way `LM_MAX_EVALUATIONS` already bounds the LM rung.

THE SEAM. `eos/njl/solver.py` calls `solve_system` at exactly three sites, all
inside `attempt`:

    790  x, err, ok = solve_system(rows, seed, ..., jac=jac)     rungs A + B
    794  reinflate rescue, from `reinflated(x)`, jac=jac         rung C
    799  differenced rescue, from the same seed, NO jac          rung D

so under `backend="fast"` (where `jac` is never None) `jac is None` IS rung D,
and C is distinguished from A+B by being the second jac-carrying call inside
one `solve_pattern`. Inside `solve_system`, `newton_solve` is rung A and each
`root` call is one MINPACK rung of B. All four are module-global names, so the
whole census is monkeypatching and no eos source is touched.

    python3 .scratch/njl-speed/proto13_rungs.py
"""
import json
import os
import sys
import time

import numpy as np

from eos import njl
import eos.general.solve as gsolve
import eos.njl.solver as nsolver

BENCH_SET = "rg_njl1"
BENCH_MODE = "beta_eq_neutrinoless"
BENCH_T = 0.0
BENCH_SPECIES = njl.SpeciesFlags(csc=True)
BENCH_BACKEND = "fast"
BENCH_NB = np.linspace(0.5, 1.55, 200)

CALLS = []          # one entry per solve_pattern call
cur = {"rec": None, "n_jac_calls": 0, "rung": None}


def install():
    real_sp = nsolver.solve_pattern
    real_ss = nsolver.solve_system
    real_newton = gsolve.newton_solve
    real_root = gsolve.root

    def spy_solve_pattern(par, mode, n_B, T, flags, pattern, **kw):
        rec = {"n_B": float(n_B), "pattern": pattern,
               "warm": kw.get("x0") is not None, "rungs": []}
        cur["rec"], cur["n_jac_calls"] = rec, 0
        t0, c0 = time.perf_counter(), time.process_time()
        pt = real_sp(par, mode, n_B, T, flags, pattern, **kw)
        rec["wall_s"] = time.perf_counter() - t0
        rec["cpu_s"] = time.process_time() - c0
        rec["converged"] = bool(pt.converged)
        rec["realised"] = pt.pattern_realised
        rec["f"] = float(pt.f)
        CALLS.append(rec)
        cur["rec"] = None
        return pt

    def spy_solve_system(residual, x0, scales_at, x0_fallback=None, tol=None,
                         jac=None):
        rec = cur["rec"]
        if jac is None:
            label = "D"
        else:
            cur["n_jac_calls"] += 1
            label = "AB" if cur["n_jac_calls"] == 1 else "C"
        prev, cur["rung"] = cur["rung"], label
        entry = {"rung": label, "stages": []}
        if rec is not None:
            rec["rungs"].append(entry)
        cur["entry"] = entry
        t0 = time.perf_counter()
        try:
            x, err, ok = real_ss(residual, x0, scales_at, x0_fallback, tol, jac)
        finally:
            cur["rung"] = prev
        entry["wall_s"] = time.perf_counter() - t0
        entry["err"], entry["ok"] = float(err), bool(ok)
        return x, err, ok

    def spy_newton(residual, jac, x0, scales_at, **kw):
        t0 = time.perf_counter()
        x, err, ok = real_newton(residual, jac, x0, scales_at, **kw)
        e = cur.get("entry")
        if e is not None and cur["rung"] is not None:
            e["stages"].append({"stage": "newton", "wall_s": time.perf_counter() - t0,
                                "err": float(err), "ok": bool(ok)})
        return x, err, ok

    def spy_root(fun, x0, **kw):
        t0 = time.perf_counter()
        sol = real_root(fun, x0, **kw)
        e = cur.get("entry")
        if e is not None and cur["rung"] is not None:
            e["stages"].append({"stage": kw.get("method", "hybr"),
                                "wall_s": time.perf_counter() - t0,
                                "nfev": int(getattr(sol, "nfev", -1)),
                                "njev": int(getattr(sol, "njev", 0) or 0)})
        return sol

    nsolver.solve_pattern = spy_solve_pattern
    nsolver.solve_system = spy_solve_system
    gsolve.newton_solve = spy_newton
    gsolve.root = spy_root


def main():
    import scipy
    stack = (f"python {sys.version.split()[0]}, numpy {np.__version__}, "
             f"scipy {scipy.__version__}")
    par = njl.Parameters.named(BENCH_SET)
    if os.environ.get("PROTO13_VX"):
        # The same census taken UNDER the bounded ladder, so the map learns
        # what the build is made of once the rescue ladder is no longer the
        # story. VX installs first; the spies below then wrap ITS
        # `solve_system`, which is what `attempt` will be calling.
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import proto13_variant as V
        V.install("VX")
    # warm the jit on two densities before the census opens
    njl.eos_table(par, BENCH_MODE, BENCH_SPECIES,
                  {"nB": BENCH_NB[:2], "T": np.array([BENCH_T])},
                  backend=BENCH_BACKEND)
    install()
    CALLS.clear()
    w0, c0 = time.perf_counter(), time.process_time()
    result = njl.eos_table(par, BENCH_MODE, BENCH_SPECIES,
                           {"nB": BENCH_NB, "T": np.array([BENCH_T])},
                           backend=BENCH_BACKEND)
    wall, cpu = time.perf_counter() - w0, time.process_time() - c0
    rows = njl.table.rows_from_result(result)
    print(f"{stack}\n  loadavg {os.getloadavg()}\n"
          f"  {len(rows)} rows  wall {wall:.1f}s cpu {cpu:.1f}s  "
          f"{1e3 * wall / len(rows):.1f} ms/pt  ({len(CALLS)} candidates)",
          flush=True)
    out = [{"n_B": float(r["n_B"]), "P": float(r["P"]), "eps": float(r["eps"]),
            "pattern": r["pattern"], "realised": r["pattern_realised"],
            "Delta": [float(r["Delta_1"]), float(r["Delta_2"]),
                      float(r["Delta_3"])]} for r in rows]
    path = (".scratch/njl-speed/proto13_rungs_VX.json"
            if os.environ.get("PROTO13_VX") else
            ".scratch/njl-speed/proto13_rungs.json")
    with open(path, "w") as fh:
        json.dump({"stack": stack, "loadavg": os.getloadavg(), "wall_s": wall,
                   "cpu_s": cpu, "n_rows": len(rows), "winners": out,
                   "calls": CALLS}, fh)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
