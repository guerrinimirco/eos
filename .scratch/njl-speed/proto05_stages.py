"""PROTOTYPE (wayfinder ticket 05) -- throwaway.

The census (proto05_instrument.py) shows the cost sits on candidates whose
FIRST damped-Newton run fails. `solve_system` already returns immediately when
that run succeeds, so a "pre-screen" buys nothing on the candidates that work:
the whole question is what the ladder below a failed Newton costs and what it
buys. This run attributes it, stage by stage.

Stages of `eos.njl.solver.solve_pattern`'s `attempt`, jac path:

    A  newton_solve(jac)                       inside solve_system #1
    B  root(hybr,jac) / root(lm,jac) / polish  rest of solve_system #1
    C  solve_system(reinflated)   -- the "converged but LEFT its layout" rescue
    D  solve_system(no jac)       -- the differenced rescue after a failure
    then, if still failed and the seed was warm: the whole ladder again from
    the COLD guess.

    python3 .scratch/njl-speed/proto05_stages.py
"""
import json
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

OUT = ".scratch/njl-speed/proto05_stages.json"
#: default enumeration, or the three-pattern list once `free` is dealt
#: with -- sys.argv[1] == "three".

records = []
current = {}

_real_solve_pattern = nsolver.solve_pattern
_real_solve_system = nsolver.solve_system
_real_newton = gsolve.newton_solve


def spy_newton(residual, jac, x0, scales_at, **kw):
    t0 = time.perf_counter()
    x, err, ok = _real_newton(residual, jac, x0, scales_at, **kw)
    if current:
        current["newton"].append([time.perf_counter() - t0, float(err),
                                  bool(ok)])
    return x, err, ok


def spy_solve_system(residual, x0, scales_at, **kw):
    """One rung. Which rung it is, is read off the call order in `attempt`."""
    t0 = time.perf_counter()
    n_before = len(current["newton"]) if current else 0
    x, err, ok = _real_solve_system(residual, x0, scales_at, **kw)
    if current:
        current["systems"].append({
            "wall_s": time.perf_counter() - t0,
            "jac": kw.get("jac") is not None,
            "err": float(err), "ok": bool(ok),
            # the Newton run this rung opened with, if it had a Jacobian
            "newton_s": (current["newton"][n_before][0]
                         if len(current["newton"]) > n_before else 0.0),
            "newton_err": (current["newton"][n_before][1]
                           if len(current["newton"]) > n_before else None),
        })
    return x, err, ok


def spy_solve_pattern(par, mode, n_B, T, flags, pattern, **kw):
    global current
    outer = current
    current = {"n_B": float(n_B), "pattern": pattern,
               "warm": kw.get("x0") is not None,
               "newton": [], "systems": []}
    t0 = time.perf_counter()
    try:
        point = _real_solve_pattern(par, mode, n_B, T, flags, pattern, **kw)
    finally:
        current["wall_s"] = time.perf_counter() - t0
        records.append(current)
        rec, current = records[-1], outer
    rec.update(converged=bool(point.converged), final_err=float(point.error),
               realised=point.pattern_realised, f=float(point.f),
               P=float(point.P))
    return point


nsolver.solve_pattern = spy_solve_pattern
nsolver.solve_system = spy_solve_system
gsolve.newton_solve = spy_newton
nsolver.newton_solve = spy_newton


def main():
    import scipy
    stack = (f"python {sys.version.split()[0]} ({sys.executable}), "
             f"numpy {np.__version__}, scipy {scipy.__version__}")
    print(stack, flush=True)
    patterns = (("unpaired", "2SC", "CFL")
                if len(sys.argv) > 1 and sys.argv[1] == "three" else None)
    print(f"patterns: {patterns}", flush=True)
    par = njl.Parameters.named(BENCH_SET)
    njl.eos_table(par, BENCH_MODE, BENCH_SPECIES,
                  {"nB": BENCH_NB[:2], "T": np.array([BENCH_T])},
                  patterns=patterns, backend=BENCH_BACKEND)
    records.clear()
    w0, c0 = time.perf_counter(), time.process_time()
    result = njl.eos_table(par, BENCH_MODE, BENCH_SPECIES,
                           {"nB": BENCH_NB, "T": np.array([BENCH_T])},
                           patterns=patterns, backend=BENCH_BACKEND)
    wall, cpu = time.perf_counter() - w0, time.process_time() - c0
    rows = njl.table.rows_from_result(result)
    print(f"n_rows {len(rows)}  wall {wall:.1f}s cpu {cpu:.1f}s "
          f"{1e3 * wall / len(rows):.1f} ms/pt", flush=True)
    with open(OUT, "w") as fh:
        json.dump({"stack": stack, "wall_s": wall, "cpu_s": cpu,
                   "n_rows": len(rows), "records": records}, fh)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
