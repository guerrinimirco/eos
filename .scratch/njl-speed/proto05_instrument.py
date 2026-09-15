"""PROTOTYPE (wayfinder ticket 05) -- throwaway. Not production code.

Instruments the PINNED benchmark (notebooks/quark_timing.py, commit 54c7be9)
so ticket 05 can see, per (n_B, pattern) candidate:

  * the SCREEN -- what the first damped-Newton run stops at. That is exactly
    what `solve_pattern(..., rescue=False)` returns, so the pre-screen's
    discriminator is measured without building it: its scaled residual, and
    the pattern its stalled iterate's gaps REALISE.
  * what the full ladder then did: final error, converged, realised pattern.
  * whether the candidate had a seed of its OWN pattern or a cross-pattern
    one from `seed_from` (which is what makes a live candidate screen badly).
  * the wall time the candidate cost.

and, per density, the baseline winner and its P -- the gate's reference table.

    python3 .scratch/njl-speed/proto05_instrument.py
"""
import json
import sys
import time

import numpy as np

from eos import njl
from eos.general.pairing import realised_pattern
import eos.general.solve as gsolve
import eos.njl.solver as nsolver

BENCH_SET = "rg_njl1"
BENCH_MODE = "beta_eq_neutrinoless"
BENCH_T = 0.0
BENCH_SPECIES = njl.SpeciesFlags(csc=True)
BENCH_BACKEND = "fast"
BENCH_NB = np.linspace(0.5, 1.55, 200)

OUT = ".scratch/njl-speed/proto05_baseline.json"

records = []
current = {}
cross_seeded = [None]         # the pattern the LAST seed_from() served

_real_solve_pattern = nsolver.solve_pattern
_real_newton = gsolve.newton_solve
_real_seed_from = nsolver.seed_from


def spy_seed_from(point, par, spec, pattern):
    # `solve` calls this immediately before solve_pattern for that candidate,
    # so "the last one served" is exactly "this candidate was cross-seeded".
    cross_seeded[0] = pattern
    return _real_seed_from(point, par, spec, pattern)


def gaps_of(x, slots):
    return [float(x[i]) if i is not None else 0.0 for i in slots]


def spy_newton(residual, jac, x0, scales_at, **kw):
    x, err, ok = _real_newton(residual, jac, x0, scales_at, **kw)
    if current:
        g = gaps_of(x, current["slots"])
        current["newton"].append([float(err), bool(ok), g,
                                  realised_pattern(g)])
    return x, err, ok


def spy_solve_pattern(par, mode, n_B, T, flags, pattern, **kw):
    global current
    outer = current
    spec = kw.get("spec")
    if spec is None:
        spec = nsolver.mode_spec(mode, leptons=None)
    names = nsolver.unknown_slots(par, spec, pattern)
    slots = [names.index(f"Delta_{e + 1}") if f"Delta_{e + 1}" in names else None
             for e in range(3)]
    x0 = kw.get("x0")
    current = {"n_B": float(n_B), "pattern": pattern,
               "warm": x0 is not None,
               "cross": cross_seeded[0] == pattern,
               "seed_gaps": gaps_of(x0, slots) if x0 is not None else None,
               "slots": slots, "newton": []}
    cross_seeded[0] = None
    t0 = time.perf_counter()
    try:
        point = _real_solve_pattern(par, mode, n_B, T, flags, pattern, **kw)
    finally:
        current["wall_s"] = time.perf_counter() - t0
        current.pop("slots")
        records.append(current)
        rec, current = records[-1], outer
    rec.update(converged=bool(point.converged), final_err=float(point.error),
               realised=point.pattern_realised, f=float(point.f),
               P=float(point.P), Delta=[float(d) for d in point.Delta])
    return point


nsolver.solve_pattern = spy_solve_pattern
nsolver.seed_from = spy_seed_from
gsolve.newton_solve = spy_newton
nsolver.newton_solve = spy_newton


def main():
    import scipy
    stack = (f"python {sys.version.split()[0]} ({sys.executable}), "
             f"numpy {np.__version__}, scipy {scipy.__version__}")
    print(stack, flush=True)

    par = njl.Parameters.named(BENCH_SET)
    axes = {"nB": BENCH_NB, "T": np.array([BENCH_T])}

    njl.eos_table(par, BENCH_MODE, BENCH_SPECIES,
                  {"nB": BENCH_NB[:2], "T": np.array([BENCH_T])},
                  patterns=None, backend=BENCH_BACKEND)
    records.clear()

    w0, c0 = time.perf_counter(), time.process_time()
    result = njl.eos_table(par, BENCH_MODE, BENCH_SPECIES, axes,
                           patterns=None, backend=BENCH_BACKEND)
    wall, cpu = time.perf_counter() - w0, time.process_time() - c0
    rows = njl.table.rows_from_result(result)
    print(f"n_rows {len(rows)}  wall {wall:.1f}s  cpu {cpu:.1f}s  "
          f"{1e3 * wall / len(rows):.1f} ms/pt", flush=True)

    winners = [{"n_B": float(r["n_B"]), "P": float(r["P"]),
                "eps": float(r["eps"]), "pattern": r["pattern"],
                "realised": r["pattern_realised"]}
               for r in rows]
    with open(OUT, "w") as fh:
        json.dump({"stack": stack, "wall_s": wall, "cpu_s": cpu,
                   "n_rows": len(rows), "records": records,
                   "winners": winners}, fh)
    print(f"wrote {OUT}: {len(records)} candidates", flush=True)


if __name__ == "__main__":
    main()
