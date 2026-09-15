"""PROTOTYPE (wayfinder ticket 05) -- throwaway. Not production code.

THE CENSUS SAID: 91.9% of the default 200-point enumeration is the `free`
candidate, and it is cross-seeded at all 200 densities. Not sometimes --
ALWAYS, and structurally:

    `eos.general.pairing.realised_pattern` returns one of the eight names of
    `_REALISED`, and 'free' is not one of them. So `solve`'s `_seeds` filter,
    `p.pattern_realised == p.pattern`, CANNOT pass for the free candidate. It
    is denied a warm start at every density of every sweep, and pays a hunt
    from the unpaired state's potentials with an asymmetric gap seed each time.

So the lever ticket 05 was chartered to test -- a loose ranking pass -- has
nothing to bite on: `solve_system` already runs Newton first and returns the
moment it succeeds, so a candidate that works is ALREADY screened for free.
What is left is a seeding defect. This module measures the one-line fix.

Variants, selected by argv[1]:

    V0   unmodified `eos.njl.solver.solve` (control)
    V1   `free` carries its converged vector forward under its own key
    V2   V1, and `free` is re-hunted COLD every FREE_RECOLD densities, so a
         capture cannot outlive that window
    V3   `free` dropped from the enumeration (the three-pattern list; what
         ticket 02 already measured, taken again under the same load)

    python3 .scratch/njl-speed/proto05_variant.py V1
"""
import json
import sys
import time

import numpy as np

from eos import njl
import eos.njl.solver as nsolver
import eos.njl.table as ntable
from eos.njl.solver import (
    SpeciesFlags, mode_spec, patterns_for, seed_from, solve_pattern,
    vacuum_solution, _refuse_fixed_YS)

BENCH_SET = "rg_njl1"
BENCH_MODE = "beta_eq_neutrinoless"
BENCH_T = 0.0
BENCH_SPECIES = njl.SpeciesFlags(csc=True)
BENCH_BACKEND = "fast"
BENCH_NB = np.linspace(0.5, 1.55, 200)

FREE_RECOLD = 10          # V2 only
_seen = {"n": 0}


def proto_solve(par, mode, n_B, T=0.0, flags=None, x0=None, patterns=None,
                vac=None, backend="reference", pair_nodes_per_panel=None,
                **fractions):
    """`eos.njl.solver.solve` with ONE line changed -- see `keeps_seed`."""
    if flags is None:
        flags = SpeciesFlags()
    leptons = fractions.pop("leptons", None)
    spec = mode_spec(mode, leptons=leptons, **fractions)
    _refuse_fixed_YS(flags, spec, 'eos.njl')
    if vac is None:
        vac = vacuum_solution(par)
    patterns = patterns_for(flags, patterns)

    seeds = ({} if x0 is None else
             dict(x0) if isinstance(x0, dict) else {patterns[0]: x0})
    if VARIANT == "V2" and _seen["n"] % FREE_RECOLD == 0:
        seeds.pop("free", None)      # one cold hunt per window
    _seen["n"] += 1

    candidates = []
    reference = None
    for pattern in patterns:
        seed = seeds.get(pattern)
        if seed is None and reference is not None:
            seed = seed_from(reference, par, spec, pattern)
        try:
            point = nsolver.solve_pattern(
                par, mode, n_B, T, flags, pattern, spec=spec, x0=seed,
                vac=vac, backend=backend,
                pair_nodes_per_panel=pair_nodes_per_panel)
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            continue
        candidates.append(point)
        if reference is None and point.converged:
            reference = point

    converged = [p for p in candidates if p.converged]
    if converged:
        winner = min(converged, key=lambda p: p.f)
        winner._seeds = {p.pattern: np.array(p.x, dtype=float)
                         for p in converged if keeps_seed(p)}
        return winner
    if candidates:
        return candidates[0]
    return nsolver.solve_pattern(par, mode, n_B, T, flags, "unpaired", spec=spec,
                         vac=vac, backend=backend,
                         pair_nodes_per_panel=pair_nodes_per_panel)


def keeps_seed(p):
    """THE ONE LINE. Today: `p.pattern_realised == p.pattern`.

    That test asks "did this candidate stay in the layout it was solved in",
    which guards a named pattern against carrying a collapsed root up the
    sweep. For 'free' it asks something unanswerable instead: 'free' is a
    SEED, not a state, so `realised_pattern` never returns it and the test
    never passes. Every state is a legal outcome of the free layout, so there
    is no collapse for the guard to catch -- only capture, which is a
    different risk and is what V2 bounds.
    """
    if p.pattern == "free":
        return VARIANT in ("V1", "V2")
    return p.pattern_realised == p.pattern


CENSUS = []


def install_census():
    import os
    if not os.environ.get("NJL_CENSUS"):
        return
    real = nsolver.solve_pattern

    def spy(par, mode, n_B, T, flags, pattern, **kw):
        t0 = time.perf_counter()
        pt = real(par, mode, n_B, T, flags, pattern, **kw)
        CENSUS.append({"n_B": float(n_B), "pattern": pattern,
                       "warm": kw.get("x0") is not None,
                       "wall_s": time.perf_counter() - t0,
                       "converged": bool(pt.converged),
                       "realised": pt.pattern_realised, "f": float(pt.f)})
        return pt

    nsolver.solve_pattern = spy


def main():
    global VARIANT
    VARIANT = sys.argv[1] if len(sys.argv) > 1 else "V0"
    import scipy
    stack = (f"python {sys.version.split()[0]} ({sys.executable}), "
             f"numpy {np.__version__}, scipy {scipy.__version__}")
    load = open("/dev/null") and __import__("os").getloadavg()
    print(f"{VARIANT}: {stack}\n  loadavg at start {load}", flush=True)

    patterns = ("unpaired", "2SC", "CFL") if VARIANT == "V3" else None
    if VARIANT in ("V1", "V2"):
        ntable.solve = proto_solve            # what table.py actually calls
        nsolver.solve = proto_solve

    par = njl.Parameters.named(BENCH_SET)
    njl.eos_table(par, BENCH_MODE, BENCH_SPECIES,
                  {"nB": BENCH_NB[:2], "T": np.array([BENCH_T])},
                  patterns=patterns, backend=BENCH_BACKEND)
    _seen["n"] = 0
    install_census()
    CENSUS.clear()

    w0, c0 = time.perf_counter(), time.process_time()
    result = njl.eos_table(par, BENCH_MODE, BENCH_SPECIES,
                           {"nB": BENCH_NB, "T": np.array([BENCH_T])},
                           patterns=patterns, backend=BENCH_BACKEND)
    wall, cpu = time.perf_counter() - w0, time.process_time() - c0
    rows = njl.table.rows_from_result(result)
    print(f"{VARIANT}: {len(rows)} rows  wall {wall:.1f}s cpu {cpu:.1f}s  "
          f"{1e3 * wall / len(rows):.1f} ms/pt  "
          f"loadavg end {__import__('os').getloadavg()}", flush=True)
    out = [{"n_B": float(r["n_B"]), "P": float(r["P"]), "eps": float(r["eps"]),
            "pattern": r["pattern"], "realised": r["pattern_realised"],
            "Delta": [float(r["Delta_1"]), float(r["Delta_2"]),
                      float(r["Delta_3"])]}
           for r in rows]
    path = f".scratch/njl-speed/proto05_{VARIANT}.json"
    with open(path, "w") as fh:
        json.dump({"variant": VARIANT, "stack": stack, "wall_s": wall,
                   "cpu_s": cpu, "n_rows": len(rows), "winners": out,
                   "census": CENSUS}, fh)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
