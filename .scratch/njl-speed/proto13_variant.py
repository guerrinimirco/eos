"""PROTOTYPE (wayfinder ticket 13) -- throwaway. Not production code.

WHAT THE CENSUS (`proto13_rungs.py`) FOUND, which decides the variants here:

  * 13 CFL candidates below the CFL onset cost 132.7 s of 171.8 s (77.2%).
    At every one of them the Jacobian rungs FAIL and rung D SUCCEEDS, onto
    the 2SC root -- so `ok` is True and the warm-start cold retry never fires.
  * The ONE CFL candidate that holds its layout (n_B = 0.5686) is rescued by
    `hybr`, in 77 evaluations and 2.3 s. It never reaches `lm` and never
    reaches rung D.
  * `lm` is entered by exactly those 13 and by nobody else: 50.2 s, 29.2%,
    and it rescues nothing.
  * Rung C fires 0 times.

So the rungs past `hybr` buy 13 duplicates and nothing at all: lm 50.2 s +
rung D 36.1 s = 86.3 s, 50.2% of the build. And an evaluation cap cannot
separate the onset from the dead -- the onset uses MORE hybr evaluations (77)
than any of the 13 (31-73). The separator is not the budget and not the
residual; it is the LAYOUT.

    V0   control: today's ladder
    VX   the census's cut: a CROSS-SEEDED candidate (one handed another
         pattern's state, not its own previous point) gets one Newton and one
         `hybr` and then an answer. No `lm`, no differenced repeat, no retry
         from cold.
    VD   rung D refused. The ticket's "cheapest thing to try first" -- and a
         TRAP, because `ok` now stays False at those 13, so `solve_pattern`'s
         warm-start cold retry runs the whole ladder a second time.
    VL   layout stop: a 2SC or CFL solve whose iterate has left its layout
         stops there. The collapsed root belongs to a pattern that is in the
         enumeration and finds it more cheaply, so there is nothing to rescue.
         This is rung B BOUNDED (hybr keeps the onset, lm never runs) and
         rung D removed, by one condition rather than two rules.

    python3 .scratch/njl-speed/proto13_variant.py VL
"""
import json
import os
import sys
import time

import numpy as np

from eos import njl
from eos.general.pairing import realised_pattern
import eos.general.solve as gsolve
from eos.general.solve import (LM_MAX_EVALUATIONS, RESIDUAL_TOL,
                               newton_solve, newton_polish,  # noqa: F401
                               scaled_residual_max)
# `gsolve.newton_solve` / `gsolve.root` rather than the names imported above:
# both are what `eos.general.solve.solve_system` itself calls, so a census
# that spies on the module attributes sees VX's rungs too.
import eos.njl.solver as nsolver

BENCH_SET = "rg_njl1"
BENCH_MODE = "beta_eq_neutrinoless"
BENCH_T = 0.0
BENCH_SPECIES = njl.SpeciesFlags(csc=True)
BENCH_BACKEND = "fast"
BENCH_NB = np.linspace(0.5, 1.55, 200)

cur = {"pattern": None, "idx": None, "cross": False, "jac_calls": 0,
       "last_ok": False}
STOPS = {"layout": 0, "rung_D": 0, "cold_retry": 0, "bounded": 0}


def left_layout(x):
    """`nsolver._left_layout` without par/spec: the Delta slots of the current
    pattern, read straight off the unknown vector."""
    if cur["pattern"] not in ("2SC", "CFL") or cur["idx"] is None:
        return False
    Delta = np.array([x[i] if i is not None else 0.0 for i in cur["idx"]])
    return realised_pattern(Delta) != cur["pattern"]


def solve_system_VL(residual, x0, scales_at, x0_fallback=None, tol=None,
                    jac=None):
    """`eos.general.solve.solve_system` with ONE condition added.

    Verbatim apart from `gave_up()`: after every rung that did not reach the
    gate, a 2SC or CFL iterate that has left its layout is returned as
    non-converged instead of being carried to the next rung. It has found a
    root of another pattern -- one the enumeration is solving anyway, from its
    own seed, in a fraction of the time -- so every rung after this point is
    spent re-deriving a duplicate.
    """
    def gave_up(x):
        if left_layout(x):
            STOPS["layout"] += 1
            return True
        return False

    best_x, best_err = np.asarray(x0, dtype=float), np.inf
    if jac is None:
        # rung D: the differenced repeat of a ladder that has already run with
        # an exact Jacobian. The census measured what it converges to.
        if left_layout(np.asarray(x0, dtype=float)):
            STOPS["rung_D"] += 1
            return np.asarray(x0, dtype=float), np.inf, False
    else:
        best_x, best_err, ok = gsolve.newton_solve(residual, jac, best_x,
                                                   scales_at)
        if ok:
            return best_x, best_err, True
        if gave_up(best_x):
            return best_x, best_err, False

    attempts = [('hybr', x0), ('lm', x0)]
    if x0_fallback is not None:
        attempts.append(('hybr', x0_fallback))
    for method, guess in attempts:
        options = ({'maxiter': LM_MAX_EVALUATIONS} if method == 'lm' else None)
        sol = gsolve.root(residual, guess, method=method, tol=tol,
                          options=options, jac=jac)
        err = scaled_residual_max(residual(sol.x), scales_at(sol.x))
        if err < best_err:
            best_x, best_err = sol.x, err
        if best_err <= RESIDUAL_TOL:
            break
        if gave_up(sol.x):
            return sol.x, err, False
    if best_err > RESIDUAL_TOL:
        best_x, best_err = gsolve.newton_polish(residual, best_x, scales_at,
                                                best_err)
    return best_x, best_err, bool(best_err <= RESIDUAL_TOL)


def solve_system_VD(residual, x0, scales_at, x0_fallback=None, tol=None,
                    jac=None):
    """Rung D and nothing else."""
    if jac is None:
        STOPS["rung_D"] += 1
        return np.asarray(x0, dtype=float), np.inf, False
    return _real_ss(residual, x0, scales_at, x0_fallback, tol, jac)


def solve_VX(par, mode, n_B, T=0.0, flags=None, x0=None, patterns=None,
             vac=None, backend="reference", pair_nodes_per_panel=None,
             **fractions):
    """`eos.njl.solver.solve`, with ONE line added: where the seed came from.

    A candidate is CROSS-SEEDED when it is handed `seed_from(reference, ...)`
    -- the converged state of a DIFFERENT pattern, mapped into this one's
    layout. That seed is the enumeration's way of asking "is there a root of
    this pattern near the state we already have", and a no is an ordinary
    answer. A candidate warm-started from its OWN converged point one density
    down is a different thing: there was a root here a moment ago, so a failure
    is evidence the ladder should work against. `solve_pattern` cannot tell
    them apart -- both arrive as `x0` -- and only `solve` knows.
    """
    if flags is None:
        flags = nsolver.SpeciesFlags()
    leptons = fractions.pop("leptons", None)
    spec = nsolver.mode_spec(mode, leptons=leptons, **fractions)
    nsolver._refuse_fixed_YS(flags, spec, 'eos.njl')
    if vac is None:
        vac = nsolver.vacuum_solution(par)
    patterns = nsolver.patterns_for(flags, patterns)

    seeds = ({} if x0 is None else
             dict(x0) if isinstance(x0, dict) else {patterns[0]: x0})
    candidates = []
    reference = None
    for pattern in patterns:
        seed = seeds.get(pattern)
        if seed is None and reference is not None:
            seed = nsolver.seed_from(reference, par, spec, pattern)
        cur["cross"] = seed is not None and pattern not in seeds   # THE LINE
        cur["jac_calls"], cur["last_ok"] = 0, False
        try:
            point = nsolver.solve_pattern(
                par, mode, n_B, T, flags, pattern, spec=spec, x0=seed,
                vac=vac, backend=backend,
                pair_nodes_per_panel=pair_nodes_per_panel)
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            continue
        finally:
            cur["cross"] = False
        candidates.append(point)
        if reference is None and point.converged:
            reference = point

    converged = [p for p in candidates if p.converged]
    if converged:
        winner = min(converged, key=lambda p: p.f)
        winner._seeds = {p.pattern: np.array(p.x, dtype=float)
                         for p in converged
                         if p.pattern_realised == p.pattern}
        return winner
    if candidates:
        return candidates[0]
    return nsolver.solve_pattern(par, mode, n_B, T, flags, "unpaired",
                                 spec=spec, vac=vac, backend=backend,
                                 pair_nodes_per_panel=pair_nodes_per_panel)


def solve_system_VX(residual, x0, scales_at, x0_fallback=None, tol=None,
                    jac=None):
    """The bounded ladder a CROSS-SEEDED candidate gets: one Newton, one
    `hybr`, the polish, and then the answer -- no `lm`, no differenced repeat,
    no retry from the cold guess.

    The census picked the cut. Of the 28 candidates whose first Newton misses
    the gate, `hybr` rescues the only CFL candidate that holds its layout (the
    onset, 77 evaluations, 2.3 s). `lm` is entered by exactly the 13 that do
    NOT hold it, costs 50.2 s, and rescues none of them; rung D then costs
    36.1 s and converges all 13 onto the 2SC root -- a state the 2SC candidate
    has already reported, at the same free energy to 1.9e-10.
    """
    if not cur["cross"]:
        return _real_ss(residual, x0, scales_at, x0_fallback, tol, jac)
    if jac is None:
        STOPS["rung_D"] += 1
        return np.asarray(x0, dtype=float), np.inf, False
    cur["jac_calls"] += 1
    if cur["jac_calls"] > 1 and not cur["last_ok"]:
        # `solve_pattern` retries a failed warm start from the cold guess.
        # For a cross-seed there is no warm start to distrust -- the whole
        # ladder simply runs a second time (this is what makes VD slower than
        # the control, at 25 rung-D refusals for 13 candidates).
        #
        # `last_ok` is what separates that retry from rung C, which is ALSO a
        # second Jacobian call in the same `solve_pattern` but follows a solve
        # that CONVERGED and left its layout. Rung C fires 0 times on the
        # pinned benchmark, but `attempt`'s docstring records it finding the
        # CFL ground state at T = 20 MeV, so it is kept.
        STOPS["cold_retry"] += 1
        cur["last_ok"] = False
        return np.asarray(x0, dtype=float), np.inf, False

    best_x = np.asarray(x0, dtype=float)
    best_x, best_err, ok = gsolve.newton_solve(residual, jac, best_x, scales_at)
    if ok:
        cur["last_ok"] = True
        return best_x, best_err, True
    sol = gsolve.root(residual, x0, method='hybr', tol=tol, jac=jac)
    err = scaled_residual_max(residual(sol.x), scales_at(sol.x))
    if err < best_err:
        best_x, best_err = sol.x, err
    if best_err > RESIDUAL_TOL:
        STOPS["bounded"] += 1
        best_x, best_err = gsolve.newton_polish(residual, best_x, scales_at,
                                                best_err)
    cur["last_ok"] = bool(best_err <= RESIDUAL_TOL)
    return best_x, best_err, cur["last_ok"]


_real_ss = nsolver.solve_system


def install(variant):
    """The layout check needs to know which pattern is being solved and where
    its gaps sit in the unknown vector; only `solve_pattern` knows that."""
    real_sp = nsolver.solve_pattern

    def spy(par, mode, n_B, T, flags, pattern, **kw):
        spec = kw.get("spec")
        if spec is None:
            spec = nsolver.mode_spec(mode, leptons=kw.get("leptons"))
        names = nsolver.unknown_slots(par, spec, pattern)
        prev = (cur["pattern"], cur["idx"])
        cur["pattern"] = pattern
        cur["idx"] = [names.index(f"Delta_{e + 1}")
                      if f"Delta_{e + 1}" in names else None for e in range(3)]
        try:
            return real_sp(par, mode, n_B, T, flags, pattern, **kw)
        finally:
            cur["pattern"], cur["idx"] = prev

    if variant == "V0":
        return
    if variant == "VX":
        import eos.njl.table as ntable
        nsolver.solve = ntable.solve = solve_VX   # table.py binds its own name
        nsolver.solve_system = solve_system_VX
        return
    nsolver.solve_pattern = spy
    nsolver.solve_system = (solve_system_VL if variant == "VL"
                            else solve_system_VD)


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "V0"
    import scipy
    stack = (f"python {sys.version.split()[0]}, numpy {np.__version__}, "
             f"scipy {scipy.__version__}")
    par = njl.Parameters.named(BENCH_SET)
    install(variant)
    njl.eos_table(par, BENCH_MODE, BENCH_SPECIES,
                  {"nB": BENCH_NB[:2], "T": np.array([BENCH_T])},
                  backend=BENCH_BACKEND)          # jit warm-up, not timed
    STOPS["layout"] = STOPS["rung_D"] = 0

    w0, c0 = time.perf_counter(), time.process_time()
    result = njl.eos_table(par, BENCH_MODE, BENCH_SPECIES,
                           {"nB": BENCH_NB, "T": np.array([BENCH_T])},
                           backend=BENCH_BACKEND)
    wall, cpu = time.perf_counter() - w0, time.process_time() - c0
    rows = njl.table.rows_from_result(result)
    print(f"{variant}: {stack}\n  loadavg {os.getloadavg()}\n"
          f"  {len(rows)} rows  wall {wall:.1f}s cpu {cpu:.1f}s  "
          f"{1e3 * wall / len(rows):.1f} ms/pt   stops {STOPS}", flush=True)
    out = [{"n_B": float(r["n_B"]), "P": float(r["P"]), "eps": float(r["eps"]),
            "pattern": r["pattern"], "realised": r["pattern_realised"],
            "Delta": [float(r["Delta_1"]), float(r["Delta_2"]),
                      float(r["Delta_3"])]} for r in rows]
    path = f".scratch/njl-speed/proto13_{variant}.json"
    with open(path, "w") as fh:
        json.dump({"variant": variant, "stack": stack,
                   "loadavg": os.getloadavg(), "wall_s": wall, "cpu_s": cpu,
                   "n_rows": len(rows), "stops": dict(STOPS),
                   "winners": out}, fh)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
