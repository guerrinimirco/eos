"""Ticket 10 -> 18: which rung of `attempt`'s ladder rescues the gapless CFL
candidate that the bounded (cross-seeded) ladder loses? Every `solve_system`
call inside `solve_pattern` is logged with its methods, Jacobian, and outcome.

    PYTHONPATH=. python3 t10_rung_probe.py
"""
import eos.njl.solver as S
from eos import njl
from eos.njl.solver import mode_spec, seed_from, solve_pattern

_ss = S.solve_system
LOG = []


def logged(rows, seed, scales, **kw):
    x, err, ok = _ss(rows, seed, scales, **kw)
    LOG.append((kw.get("methods", ("hybr", "lm")),
                "analytic" if kw.get("jac") is not None else "differenced",
                f"{err:.1e}", ok))
    return x, err, ok


S.solve_system = logged
par = njl.Parameters.named("rg_njl1")
flags = njl.SpeciesFlags(csc=True)
spec = mode_spec("fixed_YC", leptons=True, Y_C=0.1)
for n_B in (0.800, 1.000):
    ref = solve_pattern(par, "fixed_YC", n_B, 0.0, flags, "unpaired",
                        spec=spec, backend="fast")
    seed = seed_from(ref, par, spec, "CFL")
    for bounded in (True, False):
        LOG.clear()
        p = solve_pattern(par, "fixed_YC", n_B, 0.0, flags, "CFL", spec=spec,
                          x0=seed, backend="fast", cross_seeded=bounded)
        print(f"n_B {n_B} {'bounded' if bounded else 'full'}: converged "
              f"{p.converged} realised {p.pattern_realised} gapless "
              f"{p.gapless} eps {p.eps:.3f}")
        for i, entry in enumerate(LOG):
            print(f"   call {i}: methods {entry[0]}, {entry[1]} Jacobian -> "
                  f"err {entry[2]} ok {entry[3]}")
