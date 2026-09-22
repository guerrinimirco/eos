"""Ticket 18: the CFL cross seed moved along the rotated charge Q~ to just past
its own unlocking edge (bisection on the state's `gapless` flag), scored for
robustness: fast bounded path over five Jacobian draws, reference full ladder
over three seed draws jittered at 1e-9 relative. Vacuum rule re-pointed as in
t10_gapless.py.

    PYTHONPATH=. python3 .scratch/njl-speed/t18_edge_seed.py ARM DENSITIES...
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                      # noqa: E402

import t10_tables as TT                                 # noqa: E402
import eos.njl.solver as S                              # noqa: E402
from eos import njl                                     # noqa: E402
from eos.njl.solver import (_state, mode_spec, seed_from, solve_pattern,  # noqa: E402
                            unknown_slots)
from eos.njl.thermodynamics import vacuum_solution      # noqa: E402

par = njl.Parameters.named("rg_njl1")
flags = njl.SpeciesFlags(csc=True)
spec = mode_spec("fixed_YC", leptons=True, Y_C=0.1)
vac = vacuum_solution(par)
names = unknown_slots(par, spec, "CFL")
ix = {n: i for i, n in enumerate(names)}
analytic = S.residual_jacobian
H, V = (int(a) for a in sys.argv[1].split("/"))
TT.set_vacuum(V)
densities = [float(d) for d in sys.argv[2:]]


def along(x, t):
    x = x.copy()
    x[ix["mu_C"]] += t
    x[ix["mu_3"]] -= t
    x[ix["mu_8"]] -= 0.5 * t
    return x


def edge_seed(seed, backend):
    def gapless(t):
        return _state(along(seed, t), par, spec, "CFL", 0.0, vac, False,
                      backend, H).gapless
    lo, hi = 0.0, 2.0 * float(np.max(seed[[ix["Delta_1"], ix["Delta_2"],
                                           ix["Delta_3"]]]))
    if gapless(lo) or not gapless(hi):
        return seed, None
    for _ in range(8):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if gapless(mid) else (mid, hi)
    return along(seed, hi), hi


def jittered(k):
    rng = np.random.default_rng(k)

    def jac(*a, **kw):
        J = analytic(*a, **kw)
        return J * (1.0 + 1.0e-6 * rng.standard_normal(J.shape))
    return jac


def mark(p):
    if p.converged and p.gapless and p.pattern_realised == "CFL":
        return "G"
    return ("c" if p.converged else ".") + ("" if p.pattern_realised == "CFL"
                                             else p.pattern_realised)


for n_B in densities:
    ref = solve_pattern(par, "fixed_YC", n_B, 0.0, flags, "unpaired",
                        spec=spec, backend="fast", pair_nodes_per_panel=H)
    seed0 = seed_from(ref, par, spec, "CFL")
    x0, t_edge = edge_seed(seed0, "fast")
    fast = []
    for jac in [analytic] + [jittered(k) for k in range(4)]:
        S.residual_jacobian = jac
        p = solve_pattern(par, "fixed_YC", n_B, 0.0, flags, "CFL", spec=spec,
                          x0=x0, backend="fast", pair_nodes_per_panel=H,
                          cross_seeded=True)
        S.residual_jacobian = analytic
        fast.append(mark(p))
    x0r, t_edge_r = edge_seed(seed0, "reference")
    refm = []
    rng = np.random.default_rng(7)
    for k in range(3):
        xk = x0r if k == 0 else x0r * (1.0 + 1.0e-9 * rng.standard_normal(x0r.shape))
        p = solve_pattern(par, "fixed_YC", n_B, 0.0, flags, "CFL", spec=spec,
                          x0=xk, backend="reference", pair_nodes_per_panel=H,
                          cross_seeded=True)
        refm.append(mark(p))
    print(f"arm {H}/{V} n_B {n_B:.3f}: edge fast {t_edge:.1f} ref {t_edge_r:.1f}"
          f"  fast {fast.count('G')}/5 [{' '.join(fast)}]  reference "
          f"{refm.count('G')}/3 [{' '.join(refm)}]", flush=True)
