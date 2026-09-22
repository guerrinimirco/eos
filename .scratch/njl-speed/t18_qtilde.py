"""Ticket 18: seeds displaced along the CFL rotated charge Q~.

The CFL cross seed's scaled Jacobian is exactly singular along
    (d mu_C, d mu_3, d mu_8) = t (1, -1, -1/2),
which shifts the nine modes by their rotated charge (u: 0,+1,+1; d, s:
-1,0,0 over r,g,b): every CFL pair is Q~-neutral, so a gapped CFL state is a
Q~ insulator and the residual is flat along it at T = 0. The gapless root
lies a finite t away. Scan t (sign of Y_C positive; one negative control):
the fast bounded path over five Jacobian draws, the reference backend's full
ladder once (it is deterministic), both at the given node rule.

    PYTHONPATH=. python3 .scratch/njl-speed/t18_qtilde.py NODES BACKENDS T...
"""
import sys

import numpy as np

import eos.njl.solver as S
from eos import njl
from eos.njl.solver import mode_spec, seed_from, solve_pattern, unknown_slots

par = njl.Parameters.named("rg_njl1")
flags = njl.SpeciesFlags(csc=True)
spec = mode_spec("fixed_YC", leptons=True, Y_C=0.1)
names = unknown_slots(par, spec, "CFL")
ix = {n: i for i, n in enumerate(names)}
analytic = S.residual_jacobian
nodes = int(sys.argv[1])
backends = sys.argv[2].split(",")
ts = [float(t) for t in sys.argv[3:]]


def jittered(k):
    rng = np.random.default_rng(k)

    def jac(*a, **kw):
        J = analytic(*a, **kw)
        return J * (1.0 + 1.0e-6 * rng.standard_normal(J.shape))
    return jac


def displaced(seed, t):
    x = seed.copy()
    x[ix["mu_C"]] += t
    x[ix["mu_3"]] -= t
    x[ix["mu_8"]] -= 0.5 * t
    return x


def mark(p):
    if p.converged and p.gapless and p.pattern_realised == "CFL":
        return "G"
    return ("c" if p.converged else ".") + ("" if p.pattern_realised == "CFL"
                                             else p.pattern_realised)


draws = [analytic] + [jittered(k) for k in range(4)]
for n_B in (0.775, 0.800, 0.900, 0.925, 0.950, 1.000, 1.075):
    ref = solve_pattern(par, "fixed_YC", n_B, 0.0, flags, "unpaired",
                        spec=spec, backend="fast", pair_nodes_per_panel=nodes)
    seed = seed_from(ref, par, spec, "CFL")
    for t in ts:
        x0 = displaced(seed, t)
        line = f"n_B {n_B:.3f} nodes {nodes} t {t:+6.0f}:"
        if "fast" in backends:
            marks = []
            for jac in draws:
                S.residual_jacobian = jac
                p = solve_pattern(par, "fixed_YC", n_B, 0.0, flags, "CFL",
                                  spec=spec, x0=x0, backend="fast",
                                  pair_nodes_per_panel=nodes,
                                  cross_seeded=True)
                S.residual_jacobian = analytic
                marks.append(mark(p))
            line += (f"  fast {sum(m == 'G' for m in marks)}/5 "
                     f"[{' '.join(marks)}]")
        if "reference" in backends:
            p = solve_pattern(par, "fixed_YC", n_B, 0.0, flags, "CFL",
                              spec=spec, x0=x0, backend="reference",
                              pair_nodes_per_panel=nodes, cross_seeded=True)
            line += f"  reference {mark(p)} f {p.f:.3f}"
        print(line, flush=True)
