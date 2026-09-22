"""Ticket 18, the control for t18_jac_path.py's part B: the bounded path
(Newton + hybrj) from the unpaired cross seed with the analytic Jacobian
REPLACED by a central difference at several steps. The 1e-4 step agrees with
the analytic Jacobian to 1e-7 on the whole path; the 1e-6 step is noisy at the
symmetric seed. If only the noisy ones reach the gapless root, the analytic
Jacobian is not what fails -- the seed sits on a basin boundary and any
perturbation of the first step picks the root.

    PYTHONPATH=. python3 .scratch/njl-speed/t18_substitution.py NODES
"""
import sys

import numpy as np

import eos.njl.solver as S
from eos import njl
from eos.njl.solver import mode_spec, residual, seed_from, solve_pattern
from eos.njl.thermodynamics import vacuum_solution

par = njl.Parameters.named("rg_njl1")
flags = njl.SpeciesFlags(csc=True)
spec = mode_spec("fixed_YC", leptons=True, Y_C=0.1)
vac = vacuum_solution(par)
analytic = S.residual_jacobian
nodes = int(sys.argv[1]) if len(sys.argv) > 1 else 24


def central(rel):
    def jac(x, names, par_, flags_, spec_, pattern, n_B, T, vac_, nodes_,
            state=None):
        x = np.asarray(x, dtype=float)
        F = np.zeros((len(x), len(x)))
        for i in range(len(x)):
            h = rel * max(abs(x[i]), 1.0)
            up, dn = x.copy(), x.copy()
            up[i] += h
            dn[i] -= h
            F[:, i] = (np.array(residual(up, par, flags, spec, pattern, n_B,
                                         T, vac, backend="fast",
                                         pair_nodes_per_panel=nodes_))
                       - np.array(residual(dn, par, flags, spec, pattern, n_B,
                                           T, vac, backend="fast",
                                           pair_nodes_per_panel=nodes_))
                       ) / (2.0 * h)
        return F
    return jac


def jittered(scale, seed=0):
    rng = np.random.default_rng(seed)

    def jac(*a, **k):
        J = analytic(*a, **k)
        return J * (1.0 + scale * rng.standard_normal(J.shape))
    return jac


arms = [("analytic", analytic)]
arms += [(f"central {rel:.0e}", central(rel)) for rel in (1e-3, 1e-4, 1e-5,
                                                          1e-6, 1e-7)]
arms += [(f"analytic x(1+{s:.0e} N)", jittered(s)) for s in (1e-6, 1e-3)]
for n_B in (0.800, 0.900, 1.000):
    ref = solve_pattern(par, "fixed_YC", n_B, 0.0, flags, "unpaired",
                        spec=spec, backend="fast", pair_nodes_per_panel=nodes)
    seed = seed_from(ref, par, spec, "CFL")
    for label, jac in arms:
        S.residual_jacobian = jac
        p = solve_pattern(par, "fixed_YC", n_B, 0.0, flags, "CFL", spec=spec,
                          x0=seed, backend="fast", pair_nodes_per_panel=nodes,
                          cross_seeded=True)
        S.residual_jacobian = analytic
        print(f"n_B {n_B:.3f} nodes {nodes} {label:24s}: converged "
              f"{p.converged!s:5s} err {p.error:.1e} realised "
              f"{p.pattern_realised:4s} gapless {p.gapless!s:5s} "
              f"eps {p.eps:10.3f}", flush=True)
