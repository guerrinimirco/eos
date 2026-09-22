"""Ticket 18: which cross seed puts the CFL candidate robustly in the gapless
root's basin? The current seed sits ON the (u<->d, r<->g) plane
Delta_1 = Delta_2, mu_3 = 0; the bounded stall is the root's mirror image.

Each variant is scored over five draws of the bounded path (Newton + hybrj):
the analytic Jacobian and four copies jittered at 1e-6 relative, so a basin
boundary shows up as a split rather than as one lucky or unlucky ticket.

    PYTHONPATH=. python3 .scratch/njl-speed/t18_seeds.py NODES
"""
import sys

import numpy as np

import eos.njl.solver as S
from eos import njl
from eos.njl.solver import (mode_spec, seed_from, solve_pattern,
                            unknown_slots, _unpack)

par = njl.Parameters.named("rg_njl1")
flags = njl.SpeciesFlags(csc=True)
spec = mode_spec("fixed_YC", leptons=True, Y_C=0.1)
names = unknown_slots(par, spec, "CFL")
ix = {n: i for i, n in enumerate(names)}
analytic = S.residual_jacobian
nodes = int(sys.argv[1])


def jittered(k):
    rng = np.random.default_rng(k)

    def jac(*a, **kw):
        J = analytic(*a, **kw)
        return J * (1.0 + 1.0e-6 * rng.standard_normal(J.shape))
    return jac


def variants(ref, seed):
    mu_C_unp = _unpack(ref.x, par, spec, "unpaired")[6]
    s = seed[ix["Delta_3"]]
    out = {"V0 current": seed.copy()}
    v = seed.copy(); v[ix["mu_C"]] = mu_C_unp
    out["V1 unpaired mu_C"] = v
    v = seed.copy(); m = 0.5 * (v[0] + v[1]); v[0] = v[1] = m
    out["V4 M_u=M_d"] = v
    for side, sign in (("+", 1.0), ("-", -1.0)):
        v = seed.copy()
        v[ix["mu_3"]] = -sign * 0.1 * s
        v[ix["Delta_1"]] = s * (1.0 + sign * 0.05)
        v[ix["Delta_2"]] = s * (1.0 - sign * 0.05)
        out[f"V2{side} off-plane, root side" if sign > 0 else
            f"V2{side} off-plane, mirror side"] = v
    return out


draws = [("analytic", analytic)] + [(f"jit{k}", jittered(k)) for k in range(4)]
for n_B in (0.775, 0.800, 0.900, 1.000, 1.075):
    ref = solve_pattern(par, "fixed_YC", n_B, 0.0, flags, "unpaired",
                        spec=spec, backend="fast", pair_nodes_per_panel=nodes)
    seed = seed_from(ref, par, spec, "CFL")
    for label, x0 in variants(ref, seed).items():
        hits, marks = 0, ""
        for _, jac in draws:
            S.residual_jacobian = jac
            p = solve_pattern(par, "fixed_YC", n_B, 0.0, flags, "CFL",
                              spec=spec, x0=x0, backend="fast",
                              pair_nodes_per_panel=nodes, cross_seeded=True)
            S.residual_jacobian = analytic
            good = bool(p.converged and p.gapless
                        and p.pattern_realised == "CFL")
            hits += good
            marks += ("G" if good else
                      ("c" if p.converged else ".")
                      + ("" if p.pattern_realised == "CFL" else
                         p.pattern_realised))
            marks += " "
        print(f"n_B {n_B:.3f} nodes {nodes} {label:28s}: {hits}/5  "
              f"[{marks.strip()}]", flush=True)
