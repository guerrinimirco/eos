"""Ticket 18's one measurement: is the analytic Jacobian poor along the path
from the unpaired cross seed into the gapless CFL root?

fixed_YC, Y_C = 0.1, leptons, T = 0, rg_njl1, csc, backend 'fast'.

A. Every Jacobian the BOUNDED path (Newton + hybrj) asks for is recorded and
   compared with a central difference of the raw residual at two steps, row by
   row as `verify.check_jacobian_parity` does.
B. The bounded path again with the analytic Jacobian REPLACED by that central
   difference. If it now reaches the gapless root, the Jacobian is what fails.
C. Analytic vs central difference on the straight line seed -> gapless root.

    PYTHONPATH=. python3 .scratch/njl-speed/t18_jac_path.py [nodes ...]
"""
import sys

import numpy as np

import eos.njl.solver as S
from eos import njl
from eos.general.pairing import gapless_breakpoints
from eos.njl.solver import (_state, mode_spec, residual, seed_from,
                            solve_pattern, unknown_slots)
from eos.njl.thermodynamics import vacuum_solution

par = njl.Parameters.named("rg_njl1")
flags = njl.SpeciesFlags(csc=True)
spec = mode_spec("fixed_YC", leptons=True, Y_C=0.1)
vac = vacuum_solution(par)
names = unknown_slots(par, spec, "CFL")
analytic = S.residual_jacobian
RECORD = []


def raw(x, n_B, nodes):
    return np.array(residual(x, par, flags, spec, "CFL", n_B, 0.0, vac,
                             backend="fast", pair_nodes_per_panel=nodes))


def central(x, n_B, nodes, rel):
    F = np.zeros((len(x), len(x)))
    for i in range(len(x)):
        h = rel * max(abs(x[i]), 1.0)
        up, dn = x.copy(), x.copy()
        up[i] += h
        dn[i] -= h
        F[:, i] = (raw(up, n_B, nodes) - raw(dn, n_B, nodes)) / (2.0 * h)
    return F


def mismatch(J, F):
    """worst row-scaled |J - F| over live rows, and where."""
    row_scale = np.abs(F).max(axis=1) + 1.0e-300
    err = np.abs(J - F) / row_scale[:, None]
    alive = np.abs(F).max(axis=1) > 1.0e-6 * np.abs(F).max()
    err[~alive] = 0.0
    r, c = np.unravel_index(np.argmax(err), err.shape)
    return float(err[r, c]), r, names[c]


def describe(x, n_B, nodes):
    st = _state(x, par, spec, "CFL", 0.0, vac, False, "fast", nodes)
    M = x[:3]
    k = gapless_breakpoints(M, st.mu_star, x[3:6], par.Lambda_medium)
    return st.gapless, len(np.unique(np.round(k, 6)))


def recording(x, *args, **kwargs):
    J = analytic(x, *args, **kwargs)
    RECORD.append((np.array(x, dtype=float), J.copy()))
    return J


def differenced(x, names_, par_, flags_, spec_, pattern, n_B, T, vac_, nodes,
                state=None):
    return central(np.asarray(x, dtype=float), n_B, nodes, 1.0e-6)


row_names = (["mass_u", "mass_d", "mass_s", "gap_1", "gap_2", "gap_3",
              "colour_3", "colour_8"] + (["Sigma_V"] if "Sigma_V" in names
                                        else []) + ["n_B", "charge"])
node_arms = [int(a) for a in sys.argv[1:]] or [24, 12]
for nodes in node_arms:
    for n_B in (0.800, 0.900, 1.000):
        ref = solve_pattern(par, "fixed_YC", n_B, 0.0, flags, "unpaired",
                            spec=spec, backend="fast",
                            pair_nodes_per_panel=nodes)
        seed = seed_from(ref, par, spec, "CFL")
        kw = dict(spec=spec, x0=seed, backend="fast",
                  pair_nodes_per_panel=nodes)

        # A: the bounded path, every Jacobian recorded
        RECORD.clear()
        S.residual_jacobian = recording
        p_an = solve_pattern(par, "fixed_YC", n_B, 0.0, flags, "CFL",
                             cross_seeded=True, **kw)
        S.residual_jacobian = analytic
        print(f"\n=== n_B {n_B:.3f} nodes {nodes}: bounded, analytic J -> "
              f"converged {p_an.converged} err {p_an.error:.1e} gapless "
              f"{p_an.gapless} eps {p_an.eps:.3f}; {len(RECORD)} Jacobians",
              flush=True)
        worst_path = 0.0
        for i, (x, J) in enumerate(RECORD):
            F4 = central(x, n_B, nodes, 1.0e-4)
            F6 = central(x, n_B, nodes, 1.0e-6)
            e4, r4, c4 = mismatch(J, F4)
            e6, r6, c6 = mismatch(J, F6)
            e46, _, _ = mismatch(F6, F4)
            gl, nk = describe(x, n_B, nodes)
            r = raw(x, n_B, nodes)
            worst_path = max(worst_path, min(e4, e6))
            print(f"  J#{i:2d} gaps ({x[3]:7.2f},{x[4]:7.2f},{x[5]:7.2f}) "
                  f"gapless {gl!s:5s} crossings {nk}: |J-FD| 1e-4 {e4:.1e} "
                  f"[{row_names[r4]}/{c4}]  1e-6 {e6:.1e} "
                  f"[{row_names[r6]}/{c6}]  FD4-FD6 {e46:.1e}", flush=True)

        # B: the same bounded path, the analytic Jacobian replaced
        S.residual_jacobian = differenced
        p_fd = solve_pattern(par, "fixed_YC", n_B, 0.0, flags, "CFL",
                             cross_seeded=True, **kw)
        S.residual_jacobian = analytic
        print(f"  B: bounded, central-difference J -> converged "
              f"{p_fd.converged} err {p_fd.error:.1e} gapless {p_fd.gapless} "
              f"eps {p_fd.eps:.3f}", flush=True)

        # C: the gapless root (full ladder), and the line seed -> root
        p_full = solve_pattern(par, "fixed_YC", n_B, 0.0, flags, "CFL",
                               cross_seeded=False, **kw)
        root = np.array(p_full.x, dtype=float)
        print(f"  full ladder -> converged {p_full.converged} gapless "
              f"{p_full.gapless} eps {p_full.eps:.3f}", flush=True)
        for t in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0):
            x = seed + t * (root - seed)
            J = analytic(x, names, par, flags, spec, "CFL", n_B, 0.0, vac,
                         nodes)
            F4 = central(x, n_B, nodes, 1.0e-4)
            F6 = central(x, n_B, nodes, 1.0e-6)
            e4, r4, c4 = mismatch(J, F4)
            e6, r6, c6 = mismatch(J, F6)
            e46, _, _ = mismatch(F6, F4)
            gl, nk = describe(x, n_B, nodes)
            print(f"  C t={t:4.2f} gapless {gl!s:5s} crossings {nk}: |J-FD| "
                  f"1e-4 {e4:.1e} [{row_names[r4]}/{c4}]  1e-6 {e6:.1e} "
                  f"[{row_names[r6]}/{c6}]  FD4-FD6 {e46:.1e}", flush=True)
