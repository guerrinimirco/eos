"""Ticket 10: does the RG vacuum half need the SAME panel rule?

The Lambda_UV vacuum pass stops its geometric panels at Lambda_UV / 2^7 = 47 MeV,
so in 2SC at high density, where M_u ~ 10 MeV, the lowest panel holds the
mass scale and N = 8 leaves delta_rho_s at 1.6e-8 (t10_quad_accuracy.log). Here
the vacuum rule gains breakpoints at the three constituent masses, and each
layout is compared with its own N = 64 value, and with the SHIPPED rule at 64.

    PYTHONPATH=. python3 t10_vac_layout.py
"""
import numpy as np

from eos import njl
from eos.general.fermi_integrals import _gauss_legendre
from eos.general.pairing import pair_block
from eos.njl.thermodynamics import RG_PANEL_RATIO

par = njl.Parameters.named("rg_njl1")
FLAGS = njl.SpeciesFlags(csc=True)
zero = np.zeros(9)


def rule_on(edges, N):
    x, w = _gauss_legendre(N)
    edges = np.unique(np.asarray(edges, dtype=float))
    lo, hi = edges[:-1, None], edges[1:, None]
    half = 0.5 * (hi - lo)
    return ((0.5 * (lo + hi) + half * x[None, :]).ravel(),
            (half * w[None, :]).ravel())


def vac(M, D, k_max, N, breaks):
    # the SHIPPED vacuum layout -- 0, k_max / 2^7 ... k_max / 2, k_max, which
    # is what panel_nodes builds at mu* = 0 with ratio 2 -- plus `breaks`.
    # (panel_nodes itself cannot add them: a breakpoint raises the floor of
    # its geometric tail, which would delete the panels below it.)
    shipped = [0.0, k_max] + [k_max / RG_PANEL_RATIO ** j for j in range(1, 8)]
    rule = rule_on(shipped + [b for b in breaks if 0.0 < b < k_max], N)
    b = pair_block(M, zero, D, 0.0, k_max, backend="fast", quadrature=rule)
    return b, rule[0].size


for pattern, n_B in (("2SC", 1.30), ("2SC", 0.90), ("CFL", 1.149)):
    st = njl.eos_point(par, "beta_eq_neutrinoless", FLAGS, n_B=n_B, T=0.0,
                       patterns=(pattern,), backend="fast").point._state
    M, D = np.array(st.M), np.array(st.Delta)
    print(f"{pattern} n_B={n_B}  M={np.round(M, 1)}")
    for k_max, label in ((par.Lambda_medium, "vac(Lambda_UV)"),
                         (par.Lambda, "vac(Lambda)   ")):
        shipped64, _ = vac(M, D, k_max, 64, [])
        for breaks, name in (([], "shipped      "), (list(M), "+ mass breaks")):
            ref, _ = vac(M, D, k_max, 64, breaks)
            for N in (6, 8, 10, 12, 24):
                b, nodes = vac(M, D, k_max, N, breaks)
                e = [float(np.max(np.abs(getattr(b, f) - getattr(ref, f)))
                           / np.max(np.abs(getattr(ref, f))))
                     for f in ("delta_rho_s", "gap_kernel")]
                s = float(np.max(np.abs(b.delta_rho_s - shipped64.delta_rho_s))
                          / np.max(np.abs(shipped64.delta_rho_s)))
                print(f"  {label} {name} N={N:2d} ({nodes:3d} nodes): rho_s "
                      f"{e[0]:.1e}  gap {e[1]:.1e}   rho_s vs shipped@64 {s:.1e}")
