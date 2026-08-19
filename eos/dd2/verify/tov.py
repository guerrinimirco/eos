"""
verify/tov.py
====================
TOV cross-check for the DD2 engine: feed the cold
beta-equilibrium EoS to the repo's TOV solver and verify M_max >= 2 M_sun,
plus report R_1.4.

Builds the core EoS from a warm-started octet beta-eq sweep, attaches the BPS
crust below the transition, and runs compute_tov_sequence.
"""
import os

import numpy as np

from eos.astro.tov.solver import (
    EOSTable_for_TOV, compute_tov_sequence, find_mmax_precise,
    generate_ec_logspace, CRUST_PATHS,
)
from eos.dd2.solver import sweep_beta_eq_octet


#: Crust–core transition density [fm^-3] (BPS table tops out at 0.08).
N_TRANSITION = 0.08


def build_core_table(par, flags, n_lo=0.05, n_hi=1.25, n_points=150):
    """
    Cold (T=0) beta-equilibrium core EoS as an EOSTable_for_TOV. Uses a
    geometric density grid and warm-started continuation through the onsets.
    """
    grid = np.geomspace(n_lo, n_hi, n_points)
    # stop_at_boundary: a Δ model may hit scalar collapse (m*->0) before n_hi;
    # take the valid prefix as the core EoS.
    points = sweep_beta_eq_octet(par, grid, flags, T=0.0,
                                 include_photons=False, stop_at_boundary=True)
    P = np.array([p.P for p in points])
    eps = np.array([p.eps for p in points])
    nB = np.array([p.n_B for p in points])
    # TOV interpolation needs a monotone-increasing P grid.
    order = np.argsort(P)
    return EOSTable_for_TOV(P=P[order], epsilon=eps[order], nB=nB[order])


def mass_radius(par, flags, crust="BPS", n_ec=180,
                e_c_min=150.0, e_c_max=3000.0):
    """
    Run the TOV sequence and return a summary dict:
        M_max [M_sun], R_Mmax [km], R_1.4 [km], e_c_max, and the raw results.

    crust: 'BPS' (default, if the file is present) or 'No'.
    """
    core = build_core_table(par, flags)
    if crust == "BPS" and not os.path.isfile(CRUST_PATHS.get("BPS", "")):
        crust = "No"
    e_c_vec = generate_ec_logspace(e_c_min, e_c_max, n_ec)
    results = compute_tov_sequence(
        core, e_c_vec, add_crust_table=crust, add_crust_mode="attach",
        n_transition=(N_TRANSITION if crust != "No" else None),
        compute_baryonic_mass=False, compute_tidal=False,
        verbose=False, backend="scipy",
    )
    idx_max, e_c_max, M_max = find_mmax_precise(results)

    # Stable branch = up to M_max; interpolate R at M = 1.4 there.
    M = results[:idx_max + 1, 4]
    R = results[:idx_max + 1, 3]
    R_14 = float(np.interp(1.4, M, R)) if M[-1] >= 1.4 > M[0] else float("nan")
    R_Mmax = float(results[idx_max, 3])
    return dict(M_max=float(M_max), R_Mmax=R_Mmax, R_1p4=R_14,
                e_c_max=float(e_c_max), results=results)
