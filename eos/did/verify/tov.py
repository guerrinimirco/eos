"""TOV cross-check for DID: the cold beta-equilibrium EoS through the
repository's stellar-structure solver.

Table VIII of arXiv:2511.15646 reports M_max = 2.245 M_sun for DID and 2.196
for DIDY, with R_1.4 = 11.99 km for both. Those numbers were computed with the
paper's own HS (nuclear-statistical-equilibrium) crust, which this repository
does not carry; the BPS crust used here differs from it below 0.08 fm^-3, and
that difference moves R_1.4 by a few hundredths of a km and M_max by a few
thousandths. The gates in `run_full_check` are set accordingly -- loose, and
loose for a reason that is stated rather than because the model disagrees.
"""
import os

import numpy as np

from eos.astro.tov.solver import (
    CRUST_PATHS, EOSTable_for_TOV, compute_tov_sequence, find_mmax_precise,
    generate_ec_logspace,
)
from eos.did.solver import solve_beta_eq_neutrinoless, warm_start
from eos.general.modes import beta_eq_neutrinoless

#: Crust-core transition density [fm^-3] (the BPS table tops out at 0.08).
N_TRANSITION = 0.08


def build_core_table(par, flags, n_lo=0.05, n_hi=1.4, n_points=200):
    """The cold beta-equilibrium core EoS as an `EOSTable_for_TOV`.

    A geometric density grid swept with a warm start, which is what carries
    the solve through the hyperon onsets; a density the solver cannot reach is
    dropped rather than ending the sweep.
    """
    spec = beta_eq_neutrinoless()
    P, eps, n_B, x0 = [], [], [], None
    for n in np.geomspace(n_lo, n_hi, n_points):
        point = solve_beta_eq_neutrinoless(par, float(n), flags, T=0.0, x0=x0)
        if not point.converged:
            x0 = None
            continue
        P.append(point.P)
        eps.append(point.eps)
        n_B.append(point.n_B)
        x0 = warm_start(point, spec)
    P, eps, n_B = np.array(P), np.array(eps), np.array(n_B)
    order = np.argsort(P)          # TOV interpolation needs P increasing
    return EOSTable_for_TOV(P=P[order], epsilon=eps[order], nB=n_B[order])


def mass_radius(par, flags, crust="BPS", n_ec=180, e_c_min=150.0,
                e_c_max=3000.0):
    """The TOV sequence, as {M_max, R_Mmax, R_1p4, e_c_max, results}."""
    core = build_core_table(par, flags)
    if crust == "BPS" and not os.path.isfile(CRUST_PATHS.get("BPS", "")):
        crust = "No"
    results = compute_tov_sequence(
        core, generate_ec_logspace(e_c_min, e_c_max, n_ec),
        add_crust_table=crust, add_crust_mode="attach",
        n_transition=(N_TRANSITION if crust != "No" else None),
        compute_baryonic_mass=False, compute_tidal=False,
        verbose=False, backend="scipy")
    idx_max, e_c_star, M_max = find_mmax_precise(results)

    M = results[:idx_max + 1, 4]           # the stable branch
    R = results[:idx_max + 1, 3]
    R_14 = float(np.interp(1.4, M, R)) if M[-1] >= 1.4 > M[0] else float("nan")
    return dict(M_max=float(M_max), R_Mmax=float(results[idx_max, 3]),
                R_1p4=R_14, e_c_max=float(e_c_star), results=results,
                crust=crust)


if __name__ == "__main__":
    from eos.did import Parameters, SpeciesFlags

    par = Parameters.default()
    for label, flags in (("DID ", SpeciesFlags(muons=False)),
                         ("DIDY", SpeciesFlags(hyperons=True, muons=False))):
        out = mass_radius(par, flags)
        print(f"{label}: M_max = {out['M_max']:.3f} M_sun, "
              f"R_Mmax = {out['R_Mmax']:.2f} km, R_1.4 = {out['R_1p4']:.2f} km "
              f"({out['crust']} crust)")
    print("paper Table VIII: DID 2.245 / 10.87 / 11.99, "
          "DIDY 2.196 / 10.91 / 11.99")
