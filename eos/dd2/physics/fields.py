"""
physics/fields.py
====================
Mean-field sector of the DD2 model (report §1.4, §1.6): algebraic vector-field
elimination, rearrangement self-energy, and the field contributions to the
energy density and pressure.

Natural units: fields in MeV, densities in MeV^3, eps/P in MeV^4.
dG_* are dGamma/dn_B in MeV^-3 (from Parametrization.couplings_at).
"""


def vector_fields(par, Gw, Gr, nB_nat, n3_nat):
    """
    omega_0 and rho_0 [MeV] from the (algebraic) vector field equations.
    n3_nat = sum_i t3_i n_i is the tau_3 = ±1 weighted isovector density.
    """
    omega0 = Gw * nB_nat / par.m_omega ** 2
    rho0 = Gr * n3_nat / par.m_rho ** 2
    return omega0, rho0


def rearrangement(dGs, dGw, dGr, sigma, omega0, rho0, nB_nat, n3_nat, ns_nat):
    """
    Rearrangement self-energy Sigma^R [MeV] — identical for all baryons.
    Enters mu and P, never eps (thermodynamic consistency).
    """
    return dGw * omega0 * nB_nat + dGr * rho0 * n3_nat - dGs * sigma * ns_nat


def field_eps_P(par, sigma, omega0, rho0):
    """
    Meson mean-field contributions (eps_field, P_field) [MeV^4].
    Scalar enters P with minus, vectors with plus (report §1.6).
    """
    s2 = par.m_sigma ** 2 * sigma ** 2
    w2 = par.m_omega ** 2 * omega0 ** 2
    r2 = par.m_rho ** 2 * rho0 ** 2
    return 0.5 * (s2 + w2 + r2), 0.5 * (-s2 + w2 + r2)
