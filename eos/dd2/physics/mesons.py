"""DD2's couplings for the thermal meson gas.

The gas itself — species list, quantum numbers, Bose sums — lives in
:mod:`eos.general.thermal_mesons` (one implementation for every model, after
Lavagno, Phys. Rev. C 81, 044909 (2010); see also arXiv:1210.0400). What is
DD2's is the arithmetic of the three independent effective potentials, using
its DENSITY-DEPENDENT couplings Gamma_iN(n_B) in place of constant ones:

    mu*_pi+ = mu_Q - Gamma_rhoN(n_B) rho0
    mu*_K+  = mu_Q - mu_S - [Gamma_omegaN - Gamma_omegaLambda](n_B) omega0
                          - 1/2 Gamma_rhoN(n_B) rho0
    mu*_K0  =        - mu_S - [Gamma_omegaN - Gamma_omegaLambda](n_B) omega0
                          + 1/2 Gamma_rhoN(n_B) rho0

The kaon's omega shift is (Gamma_omegaN - Gamma_omegaLambda): under the
additive quark picture the kaon couples through its one light quark, and the
Lambda ratio supplies the strange-sector piece — SU(6)'s 2/3 unless the
parametrization carries an explicit Lambda coupling. No rearrangement term
enters any mu*_j: the gas is a spectator to Sigma^R.

The gas contributes charge and strangeness to the solver's constraints
(charge neutrality and fixed Y_C / Y_S count baryons + mesons together), no
baryon number, and no field sources.
"""
from eos.general import thermal_mesons as _gas

#: Lambda SU(6) omega ratio for the kaon omega-shift (Gamma_omegaL = x * Gamma_omegaN).
_X_OMEGA_LAMBDA = 2.0 / 3.0


def lambda_omega_ratio(par):
    """x_omega^Lambda entering the kaon omega-shift (SU(6) 2/3 without hyperons)."""
    # map row: (mass, x_sigma, x_omega, x_rho, x_phi) -> x_omega at [2]
    return (par.hyperon_coupling_map.get("Lambda", (0, 0, _X_OMEGA_LAMBDA))[2]
            if par.hyperon_couplings else _X_OMEGA_LAMBDA)


def meson_potentials(Gw, Gr, x_omega_L, mu_Q, mu_S, omega0, rho0):
    """(mu*_pi+, mu*_K+, mu*_K0) [MeV] from DD2's couplings and fields.

    Gw, Gr are the NUCLEON couplings Gamma_{omega,rho}N(n_B) and x_omega_L
    the Lambda/nucleon omega ratio; potentials and fields in MeV.
    """
    dGw_KL = (1.0 - x_omega_L) * Gw          # (Gamma_omegaN - Gamma_omegaL)
    mu_pi = mu_Q - Gr * rho0
    mu_Kp = mu_Q - mu_S - dGw_KL * omega0 - 0.5 * Gr * rho0
    mu_K0 = -mu_S - dGw_KL * omega0 + 0.5 * Gr * rho0
    return mu_pi, mu_Kp, mu_K0


def meson_families(Gw, Gr, x_omega_L, mu_Q, mu_S, omega0, rho0,
                   include_pseudoscalars=False, include_thermal_vectors=False):
    """(mu_eff, mass, Q, S, g) per thermal meson species, at DD2's potentials."""
    mu_pi, mu_Kp, mu_K0 = meson_potentials(Gw, Gr, x_omega_L, mu_Q, mu_S,
                                           omega0, rho0)
    return _gas.meson_families(mu_pi, mu_Kp, mu_K0,
                               include_pseudoscalars, include_thermal_vectors)


def thermal_meson_charges(Gw, Gr, x_omega_L, mu_Q, mu_S, omega0, rho0, T,
                          include_pseudoscalars=False,
                          include_thermal_vectors=False):
    """(n_C, n_S) of the gas [fm^-3] — what the solver's constraints need."""
    if T <= 0.0 or not (include_pseudoscalars or include_thermal_vectors):
        return 0.0, 0.0
    mu_pi, mu_Kp, mu_K0 = meson_potentials(Gw, Gr, x_omega_L, mu_Q, mu_S,
                                           omega0, rho0)
    return _gas.thermal_meson_charges(mu_pi, mu_Kp, mu_K0, T,
                                      include_pseudoscalars,
                                      include_thermal_vectors)


def thermal_meson_thermo(par, n_B, mu_Q, mu_S, omega0, rho0, T,
                         include_pseudoscalars=False,
                         include_thermal_vectors=False):
    """Full gas thermodynamics at (n_B [fm^-3], T [MeV]) on DD2's mean field.

    Returns the dict of :func:`eos.general.thermal_mesons.thermal_meson_thermo`
    (fm-based units), with the couplings evaluated at this n_B.
    """
    if T <= 0.0 or not (include_pseudoscalars or include_thermal_vectors):
        return dict(P=0.0, e=0.0, s=0.0, n_C=0.0, n_S=0.0, mu_dot_n=0.0)
    _, Gw, Gr, _, _, _ = par.couplings_at(n_B)
    mu_pi, mu_Kp, mu_K0 = meson_potentials(Gw, Gr, lambda_omega_ratio(par),
                                           mu_Q, mu_S, omega0, rho0)
    return _gas.thermal_meson_thermo(mu_pi, mu_Kp, mu_K0, T,
                                     include_pseudoscalars,
                                     include_thermal_vectors)
