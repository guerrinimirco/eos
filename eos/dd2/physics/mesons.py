"""
physics/mesons.py
====================
Thermal meson gas on top of the baryonic mean field (report §A, Lavagno 2010).

Pseudoscalar and vector nonets treated as one-body ideal Bose gases with
effective chemical potentials tied to the SAME meson mean fields that generate
the baryon interactions (report §A.3), using DD2's DENSITY-DEPENDENT couplings
Gamma_iN(n_B) in place of the constant SFHo ones:

    mu*_pi+ = mu*_rho+ = mu_Q - Gamma_rhoN(n_B) rho0
    mu*_K+  = mu*_K*+  = mu_Q - mu_S - (Gamma_omegaN - Gamma_omegaL)(n_B) omega0
                                     - 1/2 Gamma_rhoN(n_B) rho0
    mu*_K0  = mu*_K*0  =        - mu_S - (Gamma_omegaN - Gamma_omegaL)(n_B) omega0
                                     + 1/2 Gamma_rhoN(n_B) rho0
    mu*_{j-} = -mu*_{j+};   mu*_j = 0 for strangeless neutral (pi0, eta, eta',
                                       rho0, omega, phi).

No rearrangement term enters mu*_j — the thermal mesons are spectators to
Sigma^R (report §A.3). Bose integrals use the |mu*| <= m clamp built into
bose_integrals. Off by default for cold NS (report §A.4); the thermal
rho/omega/phi are the incoherent k!=0 quanta on top of the mean field, a
modelling choice kept switchable.

Masses (report §A.4 / PDG, MeV).
"""
from eos.general.bose_integrals import solve_bose_jel

M_PI, M_K, M_ETA, M_ETAP = 140.0, 494.0, 547.0, 958.0
M_KSTAR, M_RHO, M_OMEGA, M_PHI = 892.0, 771.0, 782.0, 1020.0

#: Lambda SU(6) omega ratio for the kaon omega-shift (Gamma_omegaL = x * Gamma_omegaN).
_X_OMEGA_LAMBDA = 2.0 / 3.0


def _bose(mu_eff, T, m, g):
    """(n, P, e, s) fm-based; antiparticles are separate species here."""
    n, P, e, s, _ = solve_bose_jel(mu_eff, T, m, g, include_antiparticles=False)
    return n, P, e, s


def thermal_meson_thermo(par, n_B, mu_Q, mu_S, omega0, rho0, T,
                         include_pseudoscalars=False,
                         include_thermal_vectors=False):
    """
    Thermal meson gas contribution at (n_B [fm^-3], T [MeV]) given the baryonic
    charge/strangeness potentials and vector fields.

    Returns a dict (fm-based units, like photon_thermo): P, e, s [MeV/fm^3 or
    fm^-3], n_C, n_S [fm^-3] (net meson charge / strangeness), and
    mu_dot_n = sum_j mu*_j n_j [MeV/fm^3] for the HVH bookkeeping (the Bose gas
    Euler relation e + P = T s + mu* n holds per species).
    """
    P = e = s = n_C = n_S = mudotn = 0.0
    if T <= 0.0 or not (include_pseudoscalars or include_thermal_vectors):
        return dict(P=0.0, e=0.0, s=0.0, n_C=0.0, n_S=0.0, mu_dot_n=0.0)

    _, Gw, Gr, _, _, _ = par.couplings_at(n_B)
    x_omega_L = par.hyperon_coupling_map.get("Lambda", (0, _X_OMEGA_LAMBDA))[1] \
        if par.hyperon_couplings else _X_OMEGA_LAMBDA
    dGw_KL = (1.0 - x_omega_L) * Gw          # (Gamma_omegaN - Gamma_omegaL)

    mu_pi = mu_Q - Gr * rho0                              # mu*_pi+ = mu*_rho+
    mu_Kp = mu_Q - mu_S - dGw_KL * omega0 - 0.5 * Gr * rho0
    mu_K0 = -mu_S - dGw_KL * omega0 + 0.5 * Gr * rho0

    # (mu_eff, mass, Q, S) per species; antiparticles listed explicitly.
    families = []
    if include_pseudoscalars:
        families += [
            (mu_pi, M_PI, +1, 0, 1.0), (-mu_pi, M_PI, -1, 0, 1.0),
            (0.0, M_PI, 0, 0, 1.0),                          # pi0
            (mu_Kp, M_K, +1, -1, 1.0), (-mu_Kp, M_K, -1, +1, 1.0),   # K+, K-
            (mu_K0, M_K, 0, -1, 1.0), (-mu_K0, M_K, 0, +1, 1.0),     # K0, K0bar
            (0.0, M_ETA, 0, 0, 1.0), (0.0, M_ETAP, 0, 0, 1.0),
        ]
    if include_thermal_vectors:
        families += [
            (mu_pi, M_RHO, +1, 0, 3.0), (-mu_pi, M_RHO, -1, 0, 3.0),
            (0.0, M_RHO, 0, 0, 3.0), (0.0, M_OMEGA, 0, 0, 3.0),
            (mu_Kp, M_KSTAR, +1, -1, 3.0), (-mu_Kp, M_KSTAR, -1, +1, 3.0),
            (mu_K0, M_KSTAR, 0, -1, 3.0), (-mu_K0, M_KSTAR, 0, +1, 3.0),
            (0.0, M_PHI, 0, 0, 3.0),
        ]

    for mu_eff, m, Q, S, g in families:
        n, P_j, e_j, s_j = _bose(mu_eff, T, m, g)
        P += P_j
        e += e_j
        s += s_j
        n_C += Q * n
        n_S += S * n
        mudotn += mu_eff * n
    return dict(P=P, e=e, s=s, n_C=n_C, n_S=n_S, mu_dot_n=mudotn)
