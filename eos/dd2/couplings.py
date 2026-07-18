"""
couplings.py
====================
DD2 density-dependent coupling functionals (Typel et al., PRC 81, 015803 (2010)).

Isoscalar mesons (sigma, omega), rational Typel–Wolter form:
    Gamma_i(n) = Gamma_i(n_sat) * f_i(x),   x = n/n_sat,
    f_i(x) = a_i * (1 + b_i (x+d_i)^2) / (1 + c_i (x+d_i)^2)

Isovector rho, exponential form:
    Gamma_rho(n) = Gamma_rho(n_sat) * exp[-a_rho (x-1)]

Internal constraints (used to derive dependent coefficients):
    f_i(1) = 1     =>  a_i = (1 + c_i (1+d_i)^2) / (1 + b_i (1+d_i)^2)
    f_i''(0) = 0   =>  d_i = 1 / sqrt(3 c_i)
"""
from eos.dd2.xp import xp


def rational_f(x, a, b, c, d):
    """f_i(x) for the isoscalar sigma/omega couplings."""
    u = (x + d) ** 2
    return a * (1.0 + b * u) / (1.0 + c * u)


def rational_df(x, a, b, c, d):
    """df_i/dx (exact closed form)."""
    return 2.0 * a * (b - c) * (x + d) / (1.0 + c * (x + d) ** 2) ** 2


def rational_d2f(x, a, b, c, d):
    """d2f_i/dx2 (exact closed form)."""
    u = (x + d) ** 2
    return 2.0 * a * (b - c) * (1.0 - 3.0 * c * u) / (1.0 + c * u) ** 3


def exponential_f(x, a_rho):
    """f_rho(x) for the isovector coupling."""
    return xp.exp(-a_rho * (x - 1.0))


def exponential_df(x, a_rho):
    """df_rho/dx."""
    return -a_rho * exponential_f(x, a_rho)


def derived_d(c):
    """Dependent coefficient d_i from the f_i''(0)=0 constraint."""
    return 1.0 / xp.sqrt(3.0 * c)


def derived_a(b, c, d):
    """Dependent coefficient a_i from the f_i(1)=1 constraint."""
    u = (1.0 + d) ** 2
    return (1.0 + c * u) / (1.0 + b * u)


# =============================================================================
# HYPERON COUPLINGS (report §2.4)
# =============================================================================
_SQRT2 = 1.4142135623730951

#: SU(6) isoscalar-vector and isovector ratios x_iY = Gamma_iY / Gamma_iN,
#: and the ABSOLUTE hidden-strange g_phiY as a multiple of g_omegaN(n_sat)
#: (report §2.4a). φ is constant-coupled in DD2Y, so g_phi is stored as a
#: multiple of the nucleon ω scale rather than a density-dependent ratio.
SU6_HYPERON = {
    #             x_omega  x_rho   g_phi/g_omegaN
    "Lambda": dict(x_omega=2.0 / 3.0, x_rho=0.0, phi_over_omegaN=-_SQRT2 / 3.0),
    "Sigma+": dict(x_omega=2.0 / 3.0, x_rho=2.0, phi_over_omegaN=-_SQRT2 / 3.0),
    "Sigma0": dict(x_omega=2.0 / 3.0, x_rho=2.0, phi_over_omegaN=-_SQRT2 / 3.0),
    "Sigma-": dict(x_omega=2.0 / 3.0, x_rho=2.0, phi_over_omegaN=-_SQRT2 / 3.0),
    "Xi0": dict(x_omega=1.0 / 3.0, x_rho=1.0, phi_over_omegaN=-2.0 * _SQRT2 / 3.0),
    "Xi-": dict(x_omega=1.0 / 3.0, x_rho=1.0, phi_over_omegaN=-2.0 * _SQRT2 / 3.0),
}

#: Which measured potential each hyperon's scalar coupling is fitted to.
_POTENTIAL_KEY = {
    "Lambda": "U_Lambda", "Sigma+": "U_Sigma", "Sigma0": "U_Sigma",
    "Sigma-": "U_Sigma", "Xi0": "U_Xi", "Xi-": "U_Xi",
}


def scalar_ratio_from_potential(U_Y, x_omega, Gs_N_sat, Gw_N_sat,
                                sigma_sat, omega0_sat, SigmaR_sat):
    """
    x_sigmaY = Gamma_sigmaY / Gamma_sigmaN(n_sat) from the hyperon potential in
    SNM at saturation (report §2.4b):

        U_Y = -Gamma_sigmaY sigma + Gamma_omegaY omega0 + Sigma^R   (all at n_sat)

    Gamma_omegaY = x_omega Gamma_omegaN(n_sat). One linear equation per hyperon.
    """
    Gamma_sigmaY = (x_omega * Gw_N_sat * omega0_sat + SigmaR_sat - U_Y) / sigma_sat
    return Gamma_sigmaY / Gs_N_sat

