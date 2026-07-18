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
