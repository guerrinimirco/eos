"""
enjl/thermodynamics.py
====================
T = 0 thermodynamics of one fermion species in the extended NJL model
(Xia 2024, PRD 110, 014022 — paper Eqs. (11)-(13)).

Only quarks carry the 3-momentum cut-off Lambda (they regularize the vacuum
Dirac sea); baryons and leptons have Lambda = 0. All functions are exact
closed forms in natural units: nu, M, Lambda in MeV; densities in MeV^3;
eps/P in MeV^4.

Kinetic number/scalar density and kinetic energy density are Eqs. (11)-(13)
with x_i = nu_i / M_i and y_i = Lambda / M_i:

    n     = g nu^3 / (6 pi^2)
    n_s   = (g M^3/4pi^2) [ x*sqrt(x^2+1) - asinh(x)
                            - y*sqrt(y^2+1) + asinh(y) ]
    eps   = (g M^4/16pi^2)[ x(2x^2+1)sqrt(x^2+1) - asinh(x)
                            - y(2y^2+1)sqrt(y^2+1) + asinh(y) ]

The kinetic pressure of one species is the exact thermodynamic relation
P = nu*n - eps (equivalent to the standard closed form; verified
numerically). For quarks this leaves a vacuum constant that the EOS assembly
removes together with the E0 subtraction of Eq. (13).
"""
import math

_PI2 = math.pi ** 2


def kF_from_n(n, g):
    """Fermi momentum [MeV] from number density n [MeV^3], degeneracy g."""
    return (6.0 * _PI2 * n / g) ** (1.0 / 3.0)


def number_density_t0(kF, g):
    """n [MeV^3] = g nu^3 / (6 pi^2), Eq. (11)."""
    return g * kF ** 3 / (6.0 * _PI2)


def scalar_density_t0(kF, ms, g, Lambda=0.0):
    """Scalar density n_s [MeV^3], Eq. (12).

    Baryons/leptons: Lambda = 0, only the x-terms survive. Quarks: the
    y-terms regularize the vacuum part. At kF = 0 the quark condensate is
    n_s = -(g M^3/4pi^2)[y sqrt(y^2+1) - asinh(y)] < 0.
    """
    x = kF / ms
    y = Lambda / ms
    fx = x * math.sqrt(x * x + 1.0) - _asinh(x)
    fy = y * math.sqrt(y * y + 1.0) - _asinh(y)
    return (g * ms ** 3 / (4.0 * _PI2)) * (fx - fy)


def eps_kin_t0(kF, ms, g, Lambda=0.0):
    """Kinetic energy density [MeV^4], Eq. (13) kinetic + vacuum parts.

    Quarks: y-term subtracts the vacuum Dirac-sea energy up to the cut-off.
    Baryons/leptons: Lambda = 0, x-term only. At kF = 0, quarks keep the
    (negative) vacuum part, which E0 of Eq. (13) compensates globally.
    """
    x = kF / ms
    y = Lambda / ms
    fx = x * (2.0 * x * x + 1.0) * math.sqrt(x * x + 1.0) - _asinh(x)
    fy = y * (2.0 * y * y + 1.0) * math.sqrt(y * y + 1.0) - _asinh(y)
    return (g * ms ** 4 / (16.0 * _PI2)) * (fx - fy)


def P_kin_t0(kF, ms, g, Lambda=0.0):
    """Kinetic pressure [MeV^4] = nu*n - eps (exact for one species).

    For quarks this carries the vacuum constant of eps_kin_t0; the EOS
    assembly removes it together with the E0 subtraction of Eq. (13), and the
    full pressure is computed as P = sum_i mu_i n_i - E (Eq. (19)).
    """
    nu = math.sqrt(kF * kF + ms * ms)
    return nu * number_density_t0(kF, g) - eps_kin_t0(kF, ms, g, Lambda)


def kinetic_thermo(nu, m, g, Lambda=0.0):
    """Full T=0 kinetic thermodynamics of one species.

    Parameters:
        nu:     kinetic chemical potential nu = mu - Sigma0 [MeV]
        m:      (effective) mass [MeV]
        g:      degeneracy
        Lambda: 3-momentum cut-off [MeV]; 0 for baryons/leptons

    Returns:
        (n, P, eps, ns) in natural units (MeV^3, MeV^4, MeV^4, MeV^3).
        P is the kinetic pressure of one species = nu*n - eps.
    """
    kF2 = nu * nu - m * m
    if kF2 <= 0.0 or nu <= 0.0:
        return 0.0, 0.0, 0.0, scalar_density_t0(0.0, m, g, Lambda)
    kF = math.sqrt(kF2)
    n = number_density_t0(kF, g)
    ns = scalar_density_t0(kF, m, g, Lambda)
    eps = eps_kin_t0(kF, m, g, Lambda)
    P = P_kin_t0(kF, m, g, Lambda)
    return n, P, eps, ns


def _asinh(z):
    """asinh(z) with the 0*inf guard needed at z = 0 (vacuum kF = 0)."""
    if z <= 0.0:
        return 0.0
    return math.asinh(z)
