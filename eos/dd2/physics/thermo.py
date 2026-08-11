"""
physics/thermo.py
====================
Kinetic (ideal Fermi gas) thermodynamics for a single species.

Natural units throughout this module: energies, masses and momenta in MeV,
densities in MeV^3, energy density / pressure in MeV^4.

T = 0: exact closed forms.
T > 0: Johns-Ellis-Lattimer Fermi integrals from eos.general.fermi_integrals,
converted from their fm-based units to natural units here so that no call site
has to. Use this rather than calling the JEL routines directly.
"""
import numpy as np
from eos.general.physics_constants import hc3

_PI2 = np.pi ** 2


def kF_from_n(n, g):
    """Fermi momentum [MeV] from number density n [MeV^3], degeneracy g."""
    return (6.0 * _PI2 * n / g) ** (1.0 / 3.0)


def number_density_t0(kF, g):
    """n [MeV^3]."""
    return g * kF ** 3 / (6.0 * _PI2)


def scalar_density_t0(kF, ms, g):
    """n_s [MeV^3]."""
    if kF <= 0.0:
        return 0.0
    EF = np.sqrt(kF ** 2 + ms ** 2)
    return (g * ms / (4.0 * _PI2)) * (kF * EF - ms ** 2 * np.log((kF + EF) / ms))


def eps_kin_t0(kF, ms, g):
    """Kinetic energy density [MeV^4]."""
    if kF <= 0.0:
        return 0.0
    EF = np.sqrt(kF ** 2 + ms ** 2)
    return (g / (16.0 * _PI2)) * (kF * EF * (2.0 * kF ** 2 + ms ** 2)
                                  - ms ** 4 * np.log((kF + EF) / ms))


def P_kin_t0(kF, ms, g):
    """Kinetic pressure [MeV^4]."""
    if kF <= 0.0:
        return 0.0
    EF = np.sqrt(kF ** 2 + ms ** 2)
    return (g / (48.0 * _PI2)) * (kF * EF * (2.0 * kF ** 2 - 3.0 * ms ** 2)
                                  + 3.0 * ms ** 4 * np.log((kF + EF) / ms))


def kinetic_thermo(mu_eff, m, g, T=0.0):
    """
    Full kinetic thermodynamics of one fermion species.

    Parameters:
        mu_eff: effective (kinetic) chemical potential mu - Sigma0 [MeV]
        m:  (effective) mass [MeV]
        g:  degeneracy
        T:  temperature [MeV]

    Returns:
        (n, P, eps, s, ns) in natural units (MeV^3, MeV^4, MeV^4, MeV^3, MeV^3).
    """
    if T == 0.0:
        kF2 = mu_eff * mu_eff - m * m
        if kF2 <= 0.0 or mu_eff <= 0.0:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        kF = np.sqrt(kF2)
        if m == 0.0:
            # massless (neutrinos): the ms^4 log terms are 0*inf; use the
            # ultra-relativistic closed forms (E=k, eps = g kF^4/8pi^2, P=eps/3).
            eps = g * kF ** 4 / (8.0 * _PI2)
            return number_density_t0(kF, g), eps / 3.0, eps, 0.0, 0.0
        return (number_density_t0(kF, g), P_kin_t0(kF, m, g),
                eps_kin_t0(kF, m, g), 0.0, scalar_density_t0(kF, m, g))
    from eos.general.fermi_integrals import solve_fermi_jel
    n, P, e, s, ns = solve_fermi_jel(mu_eff, T, m, g)
    return n * hc3, P * hc3, e * hc3, s * hc3, ns * hc3
