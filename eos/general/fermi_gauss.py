"""One species as a relativistic ideal gas, by split-panel Gauss-Legendre
quadrature, in natural units.

The Fermi integrals of this repository have one home (CLAUDE.md section 7) and
JEL -- `eos.general.fermi_integrals` -- is the validated implementation there.
This module is an ALTERNATIVE ALONGSIDE it, never a replacement, and it exists
because the quark models need three things JEL's rational approximation does
not give them:

  * machine-precision agreement with the four identities
    n = dP/dmu, rho_s = -dP/dm, s = dP/dT and eps = -P + mu n + T s, which are
    what a self-consistent model's field and gap equations are differentiated
    against. JEL is accurate to about 1e-4, which is ample for a tabulated
    equation of state and not ample for a residual whose gate is 1e-10;
  * an explicit upper limit k_max on the momentum integral, because a CUT
    theory -- NJL -- carries one as a parameter and its medium integral is not
    a spectator to it;
  * the SCALAR density rho_s, which drives the mass and dielectric equations.

Its accuracy is bought by splitting the integral at the Fermi momentum rather
than by counting nodes: at T = 0.5 MeV a single panel needs 5000 nodes to
reach what three panels reach with 100, and single-panel accuracy is not even
monotone in the node count, because whether a node lands near the Fermi step
is accidental.

One statement here is physics rather than numerics: AT T = 0 A MODE WITH
mu <= m CONTRIBUTES EXACTLY ZERO, and this returns exactly zero rather than a
small number. In `eos.ccdm` that is the confinement mechanism itself -- a mode
whose dielectric-dressed mass has run above its potential is simply absent,
and smoothing it destroys the first-order deconfinement transition -- so it is
a branch, not a numerical nuisance.

Units are natural throughout: momenta, masses and potentials in MeV, densities
in MeV^3, eps and P in MeV^4, s in MeV^3. The models' own public boundaries
convert to fm.
"""
from dataclasses import dataclass
import math

import numpy as np

from eos.general.pairing import panel_nodes

_PI2 = math.pi ** 2

#: Spin degeneracy of one colour-flavour quark mode. Colour is resolved mode
#: by mode, so it is NOT in here: the familiar g = 6 of a quark flavour is
#: this 2 times the three colours summed explicitly. It is the default only
#: because both callers are quark models; anything else passes its own g.
DEGENERACY = 2.0


@dataclass(frozen=True)
class ModeThermo:
    """One colour-flavour mode's medium integrals, all from one quadrature pass.

    n and rho_s in MeV^3, eps and P in MeV^4, s in MeV^3. Antiparticles
    SUBTRACT in n and ADD in rho_s, eps and P. `P_k4` is the second standard
    pressure form, carried only as a diagnostic: `P` is the logarithm form and
    is the one every assembly uses.
    """
    n: float
    rho_s: float
    eps: float
    P: float
    s: float
    P_k4: float


#: What `kinetic_thermo` returns for a mode that is not in the medium at all.
_ABSENT = ModeThermo(n=0.0, rho_s=0.0, eps=0.0, P=0.0, s=0.0, P_k4=0.0)


def _occupation(x, T):
    """f = 1/(1 + e^(x/T)), written through tanh so it cannot overflow."""
    if T <= 0.0:
        return np.where(x < 0.0, 1.0, np.where(x > 0.0, 0.0, 0.5))
    return 0.5 * (1.0 - np.tanh(0.5 * x / T))


def _log_term(x, T):
    """T ln(1 + e^(-x/T)), and its T -> 0 limit max(-x, 0)."""
    if T <= 0.0:
        return np.maximum(-x, 0.0)
    return T * np.logaddexp(0.0, -x / T)


def _entropy_integrand(E, mu, T):
    """The entropy per state, summed over particles and antiparticles.

        s = sum_(+-) [ (x_+-/T) f_+- + ln(1 + e^(-x_+-/T)) ] ,  x_+- = E -+ mu

    Integrated rather than taken from the single-mode Euler relation
    s = (eps + P - mu n)/T. The two are equal exactly -- the identity holds
    integrand by integrand, so it survives the cutoff -- but the Euler route
    is a difference of three numbers of order 1e9 divided by T, and in a cold
    nearly-degenerate gas, where s is genuinely of order 1e-8 of them, the
    cancellation eats every significant digit. Integrating costs one more
    array in the same pass and leaves the Euler relation available as a CHECK
    rather than spending it as a definition.
    """
    if T <= 0.0:
        return np.zeros_like(E)
    out = np.zeros_like(E)
    for x in (E - mu, E + mu):
        z = x / T
        out = out + z * _occupation(x, T) + np.logaddexp(0.0, -z)
    return out


def kinetic_thermo(mu, m, T, k_max, g=DEGENERACY, quadrature=None):
    """One mode as an ideal gas cut at `k_max` [MeV].

        n     = (g/2 pi^2) int dk k^2       (f+ - f-)
        rho_s = (g/2 pi^2) int dk k^2 (m/E) (f+ + f-)
        eps   = (g/2 pi^2) int dk k^2  E    (f+ + f-)
        P     = (g/2 pi^2) int dk k^2 T[ln(1 + e^-(E-mu)/T) + ln(1 + e^-(E+mu)/T)]

    with E = sqrt(k^2 + m^2) and f-+ the Fermi functions at E -+ mu, and the
    entropy integrand of `_entropy_integrand`.

    AT T = 0 WITH |mu| <= |m| EVERY ONE OF THEM IS EXACTLY ZERO and this
    returns zero without integrating. That is not an optimisation: it is the
    statement that a mode too heavy for its own potential is not in the
    medium, which is the confinement mechanism of `eos.ccdm` and the
    threshold behaviour of every T = 0 Fermi gas.

    P is the LOGARITHM form. Against the k^4/E form it differs by the surface
    term `surface_term` below, which is 0.1% of P at (m, mu, T) = (100, 500,
    20) MeV, 10.5% at (40, 590, 30) and 39.9% at (140, 700, 50). At T = 0 with
    k_F < k_max the two agree, which is exactly why the error hides until a
    table is built at finite temperature.
    """
    if T <= 0.0 and abs(mu) <= abs(m):
        return _ABSENT
    if quadrature is None:
        k_F = math.sqrt(mu ** 2 - m ** 2) if abs(mu) > abs(m) else 0.0
        quadrature = panel_nodes([k_F] if k_F > 0.0 else [], T, k_max)
    k, w = quadrature

    E = np.sqrt(k ** 2 + m ** 2)
    f_p = _occupation(E - mu, T)
    f_m = _occupation(E + mu, T)
    weight = w * k ** 2
    pref = g / (2.0 * _PI2)

    n = pref * float(np.sum(weight * (f_p - f_m)))
    rho_s = pref * float(np.sum(weight * (m / E) * (f_p + f_m)))
    eps = pref * float(np.sum(weight * E * (f_p + f_m)))
    P = pref * float(np.sum(weight * (_log_term(E - mu, T)
                                      + _log_term(E + mu, T))))
    P_k4 = (g / (6.0 * _PI2)) * float(np.sum(w * k ** 4 / E * (f_p + f_m)))
    s = pref * float(np.sum(weight * _entropy_integrand(E, mu, T)))
    return ModeThermo(n=n, rho_s=rho_s, eps=eps, P=P, s=s, P_k4=P_k4)


def surface_term(mu, m, T, k_max, g=DEGENERACY):
    """P_log - P_k4 in closed form [MeV^4].

        (g/6 pi^2) k_max^3 T [ ln(1 + e^-(E_max - mu)/T)
                             + ln(1 + e^-(E_max + mu)/T) ]

    the boundary term of the integration by parts that turns one pressure
    integral into the other. It does not vanish when the integral is cut, and
    it is not small.
    """
    E_max = math.sqrt(k_max ** 2 + m ** 2)
    both = _log_term(np.array(E_max - mu), T) + _log_term(np.array(E_max + mu), T)
    return (g / (6.0 * _PI2)) * k_max ** 3 * float(both)


def unbounded_k_max(mu, m, T, pad=200.0):
    """A momentum ceiling at which an UNREGULARISED integrand has died [MeV].

        k_max = max(|mu|, m) + 45 T + 12 m + pad

    For a model whose medium integrals carry no cutoff of their own -- the
    colour-dielectric one -- the upper limit is a numerical choice rather than
    a parameter, and it has to clear both the Fermi surface and the thermal
    and antiparticle tails. The 12 m term is what covers the antiparticle
    contribution of a heavy confined mode, whose integrand decays on the scale
    of m rather than of T.
    """
    return max(abs(mu), abs(m)) + 45.0 * T + 12.0 * abs(m) + pad
