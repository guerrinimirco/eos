"""The ABPR thermodynamic potential and everything derived from it.

Colour-flavour locked quark matter at T = 0, as the closed-form pressure of
Alford, Braby, Paris and Reddy, Astrophys. J. 629, 969 (2005). The condensate
locks the three flavour densities together, n_u = n_d = n_s, so the phase has
a single independent potential: the common quark chemical potential mu, with
mu_B = 3 mu. The pressure is four terms,

    P(mu) = 3 a4 mu^4/(4 pi^2 (hc)^3)       free three-flavour gas + pQCD
          - 3 m_s^2 mu^2/(4 pi^2 (hc)^3)    strange mass, to O(m_s^2)
          + 3 Delta0^2 mu^2/(pi^2 (hc)^3)   CFL condensation energy
          - B/(hc)^3                        the bag

and everything else in this module is a derivative of it, which is what makes
the model thermodynamically consistent by construction rather than by check:

    n_B = dP/dmu_B = (1/3) dP/dmu       s = dP/dT = 0
    eps = -P + mu_B n_B                 f = eps - T s = eps

The two mu^2 terms are grouped as one coefficient C, since only their sum ever
appears. The sign of C decides the shape of the model: C > 0 (a gap larger
than m_s/2) approaches the conformal c_s^2 = 1/3 from above, C < 0 from below.

This module never knows which mode it is in -- it takes a chemical potential
and returns quantities. Finding the mu that meets a condition is `solver.py`.

Units are fm-based at every boundary: mu and masses in MeV, n_B in fm^-3, P
and eps in MeV/fm^3.

See `abpr.tex` for the derivation of each term and for the expansion of the
massive Fermi gas that the m_s^2 term is the leading part of.
"""
from dataclasses import dataclass

from eos.general.basis import quark_charges
from eos.general.physics_constants import hc3, PI2


# =============================================================================
# THE COEFFICIENTS OF THE POTENTIAL
# =============================================================================
def coefficients(par):
    """(A, C, B) of P = A mu^4 + C mu^2 - B, in MeV/fm^3 per power of mu.

        A = 3 a4/(4 pi^2 (hc)^3)                the free gas with its pQCD
                                                factor folded in
        C = 3 (Delta0^2 - m_s^2/4)/(pi^2 (hc)^3)  the pairing energy against
                                                the strange mass
        B = B4^4/(hc)^3                         the bag

    Written once because P, n_B, eps and c_s^2 are all built from the same
    three numbers, and because the sign of C is the one thing about a
    parameter set worth reading off directly.
    """
    A = 3.0 * par.a4 / (4.0 * PI2 * hc3)
    C = 3.0 * (par.Delta0**2 - par.m_s**2 / 4.0) / (PI2 * hc3)
    return A, C, par.B / hc3


# =============================================================================
# THE POTENTIAL AND ITS DERIVATIVES
# =============================================================================
def pressure(mu, par):
    """P(mu) in MeV/fm^3, the four terms of the ABPR potential.

        P = 3 a4 mu^4/(4 pi^2 (hc)^3)
            + 3 (Delta0^2 - m_s^2/4) mu^2/(pi^2 (hc)^3)
            - B/(hc)^3

    Args:
        mu: common quark chemical potential (MeV), mu = mu_B/3
        par: the parameter set

    Returns:
        pressure (MeV/fm^3)
    """
    A, C, B = coefficients(par)
    mu2 = mu * mu
    return A * mu2 * mu2 + C * mu2 - B


def baryon_density(mu, par):
    """n_B(mu) in fm^-3, as the derivative dP/dmu_B with mu_B = 3 mu.

        n_B = (1/3) dP/dmu
            = a4 mu^3/(pi^2 (hc)^3)
              + 2 (Delta0^2 - m_s^2/4) mu/(pi^2 (hc)^3)

    A cubic in mu with no quadratic and no constant term, which is what makes
    its inverse a closed form (`solver.mu_from_nB`).

    Args:
        mu: common quark chemical potential (MeV)
        par: the parameter set

    Returns:
        baryon density (fm^-3)
    """
    A, C, _ = coefficients(par)
    return (4.0 * A * mu * mu + 2.0 * C) * mu / 3.0


def energy_density(mu, par):
    """eps(mu) in MeV/fm^3, from the Euler relation eps = -P + mu_B n_B.

    Equivalently eps = 3 A mu^4 + C mu^2 + B: the bag enters with the opposite
    sign to the one it has in P, which is why it cancels out of eps + P.
    Because eps is DEFINED this way rather than integrated, the Euler relation
    holds by construction; `verify/` checks it all the same.

    Args:
        mu: common quark chemical potential (MeV)
        par: the parameter set

    Returns:
        energy density (MeV/fm^3)
    """
    return -pressure(mu, par) + 3.0 * mu * baryon_density(mu, par)


def entropy(mu, par):
    """s(mu) = dP/dT = 0. This model is defined at T = 0.

    Returned rather than omitted: a solved point carries s, and zero is its
    value here, not a quantity the model declines to compute.
    """
    return 0.0


def sound_speed_squared(mu, par):
    """c_s^2 = dP/deps in closed form, with no numerical differentiation.

        c_s^2 = (2 A mu^2 + C)/(6 A mu^2 + C)  ->  1/3 as mu -> infinity

    The conformal limit is approached from above when C > 0 (Delta0 > m_s/2)
    and from below when C < 0. Above the P = 0 surface the value is between 0
    and 1 for every physical parameter set; below it, where no matter exists,
    a negative C can put the denominator through zero.

    Args:
        mu: common quark chemical potential (MeV)
        par: the parameter set

    Returns:
        the squared speed of sound, dimensionless, in units of c
    """
    A, C, _ = coefficients(par)
    mu2 = mu * mu
    return (2.0 * A * mu2 + C) / (6.0 * A * mu2 + C)


# =============================================================================
# THE BLOCK AT ONE CHEMICAL POTENTIAL
# =============================================================================
@dataclass(frozen=True)
class Thermo:
    """The strongly-interacting sector at one chemical potential.

    Densities in fm^-3, P, eps and f in MeV/fm^3, s in fm^-3. There is no
    `assemble` step in this model and no separate flavour blocks to sum:
    flavour locking makes the three quark densities equal to n_B, so the
    totals ARE this record.
    """
    mu: float = 0.0         # common quark chemical potential (MeV)
    n_B: float = 0.0        # baryon density (fm^-3)
    n_C: float = 0.0        # charge density, zero by construction (fm^-3)
    n_S: float = 0.0        # strangeness density, S = +1 per s quark (fm^-3)
    P: float = 0.0
    e: float = 0.0
    s: float = 0.0
    f: float = 0.0


def thermo_from_mu(mu, par):
    """The whole phase at a given chemical potential.

    With n_u = n_d = n_s = n_B and the quantum numbers of
    `eos.general.basis`, the conserved-charge densities of the locked phase
    are

        n_B = (n_u + n_d + n_s)/3 = n_B
        n_C = (2 n_u - n_d - n_s)/3 = 0        identically
        n_S = n_s = n_B                        identically

    -- the phase is electrically neutral by construction, with no leptons, and
    maximally strange. `eos.general.basis.quark_charges` is called rather than
    the zeros being written down, so this bookkeeping cannot drift away from
    the one every other model uses.

    Args:
        mu: common quark chemical potential (MeV)
        par: the parameter set

    Returns:
        Thermo
    """
    n_B = baryon_density(mu, par)
    _, n_C, n_S = quark_charges(n_B, n_B, n_B)
    P = pressure(mu, par)
    e = energy_density(mu, par)
    s = entropy(mu, par)
    return Thermo(mu=mu, n_B=n_B, n_C=n_C, n_S=n_S, P=P, e=e, s=s, f=e)
