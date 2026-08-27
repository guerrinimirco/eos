"""The closure of the ABPR phase, and the inverse maps that reach it.

`thermodynamics.py` computes quantities from a chemical potential; this module
finds the potential that meets a condition, and assembles the solved point.

There is only one condition to meet. Colour-flavour locking fixes the
composition -- n_u = n_d = n_s, hence Y_C = 0 and Y_S = +1 identically -- so
the phase has a single independent variable and the only request it answers is
"the state at this baryon density". Following `eos.alphabag`, that request is
named after the phase rather than after an equilibrium: the mode is `cfl`.

The other four modes of this repository each raise, naming the physics rather
than pleading incompleteness; `MODE_REFUSALS` holds the reasons, and they are
spelled out in `abpr.tex`.

Nothing here iterates. n_B(mu) is a cubic in mu with no quadratic and no
constant term, and P(mu) and eps(mu) are quadratics in mu^2, so all three
inversions are closed forms. The convergence status a solved point carries is
nonetheless real: it is the residual of the closed form against the equation
it inverts, judged on the same scaled gate
(`eos.general.solve.RESIDUAL_TOL`) as every model in this repository that does
iterate, so "converged" means the same thing here as it does there.

See `abpr.tex` for the closed forms, the branch each one takes, and why that
branch is the physical one.
"""
from dataclasses import dataclass
from math import acos, cos, pi, sqrt

import numpy as np

from eos.general.basis import charge_potentials_from_quarks
from eos.general.solve import RESIDUAL_TOL, scaled_residual_max
from eos.abpr.parameters import Parameters
from eos.abpr.thermodynamics import (
    baryon_density, coefficients, energy_density, pressure,
    sound_speed_squared, thermo_from_mu,
)

#: The modes this model closes, and the conditions each one takes beyond
#: (n_B, T). `cfl` is not one of the repository's four equilibrium modes but a
#: PHASE, closed by flavour locking rather than by an equilibrium condition,
#: and it is reached the same way because it is the same kind of request -- a
#: state at (n_B, T). It takes no fraction: the gap belongs to the parameter
#: set here (see `parameters.py`), unlike in `eos.alphabag` where it selects
#: between two phases of one potential and arrives as `Delta0`.
MODE_FRACTIONS = {
    "cfl": (),
}

#: Why each of the repository's four equilibrium modes has no state in this
#: phase. All four are refused for physics, not for missing implementation:
#: locking has already fixed the composition those modes exist to determine.
MODE_REFUSALS = {
    "beta_eq_neutrinoless":
        "beta equilibrium fixes the charge potential through mu_C + mu_e = 0, "
        "but colour-flavour locking has already fixed the composition and "
        "left mu_C = 0 with no electrons to equilibrate against, so the "
        "condition has no free variable to determine. Unpaired quark matter "
        "in beta equilibrium is eos.alphabag or eos.vmit",
    "beta_eq_neutrino_trapped":
        "the same as beta_eq_neutrinoless, and in addition the phase carries "
        "no leptons of any family, so there is no Y_Le for the mode to fix",
    "fixed_YC":
        "the locked phase has Y_C = 0 identically, at every density and for "
        "every parameter set; any other Y_C asks for a state it does not "
        "have, and Y_C = 0 is the 'cfl' mode itself",
    "fixed_YC_YS":
        "the locked phase has Y_C = 0 and Y_S = +1 identically -- both are "
        "outputs of the closure, not inputs to it -- so in particular the "
        "symmetric-matter slice Y_C = 0.5, Y_S = 0 this mode exists for is "
        "not a state of deconfined locked matter",
}


def check_mode(mode):
    """Raise unless `mode` is one this model closes, saying why if it is not.

    A malformed call is a programming error and raises before any work, rather
    than being reported as a status: a sampler that would repeat it a million
    times in silence is worse off for the silence.
    """
    if mode in MODE_FRACTIONS:
        return
    if mode in MODE_REFUSALS:
        raise NotImplementedError(
            f"eos.abpr does not support mode {mode!r}: {MODE_REFUSALS[mode]}")
    raise ValueError(f"unknown mode {mode!r}; eos.abpr closes "
                     f"{list(MODE_FRACTIONS)}, and refuses "
                     f"{list(MODE_REFUSALS)} for the physics")


def check_temperature(T):
    """Raise unless T = 0, the only temperature the parametrization has."""
    if T != 0.0:
        raise NotImplementedError(
            f"eos.abpr is a T = 0 parametrization and was asked for "
            f"T = {T} MeV; the finite-temperature CFL phase, with its BCS gap "
            f"Delta(T) and its thermal sectors, is the 'cfl' mode of "
            f"eos.alphabag")


# =============================================================================
# RESULT RECORD
# =============================================================================
@dataclass
class CFLPoint:
    """One solved colour-flavour locked state.

    The same record name and the same fields as `eos.alphabag.CFLPoint`, so a
    caller comparing the analytic parametrization against the model it is the
    T = 0 limit of reads one layout.

    `converged` is judged on `error`, the residual of the closed-form inverse
    divided by the scale of the quantity it balances (see
    `eos.general.solve.scaled_residual_max`); it is dimensionless, and the
    gate is `eos.general.solve.RESIDUAL_TOL`. When `converged` is False the
    request was outside the phase -- a pressure below -B, an energy density
    below the bag -- and no other field is a physical state.

    The phase carries no electrons, flavour locking making it neutral by
    construction, so `mu_e` and `mu_nu` are zero rather than solved.
    """
    # Convergence info
    converged: bool = False
    error: float = 0.0

    # Input conditions
    n_B: float = 0.0        # baryon density (fm^-3)
    T: float = 0.0          # temperature (MeV), always zero
    Delta0: float = 0.0     # the pairing gap (MeV)
    Delta: float = 0.0      # the same: there is no Delta(T) here
    Y_C: float = 0.0        # charge fraction, zero by construction
    Y_S: float = 0.0        # strangeness fraction, +1 by construction

    # Chemical potentials (MeV)
    mu_u: float = 0.0
    mu_d: float = 0.0
    mu_s: float = 0.0
    mu_e: float = 0.0
    mu_nu: float = 0.0
    mu_B: float = 0.0
    mu_C: float = 0.0
    mu_S: float = 0.0

    # Densities (fm^-3)
    n_u: float = 0.0
    n_d: float = 0.0
    n_s: float = 0.0

    # Thermodynamics
    P_total: float = 0.0
    e_total: float = 0.0
    s_total: float = 0.0
    f_total: float = 0.0

    # Fractions
    Y_u: float = 0.0
    Y_d: float = 0.0
    Y_s: float = 0.0


# =============================================================================
# THE INVERSE MAPS, IN CLOSED FORM
# =============================================================================
def mu_from_nB(n_B, par):
    """The chemical potential at a given baryon density.

    Inverts n_B = a mu^3 + c mu, a depressed cubic (no quadratic term, no
    constant term), by Cardano. With p = c/a, q = -n_B/a and discriminant
    D = (q/2)^2 + (p/3)^3,

        mu = u - p/(3u),   u = cbrt(-q/2 + sqrt(D))            (D >= 0)

    and for D < 0 the largest of the three trigonometric roots. The second
    form -- p/(3u) rather than the textbook cbrt(-q/2 - sqrt(D)) -- avoids the
    cancellation that costs digits when |p| is small, and takes the residual
    at n_B = 3 fm^-3 from 1.4e-11 to 8.9e-16 fm^-3.

    The root taken is always the physical one. For n_B > 0 the cubic's sign
    sequence has exactly one change whatever the sign of c, so by Descartes'
    rule there is exactly one positive root; when Delta0 < m_s/2 and c is
    negative, n_B(mu) is negative below mu = sqrt(-c/a) and the positive root
    is the one above that crossing.

    Args:
        n_B: baryon density (fm^-3), positive
        par: the parameter set

    Returns:
        (mu in MeV, converged), the status judged on the residual of the
        cubic scaled by n_B
    """
    A, C, _ = coefficients(par)
    a = 4.0 * A / 3.0
    c = 2.0 * C / 3.0

    p = c / a
    q = -n_B / a
    disc = (q / 2.0) ** 2 + (p / 3.0) ** 3
    if disc >= 0.0:
        u = np.cbrt(-q / 2.0 + sqrt(disc))
        mu = float(u - p / (3.0 * u)) if u != 0.0 else 0.0
    else:
        m = 2.0 * sqrt(-p / 3.0)
        theta = acos(3.0 * q / (p * m)) / 3.0
        mu = max(m * cos(theta), m * cos(theta - 2.0 * pi / 3.0),
                 m * cos(theta - 4.0 * pi / 3.0))
    return mu, _accepted(baryon_density(mu, par) - n_B, n_B)


def mu_from_P(P_target, par):
    """The chemical potential at a given pressure.

    Inverts P = A mu^4 + C mu^2 - B, a quadratic in mu^2:

        mu = sqrt( [-C + sqrt(C^2 + 4 A (P + B))] / (2 A) )

    The + branch is the physical one: it is the larger root in mu^2, and when
    C < 0 -- the only case in which the smaller root is positive too -- it is
    the branch on which n_B and dP/dmu are positive.

    P = 0 is the surface of a self-bound star, where a bare strange star ends
    with no crust, and is the reason this inverse exists.

    Args:
        P_target: pressure (MeV/fm^3). Below -B/(hbar c)^3 there is no state
            and the returned status says so.
        par: the parameter set

    Returns:
        (mu in MeV, converged), the status judged on the residual of P scaled
        by the bag constant -- the scale of the pressure in this model, and
        the only one available at P = 0
    """
    A, C, B = coefficients(par)
    under = C * C + 4.0 * A * (P_target + B)
    if under < 0.0:
        return 0.0, False
    mu2 = (-C + sqrt(under)) / (2.0 * A)
    if mu2 < 0.0:
        return 0.0, False
    mu = sqrt(mu2)
    return mu, _accepted(pressure(mu, par) - P_target, B)


def mu_from_eps(eps_target, par):
    """The chemical potential at a given energy density.

    Inverts eps = 3 A mu^4 + C mu^2 + B, again a quadratic in mu^2:

        mu = sqrt( [-C + sqrt(C^2 + 12 A (eps - B))] / (6 A) )

    with the + branch physical for the same reason as in `mu_from_P`.

    Args:
        eps_target: energy density (MeV/fm^3). Below B/(hbar c)^3 there is no
            state and the returned status says so.
        par: the parameter set

    Returns:
        (mu in MeV, converged), the status judged on the residual of eps
        scaled by eps
    """
    A, C, B = coefficients(par)
    under = C * C + 12.0 * A * (eps_target - B)
    if under < 0.0:
        return 0.0, False
    mu2 = (-C + sqrt(under)) / (6.0 * A)
    if mu2 < 0.0:
        return 0.0, False
    mu = sqrt(mu2)
    return mu, _accepted(energy_density(mu, par) - eps_target,
                         abs(eps_target) if eps_target else B)


def _accepted(residual, scale):
    """Whether one closed-form inverse landed inside the repository's gate.

    The scale is the quantity the inverted equation balances, so that the
    tolerance means the same thing here as in a model whose rows carry mixed
    units (see `eos.general.solve.scaled_residual_max`).
    """
    return scaled_residual_max([residual], [scale]) <= RESIDUAL_TOL


# =============================================================================
# THE SOLVED POINT
# =============================================================================
def point_from_mu(mu, par, converged=True, error=0.0):
    """The totals of a locked state at a given chemical potential.

    The three flavour potentials are equal here -- that is what the ABPR
    parametrization assumes, absorbing the mass difference into its
    -3 m_s^2 mu^2/(4 pi^2) term -- so mu_C = mu_S = 0 and mu_B = 3 mu. This is
    NOT a property of CFL matter in general: locking equal densities at
    unequal masses needs unequal potentials, and `eos.alphabag.solve_cfl`
    solves for exactly that, finding mu_S of a few tens of MeV. The two
    bookkeepings are each internally consistent and differ by the m_s^4 term
    measured in `verify/run_full_check.py`.

    Args:
        mu: common quark chemical potential (MeV)
        par: the parameter set
        converged, error: the status of the inverse that produced mu

    Returns:
        CFLPoint
    """
    block = thermo_from_mu(mu, par)
    n_B = block.n_B
    mu_B, mu_C, mu_S = charge_potentials_from_quarks(mu, mu, mu)

    return CFLPoint(
        converged=converged, error=error,
        n_B=n_B, T=0.0, Delta0=par.Delta0, Delta=par.Delta0,
        Y_C=block.n_C / n_B if n_B else 0.0,
        Y_S=block.n_S / n_B if n_B else 0.0,
        mu_u=mu, mu_d=mu, mu_s=mu, mu_e=0.0, mu_nu=0.0,
        mu_B=mu_B, mu_C=mu_C, mu_S=mu_S,
        n_u=n_B, n_d=n_B, n_s=n_B,
        P_total=block.P, e_total=block.e, s_total=block.s, f_total=block.f,
        Y_u=1.0, Y_d=1.0, Y_s=1.0,
    )


def solve_cfl(par, n_B, T=0.0):
    """Colour-flavour locked quark matter at a given baryon density.

    The one closure this model has. Flavour locking n_u = n_d = n_s replaces
    the equilibrium condition an unpaired phase would need, so the "solve" is
    the inversion of n_B(mu) and nothing else.

    Args:
        par: the parameter set; required, since model parameters are
             arguments and never defaults reached for on the caller's behalf
             (CLAUDE.md section 6)
        n_B: baryon density (fm^-3)
        T: temperature (MeV); anything but zero raises, naming eos.alphabag

    Returns:
        CFLPoint; test `.converged` before using any other field.
    """
    check_temperature(T)
    mu, converged = mu_from_nB(n_B, par)
    error = abs(baryon_density(mu, par) - n_B) / n_B if n_B else 0.0
    return point_from_mu(mu, par, converged=converged, error=error)


def response_at_mu(mu, par):
    """The second-derivative quantities that exist at T = 0.

    Only the speed of sound does. The heat capacities and thermal indices of
    the CompOSE list are not defined at T = 0, and the susceptibilities
    chi_ab = dn_a/dmu_b are singular here because flavour locking leaves n_C
    and n_S with no potential to respond to.

    Returns a dict with `cs2_isothermal`: at T = 0 the isothermal and
    adiabatic derivatives coincide, and the name says which convention the
    number was computed under rather than leaving it to the arguments.
    """
    return {"cs2_isothermal": sound_speed_squared(mu, par)}
