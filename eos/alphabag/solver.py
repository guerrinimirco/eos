"""Single-point equilibrium solvers for alphaBag quark matter.

One solver per equilibrium mode, plus one for the paired phase. Every unpaired
mode fixes the baryon density; the mode supplies the rest:

    beta equilibrium (neutrinoless)   mu_C + mu_e = 0, mu_S = 0, n_C = n_e
    beta equilibrium (trapped)        ... with mu_nue kept and Y_Le fixed
    fixed Y_C                         n_C = Y_C n_B, mu_S = 0
    fixed Y_C and Y_S                 n_C = Y_C n_B, n_S = Y_S n_B

The unknowns are chemical potentials and nothing else: the thermodynamic
potential of this model is explicit in mu -- no vector field, no gap equation,
the quark masses are parameters -- so unlike a mean-field model there is no
field equation to carry along and no density in the unknown vector.

The colour-flavour locked phase is not one of the modes. It is closed by
flavour locking, n_u = n_d = n_s = n_B, which makes it electrically neutral by
construction; `solve_cfl` is its entry point and takes the pairing gap
Delta0 per call.

Reading order: the cold guesses, the assembly of a solved point, then one
function per mode, then the paired phase, then the warm start.

The thermodynamic kernels are in `thermodynamics.py`, the table driver in
`table.py`, the spec API in `api.py`. See `alphabag.tex` for the physics.

Usage:
    from eos.alphabag.solver import solve_beta_eq_neutrinoless
    point = solve_beta_eq_neutrinoless(0.8, 30.0)
    print(point.converged, point.P_total)
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional

from eos.general.physics_constants import hc, PI2
from eos.general.solve import (
    MU_SCALE_FLOOR, RESIDUAL_TOL, scaled_residual_max, solve_system,
)
from eos.general.thermodynamics_leptons import (
    electron_thermo, electron_thermo_from_density, neutrino_thermo,
    photon_thermo,
)
from eos.alphabag.parameters import Parameters
from eos.alphabag.species import SpeciesFlags
from eos.general.basis import lepton_charges, quark_charges
from eos.alphabag.thermodynamics import (
    cfl_n_correction, cfl_thermo_from_mu, gluon_thermo, quark_density,
    thermo_from_mu,
)

#: The equilibria this model closes, and the conditions each one consumes
#: beyond (n_B, T). The first four names are the repository's, shared with
#: every other model, so the same point is requested from any of them the same
#: way. `cfl` is not one of them: it is a PHASE, closed by flavour locking
#: rather than by an equilibrium condition, and it takes the pairing gap
#: instead of a fraction. It is listed here so that one table drives both.
MODE_FRACTIONS = {
    "beta_eq_neutrinoless": (),
    "beta_eq_neutrino_trapped": ("Y_Le",),
    "fixed_YC": ("Y_C",),
    "fixed_YC_YS": ("Y_C", "Y_S"),
    "cfl": ("Delta0",),
}


def _mu_scale(mu_u, mu_d):
    """The scale a potential equality is judged against: mu_B = mu_u + 2 mu_d.

    Floored, so a pathological iterate passing through mu_B = 0 cannot divide
    by zero; physical quark matter has mu_B ~ 10^3 MeV.
    """
    return max(abs(mu_u + 2.0 * mu_d), MU_SCALE_FLOOR)


# =============================================================================
# RESULT RECORDS
# =============================================================================
@dataclass
class EoSPoint:
    """One solved alphaBag state, with the status a caller must test first.

    `converged` is judged on `error`, the largest equilibrium residual after
    each has been divided by the scale of the quantity it balances (see
    `eos.general.solve.scaled_residual_max`); it is dimensionless, and the
    gate is `RESIDUAL_TOL`. When `converged` is False every other field holds
    the best iterate reached, which is not a physical state.
    """
    # Convergence info
    converged: bool = False
    error: float = 0.0

    # Input conditions
    n_B: float = 0.0        # baryon density (fm^-3)
    T: float = 0.0          # temperature (MeV)
    # Conserved-charge fractions, MEASURED on the solved state: Y_X = n_X/n_B
    # for every charge (CLAUDE.md section 2). A mode that HOLDS one of these
    # reports what it solved, not what it was asked for, and every one of them
    # is defined in every mode -- Y_Le included, not only in the trapped mode
    # that holds it.
    Y_C: float = 0.0        # non-leptonic charge fraction
    Y_S: float = 0.0        # strangeness fraction, S = +1 per s quark
    Y_Le: float = 0.0       # electron family, (n_e + n_nue)/n_B

    # Chemical potentials (MeV)
    mu_u: float = 0.0
    mu_d: float = 0.0
    mu_s: float = 0.0
    mu_e: float = 0.0
    mu_nu: float = 0.0      # electron neutrino
    mu_B: float = 0.0
    mu_C: float = 0.0
    mu_S: float = 0.0

    # Densities (fm^-3), net: antiparticles subtracted
    n_u: float = 0.0
    n_d: float = 0.0
    n_s: float = 0.0
    n_e: float = 0.0
    n_nu: float = 0.0

    # Thermodynamics (MeV/fm^3 for P, e and f; fm^-3 for s)
    P_total: float = 0.0
    e_total: float = 0.0
    s_total: float = 0.0
    f_total: float = 0.0

    # Fractions, per baryon
    Y_u: float = 0.0
    Y_d: float = 0.0
    Y_s: float = 0.0
    Y_e: float = 0.0
    Y_nu: float = 0.0


@dataclass
class CFLPoint:
    """One solved colour-flavour locked state.

    The same fields as `EoSPoint` plus the gap it was solved at, and the same
    convergence contract: `error` is the largest scaled residual and
    `converged` is that against `RESIDUAL_TOL`. The phase carries no electrons
    -- flavour locking makes it neutral by construction -- so n_e and Y_e are
    absent rather than zero-valued.
    """
    # Convergence info
    converged: bool = False
    error: float = 0.0

    # Input conditions
    n_B: float = 0.0        # baryon density (fm^-3)
    T: float = 0.0          # temperature (MeV)
    Delta0: float = 0.0     # zero-temperature gap (MeV)
    Delta: float = 0.0      # gap at temperature T (MeV)
    Y_C: float = 0.0        # charge fraction, zero by construction
    Y_S: float = 0.0        # strangeness fraction

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
    Y_e: float = 0.0
    Y_nu: float = 0.0


# =============================================================================
# COLD GUESSES
# =============================================================================
def default_guess(mode: str, n_B: float, T: float, par: Parameters,
                  Y_C: float = None,
                  two_flavour: bool = False) -> np.ndarray:
    """The cold start of one mode.

    All of them start from the massless relation n = mu^3/(pi^2 (hbar c)^3) at
    roughly one flavour per baryon, mu ~ (pi^2 n_B)^(1/3) hbar c, floored at
    50 MeV so that a vanishing density does not produce a vanishing potential.
    The modes then differ only in how the three flavours are spread around it:
    beta equilibrium puts mu_s = mu_d slightly above mu_u, and a fixed charge
    fraction raises mu_u in proportion to Y_C.

    The layouts are the unknown vectors of each mode's residual, so a guess is
    only valid within its own mode.
    """
    mu_estimate = (n_B * PI2)**(1.0/3.0) * hc
    mu_estimate = max(mu_estimate, 50.0)

    if mode == "cfl":
        mu_est = 300.0 * (n_B / 0.4)**(1.0/3.0)
        return np.array([mu_est, mu_est, mu_est])

    if mode in ("fixed_YC", "fixed_YC_YS"):
        mu_d = mu_estimate * 1.1
        mu_s = mu_d
        mu_u = mu_d * (1.0 + 0.5 * Y_C)
        if two_flavour:
            return np.array([mu_u, mu_d])
        return np.array([mu_u, mu_d, mu_s])

    # Beta equilibrium: mu_d = mu_s by strangeness equilibrium, and mu_u sits
    # one electron potential below them.
    mu_d = mu_estimate * 1.1
    mu_s = mu_d
    mu_e = mu_d * 0.1
    mu_u = mu_d - mu_e
    if mode == "beta_eq_neutrinoless":
        if two_flavour:
            return np.array([mu_u, mu_d, mu_e])
        return np.array([mu_u, mu_d, mu_s, mu_e])
    if mode == "beta_eq_neutrino_trapped":
        # A trapped electron family holds mu_nue positive and of the order of
        # the electron potential itself.
        if two_flavour:
            return np.array([mu_u, mu_d, mu_e, mu_e])
        return np.array([mu_u, mu_d, mu_s, mu_e, mu_e])
    raise ValueError(f"unknown mode {mode!r}; expected one of "
                     f"{list(MODE_FRACTIONS)}")


# =============================================================================
# ASSEMBLY OF A SOLVED POINT
# =============================================================================
def point_from_mu(
    mu_u: float, mu_d: float, mu_s: float, mu_e: float,
    T: float, par: Parameters, flags: SpeciesFlags,
    mu_nu: float = 0.0,
    converged: bool = True,
    error: float = 0.0
) -> EoSPoint:
    """The totals of an unpaired state at given potentials.

    The quark sector and the bag come from `thermodynamics.thermo_from_mu`;
    what is added here is everything that carries no conserved charge of the
    strongly-interacting matter:

        electrons (with positrons) at mu_e
        electron neutrinos (with antineutrinos) at mu_nu, where it is nonzero
        photons, gluons, and the untracked neutrino flavours at mu = 0

    Three thermal flavours are counted where the electron neutrino is
    free-streaming (mu_nu = 0) and two where it is trapped, since the trapped
    flavour is already carried at its own potential.

    Args:
        mu_u, mu_d, mu_s: quark chemical potentials (MeV)
        mu_e: electron chemical potential (MeV)
        T: temperature (MeV)
        par: the parameter set
        flags: the active sectors -- `photons`, `gluons` and
            `thermal_neutrinos` are the phase-common gases;
            `two_flavour` takes the s flavour out of the matter
        mu_nu: electron-neutrino chemical potential (MeV)
        converged, error: the status of the solve this came from

    Returns:
        EoSPoint
    """
    two_flavour = flags.two_flavour
    quark = thermo_from_mu(mu_u, mu_d, mu_s, T, par, two_flavour)

    thermo_e = electron_thermo(mu_e, T)

    P_total = quark.P + thermo_e.P
    e_total = quark.e + thermo_e.e
    s_total = quark.s + thermo_e.s

    n_nu = 0.0
    if mu_nu != 0.0:
        thermo_nu = neutrino_thermo(mu_nu, T)
        n_nu = thermo_nu.n
        P_total += thermo_nu.P
        e_total += thermo_nu.e
        s_total += thermo_nu.s

    if flags.photons:
        thermo_gamma = photon_thermo(T)
        P_total += thermo_gamma.P
        e_total += thermo_gamma.e
        s_total += thermo_gamma.s

    if flags.gluons:
        thermo_g = gluon_thermo(T, par.alpha)
        P_total += thermo_g.P
        e_total += thermo_g.e
        s_total += thermo_g.s

    if flags.thermal_neutrinos:
        thermo_nu_th = neutrino_thermo(0.0, T)
        n_thermal_flavors = 2.0 if mu_nu != 0.0 else 3.0
        P_total += n_thermal_flavors * thermo_nu_th.P
        e_total += n_thermal_flavors * thermo_nu_th.e
        s_total += n_thermal_flavors * thermo_nu_th.s

    n_B = quark.n_B
    Y_u = quark.n_u / n_B if n_B > 0 else 0.0
    Y_d = quark.n_d / n_B if n_B > 0 else 0.0
    Y_s = quark.n_s / n_B if n_B > 0 else 0.0
    Y_e = thermo_e.n / n_B if n_B > 0 else 0.0
    Y_nu = n_nu / n_B if n_B > 0 else 0.0
    n_Le, _ = lepton_charges(n_e=thermo_e.n, n_nue=n_nu)
    Y_Le = n_Le / n_B if n_B > 0 else 0.0

    f_total = e_total - T * s_total

    return EoSPoint(
        converged=converged,
        error=error,
        n_B=n_B, T=T, Y_C=quark.Y_C, Y_S=quark.Y_S,
        mu_u=mu_u, mu_d=mu_d, mu_s=mu_s, mu_e=mu_e, mu_nu=mu_nu,
        mu_B=quark.mu_B, mu_C=quark.mu_C, mu_S=quark.mu_S,
        n_u=quark.n_u, n_d=quark.n_d, n_s=quark.n_s, n_e=thermo_e.n, n_nu=n_nu,
        P_total=P_total, e_total=e_total, s_total=s_total, f_total=f_total,
        Y_u=Y_u, Y_d=Y_d, Y_s=Y_s, Y_e=Y_e, Y_nu=Y_nu, Y_Le=Y_Le
    )


def cfl_point_from_mu(
    mu_u: float, mu_d: float, mu_s: float, mu_e: float,
    T: float, Delta0: float, par: Parameters, flags: SpeciesFlags,
    mu_nu: float = 0.0,
) -> CFLPoint:
    """The totals of a paired state at given potentials, WITH a lepton gas.

    `point_from_mu` for the CFL phase. It carries the electron gas because a
    caller comparing a CFL droplet against a hadronic phase under GLOBAL
    charge neutrality needs the leptons of the whole system attached to one of
    the two, even though the paired phase is locally neutral on its own.
    `solve_cfl` does not go through here for that reason: it returns the
    phase alone.
    """
    cfl = cfl_thermo_from_mu(mu_u, mu_d, mu_s, T, Delta0, par)
    thermo_e = electron_thermo(mu_e, T)

    P_total = cfl.P + thermo_e.P
    e_total = cfl.e + thermo_e.e
    s_total = cfl.s + thermo_e.s

    n_nu = 0.0
    if mu_nu != 0.0:
        thermo_nu = neutrino_thermo(mu_nu, T)
        n_nu = thermo_nu.n
        P_total += thermo_nu.P
        e_total += thermo_nu.e
        s_total += thermo_nu.s

    if flags.photons:
        thermo_gamma = photon_thermo(T)
        P_total += thermo_gamma.P
        e_total += thermo_gamma.e
        s_total += thermo_gamma.s

    if flags.gluons:
        thermo_g = gluon_thermo(T, par.alpha)
        P_total += thermo_g.P
        e_total += thermo_g.e
        s_total += thermo_g.s

    if flags.thermal_neutrinos:
        thermo_nu_th = neutrino_thermo(0.0, T)
        n_thermal_flavors = 2.0 if mu_nu != 0.0 else 3.0
        P_total += n_thermal_flavors * thermo_nu_th.P
        e_total += n_thermal_flavors * thermo_nu_th.e
        s_total += n_thermal_flavors * thermo_nu_th.s

    n_B = cfl.n_B
    Y_e = thermo_e.n / n_B if n_B > 0 else 0.0
    Y_nu = n_nu / n_B if n_B > 0 else 0.0

    f_total = e_total - T * s_total

    return CFLPoint(
        converged=True,
        n_B=n_B, T=T, Delta0=Delta0, Delta=cfl.Delta,
        Y_C=cfl.Y_C, Y_S=cfl.Y_S,
        mu_u=mu_u, mu_d=mu_d, mu_s=mu_s, mu_e=mu_e, mu_nu=mu_nu,
        mu_B=cfl.mu_B, mu_C=cfl.mu_C, mu_S=cfl.mu_S,
        n_u=cfl.n_u, n_d=cfl.n_d, n_s=cfl.n_s,
        P_total=P_total, e_total=e_total, s_total=s_total, f_total=f_total,
        Y_u=cfl.Y_u, Y_d=cfl.Y_d, Y_s=cfl.Y_s, Y_e=Y_e, Y_nu=Y_nu,
    )


# =============================================================================
# SOLVER: BETA EQUILIBRIUM, NEUTRINOS FREE-STREAMING
# =============================================================================
def solve_beta_eq_neutrinoless(
    par: Parameters, n_B: float, T: float, flags: SpeciesFlags,
    initial_guess: Optional[np.ndarray] = None
) -> EoSPoint:
    """Charge-neutral beta equilibrium with mu_nue = 0.

    Four rows for the unknowns x = [mu_u, mu_d, mu_s, mu_e]:

        r1 = (n_u + n_d + n_s)/3 - n_B     baryon number
        r2 = n_C - n_e(mu_e, T)            total electric neutrality
        r3 = mu_d - mu_u - mu_e            beta equilibrium, mu_C + mu_e = 0
        r4 = mu_s - mu_d                   strangeness equilibrium, mu_S = 0

    with n_C = (2 n_u - n_d - n_s)/3. Row r3 is d <-> u + e- + nubar_e with
    the neutrinos free-streaming, r4 is s <-> d.

    TWO-FLAVOUR MATTER IS THIS MODE WITH THE STRANGE SECTOR OFF
    (`two_flavour`, `eos.alphabag.SpeciesFlags`), which is what two-flavour
    quark matter physically is. Row r4 and the unknown mu_s go together: a
    flavour that is not in the matter has no potential to solve for, and
    holding Y_S = 0 instead would leave mu_S undetermined with a null Jacobian
    column, which is the route CLAUDE.md section 4 forbids. So the vector is
    x = [mu_u, mu_d, mu_e] and three rows close it. mu_s is reported as the
    weak-equilibrium value mu_d -- the s <-> d relation still holds, nothing
    is populated at it -- so mu_S vanishes with the strangeness it is
    conjugate to.

    Args:
        n_B: baryon density (fm^-3)
        T: temperature (MeV)
        par: model parameters; required (CLAUDE.md section 6)
        flags: the active sectors -- `photons`, `gluons` and
            `thermal_neutrinos` are the phase-common gases;
            `two_flavour` takes the s flavour out of the matter
        initial_guess: warm start in the layout above

    Returns:
        EoSPoint; test `.converged` before using any other field.
    """

    two_flavour = flags.two_flavour
    alpha = par.alpha
    m_u, m_d, m_s = par.m_u, par.m_d, par.m_s

    cold = default_guess("beta_eq_neutrinoless", n_B, T, par,
                         two_flavour=two_flavour)
    x0 = cold if initial_guess is None else initial_guess
    x0_fallback = None if initial_guess is None else cold

    def residual(x):
        if two_flavour:
            mu_u, mu_d, mu_e = x
            mu_s = mu_d
        else:
            mu_u, mu_d, mu_s, mu_e = x

        n_u = quark_density(mu_u, T, m_u, alpha)
        n_d = quark_density(mu_d, T, m_d, alpha)
        n_s = 0.0 if two_flavour else quark_density(mu_s, T, m_s, alpha)
        n_e = electron_thermo(mu_e, T).n

        n_B_calc, n_C, _ = quark_charges(n_u, n_d, n_s)

        rows = [n_B_calc - n_B, n_C - n_e, mu_d - mu_u - mu_e]
        if not two_flavour:
            rows.append(mu_s - mu_d)
        return rows

    def scales_at(x):
        """Two densities against n_B, the potential equalities against mu_B;
        the strangeness-equilibrium row goes with the flavour."""
        mu_B = _mu_scale(x[0], x[1])
        if two_flavour:
            return [n_B, n_B, mu_B]
        return [n_B, n_B, mu_B, mu_B]

    x, error, converged = solve_system(residual, x0, scales_at, x0_fallback)
    if two_flavour:
        mu_u, mu_d, mu_e = x
        mu_s = mu_d
    else:
        mu_u, mu_d, mu_s, mu_e = x

    return point_from_mu(
        mu_u, mu_d, mu_s, mu_e, T, par, flags,
        converged=converged,
        error=error
    )


# =============================================================================
# SOLVER: BETA EQUILIBRIUM WITH TRAPPED NEUTRINOS
# =============================================================================
def solve_beta_eq_neutrino_trapped(
    par: Parameters, n_B: float, Y_Le: float, T: float, flags: SpeciesFlags,
    initial_guess: Optional[np.ndarray] = None
) -> EoSPoint:
    """Beta equilibrium with the electron family trapped at Y_Le.

    Five rows for the unknowns x = [mu_u, mu_d, mu_s, mu_e, mu_nue]: the rows
    r1, r2 and r4 of `solve_beta_eq_neutrinoless`, with

        r3 = mu_d - mu_u - mu_e + mu_nue      mu_C + mu_e - mu_nue = 0
        r5 = (n_e + n_nue)/n_B - Y_Le         lepton-family conservation

    The neutrino gas carries its antineutrinos, so n_nue is the NET density
    and Y_Le is the conserved electron-family number per baryon. The muon
    family is not tracked; Y_Lmu raises at the API boundary.

    Args:
        n_B: baryon density (fm^-3)
        Y_Le: electron-family lepton fraction (n_e + n_nue)/n_B
        T: temperature (MeV)
        par: model parameters; required (CLAUDE.md section 6)
        flags: the active sectors -- `photons`, `gluons` and
            `thermal_neutrinos` are the phase-common gases;
            `two_flavour` takes the s flavour out of the matter
        initial_guess: warm start in the layout above

    Returns:
        EoSPoint; test `.converged` before using any other field.
    """

    two_flavour = flags.two_flavour
    alpha = par.alpha
    m_u, m_d, m_s = par.m_u, par.m_d, par.m_s

    cold = default_guess("beta_eq_neutrino_trapped", n_B, T, par,
                         two_flavour=two_flavour)
    x0 = cold if initial_guess is None else initial_guess
    x0_fallback = None if initial_guess is None else cold

    def residual(x):
        if two_flavour:
            mu_u, mu_d, mu_e, mu_nu = x
            mu_s = mu_d
        else:
            mu_u, mu_d, mu_s, mu_e, mu_nu = x

        n_u = quark_density(mu_u, T, m_u, alpha)
        n_d = quark_density(mu_d, T, m_d, alpha)
        n_s = 0.0 if two_flavour else quark_density(mu_s, T, m_s, alpha)
        n_e = electron_thermo(mu_e, T).n
        n_nu = neutrino_thermo(mu_nu, T).n

        n_B_calc, n_C, _ = quark_charges(n_u, n_d, n_s)

        rows = [n_B_calc - n_B, n_C - n_e, mu_d - mu_u - mu_e + mu_nu]
        if not two_flavour:
            rows.append(mu_s - mu_d)
        rows.append((n_e + n_nu) / n_B - Y_Le)
        return rows

    def scales_at(x):
        """Two densities against n_B, the potential equalities against mu_B;
        the lepton-fraction row is already dimensionless, and the
        strangeness-equilibrium row goes with the flavour."""
        mu_B = _mu_scale(x[0], x[1])
        if two_flavour:
            return [n_B, n_B, mu_B, 1.0]
        return [n_B, n_B, mu_B, mu_B, 1.0]

    x, error, converged = solve_system(residual, x0, scales_at, x0_fallback)
    if two_flavour:
        mu_u, mu_d, mu_e, mu_nu = x
        mu_s = mu_d
    else:
        mu_u, mu_d, mu_s, mu_e, mu_nu = x

    point = point_from_mu(
        mu_u, mu_d, mu_s, mu_e, T, par, flags,
        mu_nu=mu_nu,
        converged=converged,
        error=error
    )
    return point


# =============================================================================
# SOLVER: FIXED CHARGE FRACTION
# =============================================================================
def solve_fixed_yc(
    par: Parameters, n_B: float, Y_C: float, T: float, flags: SpeciesFlags,
    leptons: bool = False,
    initial_guess: Optional[np.ndarray] = None
) -> EoSPoint:
    """Fixed non-leptonic charge fraction, with strangeness equilibrium.

    Three rows for the unknowns x = [mu_u, mu_d, mu_s]:

        r1  = (n_u + n_d + n_s)/3 - n_B     baryon number
        r2' = n_C/n_B_calc - Y_C            the charge fraction, imposed
        r4  = mu_s - mu_d                   strangeness equilibrium, mu_S = 0

    Y_C is the NON-leptonic charge fraction; total electric neutrality is a
    separate, additional condition. With `leptons=False` the result
    is electrically CHARGED quark matter, which is what a mixed-phase
    construction needs per pure phase before global neutrality is imposed.

    With `leptons=True` a neutralizing electron gas is added AFTER
    the solve, by inverting n_e(mu_e, T) = n_C for mu_e -- a one-dimensional
    inversion rather than a fourth row, because the quark sector does not
    respond to mu_e at fixed Y_C. The two calls therefore share the same quark
    state and differ only in the lepton contribution to P, eps and s.

    Args:
        n_B: baryon density (fm^-3)
        Y_C: non-leptonic charge fraction
        T: temperature (MeV)
        par: model parameters; required (CLAUDE.md section 6)
        flags: the active sectors -- `photons`, `gluons` and
            `thermal_neutrinos` are the phase-common gases;
            `two_flavour` takes the s flavour out of the matter
        leptons: add the neutralizing electron gas
        initial_guess: warm start in the layout above

    Returns:
        EoSPoint; test `.converged` before using any other field.
    """

    two_flavour = flags.two_flavour
    alpha = par.alpha
    m_u, m_d, m_s = par.m_u, par.m_d, par.m_s

    cold = default_guess("fixed_YC", n_B, T, par, Y_C=Y_C,
                         two_flavour=two_flavour)
    x0 = cold if initial_guess is None else initial_guess
    x0_fallback = None if initial_guess is None else cold

    def residual(x):
        if two_flavour:
            mu_u, mu_d = x
            mu_s = mu_d
        else:
            mu_u, mu_d, mu_s = x

        n_u = quark_density(mu_u, T, m_u, alpha)
        n_d = quark_density(mu_d, T, m_d, alpha)
        n_s = 0.0 if two_flavour else quark_density(mu_s, T, m_s, alpha)

        n_B_calc, n_C, _ = quark_charges(n_u, n_d, n_s)
        Y_C_calc = n_C / n_B_calc if n_B_calc > 0 else 0.0

        rows = [n_B_calc - n_B, Y_C_calc - Y_C]
        if not two_flavour:
            rows.append(mu_s - mu_d)
        return rows

    def scales_at(x):
        """One density against n_B, one potential equality against mu_B; the
        charge-fraction row is already dimensionless, and the equality goes
        with the flavour."""
        if two_flavour:
            return [n_B, 1.0]
        return [n_B, 1.0, _mu_scale(x[0], x[1])]

    x, error, converged = solve_system(residual, x0, scales_at, x0_fallback)
    if two_flavour:
        mu_u, mu_d = x
        mu_s = mu_d
    else:
        mu_u, mu_d, mu_s = x
    mu_e = _neutralizing_mu_e(mu_u, mu_d, mu_s, T, par, leptons,
                              initial_guess, two_flavour)

    return point_from_mu(
        mu_u, mu_d, mu_s, mu_e, T, par, flags,
        converged=converged,
        error=error
    )


# =============================================================================
# SOLVER: FIXED CHARGE AND STRANGENESS FRACTIONS
# =============================================================================
def solve_fixed_yc_ys(
    par: Parameters, n_B: float, Y_C: float, Y_S: float, T: float,
    flags: SpeciesFlags,
    leptons: bool = False,
    initial_guess: Optional[np.ndarray] = None
) -> EoSPoint:
    """Fixed charge AND strangeness fractions -- no strangeness equilibrium.

    Three rows for the unknowns x = [mu_u, mu_d, mu_s]:

        r1  = (n_u + n_d + n_s)/3 - n_B     baryon number
        r2' = n_C/n_B_calc - Y_C            the charge fraction, imposed
        r3' = n_s/n_B_calc - Y_S            the strangeness fraction, imposed

    The third row replaces strangeness equilibrium: mu_S is then an output,
    not zero. This is the mode that separates a bag model from a nucleonic
    one -- it is meaningful here, where the s quark is a degree of freedom,
    and raises in `eos.zl`, which has none.

    Leptons are handled as in `solve_fixed_yc`.

    Args:
        n_B: baryon density (fm^-3)
        Y_C: non-leptonic charge fraction
        Y_S: strangeness fraction, S = +1 per s quark
        T: temperature (MeV)
        par: model parameters; required (CLAUDE.md section 6)
        flags: the active sectors -- `photons`, `gluons` and
            `thermal_neutrinos` are the phase-common gases;
            `two_flavour` takes the s flavour out of the matter
        leptons: add the neutralizing electron gas
        initial_guess: warm start in the layout above

    THIS IS THE ONE UNPAIRED MODE THAT REFUSES `two_flavour`, and the refusal
    is the same statement the flag makes. This mode holds Y_S; with the
    strange sector off no species left in the state carries strangeness, so
    row r3' reads 0 = Y_S -- unsatisfiable for Y_S != 0, and for Y_S = 0
    satisfied at every mu_S at once, which leaves mu_S undetermined and its
    Jacobian column null. Reaching two-flavour matter by asking for Y_S = 0 is
    exactly the route CLAUDE.md section 4 forbids; it is reached by switching
    the sector off, in `beta_eq_neutrinoless`.

    Returns:
        EoSPoint; test `.converged` before using any other field.
    """
    two_flavour = flags.two_flavour
    if two_flavour:
        raise NotImplementedError(
            "solve_fixed_yc_ys: this mode holds Y_S, and with the strange "
            "sector off there is no species left to carry strangeness -- the "
            "row is unsatisfiable for Y_S != 0 and leaves mu_S undetermined "
            "for Y_S = 0. Two-flavour quark matter is 'beta_eq_neutrinoless' "
            "with SpeciesFlags(two_flavour=True), never fixed_YC_YS at "
            "Y_S = 0")

    alpha = par.alpha
    m_u, m_d, m_s = par.m_u, par.m_d, par.m_s

    cold = default_guess("fixed_YC_YS", n_B, T, par, Y_C=Y_C)
    x0 = cold if initial_guess is None else initial_guess
    x0_fallback = None if initial_guess is None else cold

    def residual(x):
        mu_u, mu_d, mu_s = x

        n_u = quark_density(mu_u, T, m_u, alpha)
        n_d = quark_density(mu_d, T, m_d, alpha)
        n_s = quark_density(mu_s, T, m_s, alpha)

        n_B_calc, n_C, _ = quark_charges(n_u, n_d, n_s)
        Y_C_calc = n_C / n_B_calc if n_B_calc > 0 else 0.0
        Y_S_calc = n_s / n_B_calc if n_B_calc > 0 else 0.0

        return [
            n_B_calc - n_B,
            Y_C_calc - Y_C,
            Y_S_calc - Y_S,
        ]

    def scales_at(x):
        """One density against n_B; both fraction rows are already
        dimensionless."""
        return [n_B, 1.0, 1.0]

    x, error, converged = solve_system(residual, x0, scales_at, x0_fallback)
    mu_u, mu_d, mu_s = x
    mu_e = _neutralizing_mu_e(mu_u, mu_d, mu_s, T, par, leptons,
                              initial_guess)

    return point_from_mu(
        mu_u, mu_d, mu_s, mu_e, T, par, flags,
        converged=converged,
        error=error
    )


def _neutralizing_mu_e(mu_u, mu_d, mu_s, T, par, leptons,
                       initial_guess, two_flavour=False):
    """mu_e of the electron gas that neutralises the solved quark charge.

    Shared by the two fixed-fraction modes, which close the same way: the
    quark state is already determined, so the electrons follow by inverting
    n_e(mu_e, T) = n_C once. Returns 0.0 where the phase is left charged.
    """
    if not leptons:
        return 0.0

    alpha = par.alpha
    n_u = quark_density(mu_u, T, par.m_u, alpha)
    n_d = quark_density(mu_d, T, par.m_d, alpha)
    n_s = 0.0 if two_flavour else quark_density(mu_s, T, par.m_s, alpha)
    _, n_C, _ = quark_charges(n_u, n_d, n_s)

    # A warm start carrying one entry past the flavour potentials brought mu_e
    # with it; there are two flavour potentials with the strange sector off
    # and three with it on.
    n_flavours = 2 if two_flavour else 3
    mu_e_guess = None
    if initial_guess is not None and len(initial_guess) > n_flavours:
        mu_e_guess = initial_guess[n_flavours]

    return electron_thermo_from_density(n_C, T, mu_e_guess=mu_e_guess).mu


# =============================================================================
# THE PAIRED PHASE
# =============================================================================
def solve_cfl(
    par: Parameters, n_B: float, T: float, Delta0: float,
    flags: SpeciesFlags,
    initial_guess: Optional[np.ndarray] = None
) -> CFLPoint:
    """Colour-flavour locked quark matter at a given density and gap.

    Not one of the equilibrium modes: the phase is closed by flavour locking
    rather than by an equilibrium condition. The condensate pairs the three
    flavours in equal numbers, so three rows for x = [mu_u, mu_d, mu_s]:

        r_q = n_q(mu_q, T, m_q, alpha) + 2 mu_q Delta^2/(pi^2 (hbar c)^3)
              - n_B = 0 ,      q = u, d, s

    Two consequences worth stating because they are easy to expect wrongly.
    The phase is ELECTRICALLY NEUTRAL BY CONSTRUCTION -- n_C = (2n_u - n_d -
    n_s)/3 vanishes identically at equal densities -- so it carries no
    electrons and Y_C comes back as round-off rather than solved. And it is
    NOT in strangeness equilibrium: equal densities at unequal masses need
    unequal potentials, so mu_S = mu_s - mu_d is nonzero and the energy per
    baryon at P = 0 is mu_B + mu_S, not mu_B.

    Which massless boson gases the locked phase has is decided by the
    condensate, not by the caller. Colour-flavour locking breaks
    SU(3)_c x U(1)_Q down to a single unbroken U(1)_Qtilde generated by a
    mixture of the electric charge and the eighth colour generator, so of the
    nine gauge bosons exactly ONE stays massless -- the rotated photon, to
    which the medium is transparent -- and the other eight acquire Meissner
    masses. Hence:

      - `flags.photons` is the rotated photon. At leading order it is a free
        massless gas of two polarizations, thermodynamically the same
        `photon_thermo(T)` the unpaired phase carries, so the sector is
        available here exactly as it is there.
      - `flags.gluons` is REFUSED. The gluon gas of the unpaired phase is
        g_g = 2 x 8 = 16 massless bosons; in this phase every one of those
        degrees of freedom is massive, so adding that closed form here would
        assume the masslessness the condensate removes. `gluons=True`
        raises rather than returning a number the phase does not have.

    The light bosonic sector the locked phase DOES have -- the superfluid
    phonon and the pseudo-Goldstone meson octet -- is not implemented; see
    docs/DEFERRED.md. It is a different sector, not a mass added to this one.

    The thermal neutrino flavours are NOT added here either, unlike in the
    unpaired solvers; see docs/DEFERRED.md.

    Args:
        n_B: baryon density (fm^-3)
        T: temperature (MeV)
        Delta0: zero-temperature pairing gap (MeV)
        par: model parameters; required (CLAUDE.md section 6)
        flags: `photons` is the rotated photon, as above; `gluons`,
            `thermal_neutrinos` and `two_flavour` are refused, each with the
            reason above
        initial_guess: warm start [mu_u, mu_d, mu_s]

    Returns:
        CFLPoint; test `.converged` before using any other field.
    """
    if flags.gluons:
        raise NotImplementedError(
            "solve_cfl: the colour-flavour-locked phase has no free thermal "
            "gluon gas -- all eight gluons are Meissner-massive, only the "
            "rotated photon stays massless. Use gluons=False here "
            "(the unpaired solvers carry the sector); the phonon and "
            "pseudo-Goldstone octet that ARE light in this phase are not "
            "implemented, see docs/DEFERRED.md")
    if flags.two_flavour:
        raise NotImplementedError(
            "solve_cfl: colour-flavour locking pairs the three flavours at "
            "equal densities, so Y_S = +1 identically and there is no "
            "strangeness fraction free to switch off. two_flavour is refused "
            "here for the same reason gluons is: the flag keeps both its "
            "values in the unpaired modes and is a statement about the phase "
            "in this one. Two-flavour quark matter is "
            "'beta_eq_neutrinoless' with two_flavour=True")
    if flags.thermal_neutrinos:
        raise NotImplementedError(
            "solve_cfl: the paired phase carries no thermal neutrino gas, "
            "where every unpaired solver adds one. The asymmetry is "
            "inherited from the first-generation CFL table builder and is "
            "preserved deliberately, because closing it moves published CFL "
            "tables; see docs/DEFERRED.md. Use thermal_neutrinos=False in "
            "the 'cfl' mode")

    alpha = par.alpha
    m_u, m_d, m_s = par.m_u, par.m_d, par.m_s

    cold = default_guess("cfl", n_B, T, par)
    x0 = cold if initial_guess is None else initial_guess
    x0_fallback = None if initial_guess is None else cold

    def paired_density(mu, m):
        """One flavour's density in the condensate, Eq. (n_q) of alphabag.tex."""
        return (quark_density(mu, T, m, alpha)
                + cfl_n_correction(mu, T, Delta0, par.tc_coeff))

    def residual(x):
        mu_u, mu_d, mu_s = x
        return np.array([
            paired_density(mu_u, m_u) - n_B,
            paired_density(mu_d, m_d) - n_B,
            paired_density(mu_s, m_s) - n_B,
        ])

    def scales_at(x):
        """Every row of the locked phase balances a flavour density."""
        return [n_B, n_B, n_B]

    x, error, converged = solve_system(residual, x0, scales_at, x0_fallback)
    mu_u, mu_d, mu_s = x

    cfl = cfl_thermo_from_mu(mu_u, mu_d, mu_s, T, Delta0, par)

    P_total = cfl.P
    e_total = cfl.e
    s_total = cfl.s

    if flags.photons and T > 0:
        photon = photon_thermo(T)
        P_total += photon.P
        e_total += photon.e
        s_total += photon.s

    f_total = e_total - T * s_total

    return CFLPoint(
        converged=converged,
        error=error,
        n_B=cfl.n_B, T=T, Delta0=Delta0, Delta=cfl.Delta,
        Y_C=cfl.Y_C, Y_S=cfl.Y_S,
        mu_u=mu_u, mu_d=mu_d, mu_s=mu_s,
        mu_B=cfl.mu_B, mu_C=cfl.mu_C, mu_S=cfl.mu_S,
        n_u=cfl.n_u, n_d=cfl.n_d, n_s=cfl.n_s,
        P_total=P_total, e_total=e_total, s_total=s_total, f_total=f_total,
        Y_u=cfl.Y_u, Y_d=cfl.Y_d, Y_s=cfl.Y_s
    )


# =============================================================================
# WARM START
# =============================================================================
def warm_start(point, mode: str, two_flavour: bool = False) -> np.ndarray:
    """The seed the next density takes from a solved point.

    The layouts are the unknown vectors of each mode's residual, so a warm
    start is only valid within its own mode. The fixed-fraction modes and the
    paired phase carry three entries and no mu_e: the electron gas of a fixed
    fraction is not an unknown but a one-dimensional inversion after the
    solve, and the paired phase has no electrons at all.

    Along a density sweep the potentials vary smoothly, which is what makes
    the continuation work.

    With `two_flavour` the vector is one entry shorter in every unpaired
    mode: mu_s is not solved for, the flavour not being in the matter.
    """
    if mode == "beta_eq_neutrinoless":
        if two_flavour:
            return np.array([point.mu_u, point.mu_d, point.mu_e])
        return np.array([point.mu_u, point.mu_d, point.mu_s, point.mu_e])
    if mode == "beta_eq_neutrino_trapped":
        if two_flavour:
            return np.array([point.mu_u, point.mu_d, point.mu_e, point.mu_nu])
        return np.array([point.mu_u, point.mu_d, point.mu_s, point.mu_e,
                         point.mu_nu])
    if mode in ("fixed_YC", "fixed_YC_YS", "cfl"):
        if two_flavour:
            return np.array([point.mu_u, point.mu_d])
        return np.array([point.mu_u, point.mu_d, point.mu_s])
    raise ValueError(f"unknown mode {mode!r}; expected one of "
                     f"{list(MODE_FRACTIONS)}")
