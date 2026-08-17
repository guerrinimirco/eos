"""Single-point equilibrium solvers for ZL nucleonic matter.

One solver per equilibrium mode. Every mode enforces the self-consistency
n_i(mu_eff_i, T, m_i) = n_i for protons and neutrons and fixes the baryon
density; the mode supplies the rest:

    beta equilibrium (neutrinoless)   mu_C + mu_e = 0, n_C = n_e
    beta equilibrium (trapped)        ... with mu_nue kept and Y_Le fixed
    fixed Y_C                         n_p = Y_C n_B, n_n = (1-Y_C) n_B
    fixed Y_C and Y_S                 RAISES: n_S = 0 identically here

The unknowns are the physical potentials, extended by mu_e where a lepton
condition is present, by mu_nue in the trapped mode, and by the densities
(n_p, n_n) wherever the composition is not fixed in advance. Keeping the
densities as unknowns rather than substituting the self-consistency into
mu_Hv_i(n_p, n_n) is what makes the residual polynomial in the interaction
potentials instead of nesting the Fermi integrals inside them; it is a
conditioning choice, not a statement about the state, which is (mu_p, mu_n, T).

Reading order: the cold guesses, the shared solve and its gate, then one
function per mode, then the warm start.

The thermodynamic kernels are in `thermodynamics.py`, the table driver in
`table.py`, the spec API in `api.py`. See `zl.tex` for the physics.

Usage:
    from eos.zl.solver import solve_beta_eq_neutrinoless
    result = solve_beta_eq_neutrinoless(n_B=0.16, T=10.0)
    print(result.converged, result.P_total)
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional
from scipy.optimize import root

from eos.general.physics_constants import hc, PI2
from eos.general.thermodynamics_leptons import (
    electron_thermo, neutrino_thermo, photon_thermo,
)
from eos.zl.parameters import Parameters
from eos.zl.thermodynamics import effective_state, thermo_from_mu_n

#: The modes this model closes, and the fractions each one consumes. The names
#: are the repository's, shared with every other model, so the same point is
#: requested from any of them the same way.
MODE_FRACTIONS = {
    "beta_eq_neutrinoless": (),
    "beta_eq_neutrino_trapped": ("Y_Le",),
    "fixed_YC": ("Y_C",),
}

#: Post-solve gate on the sum of squares of the raw equilibrium residuals.
RESIDUAL_TOL = 0.01


# =============================================================================
# RESULT
# =============================================================================
@dataclass
class EoSPoint:
    """One solved ZL state, with the status a caller must test first.

    `converged` is judged on `error`, the sum of squared equilibrium
    residuals, against `RESIDUAL_TOL`. When `converged` is False every other
    field holds the last iterate reached, which is not a physical state.
    """
    # Convergence info
    converged: bool = False
    error: float = 0.0     # sum of squared residuals

    # Inputs
    n_B: float = 0.0       # baryon density (fm^-3)
    T: float = 0.0         # temperature (MeV)
    Y_C: float = 0.0       # non-leptonic charge fraction
    Y_S: float = 0.0       # strangeness fraction (identically zero)
    Y_L: float = 0.0       # electron-family lepton fraction (trapped mode)

    # Chemical potentials (MeV)
    mu_p: float = 0.0
    mu_n: float = 0.0
    mu_e: float = 0.0
    mu_nu: float = 0.0     # electron neutrino
    mu_B: float = 0.0
    mu_C: float = 0.0
    mu_S: float = 0.0      # zero by convention: ZL carries no strangeness
    mu_L: float = 0.0

    # Densities (fm^-3)
    n_p: float = 0.0
    n_n: float = 0.0
    n_e: float = 0.0
    n_nu: float = 0.0

    # Thermodynamics (MeV/fm^3 for P and e, fm^-3 for s)
    P_total: float = 0.0
    e_total: float = 0.0
    s_total: float = 0.0

    # Fractions
    Y_p: float = 0.0
    Y_n: float = 0.0
    Y_e: float = 0.0


# =============================================================================
# COLD GUESSES
# =============================================================================
def default_guess(mode: str, n_B: float, T: float, params: Parameters,
                  Y_C: float = None, Y_Le: float = None,
                  leptons: bool = True) -> np.ndarray:
    """The cold start of one mode: a free gas plus a mean-field estimate.

    Each nucleon potential is estimated as sqrt(k_F^2 + m^2) at the Fermi
    momentum of an assumed composition, shifted by half the symmetric part of
    the interaction; mu_e follows from beta equilibrium or from charge
    neutrality, whichever the mode imposes. The layouts are the unknown
    vectors of each mode's residual, so a guess is only valid within its own
    mode.
    """
    m_p, m_n, n0 = params.m_p, params.m_n, params.n0

    if mode == "fixed_YC":
        n_p, n_n = Y_C * n_B, (1.0 - Y_C) * n_B
    else:
        # Proton fraction of beta-equilibrated matter: low and cold, rising
        # with temperature as the entropy of the leptons grows.
        Y_p_est = 0.05 + 0.15 * (T / 50.0)
        Y_p_est = max(0.01, min(Y_p_est, 0.5))
        n_p, n_n = Y_p_est * n_B, (1.0 - Y_p_est) * n_B

    kF_p = hc * (6.0 * PI2 * n_p / 2.0)**(1.0/3.0) if n_p > 0 else 0.0
    kF_n = hc * (6.0 * PI2 * n_n / 2.0)**(1.0/3.0) if n_n > 0 else 0.0
    mu_p_eff = np.sqrt(kF_p**2 + m_p**2) if n_p > 0 else m_p
    mu_n_eff = np.sqrt(kF_n**2 + m_n**2) if n_n > 0 else m_n

    if mode == "fixed_YC":
        if not leptons:
            return np.array([mu_p_eff, mu_n_eff])
        # n_e = n_p is what charge neutrality will ask for.
        kF_e = hc * (3 * PI2 * n_p)**(1.0/3.0) if n_p > 0 else 0.0
        m_e = 0.511
        mu_e = np.sqrt(kF_e**2 + m_e**2) if n_p > 0 else m_e
        return np.array([mu_p_eff, mu_n_eff, mu_e])

    # The mean field, estimated from the symmetric part of the interaction and
    # split evenly between the two species.
    V_est = 4.0 * n_p * n_n * (
        params.a0 + params.b0 * (n_B/n0)**(params.gamma - 1)) / n0
    mu_p_est = mu_p_eff + V_est * 0.5
    mu_n_est = mu_n_eff + V_est * 0.5
    mu_e_est = max(0.0, mu_n_est - mu_p_est)

    if mode == "beta_eq_neutrinoless":
        return np.array([mu_p_est, mu_n_est, mu_e_est, n_p, n_n])
    if mode == "beta_eq_neutrino_trapped":
        return np.array([mu_p_est, mu_n_est, mu_e_est, 10.0, n_p, n_n])
    raise ValueError(f"unknown mode {mode!r}; expected one of "
                     f"{list(MODE_FRACTIONS)}")


# =============================================================================
# THE SOLVE AND ITS GATE
# =============================================================================
def solve_system(residual, x0):
    """Solve one equilibrium system: Powell's hybrid method, then, if it
    reports failure, Levenberg-Marquardt. Both are bounded internally.

    Returns (x, sum of squared residuals, converged).
    """
    sol = root(residual, x0, method='hybr')
    if not sol.success:
        sol = root(residual, x0, method='lm')
    error = sum(r**2 for r in residual(sol.x))
    return sol.x, error, bool(error < RESIDUAL_TOL)


def _finish(result, mu_p, mu_n, n_p, n_n, T, params, include_photons,
            e_thermo=None, nu_thermo=None):
    """Assemble the totals of a solved state: matter, leptons, photons."""
    matter = thermo_from_mu_n(mu_p, mu_n, n_p, n_n, T, params)

    result.P_total = matter.P
    result.e_total = matter.e
    result.s_total = matter.s
    for gas in (e_thermo, nu_thermo):
        if gas is not None:
            result.P_total += gas.P
            result.e_total += gas.e
            result.s_total += gas.s
    if include_photons:
        gamma = photon_thermo(T)
        result.P_total += gamma.P
        result.e_total += gamma.e
        result.s_total += gamma.s

    result.mu_B = matter.mu_B
    result.mu_C = matter.mu_C
    return result


# =============================================================================
# SOLVER: BETA EQUILIBRIUM, NEUTRINOS FREE-STREAMING
# =============================================================================
def solve_beta_eq_neutrinoless(
    n_B: float, T: float, params: Parameters = None,
    include_photons: bool = True,
    initial_guess: Optional[np.ndarray] = None
) -> EoSPoint:
    """Charge-neutral beta equilibrium with mu_nue = 0.

    Five rows for the unknowns x = [mu_p, mu_n, mu_e, n_p, n_n]:

        r1 = n_p(mu_eff_p, T, m_p) - n_p     self-consistency
        r2 = n_n(mu_eff_n, T, m_n) - n_n     self-consistency
        r3 = n_p + n_n - n_B                 baryon number
        r4 = mu_n - mu_p - mu_e              beta equilibrium, mu_C + mu_e = 0
        r5 = n_p - n_e(mu_e, T)              total electric neutrality

    Rows r1 and r2 take the place of the field equations of a mean-field
    model: the interaction potentials mu_Hv_i(n_p, n_n) are what the densities
    have to reproduce.

    Args:
        n_B: baryon density (fm^-3)
        T: temperature (MeV)
        params: model parameters (the published set if None)
        include_photons: add a thermal photon gas to eps, P and s
        initial_guess: warm start in the layout above

    Returns:
        EoSPoint; test `.converged` before using any other field.
    """
    if params is None:
        params = Parameters.default()

    result = EoSPoint(n_B=n_B, T=T)
    x0 = (default_guess("beta_eq_neutrinoless", n_B, T, params)
          if initial_guess is None else initial_guess)

    def residual(x):
        mu_p, mu_n, mu_e, n_p, n_n = x
        state = effective_state(mu_p, mu_n, n_p, n_n, T, params)
        n_e = electron_thermo(mu_e, T, include_antiparticles=True).n
        return [state.n_p_calc - n_p,
                state.n_n_calc - n_n,
                state.n_B - n_B,
                mu_n - mu_p - mu_e,
                state.n_C - n_e]

    x, error, converged = solve_system(residual, x0)
    mu_p, mu_n, mu_e, n_p, n_n = x
    result.converged, result.error = converged, error

    result.mu_p, result.mu_n, result.mu_e = mu_p, mu_n, mu_e
    result.n_p, result.n_n = n_p, n_n
    result.Y_p = n_p / n_B
    result.Y_n = n_n / n_B
    result.Y_C = result.Y_p

    e_thermo = electron_thermo(mu_e, T, include_antiparticles=True)
    result.n_e = e_thermo.n
    result.Y_e = result.n_e / n_B

    return _finish(result, mu_p, mu_n, n_p, n_n, T, params, include_photons,
                   e_thermo=e_thermo)


# =============================================================================
# SOLVER: FIXED CHARGE FRACTION
# =============================================================================
def solve_fixed_yc(
    n_B: float, Y_C: float, T: float, params: Parameters = None,
    include_photons: bool = True,
    include_electrons: bool = True,
    initial_guess: Optional[np.ndarray] = None
) -> EoSPoint:
    """Fixed non-leptonic charge fraction, Y_C = n_p/n_B.

    The composition is known before the solve -- n_p = Y_C n_B,
    n_n = (1-Y_C) n_B -- so both densities leave the unknown vector and only
    the self-consistency rows remain:

        include_electrons=False   x = [mu_p, mu_n],        rows r1, r2
        include_electrons=True    x = [mu_p, mu_n, mu_e],  rows r1, r2 and
                                  r5 = n_e(mu_e, T) - n_p (electric neutrality)

    With `include_electrons=False` the result is electrically CHARGED
    nucleonic matter, which is what a mixed-phase construction needs per pure
    phase before global neutrality is imposed. Y_C is the non-leptonic charge
    fraction in both cases; neutrality is a separate, additional condition.

    Args:
        n_B: baryon density (fm^-3)
        Y_C: non-leptonic charge fraction
        T: temperature (MeV)
        params: model parameters (the published set if None)
        include_photons: add a thermal photon gas to eps, P and s
        include_electrons: add the neutralizing electron gas
        initial_guess: warm start in the layout above

    Returns:
        EoSPoint; test `.converged` before using any other field.
    """
    if params is None:
        params = Parameters.default()

    result = EoSPoint(n_B=n_B, T=T, Y_C=Y_C)
    n_p = Y_C * n_B
    n_n = (1.0 - Y_C) * n_B

    x0 = (default_guess("fixed_YC", n_B, T, params, Y_C=Y_C,
                        leptons=include_electrons)
          if initial_guess is None else initial_guess)

    if include_electrons:
        def residual(x):
            mu_p, mu_n, mu_e = x
            state = effective_state(mu_p, mu_n, n_p, n_n, T, params)
            n_e = electron_thermo(mu_e, T, include_antiparticles=True).n
            return [state.n_p_calc - n_p,
                    state.n_n_calc - n_n,
                    n_e - state.n_C]
    else:
        def residual(x):
            mu_p, mu_n = x
            state = effective_state(mu_p, mu_n, n_p, n_n, T, params)
            return [state.n_p_calc - n_p,
                    state.n_n_calc - n_n]

    x, error, converged = solve_system(residual, x0)
    result.converged, result.error = converged, error

    if include_electrons:
        mu_p, mu_n, mu_e = x
        e_thermo = electron_thermo(mu_e, T, include_antiparticles=True)
        result.mu_e = mu_e
        result.n_e = e_thermo.n
        result.Y_e = result.n_e / n_B
    else:
        mu_p, mu_n = x
        e_thermo = None

    result.mu_p, result.mu_n = mu_p, mu_n
    result.n_p, result.n_n = n_p, n_n
    result.Y_p, result.Y_n = Y_C, 1.0 - Y_C

    return _finish(result, mu_p, mu_n, n_p, n_n, T, params, include_photons,
                   e_thermo=e_thermo)


def solve_fixed_yc_ys(*args, **kwargs):
    """Not a solver: the mode is physically meaningless for this model.

    ZL carries protons and neutrons and nothing else, so n_S = 0 for every
    state it has. Fixing a strangeness fraction has no solution except the
    trivial Y_S = 0, and accepting the call while ignoring Y_S would return
    `fixed_YC` under a name that promised a strangeness condition.
    """
    raise NotImplementedError(
        "fixed_YC_YS is meaningless for ZL: the model has no strange degree "
        "of freedom, so n_S = 0 identically and no Y_S can be imposed. Use "
        "fixed_YC, or a model with a strange sector (eos.dd2, eos.sfho, "
        "eos.vmit).")


# =============================================================================
# SOLVER: BETA EQUILIBRIUM WITH TRAPPED NEUTRINOS
# =============================================================================
def solve_beta_eq_neutrino_trapped(
    n_B: float, Y_Le: float, T: float, params: Parameters = None,
    include_photons: bool = True,
    initial_guess: Optional[np.ndarray] = None
) -> EoSPoint:
    """Beta equilibrium with the electron family trapped at Y_Le.

    Six rows for the unknowns x = [mu_p, mu_n, mu_e, mu_nue, n_p, n_n]: the
    rows r1, r2, r3 and r5 of `solve_beta_eq_neutrinoless`, with

        r4 = mu_n - mu_p - mu_e + mu_nue      mu_C + mu_e - mu_nue = 0
        r6 = (n_e + n_nue)/n_B - Y_Le         lepton-family conservation

    The neutrino gas carries its antiparticles, so n_nue is the NET density
    and Y_Le is the conserved electron-family number per baryon. The muon
    family is not tracked; Y_Lmu raises at the API boundary.

    Args:
        n_B: baryon density (fm^-3)
        Y_Le: electron-family lepton fraction (n_e + n_nue)/n_B
        T: temperature (MeV)
        params: model parameters (the published set if None)
        include_photons: add a thermal photon gas to eps, P and s
        initial_guess: warm start in the layout above

    Returns:
        EoSPoint; test `.converged` before using any other field.
    """
    if params is None:
        params = Parameters.default()

    result = EoSPoint(n_B=n_B, T=T, Y_L=Y_Le)
    x0 = (default_guess("beta_eq_neutrino_trapped", n_B, T, params,
                        Y_Le=Y_Le)
          if initial_guess is None else initial_guess)

    def residual(x):
        mu_p, mu_n, mu_e, mu_nu, n_p, n_n = x
        state = effective_state(mu_p, mu_n, n_p, n_n, T, params)
        n_e = electron_thermo(mu_e, T, include_antiparticles=True).n
        n_nu = neutrino_thermo(mu_nu, T, include_antiparticles=True).n
        return [state.n_p_calc - n_p,
                state.n_n_calc - n_n,
                state.n_B - n_B,
                mu_n - mu_p - mu_e + mu_nu,
                state.n_C - n_e,
                (n_e + n_nu) / n_B - Y_Le]

    x, error, converged = solve_system(residual, x0)
    mu_p, mu_n, mu_e, mu_nu, n_p, n_n = x
    result.converged, result.error = converged, error

    result.mu_p, result.mu_n = mu_p, mu_n
    result.mu_e, result.mu_nu = mu_e, mu_nu
    result.n_p, result.n_n = n_p, n_n
    result.Y_p = n_p / n_B
    result.Y_n = n_n / n_B
    result.Y_C = result.Y_p

    e_thermo = electron_thermo(mu_e, T, include_antiparticles=True)
    nu_thermo = neutrino_thermo(mu_nu, T, include_antiparticles=True)
    result.n_e = e_thermo.n
    result.n_nu = nu_thermo.n
    result.Y_e = result.n_e / n_B

    return _finish(result, mu_p, mu_n, n_p, n_n, T, params, include_photons,
                   e_thermo=e_thermo, nu_thermo=nu_thermo)


# =============================================================================
# WARM START
# =============================================================================
def warm_start(point: EoSPoint, mode: str,
               leptons: bool = True) -> np.ndarray:
    """The seed the next density takes from a solved point.

    The layouts are the unknown vectors of each mode's residual, so a warm
    start is only valid within its own mode. Along a density sweep the
    potentials and densities vary smoothly, which is what makes the
    continuation work.
    """
    if mode == "beta_eq_neutrinoless":
        return np.array([point.mu_p, point.mu_n, point.mu_e,
                         point.n_p, point.n_n])
    if mode == "beta_eq_neutrino_trapped":
        return np.array([point.mu_p, point.mu_n, point.mu_e, point.mu_nu,
                         point.n_p, point.n_n])
    if mode == "fixed_YC":
        if leptons:
            return np.array([point.mu_p, point.mu_n, point.mu_e])
        return np.array([point.mu_p, point.mu_n])
    raise ValueError(f"unknown mode {mode!r}; expected one of "
                     f"{list(MODE_FRACTIONS)}")
