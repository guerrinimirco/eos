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
    from eos.zl.parameters import Parameters
    from eos.zl.species import SpeciesFlags
    result = solve_beta_eq_neutrinoless(Parameters.default(), 0.16,
                                        SpeciesFlags(photons=True), T=10.0)
    print(result.converged, result.P)
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional

from eos.general.basis import lepton_charges
from eos.general.fermi_integrals import invert_fermi_density
from eos.general.physics_constants import hc, PI2
from eos.general.solve import (
    MU_SCALE_FLOOR, RESIDUAL_TOL, scaled_residual_max, solve_system,
)
from eos.general.thermodynamics_leptons import (
    electron_thermo, neutrino_thermo, photon_thermo,
)
from eos.zl.parameters import Parameters
from eos.zl.species import SpeciesFlags
from eos.zl.thermodynamics import (
    G_NUCLEON, effective_state, interaction_potentials, thermo_from_mu_n,
)

#: The modes this model closes, and the fractions each one consumes. The names
#: are the repository's, shared with every other model, so the same point is
#: requested from any of them the same way.
MODE_FRACTIONS = {
    "beta_eq_neutrinoless": (),
    "beta_eq_neutrino_trapped": ("Y_Le",),
    "fixed_YC": ("Y_C",),
}

# =============================================================================
# RESULT
# =============================================================================
@dataclass
class EoSPoint:
    """One solved ZL state, with the status a caller must test first.

    `converged` is judged on `error`, the largest equilibrium residual after
    each has been divided by the scale of the quantity it balances (see
    `scaled_residual_max`); it is dimensionless, and the gate is
    `RESIDUAL_TOL`. When `converged` is False every other field holds the best
    iterate reached, which is not a physical state.
    """
    # Convergence info
    converged: bool = False
    error: float = 0.0     # largest scaled residual, dimensionless

    # Inputs
    n_B: float = 0.0       # baryon density (fm^-3)
    T: float = 0.0         # temperature (MeV)
    # Conserved-charge fractions, MEASURED on the solved state: Y_X = n_X/n_B
    # for every charge (CLAUDE.md section 2). A mode that HOLDS one of these
    # reports what it solved, not what it was asked for, and every one of them
    # is defined in every mode -- Y_Le included, not only in the trapped mode
    # that holds it.
    Y_C: float = 0.0       # non-leptonic charge fraction
    Y_S: float = 0.0       # strangeness fraction (identically zero)
    Y_Le: float = 0.0      # electron family, (n_e + n_nue)/n_B

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
    P: float = 0.0
    eps: float = 0.0
    s: float = 0.0

    # Fractions
    Y_p: float = 0.0
    Y_n: float = 0.0
    Y_e: float = 0.0


# =============================================================================
# COLD GUESSES
# =============================================================================
def default_guess(mode: str, n_B: float, T: float, par: Parameters,
                  Y_C: float = None, Y_Le: float = None,
                  leptons: bool = True) -> np.ndarray:
    """The cold start of one mode.

    In `fixed_YC` the composition is known, so the guess is not an estimate:
    inverting the Fermi integrals at n_i and adding mu_Hv_i(n_p, n_n) gives
    the EXACT root of the two self-consistency rows, leaving only mu_e to be
    solved for. That matters rather than merely being tidy -- mu_Hv_p reaches
    +312 MeV at n_B = 0.8 fm^-3 and Y_C = 0.1, so a guess that omits it puts
    mu_eff_p below the nucleon mass, where the T = 0 density is identically
    zero and the row has no gradient for the solver to follow.

    Where the composition is unknown the potentials are estimated as
    sqrt(k_F^2 + m^2) at the Fermi momentum of an assumed proton fraction,
    shifted by half the symmetric part of the interaction; mu_e then follows
    from beta equilibrium.

    The layouts are the unknown vectors of each mode's residual, so a guess is
    only valid within its own mode.
    """
    m_p, m_n, n0 = par.m_p, par.m_n, par.n0

    if mode == "fixed_YC":
        n_p, n_n = Y_C * n_B, (1.0 - Y_C) * n_B
        mu_Hv_p, mu_Hv_n = interaction_potentials(n_p, n_n, par)
        mu_p = invert_fermi_density(n_p, T, m_p, G_NUCLEON) + mu_Hv_p
        mu_n = invert_fermi_density(n_n, T, m_n, G_NUCLEON) + mu_Hv_n
        if not leptons:
            return np.array([mu_p, mu_n])
        # n_e = n_p is what charge neutrality will ask for.
        kF_e = hc * (3 * PI2 * n_p)**(1.0/3.0) if n_p > 0 else 0.0
        m_e = 0.511
        mu_e = np.sqrt(kF_e**2 + m_e**2) if n_p > 0 else m_e
        return np.array([mu_p, mu_n, mu_e])

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

    # The mean field, estimated from the symmetric part of the interaction and
    # split evenly between the two species.
    V_est = 4.0 * n_p * n_n * (
        par.a0 + par.b0 * (n_B/n0)**(par.gamma - 1)) / n0
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
def _mu_scale(mu_n):
    """The scale a potential equality is judged against: mu_B = mu_n."""
    return max(abs(mu_n), MU_SCALE_FLOOR)


def _finish(result, mu_p, mu_n, n_p, n_n, T, par, flags,
            e_thermo=None, nu_thermo=None):
    """Assemble the totals of a solved state: matter, leptons, photons.

    The photon gas is present exactly when `flags.photons` says so. Photons
    carry no conserved charge, so they reach eps, P and s and nothing else.
    """
    matter = thermo_from_mu_n(mu_p, mu_n, n_p, n_n, T, par)

    result.P = matter.P
    result.eps = matter.e
    result.s = matter.s
    for gas in (e_thermo, nu_thermo):
        if gas is not None:
            result.P += gas.P
            result.eps += gas.e
            result.s += gas.s
    if flags.photons:
        gamma = photon_thermo(T)
        result.P += gamma.P
        result.eps += gamma.e
        result.s += gamma.s

    result.mu_B = matter.mu_B
    result.mu_C = matter.mu_C
    return result


# =============================================================================
# SOLVER: BETA EQUILIBRIUM, NEUTRINOS FREE-STREAMING
# =============================================================================
def solve_beta_eq_neutrinoless(
    par: Parameters, n_B: float, flags: SpeciesFlags, T: float,
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
        par: model parameters; required (CLAUDE.md section 6)
        flags: the active degrees of freedom; `photons` is the only sector
            this model has, and it is honoured here rather than by a caller
        initial_guess: warm start in the layout above

    Returns:
        EoSPoint; test `.converged` before using any other field.
    """

    result = EoSPoint(n_B=n_B, T=T)
    cold = default_guess("beta_eq_neutrinoless", n_B, T, par)
    x0 = cold if initial_guess is None else initial_guess
    x0_fallback = None if initial_guess is None else cold

    def residual(x):
        mu_p, mu_n, mu_e, n_p, n_n = x
        state = effective_state(mu_p, mu_n, n_p, n_n, T, par)
        n_e = electron_thermo(mu_e, T, include_antiparticles=True).n
        return [state.n_p_calc - n_p,
                state.n_n_calc - n_n,
                state.n_B - n_B,
                mu_n - mu_p - mu_e,
                state.n_C - n_e]

    def scales_at(x):
        """Four densities against n_B, the beta condition against mu_B."""
        return [n_B, n_B, n_B, _mu_scale(x[1]), n_B]

    x, error, converged = solve_system(residual, x0, scales_at, x0_fallback)
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
    n_Le, _ = lepton_charges(n_e=result.n_e, n_nue=result.n_nu)
    result.Y_Le = n_Le / n_B

    return _finish(result, mu_p, mu_n, n_p, n_n, T, par, flags,
                   e_thermo=e_thermo)


# =============================================================================
# SOLVER: FIXED CHARGE FRACTION
# =============================================================================
def solve_fixed_yc(
    par: Parameters, n_B: float, Y_C: float, flags: SpeciesFlags, T: float,
    leptons: bool = False,
    initial_guess: Optional[np.ndarray] = None
) -> EoSPoint:
    """Fixed non-leptonic charge fraction, Y_C = n_p/n_B.

    The composition is known before the solve -- n_p = Y_C n_B,
    n_n = (1-Y_C) n_B -- so both densities leave the unknown vector and only
    the self-consistency rows remain:

        leptons=False   x = [mu_p, mu_n],        rows r1, r2
        leptons=True    x = [mu_p, mu_n, mu_e],  rows r1, r2 and
                        r5 = n_e(mu_e, T) - n_p (electric neutrality)

    With `leptons=False` the result is electrically CHARGED
    nucleonic matter, which is what a mixed-phase construction needs per pure
    phase before global neutrality is imposed. Y_C is the non-leptonic charge
    fraction in both cases; neutrality is a separate, additional condition.

    Args:
        n_B: baryon density (fm^-3)
        Y_C: non-leptonic charge fraction
        T: temperature (MeV)
        par: model parameters; required (CLAUDE.md section 6)
        flags: the active degrees of freedom; `photons` is the only sector
            this model has, and it is honoured here rather than by a caller
        leptons: add the neutralizing electron gas. NOT a species flag --
            CLAUDE.md section 5 makes it an orthogonal named argument
        initial_guess: warm start in the layout above

    Returns:
        EoSPoint; test `.converged` before using any other field.
    """

    result = EoSPoint(n_B=n_B, T=T, Y_C=Y_C)
    n_p = Y_C * n_B
    n_n = (1.0 - Y_C) * n_B

    cold = default_guess("fixed_YC", n_B, T, par, Y_C=Y_C,
                         leptons=leptons)
    x0 = cold if initial_guess is None else initial_guess
    x0_fallback = None if initial_guess is None else cold

    if leptons:
        def residual(x):
            mu_p, mu_n, mu_e = x
            state = effective_state(mu_p, mu_n, n_p, n_n, T, par)
            n_e = electron_thermo(mu_e, T, include_antiparticles=True).n
            return [state.n_p_calc - n_p,
                    state.n_n_calc - n_n,
                    n_e - state.n_C]
    else:
        def residual(x):
            mu_p, mu_n = x
            state = effective_state(mu_p, mu_n, n_p, n_n, T, par)
            return [state.n_p_calc - n_p,
                    state.n_n_calc - n_n]

    n_rows = 3 if leptons else 2

    def scales_at(x):
        """Every row of this mode balances a density."""
        return [n_B] * n_rows

    x, error, converged = solve_system(residual, x0, scales_at, x0_fallback)
    result.converged, result.error = converged, error

    if leptons:
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
    n_Le, _ = lepton_charges(n_e=result.n_e, n_nue=result.n_nu)
    result.Y_Le = n_Le / n_B

    return _finish(result, mu_p, mu_n, n_p, n_n, T, par, flags,
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
    par: Parameters, n_B: float, Y_Le: float, flags: SpeciesFlags, T: float,
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
        par: model parameters; required (CLAUDE.md section 6)
        flags: the active degrees of freedom; `photons` is the only sector
            this model has, and it is honoured here rather than by a caller
        initial_guess: warm start in the layout above

    Returns:
        EoSPoint; test `.converged` before using any other field.
    """

    result = EoSPoint(n_B=n_B, T=T)
    cold = default_guess("beta_eq_neutrino_trapped", n_B, T, par,
                         Y_Le=Y_Le)
    x0 = cold if initial_guess is None else initial_guess
    x0_fallback = None if initial_guess is None else cold

    def residual(x):
        mu_p, mu_n, mu_e, mu_nu, n_p, n_n = x
        state = effective_state(mu_p, mu_n, n_p, n_n, T, par)
        n_e = electron_thermo(mu_e, T, include_antiparticles=True).n
        n_nu = neutrino_thermo(mu_nu, T, include_antiparticles=True).n
        return [state.n_p_calc - n_p,
                state.n_n_calc - n_n,
                state.n_B - n_B,
                mu_n - mu_p - mu_e + mu_nu,
                state.n_C - n_e,
                (n_e + n_nu) / n_B - Y_Le]

    def scales_at(x):
        """Four densities against n_B, the beta condition against mu_B; the
        lepton-fraction row is already dimensionless."""
        return [n_B, n_B, n_B, _mu_scale(x[1]), n_B, 1.0]

    x, error, converged = solve_system(residual, x0, scales_at, x0_fallback)
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
    n_Le, _ = lepton_charges(n_e=result.n_e, n_nue=result.n_nu)
    result.Y_Le = n_Le / n_B

    return _finish(result, mu_p, mu_n, n_p, n_n, T, par, flags,
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
