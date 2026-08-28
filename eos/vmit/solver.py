"""Single-point equilibrium solvers for vMIT quark matter.

One solver per equilibrium mode. Every mode enforces the vector-field
self-consistency n_q(mu_eff_q, T, m_q) = n_q for the three flavours and fixes
the baryon density; the mode supplies the rest:

    beta equilibrium (neutrinoless)   mu_C + mu_e = 0, mu_S = 0, n_C = n_e
    beta equilibrium (trapped)        ... with mu_nu kept and Y_L fixed
    fixed Y_C                         n_C = Y_C n_B, mu_S = 0
    fixed Y_C and Y_S                 n_C = Y_C n_B, n_S = Y_S n_B

The unknowns are the physical potentials together with the densities that
source the vector field, (mu_u, mu_d, mu_s, n_u, n_d, n_s), extended by mu_e
where a lepton condition is present and by mu_nu in the trapped mode. Keeping
the densities as unknowns rather than substituting V = a hbar c sum_q n_q is
what makes the residual polynomial in the mean field instead of nesting the
Fermi integrals inside it.

The thermodynamic kernels are in `thermodynamics.py`; the table driver
is in `table.py`; the spec API (eos_point / eos_table) is in `api.py`. See
`vmit.tex` for the physics.

Usage:
    from eos.vmit.solver import solve_beta_eq_neutrinoless
    from eos.vmit.species import SpeciesFlags
    result = solve_beta_eq_neutrinoless(par, 0.32, 50.0, SpeciesFlags())
    print(result.converged, result.P_total)
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional

from eos.vmit.parameters import Parameters
from eos.vmit.species import SpeciesFlags
from eos.vmit.thermodynamics import (
    effective_state, thermo_from_mu_n, G_QUARK,
)
from eos.general.thermodynamics_leptons import (
    electron_thermo, neutrino_thermo, photon_thermo,
)
from eos.general.physics_constants import hc, PI2
from eos.general.solve import (
    MU_SCALE_FLOOR, RESIDUAL_TOL, scaled_residual_max, solve_system,
)


def _mu_scale(mu_u, mu_d):
    """The scale a potential equality is judged against: mu_B = mu_u + 2 mu_d."""
    return max(abs(mu_u + 2.0 * mu_d), MU_SCALE_FLOOR)


# =============================================================================
# RESULT DATACLASS
# =============================================================================
@dataclass
class EoSPoint:
    """One solved vMIT state, with the status a caller must test first.

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
    n_B: float = 0.0       # Baryon density (fm⁻³)
    T: float = 0.0         # Temperature (MeV)
    Y_C: float = 0.0       # Charge fraction
    # Strangeness fraction n_S/n_B, MEASURED on the solved flavour densities
    # through eos.general.basis.quark_charges -- what the equilibrium
    # populated, not what a mode asked for. The two agree in fixed_YC_YS to
    # the solver's own residual and come apart everywhere else, which is the
    # whole reason it is reported.
    Y_S: float = 0.0
    Y_L: float = 0.0       # Lepton fraction
    
    # Chemical potentials (MeV)
    mu_u: float = 0.0
    mu_d: float = 0.0
    mu_s: float = 0.0
    mu_e: float = 0.0
    mu_nu: float = 0.0
    mu_B: float = 0.0      # Baryon chemical potential
    mu_C: float = 0.0      # Charge chemical potential
    mu_S: float = 0.0      # Strangeness chemical potential
    
    # Densities (fm⁻³)
    n_u: float = 0.0
    n_d: float = 0.0
    n_s: float = 0.0
    n_e: float = 0.0
    n_nu: float = 0.0
    
    # Thermodynamics (MeV/fm³ for P, e; fm⁻³ for s)
    P_total: float = 0.0
    e_total: float = 0.0
    s_total: float = 0.0
    
    # Fractions
    Y_u: float = 0.0
    Y_d: float = 0.0
    Y_s: float = 0.0
    Y_e: float = 0.0
    Y_nu: float = 0.0


# =============================================================================
# COLD GUESSES
# =============================================================================
def two_flavour_state(x, mode: str, leptons: bool = True):
    """(mu_u, mu_d, mu_s, mu_e, mu_nu, n_u, n_d, n_s) from a two-flavour vector.

    With the strange sector off the unknown vector is two entries shorter than
    the three-flavour one: neither mu_s nor n_s is solved for, because a
    flavour that is not in the matter has no density to close and no potential
    to close it at. Leaving them in as unknowns pinned by their own rows was
    tried and is measurably worse -- the near-empty Jacobian columns cost the
    beta-equilibrium solve three decades of residual, 6e-11 against 1e-12 --
    which is the same conditioning hazard, in a milder form, that made holding
    Y_S = 0 the wrong way to reach this state.

    The two are filled in afterwards: n_s = 0, and mu_s = mu_d because the
    weak relation s <-> d still holds, there is simply nothing populated at
    it. That is what makes mu_S = mu_s - mu_d vanish, so the reported
    strangeness potential agrees with the zero strangeness the state carries
    and E/A = mu_B + Y_S mu_S needs no special case.
    """
    if mode == "beta_eq_neutrinoless":
        mu_u, mu_d, mu_e, n_u, n_d = x
        mu_nu = 0.0
    elif mode == "beta_eq_neutrino_trapped":
        mu_u, mu_d, mu_e, mu_nu, n_u, n_d = x
    elif mode == "fixed_YC":
        mu_u, mu_d, n_u, n_d = x[:4]
        mu_e = x[4] if leptons else 0.0
        mu_nu = 0.0
    else:
        raise ValueError(f"mode {mode!r} has no two-flavour vector")
    return mu_u, mu_d, mu_d, mu_e, mu_nu, n_u, n_d, 0.0


def default_guess(mode: str, n_B: float, T: float, par: Parameters,
                  Y_C: float = None, Y_S: float = None, Y_Le: float = None,
                  leptons: bool = True,
                  two_flavour: bool = False) -> np.ndarray:
    """The cold start of one mode.

    Where the composition is unknown the quark densities are estimated first
    -- n_u = n_d = n_B with the strange flavour thermally suppressed at low T
    and low density -- and the potentials follow as sqrt(k_F^2 + m_q^2) at the
    Fermi momentum of those densities, shifted by the vector field
    V = a hbar c sum_q n_q they imply. Where the mode fixes the composition
    the densities are not estimated but inverted from the constraints
    themselves: at fixed Y_C and Y_S,

        n_s = Y_S n_B,   n_u = (1 + Y_C) n_B,   n_d = 3 n_B - n_u - n_s

    solves the charge, baryon and strangeness rows exactly, leaving only the
    vector self-consistency for the solver to close.

    The layouts are the unknown vectors of each mode's residual, so a guess is
    only valid within its own mode.
    """
    m_u, m_d, m_s = par.m_u, par.m_d, par.m_s

    def mu_of_n(n, m):
        """sqrt(k_F^2 + m^2) at the Fermi momentum of density n."""
        kF = hc * (6.0 * PI2 * max(n, 0.0) / G_QUARK)**(1.0 / 3.0)
        return np.sqrt(kF**2 + m**2)

    def mu_e_of_n(n_e):
        """The electron potential neutralizing a charge density n_e."""
        m_e = 0.511                                       # MeV
        if n_e <= 0.0:
            return m_e
        kF_e = hc * (3.0 * PI2 * n_e)**(1.0 / 3.0)
        return np.sqrt(kF_e**2 + m_e**2)

    if mode in ("beta_eq_neutrinoless", "beta_eq_neutrino_trapped"):
        # Strange quarks are suppressed by their mass threshold at low T and
        # low density, and thermally populated as either rises.
        strange_fraction = min(0.9, max(0.01, T / 100.0 + n_B / 0.5))
        n_u, n_d = n_B, n_B
        n_s = 0.0 if two_flavour else n_B * strange_fraction
        if two_flavour:
            # Two light flavours carry the whole baryon number between them,
            # and near-neutrality without a strange quark to help needs about
            # twice as many d as u: 2 n_u - n_d = 3 n_e with few electrons.
            n_u, n_d = n_B, 2.0 * n_B

        mu_u = mu_of_n(n_u, m_u)
        mu_d = mu_of_n(n_d, m_d)
        mu_s = mu_of_n(max(n_s, 1e-6), m_s)
        mu_e = max(0.0, mu_d - mu_u)          # beta equilibrium estimate
        V = par.a * hc * (n_u + n_d + n_s)

        if mode == "beta_eq_neutrinoless":
            if two_flavour:
                return np.array([mu_u + V, mu_d + V, mu_e, n_u, n_d])
            return np.array([mu_u + V, mu_d + V, mu_s + V, mu_e,
                             n_u, n_d, n_s])
        mu_nu = 10.0
        if two_flavour:
            return np.array([mu_u + V, mu_d + V, mu_e, mu_nu, n_u, n_d])
        return np.array([mu_u + V, mu_d + V, mu_s + V, mu_e, mu_nu,
                         n_u, n_d, n_s])

    if mode == "fixed_YC":
        n_s = 0.0 if two_flavour else n_B * 0.3
        n_u = max(n_B + Y_C * n_B + n_s / 3.0, n_B * 0.3)
        n_d = max(n_B - Y_C * n_B / 2.0, n_B * 0.3)
    elif mode == "fixed_YC_YS":
        # Exact: these densities satisfy the charge, baryon and strangeness
        # rows, so only the vector self-consistency is left to solve.
        n_s = max(n_B * Y_S, 1e-10)
        n_u = max(n_B * (1.0 + Y_C), 1e-10)
        n_d = max(3.0 * n_B - n_u - n_s, 1e-10)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    if two_flavour:
        x = [mu_of_n(n_u, m_u), mu_of_n(n_d, m_d), n_u, n_d]
    else:
        x = [mu_of_n(n_u, m_u), mu_of_n(n_d, m_d), mu_of_n(n_s, m_s),
             n_u, n_d, n_s]
    if leptons:
        x.append(mu_e_of_n(n_B * Y_C))    # n_e = n_C = Y_C n_B
    return np.array(x)


# =============================================================================
# SOLVER: BETA EQUILIBRIUM
# =============================================================================
def solve_beta_eq_neutrinoless(
    par: Parameters, n_B: float, T: float, flags: SpeciesFlags,
    initial_guess: Optional[np.ndarray] = None
) -> EoSPoint:
    """
    Solve vMIT EOS in beta equilibrium with charge neutrality.
    
    7 equations, 7 unknowns: [μ_u, μ_d, μ_s, μ_e, n_u, n_d, n_s]
    
    Constraints:
        - Weak equilibrium: μ_d = μ_u + μ_e, μ_s = μ_d
        - Charge neutrality: (2/3)n_u - (1/3)n_d - (1/3)n_s - n_e = 0
        - Baryon number: (n_u + n_d + n_s)/3 = n_B

    TWO-FLAVOUR MATTER IS THIS MODE WITH THE STRANGE SECTOR OFF
    (`two_flavour`, `eos.vmit.SpeciesFlags`), which is what two-flavour quark
    matter physically is. The layout does not change and the rows do not
    change: the s flavour stops being a degree of freedom of the matter, so
    `n_s_calc` is zero and row 3 pins n_s there, while row 7 keeps mu_s tied
    to mu_d. That is deliberate -- it leaves the strange slot DETERMINED
    rather than a null Jacobian column, which is the hazard that made holding
    Y_S = 0 the wrong way to reach this state -- and mu_S = mu_s - mu_d comes
    out zero as a consequence, matching the zero strangeness the state
    carries.
    
    Args:
        par: vMIT parameters
        n_B: Baryon density (fm⁻³)
        T: Temperature (MeV)
        flags: the active sectors; `photons` adds the thermal photon gas
            and `two_flavour` takes the s flavour out of the matter
        initial_guess: Initial guess [μ_u, μ_d, μ_s, μ_e, n_u, n_d, n_s]
        
    Returns:
        EoSPoint with all thermodynamic quantities
    """
    
    two_flavour = flags.two_flavour

    result = EoSPoint(n_B=n_B, T=T)
    
    m_u, m_d, m_s = par.m_u, par.m_d, par.m_s
    
    x0_default = default_guess("beta_eq_neutrinoless", n_B, T, par,
                               two_flavour=two_flavour)
    x0 = x0_default if initial_guess is None else initial_guess
    x0_fallback = None if initial_guess is None else x0_default

    def equations(x):
        if two_flavour:
            mu_u, mu_d, mu_s, mu_e, _, n_u, n_d, n_s = two_flavour_state(
                x, "beta_eq_neutrinoless")
        else:
            mu_u, mu_d, mu_s, mu_e, n_u, n_d, n_s = x

        # Compute effective μ and densities
        qmd = effective_state(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, par,
                                  two_flavour)
        n_e = electron_thermo(mu_e, T, include_antiparticles=True).n

        rows = [qmd.n_u_calc - n_u, qmd.n_d_calc - n_d]
        if not two_flavour:
            rows.append(qmd.n_s_calc - n_s)
        rows += [qmd.n_B - n_B, qmd.n_C - n_e, mu_u + mu_e - mu_d]
        if not two_flavour:
            rows.append(mu_d - mu_s)
        return rows

    def scales_at(x):
        """Every density row against n_B, every potential equality against
        mu_B; one of each goes when the strange flavour does."""
        mu_B = _mu_scale(x[0], x[1])
        if two_flavour:
            return [n_B, n_B, n_B, n_B, mu_B]
        return [n_B, n_B, n_B, n_B, n_B, mu_B, mu_B]

    x, error, converged = solve_system(equations, x0, scales_at, x0_fallback)
    if two_flavour:
        mu_u, mu_d, mu_s, mu_e, _, n_u, n_d, n_s = two_flavour_state(
            x, "beta_eq_neutrinoless")
    else:
        mu_u, mu_d, mu_s, mu_e, n_u, n_d, n_s = x
    result.converged = converged
    result.error = error

    # Store results
    result.mu_u, result.mu_d, result.mu_s, result.mu_e = mu_u, mu_d, mu_s, mu_e
    result.n_u, result.n_d, result.n_s = n_u, n_d, n_s
    
    # Compute quark thermodynamics using helper function
    q_thermo = thermo_from_mu_n(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, par,
                                two_flavour)
    
    # Add electron contribution
    e_thermo = electron_thermo(mu_e, T, include_antiparticles=True)
    result.n_e = e_thermo.n
    result.Y_C = q_thermo.Y_C
    result.Y_S = q_thermo.Y_S
    result.Y_u = n_u / n_B 
    result.Y_d = n_d / n_B 
    result.Y_s = n_s / n_B 
    result.Y_e = result.n_e / n_B 
    
    result.P_total = q_thermo.P + e_thermo.P
    result.e_total = q_thermo.e + e_thermo.e
    result.s_total = q_thermo.s + e_thermo.s
    
    if flags.photons:
        gamma = photon_thermo(T)
        result.P_total += gamma.P
        result.e_total += gamma.e
        result.s_total += gamma.s
    
    result.mu_B = q_thermo.mu_B
    result.mu_C = q_thermo.mu_C
    result.mu_S = q_thermo.mu_S
    
    return result


# =============================================================================
# SOLVER: FIXED Y_C
# =============================================================================
def solve_fixed_yc(
    par: Parameters, n_B: float, Y_C: float, T: float, flags: SpeciesFlags,
    leptons: bool = False,
    initial_guess: Optional[np.ndarray] = None
) -> EoSPoint:
    """
    Solve vMIT EOS with fixed charge fraction Y_C (strangeness equilibrium).
    
    If leptons=False: 6 equations, 6 unknowns: [μ_u, μ_d, μ_s, n_u, n_d, n_s]
    If leptons=True:  7 equations, 7 unknowns: [μ_u, μ_d, μ_s, n_u, n_d, n_s, μ_e]
        with charge neutrality n_e(μ_e) = n_Q = n_B * Y_C
    
    Constraints:
        - Charge: n_Q = (2/3)n_u - (1/3)n_d - (1/3)n_s = n_B * Y_C
        - Baryon: (n_u + n_d + n_s)/3 = n_B
        - Strangeness eq: μ_s = μ_d

    `two_flavour` removes the strange flavour from the matter, as in
    `solve_beta_eq_neutrinoless`. Y_C is still a free fraction with it on --
    u and d carry charge between them -- which is why this mode takes the flag
    and `solve_fixed_yc_ys` refuses it.
    """
    
    two_flavour = flags.two_flavour

    result = EoSPoint(n_B=n_B, T=T, Y_C=Y_C)
    
    m_u, m_d, m_s = par.m_u, par.m_d, par.m_s
    
    x0_default = default_guess("fixed_YC", n_B, T, par, Y_C=Y_C,
                               leptons=leptons,
                               two_flavour=two_flavour)
    x0 = x0_default if initial_guess is None else initial_guess
    x0_fallback = None if initial_guess is None else x0_default

    if leptons:
        # Solve 7 equations with electron charge neutrality
        def equations(x):
            if two_flavour:
                mu_u, mu_d, mu_s, mu_e, _, n_u, n_d, n_s = two_flavour_state(
                    x, "fixed_YC", leptons=True)
            else:
                mu_u, mu_d, mu_s, n_u, n_d, n_s, mu_e = x
            
            # Compute effective μ and densities
            qmd = effective_state(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, par,
                                  two_flavour)
            n_e = electron_thermo(mu_e, T, include_antiparticles=True).n
            
            rows = [qmd.n_u_calc - n_u, qmd.n_d_calc - n_d]
            if not two_flavour:
                rows.append(qmd.n_s_calc - n_s)
            rows += [qmd.n_B - n_B, qmd.n_C - n_B * Y_C]
            if not two_flavour:
                rows.append(mu_d - mu_s)
            rows.append(n_e - qmd.n_C)      # charge neutrality: n_e = n_C
            return rows

        def scales_at(x):
            """Every density row against n_B, the mu_s = mu_d equality against
            mu_B; that equality and one density row go with the flavour."""
            if two_flavour:
                return [n_B, n_B, n_B, n_B, n_B]
            return [n_B, n_B, n_B, n_B, n_B, _mu_scale(x[0], x[1]), n_B]

        x, error, converged = solve_system(equations, x0, scales_at, x0_fallback)
        if two_flavour:
            mu_u, mu_d, mu_s, mu_e, _, n_u, n_d, n_s = two_flavour_state(
                x, "fixed_YC", leptons=True)
        else:
            mu_u, mu_d, mu_s, n_u, n_d, n_s, mu_e = x
        result.converged = converged
        result.error = error

        result.mu_u, result.mu_d, result.mu_s, result.mu_e = mu_u, mu_d, mu_s, mu_e
        result.n_u, result.n_d, result.n_s = n_u, n_d, n_s

        # Compute electron quantities
        e_thermo = electron_thermo(mu_e, T, include_antiparticles=True)
        result.n_e = e_thermo.n
        result.Y_e = result.n_e / n_B

    else:
        # Solve 6 equations without electrons
        def equations(x):
            if two_flavour:
                mu_u, mu_d, mu_s, _, _, n_u, n_d, n_s = two_flavour_state(
                    x, "fixed_YC", leptons=False)
            else:
                mu_u, mu_d, mu_s, n_u, n_d, n_s = x

            # Compute effective μ and densities
            qmd = effective_state(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, par,
                                  two_flavour)

            rows = [qmd.n_u_calc - n_u, qmd.n_d_calc - n_d]
            if not two_flavour:
                rows.append(qmd.n_s_calc - n_s)
            rows += [qmd.n_B - n_B, qmd.n_C - n_B * Y_C]
            if not two_flavour:
                rows.append(mu_d - mu_s)
            return rows

        def scales_at(x):
            """Every density row against n_B, the mu_s = mu_d equality against
            mu_B; that equality and one density row go with the flavour."""
            if two_flavour:
                return [n_B, n_B, n_B, n_B]
            return [n_B, n_B, n_B, n_B, n_B, _mu_scale(x[0], x[1])]

        x, error, converged = solve_system(equations, x0, scales_at, x0_fallback)
        if two_flavour:
            mu_u, mu_d, mu_s, _, _, n_u, n_d, n_s = two_flavour_state(
                x, "fixed_YC", leptons=False)
        else:
            mu_u, mu_d, mu_s, n_u, n_d, n_s = x
        result.converged = converged
        result.error = error

        result.mu_u, result.mu_d, result.mu_s = mu_u, mu_d, mu_s
        result.n_u, result.n_d, result.n_s = n_u, n_d, n_s

    result.Y_u = n_u / n_B 
    result.Y_d = n_d / n_B 
    result.Y_s = n_s / n_B 
    
    # Compute quark thermodynamics using helper function
    q_thermo = thermo_from_mu_n(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, par,
                                two_flavour)

    result.Y_S = q_thermo.Y_S
    result.P_total = q_thermo.P
    result.e_total = q_thermo.e
    result.s_total = q_thermo.s
    
    # Add electron thermodynamics if included
    if leptons:
        e_thermo = electron_thermo(result.mu_e, T, include_antiparticles=True)
        result.P_total += e_thermo.P
        result.e_total += e_thermo.e
        result.s_total += e_thermo.s
    
    if flags.photons:
        gamma = photon_thermo(T)
        result.P_total += gamma.P
        result.e_total += gamma.e
        result.s_total += gamma.s
    
    result.mu_B = q_thermo.mu_B
    result.mu_C = q_thermo.mu_C
    result.mu_S = q_thermo.mu_S
    
    return result


# =============================================================================
# SOLVER: FIXED Y_C AND Y_S
# =============================================================================
def solve_fixed_yc_ys(
    par: Parameters, n_B: float, Y_C: float, Y_S: float, T: float,
    flags: SpeciesFlags,
    leptons: bool = False,
    initial_guess: Optional[np.ndarray] = None
) -> EoSPoint:
    """
    Solve vMIT EOS with fixed charge fraction Y_C AND strangeness fraction Y_S.
    
    If leptons=False: 6 equations, 6 unknowns: [μ_u, μ_d, μ_s, n_u, n_d, n_s]
    If leptons=True:  7 equations, 7 unknowns: [μ_u, μ_d, μ_s, n_u, n_d, n_s, μ_e]
        with charge neutrality n_e(μ_e) = n_Q = n_B * Y_C

    THIS IS THE ONE MODE THAT REFUSES `two_flavour`, and the refusal is the
    same statement the flag makes. This mode holds Y_S; with the strange
    sector off no species left in the state carries strangeness, so the row
    n_S = Y_S n_B reads 0 = Y_S n_B -- unsatisfiable for Y_S != 0, and for
    Y_S = 0 satisfied for every mu_S at once, which leaves mu_S undetermined
    and its Jacobian column null. Reaching two-flavour matter by asking for
    Y_S = 0 is exactly the route CLAUDE.md section 4 forbids ("no sector is
    disabled implicitly because its coupling happens to be zero"); it is
    reached by switching the sector off, in `beta_eq_neutrinoless`.
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
    
    result = EoSPoint(n_B=n_B, T=T, Y_C=Y_C, Y_S=Y_S)
    
    m_u, m_d, m_s = par.m_u, par.m_d, par.m_s
    
    x0_default = default_guess("fixed_YC_YS", n_B, T, par, Y_C=Y_C,
                               Y_S=Y_S, leptons=leptons)
    x0 = x0_default if initial_guess is None else initial_guess
    x0_fallback = None if initial_guess is None else x0_default
    
    if leptons:
        # Solve 7 equations with electron charge neutrality
        def equations(x):
            mu_u, mu_d, mu_s, n_u, n_d, n_s, mu_e = x
            
            # Compute effective μ and densities
            qmd = effective_state(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, par,
                                  two_flavour)
            n_e = electron_thermo(mu_e, T, include_antiparticles=True).n
            
            eq1 = qmd.n_u_calc - n_u
            eq2 = qmd.n_d_calc - n_d
            eq3 = qmd.n_s_calc - n_s
            eq4 = qmd.n_B - n_B
            eq5 = qmd.n_C - n_B * Y_C
            eq6 = qmd.n_S - n_B * Y_S
            eq7 = n_e - qmd.n_C  # Charge neutrality: n_e = n_C
            
            return [eq1, eq2, eq3, eq4, eq5, eq6, eq7]

        def scales_at(x):
            """Every equation of this mode is a density: all against n_B."""
            return [n_B] * 7

        x, error, converged = solve_system(equations, x0, scales_at, x0_fallback)
        mu_u, mu_d, mu_s, n_u, n_d, n_s, mu_e = x
        result.converged = converged
        result.error = error
        
        result.mu_u, result.mu_d, result.mu_s, result.mu_e = mu_u, mu_d, mu_s, mu_e
        result.n_u, result.n_d, result.n_s = n_u, n_d, n_s
        
        # Compute electron quantities
        e_thermo = electron_thermo(mu_e, T, include_antiparticles=True)
        result.n_e = e_thermo.n
        result.Y_e = result.n_e / n_B
        
    else:
        # Solve 6 equations without electrons
        def equations(x):
            mu_u, mu_d, mu_s, n_u, n_d, n_s = x
            
            # Compute effective μ and densities
            qmd = effective_state(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, par,
                                  two_flavour)
            
            eq1 = qmd.n_u_calc - n_u
            eq2 = qmd.n_d_calc - n_d
            eq3 = qmd.n_s_calc - n_s
            eq4 = qmd.n_B - n_B
            eq5 = qmd.n_C - n_B * Y_C
            eq6 = qmd.n_S - n_B * Y_S
            
            return [eq1, eq2, eq3, eq4, eq5, eq6]

        def scales_at(x):
            """Every equation of this mode is a density: all against n_B."""
            return [n_B] * 6

        x, error, converged = solve_system(equations, x0, scales_at, x0_fallback)
        mu_u, mu_d, mu_s, n_u, n_d, n_s = x
        result.converged = converged
        result.error = error
        
        result.mu_u, result.mu_d, result.mu_s = mu_u, mu_d, mu_s
        result.n_u, result.n_d, result.n_s = n_u, n_d, n_s
    
    result.Y_u = n_u / n_B 
    result.Y_d = n_d / n_B 
    result.Y_s = n_s / n_B 
    
    # Compute quark thermodynamics using helper function
    q_thermo = thermo_from_mu_n(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, par,
                                two_flavour)

    result.Y_S = q_thermo.Y_S
    result.P_total = q_thermo.P
    result.e_total = q_thermo.e
    result.s_total = q_thermo.s
    
    # Add electron thermodynamics if included
    if leptons:
        e_thermo = electron_thermo(result.mu_e, T, include_antiparticles=True)
        result.P_total += e_thermo.P
        result.e_total += e_thermo.e
        result.s_total += e_thermo.s
    
    if flags.photons:
        gamma = photon_thermo(T)
        result.P_total += gamma.P
        result.e_total += gamma.e
        result.s_total += gamma.s
    
    result.mu_B = q_thermo.mu_B
    result.mu_C = q_thermo.mu_C
    result.mu_S = q_thermo.mu_S

    return result


# =============================================================================
# SOLVER: TRAPPED NEUTRINOS
# =============================================================================
def solve_beta_eq_neutrino_trapped(
    par: Parameters, n_B: float, Y_L: float, T: float, flags: SpeciesFlags,
    initial_guess: Optional[np.ndarray] = None
) -> EoSPoint:
    """
    Solve vMIT EOS with trapped neutrinos (fixed lepton fraction Y_L).
    
    8 equations, 8 unknowns: [μ_u, μ_d, μ_s, μ_e, μ_ν, n_u, n_d, n_s]

    `two_flavour` removes the strange flavour from the matter exactly as in
    `solve_beta_eq_neutrinoless`; the lepton rows are untouched, the flag
    being orthogonal to the mode.
    """
    
    two_flavour = flags.two_flavour

    result = EoSPoint(n_B=n_B, T=T, Y_L=Y_L)
    
    m_u, m_d, m_s = par.m_u, par.m_d, par.m_s
    
    x0_default = default_guess("beta_eq_neutrino_trapped", n_B, T, par,
                               Y_Le=Y_L, two_flavour=two_flavour)
    x0 = x0_default if initial_guess is None else initial_guess
    x0_fallback = None if initial_guess is None else x0_default

    def equations(x):
        if two_flavour:
            mu_u, mu_d, mu_s, mu_e, mu_nu, n_u, n_d, n_s = two_flavour_state(
                x, "beta_eq_neutrino_trapped")
        else:
            mu_u, mu_d, mu_s, mu_e, mu_nu, n_u, n_d, n_s = x
        
        # Compute effective μ and densities
        qmd = effective_state(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, par,
                                  two_flavour)
        e_thermo = electron_thermo(mu_e, T, include_antiparticles=True)
        nu_thermo = neutrino_thermo(mu_nu, T, include_antiparticles=True)
        
        n_L = e_thermo.n + nu_thermo.n
        
        rows = [qmd.n_u_calc - n_u, qmd.n_d_calc - n_d]
        if not two_flavour:
            rows.append(qmd.n_s_calc - n_s)
        rows += [qmd.n_B - n_B, qmd.n_C - e_thermo.n]   # charge neutrality
        if not two_flavour:
            rows.append(mu_d - mu_s)                    # strangeness eq
        rows += [mu_u + mu_e - mu_d - mu_nu,            # beta eq, trapped
                 n_L / n_B - Y_L]                       # lepton fraction
        return rows

    def scales_at(x):
        """Every density row against n_B, every potential equality against
        mu_B; the lepton-fraction equation is already dimensionless, and one
        density row and one equality go when the strange flavour does."""
        mu_B = _mu_scale(x[0], x[1])
        if two_flavour:
            return [n_B, n_B, n_B, n_B, mu_B, 1.0]
        return [n_B, n_B, n_B, n_B, n_B, mu_B, mu_B, 1.0]

    x, error, converged = solve_system(equations, x0, scales_at, x0_fallback)
    if two_flavour:
        mu_u, mu_d, mu_s, mu_e, mu_nu, n_u, n_d, n_s = two_flavour_state(
            x, "beta_eq_neutrino_trapped")
    else:
        mu_u, mu_d, mu_s, mu_e, mu_nu, n_u, n_d, n_s = x
    result.converged = converged
    result.error = error

    result.mu_u, result.mu_d, result.mu_s, result.mu_e, result.mu_nu = mu_u, mu_d, mu_s, mu_e, mu_nu
    result.n_u, result.n_d, result.n_s = n_u, n_d, n_s
    
    # Compute quark thermodynamics using helper function
    q_thermo = thermo_from_mu_n(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, par,
                                two_flavour)
    
    # Add lepton contributions
    e_thermo = electron_thermo(mu_e, T, include_antiparticles=True)
    nu_thermo = neutrino_thermo(mu_nu, T, include_antiparticles=True)
    
    result.n_e = e_thermo.n
    result.n_nu = nu_thermo.n
    result.Y_C = q_thermo.Y_C
    result.Y_S = q_thermo.Y_S
    result.Y_u = n_u / n_B
    result.Y_d = n_d / n_B
    result.Y_s = n_s / n_B
    result.Y_e = result.n_e / n_B
    
    result.P_total = q_thermo.P + e_thermo.P + nu_thermo.P
    result.e_total = q_thermo.e + e_thermo.e + nu_thermo.e
    result.s_total = q_thermo.s + e_thermo.s + nu_thermo.s
    
    if flags.photons:
        gamma = photon_thermo(T)
        result.P_total += gamma.P
        result.e_total += gamma.e
        result.s_total += gamma.s

    result.mu_B = q_thermo.mu_B
    result.mu_C = q_thermo.mu_C
    result.mu_S = q_thermo.mu_S

    
    return result


# =============================================================================
# WARM START
# =============================================================================
def warm_start(point: EoSPoint, mode: str,
               leptons: bool = True,
               two_flavour: bool = False) -> np.ndarray:
    """The seed the next density takes from a solved point.

    The layouts are the unknown vectors of each mode's residual, so a warm
    start is only valid within its own mode -- and, since `two_flavour`
    shortens the vector by the two strange entries, within its own flavour
    content. Along a density sweep the potentials and the flavour densities
    vary smoothly, which is what carries the continuation through the strange
    quark's onset; with the strange sector off there is no onset to carry.
    """
    if mode == "beta_eq_neutrinoless":
        if two_flavour:
            return np.array([point.mu_u, point.mu_d, point.mu_e,
                             point.n_u, point.n_d])
        return np.array([point.mu_u, point.mu_d, point.mu_s, point.mu_e,
                         point.n_u, point.n_d, point.n_s])
    if mode == "beta_eq_neutrino_trapped":
        if two_flavour:
            return np.array([point.mu_u, point.mu_d, point.mu_e, point.mu_nu,
                             point.n_u, point.n_d])
        return np.array([point.mu_u, point.mu_d, point.mu_s, point.mu_e,
                         point.mu_nu, point.n_u, point.n_d, point.n_s])
    if mode in ("fixed_YC", "fixed_YC_YS"):
        if two_flavour:
            x = [point.mu_u, point.mu_d, point.n_u, point.n_d]
        else:
            x = [point.mu_u, point.mu_d, point.mu_s,
                 point.n_u, point.n_d, point.n_s]
        if leptons:
            x.append(point.mu_e)
        return np.array(x)
    raise ValueError(f"unknown mode {mode!r}")
