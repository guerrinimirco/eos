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
    result = solve_beta_eq_neutrinoless(n_B=0.32, T=50.0)
    print(result.converged, result.P_total)
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional

from eos.vmit.parameters import Parameters
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
    Y_S: float = 0.0       # Strangeness fraction
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
def default_guess(mode: str, n_B: float, T: float, par: Parameters,
                  Y_C: float = None, Y_S: float = None, Y_Le: float = None,
                  leptons: bool = True) -> np.ndarray:
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
        n_s = n_B * strange_fraction

        mu_u = mu_of_n(n_u, m_u)
        mu_d = mu_of_n(n_d, m_d)
        mu_s = mu_of_n(max(n_s, 1e-6), m_s)
        mu_e = max(0.0, mu_d - mu_u)          # beta equilibrium estimate
        V = par.a * hc * (n_u + n_d + n_s)

        if mode == "beta_eq_neutrinoless":
            return np.array([mu_u + V, mu_d + V, mu_s + V, mu_e,
                             n_u, n_d, n_s])
        mu_nu = 10.0
        return np.array([mu_u + V, mu_d + V, mu_s + V, mu_e, mu_nu,
                         n_u, n_d, n_s])

    if mode == "fixed_YC":
        n_s = n_B * 0.3
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

    x = [mu_of_n(n_u, m_u), mu_of_n(n_d, m_d), mu_of_n(n_s, m_s),
         n_u, n_d, n_s]
    if leptons:
        x.append(mu_e_of_n(n_B * Y_C))    # n_e = n_C = Y_C n_B
    return np.array(x)


# =============================================================================
# SOLVER: BETA EQUILIBRIUM
# =============================================================================
def solve_beta_eq_neutrinoless(
    par: Parameters, n_B: float, T: float,
    include_photons: bool = True,
    initial_guess: Optional[np.ndarray] = None
) -> EoSPoint:
    """
    Solve vMIT EOS in beta equilibrium with charge neutrality.
    
    7 equations, 7 unknowns: [μ_u, μ_d, μ_s, μ_e, n_u, n_d, n_s]
    
    Constraints:
        - Weak equilibrium: μ_d = μ_u + μ_e, μ_s = μ_d
        - Charge neutrality: (2/3)n_u - (1/3)n_d - (1/3)n_s - n_e = 0
        - Baryon number: (n_u + n_d + n_s)/3 = n_B
    
    Args:
        n_B: Baryon density (fm⁻³)
        T: Temperature (MeV)
        par: vMIT parameters
        include_photons: Include photon contributions
        initial_guess: Initial guess [μ_u, μ_d, μ_s, μ_e, n_u, n_d, n_s]
        
    Returns:
        EoSPoint with all thermodynamic quantities
    """
    
    result = EoSPoint(n_B=n_B, T=T)
    
    m_u, m_d, m_s = par.m_u, par.m_d, par.m_s
    
    x0_default = default_guess("beta_eq_neutrinoless", n_B, T, par)
    x0 = x0_default if initial_guess is None else initial_guess
    x0_fallback = None if initial_guess is None else x0_default

    def equations(x):
        mu_u, mu_d, mu_s, mu_e, n_u, n_d, n_s = x

        # Compute effective μ and densities
        qmd = effective_state(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, par)
        n_e = electron_thermo(mu_e, T, include_antiparticles=True).n

        eq1 = qmd.n_u_calc - n_u
        eq2 = qmd.n_d_calc - n_d
        eq3 = qmd.n_s_calc - n_s
        eq4 = qmd.n_B - n_B
        eq5 = qmd.n_C - n_e
        eq6 = mu_u + mu_e - mu_d
        eq7 = mu_d - mu_s

        return [eq1, eq2, eq3, eq4, eq5, eq6, eq7]

    def scales_at(x):
        """Five densities against n_B, two potential equalities against mu_B."""
        return [n_B, n_B, n_B, n_B, n_B, _mu_scale(x[0], x[1]),
                _mu_scale(x[0], x[1])]

    x, error, converged = solve_system(equations, x0, scales_at, x0_fallback)
    mu_u, mu_d, mu_s, mu_e, n_u, n_d, n_s = x
    result.converged = converged
    result.error = error

    # Store results
    result.mu_u, result.mu_d, result.mu_s, result.mu_e = mu_u, mu_d, mu_s, mu_e
    result.n_u, result.n_d, result.n_s = n_u, n_d, n_s
    
    # Compute quark thermodynamics using helper function
    q_thermo = thermo_from_mu_n(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, par)
    
    # Add electron contribution
    e_thermo = electron_thermo(mu_e, T, include_antiparticles=True)
    result.n_e = e_thermo.n
    result.Y_C = q_thermo.Y_C
    result.Y_u = n_u / n_B 
    result.Y_d = n_d / n_B 
    result.Y_s = n_s / n_B 
    result.Y_e = result.n_e / n_B 
    
    result.P_total = q_thermo.P + e_thermo.P
    result.e_total = q_thermo.e + e_thermo.e
    result.s_total = q_thermo.s + e_thermo.s
    
    if include_photons:
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
    par: Parameters, n_B: float, Y_C: float, T: float,
    include_photons: bool = True,
    include_electrons: bool = True,
    initial_guess: Optional[np.ndarray] = None
) -> EoSPoint:
    """
    Solve vMIT EOS with fixed charge fraction Y_C (strangeness equilibrium).
    
    If include_electrons=False: 6 equations, 6 unknowns: [μ_u, μ_d, μ_s, n_u, n_d, n_s]
    If include_electrons=True:  7 equations, 7 unknowns: [μ_u, μ_d, μ_s, n_u, n_d, n_s, μ_e]
        with charge neutrality n_e(μ_e) = n_Q = n_B * Y_C
    
    Constraints:
        - Charge: n_Q = (2/3)n_u - (1/3)n_d - (1/3)n_s = n_B * Y_C
        - Baryon: (n_u + n_d + n_s)/3 = n_B
        - Strangeness eq: μ_s = μ_d
    """
    
    result = EoSPoint(n_B=n_B, T=T, Y_C=Y_C)
    
    m_u, m_d, m_s = par.m_u, par.m_d, par.m_s
    
    x0_default = default_guess("fixed_YC", n_B, T, par, Y_C=Y_C,
                               leptons=include_electrons)
    x0 = x0_default if initial_guess is None else initial_guess
    x0_fallback = None if initial_guess is None else x0_default

    if include_electrons:
        # Solve 7 equations with electron charge neutrality
        def equations(x):
            mu_u, mu_d, mu_s, n_u, n_d, n_s, mu_e = x
            
            # Compute effective μ and densities
            qmd = effective_state(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, par)
            n_e = electron_thermo(mu_e, T, include_antiparticles=True).n
            
            eq1 = qmd.n_u_calc - n_u
            eq2 = qmd.n_d_calc - n_d
            eq3 = qmd.n_s_calc - n_s
            eq4 = qmd.n_B - n_B
            eq5 = qmd.n_C - n_B * Y_C
            eq6 = mu_d - mu_s
            eq7 = n_e - qmd.n_C  # Charge neutrality: n_e = n_C
            
            return [eq1, eq2, eq3, eq4, eq5, eq6, eq7]

        def scales_at(x):
            """Six densities against n_B, the mu_s = mu_d equality against mu_B."""
            return [n_B, n_B, n_B, n_B, n_B, _mu_scale(x[0], x[1]), n_B]

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
            qmd = effective_state(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, par)

            eq1 = qmd.n_u_calc - n_u
            eq2 = qmd.n_d_calc - n_d
            eq3 = qmd.n_s_calc - n_s
            eq4 = qmd.n_B - n_B
            eq5 = qmd.n_C - n_B * Y_C
            eq6 = mu_d - mu_s

            return [eq1, eq2, eq3, eq4, eq5, eq6]

        def scales_at(x):
            """Five densities against n_B, the mu_s = mu_d equality against mu_B."""
            return [n_B, n_B, n_B, n_B, n_B, _mu_scale(x[0], x[1])]

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
    q_thermo = thermo_from_mu_n(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, par)
    
    result.P_total = q_thermo.P
    result.e_total = q_thermo.e
    result.s_total = q_thermo.s
    
    # Add electron thermodynamics if included
    if include_electrons:
        e_thermo = electron_thermo(result.mu_e, T, include_antiparticles=True)
        result.P_total += e_thermo.P
        result.e_total += e_thermo.e
        result.s_total += e_thermo.s
    
    if include_photons:
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
    include_photons: bool = True,
    include_electrons: bool = True,
    initial_guess: Optional[np.ndarray] = None
) -> EoSPoint:
    """
    Solve vMIT EOS with fixed charge fraction Y_C AND strangeness fraction Y_S.
    
    If include_electrons=False: 6 equations, 6 unknowns: [μ_u, μ_d, μ_s, n_u, n_d, n_s]
    If include_electrons=True:  7 equations, 7 unknowns: [μ_u, μ_d, μ_s, n_u, n_d, n_s, μ_e]
        with charge neutrality n_e(μ_e) = n_Q = n_B * Y_C
    """
    
    result = EoSPoint(n_B=n_B, T=T, Y_C=Y_C, Y_S=Y_S)
    
    m_u, m_d, m_s = par.m_u, par.m_d, par.m_s
    
    x0_default = default_guess("fixed_YC_YS", n_B, T, par, Y_C=Y_C,
                               Y_S=Y_S, leptons=include_electrons)
    x0 = x0_default if initial_guess is None else initial_guess
    x0_fallback = None if initial_guess is None else x0_default
    
    if include_electrons:
        # Solve 7 equations with electron charge neutrality
        def equations(x):
            mu_u, mu_d, mu_s, n_u, n_d, n_s, mu_e = x
            
            # Compute effective μ and densities
            qmd = effective_state(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, par)
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
            qmd = effective_state(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, par)
            
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
    q_thermo = thermo_from_mu_n(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, par)
    
    result.P_total = q_thermo.P
    result.e_total = q_thermo.e
    result.s_total = q_thermo.s
    
    # Add electron thermodynamics if included
    if include_electrons:
        e_thermo = electron_thermo(result.mu_e, T, include_antiparticles=True)
        result.P_total += e_thermo.P
        result.e_total += e_thermo.e
        result.s_total += e_thermo.s
    
    if include_photons:
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
    par: Parameters, n_B: float, Y_L: float, T: float,
    include_photons: bool = True,
    initial_guess: Optional[np.ndarray] = None
) -> EoSPoint:
    """
    Solve vMIT EOS with trapped neutrinos (fixed lepton fraction Y_L).
    
    8 equations, 8 unknowns: [μ_u, μ_d, μ_s, μ_e, μ_ν, n_u, n_d, n_s]
    """
    
    result = EoSPoint(n_B=n_B, T=T, Y_L=Y_L)
    
    m_u, m_d, m_s = par.m_u, par.m_d, par.m_s
    
    x0_default = default_guess("beta_eq_neutrino_trapped", n_B, T, par,
                               Y_Le=Y_L)
    x0 = x0_default if initial_guess is None else initial_guess
    x0_fallback = None if initial_guess is None else x0_default

    def equations(x):
        mu_u, mu_d, mu_s, mu_e, mu_nu, n_u, n_d, n_s = x
        
        # Compute effective μ and densities
        qmd = effective_state(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, par)
        e_thermo = electron_thermo(mu_e, T, include_antiparticles=True)
        nu_thermo = neutrino_thermo(mu_nu, T, include_antiparticles=True)
        
        n_L = e_thermo.n + nu_thermo.n
        
        eq1 = qmd.n_u_calc - n_u
        eq2 = qmd.n_d_calc - n_d
        eq3 = qmd.n_s_calc - n_s
        eq4 = qmd.n_B - n_B
        eq5 = qmd.n_C - e_thermo.n  # Charge neutrality
        eq6 = mu_d - mu_s  # Strangeness eq
        eq7 = mu_u + mu_e - mu_d - mu_nu  # Beta eq with neutrinos
        eq8 = n_L / n_B - Y_L  # Lepton fraction
        
        return [eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8]

    def scales_at(x):
        """Five densities against n_B, two potential equalities against mu_B;
        the lepton-fraction equation is already dimensionless."""
        mu_B = _mu_scale(x[0], x[1])
        return [n_B, n_B, n_B, n_B, n_B, mu_B, mu_B, 1.0]

    x, error, converged = solve_system(equations, x0, scales_at, x0_fallback)
    mu_u, mu_d, mu_s, mu_e, mu_nu, n_u, n_d, n_s = x
    result.converged = converged
    result.error = error

    result.mu_u, result.mu_d, result.mu_s, result.mu_e, result.mu_nu = mu_u, mu_d, mu_s, mu_e, mu_nu
    result.n_u, result.n_d, result.n_s = n_u, n_d, n_s
    
    # Compute quark thermodynamics using helper function
    q_thermo = thermo_from_mu_n(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, par)
    
    # Add lepton contributions
    e_thermo = electron_thermo(mu_e, T, include_antiparticles=True)
    nu_thermo = neutrino_thermo(mu_nu, T, include_antiparticles=True)
    
    result.n_e = e_thermo.n
    result.n_nu = nu_thermo.n
    result.Y_C = q_thermo.Y_C
    result.Y_u = n_u / n_B
    result.Y_d = n_d / n_B
    result.Y_s = n_s / n_B
    result.Y_e = result.n_e / n_B
    
    result.P_total = q_thermo.P + e_thermo.P + nu_thermo.P
    result.e_total = q_thermo.e + e_thermo.e + nu_thermo.e
    result.s_total = q_thermo.s + e_thermo.s + nu_thermo.s
    
    if include_photons:
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
               leptons: bool = True) -> np.ndarray:
    """The seed the next density takes from a solved point.

    The layouts are the unknown vectors of each mode's residual, so a warm
    start is only valid within its own mode. Along a density sweep the
    potentials and the flavour densities vary smoothly, which is what carries
    the continuation through the strange quark's onset.
    """
    if mode == "beta_eq_neutrinoless":
        return np.array([point.mu_u, point.mu_d, point.mu_s, point.mu_e,
                         point.n_u, point.n_d, point.n_s])
    if mode == "beta_eq_neutrino_trapped":
        return np.array([point.mu_u, point.mu_d, point.mu_s, point.mu_e,
                         point.mu_nu, point.n_u, point.n_d, point.n_s])
    if mode in ("fixed_YC", "fixed_YC_YS"):
        x = [point.mu_u, point.mu_d, point.mu_s,
             point.n_u, point.n_d, point.n_s]
        if leptons:
            x.append(point.mu_e)
        return np.array(x)
    raise ValueError(f"unknown mode {mode!r}")
