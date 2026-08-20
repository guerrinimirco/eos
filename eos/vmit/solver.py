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
    from eos.vmit.solver import solve_vmit_beta_eq
    result = solve_vmit_beta_eq(n_B=0.32, T=50.0)
    print(result.converged, result.P_total)
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional

from eos.vmit.parameters import Parameters, get_vmit_default
from eos.vmit.thermodynamics import (
    compute_quark_densities_for_solver, compute_vmit_thermo_from_mu_n, G_QUARK,
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
class VMITEOSResult:
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
# INITIAL GUESS GENERATION
# =============================================================================
def get_default_guess_beta_eq(n_B: float, T: float, params: Parameters) -> np.ndarray:
    """Generate initial guess for beta equilibrium: [μ_u, μ_d, μ_s, μ_e, n_u, n_d, n_s]."""
    m_u, m_d, m_s = params.m_u, params.m_d, params.m_s
    
    # Estimate quark densities with strange quark suppression at low density/temperature
    # At high T or high nB, strange quarks are thermally populated
    # At low T and low nB, strange quarks are suppressed due to mass threshold
    strange_fraction = min(0.9, max(0.01, T / 100.0 + n_B / 0.5))
    
    # Initial guess: n_u ≈ n_d ≈ n_B, n_s slightly smaller at low density
    n_u_est = n_B * 1.0
    n_d_est = n_B * 1.0
    n_s_est = n_B * strange_fraction
    
    # Fermi momentum estimates
    kF_u = hc * (6.0 * PI2 * n_u_est / G_QUARK)**(1.0/3.0) if n_u_est > 0 else 0.0
    kF_d = hc * (6.0 * PI2 * n_d_est / G_QUARK)**(1.0/3.0) if n_d_est > 0 else 0.0
    kF_s = hc * (6.0 * PI2 * max(n_s_est, 1e-6) / G_QUARK)**(1.0/3.0)
    
    mu_u_est = np.sqrt(kF_u**2 + m_u**2)
    mu_d_est = np.sqrt(kF_d**2 + m_d**2)
    mu_s_est = np.sqrt(kF_s**2 + m_s**2)
    mu_e_est = max(0.0, mu_d_est - mu_u_est)  # Beta equilibrium estimate
    
    # Add vector field estimate
    V_est = params.a * hc * (n_u_est + n_d_est + n_s_est)
    
    return np.array([mu_u_est + V_est, mu_d_est + V_est, mu_s_est + V_est, 
                     mu_e_est, n_u_est, n_d_est, n_s_est])


def get_default_guess_fixed_yc(n_B: float, Y_C: float, T: float, 
                                params: Parameters,
                                include_electrons: bool = True) -> np.ndarray:
    """
    Generate initial guess for fixed Y_C.
    
    Returns [μ_u, μ_d, μ_s, n_u, n_d, n_s] if include_electrons=False,
    Returns [μ_u, μ_d, μ_s, n_u, n_d, n_s, μ_e] if include_electrons=True.
    """
    m_u, m_d, m_s = params.m_u, params.m_d, params.m_s
    
    # Estimate densities from constraints
    n_s = n_B * 0.3  # Initial strange fraction
    n_u = n_B + Y_C * n_B + n_s / 3.0
    n_d = n_B - Y_C * n_B / 2.0
    
    n_u = max(n_u, n_B * 0.3)
    n_d = max(n_d, n_B * 0.3)
    n_s = max(n_s, n_B * 0.1)
    
    kF_u = hc * (6.0 * PI2 * n_u / G_QUARK)**(1.0/3.0) if n_u > 0 else 0.0
    kF_d = hc * (6.0 * PI2 * n_d / G_QUARK)**(1.0/3.0) if n_d > 0 else 0.0
    kF_s = hc * (6.0 * PI2 * n_s / G_QUARK)**(1.0/3.0) if n_s > 0 else 0.0
    
    mu_u_est = np.sqrt(kF_u**2 + m_u**2)
    mu_d_est = np.sqrt(kF_d**2 + m_d**2)
    mu_s_est = np.sqrt(kF_s**2 + m_s**2)
    
    if include_electrons:
        # Estimate mu_e from n_e = n_Q = n_B * Y_C (charge neutrality)
        n_e = n_B * Y_C
        kF_e = hc * (3 * PI2 * n_e)**(1.0/3.0) if n_e > 0 else 0.0
        m_e = 0.511  # MeV
        mu_e_est = np.sqrt(kF_e**2 + m_e**2) if n_e > 0 else m_e
        return np.array([mu_u_est, mu_d_est, mu_s_est, n_u, n_d, n_s, mu_e_est])
    
    return np.array([mu_u_est, mu_d_est, mu_s_est, n_u, n_d, n_s])



def get_default_guess_fixed_yc_ys(n_B: float, Y_C: float, Y_S: float, T: float,
                                    params: Parameters,
                                    include_electrons: bool = True) -> np.ndarray:
    """
    Generate initial guess for fixed Y_C AND Y_S.
    
    Returns [μ_u, μ_d, μ_s, n_u, n_d, n_s] if include_electrons=False,
    Returns [μ_u, μ_d, μ_s, n_u, n_d, n_s, μ_e] if include_electrons=True.
    
    Analytic solution for densities (ignoring temperature effects on density):
      n_s = 3 * n_B * Y_S
      n_u = n_B * (1 + Y_C)
      n_d = 3 * n_B - n_s - n_u
          = n_B * (2 - Y_C) - n_s
    """
    m_u, m_d, m_s = params.m_u, params.m_d, params.m_s
    
    # Exact densities satisfying constraints
    n_s = n_B * Y_S
    n_u = n_B * (1.0 + Y_C)
    n_d = 3.0 * n_B - n_u - n_s
    
    # Ensure positivity for robustness (though physically should be positive)
    n_u = max(n_u, 1e-10)
    n_d = max(n_d, 1e-10)
    n_s = max(n_s, 1e-10)
    
    # Calculate Fermi momenta
    kF_u = hc * (6.0 * PI2 * n_u / G_QUARK)**(1.0/3.0)
    kF_d = hc * (6.0 * PI2 * n_d / G_QUARK)**(1.0/3.0)
    kF_s = hc * (6.0 * PI2 * n_s / G_QUARK)**(1.0/3.0)
    
    # Calculate chemical potentials
    mu_u_est = np.sqrt(kF_u**2 + m_u**2)
    mu_d_est = np.sqrt(kF_d**2 + m_d**2)
    mu_s_est = np.sqrt(kF_s**2 + m_s**2)
    
    if include_electrons:
        # Estimate mu_e from n_e = n_Q = n_B * Y_C (charge neutrality)
        n_e = n_B * Y_C
        kF_e = hc * (3 * PI2 * n_e)**(1.0/3.0) if n_e > 0 else 0.0
        m_e = 0.511  # MeV
        mu_e_est = np.sqrt(kF_e**2 + m_e**2) if n_e > 0 else m_e
        return np.array([mu_u_est, mu_d_est, mu_s_est, n_u, n_d, n_s, mu_e_est])
    
    return np.array([mu_u_est, mu_d_est, mu_s_est, n_u, n_d, n_s])


def get_default_guess_trapped_neutrinos(n_B: float, Y_L: float, T: float,
                                          params: Parameters) -> np.ndarray:
    """Generate initial guess for trapped neutrinos: [μ_u, μ_d, μ_s, μ_e, μ_ν, n_u, n_d, n_s]."""
    guess_beta = get_default_guess_beta_eq(n_B, T, params)
    mu_nu_est = 10.0
    return np.array([guess_beta[0], guess_beta[1], guess_beta[2], 
                     guess_beta[3], mu_nu_est, guess_beta[4], guess_beta[5], guess_beta[6]])


# =============================================================================
# SOLVER: BETA EQUILIBRIUM
# =============================================================================
def solve_vmit_beta_eq(
    n_B: float, T: float, params: Parameters = None,
    include_photons: bool = True,
    initial_guess: Optional[np.ndarray] = None
) -> VMITEOSResult:
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
        params: vMIT parameters
        include_photons: Include photon contributions
        initial_guess: Initial guess [μ_u, μ_d, μ_s, μ_e, n_u, n_d, n_s]
        
    Returns:
        VMITEOSResult with all thermodynamic quantities
    """
    if params is None:
        params = get_vmit_default()
    
    result = VMITEOSResult(n_B=n_B, T=T)
    
    m_u, m_d, m_s = params.m_u, params.m_d, params.m_s
    
    default_guess = get_default_guess_beta_eq(n_B, T, params)
    x0 = default_guess if initial_guess is None else initial_guess
    x0_fallback = None if initial_guess is None else default_guess

    def equations(x):
        mu_u, mu_d, mu_s, mu_e, n_u, n_d, n_s = x

        # Compute effective μ and densities
        qmd = compute_quark_densities_for_solver(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, params)
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
    q_thermo = compute_vmit_thermo_from_mu_n(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, params)
    
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
def solve_vmit_fixed_yc(
    n_B: float, Y_C: float, T: float, params: Parameters = None,
    include_photons: bool = True,
    include_electrons: bool = True,
    initial_guess: Optional[np.ndarray] = None
) -> VMITEOSResult:
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
    if params is None:
        params = get_vmit_default()
    
    result = VMITEOSResult(n_B=n_B, T=T, Y_C=Y_C)
    
    m_u, m_d, m_s = params.m_u, params.m_d, params.m_s
    
    default_guess = get_default_guess_fixed_yc(n_B, Y_C, T, params,
                                               include_electrons)
    x0 = default_guess if initial_guess is None else initial_guess
    x0_fallback = None if initial_guess is None else default_guess

    if include_electrons:
        # Solve 7 equations with electron charge neutrality
        def equations(x):
            mu_u, mu_d, mu_s, n_u, n_d, n_s, mu_e = x
            
            # Compute effective μ and densities
            qmd = compute_quark_densities_for_solver(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, params)
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
            qmd = compute_quark_densities_for_solver(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, params)

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
    q_thermo = compute_vmit_thermo_from_mu_n(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, params)
    
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
def solve_vmit_fixed_yc_ys(
    n_B: float, Y_C: float, Y_S: float, T: float, params: Parameters = None,
    include_photons: bool = True,
    include_electrons: bool = True,
    initial_guess: Optional[np.ndarray] = None
) -> VMITEOSResult:
    """
    Solve vMIT EOS with fixed charge fraction Y_C AND strangeness fraction Y_S.
    
    If include_electrons=False: 6 equations, 6 unknowns: [μ_u, μ_d, μ_s, n_u, n_d, n_s]
    If include_electrons=True:  7 equations, 7 unknowns: [μ_u, μ_d, μ_s, n_u, n_d, n_s, μ_e]
        with charge neutrality n_e(μ_e) = n_Q = n_B * Y_C
    """
    if params is None:
        params = get_vmit_default()
    
    result = VMITEOSResult(n_B=n_B, T=T, Y_C=Y_C, Y_S=Y_S)
    
    m_u, m_d, m_s = params.m_u, params.m_d, params.m_s
    
    default_guess = get_default_guess_fixed_yc_ys(n_B, Y_C, Y_S, T, params,
                                                  include_electrons)
    x0 = default_guess if initial_guess is None else initial_guess
    x0_fallback = None if initial_guess is None else default_guess
    
    if include_electrons:
        # Solve 7 equations with electron charge neutrality
        def equations(x):
            mu_u, mu_d, mu_s, n_u, n_d, n_s, mu_e = x
            
            # Compute effective μ and densities
            qmd = compute_quark_densities_for_solver(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, params)
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
            qmd = compute_quark_densities_for_solver(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, params)
            
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
    q_thermo = compute_vmit_thermo_from_mu_n(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, params)
    
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
def solve_vmit_trapped_neutrinos(
    n_B: float, Y_L: float, T: float, params: Parameters = None,
    include_photons: bool = True,
    initial_guess: Optional[np.ndarray] = None
) -> VMITEOSResult:
    """
    Solve vMIT EOS with trapped neutrinos (fixed lepton fraction Y_L).
    
    8 equations, 8 unknowns: [μ_u, μ_d, μ_s, μ_e, μ_ν, n_u, n_d, n_s]
    """
    if params is None:
        params = get_vmit_default()
    
    result = VMITEOSResult(n_B=n_B, T=T, Y_L=Y_L)
    
    m_u, m_d, m_s = params.m_u, params.m_d, params.m_s
    
    default_guess = get_default_guess_trapped_neutrinos(n_B, Y_L, T, params)
    x0 = default_guess if initial_guess is None else initial_guess
    x0_fallback = None if initial_guess is None else default_guess

    def equations(x):
        mu_u, mu_d, mu_s, mu_e, mu_nu, n_u, n_d, n_s = x
        
        # Compute effective μ and densities
        qmd = compute_quark_densities_for_solver(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, params)
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
    q_thermo = compute_vmit_thermo_from_mu_n(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, params)
    
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
# RESULT TO GUESS CONVERSION
# =============================================================================
def result_to_guess(result: VMITEOSResult, eq_type: str, include_electrons: bool = True) -> np.ndarray:
    """Convert result to initial guess array for next n_B point.
    
    Args:
        result: Previous result to extract guess from
        eq_type: Equilibrium type ('beta_eq', 'fixed_yc', 'fixed_yc_ys', 'trapped_neutrinos')
        include_electrons: For fixed_yc/fixed_yc_ys modes, whether electrons are included
    """
    if eq_type == 'beta_eq':
        return np.array([result.mu_u, result.mu_d, result.mu_s, result.mu_e,
                         result.n_u, result.n_d, result.n_s])
    elif eq_type in ('fixed_yc', 'fixed_yc_ys'):
        if include_electrons:
            return np.array([result.mu_u, result.mu_d, result.mu_s,
                             result.n_u, result.n_d, result.n_s, result.mu_e])
        else:
            return np.array([result.mu_u, result.mu_d, result.mu_s,
                             result.n_u, result.n_d, result.n_s])
    elif eq_type == 'trapped_neutrinos':
        return np.array([result.mu_u, result.mu_d, result.mu_s, result.mu_e, result.mu_nu,
                         result.n_u, result.n_d, result.n_s])
    else:
        return np.array([result.mu_u, result.mu_d, result.mu_s,
                         result.n_u, result.n_d, result.n_s])
