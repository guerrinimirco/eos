"""
nmp.py
======
The maps between SFHo's couplings and the physical quantities they are
fitted to, in both directions.

    compute_nmp(par)                couplings -> nuclear-matter parameters
    compute_hyperon_potentials(par)  couplings -> U_Lambda, U_Sigma, U_Xi
    create_custom_parametrization(U_Lambda_N, ...)   the inverse of that

This module sits ABOVE `solver.py` in the import order (CLAUDE.md section
5), and it has to: every quantity here is defined by a property of the
SOLVED state -- the saturation density is where the pressure vanishes, the
effective mass is read off the converged fields -- so computing any of them
means solving symmetric matter. That is also why a constructor that inverts
one of these maps is a free function here rather than a classmethod on the
parameter dataclass, which is at the bottom of the same order.

Definitions follow the CompOSE manual (Typel et al., arXiv:2203.03209 sec.
6) and Steiner, Prakash, Lattimer & Ellis, Phys. Rept. 411 (2005).

Units:
- Energies/masses/potentials: MeV
- Densities: fm^-3
"""
import numpy as np
from typing import Optional, Tuple, Dict
from scipy.optimize import brentq

from eos.general.physics_constants import hc, hc3
from eos.sfho.species import SpeciesFlags
from eos.sfho.parameters import (
    SFHoParams, get_sfho_nucleonic, get_sfhoy_fortin, 
    get_sfhoy_star_fortin, get_sfho_2fam_phi, get_sfho_2fam,
    SU6_RATIOS, SQRT2, _get_base_sfho
)


# =============================================================================
# CONSTANTS
# =============================================================================
N_SAT = 0.158  # fm^-3, saturation density


# =============================================================================
# COMPUTE SATURATION FIELDS
# =============================================================================
def compute_saturation_fields(params: Optional[SFHoParams] = None, 
                               n_B: float = N_SAT, 
                               Y_C: float = 0.5,
                               T: float = 0.01) -> Tuple[float, float, float, float]:
    """
    Compute meson fields (σ, ω, ρ, φ) at given density in nuclear matter.
    
    Args:
        params: SFHo parameters (defaults to nucleonic SFHo)
        n_B: Baryon density (fm⁻³)
        Y_C: Charge fraction (0.5 = symmetric nuclear matter)
        T: Temperature (MeV), use small T for T→0 limit
        
    Returns:
        (sigma, omega, rho, phi) fields in MeV
    """
    from eos.sfho.solver import solve_fixed_yc
    from eos.sfho.species import SpeciesFlags
    
    if params is None:
        params = get_sfho_nucleonic()
    
    result = solve_fixed_yc(params, n_B, Y_C, SpeciesFlags(photons=False),
                           T=T)
    
    if not result.converged:
        raise RuntimeError(f"Failed to converge at n_B={n_B}, Y_C={Y_C}, T={T}")
    
    return result.sigma, result.omega, result.rho, result.phi


# =============================================================================
# COMPUTE HYPERON POTENTIAL DEPTHS
# =============================================================================
def compute_hyperon_potentials(params: SFHoParams, 
                                sigma: float = None, 
                                omega: float = None) -> Dict[str, float]:
    """
    Compute hyperon potential depths U_H^(N) at saturation in SNM.
    
    U_H = -g_σH × σ + g_ωH × ω
    
    Args:
        params: SFHo parameters with hyperon couplings
        sigma: σ field in MeV (if None, computed at n_sat)
        omega: ω field in MeV (if None, computed at n_sat)
        
    Returns:
        Dictionary with U_Λ, U_Σ, U_Ξ in MeV
    """
    if sigma is None or omega is None:
        sigma, omega, _, _ = compute_saturation_fields()
    
    potentials = {}
    
    for hyperon, label in [('lambda', 'U_Lambda'), 
                           ('sigma+', 'U_Sigma'), 
                           ('xi0', 'U_Xi')]:
        if hyperon in params.couplings_map:
            g_sigma_H = params.couplings_map[hyperon]['sigma']
            g_omega_H = params.couplings_map[hyperon]['omega']
            U_H = -g_sigma_H * sigma + g_omega_H * omega
            potentials[label] = U_H
        else:
            potentials[label] = None
            
    return potentials


# =============================================================================
# FORWARD:  couplings -> nuclear-matter parameters
# =============================================================================
#: Symmetric nuclear matter, hadrons only: no electrons, no photons. The
#: nuclear-matter parameters are properties of the strongly-interacting
#: sector, so a lepton or radiation term in eps would corrupt every one of
#: them.
SNM_FLAGS = SpeciesFlags(photons=False)

#: The temperature the T -> 0 limit is taken at. SFHo's Fermi integrals accept
#: T = 0, but the whole NMP path is finite differences of eps, and a strictly
#: cold solve puts a threshold kink exactly where the differences straddle.
#: 0.01 MeV is far below any nuclear scale and keeps the sweep smooth.
T_COLD = 0.01


def _snm(par, n_B, Y_C=0.5):
    """The solved symmetric-matter point at n_B, or a raised error."""
    from eos.sfho.solver import solve_fixed_yc

    point = solve_fixed_yc(par, n_B, Y_C, SNM_FLAGS, T=T_COLD)
    if not point.converged:
        raise RuntimeError(
            f"symmetric matter did not converge at n_B={n_B:g}, Y_C={Y_C:g} "
            f"(residual {point.error:.3e})")
    return point


def energy_per_baryon(par, n_B, Y_C=0.5):
    """E/A [MeV] of nuclear matter at n_B [fm^-3], rest mass subtracted."""
    return _snm(par, n_B, Y_C).eps / n_B - 0.5 * (par.m_n + par.m_p)


def pressure(par, n_B, Y_C=0.5):
    """P [MeV/fm^3] of nuclear matter at n_B [fm^-3]. Vanishes at n_sat."""
    return _snm(par, n_B, Y_C).P


def esym(par, n_B):
    """
    Symmetry energy E_sym(n_B) [MeV], mean-field closed form.

    Steiner, Prakash, Lattimer & Ellis, Phys. Rept. 411 (2005), Eq. (20):

        E_sym = k_F^2 / (6 E_F*) + n_B / [ 8 ( m_rho^2/g_rho^2 + 2 f ) ]

    with A = g_rho^2 f, so the second term is n_B g_rho^2 / [8 (m_rho^2 + 2A)].
    A = A(sigma, omega) is SFHo's isoscalar-isovector cross coupling, which is
    what makes L_sym adjustable at fixed E_sym in this family of models.

    This is the rho-field response, not a rearrangement of eps, so comparing
    it with the delta^2 curvature of E/A is a genuine second opinion on the
    isovector sector rather than the same computation written twice --
    `verify/run_full_check.py` runs exactly that comparison.
    """
    point = _snm(par, n_B)
    k_F = hc * (3.0 * np.pi**2 * n_B / 2.0) ** (1.0 / 3.0)
    E_F = np.sqrt(k_F**2 + point.m_eff("n")**2)
    A = par.compute_A(point.sigma, point.omega)
    kinetic = k_F**2 / (6.0 * E_F)
    potential = n_B * hc3 * par.g_rho_N**2 / (8.0 * (par.m_rho**2 + 2.0 * A))
    return kinetic + potential


def compute_nmp(par, h=1e-3, n_lo=0.12, n_hi=0.20):
    """
    Nuclear-matter parameters at saturation.

    Returns dict with n_sat [fm^-3], E_sat, K_sat, Q_sat, E_sym, L_sym,
    K_sym [MeV], m_eff_ratio, and P_sat [MeV/fm^3] (diagnostic, ~0 by
    construction). The same keys `eos.dd2.compute_nmp` returns, so one caller
    reads either model.

    The derivatives are central differences AT saturation, not derivatives of
    a spline fitted over a density range: a fit spreads the third derivative
    over the whole range it was fitted on, and `bc_type='natural'` additionally
    pins the curvature to zero at the endpoints.

    On `h`. The step has to sit above the solver's own noise and below where
    truncation bites. Measured for SFHo_Nucleonic, K_sat and Q_sat are flat
    from h = 1e-4 to 2e-3 (245.221 +- 0.001 and -467.4 +- 0.1) and start
    drifting at 4e-3, reaching 245.43 / -470.4 by 1.6e-2. The default sits in
    the middle of that plateau.

    Q_sat and K_sym are PREDICTIONS of the parametrization, not quantities any
    fit imposes; they are reported for exactly that reason.
    """
    n_sat = brentq(lambda n: pressure(par, n), n_lo, n_hi, xtol=1e-13)
    at_sat = _snm(par, n_sat)

    EA = lambda n: energy_per_baryon(par, n)
    d2 = (EA(n_sat + h) - 2.0 * EA(n_sat) + EA(n_sat - h)) / h**2
    d3 = (EA(n_sat + 2 * h) - 2.0 * EA(n_sat + h)
          + 2.0 * EA(n_sat - h) - EA(n_sat - 2 * h)) / (2.0 * h**3)
    dEs = (esym(par, n_sat + h) - esym(par, n_sat - h)) / (2.0 * h)
    d2Es = (esym(par, n_sat + h) - 2.0 * esym(par, n_sat)
            + esym(par, n_sat - h)) / h**2

    m_N = 0.5 * (par.m_n + par.m_p)
    return {
        "n_sat": n_sat,
        "E_sat": EA(n_sat),
        "m_eff_ratio": at_sat.m_eff("n") / m_N,
        "K_sat": 9.0 * n_sat**2 * d2,
        "Q_sat": 27.0 * n_sat**3 * d3,
        "E_sym": esym(par, n_sat),
        "L_sym": 3.0 * n_sat * dEs,
        "K_sym": 9.0 * n_sat**2 * d2Es,
        "P_sat": at_sat.P,
    }


# =============================================================================
# CREATE CUSTOM PARAMETRIZATION FROM POTENTIAL DEPTHS
# =============================================================================
def create_custom_parametrization(
    # Hyperon potential depths (MeV)
    U_Lambda_N: float = -30.0,
    U_Sigma_N: float = +30.0,
    U_Xi_N: float = -14.0,
    # Vector coupling enhancement factors (per hyperon family)
    # g_ωH = g_ωN × SU(6)_ratio × y_H, g_φH = g_ωN × SU(6)_ratio × y_H
    y_Lambda: float = 1.0,
    y_Sigma: float = 1.0,
    y_Xi: float = 1.0,
    # Delta couplings
    x_sigma_delta: float = 1.15,
    x_omega_delta: float = 1.0,
    x_rho_delta: float = 1.0,
    # Name
    name: str = "Custom"
) -> SFHoParams:
    """
    Create custom parametrization from target hyperon potential depths.

    The scalar coupling R_σH is determined from the target potential depth:
        U_H = -g_σH × σ + g_ωH × ω
        R_σH = (R_ωH × y_H × g_ωN × ω - U_H) / (g_σN × σ)

    σ and ω are SOLVED for at saturation, by `compute_saturation_fields`, and
    that is why this function lives in `nmp.py` rather than in `parameters.py`:
    it is an inverse map from a physical observable to a coupling, so it needs
    the solver, and `parameters.py` is the bottom of the import layer and
    cannot reach it (CLAUDE.md §5). A second copy did live there, with the two
    fields written in as constants; the constants were mutually inconsistent —
    no single density reproduces both — and the couplings they produced missed
    the requested depths by about 3 MeV, so asking for U_Λ = -30 delivered
    -33.07. Hardcoding them is also wrong in principle for an inference run,
    where the base couplings vary and the saturation fields move with them.

    Vector couplings follow SU(6) symmetry × y_H enhancement factor per family:
        g_ωΛ = g_ωN × (2/3) × y_Lambda
        g_ωΣ = g_ωN × (2/3) × y_Sigma  
        g_ωΞ = g_ωN × (1/3) × y_Xi
    
    Example: y_Lambda=1.5, y_Sigma=1.5, y_Xi=1.875 gives:
        g_ωΛ = 1.0 × g_ωN, g_ωΣ = 1.0 × g_ωN, g_ωΞ = 0.625 × g_ωN
    
    Args:
        U_Lambda_N: Λ potential depth at n_sat in SNM (MeV), ~ -30 MeV
        U_Sigma_N: Σ potential depth at n_sat in SNM (MeV), ~ +30 MeV  
        U_Xi_N: Ξ potential depth at n_sat in SNM (MeV), ~ +10 to -20 MeV
        y_Lambda: Enhancement factor for Λ (1.0 = SU(6))
        y_Sigma: Enhancement factor for Σ (1.0 = SU(6))
        y_Xi: Enhancement factor for Ξ (1.0 = SU(6))
        x_sigma_delta: R_σΔ = g_σΔ/g_σN
        x_omega_delta: R_ωΔ = g_ωΔ/g_ωN
        x_rho_delta: R_ρΔ = g_ρΔ/g_ρN
        name: Name for the parametrization
        
    Returns:
        SFHoParams with computed couplings
    """
    # Get base SFHo parameters
    p = _get_base_sfho()
    p.name = name
    
    # Compute saturation fields
    sigma, omega, _, _ = compute_saturation_fields()
    
    # SU(6) vector ratios (before enhancement)
    R_omega_Lambda_SU6 = 2.0/3.0
    R_omega_Sigma_SU6 = 2.0/3.0
    R_omega_Xi_SU6 = 1.0/3.0
    
    R_phi_Lambda_SU6 = -SQRT2/3.0
    R_phi_Sigma_SU6 = -SQRT2/3.0
    R_phi_Xi_SU6 = -2.0*SQRT2/3.0
    
    # Apply enhancement factors
    R_omega_Lambda = R_omega_Lambda_SU6 * y_Lambda
    R_omega_Sigma = R_omega_Sigma_SU6 * y_Sigma
    R_omega_Xi = R_omega_Xi_SU6 * y_Xi
    
    R_phi_Lambda = R_phi_Lambda_SU6 * y_Lambda
    R_phi_Sigma = R_phi_Sigma_SU6 * y_Sigma
    R_phi_Xi = R_phi_Xi_SU6 * y_Xi
    
    # Compute scalar coupling ratios from potential depths
    # U_H = -R_σH × g_σN × σ + R_ωH × g_ωN × ω
    # R_σH = (R_ωH × g_ωN × ω - U_H) / (g_σN × σ)
    
    def compute_R_sigma(U_H: float, R_omega: float) -> float:
        return (R_omega * p.g_omega_N * omega - U_H) / (p.g_sigma_N * sigma)
    
    R_sigma_Lambda = compute_R_sigma(U_Lambda_N, R_omega_Lambda)
    R_sigma_Sigma = compute_R_sigma(U_Sigma_N, R_omega_Sigma)
    R_sigma_Xi = compute_R_sigma(U_Xi_N, R_omega_Xi)
    
    # Lambda couplings
    p.couplings_map['lambda'] = {
        'sigma': R_sigma_Lambda * p.g_sigma_N,
        'omega': R_omega_Lambda * p.g_omega_N,
        'phi': R_phi_Lambda * p.g_omega_N,
        'rho': 0.0,
    }
    
    # Sigma couplings (all Σ+, Σ0, Σ-)
    sigma_couplings = {
        'sigma': R_sigma_Sigma * p.g_sigma_N,
        'omega': R_omega_Sigma * p.g_omega_N,
        'phi': R_phi_Sigma * p.g_omega_N,
        'rho': 2.0 * p.g_rho_N,
    }
    for s_name in ['sigma+', 'sigma0', 'sigma-']:
        p.couplings_map[s_name] = sigma_couplings.copy()
    
    # Xi couplings
    xi_couplings = {
        'sigma': R_sigma_Xi * p.g_sigma_N,
        'omega': R_omega_Xi * p.g_omega_N,
        'phi': R_phi_Xi * p.g_omega_N,
        'rho': 1.0 * p.g_rho_N,
    }
    for x_name in ['xi0', 'xi-']:
        p.couplings_map[x_name] = xi_couplings.copy()
    
    # Delta couplings
    delta_couplings = {
        'sigma': x_sigma_delta * p.g_sigma_N,
        'omega': x_omega_delta * p.g_omega_N,
        'phi': 0.0,  # Deltas don't couple to φ
        'rho': x_rho_delta * p.g_rho_N,
    }
    for d_name in ['delta++', 'delta+', 'delta0', 'delta-']:
        p.couplings_map[d_name] = delta_couplings.copy()
    
    return p


# =============================================================================
# A SUMMARY, FOR READING RATHER THAN FOR SOLVING
# =============================================================================
#: Steiner, Hempel & Fischer, ApJ 774 (2013) 17, as tabulated by Fortin,
#: Oertel & Providencia, PASA 35 (2018) e044 Table 2 and by the CompOSE
#: HS(SFHo) entry. Q_sat and K_sym are not fitted quantities and carry no
#: published value here; `compute_nmp` reports them as predictions.
PUBLISHED_NMP = {
    "n_sat": 0.1583, "E_sat": -16.19, "m_eff_ratio": 0.76,
    "K_sat": 245.4, "E_sym": 31.57, "L_sym": 47.10,
}


def print_nmp_summary(par=None):
    """Print the nuclear-matter parameters beside their published values.

    A reading aid, not part of any solve: nothing in `eos` calls it, and
    `verify/run_full_check.py` is what asserts the agreement.
    """
    par = par or get_sfho_nucleonic()
    nmp = compute_nmp(par)
    print(f"nuclear-matter parameters, {getattr(par, 'name', '?')}")
    print(f"{'':14s} {'this model':>12s} {'published':>12s} {'difference':>12s}")
    for key in ("n_sat", "E_sat", "m_eff_ratio", "K_sat", "E_sym", "L_sym"):
        got, want = nmp[key], PUBLISHED_NMP[key]
        print(f"  {key:12s} {got:12.4f} {want:12.4f} {got - want:+12.4f}")
    print("  predictions, imposed by no fit:")
    for key in ("Q_sat", "K_sym"):
        print(f"  {key:12s} {nmp[key]:12.4f}")
    print(f"  {'P_sat':12s} {nmp['P_sat']:12.3e}   (zero by construction)")


if __name__ == "__main__":
    print_nmp_summary()

