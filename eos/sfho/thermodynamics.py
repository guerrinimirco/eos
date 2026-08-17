"""
thermodynamics.py
==================
Model-dependent hadron thermodynamics for SFHo-type RMF models.

This module computes thermodynamic quantities for:
- Baryons (nucleons, hyperons, deltas) in mean-field approximation
- Pseudoscalar mesons (pions, kaons, etas) as free Bose gas

Units:
- Energies/masses: MeV
- Lengths: fm
- Number density: fm⁻³
- Pressure/energy density: MeV/fm³
- Entropy density: fm⁻³
- Meson fields: MeV

References:
- Fortin, Oertel, Providência, PASA 35 (2018) e044
- Steiner, Hempel, Fischer, ApJ 774 (2013) 17
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

from eos.general.physics_constants import hc, hc3
from eos.general.particles import Particle
from eos.general.fermi_integrals import solve_fermi_jel
from eos.general.bose_integrals import solve_bose_jel
from eos.general.state import PhaseThermo
from eos.sfho.parameters import SFHoParams


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class HadronState:
    """
    Thermodynamic state for a single hadron species.
    
    Attributes:
        n: Number density (fm⁻³) - net baryon number
        ns: Scalar density (fm⁻³) - for σ field equation
        P: Pressure contribution (MeV/fm³)
        e: Energy density contribution (MeV/fm³)
        s: Entropy density contribution (fm⁻³)
        mu_eff: Effective chemical potential (MeV)
        m_eff: Effective mass (MeV)
    """
    n: float      # Number density
    ns: float     # Scalar density
    P: float      # Pressure
    e: float      # Energy density
    s: float      # Entropy density
    mu_eff: float # Effective chemical potential
    m_eff: float  # Effective mass
    
    def __repr__(self):
        return (f"HadronState(n={self.n:.4e}, ns={self.ns:.4e}, "
                f"P={self.P:.4e}, e={self.e:.4e})")

@dataclass
class HadronThermoResult:
    """
    Complete thermodynamic result for all hadrons.
    
    Attributes:
        states: Dictionary of individual hadron states
        n_B: Total baryon number density (fm⁻³)
        n_C: Total charge density (fm⁻³)
        n_S: Total strangeness density (fm⁻³)
        P_hadrons: Total hadron pressure (MeV/fm³)
        e_hadrons: Total hadron energy density (MeV/fm³)
        s_hadrons: Total hadron entropy density (fm⁻³)
        src_sigma: Source term for σ equation (fm⁻³)
        src_omega: Source term for ω equation (fm⁻³)
        src_rho: Source term for ρ equation (fm⁻³)
        src_phi: Source term for φ equation (fm⁻³)
    """
    states: Dict[str, HadronState]
    n_B: float       # Total baryon density
    n_C: float       # Total charge density
    n_S: float       # Total strangeness density
    P_hadrons: float # Hadron pressure
    e_hadrons: float # Hadron energy density
    s_hadrons: float # Hadron entropy density
    src_sigma: float # σ source
    src_omega: float # ω source
    src_rho: float   # ρ source
    src_phi: float   # φ source

@dataclass
class MesonThermoResult:
    """
    Thermodynamic result for pseudoscalar mesons (π, K, η).
    
    Mesons are treated as free Bose gas with chemical potentials:
    - π⁺: μ = +μ_C
    - π⁻: μ = -μ_C
    - π⁰: μ = 0
    - K⁺: μ = +μ_C - μ_S
    - K⁰: μ = -μ_S
    - K⁻: μ = -μ_C + μ_S
    - K̄⁰: μ = +μ_S
    - η, η': μ = 0
    
    Attributes:
        n_C_mesons: Total meson charge density (fm⁻³)
        n_S_mesons: Total meson strangeness density (fm⁻³)
        P_mesons: Total meson pressure (MeV/fm³)
        e_mesons: Total meson energy density (MeV/fm³)
        s_mesons: Total meson entropy density (fm⁻³)
        mu_dot_n_mesons: Σ_i μ*_i n_i over the gas (MeV/fm³), at the EFFECTIVE
            potentials above. The vector fields shift each μ* but are sourced
            by the baryons alone, so this is the combination the Euler identity
            of the whole system closes on; `eos.dd2.thermodynamics.assemble`
            adds the gas the same way.
        densities: Dictionary of individual meson densities
    """
    n_C_mesons: float  # Meson charge density
    n_S_mesons: float  # Meson strangeness density
    P_mesons: float    # Meson pressure
    e_mesons: float    # Meson energy density
    s_mesons: float    # Meson entropy density
    mu_dot_n_mesons: float       # Σ_i μ*_i n_i over the gas
    densities: Dict[str, float]  # Individual meson densities


# =============================================================================
# MAIN THERMODYNAMICS FUNCTIONS
# =============================================================================


def baryon_thermo(
    T: float,
    mu_B: float, mu_C: float, mu_S: float,
    sigma: float, omega: float, rho: float, phi: float,
    particles: List[Particle],
    params: SFHoParams
) -> HadronThermoResult:
    """
    Compute thermodynamic quantities for all hadron species.
    
    Given temperature, chemical potentials, and meson fields, this function:
    1. Computes effective masses: M*_j = m_j - g_σj × σ
    2. Computes effective chemical potentials: 
       μ*_j = B_j×μ_B + C_j×μ_C + S_j×μ_S - g_ωj×ω - g_ρj×I₃j×ρ - g_φj×φ
    3. Evaluates Fermi integrals for (n, P, e, s, n_s)
    4. Computes source terms for field equations
    
    Args:
        T: Temperature (MeV)
        mu_B: Baryon chemical potential (MeV)
        mu_C: Charge chemical potential (MeV)
        mu_S: Strangeness chemical potential (MeV)
        sigma: σ-meson field (MeV)
        omega: ω-meson field (MeV)
        rho: ρ-meson field (MeV)
        phi: φ-meson field (MeV)
        particles: List of Particle objects to include
        params: SFHoParams with model parameters
        
    Returns:
        HadronThermoResult with all thermodynamic quantities
    """
    states = {}
    
    # Initialize totals
    n_B_tot = 0.0
    n_C_tot = 0.0
    n_S_tot = 0.0
    P_tot = 0.0
    e_tot = 0.0
    s_tot = 0.0
    
    # Initialize source terms
    src_sigma = 0.0
    src_omega = 0.0
    src_rho = 0.0
    src_phi = 0.0
    
    for p in particles:
        # 1. Get meson-baryon couplings
        g_s = params.get_coupling(p.name, 'sigma')
        g_w = params.get_coupling(p.name, 'omega')
        g_r = params.get_coupling(p.name, 'rho')
        g_p = params.get_coupling(p.name, 'phi')
        
        # 2. Get baryon mass from parametrization (not from Particle object)
        # This allows for different mass values in different parametrizations
        m_baryon = params.get_baryon_mass(p.name)
        if m_baryon == 0.0:
            # Fall back to Particle mass if not in parametrization
            m_baryon = p.mass
        
        # 3. Effective mass: M* = m - g_σ × σ
        m_eff = m_baryon - g_s * sigma
        
        # Ensure positive effective mass (can become negative at high density)
        if m_eff < 0:
            m_eff = 1e-3  # Small positive value
        
        # 4. Effective chemical potential
        # μ_j = B_j×μ_B + C_j×μ_C + S_j×μ_S
        mu_physical = p.baryon_no * mu_B + p.charge * mu_C + p.strangeness * mu_S
        
        # Vector field shifts
        # μ* = μ - g_ω×ω - g_ρ×I₃×ρ - g_φ×φ
        vector_shift = g_w * omega + g_r * p.isospin_3 * rho + g_p * phi
        mu_eff = mu_physical - vector_shift
        
        # 5. Compute Fermi integrals
        # solve_fermi_jel returns (n, P, e, s, ns)
        n, P, e, s, ns = solve_fermi_jel(mu_eff, T, m_eff, p.g_degen,
                                          include_antiparticles=True)
        
        # Store individual state
        states[p.name] = HadronState(
            n=n, ns=ns, P=P, e=e, s=s, mu_eff=mu_eff, m_eff=m_eff
        )
        
        # 5. Accumulate totals
        # n is the NET number density (particles - antiparticles)
        n_B_tot += p.baryon_no * n
        n_C_tot += p.charge * n
        n_S_tot += p.strangeness * n
        P_tot += P
        e_tot += e
        s_tot += s
        
        # 6. Source terms for field equations
        # σ couples to scalar density
        src_sigma += g_s * ns
        # ω, φ couple to number density
        src_omega += g_w * n
        src_phi += g_p * n
        # ρ couples to isospin-weighted density
        src_rho += g_r * p.isospin_3 * n
    
    return HadronThermoResult(
        states=states,
        n_B=n_B_tot,
        n_C=n_C_tot,
        n_S=n_S_tot,
        P_hadrons=P_tot,
        e_hadrons=e_tot,
        s_hadrons=s_tot,
        src_sigma=src_sigma,
        src_omega=src_omega,
        src_rho=src_rho,
        src_phi=src_phi
    )


def field_residuals(
    sigma: float, omega: float, rho: float, phi: float,
    src_sigma: float, src_omega: float, src_rho: float, src_phi: float,
    params: SFHoParams
) -> Tuple[float, float, float, float]:
    """
    Compute residuals of the meson field equations.
    
    The field equations are (in mean-field approximation):
    
    σ: m²_σ σ + g₂σ² + g₃σ³ - ∂A/∂σ ρ² = Σⱼ g_σⱼ n^s_j × (ℏc)³
    ω: m²_ω ω + c₃ω³ + ∂A/∂ω ρ² = Σⱼ g_ωⱼ nⱼ × (ℏc)³
    ρ: m²_ρ ρ + c₄ρ³ + 2Aρ = Σⱼ g_ρⱼ I₃ⱼ nⱼ × (ℏc)³
    φ: m²_φ φ = Σⱼ g_φⱼ nⱼ × (ℏc)³
    
    Residual = LHS - RHS (should be zero at solution)
    
    Args:
        sigma, omega, rho, phi: Meson fields (MeV)
        src_sigma, src_omega, src_rho, src_phi: Source terms (fm⁻³)
        params: Model parameters
        
    Returns:
        Tuple of (res_sigma, res_omega, res_rho, res_phi) in MeV³
    """
    # Convert sources from fm⁻³ to MeV³
    # Source × (ℏc)³ gives MeV³
    
    # σ equation
    # m²σ + g₂σ² + g₃σ³ - (∂A/∂σ)ρ² = g_σ n_s × hc³
    dU_dsigma = params.g2 * sigma**2 + params.g3 * sigma**3
    dA_dsigma = params.compute_dA_dsigma(sigma)
    
    lhs_sigma = params.m_sigma**2 * sigma + dU_dsigma - dA_dsigma * rho**2
    rhs_sigma = src_sigma * hc3
    res_sigma = lhs_sigma - rhs_sigma
    
    # ω equation
    # m²ω + c₃ω³ + (∂A/∂ω)ρ² = g_ω n × hc³
    dA_domega = params.compute_dA_domega(omega)
    
    lhs_omega = params.m_omega**2 * omega + params.c3 * omega**3 + dA_domega * rho**2
    rhs_omega = src_omega * hc3
    res_omega = lhs_omega - rhs_omega
    
    # ρ equation
    # m²ρ + c₄ρ³ + 2Aρ = g_ρ I₃ n × hc³
    A = params.compute_A(sigma, omega)
    
    lhs_rho = params.m_rho**2 * rho + params.c4 * rho**3 + 2.0 * A * rho
    rhs_rho = src_rho * hc3
    res_rho = lhs_rho - rhs_rho
    
    # φ equation (linear, no self-interactions)
    # m²φ = g_φ n × hc³
    lhs_phi = params.m_phi**2 * phi
    rhs_phi = src_phi * hc3
    res_phi = lhs_phi - rhs_phi
    
    return res_sigma, res_omega, res_rho, res_phi


def meson_field_thermo(
    sigma: float, omega: float, rho: float, phi: float,
    params: SFHoParams
) -> Tuple[float, float]:
    """
    Compute meson field contributions to pressure and energy density.
    
    The meson Lagrangian contributes to the thermodynamics:

    P_meson = -V(σ) + ½m²_ω ω² + (c₃/4)ω⁴
              + ½m²_ρ ρ² + (c₄/4)ρ⁴ + Aρ²
              + ½m²_φ φ²

    e_meson = +V(σ) + ½m²_ω ω² + (3c₃/4)ω⁴ + ω(∂A/∂ω)ρ²
              + ½m²_ρ ρ² + (3c₄/4)ρ⁴ + Aρ²
              + ½m²_φ φ²

    where V(σ) = ½m²_σ σ² + (g₂/3)σ³ + (g₃/4)σ⁴

    Note: The sign conventions follow from the mean-field Lagrangian.
    The attractive σ field contributes negatively to pressure.

    The ω(∂A/∂ω)ρ² term in e_meson is the partner of the ∂A/∂ω ρ² source in
    the ω field equation (`field_residuals`; Fortin, Oertel & Providência
    2018, Eq. 8, and Steiner, Prakash, Lattimer & Ellis 2005, Eq. 56). A is a
    function of ω as well as σ — A(σ,ω) = g²_ρN [Σᵢ aᵢσⁱ + Σⱼ bⱼω^2ʲ] — so
    eliminating the ω source through its own field equation, which is what
    turns ½m²_ω ω² into ½m²_ω ω² + (3c₃/4)ω⁴, leaves this term behind as well:

        e_ω = -½m²_ω ω² - (c₃/4)ω⁴ + ω·src_ω
            = ½m²_ω ω² + (3c₃/4)ω⁴ + ω(∂A/∂ω)ρ²

    Keeping only b₁ gives A = g²_ρ b₁ ω², so ω(∂A/∂ω)ρ² = 2Aρ² and the cross
    term enters e as 3Aρ² against Aρ² in P — the familiar factor of three of
    the Horowitz-Piekarewicz Λ_v ω²ρ² coupling this generalises. σ has no such
    partner: it reaches the baryons through m* and the scalar density rather
    than through μ, so V(σ) cancels between e and P and the aᵢ keep
    coefficient one.

    Args:
        sigma, omega, rho, phi: Meson fields (MeV)
        params: Model parameters
        
    Returns:
        Tuple of (P_meson, e_meson) in MeV/fm³
    """
    # Full scalar potential V(σ) including mass term
    sigma_sq = sigma**2
    V_sigma = (0.5 * params.m_sigma**2 * sigma_sq 
               + (params.g2 / 3.0) * sigma**3 
               + (params.g3 / 4.0) * sigma**4)
    
    # Vector contributions
    omega_sq = omega**2
    rho_sq = rho**2
    phi_sq = phi**2
    
    # ρ contribution (including A-function)
    A = params.compute_A(sigma, omega)
    dA_domega = params.compute_dA_domega(omega)

    # ω contribution. The energy density also carries ω(∂A/∂ω)ρ², the partner
    # of the ∂A/∂ω ρ² source in the ω field equation — see the docstring.
    P_omega = 0.5 * params.m_omega**2 * omega_sq + (params.c3 / 4.0) * omega**4
    e_omega = (0.5 * params.m_omega**2 * omega_sq
               + (3.0 * params.c3 / 4.0) * omega**4
               + omega * dA_domega * rho_sq)

    P_rho = 0.5 * params.m_rho**2 * rho_sq + (params.c4 / 4.0) * rho**4 + A * rho_sq
    e_rho = 0.5 * params.m_rho**2 * rho_sq + (3.0 * params.c4 / 4.0) * rho**4 + A * rho_sq
    
    # φ contribution
    P_phi = 0.5 * params.m_phi**2 * phi_sq
    e_phi = 0.5 * params.m_phi**2 * phi_sq
    
    # Total (in MeV⁴, need to convert to MeV/fm³)
    # Divide by (ℏc)³ to get MeV/fm³
    P_meson = (-V_sigma + P_omega + P_rho + P_phi) / hc3
    e_meson = (V_sigma + e_omega + e_rho + e_phi) / hc3
    
    return P_meson, e_meson


def thermal_meson_thermo(
    T: float,
    mu_C: float, mu_S: float,
    omega: float, rho: float,
    params: SFHoParams,
    include_pions: bool = True,
    include_kaons: bool = True,
    include_etas: bool = True
) -> MesonThermoResult:
    """
    Compute thermodynamic quantities for pseudoscalar mesons (π, K, η).
    
    Mesons are treated as free Bose gas with effective chemical potentials
    shifted by the vector meson fields:
    
    Pions (I=1, S=0):
        π⁺: μ_eff = +μ_C - g_ρN × ρ
        π⁻: μ_eff = -μ_C + g_ρN × ρ
        π⁰: μ_eff = 0
        
    Kaons (I=1/2, S=±1):
        K⁺:  μ_eff = +μ_C - μ_S - (g_ωN - g_ωΛ) × ω - (1/2) g_ρN × ρ
        K⁰:  μ_eff = -μ_S - (g_ωN - g_ωΛ) × ω + (1/2) g_ρN × ρ
        K⁻:  μ_eff = -μ_C + μ_S + (g_ωN - g_ωΛ) × ω + (1/2) g_ρN × ρ
        K̄⁰:  μ_eff = +μ_S + (g_ωN - g_ωΛ) × ω - (1/2) g_ρN × ρ
        
    Eta (I=0, S=0):
        η, η': μ_eff = 0
    
    Note: Bose-Einstein condensation occurs when μ_eff → m. This function
    does not handle the condensed phase.
    
    Args:
        T: Temperature (MeV)
        mu_C: Charge chemical potential (MeV)
        mu_S: Strangeness chemical potential (MeV)
        omega: ω-meson field (MeV)
        rho: ρ-meson field (MeV)
        params: SFHoParams (for meson masses and couplings)
        include_pions: Include π mesons
        include_kaons: Include K mesons
        include_etas: Include η, η' mesons
        
    Returns:
        MesonThermoResult with all meson thermodynamics
    """
    # Initialize totals
    n_C_tot = 0.0
    n_S_tot = 0.0
    P_tot = 0.0
    e_tot = 0.0
    s_tot = 0.0
    # Σ_i μ*_i n_i. Only the charged and strange members contribute: π⁰, η and
    # η' sit at μ* = 0 exactly, which is why an energy-density term dropped for
    # one of them shows up in the Euler identity and nowhere else.
    mu_dot_n = 0.0
    densities = {}

    if T <= 0:
        return MesonThermoResult(
            n_C_mesons=0.0, n_S_mesons=0.0,
            P_mesons=0.0, e_mesons=0.0, s_mesons=0.0,
            mu_dot_n_mesons=0.0,
            densities={}
        )
    
    # Get relevant couplings from params
    g_rho_N = params.get_coupling('n', 'rho')  # Nucleon-rho coupling
    g_omega_N = params.get_coupling('n', 'omega')  # Nucleon-omega coupling
    g_omega_Lambda = params.get_coupling('Lambda', 'omega')  # Lambda-omega coupling
    
    # Omega shift for kaons: (g_ωN - g_ωΛ)
    delta_g_omega = g_omega_N - g_omega_Lambda
    
    # Pions (g=1 for each, spin-0)
    if include_pions:
        m_pi = params.m_pi_pm
        
        # π⁺: μ_eff = +μ_C - g_ρN × ρ
        mu_pip_eff = mu_C - g_rho_N * rho
        if abs(mu_pip_eff) < m_pi:  # No condensation
            n_pip, P_pip, e_pip, s_pip, _ = solve_bose_jel(mu_pip_eff, T, m_pi, g=1.0, include_antiparticles=False)
            densities['pi+'] = n_pip
            n_C_tot += n_pip  # Q = +1
            mu_dot_n += mu_pip_eff * n_pip
            P_tot += P_pip
            e_tot += e_pip
            s_tot += s_pip
        else:
            densities['pi+'] = 0.0
        
        # π⁻: μ_eff = -μ_C + g_ρN × ρ
        mu_pim_eff = -mu_C + g_rho_N * rho
        if abs(mu_pim_eff) < m_pi:
            n_pim, P_pim, e_pim, s_pim, _ = solve_bose_jel(mu_pim_eff, T, m_pi, g=1.0, include_antiparticles=False)
            densities['pi-'] = n_pim
            n_C_tot -= n_pim  # Q = -1
            mu_dot_n += mu_pim_eff * n_pim
            P_tot += P_pim
            e_tot += e_pim
            s_tot += s_pim
        else:
            densities['pi-'] = 0.0
        
        # π⁰: μ_eff = 0
        m_pi0 = params.m_pi_0
        n_pi0, P_pi0, e_pi0, s_pi0, _ = solve_bose_jel(0.0, T, m_pi0, g=1.0, include_antiparticles=False)
        densities['pi0'] = n_pi0
        P_tot += P_pi0
        e_tot += e_pi0
        s_tot += s_pi0
    
    # Kaons (g=1 for each)
    if include_kaons:
        m_k_pm = params.m_kaon_pm
        m_k_0 = params.m_kaon_0
        
        # K⁺ (us̄): Q=+1, S=-1 → μ_eff = μ_C - μ_S - Δg_ω × ω - (1/2) g_ρN × ρ
        mu_kp_eff = mu_C - mu_S - delta_g_omega * omega - 0.5 * g_rho_N * rho
        if abs(mu_kp_eff) < m_k_pm:
            n_kp, P_kp, e_kp, s_kp, _ = solve_bose_jel(mu_kp_eff, T, m_k_pm, g=1.0, include_antiparticles=False)
            densities['K+'] = n_kp
            n_C_tot += n_kp      # Q = +1
            n_S_tot -= n_kp      # S = -1
            mu_dot_n += mu_kp_eff * n_kp
            P_tot += P_kp
            e_tot += e_kp
            s_tot += s_kp
        else:
            densities['K+'] = 0.0
        
        # K⁰ (ds̄): Q=0, S=-1 → μ_eff = -μ_S - Δg_ω × ω + (1/2) g_ρN × ρ
        mu_k0_eff = -mu_S - delta_g_omega * omega + 0.5 * g_rho_N * rho
        if abs(mu_k0_eff) < m_k_0:
            n_k0, P_k0, e_k0, s_k0, _ = solve_bose_jel(mu_k0_eff, T, m_k_0, g=1.0, include_antiparticles=False)
            densities['K0'] = n_k0
            n_S_tot -= n_k0      # S = -1
            mu_dot_n += mu_k0_eff * n_k0
            P_tot += P_k0
            e_tot += e_k0
            s_tot += s_k0
        else:
            densities['K0'] = 0.0
        
        # K⁻ (ūs): Q=-1, S=+1 → μ_eff = -μ_C + μ_S + Δg_ω × ω + (1/2) g_ρN × ρ
        mu_km_eff = -mu_C + mu_S + delta_g_omega * omega + 0.5 * g_rho_N * rho
        if abs(mu_km_eff) < m_k_pm:
            n_km, P_km, e_km, s_km, _ = solve_bose_jel(mu_km_eff, T, m_k_pm, g=1.0, include_antiparticles=False)
            densities['K-'] = n_km
            n_C_tot -= n_km      # Q = -1
            n_S_tot += n_km      # S = +1
            mu_dot_n += mu_km_eff * n_km
            P_tot += P_km
            e_tot += e_km
            s_tot += s_km
        else:
            densities['K-'] = 0.0
        
        # K̄⁰ (d̄s): Q=0, S=+1 → μ_eff = +μ_S + Δg_ω × ω - (1/2) g_ρN × ρ
        mu_k0bar_eff = mu_S + delta_g_omega * omega - 0.5 * g_rho_N * rho
        if abs(mu_k0bar_eff) < m_k_0:
            n_k0bar, P_k0bar, e_k0bar, s_k0bar, _ = solve_bose_jel(mu_k0bar_eff, T, m_k_0, g=1.0, include_antiparticles=False)
            densities['K0_bar'] = n_k0bar
            n_S_tot += n_k0bar   # S = +1
            mu_dot_n += mu_k0bar_eff * n_k0bar
            P_tot += P_k0bar
            e_tot += e_k0bar
            s_tot += s_k0bar
        else:
            densities['K0_bar'] = 0.0
    
    # Eta mesons (g=1, μ_eff=0)
    if include_etas:
        # η
        m_eta = params.m_eta
        n_eta, P_eta, e_eta, s_eta, _ = solve_bose_jel(0.0, T, m_eta, g=1.0, include_antiparticles=False)
        densities['eta'] = n_eta
        P_tot += P_eta
        e_tot += e_eta
        s_tot += s_eta
        
        # η'
        m_etap = params.m_eta_prime
        n_etap, P_etap, e_etap, s_etap, _ = solve_bose_jel(0.0, T, m_etap, g=1.0, include_antiparticles=False)
        densities['eta_prime'] = n_etap
        P_tot += P_etap
        e_tot += e_etap
        s_tot += s_etap
    
    return MesonThermoResult(
        n_C_mesons=n_C_tot,
        n_S_mesons=n_S_tot,
        P_mesons=P_tot,
        e_mesons=e_tot,
        s_mesons=s_tot,
        mu_dot_n_mesons=mu_dot_n,
        densities=densities
    )


def thermo_from_mu(
    mu_B: float, mu_C: float, mu_S: float,
    sigma: float, omega: float, rho: float, phi: float,
    T: float,
    particles: List[Particle],
    params: SFHoParams,
    include_pseudoscalar_mesons: bool = False
) -> PhaseThermo:
    """
    The matter state at (μ_B, μ_C, μ_S, σ, ω, ρ, φ, T), as a `PhaseThermo`.

    MATTER ONLY — baryons, the meson mean fields and any thermal π/K/η gas.
    No leptons and no photons: those are shared by the whole system rather
    than belonging to a phase, so `solver.py` adds them.

    The gas carries electric charge and strangeness, so it enters n_C and n_S
    (CLAUDE.md §2) — through `extra_charges`, because most of its members are
    not yet in `eos.general.particles` and so cannot be summed as species.
    `eos.dd2.thermodynamics.assemble` does the same, and both are the same
    record, so a caller reads either model's state the same way.

    Σᵢ μᵢ nᵢ is supplied rather than derived: the baryons enter at their full
    potentials μᵢ = Bᵢμ_B + Cᵢμ_C + Sᵢμ_S, but the gas enters at its EFFECTIVE
    potentials, since the ω and ρ shifts it carries have no partner in the
    field energy to cancel against (they are sourced by the baryons alone).

    SFHo's couplings are constants, so there is no rearrangement self-energy:
    Σ^R = 0, and it is stated rather than omitted.

    Args:
        mu_B, mu_C, mu_S: Conserved charge chemical potentials (MeV)
        sigma, omega, rho, phi: Meson fields (MeV)
        T: Temperature (MeV)
        particles: List of baryon species
        params: Model parameters
        include_pseudoscalar_mesons: Include π, K, η contributions

    Returns:
        PhaseThermo — the shared record of `eos.general.state`
    """
    # Compute baryon thermodynamics
    hadron_result = baryon_thermo(
        T, mu_B, mu_C, mu_S, sigma, omega, rho, phi, particles, params
    )

    # Mean-field meson contributions (σ, ω, ρ, φ)
    P_mf, e_mf = meson_field_thermo(sigma, omega, rho, phi, params)

    # Total from baryons + mean-field mesons
    P_total = hadron_result.P_hadrons + P_mf
    e_total = hadron_result.e_hadrons + e_mf
    s_total = hadron_result.s_hadrons

    densities, mu_eff_i, m_eff_i = {}, {}, {}
    mu_dot_n = 0.0
    for p in particles:
        st = hadron_result.states[p.name]
        densities[p.name] = st.n
        mu_eff_i[p.name] = st.mu_eff
        m_eff_i[p.name] = st.m_eff
        mu_dot_n += (p.baryon_no * mu_B + p.charge * mu_C
                     + p.strangeness * mu_S) * st.n

    gas_C = gas_S = 0.0
    # Optional: pseudoscalar mesons (π, K, η)
    if include_pseudoscalar_mesons:
        meson_result = thermal_meson_thermo(T, mu_C, mu_S, omega, rho, params)
        P_total += meson_result.P_mesons
        e_total += meson_result.e_mesons
        s_total += meson_result.s_mesons
        gas_C = meson_result.n_C_mesons
        gas_S = meson_result.n_S_mesons
        mu_dot_n += meson_result.mu_dot_n_mesons

    return PhaseThermo.assemble(
        T=T, mu_B=mu_B, mu_C=mu_C, mu_S=mu_S,
        fields={"sigma": sigma, "omega": omega, "rho": rho, "phi": phi},
        densities=densities, mu_eff_i=mu_eff_i, m_eff_i=m_eff_i,
        P=P_total, eps=e_total, s=s_total, mu_dot_n=mu_dot_n,
        Sigma_R=0.0,
        extra_charges=(0.0, gas_C, gas_S),
    )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def get_residual_vector(
    fields: np.ndarray,
    T: float,
    mu_B: float, mu_C: float, mu_S: float,
    particles: List[Particle],
    params: SFHoParams
) -> np.ndarray:
    """
    Compute residual vector for self-consistent field solver.
    
    This function is designed to be used with scipy.optimize.fsolve or similar.
    
    Args:
        fields: Array [sigma, omega, rho, phi] of meson fields (MeV)
        T: Temperature (MeV)
        mu_B, mu_C, mu_S: Chemical potentials (MeV)
        particles: List of hadron species
        params: Model parameters
        
    Returns:
        Array of residuals [res_σ, res_ω, res_ρ, res_φ]
    """
    sigma, omega, rho, phi = fields
    
    # Compute hadron thermodynamics to get source terms
    result = baryon_thermo(
        T, mu_B, mu_C, mu_S, sigma, omega, rho, phi, particles, params
    )
    
    # Compute field equation residuals
    res_sigma, res_omega, res_rho, res_phi = field_residuals(
        sigma, omega, rho, phi,
        result.src_sigma, result.src_omega, result.src_rho, result.src_phi,
        params
    )
    
    return np.array([res_sigma, res_omega, res_rho, res_phi])


# =============================================================================
# SELF-TEST
# =============================================================================
