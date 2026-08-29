"""
sfho_parameters.py
==================
SFHo Relativistic Mean Field model parameters.

Contains parametrizations for:
- Nucleonic matter (Steiner et al. 2013)
- Hyperonic matter SFHoY (Fortin et al. 2017)
- Hyperonic matter SFHoY* with SU(6) vector couplings (Fortin et al. 2017)
- Hyperons + Deltas (SFHo_HD) as in Mathematica notebook
- General parametrization with customizable scalar couplings

Units:
- Masses: MeV
- Couplings: dimensionless (g)
- Length: fm

References:
- Steiner, Hempel, Fischer, ApJ 774 (2013) 17
- Fortin, Oertel, Providência, PASA 35 (2018) e044
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict
from eos.general.physics_constants import hc


# =============================================================================
# SU(6) SYMMETRY RATIOS FOR VECTOR MESONS
# =============================================================================
# These are the standard SU(6) ratios relative to nucleon couplings
# Based on quark counting and ideal ω-φ mixing

SQRT2 = np.sqrt(2.0)

# SU(6) ratios: R_M = g_MH / g_MN (or g_MH / g_ωN for φ)
SU6_RATIOS = {
    # Lambda (uds): I=0, S=1 (in our convention)
    'lambda': {
        'omega': 2.0/3.0,
        'rho': 0.0,
        'phi': -SQRT2/3.0,  # ≈ -0.4714
    },
    # Sigma (uus, uds, dds): I=1, S=1
    'sigma': {
        'omega': 2.0/3.0,
        'rho': 2.0,  # Isospin factor
        'phi': -SQRT2/3.0,
    },
    # Xi (uss, dss): I=1/2, S=2
    'xi': {
        'omega': 1.0/3.0,
        'rho': 1.0,
        'phi': -2.0*SQRT2/3.0,  # ≈ -0.9428
    },
    # Delta (uuu, uud, udd, ddd): I=3/2, S=0
    'delta': {
        'omega': 1.0,
        'rho': 1.0,
        'phi': 0.0,  # No strangeness
    },
}


#: Which SU(6) multiplet each hyperon belongs to, and hence which of the nine
#: breaking factors y_M_H scales its vector couplings.
MULTIPLET = {
    'lambda': 'Lambda',
    'sigma+': 'Sigma', 'sigma0': 'Sigma', 'sigma-': 'Sigma',
    'xi0': 'Xi', 'xi-': 'Xi',
}

#: The key each multiplet carries in `SU6_RATIOS`.
_SU6_KEY = {'Lambda': 'lambda', 'Sigma': 'sigma', 'Xi': 'xi'}


def vector_ratios(multiplet, y_omega, y_rho, y_phi):
    """(x_omega, x_rho, x_phi) of a hyperon multiplet: SU(6) times a factor.

        x_omegaY = y_omega * SU6_RATIOS[..]['omega']
        x_rhoY   = y_rho   * SU6_RATIOS[..]['rho']
        x_phiY   = y_phi   * SU6_RATIOS[..]['phi']

    SU(6) spin-flavour symmetry with ideal omega-phi mixing is a quark-model
    ASSUMPTION, not a measurement, so each vertex carries a factor a sampler
    may vary; y = 1 is SU(6) exactly, which is SFHoY*. SFHoY breaks it exactly
    in this form: Fortin, Oertel & Providencia, PASA 35 (2018) e044 §2.2 reads
    "we rescale the omega and phi-meson hyperon couplings as follows:
    g_MLambda = 1.5 g_MLambda(SU(6)), g_MSigma = 1.5 g_MSigma(SU(6)),
    g_MXi = 1.875 g_MXi(SU(6))" -- omega and phi, per multiplet, with rho left
    at SU(6). Their Table 1 lists the result (R_omegaLambda = 1,
    R_omegaXi = 0.62, R_phiXi = -1.77), which is where these factors are taken
    from rather than from the arithmetic 1.5 x 2/3 = 1 coming out round.

    **y_phi multiplies a NEGATIVE ratio** (-sqrt(2)/3 for Lambda and Sigma,
    -2 sqrt(2)/3 for Xi), so y_phi > 1 makes g_phiY MORE negative, not more
    repulsive-looking; the phi repulsion in matter goes as g_phiY^2 either
    way. `y_phi = 0` in every multiplet is a set with no hidden-strange
    sector at all -- which is what SFHo_2fam is.
    """
    su6 = SU6_RATIOS[_SU6_KEY[multiplet]]
    return (su6['omega'] * y_omega, su6['rho'] * y_rho, su6['phi'] * y_phi)

# =============================================================================
# PARAMETER DATACLASS
# =============================================================================
@dataclass
class Parameters:
    """
    Holds the parameters for the SFHo Relativistic Mean Field model.
    
    The model includes:
    - Nucleon-meson couplings
    - Non-linear σ-meson self-interactions
    - Vector meson self-interactions (ω, ρ)
    - Symmetry energy A-function (σ-ω-ρ mixing)
    - Hyperon/Delta couplings via ratio maps
    
    Note on masses:
    - Baryon masses here are the ones used in RMF calculations
    - They may differ slightly from PDG values in particles.py
    - particles.py contains reference PDG masses
    - These parametrization masses should be used for thermodynamics
    
    Field equations (mean-field approximation):
        m_σ² σ + g₂σ² + g₃σ³ = Σⱼ g_σⱼ n^s_j × (ℏc)³ + ∂A/∂σ ρ²
        m_ω² ω + c₃ω³ = Σⱼ g_ωⱼ nⱼ × (ℏc)³ - ∂A/∂ω ρ²
        m_ρ² ρ + c₄ρ³ + 2Aρ = Σⱼ g_ρⱼ I₃ⱼ nⱼ × (ℏc)³
        m_φ² φ = Σⱼ g_φⱼ nⱼ × (ℏc)³
    """
    # Name identifier for the parametrization
    name: str = "SFHo"
    
    # ---------------------------------------------------------
    # 1. Baryon Masses (MeV) - used in RMF calculations
    # ---------------------------------------------------------
    # Nucleons
    m_n: float = 939.565346   # Neutron mass
    m_p: float = 938.272013   # Proton mass
    
    # Hyperons (Fortin 2017 values for SFHoY)
    m_lambda: float = 1116.0      # Λ
    m_sigma_p: float = 1189.0     # Σ⁺
    m_sigma_0: float = 1193.0     # Σ⁰
    m_sigma_m: float = 1197.0     # Σ⁻
    m_xi_0: float = 1315.0        # Ξ⁰
    m_xi_m: float = 1321.0        # Ξ⁻
    
    # Delta resonances
    m_delta: float = 1232.0       # All Δ states (same mass approximation)
    
    # ---------------------------------------------------------
    # 2. Meson Masses (MeV) - mean-field mesons
    # ---------------------------------------------------------
    # Values from CompOSE table (fm^-1) converted to MeV using hc
    # m [MeV] = m [fm^-1] × hc
    m_sigma: float = 2.3689528914 * hc   # σ (scalar-isoscalar)  467.458 MeV
    m_omega: float = 3.9655047020 * hc   # ω (vector-isoscalar)  782.501 MeV
    m_rho: float = 3.8666788766 * hc     # ρ (vector-isovector)  763.000 MeV
    m_phi: float = 1020.0                # φ (vector-isoscalar, hidden strangeness)
    
    # ---------------------------------------------------------
    # 3. Nucleon Couplings (dimensionless g)
    # ---------------------------------------------------------
    g_sigma_N: float = 0.0
    g_omega_N: float = 0.0
    g_rho_N: float = 0.0
    g_phi_N: float = 0.0   # Usually 0 for nucleons

    # ---------------------------------------------------------
    # 4. Non-linear Scalar Potential Parameters
    # ---------------------------------------------------------
    # U(σ) = (g₂/3)σ³ + (g₃/4)σ⁴
    # dU/dσ = g₂σ² + g₃σ³
    g2: float = 0.0  # MeV (dimension of mass)
    g3: float = 0.0  # dimensionless
    
    # ---------------------------------------------------------
    # 5. Vector Self-Interaction Parameters
    # ---------------------------------------------------------
    # L = ... + (c₃/4)ω⁴ + (c₄/4)ρ⁴
    # Field eq: m²ω + c₃ω³ = ...
    c3: float = 0.0  # coefficient for ω⁴ term
    c4: float = 0.0  # coefficient for ρ⁴ term

    # ---------------------------------------------------------
    # 6. Symmetry Energy A-function Coefficients
    # ---------------------------------------------------------
    # A(σ,ω) = Σᵢ aᵢσⁱ + Σⱼ bⱼω^(2j)
    # Affects ρ-meson field equation and symmetry energy
    a_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(7))
    b_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(4))

    # ---------------------------------------------------------
    # 7. Couplings for Hyperons/Deltas
    # ---------------------------------------------------------
    # couplings_map[particle_name][meson] = absolute coupling value, and it
    # holds only what is FITTED: for a hyperon that is the scalar coupling
    # alone, g_sigmaH, fixed by the potential depth U_H below. The three
    # vector couplings are derived from SU(6) times the nine factors, by
    # `get_coupling`, so a factor has one place to be changed. Deltas carry
    # all four entries: their scalar has no measured depth in this model.
    # An entry's PRESENCE is what declares the sector (see species.py).
    couplings_map: Dict[str, Dict[str, float]] = field(default_factory=dict)

    #: Hyperon single-particle potentials in SNM at saturation [MeV], the
    #: measurement the scalar couplings above are inverted from
    #: (`nmp.from_potential_depths`). Fortin, Oertel & Providencia 2018 take
    #: -30 / +30 / -14 for SFHoY and SFHoY*.
    U_Lambda: float = -30.0
    U_Sigma: float = 30.0
    U_Xi: float = -14.0

    #: SU(6)-breaking factors for the hyperon VECTOR couplings, one per
    #: (meson, multiplet) pair: x_MY = y_M_Y * SU(6). y = 1 everywhere IS
    #: SU(6), which is SFHoY*. Nine, not three, because SFHoY scales omega
    #: AND phi by 1.5 / 1.5 / 1.875 while leaving rho at SU(6) -- a per-meson
    #: factor could not express that, and nor could a per-multiplet one for a
    #: set that broke omega and phi differently. `y_phi_*` multiplies a
    #: NEGATIVE ratio; `y_phi_* = 0` is a set with no phi sector, which is
    #: SFHo_2fam. See `vector_ratios`.
    y_omega_Lambda: float = 1.0
    y_omega_Sigma: float = 1.0
    y_omega_Xi: float = 1.0
    y_rho_Lambda: float = 1.0
    y_rho_Sigma: float = 1.0
    y_rho_Xi: float = 1.0
    y_phi_Lambda: float = 1.0
    y_phi_Sigma: float = 1.0
    y_phi_Xi: float = 1.0

    @property
    def su6_breaking(self):
        """{multiplet: (y_omega, y_rho, y_phi)} — the nine factors as a table."""
        return {
            'Lambda': (self.y_omega_Lambda, self.y_rho_Lambda, self.y_phi_Lambda),
            'Sigma': (self.y_omega_Sigma, self.y_rho_Sigma, self.y_phi_Sigma),
            'Xi': (self.y_omega_Xi, self.y_rho_Xi, self.y_phi_Xi),
        }

    def get_coupling(self, particle_name: str, meson: str) -> float:
        """
        Returns the meson-baryon coupling constant g_{MB}.
        
        Args:
            particle_name: Particle name (case-insensitive)
            meson: 'sigma', 'omega', 'rho', or 'phi'
            
        Returns:
            Coupling constant (dimensionless)

        A hyperon's three VECTOR couplings are not stored: they are SU(6)
        times the breaking factor of that (meson, multiplet) pair, evaluated
        here, so changing a factor moves the coupling with nothing else to
        update. Changing a `y_omega_*` at fixed g_sigmaH MOVES the potential
        depth, because U_H = -g_sigmaH sigma + g_omegaH omega holds both; to
        hold the depth instead, re-run `nmp.from_potential_depths` on the
        rescaled par. Which of the two is held is the caller's physics
        (`nmp.invert_nmp`'s `hold_hyperons` argument).
        """
        p_name = particle_name.lower()

        # The coupling map holds what is fitted; the SU(6) table supplies the
        # hyperon vector couplings around it.
        if p_name in self.couplings_map:
            if meson in self.couplings_map[p_name]:
                return self.couplings_map[p_name][meson]
            if p_name in MULTIPLET:
                mult = MULTIPLET[p_name]
                x_omega, x_rho, x_phi = vector_ratios(
                    mult, *self.su6_breaking[mult])
                if meson == 'omega':
                    return x_omega * self.g_omega_N
                if meson == 'rho':
                    return x_rho * self.g_rho_N
                if meson == 'phi':
                    return x_phi * self.g_omega_N
            return 0.0

        # Handle nucleons
        if p_name in ['n', 'p', 'neutron', 'proton']:
            if meson == 'sigma':
                return self.g_sigma_N
            elif meson == 'omega':
                return self.g_omega_N
            elif meson == 'rho':
                return self.g_rho_N
            elif meson == 'phi':
                return self.g_phi_N
        
        return 0.0
    
    def compute_A(self, sigma: float, omega: float) -> float:
        """
        Compute the A-function for symmetry energy.
        
        A(σ,ω) = g_ρN² × f(σ,ω) = g_ρN² × [Σᵢ aᵢσⁱ + Σⱼ bⱼω^(2j)]
        
        Note: CompOSE coefficients define f (Steiner 2005 Eq. 15), 
        and A = g_ρ² × f per Fortin 2017.
        """
        f = 0.0
        for i in range(1, len(self.a_coeffs)):
            f += self.a_coeffs[i] * sigma**i
        for j in range(1, len(self.b_coeffs)):
            f += self.b_coeffs[j] * omega**(2*j)
        return self.g_rho_N**2 * f
    
    def compute_dA_dsigma(self, sigma: float) -> float:
        """Compute ∂A/∂σ = g_ρN² × ∂f/∂σ"""
        df = 0.0
        for i in range(1, len(self.a_coeffs)):
            df += i * self.a_coeffs[i] * sigma**(i-1)
        return self.g_rho_N**2 * df
    
    def compute_dA_domega(self, omega: float) -> float:
        """Compute ∂A/∂ω = g_ρN² × ∂f/∂ω"""
        df = 0.0
        for j in range(1, len(self.b_coeffs)):
            df += 2*j * self.b_coeffs[j] * omega**(2*j - 1)
        return self.g_rho_N**2 * df

    def compute_d2A_dsigma2(self, sigma: float) -> float:
        """Compute ∂²A/∂σ² = g_ρN² × ∂²f/∂σ².

        Needed by `nmp.snm_derivatives`, which differentiates E_sym twice
        along the density axis and so meets A through σ(n) and ω(n). There
        is no ∂²A/∂σ∂ω: f = Σᵢ aᵢσⁱ + Σⱼ bⱼω^(2j) is separable, and that is
        a property of the SFHo form rather than a truncation.
        """
        d2f = 0.0
        for i in range(2, len(self.a_coeffs)):
            d2f += i * (i - 1) * self.a_coeffs[i] * sigma**(i - 2)
        return self.g_rho_N**2 * d2f

    def compute_d2A_domega2(self, omega: float) -> float:
        """Compute ∂²A/∂ω² = g_ρN² × ∂²f/∂ω²; see `compute_d2A_dsigma2`."""
        d2f = 0.0
        for j in range(1, len(self.b_coeffs)):
            d2f += 2*j * (2*j - 1) * self.b_coeffs[j] * omega**(2*j - 2)
        return self.g_rho_N**2 * d2f
    
    def get_baryon_mass(self, particle_name: str) -> float:
        """
        Get the baryon mass for RMF calculations.
        
        These masses are specific to the parametrization and may differ
        from PDG values. Use these for thermodynamic calculations.
        
        Args:
            particle_name: Particle name (case-insensitive)
            
        Returns:
            Mass in MeV
        """
        p_name = particle_name.lower()
        
        # Nucleons
        if p_name in ['p', 'proton']:
            return self.m_p
        elif p_name in ['n', 'neutron']:
            return self.m_n
        # Hyperons
        elif p_name == 'lambda':
            return self.m_lambda
        elif p_name in ['sigma+', 'sigmap']:
            return self.m_sigma_p
        elif p_name in ['sigma0']:
            return self.m_sigma_0
        elif p_name in ['sigma-', 'sigmam']:
            return self.m_sigma_m
        elif p_name in ['xi0']:
            return self.m_xi_0
        elif p_name in ['xi-', 'xim']:
            return self.m_xi_m
        # Deltas
        elif p_name.startswith('delta'):
            return self.m_delta
        else:
            return 0.0

    @classmethod
    def default(cls) -> "Parameters":
        """The nucleon-only SFHo table (Steiner, Hempel & Fischer 2013).

        The CompOSE table values, with no hyperon or Delta couplings declared;
        it is the set `nmp.py` reports the published nuclear-matter parameters
        against and the one `test/baseline` is frozen at.
        """
        return _nucleonic()

    @classmethod
    def named(cls, name: str) -> "Parameters":
        """One of the published sets; see `PUBLISHED_SETS` for what each is."""
        if name not in PUBLISHED_SETS:
            raise KeyError(f"unknown SFHo parameter set {name!r}; published: "
                           f"{sorted(PUBLISHED_SETS)}")
        return PUBLISHED_SETS[name]()


# =============================================================================
# BASE SFHo PARAMETERS (CompOSE table values)
# =============================================================================
def _get_base_sfho() -> Parameters:
    """
    Returns base SFHo nuclear parameters from CompOSE table.
    
    Reference: Steiner, Hempel, Fischer, ApJ 774 (2013) 17
    CompOSE table parameters for exact reproducibility.
    
    Nuclear matter properties at saturation:
    - n_sat = 0.1583 fm⁻³
    - E_0 = 16.19 MeV
    - K = 245.4 MeV
    - J = 31.57 MeV
    - L = 47.10 MeV
    - K_sym = -205.4 MeV
    """
    p = Parameters(name="SFHo")
    
    # Couplings from CompOSE table (c = g/m in fm)
    # g = c × m / hc
    c_sigma = 3.1791606374  # fm
    c_omega = 2.2752188529  # fm
    c_rho = 2.4062374629    # fm
    
    p.g_sigma_N = c_sigma / hc * p.m_sigma
    p.g_omega_N = c_omega / hc * p.m_omega
    p.g_rho_N = c_rho / hc * p.m_rho
    p.g_phi_N = 0.0  # Nucleons don't couple to φ

    # Non-linear σ potential parameters from CompOSE table
    # U = (b·M·g³_σ/3)σ³ + (c·g⁴_σ/4)σ⁴
    b_val = 7.3536466626e-3
    c_val = -3.8202821956e-3
    p.g2 = b_val * p.m_n * (p.g_sigma_N**3)
    p.g3 = c_val * (p.g_sigma_N**4)

    # Vector self-interaction parameters from CompOSE table
    # c3 = (ζ/6) × g_ωN⁴, c4 = (ξ/6) × g_ρN⁴
    zeta = -1.6155896062e-3
    xi = 4.1286242877e-3
    p.c3 = (zeta / 6.0) * (p.g_omega_N**4)
    p.c4 = (xi / 6.0) * (p.g_rho_N**4)


    # Symmetry energy A-function coefficients from CompOSE table
    # Per Steiner 2005 Eq. 13: Lagrangian has g_ρ² f(σ,ω) ρ²
    # Per Fortin 2017: A = g_ρ² × f, where f = Σᵢ aᵢσⁱ + Σⱼ bⱼω^(2j)
    # CompOSE gives the f coefficients (Steiner form), NOT A coefficients
    # The g_ρ² multiplication is done in compute_A()
    
    # a coefficients: a[i] has units fm^(i-1) in CompOSE table
    # Stored as: a_coeffs[i] = a_i × hc^(2-i) so that f has units MeV² with σ in MeV
    p.a_coeffs = np.zeros(7)
    p.a_coeffs[1] = -1.9308602647e-1 * hc            # a₁ [fm⁻¹] → MeV
    p.a_coeffs[2] = 5.6150318121e-1                   # a₂ [1] → dimensionless
    p.a_coeffs[3] = 2.8617603774e-1 / hc              # a₃ [fm] → MeV⁻¹
    p.a_coeffs[4] = 2.7717729776 / (hc**2)            # a₄ [fm²] → MeV⁻²
    p.a_coeffs[5] = 1.2307286924 / (hc**3)            # a₅ [fm³] → MeV⁻³
    p.a_coeffs[6] = 6.1480060734e-1 / (hc**4)         # a₆ [fm⁴] → MeV⁻⁴
    
    # b coefficients: b[j] has units fm^(2j-2) in CompOSE table
    # Stored as: b_coeffs[j] = b_j × hc^(2-2j) so that f has units MeV² with ω in MeV
    p.b_coeffs = np.zeros(4)
    p.b_coeffs[1] = 5.5118461115                      # b₁ [1] → dimensionless
    p.b_coeffs[2] = -1.8007283681 / (hc**2)           # b₂ [fm²] → MeV⁻²
    p.b_coeffs[3] = 4.2610479708e2 / (hc**4)          # b₃ [fm⁴] → MeV⁻⁴

    return p


# =============================================================================
# PARAMETRIZATION FACTORY FUNCTIONS
# =============================================================================

def _nucleonic() -> Parameters:
    """
    SFHo with nucleons only (Steiner et al. 2013).
    
    Use for pure nuclear matter calculations.
    """
    p = _get_base_sfho()
    p.name = "SFHo_Nucleonic"
    return p


def _sfhoy_fortin() -> Parameters:
    """
    SFHoY parametrization from Fortin et al. 2017 (PASA 35, e044).

    Features:
    - SU(6) vector couplings broken by y = 1.5 (Λ, Σ) and 1.875 (Ξ) on the
      ω and φ vertices, ρ left at SU(6) — §2.2 of the reference
    - Scalar couplings fitted to the hyperon potential depths
    - Supports M_max ≈ 2.0 M_sun for cold NS

    Hyperon potential depths at saturation (SNM, n_sat = 0.158 fm⁻³):
    - U_Λ^(N) = -30 MeV
    - U_Σ^(N) = +30 MeV
    - U_Ξ^(N) = -14 MeV

    The factors give the reference's Table 1 back:
    - R_ωΛ = 1.5 × (2/3) = 1.0,  R_φΛ = 1.5 × (-√2/3) = -0.71
    - R_ωΣ = 1.5 × (2/3) = 1.0,  R_φΣ = -0.71
    - R_ωΞ = 1.875 × (1/3) = 0.625, R_φΞ = 1.875 × (-2√2/3) = -1.77
    """
    p = _get_base_sfho()
    p.name = "SFHoY_Fortin"
    p.U_Lambda, p.U_Sigma, p.U_Xi = -30.0, 30.0, -14.0

    # SU(6) broken on ω and φ, per multiplet (Fortin et al. 2018 §2.2).
    p.y_omega_Lambda = p.y_phi_Lambda = 1.5
    p.y_omega_Sigma = p.y_phi_Sigma = 1.5
    p.y_omega_Xi = p.y_phi_Xi = 1.875

    # The scalar couplings are the published fit to the depths above; the
    # vector ones follow from the factors and are not stored.
    p.couplings_map['lambda'] = {'sigma': 0.854315 * p.g_sigma_N}
    for name in ['sigma+', 'sigma0', 'sigma-']:
        p.couplings_map[name] = {'sigma': 0.586611 * p.g_sigma_N}
    for name in ['xi0', 'xi-']:
        p.couplings_map[name] = {'sigma': 0.512754 * p.g_sigma_N}

    return p


def _sfhoy_star_fortin() -> Parameters:
    """
    SFHoY* parametrization from Fortin et al. 2017.

    Hyperon potential depths at saturation (SNM, n_sat = 0.158 fm⁻³):
    - U_Λ^(N) = -30 MeV
    - U_Σ^(N) = +30 MeV
    - U_Ξ^(N) = -14 MeV

    Features:
    - SU(6) vector couplings: every breaking factor is 1, so R_ωΛ = R_ωΣ =
      2/3, R_ωΞ = 1/3
    - Scalar couplings from potential depths:
      - R_σΛ = 0.6142, R_σΣ = 0.3461, R_σΞ = 0.3026
    - Does NOT satisfy 2 M_sun constraint (M_max ≈ 1.75 M_sun)
    """
    p = _get_base_sfho()
    p.name = "SFHoY*_Fortin"
    p.U_Lambda, p.U_Sigma, p.U_Xi = -30.0, 30.0, -14.0

    p.couplings_map['lambda'] = {'sigma': 0.614161 * p.g_sigma_N}
    for name in ['sigma+', 'sigma0', 'sigma-']:
        p.couplings_map[name] = {'sigma': 0.346456 * p.g_sigma_N}
    for name in ['xi0', 'xi-']:
        p.couplings_map[name] = {'sigma': 0.302619 * p.g_sigma_N}

    return p


def _two_family_phi() -> Parameters:
    """
    SFHoYD: SFHo with Hyperons and Deltas - includes phi meson coupling.

    Hyperon potential depths at saturation (SNM, n_sat = 0.158 fm⁻³):
    - U_Λ^(N) = -28 MeV (different from SFHoY* which uses -30)
    - U_Σ^(N) = +30 MeV
    - U_Ξ^(N) = -18 MeV (different from SFHoY* which uses -14)

    Features:
    - SU(6) vector couplings: every breaking factor is 1, so R_ωΛ = R_ωΣ =
      2/3, R_ωΞ = 1/3
    - Scalar couplings from potential depths:
      - R_σΛ = 0.6052, R_σΣ = 0.3461, R_σΞ = 0.3205
    - Delta couplings: R_σ = 1.15, R_ω = 1, R_ρ = 1, R_φ = 0
    - Hyperons couple to phi meson (hidden strangeness)

    This is the SFHoYD parametrization used in Mathematica.
    """
    p = _get_base_sfho()
    p.name = "SFHo_2fam_phi"
    p.U_Lambda, p.U_Sigma, p.U_Xi = -28.0, 30.0, -18.0

    p.couplings_map['lambda'] = {'sigma': 0.605237 * p.g_sigma_N}
    for name in ['sigma+', 'sigma0', 'sigma-']:
        p.couplings_map[name] = {'sigma': 0.346456 * p.g_sigma_N}
    for name in ['xi0', 'xi-']:
        p.couplings_map[name] = {'sigma': 0.320466 * p.g_sigma_N}

    # Deltas: no measured depth in this model, so all four are stored.
    delta_couplings = {
        'sigma': 1.15 * p.g_sigma_N,
        'omega': 1.0 * p.g_omega_N,
        'phi': 0.0,
        'rho': 1.0 * p.g_rho_N,
    }
    for name in ['delta++', 'delta+', 'delta0', 'delta-']:
        p.couplings_map[name] = delta_couplings.copy()

    return p


def _two_family() -> Parameters:
    """
    SFHo with Hyperons and Deltas - NO phi meson coupling (2-family without phi).

    Features:
    - SFHo_2fam_phi scalar couplings for hyperons
    - SU(6) ω and ρ couplings for hyperons
    - g_phi = 0 for ALL hyperons: the three y_phi factors are zero, which is
      the statement that the hidden-strange sector is absent (CLAUDE.md §4 —
      the sector is controlled by its coupling, and there is no flag)
    - Delta couplings: R_σ = 1.15, R_ω = 1, R_ρ = 1, R_φ = 0
    """
    p = _two_family_phi()
    p.name = "SFHo_2fam"
    p.y_phi_Lambda = p.y_phi_Sigma = p.y_phi_Xi = 0.0
    return p


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_params_summary(params: Parameters) -> None:
    """Print a summary of the parametrization."""
    print(f"Parametrization: {params.name}")
    print("=" * 60)
    print("\nNucleon couplings:")
    print(f"  g_σN = {params.g_sigma_N:.5f}")
    print(f"  g_ωN = {params.g_omega_N:.5f}")
    print(f"  g_ρN = {params.g_rho_N:.5f}")
    print(f"  g_φN = {params.g_phi_N:.5f}")
    
    print("\nNon-linear parameters:")
    print(f"  g2 = {params.g2:.4e} MeV")
    print(f"  g3 = {params.g3:.4e}")
    print(f"  c3 = {params.c3:.4e}")
    print(f"  c4 = {params.c4:.4e}")
    
    if params.couplings_map:
        print("\nHyperon/Delta coupling ratios (R = g_MH / g_MN, φ over g_ωN):")
        for particle in params.couplings_map:
            Rs = params.get_coupling(particle, 'sigma') / params.g_sigma_N
            Rw = params.get_coupling(particle, 'omega') / params.g_omega_N
            Rr = params.get_coupling(particle, 'rho') / params.g_rho_N
            Rp = params.get_coupling(particle, 'phi') / params.g_omega_N
            print(f"  {particle:10s}: R_σ={Rs:.3f}, R_ω={Rw:.3f}, R_ρ={Rr:.3f}, R_φ={Rp:.3f}")


#: The published parameter sets, by the name each one carries in its `name`
#: field. The values are BUILDERS, not instances: a `Parameters` holds mutable
#: coupling maps, so a shared instance would be global mutable state
#: (CLAUDE.md section 6). Reach them through `Parameters.named(...)`.
#:
#:   SFHo_Nucleonic   nucleons only, the CompOSE SFHo table; `default()`.
#:   SFHoY_Fortin     + hyperons, SU(6) omega and phi broken by y = 1.5 (Λ, Σ)
#:                    and 1.875 (Ξ), scalar couplings from U_Y (Fortin 2017).
#:   SFHoY*_Fortin    + hyperons, every breaking factor 1, i.e. SU(6) vector
#:                    couplings (same reference).
#:   SFHo_2fam_phi    + hyperons and Deltas, SU(6) vectors, hyperons coupled
#:                    to phi.
#:   SFHo_2fam        as SFHo_2fam_phi with y_phi = 0 in every multiplet, so
#:                    g_phi = 0 for every strange baryon.
PUBLISHED_SETS = {
    'SFHo_Nucleonic': _nucleonic,
    'SFHoY_Fortin': _sfhoy_fortin,
    'SFHoY*_Fortin': _sfhoy_star_fortin,
    'SFHo_2fam_phi': _two_family_phi,
    'SFHo_2fam': _two_family,
}


# =============================================================================
# SELF-TEST
# =============================================================================
if __name__ == "__main__":
    print("SFHo Parameters Module")
    print("=" * 70)
    
    # Test all parametrizations
    for name in PUBLISHED_SETS:
        print(f"\n{'='*70}")
        print_params_summary(Parameters.named(name))
    
    # Verify coupling retrieval
    print("\n" + "=" * 70)
    print("Testing coupling retrieval for SFHo_2fam_phi:")
    print("-" * 50)
    p = Parameters.named('SFHo_2fam_phi')
    test_particles = ['proton', 'neutron', 'lambda', 'sigma+', 'xi-', 'delta++']
    
    print(f"{'Particle':<12} {'σ':>10} {'ω':>10} {'ρ':>10} {'φ':>10}")
    print("-" * 54)
    for part in test_particles:
        gs = p.get_coupling(part, 'sigma')
        gw = p.get_coupling(part, 'omega')
        gr = p.get_coupling(part, 'rho')
        gp = p.get_coupling(part, 'phi')
        print(f"{part:<12} {gs:>10.4f} {gw:>10.4f} {gr:>10.4f} {gp:>10.4f}")