"""
parametrization.py
====================
DD2 parametrization container and construction routes.

Routes implemented:
    default()             — published DD2 table (Typel et al. 2010,
                            Tables II–III)
    from_microscopic(...) — user-supplied coefficients; dependent a_i, d_i derived
                            from the internal constraints when omitted

The inverse maps that build a Parameters from physical observables live one
layer up, where they can solve saturated matter: `nmp.from_nmp` for the
nuclear-matter parameters.

Every construction is validated in __post_init__ (the M0 ingest gate):
positivity, f_i(1)=1 and d_i=1/sqrt(3 c_i) to INGEST_TOL.

Units: n_sat in fm^-3, masses in MeV, couplings dimensionless.
"""
import math
from dataclasses import dataclass

from eos.general.physics_constants import hc3
from eos.dd2.couplings import (
    rational_f, rational_df, exponential_f, derived_a, derived_d,
    SU6_HYPERON, DD2Y_HYPERON, _POTENTIAL_KEY, scalar_ratio_from_potential,
)

#: Ingest-identity tolerance: the published table obeys the internal
#: constraints to 6 significant digits.
INGEST_TOL = 2.0e-6


@dataclass(frozen=True)
class Parameters:
    n_sat: float                 # fm^-3 (also the coupling reference density)
    m_n: float                   # MeV
    m_p: float                   # MeV
    m_sigma: float               # MeV
    m_omega: float               # MeV
    m_rho: float                 # MeV
    gamma_sigma: float           # Gamma_sigma(n_sat)
    a_sigma: float
    b_sigma: float
    c_sigma: float
    d_sigma: float
    gamma_omega: float           # Gamma_omega(n_sat)
    a_omega: float
    b_omega: float
    c_omega: float
    d_omega: float
    gamma_rho: float             # Gamma_rho(n_sat)
    a_rho: float                 # exponential slope
    m_phi: float = 1019.45       # MeV, hidden-strange vector (inert w/o hyperons)
    #: Hyperon single-particle potentials in SNM at saturation [MeV] (report
    #: §2.4b); the scalar hyperon couplings are inverted from these.
    U_Lambda: float = -30.0
    U_Sigma: float = 30.0
    U_Xi: float = -18.0          # DD2Y (Marques 2017); -14 is SFHoY
    #: Per-hyperon coupling data, hashable tuple of rows
    #: (name, mass [MeV], x_sigma, x_omega, x_rho, x_phi); empty for
    #: nucleon-only params. All x_i are ratios g_iY/g_iN inheriting the nucleon
    #: density dependence f_i(x); x_phi = g_phiY/g_omegaN inherits f_omega, so
    #: Gamma_phiY(n_B) = x_phi * Gamma_omegaN(n_B) (DD2Y: density-dependent φ).
    hyperon_couplings: tuple = ()
    #: Δ-isobar coupling ratios x_iΔ = Γ_iΔ/Γ_iN (; free params,
    #: literature x_Δσ~1.0–1.3, x_Δω~1.0). Δ has S=0, so no φ coupling.
    #: Inert unless SpeciesFlags.deltas is on. Defaults: Δ couples like the
    #: nucleon to σ/ω/ρ (within the literature range; override per model).
    x_Delta_sigma: float = 1.0
    x_Delta_omega: float = 1.0
    x_Delta_rho: float = 1.0
    #: Nucleon-mass convention of the uniform-matter kernel:
    #: "average"  — both nucleons carry (m_n+m_p)/2. Convention of
    #:              the published-table convention, and the one the
    #:              regression gates are stated against.
    #: "physical" — per-species m_n, m_p (HS(DD2)/CompOSE convention;
    #:              needed for <0.1% isovector agreement with the tables).
    nucleon_mass_mode: str = "average"

    def kernel_masses(self):
        """(m_kernel_n, m_kernel_p) per the nucleon_mass_mode convention."""
        if self.nucleon_mass_mode == "average":
            m = self.m_nucleon
            return m, m
        return self.m_n, self.m_p

    def __post_init__(self):
        if self.nucleon_mass_mode not in ("average", "physical"):
            raise ValueError(
                f"nucleon_mass_mode must be 'average' or 'physical', "
                f"got {self.nucleon_mass_mode!r}")
        for name in ("n_sat", "m_n", "m_p", "m_sigma", "m_omega", "m_rho",
                     "gamma_sigma", "gamma_omega", "gamma_rho",
                     "c_sigma", "c_omega"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"Parameters: {name} must be > 0, "
                                 f"got {getattr(self, name)}")
        for meson in ("sigma", "omega"):
            a = getattr(self, f"a_{meson}")
            b = getattr(self, f"b_{meson}")
            c = getattr(self, f"c_{meson}")
            d = getattr(self, f"d_{meson}")
            err_f1 = abs(rational_f(1.0, a, b, c, d) - 1.0)
            if err_f1 > INGEST_TOL:
                raise ValueError(
                    f"Parameters: f_{meson}(1) = 1 violated by {err_f1:.2e} "
                    f"(constraint: a_{meson} is dependent on b, c, d)")
            err_d = abs(d - derived_d(c))
            if err_d > INGEST_TOL:
                raise ValueError(
                    f"Parameters: d_{meson} = 1/sqrt(3 c_{meson}) violated "
                    f"by {err_d:.2e} (constraint: f_{meson}''(0) = 0)")

    # ------------------------------------------------------------- properties
    @property
    def m_nucleon(self):
        """Average nucleon mass used by the uniform-matter kernel (MeV)."""
        return 0.5 * (self.m_n + self.m_p)

    @property
    def hyperon_coupling_map(self):
        """{name: (mass, x_sigma, x_omega, x_rho, x_phi)} from the tuple."""
        return {row[0]: row[1:] for row in self.hyperon_couplings}

    # ------------------------------------------------------------- couplings
    def couplings_at(self, n_B):
        """
        All couplings and their density derivatives at total baryon density
        n_B [fm^-3].

        Returns:
            (G_sigma, G_omega, G_rho, dG_sigma, dG_omega, dG_rho)
            with G_i dimensionless and dG_i = dGamma_i/dn_B in MeV^-3
            (derivative w.r.t. the natural-unit density n_B * hc^3).
        """
        x = n_B / self.n_sat
        nsat_nat = self.n_sat * hc3
        Gs = self.gamma_sigma * rational_f(x, self.a_sigma, self.b_sigma,
                                           self.c_sigma, self.d_sigma)
        Gw = self.gamma_omega * rational_f(x, self.a_omega, self.b_omega,
                                           self.c_omega, self.d_omega)
        Gr = self.gamma_rho * exponential_f(x, self.a_rho)
        dGs = self.gamma_sigma * rational_df(x, self.a_sigma, self.b_sigma,
                                             self.c_sigma, self.d_sigma) / nsat_nat
        dGw = self.gamma_omega * rational_df(x, self.a_omega, self.b_omega,
                                             self.c_omega, self.d_omega) / nsat_nat
        dGr = -self.a_rho * Gr / nsat_nat
        return Gs, Gw, Gr, dGs, dGw, dGr

    # ----------------------------------------------------------- constructors
    @classmethod
    def default(cls):
        """Published DD2 table, transcribed verbatim."""
        return cls(
            n_sat=0.149065,
            m_n=939.56536, m_p=938.27203,
            m_sigma=546.212459, m_omega=783.0, m_rho=763.0,
            gamma_sigma=10.686681, a_sigma=1.357630, b_sigma=0.634442,
            c_sigma=1.005358, d_sigma=0.575810,
            gamma_omega=13.342362, a_omega=1.369718, b_omega=0.496475,
            c_omega=0.817753, d_omega=0.638452,
            gamma_rho=3.626940, a_rho=0.518903,
        )

    @classmethod
    def from_microscopic(cls, *, n_sat, gamma_sigma, b_sigma, c_sigma,
                         gamma_omega, b_omega, c_omega, gamma_rho, a_rho,
                         a_sigma=None, d_sigma=None, a_omega=None, d_omega=None,
                         m_n=939.56536, m_p=938.27203,
                         m_sigma=546.212459, m_omega=783.0, m_rho=763.0):
        """
        User-supplied microscopic coefficients. Omitted a_i/d_i are derived
        from the internal constraints; supplied ones are validated against them.
        Masses default to the DD2 values.
        """
        if c_sigma <= 0.0 or c_omega <= 0.0:
            raise ValueError("Parameters: c_sigma and c_omega must be > 0 "
                             f"(got {c_sigma}, {c_omega})")
        if d_sigma is None:
            d_sigma = float(derived_d(c_sigma))
        if a_sigma is None:
            a_sigma = float(derived_a(b_sigma, c_sigma, d_sigma))
        if d_omega is None:
            d_omega = float(derived_d(c_omega))
        if a_omega is None:
            a_omega = float(derived_a(b_omega, c_omega, d_omega))
        return cls(
            n_sat=n_sat, m_n=m_n, m_p=m_p,
            m_sigma=m_sigma, m_omega=m_omega, m_rho=m_rho,
            gamma_sigma=gamma_sigma, a_sigma=a_sigma, b_sigma=b_sigma,
            c_sigma=c_sigma, d_sigma=d_sigma,
            gamma_omega=gamma_omega, a_omega=a_omega, b_omega=b_omega,
            c_omega=c_omega, d_omega=d_omega,
            gamma_rho=gamma_rho, a_rho=a_rho,
        )

    @classmethod
    def named(cls, name):
        """A published set by name.

        'DD2' is the nucleon table `default()` returns; 'DD2Y' adds the
        hyperon octet. They are different parameterisations, not one set read
        through different species flags, so both are listed here.
        """
        known = {"DD2": cls.default, "DD2Y": cls._dd2y}
        if name not in known:
            raise KeyError(f"unknown DD2 parameter set {name!r}; "
                           f"available: {sorted(known)}")
        return known[name]()

    @classmethod
    def _dd2y(cls):
        """
        DD2Y (Marques et al. 2017; Fortin et al. 2017 Table 1): DD2 nucleon
        sector + hyperon octet with the published R-couplings hardcoded (report
        §2.4). Scalar ratios R_sigma = {Λ:0.62, Σ:0.48, Ξ:0.32}; SU(6) vectors
        R_omega, R_rho, R_phi; DD2Y hyperon masses. The φ coupling is
        density-dependent, inheriting f_omega (B4): Γ_phiY(n_B) = x_phi ·
        Γ_omegaN(n_B). This is the bit-exact DD2Y validation setting.

        For non-DD2Y potentials, use `eos.dd2.nmp.from_hyperon_potentials`.
        """
        from dataclasses import replace
        rows = []
        for name, su6 in SU6_HYPERON.items():
            dd2y = DD2Y_HYPERON[name]
            rows.append((name, dd2y["mass"], dd2y["R_sigma"],
                         su6["x_omega"], su6["x_rho"], su6["phi_over_omegaN"]))
        return replace(cls.default(),
                       hyperon_couplings=tuple(rows))


def _check_dd2_defaults():
    """Standalone ingest check, mirrors the M0 gate."""
    p = Parameters.default()
    for meson in ("sigma", "omega"):
        a, b, c, d = (getattr(p, f"{k}_{meson}") for k in "abcd")
        print(f"{meson}: |f(1)-1| = {abs(rational_f(1.0, a, b, c, d) - 1.0):.2e}, "
              f"|d - 1/sqrt(3c)| = {abs(d - 1.0 / math.sqrt(3.0 * c)):.2e}")


if __name__ == "__main__":
    _check_dd2_defaults()
