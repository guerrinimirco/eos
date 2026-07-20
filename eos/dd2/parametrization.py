"""
parametrization.py
====================
DD2 parametrization container and construction routes (report §3.2).

Routes implemented:
    from_dd2_defaults()   — published DD2 table (Typel et al. 2010, Tables II–III)
    from_microscopic(...) — user-supplied coefficients; dependent a_i, d_i derived
                            from the internal constraints when omitted
    from_nmp(...)         — NMP inversion cascade, milestone M8 (not yet built)

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
#: constraints to 6 significant digits (report §2.1).
INGEST_TOL = 2.0e-6


@dataclass(frozen=True)
class Parametrization:
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
    #: Δ-isobar coupling ratios x_iΔ = Γ_iΔ/Γ_iN (report §2.4; free params,
    #: literature x_Δσ~1.0–1.3, x_Δω~1.0). Δ has S=0, so no φ coupling.
    #: Inert unless SpeciesFlags.deltas is on. Defaults: Δ couples like the
    #: nucleon to σ/ω/ρ (within the literature range; override per model).
    x_Delta_sigma: float = 1.0
    x_Delta_omega: float = 1.0
    x_Delta_rho: float = 1.0
    #: Nucleon-mass convention of the uniform-matter kernel:
    #: "average"  — both nucleons carry (m_n+m_p)/2. Convention of
    #:              dd2_reference_validation.py; the §2.7 golden points and
    #:              the M1/M2 gates are only reproducible in this mode.
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
                raise ValueError(f"Parametrization: {name} must be > 0, "
                                 f"got {getattr(self, name)}")
        for meson in ("sigma", "omega"):
            a = getattr(self, f"a_{meson}")
            b = getattr(self, f"b_{meson}")
            c = getattr(self, f"c_{meson}")
            d = getattr(self, f"d_{meson}")
            err_f1 = abs(rational_f(1.0, a, b, c, d) - 1.0)
            if err_f1 > INGEST_TOL:
                raise ValueError(
                    f"Parametrization: f_{meson}(1) = 1 violated by {err_f1:.2e} "
                    f"(constraint: a_{meson} is dependent on b, c, d)")
            err_d = abs(d - derived_d(c))
            if err_d > INGEST_TOL:
                raise ValueError(
                    f"Parametrization: d_{meson} = 1/sqrt(3 c_{meson}) violated "
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
    def from_dd2_defaults(cls):
        """Published DD2 table, transcribed verbatim (report §2.1)."""
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
            raise ValueError("Parametrization: c_sigma and c_omega must be > 0 "
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
    def from_dd2y_defaults(cls):
        """
        DD2Y (Marques et al. 2017; Fortin et al. 2017 Table 1): DD2 nucleon
        sector + hyperon octet with the published R-couplings hardcoded (report
        §2.4). Scalar ratios R_sigma = {Λ:0.62, Σ:0.48, Ξ:0.32}; SU(6) vectors
        R_omega, R_rho, R_phi; DD2Y hyperon masses. The φ coupling is
        density-dependent, inheriting f_omega (B4): Γ_phiY(n_B) = x_phi ·
        Γ_omegaN(n_B). This is the bit-exact DD2Y validation setting.

        For non-DD2Y potentials, use from_hyperon_potentials().
        """
        from dataclasses import replace
        rows = []
        for name, su6 in SU6_HYPERON.items():
            dd2y = DD2Y_HYPERON[name]
            rows.append((name, dd2y["mass"], dd2y["R_sigma"],
                         su6["x_omega"], su6["x_rho"], su6["phi_over_omegaN"]))
        return replace(cls.from_dd2_defaults(),
                       hyperon_couplings=tuple(rows))

    @classmethod
    def from_hyperon_potentials(cls, U_Lambda=-30.0, U_Sigma=30.0, U_Xi=-18.0):
        """
        DD2-nucleon + hyperon octet with SU(6) vector couplings and scalar
        couplings *inverted* from the hyperon potentials U_Y in SNM at
        saturation (report §2.4b). This is the mechanism that regenerates the
        DD2Y R_sigma table (U_Xi = -18) and the route for non-DD2Y potentials.
        Hyperon masses default to the DD2Y (Marques) values.
        """
        from dataclasses import replace
        from eos.dd2.solver import solve_snm  # local import breaks the cycle

        base = replace(cls.from_dd2_defaults(),
                       U_Lambda=U_Lambda, U_Sigma=U_Sigma, U_Xi=U_Xi)
        sat = solve_snm(base, base.n_sat)
        Gs_sat, Gw_sat, _, _, _, _ = base.couplings_at(base.n_sat)
        U_map = {"U_Lambda": U_Lambda, "U_Sigma": U_Sigma, "U_Xi": U_Xi}

        rows = []
        for name, su6 in SU6_HYPERON.items():
            x_sigma = scalar_ratio_from_potential(
                U_map[_POTENTIAL_KEY[name]], su6["x_omega"], Gs_sat, Gw_sat,
                sat.sigma, sat.omega0, sat.Sigma_R)
            rows.append((name, DD2Y_HYPERON[name]["mass"], x_sigma,
                         su6["x_omega"], su6["x_rho"], su6["phi_over_omegaN"]))
        return replace(base, hyperon_couplings=tuple(rows))

    @classmethod
    def from_delta_potential(cls, U_Delta=-50.0, x_wD=1.0, x_rD=1.0, base=None):
        """
        Δ-isobar couplings from the Δ single-particle potential in SNM at
        saturation (report v11 §2.4). There is no canonical DD2Δ coupling
        table, so the default is universal coupling (x_Δσ = x_Δω = x_Δρ = 1);
        this constructor instead fixes x_Δσ by inverting

            U_Δ = -x_Δσ Γ_σN σ̄ + x_Δω Γ_ωN ω0 + Σ^R      (all at n_sat)

        for a chosen Δ potential (literature U_Δ ∈ [-100, -50] MeV, default -50)
        and vector ratios x_wD, x_rD. base: an existing Parametrization to attach
        the Δ sector to (e.g. a DD2Y octet); defaults to nucleonic DD2.
        """
        from dataclasses import replace
        from eos.dd2.solver import solve_snm  # local import breaks the cycle

        if not (-100.0 <= U_Delta <= -50.0):
            raise ValueError(
                f"U_Delta = {U_Delta} MeV outside the literature range "
                f"[-100, -50]; pass an explicit value in range or widen it")
        base = base or cls.from_dd2_defaults()
        sat = solve_snm(base, base.n_sat)
        Gs_sat, Gw_sat, _, _, _, _ = base.couplings_at(base.n_sat)
        x_Delta_sigma = scalar_ratio_from_potential(
            U_Delta, x_wD, Gs_sat, Gw_sat, sat.sigma, sat.omega0, sat.Sigma_R)
        return replace(base, x_Delta_sigma=x_Delta_sigma,
                       x_Delta_omega=x_wD, x_Delta_rho=x_rD)

    @classmethod
    def from_nmp(cls, nmp, m_sigma=546.212459, return_status=False):
        """
        Invert nuclear-matter parameters to DD2 nucleon couplings (report §2.5
        cascade). nmp: dict with {n_sat, E_sat, m_eff_ratio, K_sat, Q_sat,
        E_sym, L_sym}. Returns the Parametrization, or (Parametrization,
        InversionStatus) if return_status.

        Hyperon/Δ sectors attach on top (same SU(6)+U_Y recipe as
        from_dd2y_defaults / the x_Delta_* fields) once the nucleon sector is
        set; not folded in here.
        """
        from eos.dd2.nmp_inverter import invert_nmp
        par, status = invert_nmp(nmp, m_sigma=m_sigma)
        return (par, status) if return_status else par


def _check_dd2_defaults():
    """Standalone ingest check, mirrors the M0 gate."""
    p = Parametrization.from_dd2_defaults()
    for meson in ("sigma", "omega"):
        a, b, c, d = (getattr(p, f"{k}_{meson}") for k in "abcd")
        print(f"{meson}: |f(1)-1| = {abs(rational_f(1.0, a, b, c, d) - 1.0):.2e}, "
              f"|d - 1/sqrt(3c)| = {abs(d - 1.0 / math.sqrt(3.0 * c)):.2e}")


if __name__ == "__main__":
    _check_dd2_defaults()
