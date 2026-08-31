"""Parameter container for the chiral colour-dielectric model.

Three tiers, and the split is what makes the model usable in an inference run
(CLAUDE.md section 6: MODEL PARAMETERS ARE ARGUMENTS):

  tier 1  fixed by VACUUM DATA and never sampled: f_pi, m_pi, f_K, m_K, the
          current masses, m_zeta, m_phi, m_omega. Everything in `derived`
          below follows from them by closed-form algebra -- the Mexican-hat
          couplings, the explicit-breaking terms, the dilaton scale -- so the
          scalar sector has no free normalisation left once they are chosen.
  tier 2  structural choices, declared per run rather than sampled: the
          dielectric exponent p (locked at 1) and q, which decides whether the
          diquark coupling is dressed by the dielectric (G_D -> G_D/chi^q).
          They are fields here because they change the equations.
  tier 3  the Bayesian vector: B_g^(1/4), g_q, g_s, m_sigma, gbar_omega, n_c,
          and at L3 G_D and Lambda.

Everything else in the EoS -- phi, sigma, zeta, omega_0, the gaps, mu_3, mu_8,
mu_C -- is an internal unknown solved at each (mu_B, T), never a parameter.

The levels the specification names, which are what the flags and a zero
coupling select rather than separate models:

    L0  quarks in the dielectric, no vector, no pairing
    L1  + the density-dependent vector repulsion  (gbar_omega != 0)
    L2  the same at finite temperature
    L3  + colour superconductivity                (SpeciesFlags(csc=True))

    docs/ccdm_implementation.md            -- the implementation specification
    Drago, Fiolhais, Tambini [hep-ph/9503462]
    Ghosh, Phatak, Phys. Rev. C 52, 2195 (1995) [nucl-th/9509017]

Units are MeV powers throughout: masses, potentials and the dilaton in MeV,
B_g in MeV^4, G_D in MeV^-2, n_c in fm^-3 (a density a reader holds in fm).
`ccdm.tex` writes out every equation these enter.
"""
from dataclasses import dataclass
from functools import lru_cache
import math

_SQRT2 = math.sqrt(2.0)


@dataclass(frozen=True)
class Derived:
    """The vacuum-fixed constants of the scalar and glue potentials.

    Not inputs and not sampled: closed-form consequences of tier 1 and of
    (m_sigma, B_g). `Parameters.derived` builds one, and
    `eos.ccdm.verify.run_full_check` asserts what section 9.1 of the
    specification asserts -- V(sigma_0, zeta_0) = 0, U(phi_0) = 0, and that
    the curvatures of V return m_sigma and m_zeta.

    V_ZETA2 IS NEGATIVE at the baseline m_zeta = 980 MeV: the strange quartic
    is convex, explicit breaking dominating, so the strange sector does not
    break chirally on its own in this truncation. The sign flips between
    m_zeta = 1100 and 1150 MeV. Never assume it is positive, and never write
    v_zeta as a square root.
    """
    sigma_0: float          # [MeV]  the light condensate in vacuum, = f_pi
    zeta_0: float           # [MeV]  the strange condensate in vacuum
    eps_sigma: float        # [MeV^3] explicit chiral breaking, light
    eps_zeta: float         # [MeV^3] explicit chiral breaking, strange
    lam: float              # quartic coupling, light
    v2: float               # [MeV^2] Mexican-hat radius squared, light
    lam_zeta: float         # quartic coupling, strange
    v_zeta2: float          # [MeV^2] Mexican-hat radius squared, strange
    C_0: float              # [MeV^4] the constant that puts V(sigma_0, zeta_0) = 0
    phi_0: float            # [MeV]  the vacuum dilaton, 4 sqrt(B_g)/m_phi
    B_g: float              # [MeV^4] the glue bag scale


@dataclass(frozen=True)
class Parameters:
    """One parameter point of the colour-dielectric model.

    Frozen, so it is hashable and safely shared between processes; every
    solver takes it as its first argument and none of them reaches for a
    default on the caller's behalf.
    """
    # --- tier 1: fixed by vacuum data -----------------------------------
    f_pi: float = 93.0               # [MeV]
    m_pi: float = 138.0
    f_K: float = 113.0
    m_K: float = 496.0
    m_u: float = 5.0                 # current masses [MeV]
    m_d: float = 5.0
    m_s: float = 95.0
    m_zeta: float = 980.0            # the strange scalar
    m_phi: float = 1600.0            # the scalar glueball (lattice)
    m_omega: float = 783.0

    # --- tier 3: the sampled vector -------------------------------------
    #: B_g^(1/4) [MeV]: the glue bag scale, which sets phi_0 and the
    #: deconfinement onset. Prior support 120-250 MeV.
    B_g_quarter: float = 150.0
    #: The light-quark coupling, M*_(u,d) = (g_q sigma + m_(u,d))/chi. Pinned
    #: at 3.0 by the specification's section 10 table, which quotes
    #: M*_(u,d) = 826 MeV at phi_bar = 0.90 and 1531 MeV at 0.95 in the
    #: confined branch; both invert to g_q = 3.00. Prior support 3-6.
    g_q: float = 3.0
    #: The strange coupling, M*_s = (g_s zeta + m_s)/chi. NOT pinned by the
    #: specification -- 3.0 is the flavour-symmetric choice g_s = g_q, and the
    #: prior support is 3-8. It is a calibration knob, said so here rather
    #: than presented as a measured value.
    g_s: float = 3.0
    #: The light scalar mass [MeV], which fixes lambda and v. Prior 450-700.
    m_sigma: float = 550.0
    #: The vector coupling at vanishing density, g_omega(0). NOT pinned by the
    #: specification either; 4.0 is mid-prior (support 0-12). Zero switches
    #: the vector sector off entirely, which is the L1 -> L0 reduction.
    gbar_omega: float = 4.0
    #: The density scale of the vector coupling's decay [fm^-3]. Prior 0.3-3.
    n_c: float = 1.0
    #: The diquark coupling [MeV^-2], used only when SpeciesFlags(csc=True).
    #: Calibrated here rather than quoted: at this value the gap sits inside
    #: the 20-150 MeV window the specification asks for at mu_q ~ 450 MeV --
    #: 30.0 MeV in the 2SC channel and 119.4 MeV in the CFL one at
    #: mu_B = 1450 MeV, mu_C = -30 MeV, T = 0. Below ~4.5e-6 the 2SC gap
    #: equation has no root but the trivial one at this potential, so the
    #: window has a floor as well as a ceiling.
    G_D: float = 5.0e-6
    #: The pairing cutoff [MeV]. It applies to the PAIRING INTEGRAL ONLY --
    #: the medium integrals of this model are unregularised and terminate at
    #: their own Fermi momenta -- and it implies the validity ceiling
    #: `mu_ceiling` below. Nearly degenerate with G_D over 550-800 MeV.
    Lambda: float = 600.0

    # --- tier 2: structural choices -------------------------------------
    #: The dielectric exponent, chi = (1 - phi_bar^4)^p. LOCKED at 1: p and
    #: the bracket are meaningful only as a pair, and squaring the bracket
    #: silently doubles the confining-end exponent.
    p: int = 1
    #: Whether the diquark coupling is dressed by the dielectric,
    #: G_D -> G_D/chi^q. q = 1 is the exponent a gluon-exchange origin gives
    #: and the largest that leaves the pairing channel confined (the criterion
    #: is q <= p); q = 0 leaves G_D bare. A declared discrete choice at
    #: calibration, not a sampled parameter.
    q: int = 0

    def __post_init__(self):
        if self.p != 1:
            raise NotImplementedError(
                f"eos.ccdm locks the dielectric exponent at p = 1; got "
                f"p = {self.p}. The specification fixes it there because chi "
                f"and p are meaningful only as the pair chi^p, and the "
                f"derived constants and the q <= p criterion assume it")
        if self.q not in (0, 1):
            raise ValueError(
                f"q must be 0 or 1 (bare or gluon-exchange dielectric "
                f"dressing of G_D); got {self.q}. q > p = 1 would deconfine "
                f"the pairing channel -- see section 10 of "
                f"docs/ccdm_implementation.md")

    # ------------------------------------------------------------ derived
    @property
    def B_g(self):
        """The glue bag scale [MeV^4]."""
        return self.B_g_quarter ** 4

    @property
    def current_masses(self):
        """(m_u, m_d, m_s) [MeV], in the flavour order of `eos.general.pairing`."""
        return (self.m_u, self.m_d, self.m_s)

    @property
    def derived(self):
        """The vacuum-fixed constants of section 8, in closed form.

            sigma_0    = f_pi
            zeta_0     = sqrt2 f_K - f_pi/sqrt2
            eps_sigma  = f_pi m_pi^2
            eps_zeta   = sqrt2 f_K m_K^2 - (f_pi/sqrt2) m_pi^2
            lambda     = (m_sigma^2 - m_pi^2)/(2 f_pi^2)
            v^2        = f_pi^2 - m_pi^2/lambda
            lambda_z   = (m_zeta^2 - eps_zeta/zeta_0)/(2 zeta_0^2)
            v_zeta^2   = zeta_0^2 - eps_zeta/(lambda_z zeta_0)
            phi_0      = 4 sqrt(B_g)/m_phi
            C_0        from V(sigma_0, zeta_0) = 0

        At the shipped point these come out as zeta_0 = 94.045 MeV,
        lambda = 16.387, v = 86.527 MeV, lambda_zeta = 31.414,
        v_zeta^2 = -4039.3 MeV^2, C_0 = 2.4352e9 MeV^4, phi_0 = 56.25 MeV --
        the specification's section 8 numbers to every digit it quotes.

        MEMOIZED: this is a pure function of a frozen, hashable `Parameters`,
        and the residual reads it several times per evaluation -- 137.8 times
        per solved point unpaired, 2665 paired. `Derived` is itself frozen and
        holds only floats, so the shared object cannot be written through.
        """
        return _derived(self)

    @property
    def _derived_uncached(self):
        """The closed forms themselves; `derived` is the memoized entry."""
        sigma_0 = self.f_pi
        zeta_0 = _SQRT2 * self.f_K - self.f_pi / _SQRT2
        eps_sigma = self.f_pi * self.m_pi ** 2
        eps_zeta = (_SQRT2 * self.f_K * self.m_K ** 2
                    - (self.f_pi / _SQRT2) * self.m_pi ** 2)
        lam = (self.m_sigma ** 2 - self.m_pi ** 2) / (2.0 * self.f_pi ** 2)
        v2 = self.f_pi ** 2 - self.m_pi ** 2 / lam
        lam_zeta = ((self.m_zeta ** 2 - eps_zeta / zeta_0)
                    / (2.0 * zeta_0 ** 2))
        v_zeta2 = zeta_0 ** 2 - eps_zeta / (lam_zeta * zeta_0)

        # C_0 is whatever puts the physical vacuum at zero, so that Omega
        # needs no vacuum subtraction anywhere downstream.
        bare = (0.25 * lam * (sigma_0 ** 2 - v2) ** 2
                + 0.25 * lam_zeta * (zeta_0 ** 2 - v_zeta2) ** 2
                - eps_sigma * sigma_0 - eps_zeta * zeta_0)
        return Derived(sigma_0=sigma_0, zeta_0=zeta_0, eps_sigma=eps_sigma,
                       eps_zeta=eps_zeta, lam=lam, v2=v2, lam_zeta=lam_zeta,
                       v_zeta2=v_zeta2, C_0=-bare,
                       phi_0=4.0 * math.sqrt(self.B_g) / self.m_phi,
                       B_g=self.B_g)

    @property
    def mu_ceiling(self):
        """sqrt(Lambda^2 + m_s^2) [MeV]: where the PAIRING sector stops being
        trustworthy.

        The pairing integral is cut at Lambda while the medium integrals are
        not, so above this potential the paired Fermi surface has left the
        region the cutoff describes. Declared rather than exceeded
        (section 6.5 of the specification); `eos.ccdm.solver` reports a point
        past it rather than refusing it, because a sampler must be able to
        score the point.
        """
        return math.sqrt(self.Lambda ** 2 + self.m_s ** 2)

    # ------------------------------------------------- published sets
    @classmethod
    def default(cls):
        """The shipped set: the specification's baseline.

        B_g^(1/4) = 150 MeV and m_sigma = 550 MeV, which together give the
        derived bag constant B_eff = B_g + B_chi = (239.7 MeV)^4 =
        429.4 MeV/fm^3 -- the specification's section 4.4 gate. g_q = 3.0 is
        pinned by its section 10 table; g_s, gbar_omega and n_c are
        calibration knobs at documented mid-prior values, and `eos.ccdm` says
        so rather than dressing them as measurements.
        """
        return cls()

    @classmethod
    def named(cls, name):
        """One of the published sets; see `PUBLISHED_SETS` for what each is."""
        if name not in PUBLISHED_SETS:
            raise KeyError(f"unknown CCDM parameter set {name!r}; published: "
                           f"{sorted(PUBLISHED_SETS)}")
        return cls(**PUBLISHED_SETS[name])


@lru_cache(maxsize=32)
def _derived(par):
    """`Parameters.derived`, memoized on the frozen parameter object.

    The bound is what keeps an inference run -- which varies `par` every call
    and therefore misses every time -- from growing a cache it never reads.
    """
    return par._derived_uncached


#: The published parameter points.
#:
#:   baseline   the shipped default (see `Parameters.default`).
#:   novector   the same with gbar_omega = 0: the L1 -> L0 reduction, which
#:              the verify suite uses to show that switching the vector
#:              coupling off returns the model without one, mode by mode.
#:   dressed    the same with q = 1, the gluon-exchange dielectric dressing of
#:              the diquark coupling, G_D -> G_D/chi. It strengthens pairing
#:              where the dielectric is still partly opaque, which is the
#:              mildest form of the de Carvalho mechanism this model admits
#:              (section 10 of the specification).
#:   stiff      a heavier glue scale, B_g^(1/4) = 190 MeV, which pushes the
#:              deconfinement onset up. Carried because the branch-enumeration
#:              check wants a second onset density to compare against.
PUBLISHED_SETS = {
    "baseline": {},
    "novector": dict(gbar_omega=0.0),
    "dressed": dict(q=1),
    "stiff": dict(B_g_quarter=190.0),
}
