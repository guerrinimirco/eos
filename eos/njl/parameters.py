"""Parameter container for the three-flavour NJL model with colour
superconductivity.

Three tiers, and the split is what makes the model usable in an inference run
(CLAUDE.md section 6: MODEL PARAMETERS ARE ARGUMENTS):

  tier 1  fixed by vacuum physics, never sampled -- the RKH set below. These
          are what m_pi, f_pi, m_K and m_eta' bought; re-sampling any one of
          them breaks the vacuum phenomenology the whole model is anchored to,
          and they move only if the entire vacuum fit is redone.
  tier 2  structural choices, declared per run rather than sampled: the
          regularization scheme, the vector-coupling FORM, the lepton content.
          They are fields here because they change the equations, not a number.
  tier 3  the Bayesian vector: eta_D, eta_V (or G_V0/G_S and M_g).

Everything else in the EoS -- mu_3, mu_8, mu_C, the gaps, the masses, the
condensates -- is an internal unknown solved at each (n_B, T), never a
parameter.

    Rehberg, Klevansky, Huefner, Phys. Rev. C 53, 410 (1996)
        [arXiv:hep-ph/9506436]                        -- the RKH set
    Kunkel, Rather et al. [arXiv:2607.11537]          -- the "kunkel" set
    Buballa, Phys. Rept. 407, 205 (2005)              -- the review

Units are MeV powers throughout: Lambda and the masses in MeV, G_S in MeV^-2,
K in MeV^-5. `njl.tex` writes out every equation these enter.
"""
from dataclasses import dataclass

#: The vector-coupling forms `eos.njl.couplings` implements. A structural
#: (tier-2) choice: which one is in use changes the asymptotic sound speed,
#: not merely a number in it.
VECTOR_FORMS = ("constant", "power_law", "gluon_exchange")


@dataclass(frozen=True)
class Parameters:
    """One parameter point of the NJL model.

    Frozen, so it is hashable and safely shared between processes; every
    solver takes it as its first argument and none of them reaches for a
    default on the caller's behalf.
    """
    # --- tier 1: the RKH vacuum fit -------------------------------------
    Lambda: float = 602.3            # three-momentum cutoff [MeV]
    GS_Lambda2: float = 1.835        # G_S Lambda^2, dimensionless
    K_Lambda5: float = 12.36         # K Lambda^5, dimensionless
    m_u: float = 5.5                 # current masses [MeV]
    m_d: float = 5.5
    m_s: float = 140.7

    # --- tier 3: the sampled couplings, as ratios to G_S ----------------
    #: eta_D = G_D/G_S. The Fierz value is 0.75. NOTE that eta_D = 1 does NOT
    #: mean "equally strong channels": the condensation costs are
    #: sum_f (M_f - m_f)^2/(8 G_S) in the scalar channel against
    #: sum_eta Delta_eta^2/(4 G_D) in the diquark one, and the factor 2
    #: between them is channel and Fierz counting. eta_D also has to absorb
    #: the 't Hooft--diquark cross-term, which this model omits (section 2.1
    #: of docs/njl_csc_implementation.md); a paper using it should say so.
    eta_D: float = 0.75
    #: eta_V = G_V/G_S, the constant-vector-coupling variant.
    eta_V: float = 0.0

    # --- tier 2: structural choices -------------------------------------
    #: Which G_V(n_q) form `eos.njl.couplings` evaluates. "constant" is
    #: eta_V G_S; the other two are functions of the state and carry a
    #: rearrangement self-energy (see `couplings.py`).
    vector_form: str = "constant"
    #: Power-law variant G_V = G_V0 (n_ref/n_q)^alpha. alpha = 2/3 is the
    #: conformal point: there, and only there, the vector term's own pressure
    #: is exactly one third of its own energy density at EVERY density, so
    #: c_s^2 -> 1/3 rather than running away to 1.
    alpha: float = 2.0 / 3.0
    #: The reference quark density of the power-law form [fm^-3]; three times
    #: nuclear saturation density, so G_V0 is the coupling at n_B = n_sat.
    n_ref: float = 0.48
    #: Gluon-exchange variant G_V = G_V0 / [1 + 8 k_F^2/(9 M_g^2)], with
    #: G_V0 = G_V0_over_GS * G_S and a non-perturbative gluon mass M_g. Its
    #: effective exponent reaches 2/3 by itself, with no tuning.
    G_V0_over_GS: float = 0.5
    M_g: float = 500.0               # [MeV]
    #: lambda = Lambda_UV/Lambda, the regularization control. lambda = 1 is
    #: conventional sharp-cutoff regularization. Anything else needs the
    #: RG-consistent counterterm, which is NOT implemented -- see
    #: docs/DEFERRED.md -- and raises rather than returning a divergent
    #: answer.
    lambda_UV: float = 1.0

    # ------------------------------------------------------------ derived
    @property
    def G_S(self):
        """Scalar coupling [MeV^-2]."""
        return self.GS_Lambda2 / self.Lambda ** 2

    @property
    def K(self):
        """'t Hooft determinant coupling [MeV^-5]."""
        return self.K_Lambda5 / self.Lambda ** 5

    @property
    def G_D(self):
        """Diquark coupling [MeV^-2]."""
        return self.eta_D * self.G_S

    @property
    def current_masses(self):
        """(m_u, m_d, m_s) [MeV], in the flavour order of `eos.general.pairing`."""
        return (self.m_u, self.m_d, self.m_s)

    @property
    def Lambda_medium(self):
        """The cutoff on the MEDIUM integrals [MeV], lambda times Lambda.

        The Dirac sea keeps the vacuum cutoff whatever lambda is; only the
        medium integral is taken to the larger scale, which is the whole
        content of RG consistency.
        """
        return self.lambda_UV * self.Lambda

    # ------------------------------------------------- published sets
    @classmethod
    def default(cls):
        """The shipped set: RKH, Fierz eta_D = 0.75, no vector coupling.

        Sharp cutoff (lambda = 1). This is the set every verified number in
        docs/njl_csc_implementation.md was produced at, and the one
        `test/baseline` is frozen at.
        """
        return cls()

    @classmethod
    def named(cls, name):
        """One of the published sets; see `PUBLISHED_SETS` for what each is."""
        if name not in PUBLISHED_SETS:
            raise KeyError(f"unknown NJL parameter set {name!r}; published: "
                           f"{sorted(PUBLISHED_SETS)}")
        return cls(**PUBLISHED_SETS[name])


#: The published parameter points.
#:
#:   rkh              the shipped default (see `Parameters.default`).
#:   kunkel           the couplings of Kunkel, Rather et al.
#:                    [arXiv:2607.11537]: a strong diquark channel and a
#:                    substantial vector repulsion. THEIR calculation is
#:                    RG-consistent at lambda ~ 10, and this set is at
#:                    lambda = 1, because the counterterm that makes lambda > 1
#:                    finite is not implemented here (docs/DEFERRED.md). The
#:                    couplings are theirs; the regularization is not, and the
#:                    two are not independent -- RG-consistent gaps run almost
#:                    90% above sharp-cutoff ones. Use it as a strong-coupling
#:                    point, not as a reproduction of that paper.
#:   gluon_exchange   the recommended vector variant: the gluon-exchange form
#:                    of section 9.5 of the specification, whose effective
#:                    exponent reaches the conformal 2/3 without tuning. It
#:                    gives a sound-speed peak near n_B ~ 0.8 fm^-3 followed
#:                    by an approach to 1/3 from above, which is the shape a
#:                    compact-star EoS wants.
PUBLISHED_SETS = {
    "rkh": {},
    "kunkel": dict(eta_D=1.45, eta_V=0.7),
    "gluon_exchange": dict(vector_form="gluon_exchange",
                           G_V0_over_GS=0.5, M_g=500.0),
}
