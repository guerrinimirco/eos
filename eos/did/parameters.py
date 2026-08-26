"""The DID parameters: the numbers that pin down the coupling functionals.

`couplings.py` holds the FORM, this holds the numbers, and the evaluation
lives on the parameter object (CLAUDE.md section 5). Every number here is an
argument -- `Parameters` is passed into every entry point and nothing in this
package reads a module-level constant -- so a Bayesian run varies couplings
across millions of calls without editing a source file (section 6).

THE PUBLISHED SET is the maximum-likelihood estimate of Frohaug, Maslov,
Dexheimer et al., arXiv:2511.15646, Table II, transcribed digit for digit,
with the saturation density from their Table III and the meson masses from
their Table I. Baryon masses are not stored: they are the PDG values the paper
uses and come from `eos.general.particles`, the repository's single table
(section 7).

DID and DIDY are ONE parameter set. The hyperon couplings in Table II were
fitted together with the nucleonic ones, and "DIDY" names the same numbers
with the hyperon octet switched on -- which is a `SpeciesFlags` choice, not a
parameter set. `named("DIDY")` therefore returns the same object as
`default()`, and says so.

Two conventions to keep straight, both stated once here and used everywhere:

  tau_3 = 2 I_3, normalised to +/-1 for the nucleons, so the Sigma triplet
  carries +/-2 and the Delta quartet +/-3. This is the paper's normalisation
  and it is the one the isospin asymmetry beta = sum_i tau_3i n_i / n_B and
  the rho coupling g_rho i tau_3i rho are both written in. It is NOT
  `Particle.t3` from `eos.general.particles`, which is the DD2 rho-coupling
  convention where the Sigma factor of two is carried by the coupling ratio
  x_rho = 2 instead; DID has independent g_rho Y and no such ratio, so it
  reads tau_3 off the isospin projection directly (`tau3` below).

  S = +1 per s quark (CLAUDE.md section 2), so Lambda has S = +1 and Xi has
  S = +2 -- the opposite of the paper's PDG sign. It cancels out of every mode
  with mu_S = 0 and is handled by the shared basis maps everywhere else.

Units: n_0 in fm^-3, masses in MeV, couplings dimensionless.
"""
from dataclasses import dataclass, replace
from typing import Mapping

from eos.did.couplings import (
    ALPHA_IDEAL, TAN_THETA_IDEAL, coupling, g8_from_aggregate,
    su3_vector_ratios,
)

#: The multiplets a coupling is declared for. Every active baryon belongs to
#: exactly one, and the Delta quartet inherits the nucleon's numbers scaled by
#: its ratios (the extension of section "Delta isobars" in did.tex).
MULTIPLETS = ("N", "Lambda", "Sigma", "Xi", "Delta")

#: species name -> multiplet.
MULTIPLET_OF = {
    "p": "N", "n": "N",
    "Lambda": "Lambda",
    "Sigma+": "Sigma", "Sigma0": "Sigma", "Sigma-": "Sigma",
    "Xi0": "Xi", "Xi-": "Xi",
    "Delta++": "Delta", "Delta+": "Delta", "Delta0": "Delta",
    "Delta-": "Delta",
}


def tau3(particle):
    """tau_3 = 2 I_3 in the paper's normalisation (+/-1 for the nucleons).

    See the module docstring for why this is not `Particle.t3`.
    """
    return 2.0 * particle.isospin_3


@dataclass(frozen=True)
class Parameters:
    """One DID parameterisation.

    The sigma, omega/phi and rho sectors each carry a shape (a, b, c, d) and,
    per multiplet, the two branch strengths g^{S(0)} (fitted in isospin-
    symmetric matter) and g^{N(0)} (in neutron matter). The paper fits the two
    nucleon branches independently and ties the hyperon branches to them by
    the nucleon ratio -- "the g^{N(0)}_{sigma Y} are in the same proportions to
    g^{N(0)}_{sigma N} as their ISM counterparts" -- which is what
    `_branch_pair` below implements, so a hyperon carries one fitted number
    per meson rather than two.

    The omega and phi couplings are not stored: they are DERIVED from the
    aggregated strength g~_omegaN and the SU(3) ratio z through
    `couplings.g8_from_aggregate`, because that combination is what the
    Bayesian analysis varies (Eq. 52). Storing g_omegaN and g_phiN separately
    would make the fitted quantity a function of two stored ones and let them
    drift apart.
    """
    #: Saturation density [fm^-3]. Also the reference density x = n_B/n_0 of
    #: every coupling, and calibrated so that P(n_0) = 0 in ISM at T = 0.
    n_0: float

    # --- meson masses [MeV] (paper Table I; DD2Y values) -------------------
    m_sigma: float
    m_omega: float
    m_phi: float
    m_rho: float

    # --- sigma sector ------------------------------------------------------
    g_sigma_N_S: float
    g_sigma_N_N: float
    a_sigma: float
    b_sigma: float               # irrelevant while c_sigma is infinite
    c_sigma: float               # inf: no high-density flattening of sigma
    d_sigma: float               # likewise irrelevant
    g_sigma_Lambda_S: float
    g_sigma_Sigma_S: float
    g_sigma_Xi_S: float

    # --- omega / phi sector (SU(3), one octet coupling per branch) ---------
    g_tilde_omega_N_S: float     # the aggregated omega-phi strength, Eq. 52
    g_tilde_omega_N_N: float
    z: float                     # g_1/g_8
    a_omega: float
    b_omega: float
    c_omega: float
    d_omega: float

    # --- rho sector --------------------------------------------------------
    g_rho_N_S: float
    g_rho_N_N: float
    a_rho: float
    b_rho: float
    c_rho: float
    d_rho: float
    g_rho_Sigma_S: float
    g_rho_Xi_S: float            # g_rho Lambda = 0 identically (I = 0)

    #: SU(3) mixing, fixed rather than fitted: ideal omega-phi mixing and
    #: alpha = F/(D+F) = 1. Carried as parameters so a study may vary them.
    alpha: float = ALPHA_IDEAL
    tan_theta: float = TAN_THETA_IDEAL

    #: Delta(1232) coupling ratios x_iDelta = g_iDelta/g_iN, multiplying BOTH
    #: nucleon branches, so the Delta inherits the density AND isospin
    #: dependence of the nucleon vertex. Not in arXiv:2511.15646 (which has no
    #: Deltas); 1.0 is universal coupling, and `nmp.delta_ratios_from_potential`
    #: inverts x_Delta_sigma from a chosen U_Delta instead.
    x_Delta_sigma: float = 1.0
    x_Delta_omega: float = 1.0
    x_Delta_rho: float = 1.0

    def __post_init__(self):
        for name in ("n_0", "m_sigma", "m_omega", "m_phi", "m_rho"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"Parameters: {name} must be > 0, got "
                                 f"{getattr(self, name)}")
        if self.d_sigma == 0.0 or self.d_omega == 0.0 or self.d_rho == 0.0:
            raise ValueError("Parameters: a transition width d_M must be "
                             "non-zero (use c_M = inf to disable the "
                             "high-density plateau, as the sigma does)")

    # ------------------------------------------------------------ branches
    def _branch_pair(self, g_S, nucleon_S, nucleon_N):
        """(g^{S(0)}, g^{N(0)}) of a hyperon vertex from its fitted ISM value.

        The neutron-matter branch is tied to the symmetric one by the nucleon
        ratio of the same meson (paper, Section IV.1). A vertex whose ISM
        value is zero stays zero in both branches, which is what g_rho Lambda
        is.
        """
        if nucleon_S == 0.0:
            return g_S, g_S
        return g_S, g_S * nucleon_N / nucleon_S

    @property
    def g8(self):
        """(g_8^S, g_8^N): the SU(3) octet vector coupling of each branch."""
        return tuple(
            g8_from_aggregate(g_tilde, self.z, self.m_omega, self.m_phi,
                              self.alpha, self.tan_theta)
            for g_tilde in (self.g_tilde_omega_N_S, self.g_tilde_omega_N_N))

    def strengths(self) -> Mapping:
        """{(meson, multiplet): (g^{S(0)}, g^{N(0)})}, every vertex the model has.

        This is the whole parameter content in the shape the thermodynamics
        wants it: two numbers per vertex, to be blended by beta and scaled by
        the meson's shape function.
        """
        g8_S, g8_N = self.g8
        ratios = su3_vector_ratios(self.z, self.alpha, self.tan_theta)

        out = {
            ("sigma", "N"): (self.g_sigma_N_S, self.g_sigma_N_N),
            ("rho", "N"): (self.g_rho_N_S, self.g_rho_N_N),
        }
        for multiplet, g_S in (("Lambda", self.g_sigma_Lambda_S),
                               ("Sigma", self.g_sigma_Sigma_S),
                               ("Xi", self.g_sigma_Xi_S)):
            out[("sigma", multiplet)] = self._branch_pair(
                g_S, self.g_sigma_N_S, self.g_sigma_N_N)
        for multiplet, g_S in (("Lambda", 0.0),
                               ("Sigma", self.g_rho_Sigma_S),
                               ("Xi", self.g_rho_Xi_S)):
            out[("rho", multiplet)] = self._branch_pair(
                g_S, self.g_rho_N_S, self.g_rho_N_N)
        for multiplet, (r_omega, r_phi) in ratios.items():
            out[("omega", multiplet)] = (g8_S * r_omega, g8_N * r_omega)
            out[("phi", multiplet)] = (g8_S * r_phi, g8_N * r_phi)

        # The Delta quartet: the nucleon vertex times a ratio, both branches.
        for meson, ratio in (("sigma", self.x_Delta_sigma),
                             ("omega", self.x_Delta_omega),
                             ("rho", self.x_Delta_rho)):
            g_S, g_N = out[(meson, "N")]
            out[(meson, "Delta")] = (ratio * g_S, ratio * g_N)
        out[("phi", "Delta")] = (0.0, 0.0)      # S = 0, no hidden-strange vertex
        return out

    def shapes(self) -> Mapping:
        """{meson: (a, b, c, d)}: the density shape of each coupling.

        The phi shares the omega's shape, as it must: both come from the same
        octet coupling g_8 and the fit varies one aggregated strength for the
        pair.
        """
        return {
            "sigma": (self.a_sigma, self.b_sigma, self.c_sigma, self.d_sigma),
            "omega": (self.a_omega, self.b_omega, self.c_omega, self.d_omega),
            "phi": (self.a_omega, self.b_omega, self.c_omega, self.d_omega),
            "rho": (self.a_rho, self.b_rho, self.c_rho, self.d_rho),
        }

    # ----------------------------------------------------------- couplings
    def couplings_at(self, n_B, beta):
        """Every coupling and both its derivatives at the state (n_B, beta).

        Returns {(meson, multiplet): (g, dg/dn_B, dg/dbeta)} with dg/dn_B in
        fm^3 (per fm^-3, so that Sigma^r comes out in MeV directly) and
        dg/dbeta dimensionless.

        Both rearrangement self-energies are built from this one call, which
        is why it returns the derivatives rather than offering them
        separately: computing g without them is never what the model needs.
        """
        x = n_B / self.n_0
        shapes = self.shapes()
        out = {}
        for (meson, multiplet), (g_S, g_N) in self.strengths().items():
            g, dg_dx, dg_dbeta = coupling(g_S, g_N, x, beta, *shapes[meson])
            out[(meson, multiplet)] = (g, dg_dx / self.n_0, dg_dbeta)
        return out

    # -------------------------------------------------------- named sets
    @classmethod
    def default(cls):
        """The DID maximum-likelihood set of arXiv:2511.15646, Table II.

        The transition-zone parameters of the vectors (c = 3.5, d = 1.8, in
        units of n_0) and the plateaus b_omega = 0.80, b_rho = 0.40 were fixed
        a priori rather than sampled, and c_sigma = infinity likewise; the
        remaining 15 numbers are the fit. n_0 is from Table III, the value at
        which this set gives P = 0 in symmetric matter at T = 0.

        A NEW set is `dataclasses.replace(Parameters.default(), a_sigma=...)`.
        Twenty-nine of the thirty-four fields carry no default, so bare
        field-by-field construction means supplying all twenty-nine: that is
        deliberate for a functional whose couplings are a joint fit, where a
        partially specified set is a silently wrong one rather than a
        convenient one. `with_deltas` is the constructor for the Delta
        extension. FROM nuclear-matter parameters there is no route:
        `nmp.invert_nmp` raises, because DID's couplings are the maximum-
        likelihood point of a Bayesian analysis over 18 observables and the
        model carries two inequivalent symmetry energies, so the list to
        impose is itself undetermined. `nmp.compute_nmp` is the forward
        direction.
        """
        return cls(
            n_0=0.15880045,
            m_sigma=550.0, m_omega=783.0, m_phi=1020.0, m_rho=763.0,
            g_sigma_N_S=8.94873669, g_sigma_N_N=8.89241948,
            a_sigma=0.16394393, b_sigma=0.0, c_sigma=float("inf"), d_sigma=1.8,
            g_sigma_Lambda_S=7.51077621,
            g_sigma_Sigma_S=6.26418057,
            g_sigma_Xi_S=6.53781517,
            g_tilde_omega_N_S=10.82857726, g_tilde_omega_N_N=11.00228164,
            z=0.07720445,
            a_omega=0.15313180, b_omega=0.80, c_omega=3.5, d_omega=1.8,
            g_rho_N_S=3.23020263, g_rho_N_N=2.59340047,
            a_rho=0.39223762, b_rho=0.40, c_rho=3.5, d_rho=1.8,
            g_rho_Sigma_S=0.00545444, g_rho_Xi_S=1.11415631,
        )

    @classmethod
    def named(cls, name):
        """A published set by name.

        'DID' and 'DIDY' are the same numbers: the hyperon couplings were
        fitted with the rest, and what distinguishes DIDY is
        `SpeciesFlags(hyperons=True)`, not a different parameterisation.
        """
        known = {"DID": cls.default, "DIDY": cls.default}
        if name not in known:
            raise KeyError(f"unknown DID parameter set {name!r}; "
                           f"available: {sorted(known)}")
        return known[name]()

    def with_deltas(self, x_sigma=1.0, x_omega=1.0, x_rho=1.0):
        """A copy carrying Delta coupling ratios (the extension, see did.tex)."""
        return replace(self, x_Delta_sigma=x_sigma, x_Delta_omega=x_omega,
                       x_Delta_rho=x_rho)
