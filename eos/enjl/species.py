"""The degrees of freedom of the extended NJL model, and their quantum numbers.

Eight species in three families: baryons p, n, Lambda (g = 2, B = 1); quarks
u, d, s (g = 6, B = 1/3); leptons e, mu (g = 2, B = 0). That set is fixed by
the model of Xia, Phys. Rev. D 110, 014022 (2024) -- it is not a configurable
composition -- and `SpeciesFlags` below says so rather than offering switches
that do nothing.

Conventions are this repository's (CLAUDE.md section 2), not the PDG's:
STRANGENESS IS S = +1 PER s QUARK, so Lambda and the s quark both carry
S = +1; and C, the charge that enters n_C and Y_C, is the charge of strongly
interacting matter only, leptons excluded. The physical electric charge q_i --
which DOES include the leptons -- is what the paper's beta-equilibrium
condition Eq. (23) and its neutrality condition Eq. (24) are written in, and
it is carried here separately as `CHARGE`.

Two model-specific tables sit beside the quantum numbers, because in this
model a baryon is a three-quark cluster and almost everything is built from
that fact:

  VALENCE        N^q_i, the valence content of each baryon, which appears in
                 the baryon mass Eq. (4), the effective scalar density
                 Eq. (6), the omega source Eq. (9) and the rearrangement term
                 Eq. (17);
  ISOSPIN        tau_i, the eigenvalue the rho couples to.
"""
from dataclasses import dataclass

#: Every species, in the fixed order every sum in this model runs over. The
#: order is part of the arithmetic: it is what makes a sum reproducible.
SPECIES = ("p", "n", "Lambda", "u", "d", "s", "e", "mu")

BARYONS = ("p", "n", "Lambda")
QUARKS = ("u", "d", "s")
LEPTONS = ("e", "mu")

#: Degeneracy: spin x colour for the quarks, spin alone for baryons and
#: leptons.
DEGENERACY = {"p": 2.0, "n": 2.0, "Lambda": 2.0,
              "u": 6.0, "d": 6.0, "s": 6.0,
              "e": 2.0, "mu": 2.0}

#: Baryon number B_i.
BARYON_NUMBER = {"p": 1.0, "n": 1.0, "Lambda": 1.0,
                 "u": 1.0 / 3.0, "d": 1.0 / 3.0, "s": 1.0 / 3.0,
                 "e": 0.0, "mu": 0.0}

#: Physical electric charge q_i, LEPTONS INCLUDED. This is the charge of the
#: paper's Eqs. (23)-(24). The non-leptonic charge C of CLAUDE.md section 2 is
#: the same table with the two leptons dropped, and `thermodynamics.assemble`
#: is where that distinction is made.
CHARGE = {"p": 1.0, "n": 0.0, "Lambda": 0.0,
          "u": 2.0 / 3.0, "d": -1.0 / 3.0, "s": -1.0 / 3.0,
          "e": -1.0, "mu": -1.0}

#: Strangeness, S = +1 per s quark -- the OPPOSITE of the PDG convention.
STRANGENESS = {"p": 0.0, "n": 0.0, "Lambda": 1.0,
               "u": 0.0, "d": 0.0, "s": 1.0,
               "e": 0.0, "mu": 0.0}

#: Isospin eigenvalue tau_i for the rho coupling.
ISOSPIN = {"p": 1.0, "n": -1.0, "Lambda": 0.0,
           "u": 1.0, "d": -1.0, "s": 0.0,
           "e": 0.0, "mu": 0.0}

#: Valence-quark content N^q_i = (N^u, N^d, N^s). A quark is its own single
#: valence quark; a lepton has none.
VALENCE = {"p": (2, 1, 0), "n": (1, 2, 0), "Lambda": (1, 1, 1),
           "u": (1, 0, 0), "d": (0, 1, 0), "s": (0, 0, 1),
           "e": (0, 0, 0), "mu": (0, 0, 0)}

#: N_i = sum_q N^q_i, the source weight of the omega field. Three for every
#: baryon and one for every quark: the omega couples to quark number, so a
#: baryon couples three times as strongly because it contains three quarks.
VALENCE_TOTAL = {sp: float(sum(content)) for sp, content in VALENCE.items()}


def current_masses(par):
    """{flavour: m_q0} [MeV], the bare quark masses of the NJL Lagrangian."""
    return {"u": par.m_u0, "d": par.m_d0, "s": par.m_s0}


def coupling_rescalings(par):
    """{species: f_i}, the vector-coupling rescaling of paper Sec. II.B.

    f_p = f_n = 1 by construction; f_Lambda is fitted to the Lambda potential
    depth U_Lambda(n_sat) = -30 MeV; the three quarks share one value f_q,
    which is a parameter of the study. Leptons do not couple to the vector
    fields, f = 0.
    """
    return {"p": 1.0, "n": 1.0, "Lambda": par.f_Lambda,
            "u": par.f_q, "d": par.f_q, "s": par.f_q,
            "e": 0.0, "mu": 0.0}


def valence_count(name, flavour):
    """N^q_i: valence content of baryon `name` for `flavour`, Eqs. (4), (6)."""
    return VALENCE[name][QUARKS.index(flavour)]


#: Why each flag cannot be moved from the value the model fixes it at, in the
#: words the error message uses.
_WHY_FIXED = {
    "hyperons": (
        True,
        "the model's baryon set is (p, n, Lambda) and the Lambda is in it: "
        "f_Lambda is a fitted parameter and the Lambda enters every sum, the "
        "vector sources and the rearrangement terms. Sigma and Xi are not in "
        "the model at all -- the paper does not carry them -- so this flag is "
        "on and cannot be moved either way"),
    "muons": (
        True,
        "the muon family is a degree of freedom of this model and is "
        "populated in beta equilibrium above mu_e = m_mu, which the shipped "
        "parameters reach between n_B = 0.1 and 0.2 fm^-3"),
    "deltas": (
        False,
        "the Delta(1232) is not among the model's baryon clusters; the paper "
        "carries p, n and Lambda only"),
    "thermal_mesons": (
        False,
        "sigma, omega and rho are auxiliary fields here, eliminated in favour "
        "of g^2/m^2, and the model leaves their bare masses undetermined for "
        "exactly that reason -- there is no meson with a mass to put in a "
        "thermal gas. There is no pion or kaon sector either"),
    "photons": (
        False,
        "a thermal sector, identically zero at T = 0, which is the only "
        "temperature this model has"),
    "thermal_neutrinos": (
        False,
        "a thermal sector, identically zero at T = 0, which is the only "
        "temperature this model has"),
}


@dataclass(frozen=True)
class SpeciesFlags:
    """Which degrees of freedom are active: the ones the model has, and no more.

    The flag names are the ones every model in this repository uses, so a
    caller switching between models writes the same thing. Here every value is
    fixed by the model rather than chosen by the caller, and moving one raises
    naming the reason (`_WHY_FIXED`) rather than being quietly ignored: a
    sector that is off must be off because its flag says so, never because the
    model happened not to look at it.

    Nucleons and the three quark flavours are always present and have no flag,
    the way nucleons have none in a hadronic model.
    """
    hyperons: bool = True
    muons: bool = True
    deltas: bool = False
    thermal_mesons: bool = False
    photons: bool = False
    thermal_neutrinos: bool = False

    def __post_init__(self):
        for flag, (fixed_at, why) in _WHY_FIXED.items():
            if getattr(self, flag) is not fixed_at:
                raise NotImplementedError(
                    f"SpeciesFlags: {flag} is fixed at {fixed_at} in "
                    f"eos.enjl -- {why}")
