"""The degrees of freedom of the colour-dielectric model, and the two things
it has to ENUMERATE rather than solve.

Nine colour-flavour quark modes, three leptons, and the thermal sectors that
carry no conserved charge. The quark quantum numbers are NOT re-declared here:
they are `eos.general.pairing`'s, imported, because the pairing sector indexes
its 9x9 matrices by the same modes in the same order and two copies of one
table are two chances to disagree (CLAUDE.md section 7). Neither are the
pairing PATTERNS: which gaps a named pattern makes free is a property of the
gap matrix, shared with `eos.njl`.

Conventions are this repository's: STRANGENESS IS S = +1 PER s QUARK, the
opposite of the PDG sign, and C is the charge of strongly-interacting matter
only -- the leptons are excluded from it and enter through the separate
condition of total electric neutrality.

Two enumerations, not one
-------------------------
`eos.njl` enumerates pairing patterns. This model enumerates those AND the
chiral/dielectric BRANCHES, because below the deconfinement onset two chiral
branches coexist at fixed dilaton: a confined one, where sigma is near f_pi,
the dielectric is nearly opaque and the quarks are too heavy to appear at all,
and a restored one, where sigma has collapsed and the quarks are present. They
are both genuine roots of the same field equations.

A solver that alternates between updating sigma and omega_0 two-cycles between
them and exits with a MIXED state -- sigma from one branch, omega_0 from the
other -- which reads as a spuriously deep minimum at zero quark density. So
each branch is seeded separately, solved to self-consistency, and compared by
Omega; a branch that fails to converge is reported missing, never replaced by
a neighbour, because silently substituting a converged neighbour is how a fake
phase boundary appears.

That first-order coexistence IS the deconfinement transition of this model,
and it is why the raw branches may violate dP/dn_B >= 0 between them: CLAUDE.md
section 8 allows exactly that before a construction (Maxwell, Gibbs or the
eta-mixed phase of `eos.mixed`) resolves it.
"""
from dataclasses import dataclass

from eos.general.fermi_integrals import DEGENERACY  # noqa: F401
from eos.general.pairing import (
    CHARGE, COLOURS, DEFAULT_PATTERNS, FLAVOURS, FLAVOUR_OF_MODE, MODES,
    N_MODES, PATTERNS, STRANGENESS, pattern_mask, pattern_seed,
)

#: Density-dictionary keys for the nine modes, in MODES order: 'u_r', 'u_g',
#: ... The flavour totals 'u', 'd', 's' are what the conserved-charge sums of
#: `eos.general.basis` understand, and are what a `PhaseThermo` from this
#: model is keyed by.
MODE_NAMES = tuple(f"{f}_{a}" for f, a in MODES)


#: Why a flag this model does not have is fixed where it is, in the words the
#: error message uses.
_WHY_FIXED = {
    "hyperons": (
        False,
        "this is a quark model: there are no baryons in it to be strange. "
        "Strangeness enters through the s quark, which is always present"),
    "deltas": (
        False,
        "this is a quark model and carries no baryon resonances"),
    "thermal_mesons": (
        False,
        "the mesons of this Lagrangian -- sigma, pi, zeta and the dilaton -- "
        "are the mean fields that give the quarks their masses, not a "
        "separate thermal population. Giving them one would double-count the "
        "condensate they already are, and the mesonic fluctuations that "
        "would populate them are beyond mean field"),
}


@dataclass(frozen=True)
class SpeciesFlags:
    """Which degrees of freedom are active.

    The names are the repository's, so a caller switching between models
    writes the same thing. Setting a flag this model does not implement RAISES
    naming the reason (`_WHY_FIXED`); it is never quietly ignored, because a
    sector that is off must be off because its flag says so.

    csc
        The colour-superconducting sector (L3): the three gaps become
        unknowns, the pairing correction enters Omega, eps, s and every
        density, and the two colour chemical potentials are solved from colour
        neutrality WITHIN the pattern. Off, the quarks are unpaired and
        mu_3 = mu_8 = 0 identically.
    muons
        The muon lepton family, in leptonic equilibrium at mu_mu = mu_e.
    thermal_neutrinos
        Neutrino flavours not tracked in the composition, as mu = 0 gases:
        they contribute to eps, P and s only.
    photons
        Blackbody photons; eps, P and s only, no conserved charge.
    two_flavour
        The strange QUARK sector, off. With it on the matter is u and d only:
        the three s modes carry no medium, so n_s = 0, Y_S = 0 and mu_S = 0
        identically, and the flavour leaves the matter rather than being
        emptied by a fraction that happens to vanish (CLAUDE.md section 4).
        This is how two-flavour quark matter is reached -- 
        `beta_eq_neutrinoless` with the sector off -- and it is the upper half
        of the Bodmer-Witten window; see `zero_pressure_point`.

        THE s CONDENSATE STAYS. Only the s FERMI SEA is emptied: phi_s is
        still solved from its own gap equation and still feeds M_u and M_d
        through the 't Hooft determinant 2 K phi_d phi_s, because the s
        condensate belongs to the QCD vacuum and not to the matter. Dropping
        it would move M_u, M_d and the subtracted vacuum constant, changing
        the model rather than the flavour content asked of it.

        RAISES under any pattern that condenses a diquark containing an s
        quark -- CFL, uSC, dSC, free -- since with no s quark there is nothing
        to pair; those patterns leave the default enumeration and an explicit
        request for one is refused. The flag keeps both its values in the
        'unpaired' and '2SC' patterns and is a statement about the phase in
        the rest, exactly as `eos.alphabag.SpeciesFlags.gluons` is.
    """
    csc: bool = False
    muons: bool = False
    thermal_neutrinos: bool = False
    photons: bool = False
    two_flavour: bool = False
    hyperons: bool = False
    deltas: bool = False
    thermal_mesons: bool = False

    def __post_init__(self):
        for flag, (fixed_at, why) in _WHY_FIXED.items():
            if getattr(self, flag) is not fixed_at:
                raise NotImplementedError(
                    f"SpeciesFlags: {flag} is fixed at {fixed_at} in "
                    f"eos.ccdm -- {why}")


# =============================================================================
# THE CHIRAL / DIELECTRIC BRANCHES
# =============================================================================
# Where a solve of each branch starts, as (Phi, sigma/sigma_0, zeta/zeta_0),
# with Phi = phi_bar^4 the dilaton solve variable of section 1 of the
# specification. A branch is a DECLARATION, like a mode or a pattern: it adds
# no code, it says which root the solve is aimed at.
#
# The confined seed sits just inside the guard, Phi -> 1, where chi -> 0 and
# the effective masses diverge: that is the confining vacuum, and starting
# exactly at Phi = 1 would overflow M* rather than approach it.
#
# The partially restored seed exists because the transition is first order and
# the intermediate root is a real one over a window of densities -- not
# because the other two sometimes fail. Solving all three and comparing Omega
# is what makes the onset a located crossing rather than an artefact of which
# seed happened to be used.

BRANCHES = {
    "confined": (1.0 - 1.0e-6, 1.0, 1.0),
    "restored": (0.4 ** 4, 0.0, 0.0),
    "partial": (0.4 ** 4, 0.5, 0.7),
}

#: The branches enumerated at a fixed POTENTIAL, in the order they are tried.
#: Omega decides between them; the order only decides which of two exactly
#: degenerate answers is reported. The confined branch is in here because it
#: is what the deconfined one has to beat: it carries no quarks, so its
#: pressure is exactly zero, and the onset is where the deconfined pressure
#: crosses it.
POTENTIAL_BRANCHES = ("confined", "restored", "partial")

#: The branches enumerated at a fixed DENSITY. The confined branch is absent
#: on purpose rather than by oversight: with the dielectric closed the quarks
#: are not in the medium at all, so n_B = 0 identically and no nonzero density
#: row can be met. Seeding it anyway hands the root finder a Jacobian whose
#: field columns are all zero -- every integral is exactly zero, so no field
#: moves anything -- and the solve fails after doing the work. A mode that
#: fixes n_B > 0 is by construction asking about deconfined matter.
DENSITY_BRANCHES = ("restored", "partial")


def branch_seed(par, branch):
    """The starting (Phi, sigma, zeta) of one branch, in MeV.

    The stored fractions are of the VACUUM condensates, so a parameter point
    with a different m_sigma or f_K gets a seed scaled to its own vacuum
    rather than to the shipped one.
    """
    if branch not in BRANCHES:
        raise ValueError(f"unknown branch {branch!r}; eos.ccdm enumerates "
                         f"{sorted(BRANCHES)}")
    Phi, f_sigma, f_zeta = BRANCHES[branch]
    d = par.derived
    return Phi, f_sigma * d.sigma_0, f_zeta * d.zeta_0
