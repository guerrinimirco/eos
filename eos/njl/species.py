"""The degrees of freedom of the NJL model, and their quantum numbers.

Nine colour-flavour quark modes, three leptons, and the thermal sectors that
carry no conserved charge. The quark quantum numbers are NOT re-declared here:
they are `eos.general.pairing`'s, imported, because the pairing sector indexes
its 9x9 matrices by the same modes in the same order and two copies of one
table are two chances to disagree (CLAUDE.md section 7).

Conventions are this repository's: STRANGENESS IS S = +1 PER s QUARK, the
opposite of the PDG sign, and C is the charge of strongly-interacting matter
only -- the leptons are excluded from it and enter through the separate
condition of total electric neutrality.

The one flag that changes the equations rather than adding a sector is `csc`.
With it off the model is ordinary three-flavour NJL: no gap matrix, no
Bogoliubov-de Gennes problem, no colour chemical potentials, and mu_3 = mu_8 =
0 identically. With it on, the gaps become unknowns, colour neutrality becomes
two rows, and a point carries the pairing pattern it was solved in.
"""
from dataclasses import dataclass

#: The spin degeneracy of one colour-flavour mode, re-exported from
#: `eos.general.fermi_integrals` where the ideal-gas integrals live.
from eos.general.fermi_integrals import DEGENERACY  # noqa: F401
from eos.general.pairing import (
    CHARGE, COLOURS, DEFAULT_PATTERNS, FLAVOURS, FLAVOUR_OF_MODE, MODES,
    N_MODES, PATTERNS, STRANGENESS, pattern_mask, pattern_seed,
)

#: Density-dictionary keys for the nine modes, in MODES order: 'u_r', 'u_g',
#: ... These are what a `PhaseThermo.densities` from this model is keyed by
#: when the colour resolution matters; the flavour totals 'u', 'd', 's' are
#: what the conserved-charge sums of `eos.general.basis` understand.
MODE_NAMES = tuple(f"{f}_{a}" for f, a in MODES)

#: Spin times colour, the degeneracy of the Dirac sea integral of one flavour.
DEGENERACY_SEA = 6.0


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
        "the mesons of this Lagrangian are the auxiliary fields of the "
        "four-fermion terms -- sigma, pi and the diquark -- eliminated in "
        "favour of G_S, K and G_D. They have no independent thermal "
        "population here, and the mesonic fluctuations that would give them "
        "one are beyond mean field"),
}


@dataclass(frozen=True)
class SpeciesFlags:
    """Which degrees of freedom are active.

    The names are the repository's, so a caller switching between models
    writes the same thing. Setting a flag this model does not implement RAISES
    naming the reason (`_WHY_FIXED`); it is never quietly ignored, because a
    sector that is off must be off because its flag says so.

    csc
        The colour-superconducting sector: the three gaps become unknowns, the
        pairing correction enters Omega, eps, s and every density, and the two
        colour chemical potentials are solved from colour neutrality. Off, the
        model is ordinary unpaired NJL.
    muons
        The muon lepton family, in leptonic equilibrium at mu_mu = mu_e.
    thermal_neutrinos
        Neutrino flavours not tracked in the composition, as mu = 0 gases:
        they contribute to eps, P and s only.
    photons
        Blackbody photons; eps, P and s only, no conserved charge.
    """
    csc: bool = False
    muons: bool = False
    thermal_neutrinos: bool = False
    photons: bool = False
    hyperons: bool = False
    deltas: bool = False
    thermal_mesons: bool = False

    def __post_init__(self):
        for flag, (fixed_at, why) in _WHY_FIXED.items():
            if getattr(self, flag) is not fixed_at:
                raise NotImplementedError(
                    f"SpeciesFlags: {flag} is fixed at {fixed_at} in "
                    f"eos.njl -- {why}")


#: The pairing patterns and their seeds are `eos.general.pairing`'s, re-exported
#: here so that `eos.njl.species` remains the one place a caller asks this
#: model what degrees of freedom it has. Which gaps a named pattern makes free
#: is a property of the gap matrix, shared with the colour-dielectric model,
#: not of the Lagrangian that supplies G_D (CLAUDE.md section 7).
