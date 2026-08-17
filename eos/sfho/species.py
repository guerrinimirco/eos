"""
species.py
==========
Which degrees of freedom are active — every one an explicit named boolean.

The flag names are the repository's, so a caller switching between models
writes the same thing (CLAUDE.md §4). Nucleons are always present; everything
else is a flag, and a flag a model has not wired RAISES rather than being
quietly ignored. A sector that is off must be off because its flag says so.

SFHo adds one check the other models do not need. Its parameter sets do not
all cover the same species: `SFHo_Nucleonic` carries no hyperon and no Delta
couplings at all, and `SFHoY_Fortin` carries hyperons but no Deltas. Every
coupling in an absent sector is stored as 0.0, so enabling that sector without
a parametrization to match it silently gives a FREE gas of those baryons —
right degeneracy, right mass, no interaction — which converges happily and is
not the model anyone asked for. `check_couplings` turns that into an error and
names the sets that do carry the sector.
"""
from dataclasses import dataclass

from eos.general.particles import NUCLEONS, HYPERONS_OCTET, DELTAS


@dataclass(frozen=True)
class SpeciesFlags:
    """Active degrees of freedom beyond the nucleons.

    hyperons          -- Lambda, Sigma, Xi. Needs a parametrization carrying
                         hyperon couplings (SFHoY_Fortin, SFHoY*_Fortin,
                         SFHo_2fam, SFHo_2fam_phi).
    deltas            -- the Delta(1232) quartet. Needs SFHo_2fam or
                         SFHo_2fam_phi; the others have no Delta couplings.
    muons             -- the muon lepton family. NOT WIRED: SFHo's leptons are
                         electrons, and electron neutrinos in the trapped mode.
    thermal_mesons    -- the thermal pi, K, eta Bose gas. Carries electric
                         charge and strangeness, so it enters n_C and n_S and
                         hence neutrality and the fixed-fraction conditions,
                         not only eps, P and s.
    thermal_neutrinos -- neutrino flavours NOT tracked in the composition,
                         carried as mu = 0 gases; they contribute to eps, P
                         and s only.
    photons           -- a thermal photon gas. Carries no conserved charge and
                         matters only at T > 0.
    phi_field         -- the hidden-strange vector phi. SFHo always carries it
                         in the unknown vector, so False raises: dropping the
                         field is a change to the system's size, not a flag.
    """
    hyperons: bool = False
    deltas: bool = False
    muons: bool = False
    thermal_mesons: bool = False
    thermal_neutrinos: bool = False
    photons: bool = True
    phi_field: bool = True

    def __post_init__(self):
        if self.muons:
            raise NotImplementedError(
                "SpeciesFlags: the muon lepton family is not wired in SFHo — "
                "it appears in no residual, no neutrality row and no total "
                "(see docs/DEFERRED.md). eos.dd2 implements it.")
        if not self.phi_field:
            raise NotImplementedError(
                "SpeciesFlags: SFHo always solves the phi field; phi_field="
                "False would shrink the unknown vector, which is a solver "
                "change rather than a flag (see docs/DEFERRED.md)")

    @property
    def has_strange_baryons(self):
        return self.hyperons


def active_baryons(flags):
    """Ordered list of active baryon Particles for the given flags.

    The order is `eos.general.particles`', and it is the order the per-species
    sums run in, so it is part of the numerics and not a presentation choice.
    """
    baryons = list(NUCLEONS)
    if flags.hyperons:
        baryons += list(HYPERONS_OCTET)
    if flags.deltas:
        baryons += list(DELTAS)
    return baryons


#: sector -> (the baryons it switches on, the published sets that couple them).
#: Used only for the error message; the check itself reads the couplings.
_SECTOR_SETS = {
    "hyperons": ("SFHoY_Fortin, SFHoY*_Fortin, SFHo_2fam, SFHo_2fam_phi"),
    "deltas": ("SFHo_2fam, SFHo_2fam_phi"),
}


def check_couplings(par, flags):
    """Raise if a sector is enabled that `par` does not actually couple.

    Every coupling of an absent sector is stored as 0.0, so the solve would
    succeed and return a free baryon gas. This is CLAUDE.md §4's rule read
    from the parameter side: no sector is enabled or disabled implicitly
    because its coupling happens to be zero.
    """
    sectors = (("hyperons", flags.hyperons, HYPERONS_OCTET),
               ("deltas", flags.deltas, DELTAS))
    for name, wanted, group in sectors:
        if not wanted:
            continue
        strongest = max(abs(par.get_coupling(b.name, meson))
                        for b in group for meson in ("sigma", "omega", "rho"))
        if strongest == 0.0:
            raise ValueError(
                f"SpeciesFlags({name}=True) with parametrization "
                f"{getattr(par, 'name', '?')!r}, which has every {name} "
                f"coupling at zero — the solve would return a FREE gas of "
                f"those baryons, not the model. Sets that couple them: "
                f"{_SECTOR_SETS[name]}.")
