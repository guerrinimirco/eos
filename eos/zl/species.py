"""Which degrees of freedom are active in a ZL calculation.

The flag names are the ones every model in this repository uses, so a caller
switching between models writes the same thing. Nucleons are always present;
everything else is an explicit named boolean, and a flag this model has not
wired RAISES rather than being quietly ignored -- a sector that is off must be
off because its flag says so, never because the model happened not to look at
it.

ZL is a nucleonic functional: its interaction is written in n_p and n_n alone,
so there is no coupling for a hyperon, a Delta or a meson to carry and those
sectors are absent rather than merely unimplemented. Electrons are present
wherever the mode has a lepton condition (both beta-equilibrium modes, and
`fixed_YC` with `leptons=True`); that follows from the mode, not from a flag.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class SpeciesFlags:
    """Active degrees of freedom beyond the nucleons.

    photons           -- a thermal photon gas; contributes to eps, P and s and
                         carries no conserved charge. Matters only at T > 0.
    muons             -- the muon lepton family. Not wired: ZL's leptons are
                         electrons, and electron neutrinos in the trapped mode.
    hyperons, deltas,
    thermal_mesons    -- sectors the functional has no couplings for.
    thermal_neutrinos -- neutrino flavours NOT tracked in the composition,
                         carried as mu = 0 gases. Not wired.
    """
    photons: bool = True
    muons: bool = False
    hyperons: bool = False
    deltas: bool = False
    thermal_mesons: bool = False
    thermal_neutrinos: bool = False

    def __post_init__(self):
        for flag in ("hyperons", "deltas", "thermal_mesons"):
            if getattr(self, flag):
                raise NotImplementedError(
                    f"SpeciesFlags: {flag} has no coupling in the "
                    f"Zhao-Lattimer functional, which is written in n_p and "
                    f"n_n alone; the sector is absent from the model, not "
                    f"merely unimplemented")
        if self.muons:
            raise NotImplementedError(
                "SpeciesFlags: the muon lepton family is not wired in ZL "
                "(see docs/DEFERRED.md)")
        if self.thermal_neutrinos:
            raise NotImplementedError(
                "SpeciesFlags: thermal_neutrinos (untracked mu = 0 neutrino "
                "gases) are not wired in ZL (see docs/DEFERRED.md)")
