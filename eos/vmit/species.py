"""Which degrees of freedom are active in a vMIT calculation.

The flag names are the ones every model in this repository uses, so a caller
switching between a hadronic and a quark phase writes the same thing. Most of
them have no meaning for deconfined quark matter -- there are no hyperons in a
quark phase, only strange quarks -- and asking for one raises rather than being
quietly ignored: a sector that is off must be off because its flag says so,
never because the model happened not to look at it.

The three light flavours u, d, s are always present, the way nucleons always
are in a hadronic model. Electrons are present wherever the mode has a lepton
condition (both beta-equilibrium modes, and the fixed-fraction modes with
`leptons=True`); that follows from the mode, not from a flag.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class SpeciesFlags:
    """Active degrees of freedom beyond the three quark flavours.

    photons          -- a thermal photon gas; contributes to eps, P and s and
                        carries no conserved charge. Matters only at T > 0.
    muons            -- the muon lepton family. Not wired: vMIT's lepton
                        sector is electrons (and, in the trapped mode,
                        electron neutrinos) only.
    hyperons,
    deltas,
    thermal_mesons   -- hadronic sectors, meaningless in a deconfined phase.
    thermal_neutrinos-- neutrino flavours not tracked in the composition,
                        carried as mu = 0 gases. Not wired.
    """
    photons: bool = False
    muons: bool = False
    hyperons: bool = False
    deltas: bool = False
    thermal_mesons: bool = False
    thermal_neutrinos: bool = False

    def __post_init__(self):
        for flag in ("hyperons", "deltas", "thermal_mesons"):
            if getattr(self, flag):
                raise NotImplementedError(
                    f"SpeciesFlags: {flag} is a hadronic sector and has no "
                    f"meaning in deconfined quark matter; strangeness enters "
                    f"vMIT through the s quark")
        if self.muons:
            raise NotImplementedError(
                "SpeciesFlags: the muon lepton family is not wired in vMIT "
                "(see docs/DEFERRED.md)")
        if self.thermal_neutrinos:
            raise NotImplementedError(
                "SpeciesFlags: thermal_neutrinos (untracked mu = 0 neutrino "
                "gases) are not wired in vMIT (see docs/DEFERRED.md)")
