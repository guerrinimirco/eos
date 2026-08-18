"""Which degrees of freedom are active in an alphaBag calculation.

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

`gluons` is this model's own sector, which no other model in the repository
has: a thermal gluon gas with its own alpha_s correction. It contributes to
eps, P and s only, carries no conserved charge, and vanishes at T = 0.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class SpeciesFlags:
    """Active degrees of freedom beyond the three quark flavours.

    photons           -- a thermal photon gas; contributes to eps, P and s and
                         carries no conserved charge. Matters only at T > 0.
    gluons            -- a thermal gluon gas, 16 massless bosons at mu = 0
                         with the correction factor 1 - 15 alpha_s/(4 pi).
                         Same bookkeeping as the photons, and this model's
                         own sector.
    thermal_neutrinos -- the neutrino flavours NOT tracked in the composition,
                         carried as mu = 0 gases: three where the electron
                         neutrino is free-streaming, two where it is trapped
                         and therefore already counted at its own potential.
    muons             -- the muon lepton family. Not wired: alphaBag's lepton
                         sector is electrons (and, in the trapped mode,
                         electron neutrinos) only.
    hyperons,
    deltas,
    thermal_mesons    -- hadronic sectors, meaningless in a deconfined phase.
    """
    photons: bool = True
    gluons: bool = True
    thermal_neutrinos: bool = True
    muons: bool = False
    hyperons: bool = False
    deltas: bool = False
    thermal_mesons: bool = False

    def __post_init__(self):
        for flag in ("hyperons", "deltas", "thermal_mesons"):
            if getattr(self, flag):
                raise NotImplementedError(
                    f"SpeciesFlags: {flag} is a hadronic sector and has no "
                    f"meaning in deconfined quark matter; strangeness enters "
                    f"alphaBag through the s quark")
        if self.muons:
            raise NotImplementedError(
                "SpeciesFlags: the muon lepton family is not wired in "
                "alphaBag (see docs/DEFERRED.md)")
