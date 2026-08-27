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
eps, P and s only, carries no conserved charge, and vanishes at T = 0. It
defaults to False like every other sector here: a flag with two legal values
is a default and must be off unless asked for, whether or not its name is one
the whole repository shares. A bag model without a thermal gluon gas is the
standard MIT configuration, not a broken one, which is why this flag defaults
rather than being fixed the way `abpr` fixes it.
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
                         own sector. In the `cfl` mode the gluons are massive
                         through the Meissner effect and their thermal
                         population is suppressed, so the free gas this flag
                         adds is not part of the paired phase's own potential;
                         see the model document.
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
    photons: bool = False
    gluons: bool = False
    thermal_neutrinos: bool = False
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
