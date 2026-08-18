"""Which degrees of freedom are active in an ABPR calculation: none of them.

The flag names are the ones every model in this repository uses, so a caller
switching between a hadronic and a quark phase writes the same thing. Every
one of them is off here, and for two different reasons that are both physics
rather than unfinished work:

  - the hadronic sectors (hyperons, deltas, thermal mesons) have no meaning in
    a deconfined phase, and strangeness enters through the s quark;
  - the leptons have nothing to do: colour-flavour locking makes the phase
    electrically neutral by construction, with n_C = 0 identically, so there
    are no electrons or muons to neutralize it;
  - the thermal sectors (photons, gluons, thermal neutrinos) are identically
    zero at T = 0, which is the only temperature this model has.

The three flavours u, d, s are always present, the way nucleons always are in
a hadronic model, and locked to equal densities. Asking for any sector raises
rather than being quietly ignored: a sector that is off must be off because
its flag says so, never because the model happened not to look at it.
"""
from dataclasses import dataclass

#: Why each flag cannot be switched on, in the words the error message uses.
_WHY_OFF = {
    "hyperons": "a hadronic sector with no meaning in a deconfined phase; "
                "strangeness enters ABPR through the s quark",
    "deltas": "a hadronic sector with no meaning in a deconfined phase",
    "thermal_mesons": "a hadronic sector with no meaning in a deconfined "
                      "phase",
    "muons": "the CFL phase is electrically neutral by construction "
             "(n_C = 0), so it carries no leptons of any family",
    "photons": "a thermal sector, identically zero at T = 0, which is the "
               "only temperature this model has; for T > 0 use the 'cfl' "
               "mode of eos.alphabag",
    "gluons": "a thermal sector, identically zero at T = 0, and in the CFL "
              "phase the gluons are Meissner-massive besides; for T > 0 use "
              "the 'cfl' mode of eos.alphabag",
    "thermal_neutrinos": "a thermal sector, identically zero at T = 0, which "
                         "is the only temperature this model has",
}


@dataclass(frozen=True)
class SpeciesFlags:
    """Active degrees of freedom beyond the three locked quark flavours.

    All False, and setting any of them raises with the reason above. The
    dataclass exists so that the uniform API of this repository takes the same
    `species` argument here as everywhere else, and so that a caller who
    switches a sector on is told why it is empty rather than getting a table
    that silently ignored the request.
    """
    photons: bool = False
    gluons: bool = False
    thermal_neutrinos: bool = False
    muons: bool = False
    hyperons: bool = False
    deltas: bool = False
    thermal_mesons: bool = False

    def __post_init__(self):
        for flag, why in _WHY_OFF.items():
            if getattr(self, flag):
                raise NotImplementedError(f"SpeciesFlags: {flag} is {why}")
