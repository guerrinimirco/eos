"""
mixed/charges.py
================
Which conserved charges are equilibrated, and how — the single declaration that
configures the whole mixed-phase solver.

*Internal module.* The four named modes below are re-exported from
``eos.mixed``; import them from there.

A hadron-quark mixed phase is a two-phase equilibrium in which each conserved
quantity in {B, C, S, L_e} may be treated in one of three ways:

    GLOBAL         the potential is shared, mu_j^H = mu_j^Q, and the charge is
                   conserved only in the volume average
                   (1-chi) n_j^H + chi n_j^Q = Y_j n_B
    LOCAL          the potential is per-phase; the charge is conserved inside
                   each phase separately
    NOT_CONSERVED  the charge is not conserved at all: mu_j^I = 0 in each phase
                   and the potential is eliminated from the unknown vector

`ChargeSpec` records that choice per charge, and *nothing else in the engine
enumerates equilibrium modes*: the unknown vector (`mixed_slots`), the residual
list (`residual`) and the Jacobian are all assembled by reading the
regimes off a ChargeSpec. The four named modes are therefore four
configurations of one solver, and an unnamed combination is constructible by
instantiating `ChargeSpec` directly.

The three regimes are not a third thing beside `eos.general.modes`: they are
that module's two-valued Conservation refined by ONE extra axis, which is all
a second phase adds. A ChargeSpec is a `ModeSpec` plus a locality per charge,
so the four modes of CLAUDE.md section 3 are declared once, in `general/`, and
this module says only where each conserved charge is conserved. NOT_CONSERVED
is `general`'s EQUILIBRATED seen from here.

Baryon number is GLOBAL in every mode: mu_B^H = mu_B^Q is what makes the two
phases coexist at all.

Sign conventions (repo-wide, see CLAUDE.md §2 and eos/general/particles.py):

    S = +1 per s-quark   (Lambda S=+1, Xi S=+2, the s quark itself S=+1).
                         This is the OPPOSITE of the PDG convention.
    C                    is the NON-leptonic electric charge: C = Q for hadrons
                         and quarks, C = 0 for leptons. Total electric-charge
                         neutrality is a separate, additional condition.

This module is data only — quantum numbers and the regime declaration. It does
no solving.
"""
from dataclasses import dataclass, field
from enum import Enum

from eos.general.particles import Up, Down, Strange
from eos.general.basis import quark_charges
from eos.general import modes
from eos.general.modes import ModeSpec
from eos.dd2.species import hadronic_qn, hadronic_charges


class Regime(Enum):
    """How one conserved charge is treated across the two phases."""
    GLOBAL = "global"
    LOCAL = "local"
    NOT_CONSERVED = "not_conserved"


class Locality(Enum):
    """On which volume a CONSERVED charge is conserved.

    The second axis a two-phase system needs, and the only one `eos.general`
    does not carry: on the volume average (GLOBAL, one shared potential) or
    inside each phase separately (LOCAL, a potential per phase). It says
    nothing about whether the charge is held at all -- that is the mode's job.
    """
    GLOBAL = "global"
    LOCAL = "local"


@dataclass(frozen=True)
class ChargeSpec:
    """
    One mixed-phase configuration: a mode, plus where each charge is conserved.

    A two-phase equilibrium needs exactly one axis more than a single phase, so
    that is exactly what this adds. `mode` is the shared declaration from
    `eos.general.modes` -- WHICH charges are held, at what fractions, and
    whether neutralizing leptons are present -- and the three locality fields
    say, for each held charge, on which volume. `Regime` is the two composed,
    and is what the rest of the engine reads:

        the mode does not hold it      ->  NOT_CONSERVED
        held, Locality.GLOBAL          ->  GLOBAL
        held, Locality.LOCAL           ->  LOCAL

    B is GLOBAL in every mode and is not stored: the two phases coexist only if
    the baryon potential matches, mu_B^H = mu_B^Q.

    Nothing else in the engine enumerates equilibrium modes -- the unknown
    vector (`mixed_slots`), the residual list (`residual`) and the
    Jacobian are all assembled by reading the regimes off a ChargeSpec -- so
    the four named modes below are four configurations of one solver, and an
    unnamed combination is constructible by passing a `ModeSpec` directly.
    """
    mode: ModeSpec = field(default_factory=ModeSpec)
    C_locality: Locality = Locality.GLOBAL
    S_locality: Locality = Locality.GLOBAL
    L_e_locality: Locality = Locality.GLOBAL

    # The mode validates its own targets and survives pickling on its own
    # (`eos.mixed.scan` runs a parameter scan across processes, and CLAUDE.md
    # section 6 requires that), so neither is restated here. Three enums beside
    # it are picklable as they stand.

    def _regime(self, charge, locality):
        if not self.mode.is_fixed(charge):
            return Regime.NOT_CONSERVED
        return Regime.GLOBAL if locality is Locality.GLOBAL else Regime.LOCAL

    @property
    def B(self):
        return Regime.GLOBAL

    @property
    def C(self):
        return self._regime("C", self.C_locality)

    @property
    def S(self):
        return self._regime("S", self.S_locality)

    @property
    def L_e(self):
        return self._regime("L_e", self.L_e_locality)

    @property
    def targets(self):
        """The fixed fractions, in CLAUDE.md section 5's condition names."""
        return self.mode.targets

    @property
    def yc_leptons(self):
        """Are neutralizing leptons present in the fixed-Y_C mode?

        False wherever C is not held: with the charge potential fixed by beta
        equilibrium instead, there is no leptonless flavour to choose between.
        """
        return self.mode.is_fixed("C") and self.mode.leptons


# =============================================================================
# QUANTUM NUMBERS
# =============================================================================
#: Quark flavor -> (B, C, S), taken from eos/general/particles.py so the repo
#: has exactly one table of quantum numbers.
QUARK_QN = {q.name: (q.baryon_no, q.charge, q.strangeness)
            for q in (Up, Down, Strange)}


# `quark_charges` is NOT redefined here. The basis change (n_u, n_d, n_s) ->
# (n_B, n_C, n_S) is declared once, in `eos.general.basis` (CLAUDE.md section
# 2); this re-export keeps `eos.mixed.quark_charges` working while leaving the
# engine one function of that name in scope rather than two.


# The hadronic quantum numbers depend only on the DD2 species set, so they live
# with it in eos/dd2/species.py and are re-exported here beside the quark ones.


# =============================================================================
# THE NAMED EQUILIBRIUM MODES
# =============================================================================
# Each is one line: a regime choice per charge. The independent variables the
# resulting table is a function of are listed with each factory.
#
#   mode                       | B      | C      | S      | L_e    | table axes
#   ---------------------------|--------|--------|--------|--------|-----------
#   beta_eq_neutrinoless       | global | -      | -      | -      | nB, T
#   beta_eq_neutrino_trapped   | global | -      | -      | global | nB, Y_Le, T
#   fixed_YC                   | global | global | -      | -      | nB, Y_C, T
#   fixed_YC_YS                | global | global | global | -      | nB,Y_C,Y_S,T
#
# ("-" = NOT_CONSERVED.)

def beta_eq_neutrinoless():
    """Neutrino-transparent beta equilibrium. Independent variables (n_B, T).

    Nothing but baryon number is conserved: strangeness self-equilibrates
    (mu_S = 0) and the charge potential is fixed by the beta condition
    mu_C = -mu_e, so no charge fraction is imposed. This is the cold /
    neutrino-transparent neutron-star condition.
    """
    return ChargeSpec(modes.beta_eq_neutrinoless())


def beta_eq_neutrino_trapped(Y_Le):
    """Beta equilibrium with trapped neutrinos at fixed total electron lepton
    fraction Y_Le = (n_e + n_nue)/n_B. Independent variables (n_B, Y_Le, T).

    L_e is GLOBAL: the trapped neutrinos are treated as one uniform gas shared
    by both phases (mu_nu^H = mu_nu^Q) with no local component and no eta
    weighting. That is a modelling assumption — neutrinos have a mean free path
    far larger than the mixed-phase structures, so they cannot be localized in
    a droplet — and it is worth surfacing in output metadata.
    """
    return ChargeSpec(modes.beta_eq_neutrino_trapped(Y_Le))


def fixed_YC(Y_C, *, leptons=False):
    """Fixed non-leptonic charge fraction Y_C. Independent variables (n_B,Y_C,T).

    Two flavors:
      leptons=False  leptonless — a charged slice of matter, the CompOSE
                     general-purpose (n_B, T, Y_q) convention. eta-independent,
                     since with no leptons there is no neutrality to localize.
      leptons=True   neutralizing electrons (and muons if enabled) are present,
                     so the total system is electrically neutral while the
                     hadron+quark charge fraction is held at Y_C.
    """
    return ChargeSpec(modes.fixed_YC(Y_C, leptons=leptons))


def fixed_YC_YS(Y_C, Y_S, *, leptons=False):
    """Fixed Y_C and fixed strangeness fraction Y_S, both globally conserved.
    Independent variables (n_B, Y_C, Y_S, T).

    Strangeness is GLOBAL — mu_S^H = mu_S^Q, and Y_S counts the strangeness of
    both phases together, consistent with how Y_C is treated and with the
    tabulated-EoS convention. The alternative per-phase reading (S LOCAL, each
    phase separately at its own Y_S) is not wired; `mixed_slots` raises for it
    rather than mis-assembling silently.

    `leptons` selects the charge flavor exactly as `fixed_YC`.
    """
    return ChargeSpec(modes.fixed_YC_YS(Y_C, Y_S, leptons=leptons))
