"""
mixed/equilibrium/charges.py
============================
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
list (`mixed_residual`) and the Jacobian are all assembled by reading the
regimes off a ChargeSpec. The four named modes are therefore four
configurations of one solver, and an unnamed combination is constructible by
instantiating `ChargeSpec` directly.

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
from types import MappingProxyType
from typing import Mapping

from eos.general.particles import Up, Down, Strange
from eos.dd2.species import hadronic_qn, hadronic_charges


class Regime(Enum):
    """How one conserved charge is treated across the two phases."""
    GLOBAL = "global"
    LOCAL = "local"
    NOT_CONSERVED = "not_conserved"


#: charge name -> the `targets` key carrying its fixed fraction (GLOBAL/LOCAL).
_TARGET_KEY = {"C": "Y_C", "S": "Y_S", "L_e": "Y_L"}


@dataclass(frozen=True)
class ChargeSpec:
    """
    Per-charge regime assignment for one mixed-phase configuration.

    `B` is always GLOBAL. `C`, `S`, `L_e` default to NOT_CONSERVED, which is
    plain beta equilibrium. `targets` carries the fixed fractions (`Y_C`,
    `Y_S`, `Y_L`) that the GLOBAL/LOCAL charges need. `yc_leptons` picks
    between the two fixed-Y_C flavors: leptonless (False, the CompOSE
    (n_B, T, Y_q) convention) or with neutralizing leptons present (True).

    Prefer the named factories at the bottom of this module; build a ChargeSpec
    directly only for a combination they do not cover.
    """
    B: Regime = Regime.GLOBAL
    C: Regime = Regime.NOT_CONSERVED
    S: Regime = Regime.NOT_CONSERVED
    L_e: Regime = Regime.NOT_CONSERVED
    targets: Mapping[str, float] = field(default_factory=dict)
    yc_leptons: bool = False

    def __post_init__(self):
        if self.B is not Regime.GLOBAL:
            raise ValueError(
                "B must be GLOBAL in every mode: the two phases coexist only "
                "if the baryon potential matches, mu_B^H = mu_B^Q")
        # A charge that fixes a fraction (GLOBAL/LOCAL) needs its target, and
        # only then; a NOT_CONSERVED charge must not carry one.
        for charge, key in _TARGET_KEY.items():
            fixes = getattr(self, charge) in (Regime.GLOBAL, Regime.LOCAL)
            given = key in self.targets
            if fixes and not given:
                raise ValueError(
                    f"{charge} is {getattr(self, charge).name} but no {key} "
                    f"target was provided in `targets`")
            if not fixes and given:
                raise ValueError(
                    f"{charge} is NOT_CONSERVED; a {key} target is meaningless")
        if self.yc_leptons and self.C is not Regime.GLOBAL:
            raise ValueError(
                "yc_leptons only applies when C is GLOBAL (fixed Y_C)")
        # Normalize to an immutable mapping so a frozen ChargeSpec really is
        # frozen (dataclass freezes the field binding, not the dict contents).
        object.__setattr__(self, "targets",
                           MappingProxyType(dict(self.targets)))

    # A mappingproxy cannot be pickled, and a ChargeSpec has to survive being
    # sent to a worker process: `eos.mixed.scan` runs a parameter scan across
    # processes, and CLAUDE.md section 6 requires model objects to be
    # picklable so multiprocessing and MPI work. Unwrap for the wire and
    # re-wrap on arrival, so the immutability holds in every process without
    # costing the portability.
    def __getstate__(self):
        state = dict(self.__dict__)
        state["targets"] = dict(state["targets"])
        return state

    def __setstate__(self, state):
        for name, value in state.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "targets",
                           MappingProxyType(dict(state["targets"])))


# =============================================================================
# QUANTUM NUMBERS
# =============================================================================
#: Quark flavor -> (B, C, S), taken from eos/general/particles.py so the repo
#: has exactly one table of quantum numbers.
QUARK_QN = {q.name: (q.baryon_no, q.charge, q.strangeness)
            for q in (Up, Down, Strange)}


def quark_charges(n_u, n_d, n_s):
    """(n_B, n_C, n_S) of the quark phase from the flavor densities.

    n_B = (n_u+n_d+n_s)/3, n_C = (2 n_u - n_d - n_s)/3, n_S = n_s — the same
    convention eos/vmit uses. Output units follow the input.
    """
    n_B = n_C = n_S = 0.0
    for n, (B, C, S) in zip((n_u, n_d, n_s), QUARK_QN.values()):
        n_B += B * n
        n_C += C * n
        n_S += S * n
    return n_B, n_C, n_S


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
#   beta_eq_neutrino_trapped   | global | -      | -      | global | nB, Y_L, T
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
    return ChargeSpec()


def beta_eq_neutrino_trapped(Y_L):
    """Beta equilibrium with trapped neutrinos at fixed total electron lepton
    fraction Y_L = (n_e + n_nue)/n_B. Independent variables (n_B, Y_L, T).

    L_e is GLOBAL: the trapped neutrinos are treated as one uniform gas shared
    by both phases (mu_nu^H = mu_nu^Q) with no local component and no eta
    weighting. That is a modelling assumption — neutrinos have a mean free path
    far larger than the mixed-phase structures, so they cannot be localized in
    a droplet — and it is worth surfacing in output metadata.
    """
    return ChargeSpec(L_e=Regime.GLOBAL, targets={"Y_L": Y_L})


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
    return ChargeSpec(C=Regime.GLOBAL, targets={"Y_C": Y_C},
                      yc_leptons=leptons)


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
    return ChargeSpec(C=Regime.GLOBAL, S=Regime.GLOBAL,
                      targets={"Y_C": Y_C, "Y_S": Y_S}, yc_leptons=leptons)
