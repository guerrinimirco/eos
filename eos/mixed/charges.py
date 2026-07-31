"""
mixed/charges.py
================
Regime assignment for the DD2+vMIT eta-mixed-phase engine
(docs/phase2/SPECIFICATION_AND_PLAN.md §1.5; docs/phase2/STEP0_AUDIT.md §3).

Each conserved quantity in {B, C, S, L_e} independently sits in one of three
regimes, and the regime alone determines how it contributes to the mixed-phase
solver's unknown vector and residual list (Appendix A of the thesis):

    GLOBAL         mu_j^H = mu_j^Q            one shared unknown       (A.94)
    LOCAL          absorbed into baryon rel.  per-phase unknowns       (A.93)
    NOT_CONSERVED  mu_j^I = 0 in each phase   eliminated

The four named modes (A-D) are four *configurations* of this assignment, not
four solvers — see modes.py. An unnamed combination must be constructible by
instantiating `ChargeSpec` directly.

This module is DATA ONLY. It declares the regime assignment (`ChargeSpec`) and
the per-species quantum-number tables used to build n_B, n_C, n_S in each
phase. No solving logic lives here; the P1 residual assembly reads it.

Sign conventions (CLAUDE.md §2, reused from eos/general/particles.py, not
re-tabulated): S = +1 per s-quark (Lambda S=+1, Xi S=+2); C is the
NON-leptonic electric charge (C = Q for hadrons/quarks, 0 for leptons).
"""
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from eos.dd2.species import active_baryons


class Regime(Enum):
    """How a conserved charge is treated across the two phases (§1.5)."""
    GLOBAL = "global"
    LOCAL = "local"
    NOT_CONSERVED = "not_conserved"


#: charge name -> the `targets` key carrying its fixed fraction (when GLOBAL/LOCAL).
_TARGET_KEY = {"C": "Y_C", "S": "Y_S", "L_e": "Y_L"}


@dataclass(frozen=True)
class ChargeSpec:
    """
    Per-charge regime assignment for one mixed-phase configuration
    (spec §1.5, §3.2).

    `B` is always GLOBAL (baryon number matches across phases in every mode,
    Eq. 3.24). `C`, `S`, `L_e` default to NOT_CONSERVED, i.e. Mode A
    (beta-equilibrium, no fixed fractions). `targets` carries the fixed
    fractions (`Y_C`, `Y_S`, `Y_L`) required by the GLOBAL/LOCAL charges.
    `yc_leptons` selects between the two fixed-Y_C flavors of §1.6:
    leptonless (False) vs. neutralizing leptons (True).
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
                "B must be GLOBAL in every mode: baryon number matches across "
                "phases, mu_B^H = mu_B^Q (spec §1.4, Eq. 3.24)")
        # A charge that fixes a fraction (GLOBAL/LOCAL) needs its target and
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
                "yc_leptons (§1.6 flavor 2b) only applies when C is GLOBAL "
                "(fixed Y_C)")
        # Normalize to an immutable mapping so a frozen ChargeSpec is truly
        # frozen (dataclass only freezes the field binding, not dict contents).
        object.__setattr__(self, "targets",
                           MappingProxyType(dict(self.targets)))


# =============================================================================
# QUANTUM-NUMBER TABLES  (data the P1 residual assembly consumes)
# =============================================================================

#: Quark (B, C, S) per flavor. S = +1 per s-quark (CLAUDE.md §2); C = Q.
QUARK_QN = {
    "u": (1.0 / 3.0, 2.0 / 3.0, 0.0),
    "d": (1.0 / 3.0, -1.0 / 3.0, 0.0),
    "s": (1.0 / 3.0, -1.0 / 3.0, 1.0),
}


def quark_charges(n_u, n_d, n_s):
    """(n_B, n_C, n_S) of the quark phase from flavor densities (spec §1.3).

    n_B^Q=(n_u+n_d+n_s)/3, n_C^Q=(2n_u-n_d-n_s)/3, n_S^Q=n_s. Same convention
    vMIT itself uses (thermodynamics_quarks.py); units follow the input.
    """
    n_B = (n_u + n_d + n_s) / 3.0
    n_C = (2.0 * n_u - n_d - n_s) / 3.0
    n_S = n_s
    return n_B, n_C, n_S


def hadronic_qn(flags):
    """(name, B, C, S) for each active baryon under `flags` (spec §1.3).

    C is the non-leptonic charge (= electric charge for baryons); S carries the
    repo sign (+1 per s-quark). Reused straight from eos/general/particles.py.
    """
    return tuple((b.name, b.baryon_no, b.charge, b.strangeness)
                 for b in active_baryons(flags))


def hadronic_charges(flags, densities):
    """(n_B, n_C, n_S) of the hadronic phase from a {name: n} map (spec §1.3).

    `densities` and the returned quantities share whatever units the caller
    passes in. Baryons absent from the map contribute zero.
    """
    n_B = n_C = n_S = 0.0
    for name, B, C, S in hadronic_qn(flags):
        n = densities.get(name, 0.0)
        n_B += B * n
        n_C += C * n
        n_S += S * n
    return n_B, n_C, n_S
