"""
eos.mixed
=========
Phase-2 DD2 + vMIT eta-mixed-phase equation-of-state engine
(docs/phase2/SPECIFICATION_AND_PLAN.md).

P0 (scaffolding) public surface: the regime assignment (`ChargeSpec`, the four
`mode_*` factories), the unknown-vector owner (`MixedState`), and the two
engine adapters (`quark_phase`, `hadronic_phase`) returning a uniform
`PhaseThermo`. The equilibrium solver, eta split, and residual assembly are
later milestones (P1+).
"""
from eos.mixed.charges import (
    ChargeSpec, Regime, QUARK_QN, quark_charges, hadronic_qn, hadronic_charges,
)
from eos.mixed.modes import mode_A, mode_B, mode_C, mode_D
from eos.mixed.state import MixedState, charge_potential_slots
from eos.mixed.phases import PhaseThermo, quark_phase, hadronic_phase

__all__ = [
    "ChargeSpec", "Regime", "QUARK_QN", "quark_charges", "hadronic_qn",
    "hadronic_charges", "mode_A", "mode_B", "mode_C", "mode_D",
    "MixedState", "charge_potential_slots",
    "PhaseThermo", "quark_phase", "hadronic_phase",
]
