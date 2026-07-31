"""
mixed/state.py
==============
`MixedState` — the single owner of the mixed-phase unknown-vector layout
(docs/phase2/SPECIFICATION_AND_PLAN.md §3.2).

Every mapping between the flat solver vector and named physical quantities
lives here and nowhere else. Nothing outside this module may index the vector
positionally. That is what makes later additions — the deferred muon e_L/e_G
split (§1.7), or D-local's per-phase mu_S — a targeted change to one slot list
rather than a rewrite of the residual code.

`MixedState` is a generic pack/unpack mechanism over an ordered tuple of slot
names. `charge_potential_slots(spec)` derives the conserved-charge-potential
slots from a ChargeSpec's regime assignment (the settled part of the layout).
The electron / neutrino e_L/e_G slots that carry the eta split (§1.1-1.4) are
added by the P1 residual assembly, using this same mechanism.

Solver unknowns are KINETIC potentials where a species potential would
otherwise be used (nu_i = mu_i - Sigma0_i; CLAUDE.md §2, spec §3.2) — the
charge-potential slots below (mu_B, mu_C, mu_S) are the conserved-charge
multipliers of that decomposition and warm-start well across a density sweep.
"""
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from eos.mixed.charges import ChargeSpec, Regime


def _charge_slots(name, regime):
    """Slot names a single charge contributes, by regime (spec §1.5, audit §3).

    GLOBAL -> one shared potential; LOCAL -> per-phase (H, Q); NOT_CONSERVED ->
    none (the potential is eliminated, mu_j^I = 0 in each phase).
    """
    if regime is Regime.GLOBAL:
        return [f"mu_{name}"]
    if regime is Regime.LOCAL:
        return [f"mu_{name}_H", f"mu_{name}_Q"]
    return []


def charge_potential_slots(spec: ChargeSpec):
    """Ordered conserved-charge-potential slots implied by `spec` (spec §3.3).

    Covers `chi` and the B/S/C charge potentials — the part of the layout fixed
    purely by the regime assignment. B is always GLOBAL, so `mu_B` is always a
    shared unknown. The electron/neutrino eta-split slots are NOT included here;
    P1 appends them (see module docstring).
    """
    slots = ["chi", "mu_B"]                    # chi; B GLOBAL (Eq. 3.24)
    slots += _charge_slots("S", spec.S)
    slots += _charge_slots("C", spec.C)
    return tuple(slots)


@dataclass(frozen=True)
class MixedState:
    """Bidirectional map between a flat unknown vector and named slots.

    Build with `MixedState.for_charges(spec)` for the charge-potential layout,
    or `MixedState(slots)` directly with any ordered slot names (P1 uses the
    latter to append the electron sector).
    """
    slots: tuple

    @classmethod
    def for_charges(cls, spec: ChargeSpec):
        return cls(charge_potential_slots(spec))

    def __len__(self):
        return len(self.slots)

    def index(self, name: str) -> int:
        return self.slots.index(name)

    def pack(self, values: Mapping[str, float]) -> np.ndarray:
        """Named values -> flat vector in slot order. Missing slot -> KeyError."""
        return np.array([values[name] for name in self.slots], dtype=float)

    def unpack(self, x: Sequence[float]) -> dict:
        """Flat vector -> {slot: value}. Length must match."""
        if len(x) != len(self.slots):
            raise ValueError(
                f"vector length {len(x)} != {len(self.slots)} slots {self.slots}")
        return {name: float(x[i]) for i, name in enumerate(self.slots)}
