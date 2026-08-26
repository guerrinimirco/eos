"""
general/sound_speeds.py
=======================
The two sound speeds a composition g-mode is built from, and the table that
carries them from a model to `eos.astro.gmode`.

Why two speeds, and why they are the contract
---------------------------------------------
The Brunt-Vaisala frequency of a composition g-mode is (Zhao and Lattimer,
"Universal relations for neutron star g-mode oscillations", arXiv:2204.03037,
Eq. (1); Jaikumar, Semposki, Prakash and Constantinou, Phys. Rev. D 103,
123009 (2021), Eqs. (2)-(5))

    N^2 = g^2 (1/c_e^2 - 1/c_s^2) e^{nu - lambda}

so the mode exists only through the DIFFERENCE of

* the **equilibrium** speed `c_e^2 = dP/deps` along the equilibrated sequence,
  the one the TOV equations already see, and
* the **frozen** speed `c_s^2 = (dP/deps)_x` at fixed chemical composition `x`,
  the response of a fluid element compressed faster than the reactions that
  would restore equilibrium.

Given one of them alone `N^2` is identically zero: a single-sound-speed
equation of state supports no composition g-mode at all. That is why the two
speeds together, and not a composition derivative, are what a model owes the
mode solver. `eos_response` (CLAUDE.md section 5) already computes both per
model, under `frozen='equilibrium'` and `frozen='composition'`.

This module is the layer both sides may import (CLAUDE.md section 1): a model
or a composite engine PRODUCES an `EOSTable_for_gmode`, and `eos.astro.gmode`
CONSUMES one. Neither imports the other, exactly as `EOSTable_for_TOV` in
`general/state.py` -- which this table extends -- mediates the model/TOV
boundary.

Zero temperature
----------------
The tables here are T = 0. Zhao's operative clause is "without varying
chemical composition", not the vanishing temperature: at T = 0 with a
composition that varies along the sequence, `c_e != c_s` and the g-mode is
nonzero -- the DD2-with-leptons case. What T = 0 removes is the THERMAL axis,
leaving the composition axis intact, so a point carries exactly two numbers
and neither needs a name saying which thermal variable was held. Finite T adds
that axis, and the isothermal/adiabatic distinction with it; it is future work
and this table does not pretend to cover it.

Units are the repository's fm-based public ones: P and eps in MeV/fm^3, n_B in
fm^-3. Sound speeds are dimensionless, in units of c.
"""
from dataclasses import dataclass

import numpy as np

from eos.general.state import EOSTable_for_TOV


def sound_speed_eq(P, eps):
    """Equilibrium c_e^2 = dP/deps along a solved sequence.

    `P` and `eps` are parallel arrays in MeV/fm^3, ascending in density -- an
    `EOSTable_for_TOV`'s columns, or the stitched columns a notebook
    assembles. A central finite difference on the sequence itself, so it costs
    no extra solves; the sequence must be dense enough that the derivative is
    meaningful, which through a transition window means resolving the window.

    Arrays rather than a table object, so that the pure phases, a stitched
    core equation of state and a notebook's own columns all go through one
    function.
    """
    P = np.asarray(P, dtype=float)
    eps = np.asarray(eps, dtype=float)
    if P.size < 2:
        return np.full(P.shape, np.nan)
    return np.gradient(P, eps)


def sound_speed_frozen(compressed_state, rel_dn=1e-3):
    """Frozen c_s^2 = (dP/deps)_x, by central difference on a compression.

    `compressed_state(scale)` returns `(P, eps)` in MeV/fm^3 for the state
    compressed by the factor `scale` at FIXED composition -- supplying it is
    the model's half of the work, since only the model knows how to re-solve
    its own fields at a new density without letting the composition move.
    This function is the difference quotient alone, and lives here so that
    every model spells the derivative, its step and its degenerate case the
    same way (CLAUDE.md section 7).

    Returns nan rather than raising when the two states are not ordered in
    energy density, so one bad density leaves a gap in a curve instead of
    destroying it.
    """
    P_lo, eps_lo = compressed_state(1.0 - rel_dn)
    P_hi, eps_hi = compressed_state(1.0 + rel_dn)
    if not (eps_hi > eps_lo):
        return float("nan")
    return (P_hi - P_lo) / (eps_hi - eps_lo)


def cs2_frozen_isobaric(cs2_H, cs2_Q, chi):
    """Combine per-phase frozen sound speeds at a common pressure.

        1/c_s,mix^2 = (1 - chi)/c_s,H^2 + chi/c_s,Q^2

    cs2_H, cs2_Q : frozen c^2 of the hadronic and quark phases, each evaluated
                   at *its own* density in the mixture (dimensionless)
    chi          : quark volume fraction in [0, 1]

    The volume-fraction-weighted RECIPROCAL combination: strong and
    electromagnetic processes are fast on a g-mode period, so the two phases
    share one pressure perturbation and each answers it with its own deps.
    It follows from eps_mix = (1 - chi) eps_H + chi eps_Q at P_H = P_Q, and is
    the relation used in the g-mode literature (Jaikumar et al., Phys. Rev. D
    103, 123009 (2021), Eq. (75)).

    Scalars or broadcastable arrays. At chi = 0 this returns cs2_H exactly and
    at chi = 1 it returns cs2_Q, so the pure phases need no special-casing;
    whichever phase carries zero weight may be passed as nan.
    """
    chi = np.clip(np.asarray(chi, dtype=float), 0.0, 1.0)
    cs2_H = np.asarray(cs2_H, dtype=float)
    cs2_Q = np.asarray(cs2_Q, dtype=float)
    # Drop the absent phase before dividing, so a nan there cannot poison a
    # limit that does not depend on it.
    inv_H = np.where(chi < 1.0, (1.0 - chi) / np.where(chi < 1.0, cs2_H, 1.0), 0.0)
    inv_Q = np.where(chi > 0.0, chi / np.where(chi > 0.0, cs2_Q, 1.0), 0.0)
    total = inv_H + inv_Q
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total > 0.0, 1.0 / total, np.nan)


@dataclass
class EOSTable_for_gmode(EOSTable_for_TOV):
    """The (P, eps, n_B) table plus the two sound speeds, at T = 0.

    This is the CONTRACT between the models and `eos.astro.gmode`, the
    composition counterpart of `EOSTable_for_TOV`: a model or a composite
    engine produces one, and the mode solver consumes it. It lives in
    `general/` for the reason that one does -- both sides may import this
    layer, and a model may not import `astro/` (CLAUDE.md section 1).

    It IS an `EOSTable_for_TOV`, so the stellar structure the mode needs is
    integrated from the same object that carries the buoyancy, and anything
    that already takes a TOV table takes this one.

    P, epsilon, nB  : inherited, MeV/fm^3 and fm^-3, ascending in density
    cs2_equilibrium : c_e^2 = dP/deps along the equilibrated sequence
    cs2_frozen      : c_s^2 = (dP/deps)_x at fixed composition

    The names say which axis is held: BOTH speeds are taken at T = 0, so the
    thermal axis distinguishes nothing here and naming it would only suggest a
    freedom the table does not carry. What separates them is composition,
    which is what `equilibrium` against `frozen` says.

    Nothing here validates monotonicity, causality, or convective stability
    (`cs2_frozen >= cs2_equilibrium`, i.e. N^2 >= 0). A raw model branch may
    legitimately violate the first two inside a first-order transition
    (section 8), and the third fails loudly and usefully in a `verify/` suite
    when the two speeds have been computed for different fluids -- one with
    the neutralizing leptons and one without. `check_lengths` is the one thing
    asserted, because a length mismatch is a silent misalignment rather than a
    physics statement.

    `EOSTable_for_TOV.from_file` reads three columns and cannot build this
    table; construct it from a model's `eos_response` output instead.
    """
    cs2_equilibrium: np.ndarray
    cs2_frozen: np.ndarray

    def __post_init__(self):
        super().__post_init__()
        self.cs2_equilibrium = np.asarray(self.cs2_equilibrium)
        self.cs2_frozen = np.asarray(self.cs2_frozen)
        self.check_lengths()

    def check_lengths(self):
        """All five columns are parallel; a mismatch is a misalignment."""
        sizes = {"P": self.P.size, "epsilon": self.epsilon.size,
                 "nB": self.nB.size,
                 "cs2_equilibrium": self.cs2_equilibrium.size,
                 "cs2_frozen": self.cs2_frozen.size}
        if len(set(sizes.values())) != 1:
            raise ValueError(
                f"EOSTable_for_gmode columns must have equal length, got {sizes}")

    @classmethod
    def from_columns(cls, P, epsilon, nB, cs2_frozen, cs2_equilibrium=None):
        """Build from a model's columns, differencing the table for `c_e`.

        `cs2_equilibrium` defaults to `sound_speed_eq(P, epsilon)`, the
        derivative of the table the model already produced; pass it explicitly
        to use a model's own `eos_response(frozen='equilibrium')` value
        instead, which re-solves at each point rather than differencing the
        grid.
        """
        if cs2_equilibrium is None:
            cs2_equilibrium = sound_speed_eq(P, epsilon)
        return cls(P=P, epsilon=epsilon, nB=nB,
                   cs2_equilibrium=cs2_equilibrium, cs2_frozen=cs2_frozen)


__all__ = [
    "sound_speed_eq", "sound_speed_frozen", "cs2_frozen_isobaric",
    "EOSTable_for_gmode",
]
