"""
mixed/species.py
================
SpeciesFlags for the composite engine — CLAUDE.md section 4's six names.

A composite engine is not a model, but section 4 binds it exactly as it binds
one: every degree of freedom beyond the nucleons is an explicit named boolean,
the names are identical everywhere, and no sector is ever switched on
implicitly. All six default to False.

The six split in two, and that split is the engine's own geometry:

PER-PHASE — hyperons, deltas, thermal_mesons, and the muons that join the
    neutralizing lepton domains. These are degrees of freedom OF THE MODELS
    BEING COUPLED; the engine implements none of them and delegates, handing
    each `Phase` its own model's flags. A model's flag object carries these
    same six names plus that model's private physics (DD2's
    matter-composition neutrinos), which section 4 does not name and the
    engine never reads — which is why any model's flag object serves as the
    engine's too, through `mixture_flags` below.

PHASE-COMMON — photons and thermal_neutrinos. These belong to neither phase:
    like the eta-split leptons of `eos.mixed.thermodynamics` they are uniform
    across the whole mixture and are counted ONCE, at the mixture level. That
    is why every shipped adapter hands the phase it wraps a flag object with
    `photons=False` (`eos.mixed.adapters`): the phases contribute matter, the
    mixture contributes the radiation. With the flag here, that hardcoded
    False is correct by construction rather than correct by accident. The one
    exception is a phase's `wing_sweep`, whose rows are stitched into the
    hybrid table as they stand, with no mixture layer above them to add the
    radiation: those carry the caller's own `photons`. `zl_phase`,
    `vmit_phase` and `alphabag_phase` are built from parameters alone and are
    handed no flag object, so their wings have none to follow and carry
    `photons=False` -- which agrees with the mixture whenever this flag is at
    its default. `alphabag_phase` carries its gluon gas and thermal neutrino
    gases off for the same reason: both are phase-common thermal sectors and
    neither has a caller flag here to follow.

The engine's OWN physics — eta, the quark volume fraction chi — is not a
species flag and is not carried here; it is an argument of the solve.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class SpeciesFlags:
    hyperons: bool = False              # Λ, Σ, Ξ octet — delegated to a phase
    deltas: bool = False                # Δ quartet — delegated to a phase
    muons: bool = False                 # e always on; μ optional
    thermal_mesons: bool = False        # thermal π,K Bose gas — a phase's own
    thermal_neutrinos: bool = False     # the ν flavours a mode does NOT track:
                                        # phase-common, unwired here, raises
    photons: bool = False               # radiation, phase-common (T > 0 only)

    def __post_init__(self):
        if self.thermal_neutrinos:
            raise NotImplementedError(
                "SpeciesFlags: thermal_neutrinos — the neutrino flavours a "
                "mode does not track, carried as mu = 0 gases — is not wired "
                "in the mixed engine. It is NOT the trapped electron "
                "neutrino of beta_eq_neutrino_trapped, which is matter "
                "composition and comes from the mode, not from a flag")


def mixture_flags(flags):
    """The engine's own six flags, read off whatever flags object it was given.

    `None` means every sector off. A `SpeciesFlags` passes through unchanged.
    Anything else is a coupled model's own flag object: every model carries
    section 4's six names, so the engine reads those six and ignores that
    model's private physics. This is what lets the DD2 + vMIT front door hand
    one `eos.dd2.SpeciesFlags` to both the hadronic phase and the mixture.
    """
    if flags is None:
        return SpeciesFlags()
    if isinstance(flags, SpeciesFlags):
        return flags
    return SpeciesFlags(
        hyperons=bool(getattr(flags, "hyperons", False)),
        deltas=bool(getattr(flags, "deltas", False)),
        muons=bool(getattr(flags, "muons", False)),
        thermal_mesons=bool(getattr(flags, "thermal_mesons", False)),
        thermal_neutrinos=bool(getattr(flags, "thermal_neutrinos", False)),
        photons=bool(getattr(flags, "photons", False)),
    )
