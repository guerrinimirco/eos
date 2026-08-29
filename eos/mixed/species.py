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
    across the whole mixture and are counted ONCE, at the mixture level. So
    on the MIXTURE path — a phase's `thermo`, evaluated at fixed potentials
    and weighted by chi — the phases contribute matter and the mixture
    contributes the radiation, and no adapter's `thermo` adds a photon gas.

    A `wing_sweep` is the other path and takes the opposite rule. Its rows
    are that phase's own pure solve, stitched into the hybrid table AS THEY
    STAND with no mixture layer above them to add the radiation, so a wing
    carries the CALLER'S OWN `photons` — every adapter, without exception.
    The two paths meet at n_offset, where chi = 1 and the last window row and
    the first wing row describe the same matter; a wing short of the gas puts
    a step of P_gamma = pi^2 T^4/45(hc)^3 there, which is 0.023 MeV/fm^3 at
    T = 30 and nothing the physics puts in a table CLAUDE.md section 8
    requires be monotone. That is why every shipped adapter takes a `flags`
    argument: `zl_phase`, `vmit_phase` and `alphabag_phase` used to be built
    from parameters alone and could only hardcode `photons=False`, and the
    DD2 + vMIT front door — one flags object into `default_pair` and into the
    mixture — produced exactly that step at T > 0.

    A sector the wing could solve and the mixture cannot match would be the
    same defect wearing another name, so it RAISES rather than becoming a
    wing-only gas: `alphabag_phase` refuses `gluons` and `two_flavour`,
    `vmit_phase` refuses `two_flavour` (CLAUDE.md section 4 — never a silent
    no-op). `thermal_neutrinos` is refused here, at the mixture level, for
    every pairing.

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
