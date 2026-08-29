"""
species.py
====================
SpeciesFlags — every degree of freedom is an explicit named boolean.

No sector is ever switched on or off implicitly by "its coupling happens to be
zero": if a species is absent, its flag is False. Setting a flag that is not
yet wired raises rather than silently doing nothing, so a table can never
quietly omit a sector the caller asked for.

All six names of CLAUDE.md section 4 are carried. Five are wired: hyperons,
Delta isobars, muons, photons and thermal_mesons -- the last being section 4's
"pi, K", with the optional vector nonet behind the secondary flag
thermal_vectors. `thermal_neutrinos` is carried and RAISES: the flavours a
mode does not track are not wired here, and dd2's own `neutrinos` is the
matter-composition electron neutrino of the trapped modes, a different sector.

Also wired, as dd2's own physics: trapped neutrinos. The hidden-strange
scalar sigma* is not.

The hidden-strange VECTOR phi has no flag, because dd2 already carries its
coupling: x_phiY = g_phiY/g_omegaN is SU(6) times the free factor y_phi_Y, one
per multiplet. The sector is on exactly when hyperons are present and some
x_phiY is nonzero (`par.has_phi_coupling`), and the SU(6) column has no zero in
it, so `y_phi_Lambda = y_phi_Sigma = y_phi_Xi = 0.0` is the whole statement
that there is no phi, made where every other model number is made. A boolean beside
it would have been a second way to say the same thing, and not one an
inference sampler could vary.

EVERY flag here defaults to False, dd2's own included. A flag with two legal
values is a default and must be off unless asked for; a flag with only one
legal value raises on the other instead of defaulting (sigma_star,
thermal_neutrinos).
"""
from dataclasses import dataclass

# The baryon quantum-number maps are general-purpose -- they read only the
# shared `Particle` objects and a flags object carrying `hyperons`/`deltas`
# -- so their single home is `general/basis` (CLAUDE.md section 7). They are
# re-exported here because every dd2 caller reaches for them beside the
# flags they take.
from eos.general.basis import (active_baryons, hadronic_qn,
                               hadronic_charges)


@dataclass(frozen=True)
class SpeciesFlags:
    hyperons: bool = False              # Λ, Σ, Ξ octet
    deltas: bool = False                # Δ quartet
    muons: bool = False                 # e always on; μ optional
    thermal_mesons: bool = False        # thermal π,K,η,η' Bose gas
    thermal_neutrinos: bool = False     # the ν flavours a mode does NOT track
                                        # (the τ family): unwired here, raises
    photons: bool = False               # radiation (matters only at T>0)
    neutrinos: bool = False             # matter-composition ν_e of the trapped
                                        # modes — NOT thermal_neutrinos above
    sigma_star: bool = False            # hidden-strange SCALAR σ* (later)
    thermal_vectors: bool = False       # thermal ρ,ω,K*,φ Bose gas: section 4's
                                        # "optionally the vector nonet"

    def __post_init__(self):
        # `neutrinos` is wired: it is the trapped-Y_Le mode's electron
        # neutrino, reached through solve_beta_eq_neutrino_trapped.
        if self.sigma_star:
            raise NotImplementedError(
                "SpeciesFlags: sigma_star (hidden-strange scalar) is not wired")
        if self.thermal_neutrinos:
            raise NotImplementedError(
                "SpeciesFlags: thermal_neutrinos -- the neutrino flavours a "
                "mode does not track, carried as mu = 0 gases -- is not wired "
                "in dd2. It is NOT `neutrinos`, which is the "
                "matter-composition electron neutrino of the trapped modes")

    @property
    def has_strange_baryons(self):
        return self.hyperons  # (or deltas carrying strangeness — none do)
