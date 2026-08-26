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

Also wired, as dd2's own physics: trapped neutrinos and the hidden-strange
vector phi. The hidden-strange scalar sigma* is not.
"""
from dataclasses import dataclass

from eos.general.particles import NUCLEONS, HYPERONS_OCTET, DELTAS


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
    phi_field: bool = True              # hidden-strange VECTOR φ (DD2Y default)
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


def active_baryons(flags):
    """Ordered list of active baryon Particles for the given flags."""
    baryons = list(NUCLEONS)
    if flags.hyperons:
        baryons += list(HYPERONS_OCTET)
    if flags.deltas:
        baryons += list(DELTAS)
    return baryons


def hadronic_qn(flags):
    """(name, B, C, S) for each baryon active under `flags`.

    Strangeness is S = +1 per s-quark, so Lambda has S = +1 and Xi has S = +2.
    That is the opposite of the PDG sign and is used consistently throughout
    this repository.
    """
    return tuple((b.name, b.baryon_no, b.charge, b.strangeness)
                 for b in active_baryons(flags))


def hadronic_charges(flags, densities):
    """(n_B, n_C, n_S) of hadronic matter from a {name: n} map.

    n_C is the NON-leptonic electric charge density; total electric neutrality
    is a separate condition that also counts the leptons. `densities` and the
    result share whatever units the caller passes in, and baryons absent from
    the map contribute zero.
    """
    n_B = n_C = n_S = 0.0
    for name, B, C, S in hadronic_qn(flags):
        n = densities.get(name, 0.0)
        n_B += B * n
        n_C += C * n
        n_S += S * n
    return n_B, n_C, n_S
