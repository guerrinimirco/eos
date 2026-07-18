"""
species.py
====================
SpeciesFlags — every degree of freedom is an explicit named boolean
(report §3.6, prompt ground rule 2). No sector is switched on/off implicitly
by "its coupling happens to be zero".

Milestone wiring status:
    hyperons, muons, neutrinos-off, photons, phi_field  — M4
    deltas                                              — M5 (this milestone)
    sigma_star, include_pseudoscalars,
    include_thermal_vectors                             — later milestones
Setting a not-yet-wired flag raises (no silent no-op).
"""
from dataclasses import dataclass

from eos.general.particles import NUCLEONS, HYPERONS_OCTET, DELTAS


@dataclass(frozen=True)
class SpeciesFlags:
    hyperons: bool = False              # Λ, Σ, Ξ octet
    deltas: bool = False                # Δ quartet
    muons: bool = True                  # e always on; μ optional
    neutrinos: bool = False             # only trapped / fixed-Y_L modes
    photons: bool = True                # radiation (matters only at T>0)
    phi_field: bool = True              # hidden-strange VECTOR φ (DD2Y default)
    sigma_star: bool = False            # hidden-strange SCALAR σ* (later)
    include_pseudoscalars: bool = False  # thermal π,K,η,η' Bose gas (M7)
    include_thermal_vectors: bool = False  # thermal ρ,ω,K*,φ Bose gas (M7)

    def __post_init__(self):
        unwired = []
        if self.sigma_star:
            unwired.append("sigma_star (later milestone)")
        if self.include_pseudoscalars:
            unwired.append("include_pseudoscalars (M7)")
        if self.include_thermal_vectors:
            unwired.append("include_thermal_vectors (M7)")
        if self.neutrinos:
            unwired.append("neutrinos (M6)")
        if unwired:
            raise NotImplementedError(
                "SpeciesFlags: not yet wired at this milestone: "
                + ", ".join(unwired))

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
