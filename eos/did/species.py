"""Which degrees of freedom are active -- every one an explicit named boolean.

The flag names are the repository's (CLAUDE.md section 4), so a caller
switching between models writes the same thing. Nucleons are always present;
everything else is a flag, and a flag DID has not wired raises rather than
being quietly ignored.
"""
from dataclasses import dataclass

from eos.general.particles import NUCLEONS, HYPERONS_OCTET, DELTAS


@dataclass(frozen=True)
class SpeciesFlags:
    """Active degrees of freedom beyond the nucleons.

    hyperons          -- the Lambda, Sigma, Xi octet. Switching it on is what
                         turns DID into DIDY; the couplings are the same
                         parameter set (see `parameters.Parameters`).
    deltas            -- the Delta(1232) quartet. NOT in arXiv:2511.15646:
                         an extension of this implementation, coupled through
                         the ratios x_iDelta on the parameter object.
    muons             -- the muon lepton family.
    thermal_mesons    -- the thermal pi, K Bose gas. It carries electric
                         charge and strangeness, so it enters n_C and n_S and
                         hence neutrality and the fixed-fraction conditions,
                         not only eps, P and s.
    thermal_neutrinos -- neutrino flavours NOT tracked in the composition,
                         carried as mu = 0 gases; they contribute to eps, P
                         and s only.
    photons           -- a thermal photon gas; matters only at T > 0.
    phi_field         -- the hidden-strange vector phi. Unlike DD2Y, DID
                         couples the phi to the NUCLEON as well (SU(3) with
                         z != 1/sqrt6 gives g_phiN != 0, and at the published
                         set g_phiN = -5.2), so the field is part of the
                         system at every composition and False raises:
                         dropping it would change the model, not a sector.
    """
    hyperons: bool = False
    deltas: bool = False
    muons: bool = True
    thermal_mesons: bool = False
    thermal_neutrinos: bool = False
    photons: bool = True
    phi_field: bool = True

    def __post_init__(self):
        if not self.phi_field:
            raise NotImplementedError(
                "SpeciesFlags: phi_field=False is not a DID configuration -- "
                "the SU(3) vector sector gives the nucleon a phi coupling "
                "(g_phiN = -5.2 at the published set), so the phi field is "
                "part of the model at every composition")

    @property
    def has_strange_baryons(self):
        return self.hyperons


def active_baryons(flags):
    """The active baryon `Particle` objects, in a fixed order."""
    baryons = list(NUCLEONS)
    if flags.hyperons:
        baryons += list(HYPERONS_OCTET)
    if flags.deltas:
        baryons += list(DELTAS)
    return baryons
