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

    The hidden-strange vector phi has no flag, and unlike DD2Y or SFHo it has
    no coupling that can switch it off either. DID stores neither g_omega nor
    g_phi: both are DERIVED from the aggregated strength g~_omegaN and the
    SU(3) ratio z through `couplings.g8_from_aggregate` (Eq. 52), because that
    combination is what the Bayesian analysis varies. The per-multiplet ratios
    g_phi/g_8 = -tan(theta) - c_i(z, alpha) then have no common zero: at ideal
    mixing z = 1/sqrt6 kills only the nucleon's, tan(theta) = 0 only Lambda's
    and Sigma's. So the phi is part of the model at every composition and at
    every parameter set -- a structural statement, with no input that could
    ask otherwise and nothing left to refuse.
    """
    hyperons: bool = False
    deltas: bool = False
    muons: bool = False
    thermal_mesons: bool = False
    thermal_neutrinos: bool = False
    photons: bool = False

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
