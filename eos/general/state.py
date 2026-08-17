"""The records every model passes around: what a thermodynamic state IS.

One shape for one job (CLAUDE.md section 13). A hadronic mean-field model, a
bag model and a mixed phase all describe matter with the same quantities, so
they return the same record and a caller does not learn a new one per model.

What defines a state
--------------------
Always the chemical potentials and the temperature, plus **the model's own
internal self-consistent solution** -- and only that last part differs between
models:

    dd2, sfho      the meson mean fields          {sigma, omega0, rho0, phi0}
    enjl           the constituent masses         {M_u, M_d, M_s}
    vmit           the vector field               {V}
    zl             the interaction potentials     {mu_Hv_p, mu_Hv_n}
    alphabag, abpr none -- explicit in mu         {}

so `fields` carries it under the model's own names and is simply empty where
the model has no self-consistency. Everything else in `PhaseThermo` follows
from those.

Note that this is NOT the same thing as a solver's unknown vector. vMIT solves
for (mu_u, mu_d, mu_s, mu_e, n_u, n_d, n_s) because carrying the densities
keeps the residual polynomial in V instead of nesting the Fermi integrals
inside it -- a conditioning choice that belongs to `solver.py`. The state is
still (mu_q, V, T). Which numbers a solver iterates on is numerics; which
numbers define the state is physics.

Every species is equal
----------------------
`densities`, `mu_i`, `mu_eff_i` and `m_eff_i` are all keyed by species name,
so a neutron is an entry exactly as Lambda and Delta++ are. Nothing here
privileges nucleons. That matters for more than tidiness: the effective
potential mu_eff_i and the effective mass m_eff_i are NOT recoverable from a
record that omits them, because they depend on the fields and on each
species' own coupling ratios -- so a composition g-mode or a frozen-composition
response would have to re-derive them.

Units are fm-based on every boundary, as everywhere else in `eos`: densities
fm^-3, P and eps MeV/fm^3, T and potentials MeV.
"""
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Optional

from eos.general.basis import charges_from_densities, species_potential


def _frozen(mapping):
    """A read-only view of a mapping, so a frozen record is genuinely frozen.

    `dataclass(frozen=True)` freezes the field binding, not the dict behind
    it. The proxy cannot be pickled, so the records below unwrap for the wire
    and re-wrap on arrival -- CLAUDE.md section 6 requires model objects to
    survive being sent to a worker process.
    """
    return MappingProxyType(dict(mapping or {}))


class _PicklableFrozenMaps:
    """Pickle support for a frozen dataclass holding read-only mappings."""

    #: Names of the fields normalised to a read-only view by __post_init__.
    _MAP_FIELDS = ()

    def __getstate__(self):
        state = dict(self.__dict__)
        for name in self._MAP_FIELDS:
            state[name] = dict(state[name])
        return state

    def __setstate__(self, state):
        for name, value in state.items():
            object.__setattr__(self, name, value)
        for name in self._MAP_FIELDS:
            object.__setattr__(self, name, _frozen(state[name]))


@dataclass(frozen=True)
class PhaseThermo(_PicklableFrozenMaps):
    """One phase of strongly-interacting matter, fully solved.

    MATTER ONLY: no leptons and no photons. Those are not part of a phase --
    they are shared by the whole system, and in a mixed phase they may be
    shared differently from the matter (see `eos.mixed`). `LeptonThermo`
    carries them.

    A thermal meson gas, on the other hand, IS part of the phase and belongs in
    `densities` alongside the baryons: pi and K carry electric charge and
    strangeness, so counting them is what makes `n_C` and `n_S` the totals that
    neutrality and the fixed-Y_C / fixed-Y_S conditions are stated in terms of
    (CLAUDE.md section 2). Leaving them out gives the baryon-only fractions,
    which at T = 40 MeV differ by 10-20 percent.
    """
    T: float                            # MeV
    #: Conserved-charge potentials. mu_C is the NON-leptonic charge potential,
    #: mu_C = mu_p - mu_n, so beta equilibrium reads mu_C + mu_e = 0.
    mu_B: float
    mu_C: float
    mu_S: float
    #: The model's own self-consistent solution, in its own names (see module
    #: docstring). Empty for a model that has none.
    fields: Mapping[str, float]
    #: Every active species: baryons, quarks, and any thermal meson gas.
    densities: Mapping[str, float]      # fm^-3
    #: mu_i = B_i mu_B + C_i mu_C + S_i mu_S, derived, never independent.
    mu_i: Mapping[str, float]           # MeV
    #: mu_eff_i = mu_i - Sigma0_i, the potential that enters the Fermi
    #: integrals. Carried because it cannot be reconstructed without the
    #: fields and the per-species couplings.
    mu_eff_i: Mapping[str, float]       # MeV
    #: The Dirac effective masses, per species and for the same reason.
    m_eff_i: Mapping[str, float]        # MeV
    n_B: float                          # fm^-3
    n_C: float
    n_S: float
    P: float                            # MeV/fm^3
    eps: float                          # MeV/fm^3
    s: float                            # fm^-3
    #: sum_i mu_i n_i over this phase, the Euler sum. Supplied by the model
    #: rather than derived here, because a mean field splits the field energy
    #: between mu and P in a way only the model knows: a thermal meson gas, for
    #: instance, contributes sum_j mu*_j n_j at its EFFECTIVE potentials.
    mu_dot_n: float                     # MeV/fm^3
    #: Rearrangement self-energy; enters mu and P, never eps. Zero for a model
    #: whose couplings do not depend on density.
    Sigma_R: float = 0.0                # MeV
    #: max_j |mu*_j| / m_j over the thermal meson gas; 0 without one. At 1 the
    #: gas Bose-condenses and the ideal-gas expressions stop describing it, so
    #: every consumer of this record has to be able to see it: `solve_bose_jel`
    #: caps mu at m rather than diverging, which means a condensed phase comes
    #: back looking perfectly converged. Carried on the phase rather than
    #: recomputed by each consumer because it is a property OF the phase --
    #: `eos.mixed` reads it across the phase-adapter contract for exactly that
    #: reason (see docs/DEFERRED.md for what a condensate would need).
    condensation: float = 0.0

    _MAP_FIELDS = ("fields", "densities", "mu_i", "mu_eff_i", "m_eff_i")

    def __post_init__(self):
        for name in self._MAP_FIELDS:
            object.__setattr__(self, name, _frozen(getattr(self, name)))

    # --- fractions: always relative to n_B (CLAUDE.md section 2) -----------
    @property
    def Y_C(self):
        """Total non-leptonic charge fraction, meson gas included."""
        return self.n_C / self.n_B if self.n_B else 0.0

    @property
    def Y_S(self):
        """Total strangeness fraction, +1 per s quark, meson gas included."""
        return self.n_S / self.n_B if self.n_B else 0.0

    def Y(self, name):
        """Population fraction n_i / n_B of one species (0 if absent)."""
        return self.densities.get(name, 0.0) / self.n_B if self.n_B else 0.0

    @property
    def free_energy_density(self):
        """f = eps - T s [MeV/fm^3]."""
        return self.eps - self.T * self.s

    def euler_residual(self):
        """(eps + P - T s - sum_i mu_i n_i) / eps.

        The Euler / Hugenholtz-Van Hove identity as a number to test rather
        than an assertion to trip: CLAUDE.md section 8 requires it to hold to
        about 1e-8 relative, and `verify/` is where that is enforced.
        """
        if self.eps == 0.0:
            return 0.0
        return (self.eps + self.P - self.T * self.s - self.mu_dot_n) / self.eps

    @classmethod
    def assemble(cls, T, mu_B, mu_C, mu_S, fields, densities, mu_eff_i,
                 m_eff_i, P, eps, s, mu_dot_n, Sigma_R=0.0,
                 extra_charges=(0.0, 0.0, 0.0), condensation=0.0):
        """Build a state, deriving everything section 2 says is derived.

        The caller supplies what only the model knows -- its fields, its
        species densities and effective quantities, and the bulk P, eps, s.
        The conserved-charge densities and the species potentials are computed
        here, from `eos.general.basis`, so that no model carries its own copy
        of those algebraic maps and two models cannot disagree about what
        n_C means.

        `extra_charges` is (n_B, n_C, n_S) carried by species that contribute
        conserved charge but are not listed individually in `densities`. It
        exists for one reason: a thermal meson gas carries charge and
        strangeness, and most of its members (the vector nonet, K0bar) are not
        yet in `eos.general.particles`, so they cannot be summed through the
        quantum-number table. Once they are, the gas becomes ordinary species
        entries and this argument goes -- see docs/DEFERRED.md. Species that
        carry NO conserved charge (pi0, eta, omega, and photons) are never
        listed either way; they contribute to eps, P and s alone.
        """
        n_B, n_C, n_S = charges_from_densities(densities)
        extra_B, extra_C, extra_S = extra_charges
        mu_i = {name: species_potential(name, mu_B, mu_C, mu_S)
                for name in densities}
        return cls(T=T, mu_B=mu_B, mu_C=mu_C, mu_S=mu_S, fields=fields,
                   densities=densities, mu_i=mu_i, mu_eff_i=mu_eff_i,
                   m_eff_i=m_eff_i, n_B=n_B + extra_B, n_C=n_C + extra_C,
                   n_S=n_S + extra_S,
                   P=P, eps=eps, s=s, mu_dot_n=mu_dot_n, Sigma_R=Sigma_R,
                   condensation=condensation)


@dataclass(frozen=True)
class LeptonThermo(_PicklableFrozenMaps):
    """The lepton sector of a state.

    Separate from `PhaseThermo` because leptons are not part of a phase: they
    do not feel the strong mean fields, they carry no B or S, and they are
    excluded from C by convention (CLAUDE.md section 2). In a mixed phase they
    may also be distributed differently from the matter -- locally inside each
    phase, globally over the mixture, or a weighted mixture of the two.

    All four potentials are explicit, which is the point. The muon family is
    usually taken to be transparent, mu_numu = 0, and writing that as a stored
    zero rather than leaving it out makes the assumption visible and gives the
    trapped-muon mode of CLAUDE.md section 3 somewhere to live.
    """
    mu_e: float = 0.0                   # MeV
    mu_mu: float = 0.0
    mu_nue: float = 0.0
    mu_numu: float = 0.0
    #: {'e-': n_e, 'mu-': n_mu, 'nu_e': n_nue, 'nu_mu': n_numu} [fm^-3], net
    #: (particle minus antiparticle). The charged-lepton keys are the names in
    #: `eos.general.particles`, so a caller can look their quantum numbers up.
    densities: Mapping[str, float] = field(default_factory=dict)
    P: float = 0.0                      # MeV/fm^3
    eps: float = 0.0
    s: float = 0.0                      # fm^-3
    mu_dot_n: float = 0.0               # MeV/fm^3

    _MAP_FIELDS = ("densities",)

    def __post_init__(self):
        object.__setattr__(self, "densities", _frozen(self.densities))

    @property
    def n_charged(self):
        """Net negatively-charged lepton density, what neutrality balances.

        Neutrinos are excluded: they carry lepton number, not charge.
        """
        return self.densities.get("e-", 0.0) + self.densities.get("mu-", 0.0)


@dataclass(frozen=True)
class EoSPoint(_PicklableFrozenMaps):
    """What a mode returns: the conditions it was asked for, and the state.

    The independent variables come first because they are what identifies the
    point -- (n_B, T) plus whichever fractions the mode fixes -- and the solved
    state follows. `P`, `eps` and `s` are the TOTALS: matter plus leptons plus
    photons, which is what a structure solver integrates, while `matter.P` and
    friends are the phase alone.

    Non-convergence is a value here, not an exception (CLAUDE.md section 6):
    `converged` is judged on `residual`, the largest scaled equilibrium
    residual, so a sampler can score a bad point and move on.
    """
    mode: str
    n_B: float                          # fm^-3
    T: float                            # MeV
    #: The fractions the mode fixed: {'Y_C': ..., 'Y_S': ..., 'Y_Le': ...}.
    #: Empty for beta equilibrium, which fixes none.
    conditions: Mapping[str, float]
    matter: PhaseThermo
    leptons: Optional[LeptonThermo]
    P: float                            # MeV/fm^3, TOTAL
    eps: float                          # MeV/fm^3, TOTAL
    s: float                            # fm^-3, TOTAL
    converged: bool = True
    residual: float = 0.0

    _MAP_FIELDS = ("conditions",)

    def __post_init__(self):
        object.__setattr__(self, "conditions", _frozen(self.conditions))

    @property
    def entropy_per_baryon(self):
        """S/A = s / n_B, the axis an isentrope is drawn along."""
        return self.s / self.n_B if self.n_B else 0.0

    @property
    def free_energy_density(self):
        """f = eps - T s [MeV/fm^3], totals."""
        return self.eps - self.T * self.s
