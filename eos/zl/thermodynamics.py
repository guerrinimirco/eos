"""Thermodynamic quantities of ZL nucleonic matter, computed FROM the state.

The state is (mu_p, mu_n, T) together with the densities (n_p, n_n), which
play here the part a meson mean field plays in a relativistic mean-field
model: the interaction is a functional of the densities alone,

    V(n_p, n_n) = 4 n_p n_n [a0/n0 + b0/n0 u^(gamma-1)]
                + (n_n - n_p)^2 [a1/n0 + b1/n0 u^(gamma1-1)],   u = n_B/n0

so the potential it adds to each species is mu_Hv_i = dV/dn_i and the kinetic
(effective) potential entering the Fermi integrals is mu_eff_i = mu_i -
mu_Hv_i. The model's self-consistency is the fixed point
n_i = n_i(mu_eff_i, T, m_i), and it is `solver.py` that finds it: nothing in
this module knows which equilibrium mode it is in.

There is no scalar field, so no gap equation and no effective mass -- the
nucleons keep their vacuum mass -- and the entropy is purely kinetic, since V
carries no explicit temperature.

Reading order: one species as an ideal gas, then the interaction, then the
blocks that assemble the two.

Units at every boundary: densities in fm^-3, potentials and T in MeV, P and
eps in MeV/fm^3.

Reference: T. Zhao and J. M. Lattimer, Phys. Rev. D 102, 023021 (2020).
"""
from dataclasses import dataclass
from typing import Tuple

from eos.general import particles
from eos.general.fermi_integrals import solve_fermi_jel, invert_fermi_density
from eos.zl.parameters import Parameters

#: Spin degeneracy of a nucleon. There is no colour factor and no isospin
#: factor: the two species are summed explicitly.
G_NUCLEON = particles.get_particle("p").g_degen


# =============================================================================
# ONE SPECIES AS AN IDEAL GAS
# =============================================================================
@dataclass
class NucleonThermo:
    """One nucleon species as a free Fermi gas at its effective potential."""
    n: float = 0.0      # number density (fm^-3)
    P: float = 0.0      # pressure (MeV/fm^3)
    e: float = 0.0      # energy density (MeV/fm^3)
    s: float = 0.0      # entropy density (fm^-3)


def kinetic_thermo(mu_eff: float, T: float, m: float,
                   include_antiparticles: bool = True) -> NucleonThermo:
    """One species as an ideal Fermi gas of mass `m` at potential `mu_eff`.

    Evaluates the standard integrals over the Fermi-Dirac occupations,

        n   = (g/2pi^2) int dk k^2 [f(E-mu*) - f(E+mu*)]
        P   = (g/6pi^2) int dk k^4/E [f(E-mu*) + f(E+mu*)]
        eps = (g/2pi^2) int dk k^2 E [f(E-mu*) + f(E+mu*)]

    with E = sqrt(k^2 + m^2), g = 2, and s = (eps + P - mu* n)/T from the
    free-gas Euler relation -- which holds at the EFFECTIVE potential, and is
    why the interaction never enters the entropy. Antiparticles are included
    by default; at T = 0 their terms vanish and the integrals reduce to the
    familiar closed forms in the Fermi momentum.

    The integrals themselves are not implemented here: they come from
    `eos.general.fermi_integrals`, through the Johns-Ellis-Lattimer analytic
    approximation [Johns, Ellis & Lattimer, ApJ 473, 1020 (1996)], uniformly
    valid from the degenerate to the non-degenerate limit and exact at T = 0.
    """
    # solve_fermi_jel also returns the scalar density n_s, which this model
    # has no use for: with no scalar field there is nothing for it to source.
    n, P, e, s = solve_fermi_jel(
        mu_eff, T, m, G_NUCLEON,
        include_antiparticles=include_antiparticles)[:4]
    return NucleonThermo(n=n, P=P, e=e, s=s)


# =============================================================================
# THE INTERACTION
# =============================================================================
def interaction_energy(n_p: float, n_n: float, params: Parameters) -> float:
    """The interaction energy density V(n_p, n_n), in MeV/fm^3.

        V = 4 n_p n_n [a0/n0 + b0/n0 u^(gamma-1)]
          + (n_n - n_p)^2 [a1/n0 + b1/n0 u^(gamma1-1)]

    with u = n_B/n0. Per baryon, with delta = (n_n-n_p)/n_B, this is
    (1-delta^2)[a0 u + b0 u^gamma] + delta^2 [a1 u + b1 u^gamma1]: the first
    bracket is a proton-neutron CROSS interaction, so both brackets enter the
    symmetry energy.

    Contributes to eps; the matching contribution to P is
    `interaction_pressure`, and they differ (Zhao & Lattimer 2020).
    """
    n_B = n_p + n_n
    if n_B < 1e-15:
        return 0.0

    n0 = params.n0
    u = n_B / n0

    cross = 4.0 * n_p * n_n * (
        params.a0 / n0 + params.b0 / n0 * u**(params.gamma - 1))
    isovector = (n_n - n_p)**2 * (
        params.a1 / n0 + params.b1 / n0 * u**(params.gamma1 - 1))
    return cross + isovector


def interaction_pressure(n_p: float, n_n: float, params: Parameters) -> float:
    """The interaction contribution to the pressure, in MeV/fm^3.

        P_int = sum_i n_i mu_Hv_i - V
              = 4 n_p n_n [a0/n0 + gamma b0/n0 u^(gamma-1)]
              + (n_n - n_p)^2 [a1/n0 + gamma1 b1/n0 u^(gamma1-1)]

    The closed form on the second line is what is evaluated; the identity
    between the two is checked in `verify/`. Only the power-law pieces pick up
    the extra gamma, gamma1 -- that difference from `interaction_energy` is
    the whole numerical content of "V is a functional of the densities and
    nothing else", and it is what makes the Euler relation hold at the
    physical potentials with no rearrangement term left over.
    """
    n_B = n_p + n_n
    if n_B < 1e-15:
        return 0.0

    n0 = params.n0
    u = n_B / n0
    gamma, gamma1 = params.gamma, params.gamma1
    a0, b0, a1, b1 = params.a0, params.b0, params.a1, params.b1

    cross = 4.0 * n_p * n_n * (a0 / n0 + gamma * b0 / n0 * u**(gamma - 1))
    isovector = (n_n - n_p)**2 * (a1 / n0 + gamma1 * b1 / n0 * u**(gamma1 - 1))
    return cross + isovector


def interaction_potentials(n_p: float, n_n: float,
                           params: Parameters) -> Tuple[float, float]:
    """(mu_Hv_p, mu_Hv_n), the interaction part of each chemical potential.

    mu_Hv_i = dV/dn_i at the other density fixed. Differentiating
    `interaction_energy`, with du/dn_i = 1/n0,

        mu_Hv_p = 4 n_n [a0/n0 + b0/n0 u^(g-1)]
                - 2 (n_n-n_p) [a1/n0 + b1/n0 u^(g1-1)]
                + 4 b0 n_p n_n (g-1) u^(g-2) / n0^2
                + b1 (n_n-n_p)^2 (g1-1) u^(g1-2) / n0^2

    and mu_Hv_n is the same with n_p <-> n_n in the first term and the sign of
    the second reversed. The last two terms come from differentiating the
    powers of u = n_B/n0 and are therefore COMMON to both species, since u
    depends on the densities only through their sum.
    """
    n_B = n_p + n_n
    if n_B < 1e-15:
        return 0.0, 0.0

    n0 = params.n0
    u = n_B / n0
    gamma, gamma1 = params.gamma, params.gamma1
    a0, b0, a1, b1 = params.a0, params.b0, params.a1, params.b1

    cross_p = 4.0 * n_n * (a0/n0 + b0/n0 * u**(gamma - 1))
    cross_n = 4.0 * n_p * (a0/n0 + b0/n0 * u**(gamma - 1))
    isovector = 2.0 * (n_n - n_p) * (a1/n0 + b1/n0 * u**(gamma1 - 1))
    # The two terms below come from d/dn_i of the powers of u = n_B/n0, so
    # they are the SAME for both species.
    du_cross = 4.0 * b0 * n_n * n_p * u**(gamma - 2) * (gamma - 1) / n0**2
    du_isovector = b1 * (n_n - n_p)**2 * u**(gamma1 - 2) * (gamma1 - 1) / n0**2

    mu_Hv_p = cross_p + -isovector + du_cross + du_isovector
    mu_Hv_n = cross_n + isovector + du_cross + du_isovector
    return mu_Hv_p, mu_Hv_n


def effective_potentials(mu_p: float, mu_n: float, n_p: float, n_n: float,
                         params: Parameters) -> Tuple[float, float]:
    """(mu_eff_p, mu_eff_n) = (mu_i - mu_Hv_i), the potentials the gases see.

    This is the ZL counterpart of subtracting the vector self-energy in a
    mean-field model: mu_eff_i is what enters `kinetic_thermo`, and the model
    is solved when n_i(mu_eff_i, T, m_i) reproduces the n_i that generated
    mu_Hv_i.
    """
    mu_Hv_p, mu_Hv_n = interaction_potentials(n_p, n_n, params)
    return mu_p - mu_Hv_p, mu_n - mu_Hv_n


@dataclass
class EffectiveState:
    """The one evaluation a residual needs: potentials in, densities out.

    Carries the state it was evaluated at (`mu_i`, `n_i`), the effective
    potentials it implies, and the densities those produce (`n_i_calc`). The
    model is self-consistent where `n_i_calc == n_i`.
    """
    # State: physical chemical potentials (MeV) and densities (fm^-3)
    mu_p: float = 0.0
    mu_n: float = 0.0
    n_p: float = 0.0
    n_n: float = 0.0
    # Effective (kinetic) chemical potentials (MeV)
    mu_eff_p: float = 0.0
    mu_eff_n: float = 0.0
    # Densities the effective potentials produce (fm^-3)
    n_p_calc: float = 0.0
    n_n_calc: float = 0.0

    @property
    def n_B(self) -> float:
        """Baryon density of the state, n_B = n_p + n_n."""
        return self.n_p + self.n_n

    @property
    def n_C(self) -> float:
        """Non-leptonic charge density, n_C = n_p (leptons excluded)."""
        return self.n_p

    @property
    def n_S(self) -> float:
        """Strangeness density: identically zero, ZL carries no strangeness."""
        return 0.0

    @property
    def mu_B(self) -> float:
        """Baryon chemical potential, mu_B = mu_n."""
        return self.mu_n

    @property
    def mu_C(self) -> float:
        """Charge chemical potential, mu_C = mu_p - mu_n."""
        return self.mu_p - self.mu_n


def effective_state(mu_p: float, mu_n: float, n_p: float, n_n: float,
                    T: float, params: Parameters) -> EffectiveState:
    """Evaluate the self-consistency condition at one state.

    Takes the physical potentials and the densities that source the
    interaction, and returns both the effective potentials and the densities
    those produce. Every mode's residual is built on the two differences
    `n_i_calc - n_i`.
    """
    mu_eff_p, mu_eff_n = effective_potentials(mu_p, mu_n, n_p, n_n, params)
    return EffectiveState(
        mu_p=mu_p, mu_n=mu_n, n_p=n_p, n_n=n_n,
        mu_eff_p=mu_eff_p, mu_eff_n=mu_eff_n,
        n_p_calc=kinetic_thermo(mu_eff_p, T, params.m_p).n,
        n_n_calc=kinetic_thermo(mu_eff_n, T, params.m_n).n,
    )


# =============================================================================
# THE BLOCK: SUMS OVER THE SPECIES
# =============================================================================
@dataclass
class MatterThermo:
    """The strongly-interacting sector at one state: densities, charges, P,
    eps, s and the conserved-charge potentials. Leptons and photons are added
    on top by the mode, not here.
    """
    n_p: float = 0.0       # proton density (fm^-3)
    n_n: float = 0.0       # neutron density (fm^-3)
    Y_p: float = 0.0       # proton fraction n_p/n_B
    Y_n: float = 0.0       # neutron fraction n_n/n_B
    T: float = 0.0         # temperature (MeV)
    mu_p: float = 0.0      # proton chemical potential (MeV)
    mu_n: float = 0.0      # neutron chemical potential (MeV)
    P: float = 0.0         # pressure (MeV/fm^3)
    e: float = 0.0         # energy density (MeV/fm^3)
    s: float = 0.0         # entropy density (fm^-3)
    f: float = 0.0         # free energy density f = e - Ts (MeV/fm^3)
    Y_C: float = 0.0       # non-leptonic charge fraction
    Y_S: float = 0.0       # strangeness fraction (identically zero)
    n_B: float = 0.0       # baryon density (fm^-3)
    n_C: float = 0.0       # non-leptonic charge density (fm^-3)
    n_S: float = 0.0       # strangeness density (fm^-3)
    mu_B: float = 0.0      # baryon chemical potential (MeV)
    mu_C: float = 0.0      # charge chemical potential (MeV)
    mu_S: float = 0.0      # strangeness potential: zero by convention


def thermo_from_mu_n(mu_p: float, mu_n: float, n_p: float, n_n: float,
                     T: float, params: Parameters = None) -> MatterThermo:
    """The matter block at given potentials AND densities -- no solve.

    This is the evaluation layer: it does not enforce the self-consistency
    n_i = n_i(mu_eff_i), it evaluates everything at the state handed to it,
    and reports the densities the Fermi integrals actually returned. A
    residual calls it on every iteration; `thermo_from_mu` is the same thing
    with the fixed point solved for.

    Assembles

        P   = P_p + P_n + P_int         eps = eps_p + eps_n + V
        s   = s_p + s_n                 f   = eps - T s

    with the kinetic parts from `kinetic_thermo` at mu_eff_i and the
    interaction from `interaction_pressure` / `interaction_energy`. The
    conserved charges follow the repository basis: n_B = n_p + n_n,
    n_C = n_p (leptons excluded), n_S = 0, and mu_B = mu_n,
    mu_C = mu_p - mu_n, so beta equilibrium reads mu_C + mu_e = 0.

    The Euler relation eps + P = T s + sum_i mu_i n_i holds identically at the
    PHYSICAL potentials, because sum_i n_i mu_Hv_i = V + P_int.
    """
    if params is None:
        params = Parameters.default()

    mu_eff_p, mu_eff_n = effective_potentials(mu_p, mu_n, n_p, n_n, params)

    thermo_p = kinetic_thermo(mu_eff_p, T, params.m_p)
    thermo_n = kinetic_thermo(mu_eff_n, T, params.m_n)

    n_p_calc, n_n_calc = thermo_p.n, thermo_n.n
    n_B = n_p_calc + n_n_calc
    n_C = n_p_calc
    n_S = 0.0

    P = thermo_p.P + thermo_n.P + interaction_pressure(n_p_calc, n_n_calc,
                                                       params)
    e = thermo_p.e + thermo_n.e + interaction_energy(n_p_calc, n_n_calc,
                                                     params)
    s = thermo_p.s + thermo_n.s          # V carries no T: entropy is kinetic

    return MatterThermo(
        n_p=n_p_calc, n_n=n_n_calc, n_B=n_B, n_C=n_C, n_S=n_S,
        Y_p=n_p_calc / n_B, Y_n=n_n_calc / n_B, Y_C=n_C / n_B, Y_S=n_S / n_B,
        T=T,
        mu_p=mu_p, mu_n=mu_n, mu_B=mu_n, mu_C=mu_p - mu_n, mu_S=0.0,
        P=P, e=e, s=s, f=e - s * T,
    )


def thermo_from_mu(mu_p: float, mu_n: float, T: float,
                   params: Parameters = None) -> MatterThermo:
    """The matter block at given chemical potentials, solving the state.

    The densities are not known in advance from (mu_p, mu_n, T), because the
    interaction potentials depend on them; the fixed point

        n_i = n_i(mu_i - mu_Hv_i(n_p, n_n), T, m_i)

    is solved here for the pair (n_p, n_n), starting from the free-gas
    densities at the physical potentials. This is the self-consistent layer of
    the thermodynamics -- a solve, but of nothing mode-dependent, and it is
    the surface a phase-adapter would consume.
    """
    from scipy.optimize import root

    if params is None:
        params = Parameters.default()

    def self_consistency(x):
        n_p, n_n = x
        state = effective_state(mu_p, mu_n, n_p, n_n, T, params)
        return [state.n_p_calc - n_p, state.n_n_calc - n_n]

    guess_p = kinetic_thermo(mu_p, T, params.m_p).n
    guess_n = kinetic_thermo(mu_n, T, params.m_n).n
    x0 = [max(guess_p, 1e-6), max(guess_n, 1e-6)]

    sol = root(self_consistency, x0, method='hybr')
    if not sol.success:
        sol = root(self_consistency, x0, method='lm')

    n_p, n_n = sol.x
    return thermo_from_mu_n(mu_p, mu_n, n_p, n_n, T, params)


def thermo_from_n(n_n: float, n_p: float, T: float,
                  params: Parameters = None) -> MatterThermo:
    """The matter block at given (n_n, n_p, T) -- the inverse direction.

    With the densities given, no fixed point is needed: they determine the
    interaction potentials outright, the Fermi integrals are inverted for
    mu_eff_i, and the physical potentials follow as mu_i = mu_eff_i + mu_Hv_i.

    The species densities, not (n_B, Y_C): the two are the same information
    for nucleons, but Y_C is the fraction a MODE holds, and this module never
    knows which mode it is in (CLAUDE.md section 5). A caller that has
    (n_B, Y_C) passes n_p = Y_C n_B and n_n = (1 - Y_C) n_B.
    """
    if params is None:
        params = Parameters.default()

    mu_eff_p = invert_fermi_density(n_p, T, params.m_p, G_NUCLEON)
    mu_eff_n = invert_fermi_density(n_n, T, params.m_n, G_NUCLEON)

    mu_Hv_p, mu_Hv_n = interaction_potentials(n_p, n_n, params)
    return thermo_from_mu_n(mu_eff_p + mu_Hv_p, mu_eff_n + mu_Hv_n,
                            n_p, n_n, T, params)
