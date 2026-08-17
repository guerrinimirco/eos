"""
mixed/adapters.py
=================
The phase-adapter contract: the ONE surface through which this engine touches
a bulk equation of state. Everything the mixed-phase solve knows about DD2 and
about vMIT enters here and nowhere else, so pairing a different hadronic or
quark model is writing an adapter, not editing the solver.

An adapter is a function

    adapter(<the phase's conserved-charge potentials>, T, ...) -> PhaseThermo

that solves that phase's OWN internal self-consistency — meson fields for a
relativistic mean field, flavor densities for a vector bag — at the given
potentials, and reports the block in `PhaseThermo`: the species densities, the
conserved-charge densities n_B, n_C, n_S, the bulk P, eps and s, and the
conserved-charge potentials mu_B, mu_C, mu_S.

Both adapters report the SAME conserved-charge decomposition,

    mu_i = B_i mu_B + C_i mu_C + S_i mu_S

so the mixed-phase solver can match potentials across phases without knowing
which engine produced them. There is deliberately NO eta, NO mixing and NO
charge neutrality here: those are conditions on the *pair* of phases and belong
to the mixed residual. An adapter is a description of one phase in isolation.

An adapter must be a *deterministic function of its arguments*: the same
potentials must give the same block, to the last digit, however the point was
reached. The mixed residual is differentiated by finite differences, so an
adapter that remembers the previous trial point would corrupt the Jacobian.
This is why the optional warm start `x0` is an explicit argument rather than
hidden state.

Units on the boundary are fm-based (fm^-3, MeV/fm^3, MeV). The hadronic side
converts DD2's internal natural units through hc3; the quark side is already
fm-based.

Warm starting matters here more than anywhere else in the engine. The outer
Newton iteration calls these adapters once per residual evaluation, so a cold
seed — in particular re-running a full beta-equilibrium solve just to get a
starting field configuration — dominates the cost. Both adapters accept an
`x0`, and `MixedCtx` caches the converged internal state between calls.
"""
from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np
from scipy.optimize import root

from eos.general.physics_constants import hc3
from eos.dd2.thermodynamics import (
    baryon_kinetics, self_consistency_residual, thermo_at_potentials,
)
from eos.dd2.thermodynamics import thermal_meson_thermo
from eos.dd2.solver import solve_beta_eq_octet
from eos.vmit.parameters import get_vmit_default
from eos.vmit.thermodynamics import (
    compute_quark_matter_thermo_from_mu, compute_vmit_thermo_from_mu_n,
)

#: Post-solve residual gate for a phase-internal solve. Matches the tolerance
#: eos/dd2/solver.py accepts its own equilibrium solves at.
RESIDUAL_TOL = 1.0e-10


@dataclass(frozen=True)
class PhaseThermo:
    """One phase's thermodynamic block, fm-based.

    densities : {species name -> n [fm^-3]}
    n_B, n_C, n_S : baryon / non-leptonic-charge / strangeness density [fm^-3]
    P, eps : pressure / energy density [MeV/fm^3]  (matter only, no leptons)
    s : entropy density [fm^-3]
    mu_B, mu_C, mu_S : conserved-charge potentials [MeV]
    mu_i : {species name -> mu [MeV]}, mu_i = B_i mu_B + C_i mu_C + S_i mu_S
    mu_dot_n : sum_i mu_i n_i [MeV/fm^3], including any thermal meson gas
    condensation : max_j |mu*_j| / m_j over this phase's thermal meson gas,
        0 without one. Part of the contract because a phase that has
        Bose-condensed is OUTSIDE its model, and the mixed residual cannot see
        that any other way: `solve_bose_jel` caps mu at m rather than
        diverging, so a condensed phase reports a perfectly converged block.
        A quark phase has no meson gas and leaves it at 0.
    """
    densities: Mapping[str, float]
    n_B: float
    n_C: float
    n_S: float
    P: float
    eps: float
    s: float
    mu_B: float
    mu_C: float
    mu_S: float
    mu_i: Mapping[str, float]
    mu_dot_n: float = 0.0
    condensation: float = 0.0


# =============================================================================
# QUARK PHASE  (vMIT)
# =============================================================================
def quark_phase(mu_u, mu_d, mu_s, T=0.0, params=None):
    """vMIT quark phase at fixed physical quark chemical potentials.

    The vMIT vector interaction makes mu_eff = mu - V(n) density-dependent, so
    the flavor densities are found by a small root solve
    (`compute_quark_matter_thermo_from_mu`) before the full block is assembled.
    Quark sector only: no leptons, no photons.

    Charge decomposition (vMIT's convention, shared with the hadronic side):
    mu_B = mu_u + 2 mu_d, mu_C = mu_u - mu_d, mu_S = mu_s - mu_d.

    # ponytail: no warm start. The quark solve is ~7% of a mixed-point solve
    # (three unknowns, seeded at the free-gas limit and converging in a few
    # steps), and adding one would mean reaching into the frozen eos/vmit
    # engine. Revisit only if a profile says the quark side has become hot.
    """
    if params is None:
        params = get_vmit_default()
    n_u, n_d, n_s, _P, _e, _s, _nB = compute_quark_matter_thermo_from_mu(
        mu_u, mu_d, mu_s, T, params)
    th = compute_vmit_thermo_from_mu_n(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, params)
    return PhaseThermo(
        densities={"u": th.n_u, "d": th.n_d, "s": th.n_s},
        n_B=th.n_B, n_C=th.n_C, n_S=th.n_S,
        P=th.P, eps=th.e, s=th.s,
        mu_B=th.mu_B, mu_C=th.mu_C, mu_S=th.mu_S,
        mu_i={"u": mu_u, "d": mu_d, "s": mu_s},
        mu_dot_n=mu_u * th.n_u + mu_d * th.n_d + mu_s * th.n_s,
)


# =============================================================================
# HADRONIC PHASE  (DD2)
# =============================================================================
def hadronic_seed(par, flags, T, n_B_guess):
    """Starting field configuration for the phase-internal solve: the fields of
    a charge-neutral beta-equilibrium DD2 solve at `n_B_guess`.

    Expensive (a full DD2 solve) but guaranteed physical, m* > 0, which a
    linearized estimate is not at high density.

    Note what this does *not* depend on: the charge potentials. Within one
    mixed-phase solve, where only those potentials vary, the seed is a
    constant — so compute it once and pass it back in through `hadronic_phase`'s
    `x0` rather than rebuilding it on every residual evaluation. Doing so is
    what makes the mixed solve affordable, and because the seed is identical
    every time, it changes no converged number.

    The thermal meson gas is switched OFF for the seed, and must be. The gas
    sources none of the four field equations -- it rides on them -- so it
    changes nothing this function returns except whether it can return at all:
    `eos.dd2` REFUSES a Bose-condensed gas by raising, and in beta equilibrium
    the gas is condensed above n_B ~ 0.3 fm^-3 at these temperatures, so a
    seed built with the flags passed through simply raised. That made the
    hadronic adapter unusable with the gas on, and it hid the condensation
    check downstream, which never got the chance to run.

    Returns [sigma, omega0, rho0, (phi0), nB_nat]; the density is in natural
    units, the rest in MeV.
    """
    fields_only = replace(flags, include_pseudoscalars=False,
                          include_thermal_vectors=False)
    base = solve_beta_eq_octet(par, n_B_guess, fields_only, T=T,
                               include_photons=False, check_consistency=False)
    x = [base.sigma, base.omega0, base.rho0]
    if flags.phi_field and flags.hyperons:
        x.append(base.phi0 if base.phi0 != 0.0 else -1.0e-3)
    x.append(n_B_guess * hc3)
    return x


def hadronic_phase(par, flags, mu_tilde_B, mu_C, mu_S=0.0, T=0.0,
                   n_B_guess=0.2, x0=None, return_state=False):
    """DD2 hadronic phase at fixed kinetic charge potentials.

    Inputs are the KINETIC baryon potential `mu_tilde_B = mu_B - Sigma^R` (which
    keeps the rearrangement self-energy and its density circularity out of the
    iteration — see CLAUDE.md §2), the non-leptonic charge potential `mu_C`
    (= mu_C) and the strangeness potential `mu_S`.

    A thin call to `eos.dd2.thermodynamics.thermo_at_potentials`, which solves
    the DD-RMF meson fields and the phase's own baryon density
    self-consistently. dd2 owns that solve, because it is dd2's own
    self-consistency and nothing about it depends on a mode; this adapter's job
    is only to present the result on the phase-adapter contract. Until that
    commit the residual, its seed and its gate lived here as a SECOND
    implementation of dd2's field equations, free to drift from the first.
    Returns an fm-based `PhaseThermo` with the physical
    `mu_B = mu_tilde_B + Sigma^R` restored at assembly.

    If the species flags enable a thermal meson gas, it is added at the
    converged potentials and fields exactly as `eos/dd2/solver.solve_octet`
    does: an additive ideal Bose gas contributing P, eps, s and its own
    mu*_j n_j to the chemical-potential sum.

    The gas is NOT a spectator to the charge bookkeeping. A thermal pi/K gas
    carries net electric charge and strangeness, so it enters n_C and n_S --
    and hence neutrality, the fixed-Y_C condition and the mixed-phase charge
    residuals -- not only eps, P and s. At T = 40 MeV the pion gas carries about
    15% of the non-leptonic charge, so treating it as a spectator is not a
    small error.

    `x0` is the starting field configuration; pass the (constant, per-solve)
    result of `hadronic_seed` rather than letting this rebuild it every call.
    It must be a *deterministic* function of the caller's fixed inputs — seeding
    from the previous trial point's converged state instead makes this function
    non-reproducible at a given set of potentials, which in turn corrupts any
    finite-difference Jacobian taken of the mixed residual.

    `return_state=True` also returns the converged internal vector, so the
    Jacobian can differentiate the phase without re-solving it.
    """
    # dd2 seeds itself from its own field equations, which keeps a mode out of
    # its thermodynamics but is weaker than a solved starting point. Here the
    # stronger seed is available and this is the hot path, so supply it: a
    # charge-neutral beta-equilibrium solve at n_B_guess, which is physical by
    # construction and independent of the charge potentials.
    state, internal = thermo_at_potentials(
        par, flags, mu_tilde_B, mu_C, mu_S, T=T, n_B_guess=n_B_guess,
        x0=x0,
        x0_fallback=lambda: hadronic_seed(par, flags, T, n_B_guess),
        return_state=True)
    th = PhaseThermo(
        densities=dict(state.densities), n_B=state.n_B, n_C=state.n_C,
        n_S=state.n_S, P=state.P, eps=state.eps, s=state.s,
        mu_B=state.mu_B, mu_C=state.mu_C, mu_S=state.mu_S,
        mu_i=dict(state.mu_i), mu_dot_n=state.mu_dot_n,
        condensation=state.condensation)
    return (th, internal) if return_state else th
