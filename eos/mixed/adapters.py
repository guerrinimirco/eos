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
import numpy as np
from scipy.optimize import root

from eos.general.physics_constants import hc, hc3
from eos.general.state import PhaseThermo
from eos.dd2.thermodynamics import thermo_at_potentials
from eos.dd2.solver import solve_beta_eq_octet
from eos.vmit.parameters import get_vmit_default
from eos.vmit.thermodynamics import (
    compute_quark_matter_thermo_from_mu, compute_vmit_thermo_from_mu_n,
)

#: Post-solve residual gate for a phase-internal solve. Matches the tolerance
#: eos/dd2/solver.py accepts its own equilibrium solves at.
RESIDUAL_TOL = 1.0e-10


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
    # The single universal vector shift V = a hc (n_u + n_d + n_s); it is what
    # separates the physical potentials from the effective ones.
    V = params.a * hc * (th.n_u + th.n_d + th.n_s)
    return PhaseThermo(
        T=T,
        densities={"u": th.n_u, "d": th.n_d, "s": th.n_s},
        fields={"V": V},
        n_B=th.n_B, n_C=th.n_C, n_S=th.n_S,
        P=th.P, eps=th.e, s=th.s,
        mu_B=th.mu_B, mu_C=th.mu_C, mu_S=th.mu_S,
        mu_i={"u": mu_u, "d": mu_d, "s": mu_s},
        mu_eff_i={"u": mu_u - V, "d": mu_d - V, "s": mu_s - V},
        m_eff_i={"u": params.m_u, "d": params.m_d, "s": params.m_s},
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
    # dd2's thermodynamics already returns the shared eos.general.state
    # record, so present it as-is: same floats, no re-packaging.
    return (state, internal) if return_state else state


# =============================================================================
# ENJL PHASES  (one functional, three branches)
# =============================================================================
# ENJL is not a second model to pair with a first: it is ONE thermodynamic
# potential whose local equations admit three self-consistent solutions at the
# same potentials -- chirally broken hadronic, chirally restored but still
# confined (baryons AND quarks), and fully deconfined. A first-order transition
# in this model is therefore a construction between two BRANCHES of one
# functional, and the pairing is two adapters over one model rather than one
# adapter over each of two.
#
# `eos.enjl` knows nothing about any of this. It exposes `thermo_from_mu`, the
# block at given potentials; the branch vocabulary lives here, on the composite
# engine's side of the layering, exactly as the DD2 and vMIT adapters above do.

#: The three branches, as declared values. A construction pairs any two; which
#: pairs are realized is a property of the parameter set, not of this list.
#: Recorded in docs/enjl/PHASE_TRANSITION_DESIGN.md section 1b: the author's own
#: worksheet never reaches the deconfined branch, because it is a continuation
#: started on the hadronic side -- the branch is unvisited, not excluded.
ENJL_BRANCHES = ("broken", "restored", "deconfined")

#: Constituent mass above which a light flavour counts as chirally broken. The
#: gap equation returns M_q = m_q0 exactly once nbar^s_q is capped, and the
#: broken branch sits at 100-370 MeV, so anything in between separates them.
_CHIRAL_SPLIT = 50.0

#: Baryon fraction below which a state counts as fully deconfined. The author's
#: tables reach ~1e-5 fm^-3 of residual baryons past deconfinement, so an
#: absolute floor would be parameter-dependent; this is relative to n_B.
_DECONFINED_BARYON_FRACTION = 1.0e-4


def enjl_branch_of(point):
    """Which branch a converged `eos.enjl` point is on: the post-solve check.

    Reads the state, not the seed. `enjl_phase` calls this on what came back
    and raises if it is not the branch that was asked for, so a solve that
    slid to a neighbouring root says so instead of returning it quietly.
    """
    light = min(point.M_q["u"], point.M_q["d"])
    if light > _CHIRAL_SPLIT:
        return "broken"
    n_b = point.n_b
    baryons = point.n["p"] + point.n["n"] + point.n["Lambda"]
    if n_b > 0.0 and baryons <= _DECONFINED_BARYON_FRACTION * n_b:
        return "deconfined"
    return "restored"


def enjl_branch_seed(par, branch, mu_B, mu_C=0.0, mu_S=0.0, T=0.0):
    """Starting points for `branch`, as a PURE function of the arguments.

    Returns a list of nine-vectors (M_u, M_d, M_s, n_B, n_B^Q, g_omega omega,
    g_rho rho, Sigma^R_b, Sigma^R_q) in natural units, to be tried in order.

    Nothing here may depend on a previously solved point. That is a stronger
    requirement for ENJL than for DD2: a mixed residual is differentiated by
    finite differences, and here the seed does not merely set how fast the
    solve converges, IT CHOOSES WHICH ROOT IT CONVERGES TO. An adapter that
    warm-started from the previous trial point would change branch partway
    through a Jacobian column and return a matrix that is the derivative of
    nothing. Measured, at the f_q = 0.7, B = 1 coexistence potentials: a broken
    seed gives n_B = 0.517 fm^-3 and a restored one 0.555 fm^-3 from the same
    arguments.

    What varies between branches is the pair (quark masses, quark baryon
    density): the broken branch starts from the vacuum constituent masses with
    no quarks, the restored branch from the current masses with a fifth of the
    baryon density already in quarks, and the deconfined branch from the
    current masses with ALL of it in quarks, which is what puts the baryon
    Fermi momenta below threshold and keeps them there.

    The density ladder is a spread rather than a single value because the map
    mu_B -> n_B is not one-to-one -- that is the whole reason this function
    exists -- so no closed-form estimate can be right on every branch at once.
    Trying several is still a pure function of the arguments; remembering one
    would not be.
    """
    if T != 0.0:
        raise NotImplementedError(
            f"eos.enjl is a T = 0 model; got T = {T} MeV")
    if branch not in ENJL_BRANCHES:
        raise ValueError(
            f"unknown ENJL branch {branch!r}; expected one of {ENJL_BRANCHES}")

    from eos.enjl.thermodynamics import VACUUM_GUESS

    if branch == "broken":
        masses, quark_fraction = VACUUM_GUESS, 0.0
    else:
        masses = (par.m_u0, par.m_d0, par.m_s0 + 100.0)
        quark_fraction = 1.0 if branch == "deconfined" else 0.2

    # A spread of densities either side of a linear-in-mu_B estimate, which is
    # roughly right above 1 GeV and only has to land in the right basin below.
    scale = max(0.05, abs(mu_B) / 1900.0)
    seeds = []
    for factor in (1.0, 0.35, 2.5):
        n_B0 = scale * factor * hc3
        seeds.append([masses[0], masses[1], masses[2], n_B0,
                      quark_fraction * n_B0,
                      par.Gamma_w(n_B0) * 3.0 * n_B0, 0.0, 0.0, 0.0])
    return seeds


def enjl_phase(par, branch, mu_B, mu_C=0.0, mu_S=0.0, T=0.0):
    """One ENJL branch at fixed conserved-charge potentials -> `PhaseThermo`.

    The phase-adapter contract of CLAUDE.md section 5, with `branch` promoted
    to a declared argument. Everything the mixed-phase solve knows about ENJL
    enters here.

    Takes the PHYSICAL mu_B, unlike `hadronic_phase` above, which takes DD2's
    kinetic mu_B - Sigma^R. ENJL can, because it carries Sigma^R_b and
    Sigma^R_q as unknowns with their defining equations as residual rows, so
    the circularity that makes dd2 prefer the kinetic potential is inside its
    residual rather than around it. Nothing is subtracted or restored on this
    boundary; validated by round-tripping solved beta-equilibrium states
    through (mu_B, mu_C) and back, which returns n_B to 1e-15 relative, and
    against the author's own tables, which it reproduces in n_B and in
    P (matter + leptons) to 3e-7 relative over 0.1-8 fm^-3.

    Quark matter and hadronic matter are the same functional here, so this one
    function serves both sides of a construction; the two adapters of a pair
    differ only in the branch they declare. The density-dependent couplings
    alpha_S, Gamma_omega and Gamma_rho, and the rearrangement terms built from
    them, are evaluated at THIS phase's own n_B, which is the convention
    `hadronic_phase` already follows for DD2, itself a density-dependent RMF.

    No leptons, no neutrality, no held fraction: those are conditions on a
    system and this describes matter.

    Raises RuntimeError if no seed converges, or if the converged state is not
    on `branch` -- a silent hop is the one failure this contract cannot absorb.
    """
    from eos.enjl.thermodynamics import thermo_from_mu

    if T != 0.0:
        raise NotImplementedError(
            f"eos.enjl is a T = 0 model; got T = {T} MeV")

    last = None
    for x0 in enjl_branch_seed(par, branch, mu_B, mu_C, mu_S, T=T):
        try:
            point = thermo_from_mu(par, mu_B, mu_C, mu_S, T=T, x0=x0)
        except RuntimeError as exc:                   # this seed missed
            last = exc
            continue
        if enjl_branch_of(point) == branch:
            return enjl_phase_thermo(point, mu_B, mu_C, mu_S)
    if last is not None:
        raise RuntimeError(
            f"ENJL {branch} branch did not converge at mu_B={mu_B:.3f}, "
            f"mu_C={mu_C:.3f}, mu_S={mu_S:.3f} MeV: {last}")
    raise RuntimeError(
        f"ENJL solve at mu_B={mu_B:.3f}, mu_C={mu_C:.3f}, mu_S={mu_S:.3f} MeV "
        f"converged but never onto the {branch!r} branch")


def enjl_phase_thermo(point, mu_B, mu_C, mu_S):
    """An `eos.enjl` point, in natural units, as an fm-based `PhaseThermo`."""
    n = {sp: value / hc3 for sp, value in point.n.items()}
    mu_i = {sp: point.mu[sp] for sp in point.n}
    m_eff = dict(point.M_b, **point.M_q)
    # T = 0, so the effective (kinetic) potential is the Fermi energy.
    mu_eff = {sp: float(np.sqrt(point.kF[sp] ** 2 + m_eff[sp] ** 2))
              for sp in point.kF if sp in m_eff}
    return PhaseThermo(
        T=0.0,
        densities=n,
        fields={"gomega_omega": point.gomega_omega,
                "grho_rho": point.grho_rho,
                "SigmaR_b": point.SigmaR_b, "SigmaR_q": point.SigmaR_q},
        n_B=point.n_b / hc3, n_C=point.n_C / hc3, n_S=point.n_S / hc3,
        P=point.P / hc3, eps=point.eps / hc3, s=point.s / hc3,
        mu_B=mu_B, mu_C=mu_C, mu_S=mu_S,
        mu_i=mu_i, mu_eff_i=mu_eff, m_eff_i=m_eff,
        mu_dot_n=sum(mu_i[sp] * n[sp] for sp in n),
        condensation=0.0)
