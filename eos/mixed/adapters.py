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
from eos.general.basis import quark_charges, quark_potentials
from eos.general.state import PhaseThermo
from eos.dd2.species import SpeciesFlags as DD2SpeciesFlags
from eos.dd2.thermodynamics import thermo_from_mu as _dd2_at_mu
from eos.dd2.solver import solve_beta_eq_neutrinoless, solve, sweep
from eos.mixed.charges import Regime
from eos.vmit.parameters import Parameters as VMITParameters
from eos.vmit.species import SpeciesFlags as VMITFlags
from eos.vmit.thermodynamics import (
    thermo_from_mu as _vmit_from_mu, thermo_from_n as _vmit_from_n,
    thermo_from_mu_n as _vmit_from_mu_n,
)

#: Post-solve residual gate for a phase-internal solve. Matches the tolerance
#: eos/dd2/solver.py accepts its own equilibrium solves at.
RESIDUAL_TOL = 1.0e-10

#: The flags every vMIT solve behind this adapter is made with. Photons are
#: phase-common and are counted once at the mixture level
#: (`eos.mixed.species`), so the phase contributes matter only: the cold start
#: discards P, eps and s outright -- it reads potentials -- and the wing
#: agrees with the mixture's own all-False default. `vmit_phase` takes no
#: caller flags, so unlike `dd2_phase` its wing cannot follow one; the same
#: holds for `zl_phase` and `alphabag_phase`.
_VMIT_MATTER_ONLY = VMITFlags(photons=False)


@dataclass(frozen=True)
class Phase:
    """One phase of a pairing, as the solver consumes it.

    A `Phase` bundles the adapter callable with everything the engine must
    KNOW about it but must not ASSUME: which flavour of baryon potential its
    slot carries, how it may be seeded, and which optional capabilities it
    provides. Each factory below closes over its model's parameters, so
    parameters remain arguments (CLAUDE.md section 6) and the engine never
    sees a model type.

    The engine labels the two phases of a pair POSITIONALLY as H and Q — the
    historical hadronic/quark roles, generally the low- and high-density
    phase — in its slot names, result fields (`th_H`, `th_Q`) and lepton
    domains. `name` is the human label used in messages and reported branch
    labels; for a same-model pairing (two ENJL branches) it is what tells the
    two apart.

    thermo(mu, mu_C, mu_S, T, n_B_guess=None, x0=None, return_state=False)
        The phase's block at given conserved-charge potentials, as a shared
        `PhaseThermo`. `mu` is the slot potential — kinetic or physical per
        `potential_kind`. `n_B_guess` and `x0` are seeding hints an adapter
        may ignore; with `return_state=True` it returns `(block, state)`,
        where `state` is an opaque internal vector (or None) that a Jacobian
        may differentiate without re-solving.
    potential_kind : "kinetic" | "physical"
        What the baryon slot carries. DD2 wants the kinetic
        mu_tilde_B = mu_B - Sigma^R because its rearrangement term depends on
        the density it helps determine; models that carry Sigma^R inside
        their own residual (ENJL) or have none (SFHo, vMIT) take the
        physical potential. The residual's baryon-matching row reads the
        PHYSICAL potential from either kind, so the two may be mixed freely.
    seed(T, n_B_guess) -> x0
        The adapter-internal starting configuration, a pure function of its
        arguments. `seed_cacheable=False` forbids the solver's per-solve
        memoization: for a branch-declared adapter the seed CHOOSES THE ROOT,
        so caching would change physics, not speed (the ENJL rule).
    cold_start(n_B, T) -> (mu_slot, mu_e, mu_B_physical)
        A physical starting point for the mixed unknown vector, from the
        phase's own equilibrium at density n_B. May raise; the solver then
        falls back to the partner phase's physical values. None means the
        phase cannot cold-start and a warm start must be supplied.
    supports_S : bool
        False for a strangeness-free model (ZL): a spec that conserves S
        globally, or any nonzero mu_S, raises before a solve.
    max_T : float or None
        Highest temperature the adapter supports; None is unlimited. 0.0 for
        the T = 0 ENJL surface: T > 0 raises before a solve.
    wing_sweep(spec, n_B_grid, T) -> [(n_B, P, eps), ...]
        The PURE phase swept at the spec's own equilibrium, for the hybrid
        table's wings. None: `hybrid_table` raises naming the phase.
    frozen_thermo(th, scale, T) -> (P, eps)
        This phase compressed by `scale` at frozen composition — the
        responses capability (see `eos.mixed.responses` for the convention).
        None: `sound_speed_frozen` raises naming the phase.
    jacobian_block(mu, mu_C, mu_S, T, state, th) -> ndarray
        Analytic d(n_B, n_C, n_S, P[, mu_B])/d(mu, mu_C, mu_S) block, rows
        [n_B, n_C, n_S, P] plus a fifth mu_B row for a kinetic-kind phase.
        Optional by design (the handoff rule): absent means the solver uses
        its numeric Jacobian, which is the reference path anyway.
    """
    name: str
    thermo: object
    potential_kind: str
    seed: object = None
    seed_cacheable: bool = True
    cold_start: object = None
    supports_S: bool = True
    max_T: float = None
    wing_sweep: object = None
    frozen_thermo: object = None
    jacobian_block: object = None

    def __post_init__(self):
        if self.potential_kind not in ("kinetic", "physical"):
            raise ValueError(f"potential_kind must be 'kinetic' or "
                             f"'physical', got {self.potential_kind!r}")

    def slot(self, position):
        """This phase's baryon-potential slot name at positional label
        `position` ('H' or 'Q'): the prefix says what the slot carries."""
        prefix = "mu_tilde_B_" if self.potential_kind == "kinetic" else "mu_B_"
        return f"{prefix}{position}"


# =============================================================================
# QUARK PHASE  (vMIT)
# =============================================================================
def quark_phase(mu_u, mu_d, mu_s, T=0.0, params=None):
    """vMIT quark phase at fixed physical quark chemical potentials.

    The vMIT vector interaction makes mu_eff = mu - V(n) density-dependent, so
    the flavor densities are found by a small root solve
    (`eos.vmit.thermodynamics.thermo_from_mu`) before the full block is
    assembled.
    Quark sector only: no leptons, no photons.

    Charge decomposition (vMIT's convention, shared with the hadronic side):
    mu_B = mu_u + 2 mu_d, mu_C = mu_u - mu_d, mu_S = mu_s - mu_d.

    # ponytail: no warm start. The quark solve is ~7% of a mixed-point solve
    # (three unknowns, seeded at the free-gas limit and converging in a few
    # steps), and adding one would mean reaching into the frozen eos/vmit
    # engine. Revisit only if a profile says the quark side has become hot.
    """
    if params is None:
        params = VMITParameters.default()
    n_u, n_d, n_s, _P, _e, _s, _nB = _vmit_from_mu(mu_u, mu_d, mu_s, T,
                                                   params)
    th = _vmit_from_mu_n(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, params)
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
    fields_only = replace(flags, thermal_mesons=False,
                          thermal_vectors=False, photons=False)
    base = solve_beta_eq_neutrinoless(par, n_B_guess, fields_only, T=T,
                               check_consistency=False)
    fields = base.matter.fields
    x = [fields["sigma"], fields["omega0"], fields["rho0"]]
    if flags.hyperons and par.has_phi_coupling:
        x.append(fields["phi0"] if fields["phi0"] != 0.0 else -1.0e-3)
    x.append(n_B_guess * hc3)
    return x


def hadronic_phase(par, flags, mu_tilde_B, mu_C, mu_S=0.0, T=0.0,
                   n_B_guess=0.2, x0=None, return_state=False):
    """DD2 hadronic phase at fixed kinetic charge potentials.

    Inputs are the KINETIC baryon potential `mu_tilde_B = mu_B - Sigma^R` (which
    keeps the rearrangement self-energy and its density circularity out of the
    iteration — see CLAUDE.md §2), the non-leptonic charge potential `mu_C`
    (= mu_C) and the strangeness potential `mu_S`.

    A thin call to `eos.dd2.thermodynamics.thermo_from_mu`, which solves
    the DD-RMF meson fields and the phase's own baryon density
    self-consistently. dd2 owns that solve, because it is dd2's own
    self-consistency and nothing about it depends on a mode; this adapter's job
    is only to present the result on the phase-adapter contract. Until that
    commit the residual, its seed and its gate lived here as a SECOND
    implementation of dd2's field equations, free to drift from the first.
    Returns an fm-based `PhaseThermo` with the physical
    `mu_B = mu_tilde_B + Sigma^R` restored at assembly.

    If the species flags enable a thermal meson gas, it is added at the
    converged potentials and fields exactly as `eos/dd2/solver.solve`
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
    state, internal = _dd2_at_mu(
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
    # The effective (kinetic) potential is carried by the point itself. It was
    # rebuilt here from kF while `eos.enjl` was a T = 0 model, where the two
    # agree; at T > 0 there is no sharp Fermi surface and only nu is defined.
    mu_eff = {sp: float(point.nu[sp]) for sp in point.nu if sp in m_eff}
    return PhaseThermo(
        T=point.T,
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


def _dd2_wing_kwargs(spec, flags):
    """`sweep` mode arguments for the DD2 wing, from the spec.

    The dispatch reads the spec's regimes, not a mode name, so any regime
    combination the window can solve gets a consistent wing. Beta equilibrium
    reduces to exactly the call `sweep` makes.
    """
    if spec.C is Regime.NOT_CONSERVED:                  # beta equilibrium
        kw = dict(charge_mode="neutral")
        if spec.L_e is Regime.GLOBAL:                   # trapped neutrinos
            if not flags.neutrinos:
                raise ValueError(
                    "a trapped-neutrino hybrid needs "
                    "SpeciesFlags(neutrinos=True): the hadronic wing solves "
                    "at fixed Y_Le and must carry the neutrino population")
            kw.update(lepton_mode="trapped", Y_Le=spec.targets["Y_Le"])
        return kw
    kw = dict(charge_mode="fixed", Y_C=spec.targets["Y_C"],
              yc_leptons=spec.yc_leptons)
    if spec.S is Regime.GLOBAL:
        kw.update(strange_mode="fixed", Y_S=spec.targets["Y_S"])
    return kw


def _vmit_wing_solve(spec, n_B, T, params):
    """One pure vMIT point at the spec's equilibrium.

    There is no naming difference left to absorb: vmit names the condition
    Y_Le, as the engine and CLAUDE.md section 5 do. Each point cold-starts
    from vmit's own default guess
    — those solves are cheap and robust, and a cold start keeps every wing
    row exactly reproducible by the pure model's own call at the same
    conditions, which is what test/mixed/test_hybrid_modes.py asserts.
    """
    from eos.vmit.solver import (
        solve_beta_eq_neutrinoless as _vmit_beta,
        solve_fixed_yc as _vmit_yc,
        solve_fixed_yc_ys as _vmit_yc_ys,
        solve_beta_eq_neutrino_trapped as _vmit_trapped,
    )
    if spec.C is Regime.NOT_CONSERVED:                  # beta equilibrium
        if spec.L_e is Regime.GLOBAL:
            return _vmit_trapped(params, n_B, spec.targets["Y_Le"], T,
                                 _VMIT_MATTER_ONLY)
        return _vmit_beta(params, n_B, T, _VMIT_MATTER_ONLY)
    if spec.S is Regime.GLOBAL:
        return _vmit_yc_ys(params, n_B, spec.targets["Y_C"],
                           spec.targets["Y_S"], T, _VMIT_MATTER_ONLY,
                           leptons=spec.yc_leptons)
    return _vmit_yc(params, n_B, spec.targets["Y_C"], T, _VMIT_MATTER_ONLY,
                    leptons=spec.yc_leptons)



def _dd2_frozen_block(par, flags, n_B, Y_C, Y_S, T, x0=None):
    """(P, eps, n_C) of DD2 matter at `n_B` with Y_C and Y_S held.

    Matter only: no leptons, no photons. `x0` is the octet unknown vector to
    start from and is what keeps this solvable deep in a mixed phase.
    """
    # strange_mode='fixed' only when strangeness can actually vary; for
    # nucleonic matter it would add an inert unknown to the solve.
    strange = "fixed" if flags.has_strange_baryons else "eq"
    p = solve(par, n_B, replace(flags, photons=False), T=T, x0=x0,
                    charge_mode="fixed", Y_C=Y_C,
                    strange_mode=strange, Y_S=Y_S, yc_leptons=False,
                    check_consistency=False)
    return p.P, p.eps, Y_C * n_B


def _vmit_frozen_block(params, n_u, n_d, n_s, T):
    """(P, eps, n_C) of vMIT matter at the given flavour densities.

    Rescaling the flavour densities freezes the quark composition exactly — no
    re-solve, and no chance of the solve drifting to another root.
    """
    _mu_u, _mu_d, _mu_s, P, eps, _s, _n_B = _vmit_from_n(n_u, n_d, n_s, T,
                                                         params)
    return P, eps, quark_charges(n_u, n_d, n_s)[1]


# =============================================================================
# THE SHIPPED PAIRINGS
# =============================================================================
# Each factory closes over one model's parameters and returns a `Phase`.
# A pairing is any two of them, and every one of them is written the same
# way; `default_pair` names the DD2 + vMIT one for convenience only.

def dd2_phase(par, flags):
    """The DD2 hadronic phase as a `Phase` (kinetic potential slot)."""
    if getattr(flags, "sigma_star", False):
        raise NotImplementedError(
            "SpeciesFlags.sigma_star (hidden-strange scalar) is not wired in "
            "the hadronic phase")

    def thermo(mu, mu_C, mu_S, T, n_B_guess=None, x0=None,
               return_state=False):
        return hadronic_phase(par, flags, mu, mu_C, mu_S, T=T,
                              n_B_guess=(0.2 if n_B_guess is None
                                         else n_B_guess),
                              x0=x0, return_state=return_state)

    def seed(T, n_B_guess):
        return hadronic_seed(par, flags, T, n_B_guess)

    def cold_start(n_B, T):
        # A charge-neutral beta-equilibrium solve: physical by construction
        # and independent of the charge potentials. The meson gas is switched
        # off for the same reason `hadronic_seed` switches it off.
        seed_flags = replace(flags, thermal_mesons=False,
                             thermal_vectors=False, photons=False)
        base = solve_beta_eq_neutrinoless(par, n_B, seed_flags, T=T,
                                   check_consistency=False)
        m = base.matter
        return m.mu_B - m.Sigma_R, base.leptons.mu_e, m.mu_B

    def wing_sweep(spec, n_B_grid, T):
        # The pure DD2 wing at the spec's own equilibrium, warm-started with
        # dd2's own continuation; a point past the scalar-collapse boundary
        # ends the sweep rather than raising.
        kw = _dd2_wing_kwargs(spec, flags)
        return [(p.n_B, p.P, p.eps)
                for p in sweep(par, n_B_grid, flags, T=T,
                                     stop_at_boundary=True, **kw)]

    def frozen_thermo(th, scale, T, mu_slot=None):
        # This phase compressed by `scale` with its own Y_C and Y_S held (the
        # frozen convention of eos.mixed.responses). The octet solve is
        # seeded from the phase's state at its own slot potential; without
        # that the fallback seeds from nucleonic beta equilibrium, which a
        # mixed phase's hadronic component is nowhere near, and roughly one
        # point in six comes back nan. The seed is deterministic, so both
        # stencil points of a derivative receive the same vector.
        x0 = None
        if mu_slot is not None and th.n_B > 0.0:
            try:
                _th, state = hadronic_phase(par, flags, mu_slot, th.mu_C,
                                            th.mu_S, T=T, n_B_guess=th.n_B,
                                            return_state=True)
                x0 = list(state["x_phase"][:-1]) + [mu_slot, th.mu_C]
                if flags.has_strange_baryons:
                    x0.append(th.mu_S)
            except RuntimeError:
                x0 = None
        return _dd2_frozen_block(par, flags, th.n_B * scale,
                                 th.n_C / th.n_B, th.n_S / th.n_B, T, x0=x0)

    try:                       # backends/ is deletable; absent means numeric
        from eos.mixed.backends.jacobian import _hadronic_block
        jac = (lambda mu, mu_C, mu_S, T, state, th:
               _hadronic_block(par, flags, mu, mu_C, mu_S, T, state))
    except ImportError:
        jac = None

    return Phase(name="DD2", thermo=thermo, potential_kind="kinetic",
                 seed=seed, cold_start=cold_start, wing_sweep=wing_sweep,
                 frozen_thermo=frozen_thermo, jacobian_block=jac)


def vmit_phase(params=None):
    """The vMIT quark phase as a `Phase` (physical potential slot).

    The (mu_B, mu_C, mu_S) -> (mu_u, mu_d, mu_s) rotation happens HERE, so
    the engine hands every adapter the same conserved-charge potentials and
    no flavour basis leaks past this closure.
    """
    if params is None:
        params = VMITParameters.default()

    def thermo(mu, mu_C, mu_S, T, n_B_guess=None, x0=None,
               return_state=False):
        mu_u, mu_d, mu_s = quark_potentials(mu, mu_C, mu_S)
        th = quark_phase(mu_u, mu_d, mu_s, T=T, params=params)
        return (th, None) if return_state else th

    def cold_start(n_B, T):
        from eos.vmit.solver import solve_beta_eq_neutrinoless as _vmit_beta
        # Potentials only: P, eps and s are discarded, so the photon gas
        # would be dead weight even if the mixture did not count it itself.
        q = _vmit_beta(params, n_B, T, _VMIT_MATTER_ONLY)
        return q.mu_B, q.mu_e, q.mu_B

    def wing_sweep(spec, n_B_grid, T):
        # The pure vMIT wing at the spec's own equilibrium. A point that
        # fails or does not converge is a hole, matching the window sweep.
        out = []
        for n in n_B_grid:
            try:
                q = _vmit_wing_solve(spec, float(n), T, params)
            except Exception:
                continue
            if not q.converged:
                continue
            out.append((q.n_B, q.P_total, q.e_total))
        return out

    def frozen_thermo(th, scale, T, mu_slot=None):
        # Rescaling the flavour densities freezes the composition exactly.
        return _vmit_frozen_block(params, th.densities["u"] * scale,
                                  th.densities["d"] * scale,
                                  th.densities["s"] * scale, T)

    try:
        from eos.mixed.backends.jacobian import _quark_block
        jac = (lambda mu, mu_C, mu_S, T, state, th:
               _quark_block(mu, mu_C, mu_S, T, params, th))
    except ImportError:
        jac = None

    return Phase(name="vMIT", thermo=thermo, potential_kind="physical",
                 cold_start=cold_start, wing_sweep=wing_sweep,
                 frozen_thermo=frozen_thermo, jacobian_block=jac)


def default_pair(par, flags=None, vmit_params=None):
    """The DD2 + vMIT pairing, as a `Phase` pair.

    A named convenience for ONE pairing, and nothing more: it occupies no
    position in any signature that `(sfho_phase(...), njl_phase(...))` cannot
    occupy equally, which is the whole of what makes DD2 and vMIT ordinary
    here. `flags` is DD2's own `SpeciesFlags` (the hadronic phase's per-model
    sectors); the mixture's phase-common ones are the engine's `species=`
    argument, not this.
    """
    return dd2_phase(par, DD2SpeciesFlags() if flags is None else flags), \
           vmit_phase(vmit_params)


def sfho_phase(par, flags):
    """The SFHo hadronic phase as a `Phase` (physical potential slot).

    SFHo's couplings are constants — no rearrangement term — so the slot
    carries the physical mu_B and its adapter surface is
    `eos.sfho.thermodynamics.thermo_from_mu` (four meson fields solved,
    the densities following outright from the potentials and fields).
    """
    from eos.sfho.solver import (
        solve_beta_eq_neutrinoless as _sfho_beta,
        solve_beta_eq_neutrino_trapped as _sfho_trapped,
        solve_fixed_yc as _sfho_yc,
        solve_fixed_yc_ys as _sfho_yc_ys,
    )
    from eos.sfho.thermodynamics import thermo_from_mu as _sfho_at_mu

    def thermo(mu, mu_C, mu_S, T, n_B_guess=None, x0=None,
               return_state=False):
        return _sfho_at_mu(par, flags, mu, mu_C, mu_S, T=T,
                           n_B_guess=(0.2 if n_B_guess is None
                                      else n_B_guess),
                           x0=x0, return_state=return_state)

    def cold_start(n_B, T):
        p = _sfho_beta(par, n_B, flags, T=T)
        if not p.converged:
            raise RuntimeError(f"sfho cold start failed at n_B={n_B}")
        return p.matter.mu_B, p.leptons.mu_e, p.matter.mu_B

    def _wing_point(spec, n_B, T):
        # ponytail: cold start per point; sfho's density-scaled default guess
        # converges on wing-like states. Warm-start via sfho's own
        # warm_start(point, spec) if a wing ever shows holes.
        if spec.C is Regime.NOT_CONSERVED:
            if spec.L_e is Regime.GLOBAL:
                return _sfho_trapped(par, n_B, spec.targets["Y_Le"], flags,
                                     T=T)
            return _sfho_beta(par, n_B, flags, T=T)
        if spec.S is Regime.GLOBAL:
            return _sfho_yc_ys(par, n_B, spec.targets["Y_C"],
                               spec.targets["Y_S"], flags, T=T,
                               leptons=spec.yc_leptons)
        return _sfho_yc(par, n_B, spec.targets["Y_C"], flags, T=T,
                        leptons=spec.yc_leptons)

    def wing_sweep(spec, n_B_grid, T):
        out = []
        for n in n_B_grid:
            try:
                p = _wing_point(spec, float(n), T)
            except Exception:
                continue
            if not p.converged:
                continue
            out.append((p.n_B, p.P, p.eps))
        return out

    def frozen_thermo(th, scale, T, mu_slot=None):
        # Y_C and Y_S held while the species re-equilibrate within them,
        # matter only — the same convention as the DD2 block.
        n = th.n_B * scale
        if flags.hyperons:
            p = _sfho_yc_ys(par, n, th.n_C / th.n_B, th.n_S / th.n_B, flags,
                            T=T, leptons=False)
        else:
            p = _sfho_yc(par, n, th.n_C / th.n_B, flags, T=T, leptons=False)
        if not p.converged:
            raise RuntimeError(f"sfho frozen block failed at n_B={n}")
        return p.P, p.eps, (th.n_C / th.n_B) * n

    return Phase(name="SFHo", thermo=thermo, potential_kind="physical",
                 cold_start=cold_start, wing_sweep=wing_sweep,
                 frozen_thermo=frozen_thermo)


def did_seed(par, flags, T, n_B_guess):
    """Starting state for the DID phase-internal solve: a solved beta-equilibrium
    point at `n_B_guess`, in the layout `did.thermodynamics` iterates on.

    Expensive (a full DID solve) but guaranteed physical, and -- this is the
    point -- INDEPENDENT of the charge potentials, which are the only thing
    that varies within one mixed-phase solve. So it is computed once and
    passed back through `x0`, and because it is identical every time it
    changes no converged number.

    The thermal meson gas is switched off for the seed, as it is for DD2: the
    gas sources none of the field equations, so it changes nothing here except
    whether the seed can be built at all.
    """
    from eos.did.solver import solve_beta_eq_neutrinoless as _did_beta

    seed_flags = replace(flags, thermal_mesons=False)
    point = _did_beta(par, n_B_guess, seed_flags, T=T)
    if not point.converged:
        raise RuntimeError(
            f"DID seed failed at n_B={n_B_guess}, T={T} "
            f"(residual {point.error:.2e})")
    return [point.sigma, point.omega, point.rho, point.phi, point.beta,
            point.Sigma_t, point.n_B]


def did_phase(par, flags):
    """The DID hadronic phase as a `Phase` (kinetic potential slot).

    DID's couplings depend on the density AND on the isospin asymmetry, so its
    phase-internal solve closes seven equations rather than four: the meson
    fields, the phase's own density, its asymmetry beta and the isospin
    rearrangement self-energy Sigma^t. All of that is `eos.did`'s own business
    and stays inside `eos.did.thermodynamics.thermo_from_mu`; what
    reaches the engine is the same `PhaseThermo` every other adapter returns.

    The slot carries the KINETIC potential mu~_B = mu_B - Sigma^r, as DD2's
    does and for the same reason: the density rearrangement term is a function
    of the density this solve is finding. The SECOND rearrangement term is not
    absorbed that way -- it is weighted by (tau_3i - beta) and so differs per
    species -- and stays inside the phase, which is exactly what the adapter
    contract allows a phase to keep to itself.
    """
    from eos.did.solver import (
        solve_beta_eq_neutrinoless as _did_beta,
        solve_beta_eq_neutrino_trapped as _did_trapped,
        solve_fixed_yc as _did_yc,
        solve_fixed_yc_ys as _did_yc_ys,
    )
    from eos.did.thermodynamics import thermo_from_mu as _did_at_mu

    def thermo(mu, mu_C, mu_S, T, n_B_guess=None, x0=None,
               return_state=False):
        return _did_at_mu(par, flags, mu, mu_C, mu_S, T=T,
                          n_B_guess=(0.2 if n_B_guess is None else n_B_guess),
                          x0=x0,
                          x0_fallback=lambda: did_seed(par, flags, T,
                                                       0.2 if n_B_guess is None
                                                       else n_B_guess),
                          return_state=return_state)

    def seed(T, n_B_guess):
        return did_seed(par, flags, T, n_B_guess)

    def cold_start(n_B, T):
        point = _did_beta(par, n_B, flags, T=T)
        if not point.converged:
            raise RuntimeError(f"DID cold start failed at n_B={n_B}, T={T}")
        return point.mu_B - point.Sigma_r, point.mu_e, point.mu_B

    def _wing_point(spec, n_B, T, x0=None):
        if spec.C is Regime.NOT_CONSERVED:
            if spec.L_e is Regime.GLOBAL:
                return _did_trapped(par, n_B, spec.targets["Y_Le"], flags, T=T,
                                    x0=x0)
            return _did_beta(par, n_B, flags, T=T, x0=x0)
        if spec.S is Regime.GLOBAL:
            return _did_yc_ys(par, n_B, spec.targets["Y_C"],
                              spec.targets["Y_S"], flags, T=T,
                              leptons=spec.yc_leptons, x0=x0)
        return _did_yc(par, n_B, spec.targets["Y_C"], flags, T=T,
                       leptons=spec.yc_leptons, x0=x0)

    def wing_sweep(spec, n_B_grid, T):
        # The pure DID wing at the spec's own equilibrium. Warm-started along
        # the grid, because the wing runs through the hyperon onsets, where a
        # cold start at one density lands on the branch below the threshold.
        out, x0 = [], None
        for n in n_B_grid:
            try:
                point = _wing_point(spec, float(n), T, x0=x0)
            except (RuntimeError, ValueError):
                x0 = None
                continue
            if not point.converged:
                x0 = None
                continue
            out.append((point.n_B, point.P, point.eps))
            x0 = _did_warm(point, spec, flags)
        return out

    def frozen_thermo(th, scale, T, mu_slot=None):
        # This phase compressed by `scale` with its own Y_C and Y_S held (the
        # frozen convention of eos.mixed.responses), matter only. Seeded from
        # the phase's own state so the two stencil points of a derivative
        # start from the same deterministic vector.
        n_B = th.n_B * scale
        Y_C, Y_S = th.n_C / th.n_B, th.n_S / th.n_B
        if flags.hyperons:
            point = _did_yc_ys(par, n_B, Y_C, Y_S, flags, T=T, leptons=False)
        else:
            point = _did_yc(par, n_B, Y_C, flags, T=T, leptons=False)
        if not point.converged:
            raise RuntimeError(f"DID frozen block failed at n_B={n_B}")
        return point.P, point.eps, Y_C * n_B

    return Phase(name="DID", thermo=thermo, potential_kind="kinetic",
                 seed=seed, cold_start=cold_start, wing_sweep=wing_sweep,
                 frozen_thermo=frozen_thermo)


def _did_warm(point, spec, flags):
    """The DID warm start for a wing sweep, in the mode the spec declares."""
    from eos.did.solver import warm_start as _warm
    from eos.general.modes import (
        beta_eq_neutrinoless, beta_eq_neutrino_trapped, fixed_YC, fixed_YC_YS,
    )
    if spec.C is Regime.NOT_CONSERVED:
        mode = (beta_eq_neutrino_trapped(spec.targets["Y_Le"])
                if spec.L_e is Regime.GLOBAL else beta_eq_neutrinoless())
    elif spec.S is Regime.GLOBAL:
        mode = fixed_YC_YS(spec.targets["Y_C"], spec.targets["Y_S"],
                           leptons=spec.yc_leptons)
    else:
        mode = fixed_YC(spec.targets["Y_C"], leptons=spec.yc_leptons)
    return _warm(point, mode)


def zl_phase(params=None):
    """The ZL nucleonic phase as a `Phase` (physical potential slot).

    The species-basis rotation happens here — mu_p = mu_B + mu_C,
    mu_n = mu_B — and ZL carries no strangeness: `supports_S=False`, so any
    mode that conserves S globally raises before a solve, and mu_S never
    reaches the model.
    """
    from eos.zl.parameters import Parameters as ZLParameters
    from eos.zl.species import SpeciesFlags as ZLFlags
    from eos.zl.thermodynamics import thermo_from_mu as _zl_from_mu
    from eos.zl.thermodynamics import thermo_from_n as _zl_from_n
    from eos.zl.solver import (
        solve_beta_eq_neutrinoless as _zl_beta,
        solve_beta_eq_neutrino_trapped as _zl_trapped,
        solve_fixed_yc as _zl_yc,
    )
    if params is None:
        params = ZLParameters.default()
    # Photons are phase-common and are counted once at the mixture level
    # (`eos.mixed.species`), so the phase contributes matter only. The cold
    # start discards P, eps and s outright -- it reads potentials -- and the
    # wing agrees with the mixture's own all-False default. This phase takes
    # no caller flags, so unlike `dd2_phase` its wing cannot follow one; see
    # the note in `eos/mixed/species.py`.
    flags = ZLFlags(photons=False)

    def _as_record(m, mu_p, mu_n, T):
        return PhaseThermo(
            T=T, mu_B=m.mu_B, mu_C=m.mu_C, mu_S=0.0,
            # ZL's interaction potentials are functions of (n_p, n_n) and are
            # not carried on its MatterThermo; nothing downstream reads them.
            fields={},
            densities={"p": m.n_p, "n": m.n_n},
            mu_i={"p": mu_p, "n": mu_n},
            mu_eff_i={}, m_eff_i={},
            n_B=m.n_B, n_C=m.n_C, n_S=0.0,
            P=m.P, eps=m.e, s=m.s,
            mu_dot_n=mu_p * m.n_p + mu_n * m.n_n)

    def thermo(mu, mu_C, mu_S, T, n_B_guess=None, x0=None,
               return_state=False):
        mu_p, mu_n = mu + mu_C, mu
        m = _zl_from_mu(mu_p, mu_n, T, params=params)
        th = _as_record(m, mu_p, mu_n, T)
        return (th, None) if return_state else th

    def cold_start(n_B, T):
        p = _zl_beta(params, n_B, flags, T)
        if not p.converged:
            raise RuntimeError(f"zl cold start failed at n_B={n_B}")
        return p.mu_B, p.mu_e, p.mu_B

    def _wing_point(spec, n_B, T):
        if spec.C is Regime.NOT_CONSERVED:
            if spec.L_e is Regime.GLOBAL:
                return _zl_trapped(params, n_B, spec.targets["Y_Le"],
                                   flags, T)
            return _zl_beta(params, n_B, flags, T)
        return _zl_yc(params, n_B, spec.targets["Y_C"], flags, T,
                      leptons=spec.yc_leptons)

    def wing_sweep(spec, n_B_grid, T):
        out = []
        for n in n_B_grid:
            try:
                p = _wing_point(spec, float(n), T)
            except Exception:
                continue
            if not p.converged:
                continue
            out.append((p.n_B, p.P_total, p.e_total))
        return out

    def frozen_thermo(th, scale, T, mu_slot=None):
        # Two species: holding Y_C freezes the composition exactly, and the
        # inverse direction needs no fixed point at all.
        Y_C = th.n_C / th.n_B
        n_B = th.n_B * scale
        m = _zl_from_n((1.0 - Y_C) * n_B, Y_C * n_B, T, params=params)
        return m.P, m.e, Y_C * n_B

    return Phase(name="ZL", thermo=thermo, potential_kind="physical",
                 cold_start=cold_start, supports_S=False,
                 wing_sweep=wing_sweep, frozen_thermo=frozen_thermo)


def alphabag_phase(params=None):
    """The alphaBag quark phase as a `Phase` (physical potential slot).

    `eos.alphabag.thermodynamics.thermo_from_mu` is a pure evaluation — the
    perturbative-QCD-corrected couplings are explicit in mu, so there is no
    internal solve at all. The flavour rotation happens here, as in
    `vmit_phase`. No `frozen_thermo`: alphabag exposes no
    thermo-at-given-densities surface, so the frozen-composition responses
    raise for a pairing that includes it (docs/DEFERRED.md).
    """
    from eos.alphabag.parameters import Parameters as ABParameters
    from eos.alphabag.species import SpeciesFlags as ABFlags
    from eos.alphabag.thermodynamics import thermo_from_mu as _ab_from_mu
    from eos.alphabag.solver import (
        solve_beta_eq_neutrinoless as _ab_beta,
        solve_beta_eq_neutrino_trapped as _ab_trapped,
        solve_fixed_yc as _ab_yc,
        solve_fixed_yc_ys as _ab_yc_ys,
    )
    if params is None:
        params = ABParameters.default()
    # Matter only, for the reason `_VMIT_MATTER_ONLY` states: the photon gas
    # is phase-common and is counted once at the mixture level, the cold start
    # discards P, eps and s outright, and the wing agrees with the mixture's
    # own all-False default. The gluon gas and the thermal neutrino gases go
    # with it -- both are phase-common thermal sectors in the same sense, and
    # this phase takes no caller flags to follow. See `eos/mixed/species.py`.
    flags = ABFlags()

    def thermo(mu, mu_C, mu_S, T, n_B_guess=None, x0=None,
               return_state=False):
        mu_u, mu_d, mu_s = quark_potentials(mu, mu_C, mu_S)
        m = _ab_from_mu(mu_u, mu_d, mu_s, T, params)
        th = PhaseThermo(
            T=T, mu_B=m.mu_B, mu_C=m.mu_C, mu_S=m.mu_S,
            fields={},                     # couplings explicit in mu: no field
            densities={"u": m.n_u, "d": m.n_d, "s": m.n_s},
            mu_i={"u": mu_u, "d": mu_d, "s": mu_s},
            mu_eff_i={}, m_eff_i={},
            n_B=m.n_B, n_C=m.n_C, n_S=m.n_S,
            P=m.P, eps=m.e, s=m.s,
            mu_dot_n=mu_u * m.n_u + mu_d * m.n_d + mu_s * m.n_s)
        return (th, None) if return_state else th

    def cold_start(n_B, T):
        p = _ab_beta(params, n_B, T, flags)
        if not p.converged:
            raise RuntimeError(f"alphabag cold start failed at n_B={n_B}")
        return p.mu_B, p.mu_e, p.mu_B

    def _wing_point(spec, n_B, T):
        if spec.C is Regime.NOT_CONSERVED:
            if spec.L_e is Regime.GLOBAL:
                return _ab_trapped(params, n_B, spec.targets["Y_Le"], T,
                                   flags)
            return _ab_beta(params, n_B, T, flags)
        if spec.S is Regime.GLOBAL:
            return _ab_yc_ys(params, n_B, spec.targets["Y_C"],
                             spec.targets["Y_S"], T, flags,
                             leptons=spec.yc_leptons)
        return _ab_yc(params, n_B, spec.targets["Y_C"], T, flags,
                      leptons=spec.yc_leptons)

    def wing_sweep(spec, n_B_grid, T):
        out = []
        for n in n_B_grid:
            try:
                p = _wing_point(spec, float(n), T)
            except Exception:
                continue
            if not p.converged:
                continue
            out.append((p.n_B, p.P_total, p.e_total))
        return out

    return Phase(name="alphaBag", thermo=thermo, potential_kind="physical",
                 cold_start=cold_start, wing_sweep=wing_sweep)


def njl_phase(par, flags=None, patterns=None):
    """The three-flavour NJL quark phase as a `Phase` (physical potential slot).

    Two things make this adapter different from every other quark one here.

    COLOUR NEUTRALITY IS CLOSED INSIDE IT. A colour-superconducting phase must
    carry mu_3 and mu_8 to be colour neutral, and those are not conserved
    charges of the mixed system -- no hadronic phase has them, and there is
    nothing across the interface for them to equilibrate with. So they are
    solved within the phase, by `eos.njl.thermodynamics.thermo_from_mu`, and
    the engine never learns they exist. That is exactly what the phase-adapter
    contract asks for: "solving the phase's own internal self-consistency at
    those fixed potentials".

    THE SEED CHOOSES THE ROOT, so `seed_cacheable=False` -- the ENJL rule. The
    gap equation has three roots at any Fermi-surface mismatch, so a cached
    seed would not merely change how fast a point is reached but which state
    is reached. The adapter enumerates the pairing patterns at every call and
    keeps the one with the largest pressure, which at fixed potentials is the
    stable one; the winner's label rides on the returned block's `fields`
    alongside the gaps and the colour potentials, since a mixed table that
    does not say which quark phase it found is not reporting its own result.

    `frozen_thermo` is absent: NJL exposes no thermo-at-given-densities
    surface, so the frozen-composition responses raise for a pairing that
    includes it (docs/DEFERRED.md).
    """
    from eos.njl.parameters import Parameters as NJLParameters
    from eos.njl.species import DEFAULT_PATTERNS, SpeciesFlags as NJLFlags
    from eos.njl.thermodynamics import thermo_from_mu, vacuum_solution
    from eos.njl.solver import (
        solve_beta_eq_neutrinoless as _njl_beta,
        solve_beta_eq_neutrino_trapped as _njl_trapped,
        solve_fixed_yc as _njl_yc,
        solve_fixed_yc_ys as _njl_yc_ys,
    )
    if par is None:
        par = NJLParameters.default()
    if flags is None:
        flags = NJLFlags()
    if patterns is None:
        patterns = DEFAULT_PATTERNS if flags.csc else ("unpaired",)
    vac = vacuum_solution(par)

    def _block(st):
        n = st.n_flavour / hc3
        mu_u, mu_d, mu_s = quark_potentials(st.mu_B, st.mu_C, st.mu_S)
        fields = {"M_u": st.M[0], "M_d": st.M[1], "M_s": st.M[2],
                  "Delta_1": st.Delta[0], "Delta_2": st.Delta[1],
                  "Delta_3": st.Delta[2], "mu_3": st.mu_3, "mu_8": st.mu_8,
                  "Sigma_V": st.Sigma_V}
        return PhaseThermo(
            T=st.T, mu_B=st.mu_B, mu_C=st.mu_C, mu_S=st.mu_S, fields=fields,
            densities={"u": n[0], "d": n[1], "s": n[2]},
            mu_i={"u": mu_u, "d": mu_d, "s": mu_s},
            mu_eff_i={"u": mu_u - st.Sigma_V, "d": mu_d - st.Sigma_V,
                      "s": mu_s - st.Sigma_V},
            m_eff_i={"u": st.M[0], "d": st.M[1], "s": st.M[2]},
            n_B=st.n_B_fm, n_C=st.n_C_fm, n_S=st.n_S_fm,
            P=st.P_fm, eps=st.eps_fm, s=st.s_fm,
            mu_dot_n=st.mu_dot_n / hc3)

    def thermo(mu, mu_C, mu_S, T, n_B_guess=None, x0=None,
               return_state=False):
        seeds = dict(x0) if isinstance(x0, dict) else {}
        best = best_state = None
        for pattern in patterns:
            st, ok, _ = thermo_from_mu(par, mu, mu_C, mu_S, T,
                                       pattern=pattern,
                                       x0=seeds.get(pattern), vac=vac)
            if not ok:
                continue
            if best is None or st.P > best.P:
                best, best_state = st, {pattern: _internal_vector(st, par,
                                                                  pattern)}
        if best is None:
            raise RuntimeError(
                f"eos.njl: no pairing pattern converged at mu_B={mu:g}, "
                f"mu_C={mu_C:g}, mu_S={mu_S:g}, T={T:g} MeV")
        th = _block(best)
        return (th, best_state) if return_state else th

    def cold_start(n_B, T):
        p = _njl_beta(par, n_B, T, flags=flags, patterns=patterns)
        if not p.converged:
            raise RuntimeError(f"eos.njl cold start failed at n_B={n_B}")
        return p.mu_B, p.mu_e, p.mu_B

    def _wing_point(spec, n_B, T):
        if spec.C is Regime.NOT_CONSERVED:
            if spec.L_e is Regime.GLOBAL:
                return _njl_trapped(par, n_B, spec.targets["Y_Le"], T,
                                    flags=flags, patterns=patterns)
            return _njl_beta(par, n_B, T, flags=flags, patterns=patterns)
        if spec.S is Regime.GLOBAL:
            return _njl_yc_ys(par, n_B, spec.targets["Y_C"],
                              spec.targets["Y_S"], T,
                              flags=flags, leptons=spec.yc_leptons,
                              patterns=patterns)
        return _njl_yc(par, n_B, spec.targets["Y_C"], T, flags=flags,
                       leptons=spec.yc_leptons, patterns=patterns)

    def wing_sweep(spec, n_B_grid, T):
        out = []
        for n in n_B_grid:
            try:
                p = _wing_point(spec, float(n), T)
            except Exception:
                continue
            if p.converged:
                out.append((p.n_B, p.P_total, p.e_total))
        return out

    return Phase(name="NJL", thermo=thermo, potential_kind="physical",
                 seed_cacheable=False, cold_start=cold_start,
                 wing_sweep=wing_sweep)


def _internal_vector(st, par, pattern):
    """The internal unknown vector of a solved NJL state, for a warm start.

    The layout is `eos.njl.thermodynamics.internal_unknowns`, rebuilt from the
    state rather than carried out of the solve, so the adapter stays a pure
    function of its arguments -- the mixed residual is finite-differenced, and
    an adapter that remembered its previous trial point would corrupt the
    Jacobian.
    """
    from eos.njl.species import pattern_mask
    from eos.njl.thermodynamics import has_vector
    mask = pattern_mask(pattern)
    vector = list(st.M)
    vector += [st.Delta[eta] for eta in range(3) if mask[eta]]
    if any(mask):
        vector += [st.mu_3, st.mu_8]
    if has_vector(par):
        vector.append(st.Sigma_V)
    return np.array(vector, dtype=float)


def ccdm_phase(par, flags=None, branches=None, patterns=None):
    """The chiral colour-dielectric quark phase as a `Phase` (physical slot).

    Three things make this adapter different from the other quark ones here,
    and all three are properties of the model rather than of the engine.

    IT ENUMERATES TWO THINGS, NOT ONE. `eos.njl` chooses a pairing pattern by
    pressure; this model chooses a chiral/dielectric BRANCH as well, and the
    two cannot be chosen one after the other because which pattern survives
    depends on the strange quark's effective mass, which is a property of the
    branch. So the adapter walks the product and keeps the largest pressure,
    which at fixed potentials is the stable state. The winners ride on the
    returned block's `fields`, since a mixed table that does not say which
    quark phase it found is not reporting its own result.

    THE CONFINED BRANCH IS ENUMERATED HERE AND NOT AT FIXED DENSITY. Its
    pressure is exactly zero -- the dielectric has closed and there are no
    quarks -- so it is not a state a hybrid construction wants returned, but
    it IS what the deconfined branch has to beat, and the crossing is the
    deconfinement onset. It is therefore excluded by default and available by
    passing `branches=('confined', ...)` to locate that onset explicitly.

    COLOUR NEUTRALITY IS CLOSED INSIDE IT, exactly as in `njl_phase`: mu_3 and
    mu_8 are not conserved charges of the mixed system -- no hadronic phase
    has them and there is nothing across the interface for them to
    equilibrate with -- so `eos.ccdm.thermodynamics.thermo_from_mu` solves
    them within the pattern and the engine never learns they exist.

    THE SEED CHOOSES THE ROOT, so `seed_cacheable=False` -- the ENJL rule, and
    doubly so here: the seed picks the chiral branch as well as the gap root,
    so a cached seed would not merely change how fast a point is reached but
    which state is reached.

    `frozen_thermo` is absent: CCDM exposes no thermo-at-given-densities
    surface, so the frozen-composition responses raise for a pairing that
    includes it (docs/DEFERRED.md).
    """
    from eos.ccdm.parameters import Parameters as CCDMParameters
    from eos.ccdm.species import (
        DEFAULT_PATTERNS, DENSITY_BRANCHES, SpeciesFlags as CCDMFlags,
    )
    from eos.ccdm.thermodynamics import thermo_from_mu
    from eos.ccdm.solver import (
        solve_beta_eq_neutrinoless as _ccdm_beta,
        solve_beta_eq_neutrino_trapped as _ccdm_trapped,
        solve_fixed_yc as _ccdm_yc,
        solve_fixed_yc_ys as _ccdm_yc_ys,
    )
    if par is None:
        par = CCDMParameters.default()
    if flags is None:
        flags = CCDMFlags()
    if branches is None:
        branches = DENSITY_BRANCHES
    if patterns is None:
        patterns = DEFAULT_PATTERNS if flags.csc else ("unpaired",)

    def _block(st):
        n = st.n_flavour / hc3
        mu_u, mu_d, mu_s = quark_potentials(st.mu_B, st.mu_C, st.mu_S)
        fields = {"branch": st.branch, "pattern": st.pattern,
                  "phi_bar": st.phi_bar, "chi_diel": st.chi,
                  "sigma": st.sigma, "zeta": st.zeta, "omega_0": st.omega_0,
                  "Sigma_R": st.Sigma_R, "Sigma_V": st.Sigma_V,
                  "M_u": st.M_star[0], "M_d": st.M_star[1],
                  "M_s": st.M_star[2],
                  # reported as magnitudes: the sign of each gap is a gauge
                  # (see `eos.ccdm.solver.point_from_state`)
                  "Delta_1": abs(st.Delta[0]), "Delta_2": abs(st.Delta[1]),
                  "Delta_3": abs(st.Delta[2]),
                  "mu_3": st.mu_3, "mu_8": st.mu_8}
        return PhaseThermo(
            T=st.T, mu_B=st.mu_B, mu_C=st.mu_C, mu_S=st.mu_S, fields=fields,
            densities={"u": n[0], "d": n[1], "s": n[2]},
            mu_i={"u": mu_u, "d": mu_d, "s": mu_s},
            mu_eff_i={"u": mu_u - st.Sigma_V, "d": mu_d - st.Sigma_V,
                      "s": mu_s - st.Sigma_V},
            m_eff_i={"u": st.M_star[0], "d": st.M_star[1],
                     "s": st.M_star[2]},
            n_B=st.n_B_fm, n_C=st.n_C_fm, n_S=st.n_S_fm,
            P=st.P_fm, eps=st.eps_fm, s=st.s_fm,
            mu_dot_n=st.mu_dot_n / hc3)

    def thermo(mu, mu_C, mu_S, T, n_B_guess=None, x0=None,
               return_state=False):
        seeds = dict(x0) if isinstance(x0, dict) else {}
        best = best_state = None
        for branch in branches:
            for pattern in patterns:
                st, ok, _ = thermo_from_mu(
                    par, mu, mu_C, mu_S, T, branch=branch, pattern=pattern,
                    x0=seeds.get((branch, pattern)))
                if not ok:
                    continue
                if best is None or st.P > best.P:
                    best = st
                    best_state = {(branch, pattern):
                                  _ccdm_internal_vector(st, par, pattern)}
        if best is None:
            raise RuntimeError(
                f"eos.ccdm: no branch/pattern converged at mu_B={mu:g}, "
                f"mu_C={mu_C:g}, mu_S={mu_S:g}, T={T:g} MeV")
        th = _block(best)
        return (th, best_state) if return_state else th

    def cold_start(n_B, T):
        p = _ccdm_beta(par, n_B, T, flags=flags, branches=branches,
                       patterns=patterns)
        if not p.converged:
            raise RuntimeError(f"eos.ccdm cold start failed at n_B={n_B}")
        return p.mu_B, p.mu_e, p.mu_B

    def _wing_point(spec, n_B, T):
        kw = dict(flags=flags, branches=branches, patterns=patterns)
        if spec.C is Regime.NOT_CONSERVED:
            if spec.L_e is Regime.GLOBAL:
                return _ccdm_trapped(par, n_B, spec.targets["Y_Le"], T, **kw)
            return _ccdm_beta(par, n_B, T, **kw)
        if spec.S is Regime.GLOBAL:
            return _ccdm_yc_ys(par, n_B, spec.targets["Y_C"],
                               spec.targets["Y_S"],
                               T, leptons=spec.yc_leptons, **kw)
        return _ccdm_yc(par, n_B, spec.targets["Y_C"], T,
                        leptons=spec.yc_leptons, **kw)

    def wing_sweep(spec, n_B_grid, T):
        out = []
        for n in n_B_grid:
            try:
                p = _wing_point(spec, float(n), T)
            except Exception:
                continue
            if p.converged:
                out.append((p.n_B, p.P_total, p.e_total))
        return out

    return Phase(name="CCDM", thermo=thermo, potential_kind="physical",
                 seed_cacheable=False, cold_start=cold_start,
                 wing_sweep=wing_sweep)


def _ccdm_internal_vector(st, par, pattern):
    """The internal unknown vector of a solved CCDM state, for a warm start.

    The layout is `eos.ccdm.thermodynamics.internal_unknowns`, rebuilt from
    the state rather than carried out of the solve, so the adapter stays a
    pure function of its arguments -- the mixed residual is finite-differenced,
    and an adapter that remembered its previous trial point would corrupt the
    Jacobian.
    """
    from eos.ccdm.couplings import has_vector
    from eos.ccdm.species import pattern_mask
    mask = pattern_mask(pattern)
    vector = [st.Phi, st.sigma, st.zeta]
    if has_vector(par):
        vector.append(st.Sigma_V)
    vector += [st.Delta[eta] for eta in range(3) if mask[eta]]
    if any(mask):
        vector += [st.mu_3, st.mu_8]
    return np.array(vector, dtype=float)


def enjl_branch_pair(par, branches=("broken", "restored")):
    """Two `Phase`s over ONE functional: an ENJL construction pairs branches.

    Each phase is `enjl_phase` at a declared branch of the same parameter
    set — the same thermodynamic potential admits several self-consistent
    states at one set of potentials, and a first-order transition in this
    model is a construction between two of them.

    The Phase declarations carry the ENJL rules: physical potentials on both
    slots (the rearrangement terms are unknowns inside its residual),
    `max_T = 0` (the surface is T = 0 only), NO seed cache and NO cold start —
    the seed ladder inside `enjl_phase` is a pure function of the arguments
    because it CHOOSES THE BRANCH, and an eta < 1 solve is seeded from a
    located eta = 1 Maxwell point rather than from any equilibrium of its
    own. No wings and no frozen block yet: the multi-window construction
    driver is the follow-up that needs them (docs/DEFERRED.md).
    """
    def make(branch):
        if branch not in ENJL_BRANCHES:
            raise ValueError(f"unknown ENJL branch {branch!r}; "
                             f"expected one of {ENJL_BRANCHES}")

        def thermo(mu, mu_C, mu_S, T, n_B_guess=None, x0=None,
                   return_state=False):
            th = enjl_phase(par, branch, mu, mu_C=mu_C, mu_S=mu_S, T=T)
            return (th, None) if return_state else th

        return Phase(name=branch, thermo=thermo, potential_kind="physical",
                     seed_cacheable=False, max_T=0.0)

    lo, hi = branches
    return make(lo), make(hi)
