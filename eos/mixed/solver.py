"""
mixed/solver.py
===============
The mixed-phase equilibrium conditions and the solves that close them.

*Public API* (re-exported from `eos.mixed`): `Result`, `solve`,
`sweep`, `seed_across_eta`.

At fixed (n_B, T, eta) plus whatever fractions the mode fixes, hadron-quark
coexistence is one nonlinear system. Its unknown vector and its residual list
are both *derived* from the per-charge regime assignment (`ChargeSpec`) — they
are not enumerated per named mode. Adding an unnamed combination of regimes
therefore needs no new code.

Because the phase adapters already absorb each phase's internal
self-consistency (fields and densities at given potentials), the unknowns here
are only conserved-charge potentials, chi, and the eta-split lepton potentials
— four to nine numbers, rather than the full per-species (mu_i, n_i) vector a
direct formulation would carry. The quantities computed FROM a solved state —
the lepton domains, the volume-averaged totals, the Euler identity — live in
`eos.mixed.thermodynamics`, which never knows which mode it is in.

The eta parameter
-----------------
eta interpolates continuously between the two standard constructions of a
first-order transition, by choosing how much of the electric-charge neutrality
is enforced *locally* (inside each phase) rather than *globally* (on the
volume average):

    eta = 0   Gibbs construction. Only the volume average is neutral, so each
              phase may be charged, the two phases exchange charge freely, and
              the pressure rises continuously through the mixed window.
    eta = 1   Maxwell construction. Each phase is separately neutral, no charge
              is exchanged, and the mixed window collapses to a constant-pressure
              plateau with a genuine density jump.
    0<eta<1   both a local and a global neutralizing lepton population exist,
              weighted eta and 1-eta. Physically this stands in for the finite
              surface tension and Coulomb energy of the mixed-phase structures,
              which suppress charge separation without forbidding it.

Concretely, when leptons are present the electron gas splits into a local part
(one potential per phase, mu_eL^H and mu_eL^Q, each neutralizing its own phase)
and a global part (a single mu_eG neutralizing the average). Only the local
part enters mechanical equilibrium, since only it lives inside the structures
whose pressures must balance. Electrons and, when enabled, muons share each
domain in weak equilibrium, mu_mu = mu_e - mu_nue — muons add no unknowns.

chi is the quark volume fraction and is deliberately *not* clamped to [0, 1]:
it is the continuation variable that tells the caller which regime a density is
in. chi <= 0 means the point is still pure hadronic, chi >= 1 that it is
already pure quark, and 0 < chi < 1 that it lies in the mixed window.
`eos.mixed.boundaries` locates the phase boundaries from exactly that.

Units are fm-based throughout. n_C is the NON-leptonic charge density; the
leptons neutralize it.

This file reads guesses -> the unknown vector and its context -> the phases at
trial potentials -> the residual -> the solve -> the density sweep.
"""
from dataclasses import dataclass, field, replace

from scipy.optimize import root

from eos.general.basis import quark_potentials
from eos.general.thermodynamics_leptons import neutrino_thermo
from eos.mixed.species import SpeciesFlags, mixture_flags
from eos.mixed.charges import (
    ChargeSpec, Regime,
    beta_eq_neutrinoless, beta_eq_neutrino_trapped, fixed_YC, fixed_YC_YS,
)
from eos.mixed.adapters import PhaseThermo
from eos.mixed.thermodynamics import (
    LeptonDomain, charged_leptons, assemble, euler_residual,
)

#: Post-solve residual gate, matching the tolerance eos/dd2 accepts.
RESIDUAL_TOL = 1.0e-10

#: Relative tolerance on the Euler / Hugenholtz-Van Hove identity.
HVH_RTOL = 1.0e-8


@dataclass
class Result:
    """One solved mixed-phase state.

    P, eps, s are the TOTALS: both matter phases volume-averaged, plus the
    eta-split leptons, plus photons and any trapped neutrinos.
    """
    converged: bool
    error: float
    n_B: float                  # fm^-3
    T: float                    # MeV
    eta: float
    chi: float                  # quark volume fraction (see module docstring)
    th_H: PhaseThermo
    th_Q: PhaseThermo
    potentials: dict            # solved unknown-vector slots
    mu_B: float                 # matched physical baryon potential [MeV]
    P: float                    # MeV/fm^3
    eps: float                  # MeV/fm^3
    s: float                    # fm^-3
    extras: dict = field(default_factory=dict)

    @property
    def in_mixed_phase(self):
        return 0.0 < self.chi < 1.0

    @property
    def phase(self):
        """'H' | 'mix' | 'Q' — which regime this density is in."""
        if self.chi <= 0.0:
            return "H"
        return "mix" if self.chi < 1.0 else "Q"


# ---------------------------------------------------------------------------
# guesses
# ---------------------------------------------------------------------------

def default_guess(ctx):
    """Physical cold start: each phase at its OWN pure equilibrium point.

    Each phase's `cold_start` supplies its slot potential and its electron
    potential from the phase's own beta equilibrium at the target density.
    That matches the eta=1 structure — where each phase is separately neutral —
    far better than a single shared mu_e would, and is a sound seed at eta=0
    too. chi starts mid-window. If the second phase's cold start does not
    converge (or the phase has none), its seed falls back to the first
    phase's physical potentials — for DD2+vMIT exactly the historical
    behaviour of seeding a failed vMIT solve from the hadronic point.

    A pair whose FIRST phase has no cold start cannot be seeded here: the
    caller must pass a warm start `x0` (the ENJL construction always seeds
    from a located Maxwell point). That is a raise, not a silent guess.
    """
    p0, p1 = ctx.pair
    n_seed = ctx.n_B if ctx.n_B is not None else ctx.n_B_guess
    if p0.cold_start is None:
        raise RuntimeError(
            f"the ({p0.name}, {p1.name}) pair has no cold start; "
            f"pass a warm start x0 in the slot order of mixed_slots")
    mu0, mu_e0, mu_phys0 = p0.cold_start(n_seed, ctx.T)
    try:
        if p1.cold_start is None:
            raise RuntimeError
        mu1, mu_e1, _ = p1.cold_start(n_seed, ctx.T)
    except Exception:
        mu1, mu_e1 = mu_phys0, mu_e0
    seed = {
        ctx.slots[0]: mu0,
        ctx.slots[1]: mu1,
        "chi": 0.5,
        "n_B": n_seed,                 # the unknown of the fixed-chi layout
        "mu_eL_H": mu_e0,
        "mu_eL_Q": mu_e1,
        "mu_eG": mu_e0,
        # Fixed-Y_C charge potentials: the beta-equilibrium value mu_C = -mu_e.
        "mu_C_H": -mu_e0,
        "mu_C_Q": -mu_e1,
        "mu_C": -mu_e0,
        # Strangeness self-equilibrating, neutrinos transparent.
        "mu_S": 0.0,
        "mu_nue": 0.0,
    }
    return [seed[name] for name in ctx.slots]


def warm_start(point, slots):
    """The seed taken from a previously solved point, in slot order."""
    return [point.potentials[name] for name in slots]


def seed_across_eta(result, spec, eta, phases=None):
    """Re-express a solved point as a starting vector for a different eta.

    eta changes the *shape* of the unknown vector — at eta = 0 only a global
    lepton potential exists, at eta = 1 only the two local ones, in between all
    three — so a solution at one eta cannot be handed to another directly.
    This maps it across, using the physical correspondence between the
    populations that appear and disappear:

      going towards eta = 1  the local potentials start from the global one,
                             which is the value they must approach as the
                             global population loses its weight;
      going towards eta = 0  the global potential starts from the volume
                             average of the two local ones.

    Cold-starting each eta separately is the fragile part of an eta scan,
    especially near the Maxwell endpoint and with a soft hadronic phase; walking
    eta in small steps from a converged neighbour is what makes the scan hold
    together. Returns a list in the slot order of `mixed_slots(spec, eta)`.
    """
    p = dict(result.potentials)
    chi = p.get("chi", result.chi)
    mu_eG = p.get("mu_eG")
    mu_eL_H, mu_eL_Q = p.get("mu_eL_H"), p.get("mu_eL_Q")
    if mu_eL_H is None:                    # came from eta = 0
        mu_eL_H = mu_eL_Q = mu_eG
    if mu_eG is None:                      # came from eta = 1
        mu_eG = (1.0 - chi) * mu_eL_H + chi * mu_eL_Q
    filled = dict(p, mu_eG=mu_eG, mu_eL_H=mu_eL_H, mu_eL_Q=mu_eL_Q)
    return [filled[name] for name in mixed_slots(spec, eta, phases)]


# ---------------------------------------------------------------------------
# the unknown vector and its context
# ---------------------------------------------------------------------------

def has_leptons(spec: ChargeSpec):
    """Are neutralizing leptons present?

    Beta equilibrium always needs them. Fixed-Y_C needs them only in the
    'with neutralizing leptons' flavor; leptonless fixed-Y_C is a charged slice
    of matter with no neutrality condition at all, and is eta-independent.
    """
    return (spec.C is Regime.NOT_CONSERVED
            or (spec.C is Regime.GLOBAL and spec.yc_leptons))


def mixed_slots(spec: ChargeSpec, eta: float, pair=None, fixed_chi=False):
    """Ordered unknown-vector slot names implied by `spec` at `eta`.

    Always present: the two baryon-potential slots and chi. The positions are
    labelled H and Q; each slot's PREFIX says what it carries, as declared by
    that phase's `potential_kind` — `mu_tilde_B_<pos>` for a kinetic-potential
    phase, `mu_B_<pos>` for a physical-potential one. Without a `pair` the
    names are the (kinetic H, physical Q) ones every hadron-quark pairing with
    a density-dependent hadronic phase derives, which is what lets a caller ask
    for the slot ORDER of a spec without holding a pairing.

    `fixed_chi=True` swaps exactly one slot: chi becomes a GIVEN and the total
    density n_B becomes the unknown in its place. Every residual row is
    unchanged — only which symbol is solved for moves — which is what lets a
    phase boundary be located in one solve: chi = 0 returns n_onset and
    chi = 1 returns n_offset exactly, with no grid resolution in the answer
    (`eos.mixed.boundaries.solve_fixed_chi`).

    Then, by regime: a GLOBAL C contributes the charge potential(s) — per-phase
    (mu_C_H, mu_C_Q, tied by the eta-shifted matching condition) when
    neutralizing leptons are present, or a single shared mu_C when leptonless.
    A GLOBAL S contributes mu_S; a GLOBAL L_e contributes mu_nue.

    Finally the lepton populations activate by eta: the local potentials
    (mu_eL_H, mu_eL_Q) exist iff eta > 0, the global one (mu_eG) iff eta < 1.
    Muons ride on these same potentials and add no slots.
    """
    if spec.S is Regime.LOCAL:
        raise NotImplementedError(
            "per-phase strangeness conservation (S LOCAL) is not wired; use "
            "fixed_YC_YS, which conserves strangeness globally over H+Q")
    if spec.L_e is Regime.LOCAL:
        raise NotImplementedError(
            "local lepton number (L_e LOCAL) is not a defined mode: the "
            "neutrino mean free path is far larger than the mixed-phase "
            "structures, so neutrinos cannot be localized in one phase")
    if spec.L_e is Regime.GLOBAL and spec.C is not Regime.NOT_CONSERVED:
        raise NotImplementedError(
            "trapped neutrinos are defined on top of beta-equilibrium charge; "
            "combining fixed Y_C with a fixed Y_Le is not a defined mode")
    if pair is not None and spec.S is Regime.GLOBAL:
        for phase in pair:
            if not phase.supports_S:
                raise NotImplementedError(
                    f"the {phase.name} phase carries no strangeness; a mode "
                    f"that conserves S globally cannot include it")
    # ponytail: exact-endpoint activation (eta>0 / eta<1). Very close to an
    # endpoint (|eta| or |1-eta| <~ 1e-3) the just-activated lepton population
    # carries almost no weight in the residual, so its potential is a near
    # spectator and the Jacobian is near-singular; a cold start can stall.
    # Interior eta on practical grids is fine. If a near-endpoint eta is
    # genuinely needed, warm-start it from the eta=0/1 solution rather than
    # snapping the activation threshold.
    if pair is None:
        slots = ["mu_tilde_B_H", "mu_B_Q"]
    else:
        slots = [pair[0].slot("H"), pair[1].slot("Q")]
    slots += ["n_B" if fixed_chi else "chi"]
    if spec.C is Regime.GLOBAL:
        slots += ["mu_C_H", "mu_C_Q"] if spec.yc_leptons else ["mu_C"]
    if spec.S is Regime.GLOBAL:
        slots += ["mu_S"]                  # shared, matched across phases
    if spec.L_e is Regime.GLOBAL:
        slots += ["mu_nue"]                  # trapped-neutrino potential
    if has_leptons(spec) and eta > 0.0:
        slots += ["mu_eL_H", "mu_eL_Q"]
    if has_leptons(spec) and eta < 1.0:
        slots += ["mu_eG"]
    return tuple(slots)


@dataclass
class MixedCtx:
    """Everything the residual needs beyond the unknown vector.

    `cache` memoizes the hadronic phase's starting field configuration. That
    seed depends only on (par, flags, T, n_B_guess) — none of which vary while
    the outer solver iterates on the charge potentials — so it is a constant
    that the original formulation recomputed, at the cost of a full
    beta-equilibrium solve, on every residual evaluation.

    Memoizing a constant is safe in a way that carrying the previous trial
    point's converged state forward is not: the latter would make the residual
    depend on the path taken to reach a point, and the outer finite-difference
    Jacobian would then differentiate that path dependence along with the
    physics.
    """
    spec: ChargeSpec
    eta: float
    n_B: float          # target total baryon density [fm^-3]
    T: float
    pair: tuple         # the two `Phase` objects, positions (H, Q)
    slots: tuple
    n_B_guess: float    # seed density for a phase-internal solve
    # The engine's own section-4 flags: the phase-common sectors it
    # consumes itself (photons), and the muons of the eta-split lepton
    # domains. The per-phase sectors travel inside each `Phase`.
    species: SpeciesFlags = SpeciesFlags()
    chi: float = None   # the imposed volume fraction in the fixed-chi layout
    # residual normalization, so every row is dimensionless and O(1)
    n_scale: float = 1.0
    mu_scale: float = 100.0
    cache: dict = field(default_factory=dict)

    def phase_seed(self, phase, position):
        """One phase's starting configuration, memoized per solve — unless
        the phase forbids it: for a branch-declared adapter the seed CHOOSES
        THE ROOT, so a cache would change physics, not speed."""
        if phase.seed is None:
            return None
        if not phase.seed_cacheable:
            return phase.seed(self.T, self.n_B_guess)
        key = f"seed_{position}"
        seed = self.cache.get(key)
        if seed is None:
            seed = self.cache[key] = phase.seed(self.T, self.n_B_guess)
        return seed

    def hadronic_seed(self):
        """The historical name for position H's seed; prefer `phase_seed`."""
        return self.phase_seed(self.pair[0], "H")


def build_mixed_ctx(spec, eta, n_B, phases, T=0.0, n_B_guess=None,
                    chi=None, species=None):
    """Context for one solve. Exactly one of `n_B` (the given total density,
    chi unknown) and `chi` (the imposed volume fraction, n_B unknown) is set;
    the second flavour is the fixed-chi layout of `mixed_slots` and needs
    `n_B_guess` because the target density is now an unknown.

    `phases` is the pairing: two `Phase` objects, each closing over its own
    model's parameters. It IS the engine's parameter argument (CLAUDE.md
    section 5) — no model's parameter set reaches this function by any other
    route, and `eos.mixed.adapters.default_pair` is how a caller who wants the
    DD2 + vMIT pairing builds one.

    `species` is the ENGINE's own `SpeciesFlags` (CLAUDE.md section 4): the
    phase-common photon gas and the muons of the eta-split lepton domains. It
    defaults to every sector off, because no phase can answer a question about
    the shared leptons — the per-phase sectors travel inside each `Phase`.
    """
    if not (0.0 <= eta <= 1.0):
        raise ValueError(f"eta must be in [0, 1], got {eta}")
    if (n_B is None) == (chi is None):
        raise ValueError("exactly one of n_B (given) / chi (imposed) "
                         "must be set")
    if n_B is None and n_B_guess is None:
        raise ValueError("the fixed-chi layout needs n_B_guess: the density "
                         "is an unknown and its solve must start somewhere")
    if phases is None:
        raise ValueError(
            "the mixed engine needs a pairing: phases=(Phase, Phase). For the "
            "DD2 + vMIT hybrid write "
            "phases=eos.mixed.adapters.default_pair(par, flags, vmit_params)")
    for phase in phases:
        if phase.max_T is not None and T > phase.max_T:
            raise NotImplementedError(
                f"the {phase.name} phase is wired for T <= {phase.max_T} "
                f"MeV only, got T={T}")
    species = mixture_flags(species)
    slots = mixed_slots(spec, eta, phases, fixed_chi=(n_B is None))
    return MixedCtx(
        spec=spec, eta=eta, n_B=n_B, T=T, pair=tuple(phases), slots=slots,
        chi=chi, species=species,
        n_B_guess=(n_B if n_B_guess is None else n_B_guess),
        n_scale=max(n_B if n_B is not None else n_B_guess, 0.01),
)


# ---------------------------------------------------------------------------
# the phases at trial potentials
# ---------------------------------------------------------------------------

def evaluate_phases(x, ctx):
    """Solve both phases at the trial potentials.

    Returns (th_H, th_Q, slot_dict, extras), where `extras` carries the
    per-domain `LeptonDomain` blocks and the trapped-neutrino block that the
    neutrality and lepton-number residuals consume.

    The non-leptonic charge potential of each phase is set by the C regime.
    In beta equilibrium it is not an unknown at all: it is eliminated by the
    beta condition mu_C + mu_e = mu_nue, applied with the eta-weighted electron
    potential of that phase.
    """
    d = dict(zip(ctx.slots, x))
    eta, spec = ctx.eta, ctx.spec
    lep = has_leptons(spec)
    muons = ctx.species.muons
    mu_eL_H = d.get("mu_eL_H", 0.0)
    mu_eL_Q = d.get("mu_eL_Q", 0.0)
    mu_eG = d.get("mu_eG", 0.0)
    mu_nue = d.get("mu_nue", 0.0)          # trapped-neutrino potential; 0 if none

    if spec.C is Regime.NOT_CONSERVED:                  # beta-equilibrium
        mu_C_H = mu_nue - (eta * mu_eL_H + (1.0 - eta) * mu_eG)
        mu_C_Q = mu_nue - (eta * mu_eL_Q + (1.0 - eta) * mu_eG)
    elif spec.yc_leptons:                               # fixed Y_C, per-phase
        mu_C_H, mu_C_Q = d["mu_C_H"], d["mu_C_Q"]
    else:                                               # fixed Y_C, leptonless
        mu_C_H = mu_C_Q = d["mu_C"]

    # Strangeness potential: 0 when strangeness self-equilibrates, otherwise
    # the shared unknown matched across both phases.
    mu_S = d.get("mu_S", 0.0)

    p_H, p_Q = ctx.pair
    th_H, state_H = p_H.thermo(d[ctx.slots[0]], mu_C_H, mu_S, ctx.T,
                               n_B_guess=ctx.n_B_guess,
                               x0=ctx.phase_seed(p_H, "H"),
                               return_state=True)
    ctx.cache["state_H"] = state_H

    th_Q, state_Q = p_Q.thermo(d[ctx.slots[1]], mu_C_Q, mu_S, ctx.T,
                               n_B_guess=ctx.n_B_guess,
                               x0=ctx.phase_seed(p_Q, "Q"),
                               return_state=True)
    ctx.cache["state_Q"] = state_Q

    zero = LeptonDomain()
    extras = dict(
        L_H=charged_leptons(mu_eL_H, ctx.T, muons, mu_nue) if (lep and eta > 0) else zero,
        L_Q=charged_leptons(mu_eL_Q, ctx.T, muons, mu_nue) if (lep and eta > 0) else zero,
        G=charged_leptons(mu_eG, ctx.T, muons, mu_nue) if (lep and eta < 1) else zero,
        nu=neutrino_thermo(mu_nue, ctx.T) if spec.L_e is Regime.GLOBAL else None,
)
    return th_H, th_Q, d, extras


# ---------------------------------------------------------------------------
# the residual
# ---------------------------------------------------------------------------

def residual(x, ctx):
    """Dimensionless residual vector, assembled by regime.

    Rows, in order:
      1. baryon potentials match across the phases (B is GLOBAL in every mode);
      2. the volume average reproduces the target baryon density;
      3. mechanical equilibrium — the phase pressures balance, each carrying
         its own *local* lepton pressure. The global lepton and photon
         pressures are common to both phases and cancel.
    then, by regime:
      4. GLOBAL C: charge conservation on the average, plus (with neutralizing
         leptons) the eta-shifted charge matching between phases;
      5. GLOBAL S: strangeness conservation on the average over both phases;
      6. GLOBAL L_e: lepton-number conservation, counting the eta-weighted
         electrons and the global neutrinos;
      7. neutrality: locally in each phase when eta > 0, and on the volume
         average when eta < 1.

    A phase solve that fails at a trial point — the outer Newton stepped into a
    region where a bulk engine cannot converge — returns a large penalty
    residual so the solver backs off, rather than aborting the whole solve.
    """
    try:
        th_H, th_Q, d, extras = evaluate_phases(x, ctx)
    except RuntimeError:
        return [1.0e6] * len(ctx.slots)
    chi = d.get("chi", ctx.chi)          # unknown, or imposed (fixed-chi)
    n_B = d.get("n_B", ctx.n_B)          # given, or unknown (fixed-chi)
    eta, spec = ctx.eta, ctx.spec
    lep = has_leptons(spec)
    ns, mus = ctx.n_scale, ctx.mu_scale
    L_H, L_Q, G = extras["L_H"], extras["L_Q"], extras["G"]

    P_H_eff = th_H.P + eta * L_H.P
    P_Q_eff = th_Q.P + eta * L_Q.P
    Ps = max(abs(P_H_eff), abs(P_Q_eff), 1.0)

    # (1) the PHYSICAL baryon potentials match. A physical-kind slot carries
    # mu_B itself; a kinetic-kind phase restores mu_B = mu_tilde_B + Sigma^R
    # in its block, so th.mu_B is read instead of the slot.
    p_H, p_Q = ctx.pair
    mu_B_H = d[ctx.slots[0]] if p_H.potential_kind == "physical" else th_H.mu_B
    mu_B_Q = d[ctx.slots[1]] if p_Q.potential_kind == "physical" else th_Q.mu_B
    res = [
        (mu_B_H - mu_B_Q) / mus,                                      # (1)
        ((1.0 - chi) * th_H.n_B + chi * th_Q.n_B - n_B) / ns,         # (2)
        (P_H_eff - P_Q_eff) / Ps,                                     # (3)
    ]
    if spec.C is Regime.GLOBAL:                                       # (4)
        Y_C = spec.targets["Y_C"]
        res.append(((1.0 - chi) * th_H.n_C + chi * th_Q.n_C
                    - Y_C * n_B) / ns)
        if spec.yc_leptons:
            # The local electron potential shifts the charge matching between
            # phases; the global one does not appear, because it neutralizes
            # the average rather than either phase.
            res.append((d["mu_C_H"] + eta * d.get("mu_eL_H", 0.0)
                        - d["mu_C_Q"] - eta * d.get("mu_eL_Q", 0.0)) / mus)
    if spec.S is Regime.GLOBAL:                                       # (5)
        Y_S = spec.targets["Y_S"]
        res.append(((1.0 - chi) * th_H.n_S + chi * th_Q.n_S
                    - Y_S * n_B) / ns)
    if spec.L_e is Regime.GLOBAL:                                     # (6)
        Y_Le = spec.targets["Y_Le"]
        n_e_avg = (eta * ((1.0 - chi) * L_H.n_e + chi * L_Q.n_e)
                   + (1.0 - eta) * G.n_e)
        res.append((n_e_avg + extras["nu"].n - Y_Le * n_B) / ns)
    if lep and eta > 0.0:                                             # (7) local
        res.append((th_H.n_C - L_H.n) / ns)
        res.append((th_Q.n_C - L_Q.n) / ns)
    if lep and eta < 1.0:                                             # (7) global
        res.append(((1.0 - chi) * th_H.n_C + chi * th_Q.n_C - G.n) / ns)
    return res


# ---------------------------------------------------------------------------
# the solve
# ---------------------------------------------------------------------------

def _jac_with_fallback(ctx):
    """Wrap the analytic Jacobian so a trial point where a phase solve fails
    still yields a usable matrix.

    The residual answers such a point with a large penalty rather than an
    exception; the Jacobian mirrors that by falling back to a finite difference
    of the (penalised) residual, so the outer solver backs off instead of
    aborting.

    Returns None when `backends/` is absent, and for a pair in which either
    phase advertises no analytic block: the backends are deletable
    (CLAUDE.md section 9) and the analytic Jacobian is an OPTIONAL capability
    a phase may carry, so `analytic_jac=True` then means the solver's own
    numeric Jacobian rather than an error.
    """
    import numpy as np
    if any(phase.jacobian_block is None for phase in ctx.pair):
        return None
    try:
        from eos.mixed.backends.jacobian import mixed_jacobian
    except ImportError:
        return None

    def jac(x, ctx_):
        try:
            return mixed_jacobian(x, ctx_)
        except (RuntimeError, np.linalg.LinAlgError):
            n = len(ctx_.slots)
            J = np.zeros((len(residual(x, ctx_)), n))
            for i in range(n):
                h = max(1e-4, 1e-6 * abs(x[i]))
                xp, xm = list(x), list(x)
                xp[i] += h
                xm[i] -= h
                J[:, i] = (np.array(residual(xp, ctx_))
                           - np.array(residual(xm, ctx_))) / (2.0 * h)
            return J
    return jac


def solve(phases, n_B, eta, spec, T=0.0, x0=None, n_B_guess=None,
                check_consistency=True, analytic_jac=False, species=None):
    """Solve the mixed phase at (n_B, T, eta) for the regime assignment `spec`.

    phases      : the pairing — two `Phase` objects, each closing over its own
                  model's parameters. This is the engine's parameter argument
                  (CLAUDE.md section 5); `adapters.default_pair(par, flags,
                  vmit_params)` builds the DD2 + vMIT one.
    n_B         : total baryon density [fm^-3]
    eta         : local-neutrality fraction in [0, 1] (0 Gibbs, 1 Maxwell)
    spec        : `ChargeSpec` from one of the named mode factories
    x0          : optional warm start, in the slot order of
                  `mixed_slots(spec, eta, phases)`
    analytic_jac: supply the hand-assembled Jacobian instead of the solver's
                  own numeric one. The numeric path is the correctness oracle.
                  Honoured only when both phases advertise a Jacobian block.
    species     : the ENGINE's own `SpeciesFlags` — the phase-common photon
                  gas and the muons of the eta-split lepton domains. Defaults
                  to every sector off, because no phase can answer a question
                  about the shared leptons.

    Returns a `Result`; raises RuntimeError if the residual gate is not
    met from any guess, so non-convergence is never silent.
    """
    ctx = build_mixed_ctx(spec, eta, n_B, phases, T=T,
                          n_B_guess=n_B_guess, species=species)
    jac = _jac_with_fallback(ctx) if analytic_jac else None

    # Lazy: the cold start costs a full DD2 solve plus a full vMIT solve, so it
    # is built only if `x0` is missing or its solve does not converge. In a
    # warm-started sweep it is never evaluated, which is the hot-path win.
    def guesses():
        if x0 is not None:
            yield list(x0)
        yield default_guess(ctx)

    sol = None
    for guess in guesses():
        ctx.cache.clear()          # a fresh guess invalidates the phase cache
        sol = root(residual, guess, args=(ctx,), method="hybr",
                   tol=1e-12, jac=jac)
        res_max = max(abs(r) for r in residual(sol.x, ctx))
        if res_max <= RESIDUAL_TOL:
            break
    else:
        raise RuntimeError(
            f"mixed solve failed at n_B={n_B}, T={T}, eta={eta}: {sol.message} "
            f"(max residual {res_max:.2e}, tol {RESIDUAL_TOL:.0e})")

    return result_from_root(sol.x, ctx, res_max,
                            check_consistency=check_consistency)


def result_from_root(x, ctx, res_max, check_consistency=True):
    """A `Result` from a residual root, in either slot layout.

    Assembles the totals, refuses a Bose-condensed state, and asserts the
    consistency identities. Shared by `solve` and by the fixed-chi
    boundary solve in `eos.mixed.boundaries`, whose only difference is which
    of (n_B, chi) was the unknown.
    """
    th_H, th_Q, d, extras = evaluate_phases(x, ctx)
    chi = d.get("chi", ctx.chi)
    n_B = d.get("n_B", ctx.n_B)
    T, eta = ctx.T, ctx.eta
    L_H, L_Q, G, nu = extras["L_H"], extras["L_Q"], extras["G"], extras["nu"]

    P_total, eps_total, s_total, mu_dot_n = assemble(
        chi, eta, th_H, th_Q, L_H, L_Q, G, nu,
        mu_nue=d.get("mu_nue", 0.0), T=T, photons=ctx.species.photons)

    result = Result(
        converged=True, error=res_max, n_B=n_B, T=T, eta=eta, chi=chi,
        th_H=th_H, th_Q=th_Q, potentials=d, mu_B=th_H.mu_B,
        P=P_total, eps=eps_total, s=s_total, extras=extras)

    # Bose-Einstein condensation is not implemented, so a condensed hadronic
    # phase is outside the model rather than a hard point. It has to be checked
    # HERE and not left to the caller: `solve_bose_jel` caps mu*_j at m_j
    # instead of diverging, so the phase converges happily on a saturated gas
    # that is not the physical state, and every residual row it feeds -- the
    # charge rows above all, since the gas carries n_C and n_S -- is then
    # quietly wrong. Refused rather than returned, matching how `eos.dd2`
    # refuses the same condition; `eos.mixed.eos_point` turns it into a status.
    # The quark phase has no meson gas and reports 0.
    condensation = max(th_H.condensation, th_Q.condensation)
    if condensation >= 1.0:
        raise ValueError(
            f"the hadronic phase's thermal meson gas Bose-condenses at "
            f"n_B={n_B}, T={T}, eta={eta} (max |mu*|/m = {condensation:.3f}); "
            f"a condensate is not implemented, so this state is outside the "
            f"model")

    if check_consistency and result.in_mixed_phase:
        dP = (th_H.P + eta * L_H.P) - (th_Q.P + eta * L_Q.P)
        if abs(dP) > 1e-6 * max(abs(P_total), 1.0):
            raise ValueError(
                f"mechanical equilibrium violated at n_B={n_B}, eta={eta}: "
                f"dP={dP:.2e} MeV/fm^3")
        # Euler / Hugenholtz-Van Hove for the whole mixture (CLAUDE.md §8):
        # eps + P = T s + sum_i mu_i n_i.
        hvh = euler_residual(P_total, eps_total, s_total, mu_dot_n, T)
        if abs(hvh) > HVH_RTOL:
            raise ValueError(
                f"Euler / Hugenholtz-Van Hove violated at n_B={n_B}, T={T}, "
                f"eta={eta}: |{hvh:.2e}| > {HVH_RTOL:.0e} — a thermal or "
                f"lepton term is inconsistent")
    return result


# ---------------------------------------------------------------------------
# the modes (CLAUDE.md section 3, named per section 13)
# ---------------------------------------------------------------------------
# Each is the general solve at the ChargeSpec its mode factory declares — the
# residual is never duplicated per mode. `**kw` passes solve's tuning
# (x0, n_B_guess, species, check_consistency, analytic_jac) through.

def solve_beta_eq_neutrinoless(phases, n_B, eta, T=0.0, **kw):
    """Beta equilibrium, free-streaming neutrinos, charge neutral: (n_B, T)."""
    return solve(phases, n_B, eta, beta_eq_neutrinoless(), T=T, **kw)


def solve_beta_eq_neutrino_trapped(phases, n_B, Y_Le, eta, T=0.0, **kw):
    """Beta equilibrium with trapped neutrinos: (n_B, Y_Le, T)."""
    return solve(phases, n_B, eta, beta_eq_neutrino_trapped(Y_Le), T=T, **kw)


def solve_fixed_yc(phases, n_B, Y_C, eta, T=0.0, leptons=True, **kw):
    """Fixed non-leptonic charge fraction: (n_B, Y_C, T).

    leptons=True adds electrons (and muons if enabled) enforcing total
    neutrality; leptons=False is a charged, eta-independent slice.
    """
    return solve(phases, n_B, eta, fixed_YC(Y_C, leptons=leptons), T=T, **kw)


def solve_fixed_yc_ys(phases, n_B, Y_C, Y_S, eta, T=0.0, leptons=True, **kw):
    """Fixed charge and strangeness fractions: (n_B, Y_C, Y_S, T)."""
    return solve(phases, n_B, eta, fixed_YC_YS(Y_C, Y_S, leptons=leptons),
                       T=T, **kw)


# ---------------------------------------------------------------------------
# the density sweep
# ---------------------------------------------------------------------------

def sweep(phases, n_B_grid, eta, spec, T=0.0,
                max_bisect=6, x0=None, analytic_jac=False,
                mixed_only=False, nH0=None, species=None):
    """Warm-started sweep over `n_B_grid` at fixed eta.

    A cold seed only converges near the transition onset; through the window
    the previous solved density is by far the best predictor. So the sweep is
    warm-started along n_B, and when a step misses, it is bisected rather than
    abandoned — the same continuation tactic `eos/dd2/solver.sweep` uses.

    Returns a list of `Result` in grid order. Each solved point seeds the
    next; a missed step is bisected up to `max_bisect` levels, and a density
    that still fails is skipped, leaving a hole rather than aborting the sweep.

    `x0` seeds the first point (otherwise the physical cold start is used).
    `nH0` seeds that first point's *hadronic phase* density; see `solve_from`
    below for why the total density is the wrong guess once chi is appreciable.
    Pass it whenever `x0` comes from a solved point deep inside the window,
    otherwise the first step re-derives that point from a guess known to stall.
    `mixed_only=True` keeps only the points genuinely inside the window.
    """
    slots = mixed_slots(spec, eta, phases)

    def solve_from(n_B, seed, nH):
        # `nH` seeds the first phase's own internal solve. It must be that
        # phase's density, NOT the total: deep in the window the low-density
        # phase thins out and stops tracking n_B at all, and seeding it at
        # the total density walks it towards collapse and stalls the solve.
        return solve(phases, n_B, eta, spec, T=T, x0=seed,
                           n_B_guess=(nH if nH is not None else n_B),
                           check_consistency=False, analytic_jac=analytic_jac,
                           species=species)

    def step(n_prev, n_target, seed, nH, depth):
        try:
            return solve_from(n_target, seed, nH)
        except RuntimeError:
            if depth >= max_bisect or n_prev is None:
                raise
            n_mid = 0.5 * (n_prev + n_target)
            p_mid = step(n_prev, n_mid, seed, nH, depth + 1)
            return step(n_mid, n_target, warm_start(p_mid, slots),
                        p_mid.th_H.n_B, depth + 1)

    out, seed, n_prev, nH = [], x0, None, nH0
    for n_B in n_B_grid:
        try:
            p = step(n_prev, float(n_B), seed, nH, 0)
        except RuntimeError:
            continue
        out.append(p)
        seed, n_prev, nH = warm_start(p, slots), float(n_B), p.th_H.n_B
    return [r for r in out if r.in_mixed_phase] if mixed_only else out
