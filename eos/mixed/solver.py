"""
mixed/equilibrium/residual.py
=============================
The mixed-phase equilibrium conditions, assembled from a `ChargeSpec` and eta.

*Internal module.* Driven by `eos.mixed.solve_mixed`.

At fixed (n_B, T, eta) plus whatever fractions the mode fixes, hadron-quark
coexistence is one nonlinear system. Its unknown vector and its residual list
are both *derived* from the per-charge regime assignment — they are not
enumerated per named mode. Adding an unnamed combination of regimes therefore
needs no new code.

Because the phase adapters already absorb each phase's internal
self-consistency (fields and densities at given potentials), the unknowns here
are only conserved-charge potentials, chi, and the eta-split lepton potentials
— four to nine numbers, rather than the full per-species (mu_i, n_i) vector a
direct formulation would carry.

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
whose pressures must balance.

Charged leptons
---------------
Electrons and, when enabled, muons share each neutrality domain. They are in
weak equilibrium with each other, mu_mu = mu_e - mu_nue (with mu_nue the trapped
neutrino potential, zero for transparent matter), which is the same relation
eos/dd2 uses. Muons therefore add no unknowns: they add their density to each
neutrality condition and their pressure to mechanical equilibrium.

Units are fm-based throughout. n_C is the NON-leptonic charge density; the
leptons neutralize it.
"""
from dataclasses import dataclass, field

from eos.general.thermodynamics_leptons import (
    electron_thermo, muon_thermo, neutrino_thermo,
)
from eos.mixed.equilibrium.charges import ChargeSpec, Regime
from eos.mixed.adapters import hadronic_phase, hadronic_seed, quark_phase


def _quark_mus_from_charges(mu_B, mu_C, mu_S):
    """(mu_B, mu_C, mu_S) -> (mu_u, mu_d, mu_s), inverting vMIT's convention
    mu_B = mu_u + 2 mu_d, mu_C = mu_u - mu_d, mu_S = mu_s - mu_d."""
    mu_u = (mu_B + 2.0 * mu_C) / 3.0
    mu_d = (mu_B - mu_C) / 3.0
    mu_s = mu_d + mu_S
    return mu_u, mu_d, mu_s


@dataclass(frozen=True)
class LeptonDomain:
    """The negatively-charged leptons sharing one neutrality domain.

    `n`, `P`, `e`, `s` are the electron plus (if enabled) muon totals;
    `mu_dot_n` is mu_e n_e + mu_mu n_mu, and `kappa` = dn/dmu_e, which is what
    the Jacobian needs (mu_mu tracks mu_e one-for-one, so the muon
    susceptibility simply adds).
    """
    n: float = 0.0
    P: float = 0.0
    e: float = 0.0
    s: float = 0.0
    mu_dot_n: float = 0.0
    n_e: float = 0.0
    n_mu: float = 0.0


def charged_leptons(mu_e, T, muons, mu_nue=0.0):
    """Electron (+muon) thermodynamics in one neutrality domain.

    Muons are in equilibrium with electrons at mu_mu = mu_e - mu_nue: for
    neutrino-transparent matter that is mu_mu = mu_e, and with trapped
    electron-neutrinos the muon family stays transparent, matching
    eos/dd2/solver.py.
    """
    e = electron_thermo(mu_e, T)
    if not muons:
        return LeptonDomain(n=e.n, P=e.P, e=e.e, s=e.s,
                            mu_dot_n=mu_e * e.n, n_e=e.n)
    mu_mu = mu_e - mu_nue
    m = muon_thermo(mu_mu, T)
    return LeptonDomain(
        n=e.n + m.n, P=e.P + m.P, e=e.e + m.e, s=e.s + m.s,
        mu_dot_n=mu_e * e.n + mu_mu * m.n, n_e=e.n, n_mu=m.n)


def has_leptons(spec: ChargeSpec):
    """Are neutralizing leptons present?

    Beta equilibrium always needs them. Fixed-Y_C needs them only in the
    'with neutralizing leptons' flavor; leptonless fixed-Y_C is a charged slice
    of matter with no neutrality condition at all, and is eta-independent.
    """
    return (spec.C is Regime.NOT_CONSERVED
            or (spec.C is Regime.GLOBAL and spec.yc_leptons))


def mixed_slots(spec: ChargeSpec, eta: float, flags=None):
    """Ordered unknown-vector slot names implied by `spec` at `eta`.

    Always present: the hadronic kinetic baryon potential (mu_tilde_B_H), the
    quark physical baryon potential (mu_B_Q), and chi.

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
    if flags is not None and flags.sigma_star:
        raise NotImplementedError(
            "SpeciesFlags.sigma_star (hidden-strange scalar) is not wired in "
            "the hadronic phase")
    # ponytail: exact-endpoint activation (eta>0 / eta<1). Very close to an
    # endpoint (|eta| or |1-eta| <~ 1e-3) the just-activated lepton population
    # carries almost no weight in the residual, so its potential is a near
    # spectator and the Jacobian is near-singular; a cold start can stall.
    # Interior eta on practical grids is fine. If a near-endpoint eta is
    # genuinely needed, warm-start it from the eta=0/1 solution rather than
    # snapping the activation threshold.
    slots = ["mu_tilde_B_H", "mu_B_Q", "chi"]
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
    par: object         # DD2 Parametrization
    flags: object       # SpeciesFlags
    vmit_params: object
    slots: tuple
    n_B_guess: float    # seed density for the hadronic phase-internal solve
    # residual normalization, so every row is dimensionless and O(1)
    n_scale: float = 1.0
    mu_scale: float = 100.0
    cache: dict = field(default_factory=dict)

    def hadronic_seed(self):
        """The per-solve constant starting configuration, computed once."""
        seed = self.cache.get("seed_H")
        if seed is None:
            seed = self.cache["seed_H"] = hadronic_seed(
                self.par, self.flags, self.T, self.n_B_guess)
        return seed


def build_mixed_ctx(spec, eta, n_B, par, flags, vmit_params, T=0.0,
                    n_B_guess=None):
    if not (0.0 <= eta <= 1.0):
        raise ValueError(f"eta must be in [0, 1], got {eta}")
    slots = mixed_slots(spec, eta, flags)
    return MixedCtx(
        spec=spec, eta=eta, n_B=n_B, T=T, par=par, flags=flags,
        vmit_params=vmit_params, slots=slots,
        n_B_guess=(n_B if n_B_guess is None else n_B_guess),
        n_scale=max(n_B, 0.01),
)


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
    muons = bool(ctx.flags.muons)
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

    th_H, state_H = hadronic_phase(
        ctx.par, ctx.flags, d["mu_tilde_B_H"], mu_C_H, mu_S, T=ctx.T,
        n_B_guess=ctx.n_B_guess, x0=ctx.hadronic_seed(), return_state=True)
    ctx.cache["state_H"] = state_H

    mu_u, mu_d, mu_s = _quark_mus_from_charges(d["mu_B_Q"], mu_C_Q, mu_S)
    th_Q = quark_phase(mu_u, mu_d, mu_s, T=ctx.T, params=ctx.vmit_params)

    zero = LeptonDomain()
    extras = dict(
        L_H=charged_leptons(mu_eL_H, ctx.T, muons, mu_nue) if (lep and eta > 0) else zero,
        L_Q=charged_leptons(mu_eL_Q, ctx.T, muons, mu_nue) if (lep and eta > 0) else zero,
        G=charged_leptons(mu_eG, ctx.T, muons, mu_nue) if (lep and eta < 1) else zero,
        nu=neutrino_thermo(mu_nue, ctx.T) if spec.L_e is Regime.GLOBAL else None,
)
    return th_H, th_Q, d, extras


def mixed_residual(x, ctx):
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
    chi, eta, spec = d["chi"], ctx.eta, ctx.spec
    lep = has_leptons(spec)
    ns, mus = ctx.n_scale, ctx.mu_scale
    L_H, L_Q, G = extras["L_H"], extras["L_Q"], extras["G"]

    P_H_eff = th_H.P + eta * L_H.P
    P_Q_eff = th_Q.P + eta * L_Q.P
    Ps = max(abs(P_H_eff), abs(P_Q_eff), 1.0)

    res = [
        (th_H.mu_B - d["mu_B_Q"]) / mus,                              # (1)
        ((1.0 - chi) * th_H.n_B + chi * th_Q.n_B - ctx.n_B) / ns,     # (2)
        (P_H_eff - P_Q_eff) / Ps,                                     # (3)
    ]
    if spec.C is Regime.GLOBAL:                                       # (4)
        Y_C = spec.targets["Y_C"]
        res.append(((1.0 - chi) * th_H.n_C + chi * th_Q.n_C
                    - Y_C * ctx.n_B) / ns)
        if spec.yc_leptons:
            # The local electron potential shifts the charge matching between
            # phases; the global one does not appear, because it neutralizes
            # the average rather than either phase.
            res.append((d["mu_C_H"] + eta * d.get("mu_eL_H", 0.0)
                        - d["mu_C_Q"] - eta * d.get("mu_eL_Q", 0.0)) / mus)
    if spec.S is Regime.GLOBAL:                                       # (5)
        Y_S = spec.targets["Y_S"]
        res.append(((1.0 - chi) * th_H.n_S + chi * th_Q.n_S
                    - Y_S * ctx.n_B) / ns)
    if spec.L_e is Regime.GLOBAL:                                     # (6)
        Y_Le = spec.targets["Y_Le"]
        n_e_avg = (eta * ((1.0 - chi) * L_H.n_e + chi * L_Q.n_e)
                   + (1.0 - eta) * G.n_e)
        res.append((n_e_avg + extras["nu"].n - Y_Le * ctx.n_B) / ns)
    if lep and eta > 0.0:                                             # (7) local
        res.append((th_H.n_C - L_H.n) / ns)
        res.append((th_Q.n_C - L_Q.n) / ns)
    if lep and eta < 1.0:                                             # (7) global
        res.append(((1.0 - chi) * th_H.n_C + chi * th_Q.n_C - G.n) / ns)
    return res
