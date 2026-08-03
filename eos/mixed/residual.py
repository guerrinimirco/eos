"""
mixed/residual.py
=================
Regime-driven residual assembly for the eta-mixed-phase solver
(docs/phase2/SPECIFICATION_AND_PLAN.md §1.4-§1.5, §3.3).

The mixed-phase equilibrium at fixed (n_B, T, eta) [+ mode targets] is one
nonlinear system. Its unknown vector and residual list are assembled from the
per-charge regime assignment (ChargeSpec) and eta — NOT enumerated per named
mode. The four modes are configurations of THIS function; an unnamed
combination works without new code (spec §1.5, the central design constraint).

What P0's adapters already absorb: each phase's internal self-consistency
(fields / densities given the conserved-charge potentials). So the mixed
unknowns are conserved-charge potentials + chi + the eta-split electron
potentials — not the per-species (mu_i, n_i) that zlvmit carried. That collapses
zlvmit's 12/13/14-unknown systems to 4/5/6 here.

Milestone status: Mode A (C NOT_CONSERVED, beta-eq) and Mode C (C GLOBAL, fixed
Y_C — both §1.6 lepton flavors) at T>=0 are wired. fixed-Y_S (S conserved) and
trapped neutrinos raise NotImplementedError rather than silently mis-assembling
(CLAUDE.md §5).

The C sector is the ONLY difference between Mode A and Mode C, and it is driven
by the regime (spec §1.5):
  - C NOT_CONSERVED (Mode A): mu_C^I is eliminated by the beta-eq closure
    mu_C^I = -(eta mu_eL^I + (1-eta) mu_eG)  (Eqs. 3.56-3.57);
  - C GLOBAL (Mode C): mu_C^H, mu_C^Q are unknowns, closed by charge
    conservation (1-chi)n_C^H + chi n_C^Q = Y_C n_B and the eta-shifted matching
    mu_C^H + eta mu_eL^H = mu_C^Q + eta mu_eL^Q  (Eq. 3.27; note mu_eG is ABSENT
    here — the audit §3 asymmetry, do not symmetrize). Leptonless (§1.6 2a):
    mu_C is shared and no electrons/neutrality exist (a charged CompOSE slice).

Sign/units: fm-based throughout (adapters + electron_thermo are fm-based).
n_C is the non-leptonic charge; electrons neutralize it (n_C - n_e = 0).
"""
from dataclasses import dataclass

from eos.general.thermodynamics_leptons import electron_thermo, neutrino_thermo
from eos.mixed.charges import ChargeSpec, Regime
from eos.mixed.phases import hadronic_phase, quark_phase


def _quark_mus_from_charges(mu_B, mu_C, mu_S):
    """(mu_B, mu_C, mu_S) -> (mu_u, mu_d, mu_s), inverting vMIT's convention
    mu_B=mu_u+2mu_d, mu_C=mu_u-mu_d, mu_S=mu_s-mu_d."""
    mu_u = (mu_B + 2.0 * mu_C) / 3.0
    mu_d = (mu_B - mu_C) / 3.0
    mu_s = mu_d + mu_S
    return mu_u, mu_d, mu_s


def has_leptons(spec: ChargeSpec):
    """Are neutralizing electrons present? Beta-equilibrium always needs them;
    fixed-Y_C only in the 'with neutralizing leptons' flavor (§1.6, yc_leptons).
    Leptonless fixed-Y_C is a charged CompOSE (n_B,T,Y_q) slice — no electrons,
    no neutrality, eta-independent.
    """
    return (spec.C is Regime.NOT_CONSERVED
            or (spec.C is Regime.GLOBAL and spec.yc_leptons))


def mixed_slots(spec: ChargeSpec, eta: float):
    """Ordered unknown-vector slot names for `spec` at `eta` (spec §1.5, §3.3).

    Always: H kinetic baryon potential (mu_tilde_B_H), Q physical baryon
    potential (mu_B_Q), chi. C GLOBAL adds the charge potential(s): per-phase
    (mu_C_H, mu_C_Q, tied by Eq. 3.27) with neutralizing leptons, or a single
    shared mu_C when leptonless. Electron populations activate by eta when
    leptons are present: local e_L (per phase) iff eta>0; global e_G iff eta<1.
    The named modes and the eta endpoints are all configurations of this rule.
    """
    if spec.S is Regime.LOCAL:
        raise NotImplementedError(
            "D-local (per-phase Y_S, S LOCAL, combined (A.93) baryon relation) "
            "is deferred behind a flag — audit §2")
    if spec.L_e is Regime.LOCAL:
        raise NotImplementedError(
            "local lepton number (L_e LOCAL) is not a defined mode")
    if spec.L_e is Regime.GLOBAL and spec.C is not Regime.NOT_CONSERVED:
        raise NotImplementedError(
            "trapped neutrinos combine with beta-eq charge (Mode B); "
            "fixed-Y_C + trapped-nu is not a named mode")
    # ponytail: exact-endpoint activation (eta>0 / eta<1). Very close to an
    # endpoint (|eta| or |1-eta| <~ 1e-3) the just-activated electron population
    # carries ~zero weight in the residual, so its potential is a near-spectator
    # and the Jacobian is near-singular -> cold-start solves can stall. Interior
    # eta on practical grids is fine; if a near-endpoint eta is needed, warm-
    # start it from the eta=0/1 solution (P8 boundary work) rather than snapping
    # the activation threshold.
    slots = ["mu_tilde_B_H", "mu_B_Q", "chi"]
    if spec.C is Regime.GLOBAL:
        slots += ["mu_C_H", "mu_C_Q"] if spec.yc_leptons else ["mu_C"]
    if spec.S is Regime.GLOBAL:
        slots += ["mu_S"]                      # shared (D-global); plain-matched
    if spec.L_e is Regime.GLOBAL:
        slots += ["mu_L"]                      # trapped-nu potential (global, §1.7)
    if has_leptons(spec) and eta > 0.0:
        slots += ["mu_eL_H", "mu_eL_Q"]
    if has_leptons(spec) and eta < 1.0:
        slots += ["mu_eG"]
    return tuple(slots)


@dataclass
class MixedCtx:
    """Everything the residual needs beyond the unknown vector."""
    spec: ChargeSpec
    eta: float
    n_B: float          # target total baryon density [fm^-3]
    T: float
    par: object         # DD2 Parametrization
    flags: object       # SpeciesFlags
    vmit_params: object
    slots: tuple
    n_B_guess: float    # seed for the hadronic phase-internal solve
    # residual normalization (dimensionless residuals, spec §3.3)
    n_scale: float = 1.0
    mu_scale: float = 100.0
    P_scale: float = 1.0


def build_mixed_ctx(spec, eta, n_B, par, flags, vmit_params, T=0.0,
                    n_B_guess=None):
    if not (0.0 <= eta <= 1.0):
        raise ValueError(f"eta must be in [0, 1], got {eta}")
    slots = mixed_slots(spec, eta)
    return MixedCtx(
        spec=spec, eta=eta, n_B=n_B, T=T, par=par, flags=flags,
        vmit_params=vmit_params, slots=slots,
        n_B_guess=(n_B if n_B_guess is None else n_B_guess),
        n_scale=max(n_B, 0.01),
    )


def evaluate_phases(x, ctx):
    """Solve both phases at the trial potentials; return (th_H, th_Q, extras).

    `extras` carries the electron densities and potentials used by the
    neutrality residuals. Beta-equilibrium (Mode A) closes the non-leptonic
    charge potential per phase as mu_C^I = -(eta mu_eL^I + (1-eta) mu_eG)
    (Eqs. 3.56-3.57); strangeness is not conserved so mu_S^I = 0.
    """
    d = dict(zip(ctx.slots, x))
    eta = ctx.eta
    spec = ctx.spec
    lep = has_leptons(spec)
    mu_eL_H = d.get("mu_eL_H", 0.0)
    mu_eL_Q = d.get("mu_eL_Q", 0.0)
    mu_eG = d.get("mu_eG", 0.0)

    # Trapped-neutrino potential (global, no eta split — §1.7); 0 in Mode A.
    mu_L = d.get("mu_L", 0.0)

    # Non-leptonic charge potential per phase, by C regime:
    if spec.C is Regime.NOT_CONSERVED:                  # beta-eq closure
        # ... shifted by mu_L when neutrinos are trapped (Mode B): mu_C + mu_e =
        # mu_L per phase; reduces to Mode A (mu_L=0) exactly.
        mu_C_H = mu_L - (eta * mu_eL_H + (1.0 - eta) * mu_eG)
        mu_C_Q = mu_L - (eta * mu_eL_Q + (1.0 - eta) * mu_eG)
    elif spec.yc_leptons:                               # Mode C 2b: per-phase
        mu_C_H, mu_C_Q = d["mu_C_H"], d["mu_C_Q"]
    else:                                               # Mode C 2a: shared
        mu_C_H = mu_C_Q = d["mu_C"]

    # Strangeness potential: 0 if self-equilibrating (S NOT_CONSERVED, mu_S=0),
    # else the shared D-global unknown (plain-matched across phases, audit §2).
    mu_S = d.get("mu_S", 0.0)

    th_H = hadronic_phase(ctx.par, ctx.flags, d["mu_tilde_B_H"], mu_C_H, mu_S,
                          T=ctx.T, n_B_guess=ctx.n_B_guess)
    mu_u, mu_d, mu_s = _quark_mus_from_charges(d["mu_B_Q"], mu_C_Q, mu_S)
    th_Q = quark_phase(mu_u, mu_d, mu_s, T=ctx.T, params=ctx.vmit_params)

    e_L_H = electron_thermo(mu_eL_H, ctx.T) if (lep and eta > 0.0) else None
    e_L_Q = electron_thermo(mu_eL_Q, ctx.T) if (lep and eta > 0.0) else None
    e_G = electron_thermo(mu_eG, ctx.T) if (lep and eta < 1.0) else None
    nu = neutrino_thermo(mu_L, ctx.T) if spec.L_e is Regime.GLOBAL else None
    extras = dict(
        n_eL_H=e_L_H.n if e_L_H else 0.0, P_eL_H=e_L_H.P if e_L_H else 0.0,
        n_eL_Q=e_L_Q.n if e_L_Q else 0.0, P_eL_Q=e_L_Q.P if e_L_Q else 0.0,
        n_eG=e_G.n if e_G else 0.0, P_eG=e_G.P if e_G else 0.0,
        n_nue=nu.n if nu else 0.0, P_nue=nu.P if nu else 0.0,
    )
    return th_H, th_Q, d, extras


def mixed_residual(x, ctx):
    """Dimensionless residual vector (spec §3.3), regime-driven.

    A phase solve that fails at a trial point (the outer Newton stepped into a
    region where a pure-phase engine cannot converge) returns a large penalty
    residual so hybr backs off, rather than aborting the whole solve — the same
    contract the adapters use internally for m* <= 0.
    """
    try:
        th_H, th_Q, d, extras = evaluate_phases(x, ctx)
    except RuntimeError:
        return [1.0e6] * len(ctx.slots)
    chi, eta = d["chi"], ctx.eta
    spec = ctx.spec
    lep = has_leptons(spec)
    ns, mus = ctx.n_scale, ctx.mu_scale
    # Mechanical equilibrium carries the eta-weighted LOCAL electron pressure
    # (spec §1.4): the e_G and photon parts are phase-common and cancel. Reduces
    # to matter-only P_H = P_Q at eta = 0 or when leptonless.
    P_H_eff = th_H.P + eta * extras["P_eL_H"]
    P_Q_eff = th_Q.P + eta * extras["P_eL_Q"]
    Ps = max(abs(P_H_eff), abs(P_Q_eff), 1.0)

    res = [
        (th_H.mu_B - d["mu_B_Q"]) / mus,                       # B GLOBAL (3.24)
        ((1.0 - chi) * th_H.n_B + chi * th_Q.n_B - ctx.n_B) / ns,   # baryon (3.6)
        (P_H_eff - P_Q_eff) / Ps,                              # mechanical (3.46)
    ]
    if spec.C is Regime.GLOBAL:                                # fixed Y_C (Mode C)
        Y_C = spec.targets["Y_C"]
        res.append(((1.0 - chi) * th_H.n_C + chi * th_Q.n_C
                    - Y_C * ctx.n_B) / ns)                     # charge cons (3.7)
        if spec.yc_leptons:
            # eta-shifted charge matching (3.27); mu_eG is ABSENT (audit §3).
            res.append((d["mu_C_H"] + eta * d.get("mu_eL_H", 0.0)
                        - d["mu_C_Q"] - eta * d.get("mu_eL_Q", 0.0)) / mus)
    if spec.S is Regime.GLOBAL:                                # fixed Y_S (Mode D)
        Y_S = spec.targets["Y_S"]
        res.append(((1.0 - chi) * th_H.n_S + chi * th_Q.n_S
                    - Y_S * ctx.n_B) / ns)                     # strangeness cons
    if spec.L_e is Regime.GLOBAL:                              # trapped nu (Mode B)
        Y_L = spec.targets["Y_L"]
        # electron lepton number = averaged electrons + (global) neutrinos, §1.7
        n_e_avg = (eta * ((1.0 - chi) * extras["n_eL_H"] + chi * extras["n_eL_Q"])
                   + (1.0 - eta) * extras["n_eG"])
        res.append((n_e_avg + extras["n_nue"] - Y_L * ctx.n_B) / ns)  # Y_L cons
    if lep and eta > 0.0:                                      # local neutrality
        res.append((th_H.n_C - extras["n_eL_H"]) / ns)         # (3.8)
        res.append((th_Q.n_C - extras["n_eL_Q"]) / ns)         # (3.9)
    if lep and eta < 1.0:                                      # global neutrality
        res.append(((1.0 - chi) * th_H.n_C + chi * th_Q.n_C
                    - extras["n_eG"]) / ns)                     # (3.10)
    return res
