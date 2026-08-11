"""
mixed/solvers/point.py
======================
`solve_mixed` — one hadron-quark mixed-phase equilibrium point.

*Public API* (re-exported from `eos.mixed`).

Drives the regime-assembled residual through MINPACK's hybrid Powell solver and
returns a `MixedResult` carrying both phase blocks, the eta-split lepton
potentials, and the total thermodynamics of the mixture.

chi is the quark volume fraction and is deliberately *not* clamped to [0, 1]:
it is the continuation variable that tells the caller which regime a density is
in. chi <= 0 means the point is still pure hadronic, chi >= 1 that it is
already pure quark, and 0 < chi < 1 that it lies in the mixed window. The table
builders use exactly that to locate the phase boundaries.
"""
from dataclasses import dataclass, field

from scipy.optimize import root

from eos.general.thermodynamics_leptons import photon_thermo
from eos.dd2.solver import solve_beta_eq_octet
from eos.mixed.equilibrium.residual import (
    build_mixed_ctx, mixed_residual, evaluate_phases, has_leptons,
)
from eos.mixed.adapters import PhaseThermo

#: Post-solve residual gate, matching the tolerance eos/dd2 accepts.
RESIDUAL_TOL = 1.0e-10

#: Relative tolerance on the Euler / Hugenholtz-Van Hove identity.
HVH_RTOL = 1.0e-8


@dataclass
class MixedResult:
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


def default_guess(ctx):
    """Physical cold start: each phase at its OWN pure beta-equilibrium point.

    The hadronic side supplies mu_tilde_B and its electron potential, the quark
    side (solved separately at the same density) supplies mu_B_Q and its own.
    That matches the eta=1 structure — where each phase is separately neutral —
    far better than a single shared mu_e would, and is a sound seed at eta=0
    too. chi starts mid-window. If vMIT does not converge at this density the
    quark seed falls back to the hadronic potentials.
    """
    from eos.vmit.eos import solve_vmit_beta_eq
    base = solve_beta_eq_octet(ctx.par, ctx.n_B, ctx.flags, T=ctx.T,
                               include_photons=False, check_consistency=False)
    mu_tilde_B = base.mu_n - base.Sigma_R
    try:
        q = solve_vmit_beta_eq(ctx.n_B, ctx.T, params=ctx.vmit_params)
        mu_B_Q, mu_eL_Q = q.mu_B, q.mu_e
    except Exception:
        mu_B_Q, mu_eL_Q = base.mu_n, base.mu_e
    seed = {
        "mu_tilde_B_H": mu_tilde_B,
        "mu_B_Q": mu_B_Q,
        "chi": 0.5,
        "mu_eL_H": base.mu_e,
        "mu_eL_Q": mu_eL_Q,
        "mu_eG": base.mu_e,
        # Fixed-Y_C charge potentials: the beta-equilibrium value mu_C = -mu_e.
        "mu_C_H": -base.mu_e,
        "mu_C_Q": -mu_eL_Q,
        "mu_C": -base.mu_e,
        # Strangeness self-equilibrating, neutrinos transparent.
        "mu_S": 0.0,
        "mu_L": 0.0,
    }
    return [seed[name] for name in ctx.slots]


def _jac_with_fallback(ctx):
    """Wrap the analytic Jacobian so a trial point where a phase solve fails
    still yields a usable matrix.

    The residual answers such a point with a large penalty rather than an
    exception; the Jacobian mirrors that by falling back to a finite difference
    of the (penalised) residual, so the outer solver backs off instead of
    aborting.
    """
    import numpy as np
    from eos.mixed.equilibrium.jacobian import mixed_jacobian

    def jac(x, ctx_):
        try:
            return mixed_jacobian(x, ctx_)
        except (RuntimeError, np.linalg.LinAlgError):
            n = len(ctx_.slots)
            J = np.zeros((len(mixed_residual(x, ctx_)), n))
            for i in range(n):
                h = max(1e-4, 1e-6 * abs(x[i]))
                xp, xm = list(x), list(x)
                xp[i] += h
                xm[i] -= h
                J[:, i] = (np.array(mixed_residual(xp, ctx_))
                           - np.array(mixed_residual(xm, ctx_))) / (2.0 * h)
            return J
    return jac


def solve_mixed(par, flags, n_B, eta, spec, vmit_params=None, T=0.0,
                x0=None, n_B_guess=None, check_consistency=True,
                analytic_jac=False):
    """Solve the mixed phase at (n_B, T, eta) for the regime assignment `spec`.

    par         : DD2 `Parametrization`
    flags       : `SpeciesFlags` — which baryons, leptons and meson gases exist
    n_B         : total baryon density [fm^-3]
    eta         : local-neutrality fraction in [0, 1] (0 Gibbs, 1 Maxwell)
    spec        : `ChargeSpec` from one of the named mode factories
    x0          : optional warm start, in the slot order of
                  `mixed_slots(spec, eta, flags)`
    analytic_jac: supply the hand-assembled Jacobian instead of the solver's
                  own numeric one. The numeric path is the correctness oracle.

    Returns a `MixedResult`; raises RuntimeError if the residual gate is not
    met from any guess, so non-convergence is never silent.
    """
    if vmit_params is None:
        from eos.vmit.parameters import get_vmit_default
        vmit_params = get_vmit_default()
    ctx = build_mixed_ctx(spec, eta, n_B, par, flags, vmit_params, T=T,
                          n_B_guess=n_B_guess)
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
        sol = root(mixed_residual, guess, args=(ctx,), method="hybr",
                   tol=1e-12, jac=jac)
        res_max = max(abs(r) for r in mixed_residual(sol.x, ctx))
        if res_max <= RESIDUAL_TOL:
            break
    else:
        raise RuntimeError(
            f"mixed solve failed at n_B={n_B}, T={T}, eta={eta}: {sol.message} "
            f"(max residual {res_max:.2e}, tol {RESIDUAL_TOL:.0e})")

    th_H, th_Q, d, extras = evaluate_phases(sol.x, ctx)
    chi = d["chi"]
    lep = has_leptons(spec)
    L_H, L_Q, G, nu = extras["L_H"], extras["L_Q"], extras["G"], extras["nu"]

    # Totals. The matter phases are volume-averaged. The local leptons carry
    # weight eta and are themselves volume-averaged; the global leptons carry
    # weight 1-eta and are uniform. Photons and trapped neutrinos are uniform
    # across the whole mixture and are counted once.
    ph = photon_thermo(T) if T > 0.0 else None
    P_g, e_g, s_g = (ph.P, ph.e, ph.s) if ph else (0.0, 0.0, 0.0)
    P_nu, e_nu, s_nu = (nu.P, nu.e, nu.s) if nu else (0.0, 0.0, 0.0)

    def avg_local(attr):
        return (1.0 - chi) * getattr(L_H, attr) + chi * getattr(L_Q, attr)

    # Pressure is uniform, so it is read off either phase (they are equal by
    # the mechanical-equilibrium row) plus the phase-common parts.
    P_total = (th_H.P + eta * L_H.P + (1.0 - eta) * G.P + P_nu + P_g)
    eps_total = ((1.0 - chi) * th_H.eps + chi * th_Q.eps
                 + eta * avg_local("e") + (1.0 - eta) * G.e + e_nu + e_g)
    s_total = ((1.0 - chi) * th_H.s + chi * th_Q.s
               + eta * avg_local("s") + (1.0 - eta) * G.s + s_nu + s_g)

    # sum_i mu_i n_i over both phases and every lepton species; photons have
    # mu = 0. Weighted exactly as eps and s are.
    mu_dot_n = (1.0 - chi) * th_H.mu_dot_n + chi * th_Q.mu_dot_n
    if lep:
        mu_dot_n += eta * avg_local("mu_dot_n") + (1.0 - eta) * G.mu_dot_n
    if nu is not None:
        mu_dot_n += d["mu_L"] * nu.n                      # mu_nue = mu_L

    result = MixedResult(
        converged=True, error=res_max, n_B=n_B, T=T, eta=eta, chi=chi,
        th_H=th_H, th_Q=th_Q, potentials=d, mu_B=th_H.mu_B,
        P=P_total, eps=eps_total, s=s_total, extras=extras)

    if check_consistency and result.in_mixed_phase:
        dP = (th_H.P + eta * L_H.P) - (th_Q.P + eta * L_Q.P)
        if abs(dP) > 1e-6 * max(abs(P_total), 1.0):
            raise ValueError(
                f"mechanical equilibrium violated at n_B={n_B}, eta={eta}: "
                f"dP={dP:.2e} MeV/fm^3")
        # Euler / Hugenholtz-Van Hove for the whole mixture (CLAUDE.md §7):
        # eps + P = T s + sum_i mu_i n_i.
        hvh = (eps_total + P_total - T * s_total - mu_dot_n) / eps_total
        if abs(hvh) > HVH_RTOL:
            raise ValueError(
                f"Euler / Hugenholtz-Van Hove violated at n_B={n_B}, T={T}, "
                f"eta={eta}: |{hvh:.2e}| > {HVH_RTOL:.0e} — a thermal or "
                f"lepton term is inconsistent")
    return result
