"""
mixed/solver.py
==============
Single-point mixed-phase solve (docs/phase2/SPECIFICATION_AND_PLAN.md §3.4),
milestone P1: nucleons, Mode A (beta-equilibrium), T >= 0, the eta endpoints.

solve_mixed() drives the regime-driven residual of residual.py through MINPACK
hybr (numeric Jacobian for now; the analytic Jacobian is P7). It warm-starts
from the pure hadronic beta-equilibrium solution at the same n_B. The result
carries both phase blocks, the eta-split electron potentials, and the total
(matter + leptons + photons) thermodynamics of the mixed phase.

chi outside (0, 1) means the trial density is not in the mixed window (clamp to
the pure phase upstream); the P1 tests locate the window by scanning n_B.
"""
from dataclasses import dataclass, field

from scipy.optimize import root

from eos.general.thermodynamics_leptons import (
    electron_thermo, neutrino_thermo, photon_thermo,
)
from eos.dd2.solver import solve_beta_eq_octet
from eos.mixed.phases import PhaseThermo
from eos.mixed.residual import (
    build_mixed_ctx, mixed_residual, evaluate_phases, has_leptons,
)

#: Post-solve residual gate (Phase-1 RESIDUAL_TOL).
RESIDUAL_TOL = 1.0e-10


@dataclass
class MixedResult:
    converged: bool
    error: float
    n_B: float
    T: float
    eta: float
    chi: float
    th_H: PhaseThermo
    th_Q: PhaseThermo
    potentials: dict            # solved unknown-vector slots
    mu_B: float                 # matched physical baryon potential
    P: float                    # total pressure [MeV/fm^3] (matter+leptons+photons)
    eps: float                  # total energy density [MeV/fm^3]
    s: float                    # total entropy density [fm^-3]
    extras: dict = field(default_factory=dict)

    @property
    def in_mixed_phase(self):
        return 0.0 < self.chi < 1.0


def _default_guess(ctx):
    """Warm start from each phase's OWN pure beta-equilibrium point at n_B.

    The hadronic side sets mu_tilde_B (kinetic) and its electron potential; the
    quark side (solved separately) sets mu_B_Q and its electron potential. This
    matches the eta=1 Maxwell structure (each phase separately neutral) far
    better than a shared-mu_e seed, and is a fine seed at eta=0 too. chi seeded
    mid-window. Quark seed falls back to the hadronic mu_B if vMIT does not
    converge at this density.
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
        # Mode C charge potential(s): the beta-eq value mu_C = -mu_e is a fine seed.
        "mu_C_H": -base.mu_e,
        "mu_C_Q": -mu_eL_Q,
        "mu_C": -base.mu_e,
        # Mode D strangeness potential: self-equilibrating value mu_S = 0.
        "mu_S": 0.0,
        # Mode B trapped-neutrino potential: neutrino-transparent value mu_L = 0.
        "mu_L": 0.0,
    }
    return [seed[name] for name in ctx.slots]


def solve_mixed(par, flags, n_B, eta, spec, vmit_params=None, T=0.0,
                x0=None, n_B_guess=None, check_consistency=True):
    """Solve the mixed phase at (n_B, T, eta) for regime assignment `spec`.

    Returns a MixedResult. Raises RuntimeError if the residual gate is not met
    from any guess (no silent non-convergence). `x0` is an optional warm-start
    vector (slot order of residual.mixed_slots(spec, eta)); otherwise the pure
    hadronic beta-eq seed is used.
    """
    if vmit_params is None:
        from eos.vmit.parameters import get_vmit_default
        vmit_params = get_vmit_default()
    ctx = build_mixed_ctx(spec, eta, n_B, par, flags, vmit_params, T=T,
                          n_B_guess=n_B_guess)

    guesses = []
    if x0 is not None:
        guesses.append(list(x0))
    guesses.append(_default_guess(ctx))

    sol = None
    for guess in guesses:
        sol = root(mixed_residual, guess, args=(ctx,), method="hybr", tol=1e-12)
        res_max = max(abs(r) for r in mixed_residual(sol.x, ctx))
        if res_max <= RESIDUAL_TOL:
            break
    else:
        raise RuntimeError(
            f"mixed solve failed at n_B={n_B}, T={T}, eta={eta}: {sol.message} "
            f"(max residual {res_max:.2e}, tol {RESIDUAL_TOL:.0e})")

    th_H, th_Q, d, extras = evaluate_phases(sol.x, ctx)
    chi = d["chi"]

    # Total thermodynamics: matter phases volume-averaged; electrons are
    # eta[(1-chi)e_L^H + chi e_L^Q] + (1-eta) e_G; photons phase-common (once).
    # Pressure is uniform = matched matter+local-electron pressure + global e + g.
    lep = has_leptons(spec)
    ph = photon_thermo(T) if T > 0.0 else None
    P_g = ph.P if ph else 0.0
    e_g = ph.e if ph else 0.0
    s_g = ph.s if ph else 0.0
    eL_H = electron_thermo(d.get("mu_eL_H", 0.0), T) if (lep and eta > 0.0) else None
    eL_Q = electron_thermo(d.get("mu_eL_Q", 0.0), T) if (lep and eta > 0.0) else None
    eG = electron_thermo(d.get("mu_eG", 0.0), T) if (lep and eta < 1.0) else None

    def avg_e(a):    # volume-averaged local-electron attribute
        return ((1.0 - chi) * getattr(eL_H, a) + chi * getattr(eL_Q, a)
                if eL_H is not None else 0.0)
    eG_e = (lambda a: getattr(eG, a) if eG is not None else 0.0)

    # trapped neutrinos (Mode B): global, uniform — add once like photons.
    nu = neutrino_thermo(d["mu_L"], T) if "mu_L" in d else None
    P_nu = nu.P if nu else 0.0
    e_nu = nu.e if nu else 0.0
    s_nu = nu.s if nu else 0.0

    P_total = (th_H.P + eta * extras["P_eL_H"]
               + (1.0 - eta) * eG_e("P") + P_nu + P_g)
    eps_total = ((1.0 - chi) * th_H.eps + chi * th_Q.eps
                 + eta * avg_e("e") + (1.0 - eta) * eG_e("e") + e_nu + e_g)
    s_total = ((1.0 - chi) * th_H.s + chi * th_Q.s
               + eta * avg_e("s") + (1.0 - eta) * eG_e("s") + s_nu + s_g)

    # sum_i mu_i n_i over both phases (baryons) + the eta-split electrons; photons
    # carry mu=0. Volume-weighted to match eps/s.
    munH = sum(th_H.mu_i[n] * dn for n, dn in th_H.densities.items())
    munQ = sum(th_Q.mu_i[n] * dn for n, dn in th_Q.densities.items())
    mu_dot_n = (1.0 - chi) * munH + chi * munQ
    if lep and eta > 0.0:
        mu_dot_n += eta * ((1.0 - chi) * d["mu_eL_H"] * extras["n_eL_H"]
                           + chi * d["mu_eL_Q"] * extras["n_eL_Q"])
    if lep and eta < 1.0:
        mu_dot_n += (1.0 - eta) * d["mu_eG"] * extras["n_eG"]
    if "mu_L" in d:                                      # trapped neutrinos
        mu_dot_n += d["mu_L"] * extras["n_nue"]         # mu_nue = mu_L

    result = MixedResult(
        converged=True, error=res_max, n_B=n_B, T=T, eta=eta, chi=chi,
        th_H=th_H, th_Q=th_Q, potentials=d, mu_B=th_H.mu_B,
        P=P_total, eps=eps_total, s=s_total, extras=extras)

    if check_consistency and result.in_mixed_phase:
        # mechanical equilibrium: matter + eta local-electron pressures match.
        dP = ((th_H.P + eta * extras["P_eL_H"])
              - (th_Q.P + eta * extras["P_eL_Q"]))
        if abs(dP) > 1e-6 * max(abs(P_total), 1.0):
            raise ValueError(
                f"mechanical equilibrium violated at n_B={n_B}, eta={eta}: "
                f"dP={dP:.2e}")
        # Euler / Hugenholtz-Van Hove for the whole mixed phase (CLAUDE.md §7):
        # eps + P = T s + sum_i mu_i n_i, to Phase-1's HVH tolerance.
        hvh = (eps_total + P_total - T * s_total - mu_dot_n) / eps_total
        if abs(hvh) > 1e-8:
            raise ValueError(
                f"Euler/HVH violated at n_B={n_B}, T={T}, eta={eta}: "
                f"|{hvh:.2e}| > 1e-8 (a thermal or electron term is inconsistent)")
    return result
