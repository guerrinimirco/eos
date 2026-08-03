"""
mixed/phases.py
===============
Thin adapters presenting a uniform per-phase interface over the two validated
engines (docs/phase2/SPECIFICATION_AND_PLAN.md §3.2): given the phase's
conserved-charge chemical potentials and (T, flags), return a `PhaseThermo`
block — densities, (n_B, n_C, n_S), P, eps, s, and the conserved-charge
potentials (mu_B, mu_C, mu_S) in the SAME decomposition for both phases so the
P1 solver can match them (mu_i = B_i mu_B + C_i mu_C + S_i mu_S).

There is NO eta, NO mixing, and NO charge neutrality here: those are
mixed-phase / global conditions imposed by the P1 solver, not by a single
phase. Each adapter solves only the phase's internal self-consistency. This
given-potential evaluation is the seam every later milestone drives.

Units: fm-based on the boundary (fm^-3, MeV/fm^3, MeV), per CLAUDE.md §3. The
hadronic side converts DD2's internal natural units via hc3; the quark side is
already fm-based.

Derivative blocks (spec §3.2) are declared optional with a finite-difference
fallback and are NOT implemented here — the analytic blocks are milestone P7.
"""
from dataclasses import dataclass
from typing import Mapping

from scipy.optimize import root

from eos.general.physics_constants import hc3
from eos.dd2.physics.octet import (
    build_octet_ctx, _baryon_kinetics, assemble_octet,
)
from eos.dd2.solver import solve_beta_eq_octet
from eos.vmit.parameters import get_vmit_default
from eos.vmit.thermodynamics_quarks import (
    compute_quark_matter_thermo_from_mu, compute_vmit_thermo_from_mu_n,
)

#: Post-solve residual gate for the hadronic phase solve (matches the Phase-1
#: RESIDUAL_TOL used by eos/dd2/solver.py).
RESIDUAL_TOL = 1.0e-10


@dataclass(frozen=True)
class PhaseThermo:
    """One phase's thermodynamic block, fm-based (spec §3.2).

    densities : {species name -> n [fm^-3]}
    n_B, n_C, n_S : baryon / non-leptonic-charge / strangeness density [fm^-3]
    P, eps : pressure / energy density [MeV/fm^3]  (matter only; no leptons)
    s : entropy density [fm^-3]
    mu_B, mu_C, mu_S : conserved-charge potentials [MeV]
    mu_i : {species name -> mu [MeV]}, mu_i = B_i mu_B + C_i mu_C + S_i mu_S
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


# =============================================================================
# QUARK PHASE  (vMIT)
# =============================================================================
def quark_phase(mu_u, mu_d, mu_s, T=0.0, params=None):
    """vMIT quark phase at fixed physical quark chemical potentials (spec §3.2).

    Wraps `compute_quark_matter_thermo_from_mu` (which root-solves the
    self-consistent densities, since mu_eff = mu - V(n)) and feeds the solved
    densities into `compute_vmit_thermo_from_mu_n` for the full block. Quark
    sector only: NO leptons, NO photons. mu_C = mu_u - mu_d, mu_S = mu_s - mu_d,
    mu_B = mu_u + 2 mu_d (vMIT's convention; the same conserved-charge
    decomposition the hadronic phase reports).
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
    )


# =============================================================================
# HADRONIC PHASE  (DD2)
# =============================================================================
def _hadronic_default_guess(par, flags, ctx, T, n_B_guess):
    """Physical field seed for the given-potential solve.

    Seeds sigma, omega0, rho0, (phi0) from a charge-neutral beta-equilibrium
    DD2 solve at n_B_guess — the same tactic default_octet_guess uses, and a
    guaranteed-physical starting point (m* > 0) that the pure linearized
    estimate is not at high density. It is only a seed: hybr converges from it
    to the given-potential solution. The P1 solver warm-starts and skips this.
    n_B is carried in natural units.
    """
    base = solve_beta_eq_octet(par, n_B_guess, flags, T=T,
                               include_photons=False, check_consistency=False)
    x = [base.sigma, base.omega0, base.rho0]
    if ctx.has_phi:
        x.append(base.phi0 if base.phi0 != 0.0 else -1.0e-3)
    x.append(n_B_guess * hc3)
    return x


def _hadronic_residual(x, ctx, par, flags, mu_tilde_B, mu_Q, mu_S):
    """Field gaps + baryon-density self-consistency at fixed charge potentials.

    Unknowns x = [sigma, omega0, rho0, (phi0), nB_nat]. DD2's couplings are
    density-dependent, so nB_nat is an unknown and the couplings are recomputed
    at the current density each iteration (self-consistent local density). No
    neutrality / Y_C closure row — mu_Q is an INPUT here, unlike solve_octet.
    """
    sigma, omega0, rho0 = x[0], x[1], x[2]
    i = 3
    phi0 = x[i] if ctx.has_phi else 0.0
    i += int(ctx.has_phi)
    nB_nat = x[i]
    if nB_nat <= 0.0:
        return [1.0e6] * len(x)

    Gs, Gw, Gr, dGs, dGw, dGr = par.couplings_at(nB_nat / hc3)
    # OctetCtx is mutable; refresh the density-dependent couplings + target so
    # the reused _baryon_kinetics / assemble_octet see the current density.
    ctx.Gs_N, ctx.Gw_N, ctx.Gr_N = Gs, Gw, Gr
    ctx.dGs_N, ctx.dGw_N, ctx.dGr_N = dGs, dGw, dGr
    ctx.nB_nat = nB_nat

    kin = _baryon_kinetics(ctx, sigma, omega0, rho0, phi0,
                           mu_tilde_B, mu_Q, mu_S)
    if kin is None:                                  # m* <= 0: outside domain
        return [1.0e6] * len(x)

    src_s = src_w = src_r = src_phi = n_tot = 0.0
    for (spec, nu, ms, n, ns, eps, P, s) in kin:
        _mass, _Q, t3, _g, xs, xw, xr, xphi, _S = spec
        src_s += xs * Gs * ns
        src_w += xw * Gw * n
        src_r += xr * Gr * t3 * n
        src_phi += xphi * Gw * n
        n_tot += n

    res = [
        (sigma - src_s / ctx.m_sigma2) / ctx.mbar,
        (omega0 - src_w / ctx.m_omega2) / ctx.mbar,
        (rho0 - src_r / ctx.m_rho2) / ctx.mbar,
    ]
    if ctx.has_phi:
        res.append((phi0 - src_phi / ctx.m_phi2) / ctx.mbar)
    res.append(n_tot / nB_nat - 1.0)                 # density self-consistency
    return res


def hadronic_phase(par, flags, mu_tilde_B, mu_Q, mu_S=0.0, T=0.0,
                   n_B_guess=0.2, x0=None, return_state=False):
    """DD2 hadronic phase at fixed kinetic charge potentials (spec §3.2).

    Inputs are the KINETIC baryon potential `mu_tilde_B = mu_B - Sigma^R` (the
    octet solver's natural unknown; avoids the Sigma^R circularity — CLAUDE.md
    §2), the non-leptonic charge potential `mu_Q` (= mu_C), and the strangeness
    potential `mu_S`. Solves the DD-RMF meson fields and the phase baryon
    density self-consistently; NO leptons and NO neutrality (mixed-phase
    conditions the P1 solver imposes). Returns fm-based `PhaseThermo`, with
    physical `mu_B = mu_tilde_B + Sigma^R` restored at assembly.

    Reuses eos/dd2 octet kinetics + `assemble_octet` (charge_mode='fixed',
    leptonless), so the block is the DD2 hadronic sector to round-off.

    `return_state=True` additionally returns a dict with the converged internal
    unknown vector `x_phase` (fields + nB_nat), `strange_mode`, and `ctx` — the
    pieces the analytic Jacobian (eos/mixed/jacobian.py) needs to differentiate
    the phase without re-solving it.
    """
    strange_mode = "fixed" if mu_S != 0.0 else "eq"
    ctx = build_octet_ctx(par, n_B_guess, flags, T=T, charge_mode="fixed",
                          strange_mode=strange_mode, Y_C=0.0, Y_S=0.0)

    guesses = []
    if x0 is not None:
        guesses.append(x0)
    guesses.append(_hadronic_default_guess(par, flags, ctx, T, n_B_guess))

    sol = None
    for guess in guesses:
        sol = root(_hadronic_residual, guess,
                   args=(ctx, par, flags, mu_tilde_B, mu_Q, mu_S),
                   method="hybr", tol=1e-13)
        res_max = max(abs(r) for r in _hadronic_residual(
            sol.x, ctx, par, flags, mu_tilde_B, mu_Q, mu_S))
        if res_max <= RESIDUAL_TOL:
            break
    else:
        raise RuntimeError(
            f"hadronic_phase solve failed at mu_tilde_B={mu_tilde_B}, "
            f"mu_Q={mu_Q}, mu_S={mu_S}, T={T}: {sol.message} "
            f"(max residual {res_max:.2e}, tol {RESIDUAL_TOL:.0e})")

    # Assemble via the DD2 path at the converged density (couplings frozen at
    # the solved n_B, matching the residual's last iteration).
    sigma, omega0, rho0 = sol.x[0], sol.x[1], sol.x[2]
    i = 3
    phi0 = sol.x[i] if ctx.has_phi else 0.0
    i += int(ctx.has_phi)
    nB_nat = sol.x[i]
    a_ctx = build_octet_ctx(par, nB_nat / hc3, flags, T=T, charge_mode="fixed",
                            strange_mode=strange_mode, Y_C=0.0, Y_S=0.0)
    x_oct = [sigma, omega0, rho0]
    if ctx.has_phi:
        x_oct.append(phi0)
    x_oct += [mu_tilde_B, mu_Q]
    if a_ctx.has_muS:
        x_oct.append(mu_S)
    st = assemble_octet(x_oct, a_ctx)

    n_B = nB_nat / hc3
    densities = st["densities"]
    mu_B, mu_C = st["mu_B"], st["mu_Q"]
    mu_i = {b.name: mu_B * b.baryon_no + mu_C * b.charge + mu_S * b.strangeness
            for b in ctx.baryons}
    th = PhaseThermo(
        densities=densities,
        n_B=n_B, n_C=st["Y_C"] * n_B, n_S=st["Y_S"] * n_B,
        P=st["P"] / hc3, eps=st["eps"] / hc3, s=st["s"] / hc3,
        mu_B=mu_B, mu_C=mu_C, mu_S=mu_S, mu_i=mu_i,
    )
    if return_state:
        return th, dict(x_phase=list(sol.x), strange_mode=strange_mode, ctx=ctx)
    return th
