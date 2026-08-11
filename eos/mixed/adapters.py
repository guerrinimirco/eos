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
from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.optimize import root

from eos.general.physics_constants import hc3
from eos.dd2.physics.octet import build_octet_ctx, assemble_octet
from eos.dd2.thermodynamics import baryon_kinetics
from eos.dd2.physics.kernel_numba import meson_sources_t0, _NUMBA_OK
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

    Returns [sigma, omega0, rho0, (phi0), nB_nat]; the density is in natural
    units, the rest in MeV.
    """
    base = solve_beta_eq_octet(par, n_B_guess, flags, T=T,
                               include_photons=False, check_consistency=False)
    x = [base.sigma, base.omega0, base.rho0]
    if flags.phi_field and flags.hyperons:
        x.append(base.phi0 if base.phi0 != 0.0 else -1.0e-3)
    x.append(n_B_guess * hc3)
    return x


def _hadronic_residual(x, ctx, par, flags, mu_tilde_B, mu_C, mu_S):
    """Meson field gaps plus baryon-density self-consistency at fixed charge
    potentials.

    Unknowns x = [sigma, omega0, rho0, (phi0), nB_nat]. The phase density is an
    unknown here — unlike a normal DD2 solve, where it is given — because in a
    mixed phase only the *average* density is prescribed. DD2's couplings are
    density-dependent, so they are re-evaluated at the current nB_nat on every
    iteration. There is no neutrality or Y_C row: mu_C is an input.

    At T=0 the per-species loop runs in the jitted `meson_sources_t0` kernel,
    which is the same closed form the NumPy path uses; T>0 keeps the NumPy path
    because the JEL integrals do not jit. The two agree to machine precision —
    see test/mixed/test_hadronic_residual_backends.py.
    """
    sigma, omega0, rho0 = x[0], x[1], x[2]
    i = 3
    phi0 = x[i] if ctx.has_phi else 0.0
    i += int(ctx.has_phi)
    nB_nat = x[i]
    if nB_nat <= 0.0:
        return [1.0e6] * len(x)

    Gs, Gw, Gr, dGs, dGw, dGr = par.couplings_at(nB_nat / hc3)
    # OctetCtx is mutable; refresh the density-dependent couplings and the
    # density itself so the reused kinetics/assembly see the current state.
    ctx.Gs_N, ctx.Gw_N, ctx.Gr_N = Gs, Gw, Gr
    ctx.dGs_N, ctx.dGw_N, ctx.dGr_N = dGs, dGw, dGr
    ctx.nB_nat = nB_nat

    if ctx.T == 0.0 and _NUMBA_OK:
        spec_arr = getattr(ctx, "_spec_arr", None)
        if spec_arr is None:
            # Built once per phase solve, not per residual evaluation.
            spec_arr = np.asarray(ctx.specs, dtype=np.float64)
            ctx._spec_arr = spec_arr
        src_s, src_w, src_r, src_phi, n_tot = meson_sources_t0(
            spec_arr, sigma, omega0, rho0, phi0, mu_tilde_B, mu_C, mu_S,
            Gs, Gw, Gr)
        if n_tot < 0.0:                              # m* <= 0: outside domain
            return [1.0e6] * len(x)
    else:
        kin = baryon_kinetics(ctx, sigma, omega0, rho0, phi0,
                               mu_tilde_B, mu_C, mu_S)
        if kin is None:                              # m* <= 0: outside domain
            return [1.0e6] * len(x)

        src_s = src_w = src_r = src_phi = n_tot = 0.0
        for (_name, spec, nu, ms, n, ns, eps, P, s) in kin:
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


def hadronic_phase(par, flags, mu_tilde_B, mu_C, mu_S=0.0, T=0.0,
                   n_B_guess=0.2, x0=None, return_state=False):
    """DD2 hadronic phase at fixed kinetic charge potentials.

    Inputs are the KINETIC baryon potential `mu_tilde_B = mu_B - Sigma^R` (which
    keeps the rearrangement self-energy and its density circularity out of the
    iteration — see CLAUDE.md §2), the non-leptonic charge potential `mu_C`
    (= mu_C) and the strangeness potential `mu_S`.

    Solves the DD-RMF meson fields and the phase's own baryon density
    self-consistently. No leptons and no neutrality: those are mixed-phase
    conditions. Returns an fm-based `PhaseThermo` with the physical
    `mu_B = mu_tilde_B + Sigma^R` restored at assembly.

    If the species flags enable a thermal meson gas, it is added at the
    converged potentials and fields exactly as `eos/dd2/solver.solve_octet`
    does: an additive ideal Bose gas contributing P, eps, s and its own
    mu*_j n_j to the chemical-potential sum.

    The gas is NOT a spectator to the charge bookkeeping. A thermal pi/K gas
    carries net electric charge and strangeness, so it enters n_C and n_S --
    and hence neutrality, the fixed-Y_C condition and the mixed-phase charge
    residuals -- not only eps, P and s. The returned n_C and n_S come from
    `assemble_octet`'s totals, which count it, matching the DD2 sector and
    CLAUDE.md section 2. At T = 40 MeV the pion gas carries about 15% of the
    non-leptonic charge, so treating it as a spectator is not a small error.

    `x0` is the starting field configuration; pass the (constant, per-solve)
    result of `hadronic_seed` rather than letting this rebuild it every call.
    It must be a *deterministic* function of the caller's fixed inputs — seeding
    from the previous trial point's converged state instead makes this function
    non-reproducible at a given set of potentials, which in turn corrupts any
    finite-difference Jacobian taken of the mixed residual.

    `return_state=True` also returns the converged internal vector, so the
    Jacobian can differentiate the phase without re-solving it.
    """
    strange_mode = "fixed" if mu_S != 0.0 else "eq"
    ctx = build_octet_ctx(par, n_B_guess, flags, T=T, charge_mode="fixed",
                          strange_mode=strange_mode, Y_C=0.0, Y_S=0.0)

    # Lazy: the fallback seed costs a full beta-equilibrium solve, so it is
    # built only if `x0` is missing or its solve does not converge.
    def guesses():
        if x0 is not None:
            yield list(x0)
        yield hadronic_seed(par, flags, T, n_B_guess)

    sol = None
    for guess in guesses():
        sol = root(_hadronic_residual, guess,
                   args=(ctx, par, flags, mu_tilde_B, mu_C, mu_S),
                   method="hybr", tol=1e-13)
        res_max = max(abs(r) for r in _hadronic_residual(
            sol.x, ctx, par, flags, mu_tilde_B, mu_C, mu_S))
        if res_max <= RESIDUAL_TOL:
            break
    else:
        raise RuntimeError(
            f"hadronic_phase solve failed at mu_tilde_B={mu_tilde_B}, "
            f"mu_C={mu_C}, mu_S={mu_S}, T={T}: {sol.message} "
            f"(max residual {res_max:.2e}, tol {RESIDUAL_TOL:.0e})")

    sigma, omega0, rho0 = sol.x[0], sol.x[1], sol.x[2]
    i = 3
    phi0 = sol.x[i] if ctx.has_phi else 0.0
    i += int(ctx.has_phi)
    nB_nat = sol.x[i]
    n_B = nB_nat / hc3

    # `_hadronic_residual` left ctx's couplings and density at the converged
    # values on its last call, so ctx is already the assembly context.
    x_oct = [sigma, omega0, rho0]
    if ctx.has_phi:
        x_oct.append(phi0)
    x_oct += [mu_tilde_B, mu_C]
    if ctx.has_muS:
        x_oct.append(mu_S)
    st = assemble_octet(x_oct, ctx)

    densities = st["densities"]
    mu_B, mu_C = st["mu_B"], st["mu_C"]
    mu_i = {b.name: mu_B * b.baryon_no + mu_C * b.charge + mu_S * b.strangeness
            for b in ctx.baryons}
    P, eps, s = st["P"] / hc3, st["eps"] / hc3, st["s"] / hc3
    mu_dot_n = st["mu_dot_n"] / hc3

    if (flags.include_pseudoscalars or flags.include_thermal_vectors) and T > 0:
        mg = thermal_meson_thermo(
            par, n_B, mu_C, mu_S, omega0, rho0, T,
            include_pseudoscalars=flags.include_pseudoscalars,
            include_thermal_vectors=flags.include_thermal_vectors)
        P += mg["P"]
        eps += mg["e"]
        s += mg["s"]
        mu_dot_n += mg["mu_dot_n"]

    th = PhaseThermo(
        densities=densities,
        n_B=n_B, n_C=st["Y_C"] * n_B, n_S=st["Y_S"] * n_B,
        P=P, eps=eps, s=s,
        mu_B=mu_B, mu_C=mu_C, mu_S=mu_S, mu_i=mu_i, mu_dot_n=mu_dot_n,
)
    if return_state:
        return th, dict(x_phase=list(sol.x), strange_mode=strange_mode, ctx=ctx)
    return th
