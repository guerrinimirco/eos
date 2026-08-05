"""
enjl/eos_beta.py
=================
Beta-equilibrium, charge-neutral uniform stellar matter of the extended NJL
model (Xia 2024, PRD 110, 014022 — paper Sec. III, Eqs. (23)-(24)).

Given a total baryon density n_b, the composition (p, n, Lambda, u, d, s,
e, mu) is fixed by the beta-stability condition

    mu_i = B_i mu_b - q_i mu_e                            (paper Eq. (23))

and charge neutrality

    sum_i q_i n_i = 0                                     (paper Eq. (24))

solved simultaneously with the mean-field gap and vector-field equations of
eos.enjl.uniform. The 8 coupled unknowns (M_u, M_d, M_s, mu_b, mu_e, n_bQ,
g_w, g_r) are solved with scipy.optimize.root; the fully consistent
ENJLEoSPoint for the resulting composition is then obtained from
eos.enjl.uniform.solve_point (reusing the validated uniform solver).

The vector self-consistency g_w = Gamma_w J_omega, g_r = Gamma_r J_rho is
enforced as residual equations (the densities depend on the fields and vice
versa), so no nested fixed-point iteration is needed.
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import least_squares

from eos.enjl.parameters import ENJLParams, get_enjl_default
from eos.enjl.species import BARYONS, QUARKS, LEPTONS
from eos.enjl.thermodynamics import scalar_density_t0, number_density_t0
from eos.enjl.uniform import (
    ENJLEoSPoint, _baryon_masses, _effective_scalar_densities, _f_of, _m0_of,
    _N_BARYON, quark_masses_from_gap, solve_point,
)
from eos.general.physics_constants import hc3

#: B_i (baryon number), Q_i (electric charge), degeneracy, isospin, N-sum
_B = {"p": 1.0, "n": 1.0, "Lambda": 1.0,
      "u": 1.0 / 3.0, "d": 1.0 / 3.0, "s": 1.0 / 3.0,
      "e": 0.0, "mu": 0.0}
_Q = {"p": 1.0, "n": 0.0, "Lambda": 0.0,
      "u": 2.0 / 3.0, "d": -1.0 / 3.0, "s": -1.0 / 3.0,
      "e": -1.0, "mu": -1.0}
_TAU = {"p": 1.0, "n": -1.0, "Lambda": 0.0,
        "u": 1.0, "d": -1.0, "s": 0.0, "e": 0.0, "mu": 0.0}
_G = {"p": 2.0, "n": 2.0, "Lambda": 2.0,
      "u": 6.0, "d": 6.0, "s": 6.0, "e": 2.0, "mu": 2.0}
_NSUM = {"p": 3.0, "n": 3.0, "Lambda": 3.0,
         "u": 1.0, "d": 1.0, "s": 1.0, "e": 0.0, "mu": 0.0}
_ALL = ("p", "n", "Lambda", "u", "d", "s", "e", "mu")


@dataclass(frozen=True)
class BetaPoint:
    """Beta-equilibrium uniform-matter state at a fixed total baryon density.

    ``densities`` are in fm^-3; ``eps``/``P`` in MeV/fm^3; masses and
    chemical potentials in MeV. ``pt`` is the underlying ENJLEoSPoint
    (natural units) reused for derived quantities.
    """
    n_b_fm: float
    densities: dict
    M_q: dict
    M_b: dict
    eps: float
    P: float
    mu_b: float
    mu_e: float
    pt: ENJLEoSPoint

    @property
    def EperB(self):
        return self.pt.EperB


def _evaluate(x, par, n_b_target):
    """Full self-consistent evaluation of the 10-variable state.

    The densities, vector sources and rearrangement terms are mutually
    dependent (the chemical potentials shift by SigmaR, SigmaR depends on the
    scalar densities and sources). All self-consistencies are solved
    simultaneously: the rearrangement terms SigmaR_b, SigmaR_q are solver
    unknowns and enter the chemical-potential -> density mapping directly,
    while their defining equations appear as residuals, so every quantity that
    enters a residual equation comes from the same self-consistent state.

    Returns (kF, n, M_b, M_q, n_s_b, n_s_q, nbar, J_omega, J_rho,
    SigmaR_b, SigmaR_q, residuals) in natural units.
    """
    M_u, M_d, M_s, mu_b, mu_e, n_bQ, g_w, g_r, SigmaR_b, SigmaR_q = x
    M_q = {"u": M_u, "d": M_d, "s": M_s}
    m0 = _m0_of(par)
    f = _f_of(par)

    alpha = par.alpha_S(n_b_target)
    Gw = par.Gamma_w(n_b_target)
    Gr = par.Gamma_r(n_b_target)
    dGw = par.d_Gamma_w(n_b_target)
    dGr = par.d_Gamma_r(n_b_target)
    d_alpha = par.d_alpha_S(n_b_target)

    M_b = _baryon_masses(par, M_q, alpha, n_bQ)
    m_l = {"e": par.m_e, "mu": par.m_mu}

    def mass(sp):
        if sp in BARYONS:
            return M_b[sp]
        if sp in QUARKS:
            return M_q[sp]
        return m_l[sp]

    kF = {sp: 0.0 for sp in _ALL}
    n = {sp: 0.0 for sp in _ALL}
    n_s_b = {b: 0.0 for b in BARYONS}
    n_s_q = {q: 0.0 for q in QUARKS}
    J_omega = J_rho = 0.0

    def _kF(nu, m):
        """Fermi momentum with clamps for the solver's off-track exploration.

        Physically kF is at most a few thousand MeV even at the highest
        densities used here (n_b ~ 10 fm^-3 -> kF ~ 2200 MeV), so clamping
        avoids float overflow while never binding at a real solution.
        """
        if not (math.isfinite(nu) and math.isfinite(m)) or m <= 0.0:
            return 0.0
        if nu > 5000.0:
            return 5000.0
        if nu <= m:
            return 0.0
        k2 = (nu - m) * (nu + m)
        if not math.isfinite(k2) or k2 <= 0.0:
            return 0.0
        return min(math.sqrt(k2), 5000.0)

    for sp in _ALL:
        mu_i = _B[sp] * mu_b - _Q[sp] * mu_e
        if sp in BARYONS:
            vec = f[sp] * (3.0 * g_w + _TAU[sp] * g_r) + SigmaR_b
        elif sp in QUARKS:
            vec = f[sp] * (g_w + _TAU[sp] * g_r) + SigmaR_q
        else:
            vec = 0.0
        nu = mu_i - vec
        kF[sp] = _kF(nu, mass(sp))
        n[sp] = number_density_t0(kF[sp], _G[sp])
    n_s_b = {b: scalar_density_t0(kF[b], M_b[b], 2.0, 0.0) for b in BARYONS}
    n_s_q = {q: scalar_density_t0(kF[q], M_q[q], 6.0, par.Lambda)
             for q in QUARKS}
    J_omega = sum(f[sp] * n[sp] * _NSUM[sp] for sp in _ALL)
    J_rho = sum(f[sp] * n[sp] * _TAU[sp] for sp in _ALL)
    SigmaR_wr = 0.5 * dGw * J_omega ** 2 + 0.5 * dGr * J_rho ** 2
    SigmaR_alpha = sum(
        (sum(_N_BARYON[b][qi] * (M_q[q] - m0[q])
             for qi, q in enumerate(QUARKS)) * d_alpha) * n_s_b[b]
        for b in BARYONS)
    SigmaR_b_c = SigmaR_wr + SigmaR_alpha
    SigmaR_q_c = (1.0 / 3.0) * par.B_nat * sum(n_s_b.values()) \
        + (1.0 / 3.0) * SigmaR_b_c

    # --- effective scalar densities for the gap (Eq. (6)) ---
    nbar = _effective_scalar_densities(kF, M_q, n_s_b, alpha, par.Lambda)

    # --- residuals ---
    gap = quark_masses_from_gap(nbar, par)
    res = [M_q[q] - gap[q] for q in QUARKS]
    nB = sum(_B[sp] * n[sp] for sp in _ALL)
    nQ = sum(_Q[sp] * n[sp] for sp in _ALL)
    res.append(nB - n_b_target)
    res.append(nQ)
    res.append(n_bQ - (n["u"] + n["d"] + n["s"]) / 3.0)
    res.append(g_w - Gw * J_omega)
    res.append(g_r - Gr * J_rho)
    res.append(SigmaR_b - SigmaR_b_c)
    res.append(SigmaR_q - SigmaR_q_c)

    return (kF, n, M_b, M_q, n_s_b, n_s_q, nbar,
            J_omega, J_rho, SigmaR_b_c, SigmaR_q_c, res)


def _residual(x, par, n_b_target):
    return _evaluate(x, par, n_b_target)[-1]


def _scaled_residual(x, par, n_b_target):
    """Residuals normalized to O(1) for the least-squares solver."""
    res = _evaluate(x, par, n_b_target)[-1]
    s = [100.0, 100.0, 100.0,            # quark-mass gaps [MeV]
         n_b_target, n_b_target, n_b_target,   # density constraints [MeV^3]
         par.Gamma_w(n_b_target) * 3.0 * n_b_target,      # g_w scale
         par.Gamma_r(n_b_target) * n_b_target,            # g_r scale
         3000.0, 1000.0]                                 # SigmaR scales [MeV]
    return [r / s[i] for i, r in enumerate(res)]


#: acceptance on the scaled residuals of `_scaled_residual`, which are O(1)
#: by construction, so this is a dimensionless convergence bound
BETA_TOL = 1.0e-8


def _bounds(n_b_fm):
    """Box for the ten unknowns, widened with density.

    The chemical potentials, the vector fields and the rearrangement terms all
    grow roughly linearly with n_b — mu_b reaches ~16 GeV and g_omega*omega
    ~5 GeV at n_b = 10 fm^-3 — so a box calibrated at saturation density
    excludes the solution entirely above a few times n_0. Only the quark masses
    have a genuine, density-independent ceiling (they are bounded above by
    their vacuum values) and n_b^Q a genuine one (it cannot exceed n_b).
    """
    big = 3000.0 + 3000.0 * n_b_fm
    lo = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -big, -big, -big, -big]
    hi = [1000.0, 1000.0, 1000.0, big, 2000.0, n_b_fm * hc3,
          big, big, big, big]
    return lo, hi


def _seed_ladder(x0, n_b_fm, par, cold_start=True):
    """Starting points to try, in order of decreasing prior plausibility.

    A first-order transition makes the solution discontinuous in n_b, so the
    previous point alone cannot cross one: seeded from the low-density side the
    solver is asked for a root several hundred MeV away in the quark masses,
    and lands nowhere. The second entry is the same state with the light-quark
    condensates switched off, which is where the chirally restored branch sits,
    and it is what carries a sweep across a chiral transition.

    `cold_start` adds two parameter-free starting points for use when there is
    no previous point to continue from. They must be left out once a sweep is
    under way: a cold start that happens to converge lands on whichever branch
    it lands on, so allowing it mid-sweep lets the sequence hop between
    branches from one density to the next, which shows up as an equation of
    state that oscillates rather than one that has a transition in it.
    """
    seeds = []
    if x0 is not None:
        seeds.append(list(x0))
        restored = list(x0)
        restored[0], restored[1] = par.m_u0, par.m_d0
        restored[2] = max(par.m_s0, x0[2] * 0.5)
        seeds.append(restored)
    if cold_start:
        g_w0 = par.Gamma_w(n_b_fm * hc3) * 3.0 * n_b_fm * hc3
        mu_b0 = 950.0 + 400.0 * n_b_fm
        # nucleonic: vacuum quark masses, no quarks
        seeds.append([367.6, 367.6, 549.5, mu_b0, 130.0, 0.0,
                      g_w0, -0.1 * g_w0, 0.0, 0.0])
        # quark matter: current masses, baryons dissolved
        seeds.append([par.m_u0, par.m_d0, par.m_s0 + 100.0, mu_b0, 100.0,
                      0.9 * n_b_fm * hc3, g_w0, -0.05 * g_w0, 0.0, 0.0])
    return seeds


def solve_beta_point(n_b_fm, par=None, x0=None, cold_start=True):
    """Solve beta-equilibrium, charge-neutral uniform matter at n_b [fm^-3].

    Returns *a* root of Eqs. (23)-(24), namely the first one reached from the
    starting points tried in order. Above a first-order transition there is
    more than one: the deconfined branch and the metastable baryonic branch
    both satisfy the local equations over a finite density range, and which is
    found depends on where the search began. Selecting the stable one is the
    job of `beta_eos_table`, which has the neighbouring densities needed to do
    it. Call this directly only when a single local root is what is wanted.

    Parameters:
        n_b_fm: total baryon density [fm^-3].
        par:    ENJLParams (default paper Table I + RKH).
        x0:     initial guess (M_u, M_d, M_s, mu_b, mu_e, n_bQ, g_w, g_r,
                SigmaR_b, SigmaR_q), normally the previous point of a density
                sweep. It is the first of several starting points tried, not
                the only one.

    Returns:
        BetaPoint.
    """
    if par is None:
        par = get_enjl_default()
    n_b = n_b_fm * hc3
    lo, hi = _bounds(n_b_fm)
    x_scale = [100.0, 100.0, 100.0, 100.0, 100.0, n_b_fm * hc3,
               100.0, 100.0, 3000.0, 1000.0]

    sol_x, tried, best_residual = None, 0, float("inf")
    seen = []
    for seed in _seed_ladder(x0, n_b_fm, par, cold_start=cold_start):
        seed = [min(max(v, l), h) for v, l, h in zip(seed, lo, hi)]
        if any(all(abs(a - b) <= 1e-9 * max(1.0, abs(b)) for a, b in
                   zip(seed, other)) for other in seen):
            continue                      # duplicate of a start already tried
        seen.append(seed)
        tried += 1
        sol = least_squares(lambda x: _scaled_residual(x, par, n_b), seed,
                            bounds=(lo, hi), x_scale=x_scale,
                            xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=1500)
        residual = max(abs(r) for r in _scaled_residual(sol.x, par, n_b))
        best_residual = min(best_residual, residual)
        if residual < BETA_TOL:
            sol_x = sol.x
            break
    if sol_x is None:
        raise RuntimeError(
            f"ENJL beta-equilibrium solve failed at n_b={n_b_fm:.4f} fm^-3 "
            f"after {tried} starting points; best scaled residual "
            f"{best_residual:.3e} against a {BETA_TOL:.0e} bound")

    _, n, _, _, _, _, _, _, _, _, _, _ = _evaluate(sol_x, par, n_b)

    # final consistent uniform solution via the validated solver, seeded from
    # the quark masses just found so it stays on the branch that was solved
    pt = solve_point(n, par=par, x0=[sol_x[0], sol_x[1], sol_x[2]])
    densities = {k: v / hc3 for k, v in n.items()}
    return BetaPoint(
        n_b_fm=n_b_fm,
        densities=densities,
        M_q=pt.M_q, M_b=pt.M_b,
        eps=pt.eps / hc3, P=pt.P / hc3,
        mu_b=sol.x[3], mu_e=sol.x[4], pt=pt,
    )


def _continuation_state(p):
    """The ten-vector that warm-starts the next point of a sweep."""
    return (p.M_q["u"], p.M_q["d"], p.M_q["s"], p.mu_b, p.mu_e,
            (p.densities["u"] + p.densities["d"] + p.densities["s"])
            / 3.0 * hc3, p.pt.gomega_omega, p.pt.grho_rho,
            p.pt.SigmaR_b, p.pt.SigmaR_q)


def _sweep(nb_grid, par, x0=None):
    """Continuation along `nb_grid` in the order given; {index -> BetaPoint}.

    Cold starts are allowed only until the branch is established. After that
    the sweep continues from its own previous point or not at all, so the
    result is one branch rather than a sequence that changes branch wherever a
    cold start happens to converge somewhere else. A density that cannot be
    reached from its neighbour is left out, and the sweep carries on from the
    last point that was.
    """
    out, cur, started = {}, x0, x0 is not None
    for k, nb in enumerate(nb_grid):
        try:
            p = solve_beta_point(nb, par=par, x0=cur, cold_start=not started)
        except RuntimeError:
            continue
        out[k] = p
        cur = _continuation_state(p)
        started = True
    return out


def beta_eos_table(nb_grid, par=None, x0=None, direction="up"):
    """EOS table (P, eps, n_b, composition) along one branch of a density grid.

    This is a *continuation*, not a phase diagram: each point is warm-started
    from its neighbour, so the sequence follows one branch of the model and
    keeps following it past any first-order transition, into the metastable
    region beyond. That is deliberate. Mapping a branch and choosing between
    branches are separate steps, and the second one needs both branches — a
    Maxwell construction equates P and mu_b across the two, which cannot be
    done from a single sweep. It is also what the reference tables themselves
    contain: two of them retain a step with dP/dn_b < 0 rather than the
    coexistence plateau that would replace it.

    `direction` selects which branch is followed. "up" starts from the
    low-density, chirally broken side; "down" starts at the top of the grid
    from a deconfined guess and walks back. Where only one branch exists the
    two agree; where several do, they differ, and the difference is the
    branch structure.

    Parameters:
        nb_grid:   total baryon densities [fm^-3], ascending.
        par:       ENJLParams.
        x0:        initial guess for the first point solved.
        direction: "up" (default) or "down".

    Returns:
        points: list of BetaPoint; P, eps: arrays [MeV/fm^3]. All three are
        ordered like `nb_grid`, and shortened where a density did not converge.
    """
    if par is None:
        par = get_enjl_default()
    if direction not in ("up", "down"):
        raise ValueError(f"direction must be 'up' or 'down', got {direction!r}")
    nb_grid = list(nb_grid)
    order = nb_grid if direction == "up" else nb_grid[::-1]
    solved = _sweep(order, par, x0=x0)
    if direction == "down":
        solved = {len(nb_grid) - 1 - k: p for k, p in solved.items()}
    points = [solved[k] for k in sorted(solved)]
    P = [p.P for p in points]
    eps = [p.eps for p in points]
    return points, P, eps
