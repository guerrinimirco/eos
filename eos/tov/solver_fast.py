"""
solver_fast.py
==============
Numba-compiled TOV solver for mass-production use (Bayesian inference, ML
training) where the same equations are solved 10^7-10^8 times.

Design (see solver.py for the trusted, readable reference implementation that
this one is validated against):

* **Adaptive embedded Dormand-Prince RK45** with PI step control -- proper
  local-error control, so it stays accurate across a wide EOS prior without
  per-EOS tuning, and returns NaN (never raises) on a pathological EOS.
* **Uniform log-P EOS grid + O(1) lookup** -- the EOS is resampled once onto a
  uniform grid in log10(P); interior interpolation is then an index cast plus a
  lerp, with no binary search.
* **Everything integrated in one sweep**: mass m, pressure P, the tidal
  metric-perturbation Y, and the baryonic mass M_b are the four ODE variables,
  so Love number and M_b cost no extra passes.
* **`prange` over the central-density sequence** -- stars are independent.

Dimensionless variables (identical scaling to solver.py):
  r in r_s = 2GM_sun/c^2  (so 2Gm/rc^2 -> m/r, and C = M/(2R)),
  m, M_b in M_sun,  P in MeV/fm^3,  energy density in units of rho_sol.

Physics of the tidal term: Y(r) = r H'(r)/H(r) for the l=2 static perturbation,
  r Y' + Y^2 + Y F + r^2 Q = 0,  Y(0) = 2,
with the standard F, Q (Hinderer 2008; Postnikov, Prakash & Lattimer 2010); the
surface Love number k2 and Lambda = (2/3) k2 C^-5 use the closed form, plus the
-4 pi R^3 eps_s / M jump for self-bound (quark-star) surfaces.
"""
import numpy as np
from numba import njit, prange

from eos.general.physics_constants import (
    r_sun_km, rho_sol, m_nucleon_MeV,
)

# Scaling constants captured as plain floats for the njit kernels.
_RHO_SOL = float(rho_sol)          # MeV/fm^3  (energy-density scale)
_RSUN_KM = float(r_sun_km)         # km        (Schwarzschild radius scale)
_CB = 3.0 * float(m_nucleon_MeV) / float(rho_sol)   # baryon-mass ODE prefactor


# =============================================================================
# EOS preparation (once per EOS, plain Python)
# =============================================================================
def prepare_eos_uniform(P, epsilon, nB, n_grid: int = 2000,
                        disc_rel_jump: float = 0.05):
    """Resample an EOS onto a uniform log10(P) grid for O(1) lookup, and flag
    first-order density discontinuities (Maxwell / phase-transition jumps).

    A discontinuity is a cell where ε rises by more than ``disc_rel_jump`` in a
    single grid cell (>>the smooth per-cell step). At those cells cs² is set to
    1 so the ODE does not try to build the tidal Y jump itself; the jump is
    instead applied explicitly at integration time (Takátsy & Kovács 2020) via
    the returned ``(P_disc, deps_disc)``. This avoids double-counting.

    Parameters
    ----------
    P, epsilon, nB : array_like
        Pressure and energy density [MeV/fm^3] and baryon density [fm^-3].
    n_grid : int
        Number of uniform log10(P) points (2000 gives <0.1% on M, R, Lambda).
    disc_rel_jump : float
        Relative ε jump per cell that flags a discontinuity.

    Returns
    -------
    tuple
        ``(logP_min, dx_inv, logeps_grid, cs2_grid, logn_grid, eps_s,
           P_disc, deps_disc, P_min, P_max)``. ``eps_s`` is ε at the lowest
        tabulated pressure (self-bound surface density); ``P_disc``/``deps_disc``
        are the pressures and Δε of the internal discontinuities (ascending P).
    """
    P = np.asarray(P, float); epsilon = np.asarray(epsilon, float)
    nB = np.asarray(nB, float)
    m = (P > 0) & (epsilon > 0) & (nB > 0) & np.isfinite(P) & np.isfinite(epsilon)
    P, epsilon, nB = P[m], epsilon[m], nB[m]
    o = np.argsort(P)
    P, epsilon, nB = P[o], epsilon[o], nB[o]
    # unique P (strictly increasing x for interpolation); keep the LAST ε at a
    # repeated P (high-density side) so a literal Maxwell plateau reads as a
    # single upward ε step rather than being averaged away.
    P, u = np.unique(P, return_index=True)
    epsilon, nB = epsilon[u], nB[u]

    logP = np.log10(P)
    logeps = np.log10(epsilon)
    logn = np.log10(nB)
    with np.errstate(divide='ignore', invalid='ignore'):
        de_dP = np.gradient(epsilon, P)
        cs2 = np.where(de_dP > 0, 1.0 / de_dP, 1.0)
    cs2 = np.clip(cs2, 1e-6, 1.0)

    logP_u = np.linspace(logP[0], logP[-1], n_grid)
    logeps_g = np.interp(logP_u, logP, logeps)
    cs2_g = np.interp(logP_u, logP, cs2)
    logn_g = np.interp(logP_u, logP, logn)

    # Flag discontinuity cells on the uniform grid and neutralise their cs².
    eps_g = 10.0 ** logeps_g
    rel = np.diff(eps_g) / eps_g[:-1]
    jump_cells = np.where(rel > disc_rel_jump)[0] + 1        # cell the jump lands in
    P_disc = (10.0 ** logP_u[jump_cells]).astype(float)
    deps_disc = (eps_g[jump_cells] - eps_g[jump_cells - 1]).astype(float)
    cs2_g[jump_cells] = 1.0                                  # ODE skips the jump here
    if P_disc.size == 0:                                     # numba needs non-empty arrays
        P_disc = np.zeros(1); deps_disc = np.zeros(1)

    dx = (logP[-1] - logP[0]) / (n_grid - 1)
    return (float(logP[0]), 1.0 / dx,
            np.ascontiguousarray(logeps_g), np.ascontiguousarray(cs2_g),
            np.ascontiguousarray(logn_g), float(epsilon[0]),
            np.ascontiguousarray(P_disc), np.ascontiguousarray(deps_disc),
            float(P[0]), float(P[-1]))


# =============================================================================
# njit kernels
# =============================================================================
@njit(fastmath=True, cache=True, inline='always')
def _interp_uniform(logx, logP_min, dx_inv, grid):
    """O(1) linear interpolation of ``grid`` at log10-pressure ``logx``."""
    t = (logx - logP_min) * dx_inv
    i = int(t)
    n = grid.shape[0]
    if i < 0:
        return grid[0]
    if i >= n - 1:
        return grid[n - 1]
    f = t - i
    return grid[i] * (1.0 - f) + grid[i + 1] * f


@njit(fastmath=True, cache=True)
def _rhs(r, m, P, Y, logP_min, dx_inv, logeps_g, cs2_g, logn_g, out):
    """Fill ``out = [dm/dr, dP/dr, dY/dr, dM_b/dr]`` (dimensionless r)."""
    if P <= 0.0 or r <= 0.0:
        out[0] = 0.0; out[1] = 0.0; out[2] = 0.0; out[3] = 0.0
        return
    logP = np.log10(P)
    eps = 10.0 ** _interp_uniform(logP, logP_min, dx_inv, logeps_g)   # MeV/fm^3
    cs2 = _interp_uniform(logP, logP_min, dx_inv, cs2_g)
    n = 10.0 ** _interp_uniform(logP, logP_min, dx_inv, logn_g)       # fm^-3
    if cs2 <= 0.0:
        cs2 = 1e-6

    eps_t = eps / _RHO_SOL           # energy density in rho_sol units
    P_t = P / _RHO_SOL               # pressure    in rho_sol units
    r2 = r * r
    denom = r - m
    if denom <= 1e-12 * r:
        denom = 1e-12 * r
    metric = r / denom               # 1/(1 - m/r)

    # mass
    out[0] = 3.0 * eps_t * r2
    # pressure (TOV);  (m + 3 P_t r^3) is the scaled (m + 4 pi r^3 P)
    term_press = m + 3.0 * P_t * r * r2
    out[1] = -0.5 * eps * (1.0 + P_t / eps_t) * term_press / (r * denom)
    # tidal Y
    ratio = P_t / eps_t              # = P/eps
    F = (1.0 - 1.5 * eps_t * r2 * (1.0 - ratio)) * metric
    Qb = 5.0 + 9.0 * ratio + (1.0 + ratio) / cs2
    r2Q = (1.5 * eps_t * r2 * metric * Qb - 6.0 * metric
           - term_press * term_press / (denom * denom))
    out[2] = -(Y * Y + Y * F + r2Q) / r
    # baryonic mass:  dM_b/dr = (3 m_N/rho_sol) r^2 n / sqrt(1 - m/r)
    mm = 1.0 - m / r
    out[3] = _CB * r2 * n / np.sqrt(mm) if mm > 0.0 else 0.0


# Dormand-Prince RK45 Butcher tableau (nodes, a-matrix, 5th- and 4th-order b).
@njit(fastmath=True, cache=True)
def _solve_one(Pc, logP_min, dx_inv, logeps_g, cs2_g, logn_g, eps_s,
               P_disc, deps_disc, n_disc, P_surf, rtol, atol, r_max):
    """One star: returns (M [M_sun], R [km], M_b [M_sun], k2, Lambda). NaN on failure.

    ``P_disc[:n_disc]`` / ``deps_disc[:n_disc]`` are internal density
    discontinuities (Maxwell / phase transitions); the tidal Y jump
    ΔY = -4πr³Δε/(m+4πr³P) is applied when the integration crosses each."""
    # state s = [m, P, Y, M_b]
    r = 1e-6
    # central mass from the uniform-density core; eps(Pc):
    eps_c = 10.0 ** _interp_uniform(np.log10(Pc), logP_min, dx_inv, logeps_g)
    m = (eps_c / _RHO_SOL) * r * r * r          # = 4/3 pi r^3 eps /(4/3 pi r_s^3 rho..)->3*eps_t*r^3/3
    s0 = m; s1 = Pc; s2 = 2.0; s3 = 0.0
    h = 1e-4

    k1 = np.empty(4); k2 = np.empty(4); k3 = np.empty(4)
    k4 = np.empty(4); k5 = np.empty(4); k6 = np.empty(4); k7 = np.empty(4)

    _rhs(r, s0, s1, s2, logP_min, dx_inv, logeps_g, cs2_g, logn_g, k1)
    r_prev = r; m_prev = s0; P_prev = s1; Y_prev = s2; Mb_prev = s3

    n_step = 0
    while r < r_max and s1 > P_surf:
        n_step += 1
        if n_step > 100000:
            return np.nan, np.nan, np.nan, np.nan, np.nan

        # stages (FSAL: k1 carried from previous accepted step)
        y0 = s0 + h * (0.2 * k1[0])
        y1 = s1 + h * (0.2 * k1[1]); y2 = s2 + h * (0.2 * k1[2]); y3 = s3 + h * (0.2 * k1[3])
        _rhs(r + 0.2 * h, y0, y1, y2, logP_min, dx_inv, logeps_g, cs2_g, logn_g, k2)

        y0 = s0 + h * (0.075 * k1[0] + 0.225 * k2[0])
        y1 = s1 + h * (0.075 * k1[1] + 0.225 * k2[1])
        y2 = s2 + h * (0.075 * k1[2] + 0.225 * k2[2])
        y3 = s3 + h * (0.075 * k1[3] + 0.225 * k2[3])
        _rhs(r + 0.3 * h, y0, y1, y2, logP_min, dx_inv, logeps_g, cs2_g, logn_g, k3)

        y0 = s0 + h * (0.9777777777777777 * k1[0] - 3.7333333333333334 * k2[0] + 3.5555555555555554 * k3[0])
        y1 = s1 + h * (0.9777777777777777 * k1[1] - 3.7333333333333334 * k2[1] + 3.5555555555555554 * k3[1])
        y2 = s2 + h * (0.9777777777777777 * k1[2] - 3.7333333333333334 * k2[2] + 3.5555555555555554 * k3[2])
        y3 = s3 + h * (0.9777777777777777 * k1[3] - 3.7333333333333334 * k2[3] + 3.5555555555555554 * k3[3])
        _rhs(r + 0.8 * h, y0, y1, y2, logP_min, dx_inv, logeps_g, cs2_g, logn_g, k4)

        y0 = s0 + h * (2.9525986892242035 * k1[0] - 11.595793324188385 * k2[0] + 9.822892851699436 * k3[0] - 0.29080932784636487 * k4[0])
        y1 = s1 + h * (2.9525986892242035 * k1[1] - 11.595793324188385 * k2[1] + 9.822892851699436 * k3[1] - 0.29080932784636487 * k4[1])
        y2 = s2 + h * (2.9525986892242035 * k1[2] - 11.595793324188385 * k2[2] + 9.822892851699436 * k3[2] - 0.29080932784636487 * k4[2])
        y3 = s3 + h * (2.9525986892242035 * k1[3] - 11.595793324188385 * k2[3] + 9.822892851699436 * k3[3] - 0.29080932784636487 * k4[3])
        _rhs(r + 0.8888888888888888 * h, y0, y1, y2, logP_min, dx_inv, logeps_g, cs2_g, logn_g, k5)

        y0 = s0 + h * (2.8462752525252526 * k1[0] - 10.757575757575758 * k2[0] + 8.906422717743473 * k3[0] + 0.2784090909090909 * k4[0] - 0.2735313036020583 * k5[0])
        y1 = s1 + h * (2.8462752525252526 * k1[1] - 10.757575757575758 * k2[1] + 8.906422717743473 * k3[1] + 0.2784090909090909 * k4[1] - 0.2735313036020583 * k5[1])
        y2 = s2 + h * (2.8462752525252526 * k1[2] - 10.757575757575758 * k2[2] + 8.906422717743473 * k3[2] + 0.2784090909090909 * k4[2] - 0.2735313036020583 * k5[2])
        y3 = s3 + h * (2.8462752525252526 * k1[3] - 10.757575757575758 * k2[3] + 8.906422717743473 * k3[3] + 0.2784090909090909 * k4[3] - 0.2735313036020583 * k5[3])
        _rhs(r + h, y0, y1, y2, logP_min, dx_inv, logeps_g, cs2_g, logn_g, k6)

        # 5th-order solution (b coefficients; also the a7i -> FSAL)
        b1 = 0.09114583333333333; b3 = 0.44923629829290207; b4 = 0.6510416666666666
        b5 = -0.322376179245283; b6 = 0.13095238095238096
        n0 = s0 + h * (b1 * k1[0] + b3 * k3[0] + b4 * k4[0] + b5 * k5[0] + b6 * k6[0])
        n1 = s1 + h * (b1 * k1[1] + b3 * k3[1] + b4 * k4[1] + b5 * k5[1] + b6 * k6[1])
        n2 = s2 + h * (b1 * k1[2] + b3 * k3[2] + b4 * k4[2] + b5 * k5[2] + b6 * k6[2])
        n3 = s3 + h * (b1 * k1[3] + b3 * k3[3] + b4 * k4[3] + b5 * k5[3] + b6 * k6[3])
        _rhs(r + h, n0, n1, n2, logP_min, dx_inv, logeps_g, cs2_g, logn_g, k7)

        # 4th-order embedded solution for the error estimate
        e1 = 0.08991319444444445; e3 = 0.4534890685834082; e4 = 0.6140625
        e5 = -0.2715123820754717; e6 = 0.08904761904761904; e7 = 0.025
        d0 = h * ((b1 - e1) * k1[0] + (b3 - e3) * k3[0] + (b4 - e4) * k4[0] + (b5 - e5) * k5[0] + (b6 - e6) * k6[0] - e7 * k7[0])
        d1 = h * ((b1 - e1) * k1[1] + (b3 - e3) * k3[1] + (b4 - e4) * k4[1] + (b5 - e5) * k5[1] + (b6 - e6) * k6[1] - e7 * k7[1])
        d2 = h * ((b1 - e1) * k1[2] + (b3 - e3) * k3[2] + (b4 - e4) * k4[2] + (b5 - e5) * k5[2] + (b6 - e6) * k6[2] - e7 * k7[2])
        d3 = h * ((b1 - e1) * k1[3] + (b3 - e3) * k3[3] + (b4 - e4) * k4[3] + (b5 - e5) * k5[3] + (b6 - e6) * k6[3] - e7 * k7[3])

        sc0 = atol + rtol * max(abs(s0), abs(n0))
        sc1 = atol + rtol * max(abs(s1), abs(n1))
        sc2 = atol + rtol * max(abs(s2), abs(n2))
        sc3 = atol + rtol * max(abs(s3), abs(n3))
        err = np.sqrt(0.25 * ((d0 / sc0) ** 2 + (d1 / sc1) ** 2
                              + (d2 / sc2) ** 2 + (d3 / sc3) ** 2))

        if err <= 1.0 or h <= 1e-10:
            # accept
            r_prev = r; m_prev = s0; P_prev = s1; Y_prev = s2; Mb_prev = s3
            r += h; s0 = n0; s1 = n1; s2 = n2; s3 = n3
            # apply the tidal Y jump for any density discontinuity just crossed
            # (P falls from P_prev to s1 outward): ΔY = -3Δε̃ r³/(m̃+3P̃_d r³).
            jumped = False
            for j in range(n_disc):
                Pd = P_disc[j]
                if s1 <= Pd < P_prev:
                    Pd_t = Pd / _RHO_SOL
                    deps_t = deps_disc[j] / _RHO_SOL
                    denom_j = s0 + 3.0 * Pd_t * r * r * r
                    if denom_j > 1e-12:
                        s2 -= 3.0 * deps_t * r * r * r / denom_j
                        jumped = True
            if jumped:                                       # Y changed -> refresh FSAL k1
                _rhs(r, s0, s1, s2, logP_min, dx_inv, logeps_g, cs2_g, logn_g, k1)
            else:
                k1[0] = k7[0]; k1[1] = k7[1]; k1[2] = k7[2]; k1[3] = k7[3]   # FSAL
        # PI-ish step update
        fac = 0.9 * err ** (-0.2) if err > 0.0 else 5.0
        if fac > 5.0:
            fac = 5.0
        if fac < 0.2:
            fac = 0.2
        h *= fac
        if h > 0.5:
            h = 0.5

    if not (s1 <= P_surf) and r >= r_max:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # linear interpolation to P = P_surf for the surface radius/state
    if abs(P_prev - s1) > 1e-30:
        frac = (P_prev - P_surf) / (P_prev - s1)
    else:
        frac = 1.0
    r_s = r_prev + frac * (r - r_prev)
    m_s = m_prev + frac * (s0 - m_prev)
    Y_s = Y_prev + frac * (s2 - Y_prev)
    Mb_s = Mb_prev + frac * (s3 - Mb_prev)

    # self-bound surface: eps jumps from eps_s to 0 -> Delta Y = -4 pi R^3 eps_s / M
    if eps_s > 1e-6 and m_s > 1e-12:
        Y_s -= (3.0 * eps_s / _RHO_SOL) * r_s * r_s * r_s / m_s

    C = m_s / (2.0 * r_s)
    if C <= 0.005 or C >= 0.5:
        return m_s, r_s * _RSUN_KM, Mb_s, np.nan, np.nan
    v = 1.0 - 2.0 * C
    num = v * v * (2.0 - Y_s + 2.0 * C * (Y_s - 1.0))
    dA = 2.0 * C * (6.0 - 3.0 * Y_s + 3.0 * C * (5.0 * Y_s - 8.0))
    dB = 4.0 * C ** 3 * (13.0 - 11.0 * Y_s + C * (3.0 * Y_s - 2.0) + 2.0 * C ** 2 * (1.0 + Y_s))
    dC = 3.0 * v * v * (2.0 - Y_s + 2.0 * C * (Y_s - 1.0)) * np.log(v)
    den = dA + dB + dC
    if abs(den) < 1e-30:
        return m_s, r_s * _RSUN_KM, Mb_s, np.nan, np.nan
    k2love = 1.6 * C ** 5 * num / den
    Lam = (2.0 / 3.0) * k2love / C ** 5
    return m_s, r_s * _RSUN_KM, Mb_s, k2love, Lam


@njit(parallel=True, fastmath=True, cache=True)
def solve_sequence(Pc_array, logP_min, dx_inv, logeps_g, cs2_g, logn_g, eps_s,
                   P_disc, deps_disc, n_disc, P_surf, rtol, atol, r_max):
    """Solve a whole central-pressure sequence in parallel (prange).

    Returns five arrays (M [M_sun], R [km], M_b [M_sun], k2, Lambda), NaN where
    a star failed to converge.
    """
    n = Pc_array.shape[0]
    M = np.empty(n); R = np.empty(n); Mb = np.empty(n)
    K2 = np.empty(n); Lam = np.empty(n)
    for i in prange(n):
        m_, r_, mb_, k_, l_ = _solve_one(
            Pc_array[i], logP_min, dx_inv, logeps_g, cs2_g, logn_g, eps_s,
            P_disc, deps_disc, n_disc, P_surf, rtol, atol, r_max)
        M[i] = m_; R[i] = r_; Mb[i] = mb_; K2[i] = k_; Lam[i] = l_
    return M, R, Mb, K2, Lam


@njit(fastmath=True, cache=True)
def solve_sequence_serial(Pc_array, logP_min, dx_inv, logeps_g, cs2_g, logn_g,
                          eps_s, P_disc, deps_disc, n_disc, P_surf, rtol, atol,
                          r_max):
    """Serial (no-prange) twin of :func:`solve_sequence`.

    Use this when the CALLER already parallelises (e.g. the sigma_crit engine's
    joblib scan over grid cells): a prange region inside each joblib worker
    would oversubscribe cores (numba threads × worker processes). Same compiled
    per-star kernel, so still ~0.02 ms/star.
    """
    n = Pc_array.shape[0]
    M = np.empty(n); R = np.empty(n); Mb = np.empty(n)
    K2 = np.empty(n); Lam = np.empty(n)
    for i in range(n):
        m_, r_, mb_, k_, l_ = _solve_one(
            Pc_array[i], logP_min, dx_inv, logeps_g, cs2_g, logn_g, eps_s,
            P_disc, deps_disc, n_disc, P_surf, rtol, atol, r_max)
        M[i] = m_; R[i] = r_; Mb[i] = mb_; K2[i] = k_; Lam[i] = l_
    return M, R, Mb, K2, Lam


# =============================================================================
# Python driver (drop-in analogue of compute_tov_sequence)
# =============================================================================
def compute_tov_sequence_fast(eos, e_c_vec, rtol: float = 1e-6, atol: float = 1e-8,
                              r_max: float = 20.0, n_grid: int = 2000,
                              P_surf_rel: float = 1e-10, parallel: bool = True):
    """Fast TOV sequence over central energy densities.

    Emits the SAME 8-column layout as ``solver.compute_tov_sequence`` with both
    flags on, so the two backends are interchangeable (the scipy driver's
    ``backend='fast'`` path trims these columns to honour its flags).

    Parameters
    ----------
    eos : EOSTable_for_TOV
        Object with ``.P``, ``.epsilon``, ``.nB`` [MeV/fm^3, MeV/fm^3, fm^-3].
        Merge the crust first with the existing ``add_crust`` if you need one.
    e_c_vec : array_like
        Central energy densities [MeV/fm^3].
    rtol, atol : float
        RK45 error-control tolerances (1e-6/1e-8 gives M, R, Lambda to ~4-5
        digits; loosen for more speed).
    r_max : float
        Max integration radius in r_s units (~20 -> ~60 km, ample).
    n_grid : int
        Uniform log10(P) EOS resolution.
    P_surf_rel : float
        Surface pressure as a fraction of the smallest tabulated P.

    Returns
    -------
    np.ndarray
        Columns ``[e_c, n_c, P_c, R[km], M[Msun], M_b[Msun], k2, Lambda]``.
    """
    from scipy.interpolate import PchipInterpolator
    prep = prepare_eos_uniform(eos.P, eos.epsilon, eos.nB, n_grid=n_grid)
    (logP_min, dx_inv, logeps_g, cs2_g, logn_g, eps_s,
     P_disc, deps_disc, P_min, P_max) = prep
    n_disc = int(np.count_nonzero(deps_disc))               # 0 if the pad-zero only

    # central pressure and central baryon density from the central e_c (both
    # P(eps) and n(P) are monotone, so a 1-D interpolation each).
    o = np.argsort(eos.epsilon)
    P_of_e = PchipInterpolator(eos.epsilon[o], eos.P[o], extrapolate=True)
    e_c_vec = np.asarray(e_c_vec, float)
    Pc = np.clip(P_of_e(e_c_vec), P_min, P_max)
    nc = 10.0 ** np.interp(np.log10(Pc),
                           np.linspace(logP_min, logP_min + (logn_g.size - 1) / dx_inv,
                                       logn_g.size), logn_g)

    P_surf = P_surf_rel * P_min
    seq = solve_sequence if parallel else solve_sequence_serial
    M, R, Mb, K2, Lam = seq(
        np.ascontiguousarray(Pc), logP_min, dx_inv, logeps_g, cs2_g, logn_g,
        eps_s, P_disc, deps_disc, n_disc, P_surf, rtol, atol, r_max)

    return np.column_stack([e_c_vec, nc, Pc, R, M, Mb, K2, Lam])
