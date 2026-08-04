"""
physics/kernel_numba.py
====================
Numba-jitted T=0 octet residual and analytic Jacobian — the eos_fast hot path
. Mirrors octet_residual / octet_jacobian
exactly at T=0, where the kinetic thermodynamics are closed form and jittable.

T>0 is NOT jitted: the kinetic densities come from the scipy JEL integrals,
which Numba cannot trace (report D3). solve_octet therefore routes T=0 through
these kernels and falls back to the NumPy analytic path at T>0. Correctness is
guarded by backend parity (kernel_numba vs octet.py) in the M9 gate.

Array layout (built once per solve by build_numba_arrays):
    spec   : (N, 9) [mass, Q, t3, g, x_sigma, x_omega, x_rho, x_phi, S]
    params : [Gs_N, Gw_N, Gr_N, m_s2, m_w2, m_r2, m_p2, nB_nat, mbar,
              m_e, m_mu, Y_C, Y_S, Y_L]
    flags  : [has_phi, has_muS, has_muL, charge_neutral, include_muons]  (int)
"""
import numpy as np

try:
    from numba import njit
    _NUMBA_OK = True
except ImportError:                       # pragma: no cover - numba is a dep
    _NUMBA_OK = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def deco(f):
            return f
        return deco


_PI2 = np.pi ** 2


@njit(cache=True)
def _n_ns_t0(nu, m, g):
    """(n, ns) at T=0 [natural units]; massless has no scalar density."""
    kF2 = nu * nu - m * m
    if kF2 <= 0.0 or nu <= 0.0:
        return 0.0, 0.0
    kF = np.sqrt(kF2)
    n = g * kF * kF2 / (6.0 * _PI2)
    if m == 0.0:
        return n, 0.0
    ns = (g * m / (4.0 * _PI2)) * (kF * nu - m * m * np.log((kF + nu) / m))
    return n, ns


@njit(cache=True)
def _derivs_t0(nu, m, g):
    """(dn/dnu, dn/dm, dns/dnu, dns/dm) at T=0 (closed form)."""
    kF2 = nu * nu - m * m
    if kF2 <= 0.0 or nu <= 0.0:
        return 0.0, 0.0, 0.0, 0.0
    kF = np.sqrt(kF2)
    if m == 0.0:
        return g * nu * nu / (2.0 * _PI2), 0.0, 0.0, 0.0
    A = g * kF * nu / (2.0 * _PI2)
    B = -g * kF * m / (2.0 * _PI2)
    D = (g / (4.0 * _PI2)) * (kF * nu - 3.0 * m * m * np.log((kF + nu) / m))
    return A, B, -B, D


@njit(cache=True)
def residual_t0_jit(x, spec, params, flags):
    """T=0 octet residual (mirrors octet.octet_residual)."""
    Gs_N, Gw_N, Gr_N = params[0], params[1], params[2]
    m_s2, m_w2, m_r2, m_p2 = params[3], params[4], params[5], params[6]
    nB_nat, mbar, m_e, m_mu = params[7], params[8], params[9], params[10]
    Y_C, Y_S, Y_L = params[11], params[12], params[13]
    has_phi, has_muS, has_muL = flags[0], flags[1], flags[2]
    neutral, inc_mu = flags[3], flags[4]
    N = spec.shape[0]
    n_unk = 5 + has_phi + has_muS + has_muL

    sigma, omega0, rho0 = x[0], x[1], x[2]
    idx = 3
    phi0 = x[idx] if has_phi == 1 else 0.0
    idx += has_phi
    mutB, muQ = x[idx], x[idx + 1]
    idx += 2
    muS = x[idx] if has_muS == 1 else 0.0
    idx += has_muS
    muL = x[idx] if has_muL == 1 else 0.0

    res = np.empty(n_unk)
    src_s = src_w = src_r = src_phi = 0.0
    n_tot = charge = strange = 0.0
    for i in range(N):
        mass, Q, t3, g = spec[i, 0], spec[i, 1], spec[i, 2], spec[i, 3]
        xs, xw, xr, xphi, S = (spec[i, 4], spec[i, 5], spec[i, 6],
                               spec[i, 7], spec[i, 8])
        ms = mass - xs * Gs_N * sigma
        if ms <= 0.0:
            for k in range(n_unk):
                res[k] = 1.0e6
            return res
        nu = (mutB + Q * muQ + S * muS - xw * Gw_N * omega0
              - xr * Gr_N * t3 * rho0 - xphi * Gw_N * phi0)
        n, ns = _n_ns_t0(nu, ms, g)
        src_s += xs * Gs_N * ns
        src_w += xw * Gw_N * n
        src_r += xr * Gr_N * t3 * n
        src_phi += xphi * Gw_N * n
        n_tot += n
        charge += Q * n
        strange += S * n

    res[0] = (sigma - src_s / m_s2) / mbar
    res[1] = (omega0 - src_w / m_w2) / mbar
    res[2] = (rho0 - src_r / m_r2) / mbar
    row = 3
    if has_phi == 1:
        res[row] = (phi0 - src_phi / m_p2) / mbar
        row += 1
    res[row] = n_tot / nB_nat - 1.0
    row += 1
    if neutral == 1:
        n_e = _n_ns_t0(muL - muQ, m_e, 2.0)[0]
        n_mu = _n_ns_t0(-muQ, m_mu, 2.0)[0] if inc_mu == 1 else 0.0
        res[row] = (charge - n_e - n_mu) / nB_nat
    else:
        res[row] = charge / nB_nat - Y_C
    row += 1
    if has_muS == 1:
        res[row] = strange / nB_nat - Y_S
        row += 1
    if has_muL == 1:
        n_e = _n_ns_t0(muL - muQ, m_e, 2.0)[0]
        n_nue = _n_ns_t0(muL, 0.0, 1.0)[0]
        res[row] = (n_e + n_nue) / nB_nat - Y_L
    return res


@njit(cache=True)
def jacobian_t0_jit(x, spec, params, flags):
    """T=0 analytic Jacobian dR/dx (mirrors jacobian.octet_jacobian)."""
    Gs_N, Gw_N, Gr_N = params[0], params[1], params[2]
    m_s2, m_w2, m_r2, m_p2 = params[3], params[4], params[5], params[6]
    nB_nat, mbar, m_e, m_mu = params[7], params[8], params[9], params[10]
    has_phi, has_muS, has_muL = flags[0], flags[1], flags[2]
    neutral, inc_mu = flags[3], flags[4]
    N = spec.shape[0]
    n_unk = 5 + has_phi + has_muS + has_muL

    c_phi = 3
    c_mutB = 3 + has_phi
    c_muQ = c_mutB + 1
    c_muS = c_muQ + 1
    c_muL = c_muQ + 1 + has_muS

    sigma, omega0, rho0 = x[0], x[1], x[2]
    phi0 = x[c_phi] if has_phi == 1 else 0.0
    mutB, muQ = x[c_mutB], x[c_muQ]
    muS = x[c_muS] if has_muS == 1 else 0.0
    muL = x[c_muL] if has_muL == 1 else 0.0

    J = np.zeros((n_unk, n_unk))
    dsrc_s = np.zeros(n_unk)
    dsrc_w = np.zeros(n_unk)
    dsrc_r = np.zeros(n_unk)
    dsrc_phi = np.zeros(n_unk)
    dn_tot = np.zeros(n_unk)
    dcharge = np.zeros(n_unk)
    dstrange = np.zeros(n_unk)

    for i in range(N):
        mass, Q, t3, g = spec[i, 0], spec[i, 1], spec[i, 2], spec[i, 3]
        xs, xw, xr, xphi, S = (spec[i, 4], spec[i, 5], spec[i, 6],
                               spec[i, 7], spec[i, 8])
        ms = mass - xs * Gs_N * sigma
        if ms <= 0.0:
            J2 = np.zeros((n_unk, n_unk))
            for k in range(n_unk):
                J2[k, k] = 1.0
            return J2
        nu = (mutB + Q * muQ + S * muS - xw * Gw_N * omega0
              - xr * Gr_N * t3 * rho0 - xphi * Gw_N * phi0)
        A, B, C, D = _derivs_t0(nu, ms, g)

        dnu = np.zeros(n_unk)
        dnu[1] = -xw * Gw_N
        dnu[2] = -xr * Gr_N * t3
        if has_phi == 1:
            dnu[c_phi] = -xphi * Gw_N
        dnu[c_mutB] = 1.0
        dnu[c_muQ] = Q
        if has_muS == 1:
            dnu[c_muS] = S
        dm = np.zeros(n_unk)
        dm[0] = -xs * Gs_N

        for k in range(n_unk):
            dn_ik = A * dnu[k] + B * dm[k]
            dns_ik = C * dnu[k] + D * dm[k]
            dsrc_s[k] += xs * Gs_N * dns_ik
            dsrc_w[k] += xw * Gw_N * dn_ik
            dsrc_r[k] += xr * Gr_N * t3 * dn_ik
            dsrc_phi[k] += xphi * Gw_N * dn_ik
            dn_tot[k] += dn_ik
            dcharge[k] += Q * dn_ik
            dstrange[k] += S * dn_ik

    for k in range(n_unk):
        J[0, k] = -dsrc_s[k] / m_s2 / mbar
        J[1, k] = -dsrc_w[k] / m_w2 / mbar
        J[2, k] = -dsrc_r[k] / m_r2 / mbar
    J[0, 0] += 1.0 / mbar
    J[1, 1] += 1.0 / mbar
    J[2, 2] += 1.0 / mbar
    row = 3
    if has_phi == 1:
        for k in range(n_unk):
            J[row, k] = -dsrc_phi[k] / m_p2 / mbar
        J[row, c_phi] += 1.0 / mbar
        row += 1
    for k in range(n_unk):
        J[row, k] = dn_tot[k] / nB_nat
    row += 1
    if neutral == 1:
        Ae = _derivs_t0(muL - muQ, m_e, 2.0)[0]
        Amu = _derivs_t0(-muQ, m_mu, 2.0)[0] if inc_mu == 1 else 0.0
        for k in range(n_unk):
            J[row, k] = dcharge[k] / nB_nat
        J[row, c_muQ] += (Ae + Amu) / nB_nat        # -(dn_e+dn_mu) at muQ
        if has_muL == 1:
            J[row, c_muL] += (-Ae) / nB_nat
    else:
        for k in range(n_unk):
            J[row, k] = dcharge[k] / nB_nat
    row += 1
    if has_muS == 1:
        for k in range(n_unk):
            J[row, k] = dstrange[k] / nB_nat
        row += 1
    if has_muL == 1:
        Ae = _derivs_t0(muL - muQ, m_e, 2.0)[0]
        Anue = _derivs_t0(muL, 0.0, 1.0)[0]
        J[row, c_muQ] = -Ae / nB_nat
        J[row, c_muL] = (Ae + Anue) / nB_nat
    return J


def build_numba_arrays(ctx):
    """Flat (spec, params, flags) arrays from an OctetCtx (built once/solve)."""
    spec = np.array(ctx.specs, dtype=np.float64)
    params = np.array([ctx.Gs_N, ctx.Gw_N, ctx.Gr_N, ctx.m_sigma2, ctx.m_omega2,
                       ctx.m_rho2, ctx.m_phi2, ctx.nB_nat, ctx.mbar, ctx.m_e,
                       ctx.m_mu, ctx.Y_C, ctx.Y_S, ctx.Y_L], dtype=np.float64)
    flags = np.array([int(ctx.has_phi), int(ctx.has_muS), int(ctx.has_muL),
                      int(ctx.charge_mode == "neutral"), int(ctx.include_muons)],
                     dtype=np.int64)
    return spec, params, flags
