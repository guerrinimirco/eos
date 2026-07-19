"""
physics/jacobian.py
====================
Hand-coded analytic Jacobian of the octet residual (report §3.0/§3.5, M9).

This is the "second backend" derivative engine: the report's JAX autodiff path
is deferred (the JEL/Bose core and the T=0 threshold kink do not trace cleanly,
report D3), so the exact solver Jacobian is supplied analytically instead. It
makes the octet solve a true Newton step (dense J via MINPACK hybrj) and gives
M10 the analytic thermodynamic derivatives (susceptibilities, exact c_s^2)
without table finite-differencing.

Kinetic derivatives d{n,ns}/d{nu,m*} are closed-form at T=0 and a per-species
finite difference of kinetic_thermo at T>0 (the JEL routine exposes no exact
derivative; differencing one species is far cleaner than differencing the whole
residual). Everything else — the field-equation structure, the chain rule
m*(sigma) / nu(fields, potentials), and the algebraic lepton/constraint rows —
is analytic.

Kernel kept in the xp namespace so a Numba/JAX wrapper is a drop-in later
(profiling-driven phase 2; eos_ref is sufficient now).
"""
import numpy as np

from eos.dd2.xp import xp
from eos.dd2.physics.thermo import kinetic_thermo

_PI2 = np.pi ** 2


def kinetic_derivs(nu, m, g, T=0.0):
    """
    (dn/dnu, dn/dm*, dns/dnu, dns/dm*) for one fermion species [natural units].

    T=0: closed form (validated to ~1e-9 vs finite difference). Note the
    Maxwell-like symmetry dns/dnu = -dn/dm* = g kF m*/(2 pi^2). Massless
    (neutrino) has no scalar response.
    """
    if T == 0.0:
        kF2 = nu * nu - m * m
        if kF2 <= 0.0 or nu <= 0.0:
            return 0.0, 0.0, 0.0, 0.0
        kF = xp.sqrt(kF2)
        if m == 0.0:
            return g * nu * nu / (2.0 * _PI2), 0.0, 0.0, 0.0
        dn_dnu = g * kF * nu / (2.0 * _PI2)
        dn_dm = -g * kF * m / (2.0 * _PI2)
        L = xp.log((kF + nu) / m)          # EF = nu at T=0
        dns_dm = (g / (4.0 * _PI2)) * (kF * nu - 3.0 * m * m * L)
        return dn_dnu, dn_dm, -dn_dm, dns_dm
    # T>0: central difference of the JEL kinetic thermo (per species).
    hn = 1e-3 * max(abs(nu), 1.0)
    hm = 1e-3 * max(m, 1.0)
    np1 = kinetic_thermo(nu + hn, m, g, T)
    nm1 = kinetic_thermo(nu - hn, m, g, T)
    mp1 = kinetic_thermo(nu, m + hm, g, T)
    mm1 = kinetic_thermo(nu, m - hm, g, T)
    dn_dnu = (np1[0] - nm1[0]) / (2.0 * hn)
    dns_dnu = (np1[4] - nm1[4]) / (2.0 * hn)
    dn_dm = (mp1[0] - mm1[0]) / (2.0 * hm)
    dns_dm = (mp1[4] - mm1[4]) / (2.0 * hm)
    return dn_dnu, dn_dm, dns_dnu, dns_dm


def _columns(ctx):
    """Column index of each unknown (matches octet._unpack ordering)."""
    c = dict(sigma=0, omega0=1, rho0=2)
    i = 3
    if ctx.has_phi:
        c["phi0"] = i
        i += 1
    c["mutB"], c["muQ"] = i, i + 1
    i += 2
    if ctx.has_muS:
        c["muS"] = i
        i += 1
    if ctx.has_muL:
        c["muL"] = i
    return c


def octet_jacobian(x, ctx):
    """
    Analytic dR/dx of octet_residual (dense, n_unknowns x n_unknowns).

    Row/column order matches octet_residual / _unpack. Returns np.eye as a
    benign fallback when x is outside the physical domain (m*<=0), where the
    residual itself returns its 1e6 sentinel and MINPACK damps the step.
    """
    from eos.dd2.physics.octet import _unpack, _baryon_kinetics

    n = ctx.n_unknowns
    sigma, omega0, rho0, phi0, mutB, muQ, muS, muL = _unpack(x, ctx)
    kin = _baryon_kinetics(ctx, sigma, omega0, rho0, phi0, mutB, muQ, muS)
    if kin is None:
        return np.eye(n)

    col = _columns(ctx)
    J = np.zeros((n, n))

    # accumulators: gradients (over columns) of the residual source sums
    dsrc_s = np.zeros(n)   # sum xs Gs_N dns
    dsrc_w = np.zeros(n)   # sum xw Gw_N dn
    dsrc_r = np.zeros(n)   # sum xr Gr_N t3 dn
    dsrc_phi = np.zeros(n)  # sum xphi Gw_N dn
    dn_tot = np.zeros(n)
    dcharge = np.zeros(n)
    dstrange = np.zeros(n)

    for (spec, nu, ms, n_i, ns_i, eps_i, P_i, s_i) in kin:
        mass, Q, t3, g, xs, xw, xr, xphi, S = spec
        A, B, C, D = kinetic_derivs(nu, ms, g, ctx.T)   # dn/dnu,dn/dm,dns/dnu,dns/dm

        dnu = np.zeros(n)
        dnu[col["omega0"]] = -xw * ctx.Gw_N
        dnu[col["rho0"]] = -xr * ctx.Gr_N * t3
        if ctx.has_phi:
            dnu[col["phi0"]] = -xphi * ctx.Gw_N
        dnu[col["mutB"]] = 1.0
        dnu[col["muQ"]] = Q
        if ctx.has_muS:
            dnu[col["muS"]] = S
        dm = np.zeros(n)
        dm[col["sigma"]] = -xs * ctx.Gs_N

        dn_i = A * dnu + B * dm
        dns_i = C * dnu + D * dm

        dsrc_s += xs * ctx.Gs_N * dns_i
        dsrc_w += xw * ctx.Gw_N * dn_i
        dsrc_r += xr * ctx.Gr_N * t3 * dn_i
        dsrc_phi += xphi * ctx.Gw_N * dn_i
        dn_tot += dn_i
        dcharge += Q * dn_i
        dstrange += S * dn_i

    # field-equation rows (scaled by mbar), R = (field - src/m^2)/mbar
    row = 0
    e = lambda k: np.eye(n)[col[k]]
    J[row] = (e("sigma") - dsrc_s / ctx.m_sigma2) / ctx.mbar
    J[row + 1] = (e("omega0") - dsrc_w / ctx.m_omega2) / ctx.mbar
    J[row + 2] = (e("rho0") - dsrc_r / ctx.m_rho2) / ctx.mbar
    row += 3
    if ctx.has_phi:
        J[row] = (e("phi0") - dsrc_phi / ctx.m_phi2) / ctx.mbar
        row += 1

    # baryon number
    J[row] = dn_tot / ctx.nB_nat
    row += 1

    # charge row
    if ctx.charge_mode == "neutral":
        mu_e = muL - muQ
        Ae = kinetic_derivs(mu_e, ctx.m_e, 2.0, ctx.T)[0]
        dn_e = np.zeros(n)
        dn_e[col["muQ"]] = -Ae
        if ctx.has_muL:
            dn_e[col["muL"]] = Ae
        dn_mu = np.zeros(n)
        if ctx.include_muons:
            Amu = kinetic_derivs(-muQ, ctx.m_mu, 2.0, ctx.T)[0]
            dn_mu[col["muQ"]] = -Amu
        J[row] = (dcharge - dn_e - dn_mu) / ctx.nB_nat
    else:
        J[row] = dcharge / ctx.nB_nat
    row += 1

    # strangeness row
    if ctx.has_muS:
        J[row] = dstrange / ctx.nB_nat
        row += 1

    # lepton (Y_L) row
    if ctx.has_muL:
        Ae = kinetic_derivs(muL - muQ, ctx.m_e, 2.0, ctx.T)[0]
        Anue = kinetic_derivs(muL, 0.0, 1.0, ctx.T)[0]
        dY = np.zeros(n)
        dY[col["muQ"]] = -Ae
        dY[col["muL"]] = Ae + Anue
        J[row] = dY / ctx.nB_nat

    return J
