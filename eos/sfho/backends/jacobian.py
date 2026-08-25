"""
backends/jacobian.py
====================
Hand-coded analytic Jacobian of the SFHo residual.

The exact solver Jacobian, supplied analytically because automatic
differentiation does not survive this integrand: the JEL/Bose cores and the
T = 0 threshold kink do not trace cleanly. It turns the mean-field solve into a
true Newton step (dense J through MINPACK's hybrj) and gives the thermodynamic
derivatives -- the susceptibility matrix chi_ab = dn_a/dmu_b, the exact sound
speed -- without differencing a table.

Everything analytic except the two places where the underlying integral exposes
no derivative:

    kinetic d{n,ns}/d{mu_eff,m*}    closed form at T = 0 (the JEL routine drops
                                    to the same exact expressions there), a
                                    per-species central difference at T > 0
    thermal meson gas               a central difference of its (n_C, n_S) in
                                    the four unknowns it depends on

Differencing one species, or one gas block, is far cleaner than differencing
the whole residual: the field-equation structure, the chain rule
m*(sigma) / mu_eff(fields, potentials), and the algebraic lepton rows are all
written out.

Units are the module's own throughout (CLAUDE.md section 5): densities fm^-3,
potentials and fields MeV, so a kinetic derivative is fm^-3 MeV^-1 and a source
derivative is multiplied by (hbar c)^3 to reach the MeV^3 of a field equation.

NOT differentiated here: the entropy row of an isentropic solve, and with it
the temperature unknown. `solver.py` sends those to the finite-difference
reference path instead. `backends/` is deletable (CLAUDE.md section 5): remove
this directory and every number is unchanged, only slower.

References:
- Fortin, Oertel, Providencia, PASA 35 (2018) e044
- Steiner, Hempel, Fischer, ApJ 774 (2013) 17
"""
import numpy as np

from eos.general.physics_constants import hc3
from eos.general.particles import Electron, Neutrino
from eos.general.fermi_integrals import solve_fermi_jel

_PI2 = np.pi ** 2

#: Below this the JEL routine evaluates the exact degenerate expressions
#: (`eos.general.fermi_integrals`), so the closed-form derivatives below are
#: the derivatives of what the residual actually computed.
_T0 = 1.0e-4


def kinetic_derivs(mu_eff, T, m, g):
    """(dn/dmu_eff, dn/dm*, dns/dmu_eff, dns/dm*) for one fermion species.

    Densities in fm^-3, potentials and masses in MeV, so every entry is
    fm^-3 MeV^-1.

    At T = 0 these are closed forms of the degenerate gas, with the
    antiparticle branch (mu_eff < -m) carried by |mu_eff| and the sign of
    mu_eff, exactly as `solve_fermi_t0` builds n itself. Note the
    Maxwell-like symmetry dns/dmu_eff = -dn/dm*: both are
    sign(mu) g kF m*/(2 pi^2 (hbar c)^3). A massless species (the neutrino)
    has no scalar response at all.

    At T > 0 the JEL routine exposes no exact derivative, so this is a central
    difference of that one species.
    """
    if T < _T0:
        mu_abs = abs(mu_eff)
        if mu_abs <= m:
            return 0.0, 0.0, 0.0, 0.0
        kF = np.sqrt(mu_abs * mu_abs - m * m)
        sign = 1.0 if mu_eff > 0.0 else -1.0
        if m == 0.0:
            return g * mu_abs * mu_abs / (2.0 * _PI2 * hc3), 0.0, 0.0, 0.0
        dn_dmu = g * kF * mu_abs / (2.0 * _PI2 * hc3)
        dn_dm = -sign * g * kF * m / (2.0 * _PI2 * hc3)
        L = np.log((kF + mu_abs) / m)
        dns_dm = (g / (4.0 * _PI2 * hc3)) * (kF * mu_abs - 3.0 * m * m * L)
        return dn_dmu, dn_dm, -dn_dm, dns_dm

    hmu = 1e-5 * max(abs(mu_eff), 1.0)
    hm = 1e-5 * max(m, 1.0)
    up_mu = solve_fermi_jel(mu_eff + hmu, T, m, g)
    dn_mu = solve_fermi_jel(mu_eff - hmu, T, m, g)
    up_m = solve_fermi_jel(mu_eff, T, m + hm, g)
    dn_m = solve_fermi_jel(mu_eff, T, m - hm, g)
    return ((up_mu[0] - dn_mu[0]) / (2.0 * hmu),
            (up_m[0] - dn_m[0]) / (2.0 * hm),
            (up_mu[4] - dn_mu[4]) / (2.0 * hmu),
            (up_m[4] - dn_m[4]) / (2.0 * hm))


def _d2A(par, sigma, omega):
    """(d2A/dsigma2, d2A/domega2) of A(sigma,omega) [MeV^0 and MeV^-2 x MeV^2].

    A = g_rhoN^2 [sum_i a_i sigma^i + sum_j b_j omega^2j] is SEPARABLE, so the
    mixed second derivative is identically zero and does not appear.
    """
    d2_sigma = 0.0
    for i in range(2, len(par.a_coeffs)):
        d2_sigma += i * (i - 1) * par.a_coeffs[i] * sigma ** (i - 2)
    d2_omega = 0.0
    for j in range(1, len(par.b_coeffs)):
        k = 2 * j
        d2_omega += k * (k - 1) * par.b_coeffs[j] * omega ** (k - 2)
    return par.g_rho_N ** 2 * d2_sigma, par.g_rho_N ** 2 * d2_omega


def columns(sys):
    """Column index of each unknown, matching `solver._unpack`."""
    c = dict(sigma=0, omega=1, rho=2, phi=3, mu_B=4, mu_C=5)
    i = 6
    if sys.spec.is_fixed("S"):
        c["mu_S"] = i
        i += 1
    if sys.spec.is_fixed("L_e"):
        c["mu_nue"] = i
        i += 1
    return c


def _dgas(sys, col, n, omega, rho, mu_C, mu_S, which):
    """Gradient over the unknown columns of the thermal gas's n_C / n_S.

    `which` is 0 for n_C and 1 for n_S. Central differences in the four
    unknowns the gas depends on -- it enters mu*_j through mu_C, mu_S and the
    two vector fields and nothing else (`thermodynamics.meson_potentials`).
    Identically zero when the gas is off, so the caller can add it blind.
    """
    from eos.sfho.thermodynamics import thermal_meson_thermo

    d = np.zeros(n)
    T = sys.T
    if not sys.thermal_mesons or T is None or T <= 0.0:
        return d
    key = "n_C" if which == 0 else "n_S"
    args = dict(mu_C=mu_C, mu_S=mu_S, omega=omega, rho=rho)
    knobs = ["mu_C", "omega", "rho"]
    if sys.spec.is_fixed("S"):
        knobs.append("mu_S")
    for name in knobs:
        h = max(1e-3, 1e-4 * abs(args[name]))
        hi, lo = dict(args), dict(args)
        hi[name] += h
        lo[name] -= h
        d[col[name]] = (
            thermal_meson_thermo(T, hi["mu_C"], hi["mu_S"], hi["omega"],
                                 hi["rho"], sys.params)[key]
            - thermal_meson_thermo(T, lo["mu_C"], lo["mu_S"], lo["omega"],
                                   lo["rho"], sys.params)[key]) / (2.0 * h)
    return d


def residual_jacobian(x, sys):
    """Analytic dR/dx of `solver.residual` (dense, n_unknowns x n_unknowns).

    Row and column order match `solver.residual` and `solver._unpack`. The
    caller is responsible for not asking on an isentropic solve, where the
    entropy row and the temperature unknown are not differentiated here.
    """
    from eos.sfho.solver import _unpack, unknown_names, FIELD_SCALE

    par, spec = sys.params, sys.spec
    sigma, omega, rho, phi, mu_B, mu_C, mu_S, mu_nue, T = _unpack(x, sys)
    n = len(unknown_names(sys))
    col = columns(sys)
    J = np.zeros((n, n))

    # Gradients (over the unknown columns) of the four field sources and of the
    # three conserved-charge densities. Accumulated over the active baryons.
    dsrc_sigma = np.zeros(n)
    dsrc_omega = np.zeros(n)
    dsrc_rho = np.zeros(n)
    dsrc_phi = np.zeros(n)
    dn_B = np.zeros(n)
    dn_C = np.zeros(n)
    dn_S = np.zeros(n)

    for p in sys.particles:
        g_s = par.get_coupling(p.name, 'sigma')
        g_w = par.get_coupling(p.name, 'omega')
        g_r = par.get_coupling(p.name, 'rho')
        g_p = par.get_coupling(p.name, 'phi')
        m_baryon = par.get_baryon_mass(p.name) or p.mass

        # `baryon_thermo` floors a negative effective mass at 1e-3 MeV; where
        # it does, m* has stopped depending on sigma and so has this row.
        m_eff = m_baryon - g_s * sigma
        dm_dsigma = -g_s
        if m_eff < 0:
            m_eff = 1e-3
            dm_dsigma = 0.0

        mu_eff = (p.baryon_no * mu_B + p.charge * mu_C + p.strangeness * mu_S
                  - (g_w * omega + g_r * p.isospin_3 * rho + g_p * phi))

        dn_dmu, dn_dm, dns_dmu, dns_dm = kinetic_derivs(
            mu_eff, T, m_eff, p.g_degen)

        dmu_eff = np.zeros(n)
        dmu_eff[col["omega"]] = -g_w
        dmu_eff[col["rho"]] = -g_r * p.isospin_3
        dmu_eff[col["phi"]] = -g_p
        dmu_eff[col["mu_B"]] = p.baryon_no
        dmu_eff[col["mu_C"]] = p.charge
        if spec.is_fixed("S"):
            dmu_eff[col["mu_S"]] = p.strangeness
        dm = np.zeros(n)
        dm[col["sigma"]] = dm_dsigma

        dn_i = dn_dmu * dmu_eff + dn_dm * dm
        dns_i = dns_dmu * dmu_eff + dns_dm * dm

        dsrc_sigma += g_s * dns_i
        dsrc_omega += g_w * dn_i
        dsrc_rho += g_r * p.isospin_3 * dn_i
        dsrc_phi += g_p * dn_i
        dn_B += p.baryon_no * dn_i
        dn_C += p.charge * dn_i
        dn_S += p.strangeness * dn_i

    # The thermal pi/K/eta gas carries charge and strangeness but no baryon
    # number, so it enters those two rows only.
    dn_C += _dgas(sys, col, n, omega, rho, mu_C, mu_S, 0)
    dn_S += _dgas(sys, col, n, omega, rho, mu_C, mu_S, 1)

    # ---------------------------------------------------------- field rows
    # Each is (LHS(fields) - src (hbar c)^3) / (m^2 FIELD_SCALE); the LHS
    # derivatives below are the self-interaction terms of `field_residuals`.
    A_sigma = par.compute_dA_dsigma(sigma)
    A_omega = par.compute_dA_domega(omega)
    A = par.compute_A(sigma, omega)
    d2A_sigma, d2A_omega = _d2A(par, sigma, omega)

    unit = np.eye(n)
    row_sigma = -hc3 * dsrc_sigma
    row_sigma += (par.m_sigma ** 2 + 2.0 * par.g2 * sigma
                  + 3.0 * par.g3 * sigma ** 2
                  - d2A_sigma * rho ** 2) * unit[col["sigma"]]
    row_sigma += (-2.0 * A_sigma * rho) * unit[col["rho"]]
    J[0] = row_sigma / (par.m_sigma ** 2 * FIELD_SCALE)

    row_omega = -hc3 * dsrc_omega
    row_omega += (par.m_omega ** 2 + 3.0 * par.c3 * omega ** 2
                  + d2A_omega * rho ** 2) * unit[col["omega"]]
    row_omega += (2.0 * A_omega * rho) * unit[col["rho"]]
    J[1] = row_omega / (par.m_omega ** 2 * FIELD_SCALE)

    row_rho = -hc3 * dsrc_rho
    row_rho += (par.m_rho ** 2 + 3.0 * par.c4 * rho ** 2
                + 2.0 * A) * unit[col["rho"]]
    row_rho += (2.0 * A_sigma * rho) * unit[col["sigma"]]
    row_rho += (2.0 * A_omega * rho) * unit[col["omega"]]
    J[2] = row_rho / (par.m_rho ** 2 * FIELD_SCALE)

    row_phi = -hc3 * dsrc_phi + par.m_phi ** 2 * unit[col["phi"]]
    J[3] = row_phi / (par.m_phi ** 2 * FIELD_SCALE)

    # ------------------------------------------------------ conserved rows
    J[4] = dn_B
    row = 5

    if spec.is_fixed("C"):
        J[row] = dn_C
    else:
        # Electric neutrality, n_C = n_e, with mu_e = mu_nue - mu_C.
        mu_e = mu_nue - mu_C
        dne_dmu = kinetic_derivs(mu_e, T, Electron.mass, Electron.g_degen)[0]
        dn_e = np.zeros(n)
        dn_e[col["mu_C"]] = -dne_dmu
        if spec.is_fixed("L_e"):
            dn_e[col["mu_nue"]] = dne_dmu
        J[row] = dn_C - dn_e
    row += 1

    if spec.is_fixed("S"):
        J[row] = dn_S
        row += 1

    if spec.is_fixed("L_e"):
        mu_e = mu_nue - mu_C
        dne_dmu = kinetic_derivs(mu_e, T, Electron.mass, Electron.g_degen)[0]
        dnu_dmu = kinetic_derivs(mu_nue, T, Neutrino.mass,
                                 Neutrino.g_degen)[0]
        dY = np.zeros(n)
        dY[col["mu_C"]] = -dne_dmu
        dY[col["mu_nue"]] = dne_dmu + dnu_dmu
        J[row] = dY / sys.n_B

    return J
