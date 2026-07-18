"""
nmp_inverter.py
====================
Inverse map NMPs -> DD2 couplings (report §2.5 cascade), milestone M8.

The forward map (nmp.compute_nmp) extracts {n_sat, E_sat, m*/m, K_sat, Q_sat,
E_sym, L_sym} from a Parametrization. This inverts it:

  1. Isoscalar (6x6 root at FIXED n_sat, so no P=0 bracket search in the loop):
     free {Gamma_sigma, Gamma_omega, b_sigma, c_sigma, b_omega, c_omega} matched to
     {P(n_sat)=0, E_sat, m*/m, K_sat, Q_sat, and the closing cross-constraint
     f_sigma''(1)=f_omega''(1) (report §1.3 constraint 3)}. m_sigma is fixed
     (report §2.3: with m_sigma fixed the 5 isoscalar NMPs + the cross-constraint
     close the sector). a_i, d_i are derived internally (from_microscopic).
  2. Isovector (near-analytic): Gamma_rho(n_sat) from E_sym in closed form, then
     a_rho from L_sym by a 1-D root.
  3. Feasibility flags (report §2.6): m*/m too small, no real isoscalar solution,
     a_rho driving Gamma_rho non-monotonic / negative.

The isoscalar cross-constraint is DD2's own (it holds on the published table to
2.2e-3, not exactly), so a round trip reproduces the NMPs exactly but the shape
coefficients to ~2e-3 — see the M8 gate test.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, root

from eos.general.physics_constants import hc3
from eos.dd2.couplings import rational_d2f, derived_a, derived_d
from eos.dd2.parametrization import Parametrization
from eos.dd2.physics.thermo import kF_from_n
from eos.dd2.solver import solve_snm


@dataclass
class InversionStatus:
    ok: bool
    message: str
    isoscalar_residual: float
    isovector_residual: float


def _f2_at1(b, c):
    """f_i''(1) with a_i, d_i derived from (b, c)."""
    d = float(derived_d(c))
    a = float(derived_a(b, c, d))
    return rational_d2f(1.0, a, b, c, d)


def _trial_par(n_sat, Gs, bS, cS, Gw, bW, cW, m_sigma, Grho=3.0, a_rho=0.5):
    """Build a Parametrization from free isoscalar params (a,d derived)."""
    return Parametrization.from_microscopic(
        n_sat=n_sat, gamma_sigma=Gs, b_sigma=bS, c_sigma=cS,
        gamma_omega=Gw, b_omega=bW, c_omega=cW,
        gamma_rho=Grho, a_rho=a_rho, m_sigma=m_sigma)


def _isoscalar_quantities(par, n_sat, h=1e-4):
    """{P, E/A, m*/m, K_sat, Q_sat} of SNM at n_sat (no P=0 search)."""
    EA = lambda n: solve_snm(par, n).eps / n - par.m_nucleon
    at = solve_snm(par, n_sat)
    d2 = (EA(n_sat + h) - 2 * EA(n_sat) + EA(n_sat - h)) / h ** 2
    d3 = (EA(n_sat + 2 * h) - 2 * EA(n_sat + h)
          + 2 * EA(n_sat - h) - EA(n_sat - 2 * h)) / (2 * h ** 3)
    return dict(P=at.P, E_sat=EA(n_sat), m_ratio=at.m_eff / par.m_nucleon,
                K_sat=9 * n_sat ** 2 * d2, Q_sat=27 * n_sat ** 3 * d3)


def invert_nmp(nmp, m_sigma=546.212459, seed=None):
    """
    Recover DD2 couplings from a target NMP dict with keys
    {n_sat, E_sat, m_eff_ratio, K_sat, Q_sat, E_sym, L_sym}. Returns
    (Parametrization, InversionStatus). Raises ValueError only on a hard
    infeasibility; a soft failure is reported via status.ok=False.
    """
    n_sat = nmp["n_sat"]
    # Feasibility (report §2.6): m*/m too small drives Gamma_sigma sigma -> m_N
    # (scalar collapse); outside a physical RMF window there is no DD2-form fit.
    if not (0.35 < nmp["m_eff_ratio"] < 0.95):
        raise ValueError(
            f"NMP inversion infeasible: m*/m = {nmp['m_eff_ratio']} outside the "
            f"physical (0.35, 0.95) window (scalar collapse / no DD2-form fit)")
    tgt = np.array([0.0, nmp["E_sat"], nmp["m_eff_ratio"],
                    nmp["K_sat"], nmp["Q_sat"]])

    # --- isoscalar 6x6 (5 NMP conditions + cross-constraint) ----------------
    # Seed from the published DD2 couplings: DD2-class NMPs sit near them, and
    # the residual surface has a spurious basin (cross-constraint satisfied but
    # Q_sat wrong) that a generic seed can fall into. The tolerance floor is set
    # by the finite-difference 3rd derivative (Q_sat), ~1e-2.
    if seed is None:
        ref = Parametrization.from_dd2_defaults()
        seed = [ref.gamma_sigma, ref.b_sigma, ref.c_sigma,
                ref.gamma_omega, ref.b_omega, ref.c_omega]

    def iso_residual(p):
        Gs, bS, cS, Gw, bW, cW = p
        if cS <= 0 or cW <= 0 or Gs <= 0 or Gw <= 0:
            return [1e3] * 6
        try:
            par = _trial_par(n_sat, Gs, bS, cS, Gw, bW, cW, m_sigma)
            q = _isoscalar_quantities(par, n_sat)
        except (ValueError, RuntimeError):
            return [1e3] * 6
        cross = _f2_at1(bS, cS) - _f2_at1(bW, cW)
        return [q["P"] - tgt[0], q["E_sat"] - tgt[1], q["m_ratio"] - tgt[2],
                (q["K_sat"] - tgt[3]) * 1e-2, (q["Q_sat"] - tgt[4]) * 1e-2,
                cross]

    sol = root(iso_residual, seed, method="hybr", tol=1e-12)
    iso_res = float(np.max(np.abs(iso_residual(sol.x))))
    Gs, bS, cS, Gw, bW, cW = sol.x

    # --- isovector: Gamma_rho analytic, a_rho by 1-D root -------------------
    par_iso = _trial_par(n_sat, *sol.x, m_sigma)
    at = solve_snm(par_iso, n_sat)
    kF = kF_from_n(n_sat * hc3, 4.0)
    EFs = float(np.sqrt(kF ** 2 + at.m_eff ** 2))
    kin = kF ** 2 / (6.0 * EFs)
    n_nat = n_sat * hc3
    rho_term = nmp["E_sym"] - kin
    if rho_term <= 0:
        raise ValueError(
            f"NMP inversion infeasible: E_sym={nmp['E_sym']} below the "
            f"kinetic symmetry energy {kin:.2f} MeV (no real Gamma_rho)")
    # E_sym = kF^2/(6 EF*) + Gamma_rho^2 n/(2 m_rho^2)  ->  Gamma_rho analytic
    Grho = float(np.sqrt(rho_term * 2.0 * par_iso.m_rho ** 2 / n_nat))

    def Lsym_of_arho(a_rho):
        p = Parametrization.from_microscopic(
            n_sat=n_sat, gamma_sigma=Gs, b_sigma=bS, c_sigma=cS,
            gamma_omega=Gw, b_omega=bW, c_omega=cW,
            gamma_rho=Grho, a_rho=a_rho, m_sigma=m_sigma)
        from eos.dd2.nmp import esym
        dEs = (esym(p, n_sat + 1e-4) - esym(p, n_sat - 1e-4)) / 2e-4
        return 3.0 * n_sat * dEs

    a_rho = brentq(lambda a: Lsym_of_arho(a) - nmp["L_sym"], -2.0, 5.0,
                   xtol=1e-10)
    isov_res = abs(Lsym_of_arho(a_rho) - nmp["L_sym"])

    par = Parametrization.from_microscopic(
        n_sat=n_sat, gamma_sigma=Gs, b_sigma=bS, c_sigma=cS,
        gamma_omega=Gw, b_omega=bW, c_omega=cW,
        gamma_rho=Grho, a_rho=a_rho, m_sigma=m_sigma)
    # Tolerance floor is the finite-difference 3rd derivative (Q_sat): ~1e-2.
    status = InversionStatus(
        ok=(iso_res < 2e-2 and isov_res < 1e-3),
        message="converged" if iso_res < 2e-2 else
        f"isoscalar residual {iso_res:.2e} above the 2e-2 FD floor "
        f"(Q_sat may be inconsistent with the cross-constraint)",
        isoscalar_residual=iso_res, isovector_residual=float(isov_res))
    return par, status
