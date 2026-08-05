"""
nmp_inverter.py
====================
Inverse map: nuclear-matter parameters -> DD2 couplings.

The forward map (nmp.compute_nmp) extracts {n_sat, E_sat, m*/m, K_sat, Q_sat,
E_sym, L_sym} from a Parametrization. This inverts it:

  1. Isoscalar (6x6 root at FIXED n_sat, so no P=0 bracket search in the loop):
     free {Gamma_sigma, Gamma_omega, b_sigma, c_sigma, b_omega, c_omega} matched to
     {P(n_sat)=0, E_sat, m*/m, K_sat, Q_sat, and the closing cross-constraint
     f_sigma''(1)=f_omega''(1)}. m_sigma is fixed: with it fixed, the five
     isoscalar NMPs plus the cross-constraint
     close the sector). a_i, d_i are derived internally (from_microscopic).
  2. Isovector (near-analytic): Gamma_rho(n_sat) from E_sym in closed form, then
     a_rho from L_sym by a 1-D root.
  3. Feasibility flags: m*/m too small, no real isoscalar solution,
     a_rho driving Gamma_rho non-monotonic / negative.

The isoscalar cross-constraint is DD2's own (it holds on the published table to
2.2e-3, not exactly), so a round trip reproduces the NMPs exactly but the shape
coefficients to ~2e-3 — see the M8 gate test.

What limits which NMPs invert: the seed, not the physics
--------------------------------------------------------
A single solve from the published DD2 couplings converges only for targets near
DD2's own (K_sat, Q_sat), and the set it reaches traces a narrow band through
that seed point. That band is a picture of one basin of attraction, NOT of the
feasible set, and reading it as physics is the mistake this module exists to
prevent. Measured on a 187-cell (K_sat, Q_sat) grid over K_sat = 200-300 and
Q_sat = 0-400 MeV:

    restarts     0      32      64
    inverting    7/187  68/187  115/187

The single seed reports 4% of the plane as representable; sixty-four restarts
find 61% of it, still without saturating, and the remaining failures are
scattered rather than bounding a region — so they are very likely further seed
failures too. The residual surface has a spurious basin in which the
cross-constraint is satisfied but Q_sat is wrong, and which basin a solve lands
in is a property of where it started. Hence N_RESTARTS.

Do NOT infer a (K_sat, Q_sat) constraint curve from a scan run at low
`n_restarts`. Six couplings against six conditions makes the system square,
which bounds how much freedom there is in principle, but it does not put the
feasible set on a curve, and the numbers above are what that looks like in
practice.

The remaining numerical limit is the stencil
--------------------------------------------
Q_sat is a third finite difference of E/A, which is itself the output of a
nonlinear solve, and the h = 1e-4 step used here and in the forward map is just
past the truncation/roundoff optimum (~3e-4 to 1e-3): Q_sat carries ~0.1 MeV of
stencil noise at h = 1e-4 and diverges outright by h = 1e-6. ISO_GATE is set by
that noise. The forward map (nmp.compute_nmp) uses the IDENTICAL stencil on
purpose, so the finite-difference bias cancels exactly on a round trip; any
change to h must be made in both places together or the round trip stops
reproducing its own inputs.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, root

from eos.general.physics_constants import hc3
from eos.dd2.couplings import rational_d2f, derived_a, derived_d
from eos.dd2.parametrization import Parametrization
from eos.dd2.physics.thermo import kF_from_n
from eos.dd2.solver import solve_snm


#: Gate on the isoscalar residual. It is set by the finite-difference third
#: derivative that produces Q_sat, not by the physics: with the h = 1e-4 stencil
#: used here and in the forward map, Q_sat carries ~0.1 MeV of stencil noise
#: (scaled by 1e-2 in the residual vector), and a tighter gate would start
#: rejecting converged solutions for a reason that has nothing to do with
#: whether the NMPs are representable.
ISO_GATE = 2e-2

#: Perturbed restarts attempted when the first isoscalar solve misses the gate.
#: They run ONLY on a miss, so an NMP set that inverts from the DD2 seed costs
#: exactly what it did before. What they buy is large and does not saturate —
#: on a 187-cell (K_sat, Q_sat) grid, 7 cells invert from the single seed, 68 at
#: 32 restarts and 115 at 64 — so this default is a compromise with scan cost
#: (a miss costs ~n_restarts x 40 ms) and not a converged answer. Raise it when
#: mapping a boundary matters more than the wall clock.
N_RESTARTS = 32


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


def invert_nmp(nmp, m_sigma=546.212459, seed=None, n_restarts=N_RESTARTS):
    """
    Recover DD2 couplings from a target NMP dict with keys
    {n_sat, E_sat, m_eff_ratio, K_sat, Q_sat, E_sym, L_sym}. Returns
    (Parametrization, InversionStatus). Raises ValueError only on a hard
    infeasibility; a soft failure is reported via status.ok=False.

    `n_restarts` perturbed seeds are tried when the first isoscalar solve misses
    `ISO_GATE`. This is not a refinement: it is what separates "these NMPs have
    no DD-RMF realisation" from "this seed could not find it", and it changes
    the answer for most of the (K_sat, Q_sat) plane — see the note on
    N_RESTARTS. Set it to 0 for the old single-seed behaviour.
    """
    n_sat = nmp["n_sat"]
    # Feasibility: m*/m too small drives Gamma_sigma sigma -> m_N
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
    best_x = sol.x
    iso_res = float(np.max(np.abs(iso_residual(best_x))))

    # A miss is more often the spurious basin than an NMP set with no DD-RMF
    # realisation, so retry from jittered seeds and keep the best. Deterministic
    # by construction: the same NMP must invert identically on every run and in
    # every parallel worker, so the generator is seeded with a constant rather
    # than left to entropy.
    if iso_res >= ISO_GATE and n_restarts:
        rng = np.random.default_rng(0)
        base = np.asarray(seed, dtype=float)
        for _ in range(n_restarts):
            try:
                trial = root(iso_residual, base * rng.uniform(0.75, 1.35, 6),
                             method="hybr", tol=1e-12)
                res = float(np.max(np.abs(iso_residual(trial.x))))
            except Exception:      # a jittered seed that will not build a
                continue           # trial parametrization is not a finding
            if res < iso_res:
                best_x, iso_res = trial.x, res
            if iso_res < ISO_GATE:
                break

    Gs, bS, cS, Gw, bW, cW = best_x

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
    status = InversionStatus(
        ok=(iso_res < ISO_GATE and isov_res < 1e-3),
        message="converged" if iso_res < ISO_GATE else
        f"isoscalar residual {iso_res:.2e} above the {ISO_GATE:.0e} FD floor "
        f"after {n_restarts} restarts (Q_sat is probably inconsistent with the "
        f"cross-constraint at this K_sat)",
        isoscalar_residual=iso_res, isovector_residual=float(isov_res))
    return par, status
