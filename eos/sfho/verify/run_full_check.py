"""
verify/run_full_check.py
========================
Single entry point for the SFHo physics-invariant suite.

Each check returns a structured pass/fail and a maximum error, never a bare
print, so a caller can score the model rather than read a log.

Checks:
  1. Euler / Hugenholtz-Van Hove, every mode — the identity of CLAUDE.md §8,
     eps + P = T s + sum_i mu_i n_i;
  2. nuclear-matter parameters — E_sat, E_sym and L against the published
     SFHo values;
  3. the analytic nuclear-matter derivatives against the stencils they
     replaced — K_sat, Q_sat, L_sym and K_sym must be the h -> 0 limit of the
     finite differences `nmp.compute_nmp` used to take;
  4. the symmetry energy two ways — the delta^2 curvature of E/A against the
     analytic formula of the model's source paper;
  5. causality and monotonicity — 0 <= c_s^2 <= 1 and P non-decreasing along
     a cold beta-equilibrium sweep;
  6. CompOSE HS(SFHo) comparison, when the table is present;
  7. backend parity — the analytic Jacobian of `backends/` against a central
     difference of the residual, CLAUDE.md §9's gate on the fast flavour;
  8. the susceptibility matrix against the inverse map dmu_a/dn_b, which the
     fixed-Y_C-Y_S mode supplies without touching the Jacobian;
  9. the NMP forward and inverse maps against each other, at targets away
     from the published set so the seed is not the answer.

Why check 1 earns its place: SFHo shipped for a long time with an energy
density missing the omega(dA/domega) rho^2 partner of the omega field
equation's cross term. Nothing caught it, because it enters no residual — the
solver converged perfectly on an eps that was wrong by up to 1.8 percent in
asymmetric matter, which put E_sym at 18.7 MeV instead of 31.6 and made L
negative. The Euler identity localises exactly that class of error, and it is
cheap.

It caught nothing twice, though, for the sibling error in the thermal meson
gas: the eta's energy density was dropped from the same accumulation. The
check ran with the gas OFF, and the eta is 548 MeV, so it needs T near 50 MeV
before there is an eta to miss. Both are covered now — the gas enters at its
effective potentials, and the meson cases sit at T = 50.
"""
from dataclasses import dataclass, field

import numpy as np

from eos.sfho.parameters import Parameters
from eos.sfho.species import SpeciesFlags, active_baryons
from eos.sfho.solver import (
    solve_beta_eq_neutrinoless, solve_fixed_yc, solve_fixed_yc_ys,
    solve_beta_eq_neutrino_trapped,
)
from eos.sfho.nmp import esym as symmetry_energy_analytic
from eos.sfho.thermodynamics import thermal_meson_thermo

NUCLEONS = SpeciesFlags()
NUCLEONS_NOGAMMA = SpeciesFlags(photons=False)
WITH_HYPERONS = SpeciesFlags(hyperons=True)
WITH_MESONS = SpeciesFlags(thermal_mesons=True)

#: Steiner, Hempel & Fischer, ApJ 774 (2013) 17, as tabulated by Fortin,
#: Oertel & Providencia, PASA 35 (2018) e044, Table 2.
PUBLISHED = dict(n_sat=0.158, E_sat=-16.2, E_sym=31.6, L=47.1)


@dataclass
class CheckResult:
    name: str
    passed: bool
    max_error: float
    detail: str = ""


@dataclass
class FullCheckReport:
    results: list = field(default_factory=list)

    @property
    def all_passed(self):
        return all(r.passed for r in self.results)

    def __str__(self):
        lines = [f"SFHo run_full_check: {'PASS' if self.all_passed else 'FAIL'}"]
        for r in self.results:
            tag = "ok " if r.passed else "FAIL"
            lines.append(f"  [{tag}] {r.name:24s} max_err={r.max_error:.2e}"
                         f"  {r.detail}")
        return "\n".join(lines)


# =============================================================================
# 1. EULER / HUGENHOLTZ-VAN HOVE
# =============================================================================
def euler_residual(par, result, flags):
    """(eps + P - T s - sum_i mu_i n_i) / eps for one solved point.

    The baryon potentials are rebuilt from the conserved-charge basis,
    mu_i = B_i mu_B + C_i mu_C + S_i mu_S (CLAUDE.md §2), and the leptons are
    added at their own potentials. Photons need no term: they carry mu = 0 and
    satisfy eps + P = T s on their own.

    The thermal meson gas enters at its EFFECTIVE potentials — mu*_i, carrying
    the omega and rho shifts — because those fields are sourced by the baryons
    alone, so the shift has no partner in the field energy to cancel against.
    `eos.dd2.thermodynamics.assemble` sums the gas the same way.
    """
    mu_dot_n = 0.0
    for p in active_baryons(flags):
        n = result.matter.densities.get(p.name, 0.0)
        mu = (p.baryon_no * result.matter.mu_B + p.charge * result.matter.mu_C
              + p.strangeness * result.matter.mu_S)
        mu_dot_n += mu * n
    lep = result.leptons
    mu_dot_n += (lep.mu_e * lep.densities["e-"]
                 + lep.mu_nue * lep.densities["nu_e"])
    if flags.thermal_mesons:
        m = result.matter
        gas = thermal_meson_thermo(result.T, m.mu_C, m.mu_S,
                                   m.fields["omega"], m.fields["rho"], par)
        mu_dot_n += gas["mu_dot_n"]
    if result.eps == 0.0:
        return 0.0
    return ((result.eps + result.P - result.T * result.s
             - mu_dot_n) / result.eps)


def _check_euler(nuc, hyp, grid):
    worst, where = 0.0, ""
    for n_B in grid:
        cases = [
            ("beta T=0", nuc, NUCLEONS,
             solve_beta_eq_neutrinoless(nuc, n_B, NUCLEONS, T=0.0)),
            ("beta T=30", nuc, NUCLEONS,
             solve_beta_eq_neutrinoless(nuc, n_B, NUCLEONS, T=30.0)),
            ("beta hyperons", hyp, WITH_HYPERONS,
             solve_beta_eq_neutrinoless(hyp, n_B, WITH_HYPERONS, T=10.0)),
            ("fixed Y_C", nuc, NUCLEONS,
             solve_fixed_yc(nuc, n_B, 0.1, NUCLEONS, T=10.0)),
            ("fixed Y_C leptons", nuc, NUCLEONS,
             solve_fixed_yc(nuc, n_B, 0.1, NUCLEONS, T=10.0, leptons=True)),
            ("fixed Y_C Y_S", hyp, WITH_HYPERONS,
             solve_fixed_yc_ys(hyp, n_B, 0.4, 0.1, WITH_HYPERONS, T=10.0)),
            ("trapped", nuc, NUCLEONS,
             solve_beta_eq_neutrino_trapped(nuc, n_B, 0.4, NUCLEONS, T=10.0)),
            ("isentropic", nuc, NUCLEONS,
             solve_beta_eq_neutrinoless(nuc, n_B, NUCLEONS, SnB=1.0)),
            # T = 50 MeV, not 30: the eta is 548 MeV, so below about 40 MeV its
            # population underflows to zero and a dropped eta term is invisible.
            ("beta mesons T=50", nuc, WITH_MESONS,
             solve_beta_eq_neutrinoless(nuc, n_B, WITH_MESONS, T=50.0)),
            ("fixed Y_C mesons T=50", nuc, WITH_MESONS,
             solve_fixed_yc(nuc, n_B, 0.1, WITH_MESONS, T=50.0, leptons=True)),
        ]
        for tag, par, flags, r in cases:
            if not r.converged:
                continue
            err = abs(euler_residual(par, r, flags))
            if err > worst:
                worst, where = err, f"{tag} at n_B={n_B:g}"
    return CheckResult("Euler / HVH, all modes", worst < 1e-8, worst, where)


# =============================================================================
# 2, 3. NUCLEAR MATTER PARAMETERS
# =============================================================================
def _energy_per_baryon(par, n_B, Y_C):
    r = solve_fixed_yc(par, n_B, Y_C, NUCLEONS_NOGAMMA, T=0.01)
    m_N = 0.5 * (par.m_n + par.m_p)
    return r.eps / n_B - m_N


def symmetry_energy(par, n_B, delta=0.05):
    """E_sym from the delta^2 curvature of E/A, delta = 1 - 2 Y_C.

    A SYMMETRIC second difference. A one-sided one leaks the cubic term and
    is wrong here by several MeV -- large enough to hide the missing energy
    term this suite exists to catch.
    """
    e_plus = _energy_per_baryon(par, n_B, 0.5 * (1 - delta))
    e_zero = _energy_per_baryon(par, n_B, 0.5)
    e_minus = _energy_per_baryon(par, n_B, 0.5 * (1 + delta))
    return (e_plus + e_minus - 2 * e_zero) / (2 * delta**2)


def _check_nmp(par):
    n_sat = PUBLISHED["n_sat"]
    E_sat = _energy_per_baryon(par, n_sat, 0.5)
    E_sym = symmetry_energy(par, n_sat)
    h = 0.004
    L = 3.0 * n_sat * (symmetry_energy(par, n_sat + h)
                       - symmetry_energy(par, n_sat - h)) / (2.0 * h)
    errs = {
        "E_sat": abs(E_sat - PUBLISHED["E_sat"]) / abs(PUBLISHED["E_sat"]),
        "E_sym": abs(E_sym - PUBLISHED["E_sym"]) / PUBLISHED["E_sym"],
        "L": abs(L - PUBLISHED["L"]) / PUBLISHED["L"],
    }
    worst = max(errs.values())
    return CheckResult(
        "published NMPs", worst < 2e-2, worst,
        f"E_sat={E_sat:.2f} E_sym={E_sym:.2f} L={L:.2f}")


def _check_analytic_derivatives(par):
    """The analytic K_sat, Q_sat, L_sym and K_sym are the stencil's limit.

    `nmp.snm_derivatives` differentiates the closed forms of symmetric matter
    by hand rather than differencing the solver, so nothing inside it is
    checked by the solver agreeing with itself. What ties the two together is
    that the finite differences it replaced must CONVERGE TO IT: each is a
    central difference of order h^2, so Richardson-extrapolating the pair
    (h, h/2) removes the leading error, and the analytic value must sit
    within the pair's own scatter of that extrapolation.

    Self-calibrating on purpose -- the scatter |d(h) - d(h/2)| is what the
    stencil itself says its accuracy is, so there is no tolerance to tune and
    none to loosen.

    h = 8e-3, and that IS a measured choice, made for the opposite reason to
    dd2's. There the small-h end failed on roundoff, which jittered with the
    interpreter. Here the two stacks agree to four digits at every h, because
    what stops the stencil converging is not roundoff but `hybr`'s own xtol:
    the solver returns a state whose density is up to 5e-11 relative away from
    the one it was asked for, and dividing by the requested density leaves a
    SMOOTH ~5e-08 MeV wobble on E/A. Differentiated, that is a fixed offset
    from the analytic value -- 1.9e-06 relative on K_sat and 2.8e-04 on Q_sat,
    the same at h = 1e-3 and at h = 5e-3 -- so the estimator is honest only
    once the scatter has grown past it. Worst of the four, Q_sat every time:

        h        1.5e-3   2e-3    4e-3    6e-3    8e-3    1.2e-2
        py3.9     4.674   3.152   0.800   0.356   0.279   0.089
        py3.14    4.674   3.244   0.800   0.356   0.279   0.089

    (anaconda 3.9.7 / numpy 1.26.4 / scipy 1.13.1 against python.org 3.14.2 /
    numpy 2.3.5 / scipy 1.17.0.) At 8e-3 both stacks pass with 3.6x of margin
    and the ratio is still falling, so anywhere above ~5e-3 does the same job.

    That the floor is the SOLVER's and not the derivation's is measurable
    twice over: the analytic values reproduce across those two stacks to
    5.9e-14 (K_sat), 2.2e-13 (Q_sat), 7.0e-15 (L_sym) and 6.8e-16 (K_sym),
    and stencilling a hand-solved gap equation instead of the solver moves the
    K_sat agreement from 1.9e-06 to 1.4e-06 while the offset vanishes.
    """
    from scipy.optimize import brentq
    from eos.sfho.nmp import (snm_derivatives, energy_per_baryon, esym,
                              pressure)

    n_sat = brentq(lambda n: pressure(par, n), 0.12, 0.20, xtol=1e-13)
    analytic = snm_derivatives(par, n_sat)

    def stencils(h):
        EA = lambda n: energy_per_baryon(par, n)
        ES = lambda n: esym(par, n)
        return {
            "K_sat": 9 * n_sat ** 2 * (EA(n_sat + h) - 2 * EA(n_sat)
                                       + EA(n_sat - h)) / h ** 2,
            "Q_sat": 27 * n_sat ** 3 * (EA(n_sat + 2 * h) - 2 * EA(n_sat + h)
                                        + 2 * EA(n_sat - h)
                                        - EA(n_sat - 2 * h)) / (2 * h ** 3),
            "L_sym": 3 * n_sat * (ES(n_sat + h) - ES(n_sat - h)) / (2 * h),
            "K_sym": 9 * n_sat ** 2 * (ES(n_sat + h) - 2 * ES(n_sat)
                                       + ES(n_sat - h)) / h ** 2,
        }

    h = 8e-3                      # past the solver's floor: see the docstring
    coarse, fine = stencils(h), stencils(h / 2)
    worst, worst_key = 0.0, ""
    for key, value in analytic.items():
        richardson = (4.0 * fine[key] - coarse[key]) / 3.0
        scatter = abs(fine[key] - coarse[key])
        ratio = abs(value - richardson) / scatter
        if ratio > worst:
            worst, worst_key = ratio, key
    return CheckResult("analytic NMP derivatives", worst < 1.0, worst,
                       f"vs Richardson(h={h:.1e}, h/2), worst {worst_key} at "
                       f"{worst:.3f} of the stencil's own scatter")


def _check_esym_two_ways(par):
    """The delta^2 curvature of E/A against the rho-field closed form.

    Two independent routes to E_sym: this one differentiates the energy
    density twice in the isospin asymmetry, `eos.sfho.nmp.esym` evaluates the
    rho response of Steiner, Prakash, Lattimer & Ellis Eq. (20). An error in
    eps moves the first and leaves the second alone, which is the whole point
    of running both -- so the analytic side is IMPORTED rather than copied
    here, but the curvature side stays local and must not be replaced by a
    call to `compute_nmp`.
    """
    n_sat = PUBLISHED["n_sat"]
    curvature = symmetry_energy(par, n_sat)
    analytic = symmetry_energy_analytic(par, n_sat)
    err = abs(curvature - analytic) / analytic
    return CheckResult(
        "E_sym, curvature vs Eq. (20)", err < 1e-2, err,
        f"{curvature:.2f} vs {analytic:.2f} MeV")


# =============================================================================
# 4. CAUSALITY AND MONOTONICITY
# =============================================================================
def _check_causality(par, grid):
    P, eps = [], []
    for n_B in grid:
        r = solve_beta_eq_neutrinoless(par, n_B, NUCLEONS, T=0.0)
        if not r.converged:
            return CheckResult("causality, monotone P", False, 1.0,
                               f"no convergence at n_B={n_B:g}")
        P.append(r.P)
        eps.append(r.eps)
    P, eps = np.asarray(P), np.asarray(eps)
    cs2 = np.gradient(P, eps)
    dP = np.diff(P)
    worst = max(float(np.max(cs2)) - 1.0, -float(np.min(cs2)),
                -float(np.min(dP)) / float(np.max(P)))
    return CheckResult("causality, monotone P", worst <= 0.0, max(worst, 0.0),
                       f"c_s^2 in [{cs2.min():.3f}, {cs2.max():.3f}]")


# =============================================================================
# 5. COMPOSE
# =============================================================================
def _check_compose(par, compose_dir=None):
    """Compare P, eps, s and mu_B with the published HS(SFHo) table.

    Three charge fractions at T = 10 MeV, over the uniform-matter density
    range. The conditions and why they are those conditions are in
    `verify/compose.py`; here it is only the pass/fail. Skipped where the
    table is not on this machine, since it is a download rather than repository
    data.
    """
    from eos.sfho.verify import compose as _compose

    compose_dir = compose_dir or _compose.SFHO_COMPOSE
    if not _compose.available(compose_dir):
        return CheckResult("CompOSE HS(SFHo)", True, 0.0, "skipped (no table)")
    worst, where = 0.0, ""
    for Y_C in (0.5, 0.3, 0.1):
        out = _compose.compare_slice(par, NUCLEONS, compose_dir=compose_dir,
                                     T=10.0, Y_C=Y_C,
                                     nB_min=0.2, nB_max=0.6)
        err = max(out["max_err_P"], out["max_err_eps"])
        if err > worst:
            worst, where = err, f"Y_q={out['Y_C']:.2f}"
    return CheckResult("CompOSE HS(SFHo)", worst < 2e-3, worst,
                       f"T=10.0 MeV, worst at {where}")


# =============================================================================
# 6. BACKEND PARITY
# =============================================================================
def _fd_jacobian(x, sys):
    """dR/dx by central differences of `solver.residual` -- the reference."""
    from eos.sfho.solver import residual
    x = np.asarray(x, dtype=float)
    rows = len(residual(x, sys))
    J = np.zeros((rows, len(x)))
    for k in range(len(x)):
        h = max(1e-5, 1e-6 * abs(x[k]))
        up, lo = x.copy(), x.copy()
        up[k] += h
        lo[k] -= h
        J[:, k] = (np.asarray(residual(up, sys))
                   - np.asarray(residual(lo, sys))) / (2.0 * h)
    return J


def _check_jacobian(nuc, hyp):
    """The analytic Jacobian against a central difference of the residual.

    CLAUDE.md §9's gate on the accelerated flavour: `backends/jacobian` writes
    out the same derivative the reference path builds numerically, so the two
    must agree. Each entry is compared against the largest entry of its own
    row, because a Jacobian row spans many orders of magnitude -- a field
    equation's diagonal is m^2 and its coupling to a distant potential is not.

    The kinetic derivatives are closed forms at T = 0 and central differences
    at T > 0, so the T = 0 cases agree ~1e-9 and the T > 0 ones ~1e-8; both are
    far inside the residual gate the solve is judged on.
    """
    from eos.general import modes
    from eos.sfho.solver import _system, solve, warm_start
    from eos.sfho.backends.jacobian import residual_jacobian

    cases = [
        ("beta T=0", nuc, NUCLEONS, modes.beta_eq_neutrinoless(), 0.16, 0.0),
        ("beta T=10", nuc, NUCLEONS, modes.beta_eq_neutrinoless(), 0.16, 10.0),
        ("beta hyperons", hyp, WITH_HYPERONS,
         modes.beta_eq_neutrinoless(), 0.5, 10.0),
        ("fixed Y_C", nuc, NUCLEONS, modes.fixed_YC(0.3), 0.16, 10.0),
        ("fixed Y_C Y_S", hyp, WITH_HYPERONS,
         modes.fixed_YC_YS(0.3, 0.1), 0.5, 10.0),
        ("trapped", nuc, NUCLEONS,
         modes.beta_eq_neutrino_trapped(0.4), 0.16, 30.0),
        ("mesons T=50", nuc, WITH_MESONS, modes.fixed_YC(0.3), 0.3, 50.0),
    ]
    worst, where = 0.0, ""
    for tag, par, flags, spec, n_B, T in cases:
        sys = _system(par, flags, spec, n_B, T=T)
        point = solve(sys)
        if not point.converged:
            return CheckResult("Jacobian vs finite difference", False, 1.0,
                               f"{tag}: no converged state to check at")
        x = warm_start(point, spec)
        J_a = residual_jacobian(x, sys)
        J_f = _fd_jacobian(x, sys)
        scale = np.maximum(np.abs(J_f).max(axis=1, keepdims=True), 1e-300)
        err = float(np.max(np.abs(J_a - J_f) / scale))
        if err > worst:
            worst, where = err, tag
    return CheckResult("Jacobian vs finite difference", worst < 1e-6, worst,
                       f"worst at {where}")


def _check_nmp_roundtrip():
    """compute_nmp(invert_nmp(x)) = x — the forward and inverse maps agree.

    The two directions share the NMP list, its ordering and its stencils, so
    this is the check that they also share their PHYSICS: the inverse imposes
    {P(n_sat) = 0, E_sat, m*/m, K_sat} on the isoscalar side and
    {E_sym, L_sym} on the isovector one, and the forward map recomputes all six
    from the recovered couplings by a route that knows nothing about how they
    were found.

    Three targets away from the published set, so the seed is not the answer.
    Q_sat and K_sym are excluded: this closure predicts them rather than
    imposing them, and a prediction has nothing to round-trip against.
    """
    from eos.sfho.nmp import compute_nmp, invert_nmp

    keys = ("n_sat", "E_sat", "m_eff_ratio", "K_sat", "E_sym", "L_sym")
    targets = [
        dict(n_sat=0.160, E_sat=-16.00, m_eff_ratio=0.75, K_sat=240.0,
             E_sym=32.0, L_sym=60.0),
        dict(n_sat=0.155, E_sat=-15.80, m_eff_ratio=0.70, K_sat=260.0,
             E_sym=30.0, L_sym=40.0),
        dict(n_sat=0.165, E_sat=-16.50, m_eff_ratio=0.80, K_sat=220.0,
             E_sym=34.0, L_sym=90.0),
    ]
    worst, where = 0.0, ""
    for target in targets:
        par, status = invert_nmp(**target)
        if not status.ok:
            return CheckResult("NMP forward/inverse", False, 1.0,
                               f"L_sym={target['L_sym']:g}: {status.message}")
        forward = compute_nmp(par)
        for key in keys:
            err = abs(forward[key] - target[key]) / max(abs(target[key]), 1e-3)
            if err > worst:
                worst, where = err, f"{key} at L_sym={target['L_sym']:g}"
    return CheckResult("NMP forward/inverse", worst < 1e-5, worst,
                       f"worst on {where}")


def _check_susceptibilities(hyp, n_B=0.8, T=10.0, rel=1e-4):
    """chi_ab = dn_a/dmu_b against the inverse map, dmu_a/dn_b.

    The susceptibility matrix is the one response with no finite-difference
    twin inside the model -- the solver never varies mu_B, mu_C and mu_S
    independently, so there is no sequence to walk along. The independent route
    is the OTHER direction: the fixed-Y_C-Y_S mode imposes (n_B, n_C, n_S) and
    reports (mu_B, mu_C, mu_S), so re-solving it at perturbed charges gives
    dmu_a/dn_b without touching the Jacobian, and the two matrices must
    multiply to the identity.

    Hyperons at n_B = 0.8 fm^-3 deliberately. Lower down the strangeness
    density is essentially zero -- 2.5e-07 fm^-3 at n_B = 0.16, T = 10 MeV --
    and where a conserved charge is carried by no populated species the
    residual is flat in its potential, so the numerical dmu_S/dn_S is
    meaningless there. That is a property of the model rather than of chi.
    """
    from eos.sfho.solver import _system, solve
    from eos.general import modes
    from eos.sfho.backends.responses_jac import susceptibilities

    flags = WITH_HYPERONS
    base = solve(_system(hyp, flags, modes.beta_eq_neutrinoless(), n_B, T=T))
    if not base.converged:
        return CheckResult("chi_ab vs dmu/dn", False, 1.0,
                           f"no beta-eq state at n_B={n_B:g}")
    charges = [n_B, base.matter.n_C, base.matter.n_S]
    dmu_dn = np.zeros((3, 3))
    for b in range(3):
        step = rel * max(abs(charges[b]), 1e-3)
        ends = []
        for sign in (+1.0, -1.0):
            perturbed = list(charges)
            perturbed[b] += sign * step
            nB_, nC_, nS_ = perturbed
            spec = modes.fixed_YC_YS(nC_ / nB_, nS_ / nB_)
            point = solve(_system(hyp, flags, spec, nB_, T=T))
            if not point.converged:
                return CheckResult("chi_ab vs dmu/dn", False, 1.0,
                                   "perturbed fixed-Y_C-Y_S solve failed")
            m = point.matter
            ends.append(np.array([m.mu_B, m.mu_C, m.mu_S]))
        dmu_dn[:, b] = (ends[0] - ends[1]) / (2.0 * step)
    chi = susceptibilities(hyp, n_B, flags, T=T)
    err = float(np.max(np.abs(chi @ dmu_dn - np.eye(3))))
    return CheckResult("chi_ab vs dmu/dn", err < 1e-4, err,
                       f"n_B={n_B:g} fm^-3, T={T:g} MeV, hyperons")


# =============================================================================
def run_full_check(par=None, hyp=None, grid=None):
    """Run the SFHo verification suite; returns a structured FullCheckReport."""
    par = par or Parameters.default()
    hyp = hyp or Parameters.named("SFHoY_Fortin")
    grid = np.array(grid) if grid is not None else np.array([0.16, 0.32, 0.64])

    report = FullCheckReport()
    report.results.append(_check_euler(par, hyp, grid))
    report.results.append(_check_nmp(par))
    report.results.append(_check_analytic_derivatives(par))
    report.results.append(_check_esym_two_ways(par))
    report.results.append(_check_causality(par, np.linspace(0.1, 1.0, 25)))
    report.results.append(_check_compose(par))
    report.results.append(_check_jacobian(par, hyp))
    report.results.append(_check_susceptibilities(hyp))
    report.results.append(_check_nmp_roundtrip())
    return report


if __name__ == "__main__":
    print(run_full_check())
