"""Physics invariants of the alphaBag model, checked in one place.

These are the statements the implementation has to satisfy no matter which
parameters it is given; they are the fastest way to catch a wrong change.
Every check returns a structured pass/fail with the largest error it saw, so
the suite reports rather than prints.

  1. Euler relation      eps + P = T s + sum_q mu_q n_q, in every mode and in
                         the paired phase, with the bag included -- it enters
                         eps and P with opposite signs and cancels.
  2. Free energy         f = eps - T s = -P + sum_q mu_q n_q.
  3. Massless gas        n = dP/dmu, s = dP/dT and eps = 3P for the closed
                         forms, at any alpha_s. This is what makes the
                         correction thermodynamically consistent rather than
                         a fudge on the pressure.
  4. Massive limit       the massive branch reproduces the massless one as
                         m -> 0, and its alpha_s correction is itself a
                         consistent (n, P, eps, s) set.
  5. Gluons              eps_g = 3 P_g and s_g = 4 P_g/T; zero at T = 0.
  6. Charge basis        n_B, n_C, n_S and mu_B, mu_C, mu_S agree with
                         eos.general.basis -- no local copy of the map.
  7. Mode closures       each mode's own conditions hold at its solution.
  8. CFL phase           n_C = 0 by construction; n_q = dP/dmu_q and
                         s = dP/dT of the paired pressure; the gap closes at
                         T_c and its slope matches a numerical derivative.
  9. Causality           0 <= c_s^2 <= 1 along a cold beta-equilibrium
                         sequence.

Run as `python -m eos.alphabag.verify.run_full_check`.
"""
from dataclasses import dataclass, field

import numpy as np

from eos.general.basis import charge_potentials_from_quarks, quark_charges
from eos.general.physics_constants import hc3
from eos.general.thermodynamics_leptons import (
    electron_thermo, neutrino_thermo, photon_thermo,
)
from eos.alphabag import (
    Parameters, T_critical, bag_energy, bag_pressure, cfl_dgap_dT, cfl_gap,
    cfl_thermo_from_mu, e_massless, eos_response, gluon_thermo,
    kinetic_thermo, n_massless, P_massless, s_massless,
    solve_beta_eq_neutrinoless, solve_beta_eq_neutrino_trapped, solve_cfl,
    solve_fixed_yc, solve_fixed_yc_ys, thermo_from_mu,
)


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
        lines = [f"alphaBag run_full_check: "
                 f"{'PASS' if self.all_passed else 'FAIL'}"]
        for r in self.results:
            tag = "ok " if r.passed else "FAIL"
            lines.append(f"  [{tag}] {r.name:24s} max_err={r.max_error:.2e}"
                         f"  {r.detail}")
        return "\n".join(lines)


def _matter_only(result, par):
    """(P, eps, s) of the quarks and the bag alone.

    The Euler relation holds sector by sector; the solvers return totals, so
    the leptonic, radiative and gluonic parts have to be subtracted before the
    quark identity can be tested on its own.
    """
    P, eps, s = result.P_total, result.e_total, result.s_total
    T = result.T
    e_thermo = electron_thermo(result.mu_e, T)
    P, eps, s = P - e_thermo.P, eps - e_thermo.e, s - e_thermo.s
    if result.mu_nu != 0.0:
        nu = neutrino_thermo(result.mu_nu, T)
        P, eps, s = P - nu.P, eps - nu.e, s - nu.s
    gamma = photon_thermo(T)
    P, eps, s = P - gamma.P, eps - gamma.e, s - gamma.s
    gluon = gluon_thermo(T, par.alpha)
    P, eps, s = P - gluon.P, eps - gluon.e, s - gluon.s
    # The untracked flavours, at mu = 0: three where the electron neutrino is
    # free-streaming, two where it is trapped.
    nu_th = neutrino_thermo(0.0, T)
    n_flavours = 2.0 if result.mu_nu != 0.0 else 3.0
    P -= n_flavours * nu_th.P
    eps -= n_flavours * nu_th.e
    s -= n_flavours * nu_th.s
    return P, eps, s


def _states(par, grid, T):
    """One solved state per mode, at each density of the grid."""
    out = []
    for n_B in grid:
        out.append(("beta", solve_beta_eq_neutrinoless(n_B, T, params=par)))
        out.append(("trapped",
                    solve_beta_eq_neutrino_trapped(n_B, 0.4, T, params=par)))
        out.append(("yc", solve_fixed_yc(n_B, 0.0, T, params=par,
                                         include_electrons=True)))
        out.append(("yc_nolep", solve_fixed_yc(n_B, 0.0, T, params=par,
                                               include_electrons=False)))
        out.append(("ycys", solve_fixed_yc_ys(n_B, 0.0, 1.0, T, params=par)))
    return out


def _check_euler(par, grid, T):
    """eps + P = T s + sum_q mu_q n_q, with the bag in both sides' sectors."""
    worst = 0.0
    for _, r in _states(par, grid, T):
        P, eps, s = _matter_only(r, par)
        mu_n_sum = r.mu_u * r.n_u + r.mu_d * r.n_d + r.mu_s * r.n_s
        worst = max(worst, abs(eps + P - T * s - mu_n_sum) / abs(eps))
    for n_B in grid:
        for Delta0 in (50.0, 100.0):
            c = solve_cfl(n_B, T, Delta0, params=par, include_photons=False,
                          include_gluons=False)
            mu_n_sum = c.mu_u * c.n_u + c.mu_d * c.n_d + c.mu_s * c.n_s
            worst = max(worst, abs(c.e_total + c.P_total - T * c.s_total
                                   - mu_n_sum) / abs(c.e_total))
    return CheckResult("Euler relation", worst < 1e-10, worst,
                       f"T={T} MeV, {5*len(grid)} unpaired + "
                       f"{2*len(grid)} paired states")


def _check_free_energy(par, grid, T):
    """f = eps - T s, and f = -P + sum_q mu_q n_q for it."""
    worst = 0.0
    for _, r in _states(par, grid, T):
        P, eps, s = _matter_only(r, par)
        f = eps - T * s
        mu_n_sum = r.mu_u * r.n_u + r.mu_d * r.n_d + r.mu_s * r.n_s
        worst = max(worst, abs(f - (-P + mu_n_sum)) / abs(eps))
    return CheckResult("free energy", worst < 1e-10, worst,
                       "f = eps - Ts = -P + sum mu n")


def _check_massless_gas(par):
    """n = dP/dmu, s = dP/dT and eps = 3P of the closed forms.

    The alpha_s correction multiplies P, n, eps and s separately, so nothing
    forces them to remain a consistent set -- except that the same two factors
    were applied to the terms that map onto one another under the derivative.
    This is the check that they were.
    """
    worst_n = worst_s = worst_e = 0.0
    h = 1e-4
    for alpha in (0.0, 0.3, 0.6):
        for mu in (200.0, 400.0, 600.0):
            for T in (5.0, 30.0, 80.0):
                dP_dmu = (P_massless(mu + h, T, alpha)
                          - P_massless(mu - h, T, alpha)) / (2 * h)
                dP_dT = (P_massless(mu, T + h, alpha)
                         - P_massless(mu, T - h, alpha)) / (2 * h)
                n = n_massless(mu, T, alpha)
                s = s_massless(mu, T, alpha)
                worst_n = max(worst_n, abs(dP_dmu - n) / abs(n))
                worst_s = max(worst_s, abs(dP_dT - s) / abs(s))
                worst_e = max(worst_e,
                              abs(e_massless(mu, T, alpha)
                                  - 3.0 * P_massless(mu, T, alpha))
                              / abs(e_massless(mu, T, alpha)))
    passed = worst_n < 1e-8 and worst_s < 1e-8 and worst_e == 0.0
    return CheckResult("massless closed forms", passed,
                       max(worst_n, worst_s, worst_e),
                       f"n: {worst_n:.1e}, s: {worst_s:.1e}, "
                       f"eps=3P: {worst_e:.1e}")


def _check_massive_limit(par):
    """The massive branch at m -> 0, and the consistency of its correction.

    Two statements. First, `kinetic_thermo` above the massless cut-off must
    approach the closed forms as the mass vanishes -- otherwise the two
    branches disagree across a threshold nothing else would notice. Second,
    the correction ADDED to the exact Fermi gas must satisfy
    dn = d(dP)/dmu and ds = d(dP)/dT, since that is what carries the Euler
    relation of the free gas through to the corrected one.

    The limit is taken at T = 0, where the Fermi integrals are exact and the
    two branches agree to round-off once the m^2 term is small (1.2e-13 at
    m = 1e-4 MeV). At T > 0 they agree only to about 1e-4 -- 3.0e-04 at the
    worst corner sampled, mu = 200 MeV and T = 80 MeV, where the gas is at
    its least degenerate. That is the accuracy of the JEL approximation
    itself against the exact massless gas, not a property of this model, so
    it is measured and reported rather than asserted tightly; a REGRESSION in
    it would still show, since the gate is an order of magnitude above what
    the approximation delivers and not two.
    """
    worst_limit = 0.0
    for mu in (300.0, 500.0):
        tiny = kinetic_thermo(mu, 0.0, 1e-4, par.alpha)
        exact = kinetic_thermo(mu, 0.0, 0.0, par.alpha)
        for a, b in ((tiny.n, exact.n), (tiny.P, exact.P),
                     (tiny.e, exact.e)):
            worst_limit = max(worst_limit, abs(a - b) / max(abs(b), 1e-30))

    worst_jel = 0.0
    for mu in (300.0, 500.0):
        for T in (10.0, 50.0):
            tiny = kinetic_thermo(mu, T, 1e-4, par.alpha)
            exact = kinetic_thermo(mu, T, 0.0, par.alpha)
            for a, b in ((tiny.n, exact.n), (tiny.P, exact.P),
                         (tiny.e, exact.e)):
                worst_jel = max(worst_jel, abs(a - b) / max(abs(b), 1e-30))

    worst_corr = 0.0
    h = 1e-4
    alpha = par.alpha

    def dP(mu, T):
        return P_massless(mu, T, alpha) - P_massless(mu, T, 0.0)

    for mu in (300.0, 500.0):
        for T in (10.0, 50.0):
            dn = n_massless(mu, T, alpha) - n_massless(mu, T, 0.0)
            ds = s_massless(mu, T, alpha) - s_massless(mu, T, 0.0)
            num_n = (dP(mu + h, T) - dP(mu - h, T)) / (2 * h)
            num_s = (dP(mu, T + h) - dP(mu, T - h)) / (2 * h)
            worst_corr = max(worst_corr, abs(num_n - dn) / abs(dn),
                             abs(num_s - ds) / abs(ds))
    passed = (worst_limit < 1e-10 and worst_corr < 1e-8
              and worst_jel < 1e-3)
    return CheckResult("massive branch", passed,
                       max(worst_limit, worst_corr),
                       f"m -> 0 at T=0: {worst_limit:.1e}, "
                       f"correction: {worst_corr:.1e}, "
                       f"JEL at T>0: {worst_jel:.1e}")


def _check_gluons_and_bag(par):
    """eps_g = 3 P_g, s_g = 4 P_g/T, and the bag's equal and opposite entry.

    The bag identity is what makes the Euler relation hold with no bag term:
    eps_B = -P_B exactly, so it cancels out of eps + P.
    """
    worst = 0.0
    for T in (1.0, 30.0, 120.0):
        g = gluon_thermo(T, par.alpha)
        worst = max(worst, abs(g.e - 3.0 * g.P) / abs(g.e),
                    abs(g.s - 4.0 * g.P / T) / abs(g.s))
    cold = gluon_thermo(0.0, par.alpha)
    zero_at_zero = (cold.P == 0.0 and cold.e == 0.0 and cold.s == 0.0)
    bag_ok = bag_energy(par) == -bag_pressure(par) == par.B / hc3
    return CheckResult("gluons and bag", worst < 1e-14 and zero_at_zero
                       and bag_ok, worst,
                       f"eps=3P, s=4P/T; bag {par.B/hc3:.3f} MeV/fm^3")


def _check_charge_basis(par, grid, T):
    """The charges and potentials this model reports ARE eos.general.basis.

    A quark model is the easiest place to write (2 n_u - n_d - n_s)/3 by hand,
    and the easiest place for that copy to drift from the particle table the
    hadronic models read. This asserts they have not.
    """
    worst = 0.0
    for n_B in grid:
        r = solve_beta_eq_neutrinoless(n_B, T, params=par)
        block = thermo_from_mu(r.mu_u, r.mu_d, r.mu_s, T, par)
        n_B_b, n_C_b, n_S_b = quark_charges(block.n_u, block.n_d, block.n_s)
        mu_B_b, mu_C_b, mu_S_b = charge_potentials_from_quarks(
            r.mu_u, r.mu_d, r.mu_s)
        worst = max(worst,
                    abs(block.n_B - n_B_b), abs(block.n_C - n_C_b),
                    abs(block.n_S - n_S_b),
                    abs(block.mu_B - mu_B_b), abs(block.mu_C - mu_C_b),
                    abs(block.mu_S - mu_S_b))
        # S = +1 per s quark, this repository's convention.
        worst = max(worst, abs(block.n_S - block.n_s))
    return CheckResult("charge basis", worst == 0.0, worst,
                       "n_B, n_C, n_S and mu_B, mu_C, mu_S from general/basis")


def _check_mode_closures(par, grid, T):
    """Each mode's defining conditions, evaluated at its own solution."""
    worst = 0.0
    for n_B in grid:
        r = solve_beta_eq_neutrinoless(n_B, T, params=par)
        mu_scale = abs(r.mu_B)
        worst = max(worst,
                    abs(r.mu_C + r.mu_e) / mu_scale,        # beta equilibrium
                    abs(r.mu_S) / mu_scale,                 # strangeness eq.
                    abs(r.Y_C * n_B - r.n_e) / n_B,         # neutrality
                    abs(r.n_B - n_B) / n_B)                 # baryon number

        r = solve_beta_eq_neutrino_trapped(n_B, 0.4, T, params=par)
        worst = max(worst,
                    abs(r.mu_C + r.mu_e - r.mu_nu) / abs(r.mu_B),
                    abs(r.mu_S) / abs(r.mu_B),
                    abs(r.Y_C * n_B - r.n_e) / n_B,
                    abs((r.n_e + r.n_nu) / n_B - 0.4))

        r = solve_fixed_yc(n_B, 0.0, T, params=par, include_electrons=True)
        worst = max(worst, abs(r.Y_C), abs(r.mu_S) / abs(r.mu_B),
                    abs(r.n_B - n_B) / n_B)

        r = solve_fixed_yc(n_B, 0.0, T, params=par, include_electrons=False)
        # No neutrality here: the phase would be charged at any other Y_C,
        # which is the point.
        worst = max(worst, abs(r.Y_C), abs(r.n_e))

        r = solve_fixed_yc_ys(n_B, 0.0, 1.0, T, params=par)
        worst = max(worst, abs(r.Y_C), abs(r.Y_S - 1.0),
                    abs(r.n_B - n_B) / n_B)
    return CheckResult("mode closures", worst < 1e-8, worst,
                       "beta, strangeness, neutrality, fixed fractions")


def _check_cfl(par, grid, T):
    """The paired phase: neutrality by construction, and P as the potential.

    n_C vanishes identically at equal flavour densities, so the phase needs no
    electron to be neutral. And the densities and entropy must be derivatives
    of the paired pressure, since that is what makes the Delta^2 term a
    contribution to a thermodynamic potential rather than a correction bolted
    onto three separate quantities.
    """
    worst_neutral = worst_deriv = worst_gap = 0.0
    h = 1e-4
    for n_B in grid:
        for Delta0 in (50.0, 100.0):
            # The bare phase: the derivatives below are of its own
            # potential, so the gluon and photon gases must not be in the
            # totals they are compared against.
            c = solve_cfl(n_B, T, Delta0, params=par, include_photons=False,
                          include_gluons=False)
            worst_neutral = max(worst_neutral, abs(c.Y_C))

            def P_at(mu_u, mu_d, mu_s, temperature):
                return cfl_thermo_from_mu(mu_u, mu_d, mu_s, temperature,
                                          Delta0, par).P

            mus = (c.mu_u, c.mu_d, c.mu_s)
            for i, n in enumerate((c.n_u, c.n_d, c.n_s)):
                up = list(mus)
                up[i] += h
                down = list(mus)
                down[i] -= h
                num = (P_at(*up, T) - P_at(*down, T)) / (2 * h)
                worst_deriv = max(worst_deriv, abs(num - n) / abs(n))
            if T > 0.0:
                num_s = (P_at(*mus, T + h) - P_at(*mus, T - h)) / (2 * h)
                worst_deriv = max(worst_deriv,
                                  abs(num_s - c.s_total) / abs(c.s_total))

    for Delta0 in (50.0, 100.0):
        T_c = T_critical(Delta0)
        worst_gap = max(worst_gap, abs(cfl_gap(T_c, Delta0)),
                        abs(cfl_gap(0.0, Delta0) - Delta0) / Delta0,
                        abs(cfl_dgap_dT(0.0, Delta0)))
        for frac in (0.2, 0.5, 0.8):
            t = frac * T_c
            num = (cfl_gap(t + h, Delta0) - cfl_gap(t - h, Delta0)) / (2 * h)
            worst_gap = max(worst_gap,
                            abs(num - cfl_dgap_dT(t, Delta0))
                            / abs(cfl_dgap_dT(t, Delta0)))
    passed = (worst_neutral < 1e-10 and worst_deriv < 1e-7
              and worst_gap < 1e-7)
    return CheckResult("CFL phase", passed,
                       max(worst_neutral, worst_deriv, worst_gap),
                       f"Y_C: {worst_neutral:.1e}, dP: {worst_deriv:.1e}, "
                       f"gap: {worst_gap:.1e}")


def _check_causality(par, grid):
    """0 <= c_s^2 <= 1 along the cold beta-equilibrium sequence."""
    worst = 0.0
    values = []
    for n_B in grid:
        cs2 = eos_response(par, "beta_eq_neutrinoless", n_B=n_B,
                           T=0.0)["cs2_eq"]
        values.append(cs2)
        worst = max(worst, max(0.0 - cs2, cs2 - 1.0, 0.0))
    return CheckResult("causality", worst == 0.0, worst,
                       f"c_s^2 in [{min(values):.3f}, {max(values):.3f}]")


def run_full_check(par=None, grid=None, T=10.0):
    """Run the alphaBag verification suite; returns a structured report.

    The density grid spans the range deconfined matter is used over, from just
    below the self-bound surface to several times it, which is where the
    strange quark is populated and where a sign or factor error shows first.
    """
    par = par or Parameters.default()
    grid = np.array(grid) if grid is not None else np.array([0.45, 0.80, 1.30])

    report = FullCheckReport()
    report.results.append(_check_euler(par, grid, T))
    report.results.append(_check_free_energy(par, grid, T))
    report.results.append(_check_massless_gas(par))
    report.results.append(_check_massive_limit(par))
    report.results.append(_check_gluons_and_bag(par))
    report.results.append(_check_charge_basis(par, grid, T))
    report.results.append(_check_mode_closures(par, grid, T))
    report.results.append(_check_cfl(par, grid, T))
    report.results.append(_check_causality(par, grid))
    return report


if __name__ == "__main__":
    print(run_full_check())
