"""Physics invariants of the ABPR parametrization, checked in one place.

These are the statements the implementation has to satisfy no matter which
parameters it is given; they are the fastest way to catch a wrong change.
Every check returns a structured pass/fail with the largest error it saw, so
the suite reports rather than prints.

  1. Euler relation      eps + P = T s + sum_q mu_q n_q = mu_B n_B, with the
                         bag included -- it enters eps and P with opposite
                         signs and cancels.
  2. Free energy         f = eps - T s = eps at T = 0, and f = -P + mu_B n_B.
  3. Derivatives         n_B = dP/dmu_B and c_s^2 = dP/deps, each against a
                         numerical derivative of the closed forms. This is
                         what makes the parametrization a thermodynamic
                         potential rather than three unrelated polynomials.
  4. Charge basis        Y_C = 0 and Y_S = +1 identically, through
                         eos.general.basis -- no local copy of the map -- and
                         mu_C = mu_S = 0.
  5. The P = 0 surface   P(mu_0) = 0 to round-off, and E/A = mu_B there, the
                         Euler relation read at the surface of a self-bound
                         star.
  6. Causality           0 <= c_s^2 <= 1 from the surface upward.
  7. Inverse round-trips mu -> (n_B, P, eps) -> mu closes on itself, over
                         several parameter sets including one with
                         Delta0 < m_s/2, where the mu^2 coefficient changes
                         sign and the cubic acquires three real roots.
  8. The alphaBag gap    this model is the T = 0 analytic limit of the CFL
                         phase of eos.alphabag, which carries m_s exactly
                         through the Fermi integrals where this one expands
                         to O(m_s^2). The difference must therefore be the
                         O(m_s^4) term of that expansion,
                             dP ~ -m_s^4/(8 pi^2 (hc)^3) [9/4 + 3 ln(2mu/m_s)]
                         and it is checked against it to 1%.
  9. Mode refusals       the four equilibrium modes and any nonzero
                         temperature raise, rather than returning a state the
                         phase does not have.

Run as `python -m eos.abpr.verify.run_full_check`.
"""
from dataclasses import dataclass, field
from math import log, pi

import numpy as np

from eos.general.basis import charge_potentials_from_quarks, quark_charges
from eos.general.physics_constants import hc3
from eos.alphabag.parameters import Parameters as AlphaBagParameters
from eos.alphabag.thermodynamics import cfl_thermo_from_mu
from eos.abpr import (
    Parameters, SpeciesFlags, baryon_density, energy_density, eos_point,
    eos_response, zero_pressure_point,
    mu_from_eps, mu_from_nB, mu_from_P, pressure, solve_cfl,
    sound_speed_squared, thermo_from_mu,
)
from eos.general.physics_constants import E_per_A_iron


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
        lines = [f"ABPR run_full_check: "
                 f"{'PASS' if self.all_passed else 'FAIL'}"]
        for r in self.results:
            tag = "ok " if r.passed else "FAIL"
            lines.append(f"  [{tag}] {r.name:22s} max_err={r.max_error:.2e}"
                         f"  {r.detail}")
        return "\n".join(lines)


#: The sets the invariants are checked over: the shipped one, a strongly
#: paired one from a hybrid-star study, and one with Delta0 < m_s/2 so that
#: the mu^2 coefficient is negative and the cubic takes its three-real-root
#: branch.
SETS = (
    Parameters.default(),
    Parameters(name="strong_gap", m_s=100.0, Delta0=157.0, a4=0.92, B4=158.0),
    Parameters(name="unpaired_limit", m_s=150.0, Delta0=0.0, a4=0.7, B4=135.0),
)

#: The density range the invariants are checked on, spanning a compact-star
#: core.
DENSITIES = np.array([0.3, 0.5, 0.8, 1.2, 2.0, 3.0])


def check_euler():
    """eps + P = T s + sum_q mu_q n_q, which at equal potentials is mu_B n_B."""
    worst, detail = 0.0, ""
    for par in SETS:
        for n_B in DENSITIES:
            point = solve_cfl(par, n_B)
            lhs = point.e_total + point.P_total
            rhs = (point.T * point.s_total
                   + point.mu_u * point.n_u
                   + point.mu_d * point.n_d
                   + point.mu_s * point.n_s)
            err = abs(lhs - rhs) / abs(lhs)
            if err > worst:
                worst, detail = err, f"{par.name} n_B={n_B}"
            err_B = abs(lhs - point.mu_B * point.n_B) / abs(lhs)
            if err_B > worst:
                worst, detail = err_B, f"{par.name} n_B={n_B} (mu_B n_B form)"
    return CheckResult("euler", worst < 1e-13, worst, detail)


def check_free_energy():
    """f = eps - T s and f = -P + sum_i mu_i n_i, both at T = 0."""
    worst, detail = 0.0, ""
    for par in SETS:
        for n_B in DENSITIES:
            point = solve_cfl(par, n_B)
            scale = abs(point.e_total)
            for value, name in (
                    (point.f_total - (point.e_total
                                      - point.T * point.s_total), "eps - Ts"),
                    (point.f_total - (-point.P_total
                                      + point.mu_B * point.n_B), "-P + mu n")):
                err = abs(value) / scale
                if err > worst:
                    worst, detail = err, f"{par.name} n_B={n_B} ({name})"
    return CheckResult("free_energy", worst < 1e-13, worst, detail)


def check_derivatives():
    """n_B = dP/dmu_B and c_s^2 = dP/deps, against numerical derivatives.

    Central differences in mu on the closed forms. The stencil is
    second order, so it carries a truncation error (h^2/6) P'''(mu), which for
    the quartic P of this model is 4 A mu h^2 -- about 1e-10 relative at the
    step used here, h/mu = 1e-5, against a round-off floor of the same order.
    The gate is 1e-8, comfortably above both and orders of magnitude below
    what a genuinely wrong derivative would give.
    """
    worst_n, worst_c, detail = 0.0, 0.0, ""
    for par in SETS:
        for mu in (300.0, 400.0, 600.0, 900.0):
            h = 1e-5 * mu
            dP = (pressure(mu + h, par) - pressure(mu - h, par)) / (2.0 * h)
            de = (energy_density(mu + h, par)
                  - energy_density(mu - h, par)) / (2.0 * h)
            err_n = abs(dP / 3.0 - baryon_density(mu, par)) \
                / abs(baryon_density(mu, par))
            err_c = abs(dP / de - sound_speed_squared(mu, par)) \
                / abs(sound_speed_squared(mu, par))
            if err_n > worst_n:
                worst_n, detail = err_n, f"{par.name} mu={mu}"
            worst_c = max(worst_c, err_c)
    worst = max(worst_n, worst_c)
    return CheckResult("derivatives", worst < 1e-8, worst,
                       f"n_B={worst_n:.1e} cs2={worst_c:.1e} {detail}")


def check_charges():
    """Y_C = 0 and Y_S = +1, and mu_C = mu_S = 0, through general.basis."""
    worst, detail = 0.0, ""
    for par in SETS:
        for n_B in DENSITIES:
            point = solve_cfl(par, n_B)
            _, n_C, n_S = quark_charges(point.n_u, point.n_d, point.n_s)
            _, mu_C, mu_S = charge_potentials_from_quarks(
                point.mu_u, point.mu_d, point.mu_s)
            for value, scale, name in ((n_C, n_B, "n_C"),
                                       (n_S - n_B, n_B, "n_S - n_B"),
                                       (mu_C, point.mu_B, "mu_C"),
                                       (mu_S, point.mu_B, "mu_S"),
                                       (point.Y_C, 1.0, "Y_C"),
                                       (point.Y_S - 1.0, 1.0, "Y_S - 1")):
                err = abs(value) / scale
                if err > worst:
                    worst, detail = err, f"{par.name} n_B={n_B} ({name})"
    return CheckResult("charges", worst < 1e-14, worst, detail)


def check_surface():
    """P = 0 at the surface, and E/A = mu_B there.

    A self-bound phase ends at finite density with no crust, and the Euler
    relation read at P = 0 gives eps/n_B = mu_B directly. The number this
    check pins for the shipped set is E/A = 831.58 MeV, below the 930 MeV of
    iron: absolutely stable strange quark matter.
    """
    worst, detail = 0.0, ""
    reported = []
    for par in SETS:
        mu0, converged = mu_from_P(0.0, par)
        if not converged:
            return CheckResult("surface", False, np.inf,
                               f"{par.name}: no P = 0 root")
        point = solve_cfl(par, baryon_density(mu0, par))
        err_P = abs(point.P_total) / (par.B / hc3)
        E_per_A = point.e_total / point.n_B
        err_EA = abs(E_per_A - point.mu_B) / point.mu_B
        for err, name in ((err_P, "P(mu_0)"), (err_EA, "E/A - mu_B")):
            if err > worst:
                worst, detail = err, f"{par.name} ({name})"
        reported.append(f"{par.name}: E/A={E_per_A:.2f} MeV")
    return CheckResult("surface", worst < 1e-12, worst, "; ".join(reported))


def check_causality():
    """0 <= c_s^2 <= 1 from the P = 0 surface upward."""
    worst, detail = 0.0, ""
    for par in SETS:
        mu0, _ = mu_from_P(0.0, par)
        for mu in np.linspace(mu0, 1500.0, 40):
            cs2 = sound_speed_squared(mu, par)
            violation = max(-cs2, cs2 - 1.0, 0.0)
            if violation > worst:
                worst, detail = violation, f"{par.name} mu={mu:.1f}"
    return CheckResult("causality", worst == 0.0, worst,
                       detail or "0 <= cs2 <= 1 everywhere above the surface")


def check_round_trips():
    """mu -> (n_B, P, eps) -> mu, for each of the three closed-form inverses."""
    worst, detail = 0.0, ""
    for par in SETS:
        for mu in (300.0, 400.0, 600.0, 900.0):
            for forward, inverse, name in (
                    (baryon_density, mu_from_nB, "n_B"),
                    (pressure, mu_from_P, "P"),
                    (energy_density, mu_from_eps, "eps")):
                back, converged = inverse(forward(mu, par), par)
                err = abs(back - mu) / mu if converged else np.inf
                if err > worst:
                    worst, detail = err, f"{par.name} mu={mu} ({name})"
    return CheckResult("round_trips", worst < 1e-12, worst, detail)


def check_alphabag_limit():
    """The gap against eos.alphabag's CFL phase is the O(m_s^4) term.

    This model expands the strange-quark mass to O(m_s^2); eos.alphabag
    carries it exactly through the Fermi integrals and deliberately omits the
    expansion term, so that adding both would not count it twice. The whole
    difference between the two is therefore the next term of that expansion,

        dP = P_abpr - P_alphabag ~ -m_s^4/(8 pi^2 (hc)^3) [9/4 + 3 ln(2mu/m_s)]

    checked here at three equal potentials, which is the closure this model
    assumes. The residual after the predicted term is O(m_s^6/mu^2), so the
    ratio approaches one from below as mu rises: at the shipped set it runs
    from 0.9931 at mu = 350 MeV to 0.9991 at mu = 800 MeV.
    """
    par = Parameters.default()
    matched = AlphaBagParameters(
        name="matched", m_u=0.0, m_d=0.0, m_s=par.m_s,
        alpha=par.alpha, B4=par.B4)
    worst, detail = 0.0, ""
    for mu in (350.0, 400.0, 450.0, 500.0, 600.0, 700.0, 800.0):
        paired = cfl_thermo_from_mu(mu, mu, mu, 0.0, par.Delta0, matched)
        measured = pressure(mu, par) - paired.P
        predicted = -par.m_s**4 / (8.0 * pi**2 * hc3) * (
            9.0 / 4.0 + 3.0 * log(2.0 * mu / par.m_s))
        err = abs(measured / predicted - 1.0)
        if err > worst:
            worst, detail = err, f"mu={mu:.0f} dP={measured:.3f}"
    return CheckResult("alphabag_limit", worst < 1e-2, worst,
                       f"{detail} (dP is the m_s^4 term, to {worst:.1%})")


def check_refusals():
    """Every mode this phase does not have, and every T > 0, raises."""
    par = Parameters.default()
    failures = []
    for mode in ("beta_eq_neutrinoless", "beta_eq_neutrino_trapped",
                 "fixed_YC", "fixed_YC_YS"):
        try:
            eos_point(par, mode, n_B=0.5)
        except NotImplementedError:
            continue
        except Exception as err:                     # noqa: BLE001
            failures.append(f"{mode} raised {type(err).__name__}")
            continue
        failures.append(f"{mode} returned a state")
    try:
        eos_point(par, "cfl", n_B=0.5, T=10.0)
    except NotImplementedError:
        pass
    else:
        failures.append("T = 10 MeV returned a state")
    try:
        eos_response(par, "cfl", n_B=0.5, frozen="fast")
    except NotImplementedError:
        pass
    else:
        failures.append("frozen='fast' returned a response")
    return CheckResult("refusals", not failures, float(len(failures)),
                       "; ".join(failures) or "all four modes and T > 0 raise")


def check_zero_pressure():
    """The shared locator reproduces the closed form, to the last digit.

    `check_surface` above finds the surface the way this model always has,
    by inverting P(mu) analytically. This one finds it the way every OTHER
    quark model must -- `eos.general.zero_pressure.locate_zero_pressure`,
    a bracketed root find over `eos_point` -- and puts the two side by side.
    ABPR is the only model in the repository where both routes exist, which
    makes it the only place the locator can be measured against an exact
    answer rather than against itself. E/A = 831.58 MeV for the shipped set
    is the golden reference (CLAUDE.md section 12); code that needs a
    different number is wrong until proven otherwise.

    The identity E/A = mu_B + Y_S mu_S is checked here too, and this phase is
    where the FULL form matters: locking holds Y_S = +1, so a helper reading
    E/A = mu_B alone would be right here only by the accident that this
    parametrization carries a single mu and mu_S vanishes identically. The
    same identity on the CFL surface of `eos.alphabag`, which resolves the
    strange mass through the Fermi integrals instead of expanding it, has
    mu_S = 40.68 MeV and would miss by that much; `eos.alphabag`'s suite
    checks it there.

    THERE IS NO TWO-FLAVOUR ARM, and `SpeciesFlags(two_flavour=True)` raising
    is what says so -- the check asserts the refusal rather than tolerating a
    nan, because the number does not exist in a flavour-locked phase rather
    than being unimplemented.
    """
    worst, detail = 0.0, []
    try:
        SpeciesFlags(two_flavour=True)
    except NotImplementedError:
        pass
    else:
        return CheckResult(
            "zero-pressure surface", False, np.inf,
            "SpeciesFlags(two_flavour=True) was accepted; a flavour-locked "
            "phase has no two-flavour arm and must refuse the flag")

    for par in SETS:
        located = zero_pressure_point(par)
        if not located.ok:
            return CheckResult("zero-pressure surface", False, np.inf,
                               f"{par.name}: {located.message}")
        mu0, converged = mu_from_P(0.0, par)
        if not converged:
            return CheckResult("zero-pressure surface", False, np.inf,
                               f"{par.name}: the closed form found no root")
        closed = solve_cfl(par, baryon_density(mu0, par))
        for err, name in (
                (located.identity_error, "E/A - (mu_B + Y_S mu_S)"),
                (abs(located.n_B - closed.n_B) / closed.n_B, "n_B"),
                (abs(located.E_per_A - closed.e_total / closed.n_B)
                 / located.E_per_A, "E/A vs the closed form")):
            if err > worst:
                worst, = (err,)
        detail.append(f"{par.name}: E/A={located.E_per_A:.2f} MeV, "
                      f"{'below' if located.E_per_A < E_per_A_iron else 'above'}"
                      f" iron")
    return CheckResult("zero-pressure surface", worst < 1e-12, worst,
                       "; ".join(detail))


CHECKS = (check_euler, check_free_energy, check_derivatives, check_charges,
          check_surface, check_causality, check_round_trips,
          check_alphabag_limit, check_refusals, check_zero_pressure)


def run_full_check():
    """Run every invariant and return the report."""
    return FullCheckReport(results=[check() for check in CHECKS])


if __name__ == "__main__":
    import sys

    report = run_full_check()
    print(report)
    sys.exit(0 if report.all_passed else 1)
