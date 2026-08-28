"""Physics invariants of the vMIT model, checked in one place.

These are the statements the implementation has to satisfy no matter which
parameters it is given; they are the fastest way to catch a wrong change.
Every check returns a structured pass/fail with the largest error it saw, so
the suite reports rather than prints.

  1. Euler relation      eps + P = T s + sum_q mu_q n_q, in every mode.
  2. Free energy         f = eps - T s.
  3. Mean field          eps_V = P_V, the bag enters eps and P with opposite
                         signs, and V = a hbar c sum_q n_q is self-consistent.
  4. Mode closures       each mode's own conditions hold at its solution.
  5. Free-gas limit      with a = 0 and B = 0 the model IS a free quark gas.
  6. Residual gate       every solved state is inside RESIDUAL_TOL.
  7. Causality           0 <= c_s^2 <= 1 along a cold beta-equilibrium
                         sequence.
"""
from dataclasses import dataclass, field

import numpy as np

from eos.general.basis import quark_charges, charge_potentials_from_quarks
from eos.general.physics_constants import hc, hc3
from eos.vmit import (
    zero_pressure_point,
    RESIDUAL_TOL, SpeciesFlags, Parameters, eos_point, eos_response,
    kinetic_thermo, vector_field, Parameters,
    solve_beta_eq_neutrinoless, solve_fixed_yc, solve_fixed_yc_ys,
    solve_beta_eq_neutrino_trapped,
)
from eos.general.thermodynamics_leptons import (
    electron_thermo, neutrino_thermo, photon_thermo,
)


#: Every solver call in this suite carries the photon gas, deliberately:
#: `_quark_only` subtracts exactly one photon gas off each state before the
#: Euler and free-energy identities are tested, so the gas has to be added
#: and removable rather than woven into the equations.
GAMMA = SpeciesFlags(photons=True)


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
        lines = [f"vMIT run_full_check: "
                 f"{'PASS' if self.all_passed else 'FAIL'}"]
        for r in self.results:
            tag = "ok " if r.passed else "FAIL"
            lines.append(f"  [{tag}] {r.name:22s} max_err={r.max_error:.2e}"
                         f"  {r.detail}")
        return "\n".join(lines)


def _quark_only(result, par):
    """(P, eps, s) of the quarks alone, with leptons and photons removed.

    The Euler relation holds sector by sector; the solvers return totals, so
    the leptonic and radiative parts have to be subtracted before the quark
    identity can be tested on its own.
    """
    P, eps, s = result.P_total, result.e_total, result.s_total
    T = result.T
    if result.n_e != 0.0 or result.mu_e != 0.0:
        e_thermo = electron_thermo(result.mu_e, T, include_antiparticles=True)
        P, eps, s = P - e_thermo.P, eps - e_thermo.e, s - e_thermo.s
    if result.n_nu != 0.0 or result.mu_nu != 0.0:
        nu = neutrino_thermo(result.mu_nu, T, include_antiparticles=True)
        P, eps, s = P - nu.P, eps - nu.e, s - nu.s
    gamma = photon_thermo(T)
    # Photons are on by default in every solver call this suite makes.
    return P - gamma.P, eps - gamma.e, s - gamma.s


def _euler_error(result, par):
    """|eps + P - T s - sum_q mu_q n_q| / eps for the quark sector."""
    P, eps, s = _quark_only(result, par)
    mu_n = (result.mu_u * result.n_u + result.mu_d * result.n_d
            + result.mu_s * result.n_s)
    return abs(eps + P - result.T * s - mu_n) / abs(eps)


def _states(par, grid, T):
    """One solved state per mode, at each density of the grid."""
    out = []
    for n_B in grid:
        out.append(("beta", solve_beta_eq_neutrinoless(par, n_B, T, GAMMA)))
        out.append(("yc", solve_fixed_yc(par, n_B, 0.3, T, GAMMA)))
        out.append(("yc_nolep", solve_fixed_yc(par, n_B, 0.3, T, GAMMA,
                                               leptons=False)))
        out.append(("ycys", solve_fixed_yc_ys(par, n_B, 0.0, 1.0, T, GAMMA)))
        out.append(("trapped", solve_beta_eq_neutrino_trapped(par, n_B, 0.4, T,
                                                              GAMMA)))
    return out


def _check_euler(par, grid, T):
    errs = [_euler_error(r, par) for _, r in _states(par, grid, T)]
    worst = max(errs)
    return CheckResult("Euler relation", worst < 1e-8, worst,
                       f"{len(errs)} states, T={T} MeV")


def _check_free_energy(par, grid, T):
    """f = eps - T s, the definition, and f = -P + sum_i mu_i n_i for it."""
    worst = 0.0
    for _, r in _states(par, grid, T):
        P, eps, s = _quark_only(r, par)
        f = eps - T * s
        mu_n = (r.mu_u * r.n_u + r.mu_d * r.n_d + r.mu_s * r.n_s)
        worst = max(worst, abs(f - (-P + mu_n)) / abs(eps))
    return CheckResult("free energy", worst < 1e-8, worst,
                       "f = eps - Ts = -P + sum mu n")


def _check_mean_field(par, grid, T):
    """The vector field is self-consistent and enters eps and P equally; the
    bag enters them oppositely."""
    worst = 0.0
    for _, r in _states(par, grid, T):
        V_direct = par.a * hc * (r.n_u + r.n_d + r.n_s)
        V_module = vector_field(r.n_u, r.n_d, r.n_s, par)
        worst = max(worst, abs(V_direct - V_module) / max(abs(V_direct), 1.0))
        # mu_eff = mu - V must reproduce the density the solver settled on
        for mu, n, m in ((r.mu_u, r.n_u, par.m_u), (r.mu_d, r.n_d, par.m_d),
                         (r.mu_s, r.n_s, par.m_s)):
            n_from_mu = kinetic_thermo(mu - V_module, T, m).n
            worst = max(worst, abs(n_from_mu - n) / max(abs(n), 1e-12))
    return CheckResult("vector self-consistency", worst < 1e-8, worst,
                       "V = a hbar c sum n_q, n_q = n(mu_q - V)")


def _check_bag_signs(par):
    """eps_B = -P_B = B/(hbar c)^3, and eps_V = P_V."""
    from eos.vmit import (bag_energy, bag_pressure,
                          vector_energy, vector_pressure)
    e_B, P_B = bag_energy(par), bag_pressure(par)
    errs = [abs(e_B + P_B) / abs(e_B),
            abs(e_B - par.B / hc3) / abs(e_B)]
    for n in (0.3, 1.0, 2.0):
        e_V = vector_energy(n, n, n, par)
        P_V = vector_pressure(n, n, n, par)
        errs.append(abs(e_V - P_V) / max(abs(e_V), 1e-30))
    worst = max(errs)
    return CheckResult("bag / vector signs", worst < 1e-14, worst,
                       "eps_B = -P_B, eps_V = +P_V")


def _check_mode_closures(par, grid, T):
    """Each mode's defining conditions, evaluated at its own solution."""
    worst = 0.0
    for n_B in grid:
        r = solve_beta_eq_neutrinoless(par, n_B, T, GAMMA)
        _, mu_C, mu_S = charge_potentials_from_quarks(r.mu_u, r.mu_d, r.mu_s)
        _, n_C, _ = quark_charges(r.n_u, r.n_d, r.n_s)
        mu_scale = abs(r.mu_B)
        worst = max(worst,
                    abs(mu_C + r.mu_e) / mu_scale,     # beta equilibrium
                    abs(mu_S) / mu_scale,              # strangeness equilibrium
                    abs(n_C - r.n_e) / n_B)            # electric neutrality

        r = solve_fixed_yc(par, n_B, 0.3, T, GAMMA)
        _, n_C, _ = quark_charges(r.n_u, r.n_d, r.n_s)
        _, _, mu_S = charge_potentials_from_quarks(r.mu_u, r.mu_d, r.mu_s)
        worst = max(worst, abs(n_C - 0.3 * n_B) / n_B,
                    abs(mu_S) / abs(r.mu_B), abs(n_C - r.n_e) / n_B)

        r = solve_fixed_yc_ys(par, n_B, 0.0, 1.0, T, GAMMA)
        _, n_C, n_S = quark_charges(r.n_u, r.n_d, r.n_s)
        worst = max(worst, abs(n_C - 0.0) / n_B, abs(n_S - 1.0 * n_B) / n_B)

        r = solve_beta_eq_neutrino_trapped(par, n_B, 0.4, T, GAMMA)
        _, mu_C, _ = charge_potentials_from_quarks(r.mu_u, r.mu_d, r.mu_s)
        worst = max(worst, abs(mu_C + r.mu_e - r.mu_nu) / abs(r.mu_B),
                    abs((r.n_e + r.n_nu) / n_B - 0.4))
    return CheckResult("mode closures", worst < 1e-8, worst,
                       "beta, strangeness, neutrality, fixed fractions")


def _check_free_gas_limit(par, grid, T):
    """With a = 0 and B = 0 the model is three free Fermi gases.

    This is the check that the interaction terms are additions to a correct
    kinetic sector rather than a correction hiding one: turn them off and the
    solved state must equal the Fermi integrals evaluated at the physical
    potentials themselves.
    """
    free = Parameters(name="free", m_u=par.m_u, m_d=par.m_d, m_s=par.m_s,
                      a=0.0, B4=0.0)
    worst = 0.0
    for n_B in grid:
        r = solve_beta_eq_neutrinoless(free, n_B, T, GAMMA)
        gases = [kinetic_thermo(mu, T, m) for mu, m in
                 ((r.mu_u, free.m_u), (r.mu_d, free.m_d), (r.mu_s, free.m_s))]
        P_kin = sum(g.P for g in gases)
        eps_kin = sum(g.e for g in gases)
        P_q, eps_q, _ = _quark_only(r, free)
        worst = max(worst, abs(P_q - P_kin) / abs(eps_kin),
                    abs(eps_q - eps_kin) / abs(eps_kin))
    return CheckResult("free-gas limit", worst < 1e-10, worst,
                       "a = 0, B = 0 reproduces the Fermi integrals")


def _check_residual_gate(par, grid, T):
    """Every state this suite solved is inside the gate it claims."""
    worst = max(r.error for _, r in _states(par, grid, T) if r.converged)
    n_bad = sum(1 for _, r in _states(par, grid, T) if not r.converged)
    return CheckResult("residual gate", n_bad == 0 and worst <= RESIDUAL_TOL,
                       worst, f"{n_bad} unconverged, tol {RESIDUAL_TOL:.0e}")


def _check_causality(par, grid):
    """0 <= c_s^2 <= 1 along the cold beta-equilibrium sequence.

    A grid point the response cannot reach comes back as nan rather than an
    exception (CLAUDE.md section 6), and nan loses every comparison -- so
    `max` would propagate the incumbent and the point would be absorbed, the
    check passing over a grid it never evaluated. It is failed explicitly
    instead, naming the density: a check that cannot fail is not a check.
    That guard also makes the min/max in the message below safe, since no
    non-finite value can reach `values`.
    """
    worst = 0.0
    values = []
    for n_B in grid:
        cs2 = eos_response(par, "beta_eq_neutrinoless", n_B=n_B,
                           T=0.0)["cs2_isothermal"]
        if not np.isfinite(cs2):
            return CheckResult("causality", False, float("inf"),
                               f"c_s^2 is not finite at n_B = {n_B:.3f} "
                               f"fm^-3: the response did not converge there, "
                               f"so this grid was not evaluated")
        values.append(cs2)
        worst = max(worst, max(0.0 - cs2, cs2 - 1.0, 0.0))
    return CheckResult("causality", worst == 0.0, worst,
                       f"c_s^2 in [{min(values):.3f}, {max(values):.3f}]")


def _check_zero_pressure(par):
    """E/A = mu_B + Y_S mu_S at the self-bound surface, both flavour contents.

    THE IDENTITY IS THE INVARIANT; THE ENERGIES ARE REPORTED. At T = 0 the
    Euler relation read at P = 0 gives eps/n_B as the Gibbs energy per baryon
    exactly, so a located root that misses `mu_B + Y_S mu_S` is a root of
    something other than P -- the one failure a locator driven by a callable
    can have that nothing else would catch.

    WHERE EACH ENERGY SITS AGAINST IRON IS NOT ASSERTED. Three-flavour E/A
    below the 930.4 MeV of iron means absolutely stable strange quark matter,
    and two-flavour E/A above it means ordinary nuclei are safe, but both are
    properties of the PARAMETER SET: a legitimately excluded point is a normal
    draw for a sampler, and a suite that failed on one would be asserting the
    Bodmer-Witten hypothesis rather than checking an implementation. The
    numbers go in the detail line instead.

    A set with no self-bound surface, and a flavour content this phase
    refuses, are reported the same way and fail nothing.
    """
    worst, detail = 0.0, []
    for two_flavour in (False, True):
        try:
            flags = SpeciesFlags(two_flavour=two_flavour)
        except NotImplementedError:
            detail.append("two-flavour: no such arm in this phase")
            continue
        surface = zero_pressure_point(par, flags)
        label = "two" if two_flavour else "three"
        if not surface.ok:
            detail.append(f"{label}-flavour: {surface.message}")
            continue
        worst = max(worst, surface.identity_error)
        detail.append(
            f"{label}-flavour: E/A={surface.E_per_A:.2f} MeV at "
            f"n_B={surface.n_B:.4f} fm^-3, Y_S={surface.Y_S:.4f}, "
            f"{'below' if surface.below_iron else 'above'} iron")
    return CheckResult("zero-pressure surface", worst < 1e-12, worst,
                       "; ".join(detail))


def run_full_check(par=None, grid=None, T=10.0):
    """Run the vMIT verification suite; returns a structured report.

    The density grid stays above the bag's unbinding point, where quark matter
    exists at all: below it there is no positive-pressure solution to check.
    """
    par = par or Parameters.default()
    grid = np.array(grid) if grid is not None else np.array([0.45, 0.80, 1.30])

    report = FullCheckReport()
    report.results.append(_check_euler(par, grid, T))
    report.results.append(_check_free_energy(par, grid, T))
    report.results.append(_check_mean_field(par, grid, T))
    report.results.append(_check_bag_signs(par))
    report.results.append(_check_mode_closures(par, grid, T))
    report.results.append(_check_free_gas_limit(par, grid, T))
    report.results.append(_check_residual_gate(par, grid, T))
    report.results.append(_check_causality(par, grid))
    report.results.append(_check_zero_pressure(par))
    return report


if __name__ == "__main__":
    print(run_full_check())
