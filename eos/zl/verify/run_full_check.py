"""Physics invariants of the ZL model, checked in one place.

These are the statements the implementation has to satisfy no matter which
parameters it is given; they are the fastest way to catch a wrong change.
Every check returns a structured pass/fail with the largest error it saw, so
the suite reports rather than prints.

  1. Euler relation      eps + P = T s + mu_p n_p + mu_n n_n, in every mode.
  2. Free energy         f = eps - T s = -P + sum_i mu_i n_i.
  3. Interaction         mu_Hv_i = dV/dn_i (against a numerical derivative),
                         and P_int = sum_i n_i mu_Hv_i - V, which is the
                         identity that makes check 1 hold with no
                         rearrangement term left over.
  4. Mode closures       each mode's own conditions hold at its solution.
  5. Free gas limit      with a0 = b0 = a1 = b1 = 0 the model IS two free
                         Fermi gases at the physical potentials.
  6. Isospin symmetry    at Y_C = 0.5 the two species are interchangeable:
                         mu_p = mu_n and mu_C = 0.
  7. Residual gate       every solved state is inside RESIDUAL_TOL.
  8. Causality           0 <= c_s^2 <= 1 along a cold beta-equilibrium
                         sequence.
  9. No strangeness      fixed_YC_YS raises rather than ignoring Y_S.

Run as `python -m eos.zl.verify.run_full_check`.
"""
from dataclasses import dataclass, field

import numpy as np

from eos.general.thermodynamics_leptons import (
    electron_thermo, neutrino_thermo, photon_thermo,
)
from eos.zl import (
    MODE_FRACTIONS, Parameters, RESIDUAL_TOL, SpeciesFlags,
    eos_point, eos_response, interaction_energy, interaction_potentials,
    interaction_pressure, kinetic_thermo, solve_beta_eq_neutrinoless,
    solve_beta_eq_neutrino_trapped, solve_fixed_yc, solve_fixed_yc_ys,
    thermo_from_mu_n,
)
from eos.zl.nmp import compute_nmp


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
        lines = [f"ZL run_full_check: "
                 f"{'PASS' if self.all_passed else 'FAIL'}"]
        for r in self.results:
            tag = "ok " if r.passed else "FAIL"
            lines.append(f"  [{tag}] {r.name:24s} max_err={r.max_error:.2e}"
                         f"  {r.detail}")
        return "\n".join(lines)


def _matter_only(result):
    """(P, eps, s) of the nucleons alone, with leptons and photons removed.

    The Euler relation holds sector by sector; the solvers return totals, so
    the leptonic and radiative parts have to be subtracted before the nucleon
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


def _states(par, grid, T):
    """One solved state per mode, at each density of the grid."""
    out = []
    for n_B in grid:
        out.append(("beta", solve_beta_eq_neutrinoless(n_B, T, params=par)))
        out.append(("yc", solve_fixed_yc(n_B, 0.3, T, params=par)))
        out.append(("yc_nolep", solve_fixed_yc(n_B, 0.3, T, params=par,
                                               include_electrons=False)))
        out.append(("trapped", solve_beta_eq_neutrino_trapped(n_B, 0.4, T,
                                                              params=par)))
    return out


def _check_euler(par, grid, T):
    """eps + P = T s + mu_p n_p + mu_n n_n, at the PHYSICAL potentials."""
    worst = 0.0
    for _, r in _states(par, grid, T):
        P, eps, s = _matter_only(r)
        mu_n_sum = r.mu_p * r.n_p + r.mu_n * r.n_n
        worst = max(worst, abs(eps + P - T * s - mu_n_sum) / abs(eps))
    return CheckResult("Euler relation", worst < 1e-8, worst,
                       f"T={T} MeV, {4*len(grid)} states")


def _check_free_energy(par, grid, T):
    """f = eps - T s, and f = -P + sum_i mu_i n_i for it."""
    worst = 0.0
    for _, r in _states(par, grid, T):
        P, eps, s = _matter_only(r)
        f = eps - T * s
        mu_n_sum = r.mu_p * r.n_p + r.mu_n * r.n_n
        worst = max(worst, abs(f - (-P + mu_n_sum)) / abs(eps))
    return CheckResult("free energy", worst < 1e-8, worst,
                       "f = eps - Ts = -P + sum mu n")


def _check_interaction(par):
    """mu_Hv_i = dV/dn_i, and P_int = sum_i n_i mu_Hv_i - V.

    The first is a numerical derivative of `interaction_energy` against the
    closed form in `interaction_potentials`; the second is the identity that
    turns the closed form of `interaction_pressure` into the thermodynamic
    pressure, and is what leaves no rearrangement term outside P.
    """
    worst_deriv, worst_pressure = 0.0, 0.0
    for n_B in (0.05, 0.16, 0.5, 1.0):
        for Y_C in (0.0, 0.1, 0.5, 0.9, 1.0):
            n_p, n_n = Y_C * n_B, (1.0 - Y_C) * n_B
            mu_Hv_p, mu_Hv_n = interaction_potentials(n_p, n_n, par)

            h = 1e-6 * n_B
            dV_p = (interaction_energy(n_p + h, n_n, par)
                    - interaction_energy(n_p - h, n_n, par)) / (2 * h)
            dV_n = (interaction_energy(n_p, n_n + h, par)
                    - interaction_energy(n_p, n_n - h, par)) / (2 * h)
            scale = max(abs(mu_Hv_p), abs(mu_Hv_n), 1.0)
            worst_deriv = max(worst_deriv, abs(dV_p - mu_Hv_p) / scale,
                              abs(dV_n - mu_Hv_n) / scale)

            P_closed = interaction_pressure(n_p, n_n, par)
            P_legendre = (n_p * mu_Hv_p + n_n * mu_Hv_n
                          - interaction_energy(n_p, n_n, par))
            worst_pressure = max(worst_pressure,
                                 abs(P_closed - P_legendre)
                                 / max(abs(P_closed), 1e-30))
    passed = worst_deriv < 1e-6 and worst_pressure < 1e-12
    return CheckResult("interaction identities", passed,
                       max(worst_deriv, worst_pressure),
                       f"dV/dn_i: {worst_deriv:.1e}, "
                       f"P_int: {worst_pressure:.1e}")


def _check_mode_closures(par, grid, T):
    """Each mode's defining conditions, evaluated at its own solution."""
    worst = 0.0
    for n_B in grid:
        r = solve_beta_eq_neutrinoless(n_B, T, params=par)
        mu_scale = abs(r.mu_B)
        worst = max(worst,
                    abs(r.mu_C + r.mu_e) / mu_scale,   # beta equilibrium
                    abs(r.n_p - r.n_e) / n_B,          # electric neutrality
                    abs(r.n_p + r.n_n - n_B) / n_B)    # baryon number

        r = solve_fixed_yc(n_B, 0.3, T, params=par)
        worst = max(worst, abs(r.n_p - 0.3 * n_B) / n_B,
                    abs(r.n_p - r.n_e) / n_B)

        r = solve_fixed_yc(n_B, 0.3, T, params=par, include_electrons=False)
        # No neutrality here: the phase is charged, which is the point.
        worst = max(worst, abs(r.n_p - 0.3 * n_B) / n_B, abs(r.n_e))

        r = solve_beta_eq_neutrino_trapped(n_B, 0.4, T, params=par)
        worst = max(worst, abs(r.mu_C + r.mu_e - r.mu_nu) / abs(r.mu_B),
                    abs(r.n_p - r.n_e) / n_B,
                    abs((r.n_e + r.n_nu) / n_B - 0.4))
    return CheckResult("mode closures", worst < 1e-8, worst,
                       "beta, neutrality, baryon number, fixed fractions")


def _check_free_gas_limit(par, grid, T):
    """With every interaction coefficient zero the model is two free gases.

    This is the check that the interaction is an addition to a correct kinetic
    sector rather than a correction hiding one: turn it off and the solved
    state must equal the Fermi integrals evaluated at the physical potentials
    themselves.
    """
    free = Parameters(name="free", m_p=par.m_p, m_n=par.m_n, n0=par.n0,
                      a0=0.0, b0=0.0, gamma=par.gamma,
                      a1=0.0, b1=0.0, gamma1=par.gamma1)
    worst = 0.0
    for n_B in grid:
        r = solve_beta_eq_neutrinoless(n_B, T, params=free)
        gases = [kinetic_thermo(r.mu_p, T, free.m_p),
                 kinetic_thermo(r.mu_n, T, free.m_n)]
        P_kin = sum(g.P for g in gases)
        eps_kin = sum(g.e for g in gases)
        P, eps, _ = _matter_only(r)
        worst = max(worst, abs(P - P_kin) / abs(eps_kin),
                    abs(eps - eps_kin) / abs(eps_kin))
    return CheckResult("free-gas limit", worst < 1e-10, worst,
                       "a0 = b0 = a1 = b1 = 0 reproduces the Fermi integrals")


def _check_isospin_symmetry(par, grid, T):
    """At Y_C = 0.5 the two species are interchangeable.

    Both nucleons carry the same mass and the functional is symmetric under
    n_p <-> n_n, so symmetric matter must come back with mu_p = mu_n exactly,
    i.e. mu_C = 0. This catches a sign slip in the isovector term, which is
    otherwise invisible in the totals.
    """
    worst = 0.0
    for n_B in grid:
        r = solve_fixed_yc(n_B, 0.5, T, params=par, include_electrons=False)
        worst = max(worst, abs(r.mu_C) / abs(r.mu_B))
        block = thermo_from_mu_n(r.mu_p, r.mu_n, r.n_p, r.n_n, T, par)
        worst = max(worst, abs(block.n_p - block.n_n) / n_B)
    return CheckResult("isospin symmetry", worst < 1e-10, worst,
                       "Y_C = 0.5 gives mu_p = mu_n")


def _check_residual_gate(par, grid, T):
    """Every state this suite solved is inside the gate it claims."""
    states = _states(par, grid, T)
    worst = max(r.error for _, r in states if r.converged)
    n_bad = sum(1 for _, r in states if not r.converged)
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
                           T=0.0)["cs2_eq"]
        if not np.isfinite(cs2):
            return CheckResult("causality", False, float("inf"),
                               f"c_s^2 is not finite at n_B = {n_B:.3f} "
                               f"fm^-3: the response did not converge there, "
                               f"so this grid was not evaluated")
        values.append(cs2)
        worst = max(worst, max(0.0 - cs2, cs2 - 1.0, 0.0))
    return CheckResult("causality", worst == 0.0, worst,
                       f"c_s^2 in [{min(values):.3f}, {max(values):.3f}]")


def _check_no_strangeness(par):
    """fixed_YC_YS raises, and every solved state reports n_S = 0.

    The mode is meaningless rather than unimplemented: silently ignoring Y_S
    would return fixed_YC under a name that promised a strangeness condition.
    """
    raised_solver, raised_api = False, False
    try:
        solve_fixed_yc_ys(0.16, 0.3, 0.0, 10.0)
    except NotImplementedError:
        raised_solver = True
    try:
        eos_point(par, "fixed_YC_YS", SpeciesFlags(), n_B=0.16, T=10.0,
                  Y_C=0.3, Y_S=0.0)
    except NotImplementedError:
        raised_api = True
    listed = "fixed_YC_YS" not in MODE_FRACTIONS
    r = solve_beta_eq_neutrinoless(0.16, 10.0, params=par)
    n_S_zero = (r.Y_S == 0.0 and r.mu_S == 0.0)
    passed = raised_solver and raised_api and listed and n_S_zero
    return CheckResult("no strangeness", passed, 0.0 if passed else 1.0,
                       "fixed_YC_YS raises; n_S = mu_S = 0")


def _check_nmp(par):
    """The forward NMP map reproduces the published Constantinou et al. set.

    n_sat = 0.15951 fm^-3, E_sat = -16.00, K_sat = 250.2, E_sym = 30.85,
    L_sym = 41.26 MeV. These are quoted in `parameters.py` and in `zl.tex`;
    until `nmp.py` existed nothing in the repository reproduced them, so they
    were provenance rather than a check.

    Every one is a PREDICTION -- ZL imposes no saturation condition, so n_sat
    is found from P = 0 rather than declared. The tolerances are the published
    precision: the quoted values carry 4-5 significant figures, and the
    computed ones agree to the last of them.

    Q_sat and K_sym are computed too but not pinned: they are not in the
    published set, and a check cannot assert a number nobody published.
    """
    got = compute_nmp(par)
    want = {"n_sat": (0.15951, 5e-5), "E_sat": (-16.00, 5e-3),
            "K_sat": (250.2, 5e-2), "E_sym": (30.85, 5e-3),
            "L_sym": (41.26, 2e-2)}
    worst, failed = 0.0, []
    for name, (target, tol) in want.items():
        delta = abs(got[name] - target)
        worst = max(worst, delta / tol)
        if delta > tol:
            failed.append(f"{name}={got[name]:.5f} vs {target} (|d|={delta:.2e})")
    passed = not failed
    return CheckResult("nuclear-matter parameters", passed, worst,
                       "published NMPs reproduced" if passed
                       else "; ".join(failed))


def run_full_check(par=None, grid=None, T=10.0):
    """Run the ZL verification suite; returns a structured report.

    The density grid spans from below saturation to several times it, which is
    where the power-law terms of the functional dominate and where a sign or
    exponent error shows up first.
    """
    par = par or Parameters.default()
    grid = np.array(grid) if grid is not None else np.array([0.08, 0.16, 0.48])

    report = FullCheckReport()
    report.results.append(_check_euler(par, grid, T))
    report.results.append(_check_free_energy(par, grid, T))
    report.results.append(_check_interaction(par))
    report.results.append(_check_mode_closures(par, grid, T))
    report.results.append(_check_free_gas_limit(par, grid, T))
    report.results.append(_check_isospin_symmetry(par, grid, T))
    report.results.append(_check_residual_gate(par, grid, T))
    report.results.append(_check_causality(par, grid))
    report.results.append(_check_no_strangeness(par))
    report.results.append(_check_nmp(par))
    return report


if __name__ == "__main__":
    print(run_full_check())
