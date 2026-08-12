"""
verify/run_full_check.py
====================
Single entry point orchestrating the DD2 verification suite.
Each check returns a structured pass/fail + max error, never a bare print.

Checks (those with data/deps available in eos_ref):
  1. golden points — the day-one smoke test;
  2. thermodynamic identities — HVH, beta-eq residual across a grid;
  3. TOV cross-check — M_max >= 2 M_sun for the cold beta-eq EoS;
  4. coefficient sanity — causality 0 <= c_s^2 <= 1, thermal index > 1;
  5. CompOSE comparison — nucleonic HS(DD2) slice (< 1e-3) when the table is
     present.
Backend parity (eos_ref vs eos_fast) is the M9 check, added with that backend.
"""
from dataclasses import dataclass, field
import os

import numpy as np

from eos.dd2 import (
    Parametrization, SpeciesFlags, solve_snm, solve_beta_eq_octet, solve_octet,
)
from eos.dd2.responses import sound_speed_eq, thermal_index
from eos.dd2.verify.tov import mass_radius


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
        lines = [f"DD2 run_full_check: {'PASS' if self.all_passed else 'FAIL'}"]
        for r in self.results:
            tag = "ok " if r.passed else "FAIL"
            lines.append(f"  [{tag}] {r.name:22s} max_err={r.max_error:.2e}"
                         f"  {r.detail}")
        return "\n".join(lines)


# (b) golden SNM row at n_B = 0.16
_GOLDEN_SNM_016 = dict(mstar_m=0.54247, eps=147.6750, P=0.34289, mu_B=925.112)


def _check_golden(par):
    p = solve_snm(par, 0.16)
    errs = {
        "m*/m": abs(p.m_eff / par.m_nucleon / _GOLDEN_SNM_016["mstar_m"] - 1),
        "eps": abs(p.eps / _GOLDEN_SNM_016["eps"] - 1),
        "P": abs(p.P / _GOLDEN_SNM_016["P"] - 1),
        "mu_B": abs(p.mu_n / _GOLDEN_SNM_016["mu_B"] - 1),
    }
    m = max(errs.values())
    return CheckResult("golden points", m < 1e-4, m,
                       f"SNM(0.16) vs(b)")


def _check_identities(par, flags, grid):
    worst = 0.0
    for n_B in grid:
        p = solve_beta_eq_octet(par, n_B, flags)
        worst = max(worst, abs(p.hvh_rel), abs(p.mu_n - p.mu_p - p.mu_e) / p.mu_n)
    return CheckResult("thermo identities", worst < 1e-8, worst,
                       "HVH + beta residual over grid")


def _check_tov(par, flags):
    r = mass_radius(par, flags)
    return CheckResult("TOV M_max>=2", r["M_max"] >= 2.0,
                       max(0.0, 2.0 - r["M_max"]),
                       f"M_max={r['M_max']:.3f} R_1.4={r['R_1p4']:.2f}km")


def _check_responses(par, flags, grid):
    worst_cs = 0.0
    bad = False
    for n_B in grid:
        cs2 = sound_speed_eq(par, n_B, flags)
        if not (0.0 <= cs2 <= 1.0):
            bad = True
        worst_cs = max(worst_cs, cs2)
    gth = thermal_index(par, 0.16, flags, T=10.0)
    ok = (not bad) and (1.0 < gth < 2.5)
    return CheckResult("responses", ok, worst_cs,
                       f"max c_s^2={worst_cs:.3f} Gamma_th={gth:.3f}")


def _check_coeff_cross(par, flags, grid):
    """Analytic-from-Jacobian c_s^2 vs finite-difference:
    independent-method agreement on the equilibrium sound speed."""
    from eos.dd2.backends import responses_jac as _jc
    from eos.dd2.responses import sound_speed_eq as _cs_fd
    worst = 0.0
    for n_B in grid:
        cj = _jc.sound_speed_eq(par, float(n_B), flags)
        cf = _cs_fd(par, float(n_B), flags)
        worst = max(worst, abs(cj / cf - 1.0))
    return CheckResult("coeff analytic~FD", worst < 1e-3, worst,
                       "c_s^2 from Jacobian vs finite-difference")


def _check_backend_parity(par, flags, grid):
    """eos_fast (analytic Jacobian) vs eos_ref (numeric): same root (report
    §3.7 check 4). Same math, different derivative path — agree to ~round-off."""
    worst = 0.0
    for n_B in grid:
        a = solve_octet(par, n_B, flags, include_photons=False, analytic_jac=True)
        r = solve_octet(par, n_B, flags, include_photons=False, analytic_jac=False)
        for f in ("eps", "P", "mu_n", "sigma", "omega0"):
            va, vr = getattr(a, f), getattr(r, f)
            if abs(vr) > 1e-9:
                worst = max(worst, abs(va / vr - 1.0))
    return CheckResult("backend parity", worst < 1e-6, worst,
                       "eos_fast (analytic J) vs eos_ref")


def _check_compose(par):
    from eos.dd2.verify.compose import DD2_COMPOSE, compare_slice
    if not os.path.isfile(os.path.join(DD2_COMPOSE, "eos.thermo")):
        return CheckResult("CompOSE HS(DD2)", True, 0.0, "skipped (no table)")
    r = compare_slice(par, DD2_COMPOSE, T=1.0, Y_q=0.5, nB_min=0.14, nB_max=0.6)
    m = max(r["max_err_P"], r["max_err_eps"])
    return CheckResult("CompOSE HS(DD2)", m < 1e-3, m, "nucleonic T=1 Yq=0.5")


def run_full_check(par=None, flags=None, grid=None, include_tov=True):
    """
    Run the DD2 verification suite. Returns a FullCheckReport (structured
    pass/fail + max error per check).
    """
    par = par or Parametrization.from_dd2_defaults()
    flags = flags or SpeciesFlags(hyperons=False, phi_field=False)
    grid = np.array(grid) if grid is not None else np.array([0.1, 0.16, 0.3, 0.5])

    report = FullCheckReport()
    report.results.append(_check_golden(par))
    report.results.append(_check_identities(par, flags, grid))
    report.results.append(_check_responses(par, flags, grid))
    report.results.append(_check_coeff_cross(par, flags, grid))
    report.results.append(_check_backend_parity(par, flags, grid))
    report.results.append(_check_compose(par))
    if include_tov:
        report.results.append(_check_tov(par, flags))
    return report


if __name__ == "__main__":
    print(run_full_check())
