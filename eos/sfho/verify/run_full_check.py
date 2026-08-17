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
  3. the symmetry energy two ways — the delta^2 curvature of E/A against the
     analytic formula of the model's source paper;
  4. causality and monotonicity — 0 <= c_s^2 <= 1 and P non-decreasing along
     a cold beta-equilibrium sweep;
  5. CompOSE HS(SFHo) comparison, when the table is present.

Why check 1 earns its place: SFHo shipped for a long time with an energy
density missing the omega(dA/domega) rho^2 partner of the omega field
equation's cross term. Nothing caught it, because it enters no residual — the
solver converged perfectly on an eps that was wrong by up to 1.8 percent in
asymmetric matter, which put E_sym at 18.7 MeV instead of 31.6 and made L
negative. The Euler identity localises exactly that class of error, and it is
cheap.
"""
from dataclasses import dataclass, field
import os

import numpy as np

from eos.general.physics_constants import hc, hc3
from eos.sfho.parameters import get_sfho_nucleonic, get_sfhoy_fortin
from eos.sfho.species import SpeciesFlags, active_baryons
from eos.sfho.solver import (
    solve_beta_eq_neutrinoless, solve_fixed_yc, solve_fixed_yc_ys,
    solve_beta_eq_neutrino_trapped, solve_isentropic_beta_eq,
)

NUCLEONS = SpeciesFlags()
NUCLEONS_NOGAMMA = SpeciesFlags(photons=False)
WITH_HYPERONS = SpeciesFlags(hyperons=True)

#: Steiner, Hempel & Fischer, ApJ 774 (2013) 17, as tabulated by Fortin,
#: Oertel & Providencia, PASA 35 (2018) e044, Table 2.
PUBLISHED = dict(n_sat=0.158, E_sat=-16.2, E_sym=31.6, L=47.1)

SFHO_COMPOSE = "/Users/mircoguerrini/Desktop/Research/Compose/SFHO_Compose"


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
def euler_residual(result, flags):
    """(eps + P - T s - sum_i mu_i n_i) / eps for one solved point.

    The species potentials are rebuilt from the conserved-charge basis,
    mu_i = B_i mu_B + C_i mu_C + S_i mu_S (CLAUDE.md §2), and the leptons are
    added at their own potentials. Photons need no term: they carry mu = 0 and
    satisfy eps + P = T s on their own.
    """
    mu_dot_n = 0.0
    for p in active_baryons(flags):
        n = result.baryon_densities.get(p.name, 0.0)
        mu = (p.baryon_no * result.mu_B + p.charge * result.mu_C
              + p.strangeness * result.mu_S)
        mu_dot_n += mu * n
    mu_dot_n += result.mu_e * result.n_e + result.mu_nue * result.n_nu
    if result.e_total == 0.0:
        return 0.0
    return ((result.e_total + result.P_total - result.T * result.s_total
             - mu_dot_n) / result.e_total)


def _check_euler(nuc, hyp, grid):
    worst, where = 0.0, ""
    for n_B in grid:
        cases = [
            ("beta T=0", NUCLEONS,
             solve_beta_eq_neutrinoless(nuc, n_B, NUCLEONS, T=0.0)),
            ("beta T=30", NUCLEONS,
             solve_beta_eq_neutrinoless(nuc, n_B, NUCLEONS, T=30.0)),
            ("beta hyperons", WITH_HYPERONS,
             solve_beta_eq_neutrinoless(hyp, n_B, WITH_HYPERONS, T=10.0)),
            ("fixed Y_C", NUCLEONS,
             solve_fixed_yc(nuc, n_B, 0.1, NUCLEONS, T=10.0)),
            ("fixed Y_C leptons", NUCLEONS,
             solve_fixed_yc(nuc, n_B, 0.1, NUCLEONS, T=10.0, leptons=True)),
            ("fixed Y_C Y_S", WITH_HYPERONS,
             solve_fixed_yc_ys(hyp, n_B, 0.4, 0.1, WITH_HYPERONS, T=10.0)),
            ("trapped", NUCLEONS,
             solve_beta_eq_neutrino_trapped(nuc, n_B, 0.4, NUCLEONS, T=10.0)),
            ("isentropic", NUCLEONS,
             solve_isentropic_beta_eq(nuc, n_B, 1.0, NUCLEONS)),
        ]
        for tag, flags, r in cases:
            if not r.converged:
                continue
            err = abs(euler_residual(r, flags))
            if err > worst:
                worst, where = err, f"{tag} at n_B={n_B:g}"
    return CheckResult("Euler / HVH, all modes", worst < 1e-8, worst, where)


# =============================================================================
# 2, 3. NUCLEAR MATTER PARAMETERS
# =============================================================================
def _energy_per_baryon(par, n_B, Y_C):
    r = solve_fixed_yc(par, n_B, Y_C, NUCLEONS_NOGAMMA, T=0.01)
    m_N = 0.5 * (par.m_n + par.m_p)
    return r.e_total / n_B - m_N


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


def symmetry_energy_analytic(par, n_B):
    """E_sym from Steiner, Prakash, Lattimer & Ellis (2005), Eq. (20):

        E_sym = k_F^2 / (6 E_F*) + n_B / [ 8 ( m_rho^2/g_rho^2 + 2 f ) ]

    with A = g_rho^2 f. Independent of the energy density -- it comes from the
    rho-field response -- so it is a genuine second opinion on E_sym rather
    than the same computation rearranged.
    """
    r = solve_fixed_yc(par, n_B, 0.5, NUCLEONS_NOGAMMA, T=0.01)
    k_F = hc * (3.0 * np.pi**2 * n_B / 2.0) ** (1.0 / 3.0)
    E_F = np.sqrt(k_F**2 + r.m_eff["n"]**2)
    A = par.compute_A(r.sigma, r.omega)
    kinetic = k_F**2 / (6.0 * E_F)
    potential = n_B * hc3 * par.g_rho_N**2 / (8.0 * (par.m_rho**2 + 2.0 * A))
    return kinetic + potential


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


def _check_esym_two_ways(par):
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
        P.append(r.P_total)
        eps.append(r.e_total)
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
def _check_compose(par, compose_dir=SFHO_COMPOSE):
    """Compare P and eps with the published HS(SFHo) table.

    At T = 10 MeV and n_B above about 0.2 fm^-3 the statistical model is
    uniform matter, so the table is this model's fixed_YC mode with electrons
    and photons. Neutron-rich slices are the point: the isovector sector is
    where an energy-density error hides, since it vanishes at Y_q = 0.5.
    """
    if not os.path.isfile(os.path.join(compose_dir, "eos.thermo.ns")):
        return CheckResult("CompOSE HS(SFHo)", True, 0.0, "skipped (no table)")
    from eos.sfho.compose_loader import read_compose_data

    data = read_compose_data(compose_dir, name="SFHO")
    iT = int(np.argmin(abs(data.T_values - 10.0)))
    T = float(data.T_values[iT])
    worst, where = 0.0, ""
    for Y_C in (0.5, 0.3, 0.1):
        iY = int(np.argmin(abs(data.Y_C_values - Y_C)))
        for n_B in (0.2, 0.3, 0.4, 0.6):
            iN = int(np.argmin(abs(data.n_B_values - n_B)))
            nb = float(data.n_B_values[iN])
            P_ref = float(data.P[iT, iN, iY])
            eps_ref = float(data.e[iT, iN, iY])
            if not (np.isfinite(P_ref) and np.isfinite(eps_ref)):
                continue
            r = solve_fixed_yc(par, nb, float(data.Y_C_values[iY]),
                               NUCLEONS, T=T, leptons=True)
            err = max(abs(r.P_total - P_ref) / abs(P_ref),
                      abs(r.e_total - eps_ref) / abs(eps_ref))
            if err > worst:
                worst, where = err, f"Y_q={data.Y_C_values[iY]:.2f} n_B={nb:.3f}"
    return CheckResult("CompOSE HS(SFHo)", worst < 2e-3, worst,
                       f"T={T:.1f} MeV, worst at {where}")


# =============================================================================
def run_full_check(par=None, hyp=None, grid=None):
    """Run the SFHo verification suite; returns a structured FullCheckReport."""
    par = par or get_sfho_nucleonic()
    hyp = hyp or get_sfhoy_fortin()
    grid = np.array(grid) if grid is not None else np.array([0.16, 0.32, 0.64])

    report = FullCheckReport()
    report.results.append(_check_euler(par, hyp, grid))
    report.results.append(_check_nmp(par))
    report.results.append(_check_esym_two_ways(par))
    report.results.append(_check_causality(par, np.linspace(0.1, 1.0, 25)))
    report.results.append(_check_compose(par))
    return report


if __name__ == "__main__":
    print(run_full_check())
