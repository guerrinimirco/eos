"""
verify/run_full_check.py
====================
Single entry point orchestrating the DD2 verification suite.
Each check returns a structured pass/fail + max error, never a bare print.

Checks (those with data/deps available in eos_ref):
  1. golden points — the day-one smoke test;
  2. thermodynamic identities — HVH, beta-eq residual across a grid;
  3. TOV cross-check — M_max >= 2 M_sun for the cold beta-eq EoS;
  4. free energy — f = eps - T s = -P + sum_i mu_i n_i over the totals;
  5. rearrangement — Sigma^R enters mu and P and NEVER eps (CLAUDE.md §8),
     the invariant that catches a wrong density-dependent RMF;
  6. coefficient sanity — causality 0 <= c_s^2 <= 1, thermal index > 1;
  7. delivered table — the `EOSTable_for_TOV` this model builds has P
     non-decreasing in n_B and 0 <= c_s^2 <= 1 (§8's delivery gate, owed by
     whoever builds a table);
  8. CompOSE comparison — nucleonic HS(DD2) slice (< 1e-3) when the table is
     present.
Backend parity (eos_ref vs eos_fast) is checked alongside.
"""
from dataclasses import dataclass, field
import os

import numpy as np

from eos.dd2 import (
    Parameters, SpeciesFlags, solve_snm, solve_beta_eq_neutrinoless, solve,
)
from eos.dd2.responses import sound_speed_eq, thermal_index
from eos.dd2.table import build_core_table
from eos.dd2.thermodynamics import (
    baryon_kinetics, build_matter_ctx, field_eps_P, thermal_meson_thermo,
)
from eos.general.physics_constants import hc3


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
        "m*/m": abs(p.matter.m_eff_i["n"] / par.m_nucleon
                    / _GOLDEN_SNM_016["mstar_m"] - 1),
        "eps": abs(p.eps / _GOLDEN_SNM_016["eps"] - 1),
        "P": abs(p.P / _GOLDEN_SNM_016["P"] - 1),
        "mu_B": abs(p.matter.mu_B / _GOLDEN_SNM_016["mu_B"] - 1),
    }
    m = max(errs.values())
    return CheckResult("golden points", m < 1e-4, m,
                       f"SNM(0.16) vs(b)")


def _check_identities(par, flags, grid):
    worst = 0.0
    for n_B in grid:
        p = solve_beta_eq_neutrinoless(par, n_B, flags)
        worst = max(worst, abs(p.euler_residual()),
                     abs(-p.matter.mu_C - p.leptons.mu_e) / p.matter.mu_B)
    return CheckResult("thermo identities", worst < 1e-8, worst,
                       "HVH + beta residual over grid")


def _check_free_energy(par, flags, grid):
    """f = eps - T s, and f = -P + sum_i mu_i n_i, over the TOTALS.

    CLAUDE.md section 8 lists this beside the Euler relation, and at the level
    of algebra the two are the same statement rearranged. What makes it worth
    running anyway is WHICH SIDE each is read from: the Euler check above goes
    through `EoSPoint.euler_residual`, while this one reads the public
    `free_energy_density` property that every consumer of a point actually
    calls -- a notebook drawing f(n_B), `nucleation` taking a free-energy
    difference across a phase boundary. A property that returned eps + T s, or
    the matter's f where the caller asked for the totals, would leave the
    Euler check untouched and every free-energy figure wrong.
    """
    worst = 0.0
    for n_B in grid:
        p = solve_beta_eq_neutrinoless(par, n_B, flags)
        mu_dot_n = p.matter.mu_dot_n
        if p.leptons is not None:
            mu_dot_n += p.leptons.mu_dot_n
        f_property = p.free_energy_density
        f_identity = -p.P + mu_dot_n
        worst = max(worst, abs(f_property - f_identity) / abs(p.eps))
    return CheckResult("free energy", worst < 1e-8, worst,
                       "f = eps - Ts = -P + sum mu_i n_i")


def _check_rearrangement(par, flags, grid):
    """Sigma^R enters mu and P and NEVER eps (CLAUDE.md section 8).

    DD2's couplings depend on density, so it carries a rearrangement
    self-energy, and this is the invariant that catches a wrong
    density-dependent RMF: get the term's placement wrong and the model is
    still an equation of state, just not a thermodynamically consistent one.

    Two falsifiable identities on each solved state, taken apart rather than
    asserted. Writing the baryon sums at the solved fields as

        S_P = sum_i P_i ,  S_eps = sum_i eps_i

    with (eps_field, P_field) the meson mean-field terms and (P_gas, eps_gas)
    the thermal meson gas, `thermodynamics.assemble` must satisfy, exactly,

        P   - (S_P + P_field + P_gas)     = Sigma^R n_B     the term IS in P
        eps - (S_eps + eps_field + eps_gas) = 0             and is NOT in eps

    The kinetic sums are recomputed here from the state's own fields and
    potentials -- mu~_B = mu_B - Sigma^R is the kinetic potential the
    per-species integrals are evaluated at -- so the check does not read back
    the assembly it is testing.

    The size of the term is reported too, because an identity both sides
    satisfy trivially proves nothing: at n_B = 0.5 fm^-3 it is 4 percent of
    eps, so neither line is passing by being small.
    """
    worst = 0.0
    size = 0.0
    for n_B in grid:
        st = solve_beta_eq_neutrinoless(par, n_B, flags).matter
        ctx = build_matter_ctx(par, st.n_B, flags, st.T)
        f = st.fields
        kin = baryon_kinetics(ctx, f["sigma"], f["omega0"], f["rho0"],
                              f["phi0"], st.mu_B - st.Sigma_R, st.mu_C, st.mu_S)
        S_eps = sum(entry[6] for entry in kin) / hc3
        S_P = sum(entry[7] for entry in kin) / hc3
        eps_field, P_field = field_eps_P(par, f["sigma"], f["omega0"],
                                         f["rho0"], f["phi0"])
        gas = thermal_meson_thermo(
            par, st.n_B, st.mu_C, st.mu_S, f["omega0"], f["rho0"], st.T,
            include_pseudoscalars=ctx.include_pseudoscalars,
            include_thermal_vectors=ctx.include_thermal_vectors)
        in_P = abs((st.P - (S_P + P_field / hc3 + gas["P"]))
                   - st.Sigma_R * st.n_B)
        not_in_eps = abs(st.eps - (S_eps + eps_field / hc3 + gas["e"]))
        worst = max(worst, in_P / abs(st.eps), not_in_eps / abs(st.eps))
        size = max(size, abs(st.Sigma_R * st.n_B) / abs(st.eps))
    return CheckResult(
        "rearrangement", worst < 1e-12 and size > 1e-6, worst,
        f"Sigma^R is in P and not in eps; at its largest it carries "
        f"{100 * size:.1f}% of eps")


def _check_delivered_table(par, flags):
    """The table this model hands a structure solver is deliverable.

    CLAUDE.md section 8: P non-decreasing in n_B and 0 <= c_s^2 <= 1, checked
    before integration, and owed by whoever BUILDS the table -- which
    `eos.dd2.table.build_core_table` does.

    IT IS STATED AGAINST n_B, NOT AGAINST THE ROW ORDER, and that is the whole
    content of the check: `build_core_table` returns its rows sorted by P
    (`np.argsort(P)`, because TOV interpolates on a monotone P grid), so
    `np.diff(P) >= 0` holds by construction and would be a check that cannot
    fail. What the sort does NOT repair is the density column: a branch whose
    P falls with n_B comes back with n_B out of order, and it is that
    permutation this check looks for.
    """
    table = build_core_table(par, flags)
    if len(table.P) < 3:
        return CheckResult("delivered table", False, float("inf"),
                           f"only {len(table.P)} rows built")
    dn = np.diff(table.nB)
    cs2 = np.diff(table.P) / np.diff(table.epsilon)
    worst_n = abs(min(dn.min(), 0.0))
    worst_cs = max(-cs2.min(), cs2.max() - 1.0, 0.0)
    passed = bool(dn.min() >= 0.0 and cs2.min() >= 0.0 and cs2.max() <= 1.0)
    return CheckResult(
        "delivered table", passed, max(worst_n, worst_cs),
        f"{len(table.P)} rows, n_B = {table.nB[0]:.3f}-{table.nB[-1]:.3f} "
        f"fm^-3, c_s^2 in [{cs2.min():.3f}, {cs2.max():.3f}]")


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
        a = solve(par, n_B, flags, include_photons=False, analytic_jac=True)
        r = solve(par, n_B, flags, include_photons=False, analytic_jac=False)
        for va, vr in ((a.eps, r.eps), (a.P, r.P),
                       (a.matter.mu_B, r.matter.mu_B),
                       (a.matter.fields["sigma"], r.matter.fields["sigma"]),
                       (a.matter.fields["omega0"], r.matter.fields["omega0"])):
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


def run_full_check(par=None, flags=None, grid=None):
    """
    Run the DD2 verification suite. Returns a FullCheckReport (structured
    pass/fail + max error per check).

    The M_max >= 2 M_sun check is NOT here: running a stellar sequence means
    importing `eos.astro`, which CLAUDE.md section 1 does not allow a model to
    do. It is `test/dd2/dd2_tov_sequence.py` plus `test/dd2/test_dd2_m4_tov.py`,
    and the model's own half of that contract -- `build_core_table`, returning
    an `EOSTable_for_TOV` -- is `eos.dd2.table`.
    """
    par = par or Parameters.default()
    flags = flags or SpeciesFlags(hyperons=False, phi_field=False)
    grid = np.array(grid) if grid is not None else np.array([0.1, 0.16, 0.3, 0.5])

    report = FullCheckReport()
    report.results.append(_check_golden(par))
    report.results.append(_check_identities(par, flags, grid))
    report.results.append(_check_free_energy(par, flags, grid))
    report.results.append(_check_rearrangement(par, flags, grid))
    report.results.append(_check_responses(par, flags, grid))
    report.results.append(_check_coeff_cross(par, flags, grid))
    report.results.append(_check_backend_parity(par, flags, grid))
    report.results.append(_check_delivered_table(par, flags))
    report.results.append(_check_compose(par))
    return report


if __name__ == "__main__":
    print(run_full_check())
