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
from dataclasses import dataclass, field, replace
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
    bare = replace(flags, photons=False)
    for n_B in grid:
        a = solve(par, n_B, bare, analytic_jac=True)
        r = solve(par, n_B, bare, analytic_jac=False)
        for va, vr in ((a.eps, r.eps), (a.P, r.P),
                       (a.matter.mu_B, r.matter.mu_B),
                       (a.matter.fields["sigma"], r.matter.fields["sigma"]),
                       (a.matter.fields["omega0"], r.matter.fields["omega0"])):
            if abs(vr) > 1e-9:
                worst = max(worst, abs(va / vr - 1.0))
    return CheckResult("backend parity", worst < 1e-6, worst,
                       "eos_fast (analytic J) vs eos_ref")


def _check_restarts_extend_the_basin(par):
    """The perturbed restarts reach targets the published seed does not.

    `nmp.N_RESTARTS = 32` exists because a single solve from the DD2 couplings
    maps one basin of attraction and reads it as the feasible set. That is a
    claim about the residual surface, not about the loop -- the loop keeping
    the best of N tries is monotone by construction and asserts nothing -- so
    what is checked is that the extra tries CHANGE THE ANSWER on a grid where
    the single seed fails outright.

    A count over a grid, deliberately, rather than a verdict at one cell: a
    single cell's verdict is decided in its target's last bits (ticket 67
    measured eight targets at three perturbations each, none holding across
    its own three), while the count is stable. Measured on python.org 3.14.2 /
    numpy 2.3.5 / scipy 1.17.0: 22/30 at zero restarts, 27/30 at 32.

    The DEFAULT closure, over (K_sat, m*/m). It used to be the Q_sat-imposing
    one over (K_sat, Q_sat), because a Q_sat row carrying a third finite
    difference made that the harder residual surface -- 0/9 at zero restarts
    against 4/9 at 32. Analytic derivatives removed the difficulty rather than
    the check: that closure now reaches 30/30 at zero restarts over a grid
    three times wider (K_sat 150-350, Q_sat -400 to 800), so there is nothing
    left there for restarts to find. The basin structure the restarts exist
    for is real and survives, and it is the default closure that shows it.

    Here rather than in `test/dd2/` because it is a property of the inverse
    map's basin structure measured over a grid, the same class as the forward
    and inverse maps agreeing, and because it costs ~10 s.
    """
    from eos.dd2.nmp import compute_nmp, invert_nmp

    ref = compute_nmp(par)
    six = {k: ref[k] for k in ("n_sat", "E_sat", "m_eff_ratio", "K_sat",
                               "E_sym", "L_sym")}
    cells = [(K, M) for K in (160.0, 200.0, 240.0, 280.0, 320.0)
             for M in (0.40, 0.50, 0.60, 0.70, 0.80, 0.90)]
    reached = {}
    for n_restarts in (0, 32):
        n_ok = 0
        for K_sat, m_ratio in cells:
            try:
                _, status = invert_nmp(dict(six, K_sat=K_sat,
                                            m_eff_ratio=m_ratio),
                                       n_restarts=n_restarts)
            except ValueError:      # m*/m outside the physical window
                continue
            n_ok += bool(status.ok)
        reached[n_restarts] = n_ok
    gained = reached[32] - reached[0]
    return CheckResult("restarts extend the basin", gained > 0, float(gained),
                       f"{reached[0]}/{len(cells)} cells at 0 restarts, "
                       f"{reached[32]}/{len(cells)} at 32")


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

    h = 1.5e-3 rather than the plateau's left edge, and that IS a measured
    choice. The estimator is only honest where truncation dominates the pair;
    at small h the scatter it divides by is roundoff, and for Q_sat -- a third
    difference, so roundoff grows as h^-3 -- the ratio then jitters with the
    interpreter. Measured on the published set, worst of the four:

        h        4e-4    6e-4    8e-4    1e-3    1.5e-3   2e-3
        py3.9    0.509   0.040   0.033   0.016   0.003    0.001
        py3.14   0.403   0.633   0.038   0.022   0.003    0.000

    (anaconda 3.9.7 / numpy 1.26.4 / scipy 1.13.1 against python.org 3.14.2 /
    numpy 2.3.5 / scipy 1.17.0; the worst column is Q_sat in every case, the
    other three sit below 1e-3 everywhere.) At 1.5e-3 the two stacks agree and
    the check passes with 300x of margin; at 6e-4 it would pass with 1.6x on
    one of them, which is a check waiting to fail for a reason that is not
    physics.
    """
    from scipy.optimize import brentq
    from eos.dd2.nmp import snm_derivatives, energy_per_baryon, esym
    from eos.dd2.solver import solve_snm_t0

    n_sat = brentq(lambda n: solve_snm_t0(par, n).P, 0.12, 0.18, xtol=1e-12)
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

    h = 1.5e-3                    # truncation-dominated: see the docstring
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


def _check_compose(par):
    from eos.dd2.verify.compose import DD2_COMPOSE, compare_slice
    if not os.path.isfile(os.path.join(DD2_COMPOSE, "eos.thermo")):
        return CheckResult("CompOSE HS(DD2)", True, 0.0, "skipped (no table)")
    r = compare_slice(par, DD2_COMPOSE, T=1.0, Y_q=0.5, nB_min=0.14, nB_max=0.6)
    m = max(r["max_err_P"], r["max_err_eps"])
    return CheckResult("CompOSE HS(DD2)", m < 1e-3, m, "nucleonic T=1 Yq=0.5")


def _check_hyperon_depths(par):
    """The hyperon closure's two halves agree: SU(6) x y, then the depths.

    U_Y = -Gamma_sigmaY sigma + Gamma_omegaY omega0 + Sigma^R holds the scalar
    and vector couplings TOGETHER, so a set whose vector couplings are SU(6)
    times a factor is only consistent if the scalar couplings were inverted
    AFTER the rescaling. Invert on a broken base and require the depths back;
    then check that y = 1 reproduces the published DD2Y vector column exactly
    (Fortin, Oertel & Providencia, PASA 35 (2018) e044, Table 1).
    """
    from dataclasses import replace
    from eos.dd2.couplings import SU6_HYPERON
    from eos.dd2.nmp import from_hyperon_potentials

    su6_exact = max(
        max(abs(row[2] - SU6_HYPERON[name]["x_omega"]),
            abs(row[3] - SU6_HYPERON[name]["x_rho"]),
            abs(row[4] - SU6_HYPERON[name]["phi_over_omegaN"]))
        for name, row in Parameters.named("DD2Y").hyperon_coupling_map.items())

    depths = dict(U_Lambda=-30.0, U_Sigma=30.0, U_Xi=-18.0)
    broken = from_hyperon_potentials(
        base=replace(par, y_omega_Lambda=1.5, y_phi_Lambda=1.5,
                     y_omega_Xi=1.875, y_phi_Xi=1.875), **depths)
    sat = solve_snm(broken, broken.n_sat)
    Gs, Gw, _, _, _, _ = broken.couplings_at(broken.n_sat)
    err = 0.0
    for name, key in (("Lambda", "U_Lambda"), ("Sigma-", "U_Sigma"),
                      ("Xi-", "U_Xi")):
        _, x_sigma, x_omega, _, _ = broken.hyperon_coupling_map[name]
        U = (-x_sigma * Gs * sat.matter.fields["sigma"]
             + x_omega * Gw * sat.matter.fields["omega0"] + sat.matter.Sigma_R)
        err = max(err, abs(U - depths[key]))

    worst = max(err, su6_exact)
    return CheckResult(
        "hyperon depths vs SU(6) breaking", worst < 1e-8, worst,
        f"U_Y held to {err:.1e} MeV on a broken base; DD2Y is SU(6) to "
        f"{su6_exact:.1e}")


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
    flags = flags or SpeciesFlags(hyperons=False)
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
    report.results.append(_check_analytic_derivatives(par))
    report.results.append(_check_restarts_extend_the_basin(par))
    report.results.append(_check_compose(par))
    report.results.append(_check_hyperon_depths(par))
    return report


if __name__ == "__main__":
    print(run_full_check())
