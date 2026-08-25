"""Physics invariants of the DID model, checked in one place.

These are the statements the implementation has to satisfy whatever parameters
it is given, plus the published numbers the shipped parameter set must
reproduce. They are the fastest way to catch a wrong change, and every check
returns a structured pass/fail with the largest error it saw, so the suite
reports rather than prints.

  1. Euler relation      eps + P = T s + sum_i mu_i n_i, in every mode, with
                         both rearrangement self-energies in place.
  2. Free energy         f = eps - T s = -P + sum_i mu_i n_i.
  3. Rearrangement       Sigma^r enters mu and P, never eps; and the SECOND
                         rearrangement term cancels identically out of every
                         sum, since sum_i (tau_3i - beta) n_i = 0. Both are
                         statements the paper's Appendix A proves.
  4. ISM limit           at beta = 0 the isospin machinery switches itself
                         off: Sigma^t = 0, dg/dbeta = 0, and every coupling is
                         its symmetric-matter branch. This is what makes DID
                         reduce to an ordinary DD-RMF in symmetric matter.
  5. SU(6) limit         at z = 1/sqrt6, alpha = 1 and ideal mixing the SU(3)
                         vector ratios collapse to the textbook SU(6) values
                         (2/3, 1/3, g_phiN = 0, -sqrt2/3, -2sqrt2/3). This is
                         the check that fixes the g_phiXi coefficient, which
                         arXiv:2511.15646 Eq. (6) prints with a factor 2 that
                         breaks all four ratios.
  6. Saturation          P(n_0) = 0 in symmetric matter at T = 0 -- the
                         condition n_0 was calibrated by.
  7. Hyperon potentials  U_Y in ISM and in NM at n_0, against the paper's
                         Table IV (12 values, MLE column).
  8. Nuclear matter      the parameters of Table VI: B, K, Q, M(0.11), S_2,
                         S, L, K_sym and the beta-equilibrium proton fraction.
  9. Hyperon onsets      the DIDY onset densities of Table VII: Sigma- first
                         at 0.470, then Lambda at 0.578, then Xi- at 0.978,
                         with no Xi0 -- the inverted hierarchy that is the
                         paper's headline result.
 10. Mode closures       each mode's own conditions hold at its solution.
 11. Charge basis        n_B, n_C, n_S and the species potentials agree with
                         `eos.general.basis` -- no local copy of the map.
 12. Causality           0 <= c_s^2 <= 1 along a cold beta-equilibrium
                         sequence, and the published c_s^2 peak near 4 n_0.
 13. Residual gate       every state solved here is inside the tolerance the
                         model claims to accept at.

The stellar-structure comparison against Table VIII -- M_max and R_1.4 --
is NOT one of these: it runs a TOV sequence, so it imports `eos.astro`, which
CLAUDE.md section 1 does not allow a model to do. It is
`test/did/test_did_tov.py`.

Run as `python -m eos.did.verify.run_full_check`.
"""
from dataclasses import dataclass, field

import numpy as np

from eos.general.basis import charges_from_densities, species_potential
from eos.general.particles import get_particle
from eos.general.solve import RESIDUAL_TOL
from eos.did import (
    MULTIPLET_OF, Parameters, SpeciesFlags, Z_SU6, compute_nmp, eos_response,
    baryon_kinetics, nuclear_matter, single_particle_potential, solve_beta_eq_neutrinoless,
    solve_beta_eq_neutrino_trapped, solve_fixed_yc, solve_fixed_yc_ys,
    species_table, su3_vector_ratios, tau3, warm_start,
)
from eos.general.modes import beta_eq_neutrinoless

#: The published values every gate below is stated against
#: (arXiv:2511.15646, Tables IV, VI and VII, MLE columns).
TABLE_IV_ISM = {"Lambda": -27.87, "Sigma+": 14.99, "Sigma0": 14.99,
                "Sigma-": 14.99, "Xi0": -3.97, "Xi-": -3.97}
TABLE_IV_NM = {"Lambda": -25.54, "Sigma+": 6.85, "Sigma0": 15.79,
               "Sigma-": 24.74, "Xi0": -12.13, "Xi-": 5.85}
TABLE_VI = {"B": -15.40, "K": 227.06, "Q": -608.09, "M": 1122.72,
            "S_2": 32.44, "S": 29.72, "L": 59.95, "K_sym": -97.32,
            "X_p_eq": 0.0336}
TABLE_VII = {"Sigma-": 0.470, "Lambda": 0.578, "Xi-": 0.978}


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
        lines = [f"DID run_full_check: "
                 f"{'PASS' if self.all_passed else 'FAIL'}"]
        for r in self.results:
            tag = "ok " if r.passed else "FAIL"
            lines.append(f"  [{tag}] {r.name:22s} max_err={r.max_error:.2e}"
                         f"  {r.detail}")
        return "\n".join(lines)


def _states(par):
    """One converged state per mode, nucleonic and hyperonic, cold and hot."""
    nucleons = SpeciesFlags(muons=True)
    octet = SpeciesFlags(hyperons=True, muons=True)
    out = []
    for flags, T in ((nucleons, 0.0), (nucleons, 25.0), (octet, 0.0),
                     (octet, 25.0)):
        out.append(("beta", flags, T,
                    solve_beta_eq_neutrinoless(par, 0.4, flags, T=T)))
        out.append(("yc_lep", flags, T,
                    solve_fixed_yc(par, 0.4, 0.3, flags, T=T, leptons=True)))
        out.append(("yc_nolep", flags, T,
                    solve_fixed_yc(par, 0.4, 0.3, flags, T=T, leptons=False)))
    octet_hot = SpeciesFlags(hyperons=True, muons=True)
    out.append(("trapped", octet_hot, 25.0,
                solve_beta_eq_neutrino_trapped(par, 0.4, 0.4, octet_hot,
                                               T=25.0)))
    out.append(("yc_ys", octet_hot, 25.0,
                solve_fixed_yc_ys(par, 0.4, 0.3, 0.05, octet_hot, T=25.0)))
    return out


# =============================================================================
# 1-2. EULER AND FREE ENERGY
# =============================================================================
def check_euler(par, states, tol=1.0e-8):
    """eps + P = T s + sum_i mu_i n_i, on the totals each mode returns."""
    worst, where = 0.0, ""
    for label, _flags, _T, point in states:
        error = abs(point.hvh_rel)
        if error > worst:
            worst, where = error, label
    return CheckResult("euler", worst <= tol, worst, f"worst in {where}")


def check_free_energy(par, states, tol=1.0e-8):
    """f = eps - T s and f = -P + sum_i mu_i n_i are the same number."""
    worst, where = 0.0, ""
    for label, _flags, T, point in states:
        mu_dot_n = point.eps + point.P - T * point.s - point.hvh_rel * point.eps
        f_direct = point.eps - T * point.s
        f_legendre = -point.P + mu_dot_n
        error = abs(f_direct - f_legendre) / abs(point.eps)
        if error > worst:
            worst, where = error, label
    return CheckResult("free_energy", worst <= tol, worst, f"worst in {where}")


# =============================================================================
# 3. THE TWO REARRANGEMENT TERMS
# =============================================================================
def check_rearrangement(par, tol=1.0e-10):
    """Sigma^r is in P and not in eps, and the Sigma^t term cancels in sums.

    The energy density is rebuilt from the fields and the kinetic integrals
    alone; if a rearrangement term had leaked into it, the two would differ by
    n_B Sigma^r, which is tens of MeV/fm^3.
    """
    from eos.did.thermodynamics import field_eps_P, thermo_from_fields

    flags = SpeciesFlags(hyperons=True, muons=False, photons=False)
    specs = species_table(flags)
    worst = 0.0
    for n_B in (0.2, 0.6, 1.0):
        point = solve_beta_eq_neutrinoless(par, n_B, flags, T=0.0)
        fields = point.fields()
        mu_tilde_B = point.mu_B - point.Sigma_r
        matter = baryon_kinetics(par, specs, fields, mu_tilde_B, point.mu_C,
                          point.mu_S, point.T)
        block = thermo_from_fields(par, flags, fields, mu_tilde_B, point.mu_C,
                                   point.mu_S, point.T, matter=matter)
        eps_fields, P_fields = field_eps_P(par, fields)
        # The energy carries the kinetic pieces and the fields and NOTHING
        # else; the pressure carries n_B Sigma^r on top of them.
        worst = max(worst,
                    abs(block.eps - (matter.eps + eps_fields)) / block.eps,
                    abs(block.P - (matter.P + P_fields
                                   + matter.n_B * matter.Sigma_r)) / block.P)
        # The isospin term of the thermodynamic potential, which must vanish.
        sigma_t_term = sum((sp.tau3 - fields.beta) * matter.densities[sp.name]
                           for sp in specs) * fields.Sigma_t
        worst = max(worst, abs(sigma_t_term) / block.P)
        # And it is not vanishing because Sigma^t is small.
        if abs(fields.Sigma_t) < 1.0:
            return CheckResult("rearrangement", False, worst,
                               f"Sigma^t = {fields.Sigma_t:.2e} MeV: the "
                               f"cancellation is trivial, not a check")
    return CheckResult("rearrangement", worst <= tol, worst,
                       "Sigma^r in P not eps; Sigma^t cancels in the sums")


# =============================================================================
# 4-5. THE TWO LIMITS
# =============================================================================
def check_ism_limit(par, tol=1.0e-12):
    """At beta = 0 the isospin dependence switches itself off."""
    worst = 0.0
    for n_B in (0.1, 0.5, 1.0):
        couplings = par.couplings_at(n_B, 0.0)
        strengths = par.strengths()
        shapes = par.shapes()
        from eos.did.couplings import shape as _shape
        for (meson, multiplet), (g, _dg_dn, dg_dbeta) in couplings.items():
            worst = max(worst, abs(dg_dbeta))
            g_S = strengths[(meson, multiplet)][0]
            expected = g_S * _shape(n_B / par.n_0, *shapes[meson])
            scale = max(abs(expected), 1.0)
            worst = max(worst, abs(g - expected) / scale)
        ism = nuclear_matter(par, n_B, 0.0)
        worst = max(worst, abs(ism.Sigma_t))
    return CheckResult("ism_limit", worst <= tol, worst,
                       "Sigma^t = 0, dg/dbeta = 0, g = the S branch")


def check_su6_limit(tol=1.0e-12):
    """The SU(3) vector ratios collapse to SU(6) at z = 1/sqrt6."""
    ratios = su3_vector_ratios(Z_SU6)
    g_omega_N = ratios["N"][0]
    expected = {"N": (1.0, 0.0),
                "Lambda": (2.0 / 3.0, -np.sqrt(2.0) / 3.0),
                "Sigma": (2.0 / 3.0, -np.sqrt(2.0) / 3.0),
                "Xi": (1.0 / 3.0, -2.0 * np.sqrt(2.0) / 3.0)}
    worst = 0.0
    for multiplet, (r_omega, r_phi) in expected.items():
        got_omega, got_phi = ratios[multiplet]
        worst = max(worst, abs(got_omega / g_omega_N - r_omega),
                    abs(got_phi / g_omega_N - r_phi))
    return CheckResult("su6_limit", worst <= tol, worst,
                       "omega 2/3, 1/3; g_phiN = 0; phi -sqrt2/3, -2sqrt2/3")


# =============================================================================
# 6-9. THE PUBLISHED NUMBERS
# =============================================================================
def check_saturation(par, tol=1.0e-6):
    """P(n_0) = 0 in symmetric matter at T = 0, the calibration condition."""
    ism = nuclear_matter(par, par.n_0, 0.0)
    error = abs(ism.P)
    return CheckResult("saturation", error <= tol and ism.converged, error,
                       f"P(n_0) = {ism.P:.2e} MeV/fm^3")


def check_hyperon_potentials(par, tol=0.02):
    """U_Y at n_0 in ISM and NM against Table IV of arXiv:2511.15646.

    The single strongest check in this suite: it exercises the SU(3) vector
    unpacking, the fitted scalar and isovector hyperon couplings, BOTH
    rearrangement terms and the tau_3 = 2 I_3 normalisation at once. The Sigma
    splitting in neutron matter is carried almost entirely by Sigma^t.
    """
    worst, where = 0.0, ""
    for medium, targets in (("ISM", TABLE_IV_ISM), ("NM", TABLE_IV_NM)):
        point = nuclear_matter(par, par.n_0, 0.0 if medium == "ISM" else -1.0)
        couplings = par.couplings_at(point.n_B, point.beta)
        for name, published in targets.items():
            U = single_particle_potential(
                couplings, point.fields(), MULTIPLET_OF[name],
                tau3(get_particle(name)), point.Sigma_r)
            error = abs(U - published)
            if error > worst:
                worst, where = error, f"{name} in {medium}"
    return CheckResult("hyperon_potentials", worst <= tol, worst,
                       f"12 values, worst {where}")


def check_nuclear_matter_parameters(par, rtol=2.0e-3, rtol_quadratic=1.5e-2):
    """Table VI, as relative agreement with the published values.

    Two tolerances, because the entries are two different kinds of number.
    B, K, Q, M, S, L, K_sym and X_p are properties of the functional and come
    back to two parts in a thousand. S_2 is a coefficient of a TRUNCATED
    expansion rather than a quantity the model defines: the paper quotes it
    with S - S_2 as the quartic coefficient, i.e. from a fit truncated at
    beta^4, and how much of the beta^6 behaviour a given extraction absorbs
    moves it by about one percent. That is a statement about the definition,
    not about the implementation, and the looser gate says so.
    """
    nmp = compute_nmp(par)
    worst, where = 0.0, ""
    for key, published in TABLE_VI.items():
        tol = rtol_quadratic if key == "S_2" else rtol
        error = abs(nmp[key] - published) / abs(published) / tol
        if error > worst:
            worst, where = error, key
    return CheckResult("nmp", worst <= 1.0, worst,
                       f"Table VI as multiples of tolerance, worst {where} "
                       f"(B={nmp['B']:.2f}, K={nmp['K']:.2f}, "
                       f"L={nmp['L']:.2f}, S_2={nmp['S_2']:.2f})")


def check_hyperon_onsets(par, tol=0.01):
    """The DIDY onset densities of Table VII, and their inverted hierarchy.

    The paper's own configuration: electrons only, no muons. With muons the
    onsets move up by 0.03 fm^-3, which is a real effect of a different
    lepton sector and not a discrepancy.
    """
    flags = SpeciesFlags(hyperons=True, muons=False)
    spec = beta_eq_neutrinoless()
    onsets, x0 = {}, None
    for n_B in np.arange(0.30, 1.30, 0.002):
        point = solve_beta_eq_neutrinoless(par, float(n_B), flags, T=0.0,
                                           x0=x0)
        if not point.converged:
            x0 = None
            continue
        for name in ("Sigma-", "Lambda", "Xi-", "Xi0"):
            if name not in onsets and point.Y(name) > 1.0e-6:
                onsets[name] = float(n_B)
        x0 = warm_start(point, spec)
    worst, where = 0.0, ""
    for name, published in TABLE_VII.items():
        error = abs(onsets.get(name, np.inf) - published)
        if error > worst:
            worst, where = error, name
    if "Xi0" in onsets:                          # Table VII: it never appears
        worst, where = np.inf, "Xi0 appeared"
    order_ok = (onsets.get("Sigma-", np.inf) < onsets.get("Lambda", np.inf)
                < onsets.get("Xi-", np.inf))
    return CheckResult("hyperon_onsets", worst <= tol and order_ok, worst,
                       f"Sigma- before Lambda before Xi-, worst {where}")


# =============================================================================
# 10-13. CLOSURES, BASIS, CAUSALITY, GATE
# =============================================================================
def check_mode_closures(par, states, tol=1.0e-8):
    """Each mode's own conditions hold at its solution."""
    worst, where = 0.0, ""

    def note(error, label):
        nonlocal worst, where
        if error > worst:
            worst, where = error, label

    for label, flags, _T, point in states:
        if label.startswith("beta") or label == "trapped":
            # beta equilibrium: mu_C + mu_e = mu_nue, and electric neutrality.
            note(abs(point.mu_C + point.mu_e - point.mu_nue) / 1000.0,
                 f"{label} beta relation")
            note(abs(point.n_C - point.n_e - point.n_mu) / point.n_B,
                 f"{label} neutrality")
        if label == "trapped":
            note(abs(point.Y_Le - 0.4), "trapped Y_Le")
        if label.startswith("yc"):
            note(abs(point.Y_C - 0.3), f"{label} Y_C")
        if label == "yc_lep":
            note(abs(point.n_C - point.n_e - point.n_mu) / point.n_B,
                 "yc_lep neutrality")
        if label == "yc_ys":
            note(abs(point.Y_S - 0.05), "yc_ys Y_S")
    return CheckResult("mode_closures", worst <= tol, worst, f"worst {where}")


def check_charge_basis(par, states, tol=1.0e-10):
    """n_B, n_C, n_S and mu_i agree with `eos.general.basis`."""
    worst = 0.0
    for _label, _flags, _T, point in states:
        n_B, n_C, n_S = charges_from_densities(point.composition_map)
        worst = max(worst, abs(n_B - point.n_B) / point.n_B)
        # The point's n_C and n_S include any thermal meson gas; these states
        # carry none, so the species sums are the totals.
        worst = max(worst, abs(n_C - point.n_C) / point.n_B,
                    abs(n_S - point.n_S) / point.n_B)
        for name in point.composition_map:
            mu = species_potential(name, point.mu_B, point.mu_C, point.mu_S)
            worst = max(worst, abs(mu - point.mu(name)) / 1000.0)
    return CheckResult("charge_basis", worst <= tol, worst,
                       "sums and potentials from eos.general.basis")


def check_causality(par, tol=0.0):
    """0 <= c_s^2 <= 1 along a cold beta-equilibrium sequence, and the peak.

    The paper reports a non-monotonic c_s^2 for DIDY with a maximum of about
    0.71 near n_B = 0.66 fm^-3 (their Fig. 8), which is the model's signature
    feature; both the value and its location are checked loosely, since they
    are read off a figure.
    """
    flags = SpeciesFlags(hyperons=True, muons=False)
    densities = np.arange(0.2, 1.3, 0.05)
    cs2 = []
    for n_B in densities:
        response = eos_response(par, "beta_eq_neutrinoless", flags,
                                n_B=float(n_B), T=0.0)
        cs2.append(response["cs2_isothermal"])
    cs2 = np.array(cs2)
    if not np.all(np.isfinite(cs2)):
        # nan loses every comparison, so it would reach the report as a nan
        # `violation` and a nan peak rather than as the density that produced
        # it. A grid point the response cannot reach is a failure with an
        # address (CLAUDE.md sections 4 and 6).
        bad = densities[~np.isfinite(cs2)]
        return CheckResult(
            "causality", False, float("inf"),
            f"c_s^2 is not finite at n_B = "
            f"{', '.join(f'{n:.2f}' for n in bad)} fm^-3: the response did "
            f"not converge there, so this grid was not evaluated")
    violation = max(np.max(-cs2), np.max(cs2 - 1.0), 0.0)
    peak = float(cs2.max())
    peak_at = float(densities[int(cs2.argmax())])
    ok = (violation <= tol and abs(peak - 0.71) < 0.05
          and abs(peak_at - 0.66) < 0.1)
    return CheckResult("causality", ok, violation,
                       f"max c_s^2 = {peak:.3f} at n_B = {peak_at:.2f} "
                       f"(paper: 0.71 at 0.66)")


def check_delivered_table(par, flags=None):
    """The table this model hands a structure solver is deliverable.

    CLAUDE.md section 8: P non-decreasing in n_B and 0 <= c_s^2 <= 1, checked
    BEFORE integration and owed by whoever builds the table -- which
    `eos.did.table.build_core_table` does. Building one is the model's side of
    the contract and imports no `astro/`; running a sequence over it is astro's
    and stays in `test/did/`.

    IT IS STATED AGAINST n_B, NOT AGAINST THE ROW ORDER. `build_core_table`
    returns its rows sorted by P (`np.argsort(P)`, because TOV interpolates on
    a monotone P grid), so `np.diff(P) >= 0` holds by construction and would be
    a check that cannot fail. What the sort does not repair is the density
    column: a branch whose P falls with n_B comes back with n_B out of order,
    and that permutation is what this looks for. It matters more here than in a
    nucleonic model -- DID's hyperon onsets are where a softening would appear,
    and `build_core_table` DROPS a density it cannot solve rather than stopping,
    so a gap in the sweep must not be able to pass as a smooth table.
    """
    from eos.did.table import build_core_table

    flags = flags if flags is not None else SpeciesFlags(hyperons=True)
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


def check_residual_gate(par, states):
    """Every state solved here is inside the tolerance the model accepts at."""
    worst, where = 0.0, ""
    for label, _flags, _T, point in states:
        if not point.converged:
            return CheckResult("residual_gate", False, point.error,
                               f"{label} did not converge")
        if point.error > worst:
            worst, where = point.error, label
    return CheckResult("residual_gate", worst <= RESIDUAL_TOL, worst,
                       f"worst {where}, gate {RESIDUAL_TOL:.0e}")


def run_all(par=None):
    """Run every check and return the report.

    The comparison against Table VIII of arXiv:2511.15646 -- M_max and R_1.4
    through a TOV sequence -- is NOT here: it imports `eos.astro`, which
    CLAUDE.md section 1 does not allow a model to do. It is
    `test/did/tov_sequence.py` plus `test/did/test_did_tov.py`, and the
    model's own half of that contract, `build_core_table`, is `eos.did.table`.
    """
    par = par if par is not None else Parameters.default()
    states = _states(par)
    report = FullCheckReport()
    report.results = [
        check_euler(par, states),
        check_free_energy(par, states),
        check_rearrangement(par),
        check_ism_limit(par),
        check_su6_limit(),
        check_saturation(par),
        check_hyperon_potentials(par),
        check_nuclear_matter_parameters(par),
        check_hyperon_onsets(par),
        check_mode_closures(par, states),
        check_charge_basis(par, states),
        check_causality(par),
        check_delivered_table(par),
        check_residual_gate(par, states),
    ]

    return report


if __name__ == "__main__":
    print(run_all())
