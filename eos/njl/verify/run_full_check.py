"""Physics invariants of the NJL model, checked in one place.

These are the statements the implementation has to satisfy whatever parameters
it is given, plus the published numbers the shipped parameter set must
reproduce. They are the fastest way to catch a wrong change, and every check
returns a structured pass/fail with the largest error it saw, so the suite
reports rather than prints.

  1. Base integrals     N = dP/dmu, Rs = -dP/dM, S = dP/dT and the per-mode
                        Euler relation, at five (M, mu, T, cutoff) points
                        spanning T = 0.5-50 MeV, cut and uncut.
  2. Surface term       P_log - P_k4 against its closed form. It is 0.1% of P
                        at (100, 500, 20) MeV and 39.9% at (140, 700, 50), so
                        a model that used the k^4/E form when cut would be
                        wrong by tens of percent at finite T and right at
                        T = 0, which is the way that error hides.
  3. RKH vacuum         M_u = 367.648, M_s = 549.479, (-phi_u)^(1/3) = 241.946,
                        (-phi_s)^(1/3) = 257.688, f_pi = 92.391 MeV, all
                        against the published fit with no fitting here; and
                        Omega_vac = eps_vac exactly, which is what makes the
                        vacuum constant cancel out of Euler.
  4. Bag constant       B_eff = (228.93 MeV)^4 = 357.49 MeV/fm^3, the derived
                        vacuum pressure difference across chiral restoration.
  5. Field energy       2 G_S sum phi^2 = sum (M - m)^2/(8 G_S) at K = 0,
                        exactly -- AND its failure by 34% with the determinant
                        term on, because a suite that only checked the K = 0
                        case would pass for an implementation that had dropped
                        the determinant.
  6. Euler              eps + P = T s + sum_j mu_j n_j at every solved point,
                        in every mode, paired and unpaired. Three assembly
                        bugs were caught by this during development, each of
                        which produced a plausible-looking equation of state.
  7. Free energy        f = eps - T s and f = -P + sum_j mu_j n_j.
  8. n = -dOmega/dmu    n_B against a finite difference of P along the neutral
                        solution. This is the check that distinguishes a
                        thermodynamically consistent gap equation from one
                        whose scalar densities are merely plausible: the
                        paired scalar density enters the mass equation, and
                        with its sign flipped this identity fails by 2.6e-4
                        where it otherwise holds to the finite-difference
                        floor.
  9. Solved anchor      the unpaired and 2SC neutral points at mu_B = 1500 MeV,
                        T = 0, eta_D = 0.75 (section 6 of the specification).
 10. Colour neutrality  n_3 and n_8 vanish identically in an unpaired phase at
                        mu_3 = mu_8 = 0, and are driven to zero by the two
                        colour potentials in a paired one.
 11. Pairing limit      the csc sector switches itself off: at Delta = 0 every
                        pairing correction is exactly zero and the paired code
                        path reproduces the unpaired state to the last bit.
 12. Paired entropy     the entropy of a gapped phase is exponentially
                        suppressed, s_paired/s_unpaired ~ 2e-4 at T = 5 MeV.
                        A merger simulation that used the unpaired entropy in
                        a gapped phase would not be approximately wrong.
 13. Mode closures      each mode's own conditions hold at its solution.
 14. Charge basis       n_B, n_C and n_S agree with `eos.general.basis` -- no
                        local copy of the map.
 15. Residual gate      every state solved here is inside the tolerance the
                        model claims to accept at.
 16. Saturation         the sharp cutoff freezes n_B at Lambda^3/pi^2 =
                        2.881 fm^-3. That ceiling is the regularization's, not
                        the solver's, and it is why the conformal asymptotics
                        of section 9 cannot be exhibited at lambda = 1 at all.
 17. Sound speed        0 <= c_s^2 <= 1 along a cold beta-equilibrium
                        sequence, rising towards 1/3 from above. Both off by
                        default (a density sweep with re-solved derivatives);
                        `--sound` switches them on.

The colour-pairing machinery itself -- gap-matrix multiplicities, the BdG
structure identities, the 2SC closed form, the BCS logarithm, the gap-kernel
sign structure and the Clogston limit -- belongs to `eos.general.pairing`,
which is shared with the colour-dielectric model, and is checked in
`test/general/test_pairing.py` where every other `eos.general` module is
checked.

Run as `python -m eos.njl.verify.run_full_check`.
"""
from dataclasses import dataclass, field

import numpy as np

from eos.general.basis import quark_charges
from eos.general.pairing import colour_densities, pair_block
from eos.general.physics_constants import hc3
from eos.general.solve import RESIDUAL_TOL
from eos.njl import (
    zero_pressure_point,
    Parameters, SpeciesFlags, bag_constant, condensate_energy, condensates,
    eos_response, eos_table, f_pi, kinetic_thermo, solve,
    solve_beta_eq_neutrinoless,
    solve_beta_eq_neutrino_trapped, solve_fixed_yc, solve_fixed_yc_ys,
    state_at, surface_term, vacuum_solution,
)

#: The published vacuum outputs of the RKH fit (Rehberg, Klevansky, Huefner,
#: Phys. Rev. C 53, 410 (1996)), as reproduced in section 4 of
#: docs/njl_csc_implementation.md.
RKH_VACUUM = {"M_u": 367.648, "M_s": 549.479, "phi_u_cbrt": 241.946,
              "phi_s_cbrt": 257.688, "f_pi": 92.391}

#: B_eff^(1/4) [MeV] -- a derived quantity here, not an input.
B_EFF_QUARTER = 228.93

#: The solved neutral points at mu_B = 1500 MeV, T = 0, eta_D = 0.75, no
#: vector coupling (section 6 of the specification). M_u and M_d of the 2SC
#: row are NOT gated: see `check_anchor`.
ANCHOR_UNPAIRED = {"M_u": 9.84, "M_d": 8.55, "M_s": 265.59, "mu_C": -34.20,
                   "n_B": 1.4319, "P": 302.12}
ANCHOR_2SC = {"M_s": 243.13, "mu_C": -62.27, "mu_8": -2.46, "Delta_3": 95.50,
              "n_B": 1.4887, "P": 324.75}


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
        lines = [f"NJL run_full_check: "
                 f"{'PASS' if self.all_passed else 'FAIL'}"]
        for r in self.results:
            tag = "ok " if r.passed else "FAIL"
            lines.append(f"  [{tag}] {r.name:22s} max_err={r.max_error:.2e}"
                         f"  {r.detail}")
        return "\n".join(lines)


def _states(par, include_csc=True):
    """One converged point per mode, cold and hot, unpaired and (optionally)
    with the pairing sector on."""
    plain = SpeciesFlags(csc=False)
    out = []
    for T in (0.0, 25.0):
        out.append(("beta", plain, T,
                    solve_beta_eq_neutrinoless(par, 1.0, T, plain)))
        out.append(("yc_lep", plain, T,
                    solve_fixed_yc(par, 1.0, 0.1, T, plain, leptons=True)))
        out.append(("yc_nolep", plain, T,
                    solve_fixed_yc(par, 1.0, 0.1, T, plain, leptons=False)))
    out.append(("trapped", plain, 25.0,
                solve_beta_eq_neutrino_trapped(par, 1.0, 0.2, 25.0, plain)))
    out.append(("yc_ys", plain, 25.0,
                solve_fixed_yc_ys(par, 1.0, 0.0, 0.3, 25.0, plain)))
    if include_csc:
        paired = SpeciesFlags(csc=True)
        out.append(("beta_2SC", paired, 0.0,
                    solve(par, "beta_eq_neutrinoless", 1.4887, 0.0, paired,
                          patterns=("2SC",))))
        out.append(("beta_2SC_hot", paired, 20.0,
                    solve(par, "beta_eq_neutrinoless", 1.4, 20.0, paired,
                          patterns=("2SC",))))
    return out


# =============================================================================
# 1-2. THE BASE INTEGRALS
# =============================================================================
_INTEGRAL_CASES = ((100.0, 500.0, 20.0, 602.3), (40.0, 590.0, 30.0, 602.3),
                   (140.0, 700.0, 5.0, 602.3), (140.0, 700.0, 50.0, 602.3),
                   (300.0, 400.0, 10.0, 1.0e6))


def check_base_integrals(tol=1.0e-6):
    """N = dP/dmu, Rs = -dP/dM, S = dP/dT, and eps = -P + mu N + T S.

    Two pieces of care, both about the TEST rather than the physics.

    Every point of the stencil is evaluated on the SAME quadrature rule, built
    once at the centre. Rebuilding it per point would move the panel
    breakpoints with the variable being differentiated, and a difference of
    two 1e9 pressures divided by 2e-3 amplifies that motion into the fourth
    digit -- a test of the panel layout, not of the identity.

    And each row is judged as |fd - analytic| <= tol |analytic| + noise, with
    noise the round-off floor of a central difference, 4 eps |P| / (2h). In a
    cold nearly-degenerate gas the entropy is genuinely ten orders of
    magnitude below the pressure -- s = 0.275 against P = 1.66e9 at
    (140, 700, 5) MeV -- so the finite difference cannot resolve it to better
    than a part in 1e4 however good the integrand is. Judging that row against
    s alone would report a failure of double precision as a failure of the
    model.
    """
    from eos.general.pairing import panel_nodes
    import math

    machine = 4.0 * np.finfo(float).eps
    worst, where = 0.0, ""
    h = 1.0e-3
    for m, mu, T, cut in _INTEGRAL_CASES:
        k_F = math.sqrt(mu ** 2 - m ** 2) if abs(mu) > abs(m) else 0.0
        rule = panel_nodes([k_F] if k_F > 0.0 else [], T, cut)

        def at(mu_, m_, T_):
            return kinetic_thermo(mu_, m_, T_, cut, quadrature=rule)

        r = at(mu, m, T)
        noise = machine * abs(r.P) / (2.0 * h)
        rows = [("N", (at(mu + h, m, T).P - at(mu - h, m, T).P) / (2.0 * h),
                 r.n),
                ("Rs", -(at(mu, m + h, T).P - at(mu, m - h, T).P) / (2.0 * h),
                 r.rho_s),
                ("S", (at(mu, m, T + h).P - at(mu, m, T - h).P) / (2.0 * h),
                 r.s),
                ("Euler", r.eps, -r.P + mu * r.n + T * r.s)]
        for name, got, analytic in rows:
            floor = 0.0 if name == "Euler" else noise
            error = max(abs(got - analytic) - floor, 0.0) / abs(analytic)
            if error > worst:
                worst, where = error, f"{name} at (M,mu,T)=({m},{mu},{T})"
    return CheckResult("base integrals", worst < tol, worst, where)


def check_surface_term(tol=1.0e-8):
    """P_log - P_k4 against its closed form, and that it is NOT small."""
    worst, biggest = 0.0, 0.0
    for m, mu, T, cut in _INTEGRAL_CASES[:4]:
        r = kinetic_thermo(mu, m, T, cut)
        difference = r.P - r.P_k4
        worst = max(worst, abs(surface_term(mu, m, T, cut) / difference - 1.0))
        biggest = max(biggest, difference / r.P)
    return CheckResult("surface term", worst < tol and biggest > 0.3, worst,
                       f"largest share of P: {biggest:.3f}")


# =============================================================================
# 3-5. THE VACUUM
# =============================================================================
def check_vacuum(par, tol=2.0e-5):
    """The RKH vacuum, and Omega_vac = eps_vac exactly."""
    vac = vacuum_solution(par)
    got = {"M_u": vac.M[0], "M_s": vac.M[2],
           "phi_u_cbrt": (-vac.phi[0]) ** (1.0 / 3.0),
           "phi_s_cbrt": (-vac.phi[2]) ** (1.0 / 3.0), "f_pi": vac.f_pi}
    worst, where = 0.0, ""
    for name, published in RKH_VACUUM.items():
        error = abs(got[name] / published - 1.0)
        if error > worst:
            worst, where = error, name
    identity = abs(vac.Omega - vac.eps)
    return CheckResult("RKH vacuum", worst < tol and identity == 0.0, worst,
                       f"worst {where}; Omega_vac - eps_vac = {identity:g}")


def check_bag_constant(par, tol=1.0e-4):
    """B_eff^(1/4) = 228.93 MeV, the derived vacuum pressure difference."""
    quarter = bag_constant(par) ** 0.25
    error = abs(quarter / B_EFF_QUARTER - 1.0)
    return CheckResult("bag constant", error < tol, error,
                       f"B_eff^(1/4) = {quarter:.2f} MeV "
                       f"= {bag_constant(par) / hc3:.2f} MeV/fm^3")


def check_field_energy_identity(par, tol=1.0e-12, expected_ratio=0.6616,
                                ratio_tol=1.0e-3):
    """2 G_S sum phi^2 = sum (M - m)^2/(8 G_S) at K = 0, and its 34% failure
    with the determinant term on.

    Both halves matter. The identity is a clean unit test of the scalar
    sector, but ONLY at K = 0; quoted as a check on the full model it would
    pass for an implementation that had silently dropped the determinant.
    """
    def ratio(parameters):
        vac = vacuum_solution(parameters)
        m = np.array(parameters.current_masses)
        left = 2.0 * parameters.G_S * float(np.sum(vac.phi ** 2))
        right = float(np.sum((vac.M - m) ** 2)) / (8.0 * parameters.G_S)
        return left / right

    from dataclasses import replace
    at_zero_K = abs(ratio(replace(par, K_Lambda5=0.0)) - 1.0)
    with_K = ratio(par)
    passed = at_zero_K < tol and abs(with_K - expected_ratio) < ratio_tol
    return CheckResult("field-energy identity", passed, at_zero_K,
                       f"ratio with K on = {with_K:.4f} (expected "
                       f"{expected_ratio}, i.e. a 34% failure)")


# =============================================================================
# 6-8. THERMODYNAMIC CONSISTENCY
# =============================================================================
def check_euler(par, states, tol=1.0e-8):
    """eps + P = T s + sum_j mu_j n_j, on the matter block of every state."""
    worst, where = 0.0, ""
    for label, _flags, _T, point in states:
        error = abs(point.state.euler_residual())
        if error > worst:
            worst, where = error, label
    return CheckResult("Euler", worst < tol, worst, f"worst in {where}")


def check_free_energy(par, states, tol=1.0e-8):
    """f = eps - T s, and f = -P + sum_j mu_j n_j, on the matter block."""
    worst, where = 0.0, ""
    for label, _flags, _T, point in states:
        st = point.state
        f_direct = st.eps - st.T * st.s
        f_legendre = -st.P + st.mu_dot_n
        scale = max(abs(f_direct), 1.0)
        error = abs(f_direct - f_legendre) / scale
        if error > worst:
            worst, where = error, label
    return CheckResult("free energy", worst < tol, worst, f"worst in {where}")


def check_density_derivative(par, tol=1.0e-5):
    """n_B = dP/dmu_B along the neutral solution, unpaired and paired.

    The identity that decides whether the gap equation is stationarity or
    merely plausible. A central difference in n_B is what makes it a test of
    the SOLUTION rather than of the assembly.
    """
    worst, where = 0.0, ""
    for flags, n_B, patterns in ((SpeciesFlags(csc=False), 1.0, None),
                                 (SpeciesFlags(csc=True), 1.4887, ("2SC",))):
        dn = 1.0e-2 * n_B
        points = [solve(par, "beta_eq_neutrinoless", n, 0.0, flags,
                        patterns=patterns)
                  for n in (n_B - dn, n_B, n_B + dn)]
        if not all(p.converged for p in points):
            return CheckResult("n = dP/dmu_B", False, float("inf"),
                               "a stencil point did not converge")
        slope = ((points[2].P_total - points[0].P_total)
                 / (points[2].mu_B - points[0].mu_B))
        error = abs(slope / points[1].n_B - 1.0)
        if error > worst:
            worst, where = error, ("paired" if patterns else "unpaired")
    return CheckResult("n = dP/dmu_B", worst < tol, worst, f"worst {where}")


# =============================================================================
# 9. THE SOLVED ANCHOR
# =============================================================================
def check_anchor(par, tol=1.0e-3):
    """The neutral points at mu_B = 1500 MeV, T = 0, eta_D = 0.75.

    M_u and M_d of the 2SC row are deliberately NOT gated. The specification
    reports (11.96, 7.65) there and this implementation gives (9.73, 8.90);
    the difference is the SIGN of the hole amplitudes in the paired scalar
    density, and the identity n_B = dP/dmu_B decides it -- it holds to the
    finite-difference floor with the sign used here and fails by 2.6e-4 with
    the other. Everything the two agree on is gated: M_s, Delta_3, mu_8, mu_C,
    n_B and P. Near chiral restoration M - m is the small difference of two
    large numbers, which is why a percent-level change in one scalar density
    moves the light masses by twenty percent and moves nothing else at all.
    See docs/DEFERRED.md.
    """
    worst, where = 0.0, ""

    def compare(got, reference, label):
        nonlocal worst, where
        for name, published in reference.items():
            error = abs(got[name] - published) / max(abs(published), 1.0)
            if error > worst:
                worst, where = error, f"{label}.{name}"

    plain = SpeciesFlags(csc=False)
    p = solve(par, "beta_eq_neutrinoless", ANCHOR_UNPAIRED["n_B"], 0.0, plain)
    compare({"M_u": p.M[0], "M_d": p.M[1], "M_s": p.M[2], "mu_C": p.mu_C,
             "n_B": p.n_B, "P": p.P_total}, ANCHOR_UNPAIRED, "unpaired")

    paired = SpeciesFlags(csc=True)
    q = solve(par, "beta_eq_neutrinoless", ANCHOR_2SC["n_B"], 0.0, paired,
              patterns=("2SC",))
    compare({"M_s": q.M[2], "mu_C": q.mu_C, "mu_8": q.mu_8,
             "Delta_3": q.Delta[2], "n_B": q.n_B, "P": q.P_total},
            ANCHOR_2SC, "2SC")
    return CheckResult("solved anchor", worst < tol, worst,
                       f"worst {where}; 2SC M_u = {q.M[0]:.2f}, "
                       f"M_d = {q.M[1]:.2f} (ungated, see docstring)")


# =============================================================================
# 10-12. THE PAIRING SECTOR
# =============================================================================
def check_colour_neutrality(par, tol=1.0e-10):
    """n_3 = n_8 = 0: identically in an unpaired phase, by solve in a paired one.

    The unpaired half is what justifies pinning mu_8 rather than solving for
    it: with no gap, n_3 responds only to mu_3 and n_8 only to mu_8, and both
    vanish exactly at zero. Letting a root finder hunt for mu_8 in an unpaired
    region is a documented way to lose an afternoon.
    """
    vac = vacuum_solution(par)
    st = state_at(par, vac.M, (0.0, 0.0, 0.0), 0.0, 1400.0, -40.0, 0.0,
                  0.0, 0.0, 0.0, vac=vac)
    scale = max(abs(st.n_q), 1.0)
    unpaired = max(abs(st.n_3), abs(st.n_8)) / scale

    paired = solve(par, "beta_eq_neutrinoless", 1.4887, 0.0,
                   SpeciesFlags(csc=True), patterns=("2SC",))
    solved = max(abs(paired.state.n_3), abs(paired.state.n_8)) / max(
        abs(paired.state.n_q), 1.0)
    worst = max(unpaired, solved)
    return CheckResult("colour neutrality", worst < tol, worst,
                       f"unpaired {unpaired:.1e}, solved {solved:.1e}")


def check_pairing_off_limit(par, tol=0.0):
    """At Delta = 0 the pairing sector is EXACTLY absent, not merely small.

    Every entry of the block is identically zero, so the unpaired phase is a
    clean limit of the same code and a paired solve reduces to the unpaired
    one bit for bit rather than to quadrature accuracy.
    """
    vac = vacuum_solution(par)
    mu_star = np.full(9, 450.0)
    block = pair_block(vac.M, mu_star, (0.0, 0.0, 0.0), 20.0, par.Lambda)
    worst = max(abs(block.delta_omega), abs(block.delta_s),
                float(np.max(np.abs(block.delta_n))),
                float(np.max(np.abs(block.delta_rho_s))),
                float(np.max(np.abs(block.gap_kernel))))
    return CheckResult("pairing off limit", worst <= tol, worst,
                       "delta_omega, delta_n, delta_rho_s, delta_s and the "
                       "kernels all exactly zero")


def check_paired_entropy(par, T=5.0, ratio_max=1.0e-3):
    """The entropy of a FULLY gapped phase is exponentially suppressed.

    At M* = 60, mu* = 450, Delta = 60 MeV in CFL -- where all nine modes are
    gapped -- the ratio of paired to unpaired entropy is 2.0e-4 at T = 5 MeV,
    3.3e-2 at 10, 3.0e-1 at 20 and 7.6e-1 at 50: four orders of magnitude at
    the coldest. A merger equation of state that used the unpaired entropy in
    a gapped phase would not be approximately wrong, it would be qualitatively
    wrong.

    CFL rather than 2SC deliberately. In 2SC only four of the nine modes pair,
    so five carry their full entropy and the ratio saturates near 1/2 -- which
    would say nothing about the suppression.
    """
    M = np.full(3, 60.0)
    mu_star = np.full(9, 450.0)
    unpaired = sum(kinetic_thermo(mu_star[j], M[0], T, par.Lambda).s
                   for j in range(9))
    block = pair_block(M, mu_star, (60.0, 60.0, 60.0), T, par.Lambda)
    ratio = (unpaired + block.delta_s) / unpaired
    return CheckResult("paired entropy", 0.0 < ratio < ratio_max, ratio,
                       f"s_paired/s_unpaired = {ratio:.2e} at T = {T} MeV "
                       f"(CFL, all nine modes gapped)")


# =============================================================================
# 13-15. THE MODES
# =============================================================================
def check_mode_closures(par, states, tol=1.0e-8):
    """Each mode's own conditions hold at its solution."""
    worst, where = 0.0, ""

    def note(error, label):
        nonlocal worst, where
        if error > worst:
            worst, where = error, label

    for label, flags, T, point in states:
        n_B = point.n_B
        if label.startswith("beta"):
            note(abs(point.mu_C + point.mu_e - point.mu_nu) / 1.0e3,
                 f"{label}: mu_C + mu_e = mu_nue")
            note(abs(point.Y_C * n_B - (point.n_e + point.n_mu)) / n_B,
                 f"{label}: neutrality")
            note(abs(point.mu_S) / 1.0e3, f"{label}: mu_S = 0")
        if label == "trapped":
            note(abs((point.n_e + point.n_nu) / n_B - 0.2),
                 "trapped: Y_Le")
        if label.startswith("yc"):
            target = 0.0 if label == "yc_ys" else 0.1
            note(abs(point.Y_C - target), f"{label}: Y_C")
        if label == "yc_ys":
            note(abs(point.Y_S - 0.3), "yc_ys: Y_S")
    return CheckResult("mode closures", worst < tol, worst, f"worst {where}")


def check_charge_basis(par, states, tol=1.0e-10):
    """n_B, n_C and n_S agree with `eos.general.basis` and with the colour sums."""
    worst, where = 0.0, ""
    for label, _flags, _T, point in states:
        n_B, n_C, n_S = quark_charges(point.n_u, point.n_d, point.n_s)
        scale = max(abs(point.n_B), 1.0e-6)
        for got, mine, name in ((n_B, point.n_B, "n_B"),
                                (n_C, point.Y_C * point.n_B, "n_C"),
                                (n_S, point.Y_S * point.n_B, "n_S")):
            error = abs(got - mine) / scale
            if error > worst:
                worst, where = error, f"{label}.{name}"
        n_3, n_8 = colour_densities(point.state.n_modes)
        for got, mine, name in ((n_3, point.state.n_3, "n_3"),
                                (n_8, point.state.n_8, "n_8")):
            error = abs(got - mine) / max(abs(point.state.n_q), 1.0)
            if error > worst:
                worst, where = error, f"{label}.{name}"
    return CheckResult("charge basis", worst < tol, worst, f"worst {where}")


def check_residual_gate(par, states):
    """Every state solved here is inside the tolerance the model accepts at."""
    worst, where = 0.0, ""
    for label, _flags, _T, point in states:
        if not point.converged or point.error > RESIDUAL_TOL:
            if point.error > worst:
                worst, where = point.error, label
    return CheckResult("residual gate", worst == 0.0, worst,
                       f"worst {where}" if where else "all inside the gate")


# =============================================================================
# 16. THE CONFORMAL LIMIT
# =============================================================================
def check_saturation_density(par, tol=1.0e-3):
    """The sharp cutoff saturates the density at n_B = Lambda^3/pi^2.

    With every integral cut at Lambda the nine modes can hold no more than
    n_q = 3 Lambda^3/pi^2, so n_B freezes at 2.881 fm^-3 for the RKH cutoff.
    That is not a solver failure but the regularization's own ceiling, and it
    is the reason the conformal asymptotics of section 9 CANNOT be exhibited
    under sharp-cutoff regularization: c_s^2 -> 1/3 is a statement about
    n_B -> infinity, and this model has no densities above 2.881 fm^-3 to
    approach it through. Reaching them is what lambda = Lambda_UV/Lambda > 1
    is for, and that needs the counterterm recorded in docs/DEFERRED.md.
    """
    ceiling = par.Lambda ** 3 / np.pi ** 2 / hc3
    flags = SpeciesFlags(csc=False)
    # Warm-started: a COLD guess stops converging around 2.1 fm^-3, because it
    # comes from the massless relation, which knows nothing about a cutoff and
    # so cannot see mu_B running away as the ceiling is approached.
    table = eos_table(par, "beta_eq_neutrinoless", flags,
                      axes={"nB": np.linspace(1.5, 2.95, 12), "T": [0.0]})
    solved = [point.n_B for point in table.points[0]]
    passed = (abs(ceiling - 2.881) < 1e-3 and solved
              and max(solved) < ceiling)
    return CheckResult("saturation density", passed,
                       abs(ceiling / 2.881 - 1.0),
                       f"ceiling Lambda^3/pi^2 = {ceiling:.3f} fm^-3; "
                       f"highest solved {max(solved) if solved else 0:.3f}")


def check_sound_speed(par, tol=0.05):
    """c_s^2 is causal and rising along a cold beta-equilibrium sequence.

    Only causality and monotonicity are asserted, and that is all this
    regularization can support. The conformal limit is NOT testable at
    lambda = 1: the ceiling of `check_saturation_density` sits at 2.881 fm^-3,
    and c_s^2 -> 1/3 is a statement about densities the sharp cutoff cannot
    supply. In the window it can, c_s^2 is still climbing towards 1/3 from
    BELOW -- 0.236 at n_B = 1.5 and 0.284 at 2.0 fm^-3 -- because the strange
    quark is still massive there. The specification's "from above", with the
    pairing correction +2 Delta^2/9 mu^2, is the asymptotic statement, and
    reaching it needs lambda > 1 (docs/DEFERRED.md).

    What this check would catch is the failure that matters: a constant G_V
    sends c_s^2 towards 1 rather than 1/3, which is the whole reason
    `eos.njl.couplings` carries the density-dependent forms.
    """
    flags = SpeciesFlags(csc=False)
    densities = (1.5, 2.0)
    values = [eos_response(par, "beta_eq_neutrinoless", flags, n_B=n,
                           T=0.0)["cs2_isothermal"] for n in densities]
    bad = [n for n, v in zip(densities, values) if not np.isfinite(v)]
    if bad:
        # A density the response cannot reach returns nan (CLAUDE.md section
        # 6), and every comparison below is False against one -- which fails
        # the check for the wrong reason and reports the nan instead of the
        # density. Fail it here, with the address.
        return CheckResult(
            "sound speed", False, float("inf"),
            f"c_s^2 is not finite at n_B = "
            f"{', '.join(f'{n:.1f}' for n in bad)} fm^-3: the response did "
            f"not converge there, so this sequence was not evaluated")
    passed = all(0.0 <= v <= 1.0 for v in values) and values[1] > values[0]
    return CheckResult("sound speed", passed, max(values),
                       f"c_s^2 = {values[0]:.4f}, {values[-1]:.4f} "
                       f"at n_B = 1.5, 2.0 fm^-3 (still below 1/3: the "
                       f"conformal limit needs lambda > 1)")


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
        surface = zero_pressure_point(par, flags,
                                        n_lo=0.25, n_hi=0.60,
                                        n_scan=15)
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


def run_all(par=None, include_csc=True, include_sound=True):
    """Run every check and return the report.

    `include_csc` adds the paired states, which are the expensive half (a
    paired solve diagonalises an 18x18 matrix at every quadrature node).

    `include_sound` adds the density-ceiling and sound-speed checks and is ON:
    causality is an invariant (CLAUDE.md section 8), and an invariant that only
    runs when a CLI flag is passed is not one. It used to default to False. Cost
    is not the reason to keep it off and was measured before the default moved:
    the two together are about 6 s against the suite's 3 s, so they are not
    `slow` either. The flag survives so a caller re-running the cheap invariants
    in a loop can still say so, and `--no-sound` is its command line.
    """
    par = par if par is not None else Parameters.default()
    states = _states(par, include_csc=include_csc)
    report = FullCheckReport()
    report.results = [
        check_base_integrals(),
        check_surface_term(),
        check_vacuum(par),
        check_bag_constant(par),
        check_field_energy_identity(par),
        check_euler(par, states),
        check_free_energy(par, states),
        check_mode_closures(par, states),
        check_charge_basis(par, states),
        check_residual_gate(par, states),
        _check_zero_pressure(par),
    ]
    if include_csc:
        report.results += [
            check_density_derivative(par),
            check_anchor(par),
            check_colour_neutrality(par),
            check_pairing_off_limit(par),
            check_paired_entropy(par),
        ]
    if include_sound:
        report.results += [check_saturation_density(par),
                           check_sound_speed(par)]
    return report


if __name__ == "__main__":
    import sys

    print(run_all(include_csc="--no-csc" not in sys.argv,
                  include_sound="--no-sound" not in sys.argv))
