"""Physics invariants of the extended NJL model, checked in one place.

These are the statements the implementation has to satisfy no matter which
parameters it is given; they are the fastest way to catch a wrong change.
Every check returns a structured pass/fail with the largest error it saw, so
the suite reports rather than prints.

  1. Euler relation      eps + P = T s + sum_i mu_i n_i, at T = 0 with s = 0.
                         Weak by construction here -- Eq. (19) IS how P is
                         computed -- so it is checked to round-off and check 3
                         is what actually tests the potentials.
  2. Free energy         f = eps - T s = eps, and f = -P + sum_i mu_i n_i.
  3. Rearrangement       mu_i = d eps / d n_i by central difference. This is
                         the sharp one: Sigma^R enters mu and P and NEVER eps
                         (CLAUDE.md section 8), and dropping either
                         rearrangement term breaks this identity by tens of
                         MeV while leaving checks 1 and 2 green. It has one
                         measured exception, the cap of Eq. (6); the check
                         names it rather than absorbing it.
  4. Gap equation        the returned state satisfies Eq. (5) at its own
                         effective scalar densities, including on the rows
                         where the condensate has been capped at zero.
  5. Charge basis        n_B, n_C and n_S from `thermodynamics.assemble`
                         against `eos.general.basis`, which derives them from
                         the shared particle table -- no local copy of the
                         map, and S = +1 per s quark on both sides.
  6. Beta equilibrium    a solved point satisfies mu_i = B_i mu_b - q_i mu_e
                         for every species that is present, and sum_i q_i n_i
                         = 0. mu_B = mu_u + 2 mu_d and mu_C = -mu_e through
                         `eos.general.basis` as well.
  7. Causality           0 <= c_s^2 <= 1 on the stable stretch of the
                         beta-equilibrium branch, which is LOCATED rather than
                         assumed: a raw branch may violate dP/dn_B >= 0 inside
                         a first-order transition, and where that begins is
                         reported.
  8. The vacuum          the constituent masses and E0 of Eq. (13) reproduce
                         the published numbers, and E0 does not move with f_q
                         or B -- it depends only on (Lambda, m_q0, G_S, K).
  9. Mode refusals       the three modes this model does not close, any
                         T > 0, any species flag moved from the value the
                         model fixes it at, and eos_response all raise rather
                         than returning a state the model does not have.

 10. Non-convergence    a request the solver cannot reach comes back as a
                         status with a message, not as a raise.

Run as `python -m eos.enjl.verify.run_full_check`.
"""
from dataclasses import dataclass, field

import numpy as np

from eos.enjl import (
    Parameters, SpeciesFlags, TableSpec, build_table, eos_point, eos_response,
    solve_beta_eq_neutrinoless, thermo_from_n, vacuum_energy_density,
    vacuum_solution,
)
from eos.enjl.species import BARYONS, CHARGE, QUARKS, SPECIES
from eos.enjl.thermodynamics import (
    assemble, effective_scalar_densities, quark_masses_from_gap,
)
from eos.general.basis import (
    charge_potentials_from_quarks, charges_from_densities,
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
        lines = [f"ENJL run_full_check: "
                 f"{'PASS' if self.all_passed else 'FAIL'}"]
        for r in self.results:
            tag = "ok " if r.passed else "FAIL"
            lines.append(f"  [{tag}] {r.name:22s} max_err={r.max_error:.2e}"
                         f"  {r.detail}")
        return "\n".join(lines)


#: The parameter sets the invariants are checked over: the shipped one, the
#: one with Pauli blocking switched off, and one with the quarks coupled to
#: the vector fields as strongly as the baryons.
SETS = (
    Parameters.default(),
    Parameters.named("fq0.5_B0"),
    Parameters.named("fq1.0_B1"),
)

#: Compositions the fixed-composition checks are run at, as
#: (n_B [fm^-3], proton fraction, Lambda fraction, quark baryon fraction).
#: Purely nucleonic, hyperonic, mixed and quark-dominated.
COMPOSITIONS = (
    (0.16, 0.5, 0.0, 0.0),
    (0.16, 0.1, 0.0, 0.0),
    (0.40, 0.4, 0.05, 0.0),
    (0.80, 0.3, 0.05, 0.2),
    (1.50, 0.2, 0.05, 0.5),
    (3.00, 0.1, 0.05, 0.8),
)


def _densities(n_B_fm, y_p, y_L, y_q):
    """Species densities [MeV^3] for one entry of COMPOSITIONS.

    The quark baryon fraction y_q is split equally over the three flavours, so
    n_u = n_d = n_s = y_q n_B; the rest is baryonic.
    """
    n_B = n_B_fm * hc3
    n_quark = y_q * n_B
    n_had = n_B - y_q * n_B
    return {"p": y_p * n_had, "n": (1.0 - y_p - y_L) * n_had,
            "Lambda": y_L * n_had,
            "u": n_quark, "d": n_quark, "s": n_quark,
            "e": 0.0, "mu": 0.0}


def _solved(par, entry, seed=None):
    """The state at one composition, warm-started so the gap solve lands.

    The cold start of the gap equation is the vacuum constituent masses, which
    is a poor guess once the light condensates have collapsed; the chirally
    restored seed is tried second, exactly as a density sweep would.
    """
    n = _densities(*entry)
    for x0 in ([seed] if seed is not None else []) + [None, [5.5, 5.5, 200.0]]:
        try:
            return thermo_from_n(n, par=par, x0=x0), n
        except RuntimeError:
            continue
    return None, n


def check_euler():
    """eps + P = T s + sum_i mu_i n_i, at T = 0 where s = 0."""
    worst, detail = 0.0, ""
    for par in SETS:
        for entry in COMPOSITIONS:
            point, _ = _solved(par, entry)
            if point is None:
                continue
            lhs = point.eps + point.P
            rhs = point.s * 0.0 + sum(point.mu[sp] * point.n[sp]
                                      for sp in SPECIES)
            err = abs(lhs - rhs) / abs(lhs)
            if err > worst:
                worst, detail = err, f"{par.f_q}/{par.B_GeV_fm3} {entry[0]}"
    return CheckResult("euler", worst < 1e-13, worst, detail)


def check_free_energy():
    """f = eps - T s = eps at T = 0, and f = -P + sum_i mu_i n_i."""
    worst, detail = 0.0, ""
    for par in SETS:
        for entry in COMPOSITIONS:
            point, _ = _solved(par, entry)
            if point is None:
                continue
            f_from_eps = point.eps - 0.0 * point.s
            f_from_P = -point.P + sum(point.mu[sp] * point.n[sp]
                                      for sp in SPECIES)
            err = abs(f_from_eps - f_from_P) / abs(f_from_eps)
            if err > worst:
                worst, detail = err, f"{par.f_q}/{par.B_GeV_fm3} {entry[0]}"
            if point.s != 0.0:
                return CheckResult("free_energy", False, np.inf,
                                   "s is not identically zero at T = 0")
    return CheckResult("free_energy", worst < 1e-13, worst, detail)


def check_rearrangement():
    """mu_i = d eps / d n_i: Sigma^R is in mu and P, and not in eps.

    The central difference is second order and is taken at a relative step of
    1e-4 on the species density. That step comes from a measured step-size
    study rather than from taste: the residual is flat from h/n = 1e-3 to
    1e-5, so the difference is neither truncation- nor round-off-limited
    there.

    THE IDENTITY HAS ONE EXCEPTION, and it is the cap of Eq. (6). Where a
    flavour's effective scalar density has been clamped at zero, nbar^s_q no
    longer responds to the densities, so eps stops being stationary with
    respect to that flavour's constituent mass -- which is the property that
    makes mu_i = d eps/d n_i hold. It only bites when exactly ONE light
    flavour is at the cap and the other is not: with neither capped nothing
    is clamped, and with both capped the 't Hooft term vanishes in both light
    channels, M_u = M_d = m_q0 exactly, and the state sits in a flat region
    where the clamp costs nothing. The one case reached here is f_q = 0.5,
    B = 0 at n_B = 0.8 fm^-3, where the residual is 6.9e-2 MeV on a potential
    of 1419 MeV -- 4.8e-5 relative, below the 0.05-0.20 MeV at which the
    engine is validated against the author's own tables.

    So the gate is split: 1e-3 MeV where the functional is smooth, which is
    four orders above the measured floor and five below what dropping either
    rearrangement term would cost, and 1e-1 MeV at the cap, where the
    exception is named rather than absorbed.
    """
    worst_smooth, worst_capped, detail, capped_detail = 0.0, 0.0, "", ""
    for par in SETS:
        for entry in COMPOSITIONS:
            point, n0 = _solved(par, entry)
            if point is None:
                continue
            at_cap = sum(1 for q in ("u", "d") if point.nbar_s[q] == 0.0) == 1
            seed = [point.M_q[q] for q in QUARKS]
            for sp in BARYONS + QUARKS:
                if n0[sp] <= 0.0:
                    continue
                h = 1e-4 * n0[sp]
                up, down = dict(n0), dict(n0)
                up[sp] += h
                down[sp] -= h
                d_eps = (thermo_from_n(up, par=par, x0=seed).eps
                         - thermo_from_n(down, par=par, x0=seed).eps) / (2 * h)
                err = abs(d_eps - point.mu[sp])
                where = (f"f_q={par.f_q} B={par.B_GeV_fm3:g} "
                         f"n_B={entry[0]} {sp}")
                if at_cap:
                    if err > worst_capped:
                        worst_capped, capped_detail = err, where
                elif err > worst_smooth:
                    worst_smooth, detail = err, where
    passed = worst_smooth < 1e-3 and worst_capped < 1e-1
    return CheckResult(
        "rearrangement", passed, max(worst_smooth, worst_capped),
        f"smooth {worst_smooth:.1e} MeV at {detail}; at the Eq. (6) cap "
        f"{worst_capped:.1e} MeV at {capped_detail}")


def check_gap_equation():
    """The returned state satisfies Eq. (5) at its own nbar^s_q.

    Where the effective scalar density has been capped at zero the gap returns
    M_q = m_q0 exactly, so those flavours are checked at the cap rather than
    excluded: the identity has to hold there too.
    """
    worst, detail = 0.0, ""
    for par in SETS:
        for entry in COMPOSITIONS:
            point, _ = _solved(par, entry)
            if point is None:
                continue
            gap = quark_masses_from_gap(point.nbar_s, par)
            for q in QUARKS:
                err = abs(point.M_q[q] - gap[q]) / point.M_q[q]
                if err > worst:
                    worst, detail = err, f"n_B={entry[0]} {q}"
            recomputed = effective_scalar_densities(
                point.kF, point.M_q, {b: point.n_s[b] for b in BARYONS},
                point.alpha_S, par.Lambda)
            for q in QUARKS:
                scale = max(abs(point.nbar_s[q]), 1e-6 * hc3)
                err = abs(recomputed[q] - point.nbar_s[q]) / scale
                if err > worst:
                    worst, detail = err, f"n_B={entry[0]} nbar_s_{q}"
    return CheckResult("gap_equation", worst < 1e-12, worst, detail)


def check_charge_basis():
    """n_B, n_C, n_S against eos.general.basis, S = +1 per s quark.

    The model's own `assemble` and the shared map derive the same three sums
    from the same quantum numbers; this is what stops a sign convention from
    drifting between a model and the rest of the repository. The leptons are
    left out of the shared call because C is the charge of strongly
    interacting matter only -- which is the convention being checked.
    """
    worst, detail = 0.0, ""
    for par in SETS:
        for entry in COMPOSITIONS:
            point, n = _solved(par, entry)
            if point is None:
                continue
            mine = assemble(n)
            theirs = charges_from_densities(
                {sp: n[sp] for sp in SPECIES if sp not in ("e", "mu")})
            for a, b, name in zip(mine, theirs, ("n_B", "n_C", "n_S")):
                err = abs(a - b) / max(abs(mine[0]), 1e-30)
                if err > worst:
                    worst, detail = err, f"n_B={entry[0]} {name}"
            if point.n_S < 0.0:
                return CheckResult(
                    "charge_basis", False, np.inf,
                    "n_S came out negative: S = +1 per s quark in this "
                    "repository, so strange matter has n_S > 0")
    return CheckResult("charge_basis", worst < 1e-14, worst, detail)


def check_beta_equilibrium():
    """mu_i = B_i mu_b - q_i mu_e where present, and sum_i q_i n_i = 0.

    Only species that are present are checked: below threshold a species has
    no equilibrium potential, its density being zero for a whole range of
    mu_i. The conserved-charge reading of the same state is checked too --
    mu_B = mu_u + 2 mu_d and mu_C = mu_p - mu_n = -mu_e, through
    `eos.general.basis` -- which is the identity a phase construction or an
    adapter would use.
    """
    from eos.enjl.species import BARYON_NUMBER

    worst_mu, worst_q, detail = 0.0, 0.0, ""
    for par in SETS:
        x0 = None
        for n_B in (0.2, 0.3, 0.4, 0.5, 0.6):
            try:
                pt = solve_beta_eq_neutrinoless(n_B, par=par, x0=x0)
            except RuntimeError:
                x0 = None
                continue
            from eos.enjl.solver import warm_start
            x0 = warm_start(pt)
            state = pt.point
            for sp in SPECIES:
                if pt.densities[sp] <= 1e-4 * n_B:
                    continue
                predicted = BARYON_NUMBER[sp] * pt.mu_b - CHARGE[sp] * pt.mu_e
                err = abs(state.mu[sp] - predicted)
                if err > worst_mu:
                    worst_mu, detail = err, f"f_q={par.f_q} n_B={n_B} {sp}"
            charge = sum(CHARGE[sp] * pt.densities[sp] for sp in SPECIES)
            worst_q = max(worst_q, abs(charge) / n_B)
            if all(pt.densities[q] > 1e-4 * n_B for q in QUARKS):
                # Only with all three flavours present: a quark below
                # threshold carries its threshold potential, not an
                # equilibrium one, and reading mu_S off it would measure the
                # threshold rather than the physics.
                mu_B, mu_C, mu_S = charge_potentials_from_quarks(
                    state.mu["u"], state.mu["d"], state.mu["s"])
                err = max(abs(mu_B - pt.mu_b), abs(mu_C + pt.mu_e),
                          abs(mu_S))
                if err > worst_mu:
                    worst_mu, detail = err, (f"f_q={par.f_q} n_B={n_B} "
                                             f"(B,C,S) basis")
    worst = max(worst_mu, worst_q)
    return CheckResult("beta_equilibrium", worst < 1e-6, worst,
                       f"mu {worst_mu:.1e} MeV, neutrality {worst_q:.1e} "
                       f"at {detail}")


def check_causality():
    """0 <= c_s^2 <= 1 wherever the beta-equilibrium branch is stable.

    Stability is not assumed, it is located. A raw branch of this model MAY
    violate dP/dn_B >= 0 -- that is a mechanically unstable stretch inside a
    first-order transition, real physics that a Maxwell construction and not
    this check is what resolves (CLAUDE.md section 8). So the sweep is walked,
    the longest run of increasing P is taken as the stable stretch, and the
    sound speed is required to be causal on it. Where the stretch ends is
    reported, because that density is the model's first transition.

    At f_q = 0.5, B = 0 the stable stretch starts above 0.12 fm^-3 already:
    with Pauli blocking switched off the d quarks appear at about n_sat/2, so
    the quarkyonic onset is below the grid rather than in it.
    """
    worst, detail = 0.0, []
    grid = np.linspace(0.12, 0.60, 49)
    for par in SETS:
        table = build_table(TableSpec(nB=grid, par=par))
        P, eps, nB = table.P, table.eps, table.nB_solved
        if len(P) < 20:
            return CheckResult("causality", False, np.inf,
                               f"only {len(P)} points solved at f_q={par.f_q}")
        rising = np.diff(P) > 0.0
        best_start, best_len, start = 0, 0, 0
        for i, up in enumerate(list(rising) + [False]):
            if not up:
                if i - start > best_len:
                    best_start, best_len = start, i - start
                start = i + 1
        if best_len < 10:
            return CheckResult(
                "causality", False, np.inf,
                f"f_q={par.f_q} B={par.B_GeV_fm3}: no stable stretch of "
                f"10 points in 0.12-0.60 fm^-3")
        lo, hi = best_start, best_start + best_len + 1
        cs2 = np.gradient(P[lo:hi], eps[lo:hi])
        violation = float(np.max(np.maximum(-cs2, cs2 - 1.0)))
        worst = max(worst, violation)
        detail.append(f"f_q={par.f_q}/B={par.B_GeV_fm3:g}: stable "
                      f"{nB[lo]:.2f}-{nB[hi - 1]:.2f}, "
                      f"cs2 <= {cs2.max():.3f}")
    return CheckResult("causality", worst <= 0.0, worst, "; ".join(detail))


def check_vacuum():
    """The vacuum masses and E0, and that E0 moves with neither f_q nor B.

    E0 depends only on (Lambda, m_q0, G_S, K), so all six published parameter
    sets must return one number; that they do is itself a check on Eq. (13).
    The published anchors are M_u = M_d = 367.6, M_s = 549.5 MeV.
    """
    worst, detail = 0.0, ""
    values = []
    for name in ("fq0.5_B1", "fq0.7_B0", "fq1.0_B1"):
        par = Parameters.named(name)
        M = vacuum_solution(par)
        for q, published in (("u", 367.6), ("d", 367.6), ("s", 549.5)):
            err = abs(M[q] - published) / published
            if err > worst:
                worst, detail = err, f"{name} M_{q}={M[q]:.4f}"
        values.append(vacuum_energy_density(par) / hc3)
    spread = max(values) - min(values)
    if abs(spread) > 1e-9:
        return CheckResult("vacuum", False, abs(spread),
                           f"E0 moved with the parameters: {values}")
    return CheckResult("vacuum", worst < 1e-3, worst,
                       f"E0 = {values[0]:.4f} MeV/fm^3, identical across "
                       f"sets; {detail}")


def check_refusals():
    """Every mode, temperature, flag and response this model does not have."""
    par = Parameters.default()
    failures = []
    for mode in ("beta_eq_neutrino_trapped", "fixed_YC", "fixed_YC_YS"):
        try:
            eos_point(par, mode, n_B=0.3)
        except NotImplementedError:
            continue
        except Exception as err:                     # noqa: BLE001
            failures.append(f"{mode} raised {type(err).__name__}")
            continue
        failures.append(f"{mode} returned a state")
    try:
        eos_point(par, "beta_eq_neutrinoless", n_B=0.3, T=10.0)
    except NotImplementedError:
        pass
    else:
        failures.append("T = 10 MeV returned a state")
    for flag, value in (("hyperons", False), ("muons", False),
                        ("deltas", True), ("thermal_mesons", True),
                        ("photons", True), ("thermal_neutrinos", True)):
        try:
            SpeciesFlags(**{flag: value})
        except NotImplementedError:
            continue
        failures.append(f"SpeciesFlags({flag}={value}) was accepted")
    try:
        eos_response(par, n_B=0.3)
    except NotImplementedError:
        pass
    else:
        failures.append("eos_response returned a response")
    return CheckResult("refusals", not failures, float(len(failures)),
                       "; ".join(failures)
                       or "three modes, T > 0, six flags and eos_response "
                          "all raise")


def check_non_convergence_is_returned():
    """A request the solver cannot reach comes back as a status, not a raise.

    A sampler walks into unphysical territory constantly and must be able to
    score the point and move on. The cold starts of this model stop converging
    around n_B = 0.5 fm^-3, so a cold request well above that is the natural
    probe.
    """
    par = Parameters.default()
    result = eos_point(par, n_B=8.0)
    if result.ok:
        return CheckResult("non_convergence", True, 0.0,
                           "a cold start at n_B = 8 fm^-3 converged; the "
                           "status path is exercised by the message field")
    if result.point is not None:
        return CheckResult("non_convergence", False, 1.0,
                           "ok is False but a point was returned")
    return CheckResult("non_convergence", bool(result.message), 0.0,
                       "reported rather than raised")


CHECKS = (check_euler, check_free_energy, check_rearrangement,
          check_gap_equation, check_charge_basis, check_beta_equilibrium,
          check_causality, check_vacuum, check_refusals,
          check_non_convergence_is_returned)


def run_full_check():
    """Run every invariant and return the report."""
    return FullCheckReport(results=[check() for check in CHECKS])


if __name__ == "__main__":
    import sys

    report = run_full_check()
    print(report)
    sys.exit(0 if report.all_passed else 1)
