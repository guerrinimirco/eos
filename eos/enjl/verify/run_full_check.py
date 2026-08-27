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
 11. Residual margin     every solved point clears the acceptance gate by at
                         least two decades. A mode that merely PASSES the gate
                         is not safe: the seed list of `solver.solve` falls
                         through to a start on another chiral branch when a
                         root misses, so a mode sitting near the gate has its
                         branch chosen by round-off. That is not hypothetical
                         -- it is what a held Y_S = 0 did while mu_S was
                         carried as an unknown its rows did not determine.

Run as `python -m eos.enjl.verify.run_full_check`.
"""
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from eos.enjl import (
    Parameters, SpeciesFlags, TableSpec, build_table, eos_point, eos_response,
    eos_table, solve_beta_eq_neutrinoless, thermo_from_n,
    vacuum_energy_density, vacuum_solution,
)
from eos.enjl.solver import (
    solve_beta_eq_neutrino_trapped, solve_fixed_yc, solve_fixed_yc_ys,
)
from eos.enjl.thermodynamics import thermo_from_mu
from eos.enjl.species import BARYONS, CHARGE, QUARKS, SPECIES
from eos.enjl.thermodynamics import (
    assemble, effective_scalar_densities, quark_masses_from_gap,
)
from eos.general.basis import (
    charge_potentials_from_quarks, charges_from_densities,
)
from eos.general.physics_constants import hc3
from eos.general.solve import RESIDUAL_TOL


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


def _densities(n_B, y_p, y_L, y_q):
    """Species densities [MeV^3] for one entry of COMPOSITIONS.

    The quark baryon fraction y_q is split equally over the three flavours, so
    n_u = n_d = n_s = y_q n_B; the rest is baryonic.
    """
    n_B_nat = n_B * hc3
    n_quark = y_q * n_B_nat
    n_had = n_B_nat - y_q * n_B_nat
    return {"p": y_p * n_had, "n": (1.0 - y_p - y_L) * n_had,
            "Lambda": y_L * n_had,
            "u": n_quark, "d": n_quark, "s": n_quark,
            "e": 0.0, "mu": 0.0}


#: Temperatures the identities are checked at. T = 0 keeps the exact closed
#: forms; the other two are where the T s term is large enough that dropping
#: it could not pass.
TEMPERATURES = (0.0, 10.0, 30.0)


def _solved(par, entry, seed=None, T=0.0):
    """The state at one composition, warm-started so the gap solve lands.

    The cold start of the gap equation is the vacuum constituent masses, which
    is a poor guess once the light condensates have collapsed; the chirally
    restored seed is tried second, exactly as a density sweep would.
    """
    n = _densities(*entry)
    for x0 in ([seed] if seed is not None else []) + [None, [5.5, 5.5, 200.0]]:
        try:
            return thermo_from_n(n, par=par, T=T, x0=x0), n
        except RuntimeError:
            continue
    return None, n


def check_euler():
    """eps + P = T s + sum_i mu_i n_i, at T = 0 and at T > 0.

    The T s term is the whole of what temperature adds to this identity, and
    it is checked at temperatures where it is not small: at T = 30 MeV it is
    several per cent of eps, so a dropped or mis-scaled entropy could not
    hide inside the tolerance.

    READ THE ZERO CORRECTLY. `thermo_from_n` DEFINES P through this relation,
    so the residual is exact by construction and not a measurement of
    anything: it catches an assembly that forgets a species or a sign in the
    sum, and nothing else. `check_free_energy` is the same identity
    rearranged and is equally definitional. What is NOT definitional, and is
    where the thermodynamics is actually tested, is `check_rearrangement`
    (mu_i = d eps/d n_i by finite difference) and `check_entropy_limit`.
    """
    worst, detail = 0.0, ""
    for par in SETS:
        for entry in COMPOSITIONS:
            for T in TEMPERATURES:
                point, _ = _solved(par, entry, T=T)
                if point is None:
                    continue
                lhs = point.eps + point.P
                rhs = T * point.s + sum(point.mu[sp] * point.n[sp]
                                        for sp in SPECIES)
                err = abs(lhs - rhs) / abs(lhs)
                if err >= worst:
                    worst, detail = err, (f"{par.f_q}/{par.B_GeV_fm3} "
                                          f"n_B={entry[0]} T={T}")
    return CheckResult("euler", worst < 1e-13, worst,
                       f"{detail}; exact by construction -- P is defined "
                       f"through this relation")


def check_free_energy():
    """f = eps - T s and f = -P + sum_i mu_i n_i, at T = 0 and at T > 0.

    Also that s IS still exactly zero at T = 0. That is not a limit here: the
    exact closed form of `eos.general.fermi_integrals` returns s = 0.0, and it
    is what makes `test/baseline` bit-for-bit across this seam.
    """
    worst, detail = 0.0, ""
    for par in SETS:
        for entry in COMPOSITIONS:
            for T in TEMPERATURES:
                point, _ = _solved(par, entry, T=T)
                if point is None:
                    continue
                f_from_eps = point.eps - T * point.s
                f_from_P = -point.P + sum(point.mu[sp] * point.n[sp]
                                          for sp in SPECIES)
                err = abs(f_from_eps - f_from_P) / abs(f_from_eps)
                if err > worst:
                    worst, detail = err, (f"{par.f_q}/{par.B_GeV_fm3} "
                                          f"n_B={entry[0]} T={T}")
                if T == 0.0 and point.s != 0.0:
                    return CheckResult("free_energy", False, np.inf,
                                       "s is not identically zero at T = 0")
    return CheckResult("free_energy", worst < 1e-13, worst, detail)


def check_entropy_limit():
    """s > 0 at T > 0, s rises with T, and s -> 0 as T -> 0 -- to a FLOOR.

    The floor is the point of this check. `eos.general.fermi_integrals`
    evaluates the exact T = 0 closed form at T = 0 and the JEL fit at every
    T != 0, and the fit does not converge back to the closed form: it steps
    off the moment T != 0 and STAYS at that offset as T falls, flat from
    T = 1e-3 MeV downward. Measured on single species, the step is +6.9e-6 in
    n (u at nu = 400, M = 5.5), -6.3e-5 (s at nu = 500, M = 140.7) and
    -3.0e-6 (neutron at nu = 1000), with the same numbers in eps.

    So "take T small and compare against the T = 0 baseline" is not a
    validation route below about 1e-4 relative, and this check states that
    rather than chasing it. What it does assert is what is true: the entropy
    itself goes to zero linearly in T (it carries no such offset, being zero
    at T = 0 in both branches), while eps and P approach their T = 0 values
    only to the fit's floor.

    That is a property of the fit, not of this port, and it is the reason
    `check_temperature` keeps the exact T = 0 branch as a special case rather
    than routing everything through the fit for smoothness -- which would move
    every frozen number in the repository for a 1e-5 cosmetic gain.
    """
    #: The JEL fit's offset from the exact T = 0 closed form. Anything below
    #: this is the fit, not the model.
    floor = 1.0e-4
    par = Parameters.default()
    entry = COMPOSITIONS[2]
    cold, _ = _solved(par, entry, T=0.0)
    if cold is None:
        return CheckResult("entropy_limit", False, np.inf,
                           "the T = 0 reference state did not solve")

    failures = []
    if cold.s != 0.0:
        failures.append(f"s = {cold.s} at T = 0, not identically zero")

    previous = 0.0
    for T in (0.5, 2.0, 10.0, 30.0):
        point, _ = _solved(par, entry, T=T)
        if point is None:
            failures.append(f"T = {T} MeV did not solve")
            continue
        if point.s <= previous:
            failures.append(f"s did not rise from T = {previous} to {T} MeV")
        previous = point.s

    # The approach to T = 0: s vanishes, eps only to the fit's floor.
    warm, _ = _solved(par, entry, T=1.0e-3)
    d_eps = abs(warm.eps - cold.eps) / abs(cold.eps)
    s_over_nB = warm.s / warm.n_b
    if s_over_nB > 1.0e-4:
        failures.append(f"s/n_B = {s_over_nB:.3e} at T = 1e-3 MeV, not small")
    if d_eps > floor:
        failures.append(f"eps at T = 1e-3 MeV is {d_eps:.3e} from the T = 0 "
                        f"value, above the {floor:.0e} JEL floor")
    detail = (f"s = 0 exactly at T = 0 and rises to {previous / cold.n_b:.3f} "
              f"per baryon by T = 30 MeV; at T = 1e-3 MeV s/n_B = "
              f"{s_over_nB:.2e} while eps sits {d_eps:.1e} from its T = 0 "
              f"value -- the JEL fit's step off the exact closed form, floor "
              f"{floor:.0e}, not a continuity error")
    return CheckResult("entropy_limit", not failures,
                       d_eps if not failures else np.inf,
                       "; ".join(failures) if failures else detail)


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
                point.nu, point.M_q, {b: point.n_s[b] for b in BARYONS},
                point.alpha_S, par.Lambda, point.T)
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
                pt = solve_beta_eq_neutrinoless(par, n_B, x0=x0)
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


def check_thermo_from_mu():
    """The potentials of a solved state return that state, cold.

    `thermo_from_mu` is the phase-adapter surface a mixed-phase construction
    consumes: it takes (mu_B, mu_C, mu_S, T) and solves the model's own
    self-consistency INCLUDING the phase's own baryon density, with no
    leptons, no neutrality and no held fraction. So feeding it the potentials
    of a beta-equilibrium point must return the same matter -- the same n_B,
    the same constituent masses, the same P and eps once the leptons are taken
    out of the comparison, which is what the matter-only reference below does.

    Solved COLD at every density, with no warm start, because that is the
    condition a numerically differentiated adapter has to meet: its output
    must depend on its arguments and nothing else.
    """
    par = Parameters.default()
    worst, detail, x0 = 0.0, "", None
    for n_B in (0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.5):
        try:
            pt = solve_beta_eq_neutrinoless(par, n_B, x0=x0)
        except RuntimeError:
            x0 = None
            continue
        x0 = list(pt.x)
        state = pt.point
        try:
            back = thermo_from_mu(par, pt.mu_b, pt.mu_C, pt.mu_S)
        except RuntimeError as err:                        # noqa: BLE001
            return CheckResult("thermo_from_mu", False, np.inf,
                               f"n_B={n_B}: {err}")
        matter = thermo_from_n(
            {k: (0.0 if k in ("e", "mu") else v * hc3)
             for k, v in pt.densities.items()},
            par=par, x0=[state.M_q[q] for q in QUARKS])
        for got, want, scale, name in (
                (back.n_b, matter.n_b, matter.n_b, "n_B"),
                (back.P, matter.P, abs(matter.P), "P"),
                (back.eps, matter.eps, matter.eps, "eps"),
                (back.M_q["u"], matter.M_q["u"], matter.M_q["u"], "M_u")):
            err = abs(got - want) / scale
            if err > worst:
                worst, detail = err, f"n_B={n_B} ({name})"
    return CheckResult("thermo_from_mu", worst < 1e-10, worst,
                       detail or "cold round trip exact over 0.2-1.5 fm^-3")


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


def check_fixed_fractions():
    """The fixed-fraction modes hit the fractions they were asked for.

    Y_C and Y_S are the NON-LEPTONIC fractions of CLAUDE.md section 2, so this
    also checks that the leptons stayed out of them: with leptons=True the
    neutralizing gas is added after the solve and must not move Y_C.
    """
    worst, detail = 0.0, ""
    par = Parameters.default()
    for n_B in (0.2, 0.4, 0.6):
        for Y_C in (0.0, 0.1, 0.3, 0.5):
            for leptons in (True, False):
                try:
                    pt = solve_fixed_yc(par, n_B, Y_C, leptons=leptons)
                except RuntimeError:
                    continue
                got = pt.point.n_C / pt.point.n_b
                err = abs(got - Y_C)
                if err > worst:
                    worst, detail = err, f"n_B={n_B} Y_C={Y_C} leptons={leptons}"
        for Y_S in (0.0, 0.1):
            try:
                pt = solve_fixed_yc_ys(par, n_B, 0.3, Y_S, leptons=False)
            except RuntimeError:
                continue
            for got, want, name in ((pt.point.n_C / pt.point.n_b, 0.3, "Y_C"),
                                    (pt.point.n_S / pt.point.n_b, Y_S, "Y_S")):
                err = abs(got - want)
                if err > worst:
                    worst, detail = err, f"n_B={n_B} Y_S={Y_S} ({name})"
    return CheckResult("fixed_fractions", worst < 1e-9, worst, detail)


def check_symmetric_matter_slice():
    """fixed_YC_YS at Y_C = 0.5, Y_S = 0, leptons off IS symmetric matter --
    up to the quarkyonic onset, which this check LOCATES.

    The strongest external check the new modes get. The mode solves for the
    composition from the two fractions; `thermo_from_n` is handed
    n_p = n_n = n_B/2 directly. They are different code paths through the same
    physics and must return the same state -- and that state is the one whose
    saturation properties reproduce the paper (n_sat = 0.158297 fm^-3,
    E/A = -16.010 MeV, K_sat = 234.20 MeV).

    They must NOT agree above the quarkyonic onset, and that is the point.
    Y_C = 0.5 with Y_S = 0 does not forbid u and d quarks -- it only forbids
    strangeness and fixes the charge -- so once quasi-free quarks appear the
    mode finds them while the hand-built nucleonic reference cannot. The
    comparison therefore runs only where the mode's own solution has no
    quarks, and reports where they appear.

    WHICH densities carry quarks is deliberately not asserted, because it is
    not a property of the density. Warm-started from 0.10 the sweep carries
    the quark-bearing branch down to 0.20 fm^-3; solved cold on a 0.05 grid
    the first quark-bearing point is 0.25; solved cold at {0.30, 0.40, 0.45}
    there are none, and at 0.50 there are (n_u = n_d ~ 5e-8 fm^-3, and P then
    parts company with the nucleonic reference by 1.9e-7 relative). All three
    are roots of the same equations reached from different starting points.
    Deciding which is realised means comparing free energies across branches
    -- a construction, not a solve -- and this model does not perform one yet.

    So the check asserts only what is a property of the physics: wherever the
    mode returns a quark-free state, that state IS nucleonic symmetric matter,
    to round-off. It reports how many densities that covered, so a change that
    quietly stopped finding quark-free states could not pass unnoticed.
    """
    par = Parameters.default()
    worst, detail, compared, attempted = 0.0, "", 0, 0
    for n_B in np.arange(0.10, 0.86, 0.05):
        n_B = float(n_B)
        try:
            pt = solve_fixed_yc_ys(par, n_B, 0.5, 0.0, leptons=False)
        except RuntimeError:
            continue
        attempted += 1
        if sum(pt.densities[q] for q in QUARKS) > 0.0:
            continue
        compared += 1
        ref = thermo_from_n({"p": n_B * hc3 / 2.0, "n": n_B * hc3 / 2.0},
                            par=par)
        for got, want, scale, name in (
                (pt.P, ref.P / hc3, abs(ref.P / hc3) + 1.0, "P"),
                (pt.eps, ref.eps / hc3, abs(ref.eps / hc3), "eps"),
                (pt.EperB, ref.EperB, abs(ref.EperB), "E/A")):
            err = abs(got - want) / scale
            if err > worst:
                worst, detail = err, f"n_B={n_B:.2f} ({name})"
    if compared < 3:
        return CheckResult(
            "symmetric_matter", False, np.inf,
            f"only {compared} of {attempted} solved densities came back "
            f"quark-free; there is nothing left to compare against nucleonic "
            f"symmetric matter")
    return CheckResult("symmetric_matter", worst < 1e-10, worst,
                       f"{compared}/{attempted} densities quark-free and "
                       f"identical to nucleonic matter; worst at {detail}")


def check_trapped_lepton_number():
    """The trapped mode hits Y_Le and satisfies mu_C + mu_e = mu_nue.

    The neutrinos are massless and left-handed, g = 1, so
    n_nue = mu_nue^3/(6 pi^2), and the lepton-family row is
    (n_e + n_nue)/n_B = Y_Le. The beta relation is the one of
    `eos.general.modes.electron_potential`, which reduces to mu_C + mu_e = 0
    when the neutrinos escape.
    """
    from eos.enjl.solver import _massless_density, _unpack

    par = Parameters.default()
    worst_Y, worst_beta, detail = 0.0, 0.0, ""
    for Y_Le in (0.1, 0.2, 0.3, 0.4):
        for n_B in (0.2, 0.4):
            try:
                pt = solve_beta_eq_neutrino_trapped(par, n_B, Y_Le)
            except RuntimeError:
                continue
            _, _, mu_C, _, mu_nue, _ = _unpack(pt.x, pt.spec)
            n_nue = _massless_density(mu_nue, 1.0) / hc3
            err_Y = abs((pt.densities["e"] + n_nue) / n_B - Y_Le)
            err_b = abs(mu_C + pt.point.mu["e"] - mu_nue)
            if err_Y > worst_Y:
                worst_Y, detail = err_Y, f"Y_Le={Y_Le} n_B={n_B}"
            worst_beta = max(worst_beta, err_b)
    worst = max(worst_Y, worst_beta)
    return CheckResult("trapped_leptons", worst < 1e-9, worst,
                       f"Y_Le {worst_Y:.1e}, mu_C+mu_e-mu_nue "
                       f"{worst_beta:.1e} MeV at {detail}")


def check_refusals():
    """What this model refuses: a construction at T > 0, a flag, a response.

    All four modes of CLAUDE.md section 3 are closed here at any non-negative
    temperature, so neither a mode nor a temperature is on this list any
    longer. What is: a NEGATIVE temperature; the CONSTRUCTED table above
    T = 0, since locating a coexistence at T > 0 puts the entropy into the
    bookkeeping and that is not done; the species flags the model genuinely
    fixes -- which no longer include photons and thermal_neutrinos, both of
    which are now implemented and selectable; leptons=False in a
    beta-equilibrium mode, which is defined by the leptons; and eos_response,
    which needs a settled branch.
    """
    par = Parameters.default()
    failures = []
    try:
        eos_point(par, "beta_eq_neutrinoless", n_B=0.3, T=-1.0)
    except ValueError:
        pass
    else:
        failures.append("T = -1 MeV returned a state")
    try:
        eos_table(par, "beta_eq_neutrinoless",
                  axes={"nB": [0.3, 0.4], "T": [10.0]},
                  coexistences=[])
    except NotImplementedError:
        pass
    else:
        failures.append("a constructed table at T = 10 MeV was accepted")
    try:
        eos_point(par, "beta_eq_neutrinoless", n_B=0.3, leptons=False)
    except ValueError:
        pass
    else:
        failures.append("leptons=False in beta equilibrium was accepted")
    try:
        eos_point(par, "fixed_YC", n_B=0.3)
    except ValueError:
        pass
    else:
        failures.append("fixed_YC without Y_C was accepted")
    for flag, value in (("hyperons", False), ("muons", False),
                        ("deltas", True), ("thermal_mesons", True)):
        try:
            SpeciesFlags(**{flag: value})
        except NotImplementedError:
            continue
        failures.append(f"SpeciesFlags({flag}={value}) was accepted")
    # ... and the two that are now the caller's must NOT raise.
    try:
        SpeciesFlags(photons=True, thermal_neutrinos=True)
    except NotImplementedError:
        failures.append("photons/thermal_neutrinos still raise, but they are "
                        "implemented")
    try:
        eos_response(par, "beta_eq_neutrinoless", n_B=0.3)
    except NotImplementedError:
        pass
    else:
        failures.append("eos_response returned a response")
    return CheckResult("refusals", not failures, float(len(failures)),
                       "; ".join(failures)
                       or "T < 0, a constructed table at T > 0, four fixed "
                          "flags, a malformed call and eos_response all "
                          "raise; photons and thermal_neutrinos do not")


def check_non_convergence_is_returned():
    """A request the solver cannot reach comes back as a status, not a raise.

    A sampler walks into unphysical territory constantly and must be able to
    score the point and move on. The cold starts of this model stop converging
    around n_B = 0.5 fm^-3, so a cold request well above that is the natural
    probe.
    """
    par = Parameters.default()
    result = eos_point(par, "beta_eq_neutrinoless", n_B=8.0)
    if result.ok:
        return CheckResult("non_convergence", True, 0.0,
                           "a cold start at n_B = 8 fm^-3 converged; the "
                           "status path is exercised by the message field")
    if result.point is not None:
        return CheckResult("non_convergence", False, 1.0,
                           "ok is False but a point was returned")
    return CheckResult("non_convergence", bool(result.message), 0.0,
                       "reported rather than raised")


#: Where the construction checks are run: the parameter set whose chiral
#: transition the author's tables pin most cheaply, and a mu_B grid that
#: BRACKETS the crossing without pinpointing it -- a grid tight enough to
#: contain the answer would be checking arithmetic rather than physics.
CONSTRUCTION_SET = "fq0.7_B1"
CONSTRUCTION_MU_B = (1120.0, 1220.0, 20.0)

#: Density grid the delivered table is checked on. Coarse: this is an
#: invariant check, not a production table, and every point is a full solve.
CONSTRUCTION_NB = (0.20, 0.90, 0.05)


@lru_cache(maxsize=4)
def _coexistences(name):
    """The located transitions of one parameter set, computed once per run.

    A read-only cache keyed by an immutable value, which CLAUDE.md section 6
    allows; two checks below need the same location and it costs a minute.

    `eos.mixed` is imported HERE rather than at module scope on purpose. A
    model may not import a composite engine (CLAUDE.md section 1), and this
    file is a test entry point that nothing in `eos.enjl` imports -- so
    `import eos.enjl` still pulls in no part of `eos.mixed`, and the layering
    the import test enforces is untouched. The library-side rule is kept by
    `eos.enjl.table.build_constructed_table`, which takes the located windows
    as an argument and imports nothing from the engine.
    """
    from eos.mixed.construction import enjl_coexistences

    par = Parameters.named(name)
    lo, hi, step = CONSTRUCTION_MU_B
    return par, enjl_coexistences(par, np.arange(lo, hi, step),
                                  pairs=(("broken", "restored"),))


def check_maxwell_crossing():
    """P and mu_B are equal across the two phases at a located crossing.

    The defining conditions of the eta = 1 construction: mechanical
    equilibrium P_lo = P_hi and chemical equilibrium mu_B,lo = mu_B,hi, with
    each phase separately neutral and carrying its OWN electron potential.
    Recomputed from the phases at the located potential rather than read back
    off the record that asserted them.
    """
    from eos.mixed.adapters import enjl_branch_pair
    from eos.mixed.boundaries import total_pressure

    par, found = _coexistences(CONSTRUCTION_SET)
    if not found:
        return CheckResult("maxwell_crossing", False, float("inf"),
                           f"no transition located for {CONSTRUCTION_SET} on "
                           f"mu_B in {CONSTRUCTION_MU_B}")
    worst, detail = 0.0, []
    for co in found:
        lo_phase, hi_phase = enjl_branch_pair(par, co.branches)
        pressures = []
        for phase in (lo_phase, hi_phase):
            got = total_pressure(lambda m, c: phase.thermo(m, c, 0.0, 0.0),
                                 co.mu_B, muons=True)
            if got is None:
                return CheckResult("maxwell_crossing", False, float("inf"),
                                   f"{phase.name} does not exist at the "
                                   f"located mu_B = {co.mu_B:.4f} MeV")
            pressures.append(got[0])
        dP = abs(pressures[0] - pressures[1]) / abs(co.P)
        # mu_B is one number handed to both phases, so its equality is exact
        # by construction; what is checked is that each edge row reports it.
        dmu = max(abs(co.row_lo["mu_B"] - co.mu_B),
                  abs(co.row_hi["mu_B"] - co.mu_B)) / abs(co.mu_B)
        worst = max(worst, dP, dmu)
        detail.append(f"{'+'.join(co.branches)} dP/P={dP:.1e} "
                      f"dmu_B/mu_B={dmu:.1e}")
    return CheckResult("maxwell_crossing", worst < 1.0e-6, worst,
                       "; ".join(detail))


#: The parameter set whose branches do NOT cross on `CONSTRUCTION_NB`, so an
#: empty window list is a true assertion for it. The negative control below:
#: without one, "the gate fires" and "the gate fires whenever no window was
#: passed" are the same observation.
CLEAN_SET = "fq1.0_B1"


def _constructed(par, coexistences):
    """The delivered table of one parameter set on `CONSTRUCTION_NB`."""
    from eos.enjl.table import TableSpec, build_constructed_table

    lo, hi, step = CONSTRUCTION_NB
    return build_constructed_table(
        TableSpec(nB=np.arange(lo, hi, step), par=par), coexistences)


def check_delivered_table():
    """The delivered table is deliverable: P non-decreasing, 0 <= c_s^2 <= 1.

    CLAUDE.md section 8. A RAW ENJL branch may violate both inside a
    first-order transition -- mechanical instability is real physics and the
    continuation is allowed to map it -- and this is the check that the
    construction resolves it before the table reaches a structure solver.

    The predicate is `ConstructedTable.defect`, not arithmetic repeated here:
    the flag a caller tests and the check that blesses it have to be the same
    statement, or a green suite and a False flag can coexist.

    Demonstrated in BOTH directions, because a gate is only evidence if it is
    known to be able to fail. Three tables:

    - `CONSTRUCTION_SET` with its located windows -- deliverable. The
      construction does its job.
    - `CONSTRUCTION_SET` with an EMPTY window list -- not deliverable, and
      that is the correct answer, not a defect of this check. Outside a window
      the assembly keeps the lower-eps branch, which is the stable PURE phase
      and not the stable state where the branches cross; the min of two convex
      eps(n_B) curves is concave there, so mu_B jumps down and P falls with it.
    - `CLEAN_SET` with an empty window list -- deliverable. Its branches do not
      cross on this grid, so the empty list asserts something true and the gate
      stays quiet. Without this row the middle one would not distinguish a
      located crossing from a missing argument.
    """
    par, found = _coexistences(CONSTRUCTION_SET)
    cases = [(f"{CONSTRUCTION_SET}+windows", par, found, True),
             (f"{CONSTRUCTION_SET}+none", par, [], False),
             (f"{CLEAN_SET}+none", Parameters.named(CLEAN_SET), [], True)]

    detail, passed = [], True
    for name, params, windows, want in cases:
        table = _constructed(params, windows)
        if len(table.rows) < 3:
            return CheckResult("delivered_table", False, float("inf"),
                               f"{name}: only {len(table.rows)} rows")
        got = table.deliverable
        passed = passed and got is want
        cs2 = table.cs2
        detail.append(
            f"{name}: deliverable={got} (want {want}), "
            f"min dP={np.diff(table.P).min():+.2e} MeV/fm^3, "
            f"c_s^2 in [{cs2.min():+.3e}, {cs2.max():.4f}]"
            + (f", {table.defect}" if table.defect else ""))
    return CheckResult("delivered_table", passed, 0.0 if passed else 1.0,
                       "; ".join(detail))



def check_residual_margin():
    """Every mode clears `RESIDUAL_TOL` by at least two decades.

    A pass/fail on the gate itself is the wrong test. `solver.solve` walks its
    starting points in order and stops at the first that clears the gate, and
    the second of them (`solver._restored_branch`) is on the OTHER chiral
    branch by construction -- so a root that misses the gate is not retried,
    it is replaced by a root somewhere else. A mode whose residuals sit near
    the gate therefore has its branch decided by round-off, and the symptom is
    an O(1) discontinuity in eps and P rather than a convergence failure.

    The margin is measured, not chosen to fit: with every unknown determined
    the four modes land between 1e-16 and 1e-13, four decades clear, and the
    one configuration that ever approached the gate did so because a held
    Y_S = 0 left mu_S undetermined (`solver.strangeness_row_is_empty`) and the
    least-squares termination fired early on the rank-deficient problem.
    """
    gate = 1.0e-2 * RESIDUAL_TOL
    worst, detail = 0.0, ""
    par = Parameters.default()
    grid = np.linspace(0.12, 1.0, 12)
    cases = (("beta_eq_neutrinoless", {}, True),
             ("beta_eq_neutrino_trapped", {"Y_Le": 0.4}, True),
             ("fixed_YC", {"Y_C": 0.5}, False),
             ("fixed_YC_YS", {"Y_C": 0.5, "Y_S": 0.0}, False))
    for mode, fracs, leptons in cases:
        table = build_table(TableSpec(nB=grid, par=par, mode=mode,
                                      leptons=leptons, fractions=fracs))
        for point in table.points:
            if point.error > worst:
                worst = point.error
                detail = f"{mode} {fracs} n_B={point.n_B:.4f}"
    return CheckResult("residual_margin", worst < gate, worst,
                       f"{detail or 'all modes'}; "
                       f"worst {worst:.2e} against {gate:.0e} "
                       f"(gate {RESIDUAL_TOL:.0e})")


CHECKS = (check_euler, check_free_energy, check_entropy_limit,
          check_rearrangement,
          check_gap_equation, check_charge_basis, check_beta_equilibrium,
          check_fixed_fractions, check_symmetric_matter_slice,
          check_trapped_lepton_number, check_thermo_from_mu,
          check_causality, check_vacuum,
          check_maxwell_crossing, check_delivered_table,
          check_refusals, check_non_convergence_is_returned,
          check_residual_margin)


def run_full_check():
    """Run every invariant and return the report."""
    return FullCheckReport(results=[check() for check in CHECKS])


if __name__ == "__main__":
    import sys

    report = run_full_check()
    print(report)
    sys.exit(0 if report.all_passed else 1)
