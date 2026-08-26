"""Physics invariants of `eos.general`, checked in one place.

`general/` is not a model, but it is the single home of the Fermi and Bose
integrals (CLAUDE.md section 7), the conserved-charge basis maps (section 2)
and the thermal meson gas — the pieces every model's correctness rests on, and
the ones a wrong result is hardest to trace back to. This suite checks those
shared pieces against each other, which is the only way they CAN be checked:
there is no model above them to disagree with.

  1. Fermi integrals   JEL against the alternatives section 7 requires be
                       validated against it — the scipy quadrature and the
                       Gauss-Laguerre rule.
  2. Bose integrals    the same, for the boson family.
  3. Basis maps        every map in `eos.general.basis` against the species
                       quantum numbers of `eos.general.particles`, over the
                       WHOLE table rather than a handful of named species.
  4. Meson gas         the thermal nonet's local (Q, S) table against the same
                       particle table, and the gas's per-species Euler
                       relation.
  5. T = 0 limits      the closed forms against the finite-T integrals as
                       T -> 0, including the Sommerfeld T^2 approach that
                       distinguishes a correct limit from a wrong constant.

WHAT IS DELIBERATELY NOT HERE. The split-panel Gauss-Legendre gas is the third
alternative section 7 speaks of, and its validation against JEL already exists
as `test/general/test_fermi_gauss.py` — repeating it here would be the
duplication section 12 warns against, not a second opinion. Nor does this
suite owe section 8's delivery gate: `general/` has no `table.py` and hands no
table to a structure solver, so P-monotonicity and causality are somebody
else's invariant.

Run it:

    python -m eos.general.verify.run_full_check
"""
from dataclasses import dataclass, field

import numpy as np

from eos.general.basis import (
    QUARK_FLAVOURS, charges_of, charges_from_densities, quark_charges,
    species_potential, quark_potentials, charge_potentials_from_quarks,
    baryon_potentials, projection_residual, undetermined_potential,
)
from eos.general.bose_integrals import (
    solve_bose_jel, solve_bose_gl, Bose_Numerical,
)
from eos.general.fermi_integrals import (
    solve_fermi_jel, solve_fermi_gl, solve_fermi_t0, Fermi_Numerical,
)
from eos.general.solve import undetermined_unknowns
from eos.general.particles import (
    BARYONS_ALL, LEPTONS, MESONS_PSEUDOSCALAR, QUARKS, Neutrino, get_particle,
)
from eos.general.thermal_mesons import meson_families, thermal_meson_thermo


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
        lines = [f"general run_full_check: "
                 f"{'PASS' if self.all_passed else 'FAIL'}"]
        for r in self.results:
            tag = "ok " if r.passed else "FAIL"
            lines.append(f"  [{tag}] {r.name:24s} max_err={r.max_error:.2e}"
                         f"  {r.detail}")
        return "\n".join(lines)


# =============================================================================
# THE INTEGRALS
# =============================================================================
#: (m, mu, T, g) in MeV, spanning the corners the models actually solve in:
#: cold and warm nucleons, leptons, and light quarks at merger temperatures.
FERMI_POINTS = [
    (939.0, 1000.0, 5.0, 2.0),
    (939.0, 1000.0, 30.0, 2.0),
    (939.0, 1200.0, 50.0, 2.0),
    (939.0, 960.0, 1.0, 2.0),
    (0.511, 120.0, 10.0, 2.0),
    (105.66, 200.0, 30.0, 2.0),
    (5.0, 400.0, 20.0, 6.0),
    (100.0, 500.0, 20.0, 6.0),
    (100.0, 500.0, 0.5, 6.0),
    (300.0, 450.0, 30.0, 6.0),
    (140.0, 700.0, 50.0, 6.0),
]

#: (m, mu, T, g) for the boson family: the thermal nonet at merger
#: temperatures, at both signs of the effective potential and up against the
#: condensation edge (mu -> m at the last point).
BOSE_POINTS = [
    (139.57039, 100.0, 30.0, 1.0),
    (139.57039, 0.0, 50.0, 1.0),
    (139.57039, -100.0, 20.0, 1.0),
    (493.677, 200.0, 50.0, 1.0),
    (497.611, 0.0, 80.0, 1.0),
    (547.862, 0.0, 60.0, 1.0),
    (775.26, 0.0, 100.0, 3.0),
    (139.57039, 130.0, 10.0, 1.0),
]

#: JEL is a rational approximation, quoted at ~1e-4 (Johns, Ellis & Lattimer,
#: ApJ 473, 1020). An alternative agreeing with it to 2e-3 over the grids
#: above is the SAME integral evaluated a different way; the measured worst is
#: reported in every detail string, so a drift toward this bound is visible
#: long before it crosses.
INTEGRAL_TOL = 2.0e-3

#: Below this degeneracy parameter, T / (mu - m), a 30-node Gauss-Laguerre
#: rule cannot resolve the Fermi step and `solve_fermi_gl` is not usable: at
#: T / (mu - m) = 0.001 it returns a density three orders of magnitude wrong.
#: Its own T < 1e-4 analytic fallback sits far below the breakdown, so the
#: window between is real. That is a property of the rule, not of JEL, so the
#: comparison is made where the rule applies and the boundary is named here
#: rather than left implicit in a hand-picked grid.
GL_MIN_DEGENERACY = 0.1


def _scaled_errors(candidate, reference):
    """|candidate - reference| for (n, P, eps, s, n_s), each over its own scale.

    Every quantity but one is measured against ITSELF, which is what keeps the
    check sharp: normalising the pressure by the energy density instead would
    divide a degenerate gas's error by P / eps ~ 0.1 and let a half-percent
    error in P through a 2e-3 tolerance. The scalar density is the exception
    and is measured against `n`, because n_s / n -> 0 for a nearly massless
    species: at m = 0.511 MeV a relative error in n_s reaches 4 percent while
    the absolute quantity is negligible beside the density, and normalising by
    itself would turn a physically empty corner into the loudest number in the
    suite.
    """
    candidate = np.asarray(candidate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    n, P, eps, s, n_s = reference
    scales = np.array([abs(n), abs(P), abs(eps),
                       abs(s) if s != 0.0 else 1.0,
                       max(abs(n_s), abs(n))])
    return np.abs(candidate - reference) / np.maximum(scales, 1e-30)


def check_fermi_alternatives(tol=INTEGRAL_TOL):
    """JEL against the scipy quadrature and the Gauss-Laguerre rule.

    Section 7's rule stated as a check: JEL is the validated implementation
    and the alternatives are supplements, each validated against it. Two
    alternatives, two domains — `Fermi_Numerical` integrates the distribution
    directly and is exact everywhere on the grid, `solve_fermi_gl` only where
    the gas is not strongly degenerate (see `GL_MIN_DEGENERACY`).
    """
    worst_quad = 0.0
    worst_gl = 0.0
    n_gl = 0
    for m, mu, T, g in FERMI_POINTS:
        jel = solve_fermi_jel(mu, T, m, g)
        worst_quad = max(worst_quad,
                         _scaled_errors(jel, Fermi_Numerical(mu, T, m, g)).max())
        if T / (mu - m) >= GL_MIN_DEGENERACY:
            n_gl += 1
            worst_gl = max(worst_gl,
                           _scaled_errors(jel, solve_fermi_gl(mu, T, m, g)).max())
    worst = float(max(worst_quad, worst_gl))
    return CheckResult(
        "Fermi: JEL vs alts", worst < tol, worst,
        f"scipy {worst_quad:.1e} over {len(FERMI_POINTS)} points, "
        f"Gauss-Laguerre {worst_gl:.1e} over {n_gl}")


def check_bose_alternatives(tol=INTEGRAL_TOL):
    """The same for the boson family, which nothing else in the repository
    cross-validates — and which the thermal meson gas is built on."""
    worst_quad = 0.0
    worst_gl = 0.0
    for m, mu, T, g in BOSE_POINTS:
        jel = solve_bose_jel(mu, T, m, g)
        worst_quad = max(worst_quad,
                         _scaled_errors(jel, Bose_Numerical(mu, T, m, g)).max())
        worst_gl = max(worst_gl,
                       _scaled_errors(jel, solve_bose_gl(mu, T, m, g)).max())
    worst = float(max(worst_quad, worst_gl))
    return CheckResult(
        "Bose: JEL vs alts", worst < tol, worst,
        f"scipy {worst_quad:.1e}, Gauss-Laguerre {worst_gl:.1e}, over "
        f"{len(BOSE_POINTS)} points")


# =============================================================================
# THE BASIS MAPS
# =============================================================================
def check_basis_against_species_table(tol=1.0e-13):
    """Every map in `basis` against the quantum numbers it claims to be built
    from, over the whole particle table.

    Four statements, and the point of sweeping the table rather than naming
    species is that a single wrong row cannot hide behind the ones somebody
    thought to test:

      * Gell-Mann-Nishijima, Q = I_3 + (B - S)/2, in THIS repository's
        convention S = +1 per s quark. Under the PDG sign the same table would
        fail on every strange species, which is what makes the sweep the
        cheapest test of the convention.
      * C = Q for hadrons and quarks, C = 0 for leptons (section 2: C is the
        charge of strongly-interacting matter only).
      * mu_i = B_i mu_B + C_i mu_C + S_i mu_S, evaluated against the row's own
        quantum numbers, at potentials chosen so that no two terms can cancel.
      * the density sums n_B, n_C, n_S from the same rows, and the quark
        potential map inverted.
    """
    mu_B, mu_C, mu_S = 1000.0, -70.0, 130.0
    errors = []
    failures = []

    hadrons = BARYONS_ALL + MESONS_PSEUDOSCALAR + list(QUARK_FLAVOURS)
    for p in hadrons:
        Q_gmn = p.isospin_3 + (p.baryon_no - p.strangeness) / 2.0
        errors.append(abs(Q_gmn - p.charge))
        errors.append(abs(p.strong_charge - p.charge))
        B, C, S = charges_of(p)
        if (B, C, S) != (p.baryon_no, p.charge, p.strangeness):
            failures.append(f"charges_of({p.name}) disagrees with the table")
        errors.append(abs(species_potential(p, mu_B, mu_C, mu_S)
                          - (B * mu_B + C * mu_C + S * mu_S)))
        # The name lookup must reach the same row as the object.
        if charges_of(p.name) != (B, C, S):
            failures.append(f"{p.name} by name and by object disagree")

    for lepton in LEPTONS + [Neutrino]:
        if lepton.strong_charge != 0.0:
            failures.append(f"{lepton.name} carries strong charge")

    # Densities: the sum over the table, against the map.
    densities = {p.name: 0.05 + 0.01 * i for i, p in enumerate(BARYONS_ALL)}
    densities["e-"] = 0.03          # a lepton, which must not enter any sum
    n_B = sum(get_particle(k).baryon_no * v for k, v in densities.items())
    n_C = sum(get_particle(k).strong_charge * v for k, v in densities.items()
              if not get_particle(k).is_lepton)
    n_S = sum(get_particle(k).strangeness * v for k, v in densities.items())
    got = charges_from_densities(densities)
    errors += [abs(a - b) for a, b in zip(got, (n_B, n_C, n_S))]

    # Quarks: the flavour sums, and the potential map both ways.
    n_u, n_d, n_s = 0.4, 0.5, 0.3
    from_flavours = quark_charges(n_u, n_d, n_s)
    by_hand = ((n_u + n_d + n_s) / 3.0,
               (2.0 * n_u - n_d - n_s) / 3.0,
               n_s)
    errors += [abs(a - b) for a, b in zip(from_flavours, by_hand)]
    round_trip = charge_potentials_from_quarks(
        *quark_potentials(mu_B, mu_C, mu_S))
    errors += [abs(a - b) / abs(b) for a, b in
               zip(round_trip, (mu_B, mu_C, mu_S))]

    # The octet map is the same projection, species by species.
    for name, mu_i in baryon_potentials(mu_B, mu_C, mu_S).items():
        errors.append(abs(mu_i - species_potential(name, mu_B, mu_C, mu_S)))

    worst = float(max(errors)) / max(abs(mu_B), 1.0)
    return CheckResult(
        "basis vs species table", worst <= tol and not failures, worst,
        "; ".join(failures) if failures else
        f"{len(hadrons)} hadrons and quarks, {len(LEPTONS) + 1} leptons")


def check_meson_gas(tol=1.0e-10):
    """The thermal nonet's local (Q, S) table against `particles`, and the
    gas's own Euler relation.

    `thermal_mesons` carries charge and strangeness inline beside each species
    rather than looking them up, because it also lists the vector nonet, which
    the particle table does not have. The pseudoscalar half IS in the table,
    and it is the half where the sign convention bites: K+ (u sbar) carries
    S = -1 and K- (ubar s) carries S = +1, the opposite of the PDG sign. A
    flipped kaon row would move strangeness in every fixed-Y_S solve that
    enables the gas and would not show up in P or eps at all — so it is
    checked against the one table section 2 declares.

    The second half is the identity a Bose gas satisfies species by species,
    eps + P = T s + sum_j mu*_j n_j, evaluated on the full nonet.
    """
    errors = []
    failures = []
    for name, _mu_eff, _m, Q, S, _g in meson_families(
            0.0, 0.0, 0.0, include_pseudoscalars=True,
            include_thermal_vectors=True):
        p = get_particle(name)
        if p is None:
            continue                      # the vector nonet has no table row
        if (Q, S) != (p.charge, p.strangeness):
            failures.append(f"{name}: gas says (Q={Q}, S={S}), table says "
                            f"(Q={p.charge:.0f}, S={p.strangeness:.0f})")

    T = 40.0
    gas = thermal_meson_thermo(60.0, 90.0, 30.0, T,
                               include_pseudoscalars=True,
                               include_thermal_vectors=True)
    euler = gas["e"] + gas["P"] - T * gas["s"] - gas["mu_dot_n"]
    errors.append(abs(euler) / abs(gas["e"]))

    worst = float(max(errors))
    return CheckResult(
        "thermal meson gas", worst <= tol and not failures, worst,
        "; ".join(failures) if failures else
        f"nonet quantum numbers, Euler at T = {T:g} MeV")


# =============================================================================
# THE T = 0 LIMITS
# =============================================================================
#: (m, mu, g): degenerate points for the T -> 0 approach.
T0_POINTS = [
    (939.0, 1000.0, 2.0),
    (100.0, 500.0, 6.0),
    (300.0, 450.0, 6.0),
    (5.0, 400.0, 6.0),
]

#: The temperatures the approach is measured at, halving.
T0_LADDER = (0.4, 0.2, 0.1)


def check_t0_limit(tol=1.0e-4):
    """The T = 0 closed forms against the finite-T integrals as T -> 0.

    `solve_fermi_t0` is a separate implementation — five closed forms, not a
    limit taken numerically — so nothing but this comparison ties it to the
    finite-T integral it is supposed to be the limit of.

    TWO STATEMENTS, AND THE SECOND IS THE ONE THAT BITES. The first is that
    the difference is small at the lowest temperature. The second is that it
    falls like T^2, the Sommerfeld expansion's leading correction: halving T
    must quarter the error. A closed form carrying a wrong CONSTANT — a
    dropped log term, a factor in the pressure — passes the first statement at
    a loose enough tolerance and fails the second outright, because its error
    stops falling. The exact quadrature is used as the finite-T side rather
    than JEL, whose own ~1e-4 approximation error would itself floor the
    ladder at the third rung and hide exactly that signature.
    """
    worst = 0.0
    worst_ratio_error = 0.0
    for m, mu, g in T0_POINTS:
        cold = np.asarray(solve_fermi_t0(mu, m, g), dtype=float)
        ladder = []
        for T in T0_LADDER:
            hot = np.asarray(Fermi_Numerical(mu, T, m, g), dtype=float)
            # The entropy is not part of the comparison: it vanishes at T = 0
            # and is linear in T, so it carries no information about the
            # closed forms and would fail a T^2 ratio it never obeyed.
            err = _scaled_errors(hot, cold)
            ladder.append(max(err[0], err[1], err[2], err[4]))
        worst = max(worst, float(ladder[-1]))
        for coarse, fine in zip(ladder, ladder[1:]):
            # T halves, so a T^2 error quarters; 3 to 5 leaves room for the
            # quadrature's own noise without admitting a constant offset,
            # whose ratio is 1.
            ratio = coarse / fine if fine > 0.0 else float("inf")
            if not 3.0 <= ratio <= 5.0:
                worst_ratio_error = max(worst_ratio_error, abs(ratio - 4.0))
    passed = worst < tol and worst_ratio_error == 0.0
    return CheckResult(
        "T -> 0 limit", passed, worst,
        f"{len(T0_POINTS)} points, T = {T0_LADDER[-1]} MeV; the T^2 approach "
        f"{'holds' if worst_ratio_error == 0.0 else 'FAILS'}")


def check_undetermined_potential_screen(tol=1.0e-12):
    """The two screens of `basis` that read a SOLVED STATE, both directions.

    A conserved-charge potential that no populated species carries is not
    pinned by the equations: its residual row reads 0 = 0, the solver stops
    wherever its path ran out, and the number it reports is round-off. That is
    legitimate physics -- a locked phase has no free charge fraction, and
    strangeness at Y_S = 0 with no strange species has nothing to fix mu_S --
    but it is indistinguishable from a regression unless something can name
    it, and every instance so far cost a session of hand analysis.

    `basis.projection_residual` is the per-point half: mu_i must equal its own
    projection B_i mu_B + C_i mu_C + S_i mu_S. It catches a species potential
    carried as an independent unknown, at one point, with no second run.

    `basis.undetermined_potential` is the differential half, and it exists
    because the first CANNOT see this: an undetermined potential satisfies the
    projection at every point and still lands elsewhere on the next run. Given
    what each species moved between two runs of one case, it asks whether the
    shifts are one common delta times each species' own coefficient. The
    fingerprint is exact -- Xi carries twice what Lambda carries because
    S_Xi = 2 S_Lambda -- and it is READABLE only because the potentials are
    derived by projection (CLAUDE.md section 2); a model carrying ad-hoc
    species potentials would show the same failure as unstructured drift.

    `solve.undetermined_unknowns` is the third limb, and the only one that
    fires before the damage is done. An undetermined potential is a
    CONDITIONING hazard, not only a reporting one: carried as an unknown with
    no equation it is a null column in the Jacobian, the least-squares
    termination fires early on the rank-deficient problem, and the residual of
    the whole solve sits decades above what the model's other modes reach --
    close enough to the gate for round-off to decide which side a point lands
    on. So this one reads the column directly, rather than waiting for the
    symptom.

    All three are checked in both directions, because a screen that cannot
    fail is not a screen: a consistent state, a consistent drift and a
    full-rank Jacobian must pass, while a single perturbed species, a wrong
    ratio, a species carrying none of the charge that moved anyway, and a
    column fourteen decades down must each be caught.
    """
    mu_B, mu_C, mu_S = 1000.0, -70.0, 130.0
    strange = [p for p in BARYONS_ALL if p.strangeness]
    errors = []
    failures = []

    # The identity, over the whole table rather than a chosen few.
    state = {p.name: species_potential(p, mu_B, mu_C, mu_S)
             for p in BARYONS_ALL}
    worst, carrier = projection_residual(state, mu_B, mu_C, mu_S)
    errors.append(worst / mu_B)
    if carrier is not None:
        failures.append(f"a consistent state named {carrier}")

    # ... and it must SEE a species carried as its own unknown.
    broken = dict(state)
    broken["Lambda"] += 1.0e-6
    seen, named = projection_residual(broken, mu_B, mu_C, mu_S)
    if named != "Lambda" or seen < 5.0e-7:
        failures.append(f"a 1e-6 MeV perturbation of Lambda was read as "
                        f"{seen:.3e} on {named!r}")

    # The differential: one undetermined mu_S, seen through S_i.
    delta = 0.37
    drift = {p.name: delta * p.strangeness for p in BARYONS_ALL}
    got, reason = undetermined_potential(drift, "S")
    if got is None:
        failures.append(f"an exact mu_S drift was rejected: {reason}")
    else:
        errors.append(abs(got - delta) / delta)

    # ... and the two ways it must fail. A wrong ratio is a physics change
    # wearing the pattern; a species with S_i = 0 that moved is one too, and
    # it is the case the ratio test alone cannot see.
    wrong_ratio = dict(drift)
    wrong_ratio[strange[0].name] *= 1.5
    if undetermined_potential(wrong_ratio, "S")[0] is not None:
        failures.append(f"a 1.5x wrong ratio on {strange[0].name} passed")
    nucleon_moved = dict(drift)
    nucleon_moved["n"] = 1.0e-9
    if undetermined_potential(nucleon_moved, "S")[0] is not None:
        failures.append("a neutron that moved under a mu_S drift passed")

    # The third limb, and the only one that fires BEFORE the damage: an
    # unknown with no equation is a null column of the Jacobian, and
    # `eos.general.solve.undetermined_unknowns` reads it off directly rather
    # than waiting for the residual to sit decades high.
    slots = ("sigma", "omega", "mu_B", "mu_C", "mu_S")
    well_posed = np.linalg.qr(np.arange(1.0, 26.0).reshape(5, 5)
                              + 5.0 * np.eye(5))[0]
    if undetermined_unknowns(well_posed, slots):
        failures.append("a full-rank Jacobian was called under-determined")
    null_column = well_posed.copy()
    null_column[:, 4] = 0.0
    if undetermined_unknowns(null_column, slots) != ["mu_S"]:
        failures.append("a mu_S column of exact zeros was not caught")
    near_null = well_posed.copy()
    near_null[:, 4] = 1.0e-14 * near_null[:, 0]
    if undetermined_unknowns(near_null, slots) != ["mu_S"]:
        failures.append("a mu_S column 14 decades down was not caught")

    worst = float(max(errors))
    return CheckResult(
        "undetermined-potential screen", worst <= tol and not failures, worst,
        "; ".join(failures) if failures else
        f"{len(BARYONS_ALL)} species, {len(strange)} carrying S; "
        f"all three limbs fail on a broken input")


def run_full_check():
    """Run the `eos.general` verification suite; returns a structured report."""
    report = FullCheckReport()
    report.results.append(check_fermi_alternatives())
    report.results.append(check_bose_alternatives())
    report.results.append(check_basis_against_species_table())
    report.results.append(check_undetermined_potential_screen())
    report.results.append(check_meson_gas())
    report.results.append(check_t0_limit())
    return report


if __name__ == "__main__":
    print(run_full_check())
