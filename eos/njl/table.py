"""Table driver for NJL: a `TableSpec`, and `build_table` that solves it.

A table is a set of lines -- one per temperature (or entropy per baryon) and
per combination of the fractions the mode fixes -- each swept along the baryon
density with a warm start. That loop is `eos.general.tabulate`, shared with
every other model; what this module supplies is the NJL-specific part: which
solve a mode name means, what a solved point carries into the next warm start,
and how a point flattens into a row.

    spec = TableSpec(Parameters.default(), "beta_eq_neutrinoless",
                     axes={"nB": np.linspace(0.4, 2.0, 80), "T": [0.0, 30.0]},
                     include=SpeciesFlags(csc=True))
    result = build_table(spec, verbose=True)

The sweep bisects a missed step back towards the last solved point, which is
what carries it across the two thresholds this model has along a cold density
axis: the strange quark's onset, and the pairing onset where the winning
pattern changes.

A warm start here is keyed by PATTERN (see `eos.njl.solver.warm_start`), so a
line carries one seed per candidate and each pattern continues from its own
previous root. What keeps the enumeration honest is not withholding those
seeds but `pattern_realised`: a candidate that COLLAPSED -- came back in a
state other than the layout it was solved in -- carries no seed forward and
starts cold at the next density, so a rival it fell onto cannot capture it for
the rest of the line.

Progress reporting is the shared callback of CLAUDE.md section 5 -- one
dictionary shape for every model, invoked once per completed line -- with one
key added, `pattern`, because which phase a line ended in is the thing a
reader of a CSC table wants to know first.

A spec that sets `solve_nodes` is answered by `build_fast_table` instead,
which turns the sweep inside out: it solves a few densities per PAIRING
PATTERN and interpolates along each branch, rather than solving every density
and enumerating the patterns at each. The two are the same request at
different accuracies -- same modes, same `TableResult`, same rows, same rule
for which candidate wins -- and which one runs is a property of the spec, so
`build_table` dispatches rather than the caller choosing a function.
"""
import time
from dataclasses import dataclass, field, fields

import numpy as np
from scipy.interpolate import CubicHermiteSpline, CubicSpline

from eos.general.tabulate import (
    TEMPERATURE_AXES, lines_from_axes, print_progress, sweep_lines,
    temperature_at_entropy,
)
from eos.general.modes import resolve_leptons
from eos.njl.parameters import Parameters
from eos.njl.solver import (
    MODE_FRACTIONS, EoSPoint, patterns_for, solve, solve_pattern, warm_start,
)
from eos.njl.species import SpeciesFlags

#: How many times a missed density step may be halved back towards the last
#: solved point. Two thresholds earn it here: the strange quark's onset and
#: the pairing onset.
MAX_BISECT = 6


def solve_at(par, mode, n_B, conditions, flags, leptons=None, x0=None,
             backend="reference", patterns=None, pair_nodes_per_panel=None):
    """One point of a table: the mode's solve at this density and line.

    `conditions` carries the line's temperature (`T`) or entropy per baryon
    (`SnB`) and whichever fractions the mode fixes, under the spec names of
    CLAUDE.md section 5. Non-convergence comes back on the result, not as an
    exception.
    """
    fractions = {k: v for k, v in conditions.items()
                 if k in MODE_FRACTIONS[mode]}
    if "SnB" in conditions:
        def entropy_at(T):
            point = solve(par, mode, n_B, T, flags, x0, leptons=leptons,
                          backend=backend, patterns=patterns,
                          pair_nodes_per_panel=pair_nodes_per_panel,
                          **fractions)
            return point.s / point.n_B if point.n_B else 0.0
        T = temperature_at_entropy(entropy_at, conditions["SnB"])
    else:
        T = conditions["T"]
    return solve(par, mode, n_B, T, flags, x0, leptons=leptons,
                 backend=backend, patterns=patterns,
                 pair_nodes_per_panel=pair_nodes_per_panel, **fractions)


def quark_row(point):
    """Flatten one solved point into a table row.

    Keyed the way `eos.alphabag.table.quark_row` and `eos.vmit.table` key
    theirs, so a quark table and a hadronic one concatenate without renaming:
    chi = 1 and phase = 'Q' say the matter is entirely deconfined. The pairing
    columns are the ones no other quark table has, and they are results rather
    than inputs: which pattern won, the three gaps, the two colour potentials
    the phase had to carry to be colour neutral, and whether the state is
    gapless.
    """
    n_B = point.n_B
    row = dict(n_B=n_B, T=point.T, chi=1.0, phase="Q",
               P=point.P, eps=point.eps, s=point.s,
               S_per_B=(point.s / n_B if n_B else 0.0),
               mu_B=point.mu_B, mu_C=point.mu_C, mu_S=point.mu_S,
               mu_e=point.mu_e,
               Y_C=point.Y_C, Y_S=point.Y_S,
               Y_u=point.Y_u, Y_d=point.Y_d, Y_s=point.Y_s, Y_e=point.Y_e,
               M_u=point.M[0], M_d=point.M[1], M_s=point.M[2],
               pattern=point.pattern,
               pattern_realised=point.pattern_realised, gapless=point.gapless,
               Delta_1=point.Delta[0], Delta_2=point.Delta[1],
               Delta_3=point.Delta[2], mu_3=point.mu_3, mu_8=point.mu_8)
    row["Y_mu-"] = point.n_mu / n_B if n_B else 0.0
    if point.mu_nu:
        row["Y_nue"] = point.Y_nu
        row["mu_nue"] = point.mu_nu
    return row


@dataclass
class TableSpec:
    """One table request.

    axes  : {'nB': grid, exactly one of 'T'/'SnB': grid, and optionally any
            fraction the mode fixes ('Y_C', 'Y_S', 'Y_Le') as a further axis}
    fixed : scalar values for the fractions the mode needs and the axes do not
            sweep
    leptons: for the fixed-fraction modes, whether neutralizing leptons are
            added so the total system is electrically neutral. In the
            beta-equilibrium modes the leptons are constitutive, so True is
            redundant and ignored and False raises -- here rather than inside
            the sweep, where skip_errors would swallow it.
    backend: which flavour of the medium integrals to use, 'reference' or
            'fast' (CLAUDE.md section 9). The default is the reference path,
            so a table is the same table it has always been; 'fast' is the
            jitted kernel of `eos.njl.backends`, which agrees to round-off
            rather than bit for bit.
    patterns: restrict the pairing enumeration to these candidates, exactly as
            `eos_point`'s argument of the same name does; None enumerates the
            default set. ('unpaired', '2SC', 'CFL') is the recommended
            fast restriction: the asymmetric 'free' seed exists to let a
            CFL-layout solve fall to a state that is not CFL, and where no
            such state exists it burns its whole retry ladder discovering so
            -- measured at 90% of one paired point's cost. Restricting is a
            declaration that uSC/dSC-like states are not being hunted, which
            is a physics choice the caller makes explicitly.

            MEASURED, on a 9-point csc=True table over n_B = 1.0 to 1.4 fm^-3
            at T = 0: 63.9 s with the defaults, 7.5 s with backend='fast',
            3.7 s with backend='fast' and this restriction -- 17.5x, agreeing
            with the default table to 1.2e-10 relative in P and 6.2e-12 in
            eps. Neither argument changes the equations; both are declarations
            the caller makes.
    pair_nodes_per_panel: Gauss-Legendre nodes per panel of the PAIRING
            quadrature; None keeps the shipped rule (24). Lowering it is the
            third speed lever and the only one that moves numbers. Measured on
            the same 9-point csc=True table: 11.5 s at 24 nodes, 6.2 s at 16
            and 3.6 s at 12, with P moving by 3e-10 and 4e-10 relative -- both
            above the 1e-10 the `test/baseline` entries are frozen at, and a
            point can miss the convergence gate outright at 16. The default is
            unchanged and this is an argument the caller sets deliberately.
    solve_nodes: solve this many densities PER BRANCH and interpolate the
            rest (`build_fast_table`). None, the default, solves every
            requested density. `beta_eq_neutrinoless` on a temperature axis
            ONLY -- the interpolation rests on an identity the other modes do
            not satisfy, and asking for one of them raises rather than
            returning a table that is wrong by percents (`FAST_MODES`).

            This is the fourth speed lever and the largest one, and like
            `pair_nodes_per_panel` it moves numbers: an interpolated point is
            accurate to the spline, not to the solver's gate. What makes it cheap is that the branches are SMOOTH in mu_B
            while the enumeration over them is not, so the expensive thing is
            done a few times and the smooth thing many.

            MEASURED, on 200-point beta_eq_neutrinoless tables at T = 0 with
            `Parameters.named("rg_njl1")`, eta_D = 1.45, backend='fast' and
            patterns=('2SC', 'CFL'), against the same grid fully solved. The
            shipped defaults cost 6638 ms/point and the pattern restriction
            alone 914; the two densities straddling the branch crossing are
            excluded, since there the two tables differ over WHERE the
            first-order transition sits and not over a branch:

              n_B = 0.50 to 1.55 fm^-3, a hybrid-star quark core
                 8 nodes   18 ms/pt   max deps/eps 2.1e-3   median 2.2e-6
                16 nodes   33 ms/pt   max deps/eps 5.8e-4   median 1.5e-6
                24 nodes   38 ms/pt   max deps/eps 9.2e-5   median 1.5e-6

              n_B = 0.30 to 1.55 fm^-3, down through chiral restoration
                 8 nodes   21 ms/pt   max deps/eps 1.8e-2   median 9.6e-6
                16 nodes   28 ms/pt   max deps/eps 6.1e-3   median 3.0e-6
                24 nodes   37 ms/pt   max deps/eps 1.7e-3   median 2.9e-6

            THE ERROR IS NOT SPREAD EVENLY, which is the number to read here:
            the median is a part in 1e6 either way and the maximum is spent
            almost entirely in one place, the low-density end of the 2SC
            branch where the constituent masses are still falling. A CFL
            branch is smooth enough that ten nodes reproduce it to 3e-6 in P;
            a 2SC branch carried down through chiral restoration is not, and
            wants the node count above. Start the grid above the restoration
            region and eight nodes is plenty.
    """
    par: Parameters = field(default_factory=Parameters.default)
    mode: str = "beta_eq_neutrinoless"
    axes: dict = field(default_factory=dict)
    include: SpeciesFlags = field(default_factory=SpeciesFlags)
    fixed: dict = field(default_factory=dict)
    leptons: bool = None
    backend: str = "reference"
    patterns: tuple = None
    pair_nodes_per_panel: int = None
    #: Solve this many densities PER BRANCH and interpolate the rest -- the
    #: fast table mode of `build_fast_table`. None (the default) solves every
    #: requested density and is the exact path.
    solve_nodes: int = None

    def __post_init__(self):
        if "nB" not in self.axes:
            raise ValueError("TableSpec.axes must contain 'nB'")
        if self.mode not in MODE_FRACTIONS:
            raise ValueError(f"unknown mode {self.mode!r}; expected one of "
                             f"{list(MODE_FRACTIONS)}")
        temp_keys = [k for k in self.axes if k in TEMPERATURE_AXES]
        if len(temp_keys) != 1:
            raise ValueError("TableSpec.axes needs exactly one of 'T' / 'SnB'")
        supplied = set(self.axes) | set(self.fixed)
        for key in MODE_FRACTIONS[self.mode]:
            if key not in supplied:
                raise ValueError(f"mode {self.mode!r} needs {key!r}, as an "
                                 f"axis or in fixed")
        self.leptons = resolve_leptons(self.mode, self.leptons, default=False)
        # Validate a pattern restriction NOW: inside the sweep, skip_errors
        # would swallow the per-point ValueError and return an empty table.
        patterns_for(self.include, self.patterns)


@dataclass
class TableResult:
    spec: TableSpec
    nB: np.ndarray
    #: One entry per line, parallel to `points`: the conditions it was solved
    #: at ({'T' or 'SnB': ..., and whichever fractions the mode fixes}).
    lines: list
    #: points[i_line][i_nB]. With skip_errors a line is shorter than `nB`.
    points: list

    @property
    def P(self):
        return [np.array([p.P for p in line]) for line in self.points]

    @property
    def eps(self):
        return [np.array([p.eps for p in line]) for line in self.points]

    @property
    def nB_solved(self):
        return [np.array([p.n_B for p in line]) for line in self.points]


def build_table(spec, skip_errors=True, rows=False, progress=None,
                verbose=False):
    """Solve a `TableSpec` over the product of its temperature and fraction axes.

    rows=False (default) returns a `TableResult`; rows=True returns the long
    format `eos.general.table_io` writes -- one flat dict per solved point.

    skip_errors=True drops points the solver could not reach instead of
    aborting the table, which is what a parameter scan needs. progress/verbose
    report per line, in the shape every table builder in this repository uses,
    plus the `pattern` the line ended in.

    A spec that sets `solve_nodes` is handed to `build_fast_table`, which
    solves that many densities per BRANCH and interpolates the rest. It is the
    same request answered to a different accuracy, so it is dispatched here
    rather than being a second public entry point.
    """
    if spec.solve_nodes is not None:
        return build_fast_table(spec, skip_errors=skip_errors, rows=rows,
                                progress=progress, verbose=verbose)
    axes = {k: v for k, v in spec.axes.items() if k != "nB"}
    lines = lines_from_axes(axes, fixed=spec.fixed)

    # The last converged point of the line in progress, so the per-line report
    # can name the pattern the line ended in. sweep_lines hands the callback
    # the line's bookkeeping, not its points.
    last = {}

    def solve_one(n_B, conditions, x0):
        point = solve_at(spec.par, spec.mode, n_B, conditions, spec.include,
                         leptons=spec.leptons, x0=x0, backend=spec.backend,
                         patterns=spec.patterns,
                         pair_nodes_per_panel=spec.pair_nodes_per_panel)
        if point is not None and point.converged:
            last["point"] = point
        return point

    def report(info):
        point = last.pop("point", None)
        info = dict(info, pattern=point.pattern if point else None)
        if progress is not None:
            progress(info)
        if verbose:
            print_progress(info)

    points = sweep_lines(lines, spec.axes["nB"], solve_one,
                         warm_start=warm_start, skip_errors=skip_errors,
                         progress=report if (progress or verbose) else None,
                         mode=spec.mode, max_bisect=MAX_BISECT)

    result = TableResult(spec=spec,
                         nB=np.asarray(spec.axes["nB"], dtype=float),
                         lines=lines, points=points)
    return rows_from_result(result) if rows else result


# =============================================================================
# THE FAST TABLE: SOLVE THE BRANCHES, INTERPOLATE ALONG THEM
# =============================================================================
# The enumeration of section 3's patterns is what costs, and it is not smooth:
# which candidate wins changes discontinuously with density. Each candidate
# taken ALONE is smooth, which is the asymmetry `solve_nodes` trades on --
# solve a few points per branch, interpolate along each branch, and do the
# comparison between them on the interpolants.
#
# Measured on `Parameters.named("rg_njl1")` at T = 0, eta_D = 1.45,
# beta_eq_neutrinoless, backend='fast', one pattern at a time and warm
# started: unpaired 6 ms/point, 2SC 63 ms/point, CFL 443 ms/point. The whole
# enumeration through `build_table` is 6638 ms/point. Over n_B = 0.25 to 1.6
# the unpaired branch never wins -- f_2SC - f_unpaired is at most -103
# MeV/fm^3 and falls monotonically -- and f_CFL - f_2SC is monotone with one
# zero, so the branches cross exactly once and the crossing falls out of the
# interpolants rather than being searched for.

#: The pattern whose LAYOUT is deliberately free to collapse: it exists so a
#: CFL-layout solve can fall to a state that is not CFL, which means it names
#: no branch of its own and cannot be interpolated ALONG. It is dropped from
#: the fast driver's default enumeration and raises when asked for by name --
#: the same split `eos.njl.solver.patterns_for` already makes between a
#: candidate that loses and a call that asks for something it cannot have.
NOT_A_BRANCH = "free"

#: The modes `solve_nodes` can answer, and it is exactly one.
#:
#: The interpolation rests on dP/dmu_B = n_B holding ALONG THE SWEEP, which is
#: what lets a branch be a Hermite spline of P(mu_B) whose slope is data. That
#: is not a property of the model but of the MODE, because Gibbs-Duhem at
#: fixed T reads
#:
#:     dP = n_B dmu_B + n_C dmu_C + n_S dmu_S + (lepton terms)
#:
#: and only beta equilibrium kills the rest of it. There mu_S = 0 (strangeness
#: self-equilibrates), the leptons are tied to the matter by mu_C + mu_e = 0,
#: and neutrality makes n_C = n_e, so n_C dmu_C + n_e dmu_e = 0 identically.
#: A fixed-Y_C sweep breaks this even with neutralizing leptons: there mu_C is
#: an independent unknown and mu_C + mu_e = 0 is exactly the condition that
#: does NOT hold, so the two terms no longer cancel. Trapping breaks it too --
#: mu_C + mu_e = mu_nue there -- and fixing Y_S adds a live mu_S term.
#:
#: MEASURED, |dP/dmu_B / n_B - 1| along a 13-point 2SC sweep at T = 0 with
#: `Parameters.named("rg_njl1")`: beta_eq_neutrinoless 4.0e-4, and then
#: fixed_YC 3.0e-2 leptonless and 4.2e-2 with leptons, beta_eq_neutrino_trapped
#: 6.9e-2, fixed_YC_YS 6.0e-2. Two orders between the one that holds and the
#: four that do not, and the four are wrong by percents in P -- which is why
#: this is a raise and not a documented caveat. `solve_nodes=None` solves
#: every mode exactly, as it always has (docs/DEFERRED.md).
FAST_MODES = ("beta_eq_neutrinoless",)


#: How far above the convergence gate a probe may stall and still be taken for
#: a HARD POINT rather than the end of the branch. `eos.general.solve`'s gate
#: is 1e-10, so this is ten thousand gate widths -- deliberately generous,
#: because the two cases are separated by nine orders of magnitude and the
#: cost of being wrong is asymmetric: a branch cut short is a wrong table,
#: while a needless retry is a fraction of a second.
#:
#: MEASURED, walking down a ten-node ladder at T = 0 with
#: `Parameters.named("rg_njl1")`, eta_D = 1.45, backend='fast'. The 2SC branch
#: at n_B = 0.994 fm^-3 is a hard point: its probe stalls at 1.5e-10, one gate
#: width out, with the gaps already right, and the full ladder reaches the
#: root in 224 ms. The CFL branch past its onset at n_B = 0.439 fm^-3 has
#: ENDED: its probe stalls at 2.0e-01, and the full ladder spends 24.2 s
#: discovering that the root it finds is the 2SC one.
BRANCH_END_RESIDUAL = 1.0e-6


def branch_ladder(spec, pattern, nB_nodes, T, fractions):
    """One pattern walked DOWN a coarse density ladder, warm started.

    Downward because every pattern exists at the top of a density range and
    the paired ones end somewhere below: walking down, the seed is always the
    root one step away, and the walk stops where the pattern stops. Walking up
    would start each pattern cold in the region where it does not exist, which
    is the most expensive place this model has.

    Every node after the first is PROBED first -- one damped-Newton run from
    the previous root and no hunt, `eos.njl.solver.solve_pattern`'s
    `rescue=False` -- and the probe's own residual decides what the failure
    meant. That is the whole of what makes this ladder cheap, and it rests on
    the two failures being nine orders of magnitude apart (see
    `BRANCH_END_RESIDUAL`):

      * probe converged, in its own layout -> the node is solved, walk on;
      * probe converged onto ANOTHER layout -- a CFL vector whose two s-quark
        gaps came back zero is the 2SC state -- -> the branch has ended here,
        and no residual will say otherwise;
      * probe stalled NEAR a root -> a hard point, not an end. The full
        ladder runs from the same seed, and only its verdict is final;
      * probe stalled far from one -> the branch has ended.

    Returns the solved points in ASCENDING density, which is the order the
    interpolants below want.
    """
    def attempt(n_B, seed, rescue):
        return solve_pattern(
            spec.par, spec.mode, float(n_B), T, spec.include, pattern,
            x0=seed, leptons=spec.leptons, backend=spec.backend,
            pair_nodes_per_panel=spec.pair_nodes_per_panel,
            rescue=rescue, **fractions)

    points, seed = [], None
    for n_B in sorted(np.asarray(nB_nodes, dtype=float), reverse=True):
        point = attempt(n_B, seed, rescue=(seed is None))
        if not point.converged and point.error <= BRANCH_END_RESIDUAL:
            point = attempt(n_B, seed, rescue=True)
        if not (point.converged and point.pattern_realised == pattern):
            break
        seed = np.array(point.x, dtype=float)
        points.append(point)
    points.reverse()
    return points


#: Fields of an `EoSPoint` that an interpolated point does NOT carry. The
#: three underscored ones are the model's own working record in natural units
#: -- the state, the unknown vector, the per-pattern seeds -- and none of them
#: survives interpolation: there is no solved state between two solved states.
#: Dropping them is what stops a caller re-seeding from a point that was never
#: solved. The rest are strings and flags, carried from the branch instead.
INTERPOLATED_DROP = ("_state", "x", "_seeds")
INTERPOLATED_CARRY = ("mode", "pattern", "pattern_realised", "converged")


#: Relative slack at a branch's ends, so a grid endpoint does not fall off the
#: branch that was solved to reach it (`Branch.covers`). Two orders above
#: `eos.general.solve.RESIDUAL_TOL`, which is the scale of the offset between
#: the density a node was asked for and the one it converged to, and many
#: orders below any grid spacing.
BRANCH_EDGE_SLACK = 1.0e-8


class Branch:
    """One pairing pattern's solved points, as a function of n_B.

    P and n_B come from a cubic HERMITE spline of P(mu_B) whose slope is the
    solved n_B, because dP/dmu_B = n_B at fixed T is the Gibbs-Duhem relation
    and interpolating it as data rather than fitting it independently is what
    makes the interpolated points thermodynamically consistent with each
    other. The density is inverted exactly, as a root of that spline's own
    derivative (`PPoly.solve`), so the node count is the only error in P.

    Every OTHER quantity -- eps, the masses, the gaps, the colour potentials,
    the fractions -- is an ordinary cubic spline in mu_B. So an interpolated
    point satisfies the Euler relation of CLAUDE.md section 8 to the accuracy
    of the interpolation and not to the solver's 1e-8 -- a part in 1e6 in the
    median and a part in 1e3 at worst, with the spread and where it sits
    measured in `TableSpec`. `solve_nodes=None` satisfies it exactly.

    `gapless` is carried from the nearer solved node rather than interpolated:
    it is a statement about a spectrum, and there is no half of one.
    """

    def __init__(self, pattern, points):
        if len(points) < 4:
            raise ValueError(f"a {pattern!r} branch needs at least 4 solved "
                             f"nodes to interpolate, got {len(points)}")
        self.pattern = pattern
        self.points = points
        mu = np.array([p.mu_B for p in points], dtype=float)
        self.P = CubicHermiteSpline(mu, np.array([p.P for p in points]),
                                    np.array([p.n_B for p in points]))
        self.n_B = self.P.derivative()
        self.mu_lo, self.mu_hi = mu[0], mu[-1]
        self.nB_lo, self.nB_hi = points[0].n_B, points[-1].n_B
        self._scalar, self._vector = {}, {}
        for column in fields(EoSPoint):
            name = column.name
            if name in INTERPOLATED_DROP or name in INTERPOLATED_CARRY:
                continue
            values = [getattr(point, name) for point in points]
            if name in ("Delta", "M"):
                self._vector[name] = CubicSpline(
                    mu, np.array([list(v) for v in values], dtype=float))
            elif isinstance(values[0], bool):
                continue
            else:
                self._scalar[name] = CubicSpline(
                    mu, np.array(values, dtype=float))

    def covers(self, n_B):
        """Is this density on the branch, allowing for where its ends ARE?

        A solved point sits at the density it was ASKED for only to the
        solver's own gate: a ladder node requested at 1.55 fm^-3 comes back at
        1.5499999999944. Judged strictly, the last density of a grid then
        falls off the branch that was built to reach it and is answered by
        whichever rival still covers it -- which is how a table's top row
        ends up 5% out while every row below it is exact to 1e-5.
        """
        return (self.nB_lo * (1.0 - BRANCH_EDGE_SLACK) <= n_B
                <= self.nB_hi * (1.0 + BRANCH_EDGE_SLACK))

    def mu_at(self, n_B):
        """The mu_B where this branch reaches density n_B, exactly.

        n_B is monotone in mu_B along a branch, so the derivative spline meets
        the target once inside the range; `extrapolate=False` is what keeps a
        root of a neighbouring polynomial piece from being returned. A target
        inside the slack of `covers` but outside the solved range is answered
        by the end node itself rather than by extrapolating to it.
        """
        n_B = float(n_B)
        if not self.covers(n_B):
            raise ValueError(f"the {self.pattern!r} branch does not reach "
                             f"n_B = {n_B:g} fm^-3")
        if n_B <= self.nB_lo:
            return self.mu_lo
        if n_B >= self.nB_hi:
            return self.mu_hi
        roots = self.n_B.solve(n_B, extrapolate=False)
        inside = [r for r in roots if self.mu_lo <= r <= self.mu_hi]
        if not inside:
            raise ValueError(f"the {self.pattern!r} branch has no mu_B at "
                             f"n_B = {n_B:g} fm^-3; is it monotone?")
        return float(inside[0])

    def at(self, n_B):
        """The interpolated `EoSPoint` at this density."""
        mu = self.mu_at(n_B)
        near = min(self.points, key=lambda p: abs(p.mu_B - mu))
        values = {name: float(spline(mu))
                  for name, spline in self._scalar.items()}
        for name, spline in self._vector.items():
            values[name] = tuple(float(v) for v in spline(mu))
        values.update(n_B=float(n_B), mu_B=mu, P=float(self.P(mu)),
                      gapless=near.gapless,
                      error=max(p.error for p in self.points))
        for name in INTERPOLATED_CARRY:
            values[name] = getattr(near, name)
        return EoSPoint(**values)


def fast_line(spec, conditions, nB_out, patterns, skip_errors=True):
    """One line of a fast table: the branches, then the winner at each density.

    The winner is the converged candidate of lowest f = eps - T s, which is
    the rule `eos.njl.solver.solve` applies at a solved point and is applied
    here to the interpolants instead. It is deliberately the SAME rule: this
    is a speed lever, not a different construction, and a fast table and an
    exact one are two accuracies of one answer rather than two answers. A
    first-order transition therefore still shows up as the branch swap it is,
    with the mechanical instability across it intact and left for the
    construction of CLAUDE.md section 8 to resolve.

    A pattern that reaches fewer than four of the ladder's nodes IS NOT
    REPORTED: a cubic spline cannot be built through three points, so there is
    no branch to compare. That is the one thing this mode can miss that the
    exact path does not -- a phase occupying a sliver of the range narrower
    than the node spacing -- and raising `solve_nodes` is what resolves it,
    which is why the error below names the node count rather than the physics.
    """
    if "SnB" in conditions:
        raise NotImplementedError(
            "eos.njl: solve_nodes is wired for a temperature axis only. An "
            "entropy-per-baryon axis puts an outer solve for T around every "
            "node, which is a second continuation the ladder does not carry -- "
            "and the identity of `FAST_MODES` holds at fixed T, not at fixed "
            "S/n_B. Use axes={'T': ...} or solve_nodes=None "
            "(docs/DEFERRED.md)")
    T = conditions["T"]
    fractions = {k: v for k, v in conditions.items()
                 if k in MODE_FRACTIONS[spec.mode]}
    nodes = np.linspace(float(nB_out[0]), float(nB_out[-1]), spec.solve_nodes)

    branches = []
    for pattern in patterns:
        points = branch_ladder(spec, pattern, nodes, T, fractions)
        if len(points) >= 4:
            branches.append(Branch(pattern, points))
    if not branches:
        if skip_errors:
            return []
        raise RuntimeError(
            f"eos.njl: no pattern of {tuple(patterns)!r} held a branch of at "
            f"least 4 nodes over n_B = {nB_out[0]:g} to {nB_out[-1]:g} fm^-3 "
            f"at T = {T:g} MeV; raise solve_nodes or narrow the density range")

    line = []
    for n_B in np.asarray(nB_out, dtype=float):
        candidates = [b.at(n_B) for b in branches if b.covers(n_B)]
        if candidates:
            line.append(min(candidates, key=lambda p: p.f))
        elif not skip_errors:
            raise RuntimeError(f"eos.njl: no branch covers n_B = {n_B:g} "
                               f"fm^-3 at T = {T:g} MeV")
    return line


def build_fast_table(spec, skip_errors=True, rows=False, progress=None,
                     verbose=False):
    """`build_table` for a spec that sets `solve_nodes`: solve the branches.

    Same signature, same `TableResult`, same rows. What differs is that most
    of the returned points were interpolated along a branch rather than
    solved, so they carry no `_state` and no unknown vector and cannot seed
    anything; `result.spec.solve_nodes` is what says a table is one of these.

    The progress dictionary is section 5's, with `pattern` as `build_table`
    reports it and one key added: `n_nodes`, the number of densities actually
    solved on the line, which is the whole of what this mode is about.
    """
    if spec.mode not in FAST_MODES:
        raise NotImplementedError(
            f"eos.njl: solve_nodes cannot answer mode {spec.mode!r}. The "
            f"branch interpolation needs dP/dmu_B = n_B along the sweep, and "
            f"of this model's modes only {FAST_MODES[0]!r} satisfies it -- "
            f"see `FAST_MODES` for why, and docs/DEFERRED.md. Use "
            f"solve_nodes=None, which solves every mode exactly")
    patterns = patterns_for(spec.include, spec.patterns)
    if spec.patterns is None:
        patterns = tuple(p for p in patterns if p != NOT_A_BRANCH)
    elif NOT_A_BRANCH in patterns:
        raise ValueError(
            f"pattern {NOT_A_BRANCH!r} names no branch: its layout is free to "
            f"collapse, which is what it is FOR, so there is nothing to "
            f"interpolate along. Drop it from `patterns`, or leave "
            f"`solve_nodes` unset to enumerate it point by point")

    axes = {k: v for k, v in spec.axes.items() if k != "nB"}
    lines = lines_from_axes(axes, fixed=spec.fixed)
    nB_out = np.asarray(spec.axes["nB"], dtype=float)

    points = []
    for index, conditions in enumerate(lines, start=1):
        started = time.perf_counter()
        line = fast_line(spec, conditions, nB_out, patterns,
                         skip_errors=skip_errors)
        points.append(line)
        if progress is None and not verbose:
            continue
        temp_key = "SnB" if "SnB" in conditions else "T"
        info = dict(mode=spec.mode, line=index, n_lines=len(lines),
                    temp_key=temp_key, temp=conditions[temp_key],
                    fracs={k: v for k, v in conditions.items()
                           if k != temp_key},
                    n_solved=len(line), n_requested=len(nB_out),
                    elapsed_s=time.perf_counter() - started,
                    pattern=(line[-1].pattern if line else None),
                    n_nodes=spec.solve_nodes)
        if progress is not None:
            progress(info)
        if verbose:
            print_progress(info)

    result = TableResult(spec=spec, nB=nB_out, lines=lines, points=points)
    return rows_from_result(result) if rows else result


def rows_from_result(result):
    """A solved `TableResult` as long-format rows.

    Separate from `build_table` so a table already in hand can be written out
    without being solved a second time.
    """
    out = []
    for conditions, line in zip(result.lines, result.points):
        for point in line:
            row = quark_row(point)
            for key, value in conditions.items():
                row.setdefault(key, value)
            out.append(row)
    return out
