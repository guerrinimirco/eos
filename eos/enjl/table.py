"""The grid driver: a warm-started continuation along the density axis.

A table of this model is a CONTINUATION, not a phase diagram. Each point is
warm-started from its neighbour, so the sequence follows one branch of the
model and keeps following it past any first-order transition, into the
metastable region beyond. That is deliberate. Mapping a branch and choosing
between branches are separate steps, and the second one needs both branches at
once -- a Maxwell construction equates P and the Gibbs free energy per baryon
g = (eps + P)/n_B across them, which no single sweep can do. In beta
equilibrium with neutrality g reduces to mu_B, which is why the located
transitions are indexed by mu_B; at a held (Y_C, Y_S) it does not, and
g = mu_B + Y_C mu_C + Y_S mu_S is the quantity that must match. It is also
what the author's own tables contain: two of them
retain a step with dP/dn_B < 0 rather than the coexistence plateau that would
replace it.

`direction` therefore selects the branch rather than merely the order of the
loop. Where only one branch exists the two agree; where several do, they
differ, and the difference IS the branch structure.

The loop, the skipping, the timing and the progress reporting are
`eos.general.tabulate`, as in every model; what this module supplies is how
one point is solved, what part of a solved point seeds the next, and the
direction.
"""
from dataclasses import dataclass, field, replace

import numpy as np

from eos.enjl.parameters import Parameters
from eos.enjl.solver import (
    BetaPoint, check_mode, check_temperature, solve, solve_at_entropy,
    warm_start,
)
from eos.enjl.species import SpeciesFlags
from eos.general.tabulate import lines_from_axes, sweep_lines

DIRECTIONS = ("up", "down")


@dataclass
class TableSpec:
    """What to solve: the mode, its fractions, the density axis, the branch.

    `T` is ONE temperature, the line this table is solved along; give `SnB`
    instead to solve an isentrope, where the temperature is found per density
    by an outer 1-D solve (`solver.solve_at_entropy`). Exactly one of the two
    is given. `fractions` carries the mode's own conditions under CLAUDE.md
    section 5's names -- Y_C, Y_S, Y_Le -- and is empty for
    beta_eq_neutrinoless.

    `species` is the SpeciesFlags; only `photons` and `thermal_neutrinos` are
    the caller's, and neither enters an equation.
    """
    nB: np.ndarray
    mode: str = "beta_eq_neutrinoless"
    par: Parameters = field(default_factory=Parameters.default)
    direction: str = "up"
    T: float = 0.0
    SnB: float = None
    x0: list = None
    leptons: bool = None
    species: SpeciesFlags = None
    fractions: dict = field(default_factory=dict)


@dataclass
class TableResult:
    """A solved ENJL grid: the conditions of each line, and its points.

    There is exactly one line: a table here is a density CONTINUATION, and
    that is what carries the sweep, so a second temperature or a second
    fraction is a second call rather than a second column. `points` is shorter
    than `nB` wherever a density could not be reached from its neighbour.
    """
    par: Parameters
    mode: str
    direction: str
    nB: np.ndarray
    lines: list
    points: list
    fractions: dict = field(default_factory=dict)

    @property
    def P(self):
        """Pressure along the solved densities [MeV/fm^3]."""
        return np.array([p.P for p in self.points])

    @property
    def eps(self):
        """Energy density along the solved densities [MeV/fm^3]."""
        return np.array([p.eps for p in self.points])

    @property
    def nB_solved(self):
        """The densities that were actually reached [fm^-3]."""
        return np.array([p.n_B for p in self.points])


def build_table(spec, progress=None, verbose=False):
    """Solve one branch of the model along `spec.nB`.

    Cold starts are allowed only until the branch is established; after that
    the sweep continues from its own previous point or not at all, so the
    result is one branch rather than a sequence that changes branch wherever a
    cold start happens to converge somewhere else. A density that cannot be
    reached from its neighbour is left out, and the sweep carries on from the
    last one that was -- which is why the shared driver is asked NOT to reset
    the warm start on a failure.

    progress : callable, invoked once per completed line -- there is one --
        with the dict every table builder in this repository reports:
        {mode, line, n_lines, temp_key, temp, fracs, n_solved, n_requested,
        elapsed_s}. verbose=True installs the built-in one-line printer.

    Returns:
        TableResult, its `points` ordered by ascending density whichever
        direction was swept.
    """
    check_mode(spec.mode)
    check_temperature(spec.T)
    if spec.SnB is not None and spec.T:
        raise ValueError("give exactly one of T / SnB")
    if spec.direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, "
                         f"got {spec.direction!r}")

    grid = np.atleast_1d(np.asarray(spec.nB, dtype=float))
    order = grid if spec.direction == "up" else grid[::-1]

    def solve_one(n_B, conditions, x0):
        seed = spec.x0 if x0 is None and spec.x0 is not None else x0
        common = dict(x0=seed, cold_start=seed is None,
                      leptons=spec.leptons, species=spec.species)
        if spec.SnB is None:
            return solve(spec.par, spec.mode, n_B, T=spec.T,
                         **common, **spec.fractions)
        # An isentrope: the temperature is a different number at every
        # density, so it is solved for at every density.
        return solve_at_entropy(spec.par, spec.mode, n_B, spec.SnB,
                                **common, **spec.fractions)

    axis = {"T": [spec.T]} if spec.SnB is None else {"SnB": [spec.SnB]}
    lines = lines_from_axes(axis, fixed=spec.fractions)
    solved = sweep_lines(lines, order, solve_one, warm_start=warm_start,
                         progress=progress, verbose=verbose, mode=spec.mode,
                         reset_on_failure=False)
    points = solved[0]
    if spec.direction == "down":
        points = points[::-1]
    return TableResult(par=spec.par, mode=spec.mode, direction=spec.direction,
                       nB=grid, lines=lines, points=points,
                       fractions=dict(spec.fractions))


def beta_row(point):
    """One solved point as a flat table row, fm-based.

    The keys a structure solver and the plotting code read, plus the
    composition and the constituent masses, which are what this model is about.
    `chi` is the fraction of the baryon density carried by deconfined quarks --
    the `fq` column of the author's tables -- and it plays the part the quark
    volume fraction plays in a two-engine mixed phase, so a reader of both
    finds it under the same name.
    """
    state = point.point
    n = point.densities
    return dict(n_B=point.n_B, T=point.T,
                P=point.P, eps=point.eps, s=point.s,
                S_per_B=point.s / point.n_B if point.n_B > 0 else 0.0,
                chi=state.n_bQ / state.n_b if state.n_b > 0 else 0.0,
                mu_B=point.mu_b, mu_C=point.mu_C, mu_S=point.mu_S,
                mu_e=point.mu_e,
                Y_C=state.n_C / state.n_b if state.n_b > 0 else 0.0,
                Y_S=state.n_S / state.n_b if state.n_b > 0 else 0.0,
                n_p=n["p"], n_n=n["n"], n_Lambda=n["Lambda"],
                n_u=n["u"], n_d=n["d"], n_s=n["s"],
                n_e=n["e"], n_mu=n["mu"],
                M_u=state.M_q["u"], M_d=state.M_q["d"], M_s=state.M_q["s"],
                M_p=state.M_b["p"], M_n=state.M_b["n"],
                M_Lambda=state.M_b["Lambda"])


# --------------------------------------------------------------------------
# The construction: from branch continuations to ONE delivered table
# --------------------------------------------------------------------------
#
# `build_table` above maps a BRANCH and keeps following it past a first-order
# transition, into the metastable region beyond. What a structure solver needs
# is the other object: the stable EoS, with each transition replaced by the
# construction across it. That is what this section assembles.
#
# Where the transitions ARE is not computed here. Locating one needs both
# branches at once and the eta = 1 lepton bookkeeping, which lives in
# `eos.mixed` -- a composite engine, downstream of this model (CLAUDE.md
# section 1), so a model may not import it. The located windows therefore
# arrive as an ARGUMENT, `eos.mixed.construction.enjl_coexistences` produces
# them, and the assembly below is pure ENJL.

#: Tolerance on the two conditions of CLAUDE.md section 8, `ConstructedTable
#: .defect` below. A delivered table is exactly flat in P across every
#: constructed window, so a centred difference of it returns zero up to
#: rounding; this is the size of that differencing noise and not a relaxed
#: physical bound.
DELIVERY_TOL = 1.0e-9

#: Row keys that are densities, and so volume-average across a plateau.
_LEVER_KEYS = ("eps", "s", "n_p", "n_n", "n_Lambda", "n_u", "n_d", "n_s",
               "n_e", "n_mu")

#: Row keys that belong to ONE phase and have no value in a two-phase mixture:
#: each phase carries its own, and averaging them would invent a state that is
#: nowhere in the mixture. Reported as nan on plateau rows. mu_e and mu_C are
#: here because at eta = 1 the two phases neutralize separately and their
#: electron potentials genuinely differ -- that difference IS eta = 1.
_UNDEFINED_ON_PLATEAU = ("mu_C", "mu_e", "M_u", "M_d", "M_s",
                         "M_p", "M_n", "M_Lambda")


def plateau_row(co, n_B):
    """One row inside a coexistence window: the constant-pressure segment.

    The lever rule. `phase_fraction` is the volume fraction of the
    high-density phase, fixed by n_B = (1 - f) n_lo + f n_hi, and every
    density averages with the same weights. P is the coexistence value,
    uniform across the window -- that is what makes the segment flat.

    **mu_B and mu_S are written only where the two edges agree on them.** The
    quantity coexistence equates is the Gibbs free energy per baryon,
    g = mu_B + Y_C mu_C + Y_S mu_S, and mu_B is that only where the two
    fraction terms drop out -- in beta equilibrium with neutrality, where
    mu_S = 0 and the mu_C term cancels against the leptons. Under the
    composition closure each phase carries its own (mu_C, mu_S) and its own
    mu_B, so neither is a property of the mixture, and both join the potentials
    already reported as nan below. Averaging them would invent a state that is
    nowhere in the mixture.

    Equality is tested exactly, and that is not fragile: under the beta closure
    mu_B is ONE number handed to both phases, so it is a bitwise identity;
    under the composition closure the two are independent unknowns, and where
    they do coincide -- a symmetric composition puts mu_C = mu_S = 0 on both
    sides, which is g = mu_B again -- the value written is true. Both outcomes
    of the test are true statements about the mixture.

    `co` is anything carrying the fields of
    `eos.mixed.construction.Coexistence`; this module does not import it.
    """
    frac = (n_B - co.n_B_lo) / (co.n_B_hi - co.n_B_lo)
    lo, hi = co.row_lo, co.row_hi
    shared = lambda a, b: a if a == b else float("nan")
    row = {"n_B": float(n_B), "T": lo["T"], "P": co.P,
           "mu_B": shared(co.mu_B_lo, co.mu_B_hi),
           "mu_S": shared(lo["mu_S"], hi["mu_S"]),
           "phase_fraction": float(frac),
           "branch": f"{co.branches[0]}+{co.branches[1]}"}
    for key in _LEVER_KEYS:
        row[key] = (1.0 - frac) * lo[key] + frac * hi[key]
    # s/n_B is a ratio of two volume averages, not an average of two ratios:
    # each edge's S/n_B is relative to its OWN n_B, as its Y are.
    row["S_per_B"] = row["s"] / n_B if n_B > 0 else 0.0
    # The fractions follow from the averaged densities, not from averaging the
    # fractions: both edges' Y are relative to their OWN n_B, not to this one.
    for key, edge in (("chi", "chi"), ("Y_C", "Y_C"), ("Y_S", "Y_S")):
        row[key] = ((1.0 - frac) * lo[edge] * co.n_B_lo
                    + frac * hi[edge] * co.n_B_hi) / n_B
    for key in _UNDEFINED_ON_PLATEAU:
        row[key] = float("nan")
    return row


@dataclass
class ConstructedTable:
    """The DELIVERED ENJL EoS: rows, plus the windows that were constructed.

    A mixed table is "rows + windows" and the windows are part of the result,
    not a by-product (CLAUDE.md section 5). `rows` is in ascending density and
    may be shorter than `nB` where no branch reached a density.

    Every row carries `branch`: on a pure row the CONTINUATION that supplied
    it, "up" or "down", and on a constructed segment the two branch labels
    joined, "<lo>+<hi>". A pure row does not name a branch of the model
    because naming one means reading the state (`enjl_branch_of`), and that
    vocabulary lives with the composite engine, not here. `phase_fraction` is
    nan on a pure row and the volume fraction of the high-density phase on a
    constructed one.
    """
    par: Parameters
    nB: np.ndarray
    rows: list
    windows: list
    eta: float = 1.0
    T: float = 0.0

    @property
    def P(self):
        """Pressure along the delivered densities [MeV/fm^3]."""
        return np.array([r["P"] for r in self.rows])

    @property
    def eps(self):
        """Energy density along the delivered densities [MeV/fm^3]."""
        return np.array([r["eps"] for r in self.rows])

    @property
    def nB_solved(self):
        """The densities actually delivered [fm^-3]."""
        return np.array([r["n_B"] for r in self.rows])

    @property
    def cs2(self):
        """dP/deps along the delivered table, by centred differences.

        The plateau is flat in P and rising in eps, so c_s^2 = 0 there
        exactly; a construction is what makes the delivered table causal
        (CLAUDE.md section 8) and this is the quantity that says so.
        """
        P, eps = self.P, self.eps
        if len(P) < 2:
            return np.zeros_like(P)
        return np.gradient(P, eps)

    @property
    def defect(self):
        """Why this table may not be delivered to a structure solver, or None.

        CLAUDE.md section 8: a table DELIVERED to a structure solver has P
        non-decreasing in n_B and 0 <= c_s^2 <= 1. A raw branch may violate
        both inside a first-order transition -- mechanical instability is real
        physics and `build_table` is allowed to map it -- and this reports on
        the CONSTRUCTED table, where a construction was supposed to resolve
        that. The gate belongs to whoever builds the table, and this module
        builds this one.

        The usual cause is a window that was never located. Outside a window
        the assembly keeps whichever branch has the lower energy density,
        which selects the stable PURE phase -- and that is the stable state
        only where the branches do not cross. Where they do, the min of two
        convex eps(n_B) curves is CONCAVE at the crossing: mu_B = deps/dn_B
        jumps down across it and P = mu_B n_B - eps falls with it. So an
        unlocated window does not merely leave a segment unconstructed, it
        leaves a pressure drop the assembly manufactured. Passing the located
        windows removes it.
        """
        P = self.P
        if len(P) < 2:
            return None
        n_B = self.nB_solved
        dP = np.diff(P)
        i = int(np.argmin(dP))
        if dP[i] < -DELIVERY_TOL:
            return (f"P falls by {-dP[i]:.3f} MeV/fm^3 between "
                    f"n_B = {n_B[i]:.4f} and {n_B[i + 1]:.4f} fm^-3")
        cs2 = self.cs2
        i = int(np.argmin(cs2))
        if cs2[i] < -DELIVERY_TOL:
            return f"c_s^2 = {cs2[i]:+.3e} at n_B = {n_B[i]:.4f} fm^-3"
        i = int(np.argmax(cs2))
        if cs2[i] > 1.0:
            return f"c_s^2 = {cs2[i]:.4f} > 1 at n_B = {n_B[i]:.4f} fm^-3"
        return None

    @property
    def deliverable(self):
        """True where `defect` is None: this table may go to a structure solver."""
        return self.defect is None


def build_constructed_table(spec, coexistences, eta=1.0, progress=None,
                            verbose=False):
    """The stable EoS: raw branches outside the windows, plateaus across them.

    Two continuations are swept, "up" from the low-density chirally broken
    side and "down" from the top of the grid, so both branches of every
    transition are in hand. Outside a window the delivered point is whichever
    of them has the LOWER energy density at that n_B, and inside a window it
    is the plateau of `plateau_row`.

    At T = 0 the free energy IS eps and the two roots carry identical
    conserved charges, so the lower-eps root is the stable one -- in beta
    equilibrium with neutrality and equally at a held (Y_C, Y_S), which is why
    this needs no branch bookkeeping. What it does not buy is a deliverable
    table: min-eps selects the stable PURE phase, and that is the stable state
    only OUTSIDE a coexistence window. Inside one the stable state is a
    mixture, min-eps manufactures a drop in P at the crossing, and the result
    reports `deliverable = False`. See `ConstructedTable.defect`.

    Parameters
    ----------
    spec : TableSpec
        As `build_table` takes it. `spec.direction` is ignored: a construction
        needs both directions and sweeps both.
    coexistences : list
        The located windows, from `eos.mixed.construction.enjl_coexistences`.
        An ARGUMENT because a model may not import a composite engine
        (CLAUDE.md section 1). An empty list is legal and asserts that this
        grid holds no transition: where that is true the result is the stable
        branch with no constructed segment in it, and where it is false the
        result reports `deliverable = False` rather than pretending. Test it.
    eta : float
        The construction. Only eta = 1, the Maxwell construction with each
        phase separately neutral, is implemented; anything else raises. An
        eta < 1 delivered table needs the mixed system solved at every density
        inside the window, seeded from the eta = 1 point located here.
    progress : callable
        Invoked once for the single line, with the dictionary every table
        builder in this repository reports, plus `eta` and `windows` as the
        mixed builder adds them. verbose=True installs the built-in printer.

    Returns:
        ConstructedTable.
    """
    import time

    check_temperature(spec.T)
    if spec.T != 0.0 or spec.SnB:
        raise NotImplementedError(
            f"eos.enjl builds a CONSTRUCTED table at T = 0 only; got "
            f"T = {spec.T} MeV, SnB = {spec.SnB}. The raw continuation "
            f"`build_table` is closed at any temperature -- what is not is "
            f"the construction across a window: locating a coexistence at "
            f"T > 0 means equating the Gibbs free energies rather than "
            f"P and mu_B alone, so the entropy enters the coexistence "
            f"bookkeeping, and the plateau's lever rule then averages it "
            f"too. `eos.mixed.construction` raises for the same reason")
    if eta != 1.0:
        raise NotImplementedError(
            f"eos.enjl delivers the eta = 1 (Maxwell) construction; got "
            f"eta = {eta}. An eta < 1 table needs the mixed system solved at "
            f"every density inside the window, seeded from the eta = 1 point "
            f"-- see eos.mixed.solver and docs/DEFERRED.md")
    if verbose and progress is None:
        from eos.general.tabulate import print_progress
        progress = print_progress

    t0 = time.time()
    grid = np.sort(np.atleast_1d(np.asarray(spec.nB, dtype=float)))
    windows = sorted(coexistences, key=lambda w: w.n_B_lo)

    # `BetaPoint.n_B` is the density the point was ASKED at, passed
    # through unchanged, so the two sweeps and the grid key alike exactly.
    #
    # The downward sweep is SEEDED FROM THE TOP OF THE UPWARD ONE rather than
    # cold-started, and that is not an optimization. This model's cold starts
    # stop converging around 0.5 fm^-3, so a downward continuation left to
    # find its own first point fails at every density above that and only
    # catches on once it has walked down to where a cold start works -- at
    # which point it is retracing the upward sweep's branch instead of the
    # high-density one, and the pair of sweeps holds one branch rather than
    # two. Measured on f_q = 0.7, B = 1 over 0.30-0.78 fm^-3: unseeded, the
    # two sweeps agree to the last digit up to 0.54 and the downward one has
    # nothing above it. The upward sweep's last point is on the high-density
    # branch (a continuation changes branch where its own ends), so it is what
    # puts the downward sweep there.
    branches = {}
    up = build_table(replace(spec, direction="up"))
    sweeps = [("up", up)]
    if up.points:
        sweeps.append(("down", build_table(
            replace(spec, direction="down", x0=warm_start(up.points[-1])))))
    for direction, result in sweeps:
        for point in result.points:
            row = beta_row(point)
            row["branch"] = direction
            row["phase_fraction"] = float("nan")
            branches.setdefault(round(point.n_B, 10), []).append(row)

    rows = []
    for n_B in grid:
        window = next((w for w in windows if w.contains(n_B)), None)
        if window is not None:
            rows.append(plateau_row(window, float(n_B)))
            continue
        candidates = branches.get(round(float(n_B), 10), ())
        if not candidates:
            continue
        rows.append(min(candidates, key=lambda r: r["eps"]))

    if progress is not None:
        progress({"mode": spec.mode, "line": 1, "n_lines": 1,
                  "temp_key": "T", "temp": spec.T, "fracs": dict(spec.fractions),
                  "n_solved": len(rows), "n_requested": len(grid),
                  "elapsed_s": time.time() - t0,
                  "eta": eta, "windows": list(windows)})
    return ConstructedTable(par=spec.par, nB=grid, rows=rows,
                            windows=list(windows), eta=eta, T=spec.T)


__all__ = ["BetaPoint", "ConstructedTable", "DELIVERY_TOL", "DIRECTIONS",
           "TableSpec", "TableResult", "beta_row",
           "build_constructed_table", "build_table", "plateau_row"]
