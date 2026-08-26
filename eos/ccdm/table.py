"""Table driver for CCDM: a `TableSpec`, and `build_table` that solves it.

A table is a set of lines -- one per temperature (or entropy per baryon) and
per combination of the fractions the mode fixes -- each swept along the baryon
density with a warm start. That loop is `eos.general.tabulate`, shared with
every other model; what this module supplies is the CCDM-specific part: which
solve a mode name means, what a solved point carries into the next warm start,
and how a point flattens into a row.

    spec = TableSpec(Parameters.default(), "beta_eq_neutrinoless",
                     axes={"nB": np.linspace(0.8, 2.0, 40), "T": [0.0, 30.0]},
                     include=SpeciesFlags(csc=True))
    result = build_table(spec, verbose=True)

The sweep bisects a missed step back towards the last solved point, which is
what carries it across the thresholds this model has along a cold density
axis: the strange quark's onset, the pairing onset where the winning pattern
changes, and -- the one no other quark model here has -- the chiral/dielectric
branch change, where the fields move discontinuously.

A warm start is keyed by the CANDIDATE, the (branch, pattern) pair (see
`eos.ccdm.solver.warm_start`), so a line carries the seed of whichever
candidate last won and lets the others start cold. That is deliberate: a
candidate seeded only from itself would never be displaced by a rival, and the
point of the enumeration is that it can be.

WHERE THE DENSITY AXIS SHOULD START. Below the deconfinement onset the
deconfined branch is at negative pressure and, further down, has no root at
fixed density at all -- the quarks are simply not there. That is physics, not
a solver limitation, and a table asked for it reports the points as missing
rather than inventing them. A hybrid equation of state gets its low-density
half from a hadronic model through `eos.mixed`, which is what the transition
observables are for.

Progress reporting is the shared callback of CLAUDE.md section 5 -- one
dictionary shape for every model, invoked once per completed line -- with two
keys added, `branch` and `pattern`, because which phase a line ended in is the
thing a reader of this table wants to know first.
"""
from dataclasses import dataclass, field

import numpy as np

from eos.general.modes import resolve_leptons
from eos.ccdm.parameters import Parameters
from eos.ccdm.solver import MODE_FRACTIONS, solve, warm_start
from eos.ccdm.species import SpeciesFlags
from eos.general.tabulate import (
    TEMPERATURE_AXES, lines_from_axes, print_progress, sweep_lines,
    temperature_at_entropy,
)

#: How many times a missed density step may be halved back towards the last
#: solved point. Three thresholds earn it here: the strange quark's onset, the
#: pairing onset, and the chiral/dielectric branch change.
MAX_BISECT = 6


def solve_at(par, mode, n_B, conditions, flags, leptons=True, x0=None):
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
            point = solve(mode, n_B, T, par, flags, x0, leptons=leptons,
                          **fractions)
            return point.s_total / point.n_B if point.n_B else 0.0
        T = temperature_at_entropy(entropy_at, conditions["SnB"])
    else:
        T = conditions["T"]
    return solve(mode, n_B, T, par, flags, x0, leptons=leptons, **fractions)


def quark_row(point):
    """Flatten one solved point into a table row.

    Keyed the way `eos.njl.table.quark_row` and `eos.alphabag.table` key
    theirs, so a quark table and a hadronic one concatenate without renaming:
    chi = 1 and phase = 'Q' say the matter is entirely deconfined. NOTE that
    `chi` here is the MIXED-PHASE quark volume fraction, the shared column
    name, and is not this model's dielectric function -- that one is
    `chi_diel`, and the collision is why it carries a suffix.

    The columns no other quark table has are results rather than inputs: the
    branch and pattern that won, the four fields, the three gaps, the two
    colour potentials the phase had to carry to be colour neutral, and the two
    flags that say when a comparison or a cutoff has stopped meaning what it
    usually means.
    """
    n_B = point.n_B
    row = dict(n_B=n_B, T=point.T, chi=1.0, phase="Q",
               P=point.P_total, eps=point.e_total, s=point.s_total,
               S_per_B=(point.s_total / n_B if n_B else 0.0),
               mu_B=point.mu_B, mu_C=point.mu_C, mu_S=point.mu_S,
               mu_e=point.mu_e,
               Y_C=point.Y_C, Y_S=point.Y_S,
               Y_u=point.Y_u, Y_d=point.Y_d, Y_s=point.Y_s, Y_e=point.Y_e,
               M_u=point.M_star[0], M_d=point.M_star[1], M_s=point.M_star[2],
               branch=point.branch, pattern=point.pattern,
               gapless=point.gapless, beyond_cutoff=point.beyond_cutoff,
               phi_bar=point.phi_bar, chi_diel=point.chi,
               sigma=point.sigma, zeta=point.zeta, omega_0=point.omega_0,
               Sigma_R=point.Sigma_R,
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
    """
    par: Parameters = field(default_factory=Parameters.default)
    mode: str = "beta_eq_neutrinoless"
    axes: dict = field(default_factory=dict)
    include: SpeciesFlags = field(default_factory=SpeciesFlags)
    fixed: dict = field(default_factory=dict)
    leptons: bool = True

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
        self.leptons = resolve_leptons(self.mode, self.leptons, default=True)


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
        return [np.array([p.P_total for p in line]) for line in self.points]

    @property
    def eps(self):
        return [np.array([p.e_total for p in line]) for line in self.points]

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
    plus the `branch` and `pattern` the line ended in.
    """
    axes = {k: v for k, v in spec.axes.items() if k != "nB"}
    lines = lines_from_axes(axes, fixed=spec.fixed)

    # The last converged point of the line in progress, so the per-line report
    # can name the phase the line ended in. sweep_lines hands the callback the
    # line's bookkeeping, not its points.
    last = {}

    def solve_one(n_B, conditions, x0):
        point = solve_at(spec.par, spec.mode, n_B, conditions, spec.include,
                         leptons=spec.leptons, x0=x0)
        if point is not None and point.converged:
            last["point"] = point
        return point

    def report(info):
        point = last.pop("point", None)
        info = dict(info, branch=point.branch if point else None,
                    pattern=point.pattern if point else None)
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
