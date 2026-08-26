"""Table driver for DID: a `TableSpec`, and `build_table` that solves it.

A table is a set of lines -- one per temperature (or entropy per baryon) and
per combination of the fractions the mode fixes -- each swept along the baryon
density with a warm start. That loop is not written here: it is
`eos.general.tabulate`, shared with every other model. What this module
supplies is the DID-specific part, which is three things: which solve a mode
name means, what a solved point carries into the next warm start, and how a
point flattens into a table row.

    spec = TableSpec(Parameters.default(), "beta_eq_neutrinoless",
                     axes={"nB": np.linspace(0.05, 1.2, 120), "T": [0.0, 30.0]},
                     include=SpeciesFlags(hyperons=True))
    result = build_table(spec, verbose=True)

The density sweep bisects a missed step back towards the last solved point,
which is what carries it through the hyperon onsets: at T = 0 a threshold
inside one grid interval leaves the previous point's answer outside the new
basin, and halving the interval gives the continuation a seed on the far side.

Progress reporting is the shared callback of CLAUDE.md section 5 -- one
dictionary shape for every model, invoked once per completed line.
"""
from dataclasses import dataclass, field

import numpy as np

from eos.general.modes import beta_eq_neutrinoless
from eos.general.state import EOSTable_for_TOV
from eos.general.tabulate import TEMPERATURE_AXES, lines_from_axes, sweep_lines
from eos.did.parameters import Parameters
from eos.did.solver import (
    EoSPoint, MODE_FRACTIONS, mode_spec, solve_beta_eq_neutrinoless,
    solve_mode, warm_start,
)
from eos.did.species import SpeciesFlags

#: How many times a missed density step may be halved back towards the last
#: solved point. DID crosses hyperon thresholds along a cold sweep, so the
#: bisection earns its keep here (`eos.general.tabulate` defaults it to 0 for
#: models with nothing to walk through).
MAX_BISECT = 6


def solve_at(par, mode, n_B, conditions, flags, leptons=None, x0=None):
    """One point of a table: the mode's solve at this density and line.

    `conditions` carries the line's temperature (`T`) or entropy per baryon
    (`SnB`) and whichever fractions the mode fixes, under the spec names of
    CLAUDE.md section 5. Non-convergence comes back on the result, not as an
    exception.
    """
    spec = mode_spec(mode, dict(conditions, leptons=leptons))
    if "SnB" in conditions:
        return solve_mode(par, n_B, flags, spec, SnB=conditions["SnB"], x0=x0)
    return solve_mode(par, n_B, flags, spec, T=conditions["T"], x0=x0)


def hadronic_row(point: EoSPoint):
    """Flatten one solved point into a table row.

    Keyed the way `eos.dd2.table.hadronic_row` and `eos.mixed` key theirs, so
    a purely hadronic table and a hybrid table concatenate without renaming:
    chi = 0 and phase = 'H' say the matter is entirely hadronic. Y_C and Y_S
    are the TOTAL non-leptonic fractions, thermal meson gas included, which is
    what the fixed-fraction conditions are stated in terms of.
    """
    n_B = point.n_B
    row = dict(n_B=n_B, T=point.T, chi=0.0, phase="H",
               P=point.P, eps=point.eps, s=point.s,
               S_per_B=point.entropy_per_baryon,
               mu_B=point.mu_B, mu_C=point.mu_C, mu_S=point.mu_S,
               mu_e=point.mu_e, mu_nue=point.mu_nue,
               Y_C=point.Y_C, Y_S=point.Y_S,
               Y_e=point.Y_e, beta=point.beta,
               sigma=point.sigma, omega=point.omega, rho=point.rho,
               phi=point.phi, Sigma_r=point.Sigma_r, Sigma_t=point.Sigma_t)
    row["Y_mu-"] = point.n_mu / n_B if n_B else 0.0
    for name, n in point.composition_map.items():
        row[f"Y_{name}"] = n / n_B if n_B else 0.0
    if point.n_nu:
        row["Y_nue"] = point.n_nu / n_B
    return row


@dataclass
class TableSpec:
    """One table request.

    axes  : {'nB': grid, exactly one of 'T'/'SnB': grid, and optionally any
            fraction the mode fixes ('Y_C', 'Y_S', 'Y_Le') as a further axis}
    fixed : scalar values for the fractions the mode needs and the axes do not
            sweep
    leptons: for the fixed-fraction modes, whether neutralizing leptons are
            added so the total system is electrically neutral. None leaves it
            at DID's leptonless default; in the beta-equilibrium modes the
            leptons are constitutive, so True is redundant and ignored and
            False raises.
    """
    par: Parameters = field(default_factory=Parameters.default)
    mode: str = "beta_eq_neutrinoless"
    axes: dict = field(default_factory=dict)
    include: SpeciesFlags = field(default_factory=SpeciesFlags)
    fixed: dict = field(default_factory=dict)
    leptons: bool = None

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


@dataclass
class TableResult:
    spec: TableSpec
    nB: np.ndarray
    #: One entry per line, parallel to `points`: the conditions it was solved
    #: at ({'T' or 'SnB': ..., and whichever fractions the mode fixes}).
    lines: list
    #: points[i_line][i_nB]. With skip_errors a line is shorter than `nB`.
    points: list


def build_table(spec, skip_errors=True, rows=False, progress=None,
                verbose=False):
    """Solve a `TableSpec` over the product of its temperature and fraction axes.

    rows=False (default) returns a `TableResult`; rows=True returns the long
    format `eos.general.table_io` writes -- one flat dict per solved point.

    skip_errors=True drops points the solver could not reach instead of
    aborting the table, which is what a parameter scan needs: a grid always
    has corners where uniform matter has no solution (the liquid-gas spinodal
    at low T and low density, above all). progress/verbose report per line, in
    the shape every table builder in this repository uses.
    """
    axes = {k: v for k, v in spec.axes.items() if k != "nB"}
    lines = lines_from_axes(axes, fixed=spec.fixed)
    isentropic = "SnB" in axes

    def solve(n_B, conditions, x0):
        return solve_at(spec.par, spec.mode, n_B, conditions, spec.include,
                        leptons=spec.leptons, x0=x0)

    # Which unknowns the vector carries depends on the MODE, not on the
    # values it holds fixed, so one declaration serves every line.
    layout = mode_spec(spec.mode, dict(lines[0], leptons=spec.leptons))

    def seed(point):
        return warm_start(point, layout, isentropic=isentropic)

    points = sweep_lines(lines, spec.axes["nB"], solve, warm_start=seed,
                         skip_errors=skip_errors, progress=progress,
                         verbose=verbose, mode=spec.mode,
                         max_bisect=MAX_BISECT)
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
            row = hadronic_row(point)
            # The line's conditions are recorded, but never on top of a
            # quantity the row already carries: Y_C in the row is the charge
            # the solved state turned out to have, which is what a table is
            # for, not the value that was asked for.
            for key, value in conditions.items():
                row.setdefault(key, value)
            out.append(row)
    return out


# =============================================================================
# THE CORE TABLE A STRUCTURE SOLVER INTEGRATES
# =============================================================================
# `EOSTable_for_TOV` is the contract between a model and `eos.astro` and lives
# in `eos.general.state`, which both layers may import (CLAUDE.md section 1);
# building one is the model's side of it. Running the sequence over it is
# astro's, and lives in `test/did/did_tov_sequence.py`.

#: Crust-core transition density [fm^-3] (the BPS table tops out at 0.08).
N_TRANSITION = 0.08


def build_core_table(par, flags, n_lo=0.05, n_hi=1.4, n_points=200):
    """The cold beta-equilibrium core EoS as an `EOSTable_for_TOV`.

    A geometric density grid swept with a warm start, which is what carries
    the solve through the hyperon onsets; a density the solver cannot reach is
    dropped rather than ending the sweep.
    """
    spec = beta_eq_neutrinoless()
    P, eps, n_B, x0 = [], [], [], None
    for n in np.geomspace(n_lo, n_hi, n_points):
        point = solve_beta_eq_neutrinoless(par, float(n), flags, T=0.0, x0=x0)
        if not point.converged:
            x0 = None
            continue
        P.append(point.P)
        eps.append(point.eps)
        n_B.append(point.n_B)
        x0 = warm_start(point, spec)
    P, eps, n_B = np.array(P), np.array(eps), np.array(n_B)
    order = np.argsort(P)          # TOV interpolation needs P increasing
    return EOSTable_for_TOV(P=P[order], epsilon=eps[order], nB=n_B[order])
