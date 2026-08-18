"""The grid driver: a warm-started continuation along the density axis.

A table of this model is a CONTINUATION, not a phase diagram. Each point is
warm-started from its neighbour, so the sequence follows one branch of the
model and keeps following it past any first-order transition, into the
metastable region beyond. That is deliberate. Mapping a branch and choosing
between branches are separate steps, and the second one needs both branches at
once -- a Maxwell construction equates P and mu_B across them, which no single
sweep can do. It is also what the author's own tables contain: two of them
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
from dataclasses import dataclass, field

import numpy as np

from eos.enjl.parameters import Parameters
from eos.enjl.solver import (
    BetaPoint, check_mode, check_temperature, solve_beta_eq_neutrinoless,
    warm_start,
)
from eos.general.tabulate import lines_from_axes, sweep_lines

DIRECTIONS = ("up", "down")


@dataclass
class TableSpec:
    """What to solve: the mode, the density axis, and which branch to follow.

    The temperature axis is accepted for the shape every model's table spec
    has, and must be [0.0]: this is a T = 0 model.
    """
    nB: np.ndarray
    mode: str = "beta_eq_neutrinoless"
    par: Parameters = field(default_factory=Parameters.default)
    direction: str = "up"
    T: float = 0.0
    x0: list = None


@dataclass
class TableResult:
    """A solved ENJL grid: the conditions of each line, and its points.

    There is exactly one line, T = 0, because the model has one temperature
    and the mode it closes fixes no fraction to sweep. `points` is shorter
    than `nB` wherever a density could not be reached from its neighbour.
    """
    par: Parameters
    mode: str
    direction: str
    nB: np.ndarray
    lines: list
    points: list

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
        return np.array([p.n_b_fm for p in self.points])


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
    if spec.direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, "
                         f"got {spec.direction!r}")

    grid = np.atleast_1d(np.asarray(spec.nB, dtype=float))
    order = grid if spec.direction == "up" else grid[::-1]

    def solve(n_B, conditions, x0):
        seed = spec.x0 if x0 is None and spec.x0 is not None else x0
        return solve_beta_eq_neutrinoless(n_B, par=spec.par, x0=seed,
                                          cold_start=seed is None)

    lines = lines_from_axes({"T": [spec.T]})
    solved = sweep_lines(lines, order, solve, warm_start=warm_start,
                         progress=progress, verbose=verbose, mode=spec.mode,
                         reset_on_failure=False)
    points = solved[0]
    if spec.direction == "down":
        points = points[::-1]
    return TableResult(par=spec.par, mode=spec.mode, direction=spec.direction,
                       nB=grid, lines=lines, points=points)


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
    return dict(n_B=point.n_b_fm, T=0.0,
                P=point.P, eps=point.eps, s=0.0, S_per_B=0.0,
                chi=state.n_bQ / state.n_b if state.n_b > 0 else 0.0,
                mu_B=point.mu_b, mu_C=-point.mu_e, mu_S=0.0, mu_e=point.mu_e,
                Y_C=state.n_C / state.n_b if state.n_b > 0 else 0.0,
                Y_S=state.n_S / state.n_b if state.n_b > 0 else 0.0,
                n_p=n["p"], n_n=n["n"], n_Lambda=n["Lambda"],
                n_u=n["u"], n_d=n["d"], n_s=n["s"],
                n_e=n["e"], n_mu=n["mu"],
                M_u=state.M_q["u"], M_d=state.M_q["d"], M_s=state.M_q["s"],
                M_p=state.M_b["p"], M_n=state.M_b["n"],
                M_Lambda=state.M_b["Lambda"])


__all__ = ["BetaPoint", "DIRECTIONS", "TableSpec", "TableResult",
           "beta_row", "build_table"]
