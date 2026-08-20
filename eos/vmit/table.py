"""Table driver for vMIT: a TableSpec, and build_table() that solves it.

A table is a set of lines -- one per temperature and per combination of the
fractions the mode fixes -- each swept along the baryon density with a warm
start. That loop is not written here: it is `eos.general.tabulate`, shared with
every other model. What this module supplies is the vMIT-specific part, which
is only three things: which solver a mode name means, what a solved point
carries into the next warm start, and how a point flattens into a table row.

    spec = TableSpec(params, "beta_eq_neutrinoless",
                     axes={"nB": np.linspace(0.3, 1.5, 60), "T": [0.0, 30.0]})
    result = build_table(spec, verbose=True)
"""
from dataclasses import dataclass, field

import numpy as np

from eos.general.tabulate import (
    lines_from_axes, sweep_lines, TEMPERATURE_AXES,
)
from eos.vmit.parameters import Parameters, get_vmit_default
from eos.vmit.species import SpeciesFlags
from eos.vmit.solver import (
    solve_vmit_beta_eq, solve_vmit_fixed_yc, solve_vmit_fixed_yc_ys,
    solve_vmit_trapped_neutrinos, result_to_guess,
)

#: Mode name -> the fixed fractions it consumes, as a `TableSpec.axes` grid or
#: as a scalar in `TableSpec.fixed`. The names are the repository's, shared
#: with eos.dd2 and eos.mixed, so a quark table and a hadronic table are
#: requested the same way.
MODE_FRACTIONS = {
    "beta_eq_neutrinoless": (),
    "beta_eq_neutrino_trapped": ("Y_Le",),
    "fixed_YC": ("Y_C",),
    "fixed_YC_YS": ("Y_C", "Y_S"),
}

#: Mode name -> the guess layout `result_to_guess` must produce for it. The
#: unknown vectors differ between modes, so a warm start is only valid within
#: its own mode.
_GUESS_KIND = {
    "beta_eq_neutrinoless": "beta_eq",
    "beta_eq_neutrino_trapped": "trapped_neutrinos",
    "fixed_YC": "fixed_yc",
    "fixed_YC_YS": "fixed_yc_ys",
}


def solve_at(params, mode, n_B, conditions, species, leptons, x0=None):
    """One point of a table: dispatch to the mode's solver.

    `conditions` carries the line's temperature (`T`) and whichever fractions
    the mode fixes, under the spec names. Non-convergence comes back on the
    result, not as an exception.
    """
    T = conditions["T"]
    photons = species.photons
    if mode == "beta_eq_neutrinoless":
        return solve_vmit_beta_eq(n_B, T, params, include_photons=photons,
                                  initial_guess=x0)
    if mode == "beta_eq_neutrino_trapped":
        return solve_vmit_trapped_neutrinos(n_B, conditions["Y_Le"], T, params,
                                            include_photons=photons,
                                            initial_guess=x0)
    if mode == "fixed_YC":
        return solve_vmit_fixed_yc(n_B, conditions["Y_C"], T, params,
                                   include_photons=photons,
                                   include_electrons=leptons,
                                   initial_guess=x0)
    if mode == "fixed_YC_YS":
        return solve_vmit_fixed_yc_ys(n_B, conditions["Y_C"],
                                      conditions["Y_S"], T, params,
                                      include_photons=photons,
                                      include_electrons=leptons,
                                      initial_guess=x0)
    raise ValueError(f"unknown mode {mode!r}; expected one of "
                     f"{list(MODE_FRACTIONS)}")


def quark_row(result, mode):
    """Flatten one solved point into a table row.

    Keyed the way `eos.dd2.table.hadronic_row` keys a hadronic point, so a
    pure-quark table and a hadronic one concatenate without renaming: chi = 1
    and phase = 'Q' say the matter is entirely deconfined.
    """
    n_B = result.n_B
    row = dict(n_B=n_B, T=result.T, chi=1.0, phase="Q",
               P=result.P_total, eps=result.e_total, s=result.s_total,
               S_per_B=(result.s_total / n_B if n_B else 0.0),
               mu_B=result.mu_B, mu_C=result.mu_C, mu_S=result.mu_S,
               mu_e=result.mu_e,
               Y_C=_charge_fraction(result),
               Y_S=result.n_s / n_B,
               Y_u=result.n_u / n_B, Y_d=result.n_d / n_B,
               Y_s=result.n_s / n_B, Y_e=result.n_e / n_B)
    if mode == "beta_eq_neutrino_trapped":
        row["Y_nue"] = result.n_nu / n_B
        row["mu_nue"] = result.mu_nu
    return row


def _charge_fraction(result):
    """Y_C = n_C/n_B of the strongly-interacting matter, leptons excluded."""
    from eos.general.basis import quark_charges
    n_B, n_C, _ = quark_charges(result.n_u, result.n_d, result.n_s)
    return n_C / n_B


@dataclass
class TableSpec:
    """One table request.

    axes  : {'nB': grid, exactly one of 'T'/'SnB': grid, and optionally any
            fraction the mode fixes ('Y_C', 'Y_S', 'Y_Le') as a further axis}
    fixed : scalar values for the fractions the mode needs and the axes do not
            sweep
    leptons: for the fixed-fraction modes, whether neutralizing electrons are
            added. Has no meaning in the beta-equilibrium modes, where leptons
            are what the equilibrium is about.
    """
    params: Parameters = field(default_factory=get_vmit_default)
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
        if temp_keys[0] == "SnB":
            raise NotImplementedError(
                "vMIT tables take a temperature axis; the entropy-per-baryon "
                "axis is not wired (see docs/DEFERRED.md)")
        # Every fraction the mode fixes must arrive, as an axis or a scalar.
        supplied = set(self.axes) | set(self.fixed)
        for key in MODE_FRACTIONS[self.mode]:
            if key not in supplied:
                raise ValueError(f"mode {self.mode!r} needs {key!r}, as an "
                                 f"axis or in fixed")


@dataclass
class TableResult:
    spec: TableSpec
    nB: np.ndarray
    #: One entry per line, parallel to `points`: the conditions that line was
    #: solved at ({'T': ..., and whichever fractions the mode fixes}).
    lines: list
    #: points[i_line][i_nB], the solved states of each line. With
    #: skip_errors a line is shorter than `nB`.
    points: list


def build_table(spec, skip_errors=True, rows=False, progress=None,
                verbose=False):
    """Solve a TableSpec over the product of its temperature and fraction axes.

    Within a line the density axis is swept with a warm start: each solved
    point seeds the next, which is what carries the solve through the strange
    quark's onset. A warm start is only valid within its own mode, since the
    unknown vectors differ.

    rows=False (default) returns a `TableResult`. rows=True instead returns
    the long format `eos.general.table_io` writes -- one flat dict per solved
    point.

    skip_errors=True drops points the solver could not reach instead of
    aborting the table; progress/verbose report per line, in the shape every
    table builder in this repository uses.
    """
    axes = {k: v for k, v in spec.axes.items() if k != "nB"}
    lines = lines_from_axes(axes, fixed=spec.fixed)
    guess_kind = _GUESS_KIND[spec.mode]

    def solve(n_B, conditions, x0):
        return solve_at(spec.params, spec.mode, n_B, conditions, spec.include,
                        spec.leptons, x0=x0)

    def warm_start(point):
        return result_to_guess(point, guess_kind,
                               include_electrons=spec.leptons)

    points = sweep_lines(lines, spec.axes["nB"], solve, warm_start=warm_start,
                         skip_errors=skip_errors, progress=progress,
                         verbose=verbose, mode=spec.mode)
    result = TableResult(spec=spec, nB=np.asarray(spec.axes["nB"], dtype=float),
                         lines=lines, points=points)
    return rows_from_result(result) if rows else result


def rows_from_result(result):
    """A solved `TableResult` as the long-format rows `eos.general.table_io`
    writes. Separate from `build_table` so a table already in hand can be
    written out without being solved a second time.
    """
    out = []
    for conditions, line in zip(result.lines, result.points):
        for point in line:
            row = quark_row(point, result.spec.mode)
            # The line's conditions are recorded, but never on top of a
            # quantity the row already carries: Y_C in the row is the charge
            # the solved state turned out to have, which is what a table is
            # for, not the value that was asked for.
            for key, value in conditions.items():
                row.setdefault(key, value)
            out.append(row)
    return out
