"""Table driver for ZL: a TableSpec, and build_table() that solves it.

A table is a set of lines -- one per temperature and per combination of the
fractions the mode fixes -- each swept along the baryon density with a warm
start. That loop is not written here: it is `eos.general.tabulate`, shared with
every other model. What this module supplies is the ZL-specific part, which is
only three things: which solver a mode name means, what a solved point carries
into the next warm start, and how a point flattens into a table row.

    spec = TableSpec(params, "beta_eq_neutrinoless",
                     axes={"nB": np.linspace(0.05, 1.2, 100), "T": [0.1, 30.0]})
    result = build_table(spec, verbose=True)

The settings-object interface at the bottom (`TableSettings`, `compute_table`)
is the first-generation one, kept because the ZLvMIT notebook drives ZL
through it; it is a translation layer over `build_table`, not a second
sweep.
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from eos.general.tabulate import (
    lines_from_axes, sweep_lines, TEMPERATURE_AXES,
)
from eos.general.modes import resolve_leptons
from eos.zl.parameters import Parameters
from eos.zl.solver import (
    EoSPoint, MODE_FRACTIONS, solve_beta_eq_neutrinoless,
    solve_beta_eq_neutrino_trapped, solve_fixed_yc, solve_fixed_yc_ys,
    warm_start,
)
from eos.zl.species import SpeciesFlags


def solve_at(params, mode, n_B, conditions, species, leptons, x0=None):
    """One point of a table: dispatch to the mode's solver.

    `conditions` carries the line's temperature (`T`) and whichever fractions
    the mode fixes, under the spec names. Non-convergence comes back on the
    result, not as an exception.
    """
    T = conditions["T"]
    photons = species.photons
    if mode == "beta_eq_neutrinoless":
        return solve_beta_eq_neutrinoless(params, n_B, T,
                                          include_photons=photons,
                                          initial_guess=x0)
    if mode == "beta_eq_neutrino_trapped":
        return solve_beta_eq_neutrino_trapped(params, n_B,
                                              conditions["Y_Le"], T,
                                              include_photons=photons,
                                              initial_guess=x0)
    if mode == "fixed_YC":
        return solve_fixed_yc(params, n_B, conditions["Y_C"], T,
                              include_photons=photons,
                              include_electrons=leptons, initial_guess=x0)
    if mode == "fixed_YC_YS":
        return solve_fixed_yc_ys()      # raises, naming the physics
    raise ValueError(f"unknown mode {mode!r}; expected one of "
                     f"{list(MODE_FRACTIONS)}")


def nucleon_row(result, mode):
    """Flatten one solved point into a table row.

    Keyed the way `eos.vmit.table.quark_row` and `eos.dd2.table.hadronic_row`
    key their points, so a nucleonic table and a quark one concatenate without
    renaming: chi = 0 and phase = 'H' say the matter is entirely hadronic.
    Y_S is written explicitly as zero rather than omitted -- the column exists
    in the shared layout and ZL's answer for it is a number, not a gap.
    """
    n_B = result.n_B
    row = dict(n_B=n_B, T=result.T, chi=0.0, phase="H",
               P=result.P_total, eps=result.e_total, s=result.s_total,
               S_per_B=(result.s_total / n_B if n_B else 0.0),
               mu_B=result.mu_B, mu_C=result.mu_C, mu_S=0.0,
               mu_e=result.mu_e,
               Y_C=(result.n_p / n_B), Y_S=0.0,
               Y_p=result.Y_p, Y_n=result.Y_n, Y_e=result.Y_e)
    if mode == "beta_eq_neutrino_trapped":
        row["Y_nue"] = result.n_nu / n_B
        row["mu_nue"] = result.mu_nu
    return row


@dataclass
class TableSpec:
    """One table request.

    axes  : {'nB': grid, exactly one of 'T'/'SnB': grid, and optionally any
            fraction the mode fixes ('Y_C', 'Y_Le') as a further axis}
    fixed : scalar values for the fractions the mode needs and the axes do not
            sweep
    leptons: for `fixed_YC`, whether neutralizing electrons are added. In the
            beta-equilibrium modes the leptons are constitutive, so True is
            redundant and ignored and False raises.
    """
    params: Parameters = field(default_factory=Parameters.default)
    mode: str = "beta_eq_neutrinoless"
    axes: dict = field(default_factory=dict)
    include: SpeciesFlags = field(default_factory=SpeciesFlags)
    fixed: dict = field(default_factory=dict)
    leptons: bool = True

    def __post_init__(self):
        if "nB" not in self.axes:
            raise ValueError("TableSpec.axes must contain 'nB'")
        if self.mode == "fixed_YC_YS":
            solve_fixed_yc_ys()          # raises, naming the physics
        if self.mode not in MODE_FRACTIONS:
            raise ValueError(f"unknown mode {self.mode!r}; expected one of "
                             f"{list(MODE_FRACTIONS)}")
        temp_keys = [k for k in self.axes if k in TEMPERATURE_AXES]
        if len(temp_keys) != 1:
            raise ValueError("TableSpec.axes needs exactly one of 'T' / 'SnB'")
        if temp_keys[0] == "SnB":
            raise NotImplementedError(
                "ZL tables take a temperature axis; the entropy-per-baryon "
                "axis is not wired (see docs/DEFERRED.md)")
        # Every fraction the mode fixes must arrive, as an axis or a scalar.
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
    point seeds the next, which is what carries the solve through the stiff
    rise of the interaction potentials. A warm start is only valid within its
    own mode, since the unknown vectors differ.

    rows=False (default) returns a `TableResult`. rows=True instead returns
    the long format `eos.general.table_io` writes -- one flat dict per solved
    point.

    skip_errors=True drops points the solver could not reach instead of
    aborting the table; progress/verbose report per line, in the shape every
    table builder in this repository uses.
    """
    axes = {k: v for k, v in spec.axes.items() if k != "nB"}
    lines = lines_from_axes(axes, fixed=spec.fixed)

    def solve(n_B, conditions, x0):
        return solve_at(spec.params, spec.mode, n_B, conditions, spec.include,
                        spec.leptons, x0=x0)

    def seed(point):
        return warm_start(point, spec.mode, leptons=spec.leptons)

    points = sweep_lines(lines, spec.axes["nB"], solve, warm_start=seed,
                         skip_errors=skip_errors, progress=progress,
                         verbose=verbose, mode=spec.mode)
    result = TableResult(spec=spec,
                         nB=np.asarray(spec.axes["nB"], dtype=float),
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
            row = nucleon_row(point, result.spec.mode)
            # The line's conditions are recorded, but never on top of a
            # quantity the row already carries: Y_C in the row is the charge
            # the solved state turned out to have, which is what a table is
            # for, not the value that was asked for.
            for key, value in conditions.items():
                row.setdefault(key, value)
            out.append(row)
    return out


# =============================================================================
# THE FIRST-GENERATION SETTINGS INTERFACE, kept for the ZLvMIT notebook
# =============================================================================
#: The legacy equilibrium names, and the repository mode each one means.
_LEGACY_MODES = {
    'beta_eq': "beta_eq_neutrinoless",
    'fixed_yc': "fixed_YC",
    'trapped_neutrinos': "beta_eq_neutrino_trapped",
}

#: The legacy name of each mode's fraction axes, and the spec name it maps to.
_LEGACY_FRACTIONS = {
    'beta_eq': (),
    'fixed_yc': (("Y_C_values", "Y_C"),),
    'trapped_neutrinos': (("Y_L_values", "Y_Le"),),
}


@dataclass
class TableSettings:
    """Configuration for a ZL table.

    equilibrium: 'beta_eq', 'fixed_yc' or 'trapped_neutrinos'.
    """
    # Model parameters
    params: Optional[Parameters] = None  # None = use the published set

    # Equilibrium type
    equilibrium: str = 'beta_eq'

    # Grid definition
    n_B_values: np.ndarray = field(
        default_factory=lambda: np.linspace(0.1, 12, 300) * 0.16)
    T_values: List[float] = field(default_factory=lambda: [10.0])

    # Constraint parameters (depending on equilibrium mode)
    Y_C_values: List[float] = field(default_factory=lambda: [0.3])
    Y_L_values: List[float] = field(default_factory=lambda: [0.4])

    # Options
    include_photons: bool = True
    include_leptons: bool = True    # only for fixed_yc

    # Output control
    print_results: bool = True
    print_first_n: int = 5
    print_errors: bool = True
    print_timing: bool = True

    # File output
    save_to_file: bool = False
    output_filename: Optional[str] = None  # auto-generate if None


def compute_table(settings: TableSettings) -> Dict[Tuple, List[EoSPoint]]:
    """Solve the settings grid; returns {(T, [fraction...]): [points]}.

    The key of each line is the tuple of its grid values in the legacy order
    (T first, then the mode's fractions), and lines are ordered the way nested
    loops over those axes would produce them. Each line is parallel to
    `n_B_values`, so a caller may index it positionally: a density the solver
    could not reach aborts the call rather than silently shortening the line.
    """
    eq_type = settings.equilibrium.lower()
    if eq_type not in _LEGACY_MODES:
        raise ValueError(f"Unknown equilibrium type: {eq_type}")

    params = settings.params
    if params is None:
        params = Parameters.default()
    frac_axes = _LEGACY_FRACTIONS[eq_type]

    axes = {"nB": np.asarray(settings.n_B_values),
            "T": np.atleast_1d(np.asarray(settings.T_values, dtype=float))}
    for legacy_name, spec_name in frac_axes:
        axes[spec_name] = np.atleast_1d(
            np.asarray(getattr(settings, legacy_name), dtype=float))

    spec = TableSpec(params=params, mode=_LEGACY_MODES[eq_type], axes=axes,
                     include=SpeciesFlags(photons=settings.include_photons),
                     # None where the mode has no such flag: in a
                     # beta-equilibrium mode the leptons are constitutive,
                     # and an explicit False there is refused.
                     leptons=(settings.include_leptons
                              if _LEGACY_MODES[eq_type] == "fixed_YC"
                              else None))
    verbose = settings.print_results or settings.print_timing
    result = build_table(spec, skip_errors=False, verbose=verbose)

    all_results = {}
    for conditions, line in zip(result.lines, result.points):
        key = tuple([conditions["T"]]
                    + [conditions[spec_name] for _, spec_name in frac_axes])
        all_results[key] = line

    if settings.save_to_file:
        save_results(all_results, settings, params, eq_type)

    return all_results


def save_results(all_results: Dict[Tuple, List[EoSPoint]],
                 settings: TableSettings, params: Parameters, eq_type: str):
    """Write the solved lines out in the legacy column layout."""
    if settings.output_filename:
        filename = settings.output_filename
    else:
        from eos import REPO_ROOT
        filename = str(REPO_ROOT / "output" / f"zl_{params.name}_{eq_type}.dat")

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w') as f:
        f.write(f"# ZL EOS Table: {params.name}\n")
        f.write(f"# Equilibrium: {eq_type}\n")

        # Format: inputs first, then mu, then Y, then thermodynamics
        if eq_type == 'fixed_yc':
            columns = ['n_B', 'Y_C', 'T', 'mu_p', 'mu_n', 'mu_e',
                       'Y_p', 'Y_n', 'Y_e',
                       'P_total', 'e_total', 's_total', 'f_total', 'converged']
        elif eq_type == 'trapped_neutrinos':
            columns = ['n_B', 'Y_L', 'T', 'mu_p', 'mu_n', 'mu_e', 'mu_nu',
                       'Y_p', 'Y_n', 'Y_e', 'Y_nu',
                       'P_total', 'e_total', 's_total', 'f_total', 'converged']
        else:  # beta_eq
            columns = ['n_B', 'T', 'mu_p', 'mu_n', 'mu_e', 'Y_p', 'Y_n', 'Y_e',
                       'P_total', 'e_total', 's_total', 'f_total', 'converged']

        f.write("# " + " ".join(f"{col:>14}" for col in columns) + "\n")

        trapped = eq_type == 'trapped_neutrinos'
        for key, results in all_results.items():
            Y_L_val = key[1] if trapped and len(key) > 1 else 0.0

            for r in results:
                if r.converged:
                    f_total = r.e_total - r.s_total * r.T
                    if eq_type == 'fixed_yc':
                        row = [r.n_B, r.Y_C, r.T, r.mu_p, r.mu_n, r.mu_e,
                               r.Y_p, r.Y_n, r.Y_e, r.P_total, r.e_total,
                               r.s_total, f_total, 1]
                    elif eq_type == 'trapped_neutrinos':
                        Y_nu = r.n_nu / r.n_B if r.n_B > 0 else 0.0
                        row = [r.n_B, Y_L_val, r.T, r.mu_p, r.mu_n, r.mu_e,
                               r.mu_nu, r.Y_p, r.Y_n, r.Y_e, Y_nu, r.P_total,
                               r.e_total, r.s_total, f_total, 1]
                    else:  # beta_eq
                        row = [r.n_B, r.T, r.mu_p, r.mu_n, r.mu_e,
                               r.Y_p, r.Y_n, r.Y_e, r.P_total, r.e_total,
                               r.s_total, f_total, 1]
                    f.write(" ".join(f"{v:>14.6e}" if isinstance(v, float)
                                     else f"{v:>14}" for v in row) + "\n")

    print(f"\nSaved to: {filename}")
