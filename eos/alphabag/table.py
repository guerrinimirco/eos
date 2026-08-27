"""Table driver for alphaBag: a TableSpec, and build_table() that solves it.

A table is a set of lines -- one per temperature and per combination of the
fractions the mode fixes (or, for the paired phase, per pairing gap) -- each
swept along the baryon density with a warm start. That loop is not written
here: it is `eos.general.tabulate`, shared with every other model. What this
module supplies is the alphaBag-specific part, which is only three things:
which solver a mode name means, what a solved point carries into the next warm
start, and how a point flattens into a table row.

    spec = TableSpec(Parameters.default(), "beta_eq_neutrinoless",
                     axes={"nB": np.linspace(0.3, 2.0, 100), "T": [0.0, 30.0]})
    result = build_table(spec, verbose=True)

The colour-flavour locked phase is driven through the same spec, as the mode
name `cfl` with a `Delta0` axis: it is a phase rather than an equilibrium, but
it is a line-per-parameter sweep along density like any other and there is no
reason for it to have a second driver.

The settings-object interface at the bottom (`TableSettings`, `compute_table`,
`save_results`) is the first-generation one, kept because the 2fam PNS
nucleation study drives alphaBag through it and its `.dat` files are a format
on disk; it is a translation layer over `build_table`, not a second sweep.
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from eos.general.tabulate import (
    lines_from_axes, sweep_lines, TEMPERATURE_AXES,
)
from eos.general.modes import resolve_leptons
from eos.alphabag.parameters import Parameters
from eos.alphabag.solver import (
    MODE_FRACTIONS, solve_beta_eq_neutrinoless, solve_beta_eq_neutrino_trapped,
    solve_cfl, solve_fixed_yc, solve_fixed_yc_ys, warm_start,
)
from eos.alphabag.species import SpeciesFlags


#: How many times a missed density step may be halved back towards the last
#: solved point. alphaBag has two real thresholds along a density sweep -- the
#: strange quark's onset, and the gap closing at T_c in the paired phase --
#: so a warm start carried across one grid interval can land outside the
#: basin, which is what a bisected step walks through.
MAX_BISECT = 6


def solve_at(params, mode, n_B, conditions, species, leptons, x0=None):
    """One point of a table: dispatch to the mode's solver.

    `conditions` carries the line's temperature (`T`) and whichever fractions
    the mode fixes -- or, for `cfl`, the pairing gap -- under the spec names.
    Non-convergence comes back on the result, not as an exception.
    """
    T = conditions["T"]
    photons, gluons = species.photons, species.gluons
    neutrinos = species.thermal_neutrinos
    two_flavour = species.two_flavour
    if mode == "beta_eq_neutrinoless":
        return solve_beta_eq_neutrinoless(
            params, n_B, T, include_photons=photons, include_gluons=gluons,
            include_thermal_neutrinos=neutrinos, initial_guess=x0,
            two_flavour=two_flavour)
    if mode == "beta_eq_neutrino_trapped":
        return solve_beta_eq_neutrino_trapped(
            params, n_B, conditions["Y_Le"], T, include_photons=photons,
            include_gluons=gluons, include_thermal_neutrinos=neutrinos,
            initial_guess=x0, two_flavour=two_flavour)
    if mode == "fixed_YC":
        return solve_fixed_yc(
            params, n_B, conditions["Y_C"], T, include_photons=photons,
            include_gluons=gluons, include_electrons=leptons,
            include_thermal_neutrinos=neutrinos, initial_guess=x0,
            two_flavour=two_flavour)
    if mode == "fixed_YC_YS":
        return solve_fixed_yc_ys(
            params, n_B, conditions["Y_C"], conditions["Y_S"], T,
            include_photons=photons, include_gluons=gluons,
            include_electrons=leptons, include_thermal_neutrinos=neutrinos,
            initial_guess=x0, two_flavour=two_flavour)
    if mode == "cfl":
        # The paired phase carries no lepton condition, and two of the
        # thermal sectors are not its physics. `gluons` is refused inside
        # solve_cfl (the eight gluons are Meissner-massive; only the rotated
        # photon stays massless), and the thermal neutrino gas is refused
        # here, because the paired phase has never carried it -- see
        # docs/DEFERRED.md. Neither is silently dropped: section 4 of
        # CLAUDE.md requires a sector a model does not implement to raise
        # rather than be ignored.
        if two_flavour:
            raise NotImplementedError(
                "alphaBag 'cfl': colour-flavour locking pairs the three "
                "flavours at equal densities, so Y_S = +1 identically and "
                "there is no strangeness fraction free to switch off. "
                "two_flavour is refused here for the same reason gluons is: "
                "the flag keeps both its values in the unpaired modes and is "
                "a statement about the phase in this one. Two-flavour quark "
                "matter is 'beta_eq_neutrinoless' with two_flavour=True")
        if neutrinos:
            raise NotImplementedError(
                "alphaBag 'cfl': the paired phase carries no thermal "
                "neutrino gas, where every unpaired solver adds one. The "
                "asymmetry is inherited from the first-generation CFL table "
                "builder and is preserved deliberately, because closing it "
                "moves published CFL tables; see docs/DEFERRED.md. Use "
                "thermal_neutrinos=False in the 'cfl' mode")
        return solve_cfl(params, n_B, T, conditions["Delta0"],
                         include_photons=photons, include_gluons=gluons,
                         initial_guess=x0)
    raise ValueError(f"unknown mode {mode!r}; expected one of "
                     f"{list(MODE_FRACTIONS)}")


def quark_row(result, mode):
    """Flatten one solved point into a table row.

    Keyed the way `eos.vmit.table.quark_row` and `eos.dd2.table.hadronic_row`
    key their points, so a quark table and a hadronic one concatenate without
    renaming: chi = 1 and phase = 'Q' say the matter is entirely deconfined.
    The paired phase adds the gap it was solved at, which is the one thing a
    reader cannot recover from the other columns.
    """
    n_B = result.n_B
    row = dict(n_B=n_B, T=result.T, chi=1.0, phase="Q",
               P=result.P_total, eps=result.e_total, s=result.s_total,
               S_per_B=(result.s_total / n_B if n_B else 0.0),
               mu_B=result.mu_B, mu_C=result.mu_C, mu_S=result.mu_S,
               mu_e=result.mu_e,
               Y_C=result.Y_C, Y_S=result.Y_S,
               Y_u=result.Y_u, Y_d=result.Y_d, Y_s=result.Y_s,
               Y_e=result.Y_e)
    if mode == "beta_eq_neutrino_trapped":
        row["Y_nue"] = result.Y_nu
        row["mu_nue"] = result.mu_nu
    if mode == "cfl":
        row["Delta0"] = result.Delta0
        row["Delta"] = result.Delta
    return row


@dataclass
class TableSpec:
    """One table request.

    axes  : {'nB': grid, exactly one of 'T'/'SnB': grid, and optionally any
            condition the mode fixes ('Y_C', 'Y_S', 'Y_Le', or 'Delta0' for
            the paired phase) as a further axis}
    fixed : scalar values for the conditions the mode needs and the axes do
            not sweep
    leptons: for the fixed-fraction modes, whether neutralizing electrons are
            added. None leaves it at alphaBag's leptonless default; in the
            beta-equilibrium modes the leptons are constitutive, so True is
            redundant and ignored and False raises. The paired phase is
            neutral by construction.
    """
    params: Parameters = field(default_factory=Parameters.default)
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
        if temp_keys[0] == "SnB":
            raise NotImplementedError(
                "alphaBag tables take a temperature axis; the "
                "entropy-per-baryon axis is not wired "
                "(see docs/DEFERRED.md)")
        # Every condition the mode fixes must arrive, as an axis or a scalar.
        supplied = set(self.axes) | set(self.fixed)
        for key in MODE_FRACTIONS[self.mode]:
            if key not in supplied:
                raise ValueError(f"mode {self.mode!r} needs {key!r}, as an "
                                 f"axis or in fixed")
        self.leptons = resolve_leptons(self.mode, self.leptons, default=False)


@dataclass
class TableResult:
    spec: TableSpec
    nB: np.ndarray
    #: One entry per line, parallel to `points`: the conditions that line was
    #: solved at ({'T': ..., and whichever conditions the mode fixes}).
    lines: list
    #: points[i_line][i_nB], the solved states of each line. With
    #: skip_errors a line is shorter than `nB`.
    points: list


def build_table(spec, skip_errors=True, rows=False, progress=None,
                verbose=False):
    """Solve a TableSpec over the product of its temperature and fraction axes.

    Within a line the density axis is swept with a warm start: each solved
    point seeds the next, and where a solve misses the step is bisected, which
    is what carries the sweep through the strange-quark onset rather than
    stopping at it. A warm start is only valid within its own mode, since the
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

    def solve(n_B, conditions, x0):
        return solve_at(spec.params, spec.mode, n_B, conditions, spec.include,
                        spec.leptons, x0=x0)

    def seed(point):
        return warm_start(point, spec.mode,
                          two_flavour=spec.include.two_flavour)

    points = sweep_lines(lines, spec.axes["nB"], solve, warm_start=seed,
                         skip_errors=skip_errors, progress=progress,
                         verbose=verbose, mode=spec.mode,
                         max_bisect=MAX_BISECT)
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
            row = quark_row(point, result.spec.mode)
            # The line's conditions are recorded, but never on top of a
            # quantity the row already carries: Y_C in the row is the charge
            # the solved state turned out to have, which is what a table is
            # for, not the value that was asked for.
            for key, value in conditions.items():
                row.setdefault(key, value)
            out.append(row)
    return out


# =============================================================================
# THE FIRST-GENERATION SETTINGS INTERFACE, kept for the 2fam PNS study
# =============================================================================
#: The legacy phase/equilibrium names, and the repository mode each one means.
_LEGACY_MODES = {
    ('unpaired', 'beta_eq'): "beta_eq_neutrinoless",
    ('unpaired', 'fixed_yc'): "fixed_YC",
    ('unpaired', 'fixed_yc_ys'): "fixed_YC_YS",
    ('cfl', None): "cfl",
}

#: The legacy name of each mode's condition axes, and the spec name it maps to.
_LEGACY_AXES = {
    "beta_eq_neutrinoless": (),
    "fixed_YC": (("Y_C_values", "Y_C"),),
    "fixed_YC_YS": (("Y_C_values", "Y_C"), ("Y_S_values", "Y_S")),
    "cfl": (("Delta0_values", "Delta0"),),
}


@dataclass
class TableSettings:
    """Configuration for an alphaBag table.

    phase: 'unpaired' or 'cfl'.
    equilibrium: 'beta_eq', 'fixed_yc' or 'fixed_yc_ys' (unpaired only).
    """
    # Model parameters. `params` wins; otherwise the three easy-access numbers
    # build a set, each falling back to the shipped default.
    params: Optional[Parameters] = None
    alpha: Optional[float] = None
    B4: Optional[float] = None
    m_s: Optional[float] = None

    # Phase and equilibrium
    phase: str = 'unpaired'
    equilibrium: str = 'beta_eq'

    # CFL gap
    Delta0_values: List[float] = field(default_factory=lambda: [100.0])

    # Grid definition
    n_B_values: np.ndarray = field(
        default_factory=lambda: np.linspace(0.1, 10, 100) * 0.16)
    T_values: List[float] = field(default_factory=lambda: [10.0])

    # Constraint parameters (unpaired fixed-fraction modes)
    Y_C_values: List[float] = field(default_factory=lambda: [0.0])
    Y_S_values: List[float] = field(default_factory=lambda: [0.0])

    # Options
    include_photons: bool = True
    include_gluons: bool = True
    include_electrons: bool = False
    include_thermal_neutrinos: bool = True

    # Output control
    print_results: bool = True
    print_first_n: int = 5
    print_errors: bool = True
    print_timing: bool = True

    # File output
    save_to_file: bool = False
    output_filename: Optional[str] = None

    def to_params(self) -> Parameters:
        """The parameter set this configuration means."""
        if self.params is not None:
            return self.params
        if self.alpha is None and self.B4 is None and self.m_s is None:
            return Parameters.default()
        default = Parameters.default()
        return Parameters(
            name="alphabag_custom",
            alpha=default.alpha if self.alpha is None else self.alpha,
            B4=default.B4 if self.B4 is None else self.B4,
            m_s=default.m_s if self.m_s is None else self.m_s,
        )


def compute_table(settings: TableSettings) -> Dict[Tuple, List]:
    """Solve the settings grid; returns {(T, [condition...]): [points]}.

    The key of each line is the tuple of its grid values in the legacy order
    (T first, then the mode's conditions -- the fractions for an unpaired
    fixed-fraction table, the pairing gap for a CFL one), and lines are
    ordered the way nested loops over those axes would produce them. Each line
    is parallel to `n_B_values`, so a caller may index it positionally: a
    density the solver could not reach aborts the call rather than silently
    shortening the line.
    """
    phase = settings.phase.lower()
    if phase == 'cfl':
        key = ('cfl', None)
    else:
        key = (phase, settings.equilibrium.lower())
    if key not in _LEGACY_MODES:
        raise ValueError(f"unknown phase/equilibrium {key}; expected one of "
                         f"{list(_LEGACY_MODES)}")
    mode = _LEGACY_MODES[key]
    params = settings.to_params()
    condition_axes = _LEGACY_AXES[mode]

    axes = {"nB": np.asarray(settings.n_B_values),
            "T": np.atleast_1d(np.asarray(settings.T_values, dtype=float))}
    for legacy_name, spec_name in condition_axes:
        axes[spec_name] = np.atleast_1d(
            np.asarray(getattr(settings, legacy_name), dtype=float))

    spec = TableSpec(
        params=params, mode=mode, axes=axes,
        include=SpeciesFlags(
            photons=settings.include_photons,
            gluons=settings.include_gluons,
            thermal_neutrinos=settings.include_thermal_neutrinos),
        # None where the mode has no such flag: in a beta-equilibrium mode
        # the leptons are constitutive, and an explicit False there is refused.
        leptons=(settings.include_electrons
                 if mode in ("fixed_YC", "fixed_YC_YS") else None))
    verbose = settings.print_results or settings.print_timing
    result = build_table(spec, skip_errors=False, verbose=verbose)

    all_results = {}
    for conditions, line in zip(result.lines, result.points):
        line_key = tuple([conditions["T"]]
                         + [conditions[spec_name]
                            for _, spec_name in condition_axes])
        all_results[line_key] = line

    if settings.save_to_file:
        save_results(all_results, settings, params, mode)

    return all_results


#: The legacy column layout, per mode. These are a format on disk: the 2fam
#: PNS nucleation study wrote its quark tables this way.
_LEGACY_COLUMNS = {
    "cfl": ['n_B', 'T', 'Delta0', 'Delta', 'mu_u', 'mu_d', 'mu_s',
            'P', 'e', 's', 'f'],
    "beta_eq_neutrinoless": ['n_B', 'T', 'mu_u', 'mu_d', 'mu_s', 'mu_e',
                             'Y_u', 'Y_d', 'Y_s', 'P_total', 'e_total',
                             's_total', 'converged'],
    "fixed": ['n_B', 'Y_C', 'T', 'mu_u', 'mu_d', 'mu_s', 'Y_u', 'Y_d', 'Y_s',
              'P_total', 'e_total', 's_total', 'converged'],
}


def save_results(all_results: Dict[Tuple, List], settings: TableSettings,
                 params: Parameters, mode: str):
    """Write the solved lines out in the legacy column layout.

    Non-converged points are dropped from an unpaired table; a CFL table
    writes every point, converged or not, as it always has.
    """
    if settings.output_filename:
        filename = settings.output_filename
    else:
        from eos import REPO_ROOT
        phase = "cfl" if mode == "cfl" else "unpaired"
        filename = str(REPO_ROOT / "output" /
                       f"alphabag_{phase}_B{int(params.B4)}_"
                       f"alpha{params.alpha}.dat")

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if mode == "cfl":
        columns = _LEGACY_COLUMNS["cfl"]
    elif mode == "beta_eq_neutrinoless":
        columns = _LEGACY_COLUMNS["beta_eq_neutrinoless"]
    else:
        columns = _LEGACY_COLUMNS["fixed"]

    with open(filename, 'w') as f:
        f.write(f"# AlphaBag EOS Table ({'cfl' if mode == 'cfl' else 'unpaired'})\n")
        f.write(f"# Parameters: B^1/4={params.B4} MeV, "
                f"α_s={params.alpha}, m_s={params.m_s} MeV\n")
        f.write(f"# Equilibrium: {settings.equilibrium}\n")

        components = []
        if settings.include_photons:
            components.append("photons")
        if settings.include_gluons:
            components.append("gluons")
        if settings.include_electrons:
            components.append("electrons")
        if settings.include_thermal_neutrinos:
            components.append("thermal_neutrinos")
        f.write(f"# Components: "
                f"{', '.join(components) if components else 'quarks only'}\n")
        f.write("# " + " ".join(f"{col:>14}" for col in columns) + "\n")

        for line in all_results.values():
            for r in line:
                if mode == "cfl":
                    row = [r.n_B, r.T, r.Delta0, r.Delta, r.mu_u, r.mu_d,
                           r.mu_s, r.P_total, r.e_total, r.s_total, r.f_total]
                else:
                    if not r.converged:
                        continue
                    if mode == "beta_eq_neutrinoless":
                        row = [r.n_B, r.T, r.mu_u, r.mu_d, r.mu_s, r.mu_e,
                               r.Y_u, r.Y_d, r.Y_s, r.P_total, r.e_total,
                               r.s_total, 1]
                    else:
                        row = [r.n_B, r.Y_C, r.T, r.mu_u, r.mu_d, r.mu_s,
                               r.Y_u, r.Y_d, r.Y_s, r.P_total, r.e_total,
                               r.s_total, 1]

                f.write(" ".join(f"{v:>14.6e}" if isinstance(v, float)
                                 else f"{v:>14}" for v in row) + "\n")

    print(f"\nSaved to: {filename}")
