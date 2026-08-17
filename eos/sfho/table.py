"""
table.py
========
The grid driver: a `TableSpec` naming the parametrization, the mode, the axes
and the active species, and `build_table` solving that grid.

One driver, not one per mode. A mode is a declaration (`eos.general.modes`),
`MODES` below says which declaration each table mode name means, and the sweep
never branches on it again. The density axis is the stiff one, so it is swept
innermost and warm-started -- the continuation tactics are on
`_continuation_guess`. The temperature axis is given either as 'T' or, per
CLAUDE.md section 3, as entropy per baryon 'SnB', which moves T into the
unknown vector.

`eos.dd2.table` has the same three names with the same meanings -- `TableSpec`,
`build_table`, `TableResult` -- and `build_table(rows=True)` emits the same row
keys, so a purely hadronic table from either model concatenates with a hybrid
table from `eos.mixed` without renaming a column.

The second half of this file is the first-generation .dat writer and reader
(`save_results`, `load_eos_table`, `build_interpolators`) and the settings
object that drives them. That is the on-disk format the published 2fam PNS
tables were written in, so it is kept as it stands; `compute_table` is now a
thin adapter onto `build_table` rather than a second sweep.

References:
- Fortin, Oertel, Providencia, PASA 35 (2018) e044
- Steiner, Hempel, Fischer, ApJ 774 (2013) 17
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any, Tuple
from itertools import product

from eos.general import modes
from eos.sfho.species import SpeciesFlags
from eos.sfho.solver import (
    EoSPoint, solve_mode, default_guess, warm_start,
)
from eos.sfho.parameters import (
    SFHoParams,
    get_sfho_nucleonic,
    get_sfhoy_fortin,
    get_sfhoy_star_fortin,
    get_sfho_2fam_phi,
    get_sfho_2fam
)


# =============================================================================
# THE MODES A TABLE CAN BE BUILT IN
# =============================================================================

#: Table mode name -> the `eos.general.modes` factory that declares it, plus
#: whether neutralizing leptons are present. The names are CLAUDE.md section
#: 3's, and match the ones `eos.dd2.table` and `eos.mixed` offer, so the same
#: table is requested from any of the three the same way:
#:
#:   beta_eq_neutrinoless      charge-neutral beta equilibrium, neutrinos escape
#:   beta_eq_neutrino_trapped  ... with the electron family trapped at Y_Le
#:   fixed_YC                  fixed non-leptonic charge fraction, no leptons
#:                             (charged matter -- what a mixed phase needs)
#:   fixed_YC_neutral          ... plus the neutralizing electrons
#:   fixed_YC_YS               charge and strangeness fixed, no leptons
#:   fixed_YC_YS_neutral       ... plus the neutralizing electrons
MODES = {
    "beta_eq_neutrinoless":     dict(spec=modes.beta_eq_neutrinoless),
    "beta_eq_neutrino_trapped": dict(spec=modes.beta_eq_neutrino_trapped),
    "fixed_YC":                 dict(spec=modes.fixed_YC, leptons=False),
    "fixed_YC_neutral":         dict(spec=modes.fixed_YC, leptons=True),
    "fixed_YC_YS":              dict(spec=modes.fixed_YC_YS, leptons=False),
    "fixed_YC_YS_neutral":      dict(spec=modes.fixed_YC_YS, leptons=True),
}

#: mode name -> the fractions it consumes, in the order its factory takes
#: them. Supplied either as a `TableSpec.axes` grid, to be swept, or as a
#: scalar in `TableSpec.fixed`.
MODE_FRACTIONS = {
    "beta_eq_neutrinoless": (),
    "beta_eq_neutrino_trapped": ("Y_Le",),
    "fixed_YC": ("Y_C",),
    "fixed_YC_neutral": ("Y_C",),
    "fixed_YC_YS": ("Y_C", "Y_S"),
    "fixed_YC_YS_neutral": ("Y_C", "Y_S"),
}

#: The equilibrium names the settings object at the bottom of this file uses,
#: mapped onto the modes above. `include_electrons` picks the neutral flavour
#: of a fixed-fraction mode, and the two isentropic names are the same modes
#: on the 'SnB' axis rather than modes of their own.
_LEGACY_EQUILIBRIA = {
    "beta_eq": ("beta_eq_neutrinoless", "T"),
    "trapped_neutrinos": ("beta_eq_neutrino_trapped", "T"),
    "fixed_yc": ("fixed_YC", "T"),
    "fixed_yc_ys": ("fixed_YC_YS", "T"),
    "isentropic_beta_eq": ("beta_eq_neutrinoless", "SnB"),
    "isentropic_trapped": ("beta_eq_neutrino_trapped", "SnB"),
}


def mode_spec(mode: str, fracs: Dict[str, float]) -> modes.ModeSpec:
    """The `ModeSpec` a table mode name declares, at these fractions.

    One place where a name becomes a declaration; nothing downstream of here
    branches on the name again.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {list(MODES)}")
    entry = dict(MODES[mode])
    factory = entry.pop("spec")
    values = []
    for key in MODE_FRACTIONS[mode]:
        if key not in fracs:
            raise ValueError(f"mode {mode!r} needs fixed[{key!r}]")
        values.append(fracs[key])
    return factory(*values, **entry)


# =============================================================================
# ONE POINT AS A ROW
# =============================================================================

def hadronic_row(r: EoSPoint) -> Dict[str, Any]:
    """Flatten one solved point into a dict row.

    Keyed the way `eos.dd2.table.hadronic_row` and `eos.mixed.composition_row`
    key theirs, so a pure-hadronic table and a hybrid table concatenate without
    renaming anything. chi = 0 and phase = 'H': no quark matter is present.

    Y_C and Y_S are the TOTAL non-leptonic fractions read off the solved state,
    so they count the thermal meson gas as well as the baryons (CLAUDE.md
    section 2) -- at T = 40 MeV with pions the two differ by 10 to 20 percent.
    """
    n_B = r.n_B
    row = dict(n_B=n_B, T=r.T, chi=0.0, phase="H",
               P=r.P, eps=r.eps, s=r.s,
               S_per_B=r.entropy_per_baryon,
               mu_B=r.mu_B, Y_C=r.Y_C, Y_S=r.Y_S,
               mu_e=r.mu_e, mu_S=r.mu_S, mu_nue=r.mu_nue,
               Y_e=r.n_e / n_B, **{"Y_mu-": 0.0})
    for name, n in r.composition:
        row[f"Y_{name}"] = n / n_B
    if r.n_nu:
        row["Y_nue"] = r.n_nu / n_B
    return row


# =============================================================================
# THE REQUEST AND THE RESULT
# =============================================================================

@dataclass
class TableSpec:
    """One table request.

    parametrization: the SFHoParams to solve with -- an argument, never module
        state, because inference varies it (CLAUDE.md section 6)
    mode : a key of MODES
    axes : {'nB': grid, exactly one of 'T'/'SnB': grid, and optionally any of
           'Y_C'/'Y_S'/'Y_Le': grid to sweep that fraction as a further axis}
    include: the active degrees of freedom
    fixed: scalar values for the fractions the mode needs that are not swept
    """
    parametrization: SFHoParams
    mode: str
    axes: dict
    include: SpeciesFlags = field(default_factory=SpeciesFlags)
    fixed: dict = field(default_factory=dict)

    def __post_init__(self):
        if "nB" not in self.axes:
            raise ValueError("TableSpec.axes must contain 'nB'")
        temp_axes = [k for k in self.axes if k in ("T", "SnB")]
        if len(temp_axes) != 1:
            raise ValueError("TableSpec.axes needs exactly one of 'T' / 'SnB'")
        self._temp_key = temp_axes[0]
        if self.mode not in MODES:
            raise ValueError(f"unknown mode {self.mode!r}; expected one of "
                             f"{list(MODES)}")
        self._frac_keys = [k for k in ("Y_C", "Y_S", "Y_Le") if k in self.axes]
        # Validate early that every fraction the mode needs is supplied, by an
        # axis or a scalar; an axis value stands in here only for the check.
        probe = dict(self.fixed)
        probe.update({k: 0.0 for k in self._frac_keys})
        mode_spec(self.mode, probe)


@dataclass
class TableResult:
    spec: TableSpec
    nB: np.ndarray
    temp_values: np.ndarray               # the T or S/A grid
    temp_key: str                         # 'T' or 'SnB'
    points: list                          # points[i_combo][i_nB] EoSPoint
    #: [(temperature value, {fraction: value}), ...], parallel to `points`.
    #: One entry per line; with no fraction axes it is one entry per
    #: temperature and the dict is empty.
    combos: list = None


def _print_progress(info):
    """The built-in progress printer (verbose=True)."""
    fracs = "".join(f" {k}={v:g}" for k, v in info["fracs"].items())
    print(f"[{info['line']}/{info['n_lines']}] {info['mode']} "
          f"{info['temp_key']}={info['temp']:g}{fracs}: "
          f"{info['n_solved']}/{info['n_requested']} points "
          f"in {info['elapsed_s']:.1f}s")


# =============================================================================
# THE WARM-STARTED SWEEP
# =============================================================================

def _continuation_guess(seeds, seed_nB, n_B):
    """The seed for the next density from the ones already solved on this line.

    Linear extrapolation in n_B from the last two converged points, which is
    what makes a fine density sweep cheap: every unknown varies smoothly with
    n_B away from a threshold, so a first-order step lands close enough that
    the solve takes a few iterations. With only one point behind it there is
    nothing to extrapolate along and that point is reused as it stands.

    Returns None when the line is empty, and the caller falls back to the
    cold start.
    """
    if len(seeds) >= 2 and abs(seed_nB[-1] - seed_nB[-2]) > 1e-15:
        slope = (seeds[-1] - seeds[-2]) / (seed_nB[-1] - seed_nB[-2])
        return seeds[-1] + slope * (n_B - seed_nB[-1])
    if seeds:
        return seeds[-1].copy()
    return None


def build_table(spec: TableSpec, skip_errors: bool = False,
                rows: bool = False, progress=None, verbose: bool = False):
    """
    Solve the TableSpec grid over the product of its temperature and fraction
    axes. Within each combination the n_B sweep is warm-started along density
    (the stiff axis); the 'SnB' axis puts T in the unknown vector instead of
    imposing it.

    rows=False (default) returns a `TableResult`, whose `points` are indexed
    [i_combination][i_nB] -- one line per (temperature, fractions) pair, in the
    order `TableResult.combos` records.

    rows=True instead returns `(rows, {})` in the long format
    `eos.dd2.build_table` and `eos.mixed.build_mixed_table` return -- one flat
    dict per CONVERGED point, ready for `eos.general.table_io`. The empty
    second element is where the mixed builder returns its phase windows, which
    purely hadronic matter has none of; it is returned anyway so the two calls
    unpack the same way.

    skip_errors: SFHo reports non-convergence as a return value rather than an
    exception (CLAUDE.md section 6), so this chooses what happens to such a
    point rather than whether it is fatal: True drops it from its line and
    resets the warm start, False keeps it in place with converged=False for
    the caller to filter. Either way `rows=True` emits converged points only.

    progress: optional callable, invoked once per completed line with a dict
    {mode, line, n_lines, temp_key, temp, fracs, n_solved, n_requested,
    elapsed_s} -- the same shape every table builder in this repository uses.
    Default is silent; verbose=True installs the built-in one-line printer.
    Deep solver code never prints.
    """
    if verbose and progress is None:
        progress = _print_progress
    par = spec.parametrization
    flags = spec.include
    nB = np.asarray(spec.axes["nB"], dtype=float)
    temp = np.asarray(spec.axes[spec._temp_key], dtype=float)
    isentropic = spec._temp_key == "SnB"
    frac_keys = spec._frac_keys
    frac_grids = [np.atleast_1d(np.asarray(spec.axes[k], float))
                  for k in frac_keys]
    n_lines = len(temp) * max(1, int(np.prod([len(g) for g in frac_grids])))

    points, combos = [], []
    previous_line = None      # the line before this one, seeds it point by point
    # Fractions outermost, temperature innermost, so consecutive lines differ
    # by a step in T (or S/A) at the same composition. That is the smaller step
    # for every unknown, which is what makes the line-to-line seeding below
    # worth having; `TableResult.combos` records the order it produced.
    for combo in product(*frac_grids) if frac_grids else [()]:
        for tv in temp:
            fracs = dict(spec.fixed)
            fracs.update(zip(frac_keys, (float(c) for c in combo)))
            spec_mode = mode_spec(spec.mode, fracs)
            T = None if isentropic else float(tv)
            SnB = float(tv) if isentropic else None
            # The cold start needs a temperature even when T is an unknown;
            # 10 MeV is the solver's own default seed for that case.
            T_seed = 10.0 if isentropic else float(tv)

            combos.append((float(tv), dict(zip(frac_keys, map(float, combo)))))
            t_line = time.time()
            line, seeds, seed_nB = [], [], []
            for i, n in enumerate(nB):
                cold = default_guess(spec_mode, float(n), T_seed, par,
                                     SnB=SnB)
                x0 = _continuation_guess(seeds, seed_nB, float(n))
                if x0 is None and previous_line is not None and i < len(previous_line):
                    # Nothing solved yet on this line: take the point at the
                    # same density from the previous one. A step in T or in a
                    # fraction moves the unknowns far less than a step in n_B.
                    neighbour = previous_line[i]
                    if neighbour.converged:
                        x0 = warm_start(neighbour, spec_mode, isentropic)
                if x0 is None:
                    x0 = cold
                r = solve_mode(par, float(n), flags, spec_mode, T=T, SnB=SnB,
                               x0=x0)
                if not r.converged and x0 is not cold:
                    r = solve_mode(par, float(n), flags, spec_mode, T=T,
                                   SnB=SnB, x0=cold)
                if r.converged:
                    seeds.append(warm_start(r, spec_mode, isentropic))
                    seed_nB.append(float(n))
                    if len(seeds) > 2:
                        seeds.pop(0)
                        seed_nB.pop(0)
                    line.append(r)
                elif skip_errors:
                    seeds, seed_nB = [], []     # reset the warm start past a gap
                else:
                    line.append(r)
            points.append(line)
            previous_line = line
            if progress is not None:
                progress(dict(mode=spec.mode, line=len(points),
                              n_lines=n_lines, temp_key=spec._temp_key,
                              temp=float(tv), fracs=combos[-1][1],
                              n_solved=sum(1 for r in line if r.converged),
                              n_requested=len(nB),
                              elapsed_s=time.time() - t_line))

    result = TableResult(spec=spec, nB=nB, temp_values=temp,
                         temp_key=spec._temp_key, points=points, combos=combos)
    return (rows_from_result(result), {}) if rows else result


def rows_from_result(result: TableResult) -> List[Dict[str, Any]]:
    """A solved `TableResult` to the long-format rows `eos.general.table_io`
    writes -- the same shape `eos.dd2.rows_from_result` produces.

    Separate from `build_table` so a table already solved for its
    `TableResult` can be written out without being solved a second time.
    Non-converged points are dropped here: a row is a state, and a point that
    did not converge is not one.
    """
    out = []
    for (tv, fracs), line in zip(result.combos, result.points):
        for r in line:
            if not r.converged:
                continue
            row = hadronic_row(r)
            row.update(fracs)
            if result.temp_key == "SnB":
                row["SnB"] = tv
            out.append(row)
    return out



# =============================================================================
# THE SETTINGS-OBJECT INTERFACE
# =============================================================================
# The first-generation script interface: one object describing a whole run, a
# single call, and a .dat file out. It is kept because the published 2fam PNS
# nucleation tables were produced through it and are read back through
# `load_eos_table` below, so its column layout is a format on disk and not an
# implementation detail. It is now an ADAPTER onto `build_table` -- the sweep
# lives in one place, and this half only translates names and writes files.


@dataclass
class TableSettings:
    """
    Configuration for SFHo EOS table generation (supports multi-dimensional grids).

    Equilibrium types:
    - 'beta_eq': Beta equilibrium with charge neutrality
    - 'fixed_yc': Fixed charge fraction Y_C
    - 'fixed_yc_ys': Fixed Y_C and Y_S (requires hyperons)
    - 'trapped_neutrinos': Trapped neutrinos with fixed Y_Le
    - 'isentropic_beta_eq', 'isentropic_trapped': the same two beta-equilibrium
      modes with entropy per baryon imposed instead of temperature

    Custom parametrization:
        Use custom_params to pass a SFHoParams object directly. Example:

        from eos.sfho.parameters import create_custom_parametrization

        my_params = create_custom_parametrization(
            U_Lambda_N=-28.0, U_Sigma_N=+30.0, U_Xi_N=-18.0,
            name="MyCustom"
        )
        settings = TableSettings(
            custom_params=my_params,
            particle_content='nucleons_hyperons'
        )
    """
    # Model selection
    parametrization: str = 'sfho'        # 'sfho', 'sfhoy', 'sfhoy_star', '2fam_phi'
    particle_content: str = 'nucleons'   # 'nucleons', 'nucleons_hyperons', 'nucleons_hyperons_deltas'
    equilibrium: str = 'beta_eq'         # a key of _LEGACY_EQUILIBRIA
    custom_params: Any = None            # SFHoParams object for custom parametrization

    # Grid definition
    n_B_values: np.ndarray = field(default_factory=lambda: np.logspace(-2, 0, 50) * 0.16)
    T_values: List[float] = field(default_factory=lambda: [10.0])
    S_values: List[float] = field(default_factory=lambda: [1.0])  # Entropy per baryon for isentropic

    # Constraint parameters - can be single values OR arrays for multidimensional tables
    Y_C_values: Union[float, List[float], None] = None
    Y_S_values: Union[float, List[float], None] = None
    Y_L_values: Union[float, List[float], None] = None

    # Options
    include_muons: bool = False
    include_photons: bool = True
    include_electrons: bool = False      # For fixed_yc modes: add electrons for charge neutrality
    include_thermal_neutrinos: bool = False  # Add thermal neutrinos with μ_ν=0
    include_pseudoscalar_mesons: bool = False

    # Output control
    print_results: bool = True
    print_first_n: int = 5
    print_errors: bool = True
    print_timing: bool = True

    # File output
    save_to_file: bool = False
    output_filename: Optional[str] = None
    output_columns: List[str] = field(default_factory=lambda: [
        'n_B', 'T',
        'sigma', 'omega', 'rho', 'phi',
        'mu_B', 'mu_C', 'mu_S', 'mu_e', 'mu_nue',
        'P_total', 'e_total', 's_total',
        'Y_C', 'Y_S', 'Y_Le',
        'converged'
    ])


def _to_list(val):
    """Convert value to list if not None."""
    if val is None:
        return [None]
    if isinstance(val, (list, np.ndarray)):
        return list(val)
    return [val]


def _get_params(settings: TableSettings) -> SFHoParams:
    """Get SFHoParams from settings."""
    if settings.custom_params is not None:
        return settings.custom_params

    param_map = {
        'sfho': get_sfho_nucleonic,
        'sfhoy': get_sfhoy_fortin,
        'sfhoy_star': get_sfhoy_star_fortin,
        '2fam_phi': get_sfho_2fam_phi,
        '2fam': get_sfho_2fam,
    }

    if settings.parametrization.lower() in param_map:
        return param_map[settings.parametrization.lower()]()
    else:
        raise ValueError(f"Unknown parametrization: {settings.parametrization}")


def _get_flags(settings: TableSettings) -> SpeciesFlags:
    """The species flags this table asks for (CLAUDE.md section 4)."""
    content = settings.particle_content.lower()
    known = ('nucleons', 'nucleons_hyperons', 'nucleons_hyperons_deltas')
    if content not in known:
        raise ValueError(f"Unknown particle content: {settings.particle_content}")
    return SpeciesFlags(
        hyperons='hyperons' in content,
        deltas='deltas' in content,
        muons=settings.include_muons,
        thermal_mesons=settings.include_pseudoscalar_mesons,
        thermal_neutrinos=settings.include_thermal_neutrinos,
        photons=settings.include_photons,
    )


def _settings_to_spec(settings: TableSettings) -> Tuple[TableSpec, List[str]]:
    """(TableSpec, the names of the grid axes beyond n_B) for one settings object.

    The names come back because the .dat writer puts the independent variables
    in the leading columns and keys `compute_table`'s result dict by them.
    """
    eq = settings.equilibrium.lower()
    if eq not in _LEGACY_EQUILIBRIA:
        raise ValueError(f"Unknown equilibrium type: {settings.equilibrium}")
    mode, temp_key = _LEGACY_EQUILIBRIA[eq]
    if mode in ("fixed_YC", "fixed_YC_YS") and settings.include_electrons:
        mode += "_neutral"

    grids = {"Y_C": settings.Y_C_values, "Y_S": settings.Y_S_values,
             "Y_Le": settings.Y_L_values}
    axes = {"nB": np.asarray(settings.n_B_values, dtype=float)}
    frac_names = []
    for key in MODE_FRACTIONS[mode]:
        values = _to_list(grids[key])
        if values == [None]:
            raise ValueError(f"equilibrium {settings.equilibrium!r} needs "
                             f"{key} values")
        axes[key] = np.asarray(values, dtype=float)
        frac_names.append(key)
    axes[temp_key] = np.asarray(
        settings.T_values if temp_key == "T" else settings.S_values, dtype=float)

    spec = TableSpec(parametrization=_get_params(settings), mode=mode,
                     axes=axes, include=_get_flags(settings))
    # 'S' rather than 'SnB': the .dat column has been called S since the first
    # tables were written, and files on disk carry that header.
    return spec, frac_names + [temp_key if temp_key == "T" else "S"]


def compute_table(settings: TableSettings) -> Dict[Tuple, List[EoSPoint]]:
    """Solve the grid a `TableSettings` describes.

    An adapter onto `build_table`: same sweep, same warm start, keyed and
    printed the way this interface always was. Returns {grid point -> the n_B
    line solved there}, where a grid point is the tuple of the fractions and
    the temperature (or entropy) that line was solved at, in the order the
    columns are written.
    """
    spec, param_names = _settings_to_spec(settings)
    n_points = len(spec.axes["nB"])

    if settings.print_results:
        params = spec.parametrization
        print("=" * 70)
        print("SFHo EOS TABLE GENERATION")
        print("=" * 70)
        print(f"\nModel: {getattr(params, 'name', settings.parametrization)}")
        print(f"Particles: {settings.particle_content}")
        print(f"Equilibrium: {settings.equilibrium} -> mode {spec.mode!r}")
        print(f"\nDensity grid: {n_points} points")
        print(f"  n_B range: [{spec.axes['nB'][0]:.4e}, "
              f"{spec.axes['nB'][-1]:.4e}] fm^-3")
        print(f"  Parameters: {param_names}")
        print()

    total_start = time.time()
    result = build_table(spec, verbose=settings.print_timing)
    total_elapsed = time.time() - total_start

    all_results = {}
    for (tv, fracs), line in zip(result.combos, result.points):
        key = tuple(fracs[k] for k in param_names[:-1]) + (float(tv),)
        all_results[key] = line
        if settings.print_errors:
            failed = [r.n_B for r in line if not r.converged]
            if failed:
                print(f"  {key}: {len(failed)} points did not converge, "
                      f"n_B from {min(failed):.4e} to {max(failed):.4e}")

    if settings.print_timing:
        n_total = n_points * len(all_results)
        print("\n" + "=" * 70)
        print(f"Total time: {total_elapsed:.2f} s")
        print(f"Average: {total_elapsed * 1000 / n_total:.1f} ms/point")

    if settings.save_to_file:
        save_results(all_results, settings, param_names)

    return all_results


#: .dat column name -> the attribute of `EoSPoint` that fills it. The two
#: differ because the column names are a FORMAT ON DISK -- the published
#: 2fam PNS tables carry these headers and are read back by `load_eos_table`
#: -- while the record now uses the repository's names (P, eps, s).
_COLUMN_ATTR = {
    'P_total': 'P', 'e_total': 'eps', 's_total': 's',
    'f_total': 'free_energy_density',
}


def save_results(all_results: Dict[Tuple, List[EoSPoint]], 
                 settings: TableSettings,
                 param_names: List[str]):
    """Save results to file."""
    params = _get_params(settings)
    
    if settings.output_filename:
        filename = settings.output_filename
    else:
        # Auto-generate filename with all relevant info
        from eos import REPO_ROOT
        filename = str(REPO_ROOT / "sfho_tables_output" / f"eos_{settings.parametrization}_{settings.particle_content}_{settings.equilibrium}.dat")
    
    import os
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    
    with open(filename, 'w') as f:
        f.write(f"# SFHo EOS Table: {settings.parametrization}, {settings.particle_content}\n")
        f.write(f"# Equilibrium: {settings.equilibrium}\n")
        
        # Components included
        components = []
        if settings.include_photons:
            components.append("photons")
        if settings.include_electrons:
            components.append("electrons")
        if settings.include_thermal_neutrinos:
            components.append("thermal_neutrinos")
        if settings.include_pseudoscalar_mesons:
            components.append("pseudoscalar_mesons")
        f.write(f"# Components: {', '.join(components) if components else 'hadrons only'}\n")
        
        # Build column list with independent variables first
        all_columns = list(settings.output_columns)
        
        # Ensure n_B is first
        if 'n_B' in all_columns:
            all_columns.remove('n_B')
        # Ensure T is present
        if 'T' in all_columns:
            all_columns.remove('T')
            
        # Add independent variables at the beginning: n_B, constraint params, T
        ind_vars = ['n_B']
        for pname in param_names:
            if pname not in ['n_B', 'T'] and pname not in ind_vars:
                ind_vars.append(pname)
        ind_vars.append('T')
        
        all_columns = ind_vars + [c for c in all_columns if c not in ind_vars]
        
        f.write("# " + " ".join(f"{col:>14}" for col in all_columns) + "\n")
        
        for grid_param, results in all_results.items():
            param_dict = dict(zip(param_names, grid_param))
            for r in results:
                if r.converged:
                    row = []
                    for col in all_columns:
                        if col in param_dict:
                            val = param_dict[col]
                        elif col == 'Y_C':
                            val = r.Y_C
                        elif col == 'Y_S':
                            val = r.Y_S
                        elif col == 'Y_Le':
                            val = getattr(r, 'Y_Le', 0.0)
                        elif col == 'mu_e':
                            val = getattr(r, 'mu_e', 0.0)
                        elif col == 'mu_C':
                            val = r.mu_C
                        else:
                            val = getattr(r, _COLUMN_ATTR.get(col, col), 0.0)
                        if val is None:
                            val = 0.0
                        if isinstance(val, bool):
                            val = 1 if val else 0
                        row.append(f"{val:>14.6e}" if isinstance(val, float) else f"{val:>14}")
                    f.write(" ".join(row) + "\n")
    
    print(f"\nSaved to: {filename}")


def results_to_arrays(results: List[EoSPoint]) -> Dict[str, np.ndarray]:
    """Convert list of EoSPoint to dictionary of numpy arrays."""
    attrs = [
        'n_B', 'T', 'P_total', 'e_total', 's_total', 'f_total',
        'sigma', 'omega', 'rho', 'phi', 'mu_B', 'mu_C', 'mu_S', 'mu_nue', 'mu_e',
        'Y_C', 'Y_S', 'n_C', 'n_e', 'error'
    ]   # keyed by COLUMN name; see _COLUMN_ATTR
    arrays = {}
    
    # Filter to converged only
    converged_results = [r for r in results if r.converged]
    
    for attr in attrs:
        vals = []
        for r in converged_results:
            val = getattr(r, _COLUMN_ATTR.get(attr, attr), np.nan)
            vals.append(val if val is not None else np.nan)
        arrays[attr] = np.array(vals)
    
    arrays['converged'] = np.array([r.converged for r in results])
    
    return arrays


#==============================================================================
# TABLE LOADING AND INTERPOLATION
#==============================================================================

# Column mappings for each equilibrium type
COLUMN_MAPS = {
    'beta_eq': {
        'n_B': 0, 'T': 1, 'sigma': 2, 'omega': 3, 'rho': 4, 'phi': 5,
        'mu_B': 6, 'mu_C': 7, 'mu_S': 8, 'mu_e': 9, 'mu_nue': 10,
        'P_total': 11, 'e_total': 12, 's_total': 13,
        'Y_C': 14, 'Y_S': 15, 'Y_Le': 16, 'converged': 17
    },
    'trapped_neutrinos': {
        'n_B': 0, 'Y_Le': 1, 'T': 2, 'sigma': 3, 'omega': 4, 'rho': 5, 'phi': 6,
        'mu_B': 7, 'mu_C': 8, 'mu_S': 9, 'mu_e': 10, 'mu_nue': 11,
        'P_total': 12, 'e_total': 13, 's_total': 14,
        'Y_C': 15, 'Y_S': 16, 'converged': 17
    },
    'fixed_yc': {
        'n_B': 0, 'Y_C': 1, 'T': 2, 'sigma': 3, 'omega': 4, 'rho': 5, 'phi': 6,
        'mu_B': 7, 'mu_C': 8, 'mu_S': 9, 'mu_e': 10, 'mu_nue': 11,
        'P_total': 12, 'e_total': 13, 's_total': 14,
        'Y_S': 15, 'Y_Le': 16, 'converged': 17
    },
    'isentropic_beta_eq': {
        'n_B': 0, 'S': 1, 'T': 2, 'sigma': 3, 'omega': 4, 'rho': 5, 'phi': 6,
        'mu_B': 7, 'mu_C': 8, 'mu_S': 9, 'mu_e': 10, 'mu_nue': 11,
        'P_total': 12, 'e_total': 13, 's_total': 14,
        'Y_C': 15, 'Y_S': 16, 'Y_Le': 17, 'converged': 18
    },
    'isentropic_trapped': {
        'n_B': 0, 'Y_Le': 1, 'S': 2, 'T': 3, 'sigma': 4, 'omega': 5, 'rho': 6, 'phi': 7,
        'mu_B': 8, 'mu_C': 9, 'mu_S': 10, 'mu_e': 11, 'mu_nue': 12,
        'P_total': 13, 'e_total': 14, 's_total': 15,
        'Y_C': 16, 'Y_S': 17, 'converged': 18
    },
}

# Grid axes for each equilibrium type (order matters for reshaping)
GRID_AXES = {
    'beta_eq': ['n_B', 'T'],
    'trapped_neutrinos': ['n_B', 'Y_Le', 'T'],
    'fixed_yc': ['n_B', 'Y_C', 'T'],
    'isentropic_beta_eq': ['n_B', 'S'],
    'isentropic_trapped': ['n_B', 'Y_Le', 'S'],
}


@dataclass
class EOSTableData:
    """Container for loaded EOS table with structured grids."""
    eq_type: str
    grids: Dict[str, np.ndarray]      # {'n_B': array, 'T': array, ...}
    data: Dict[str, np.ndarray]       # {'P_total': 2D/3D array, ...}
    filepath: str = ""

    def __repr__(self):
        axes = list(self.grids.keys())
        shapes = [f"{k}={len(v)}" for k, v in self.grids.items()]
        return f"EOSTableData(eq_type='{self.eq_type}', axes={axes}, shape=({', '.join(shapes)}))"


def load_eos_table(filepath: str, eq_type: str) -> EOSTableData:
    """
    Load an EOS table from file and return structured grids.

    Parameters:
        filepath: Path to the .dat file
        eq_type: Equilibrium type - 'beta_eq', 'trapped_neutrinos', 'fixed_yc',
                 'isentropic_beta_eq', or 'isentropic_trapped'

    Returns:
        EOSTableData with:
        - grids: dict of 1D arrays for each axis (n_B, T, Y_Le, etc.)
        - data: dict of N-dimensional arrays for each quantity
                e.g., data['P_total'][i, j] for beta_eq (2D)
                      data['P_total'][i, j, k] for trapped (3D)

    Example:
        >>> table = load_eos_table('eos_betaeq.dat', 'beta_eq')
        >>> P = table.data['P_total']  # Shape: (n_nB, n_T)
        >>> nB = table.grids['n_B']    # 1D array of n_B values
        >>> T = table.grids['T']       # 1D array of T values
        >>> print(f"P(n_B[10], T[5]) = {P[10, 5]} MeV/fm^3")
    """
    if eq_type not in COLUMN_MAPS:
        raise ValueError(f"Unknown eq_type: {eq_type}. "
                        f"Valid options: {list(COLUMN_MAPS.keys())}")

    col_map = COLUMN_MAPS[eq_type]
    axes = GRID_AXES[eq_type]

    # Load raw data
    raw_data = np.loadtxt(filepath, comments='#')
    print(f"Loaded {len(raw_data)} points from {filepath}")

    # Extract unique grid values for each axis
    grids = {}
    for axis in axes:
        grids[axis] = np.unique(raw_data[:, col_map[axis]])

    # Determine grid shape
    shape = tuple(len(grids[axis]) for axis in axes)
    n_points_expected = np.prod(shape)

    if len(raw_data) != n_points_expected:
        print(f"  Warning: Expected {n_points_expected} points for complete grid, got {len(raw_data)}")

    # Columns to extract (exclude grid axes and converged flag)
    exclude = set(axes) | {'converged'}
    columns = [c for c in col_map.keys() if c not in exclude]

    # Build structured arrays using vectorized approach
    data = {}

    # Create index mapping: for each raw data row, find its grid indices
    indices = []
    for axis in axes:
        axis_values = raw_data[:, col_map[axis]]
        grid_values = grids[axis]
        # Find index of each value in the grid
        idx = np.searchsorted(grid_values, axis_values)
        indices.append(idx)
    indices = tuple(indices)

    # Fill in data arrays
    for col in columns:
        arr = np.full(shape, np.nan)
        arr[indices] = raw_data[:, col_map[col]]
        data[col] = arr

    # Add derived quantities
    # f_total = e_total - T * s_total (free energy density)
    if 'e_total' in data and 's_total' in data:
        if eq_type in ['beta_eq', 'trapped_neutrinos', 'fixed_yc']:
            # T is one of the axes
            T_idx = axes.index('T')
            T_broadcast_shape = [1] * len(axes)
            T_broadcast_shape[T_idx] = len(grids['T'])
            T_grid = grids['T'].reshape(T_broadcast_shape)
            data['f_total'] = data['e_total'] - T_grid * data['s_total']
        elif 'T' in data:
            # T is computed (isentropic cases)
            data['f_total'] = data['e_total'] - data['T'] * data['s_total']

    # mu_nue is read straight from its column. It used to be reconstructed
    # here as mu_e + mu_nu, which was wrong twice over: the relation is
    # mu_nue = mu_e + mu_C, and the mu_nu column was written from a result
    # field the solvers never set, so it was a column of zeros. Tables written
    # before that fix carry zeros in this column.

    # Print summary
    print(f"  Equilibrium: {eq_type}")
    for axis in axes:
        print(f"  {axis}: [{grids[axis][0]:.4g}, {grids[axis][-1]:.4g}], {len(grids[axis])} points")

    return EOSTableData(
        eq_type=eq_type,
        grids=grids,
        data=data,
        filepath=filepath
    )


class _EdgeSnapped:
    """RegularGridInterpolator that absorbs float round-trip noise at the edges.

    Why this exists. The grid axes are written to the .dat file with '%e', i.e.
    7 significant digits, so a node rebuilt in memory need not be bit-identical
    to its on-disk twin. The classic offender is the last node of an arange:

        np.arange(0.10, 0.401, 0.05)[-1] == 0.40000000000000013

    while the file reads back exactly 0.4. Asking for Y_Le there is asking for a
    point *outside* the interpolation domain, and RegularGridInterpolator then
    fills the ENTIRE query with NaN -- one ULP silently blanks a whole EoS
    table, and the failure only surfaces much later (e.g. "EOS has 0 finite
    rows" when the crust is spliced on).

    Mechanics. Before delegating, every coordinate that sits within `rtol` of an
    axis endpoint is snapped onto that endpoint. This carries no physics: it
    moves a query by at most a part in 1e9. A coordinate that misses the grid by
    more than that (T = 200 MeV on a grid stopping at 100) is left alone and
    still returns NaN, because that one is a genuine out-of-range request.
    """

    __slots__ = ('_rgi', '_lo', '_hi', '_tol')

    def __init__(self, rgi, grids, rtol: float = 1e-9):
        self._rgi = rgi
        # Axes are sorted ascending by load_eos_table, so [0] / [-1] are the ends.
        self._lo = [float(g[0]) for g in grids]
        self._hi = [float(g[-1]) for g in grids]
        self._tol = [rtol * max(abs(lo), abs(hi))
                     for lo, hi in zip(self._lo, self._hi)]

    def __call__(self, xi):
        # Only the tuple-of-coordinates call style is snapped; the (n, ndim)
        # array style is passed straight through (nothing here uses it).
        if not isinstance(xi, (tuple, list)):
            return self._rgi(xi)
        return self._rgi(tuple(
            self._snap(c, lo, hi, tol)
            for c, lo, hi, tol in zip(xi, self._lo, self._hi, self._tol)))

    @staticmethod
    def _snap(c, lo, hi, tol):
        c = np.asarray(c, dtype=float)
        c = np.where(np.abs(c - lo) <= tol, lo, c)
        return np.where(np.abs(c - hi) <= tol, hi, c)


def build_interpolators(table: EOSTableData,
                        method: str = 'linear',
                        bounds_error: bool = False,
                        fill_value: float = np.nan) -> Dict[str, Any]:
    """
    Build interpolation functions from loaded EOS table data.

    Parameters:
        table: EOSTableData from load_eos_table()
        method: Interpolation method ('linear', 'nearest', 'cubic', etc.)
        bounds_error: If True, raise error for out-of-bounds queries
        fill_value: Value to return for out-of-bounds queries

    Returns:
        Dict with:
        - 'interpolators': dict of RegularGridInterpolator for each quantity
        - 'grids': reference to the grid arrays
        - 'axes': list of axis names in order
        - Convenience functions for common quantities

    Example:
        >>> table = load_eos_table('eos_betaeq.dat', 'beta_eq')
        >>> interp = build_interpolators(table)
        >>>
        >>> # Using interpolators directly
        >>> P = interp['interpolators']['P_total']((0.16, 10.0))
        >>>
        >>> # Using convenience functions (beta_eq)
        >>> P = interp['P'](0.16, 10.0)
        >>> eps = interp['eps'](0.16, 10.0)
        >>>
        >>> # For trapped neutrinos (3 arguments)
        >>> P = interp['P'](0.16, 0.4, 50.0)  # n_B, Y_Le, T
    """
    from scipy.interpolate import RegularGridInterpolator

    axes = GRID_AXES[table.eq_type]
    grid_tuple = tuple(table.grids[axis] for axis in axes)

    interpolators = {}
    for name, arr in table.data.items():
        interpolators[name] = _EdgeSnapped(RegularGridInterpolator(
            grid_tuple, arr,
            method=method,
            bounds_error=bounds_error,
            fill_value=fill_value
        ), grid_tuple)

    result = {
        'interpolators': interpolators,
        'grids': table.grids,
        'axes': axes,
        'eq_type': table.eq_type,
    }

    # Add convenience functions based on equilibrium type
    if table.eq_type == 'beta_eq':
        # 2D: f(n_B, T)
        result['P'] = lambda nB, T: interpolators['P_total']((nB, T))
        result['eps'] = lambda nB, T: interpolators['e_total']((nB, T))
        result['s'] = lambda nB, T: interpolators['s_total']((nB, T))
        result['f'] = lambda nB, T: interpolators['f_total']((nB, T))
        result['mu_B'] = lambda nB, T: interpolators['mu_B']((nB, T))
        result['mu_C'] = lambda nB, T: interpolators['mu_C']((nB, T))
        result['mu_S'] = lambda nB, T: interpolators['mu_S']((nB, T))
        result['mu_e'] = lambda nB, T: interpolators['mu_e']((nB, T))
        result['Y_C'] = lambda nB, T: interpolators['Y_C']((nB, T))
        result['Y_S'] = lambda nB, T: interpolators['Y_S']((nB, T))

    elif table.eq_type == 'trapped_neutrinos':
        # 3D: f(n_B, Y_Le, T)
        result['P'] = lambda nB, YL, T: interpolators['P_total']((nB, YL, T))
        result['eps'] = lambda nB, YL, T: interpolators['e_total']((nB, YL, T))
        result['s'] = lambda nB, YL, T: interpolators['s_total']((nB, YL, T))
        result['f'] = lambda nB, YL, T: interpolators['f_total']((nB, YL, T))
        result['mu_B'] = lambda nB, YL, T: interpolators['mu_B']((nB, YL, T))
        result['mu_C'] = lambda nB, YL, T: interpolators['mu_C']((nB, YL, T))
        result['mu_S'] = lambda nB, YL, T: interpolators['mu_S']((nB, YL, T))
        result['mu_nue'] = lambda nB, YL, T: interpolators['mu_nue']((nB, YL, T))
        result['mu_e'] = lambda nB, YL, T: interpolators['mu_e']((nB, YL, T))
        result['Y_C'] = lambda nB, YL, T: interpolators['Y_C']((nB, YL, T))
        result['Y_S'] = lambda nB, YL, T: interpolators['Y_S']((nB, YL, T))

    elif table.eq_type == 'fixed_yc':
        # 3D: f(n_B, Y_C, T)
        result['P'] = lambda nB, YC, T: interpolators['P_total']((nB, YC, T))
        result['eps'] = lambda nB, YC, T: interpolators['e_total']((nB, YC, T))
        result['s'] = lambda nB, YC, T: interpolators['s_total']((nB, YC, T))
        result['f'] = lambda nB, YC, T: interpolators['f_total']((nB, YC, T))
        result['mu_B'] = lambda nB, YC, T: interpolators['mu_B']((nB, YC, T))
        result['mu_C'] = lambda nB, YC, T: interpolators['mu_C']((nB, YC, T))
        result['mu_S'] = lambda nB, YC, T: interpolators['mu_S']((nB, YC, T))
        result['mu_e'] = lambda nB, YC, T: interpolators['mu_e']((nB, YC, T))
        result['Y_S'] = lambda nB, YC, T: interpolators['Y_S']((nB, YC, T))

    elif table.eq_type == 'isentropic_beta_eq':
        # 2D: f(n_B, S)
        result['P'] = lambda nB, S: interpolators['P_total']((nB, S))
        result['eps'] = lambda nB, S: interpolators['e_total']((nB, S))
        result['s'] = lambda nB, S: interpolators['s_total']((nB, S))
        result['T'] = lambda nB, S: interpolators['T']((nB, S))
        result['f'] = lambda nB, S: interpolators['f_total']((nB, S))
        result['mu_B'] = lambda nB, S: interpolators['mu_B']((nB, S))
        result['mu_C'] = lambda nB, S: interpolators['mu_C']((nB, S))
        result['mu_S'] = lambda nB, S: interpolators['mu_S']((nB, S))
        result['mu_e'] = lambda nB, S: interpolators['mu_e']((nB, S))
        result['mu_nue'] = lambda nB, S: interpolators['mu_nue']((nB, S))
        result['Y_C'] = lambda nB, S: interpolators['Y_C']((nB, S))
        result['Y_S'] = lambda nB, S: interpolators['Y_S']((nB, S))

    elif table.eq_type == 'isentropic_trapped':
        # 3D: f(n_B, Y_Le, S)
        result['P'] = lambda nB, YL, S: interpolators['P_total']((nB, YL, S))
        result['eps'] = lambda nB, YL, S: interpolators['e_total']((nB, YL, S))
        result['s'] = lambda nB, YL, S: interpolators['s_total']((nB, YL, S))
        result['T'] = lambda nB, YL, S: interpolators['T']((nB, YL, S))
        result['f'] = lambda nB, YL, S: interpolators['f_total']((nB, YL, S))
        result['mu_B'] = lambda nB, YL, S: interpolators['mu_B']((nB, YL, S))
        result['mu_C'] = lambda nB, YL, S: interpolators['mu_C']((nB, YL, S))
        result['mu_S'] = lambda nB, YL, S: interpolators['mu_S']((nB, YL, S))
        result['mu_e'] = lambda nB, YL, S: interpolators['mu_e']((nB, YL, S))
        result['mu_nue'] = lambda nB, YL, S: interpolators['mu_nue']((nB, YL, S))
        result['Y_C'] = lambda nB, YL, S: interpolators['Y_C']((nB, YL, S))
        result['Y_S'] = lambda nB, YL, S: interpolators['Y_S']((nB, YL, S))

    return result


#==============================================================================
# MAIN
#==============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("SFHo EOS TABLE GENERATOR")
    print("=" * 70 + "\n")

    spec = TableSpec(
        parametrization=get_sfho_2fam_phi(),
        mode="beta_eq_neutrinoless",
        axes={"nB": np.linspace(0.1, 10, 40) * 0.1583, "T": [0.0, 10.0]},
        include=SpeciesFlags(),
    )
    result = build_table(spec, verbose=True)
    for (T, _), line in zip(result.combos, result.points):
        P = [r.P for r in line if r.converged]
        print(f"T={T:5.1f} MeV: {len(P)}/{len(spec.axes['nB'])} converged, "
              f"P in [{min(P):.4e}, {max(P):.4e}] MeV/fm^3")
    print("=" * 70 + "\n")
