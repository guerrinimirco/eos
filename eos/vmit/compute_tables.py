"""Settings-object table generation, kept for the ZLvMIT notebook.

`VMITTableSettings` + `compute_vmit_table` are the first-generation interface:
one object carrying the grid, the mode and the printing options, returning a
dict keyed by the grid point. New code should use `eos.vmit.eos_table` (or
`eos.vmit.table.build_table`), which takes the repository's mode names and
reports through the progress callback every model shares.

Nothing is solved here. The sweep, the warm start and the timing are
`eos.general.tabulate` by way of `eos.vmit.table`; this module only translates
names and reshapes the result. The writer below stays because the on-disk
column layout is read by the notebook.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

from eos.vmit.solver import EoSPoint
from eos.vmit.parameters import Parameters
from eos.vmit.species import SpeciesFlags
from eos.vmit.table import TableSpec, build_table

#: The legacy equilibrium names, and the repository mode each one means.
_LEGACY_MODES = {
    'beta_eq': "beta_eq_neutrinoless",
    'fixed_yc': "fixed_YC",
    'fixed_yc_ys': "fixed_YC_YS",
    'trapped_neutrinos': "beta_eq_neutrino_trapped",
}

#: The legacy name of each mode's fraction axes, and the spec name it maps to.
_LEGACY_FRACTIONS = {
    'beta_eq': (),
    'fixed_yc': (("Y_C_values", "Y_C"),),
    'fixed_yc_ys': (("Y_C_values", "Y_C"), ("Y_S_values", "Y_S")),
    'trapped_neutrinos': (("Y_L_values", "Y_Le"),),
}


@dataclass
class VMITTableSettings:
    """Configuration for a vMIT table.

    equilibrium: 'beta_eq', 'fixed_yc', 'fixed_yc_ys' or 'trapped_neutrinos'.
    """
    # Model parameters
    params: Optional[Parameters] = None  # None = use default

    # Equilibrium type
    equilibrium: str = 'beta_eq'

    # Grid definition
    n_B_values: np.ndarray = field(default_factory=lambda: np.linspace(0.1, 10, 100) * 0.16)
    T_values: List[float] = field(default_factory=lambda: [10.0])

    # Constraint parameters (depending on equilibrium mode)
    Y_C_values: List[float] = field(default_factory=lambda: [0.0])  # For fixed_yc
    Y_S_values: List[float] = field(default_factory=lambda: [0.0])  # For fixed_yc_ys
    Y_L_values: List[float] = field(default_factory=lambda: [0.4])  # For trapped_neutrinos

    # Options
    include_photons: bool = True
    include_leptons: bool = True  # Only for fixed_yc/fixed_yc_ys: include electrons

    # Output control
    print_results: bool = True
    print_first_n: int = 5
    print_errors: bool = True
    print_timing: bool = True

    # File output
    save_to_file: bool = False
    output_filename: Optional[str] = None  # Auto-generate if None


def compute_vmit_table(settings: VMITTableSettings) -> Dict[Tuple, List[EoSPoint]]:
    """Solve the settings grid; returns {(T, [fractions...]): [results]}.

    The key of each line is the tuple of its grid values in the legacy order
    (T first, then the mode's fractions), and lines are ordered the way nested
    loops over those axes would produce them.
    """
    eq_type = settings.equilibrium.lower()
    if eq_type not in _LEGACY_MODES:
        raise ValueError(f"Unknown equilibrium type: {eq_type}")

    params = settings.params if settings.params is not None else Parameters.default()
    frac_axes = _LEGACY_FRACTIONS[eq_type]

    axes = {"nB": np.asarray(settings.n_B_values),
            "T": np.atleast_1d(np.asarray(settings.T_values, dtype=float))}
    for legacy_name, spec_name in frac_axes:
        axes[spec_name] = np.atleast_1d(
            np.asarray(getattr(settings, legacy_name), dtype=float))

    spec = TableSpec(params=params, mode=_LEGACY_MODES[eq_type], axes=axes,
                     include=SpeciesFlags(photons=settings.include_photons),
                     leptons=settings.include_leptons)
    result = build_table(spec, skip_errors=False,
                         verbose=settings.print_results or settings.print_timing)

    all_results = {}
    for conditions, line in zip(result.lines, result.points):
        key = tuple([conditions["T"]]
                    + [conditions[spec_name] for _, spec_name in frac_axes])
        all_results[key] = line

    if settings.save_to_file:
        save_vmit_results(all_results, settings, params, eq_type)

    return all_results


def save_vmit_results(
    all_results: Dict[Tuple, List[EoSPoint]],
    settings: VMITTableSettings,
    params: Parameters,
    eq_type: str
):
    """Write the solved lines out in the legacy column layout."""
    if settings.output_filename:
        filename = settings.output_filename
    else:
        from eos import REPO_ROOT
        filename = str(REPO_ROOT / "output" / f"vmit_B{int(params.B4)}_a{params.a}_{eq_type}.dat")

    import os
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w') as f:
        f.write(f"# vMIT EOS Table: {params.name}\n")
        f.write(f"# Parameters: B^1/4={params.B4} MeV, a={params.a} fm^2\n")
        f.write(f"# Equilibrium: {eq_type}\n")

        # Format: inputs first, then mu, then Y, then thermodynamics
        if eq_type == 'fixed_yc':
            columns = ['n_B', 'Y_C', 'T', 'mu_u', 'mu_d', 'mu_s', 'mu_e', 'Y_u', 'Y_d', 'Y_s', 'Y_e',
                       'P_total', 'e_total', 's_total', 'f_total', 'converged']
        elif eq_type == 'fixed_yc_ys':
            columns = ['n_B', 'Y_C', 'Y_S', 'T', 'mu_u', 'mu_d', 'mu_s', 'mu_e', 'Y_u', 'Y_d', 'Y_s', 'Y_e',
                       'P_total', 'e_total', 's_total', 'f_total', 'converged']
        elif eq_type == 'trapped_neutrinos':
            columns = ['n_B', 'Y_L', 'T', 'mu_u', 'mu_d', 'mu_s', 'mu_e', 'mu_nu', 'Y_u', 'Y_d', 'Y_s', 'Y_e', 'Y_nu',
                       'P_total', 'e_total', 's_total', 'f_total', 'converged']
        else:  # beta_eq
            columns = ['n_B', 'T', 'mu_u', 'mu_d', 'mu_s', 'mu_e', 'Y_u', 'Y_d', 'Y_s', 'Y_e',
                       'P_total', 'e_total', 's_total', 'f_total', 'converged']

        f.write("# " + " ".join(f"{col:>14}" for col in columns) + "\n")

        for params_tuple, results in all_results.items():
            # Get Y_C, Y_S, Y_L from grid params depending on mode
            Y_C_val = params_tuple[1] if eq_type in ('fixed_yc', 'fixed_yc_ys') and len(params_tuple) > 1 else 0.0
            Y_S_val = params_tuple[2] if eq_type == 'fixed_yc_ys' and len(params_tuple) > 2 else 0.0
            Y_L_val = params_tuple[1] if eq_type == 'trapped_neutrinos' and len(params_tuple) > 1 else 0.0

            for r in results:
                if r.converged:
                    f_total = r.e_total - r.s_total * r.T

                    if eq_type == 'fixed_yc':
                        row = [r.n_B, Y_C_val, r.T, r.mu_u, r.mu_d, r.mu_s, r.mu_e,
                               r.Y_u, r.Y_d, r.Y_s, r.Y_e, r.P_total, r.e_total, r.s_total, f_total, 1]
                    elif eq_type == 'fixed_yc_ys':
                        row = [r.n_B, Y_C_val, Y_S_val, r.T, r.mu_u, r.mu_d, r.mu_s, r.mu_e,
                               r.Y_u, r.Y_d, r.Y_s, r.Y_e, r.P_total, r.e_total, r.s_total, f_total, 1]
                    elif eq_type == 'trapped_neutrinos':
                        Y_nu = r.n_nu / r.n_B if r.n_B > 0 else 0.0
                        row = [r.n_B, Y_L_val, r.T, r.mu_u, r.mu_d, r.mu_s, r.mu_e, r.mu_nu,
                               r.Y_u, r.Y_d, r.Y_s, r.Y_e, Y_nu, r.P_total, r.e_total, r.s_total, f_total, 1]
                    else:  # beta_eq
                        row = [r.n_B, r.T, r.mu_u, r.mu_d, r.mu_s, r.mu_e,
                               r.Y_u, r.Y_d, r.Y_s, r.Y_e, r.P_total, r.e_total, r.s_total, f_total, 1]
                    f.write(" ".join(f"{v:>14.6e}" if isinstance(v, float) else f"{v:>14}" for v in row) + "\n")

    print(f"\nSaved to: {filename}")


def results_to_arrays(results: List[EoSPoint]) -> Dict[str, np.ndarray]:
    """Converged points of one line as {quantity: array}, ready for plotting."""
    input_attrs = ['n_B', 'T', 'Y_C', 'Y_S', 'Y_L']
    mu_attrs = ['mu_u', 'mu_d', 'mu_s', 'mu_e', 'mu_nu']
    Y_attrs = ['Y_u', 'Y_d', 'Y_s', 'Y_e', 'Y_nu']
    thermo_attrs = ['P_total', 'e_total', 's_total', 'f_total']
    other_attrs = ['error']

    all_attrs = input_attrs + mu_attrs + Y_attrs + thermo_attrs + other_attrs

    arrays = {}
    for attr in all_attrs:
        try:
            arrays[attr] = np.array([getattr(r, attr) for r in results if r.converged])
        except AttributeError:
            # Skip attributes that don't exist (e.g. mu_nu for non-trapped mode)
            pass

    # f = e - Ts, if the result did not carry it
    if 'f_total' not in arrays and 'e_total' in arrays and 's_total' in arrays:
        T_arr = np.array([r.T for r in results if r.converged])
        arrays['f_total'] = arrays['e_total'] - arrays['s_total'] * T_arr

    arrays['converged'] = np.array([r.converged for r in results])
    return arrays
