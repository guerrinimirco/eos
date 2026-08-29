"""
SFHo nonlinear relativistic mean-field equation-of-state engine.

Nucleons, hyperons and Delta isobars in a nonlinear RMF with the
sigma-omega-rho cross coupling A(sigma, omega), plus leptons, photons and an
optional thermal pi/K/eta gas, at zero and finite temperature. Which degrees of
freedom are active is declared explicitly through `SpeciesFlags`; the
equilibrium condition is chosen through the mode names in `table.py`.

SFHo carries no `couplings.py`: its meson-baryon couplings are constants, and
its density dependence is the nonlinear self-interaction of the fields, which
is thermodynamics.

References:
- Steiner, Hempel & Fischer, ApJ 774 (2013) 17
- Fortin, Oertel & Providencia, PASA 35 (2018) e044
"""
from eos.sfho.parameters import Parameters, PUBLISHED_SETS
from eos.sfho.species import (
    SpeciesFlags, active_baryons, check_couplings,
)
from eos.sfho.solver import (
    EoSPoint, warm_start, default_guess, unknown_names,
    solve_mode, solve_beta_eq_neutrinoless, solve_beta_eq_neutrino_trapped,
    solve_fixed_yc, solve_fixed_yc_ys,
)
from eos.sfho.table import (
    TableSpec, TableResult, build_table, mode_spec,
    MODES, MODE_FRACTIONS, hadronic_row, rows_from_result,
    TableSettings, compute_table, save_results, results_to_arrays,
    load_eos_table, build_interpolators, EOSTableData,
)
from eos.sfho.nmp import (
    compute_nmp, snm_derivatives, energy_per_baryon, pressure, esym,
    from_potential_depths, compute_saturation_fields,
    compute_hyperon_potentials, PUBLISHED_NMP, PUBLISHED_NMP_EXACT,
)
from eos.sfho.api import (
    eos_point, eos_table, eos_response, PointResult, RESPONSE_FREEZES,
)

__all__ = [
    "Parameters", "PUBLISHED_SETS", "SpeciesFlags", "active_baryons", "check_couplings",
    "EoSPoint", "warm_start", "default_guess", "unknown_names",
    "solve_mode", "solve_beta_eq_neutrinoless", "solve_beta_eq_neutrino_trapped",
    "solve_fixed_yc", "solve_fixed_yc_ys",
    "TableSpec", "TableResult", "build_table", "mode_spec",
    "MODES", "MODE_FRACTIONS", "hadronic_row", "rows_from_result",
    "TableSettings", "compute_table", "save_results", "results_to_arrays",
    "load_eos_table", "build_interpolators", "EOSTableData",
    "compute_nmp", "snm_derivatives", "energy_per_baryon", "pressure", "esym",
    "from_potential_depths", "compute_saturation_fields",
    "compute_hyperon_potentials", "PUBLISHED_NMP", "PUBLISHED_NMP_EXACT",
    "eos_point", "eos_table", "eos_response", "PointResult",
    "RESPONSE_FREEZES",
]
