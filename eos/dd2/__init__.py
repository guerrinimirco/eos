"""
DD2 density-dependent relativistic mean-field equation-of-state engine.

Nucleons, hyperons and Delta isobars in a density-dependent RMF, with leptons,
photons and an optional thermal meson gas, at zero and finite temperature.
Which degrees of freedom are active is declared explicitly through
`SpeciesFlags`; the equilibrium condition is chosen through the mode names in
`table.py`. See `eos.mixed` for the hybrid hadron-quark extension.
"""
from eos.dd2.parameters import Parameters
from eos.dd2.species import (
    SpeciesFlags, active_baryons, hadronic_qn, hadronic_charges,
)
from eos.dd2.solver import (
    EoSPoint, nucleon_warm_start, warm_start,
    solve_beta_eq, solve_beta_eq_t0,
    solve, solve_beta_eq_neutrinoless, solve_fixed_yc,
    solve_beta_eq_neutrino_trapped,
    solve_hadronic,
    sweep,
    solve_composition, solve_composition_t0,
    solve_snm, solve_snm_t0,
)
from eos.dd2.solver import MODES, MODE_FRACTIONS
from eos.dd2.table import (
    TableSpec, TableResult, build_table, solve_at_entropy,
    hadronic_row, rows_from_result,
)
from eos.general.table_io import save_table, load_table, export_csv
from eos.dd2.nmp import compute_nmp, energy_per_baryon, esym
from eos.dd2.nmp import invert_nmp, from_nmp, InversionStatus
from eos.dd2.nmp import from_hyperon_potentials, from_delta_potential
from eos.dd2.nmp import build_parametrization, SECTOR_KEYS
from eos.dd2.responses import (
    sound_speed_eq, sound_speed_isothermal_frozen,
    sound_speed_adiabatic_frozen, adiabatic_index, thermal_index,
    heat_capacity_V, snm_sound_speed,
)
# Nothing from `backends/` is re-exported here. CLAUDE.md section 5 defines
# that package by the property that deleting it changes no number, and a name
# on the package surface would make `import eos.dd2` fail instead. Reach the
# analytic Jacobian and its susceptibilities at eos.dd2.backends.jacobian and
# eos.dd2.backends.responses_jac, which is where a reader looking for them
# would go.
from eos.dd2.api import (
    eos_point, eos_table, eos_response, PointResult,
    RESPONSE_FREEZES,
)

__all__ = [
    "Parameters", "SpeciesFlags", "active_baryons",
    "hadronic_qn", "hadronic_charges",
    "invert_nmp", "from_nmp", "InversionStatus",
    "from_hyperon_potentials", "from_delta_potential",
    "build_parametrization", "SECTOR_KEYS",
    "EoSPoint", "nucleon_warm_start", "warm_start",
    "solve_beta_eq", "solve_beta_eq_t0",
    "solve", "solve_beta_eq_neutrinoless", "solve_fixed_yc",
    "solve_beta_eq_neutrino_trapped", "solve_hadronic", "sweep",
    "solve_composition", "solve_composition_t0",
    "solve_snm", "solve_snm_t0",
    "TableSpec", "TableResult", "build_table", "solve_at_entropy",
    "MODES", "MODE_FRACTIONS", "hadronic_row", "rows_from_result",
    "save_table", "load_table", "export_csv",
    "compute_nmp", "energy_per_baryon", "esym",
    "sound_speed_eq", "sound_speed_isothermal_frozen",
    "sound_speed_adiabatic_frozen", "adiabatic_index",
    "thermal_index", "heat_capacity_V", "snm_sound_speed",
]
