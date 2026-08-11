"""
DD2 density-dependent relativistic mean-field equation-of-state engine.

Nucleons, hyperons and Delta isobars in a density-dependent RMF, with leptons,
photons and an optional thermal meson gas, at zero and finite temperature.
Which degrees of freedom are active is declared explicitly through
`SpeciesFlags`; the equilibrium condition is chosen through the mode names in
`table.py`. See `eos.mixed` for the hybrid hadron-quark extension.
"""
from eos.dd2.parameters import Parametrization
from eos.dd2.species import (
    SpeciesFlags, active_baryons, hadronic_qn, hadronic_charges,
)
from eos.dd2.solver import (
    EoSPoint, beta_warm_start, octet_warm_start,
    solve_beta_eq, solve_beta_eq_t0,
    solve_octet, solve_beta_eq_octet, solve_fixed_yc_octet, solve_yl_octet,
    solve_hadronic,
    sweep_beta_eq_octet, sweep_octet,
    solve_composition, solve_composition_t0,
    solve_snm, solve_snm_t0,
)
from eos.dd2.table import (
    TableSpec, TableResult, build_table, solve_octet_at_entropy,
    MODES, MODE_FRACTIONS, hadronic_row, rows_from_result,
)
from eos.general.table_io import save_table, load_table, export_csv
from eos.dd2.physics.jacobian import octet_jacobian, kinetic_derivs
from eos.dd2.nmp import compute_nmp, energy_per_baryon, esym
from eos.dd2.nmp import invert_nmp, from_nmp, InversionStatus
from eos.dd2.responses import (
    sound_speed_eq, sound_speed_adiabatic, adiabatic_index, thermal_index,
    heat_capacity_V, snm_sound_speed,
)
from eos.dd2.responses_jac import susceptibilities, SUSCEPT_LABELS
from eos.dd2.api import (
    eos_point, eos_table, eos_response, PointResult,
    RESPONSE_FREEZES,
)

__all__ = [
    "Parametrization", "SpeciesFlags", "active_baryons",
    "hadronic_qn", "hadronic_charges",
    "invert_nmp", "from_nmp", "InversionStatus",
    "EoSPoint", "beta_warm_start", "octet_warm_start",
    "solve_beta_eq", "solve_beta_eq_t0",
    "solve_octet", "solve_beta_eq_octet", "solve_fixed_yc_octet",
    "solve_yl_octet", "solve_hadronic", "sweep_beta_eq_octet", "sweep_octet",
    "solve_composition", "solve_composition_t0",
    "solve_snm", "solve_snm_t0",
    "TableSpec", "TableResult", "build_table", "solve_octet_at_entropy",
    "MODES", "MODE_FRACTIONS", "hadronic_row", "rows_from_result",
    "save_table", "load_table", "export_csv",
    "octet_jacobian", "kinetic_derivs",
    "compute_nmp", "energy_per_baryon", "esym",
    "sound_speed_eq", "sound_speed_adiabatic", "adiabatic_index",
    "thermal_index", "heat_capacity_V", "snm_sound_speed",
    "susceptibilities", "SUSCEPT_LABELS",
]
