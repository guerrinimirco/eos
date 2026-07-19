"""
DD2 density-dependent RMF equation-of-state engine (Phase 1).

Physics specification: DD2_EoS_Physics_Report.md; executable specification
for M0–M2: dd2_reference_validation.py.
"""
from eos.dd2.parametrization import Parametrization
from eos.dd2.species import SpeciesFlags, active_baryons
from eos.dd2.solver import (
    EoSPoint, beta_warm_start, octet_warm_start,
    solve_beta_eq, solve_beta_eq_t0,
    solve_octet, solve_beta_eq_octet, solve_fixed_yc_octet, solve_yl_octet,
    sweep_beta_eq_octet, sweep_octet,
    solve_composition, solve_composition_t0,
    solve_snm, solve_snm_t0,
)
from eos.dd2.table import (
    TableSpec, TableResult, build_table, solve_octet_at_entropy,
)
from eos.dd2.physics.jacobian import octet_jacobian, kinetic_derivs
from eos.dd2.nmp import compute_nmp, energy_per_baryon, esym
from eos.dd2.nmp_inverter import invert_nmp, InversionStatus
from eos.dd2.coefficients import (
    sound_speed_eq, sound_speed_adiabatic, adiabatic_index, thermal_index,
    heat_capacity_V, snm_sound_speed,
)

__all__ = [
    "Parametrization", "SpeciesFlags", "active_baryons",
    "invert_nmp", "InversionStatus",
    "EoSPoint", "beta_warm_start", "octet_warm_start",
    "solve_beta_eq", "solve_beta_eq_t0",
    "solve_octet", "solve_beta_eq_octet", "solve_fixed_yc_octet",
    "solve_yl_octet", "sweep_beta_eq_octet", "sweep_octet",
    "solve_composition", "solve_composition_t0",
    "solve_snm", "solve_snm_t0",
    "TableSpec", "TableResult", "build_table", "solve_octet_at_entropy",
    "octet_jacobian", "kinetic_derivs",
    "compute_nmp", "energy_per_baryon", "esym",
    "sound_speed_eq", "sound_speed_adiabatic", "adiabatic_index",
    "thermal_index", "heat_capacity_V", "snm_sound_speed",
]
