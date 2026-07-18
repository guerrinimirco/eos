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
    solve_beta_eq_octet, sweep_beta_eq_octet,
    solve_composition, solve_composition_t0,
    solve_snm, solve_snm_t0,
)
from eos.dd2.nmp import compute_nmp, energy_per_baryon, esym

__all__ = [
    "Parametrization", "SpeciesFlags", "active_baryons",
    "EoSPoint", "beta_warm_start", "octet_warm_start",
    "solve_beta_eq", "solve_beta_eq_t0",
    "solve_beta_eq_octet", "sweep_beta_eq_octet",
    "solve_composition", "solve_composition_t0",
    "solve_snm", "solve_snm_t0",
    "compute_nmp", "energy_per_baryon", "esym",
]
