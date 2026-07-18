"""
DD2 density-dependent RMF equation-of-state engine (Phase 1).

Physics specification: DD2_EoS_Physics_Report.md; executable specification
for M0–M2: dd2_reference_validation.py.
"""
from eos.dd2.parametrization import Parametrization
from eos.dd2.solver import (
    EoSPoint, solve_beta_eq_t0, solve_composition_t0, solve_snm_t0,
)
from eos.dd2.nmp import compute_nmp, energy_per_baryon, esym

__all__ = [
    "Parametrization", "EoSPoint",
    "solve_beta_eq_t0", "solve_composition_t0", "solve_snm_t0",
    "compute_nmp", "energy_per_baryon", "esym",
]
