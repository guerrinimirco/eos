"""
DD2 density-dependent RMF equation-of-state engine (Phase 1).

Physics specification: DD2_EoS_Physics_Report.md; executable specification
for M0–M2: dd2_reference_validation.py.
"""
from eos.dd2.parametrization import Parametrization

__all__ = ["Parametrization"]
