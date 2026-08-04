"""
eos.enjl — extended NJL equation-of-state engine
====================================================

Reproduces, in Python, the method and results of

  * Xia 2024, "Quarkyonic matter and compact stars in an extended NJL model",
    PRD 110, 014022 (arXiv:2405.02946)   [uniform matter + TOV]
  * Xia, Maruyama, Yasutake & Tatsumi 2026, "Mixed phases of hadron-quark
    matter with Thomas-Fermi approximation" (arXiv:2409.12489)
    [Thomas-Fermi / Wigner-Seitz mixed phases]

Package layout:
    parameters.py      — ENJLParams (RKH + Table I couplings + B, f_q)
    species.py         — species table (baryons, quarks, leptons)
    thermodynamics.py  — T=0 Fermi-gas integrals with the Lambda cut-off
    uniform.py         — self-consistent uniform-matter solver (Paper 1)
    eos_beta.py        — beta equilibrium / charge neutrality + tables (Paper 1)
    star.py            — TOV star sequences (Paper 1)
    tfa/               — Thomas-Fermi approximation engine (Paper 2)

Physics conventions follow CLAUDE.md: strangeness S = +1 per s-quark, charge
C excludes leptons, natural units (MeV powers) internally with fm-based
public APIs.
"""
from eos.enjl.parameters import ENJLParams, get_enjl_default
from eos.enjl.species import make_species, BARYONS, QUARKS, LEPTONS
from eos.enjl.uniform import solve_point, ENJLEoSPoint

__all__ = [
    "ENJLParams", "get_enjl_default",
    "make_species", "BARYONS", "QUARKS", "LEPTONS",
    "solve_point", "ENJLEoSPoint",
]
