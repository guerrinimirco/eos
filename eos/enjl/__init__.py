"""eos.enjl -- the extended Nambu-Jona-Lasinio model of dense matter at T = 0.

Baryons and quarks described by ONE functional: a baryon is a three-quark
cluster whose mass is built from the same constituent masses the NJL gap
equation determines, so the chiral, quarkyonic and deconfinement transitions
come out of a single mean field rather than out of two models joined at a
boundary. Density-dependent couplings, and therefore rearrangement
self-energies; a three-momentum cut-off entering as a temperature-independent
vacuum subtraction on the quark sector alone.

    C.-J. Xia, Phys. Rev. D 110, 014022 (2024) [arXiv:2405.02946]

`enjl.tex` states every equation the code solves and the equilibrium residual
row by row. All four modes of CLAUDE.md section 3 are closed, at T = 0; a
mode is a declaration (`eos.general.modes.ModeSpec`) that the residual
assembly reads, not a per-mode function. What this model refuses is a
temperature, not a mode.

Layout:
    parameters.py      Parameters: the RKH set, Table I, f_q and B
    species.py         the eight species, their quantum numbers, SpeciesFlags
    thermodynamics.py  quantities computed FROM the state: the gap equation,
                       the mean fields, eps, P, s, the conserved-charge sums
    solver.py          the equilibrium conditions and the solve that closes
                       them, one per mode, assembled from the declaration
    table.py           the warm-started branch continuation along n_B
    api.py             eos_point / eos_table / eos_response
    verify/            the physics invariants

Conventions are CLAUDE.md's: strangeness S = +1 per s quark, C the charge of
strongly interacting matter only, natural units (MeV powers) internally and
fm-based units at every public boundary.
"""
from eos.enjl.parameters import Parameters, PUBLISHED_SETS, RHO_FACTOR
from eos.enjl.species import SpeciesFlags
from eos.enjl.thermodynamics import (
    EoSPoint, kinetic_thermo, thermo_from_mu, thermo_from_n,
    vacuum_energy_density, vacuum_solution,
)
from eos.enjl.solver import (
    BetaPoint, mode_spec, solve, solve_beta_eq_neutrino_trapped,
    solve_beta_eq_neutrinoless, solve_fixed_yc, solve_fixed_yc_ys,
)
from eos.enjl.table import TableSpec, TableResult, build_table
from eos.enjl.api import PointResult, eos_point, eos_response, eos_table

__all__ = [
    "Parameters", "PUBLISHED_SETS", "RHO_FACTOR", "SpeciesFlags",
    "EoSPoint", "kinetic_thermo", "thermo_from_mu", "thermo_from_n",
    "vacuum_energy_density", "vacuum_solution",
    "BetaPoint", "mode_spec", "solve", "solve_beta_eq_neutrinoless",
    "solve_beta_eq_neutrino_trapped", "solve_fixed_yc", "solve_fixed_yc_ys",
    "TableSpec", "TableResult", "build_table",
    "PointResult", "eos_point", "eos_response", "eos_table",
]
