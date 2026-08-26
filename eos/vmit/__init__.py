"""
vMIT: the MIT bag model with a repulsive vector interaction between quarks.

Three quark flavours (u, d, s) confined by a bag of constant energy density B,
with a flavour-blind isoscalar-vector field whose coupling enters through the
single combination a = g_V^2/m_V^2. The quark masses are the current masses
and are parameters, so the only mean field is the vector one; that is what
makes the model cheap enough to scan over. Leptons and photons are added by
the mode and the species flags. See `vmit.tex` for the physics and
`eos.mixed` for the hybrid hadron-quark construction this is the quark half
of.
"""
from eos.vmit.parameters import Parameters
from eos.vmit.species import SpeciesFlags
from eos.vmit.thermodynamics import (
    QuarkThermo, MatterThermo, QuarkMuDensity,
    kinetic_thermo, quark_density,
    vector_field, vector_pressure, vector_energy,
    bag_pressure, bag_energy,
    effective_potential, physical_potentials, effective_potentials,
    effective_state,
    thermo_from_mu_n,
    thermo_from_n, thermo_from_mu,
)
from eos.vmit.solver import (
    EoSPoint, RESIDUAL_TOL, scaled_residual_max, solve_system,
    solve_beta_eq_neutrinoless, solve_fixed_yc, solve_fixed_yc_ys,
    solve_beta_eq_neutrino_trapped, warm_start, default_guess,
)
from eos.vmit.table import (
    TableSpec, TableResult, build_table, rows_from_result, quark_row,
    solve_at, MODE_FRACTIONS,
)
from eos.vmit.api import (
    eos_point, eos_table, eos_response, PointResult, RESPONSE_FREEZES,
)
from eos.general.table_io import save_table, load_table, export_csv

__all__ = [
    "Parameters", "SpeciesFlags",
    "QuarkThermo", "MatterThermo", "QuarkMuDensity",
    "kinetic_thermo", "quark_density",
    "vector_field", "vector_pressure", "vector_energy",
    "bag_pressure", "bag_energy",
    "effective_potential", "physical_potentials",
    "effective_potentials", "effective_state",
    "thermo_from_mu_n",
    "thermo_from_n", "thermo_from_mu",
    "EoSPoint", "RESIDUAL_TOL", "scaled_residual_max", "solve_system",
    "solve_beta_eq_neutrinoless", "solve_fixed_yc", "solve_fixed_yc_ys",
    "solve_beta_eq_neutrino_trapped", "warm_start", "default_guess",
    "TableSpec", "TableResult", "build_table", "rows_from_result",
    "quark_row", "solve_at", "MODE_FRACTIONS",
    "eos_point", "eos_table", "eos_response", "PointResult",
    "RESPONSE_FREEZES",
    "save_table", "load_table", "export_csv",
]
