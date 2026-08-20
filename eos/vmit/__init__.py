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
from eos.vmit.parameters import Parameters, get_vmit_default, get_vmit_custom
from eos.vmit.species import SpeciesFlags
from eos.vmit.thermodynamics import (
    QuarkThermo, VMITThermo, QuarkMuDensity,
    compute_quark_thermo, compute_quark_density,
    compute_vector_field, compute_vector_pressure, compute_vector_energy,
    compute_bag_pressure, compute_bag_energy,
    compute_mu_effective, compute_mu_physical, compute_effective_mu_quarks,
    compute_quark_densities_for_solver,
    compute_vmit_thermo_from_mu_n,
    compute_quark_matter_thermo_from_n, compute_quark_matter_thermo_from_mu,
)
from eos.vmit.solver import (
    VMITEOSResult, RESIDUAL_TOL, scaled_residual_max, solve_system,
    solve_vmit_beta_eq, solve_vmit_fixed_yc, solve_vmit_fixed_yc_ys,
    solve_vmit_trapped_neutrinos, result_to_guess,
    get_default_guess_beta_eq, get_default_guess_fixed_yc,
    get_default_guess_fixed_yc_ys, get_default_guess_trapped_neutrinos,
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
    "Parameters", "get_vmit_default", "get_vmit_custom", "SpeciesFlags",
    "QuarkThermo", "VMITThermo", "QuarkMuDensity",
    "compute_quark_thermo", "compute_quark_density",
    "compute_vector_field", "compute_vector_pressure", "compute_vector_energy",
    "compute_bag_pressure", "compute_bag_energy",
    "compute_mu_effective", "compute_mu_physical",
    "compute_effective_mu_quarks", "compute_quark_densities_for_solver",
    "compute_vmit_thermo_from_mu_n",
    "compute_quark_matter_thermo_from_n", "compute_quark_matter_thermo_from_mu",
    "VMITEOSResult", "RESIDUAL_TOL", "scaled_residual_max", "solve_system",
    "solve_vmit_beta_eq", "solve_vmit_fixed_yc", "solve_vmit_fixed_yc_ys",
    "solve_vmit_trapped_neutrinos", "result_to_guess",
    "get_default_guess_beta_eq", "get_default_guess_fixed_yc",
    "get_default_guess_fixed_yc_ys", "get_default_guess_trapped_neutrinos",
    "TableSpec", "TableResult", "build_table", "rows_from_result",
    "quark_row", "solve_at", "MODE_FRACTIONS",
    "eos_point", "eos_table", "eos_response", "PointResult",
    "RESPONSE_FREEZES",
    "save_table", "load_table", "export_csv",
]
