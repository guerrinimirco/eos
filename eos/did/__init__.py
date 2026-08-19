"""DID / DIDY: a relativistic mean field with density- AND isospin-dependent
couplings.

The model of Frohaug, Maslov, Dexheimer, Grefa, Jahan, Ratti and Restrepo,
arXiv:2511.15646: a DD-RMF whose baryon-meson couplings depend on the isospin
asymmetry beta = sum_i tau_3i n_i / n_B as well as on the baryon density, which
is what lets it reproduce the HALQCD-based hyperon potentials in NEUTRON matter
as well as in symmetric matter -- and, through the later hyperon onsets that
follow, keep M_max above 2 M_sun with a full hyperon octet.

    DID    nucleons only          SpeciesFlags()
    DIDY   + the hyperon octet    SpeciesFlags(hyperons=True)

both from the same published parameter set, `Parameters.default()`. Delta
isobars, muons, a thermal meson gas and thermal neutrinos are extensions of
this implementation beyond the paper and are declared the same way.

See `did.tex` / `did.md` for the equations, and `verify/run_full_check.py` for
the physics invariants and the published values they are checked against.
"""
from eos.did.parameters import MULTIPLET_OF, MULTIPLETS, Parameters, tau3
from eos.did.couplings import (
    ALPHA_IDEAL, E_ISOSPIN, TAN_THETA_IDEAL, Z_SU6, blend, coupling,
    g8_from_aggregate, shape, su3_vector_ratios,
)
from eos.did.species import SpeciesFlags, active_baryons
from eos.did.thermodynamics import (
    Fields, Matter, Species, cold_start, evaluate, field_estimate,
    kinetic_thermo, mean_fields, meson_potentials, single_particle_potential,
    species_table, thermal_meson_thermo, thermo_at_potentials, thermo_from_mu,
)
from eos.did.solver import (
    EoSPoint, MODE_FRACTIONS, System, assemble, default_guess, mode_spec,
    residual, solve, solve_beta_eq_neutrinoless, solve_beta_eq_neutrino_trapped,
    solve_fixed_yc, solve_fixed_yc_ys, solve_mode, unknown_names, warm_start,
)
from eos.did.nmp import (
    compute_nmp, crossover_M, delta_ratios_from_potential, energy_per_baryon,
    nuclear_matter, symmetry_energy_full, symmetry_energy_quadratic,
)
from eos.did.table import (
    TableResult, TableSpec, build_table, hadronic_row, rows_from_result,
)
from eos.did.api import (
    PointResult, RESPONSE_FREEZES, eos_point, eos_response, eos_table,
)
from eos.general.table_io import export_csv, load_table, save_table

__all__ = [
    "Parameters", "MULTIPLETS", "MULTIPLET_OF", "tau3",
    "shape", "blend", "coupling", "su3_vector_ratios", "g8_from_aggregate",
    "E_ISOSPIN", "ALPHA_IDEAL", "TAN_THETA_IDEAL", "Z_SU6",
    "SpeciesFlags", "active_baryons",
    "Fields", "Matter", "Species", "species_table", "kinetic_thermo",
    "mean_fields", "evaluate", "field_estimate", "cold_start",
    "meson_potentials", "thermal_meson_thermo", "single_particle_potential",
    "thermo_from_mu", "thermo_at_potentials",
    "EoSPoint", "System", "MODE_FRACTIONS", "unknown_names", "residual",
    "assemble", "default_guess", "warm_start", "mode_spec", "solve",
    "solve_beta_eq_neutrinoless", "solve_beta_eq_neutrino_trapped",
    "solve_fixed_yc", "solve_fixed_yc_ys", "solve_mode",
    "compute_nmp", "energy_per_baryon", "nuclear_matter",
    "symmetry_energy_quadratic", "symmetry_energy_full", "crossover_M",
    "delta_ratios_from_potential",
    "TableSpec", "TableResult", "build_table", "hadronic_row",
    "rows_from_result",
    "eos_point", "eos_table", "eos_response", "PointResult",
    "RESPONSE_FREEZES",
    "save_table", "load_table", "export_csv",
]
