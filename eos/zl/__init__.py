"""
ZL: the Zhao-Lattimer nucleonic density functional.

Protons and neutrons as free Fermi gases at their vacuum mass, plus an
interaction energy density built from a proton-neutron cross term and an
isovector term, each with a linear and a power-law piece in n_B/n0. There is
no scalar field and therefore no effective mass: the whole self-consistency of
the model is between the densities and the interaction potentials
mu_Hv_i = dV/dn_i they generate. Six numbers set the six lowest nuclear-matter
parameters almost independently, which is what the model is for.

Leptons and photons are added by the mode and the species flags. See `zl.tex`
for the physics and `eos.zlvmit` for the first-generation hybrid this is the
hadronic half of.

Reference: T. Zhao and J. M. Lattimer, Phys. Rev. D 102, 023021 (2020).
"""
from eos.zl.parameters import Parameters
from eos.zl.species import SpeciesFlags
from eos.zl.thermodynamics import (
    NucleonThermo, MatterThermo, EffectiveState, G_NUCLEON,
    kinetic_thermo, interaction_energy, interaction_pressure,
    interaction_potentials, effective_potentials, effective_state,
    thermo_from_mu_n, thermo_from_mu, thermo_from_n,
)
from eos.zl.solver import (
    EoSPoint, MODE_FRACTIONS, RESIDUAL_TOL, scaled_residual_max,
    solve_system,
    solve_beta_eq_neutrinoless, solve_beta_eq_neutrino_trapped,
    solve_fixed_yc, solve_fixed_yc_ys, default_guess, warm_start,
)
from eos.zl.table import (
    TableSpec, TableResult, build_table, rows_from_result, nucleon_row,
    solve_at, TableSettings, compute_table, save_results,
)
from eos.zl.api import (
    eos_point, eos_table, eos_response, PointResult, RESPONSE_FREEZES,
)

from eos.zl.nmp import (
    compute_nmp, energy_per_baryon, from_nmp, invert_nmp, nuclear_matter,
    pressure, saturation_density, symmetry_energy,
)

__all__ = [
    "Parameters", "SpeciesFlags",
    "NucleonThermo", "MatterThermo", "EffectiveState", "G_NUCLEON",
    "kinetic_thermo", "interaction_energy", "interaction_pressure",
    "interaction_potentials", "effective_potentials", "effective_state",
    "thermo_from_mu_n", "thermo_from_mu", "thermo_from_n",
    "EoSPoint", "MODE_FRACTIONS", "RESIDUAL_TOL", "scaled_residual_max",
    "solve_system",
    "solve_beta_eq_neutrinoless", "solve_beta_eq_neutrino_trapped",
    "solve_fixed_yc", "solve_fixed_yc_ys", "default_guess", "warm_start",
    "TableSpec", "TableResult", "build_table", "rows_from_result",
    "nucleon_row", "solve_at", "TableSettings", "compute_table",
    "save_results",
    "compute_nmp", "invert_nmp", "from_nmp", "nuclear_matter",
    "energy_per_baryon", "pressure", "saturation_density", "symmetry_energy",
    "eos_point", "eos_table", "eos_response", "PointResult",
    "RESPONSE_FREEZES",
]
