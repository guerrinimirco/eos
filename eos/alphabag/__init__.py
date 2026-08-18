"""
alphaBag: deconfined quark matter in a bag, with the leading perturbative QCD
correction, unpaired and colour-flavour locked.

Three light flavours as ideal gases inside a bag, with one constant coupling
alpha_s multiplying the free-gas terms -- a softening rather than the
stiffening a vector field gives, which is what makes this model and `eos.vmit`
two different mechanisms and not two parametrisations of one. Thermal gluons
are this model's own sector. There is no scalar field and no gap equation: the
quark masses are parameters and the thermodynamic potential is explicit in mu,
so the only solving to do is that of the equilibrium conditions.

A second phase lives here too: the colour-flavour locked condensate, the same
quark gas plus a Delta^2 mu^2 pairing term, closed by flavour locking
(n_u = n_d = n_s) rather than by an equilibrium condition and therefore
electrically neutral with no electrons at all.

See `alphabag.tex` for the physics, and `eos.abpr` for the analytic T = 0
parametrization of the same paired phase.

References: T. Fischer et al., Astrophys. J. Suppl. Ser. 194, 39 (2011);
M. Alford, M. Braby, M. Paris and S. Reddy, Astrophys. J. 629, 969 (2005).
"""
from eos.alphabag.parameters import Parameters
from eos.alphabag.species import SpeciesFlags
from eos.alphabag.thermodynamics import (
    QuarkThermo, MatterThermo, CFLThermo, G_QUARK,
    P_massless, e_massless, n_massless, s_massless,
    fermi_thermo, kinetic_thermo, quark_density,
    bag_pressure, bag_energy, gluon_thermo, thermo_from_mu,
    T_critical, cfl_gap, cfl_dgap_dT,
    cfl_P_correction, cfl_n_correction, cfl_s_correction, cfl_thermo_from_mu,
)
from eos.alphabag.solver import (
    EoSPoint, CFLPoint, MODE_FRACTIONS,
    default_guess, point_from_mu, cfl_point_from_mu,
    solve_beta_eq_neutrinoless, solve_beta_eq_neutrino_trapped,
    solve_fixed_yc, solve_fixed_yc_ys, solve_cfl, warm_start,
)
from eos.alphabag.table import (
    TableSpec, TableResult, build_table, rows_from_result, quark_row,
    solve_at, TableSettings, compute_table, save_results,
)
from eos.alphabag.api import (
    eos_point, eos_table, eos_response, PointResult, RESPONSE_FREEZES,
)

__all__ = [
    "Parameters", "SpeciesFlags",
    "QuarkThermo", "MatterThermo", "CFLThermo", "G_QUARK",
    "P_massless", "e_massless", "n_massless", "s_massless",
    "fermi_thermo", "kinetic_thermo", "quark_density",
    "bag_pressure", "bag_energy", "gluon_thermo", "thermo_from_mu",
    "T_critical", "cfl_gap", "cfl_dgap_dT",
    "cfl_P_correction", "cfl_n_correction", "cfl_s_correction",
    "cfl_thermo_from_mu",
    "EoSPoint", "CFLPoint", "MODE_FRACTIONS",
    "default_guess", "point_from_mu", "cfl_point_from_mu",
    "solve_beta_eq_neutrinoless", "solve_beta_eq_neutrino_trapped",
    "solve_fixed_yc", "solve_fixed_yc_ys", "solve_cfl", "warm_start",
    "TableSpec", "TableResult", "build_table", "rows_from_result",
    "quark_row", "solve_at", "TableSettings", "compute_table", "save_results",
    "eos_point", "eos_table", "eos_response", "PointResult",
    "RESPONSE_FREEZES",
]
