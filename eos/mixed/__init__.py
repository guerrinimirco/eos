"""
eos.mixed
=========
Hadron-quark mixed-phase equations of state: the DD2 density-dependent
relativistic mean-field hadronic engine (`eos.dd2`) coupled to the vMIT
vector-bag quark engine (`eos.vmit`) across a first-order phase transition.

Everything you need is imported from this module. The package is laid out the
way the models are (CLAUDE.md §5): `charges` declares the equilibria,
`thermodynamics` computes quantities from the state, `solver` finds the state,
`boundaries` locates the transition window, `hybrid` stitches the complete
hybrid equation of state, `table` drives grids, `api` is the uniform surface,
and `backends/` holds the deletable accelerated paths.

What the engine does
--------------------
Given a density, a temperature and an equilibrium mode, it finds the state in
which a hadronic phase and a quark phase coexist: matched baryon chemical
potentials, balanced pressures, and whatever charges the mode conserves. The
quark volume fraction chi comes out of the solve, and its value says which
regime the point is in — chi <= 0 pure hadronic, chi >= 1 pure quark, in
between a genuine mixed phase.

A continuous parameter eta interpolates between the two standard constructions
of the transition by choosing how much of electric-charge neutrality is imposed
locally rather than globally: eta = 0 is Gibbs (pressure rises through the
mixed window), eta = 1 is Maxwell (a constant-pressure plateau and a density
jump), and intermediate values stand in for the finite surface tension and
Coulomb cost of the mixed-phase structures.

Equilibrium modes
-----------------
Each is a choice of how {B, C, S, L_e} are conserved, and all four run through
one solver rather than four:

    beta_eq_neutrinoless()              (n_B, T)
    beta_eq_neutrino_trapped(Y_Le)       (n_B, Y_Le, T)
    fixed_YC(Y_C, leptons=...)          (n_B, Y_C, T)
    fixed_YC_YS(Y_C, Y_S, leptons=...)  (n_B, Y_C, Y_S, T)

Here C is the NON-leptonic electric charge (hadrons and quarks only) and
strangeness counts +1 per s-quark — see `eos.mixed.charges` and
CLAUDE.md §2 before assuming either sign.

Typical use
-----------
The three entry points every model in this repository exposes, with `eta` and
the quark parameters added because a hybrid equation of state has a transition
and two models in it:

    from eos.dd2 import Parametrization, SpeciesFlags
    from eos.mixed import eos_point, eos_table, eos_response

    par = Parametrization.from_dd2_defaults()
    flags = SpeciesFlags(hyperons=True, deltas=True, muons=True)

    state = eos_point(par, "beta_eq_neutrinoless", flags,
                      n_B=0.6, T=0.0, eta=0.5)
    rows, windows = eos_table(par, "beta_eq_neutrinoless", flags,
                              axes={"nB": nB_grid, "T": [0.0, 10.0]}, eta=0.5)

A mixed table is rows PLUS windows: the phase boundaries found on each line are
part of the answer, not something to recover afterwards from the rows.
`MixedTableSpec` / `build_mixed_table` remain available underneath for a table
described as an object rather than as arguments.

Units are fm-based on every public boundary: densities fm^-3, pressure and
energy density MeV/fm^3, temperature and chemical potentials MeV.
"""
# --- equilibrium modes and the regime declaration behind them ---------------
from eos.mixed.charges import (
    ChargeSpec, Regime, Locality,
    beta_eq_neutrinoless, beta_eq_neutrino_trapped, fixed_YC, fixed_YC_YS,
    QUARK_QN, quark_charges, hadronic_qn, hadronic_charges,
)

# --- the phase-adapter contract: the only surface onto a bulk engine --------
from eos.mixed.adapters import (
    PhaseThermo, hadronic_phase, hadronic_seed, quark_phase,
)

# --- solving one point, and sweeping density -------------------------------
from eos.mixed.solver import (
    MixedResult, solve_mixed, sweep_mixed, find_mixed_window, seed_across_eta,
)
from eos.mixed.boundaries import MixedWindow, locate_window

# --- generating, stitching and storing tables ------------------------------
from eos.mixed.table import (
    MixedTableSpec, build_mixed_table, solve_mixed_at_entropy,
    make_charge_spec, composition_row, MODE_FRACTIONS,
)
from eos.mixed.hybrid import (
    MixedEoSTable, build_mixed_eos_table, mass_radius_mixed,
)
from eos.general.table_io import save_table, load_table, export_csv

# --- the uniform model API, the same three entry points every model has -----
from eos.mixed.api import (
    PointResult, eos_point, eos_table, eos_response, RESPONSE_FREEZES,
)

# --- thermodynamic responses --------------------------------------------
from eos.mixed.responses import (
    sound_speed_eq, sound_speed_frozen, sound_speed_frozen_hadronic,
    sound_speed_frozen_quark, adiabatic_index, frozen_along,
)

# --- where in parameter space a hybrid equation of state exists -------------
from eos.mixed.scan import (
    scan_parameters, scan_point, scan_hadronic, scan_hadronic_point,
    build_parametrization, grid_samples, NMP_KEYS, VMIT_KEYS,
    DEFAULT_HYPERON_POTENTIALS, DEFAULT_U_DELTA,
)

__all__ = [
    # modes
    "beta_eq_neutrinoless", "beta_eq_neutrino_trapped",
    "fixed_YC", "fixed_YC_YS", "MODE_FRACTIONS", "make_charge_spec",
    # regime machinery, for combinations the named modes do not cover
    "ChargeSpec", "Regime", "Locality",
    # quantum numbers
    "QUARK_QN", "quark_charges", "hadronic_qn", "hadronic_charges",
    # phase adapters
    "PhaseThermo", "hadronic_phase", "hadronic_seed", "quark_phase",
    # solving
    "solve_mixed", "MixedResult",
    "sweep_mixed", "locate_window", "find_mixed_window", "MixedWindow",
    "seed_across_eta",
    "solve_mixed_at_entropy",
    # tables
    "MixedTableSpec", "build_mixed_table", "composition_row",
    "MixedEoSTable", "build_mixed_eos_table", "mass_radius_mixed",
    "save_table", "load_table", "export_csv",
    # the uniform model API
    "eos_point", "eos_table", "eos_response", "PointResult",
    "RESPONSE_FREEZES",
    # responses
    "sound_speed_eq", "sound_speed_frozen", "sound_speed_frozen_hadronic",
    "sound_speed_frozen_quark", "adiabatic_index", "frozen_along",
    # parameter-space scan
    "scan_parameters", "scan_point", "scan_hadronic", "scan_hadronic_point",
    "build_parametrization", "grid_samples", "NMP_KEYS", "VMIT_KEYS",
    "DEFAULT_HYPERON_POTENTIALS", "DEFAULT_U_DELTA",
]
