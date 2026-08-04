"""
eos.mixed
=========
Hadron-quark mixed-phase equations of state: the DD2 density-dependent
relativistic mean-field hadronic engine (`eos.dd2`) coupled to the vMIT
vector-bag quark engine (`eos.vmit`) across a first-order phase transition.

Everything you need is imported from this module. The subpackages
(`equilibrium`, `solvers`, `tables`) are internal structure.

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
    beta_eq_neutrino_trapped(Y_L)       (n_B, Y_L, T)
    fixed_YC(Y_C, leptons=...)          (n_B, Y_C, T)
    fixed_YC_YS(Y_C, Y_S, leptons=...)  (n_B, Y_C, Y_S, T)

Here C is the NON-leptonic electric charge (hadrons and quarks only) and
strangeness counts +1 per s-quark — see `eos.mixed.equilibrium.charges` and
CLAUDE.md §2 before assuming either sign.

Typical use
-----------
    from eos.dd2 import Parametrization, SpeciesFlags
    from eos.mixed import MixedTableSpec, build_mixed_table

    par = Parametrization.from_dd2y_defaults()
    flags = SpeciesFlags(hyperons=True, deltas=True, muons=True)
    rows, windows = build_mixed_table(MixedTableSpec(
        par, flags, "beta_eq_neutrinoless",
        axes={"nB": nB_grid, "T": [0.0, 10.0]}, eta=0.5))

Units are fm-based on every public boundary: densities fm^-3, pressure and
energy density MeV/fm^3, temperature and chemical potentials MeV.
"""
# --- equilibrium modes and the regime declaration behind them ---------------
from eos.mixed.equilibrium.charges import (
    ChargeSpec, Regime,
    beta_eq_neutrinoless, beta_eq_neutrino_trapped, fixed_YC, fixed_YC_YS,
    QUARK_QN, quark_charges, hadronic_qn, hadronic_charges,
)

# --- solving one point, and sweeping density -------------------------------
from eos.mixed.solvers.phases import PhaseThermo, hadronic_phase, quark_phase
from eos.mixed.solvers.point import MixedResult, solve_mixed
from eos.mixed.solvers.sweep import (
    MixedWindow, sweep_mixed, locate_window, find_mixed_window,
    seed_across_eta,
)

# --- generating, stitching and storing tables ------------------------------
from eos.mixed.tables.generate import (
    MixedTableSpec, build_mixed_table, solve_mixed_at_entropy,
    make_charge_spec, composition_row, MODE_FRACTIONS,
)
from eos.mixed.tables.core_eos import (
    MixedEoSTable, build_mixed_eos_table, mass_radius_mixed,
)
from eos.general.table_io import save_table, load_table, export_csv

# --- where in parameter space a hybrid equation of state exists -------------
from eos.mixed.scan import (
    scan_parameters, scan_point, grid_samples, NMP_KEYS, VMIT_KEYS,
)

__all__ = [
    # modes
    "beta_eq_neutrinoless", "beta_eq_neutrino_trapped",
    "fixed_YC", "fixed_YC_YS", "MODE_FRACTIONS", "make_charge_spec",
    # regime machinery, for combinations the named modes do not cover
    "ChargeSpec", "Regime",
    # quantum numbers
    "QUARK_QN", "quark_charges", "hadronic_qn", "hadronic_charges",
    # solving
    "solve_mixed", "MixedResult", "PhaseThermo",
    "hadronic_phase", "quark_phase",
    "sweep_mixed", "locate_window", "find_mixed_window", "MixedWindow",
    "seed_across_eta",
    "solve_mixed_at_entropy",
    # tables
    "MixedTableSpec", "build_mixed_table", "composition_row",
    "MixedEoSTable", "build_mixed_eos_table", "mass_radius_mixed",
    "save_table", "load_table", "export_csv",
    # parameter-space scan
    "scan_parameters", "scan_point", "grid_samples", "NMP_KEYS", "VMIT_KEYS",
]
