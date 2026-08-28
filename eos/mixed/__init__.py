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
The three entry points every model in this repository exposes, with `eta`
added because a hybrid equation of state has a transition in it. The parameter
argument is the PAIRING — two `Phase` objects, each closing over its own
model's parameters (CLAUDE.md section 5) — so no model holds a privileged
position in a signature:

    from eos.mixed import (eos_point, eos_table, SpeciesFlags,
                           default_pair, sfho_phase, njl_phase)

    species = SpeciesFlags(hyperons=True, deltas=True, muons=True)

    pair = default_pair(dd2_par, dd2_flags, vmit_par)   # the DD2 + vMIT hybrid
    pair = (sfho_phase(sfho_par, sfho_flags), njl_phase(njl_par))   # or another

    state = eos_point(pair, "beta_eq_neutrinoless", species,
                      n_B=0.6, T=0.0, eta=0.5)
    rows, windows = eos_table(pair, "beta_eq_neutrinoless", species,
                              axes={"nB": nB_grid, "T": [0.0, 10.0]}, eta=0.5)

A mixed table is rows PLUS windows: the phase boundaries found on each line are
part of the answer, not something to recover afterwards from the rows.
`TableSpec` / `build_table` remain available underneath for a table
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

# --- the engine's own species flags (CLAUDE.md section 4) ------------------
from eos.mixed.species import SpeciesFlags, mixture_flags

# --- the phase-adapter contract: the only surface onto a bulk engine --------
from eos.mixed.adapters import (
    Phase, PhaseThermo, hadronic_phase, hadronic_seed, quark_phase,
    dd2_phase, vmit_phase, default_pair,
    sfho_phase, zl_phase, alphabag_phase, enjl_branch_pair,
    ccdm_phase, did_phase, did_seed, njl_phase,
)

# --- where a multi-branch model's transitions are, as data ------------------
from eos.mixed.construction import (
    CONSTRUCTION_PAIRS, Coexistence, enjl_coexistences,
    enjl_composition_coexistences,
)

# --- solving one point, and sweeping density -------------------------------
from eos.mixed.solver import (
    Result, solve, sweep, seed_across_eta,
    solve_beta_eq_neutrinoless, solve_beta_eq_neutrino_trapped,
    solve_fixed_yc, solve_fixed_yc_ys,
)
from eos.mixed.boundaries import (
    Window, locate_window, locate_windows, refine_window,
    solve_fixed_chi, MaxwellPoint, locate_maxwell,
    locate_maxwell_composition, composition_phase, neutral_phase,
)

# --- generating, stitching and storing tables ------------------------------
from eos.mixed.table import (
    TableSpec, build_table, solve_at_entropy,
    make_charge_spec, composition_row, MODE_FRACTIONS,
)
from eos.mixed.hybrid import (
    EoSTable, build_hybrid_table, mass_radius_mixed,
)
from eos.general.table_io import save_table, load_table, export_csv

# --- the uniform model API, the same three entry points every model has -----
from eos.mixed.api import (
    PointResult, HybridResult, eos_point, eos_table, eos_response,
    hybrid_table, RESPONSE_FREEZES,
)

# --- thermodynamic responses --------------------------------------------
from eos.mixed.responses import (
    sound_speed_eq, sound_speed_frozen, sound_speed_frozen_pure,
    adiabatic_index, frozen_along,
)

__all__ = [
    # modes
    "beta_eq_neutrinoless", "beta_eq_neutrino_trapped",
    "fixed_YC", "fixed_YC_YS", "MODE_FRACTIONS", "make_charge_spec",
    # regime machinery, for combinations the named modes do not cover
    "ChargeSpec", "Regime", "Locality",
    # the engine's own species flags
    "SpeciesFlags", "mixture_flags",
    # quantum numbers
    "QUARK_QN", "quark_charges", "hadronic_qn", "hadronic_charges",
    # phase adapters and pairings
    "Phase", "PhaseThermo", "hadronic_phase", "hadronic_seed", "quark_phase",
    "dd2_phase", "vmit_phase", "default_pair",
    "sfho_phase", "zl_phase", "alphabag_phase", "enjl_branch_pair",
    "CONSTRUCTION_PAIRS", "Coexistence", "enjl_coexistences",
    "enjl_composition_coexistences",
    "ccdm_phase", "did_phase", "did_seed", "njl_phase",
    # solving
    "solve", "Result",
    "solve_beta_eq_neutrinoless", "solve_beta_eq_neutrino_trapped",
    "solve_fixed_yc", "solve_fixed_yc_ys",
    "sweep", "locate_window", "locate_windows",
    "MaxwellPoint", "locate_maxwell", "locate_maxwell_composition",
    "composition_phase", "neutral_phase",
    "Window",
    "solve_fixed_chi", "refine_window",
    "seed_across_eta",
    "solve_at_entropy",
    # tables
    "TableSpec", "build_table", "composition_row",
    "EoSTable", "build_hybrid_table", "mass_radius_mixed",
    "save_table", "load_table", "export_csv",
    # the uniform model API
    "eos_point", "eos_table", "eos_response", "PointResult",
    "hybrid_table", "HybridResult", "RESPONSE_FREEZES",
    # responses
    "sound_speed_eq", "sound_speed_frozen", "sound_speed_frozen_pure",
    "adiabatic_index", "frozen_along",
]
