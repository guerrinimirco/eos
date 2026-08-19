"""eos.njl -- three-flavour Nambu-Jona-Lasinio quark matter, with and without
colour superconductivity, at finite temperature.

A contact four-fermion theory of the light quarks: the scalar channel that
breaks chiral symmetry and gives the quarks their constituent masses, the
't Hooft determinant that ties the three flavours together and fixes the
eta', the diquark channel that condenses into a colour superconductor, and a
vector channel whose repulsion is what a compact star's high-density pressure
needs. Everything except the current masses is generated: the constituent
masses come out of a gap equation, the bag constant is a derived vacuum
pressure difference rather than an input, and which colour-superconducting
pattern the matter is in is an OUTCOME chosen by free energy, not a
declaration.

    Rehberg, Klevansky, Huefner, Phys. Rev. C 53, 410 (1996)
        [arXiv:hep-ph/9506436]                       -- the RKH parameter set
    Buballa, Phys. Rept. 407, 205 (2005) [arXiv:hep-ph/0402234]
    Ruester, Werth, Buballa, Shovkovy, Rischke, Phys. Rev. D 72, 034004 (2005)
        [arXiv:hep-ph/0503184]                       -- neutral three-flavour CSC
    Alford, Schmitt, Rajagopal, Schaefer, Rev. Mod. Phys. 80, 1455 (2008)
        [arXiv:0709.4635]                            -- the review

`njl.tex` states every equation the code solves and the residual row by row;
`docs/njl_csc_implementation.md` is the implementation specification it
follows, and is the authority wherever the two differ.

All four modes of CLAUDE.md section 3 are closed, at any temperature. The
pairing sector is a flag (`SpeciesFlags(csc=True)`), not a separate model, and
with it off this is ordinary unpaired NJL.

Layout:
    parameters.py      Parameters: the RKH set and the three coupling tiers
    couplings.py       G_V as a function of the state, and its rearrangement
    species.py         SpeciesFlags, the quantum numbers, the gap patterns
    thermodynamics.py  quantities computed FROM the state: the cut medium
                       integrals, the Dirac sea, the gap equation, eps, P, s,
                       and the internal solve at fixed potentials
    solver.py          the equilibrium conditions, the mode residual, and the
                       enumeration that chooses the pairing pattern
    table.py           the warm-started density sweep + progress callback
    api.py             eos_point / eos_table / eos_response
    responses.py       second derivatives, by re-solved finite differences
    verify/            the physics invariants

The pairing machinery itself is NOT here: the gap matrix, the
Bogoliubov-de Gennes problem, the pairing correction to Omega and the
Hellmann-Feynman gap kernels are `eos.general.pairing`, shared with the
colour-dielectric model, because the pairing sector of the two is the same
sector (CLAUDE.md section 7).

Conventions are this repository's: strangeness S = +1 per s quark, C the
charge of strongly interacting matter only, natural units (MeV powers)
internally and fm-based units at every public boundary. The colour generator
normalisation is T_8 = diag(1, 1, -2)/3; converting mu_8 to the literature's
two other conventions is documented in `eos.general.pairing`.
"""
from eos.njl.parameters import Parameters, PUBLISHED_SETS, VECTOR_FORMS
from eos.njl.couplings import (
    effective_exponent, vector_coupling, vector_energy, vector_self_energy,
)
from eos.njl.species import (
    DEFAULT_PATTERNS, MODE_NAMES, PATTERNS, SpeciesFlags, pattern_mask,
    pattern_seed,
)
from eos.njl.thermodynamics import (
    ModeThermo, NJLState, Vacuum, bag_constant, condensates,
    condensate_energy, f_pi, kinetic_thermo, masses_from_condensates,
    has_vector, sea_energy, sea_scalar_density, state_at, surface_term,
    thermo_from_mu, vacuum_solution,
)
from eos.njl.solver import (
    EoSPoint, MODE_FRACTIONS, default_guess, mode_spec, residual, solve,
    solve_beta_eq_neutrino_trapped, solve_beta_eq_neutrinoless,
    solve_fixed_yc, solve_fixed_yc_ys, solve_pattern, warm_start,
)
from eos.njl.table import TableSpec, TableResult, build_table, quark_row
from eos.njl.api import PointResult, eos_point, eos_response, eos_table

__all__ = [
    "Parameters", "PUBLISHED_SETS", "VECTOR_FORMS",
    "vector_coupling", "vector_energy", "vector_self_energy",
    "effective_exponent",
    "SpeciesFlags", "PATTERNS", "DEFAULT_PATTERNS", "MODE_NAMES",
    "pattern_mask", "pattern_seed",
    "ModeThermo", "NJLState", "Vacuum", "kinetic_thermo", "surface_term",
    "sea_energy", "sea_scalar_density", "condensates", "condensate_energy",
    "masses_from_condensates", "f_pi", "vacuum_solution", "bag_constant",
    "state_at", "thermo_from_mu", "has_vector",
    "EoSPoint", "MODE_FRACTIONS", "mode_spec", "default_guess", "warm_start",
    "residual", "solve", "solve_pattern", "solve_beta_eq_neutrinoless",
    "solve_beta_eq_neutrino_trapped", "solve_fixed_yc", "solve_fixed_yc_ys",
    "TableSpec", "TableResult", "build_table", "quark_row",
    "PointResult", "eos_point", "eos_table", "eos_response",
]
