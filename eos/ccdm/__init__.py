"""eos.ccdm -- chiral colour-dielectric quark matter, with and without colour
superconductivity, at finite temperature.

A model in which confinement and chiral symmetry breaking are two faces of one
mechanism. A dilaton field carries the gluon condensate; the dielectric
function chi = (1 - phi_bar^4)^p built from it measures how transparent the
medium is to colour, and it sits in the DENOMINATOR of the quark masses,

    M*_u,d = (g_q sigma + m_u,d)/chi ,     M*_s = (g_s zeta + m_s)/chi

so that as the condensate reaches its vacuum value the medium goes opaque, the
effective masses diverge and the quarks leave the medium entirely. That is not
a suppression to be smoothed: at T = 0 a mode with M* >= mu* contributes
IDENTICALLY zero, and it is what makes deconfinement first order here rather
than a crossover. The chiral sector -- a Mexican hat in the light and strange
condensates with explicit breaking fixed by f_pi, m_pi, f_K and m_K -- supplies
the larger part of the effective bag constant: B_g^(1/4) = 150 MeV against
B_chi^(1/4) = 230 MeV, giving B_eff = (240 MeV)^4 = 429 MeV/fm^3, which is a
DERIVED number and not an input.

    docs/ccdm_implementation.md    -- the implementation specification, and
                                      the AUTHORITY wherever it and the code
                                      or `ccdm.tex` differ (with the two
                                      documented exceptions below)
    Drago, Fiolhais, Tambini [arXiv:hep-ph/9503462]
    Ghosh, Phatak, Phys. Rev. C 52, 2195 (1995) [arXiv:nucl-th/9509017]
    Alberico, Drago, Ratti [arXiv:hep-ph/0110091]
    Maieron, Baldo, Burgio, Schulze [arXiv:nucl-th/0404089]
    Alford, Schmitt, Rajagopal, Schaefer, Rev. Mod. Phys. 80, 1455 (2008)
        [arXiv:0709.4635]                            -- the pairing review

TWO CORRECTIONS TO THE SPECIFICATION, both forced by the Euler audit its own
section 9.6 mandates and both written out in `ccdm.tex` beside its forms: eps
takes +(1/2) m_omega^2 omega_0^2 rather than minus (a repulsive interaction
adds energy), and Omega carries -Sigma_R n_q, the rearrangement term the
specification puts in mu* without its counterpart. With them the Euler
relation holds to machine precision at every solved point, paired and
unpaired, at T = 0 and above; without either it does not.

All four modes of CLAUDE.md section 3 are closed, at any temperature, and the
specification's closure rows map onto them one for one (see
`eos.ccdm.solver`). The pairing sector is a flag (`SpeciesFlags(csc=True)`),
not a separate model.

Layout:
    parameters.py      Parameters: the vacuum-fixed block, the derived
                       constants, and the three coupling tiers
    couplings.py       g_omega as a function of the state, and the
                       rearrangement self-energy it owes
    species.py         SpeciesFlags, the quantum numbers, the chiral branches
    thermodynamics.py  quantities computed FROM the state: the dielectric, the
                       two potentials, the medium integrals, eps, P, s, and
                       the internal solve at fixed potentials
    solver.py          the equilibrium conditions, the mode residual, and the
                       two enumerations that choose the branch and the pattern
    table.py           the warm-started density sweep + progress callback
    api.py             eos_point / eos_table / eos_response
    responses.py       second derivatives, by re-solved finite differences
    verify/            the physics invariants

The pairing machinery itself is NOT here: the gap matrix, the
Bogoliubov-de Gennes problem, the pairing correction to Omega and the
Hellmann-Feynman gap kernels are `eos.general.pairing`, shared with the NJL
model, because the pairing sector of the two is the same sector (CLAUDE.md
section 7). The ideal-gas integrals are `eos.general.fermi_integrals` for the
same reason -- the split-panel method there, in natural units, not the
fm-based JEL one the hadronic models call.

Conventions are this repository's: strangeness S = +1 per s quark, C the
charge of strongly interacting matter only, natural units (MeV powers)
internally and fm-based units at every public boundary. The colour generator
normalisation is T_8 = diag(1, 1, -2)/3; converting mu_8 to the literature's
two other conventions is documented in `eos.general.pairing`.
"""
from eos.ccdm.parameters import Derived, Parameters, PUBLISHED_SETS
from eos.ccdm.couplings import (
    diquark_coupling, has_vector, rearrangement, vector_coupling,
    vector_coupling_derivative, vector_field, vector_self_energy,
)
from eos.ccdm.species import (
    BRANCHES, DEFAULT_PATTERNS, DENSITY_BRANCHES, MODE_NAMES, PATTERNS,
    POTENTIAL_BRANCHES, SpeciesFlags, branch_seed, pattern_mask, pattern_seed,
)
from eos.ccdm.thermodynamics import (
    CCDMState, bag_constant, chiral_derivatives, chiral_potential, dielectric,
    effective_masses, glue_derivative, glue_potential, guard_phi, mode_thermo,
    state_at, thermo_from_mu,
)
from eos.ccdm.solver import (
    EoSPoint, MODE_FRACTIONS, candidates_for, default_guess, mode_spec,
    residual, solve, solve_beta_eq_neutrino_trapped,
    solve_beta_eq_neutrinoless, solve_candidate, solve_fixed_yc,
    solve_fixed_yc_ys, warm_start,
)
from eos.ccdm.table import TableSpec, TableResult, build_table, quark_row
from eos.ccdm.api import PointResult, eos_point, eos_response, eos_table

__all__ = [
    "Parameters", "Derived", "PUBLISHED_SETS",
    "vector_coupling", "vector_coupling_derivative", "vector_field",
    "vector_self_energy", "rearrangement", "diquark_coupling", "has_vector",
    "SpeciesFlags", "BRANCHES", "PATTERNS", "DEFAULT_PATTERNS",
    "DENSITY_BRANCHES", "POTENTIAL_BRANCHES", "MODE_NAMES", "branch_seed",
    "pattern_mask", "pattern_seed",
    "CCDMState", "dielectric", "effective_masses", "glue_potential",
    "glue_derivative", "chiral_potential", "chiral_derivatives", "guard_phi",
    "mode_thermo", "bag_constant", "state_at", "thermo_from_mu",
    "EoSPoint", "MODE_FRACTIONS", "mode_spec", "default_guess", "warm_start",
    "residual", "candidates_for", "solve", "solve_candidate",
    "solve_beta_eq_neutrinoless", "solve_beta_eq_neutrino_trapped",
    "solve_fixed_yc", "solve_fixed_yc_ys",
    "TableSpec", "TableResult", "build_table", "quark_row",
    "PointResult", "eos_point", "eos_response", "eos_table",
]
