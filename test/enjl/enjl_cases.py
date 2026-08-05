"""Shared setup for the extended-NJL reference-table tests.

Both `test_enjl_reference.py` (do the tables satisfy the model's identities?)
and `test_enjl_fixed_composition.py` (does the solver reproduce the tables?)
need the same three things: the parameter set that produced each file, the mask
of rows that are genuine solver output, and the bridge between the model's
species names and the tables' column suffixes. They live here so the two test
modules cannot drift apart on any of them.

This is not a test module; pytest does not collect it.
"""
from functools import lru_cache

import numpy as np

from eos.enjl.parameters import ENJLParams
from eos.enjl.species import make_species
from eos.enjl.thermodynamics import kF_from_n, scalar_density_t0
from eos.general.physics_constants import hc3

from reference import PARAMETER_SETS, bad_rows, load_reference, solved_rows

#: model species name -> reference-table column suffix. The tables write "L"
#: for the Lambda hyperon where `eos.enjl` writes "Lambda"; every other name
#: coincides.
REF_COL = {"p": "p", "n": "n", "Lambda": "L",
           "u": "u", "d": "d", "s": "s", "e": "e", "mu": "mu"}
BARYONS = ("p", "n", "Lambda")
QUARKS = ("u", "d", "s")


@lru_cache(maxsize=None)
def case(filename):
    """(columns, usable-row mask, params, alpha_S per row) for one file.

    The mask keeps only rows that are genuine solver output: `solved_rows`
    drops the interpolated mixed-phase rows of the f_q = 0.5 file, `bad_rows`
    the non-converged rows of the f_q = 0.7, B = 0 file.
    """
    f_q, B_GeV = PARAMETER_SETS[filename]
    par = ENJLParams(f_q=f_q, B_GeV_fm3=B_GeV)
    col = load_reference(filename)
    ok = solved_rows(col) & ~bad_rows(col, filename)
    alpha = np.array([par.alpha_S(nb * hc3) for nb in col["nB"]])
    return col, ok, par, alpha


def species_of(par):
    """{name -> Species} carrying the conserved charges B, Q and the content N."""
    return {sp.name: sp for sp in make_species(par.f_Lambda, par.f_q)}


def table_masses(col, i):
    """The row's own (M_u, M_d, M_s) [MeV], as a gap-solve starting point."""
    return [col["Mu"][i], col["Md"][i], col["Ms"][i]]


def row_densities(col, i):
    """The row's species densities in MeV^3, keyed for `solve_point`."""
    return {name: col["n" + suffix][i] * hc3
            for name, suffix in REF_COL.items()}


def chirally_restored(col, par, q):
    """Rows where M_q sits exactly at m_q0 and `Sigmaq` is written as 0.

    The condensate of flavor q has vanished there. The tables report the
    vanishing effective scalar density as an exact zero, which is the value the
    model's own cap produces, but it means those rows carry no information
    about the size of nbar^s_q and so cannot test Eq. (6).
    """
    m0 = {"u": par.m_u0, "d": par.m_d0, "s": par.m_s0}[q]
    return np.abs(col["M" + q] - m0) < 1.0e-9


def scalar_density_column(n_fm, mass, g, Lambda):
    """n^s [fm^-3] along a column, through the shipped Eq. (12) closed form.

    `scalar_density_t0` is scalar-valued (it is the T = 0 reference path and
    stays that way), so the sweep is an explicit comprehension rather than a
    second, vectorized copy of the same algebra.
    """
    kF = kF_from_n(np.asarray(n_fm) * hc3, g)
    return np.array([scalar_density_t0(k, m, g, Lambda)
                     for k, m in zip(kF, np.asarray(mass))]) / hc3


def worst(residual, mask):
    """Largest |residual| over the masked rows; 0.0 if the mask is empty."""
    return float(np.nanmax(np.abs(residual[mask]))) if mask.any() else 0.0
