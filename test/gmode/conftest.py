"""Shared fixtures for the g-mode tests.

Building a DD2 background means a beta-equilibrium sweep plus a frozen sound
speed at every density, so it is built once per session and reused.
"""
import numpy as np
import pytest

from eos.dd2.parametrization import Parametrization
from eos.dd2.species import SpeciesFlags
from eos.dd2.solver import sweep_beta_eq_octet
from eos.dd2.coefficients import sound_speed_eq
from eos.gmode.sound_speeds import cs2_frozen_nucleonic
from eos.gmode.background import with_crust
from eos.tov.solver import EOSTable_for_TOV


@pytest.fixture(scope="session")
def dd2_par():
    return Parametrization.from_dd2_defaults()


@pytest.fixture(scope="session")
def dd2_flags():
    return SpeciesFlags(muons=True)


@pytest.fixture(scope="session")
def dd2_eos(dd2_par, dd2_flags):
    """(EOSTable_for_TOV, cs2_eq, cs2_ad, n_B, Y_p) for cold DD2 npemu matter.

    Both sound speeds are computed the same way -- proper central differences on
    re-solved states, with the neutralising leptons included in both -- so that
    their difference is the composition effect and not a mismatch of
    conventions.
    """
    grid = np.geomspace(0.08, 1.2, 110)
    pts = sweep_beta_eq_octet(dd2_par, grid, dd2_flags, T=0.0,
                              include_photons=False, stop_at_boundary=True)
    P = np.array([p.P for p in pts])
    eps = np.array([p.eps for p in pts])
    n_B = np.array([p.n_B for p in pts])
    Y_p = np.array([p.Y_p for p in pts])

    cs2_eq = np.array([sound_speed_eq(dd2_par, n, dd2_flags, T=0.0)
                       for n in n_B])
    cs2_ad = np.array([cs2_frozen_nucleonic(dd2_par, n, y, muons=True)
                       for n, y in zip(n_B, Y_p)])

    core = EOSTable_for_TOV(P=P, epsilon=eps, nB=n_B)
    full, ceq, cad = with_crust(core, cs2_eq, cs2_ad, crust="BPS",
                                n_transition=0.08)
    return full, ceq, cad, n_B, Y_p


@pytest.fixture(scope="session")
def polytrope():
    """A Gamma = 2 polytrope, as a background with no composition at all.

    Returns (eos, cs2). Because it has one sound speed, N^2 vanishes and it must
    support no g-mode -- the sharpest available null test.
    """
    eps = np.logspace(np.log10(0.5), np.log10(4000.0), 1500)
    P = 2.0e-4 * eps**2
    eos = EOSTable_for_TOV(P=P, epsilon=eps, nB=eps / 939.0)
    return eos, np.gradient(P, eps)
