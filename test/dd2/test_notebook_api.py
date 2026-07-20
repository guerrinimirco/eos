"""
Smoke test for eos/dd2/notebook_api.py: every plot_* returns a Figure and every
compute_* returns finite numbers, on a coarse nucleonic grid (fast).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure

import pytest

from eos.dd2 import Parametrization
from eos.dd2 import notebook_api as api

PAR = Parametrization.from_dd2_defaults()
GRID = np.geomspace(0.06, 0.9, 10)


@pytest.fixture(scope="module")
def tov():
    return api.compute_tov(PAR, api.NUCLEONIC, n_ec=30)


@pytest.mark.parametrize("fn", [
    api.plot_p_vs_nb, api.plot_composition, api.plot_sound_speed,
    api.plot_p_vs_nb_snm, api.plot_pnm_chiral,
])
def test_simple_plots(fn):
    fig = fn(PAR, grid=GRID) if fn is api.plot_pnm_chiral \
        else fn(PAR, grid=GRID)
    assert isinstance(fig, Figure)


def test_isentropic_plot():
    fig = api.plot_isentropic_T(PAR, S_values=(2.0,), grid=GRID[:6])
    assert isinstance(fig, Figure)


def test_heat_capacity_plot():
    fig = api.plot_heat_capacity(PAR, grid=GRID[:6], T=10.0)
    assert isinstance(fig, Figure)


def test_tov_plots(tov):
    assert np.isfinite(tov["M_max"]) and tov["M_max"] > 1.5
    assert np.isfinite(tov["R_1p4"])
    assert isinstance(api.plot_mass_radius(PAR, tov=tov), Figure)
    assert isinstance(api.plot_lambda_mass(PAR, tov=tov), Figure)
    assert isinstance(api.plot_mr_with_constraints(PAR, tov=tov), Figure)


def test_nmp_comparison():
    cmp = api.nmp_comparison(PAR)
    for key, (val, ref, d) in cmp.items():
        assert np.isfinite(val) and np.isfinite(d)
    # DD2 par reproduces the DD2 reference to well within a percent-ish.
    assert cmp["n_sat"][2] < 1e-3
    assert isinstance(api.format_nmp_comparison(PAR), str)


def test_nmp_inversion_pass():
    par2, status = api.build_nmp_par()
    assert status.ok, status.message
    got = api.nmp_comparison(par2)
    # the L_sym nudge landed
    assert got["L_sym"][0] == pytest.approx(70.0, abs=0.5)
