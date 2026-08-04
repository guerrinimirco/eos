"""
The two sound speeds of a hadron-quark mixture.

The sharp checks are the pure-phase limits — at chi = 0 the frozen sound speed
must reproduce the DD2 nucleonic one exactly — and the Maxwell contrast, which
is the whole reason for having two definitions: across a constant-pressure
plateau c_eq goes to zero while c_ad, holding the phase fraction fixed, does
not.
"""
from dataclasses import replace

import numpy as np
import pytest

from eos.dd2 import Parametrization, SpeciesFlags
from eos.dd2.coefficients import sound_speed_adiabatic
from eos.mixed import (
    beta_eq_neutrinoless, locate_window, sweep_mixed,
    sound_speed_eq, sound_speed_frozen, adiabatic_index, frozen_along,
)
from eos.vmit.parameters import get_vmit_default


@pytest.fixture(scope="module")
def setup():
    par = Parametrization.from_dd2_defaults()
    flags = SpeciesFlags(hyperons=False, phi_field=False, muons=False)
    grid = np.linspace(0.1 * par.n_sat, 12.0 * par.n_sat, 200)
    return par, flags, grid, get_vmit_default()


def _window_states(setup, eta):
    par, flags, grid, vmit = setup
    spec = beta_eq_neutrinoless()
    w = locate_window(par, flags, grid, eta, spec, vmit_params=vmit, T=0.0)
    assert w.exists, f"no window at eta={eta}"
    inside = grid[(grid >= w.n_onset) & (grid <= w.n_offset)]
    return sweep_mixed(par, flags, inside, eta, spec, vmit_params=vmit, T=0.0)


def test_eq_is_dP_deps():
    """sound_speed_eq is exactly dP/deps — pinned on an analytic case where the
    answer is known, so the definition cannot drift."""
    eps = np.linspace(100.0, 500.0, 50)
    P = 0.25 * eps + 7.0                       # c^2 = 1/4 everywhere
    assert np.allclose(sound_speed_eq(P, eps), 0.25)
    assert np.isnan(sound_speed_eq([1.0], [1.0])).all()


def test_adiabatic_index_relation():
    eps = np.linspace(100.0, 500.0, 50)
    P = 0.25 * eps + 7.0
    assert np.allclose(adiabatic_index(P, eps), (eps + P) / P * 0.25)


def test_pure_hadronic_limit_matches_dd2(setup):
    """chi = 0, leptons off: must reproduce eos.dd2 sound_speed_adiabatic.

    That function is nucleonic matter at fixed Y_p with NO leptons, so the
    comparison is only like-for-like with leptons=False. With them on the
    values differ by a few percent, which is physics, not error.
    """
    par, flags, _grid, vmit = setup
    r = _window_states(setup, 0.0)[10]
    n_H = r.th_H.n_B
    Y_C = r.th_H.n_C / n_H

    got = sound_speed_frozen(par, flags, replace(r, chi=0.0), vmit_params=vmit,
                             leptons=False)
    want = sound_speed_adiabatic(par, n_H, Y_C, T=0.0)
    assert got == pytest.approx(want, rel=1e-8)

    with_leptons = sound_speed_frozen(par, flags, replace(r, chi=0.0),
                                      vmit_params=vmit, leptons=True)
    assert with_leptons != pytest.approx(want, rel=1e-4), \
        "leptons must change the answer; if not, they are not being added"


def test_pure_quark_limit_is_physical(setup):
    par, flags, _grid, vmit = setup
    r = _window_states(setup, 0.0)[10]
    c = sound_speed_frozen(par, flags, replace(r, chi=1.0), vmit_params=vmit)
    assert 0.0 < c <= 1.0


@pytest.mark.parametrize("eta", [0.0, 0.3, 1.0])
def test_frozen_exceeds_equilibrium_in_the_window(setup, eta):
    par, flags, _grid, vmit = setup
    rs = _window_states(setup, eta)
    P = np.array([r.P for r in rs])
    eps = np.array([r.eps for r in rs])
    c_eq = sound_speed_eq(P, eps)
    c_ad = frozen_along(par, flags, rs, vmit_params=vmit)

    good = np.isfinite(c_eq) & np.isfinite(c_ad)
    assert good.sum() == len(rs), "frozen sound speed left gaps in the window"
    assert np.all(c_ad[good] > c_eq[good])
    assert np.all(c_ad[good] > 0.0) and np.all(c_ad[good] <= 1.0)
    assert np.all(c_eq[good] >= -1e-6) and np.all(c_eq[good] <= 1.0)


def test_maxwell_plateau_collapses_c_eq_only(setup):
    """The contrast that justifies having both definitions."""
    par, flags, _grid, vmit = setup
    rs = _window_states(setup, 1.0)
    P = np.array([r.P for r in rs])
    eps = np.array([r.eps for r in rs])

    c_eq = sound_speed_eq(P, eps)
    c_ad = frozen_along(par, flags, rs, vmit_params=vmit)
    assert np.max(np.abs(c_eq)) < 1e-6, "eta=1 is a plateau: c_eq must vanish"
    assert np.min(c_ad) > 0.1, "c_ad must NOT collapse — chi is frozen"


def test_chi_outside_unit_interval_is_clamped(setup):
    """solve_mixed does not clamp chi (its sign classifies the phase), but a
    frozen sound speed at chi < 0 is undefined; the pure wing is the answer."""
    par, flags, _grid, vmit = setup
    r = _window_states(setup, 0.0)[10]
    lo = sound_speed_frozen(par, flags, replace(r, chi=-0.4), vmit_params=vmit)
    at0 = sound_speed_frozen(par, flags, replace(r, chi=0.0), vmit_params=vmit)
    assert lo == pytest.approx(at0, rel=1e-12)
