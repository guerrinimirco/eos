"""The stellar background: does it reproduce eos.tov, and is the metric right?

`eos.gmode.background` re-integrates the TOV equations because it needs the
radial profiles and the metric function nu, which `eos.tov.solver` does not
retain. That makes agreement with `eos.tov` on M and R the first thing to
check: the two are independent integrations of the same equations in different
unit systems.
"""
import numpy as np
import pytest

from eos.gmode.background import build_background, KM_PER_MSUN
from eos.tov.solver import solve_tov_single, _create_interpolators


def test_mass_radius_agree_with_eos_tov(polytrope):
    """M and R match eos.tov.solve_tov_single on the same table."""
    eos, cs2 = polytrope
    bg = build_background(eos, cs2, cs2, e_c=900.0, n_points=400)

    P_of_e, e_of_P, n_of_P = _create_interpolators(eos)
    ref = solve_tov_single(900.0, eos, P_of_e, e_of_P, n_of_P,
                           compute_baryonic=False, compute_tidal=False)

    # Two independent integrations with different surface criteria; a few parts
    # in 1e4 is the level at which they can be expected to agree.
    assert bg.M_msun == pytest.approx(ref.M, rel=1e-3)
    assert bg.R == pytest.approx(ref.R, rel=1e-3)


def test_metric_matches_exterior_schwarzschild(polytrope):
    """e^nu and e^lambda join the vacuum solution at the surface."""
    eos, cs2 = polytrope
    bg = build_background(eos, cs2, cs2, e_c=900.0, n_points=400)
    M, R = bg.M, bg.R

    assert bg.e_nu[-1] == pytest.approx(1.0 - 2.0 * M / R, rel=1e-10)
    assert bg.e_lam[-1] == pytest.approx(1.0 / (1.0 - 2.0 * M / R), rel=1e-10)
    # e^nu <= 1 everywhere, and rises monotonically outwards.
    assert np.all(bg.e_nu <= 1.0)
    assert np.all(np.diff(bg.e_nu) > 0.0)


def test_gravity_is_half_the_metric_gradient(polytrope):
    """g = -(dP/dr)/(eps+P) = nu'/2, the relation used to avoid a nu ODE."""
    eos, cs2 = polytrope
    bg = build_background(eos, cs2, cs2, e_c=900.0, n_points=1200)
    nu = np.log(bg.e_nu)
    # Interior only: a centred difference is one-sided at the endpoints, where
    # g varies fastest.
    sl = slice(5, -5)
    dnu = np.gradient(nu, bg.r)[sl]
    assert np.allclose(bg.g[sl], 0.5 * dnu, rtol=5e-3)


def test_equal_sound_speeds_give_zero_buoyancy(polytrope):
    """One sound speed means no composition gradient, hence N^2 == 0 exactly."""
    eos, cs2 = polytrope
    bg = build_background(eos, cs2, cs2, e_c=900.0, n_points=300)
    assert np.all(bg.N2 == 0.0)


def test_buoyancy_positive_when_frozen_speed_is_larger(polytrope):
    """c_ad > c_eq is convective stability, N^2 >= 0."""
    eos, cs2 = polytrope
    bg = build_background(eos, cs2, cs2 + 0.002, e_c=900.0, n_points=300)
    assert np.all(bg.N2 >= 0.0)
    assert bg.N2.max() > 0.0


def test_solve_at_target_mass(polytrope):
    """The central-density search lands on the requested mass."""
    eos, cs2 = polytrope
    bg = build_background(eos, cs2, cs2, M_target=1.4, n_points=200)
    assert bg.M_msun == pytest.approx(1.4, abs=2e-3)


def test_target_mass_above_maximum_raises(polytrope):
    """Asking for a mass the equation of state cannot support fails loudly."""
    eos, cs2 = polytrope
    with pytest.raises(ValueError, match="not reached by this equation"):
        build_background(eos, cs2, cs2, M_target=50.0, n_points=100)


def test_dd2_background_is_physical(dd2_eos):
    """A real DD2 star: sane M, R, positive buoyancy, causal sound speeds."""
    eos, ceq, cad, _n, _y = dd2_eos
    bg = build_background(eos, ceq, cad, M_target=1.4, n_points=500)

    assert bg.M_msun == pytest.approx(1.4, abs=5e-3)
    assert 11.0 < bg.R < 15.0                       # DD2 is a stiff EoS
    assert np.all(np.isfinite(bg.N2))
    assert np.all(bg.N2 >= 0.0), "DD2 must be convectively stable"
    assert np.all(bg.cs2_eq > 0.0) and np.all(bg.cs2_eq <= 1.0)
    assert bg.M == pytest.approx(bg.M_msun * KM_PER_MSUN, rel=1e-12)
