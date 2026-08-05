"""Beta-equilibration rates: the temperature that decides whether a g-mode lives.

The calibration point is Alford, Harris, Harutyunyan and Sedrakian,
arXiv:1907.03795, Fig. 2(a): for the DD2 equation of state subjected to a 1 kHz
density oscillation, the bulk viscosity -- which peaks exactly where the
equilibration rate matches the oscillation frequency, gamma = omega -- reaches
its maximum at about 3 MeV using exact Urca rates, and the paper states that
the Fermi surface approximation puts the peak 1-2 MeV *higher*, so near
4-5 MeV. It also reports the peak temperature to be essentially independent of
density for DD2. Both features are checked here, since this module implements
the Fermi-surface rates.

DD2 is the right EoS for this test because it has no direct Urca threshold at
any density in the Fermi surface approximation, so the comparison isolates the
modified Urca channel.
"""
import numpy as np
import pytest
from scipy.optimize import brentq

from eos.dd2.solver import solve_beta_eq_octet
from eos.gmode.rates import (
    equilibration_rate, susceptibility_A, lambda_direct_urca,
    lambda_modified_urca,
)

OMEGA_1KHZ = 2.0 * np.pi * 1000.0


def _beta_Y_p(par, flags, n_B, T=1.0):
    return solve_beta_eq_octet(par, n_B, flags, T=T).Y_p


@pytest.mark.parametrize("n_over_n0", [1.0, 3.0, 5.0])
def test_resonance_temperature_matches_published_dd2(dd2_par, dd2_flags,
                                                     n_over_n0):
    """gamma = omega(1 kHz) near 4-5 MeV, as the Fermi-surface rates give."""
    n_B = n_over_n0 * 0.16
    Y_p = _beta_Y_p(dd2_par, dd2_flags, n_B)

    T_cross = brentq(
        lambda T: equilibration_rate(dd2_par, n_B, Y_p, T) - OMEGA_1KHZ,
        0.5, 20.0)
    assert 3.5 < T_cross < 6.5, f"gamma = omega at T = {T_cross:.2f} MeV"


def test_resonance_temperature_is_density_independent(dd2_par, dd2_flags):
    """For DD2 the peak sits at the same temperature at every density."""
    crossings = []
    for n_over_n0 in (1.0, 2.0, 3.0, 5.0):
        n_B = n_over_n0 * 0.16
        Y_p = _beta_Y_p(dd2_par, dd2_flags, n_B)
        crossings.append(brentq(
            lambda T: equilibration_rate(dd2_par, n_B, Y_p, T) - OMEGA_1KHZ,
            0.5, 20.0))
    assert max(crossings) - min(crossings) < 1.0, crossings


def test_cold_matter_is_frozen(dd2_par, dd2_flags):
    """At 0.1 MeV the rate is far below a g-mode frequency: composition frozen.

    This is the regime in which the frozen sound speed, and hence the whole
    cold-star g-mode calculation, is the correct limit.
    """
    n_B = 0.32
    Y_p = _beta_Y_p(dd2_par, dd2_flags, n_B)
    gamma = equilibration_rate(dd2_par, n_B, Y_p, 0.1)
    assert gamma < 1e-3 * OMEGA_1KHZ


def test_zero_temperature_rate_vanishes(dd2_par):
    assert equilibration_rate(dd2_par, 0.32, 0.1, 0.0) == 0.0


def test_rate_scaling_with_temperature(dd2_par, dd2_flags):
    """Modified Urca gives gamma ~ T^6 where it dominates."""
    n_B = 0.32
    Y_p = _beta_Y_p(dd2_par, dd2_flags, n_B)
    lo = equilibration_rate(dd2_par, n_B, Y_p, 0.5, processes="modified")
    hi = equilibration_rate(dd2_par, n_B, Y_p, 1.0, processes="modified")
    assert hi / lo == pytest.approx(2.0**6, rel=0.05)


def test_susceptibility_is_positive(dd2_par, dd2_flags):
    """A = (d mu_Delta / d n_n) > 0 for any isospin-stable equation of state."""
    for n_over_n0 in (1.0, 2.0, 4.0):
        n_B = n_over_n0 * 0.16
        Y_p = _beta_Y_p(dd2_par, dd2_flags, n_B)
        assert susceptibility_A(dd2_par, n_B, Y_p, T=1.0) > 0.0


def test_direct_urca_is_blocked_below_threshold():
    """The triangle inequality switches direct Urca off, and on again above."""
    # p_Fn > p_Fp + p_Fe: forbidden
    assert lambda_direct_urca(400.0, 100.0, 100.0, 600.0, 600.0, 1.0) == 0.0
    # p_Fn < p_Fp + p_Fe: allowed
    assert lambda_direct_urca(150.0, 100.0, 100.0, 600.0, 600.0, 1.0) > 0.0


def test_modified_urca_has_no_threshold():
    """It is the only channel below the direct Urca onset, so it must not vanish."""
    lam = lambda_modified_urca(400.0, 100.0, 100.0, 939.6, 938.3, 1.0)
    assert lam > 0.0
