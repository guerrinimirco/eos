"""The two sound speeds, their mixed-phase conventions, and finite rates.

The g-mode is driven entirely by c_ad^2 - c_eq^2, so these quantities have to
be right before any eigenfrequency means anything.
"""
import numpy as np
import pytest

from eos.dd2.coefficients import sound_speed_adiabatic, sound_speed_eq
from eos.gmode.sound_speeds import (
    cs2_frozen_isobaric, cs2_frozen_nucleonic, cs2_dynamical, bulk_viscosity,
)


def test_leptonless_limit_reproduces_dd2_adiabatic(dd2_par):
    """With the leptons switched off this is eos.dd2's own frozen sound speed.

    That function is the validated reference; agreement to machine precision
    shows the only difference in the default path is the deliberate one, the
    neutralising lepton gas.
    """
    for n_B, Y_p in ((0.16, 0.05), (0.32, 0.09), (0.64, 0.12)):
        ours = cs2_frozen_nucleonic(dd2_par, n_B, Y_p, leptons=False)
        ref = sound_speed_adiabatic(dd2_par, n_B, Y_p)
        assert ours == pytest.approx(ref, rel=1e-10)


def test_leptons_change_the_frozen_speed_appreciably(dd2_par, dd2_flags):
    """Including the leptons is not a small correction for this purpose.

    In absolute terms the shift is only a few times 1e-4 in c^2, but the
    quantity that drives the g-mode is c_ad^2 - c_eq^2, which is itself only a
    few times 1e-3 here. The lepton term is a sizeable fraction of the entire
    signal, which is why both sound speeds must be computed for the same
    fluid.
    """
    n_B, Y_p = 0.32, 0.09
    with_lep = cs2_frozen_nucleonic(dd2_par, n_B, Y_p, leptons=True)
    without = cs2_frozen_nucleonic(dd2_par, n_B, Y_p, leptons=False)
    buoyancy = with_lep - sound_speed_eq(dd2_par, n_B, dd2_flags, T=0.0)

    assert abs(with_lep - without) > 1e-4
    assert abs(with_lep - without) > 0.1 * abs(buoyancy)


def test_dd2_is_convectively_stable(dd2_eos):
    """c_ad^2 > c_eq^2 at every density, computed consistently."""
    _eos, ceq, cad, _n, _y = dd2_eos
    diff = cad - ceq
    assert np.all(diff > -1e-12), f"min(c_ad^2 - c_eq^2) = {diff.min()}"
    assert diff.max() > 1e-4, "no composition gradient means no g-mode"


# --------------------------------------------------------------- mixed phase

def test_isobaric_reduces_to_the_pure_phases():
    """chi = 0 gives the hadronic value, chi = 1 the quark one."""
    assert cs2_frozen_isobaric(0.3, 0.5, 0.0) == pytest.approx(0.3)
    assert cs2_frozen_isobaric(0.3, 0.5, 1.0) == pytest.approx(0.5)
    # the absent phase may be unknown without poisoning the limit
    assert cs2_frozen_isobaric(0.3, np.nan, 0.0) == pytest.approx(0.3)
    assert cs2_frozen_isobaric(np.nan, 0.5, 1.0) == pytest.approx(0.5)


def test_isobaric_is_the_reciprocal_combination():
    """1/c^2 = (1-chi)/c_H^2 + chi/c_Q^2, resistors in parallel."""
    cs2_H, cs2_Q, chi = 0.25, 0.60, 0.4
    expect = 1.0 / ((1 - chi) / cs2_H + chi / cs2_Q)
    assert cs2_frozen_isobaric(cs2_H, cs2_Q, chi) == pytest.approx(expect)
    # It lies between the two phase values, and below the volume-weighted mean:
    # the soft phase dominates a series combination.
    assert cs2_H < expect < cs2_Q
    assert expect < (1 - chi) * cs2_H + chi * cs2_Q


def test_isobaric_is_monotone_in_chi():
    chi = np.linspace(0.0, 1.0, 21)
    out = cs2_frozen_isobaric(0.2, 0.7, chi)
    assert np.all(np.diff(out) > 0.0)


# ------------------------------------------------------------ finite rates

def test_dynamical_sound_speed_limits():
    """gamma -> 0 recovers frozen, gamma -> infinity recovers equilibrium."""
    cs2_eq, cs2_ad, omega = 0.20, 0.28, 2 * np.pi * 300.0
    frozen = cs2_dynamical(cs2_eq, cs2_ad, 1e-8 * omega, omega)
    equil = cs2_dynamical(cs2_eq, cs2_ad, 1e8 * omega, omega)
    assert frozen.real == pytest.approx(cs2_ad, rel=1e-6)
    assert equil.real == pytest.approx(cs2_eq, rel=1e-6)
    assert abs(frozen.imag) < 1e-6
    assert abs(equil.imag) < 1e-6


def test_dynamical_real_part_is_bounded_by_the_two_limits():
    cs2_eq, cs2_ad, omega = 0.20, 0.28, 2 * np.pi * 300.0
    gamma = np.logspace(-2, 6, 40) * omega
    dy = cs2_dynamical(cs2_eq, cs2_ad, gamma, omega)
    assert np.all(dy.real >= cs2_eq - 1e-12)
    assert np.all(dy.real <= cs2_ad + 1e-12)


def test_dissipation_is_positive_and_peaks_at_resonance():
    """Im[c_dy^2] >= 0, maximal where gamma = omega."""
    cs2_eq, cs2_ad, omega = 0.20, 0.28, 2 * np.pi * 300.0
    gamma = np.logspace(-3, 3, 601) * omega
    dy = cs2_dynamical(cs2_eq, cs2_ad, gamma, omega)
    assert np.all(dy.imag >= -1e-15)
    assert gamma[np.argmax(dy.imag)] == pytest.approx(omega, rel=0.05)


def test_bulk_viscosity_tracks_the_imaginary_part():
    omega = 2 * np.pi * 300.0
    dy = cs2_dynamical(0.20, 0.28, omega, omega)
    zeta = bulk_viscosity(dy, 400.0, 60.0, omega)
    assert zeta == pytest.approx((400.0 + 60.0) * dy.imag / omega)
    assert zeta > 0.0
