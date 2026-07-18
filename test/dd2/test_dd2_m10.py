"""
M10 gate: thermodynamic coefficients + full verification suite.

Gate (report §3.y M10): analytic-vs-numeric coefficient agreement; all verify
checks green on eos_ref. Backend parity (eos_ref vs eos_fast) is the M9 check.

Here the "numeric" cross-check is a second, independent finite-difference of
each coefficient (different step / neighbouring method); the JAX-autodiff
"analytic" value is added at M9 and held to the same agreement.
"""
import numpy as np
import pytest

from eos.dd2 import Parametrization, SpeciesFlags
from eos.dd2.coefficients import (
    sound_speed_eq, sound_speed_adiabatic, thermal_index, heat_capacity_V,
    snm_sound_speed,
)
from eos.dd2.verify.run_full_check import run_full_check


@pytest.fixture(scope="module")
def par():
    return Parametrization.from_dd2_defaults()


@pytest.fixture(scope="module")
def flags():
    return SpeciesFlags(hyperons=False, phi_field=False)


def test_sound_speed_causal_and_rising(par, flags):
    cs2 = [sound_speed_eq(par, n, flags) for n in (0.16, 0.32, 0.6, 1.0)]
    assert all(0.0 <= c <= 1.0 for c in cs2)         # causal
    assert cs2[0] < cs2[1] < cs2[2] < cs2[3]         # stiffens with density


def test_sound_speed_step_convergence(par, flags):
    # independent numeric cross-check: two FD steps agree (Richardson-stable)
    c1 = sound_speed_eq(par, 0.4, flags, rel_dn=1e-3)
    c2 = sound_speed_eq(par, 0.4, flags, rel_dn=5e-4)
    assert c1 == pytest.approx(c2, rel=1e-3)


def test_adiabatic_equals_frozen_at_T0(par):
    # at T=0 the adiabatic (frozen-composition) c_s^2 is the fixed-composition
    # derivative; cross-check SNM against the composition route at Y_p=0.5
    cs_snm = snm_sound_speed(par, 0.3)
    cs_comp = sound_speed_adiabatic(par, 0.3, Y_p=0.5)
    assert cs_snm == pytest.approx(cs_comp, rel=1e-6)


def test_thermal_index_physical(par, flags):
    for n in (0.16, 0.4):
        gth = thermal_index(par, n, flags, T=10.0)
        assert 1.0 < gth < 2.5                       # typical nucleonic range


def test_heat_capacity_positive(par, flags):
    for T in (5.0, 20.0):
        assert heat_capacity_V(par, 0.16, flags, T=T) > 0.0


def test_run_full_check_all_pass(par, flags):
    report = run_full_check(par, flags,
                            grid=[0.1, 0.16, 0.3, 0.5], include_tov=False)
    assert report.all_passed, str(report)
    names = {r.name for r in report.results}
    assert {"golden points", "thermo identities", "coefficients"} <= names


def test_run_full_check_tov(par, flags):
    report = run_full_check(par, flags, grid=[0.16, 0.3], include_tov=True)
    tov = next(r for r in report.results if r.name == "TOV M_max>=2")
    assert tov.passed


if __name__ == "__main__":
    print(run_full_check())
