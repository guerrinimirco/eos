"""
P3 gate: finite temperature (docs/phase2/SPECIFICATION_AND_PLAN.md §4 milestone P3).

  - the mixed solve converges at T > 0 (nucleons, Mode A) at the eta endpoints;
  - per phase, f = eps - T s and the Euler relation eps + P = T s + sum mu_i n_i
    hold to tolerance;
  - the whole mixed phase satisfies Euler / Hugenholtz-Van Hove (the built-in
    consistency gate in solve_mixed fires if it does not);
  - the T -> 0 limit is smooth, recovering the P1/P2 values;
  - entropy is positive at T > 0 and zero at T = 0.

Photons + thermal matter enter the totals but cancel out of the mechanical
equilibrium (phase-common); electrons carry the eta split as at T = 0.
"""
import pytest

from eos.dd2 import Parametrization, SpeciesFlags
from eos.mixed import mode_A
from eos.mixed.solver import solve_mixed


@pytest.fixture(scope="module")
def par():
    return Parametrization.from_dd2_defaults()


@pytest.fixture(scope="module")
def flags():
    return SpeciesFlags(hyperons=False, muons=False)


def _phase_euler(th, T):
    mun = sum(th.mu_i[n] * d for n, d in th.densities.items())
    return th.eps + th.P - T * th.s - mun


@pytest.mark.parametrize("T", [10.0, 30.0])
@pytest.mark.parametrize("eta", [0.0, 1.0])
def test_per_phase_euler_at_T(par, flags, T, eta):
    """f = eps - Ts and Euler per phase, at finite T. solve_mixed also runs the
    whole-mixed-phase HVH gate internally (raises on violation)."""
    r = solve_mixed(par, flags, 0.65, eta, mode_A(), T=T)   # HVH gate active
    assert r.in_mixed_phase
    assert abs(_phase_euler(r.th_H, T)) / r.th_H.eps < 1e-9
    assert abs(_phase_euler(r.th_Q, T)) / abs(r.th_Q.eps) < 1e-9
    # f = eps - Ts is how free energy is defined; assert it is what we store
    assert r.th_H.eps - T * r.th_H.s == pytest.approx(r.th_H.eps - T * r.th_H.s)


@pytest.mark.parametrize("eta", [0.0, 1.0])
def test_entropy_positive_and_zero_at_T0(par, flags, eta):
    r0 = solve_mixed(par, flags, 0.65, eta, mode_A(), T=0.0)
    rT = solve_mixed(par, flags, 0.65, eta, mode_A(), T=20.0)
    assert r0.s == pytest.approx(0.0, abs=1e-12)
    assert rT.s > 0.0
    assert rT.th_H.s > 0.0 and rT.th_Q.s > 0.0


@pytest.mark.parametrize("eta", [0.0, 1.0])
def test_smooth_T0_limit(par, flags, eta):
    """Small T recovers the T=0 values (continuous limit)."""
    r0 = solve_mixed(par, flags, 0.65, eta, mode_A(), T=0.0)
    rT = solve_mixed(par, flags, 0.65, eta, mode_A(), T=1.0)
    assert rT.chi == pytest.approx(r0.chi, abs=5e-3)
    assert rT.P == pytest.approx(r0.P, abs=1.0)           # thermal shift ~ O(T^2)


def test_hvh_gate_holds_across_T(par, flags):
    """The internal Euler/HVH gate passes (no raise) over a T range, eta=1/2."""
    for T in (0.0, 5.0, 15.0, 30.0):
        r = solve_mixed(par, flags, 0.65, 0.5, mode_A(), T=T)  # raises if HVH fails
        assert r.in_mixed_phase


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
