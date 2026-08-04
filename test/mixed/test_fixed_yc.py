"""
Fixed non-leptonic charge fraction Y_C, in both lepton flavors.

  - fixing Y_C at the value beta equilibrium produces for itself must return the
    beta-equilibrium state. Two different unknown vectors agreeing on one
    physical state is the strongest available check that the regime machinery
    is right;
  - at eta=1 with Y_C held fixed the total pressure is NOT constant across the
    mixed phase. That is the physics point of the framework, and asserting a
    plateau here would be asserting a bug — in deliberate contrast to the
    beta-equilibrium Maxwell case;
  - the leptonless flavor is eta-independent and pins Y_C exactly: with no
    leptons there is no neutrality condition to localize;
  - fixing Y_C shares one residual with beta equilibrium, so wiring it leaves
    the beta-equilibrium answers untouched.

Nucleons and electrons, T=0. The neutralizing-lepton flavor populates electrons
so that the total system is electrically neutral.
"""
import numpy as np
import pytest

from eos.dd2 import Parametrization, SpeciesFlags
from eos.mixed import beta_eq_neutrinoless, fixed_YC
from eos.mixed.solvers.point import solve_mixed
from eos.mixed.solvers.sweep import sweep_mixed


@pytest.fixture(scope="module")
def par():
    return Parametrization.from_dd2_defaults()


@pytest.fixture(scope="module")
def flags():
    return SpeciesFlags(hyperons=False, muons=False)


def _Y_C(r):
    return ((1 - r.chi) * r.th_H.n_C + r.chi * r.th_Q.n_C) / r.n_B


@pytest.mark.parametrize("eta", [0.0, 0.5, 1.0])
def test_fixed_yc_reproduces_beta_eq(par, flags, eta):
    """Fix Y_C to beta equilibrium's self-consistent value -> recover beta equilibrium exactly."""
    rA = solve_mixed(par, flags, 0.65, eta, beta_eq_neutrinoless())
    rC = solve_mixed(par, flags, 0.65, eta, fixed_YC(_Y_C(rA), leptons=True))
    assert rC.chi == pytest.approx(rA.chi, abs=1e-8)
    assert rC.P == pytest.approx(rA.P, rel=1e-8)
    assert rC.th_H.densities["p"] == pytest.approx(rA.th_H.densities["p"], rel=1e-7)


def test_fixed_yc_eta1_is_not_a_plateau(par, flags):
    """eta=1 with Y_C fixed: P VARIES across the window (contrast beta-equilibrium Maxwell)."""
    grid = np.arange(0.45, 0.85, 0.05)
    mixed = [r for r in sweep_mixed(par, flags, grid, 1.0, fixed_YC(0.10, leptons=True))
             if r.in_mixed_phase]
    assert len(mixed) >= 3
    Ps = [r.P for r in mixed]
    assert max(Ps) - min(Ps) > 5.0             # NOT constant (the physics point)


def test_charge_conservation(par, flags):
    """Total non-leptonic charge fraction equals the fixed Y_C target."""
    for eta in (0.0, 1.0):
        r = solve_mixed(par, flags, 0.65, eta, fixed_YC(0.10, leptons=True))
        assert _Y_C(r) == pytest.approx(0.10, rel=1e-8)


def test_leptonless_is_eta_independent(par, flags):
    """2a leptonless: charged slice, Y_C pinned, no eta dependence."""
    states = [solve_mixed(par, flags, 0.65, eta, fixed_YC(0.30, leptons=False))
              for eta in (0.0, 0.5, 1.0)]
    for r in states:
        assert _Y_C(r) == pytest.approx(0.30, rel=1e-8)
    assert states[0].chi == pytest.approx(states[1].chi, abs=1e-10)
    assert states[1].P == pytest.approx(states[2].P, rel=1e-10)


def test_beta_eq_unchanged_by_fixed_yc_wiring(par, flags):
    """Regression: beta equilibrium endpoints unchanged after the C-sector refactor."""
    r0 = solve_mixed(par, flags, 0.65, 0.0, beta_eq_neutrinoless())
    r1 = solve_mixed(par, flags, 0.65, 1.0, beta_eq_neutrinoless())
    assert r0.chi == pytest.approx(0.2469, abs=1e-3)     # Gibbs
    assert r1.P == pytest.approx(258.08, abs=0.1)        # Maxwell plateau value


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
