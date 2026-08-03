"""
P4 gate: Mode C, fixed charge fraction Y_C (docs/phase2/SPECIFICATION_AND_PLAN.md
§4 milestone P4), both §1.6 lepton flavors.

  - Mode C at Y_C = the value Mode A self-consistently produces must return
    Mode A's state (the strongest test that the regime machinery is right);
  - at eta=1 with Y_C fixed, total pressure is NOT constant across the mixed
    phase -- this is the physics point of the framework (a plateau assertion
    here would be asserting the bug), in deliberate contrast to Mode A Maxwell;
  - the leptonless flavor (2a) is eta-independent and pins Y_C exactly (a
    charged CompOSE (n_B,T,Y_q) slice, no electrons);
  - Mode C shares ONE regime-driven residual with Mode A (spec §1.5): wiring it
    does not change Mode A.

Nucleons + electrons, T=0, DD2+vMIT. Mode C 2b uses neutralizing electrons.
"""
import numpy as np
import pytest

from eos.dd2 import Parametrization, SpeciesFlags
from eos.mixed import mode_A, mode_C
from eos.mixed.solver import solve_mixed
from eos.mixed.continuation import sweep_mixed


@pytest.fixture(scope="module")
def par():
    return Parametrization.from_dd2_defaults()


@pytest.fixture(scope="module")
def flags():
    return SpeciesFlags(hyperons=False, muons=False)


def _Y_C(r):
    return ((1 - r.chi) * r.th_H.n_C + r.chi * r.th_Q.n_C) / r.n_B


@pytest.mark.parametrize("eta", [0.0, 0.5, 1.0])
def test_mode_C_reproduces_mode_A(par, flags, eta):
    """Fix Y_C to Mode A's self-consistent value -> recover Mode A exactly."""
    rA = solve_mixed(par, flags, 0.65, eta, mode_A())
    rC = solve_mixed(par, flags, 0.65, eta, mode_C(_Y_C(rA), yc_leptons=True))
    assert rC.chi == pytest.approx(rA.chi, abs=1e-8)
    assert rC.P == pytest.approx(rA.P, rel=1e-8)
    assert rC.th_H.densities["p"] == pytest.approx(rA.th_H.densities["p"], rel=1e-7)


def test_fixed_yc_eta1_is_not_a_plateau(par, flags):
    """eta=1 with Y_C fixed: P VARIES across the window (contrast Mode A Maxwell)."""
    grid = np.arange(0.45, 0.85, 0.05)
    mixed = [r for r in sweep_mixed(par, flags, grid, 1.0, mode_C(0.10, yc_leptons=True))
             if r.in_mixed_phase]
    assert len(mixed) >= 3
    Ps = [r.P for r in mixed]
    assert max(Ps) - min(Ps) > 5.0             # NOT constant (the physics point)


def test_charge_conservation(par, flags):
    """Total non-leptonic charge fraction equals the fixed Y_C target."""
    for eta in (0.0, 1.0):
        r = solve_mixed(par, flags, 0.65, eta, mode_C(0.10, yc_leptons=True))
        assert _Y_C(r) == pytest.approx(0.10, rel=1e-8)


def test_leptonless_is_eta_independent(par, flags):
    """2a leptonless: charged slice, Y_C pinned, no eta dependence."""
    states = [solve_mixed(par, flags, 0.65, eta, mode_C(0.30, yc_leptons=False))
              for eta in (0.0, 0.5, 1.0)]
    for r in states:
        assert _Y_C(r) == pytest.approx(0.30, rel=1e-8)
    assert states[0].chi == pytest.approx(states[1].chi, abs=1e-10)
    assert states[1].P == pytest.approx(states[2].P, rel=1e-10)


def test_mode_A_unchanged_by_mode_C_wiring(par, flags):
    """Regression: Mode A endpoints unchanged after the C-sector refactor."""
    r0 = solve_mixed(par, flags, 0.65, 0.0, mode_A())
    r1 = solve_mixed(par, flags, 0.65, 1.0, mode_A())
    assert r0.chi == pytest.approx(0.2469, abs=1e-3)     # Gibbs
    assert r1.P == pytest.approx(258.08, abs=0.1)        # Maxwell plateau value


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
