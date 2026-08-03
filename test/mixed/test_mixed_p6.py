"""
P6 gate: hyperons on -> nonzero HADRONIC strangeness in the mixed phase
(docs/phase2/SPECIFICATION_AND_PLAN.md §4 milestone P6).

  - with hyperons the hadronic phase carries n_S^H != 0, and the verified
    P_H = P_Q reduction (audit §1) holds numerically despite it;
  - the whole-mixed-phase Euler/HVH gate still holds (solve_mixed asserts it);
  - the S sector sums BOTH phases: fixing Y_S to Mode A's value reproduces
    Mode A even when n_S^H != 0 (strangeness conservation is global over H+Q);
  - SpeciesFlags gating: hyperons off -> n_S^H = 0 exactly.

DD2Y (from_dd2y_defaults) + phi field are needed for the hyperon couplings.
vMIT B4=160 places the transition in the hyperonic regime (with the default
B4=180 the softer hyperonic EoS has no quark transition here -- a physics fact,
not tuning). Tested at eta=0; the eta=1 Maxwell + hyperon solve is a known
continuation-robustness gap deferred to P8 (better predictors near the boundary).
"""
import numpy as np
import pytest

from eos.dd2 import Parametrization, SpeciesFlags
from eos.vmit.parameters import get_vmit_custom
from eos.mixed import mode_A
from eos.mixed.charges import ChargeSpec, Regime
from eos.mixed.solver import solve_mixed
from eos.mixed.continuation import sweep_mixed


@pytest.fixture(scope="module")
def par_y():
    return Parametrization.from_dd2y_defaults()


@pytest.fixture(scope="module")
def flags_y():
    return SpeciesFlags(hyperons=True, phi_field=True, muons=False)


@pytest.fixture(scope="module")
def vp():
    return get_vmit_custom(B4=160.0)


@pytest.fixture(scope="module")
def window(par_y, flags_y, vp):
    grid = np.arange(0.40, 0.90, 0.05)
    return [r for r in sweep_mixed(par_y, flags_y, grid, 0.0, mode_A(), vmit_params=vp)
            if r.in_mixed_phase]


def test_hadronic_strangeness_present(window):
    """The hadronic phase carries strangeness (hyperons populated)."""
    assert len(window) >= 4
    assert max(r.th_H.n_S for r in window) > 0.05     # genuinely strange
    # hyperons actually in the composition
    assert any(any(k.startswith(("Lambda", "Sigma", "Xi")) and v > 1e-4
                   for k, v in r.th_H.densities.items()) for r in window)


def test_mechanical_equilibrium_survives_strangeness(window):
    """P_H = P_Q holds numerically with n_S^H != 0 (audit §1 reduction)."""
    for r in window:
        assert r.th_H.n_S > 0.0 or r.n_B < 0.5        # strangeness onsets in-window
        assert r.th_H.P == pytest.approx(r.th_Q.P, rel=1e-8)  # eta=0: matter-only


def test_fixed_ys_reproduces_mode_A_with_hyperons(par_y, flags_y, vp):
    """Global strangeness conservation sums H+Q: fixing Y_S to Mode A's value
    reproduces Mode A even though n_S^H != 0."""
    rA = solve_mixed(par_y, flags_y, 0.7, 0.0, mode_A(), vmit_params=vp)
    Y_S = ((1 - rA.chi) * rA.th_H.n_S + rA.chi * rA.th_Q.n_S) / rA.n_B
    assert rA.th_H.n_S > 0.0                            # genuinely a strange point
    rD = solve_mixed(par_y, flags_y, 0.7, 0.0,
                     ChargeSpec(S=Regime.GLOBAL, targets={"Y_S": Y_S}), vmit_params=vp)
    assert rD.chi == pytest.approx(rA.chi, abs=1e-8)
    assert rD.potentials["mu_S"] == pytest.approx(0.0, abs=1e-6)


def test_speciesflags_gating(vp):
    """Hyperons off -> the hadronic phase has exactly zero strangeness."""
    r = solve_mixed(Parametrization.from_dd2_defaults(),
                    SpeciesFlags(hyperons=False, muons=False),
                    0.7, 0.0, mode_A(), vmit_params=vp)
    assert r.th_H.n_S == 0.0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
