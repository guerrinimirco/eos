"""
The strangeness and lepton-number sectors: fixed Y_S, and trapped neutrinos.

Cross-mode reproduction is the theme — a mode with an extra conserved charge,
fixed at the value a simpler mode produces for itself, must return that simpler
mode's state with the extra potential going to zero:

  - fixing Y_S at the fixed-Y_C mode's own value returns that state, mu_S -> 0;
  - fixing Y_L at beta equilibrium's own value returns beta equilibrium,
    mu_L -> 0, at T=0 where trapped neutrinos vanish anyway;
  - away from those points the constraints bite: Y_S is genuinely pinned, and a
    higher Y_L genuinely traps neutrinos (mu_L > 0, n_nue > 0) at T > 0;
  - both sectors are regime-driven additions, so the simpler modes are
    unchanged and the unwired readings raise rather than mis-assemble.
"""
import pytest

from eos.dd2 import Parametrization, SpeciesFlags
from eos.mixed import beta_eq_neutrinoless, fixed_YC, fixed_YC_YS, beta_eq_neutrino_trapped, ChargeSpec, Regime
from eos.mixed.equilibrium.residual import mixed_slots
from eos.mixed.solvers.point import solve_mixed


@pytest.fixture(scope="module")
def par():
    return Parametrization.from_dd2_defaults()


@pytest.fixture(scope="module")
def flags():
    return SpeciesFlags(hyperons=False, muons=False)


def _Y_C(r):
    return ((1 - r.chi) * r.th_H.n_C + r.chi * r.th_Q.n_C) / r.n_B


def _Y_S(r):
    return ((1 - r.chi) * r.th_H.n_S + r.chi * r.th_Q.n_S) / r.n_B


# ------------------------------------------------------------ cross-checks
@pytest.mark.parametrize("eta", [0.0, 0.5, 1.0])
def test_fixed_ys_reproduces_fixed_yc(par, flags, eta):
    """Fix Y_S to the fixed-Y_C state's self-consistent value -> recover the fixed-Y_C mode, mu_S -> 0."""
    rC = solve_mixed(par, flags, 0.65, eta, fixed_YC(0.10, leptons=True))
    rD = solve_mixed(par, flags, 0.65, eta,
                     fixed_YC_YS(0.10, _Y_S(rC), leptons=True))
    assert rD.chi == pytest.approx(rC.chi, abs=1e-8)
    assert rD.P == pytest.approx(rC.P, rel=1e-8)
    assert rD.potentials["mu_S"] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("eta", [0.0, 0.5, 1.0])
def test_trapped_reproduces_beta_eq(par, flags, eta):
    """Fix Y_L to beta equilibrium's value (= its charge fraction at T=0) -> recover beta equilibrium,
    mu_L -> 0 (no neutrinos needed)."""
    rA = solve_mixed(par, flags, 0.65, eta, beta_eq_neutrinoless(), T=0.0)
    rB = solve_mixed(par, flags, 0.65, eta, beta_eq_neutrino_trapped(_Y_C(rA)), T=0.0)
    assert rB.chi == pytest.approx(rA.chi, abs=1e-8)
    assert rB.P == pytest.approx(rA.P, rel=1e-8)
    assert rB.potentials["mu_L"] == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------- targets are hit
def test_fixed_ys_pins_Y_S(par, flags):
    r = solve_mixed(par, flags, 0.65, 0.0, fixed_YC_YS(0.10, 0.30, leptons=True))
    assert _Y_S(r) == pytest.approx(0.30, rel=1e-7)
    assert _Y_C(r) == pytest.approx(0.10, rel=1e-7)


def test_trapped_mode_traps_neutrinos(par, flags):
    """A high Y_L at T>0 genuinely populates trapped neutrinos."""
    r = solve_mixed(par, flags, 0.65, 0.0, beta_eq_neutrino_trapped(0.30), T=30.0)
    assert r.potentials["mu_L"] > 1.0            # neutrino potential switched on
    assert r.extras["nu"].n > 0.0               # trapped neutrinos present


# ------------------------------------------------- regime discipline / guards
def test_modes_A_C_unchanged(par, flags):
    assert solve_mixed(par, flags, 0.65, 1.0, beta_eq_neutrinoless()).P == pytest.approx(258.08, abs=0.1)
    rC = solve_mixed(par, flags, 0.65, 0.0, fixed_YC(0.10, leptons=True))
    assert _Y_C(rC) == pytest.approx(0.10, rel=1e-7)


def test_d_local_raises():
    with pytest.raises(NotImplementedError):     # per-phase Y_S is not wired
        mixed_slots(ChargeSpec(S=Regime.LOCAL, targets={"Y_S": 0.1}), 0.0)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
