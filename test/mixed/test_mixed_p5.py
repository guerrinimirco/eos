"""
P5 gate: Mode D (Y_S global) and Mode B (trapped neutrinos)
(docs/phase2/SPECIFICATION_AND_PLAN.md §4 milestone P5). The strongest available
test of the regime machinery is cross-mode reproduction:

  - Mode D with Y_S set to the value Mode C self-consistently produces returns
    Mode C's state (and mu_S -> 0 there);
  - Mode B with Y_L set to Mode A's value returns Mode A's state (mu_L -> 0),
    at T=0 where trapped neutrinos vanish;
  - Mode D fixes Y_S; Mode B at a higher Y_L genuinely traps neutrinos (mu_L>0,
    n_nue>0) at T>0;
  - the S and L_e sectors are regime-driven additions (spec §1.5): Modes A and C
    are unchanged, and the deferred readings (D-local S LOCAL) raise.

D-global is the default (audit §2). Nucleons + electrons; strangeness lives in
the quark phase (n_S^H = 0 without hyperons).
"""
import pytest

from eos.dd2 import Parametrization, SpeciesFlags
from eos.mixed import mode_A, mode_C, mode_D, mode_B, ChargeSpec, Regime
from eos.mixed.residual import mixed_slots
from eos.mixed.solver import solve_mixed


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
def test_mode_D_reproduces_mode_C(par, flags, eta):
    """Fix Y_S to Mode C's self-consistent value -> recover Mode C, mu_S -> 0."""
    rC = solve_mixed(par, flags, 0.65, eta, mode_C(0.10, yc_leptons=True))
    rD = solve_mixed(par, flags, 0.65, eta,
                     mode_D(0.10, _Y_S(rC), yc_leptons=True))
    assert rD.chi == pytest.approx(rC.chi, abs=1e-8)
    assert rD.P == pytest.approx(rC.P, rel=1e-8)
    assert rD.potentials["mu_S"] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("eta", [0.0, 0.5, 1.0])
def test_mode_B_reproduces_mode_A(par, flags, eta):
    """Fix Y_L to Mode A's value (= its charge fraction at T=0) -> recover Mode A,
    mu_L -> 0 (no neutrinos needed)."""
    rA = solve_mixed(par, flags, 0.65, eta, mode_A(), T=0.0)
    rB = solve_mixed(par, flags, 0.65, eta, mode_B(_Y_C(rA)), T=0.0)
    assert rB.chi == pytest.approx(rA.chi, abs=1e-8)
    assert rB.P == pytest.approx(rA.P, rel=1e-8)
    assert rB.potentials["mu_L"] == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------- targets are hit
def test_mode_D_fixes_Y_S(par, flags):
    r = solve_mixed(par, flags, 0.65, 0.0, mode_D(0.10, 0.30, yc_leptons=True))
    assert _Y_S(r) == pytest.approx(0.30, rel=1e-7)
    assert _Y_C(r) == pytest.approx(0.10, rel=1e-7)


def test_mode_B_traps_neutrinos(par, flags):
    """A high Y_L at T>0 genuinely populates trapped neutrinos."""
    r = solve_mixed(par, flags, 0.65, 0.0, mode_B(0.30), T=30.0)
    assert r.potentials["mu_L"] > 1.0            # neutrino potential switched on
    assert r.extras["n_nue"] > 0.0               # trapped neutrinos present


# ------------------------------------------------- regime discipline / guards
def test_modes_A_C_unchanged(par, flags):
    assert solve_mixed(par, flags, 0.65, 1.0, mode_A()).P == pytest.approx(258.08, abs=0.1)
    rC = solve_mixed(par, flags, 0.65, 0.0, mode_C(0.10, yc_leptons=True))
    assert _Y_C(rC) == pytest.approx(0.10, rel=1e-7)


def test_d_local_raises():
    with pytest.raises(NotImplementedError):     # S LOCAL (D-local) deferred, audit §2
        mixed_slots(ChargeSpec(S=Regime.LOCAL, targets={"Y_S": 0.1}), 0.0)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
