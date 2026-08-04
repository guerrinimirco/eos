"""
Beta equilibrium at T=0: the two limiting constructions of the transition.

  - eta=1 reproduces a MAXWELL construction — total pressure is constant across
    the mixed phase (one transition pressure) while chi sweeps 0 -> 1;
  - eta=0 reproduces a GIBBS construction — total pressure varies monotonically
    across the mixed phase;
  - both come out of ONE residual, assembled from the charge regimes, with no
    per-mode code anywhere;
  - at every mixed point the baryon potentials match, the eta-weighted phase
    pressures balance, and baryon number and charge neutrality hold to solver
    tolerance.

Nucleons and electrons only, so the checks isolate the eta split itself; the
muon sector is exercised in test_muons_and_mesons.py.
"""
import numpy as np
import pytest

from eos.dd2 import Parametrization, SpeciesFlags
from eos.vmit.parameters import get_vmit_default
from eos.mixed import beta_eq_neutrinoless
from eos.mixed.equilibrium.residual import mixed_slots, build_mixed_ctx, mixed_residual
from eos.mixed.solvers.point import solve_mixed, RESIDUAL_TOL
from eos.mixed.solvers.sweep import sweep_mixed


@pytest.fixture(scope="module")
def par():
    return Parametrization.from_dd2_defaults()


@pytest.fixture(scope="module")
def flags():
    return SpeciesFlags(hyperons=False, muons=False)


@pytest.fixture(scope="module")
def grid():
    return np.arange(0.40, 0.85, 0.05)          # brackets the DD2+vMIT window


@pytest.fixture(scope="module")
def gibbs(par, flags, grid):
    return [r for r in sweep_mixed(par, flags, grid, 0.0, beta_eq_neutrinoless())
            if r.in_mixed_phase]


@pytest.fixture(scope="module")
def maxwell(par, flags, grid):
    return [r for r in sweep_mixed(par, flags, grid, 1.0, beta_eq_neutrinoless())
            if r.in_mixed_phase]


# ---------------------------------------------------------------- endpoints
def test_maxwell_plateau(maxwell):
    """eta=1: constant total pressure across the mixed phase (Maxwell)."""
    assert len(maxwell) >= 3
    P = [r.P for r in maxwell]
    chi = [r.chi for r in maxwell]
    assert max(P) - min(P) < 1.0e-3            # flat to solver tolerance
    assert chi == sorted(chi) and chi[-1] - chi[0] > 0.1   # chi genuinely sweeps


def test_gibbs_monotonic(gibbs):
    """eta=0: total pressure varies monotonically across the mixed phase."""
    assert len(gibbs) >= 3
    P = [r.P for r in gibbs]
    assert P == sorted(P)                       # monotone increasing in n_B
    assert max(P) - min(P) > 10.0               # genuinely varying (not a plateau)


def test_maxwell_is_not_gibbs(gibbs, maxwell):
    """The two constructions are physically distinct (P spread differs)."""
    assert (max(r.P for r in gibbs) - min(r.P for r in gibbs)) > 10.0
    assert (max(r.P for r in maxwell) - min(r.P for r in maxwell)) < 1.0e-3


# ------------------------------------------------------- equilibrium holds
@pytest.mark.parametrize("eta", [0.0, 1.0])
def test_equilibrium_conditions(par, flags, eta):
    r = solve_mixed(par, flags, 0.7, eta, beta_eq_neutrinoless())
    assert r.in_mixed_phase
    # baryon-number conservation
    assert ((1 - r.chi) * r.th_H.n_B + r.chi * r.th_Q.n_B
            == pytest.approx(0.7, rel=1e-8))
    # baryon chemical equilibrium mu_B^H = mu_B^Q
    assert r.th_H.mu_B == pytest.approx(r.potentials["mu_B_Q"], abs=1e-6)
    # mechanical equilibrium P_H + eta P_eL_H = P_Q + eta P_eL_Q
    pH = r.th_H.P + eta * r.extras["L_H"].P
    pQ = r.th_Q.P + eta * r.extras["L_Q"].P
    assert pH == pytest.approx(pQ, rel=1e-6)
    # residual gate
    ctx = build_mixed_ctx(beta_eq_neutrinoless(), eta, 0.7, par, flags, get_vmit_default())
    x = [r.potentials[s] for s in mixed_slots(beta_eq_neutrinoless(), eta)]
    assert max(abs(v) for v in mixed_residual(x, ctx)) < RESIDUAL_TOL


def test_charge_neutrality(par, flags):
    """eta=0: global neutrality; eta=1: local neutrality per phase."""
    r0 = solve_mixed(par, flags, 0.7, 0.0, beta_eq_neutrinoless())
    glob = (1 - r0.chi) * r0.th_H.n_C + r0.chi * r0.th_Q.n_C - r0.extras["G"].n
    assert glob == pytest.approx(0.0, abs=1e-8)

    r1 = solve_mixed(par, flags, 0.7, 1.0, beta_eq_neutrinoless())
    assert r1.th_H.n_C - r1.extras["L_H"].n == pytest.approx(0.0, abs=1e-8)
    assert r1.th_Q.n_C - r1.extras["L_Q"].n == pytest.approx(0.0, abs=1e-8)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
