"""
Continuity in eta: the two limiting constructions are the ends of one family.

The same residual that produces Gibbs at eta=0 and Maxwell at eta=1 must give
smooth, monotone behaviour in between — with no new code, since eta enters as a
weight rather than as a branch. Checks:

  - at a fixed density inside the mixed phase, P(eta) and chi(eta) are
    continuous and monotone (pressure rises Gibbs -> Maxwell, chi falls);
  - the spread of pressure across the mixed phase collapses monotonically with
    eta, reaching the flat Maxwell plateau exactly at eta=1;
  - the density width of the mixed window does not grow with eta — it shrinks,
    with Maxwell the narrowest.

Nucleons and electrons only, T=0, beta equilibrium.
"""
import numpy as np
import pytest

from eos.dd2 import Parametrization, SpeciesFlags
from eos.mixed import beta_eq_neutrinoless
from eos.mixed.solvers.point import solve_mixed
from eos.mixed.solvers.sweep import sweep_mixed


@pytest.fixture(scope="module")
def par():
    return Parametrization.from_dd2_defaults()


@pytest.fixture(scope="module")
def flags():
    return SpeciesFlags(hyperons=False, muons=False)


def test_continuity_and_monotonicity_in_eta(par, flags):
    """P(eta) up, chi(eta) down, both smooth, at a fixed mixed-phase density."""
    n_B = 0.65
    etas = np.linspace(0.0, 1.0, 6)
    P, chi = [], []
    for eta in etas:
        r = solve_mixed(par, flags, n_B, eta, beta_eq_neutrinoless())
        assert r.in_mixed_phase
        P.append(r.P)
        chi.append(r.chi)
    assert P == sorted(P)                         # P monotone increasing in eta
    assert chi == sorted(chi, reverse=True)       # chi monotone decreasing
    # continuity: no jumps between neighbouring eta (bounded finite differences)
    dP = np.diff(P)
    assert dP.max() < 3.0 * dP.mean()             # smooth, no discontinuous step


def test_pressure_spread_collapses_with_eta(par, flags):
    """Mixed-phase P spread: Gibbs (large) -> Maxwell (flat plateau)."""
    grid = np.arange(0.40, 0.90, 0.05)

    def spread(eta):
        mixed = [r for r in sweep_mixed(par, flags, grid, eta, beta_eq_neutrinoless())
                 if r.in_mixed_phase]
        assert len(mixed) >= 3
        Ps = [r.P for r in mixed]
        return max(Ps) - min(Ps)

    s0, s5, s1 = spread(0.0), spread(0.5), spread(1.0)
    assert s0 > s5 > s1                            # monotone collapse
    assert s1 < 1.0e-3                             # Maxwell plateau


def test_mixed_window_does_not_grow_with_eta(par, flags):
    """The Maxwell (eta=1) density window is no wider than the Gibbs one."""
    grid = np.arange(0.40, 0.90, 0.05)

    def width(eta):
        nBs = [r.n_B for r in sweep_mixed(par, flags, grid, eta, beta_eq_neutrinoless())
               if r.in_mixed_phase]
        return max(nBs) - min(nBs)

    assert width(1.0) <= width(0.0) + 1e-9


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
