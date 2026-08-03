"""
P2 gate: continuous eta (docs/phase2/SPECIFICATION_AND_PLAN.md §4 milestone P2).

The same regime-driven residual that gives the eta endpoints must give smooth,
monotone behaviour in between — with NO new code (spec §1.5). Checks:

  - at a fixed density in the mixed phase, P(eta) and chi(eta) are continuous
    and monotone (P rises Gibbs->Maxwell, chi falls);
  - the mixed-phase pressure spread collapses monotonically with eta, reaching
    the flat Maxwell plateau at eta=1;
  - the mixed-phase density window does not grow with eta (it shrinks: the
    Maxwell window is the narrowest).

Nucleons + electrons only (muons off), T=0, Mode A — as P1.
"""
import numpy as np
import pytest

from eos.dd2 import Parametrization, SpeciesFlags
from eos.mixed import mode_A
from eos.mixed.solver import solve_mixed
from eos.mixed.continuation import sweep_mixed


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
        r = solve_mixed(par, flags, n_B, eta, mode_A())
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
        mixed = [r for r in sweep_mixed(par, flags, grid, eta, mode_A())
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
        nBs = [r.n_B for r in sweep_mixed(par, flags, grid, eta, mode_A())
               if r.in_mixed_phase]
        return max(nBs) - min(nBs)

    assert width(1.0) <= width(0.0) + 1e-9


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
