"""
Does a truncated sweep truncate for the right reason?

`sweep_octet(stop_at_boundary=True)` exists for one physical situation: a Delta
model drives the Dirac effective mass to zero at high density, past which there
is no mean-field solution at all, so the sweep returns the branch that exists
instead of raising. That is the scalar-collapse feasibility boundary.

The failure this file pins is the other reason a point can fail — the warm-start
continuation losing the basin at one density — arriving through the same door.
Ending the sweep there discards every density above it, and nothing in the
returned list says so: a branch cut short at 1.01 fm^-3 by one missed solve is
the same object as one that genuinely ends at 1.55. It showed up as a single
temperature in a T-sweep returning 168 of 300 points where both its neighbours
returned 255, which is not a shape any physical boundary has.

`max_skip` is what separates them. Past the boundary every remaining density
fails, so a run of misses still ends the sweep; an isolated one is skipped and
the branch above it survives.
"""
import numpy as np
import pytest

from eos.dd2 import Parametrization, SpeciesFlags
from eos.dd2.solver import sweep_beta_eq_octet
import eos.dd2.solver as solver_mod

FLAGS = SpeciesFlags(hyperons=False, deltas=False, muons=True)
GRID = np.linspace(0.10, 0.40, 12)


@pytest.fixture(scope="module")
def par():
    return Parametrization.from_dd2_defaults()


def _fail_at(monkeypatch, densities):
    """Make `solve_octet` raise at `densities` and behave normally elsewhere.

    Injected rather than hunted for in a real parametrization: the miss is a
    property of the continuation's path, so it moves with the grid and cannot
    be pinned to a density in a test that is supposed to stay true.
    """
    real = solver_mod.solve_octet

    def patched(par_, n_B, *a, **kw):
        if any(abs(n_B - d) < 1e-12 for d in densities):
            raise RuntimeError(f"injected failure at n_B={n_B}")
        return real(par_, n_B, *a, **kw)

    monkeypatch.setattr(solver_mod, "solve_octet", patched)


def test_clean_sweep_returns_every_point(par):
    """The control: nothing fails, so nothing is skipped or cut."""
    pts = sweep_beta_eq_octet(par, GRID, FLAGS, include_photons=False,
                              stop_at_boundary=True)
    assert len(pts) == len(GRID)
    assert pts[-1].n_B == pytest.approx(GRID[-1])


def test_one_missed_density_is_a_hole_not_the_end(par, monkeypatch):
    """The regression. One unsolvable density must cost one point, not the tail.

    Before `max_skip` this returned 4 of 12 points and looked exactly like a
    branch that ends at 0.18 fm^-3.
    """
    _fail_at(monkeypatch, [GRID[4]])
    pts = sweep_beta_eq_octet(par, GRID, FLAGS, include_photons=False,
                              stop_at_boundary=True, max_bisect=1)
    assert len(pts) == len(GRID) - 1, "the tail above the hole was lost"
    assert pts[-1].n_B == pytest.approx(GRID[-1]), "sweep stopped early"
    assert not any(abs(p.n_B - GRID[4]) < 1e-12 for p in pts)


def test_a_run_of_failures_still_ends_the_sweep(par, monkeypatch):
    """The boundary must still stop it. Past scalar collapse nothing solves, so
    skipping has to give up rather than grind through the rest of the grid."""
    _fail_at(monkeypatch, list(GRID[6:]))
    pts = sweep_beta_eq_octet(par, GRID, FLAGS, include_photons=False,
                              stop_at_boundary=True, max_bisect=1)
    assert len(pts) == 6
    assert pts[-1].n_B == pytest.approx(GRID[5])


def test_max_skip_zero_restores_stopping_at_the_first_miss(par, monkeypatch):
    """The old behaviour stays reachable, so a caller that wants a strictly
    contiguous branch can still demand one."""
    _fail_at(monkeypatch, [GRID[4]])
    pts = sweep_beta_eq_octet(par, GRID, FLAGS, include_photons=False,
                              stop_at_boundary=True, max_bisect=1, max_skip=0)
    assert len(pts) == 4


def test_the_last_point_says_which_reason_it_was(par, monkeypatch):
    """The diagnostic a caller needs, and the reason `max_skip` alone is not
    enough: a sweep that ends at the scalar boundary has m_eff ~ 0 there, one
    that gave up ends with a healthy effective mass. Same list type, and only
    this tells them apart."""
    _fail_at(monkeypatch, list(GRID[6:]))
    pts = sweep_beta_eq_octet(par, GRID, FLAGS, include_photons=False,
                              stop_at_boundary=True, max_bisect=1)
    # At the real boundary this ratio is ~4e-4; here the branch is healthy and
    # nowhere near collapse, which is the whole distinction.
    assert pts[-1].m_eff / par.m_n > 0.1, (
        "an injected miss must not look like scalar collapse")
