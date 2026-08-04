"""
The jitted and NumPy evaluations of the mixed-phase hadronic residual must
agree.

`_hadronic_residual` runs its per-species loop in the Numba kernel
`meson_sources_t0` at T = 0 and in the NumPy `_baryon_kinetics` path otherwise.
Both are the same closed form, so at T = 0 they must agree to machine
precision — the NumPy path is the oracle when they do not.
"""
import numpy as np
import pytest

import eos.mixed.solvers.phases as phases
from eos.dd2 import Parametrization, SpeciesFlags
from eos.mixed import (
    beta_eq_neutrinoless, locate_window, sweep_mixed, solve_mixed,
)
from eos.vmit.parameters import get_vmit_default


@pytest.fixture(scope="module")
def setup():
    par = Parametrization.from_dd2_defaults()
    flags = SpeciesFlags(hyperons=False, phi_field=False, muons=False)
    return par, flags, get_vmit_default()


@pytest.fixture
def numpy_path(monkeypatch):
    """Force `_hadronic_residual` down its NumPy branch."""
    monkeypatch.setattr(phases, "_NUMBA_OK", False)


# Densities a COLD start reaches. Deeper into the window the mixed solve needs
# the warm start a sweep provides — that is a property of the solve, not of the
# residual backend, and `test_sweep_agrees` covers the full window.
@pytest.mark.parametrize("n_B", [0.5, 0.6, 0.7])
def test_single_point_agrees(setup, monkeypatch, n_B):
    par, flags, vmit = setup
    spec = beta_eq_neutrinoless()
    jit = solve_mixed(par, flags, n_B, 0.0, spec, vmit_params=vmit, T=0.0)
    monkeypatch.setattr(phases, "_NUMBA_OK", False)
    ref = solve_mixed(par, flags, n_B, 0.0, spec, vmit_params=vmit, T=0.0)

    assert jit.P == pytest.approx(ref.P, rel=1e-10)
    assert jit.eps == pytest.approx(ref.eps, rel=1e-10)
    assert jit.chi == pytest.approx(ref.chi, abs=1e-10)
    assert jit.mu_B == pytest.approx(ref.mu_B, rel=1e-10)


def test_sweep_agrees(setup, monkeypatch):
    par, flags, vmit = setup
    spec = beta_eq_neutrinoless()
    grid = np.linspace(0.1 * par.n_sat, 12.0 * par.n_sat, 120)
    window = locate_window(par, flags, grid, 0.0, spec, vmit_params=vmit,
                           T=0.0)
    assert window.exists
    inside = grid[(grid >= window.n_onset) & (grid <= window.n_offset)]

    jit = sweep_mixed(par, flags, inside, 0.0, spec, vmit_params=vmit, T=0.0)
    monkeypatch.setattr(phases, "_NUMBA_OK", False)
    ref = sweep_mixed(par, flags, inside, 0.0, spec, vmit_params=vmit, T=0.0)

    assert len(jit) == len(ref)
    for a, b in zip(jit, ref):
        assert a.n_B == pytest.approx(b.n_B)
        assert a.P == pytest.approx(b.P, rel=1e-9)
        assert a.eps == pytest.approx(b.eps, rel=1e-9)
        assert a.chi == pytest.approx(b.chi, abs=1e-9)


def test_finite_T_uses_numpy_path(setup):
    """T > 0 has no jitted branch (the JEL integrals do not trace), so it must
    still solve — this guards the branch condition itself, not the numbers."""
    par, flags, vmit = setup
    r = solve_mixed(par, flags, 0.7, 0.0, beta_eq_neutrinoless(),
                    vmit_params=vmit, T=20.0)
    assert r.converged and r.P > 0.0 and r.s > 0.0
