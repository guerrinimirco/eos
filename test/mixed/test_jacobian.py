"""
The hand-assembled analytic Jacobian of the mixed residual.

Same contract as the DD2 analytic Jacobian: an exact Jacobian handed to the
same root finder, checked against a finite difference, with the numeric-Jacobian
path as the correctness oracle (CLAUDE.md §4).

  - `mixed_jacobian` matches a finite-difference Jacobian of `mixed_residual`
    at the converged root, across EVERY mode, at both eta endpoints and at
    T > 0 — which exercises the whole assembly: the quark Sherman-Morrison
    block, the hadronic implicit-function block, the lepton and neutrino
    blocks, and the chain rule through the per-phase charge potentials;
  - backend parity: solving with the analytic Jacobian reaches the SAME root as
    solving without it, by a different path.
"""
import numpy as np
import pytest

from eos.dd2 import Parametrization, SpeciesFlags
from eos.vmit.parameters import get_vmit_default
from eos.mixed import beta_eq_neutrinoless, fixed_YC, fixed_YC_YS, beta_eq_neutrino_trapped
from eos.mixed.solvers.point import solve_mixed
from eos.mixed.equilibrium.residual import build_mixed_ctx, mixed_residual, mixed_slots
from eos.mixed.equilibrium.jacobian import mixed_jacobian


@pytest.fixture(scope="module")
def par():
    return Parametrization.from_dd2_defaults()


@pytest.fixture(scope="module")
def flags():
    return SpeciesFlags(hyperons=False, muons=False)


@pytest.fixture(scope="module")
def vp():
    return get_vmit_default()


def _fd_jac(x, ctx, h=1e-4):
    x = np.asarray(x, float)
    n = len(x)
    m = len(mixed_residual(x, ctx))
    J = np.zeros((m, n))
    for j in range(n):
        d = max(h, 1e-6 * abs(x[j]))
        xp, xm = x.copy(), x.copy()
        xp[j] += d
        xm[j] -= d
        J[:, j] = (np.array(mixed_residual(xp, ctx))
                   - np.array(mixed_residual(xm, ctx))) / (2.0 * d)
    return J


# label, eta, spec-factory, n_B, T -- one per mode / endpoint / T>0
_CASES = [
    ("A_eta0",   0.0, lambda: beta_eq_neutrinoless(),                       0.60, 0.0),
    ("A_eta1",   1.0, lambda: beta_eq_neutrinoless(),                       0.72, 0.0),
    ("A_Tpos",   0.5, lambda: beta_eq_neutrinoless(),                       0.60, 15.0),
    ("C_eta0.5", 0.5, lambda: fixed_YC(0.10, leptons=True),  0.60, 0.0),
    ("D_eta0",   0.0, lambda: fixed_YC_YS(0.10, 0.20, leptons=True), 0.60, 0.0),
    ("B_Tpos",   0.0, lambda: beta_eq_neutrino_trapped(0.30),                   0.60, 20.0),
]


@pytest.mark.parametrize("label,eta,make_spec,n_B,T", _CASES)
def test_analytic_matches_fd(par, flags, vp, label, eta, make_spec, n_B, T):
    """mixed_jacobian == FD(mixed_residual) at the converged root, every mode."""
    spec = make_spec()
    r = solve_mixed(par, flags, n_B, eta, spec, vmit_params=vp, T=T,
                    check_consistency=False)
    slots = mixed_slots(spec, eta)
    x = np.array([r.potentials[s] for s in slots])
    ctx = build_mixed_ctx(spec, eta, n_B, par, flags, vp, T=T)
    Ja = mixed_jacobian(x, ctx)
    Jn = _fd_jac(x, ctx)
    assert Ja.shape == Jn.shape == (len(x), len(x))
    # T=0 is exact; T>0 is JEL-floor-limited -> a looser but still tight bound.
    tol = 1e-6 if T == 0.0 else 1e-5
    assert np.abs(Ja - Jn).max() < tol, f"{label}: max |dJ| = {np.abs(Ja-Jn).max():.2e}"


@pytest.mark.parametrize("label,eta,make_spec,n_B,T", _CASES)
def test_backend_parity(par, flags, vp, label, eta, make_spec, n_B, T):
    """Analytic-Jacobian solve reaches the same root as the numeric one."""
    spec = make_spec()
    r = solve_mixed(par, flags, n_B, eta, spec, vmit_params=vp, T=T,
                    check_consistency=False)
    slots = mixed_slots(spec, eta)
    x0 = [r.potentials[s] for s in slots]
    ra = solve_mixed(par, flags, n_B, eta, spec, vmit_params=vp, T=T,
                     check_consistency=False, analytic_jac=True, x0=x0)
    assert ra.converged
    assert ra.chi == pytest.approx(r.chi, abs=1e-8)
    assert ra.P == pytest.approx(r.P, rel=1e-8)


def test_analytic_jac_solves_from_default_seed(par, flags, vp):
    """The analytic path is self-sufficient: it converges from the cold seed at
    an interior mixed point (not only warm-started)."""
    r = solve_mixed(par, flags, 0.6, 0.0, beta_eq_neutrinoless(), vmit_params=vp,
                    analytic_jac=True)
    assert r.converged and 0.0 < r.chi < 1.0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
