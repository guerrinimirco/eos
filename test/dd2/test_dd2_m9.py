"""
M9 gate: the eos_fast backend = hand-coded analytic Jacobian.

The report's JAX autodiff path is deferred (the JEL/Bose core and T=0 threshold
kink do not trace cleanly, report D3); the authorized fallback is the exact
analytic Jacobian supplied to the same MINPACK solver. Gate:

- the analytic Jacobian matches a finite-difference Jacobian of the octet
  residual (exact at T=0, JEL-floor-limited at T>0), across every mode;
- backend parity: the analytic-Jacobian (eos_fast) solve reproduces the
  numeric-Jacobian (eos_ref) solve at every golden/hyperonic point and across
  a table — same root, different path.
"""
import numpy as np
import pytest

from eos.dd2 import (
    Parametrization, SpeciesFlags, solve_octet, octet_warm_start,
    solve_beta_eq_octet, solve_yl_octet, solve_fixed_yc_octet,
    sweep_beta_eq_octet,
)
from eos.dd2.physics.octet import build_octet_ctx, octet_residual
from eos.dd2.physics.jacobian import octet_jacobian, kinetic_derivs
from eos.dd2.physics.kernel_numba import (
    residual_t0_jit, jacobian_t0_jit, build_numba_arrays,
)
from eos.dd2.physics.thermo import kinetic_thermo


@pytest.fixture(scope="module")
def par():
    return Parametrization.from_dd2y_defaults()


def _fd_jac(x, ctx, h=1e-6):
    x = np.asarray(x, float)
    n = len(x)
    J = np.zeros((n, n))
    for j in range(n):
        d = h * max(abs(x[j]), 1.0)
        xp_, xm_ = x.copy(), x.copy()
        xp_[j] += d
        xm_[j] -= d
        J[:, j] = (np.array(octet_residual(xp_, ctx))
                   - np.array(octet_residual(xm_, ctx))) / (2.0 * d)
    return J


def test_kinetic_derivs_T0_closed_form():
    # closed-form d{n,ns}/d{nu,m*} vs central difference of kinetic_thermo
    nu, m, g = 900.0, 480.0, 2.0
    a = kinetic_derivs(nu, m, g, 0.0)
    h = 1e-4
    fd = (
        (kinetic_thermo(nu + h, m, g)[0] - kinetic_thermo(nu - h, m, g)[0]) / (2 * h),
        (kinetic_thermo(nu, m + h, g)[0] - kinetic_thermo(nu, m - h, g)[0]) / (2 * h),
        (kinetic_thermo(nu + h, m, g)[4] - kinetic_thermo(nu - h, m, g)[4]) / (2 * h),
        (kinetic_thermo(nu, m + h, g)[4] - kinetic_thermo(nu, m - h, g)[4]) / (2 * h),
    )
    for an, fdv in zip(a, fd):
        assert an == pytest.approx(fdv, rel=1e-6)


@pytest.mark.parametrize("T", [0.0, 15.0])
def test_analytic_jac_matches_fd(par, T):
    fY = SpeciesFlags(hyperons=True, phi_field=True)
    fNu = SpeciesFlags(hyperons=True, phi_field=True, neutrinos=True)
    tol = 1e-7 if T == 0.0 else 3e-4       # T>0 kinetic derivs are FD (JEL floor)

    cases = [
        (build_octet_ctx(par, 0.5, fY, T=T),
         octet_warm_start(solve_beta_eq_octet(par, 0.5, fY, T=T,
                                              include_photons=False),
                          True, False, False)),
        (build_octet_ctx(par, 0.6, fY, T=T, charge_mode="fixed", Y_C=0.1,
                         strange_mode="fixed", Y_S=0.05),
         octet_warm_start(solve_fixed_yc_octet(par, 0.6, 0.1, fY, T=T, Y_S=0.05),
                          True, True, False)),
        (build_octet_ctx(par, 0.5, fNu, T=T, lepton_mode="trapped", Y_L=0.3),
         octet_warm_start(solve_yl_octet(par, 0.5, 0.3, fNu, T=T),
                          True, False, True)),
    ]
    for ctx, x in cases:
        Ja = octet_jacobian(x, ctx)
        Jf = _fd_jac(x, ctx)
        scale = max(np.max(np.abs(Jf)), 1e-30)
        assert np.max(np.abs(Ja - Jf)) / scale < tol


def test_numba_kernel_parity(par):
    # the jitted T=0 residual/Jacobian are machine-identical to the NumPy
    # kernel across every mode (this is what solve_octet routes T=0 through).
    fY = SpeciesFlags(hyperons=True, phi_field=True)
    fNu = SpeciesFlags(hyperons=True, phi_field=True, neutrinos=True)
    cases = [
        (build_octet_ctx(par, 0.5, fY),
         octet_warm_start(solve_beta_eq_octet(par, 0.5, fY,
                                              include_photons=False),
                          True, False, False)),
        (build_octet_ctx(par, 0.6, fY, charge_mode="fixed", Y_C=0.1,
                         strange_mode="fixed", Y_S=0.05),
         octet_warm_start(solve_fixed_yc_octet(par, 0.6, 0.1, fY, Y_S=0.05),
                          True, True, False)),
        (build_octet_ctx(par, 0.5, fNu, lepton_mode="trapped", Y_L=0.3),
         octet_warm_start(solve_yl_octet(par, 0.5, 0.3, fNu),
                          True, False, True)),
    ]
    for ctx, x in cases:
        x = np.asarray(x, float)
        spec, prm, flg = build_numba_arrays(ctx)
        assert np.max(np.abs(np.array(octet_residual(x, ctx))
                             - residual_t0_jit(x, spec, prm, flg))) < 1e-12
        assert np.max(np.abs(octet_jacobian(x, ctx)
                             - jacobian_t0_jit(x, spec, prm, flg))) < 1e-12


@pytest.mark.parametrize("T", [0.0, 20.0])
def test_backend_parity(par, T):
    # eos_fast (analytic J) must reproduce eos_ref (numeric J): same root.
    fY = SpeciesFlags(hyperons=True, phi_field=True)
    fN = SpeciesFlags(hyperons=False, phi_field=False)
    configs = [
        (fN, dict(charge_mode="neutral")),
        (fY, dict(charge_mode="neutral")),
        (fY, dict(charge_mode="fixed", Y_C=0.15)),
    ]
    for flags, kw in configs:
        for n_B in (0.16, 0.3, 0.5, 0.8):
            a = solve_octet(par, n_B, flags, T=T, include_photons=False,
                            analytic_jac=True, **kw)
            r = solve_octet(par, n_B, flags, T=T, include_photons=False,
                            analytic_jac=False, **kw)
            for f in ("eps", "P", "s", "mu_n", "mu_p", "sigma", "omega0",
                      "rho0", "phi0"):
                va, vr = getattr(a, f), getattr(r, f)
                if abs(vr) > 1e-9:
                    assert va == pytest.approx(vr, rel=1e-6)


def test_fast_backend_sweep_matches(par):
    # the eos_fast backend works through a warm-started hyperon sweep.
    flags = SpeciesFlags(hyperons=True, phi_field=True)
    grid = np.linspace(0.1, 1.0, 25)
    ref = sweep_beta_eq_octet(par, grid, flags, include_photons=False)
    fast = sweep_beta_eq_octet(par, grid, flags, include_photons=False,
                               analytic_jac=True)
    assert len(ref) == len(fast)
    for a, b in zip(fast, ref):
        assert a.P == pytest.approx(b.P, rel=1e-6)
        assert a.eps == pytest.approx(b.eps, rel=1e-6)
