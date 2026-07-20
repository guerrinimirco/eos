"""
M10 (Jacobian) gate: thermodynamic coefficients wired onto octet_jacobian
(report §3.5, §3.7 check 5). The analytic-from-J values must agree with the
independent finite-difference oracle (coefficients.py).

- equilibrium c_s^2 from Jacobian tangent steps vs FD-of-resolves;
- susceptibilities chi_ab from the field-response block: symmetric, positive
  baryon susceptibility, and matching a direct grand-canonical FD;
- C_V, C_P per baryon: C_V matches a per-baryon FD (T>0, JEL floor ~1%),
  C_P >= C_V.
"""
import numpy as np
import pytest
from scipy.optimize import root

from eos.dd2 import Parametrization, SpeciesFlags
from eos.dd2 import coefficients as fd
from eos.dd2 import coefficients_jac as jc
from eos.dd2.solver import solve_beta_eq_octet
from eos.dd2.physics.octet import build_octet_ctx, octet_residual, assemble_octet


@pytest.fixture(scope="module")
def par():
    return Parametrization.from_dd2y_defaults()


@pytest.fixture(scope="module")
def flags():
    return SpeciesFlags(hyperons=True, phi_field=True)


@pytest.mark.parametrize("n_B", [0.2, 0.4, 0.8])
def test_sound_speed_eq_from_jacobian(par, flags, n_B):
    cj = jc.sound_speed_eq(par, n_B, flags)
    cf = fd.sound_speed_eq(par, n_B, flags)
    assert cj == pytest.approx(cf, rel=1e-3)
    assert 0.0 < cj < 1.0


def test_susceptibility_symmetric_positive(par, flags):
    for n_B in (0.16, 0.4, 0.8):
        chi = jc.susceptibilities(par, n_B, flags)
        assert np.max(np.abs(chi - chi.T)) / np.max(np.abs(chi)) < 1e-8  # symmetric
        assert chi[0, 0] > 0.0                    # baryon susceptibility > 0


def _grand_canonical_charges(par, n_B, flags, muB, muQ, muS, T=0.0):
    """Solve the field gap equations at fixed (muB,muQ,muS) (couplings frozen
    at n_B) and return (n_tot, charge, strange) in natural units — the direct
    grand-canonical densities for an independent susceptibility FD."""
    ctx = build_octet_ctx(par, n_B, flags, T=T, charge_mode="fixed",
                          strange_mode="fixed")
    has_phi = flags.phi_field and flags.hyperons
    n_f = 3 + int(has_phi)
    base = solve_beta_eq_octet(par, n_B, flags, T=T, include_photons=False)
    F0 = [base.sigma, base.omega0, base.rho0] + ([base.phi0] if has_phi else [])

    def field_res(F):
        x = list(F) + [muB, muQ, muS]
        return octet_residual(np.array(x), ctx)[:n_f]

    sol = root(field_res, F0, method="hybr", tol=1e-13)
    st = assemble_octet(np.array(list(sol.x) + [muB, muQ, muS]), ctx)
    return (st["n_tot"], st["Y_C"] * ctx.nB_nat, st["Y_S"] * ctx.nB_nat)


def test_susceptibility_vs_direct_fd(par, flags):
    n_B = 0.5
    base = solve_beta_eq_octet(par, n_B, flags, include_photons=False)
    muB, muQ, muS = base.mu_n - base.Sigma_R, base.mu_p - base.mu_n, 0.0
    chi = jc.susceptibilities(par, n_B, flags)
    h = 1e-3
    # FD columns: perturb muB, muQ, muS; rows are (n_tot, charge, strange)
    for b, base_mu in enumerate((muB, muQ, muS)):
        mus = [muB, muQ, muS]
        mus[b] = base_mu + h
        np_ = _grand_canonical_charges(par, n_B, flags, *mus)
        mus[b] = base_mu - h
        nm_ = _grand_canonical_charges(par, n_B, flags, *mus)
        col_fd = [(np_[a] - nm_[a]) / (2 * h) for a in range(3)]
        for a in range(3):
            assert chi[a, b] == pytest.approx(col_fd[a], rel=2e-3, abs=1e-2)


@pytest.mark.parametrize("n_B", [0.16, 0.4])
def test_heat_capacity_from_jacobian(par, flags, n_B):
    T = 20.0
    cvj = jc.heat_capacity_V(par, n_B, flags, T=T)
    cpj = jc.heat_capacity_P(par, n_B, flags, T=T)
    # per-baryon FD reference for C_V
    lo = solve_beta_eq_octet(par, n_B, flags, T=T - 0.05, include_photons=False)
    hi = solve_beta_eq_octet(par, n_B, flags, T=T + 0.05, include_photons=False)
    cv_fd = T * (hi.s / hi.n_B - lo.s / lo.n_B) / 0.1
    assert cvj == pytest.approx(cv_fd, rel=2e-2)      # T>0 JEL floor
    assert cpj >= cvj                                  # C_P >= C_V always
