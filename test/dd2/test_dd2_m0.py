"""
M0 gate: DD2 scaffolding.

Ingest identities on the published table (f_i(1)=1, d_i=1/sqrt(3 c_i)),
constructor validation, particle-registry t3, and the JEL kinetic wrapper.
"""
import math

import numpy as np
import pytest

from eos.dd2 import Parametrization
from eos.dd2.couplings import rational_f, derived_a, derived_d
from eos.dd2.parametrization import INGEST_TOL
from eos.dd2.physics.thermo import (
    kF_from_n, kinetic_thermo, number_density_t0,
)
from eos.general.particles import Neutron, Proton, SigmaM, Xi0, Lambda
from eos.general.physics_constants import hc3


def _coeffs(par, meson):
    return tuple(getattr(par, f"{k}_{meson}") for k in "abcd")


def test_ingest_f_at_saturation_is_one():
    par = Parametrization.from_dd2_defaults()
    for meson in ("sigma", "omega"):
        a, b, c, d = _coeffs(par, meson)
        assert abs(rational_f(1.0, a, b, c, d) - 1.0) < INGEST_TOL


def test_ingest_d_constraint():
    par = Parametrization.from_dd2_defaults()
    for meson in ("sigma", "omega"):
        _, _, c, d = _coeffs(par, meson)
        assert abs(d - 1.0 / math.sqrt(3.0 * c)) < INGEST_TOL


def test_from_microscopic_derives_dependent_coeffs():
    ref = Parametrization.from_dd2_defaults()
    par = Parametrization.from_microscopic(
        n_sat=ref.n_sat,
        gamma_sigma=ref.gamma_sigma, b_sigma=ref.b_sigma, c_sigma=ref.c_sigma,
        gamma_omega=ref.gamma_omega, b_omega=ref.b_omega, c_omega=ref.c_omega,
        gamma_rho=ref.gamma_rho, a_rho=ref.a_rho,
    )
    # Derived a_i, d_i must reproduce the published (rounded) values.
    for meson in ("sigma", "omega"):
        for k in ("a", "d"):
            assert getattr(par, f"{k}_{meson}") == pytest.approx(
                getattr(ref, f"{k}_{meson}"), abs=5e-6)


def test_inconsistent_coefficients_raise():
    ref = Parametrization.from_dd2_defaults()
    with pytest.raises(ValueError, match="d_omega"):
        Parametrization.from_microscopic(
            n_sat=ref.n_sat,
            gamma_sigma=ref.gamma_sigma, b_sigma=ref.b_sigma, c_sigma=ref.c_sigma,
            gamma_omega=ref.gamma_omega, b_omega=ref.b_omega, c_omega=ref.c_omega,
            gamma_rho=ref.gamma_rho, a_rho=ref.a_rho,
            d_omega=0.5,  # violates f''(0)=0
        )
    with pytest.raises(ValueError, match="must be > 0"):
        Parametrization.from_microscopic(
            n_sat=ref.n_sat,
            gamma_sigma=ref.gamma_sigma, b_sigma=ref.b_sigma, c_sigma=-1.0,
            gamma_omega=ref.gamma_omega, b_omega=ref.b_omega, c_omega=ref.c_omega,
            gamma_rho=ref.gamma_rho, a_rho=ref.a_rho,
        )


def test_particle_registry_t3():
    # tau_3 = ±1 convention — the load-bearing isospin choice.
    assert Proton.t3 == +1.0
    assert Neutron.t3 == -1.0
    assert Lambda.t3 == 0.0
    assert SigmaM.t3 == -1.0
    assert Xi0.t3 == +1.0


def test_kinetic_thermo_t0_free_gas_identity():
    # At T=0: e + P = nu * n exactly, and n reproduces g kF^3 / 6 pi^2.
    for nu, m, g in [(600.0, 528.0, 4.0), (950.0, 939.0, 2.0), (300.0, 0.511, 2.0)]:
        n, P, e, s, ns = kinetic_thermo(nu, m, g, T=0.0)
        assert s == 0.0
        kF = math.sqrt(nu ** 2 - m ** 2)
        assert n == pytest.approx(number_density_t0(kF, g), rel=1e-12)
        assert e + P == pytest.approx(nu * n, rel=1e-12)
    # Below threshold: vacuum.
    assert kinetic_thermo(500.0, 939.0, 2.0, T=0.0) == (0.0, 0.0, 0.0, 0.0, 0.0)


def test_kinetic_thermo_jel_branch_units():
    # T>0 branch (JEL, natural units) vs Gauss-Laguerre reference.
    from eos.general.fermi_integrals import solve_fermi_gl
    nu, m, g, T = 900.0, 939.0, 2.0, 10.0
    got = kinetic_thermo(nu, m, g, T=T)
    ref = tuple(v * hc3 for v in solve_fermi_gl(nu, T, m, g))
    for a, b in zip(got, ref):
        assert a == pytest.approx(b, rel=2e-3)
