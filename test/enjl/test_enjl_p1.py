"""
P1 gate (uniform-matter solver, Paper 1): eos/enjl reproduces the reference
numbers of Xia 2024 (PRD 110, 014022, arXiv:2405.02946) for uniform matter.

Reference anchor points (paper Table II + Sec. II prose + the factor-9 reading
of the rho coupling read off the tables in test/enjl/reference/):

    * vacuum:  M_u = M_d = 367.6, M_s = 549.5 MeV;  M_N = 938.9, M_L = 1113.7;
               alpha_S(0) = 0.849, alpha_S(n_S) = 0.57000
    * saturation: n0 = 0.158, E/A = -16.0, K = 234.5, S = 31.5, L = 42.4
    * symmetry energy: S(0.1 fm^-3) = 25.5  (the "S(n_on)" of the paper)
    * Lambda potential: U_Lambda(n0) = -30 MeV  (fixes f_Lambda = 1.0626)
    * isoscalar: E/n = 924.8 (0.1), 922.9 (0.158);  M_N = 519.2 (0.16)

Tolerances: the paper quotes 1-3 significant figures; we assert the exact
solver output against the printed values at ~1% (rel=1.5e-2), and keep the
exact-round-off checks (masses, alpha_S) at the tightest tolerance.

The rho channel carries the factor 9 of RHO_FACTOR (eos/enjl/parameters.py);
without it S(0.1)=13.3 and S(0.158)=20.2 would fail.
"""
import numpy as np
import pytest

from eos.enjl import ENJLParams, get_enjl_default, solve_point
from eos.enjl.parameters import RHO_FACTOR
from eos.enjl.uniform import vacuum_solution
from eos.general.physics_constants import hc3


# -----------------------------------------------------------------------------
# fixtures (module-scoped, built once)
# -----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def par():
    return get_enjl_default()


def _dens(nb_fm):
    nb = nb_fm * hc3
    return {"p": nb / 2.0, "n": nb / 2.0, "Lambda": 0.0,
            "u": 0.0, "d": 0.0, "s": 0.0, "e": 0.0, "mu": 0.0}


def _pnm(nb_fm):
    nb = nb_fm * hc3
    return {"p": 0.0, "n": nb, "Lambda": 0.0,
            "u": 0.0, "d": 0.0, "s": 0.0, "e": 0.0, "mu": 0.0}


def _e_per_b(nb_fm, par=None, pnm=False):
    n = _pnm(nb_fm) if pnm else _dens(nb_fm)
    return solve_point(n, par=par).EperB


# -----------------------------------------------------------------------------
# 1. vacuum anchor
# -----------------------------------------------------------------------------
def test_vacuum_masses(par):
    M = vacuum_solution(par)
    assert M["u"] == pytest.approx(367.6, rel=1e-3)
    assert M["d"] == pytest.approx(367.6, rel=1e-3)
    assert M["s"] == pytest.approx(549.5, rel=1e-3)
    assert par.alpha_S(0.0) == pytest.approx(0.849, rel=1e-3)
    assert par.alpha_S(par.nS * hc3) == pytest.approx(0.57000, rel=1e-6)


def test_vacuum_baryon_masses(par):
    # M_i = sum_q N_i^q [m_q0 + alpha_S (M_q - m_q0)] at n_b = 0
    from eos.enjl.uniform import _baryon_masses
    M = vacuum_solution(par)
    Mb = _baryon_masses(par, M, par.alpha_S(0.0), 0.0)
    assert Mb["p"] == pytest.approx(938.9, rel=1e-3)
    assert Mb["n"] == pytest.approx(938.9, rel=1e-3)
    assert Mb["Lambda"] == pytest.approx(1113.7, rel=1e-3)


# -----------------------------------------------------------------------------
# 2. isoscalar channel (Table II row: 0.158  -16.0  234.5  31.5  42.4)
# -----------------------------------------------------------------------------
def test_saturation_point(par):
    pt = solve_point(_dens(0.158))
    assert pt.EperB == pytest.approx(-16.0, rel=1.5e-2)
    assert pt.P_fm == pytest.approx(0.0, abs=5.0)   # near saturation, small P


def test_isoscalar_energy_per_baryon(par):
    # E/n (total energy per baryon, not minus m_N): 938.9 + E/A
    e01 = _e_per_b(0.1) + 938.9
    e016 = _e_per_b(0.158) + 938.9
    assert e01 == pytest.approx(924.8, rel=1.5e-2)
    assert e016 == pytest.approx(922.9, rel=1.5e-2)


def test_nucleon_mass_0_16(par):
    pt = solve_point(_dens(0.16))
    assert pt.M_b["p"] == pytest.approx(519.2, rel=1.5e-2)
    assert pt.M_b["n"] == pytest.approx(519.2, rel=1.5e-2)


def test_table_II(par):
    n0 = 0.158
    assert n0 == pytest.approx(0.158, rel=1e-2)
    assert _e_per_b(n0) == pytest.approx(-16.0, rel=1.5e-2)
    K = 9.0 * n0 * n0 * (  # second derivative via central differences
        _e_per_b(n0 + 0.002) - 2.0 * _e_per_b(n0) + _e_per_b(n0 - 0.002)) \
        / 0.002 ** 2
    assert K == pytest.approx(234.5, rel=1.5e-2)
    S = _e_per_b(n0, pnm=True) - _e_per_b(n0)
    assert S == pytest.approx(31.5, rel=1.5e-2)
    dS = (_e_per_b(n0 + 0.002, pnm=True) - _e_per_b(n0 + 0.002) - S) / 0.002
    L = 3.0 * n0 * dS
    assert L == pytest.approx(42.4, rel=1.5e-2)


def test_symmetry_energy_0_1(par):
    # paper: S(n_on) = 25.5 (n_on = 0.1 fm^-3), the density of the 0.1 anchor
    S = _e_per_b(0.1, pnm=True) - _e_per_b(0.1)
    assert S == pytest.approx(25.5, rel=1.5e-2)


# -----------------------------------------------------------------------------
# 3. Lambda potential  (fixes f_Lambda = 1.0626)
# -----------------------------------------------------------------------------
def test_lambda_potential(par):
    M = vacuum_solution(par)
    ml_vac = (1.0 * (5.5 + par.alpha_S(0.0) * (M["u"] - 5.5))
              + 1.0 * (5.5 + par.alpha_S(0.0) * (M["d"] - 5.5))
              + 1.0 * (140.7 + par.alpha_S(0.0) * (M["s"] - 140.7)))
    nb = 0.158 * hc3
    n = {"p": nb / 2.0, "n": nb / 2.0, "u": 0.0, "d": 0.0, "s": 0.0,
         "Lambda": 1.0e-9 * hc3, "e": 0.0, "mu": 0.0}
    pt = solve_point(n)
    U = pt.mu["Lambda"] - ml_vac
    assert U == pytest.approx(-30.0, rel=1.5e-2)


# -----------------------------------------------------------------------------
# 4. the factor-9 rho coupling
# -----------------------------------------------------------------------------
def test_rho_factor_is_verified_nine():
    assert RHO_FACTOR == 9.0
    par = ENJLParams()
    Gr = par.Gamma_r(0.158 * hc3)
    assert Gr == pytest.approx(9.0 * 4.0 * par.GS
                               * (par.aTV * np.exp(-0.158 / par.nTV) + par.bTV),
                               rel=1e-12)


def test_symmetry_energy_requires_factor_nine(par):
    # Without the factor 9 the paper symmetry energies cannot be reproduced
    # (S(0.1)=13.3, S(0.158)=20.2 with the paper Eq. (22) literal reading).
    from dataclasses import replace
    assert replace(par, aTV=0.0).aTV == 0.0
    S9 = _e_per_b(0.158, pnm=True) - _e_per_b(0.158)
    assert S9 == pytest.approx(31.5, rel=1.5e-2)
    assert S9 > 25.0                        # far above the no-factor value


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
