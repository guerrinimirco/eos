"""
M3 gate: finite temperature (JEL integrals, entropy, free energy).

Gate items testable without external data: the T -> 0 limit reproduces the
M1/M2 (golden-checked) T = 0 states to the JEL accuracy floor, and the
Hugenholtz–Van Hove identity with the Ts term holds at round-off across
(n_B, T). The CompOSE DD2 finite-T slice (< 0.1%) is data-blocked: no DD2
CompOSE table is present locally (only SFHO/SFHOY under Research/Compose).

Tolerances below are ~3x the measured JEL floor at T = 0.01 MeV.
"""
import pytest

from eos.dd2 import (
    Parametrization, solve_beta_eq, solve_composition, solve_snm,
)
from eos.general.physics_constants import hc3
from eos.general.thermodynamics_leptons import photon_thermo


@pytest.fixture(scope="module")
def par():
    return Parametrization.from_dd2_defaults()


def test_t0_limit_snm(par):
    cold = solve_snm(par, 0.16, T=0.0)      # golden-gated in M1
    warm = solve_snm(par, 0.16, T=0.01)
    assert warm.eps == pytest.approx(cold.eps, rel=1e-6)
    assert abs(warm.P - cold.P) < 2e-4
    assert abs(warm.mu_n - cold.mu_n) < 2e-3
    assert warm.m_eff == pytest.approx(cold.m_eff, rel=1e-5)
    assert 0.0 < warm.s < 5e-4


def test_t0_limit_beta_eq(par):
    cold = solve_beta_eq(par, 0.16, T=0.0)  # golden-gated in M2
    warm = solve_beta_eq(par, 0.16, T=0.01)
    assert warm.Y_p == pytest.approx(cold.Y_p, rel=1e-4)
    assert warm.eps == pytest.approx(cold.eps, rel=1e-6)
    assert abs(warm.P - cold.P) < 2e-4
    assert abs(warm.mu_e - cold.mu_e) < 2e-3


@pytest.mark.parametrize("T", [1.0, 10.0, 30.0])
@pytest.mark.parametrize("n_B", [0.08, 0.16, 0.32])
def test_hvh_with_Ts(par, n_B, T):
    b = solve_beta_eq(par, n_B, T=T)
    assert abs(b.hvh_rel) < 1e-12
    c = solve_composition(par, 0.7 * n_B, 0.3 * n_B, T=T)
    assert abs(c.hvh_rel) < 1e-12
    # entropy and free energy are consistent by construction; spot-check sign
    assert b.s > 0.0
    assert b.free_energy_density == pytest.approx(b.eps - T * b.s, rel=1e-12)


def test_entropy_monotone_in_T(par):
    s_vals = [solve_beta_eq(par, 0.16, T=T).s for T in [0.5, 1.0, 5.0, 20.0]]
    assert all(a < b for a, b in zip(s_vals, s_vals[1:]))


def test_charge_closures_at_finite_T(par):
    p = solve_beta_eq(par, 0.16, T=15.0)
    assert p.n_p == pytest.approx(p.n_e + p.n_mu, rel=1e-8)
    assert p.mu_n - p.mu_p == pytest.approx(p.mu_e, abs=1e-6)


def test_photon_flag(par):
    T = 20.0
    on = solve_beta_eq(par, 0.16, T=T, include_photons=True)
    off = solve_beta_eq(par, 0.16, T=T, include_photons=False)
    ph = photon_thermo(T)
    # photons are chargeless spectators: identical chemistry, additive thermo
    assert on.Y_p == pytest.approx(off.Y_p, rel=1e-12)
    assert on.eps - off.eps == pytest.approx(ph.e, rel=1e-10)
    assert on.P - off.P == pytest.approx(ph.P, rel=1e-10)
    assert on.s - off.s == pytest.approx(ph.s, rel=1e-10)
    # photons drop out of the T=0 solve entirely
    cold = solve_beta_eq(par, 0.16, T=0.0, include_photons=True)
    assert cold.s == 0.0


if __name__ == "__main__":
    par = Parametrization.from_dd2_defaults()
    for T in (0.01, 0.1):
        c, w = solve_snm(par, 0.16, 0.0), solve_snm(par, 0.16, T)
        print(f"SNM T={T}: |d_eps|rel={abs(w.eps/c.eps-1):.1e} "
              f"|dP|={abs(w.P-c.P):.1e} |dmu|={abs(w.mu_n-c.mu_n):.1e} "
              f"HVH={w.hvh_rel:.1e}")
    for T in (1.0, 10.0, 30.0):
        b = solve_beta_eq(par, 0.16, T=T)
        print(f"beta T={T}: Yp={b.Y_p:.4f} s/nB={b.s/b.n_B:.4f} "
              f"P={b.P:.3f} HVH={b.hvh_rel:.1e}")
