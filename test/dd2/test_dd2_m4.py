"""
M4 gate (fast physics): hyperons + φ.

- octet solver reduces to the nucleon solver when hyperons are off (physics
  consistency without touching the gated M2/M3 path);
- hyperon scalar couplings round-trip the input potentials;
- Λ onset near the DD2Y literature value (~2.2 n_sat), correct onset order;
- Hugenholtz–Van Hove holds through every onset;
- φ field present iff hyperons + phi_field; SpeciesFlags guards fire.

The TOV M_max >= 2 gate and the DD2Y CompOSE comparison are the slower
test_dd2_m4_tov.py (need the downloaded table / TOV integration).
"""
import numpy as np
import pytest

from eos.dd2 import (
    Parametrization, SpeciesFlags, solve_beta_eq, solve_beta_eq_octet,
    sweep_beta_eq_octet,
)


@pytest.fixture(scope="module")
def par_y():
    return Parametrization.from_dd2y_defaults()


@pytest.fixture(scope="module")
def flags_y():
    return SpeciesFlags(hyperons=True, phi_field=True)


def test_octet_reduces_to_nucleon():
    # Hyperons off: the general octet path must match the gated nucleon path.
    par = Parametrization.from_dd2_defaults()
    flags = SpeciesFlags(hyperons=False, phi_field=False)
    for n_B in (0.10, 0.16, 0.30, 0.60):
        o = solve_beta_eq_octet(par, n_B, flags)
        n = solve_beta_eq(par, n_B)
        assert o.Y_p == pytest.approx(n.Y_p, rel=1e-9)
        assert o.eps == pytest.approx(n.eps, rel=1e-9)
        assert o.P == pytest.approx(n.P, rel=1e-9)
        assert o.phi0 == 0.0


def test_dd2y_coupling_table(par_y):
    # Bit-exact DD2Y R-couplings (Fortin 2017 Table 1): R_sigma hardcoded,
    # SU(6) vectors, DD2Y masses, density-dependent φ (x_phi = R_phi ratio).
    _s2 = 1.4142135623730951
    expect = {  # (mass, x_sigma, x_omega, x_rho, x_phi)
        "Lambda": (1115.683, 0.62, 2 / 3, 0.0, -_s2 / 3),
        "Sigma-": (1190.0, 0.48, 2 / 3, 2.0, -_s2 / 3),
        "Xi-": (1321.68, 0.32, 1 / 3, 1.0, -2 * _s2 / 3),
    }
    for name, row in expect.items():
        assert par_y.hyperon_coupling_map[name] == pytest.approx(row, rel=1e-12)


def test_scalar_potential_inversion():
    # The U_Y inversion (report §2.4b) regenerates the DD2Y R_sigma table when
    # U_Xi = -18 (B2): the mechanism and the hardcoded table agree.
    # (approximate: R_sigma is 2-digit-rounded and the inversion is sensitive
    # to the saturation σ-solve — report §2.4 "sensitive to the φ solve".)
    par = Parametrization.from_hyperon_potentials()   # U_Λ,Σ,Ξ = -30,30,-18
    for name, R_sigma in (("Lambda", 0.62), ("Sigma-", 0.48), ("Xi-", 0.32)):
        assert par.hyperon_coupling_map[name][1] == pytest.approx(R_sigma, abs=1e-2)


def test_lambda_onset_and_order(par_y, flags_y):
    grid = np.linspace(0.28, 0.55, 28)
    pts = sweep_beta_eq_octet(par_y, grid, flags_y, include_photons=False)
    YL = np.array([p.Y("Lambda") for p in pts])
    onset = grid[np.argmax(YL > 1e-4)]
    # DD2Y literature: Lambda first, around 2.0-2.4 n_sat.
    assert 2.0 * par_y.n_sat < onset < 2.4 * par_y.n_sat
    # Lambda is the first hyperon to appear.
    last = pts[-1]
    assert last.Y("Lambda") > last.Y("Sigma-")
    assert last.Y("Lambda") > last.Y("Xi-")
    assert last.Y("Sigma+") < 1e-6            # Sigma+ suppressed in cold NS


def test_hvh_through_onsets(par_y, flags_y):
    from eos.general.particles import get_particle
    grid = np.geomspace(0.10, 1.2, 40)
    for p in sweep_beta_eq_octet(par_y, grid, flags_y, include_photons=False):
        assert abs(p.hvh_rel) < 1e-11
        assert p.mu_n - p.mu_p == pytest.approx(p.mu_e, abs=1e-6)
        # charge neutrality including hyperon charges
        Q = sum(get_particle(name).charge * n for name, n in p.composition)
        assert Q - (p.n_e + p.n_mu) == pytest.approx(0.0, abs=1e-9)


def test_phi_field_presence(par_y):
    on = solve_beta_eq_octet(par_y, 0.8, SpeciesFlags(hyperons=True, phi_field=True))
    off = solve_beta_eq_octet(par_y, 0.8, SpeciesFlags(hyperons=True, phi_field=False))
    assert on.phi0 != 0.0                     # φ sourced by hyperons
    assert off.phi0 == 0.0
    # φ repulsion stiffens: pressure is higher with the field on
    assert on.P > off.P


def test_speciesflags_guards():
    # sigma_star is not yet wired; deltas (M5) and neutrinos (M6) are.
    with pytest.raises(NotImplementedError, match="sigma_star"):
        SpeciesFlags(sigma_star=True)


if __name__ == "__main__":
    par = Parametrization.from_dd2y_defaults()
    flags = SpeciesFlags(hyperons=True, phi_field=True)
    grid = np.linspace(0.1, 1.2, 34)
    pts = sweep_beta_eq_octet(par, grid, flags, include_photons=False)
    YL = np.array([p.Y("Lambda") for p in pts])
    onset = grid[np.argmax(YL > 1e-4)]
    print(f"Lambda onset = {onset:.3f} fm^-3 = {onset / par.n_sat:.2f} n_sat")
    print(f"max |HVH| = {max(abs(p.hvh_rel) for p in pts):.1e}")
