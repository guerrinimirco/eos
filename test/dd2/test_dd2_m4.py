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
    sweep_beta_eq_octet, solve_snm,
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


def test_scalar_coupling_roundtrip(par_y):
    # U_Y = -Gamma_sigmaY sigma + Gamma_omegaY omega0 + Sigma^R at saturation.
    sat = solve_snm(par_y, par_y.n_sat)
    Gs, Gw, _, _, _, _ = par_y.couplings_at(par_y.n_sat)
    targets = {"Lambda": -30.0, "Sigma-": 30.0, "Xi-": -14.0}
    for name, U_target in targets.items():
        xs, xw, xr, gphi = par_y.hyperon_coupling_map[name]
        U = -xs * Gs * sat.sigma + xw * Gw * sat.omega0 + sat.Sigma_R
        assert U == pytest.approx(U_target, abs=1e-6)


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
    with pytest.raises(NotImplementedError, match="deltas"):
        SpeciesFlags(deltas=True)
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
