"""
M6 remainder gate: Y_L / neutrino-trapped mode, entropy-per-baryon (S=s/n_B)
axis, and the TableSpec driver.

Validated:
- trapped Y_L hits the target electron lepton fraction (n_e+n_nue)/n_B exactly,
  with the trapped beta relation mu_n-mu_p = mu_e-mu_nue and HVH holding;
- neutrino trapping stiffens (higher Y_L -> higher P) and mu_L>0;
- the transparent limit is unchanged (mu_L=0, no neutrinos);
- the entropy axis recovers the T hitting a target s/n_B;
- build_table produces the (nB x T) and (nB x SnB) grids for each mode.
"""
import numpy as np
import pytest

from eos.dd2 import (
    Parametrization, SpeciesFlags, solve_yl_octet, solve_beta_eq_octet,
    solve_octet_at_entropy, solve_fixed_yc_octet, TableSpec, build_table,
)
from eos.general.particles import get_particle


def _hadronic_YC(p):
    return sum(get_particle(n).charge * d for n, d in p.composition) / p.n_B


@pytest.fixture(scope="module")
def par_y():
    return Parametrization.from_dd2y_defaults()


@pytest.fixture(scope="module")
def flags_nu():
    return SpeciesFlags(hyperons=True, phi_field=True, neutrinos=True)


@pytest.mark.parametrize("Y_L", [0.3, 0.4])
@pytest.mark.parametrize("T", [0.0, 30.0])
def test_trapped_yl_hits_target(par_y, flags_nu, Y_L, T):
    p = solve_yl_octet(par_y, 0.5, Y_L, flags_nu, T=T)
    assert (p.n_e + p.n_nu) / p.n_B == pytest.approx(Y_L, abs=1e-9)
    # trapped beta equilibrium: mu_n - mu_p = mu_e - mu_nue (mu_nue = mu_L)
    assert p.mu_n - p.mu_p == pytest.approx(p.mu_e - p.mu_L, abs=1e-6)
    assert abs(p.hvh_rel) < 1e-10
    assert p.mu_L > 0.0 and p.n_nu > 0.0


def test_trapping_stiffens(par_y, flags_nu):
    lo = solve_yl_octet(par_y, 0.6, 0.2, flags_nu, T=10.0)
    hi = solve_yl_octet(par_y, 0.6, 0.4, flags_nu, T=10.0)
    assert hi.P > lo.P                       # trapped neutrinos add pressure


def test_transparent_limit_unchanged(par_y, flags_nu):
    # solve_yl at the self-consistent (free) Y_L would reproduce beta-eq; here
    # just assert the transparent beta path carries no neutrinos / mu_L.
    t = solve_beta_eq_octet(par_y, 0.5, flags_nu, include_photons=False)
    assert t.mu_L == 0.0 and t.n_nu == 0.0


def test_neutrino_flag_required(par_y):
    flags = SpeciesFlags(hyperons=True, phi_field=True)   # neutrinos off
    with pytest.raises(ValueError, match="neutrinos"):
        solve_yl_octet(par_y, 0.5, 0.3, flags)


def test_entropy_axis_recovers_T(par_y):
    flags = SpeciesFlags(hyperons=True, phi_field=True)
    S_target = 1.5
    p = solve_octet_at_entropy(par_y, 0.5, S_target, flags, charge_mode="neutral")
    assert p.s / p.n_B == pytest.approx(S_target, rel=1e-4)
    assert p.T > 0.0
    assert abs(p.hvh_rel) < 1e-9


def test_tablespec_T_axis(par_y):
    spec = TableSpec(
        parametrization=par_y, mode="beta_eq_neutrinoless",
        axes={"nB": np.linspace(0.15, 0.6, 5), "T": [0.0, 20.0]},
        include=SpeciesFlags(hyperons=True, phi_field=True),
        want_coeffs=True,
    )
    res = build_table(spec)
    assert len(res.points) == 2 and len(res.points[0]) == 5
    # causal sound speed on each line
    for line in res.cs2_eq:
        assert np.all((line >= 0.0) & (line <= 1.0))


def test_tablespec_entropy_axis_and_yc(par_y):
    spec = TableSpec(
        parametrization=par_y, mode="fixed_YC",
        axes={"nB": np.linspace(0.2, 0.6, 4), "SnB": [1.0]},
        include=SpeciesFlags(hyperons=True, phi_field=True),
        fixed={"Y_C": 0.2},
    )
    res = build_table(spec)
    line = res.points[0]
    assert res.temp_key == "SnB"
    for p in line:
        assert p.s / p.n_B == pytest.approx(1.0, rel=1e-4)
        assert p.T > 0.0


# --- fixed-Y_C flavor 2b: neutralizing leptons ---------
def test_yc_2b_hadronic_matches_2a(par_y):
    # leptons don't source the mean fields: the 2b hadronic solve == 2a.
    fY = SpeciesFlags(hyperons=True, phi_field=True)
    a = solve_fixed_yc_octet(par_y, 0.6, 0.15, fY)                 # 2a leptonless
    b = solve_fixed_yc_octet(par_y, 0.6, 0.15, fY, leptons=True)   # 2b
    assert dict(b.composition) == pytest.approx(dict(a.composition), rel=1e-9)
    assert b.P > a.P                       # 2b adds lepton pressure
    assert a.n_e == 0.0                    # 2a carries no leptons


def test_yc_2b_electron_only(par_y):
    f = SpeciesFlags(hyperons=True, phi_field=True, muons=False)
    p = solve_fixed_yc_octet(par_y, 0.6, 0.15, f, leptons=True)
    assert p.Y_e == pytest.approx(0.15, rel=1e-6)     # n_e = Y_C n_B
    assert p.Y_mu == 0.0
    assert _hadronic_YC(p) - p.Y_e - p.Y_mu == pytest.approx(0.0, abs=1e-9)  # neutral
    assert abs(p.hvh_rel) < 1e-10


def test_yc_2b_electrons_and_muons(par_y):
    f = SpeciesFlags(hyperons=True, phi_field=True)     # muons on
    p = solve_fixed_yc_octet(par_y, 0.6, 0.15, f, leptons=True)
    assert p.Y_e + p.Y_mu == pytest.approx(0.15, rel=1e-6)   # Y_C = Y_e + Y_mu
    assert p.Y_mu > 0.0                                       # muons populated
    assert _hadronic_YC(p) - p.Y_e - p.Y_mu == pytest.approx(0.0, abs=1e-9)
    assert abs(p.hvh_rel) < 1e-10


def test_yc_2b_composes_with_ys(par_y):
    f = SpeciesFlags(hyperons=True, phi_field=True)
    p = solve_fixed_yc_octet(par_y, 0.7, 0.1, f, leptons=True, Y_S=0.05)
    S = sum(get_particle(n).strangeness * d for n, d in p.composition) / p.n_B
    assert S == pytest.approx(0.05, abs=1e-9)
    assert p.Y_e + p.Y_mu == pytest.approx(0.1, rel=1e-6)
    assert p.mu_S != 0.0
    assert abs(p.hvh_rel) < 1e-10
