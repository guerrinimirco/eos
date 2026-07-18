"""
M6 gate (partial): fixed-Y_C and fixed-Y_C+Y_S modes (report §1.7 modes 2/3).

The unified charge-vector residual now switches the charge sector between
'neutral' (beta eq, leptons) and 'fixed' (hadronic Y_C, no leptons — the
CompOSE general-purpose convention), and optionally fixes the strangeness
fraction Y_S (adds mu_S). Y_L / neutrino-trapped mode is deferred (neutrinos
not yet wired).

Validated here:
- fixed-Y_C nucleonic == solve_composition at the matching (n_n, n_p) —
  the mode's physics is the M1/M3-gated kernel with mu_Q as the Lagrange
  multiplier for the charge constraint;
- fixed-Y_C octet recovers the beta-eq composition when Y_C is set to the
  beta-eq hadronic charge fraction;
- targets are hit exactly; HVH holds; fixed-Y_C carries no leptons;
- fixed Y_S pins the strangeness fraction with mu_S free.
"""
import pytest

from eos.dd2 import (
    Parametrization, SpeciesFlags, solve_beta_eq_octet, solve_composition,
    solve_fixed_yc_octet,
)


@pytest.fixture(scope="module")
def par_y():
    return Parametrization.from_dd2y_defaults()


def test_fixed_yc_nucleonic_matches_composition():
    # Fixed hadronic Y_C with only nucleons == fixing (n_n, n_p) directly.
    par = Parametrization.from_dd2_defaults()
    flags = SpeciesFlags(hyperons=False, phi_field=False)
    for n_B, Y_C in [(0.16, 0.3), (0.32, 0.1), (0.5, 0.2)]:
        f = solve_fixed_yc_octet(par, n_B, Y_C, flags)
        c = solve_composition(par, (1.0 - Y_C) * n_B, Y_C * n_B)
        assert f.eps == pytest.approx(c.eps, rel=1e-8)
        assert f.P == pytest.approx(c.P, rel=1e-8)
        assert f.n_p / f.n_B == pytest.approx(Y_C, abs=1e-10)
        assert f.n_e == 0.0 and f.n_mu == 0.0     # CompOSE: no leptons


def test_fixed_yc_recovers_beta_composition(par_y):
    from eos.general.particles import get_particle
    flags = SpeciesFlags(hyperons=True, phi_field=True)
    b = solve_beta_eq_octet(par_y, 0.5, flags)
    Y_C_had = sum(get_particle(name).charge * n for name, n in b.composition) \
        / b.n_B
    f = solve_fixed_yc_octet(par_y, 0.5, Y_C_had, flags)
    # same hadronic charge -> same composition (leptons aside)
    assert f.Y("Lambda") == pytest.approx(b.Y("Lambda"), rel=1e-6)
    assert f.Y("p") == pytest.approx(b.Y("p"), rel=1e-6)
    assert abs(f.hvh_rel) < 1e-11


@pytest.mark.parametrize("Y_C", [0.05, 0.1, 0.3])
def test_fixed_yc_hits_target_and_hvh(par_y, Y_C):
    from eos.general.particles import get_particle
    flags = SpeciesFlags(hyperons=True, phi_field=True)
    f = solve_fixed_yc_octet(par_y, 0.6, Y_C, flags)
    got = sum(get_particle(name).charge * n for name, n in f.composition) / f.n_B
    assert got == pytest.approx(Y_C, abs=1e-9)
    assert abs(f.hvh_rel) < 1e-11


def test_fixed_yc_ys(par_y):
    from eos.general.particles import get_particle
    flags = SpeciesFlags(hyperons=True, phi_field=True)
    # pin the strangeness fraction below the free-equilibrium value (which is
    # ~0.25 at this density). S = +1 per s-quark, so Y_S >= 0 at T=0.
    Y_S_target = 0.05
    f = solve_fixed_yc_octet(par_y, 0.7, 0.1, flags, Y_S=Y_S_target)
    S = sum(get_particle(name).strangeness * n for name, n in f.composition) \
        / f.n_B
    assert S == pytest.approx(Y_S_target, abs=1e-9)
    assert f.mu_S != 0.0                          # strangeness potential active
    assert abs(f.hvh_rel) < 1e-11


if __name__ == "__main__":
    par = Parametrization.from_dd2y_defaults()
    flags = SpeciesFlags(hyperons=True, phi_field=True)
    for Y_C in (0.05, 0.1, 0.3):
        f = solve_fixed_yc_octet(par, 0.6, Y_C, flags)
        print(f"Y_C={Y_C}: Y_Lambda={f.Y('Lambda'):.4f} "
              f"Y_Sigma-={f.Y('Sigma-'):.4f} P={f.P:.3f} HVH={f.hvh_rel:.0e}")
