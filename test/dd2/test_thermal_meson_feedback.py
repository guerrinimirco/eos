"""
Self-consistent coupling of the thermal meson gas to the charge and
strangeness constraints.

The thermal pseudoscalar/vector nonets carry electric charge and strangeness,
so they belong inside the constraints the solver imposes, not only in the
thermodynamic totals added afterwards:

    neutrality  :  n_C^baryons + n_C^mesons = n_e + n_mu
    fixed Y_C   :  (n_C^baryons + n_C^mesons)/n_B = Y_C
    fixed Y_S   :  (n_S^baryons + n_S^mesons)/n_B = Y_S

They carry no baryon number, so the n_B constraint stays baryons-only, and
they do not source the sigma/omega/rho/phi field equations — an ideal Bose gas
rides on the mean fields rather than generating them.

The gas vanishes at T = 0, so every result here is unchanged for a cold star;
the feedback matters at the temperatures of supernova and merger matter, where
the thermal K-/pi- population is a percent-level fraction of n_B.
"""
import numpy as np
import pytest

from eos.dd2 import Parametrization, SpeciesFlags, solve_octet
from eos.dd2.physics.mesons import lambda_omega_ratio, thermal_meson_charges
from eos.dd2.physics.octet import (build_octet_ctx, meson_charges_nat,
                                   octet_residual)
from eos.dd2.physics.jacobian import octet_jacobian
from eos.general.particles import get_particle


@pytest.fixture(scope="module")
def par():
    return Parametrization.from_dd2y_defaults()


def _flags(ps=False, tv=False, hyperons=True):
    return SpeciesFlags(hyperons=hyperons, muons=True,
                        include_pseudoscalars=ps, include_thermal_vectors=tv)


def _meson_charge_at(par, point, ps=True, tv=False):
    """(n_C, n_S) [fm^-3] of the gas at a converged point's potentials."""
    _, Gw, Gr, _, _, _ = par.couplings_at(point.n_B)
    return thermal_meson_charges(
        Gw, Gr, lambda_omega_ratio(par),
        -point.mu_e, point.mu_S, point.omega0, point.rho0, point.T,
        include_pseudoscalars=ps, include_thermal_vectors=tv)


def _baryon_charge(point):
    return sum(get_particle(name).charge * n for name, n in point.composition)


# --------------------------------------------------------------- neutrality
def test_neutrality_counts_the_meson_charge(par):
    """Neutrality closes on baryons + mesons, and NOT on baryons alone."""
    p = solve_octet(par, 0.4, _flags(ps=True), T=60.0)
    n_C_meson, _ = _meson_charge_at(par, p)
    leptons = p.n_e + p.n_mu

    assert _baryon_charge(p) + n_C_meson == pytest.approx(leptons, abs=1e-12)
    # the gas is not a rounding effect at this temperature: dropping it would
    # violate neutrality at the percent level of n_B
    assert abs(n_C_meson) > 1e-3 * p.n_B
    assert abs(_baryon_charge(p) - leptons) > 1e-3 * p.n_B


def test_thermal_mesons_are_negatively_charged_in_neutron_rich_matter(par):
    """mu_Q < 0 favours pi-/K-, so the gas pushes Y_p up and Y_e down."""
    base = solve_octet(par, 0.4, _flags(), T=60.0)
    fed = solve_octet(par, 0.4, _flags(ps=True), T=60.0)
    n_C_meson, _ = _meson_charge_at(par, fed)

    assert n_C_meson < 0.0
    assert fed.Y_p > base.Y_p          # protons make up the deficit
    assert fed.Y_e < base.Y_e          # fewer electrons needed


def test_no_feedback_at_T0_or_with_flags_off(par):
    """Cold matter and flags-off are bit-identical to the uncoupled solve."""
    for T in (0.0, 40.0):
        base = solve_octet(par, 0.3, _flags(), T=T)
        off = solve_octet(par, 0.3, _flags(ps=False, tv=False), T=T)
        assert off.P == base.P and off.n_p == base.n_p

    cold_base = solve_octet(par, 0.3, _flags(), T=0.0)
    cold_fed = solve_octet(par, 0.3, _flags(ps=True, tv=True), T=0.0)
    assert cold_fed.P == pytest.approx(cold_base.P, rel=1e-14)
    assert cold_fed.n_p == pytest.approx(cold_base.n_p, rel=1e-14)


# ------------------------------------------------------- charge/strangeness
def test_mesons_carry_no_baryon_number(par):
    """n_B is met by the baryons alone, gas or no gas."""
    n_B = 0.4
    p = solve_octet(par, n_B, _flags(ps=True, tv=True), T=60.0)
    assert sum(n for _, n in p.composition) == pytest.approx(n_B, rel=1e-10)


def test_fixed_YS_counts_thermal_kaons(par):
    """With Y_S imposed, thermal kaon strangeness is part of the budget."""
    n_B, Y_S = 0.4, 0.05
    p = solve_octet(par, n_B, _flags(ps=True), T=60.0, charge_mode="fixed",
                    Y_C=0.3, strange_mode="fixed", Y_S=Y_S)
    _, n_S_meson = _meson_charge_at(par, p)
    n_S_baryon = sum(get_particle(name).strangeness * n
                     for name, n in p.composition)

    assert (n_S_baryon + n_S_meson) / n_B == pytest.approx(Y_S, abs=1e-10)
    assert abs(n_S_meson) > 1e-4 * n_B      # kaons genuinely contribute


def test_fixed_YC_counts_meson_charge(par):
    n_B, Y_C = 0.4, 0.3
    p = solve_octet(par, n_B, _flags(ps=True), T=60.0, charge_mode="fixed",
                    Y_C=Y_C)
    n_C_meson, _ = _meson_charge_at(par, p)
    assert (_baryon_charge(p) + n_C_meson) / n_B == pytest.approx(Y_C, abs=1e-10)


# ------------------------------------------------------------- consistency
def test_hvh_still_holds_with_feedback(par):
    for T in (20.0, 60.0, 90.0):
        p = solve_octet(par, 0.4, _flags(ps=True, tv=True), T=T)
        assert abs(p.hvh_rel) < 1e-10


@pytest.mark.parametrize("mode", ["neutral", "fixed"])
def test_analytic_jacobian_matches_finite_difference(par, mode):
    """The meson block of the analytic Jacobian against a central difference.

    Required of every analytic Jacobian in this repository; the meson rows are
    themselves differenced, so this checks they land in the right columns and
    carry the right sign, not that they beat the difference in accuracy.
    """
    kw = dict(charge_mode=mode)
    if mode == "fixed":
        kw.update(Y_C=0.3, strange_mode="fixed", Y_S=0.05)
    p = solve_octet(par, 0.4, _flags(ps=True, tv=True), T=60.0, **kw)

    ctx = build_octet_ctx(par, 0.4, _flags(ps=True, tv=True), T=60.0, **kw)
    x = [p.sigma, p.omega0, p.rho0, p.phi0, p.mu_n - p.Sigma_R, -p.mu_e]
    if ctx.has_muS:
        x.append(p.mu_S)
    x = np.array(x, dtype=float)

    J = np.asarray(octet_jacobian(x, ctx))
    fd = np.zeros_like(J)
    for i in range(len(x)):
        h = max(1e-5, 1e-6 * abs(x[i]))
        hi, lo = x.copy(), x.copy()
        hi[i] += h
        lo[i] -= h
        fd[:, i] = (np.array(octet_residual(hi, ctx))
                    - np.array(octet_residual(lo, ctx))) / (2.0 * h)

    scale = max(np.abs(fd).max(), 1e-12)
    assert np.abs(J - fd).max() / scale < 1e-4


def test_fast_and_ref_backends_agree(par):
    """analytic-Jacobian and finite-difference solves land on the same state."""
    fast = solve_octet(par, 0.4, _flags(ps=True, tv=True), T=60.0,
                       analytic_jac=True)
    ref = solve_octet(par, 0.4, _flags(ps=True, tv=True), T=60.0,
                      analytic_jac=False)
    assert fast.P == pytest.approx(ref.P, rel=1e-9)
    assert fast.n_p == pytest.approx(ref.n_p, rel=1e-9)
    assert fast.mu_e == pytest.approx(ref.mu_e, rel=1e-9)


# ------------------------------------------------------------------- units
def test_meson_charges_nat_is_hc3_of_the_fm_value(par):
    from eos.general.physics_constants import hc3
    ctx = build_octet_ctx(par, 0.4, _flags(ps=True), T=60.0)
    nat = meson_charges_nat(ctx, mu_Q=-80.0, mu_S=0.0, omega0=120.0, rho0=-8.0)
    fm = thermal_meson_charges(ctx.Gw_N, ctx.Gr_N, ctx.x_omega_L, -80.0, 0.0,
                               120.0, -8.0, 60.0, include_pseudoscalars=True)
    assert nat[0] == pytest.approx(fm[0] * hc3, rel=1e-14)
    assert nat[1] == pytest.approx(fm[1] * hc3, rel=1e-14)


if __name__ == "__main__":
    par = Parametrization.from_dd2y_defaults()
    print(f"{'T':>6} {'n_C^bar':>10} {'n_C^mes':>10} {'n_e+n_mu':>10} {'Y_p':>8}")
    for T in (0.0, 20.0, 40.0, 60.0, 80.0):
        p = solve_octet(par, 0.4, _flags(ps=True), T=T)
        mC, _ = _meson_charge_at(par, p)
        print(f"{T:6.1f} {_baryon_charge(p):10.5f} {mC:10.5f} "
              f"{p.n_e + p.n_mu:10.5f} {p.Y_p:8.5f}")
