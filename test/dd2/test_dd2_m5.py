"""
M5 gate: Δ isobars (ratio couplings, τ₃ = 2 I₃ quartet).

Gate (report §3.y M5): Δ onset vs published DD2Δ; stability preserved.
- Δ⁻ appears first (neutron-rich matter favors the negative isobar), near the
  DD2Δ literature onset ~2 n_sat for x_Δσ = x_Δω = 1;
- the EoS stays causal (0 < c_s² < 1) and mechanically stable (dP/dn > 0)
  throughout its valid range, with HVH at round-off;
- the high-density scalar-collapse boundary (m* → 0, a real DD-RMF Δ feature
  with x_Δσ ~ 1) is detected and flagged, not silently truncated.
"""
import numpy as np
import pytest

from eos.dd2 import (
    Parametrization, SpeciesFlags, solve_beta_eq_octet, sweep_beta_eq_octet,
)
from eos.general.particles import DeltaPP, DeltaP, Delta0, DeltaM


@pytest.fixture(scope="module")
def par():
    return Parametrization.from_dd2_defaults()


@pytest.fixture(scope="module")
def flags_d():
    return SpeciesFlags(hyperons=False, deltas=True, phi_field=False)


def test_delta_t3_quartet():
    # τ₃ = 2 I₃ (report §1.4 rule): {+3, +1, -1, -3}.
    assert (DeltaPP.t3, DeltaP.t3, Delta0.t3, DeltaM.t3) == (3.0, 1.0, -1.0, -3.0)


def test_delta_onset_and_order(par, flags_d):
    grid = np.geomspace(0.10, 0.85, 60)
    pts = sweep_beta_eq_octet(par, grid, flags_d, include_photons=False)
    N = np.array([p.n_B for p in pts])
    YDm = np.array([p.Y("Delta-") for p in pts])
    onset = N[np.argmax(YDm > 1e-4)]
    # DD2Δ literature: Δ⁻ first, near ~2 n_sat for x_Δσ = x_Δω = 1.
    assert 1.8 * par.n_sat < onset < 2.5 * par.n_sat
    last = pts[-1]
    assert last.Y("Delta-") > last.Y("Delta0") > last.Y("Delta+")
    assert last.Y("Delta++") < 1e-4          # Δ++ suppressed in neutron-rich NS


def test_delta_stability_and_hvh(par, flags_d):
    grid = np.geomspace(0.08, 0.9, 70)
    pts = sweep_beta_eq_octet(par, grid, flags_d, include_photons=False,
                              stop_at_boundary=True)
    N = np.array([p.n_B for p in pts])
    P = np.array([p.P for p in pts])
    eps = np.array([p.eps for p in pts])
    assert np.all(np.diff(P) > 0)                    # mechanically stable
    cs2 = np.diff(P) / np.diff(eps)
    assert np.all((cs2 > 0) & (cs2 < 1))             # causal
    assert max(abs(p.hvh_rel) for p in pts) < 1e-11


def test_scalar_collapse_flagged(par, flags_d):
    grid = np.geomspace(0.10, 1.3, 80)
    # default: raise past the collapse boundary (no silent truncation)
    with pytest.raises(RuntimeError):
        sweep_beta_eq_octet(par, grid, flags_d, include_photons=False)
    # boundary mode: return the valid prefix and stop cleanly before m* -> 0
    pts = sweep_beta_eq_octet(par, grid, flags_d, include_photons=False,
                              stop_at_boundary=True)
    assert 5.5 * par.n_sat < pts[-1].n_B < 7.0 * par.n_sat
    assert pts[-1].m_eff / par.m_nucleon < 0.05      # near scalar collapse


def test_delta_with_hyperons(par):
    # Full baryon set (octet + Δ + φ) solves and stays consistent.
    pary = Parametrization.from_dd2y_defaults()
    flags = SpeciesFlags(hyperons=True, deltas=True, phi_field=True)
    p = solve_beta_eq_octet(pary, 0.9, flags)
    assert p.Y("Delta-") > 0.0 and p.Y("Lambda") > 0.0
    assert abs(p.hvh_rel) < 1e-11


def _onset(par, flags, grid, name="Delta-"):
    pts = sweep_beta_eq_octet(par, grid, flags, include_photons=False,
                              stop_at_boundary=True)
    Y = np.array([p.Y(name) for p in pts])
    if not (Y > 1e-4).any():
        return np.inf                        # not yet onset within the grid
    return np.array([p.n_B for p in pts])[np.argmax(Y > 1e-4)]


def test_delta_potential_constructor(par):
    # report v11 §2.4: no canonical DD2Δ table -> calibrate x_Δσ from a chosen
    # (U_Δ, x_Δω) point. Validate the potential round-trips and the onset is
    # physical for that calibration.
    from eos.dd2 import solve_snm
    U_target, x_wD = -75.0, 1.0
    pd = Parametrization.from_delta_potential(U_Delta=U_target, x_wD=x_wD)
    sat = solve_snm(pd, pd.n_sat)
    Gs, Gw, _, _, _, _ = pd.couplings_at(pd.n_sat)
    U = (-pd.x_Delta_sigma * Gs * sat.sigma + pd.x_Delta_omega * Gw * sat.omega0
         + sat.Sigma_R)
    assert U == pytest.approx(U_target, abs=1e-6)     # round-trips
    assert pd.x_Delta_omega == x_wD
    # chosen-point onset is in the physical DD2Δ range
    flags = SpeciesFlags(hyperons=False, deltas=True, phi_field=False)
    onset = _onset(pd, flags, np.geomspace(0.15, 0.7, 50))
    assert 1.8 * pd.n_sat < onset < 2.6 * pd.n_sat


def test_delta_potential_range_guard():
    # U_Δ outside the literature range is flagged, not silently extrapolated.
    with pytest.raises(ValueError, match="range"):
        Parametrization.from_delta_potential(U_Delta=-20.0)


def test_delta_coupling_ratios_configurable():
    from dataclasses import replace
    par = Parametrization.from_dd2_defaults()
    flags = SpeciesFlags(hyperons=False, deltas=True, phi_field=False)
    grid = np.geomspace(0.15, 0.6, 50)
    # Scalar attraction is the monotone onset knob: more x_Δσ -> earlier Δ⁻.
    onset_more = _onset(replace(par, x_Delta_sigma=1.2), flags, grid)
    onset_less = _onset(replace(par, x_Delta_sigma=0.8), flags, grid)
    assert onset_more < _onset(par, flags, grid) < onset_less


if __name__ == "__main__":
    par = Parametrization.from_dd2_defaults()
    flags = SpeciesFlags(hyperons=False, deltas=True, phi_field=False)
    pts = sweep_beta_eq_octet(par, np.geomspace(0.1, 1.3, 80), flags,
                              include_photons=False, stop_at_boundary=True)
    N = np.array([p.n_B for p in pts])
    onset = N[np.argmax(np.array([p.Y("Delta-") for p in pts]) > 1e-4)]
    print(f"Δ⁻ onset = {onset:.3f} fm^-3 = {onset / par.n_sat:.2f} n_sat")
    print(f"valid to {N[-1]:.3f} fm^-3 = {N[-1] / par.n_sat:.2f} n_sat "
          f"(scalar collapse, m*/m = {pts[-1].m_eff / par.m_nucleon:.3f})")
