"""
M4 gate (slow): TOV cross-check and DD2Y CompOSE comparison.

TOV M_max >= 2 M_sun is the load-bearing DD2Y gate. The DD2 nucleonic run is
the pipeline validation: it must reproduce the published DD2 neutron-star
point (M_max ~ 2.42, R_1.4 ~ 13.2 km) closely.

DD2Y CompOSE: with the exact Fortin/Marques R-couplings hardcoded (report v8
§2.4) and density-dependent φ, DD2Y now reproduces to <0.1% in eps:
- fixed-Y_q general-purpose table (no lepton cancellation): eps ~0.02%,
  P ~0.17% — the couplings, isolated;
- cold-NS β-eq table: eps ~0.09%, mu_B ~0.23%; P median ~0.9% / max ~2.5%
  because β-eq P is a small difference of large numbers with lepton
  cancellation, which amplifies any residual (0.33 vs 1/3 Ξ rounding, table
  interpolation). eps and mu_B are the convention-robust indicators.
"""
import os

import pytest

from eos.dd2 import Parametrization, SpeciesFlags, sweep_beta_eq_octet
from eos.dd2.verify.compose import (
    DD2Y_NS, DD2Y_COMPOSE, compare_ns, compare_slice,
)
from eos.dd2.verify.tov import mass_radius

import numpy as np


def test_tov_dd2_nucleonic_pipeline():
    # Pipeline validation: DD2 (no hyperons) vs published NS point.
    par = Parametrization.from_dd2_defaults()
    r = mass_radius(par, SpeciesFlags(hyperons=False, phi_field=False))
    assert r["M_max"] == pytest.approx(2.42, abs=0.05)
    assert r["R_1p4"] == pytest.approx(13.2, abs=0.4)


def test_tov_dd2y_mmax_over_two():
    # Load-bearing DD2Y gate.
    par = Parametrization.from_dd2y_defaults()
    r = mass_radius(par, SpeciesFlags(hyperons=True, phi_field=True))
    assert r["M_max"] >= 2.0
    # hyperon softening: below the nucleonic M_max, above the 2 Msun gate
    assert r["M_max"] < 2.42


@pytest.mark.skipif(not os.path.isfile(DD2Y_NS),
                    reason="DD2Y CompOSE NS table not downloaded")
def test_dd2y_compose_eps_mu():
    from dataclasses import replace
    par = replace(Parametrization.from_dd2y_defaults(),
                  nucleon_mass_mode="physical")
    flags = SpeciesFlags(hyperons=True, phi_field=True)
    grid = np.geomspace(0.06, 1.25, 130)
    pts = sweep_beta_eq_octet(par, grid, flags, include_photons=False)
    cmp = compare_ns(pts, nB_min=0.2, nB_max=1.0)
    # exact DD2Y R-couplings: eps/mu_B now <0.1%/0.3% (were 4-8%).
    assert cmp["max_err_eps"] < 2e-3
    assert cmp["max_err_muB"] < 5e-3
    # β-eq P amplifies the residual via lepton cancellation (module docstring).
    assert cmp["med_err_P"] < 2e-2


@pytest.mark.skipif(not os.path.isfile(os.path.join(DD2Y_COMPOSE, "eos.thermo")),
                    reason="DD2Y CompOSE general-purpose table not downloaded")
def test_dd2y_compose_fixed_yq():
    # Fixed-Y_q general-purpose table (no lepton cancellation) isolates the
    # couplings: <0.1% eps, <0.5% P over the uniform region.
    from dataclasses import replace
    par = replace(Parametrization.from_dd2y_defaults(),
                  nucleon_mass_mode="physical")
    flags = SpeciesFlags(hyperons=True, phi_field=True)
    r = compare_slice(par, DD2Y_COMPOSE, T=0.1, Y_q=0.3,
                      nB_min=0.2, nB_max=0.8, flags=flags, name="DD2Y")
    assert r["max_err_eps"] < 1e-3
    assert r["max_err_P"] < 5e-3
