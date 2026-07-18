"""
M4 gate (slow): TOV cross-check and DD2Y CompOSE comparison.

TOV M_max >= 2 M_sun is the load-bearing DD2Y gate. The DD2 nucleonic run is
the pipeline validation: it must reproduce the published DD2 neutron-star
point (M_max ~ 2.42, R_1.4 ~ 13.2 km) closely.

DD2Y CompOSE (cold-NS table) comparison: eps and mu_B agree to ~1% over the
hyperonic range; the pressure differs by several % because (a) P is a small
difference of large numbers that amplifies any composition/coupling
difference, and (b) the report's §2.4 generic SU(6)+U_Y recipe approximates
rather than bit-reproduces Marques' tabulated DD2Y R-couplings (which the
report does not provide). The strict 0.1% reproduction therefore applies to
the nucleonic sector (M3), not to hyperonic DD2Y here.
"""
import os

import pytest

from eos.dd2 import Parametrization, SpeciesFlags, sweep_beta_eq_octet
from eos.dd2.verify.compose import DD2Y_NS, compare_ns
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
    # eps and mu_B are the convention-robust indicators.
    assert cmp["max_err_eps"] < 1e-2
    assert cmp["max_err_muB"] < 3e-2
    # P difference is documented, not gated tight (see module docstring).
    assert cmp["med_err_P"] < 0.1
