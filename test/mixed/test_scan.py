"""
The parameter-space scan: does a hybrid equation of state exist here?

Two things must hold for a sample to count, and the scan must report — not
raise — when either fails. A scan that aborts on the first pathological sample
cannot map a boundary, and the pathological samples ARE the boundary.
"""
import numpy as np
import pytest

from eos.dd2 import Parametrization, SpeciesFlags, compute_nmp
from eos.mixed import scan_parameters, scan_point, grid_samples, NMP_KEYS

FLAGS = SpeciesFlags(hyperons=False, phi_field=False, muons=False)
GRID = np.linspace(0.05, 1.6, 110)


@pytest.fixture(scope="module")
def dd2_nmp():
    return compute_nmp(Parametrization.from_dd2_defaults())


def test_known_good_sample_finds_a_window(dd2_nmp):
    row = scan_point(dd2_nmp, {"B4": 180.0}, FLAGS, GRID)
    assert row["status"] == "ok", row["status"]
    assert row["inversion_ok"] == 1.0
    assert row["window_exists"] == 1.0
    assert 0.0 < row["n_onset"] < row["n_offset"]
    assert row["B4"] == 180.0


def test_unrepresentable_nmp_is_reported_not_raised(dd2_nmp):
    """K_sat = 10 GeV has no DD-RMF realisation. The scan must say so."""
    row = scan_point(dict(dd2_nmp, K_sat=1.0e4), {"B4": 180.0}, FLAGS, GRID)
    assert row["inversion_ok"] == 0.0
    assert row["status"] != "ok"
    assert np.isnan(row["n_onset"])


def test_large_bag_constant_removes_the_transition(dd2_nmp):
    """A high enough bag constant makes quark matter unfavourable everywhere.
    'No window' is the correct physical answer, not a solver failure — the
    scan must record it and carry on."""
    row = scan_point(dd2_nmp, {"B4": 400.0}, FLAGS, GRID)
    assert row["inversion_ok"] == 1.0
    assert row["window_exists"] == 0.0
    assert row["status"] == "no_window"


def test_onset_rises_with_the_bag_constant(dd2_nmp):
    """Raising B^1/4 costs more to make quark matter, so the transition starts
    at a higher density. A scan that got this backwards would be wired wrong."""
    onsets = [scan_point(dd2_nmp, {"B4": b}, FLAGS, GRID)["n_onset"]
              for b in (180.0, 220.0, 260.0)]
    assert all(np.isfinite(o) for o in onsets), onsets
    assert onsets[0] < onsets[1] < onsets[2], onsets


def test_scan_is_the_product_of_its_axes(dd2_nmp):
    vmit = grid_samples(B4=[180.0, 260.0])
    rows = scan_parameters([dd2_nmp], vmit, FLAGS, GRID)
    assert len(rows) == 2
    assert {r["B4"] for r in rows} == {180.0, 260.0}
    assert all(set(NMP_KEYS) <= set(r) for r in rows)
    assert all(r["seconds"] >= 0.0 for r in rows)


def test_progress_callback_sees_every_row(dd2_nmp):
    seen = []
    rows = scan_parameters([dd2_nmp], grid_samples(B4=[180.0]), FLAGS, GRID,
                           progress=seen.append)
    assert len(seen) == len(rows) == 1


def test_missing_nmp_key_raises(dd2_nmp):
    """A malformed request is a caller bug, not a finding — that one raises."""
    with pytest.raises(ValueError, match="missing"):
        scan_point({k: v for k, v in dd2_nmp.items() if k != "K_sat"},
                   {"B4": 180.0}, FLAGS, GRID)


def test_tov_columns_appear_and_are_physical(dd2_nmp):
    row = scan_point(dd2_nmp, {"B4": 180.0}, FLAGS, GRID, tov=True)
    assert row["status"] == "ok", row["status"]
    assert 1.5 < row["M_max"] < 3.0, row["M_max"]
    assert 9.0 < row["R_Mmax"] < 16.0, row["R_Mmax"]
    assert 0.0 < row["cs2_max"] <= 1.0, row["cs2_max"]


def test_tov_columns_are_nan_without_a_window(dd2_nmp):
    """No window means no hybrid core to integrate. The columns must be present
    and nan, not absent — a ragged table is worse than an empty cell."""
    row = scan_point(dd2_nmp, {"B4": 400.0}, FLAGS, GRID, tov=True)
    assert row["window_exists"] == 0.0
    assert np.isnan(row["M_max"]) and np.isnan(row["R_1p4"])


def test_tov_off_by_default(dd2_nmp):
    row = scan_point(dd2_nmp, {"B4": 180.0}, FLAGS, GRID)
    assert "M_max" not in row


def test_grid_samples_is_a_product():
    out = grid_samples(B4=[150.0, 180.0], a=[0.0, 0.2, 0.4])
    assert len(out) == 6
    assert {(d["B4"], d["a"]) for d in out} == {
        (b, a) for b in (150.0, 180.0) for a in (0.0, 0.2, 0.4)}
