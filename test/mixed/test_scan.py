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


def test_eos_guard_rejects_a_descending_table():
    """A stitched table that steps DOWN in pressure is not an equation of
    state. Integrating it returns confident nonsense (maximum masses in the
    hundreds of solar masses), so it must be caught before TOV, not after."""
    from types import SimpleNamespace
    from eos.mixed.scan import eos_is_physical

    n = np.linspace(0.1, 1.0, 40)
    good = SimpleNamespace(P=np.linspace(1.0, 400.0, 40),
                           eps=np.linspace(100.0, 1200.0, 40), n_B=n)
    assert eos_is_physical(good)[0]

    bad_P = good.P.copy()
    bad_P[15:] -= 60.0                       # the onset step-down seen in practice
    ok, reason = eos_is_physical(SimpleNamespace(P=bad_P, eps=good.eps, n_B=n))
    assert not ok and "P_not_monotone" in reason

    # superluminal: dP/deps > 1
    ok, reason = eos_is_physical(
        SimpleNamespace(P=np.linspace(1.0, 3000.0, 40), eps=good.eps, n_B=n))
    assert not ok and "cs2_out_of_range" in reason


def test_maxwell_plateau_passes_the_guard():
    """A flat plateau is dP = 0, which must be allowed — rejecting it would
    reject every eta = 1 table."""
    from types import SimpleNamespace
    from eos.mixed.scan import eos_is_physical

    P = np.concatenate([np.linspace(1.0, 200.0, 20),
                        np.full(10, 200.0),
                        np.linspace(200.0, 400.0, 20)])
    eps = np.linspace(100.0, 1400.0, P.size)
    assert eos_is_physical(
        SimpleNamespace(P=P, eps=eps, n_B=np.linspace(0.1, 1.2, P.size)))[0]


def test_build_parametrization_attaches_hyperons_and_deltas(dd2_nmp):
    """from_nmp inverts the NUCLEON sector only, so the strange and resonant
    sectors have to be attached on top or a hyperon flag fails on a lookup."""
    from eos.dd2 import SpeciesFlags
    from eos.mixed import build_parametrization

    flags = SpeciesFlags(hyperons=True, deltas=True, muons=True,
                         phi_field=True)
    par, stage, msg = build_parametrization(dd2_nmp, flags)
    assert stage == "ok", msg
    assert par.hyperon_coupling_map, "hyperon couplings were not attached"
    assert par.x_Delta_sigma != 0.0

    nucleonic, stage2, _ = build_parametrization(
        dd2_nmp, SpeciesFlags(hyperons=False, deltas=False, phi_field=False))
    assert stage2 == "ok"
    assert not nucleonic.hyperon_coupling_map


def test_build_parametrization_reports_the_two_failures_apart(dd2_nmp):
    from eos.dd2 import SpeciesFlags
    from eos.mixed import build_parametrization

    flags = SpeciesFlags(hyperons=True, deltas=True, phi_field=True)
    par, stage, _ = build_parametrization(dict(dd2_nmp, K_sat=1.0e4), flags)
    assert stage == "inversion_failed" and par is None
    # U_Delta outside the literature range is rejected by the constructor
    par, stage, _ = build_parametrization(dd2_nmp, flags, U_Delta=+10.0)
    assert stage == "sectors_failed" and par is None


def test_hadronic_stage_reports_each_check(dd2_nmp):
    from eos.dd2 import SpeciesFlags
    from eos.mixed import scan_hadronic_point

    flags = SpeciesFlags(hyperons=True, deltas=True, muons=True,
                         phi_field=True)
    row = scan_hadronic_point(dd2_nmp, flags, np.linspace(0.05, 1.6, 60),
                              tov=True)
    assert row["inversion_ok"] == 1.0 and row["sectors_ok"] == 1.0
    # Deltas drive scalar collapse before the top of the grid; the sweep is
    # expected to truncate, and what matters is how far it got.
    assert np.isfinite(row["n_sweep_max"]) and row["n_sweep_max"] > 1.0
    assert row["status"] == "ok", row["status"]
    assert 1.8 < row["M_max_had"] < 2.6, row["M_max_had"]

    bad = scan_hadronic_point(dict(dd2_nmp, K_sat=1.0e4), flags,
                              np.linspace(0.05, 1.6, 40), tov=False)
    assert bad["status"] == "inversion_failed"
    assert bad["inversion_ok"] == 0.0


def test_K_sat_cannot_be_moved_at_fixed_Q_sat(dd2_nmp):
    """Pins the documented limitation rather than leaving it folklore: the
    isoscalar cross-constraint ties K_sat to Q_sat, so K_sat alone does not
    move. If this ever starts passing, the inverter got better and the scan
    docs should be updated."""
    from eos.dd2 import Parametrization

    _, at_dd2 = Parametrization.from_nmp(dd2_nmp, return_status=True)
    assert at_dd2.ok, "the DD2 point itself must invert"
    for K in (220.0, 260.0):
        _, st = Parametrization.from_nmp(dict(dd2_nmp, K_sat=K),
                                         return_status=True)
        assert not st.ok, f"K_sat={K} unexpectedly inverts at fixed Q_sat"


def test_L_sym_is_free(dd2_nmp):
    """The isovector inversion is near-analytic and must converge across the
    whole literature range — this is what makes L_sym the usable scan axis."""
    from eos.dd2 import Parametrization

    for L in (30.0, 55.0, 85.0, 100.0):
        _, st = Parametrization.from_nmp(dict(dd2_nmp, L_sym=L),
                                         return_status=True)
        assert st.ok, f"L_sym={L} failed: {st.message}"
        assert st.isovector_residual < 1e-6


def test_sector_potentials_ride_in_the_sample(dd2_nmp):
    """The hyperon and Delta potentials must be scannable the same way an NMP
    is — carried in the sample dict — or a grid over them needs a separate loop
    outside the scan, which is exactly what the scan exists to avoid."""
    from eos.dd2 import SpeciesFlags
    from eos.mixed import build_parametrization

    flags = SpeciesFlags(hyperons=True, deltas=True, muons=True,
                         phi_field=True)
    sample = dict(dd2_nmp, U_Xi=-5.0, U_Delta=-80.0, x_wD=1.1)

    par, stage, msg = build_parametrization(sample, flags)
    assert stage == "ok", msg
    assert par.U_Xi == -5.0
    assert par.x_Delta_omega == 1.1
    # U_Delta enters only through the inverted scalar ratio, so the proof that
    # it was used is that x_Delta_sigma moved off its default.
    default, _, _ = build_parametrization(dict(dd2_nmp), flags)
    assert par.x_Delta_sigma != default.x_Delta_sigma

    # ...and the values reach the row, so a scan table is self-describing.
    row = scan_point(sample, {"B4": 180.0}, flags, GRID)
    assert (row["U_Xi"], row["U_Delta"], row["x_wD"]) == (-5.0, -80.0, 1.1)
    assert row["U_Lambda"] == -30.0             # untouched default


def test_sector_columns_are_nan_when_the_flag_is_off(dd2_nmp):
    """A U_Delta column on a run with no Deltas would claim a value the model
    never used."""
    row = scan_point(dd2_nmp, {"B4": 180.0}, FLAGS, GRID)   # no hyperons/Deltas
    assert np.isnan(row["U_Delta"]) and np.isnan(row["U_Lambda"])


def test_grid_samples_is_a_product():
    out = grid_samples(B4=[150.0, 180.0], a=[0.0, 0.2, 0.4])
    assert len(out) == 6
    assert {(d["B4"], d["a"]) for d in out} == {
        (b, a) for b in (150.0, 180.0) for a in (0.0, 0.2, 0.4)}
