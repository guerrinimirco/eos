"""
Rotating-star models: table conversion, solver invocation, and the physics
invariants of `eos/tov/rotating.py` and `eos/tov/rns_backend.py`.

Everything that needs the external rotating-star solver is skipped when it is
not installed, following the pattern used for external data elsewhere in this
suite.
"""

import os
import tempfile

import numpy as np
import pytest

from eos.general.physics_constants import (
    MEV_FM3_TO_DYNE_CM2,
    MEV_FM3_TO_G_CM3,
)
from eos.tov import rotating as rot
from eos.tov import rns_backend as rns
from eos.tov.solver import EOSTable_for_TOV

# Reference tables shipped with the solver, used for the golden-value checks.
_RNS_ROOT = "/Users/mircoguerrini/Desktop/Research/rns-main-official"
EOSC_PATH = os.path.join(_RNS_ROOT, "source", "eos", "eosC")

have_rns = rns.have_rns()
have_eosc = os.path.isfile(EOSC_PATH)

needs_rns = pytest.mark.skipif(not have_rns, reason="rns binary not found")
needs_eosc = pytest.mark.skipif(not have_eosc, reason="eosC table not found")

KEPLER_M = rot.KEPLER_COLUMNS.index("M")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def eosc_table():
    """The solver's own eosC, converted into this repository's units."""
    rho, p, h, n0 = np.loadtxt(EOSC_PATH, skiprows=1).T
    return EOSTable_for_TOV(P=p / MEV_FM3_TO_DYNE_CM2,
                            epsilon=rho / MEV_FM3_TO_G_CM3,
                            nB=n0 / 1.0e39)


@pytest.fixture(scope="module")
def eosc_reference():
    """Raw eosC columns, for comparing against the solver's own converter."""
    return np.loadtxt(EOSC_PATH, skiprows=1).T


@pytest.fixture(scope="module")
def dd2_core():
    """Cold beta-equilibrium DD2 core table, without a crust."""
    from eos.dd2 import SpeciesFlags
    from eos.dd2.parametrization import Parametrization
    from eos.dd2.verify.tov import build_core_table

    return build_core_table(Parametrization.from_dd2_defaults(), SpeciesFlags(),
                            n_lo=0.05, n_hi=1.2, n_points=120)


@pytest.fixture(scope="module")
def dd2_path(dd2_core):
    """DD2 core plus BPS crust, written in the solver's format."""
    return rot.prepare_rotating_eos(dd2_core)


@pytest.fixture(scope="module")
def dd2_scan(dd2_path):
    """One axis-ratio scan at a central density near the maximum mass."""
    return rot.rratio_scan(dd2_path, 800.0, n=12)


# ---------------------------------------------------------------------------
# Table conversion
# ---------------------------------------------------------------------------

@needs_eosc
def test_enthalpy_matches_reference_converter(eosc_table, eosc_reference):
    """
    The enthalpy column must reproduce the one produced by the solver's own
    converter, HnG.c, which integrates dh = dp/(e+p) with Simpson's rule on
    16003 divisions.

    This is the column with no independent check inside the solver: it inverts
    the table through both `p_at_h` and `e_at_p`, so an inconsistent enthalpy
    shows up only as a field iteration that will not settle. Integrating on the
    table's own log-spaced rows instead of a refined grid gives a 3% error here
    and a 1% error in the resulting stellar masses.
    """
    _, _, h_ref, _ = eosc_reference
    _, _, h, _ = rns.rns_columns(eosc_table)

    rel = np.abs(h[1:] / h_ref[1:] - 1.0)
    # Measured: median 2.7e-4, max 9.1e-3. The maximum sits in the outermost
    # crust, where the two schemes interpolate a 96-point table differently and
    # the enclosed mass is negligible.
    assert np.median(rel) < 1e-3
    assert rel.max() < 2e-2
    assert h[0] == 1.0        # the floor value the solver expects


@needs_eosc
def test_written_table_is_well_formed(eosc_table):
    """Every column strictly increasing, and the table reaches the surface."""
    with tempfile.TemporaryDirectory() as tmp:
        path = rns.write_rns_eos(eosc_table, os.path.join(tmp, "eos.rns"))
        with open(path) as fh:
            n_declared = int(fh.readline())
        rho, p, h, n0 = np.loadtxt(path, skiprows=1).T

    assert n_declared == rho.size <= rns.MAX_EOS_ROWS
    for name, col in (("rho", rho), ("p", p), ("h", h), ("n0", n0)):
        assert np.all(np.isfinite(col)), name
        assert np.all(col > 0.0), name
    for name, col in (("rho", rho), ("p", p), ("h", h)):
        assert np.all(np.diff(col) > 0.0), f"{name} must be strictly increasing"
    assert rho.min() <= rns.RNS_RHO_SURFACE * rns.SURFACE_TOLERANCE


def test_row_cap_is_enforced(eosc_table):
    """
    Asking for more rows than the solver's fixed-size arrays hold must raise.

    The arrays are `double[201]` and are filled without a bound check, so an
    oversized table corrupts memory instead of failing.
    """
    with pytest.raises(ValueError, match="at most 200"):
        rns.rns_columns(eosc_table, n_points=500)


def test_dense_table_is_thinned(dd2_core):
    """A table with more rows than the cap is resampled, not rejected."""
    fine = EOSTable_for_TOV(
        P=np.geomspace(dd2_core.P.min(), dd2_core.P.max(), 4000),
        epsilon=np.geomspace(dd2_core.epsilon.min(), dd2_core.epsilon.max(), 4000),
        nB=np.geomspace(dd2_core.nB.min(), dd2_core.nB.max(), 4000),
    )
    merged = rot.prepare_rotating_eos(fine, path=None)
    rho = np.loadtxt(merged, skiprows=1)[:, 0]
    assert rho.size <= rns.MAX_EOS_ROWS


def test_missing_crust_is_rejected(dd2_core):
    """
    A core-only table stops seven decades above the solver's fixed surface
    density, where no model can converge. That must be an error, not a silent
    non-convergence.
    """
    with pytest.raises(ValueError, match="surface"):
        rns.rns_columns(dd2_core)


def test_crust_is_attached_down_to_the_surface(dd2_path):
    """The prepared table reaches the surface the solver has pinned."""
    rho = np.loadtxt(dd2_path, skiprows=1)[:, 0]
    assert rho.min() <= rns.RNS_RHO_SURFACE * rns.SURFACE_TOLERANCE
    assert rho.min() < 1.0e2      # i.e. a real crust, not a truncated one


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def test_parse_output_handles_undefined_entries():
    """`---` marks a quantity the solver leaves undefined; it must become NaN."""
    text = """
    eosC,  MDIVxSDIV=65x129,  accuracy=1e-05
  ------------------------------------------
  2.00000e+00  e_c           (10^15 gr/cm^3)
  1.79249e+00  M             (M_sun)
  2.04981e+00  M_0           (M_sun)
  1.07698e+01  R_e           (km)
  0.00000e+00  Omega         (10^4 s^-1)
  1.37951e+00  Omega_p       (10^4 s^-1)
  0.00000e+00  T/W
  0.00000e+00  cJ/GM_sun^2
      ---      I             (10^45 gr cm^2)
 -2.63092e-01  Z_f
  1.00000e+00  r_p/r_e
  ------------------------------------------
"""
    res = rns.parse_rns_output(text)
    assert res["M"] == pytest.approx(1.79249)
    assert res["e_c"] == pytest.approx(2.0e15)
    assert np.isnan(res["I"])
    assert res["Z_f"] == pytest.approx(-0.263092)   # negative values parse
    assert res["Omega"] == pytest.approx(0.0)
    assert res["Omega_K"] == pytest.approx(1.37951e4)
    assert res["freq_K"] == pytest.approx(1.37951e4 / (2 * np.pi))


def test_parse_output_ignores_noise():
    """Rules and banners must not be mistaken for data."""
    assert rns.parse_rns_output("------------------\nnonsense\n") == {}


# ---------------------------------------------------------------------------
# Golden values shipped with the solver
# ---------------------------------------------------------------------------

@needs_rns
@needs_eosc
@pytest.mark.slow
@pytest.mark.parametrize("task,kwargs,expected", [
    ("model", dict(r_ratio=0.59),
     dict(M=2.13324, M_0=2.43446, R_e=13.9518, Omega_1e4=0.961702, J=2.89653)),
    ("kepler", dict(), dict(M=2.13633, M_0=2.43798)),
    ("static", dict(), dict(M=1.79249, R_e=10.7698)),
])
def test_reference_examples(task, kwargs, expected):
    """
    Reproduce the worked examples distributed with the solver (`examples.test`,
    eosC at a central density of 2e15 g/cm^3).

    Same binary and same table, so this is a check on invocation and parsing
    and must match to the digits the solver prints.
    """
    res = rns.run_rns(EOSC_PATH, task, 2e15, **kwargs)
    assert res is not None, f"{task} did not converge"
    for key, want in expected.items():
        assert res[key] == pytest.approx(want, rel=1e-5), key


@needs_rns
@needs_eosc
@pytest.mark.slow
def test_writer_roundtrip_reproduces_reference(eosc_table):
    """
    Converting eosC out of the solver's format and back must return the same
    star. This is the end-to-end check on the unit conversions and the enthalpy
    integration together.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = rns.write_rns_eos(eosc_table, os.path.join(tmp, "eos.rns"))
        ours = rns.run_rns(path, "model", 2e15, r_ratio=0.59)
    theirs = rns.run_rns(EOSC_PATH, "model", 2e15, r_ratio=0.59)

    assert ours is not None and theirs is not None
    # Measured: 6e-5 in M, 4e-4 in R_e. The residual is the enthalpy column,
    # which the reference converter integrates with a different quadrature.
    assert ours["M"] == pytest.approx(theirs["M"], rel=1e-3)
    assert ours["R_e"] == pytest.approx(theirs["R_e"], rel=2e-3)


@needs_rns
@needs_eosc
@pytest.mark.slow
def test_long_eos_path_still_converges(eosc_table):
    """
    The solver reads the table path into a `char[80]` with `sscanf("%s")`, so a
    longer path overruns it and silently corrupts the globals that follow. A
    temporary directory alone exceeds this on macOS. The runner must therefore
    be immune to how long the caller's path is.
    """
    with tempfile.TemporaryDirectory() as tmp:
        deep = os.path.join(tmp, "d" * 60, "e" * 60)
        os.makedirs(deep)
        path = rns.write_rns_eos(eosc_table, os.path.join(deep, "a" * 40 + ".rns"))
        assert len(path) > rns.MAX_PATH_CHARS, "test needs an over-long path"
        res = rns.run_rns(path, "model", 2e15, r_ratio=0.59)

    assert res is not None, "over-long EOS path broke the run"
    assert res["M"] == pytest.approx(2.13324, rel=1e-3)


# ---------------------------------------------------------------------------
# Physics invariants
# ---------------------------------------------------------------------------

@needs_rns
@pytest.mark.slow
def test_kepler_model_is_at_mass_shedding(dd2_path):
    """
    At the mass-shedding limit the fluid angular velocity equals the orbital
    angular velocity of a particle at the equator. The solver reports both
    independently, so their agreement is a free check that the Keplerian search
    actually reached the limit.
    """
    kep = rot.kepler_model(dd2_path, 800.0)
    assert kep.converged, kep.note
    assert abs(kep.Omega / kep.Omega_K - 1.0) < 1e-3    # measured 8e-5
    assert 800.0 < kep.freq < 2000.0                    # nucleonic EOS
    assert 0.5 < kep.r_ratio < 0.7                      # oblate, not spherical


@needs_rns
@pytest.mark.slow
def test_scan_is_monotone_and_converges(dd2_scan):
    """
    Every fixed-axis-ratio model must converge, and Omega, M, M_0 and J must
    all increase as the star is spun up. The inversion used for every physical
    target assumes exactly this.
    """
    assert all(s.converged for s in dd2_scan), \
        [s.note for s in dd2_scan if not s.converged]

    r = np.array([s.r_ratio for s in dd2_scan])
    assert np.all(np.diff(r) > 0), "scan must be ordered by axis ratio"
    for name in ("Omega", "M", "M_0", "J"):
        values = np.array([getattr(s, name) for s in dd2_scan])
        assert np.all(np.diff(values) <= 1e-12), f"{name} is not monotone"

    assert dd2_scan[-1].r_ratio == pytest.approx(1.0)
    assert dd2_scan[-1].Omega == pytest.approx(0.0)     # non-rotating endpoint


@needs_rns
@pytest.mark.slow
def test_rotation_raises_the_maximum_mass_by_about_20_percent(dd2_core):
    """
    Uniform rotation supports about 20% more mass than the non-rotating limit,
    largely independently of the equation of state (Breu & Rezzolla 2016, MNRAS
    459, 646, find 1.203 +/- 0.022).
    """
    grid = np.geomspace(400.0, 1500.0, 8)
    kepler = rot.kepler_sequence(dd2_core, grid)
    static = rot.static_cross_check(dd2_core, grid)

    m_max_rot = np.nanmax(kepler[:, KEPLER_M])
    m_max_static = np.nanmax(static[:, 3])
    assert m_max_rot / m_max_static == pytest.approx(1.20, abs=0.06)


@needs_rns
@pytest.mark.slow
def test_static_limit_agrees_with_the_tov_solver(dd2_core):
    """
    At unit axis ratio the rotating code must reproduce this repository's TOV
    solver on the same equation of state.

    The two do not agree exactly, and should not be expected to: the rotating
    code pins its own surface at 7.8 g/cm^3 and works from a table thinned to
    200 rows, while `compute_tov_sequence` integrates the supplied table
    directly with an adaptive scheme. Measured on DD2 over 300-1400 MeV/fm^3:
    at most 2.5e-3 in mass and 1.4e-3 in radius, median 2.0e-4 and 1.0e-3. The
    bounds below are those measured values with room for solver noise; they are
    not a target to be loosened if a change makes them fail.
    """
    grid = np.geomspace(300.0, 1400.0, 8)
    out = rot.static_cross_check(dd2_core, grid)

    assert np.all(np.isfinite(out[:, 1:5])), "a static model failed"
    assert np.nanmax(out[:, 5]) < 5e-3, "mass disagreement"
    assert np.nanmax(out[:, 6]) < 3e-3, "radius disagreement"

    # And the sequence itself must still reproduce the published DD2 numbers.
    assert 2.35 < np.nanmax(out[:, 1]) < 2.50          # M_max [M_sun]


# ---------------------------------------------------------------------------
# Inversion
# ---------------------------------------------------------------------------

@needs_rns
@pytest.mark.slow
@pytest.mark.parametrize("quantity", ["freq", "J", "M", "M_0"])
def test_inversion_recovers_a_known_scan_point(dd2_core, dd2_scan, quantity):
    """
    Take a converged model out of the scan, ask for the target it realises, and
    the same axis ratio must come back. This exercises the whole path: monotone
    interpolation, bracketing, root find, and the confirming run.
    """
    reference = dd2_scan[len(dd2_scan) // 2]
    target = getattr(reference, quantity)

    got = rot.rotating_model(dd2_core, 800.0, scan=dd2_scan,
                             **{quantity: target})
    assert got.converged, got.note
    assert got.r_ratio == pytest.approx(reference.r_ratio, abs=1e-3)
    assert got.M == pytest.approx(reference.M, rel=1e-4)


@needs_rns
@pytest.mark.slow
def test_targets_beyond_the_kepler_limit_are_reported_not_invented(dd2_core,
                                                                   dd2_scan):
    """
    No uniformly rotating star exists above the mass-shedding frequency. The
    request must come back unconverged and explained, rather than as an
    extrapolated model or an exception that would abort a sweep.
    """
    got = rot.rotating_model(dd2_core, 800.0, freq=1e4, scan=dd2_scan)
    assert not got.converged
    assert "Keplerian" in got.note
    assert np.isnan(got.M)


@needs_rns
@pytest.mark.slow
def test_grid_stays_rectangular_across_the_kepler_limit(dd2_core):
    """
    A constant-frequency grid that overhangs the mass-shedding limit at low
    central density must return NaN for the points that do not exist, keeping
    the array shape predictable for plotting.
    """
    e_c = np.geomspace(400.0, 1200.0, 4)
    freqs = [0.0, 600.0, 1200.0, 1600.0]
    grid = rot.rotating_grid(dd2_core, e_c, freq_grid=freqs, n_scan=10)

    assert grid.shape == (len(e_c) * len(freqs), len(rot.GRID_COLUMNS))
    freq_out = grid[:, rot.GRID_COLUMNS.index("freq")]
    requested = np.tile(freqs, len(e_c))

    resolved = np.isfinite(freq_out)
    assert resolved.any() and not resolved.all(), \
        "grid should straddle the Kepler limit for this test to mean anything"
    assert np.allclose(freq_out[resolved], requested[resolved], atol=1e-4)
    # The lowest central density cannot reach the highest frequency.
    assert not resolved[3]


def test_rotating_model_requires_exactly_one_target(dd2_core):
    """Two targets, or none, is a caller error and must say so."""
    with pytest.raises(ValueError, match="exactly one"):
        rot.rotating_model(dd2_core, 800.0, freq=500.0, J=1.0)
    with pytest.raises(ValueError, match="exactly one"):
        rot.rotating_model(dd2_core, 800.0)


def test_unknown_backend_is_refused(dd2_core):
    """Only the wired-up backend may be requested."""
    with pytest.raises(NotImplementedError, match="lorene"):
        rot.rotating_model(dd2_core, 800.0, kepler=True, backend="lorene")


def test_unstable_solver_tasks_are_refused():
    """
    The solver's fixed-quantity tasks wrap the field iteration in a second
    outer loop and are the ones that lose convergence. They must not be
    reachable by accident.
    """
    for task in ("omega", "jmoment", "gmass", "rmass"):
        with pytest.raises(ValueError, match="deliberately unsupported"):
            rns.run_rns("unused", task, 1e15)


# ---------------------------------------------------------------------------
# Turning-point stability
# ---------------------------------------------------------------------------

def test_turning_point_finds_the_mass_maximum():
    """
    On a smooth sequence with one maximum, the turning point is that maximum
    and the stable set is everything up to it. The grid deliberately misses the
    peak, so the interpolated location must beat the grid spacing.
    """
    x = np.linspace(0.4, 1.6, 25)
    M = 2.0 - 3.0 * (x - 0.93) ** 2

    stable, x_crit, M_crit = rot.turning_point(x, M)

    assert x_crit == pytest.approx(0.93, abs=0.005)
    assert M_crit == pytest.approx(2.0, abs=1e-3)
    assert np.array_equal(stable, x <= x_crit)


def test_turning_point_takes_the_first_maximum_not_the_largest():
    """
    A first-order transition can give a second, higher peak past the dip. The
    first branch still destabilises at its own turning point, so the larger
    maximum must not be the one reported.
    """
    x = np.linspace(0.4, 1.6, 61)
    first, second = 0.7, 1.3
    M = (1.6 + 0.4 * np.exp(-((x - first) / 0.18) ** 2)
         + 0.5 * np.exp(-((x - second) / 0.18) ** 2))

    assert M[np.argmin(abs(x - second))] > M[np.argmin(abs(x - first))]

    stable, x_crit, M_crit = rot.turning_point(x, M)

    assert x_crit == pytest.approx(first, abs=0.01)
    assert not stable[x > first + 0.05].any()


def test_a_sequence_that_never_turns_over_is_all_stable():
    """
    A grid that stops before the maximum has no turning point on it. That must
    be said with NaN rather than reported as a maximum at the last point, which
    would silently truncate a sequence that needs extending instead.
    """
    x = np.linspace(0.4, 0.9, 20)
    M = 2.0 - 3.0 * (x - 1.3) ** 2            # still rising at the last point

    stable, x_crit, M_crit = rot.turning_point(x, M)

    assert stable.all()
    assert np.isnan(x_crit) and np.isnan(M_crit)


def test_noise_below_the_precision_is_not_a_turning_point():
    """
    Solver noise on a flat stretch produces local maxima that are not physical.
    Only a drop larger than `precision` counts.
    """
    x = np.linspace(0.4, 1.0, 31)
    M = 1.9 + 1e-4 * np.sin(9.0 * x)          # wiggles, no real turnover

    stable, x_crit, _ = rot.turning_point(x, M, precision=1e-3)

    assert stable.all() and np.isnan(x_crit)


def test_failed_models_are_skipped_and_never_marked_stable():
    """
    A sweep keeps non-converged points as NaN. They must not break the search,
    and must not come back flagged as stable models.
    """
    x = np.linspace(0.4, 1.6, 25)
    M = 2.0 - 3.0 * (x - 0.93) ** 2
    M[3] = np.nan                              # a failed model below the peak
    M[-2] = np.nan                             # and one above it

    stable, x_crit, _ = rot.turning_point(x, M)

    assert x_crit == pytest.approx(0.93, abs=0.005)
    assert not stable[3] and not stable[-2]
    assert stable[2] and stable[4]
