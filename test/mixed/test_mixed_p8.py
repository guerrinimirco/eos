"""
P8 gate: full EoS table across the transition + TOV M(R) / tidal
(docs/phase2/SPECIFICATION_AND_PLAN.md §4 milestone P8).

  - build_mixed_eos_table stitches pure-hadronic + eta-mixed + pure-quark into
    one monotone (P, eps, n_B) core EoS with no interior holes; chi spans 0..1
    and the pure wings are cut at the mixed pressure range (no P double-back);
  - eta=1 (Maxwell) leaves a constant-P plateau that eos/tov detects
    (_detect_maxwell_construction) -> the Takatsy-Kovacs tidal Delta-Y jump is
    applied automatically at the density discontinuity; eta=0 (Gibbs) has no
    plateau (P rises through the window) so there is nothing to correct;
  - eos/tov integrates the table: M_max is physical (~2 Msun), radii physical,
    and the construction softens the EoS smoothly (Gibbs M_max <= Maxwell);
  - tidal Lambda(M) is produced -- the discontinuity correction needs no flag
    on our side, it rides on the plateau the table already carries.

Nucleons + electrons, default vMIT (B4=180): a mixed window ~0.44-1.0 fm^-3.
TOV uses the fast numba backend for test speed (both backends apply the jump).
"""
import numpy as np
import pytest

from eos.dd2 import Parametrization, SpeciesFlags
from eos.mixed import mode_A
from eos.mixed.table import build_mixed_eos_table, mass_radius_mixed
from eos.tov.solver import _detect_maxwell_construction


@pytest.fixture(scope="module")
def par():
    return Parametrization.from_dd2_defaults()


@pytest.fixture(scope="module")
def flags():
    return SpeciesFlags(hyperons=False, muons=False)


@pytest.fixture(scope="module")
def grid():
    return np.round(np.arange(0.08, 1.45, 0.03), 3)


@pytest.fixture(scope="module")
def tables(par, flags, grid):
    return {eta: build_mixed_eos_table(par, flags, grid, eta, mode_A())
            for eta in (0.0, 0.5, 1.0)}


# ---------------------------------------------------------------- table shape
@pytest.mark.parametrize("eta", [0.0, 0.5, 1.0])
def test_table_monotone_and_spans_transition(tables, eta):
    """One monotone core EoS: P non-decreasing, chi runs 0 -> 1 with a genuine
    mixed segment in between."""
    t = tables[eta]
    assert np.all(np.diff(t.P) > -1e-6), "P doubles back (wing not cut at plateau)"
    assert np.all(np.diff(t.eps) > -1e-6), "eps not monotone"
    assert t.chi.min() < 0.05 and t.chi.max() > 0.95, "chi does not span 0..1"
    assert (t.phase == "mix").sum() >= 3, "no resolved mixed window"
    assert t.onset < t.offset, "onset/offset not ordered"


def test_no_interior_holes_in_mixed_window(tables, grid):
    """Every grid point inside the converged window [onset, offset] is present
    (the numeric-Jacobian stall band is interpolation-filled)."""
    for eta in (0.0, 0.5, 1.0):
        t = tables[eta]
        present = set(np.round(t.n_B, 3))
        interior = [g for g in grid if t.onset < g < t.offset]
        missing = [g for g in interior if g not in present]
        assert not missing, f"eta={eta}: holes inside window at {missing}"


# --------------------------------------------------- Maxwell plateau / tidal
def test_maxwell_plateau_detected_only_at_eta1(tables):
    """The tidal Delta-Y discontinuity correction is triggered by a detectable
    constant-P plateau: eta=1 has one (Maxwell), eta=0 does not (Gibbs)."""
    assert _detect_maxwell_construction(tables[1.0].to_tov()) is not None
    assert _detect_maxwell_construction(tables[0.0].to_tov()) is None
    assert np.isfinite(tables[1.0].P_trans)


# ------------------------------------------------------------------- TOV M(R)
@pytest.fixture(scope="module")
def mr(par, flags, grid):
    return {eta: mass_radius_mixed(par, flags, grid, eta, mode_A(),
                                   n_ec=40, backend="fast")
            for eta in (0.0, 1.0)}


@pytest.mark.parametrize("eta", [0.0, 1.0])
def test_tov_integrates_physical(mr, eta):
    """M(R) integrates to a physical sequence with tidal Lambda."""
    m = mr[eta]
    res = m["results"]
    assert res.shape[1] == 7, "expected columns e_c,n_c,P_c,R,M,k2,Lambda"
    assert 1.9 < m["M_max"] < 2.6, f"M_max={m['M_max']} unphysical"
    assert 10.0 < m["R_Mmax"] < 15.0
    assert 11.0 < m["R_1p4"] < 15.0
    # stable branch (up to M_max) has finite, positive tidal deformability
    idx = int(np.argmax(res[:, 4]))
    Lam = res[:idx + 1, 6]
    assert np.all(np.isfinite(Lam)) and np.all(Lam > 0.0)


def test_construction_softens_smoothly(mr):
    """Gibbs (eta=0) softens the EoS more than Maxwell (eta=1): lower M_max, but
    smoothly (the two constructions are the endpoints of one continuous eta)."""
    assert mr[0.0]["M_max"] <= mr[1.0]["M_max"] + 0.02
    assert mr[1.0]["M_max"] - mr[0.0]["M_max"] < 0.5   # not a wild jump


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
