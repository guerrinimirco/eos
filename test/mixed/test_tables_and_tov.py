"""
The stitched core equation of state and its TOV integration.

  - build_mixed_eos_table joins pure hadronic, eta-mixed and pure quark
    segments into one monotone (P, eps, n_B) table with no interior holes; chi
    spans 0 to 1 and the pure wings are cut at the transition boundaries, so
    the pressure never doubles back;
  - eta=1 (Maxwell) leaves a constant-pressure plateau that eos/tov detects on
    its own, applying the Takatsy-Kovacs tidal correction at the density
    discontinuity; eta=0 (Gibbs) has no plateau — pressure rises through the
    window — so there is nothing to correct;
  - eos/tov integrates the table to a physical maximum mass and radii, and the
    construction softens the equation of state smoothly;
  - the tidal deformability Lambda(M) comes out without any special flag.
"""
import numpy as np
import pytest

from eos.dd2 import Parametrization, SpeciesFlags
from eos.mixed import beta_eq_neutrinoless
from eos.mixed.tables.core_eos import build_mixed_eos_table, mass_radius_mixed
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
    return {eta: build_mixed_eos_table(par, flags, grid, eta, beta_eq_neutrinoless())
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
    assert t.n_onset < t.n_offset, "onset/offset not ordered"


def test_no_interior_holes_in_mixed_window(tables, grid):
    """Every grid point inside the converged window [onset, offset] is present
    (the numeric-Jacobian stall band is interpolation-filled)."""
    for eta in (0.0, 0.5, 1.0):
        t = tables[eta]
        present = set(np.round(t.n_B, 3))
        interior = [g for g in grid if t.n_onset < g < t.n_offset]
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
    return {eta: mass_radius_mixed(par, flags, grid, eta, beta_eq_neutrinoless(),
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
