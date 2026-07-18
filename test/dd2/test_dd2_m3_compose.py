"""
M3 gate, CompOSE part: DD2 finite-T slices vs HS(DD2) (CompOSE id 18).

Engine in the "physical" nucleon-mass convention (HS(DD2) uses m_n != m_p;
the golden-point "average" convention costs up to ~4e-3 on P in asymmetric
matter). Compared in the uniform-matter region: the density cut rises with
T because HS light clusters survive to higher density when hot.

Gate: < 1e-3 relative on P, eps, s (report §3.x). Skipped when the CompOSE
tables are not present.
"""
import os

import pytest

from eos.dd2 import Parametrization
from eos.dd2.verify.compose import DD2_COMPOSE, compare_slice

pytestmark = pytest.mark.skipif(
    not os.path.isfile(os.path.join(DD2_COMPOSE, "eos.thermo")),
    reason="HS(DD2) CompOSE table not downloaded")

# (T [MeV], Y_q, nB_min [fm^-3]) — nB_min above the cluster region
SLICES = [
    (1.0, 0.5, 0.14),
    (4.786, 0.5, 0.14),
    (4.786, 0.1, 0.14),
    (20.893, 0.3, 0.20),
    (20.893, 0.5, 0.20),
    (47.863, 0.1, 0.22),
    (47.863, 0.5, 0.22),
    (100.0, 0.4, 0.25),
]

GATE = 1e-3


@pytest.fixture(scope="module")
def par():
    return Parametrization.from_dd2_defaults()


@pytest.mark.parametrize("T,Y_q,nB_min", SLICES)
def test_compose_slice(par, T, Y_q, nB_min):
    r = compare_slice(par, DD2_COMPOSE, T=T, Y_q=Y_q,
                      nB_min=nB_min, nB_max=0.6)
    assert r["n_points"] >= 10
    assert r["max_err_P"] < GATE
    assert r["max_err_eps"] < GATE
    assert r["max_err_s"] < GATE
    assert r["max_err_muB"] < GATE
