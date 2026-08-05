"""A grid node that is one ULP outside the table must not blank the query.

Guards `_EdgeSnapped` in eos/sfho/compute_tables.py: writing an axis to disk
with '%e' rounds it, so the in-memory node can land just past the on-disk edge
and RegularGridInterpolator would fill the whole query with NaN.
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from eos.sfho.compute_tables import _EdgeSnapped


def test_edge_snap():
    nB = np.linspace(0.1, 1.0, 5)
    YL = np.array([0.1, 0.2, 0.3, 0.4])          # as read back from the file
    f = _EdgeSnapped(
        RegularGridInterpolator((nB, YL), np.outer(nB, YL),
                                bounds_error=False, fill_value=np.nan),
        (nB, YL))

    YL_mem = np.arange(0.10, 0.401, 0.05)[-1]     # 0.40000000000000013
    assert YL_mem > YL[-1], "float drift gone -- this test no longer guards anything"

    # the ULP overshoot is snapped back onto the edge
    assert np.all(np.isfinite(f((nB, YL_mem)))), "edge node returned NaN"
    np.testing.assert_allclose(f((nB, YL_mem)), f((nB, 0.4)))

    # a real out-of-range query is still NaN, not clamped
    assert np.all(np.isnan(f((nB, 0.5))))
    assert np.all(np.isnan(f((nB, 0.05))))


if __name__ == '__main__':
    test_edge_snap()
    print("ok")
