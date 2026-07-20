"""
Fast-path speed regression guard: the analytic-Jacobian β-eq sweep must stay
under a generous per-point ceiling after Numba warmup, so future regressions in
solver conditioning are caught. Short grid so it doesn't bloat CI.
"""
import time

import numpy as np
import pytest

from eos.dd2 import Parametrization, SpeciesFlags, sweep_beta_eq_octet

#: Generous per-point ceiling [ms] (anchor ~0.35 ms/pt on the dev machine).
CEILING_MS_PER_PT = 2.0


@pytest.mark.slow
def test_fast_path_under_ceiling():
    par = Parametrization.from_dd2_defaults()
    flags = SpeciesFlags(hyperons=False, phi_field=False)
    grid = np.geomspace(0.06, 1.2, 60)

    # discard the first call (Numba compile)
    sweep_beta_eq_octet(par, grid[:5], flags, include_photons=False,
                        analytic_jac=True)
    t0 = time.perf_counter()
    pts = sweep_beta_eq_octet(par, grid, flags, include_photons=False,
                              analytic_jac=True)
    ms_per_pt = 1e3 * (time.perf_counter() - t0) / len(grid)

    assert len(pts) == len(grid)
    assert ms_per_pt < CEILING_MS_PER_PT, \
        f"fast-path sweep {ms_per_pt:.2f} ms/pt exceeds {CEILING_MS_PER_PT} ms/pt"
