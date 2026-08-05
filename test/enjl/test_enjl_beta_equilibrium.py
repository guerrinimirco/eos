"""The beta-equilibrium solver reproduces the reference tables' composition.

`test_enjl_fixed_composition.py` hands the solver each row's own densities.
Here it is given only n_b and has to produce the composition itself, from
Eqs. (23)-(24) of Xia 2024 (PRD 110, 014022) solved simultaneously with the
mean field. That is a much stronger statement: the onset density of every
species, and the way the composition rearranges through each threshold, is a
prediction rather than an input.

**What is gated and what is not.** The model has more than one solution branch
above each of its first-order transitions -- chirally broken, chirally restored
but confined, and fully deconfined -- and all of them satisfy the local
equations over a finite density range. A continuation follows the branch it
started on and keeps following it past the transition, into the metastable
region beyond, which is exactly what `beta_eos_table` documents itself as
doing. Choosing between the branches needs a Maxwell construction and is an
open question in this implementation, so it is not gated here.

What *is* gated is everything below each table's first transition, and there
the agreement is not marginal: the chemical potential matches to 0.1 MeV or
better on all five parameter sets, and every species density present above 1%
of n_b to about 1e-3 relative. Both are two to three orders of magnitude
inside the 0.5 MeV and 1% that would be needed to call the composition
reproduced. The upper limits below were not chosen; each one is where the
solver and the table part company, and each coincides with that file's own
recorded first-order transition to the grid spacing.
"""
import numpy as np
import pytest

from eos.enjl.eos_beta import beta_eos_table, solve_beta_point

from enjl_cases import REF_COL, case
from reference import PARAMETER_SETS, baryon_potential

#: Highest density [fm^-3] at which the upward branch still coincides with the
#: table, per file, and the residuals reached over that range. Every entry sits
#: at the low edge of that parameter set's first first-order transition, as
#: tabulated from the off-grid coexistence rows:
#:
#:   fq1.0_B0  0.6438  vs the recorded chiral coexistence edge 0.643752
#:   fq1.0_B1  0.63    vs 0.637711
#:   fq0.7_B0  0.59    the unstable dP/dn_b < 0 step this file retains
#:   fq0.7_B1  0.4486  vs 0.448564
#:   fq0.5_B1  1.24    vs 1.248480, the *second* transition -- this set's
#:                     first one, at 0.322-0.459, is crossed correctly
#:
#: Tolerances are the worst residual observed plus 5%, rounded up.
BRANCH_LIMIT = {
    "Beta_fq1.0_B0.dat": dict(n_max=0.6438, mu=7.0e-2, dens=1.5e-3),
    "Beta_fq1.0_B1.dat": dict(n_max=0.6300, mu=1.1e-1, dens=1.1e-3),
    "Beta_fq0.7_B0.dat": dict(n_max=0.5900, mu=1.9e-4, dens=1.4e-5),
    "Beta_fq0.7_B1.dat": dict(n_max=0.4486, mu=3.9e-2, dens=1.1e-3),
    "Beta_fq0.5_B1.dat": dict(n_max=1.2400, mu=5.3e-4, dens=1.6e-5),
}

#: a species counts for the density comparison above this fraction of n_b;
#: below it the table's own value is at the resolution of its printed digits
PRESENT = 1.0e-2

FILES = pytest.mark.parametrize("filename", sorted(PARAMETER_SETS))


@FILES
def test_reproduces_table_composition_below_transition(filename):
    """mu_b and every present species density, predicted rather than read."""
    limit = BRANCH_LIMIT[filename]
    col, ok, par, _ = case(filename)
    idx = np.flatnonzero(ok & (col["nB"] <= limit["n_max"] + 1e-9))
    assert len(idx) > 30, "too few densities to be a meaningful sweep"

    mu_table = baryon_potential(col)
    points, _, _ = beta_eos_table(col["nB"][idx], par=par)
    assert len(points) == len(idx), "a density in the sweep failed to converge"

    for pt, i in zip(points, idx):
        assert abs(pt.mu_b - mu_table[i]) <= limit["mu"], col["nB"][i]
        for name, suffix in REF_COL.items():
            n_table = col["n" + suffix][i]
            if n_table > PRESENT * col["nB"][i]:
                rel = abs(pt.densities[name] - n_table) / n_table
                assert rel <= limit["dens"], (col["nB"][i], name, rel)


@pytest.mark.parametrize("f_q,B", [(0.7, 1.0), (1.0, 0.0)])
def test_sweep_covers_the_full_density_range(f_q, B):
    """A sweep reaches every density from 0.05 to 10 fm^-3, transitions included.

    Which branch it lands on above a transition is a separate question (see the
    module docstring); that it lands somewhere at every density is what this
    checks, and it is what the box on the unknowns has to deliver. A box
    calibrated at saturation density fails here well before 10 fm^-3, because
    mu_b reaches about 16 GeV and g_omega*omega about 5 GeV, and puts the
    solution -- and at n_b = 5 the starting point itself -- outside the
    feasible region.

    The grid is uniform on purpose. The tables' own solved rows are not: the
    f_q = 0.5 file has 0.15 fm^-3 gaps where its interpolated mixed-phase rows
    have been masked away, and a continuation cannot be expected to step across
    those.
    """
    from eos.enjl.parameters import ENJLParams
    par = ENJLParams(f_q=f_q, B_GeV_fm3=B)
    grid = np.round(np.arange(0.05, 10.001, 0.05), 4)
    points, _, _ = beta_eos_table(grid, par=par)
    assert len(points) == len(grid), (
        f"{len(grid) - len(points)} of {len(grid)} densities unreached")
    n_b = np.array([p.n_b_fm for p in points])
    assert np.allclose(n_b, grid, rtol=1e-6), "a point drifted off its density"


def test_high_density_needs_a_widened_box():
    """A density where the saturation-scale bounds excluded the solution.

    At n_b = 5 fm^-3 the baryon chemical potential is about 8.4 GeV, so the
    3 GeV ceiling that suited nuclear densities made the starting point itself
    infeasible and the solve could not even begin. This pins the fix.
    """
    _, _, par, _ = case("Beta_fq0.7_B1.dat")
    pt = solve_beta_point(5.0, par=par)
    assert pt.mu_b > 3000.0
    assert pt.P > 0.0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
