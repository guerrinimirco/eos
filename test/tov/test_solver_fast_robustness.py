"""
The TOV layer must not build unphysical tables, and must not read out of bounds
when handed one.

Two defects met here. `_attach_crust` spliced a crust and a core on density
alone, so the merged table stepped DOWN in pressure at the join — the crust and
the core are independent calculations and do not agree on P at the same n_B.
And the Numba backend turned the resulting non-monotone eps(P) into confident
nonsense rather than into a failure.

These are unit tests on `eos.tov.solver_fast` internals rather than on a
mass-radius result, deliberately. The failures they guard against were all
silent: the backend returned finite, confident, wrong numbers — maximum masses
of order 1e200 M_sun that changed from run to run — for equations of state that
the scipy reference integrated correctly. A parity test comparing the two
backends did exist, but only at eta = 0 and eta = 1 and only without a crust,
which is precisely the corner where all three defects hide.
"""
import numpy as np
import pytest

from eos.tov.solver_fast import prepare_eos_uniform, _interp_uniform


def _polytrope(n_lo=1e-8, n_hi=1.5, n=400, K=100.0, gamma=2.0):
    """A smooth, strictly monotone EOS with no phase transition anywhere.

    The density range spans ~8 decades on purpose: attaching a crust widens an
    EOS table's pressure span from ~4 decades to ~12, and the bug this guards
    against was a discontinuity criterion whose meaning depended on that span.
    """
    nB = np.geomspace(n_lo, n_hi, n)
    eps = 939.0 * nB + K * nB ** gamma
    P = K * (gamma - 1.0) * nB ** gamma
    return P, eps, nB


@pytest.mark.slow
def test_attached_crust_leaves_pressure_monotone():
    """The join must not step down in P.

    The crust and the core are independent calculations and disagree on P at
    the same n_B — for this parametrization BPS gives 0.406 MeV/fm^3 at
    n_B = 0.080 fm^-3 where the core gives 0.225 — so splitting on density
    alone inverts P at the join. That is not a blemish: eps(P) becomes
    double-valued, the implied cs^2 goes negative, and a TOV integration
    crossing it diverges.

    Whether it bites depends on the parametrization, which is why it went
    unnoticed — the DD2 defaults happen to join cleanly. The soft, Delta-rich
    sample used here does not, and it is the kind of sample the parameter scans
    generate by the hundred.
    """
    import os
    from eos.tov.solver import add_crust, CRUST_PATHS
    from eos.dd2.verify.tov import N_TRANSITION

    if not os.path.isfile(CRUST_PATHS.get("BPS", "")):
        pytest.skip("BPS crust table not available on this machine")

    core = _soft_hybrid_core(eta=0.3)
    merged = add_crust(core.to_tov(), "BPS", mode="attach",
                       n_transition=N_TRANSITION)
    dP = np.diff(merged.P)
    assert np.all(dP >= 0), (
        f"{int((dP < 0).sum())} pressure inversion(s) at the crust join, "
        f"worst dP = {dP.min():.3e} MeV/fm^3")
    assert np.all(np.diff(merged.nB) > 0), "n_B not increasing across the join"


def _soft_hybrid_core(eta):
    """A stitched DD2+vMIT core whose low-density branch undercuts the BPS
    crust. Soft (L_sym = 30) and Delta-rich, which is what makes the join bite;
    the DD2 defaults happen to join cleanly and hide the problem."""
    from eos.dd2 import Parametrization, SpeciesFlags, compute_nmp
    from eos.vmit.parameters import get_vmit_custom
    from eos.mixed import (beta_eq_neutrinoless, build_mixed_eos_table,
                           build_parametrization)

    flags = SpeciesFlags(hyperons=True, deltas=True, muons=True,
                         phi_field=True, photons=True)
    sample = dict(compute_nmp(Parametrization.from_dd2_defaults()),
                  K_sat=250.0, Q_sat=100.0, L_sym=30.0, U_Lambda=-30.0,
                  U_Sigma=30.0, U_Xi=-18.0, U_Delta=-50.0, x_wD=1.2, x_rD=1.0)
    par, status, msg = build_parametrization(sample, flags)
    assert status == "ok", msg
    grid = np.linspace(0.1 * par.n_sat, 12.0 * par.n_sat, 300)
    core = build_mixed_eos_table(par, flags, grid, eta, beta_eq_neutrinoless(),
                                 vmit_params=get_vmit_custom(B4=180.0, a=0.15,
                                                             m_s=150.0), T=0.0)
    assert core.has_transition, "no transition: nothing to test"
    return core


def test_a_pressure_inversion_gives_nan_not_a_confident_wrong_answer():
    """The failure mode that started all of this, in miniature.

    `solver._attach_crust` concatenates a crust and a core at a fixed density
    without reconciling their pressures, so the merged table can step DOWN in P
    at the join. Sorting by P then makes eps(P) non-monotone, the integrator
    diverges there, and the old kernel turned that into finite nonsense —
    maximum masses around 1e200 M_sun, differing from run to run.

    A star the backend cannot integrate must come back NaN. Being wrong is
    allowed to look like failure; it is not allowed to look like an answer.
    """
    from eos.tov.solver import EOSTable_for_TOV
    from eos.tov.solver_fast import compute_tov_sequence_fast

    P, eps, nB = _polytrope(n_lo=1e-6)
    i = 120
    P = P.copy()
    P[:i] *= 2.5                       # the join: crust side sits too high

    res = compute_tov_sequence_fast(
        EOSTable_for_TOV(P=P, epsilon=eps, nB=nB),
        np.geomspace(200.0, 2000.0, 40), parallel=False)
    M, R = res[:, 4], res[:, 3]
    finite = np.isfinite(M)
    assert np.all((M[finite] > 0.0) & (M[finite] < 10.0)), (
        f"absurd masses from a non-monotone table: max M = {np.nanmax(M):.3e}")
    assert np.all((R[finite] > 1.0) & (R[finite] < 200.0)), (
        f"absurd radii: max R = {np.nanmax(R):.3e}")


def test_maxwell_plateau_is_found_with_a_positive_jump():
    """A genuine constant-P plateau must be detected, with Delta eps > 0."""
    P, eps, nB = _polytrope()
    i = 250
    P = P.copy(); eps = eps.copy()
    P[i:i + 20] = P[i]                       # constant-P plateau
    eps[i:i + 20] = np.linspace(eps[i], eps[i] * 1.4, 20)
    P[i + 20:] += P[i] - P[i + 20] + 1e-6    # keep the table monotone after it

    *_, P_disc, deps_disc, _, _ = prepare_eos_uniform(P, eps, nB)
    assert np.count_nonzero(deps_disc) == 1, "the plateau was missed"
    assert deps_disc[0] > 0.0, "Delta eps came out non-positive"
    assert deps_disc[0] == pytest.approx(eps[i + 19] - eps[i], rel=1e-6)


def test_plateau_survives_arbitrary_eps_order_within_it():
    """Equal pressures carry no ordering information.

    A Maxwell plateau's pressures agree to ~1e-14 relative, so a plain
    argsort(P) leaves its points in arbitrary epsilon order. Taking the
    plateau's endpoint difference then yields a NEGATIVE Delta eps and the jump
    is silently dropped. Shuffling the plateau here reproduces that exactly.
    """
    P, eps, nB = _polytrope()
    i = 250
    P = P.copy(); eps = eps.copy(); nB = nB.copy()
    P[i:i + 20] = P[i]
    eps[i:i + 20] = np.linspace(eps[i], eps[i] * 1.4, 20)
    P[i + 20:] += P[i] - P[i + 20] + 1e-6
    order = np.arange(len(P))
    rng = np.random.default_rng(0)
    order[i:i + 20] = rng.permutation(order[i:i + 20])

    *_, deps_disc, _, _ = prepare_eos_uniform(P[order], eps[order], nB[order])
    assert np.count_nonzero(deps_disc) == 1, "shuffling the plateau hid it"
    assert deps_disc[0] > 0.0


@pytest.mark.slow
@pytest.mark.parametrize("eta", [0.3, 0.6])
def test_crusted_hybrid_star_agrees_with_the_reference(eta):
    """The real reproducer, at the intermediate eta the parity test never ran.

    `test_tov_backend_parity` compares the backends only at eta = 0 and eta = 1
    and only with `crust='No'`. Both blind spots were needed to hide this: with
    a BPS crust attached and 0 < eta < 1 the fast backend returned maximum
    masses of order 1e200 M_sun — silently, and differing between runs of the
    same input, because a diverged step was indexing the EOS lookup table out
    of bounds. The scipy reference integrates the identical table correctly, so
    a disagreement here is the fast path being wrong.
    """
    from eos.dd2 import Parametrization, SpeciesFlags, compute_nmp
    from eos.vmit.parameters import get_vmit_custom
    from eos.mixed import (beta_eq_neutrinoless, build_parametrization,
                           mass_radius_mixed)

    flags = SpeciesFlags(hyperons=True, deltas=True, muons=True,
                         phi_field=True, photons=True)
    sample = dict(compute_nmp(Parametrization.from_dd2_defaults()),
                  K_sat=250.0, Q_sat=100.0, L_sym=30.0, U_Lambda=-30.0,
                  U_Sigma=30.0, U_Xi=-18.0, U_Delta=-50.0, x_wD=1.2, x_rD=1.0)
    par = build_parametrization(sample, flags)[0]
    vmit = get_vmit_custom(B4=180.0, a=0.15, m_s=150.0)
    grid = np.linspace(0.1 * par.n_sat, 12.0 * par.n_sat, 300)
    spec = beta_eq_neutrinoless()
    core = _soft_hybrid_core(eta)

    kw = dict(vmit_params=vmit, T=0.0, table=core, n_ec=100,
              compute_tidal=False, tov_parallel=False)
    fast = mass_radius_mixed(par, flags, grid, eta, spec, backend="fast", **kw)
    ref = mass_radius_mixed(par, flags, grid, eta, spec, backend="scipy", **kw)
    assert fast["M_max"] == pytest.approx(ref["M_max"], abs=6e-3), (
        f"eta={eta}: fast {fast['M_max']:.4g} vs scipy {ref['M_max']:.4g}")


@pytest.mark.parametrize("bad", [np.nan, -np.inf, np.inf, -1e300, 1e300])
def test_interpolator_stays_in_bounds_for_any_input(bad):
    """`int(NaN)` is an arbitrary integer and the kernel has no bounds checking.

    A diverged step feeding NaN into the lookup used to index out of bounds and
    return whatever that memory held — which is where the 1e200 M_sun masses
    came from, and why they were not reproducible. Every input must land inside
    the tabulated range.
    """
    grid = np.linspace(1.0, 2.0, 64)
    got = _interp_uniform(bad, 0.0, 10.0, grid)
    assert grid[0] <= got <= grid[-1], f"out-of-range lookup: {got}"
