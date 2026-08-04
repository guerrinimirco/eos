"""
Are the reported phase boundaries where chi actually crosses 0 and 1?

`locate_window` returns two numbers that go straight onto a coexistence diagram,
so a boundary that is merely *plausible* is worse than one that is missing: it
draws a smooth curve with a kink in it and nothing says the kink is an artefact.

Both failure modes pinned here produced exactly that, on a hyperon + Delta model
where the probe steps are large enough to make the mixed solve miss:

* the coarse probe scan dropped every point above its first failure, so the
  chi >= 1 side was never seen, and the offset fell through to whichever probe
  happened to converge last;
* the bisection accepted the midpoint of its starting bracket whenever a
  midpoint probe failed, reporting a boundary up to half a probe spacing wrong
  as though it had converged.

Neither raised, and both moved with temperature, which is what put visible
zig-zags in the (n_B, T) coexistence curves.
"""
import numpy as np
import pytest

from eos.dd2 import Parametrization, SpeciesFlags
from eos.mixed import beta_eq_neutrinoless, locate_window, sweep_mixed
from eos.vmit.parameters import get_vmit_custom

# The configuration that exposed both bugs: hyperons and Deltas make the
# hadronic branch soft enough that the mixed solve misses across a coarse step.
FLAGS = SpeciesFlags(hyperons=True, deltas=True, muons=True, phi_field=True,
                     photons=True)
GRID = np.linspace(0.0149077, 1.788924, 300)
TOL = 0.5 * float(np.min(np.diff(GRID)))


@pytest.fixture(scope="module")
def par():
    return Parametrization.from_delta_potential(
        U_Delta=-100.0, x_wD=1.2,
        base=Parametrization.from_hyperon_potentials(
            U_Lambda=-30.0, U_Sigma=30.0, U_Xi=-18.0))


@pytest.fixture(scope="module")
def vmit():
    return get_vmit_custom(B4=180.0, a=0.15, m_s=150.0)


def _window(par, vmit, eta, T):
    return locate_window(par, FLAGS, GRID, eta, beta_eq_neutrinoless(),
                         vmit_params=vmit, T=T)


@pytest.mark.parametrize("eta,T", [(0.3, 0.0), (0.3, 25.0), (0.6, 0.0)])
def test_boundaries_are_bracketed_crossings(par, vmit, eta, T):
    """A finite boundary must have probes on both sides of the crossing.

    This is the invariant both bugs broke, and it is what separates a located
    crossing from a fabricated one: if the offset is real, some probe below it
    has chi < 1 and some probe above it has chi > 1. A value produced by the
    'grid ended inside the window' fallback, or by giving up mid-bisection, has
    nothing above it at all.
    """
    w = _window(par, vmit, eta, T)
    assert w.exists, f"no window at eta={eta}, T={T}"

    chi = np.array([r.chi for r in w.probes])
    n_B = np.array([r.n_B for r in w.probes])
    for target, boundary in ((0.0, w.n_onset), (1.0, w.n_offset)):
        below = chi[n_B <= boundary + TOL]
        above = chi[n_B >= boundary - TOL]
        assert below.size and below.min() <= target, (
            f"chi={target:.0f} boundary {boundary:.4f} has no probe below it")
        assert above.size and above.max() >= target, (
            f"chi={target:.0f} boundary {boundary:.4f} has no probe above it — "
            "it was not located, it was guessed")


@pytest.mark.parametrize("eta,n_onset,n_offset", [
    # Reference values from a dense sweep_mixed over 0.95-1.45 fm^-3 in steps
    # of 0.005, reading the crossings off chi by linear interpolation. Before
    # the fix the offsets came back 1.1437 (eta=0.3) and 1.1437 (eta=0.6) —
    # wrong by ~11 grid spacings, and silently so.
    (0.3, 0.9989, 1.1753),
    (0.6, 0.9995, 1.1709),
    (1.0, 0.9995, 1.1702),
])
def test_boundaries_match_a_dense_sweep(par, vmit, eta, n_onset, n_offset):
    """The cheap locator must agree with the expensive sweep it stands in for.

    `locate_window` exists only to avoid solving the mixed system everywhere,
    so the moment it disagrees with a dense sweep it has stopped being an
    optimisation and started being a different answer. Tolerance is a few times
    the bisection tolerance, which is what the locator promises.
    """
    w = _window(par, vmit, eta, T=0.0)
    assert w.exists
    assert w.n_onset == pytest.approx(n_onset, abs=4 * TOL)
    assert w.n_offset == pytest.approx(n_offset, abs=4 * TOL)


def test_offset_shrinks_smoothly_with_temperature(par, vmit):
    """The zig-zag test, stated as a bound on the step between neighbours.

    The visible symptom was an offset that jumped by ~0.15 fm^-3 between
    adjacent temperatures and came back. Physically the window closes smoothly,
    so consecutive temperatures 5 MeV apart cannot move the boundary that far.
    """
    Ts = [15.0, 20.0, 25.0, 30.0]
    offsets = np.array([_window(par, vmit, 0.3, T).n_offset for T in Ts])
    assert np.all(np.isfinite(offsets)), dict(zip(Ts, offsets))
    steps = np.diff(offsets)
    assert np.all(steps < 0.0), f"offset must fall with T, got {offsets}"
    assert np.all(np.abs(steps) < 0.06), f"jump in the boundary: {offsets}"


@pytest.mark.parametrize("hint", [
    None,
    (0.98, 1.18),            # roughly right, as a neighbouring T would give
    (0.20, 0.40),            # badly wrong: the window is nowhere near here
    (1.40, 1.70),            # badly wrong the other way
])
def test_a_hint_only_accelerates_the_search(par, vmit, hint):
    """The chained hint must never be able to change the answer.

    `build_mixed_table` feeds each temperature the window the previous one
    found, which is what keeps the boundary continuous where a cold search
    stops converging. That is only safe if a hint that has gone stale — the
    window jumped, or vanished — is detected and discarded rather than
    returning the edge of the search box as a phase boundary.
    """
    from eos.mixed.tables.generate import MixedTableSpec, _locate_chained

    spec = MixedTableSpec(par, FLAGS, "beta_eq_neutrinoless",
                          axes={"nB": GRID, "T": [0.0]}, eta=0.3,
                          vmit_params=vmit)
    w = _locate_chained(spec, GRID, beta_eq_neutrinoless(), vmit, 0.0, hint)
    assert w.exists, f"hint {hint} lost the window entirely"
    assert w.n_onset == pytest.approx(0.9989, abs=4 * TOL)
    assert w.n_offset == pytest.approx(1.1753, abs=4 * TOL)


def test_nH0_seeds_the_first_point_inside_the_window(par, vmit):
    """`sweep_mixed(nH0=...)` is what lets a sweep restart deep in the window.

    Without it the first point guesses the hadronic phase at the *total*
    density; once chi is appreciable the two have diverged and the solve walks
    into scalar collapse. The bisection in `locate_window` restarts from solved
    points deep inside the window on every step, so it depends on this.
    """
    spec = beta_eq_neutrinoless()
    seed_grid = np.linspace(0.95, 1.15, 30)     # stops short of the offset
    deep = sweep_mixed(par, FLAGS, seed_grid, 0.3, spec, vmit_params=vmit,
                       T=0.0)
    start = deep[-1]
    assert 0.0 < start.chi < 1.0, "fixture point is not inside the window"
    assert start.th_H.n_B < start.n_B - 0.05, "phases have not diverged yet"

    from eos.mixed.equilibrium.residual import mixed_slots
    slots = mixed_slots(spec, 0.3, FLAGS)
    x0 = [start.potentials[s] for s in slots]
    target = [start.n_B, start.n_B + 0.04]

    with_seed = sweep_mixed(par, FLAGS, target, 0.3, spec, vmit_params=vmit,
                            T=0.0, x0=x0, nH0=start.th_H.n_B)
    assert len(with_seed) == 2, "nH0 restart lost a point it should have kept"
    assert with_seed[0].chi == pytest.approx(start.chi, abs=1e-6)
