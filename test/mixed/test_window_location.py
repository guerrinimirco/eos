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


def test_a_dropped_probe_below_the_onset_does_not_hide_the_transition():
    """The onset must be found even when no probe lands on the hadronic side.

    `sweep_mixed` drops a density it cannot solve, and below the onset the
    mixed system has no solution, so those probes vanish and the probe set can
    begin above the onset with every chi > 0. There is then no sign change to
    bracket and the transition is reported as absent — while its chi = 1 side
    sits located at 1.04, which is the tell.

    This parametrization does exactly that: its lowest surviving probe comes
    out at chi = +0.0024, a hair above the crossing. Moving U_Lambda to -30
    shifts the onset by ~0.005 fm^-3, the same probe lands at chi = -0.0009,
    and the search succeeds — so the two differ only in which side of one
    probe the boundary falls on, and both must give the same window.
    """
    grid = np.linspace(0.05, 1.6, 80)
    vm = get_vmit_custom(B4=170.0, a=0.20, m_s=150.0)

    def window_for(U_Lambda):
        p = Parametrization.from_nmp(dict(
            n_sat=0.149077, E_sat=-16.02, m_eff_ratio=0.5625, K_sat=290.0,
            Q_sat=300.0, E_sym=31.67, L_sym=50.0))
        p = Parametrization.from_hyperon_potentials(
            U_Lambda=U_Lambda, U_Sigma=30.0, U_Xi=-10.0, base=p)
        p = Parametrization.from_delta_potential(
            U_Delta=-50.0, x_wD=1.20, x_rD=1.00, base=p)
        return locate_window(p, FLAGS, grid, 0.0, beta_eq_neutrinoless(),
                             vmit_params=vm, T=0.0)

    tol = 0.5 * float(np.min(np.diff(grid)))
    w25, w30 = window_for(-25.0), window_for(-30.0)
    assert w30.exists, "the control case regressed"
    assert w25.exists, (
        f"onset lost: reason={w25.reason}, offset={w25.n_offset:.4f} was "
        "located, so a transition exists")
    # Same transition seen from either side of one probe: U_Lambda moves the
    # onset a little, not by the probe spacing (0.14) the bug swung it by.
    assert abs(w25.n_onset - w30.n_onset) < 6 * tol, (
        f"{w25.n_onset:.4f} vs {w30.n_onset:.4f}")


@pytest.mark.parametrize("n_probe", [8, 12, 20])
@pytest.mark.parametrize("n_lo", [0.03, 0.05, 0.08])
def test_the_boundaries_do_not_depend_on_where_the_probes_land(
        par, vmit, n_probe, n_lo):
    """The located window must be a property of the physics, not of the grid.

    The failure this pins is invisible to any single-grid test: it turns on
    whether some probe happens to fall below the onset, so it appears and
    disappears as the grid start or the probe count is nudged. Invariance under
    both is the property that actually holds, and asserting it is the only way
    to catch a locator that is right by luck.

    Reference values are the module's dense-sweep numbers, so this also pins
    the walk fallback to the same answer the bracketing path gives.
    """
    grid = np.linspace(n_lo, 1.788924, 300)
    w = locate_window(par, FLAGS, grid, 0.3, beta_eq_neutrinoless(),
                      vmit_params=vmit, T=0.0, n_probe=n_probe)
    assert w.exists, f"lost the window at n_probe={n_probe}, n_lo={n_lo}"
    assert w.n_onset == pytest.approx(0.9989, abs=6 * TOL)
    assert w.n_offset == pytest.approx(1.1753, abs=6 * TOL)


def test_reason_separates_physics_from_a_failed_location(par, vmit):
    """`no_transition` and the three locator failures must be distinguishable.

    A scan that labels them all 'no window' cannot say how much of its reject
    count is physics — which is what made a 27% bucket of unlocated onsets
    indistinguishable from parametrizations that genuinely have no transition.
    """
    from eos.mixed.solvers.sweep import MixedWindow

    assert MixedWindow(0.9, 1.2, []).reason == "ok"
    assert MixedWindow(np.nan, np.nan, []).reason == "no_transition"
    assert MixedWindow(np.nan, 1.2, []).reason == "onset_unbracketed"
    assert MixedWindow(0.9, np.nan, []).reason == "offset_unbracketed"
    assert MixedWindow(1.2, 0.9, []).reason == "crossings_out_of_order"

    # A bag constant high enough that quark matter never pays: physics, and it
    # must say so rather than reporting a location failure.
    far = get_vmit_custom(B4=400.0, a=vmit.a, m_s=vmit.m_s)
    w = locate_window(par, FLAGS, GRID, 0.3, beta_eq_neutrinoless(),
                      vmit_params=far, T=0.0)
    assert w.reason == "no_transition"


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
