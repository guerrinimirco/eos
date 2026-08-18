"""
mixed/boundaries.py
===================
Location of the mixed-phase window: where the first-order transition sits on a
density line.

*Public API* (re-exported from `eos.mixed`): `MixedWindow`, `locate_window`.

Solving the mixed system at a density that turns out to be pure phase is
wasted work, and most of a realistic grid is pure phase. `locate_window` finds
the two boundaries first, using chi as the indicator (chi <= 0 hadronic,
chi >= 1 quark, in between mixed), so the expensive mixed solve runs only
where it is the answer.

There is a trap in that point worth stating plainly, because it produced
silently missing transitions. chi is only readable where the mixed system has a
solution, and below the onset it has none — so a density down there does not
report chi <= 0, it drops out of the sweep entirely. The probe set can
therefore begin *above* the onset with every chi > 0, whereupon no sign change
exists to bracket and a real transition is reported as absent. `locate_window`
answers that by walking to the boundary from a point known to be inside the
window (`walk_to_crossing`), where a step that will not converge is itself the
hadronic side rather than missing information.

The empirical bracketing heuristics live here rather than in the residual,
mirroring the split between physics and continuation.
"""
from dataclasses import dataclass

import numpy as np

from eos.mixed.solver import _as_x0, mixed_slots, sweep_mixed

#: Most single-tolerance steps `locate_window` will take when it has to walk to
#: boundary -- so this is a cost ceiling, not a working limit.
MAX_WALK = 64


@dataclass
class MixedWindow:
    """Where the first-order transition sits on one density line.

    n_onset  : density at which chi reaches 0 — the last hadronic point
    n_offset : density at which chi reaches 1 — the first pure quark point
    Both are nan when there is no transition on the grid (the quark phase never
    becomes favourable for these parameters, which is a physics outcome, not a
    failure). `probes` keeps the solved points used to find the boundaries so a
    caller can reuse them.
    """
    n_onset: float
    n_offset: float
    probes: list

    @property
    def exists(self):
        """True only for a well-ordered window. chi is a solved quantity, not a
        monotone parameter, so a sparse or noisy probe set can bracket the two
        crossings out of order; that is not a transition, it is a failed
        location, and callers must not treat it as one."""
        return (np.isfinite(self.n_onset) and np.isfinite(self.n_offset)
                and self.n_offset > self.n_onset)

    @property
    def reason(self):
        """Why `exists` is False, as a distinct label; 'ok' when it is True.

        The four outcomes below are not interchangeable, and reporting them as
        one label costs the caller the ability to tell physics from failure: a
        parametrization with no transition and one whose boundary was never
        bracketed both look like 'no window' in a scan table, so a scan cannot
        say how much of its reject count is real.

          no_transition          chi never crossed either target on this grid
          onset_unbracketed      chi = 1 was located, chi = 0 was not
          offset_unbracketed     chi = 0 was located, chi = 1 was not
          crossings_out_of_order both located, but offset <= onset
        """
        if self.exists:
            return "ok"
        on, off = np.isfinite(self.n_onset), np.isfinite(self.n_offset)
        if on and off:
            return "crossings_out_of_order"
        if off:
            return "onset_unbracketed"
        if on:
            return "offset_unbracketed"
        return "no_transition"

    def contains(self, n_B):
        return self.exists and self.n_onset <= n_B <= self.n_offset


def locate_window(par, flags, n_B_grid, eta, spec, vmit_params=None, T=0.0,
                  n_probe=12, tol=None, analytic_jac=False, x0=None,
                  hint=None, max_refine=2):
    """Find the mixed window on `n_B_grid` by bracketing the chi crossings.

    Probes the grid coarsely, reading chi as the regime indicator (chi <= 0
    hadronic, chi >= 1 quark), then bisects the chi = 0 and chi = 1 crossings
    to `tol` (default: half a grid spacing). In exchange for a couple of dozen
    solves, the caller can then skip the mixed system everywhere outside the
    window, which on a realistic density grid is most of it.

    `hint` is an optional (n_lo, n_hi) span to concentrate the probes in, for
    when a neighbouring temperature or eta has already shown roughly where the
    transition sits. Do not assume a window at one eta brackets the window at
    another: raising eta narrows the window but also moves it, sometimes to
    lower density, so a hint taken from a different eta should be generous.

    Without a hint the probe set is refined up to `max_refine` times, each time
    tripling the density of probes over the sub-range that brackets the
    crossings, which recovers narrow windows at the cost of a few more solves.
    """
    grid = np.asarray(n_B_grid, dtype=float)
    if grid.size < 2:
        return MixedWindow(np.nan, np.nan, [])
    if tol is None:
        tol = 0.5 * float(np.min(np.diff(np.sort(grid))))
    slots = mixed_slots(spec, eta, flags)

    def scan(lo, hi, count):
        """Solve at `count` densities across [lo, hi], warm-started along the way.

        Delegates to `sweep_mixed` so the probes get the same continuation a
        table sweep gets: a step that misses is bisected rather than dropped,
        and the warm start survives a point that is skipped anyway.

        That matters more here than it looks. The probe spacing is the whole
        grid divided by `n_probe`, which is a far bigger step than any table
        sweep takes, and a step that large inside the window regularly misses.
        Dropping those points loses every probe above the first failure, so the
        chi >= 1 side of the transition is never seen, the refinement below
        concludes there is no transition, and the offset falls through to the
        last probe that happened to converge -- a boundary that then jumps
        around from one temperature to the next.
        """
        return sweep_mixed(par, flags, np.unique(np.linspace(lo, hi, count)),
                           eta, spec, vmit_params=vmit_params, T=T, x0=x0,
                           analytic_jac=analytic_jac)

    lo, hi = (float(grid[0]), float(grid[-1])) if hint is None else (
        max(float(grid[0]), float(hint[0])), min(float(grid[-1]), float(hint[1])))
    probes = scan(lo, hi, min(n_probe, grid.size))

    # Narrow the search to the stretch that actually straddles the transition,
    # then re-probe it more finely. A window thinner than the coarse spacing is
    # invisible to the first pass but obvious to the second.
    for _ in range(max_refine if hint is None else 0):
        partial = [r for r in probes if 0.0 < r.chi < 1.0]
        below = [r.n_B for r in probes if r.chi <= 0.0]
        above = [r.n_B for r in probes if r.chi >= 1.0]
        if partial and below and above:
            break                              # both crossings already bracketed
        if not (below and above):
            break                              # no transition on this grid at all
        sub_lo, sub_hi = max(below), min(above)
        if sub_hi <= sub_lo or (sub_hi - sub_lo) <= tol:
            break
        probes += scan(sub_lo, sub_hi, 3 * min(n_probe, grid.size))

    if not probes:
        return MixedWindow(np.nan, np.nan, [])

    def bisect(target, lo, hi):
        """Density at which chi crosses `target`, bracketed by (lo, hi).

        Each midpoint is reached by continuation from the low end of the
        bracket, not by a lone solve. The starting bracket is a full coarse
        probe spacing wide -- a tenth of the grid -- and a single solve across
        a gap that size misses often, especially deep in the window.

        A midpoint that still will not converge returns nan rather than the
        current bracket midpoint. Accepting the midpoint looks like a graceful
        degradation and is not: it reports a boundary up to half a probe
        spacing wrong *as if it had converged*, and since the error depends on
        where the probes happened to land it moves unpredictably from one
        temperature to the next. nan says the crossing was bracketed but not
        located, which is what actually happened.
        """
        r_lo, r_hi = lo, hi
        while (r_hi.n_B - r_lo.n_B) > tol:
            n_mid = 0.5 * (r_lo.n_B + r_hi.n_B)
            stepped = sweep_mixed(par, flags, [r_lo.n_B, n_mid], eta, spec,
                                  vmit_params=vmit_params, T=T,
                                  x0=_as_x0(r_lo, slots),
                                  nH0=r_lo.th_H.n_B,
                                  analytic_jac=analytic_jac)
            if not stepped or abs(stepped[-1].n_B - n_mid) > 1e-12:
                return np.nan
            r = stepped[-1]
            probes.append(r)
            if (r.chi - target) * (r_lo.chi - target) > 0.0:
                r_lo = r
            else:
                r_hi = r
        return 0.5 * (r_lo.n_B + r_hi.n_B)

    def crossing(target, above=None):
        """Bracket the first chi = `target` crossing among the probes.

        `above` restricts the search to densities beyond a boundary already
        found, so the offset is always looked for on the quark side of the
        onset. Without that the two crossings can be picked up in the wrong
        order — chi is a solved quantity, not a monotone parameter, and a
        coarse probe set can straddle both crossings inside one interval.
        """
        ordered = sorted(probes, key=lambda r: r.n_B)
        if above is not None and np.isfinite(above):
            ordered = [r for r in ordered if r.n_B >= above]
        for a, b in zip(ordered, ordered[1:]):
            if (a.chi - target) * (b.chi - target) <= 0.0:
                return bisect(target, a, b)
        return np.nan

    def walk_to_crossing(target, direction):
        """Step from the nearest mixed probe towards the chi = `target`
        crossing, one `tol` at a time, when the probe set never bracketed it.

        This covers the case the probe scan is blind to. `sweep_mixed` drops a
        density it cannot solve, and below the onset the mixed system has no
        solution at all, so the probe set that comes back can *begin above the
        onset* with every chi > 0. `crossing` then finds no sign change and a
        transition that plainly exists — its chi = 1 side is located — is
        reported as absent. Which side of the lowest surviving probe the onset
        falls on decides it, so the outcome moves with the parametrization, the
        temperature and the probe count, for no physical reason.

        Walking sees the boundary either way: chi crosses the target, or the
        solve stops converging. Going down those are the same event, because
        the formal mixed solution ceases to exist on the hadronic side — that
        is what a dropped probe down there means. Going up, a failure is only a
        failure; a missing pure-quark boundary is not evidence of one, so it
        returns nan rather than inventing an offset.

        Steps are `tol` and warm-started, which is the regime the mixed solve
        is reliable in, and the answer lands already refined to `tol`, so no
        bisection follows.
        """
        inside = [r for r in probes if r.in_mixed_phase]
        if not inside:
            return np.nan
        r = (min(inside, key=lambda p: p.n_B) if direction < 0
             else max(inside, key=lambda p: p.n_B))
        for _ in range(MAX_WALK):
            n_next = r.n_B + direction * tol
            if not (lo <= n_next <= hi):
                return np.nan                  # ran off the grid, not a crossing
            stepped = sweep_mixed(par, flags, [r.n_B, n_next], eta, spec,
                                  vmit_params=vmit_params, T=T,
                                  x0=_as_x0(r, slots), nH0=r.th_H.n_B,
                                  analytic_jac=analytic_jac)
            if not stepped or abs(stepped[-1].n_B - n_next) > 1e-12:
                return 0.5 * (n_next + r.n_B) if direction < 0 else np.nan
            r = stepped[-1]
            probes.append(r)
            if (r.chi <= target) if direction < 0 else (r.chi >= target):
                return r.n_B - 0.5 * direction * tol
        return np.nan

    n_onset = crossing(0.0)
    if not np.isfinite(n_onset):
        n_onset = walk_to_crossing(0.0, -1)
    n_offset = crossing(1.0, above=n_onset)
    if not np.isfinite(n_offset):
        n_offset = walk_to_crossing(1.0, +1)
    # A crossing may be missing because the grid begins or ends inside the
    # window rather than because there is no transition; fall back to the
    # extent of the probes that actually came out mixed.
    #
    # Only when the probes reached that end of the range, though. If the solves
    # stopped converging partway up, the last mixed probe is where the SOLVER
    # gave up, not where the phase boundary is, and returning it turns a
    # convergence failure into a confident wrong number -- one that moves with
    # the probe spacing and so wanders from one temperature to the next. nan
    # here means `exists` is False and the caller reports no window, which is
    # the honest answer when the boundary was never bracketed.
    mixed = sorted(r.n_B for r in probes if r.in_mixed_phase)
    if mixed:
        reached = sorted(r.n_B for r in probes)
        # "The grid ended inside the window" means two things at once: the
        # probes got to that end of the range, AND that side of the transition
        # was never seen. If a chi >= 1 probe exists, the offset was bracketed
        # and merely failed to refine, and the last mixed probe is nowhere near
        # it -- so the fallback must not fire.
        if (not np.isfinite(n_onset) and reached[0] <= lo + tol
                and not any(r.chi <= 0.0 for r in probes)):
            n_onset = mixed[0]
        if (not np.isfinite(n_offset) and reached[-1] >= hi - tol
                and not any(r.chi >= 1.0 for r in probes)):
            n_offset = mixed[-1]
    return MixedWindow(float(n_onset), float(n_offset), probes)
