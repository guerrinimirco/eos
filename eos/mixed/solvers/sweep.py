"""
mixed/solvers/sweep.py
======================
Warm-started density sweeps and location of the mixed-phase window.

*Public API* (re-exported from `eos.mixed`): `sweep_mixed`, `locate_window`,
`MixedWindow`.

Two facts shape this module.

First, a cold seed only converges near the transition onset; through the window
the previous solved density is by far the best predictor. So every sweep is
warm-started along n_B, and when a step misses, it is bisected rather than
abandoned — the same continuation tactic `eos/dd2/solver.sweep_octet` uses.

Second, and this is what makes a full table affordable: solving the mixed
system at a density that turns out to be pure phase is wasted work, and most of
a realistic grid is pure phase. `locate_window` finds the two boundaries first,
using chi as the indicator (chi <= 0 hadronic, chi >= 1 quark, in between
mixed), so the expensive mixed solve runs only where it is the answer.

The empirical bracketing heuristics live here rather than in the residual,
mirroring the Phase-1 split between physics and continuation.
"""
from dataclasses import dataclass

import numpy as np

from eos.mixed.equilibrium.residual import mixed_slots
from eos.mixed.solvers.point import solve_mixed


def _as_x0(result, slots):
    return [result.potentials[name] for name in slots]


def seed_across_eta(result, spec, eta, flags=None):
    """Re-express a solved point as a starting vector for a different eta.

    eta changes the *shape* of the unknown vector — at eta = 0 only a global
    lepton potential exists, at eta = 1 only the two local ones, in between all
    three — so a solution at one eta cannot be handed to another directly.
    This maps it across, using the physical correspondence between the
    populations that appear and disappear:

      going towards eta = 1  the local potentials start from the global one,
                             which is the value they must approach as the
                             global population loses its weight;
      going towards eta = 0  the global potential starts from the volume
                             average of the two local ones.

    Cold-starting each eta separately is the fragile part of an eta scan,
    especially near the Maxwell endpoint and with a soft hadronic phase; walking
    eta in small steps from a converged neighbour is what makes the scan hold
    together. Returns a list in the slot order of `mixed_slots(spec, eta)`.
    """
    p = dict(result.potentials)
    chi = p.get("chi", result.chi)
    mu_eG = p.get("mu_eG")
    mu_eL_H, mu_eL_Q = p.get("mu_eL_H"), p.get("mu_eL_Q")
    if mu_eL_H is None:                    # came from eta = 0
        mu_eL_H = mu_eL_Q = mu_eG
    if mu_eG is None:                      # came from eta = 1
        mu_eG = (1.0 - chi) * mu_eL_H + chi * mu_eL_Q
    filled = dict(p, mu_eG=mu_eG, mu_eL_H=mu_eL_H, mu_eL_Q=mu_eL_Q)
    return [filled[name] for name in mixed_slots(spec, eta, flags)]


def sweep_mixed(par, flags, n_B_grid, eta, spec, vmit_params=None, T=0.0,
                max_bisect=6, x0=None, analytic_jac=False,
                mixed_only=False):
    """Warm-started sweep over `n_B_grid` at fixed eta.

    Returns a list of `MixedResult` in grid order. Each solved point seeds the
    next; a missed step is bisected up to `max_bisect` levels, and a density
    that still fails is skipped, leaving a hole rather than aborting the sweep.

    `x0` seeds the first point (otherwise the physical cold start is used).
    `mixed_only=True` keeps only the points genuinely inside the window.
    """
    slots = mixed_slots(spec, eta, flags)

    def solve_from(n_B, seed, nH):
        # `nH` seeds the hadronic phase's own internal solve. It must be that
        # phase's density, NOT the total: deep in the window the two diverge
        # badly (as chi -> 1 the hadronic phase thins out and stops tracking
        # n_B at all), and seeding the phase at the total density walks it
        # towards scalar collapse and stalls the solve.
        return solve_mixed(par, flags, n_B, eta, spec, vmit_params=vmit_params,
                           T=T, x0=seed, n_B_guess=(nH if nH is not None else n_B),
                           check_consistency=False, analytic_jac=analytic_jac)

    def step(n_prev, n_target, seed, nH, depth):
        try:
            return solve_from(n_target, seed, nH)
        except RuntimeError:
            if depth >= max_bisect or n_prev is None:
                raise
            n_mid = 0.5 * (n_prev + n_target)
            p_mid = step(n_prev, n_mid, seed, nH, depth + 1)
            return step(n_mid, n_target, _as_x0(p_mid, slots),
                        p_mid.th_H.n_B, depth + 1)

    out, seed, n_prev, nH = [], x0, None, None
    for n_B in n_B_grid:
        try:
            p = step(n_prev, float(n_B), seed, nH, 0)
        except RuntimeError:
            continue
        out.append(p)
        seed, n_prev, nH = _as_x0(p, slots), float(n_B), p.th_H.n_B
    return [r for r in out if r.in_mixed_phase] if mixed_only else out


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

    def contains(self, n_B):
        return self.exists and self.n_onset <= n_B <= self.n_offset


def _chi_at(par, flags, n_B, eta, spec, vmit_params, T, seed, analytic_jac,
            nH=None):
    """One probe solve, or None if the mixed system will not converge there.

    `nH` seeds the hadronic phase's internal solve at that phase's own density
    (see `sweep_mixed`); without it, probes deep in the window seed the phase
    at the total density and stall.
    """
    try:
        return solve_mixed(par, flags, float(n_B), eta, spec,
                           vmit_params=vmit_params, T=T, x0=seed,
                           n_B_guess=(nH if nH is not None else n_B),
                           check_consistency=False, analytic_jac=analytic_jac)
    except RuntimeError:
        return None


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
        """Solve at `count` densities across [lo, hi], warm-started along the way."""
        out, seed, nH = [], x0, None
        for n in np.unique(np.linspace(lo, hi, count)):
            r = _chi_at(par, flags, n, eta, spec, vmit_params, T, seed,
                        analytic_jac, nH)
            if r is None:
                seed, nH = None, None         # reset the warm start past a gap
                continue
            out.append(r)
            seed, nH = _as_x0(r, slots), r.th_H.n_B
        return out

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
        """Density at which chi crosses `target`, bracketed by (lo, hi)."""
        r_lo, r_hi = lo, hi
        while (r_hi.n_B - r_lo.n_B) > tol:
            n_mid = 0.5 * (r_lo.n_B + r_hi.n_B)
            r = _chi_at(par, flags, n_mid, eta, spec, vmit_params, T,
                        _as_x0(r_lo, slots), analytic_jac, r_lo.th_H.n_B)
            if r is None:
                break                          # cannot refine further; accept
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

    n_onset = crossing(0.0)
    n_offset = crossing(1.0, above=n_onset)
    # A crossing may be missing because the grid begins or ends inside the
    # window rather than because there is no transition; fall back to the
    # extent of the probes that actually came out mixed.
    mixed = sorted(r.n_B for r in probes if r.in_mixed_phase)
    if not np.isfinite(n_onset) and mixed:
        n_onset = mixed[0]
    if not np.isfinite(n_offset) and mixed:
        n_offset = mixed[-1]
    return MixedWindow(float(n_onset), float(n_offset), probes)


def find_mixed_window(par, flags, n_B_grid, eta, spec, vmit_params=None, T=0.0):
    """The subset of a full sweep that is genuinely mixed (0 < chi < 1).

    Kept for callers that want every mixed point on the grid rather than just
    the boundaries; `locate_window` is much cheaper when only the boundaries
    are needed.
    """
    return sweep_mixed(par, flags, n_B_grid, eta, spec,
                       vmit_params=vmit_params, T=T, mixed_only=True)
