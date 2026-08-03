"""
mixed/continuation.py
=====================
Warm-started density sweep and mixed-window location for the mixed-phase solver
(docs/phase2/SPECIFICATION_AND_PLAN.md §3.4), milestone P1.

A cold pure-phase seed converges only near the onset; through the mixed window
the previous solved point is the reliable predictor. sweep_mixed() warm-starts
each density from its neighbour and bisects the step when the corrector misses
(the tactic eos/dd2/solver.py::sweep_octet uses). find_mixed_window() reports
the (n_B) span where 0 < chi < 1.

The empirical bracketing heuristics live here, not in the residual — mirroring
the Phase-1 split between physics and continuation.
"""
from eos.mixed.solver import solve_mixed
from eos.mixed.residual import mixed_slots


def _as_x0(result, slots):
    return [result.potentials[name] for name in slots]


def sweep_mixed(par, flags, n_B_grid, eta, spec, vmit_params=None, T=0.0,
                max_bisect=6):
    """Warm-started sweep over n_B_grid at fixed eta. Returns a list of
    MixedResult in grid order (points that fail every bisection are skipped,
    so the caller sees the convergent prefix/interior). Each solved point seeds
    the next; a missed step is bisected up to max_bisect levels.
    """
    slots = mixed_slots(spec, eta)

    def solve_from(n_B, x0):
        return solve_mixed(par, flags, n_B, eta, spec, vmit_params=vmit_params,
                           T=T, x0=x0, n_B_guess=n_B, check_consistency=False)

    def step(n_prev, n_target, x0, depth):
        try:
            return solve_from(n_target, x0)
        except RuntimeError:
            if depth >= max_bisect or n_prev is None:
                raise
            n_mid = 0.5 * (n_prev + n_target)
            p_mid = step(n_prev, n_mid, x0, depth + 1)
            return step(n_mid, n_target, _as_x0(p_mid, slots), depth + 1)

    out, x0, n_prev = [], None, None
    for n_B in n_B_grid:
        try:
            p = step(n_prev, n_B, x0, 0)
        except RuntimeError:
            continue                       # leave a hole rather than abort sweep
        out.append(p)
        x0, n_prev = _as_x0(p, slots), n_B
    return out


def find_mixed_window(par, flags, n_B_grid, eta, spec, vmit_params=None, T=0.0):
    """The subset of a sweep that is genuinely in the mixed phase (0<chi<1)."""
    return [r for r in sweep_mixed(par, flags, n_B_grid, eta, spec,
                                   vmit_params=vmit_params, T=T)
            if r.in_mixed_phase]
