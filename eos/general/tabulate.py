"""The grid driver every model's table builder is made of.

A table in this repository always has the same shape: an outer set of *lines*
-- one per temperature and per combination of the fractions the mode fixes --
and, within each line, a sweep along the stiff axis, the baryon density. The
density sweep is warm-started, because each solved point is the only good
starting guess for the next one; the lines are independent of each other.

Only two things differ between models: how one point is solved, and what part
of a solved point makes the next warm start. Both are arguments here, so the
loop, the skipping, the timing and the progress reporting are written once.

    lines = lines_from_axes({"T": [0, 10], "Y_C": [0.1, 0.5]})
    points = sweep_lines(lines, nB_grid, solve, warm_start=to_guess)

`solve(value, conditions, x0)` returns a solved point, or None (or raises
RuntimeError / ValueError) where the state does not exist. `warm_start(point)`
returns the guess to seed the next density with; leave it None for a solver
that finds its own starting point.

Deep solver code never prints. Reporting is the `progress` callback: it is
invoked once per completed line with a small dictionary, the same keys for
every model, so one printer serves them all.
"""
import time
from itertools import product

import numpy as np

#: The axis names that mean "temperature axis". A table carries exactly one:
#: `SnB` (entropy per baryon) stands in for T through an outer 1-D solve, and
#: the two are alternatives, never both.
TEMPERATURE_AXES = ("T", "SnB")


def lines_from_axes(axes, fixed=None):
    """The list of line conditions from a table's axes.

    `axes` maps axis name to grid: exactly one temperature axis (`T` or
    `SnB`) plus any fraction axes (`Y_C`, `Y_S`, `Y_Le`, ...). `fixed` supplies
    fractions the mode needs that are not swept. Returns one dict per line,
    temperature outermost, in the order the fraction axes were given -- so
    adding a fraction axis extends the table without reordering what was
    already there.
    """
    temp_keys = [k for k in axes if k in TEMPERATURE_AXES]
    if len(temp_keys) != 1:
        raise ValueError(f"axes need exactly one of {TEMPERATURE_AXES}, "
                         f"got {temp_keys}")
    temp_key = temp_keys[0]
    frac_keys = [k for k in axes if k not in TEMPERATURE_AXES]
    frac_grids = [np.atleast_1d(np.asarray(axes[k], dtype=float))
                  for k in frac_keys]

    lines = []
    for temp in np.atleast_1d(np.asarray(axes[temp_key], dtype=float)):
        for combo in (product(*frac_grids) if frac_grids else [()]):
            conditions = dict(fixed or {})
            conditions[temp_key] = float(temp)
            conditions.update(zip(frac_keys, (float(c) for c in combo)))
            lines.append(conditions)
    return lines


def split_conditions(conditions):
    """(temperature key, temperature value, {fraction: value}) of one line.

    The progress callback reports the temperature separately from the
    fractions because they are read differently: the temperature says which
    isotherm (or isentrope) is being solved, the fractions say which
    composition on it.
    """
    temp_key = next(k for k in conditions if k in TEMPERATURE_AXES)
    fracs = {k: v for k, v in conditions.items() if k not in TEMPERATURE_AXES}
    return temp_key, conditions[temp_key], fracs


def temperature_at_entropy(entropy_per_baryon_at, target, T_lo=0.2, T_hi=80.0,
                           T_cap=400.0, xtol=1e-5):
    """The temperature at which s/n_B reaches `target`, by an outer 1-D solve.

    This is what lets a table take an entropy axis wherever it takes a
    temperature axis: an isentrope is an isotherm whose temperature is solved
    for at every density. `entropy_per_baryon_at(T)` returns s/n_B of the
    state at that temperature.

    s(T) is monotone increasing at fixed density, so the bracket is well
    posed; T_hi is doubled up to T_cap until the target is bracketed, and if
    it still is not the target is unreachable and that is said rather than
    guessed at.
    """
    from scipy.optimize import brentq

    def f(T):
        return entropy_per_baryon_at(T) - target

    f_hi = f(T_hi)
    while f_hi < 0.0 and T_hi < T_cap:
        T_hi = min(2.0 * T_hi, T_cap)
        f_hi = f(T_hi)
    if f_hi < 0.0:
        raise RuntimeError(
            f"entropy per baryon {target} is unreachable below T = {T_cap} "
            f"MeV (s/n_B = {f_hi + target:.3f} there)")
    return brentq(f, T_lo, T_hi, xtol=xtol)


def accepted(point):
    """Whether a solved point is usable: not None, and not flagged as failed.

    Models report non-convergence either by returning a result carrying
    `converged = False` or by returning nothing at all; both are recognised
    here, so a model does not have to change how it reports to use this
    driver.
    """
    return point is not None and getattr(point, "converged", True) is not False


def unconverged_response(reason, quantities):
    """The dict `eos_response` returns when it could not reach the state.

    CLAUDE.md section 6: non-convergence is a return value at every public
    boundary, never an exception. The shape returned here is the SAME one the
    converged path returns -- `quantities` are the names that path would have
    carried at this T and freeze -- with nan in every one of them, so a caller
    writing `result["cs2_adiabatic"]` into an array column needs no second code
    path and the failure propagates to a plot honestly rather than as a gap the
    reader has to know about. `reason` carries the message the internal layer
    raised with; internal layers may still raise, and this is where that raise
    is turned into a status.
    """
    out = {name: np.nan for name in quantities}
    out["converged"] = False
    out["reason"] = reason
    return out


def print_progress(info):
    """The built-in one-line printer, installed by `verbose=True`."""
    fracs = "".join(f" {k}={v:g}" for k, v in info["fracs"].items())
    label = f"{info['mode']} " if info["mode"] else ""
    print(f"[{info['line']}/{info['n_lines']}] {label}"
          f"{info['temp_key']}={info['temp']:g}{fracs}: "
          f"{info['n_solved']}/{info['n_requested']} points "
          f"in {info['elapsed_s']:.1f}s")


def _step(solve, conditions, previous, target, x0, warm_start, max_bisect,
          depth):
    """One point of a sweep, bisecting the step back towards the last success.

    A warm start is only good while the state moves smoothly, and along a
    density sweep it stops being good exactly where something turns on: a
    threshold crossed inside one grid interval leaves the previous point's
    answer outside the new basin. Halving the interval and solving the
    midpoint first gives the continuation a seed on the far side of the
    threshold, and the recursion repeats until the step is small enough or
    `max_bisect` halvings have been spent.

    The midpoints are seeds only -- they are not grid values and never reach
    the returned line. Returns the solved point, or None if the step could not
    be walked.
    """
    exhausted = depth >= max_bisect or previous is None
    try:
        point = solve(target, conditions, x0)
    except (RuntimeError, ValueError):
        if exhausted:
            raise           # nothing left to try: the caller's error, intact
        point = None
    if accepted(point):
        return point
    if exhausted:
        return None

    middle = 0.5 * (previous + target)
    at_middle = _step(solve, conditions, previous, middle, x0, warm_start,
                      max_bisect, depth + 1)
    if not accepted(at_middle):
        return None
    seed = warm_start(at_middle) if warm_start is not None else None
    return _step(solve, conditions, middle, target, seed, warm_start,
                 max_bisect, depth + 1)


def sweep_lines(lines, axis_values, solve, warm_start=None, skip_errors=True,
                progress=None, verbose=False, mode="", max_bisect=0,
                reset_on_failure=True):
    """Solve every line, sweeping `axis_values` with a warm start within each.

    Returns a list of lines, each a list of solved points in the order of
    `axis_values`.

    skip_errors=True (the default) drops a point the solver could not reach
    from its line and carries on, resetting the warm start so the next point
    starts fresh; the line is then shorter than `axis_values`. This is the
    behaviour a parameter scan needs, since a grid always has corners where
    uniform matter has no solution. skip_errors=False raises instead, which is
    what one wants while developing a solver.

    reset_on_failure=False keeps the last successful warm start instead of
    dropping it, so the sweep carries on from the point it last reached rather
    than from a cold guess. That is what a model with more than one solution
    branch needs: a cold start lands on whichever branch it lands on, so
    restarting mid-sweep lets the sequence hop between branches from one grid
    value to the next, which shows up as an equation of state that oscillates
    rather than one that has a transition in it. `eos.enjl` sets it.

    max_bisect is how many times a missed step may be halved back towards the
    last solved point before it is given up on (see `_step`). It is 0 by
    default -- a model with no threshold along its density sweep has nothing
    for a bisected step to walk through, and paying for the recursion would
    buy it nothing -- and a model that crosses onsets sets it, as `eos.dd2`
    and `eos.alphabag` do.

    progress: called once per completed line with
    {mode, line, n_lines, temp_key, temp, fracs, n_solved, n_requested,
    elapsed_s}. verbose=True installs `print_progress` as that callback.
    """
    if verbose and progress is None:
        progress = print_progress
    values = np.asarray(axis_values, dtype=float)

    solved_lines = []
    for conditions in lines:
        started = time.time()
        line, x0, previous = [], None, None
        for value in values:
            try:
                point = _step(solve, conditions, previous, float(value), x0,
                              warm_start, max_bisect, 0)
            except (RuntimeError, ValueError):
                if not skip_errors:
                    raise
                point = None
            if not accepted(point):
                if not skip_errors:
                    raise RuntimeError(
                        f"no solution at value {value} "
                        f"for line {conditions}")
                if reset_on_failure:
                    x0 = None        # start the next point fresh
                continue
            line.append(point)
            previous = float(value)
            x0 = warm_start(point) if warm_start is not None else None
        solved_lines.append(line)

        if progress is not None:
            temp_key, temp, fracs = split_conditions(conditions)
            progress(dict(mode=mode, line=len(solved_lines),
                          n_lines=len(lines), temp_key=temp_key, temp=temp,
                          fracs=fracs, n_solved=len(line),
                          n_requested=len(values),
                          elapsed_s=time.time() - started))
    return solved_lines
