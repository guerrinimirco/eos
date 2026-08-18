"""Solving one equilibrium system, and judging whether it was solved.

Every model in this repository closes its modes the same way: assemble a
handful of equations, hand them to a root finder, and decide from the residual
whether what came back is a state. The deciding is the part worth writing
once, because it is the part that is easy to get wrong in a way nothing
notices -- a solver's own success flag reports that the iteration terminated,
which is a different question from whether the equations are satisfied, and a
norm of the raw residual vector is dominated by whichever row happens to carry
the largest units.

    x, error, converged = solve_system(residual, x0, scales_at)

`residual(x)` returns the equations, `scales_at(x)` returns the scale of the
quantity each one balances, and the state is accepted when the largest scaled
component is below `RESIDUAL_TOL`.

This does not know what the equations mean, and no model's physics lives here:
what a row balances, and therefore what it should be divided by, is the
model's own statement and arrives as `scales_at`.
"""
import numpy as np
from scipy.optimize import root

#: Post-solve gate on the equilibrium residuals, each divided by the scale of
#: the quantity its equation balances. One number for the whole repository, so
#: that "converged" means the same thing in every model.
RESIDUAL_TOL = 1.0e-10

#: Floor on the potential scale, so a pathological iterate passing through
#: mu_B = 0 cannot divide by zero. Physical dense matter has mu_B ~ 10^3 MeV.
MU_SCALE_FLOOR = 1.0


def scaled_residual_max(residuals, scales):
    """The largest residual once each is divided by its own scale.

    The rows of a mode carry mixed units: densities and charge conditions in
    fm^-3, of order 10^-1, fractions of order unity, and equalities between
    chemical potentials in MeV, of order 10^3. A norm of the raw vector is
    therefore dominated by whichever row happens to be largest, and accepts
    states that satisfy the others only loosely. Dividing each residual by the
    scale of the quantity it balances -- n_B for a density, mu_B for a
    potential -- makes the components comparable, so one tolerance means the
    same thing for all of them.
    """
    return max(abs(r) / s for r, s in zip(residuals, scales))


def solve_system(residual, x0, scales_at, x0_fallback=None):
    """Solve one equilibrium system and judge it on its scaled residual.

    Powell's hybrid method first, Levenberg-Marquardt if that does not reach
    the gate, and -- when the caller passed a warm start -- one more hybrid
    attempt from the mode's own cold guess, since a warm start carried across
    a threshold can land outside the basin. Three attempts at most: a
    parameter scan must always get an answer back, and every attempt is
    bounded internally.

    `scales_at(x)` returns the per-equation scales at the point x, so the
    residual is judged in dimensionless terms (see `scaled_residual_max`).

    Returns (x, scaled residual, converged) for the best attempt made.
    """
    attempts = [('hybr', x0), ('lm', x0)]
    if x0_fallback is not None:
        attempts.append(('hybr', x0_fallback))

    best_x, best_err = np.asarray(x0, dtype=float), np.inf
    for method, guess in attempts:
        sol = root(residual, guess, method=method)
        err = scaled_residual_max(residual(sol.x), scales_at(sol.x))
        if err < best_err:
            best_x, best_err = sol.x, err
        if best_err <= RESIDUAL_TOL:
            break
    return best_x, best_err, bool(best_err <= RESIDUAL_TOL)
