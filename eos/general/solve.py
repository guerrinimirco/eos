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

#: Evaluation budget for the Levenberg-Marquardt rescue attempt.
#:
#: `solve_system` follows a missed gate with LM, which exists for the solve
#: that very nearly worked: MINPACK's own termination leaves the residual
#: around 1e-9, above the gate, and one more attempt polishes it to machine
#: precision. LM cannot manufacture a root that is not there, and when the
#: conditions have none -- a density below a model's deconfinement onset, a
#: parameter point a sampler proposed outside the physical region -- it grinds
#: to scipy's default limit and reports the same failure far more slowly.
#:
#: So the budget bounds what a rescue may SPEND rather than guessing whether
#: one is possible, which is the distinction that matters: a residual-size
#: gate looks reasonable and is wrong, because a rescue's starting residual
#: says nothing about whether it will succeed. Measured over test/did,
#: test/ccdm, test/njl, test/zl, test/vmit and test/alphabag, the 127 LM calls
#: that reached the gate used at most 238 evaluations and started from
#: residuals as high as 1.4; `eos.ccdm` below its deconfinement onset, where
#: there is no root at all, spent 1404 per call. This clears every observed
#: rescue by 1.7x and cuts the doomed ones by 3.5x.
#:
#: A rescue that genuinely needed more comes back as non-convergence, which is
#: a return value the caller can score (CLAUDE.md section 6) -- never a wrong
#: answer, and never a hang.
LM_MAX_EVALUATIONS = 400


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


def solve_system(residual, x0, scales_at, x0_fallback=None, tol=None):
    """Solve one equilibrium system and judge it on its scaled residual.

    Powell's hybrid method first, Levenberg-Marquardt if that does not reach
    the gate, and -- when the caller passed a warm start -- one more hybrid
    attempt from the mode's own cold guess, since a warm start carried across
    a threshold can land outside the basin. Three attempts at most: a
    parameter scan must always get an answer back, and every attempt is
    bounded internally.

    The Levenberg-Marquardt attempt is capped at `LM_MAX_EVALUATIONS`, which
    clears every rescue measured here and stops a system with no root from
    grinding to scipy's own limit. That is what keeps a point with no root
    cheap, which matters because a sampler and a density sweep both meet those
    constantly.

    `scales_at(x)` returns the per-equation scales at the point x, so the
    residual is judged in dimensionless terms (see `scaled_residual_max`).

    `tol` is passed straight to the root finder, and is what a model whose
    rows are already dimensionless needs: MINPACK's own default termination
    leaves the residual around 1e-9, which is above the gate below, so such a
    solve would be judged non-converged however good its state was. None keeps
    the root finder's defaults, which is what every caller predating this
    argument gets.

    Returns (x, scaled residual, converged) for the best attempt made.
    """
    attempts = [('hybr', x0), ('lm', x0)]
    if x0_fallback is not None:
        attempts.append(('hybr', x0_fallback))

    best_x, best_err = np.asarray(x0, dtype=float), np.inf
    for method, guess in attempts:
        options = ({'maxiter': LM_MAX_EVALUATIONS} if method == 'lm' else None)
        sol = root(residual, guess, method=method, tol=tol, options=options)
        err = scaled_residual_max(residual(sol.x), scales_at(sol.x))
        if err < best_err:
            best_x, best_err = sol.x, err
        if best_err <= RESIDUAL_TOL:
            break
    return best_x, best_err, bool(best_err <= RESIDUAL_TOL)


def undetermined_unknowns(jacobian, names, rtol=1.0e-10):
    """Which unknowns the equations do not constrain, read off the Jacobian.

    An unknown whose residual row is identically zero -- a conserved-charge
    potential no populated species carries, say -- appears here as a COLUMN of
    the Jacobian that is zero to numerical precision. Nothing determines it:
    the solve stops wherever its path ran out and reports round-off.

    That is worth catching directly rather than through its consequences,
    because the consequences are subtle and expensive. Carried as an unknown
    with no equation, a null column makes the problem rank-deficient, and a
    least-squares termination then fires early and leaves the residual of the
    WHOLE solve decades above what the model's other modes reach -- close
    enough to `RESIDUAL_TOL` for round-off to decide which side of the gate a
    point lands on, and for a solver that answers a missed gate by trying
    another root to select the other root by round-off. So an undetermined
    potential is a CONDITIONING hazard and not only a reporting one.

    `jacobian` is (n_rows, n_unknowns) and `names` labels its columns, in the
    order the model's unknown vector carries them. Columns are compared
    against the largest column norm rather than an absolute floor, since the
    rows carry mixed units. Returns the names of the unconstrained unknowns,
    in column order; an empty list is a well-posed system.

    The cure is not to widen the tolerance but to give the unknown a row --
    pinning it at a declared value where the physics leaves it free.
    """
    columns = np.atleast_2d(np.asarray(jacobian, dtype=float))
    if columns.shape[1] != len(names):
        raise ValueError(f"jacobian has {columns.shape[1]} columns and "
                         f"{len(names)} names were given")
    norms = np.linalg.norm(columns, axis=0)
    largest = float(norms.max()) if norms.size else 0.0
    if largest == 0.0:
        return list(names)
    return [name for name, norm in zip(names, norms)
            if norm <= rtol * largest]
