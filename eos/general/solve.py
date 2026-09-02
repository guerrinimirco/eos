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
    if best_err > RESIDUAL_TOL:
        best_x, best_err = newton_polish(residual, best_x, scales_at, best_err)
    return best_x, best_err, bool(best_err <= RESIDUAL_TOL)


#: Relative steps the polish differentiates at, largest first. Central
#: differences, so the truncation error is O(h^2) and a step far below
#: MINPACK's sqrt(macheps) ~ 1.5e-8 is still meaningful -- and a SPREAD is
#: needed rather than one value, because the right step depends on the
#: unknown: a potential of order 10^3 MeV wants a coarse one, and a colour
#: potential sitting at 1e-6 MeV wants a fine one. Measured on the njl 2SC
#: point below, 1e-8 takes the residual from 1.3e-8 to 5e-11 and the
#: neighbouring two do nothing at all.
POLISH_JACOBIAN_STEPS = (1.0e-6, 1.0e-8, 1.0e-10)

#: How many Newton steps the polish may take, and how far it may back off
#: along one. A step that does not reduce the residual at full, half or
#: quarter length is a step into a different basin, and the polish stops.
POLISH_STEPS = 3
POLISH_BACKOFFS = (1.0, 0.5, 0.25)


def newton_polish(residual, x, scales_at, err, steps=POLISH_STEPS):
    """A few damped Newton steps from the best iterate a root finder reached.

    MINPACK stops on its OWN progress test, not on this repository's gate, and
    the two part company in a way that has nothing to do with the physics: its
    forward-difference Jacobian steps by sqrt(macheps)|x|, which is 5e-6 at
    the mu ~ 10^3 MeV of dense matter, so once the remaining Newton step is of
    that size it reports "not making good progress" and returns. Observed in
    `eos.njl`: a 2SC point whose scaled residual sat at 1.3e-8, three decades
    above `RESIDUAL_TOL`, with a genuine root 7e-6 away in x and a Jacobian
    conditioned at 50 -- and reached by the same solver from the same start
    under a different rounding of the same equations. A termination artefact,
    not a state that fails to exist.

    So the polish re-differentiates centrally at each of
    `POLISH_JACOBIAN_STEPS`, takes the Newton step, and backs off along it
    while that does not help. It runs ONLY after the gate has been missed, so
    it can lower a reported residual and never raise one; a system with no
    root nearby stops on the first non-improving step.

    Returns the better of (x, err) and what the polish reached.
    """
    x = np.asarray(x, dtype=float)
    best_x, best_err = x, err
    for _ in range(steps):
        r0 = np.asarray(residual(best_x), dtype=float)
        n = best_x.size
        if r0.size != n:
            return best_x, best_err          # not square: Newton has no step
        improved = False
        for relative_step in POLISH_JACOBIAN_STEPS:
            jacobian = np.empty((r0.size, n))
            for i in range(n):
                h = relative_step * max(abs(best_x[i]), 1.0)
                up, down = best_x.copy(), best_x.copy()
                up[i] += h
                down[i] -= h
                jacobian[:, i] = (np.asarray(residual(up), dtype=float)
                                  - np.asarray(residual(down), dtype=float)
                                  ) / (2.0 * h)
            try:
                step = np.linalg.lstsq(jacobian, -r0, rcond=None)[0]
            except np.linalg.LinAlgError:
                continue
            for damping in POLISH_BACKOFFS:
                trial = best_x + damping * step
                trial_err = scaled_residual_max(residual(trial),
                                                scales_at(trial))
                if trial_err < best_err:
                    best_x, best_err, improved = trial, trial_err, True
                    break
            if improved:
                break
        if not improved or best_err <= RESIDUAL_TOL:
            break
    return best_x, best_err


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
