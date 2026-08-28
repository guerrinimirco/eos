"""The uniform model API for ENJL: eos_point, eos_table, eos_response.

Every model in this repository exposes these three entry points with the same
shape, so a caller -- a table pipeline, a sampler, a downstream package --
drives any model the same way:

    eos_point(par, mode, species, n_B=..., T=0.0)   one state
    eos_table(par, mode, species, axes)             a solved grid
    eos_response(par, mode, species, ...)           second derivatives

All four modes of CLAUDE.md section 3 are closed here, at any non-negative
temperature, and entropy per baryon is accepted wherever a temperature is --
an isentrope is an isotherm whose temperature is solved for at every density.
What this model still refuses is the CONSTRUCTION at T > 0: mapping a branch
is closed, replacing a first-order window by its plateau is not.

Conditions are named exactly n_B, T, Y_C, Y_S and Y_Le. Units at this
boundary are fm-based:
n_B in fm^-3, potentials in MeV, eps and P in MeV/fm^3. Natural units stay
inside `thermodynamics.py` and `solver.py`.

NON-CONVERGENCE IS A RETURN VALUE here, as in every model: this solver has
several branches and a bounded iteration count, and a sampler must be able to
score a point it could not reach and move on. A malformed CALL is different --
an unknown mode, a mode the model refuses, a temperature it does not have, a
species flag it does not carry -- and raises before any work, because it is a
programming error that would otherwise be repeated a million times in silence.
"""
from dataclasses import dataclass

import numpy as np

from eos.enjl.parameters import Parameters
from eos.enjl.solver import (
    MODE_FRACTIONS, check_mode, check_temperature, mode_spec, solve,
    solve_at_entropy,
)
from eos.enjl.species import SpeciesFlags
from eos.enjl.table import (
    DIRECTIONS, TableSpec, beta_row, build_constructed_table, build_table,
)


@dataclass(frozen=True)
class PointResult:
    """One eos_point outcome: a convergence status the caller can test.

    When `ok` is False, `point` is None and `message` says what the request
    reached -- typically the best scaled residual of the starting points tried.
    """
    ok: bool
    message: str
    point: object = None


def _check_call(mode, species, T, SnB, conditions, leptons=None):
    """(T, SnB, species) of a request, or a raise if it is not one to make.

    `mode_spec` validates the fractions against the mode, so a missing or
    surplus condition is refused before any work rather than defaulted.
    Exactly one thermal variable is given: a temperature or an entropy per
    baryon. `SnB` comes back unresolved, because resolving it is an outer
    solve that needs the density as well.
    """
    check_mode(mode)
    mode_spec(mode, leptons=leptons, **conditions)
    if SnB is not None and T:
        raise ValueError("give exactly one of T / SnB")
    T = 0.0 if T is None else float(T)
    check_temperature(T)
    if SnB is not None and SnB < 0.0:
        raise ValueError(f"entropy per baryon must be non-negative; "
                         f"got SnB = {SnB}")
    species = SpeciesFlags() if species is None else species
    return T, SnB, species


def eos_point(par, mode, species=None, n_B=None,
              T=0.0, SnB=None, x0=None, leptons=None, **conditions):
    """One solved state; non-convergence is a return value.

    Parameters
    ----------
    par : Parameters
        The model parameters -- always an argument, never module state.
    mode : str
        Any of the four of CLAUDE.md section 3.
    leptons : bool
        For `fixed_YC` and `fixed_YC_YS`, whether the neutralizing electrons
        and muons are added. With leptons=False the result is electrically
        charged strongly-interacting matter, which is what a mixed-phase
        construction needs for each pure phase before imposing global
        neutrality. Meaningless in beta equilibrium, which is defined by the
        leptons, and passing False there raises.
    species : SpeciesFlags
        The active degrees of freedom. They are fixed by the model (p, n,
        Lambda, u, d, s, e, mu) and moving a flag raises; see
        `eos.enjl.species`.
    n_B : float
        Total baryon density [fm^-3].
    T, SnB : float
        Temperature [MeV], or entropy per baryon in its place -- exactly one
        of the two. SnB costs an outer 1-D solve for T, every evaluation of
        which is a full equilibrium solve.
    x0 : list
        Optional starting guess, normally a neighbouring point's
        `eos.enjl.solver.warm_start`. Its ORDER DEPENDS ON THE MODE:
        `eos.enjl.solver.unknown_slots(spec)` names the slots for the
        `spec` a mode builds, and returns the ten of
        `eos.enjl.solver.BASE_UNKNOWNS` -- M_u, M_d, M_s, mu_B, mu_C,
        n_bQ, gomega_omega, grho_rho, SigmaR_b, SigmaR_q -- followed by
        one potential per held charge: mu_S where the mode fixes Y_S,
        then mu_nue where it fixes Y_Le. So a vector written for
        beta_eq_neutrinoless is ten long and is the WRONG LENGTH for
        fixed_YC_YS or the trapped mode; build it from `unknown_slots`
        rather than by counting. Above a first-order transition the model
        has more than one root and the guess decides which one is found;
        without it the parameter-free cold starts are used, and those
        stop converging around n_B = 0.5 fm^-3.

    Returns
    -------
    PointResult -- test `.ok` before using `.point`.
    """
    T, SnB, species = _check_call(mode, species, T, SnB, conditions,
                                  leptons=leptons)
    if n_B is None:
        raise ValueError("n_B is required")
    if n_B <= 0.0:
        raise ValueError(f"n_B must be positive, got {n_B}")

    common = dict(x0=x0, cold_start=x0 is None, leptons=leptons,
                  species=species)
    try:
        if SnB is None:
            point = solve(par, mode, n_B, T=T, **common, **conditions)
        else:
            point = solve_at_entropy(par, mode, n_B, SnB,
                                     **common, **conditions)
    except RuntimeError as err:
        return PointResult(False, str(err))
    return PointResult(True, "converged", point)


def eos_table(par, mode, species=None, axes=None,
              direction="up", x0=None, leptons=None, rows=False,
              progress=None, verbose=False, coexistences=None,
              eta=1.0):
    """A solved grid along the density axis, following one branch.

    axes = {'nB': grid, 'T': [T]}, or 'SnB' in place of 'T' for an isentrope;
    the thermal axis may be omitted, in which case T = 0 is understood. Each
    axis carries ONE value: a table here is a density continuation and that is
    what carries the sweep, so a second temperature or a second fraction is a
    second call, not a second column. The result feeds `eos.astro.tov` and the
    plotting code directly.

    Each point is warm-started from its neighbour, and `direction` selects
    which branch is followed -- "up" from the low-density chirally broken
    side, "down" back from a deconfined guess at the top of the grid. Where
    several branches exist the two differ, and choosing between them is a
    Maxwell construction this engine does not perform; see `eos.enjl.table`.

    Above a first-order transition the returned branch may violate
    dP/dn_B >= 0. That is real physics -- a mechanically unstable branch -- and
    a table in that state must be resolved by a construction before it reaches
    a structure solver.

    `coexistences` is what performs that resolution. Given the located
    transitions -- from `eos.mixed.construction`, which is a composite engine
    and so cannot be reached from inside this model (CLAUDE.md section 1) --
    this returns the DELIVERED table instead: both
    branches swept, the stable one kept at each density, and each window
    replaced by its constant-pressure segment. `direction` is then unused,
    because a construction needs both. Without it the raw continuation is
    returned exactly as before, which is what `test/baseline` freezes.
    `eta` selects the construction and only eta = 1 (Maxwell) is implemented.

    **Which locator depends on the mode**, because a phase must be closed
    before two of them can be equated: `enjl_coexistences` closes each phase
    by neutrality and serves the beta-equilibrium modes, while
    `enjl_composition_coexistences` closes it at a held, leptonless (Y_C, Y_S)
    and serves `fixed_YC_YS` with `leptons=False`. Passing a window located
    under the wrong closure delivers a plateau at the wrong densities; the
    windows and the table have to be built for the same mode.

    An EMPTY list is legal and asserts that this grid holds no transition.
    Where that is false the table keeps the lower-eps branch across the
    crossing, which drops P there, and the returned `ConstructedTable`
    reports `deliverable = False` with `defect` naming the density. Test it
    before handing the table to a structure solver -- `rows=True` returns the
    rows alone and drops the status along with `windows`, so a caller who
    needs either asks for the object.

    progress : callable, invoked once per completed line, with the dict every
        table builder in this repository reports. verbose=True installs the
        built-in printer.
    """
    check_mode(mode)
    axes = dict(axes or {})
    nB = np.atleast_1d(np.asarray(axes.pop("nB"), dtype=float))
    entropies = axes.pop("SnB", None)
    temps = np.atleast_1d(np.asarray(axes.pop("T", [0.0]), dtype=float))
    wanted = set(MODE_FRACTIONS[mode])
    fractions = {k: axes.pop(k) for k in list(axes) if k in wanted}
    if axes:
        raise ValueError(f"mode {mode!r} takes axes nB, T (or SnB) and "
                         f"{sorted(wanted)}; got {sorted(axes)} as well")
    for key, value in list(fractions.items()):
        grid = np.atleast_1d(np.asarray(value, dtype=float))
        if len(grid) != 1:
            raise NotImplementedError(
                f"eos.enjl sweeps one fraction combination per table; the "
                f"{key} axis has {len(grid)} values. Call eos_table once per "
                f"value -- the density continuation is what carries a sweep "
                f"here, and a fraction axis would restart it")
        fractions[key] = float(grid[0])
    SnB = None
    if entropies is not None:
        grid = np.atleast_1d(np.asarray(entropies, dtype=float))
        if len(grid) != 1:
            raise NotImplementedError(
                f"eos.enjl sweeps one thermal value per table; the SnB axis "
                f"has {len(grid)} values. Call eos_table once per value -- "
                f"the density continuation is what carries a sweep here, and "
                f"a second thermal value would restart it")
        SnB = float(grid[0])
    if len(temps) != 1:
        raise NotImplementedError(
            f"eos.enjl sweeps one thermal value per table; the T axis has "
            f"{len(temps)} values. Call eos_table once per temperature -- the "
            f"density continuation is what carries a sweep here, and a second "
            f"temperature would restart it")
    T, SnB, species = _check_call(mode, species, float(temps[0]), SnB,
                                  fractions, leptons=leptons)
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, "
                         f"got {direction!r}")

    table_spec = TableSpec(nB=nB, mode=mode, par=par, direction=direction,
                           T=T, SnB=SnB, x0=x0, leptons=leptons,
                           species=species, fractions=fractions)
    if coexistences is not None:
        built = build_constructed_table(table_spec, coexistences, eta=eta,
                                        progress=progress, verbose=verbose)
        return built.rows if rows else built

    result = build_table(table_spec, progress=progress, verbose=verbose)
    if rows:
        return [beta_row(p) for p in result.points]
    return result


#: The freezes eos_response implements: none. See below.
RESPONSE_FREEZES = ()


def eos_response(par, mode, species=None,
                 frozen="equilibrium", n_B=None, T=0.0, **conditions):
    """Second-derivative quantities -- not implemented for this model.

    The CompOSE list divides in two here, and neither half is reachable yet.
    The heat capacities C_V and C_P, the thermal index and the difference
    between the isothermal and adiabatic sound speeds all need T > 0, and this
    is a T = 0 model. What is left -- c_s^2 and the susceptibilities
    chi_ab = dn_a/dmu_b -- needs a statement about which branch the derivative
    is taken along, and above the model's first first-order transition more
    than one branch satisfies the equilibrium conditions at the same density.
    Differentiating along the branch a continuation happened to reach would
    return a number whose meaning depends on the direction the table was swept
    in, which is worse than returning nothing.

    Raises NotImplementedError, naming both reasons.
    """
    raise NotImplementedError(
        "eos.enjl does not implement eos_response. The heat capacities, the "
        "thermal index and the isothermal/adiabatic distinction need T > 0 "
        "and this is a T = 0 model; c_s^2 and chi_ab = dn_a/dmu_b need the "
        "branch the derivative is taken along to be settled, and above the "
        "model's first first-order transition more than one branch satisfies "
        "the equilibrium conditions at the same density")


__all__ = ["MODE_FRACTIONS", "Parameters", "PointResult", "RESPONSE_FREEZES",
           "SpeciesFlags", "eos_point", "eos_response", "eos_table"]
