"""The uniform model API for ENJL: eos_point, eos_table, eos_response.

Every model in this repository exposes these three entry points with the same
shape, so a caller -- a table pipeline, a sampler, a downstream package --
drives any model the same way:

    eos_point(par, mode, species, n_B=..., T=0.0)   one state
    eos_table(par, mode, species, axes)             a solved grid
    eos_response(par, mode, species, ...)           second derivatives

All four modes of CLAUDE.md section 3 are closed here, at T = 0; what this
model refuses is a temperature, not a mode. Any T > 0 raises, because every
kinetic expression in the model is a zero-temperature closed form.

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
)
from eos.enjl.species import SpeciesFlags
from eos.enjl.table import DIRECTIONS, TableSpec, beta_row, build_table


@dataclass(frozen=True)
class PointResult:
    """One eos_point outcome: a convergence status the caller can test.

    When `ok` is False, `point` is None and `message` says what the request
    reached -- typically the best scaled residual of the starting points tried.
    """
    ok: bool
    message: str
    point: object = None


def _check_call(mode, species, T, SnB, conditions, leptons=True):
    """Raise unless the request is one this model can be asked at all.

    `mode_spec` validates the fractions against the mode, so a missing or
    surplus condition is refused before any work rather than defaulted.
    """
    check_mode(mode)
    mode_spec(mode, leptons=leptons, **conditions)
    if SnB is not None:
        if T is not None:
            raise ValueError("give exactly one of T / SnB")
        if SnB != 0.0:
            raise NotImplementedError(
                f"eos.enjl is a T = 0 model, where s = 0 identically and the "
                f"only entropy per baryon it reaches is zero; got "
                f"SnB = {SnB}")
        T = 0.0
    check_temperature(0.0 if T is None else T)
    SpeciesFlags() if species is None else species   # flags validate on init
    return 0.0


def eos_point(par, mode="beta_eq_neutrinoless", species=None, n_B=None,
              T=0.0, SnB=None, x0=None, leptons=True, **conditions):
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
        Temperature [MeV] or entropy per baryon; both are zero, and any other
        value raises rather than being quietly rounded to the model's one
        temperature.
    x0 : list
        Optional starting guess in the order of `eos.enjl.solver.UNKNOWNS`,
        normally a neighbouring point's `warm_start`. Above a first-order
        transition the model has more than one root and the guess decides
        which one is found; without it the parameter-free cold starts are
        used, and those stop converging around n_B = 0.5 fm^-3.

    Returns
    -------
    PointResult -- test `.ok` before using `.point`.
    """
    T = _check_call(mode, species, T, SnB, conditions, leptons=leptons)
    if n_B is None:
        raise ValueError("n_B is required")
    if n_B <= 0.0:
        raise ValueError(f"n_B must be positive, got {n_B}")

    try:
        point = solve(mode, n_B, par=par, x0=x0, cold_start=x0 is None,
                      leptons=leptons, T=T, **conditions)
    except RuntimeError as err:
        return PointResult(False, str(err))
    return PointResult(True, "converged", point)


def eos_table(par, mode="beta_eq_neutrinoless", species=None, axes=None,
              direction="up", x0=None, leptons=True, rows=False,
              progress=None, verbose=False):
    """A solved grid along the density axis, following one branch.

    axes = {'nB': grid, 'T': [0.0]}; the temperature axis may be omitted, in
    which case T = 0 is understood, since it is the only value the model has.
    The result feeds `eos.astro.tov` and the plotting code directly.

    Each point is warm-started from its neighbour, and `direction` selects
    which branch is followed -- "up" from the low-density chirally broken
    side, "down" back from a deconfined guess at the top of the grid. Where
    several branches exist the two differ, and choosing between them is a
    Maxwell construction this engine does not perform; see `eos.enjl.table`.

    Above a first-order transition the returned branch may violate
    dP/dn_B >= 0. That is real physics -- a mechanically unstable branch -- and
    a table in that state must be resolved by a construction before it reaches
    a structure solver.

    progress : callable, invoked once per completed line, with the dict every
        table builder in this repository reports. verbose=True installs the
        built-in printer.
    """
    check_mode(mode)
    axes = dict(axes or {})
    nB = np.atleast_1d(np.asarray(axes.pop("nB"), dtype=float))
    temps = np.atleast_1d(np.asarray(axes.pop("T", [0.0]), dtype=float))
    wanted = set(MODE_FRACTIONS[mode])
    fractions = {k: axes.pop(k) for k in list(axes) if k in wanted}
    if axes:
        raise ValueError(f"mode {mode!r} takes axes nB, T and "
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
    if len(temps) != 1:
        raise NotImplementedError(
            f"eos.enjl has one temperature, T = 0; got a temperature axis of "
            f"{len(temps)} values")
    T = _check_call(mode, species, float(temps[0]), None, fractions,
                    leptons=leptons)
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, "
                         f"got {direction!r}")

    result = build_table(
        TableSpec(nB=nB, mode=mode, par=par, direction=direction, T=T, x0=x0,
                  leptons=leptons, fractions=fractions),
        progress=progress, verbose=verbose)
    if rows:
        return [beta_row(p) for p in result.points]
    return result


#: The freezes eos_response implements: none. See below.
RESPONSE_FREEZES = ()


def eos_response(par, mode="beta_eq_neutrinoless", species=None,
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
