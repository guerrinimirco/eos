"""The uniform model API for the hadron-quark mixed phase: eos_point,
eos_table, eos_response.

Every engine in this repository exposes these three entry points with the same
shape, so a caller -- a table pipeline, a sampler, a downstream package --
drives the mixed phase exactly the way it drives a single-phase model:

    eos_point(par, mode, species, n_B=..., T=..., eta=...)   one state
    eos_table(par, mode, species, axes, eta=...)             a solved grid
    eos_response(par, mode, species, frozen=..., ...)        second derivatives

Modes are the repository's named equilibria: 'beta_eq_neutrinoless',
'beta_eq_neutrino_trapped', 'fixed_YC', 'fixed_YC_YS'. Conditions are named
exactly n_B, T (or SnB), Y_C, Y_S, Y_Le. Units at this boundary are fm-based:
n_B in fm^-3, T and potentials in MeV, eps and P in MeV/fm^3.

What a composite engine adds to the three entry points
------------------------------------------------------
Two arguments and one extra result, all of them the transition itself:

`eta` in [0, 1] chooses how much of electric-charge neutrality is imposed
phase by phase rather than on the mixture as a whole -- eta = 0 is the Gibbs
construction, eta = 1 the Maxwell one, and intermediate values stand in for
the surface tension and Coulomb cost of the mixed-phase structures. It is a
scalar per call, not an axis, because it changes the shape of the unknown
vector.

`vmit_params` is the quark phase's parameter set, alongside `par` for the
hadronic one: a hybrid equation of state has two models in it and both sets of
parameters are arguments (CLAUDE.md section 6).

And `eos_table` returns `(rows, windows)`, not just rows. The phase boundaries
found on each line are part of the answer -- they are what the transition
curves n_onset(T) and n_offset(T) are made of -- so they are returned, not
recovered afterwards by scanning the rows for where chi left [0, 1].

NON-CONVERGENCE IS A RETURN VALUE here, not an exception: a sampler walks into
unphysical corners constantly and must be able to score the point and move on.
A malformed CALL is different -- an unknown mode, a fraction the mode does not
take -- and raises before any solve, because it is a programming error that
would otherwise be repeated a million times in silence.
"""
from dataclasses import dataclass

import numpy as np

from eos.general.tabulate import temperature_at_entropy
from eos.dd2.species import SpeciesFlags
from eos.mixed.coefficients import sound_speed_frozen
from eos.mixed.equilibrium.residual import mixed_slots
from eos.mixed.solvers.point import MixedResult, solve_mixed
from eos.mixed.tables.generate import (
    MODE_FRACTIONS, MixedTableSpec, build_mixed_table, make_charge_spec,
)

#: The electron lepton fraction has two names. CLAUDE.md section 5 calls it
#: Y_Le; `eos.mixed` predates that rule and calls it Y_L throughout its
#: `ChargeSpec`, its `MODE_FRACTIONS` and its table axes. The public boundary
#: takes the spec name and translates here, in the one place that has to know.
_YLE_SPEC = "Y_Le"
_YLE_ENGINE = "Y_L"


@dataclass(frozen=True)
class PointResult:
    """One eos_point outcome: a convergence status the caller can test.

    `ok` is judged on the largest mixed-residual component (see
    `eos.mixed.solvers.point.RESIDUAL_TOL`); when it is False, `point` is None
    and `message` says what went wrong. A converged point may still be OUTSIDE
    the coexistence window -- test `.point.phase`, which is 'H', 'mix' or 'Q'.
    """
    ok: bool
    message: str
    point: MixedResult = None


def _to_engine_names(named):
    """Spec condition names translated to the engine's own, or raise.

    The internal name is REJECTED here rather than quietly accepted as a
    second spelling: one quantity with two accepted names at one boundary is
    exactly the drift the uniform API exists to stop.
    """
    if _YLE_ENGINE in named:
        raise ValueError(f"{_YLE_ENGINE!r} is the engine's internal name for "
                         f"the electron lepton fraction; this boundary takes "
                         f"{_YLE_SPEC!r}")
    out = {}
    for key, value in named.items():
        out[_YLE_ENGINE if key == _YLE_SPEC else key] = value
    return out


def _engine_fractions(mode, conditions):
    """The mode's fractions in the engine's own names, or raise.

    Raises unless exactly the fractions this mode takes were supplied, so a
    misspelled or surplus condition is a loud error at the call rather than a
    silently ignored argument.
    """
    if mode not in MODE_FRACTIONS:
        raise ValueError(f"unknown mode {mode!r}; expected one of "
                         f"{list(MODE_FRACTIONS)}")
    if "Y_Lmu" in conditions:
        raise NotImplementedError(
            "the mixed engine tracks one trapped lepton family; "
            "beta_eq_neutrino_trapped takes (n_B, Y_Le, T) only")
    fracs = _to_engine_names(conditions)

    def spec_name(key):
        return _YLE_SPEC if key == _YLE_ENGINE else key

    wanted = set(MODE_FRACTIONS[mode])
    given = set(fracs)
    missing = wanted - given
    if missing:
        raise ValueError(f"mode {mode!r} needs "
                         f"{sorted(spec_name(m) for m in missing)}")
    extra = given - wanted
    if extra:
        raise ValueError(f"mode {mode!r} does not take "
                         f"{sorted(spec_name(e) for e in extra)}")
    return fracs


def eos_point(par, mode, species=None, n_B=None, T=None, SnB=None, eta=0.0,
              vmit_params=None, leptons=True, x0=None, analytic_jac=False,
              check_consistency=True, **conditions):
    """One solved mixed-phase state; non-convergence is a return value.

    Parameters
    ----------
    par : dd2 Parametrization
        The hadronic parameters -- always an argument, never module state.
    mode : str
        One of the keys of `MODE_FRACTIONS`.
    species : dd2 SpeciesFlags
        The active degrees of freedom of the hadronic phase.
    n_B : float
        TOTAL baryon density of the mixture [fm^-3], i.e. volume-averaged over
        both phases -- not the density of either one.
    T, SnB : float
        Exactly one of temperature [MeV] or entropy per baryon; SnB adds an
        outer 1-D solve for T.
    eta : float
        Local-neutrality fraction in [0, 1]; 0 Gibbs, 1 Maxwell.
    vmit_params : VMITParams
        The quark phase's parameters; the published default if omitted.
    leptons : bool
        For the fixed-fraction modes: whether neutralizing leptons are present.
    x0 : sequence
        Warm start in the slot order of `mixed_slots(spec, eta, species)`.
    conditions : the fractions the mode fixes (Y_C, Y_S, Y_Le).

    Returns
    -------
    PointResult -- test `.ok` before using `.point`. A converged result whose
    chi lies outside [0, 1] is a pure phase, not a failure: that is how the
    engine reports which side of the transition the density is on.
    """
    if species is None:
        species = SpeciesFlags()
    if n_B is None:
        raise ValueError("n_B is required")
    if n_B <= 0.0:
        raise ValueError(f"n_B must be positive, got {n_B}")
    if (T is None) == (SnB is None):
        raise ValueError("exactly one of T / SnB must be given")
    if not 0.0 <= eta <= 1.0:
        raise ValueError(f"eta must lie in [0, 1], got {eta}")
    fracs = _engine_fractions(mode, conditions)      # caller errors raise here
    spec = make_charge_spec(mode, fracs, leptons=leptons)

    def solve(temperature):
        return solve_mixed(par, species, float(n_B), float(eta), spec,
                           vmit_params=vmit_params, T=float(temperature),
                           x0=x0, analytic_jac=analytic_jac,
                           check_consistency=check_consistency)

    try:
        if SnB is not None:
            T = temperature_at_entropy(lambda t: solve(t).s / n_B, float(SnB))
        point = solve(T)
    except NotImplementedError:
        raise                       # an unwired request must never be a status
    except (RuntimeError, ValueError) as err:
        return PointResult(False, str(err))
    return PointResult(True, "converged", point)


def eos_table(par, mode, species=None, axes=None, eta=0.0, fixed=None,
              leptons=True, vmit_params=None, window_only=True,
              analytic_jac=False, progress=None, verbose=False):
    """A solved grid over {n_B} x {T or SnB} [x fraction axes], with the phase
    boundaries found on each line.

    Returns `(rows, windows)`: the long-format rows, and the `MixedWindow` per
    (temperature, fraction) line. A mixed table is rows plus windows, and the
    windows are part of the result rather than something the caller recovers by
    scanning for where chi crosses 0 and 1 (CLAUDE.md section 5).

    axes follows `MixedTableSpec`: {'nB': grid, exactly one of 'T'/'SnB': grid,
    and optionally any of 'Y_C'/'Y_S'/'Y_Le' to sweep that fraction}. The
    density axis is warm-started within each line.

    window_only=True (the default) solves the mixed system only between the
    located boundaries, where it is the only thing that can answer; outside, the
    far cheaper pure-phase solvers give the same state. Set it False to solve at
    every grid point, e.g. when studying chi outside [0, 1].

    progress : callable, invoked once per completed line with a dict
        {mode, line, n_lines, temp_key, temp, fracs, n_solved, n_requested,
        elapsed_s} plus the mixed extras {eta, window}. verbose=True installs
        the built-in one-line printer.
    """
    if species is None:
        species = SpeciesFlags()
    if not 0.0 <= eta <= 1.0:
        raise ValueError(f"eta must lie in [0, 1], got {eta}")
    axes = _to_engine_names(axes or {})
    fixed = _to_engine_names(fixed or {})
    spec = MixedTableSpec(par=par, flags=species, mode=mode, axes=axes,
                          eta=float(eta), vmit_params=vmit_params, fixed=fixed,
                          leptons=leptons, window_only=window_only,
                          analytic_jac=analytic_jac)
    return build_mixed_table(spec, progress=progress, verbose=verbose)


#: The freezes eos_response implements. Each answers "what is held fixed while
#: the derivative is taken", which is what encodes which reactions are faster
#: than the perturbation:
#:
#:   'equilibrium' -- nothing is held: chi and both compositions readjust, so
#:       the mixture converts one phase into the other rather than stiffening.
#:       This is the sound speed the TOV equations take.
#:   'chi' -- the quark volume fraction is held, and with it each phase's Y_C
#:       and Y_S; the individual hadronic species still re-equilibrate within
#:       them. The mixture is compressed faster than the phases can convert.
#:
#: The remaining spec freezes -- frozen per-species composition, and frozen
#: conserved fractions with chi free -- are recorded in docs/DEFERRED.md.
RESPONSE_FREEZES = ("equilibrium", "chi")


def eos_response(par, mode, species=None, frozen="equilibrium", n_B=None,
                 T=0.0, eta=0.0, vmit_params=None, leptons=True, rel_dn=1e-3,
                 **conditions):
    """Second-derivative quantities at one mixed-phase state.

    frozen='equilibrium' -- everything re-equilibrates under the perturbation.
        The derivatives are taken along the mode's own sequence by central
        differences over a relative density step `rel_dn`:

            cs2_eq = dP/deps   at fixed T
            C_V    = (T/n_B) ds/dT   at fixed n_B, returned only at T > 0

        Through a Maxwell window (eta = 1) the pressure is constant along that
        sequence, so cs2_eq goes to zero. That is the physics, not a failure.

        ONLY DEFINED INSIDE THE COEXISTENCE WINDOW. Outside it the mixed system
        still has a root -- chi runs negative or past one -- but that root is
        an analytic continuation, not the state: at eta = 1 it sits on the
        pressure plateau at every density, so differentiating it returns zero
        for matter whose real sound speed is not small. Both quantities are
        therefore nan when the point is a pure phase, and the equilibrium
        response there comes from that phase's own `eos_response` --
        `eos.dd2.api` below the onset, `eos.vmit.api` above the offset.

    frozen='chi' -- the quark volume fraction is held fixed, and with it each
        phase's charge and strangeness fractions; `leptons` chooses whether the
        mixture is re-neutralised under the perturbation. Delegates to
        `eos.mixed.coefficients.sound_speed_frozen`, whose module docstring
        states the convention in full -- read it before comparing the number
        with one from elsewhere. Returns cs2_frozen. Defined at every density:
        outside the window chi is clipped and the answer is the pure phase's
        own frozen sound speed, so the curve joins continuously at the
        boundaries.

    Every result also carries `chi` and `phase` ('H', 'mix' or 'Q'), because in
    a composite engine which regime the point is in decides which numbers in
    the dict mean anything.

    Returns a dict of the computed quantities; raises NotImplementedError,
    naming the gap, for freezes not yet wired.
    """
    if frozen not in RESPONSE_FREEZES:
        raise NotImplementedError(
            f"frozen={frozen!r} is not wired for the mixed engine; "
            f"implemented: {RESPONSE_FREEZES} (see docs/DEFERRED.md)")
    if species is None:
        species = SpeciesFlags()
    if n_B is None:
        raise ValueError("n_B is required")
    fracs = _engine_fractions(mode, conditions)
    spec = make_charge_spec(mode, fracs, leptons=leptons)
    slots = mixed_slots(spec, float(eta), species)

    centre = eos_point(par, mode, species, n_B=n_B, T=T, eta=eta,
                       vmit_params=vmit_params, leptons=leptons, **conditions)
    if not centre.ok:
        raise RuntimeError(f"eos_response could not solve its central point "
                           f"(n_B={n_B}, T={T}, eta={eta}): {centre.message}")

    point = centre.point
    out = {"chi": point.chi, "phase": point.phase}

    if frozen == "chi":
        out["cs2_frozen"] = sound_speed_frozen(
            par, species, point, vmit_params=vmit_params, rel_dn=rel_dn,
            leptons=leptons)
        return out

    if not point.in_mixed_phase:
        # The mixed root outside the window is a continuation, not the state;
        # see the docstring. Say nan rather than differentiate it.
        out["cs2_eq"] = np.nan
        if T > 0.0:
            out["C_V"] = np.nan
        return out

    # The centre point's converged potentials seed both stencil points, which
    # is what keeps a step across a steep window on the same root.
    x0 = [point.potentials[name] for name in slots]

    def state(n, temperature):
        result = eos_point(par, mode, species, n_B=n, T=temperature, eta=eta,
                           vmit_params=vmit_params, leptons=leptons, x0=x0,
                           **conditions)
        if not result.ok:
            raise RuntimeError(
                f"eos_response could not solve its stencil point "
                f"(n_B={n}, T={temperature}): {result.message}")
        return result.point

    dn = rel_dn * n_B
    lo, hi = state(n_B - dn, T), state(n_B + dn, T)
    out["cs2_eq"] = ((hi.P - lo.P) / (hi.eps - lo.eps)
                     if hi.eps != lo.eps else np.nan)

    if T > 0.0:
        dT = rel_dn * T
        cold, hot = state(n_B, T - dT), state(n_B, T + dT)
        out["C_V"] = T * (hot.s - cold.s) / (2.0 * dT) / n_B
    return out
