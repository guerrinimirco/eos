"""The uniform model API for the hadron-quark mixed phase: eos_point,
eos_table, eos_response.

Every engine in this repository exposes these three entry points with the same
shape, so a caller -- a table pipeline, a sampler, a downstream package --
drives the mixed phase exactly the way it drives a single-phase model:

    eos_point(phases, mode, species, n_B=..., T=..., eta=...)  one state
    eos_table(phases, mode, species, axes, eta=...)            a solved grid
    eos_response(phases, mode, species, frozen=..., ...)       2nd derivatives

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

`phases` is the pairing, and it is THE parameter argument of this engine
(CLAUDE.md section 5): two `Phase` objects from `eos.mixed.adapters`, each
closing over its own model's parameters, so a hybrid's two parameter sets are
arguments (section 6) without either model taking a privileged position in a
signature. `adapters.default_pair(par, flags, vmit_params)` builds the DD2 +
vMIT pairing; `(sfho_phase(...), njl_phase(...))` builds another, and the two
are written the same way.

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

from eos.general.tabulate import (temperature_at_entropy,
                                  unconverged_response)
from eos.mixed.species import SpeciesFlags
from eos.mixed.responses import sound_speed_frozen
from eos.mixed.solver import mixed_slots
from eos.mixed.solver import Result, solve
from eos.mixed.hybrid import (EoSTable, build_hybrid_table,
                              validate_wings)
from eos.mixed.table import (
    MODE_FRACTIONS, TableSpec, build_table, make_charge_spec,
)

@dataclass(frozen=True)
class PointResult:
    """One eos_point outcome: a convergence status the caller can test.

    `ok` is judged on the largest mixed-residual component (see
    `eos.mixed.solver.RESIDUAL_TOL`); when it is False, `point` is None
    and `message` says what went wrong. A converged point may still be OUTSIDE
    the coexistence window -- test `.point.phase`, which is 'H', 'mix' or 'Q'.
    """
    ok: bool
    message: str
    point: Result = None


def _engine_fractions(mode, conditions):
    """The mode's fractions, checked against what it takes, or raise.

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
    fracs = dict(conditions)
    wanted = set(MODE_FRACTIONS[mode])
    given = set(fracs)
    missing = wanted - given
    if missing:
        raise ValueError(f"mode {mode!r} needs {sorted(missing)}")
    extra = given - wanted
    if extra:
        raise ValueError(f"mode {mode!r} does not take {sorted(extra)}")
    return fracs


def eos_point(phases, mode, species=None, n_B=None, T=None, SnB=None,
              eta=0.0, leptons=None, x0=None, analytic_jac=False,
              check_consistency=True, **conditions):
    """One solved mixed-phase state; non-convergence is a return value.

    Parameters
    ----------
    phases : (Phase, Phase)
        The pairing (see `eos.mixed.adapters`): two `Phase` objects, each
        closing over its own model's parameters. This is the engine's
        parameter argument (CLAUDE.md section 5) -- always an argument, never
        module state. `adapters.default_pair(par, flags, vmit_params)` is the
        DD2 + vMIT one.
    mode : str
        One of the keys of `MODE_FRACTIONS`.
    species : SpeciesFlags
        The active degrees of freedom: `eos.mixed.species.SpeciesFlags`,
        the engine's own set (CLAUDE.md section 4), every sector off when
        omitted. The per-phase sectors reach the models through their
        `Phase`; the phase-common ones — photons above all — are consumed
        once at the mixture level.
    n_B : float
        TOTAL baryon density of the mixture [fm^-3], i.e. volume-averaged over
        both phases -- not the density of either one.
    T, SnB : float
        Exactly one of temperature [MeV] or entropy per baryon; SnB adds an
        outer 1-D solve for T.
    eta : float
        Local-neutrality fraction in [0, 1]; 0 Gibbs, 1 Maxwell.
    leptons : bool
        For the fixed-fraction modes: whether neutralizing leptons are present.
    x0 : sequence
        Warm start in the slot order of `mixed_slots(spec, eta, phases)`.
    conditions : the fractions the mode fixes (Y_C, Y_S, Y_Le).

    Returns
    -------
    PointResult -- test `.ok` before using `.point`. A converged result whose
    chi lies outside [0, 1] is a pure phase, not a failure: that is how the
    engine reports which side of the transition the density is on.
    """
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

    def point_at(temperature):
        return solve(phases, float(n_B), float(eta), spec,
                     T=float(temperature), x0=x0, analytic_jac=analytic_jac,
                     check_consistency=check_consistency, species=species)

    try:
        if SnB is not None:
            T = temperature_at_entropy(lambda t: point_at(t).s / n_B, float(SnB))
        point = point_at(T)
    except NotImplementedError:
        raise                       # an unwired request must never be a status
    except (RuntimeError, ValueError) as err:
        return PointResult(False, str(err))
    return PointResult(True, "converged", point)


def eos_table(phases, mode, species=None, axes=None, eta=0.0, fixed=None,
              leptons=None, window_only=True,
              analytic_jac=False, refine="exact",
              progress=None, verbose=False):
    """A solved grid over {n_B} x {T or SnB} [x fraction axes], with the phase
    boundaries found on each line.

    Returns `(rows, windows)`: the long-format rows, and the `Window` per
    (temperature, fraction) line. A mixed table is rows plus windows, and the
    windows are part of the result rather than something the caller recovers by
    scanning for where chi crosses 0 and 1 (CLAUDE.md section 5).

    axes follows `TableSpec`: {'nB': grid, exactly one of 'T'/'SnB': grid,
    and optionally any of 'Y_C'/'Y_S'/'Y_Le' to sweep that fraction}. The
    density axis is warm-started within each line.

    window_only=True (the default) solves the mixed system only between the
    located boundaries, where it is the only thing that can answer; outside, the
    far cheaper pure-phase solvers give the same state. Set it False to solve at
    every grid point, e.g. when studying chi outside [0, 1].

    refine chooses the boundary resolution: "exact" (the default) finishes
    each boundary with one fixed-chi solve and marches those solves along the
    temperature axis, so the windows carry the chi crossings themselves;
    "bisect" stops at half a grid spacing.

    progress : callable, invoked once per completed line with a dict
        {mode, line, n_lines, temp_key, temp, fracs, n_solved, n_requested,
        elapsed_s} plus the mixed extras {eta, window}. verbose=True installs
        the built-in one-line printer.
    """
    if not 0.0 <= eta <= 1.0:
        raise ValueError(f"eta must lie in [0, 1], got {eta}")
    axes = dict(axes or {})
    fixed = dict(fixed or {})
    spec = TableSpec(phases=phases, mode=mode, axes=axes,
                          eta=float(eta), fixed=fixed,
                          leptons=leptons, window_only=window_only,
                          analytic_jac=analytic_jac, refine=refine,
                          species=species)
    return build_table(spec, progress=progress, verbose=verbose)


@dataclass(frozen=True)
class HybridResult:
    """One hybrid_table outcome: a convergence status the caller can test.

    When `ok` is False, `table` is None and `message` says what went wrong —
    a boundary that could not be located, or a pressure inversion too large to
    be round-off (an unresolved mechanical instability, CLAUDE.md section 8).
    """
    ok: bool
    message: str
    table: EoSTable = None


def hybrid_table(phases, mode, species=None, n_B_grid=None, eta=0.0, T=0.0,
                 leptons=None, window=None,
                 analytic_jac=False, **conditions):
    """The stitched hadronic + mixed + quark core EoS at ONE equilibrium.

    The mode holds in all three segments: the wings are the pure models' own
    per-mode solves and the window is the eta-mixed phase, so a fixed-Y_C
    hybrid is fixed-Y_C everywhere, not a fixed-Y_C window between
    beta-equilibrium wings. Only eta is specific to the mixed region.

    Parameters are those of `eos_point`, with `n_B_grid` in place of `n_B`;
    `window` optionally reuses an already-located `Window`. If there is
    no transition on the grid the table is pure hadronic.

    Returns a `HybridResult` — test `.ok` before using `.table`, whose
    `.to_tov()` is the contract into `eos.astro.tov` (§6: non-convergence is a
    return value). A malformed call — an unknown mode, a fraction the mode
    does not take, a species flag the mode needs — raises before any solve.
    """
    if n_B_grid is None:
        raise ValueError("n_B_grid is required")
    if not 0.0 <= eta <= 1.0:
        raise ValueError(f"eta must lie in [0, 1], got {eta}")
    fracs = _engine_fractions(mode, conditions)      # caller errors raise here
    spec = make_charge_spec(mode, fracs, leptons=leptons)
    # A wing a phase cannot solve at this mode is a MALFORMED CALL, so it
    # raises here, outside the try below that turns non-convergence into a
    # status: each adapter validates the spec against its own flags (the DD2
    # one refuses a trapped spec carrying no neutrino population).
    validate_wings(phases, spec, T)
    try:
        table = build_hybrid_table(phases, n_B_grid, eta, spec, T=T,
                                   analytic_jac=analytic_jac,
                                   window=window, species=species)
    except NotImplementedError:
        raise                       # an unwired request must never be a status
    except (RuntimeError, ValueError) as err:
        return HybridResult(False, str(err))
    return HybridResult(True, "ok", table)


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


def eos_response(phases, mode, species=None, frozen="equilibrium", n_B=None,
                 T=0.0, eta=0.0, leptons=None, rel_dn=1e-3, **conditions):
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
        `eos.mixed.responses.sound_speed_frozen`, whose module docstring
        states the convention in full -- read it before comparing the number
        with one from elsewhere. Returns cs2_frozen. Defined at every density:
        outside the window chi is clipped and the answer is the pure phase's
        own frozen sound speed, so the curve joins continuously at the
        boundaries.

    Every result also carries `chi` and `phase` ('H', 'mix' or 'Q'), because in
    a composite engine which regime the point is in decides which numbers in
    the dict mean anything.

    Returns a dict of the computed quantities, plus `converged` and `reason`.
    A point the engine cannot solve -- the central one or either stencil
    neighbour -- is NOT an exception: the same dict comes back with
    converged=False and nan in every quantity, so a sampler can score the
    point and move on (CLAUDE.md section 6). Note that a nan `cs2_eq` at
    converged=True is a different statement: it means the point is a pure
    phase, where the equilibrium response is that phase's own.
    Raises NotImplementedError, naming the gap, for freezes not yet wired.
    """
    if frozen not in RESPONSE_FREEZES:
        raise NotImplementedError(
            f"frozen={frozen!r} is not wired for the mixed engine; "
            f"implemented: {RESPONSE_FREEZES} (see docs/DEFERRED.md)")
    if n_B is None:
        raise ValueError("n_B is required")
    fracs = _engine_fractions(mode, conditions)
    spec = make_charge_spec(mode, fracs, leptons=leptons)
    slots = mixed_slots(spec, float(eta), phases)

    if frozen == "chi":
        names = ("chi", "cs2_frozen")
    elif T > 0.0:
        names = ("chi", "cs2_eq", "C_V")
    else:
        names = ("chi", "cs2_eq")

    centre = eos_point(phases, mode, species, n_B=n_B, T=T, eta=eta,
                       leptons=leptons, **conditions)
    if not centre.ok:
        out = unconverged_response(
            f"eos_response could not solve its central point "
            f"(n_B={n_B}, T={T}, eta={eta}): {centre.message}", names)
        out["phase"] = None      # not a quantity: which regime is unknown
        return out

    point = centre.point
    out = {"chi": point.chi, "phase": point.phase}

    if frozen == "chi":
        out["cs2_frozen"] = sound_speed_frozen(
            phases, point, rel_dn=rel_dn, leptons=leptons, species=species)
        out["converged"] = True
        out["reason"] = "converged"
        return out

    if not point.in_mixed_phase:
        # The mixed root outside the window is a continuation, not the state;
        # see the docstring. Say nan rather than differentiate it.
        out["cs2_eq"] = np.nan
        if T > 0.0:
            out["C_V"] = np.nan
        out["converged"] = True
        out["reason"] = ("outside the coexistence window the equilibrium "
                         "response is the pure phase's own")
        return out

    # The centre point's converged potentials seed both stencil points, which
    # is what keeps a step across a steep window on the same root.
    x0 = [point.potentials[name] for name in slots]

    def state(n, temperature):
        result = eos_point(phases, mode, species, n_B=n, T=temperature,
                           eta=eta, leptons=leptons, x0=x0, **conditions)
        if not result.ok:
            raise RuntimeError(
                f"eos_response could not solve its stencil point "
                f"(n_B={n}, T={temperature}): {result.message}")
        return result.point

    dn = rel_dn * n_B
    try:
        lo, hi = state(n_B - dn, T), state(n_B + dn, T)
        out["cs2_eq"] = ((hi.P - lo.P) / (hi.eps - lo.eps)
                         if hi.eps != lo.eps else np.nan)

        if T > 0.0:
            dT = rel_dn * T
            cold, hot = state(n_B, T - dT), state(n_B, T + dT)
            out["C_V"] = T * (hot.s - cold.s) / (2.0 * dT) / n_B
    except RuntimeError as err:
        # chi and the phase came from the centre point, which converged, so
        # they are kept: only the derivatives are unreachable.
        failed = unconverged_response(str(err), names)
        failed["chi"], failed["phase"] = point.chi, point.phase
        return failed

    out["converged"] = True
    out["reason"] = "converged"
    return out
