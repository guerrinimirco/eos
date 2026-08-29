"""The uniform model API for ABPR: eos_point, eos_table, eos_response.

Every model in this repository exposes these three entry points with the same
shape, so a caller -- a table pipeline, a sampler, a downstream package --
drives any model the same way:

    eos_point(par, mode, species, n_B=..., T=0.0)   one state
    eos_table(par, mode, species, axes)             a solved grid
    eos_response(par, mode, species, n_B=..., ...)  second derivatives

The mode this model closes is `cfl`, the colour-flavour locked phase, and it
is the only one: locking fixes the composition, so the four equilibrium modes
of this repository have no state here and each raises naming the physics (see
`eos.abpr.solver.MODE_REFUSALS`). T = 0 is the only temperature.

Conditions are named exactly n_B and T. Units at this boundary are fm-based:
n_B in fm^-3, potentials in MeV, eps and P in MeV/fm^3.

NON-CONVERGENCE IS A RETURN VALUE here, not an exception, as it is in every
model -- though in this one it can only mean a request outside the phase (a
pressure below -B), the inverse maps being closed forms. A malformed CALL is
different -- an unknown mode, a mode the model refuses, a temperature it does
not have, a species flag it does not carry -- and raises before any work,
because it is a programming error that would otherwise be repeated a million
times in silence.
"""
from dataclasses import dataclass, field

import numpy as np

from eos.abpr.parameters import Parameters
from eos.abpr.solver import (
    MODE_FRACTIONS, check_mode, check_temperature, mu_from_nB, point_from_mu,
    response_at_mu, solve_cfl,
)
from eos.abpr.species import SpeciesFlags
from eos.general.basis import quark_charges
from eos.general.tabulate import unconverged_response
from eos.general.zero_pressure import (
    N_HI_DEFAULT, N_LO_DEFAULT, N_SCAN_DEFAULT, locate_zero_pressure,
)


@dataclass(frozen=True)
class PointResult:
    """One eos_point outcome: a convergence status the caller can test.

    When `ok` is False, `point` is None and `message` says what the request
    reached.
    """
    ok: bool
    message: str
    point: object = None


def _check_call(mode, species, T, SnB, conditions):
    """Raise unless the request is one this model can be asked at all."""
    check_mode(mode)
    if conditions:
        raise ValueError(
            f"mode {mode!r} takes no fractions; got {sorted(conditions)}. "
            f"The pairing gap is a parameter here, not a condition -- it is "
            f"Parameters.Delta0")
    if SnB is not None:
        if T is not None:
            raise ValueError("give exactly one of T / SnB")
        if SnB != 0.0:
            raise NotImplementedError(
                f"eos.abpr is a T = 0 parametrization, where s = 0 and the "
                f"only entropy per baryon it reaches is zero; got SnB = "
                f"{SnB}. For an isentrope use the 'cfl' mode of eos.alphabag")
        T = 0.0
    check_temperature(0.0 if T is None else T)
    SpeciesFlags() if species is None else species     # flags validate on init
    return 0.0


def eos_point(par, mode="cfl", species=None, n_B=None, T=0.0, SnB=None,
              **conditions):
    """One solved state; non-convergence is a return value.

    Parameters
    ----------
    par : Parameters
        The model parameters -- always an argument, never module state.
    mode : str
        `cfl`, the only mode this model closes.
    species : SpeciesFlags
        The active degrees of freedom. Every sector is off here and switching
        one on raises; see `eos.abpr.species`.
    n_B : float
        Baryon density [fm^-3].
    T, SnB : float
        Temperature [MeV] or entropy per baryon; both are zero, and any other
        value raises rather than being quietly rounded to the model's one
        temperature.

    Returns
    -------
    PointResult -- test `.ok` before using `.point`.
    """
    T = _check_call(mode, species, T, SnB, conditions)
    if n_B is None:
        raise ValueError("n_B is required")
    if n_B <= 0.0:
        raise ValueError(f"n_B must be positive, got {n_B}")

    point = solve_cfl(par, n_B, T=T)
    if not point.converged:
        return PointResult(
            False, f"ABPR cfl inverse did not land at n_B={n_B}: "
                   f"residual {point.error:.2e}")
    return PointResult(True, "converged", point)


@dataclass
class TableResult:
    """A solved ABPR grid.

    The same shape as every other model's table result -- the conditions of
    each line, and the solved points of each line -- so the same reader walks
    an ABPR table and an alphaBag one. This model has exactly one line, T = 0,
    because it has one temperature and no fractions to sweep.
    """
    par: Parameters = field(default_factory=Parameters.default)
    mode: str = "cfl"
    nB: np.ndarray = field(default_factory=lambda: np.empty(0))
    lines: list = field(default_factory=list)
    points: list = field(default_factory=list)


def cfl_row(point):
    """One solved point as a flat table row.

    Keyed the way `eos.alphabag.table.quark_row` keys its paired points, so an
    ABPR table and an alphaBag one concatenate without renaming: chi = 1 and
    phase = 'Q' say the matter is entirely deconfined, and the gap is carried
    because it is the one column a reader cannot recover from the others.
    """
    n_B = point.n_B
    return dict(n_B=n_B, T=point.T, chi=1.0, phase="Q",
                P=point.P, eps=point.eps, s=point.s,
                S_per_B=0.0,
                mu_B=point.mu_B, mu_C=point.mu_C, mu_S=point.mu_S,
                mu_e=point.mu_e,
                Y_C=point.Y_C, Y_S=point.Y_S,
                Y_u=point.Y_u, Y_d=point.Y_d, Y_s=point.Y_s, Y_e=0.0,
                Delta0=point.Delta0, Delta=point.Delta)


def eos_table(par, mode="cfl", species=None, axes=None, rows=False,
              progress=None, verbose=False):
    """A solved grid over the density axis.

    axes = {'nB': grid, 'T': [0.0]}; the temperature axis may be omitted, in
    which case T = 0 is understood, since it is the only value the model has.
    The result feeds `eos.astro.tov` and the plotting code directly.

    There is no warm start and no bisected continuation here, and their
    absence is the physics rather than a gap: the density inverse is a closed
    form (`eos.abpr.solver.mu_from_nB`), so no point needs its neighbour. The
    grid is nevertheless walked point by point -- `solve_cfl` takes one scalar
    density and returns one `CFLPoint`, and this loops it over `nB`. Every
    point being independent, genuine array-in/array-out (CLAUDE.md section 6)
    is reachable for this model and for no other; it is a change to the solver
    signature, not to this driver, and it has not been made. Nor is there a
    skip_errors flag: a request outside the phase is a property of the target,
    not of a solve that might have gone better from a different start.

    progress : callable, invoked once per completed line -- there is one --
        with the dict every table builder in this repository reports:
        {mode, line, n_lines, temp_key, temp, fracs, n_solved, n_requested,
        elapsed_s}. verbose=True installs the built-in one-line printer.
    """
    import time

    axes = dict(axes or {})
    nB = np.atleast_1d(np.asarray(axes.pop("nB"), dtype=float))
    temps = np.atleast_1d(np.asarray(axes.pop("T", [0.0]), dtype=float))
    if axes:
        raise ValueError(f"mode {mode!r} has no further axes; got "
                         f"{sorted(axes)}")
    if len(temps) != 1:
        raise NotImplementedError(
            f"eos.abpr has one temperature, T = 0; got a temperature axis of "
            f"{len(temps)} values")
    T = _check_call(mode, species, float(temps[0]), None, {})

    started = time.time()
    points = [solve_cfl(par, float(n), T=T) for n in nB]
    if progress is None and verbose:
        from eos.general.tabulate import print_progress
        progress = print_progress
    if progress is not None:
        progress({"mode": mode, "line": 1, "n_lines": 1, "temp_key": "T",
                  "temp": T, "fracs": {},
                  "n_solved": sum(1 for p in points if p.converged),
                  "n_requested": len(nB),
                  "elapsed_s": time.time() - started})

    if rows:
        return [cfl_row(p) for p in points if p.converged]
    return TableResult(par=par, mode=mode, nB=nB, lines=[{"T": T}],
                       points=[points])


#: The freezes eos_response implements. At T = 0 there is only one: the
#: composition is locked, so nothing can be held that is not already held.
RESPONSE_FREEZES = ("equilibrium",)


def eos_response(par, mode="cfl", species=None, frozen="equilibrium",
                 n_B=None, T=0.0, **conditions):
    """Second-derivative quantities at one state, in closed form.

    frozen='equilibrium' -- the only conditioning this phase admits. Flavour
        locking holds the composition at every density, so "what is held
        fixed while the derivative is taken" has one answer and the named
        freezes of the other models (`fast`, `slow`) would expand to the same
        set.

    Returns {'cs2_isothermal': c_s^2, 'converged': True, 'reason': ...} from
    `eos.abpr.thermodynamics.sound_speed_squared`, differentiated
    analytically rather than by a stencil. At T = 0 the isothermal and
    adiabatic sound speeds coincide; the name says which convention the number
    was computed under rather than leaving it to the arguments.

    The heat capacities C_V and C_P, the thermal index and the adiabatic index
    are not defined at T = 0 and are not returned. The susceptibilities
    chi_ab = dn_a/dmu_b are singular here, flavour locking leaving n_C and n_S
    with no potential to respond to, and are not returned either.

    A density the n_B inversion cannot reach is NOT an exception: the same
    dict comes back with converged=False and nan for the sound speed, so a
    sampler can score the point and move on (CLAUDE.md section 6).
    """
    if frozen != "equilibrium":
        raise NotImplementedError(
            f"frozen={frozen!r} has no meaning in a flavour-locked phase, "
            f"where the composition is held at every density; implemented: "
            f"{RESPONSE_FREEZES}")
    T = _check_call(mode, species, T, None, conditions)
    if n_B is None:
        raise ValueError("n_B is required")
    mu, converged = mu_from_nB(n_B, par)
    if not converged:
        return unconverged_response(
            f"eos_response could not invert n_B = {n_B} fm^-3 to a quark "
            f"chemical potential", ("cs2_isothermal",))
    out = response_at_mu(mu, par)
    out["converged"] = True
    out["reason"] = "converged"
    return out


def zero_pressure_point(par, species=None, n_lo=N_LO_DEFAULT,
                        n_hi=N_HI_DEFAULT, n_scan=N_SCAN_DEFAULT):
    """E/A at the self-bound surface, P = 0 and T = 0.

    The locked phase is self-bound: P falls to zero at finite density with no
    crust below it, and eps/n_B there is the energy per baryon of a lump of
    this matter at rest. For the shipped set it is 831.58 MeV, below the
    930.4 MeV of iron, which is what "absolutely stable strange quark matter"
    means for this parametrization.

    THERE IS NO TWO-FLAVOUR ARM HERE, and the absence is physics rather than
    an omission: `cfl` is the only mode this model has, and flavour locking
    fixes Y_S = +1 identically, so no strangeness fraction is free to switch
    off. `SpeciesFlags(two_flavour=True)` therefore raises, naming the reason,
    rather than this function returning a nan for a number that does not
    exist. The two-flavour half of the Bodmer-Witten window is asked of a
    model that has an unpaired phase -- `eos.vmit`, `eos.alphabag`, `eos.njl`,
    `eos.ccdm` -- in `beta_eq_neutrinoless`.

    The root find is `eos.general.zero_pressure.locate_zero_pressure` over
    this model's own `eos_point`, deliberately, even though `mu_from_P` gives
    the same surface in closed form one line at a time: the closed form does
    not generalise past a polynomial pressure, and driving the shared locator
    with the model that has an exact answer is what measures the locator.
    `verify/run_full_check.py` compares the two.

    Returns a `ZeroPressurePoint`; test `.ok`. `below_iron` is reported, never
    asserted -- whether a set sits in the Bodmer-Witten window is a property
    of the set.
    """
    flags = SpeciesFlags() if species is None else species

    def point_at(n_B):
        result = eos_point(par, "cfl", flags, n_B=n_B, T=0.0)
        if not result.ok:
            return None
        p = result.point
        # n_B from the solved flavour densities, as in every other model: the
        # locked phase closes n_B to a residual too, and eps goes with the
        # densities rather than with the request.
        n_B_solved, _, n_S = quark_charges(p.n_u, p.n_d, p.n_s)
        return (p.P, p.eps / n_B_solved, p.mu_B,
                n_S / n_B_solved, p.mu_S)

    return locate_zero_pressure(point_at, two_flavour=False, n_lo=n_lo,
                                n_hi=n_hi, n_scan=n_scan)
