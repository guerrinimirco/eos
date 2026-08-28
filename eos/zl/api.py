"""The uniform model API for ZL: eos_point, eos_table, eos_response.

Every model in this repository exposes these three entry points with the same
shape, so a caller -- a table pipeline, a sampler, a downstream package --
drives any model the same way:

    eos_point(par, mode, species, n_B=..., T=..., ...)   one state
    eos_table(par, mode, species, axes)                  a solved grid
    eos_response(par, mode, species, frozen=..., ...)    second derivatives

Modes are the repository's named equilibria: 'beta_eq_neutrinoless',
'beta_eq_neutrino_trapped', 'fixed_YC'. The fourth, 'fixed_YC_YS', is
physically meaningless for a model with no strange degree of freedom and
raises rather than ignoring Y_S. Conditions are named exactly n_B, T (or SnB),
Y_C, Y_Le. Units at this boundary are fm-based: n_B in fm^-3, T and potentials
in MeV, eps and P in MeV/fm^3.

NON-CONVERGENCE IS A RETURN VALUE here, not an exception: a sampler walks into
unphysical corners constantly and must be able to score the point and move on.
A malformed CALL is different -- an unknown mode, a fraction the mode does not
take, a request the model does not implement -- and raises before any solve,
because it is a programming error that would otherwise be repeated a million
times in silence.
"""
from dataclasses import dataclass

from eos.general.tabulate import (temperature_at_entropy,
                                  unconverged_response)
from eos.general.modes import resolve_leptons
from eos.zl.solver import EoSPoint, MODE_FRACTIONS, solve_fixed_yc_ys
from eos.zl.species import SpeciesFlags
from eos.zl.table import TableSpec, build_table, solve_at


@dataclass(frozen=True)
class PointResult:
    """One eos_point outcome: a convergence status the caller can test.

    `ok` is judged on the largest scaled equilibrium residual (see
    `eos.zl.solver.scaled_residual_max`); when it is False, `point` is None
    and `message` says what the residual reached.
    """
    ok: bool
    message: str
    point: EoSPoint = None


def _check_conditions(mode, conditions):
    """Raise unless exactly the fractions this mode takes were supplied."""
    if mode == "fixed_YC_YS":
        solve_fixed_yc_ys()              # raises, naming the physics
    if mode not in MODE_FRACTIONS:
        raise ValueError(f"unknown mode {mode!r}; expected one of "
                         f"{list(MODE_FRACTIONS)}")
    if "Y_Lmu" in conditions:
        raise NotImplementedError(
            "ZL does not track the muon lepton family; "
            "beta_eq_neutrino_trapped takes (n_B, Y_Le, T) only")
    wanted = set(MODE_FRACTIONS[mode])
    given = set(conditions)
    missing = wanted - given
    if missing:
        raise ValueError(f"mode {mode!r} needs {sorted(missing)}")
    extra = given - wanted
    if extra:
        raise ValueError(f"mode {mode!r} does not take {sorted(extra)}")


def eos_point(par, mode, species=None, n_B=None, T=None, SnB=None,
              leptons=None, x0=None, **conditions):
    """One solved state in a named mode; non-convergence is a return value.

    Parameters
    ----------
    par : Parameters
        The model parameters -- always an argument, never module state.
    mode : str
        One of the keys of `eos.zl.solver.MODE_FRACTIONS`.
    species : SpeciesFlags
        The active degrees of freedom. Every one defaults to False, so
        `SpeciesFlags()` asks for bare nucleonic matter; `photons` is the
        only sector this model wires.
    n_B : float
        Baryon density [fm^-3].
    T, SnB : float
        Exactly one of temperature [MeV] or entropy per baryon; SnB adds an
        outer 1-D solve for T.
    leptons : bool
        For `fixed_YC`: whether neutralizing electrons are added (n_e = n_C),
        so the total system is electrically neutral. With leptons=False the
        result is charged nucleonic matter, which is what a mixed-phase
        construction needs per pure phase before imposing global neutrality.
        In the beta-equilibrium modes the leptons are constitutive rather than
        optional: leptons=True is redundant and is ignored, leptons=False
        raises.
    conditions : the fractions the mode fixes (Y_C, Y_Le).

    Returns
    -------
    PointResult -- test `.ok` before using `.point`.
    """
    if species is None:
        species = SpeciesFlags()
    if n_B is None:
        raise ValueError("n_B is required")
    if n_B <= 0.0:
        raise ValueError(f"n_B must be positive, got {n_B}")
    if (T is None) == (SnB is None):
        raise ValueError("exactly one of T / SnB must be given")
    _check_conditions(mode, conditions)          # caller errors raise here
    leptons = resolve_leptons(mode, leptons, default=False)

    def solve(temperature):
        line = dict(conditions)
        line["T"] = float(temperature)
        return solve_at(par, mode, n_B, line, species, leptons, x0=x0)

    try:
        if SnB is not None:
            T = temperature_at_entropy(
                lambda t: _entropy_per_baryon(solve(t), n_B), float(SnB))
        point = solve(T)
    except NotImplementedError:
        raise                       # an unwired request must never be a status
    except (RuntimeError, ValueError) as err:
        return PointResult(False, str(err))

    if not point.converged:
        return PointResult(
            False, f"ZL {mode} solve did not converge at n_B={n_B}, T={T}: "
                   f"largest scaled residual {point.error:.2e}")
    return PointResult(True, "converged", point)


def _entropy_per_baryon(point, n_B):
    """s/n_B of a solved point, for the entropy-axis outer solve."""
    if not point.converged:
        raise RuntimeError("the entropy solve stepped onto a state the "
                           "equilibrium solver could not reach")
    return point.s_total / n_B


def eos_table(par, mode, species=None, axes=None, fixed=None, leptons=None,
              skip_errors=True, rows=False, progress=None, verbose=False):
    """A solved grid over {n_B} x {T} [x fraction axes].

    A thin wrapper over `eos.zl.table.build_table`: axes follow TableSpec
    (axes={'nB': grid, 'T': grid, optionally 'Y_C'/'Y_Le'}), the density axis
    is warm-started, and the result feeds `eos.astro.tov` and the plotting
    code directly. skip_errors=True drops non-converged points from their line
    -- the sampler-friendly default -- rather than aborting.

    progress : callable, invoked once per completed line with a dict
        {mode, line, n_lines, temp_key, temp, fracs, n_solved, n_requested,
        elapsed_s}. verbose=True installs the built-in one-line printer.
    """
    if species is None:
        species = SpeciesFlags()
    spec = TableSpec(params=par, mode=mode, axes=dict(axes or {}),
                     include=species, fixed=dict(fixed or {}), leptons=leptons)
    return build_table(spec, skip_errors=skip_errors, rows=rows,
                       progress=progress, verbose=verbose)


#: The freezes eos_response implements. Each answers "what is held fixed while
#: the derivative is taken", which is what encodes which reactions are faster
#: than the perturbation. The remaining spec freezes (frozen per-species
#: composition, frozen conserved fractions, the leptonic re-neutralization
#: variants) are recorded in docs/DEFERRED.md.
RESPONSE_FREEZES = ("equilibrium",)


def eos_response(par, mode, species=None, frozen="equilibrium", n_B=None,
                 T=0.0, leptons=None, rel_step=1e-3, **conditions):
    """Second-derivative quantities at one state.

    frozen='equilibrium' -- everything re-equilibrates under the perturbation,
        so the derivatives are taken along the mode's own sequence:

            cs2_isothermal = dP/deps          at fixed T, along the
                                              mode's own sequence
            C_V            = (T/n_B) ds/dT    at fixed n_B

        Both by central differences over a relative step `rel_step` in the
        variable differentiated, since ZL's residual has no analytic Jacobian
        in this repository. C_V is returned only at T > 0, where it is
        defined.

        The adiabatic speed, larger by C_P/C_V at T > 0, is NOT computed:
        C_P is not among the returned quantities, so there is no factor to
        form it with. `docs/DEFERRED.md` records the gap.

    Returns a dict of the computed quantities, plus `converged` and `reason`.
    A stencil point the equilibrium solver cannot reach is NOT an exception:
    the same dict comes back with converged=False and nan in every quantity,
    so a sampler can score the point and move on (CLAUDE.md section 6).
    Raises NotImplementedError, naming the gap, for freezes not yet wired.
    """
    if frozen != "equilibrium":
        raise NotImplementedError(
            f"frozen={frozen!r} is not wired for ZL; implemented: "
            f"{RESPONSE_FREEZES} (see docs/DEFERRED.md)")
    if species is None:
        species = SpeciesFlags()
    if n_B is None:
        raise ValueError("n_B is required")
    _check_conditions(mode, conditions)

    def state(n, temperature):
        result = eos_point(par, mode, species, n_B=n, T=temperature,
                           leptons=leptons, **conditions)
        if not result.ok:
            raise RuntimeError(
                f"eos_response could not solve its stencil point "
                f"(n_B={n}, T={temperature}): {result.message}")
        return result.point

    dn = rel_step * n_B
    try:
        lo, hi = state(n_B - dn, T), state(n_B + dn, T)
        out = {"cs2_isothermal": (hi.P_total - lo.P_total) / (hi.e_total - lo.e_total)}

        if T > 0.0:
            dT = rel_step * T
            cold, hot = state(n_B, T - dT), state(n_B, T + dT)
            out["C_V"] = T * (hot.s_total - cold.s_total) / (2.0 * dT) / n_B
    except RuntimeError as err:
        return unconverged_response(
            str(err), ("cs2_isothermal", "C_V") if T > 0.0 else ("cs2_isothermal",))

    out["converged"] = True
    out["reason"] = "converged"
    return out
