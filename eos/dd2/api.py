"""The uniform model API for DD2: eos_point, eos_table, eos_response.

Every model in this repository exposes these three entry points with the same
shape, so a caller — a table pipeline, a sampler, a downstream package —
drives any model the same way:

    eos_point(par, mode, species, n_B=..., T=..., ...)   one state
    eos_table(par, mode, species, axes)                  a solved grid
    eos_response(par, mode, species, frozen=..., ...)    second derivatives

Modes are the repository's named equilibria: 'beta_eq_neutrinoless',
'beta_eq_neutrino_trapped', 'fixed_YC', 'fixed_YC_YS' (DD2 also offers the
extras in eos.dd2.MODES). Conditions are named exactly n_B, T (or SnB), Y_C,
Y_S, Y_Le. Units at this boundary are fm-based: n_B in fm^-3, T and potentials
in MeV, eps and P in MeV/fm^3.

NON-CONVERGENCE IS A RETURN VALUE here, not an exception: a sampler walks
into unphysical corners constantly and must be able to score the point and
move on. The deep solvers raise; this surface catches and reports. Their
iteration counts are bounded (MINPACK hybr), so a call returns.
"""
from dataclasses import dataclass

from eos.dd2.solver import solve, EoSPoint, _mode_kwargs
from eos.dd2.table import TableSpec, build_table, solve_at_entropy

@dataclass(frozen=True)
class PointResult:
    """One eos_point outcome: a convergence status the caller can test.

    ok is judged on the solver's residual norm (RESIDUAL_TOL) and the
    Hugenholtz–Van Hove identity; when it is False, `point` is None and
    `message` says which condition failed and where.
    """
    ok: bool
    message: str
    point: EoSPoint = None


def _check(mode, conditions):
    """The fractions this mode consumes, from spec-named conditions.

    The condition names are fixed at n_B, T, Y_C, Y_S, Y_Le, Y_Lmu (CLAUDE.md
    section 5); `leptons` is a flag rather than a condition and reaches the
    entry points as a named argument, so finding it here means a caller still
    routing it through the bag.

    Y_Lmu raises: the muon family is not tracked in DD2's trapped mode, and
    per the species-flag rules an unimplemented request must never become a
    silent no-op.
    """
    if "Y_Lmu" in conditions:
        raise NotImplementedError(
            "DD2 does not track the muon lepton family; "
            "beta_eq_neutrino_trapped takes (n_B, Y_Le, T) only")
    if "leptons" in conditions:
        raise TypeError("leptons is a flag, not a condition; pass it as the "
                        "named argument leptons=")
    return dict(conditions)


def eos_point(par, mode, species, n_B, T=None, SnB=None, leptons=None,
              x0=None, analytic_jac=True, **conditions):
    """One solved state in a named mode; non-convergence is a return value.

    Parameters
    ----------
    par : Parameters
        The model parameters — always an argument, never module state.
    mode : str
        A key of eos.dd2.MODES.
    species : SpeciesFlags
        The active degrees of freedom.
    n_B : float
        Baryon density [fm^-3].
    T, SnB : float
        Exactly one of temperature [MeV] or entropy per baryon; SnB adds an
        outer 1-D solve for T.
    leptons : bool, optional
        For fixed_YC: whether the neutralizing leptons are added, so the total
        system is electrically neutral. With leptons=False the matter is
        charged, which is what a mixed-phase construction needs per pure phase
        before imposing global neutrality; left unnamed it is DD2's leptonless
        default. A flag, not a condition (CLAUDE.md section 3), so it is a
        named argument. In the beta-equilibrium modes the leptons are
        constitutive rather than optional: leptons=True is redundant and is
        ignored, leptons=False raises.
    conditions : the fractions the mode fixes (Y_C, Y_S, Y_Le).

    Returns
    -------
    PointResult — test `.ok` before using `.point`.

    The split between raising and returning: a malformed CALL — unknown mode,
    missing fraction, both or neither of T/SnB, a request the model does not
    implement — raises before any solve, because it is a programming error
    that a sampler would otherwise re-make a million times. What the SOLVE
    says about the state — non-convergence, a violated consistency identity —
    is the return value.
    """
    if (T is None) == (SnB is None):
        raise ValueError("exactly one of T / SnB must be given")
    fr = _check(mode, dict(conditions))              # caller errors raise here
    kwargs = _mode_kwargs(mode, fr, leptons)
    try:
        if SnB is not None:
            point = solve_at_entropy(par, n_B, SnB, species, x0=x0,
                                           **kwargs)
        else:
            point = solve(par, n_B, species, T=T, x0=x0,
                                analytic_jac=analytic_jac,
                                include_photons=species.photons, **kwargs)
        return PointResult(True, "converged", point)
    except NotImplementedError:
        raise                       # an unwired request must never be a status
    except (RuntimeError, ValueError) as err:
        # RuntimeError: the solve did not reach the residual gate.
        # ValueError: HVH / beta-condition violation, or outside the domain.
        return PointResult(False, str(err))


def eos_table(par, mode, species, axes, fixed=None, leptons=None,
              skip_errors=True, progress=None, verbose=False):
    """A solved grid over {n_B} x {T or SnB} [x fraction axes].

    A thin wrapper over eos.dd2.build_table: axes and fixed follow TableSpec
    (axes={'nB': grid, 'T' or 'SnB': grid, optionally 'Y_C'/'Y_S'/'Y_Le'}),
    the density axis is warm-started with bisected continuation through
    onsets, and the result's points/rows feed eos.astro.tov and the plotting code
    directly. skip_errors=True drops non-converged points from their line
    (the sampler-friendly default) rather than aborting the table.

    leptons : as in `eos_point` — the section 3 flag, so the neutral flavour
        of a fixed-Y_C table is reachable here too and not only through a
        mode name.
    progress : callable, invoked once per completed line with a dict
        {mode, line, n_lines, temp, fracs, n_solved, n_requested, elapsed_s}.
    verbose : True installs a one-line printer as the callback.
    """
    _check(mode, dict(fixed or {}))                  # caller errors raise here
    spec = TableSpec(parametrization=par, mode=mode, axes=dict(axes),
                     include=species, fixed=dict(fixed or {}), leptons=leptons)
    return build_table(spec, skip_errors=skip_errors, progress=progress,
                       verbose=verbose)


#: The freezes eos_response implements. Each answers "what is held fixed
#: while the derivative is taken", which encodes which reactions are faster
#: than the perturbation. The remaining spec freezes (frozen conserved
#: fractions; leptonic re-neutralization variants) are recorded in
#: docs/DEFERRED.md.
RESPONSE_FREEZES = ("equilibrium", "composition")


def eos_response(par, mode, species, frozen="equilibrium", n_B=None, T=0.0,
                 Y_p=None, **conditions):
    """Second-derivative quantities at one state.

    Both freezes return BOTH sound speeds, because a freeze is a statement
    about the COMPOSITION and the key name is a statement about the THERMAL
    variable (CLAUDE.md section 5) — two axes, never one word doing both jobs:

        cs2_isothermal   (dP/deps)_T, T held on both stencil points
        cs2_adiabatic    the same at fixed entropy per baryon, larger by
                         C_P/C_V; equal to the isothermal one at T = 0

    frozen='equilibrium' — everything re-equilibrates under the perturbation:
        the two sequence sound speeds, the heat capacities C_V and C_P, and
        the susceptibility matrix chi_ab = dn_a/dmu_b for a,b in (B, C, S),
        all from the analytic Jacobian. Implemented for beta_eq_neutrinoless.

    frozen='composition' — every particle fraction held fixed (reactions
        slow): the two sound speeds and the index Gamma at proton fraction Y_p
        (nucleonic matter), by finite difference along the frozen-Y_p
        sequence. Gamma is built on cs2_isothermal (see eos.dd2.responses).

    Returns a dict of the computed quantities; raises NotImplementedError,
    naming the gap, for freezes or modes not yet wired.
    """
    if frozen == "equilibrium":
        if mode != "beta_eq_neutrinoless":
            raise NotImplementedError(
                f"eos_response(frozen='equilibrium') is wired for "
                f"beta_eq_neutrinoless only, not {mode!r} "
                f"(see docs/DEFERRED.md)")
        from eos.dd2.backends.responses_jac import (
            sound_speed_eq, heat_capacity_V, heat_capacity_P, susceptibilities,
        )
        out = {"cs2_isothermal": sound_speed_eq(par, n_B, species, T=T),
               "chi": susceptibilities(par, n_B, species, T=T)}
        if T > 0.0:
            out["C_V"] = heat_capacity_V(par, n_B, species, T)
            out["C_P"] = heat_capacity_P(par, n_B, species, T)
            out["cs2_adiabatic"] = (out["C_P"] / out["C_V"]
                                    * out["cs2_isothermal"])
        else:
            out["cs2_adiabatic"] = out["cs2_isothermal"]   # C_P/C_V -> 1
        return out
    if frozen == "composition":
        if Y_p is None:
            raise ValueError("frozen='composition' needs Y_p")
        from eos.dd2.responses import (
            sound_speed_isothermal_frozen, sound_speed_adiabatic_frozen,
            adiabatic_index,
        )
        return {"cs2_isothermal": sound_speed_isothermal_frozen(
                    par, n_B, Y_p, T=T),
                "cs2_adiabatic": sound_speed_adiabatic_frozen(
                    par, n_B, Y_p, T=T),
                "Gamma": adiabatic_index(par, n_B, Y_p, T=T)}
    raise NotImplementedError(
        f"frozen={frozen!r} is not wired; implemented: {RESPONSE_FREEZES} "
        f"(see docs/DEFERRED.md)")
