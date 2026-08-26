"""The uniform model API for SFHo: eos_point, eos_table, eos_response.

Every model in this repository exposes these three entry points with the same
shape, so a caller — a table pipeline, a sampler, a downstream package —
drives any model the same way:

    eos_point(par, mode, species, n_B=..., T=..., ...)   one state
    eos_table(par, mode, species, axes)                  a solved grid
    eos_response(par, mode, species, frozen=..., ...)    second derivatives

Modes are the repository's named equilibria: 'beta_eq_neutrinoless',
'beta_eq_neutrino_trapped', 'fixed_YC', 'fixed_YC_YS' (with the neutral
flavours of the last two selected by leptons=True). Conditions are named
exactly n_B, T (or SnB), Y_C, Y_S, Y_Le. Units at this boundary are fm-based:
n_B in fm^-3, T and potentials in MeV, eps and P in MeV/fm^3.

NON-CONVERGENCE IS A RETURN VALUE here, not an exception: a sampler walks into
unphysical corners constantly and must be able to score the point and move on.
SFHo's solvers report it that way all the way down — `EoSPoint.converged`
against the residual norm in `.error`, with a bounded iteration count — so
this surface only has to pass it on.

References:
- Fortin, Oertel, Providencia, PASA 35 (2018) e044
- Steiner, Hempel, Fischer, ApJ 774 (2013) 17
"""
from dataclasses import dataclass

from eos.general.tabulate import unconverged_response
from eos.general.thermal_mesons import condensation_message
from eos.sfho.solver import EoSPoint, solve_mode
from eos.sfho.table import MODES, TableSpec, build_table, mode_spec


@dataclass(frozen=True)
class PointResult:
    """One eos_point outcome: a convergence status the caller can test.

    ok is the solver's own verdict on its residual norm; when it is False,
    `point` is None and `message` carries the norm that failed the gate.
    """
    ok: bool
    message: str
    point: EoSPoint = None


def _check(mode, conditions):
    """The fractions this mode needs, from spec-named conditions.

    The condition names are fixed at n_B, T, Y_C, Y_S, Y_Le, Y_Lmu (CLAUDE.md
    section 5); `leptons` is a flag rather than a condition and reaches the
    entry points as a named argument, so finding it here means a caller still
    routing it through the bag.

    Y_Lmu raises: SFHo does not track the muon lepton family at all, and per
    the species-flag rules an unimplemented request must never become a silent
    no-op.
    """
    if "Y_Lmu" in conditions:
        raise NotImplementedError(
            "SFHo does not track the muon lepton family; "
            "beta_eq_neutrino_trapped takes (n_B, Y_Le, T) only")
    if "leptons" in conditions:
        raise TypeError("leptons is a flag, not a condition; pass it as the "
                        "named argument leptons=")
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of "
                         f"{list(MODES)}")
    return dict(conditions)


def eos_point(par, mode, species, n_B, T=None, SnB=None, leptons=None,
              x0=None, **conditions):
    """One solved state in a named mode; non-convergence is a return value.

    Parameters
    ----------
    par : Parameters
        The model parameters — always an argument, never module state.
    mode : str
        One of the modes above.
    species : SpeciesFlags
        The active degrees of freedom.
    n_B : float
        Baryon density [fm^-3].
    T, SnB : float
        Exactly one of temperature [MeV] or entropy per baryon; SnB puts T in
        the unknown vector (CLAUDE.md section 3).
    x0 : array, optional
        A warm start, in the unknown order `solver.unknown_names` documents.
    leptons : bool, optional
        For the fixed-fraction modes: whether the neutralizing electrons are
        added, so the total system is electrically neutral. With leptons=False
        the matter is charged, which is what a mixed-phase construction needs
        per pure phase before imposing global neutrality; left unnamed it is
        SFHo's leptonless default. A flag, not a condition (CLAUDE.md section
        3), so it is a named argument. In the beta-equilibrium modes the
        leptons are constitutive rather than optional: leptons=True is
        redundant and is ignored, leptons=False raises.
    conditions : the fractions the mode fixes (Y_C, Y_S, Y_Le).

    Returns
    -------
    PointResult — test `.ok` before using `.point`.

    The split between raising and returning: a malformed CALL — unknown mode,
    missing fraction, both or neither of T/SnB, a request the model does not
    implement — raises before any solve, because it is a programming error
    that a sampler would otherwise re-make a million times. What the SOLVE
    says about the state is the return value.
    """
    if (T is None) == (SnB is None):
        raise ValueError("exactly one of T / SnB must be given")
    fracs = _check(mode, dict(conditions))             # caller errors raise here
    spec = mode_spec(mode, fracs, leptons)
    point = solve_mode(par, n_B, species, spec, T=T, SnB=SnB, x0=x0)
    if not point.converged:
        if point.matter.condensation >= 1.0:
            return PointResult(False, condensation_message(
                point.matter.condensation, point.n_B, point.T), None)
        return PointResult(
            False, f"residual {point.error:.3e} above the mode's gate", None)
    return PointResult(True, "converged", point)


def eos_table(par, mode, species, axes, fixed=None, leptons=None,
              skip_errors=True, rows=False, progress=None, verbose=False):
    """A solved grid over {n_B} x {T or SnB} [x fraction axes].

    A thin wrapper over `eos.sfho.build_table`: axes and fixed follow
    `TableSpec` (axes={'nB': grid, 'T' or 'SnB': grid, optionally
    'Y_C'/'Y_S'/'Y_Le'}), the density axis is warm-started by linear
    continuation, and the result feeds `eos.astro.tov` and the plotting code
    directly. skip_errors=True drops non-converged points from their line (the
    sampler-friendly default) rather than leaving them in it.

    leptons : as in `eos_point` — the section 3 flag, so the neutral flavour
        of a fixed-fraction table is reachable here too and not only through
        a mode name.

    progress : callable, invoked once per completed line with a dict
        {mode, line, n_lines, temp_key, temp, fracs, n_solved, n_requested,
        elapsed_s}.
    verbose : True installs a one-line printer as the callback.
    """
    _check(mode, dict(fixed or {}))                    # caller errors raise here
    spec = TableSpec(parametrization=par, mode=mode, axes=dict(axes),
                     include=species, fixed=dict(fixed or {}),
                     leptons=leptons)
    return build_table(spec, skip_errors=skip_errors, rows=rows,
                       progress=progress, verbose=verbose)


#: The freezes eos_response implements. Each answers "what is held fixed while
#: the derivative is taken", which encodes which reactions are faster than the
#: perturbation (CLAUDE.md section 5). The freezes that hold a composition —
#: per-species Y_i, or the conserved fractions with the species free — are
#: recorded in docs/DEFERRED.md.
RESPONSE_FREEZES = ("equilibrium",)


def eos_response(par, mode, species, frozen="equilibrium", n_B=None, T=0.0,
                 leptons=None, **conditions):
    """Second-derivative quantities at one state.

    frozen='equilibrium' — nothing is held: the composition re-equilibrates
        under the perturbation. Returns

            cs2_isothermal   (dP/dn_B)_T / (deps/dn_B)_T along the mode's
                             own sequence
            cs2_adiabatic    the same at fixed entropy per baryon, larger by
                             the heat-capacity ratio (only at T > 0)
            C_V, C_P         heat capacities per baryon (only at T > 0)
            Gamma_th         thermal index against the T = 0 state of the same
                             mode (only at T > 0)
            chi              the susceptibility matrix chi_ab = dn_a/dmu_b for
                             a, b in (B, C, S), in fm^-3 MeV^-1

        The first four come from `eos.sfho.responses`, by finite differences
        along re-solved sequences. `chi` comes from the analytic Jacobian in
        `backends/`, which is the only route to it — the solver never varies
        the three potentials independently, so there is no sequence to
        difference along — and it is therefore the one quantity here that a
        deleted `backends/` costs.

    Both sound speeds are named for the thermal condition they are taken at,
    because at T > 0 they are different numbers (CLAUDE.md section 5).

    Returns a dict of the computed quantities, plus `converged` and `reason`.
    A stencil point the equilibrium solver cannot reach is NOT an exception:
    the same dict comes back with converged=False and nan in every quantity,
    so a sampler can score the point and move on (CLAUDE.md section 6).
    Raises NotImplementedError, naming the gap, for freezes not yet wired.
    """
    if frozen != "equilibrium":
        raise NotImplementedError(
            f"frozen={frozen!r} is not wired for eos.sfho; implemented: "
            f"{RESPONSE_FREEZES}. Holding a composition needs the species "
            f"fractions in the residual, which SFHo does not carry "
            f"(see docs/DEFERRED.md)")
    if n_B is None:
        raise ValueError("eos_response needs n_B [fm^-3]")
    fracs = _check(mode, dict(conditions))
    spec = mode_spec(mode, fracs, leptons)

    from eos.sfho import responses as _fd
    try:
        from eos.sfho.backends.responses_jac import susceptibilities
    except ImportError:
        # `backends/` is optional (CLAUDE.md section 5). Everything else is
        # unchanged without it; only chi_ab has no reference flavour to fall
        # back to, so it is absent from the result either way.
        susceptibilities = None

    names = ("cs2_isothermal",)
    if T > 0.0:
        names += ("C_V", "C_P", "cs2_adiabatic", "Gamma_th")
    if susceptibilities is not None:
        names += ("chi",)

    try:
        out = {"cs2_isothermal": _fd.sound_speed_isothermal(par, n_B, species,
                                                            spec, T=T)}
        if T > 0.0:
            out["C_V"] = _fd.heat_capacity_V(par, n_B, species, spec, T)
            out["C_P"] = _fd.heat_capacity_P(par, n_B, species, spec, T)
            out["cs2_adiabatic"] = _fd.sound_speed_adiabatic(par, n_B, species,
                                                             spec, T)
            out["Gamma_th"] = _fd.thermal_index(par, n_B, species, spec, T)
        if susceptibilities is not None:
            out["chi"] = susceptibilities(par, n_B, species, T=T, spec=spec)
    except RuntimeError as err:
        return unconverged_response(str(err), names)

    out["converged"] = True
    out["reason"] = "converged"
    return out
