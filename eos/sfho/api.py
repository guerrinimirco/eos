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


def _normalize(mode, conditions):
    """(mode, the fractions it needs) from spec-named conditions.

    Y_Lmu raises: SFHo does not track the muon lepton family at all, and per
    the species-flag rules an unimplemented request must never become a silent
    no-op.
    """
    if "Y_Lmu" in conditions:
        raise NotImplementedError(
            "SFHo does not track the muon lepton family; "
            "beta_eq_neutrino_trapped takes (n_B, Y_Le, T) only")
    leptons = conditions.pop("leptons", None)
    if leptons is not None:
        if mode not in ("fixed_YC", "fixed_YC_YS"):
            raise ValueError("leptons= applies to the fixed-fraction modes; "
                             "beta equilibrium is defined by the leptons")
        if leptons:
            mode += "_neutral"
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of "
                         f"{[m for m in MODES if not m.endswith('_neutral')]} "
                         f"(with leptons=True for the neutral flavours)")
    return mode, dict(conditions)


def eos_point(par, mode, species, n_B, T=None, SnB=None, x0=None,
              **conditions):
    """One solved state in a named mode; non-convergence is a return value.

    Parameters
    ----------
    par : SFHoParams
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
    conditions : the fractions the mode fixes (Y_C, Y_S, Y_Le), plus
        leptons=True/False for the fixed-fraction modes (neutralizing
        electrons on/off).

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
    mode, fracs = _normalize(mode, dict(conditions))   # caller errors raise here
    spec = mode_spec(mode, fracs)
    point = solve_mode(par, n_B, species, spec, T=T, SnB=SnB, x0=x0)
    if not point.converged:
        return PointResult(
            False, f"residual {point.error:.3e} above the mode's gate", None)
    return PointResult(True, "converged", point)


def eos_table(par, mode, species, axes, fixed=None, skip_errors=True,
              rows=False, progress=None, verbose=False):
    """A solved grid over {n_B} x {T or SnB} [x fraction axes].

    A thin wrapper over `eos.sfho.build_table`: axes and fixed follow
    `TableSpec` (axes={'nB': grid, 'T' or 'SnB': grid, optionally
    'Y_C'/'Y_S'/'Y_Le'}), the density axis is warm-started by linear
    continuation, and the result feeds `eos.astro.tov` and the plotting code
    directly. skip_errors=True drops non-converged points from their line (the
    sampler-friendly default) rather than leaving them in it.

    progress : callable, invoked once per completed line with a dict
        {mode, line, n_lines, temp_key, temp, fracs, n_solved, n_requested,
        elapsed_s}.
    verbose : True installs a one-line printer as the callback.
    """
    mode, _ = _normalize(mode, dict(fixed or {}))
    spec = TableSpec(parametrization=par, mode=mode, axes=dict(axes),
                     include=species,
                     fixed={k: v for k, v in (fixed or {}).items()
                            if k != "leptons"})
    return build_table(spec, skip_errors=skip_errors, rows=rows,
                       progress=progress, verbose=verbose)


#: The freezes eos_response implements — none yet. Each would answer "what is
#: held fixed while the derivative is taken", which encodes which reactions are
#: faster than the perturbation (CLAUDE.md section 5).
RESPONSE_FREEZES = ()


def eos_response(par, mode, species, frozen="equilibrium", **conditions):
    """Second-derivative quantities at one state — not implemented for SFHo.

    Raises, naming the gap, rather than returning a number computed under a
    conditioning nobody asked for: a second derivative is only defined once
    one says what is held fixed. `eos.dd2.eos_response` implements the
    'equilibrium' and 'composition' freezes and is the worked example; SFHo
    has no analytic Jacobian and no finite-difference response module yet.
    """
    raise NotImplementedError(
        "eos.sfho has no response functions: c_s^2, C_V, C_P, Gamma and the "
        "susceptibility matrix are not implemented for this model "
        "(see docs/DEFERRED.md). eos.dd2.eos_response has them.")
