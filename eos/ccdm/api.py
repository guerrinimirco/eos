"""The uniform model API for CCDM: eos_point, eos_table, eos_response.

Every model in this repository exposes these three entry points with the same
shape, so one caller -- a table pipeline, a sampler, a downstream package --
drives any of them the same way:

    eos_point(par, mode, species, n_B=..., T=..., ...)   one state
    eos_table(par, mode, species, axes)                  a solved grid
    eos_response(par, mode, species, frozen=..., ...)    second derivatives

Modes are the repository's named equilibria, and CCDM closes all four at any
temperature: 'beta_eq_neutrinoless', 'beta_eq_neutrino_trapped', 'fixed_YC',
'fixed_YC_YS'. Conditions are named exactly n_B, T (or SnB), Y_C, Y_S, Y_Le.
Units at this boundary are fm-based: n_B in fm^-3, T and potentials in MeV,
eps and P in MeV/fm^3.

What is NOT a condition is the chiral/dielectric BRANCH or the pairing
PATTERN. Both are outcomes, decided by free energy among the enumerated
candidates, and every returned point carries the winners along with the four
fields, the three gaps, the two colour potentials and the two flags that say
when a comparison or a cutoff has stopped meaning what it usually means. A
caller who wants one particular branch -- to draw the deconfined curve below
its own onset, or to take a one-sided derivative across the first-order
transition -- passes `branches=('restored',)` or `patterns=('2SC',)`, which
are restrictions on the enumeration rather than extra modes.

WHERE THIS MODEL IS DEFINED. At fixed density the confined branch carries no
quarks, so there is no deconfined root below the point where the branch turns
around; a solve there returns `ok = False` rather than a fabricated state. The
low-density half of a hybrid equation of state comes from a hadronic model
through `eos.mixed`, which is what `ccdm_phase` is for.

NON-CONVERGENCE IS A RETURN VALUE here, not an exception: a sampler walks into
unphysical corners constantly and must be able to score the point and move on.
A malformed CALL still raises -- an unknown mode, a missing fraction, both or
neither of T and SnB, a sector CCDM does not implement -- because that is a
programming error a sampler would otherwise re-make a million times.
"""
from dataclasses import dataclass

from eos.ccdm.solver import EoSPoint, MODE_FRACTIONS, solve
from eos.ccdm.species import SpeciesFlags
from eos.ccdm.table import TableSpec, build_table
from eos.general.tabulate import (temperature_at_entropy,
                                  unconverged_response)


@dataclass(frozen=True)
class PointResult:
    """One eos_point outcome: a convergence status the caller can test.

    `ok` is the solver's own judgement on its scaled residual; `message` says
    what happened, and `point` carries the state (it is present even when `ok`
    is False, so a caller can look at where the solve got to).
    """
    ok: bool
    message: str
    point: EoSPoint = None


def _check(mode, conditions):
    """Validate the call, and return the fractions the mode consumes."""
    if mode not in MODE_FRACTIONS:
        raise ValueError(f"unknown mode {mode!r}; expected one of "
                         f"{list(MODE_FRACTIONS)}")
    if "Y_Lmu" in conditions:
        raise NotImplementedError(
            "eos.ccdm does not track the muon lepton family as a conserved "
            "charge; beta_eq_neutrino_trapped takes (n_B, Y_Le, T). The muon "
            "SPECIES is available through SpeciesFlags(muons=True) "
            "(see docs/DEFERRED.md)")
    missing = [k for k in MODE_FRACTIONS[mode] if k not in conditions]
    if missing:
        raise ValueError(f"mode {mode!r} needs {missing}")
    extra = [k for k in conditions
             if k not in MODE_FRACTIONS[mode] and k != "leptons"]
    if extra:
        raise ValueError(f"mode {mode!r} does not take {extra}")
    return conditions


def eos_point(par, mode, species=None, n_B=None,
              T=None, SnB=None, x0=None, branches=None, patterns=None,
              **conditions):
    """One solved state in a named mode; non-convergence is a return value.

    Parameters
    ----------
    par : Parameters
        The model parameters -- always an argument, never module state.
    mode : str
        One of `MODE_FRACTIONS`.
    species : SpeciesFlags
        The active degrees of freedom; unpaired, with muons, by default.
    n_B : float
        Baryon density [fm^-3].
    T, SnB : float
        Exactly one of temperature [MeV] or entropy per baryon. SnB is an
        outer one-dimensional solve for T, since the entropy of this model is
        not a variable its residual carries.
    branches : tuple of str
        Restrict the chiral/dielectric enumeration. The default at fixed
        density is ('restored', 'partial') -- the confined branch carries no
        quarks and cannot meet a nonzero density row.
    patterns : tuple of str
        Restrict the pairing enumeration. The default enumerates unpaired,
        2SC, CFL and one asymmetric free seed when the `csc` flag is on, and
        only the unpaired one when it is off.
    conditions :
        The fractions the mode fixes (Y_C, Y_S, Y_Le), plus leptons=True/False
        for the fixed-fraction modes.
    """
    species = species if species is not None else SpeciesFlags()
    conditions = dict(_check(mode, dict(conditions)))
    if (T is None) == (SnB is None):
        raise ValueError("exactly one of T / SnB must be given")

    if SnB is not None:
        def entropy_at(temp):
            p = solve(mode, n_B, temp, par, species, x0, branches=branches,
                      patterns=patterns, **conditions)
            return p.s_total / p.n_B if p.n_B else 0.0
        try:
            T = temperature_at_entropy(entropy_at, SnB)
        except (RuntimeError, ValueError) as err:
            # An entropy target the model cannot reach is a state that does
            # not exist, not a caller error: CLAUDE.md section 6 wants it
            # scored and stepped over, never raised.
            return PointResult(False, str(err))

    point = solve(mode, n_B, T, par, species, x0, branches=branches,
                  patterns=patterns, **conditions)
    if point.converged:
        return PointResult(
            True, f"converged in branch {point.branch!r}, pattern "
                  f"{point.pattern!r}", point)
    return PointResult(False,
                       f"residual {point.error:.3e} above the gate at "
                       f"n_B={n_B:g} fm^-3; below the deconfinement onset "
                       f"this model has no deconfined root at fixed density",
                       point)


def eos_table(par, mode, species=None, axes=None,
              fixed=None, leptons=True, skip_errors=True, rows=False,
              progress=None, verbose=False):
    """A solved grid over {n_B} x {T or SnB} [x fraction axes].

    A thin wrapper over `eos.ccdm.table.build_table`: axes and fixed follow
    `TableSpec`, the density axis is warm-started with bisected continuation
    through the strange-quark, pairing and branch changes, and the result
    feeds `eos.astro.tov` and the plotting code directly. skip_errors=True
    drops non-converged points from their line rather than aborting the table,
    which is what the sub-onset densities need.

    progress : callable, invoked once per completed line with
        {mode, line, n_lines, temp_key, temp, fracs, n_solved, n_requested,
        elapsed_s} -- the same dictionary in every model -- plus `branch` and
        `pattern`, the phase the line ended in.
    verbose : True installs the shared one-line printer as that callback.
    """
    species = species if species is not None else SpeciesFlags()
    spec = TableSpec(par=par, mode=mode, axes=dict(axes or {}),
                     include=species, fixed=dict(fixed or {}),
                     leptons=leptons)
    return build_table(spec, skip_errors=skip_errors, rows=rows,
                       progress=progress, verbose=verbose)


#: The freezes `eos_response` implements. Each answers "what is held fixed
#: while the derivative is taken", which is what encodes which reactions are
#: faster than the perturbation. The composition freezes (held Y_i, held Y_C,
#: held Delta, held fields) and the susceptibility matrix chi_ab are recorded
#: in docs/DEFERRED.md.
RESPONSE_FREEZES = ("equilibrium",)


def eos_response(par, mode, species=None,
                 frozen="equilibrium", n_B=None, T=0.0, rel_dn=1e-3, dT=0.05,
                 branches=None, patterns=None, **conditions):
    """Second-derivative quantities at one state.

    frozen='equilibrium' -- nothing is held: the composition re-equilibrates
    under the perturbation, and so do both enumerations. Returns

        cs2_isothermal   (dP/dn_B)_T / (deps/dn_B)_T along the mode's own
                         sequence; at T = 0 this is the sound speed
        cs2_adiabatic    the same at fixed entropy per baryon, larger by
                         C_P/C_V; equal to the isothermal one at T = 0
        C_V, C_P         heat capacities per baryon (T > 0 only)
        Gamma_th         thermal index (T > 0 only)
        branch_changed   whether the density stencil straddled a branch or
                         pattern change, in which case every number above is
                         a CHORD ACROSS A FIRST-ORDER JUMP rather than a
                         tangent, and the derivative should be retaken
                         one-sided with `branches=` or `patterns=`

    Both sound speeds are named for the thermal variable they hold, never as a
    bare `cs2` whose meaning would depend on the arguments (CLAUDE.md section
    5). `branch_changed` is returned rather than left to the caller to
    suspect: this model has a first-order transition, and there is no way to
    see from the number alone that a stencil crossed it.

    Every result also carries `converged` and `reason`. A stencil point the
    solver cannot reach is NOT an exception: the same dict comes back with
    converged=False and nan in every quantity, so a sampler can score the
    point and move on (CLAUDE.md section 6).
    """
    from eos.ccdm import responses as _fd

    species = species if species is not None else SpeciesFlags()
    conditions = dict(_check(mode, dict(conditions)))
    if frozen not in RESPONSE_FREEZES:
        raise NotImplementedError(
            f"frozen={frozen!r} is not wired for eos.ccdm; implemented: "
            f"{RESPONSE_FREEZES}. Holding a composition needs the species "
            f"fractions carried through the solve as constraints, and holding "
            f"the gaps or the fields needs them fixed against their own "
            f"equations (see docs/DEFERRED.md)")

    kwargs = dict(branches=branches, patterns=patterns, **conditions)
    names = ("cs2_isothermal", "cs2_adiabatic", "branch_changed")
    if T > 0.0:
        names += ("C_V", "C_P", "Gamma_th")
    try:
        out = {"cs2_isothermal": _fd.sound_speed_isothermal(
            par, species, mode, n_B, T=T, rel_dn=rel_dn, **kwargs)}
        out["branch_changed"] = _fd.branch_changed(par, species, mode, n_B,
                                                   T=T, rel_dn=rel_dn,
                                                   **kwargs)
        if T > 0.0:
            out["cs2_adiabatic"] = _fd.sound_speed_adiabatic(
                par, species, mode, n_B, T=T, dT=dT, rel_dn=rel_dn, **kwargs)
            out["C_V"] = _fd.heat_capacity_V(par, species, mode, n_B, T, dT=dT,
                                             **kwargs)
            out["C_P"] = _fd.heat_capacity_P(par, species, mode, n_B, T, dT=dT,
                                             rel_dn=rel_dn, **kwargs)
            out["Gamma_th"] = _fd.thermal_index(par, species, mode, n_B, T,
                                                **kwargs)
        else:
            out["cs2_adiabatic"] = out["cs2_isothermal"]
    except RuntimeError as err:
        return unconverged_response(str(err), names)

    out["converged"] = True
    out["reason"] = "converged"
    return out
