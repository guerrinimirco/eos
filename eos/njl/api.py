"""The uniform model API for NJL: eos_point, eos_table, eos_response.

Every model in this repository exposes these three entry points with the same
shape, so one caller -- a table pipeline, a sampler, a downstream package --
drives any of them the same way:

    eos_point(par, mode, species, n_B=..., T=..., ...)   one state
    eos_table(par, mode, species, axes)                  a solved grid
    eos_response(par, mode, species, frozen=..., ...)    second derivatives

Modes are the repository's named equilibria, and NJL closes all four at any
temperature: 'beta_eq_neutrinoless', 'beta_eq_neutrino_trapped', 'fixed_YC',
'fixed_YC_YS'. Conditions are named exactly n_B, T (or SnB), Y_C, Y_S, Y_Le.
Units at this boundary are fm-based: n_B in fm^-3, T and potentials in MeV,
eps and P in MeV/fm^3.

What is NOT a condition is the pairing pattern. Which diquark condensates
survive is an outcome, decided by free energy among the enumerated candidates,
and every returned point carries the winner along with the three gaps, the two
colour potentials and whether the state is gapless. A caller who wants one
particular pattern -- to draw a branch that is not the ground state, say --
passes `patterns=('2SC',)`, which is a restriction on the enumeration rather
than a fifth mode.

NON-CONVERGENCE IS A RETURN VALUE here, not an exception: a sampler walks into
unphysical corners constantly and must be able to score the point and move on.
A malformed CALL still raises -- an unknown mode, a missing fraction, both or
neither of T and SnB, a sector NJL does not implement -- because that is a
programming error a sampler would otherwise re-make a million times.
"""
from dataclasses import dataclass

from eos.general.tabulate import temperature_at_entropy
from eos.njl.solver import EoSPoint, MODE_FRACTIONS, solve
from eos.njl.species import SpeciesFlags
from eos.njl.table import TableSpec, build_table


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
            "eos.njl does not track the muon lepton family as a conserved "
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


def eos_point(par, mode="beta_eq_neutrinoless", species=None, n_B=None,
              T=None, SnB=None, x0=None, patterns=None, **conditions):
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
    patterns : tuple of str
        Restrict the pairing enumeration to these candidates. The default
        enumerates unpaired, 2SC, CFL and one asymmetric free seed when the
        `csc` flag is on, and only the unpaired one when it is off.
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
            p = solve(mode, n_B, temp, par, species, x0, patterns=patterns,
                      **conditions)
            return p.s_total / p.n_B if p.n_B else 0.0
        T = temperature_at_entropy(entropy_at, SnB)

    point = solve(mode, n_B, T, par, species, x0, patterns=patterns,
                  **conditions)
    if point.converged:
        return PointResult(True, f"converged in pattern {point.pattern!r}",
                           point)
    return PointResult(False,
                       f"residual {point.error:.3e} above the gate at "
                       f"n_B={n_B:g} fm^-3", point)


def eos_table(par, mode="beta_eq_neutrinoless", species=None, axes=None,
              fixed=None, leptons=True, skip_errors=True, rows=False,
              progress=None, verbose=False):
    """A solved grid over {n_B} x {T or SnB} [x fraction axes].

    A thin wrapper over `eos.njl.table.build_table`: axes and fixed follow
    `TableSpec`, the density axis is warm-started with bisected continuation
    through the strange-quark and pairing onsets, and the result feeds
    `eos.tov` and the plotting code directly. skip_errors=True drops
    non-converged points from their line rather than aborting the table.

    progress : callable, invoked once per completed line with
        {mode, line, n_lines, temp_key, temp, fracs, n_solved, n_requested,
        elapsed_s} -- the same dictionary in every model -- plus `pattern`,
        the phase the line ended in.
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
#: held Delta) and the susceptibility matrix chi_ab are recorded in
#: docs/DEFERRED.md.
RESPONSE_FREEZES = ("equilibrium",)


def eos_response(par, mode="beta_eq_neutrinoless", species=None,
                 frozen="equilibrium", n_B=None, T=0.0, rel_dn=1e-3, dT=0.05,
                 patterns=None, **conditions):
    """Second-derivative quantities at one state.

    frozen='equilibrium' -- nothing is held: the composition re-equilibrates
    under the perturbation, and so does the pairing pattern. Returns

        cs2_isothermal   (dP/dn_B)_T / (deps/dn_B)_T along the mode's own
                         sequence; at T = 0 this is the sound speed
        cs2_adiabatic    the same at fixed entropy per baryon, larger by
                         C_P/C_V; equal to the isothermal one at T = 0
        C_V, C_P         heat capacities per baryon (T > 0 only)
        Gamma_th         thermal index (T > 0 only)

    Both sound speeds are named for the thermal variable they hold, never as a
    bare `cs2` whose meaning would depend on the arguments (CLAUDE.md section
    5). Pass `patterns=('2SC',)` to differentiate within one pattern instead
    of across the enumeration.
    """
    from eos.njl import responses as _fd

    species = species if species is not None else SpeciesFlags()
    conditions = dict(_check(mode, dict(conditions)))
    if frozen not in RESPONSE_FREEZES:
        raise NotImplementedError(
            f"frozen={frozen!r} is not wired for eos.njl; implemented: "
            f"{RESPONSE_FREEZES}. Holding a composition needs the species "
            f"fractions carried through the solve as constraints, and holding "
            f"the gaps needs Delta fixed against its own equation "
            f"(see docs/DEFERRED.md)")

    kwargs = dict(patterns=patterns, **conditions)
    out = {"cs2_isothermal": _fd.sound_speed_isothermal(
        par, species, mode, n_B, T=T, rel_dn=rel_dn, **kwargs)}
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
    return out
