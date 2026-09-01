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
from eos.general.basis import quark_charges
from eos.general.zero_pressure import (
    N_HI_DEFAULT, N_LO_DEFAULT, N_SCAN_DEFAULT, locate_zero_pressure,
)
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
    """Validate the call, and return the fractions the mode consumes.

    The condition names are fixed at n_B, T, Y_C, Y_S, Y_Le, Y_Lmu (CLAUDE.md
    section 5); `leptons` is a flag rather than a condition and reaches the
    entry points as a named argument, so finding it here means a caller still
    routing it through the bag.
    """
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
    if "leptons" in conditions:
        raise TypeError("leptons is a flag, not a condition; pass it as the "
                        "named argument leptons=")
    extra = [k for k in conditions if k not in MODE_FRACTIONS[mode]]
    if extra:
        raise ValueError(f"mode {mode!r} does not take {extra}")
    return conditions


def eos_point(par, mode, species=None, n_B=None,
              T=None, SnB=None, leptons=None, x0=None, branches=None,
              patterns=None, backend="reference", **conditions):
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
    leptons : bool
        For the fixed-fraction modes: whether neutralizing leptons are added,
        so the total system is electrically neutral. With leptons=False the
        result is charged matter, which is what a mixed-phase construction
        needs per pure phase before imposing global neutrality. It is a flag,
        not a condition (CLAUDE.md section 3), so it is a named argument
        rather than a member of `conditions`.
    conditions :
        The fractions the mode fixes (Y_C, Y_S, Y_Le).
    """
    species = species if species is not None else SpeciesFlags()
    conditions = dict(_check(mode, dict(conditions)))
    if (T is None) == (SnB is None):
        raise ValueError("exactly one of T / SnB must be given")

    if SnB is not None:
        def entropy_at(temp):
            p = solve(par, mode, n_B, temp, species, x0, branches=branches,
                      patterns=patterns, leptons=leptons, backend=backend,
                      **conditions)
            return p.s / p.n_B if p.n_B else 0.0
        try:
            T = temperature_at_entropy(entropy_at, SnB)
        except (RuntimeError, ValueError) as err:
            # An entropy target the model cannot reach is a state that does
            # not exist, not a caller error: CLAUDE.md section 6 wants it
            # scored and stepped over, never raised.
            return PointResult(False, str(err))

    point = solve(par, mode, n_B, T, species, x0, branches=branches,
                  patterns=patterns, leptons=leptons, backend=backend,
                  **conditions)
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
              fixed=None, leptons=None, skip_errors=True, rows=False,
              progress=None, verbose=False, backend="reference",
              branches=None, patterns=None):
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
                     leptons=leptons, backend=backend,
                     branches=branches, patterns=patterns)
    return build_table(spec, skip_errors=skip_errors, rows=rows,
                       progress=progress, verbose=verbose)


#: The freezes `eos_response` implements. Each answers "what is held fixed
#: while the derivative is taken", which is what encodes which reactions are
#: faster than the perturbation. The composition freezes (held Y_i, held Y_C,
#: held Delta, held fields) and the susceptibility matrix chi_ab are recorded
#: in docs/DEFERRED.md.
RESPONSE_FREEZES = ("equilibrium",)


def eos_response(par, mode, species=None,
                 frozen="equilibrium", n_B=None, T=0.0, leptons=None,
                 rel_dn=1e-3, dT=0.05, branches=None, patterns=None,
                 backend="reference", **conditions):
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

    `leptons` is the same flag `eos_point` takes: a named argument rather
    than a member of `conditions` (CLAUDE.md section 5), and it holds through
    every point of the stencil.

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

    # One memo per call: the response functions share one 6-state stencil
    # and previously re-solved it 20 times between them. Identical arguments
    # re-solve to the identical point, so this changes no number.
    kwargs = dict(branches=branches, patterns=patterns, leptons=leptons,
                  backend=backend, _memo={}, **conditions)
    names = ("cs2_isothermal", "cs2_adiabatic", "branch_changed")
    if T > 0.0:
        names += ("C_V", "C_P", "Gamma_th")
    try:
        out = {"cs2_isothermal": _fd.sound_speed_isothermal(
            par, species, mode, n_B, T=T, rel_dn=rel_dn, dT=dT, **kwargs)}
        out["branch_changed"] = _fd.branch_changed(par, species, mode, n_B,
                                                   T=T, rel_dn=rel_dn,
                                                   **kwargs)
        if T > 0.0:
            out["cs2_adiabatic"] = _fd.sound_speed_adiabatic(
                par, species, mode, n_B, T=T, dT=dT, rel_dn=rel_dn, **kwargs)
            out["C_V"] = _fd.heat_capacity_V(par, species, mode, n_B, T, dT=dT,
                                             rel_dn=rel_dn, **kwargs)
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


def zero_pressure_point(par, species=None, n_lo=N_LO_DEFAULT,
                        n_hi=N_HI_DEFAULT, n_scan=N_SCAN_DEFAULT):
    """E/A at the self-bound surface: P = 0, T = 0, beta_eq_neutrinoless.

    A parametrization whose pressure crosses zero at finite density describes
    SELF-BOUND matter: the phase ends there with no crust below it, and

        E/A = eps/n_B

    at that density is the energy per baryon of a lump of it at rest. The pair
    of numbers this returns for the two flavour contents is the Bodmer-Witten
    window, and it is a two-sided gate on a parameter set:

        three-flavour E/A BELOW the 930.4 MeV of iron
            -- strange quark matter is absolutely stable;
        two-flavour E/A ABOVE it
            -- ordinary nuclei are not already decaying into quark matter.

    A set failing either is excluded, so both are wanted per sample. Which one
    this call returns is decided by `SpeciesFlags.two_flavour`:

        zero_pressure_point(par, SpeciesFlags())                    three
        zero_pressure_point(par, SpeciesFlags(two_flavour=True))    two

    THE CONTENT REQUESTED AND THE CONTENT FOUND CAN DIFFER, and the result
    carries both. A three-flavour request returns whatever strangeness the
    equilibrium actually populated at the surface, which is zero for a set
    whose surface sits below the s quark's threshold; read it off `Y_S`, not
    off `two_flavour`. Y_S is computed here through
    `eos.general.basis.quark_charges`, the shared conserved-charge map, from
    the solved flavour densities.

    The root find is `eos.general.zero_pressure.locate_zero_pressure` over
    this model's own `eos_point`: the locator takes the state as a callable,
    so it is one implementation for every model and imports none of them.
    Non-convergence, and a set with no self-bound surface at all, come back on
    the result (CLAUDE.md section 6); `below_iron` is reported, never
    asserted, because whether a set sits in the window is a property of the
    set rather than an invariant of the code.

    Returns a `ZeroPressurePoint`; test `.ok`.
    """
    flags = SpeciesFlags() if species is None else species

    def point_at(n_B):
        result = eos_point(par, "beta_eq_neutrinoless", flags, n_B=n_B, T=0.0)
        if not result.ok:
            return None
        p = result.point
        # n_B and n_S from the SOLVED flavour densities through the shared
        # conserved-charge map, not from the requested density and not from a
        # cached field: eps was computed at these densities, so this is the
        # n_B that makes eps/n_B the Gibbs energy per baryon the Euler
        # relation names.
        n_B_solved, _, n_S = quark_charges(p.n_u, p.n_d, p.n_s)
        return (p.P, p.eps / n_B_solved, p.mu_B,
                n_S / n_B_solved, p.mu_S)

    return locate_zero_pressure(point_at, two_flavour=flags.two_flavour,
                                n_lo=n_lo, n_hi=n_hi, n_scan=n_scan)
