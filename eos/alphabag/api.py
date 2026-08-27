"""The uniform model API for alphaBag: eos_point, eos_table, eos_response.

Every model in this repository exposes these three entry points with the same
shape, so a caller -- a table pipeline, a sampler, a downstream package --
drives any model the same way:

    eos_point(par, mode, species, n_B=..., T=..., ...)   one state
    eos_table(par, mode, species, axes)                  a solved grid
    eos_response(par, mode, species, frozen=..., ...)    second derivatives

Modes are the repository's named equilibria, and alphaBag closes all four:
'beta_eq_neutrinoless', 'beta_eq_neutrino_trapped', 'fixed_YC', 'fixed_YC_YS'.
The fifth name it accepts, 'cfl', is not a mode but a PHASE: the
colour-flavour locked condensate is closed by flavour locking rather than by
an equilibrium condition, and it takes the pairing gap `Delta0` where a mode
takes a fraction. It is reached the same way because it is the same kind of
request -- a state at (n_B, T) -- and giving it a second entry point would
only mean two ways to ask for one thing.

Conditions are named exactly n_B, T (or SnB), Y_C, Y_S, Y_Le, plus Delta0 for
the paired phase. Units at this boundary are fm-based: n_B in fm^-3, T and
potentials in MeV, eps and P in MeV/fm^3.

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
from eos.alphabag.solver import MODE_FRACTIONS
from eos.general.basis import quark_charges
from eos.general.zero_pressure import (
    N_HI_DEFAULT, N_LO_DEFAULT, N_SCAN_DEFAULT, locate_zero_pressure,
)
from eos.alphabag.species import SpeciesFlags
from eos.alphabag.table import TableSpec, build_table, solve_at


@dataclass(frozen=True)
class PointResult:
    """One eos_point outcome: a convergence status the caller can test.

    When `ok` is False, `point` is None and `message` says what the solve
    reached.
    """
    ok: bool
    message: str
    point: object = None


def _check_conditions(mode, conditions):
    """Raise unless exactly the conditions this mode takes were supplied."""
    if mode not in MODE_FRACTIONS:
        raise ValueError(f"unknown mode {mode!r}; expected one of "
                         f"{list(MODE_FRACTIONS)}")
    if "Y_Lmu" in conditions:
        raise NotImplementedError(
            "alphaBag does not track the muon lepton family; "
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
        One of the keys of `eos.alphabag.solver.MODE_FRACTIONS`.
    species : SpeciesFlags
        The active degrees of freedom. Every flag defaults to False, `gluons`
        included: a sector is off unless the caller asks for it.
    n_B : float
        Baryon density [fm^-3].
    T, SnB : float
        Exactly one of temperature [MeV] or entropy per baryon; SnB adds an
        outer 1-D solve for T.
    leptons : bool, optional
        For the fixed-fraction modes: whether neutralizing electrons are added
        (n_e = n_C), so the total system is electrically neutral. With
        leptons=False -- the default here, since a quark phase is most often
        wanted as one half of a mixed phase -- the result is charged quark
        matter, which is what a mixed-phase construction needs per pure phase
        before imposing global neutrality. In the beta-equilibrium modes the
        leptons are constitutive rather than optional: leptons=True is
        redundant and is ignored, leptons=False raises. The paired phase is
        neutral with no electrons at all.
    conditions : the conditions the mode fixes (Y_C, Y_S, Y_Le, or Delta0).

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
            False, f"alphaBag {mode} solve did not converge at n_B={n_B}, "
                   f"T={T}: residual {point.error:.2e}")
    return PointResult(True, "converged", point)


def _entropy_per_baryon(point, n_B):
    """s/n_B of a solved point, for the entropy-axis outer solve."""
    if not point.converged:
        raise RuntimeError("the entropy solve stepped onto a state the "
                           "equilibrium solver could not reach")
    return point.s_total / n_B


def eos_table(par, mode, species=None, axes=None, fixed=None, leptons=None,
              skip_errors=True, rows=False, progress=None, verbose=False):
    """A solved grid over {n_B} x {T} [x condition axes].

    A thin wrapper over `eos.alphabag.table.build_table`: axes follow
    TableSpec (axes={'nB': grid, 'T': grid, optionally 'Y_C'/'Y_S'/'Y_Le', or
    'Delta0' for the paired phase), the density axis is warm-started with a
    bisected step through the strange-quark onset, and the result feeds
    `eos.astro.tov` and the plotting code directly. skip_errors=True drops
    non-converged points from their line -- the sampler-friendly default --
    rather than aborting.

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
        variable differentiated, since alphaBag's residual has no analytic
        Jacobian in this repository. C_V is returned only at T > 0, where it
        is defined.

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
            f"frozen={frozen!r} is not wired for alphaBag; implemented: "
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
        return (p.P_total, p.e_total / n_B_solved, p.mu_B,
                n_S / n_B_solved, p.mu_S)

    return locate_zero_pressure(point_at, two_flavour=flags.two_flavour,
                                n_lo=n_lo, n_hi=n_hi, n_scan=n_scan)
