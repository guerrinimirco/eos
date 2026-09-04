"""The equilibrium conditions of the NJL model, and the solves that close them.

`thermodynamics.py` computes quantities FROM a state; this module FINDS the
state. It is where the mode lives -- the conditions that pick one composition
out of the many `state_at` would evaluate -- and where the PATTERN lives, which
is the second thing this model has to choose and no other model in this
repository does.

A mode is a declaration (`eos.general.modes.ModeSpec`): per conserved charge,
either its fraction is imposed and its potential is an unknown, or its
potential is set by an equilibrium relation and the fraction comes out. The
unknown vector and the rows are assembled from that declaration, so the
equations are written once each and there is no per-mode residual to drift.
The vector is

    always   M_u, M_d, M_s, mu_B, mu_C
    plus     Delta_eta   for each gap the PATTERN makes free
    plus     mu_3, mu_8  whenever the pattern pairs at all
    plus     Sigma_V     where there is a vector coupling to carry
    plus     mu_S        iff the mode holds Y_S
    plus     mu_nue      iff the mode holds Y_Le

and the rows are, in order: three mass gap equations, one gap equation per
free gap, the two colour-neutrality rows where the pattern pairs, the vector
self-energy definition, the baryon density, and the mode's own charge rows.

A pattern is NOT a mode
-----------------------
Which diquark condensates are nonzero is not something a caller declares and
not something an equilibrium condition fixes: it is decided by which candidate
minimises the free energy at the given conditions. So `solve` enumerates
seeds -- unpaired, 2SC, CFL, and one asymmetric free seed that can land on
uSC, dSC or an unequal-gap state -- solves each to self-consistency, and
returns the survivor with the lowest f = eps - T s at the fixed density. The
comparison is by FREE ENERGY because these modes fix n_B; a comparison at
fixed mu_B (which is what `thermo_from_mu` and therefore `eos.mixed` do) is by
pressure instead. The two pick the same phase wherever one phase is stable,
and they CANNOT agree inside a first-order transition, because there no pure
phase is the ground state at fixed n_B -- a mixture is.

What this function returns there is the lower envelope of f(n_B) over the
candidates, and its crossing lies INSIDE the coexistence window rather than
at either edge of it: measured on `rg_njl1` in beta equilibrium at T = 0, the
2SC and CFL branches have equal f at n_B = 0.653 fm^-3, while equal pressure
at equal mu_B puts the transition at mu_B = 1370 MeV with the window
n_B = 0.583 -> 0.736 fm^-3 (3.6 -> 4.6 n_0, the fixed-potential number the
MUSES module and Kunkel et al. both report). So the switch in a fixed-n_B
sweep is NOT the transition, and the jump in P across it is not physics: the
window is located from the branches at fixed mu_B, and the plateau between
its edges comes from the construction (Maxwell, Gibbs or the eta-mixed phase
of `eos.mixed`), as CLAUDE.md section 8 requires before any table reaches a
structure solver. Pass `patterns=` to get one branch at a time, which is what
a construction needs.

Enumeration is necessary rather than tidy. The gap equation has three roots at
any Fermi-surface mismatch -- zero, a barrier maximum, and the physical BCS
root -- so a single Newton solve returns whichever root its seed was nearest,
silently. Every point reports the pattern it was solved in for the same reason
`eos.mixed` reports its window: it is part of the answer.

Two seeding facts, both from experience rather than taste:

  * CFL is electrically neutral WITHOUT electrons, so its seed puts mu_C at
    zero. Seeded with an electron-bearing potential the solve converges to a
    spurious point with an 11% flavour-density spread;
  * in an UNPAIRED region mu_8 is unconstrained -- n_8 vanishes identically at
    mu_8 = 0 -- so it is pinned there, never solved for.

Natural units inside, fm-based at the entry points: n_B arrives in fm^-3
because that is what a caller holds, and is converted once.
"""
from dataclasses import dataclass, field
import math

import numpy as np

from eos.general.basis import lepton_charges
from eos.general.modes import (
    beta_eq_neutrino_trapped, beta_eq_neutrinoless, electron_potential,
    fixed_YC, fixed_YC_YS, muon_potential, resolve_leptons,
)
from eos.general.physics_constants import hc3
from eos.general.solve import solve_system
from eos.general.thermodynamics_leptons import (
    ThermoResult, electron_thermo, muon_thermo, neutralizing_leptons,
    neutrino_thermo, photon_thermo,
)
from eos.njl.species import (
    DEFAULT_PATTERNS, PATTERNS, SpeciesFlags, pattern_mask, pattern_seed,
    realised_pattern,
)
from eos.njl.thermodynamics import (gap_seed_scale, has_vector, state_at,
                                    vacuum_solution)

# `backends/` is optional (CLAUDE.md section 5): with the directory gone, or
# numba absent, the fast backend hands MINPACK the residual alone and every
# number is the same, only slower.
try:
    from eos.njl.backends.jacobian import residual_jacobian
except ImportError:                       # pragma: no cover - backends/ removed
    residual_jacobian = None

#: The four modes of CLAUDE.md section 3, and the fractions each takes beyond
#: (n_B, T). Every one is closed here, at any temperature.
MODE_FRACTIONS = {
    "beta_eq_neutrinoless": (),
    "beta_eq_neutrino_trapped": ("Y_Le",),
    "fixed_YC": ("Y_C",),
    "fixed_YC_YS": ("Y_C", "Y_S"),
}

MODE_FACTORIES = {
    "beta_eq_neutrinoless": beta_eq_neutrinoless,
    "beta_eq_neutrino_trapped": beta_eq_neutrino_trapped,
    "fixed_YC": fixed_YC,
    "fixed_YC_YS": fixed_YC_YS,
}

_EMPTY = ThermoResult(n=0.0, P=0.0, e=0.0, s=0.0)


def mode_spec(mode, leptons=None, **fractions):
    """The `ModeSpec` for a named mode and its fractions.

    The names and the fractions are the repository's and the factories are
    `eos.general.modes`'s, so this model cannot invent a fifth mode or spell
    an existing one differently.
    """
    if mode not in MODE_FRACTIONS:
        raise ValueError(f"unknown mode {mode!r}; eos.njl closes "
                         f"{sorted(MODE_FRACTIONS)}")
    if "Y_Lmu" in fractions:
        raise NotImplementedError(
            "eos.njl does not trap the muon lepton family: "
            "beta_eq_neutrino_trapped takes (n_B, Y_Le, T) only "
            "(docs/DEFERRED.md)")
    expected, given = set(MODE_FRACTIONS[mode]), set(fractions)
    if given != expected:
        raise ValueError(f"mode {mode!r} takes fractions {sorted(expected)}; "
                         f"got {sorted(given)}")
    leptons = resolve_leptons(mode, leptons, default=False)
    if mode.startswith("beta_eq"):
        return MODE_FACTORIES[mode](**fractions)
    return MODE_FACTORIES[mode](leptons=leptons, **fractions)


# =============================================================================
# THE UNKNOWNS
# =============================================================================
def unknown_slots(par, spec, pattern):
    """The unknown vector's names, in order (see the module docstring)."""
    names = ["M_u", "M_d", "M_s"]
    mask = pattern_mask(pattern)
    names += [f"Delta_{eta + 1}" for eta in range(3) if mask[eta]]
    if any(mask):
        names += ["mu_3", "mu_8"]
    if has_vector(par):
        names.append("Sigma_V")
    names += ["mu_B", "mu_C"]
    if spec.is_fixed("S"):
        names.append("mu_S")
    if spec.is_fixed("L_e"):
        names.append("mu_nue")
    return tuple(names)


def _unpack(x, par, spec, pattern):
    """The state variables the unknown vector carries."""
    got = {name: float(value)
           for name, value in zip(unknown_slots(par, spec, pattern), x)}
    Delta = np.array([got.get(f"Delta_{eta + 1}", 0.0) for eta in range(3)])
    return (np.array([got["M_u"], got["M_d"], got["M_s"]]), Delta,
            got.get("mu_3", 0.0), got.get("mu_8", 0.0),
            got.get("Sigma_V", 0.0), got["mu_B"], got["mu_C"],
            got.get("mu_S", 0.0), got.get("mu_nue", 0.0))


def default_guess(par, spec, pattern, n_B, T, vac=None):
    """The cold start of one mode in one pattern.

    The baryon potential comes from the free massless relation at one flavour
    per colour, floored so a vanishing density does not give a vanishing
    potential; the masses interpolate from the broken vacuum towards the
    current masses; the gaps take the pattern's own seed at a tenth of the
    quark potential, which is well above the barrier root the gap equation
    also carries.

    mu_C starts slightly negative -- unpaired strange quark matter is
    negatively charged and needs electrons -- EXCEPT in CFL, which is neutral
    without them and whose seed therefore puts mu_C at zero (section 6.3 of
    the specification).
    """
    if vac is None:
        vac = vacuum_solution(par)
    n_q = max(3.0 * n_B * hc3, 1.0)
    mu_q = max((0.5 * math.pi ** 2 * n_q) ** (1.0 / 3.0), 50.0)

    x = min(mu_q / 400.0, 1.0)
    M = vac.M * (1.0 - x) + np.array(par.current_masses) * x
    guess = list(M)

    mask = pattern_mask(pattern)
    seed = pattern_seed(pattern, gap_seed_scale(mu_q))
    guess += [seed[eta] for eta in range(3) if mask[eta]]
    if any(mask):
        guess += [0.0, -0.02 * mu_q]
    if has_vector(par):
        guess.append(0.0)

    guess += [3.0 * mu_q, 0.0 if pattern == "CFL" else -0.05 * mu_q]
    if spec.is_fixed("S"):
        guess.append(0.0)
    if spec.is_fixed("L_e"):
        guess.append(0.3 * mu_q)
    return np.array(guess, dtype=float)


def seed_from(point, par, spec, pattern):
    """A seed for `pattern`, built from an already solved point of another one.

    The masses and the potentials of a converged unpaired state are a far
    better start for a paired solve than any analytic guess, because pairing
    moves them by percents while a cold guess is out by tens of percent. What
    the seed cannot take from the unpaired state is the gaps -- there are none
    -- so those come from the pattern's own declaration, and mu_8 is scaled to
    the gap, since the colour potential a paired phase needs is of the order
    of the gap it carries and is zero without one.

    CFL still overrides mu_C to zero: it is electrically neutral WITHOUT
    electrons, and starting it from the unpaired state's electron-bearing
    potential is what sends the solve to a spurious point.
    """
    M, _, _, _, Sigma_V, mu_B, mu_C, mu_S, mu_nue = _unpack(
        point.x, par, spec, point.pattern)
    gap_scale = gap_seed_scale(mu_B / 3.0)

    guess = list(M)
    mask = pattern_mask(pattern)
    seed = pattern_seed(pattern, gap_scale)
    guess += [seed[eta] for eta in range(3) if mask[eta]]
    if any(mask):
        guess += [0.0, -0.03 * gap_scale]
    if has_vector(par):
        guess.append(Sigma_V)
    guess += [mu_B, 0.0 if pattern == "CFL" else mu_C]
    if spec.is_fixed("S"):
        guess.append(mu_S)
    if spec.is_fixed("L_e"):
        guess.append(mu_nue)
    return np.array(guess, dtype=float)


def warm_start(point):
    """The seed taken from an already solved point, as {pattern: vector}.

    Keyed by pattern because the pattern decides the vector's LAYOUT -- a 2SC
    vector has one gap in it and a CFL vector three -- so a seed handed to the
    wrong pattern is not merely a poor guess, it is the wrong length.

    EVERY candidate that held its layout is carried, not only the winner. The
    danger a warm start runs in this model is not slowness but capture: seed a
    pattern from a density where it had COLLAPSED -- a CFL layout whose two
    s-quark gaps came back zero, which is the 2SC state -- and it stays on that
    root for the rest of the sweep, so a branch that exists is never found and
    every row is labelled with a pattern it is not in. `point._seeds` is built
    against exactly that (`solve`): a candidate contributes its vector only
    where `pattern_realised` agrees with the layout it was solved in, so a
    collapsed one starts cold at the next density and can be re-found the
    moment its root appears. Before that check existed the whole enumeration
    had to start cold to stay honest; it no longer does, and a paired sweep
    costs about half what it did.
    """
    if point._seeds:
        return {pattern: np.array(x, dtype=float)
                for pattern, x in point._seeds.items()}
    return {point.pattern: np.array(point.x, dtype=float)}


# =============================================================================
# THE LEPTON SECTOR
# =============================================================================
def lepton_block(mu_e, mu_nue, T, flags):
    """The leptons at given potentials: (blocks, n_charged, n_Le).

    Electrons always, muons where the flag allows them, at
    mu_mu = mu_e - mu_nue (muon decay equilibrium with a transparent muon
    family). Neutrinos only where they are trapped, mu_nue != 0; the
    free-streaming case carries no lepton number and no pressure, which is
    what mu_nue = 0 means.
    """
    electrons = electron_thermo(mu_e, T)
    muons = muon_thermo(muon_potential(mu_e, mu_nue), T) if flags.muons else _EMPTY
    neutrinos = neutrino_thermo(mu_nue, T) if mu_nue != 0.0 else _EMPTY
    return ((electrons, muons, neutrinos), electrons.n + muons.n,
            electrons.n + neutrinos.n)


def thermal_sectors(T, flags, trapped):
    """(P, eps, s) of the sectors carrying no conserved charge [fm-based].

    Photons, and the neutrino flavours not tracked in the composition: three
    where the electron neutrino is free-streaming, two where it is trapped,
    since the trapped flavour is already counted at its own potential.
    """
    P = e = s = 0.0
    if flags.photons:
        gamma = photon_thermo(T)
        P, e, s = P + gamma.P, e + gamma.e, s + gamma.s
    if flags.thermal_neutrinos:
        nu = neutrino_thermo(0.0, T)
        n_flavours = 2.0 if trapped else 3.0
        P += n_flavours * nu.P
        e += n_flavours * nu.e
        s += n_flavours * nu.s
    return P, e, s


# =============================================================================
# THE RESIDUAL
# =============================================================================
def _state(x, par, spec, pattern, T, vac, two_flavour=False,
           backend="reference", pair_nodes_per_panel=None):
    """The `NJLState` an unknown vector describes."""
    M, Delta, mu_3, mu_8, Sigma_V, mu_B, mu_C, mu_S, _ = _unpack(
        x, par, spec, pattern)
    return state_at(par, M, Delta, Sigma_V, mu_B, mu_C, mu_S, mu_3, mu_8, T,
                    vac=vac, pattern=pattern, two_flavour=two_flavour,
                    backend=backend,
                    pair_nodes_per_panel=pair_nodes_per_panel)


def _charge_rows(x, par, flags, spec, pattern, st, T):
    """The mode's own rows: one per fraction it holds, plus neutrality.

    In a fixed-fraction mode the neutralizing leptons are NOT a row: they are
    solved after the matter, from the charge the matter turned out to carry
    (`eos.general.thermodynamics_leptons.neutralizing_leptons`), because they
    feel no field the matter feels and nothing about them feeds back.
    """
    _, _, _, _, _, _, mu_C, _, mu_nue = _unpack(x, par, spec, pattern)

    # ONE lepton block serves both rows below. The neutrality row asks these
    # leptons for their charge and the Y_Le row asks the SAME ones for their
    # lepton number, and the trapped mode carries both -- so solving them
    # twice is one Fermi integral per residual spent to learn nothing.
    n_charged = n_Le = 0.0
    if not spec.is_fixed("C") or spec.is_fixed("L_e"):
        _, n_charged, n_Le = lepton_block(
            electron_potential(mu_C, mu_nue), mu_nue, T, flags)

    rows = []
    if spec.is_fixed("C"):
        rows.append(st.n_C - spec.targets["Y_C"] * st.n_B)
    else:
        rows.append(st.n_C - n_charged)
    if spec.is_fixed("S"):
        if T == 0.0 and spec.targets["Y_S"] == 0.0:
            # At T = 0 a vanishing strangeness fraction means an EMPTY strange
            # sector, and n_S = 0 then holds for every mu_S that keeps all
            # three s modes below threshold: the row is 0 = 0 across a whole
            # interval and its Jacobian column is null. Solved as written, the
            # iteration parks on the threshold kink -- the one place a
            # one-sided finite difference is nonzero -- and stalls decades
            # above the gate. The row is instead the boundary of that
            # interval, mu*_s = M_s in the most favourable colour: the T = 0
            # convention that an absent species sits at its onset, and the
            # value dF/dn_S takes as n_S -> 0+. A potential, judged against
            # mu_B (`residual_scales`).
            rows.append(float(np.max(st.mu_star[6:9])) - float(st.M[2]))
        else:
            rows.append(st.n_S - spec.targets["Y_S"] * st.n_B)
    if spec.is_fixed("L_e"):
        rows.append(n_Le - spec.targets["Y_Le"] * st.n_B)
    return rows


def residual(x, par, flags, spec, pattern, n_B, T, vac, backend="reference",
             pair_nodes_per_panel=None):
    """The equations of one mode in one pattern, in assembly order.

    `flags.two_flavour` reaches the state and nothing else: the rows are the
    same equations, evaluated on matter the s Fermi sea has left. The three
    mass rows STAY -- M_s is still determined, by its own gap equation with
    the medium term gone -- because the s condensate is a property of the
    vacuum rather than of the matter, and the 't Hooft determinant feeds it
    into M_u and M_d whether or not an s quark is populated.
    """
    st = _state(x, par, spec, pattern, T, vac, flags.two_flavour, backend,
                pair_nodes_per_panel)
    return rows_from_state(st, x, par, flags, spec, pattern, n_B, T)


def rows_from_state(st, x, par, flags, spec, pattern, n_B, T):
    """`residual`'s rows from a state already evaluated at x."""
    mask = pattern_mask(pattern)

    rows = list(st.mass_residual)
    rows += [st.gap_residual[eta] for eta in range(3) if mask[eta]]
    if any(mask):
        rows += [st.n_3, st.n_8]
    if has_vector(par):
        rows.append(st.vector_residual)

    rows.append(st.n_B - n_B)
    rows += _charge_rows(x, par, flags, spec, pattern, st, T)
    return rows


def residual_scales(par, spec, pattern, n_B, mu_scale, T=None):
    """The scale each row balances, so one tolerance means one thing.

    A mass row is a potential, judged against mu_B. A GAP row is not:
    Delta_eta/(2 G_D) has units of MeV^3, since G_D carries MeV^-2, so it is a
    density and is judged against the quark-density scale (mu_B/3)^3/pi^2 --
    as are the two colour rows. The density and charge rows are baryon
    densities in fm^-3.

    Without this the norm would be dominated by whichever row happens to carry
    the largest units, which here is the colour pair by twenty orders of
    magnitude; and judging a gap row against a potential rather than a density
    is four orders of magnitude too strict, which makes a perfectly converged
    solve report a residual of 1e-8.
    """
    mask = pattern_mask(pattern)
    n_scale = max(n_B, 1.0e-3)
    colour_scale = max((mu_scale / 3.0) ** 3 / math.pi ** 2, 1.0)
    scales = [mu_scale] * 3
    scales += [colour_scale] * sum(mask)
    if any(mask):
        scales += [colour_scale, colour_scale]
    if has_vector(par):
        scales.append(mu_scale)
    scales.append(n_scale)
    scales.append(n_scale)
    if spec.is_fixed("S"):
        # The T = 0, Y_S = 0 row of `_charge_rows` is a threshold potential,
        # not a density; every other strangeness row is a density.
        threshold = T == 0.0 and spec.targets["Y_S"] == 0.0
        scales.append(mu_scale if threshold else n_scale)
    if spec.is_fixed("L_e"):
        scales.append(n_scale)
    return scales


# =============================================================================
# ONE SOLVED POINT
# =============================================================================
@dataclass
class EoSPoint:
    """One solved NJL state, with the status a caller must test first.

    `converged` is judged on `error`, the largest equilibrium residual once
    each has been divided by the scale of the quantity it balances; the gate
    is `eos.general.solve.RESIDUAL_TOL`. When `converged` is False every other
    field holds the best iterate reached, which is not a physical state.

    Four fields exist only because this model pairs, and each is part of the
    answer rather than a diagnostic:

        pattern   which of the enumerated candidates won, by free energy. This
                  is the name of the LAYOUT the unknown vector was solved in,
                  which is what `x` has to be unpacked by (`seed_from`,
                  `warm_start`), and it is a statement about the equations
                  rather than about the phase;
        pattern_realised
                  the pattern the solved gaps ARE
                  (`eos.general.pairing.realised_pattern`). A free gap is free
                  to come out zero, so a solve requested in one layout can
                  converge on another -- a 'CFL' solve landing on 2SC is the
                  documented case -- and where the two names differ it is this
                  one that names the physics. A caller drawing a branch reads
                  this; a caller re-seeding reads `pattern`;
        Delta     the three gaps [MeV], zero where the pattern does not pair;
        gapless   whether a quasiparticle branch has reached zero. A gapless
                  state is physical, but comparing candidates by Omega across
                  one is not, so it is reported rather than silently ranked.
    """
    converged: bool = False
    error: float = 0.0
    mode: str = ""

    n_B: float = 0.0                # fm^-3
    T: float = 0.0                  # MeV
    # Conserved-charge fractions, MEASURED on the solved state: Y_X = n_X/n_B
    # for every charge (CLAUDE.md section 2). A mode that HOLDS one of these
    # reports what it solved, not what it was asked for, and every one of them
    # is defined in every mode -- Y_Le included, not only in the trapped mode
    # that holds it.
    Y_C: float = 0.0
    Y_S: float = 0.0
    Y_Le: float = 0.0               # electron family, (n_e + n_nue)/n_B

    pattern: str = "unpaired"
    pattern_realised: str = "unpaired"
    gapless: bool = False
    Delta: tuple = (0.0, 0.0, 0.0)  # MeV
    M: tuple = (0.0, 0.0, 0.0)      # MeV

    mu_B: float = 0.0               # MeV
    mu_C: float = 0.0
    mu_S: float = 0.0
    mu_3: float = 0.0
    mu_8: float = 0.0
    mu_e: float = 0.0
    mu_nu: float = 0.0

    n_u: float = 0.0                # fm^-3
    n_d: float = 0.0
    n_s: float = 0.0
    n_e: float = 0.0
    n_mu: float = 0.0
    n_nu: float = 0.0

    #: Colour and quark densities, fm^-3, lifted off the matter block so no
    #: caller has to reach into it. n_3 and n_8 are the colour densities whose
    #: vanishing is what makes a state colour-neutral; n_q is the total quark
    #: density (3 n_B, up to the pairing sector's own bookkeeping).
    n_3: float = 0.0                # fm^-3
    n_8: float = 0.0
    n_q: float = 0.0

    P: float = 0.0            # MeV/fm^3
    eps: float = 0.0
    s: float = 0.0            # fm^-3
    f: float = 0.0            # MeV/fm^3

    Y_u: float = 0.0
    Y_d: float = 0.0
    Y_s: float = 0.0
    Y_e: float = 0.0
    Y_nu: float = 0.0

    #: The matter block and the unknown vector that produced it. `_state` is
    #: INTERNAL and carries natural units [MeV^n]: it is the model's own
    #: working record, not part of the fm-based public boundary (CLAUDE.md
    #: section 5), and the leading underscore is what says so. Everything a
    #: caller needs is lifted onto this point in fm; `euler_residual()` is
    #: dimensionless and stays reachable through it.
    _state: object = None
    x: np.ndarray = field(default_factory=lambda: np.zeros(0))

    #: What the NEXT point of a sweep may re-seed from: {pattern: vector} for
    #: every candidate of this solve that both converged and stayed in the
    #: layout it was solved in. Empty on a point that came from `solve_pattern`
    #: directly, which enumerates nothing. See `warm_start`, its one consumer.
    #:
    #: UNDERSCORED for the same reason `_state` is: these are the model's own
    #: unknown vectors in natural units, scaffolding that reached the answer
    #: rather than part of it, so they stay off the fm-based boundary of
    #: CLAUDE.md section 5 -- and that underscore is the line the baseline
    #: flattener draws, which is what keeps a seeding change out of the frozen
    #: numbers.
    _seeds: dict = field(default_factory=dict)


def point_from_state(st, par, flags, spec, mode, x, converged, error, T):
    """Assemble the totals of one state: matter, leptons and the thermal gases.

    Where the mode holds Y_C the neutralizing leptons are solved here, from
    the charge the matter carries, at the single potential that makes the
    system neutral. Where the mode is a beta equilibrium they are already
    determined, by mu_e = mu_nue - mu_C.
    """
    mu_nue = float(x[-1]) if spec.is_fixed("L_e") else 0.0

    if spec.is_fixed("C"):
        if spec.leptons:
            mu_e, electrons, muons = neutralizing_leptons(
                st.n_C, T, include_muons=flags.muons)
            neutrinos = _EMPTY
        else:
            mu_e = 0.0
            electrons = muons = neutrinos = _EMPTY
    else:
        mu_e = electron_potential(st.mu_C, mu_nue)
        (electrons, muons, neutrinos), _, _ = lepton_block(
            mu_e, mu_nue, T, flags)

    P_th, e_th, s_th = thermal_sectors(T, flags, trapped=mu_nue != 0.0)
    P = st.P + electrons.P + muons.P + neutrinos.P + P_th
    e = st.eps + electrons.e + muons.e + neutrinos.e + e_th
    s = st.s + electrons.s + muons.s + neutrinos.s + s_th

    n_B = st.n_B
    per_B = (lambda n: n / n_B if n_B else 0.0)
    n_u, n_d, n_s = st.n_flavour / hc3
    n_Le, _ = lepton_charges(n_e=electrons.n, n_nue=neutrinos.n)
    return EoSPoint(
        converged=converged, error=error, mode=mode, n_B=n_B, T=T,
        Y_C=st.n_C / n_B if n_B else 0.0,
        Y_S=st.n_S / n_B if n_B else 0.0,
        Y_Le=per_B(n_Le),
        pattern=st.pattern, pattern_realised=realised_pattern(st.Delta),
        gapless=st.gapless,
        Delta=tuple(float(d) for d in st.Delta),
        M=tuple(float(m) for m in st.M),
        mu_B=st.mu_B, mu_C=st.mu_C, mu_S=st.mu_S, mu_3=st.mu_3, mu_8=st.mu_8,
        mu_e=mu_e, mu_nu=mu_nue,
        n_u=n_u, n_d=n_d, n_s=n_s,
        n_e=electrons.n, n_mu=muons.n, n_nu=neutrinos.n,
        P=P, eps=e, s=s, f=e - T * s,
        Y_u=per_B(n_u), Y_d=per_B(n_d), Y_s=per_B(n_s),
        Y_e=per_B(electrons.n), Y_nu=per_B(neutrinos.n),
        n_3=st.n_3 / hc3, n_8=st.n_8 / hc3, n_q=st.n_q / hc3,
        _state=st, x=np.asarray(x, dtype=float))


# =============================================================================
# THE SOLVE
# =============================================================================
def _refuse_fixed_YS(flags, spec, model):
    """A mode that HOLDS Y_S has no meaning with the strange sector off.

    With no species left carrying strangeness, n_S = 0 identically: the row
    n_S = Y_S n_B is unsatisfiable for Y_S != 0, and for Y_S = 0 it is
    satisfied at every mu_S at once. mu_S is then an unknown with a null
    Jacobian column -- `eos.general.basis.undetermined_potential`'s screen
    firing, and the failure that put one `eos.enjl` mode's residual within
    round-off of the acceptance gate and let round-off pick a chiral branch.

    Reaching two-flavour matter by asking for Y_S = 0 is disabling a sector
    through a fraction that happens to vanish, which CLAUDE.md section 4
    forbids in so many words. It is reached by switching the sector off, in
    `beta_eq_neutrinoless`.
    """
    if flags.two_flavour and spec.is_fixed("S"):
        raise NotImplementedError(
            f"{model}: a mode that holds Y_S has no state with "
            f"SpeciesFlags(two_flavour=True) -- no species left carries "
            f"strangeness, so the row is unsatisfiable for Y_S != 0 and "
            f"leaves mu_S undetermined for Y_S = 0. Two-flavour quark matter "
            f"is 'beta_eq_neutrinoless' with two_flavour=True")


def solve_pattern(par, mode, n_B, T, flags, pattern, spec=None, x0=None,
                  vac=None, backend="reference", pair_nodes_per_panel=None,
                  **fractions):
    """One mode in ONE declared pattern. The pattern is not chosen here.

    `backend` selects the flavour of the medium integrals and is passed
    straight down to `state_at`: 'reference' (the default, and what
    correctness is judged against) or 'fast'. `pair_nodes_per_panel` is the
    Gauss-Legendre node count of the pairing quadrature, likewise passed
    straight down; None keeps the shipped rule. See `eos.njl.eos_point`.
    """
    if spec is None:
        # None, not True: an unnamed flag means `resolve_leptons`'s default,
        # which is False in every model. Defaulting it True here would make
        # solve_pattern(..., "fixed_YC", Y_C=0.3) add a neutralizing electron
        # gas that eos_point() -- which passes the flag explicitly -- does not,
        # and the two spellings of one call would differ by 23 MeV/fm^3 in P.
        spec = mode_spec(mode, leptons=fractions.pop("leptons", None),
                         **fractions)
    _refuse_fixed_YS(flags, spec, 'eos.njl')
    if vac is None:
        vac = vacuum_solution(par)
    if not flags.csc and pattern != "unpaired":
        raise ValueError(
            f"pattern {pattern!r} needs SpeciesFlags(csc=True): with the "
            f"colour-superconducting sector off there are no gaps to solve for")

    cold = default_guess(par, spec, pattern, n_B, T, vac)
    warm = x0 is not None
    x0 = np.asarray(x0, dtype=float) if warm else cold

    last = {}

    def raw_rows(x):
        """`residual` at x, remembering the last state: the Jacobian below
        needs the state and the rows at the point it is asked for, which is
        always the point the residual was just evaluated at."""
        key = np.asarray(x, dtype=float).tobytes()
        if last.get("key") != key:
            st = _state(x, par, spec, pattern, T, vac, flags.two_flavour,
                        backend, pair_nodes_per_panel)
            last["key"] = key
            last["state"] = st
            last["raw"] = np.asarray(rows_from_state(st, x, par, flags, spec,
                                                     pattern, n_B, T))
        return last["raw"]

    def rows(x):
        """The rows ALREADY DIVIDED by their scales.

        The root finder terminates on its own view of the residual, and the
        raw rows here span twenty orders of magnitude -- masses in MeV against
        gap and colour rows in MeV^3. Handing it a dimensionless vector, and
        the matching tolerance, is what lets it drive every row to the gate
        rather than whichever one happens to be largest (CLAUDE.md's
        `solve_system(..., tol=...)`).
        """
        return [r / s for r, s in zip(raw_rows(x), scales_at(x))]

    def scales_at(x):
        mu_B = _unpack(x, par, spec, pattern)[5]
        return residual_scales(par, spec, pattern, n_B, max(abs(mu_B), 1.0),
                               T=T)

    def unit_scales(x):
        return [1.0] * len(unknown_slots(par, spec, pattern))

    jac = None
    if backend == "fast" and residual_jacobian is not None:
        names = unknown_slots(par, spec, pattern)

        def jac(x):
            """The Jacobian of `rows`: the analytic one, scaled like them.

            The scales themselves move with mu_B (`residual_scales` reads the
            potential scale off the iterate), so the quotient rule adds
            -r_i s_i'/s_i^2 to the mu_B column; at a root that term is zero,
            away from one it keeps the Jacobian consistent with `rows`.
            """
            raw = raw_rows(x)
            J = residual_jacobian(x, names, par, flags, spec, pattern, n_B, T,
                                  vac, pair_nodes_per_panel,
                                  state=last["state"])
            scales = np.asarray(scales_at(x))
            J = J / scales[:, None]
            mu_B = _unpack(x, par, spec, pattern)[5]
            if abs(mu_B) > 1.0:
                bumped = np.asarray(residual_scales(
                    par, spec, pattern, n_B, abs(mu_B) * (1.0 + 1.0e-6), T=T))
                dscale = (bumped - scales) / (1.0e-6 * abs(mu_B)) * np.sign(mu_B)
                J[:, names.index("mu_B")] -= raw * dscale / scales ** 2
            return J

    def reinflated(x):
        """x with its gaps reset to the pattern's seed, everything else kept."""
        x = np.array(x, dtype=float)
        names = unknown_slots(par, spec, pattern)
        seed = pattern_seed(pattern, gap_seed_scale(x[names.index("mu_B")] / 3.0))
        for eta in range(3):
            name = f"Delta_{eta + 1}"
            if name in names:
                x[names.index(name)] = seed[eta]
        return x

    def attempt(seed):
        """One seed through `solve_system`, and what the Jacobian path owes
        on top.

        The Newton attempt has a basin of its own, and from the unpaired seed
        at the 2SC -> CFL switch it lands somewhere the differenced hybrid
        method does not: at T = 20 MeV it converges onto the 2SC root of the
        CFL layout while the CFL root -- the ground state there by
        0.3 MeV/fm^3 in f -- sits in the other basin, and at T = 0 it stalls.
        A candidate an enumeration never sees is a wrong ground state, not a
        slow one, so two rescues follow, each measured on the switch points
        of a 200-point fixed-Y_C sweep against the differenced solver's
        tables:

          * a solve that CONVERGED but let a gap go -- left the layout it was
            asked for -- is re-seeded with its gaps reset to the pattern's
            seed and everything else kept, and solved again. From the
            near-root masses and potentials Newton reaches the CFL root in
            three or four steps where it exists (135-175 ms, against 600+ for
            the differenced solve), and re-collapses just as fast where it
            does not, which is the common case below the onset;
          * a solve that FAILED is followed by the differenced solve from the
            same seed, the path the reference tables were built with.

        An in-layout root from any of the three wins. This is the one place
        the fast backend pays twice, and a warm-started point never reaches
        it.
        """
        x, err, ok = solve_system(rows, seed, unit_scales, tol=1.0e-13, jac=jac)
        if jac is None:
            return x, err, ok
        if ok and _left_layout(x, par, spec, pattern):
            x_re, err_re, ok_re = solve_system(rows, reinflated(x), unit_scales,
                                               tol=1.0e-13, jac=jac)
            if ok_re and not _left_layout(x_re, par, spec, pattern):
                x, err, ok = x_re, err_re, ok_re
        elif not ok:
            x_fd, err_fd, ok_fd = solve_system(rows, seed, unit_scales,
                                               tol=1.0e-13)
            if ok_fd:
                x, err, ok = x_fd, err_fd, ok_fd
        return x, err, ok

    x, err, ok = attempt(x0)
    if not ok and warm:
        # A seed that lands in the right basin can still stall just above the
        # gate -- the CFL point at n_B = 1.2 fm^-3 stops at 8e-9 from a
        # continuation seed and reaches 3e-11 from the cold one, on the SAME
        # root. So the cold start is retried in full (both methods) rather
        # than as `solve_system`'s single fallback attempt, and the better of
        # the two is kept.
        x_cold, err_cold, ok_cold = attempt(cold)
        if err_cold < err:
            x, err, ok = x_cold, err_cold, ok_cold

    st = _state(x, par, spec, pattern, T, vac, flags.two_flavour, backend,
                pair_nodes_per_panel)
    return point_from_state(st, par, flags, spec, mode, x, ok, err, T)


def _left_layout(x, par, spec, pattern):
    """Did a 2SC or CFL solve come out in some other state? A free gap may
    be zero (`check_pattern_labels`), and a solve that let one go has found
    a root of a different pattern, which that pattern's own candidate finds
    more cheaply."""
    if pattern not in ("2SC", "CFL"):
        return False
    Delta = _unpack(x, par, spec, pattern)[1]
    return realised_pattern(Delta) != pattern


def patterns_for(flags, patterns=None):
    """The pairing patterns a solve enumerates, validated against the flags.

    The default enumeration is `DEFAULT_PATTERNS` with `csc` on and only
    `unpaired` with it off; `two_flavour` removes the patterns that condense a
    diquark containing an s quark, since with the strange sector off there is
    nothing to pair. An EXPLICITLY requested pattern the flags forbid raises --
    a malformed call is a programming error, not a candidate that loses on
    free energy -- and this function is one home for that judgement, so
    `TableSpec` can make it at construction time rather than letting
    `skip_errors` swallow it point by point inside a sweep.
    """
    if patterns is None:
        patterns = DEFAULT_PATTERNS if flags.csc else ("unpaired",)
        if flags.two_flavour:
            # A diquark containing an s quark is not a state two-flavour
            # matter has, so those patterns leave the ENUMERATION rather than
            # being solved and rejected. An explicitly requested one still
            # raises, below: the same split the `csc` flag already makes
            # between a candidate that loses on free energy and a call that
            # asks for a state the flags forbid.
            patterns = tuple(p for p in patterns
                             if not (pattern_mask(p)[0] or pattern_mask(p)[1]))
    elif flags.two_flavour and any(pattern_mask(p)[0] or pattern_mask(p)[1]
                                   for p in patterns):
        raise ValueError(
            f"patterns {tuple(patterns)!r} condense a diquark containing an "
            f"s quark (Delta_1 pairs d-s, Delta_2 pairs u-s), and "
            f"SpeciesFlags(two_flavour=True) leaves no s quark to pair. The "
            f"patterns two-flavour matter has are 'unpaired' and '2SC'")
    if not patterns:
        raise ValueError(
            "no pairing pattern survives SpeciesFlags(two_flavour=True) here; "
            "'unpaired' and '2SC' are the two-flavour patterns")
    elif not flags.csc and any(p != "unpaired" for p in patterns):
        # An explicitly requested pattern the flags forbid is a malformed
        # CALL, not a bad draw, so it raises here rather than being dropped by
        # the enumeration's own tolerance for candidates that do not solve.
        raise ValueError(
            f"patterns {tuple(patterns)!r} need SpeciesFlags(csc=True): with "
            f"the colour-superconducting sector off there are no gaps to "
            f"solve for")
    unknown = [p for p in patterns if p not in PATTERNS]
    if unknown:
        raise ValueError(f"unknown patterns {unknown}; eos.njl enumerates "
                         f"{sorted(PATTERNS)}")
    return tuple(patterns)


def solve(par, mode, n_B, T=0.0, flags=None, x0=None, patterns=None,
          vac=None, backend="reference", pair_nodes_per_panel=None,
          **fractions):
    """One mode at (n_B, T), with the pairing pattern chosen by free energy.

    Every enumerated pattern is solved to self-consistency and the converged
    candidates are ranked by f = eps - T s, the right potential at fixed
    density. A candidate that did not converge is dropped, not substituted; if
    none converged, the best iterate of the first candidate comes back with
    `converged = False`, which is a value a sampler can score (CLAUDE.md
    section 6) rather than an exception it has to catch.

    `x0` is either a bare vector, which seeds the first pattern tried, or a
    {pattern: vector} mapping as `warm_start` returns, which seeds each named
    pattern and leaves the rest cold. A seed belongs to the layout it was
    solved in, so it cannot simply be handed to whichever pattern comes next.

    The winner carries `_seeds` back out, holding the vector of every candidate
    that converged AND stayed in its own layout, so the next density can
    re-seed the whole enumeration rather than only the pattern that won. A
    candidate that collapsed is deliberately absent: it would seed the next
    solve onto the root it collapsed to and keep it there for the rest of the
    sweep. See `warm_start`.
    """
    if flags is None:
        flags = SpeciesFlags()
    # None means "not named", which `resolve_leptons` turns into False --
    # the same default every other model has, and the one `eos_point`
    # already passes down (CLAUDE.md section 4: a default never ADDS
    # physics to a call that did not ask for it).
    leptons = fractions.pop("leptons", None)
    spec = mode_spec(mode, leptons=leptons, **fractions)
    _refuse_fixed_YS(flags, spec, 'eos.njl')
    if vac is None:
        vac = vacuum_solution(par)
    patterns = patterns_for(flags, patterns)

    seeds = ({} if x0 is None else
             dict(x0) if isinstance(x0, dict) else {patterns[0]: x0})

    candidates = []
    # The first converged candidate seeds every later one that has no warm
    # start of its own. In the default enumeration that is the unpaired state,
    # which is the cheapest to reach and the closest thing to a continuation
    # the paired patterns can be given.
    reference = None
    for pattern in patterns:
        seed = seeds.get(pattern)
        if seed is None and reference is not None:
            seed = seed_from(reference, par, spec, pattern)
        try:
            point = solve_pattern(
                par, mode, n_B, T, flags, pattern, spec=spec, x0=seed,
                vac=vac, backend=backend,
                pair_nodes_per_panel=pair_nodes_per_panel)
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            continue
        candidates.append(point)
        if reference is None and point.converged:
            reference = point

    converged = [p for p in candidates if p.converged]
    if converged:
        winner = min(converged, key=lambda p: p.f)
        winner._seeds = {p.pattern: np.array(p.x, dtype=float)
                         for p in converged
                         if p.pattern_realised == p.pattern}
        return winner
    if candidates:
        return candidates[0]
    return solve_pattern(par, mode, n_B, T, flags, "unpaired", spec=spec,
                         vac=vac, backend=backend,
                         pair_nodes_per_panel=pair_nodes_per_panel)


def solve_beta_eq_neutrinoless(par, n_B, T=0.0, flags=None, x0=None,
                               **kwargs):
    """Beta equilibrium with free-streaming neutrinos. Variables (n_B, T)."""
    return solve(par, "beta_eq_neutrinoless", n_B, T, flags, x0, **kwargs)


def solve_beta_eq_neutrino_trapped(par, n_B, Y_Le, T=0.0, flags=None,
                                   x0=None, **kwargs):
    """Beta equilibrium with a trapped electron family. (n_B, Y_Le, T)."""
    return solve(par, "beta_eq_neutrino_trapped", n_B, T, flags, x0,
                 Y_Le=Y_Le, **kwargs)


def solve_fixed_yc(par, n_B, Y_C, T=0.0, flags=None, x0=None,
                   leptons=False, **kwargs):
    """Fixed non-leptonic charge fraction. Variables (n_B, Y_C, T)."""
    return solve(par, "fixed_YC", n_B, T, flags, x0, Y_C=Y_C,
                 leptons=leptons, **kwargs)


def solve_fixed_yc_ys(par, n_B, Y_C, Y_S, T=0.0, flags=None, x0=None,
                      leptons=False, **kwargs):
    """Fixed charge and strangeness. Variables (n_B, Y_C, Y_S, T)."""
    return solve(par, "fixed_YC_YS", n_B, T, flags, x0, Y_C=Y_C, Y_S=Y_S,
                 leptons=leptons, **kwargs)
