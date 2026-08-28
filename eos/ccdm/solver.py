"""The equilibrium conditions of the colour-dielectric model, and the solves
that close them.

`thermodynamics.py` computes quantities FROM a state; this module FINDS the
state. It is where the mode lives -- the conditions that pick one composition
out of the many `state_at` would evaluate -- and where the two ENUMERATIONS
live, which is the thing this model has that the others do not.

A mode is a declaration (`eos.general.modes.ModeSpec`): per conserved charge,
either its fraction is imposed and its potential is an unknown, or its
potential is set by an equilibrium relation and the fraction comes out. The
unknown vector and the rows are assembled from that declaration, so the
equations are written once each and there is no per-mode residual to drift.
The vector is

    always   Phi, sigma, zeta, mu_B, mu_C
    plus     Sigma_V     where there is a vector coupling to carry
    plus     Delta_eta   for each gap the PATTERN makes free
    plus     mu_3, mu_8  whenever the pattern pairs at all
    plus     mu_S        iff the mode holds Y_S
    plus     mu_nue      iff the mode holds Y_Le

and the rows are, in order: the dilaton equation, the two scalar equations,
the vector self-energy definition, one gap equation per free gap, the two
colour-neutrality rows where the pattern pairs, the baryon density, and the
mode's own charge rows.

The closure rows of the specification map onto the repository's four modes
exactly, and the mapping is the whole of the translation:

    R1 cold star        beta_eq_neutrinoless        mu_S = 0, mu_e = -mu_C
    R3 proto-NS         beta_eq_neutrino_trapped    + Y_Le held
    R2 merger / CCSN    fixed_YC, leptons=True      LOCALLY NEUTRAL AND NOT
                                                    WEAK-EQUILIBRATED
    R4 heavy-ion        fixed_YC_YS, leptons=False
    R5 symmetric        fixed_YC_YS at Y_C = 1/2, Y_S = 0

R2 is the one worth stating twice: it imposes total electric neutrality
WITHOUT imposing beta equilibrium, which is what merger and supernova matter
is on a dynamical timescale. Weak equilibrium is a per-row closure here, never
an identity built into Omega -- hardwiring it would make that matter
unrepresentable.

Two enumerations, not a mode
----------------------------
Neither the chiral/dielectric BRANCH nor the pairing PATTERN is something a
caller declares or an equilibrium condition fixes: both are decided by which
candidate minimises the free energy at the given conditions. So `solve`
enumerates the product -- each branch seed crossed with each pattern seed --
solves every one to self-consistency, and returns the survivor with the lowest
f = eps - T s at the fixed density. The comparison is by FREE ENERGY because
these modes fix n_B; a comparison at fixed mu_B (which is what
`thermo_from_mu` and therefore `eos.mixed` do) is by pressure instead, and the
two agree.

A candidate that does not converge is DROPPED, never replaced by a neighbour:
substituting a converged neighbouring point is how a fake phase boundary
appears. Every point reports the branch and the pattern it was solved in, for
the same reason `eos.mixed` reports its window -- they are part of the answer.

Natural units inside, fm-based at the entry points: n_B arrives in fm^-3
because that is what a caller holds, and is converted once.
"""
from dataclasses import dataclass, field
import math

import numpy as np

from eos.ccdm.couplings import has_vector
from eos.ccdm.species import (
    DEFAULT_PATTERNS, DENSITY_BRANCHES, SpeciesFlags, branch_seed,
    pattern_mask, pattern_seed,
)
from eos.ccdm.thermodynamics import state_at
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
        raise ValueError(f"unknown mode {mode!r}; eos.ccdm closes "
                         f"{sorted(MODE_FRACTIONS)}")
    if "Y_Lmu" in fractions:
        raise NotImplementedError(
            "eos.ccdm does not trap the muon lepton family: "
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
    names = ["Phi", "sigma", "zeta"]
    if has_vector(par):
        names.append("Sigma_V")
    mask = pattern_mask(pattern)
    names += [f"Delta_{eta + 1}" for eta in range(3) if mask[eta]]
    if any(mask):
        names += ["mu_3", "mu_8"]
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
    return (got["Phi"], got["sigma"], got["zeta"], got.get("Sigma_V", 0.0),
            Delta, got.get("mu_3", 0.0), got.get("mu_8", 0.0),
            got["mu_B"], got["mu_C"], got.get("mu_S", 0.0),
            got.get("mu_nue", 0.0))


def default_guess(par, spec, branch, pattern, n_B, T):
    """The cold start of one mode in one branch and one pattern.

    The fields come from the branch's own seed, scaled to this parameter
    point's vacuum condensates rather than to the shipped one. The baryon
    potential comes from the free massless relation at one flavour per colour,
    floored so a vanishing density does not give a vanishing potential; the
    gaps take the pattern's own seed at a tenth of the quark potential, which
    is well above the barrier root the gap equation also carries.

    mu_C starts slightly negative -- unpaired strange quark matter is
    negatively charged and needs electrons -- EXCEPT in CFL, which is
    electrically neutral WITHOUT them and whose seed therefore puts mu_C at
    zero. Seeded with an electron-bearing potential a CFL solve converges to a
    spurious point.
    """
    n_q = max(3.0 * n_B * hc3, 1.0)
    mu_q = max((0.5 * math.pi ** 2 * n_q) ** (1.0 / 3.0), 50.0)

    Phi, sigma, zeta = branch_seed(par, branch)
    guess = [Phi, sigma, zeta]
    if has_vector(par):
        guess.append(0.0)

    mask = pattern_mask(pattern)
    seed = pattern_seed(pattern, max(0.1 * mu_q, 20.0))
    guess += [seed[eta] for eta in range(3) if mask[eta]]
    if any(mask):
        guess += [0.0, -0.02 * mu_q]

    guess += [3.0 * mu_q, 0.0 if pattern == "CFL" else -0.05 * mu_q]
    if spec.is_fixed("S"):
        # mu_S is seeded on the SIGN THAT SUPPRESSES STRANGENESS rather than
        # at zero. With S = +1 per s quark (CLAUDE.md section 2) the s modes
        # sit at mu*_s = mu_B/3 - mu_C/3 + mu_S - Sigma_V, so a negative mu_S
        # is what pushes them below their own effective mass, and seeded at
        # zero the strange sector starts fully populated. The target enters
        # linearly so a mode asking for strange matter is not handed a
        # suppressing seed.
        #
        # THIS DOES NOT MAKE Y_S = 0 CONVERGE, and it is not meant to. Once
        # M*_s rises above mu*_s the strange density is identically zero, the
        # strangeness row is satisfied for a whole RANGE of mu_S, and its
        # column of the Jacobian vanishes -- so the solve stalls on the
        # threshold with mu_S wherever its path reached. That is the
        # cross-cutting degeneracy of docs/DEFERRED.md ("A potential is only
        # pinned as tightly as its conjugate density responds"), which
        # eos.njl shows too; the seed only shortens the path to it.
        guess.append(-0.6 * mu_q * (1.0 - min(spec.targets["Y_S"], 1.0)))
    if spec.is_fixed("L_e"):
        guess.append(0.3 * mu_q)
    return np.array(guess, dtype=float)


def seed_from(point, par, spec, branch, pattern):
    """A seed for (branch, pattern), built from an already solved point.

    The fields and the potentials of a converged neighbouring candidate are a
    far better start than any analytic guess -- pairing and a change of
    pattern move them by percents where a cold guess is out by tens of
    percent. What the seed cannot take is the gaps of a pattern the reference
    did not carry, so those come from the pattern's own declaration, and mu_8
    is scaled to the gap, since the colour potential a paired phase needs is
    of the order of the gap it carries and is zero without one.

    THE FIELDS ARE NOT TAKEN ACROSS A BRANCH. A branch IS a choice of field
    root, so seeding the restored branch from the confined one's fields would
    hand it the very root it is meant to be an alternative to; when the branch
    differs the fields come from the branch's own seed and only the potentials
    are inherited.
    """
    (Phi, sigma, zeta, Sigma_V, _, _, _,
     mu_B, mu_C, mu_S, mu_nue) = _unpack(point.x, par, spec, point.pattern)
    if branch != point.branch:
        Phi, sigma, zeta = branch_seed(par, branch)
    gap_scale = max(0.1 * mu_B / 3.0, 20.0)

    guess = [Phi, sigma, zeta]
    if has_vector(par):
        guess.append(Sigma_V)

    mask = pattern_mask(pattern)
    seed = pattern_seed(pattern, gap_scale)
    guess += [seed[eta] for eta in range(3) if mask[eta]]
    if any(mask):
        guess += [0.0, -0.03 * gap_scale]

    guess += [mu_B, 0.0 if pattern == "CFL" else mu_C]
    if spec.is_fixed("S"):
        guess.append(mu_S)
    if spec.is_fixed("L_e"):
        guess.append(mu_nue)
    return np.array(guess, dtype=float)


def warm_start(point):
    """The seed taken from an already solved point, as {(branch, pattern): x}.

    Keyed by the candidate because the pattern decides the vector's LAYOUT --
    a 2SC vector has one gap in it and a CFL vector three -- so a seed handed
    to the wrong candidate is not merely a poor guess, it is the wrong length.
    A density sweep therefore carries the winning candidate's seed and lets
    the others start cold, which is also what keeps the enumeration honest: a
    candidate that only ever sees a warm start from itself can never be
    displaced.
    """
    return {(point.branch, point.pattern): np.array(point.x, dtype=float)}


# =============================================================================
# THE LEPTON SECTOR
# =============================================================================
def lepton_block(mu_e, mu_nue, T, flags):
    """The leptons at given potentials: (blocks, n_charged, n_Le).

    Electrons always, muons where the flag allows them, at
    mu_mu = mu_e - mu_nue (muon decay equilibrium with a transparent muon
    family). Neutrinos only where they are trapped, mu_nue != 0; the
    free-streaming case carries no lepton number and no pressure, which is
    what mu_nue = 0 means. Trapped neutrinos carry g = 1, not 2 -- they are
    left-handed only -- and `eos.general.thermodynamics_leptons` is where that
    is declared.
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
def _state(x, par, spec, branch, pattern, T, two_flavour=False):
    """The `CCDMState` an unknown vector describes."""
    (Phi, sigma, zeta, Sigma_V, Delta, mu_3, mu_8,
     mu_B, mu_C, mu_S, _) = _unpack(x, par, spec, pattern)
    return state_at(par, Phi, sigma, zeta, Sigma_V, Delta, mu_B, mu_C, mu_S,
                    mu_3, mu_8, T, branch=branch, pattern=pattern,
                    two_flavour=two_flavour)


def _charge_rows(x, par, flags, spec, pattern, st, T):
    """The mode's own rows: one per fraction it holds, plus neutrality.

    In a fixed-fraction mode the neutralizing leptons are NOT a row: they are
    solved after the matter, from the charge the matter turned out to carry
    (`eos.general.thermodynamics_leptons.neutralizing_leptons`), because they
    feel no field the matter feels and nothing about them feeds back.
    """
    *_, mu_C, _, mu_nue = _unpack(x, par, spec, pattern)
    rows = []
    if spec.is_fixed("C"):
        rows.append(st.n_C_fm - spec.targets["Y_C"] * st.n_B_fm)
    else:
        _, n_charged, _ = lepton_block(
            electron_potential(mu_C, mu_nue), mu_nue, T, flags)
        rows.append(st.n_C_fm - n_charged)
    if spec.is_fixed("S"):
        rows.append(st.n_S_fm - spec.targets["Y_S"] * st.n_B_fm)
    if spec.is_fixed("L_e"):
        _, _, n_Le = lepton_block(
            electron_potential(mu_C, mu_nue), mu_nue, T, flags)
        rows.append(n_Le - spec.targets["Y_Le"] * st.n_B_fm)
    return rows


def residual(x, par, flags, spec, branch, pattern, n_B, T):
    """The equations of one mode in one candidate, in assembly order.

    `flags.two_flavour` reaches the state and nothing else: the rows are the
    same equations on matter the s Fermi sea has left. The field rows STAY --
    sigma and zeta are still solved, with their medium terms gone -- because
    they are condensates of the model's vacuum rather than a population.
    """
    st = _state(x, par, spec, branch, pattern, T, flags.two_flavour)
    mask = pattern_mask(pattern)

    rows = list(st.field_residual[:3])
    if has_vector(par):
        rows.append(st.field_residual[3])
    rows += [st.gap_residual[eta] for eta in range(3) if mask[eta]]
    if any(mask):
        rows += [st.n_3, st.n_8]

    rows.append(st.n_B_fm - n_B)
    rows += _charge_rows(x, par, flags, spec, pattern, st, T)
    return rows


def residual_scales(par, spec, pattern, n_B, mu_scale):
    """The scale each row balances, so one tolerance means one thing.

    The dilaton row is an energy density, judged against B_g. The two scalar
    rows are scalar densities, judged against the explicit-breaking terms they
    balance in the vacuum. The vector row is a potential. The gap and colour
    rows are densities in MeV^3, judged against the quark-density scale
    (mu_B/3)^3/pi^2. The density and charge rows are baryon densities in
    fm^-3.

    Without this the norm would be dominated by whichever row happens to carry
    the largest units, which here is the dilaton row by twenty-two orders of
    magnitude over a charge row in fm^-3.
    """
    d = par.derived
    mask = pattern_mask(pattern)
    n_scale = max(n_B, 1.0e-3)
    colour_scale = max((mu_scale / 3.0) ** 3 / math.pi ** 2, 1.0)

    scales = [par.B_g, abs(d.eps_sigma), abs(d.eps_zeta)]
    if has_vector(par):
        scales.append(mu_scale)
    scales += [colour_scale] * sum(mask)
    if any(mask):
        scales += [colour_scale, colour_scale]
    scales.append(n_scale)
    scales += [n_scale] * (1 + int(spec.is_fixed("S"))
                           + int(spec.is_fixed("L_e")))
    return scales


# =============================================================================
# ONE SOLVED POINT
# =============================================================================
@dataclass
class EoSPoint:
    """One solved colour-dielectric state, with the status a caller must test
    first.

    `converged` is judged on `error`, the largest equilibrium residual once
    each has been divided by the scale of the quantity it balances; the gate
    is `eos.general.solve.RESIDUAL_TOL`. When `converged` is False every other
    field holds the best iterate reached, which is not a physical state.

    Five fields are part of the ANSWER rather than diagnostics:

        branch    which chiral/dielectric root won, by free energy;
        pattern   which pairing candidate won, by the same comparison;
        Delta     the three gaps [MeV], zero where the pattern does not pair;
        gapless   whether a quasiparticle branch has reached zero. A gapless
                  state is physical, but comparing candidates by Omega across
                  one is not, so it is reported rather than silently ranked;
        beyond_cutoff  whether the largest mode potential has passed
                  `Parameters.mu_ceiling`, where the PAIRING sector's sharp
                  cutoff no longer describes the Fermi surface it is cutting.
                  Declared rather than refused: the unpaired thermodynamics is
                  untouched by it, so the point is still worth returning.
    """
    converged: bool = False
    error: float = 0.0
    mode: str = ""

    n_B: float = 0.0                # fm^-3
    T: float = 0.0                  # MeV
    Y_C: float = 0.0
    Y_S: float = 0.0
    Y_L: float = 0.0

    branch: str = "restored"
    pattern: str = "unpaired"
    gapless: bool = False
    beyond_cutoff: bool = False
    Delta: tuple = (0.0, 0.0, 0.0)  # MeV
    M_star: tuple = (0.0, 0.0, 0.0)  # MeV

    phi_bar: float = 0.0
    chi: float = 0.0
    sigma: float = 0.0              # MeV
    zeta: float = 0.0               # MeV
    omega_0: float = 0.0            # MeV
    Sigma_R: float = 0.0            # MeV

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

    P_total: float = 0.0            # MeV/fm^3
    e_total: float = 0.0
    s_total: float = 0.0            # fm^-3
    f_total: float = 0.0            # MeV/fm^3

    Y_u: float = 0.0
    Y_d: float = 0.0
    Y_s: float = 0.0
    Y_e: float = 0.0
    Y_nu: float = 0.0

    #: The matter block and the unknown vector that produced it: the state,
    #: and the warm start for the next point.
    state: object = None
    x: np.ndarray = field(default_factory=lambda: np.zeros(0))


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
                st.n_C_fm, T, include_muons=flags.muons)
            neutrinos = _EMPTY
        else:
            mu_e = 0.0
            electrons = muons = neutrinos = _EMPTY
    else:
        mu_e = electron_potential(st.mu_C, mu_nue)
        (electrons, muons, neutrinos), _, _ = lepton_block(
            mu_e, mu_nue, T, flags)

    P_th, e_th, s_th = thermal_sectors(T, flags, trapped=mu_nue != 0.0)
    P = st.P_fm + electrons.P + muons.P + neutrinos.P + P_th
    e = st.eps_fm + electrons.e + muons.e + neutrinos.e + e_th
    s = st.s_fm + electrons.s + muons.s + neutrinos.s + s_th

    n_B = st.n_B_fm
    per_B = (lambda n: n / n_B if n_B else 0.0)
    n_u, n_d, n_s = st.n_flavour / hc3
    return EoSPoint(
        converged=converged, error=error, mode=mode, n_B=n_B, T=T,
        Y_C=st.n_C_fm / n_B if n_B else 0.0,
        Y_S=st.n_S_fm / n_B if n_B else 0.0,
        Y_L=per_B(electrons.n + neutrinos.n),
        branch=st.branch, pattern=st.pattern, gapless=st.gapless,
        beyond_cutoff=bool(np.max(np.abs(st.mu_star)) > par.mu_ceiling),
        # THE SIGN OF EACH GAP IS A GAUGE: Omega is invariant under flipping
        # any subset of the Delta_eta (verified -- the pairing correction is
        # identical to the last bit under all eight sign choices), and the gap
        # kernel flips with its gap, so -Delta is a root whenever Delta is.
        # The solve lands on whichever the seed was nearest; what is REPORTED
        # is the magnitude, so a table column and a plot mean one thing. The
        # signed values stay on `state` and in `x`, which is what a warm start
        # is built from.
        Delta=tuple(abs(float(d)) for d in st.Delta),
        M_star=tuple(float(m) for m in st.M_star),
        phi_bar=st.phi_bar, chi=st.chi, sigma=st.sigma, zeta=st.zeta,
        omega_0=st.omega_0, Sigma_R=st.Sigma_R,
        mu_B=st.mu_B, mu_C=st.mu_C, mu_S=st.mu_S, mu_3=st.mu_3, mu_8=st.mu_8,
        mu_e=mu_e, mu_nu=mu_nue,
        n_u=n_u, n_d=n_d, n_s=n_s,
        n_e=electrons.n, n_mu=muons.n, n_nu=neutrinos.n,
        P_total=P, e_total=e, s_total=s, f_total=e - T * s,
        Y_u=per_B(n_u), Y_d=per_B(n_d), Y_s=per_B(n_s),
        Y_e=per_B(electrons.n), Y_nu=per_B(neutrinos.n),
        state=st, x=np.asarray(x, dtype=float))


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


def solve_candidate(par, mode, n_B, T, flags, branch, pattern, spec=None,
                    x0=None, **fractions):
    """One mode in ONE declared branch and ONE declared pattern.

    Neither is chosen here: this is the solve an enumeration calls, and the
    choosing is `solve`'s.
    """
    if spec is None:
        spec = mode_spec(mode, leptons=fractions.pop("leptons", True),
                         **fractions)
    _refuse_fixed_YS(flags, spec, "eos.ccdm")
    if not flags.csc and pattern != "unpaired":
        raise ValueError(
            f"pattern {pattern!r} needs SpeciesFlags(csc=True): with the "
            f"colour-superconducting sector off there are no gaps to solve for")

    cold = default_guess(par, spec, branch, pattern, n_B, T)
    warm = x0 is not None
    x0 = np.asarray(x0, dtype=float) if warm else cold

    def rows(x):
        """The rows ALREADY DIVIDED by their scales.

        The root finder terminates on its own view of the residual, and the
        raw rows here span twenty-two orders of magnitude -- the dilaton row
        is an energy density in MeV^4 against a charge row in fm^-3. Handing
        it a dimensionless vector, and the matching tolerance, is what lets it
        drive every row to the gate rather than whichever one happens to be
        largest.
        """
        raw = residual(x, par, flags, spec, branch, pattern, n_B, T)
        return [r / s for r, s in zip(raw, scales_at(x))]

    def scales_at(x):
        mu_B = _unpack(x, par, spec, pattern)[7]
        return residual_scales(par, spec, pattern, n_B, max(abs(mu_B), 1.0))

    def unit_scales(x):
        return [1.0] * len(unknown_slots(par, spec, pattern))

    x, err, ok = solve_system(rows, x0, unit_scales, tol=1.0e-13)
    if not ok and warm:
        # A seed that lands in the right basin can still stall just above the
        # gate. The cold start is retried in full rather than as
        # `solve_system`'s single fallback attempt, and the better of the two
        # is kept.
        x_cold, err_cold, ok_cold = solve_system(rows, cold, unit_scales,
                                                 tol=1.0e-13)
        if err_cold < err:
            x, err, ok = x_cold, err_cold, ok_cold

    st = _state(x, par, spec, branch, pattern, T, flags.two_flavour)
    return point_from_state(st, par, flags, spec, mode, x,
                            bool(ok and st.valid), err, T)


def candidates_for(flags, branches=None, patterns=None):
    """The (branch, pattern) pairs a fixed-density solve enumerates.

    The product of the two enumerations, not a list either of them could
    supply on its own: which pairing pattern survives depends on the strange
    quark's effective mass, which is a property of the chiral branch, so the
    two cannot be chosen one after the other.
    """
    if branches is None:
        branches = DENSITY_BRANCHES
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
    return [(b, p) for b in branches for p in patterns]


def solve(par, mode, n_B, T=0.0, flags=None, x0=None, branches=None,
          patterns=None, **fractions):
    """One mode at (n_B, T), with the branch and the pattern chosen by free
    energy.

    Every enumerated candidate is solved to self-consistency and the converged
    ones are ranked by f = eps - T s, the right potential at fixed density. A
    candidate that did not converge is dropped, not substituted; if none
    converged, the best iterate of the first candidate comes back with
    `converged = False`, which is a value a sampler can score (CLAUDE.md
    section 6) rather than an exception it has to catch.

    `x0` is either a bare vector, which seeds the first candidate tried, or a
    {(branch, pattern): vector} mapping as `warm_start` returns, which seeds
    each named candidate and leaves the rest cold. A seed belongs to the
    layout it was solved in, so it cannot simply be handed to whichever
    candidate comes next.
    """
    if flags is None:
        flags = SpeciesFlags()
    leptons = fractions.pop("leptons", True)
    spec = mode_spec(mode, leptons=leptons, **fractions)
    _refuse_fixed_YS(flags, spec, "eos.ccdm")

    if patterns is not None and not flags.csc and any(
            p != "unpaired" for p in patterns):
        # An explicitly requested pattern the flags forbid is a malformed
        # CALL, not a bad draw, so it raises here rather than being dropped by
        # the enumeration's own tolerance for candidates that do not solve.
        raise ValueError(
            f"patterns {tuple(patterns)!r} need SpeciesFlags(csc=True): with "
            f"the colour-superconducting sector off there are no gaps to "
            f"solve for")

    wanted = candidates_for(flags, branches, patterns)
    seeds = ({} if x0 is None else
             dict(x0) if isinstance(x0, dict) else {wanted[0]: x0})

    solved = []
    # The first converged candidate seeds every later one that has no warm
    # start of its own: a converged neighbour is a far better start than any
    # analytic guess, and `seed_from` refuses to carry the FIELDS across a
    # branch, which is what keeps the branches independent.
    reference = None
    for branch, pattern in wanted:
        seed = seeds.get((branch, pattern))
        if seed is None and reference is not None:
            seed = seed_from(reference, par, spec, branch, pattern)
        try:
            point = solve_candidate(par, mode, n_B, T, flags, branch,
                                    pattern, spec=spec, x0=seed)
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            continue
        solved.append(point)
        if reference is None and point.converged:
            reference = point

    converged = [p for p in solved if p.converged]
    if converged:
        return min(converged, key=lambda p: p.f_total)
    if solved:
        return solved[0]
    return solve_candidate(par, mode, n_B, T, flags, wanted[0][0],
                           wanted[0][1], spec=spec)


def solve_beta_eq_neutrinoless(par, n_B, T=0.0, flags=None, x0=None,
                               **kwargs):
    """Beta equilibrium with free-streaming neutrinos. Variables (n_B, T).

    The specification's R1 row: mu_S = 0, mu_e = -mu_C, total electric
    neutrality including the leptons.
    """
    return solve(par, "beta_eq_neutrinoless", n_B, T, flags, x0, **kwargs)


def solve_beta_eq_neutrino_trapped(par, n_B, Y_Le, T=0.0, flags=None,
                                   x0=None, **kwargs):
    """Beta equilibrium with a trapped electron family. (n_B, Y_Le, T).

    The specification's R3 row. Trapped neutrinos carry g = 1.
    """
    return solve(par, "beta_eq_neutrino_trapped", n_B, T, flags, x0,
                 Y_Le=Y_Le, **kwargs)


def solve_fixed_yc(par, n_B, Y_C, T=0.0, flags=None, x0=None,
                   leptons=False, **kwargs):
    """Fixed non-leptonic charge fraction. Variables (n_B, Y_C, T).

    With leptons=True this is the specification's R2 row -- merger and
    supernova matter, LOCALLY NEUTRAL AND NOT WEAK-EQUILIBRATED. With
    leptons=False it is the charged pure phase a mixed-phase construction
    needs before imposing global neutrality.
    """
    return solve(par, "fixed_YC", n_B, T, flags, x0, Y_C=Y_C,
                 leptons=leptons, **kwargs)


def solve_fixed_yc_ys(par, n_B, Y_C, Y_S, T=0.0, flags=None, x0=None,
                      leptons=False, **kwargs):
    """Fixed charge and strangeness. Variables (n_B, Y_C, Y_S, T).

    The specification's R4 (heavy-ion) and R5 (symmetric, Y_C = 1/2, Y_S = 0)
    rows. Note the fractions are PER BARYON, so Y_S = n_S/n_B differs by a
    factor three from the per-quark n_s/(3 n_B) often plotted.
    """
    return solve(par, "fixed_YC_YS", n_B, T, flags, x0, Y_C=Y_C, Y_S=Y_S,
                 leptons=leptons, **kwargs)
