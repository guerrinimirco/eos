"""The equilibrium conditions of the extended NJL model, and the solve.

`thermodynamics.py` computes quantities FROM a state; this module FINDS the
state. It is where the mode lives: the conditions that pick one composition
out of the many `thermo_from_n` would evaluate.

A mode is a DECLARATION, not a function. `eos.general.modes.ModeSpec` says,
per conserved charge, whether its fraction is imposed (and its potential is an
unknown) or its potential is set by an equilibrium relation (and the fraction
comes out); `state_at` reads that declaration and assembles the unknown vector
and the rows from it. So the equations are written once each and there is no
per-mode residual to drift out of step. The unknown vector is

    always   M_u, M_d, M_s, mu_B, mu_C, n_B^Q,
             g_omega omega, g_rho rho, Sigma^R_b, Sigma^R_q
    plus     mu_S      iff the mode holds Y_S
    plus     mu_nue    iff the mode holds Y_Le

and the rows are the nine that never change -- three gap equations, baryon
number, the definition of n_B^Q, the two vector-field self-consistencies and
the two rearrangement definitions -- plus one row per held fraction and the
charge row, which is electric neutrality where C is equilibrated and
n_C = Y_C n_B where it is held.

The model of Xia, Phys. Rev. D 110, 014022 (2024) is a T = 0 model, so any
T > 0 raises. `enjl.tex` states the residual row by row.

Natural units throughout except where a name says `_fm`: n_B enters in fm^-3
at the entry points, because that is what a caller holds, and is converted
once.
"""
from dataclasses import dataclass
import math

from scipy.optimize import least_squares

from eos.enjl.parameters import Parameters
from eos.enjl.species import (
    BARYON_NUMBER, BARYONS, CHARGE, DEGENERACY, ISOSPIN, LEPTONS,
    NEUTRINO_FLAVOURS, QUARKS, SPECIES, STRANGENESS, SpeciesFlags,
    coupling_rescalings,
)
from eos.enjl.thermodynamics import (
    EoSPoint, _baryon_scalar_densities, baryon_masses,
    effective_scalar_densities, kinetic_state, mean_fields,
    quark_masses_from_gap, thermo_from_n,
)
from eos.general.modes import (
    ModeSpec, beta_eq_neutrino_trapped, beta_eq_neutrinoless, electron_potential,
    fixed_YC, fixed_YC_YS, has_leptons, muon_potential, strangeness_potential,
)
from eos.general.physics_constants import hc3
from eos.general.solve import RESIDUAL_TOL, scaled_residual_max
from eos.general.thermodynamics_leptons import neutralizing_leptons
from eos.general.tabulate import temperature_at_entropy

#: The four modes of CLAUDE.md section 3, as the factories that declare them.
#: Every one is reachable here; what this model refuses is a temperature, not
#: a mode.
MODE_FACTORIES = {
    "beta_eq_neutrinoless": beta_eq_neutrinoless,
    "beta_eq_neutrino_trapped": beta_eq_neutrino_trapped,
    "fixed_YC": fixed_YC,
    "fixed_YC_YS": fixed_YC_YS,
}

#: The fractions each mode takes beyond (n_B, T).
MODE_FRACTIONS = {
    "beta_eq_neutrinoless": (),
    "beta_eq_neutrino_trapped": ("Y_Le",),
    "fixed_YC": ("Y_C",),
    "fixed_YC_YS": ("Y_C", "Y_S"),
}


def check_mode(mode):
    """Raise unless `mode` is one of the repository's four."""
    if mode not in MODE_FACTORIES:
        raise ValueError(f"unknown mode {mode!r}; eos.enjl closes "
                         f"{sorted(MODE_FACTORIES)}")


def check_temperature(T):
    """Raise unless T >= 0.

    Every mode of this model is closed at any non-negative temperature. T = 0
    is not merely the T -> 0 limit of the rest: it keeps the exact closed
    forms of `eos.general.fermi_integrals`, which the JEL fit the T > 0 path
    uses does not converge back to (it steps off by ~1e-5 relative the moment
    T != 0 and stays there). That is a property of the fit, not of the port,
    and it is why a T -> 0 continuity check has a floor near 1e-4 -- see
    `eos.enjl.verify.run_full_check.check_entropy_limit`.
    """
    if T < 0.0:
        raise ValueError(
            f"temperature must be non-negative; got T = {T} MeV")


def mode_spec(mode, leptons=True, **fractions):
    """The `ModeSpec` for a named mode and its fractions.

    The names and the fractions are CLAUDE.md section 3's, and the factories
    are `eos.general.modes`'s, so this model cannot invent a fifth mode or
    spell an existing one differently.
    """
    check_mode(mode)
    expected = set(MODE_FRACTIONS[mode])
    given = set(fractions)
    if given != expected:
        raise ValueError(f"mode {mode!r} takes fractions {sorted(expected)}; "
                         f"got {sorted(given)}")
    if mode.startswith("beta_eq"):
        if not leptons:
            raise ValueError(
                "leptons=False has no meaning in beta equilibrium, which is "
                "defined by the leptons; it applies to fixed_YC and "
                "fixed_YC_YS, where it is the charged pure phase a "
                "mixed-phase construction needs")
        return MODE_FACTORIES[mode](**fractions)
    return MODE_FACTORIES[mode](leptons=leptons, **fractions)


# --------------------------------------------------------------------------
# The unknowns, and the guesses
# --------------------------------------------------------------------------

#: The unknowns every mode carries, in order. `enjl.tex` says why the fields
#: and the rearrangement terms are unknowns rather than recomputed inside the
#: residual: it makes every quantity entering any row come from one and the
#: same state, and replaces a nested fixed-point iteration by four more rows
#: of one outer solve. n_B^Q is carried for the same reason -- it enters the
#: baryon masses before the quark densities that define it are known.
BASE_UNKNOWNS = ("M_u", "M_d", "M_s", "mu_B", "mu_C", "n_bQ",
                 "gomega_omega", "grho_rho", "SigmaR_b", "SigmaR_q")


def strangeness_row_is_empty(spec, T):
    """Whether a held-Y_S row reads 0 = 0, leaving mu_S undetermined.

    Both strangeness carriers of this model have S = +1 -- the Lambda and the
    s quark, `species.STRANGENESS` -- and nothing in it carries S < 0. At
    T = 0 there are no antiparticles either, so every term of
    n_S = sum_i S_i n_i is non-negative and n_S = 0 forces n_Lambda = n_s = 0
    term by term. The row then vanishes identically, for every mu_S, and
    determines nothing.

    That is a T = 0 statement only: at T > 0 the Fermi tails populate both
    species (and their antiparticles, which carry S = -1), the row acquires a
    gradient in mu_S, and the potential is determined in the ordinary way.
    """
    return (spec.is_fixed("S") and T == 0.0
            and spec.targets["Y_S"] == 0.0)


def unknown_slots(spec):
    """The unknown-vector slot names implied by `spec`, in order.

    The ten of `BASE_UNKNOWNS`, then one potential per held charge. mu_C is
    NOT in that second group: it is carried always, because it is a potential
    of the matter whichever mode is asked -- what the declaration changes is
    the ROW that closes it, electric neutrality or n_C = Y_C n_B.
    """
    extra = []
    if spec.is_fixed("S"):
        extra.append("mu_S")
    if spec.is_fixed("L_e"):
        extra.append("mu_nue")
    return BASE_UNKNOWNS + tuple(extra)


def _unpack(x, spec):
    """(M_q, mu_B, mu_C, mu_S, mu_nue, inner) from the unknown vector."""
    M_q = dict(zip(QUARKS, x[:3]))
    mu_B, mu_C, n_bQ, g_w, g_r, SigmaR_b, SigmaR_q = x[3:10]
    i = 10
    if spec.is_fixed("S"):
        mu_S = x[i]
        i += 1
    else:
        mu_S = strangeness_potential(spec)
    mu_nue = x[i] if spec.is_fixed("L_e") else 0.0
    return (M_q, mu_B, mu_C, mu_S, mu_nue,
            (n_bQ, g_w, g_r, SigmaR_b, SigmaR_q))


def default_guess(mode, n_B_fm, par, spec=None, **fractions):
    """The cold starts for `mode`, in order of decreasing plausibility.

    Two parameter-free starting points, one on each side of the model's
    transitions: a nucleonic state with the vacuum constituent masses and no
    quarks, and a deconfined state with the current masses and the baryons
    dissolved. They are for use only when there is no previous point to
    continue from -- see `solve`.

    The charge potential is seeded at mu_C = -130 MeV for a beta-equilibrium
    mode, which is -mu_e at the electron potential neutron-rich matter reaches
    near saturation, and at the value symmetric matter would need for a held
    Y_C otherwise. A trapped mode seeds mu_nue from the density the neutrinos
    would carry if they held half of Y_Le n_B, which is the right order over
    the whole useful range of Y_Le: a fixed seed converges at Y_Le = 0.4 and
    misses at 0.2.
    """
    if spec is None:
        spec = mode_spec(mode, **fractions)
    n_B = n_B_fm * hc3
    g_w0 = par.Gamma_w(n_B) * 3.0 * n_B
    mu_B0 = 950.0 + 400.0 * n_B_fm
    mu_C0 = -130.0 if not spec.is_fixed("C") else -60.0

    # mu_S = 0 is the right seed only when no strangeness is demanded. Asked
    # for Y_S > 0 where nothing strange is populated, the residual has no
    # gradient to follow from there -- the Lambda and the s quark are both
    # below threshold and stay below it for any small step -- so a second
    # variant is emitted that drives strangeness in. At n_B = 0.3 fm^-3,
    # Y_C = 0.3, Y_S = 0.05 the answer is mu_S = 115 MeV, carried entirely by
    # the Lambda; seeded at zero the solve does not find it, seeded at 200 it
    # does.
    strangeness_seeds = [0.0]
    if spec.is_fixed("S") and spec.targets["Y_S"] > 0.0:
        strangeness_seeds.append(200.0)

    seeds = []
    for mu_S0 in strangeness_seeds:
        for masses, quark_fraction, charge in (
                ((367.6, 367.6, 549.5), 0.0, mu_C0),
                ((par.m_u0, par.m_d0, par.m_s0 + 100.0), 0.9 * n_B,
                 0.5 * mu_C0)):
            seed = [masses[0], masses[1], masses[2], mu_B0, charge,
                    quark_fraction, g_w0,
                    -0.1 * g_w0 if quark_fraction == 0.0 else -0.05 * g_w0,
                    0.0, 0.0]
            if spec.is_fixed("S"):
                seed.append(mu_S0)
            if spec.is_fixed("L_e"):
                n_nue = 0.5 * spec.targets["Y_Le"] * n_B
                seed.append((6.0 * math.pi ** 2 * n_nue) ** (1.0 / 3.0))
            seeds.append(seed)
    return seeds


def thermal_neutrino_flavours(spec, species):
    """How many mu = 0 neutrino flavours `species.thermal_neutrinos` adds.

    CLAUDE.md section 4 defines that flag as the flavours NOT tracked in the
    matter composition. This model tracks the electron neutrino, and only
    where a mode holds Y_Le: the muon family is transparent here
    (mu_mu = mu_e - mu_nue) and the tau family is never carried. So a
    neutrinoless mode leaves all three flavours to the thermal gas and a
    trapped one leaves two.

    Deciding this is the solver's job and not `thermodynamics.py`'s, which
    never knows which mode it is in; that module is handed the COUNT.
    """
    if not species.thermal_neutrinos:
        return 0
    return NEUTRINO_FLAVOURS - (1 if spec.is_fixed("L_e") else 0)


def warm_start(point):
    """The ten-or-more vector that seeds the next density of a sweep.

    `point` is a `BetaPoint`. The constituent masses are the natural
    continuation variables: they move smoothly where potential-based unknowns
    do not.
    """
    return list(point.x)


def _restored_branch(x0, par):
    """`x0` with the light condensates switched off.

    A first-order transition makes the solution discontinuous in n_B, so the
    previous point alone cannot cross one: seeded from the low-density side
    the solver is asked for a root several hundred MeV away in the quark
    masses, and lands nowhere. This is the same state with the light quark
    masses at their current values, which is where the chirally restored
    branch sits, and it is what carries a sweep ACROSS a chiral transition.
    """
    seed = list(x0)
    seed[0], seed[1] = par.m_u0, par.m_d0
    seed[2] = max(par.m_s0, x0[2] * 0.5)
    return seed


def _bounds(n_B_fm, spec):
    """Box for the unknowns, widened with density.

    The chemical potentials, the vector fields and the rearrangement terms all
    grow roughly linearly with n_B -- mu_B reaches 6.2 GeV and g_omega omega
    1.7 GeV already at n_B = 3.8 fm^-3 -- so a box calibrated at saturation
    density excludes the solution entirely above a few times n_sat. Only the
    quark masses have a genuine, density-independent ceiling (they are bounded
    above by their vacuum values) and n_B^Q a genuine one (it cannot exceed
    n_B).

    mu_C is negative in neutron-rich matter (mu_C = -mu_e in beta equilibrium)
    and positive where a held Y_C makes the matter proton-rich, so its box
    straddles zero.
    """
    big = 3000.0 + 3000.0 * n_B_fm
    lo = [0.0, 0.0, 0.0, 0.0, -2000.0, 0.0, -big, -big, -big, -big]
    hi = [1000.0, 1000.0, 1000.0, big, 2000.0, n_B_fm * hc3,
          big, big, big, big]
    if spec.is_fixed("S"):
        lo.append(-2000.0)
        hi.append(2000.0)
    if spec.is_fixed("L_e"):
        lo.append(0.0)
        hi.append(2000.0)
    return lo, hi


# --------------------------------------------------------------------------
# The residual
# --------------------------------------------------------------------------

def _massless_density(mu, g):
    """n [MeV^3] of a massless T = 0 gas, for the trapped neutrinos.

        n = g mu^3 / (6 pi^2),    mu > 0

    Left-handed neutrinos only, so g = 1.
    """
    if mu <= 0.0:
        return 0.0
    return g * mu ** 3 / (6.0 * math.pi ** 2)


def state_at(x, par, spec, n_B, T=0.0):
    """(densities, residuals) of the equilibrium system at `x`.

    The state is built forwards from the unknowns of `unknown_slots`: the
    density-dependent couplings at the TARGET density n_B (which the baryon
    number row makes equal to the state's own at the solution, and which keeps
    them out of the Jacobian), the baryon masses of Eq. (4), then every
    strongly interacting species' potential from the conserved-charge
    decomposition

        mu_i = B_i mu_B + C_i mu_C + S_i mu_S

    which is CLAUDE.md section 2's, and reduces to the paper's Eq. (23)
    mu_i = B_i mu_b - q_i mu_e in a beta-equilibrium mode, where mu_C = -mu_e
    and mu_S = 0. The lepton potentials come from the weak-equilibrium
    relations of `eos.general.modes` where C is equilibrated, and the leptons
    are absent from the residual where Y_C is held instead -- there n_C is
    already pinned by the charge row and the neutralizing leptons follow from
    it after the solve, entering no equation.

    Subtracting the vector and rearrangement shifts gives the kinetic
    potential nu_i, and `kinetic_state` turns that into n_i; n^s_i follows
    from the same nu. This is the FORWARD direction, potentials to densities,
    so it inverts nothing and costs no more at T > 0 than at T = 0: the
    excursion clamp is the one difference, and it moves from kF onto nu
    because at T > 0 there is no sharp Fermi surface to clamp.

    The rows, in order: the gap equation for each flavour (Eq. 5); baryon
    number against the target; the definition of n_B^Q; the two vector-field
    self-consistencies (Eqs. 9-10); the two rearrangement definitions
    (Eqs. 17-18); then the charge row and one row per held fraction.

    Natural units: n_B in MeV^3.
    """
    M_q, mu_B, mu_C, mu_S, mu_nue, inner = _unpack(x, spec)
    n_bQ, g_w, g_r, SigmaR_b, SigmaR_q = inner
    f = coupling_rescalings(par)
    alpha_S = par.alpha_S(n_B)
    M_b = baryon_masses(par, M_q, alpha_S, n_bQ)
    m_l = {"e": par.m_e, "mu": par.m_mu}
    mass = {**M_b, **M_q, **m_l}

    leptonic = has_leptons(spec) and not spec.is_fixed("C")
    if leptonic:
        mu_e = electron_potential(mu_C, mu_nue)
        mu_lepton = {"e": mu_e, "mu": muon_potential(mu_e, mu_nue)}
    else:
        mu_lepton = {"e": 0.0, "mu": 0.0}

    nu = {}
    n = {}
    for sp in SPECIES:
        if sp in LEPTONS:
            mu_i = mu_lepton[sp]
            shift = 0.0
        else:
            mu_i = (BARYON_NUMBER[sp] * mu_B + CHARGE[sp] * mu_C
                    + STRANGENESS[sp] * mu_S)
            if sp in BARYONS:
                shift = f[sp] * (3.0 * g_w + ISOSPIN[sp] * g_r) + SigmaR_b
            else:
                shift = f[sp] * (g_w + ISOSPIN[sp] * g_r) + SigmaR_q
        nu[sp], _, n[sp] = kinetic_state(mu_i - shift, mass[sp],
                                         DEGENERACY[sp], T)

    n_s_b = _baryon_scalar_densities(nu, M_b, T)
    fields = mean_fields(n, n_s_b, M_q, par, n_B)
    nbar = effective_scalar_densities(nu, M_q, n_s_b, alpha_S, par.Lambda, T)
    gap = quark_masses_from_gap(nbar, par)

    res = [M_q[q] - gap[q] for q in QUARKS]
    res.append(sum(BARYON_NUMBER[sp] * n[sp] for sp in SPECIES) - n_B)
    res.append(n_bQ - (n["u"] + n["d"] + n["s"]) / 3.0)
    res.append(g_w - fields.gomega_omega)
    res.append(g_r - fields.grho_rho)
    res.append(SigmaR_b - fields.SigmaR_b)
    res.append(SigmaR_q - fields.SigmaR_q)

    # --- the charge row, and one row per held fraction ---
    if spec.is_fixed("C"):
        n_C = sum(CHARGE[sp] * n[sp] for sp in SPECIES if sp not in LEPTONS)
        res.append(n_C - spec.targets["Y_C"] * n_B)
    else:
        res.append(sum(CHARGE[sp] * n[sp] for sp in SPECIES))
    if strangeness_row_is_empty(spec, T):
        # mu_S is pinned rather than solved for, because nothing here
        # determines it (`strangeness_row_is_empty`). Carried as an unknown
        # with no equation it is a null column in the Jacobian, and the cost
        # is not cosmetic: the least-squares termination tests fire early on
        # the rank-deficient problem and leave the scaled residual of the
        # WHOLE solve four decades higher than the model's other modes reach,
        # close enough to `eos.general.solve.RESIDUAL_TOL` that round-off
        # decides which side of it a point lands on. Zero is the value the
        # solve already returned here, since no gradient ever moved it off
        # its seed.
        res.append(mu_S)
    elif spec.is_fixed("S"):
        n_S = sum(STRANGENESS[sp] * n[sp] for sp in SPECIES)
        res.append(n_S - spec.targets["Y_S"] * n_B)
    if spec.is_fixed("L_e"):
        n_nue = _massless_density(mu_nue, 1.0)
        res.append(n["e"] + n_nue - spec.targets["Y_Le"] * n_B)
    return n, res


def residual(x, par, spec, n_B, T=0.0):
    """The equations that must vanish; see `state_at`."""
    return state_at(x, par, spec, n_B, T)[1]


def residual_scales(par, spec, n_B, T=0.0):
    """The scale each row of `residual` balances, so one gate means one thing.

    The rows carry mixed units -- MeV for the mass gaps, the fields and the
    rearrangement terms, MeV^3 for the density, charge and fraction conditions
    -- and a norm of the raw vector would be dominated by whichever row
    happens to be largest. Dividing each by the scale of the quantity it
    balances makes the components comparable, which is what
    `eos.general.solve.RESIDUAL_TOL` is then applied to.
    """
    scales = [100.0, 100.0, 100.0,                     # quark-mass gaps [MeV]
              n_B, n_B,                                # n_B, n_B^Q [MeV^3]
              par.Gamma_w(n_B) * 3.0 * n_B,            # g_omega omega [MeV]
              par.Gamma_r(n_B) * n_B,                  # g_rho rho [MeV]
              3000.0, 1000.0,                          # Sigma^R [MeV]
              n_B]                                     # the charge row
    if strangeness_row_is_empty(spec, T):
        scales.append(100.0)              # a pinned potential [MeV]
    elif spec.is_fixed("S"):
        scales.append(n_B)
    if spec.is_fixed("L_e"):
        scales.append(n_B)
    return scales


def _scaled_residual(x, par, spec, n_B, T=0.0):
    scales = residual_scales(par, spec, n_B, T)
    return [r / s for r, s in zip(residual(x, par, spec, n_B, T), scales)]


# --------------------------------------------------------------------------
# The neutralizing leptons of a held-charge mode
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# The solve
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class BetaPoint:
    """One solved state at a fixed total baryon density.

    Named for the mode this model was written around; it is what every mode
    returns. `densities` are in fm^-3 and `eps`/`P` in MeV/fm^3, the fm-based
    public units; masses and chemical potentials are in MeV in both systems.
    `point` is the underlying `EoSPoint` in natural units, carrying everything
    `thermo_from_n` returns, and `x` is the converged unknown vector, which is
    what warm-starts the next density. `s` is the entropy density [fm^-3] and
    `T` the temperature [MeV] it was solved at.

    `seed` names the starting point the accepted root was reached from --
    "warm", "restored" or "cold". Along a sweep it should read "warm" at every
    density but the first: the other two are on a DIFFERENT branch by
    construction, so a point reporting one of them mid-sweep is a point where
    the continuation changed branch, and reading it off the result is what
    distinguishes a branch that ended from a branch that was displaced.
    """
    converged: bool
    error: float
    n_b_fm: float
    T: float
    spec: ModeSpec
    densities: dict
    M_q: dict
    M_b: dict
    eps: float
    P: float
    s: float
    mu_b: float
    mu_e: float
    mu_C: float
    mu_S: float
    point: EoSPoint
    x: tuple
    seed: str = "warm"

    @property
    def EperB(self):
        return self.point.EperB


def solve(mode, n_B_fm, par=None, x0=None, cold_start=True, leptons=True,
          T=0.0, species=None, **fractions):
    """One equilibrium solve at n_B [fm^-3], for any of the four modes.

    The mode declaration decides the unknowns and the rows (`state_at`);
    everything else is common. Bounded least squares, accepted on the scaled
    residual of `residual_scales` at the repository's common gate
    `eos.general.solve.RESIDUAL_TOL`. Bounded rather than the shared
    `solve_system`: the box of `_bounds` is what keeps the iteration off the
    unphysical regions (negative masses, n_B^Q above n_B) that an unbounded
    root find walks into at these scales.

    Returns *a* root, namely the first one reached from the starting points
    tried in order. Above a first-order transition there is more than one: the
    deconfined branch and the metastable baryonic branch both satisfy the
    local equations over a finite density range, and which is found depends on
    where the search began. Selecting the stable one needs both branches at
    once -- a construction -- and is not done here or in `eos.enjl.table`.

    Parameters:
        mode:       one of CLAUDE.md section 3's four names.
        n_B_fm:     total baryon density [fm^-3].
        par:        Parameters (default: the shipped set).
        x0:         starting guess in the order of `unknown_slots(spec)`,
                    normally the previous point of a density sweep through
                    `warm_start`. It is the first of several starting points
                    tried, not the only one.
        cold_start: whether the parameter-free starts of `default_guess` may
                    be tried. They must be left out once a sweep is under way:
                    a cold start that happens to converge lands on whichever
                    branch it lands on, so allowing it mid-sweep lets the
                    sequence hop between branches from one density to the
                    next, which shows up as an equation of state that
                    oscillates rather than one that has a transition in it.
        leptons:    for the fixed-fraction modes, whether the neutralizing
                    electrons and muons are added. They enter no equation
                    either way; with leptons=False the result is charged
                    matter, which is what a mixed-phase construction needs per
                    pure phase.
        T:          temperature [MeV].
        species:    SpeciesFlags. Only `photons` and `thermal_neutrinos` are
                    the caller's here, and neither enters an equation of the
                    solve: they carry no conserved charge, so they are added
                    to eps, P and s once the composition is found.
        fractions:  the mode's own conditions, named Y_C, Y_S, Y_Le.

    Returns:
        BetaPoint.
    """
    if par is None:
        par = Parameters.default()
    check_temperature(T)
    T = float(T)
    species = SpeciesFlags() if species is None else species
    spec = mode_spec(mode, leptons=leptons, **fractions)
    n_B = n_B_fm * hc3
    lo, hi = _bounds(n_B_fm, spec)
    x_scale = [100.0, 100.0, 100.0, 100.0, 100.0, n_B,
               100.0, 100.0, 3000.0, 1000.0]
    if spec.is_fixed("S"):
        x_scale.append(100.0)
    if spec.is_fixed("L_e"):
        x_scale.append(100.0)

    seeds = []
    if x0 is not None:
        seeds.append(("warm", list(x0)))
        seeds.append(("restored", _restored_branch(x0, par)))
    if cold_start:
        seeds.extend(("cold", g)
                     for g in default_guess(mode, n_B_fm, par, spec=spec))

    solved, from_seed, tried, best_error = None, None, 0, float("inf")
    already = []
    for name, seed in seeds:
        seed = [min(max(v, l), h) for v, l, h in zip(seed, lo, hi)]
        if any(all(abs(a - b) <= 1e-9 * max(1.0, abs(b))
                   for a, b in zip(seed, other)) for other in already):
            continue                      # duplicate of a start already tried
        already.append(seed)
        tried += 1
        sol = least_squares(
            lambda x: _scaled_residual(x, par, spec, n_B, T), seed,
            bounds=(lo, hi), x_scale=x_scale,
            xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=1500)
        error = scaled_residual_max(residual(sol.x, par, spec, n_B, T),
                                    residual_scales(par, spec, n_B, T))
        best_error = min(best_error, error)
        if error <= RESIDUAL_TOL:
            solved, from_seed = sol.x, name
            break

    if solved is None:
        raise RuntimeError(
            f"ENJL {mode} solve did not converge at n_B={n_B_fm:.4f} fm^-3 "
            f"after {tried} starting points; best scaled residual "
            f"{best_error:.3e} against a {RESIDUAL_TOL:.0e} bound")

    n, _ = state_at(solved, par, spec, n_B, T)
    _, mu_B, mu_C, mu_S, mu_nue, _ = _unpack(solved, spec)

    # Where Y_C is held, the neutralizing leptons were not part of any row;
    # they are added now, from the charge density the solve pinned.
    if spec.is_fixed("C") and spec.leptons:
        n_C = sum(CHARGE[sp] * n[sp] for sp in SPECIES if sp not in LEPTONS)
        # The shared solve is fm-based; this assembly is in natural units, and
        # ENJL carries its own lepton masses (section 7), so it hands them over
        # rather than letting the general values silently replace them.
        mu_e, e_blk, mu_blk = neutralizing_leptons(
            n_C / hc3, T, include_muons=True, m_e=par.m_e, m_mu=par.m_mu)
        n["e"], n["mu"] = e_blk.n * hc3, mu_blk.n * hc3
    elif spec.is_fixed("C"):
        mu_e = 0.0
    else:
        mu_e = electron_potential(mu_C, mu_nue)

    # The reported state comes from one code path whichever mode was asked:
    # the composition just found is handed to `thermo_from_n`, seeded with the
    # constituent masses so it stays on the branch that was solved.
    point = thermo_from_n(
        n, par=par, T=T, x0=list(solved[:3]), photons=species.photons,
        thermal_neutrinos=thermal_neutrino_flavours(spec, species))
    return BetaPoint(
        converged=True, error=best_error,
        n_b_fm=n_B_fm, T=T, spec=spec,
        densities={k: v / hc3 for k, v in n.items()},
        M_q=point.M_q, M_b=point.M_b,
        eps=point.eps / hc3, P=point.P / hc3, s=point.s / hc3,
        mu_b=mu_B, mu_e=mu_e, mu_C=mu_C, mu_S=mu_S,
        point=point, x=tuple(solved), seed=from_seed,
    )


def solve_at_entropy(mode, n_B_fm, SnB, par=None, T_lo=0.2, T_hi=80.0,
                     **kwargs):
    """The state whose entropy per baryon is `SnB`: an outer 1-D solve for T.

    CLAUDE.md section 3 accepts entropy per baryon wherever it accepts a
    temperature, and this is what that means: an isentrope is an isotherm
    whose temperature is solved for at every density. s/n_B is monotone
    increasing in T at fixed density, so the bracket is well posed; the
    bracketing and the outer root are `eos.general.tabulate`'s, shared with
    every other model that takes the axis.

    SnB = 0 is answered directly at T = 0 rather than through the bracket. It
    is the only entropy the exact T = 0 branch reaches, and no positive
    bracket contains it.

    Every evaluation of the bracket is a FULL equilibrium solve, so this costs
    the outer iteration count times `solve`; `kwargs` carries `x0` through, so
    a sweep's warm start seeds every one of them.

    Raises RuntimeError if the entropy is out of reach, which is what the
    public boundary turns into a status.
    """
    if SnB == 0.0:
        return solve(mode, n_B_fm, par=par, T=0.0, **kwargs)

    def entropy_per_baryon_at(T):
        point = solve(mode, n_B_fm, par=par, T=T, **kwargs)
        return point.s / point.n_b_fm

    try:
        T = temperature_at_entropy(entropy_per_baryon_at, SnB,
                                   T_lo=T_lo, T_hi=T_hi)
    except ValueError as err:
        raise RuntimeError(
            f"ENJL could not bracket s/n_B = {SnB} at n_B={n_B_fm:.4f} fm^-3 "
            f"between T = {T_lo} and {T_hi} MeV: {err}") from err
    return solve(mode, n_B_fm, par=par, T=T, **kwargs)


def solve_beta_eq_neutrinoless(n_B_fm, par=None, x0=None, cold_start=True):
    """Beta equilibrium with free-streaming neutrinos, charge neutral.

        mu_i = B_i mu_b - q_i mu_e     Eq. (23)
        sum_i q_i n_i = 0              Eq. (24)

    i.e. mu_C + mu_e = 0 and mu_S = 0 in the conserved-charge basis.
    """
    return solve("beta_eq_neutrinoless", n_B_fm, par=par, x0=x0,
                 cold_start=cold_start)


def solve_beta_eq_neutrino_trapped(n_B_fm, Y_Le, par=None, x0=None,
                                   cold_start=True):
    """Beta equilibrium with the electron lepton family trapped at Y_Le.

    mu_nue becomes an unknown, closed by (n_e + n_nue)/n_B = Y_Le, and the
    beta relation reads mu_C + mu_e = mu_nue. The neutrinos are massless and
    left-handed, g = 1. The muon family stays transparent.
    """
    return solve("beta_eq_neutrino_trapped", n_B_fm, par=par, x0=x0,
                 cold_start=cold_start, Y_Le=Y_Le)


def solve_fixed_yc(n_B_fm, Y_C, par=None, x0=None, cold_start=True,
                   leptons=True):
    """Fixed non-leptonic charge fraction, n_C = Y_C n_B. Strangeness still
    self-equilibrates, mu_S = 0."""
    return solve("fixed_YC", n_B_fm, par=par, x0=x0, cold_start=cold_start,
                 leptons=leptons, Y_C=Y_C)


def solve_fixed_yc_ys(n_B_fm, Y_C, Y_S, par=None, x0=None, cold_start=True,
                      leptons=True):
    """Fixed charge and strangeness. Y_C = 0.5, Y_S = 0 is symmetric nuclear
    matter; mu_S becomes an unknown, determined by the strangeness demanded."""
    return solve("fixed_YC_YS", n_B_fm, par=par, x0=x0, cold_start=cold_start,
                 leptons=leptons, Y_C=Y_C, Y_S=Y_S)
