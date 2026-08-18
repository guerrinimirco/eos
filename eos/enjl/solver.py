"""The equilibrium conditions of the extended NJL model, and the solve.

`thermodynamics.py` computes quantities FROM a state; this module FINDS the
state. It is where the mode lives: the conditions that pick one composition
out of the many `thermo_from_n` would evaluate.

One mode is closed here, `beta_eq_neutrinoless` -- weak equilibrium with
free-streaming neutrinos and total electric neutrality, at T = 0. The other
three of the repository's four raise naming the physics that is missing
(`MODE_REFUSALS`), and so does any T > 0; the model of Xia, Phys. Rev. D 110,
014022 (2024) is a T = 0 model.

Reading order, which is the physics: the guesses, the residual, the solve.
`enjl.tex` states the residual row by row, with the scale each row is judged
on and the unknown vector it is a function of.

Natural units throughout except where a name says `_fm`: n_B enters in fm^-3
at the entry points, because that is what a caller holds, and is converted
once.
"""
from dataclasses import dataclass
import math

from scipy.optimize import least_squares

from eos.enjl.parameters import Parameters
from eos.enjl.species import (
    BARYON_NUMBER, BARYONS, CHARGE, DEGENERACY, ISOSPIN, QUARKS,
    SPECIES, coupling_rescalings,
)
from eos.enjl.thermodynamics import (
    EoSPoint, baryon_masses, effective_scalar_densities, kinetic_thermo,
    mean_fields, n_from_kF, quark_masses_from_gap, thermo_from_n,
)
from eos.general.physics_constants import hc3
from eos.general.solve import RESIDUAL_TOL, scaled_residual_max

#: The modes this model does not close, and what is missing from each. They
#: raise rather than returning a state, and `enjl.tex` gives the same reasons
#: at length.
MODE_REFUSALS = {
    "beta_eq_neutrino_trapped":
        "neutrinos are not among this model's degrees of freedom and the "
        "equilibrium condition mu_i = B_i mu_b - q_i mu_e has no "
        "lepton-family potential in it: mu_nu = 0 is built in. A trapped "
        "mode fixes Y_Le, and there is no mu_nue for it to fix -- closing it "
        "needs an eleventh unknown and one more row",
    "fixed_YC":
        "Y_C is the NON-LEPTONIC charge fraction, so imposing it replaces "
        "the neutrality row by n_C = Y_C n_B. With leptons off that is a "
        "one-row change and mu_e becomes -mu_C; with leptons on, total "
        "neutrality n_e + n_mu = n_C is a separate condition and the lepton "
        "potential parts company with mu_C, so the system gains an unknown. "
        "The mode's leptons flag requires both, and half of it is not a mode",
    "fixed_YC_YS":
        "the same for Y_C, and in addition mu_S = 0 identically here: weak "
        "equilibrium does not conserve strangeness, so the model as closed "
        "has no potential conjugate to n_S and Y_S is an output. Imposing "
        "Y_S means promoting mu_S to an unknown, which turns the two-potential "
        "condition into the full mu_i = B_i mu_B + C_i mu_C + S_i mu_S for "
        "all six strongly interacting species",
}

#: The mode this model closes, and the conditions it takes beyond (n_B, T).
MODE_FRACTIONS = {"beta_eq_neutrinoless": ()}


def check_mode(mode):
    """Raise unless `mode` is one this model closes."""
    if mode in MODE_FRACTIONS:
        return
    if mode in MODE_REFUSALS:
        raise NotImplementedError(
            f"eos.enjl does not close mode {mode!r}: {MODE_REFUSALS[mode]}")
    raise ValueError(f"unknown mode {mode!r}; eos.enjl closes "
                     f"{sorted(MODE_FRACTIONS)}, and refuses "
                     f"{sorted(MODE_REFUSALS)}")


def check_temperature(T):
    """Raise unless T = 0, the only temperature this model has."""
    if T != 0.0:
        raise NotImplementedError(
            f"eos.enjl is a T = 0 model: every kinetic expression in it is a "
            f"zero-temperature closed form and s = 0 identically; got "
            f"T = {T} MeV")


# --------------------------------------------------------------------------
# The unknowns, and the guesses
# --------------------------------------------------------------------------

#: The ten unknowns of the beta-equilibrium system, in the order the residual
#: and every guess use. `enjl.tex` Sec. "The residual, row by row" says why
#: the fields and the rearrangement terms are carried as unknowns rather than
#: recomputed inside the residual: it makes every quantity entering any row
#: come from one and the same state, and replaces a nested fixed-point
#: iteration by four more rows of one outer solve.
UNKNOWNS = ("M_u", "M_d", "M_s", "mu_b", "mu_e", "n_bQ",
            "gomega_omega", "grho_rho", "SigmaR_b", "SigmaR_q")


def default_guess(mode, n_B_fm, par):
    """The cold starts for `mode`, in order of decreasing plausibility.

    Two parameter-free starting points, one on each side of the model's
    transitions: a nucleonic state with the vacuum constituent masses and no
    quarks, and a deconfined state with the current masses and the baryons
    dissolved. They are for use only when there is no previous point to
    continue from -- see `solve_beta_eq_neutrinoless`.
    """
    check_mode(mode)
    n_B = n_B_fm * hc3
    g_w0 = par.Gamma_w(n_B) * 3.0 * n_B
    mu_b0 = 950.0 + 400.0 * n_B_fm
    return [
        [367.6, 367.6, 549.5, mu_b0, 130.0, 0.0, g_w0, -0.1 * g_w0, 0.0, 0.0],
        [par.m_u0, par.m_d0, par.m_s0 + 100.0, mu_b0, 100.0, 0.9 * n_B,
         g_w0, -0.05 * g_w0, 0.0, 0.0],
    ]


def warm_start(point):
    """The ten-vector that seeds the next density of a sweep from `point`.

    `point` is a `BetaPoint`. The constituent masses are the natural
    continuation variables: they move smoothly where potential-based unknowns
    do not.
    """
    state = point.point
    return [state.M_q["u"], state.M_q["d"], state.M_q["s"],
            point.mu_b, point.mu_e, state.n_bQ,
            state.gomega_omega, state.grho_rho,
            state.SigmaR_b, state.SigmaR_q]


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


def _bounds(n_B_fm):
    """Box for the ten unknowns, widened with density.

    The chemical potentials, the vector fields and the rearrangement terms all
    grow roughly linearly with n_B -- mu_b reaches 6.2 GeV and g_omega omega
    1.7 GeV already at n_B = 3.8 fm^-3 -- so a box calibrated at saturation
    density excludes the solution entirely above a few times n_sat. Only the
    quark masses have a genuine, density-independent ceiling (they are bounded
    above by their vacuum values) and n_b^Q a genuine one (it cannot exceed
    n_B).
    """
    big = 3000.0 + 3000.0 * n_B_fm
    lo = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -big, -big, -big, -big]
    hi = [1000.0, 1000.0, 1000.0, big, 2000.0, n_B_fm * hc3,
          big, big, big, big]
    return lo, hi


# --------------------------------------------------------------------------
# The residual
# --------------------------------------------------------------------------

def _fermi_momentum(nu, m):
    """Fermi momentum with the clamps a root finder's excursions need.

    Physically kF is at most a few thousand MeV even at the highest densities
    used here (n_B ~ 10 fm^-3 gives kF ~ 2200 MeV), so clamping avoids float
    overflow on an off-track iterate while never binding at a real solution.
    """
    if not (math.isfinite(nu) and math.isfinite(m)) or m <= 0.0:
        return 0.0
    if nu > 5000.0:
        return 5000.0
    if nu <= m:
        return 0.0
    k2 = (nu - m) * (nu + m)
    if not math.isfinite(k2) or k2 <= 0.0:
        return 0.0
    return min(math.sqrt(k2), 5000.0)


def state_at(x, par, n_B):
    """(densities, residuals) of the beta-equilibrium system at `x`.

    The state is built forwards from the unknowns of `UNKNOWNS`: the
    density-dependent couplings at the TARGET density n_B (which the baryon
    number row makes equal to the state's own at the solution, and which keeps
    them out of the Jacobian), the baryon masses of Eq. (4), then every
    species' potential from the equilibrium condition

        mu_i = B_i mu_b - q_i mu_e                                  Eq. (23)

    with q_i the physical electric charge, LEPTONS INCLUDED. Subtracting the
    vector and rearrangement shifts gives the kinetic potential nu_i, hence
    kF_i, n_i and n^s_i.

    The ten residuals, in order: the gap equation for each flavour (Eq. 5);
    baryon number against the target; electric neutrality (Eq. 24, the sum
    running over leptons); the definition of n_b^Q; the two vector-field
    self-consistencies (Eqs. 9-10); and the two rearrangement definitions
    (Eqs. 17-18).

    Natural units: n_B in MeV^3.
    """
    M_q = dict(zip(QUARKS, x[:3]))
    mu_b, mu_e, n_bQ, g_w, g_r, SigmaR_b, SigmaR_q = x[3:]
    f = coupling_rescalings(par)
    alpha_S = par.alpha_S(n_B)
    M_b = baryon_masses(par, M_q, alpha_S, n_bQ)
    m_l = {"e": par.m_e, "mu": par.m_mu}
    mass = {**M_b, **M_q, **m_l}

    kF = {}
    n = {}
    for sp in SPECIES:
        mu_i = BARYON_NUMBER[sp] * mu_b - CHARGE[sp] * mu_e
        if sp in BARYONS:
            shift = f[sp] * (3.0 * g_w + ISOSPIN[sp] * g_r) + SigmaR_b
        elif sp in QUARKS:
            shift = f[sp] * (g_w + ISOSPIN[sp] * g_r) + SigmaR_q
        else:
            shift = 0.0
        kF[sp] = _fermi_momentum(mu_i - shift, mass[sp])
        n[sp] = n_from_kF(kF[sp], DEGENERACY[sp])

    n_s_b = {b: kinetic_thermo(math.sqrt(kF[b] ** 2 + M_b[b] ** 2),
                               M_b[b], DEGENERACY[b], 0.0)[4]
             for b in BARYONS}
    fields = mean_fields(n, n_s_b, M_q, par, n_B)
    nbar = effective_scalar_densities(kF, M_q, n_s_b, alpha_S, par.Lambda)
    gap = quark_masses_from_gap(nbar, par)

    res = [M_q[q] - gap[q] for q in QUARKS]
    res.append(sum(BARYON_NUMBER[sp] * n[sp] for sp in SPECIES) - n_B)
    res.append(sum(CHARGE[sp] * n[sp] for sp in SPECIES))
    res.append(n_bQ - (n["u"] + n["d"] + n["s"]) / 3.0)
    res.append(g_w - fields.gomega_omega)
    res.append(g_r - fields.grho_rho)
    res.append(SigmaR_b - fields.SigmaR_b)
    res.append(SigmaR_q - fields.SigmaR_q)
    return n, res


def residual(x, par, n_B):
    """The ten equations that must vanish; see `state_at`."""
    return state_at(x, par, n_B)[1]


def residual_scales(par, n_B):
    """The scale each row of `residual` balances, so one gate means one thing.

    The rows carry mixed units -- MeV for the mass gaps, the fields and the
    rearrangement terms, MeV^3 for the density and charge conditions -- and a
    norm of the raw vector would be dominated by whichever row happens to be
    largest. Dividing each by the scale of the quantity it balances makes the
    components comparable, which is what `eos.general.solve.RESIDUAL_TOL` is
    then applied to.
    """
    return [100.0, 100.0, 100.0,                       # quark-mass gaps [MeV]
            n_B, n_B, n_B,                             # densities [MeV^3]
            par.Gamma_w(n_B) * 3.0 * n_B,              # g_omega omega [MeV]
            par.Gamma_r(n_B) * n_B,                    # g_rho rho [MeV]
            3000.0, 1000.0]                            # Sigma^R [MeV]


def _scaled_residual(x, par, n_B):
    scales = residual_scales(par, n_B)
    return [r / s for r, s in zip(residual(x, par, n_B), scales)]


# --------------------------------------------------------------------------
# The solve
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class BetaPoint:
    """One beta-equilibrium state at a fixed total baryon density.

    `densities` are in fm^-3 and `eps`/`P` in MeV/fm^3, the fm-based public
    units; masses and chemical potentials are in MeV in both systems. `point`
    is the underlying `EoSPoint` in natural units, carrying everything
    `thermo_from_n` returns.
    """
    converged: bool
    error: float
    n_b_fm: float
    densities: dict
    M_q: dict
    M_b: dict
    eps: float
    P: float
    mu_b: float
    mu_e: float
    point: EoSPoint

    @property
    def EperB(self):
        return self.point.EperB


def solve_beta_eq_neutrinoless(n_B_fm, par=None, x0=None, cold_start=True):
    """Beta-equilibrium, charge-neutral uniform matter at n_B [fm^-3], T = 0.

        mu_i = B_i mu_b - q_i mu_e     Eq. (23)
        sum_i q_i n_i = 0              Eq. (24)

    solved simultaneously with the gap equation and the vector-field
    self-consistencies; ten unknowns (`UNKNOWNS`), bounded least squares,
    accepted on the scaled residual of `residual_scales` at the repository's
    common gate `eos.general.solve.RESIDUAL_TOL`. Bounded least squares rather
    than the shared `solve_system`: the box of `_bounds` is what keeps the
    iteration off the unphysical regions (negative masses, n_b^Q above n_B)
    that an unbounded root find walks into at these scales.

    Returns *a* root, namely the first one reached from the starting points
    tried in order. Above a first-order transition there is more than one: the
    deconfined branch and the metastable baryonic branch both satisfy the
    local equations over a finite density range, and which is found depends on
    where the search began. Selecting the stable one needs both branches at
    once -- a Maxwell construction -- and is not done here or in
    `eos.enjl.table`. Call this directly only when a single local root is what
    is wanted.

    Parameters:
        n_B_fm:     total baryon density [fm^-3].
        par:        Parameters (default: the shipped set).
        x0:         starting guess in the order of `UNKNOWNS`, normally the
                    previous point of a density sweep through `warm_start`. It
                    is the first of several starting points tried, not the
                    only one.
        cold_start: whether the parameter-free starts of `default_guess` may
                    be tried. They must be left out once a sweep is under way:
                    a cold start that happens to converge lands on whichever
                    branch it lands on, so allowing it mid-sweep lets the
                    sequence hop between branches from one density to the
                    next, which shows up as an equation of state that
                    oscillates rather than one that has a transition in it.

    Returns:
        BetaPoint. Non-convergence is a return value here as at the public
        boundary: `converged` is False and `error` carries the best scaled
        residual reached.
    """
    if par is None:
        par = Parameters.default()
    n_B = n_B_fm * hc3
    lo, hi = _bounds(n_B_fm)
    x_scale = [100.0, 100.0, 100.0, 100.0, 100.0, n_B,
               100.0, 100.0, 3000.0, 1000.0]

    seeds = []
    if x0 is not None:
        seeds.append(list(x0))
        seeds.append(_restored_branch(x0, par))
    if cold_start:
        seeds.extend(default_guess("beta_eq_neutrinoless", n_B_fm, par))

    solved, tried, best_error = None, 0, float("inf")
    already = []
    for seed in seeds:
        seed = [min(max(v, l), h) for v, l, h in zip(seed, lo, hi)]
        if any(all(abs(a - b) <= 1e-9 * max(1.0, abs(b))
                   for a, b in zip(seed, other)) for other in already):
            continue                      # duplicate of a start already tried
        already.append(seed)
        tried += 1
        sol = least_squares(lambda x: _scaled_residual(x, par, n_B), seed,
                            bounds=(lo, hi), x_scale=x_scale,
                            xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=1500)
        error = scaled_residual_max(residual(sol.x, par, n_B),
                                    residual_scales(par, n_B))
        best_error = min(best_error, error)
        if error <= RESIDUAL_TOL:
            solved = sol.x
            break

    if solved is None:
        raise RuntimeError(
            f"ENJL beta-equilibrium solve did not converge at "
            f"n_B={n_B_fm:.4f} fm^-3 after {tried} starting points; best "
            f"scaled residual {best_error:.3e} against a "
            f"{RESIDUAL_TOL:.0e} bound")

    n, _ = state_at(solved, par, n_B)
    # The reported state comes from one code path whichever entry point was
    # called: the composition just found is handed back to `thermo_from_n`,
    # seeded with the constituent masses so it stays on the branch that was
    # solved.
    point = thermo_from_n(n, par=par, x0=list(solved[:3]))
    return BetaPoint(
        converged=True, error=best_error,
        n_b_fm=n_B_fm,
        densities={k: v / hc3 for k, v in n.items()},
        M_q=point.M_q, M_b=point.M_b,
        eps=point.eps / hc3, P=point.P / hc3,
        mu_b=solved[3], mu_e=solved[4], point=point,
    )
