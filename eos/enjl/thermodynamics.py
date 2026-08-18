"""Quantities of the extended NJL model computed FROM the state.

The state of this model is (densities, T) plus one genuine self-consistency:
the three constituent quark masses M_u, M_d, M_s of the NJL gap equation. This
module takes densities, solves that gap, and returns everything else --- the
baryon masses, the vector fields, the rearrangement self-energies, the
chemical potentials, eps, P, s and the conserved-charge sums. It never knows
which equilibrium mode it is in; imposing beta equilibrium or a charge
fraction is `eos.enjl.solver`.

Reading order, which is the physics: one species as a free gas, the gap
equation, the baryon masses it feeds, the mean fields, the vacuum constant,
then the assembly of one solved point.

The model is Xia, Phys. Rev. D 110, 014022 (2024) [arXiv:2405.02946]; the
equation numbers below are that paper's, and `enjl.tex` states each of them in
full. The NJL parameter set is Rehberg, Klevansky and Huefner, Phys. Rev. C
53, 410 (1996).

Units are natural inside this module: densities in MeV^3, masses and
potentials in MeV, eps and P in MeV^4. The fm-based public boundary is
`eos.enjl.api`; the `_fm` properties of `EoSPoint` are the conversion.
"""
from dataclasses import dataclass
import math

from scipy.optimize import root

from eos.enjl.parameters import Parameters
from eos.enjl.species import (
    BARYONS, DEGENERACY, ISOSPIN, LEPTONS, QUARKS, SPECIES, VALENCE,
    VALENCE_TOTAL, coupling_rescalings, current_masses,
)
from eos.general.fermi_integrals import solve_fermi_jel
from eos.general.physics_constants import hc3

_PI2 = math.pi ** 2

#: Residual bound on the gap equation, Eq. (5), in MeV. On constituent masses
#: of 5-550 MeV this is 1e-14 relative or better. The root finder's own
#: success flag is NOT the gate: at this tolerance it routinely reports "not
#: making good progress" on a converged root, because it is being asked for
#: more precision than the residual can supply.
GAP_TOL = 1.0e-12

#: The cold start for the gap equation: the vacuum constituent masses. It is a
#: poor guess once the light condensates have collapsed, which is why a
#: fixed-composition request in the chirally restored region has to be seeded
#: from a neighbouring point.
VACUUM_GUESS = (367.6, 367.6, 549.5)


# --------------------------------------------------------------------------
# One species as a free Fermi gas, and the cut-off that regularizes the sea
# --------------------------------------------------------------------------

def kF_from_n(n, g):
    """Fermi momentum [MeV] from number density n [MeV^3], Eq. (11) inverted.

        n = g kF^3 / (6 pi^2)
    """
    return (6.0 * _PI2 * n / g) ** (1.0 / 3.0)


def n_from_kF(kF, g):
    """Number density n [MeV^3] from Fermi momentum kF [MeV], Eq. (11).

        n = g kF^3 / (6 pi^2)

    The algebraic inverse of `kF_from_n`, not a Fermi integral: the shared
    integrals of `eos.general` are written in the chemical potential, and a
    solver that carries densities needs the map in both directions. The
    integrals proper -- n^s, eps, P -- come from there, through
    `kinetic_thermo`.
    """
    return g * kF ** 3 / (6.0 * _PI2)


def vacuum_scalar_density(m, g, Lambda):
    """The Dirac-sea term subtracted from the scalar density, Eq. (12).

        n^s_vac = (g m^3 / 4 pi^2) [ y sqrt(y^2+1) - arcsinh y ],  y = Lambda/m

    Positive, so the scalar density it is subtracted from is NEGATIVE in
    vacuum -- that negative value is the chiral condensate, up to the positive
    factor of Eq. (8). Zero for Lambda = 0, i.e. for baryons and leptons,
    which carry no cut-off.
    """
    if Lambda <= 0.0:
        return 0.0
    y = Lambda / m
    return (g * m ** 3 / (4.0 * _PI2)) * (y * math.sqrt(y * y + 1.0)
                                          - math.asinh(y))


def vacuum_energy(m, g, Lambda):
    """The Dirac-sea term subtracted from the energy density, Eq. (13).

        eps_vac = (g m^4/16 pi^2)[ y(2y^2+1) sqrt(y^2+1) - arcsinh y ]
    """
    if Lambda <= 0.0:
        return 0.0
    y = Lambda / m
    return (g * m ** 4 / (16.0 * _PI2)) * (
        y * (2.0 * y * y + 1.0) * math.sqrt(y * y + 1.0) - math.asinh(y))


def kinetic_thermo(nu, m, g, Lambda=0.0):
    """(n, P, eps, s, n_s) of one species at kinetic potential nu [MeV].

    The free-gas part comes from `eos.general.fermi_integrals`, the single
    home for the Fermi integrals of this repository; only the vacuum
    subtraction is written here, because it is model physics rather than an
    integral. The split is exactly

        n^s = n^s_medium(nu, m, g)  -  n^s_vac(m, g, Lambda)
        eps = eps_medium(nu, m, g)  -  eps_vac(m, g, Lambda)

    and the number density carries no vacuum term, the Dirac sea holding no
    net baryon number. THE MEDIUM INTEGRALS ARE NOT CUT OFF: the paper applies
    Lambda to the vacuum subtraction alone, which matters because nu_q exceeds
    Lambda = 602.3 MeV above n_B ~ 3 fm^-3, where a cut on the medium integral
    would truncate the physical Fermi sea.

    Because the vacuum terms depend on (m, g, Lambda) alone they are
    independent of nu and of temperature, which is what makes this function
    the single place a finite-temperature extension would touch.

    `P` is the MEDIUM pressure of the species, nu*n - eps_medium. The total
    pressure of a state is not the sum of these: it is Eq. (19), assembled in
    `assemble` from the Euler relation, which is what carries the interaction
    and rearrangement contributions.

    Natural units in and out: nu, m, Lambda in MeV; n, s, n_s in MeV^3; P and
    eps in MeV^4. Quark masses bottom out at the current mass m_q0 = 5.5 MeV,
    so the massless branch of the shared integrals is never reached.
    """
    n, P, eps, s, n_s = solve_fermi_jel(nu, 0.0, m, g,
                                        include_antiparticles=False)
    return (n * hc3, P * hc3, eps * hc3 - vacuum_energy(m, g, Lambda),
            s * hc3, n_s * hc3 - vacuum_scalar_density(m, g, Lambda))


# --------------------------------------------------------------------------
# The gap equation and the masses it fixes
# --------------------------------------------------------------------------

#: The two flavours the 't Hooft determinant term couples each flavour to.
_OTHER_FLAVOURS = {"u": ("d", "s"), "d": ("u", "s"), "s": ("u", "d")}


def quark_masses_from_gap(nbar_s, par):
    """Constituent quark masses from the gap equation, Eq. (5) with Eq. (8).

        M_u = m_u0 - 4 G_S nbar^s_u + 2 K nbar^s_d nbar^s_s

    and cyclically. The paper writes the 't Hooft determinant term as
    2 K nbar^s_u nbar^s_d nbar^s_s / nbar^s_q; that is the same number wherever
    nbar^s_q is nonzero, but it is a 0/0 at chiral restoration, where the
    condensate of a flavour vanishes. The product over the *other two* flavours
    is the same expression without the removable singularity, so it is the form
    used here.

    `nbar_s` maps "u"/"d"/"s" to the *effective* scalar densities of Eq. (6):
    the medium term, the vacuum (Dirac-sea) term regularized by the cut-off
    Lambda, and the alpha_S-weighted baryon cluster term together. The medium
    term alone is not the argument of this equation.

    Natural units: nbar_s in MeV^3, masses returned in MeV. Pure arithmetic,
    so a dict of numpy arrays evaluates the gap along a whole density sweep.
    """
    m0 = current_masses(par)
    return {q: (m0[q] - 4.0 * par.GS * nbar_s[q]
                + 2.0 * par.K * nbar_s[_OTHER_FLAVOURS[q][0]]
                * nbar_s[_OTHER_FLAVOURS[q][1]])
            for q in QUARKS}


def effective_scalar_densities(kF, M_q, n_s_b, alpha_S, Lambda):
    """nbar^s_q of Eq. (6), the source the gap equation is fed.

        nbar^s_q = n^s_q(M_q, nu_q) + alpha_S sum_{i=p,n,Lambda} N^q_i n^s_i

    A baryon is a cluster of three quarks, so its scalar density contributes
    to the condensate of each of its valence flavours, weighted by the
    structural function alpha_S.

    Capped at zero from above. nbar^s_q is the scalar condensate up to a
    positive factor (g_sigma sigma_q = 4 G_S nbar^s_q, Eq. (8)), so it is
    negative in vacuum and rises towards zero as chiral symmetry is restored;
    a positive value would be a condensate of the wrong sign and would drive
    M_q below its current mass m_q0. The bound is reached because the baryon
    cluster term is positive and grows with baryon density: in symmetric
    nucleonic matter at n_B = 10 fm^-3 the uncapped expression returns
    +0.105 fm^-3 for the u and d flavours, while the s flavour, which the
    nucleons do not feed, is still at -2.07 fm^-3 and is not capped.

    Once nbar^s_q is zero the gap equation returns M_q = m_q0 exactly, and the
    't Hooft term of the remaining flavours loses its coupling to this one.
    """
    out = {}
    for q in QUARKS:
        nu_q = math.sqrt(kF[q] ** 2 + M_q[q] ** 2)
        n_sq = kinetic_thermo(nu_q, M_q[q], DEGENERACY[q], Lambda)[4]
        cluster = sum(VALENCE[b][QUARKS.index(q)] * n_s_b[b] for b in BARYONS)
        out[q] = min(n_sq + alpha_S * cluster, 0.0)
    return out


def baryon_masses(par, M_q, alpha_S, n_bQ):
    """Baryon masses from Eq. (4).

        M_i = sum_q N^q_i [ m_q0 + alpha_S (M_q - m_q0) ] + B n_b^Q

    The first term is the three valence constituent masses, interpolated by
    alpha_S between the current and constituent values. The second is Pauli
    blocking by the deconfined quarks: n_b^Q is the baryon density carried by
    quarks, so as quarks are liberated every baryon mass is pushed up and past
    some density the baryons dissolve. B = 0 switches that mechanism off.
    """
    m0 = current_masses(par)
    out = {}
    for b in BARYONS:
        constituents = sum(
            VALENCE[b][qi] * (m0[q] + alpha_S * (M_q[q] - m0[q]))
            for qi, q in enumerate(QUARKS))
        out[b] = constituents + par.B_nat * n_bQ
    return out


# --------------------------------------------------------------------------
# The mean fields
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MeanFields:
    """The vector fields and rearrangement self-energies at one state.

    `gomega_omega` and `grho_rho` are the products g_omega omega_0 and
    g_rho rho_0 [MeV]: the model determines only g^2/m^2, so the field and its
    coupling never appear apart. `SigmaR_b` and `SigmaR_q` are the
    rearrangement self-energies [MeV] of Eqs. (17)-(18).
    """
    J_omega: float
    J_rho: float
    gomega_omega: float
    grho_rho: float
    SigmaR_b: float
    SigmaR_q: float


def mean_fields(n, n_s_b, M_q, par, n_B):
    """The vector fields and rearrangement terms, Eqs. (9)-(10) and (17)-(18).

        J_omega = sum_i f_i N_i n_i,      N_i = 3 (baryons), 1 (quarks)
        J_rho   = sum_i f_i tau_i n_i
        g_omega omega = Gamma_omega(n_B) J_omega
        g_rho rho     = Gamma_rho(n_B)   J_rho

        Sigma^R_b = 1/2 Gamma_omega'(n_B) J_omega^2
                  + 1/2 Gamma_rho'(n_B)   J_rho^2
                  + alpha_S'(n_B) sum_i [ sum_q N^q_i (M_q - m_q0) ] n^s_i
        Sigma^R_q = 1/3 B sum_i n^s_i + 1/3 Sigma^R_b

    The omega couples to quark number, so a baryon couples three times as
    strongly as a quark; the rho carries no such factor, tau_i already
    distinguishing the two nucleons.

    The asymmetry of Sigma^R_q is worth reading twice. The Pauli-blocking term
    B n_b^Q of Eq. (4) raises BARYON masses, but n_b^Q is a QUARK density, so
    differentiating with respect to a quark density acts back on the quark
    potential; the factor 1/3 is d n_b^Q / d n_q. The second term is 1/3
    Sigma^R_b for the same reason: a quark carries one third of a baryon's
    worth of the n_B that the density-dependent couplings respond to.

    `n_B` is passed rather than summed from `n` because the beta-equilibrium
    residual evaluates the couplings at the density it is TARGETING, which its
    baryon-number row makes equal to the state's own at the solution.
    """
    f = coupling_rescalings(par)
    m0 = current_masses(par)

    J_omega = 0.0
    J_rho = 0.0
    for sp in SPECIES:
        J_omega += f[sp] * n[sp] * VALENCE_TOTAL[sp]
        J_rho += f[sp] * n[sp] * ISOSPIN[sp]

    vector = 0.5 * par.d_Gamma_w(n_B) * J_omega ** 2 \
        + 0.5 * par.d_Gamma_r(n_B) * J_rho ** 2
    d_alpha = par.d_alpha_S(n_B)
    cluster = sum(
        sum(VALENCE[b][qi] * (M_q[q] - m0[q]) for qi, q in enumerate(QUARKS))
        * d_alpha * n_s_b[b]
        for b in BARYONS)
    SigmaR_b = vector + cluster
    SigmaR_q = (1.0 / 3.0) * par.B_nat * sum(n_s_b[b] for b in BARYONS) \
        + (1.0 / 3.0) * SigmaR_b

    return MeanFields(J_omega=J_omega, J_rho=J_rho,
                      gomega_omega=par.Gamma_w(n_B) * J_omega,
                      grho_rho=par.Gamma_r(n_B) * J_rho,
                      SigmaR_b=SigmaR_b, SigmaR_q=SigmaR_q)


# --------------------------------------------------------------------------
# The vacuum, and the constant it fixes
# --------------------------------------------------------------------------

_vacuum_cache = {}


def vacuum_solution(par):
    """(M_u, M_d, M_s) of the gap equation at zero density.

    No baryons, so Eq. (6) has no cluster term and the source is the Dirac-sea
    scalar density alone. The result depends only on (Lambda, m_q0, G_S, K).
    Cached on the frozen parameter object, which is a read-only cache keyed by
    an immutable value and so leaves the module stateless in every sense that
    matters to a sampler.
    """
    if par in _vacuum_cache:
        return _vacuum_cache[par]

    def residual(x):
        M_q = dict(zip(QUARKS, x))
        n_s = {q: kinetic_thermo(M_q[q], M_q[q], DEGENERACY[q], par.Lambda)[4]
               for q in QUARKS}
        gap = quark_masses_from_gap(n_s, par)
        return [x[i] - gap[q] for i, q in enumerate(QUARKS)]

    sol = root(residual, list(VACUUM_GUESS), method="hybr", tol=GAP_TOL)
    if not sol.success:
        raise RuntimeError(f"ENJL vacuum gap solve failed: {sol.message}")
    M_q = dict(zip(QUARKS, sol.x))
    _vacuum_cache[par] = M_q
    return M_q


def vacuum_energy_density(par, _vac_mass=None):
    """The constant E0 of Eq. (13), so that eps = 0 in the vacuum.

    Equation (13) evaluated at zero density: the cut-off quark sea and the
    condensate terms, with no baryons and no leptons. It is a property of the
    vacuum and moves with (Lambda, m_q0, G_S, K) and nothing else -- not with
    f_q, not with B, not with density, not with temperature. At the shipped
    parameters it is -4263.8455 MeV/fm^3, which is the constant the author's
    Maple worksheet hard-codes and the offset all five of the reference tables
    show, to 8e-7 relative.
    """
    M_q = vacuum_solution(par) if _vac_mass is None else _vac_mass
    eps = 0.0
    n_s = {}
    for q in QUARKS:
        _, _, eps_q, _, n_s[q] = kinetic_thermo(
            M_q[q], M_q[q], DEGENERACY[q], par.Lambda)
        eps += eps_q
    eps += 2.0 * par.GS * sum(n_s[q] ** 2 for q in QUARKS)
    eps -= 4.0 * par.K * n_s["u"] * n_s["d"] * n_s["s"]
    return eps


# --------------------------------------------------------------------------
# One solved point
# --------------------------------------------------------------------------

def assemble(n):
    """(n_B, n_C, n_S) from the species densities, in this repo's conventions.

    Leptons are excluded from all three: C is the charge of strongly
    interacting matter only, and a lepton carries no B or S either. Strangeness
    is S = +1 per s quark, so Lambda and s both count +1 -- the OPPOSITE of the
    PDG sign, and the same convention `eos.general.basis` derives from the
    shared particle table.

        n_B = n_p + n_n + n_Lambda + (n_u + n_d + n_s)/3
        n_C = n_p + 2 n_u/3 - (n_d + n_s)/3
        n_S = n_Lambda + n_s
    """
    n_B = n["p"] + n["n"] + n["Lambda"] + (n["u"] + n["d"] + n["s"]) / 3.0
    n_C = n["p"] - (n["d"] + n["s"]) / 3.0 + (2.0 / 3.0) * n["u"]
    n_S = n["Lambda"] + n["s"]
    return n_B, n_C, n_S


@dataclass(frozen=True)
class EoSPoint:
    """One solved state of uniform ENJL matter, in natural units.

    Densities in MeV^3, masses and potentials in MeV, eps and P in MeV^4;
    the `_fm` properties convert to the fm-based public boundary. `enjl.tex`
    Sec. "What a solved point returns" lists every field against the equation
    it comes from.
    """
    n: dict                                   # all species densities [MeV^3]
    M_q: dict                                 # M_u, M_d, M_s [MeV]
    M_b: dict                                 # M_p, M_n, M_Lambda [MeV]
    kF: dict                                  # Fermi momenta [MeV]
    n_s: dict                                 # scalar densities [MeV^3]
    nbar_s: dict                              # effective scalar densities [MeV^3]
    alpha_S: float                            # structural function
    Gw: float                                 # Gamma_omega [MeV^-2]
    Gr: float                                 # Gamma_rho [MeV^-2]
    J_omega: float                            # omega source [MeV^3]
    J_rho: float                              # rho source [MeV^3]
    gomega_omega: float                       # g_omega omega_0 [MeV]
    grho_rho: float                           # g_rho rho_0 [MeV]
    SigmaR_b: float                           # baryon rearrangement [MeV]
    SigmaR_q: float                           # quark rearrangement [MeV]
    mu: dict                                  # chemical potentials [MeV]
    eps: float                                # energy density [MeV^4]
    P: float                                  # pressure [MeV^4]
    s: float                                  # entropy density [MeV^3], zero
    n_b: float                                # total baryon density [MeV^3]
    n_bQ: float                               # quark baryon density [MeV^3]
    n_C: float                                # non-leptonic charge density [MeV^3]
    n_S: float                                # strangeness density [MeV^3]

    @property
    def n_b_fm(self):
        return self.n_b / hc3

    @property
    def eps_fm(self):
        return self.eps / hc3

    @property
    def P_fm(self):
        return self.P / hc3

    @property
    def EperB(self):
        """Energy per baryon minus the nucleon rest mass [MeV], the paper's
        Fig. 2 ordinate."""
        if self.n_b <= 0:
            return 0.0
        return (self.eps / self.n_b) - 938.9


def thermo_from_n(n, par=None, x0=None, _vac_mass=None):
    """The state at given species densities: the block at fixed composition.

    This is not one of the repository's four modes and is not named like one.
    Nothing about the composition is determined here -- all eight densities are
    the caller's -- so the only thing solved is the model's own
    self-consistency, the gap equation Eq. (5) for (M_u, M_d, M_s). Everything
    else follows algebraically: the baryon masses Eq. (4), the vector fields
    Eqs. (9)-(10), the rearrangement terms Eqs. (17)-(18), the chemical
    potentials Eqs. (14)-(16), the energy density Eq. (13) and the pressure
    Eq. (19). Figures 1-3 of the paper are evaluated this way, and they carry
    no branch ambiguity because the composition is imposed.

    Sigma^R appears in every mu_i and hence in P through Eq. (19); it does NOT
    appear in eps. That placement is what makes a density-dependent-coupling
    mean field thermodynamically consistent, and it is checked directly, as
    mu_i = d eps / d n_i, by `eos.enjl.verify`.

    Parameters:
        n:   dict keyed by "p","n","Lambda","u","d","s","e","mu" [MeV^3].
             Missing species are treated as zero density.
        par: Parameters (default: the shipped set).
        x0:  starting guess for (M_u, M_d, M_s); the vacuum masses by default,
             which is a poor guess in the chirally restored region.

    Returns:
        EoSPoint, in natural units.

    Raises:
        RuntimeError if the gap solve does not reach GAP_TOL. Non-convergence
        is a return value at the public boundary, `eos.enjl.api.eos_point`,
        which catches this.
    """
    if par is None:
        par = Parameters.default()
    f = coupling_rescalings(par)
    m_l = {"e": par.m_e, "mu": par.m_mu}

    n = {k: float(v) for k, v in n.items()}
    for sp in SPECIES:
        n.setdefault(sp, 0.0)

    n_b, n_C, n_S = assemble(n)
    n_bQ = (n["u"] + n["d"] + n["s"]) / 3.0
    alpha_S = par.alpha_S(n_b)
    kF = {sp: kF_from_n(n[sp], DEGENERACY[sp]) if n[sp] > 0 else 0.0
          for sp in SPECIES}

    def gap_residual(x):
        M_q = dict(zip(QUARKS, x))
        M_b = baryon_masses(par, M_q, alpha_S, n_bQ)
        n_s_b = _baryon_scalar_densities(kF, M_b)
        nbar = effective_scalar_densities(kF, M_q, n_s_b, alpha_S, par.Lambda)
        gap = quark_masses_from_gap(nbar, par)
        return [x[i] - gap[q] for i, q in enumerate(QUARKS)]

    guess = list(VACUUM_GUESS) if x0 is None else list(x0)
    sol = root(gap_residual, guess, method="hybr", tol=GAP_TOL)
    residual = max(abs(r) for r in gap_residual(sol.x))
    if not sol.success and residual > GAP_TOL:
        raise RuntimeError(
            f"ENJL gap solve failed at n_b={n_b / hc3:.4f} fm^-3: "
            f"residual {residual:.3e} MeV, {sol.message}")

    M_q = dict(zip(QUARKS, sol.x))
    M_b = baryon_masses(par, M_q, alpha_S, n_bQ)
    n_s_b = _baryon_scalar_densities(kF, M_b)
    nbar = effective_scalar_densities(kF, M_q, n_s_b, alpha_S, par.Lambda)
    fields = mean_fields(n, n_s_b, M_q, par, n_b)

    # --- energy density, Eq. (13), with the E0 vacuum subtraction ---
    n_s_q = {}
    eps = 0.0
    for sp in SPECIES:
        mass = M_b.get(sp, M_q.get(sp, m_l.get(sp)))
        Lambda = par.Lambda if sp in QUARKS else 0.0
        nu = math.sqrt(kF[sp] ** 2 + mass ** 2)
        _, _, eps_i, _, n_s_i = kinetic_thermo(nu, mass, DEGENERACY[sp],
                                               Lambda)
        eps += eps_i
        if sp in QUARKS:
            n_s_q[sp] = n_s_i
    eps += 2.0 * par.GS * sum(nbar[q] ** 2 for q in QUARKS)
    eps += 0.5 * par.Gamma_w(n_b) * fields.J_omega ** 2 \
        + 0.5 * par.Gamma_r(n_b) * fields.J_rho ** 2
    eps -= 4.0 * par.K * nbar["u"] * nbar["d"] * nbar["s"]
    eps -= vacuum_energy_density(par, _vac_mass=_vac_mass)

    # --- chemical potentials, Eqs. (14)-(16) ---
    mu = {}
    for b in BARYONS:
        mu[b] = math.sqrt(kF[b] ** 2 + M_b[b] ** 2) \
            + f[b] * (3.0 * fields.gomega_omega
                      + fields.grho_rho * ISOSPIN[b]) + fields.SigmaR_b
    for q in QUARKS:
        mu[q] = math.sqrt(kF[q] ** 2 + M_q[q] ** 2) \
            + f[q] * (fields.gomega_omega
                      + fields.grho_rho * ISOSPIN[q]) + fields.SigmaR_q
    for lepton in LEPTONS:
        mu[lepton] = math.sqrt(kF[lepton] ** 2 + m_l[lepton] ** 2)

    # --- pressure: the Euler relation, Eq. (19), at T = 0 ---
    P = sum(mu[sp] * n[sp] for sp in SPECIES) - eps

    return EoSPoint(
        n=n, M_q=M_q, M_b=M_b, kF=kF,
        n_s={**n_s_b, **n_s_q}, nbar_s=nbar, alpha_S=alpha_S,
        Gw=par.Gamma_w(n_b), Gr=par.Gamma_r(n_b),
        J_omega=fields.J_omega, J_rho=fields.J_rho,
        gomega_omega=fields.gomega_omega, grho_rho=fields.grho_rho,
        SigmaR_b=fields.SigmaR_b, SigmaR_q=fields.SigmaR_q,
        mu=mu, eps=eps, P=P, s=0.0,
        n_b=n_b, n_bQ=n_bQ, n_C=n_C, n_S=n_S,
    )


def _baryon_scalar_densities(kF, M_b):
    """n^s_i of the three baryons, Eq. (12) with no cut-off."""
    out = {}
    for b in BARYONS:
        nu = math.sqrt(kF[b] ** 2 + M_b[b] ** 2)
        out[b] = kinetic_thermo(nu, M_b[b], DEGENERACY[b], 0.0)[4]
    return out
