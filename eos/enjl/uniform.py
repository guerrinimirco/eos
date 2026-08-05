"""
enjl/uniform.py
====================
Uniform dense-matter solver of the extended NJL model (Xia 2024, PRD 110,
014022, arXiv:2405.02946 — Sec. II).

Given a set of number densities (baryons p,n,Lambda; quarks u,d,s; leptons
e,mu) in natural units, this solves the coupled mean-field system:

  * effective scalar densities   nbar_sq = n_sq + alpha_S sum_i N_i^q n_i^s
  * quark masses                 M_q = m_q0 - 4 G_S nbar_sq
                                   + 2 K nbar_su nbar_sd nbar_ss / nbar_sq
  * baryon masses                M_i = sum_q N_i^q [m_q0 + alpha_S (M_q - m_q0)]
                                   + B n_b^Q                     (Eq. (4))
  * vector fields                J_omega = sum_i f_i n_i sum_q N_i^q,
                                  J_rho = sum_i f_i tau_i n_i
                                  g_omega omega = Gamma_omega J_omega,
                                  g_rho rho = Gamma_rho J_rho
  * rearrangement                Sigma_b^R, Sigma_q^R            (Eqs. (17)-(18))
  * energy density               Eq. (13), pressure P = sum mu_i n_i - E

The three quark masses (M_u, M_d, M_s) are the only independent unknowns; all
baryon masses, scalar densities and fields follow algebraically. Solved with
scipy.optimize.root (hybr). All inputs/outputs in natural units (MeV powers);
fm-based wrappers live in eos.enjl.eos_beta.
"""
from dataclasses import dataclass
import math

from scipy.optimize import root

from eos.enjl.parameters import ENJLParams, get_enjl_default
from eos.enjl.species import BARYONS, QUARKS, LEPTONS
from eos.enjl.thermodynamics import kF_from_n, scalar_density_t0, eps_kin_t0
from eos.general.physics_constants import hc3

#: residual tolerance (dimensionless) for the quark-mass gap solve
GAP_TOL = 1.0e-12

#: valence-quark content of each baryon, N_i^q of Eq. (4)/(6)
_N_BARYON = {"p": (2, 1, 0), "n": (1, 2, 0), "Lambda": (1, 1, 1)}
_DEGEN = {"p": 2.0, "n": 2.0, "Lambda": 2.0, "u": 6.0, "d": 6.0, "s": 6.0,
          "e": 2.0, "mu": 2.0}
_M0 = {"u": None, "d": None, "s": None}  # filled from par
_TAU = {"p": 1.0, "n": -1.0, "Lambda": 0.0, "u": 1.0, "d": -1.0, "s": 0.0,
        "e": 0.0, "mu": 0.0}
_NSUM = {"p": 3, "n": 3, "Lambda": 3, "u": 1, "d": 1, "s": 1, "e": 0, "mu": 0}
_F = {"p": 1.0, "n": 1.0, "Lambda": None, "u": None, "d": None, "s": None,
      "e": 0.0, "mu": 0.0}


@dataclass(frozen=True)
class ENJLEoSPoint:
    """One solved uniform-matter state (natural units internally).

    Densities in MeV^3, masses/fields in MeV, eps/P in MeV^4. Convert with
    hc3/hc3 before exposing fm-based quantities.
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
    n_b: float                                # total baryon density [MeV^3]
    n_bQ: float                               # quark baryon density [MeV^3]
    n_C: float                                # non-leptonic charge density [MeV^3]

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
        """Energy per baryon minus nucleon rest mass [MeV] (Paper Fig. 2)."""
        if self.n_b <= 0:
            return 0.0
        return (self.eps / self.n_b) - 938.9


#: the two flavors the 't Hooft determinant term couples to, per flavor
_OTHER_FLAVORS = {"u": ("d", "s"), "d": ("u", "s"), "s": ("u", "d")}


def quark_masses_from_gap(nbar_s, par):
    """Quark masses from the gap equation, Eq. (5) with Eq. (8) substituted.

        M_u = m_u0 - 4 G_S nbar^s_u + 2 K nbar^s_d nbar^s_s

    and cyclically. The paper writes the 't Hooft determinant term as
    2 K nbar^s_u nbar^s_d nbar^s_s / nbar^s_q; that is the same number wherever
    nbar^s_q is nonzero, but it is a 0/0 at chiral restoration, where the
    condensate of a flavor vanishes. The product over the *other two* flavors
    is the same expression without the removable singularity, so it is the form
    used here.

    `nbar_s` maps "u"/"d"/"s" to the *effective* scalar densities of Eq. (6):
    the medium term, the vacuum (Dirac-sea) term regularized by the cut-off
    Lambda, and the alpha_S-weighted baryon cluster term together. The medium
    term alone is not the argument of this equation.

    Natural units: nbar_s in MeV^3, masses returned in MeV. Pure arithmetic,
    so a dict of numpy arrays evaluates the gap along a whole density sweep.
    """
    m0 = _m0_of(par)
    return {q: (m0[q] - 4.0 * par.GS * nbar_s[q]
                + 2.0 * par.K * nbar_s[_OTHER_FLAVORS[q][0]]
                * nbar_s[_OTHER_FLAVORS[q][1]])
            for q in QUARKS}


def _m0_of(par):
    return {"u": par.m_u0, "d": par.m_d0, "s": par.m_s0}


def _f_of(par):
    return {"p": 1.0, "n": 1.0, "Lambda": par.f_Lambda, "u": par.f_q,
            "d": par.f_q, "s": par.f_q, "e": 0.0, "mu": 0.0}


def solve_point(n, par=None, x0=None, _vac_mass=None):
    """Solve the uniform ENJL mean field for densities n [MeV^3].

    Parameters:
        n:   dict with keys in {"p","n","Lambda","u","d","s","e","mu"}.
             Missing species are treated as zero density.
        par: ENJLParams (default: paper Table I + RKH).
        x0:  initial guess for (M_u, M_d, M_s); default vacuum masses.

    Returns:
        ENJLEoSPoint.
    """
    if par is None:
        par = get_enjl_default()
    m0 = _m0_of(par)
    f = _f_of(par)
    n = {k: float(v) for k, v in n.items()}
    for k in _DEGEN:
        n.setdefault(k, 0.0)

    # --- total baryon density & quark baryon density (Eq. (7)) ---
    n_b = n["p"] + n["n"] + n["Lambda"] + (n["u"] + n["d"] + n["s"]) / 3.0
    n_bQ = (n["u"] + n["d"] + n["s"]) / 3.0
    n_C = n["p"] - (n["d"] + n["s"]) / 3.0 + (2.0 / 3.0) * n["u"]

    # --- density-dependent couplings at this n_b ---
    alpha_S = par.alpha_S(n_b)
    Gw = par.Gamma_w(n_b)
    Gr = par.Gamma_r(n_b)

    # Fermi momenta for all species
    kF = {k: kF_from_n(n[k], _DEGEN[k]) if n[k] > 0 else 0.0 for k in n}

    # --- self-consistent quark masses ---
    def residuals(x):
        M_q = {"u": x[0], "d": x[1], "s": x[2]}
        # baryon masses need n_bQ (known) and M_q; then baryon scalar densities
        M_b = _baryon_masses(par, M_q, alpha_S, n_bQ)
        n_s_b = {b: scalar_density_t0(kF[b], M_b[b], 2.0, 0.0) for b in BARYONS}
        nbar = _effective_scalar_densities(kF, M_q, n_s_b, alpha_S, par.Lambda)
        gap = quark_masses_from_gap(nbar, par)
        return [x[QUARKS.index(q)] - gap[q] for q in QUARKS]

    if x0 is None:
        x0 = [367.6, 367.6, 549.5]
    sol = root(residuals, x0, method="hybr", tol=GAP_TOL)
    # hybr reports "not making good progress" (status 5) whenever it is asked
    # for more precision than the residual can supply, which at GAP_TOL happens
    # routinely on a converged root. The residual, not the status flag, decides.
    residual = max(abs(r) for r in residuals(sol.x))
    if not sol.success and residual > GAP_TOL:
        raise RuntimeError(
            f"ENJL gap solve failed at n_b={n_b / hc3:.4f} fm^-3: "
            f"residual {residual:.3e} MeV, {sol.message}")

    M_u, M_d, M_s = sol.x
    M_q = {"u": M_u, "d": M_d, "s": M_s}
    M_b = _baryon_masses(par, M_q, alpha_S, n_bQ)
    n_s_b = {b: scalar_density_t0(kF[b], M_b[b], 2.0, 0.0) for b in BARYONS}
    nbar = _effective_scalar_densities(kF, M_q, n_s_b, alpha_S, par.Lambda)
    n_s_q = {q: scalar_density_t0(kF[q], M_q[q], 6.0, par.Lambda) for q in QUARKS}

    # --- vector fields (Eqs. (9)-(10)) ---
    J_omega = sum(f[k] * n[k] * _NSUM[k] for k in n)
    J_rho = sum(f[k] * n[k] * _TAU[k] for k in n)
    gomega_omega = Gw * J_omega
    grho_rho = Gr * J_rho

    # --- rearrangement terms (Eqs. (17)-(18)) ---
    dGw = par.d_Gamma_w(n_b)
    dGr = par.d_Gamma_r(n_b)
    d_alpha = par.d_alpha_S(n_b)
    SigmaR_wr = 0.5 * dGw * J_omega ** 2 + 0.5 * dGr * J_rho ** 2
    SigmaR_alpha = sum(
        (sum(_N_BARYON[b][qi] * (M_q[q] - m0[q])
             for qi, q in enumerate(QUARKS)) * d_alpha) * n_s_b[b]
        for b in BARYONS)
    SigmaR_b = SigmaR_wr + SigmaR_alpha
    SigmaR_q = (1.0 / 3.0) * par.B_nat * sum(n_s_b.values()) \
        + (1.0 / 3.0) * SigmaR_b

    # --- chemical potentials (Eqs. (14)-(16)) ---
    mu = {}
    for b in BARYONS:
        mu[b] = math.sqrt(kF[b] ** 2 + M_b[b] ** 2) \
            + f[b] * (3.0 * gomega_omega + grho_rho * _TAU[b]) + SigmaR_b
    for q in QUARKS:
        mu[q] = math.sqrt(kF[q] ** 2 + M_q[q] ** 2) \
            + f[q] * (gomega_omega + grho_rho * _TAU[q]) + SigmaR_q
    m_l = {"e": par.m_e, "mu": par.m_mu}
    for l in LEPTONS:
        mu[l] = math.sqrt(kF[l] ** 2 + m_l[l] ** 2)

    # --- energy density (Eq. (13)) with E0 vacuum subtraction ---
    eps = 0.0
    for k in n:
        mass = M_b.get(k, M_q.get(k, m_l.get(k, 0.0)))
        eps += eps_kin_t0(kF[k], mass, _DEGEN[k],
                          par.Lambda if k in QUARKS else 0.0)
    eps += 2.0 * par.GS * sum(nbar[q] ** 2 for q in QUARKS)
    eps += 0.5 * Gw * J_omega ** 2 + 0.5 * Gr * J_rho ** 2
    eps -= 4.0 * par.K * nbar["u"] * nbar["d"] * nbar["s"]
    eps -= vacuum_energy_density(par, _vac_mass=_vac_mass)

    # --- pressure (Eq. (19)) ---
    P = sum(mu[k] * n[k] for k in n) - eps

    return ENJLEoSPoint(
        n=n, M_q=M_q, M_b=M_b, kF=kF,
        n_s={**n_s_b, **n_s_q}, nbar_s=nbar, alpha_S=alpha_S,
        Gw=Gw, Gr=Gr, J_omega=J_omega, J_rho=J_rho,
        gomega_omega=gomega_omega, grho_rho=grho_rho,
        SigmaR_b=SigmaR_b, SigmaR_q=SigmaR_q,
        mu=mu, eps=eps, P=P, n_b=n_b, n_bQ=n_bQ, n_C=n_C,
    )


def _baryon_masses(par, M_q, alpha_S, n_bQ):
    """M_i from Eq. (4): sum_q N_i^q [m_q0 + alpha_S(M_q-m_q0)] + B n_b^Q."""
    m0 = _m0_of(par)
    out = {}
    for b in BARYONS:
        total = sum(
            _N_BARYON[b][qi] * (m0[q] + alpha_S * (M_q[q] - m0[q]))
            for qi, q in enumerate(QUARKS))
        out[b] = total + par.B_nat * n_bQ
    return out


def _effective_scalar_densities(kF, M_q, n_s_b, alpha_S, Lambda):
    """nbar_sq = n_sq(M_q,nu_q) + alpha_S sum_{i=n,p,L} N_i^q n_i^s (Eq. 6).

    Capped at zero from above. nbar_sq is the scalar condensate up to a
    positive factor (g_sigma sigma_q = 4 G_S nbar_sq, Eq. (8)), so it is
    negative in vacuum and rises towards zero as chiral symmetry is restored;
    a positive value would be a condensate of the wrong sign and would drive
    M_q below its current mass m_q0. The bound is reached in this model well
    before the quark medium term alone would reach it, because the baryon
    cluster term alpha_S sum_i N^q_i n^s_i is positive and grows with the
    baryon density: at n_b = 10 fm^-3 the uncapped expression returns
    +2.5 fm^-3 for the u flavor while the condensate has long since vanished.

    Once nbar_sq is zero the gap equation returns M_q = m_q0 exactly, and the
    't Hooft term of the remaining flavors loses its coupling to this one.
    """
    out = {}
    for q in QUARKS:
        n_sq = scalar_density_t0(kF[q], M_q[q], 6.0, Lambda)
        bar = sum(_N_BARYON[b][QUARKS.index(q)] * n_s_b[b] for b in BARYONS)
        out[q] = min(n_sq + alpha_S * bar, 0.0)
    return out


# --------------------------------------------------------------------- vacuum
_vacuum_cache = {}


def vacuum_solution(par):
    """(M_u, M_d, M_s) of the vacuum gap equation (zero densities)."""
    key = par
    if key in _vacuum_cache:
        return _vacuum_cache[key]
    n0 = {k: 0.0 for k in _DEGEN}
    sol = root(lambda x: _vacuum_residual(x, par), [367.6, 367.6, 549.5],
               method="hybr", tol=GAP_TOL)
    if not sol.success:
        raise RuntimeError(f"ENJL vacuum gap solve failed: {sol.message}")
    M_q = {"u": sol.x[0], "d": sol.x[1], "s": sol.x[2]}
    _vacuum_cache[key] = M_q
    return M_q


def _vacuum_residual(x, par):
    M_u, M_d, M_s = x
    n_s = {
        "u": scalar_density_t0(0.0, M_u, 6.0, par.Lambda),
        "d": scalar_density_t0(0.0, M_d, 6.0, par.Lambda),
        "s": scalar_density_t0(0.0, M_s, 6.0, par.Lambda),
    }
    gap = quark_masses_from_gap(n_s, par)
    return [x[QUARKS.index(q)] - gap[q] for q in QUARKS]


def vacuum_energy_density(par, _vac_mass=None):
    """Vacuum energy density E0 (Eq. (13)) so that E = 0 in the vacuum.

    Includes only the quark vacuum kinetic parts and the scalar condensate
    terms; baryons and leptons are absent in the vacuum.
    """
    M_q = vacuum_solution(par) if _vac_mass is None else _vac_mass
    eps = 0.0
    for q in QUARKS:
        eps += eps_kin_t0(0.0, M_q[q], 6.0, par.Lambda)
    n_s = {q: scalar_density_t0(0.0, M_q[q], 6.0, par.Lambda) for q in QUARKS}
    eps += 2.0 * par.GS * sum(n_s[q] ** 2 for q in QUARKS)
    eps -= 4.0 * par.K * n_s["u"] * n_s["d"] * n_s["s"]
    return eps
