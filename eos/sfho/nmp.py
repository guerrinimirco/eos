"""
nmp.py
======
The maps between SFHo's couplings and the physical quantities they are
fitted to, in both directions.

    compute_nmp(par)                couplings -> nuclear-matter parameters
    compute_hyperon_potentials(par)  couplings -> U_Lambda, U_Sigma, U_Xi
    from_potential_depths(U_Lambda_N, ...)   the inverse of that

This module sits ABOVE `solver.py` in the import order (CLAUDE.md section
5), and it has to: every quantity here is defined by a property of the
SOLVED state -- the saturation density is where the pressure vanishes, the
effective mass is read off the converged fields -- so computing any of them
means solving symmetric matter. That is also why a constructor that inverts
one of these maps is a free function here rather than a classmethod on the
parameter dataclass, which is at the bottom of the same order.

Definitions follow the CompOSE manual (Typel et al., arXiv:2203.03209 sec.
6) and Steiner, Prakash, Lattimer & Ellis, Phys. Rept. 411 (2005).

Every nuclear-matter parameter here is exact: four need no derivative, and the
four that do (`snm_derivatives`) take theirs analytically. They used to be
finite differences of quantities that are themselves the output of a nonlinear
solve, so their accuracy was bounded by the solver rather than by the step,
and the published values moved when they were replaced. On python.org 3.14.2 /
numpy 2.3.5 / scipy 1.17.0, old (stencil at the shipped h = 1e-3, symmetric
matter solved at T = 0.01 MeV) -> new (analytic, T = 0 exactly):

    n_sat        0.1582409773  ->   0.1582415032    (rel 3.3e-06)
    E_sat      -16.1723618256  -> -16.1724036674    (rel 2.6e-06)
    m*/m         0.7615642360  ->   0.7615635772    (rel 8.7e-07)
    E_sym       31.5456784436  ->  31.5457311218    (rel 1.7e-06)
    K_sat      245.2210817033  -> 245.2196926301    (rel 5.7e-06)
    Q_sat     -467.4237080394  -> -467.5470984057   (rel 2.6e-04)
    L_sym       47.0775696160  ->  47.0765975489    (rel 2.1e-05)
    K_sym     -205.3787719172  -> -205.3781863893   (rel 2.9e-06)

The first four moved only because the T -> 0 limit became T = 0 (see
`T_COLD`). Against the h-plateau MEAN of the old stencil -- h in [2e-4, 2e-3],
which is the fairer comparison, since the shipped h was one point on it --
three of the four derivative values land inside the plateau's own spread and
Q_sat lands 1.3 spreads outside it:

                plateau mean    spread     analytic      analytic - mean
    K_sat        245.221306    3.26e-03   245.219693       -1.6e-03
    Q_sat       -467.417901    9.80e-02  -467.547098       -1.3e-01
    L_sym         47.077861    3.58e-03    47.076598       -1.3e-03
    K_sym       -205.378953    2.23e-03  -205.378186       +7.7e-04

The reproducibility argument that motivated the change is measured the same
way: the OLD Q_sat at the shipped h differs by 9.7e-05 MeV between that stack
and anaconda 3.9.7 / numpy 1.26.4 / scipy 1.13.1, and their h-plateau means by
1.4e-02 MeV, while the four analytic values agree across the same two stacks
to 5.9e-14 (K_sat), 2.2e-13 (Q_sat), 7.0e-15 (L_sym) and 6.8e-16 (K_sym).

Units:
- Energies/masses/potentials: MeV
- Densities: fm^-3
"""
import copy
import numpy as np
from dataclasses import dataclass, field as dataclass_field, replace
from typing import Optional, Tuple, Dict
from scipy.optimize import brentq, root

from eos.general.physics_constants import hc, hc3
from eos.sfho.species import SpeciesFlags
from eos.sfho.parameters import (
    Parameters, MULTIPLET, vector_ratios, _get_base_sfho
)


# =============================================================================
# CONSTANTS
# =============================================================================
N_SAT = 0.158  # fm^-3, saturation density


# =============================================================================
# COMPUTE SATURATION FIELDS
# =============================================================================
def compute_saturation_fields(params: Optional[Parameters] = None, 
                               n_B: float = N_SAT, 
                               Y_C: float = 0.5,
                               T: float = 0.01) -> Tuple[float, float, float, float]:
    """
    Compute meson fields (σ, ω, ρ, φ) at given density in nuclear matter.
    
    Args:
        params: SFHo parameters (defaults to nucleonic SFHo)
        n_B: Baryon density (fm⁻³)
        Y_C: Charge fraction (0.5 = symmetric nuclear matter)
        T: Temperature (MeV), use small T for T→0 limit
        
    Returns:
        (sigma, omega, rho, phi) fields in MeV
    """
    from eos.sfho.solver import solve_fixed_yc
    from eos.sfho.species import SpeciesFlags
    
    if params is None:
        params = Parameters.default()
    
    result = solve_fixed_yc(params, n_B, Y_C, SpeciesFlags(photons=False),
                           T=T)
    
    if not result.converged:
        raise RuntimeError(f"Failed to converge at n_B={n_B}, Y_C={Y_C}, T={T}")
    
    fields = result.matter.fields
    return (fields["sigma"], fields["omega"], fields["rho"], fields["phi"])


# =============================================================================
# COMPUTE HYPERON POTENTIAL DEPTHS
# =============================================================================
def compute_hyperon_potentials(params: Parameters, 
                                sigma: float = None, 
                                omega: float = None) -> Dict[str, float]:
    """
    Compute hyperon potential depths U_H^(N) at saturation in SNM.
    
    U_H = -g_σH × σ + g_ωH × ω
    
    Args:
        params: SFHo parameters with hyperon couplings
        sigma: σ field in MeV (if None, computed at n_sat)
        omega: ω field in MeV (if None, computed at n_sat)
        
    Returns:
        Dictionary with U_Λ, U_Σ, U_Ξ in MeV
    """
    if sigma is None or omega is None:
        # On `params`, not on the default set: the depth is defined at the
        # saturation point of the parametrization being read, and an inverted
        # or rescaled base does not saturate where nucleonic SFHo does.
        sigma, omega, _, _ = compute_saturation_fields(params)
    
    potentials = {}
    
    for hyperon, label in [('lambda', 'U_Lambda'), 
                           ('sigma+', 'U_Sigma'), 
                           ('xi0', 'U_Xi')]:
        if hyperon in params.couplings_map:
            g_sigma_H = params.get_coupling(hyperon, 'sigma')
            g_omega_H = params.get_coupling(hyperon, 'omega')
            U_H = -g_sigma_H * sigma + g_omega_H * omega
            potentials[label] = U_H
        else:
            potentials[label] = None
            
    return potentials


# =============================================================================
# FORWARD:  couplings -> nuclear-matter parameters
# =============================================================================
#: Symmetric nuclear matter, hadrons only: no electrons, no photons. The
#: nuclear-matter parameters are properties of the strongly-interacting
#: sector, so a lepton or radiation term in eps would corrupt every one of
#: them.
SNM_FLAGS = SpeciesFlags(photons=False)

#: The nuclear-matter parameters are T = 0 quantities and are computed at
#: T = 0 exactly.
#:
#: This used to be 0.01 MeV, on the argument that a strictly cold solve puts
#: a threshold kink where the finite differences straddle. Two things retire
#: it. The differences are gone -- every derivative below is analytic. And the
#: kink was never in this path anyway: symmetric matter here is nucleons only
#: (`SNM_FLAGS`), and a nucleon has no threshold to cross. What the 0.01 MeV
#: did buy was JEL's approximation error, because SFHo evaluates the Fermi
#: integrals in closed form on its T = 0 branch and through JEL on the finite-T
#: one: at n_B = 0.158 the sigma field came back 1.5e-07 relative away from the
#: exact gap-equation root and eps 4.5e-08 away. That is the branch and not the
#: temperature -- T = 1e-4 and T = 0.01 MeV give the same displaced answer --
#: and it is two orders larger than the scatter the analytic derivatives
#: themselves carry across interpreters.
T_COLD = 0.0


def _snm(par, n_B, Y_C=0.5):
    """The solved symmetric-matter point at n_B, or a raised error."""
    from eos.sfho.solver import solve_fixed_yc

    point = solve_fixed_yc(par, n_B, Y_C, SNM_FLAGS, T=T_COLD)
    if not point.converged:
        raise RuntimeError(
            f"symmetric matter did not converge at n_B={n_B:g}, Y_C={Y_C:g} "
            f"(residual {point.error:.3e})")
    return point


def energy_per_baryon(par, n_B, Y_C=0.5):
    """E/A [MeV] of nuclear matter at n_B [fm^-3], rest mass subtracted."""
    return _snm(par, n_B, Y_C).eps / n_B - 0.5 * (par.m_n + par.m_p)


def pressure(par, n_B, Y_C=0.5):
    """P [MeV/fm^3] of nuclear matter at n_B [fm^-3]. Vanishes at n_sat."""
    return _snm(par, n_B, Y_C).P


def esym(par, n_B):
    """
    Symmetry energy E_sym(n_B) [MeV], mean-field closed form.

    Steiner, Prakash, Lattimer & Ellis, Phys. Rept. 411 (2005), Eq. (20):

        E_sym = k_F^2 / (6 E_F*) + n_B / [ 8 ( m_rho^2/g_rho^2 + 2 f ) ]

    with A = g_rho^2 f, so the second term is n_B g_rho^2 / [8 (m_rho^2 + 2A)].
    A = A(sigma, omega) is SFHo's isoscalar-isovector cross coupling, which is
    what makes L_sym adjustable at fixed E_sym in this family of models.

    This is the rho-field response, not a rearrangement of eps, so comparing
    it with the delta^2 curvature of E/A is a genuine second opinion on the
    isovector sector rather than the same computation written twice --
    `verify/run_full_check.py` runs exactly that comparison.
    """
    point = _snm(par, n_B)
    k_F = hc * (3.0 * np.pi**2 * n_B / 2.0) ** (1.0 / 3.0)
    E_F = np.sqrt(k_F**2 + point.matter.m_eff_i["n"]**2)
    A = par.compute_A(point.matter.fields["sigma"], point.matter.fields["omega"])
    kinetic = k_F**2 / (6.0 * E_F)
    potential = n_B * hc3 * par.g_rho_N**2 / (8.0 * (par.m_rho**2 + 2.0 * A))
    return kinetic + potential


# =============================================================================
# THE DENSITY DERIVATIVES OF SATURATED MATTER, IN CLOSED FORM
# =============================================================================
# K_sat, Q_sat, L_sym and K_sym used to be finite differences of quantities
# that are themselves the output of a nonlinear solve. They are written out
# here instead. Everything in this section is in natural units (n in MeV^3,
# kF, fields and masses in MeV) and ' means d/dn.
#
# SFHo's couplings are CONSTANTS, so unlike a density-dependent RMF there is
# no rearrangement self-energy and no coupling to differentiate; what is left
# is the two self-interacting fields. Symmetric matter at T = 0 with Y_C = 0.5
# carries n_p = n_n = n/2 exactly, so rho and phi vanish identically and the
# two nucleons are two g = 2 gases sharing one Fermi momentum but NOT one
# Dirac mass: m*_p = m_p - g_sigma sigma and m*_n = m_n - g_sigma sigma differ
# by m_n - m_p = 1.293 MeV. Abbreviating the two field polynomials
#
#     Phi(sigma) = m_sigma^2 sigma + g2 sigma^2 + g3 sigma^3
#     Psi(omega) = m_omega^2 omega + c3 omega^3
#
# the field equations at rho = 0 are
#
#     Phi(sigma) = g_sigma n_s(sigma, n),   n_s = n_s^p + n_s^n            (gap)
#     Psi(omega) = g_omega n                                            (omega)
#
# and (omega) contains no n_s at all, so omega is a function of n alone -- an
# exact one, independent of the state of the scalar sector:
#
#     omega'  = g_omega / Psi'(omega)
#     omega'' = -Psi''(omega) omega'^2 / Psi'(omega)
#
# The energy density and the mean chemical potential are
#
#     eps    = eps_kin(m*_p, kF) + eps_kin(m*_n, kF) + V(sigma)
#              + m_omega^2 omega^2/2 + 3 c3 omega^4/4
#     mu_bar = (E_F*_p + E_F*_n)/2 + g_omega omega
#
# with V(sigma) = m_sigma^2 sigma^2/2 + g2 sigma^3/3 + g3 sigma^4/4. mu_bar is
# the mean of mu_p and mu_n, and it is the right potential here because the
# mode holds Y_C = 0.5: deps = mu_p dn_p + mu_n dn_n = mu_bar dn along the
# sweep, so deps/dn = mu_bar even though mu_p != mu_n. Then, exactly as in a
# one-gas model, P = mu_bar n - eps and E/A = eps/n - m_N give
#
#     (E/A)'   = P / n^2,        P' = n mu_bar'
#     K_sat    = 9 n^2 (E/A)''  = 9 n mu_bar'                   } at P = 0,
#     Q_sat    = 27 n^3 (E/A)'''= 27 n (n mu_bar'' - 3 mu_bar') } i.e. at n_sat
#
# so the third derivative of E/A costs only the SECOND derivative of mu_bar,
# and the gap equation needs differentiating twice rather than three times.
# Writing ns_m = dn_s/dm*, ns_k = dn_s/dkF and so on, summed over the two
# gases, and using dm*_i/dn = -g_sigma sigma',
#
#     dn_s/dn = -ns_m g_sigma sigma' + ns_k kF'                          (dns)
#     D = Phi'(sigma) + g_sigma^2 ns_m
#     D sigma'  = g_sigma ns_k kF'
#     D sigma'' = -Phi''(sigma) sigma'^2
#                 + g_sigma ( g_sigma^2 ns_mm sigma'^2
#                             - 2 g_sigma ns_mk sigma' kF'
#                             + ns_kk kF'^2 + ns_k kF'' )
#
# The symmetry energy is already closed-form (`esym` above), so L_sym and
# K_sym follow from the same E_F* derivatives plus the two field derivatives
# reaching A(sigma, omega); f is separable in sigma and omega, so there is no
# mixed second partial.
#
# Z_sat, the fourth derivative, is deliberately NOT reported, for the same
# reason it is not reported in `eos.dd2`: it would need a third derivative of
# the gap, no closure imposes it and nobody quotes it.

#: One nucleon species: 2 spin states. Symmetric matter is two of these, at
#: the same kF and at Dirac masses that differ by m_n - m_p.
_G_NUCLEON = 2.0


def _ns_partials(m, kF):
    """n_s [MeV^3] of one g = 2 nucleon gas and its partials in (m*, kF).

    With E_F = sqrt(kF^2 + m*^2) and L = asinh(kF/m*),

        n_s = (g / 4 pi^2) m* [kF E_F - m*^2 L]

    The kF partials are the integrand at the surface; the m* partials are the
    moments

        dn_s/dm*    =  (g / 2 pi^2) int_0^kF k^4 / E_k^3 dk
        d2n_s/dm*^2 = -(g / 2 pi^2) 3 m* int_0^kF k^4 / E_k^5 dk

    which k = m* sinh t turns into m*^2 int (cosh^2 t - 2 + sech^2 t) dt and
    int tanh^4 t dt, both elementary. Returns
    (n_s, ns_m, ns_k, ns_mm, ns_mk, ns_kk).
    """
    E = np.sqrt(kF ** 2 + m ** 2)
    L = np.arcsinh(kF / m)
    p = _G_NUCLEON / (2.0 * np.pi ** 2)
    return (0.5 * p * m * (kF * E - m ** 2 * L),
            p * (0.5 * kF * E - 1.5 * m ** 2 * L + m ** 2 * kF / E),
            p * kF ** 2 * m / E,
            -3.0 * m * p * (L - kF / E - (kF / E) ** 3 / 3.0),
            p * kF ** 4 / E ** 3,
            p * m * kF * (2.0 * E ** 2 - kF ** 2) / E ** 3)


def snm_derivatives(par, n_B):
    """{K_sat, Q_sat, L_sym, K_sym} of symmetric matter at n_B [fm^-3].

    The nuclear-matter combinations 9 n^2 (E/A)'', 27 n^3 (E/A)''',
    3 n E_sym' and 9 n^2 E_sym'', analytically. K_sat and Q_sat are the
    saturation parameters only where P(n_B) = 0, which is where every caller
    evaluates them; the derivation is in the section header above.

    Solves symmetric matter ONCE, at n_B, and differentiates the closed forms
    around that solved point -- so it is also seven solves cheaper than the
    third-difference stencil it replaced.
    """
    point = _snm(par, n_B)
    sigma = point.matter.fields["sigma"]
    omega = point.matter.fields["omega"]
    m_star = point.matter.m_eff_i           # m*_i = m_i - g_sigma sigma
    g_sigma, g_omega, g_rho = par.g_sigma_N, par.g_omega_N, par.g_rho_N

    n = n_B * hc3
    kF = (3.0 * np.pi ** 2 * n / 2.0) ** (1.0 / 3.0)
    kF1, kF2 = kF / (3.0 * n), -2.0 * kF / (9.0 * n ** 2)

    # --- the two nucleon gases, summed --------------------------------------
    ns = ns_m = ns_k = ns_mm = ns_mk = ns_kk = 0.0
    for name in ("p", "n"):
        a, b, c, d, e, f = _ns_partials(m_star[name], kF)
        ns += a
        ns_m += b
        ns_k += c
        ns_mm += d
        ns_mk += e
        ns_kk += f

    # --- the scalar gap, differentiated implicitly --------------------------
    Phi1 = par.m_sigma ** 2 + 2.0 * par.g2 * sigma + 3.0 * par.g3 * sigma ** 2
    Phi2 = 2.0 * par.g2 + 6.0 * par.g3 * sigma
    D = Phi1 + g_sigma ** 2 * ns_m
    s1 = g_sigma * ns_k * kF1 / D
    s2 = (-Phi2 * s1 ** 2
          + g_sigma * (g_sigma ** 2 * ns_mm * s1 ** 2
                       - 2.0 * g_sigma * ns_mk * s1 * kF1
                       + ns_kk * kF1 ** 2 + ns_k * kF2)) / D

    # --- the vector field, which sees only n --------------------------------
    Psi1 = par.m_omega ** 2 + 3.0 * par.c3 * omega ** 2
    Psi2 = 6.0 * par.c3 * omega
    w1 = g_omega / Psi1
    w2 = -Psi2 * w1 ** 2 / Psi1

    # --- the Fermi energies at the moving masses and momentum ---------------
    E, E1, E2 = {}, {}, {}
    for name in ("p", "n"):
        m = m_star[name]
        E[name] = np.sqrt(kF ** 2 + m ** 2)
        E1[name] = (kF * kF1 - m * g_sigma * s1) / E[name]
        E2[name] = ((kF1 ** 2 + kF * kF2 + (g_sigma * s1) ** 2
                     - m * g_sigma * s2) / E[name] - E1[name] ** 2 / E[name])

    # --- isoscalar: mu_bar', mu_bar'' -> K_sat, Q_sat -----------------------
    mu1 = 0.5 * (E1["p"] + E1["n"]) + g_omega * w1
    mu2 = 0.5 * (E2["p"] + E2["n"]) + g_omega * w2

    # --- isovector: E_sym = kF^2/(6 E_F*_n) + n g_rho^2 / [8 (m_rho^2 + 2A)]
    u = kF ** 2
    u1, u2 = 2.0 * u / (3.0 * n), -2.0 * u / (9.0 * n ** 2)
    En, En1, En2 = E["n"], E1["n"], E2["n"]
    Es1 = u1 / (6.0 * En) - u * En1 / (6.0 * En ** 2)
    Es2 = (u2 / En - 2.0 * u1 * En1 / En ** 2 - u * En2 / En ** 2
           + 2.0 * u * En1 ** 2 / En ** 3) / 6.0

    A = par.compute_A(sigma, omega)
    A_s, A_w = par.compute_dA_dsigma(sigma), par.compute_dA_domega(omega)
    A_ss, A_ww = par.compute_d2A_dsigma2(sigma), par.compute_d2A_domega2(omega)
    A1 = A_s * s1 + A_w * w1
    A2 = A_ss * s1 ** 2 + A_ww * w1 ** 2 + A_s * s2 + A_w * w2
    Q = par.m_rho ** 2 + 2.0 * A
    R = g_rho ** 2 / Q
    R1 = -2.0 * g_rho ** 2 * A1 / Q ** 2
    R2 = -2.0 * g_rho ** 2 * A2 / Q ** 2 + 8.0 * g_rho ** 2 * A1 ** 2 / Q ** 3
    Es1 += (R + n * R1) / 8.0
    Es2 += (2.0 * R1 + n * R2) / 8.0

    return {
        "K_sat": 9.0 * n * mu1,
        "Q_sat": 27.0 * n * (n * mu2 - 3.0 * mu1),
        "L_sym": 3.0 * n * Es1,
        "K_sym": 9.0 * n ** 2 * Es2,
    }


def compute_nmp(par, n_lo=0.12, n_hi=0.20):
    """
    Nuclear-matter parameters at saturation.

    Returns dict with n_sat [fm^-3], E_sat, K_sat, Q_sat, E_sym, L_sym,
    K_sym [MeV], m_eff_ratio, and P_sat [MeV/fm^3] (diagnostic, ~0 by
    construction). The same keys `eos.dd2.compute_nmp` returns, so one caller
    reads either model. Z_sat is not reported -- see the derivative section
    above for why.

    Every entry is exact: n_sat, E_sat, m*/m and E_sym need no derivative,
    and the four that do take theirs analytically (`snm_derivatives`) rather
    than by stencil.

    Q_sat and K_sym are PREDICTIONS of the parametrization, not quantities any
    fit imposes; they are reported for exactly that reason.
    """
    n_sat = brentq(lambda n: pressure(par, n), n_lo, n_hi, xtol=1e-13)
    at_sat = _snm(par, n_sat)

    m_N = 0.5 * (par.m_n + par.m_p)
    return {
        "n_sat": n_sat,
        "E_sat": at_sat.eps / n_sat - m_N,
        "m_eff_ratio": at_sat.matter.m_eff_i["n"] / m_N,
        "E_sym": esym(par, n_sat),
        "P_sat": at_sat.P,
        **snm_derivatives(par, n_sat),
    }


# =============================================================================
# A PARAMETRIZATION FROM TARGET POTENTIAL DEPTHS
# =============================================================================
def from_potential_depths(
    # Hyperon potential depths (MeV)
    U_Lambda_N: float = -30.0,
    U_Sigma_N: float = +30.0,
    U_Xi_N: float = -14.0,
    # The base whose nucleon couplings and SU(6)-breaking factors are used
    base: Parameters = None,
    # Delta couplings (no measured depth in this model, so given as ratios)
    x_Delta_sigma: float = 1.15,
    x_Delta_omega: float = 1.0,
    x_Delta_rho: float = 1.0,
    # Name
    name: str = "Custom"
) -> Parameters:
    """
    Create a parametrization from target hyperon potential depths.

    The scalar coupling R_sigma_H is determined from the target depth:
        U_H = -g_sigma_H * sigma + g_omega_H * omega
        R_sigma_H = (R_omega_H * g_omega_N * omega - U_H) / (g_sigma_N * sigma)

    sigma and omega are SOLVED for at saturation ON `base`, by
    `compute_saturation_fields`, and that is why this function lives in
    `nmp.py` rather than in `parameters.py`: it is an inverse map from a
    physical observable to a coupling, so it needs the solver, and
    `parameters.py` is the bottom of the import layer and cannot reach it
    (CLAUDE.md section 5). A second copy did live there, with the two fields
    written in as constants; the constants were mutually inconsistent -- no
    single density reproduces both -- and the couplings they produced missed
    the requested depths by about 3 MeV, so asking for U_Lambda = -30
    delivered -33.07. Hardcoding them is also wrong in principle for an
    inference run, where the base couplings vary and the saturation fields
    move with them.

    **The vector sector comes from `base`'s nine SU(6)-breaking factors, and
    the scalar inversion runs AFTER them**, which is the whole reason this is
    one call rather than two. R_omega_H enters the equation above, so a
    rescaled vector coupling changes the scalar coupling that reproduces the
    same depth; inverting first and rescaling after would silently move U_H.
    To break SU(6), set the factors on the base and let this function close
    the depths on them:

        base = replace(Parameters.default(),
                       y_omega_Lambda=1.5, y_phi_Lambda=1.5,
                       y_omega_Sigma=1.5, y_phi_Sigma=1.5,
                       y_omega_Xi=1.875, y_phi_Xi=1.875)
        par = from_potential_depths(U_Xi_N=-14.0, base=base)

    reproduces SFHoY's couplings (Fortin, Oertel & Providencia 2018, Table 1)
    to the rounding of the published R_sigma column.

    Args:
        U_Lambda_N: Lambda depth at n_sat in SNM (MeV), ~ -30 MeV
        U_Sigma_N: Sigma depth at n_sat in SNM (MeV), ~ +30 MeV
        U_Xi_N: Xi depth at n_sat in SNM (MeV), ~ +10 to -20 MeV
        base: parametrization supplying the nucleon couplings and the nine
            breaking factors; defaults to nucleonic SFHo (SU(6) throughout)
        x_Delta_sigma: R_sigma_Delta = g_sigma_Delta/g_sigma_N
        x_Delta_omega: R_omega_Delta = g_omega_Delta/g_omega_N
        x_Delta_rho: R_rho_Delta = g_rho_Delta/g_rho_N
        name: Name for the parametrization

    Returns:
        Parameters with computed couplings
    """
    p = copy.deepcopy(base if base is not None else _get_base_sfho())
    p.name = name
    p.couplings_map = {}
    p.U_Lambda, p.U_Sigma, p.U_Xi = U_Lambda_N, U_Sigma_N, U_Xi_N

    # Saturation fields of the base, on which the depths are defined.
    sigma, omega, _, _ = compute_saturation_fields(p)

    # U_H = -R_sigma_H g_sigma_N sigma + R_omega_H g_omega_N omega, one linear
    # equation per multiplet, solved AFTER the vector ratios are known.
    depths = {'Lambda': U_Lambda_N, 'Sigma': U_Sigma_N, 'Xi': U_Xi_N}
    scalar_ratio = {}
    for multiplet, U_H in depths.items():
        R_omega, _, _ = vector_ratios(multiplet, *p.su6_breaking[multiplet])
        scalar_ratio[multiplet] = (
            (R_omega * p.g_omega_N * omega - U_H) / (p.g_sigma_N * sigma))

    for particle, multiplet in MULTIPLET.items():
        p.couplings_map[particle] = {
            'sigma': scalar_ratio[multiplet] * p.g_sigma_N}

    # Delta couplings: all four stored, there being no depth to invert.
    delta_couplings = {
        'sigma': x_Delta_sigma * p.g_sigma_N,
        'omega': x_Delta_omega * p.g_omega_N,
        'phi': 0.0,  # Deltas carry no strangeness
        'rho': x_Delta_rho * p.g_rho_N,
    }
    for d_name in ['delta++', 'delta+', 'delta0', 'delta-']:
        p.couplings_map[d_name] = delta_couplings.copy()

    return p


def _rescale_hyperon_scalars(par, base):
    """Hold g_sigma_H / g_sigma_N across an inversion; the depths move."""
    for particle, entry in base.couplings_map.items():
        par.couplings_map[particle] = {
            'sigma': entry['sigma'] / base.g_sigma_N * par.g_sigma_N,
            **{meson: value / _nucleon_coupling(base, meson)
               * _nucleon_coupling(par, meson)
               for meson, value in entry.items() if meson != 'sigma'},
        }
    return par


def _reinvert_hyperon_depths(par, base):
    """Hold U_Lambda, U_Sigma, U_Xi across an inversion; the ratios move."""
    deltas = [name for name in base.couplings_map if name.startswith('delta')]
    ratios = {}
    for meson, x_name in (('sigma', 'x_Delta_sigma'),
                          ('omega', 'x_Delta_omega'),
                          ('rho', 'x_Delta_rho')):
        g_N = _nucleon_coupling(base, meson)
        ratios[x_name] = (base.couplings_map[deltas[0]][meson] / g_N
                          if deltas else 1.0)
    rebuilt = from_potential_depths(
        U_Lambda_N=base.U_Lambda, U_Sigma_N=base.U_Sigma, U_Xi_N=base.U_Xi,
        base=par, name=par.name, **ratios)
    if not deltas:
        for name in list(rebuilt.couplings_map):
            if name.startswith('delta'):
                del rebuilt.couplings_map[name]
    return rebuilt


def _nucleon_coupling(par, meson):
    """g_MN, the coupling a hyperon ratio is measured against (phi over g_omegaN)."""
    return {'sigma': par.g_sigma_N, 'omega': par.g_omega_N,
            'rho': par.g_rho_N, 'phi': par.g_omega_N}[meson]


# =============================================================================
# INVERSE:  nuclear-matter parameters -> couplings
# =============================================================================
# The inversion is TRIANGULAR, and that is a property of the model rather than
# a solver tactic: in symmetric matter the rho field and A(sigma,omega) rho^2
# drop out of every equation, so the isoscalar sector does not see the
# isovector couplings at all. Solve four isoscalar unknowns first, then two
# isovector ones on top of the converged isoscalar point. (Strictly, m_p != m_n
# leaves a tiny isospin source and rho is not exactly zero; it enters eps as
# A rho^2 and is far below every gate here. The forward map reads the same
# solved points, so a round trip is exact regardless.)
#
# ISOSCALAR: the classical Boguta-Bodmer inversion.
#     unknowns   {g_sigma_N, g_omega_N, b, c}   at fixed m_sigma, m_omega, c3
#     conditions {P(n_sat) = 0, E_sat, m*/m, K_sat}
# Four against four, no structural closure needed. The scalar self-couplings
# are carried in the REDUCED form the published table states them in,
#     g2 = b m_N g_sigma^3,   g3 = c g_sigma^4,
# because b ~ 7e-3 and c ~ -4e-3 sit beside couplings of order 10 while g2 and
# g3 span 3e3 MeV and -12; a root finder given the raw pair is solving a badly
# scaled problem for no reason.
#
# ISOVECTOR: two conditions {E_sym, L_sym} face g_rho_N plus the NINE shape
# coefficients of A = g_rho_N^2 [sum_i a_i sigma^i + sum_j b_j omega^2j], so
# exactly two have to be freed and the rest pinned at their published values.
# The choice is physics, not bookkeeping: it decides how E_sym behaves ABOVE
# saturation, where no nuclear-matter parameter constrains it. The closure
# here frees (g_rho_N, b_1), for three measured reasons.
#
#   CONDITIONING. The 2x2 Jacobian in log-knobs at the SFHo point:
#       (g_rho, a_1)  det = +231   cond = 11.60
#       (g_rho, b_1)  det = -922   cond =  3.40
#       (g_rho, s)    det = -976   cond =  3.53      (s an overall scale on f)
#   a_1 moves L_sym by only 1.8 MeV per e-fold and is the weak lever.
#
#   REACH. Scanning g_rho over [0.5, 2] and the knob over [-2, 6] times
#   published, at E_sym held near 31.5 MeV, the accessible L_sym is
#       a_1: [ 27.1,  69.1]      b_1: [ -6.4, 146.3]      s: [-34.2,  59.2]
#   b_1 is the only one that spans an inference prior in both directions.
#
#   LITERATURE. b_1 IS the Horowitz-Piekarewicz Lambda_v omega^2 rho^2
#   coupling, PRL 86 (2001) 5647 -- keeping only b_1 gives A = g_rho^2 b_1
#   omega^2 -- which is the standard way this family of models tunes L_sym at
#   fixed E_sym. A set inverted this way stays comparable to published ones.
#
# What the closure costs: it reshapes E_sym above saturation more than a_1
# does and less than s does. All three fitted to E_sym = 31.52, L_sym = 70:
#
#     E_sym at   n_sat   2 n_sat   3 n_sat   4 n_sat
#     a_1        31.52     44.01     49.57     55.90
#     b_1        31.52     46.83     52.17     56.76
#     s          31.52     48.15     55.84     60.20
#     published  31.52     41.36     48.52     55.67
#
# so `s` -- which looks least invasive because it preserves the SHAPE of A --
# is in fact the most invasive to the physics, because scaling A changes how
# 2A competes with m_rho^2 in the denominator of E_sym as density rises.

#: Gate on the isoscalar residual. Above the noise of the K_sat second
#: difference (scaled by 1e-2 in the residual below) and far under any
#: difference that would matter to a fit.
ISO_GATE = 1e-6

#: Gate on the isovector residual, in MeV on E_sym and L_sym.
ISOV_GATE = 1e-6

#: The cross coupling has to stay a CORRECTION to the rho mass term rather
#: than a replacement for it: |2A| < m_rho^2 at saturation. That is the
#: assumption the model form is written under, and without it the isovector
#: solve has a second, mathematically valid and physically absurd branch.
#: Because E_sym's potential term is n g_rho^2 / [8 (m_rho^2 + 2A)] and
#: A = g_rho^2 f, sending g_rho to infinity does not send E_sym with it -- the
#: term saturates at n/(16 f) -- so a runaway (g_rho, b_1) can fit any target
#: the physical branch fits. Measured 2A/m_rho^2: published SFHo +0.37, every
#: fit from L_sym = 40 to 140 inside [-0.40, +0.69], and the runaway +108.9.
CROSS_COUPLING_LIMIT = 1.0

#: Perturbed restarts attempted when the first isoscalar solve misses the
#: gate. They run ONLY on a miss, so a target that inverts from the published
#: seed costs exactly what it did without them. What they buy is the
#: difference between "these NMPs have no SFHo-form realisation" and "this
#: seed could not find it".
N_RESTARTS = 16


@dataclass
class InversionStatus:
    """What the inversion achieved, as a return value rather than a raise."""
    ok: bool
    message: str
    isoscalar_residual: float
    isovector_residual: float
    #: Higher derivatives the closure does not impose, computed FORWARD from
    #: the recovered couplings with `compute_nmp`'s own closed forms:
    #: {"Q_sat": MeV, "K_sym": MeV}.
    predictions: dict = dataclass_field(default_factory=dict)


def _trial_par(base, g_sigma, g_omega, b, c, g_rho=None, b1=None):
    """A parameter set with the isoscalar (and optionally isovector) knobs set.

    Everything not named is inherited from `base`: the meson masses, c3, c4,
    and the eight shape coefficients of A the closure does not free. The
    scalar self-couplings arrive in the reduced (b, c) of the published table
    and are converted here, which is the one place that conversion lives.
    """
    par = copy.deepcopy(base)
    par.g_sigma_N = g_sigma
    par.g_omega_N = g_omega
    par.g2 = b * par.m_n * g_sigma ** 3
    par.g3 = c * g_sigma ** 4
    if g_rho is not None:
        par.g_rho_N = g_rho
    if b1 is not None:
        par.b_coeffs = np.array(base.b_coeffs, dtype=float)
        par.b_coeffs[1] = b1
    return par


def _reduced_self_couplings(par):
    """(b, c) of the published table, back out of (g2, g3)."""
    return (par.g2 / (par.m_n * par.g_sigma_N ** 3),
            par.g3 / par.g_sigma_N ** 4)


def _isoscalar_quantities(par, n_sat):
    """{P, E_sat, m_ratio, K_sat} of symmetric matter AT n_sat.

    No P = 0 search: the inversion imposes P(n_sat) = 0 as one of its
    conditions instead, so the target saturation density is where these are
    evaluated rather than something to be found first.

    K_sat comes from `snm_derivatives`, the same closed form `compute_nmp`
    reads. Forward and inverse have to differentiate identically -- while both
    were stencils the truncation bias cancelled on a round trip, and moving
    one alone would stop the inversion reproducing its own inputs.
    """
    at = _snm(par, n_sat)
    m_N = 0.5 * (par.m_n + par.m_p)
    return dict(P=at.P, E_sat=at.eps / n_sat - m_N,
                m_ratio=at.matter.m_eff_i["n"] / m_N,
                K_sat=snm_derivatives(par, n_sat)["K_sat"])


def _restart_loop(residual, seed, first, n_restarts, gate):
    """Keep the best of the first solve and up to n_restarts jittered ones.

    Deterministic by construction: the same target must invert identically on
    every run and in every parallel worker, so the generator is seeded with a
    constant rather than left to entropy.
    """
    best_x = first.x
    best_res = float(np.max(np.abs(residual(best_x))))
    if best_res >= gate and n_restarts:
        rng = np.random.default_rng(0)
        base = np.asarray(seed, dtype=float)
        for _ in range(n_restarts):
            try:
                trial = root(residual, base * rng.uniform(0.75, 1.35, base.size),
                             method="hybr", tol=1e-13)
                res = float(np.max(np.abs(residual(trial.x))))
            except Exception:      # a jittered seed that will not build a
                continue           # trial parametrization is not a finding
            if res < best_res:
                best_x, best_res = trial.x, res
            if best_res < gate:
                break
    return best_x, best_res


def invert_nmp(par_base=None, seed=None, n_restarts=N_RESTARTS,
               hold_hyperons=None, **nmp):
    """Recover SFHo couplings from a target nuclear-matter-parameter set.

    The inverse of `compute_nmp`, closed as documented above: the classical
    Boguta-Bodmer inversion in the isoscalar sector, and (g_rho_N, b_1) in the
    isovector one.

    Args:
        par_base: the parameter set everything the closure does not free is
            inherited from — meson masses, c3, c4, the eight pinned shape
            coefficients of A, and any hyperon or Delta couplings.
            Defaults to published SFHo.
        seed: (g_sigma_N, g_omega_N, b, c) to start the isoscalar solve from,
            in the reduced self-couplings of the published table. Defaults to
            `par_base`'s own values, which is the right seed for any target in
            the SFHo neighbourhood.
        n_restarts: jittered isoscalar restarts tried ONLY if the first solve
            misses ISO_GATE. Set to 0 for single-seed behaviour.
        nmp: the targets, named as `compute_nmp`'s keys —
            n_sat [fm^-3], E_sat, m_eff_ratio, K_sat, E_sym, L_sym [MeV].
            All six are required; Q_sat and K_sym are ignored if passed,
            because this closure predicts rather than imposes them.

    Returns:
        (Parameters, InversionStatus). Non-convergence is a RETURN VALUE
        (CLAUDE.md section 6): a sampler walks into unrepresentable targets
        constantly and must be able to score one and move on, so a solve that
        misses its gate comes back with `ok=False` and `None` for the
        parameters. Only a hard infeasibility raises — a target outside the
        physical window, where there is no question of a fit failing to be
        found because there is nothing to find.

    **`hold_hyperons` says what the strange sector keeps**, and it is an
    argument rather than a default because the answer is physics, not
    bookkeeping. A hyperon's scalar coupling is stored ABSOLUTELY while its
    depth U_H = -g_sigma_H sigma + g_omega_H omega is a statement about the
    nucleon couplings and fields this function has just moved, so the two
    cannot both survive an inversion:

      "ratios"  keep g_sigma_H / g_sigma_N fixed. The hyperon sector is a
                fixed multiple of the nucleon one and the depths move.
      "depths"  keep U_Lambda, U_Sigma, U_Xi fixed, re-inverting the scalar
                couplings on the new nucleon sector through
                `from_potential_depths`. The ratios move.

    Required whenever `par_base` carries hyperons; ignored (and pointless)
    otherwise. The Delta couplings are ratios in both arms -- this model
    inverts no Delta depth.
    """
    required = ("n_sat", "E_sat", "m_eff_ratio", "K_sat", "E_sym", "L_sym")
    missing = [k for k in required if k not in nmp]
    if missing:
        raise ValueError(f"invert_nmp needs {required}; missing {missing}")
    base = par_base or Parameters.default()
    n_sat = float(nmp["n_sat"])

    has_hyperons = any(name in base.couplings_map for name in MULTIPLET)
    if has_hyperons and hold_hyperons not in ("ratios", "depths"):
        raise ValueError(
            f"invert_nmp on a base carrying hyperons must say what the "
            f"strange sector holds: hold_hyperons='ratios' keeps "
            f"g_sigma_H/g_sigma_N and lets U_H move, hold_hyperons='depths' "
            f"keeps U_Lambda/U_Sigma/U_Xi and re-inverts the scalar "
            f"couplings. Got {hold_hyperons!r}.")

    # Hard infeasibility: below about 0.35 the scalar field has eaten the
    # nucleon mass (g_sigma sigma -> m_N, scalar collapse) and above about
    # 0.95 the scalar sector is doing nothing. Neither has an SFHo-form fit,
    # and neither is a solver failure to be reported.
    if not (0.35 < nmp["m_eff_ratio"] < 0.95):
        raise ValueError(
            f"NMP inversion infeasible: m*/m = {nmp['m_eff_ratio']} outside "
            f"the physical (0.35, 0.95) window (scalar collapse / no fit)")

    if seed is None:
        b0, c0 = _reduced_self_couplings(base)
        seed = [base.g_sigma_N, base.g_omega_N, b0, c0]

    targets = np.array([0.0, nmp["E_sat"], nmp["m_eff_ratio"], nmp["K_sat"]])

    def iso_residual(x):
        g_sigma, g_omega, b, c = x
        if g_sigma <= 0.0 or g_omega <= 0.0:
            return [1e3] * 4
        try:
            q = _isoscalar_quantities(_trial_par(base, g_sigma, g_omega, b, c),
                                      n_sat)
        except (ValueError, RuntimeError):
            return [1e3] * 4
        # K_sat is scaled to ~1: it is a few hundred MeV where E_sat is tens
        # and m*/m is order one, and an unscaled row would dominate the norm.
        return [q["P"] - targets[0], q["E_sat"] - targets[1],
                q["m_ratio"] - targets[2], (q["K_sat"] - targets[3]) * 1e-2]

    first = root(iso_residual, seed, method="hybr", tol=1e-13)
    x_iso, iso_res = _restart_loop(iso_residual, seed, first, n_restarts,
                                   ISO_GATE)
    g_sigma, g_omega, b, c = x_iso

    if iso_res >= ISO_GATE:
        # Fitting the isovector sector on top of an unconverged isoscalar one
        # would read the Dirac mass off a meaningless point, so there is
        # nothing to hand back.
        return None, InversionStatus(
            ok=False,
            message=f"isoscalar residual {iso_res:.2e} above the "
                    f"{ISO_GATE:.0e} gate after {n_restarts} restarts "
                    f"(the targets are probably not representable in the "
                    f"SFHo form at this K_sat)",
            isoscalar_residual=iso_res, isovector_residual=float("nan"))

    # --- isovector: (g_rho_N, b_1) against (E_sym, L_sym) -------------------
    # Both unknowns are solved together rather than one analytically: unlike
    # DD2, where E_sym is quadratic in Gamma_rho and inverts in closed form,
    # here A = g_rho_N^2 f carries g_rho_N in the DENOMINATOR of the potential
    # term as well, so the two conditions do not separate.
    iso_only = _trial_par(base, g_sigma, g_omega, b, c)
    at_sat = _snm(iso_only, n_sat)
    k_F = hc * (3.0 * np.pi ** 2 * n_sat / 2.0) ** (1.0 / 3.0)
    kinetic = k_F ** 2 / (6.0 * np.sqrt(k_F ** 2 + at_sat.matter.m_eff_i["n"] ** 2))
    if nmp["E_sym"] <= kinetic:
        raise ValueError(
            f"NMP inversion infeasible: E_sym = {nmp['E_sym']} at or below "
            f"the kinetic symmetry energy {kinetic:.2f} MeV of the converged "
            f"isoscalar point (no real g_rho_N)")

    def isov_residual(x):
        g_rho, b1 = x
        if g_rho <= 0.0:
            return [1e3] * 2
        try:
            p = _trial_par(base, g_sigma, g_omega, b, c, g_rho=g_rho, b1=b1)
            E = esym(p, n_sat)
            L = snm_derivatives(p, n_sat)["L_sym"]
        except (ValueError, RuntimeError):
            return [1e3] * 2
        return [E - nmp["E_sym"], (L - nmp["L_sym"]) * 1e-1]

    isov_seed = [base.g_rho_N, base.b_coeffs[1]]
    first_v = root(isov_residual, isov_seed, method="hybr", tol=1e-13)
    x_isov, isov_res = _restart_loop(isov_residual, isov_seed, first_v,
                                     n_restarts, ISOV_GATE)
    g_rho, b1 = x_isov

    if isov_res >= ISOV_GATE:
        return None, InversionStatus(
            ok=False,
            message=f"isovector residual {isov_res:.2e} above the "
                    f"{ISOV_GATE:.0e} gate: (E_sym, L_sym) = "
                    f"({nmp['E_sym']}, {nmp['L_sym']}) is outside what "
                    f"(g_rho_N, b_1) reaches on this isoscalar sector",
            isoscalar_residual=iso_res, isovector_residual=isov_res)

    par = _trial_par(base, g_sigma, g_omega, b, c, g_rho=g_rho, b1=b1)
    par.name = f"{getattr(base, 'name', 'SFHo')}_from_nmp"
    if has_hyperons and hold_hyperons == "depths":
        par = _reinvert_hyperon_depths(par, base)
    elif has_hyperons:
        par = _rescale_hyperon_scalars(par, base)

    at_final = _snm(par, n_sat)
    fields = at_final.matter.fields
    cross = (2.0 * par.compute_A(fields["sigma"], fields["omega"])
             / par.m_rho ** 2)
    if abs(cross) >= CROSS_COUPLING_LIMIT:
        # A converged root on the runaway branch. It reproduces the targets --
        # forward-checking it agrees -- so this is not a solver failure but a
        # statement that the target has no realisation on the branch the model
        # form assumes. Reported rather than raised (CLAUDE.md section 6).
        return None, InversionStatus(
            ok=False,
            message=f"the fit landed on the runaway cross-coupling branch: "
                    f"2A/m_rho^2 = {cross:+.2f} at saturation, against a limit "
                    f"of {CROSS_COUPLING_LIMIT} (published SFHo sits at +0.37). "
                    f"g_rho_N = {g_rho:.2f}, b_1 = {b1:.2f}. The targets are "
                    f"reproduced, but not by a physical rho sector",
            isoscalar_residual=iso_res, isovector_residual=isov_res)

    # Report what the closure did NOT impose, with the forward map itself.
    # The forward map brackets saturation in [n_lo, n_hi] and raises if the
    # recovered couplings saturate outside it -- which a target n_sat near the
    # edge of that bracket can produce. The couplings are still the answer, so
    # the predictions are dropped rather than the whole inversion: a sampler
    # must not meet an exception at a public boundary (CLAUDE.md section 6).
    try:
        full = compute_nmp(par)
        predictions = {"Q_sat": full["Q_sat"], "K_sym": full["K_sym"]}
        message = "converged"
    except (ValueError, RuntimeError) as exc:
        predictions = {}
        message = (f"converged, but the forward map could not re-locate "
                   f"saturation to predict Q_sat and K_sym ({exc})")

    return par, InversionStatus(
        ok=True, message=message,
        isoscalar_residual=iso_res, isovector_residual=isov_res,
        predictions=predictions)


#: The nine SU(6)-breaking factors, one per (vector meson, multiplet) pair.
#: The same names dd2 carries, because it is the same job: they scale the
#: hyperon VECTOR couplings, so they go on the base BEFORE the depths are
#: inverted on it (see `from_potential_depths`).
SU6_FACTOR_KEYS = ("y_omega_Lambda", "y_omega_Sigma", "y_omega_Xi",
                   "y_rho_Lambda", "y_rho_Sigma", "y_rho_Xi",
                   "y_phi_Lambda", "y_phi_Sigma", "y_phi_Xi")

#: The scalar couplings the NMP closure does NOT fit. `invert_nmp` fits six
#: numbers -- g_sigma_N, g_omega_N, g_rho_N, g2, g3 and b_coeffs[1] -- and
#: EVERYTHING else in the parametrization is inherited from `par_base`. These
#: seven are the continuous ones a sampler is most likely to want on an axis;
#: the shape arrays `a_coeffs` and `b_coeffs[2:]`, the baryon masses and the
#: hyperon sector stay reachable through `par_base` itself, which is the
#: general hook and is why this model needs no `pinned` mechanism of its own
#: beyond a name for it.
#:
#: They are not decoration. Re-inverting the SAME six NMPs on a rescaled base
#: reproduces them to ~1e-11 while moving the star: over +-30% `m_omega` moves
#: M_max by ~0.21 M_sun, and switching `g_phi_N` from its published 0 to 6
#: moves R_1.4 by ~2.2 km. Those are directions a nuclear-matter likelihood
#: cannot see at all.
PINNED_DEFAULT = ("m_sigma", "m_omega", "m_rho", "m_phi", "g_phi_N",
                  "c3", "c4")

#: Hadronic-sector knobs that may be carried INSIDE the NMP sample dict, so
#: that one dict describes the whole parametrization the way an inference
#: sample does. `x_Delta_sigma` appears and `U_Delta` does not: this model
#: inverts no Delta depth, so the quartet's scalar sector is a ratio and there
#: is no depth form to choose (contrast `eos.dd2.nmp`).
SECTOR_KEYS = (("U_Lambda", "U_Sigma", "U_Xi",
                "x_Delta_sigma", "x_Delta_omega", "x_Delta_rho")
               + SU6_FACTOR_KEYS + PINNED_DEFAULT)


def _split_sample(sample, hyperon_potentials=None, pinned=None):
    """Separate a sample dict into (nmp, sector kwargs).

    A sample may carry any of the `SECTOR_KEYS` next to the nuclear-matter
    parameters; those override the corresponding keyword arguments key by key,
    so naming one held coupling in the sample does not discard the others.
    """
    nmp = {k: v for k, v in sample.items() if k not in SECTOR_KEYS}
    pots = dict(hyperon_potentials or {})
    pots.update({k: float(sample[k]) for k in ("U_Lambda", "U_Sigma", "U_Xi")
                 if k in sample})
    return nmp, {
        "hyperon_potentials": pots,
        "su6": {k: float(sample[k]) for k in SU6_FACTOR_KEYS if k in sample},
        "pinned": {**{k: float(v) for k, v in (pinned or {}).items()},
                   **{k: float(sample[k]) for k in PINNED_DEFAULT
                      if k in sample}},
        "delta": {k: float(sample[k]) for k in
                  ("x_Delta_sigma", "x_Delta_omega", "x_Delta_rho")
                  if k in sample},
    }


def build_parametrization(nmp, flags, hyperon_potentials=None, pinned=None,
                          delta_ratios=None, par_base=None,
                          hold_hyperons=None):
    """Nuclear-matter parameters to a `Parameters` with the strange and
    resonant sectors attached, as `flags` requires.

    The same entry point `eos.dd2.nmp` carries, taking the same six
    nuclear-matter parameters, so one sample dict drives either model. It runs
    the stages this model spells out separately -- the held couplings onto the
    base, `invert_nmp` for the nucleons, the SU(6) factors, then
    `from_potential_depths` for the hyperon depths and the Delta ratios -- in
    the one order that is correct: the SU(6) factors scale the VECTOR
    couplings and the hyperon scalar couplings are inverted from the depths
    AFTER them, so imposing them first holds the depths at what was asked and
    re-fits x_sigma. Applying them to the finished parametrization does the
    opposite and moves the depths instead.

    `nmp` may also carry any of the `SECTOR_KEYS` -- the hyperon depths, the
    three Delta ratios, the nine SU(6) factors and the seven held couplings of
    `PINNED_DEFAULT`. Those take precedence over the keyword arguments, key by
    key, so a single dict can put L_sym, U_Xi, y_omega_Xi and m_omega on axes
    together.

    `par_base` is the parametrization everything unnamed is inherited from,
    defaulting to published SFHo. It is the general form of `pinned`: the
    shape arrays and the baryon masses have no sample key and are moved by
    passing a base that carries them. `hold_hyperons` is forwarded to
    `invert_nmp` and is required only when that base already carries a strange
    sector.

    Returns `(par, stage, message)`. `stage` is 'ok', 'inversion_failed' when
    the NMPs have no SFHo-form realisation, or 'sectors_failed' when they do
    but the hyperon closure does not converge on them. `par` is None unless
    `stage` is 'ok'. Non-convergence is a RETURN VALUE (CLAUDE.md section 6).
    """
    nmp, sector = _split_sample(dict(nmp), hyperon_potentials, pinned)
    base = replace(par_base or Parameters.default(), **sector["pinned"])

    par, status = invert_nmp(par_base=base, hold_hyperons=hold_hyperons, **nmp)
    if not status.ok:
        return None, "inversion_failed", status.message
    try:
        if flags.hyperons or flags.deltas:
            par = replace(par, **sector["su6"])
            ratios = dict(delta_ratios or {})
            ratios.update(sector["delta"])
            par = from_potential_depths(base=par, **{
                f"U_{name}_N": sector["hyperon_potentials"][f"U_{name}"]
                for name in ("Lambda", "Sigma", "Xi")
                if f"U_{name}" in sector["hyperon_potentials"]}, **ratios)
        elif sector["su6"]:
            par = replace(par, **sector["su6"])
    except Exception as exc:
        return None, "sectors_failed", f"{type(exc).__name__}: {exc}"
    return par, "ok", ""


def from_nmp(par_base=None, return_status=False, **nmp):
    """Nuclear-matter parameters -> an `Parameters` carrying those couplings.

    The convenience face of `invert_nmp`: same arguments, returns the
    parameters alone unless `return_status`. Raises when the inversion did not
    converge, since a caller asking only for parameters has nowhere to put a
    failure -- use `invert_nmp` directly to score a target instead of
    raising on it.

        par = from_nmp(n_sat=0.16, E_sat=-16.0, m_eff_ratio=0.75,
                       K_sat=240.0, E_sym=32.0, L_sym=60.0)
    """
    par, status = invert_nmp(par_base=par_base, **nmp)
    if not status.ok:
        raise RuntimeError(f"NMP inversion failed: {status.message}")
    return (par, status) if return_status else par


# =============================================================================
# A SUMMARY, FOR READING RATHER THAN FOR SOLVING
# =============================================================================
# Two dicts, because they answer two different questions. A reader checking
# this model against the paper's table needs the digits the paper prints; a
# caller starting an inference "around" the published set needs the numbers
# the published COUPLINGS actually produce, which the printed digits are a
# rounding of. Neither is a gate.

#: The nuclear-matter parameters as PRINTED: Steiner, Hempel & Fischer, ApJ
#: 774 (2013) 17, as tabulated by Fortin, Oertel & Providencia, PASA 35
#: (2018) e044 Table 2 and by the CompOSE HS(SFHo) entry. Q_sat and K_sym are
#: not fitted quantities and carry no published value here; `compute_nmp`
#: reports them as predictions.
PUBLISHED_NMP = {
    "n_sat": 0.1583, "E_sat": -16.19, "m_eff_ratio": 0.76,
    "K_sat": 245.4, "E_sym": 31.57, "L_sym": 47.10,
}

#: The same six at full precision: `compute_nmp(Parameters.default())` on the
#: published couplings, frozen here so that reading them costs no saturation
#: solve. Regenerate with that call.
#:
#: What the paper's rounding costs, measured as the worst relative distance
#: between the couplings `invert_nmp` returns and the published ones, over
#: (g_sigma_N, g_omega_N, g2, g3, g_rho_N):
#:
#:     from PUBLISHED_NMP                            3.7e-02
#:     from PUBLISHED_NMP with m*/m exact, rest rounded  1.5e-03
#:     from PUBLISHED_NMP_EXACT                      0
#:
#: The whole factor of 25 is ONE two-digit entry: the table prints
#: m*/m = 0.76 where the couplings give 0.761564. The inversion is exact at
#: full precision -- SFHo's published set IS a root of this closure -- so
#: what the first row measures is the paper's rounding and not the closure.
PUBLISHED_NMP_EXACT = {
    "n_sat": 0.15824150323199773, "E_sat": -16.172403667396566,
    "m_eff_ratio": 0.7615635771754136, "K_sat": 245.2196926301282,
    "E_sym": 31.54573112181783, "L_sym": 47.07659754891122,
}


def print_nmp_summary(par=None):
    """Print the nuclear-matter parameters beside their published values.

    A reading aid, not part of any solve: nothing in `eos` calls it, and
    `verify/run_full_check.py` is what asserts the agreement.
    """
    par = par or Parameters.default()
    nmp = compute_nmp(par)
    print(f"nuclear-matter parameters, {getattr(par, 'name', '?')}")
    print(f"{'':14s} {'this model':>12s} {'published':>12s} {'difference':>12s}")
    for key in ("n_sat", "E_sat", "m_eff_ratio", "K_sat", "E_sym", "L_sym"):
        got, want = nmp[key], PUBLISHED_NMP[key]
        print(f"  {key:12s} {got:12.4f} {want:12.4f} {got - want:+12.4f}")
    print("  predictions, imposed by no fit:")
    for key in ("Q_sat", "K_sym"):
        print(f"  {key:12s} {nmp[key]:12.4f}")
    print(f"  {'P_sat':12s} {nmp['P_sat']:12.3e}   (zero by construction)")


if __name__ == "__main__":
    print_nmp_summary()

