"""
nmp.py
======
The maps between SFHo's couplings and the physical quantities they are
fitted to, in both directions.

    compute_nmp(par)                couplings -> nuclear-matter parameters
    compute_hyperon_potentials(par)  couplings -> U_Lambda, U_Sigma, U_Xi
    create_custom_parametrization(U_Lambda_N, ...)   the inverse of that

This module sits ABOVE `solver.py` in the import order (CLAUDE.md section
5), and it has to: every quantity here is defined by a property of the
SOLVED state -- the saturation density is where the pressure vanishes, the
effective mass is read off the converged fields -- so computing any of them
means solving symmetric matter. That is also why a constructor that inverts
one of these maps is a free function here rather than a classmethod on the
parameter dataclass, which is at the bottom of the same order.

Definitions follow the CompOSE manual (Typel et al., arXiv:2203.03209 sec.
6) and Steiner, Prakash, Lattimer & Ellis, Phys. Rept. 411 (2005).

Units:
- Energies/masses/potentials: MeV
- Densities: fm^-3
"""
import copy
import numpy as np
from dataclasses import dataclass, field as dataclass_field
from typing import Optional, Tuple, Dict
from scipy.optimize import brentq, root

from eos.general.physics_constants import hc, hc3
from eos.sfho.species import SpeciesFlags
from eos.sfho.parameters import (
    Parameters, SU6_RATIOS, SQRT2, _get_base_sfho
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
        sigma, omega, _, _ = compute_saturation_fields()
    
    potentials = {}
    
    for hyperon, label in [('lambda', 'U_Lambda'), 
                           ('sigma+', 'U_Sigma'), 
                           ('xi0', 'U_Xi')]:
        if hyperon in params.couplings_map:
            g_sigma_H = params.couplings_map[hyperon]['sigma']
            g_omega_H = params.couplings_map[hyperon]['omega']
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

#: The temperature the T -> 0 limit is taken at. SFHo's Fermi integrals accept
#: T = 0, but the whole NMP path is finite differences of eps, and a strictly
#: cold solve puts a threshold kink exactly where the differences straddle.
#: 0.01 MeV is far below any nuclear scale and keeps the sweep smooth.
T_COLD = 0.01


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


def compute_nmp(par, h=1e-3, n_lo=0.12, n_hi=0.20):
    """
    Nuclear-matter parameters at saturation.

    Returns dict with n_sat [fm^-3], E_sat, K_sat, Q_sat, E_sym, L_sym,
    K_sym [MeV], m_eff_ratio, and P_sat [MeV/fm^3] (diagnostic, ~0 by
    construction). The same keys `eos.dd2.compute_nmp` returns, so one caller
    reads either model.

    The derivatives are central differences AT saturation, not derivatives of
    a spline fitted over a density range: a fit spreads the third derivative
    over the whole range it was fitted on, and `bc_type='natural'` additionally
    pins the curvature to zero at the endpoints.

    On `h`. The step has to sit above the solver's own noise and below where
    truncation bites. Measured for SFHo_Nucleonic, K_sat and Q_sat are flat
    from h = 1e-4 to 2e-3 (245.221 +- 0.001 and -467.4 +- 0.1) and start
    drifting at 4e-3, reaching 245.43 / -470.4 by 1.6e-2. The default sits in
    the middle of that plateau.

    Q_sat and K_sym are PREDICTIONS of the parametrization, not quantities any
    fit imposes; they are reported for exactly that reason.
    """
    n_sat = brentq(lambda n: pressure(par, n), n_lo, n_hi, xtol=1e-13)
    at_sat = _snm(par, n_sat)

    EA = lambda n: energy_per_baryon(par, n)
    d2 = (EA(n_sat + h) - 2.0 * EA(n_sat) + EA(n_sat - h)) / h**2
    d3 = (EA(n_sat + 2 * h) - 2.0 * EA(n_sat + h)
          + 2.0 * EA(n_sat - h) - EA(n_sat - 2 * h)) / (2.0 * h**3)
    dEs = (esym(par, n_sat + h) - esym(par, n_sat - h)) / (2.0 * h)
    d2Es = (esym(par, n_sat + h) - 2.0 * esym(par, n_sat)
            + esym(par, n_sat - h)) / h**2

    m_N = 0.5 * (par.m_n + par.m_p)
    return {
        "n_sat": n_sat,
        "E_sat": EA(n_sat),
        "m_eff_ratio": at_sat.matter.m_eff_i["n"] / m_N,
        "K_sat": 9.0 * n_sat**2 * d2,
        "Q_sat": 27.0 * n_sat**3 * d3,
        "E_sym": esym(par, n_sat),
        "L_sym": 3.0 * n_sat * dEs,
        "K_sym": 9.0 * n_sat**2 * d2Es,
        "P_sat": at_sat.P,
    }


# =============================================================================
# CREATE CUSTOM PARAMETRIZATION FROM POTENTIAL DEPTHS
# =============================================================================
def create_custom_parametrization(
    # Hyperon potential depths (MeV)
    U_Lambda_N: float = -30.0,
    U_Sigma_N: float = +30.0,
    U_Xi_N: float = -14.0,
    # Vector coupling enhancement factors (per hyperon family)
    # g_ωH = g_ωN × SU(6)_ratio × y_H, g_φH = g_ωN × SU(6)_ratio × y_H
    y_Lambda: float = 1.0,
    y_Sigma: float = 1.0,
    y_Xi: float = 1.0,
    # Delta couplings
    x_sigma_delta: float = 1.15,
    x_omega_delta: float = 1.0,
    x_rho_delta: float = 1.0,
    # Name
    name: str = "Custom"
) -> Parameters:
    """
    Create custom parametrization from target hyperon potential depths.

    The scalar coupling R_σH is determined from the target potential depth:
        U_H = -g_σH × σ + g_ωH × ω
        R_σH = (R_ωH × y_H × g_ωN × ω - U_H) / (g_σN × σ)

    σ and ω are SOLVED for at saturation, by `compute_saturation_fields`, and
    that is why this function lives in `nmp.py` rather than in `parameters.py`:
    it is an inverse map from a physical observable to a coupling, so it needs
    the solver, and `parameters.py` is the bottom of the import layer and
    cannot reach it (CLAUDE.md §5). A second copy did live there, with the two
    fields written in as constants; the constants were mutually inconsistent —
    no single density reproduces both — and the couplings they produced missed
    the requested depths by about 3 MeV, so asking for U_Λ = -30 delivered
    -33.07. Hardcoding them is also wrong in principle for an inference run,
    where the base couplings vary and the saturation fields move with them.

    Vector couplings follow SU(6) symmetry × y_H enhancement factor per family:
        g_ωΛ = g_ωN × (2/3) × y_Lambda
        g_ωΣ = g_ωN × (2/3) × y_Sigma  
        g_ωΞ = g_ωN × (1/3) × y_Xi
    
    Example: y_Lambda=1.5, y_Sigma=1.5, y_Xi=1.875 gives:
        g_ωΛ = 1.0 × g_ωN, g_ωΣ = 1.0 × g_ωN, g_ωΞ = 0.625 × g_ωN
    
    Args:
        U_Lambda_N: Λ potential depth at n_sat in SNM (MeV), ~ -30 MeV
        U_Sigma_N: Σ potential depth at n_sat in SNM (MeV), ~ +30 MeV  
        U_Xi_N: Ξ potential depth at n_sat in SNM (MeV), ~ +10 to -20 MeV
        y_Lambda: Enhancement factor for Λ (1.0 = SU(6))
        y_Sigma: Enhancement factor for Σ (1.0 = SU(6))
        y_Xi: Enhancement factor for Ξ (1.0 = SU(6))
        x_sigma_delta: R_σΔ = g_σΔ/g_σN
        x_omega_delta: R_ωΔ = g_ωΔ/g_ωN
        x_rho_delta: R_ρΔ = g_ρΔ/g_ρN
        name: Name for the parametrization
        
    Returns:
        Parameters with computed couplings
    """
    # Get base SFHo parameters
    p = _get_base_sfho()
    p.name = name
    
    # Compute saturation fields
    sigma, omega, _, _ = compute_saturation_fields()
    
    # SU(6) vector ratios (before enhancement)
    R_omega_Lambda_SU6 = 2.0/3.0
    R_omega_Sigma_SU6 = 2.0/3.0
    R_omega_Xi_SU6 = 1.0/3.0
    
    R_phi_Lambda_SU6 = -SQRT2/3.0
    R_phi_Sigma_SU6 = -SQRT2/3.0
    R_phi_Xi_SU6 = -2.0*SQRT2/3.0
    
    # Apply enhancement factors
    R_omega_Lambda = R_omega_Lambda_SU6 * y_Lambda
    R_omega_Sigma = R_omega_Sigma_SU6 * y_Sigma
    R_omega_Xi = R_omega_Xi_SU6 * y_Xi
    
    R_phi_Lambda = R_phi_Lambda_SU6 * y_Lambda
    R_phi_Sigma = R_phi_Sigma_SU6 * y_Sigma
    R_phi_Xi = R_phi_Xi_SU6 * y_Xi
    
    # Compute scalar coupling ratios from potential depths
    # U_H = -R_σH × g_σN × σ + R_ωH × g_ωN × ω
    # R_σH = (R_ωH × g_ωN × ω - U_H) / (g_σN × σ)
    
    def compute_R_sigma(U_H: float, R_omega: float) -> float:
        return (R_omega * p.g_omega_N * omega - U_H) / (p.g_sigma_N * sigma)
    
    R_sigma_Lambda = compute_R_sigma(U_Lambda_N, R_omega_Lambda)
    R_sigma_Sigma = compute_R_sigma(U_Sigma_N, R_omega_Sigma)
    R_sigma_Xi = compute_R_sigma(U_Xi_N, R_omega_Xi)
    
    # Lambda couplings
    p.couplings_map['lambda'] = {
        'sigma': R_sigma_Lambda * p.g_sigma_N,
        'omega': R_omega_Lambda * p.g_omega_N,
        'phi': R_phi_Lambda * p.g_omega_N,
        'rho': 0.0,
    }
    
    # Sigma couplings (all Σ+, Σ0, Σ-)
    sigma_couplings = {
        'sigma': R_sigma_Sigma * p.g_sigma_N,
        'omega': R_omega_Sigma * p.g_omega_N,
        'phi': R_phi_Sigma * p.g_omega_N,
        'rho': 2.0 * p.g_rho_N,
    }
    for s_name in ['sigma+', 'sigma0', 'sigma-']:
        p.couplings_map[s_name] = sigma_couplings.copy()
    
    # Xi couplings
    xi_couplings = {
        'sigma': R_sigma_Xi * p.g_sigma_N,
        'omega': R_omega_Xi * p.g_omega_N,
        'phi': R_phi_Xi * p.g_omega_N,
        'rho': 1.0 * p.g_rho_N,
    }
    for x_name in ['xi0', 'xi-']:
        p.couplings_map[x_name] = xi_couplings.copy()
    
    # Delta couplings
    delta_couplings = {
        'sigma': x_sigma_delta * p.g_sigma_N,
        'omega': x_omega_delta * p.g_omega_N,
        'phi': 0.0,  # Deltas don't couple to φ
        'rho': x_rho_delta * p.g_rho_N,
    }
    for d_name in ['delta++', 'delta+', 'delta0', 'delta-']:
        p.couplings_map[d_name] = delta_couplings.copy()
    
    return p


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
    #: the recovered couplings with `compute_nmp`'s own stencils:
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


def _isoscalar_quantities(par, n_sat, h=1e-3):
    """{P, E_sat, m_ratio, K_sat} of symmetric matter AT n_sat.

    No P = 0 search: the inversion imposes P(n_sat) = 0 as one of its
    conditions instead, so the target saturation density is where these are
    evaluated rather than something to be found first.
    """
    at = _snm(par, n_sat)
    EA = lambda n: energy_per_baryon(par, n)
    d2 = (EA(n_sat + h) - 2.0 * EA(n_sat) + EA(n_sat - h)) / h ** 2
    m_N = 0.5 * (par.m_n + par.m_p)
    return dict(P=at.P, E_sat=at.eps / n_sat - m_N,
                m_ratio=at.matter.m_eff_i["n"] / m_N, K_sat=9.0 * n_sat ** 2 * d2)


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


def invert_nmp(par_base=None, seed=None, n_restarts=N_RESTARTS, **nmp):
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

    The hyperon and Delta sectors are NOT refitted: they ride along from
    `par_base`, and their coupling ratios are defined against the nucleon
    couplings this function has just changed. Re-derive them with
    `create_custom_parametrization` on the result if the potential depths are
    meant to be held.
    """
    required = ("n_sat", "E_sat", "m_eff_ratio", "K_sat", "E_sym", "L_sym")
    missing = [k for k in required if k not in nmp]
    if missing:
        raise ValueError(f"invert_nmp needs {required}; missing {missing}")
    base = par_base or Parameters.default()
    n_sat = float(nmp["n_sat"])

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

    h = 1e-3

    def isov_residual(x):
        g_rho, b1 = x
        if g_rho <= 0.0:
            return [1e3] * 2
        try:
            p = _trial_par(base, g_sigma, g_omega, b, c, g_rho=g_rho, b1=b1)
            E = esym(p, n_sat)
            L = 3.0 * n_sat * (esym(p, n_sat + h) - esym(p, n_sat - h)) / (2 * h)
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

    # Report what the closure did NOT impose, with the forward map's stencils.
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
#: Steiner, Hempel & Fischer, ApJ 774 (2013) 17, as tabulated by Fortin,
#: Oertel & Providencia, PASA 35 (2018) e044 Table 2 and by the CompOSE
#: HS(SFHo) entry. Q_sat and K_sym are not fitted quantities and carry no
#: published value here; `compute_nmp` reports them as predictions.
PUBLISHED_NMP = {
    "n_sat": 0.1583, "E_sat": -16.19, "m_eff_ratio": 0.76,
    "K_sat": 245.4, "E_sym": 31.57, "L_sym": 47.10,
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

