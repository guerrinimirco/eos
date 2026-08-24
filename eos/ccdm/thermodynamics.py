"""Quantities of the chiral colour-dielectric model computed FROM the state.

The state is (Phi, sigma, zeta, Sigma_V; Delta_1, Delta_2, Delta_3; the mode
potentials; T): the dilaton in its solve variable, the two scalar condensates,
the vector SELF-ENERGY (not the field -- see `state_at`), three diquark gaps,
and the potentials that follow from
(mu_B, mu_C, mu_S, mu_3, mu_8). This module takes that state and returns
everything else -- the effective masses, the medium integrals, the two
potentials U and V, Omega, eps, s, the conserved-charge sums and the residuals
that the state must satisfy. IT NEVER KNOWS WHICH EQUILIBRIUM MODE IT IS IN;
imposing beta equilibrium or a charge fraction is `eos.ccdm.solver`.

The one thing it does close on its own is the model's INTERNAL
self-consistency -- fields, gaps and, when the phase is paired, the two colour
potentials that make it colour-neutral. That is the phase-adapter contract of
CLAUDE.md section 5 seen from inside: colour neutrality is a structural
property of a colour-superconducting phase, not a condition a caller chooses,
so `thermo_from_mu` closes it and `eos.mixed` never learns that mu_3 and mu_8
exist.

Reading order, which is the physics: the dielectric and the two potentials,
one mode as an ideal gas, the assembly of one state, the residual rows, then
the internal solve.

What is specific to THIS model, and what each mistake costs
-----------------------------------------------------------
  * THE DILATON IS SOLVED IN Phi = phi_bar^4, not in phi_bar. R_1 has a
    spurious root at phi_bar = 0 where both of its terms vanish as
    phi_bar^3 -- an artefact of the parametrization, since the Jacobian
    dPhi/dphi_bar = 4 phi_bar^3 vanishes there and not the physics. A Newton
    solve in phi_bar landed on it from three of five starting points. In Phi
    the same equation reads dU/dPhi = B_g ln Phi, which runs to -infinity
    there, and no such root exists;

  * CONFINEMENT IS A PINNING, NOT A SMOOTH SUPPRESSION. At T = 0 a mode with
    M* >= mu* contributes identically zero, and `eos.general.fermi_gauss`
    returns exactly zero rather than a small number. That IS the confinement
    mechanism -- as phi_bar -> 1 the dielectric closes, M* diverges and the
    quarks leave the medium -- and smoothing it destroys the first-order
    deconfinement transition it produces;

  * THE VECTOR SOURCE IS n_q = 3 n_B. Using n_B understates omega_0 by three
    and the repulsive energy by nine;

  * THE MEDIUM INTEGRALS ARE UNREGULARISED. They terminate at their own Fermi
    momenta; the cutoff Lambda belongs to the PAIRING integral alone, which is
    why a sharp one is admissible here and why `Parameters.mu_ceiling`
    declares where the pairing sector stops being trustworthy;

  * the paired densities, scalar densities and entropy are NOT the unpaired
    Fermi integrals. Those corrections come from `eos.general.pairing`.

Two corrections to the specification, both forced by its own section 9.6
Euler audit
-----------------------------------------------------------------------
`docs/ccdm_implementation.md` is the authority for this model, and two places
in its assembly do not survive the audit it itself mandates:

  * its section 4.3 writes eps with -(1/2) m_omega^2 omega_0^2. The sign must
    be PLUS. A repulsive vector interaction adds energy density, and Euler
    with the vector term entering P at +(1/2) m_omega^2 omega_0^2 fixes the
    same sign in eps. This is the standard density-dependent mean-field
    result: the scalar potentials enter eps positively and P negatively, the
    vector term enters BOTH positively;

  * its section 4.1 carries Sigma_R in mu* but omits the compensating
    -Sigma_R n_q from Omega. Without it n = -dOmega/dmu and Euler both fail as
    soon as g_omega depends on the density. The rearrangement term enters mu
    and P and NEVER eps (CLAUDE.md section 8), which is exactly where it is
    put here.

Both are written out in `ccdm.tex` beside the specification's forms.

Units are natural inside this module: momenta, masses and potentials in MeV,
densities in MeV^3, Omega, P and eps in MeV^4. The fm-based public boundary is
`eos.ccdm.api`.
"""
from dataclasses import dataclass
import math

import numpy as np

from eos.ccdm.couplings import (
    diquark_coupling, has_vector, rearrangement, vector_field,
    vector_self_energy,
)
from eos.ccdm.species import pattern_mask
from eos.general.fermi_gauss import kinetic_thermo, unbounded_k_max
from eos.general.pairing import (
    CHARGE, FLAVOUR_OF_MODE, N_MODES, STRANGENESS, colour_densities,
    mode_potentials, pair_block,
)
from eos.general.physics_constants import hc3
from eos.general.solve import solve_system

_PI2 = math.pi ** 2

#: The dilaton solve variable is confined to [PHI_FLOOR, PHI_CEIL].
#:
#: The ceiling is not 1 - 1e-9: at Phi = 1 exactly the dielectric closes, M*
#: is infinite and the M* rho_s term of R_1 becomes inf * 0. It is also where
#: the CONFINED branch's solution genuinely sits, since with no quarks R_1
#: reduces to B_g ln Phi, whose only root is Phi = 1. So the ceiling has to be
#: close enough that |ln Phi| = 1e-13 falls inside the residual gate the model
#: claims to accept at, and far enough that M* = 2.8e15 MeV stays an ordinary
#: double. 1 - 1e-13 is both.
PHI_FLOOR = 1.0e-14
PHI_CEIL = 1.0 - 1.0e-13

#: How many thermal widths above its potential an effective mass has to be
#: before its mode is treated as absent at T > 0. At 60 T the occupation is
#: e^-60 ~ 1e-26, so this is exact to well inside double precision; it exists
#: because the confined branch drives M* to 1e15 MeV, where integrating is not
#: wrong, merely pointless. AT T = 0 THE TEST IS EXACT AND IS
#: `eos.general.fermi_gauss`'s, not a threshold at all.
ABSENT_WIDTHS = 60.0


# =============================================================================
# THE DIELECTRIC, AND THE TWO POTENTIALS
# =============================================================================
def guard_phi(Phi):
    """Phi held inside [PHI_FLOOR, PHI_CEIL].

    A Newton step may propose a dilaton outside the interval the model is
    defined on; evaluating there gives a NaN residual and the solve dies with
    no information. Clamping evaluates at the edge instead, where
    dU/dPhi = B_g ln Phi is strongly signed and pushes the next step back
    inside.
    """
    return min(max(float(Phi), PHI_FLOOR), PHI_CEIL)


def dielectric(par, Phi):
    """chi = (1 - Phi)^p, the dielectric function, dimensionless.

    Phi = phi_bar^4 is the gluon condensate in units of its vacuum value, so
    1 - Phi is the medium's deviation from transparency and chi is LINEAR in
    it at p = 1. chi -> 0 at Phi -> 1 (confinement, M* -> infinity) and
    chi -> 1 at Phi -> 0 (perturbative, M* -> m_f).
    """
    return (1.0 - guard_phi(Phi)) ** par.p


def effective_masses(par, Phi, sigma, zeta):
    """(M*_u, M*_d, M*_s) [MeV], in the flavour order of `eos.general.pairing`.

        M*_u = (g_q sigma + m_u)/chi ,  M*_d = (g_q sigma + m_d)/chi ,
        M*_s = (g_s zeta  + m_s)/chi

    The dielectric sits in the DENOMINATOR: it is the medium's transparency to
    colour, and a quark in an opaque medium is heavy. Both mechanisms are in
    here at once -- chiral symmetry breaking through sigma and zeta, and
    confinement through chi -- which is what makes chiral restoration and
    deconfinement two aspects of one transition in this model rather than two
    independent ones.
    """
    chi = dielectric(par, Phi)
    return np.array([(par.g_q * sigma + par.m_u) / chi,
                     (par.g_q * sigma + par.m_d) / chi,
                     (par.g_s * zeta + par.m_s) / chi])


def glue_potential(par, Phi):
    """U(Phi) = B_g [Phi (ln Phi - 1) + 1] [MeV^4].

    Zero at the physical vacuum Phi = 1 and equal to B_g at the perturbative
    point Phi = 0, which is what makes B_g the GLUE part of the effective bag
    constant. The limit Phi ln Phi -> 0 is special-cased: without it the
    perturbative point returns NaN.
    """
    Phi = guard_phi(Phi)
    log_term = Phi * math.log(Phi) if Phi > 0.0 else 0.0
    return par.B_g * (log_term - Phi + 1.0)


def glue_derivative(par, Phi):
    """dU/dPhi = B_g ln Phi [MeV^4].

    The whole reason Phi is the solve variable: it runs to -infinity at the
    perturbative point, so the spurious phi_bar = 0 root of the same equation
    written in phi_bar simply does not exist here.
    """
    return par.B_g * math.log(guard_phi(Phi))


def chiral_potential(par, sigma, zeta):
    """V(sigma, zeta) [MeV^4], the Mexican hat with explicit breaking.

        V = (lambda/4)(sigma^2 - v^2)^2 + (lambda_z/4)(zeta^2 - v_zeta^2)^2
            - eps_sigma sigma - eps_zeta zeta + C_0

    v and v_zeta are CONSTANTS, not fields: the hat radii before explicit
    breaking, and not observables. The linear terms shift the true minima to
    sigma_0 = f_pi and zeta_0. C_0 is whatever puts V(sigma_0, zeta_0) = 0, so
    that Omega needs no vacuum subtraction. v_zeta^2 is NEGATIVE at the
    baseline m_zeta; it appears only squared-off inside the bracket, so
    nothing here takes its square root.
    """
    d = par.derived
    return (0.25 * d.lam * (sigma ** 2 - d.v2) ** 2
            + 0.25 * d.lam_zeta * (zeta ** 2 - d.v_zeta2) ** 2
            - d.eps_sigma * sigma - d.eps_zeta * zeta + d.C_0)


def chiral_derivatives(par, sigma, zeta):
    """(dV/dsigma, dV/dzeta) [MeV^3].

        dV/dsigma = lambda sigma (sigma^2 - v^2) - eps_sigma
        dV/dzeta  = lambda_z zeta (zeta^2 - v_zeta^2) - eps_zeta
    """
    d = par.derived
    return (d.lam * sigma * (sigma ** 2 - d.v2) - d.eps_sigma,
            d.lam_zeta * zeta * (zeta ** 2 - d.v_zeta2) - d.eps_zeta)


def bag_constant(par):
    """B_eff = B_g + B_chi [MeV^4], the field energy across the transition.

        B_g   = U(0) - U(1)          the glue part
        B_chi = V(0, 0) - V(f_pi, zeta_0)   the chiral part

    NOT an input the way a bag constant is in a bag model: the field energy is
    zero by construction at the physical vacuum and equals this at the
    perturbative point, so it is a DERIVED number. THE CHIRAL SECTOR SUPPLIES
    THE LARGER PART -- at the shipped set B_g^(1/4) = 150 MeV against
    B_chi^(1/4) = 229.9 MeV, giving B_eff = (239.7 MeV)^4 = 429.4 MeV/fm^3.
    Quoting B_g alone as "the bag constant" of this model is wrong by a factor
    of six in energy density.
    """
    d = par.derived
    B_chi = chiral_potential(par, 0.0, 0.0) - chiral_potential(par, d.sigma_0,
                                                               d.zeta_0)
    return par.B_g + B_chi


# =============================================================================
# ONE MODE AS AN IDEAL GAS
# =============================================================================
def mode_thermo(mu_star, M_star, T):
    """One colour-flavour mode's medium integrals [natural units].

    An UNREGULARISED relativistic ideal gas: the upper limit is the numerical
    one of `eos.general.fermi_gauss.unbounded_k_max`, chosen so the integrand
    has died, and not a parameter of the model.

    A mode too heavy for its own potential is ABSENT. At T = 0 that test is
    exact and lives in `kinetic_thermo`; at T > 0 it is the `ABSENT_WIDTHS`
    test here, which matters because the confined branch drives M* to 1e15 MeV
    and there is nothing to be learnt from integrating a Fermi function that
    is e^-1e13.
    """
    if T > 0.0 and M_star - abs(mu_star) > ABSENT_WIDTHS * T:
        return kinetic_thermo(0.0, M_star, 0.0, 1.0)      # the absent block
    k_max = unbounded_k_max(mu_star, M_star, T)
    return kinetic_thermo(mu_star, M_star, T, k_max)


# =============================================================================
# ONE SOLVED STATE
# =============================================================================
@dataclass(frozen=True)
class CCDMState:
    """Everything one state of colour-dielectric matter is, in natural units.

    MATTER ONLY: no leptons, no photons. Those are shared by the system rather
    than owned by the phase, and `eos.ccdm.solver` adds them.

    The residual arrays are what a solve drives to zero, carried on the state
    so a caller can see how well the state it was handed is actually solved:

        field_residual  (R_1, R_2, R_3, R_4) -- dilaton, sigma, zeta, omega_0
        gap_residual    Delta_eta/(2 G_D) - kernel_eta, on the FREE gaps

    `n_3` and `n_8` are the colour densities colour neutrality sets to zero.
    In an unpaired region both vanish identically at mu_3 = mu_8 = 0, so they
    are not solved for there -- letting a root finder hunt for mu_8 in an
    unpaired phase is a documented way to lose an afternoon.

    `valid` is False for a converged state that is not a physical one: a
    negative effective mass, which the Mexican hat admits as its reflected
    minimum. The enumeration drops such a candidate rather than ranking it.
    """
    T: float
    Phi: float                          # the dilaton solve variable, phi_bar^4
    phi_bar: float                      # Phi^(1/4)
    chi: float                          # the dielectric
    sigma: float                        # [MeV]
    zeta: float                         # [MeV]
    omega_0: float                      # [MeV]
    Sigma_V: float                      # [MeV] = g_omega omega_0 + Sigma_R
    M_star: np.ndarray                  # [MeV] per flavour
    Delta: np.ndarray                   # [MeV]
    Sigma_R: float                      # [MeV]
    mu_B: float
    mu_C: float
    mu_S: float
    mu_3: float
    mu_8: float
    mu_modes: np.ndarray                # physical mode potentials [MeV]
    mu_star: np.ndarray                 # shifted mode potentials [MeV]
    rho_s: np.ndarray                   # scalar densities per flavour [MeV^3]
    n_modes: np.ndarray                 # [MeV^3]
    n_flavour: np.ndarray               # [MeV^3]
    n_q: float
    n_B: float
    n_C: float
    n_S: float
    n_3: float
    n_8: float
    U: float                            # [MeV^4]
    V: float                            # [MeV^4]
    Omega: float                        # [MeV^4]
    P: float                            # = -Omega [MeV^4]
    eps: float                          # [MeV^4]
    s: float                            # [MeV^3]
    mu_dot_n: float                     # sum_j mu_j n_j [MeV^4]
    field_residual: np.ndarray
    gap_residual: np.ndarray
    branch: str
    pattern: str
    gapless: bool
    valid: bool
    delta_omega: float                  # the pairing correction alone [MeV^4]
    pair_cost: float                    # sum_eta Delta^2/(4 G_D) [MeV^4]

    # --- the fm-based boundary --------------------------------------------
    @property
    def n_B_fm(self):
        return self.n_B / hc3

    @property
    def P_fm(self):
        return self.P / hc3

    @property
    def eps_fm(self):
        return self.eps / hc3

    @property
    def s_fm(self):
        return self.s / hc3

    @property
    def n_C_fm(self):
        return self.n_C / hc3

    @property
    def n_S_fm(self):
        return self.n_S / hc3

    def euler_residual(self):
        """(eps + P - T s - sum_j mu_j n_j)/eps, the identity of section 8."""
        if self.eps == 0.0:
            return 0.0
        return (self.eps + self.P - self.T * self.s - self.mu_dot_n) / self.eps


def state_at(par, Phi, sigma, zeta, Sigma_V, Delta, mu_B, mu_C, mu_S, mu_3,
             mu_8, T, branch="restored", pattern="unpaired"):
    """One state, evaluated. No equilibrium condition is imposed here.

    The assembly, in the order it is written:

        chi     = (1 - Phi)^p ,  M*_f = (g_f phi_f + m_f)/chi
        mu*_j   = mu_j - Sigma_V
        omega_0 = g_omega(n_B) 3 n_B / m_omega^2          (from the densities)
        Sigma_R = (dg_omega/dn_B) omega_0 n_B

        Omega = U(Phi) + V(sigma, zeta) - (1/2) m_omega^2 omega_0^2
                - Sigma_R n_q - sum_j P_j + delta_omega_pair
                + sum_eta Delta_eta^2/(4 G_D)
        eps   = sum_j eps_j + U + V + (1/2) m_omega^2 omega_0^2 + eps_pair
        eps_pair = delta_omega_pair + D + T s_pair + sum_j mu*_j delta_n_j

    with the two corrections to the specification's own assembly documented in
    the module docstring: eps takes +(1/2) m_omega^2 omega_0^2, and Omega
    carries -Sigma_R n_q. Both are what its section 9.6 Euler audit demands,
    and `euler_residual` on the returned state is that audit.

    THE VECTOR UNKNOWN IS Sigma_V, NOT omega_0, and that is what removes the
    circularity the specification handles by nesting a third loop: omega_0
    sets mu*, which sets n_B, which sets omega_0. Carried as the total shift
    Sigma_V = g_omega omega_0 + Sigma_R, everything downstream is explicit --
    mu* follows from Sigma_V, the densities from mu*, omega_0 and Sigma_R from
    the densities -- and R_4 is the one row saying the field is the one the
    returned densities source, which is precisely the assertion the
    specification's section 9.3 asks for. It is also this repository's
    declared convention for a density-dependent coupling (CLAUDE.md
    section 2): the unknown vector uses the effective potentials, so the
    rearrangement and the large vector shift cancel out of the iteration.
    """
    Phi = guard_phi(Phi)
    chi = dielectric(par, Phi)
    M_star = effective_masses(par, Phi, sigma, zeta)
    Delta = np.asarray(Delta, dtype=float)

    mu_modes = mode_potentials(mu_B, mu_C, mu_S, mu_3, mu_8)
    mu_star = mu_modes - Sigma_V
    M_mode = M_star[FLAVOUR_OF_MODE]

    # --- the nine modes as unregularised ideal gases ----------------------
    modes = [mode_thermo(mu_star[j], M_mode[j], T) for j in range(N_MODES)]
    n_med = np.array([m.n for m in modes])
    P_med = sum(m.P for m in modes)
    eps_med = sum(m.eps for m in modes)
    s_med = sum(m.s for m in modes)
    rho_s_med = np.array([sum(modes[j].rho_s for j in range(N_MODES)
                              if FLAVOUR_OF_MODE[j] == i) for i in range(3)])

    # --- the pairing correction -------------------------------------------
    G_D = diquark_coupling(par, chi)
    block = pair_block(M_star, mu_star, Delta, T, par.Lambda)
    n_modes = n_med + block.delta_n
    rho_s = rho_s_med + block.delta_rho_s
    s = s_med + block.delta_s

    n_flavour = np.array([float(np.sum(n_modes[FLAVOUR_OF_MODE == i]))
                          for i in range(3)])
    n_q = float(np.sum(n_flavour))
    n_B = n_q / 3.0

    # --- the vector field the densities source ----------------------------
    omega_0 = vector_field(par, n_B)
    Sigma_R = rearrangement(par, n_B, omega_0)

    D = float(np.sum(Delta ** 2)) / (4.0 * G_D)
    U = glue_potential(par, Phi)
    V = chiral_potential(par, sigma, zeta)
    field_energy = 0.5 * par.m_omega ** 2 * omega_0 ** 2

    eps_pair = (block.delta_omega + D + T * block.delta_s
                + float(np.dot(mu_star, block.delta_n)))
    Omega = (U + V - field_energy - Sigma_R * n_q - P_med
             + block.delta_omega + D)
    eps = eps_med + U + V + field_energy + eps_pair

    n_3, n_8 = colour_densities(n_modes)
    dV_dsigma, dV_dzeta = chiral_derivatives(par, sigma, zeta)
    R_1 = (glue_derivative(par, Phi)
           + par.p * float(np.dot(M_star, rho_s)) / (1.0 - Phi))
    R_2 = dV_dsigma + (par.g_q / chi) * float(rho_s[0] + rho_s[1])
    R_3 = dV_dzeta + (par.g_s / chi) * float(rho_s[2])
    R_4 = Sigma_V - vector_self_energy(par, n_B)

    return CCDMState(
        T=T, Phi=Phi, phi_bar=Phi ** 0.25, chi=chi, sigma=sigma, zeta=zeta,
        omega_0=omega_0, Sigma_V=Sigma_V, M_star=M_star, Delta=Delta,
        Sigma_R=Sigma_R,
        mu_B=mu_B, mu_C=mu_C, mu_S=mu_S, mu_3=mu_3, mu_8=mu_8,
        mu_modes=mu_modes, mu_star=mu_star, rho_s=rho_s,
        n_modes=n_modes, n_flavour=n_flavour, n_q=n_q, n_B=n_B,
        n_C=float(np.dot(CHARGE, n_flavour)),
        n_S=float(np.dot(STRANGENESS, n_flavour)), n_3=n_3, n_8=n_8,
        U=U, V=V, Omega=Omega, P=-Omega, eps=eps, s=s,
        mu_dot_n=float(np.dot(mu_modes, n_modes)),
        field_residual=np.array([R_1, R_2, R_3, R_4]),
        gap_residual=Delta / (2.0 * G_D) - block.gap_kernel,
        branch=branch, pattern=pattern, gapless=block.gapless,
        valid=bool(np.all(M_star > 0.0)),
        delta_omega=block.delta_omega, pair_cost=D)


# =============================================================================
# THE INTERNAL SELF-CONSISTENCY, AT GIVEN CONSERVED-CHARGE POTENTIALS
# =============================================================================
def internal_unknowns(par, pattern):
    """The names of the unknowns `thermo_from_mu` closes, in order.

    Always the four fields. The free gaps of the pattern, and with them the
    two colour potentials, only where the pattern pairs: in an unpaired region
    n_3 and n_8 vanish identically at mu_3 = mu_8 = 0, so they are PINNED
    rather than solved.
    """
    names = ["Phi", "sigma", "zeta"]
    if has_vector(par):
        names.append("Sigma_V")
    mask = pattern_mask(pattern)
    names += [f"Delta_{eta + 1}" for eta in range(3) if mask[eta]]
    if any(mask):
        names += ["mu_3", "mu_8"]
    return tuple(names)


def unpack_internal(x, par, pattern):
    """(Phi, sigma, zeta, Sigma_V, Delta, mu_3, mu_8) from the unknown vector."""
    Phi, sigma, zeta = (float(v) for v in x[:3])
    i = 3
    Sigma_V = 0.0
    if has_vector(par):
        Sigma_V, i = float(x[3]), 4
    mask = pattern_mask(pattern)
    Delta = np.zeros(3)
    for eta in range(3):
        if mask[eta]:
            Delta[eta] = x[i]
            i += 1
    mu_3 = mu_8 = 0.0
    if any(mask):
        mu_3, mu_8 = float(x[i]), float(x[i + 1])
    return Phi, sigma, zeta, Sigma_V, Delta, mu_3, mu_8


def internal_rows(st, par, pattern):
    """The rows the internal system drives to zero, in assembly order.

        R_1  dilaton      B_g ln Phi + p sum_f M*_f rho_s,f/(1 - Phi)
        R_2  sigma        dV/dsigma + (g_q/chi)(rho_s,u + rho_s,d)
        R_3  zeta         dV/dzeta  + (g_s/chi) rho_s,s
        R_4  vector       Sigma_V - [g_omega(n_B) omega_0 + Sigma_R]
        one per free gap  Delta_eta/(2 G_D) - kernel_eta
        colour neutrality n_3 = 0 , n_8 = 0        (paired patterns only)
    """
    mask = pattern_mask(pattern)
    rows = list(st.field_residual[:3])
    if has_vector(par):
        rows.append(st.field_residual[3])
    rows += [st.gap_residual[eta] for eta in range(3) if mask[eta]]
    if any(mask):
        rows += [st.n_3, st.n_8]
    return rows


def internal_scales(par, pattern, mu_B):
    """The scale each internal row balances, so one tolerance means one thing.

    R_1 is an energy density and is judged against B_g; R_2 and R_3 are
    scalar-density rows and are judged against the explicit-breaking terms
    eps_sigma and eps_zeta, which are what they balance in the vacuum; R_4 is
    a potential, judged against mu_B/3; the gap rows and the colour rows are
    densities, judged against (mu_B/3)^3/pi^2. Without this the norm would be dominated by whichever row
    happens to carry the largest units, which here is R_1 by four orders of
    magnitude over R_2.
    """
    d = par.derived
    n_scale = max((abs(mu_B) / 3.0) ** 3 / _PI2, 1.0)
    mu_scale = max(abs(mu_B) / 3.0, 1.0)
    scales = [par.B_g, abs(d.eps_sigma), abs(d.eps_zeta)]
    if has_vector(par):
        scales.append(mu_scale)
    mask = pattern_mask(pattern)
    scales += [n_scale] * sum(mask)
    if any(mask):
        scales += [n_scale, n_scale]
    return scales


def default_internal_guess(par, branch, pattern, mu_B, T, gap_scale=None):
    """A cold start for `thermo_from_mu`: fields from the branch, gaps from
    the pattern.

    The vector field is seeded at zero rather than at a guess: R_4 is nearly
    linear in omega_0, so the first Newton step lands close, and a nonzero
    seed only risks starting on the far side of the density the coupling is
    evaluated at.
    """
    from eos.ccdm.species import branch_seed, pattern_seed
    Phi, sigma, zeta = branch_seed(par, branch)
    guess = [Phi, sigma, zeta]
    if has_vector(par):
        guess.append(0.0)

    if gap_scale is None:
        gap_scale = max(0.1 * mu_B / 3.0, 20.0)
    mask = pattern_mask(pattern)
    seed = pattern_seed(pattern, gap_scale)
    guess += [seed[eta] for eta in range(3) if mask[eta]]
    if any(mask):
        guess += [0.0, -0.02 * mu_B / 3.0]
    return np.array(guess, dtype=float)


def thermo_from_mu(par, mu_B, mu_C=0.0, mu_S=0.0, T=0.0, branch="restored",
                   pattern="unpaired", x0=None, return_state=False):
    """The state at given conserved-charge potentials, self-consistently.

    Closes the model's own internal system -- the four fields, the gaps and
    colour neutrality -- at fixed (mu_B, mu_C, mu_S, T) in ONE declared branch
    and ONE declared pairing pattern. This is the surface `eos.mixed`
    consumes: the engine hands over three potentials and a temperature and
    gets a block back, and never learns that mu_3, mu_8 or a dilaton exist.

    THE BRANCH IS DECLARED, NOT DISCOVERED, here. Choosing between branches is
    a comparison of Omega across separate self-consistent solves, and it is
    `eos.ccdm.solver`'s (and the adapter's) job, not this one's -- which is
    what keeps this function a pure map from potentials to a block.

    NON-CONVERGENCE IS A RETURN VALUE, not an exception: the returned state
    carries the best iterate reached, and `converged` says whether to believe
    it. With `return_state=True` the internal unknown vector comes back beside
    it, which is what a warm start is.
    """
    if x0 is None:
        x0 = default_internal_guess(par, branch, pattern, mu_B, T)

    scales = internal_scales(par, pattern, mu_B)

    def residual(x):
        """The rows ALREADY DIVIDED by their scales; see `internal_scales`."""
        Phi, sigma, zeta, Sigma_V, Delta, mu_3, mu_8 = unpack_internal(
            x, par, pattern)
        st = state_at(par, Phi, sigma, zeta, Sigma_V, Delta, mu_B, mu_C, mu_S,
                      mu_3, mu_8, T, branch=branch, pattern=pattern)
        return [r / s for r, s in zip(internal_rows(st, par, pattern), scales)]

    def unit_scales(x):
        return [1.0] * len(internal_unknowns(par, pattern))

    x, err, ok = solve_system(residual, np.asarray(x0, dtype=float),
                              unit_scales, tol=1.0e-13)

    Phi, sigma, zeta, Sigma_V, Delta, mu_3, mu_8 = unpack_internal(x, par,
                                                                   pattern)
    st = state_at(par, Phi, sigma, zeta, Sigma_V, Delta, mu_B, mu_C, mu_S,
                  mu_3, mu_8, T, branch=branch, pattern=pattern)
    ok = bool(ok and st.valid)
    if return_state:
        return st, ok, err, x
    return st, ok, err
