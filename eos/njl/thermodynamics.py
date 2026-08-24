"""Quantities of the three-flavour NJL model computed FROM the state.

The state is (M_u, M_d, M_s; Delta_1, Delta_2, Delta_3; Sigma_V; the mode
potentials; T): three constituent masses from the gap equation, three diquark
gaps, the vector self-energy, and the potentials that follow from
(mu_B, mu_C, mu_S, mu_3, mu_8). This module takes that state and returns
everything else -- the medium integrals, the Dirac sea, the condensates,
Omega, eps, s, the conserved-charge sums and the residuals that the state must
satisfy. IT NEVER KNOWS WHICH EQUILIBRIUM MODE IT IS IN; imposing beta
equilibrium or a charge fraction is `eos.njl.solver`.

The one thing it does close on its own is the model's INTERNAL
self-consistency -- masses, gaps, vector self-energy and, when the phase is
paired, the two colour potentials that make it colour-neutral. That is the
phase-adapter contract of CLAUDE.md section 5 seen from inside: colour
neutrality is a structural property of a colour-superconducting phase, not a
condition a caller chooses, so `thermo_from_mu` closes it and `eos.mixed`
never learns that mu_3 and mu_8 exist.

Reading order, which is the physics: one mode as a cut Fermi gas, the Dirac
sea, the condensates and the masses they determine, the vacuum, then the
assembly of one state, then the internal solve.

Three things here are specific to a CUT theory and are the ones that go wrong
quietly:

  * P COMES FROM THE LOGARITHM FORM. The two standard pressure integrals differ
    by a surface term that vanishes only when the integrand does at the upper
    limit -- which is true at T = 0 below the cutoff and false everywhere else.
    The k^4/E form is 10% low at T = 30 MeV and 40% low at T = 50 MeV. Both
    forms and the closed-form difference between them are
    `eos.general.fermi_gauss`, which is where the cut ideal gas lives now that
    the colour-dielectric model needs the same integrals (CLAUDE.md section 7);
  * the medium integral is not a spectator. At T = 0 unpaired it is
    self-limiting at k_F, but at T > 0, and in ANY paired phase, the Fermi
    surface is smeared and the cutoff enters. That is why lambda = Lambda_UV/
    Lambda exists as a parameter, and why lambda != 1 raises here rather than
    returning a divergent number (docs/DEFERRED.md);
  * the paired densities, scalar densities and entropy are NOT the unpaired
    Fermi integrals. Those corrections come from `eos.general.pairing`.

The model is the standard three-flavour NJL of Rehberg, Klevansky and Huefner,
Phys. Rev. C 53, 410 (1996), with the diquark channel of Ruester et al.,
Phys. Rev. D 72, 034004 (2005); `njl.tex` writes out every equation below and
`docs/njl_csc_implementation.md` is the implementation specification this
follows.

Units are natural inside this module: momenta, masses and potentials in MeV,
densities in MeV^3, Omega, P and eps in MeV^4. The fm-based public boundary is
`eos.njl.api`.
"""
from dataclasses import dataclass
import math

import numpy as np

from eos.general.fermi_gauss import ModeThermo, kinetic_thermo, surface_term
from eos.general.pairing import (
    CHARGE, FLAVOUR_OF_MODE, N_MODES, STRANGENESS, colour_densities,
    mode_potentials, pair_block, pattern_mask,
)
from eos.general.physics_constants import hc3
from eos.general.solve import solve_system
from eos.njl.couplings import vector_energy, vector_self_energy
from eos.njl.species import DEGENERACY_SEA

_PI2 = math.pi ** 2

#: Damping of the fixed-point iteration on the masses. A root finder on the
#: condensates diverges here -- during development it returned masses that
#: INCREASE with density -- because the gap equation's map is contractive only
#: when it is damped. 0.3 is robust across the whole mu_B range at every
#: vector variant tested.
MASS_DAMPING = 0.3

#: Iteration bound and residual gate of the vacuum fixed point [MeV].
MAX_VACUUM_ITER = 4000
VACUUM_TOL = 1.0e-12


# =============================================================================
# THE DIRAC SEA
# =============================================================================
def sea_scalar_density(m, Lambda, g=DEGENERACY_SEA):
    """The vacuum scalar density of one flavour [MeV^3], in closed form.

        (g/2 pi^2) m/2 [ Lambda sqrt(Lambda^2 + m^2)
                       - m^2 arcsinh(Lambda/m) ]

    Degeneracy g = 2 N_c = 6: spin times colour, because the sea is not
    resolved by colour -- the gap matrix is, the vacuum is not.
    """
    if m <= 0.0:
        return 0.0
    R = math.sqrt(Lambda ** 2 + m ** 2)
    return (g / (2.0 * _PI2)) * 0.5 * m * (Lambda * R
                                           - m ** 2 * math.asinh(Lambda / m))


def sea_energy(m, Lambda, g=DEGENERACY_SEA):
    """The vacuum (Dirac sea) energy density of one flavour [MeV^4].

        (g/2 pi^2) (1/8) [ Lambda (2 Lambda^2 + m^2) sqrt(Lambda^2 + m^2)
                         - m^4 arcsinh(Lambda/m) ]

    It enters BOTH Omega and eps with the same sign, which is what makes the
    vacuum constant identical in the two and Euler hold after subtracting it.
    """
    R = math.sqrt(Lambda ** 2 + m ** 2)
    asinh = math.asinh(Lambda / m) if m > 0.0 else 0.0
    return (g / (2.0 * _PI2)) * 0.125 * (Lambda * (2.0 * Lambda ** 2 + m ** 2) * R
                                         - m ** 4 * asinh)


def f_pi(M, Lambda):
    """The pion decay constant from the quark loop [MeV].

        f_pi^2 = 2 M^2 I_2 ,
        I_2 = (N_c/4 pi^2) [ arcsinh(Lambda/M) - Lambda/sqrt(Lambda^2 + M^2) ]

    The vacuum-fit diagnostic: 92.391 MeV at the RKH point against the
    published 92.4.
    """
    R = math.sqrt(Lambda ** 2 + M ** 2)
    I2 = (3.0 / (4.0 * _PI2)) * (math.asinh(Lambda / M) - Lambda / R)
    return math.sqrt(2.0 * M ** 2 * I2)


# =============================================================================
# CONDENSATES, AND THE MASSES THEY DETERMINE
# =============================================================================
def condensates(par, M, medium_rho_s=(0.0, 0.0, 0.0)):
    """phi_f = <qbar_f q_f> [MeV^3] per flavour, negative in the broken phase.

        phi_f = -rho_s,vac(M_f) + sum_a rho_s,med(f, a)

    the Dirac sea's scalar density, which the medium's own occupation
    subtracts from mode by mode. The three colours of one flavour are summed
    here, so the medium sum carries the same degeneracy 2 x 3 as the sea.
    """
    return np.array([-sea_scalar_density(M[i], par.Lambda) + medium_rho_s[i]
                     for i in range(3)])


def masses_from_condensates(par, phi):
    """The gap equation's right-hand side: M_f from the condensates [MeV].

        M_u = m_u - 4 G_S phi_u + 2 K phi_d phi_s        (and cyclic)

    THE DETERMINANT CROSS-TERMS ARE PART OF IT. Dropping the 2 K phi phi piece
    -- or mis-signing it -- is the single most common transcription error in
    this model, and it leaves a mass equation that still converges and still
    looks like NJL. Note that the convenient field-energy identity
    2 G_S sum phi^2 = sum (M - m)^2/(8 G_S) holds ONLY at K = 0, where it is
    exact to 4e-16; with the determinant on it fails by 34%.
    """
    m = par.current_masses
    G_S, K = par.G_S, par.K
    return np.array([
        m[0] - 4.0 * G_S * phi[0] + 2.0 * K * phi[1] * phi[2],
        m[1] - 4.0 * G_S * phi[1] + 2.0 * K * phi[0] * phi[2],
        m[2] - 4.0 * G_S * phi[2] + 2.0 * K * phi[0] * phi[1],
    ])


def condensate_energy(par, phi):
    """C = 2 G_S sum_f phi_f^2 - 4 K phi_u phi_d phi_s [MeV^4].

    The cost of the quark-antiquark condensates. It enters Omega and eps with
    the SAME sign, which is the second half of why the vacuum constant cancels
    out of Euler.
    """
    return (2.0 * par.G_S * float(np.sum(np.asarray(phi) ** 2))
            - 4.0 * par.K * float(phi[0] * phi[1] * phi[2]))


# =============================================================================
# THE VACUUM
# =============================================================================
@dataclass(frozen=True)
class Vacuum:
    """The vacuum solution: the constants every state is measured against.

    `Omega` and `eps` must be EQUAL -- the same combination of the sea and the
    condensate cost, with no medium, no pairing and no vector term to
    distinguish them. `vacuum_solution` asserts it, because three assembly
    bugs found during development of this model all showed up first as those
    two numbers drifting apart.
    """
    M: np.ndarray                # constituent masses [MeV]
    phi: np.ndarray              # condensates [MeV^3]
    Omega: float                 # [MeV^4]
    eps: float                   # [MeV^4]
    f_pi: float                  # [MeV]


def vacuum_solution(par, M0=(350.0, 350.0, 550.0)):
    """The chirally broken vacuum, by a DAMPED FIXED POINT on the masses.

        M <- M + lambda (M_new[phi(M)] - M) ,   lambda = 0.3

    and not a root finder on the condensates: `fsolve` on phi diverged during
    development, returning masses that increase with density. At the RKH set
    this reproduces the published vacuum with no fitting at all --
    M_u = 367.648 (published 367.7), M_s = 549.479 (549.5),
    (-phi_u)^(1/3) = 241.946 (241.9), (-phi_s)^(1/3) = 257.688 (257.7) and
    f_pi = 92.391 MeV (92.4).
    """
    M = np.array(M0, dtype=float)
    for _ in range(MAX_VACUUM_ITER):
        phi = condensates(par, M)
        M_new = masses_from_condensates(par, phi)
        step = M_new - M
        M = M + MASS_DAMPING * step
        if np.max(np.abs(step)) < VACUUM_TOL:
            break
    else:
        raise RuntimeError("the NJL vacuum fixed point did not converge in "
                           f"{MAX_VACUUM_ITER} iterations")

    phi = condensates(par, M)
    sea = sum(sea_energy(M[i], par.Lambda) for i in range(3))
    C = condensate_energy(par, phi)
    Omega = -sea + C
    return Vacuum(M=M, phi=phi, Omega=Omega, eps=Omega, f_pi=f_pi(M[0], par.Lambda))


def bag_constant(par, vac=None):
    """B_eff [MeV^4]: the vacuum pressure difference across chiral restoration.

    Omega at fixed masses, evaluated at the CURRENT (restored) masses minus at
    the broken-phase ones. It is a DERIVED quantity here, not an input the way
    a bag constant is in a bag model, and it is reported because the
    colour-dielectric companion model quotes its own B_g + B_chi against it:
    (228.93 MeV)^4 = 357.49 MeV/fm^3 at the RKH set.
    """
    if vac is None:
        vac = vacuum_solution(par)
    m = np.array(par.current_masses, dtype=float)
    phi_restored = condensates(par, m)
    Omega_restored = (-sum(sea_energy(m[i], par.Lambda) for i in range(3))
                      + condensate_energy(par, phi_restored))
    return Omega_restored - vac.Omega


# =============================================================================
# ONE SOLVED STATE
# =============================================================================
@dataclass(frozen=True)
class NJLState:
    """Everything one state of NJL matter is, in natural units.

    MATTER ONLY: no leptons, no photons. Those are shared by the system rather
    than owned by the phase, and `eos.njl.solver` adds them.

    The three residual arrays are what a solve drives to zero, carried on the
    state so that a caller can see how well the state it was handed is
    actually solved:

        mass_residual   M_f - [m_f - 4 G_S phi_f + 2 K phi_g phi_h]
        gap_residual    Delta_eta/(2 G_D) - kernel_eta, on the FREE gaps
        vector_residual Sigma_V - dW/dn_q

    `n_3` and `n_8` are the colour densities that colour neutrality sets to
    zero. In an unpaired region both vanish identically at mu_3 = mu_8 = 0, so
    they are not solved for there -- letting a root finder hunt for mu_8 in an
    unpaired phase is a documented way to lose an afternoon.
    """
    T: float
    M: np.ndarray                       # [MeV]
    Delta: np.ndarray                   # [MeV]
    Sigma_V: float                      # [MeV]
    mu_B: float
    mu_C: float
    mu_S: float
    mu_3: float
    mu_8: float
    mu_modes: np.ndarray                # physical mode potentials [MeV]
    mu_star: np.ndarray                 # mu_modes - Sigma_V [MeV]
    phi: np.ndarray                     # condensates [MeV^3]
    n_modes: np.ndarray                 # [MeV^3]
    n_flavour: np.ndarray               # [MeV^3]
    n_q: float
    n_B: float
    n_C: float
    n_S: float
    n_3: float
    n_8: float
    Omega: float                        # vacuum-subtracted [MeV^4]
    P: float                            # = -Omega [MeV^4]
    eps: float                          # vacuum-subtracted [MeV^4]
    s: float                            # [MeV^3]
    mu_dot_n: float                     # sum_j mu_j n_j [MeV^4]
    mass_residual: np.ndarray
    gap_residual: np.ndarray
    vector_residual: float
    pattern: str
    gapless: bool
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


def state_at(par, M, Delta, Sigma_V, mu_B, mu_C, mu_S, mu_3, mu_8, T,
             vac=None, pattern="unpaired", pair_nodes_per_panel=None):
    """One state, evaluated. No equilibrium condition is imposed here.

    The assembly, in the order it is written (section 6.1 of the
    specification), is

        C     = 2 G_S sum_f phi_f^2 - 4 K phi_u phi_d phi_s
        D     = sum_eta Delta_eta^2/(4 G_D)
        W     = G_V(n_q) n_q^2 ,   Sigma_V = dW/dn_q

        Omega = -sum_j P_med,j - sum_f eps_sea,f + C - (Sigma_V n_q - W)
                + delta_omega_pair + D
        eps   =  sum_j eps_med,j - sum_f eps_sea,f + C + W + eps_pair
        eps_pair = delta_omega_pair + D + T s_pair + sum_j mu*_j delta_n_j

    with the vacuum constant subtracted from Omega and eps alike. The lepton
    sector is NOT in either: a phase does not own the leptons.

    Auditing that assembly at every solved point is what catches assembly
    bugs, and each of the three found during development produced a plausible
    EoS: a sign error in eps (Euler off by O(1)); the pairing cost and
    delta_omega dropped from both Omega and eps (Euler off by 8e-3, small
    enough to pass for quadrature); and s_pair dropped, which fails only at
    T > 0.
    """
    if par.lambda_UV != 1.0:
        raise NotImplementedError(
            f"eos.njl runs at lambda = Lambda_UV/Lambda = 1 (conventional "
            f"sharp-cutoff regularization); got lambda = {par.lambda_UV}. "
            f"lambda > 1 needs the RG-consistent counterterm that cancels the "
            f"medium's logarithmic divergence -(2/pi^2) mubar^2 Delta^2 "
            f"ln Lambda_UV, which is not implemented -- see docs/DEFERRED.md. "
            f"Returning a lambda-dependent answer instead would be worse than "
            f"this exception")

    M = np.asarray(M, dtype=float)
    Delta = np.asarray(Delta, dtype=float)
    if vac is None:
        vac = vacuum_solution(par)

    mu_modes = mode_potentials(mu_B, mu_C, mu_S, mu_3, mu_8)
    mu_star = mu_modes - Sigma_V
    M_mode = M[FLAVOUR_OF_MODE]

    # --- the nine modes as cut Fermi gases -------------------------------
    modes = [kinetic_thermo(mu_star[j], M_mode[j], T, par.Lambda_medium)
             for j in range(N_MODES)]
    n_med = np.array([m.n for m in modes])
    P_med = sum(m.P for m in modes)
    eps_med = sum(m.eps for m in modes)
    s_med = sum(m.s for m in modes)
    rho_s_med = np.array([sum(modes[j].rho_s
                              for j in range(N_MODES) if FLAVOUR_OF_MODE[j] == i)
                          for i in range(3)])

    # --- the pairing correction ------------------------------------------
    kwargs = ({} if pair_nodes_per_panel is None
              else {"nodes_per_panel": pair_nodes_per_panel})
    block = pair_block(M, mu_star, Delta, T, par.Lambda_medium, **kwargs)
    n_modes = n_med + block.delta_n
    rho_s = rho_s_med + block.delta_rho_s
    s = s_med + block.delta_s

    # --- condensates, and the sums ---------------------------------------
    phi = condensates(par, M, rho_s)
    C = condensate_energy(par, phi)
    D = float(np.sum(Delta ** 2)) / (4.0 * par.G_D)
    sea = sum(sea_energy(M[i], par.Lambda) for i in range(3))

    n_flavour = np.array([float(np.sum(n_modes[FLAVOUR_OF_MODE == i]))
                          for i in range(3)])
    n_q = float(np.sum(n_flavour))
    W = vector_energy(par, n_q)
    Sigma_V_new = vector_self_energy(par, n_q)

    eps_pair = (block.delta_omega + D + T * block.delta_s
                + float(np.dot(mu_star, block.delta_n)))
    Omega = (-P_med - sea + C - (Sigma_V * n_q - W)
             + block.delta_omega + D - vac.Omega)
    eps = eps_med - sea + C + W + eps_pair - vac.eps

    n_3, n_8 = colour_densities(n_modes)
    return NJLState(
        T=T, M=M, Delta=Delta, Sigma_V=Sigma_V,
        mu_B=mu_B, mu_C=mu_C, mu_S=mu_S, mu_3=mu_3, mu_8=mu_8,
        mu_modes=mu_modes, mu_star=mu_star, phi=phi,
        n_modes=n_modes, n_flavour=n_flavour, n_q=n_q, n_B=n_q / 3.0,
        n_C=float(np.dot(CHARGE, n_flavour)),
        n_S=float(np.dot(STRANGENESS, n_flavour)), n_3=n_3, n_8=n_8,
        Omega=Omega, P=-Omega, eps=eps, s=s,
        mu_dot_n=float(np.dot(mu_modes, n_modes)),
        mass_residual=M - masses_from_condensates(par, phi),
        gap_residual=Delta / (2.0 * par.G_D) - block.gap_kernel,
        vector_residual=Sigma_V - Sigma_V_new,
        pattern=pattern, gapless=block.gapless,
        delta_omega=block.delta_omega, pair_cost=D)


# =============================================================================
# THE INTERNAL SELF-CONSISTENCY, AT GIVEN CONSERVED-CHARGE POTENTIALS
# =============================================================================
def internal_unknowns(par, pattern):
    """The names of the unknowns `thermo_from_mu` closes, in order.

    Always the three masses. The free gaps of the pattern, and with them the
    two colour potentials, only where the pattern pairs: in an unpaired region
    n_3 and n_8 vanish identically at mu_3 = mu_8 = 0, so they are PINNED
    rather than solved. Sigma_V only where there is a vector coupling to carry.
    """
    names = ["M_u", "M_d", "M_s"]
    mask = pattern_mask(pattern)
    names += [f"Delta_{eta + 1}" for eta in range(3) if mask[eta]]
    if any(mask):
        names += ["mu_3", "mu_8"]
    if has_vector(par):
        names.append("Sigma_V")
    return tuple(names)


def has_vector(par):
    """Is there a vector coupling at all? A zero one is not carried.

    Public because `solver.py` asks the same question when it lays out the
    mode's unknown vector: whether Sigma_V is an unknown is one fact, and one
    fact has one home.
    """
    if par.vector_form == "gluon_exchange":
        return par.G_V0_over_GS != 0.0
    return par.eta_V != 0.0


def default_internal_guess(par, pattern, mu_B, T, vac=None, gap_scale=None):
    """A cold start for `thermo_from_mu`: masses from the vacuum, gaps seeded
    by the pattern.

    The mass seed interpolates from the broken vacuum towards the current
    masses as mu_B rises, which is the direction the physics moves; the gap
    seed is the pattern's own (see `eos.njl.species.PATTERNS`), scaled by
    `gap_scale`, whose default is a tenth of the baryon potential -- the order
    of magnitude a strongly-coupled gap has, and comfortably above the barrier
    root the gap equation also carries.
    """
    from eos.njl.species import pattern_seed
    if vac is None:
        vac = vacuum_solution(par)
    x = min(max((mu_B / 3.0) / 400.0, 0.0), 1.0)
    m = np.array(par.current_masses, dtype=float)
    M = vac.M * (1.0 - x) + m * x
    if gap_scale is None:
        gap_scale = max(0.1 * mu_B / 3.0, 20.0)

    guess = list(M)
    mask = pattern_mask(pattern)
    seed = pattern_seed(pattern, gap_scale)
    guess += [seed[eta] for eta in range(3) if mask[eta]]
    if any(mask):
        guess += [0.0, -0.02 * mu_B / 3.0]
    if has_vector(par):
        guess.append(0.0)
    return np.array(guess, dtype=float)


def _unpack_internal(x, par, pattern):
    """(M, Delta, mu_3, mu_8, Sigma_V) from the internal unknown vector."""
    M = np.asarray(x[:3], dtype=float)
    mask = pattern_mask(pattern)
    Delta = np.zeros(3)
    i = 3
    for eta in range(3):
        if mask[eta]:
            Delta[eta] = x[i]
            i += 1
    mu_3 = mu_8 = 0.0
    if any(mask):
        mu_3, mu_8 = float(x[i]), float(x[i + 1])
        i += 2
    Sigma_V = float(x[i]) if has_vector(par) else 0.0
    return M, Delta, mu_3, mu_8, Sigma_V


def internal_residual(x, par, mu_B, mu_C, mu_S, T, vac, pattern):
    """The rows `thermo_from_mu` drives to zero, in the order it assembles them.

        three gap equations         M_f - [m_f - 4 G_S phi_f + 2 K phi_g phi_h]
        one per free gap            Delta_eta/(2 G_D) - kernel_eta
        colour neutrality           n_3 = 0 ,  n_8 = 0        (paired only)
        the vector self-energy      Sigma_V - dW/dn_q         (if there is one)
    """
    M, Delta, mu_3, mu_8, Sigma_V = _unpack_internal(x, par, pattern)
    st = state_at(par, M, Delta, Sigma_V, mu_B, mu_C, mu_S, mu_3, mu_8, T,
                  vac=vac, pattern=pattern)
    mask = pattern_mask(pattern)
    rows = list(st.mass_residual)
    rows += [st.gap_residual[eta] for eta in range(3) if mask[eta]]
    if any(mask):
        rows += [st.n_3, st.n_8]
    if has_vector(par):
        rows.append(st.vector_residual)
    return rows


def thermo_from_mu(par, mu_B, mu_C=0.0, mu_S=0.0, T=0.0, pattern="unpaired",
                   x0=None, vac=None, return_state=False):
    """The state at given conserved-charge potentials, self-consistently.

    Closes the model's own internal system -- masses, gaps, colour neutrality
    and the vector self-energy -- at fixed (mu_B, mu_C, mu_S, T) in one
    declared pairing pattern. This is the surface `eos.mixed` consumes: the
    engine hands over three potentials and a temperature and gets a block
    back, and never learns that mu_3 and mu_8 exist.

    NON-CONVERGENCE IS A RETURN VALUE, not an exception: the returned state
    carries the best iterate reached, and `converged` says whether to believe
    it. With `return_state=True` the internal unknown vector comes back beside
    it, which is what a warm start is.
    """
    if vac is None:
        vac = vacuum_solution(par)
    if x0 is None:
        x0 = default_internal_guess(par, pattern, mu_B, T, vac)

    def residual(x):
        """The rows ALREADY DIVIDED by their scales; see `internal_scales`."""
        raw = internal_residual(x, par, mu_B, mu_C, mu_S, T, vac, pattern)
        return [r / s for r, s in zip(raw, internal_scales(x, par, pattern,
                                                           mu_B))]

    def unit_scales(x):
        return [1.0] * len(internal_unknowns(par, pattern))

    x, err, ok = solve_system(residual, np.asarray(x0, dtype=float),
                              unit_scales, tol=1.0e-13)

    M, Delta, mu_3, mu_8, Sigma_V = _unpack_internal(x, par, pattern)
    st = state_at(par, M, Delta, Sigma_V, mu_B, mu_C, mu_S, mu_3, mu_8, T,
                  vac=vac, pattern=pattern)
    if return_state:
        return st, ok, err, x
    return st, ok, err


def internal_scales(x, par, pattern, mu_B):
    """The scale each internal row balances, so one tolerance means one thing.

    A mass row is a potential and is judged against mu_B/3. A GAP row is not:
    Delta_eta/(2 G_D) has units of MeV^3, since G_D carries MeV^-2, so it is a
    density and is judged against the quark-density scale (mu_B/3)^3/pi^2 --
    as are the two colour rows. Judging a gap row against a potential instead
    is four orders of magnitude too strict and makes a perfectly converged
    solve report a residual of 1e-8.
    """
    mask = pattern_mask(pattern)
    mu_scale = max(abs(mu_B) / 3.0, 1.0)
    n_scale = max(mu_scale ** 3 / _PI2, 1.0)
    scales = [mu_scale] * 3
    scales += [n_scale] * sum(mask)
    if any(mask):
        scales += [n_scale, n_scale]
    if has_vector(par):
        scales.append(mu_scale)
    return scales
