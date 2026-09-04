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
    `eos.general.fermi_integrals`, the fourth method in the one home the Fermi
    integrals of this repository have (CLAUDE.md section 7);
  * the medium integral is not a spectator. At T = 0 unpaired it is
    self-limiting at k_F, but at T > 0, and in ANY paired phase, the Fermi
    surface is smeared and the cutoff enters. That is what lambda = Lambda_UV/
    Lambda answers: the Dirac sea keeps Lambda, the medium runs to Lambda_UV,
    and `counterterm` cancels the logarithm a paired medium picks up on the
    way. lambda = 1 makes that counterterm identically zero and is the
    conventional sharp-cutoff model;
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
from dataclasses import dataclass, replace
from functools import lru_cache
import math

import numpy as np

from eos.general.fermi_integrals import (
    ABSENT, DEGENERACY, NODES_PER_PANEL, ModeThermo, _gauss_legendre,
    kinetic_thermo, surface_term,
)
from eos.general.pairing import (active_gaps, 
    CHARGE, FLAVOUR_OF_MODE, N_MODES, STRANGENESS, colour_densities,
    mode_potentials, pair_block, pair_nodes, pattern_mask,
)
from eos.general.physics_constants import hc3
from eos.general.solve import solve_system
from eos.njl.couplings import vector_energy, vector_self_energy
from eos.njl.species import DEGENERACY_SEA

# `backends/` is optional: CLAUDE.md section 5 defines it by the property that
# deleting it changes no number, only the time they take. With it gone, or
# with numba absent, `backend="fast"` raises and the reference path below is
# the whole story.
try:
    from eos.njl.backends.kernel_numba import NUMBA_OK, modes_thermo
except ImportError:                       # pragma: no cover - backends/ removed
    NUMBA_OK = False
    modes_thermo = None

_PI2 = math.pi ** 2

#: The reference Gauss-Legendre rule handed to the jitted backend. Memoized in
#: `eos.general.fermi_integrals`, so the two flavours quadrature on the same
#: nodes and cannot drift apart in the rule itself.
_GAUSS_X, _GAUSS_W = _gauss_legendre(NODES_PER_PANEL)

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
# THE RG-CONSISTENT COUNTERTERM
# =============================================================================
# Taking the medium integral to Lambda_UV >> Lambda is what removes the cutoff
# artifacts (Gholami, Hofmann and Buballa, Phys. Rev. D 111, 014021 (2025),
# arXiv:2408.06704), but the medium part of a PAIRED phase does not converge
# there: it carries a logarithmic divergence
#
#     Omega_med  ~  -(1/pi^2) sum_pairs mubar_ij^2 Delta_eta^2 ln Lambda_UV
#
# which exists only when mu != 0 AND Delta != 0 and does not scale with the
# quark masses. The counterterm below cancels it exactly. Three schemes are
# published; this is the MASSLESS one (their Eq. C7), which sets the quark
# masses to zero in the renormalization factors and is therefore the one with a
# closed form. It is the scheme the RG-consistent papers use: Gholami et al.,
# arXiv:2411.04064 section II, states it in so many words ("all quark masses
# are set to zero in the counterterm ... the massless renormalization scheme"),
# and Kunkel et al., arXiv:2607.11537, inherit it. The massive scheme is
# rejected by its own authors (it inverts the gap ordering to
# Delta_3 < Delta_1 = Delta_2 and so predicts the wrong melting pattern), so
# there is one scheme here and no scheme argument.
#
# The authors' own code -- the MUSES NJL module, Zenodo 10.5281/zenodo.18249033,
# which is what those papers are computed with -- exposes FOUR spellings of it
# under `RG_scheme`, and the closed form below is the one it calls 'analytic'.
# Its shipped default is 'minimal', which keeps only the leading logarithm
# g -> ln(Lambda_UV/Lambda); that agrees with the form below to 0.01% at
# Delta = 10 MeV but is 5% off at Delta = 250 MeV, and at eta_D = 1.45 it moves
# the 2SC window away entirely. A comparison against that module has to select
# 'analytic', not take its default.

#: The six Cooper pairs, as (eta, mode_i, mode_j) in the flavour-major mode
#: index j = 3 i_f + i_a. Delta_1 pairs d with s, Delta_2 pairs u with s and
#: Delta_3 pairs u with d; each gap has TWO pairs, the two ways of choosing the
#: colours epsilon leaves. This is Table I of Ruester et al., PRD 72, 034004
#: (2005), and it is the same partition `eos.general.pairing` block-diagonalises
#: the BdG matrix by -- (u_r, d_g, s_b) supply one member each to three pairs,
#: (u_g, d_r), (u_b, s_r) and (d_b, s_g) are the other three.
PAIRS = ((0, 4, 8), (0, 5, 7),      # Delta_1: (d_g, s_b), (d_b, s_g)
         (1, 0, 8), (1, 2, 6),      # Delta_2: (u_r, s_b), (u_b, s_r)
         (2, 0, 4), (2, 1, 3))      # Delta_3: (u_r, d_g), (u_g, d_r)

#: Geometric refinement of the pairing panels above the Fermi momenta, used
#: only at lambda > 1 where the cutoff is a decade clear of them. Halving is
#: enough: it already reaches round-off, and a finer ratio only buys nodes.
RG_PANEL_RATIO = 2.0


def counterterm_shape(Delta, Lambda, Lambda_UV):
    """g(Delta), the dimensionless shape of the massless counterterm.

        g = Lambda / sqrt(Lambda^2 + Delta^2)
          - Lambda_UV / sqrt(Lambda_UV^2 + Delta^2)
          + ln[ (Lambda_UV + sqrt(Lambda_UV^2 + Delta^2))
              / (Lambda    + sqrt(Lambda^2    + Delta^2)) ]

    Two limits carry the whole design. `g -> ln(Lambda_UV/Lambda)` for large
    Lambda_UV, which is the logarithm the medium divergence needs cancelled;
    and **g = 0 identically at Lambda_UV = Lambda**, so lambda = 1 recovers
    conventional sharp-cutoff regularization with no counterterm and no branch
    to select it. That is why `lambda_UV` is the regularization switch and
    there is no second flag beside it.
    """
    hi = math.hypot(Lambda_UV, Delta)
    lo = math.hypot(Lambda, Delta)
    return Lambda / lo - Lambda_UV / hi + math.log((Lambda_UV + hi)
                                                   / (Lambda + lo))


def counterterm(par, Delta, mu_star):
    """The counterterm block: (omega, n, gap).

    `omega` [MeV^4] is its contribution to Omega,

        omega = (1/pi^2) sum_pairs mubar_ij^2 Delta_eta^2 g(Delta_eta)

    with `mubar_ij = (mu*_i + mu*_j)/2` the mean EFFECTIVE potential of the
    pair -- effective because it is the potential that appears in the
    dispersion relations the counterterm was derived from.

    `n` [MeV^3] is the density it carries, n_j = -dOmega/dmu*_j, which every
    conserved charge and both neutrality conditions are then built from; and
    `gap` [MeV^3] is d(omega)/d(Delta_eta), which the gap equation needs. A
    counterterm added to Omega alone would leave Euler violated and the gaps
    solving the wrong equation, and both failures look like a plausible EoS.

    It carries no explicit T and no mass -- the massless scheme sets M = 0 --
    so it contributes to neither the entropy nor the scalar density.
    """
    omega = 0.0
    n = np.zeros(N_MODES)
    gap = np.zeros(3)
    if not np.any(Delta):
        return omega, n, gap

    for eta, i, j in PAIRS:
        D = float(Delta[eta])
        if D == 0.0:
            continue
        mubar = 0.5 * (mu_star[i] + mu_star[j])
        g = counterterm_shape(D, par.Lambda, par.Lambda_medium)

        omega += mubar ** 2 * D ** 2 * g / math.pi ** 2
        # -d/dmu*_i and -d/dmu*_j, with d(mubar)/d(mu*) = 1/2 on each
        share = mubar * D ** 2 * g / math.pi ** 2
        n[i] -= share
        n[j] -= share
        gap[eta] += mubar ** 2 * _shape_gap_derivative(
            D, par.Lambda, par.Lambda_medium) / math.pi ** 2
    return omega, n, gap


@lru_cache(maxsize=512)
def _vacuum_pair_block(M_bytes, Delta_bytes, k_max, nodes_per_panel, backend):
    """The pairing block at mu* = 0, T = 0, memoized on (M, Delta).

    The vacuum half of the RG split depends on the masses and the gaps and on
    NOTHING else -- not the potentials, not mu_3 or mu_8, not Sigma_V, not T.
    A numerically differenced Jacobian perturbs one unknown at a time, so every
    column that moves a potential rather than a mass or a gap asks for a block
    that has just been computed. Measured hit rate on a 250-point 2SC sweep:
    41%, for a 1.26x speedup on the whole table.

    Keyed on the arrays' bytes because a cache key must be hashable, and the
    returned arrays are sealed read-only because the cache hands every caller
    the same object (CLAUDE.md section 6 allows exactly this: a read-only cache
    keyed by immutable parameters).
    """
    M = np.frombuffer(M_bytes, dtype=float)
    Delta = np.frombuffer(Delta_bytes, dtype=float)
    zero = np.zeros(N_MODES)
    block = pair_block(M, zero, Delta, 0.0, k_max, backend=backend,
                       quadrature=pair_nodes(M, zero, 0.0, k_max,
                                             nodes_per_panel,
                                             RG_PANEL_RATIO))
    for name in ("delta_n", "delta_rho_s", "gap_kernel"):
        getattr(block, name).flags.writeable = False
    return block


def rg_pair_block(par, M, mu_star, Delta, T,
                  nodes_per_panel=NODES_PER_PANEL, **kwargs):
    """The pairing block under the RG-consistent split.

    Eq. (42) of Gholami et al. regularizes the VACUUM at Lambda and the medium
    remainder `A - A_vac` at Lambda_UV -- and `A_vac(chi)` is the vacuum at the
    SAME condensates, gaps included. The Delta-dependent Dirac sea is therefore
    a vacuum quantity and keeps the vacuum cutoff; only what is left of the
    pairing correction runs to Lambda_UV:

        block = [ hot(Lambda_UV) - vac(Lambda_UV) ] + vac(Lambda)

    where `vac` is the same block at mu* = 0, T = 0. Omitting it does not cost
    accuracy, it costs the scheme: the remainder then diverges QUADRATICALLY in
    Lambda_UV rather than logarithmically, and no counterterm of the published
    form can cancel that. Measured at (M_s, Delta_3) = (250, 100) MeV, the
    subtraction turns a drift of -5.9e11 MeV^4 between lambda = 5 and 40 into a
    logarithm whose slope agrees with `counterterm` to 0.1%.

    The two vacuum blocks are nested, so their difference is formally one
    integral over the shell [Lambda, Lambda_UV] -- one pass instead of two, and
    96 nodes instead of 384. It is NOT taken that way: measured, the shell form
    agrees to 1.8e-16 in delta_omega and bit for bit in gap_kernel, yet leaves
    the mu_B = 1500 MeV, eta_D = 1 2SC solve stalled at a residual of 9.0e-9
    where the two nested passes reach 7.6e-15. The two rules have the SAME
    delta_rho_s accuracy (1.2e-9 either way), so what the nesting buys is not
    accuracy but a quadrature error that cancels between the two blocks, and
    the mass residual is what notices. Three passes is the price.

    At lambda = 1 the two vacuum blocks are the same integral and the whole
    expression collapses to `hot(Lambda)`, which is returned directly -- so
    conventional sharp-cutoff regularization costs exactly one quadrature pass
    and reproduces its numbers bit for bit.
    """
    if par.lambda_UV == 1.0 or not any(active_gaps(Delta)):
        return pair_block(M, mu_star, Delta, T, par.Lambda_medium,
                          nodes_per_panel=nodes_per_panel, **kwargs)

    # Lambda_UV sits a decade above every Fermi momentum, so the default panel
    # layout leaves the whole tail in ONE panel and mis-integrates its 1/p
    # decay by 4e-7 relative -- enough to make a warm-started table and a cold
    # point solve disagree past their convergence gate. The geometric panels
    # cost the same node count and bring it to 3e-13.
    rule = dict(nodes_per_panel=nodes_per_panel,
                max_panel_ratio=RG_PANEL_RATIO)
    # Delta goes into the rule as well: in a gapless state the T = 0
    # occupation steps at the momenta where a quasiparticle branch crosses
    # zero, and `pair_nodes` puts a breakpoint there only when told the gaps.
    hot = pair_block(M, mu_star, Delta, T, par.Lambda_medium, **kwargs,
                     quadrature=pair_nodes(M, mu_star, T, par.Lambda_medium,
                                           Delta=Delta, **rule))
    M_bytes = np.ascontiguousarray(M, dtype=float).tobytes()
    Delta_bytes = np.ascontiguousarray(Delta, dtype=float).tobytes()
    backend = kwargs.get("backend", "reference")
    hi = _vacuum_pair_block(M_bytes, Delta_bytes, par.Lambda_medium,
                            nodes_per_panel, backend)
    lo = _vacuum_pair_block(M_bytes, Delta_bytes, par.Lambda,
                            nodes_per_panel, backend)
    return replace(
        hot,
        delta_omega=hot.delta_omega - hi.delta_omega + lo.delta_omega,
        delta_n=hot.delta_n - hi.delta_n + lo.delta_n,
        delta_rho_s=hot.delta_rho_s - hi.delta_rho_s + lo.delta_rho_s,
        delta_s=hot.delta_s - hi.delta_s + lo.delta_s,
        gap_kernel=hot.gap_kernel - hi.gap_kernel + lo.gap_kernel,
    )


def _shape_gap_derivative(Delta, Lambda, Lambda_UV):
    """d/dDelta [ Delta^2 g(Delta) ] [MeV], for the gap equation.

    Written out rather than differenced because the gap equation is solved by
    Newton and a noisy row costs more than the algebra does; `test/njl`
    checks it against a central difference of `counterterm_shape`.
    """
    hi = math.hypot(Lambda_UV, Delta)
    lo = math.hypot(Lambda, Delta)
    return (3.0 * Delta * (Lambda / lo - Lambda_UV / hi)
            + 2.0 * Delta * math.log((Lambda_UV + hi) / (Lambda + lo))
            + Delta ** 3 * (Lambda_UV / hi ** 3 - Lambda / lo ** 3))


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

    MEMOIZED on (par, M0). The vacuum is a pure function of the parameters --
    no medium, no mode, no temperature -- but `solve` asks for it once per
    solved point, and at 1.475 ms it is 4.5 times one `state_at` and 12.8% of
    an unpaired table. `Parameters` is frozen and hashable, so this is the
    read-only cache keyed by immutable parameters that CLAUDE.md section 6
    allows, and it returns the SAME object rather than an equal one: the
    arrays are therefore made read-only, as `_gauss_legendre` does with its
    rule, so no caller can poison a vacuum every later point will read.
    """
    return _vacuum_solution(par, tuple(float(m) for m in M0))


@lru_cache(maxsize=32)
def _vacuum_solution(par, M0):
    """`vacuum_solution`'s cached core; `M0` is a tuple so it can be a key.

    The bound is what keeps an inference run -- which varies `par` every call
    and therefore misses every time -- from growing a cache it never reads.
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
    # The cache hands the SAME object to every caller, so the arrays are
    # sealed: an in-place write would otherwise reach every point solved after
    # it. Same protection, and same reason, as `_gauss_legendre`'s rule.
    M.flags.writeable = False
    phi.flags.writeable = False
    return Vacuum(M=M, phi=phi, Omega=Omega, eps=Omega, f_pi=f_pi(M[0], par.Lambda))


def bag_constant(par, vac=None):
    """The FULLY RESTORED bag [MeV^4]: Omega at the current masses minus Omega
    at the broken-phase ones.

    Omega at fixed masses, evaluated at the CURRENT (restored) masses minus at
    the broken-phase ones. It is a DERIVED quantity here, not an input the way
    a bag constant is in a bag model, and it is reported because the
    colour-dielectric companion model quotes its own B_g + B_chi against it:
    (228.93 MeV)^4 = 357.49 MeV/fm^3 at the RKH set. It is the quantity Kojo,
    Powell, Song and Baym call B_q^NJL, Phys. Rev. D 91, 045003 (2015)
    [arXiv:1412.1108] Eq. (28), who get 284 MeV/fm^3 = (219 MeV)^4 for the HK
    set by the same formula.

    THIS IS NOT THE `B_eff` OF THE NJL LITERATURE, and the names collide.
    There, B_eff = B_0 - B(n) is DENSITY DEPENDENT: B_0 is the constant this
    model subtracts at every point (`Vacuum.Omega`, up to sign) and B(n) is the
    same combination of sea and condensate energies evaluated at the IN-MEDIUM
    masses -- Logoteta, Bombaci, Providencia and Vidana, Phys. Rev. D 85,
    023003 (2012) [arXiv:1203.4159] Eqs. (20)-(21). For THIS parameter set
    their B_eff runs 90-195 MeV/fm^3 over the pressures a star reaches (their
    Fig. 3), against the 357.49 MeV/fm^3 here, because at those densities the
    strange condensate is still largely intact while this function melts all
    three down to the current masses. Comparing a surface energy against the
    number below rather than against theirs overstates the bag by a factor of
    two to four. Same disease, and the same fix, as `chi_diel` in
    `eos.ccdm`: two quantities, one name, so the name says which.
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
    n_B_nat: float
    n_C_nat: float
    n_S_nat: float
    n_3: float
    n_8: float
    Omega: float                        # vacuum-subtracted [MeV^4]
    P_nat: float                        # = -Omega [MeV^4]
    eps_nat: float                      # vacuum-subtracted [MeV^4]
    s_nat: float                        # [MeV^3]
    mu_dot_n: float                     # sum_j mu_j n_j [MeV^4]
    mass_residual: np.ndarray
    gap_residual: np.ndarray
    vector_residual: float
    pattern: str
    gapless: bool
    delta_omega: float                  # the pairing correction alone [MeV^4]
    pair_cost: float                    # sum_eta Delta^2/(4 G_D) [MeV^4]

    # --- the fm-based boundary ---------------------------------------------
    # The convention, and it is CLAUDE.md section 2's: a quantity that crosses
    # the fm-based boundary of section 5 carries the BARE name and is fm-based
    # (fm^-3, MeV/fm^3); its natural-units twin carries `_nat` (MeV^n). The
    # `_nat` suffix marks a TWIN, so a natural-units field with no fm partner
    # -- `n_q`, `n_3`, `n_8`, `Omega`, `mu_dot_n`, the mode and flavour arrays
    # -- keeps its bare name and its unit comment. That is not an exception:
    # this whole record is INTERNAL (the point holds it as `_state`), so what
    # is being named here is which quantities a caller reading through the
    # boundary would get, not a promise about every field.
    @property
    def n_B(self):
        return self.n_B_nat / hc3

    @property
    def P(self):
        return self.P_nat / hc3

    @property
    def eps(self):
        return self.eps_nat / hc3

    @property
    def s(self):
        return self.s_nat / hc3

    @property
    def n_C(self):
        return self.n_C_nat / hc3

    @property
    def n_S(self):
        return self.n_S_nat / hc3

    def euler_residual(self):
        """(eps + P - T s - sum_j mu_j n_j)/eps, the identity of section 8."""
        if self.eps_nat == 0.0:
            return 0.0
        return ((self.eps_nat + self.P_nat - self.T * self.s_nat
                 - self.mu_dot_n) / self.eps_nat)


def state_at(par, M, Delta, Sigma_V, mu_B, mu_C, mu_S, mu_3, mu_8, T,
             vac=None, pattern="unpaired", pair_nodes_per_panel=None,
             two_flavour=False, backend="reference"):
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
    if par.lambda_UV < 1.0:
        raise ValueError(
            f"eos.njl needs lambda = Lambda_UV/Lambda >= 1; got "
            f"{par.lambda_UV}. The medium is integrated to Lambda_UV and the "
            f"Dirac sea to Lambda, so lambda < 1 would cut the medium BELOW "
            f"the vacuum -- not a regularization anyone has defined. "
            f"lambda = 1 is conventional sharp-cutoff regularization")

    M = np.asarray(M, dtype=float)
    Delta = np.asarray(Delta, dtype=float)
    if vac is None:
        vac = vacuum_solution(par)
    if two_flavour and (pattern_mask(pattern)[0] or pattern_mask(pattern)[1]):
        raise NotImplementedError(
            f"eos.njl: pattern {pattern!r} condenses a diquark containing an "
            f"s quark (Delta_1 pairs d-s, Delta_2 pairs u-s), which is not a "
            f"state two-flavour matter has -- with the strange sector off "
            f"there is no s quark to pair. The patterns that survive "
            f"two_flavour=True are 'unpaired' and '2SC', the u-d condensate; "
            f"the flag keeps both its values there and is a statement about "
            f"the phase in the flavour-locked ones, exactly as "
            f"eos.alphabag.SpeciesFlags.gluons is")

    mu_modes = mode_potentials(mu_B, mu_C, mu_S, mu_3, mu_8)
    mu_star = mu_modes - Sigma_V
    M_mode = M[FLAVOUR_OF_MODE]

    # --- the nine modes as cut Fermi gases -------------------------------
    # WITH THE STRANGE SECTOR OFF THE THREE s MODES CARRY NO MEDIUM, and only
    # the medium: the DIRAC SEA of the s quark stays, and so does phi_s and
    # the gap equation that determines M_s. That is the physics of two-flavour
    # quark matter -- the s Fermi sea is empty, the s condensate of the QCD
    # vacuum is not -- and it is why the flag does not touch
    # `masses_from_condensates`. Dropping phi_s from the 't Hooft determinant
    # instead would move M_u and M_d and the subtracted vacuum constant, which
    # would change the MODEL rather than the matter content it is asked about.
    if backend == "fast":
        # The SAME nine integrals, compiled: `backends/kernel_numba` writes out
        # the quadrature `kinetic_thermo` performs, and verify/ checks the two
        # against each other. It is opt-in rather than the default because the
        # two flavours sum the modes in different orders and so agree to
        # round-off rather than bit for bit, and a pattern enumeration decided
        # by a free-energy comparison can be knife-edge: at n_B = 1.2 fm^-3 the
        # CFL and 2SC candidates are close enough that a 1e-16 difference
        # selects the other one. That is a property of the physics at a
        # near-degenerate boundary, not of either flavour, and it is why
        # `test/baseline` is frozen against the reference (CLAUDE.md section 9:
        # the reference is what correctness is judged against and is never
        # bypassed).
        if not NUMBA_OK:
            raise NotImplementedError(
                "eos.njl backend='fast' needs eos/njl/backends/kernel_numba "
                "and numba; neither is required for the model, and "
                "backend='reference' computes the same numbers more slowly")
        absent = np.array([bool(two_flavour and FLAVOUR_OF_MODE[j] == 2)
                           for j in range(N_MODES)])
        block = modes_thermo(mu_star, M_mode, T, par.Lambda_medium,
                             DEGENERACY, _GAUSS_X, _GAUSS_W, absent)
        n_med = block[:, 0]
        P_med = float(np.sum(block[:, 3]))
        eps_med = float(np.sum(block[:, 2]))
        s_med = float(np.sum(block[:, 4]))
        rho_s_med = np.array([float(np.sum(block[FLAVOUR_OF_MODE == i, 1]))
                              for i in range(3)])
    elif backend != "reference":
        raise ValueError(f"unknown backend {backend!r}; eos.njl has "
                         f"'reference' and 'fast'")
    else:
        modes = [(ABSENT if two_flavour and FLAVOUR_OF_MODE[j] == 2
                  else kinetic_thermo(mu_star[j], M_mode[j], T,
                                      par.Lambda_medium))
                 for j in range(N_MODES)]
        n_med = np.array([m.n for m in modes])
        P_med = sum(m.P for m in modes)
        eps_med = sum(m.eps for m in modes)
        s_med = sum(m.s for m in modes)
        rho_s_med = np.array([sum(modes[j].rho_s
                                  for j in range(N_MODES)
                                  if FLAVOUR_OF_MODE[j] == i)
                              for i in range(3)])

    # --- the pairing correction ------------------------------------------
    kwargs = ({} if pair_nodes_per_panel is None
              else {"nodes_per_panel": pair_nodes_per_panel})
    block = rg_pair_block(par, M, mu_star, Delta, T,
                          backend=backend, **kwargs)
    ct_omega, ct_n, ct_gap = counterterm(par, Delta, mu_star)
    n_modes = n_med + block.delta_n + ct_n
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
    # The counterterm enters eps by the same route as the pairing block --
    # eps = Omega + T s + sum_i mu*_i n_i, term by term -- but with no T of
    # its own, so it brings its Omega and the density it carries and nothing
    # else. Dropping either half leaves Euler violated at the 1e-3 level,
    # which is small enough to read as quadrature error.
    eps_ct = ct_omega + float(np.dot(mu_star, ct_n))
    Omega = (-P_med - sea + C - (Sigma_V * n_q - W)
             + block.delta_omega + D + ct_omega - vac.Omega)
    eps = eps_med - sea + C + W + eps_pair + eps_ct - vac.eps

    n_3, n_8 = colour_densities(n_modes)
    return NJLState(
        T=T, M=M, Delta=Delta, Sigma_V=Sigma_V,
        mu_B=mu_B, mu_C=mu_C, mu_S=mu_S, mu_3=mu_3, mu_8=mu_8,
        mu_modes=mu_modes, mu_star=mu_star, phi=phi,
        n_modes=n_modes, n_flavour=n_flavour, n_q=n_q, n_B_nat=n_q / 3.0,
        n_C_nat=float(np.dot(CHARGE, n_flavour)),
        n_S_nat=float(np.dot(STRANGENESS, n_flavour)), n_3=n_3, n_8=n_8,
        Omega=Omega, P_nat=-Omega, eps_nat=eps, s_nat=s,
        mu_dot_n=float(np.dot(mu_modes, n_modes)),
        mass_residual=M - masses_from_condensates(par, phi),
        gap_residual=Delta / (2.0 * par.G_D) - block.gap_kernel + ct_gap,
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


def gap_seed_scale(mu_q):
    """The cold-start size of a diquark gap [MeV], from the quark potential.

        gap_scale = max(0.35 mu_q, 50)

    One rule, read by every cold start in the model, because a seed is only
    ever wrong in one direction here: the gap equation has a trivial root at
    Delta = 0 as well as the physical one, the residual between them is nearly
    FLAT, and a Newton step off a flat residual overshoots and falls back onto
    zero. A gap that is silently zero reads as an unpaired phase rather than as
    a failure, so the seed is set above the physical gap rather than below it.

    0.35 was measured, not chosen: over eta_D in (0.75, 1.0, 1.45), lambda in
    (1, 10), mu_B in (1100, 1400, 1800) MeV and both the 2SC and CFL patterns,
    the old 0.1 mu_q collapsed or failed on 20 of 36 cold starts and 0.35 mu_q
    on 3. RG-consistent gaps run well above sharp-cutoff ones -- 199 MeV
    against 151 at eta_D = 1, mu_B = 1400 -- which is what moved the rule.
    """
    return max(0.35 * mu_q, 50.0)


def default_internal_guess(par, pattern, mu_B, T, vac=None, gap_scale=None):
    """A cold start for `thermo_from_mu`: masses from the vacuum, gaps seeded
    by the pattern.

    The mass seed interpolates from the broken vacuum towards the current
    masses as mu_B rises, which is the direction the physics moves; the gap
    seed is the pattern's own (see `eos.njl.species.PATTERNS`), scaled by
    `gap_scale`, whose default is `gap_seed_scale` -- the one rule every cold
    start in this model takes its gaps from.
    """
    from eos.njl.species import pattern_seed
    if vac is None:
        vac = vacuum_solution(par)
    x = min(max((mu_B / 3.0) / 400.0, 0.0), 1.0)
    m = np.array(par.current_masses, dtype=float)
    M = vac.M * (1.0 - x) + m * x
    if gap_scale is None:
        gap_scale = gap_seed_scale(mu_B / 3.0)

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


def internal_residual(x, par, mu_B, mu_C, mu_S, T, vac, pattern,
                      backend="reference"):
    """The rows `thermo_from_mu` drives to zero, in the order it assembles them.

        three gap equations         M_f - [m_f - 4 G_S phi_f + 2 K phi_g phi_h]
        one per free gap            Delta_eta/(2 G_D) - kernel_eta
        colour neutrality           n_3 = 0 ,  n_8 = 0        (paired only)
        the vector self-energy      Sigma_V - dW/dn_q         (if there is one)
    """
    M, Delta, mu_3, mu_8, Sigma_V = _unpack_internal(x, par, pattern)
    st = state_at(par, M, Delta, Sigma_V, mu_B, mu_C, mu_S, mu_3, mu_8, T,
                  vac=vac, pattern=pattern, backend=backend)
    mask = pattern_mask(pattern)
    rows = list(st.mass_residual)
    rows += [st.gap_residual[eta] for eta in range(3) if mask[eta]]
    if any(mask):
        rows += [st.n_3, st.n_8]
    if has_vector(par):
        rows.append(st.vector_residual)
    return rows


def thermo_from_mu(par, mu_B, mu_C=0.0, mu_S=0.0, T=0.0, pattern="unpaired",
                   x0=None, vac=None, return_state=False,
                   backend="reference"):
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
        raw = internal_residual(x, par, mu_B, mu_C, mu_S, T, vac, pattern,
                                backend=backend)
        return [r / s for r, s in zip(raw, internal_scales(x, par, pattern,
                                                           mu_B))]

    def unit_scales(x):
        return [1.0] * len(internal_unknowns(par, pattern))

    x, err, ok = solve_system(residual, np.asarray(x0, dtype=float),
                              unit_scales, tol=1.0e-13)

    M, Delta, mu_3, mu_8, Sigma_V = _unpack_internal(x, par, pattern)
    st = state_at(par, M, Delta, Sigma_V, mu_B, mu_C, mu_S, mu_3, mu_8, T,
                  vac=vac, pattern=pattern, backend=backend)
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
