"""Colour pairing: the gap matrix, the BdG problem, and what a diquark
condensate does to Omega, to the densities, to the entropy and to itself.

Two models in this repository condense diquarks -- the three-flavour NJL of
`eos.njl` and the chiral colour-dielectric model of `eos.ccdm` -- and the
pairing sector of both is the SAME sector. Same nine colour-flavour modes,
same 9x9 gap matrix, same 18x18 Bogoliubov-de Gennes problem, same correction
form for the pairing piece of the thermodynamic potential, same
Hellmann-Feynman kernels for the gap equations. So it is written once here
(CLAUDE.md section 7), as pure functions of

    (M_star[3], mu_star[9], Delta[3], T, k_max)

with no model knowledge in them at all: each model feeds its own effective
masses -- gap-equation masses in NJL, dielectric-dressed medium masses in
CCDM -- its own effective potentials, its own diquark coupling and its own
cutoff, and reads back the same block.

Conventions
-----------
Nine modes j = (f, a), flavour-major, j = 3 i_f + i_a, with f in (u, d, s) and
colour a in (r, g, b). The three gaps Delta_eta, eta = 1, 2, 3, pair the two
flavours and the two colours that eta is NOT: Delta_1 pairs d with s,
Delta_2 pairs u with s, Delta_3 pairs u with d (the 2SC gap). Colour
generators are

    (T_3)_(r,g,b) = (+1/2, -1/2, 0)      (T_8)_(r,g,b) = (+1/3, +1/3, -2/3)

so T_8 = lambda_8/sqrt(3). THREE normalisations of T_8 are in circulation and
mixing them corrupts mu_8 by factors of 1.15 to 1.7: Ruester et al.,
Pagliara-Schaffner-Bielich and Kunkel et al. use the halved Gell-Mann
diag(1/2, 1/2, -1), for which mu_8^theirs = (2/sqrt(3)) mu_8^ours, and
Buballa and Steiner-Reddy-Prakash use the full lambda_8, for which
mu_8^ours = sqrt(3) mu_8^theirs. Every mu_8 this module produces or consumes is
in the diag(1,1,-2)/3 normalisation, and a comparison with a paper has to
convert.

Units are natural throughout: momenta, masses and potentials in MeV,
densities in MeV^3, Omega in MeV^4. Nothing here converts to fm; the models'
own boundaries do that.

What is easy to get wrong
-------------------------
Four things, each of which returns a plausible-looking wrong answer:

  * the pairing potential is written as a CORRECTION, a difference from the
    unpaired spectrum, so that it vanishes identically -- not merely to
    quadrature accuracy -- when Delta = 0. `pair_block` asserts that;

  * the gap kernel is NOT Delta/|E|. That form is wrong by a factor 12 in the
    gapless window, where one quasiparticle branch has gone negative and the
    two branches cancel. `pair_block` differentiates the BdG eigenvalues by
    Hellmann-Feynman instead, which carries the branch sign automatically
    because the sorted non-negative eigenvalue IS |E| and its derivative
    therefore IS sign(E) dE/dDelta;

  * paired-mode densities and entropies are NOT the unpaired Fermi integrals.
    At a 2SC point with Delta = 80 MeV the unpaired density formula is wrong
    by -21% on the paired u modes and +12% on the paired d modes, and at
    T = 5 MeV the unpaired entropy is four orders of magnitude too large;

  * the |xi_j| subtraction kinks at each of the nine Fermi momenta, and one
    quadrature panel cannot resolve nine kinks. `pair_nodes` splits there.

Antiparticle branches are not optional either: at T = 0 they contribute 8.8%
of the pairing potential at Lambda = 600 MeV and 17.1% at Lambda = 1000 MeV.

Reading order: the mode bookkeeping, the gap matrix, the BdG problem, the
2SC closed form, the quadrature, the one pass that computes everything, and
the gap-root scan.

References
----------
Alford, Schmitt, Rajagopal, Schaefer, Rev. Mod. Phys. 80, 1455 (2008)
    [arXiv:0709.4635] -- the review; the gap matrix and the patterns.
Ruester, Werth, Buballa, Shovkovy, Rischke, Phys. Rev. D 72, 034004 (2005)
    [arXiv:hep-ph/0503184] -- the neutral three-flavour phase diagram.
Steiner, Reddy, Prakash, Phys. Rev. D 66, 094007 (2002)
    [arXiv:hep-ph/0205201] -- colour neutrality and mu_8.
Buballa, Phys. Rept. 407, 205 (2005) [arXiv:hep-ph/0402234].
"""
from dataclasses import dataclass

import numpy as np

#: The three light flavours and the three colours, in the order every array
#: here is indexed by.
FLAVOURS = ("u", "d", "s")
COLOURS = ("r", "g", "b")

#: Nine colour-flavour modes, flavour-major: j = 3 i_f + i_a.
MODES = tuple((f, a) for f in FLAVOURS for a in COLOURS)
N_MODES = len(MODES)

#: The flavour index of each mode, for expanding a per-flavour quantity
#: (a mass, a scalar density) over the nine modes.
FLAVOUR_OF_MODE = np.array([FLAVOURS.index(f) for f, _ in MODES])
COLOUR_OF_MODE = np.array([COLOURS.index(a) for _, a in MODES])

#: Electric charge and strangeness per flavour. S = +1 per s quark, the
#: OPPOSITE of the PDG sign, used consistently throughout this repository
#: (CLAUDE.md section 2).
CHARGE = np.array([2.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0])
STRANGENESS = np.array([0.0, 0.0, 1.0])

#: Colour generators in the diag(1, 1, -2)/3 normalisation of the module
#: docstring, per colour (r, g, b).
T3 = np.array([0.5, -0.5, 0.0])
T8 = np.array([1.0 / 3.0, 1.0 / 3.0, -2.0 / 3.0])


def mode_potentials(mu_B, mu_C=0.0, mu_S=0.0, mu_3=0.0, mu_8=0.0):
    """The nine mode potentials mu_(f,a) [MeV], in MODES order.

        mu_(f,a) = mu_B/3 + q_f mu_C + s_f mu_S + (T_3)_a mu_3 + (T_8)_a mu_8

    The flavour part is `eos.general.basis.quark_potentials` written over the
    nine modes; the colour part exists only where pairing does, which is why
    it lives here. Both colour potentials are ZERO in an unpaired region:
    n_3 and n_8 vanish there identically at mu_3 = mu_8 = 0 and the solver
    must be told so rather than left to hunt for them.
    """
    flavour = (mu_B / 3.0 + CHARGE * mu_C + STRANGENESS * mu_S)[FLAVOUR_OF_MODE]
    colour = (T3 * mu_3 + T8 * mu_8)[COLOUR_OF_MODE]
    return flavour + colour


def colour_densities(n_modes):
    """(n_3, n_8) from the nine mode densities, as the neutrality rows use them.

        n_3 = sum_f (n_(f,r) - n_(f,g))
        n_8 = sum_f (n_(f,r) + n_(f,g) - 2 n_(f,b))

    These are the generator densities up to the constant factors 1/2 and 1/3
    of T_3 and T_8; a row that must vanish does not care about its own
    normalisation, and this is the form the literature states.
    """
    n = np.asarray(n_modes, dtype=float).reshape(3, 3)
    n_3 = float(np.sum(n[:, 0] - n[:, 1]))
    n_8 = float(np.sum(n[:, 0] + n[:, 1] - 2.0 * n[:, 2]))
    return n_3, n_8


# =============================================================================
# THE GAP MATRIX
# =============================================================================
def _levi_civita():
    eps = np.zeros((3, 3, 3))
    eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1.0
    eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1.0
    return eps


def _basis_matrices():
    """The three 9x9 matrices B_eta with G = sum_eta Delta_eta B_eta.

        (B_eta)_((f a), (g b)) = eps^(a b eta) eps_(f g eta)

    Assembled once at import: they are three constant matrices and rebuilding
    them inside a quadrature loop is the sort of cost that hides.
    """
    eps = _levi_civita()
    B = np.zeros((3, N_MODES, N_MODES))
    for eta in range(3):
        for j, (f, a) in enumerate(MODES):
            for k, (g, b) in enumerate(MODES):
                B[eta, j, k] = (eps[COLOURS.index(a), COLOURS.index(b), eta]
                                * eps[FLAVOURS.index(f), FLAVOURS.index(g), eta])
    return B


#: B_eta, eta = 1, 2, 3, in the module's mode order.
B_ETA = _basis_matrices()


def gap_matrix(Delta):
    """The 9x9 gap matrix G = sum_eta Delta_eta B_eta [MeV].

    Its eigenvalue multiplicities are a DERIVED property of the pattern and
    must never be assigned by hand. At Delta_0 = 60 MeV they come out as

        unpaired (0,0,0)   0 (x9)
        2SC      (0,0,D)   -60 (x2), 0 (x5), +60 (x2)
        CFL      (D,D,D)   -60 (x5), +60 (x3), +120 (x1)
        uSC      (0,D,D)   +-84.85 (x1 each), +-60 (x2 each), 0 (x3)
        dSC      (D,0,D)   the same spectrum as uSC

    and with independent gaps the +-sqrt(2) Delta eigenvalue of uSC/dSC
    generalises to +-sqrt(Delta_2^2 + Delta_3^2). The matrix is symmetric with
    identically zero diagonal: a gap mixes particles with holes, it does not
    shift a quasiparticle energy the way a mass does.
    """
    Delta = np.asarray(Delta, dtype=float)
    return np.einsum("e,eij->ij", Delta, B_ETA)


# =============================================================================
# THE BOGOLIUBOV-DE GENNES PROBLEM
# =============================================================================
def bdg_matrix(xi, G):
    """The 18x18 BdG matrix [[diag(xi), G], [G, -diag(xi)]] [MeV].

    `xi` may be one vector of nine, or a stack of them with shape (..., 9);
    the result carries the same leading shape, which is what lets one
    quadrature pass diagonalise every node at once.
    """
    xi = np.asarray(xi, dtype=float)
    lead = xi.shape[:-1]
    H = np.zeros(lead + (2 * N_MODES, 2 * N_MODES))
    idx = np.arange(N_MODES)
    H[..., idx, idx] = xi
    H[..., idx + N_MODES, idx + N_MODES] = -xi
    H[..., :N_MODES, N_MODES:] = G
    H[..., N_MODES:, :N_MODES] = G
    return H


def bdg_eigh(xi, G):
    """(E, V): the nine quasiparticle energies and their eigenvectors.

    `E` is the NON-NEGATIVE HALF OF THE SIGNED SPECTRUM, `sort(eigvalsh)[9:]`
    -- not the nine largest in modulus. The two prescriptions agree in value,
    because the spectrum comes in +-pairs, but only the first is a smooth
    function of the parameters through a gapless window, where a branch
    crosses zero and its partner crosses back. That smoothness is what makes
    the Hellmann-Feynman derivatives below carry the correct branch sign
    without any sign bookkeeping: the sorted eigenvalue is |E| and its
    derivative is therefore sign(E) dE/dx.

    `V[..., :, a]` is the eigenvector of `E[..., a]`, in the doubled
    (particle, hole) basis: the top nine components are the particle
    amplitudes, the bottom nine the hole amplitudes.
    """
    w, v = np.linalg.eigh(bdg_matrix(xi, G))
    return w[..., N_MODES:], v[..., :, N_MODES:]


def bdg_energies(xi, G):
    """The nine quasiparticle energies alone (see `bdg_eigh`) [MeV]."""
    w = np.linalg.eigvalsh(bdg_matrix(xi, G))
    return w[..., N_MODES:]


def twosc_dispersion(E_u, E_d, mu_u, mu_d, Delta):
    """The 2SC quasiparticle energies in closed form [MeV].

        E^+- = sqrt((Ebar - mubar)^2 + Delta^2) +- [ (E_d - E_u)/2 - dmu ]
        Ebar = (E_u + E_d)/2,  mubar = (mu_u + mu_d)/2,  dmu = (mu_d - mu_u)/2

    Valid ONLY for the 2SC pattern -- one gap, two flavours, two colours --
    where the gap matrix and the mass matrix happen to commute within the
    paired block. For a general pattern at unequal masses they do not:
    ||[G, M]||_F is 7.4e4 at M = (40, 45, 480) MeV against exactly zero at
    equal masses, so there is no closed-form dispersion and `bdg_eigh` is the
    only route.

    Kept for two reasons: it is a fast path for 2SC production runs, and it is
    the unit test of the general path, which reproduces it to 4.5e-13 MeV over
    random configurations of masses, potentials, gap and momentum.

    E^- may be NEGATIVE. That is the gapless window, not an error: the two
    branches then cancel in the gap kernel, which is the whole content of the
    Clogston-Chandrasekhar limit.
    """
    base = np.sqrt((0.5 * (E_u + E_d) - 0.5 * (mu_u + mu_d)) ** 2 + Delta ** 2)
    shift = 0.5 * (E_d - E_u) - 0.5 * (mu_d - mu_u)
    return base + shift, base - shift


# =============================================================================
# QUADRATURE
# =============================================================================
#: Gauss-Legendre nodes per panel. The panels do the work here, not the node
#: count: at T = 30 MeV, splitting at the nine Fermi momenta with 100 nodes
#: per panel reaches a relative error of 3e-14 where a single panel with 800
#: nodes reaches only 2e-7.
NODES_PER_PANEL = 24

#: How wide a thermal collar to place around each Fermi momentum, in units of
#: T. The Fermi function has fallen to e^-25 at the edge of it.
THERMAL_COLLAR = 25.0

#: Two breakpoints closer than this fraction of the cutoff are one breakpoint.
_BREAK_TOL = 1.0e-9


def pair_nodes(M_star, mu_star, T, k_max, nodes_per_panel=NODES_PER_PANEL):
    """(k, w): a panel-split Gauss-Legendre rule on [0, k_max] [MeV].

    Breakpoints at each of the nine Fermi momenta k_F,j = sqrt(mu*_j^2 - M_f^2)
    and, at T > 0, at k_F,j +- 25 T. The |xi_j| subtraction in the pairing
    potential kinks at every one of them.

    THE CUTOFF IS THE PANEL LIMIT, imposed before the panels are built. The
    tempting alternative -- build the breakpoints, then filter out the ones
    above the cutoff -- can delete the Fermi-surface break and revert silently
    to a single panel; that produced 1e-1 errors at T = 30 MeV during
    development of the specification this module implements.
    """
    M_star = np.asarray(M_star, dtype=float)[FLAVOUR_OF_MODE]
    mu_star = np.asarray(mu_star, dtype=float)
    inside = np.abs(mu_star) > M_star
    kF = np.sqrt(np.maximum(mu_star[inside] ** 2 - M_star[inside] ** 2, 0.0))
    return panel_nodes(kF, T, k_max, nodes_per_panel)


def panel_nodes(fermi_momenta, T, k_max, nodes_per_panel=NODES_PER_PANEL):
    """(k, w): a panel-split Gauss-Legendre rule on [0, k_max] [MeV].

    The general helper `pair_nodes` and the models' own medium integrals share,
    since a cut Fermi integral has the same two problems wherever it appears:
    the integrand kinks at each Fermi momentum, and one panel cannot resolve a
    kink however many nodes it is given. Breakpoints go at each momentum in
    `fermi_momenta` and, at T > 0, at +- 25 T around each.
    """
    breaks = [0.0, k_max]
    for k_F in np.atleast_1d(fermi_momenta):
        breaks.append(float(k_F))
        if T > 0.0:
            breaks.append(float(k_F) - THERMAL_COLLAR * T)
            breaks.append(float(k_F) + THERMAL_COLLAR * T)

    edges = np.unique(np.clip(np.asarray(breaks, dtype=float), 0.0, k_max))
    edges = edges[np.concatenate(([True], np.diff(edges) > _BREAK_TOL * k_max))]

    x, wx = np.polynomial.legendre.leggauss(nodes_per_panel)
    lo, hi = edges[:-1, None], edges[1:, None]
    half = 0.5 * (hi - lo)
    k = (0.5 * (lo + hi) + half * x[None, :]).ravel()
    w = (half * wx[None, :]).ravel()
    return k, w


# =============================================================================
# THE PAIRING BLOCK: ONE QUADRATURE PASS
# =============================================================================
#: A quasiparticle branch closer to zero than this fraction of the largest gap
#: means the phase is gapless: a branch has crossed the Fermi surface and the
#: BCS blocking region has opened.
GAPLESS_FRACTION = 1.0e-3

@dataclass(frozen=True)
class PairBlock:
    """Everything the pairing correction contributes, from one quadrature pass.

    Every entry is a CORRECTION -- a difference between the paired spectrum
    and the unpaired one -- so every entry is identically zero at Delta = 0
    and a model adds them to its unpaired sums without a second code path.

    delta_omega   [MeV^4]  the pairing piece of Omega, WITHOUT the condensation
                           cost sum_eta Delta_eta^2/(4 G_D), which is the
                           model's own coupling and is added by the model.
    delta_n       [MeV^3]  per mode, in MODES order: n_j = n_j^unpaired + this.
    delta_rho_s   [MeV^3]  per flavour: the scalar density the gap equation
                           for M_f must include.
    delta_s       [MeV^3]  the entropy correction. At T = 5 MeV in a gapped
                           phase this cancels the unpaired entropy to four
                           significant figures; using the unpaired value there
                           is not approximately wrong but qualitatively so.
    gap_kernel    [MeV^3]  per gap eta: the Hellmann-Feynman integral that the
                           gap equation Delta_eta/(2 G_D) = kernel_eta balances.
    min_energy    [MeV]    the smallest quasiparticle energy on the grid.
    gapless       bool     whether a branch has reached zero (see
                           `GAPLESS_FRACTION`). A gapless solution is a real
                           physical state, but pattern ENUMERATION by comparing
                           Omega is not valid across one, so it is reported.
    """
    delta_omega: float
    delta_n: np.ndarray
    delta_rho_s: np.ndarray
    delta_s: float
    gap_kernel: np.ndarray
    min_energy: float
    gapless: bool


def _phi(x, T):
    """phi(x) = x + 2 T ln(1 + e^(-x/T)), the per-branch potential [MeV].

    Through `logaddexp`, never `log(1 + exp(...))`: the naive form overflows
    at x/T of a few hundred, which a T = 1 MeV point with a 300 MeV branch
    reaches immediately.
    """
    if T <= 0.0:
        return x
    return x + 2.0 * T * np.logaddexp(0.0, -x / T)


def _dphi(x, T):
    """phi'(x) = tanh(x / 2T), the occupation factor the derivatives carry."""
    if T <= 0.0:
        return np.sign(x)
    return np.tanh(0.5 * x / T)


def _dphi_dT(x, T):
    """d phi / d T at fixed x, which is minus the entropy per branch."""
    if T <= 0.0:
        return np.zeros_like(x)
    z = x / T
    # 2z/(e^z + 1) written through tanh, which is stable at every z; the
    # naive exponential overflows at z of a few hundred, which a T = 1 MeV
    # point with a 300 MeV branch reaches immediately.
    return 2.0 * np.logaddexp(0.0, -z) + z * (1.0 - np.tanh(0.5 * z))


def pair_block(M_star, mu_star, Delta, T, k_max,
               nodes_per_panel=NODES_PER_PANEL, quadrature=None):
    """The pairing correction to Omega, n_j, rho_s,f, s and the gap equations.

    One quadrature pass, one batched diagonalisation, five results: computing
    them separately would diagonalise the same 18x18 matrices five times, and
    finite-differencing them instead was measured 40x slower and
    ill-conditioned enough to lose convergence.

    Parameters
    ----------
    M_star : (3,) effective masses per flavour [MeV]
    mu_star : (9,) effective mode potentials [MeV], in MODES order
    Delta : (3,) the three gaps [MeV]
    T : temperature [MeV]; T = 0 is a branch, not a limit -- in a fully gapped
        phase every Taylor coefficient in T vanishes, so an expansion must be
        bypassed there rather than corrected
    k_max : the pairing cutoff [MeV]
    quadrature : an explicit (k, w) rule, if the caller is holding one; by
        default `pair_nodes` builds it from the CURRENT masses and potentials,
        which move as an outer solve converges

    Both the particle branches (xi = E - mu*) and the ANTIPARTICLE branches
    (xi = E + mu*) are summed. The antiparticle piece is not a small
    correction and it grows with the cutoff: 8.8% of the particle piece at
    Lambda = 600 MeV, 17.1% at Lambda = 1000 MeV.
    """
    M_star = np.asarray(M_star, dtype=float)
    mu_star = np.asarray(mu_star, dtype=float)
    Delta = np.asarray(Delta, dtype=float)

    if not np.any(Delta):
        zero_n = np.zeros(N_MODES)
        return PairBlock(delta_omega=0.0, delta_n=zero_n,
                         delta_rho_s=np.zeros(3), delta_s=0.0,
                         gap_kernel=np.zeros(3), min_energy=0.0, gapless=False)

    if quadrature is None:
        quadrature = pair_nodes(M_star, mu_star, T, k_max, nodes_per_panel)
    k, w = quadrature
    weight = w * k ** 2

    G = gap_matrix(Delta)
    M_mode = M_star[FLAVOUR_OF_MODE]
    E_mode = np.sqrt(k[:, None] ** 2 + M_mode[None, :] ** 2)   # (nk, 9)

    omega = 0.0
    delta_n = np.zeros(N_MODES)
    delta_rho_s = np.zeros(3)
    delta_s = 0.0
    kernel = np.zeros(3)
    min_energy = np.inf

    for r in (+1.0, -1.0):                     # particles, then antiparticles
        xi = E_mode - r * mu_star[None, :]
        E, V = bdg_eigh(xi, G)                 # (nk, 9), (nk, 18, 9)
        top, bot = V[:, :N_MODES, :], V[:, N_MODES:, :]
        occ_top, occ_bot = top ** 2, bot ** 2  # the BdG matrix is real

        abs_xi = np.abs(xi)
        min_energy = min(min_energy, float(np.min(E)))

        # Omega: the correction form, a difference from the unpaired spectrum
        omega += -np.sum(weight[:, None] * (_phi(E, T) - _phi(abs_xi, T)))

        tanh_E = _dphi(E, T)                   # (nk, 9)
        tanh_xi = _dphi(xi, T)                 # signed: tanh(|xi|/2T) sign(xi)

        # densities: dxi_j/dmu_j = -r, so dH/dmu_j = -r (P_j + -P_j)
        dE_dmu = -r * np.einsum("kb,kjb->kj", tanh_E, occ_top - occ_bot)
        delta_n += np.sum(weight[:, None] * (dE_dmu + r * tanh_xi), axis=0)

        # scalar densities: dxi_j/dM_f = M_f/E_j on the modes of flavour f
        dxi_dM = M_mode[None, :] / E_mode                      # (nk, 9)
        dE_dM = np.einsum("kb,kj,kjb->kj", tanh_E, dxi_dM, occ_top - occ_bot)
        per_mode = np.sum(weight[:, None] * (dE_dM - dxi_dM * tanh_xi), axis=0)
        for i_f in range(3):
            delta_rho_s[i_f] += float(np.sum(per_mode[FLAVOUR_OF_MODE == i_f]))

        # entropy: s = -dOmega/dT at fixed spectrum
        delta_s += float(np.sum(weight[:, None]
                                * (_dphi_dT(E, T) - _dphi_dT(abs_xi, T))))

        # the gap equation kernels, Hellmann-Feynman on dH/dDelta_eta
        for eta in range(3):
            mix = 2.0 * np.einsum("kib,ij,kjb->kb", top, B_ETA[eta], bot)
            kernel[eta] += float(np.sum(weight[:, None] * tanh_E * mix))

    inv = 1.0 / (2.0 * np.pi ** 2)
    omega *= inv
    delta_n *= inv
    delta_rho_s *= -inv          # rho_s = +dOmega/dM, and the sign is in here
    delta_s *= inv
    kernel *= inv

    scale = float(np.max(np.abs(Delta)))
    return PairBlock(delta_omega=float(omega), delta_n=delta_n,
                     delta_rho_s=delta_rho_s, delta_s=float(delta_s),
                     gap_kernel=kernel, min_energy=float(min_energy),
                     gapless=bool(min_energy < GAPLESS_FRACTION * scale))


def delta_omega_pair(M_star, mu_star, Delta, T, k_max, **kwargs):
    """The pairing potential alone [MeV^4] (see `pair_block`).

    Identically zero at Delta = 0 -- exactly 0.0, not small -- which is what
    makes the unpaired phase a clean limit of the same code. In the clean
    weak-coupling limit it obeys the BCS logarithm,

        -delta_omega / [mu^2 Delta^2 (ln(2 Lambda / Delta) - 1/2)] -> 2/pi^2

    approached from below as Delta falls: 1.9955, 1.9948, 1.9941, 1.9933
    times 1/pi^2 at Delta = 2, 5, 10, 20 MeV.
    """
    return pair_block(M_star, mu_star, Delta, T, k_max, **kwargs).delta_omega


def gap_residuals(Delta, G_D, block):
    """The three gap equations, as residuals that must vanish [MeV^3].

        R_eta = Delta_eta / (2 G_D) - kernel_eta

    with `kernel_eta` the Hellmann-Feynman integral `pair_block` returns:

        kernel_eta = (1/2 pi^2) sum_(r=+-) int dk k^2
                     sum_a <V_a| [[0, B_eta], [B_eta, 0]] |V_a> tanh(E_a / 2T)

    NEVER the Delta/|E| form obtained by differentiating |E| as though every
    branch were positive. Against finite differences of delta_omega at
    mu_u = 400, mu_d = 500 MeV that form is wrong by a factor 12.0 at
    Delta = 40 MeV, 1.7 at 60 and 1.3 at 80, and it makes the gap GROW with
    the mismatch -- the opposite of the physics. The sign structure is what
    the gapless window is made of.
    """
    return np.asarray(Delta, dtype=float) / (2.0 * G_D) - block.gap_kernel


def gap_roots(residual, hi, n_scan=60, xtol=1.0e-10):
    """Every root of a one-dimensional gap equation on (0, hi], by scanning.

    With a mismatch between the paired Fermi surfaces R(Delta) has THREE
    roots: Delta = 0, a barrier maximum, and the physical BCS root. A fixed
    bracket handed to `brentq` returns whichever the bracket happens to
    contain, or fails, and neither failure is loud. So scan, then bracket each
    sign change; the caller compares the candidates by Omega.

    Delta = 0 is always a root and is NOT returned: it is the unpaired
    pattern, which is enumerated on its own.
    """
    from scipy.optimize import brentq

    grid = np.linspace(hi / n_scan, hi, n_scan)
    values = np.array([residual(d) for d in grid])
    roots = []
    for i in range(len(grid) - 1):
        if values[i] == 0.0:
            roots.append(float(grid[i]))
        elif values[i] * values[i + 1] < 0.0:
            roots.append(float(brentq(residual, grid[i], grid[i + 1],
                                      xtol=xtol)))
    return roots
