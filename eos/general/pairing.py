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

from eos.general.fermi_integrals import (
    NODES_PER_PANEL, THERMAL_COLLAR, panel_nodes,
)

# The compiled flavour of `pair_block` below follows the precedent of
# `eos.general.fermi_integrals`: the jitted twin lives beside the reference in
# the sector's one home, guarded so the module imports without numba. Without
# numba, backend='fast' falls back to the blocked-numpy diagonalisation, which
# is the same spectrum a little slower -- never a silent wrong answer.
try:
    from numba import njit
    _NUMBA_OK = True
except ImportError:                       # pragma: no cover - numba optional
    _NUMBA_OK = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]

        def deco(f):
            return f
        return deco

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


def _bdg_blocks():
    """The mode index sets the gap matrix never couples across.

    `(B_eta)_((f a),(g b)) = eps^(a b eta) eps_(f g eta)` vanishes unless the
    two modes differ in BOTH flavour and colour by the same epsilon, which
    partitions the nine modes into four groups that no gap connects:

        (ur, dg, sb)   the three colour-flavour-diagonal modes
        (ug, dr)  (ub, sr)  (db, sg)   three pairs

    The partition is a property of the BASIS matrices, not of the gaps, so it
    is the same for every pattern and for every value of Delta -- it is taken
    from the union of the three B_eta, which is the coarsest (and therefore
    always safe) grouping. `diag(xi)` is diagonal and so preserves it, and the
    18x18 BdG matrix is therefore exactly block-diagonal in one 6x6 and three
    4x4 blocks, in the doubled basis (modes, modes + 9).

    That is 6^3 + 3*4^3 = 408 flops against 18^3 = 5832, a factor 14.3, and it
    is EXACT rather than an approximation: `bdg_eigh(backend='fast')`
    reproduces the dense spectrum to round-off.
    """
    coupled = np.any(np.abs(B_ETA) > 0.0, axis=0)
    seen, blocks = set(), []
    for start in range(N_MODES):
        if start in seen:
            continue
        stack, group = [start], []
        while stack:
            j = stack.pop()
            if j in seen:
                continue
            seen.add(j)
            group.append(j)
            for k in range(N_MODES):
                if coupled[j, k] and k not in seen:
                    stack.append(k)
        blocks.append(np.array(sorted(group)))
    return tuple(blocks)


#: The four index sets of `_bdg_blocks`, assembled once at import.
BDG_BLOCKS = _bdg_blocks()


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


def bdg_eigh(xi, G, backend="reference"):
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

    backend='fast' diagonalises the four `BDG_BLOCKS` separately instead of
    the whole 18x18 at once. The decomposition is exact, so this is the same
    spectrum -- but the branches come back GROUPED BY BLOCK rather than sorted
    across all nine, and within a degenerate subspace the eigenvectors are a
    different orthonormal basis. Every consumer in `pair_block` contracts over
    the branch index, so neither difference reaches a result; nothing else may
    assume `E` is sorted.
    """
    if backend == "fast":
        return _bdg_eigh_blocked(xi, G)
    if backend != "reference":
        raise ValueError(f"unknown backend {backend!r}; eos.general.pairing "
                         f"has 'reference' and 'fast'")
    w, v = np.linalg.eigh(bdg_matrix(xi, G))
    return w[..., N_MODES:], v[..., :, N_MODES:]


def _bdg_eigh_blocked(xi, G):
    """`bdg_eigh` on the four `BDG_BLOCKS`, which is 14.3x fewer flops.

    Each block of `n` modes contributes a 2n x 2n BdG problem whose upper half
    of the spectrum is `n` branches; the four blocks supply 3 + 2 + 2 + 2 = 9
    between them. The eigenvectors are written back into the full doubled
    basis, so `V` has the same 18-row shape the dense path returns and a
    caller cannot tell which path produced it except by the branch ORDER.
    """
    xi = np.asarray(xi, dtype=float)
    lead = xi.shape[:-1]
    E = np.empty(lead + (N_MODES,))
    V = np.zeros(lead + (2 * N_MODES, N_MODES))

    filled = 0
    for idx in BDG_BLOCKS:
        n = idx.size
        sub = np.zeros(lead + (2 * n, 2 * n))
        rows = np.arange(n)
        sub[..., rows, rows] = xi[..., idx]
        sub[..., rows + n, rows + n] = -xi[..., idx]
        block_G = G[np.ix_(idx, idx)]
        sub[..., :n, n:] = block_G
        sub[..., n:, :n] = block_G

        w, v = np.linalg.eigh(sub)
        E[..., filled:filled + n] = w[..., n:]
        # back into the full basis: the block's particle rows are `idx` and
        # its hole rows are `idx + N_MODES`.
        V[..., idx, filled:filled + n] = v[..., :n, n:]
        V[..., idx + N_MODES, filled:filled + n] = v[..., n:, n:]
        filled += n
    return E, V


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
# The panel-split rule itself is `eos.general.fermi_integrals.panel_nodes`,
# imported at the top rather than repeated here: a cut Fermi integral has the
# same two problems wherever it appears -- the integrand kinks at each Fermi
# momentum, and one panel cannot resolve a kink however many nodes it is given
# -- and the Fermi integrals of this repository have one home (CLAUDE.md
# section 7). What belongs to PAIRING is which momenta to break at, which is
# `pair_nodes` below.


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


# =============================================================================
# THE COMPILED PASS
# =============================================================================
# The block structure of `_bdg_blocks`, taken one step further and written as
# loops numba can compile. The three 4x4 blocks decouple AGAIN: the gap couples
# the particle of one mode only to the hole of its partner, so each 4x4 is two
# independent 2x2 Bogoliubov problems,
#
#     A = [[xi_i, g], [g, -xi_j]]   on (particle_i, hole_j)
#     B = [[xi_j, g], [g, -xi_i]]   on (particle_j, hole_i)
#
# with closed-form eigenpairs (`twosc_dispersion` is the energy half of this
# statement). Only the (ur, dg, sb) 6x6 needs a numerical diagonalisation, by
# cyclic Jacobi -- robust at every degeneracy, unlike a cubic in closed form.
#
# ONE SELECTION RULE, stated once because it is the part that goes wrong
# quietly: `bdg_eigh` returns the non-negative member of each +- pair of the
# SIGNED spectrum. Within a 2x2 sub-block the pairs run ACROSS the two
# sub-blocks -- A's eigenvalues are m +- s and B's are -m +- s, pairing
# (m+s) with -(m+s) -- so the branch kept is |m + s| with A's upper
# eigenvector when m + s >= 0 and B's LOWER-partner vector when it is not,
# and likewise |m - s|. Inside a gapless window that is exactly the switch
# that keeps the Hellmann-Feynman derivatives carrying sign(E).

#: The mode index sets of the compiled pass, frozen from `BDG_BLOCKS`' own
#: derivation: the (ur, dg, sb) triple, and the (i, j) of the three pairs.
_BLOCK0 = np.array([0, 4, 8])
_PAIR_I = np.array([1, 2, 5])
_PAIR_J = np.array([3, 6, 7])
for _arr in (_BLOCK0, _PAIR_I, _PAIR_J):
    _arr.flags.writeable = False


@njit(cache=True)
def _phi_scalar(x, T):
    """phi(x) = x + 2 T ln(1 + e^(-x/T)), overflow-safe, scalar."""
    if T <= 0.0:
        return x
    z = -x / T
    if z > 0.0:
        log_term = z + np.log1p(np.exp(-z))
    else:
        log_term = np.log1p(np.exp(z))
    return x + 2.0 * T * log_term


@njit(cache=True)
def _dphi_scalar(x, T):
    """phi'(x) = tanh(x / 2T); sign(x) at T = 0."""
    if T <= 0.0:
        if x > 0.0:
            return 1.0
        if x < 0.0:
            return -1.0
        return 0.0
    return np.tanh(0.5 * x / T)


@njit(cache=True)
def _dphi_dT_scalar(x, T):
    """d phi / d T at fixed x; zero at T = 0."""
    if T <= 0.0:
        return 0.0
    z = x / T
    if -z > 0.0:
        log_term = -z + np.log1p(np.exp(z))
    else:
        log_term = np.log1p(np.exp(-z))
    return 2.0 * log_term + z * (1.0 - np.tanh(0.5 * z))


@njit(cache=True)
def _eig2_upper(a, b, c):
    """(lambda_max, u, v) of the symmetric [[a, c], [c, b]].

    c = 0 is an EXACT branch, not a small-norm fallback: a pattern with a
    zero gap channel reaches the diagonal case at every node, and there the
    formula vector (c, lambda - a) is (0, round-off) -- its direction is
    noise, and the noise picked the HOLE axis for a particle branch often
    enough to corrupt every occupation it touched. With c != 0 the vector is
    the larger-normed of the two analytic candidates (c, lambda - a) and
    (lambda - b, c), which is the standard conditioning trick: lambda sits
    within round-off of ONE diagonal exactly when the other candidate is
    order one.
    """
    m = 0.5 * (a + b)
    d = 0.5 * (a - b)
    s = np.sqrt(d * d + c * c)
    lam = m + s
    if c == 0.0:
        if a >= b:
            return lam, 1.0, 0.0
        return lam, 0.0, 1.0
    v1x = c
    v1y = lam - a
    v2x = lam - b
    v2y = c
    n1 = v1x * v1x + v1y * v1y
    n2 = v2x * v2x + v2y * v2y
    if n1 >= n2:
        norm = np.sqrt(n1)
        return lam, v1x / norm, v1y / norm
    norm = np.sqrt(n2)
    return lam, v2x / norm, v2y / norm


@njit(cache=True)
def _jacobi6(A, V):
    """Cyclic Jacobi on the symmetric 6x6 `A`, vectors into `V`. In place.

    Converges quadratically; a dozen sweeps is far more than the spectrum
    ever needs, and the threshold is relative to the matrix norm so a scale
    of 1e15 MeV (a confined effective mass) needs no special case.
    """
    for i in range(6):
        for j in range(6):
            V[i, j] = 1.0 if i == j else 0.0
    norm = 0.0
    for i in range(6):
        for j in range(6):
            norm += A[i, j] * A[i, j]
    if norm <= 0.0:
        return
    for _ in range(30):
        off = 0.0
        for p in range(5):
            for q in range(p + 1, 6):
                off += A[p, q] * A[p, q]
        if off <= 1.0e-30 * norm:
            break
        for p in range(5):
            for q in range(p + 1, 6):
                apq = A[p, q]
                if apq == 0.0:
                    continue
                theta = 0.5 * (A[q, q] - A[p, p]) / apq
                t = 1.0 / (abs(theta) + np.sqrt(theta * theta + 1.0))
                if theta < 0.0:
                    t = -t
                c = 1.0 / np.sqrt(t * t + 1.0)
                s = t * c
                for i in range(6):
                    aip = A[i, p]
                    aiq = A[i, q]
                    A[i, p] = c * aip - s * aiq
                    A[i, q] = s * aip + c * aiq
                for i in range(6):
                    api = A[p, i]
                    aqi = A[q, i]
                    A[p, i] = c * api - s * aqi
                    A[q, i] = s * api + c * aqi
                for i in range(6):
                    vip = V[i, p]
                    viq = V[i, q]
                    V[i, p] = c * vip - s * viq
                    V[i, q] = s * vip + c * viq


@njit(cache=True)
def _pair_pass(k, w, M_mode, mu_star, G, B_eta, T):
    """The whole of `pair_block`'s quadrature, one compiled pass.

    Returns the UNSCALED accumulators in the reference path's own convention
    -- (omega, delta_n[9], dM_per_mode[9], delta_s, kernel[3], min_energy) --
    so the wrapper applies the same 1/(2 pi^2) factors and flavour sums to
    both flavours and the two cannot drift apart in the bookkeeping.
    """
    nk = k.shape[0]
    omega = 0.0
    delta_n = np.zeros(9)
    dM_mode = np.zeros(9)
    delta_s = 0.0
    kernel = np.zeros(3)
    min_energy = np.inf

    A66 = np.empty((6, 6))
    V66 = np.empty((6, 6))
    lam6 = np.empty(6)
    order = np.empty(6, dtype=np.int64)
    E_node = np.empty(9)
    xi = np.empty(9)
    E_mode = np.empty(9)
    dxi_dM = np.empty(9)
    top = np.empty(9)
    bot = np.empty(9)

    for node in range(nk):
        kk = k[node]
        weight = w[node] * kk * kk
        for j in range(9):
            E_mode[j] = np.sqrt(kk * kk + M_mode[j] * M_mode[j])
            dxi_dM[j] = M_mode[j] / E_mode[j]

        for r in (1.0, -1.0):
            for j in range(9):
                xi[j] = E_mode[j] - r * mu_star[j]

            n_branch = 0
            # --- the (ur, dg, sb) 6x6 ------------------------------------
            for a in range(3):
                for b in range(3):
                    A66[a, b] = 0.0
                    A66[3 + a, 3 + b] = 0.0
                    A66[a, 3 + b] = G[_BLOCK0[a], _BLOCK0[b]]
                    A66[3 + a, b] = G[_BLOCK0[a], _BLOCK0[b]]
                A66[a, a] = xi[_BLOCK0[a]]
                A66[3 + a, 3 + a] = -xi[_BLOCK0[a]]
            _jacobi6(A66, V66)
            for i in range(6):
                lam6[i] = A66[i, i]
                order[i] = i
            # insertion sort, ascending: the upper half is the kept spectrum
            for i in range(1, 6):
                key = lam6[i]
                key_o = order[i]
                j6 = i - 1
                while j6 >= 0 and lam6[j6] > key:
                    lam6[j6 + 1] = lam6[j6]
                    order[j6 + 1] = order[j6]
                    j6 -= 1
                lam6[j6 + 1] = key
                order[j6 + 1] = key_o
            for pick in range(3, 6):
                E_b = lam6[pick]
                col = order[pick]
                for j in range(9):
                    top[j] = 0.0
                    bot[j] = 0.0
                for a in range(3):
                    top[_BLOCK0[a]] = V66[a, col]
                    bot[_BLOCK0[a]] = V66[3 + a, col]
                E_node[n_branch] = E_b
                _accumulate_branch(E_b, top, bot, xi, dxi_dM, r, T, weight,
                                   B_eta, delta_n, dM_mode, kernel)
                n_branch += 1

            # --- the three pairs, each two closed-form 2x2 problems ------
            for pair in range(3):
                i = _PAIR_I[pair]
                j = _PAIR_J[pair]
                g = G[i, j]
                lamA, uA, vA = _eig2_upper(xi[i], -xi[j], g)
                # the +- partner structure: A's eigenvalues are m +- s and
                # B's are -m +- s, so the non-negative member of each pair is
                # |m + s| and |m - s|, with the vector taken from whichever
                # sub-block carries that sign
                lamB, uB, vB = _eig2_upper(xi[j], -xi[i], g)
                # branch 1: |m + s| = |lamA|
                for jj in range(9):
                    top[jj] = 0.0
                    bot[jj] = 0.0
                if lamA >= 0.0:
                    E_b = lamA
                    top[i] = uA
                    bot[j] = vA
                else:
                    # the partner -(m+s) = B's lower; its vector is the
                    # orthogonal complement of B's upper
                    E_b = -lamA
                    top[j] = -vB
                    bot[i] = uB
                E_node[n_branch] = E_b
                _accumulate_branch(E_b, top, bot, xi, dxi_dM, r, T, weight,
                                   B_eta, delta_n, dM_mode, kernel)
                n_branch += 1
                # branch 2: |m - s|; m - s is A's lower, -(m - s) is B's upper
                for jj in range(9):
                    top[jj] = 0.0
                    bot[jj] = 0.0
                lamA_lo = xi[i] - xi[j] - lamA          # trace(A) - lamA
                if lamA_lo >= 0.0:
                    E_b = lamA_lo
                    top[i] = -vA
                    bot[j] = uA
                else:
                    E_b = lamB
                    top[j] = uB
                    bot[i] = vB
                E_node[n_branch] = E_b
                _accumulate_branch(E_b, top, bot, xi, dxi_dM, r, T, weight,
                                   B_eta, delta_n, dM_mode, kernel)
                n_branch += 1

            # --- the per-node scalars ------------------------------------
            for b in range(9):
                if E_node[b] < min_energy:
                    min_energy = E_node[b]
                omega -= weight * _phi_scalar(E_node[b], T)
                delta_s += weight * _dphi_dT_scalar(E_node[b], T)
            for j in range(9):
                a_xi = abs(xi[j])
                omega += weight * _phi_scalar(a_xi, T)
                delta_s -= weight * _dphi_dT_scalar(a_xi, T)
                tanh_xi = _dphi_scalar(xi[j], T)
                delta_n[j] += weight * r * tanh_xi
                dM_mode[j] -= weight * dxi_dM[j] * tanh_xi

    return omega, delta_n, dM_mode, delta_s, kernel, min_energy


@njit(cache=True)
def _accumulate_branch(E_b, top, bot, xi, dxi_dM, r, T, weight,
                       B_eta, delta_n, dM_mode, kernel):
    """One quasiparticle branch's contributions, Hellmann-Feynman throughout.

    The reference path contracts (nk, 9, 9) arrays; here the same sums run
    over the at most three nonzero components a branch has in the blocked
    basis. `delta_n` takes dE/dmu_j = -r (top_j^2 - bot_j^2) tanh(E/2T),
    `dM_mode` takes dE/dM through dxi/dM, and the gap kernels take
    2 top_i (B_eta)_ij bot_j -- the sparse handful of (i, j) pairs each
    B_eta actually couples.
    """
    tanh_E = _dphi_scalar(E_b, T)
    for j in range(9):
        occ = top[j] * top[j] - bot[j] * bot[j]
        if occ != 0.0:
            delta_n[j] += weight * (-r) * tanh_E * occ
            dM_mode[j] += weight * tanh_E * dxi_dM[j] * occ
    for eta in range(3):
        mix = 0.0
        for i in range(9):
            t_i = top[i]
            if t_i == 0.0:
                continue
            for j in range(9):
                if bot[j] != 0.0 and B_eta[eta, i, j] != 0.0:
                    mix += t_i * B_eta[eta, i, j] * bot[j]
        if mix != 0.0:
            kernel[eta] += weight * tanh_E * 2.0 * mix


def pair_block(M_star, mu_star, Delta, T, k_max,
               nodes_per_panel=NODES_PER_PANEL, quadrature=None,
               backend="reference"):
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
    backend : 'reference' (default) or 'fast', which selects how the BdG
        problem at each node is diagonalised -- see `bdg_eigh`. The two are the
        same spectrum computed two ways and agree to round-off, so the choice
        belongs to the caller that already declared one (CLAUDE.md section 9)

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

    G = gap_matrix(Delta)
    M_mode = M_star[FLAVOUR_OF_MODE]

    if backend == "fast" and _NUMBA_OK:
        # The whole pass compiled: closed-form 2x2 sub-blocks, a Jacobi 6x6,
        # and the contractions run over the handful of components a blocked
        # branch actually has. Without numba, 'fast' continues below through
        # `bdg_eigh(..., 'fast')`, the blocked-numpy diagonalisation -- the
        # same spectrum, never a silent fallback to a different answer.
        omega, delta_n, dM_mode, delta_s, kernel, min_energy = _pair_pass(
            np.ascontiguousarray(k, dtype=float),
            np.ascontiguousarray(w, dtype=float),
            np.ascontiguousarray(M_mode, dtype=float),
            np.ascontiguousarray(mu_star, dtype=float), G, B_ETA, float(T))
        inv = 1.0 / (2.0 * np.pi ** 2)
        delta_rho_s = np.array(
            [-inv * float(np.sum(dM_mode[FLAVOUR_OF_MODE == i]))
             for i in range(3)])
        scale = float(np.max(np.abs(Delta)))
        return PairBlock(delta_omega=float(omega) * inv,
                         delta_n=delta_n * inv,
                         delta_rho_s=delta_rho_s,
                         delta_s=float(delta_s) * inv,
                         gap_kernel=kernel * inv,
                         min_energy=float(min_energy),
                         gapless=bool(min_energy < GAPLESS_FRACTION * scale))

    weight = w * k ** 2
    E_mode = np.sqrt(k[:, None] ** 2 + M_mode[None, :] ** 2)   # (nk, 9)

    omega = 0.0
    delta_n = np.zeros(N_MODES)
    delta_rho_s = np.zeros(3)
    delta_s = 0.0
    kernel = np.zeros(3)
    min_energy = np.inf

    for r in (+1.0, -1.0):                     # particles, then antiparticles
        xi = E_mode - r * mu_star[None, :]
        E, V = bdg_eigh(xi, G, backend)        # (nk, 9), (nk, 18, 9)
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


# =============================================================================
# THE PAIRING PATTERNS
# =============================================================================
# Which of the three gaps a pattern makes free, and where a solve of it starts.
# A pattern is a DECLARATION, like a mode: it does not add code, it says which
# unknowns exist. The gap equation has three roots at any mismatch -- zero, a
# barrier maximum and the physical BCS root -- so which root a solve lands on
# is decided by the seed, and the seeds below are what make an enumeration an
# enumeration rather than one solve repeated.
#
# The mask says which Delta_eta are unknowns; the seed is in units of the gap
# scale the caller supplies. Recall that Delta_1 pairs d with s, Delta_2 pairs
# u with s and Delta_3 pairs u with d, so 2SC -- the pattern that survives a
# large strange-quark mass -- is the one with Delta_3 alone.
#
# 'free' carries the same freedom as CFL and differs only in seeding: started
# asymmetrically it can converge on uSC, dSC or an unequal-gap state that none
# of the named seeds would have found. That is the point of enumerating seeds
# rather than patterns.
#
# The table is here rather than in either model because both of them pair, and
# which gaps a named pattern makes free is a property of the gap matrix above,
# not of the Lagrangian that supplies G_D (CLAUDE.md section 7).

PATTERNS = {
    "unpaired": ((False, False, False), (0.0, 0.0, 0.0)),
    "2SC":      ((False, False, True),  (0.0, 0.0, 1.0)),
    "uSC":      ((False, True, True),   (0.0, 0.6, 1.0)),
    "dSC":      ((True, False, True),   (0.6, 0.0, 1.0)),
    "CFL":      ((True, True, True),    (1.0, 1.0, 1.0)),
    "free":     ((True, True, True),    (0.3, 0.6, 1.0)),
}

#: The patterns a model enumerates by default, in the order they are tried.
#: Omega decides between them; the order only decides which of two exactly
#: degenerate answers is reported.
DEFAULT_PATTERNS = ("unpaired", "2SC", "CFL", "free")


def pattern_mask(pattern):
    """Which of (Delta_1, Delta_2, Delta_3) this pattern makes unknowns."""
    if pattern not in PATTERNS:
        raise ValueError(f"unknown pairing pattern {pattern!r}; the patterns "
                         f"declared here are {sorted(PATTERNS)}")
    return PATTERNS[pattern][0]


def pattern_seed(pattern, scale):
    """The starting gaps of this pattern [MeV], at a gap scale of `scale`."""
    if pattern not in PATTERNS:
        raise ValueError(f"unknown pairing pattern {pattern!r}; the patterns "
                         f"declared here are {sorted(PATTERNS)}")
    return tuple(scale * s for s in PATTERNS[pattern][1])
