"""Colour pairing: the gap matrix, the quasiparticle spectrum, and what a
diquark condensate does to Omega, the densities, the entropy and itself.

Two models in this repository condense diquarks -- the three-flavour NJL of
`eos.njl` and the chiral colour-dielectric model of `eos.ccdm` -- and the
pairing sector of both is the SAME sector. Same nine colour-flavour modes,
same 9x9 gap matrix, same 36-state mean-field spectrum, same correction form
for the pairing piece of the thermodynamic potential, same Hellmann-Feynman
kernels for the gap equations. So it is written once here
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
mixing them corrupts mu_8 by factors of 1.15 to 1.7:

  * the halved Gell-Mann diag(1/2, 1/2, -1), for which
    mu_8^theirs = (2/sqrt(3)) mu_8^ours -- Ruester et al.,
    Pagliara-Schaffner-Bielich, and Kunkel et al.'s Eq. (7) as WRITTEN;
  * the full lambda_8, for which mu_8^ours = sqrt(3) mu_8^theirs -- Buballa,
    Steiner-Reddy-Prakash, Gholami et al. (arXiv:2411.04064 Eq. 3, which uses
    lambda_3 and lambda_8 by name), and the MUSES NJL module those two papers
    are computed with, whose paired potentials read
    mu_bar_ur_dg = mu~ + mu_Q/6 + mu_8/sqrt(3);
  * this module's own diag(1, 1, -2)/3.

The RG-consistent line of papers therefore SPLITS: Kunkel et al. print the
halved convention and compute in the full one, because their published module
is Gholami et al.'s. A comparison against either paper's CODE or against
Gholami's equations uses sqrt(3); only a comparison against the symbols
printed in Kunkel et al. uses 2/sqrt(3). Every mu_8 this module produces or
consumes is in the diag(1,1,-2)/3 normalisation.

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
    two branches cancel. `pair_block` differentiates the eigenvalues by
    Hellmann-Feynman instead, which carries the branch sign automatically
    because what enters Omega is |lambda| and its derivative therefore IS
    sign(lambda) dlambda/dDelta;

  * the spectrum is the one of the FULL DIRAC basis, not the on-shell
    Bogoliubov problem obtained by fixing E = sqrt(k^2 + M^2) first. The two
    agree exactly when the two paired quarks have equal masses and part
    company as those masses differ, so the reduction is harmless for 2SC and
    is a 6-16% error in the CFL pressure. See THE QUASIPARTICLE SPECTRUM;

  * paired-mode densities and entropies are NOT the unpaired Fermi integrals.
    At a 2SC point with Delta = 80 MeV the unpaired density formula is wrong
    by -21% on the paired u modes and +12% on the paired d modes, and at
    T = 5 MeV the unpaired entropy is four orders of magnitude too large;

  * the |xi_j| subtraction kinks at each of the nine Fermi momenta, and one
    quadrature panel cannot resolve nine kinks. `pair_nodes` splits there.

Antiparticle branches are not optional either: at T = 0 they contribute 8.8%
of the pairing potential at Lambda = 600 MeV and 17.1% at Lambda = 1000 MeV.

Reading order: the mode bookkeeping, the gap matrix, the quasiparticle
spectrum and the blocks it decomposes into, the quadrature, the unpaired
reference, the one pass that computes everything (twice: batched LAPACK and a
compiled twin), and the gap-root scan.

References
----------
Alford, Schmitt, Rajagopal, Schaefer, Rev. Mod. Phys. 80, 1455 (2008)
    [arXiv:0709.4635] -- the review; the gap matrix and the patterns.
Ruester, Werth, Buballa, Shovkovy, Rischke, Phys. Rev. D 72, 034004 (2005)
    [arXiv:hep-ph/0503184] -- the neutral three-flavour phase diagram, and
    Appendix A, which is the spectrum this module diagonalises.
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
# THE QUASIPARTICLE SPECTRUM
# =============================================================================
# The mean-field inverse propagator in the FULL DIRAC basis: four components
# per colour-flavour mode -- two particle, two antiparticle -- so 36 states at
# each momentum, with the gap mixing them. This is Ruester, Werth, Buballa,
# Shovkovy and Rischke, Phys. Rev. D 72, 034004 (2005), Appendix A.
#
# The familiar alternative solves the free Dirac problem FIRST, takes
# E = sqrt(k^2 + M^2), and pairs the resulting ON-SHELL modes in an 18x18
# Bogoliubov problem. It drops the particle-antiparticle mixing the gap
# induces, and WHAT CONTROLS THAT IS THE MASS MISMATCH OF THE PAIR: with the
# two paired quarks at equal masses the two spectra agree to 1e-13 MeV
# whatever the mass is, and they part company as the masses differ -- 3.4 MeV
# at M = (5.5, 300) MeV, Delta = 80 MeV, k = 320 MeV.
#
# So the reduction is harmless for 2SC, which pairs u with d, and it is not
# harmless for CFL, which pairs both of them with s. In CFL at Gholami et
# al.'s parameter set 1 the s quark pairs at M_s ~ 290 MeV against light
# partners near 20 MeV: the quasiparticle branches move by up to 11 MeV, the
# pressure by 6-16 %, and the whole 2SC -> CFL transition density by 9 %,
# which is how the discrepancy against the published module was found. The
# exact spectrum is therefore what this module carries, and the on-shell
# reduction is not offered as an option.
#
# The 36 states block-diagonalise: six 4x4 blocks for the six modes that pair
# pairwise, and one 12x12 for the (u_r, d_g, s_b) triple all three gaps couple.

#: Which two modes each gap pairs. eta = 1 pairs d with s, eta = 2 pairs u
#: with s, eta = 3 pairs u with d -- the same statement `B_ETA` makes, read
#: off it by `_check_blocks_against_B_ETA` below rather than asserted twice.
_PAIRED_MODES = {3: (("d", "r"), ("u", "g")),
                 2: (("s", "r"), ("u", "b")),
                 1: (("s", "g"), ("d", "b"))}

#: The triple every gap couples, which is what makes CFL one 12x12 block.
_TRIPLE_MODES = (("u", "r"), ("d", "g"), ("s", "b"))


def _spectrum_blocks(active=(True, True, True)):
    """The blocks a given set of NONZERO gaps splits the 36 states into.

    Each block is (rows, momentum, gaps) with

        rows      [(mode, s_mu, s_M), ...]   the diagonal, entry i being
                                             s_mu mu_mode + s_M M_flavour
        momentum  [(i, j), ...]              where the momentum k sits
        gaps      [(i, j, sign, eta), ...]   where Delta_eta sits

    all symmetric, and the second return is the modes those blocks cover.

    A mode NO nonzero gap touches is left out, and this is not an
    optimisation. Its four Dirac components are then exactly the unpaired ones
    the correction subtracts, so it contributes zero -- but computed as a
    difference it is zero MINUS AN ILL-CONDITIONED NUMBER: an ungapped branch
    crosses zero at its own Fermi surface, its partner at -0 sits in the same
    block, and a divide-and-conquer eigensolver mixes the two arbitrarily.
    The cancellation then fails at 1e-8, which is above `RESIDUAL_TOL` and
    stalls the solve. Leaving the mode out makes the zero exact.

    Two modes coupled by ONE gap split into two 4x4 blocks; the (u_r, d_g,
    s_b) triple, when two or three gaps connect it, is one 12x12.
    """
    index = {m: j for j, m in enumerate(MODES)}
    blocks = []
    covered = []

    for eta, (mode_a, mode_b) in _PAIRED_MODES.items():
        if not active[eta - 1]:
            continue
        a, b = index[mode_a], index[mode_b]
        covered += [a, b]
        for s in (+1.0, -1.0):
            rows = [(a, -s, +s), (a, -s, -s), (b, +s, +s), (b, +s, -s)]
            gaps = [(0, 3, -1.0, eta), (1, 2, +1.0, eta)]
            blocks.append((rows, [(0, 1), (2, 3)], gaps))

    # The triple, under whichever gaps are on: eta = 3 joins slots 0 and 1,
    # eta = 2 joins 0 and 2, eta = 1 joins 1 and 2.
    edges = [(eta, pair) for eta, pair in ((3, (0, 1)), (2, (0, 2)), (1, (1, 2)))
             if active[eta - 1]]
    for group in _connected_slots(edges):
        modes = [index[_TRIPLE_MODES[slot]] for slot in group]
        covered += modes
        inside = [(eta, (group.index(a), group.index(b)))
                  for eta, (a, b) in edges if a in group and b in group]
        if len(group) == 2:
            # one gap, two modes: the same 4x4 split the pairs above use
            eta = inside[0][0]
            a, b = modes
            for s in (+1.0, -1.0):
                rows = [(a, -s, +s), (a, -s, -s), (b, +s, +s), (b, +s, -s)]
                gaps = [(0, 3, -1.0, eta), (1, 2, +1.0, eta)]
                blocks.append((rows, [(0, 1), (2, 3)], gaps))
            continue
        rows, momentum, gaps = [], [], []
        for slot, mode in enumerate(modes):
            base = 4 * slot
            rows += [(mode, -1.0, -1.0), (mode, -1.0, +1.0),
                     (mode, +1.0, -1.0), (mode, +1.0, +1.0)]
            momentum += [(base, base + 1), (base + 2, base + 3)]
        for eta, (slot_a, slot_b) in inside:
            A, B = 4 * slot_a, 4 * slot_b
            gaps += [(A + 0, B + 3, -1.0, eta), (A + 1, B + 2, +1.0, eta),
                     (A + 2, B + 1, +1.0, eta), (A + 3, B + 0, -1.0, eta)]
        blocks.append((rows, momentum, gaps))
    return blocks, np.array(sorted(covered), dtype=np.int64)


def _connected_slots(edges):
    """The connected groups of the triple's three slots, under `edges`.

    A slot no edge reaches is a group of one, which `_spectrum_blocks` drops.
    """
    groups = []
    seen = set()
    for start in range(3):
        if start in seen:
            continue
        group, stack = [], [start]
        while stack:
            slot = stack.pop()
            if slot in seen:
                continue
            seen.add(slot)
            group.append(slot)
            for _, (a, b) in edges:
                if a == slot and b not in seen:
                    stack.append(b)
                if b == slot and a not in seen:
                    stack.append(a)
        if len(group) > 1:
            groups.append(sorted(group))
    return groups


def _block_tables(blocks):
    """`_spectrum_blocks` padded into the fixed-shape arrays numba can take."""
    n_blocks = len(blocks)
    width = max(len(rows) for rows, _, _ in blocks)
    n_mom = max(len(mom) for _, mom, _ in blocks)
    n_gap = max(len(gaps) for _, _, gaps in blocks)

    size = np.array([len(rows) for rows, _, _ in blocks], dtype=np.int64)
    row_mode = np.full((n_blocks, width), -1, dtype=np.int64)
    row_s_mu = np.zeros((n_blocks, width))
    row_s_M = np.zeros((n_blocks, width))
    mom_n = np.array([len(mom) for _, mom, _ in blocks], dtype=np.int64)
    mom_i = np.zeros((n_blocks, n_mom), dtype=np.int64)
    mom_j = np.zeros((n_blocks, n_mom), dtype=np.int64)
    gap_n = np.array([len(g) for _, _, g in blocks], dtype=np.int64)
    gap_i = np.zeros((n_blocks, n_gap), dtype=np.int64)
    gap_j = np.zeros((n_blocks, n_gap), dtype=np.int64)
    gap_s = np.zeros((n_blocks, n_gap))
    gap_eta = np.zeros((n_blocks, n_gap), dtype=np.int64)

    for b, (rows, mom, gaps) in enumerate(blocks):
        for i, (mode, s_mu, s_M) in enumerate(rows):
            row_mode[b, i], row_s_mu[b, i], row_s_M[b, i] = mode, s_mu, s_M
        for m, (i, j) in enumerate(mom):
            mom_i[b, m], mom_j[b, m] = i, j
        for g, (i, j, sign, eta) in enumerate(gaps):
            gap_i[b, g], gap_j[b, g] = i, j
            gap_s[b, g], gap_eta[b, g] = sign, eta - 1
    row_flavour = np.where(row_mode >= 0, FLAVOUR_OF_MODE[row_mode], 0)
    return (size, row_mode, row_s_mu, row_s_M, row_flavour,
            mom_n, mom_i, mom_j, gap_n, gap_i, gap_j, gap_s, gap_eta)


#: The blocks and their tables for each of the eight gap patterns, assembled
#: once at import: which gaps are nonzero decides the decomposition, and there
#: are only eight answers.
_BY_PATTERN = {}
for _mask in ((False, False, False), (True, False, False), (False, True, False),
              (False, False, True), (True, True, False), (True, False, True),
              (False, True, True), (True, True, True)):
    _blocks, _covered = _spectrum_blocks(_mask)
    _BY_PATTERN[_mask] = (_blocks, _covered,
                          _block_tables(_blocks) if _blocks else None)

#: The all-gaps-on decomposition, which is what `spectrum_matrix` and
#: `spectrum_energies` show a caller asking about the spectrum itself.
SPECTRUM_BLOCKS = _BY_PATTERN[(True, True, True)][0]


def spectrum_matrix(block, M_star, mu_star, Delta, k):
    """H(k) of one block, shape (..., n, n) over the momenta in `k` [MeV]."""
    rows, momentum, gaps = block
    k = np.asarray(k, dtype=float)
    n = len(rows)
    H = np.zeros(k.shape + (n, n))
    for i, (mode, s_mu, s_M) in enumerate(rows):
        H[..., i, i] = (s_mu * mu_star[mode]
                        + s_M * M_star[FLAVOUR_OF_MODE[mode]])
    for i, j in momentum:
        H[..., i, j] += k
        H[..., j, i] += k
    for i, j, sign, eta in gaps:
        H[..., i, j] += sign * Delta[eta - 1]
        H[..., j, i] += sign * Delta[eta - 1]
    return H


def spectrum_energies(M_star, mu_star, Delta, k):
    """The 18 non-negative quasiparticle energies at each momentum [MeV].

    The 36 eigenvalues come in +- pairs, and these are the positive half,
    sorted. At Delta = 0 they are |E_j -+ mu_j| over the nine modes, which is
    the unpaired spectrum `pair_block` subtracts.
    """
    M_star = np.asarray(M_star, dtype=float)
    mu_star = np.asarray(mu_star, dtype=float)
    Delta = np.asarray(Delta, dtype=float)
    k = np.asarray(k, dtype=float)
    lam = np.concatenate(
        [np.linalg.eigvalsh(spectrum_matrix(b, M_star, mu_star, Delta, k))
         for b in SPECTRUM_BLOCKS], axis=-1)
    return np.sort(lam, axis=-1)[..., 18:]


def _check_blocks_against_B_ETA():
    """`_PAIRED_MODES` and `_TRIPLE_MODES` against `B_ETA`, at import.

    Two statements of which modes pair would be two things to keep in step;
    this makes the block tables answerable to `B_ETA`, which `gap_matrix` and
    the counterterm already read. Cheap enough to run every import.
    """
    for eta, (mode_a, mode_b) in _PAIRED_MODES.items():
        a, b = MODES.index(mode_a), MODES.index(mode_b)
        if B_ETA[eta - 1, a, b] == 0.0:
            raise AssertionError(
                f"block table says Delta_{eta} pairs {mode_a} with {mode_b}, "
                f"which B_ETA does not")
    for eta, (slot_a, slot_b) in ((3, (0, 1)), (2, (0, 2)), (1, (1, 2))):
        a = MODES.index(_TRIPLE_MODES[slot_a])
        b = MODES.index(_TRIPLE_MODES[slot_b])
        if B_ETA[eta - 1, a, b] == 0.0:
            raise AssertionError(
                f"block table says Delta_{eta} couples the triple's slots "
                f"{slot_a} and {slot_b}, which B_ETA does not")


_check_blocks_against_B_ETA()


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


def pair_nodes(M_star, mu_star, T, k_max, nodes_per_panel=NODES_PER_PANEL,
               max_panel_ratio=None):
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
    return panel_nodes(kF, T, k_max, nodes_per_panel, max_panel_ratio)


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
# THE UNPAIRED REFERENCE
# =============================================================================
def _unpaired_reference(k, weight, M_mode, mu_star, T, covered):
    """What every entry of `PairBlock` is a correction TO.

    At Delta = 0 the 36 eigenvalues are +-(E_j -+ mu_j) over the nine modes,
    so the positive half is |xi| with xi = E_j - r mu_j for r = +-1, and it is
    written out here rather than diagonalised: it is analytic, it is the same
    for both flavours of the pass, and computing it the same way twice is what
    would let the correction fail to vanish at Delta = 0.

    `covered` names the modes to include -- the ones the blocks of
    `_spectrum_blocks` carry, since a mode absent from both sides cancels
    exactly rather than approximately.

    Returns (omega, entropy, per-mode d/dmu, per-flavour d/dM), each summed
    over the quadrature and UNSCALED by 1/(2 pi^2).
    """
    E = np.sqrt(k[:, None] ** 2 + M_mode[covered][None, :] ** 2)
    mu_star = mu_star[covered]
    flavour = FLAVOUR_OF_MODE[covered]
    omega = 0.0
    entropy = 0.0
    d_mu = np.zeros(N_MODES)
    d_M = np.zeros(3)
    for r in (+1.0, -1.0):
        xi = E - r * mu_star[None, :]
        omega += -float(np.sum(weight[:, None] * _phi(np.abs(xi), T)))
        entropy += float(np.sum(weight[:, None] * _dphi_dT(np.abs(xi), T)))
        tanh_xi = _dphi(xi, T)              # tanh(|xi|/2T) sign(xi)
        d_mu[covered] += -r * np.sum(weight[:, None] * tanh_xi, axis=0)
        per_mode = np.sum(
            weight[:, None] * tanh_xi * (M_mode[covered][None, :] / E), axis=0)
        for i_f in range(3):
            d_M[i_f] += float(np.sum(per_mode[flavour == i_f]))
    return omega, entropy, d_mu, d_M


# =============================================================================
# THE COMPILED PASS
# =============================================================================
# The same seven blocks, diagonalised by cyclic Jacobi in a compiled loop
# instead of by a batched LAPACK call. Only the PAIRED half is compiled: the
# unpaired reference above is analytic and cheap, and one copy of it is one
# fewer place for the two flavours to disagree.
#
# Hellmann-Feynman throughout, and the derivative matrices are why it is
# cheap: dH/dmu_j and dH/dM_f are DIAGONAL -- +-1 on the rows of that mode --
# so their expectation values are sums of squared eigenvector components, and
# dH/dDelta_eta has two or four nonzero entries. Nothing here builds a matrix
# product.
#
# ONE SELECTION RULE, stated once because it is the part that goes wrong
# quietly: the 36 eigenvalues come in +- pairs and Omega wants the positive
# half, which is taken as HALF THE SUM OVER ALL 36 of |lambda|. A branch that
# crosses zero inside a gapless window therefore needs no bookkeeping: its
# partner crosses back, and the derivative of |lambda| carries sign(lambda),
# which is the `sgn` factor below.


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
def _symmetric_eigh(A, V, d, e, n):
    """Eigenvalues into `d`, eigenvectors into the columns of `V`. In place.

    Householder reduction to tridiagonal form followed by the implicit-shift
    QL iteration -- EISPACK's tred2 and tql2, which is what LAPACK's dsyev
    does and what `numpy.linalg.eigh` is measured against here: over a
    thousand random symmetric matrices of order 2 to 12 the eigenvalues agree
    to 3.2e-15 relative and the residual ||A v - lambda v|| is 5.6e-15.

    Only the leading n x n of the buffers is touched, so one preallocated
    12x12 pair serves every block. Cyclic Jacobi was written first and is
    four times slower on the 12x12: 7.6 ms against 1.9 ms over a 672-node
    pass, which is most of the pairing cost.
    """
    for i in range(n):
        for j in range(n):
            V[i, j] = A[i, j]
    # --- Householder reduction to tridiagonal form (EISPACK tred2) --------
    for j in range(n):
        d[j] = V[n - 1, j]
    for i in range(n - 1, 0, -1):
        scale = 0.0
        h = 0.0
        for q in range(i):
            scale += abs(d[q])
        if scale == 0.0:
            e[i] = d[i - 1]
            for j in range(i):
                d[j] = V[i - 1, j]
                V[i, j] = 0.0
                V[j, i] = 0.0
        else:
            for q in range(i):
                d[q] /= scale
                h += d[q] * d[q]
            f = d[i - 1]
            g = np.sqrt(h)
            if f > 0.0:
                g = -g
            e[i] = scale * g
            h -= f * g
            d[i - 1] = f - g
            for j in range(i):
                e[j] = 0.0
            for j in range(i):
                f = d[j]
                V[j, i] = f
                g = e[j] + V[j, j] * f
                for q in range(j + 1, i):
                    g += V[q, j] * d[q]
                    e[q] += V[q, j] * f
                e[j] = g
            f = 0.0
            for j in range(i):
                e[j] /= h
                f += e[j] * d[j]
            hh = f / (h + h)
            for j in range(i):
                e[j] -= hh * d[j]
            for j in range(i):
                f = d[j]
                g = e[j]
                for q in range(j, i):
                    V[q, j] -= (f * e[q] + g * d[q])
                d[j] = V[i - 1, j]
                V[i, j] = 0.0
        d[i] = h
    for i in range(n - 1):
        V[n - 1, i] = V[i, i]
        V[i, i] = 1.0
        h = d[i + 1]
        if h != 0.0:
            for q in range(i + 1):
                d[q] = V[q, i + 1] / h
            for j in range(i + 1):
                g = 0.0
                for q in range(i + 1):
                    g += V[q, i + 1] * V[q, j]
                for q in range(i + 1):
                    V[q, j] -= g * d[q]
        for q in range(i + 1):
            V[q, i + 1] = 0.0
    for j in range(n):
        d[j] = V[n - 1, j]
        V[n - 1, j] = 0.0
    V[n - 1, n - 1] = 1.0
    e[0] = 0.0
    # --- the implicit-shift QL iteration (EISPACK tql2) -------------------
    for i in range(1, n):
        e[i - 1] = e[i]
    e[n - 1] = 0.0
    f = 0.0
    tst1 = 0.0
    eps = 2.0 ** -52
    for l in range(n):
        if abs(d[l]) + abs(e[l]) > tst1:
            tst1 = abs(d[l]) + abs(e[l])
        m = l
        while m < n:
            if abs(e[m]) <= eps * tst1:
                break
            m += 1
        if m > l:
            while True:
                g = d[l]
                p = (d[l + 1] - g) / (2.0 * e[l])
                r = np.hypot(p, 1.0)
                if p < 0.0:
                    r = -r
                d[l] = e[l] / (p + r)
                d[l + 1] = e[l] * (p + r)
                dl1 = d[l + 1]
                h = g - d[l]
                for i in range(l + 2, n):
                    d[i] -= h
                f += h
                p = d[m]
                c = 1.0
                c2 = c
                c3 = c
                el1 = e[l + 1]
                s = 0.0
                s2 = 0.0
                for i in range(m - 1, l - 1, -1):
                    c3 = c2
                    c2 = c
                    s2 = s
                    g = c * e[i]
                    h = c * p
                    r = np.hypot(p, e[i])
                    e[i + 1] = s * r
                    s = e[i] / r
                    c = p / r
                    p = c * d[i] - s * g
                    d[i + 1] = h + s * (c * g + s * d[i])
                    for q in range(n):
                        h = V[q, i + 1]
                        V[q, i + 1] = s * V[q, i] + c * h
                        V[q, i] = c * V[q, i] - s * h
                p = -s * s2 * c3 * el1 * e[l] / dl1
                e[l] = s * p
                d[l] = c * p
                if abs(e[l]) <= eps * tst1:
                    break
        d[l] += f
        e[l] = 0.0


@njit(cache=True)
def _pair_pass(k, w, M_star, mu_star, Delta, T, size, row_mode, row_s_mu,
               row_s_M, row_flavour, mom_n, mom_i, mom_j, gap_n, gap_i, gap_j,
               gap_s, gap_eta):
    """The paired half of `pair_block`'s quadrature, one compiled pass.

    Returns the UNSCALED accumulators (omega, d/dmu[9], d/dM[3],
    d/dDelta[3], entropy, min_energy), each in the same convention as the
    reference path's, so `pair_block` subtracts one unpaired reference from
    either.
    """
    n_blocks = size.shape[0]
    width = row_mode.shape[1]
    A = np.zeros((width, width))
    V = np.zeros((width, width))
    lam = np.zeros(width)
    work = np.zeros(width)

    omega = 0.0
    entropy = 0.0
    d_mu = np.zeros(N_MODES)
    d_M = np.zeros(3)
    d_Delta = np.zeros(3)
    min_energy = np.inf

    for node in range(k.shape[0]):
        kk = k[node]
        weight = w[node] * kk * kk
        for b in range(n_blocks):
            n = size[b]
            for i in range(n):
                for j in range(n):
                    A[i, j] = 0.0
            for i in range(n):
                mode = row_mode[b, i]
                A[i, i] = (row_s_mu[b, i] * mu_star[mode]
                           + row_s_M[b, i] * M_star[row_flavour[b, i]])
            for m in range(mom_n[b]):
                i, j = mom_i[b, m], mom_j[b, m]
                A[i, j] += kk
                A[j, i] += kk
            for g in range(gap_n[b]):
                i, j = gap_i[b, g], gap_j[b, g]
                value = gap_s[b, g] * Delta[gap_eta[b, g]]
                A[i, j] += value
                A[j, i] += value

            _symmetric_eigh(A, V, lam, work, n)

            for branch in range(n):
                energy = abs(lam[branch])
                if energy < min_energy:
                    min_energy = energy
                omega -= 0.5 * weight * _phi_scalar(energy, T)
                entropy += 0.5 * weight * _dphi_dT_scalar(energy, T)
                occ = 0.5 * weight * _dphi_scalar(energy, T)
                if lam[branch] < 0.0:
                    occ = -occ
                for i in range(n):
                    share = occ * V[i, branch] * V[i, branch]
                    d_mu[row_mode[b, i]] += row_s_mu[b, i] * share
                    d_M[row_flavour[b, i]] += row_s_M[b, i] * share
                for g in range(gap_n[b]):
                    i, j = gap_i[b, g], gap_j[b, g]
                    d_Delta[gap_eta[b, g]] += (2.0 * gap_s[b, g] * occ
                                               * V[i, branch] * V[j, branch])

    return omega, d_mu, d_M, d_Delta, entropy, min_energy


def _pair_pass_reference(k, weight, M_star, mu_star, Delta, T, blocks):
    """The paired half, batched over the quadrature with `numpy.linalg.eigh`.

    The same accumulators `_pair_pass` returns, computed the same way -- one
    Hellmann-Feynman expectation value per branch per parameter -- with the
    node loop pushed into LAPACK instead of into numba.
    """
    omega = 0.0
    entropy = 0.0
    d_mu = np.zeros(N_MODES)
    d_M = np.zeros(3)
    d_Delta = np.zeros(3)
    min_energy = np.inf

    for rows, momentum, gaps in blocks:
        H = spectrum_matrix((rows, momentum, gaps), M_star, mu_star, Delta, k)
        lam, V = np.linalg.eigh(H)                       # (nk, n), (nk, n, n)
        energy = np.abs(lam)
        min_energy = min(min_energy, float(np.min(energy)))

        omega += -0.5 * float(np.sum(weight[:, None] * _phi(energy, T)))
        entropy += 0.5 * float(np.sum(weight[:, None] * _dphi_dT(energy, T)))

        occ = 0.5 * weight[:, None] * _dphi(energy, T) * np.sign(lam)
        # dH/dmu_j and dH/dM_f are diagonal, so their expectation values are
        # the squared components of each eigenvector, summed over the rows of
        # that mode -- one contraction serves every one of the twelve.
        share = np.einsum("kb,kib->i", occ, V * V)        # (n,)
        for i, (mode, s_mu, s_M) in enumerate(rows):
            d_mu[mode] += s_mu * share[i]
            d_M[FLAVOUR_OF_MODE[mode]] += s_M * share[i]
        for i, j, sign, eta in gaps:
            d_Delta[eta - 1] += 2.0 * sign * float(
                np.sum(occ * V[:, i, :] * V[:, j, :]))

    return omega, d_mu, d_M, d_Delta, entropy, min_energy


def pair_block(M_star, mu_star, Delta, T, k_max,
               nodes_per_panel=NODES_PER_PANEL, quadrature=None,
               backend="reference"):
    """The pairing correction to Omega, n_j, rho_s,f, s and the gap equations.

    One quadrature pass, one set of diagonalisations, five results: computing
    them separately would diagonalise the same blocks five times, and
    finite-differencing them instead was measured 40x slower and
    ill-conditioned enough to lose convergence.

    Every entry is a CORRECTION to the unpaired spectrum -- the paired sum
    minus `_unpaired_reference` -- so every entry is identically zero at
    Delta = 0 and a model adds them to its unpaired sums without a second code
    path.

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
    backend : 'reference' (default) or 'fast'. The two diagonalise the same
        seven blocks, by LAPACK batched over the quadrature or by compiled
        cyclic Jacobi node by node, and agree to round-off; the choice belongs
        to the caller that already declared one (CLAUDE.md section 9). Without
        numba, 'fast' runs the reference path -- the same numbers, not a
        different answer.

    Both the particle branches and the ANTIPARTICLE branches are summed; they
    are two halves of the one spectrum here, since the Dirac basis carries
    both. The antiparticle piece is not a small correction and it grows with
    the cutoff: 8.8% of the particle piece at Lambda = 600 MeV, 17.1% at
    Lambda = 1000 MeV.
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

    blocks, covered, tables = _BY_PATTERN[tuple(Delta != 0.0)]
    if backend == "fast" and _NUMBA_OK:
        omega, d_mu, d_M, d_Delta, entropy, min_energy = _pair_pass(
            np.ascontiguousarray(k, dtype=float),
            np.ascontiguousarray(w, dtype=float),
            np.ascontiguousarray(M_star, dtype=float),
            np.ascontiguousarray(mu_star, dtype=float),
            np.ascontiguousarray(Delta, dtype=float), float(T), *tables)
    elif backend in ("reference", "fast"):
        omega, d_mu, d_M, d_Delta, entropy, min_energy = _pair_pass_reference(
            k, weight, M_star, mu_star, Delta, T, blocks)
    else:
        raise ValueError(f"unknown backend {backend!r}; eos.general.pairing "
                         f"has 'reference' and 'fast'")

    M_mode = M_star[FLAVOUR_OF_MODE]
    omega_0, entropy_0, d_mu_0, d_M_0 = _unpaired_reference(
        k, weight, M_mode, mu_star, T, covered)

    inv = 1.0 / (2.0 * np.pi ** 2)
    scale = float(np.max(np.abs(Delta)))
    return PairBlock(delta_omega=(omega - omega_0) * inv,
                     delta_n=(d_mu - d_mu_0) * inv,
                     # rho_s = +dOmega/dM, and the sign is in here
                     delta_rho_s=-(d_M - d_M_0) * inv,
                     delta_s=(entropy - entropy_0) * inv,
                     gap_kernel=d_Delta * inv,
                     min_energy=float(min_energy),
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


#: Which pattern a SOLVED set of gaps actually realises, by the mask of the
#: gaps that came out nonzero. Every one of the eight masks is named, including
#: the one `PATTERNS` does not offer as a request: (Delta_1, Delta_2) nonzero
#: with Delta_3 zero is the s-pairing state, sSC, which no seed here aims at but
#: which a free solve can land on.
_REALISED = {
    (False, False, False): "unpaired",
    (False, False, True): "2SC",
    (False, True, False): "usSC",
    (True, False, False): "dsSC",
    (False, True, True): "uSC",
    (True, False, True): "dSC",
    (True, True, False): "sSC",
    (True, True, True): "CFL",
}

#: A gap smaller than this fraction of the largest one is zero. Solved gaps a
#: pattern leaves free come back as exactly +-0.0 when the state does not
#: condense them, so this only has to separate that from a real gap.
REALISED_TOL = 1.0e-6


def realised_pattern(Delta, rel_tol=REALISED_TOL, abs_floor=1.0e-6):
    """The pattern a solved gap vector IS, which is not what was requested.

    A pattern is a DECLARATION of which gaps are free (`pattern_mask`), and a
    free gap is free to come out zero: Delta_1 = Delta_2 = 0 is a perfectly
    good root of the CFL layout, and it is the 2SC state. So a solve requested
    in one pattern can converge on another, and the requested name is then a
    statement about the unknown vector rather than about the phase.

    That is not hypothetical and it is not rare. A warm-started sweep asked for
    'CFL' from below the CFL onset converges on 2SC at its first density and
    carries that root up the whole sweep; every point comes back labelled 'CFL'
    with Delta = (0, 0, ~260). Nothing downstream could see it, because the
    only pattern a point carried was the one that had been asked for.

    Returns one of the eight names of `_REALISED`. Six of them are keys of
    `PATTERNS` and can be requested back; 'sSC', 'usSC' and 'dsSC' are
    REPORTS -- states a free seed can reach that no seed here aims at -- and
    passing one to `pattern_mask` raises, which is the intended behaviour.

    The comparison is relative to the largest gap, so it is scale-free; a
    vector whose largest gap is below `abs_floor` [MeV] is unpaired.
    """
    Delta = np.abs(np.asarray(Delta, dtype=float))
    scale = float(Delta.max()) if Delta.size else 0.0
    if scale <= abs_floor:
        return "unpaired"
    mask = tuple(bool(d > rel_tol * scale) for d in Delta)
    return _REALISED[mask]
