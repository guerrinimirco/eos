"""
mixed/construction.py
=====================
Where an ENJL parameter set's first-order transitions are, as data a model can
assemble a delivered table from.

*Public API* (re-exported from `eos.mixed`): `Coexistence`,
`enjl_coexistences`.

`eos.enjl` is a model and a model does not import a composite engine
(CLAUDE.md section 1: general/ -> models -> composite engines -> astro/), so
the ENJL branch vocabulary and everything that pairs two branches live here,
next to `enjl_phase` and for the same reason. What crosses back to the model
is a `Coexistence` -- plain numbers and two edge rows -- which
`eos.enjl.table.build_constructed_table` takes as an ARGUMENT. That keeps the
locator (which needs both branches and the eta = 1 lepton bookkeeping) on this
side of the layering and the assembly (which is pure ENJL) on the model's.

A model with three branches admits three pairings, and which of them are
realized is a property of the parameter set, not of this module. So this is a
loop over declared branch pairs calling `locate_maxwell` once each, exactly
the caller-level loop docs/enjl/PHASE_TRANSITION_DESIGN.md section 4 asks for:
the engine stays a two-phase engine and knows nothing about how many branches
a model has.
"""
from dataclasses import dataclass, field

import numpy as np

from eos.mixed.adapters import enjl_branch_pair
from eos.mixed.boundaries import locate_maxwell, neutral_phase

#: The branch pairings a construction sweeps, low-density branch first. The
#: chiral transition pairs the broken and restored branches; deconfinement
#: pairs the restored and deconfined ones. (broken, deconfined) is the third
#: pairing the three branches admit and is not swept by default: where it
#: exists it is the same transition one of the other two already found.
CONSTRUCTION_PAIRS = (("broken", "restored"), ("restored", "deconfined"))


@dataclass(frozen=True)
class Coexistence:
    """One located first-order transition, with both edge states.

    mu_B, P          : the coexistence potential [MeV] and pressure
                       [MeV/fm^3], equal across the two phases by
                       construction -- that IS the Maxwell condition.
    n_B_lo, n_B_hi   : the window edges [fm^-3]. Between them no single phase
                       is stable and the delivered EoS is this plateau.
    mu_e_lo, mu_e_hi : each phase's OWN neutralizing electron potential. They
                       differ, and that difference is what eta = 1 means:
                       neither phase borrows charge from the other.
    branches         : the two branch labels, low then high.
    row_lo, row_hi   : the edge states as flat rows, keyed as
                       `eos.enjl.table.beta_row` keys them, so the assembler
                       lever-rules them without knowing what a PhaseThermo is.

    At T = 0 the Euler relation on a neutral phase collapses to
    eps = mu_B n_B - P (the mu_C n_C and lepton terms cancel against each
    other when mu_C = -mu_e and n_C = n_e + n_mu), which is why the plateau
    needs no further solve: eps is linear in n_B across it automatically.
    """
    mu_B: float
    P: float
    n_B_lo: float
    n_B_hi: float
    mu_e_lo: float
    mu_e_hi: float
    branches: tuple
    row_lo: dict = field(default_factory=dict)
    row_hi: dict = field(default_factory=dict)

    @property
    def width(self):
        """Width of the window in density [fm^-3]."""
        return self.n_B_hi - self.n_B_lo

    def contains(self, n_B):
        return self.n_B_lo <= n_B <= self.n_B_hi


def _edge_row(th, leptons, mu_C, branch):
    """One coexistence edge as a flat row, keyed as `beta_row` keys them."""
    n = dict(th.densities)
    n_bQ = (n.get("u", 0.0) + n.get("d", 0.0) + n.get("s", 0.0)) / 3.0
    return dict(
        n_B=th.n_B, T=0.0, P=th.P + leptons.P, eps=th.eps + leptons.e,
        s=0.0, S_per_B=0.0,
        chi=n_bQ / th.n_B if th.n_B > 0.0 else 0.0,
        mu_B=th.mu_B, mu_C=mu_C, mu_S=th.mu_S, mu_e=-mu_C,
        Y_C=th.n_C / th.n_B if th.n_B > 0.0 else 0.0,
        Y_S=th.n_S / th.n_B if th.n_B > 0.0 else 0.0,
        n_p=n.get("p", 0.0), n_n=n.get("n", 0.0),
        n_Lambda=n.get("Lambda", 0.0),
        n_u=n.get("u", 0.0), n_d=n.get("d", 0.0), n_s=n.get("s", 0.0),
        n_e=leptons.n_e, n_mu=leptons.n_mu,
        M_u=th.m_eff_i["u"], M_d=th.m_eff_i["d"], M_s=th.m_eff_i["s"],
        M_p=th.m_eff_i["p"], M_n=th.m_eff_i["n"],
        M_Lambda=th.m_eff_i["Lambda"], branch=branch)


def enjl_coexistences(par, mu_B_grid, pairs=CONSTRUCTION_PAIRS, T=0.0,
                      muons=True, progress=None):
    """Every first-order transition of `par` on `mu_B_grid`, in density order.

    One `locate_maxwell` per branch pair. A pair whose two branches never both
    exist on the grid, or whose pressures do not cross on it, contributes
    nothing -- that is a physics outcome for those parameters and not a
    failure, so it is a shorter list rather than an exception.

    Duplicates are dropped: two pairings can find the same transition (the
    same physical crossing reached from either side), and a window whose edges
    both fall inside one already accepted is that, not a second transition.

    `progress`, if given, is called once per pair with
    `(branches, Coexistence or None)` -- deep solver code never prints
    (CLAUDE.md section 5).

    Returns a list of `Coexistence`, sorted by `n_B_lo`.
    """
    if T != 0.0:
        raise NotImplementedError(
            f"the ENJL construction is written at T = 0; got T = {T} MeV. "
            f"Finite T needs eos.enjl itself to leave T = 0 first")

    grid = np.asarray(mu_B_grid, dtype=float)
    found = []
    for branches in pairs:
        lo_phase, hi_phase = enjl_branch_pair(par, branches)

        def call(phase):
            return lambda mu, mu_C: phase.thermo(mu, mu_C, 0.0, T)

        point = locate_maxwell(call(lo_phase), call(hi_phase), grid, T=T,
                               muons=muons, labels=branches)
        if not point.exists:
            if progress is not None:
                progress(branches, None)
            continue

        edges = []
        for phase, branch in ((lo_phase, branches[0]),
                              (hi_phase, branches[1])):
            state = neutral_phase(call(phase), point.mu_B, T=T, muons=muons)
            if state is None:                # located, but not re-solvable
                edges = None
                break
            th, leptons, mu_C = state
            edges.append(_edge_row(th, leptons, mu_C, branch))
        if edges is None:
            if progress is not None:
                progress(branches, None)
            continue

        co = Coexistence(mu_B=point.mu_B, P=point.P,
                         n_B_lo=point.n_B_lo, n_B_hi=point.n_B_hi,
                         mu_e_lo=point.mu_e_lo, mu_e_hi=point.mu_e_hi,
                         branches=tuple(branches),
                         row_lo=edges[0], row_hi=edges[1])
        if progress is not None:
            progress(branches, co)
        if not any(w.contains(co.n_B_lo) and w.contains(co.n_B_hi)
                   for w in found):
            found.append(co)
    return sorted(found, key=lambda w: w.n_B_lo)


__all__ = ["CONSTRUCTION_PAIRS", "Coexistence", "enjl_coexistences"]
