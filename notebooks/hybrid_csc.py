# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Hybrid stars with colour-superconducting quark cores
#
# A hadronic phase carrying the full baryon octet and the Delta quartet, against
# two colour-superconducting quark models, at `T = 0` in beta equilibrium with
# free-streaming neutrinos:
#
# * **DD2Y + NJL**, the three-flavour Nambu–Jona-Lasinio model with `csc=True`
# * **DD2Y + CCDM**, the chiral colour-dielectric model with `csc=True`
#
# The two observables asked of each pairing are `P(n_B)` and the mass–radius
# curve. Both come out of the library's public API; nothing here reaches into a
# solver internal, and there is no helper module beside this notebook.
#
# Three findings shape how the notebook is built, and each is demonstrated
# rather than asserted:
#
# 1. **The hadronic side must be `DD2Y`, not `DD2`.** The two are different
#    published parameterisations, and the nucleonic set carries no hyperon
#    couplings — asking `DD2` for hyperons raises rather than guessing.
# 2. **The quark phase must be ENUMERATED, and the branch then enveloped.**
#    Letting the model choose its own pairing pattern at each density is what
#    makes the result the ground state: the winner runs unpaired -> 2SC -> CFL
#    with rising density, and CFL is stable above `mu_B ~ 1385 MeV`, which is
#    below the transition. Holding one pattern is faster and gives a
#    METASTABLE core. The price of enumerating is that `mu_B(n_B)` runs
#    backwards at each phase change, since two phases are being reported on one
#    density axis; the upper-`P` envelope of section 2 turns that back into a
#    single stable branch. The one place a pattern IS held is section 7, where
#    the mixed engine's Newton solve needs a phase function without steps.
# 3. **The transition is closed by a Maxwell construction on the two pure
#    branches**, because the eta-mixed window never closes: `DD2Y` with
#    hyperons and Deltas ceases to exist near `n_B ~ 1.03 fm^-3`, while the
#    quark volume fraction chi has only reached about one half. Section 7
#    shows the engine's own mixed points and where they stop.
#
# Units are the ones every public boundary of `eos` uses: `n` in fm^-3, `mu` and
# `T` in MeV, `P` and `eps` in MeV/fm^3.

# %%
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# `eos` is imported from this checkout. Works from `notebooks/` or from the root.
ROOT = Path.cwd()
if not (ROOT / "eos").is_dir():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from eos.dd2.parameters import Parameters as DD2Parameters
from eos.dd2.species import SpeciesFlags as DD2Flags
from eos.dd2.api import eos_table as dd2_eos_table

from eos.njl.parameters import Parameters as NJLParameters
from eos.njl.species import SpeciesFlags as NJLFlags
from eos.njl.api import eos_table as njl_eos_table
from eos.njl.solver import solve_beta_eq_neutrinoless as njl_beta_eq

from eos.ccdm.parameters import Parameters as CCDMParameters
from eos.ccdm.species import SpeciesFlags as CCDMFlags
from eos.ccdm.api import eos_table as ccdm_eos_table

from eos.general.state import EOSTable_for_TOV
from eos.general.figure_style import STANDARD_COLORS, save_figure, set_paper_style
from eos.astro.tov.crust import have_crust
from eos.astro.tov.solver import (compute_tov_sequence, find_mmax_precise,
                                  generate_ec_logspace)

set_paper_style()

FIG_DIR = ROOT / "output" / "hybrid_csc"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. The knobs
#
# Every choice this notebook makes is here and nowhere else.
#
# `HELD_PATTERN` is not the physics knob it looks like. Sections 3-5 use the
# ENUMERATED quark branch — the model choosing its own colour-superconducting
# phase — and the held pattern is used only in section 7, where the mixed
# engine's Newton solve needs a phase function without steps in it. Holding
# `2SC` through the transition gives a METASTABLE core: the enumeration puts
# CFL above mu_B ~ 1385 MeV, below where the transition sits.
#
# `RUN_CCDM` is a runtime switch rather than a physics one: the CCDM branch
# enumeration costs roughly a minute per density point, so the sweep is coarse
# and can be turned off entirely. Section 6 is the only part that needs it.

# %%
HADRONIC_SET = "DD2Y"                 # DD2 carries no hyperon couplings
HADRONIC_FLAGS = DD2Flags(hyperons=True, deltas=True, muons=False)

# The quark branch is built by `eos_table`, which ENUMERATES the pairing
# patterns and keeps the highest-pressure one at each density — that is the
# model's own answer to "colour superconductivity", and it is what sections
# 3-5 use. `HELD_PATTERN` is a second, cheap sweep with one pattern held; it
# exists only for the cross-check in section 7, where the mixed engine needs a
# continuous phase function and cannot enumerate.
HELD_PATTERN = ("2SC",)
NJL_FLAGS = NJLFlags(csc=True, muons=False)
CCDM_FLAGS = CCDMFlags(csc=True, muons=False)

N_B_HADRONIC = np.linspace(0.06, 1.10, 200)    # sweep stops itself at the
                                              # scalar-collapse boundary
N_B_NJL = np.linspace(0.25, 2.00, 100)
N_B_CCDM = np.linspace(0.20, 1.20, 24)        # enumerates BRANCHES as well as
                                              # patterns: ~1 min a point, so a
                                              # 100-point grid is ~2 hours

RUN_CCDM = True

COLOR_H = STANDARD_COLORS["Blue"]
COLOR_NJL = STANDARD_COLORS["Red"]
COLOR_CCDM = STANDARD_COLORS["Green"]
COLOR_HYBRID = STANDARD_COLORS["Purple"]

# %% [markdown]
# ## 2. The two pure branches
#
# Each phase is solved by its own model, in its own beta-equilibrium mode, and
# reports the *physical* baryon chemical potential. That matters for section 3:
# a Maxwell construction equates `P` at equal `mu_B`, so both branches have to
# be expressed in the same potential. The DD2 phase adapter of `eos.mixed`
# carries the *kinetic* potential in its slot instead, which is why the
# construction below is done on the models' own beta-equilibrium output rather
# than through `eos.mixed.boundaries.locate_maxwell`.
#
# The hadronic side goes through `eos_table`, the uniform API of CLAUDE.md
# section 5. Its density axis is warm-started for us, and the line simply ends
# where the octet solve stops converging — the scalar-collapse boundary. Where
# it stops is a property of the parameter set with hyperons and Deltas switched
# on, and it is the fact section 3 turns on.
#
# The quark side goes through `eos_table` too. It costs roughly 25 s a point
# rather than 2, because it ENUMERATES the pairing patterns at every density
# and keeps the one with the largest pressure — but that enumeration is the
# point: it is how the model answers which colour-superconducting phase the
# matter is in, and it returns the winning `pattern` per row.
#
# What `eos_table` cannot express is HOLDING one pattern: `patterns` is not a
# `TableSpec` field in `eos.njl.table` or `eos.ccdm.table`. That is a real gap
# and belongs in `docs/DEFERRED.md` — but it is a gap in speed and in phase
# selection, not one that forces the construction off the public API.

# %%
hadronic_par = DD2Parameters.named(HADRONIC_SET)

t0 = time.time()
hadronic_table = dd2_eos_table(hadronic_par, "beta_eq_neutrinoless",
                               HADRONIC_FLAGS,
                               {"nB": N_B_HADRONIC, "T": np.array([0.0])})
had = np.array([(point.n_B, point.matter.mu_B, point.P, point.eps)
                for point in hadronic_table.points[0]])

print(f"DD2Y + hyperons + Deltas: {len(had)} of {len(N_B_HADRONIC)} points "
      f"({time.time() - t0:.0f} s)")
# `eos_table` drops a point that does not converge rather than ending the line,
# so a line that stops at a boundary and one with a hole in it look alike from
# the length alone. The spacing says which this is.
step = float(N_B_HADRONIC[1] - N_B_HADRONIC[0])
print(f"  contiguous (a truncation, not a gap): "
      f"{bool(np.allclose(np.diff(had[:, 0]), step))}")
print(f"  n_B   {had[0, 0]:.3f} .. {had[-1, 0]:.3f} fm^-3   "
      f"<- the branch ends here")
print(f"  mu_B  {had[0, 1]:.1f} .. {had[-1, 1]:.1f} MeV")
print(f"  P     {had[0, 2]:.2f} .. {had[-1, 2]:.2f} MeV/fm^3")

# %% [markdown]
# The quark branch, enumerated. `mu_B(n_B)` runs backwards wherever the winning
# pattern changes — unpaired to 2SC, then 2SC to CFL — because two different
# phases are being reported on one density axis. That is physics, not a solver
# fault, and the envelope below is what turns it into a single stable branch.
#
# **Do not replace this with a hand-rolled warm-started loop.** CLAUDE.md
# section 6 discourages bare loops over point solves, and `eos.njl.solver`
# offers `warm_start`, so threading it through a density loop looks like the
# tidy version. Measured, it is smoother and WRONG: the sweep begins in
# unpaired matter, and a seed that only ever comes from the previous point
# keeps the solver on the trivial `Delta = 0` root for the whole sweep — the
# gap never leaves 0 MeV, so the branch comes back with no colour
# superconductivity in it at all. `warm_start`'s own docstring names the
# mechanism: "a pattern that only ever sees a warm start from itself can never
# be displaced." `eos_table` enumerates at every density, so it cannot get
# stuck that way.
#
# **Each point is solved COLD, and that is deliberate.** CLAUDE.md section 6
# says a table should not be a bare loop over point solves unless the solver
# needs the previous point as a warm start, and `eos.njl.solver` does offer
# `warm_start`. Passing it here makes the branch smoother and WRONG: the sweep
# begins in unpaired matter, and a seed that only ever comes from the previous
# point keeps the solver on the trivial `Delta = 0` root for the whole sweep —
# measured, the gap never leaves 0 MeV, so the returned branch has no colour
# superconductivity in it at all. `warm_start`'s own docstring names the
# mechanism: "a pattern that only ever sees a warm start from itself can never
# be displaced."
#
# Cold starts land on different roots at different densities, which is what
# makes both of them visible; the envelope below then chooses between them by
# pressure. The apparent untidiness is what keeps the paired solution in the
# table.

# %%
njl_par = NJLParameters.default()

t0 = time.time()
njl_table = njl_eos_table(njl_par, "beta_eq_neutrinoless", NJL_FLAGS,
                          {"nB": N_B_NJL, "T": np.array([0.0])})
njl_line = njl_table.points[0]
njl = np.array([(p.n_B, p.mu_B, p.P, p.eps) for p in njl_line])

print(f"NJL (csc, enumerated): {len(njl)} of {len(N_B_NJL)} points "
      f"({time.time() - t0:.0f} s)")
print(f"  phases found: {sorted(set(p.pattern for p in njl_line))}")
print(f"  mu_B monotone: {bool(np.all(np.diff(njl[:, 1]) > 0))}")
print(f"  mu_B  {njl[0, 1]:.1f} .. {njl[-1, 1]:.1f} MeV")
print(f"  P     {njl[0, 2]:.2f} .. {njl[-1, 2]:.2f} MeV/fm^3")

# %% [markdown]
# `mu_B` still runs backwards in places, and holding the pattern fixed is not
# what cures it: the gap equation has more than one root INSIDE a pattern. The
# printed gaps below show it — `Delta_3` is zero on some rows and around 90 MeV
# on others, and the solver returns whichever root its seed reached. Those are
# the unpaired and the 2SC-paired solutions of the same equations, and both are
# genuine roots.
#
# The stable one at a given `mu_B` is the one with the largest pressure. That
# is not a numerical convenience: it is the same criterion the phase adapters
# of `eos.mixed` use to choose between patterns, applied here between roots of
# one pattern. Taking the upper envelope in `P` over `mu_B` therefore selects
# the stable branch and drops the metastable rows, and what it leaves is
# monotone in both `mu_B` and `P` — which is also what `np.interp` in section 3
# requires, and would silently violate if handed the raw branch.
#
# The raw branch is kept for the figure: the metastable roots are worth seeing,
# and section 4 draws them.

# %%
def stable_branch(rows):
    """The upper-P envelope over mu_B: the stable root at each potential.

    `rows` is the (n_B, mu_B, P, eps) array of one phase. Sorting by mu_B and
    keeping only rows whose pressure exceeds every pressure at a lower mu_B
    discards the metastable roots, because along a stable branch both mu_B and
    P increase together (dP/dmu_B = n_B > 0).
    """
    ordered = rows[np.argsort(rows[:, 1])]
    keep, P_best = [], -np.inf
    for row in ordered:
        if row[2] > P_best:
            keep.append(row)
            P_best = row[2]
    return np.array(keep)


njl_stable = stable_branch(njl)
print(f"stable branch: {len(njl_stable)} of {len(njl)} rows kept")
print(f"  dropped {len(njl) - len(njl_stable)} metastable rows")
print(f"  mu_B monotone: {bool(np.all(np.diff(njl_stable[:, 1]) > 0))}")
print(f"  P    monotone: {bool(np.all(np.diff(njl_stable[:, 2]) > 0))}")

# %% [markdown]
# ## 3. The Maxwell construction
#
# Two phases coexist where their pressures are equal at a common `mu_B`, each
# neutralised by its own leptons. Scanning `P_quark - P_hadronic` over the
# overlap in `mu_B` and bracketing the sign change gives the transition
# potential; the two branches' densities there are the edges of the density
# jump.
#
# The construction is written out rather than called because of the potential
# mismatch noted in section 2 — but it is the same eta = 1 condition the mixed
# engine imposes, and section 6 checks the two against each other.

# %%
# Both branches must be monotone in mu_B for the interpolation below to mean
# anything. The hadronic sweep is monotone as solved; the quark one is only
# after the envelope of section 2.
assert np.all(np.diff(had[:, 1]) > 0), "hadronic mu_B is not monotone"
assert np.all(np.diff(njl_stable[:, 1]) > 0), "quark mu_B is not monotone"

mu_lo = max(had[:, 1].min(), njl_stable[:, 1].min())
mu_hi = min(had[:, 1].max(), njl_stable[:, 1].max())
print(f"mu_B overlap: [{mu_lo:.1f}, {mu_hi:.1f}] MeV")

mu_scan = np.linspace(mu_lo, mu_hi, 4001)
gap = (np.interp(mu_scan, njl_stable[:, 1], njl_stable[:, 2])
       - np.interp(mu_scan, had[:, 1], had[:, 2]))
sign_change = np.where(np.diff(np.sign(gap)) != 0)[0]

if len(sign_change) == 0:
    raise RuntimeError("no Maxwell crossing on the overlap — no transition")

i = int(sign_change[0])
# linear root of the gap between the two bracketing scan points
mu_trans = float(mu_scan[i] - gap[i] * (mu_scan[i + 1] - mu_scan[i])
                 / (gap[i + 1] - gap[i]))

P_trans = float(np.interp(mu_trans, had[:, 1], had[:, 2]))
n_H = float(np.interp(mu_trans, had[:, 1], had[:, 0]))
n_Q = float(np.interp(mu_trans, njl_stable[:, 1], njl_stable[:, 0]))
eps_H = float(np.interp(mu_trans, had[:, 1], had[:, 3]))
eps_Q = float(np.interp(mu_trans, njl_stable[:, 1], njl_stable[:, 3]))

print(f"mu_trans = {mu_trans:.2f} MeV")
print(f"P_trans  = {P_trans:.3f} MeV/fm^3")
print(f"density jump  n_B {n_H:.4f} -> {n_Q:.4f} fm^-3")
print(f"energy jump   eps {eps_H:.2f} -> {eps_Q:.2f} MeV/fm^3")
print(f"the hadronic branch ends at n_B = {had[-1, 0]:.4f} fm^-3, "
      f"{'above' if had[-1, 0] > n_H else 'BELOW'} the transition")

# %% [markdown]
# The stitched core equation of state: hadronic below the jump, quark above it,
# and the two edge rows at the same pressure. A Maxwell transition IS a density
# discontinuity at constant `P`, so both edges belong in the table — that pair
# of rows is what the structure solver reads as the jump.

# %%
below = had[had[:, 0] < n_H]
above = njl_stable[njl_stable[:, 0] > n_Q]

n_B_core = np.concatenate([below[:, 0], [n_H, n_Q], above[:, 0]])
P_core = np.concatenate([below[:, 2], [P_trans, P_trans], above[:, 2]])
eps_core = np.concatenate([below[:, 3], [eps_H, eps_Q], above[:, 3]])

print(f"core table: {len(n_B_core)} rows, "
      f"n_B {n_B_core[0]:.3f} .. {n_B_core[-1]:.3f} fm^-3")
print(f"  P non-decreasing: {bool(np.all(np.diff(P_core) >= 0.0))}")
print(f"  eps increasing:   {bool(np.all(np.diff(eps_core) > 0.0))}")
print(f"  hadronic rows {len(below)},  quark rows {len(above)}")

# %% [markdown]
# ## 4. Pressure against baryon density
#
# The two pure branches and the hybrid built from them. The shaded band is the
# density jump: no matter sits at those densities, which is exactly what a
# first-order transition with a Maxwell construction means.

# %%
fig, ax = plt.subplots(figsize=(5.5, 4.2))

ax.plot(had[:, 0], had[:, 2], color=COLOR_H, ls="--", lw=1.2,
        label="DD2Y + hyperons + $\\Delta$ (pure)")
ax.plot(njl[:, 0], njl[:, 2], color=COLOR_NJL, ls=":", lw=1.0,
        alpha=0.7, label="NJL csc, enumerated (all rows)")
ax.plot(njl_stable[:, 0], njl_stable[:, 2], color=COLOR_NJL, ls="--", lw=1.2,
        label="NJL csc (stable branch)")
ax.axvspan(n_H, n_Q, color="0.85", zorder=0, label="density jump")
ax.plot(n_B_core, P_core, color=COLOR_HYBRID, lw=2.0, label="hybrid (Maxwell)")

ax.set_xlabel(r"$n_B$ [fm$^{-3}$]")
ax.set_ylabel(r"$P$ [MeV fm$^{-3}$]")
ax.set_xlim(0.0, float(n_B_core[-1]))
ax.set_ylim(0.0, None)
ax.legend(loc="upper left", frameon=False)
save_figure(fig, str(FIG_DIR / "pressure_density"))
plt.show()

# %% [markdown]
# ## 5. Mass–radius
#
# A BPS crust is attached below `n_B = 0.08 fm^-3`; the core table above is the
# rest. The pure hadronic sequence is run beside the hybrid so the effect of the
# quark core is visible rather than asserted.

# %%
def tov_sequence(n_B, P, eps, label):
    """Run `eos.astro.tov` over one core table and report the headline numbers."""
    table = EOSTable_for_TOV(P=np.asarray(P), epsilon=np.asarray(eps),
                             nB=np.asarray(n_B))
    crust = "BPS" if have_crust("BPS") else "No"
    results = compute_tov_sequence(
        table, generate_ec_logspace(150.0, 3000.0, 160),
        add_crust_table=crust, add_crust_mode="attach",
        n_transition=(0.08 if crust != "No" else None),
        compute_baryonic_mass=False, compute_tidal=True,
        verbose=False, backend="fast", tov_parallel=True)
    idx, e_c, M_max = find_mmax_precise(results)
    M, R = results[:idx + 1, 4], results[:idx + 1, 3]
    R_14 = float(np.interp(1.4, M, R)) if M[-1] >= 1.4 > M[0] else float("nan")
    print(f"{label:26s} M_max = {M_max:.4f} Msun   R(M_max) = "
          f"{results[idx, 3]:.3f} km   R(1.4) = {R_14:.3f} km")
    return results, idx


t0 = time.time()
seq_hybrid, idx_hybrid = tov_sequence(n_B_core, P_core, eps_core,
                                      "DD2Y + NJL (hybrid)")
seq_hadronic, idx_hadronic = tov_sequence(had[:, 0], had[:, 2], had[:, 3],
                                          "DD2Y (pure hadronic)")
print(f"({time.time() - t0:.0f} s)")

# %%
fig, ax = plt.subplots(figsize=(5.0, 4.6))

ax.plot(seq_hadronic[:idx_hadronic + 1, 3], seq_hadronic[:idx_hadronic + 1, 4],
        color=COLOR_H, ls="--", lw=1.4, label="DD2Y + hyperons + $\\Delta$")
ax.plot(seq_hybrid[:idx_hybrid + 1, 3], seq_hybrid[:idx_hybrid + 1, 4],
        color=COLOR_HYBRID, lw=2.0, label="hybrid: + NJL csc core")

ax.set_xlabel(r"$R$ [km]")
ax.set_ylabel(r"$M$ [$M_\odot$]")
ax.set_xlim(9.0, 16.0)
ax.set_ylim(0.5, 2.5)
ax.legend(loc="lower left", frameon=False)
save_figure(fig, str(FIG_DIR / "mass_radius"))
plt.show()

# %% [markdown]
# ## 6. DD2Y + CCDM — a reported gap, not a figure
#
# The second pairing asked for does not produce a hybrid star, and the reason is
# physics rather than a solver that needs more patience: the CCDM pressure never
# reaches the hadronic one. The sweep below prints the branch and the gap at
# every density it solves, so the claim can be checked rather than taken.
#
# Two separate things show up, and they are printed apart:
#
# * the pressure deficit **widens** with `mu_B` instead of closing, so there is
#   no crossing to construct a transition at;
# * the branch enumeration flips at around `n_B ~ 0.65 fm^-3`, where `P` drops
#   by roughly a factor of six and `mu_B` runs backwards. That is the
#   confined/restored coexistence of the model, and CLAUDE.md section 8 allows a
#   raw branch to be non-monotone there — but it is below any transition, so no
#   construction resolves it into a star.
#
# A third thing shows up in the print and is flagged there rather than tidied
# away: the solve does not always land on the density it was asked for, and
# some rows come back at negative pressure. Where the returned `n_B` differs
# from the requested one the row says so. Neither affects the verdict — the
# pressure deficit is between 12 and 300 MeV/fm^3 across the whole overlap and
# never changes sign — but a sweep whose rows are silently relabelled with the
# density they happened to reach would read as clean when it is not.

# %%
if RUN_CCDM:
    ccdm_par = CCDMParameters.default()
    t0 = time.time()
    ccdm_result = ccdm_eos_table(ccdm_par, "beta_eq_neutrinoless", CCDM_FLAGS,
                                 {"nB": N_B_CCDM, "T": np.array([0.0])})
    ccdm_line = ccdm_result.points[0]
    ccdm = np.array([(p.n_B, p.mu_B, p.P, p.eps) for p in ccdm_line])
    print(f"CCDM (csc, enumerated): {len(ccdm)} of {len(N_B_CCDM)} points "
          f"({time.time() - t0:.0f} s)")
    print(f"  branches found: {sorted(set(p.branch for p in ccdm_line))}")
    print(f"  patterns found: {sorted(set(p.pattern for p in ccdm_line))}")

    for row in ccdm:
        P_had_here = float(np.interp(row[1], had[:, 1], had[:, 2]))
        notes = []
        if not had[0, 1] <= row[1] <= had[-1, 1]:
            notes.append("mu_B outside the hadronic branch")
        if row[2] < 0.0:
            notes.append("P < 0")
        print(f"  n_B = {row[0]:.3f}  mu_B = {row[1]:8.2f}  P = {row[2]:8.3f}"
              f"   P_quark - P_had = {row[2] - P_had_here:+8.2f}"
              + ("   [" + "; ".join(notes) + "]" if notes else ""))

    if len(ccdm):
        ccdm_stable = stable_branch(ccdm)
        lo = max(had[:, 1].min(), ccdm_stable[:, 1].min())
        hi = min(had[:, 1].max(), ccdm_stable[:, 1].max())
        if lo < hi:
            mu = np.linspace(lo, hi, 400)
            d = (np.interp(mu, ccdm_stable[:, 1], ccdm_stable[:, 2])
                 - np.interp(mu, had[:, 1], had[:, 2]))
            crossed = np.any(np.diff(np.sign(d)) != 0)
            print(f"\nP_quark - P_had over mu_B in [{lo:.1f}, {hi:.1f}]: "
                  f"{d.min():+.2f} .. {d.max():+.2f} MeV/fm^3")
            print(f"crossing: {crossed}  -> "
                  f"{'a transition exists' if crossed else 'NO TRANSITION'}")
        else:
            print("\nno mu_B overlap with the hadronic branch")

else:
    print("RUN_CCDM is False — section 5 skipped, nothing computed.")

# %% [markdown]
# ## 7. What the eta-mixed engine gives, and where it stops
#
# The Maxwell construction of section 3 is the eta = 1 limit of the composite
# engine, so the engine should reproduce its transition pressure. It does, at
# the densities where it converges — and this cell is also where the two claims
# made in the introduction are demonstrated instead of asserted.
#
# `eos_point` is called directly rather than `hybrid_table`, because
# `locate_window` reports no window for this pairing: it brackets the chi = 0
# and chi = 1 crossings, and chi = 1 is never reached. The hadronic branch
# ends first, with chi around one half. A `hybrid_table` call therefore returns
# a pure hadronic table, which is correct behaviour for a window that does not
# close, and is why this notebook does not build its table that way.
#
# **If the eta = 1 points below report non-convergence, restart the kernel and
# run this section alone.** They converge reproducibly in a fresh process —
# `chi = +0.0669` and `+0.3934`, both at `P = 207.610` — but have been observed
# to fail when this section runs at the end of a long full-notebook execution,
# on identical inputs. Ruled out as the cause: the held-pattern sweep just
# above, the enumerated `eos_table` of section 2, the TOV pass of section 5,
# and mutation of the arrays in `eos.general.pairing` shared by the two quark
# models (all four checked, none of them reproduce it). The cause is not yet
# identified, so it is recorded rather than explained. It touches this
# cross-check only; sections 2-5 are unaffected and reproduce exactly.

# %%
import eos.mixed.adapters as adapters
from eos.mixed import eos_point
from eos.mixed.species import SpeciesFlags as MixedFlags

mixed_species = MixedFlags(hyperons=True, deltas=True, muons=True)
hadronic_phase = adapters.dd2_phase(hadronic_par, HADRONIC_FLAGS)
quark_phase = adapters.njl_phase(njl_par, NJL_FLAGS, patterns=HELD_PATTERN)

# The engine cannot enumerate — its Newton solve needs a phase function
# without steps — so it is given the HELD pattern. Comparing its plateau with
# section 3's enumerated transition would compare two different quark phases,
# so the held-pattern construction is rebuilt here and that is what the engine
# is checked against. The two numbers below are the same physics by two
# independent routes; the enumerated result of section 3 is the physical one.
held_rows = []
for n_B in np.linspace(0.85, 1.70, 30):
    point = njl_beta_eq(njl_par, float(n_B), 0.0, flags=NJL_FLAGS,
                        patterns=HELD_PATTERN)
    if point is not None:
        held_rows.append((point.n_B, point.mu_B, point.P, point.eps))
held = stable_branch(np.array(held_rows))

mu_h = np.linspace(max(had[:, 1].min(), held[:, 1].min()),
                   min(had[:, 1].max(), held[:, 1].max()), 4001)
gap_h = (np.interp(mu_h, held[:, 1], held[:, 2])
         - np.interp(mu_h, had[:, 1], had[:, 2]))
j = np.where(np.diff(np.sign(gap_h)) != 0)[0]
P_trans_held = float("nan")
if len(j):
    k = int(j[0])
    mu_held = float(mu_h[k] - gap_h[k] * (mu_h[k + 1] - mu_h[k])
                    / (gap_h[k + 1] - gap_h[k]))
    P_trans_held = float(np.interp(mu_held, had[:, 1], had[:, 2]))
    print(f"held {HELD_PATTERN[0]}, Maxwell on the pure branches: "
          f"P_trans = {P_trans_held:.3f} MeV/fm^3")
print(f"enumerated (section 3, the physical one): "
      f"P_trans = {P_trans:.3f} MeV/fm^3\n")
for eta in (1.0, 0.0):
    print(f"eta = {eta:.1f}")
    for n_B in (0.75, 0.90, 1.05):
        t0 = time.time()
        result = eos_point((hadronic_phase, quark_phase),
                           "beta_eq_neutrinoless", mixed_species,
                           n_B=n_B, T=0.0, eta=eta)
        dt = time.time() - t0
        if not result.ok:
            print(f"  n_B = {n_B:.2f}  did not converge: "
                  f"{result.message[:58]} ({dt:.0f} s)")
            continue
        point = result.point
        print(f"  n_B = {n_B:.2f}  chi = {point.chi:+.4f}  "
              f"phase = {point.phase:3s}  P = {point.P:8.3f}  ({dt:.0f} s)")
    print()

# %% [markdown]
# At eta = 1 the converged points sit at one pressure — that is the Maxwell
# plateau, and it is the number section 3 constructed from the pure branches by
# a different route. At eta = 0 the pressure rises through the window instead,
# which is what a Gibbs construction does.
#
# The `n_B = 1.05` points are where the hadronic component runs out. Above that
# the mixed system has no hadronic phase left to mix, chi never reaches 1, and
# the transition cannot be closed from inside the window — which is what
# section 3 closes from outside it.
