# notebooks/hybrid_eos — figures, the TOV pass, and the swap cell

Type: task
Status: resolved
Blocked by: 58
Parent: ../map.md

## Question

Three parts.

1. **Figures** for the headline DD2 + vMIT construction: P vs n_B with the
   window marked, the quark volume fraction chi across the transition, the
   per-phase decomposition of each conserved charge, and the transition curves
   n_onset / n_offset. These are the composite engine's own observables (§5) and
   are why `mixed` earned a notebook.

2. **The TOV pass.** `HybridResult.table.to_tov()` is the declared contract into
   `eos.astro.tov`, and `mixed/hybrid.py` and `mixed/scan.py` already import it —
   the one §1 exception. End on M–R. Run §8's gate (P non-decreasing,
   0 <= c_s^2 <= 1) BEFORE integrating and report its status, never a mass
   computed past a failed gate.

3. **The swap cell.** Re-run with **DID + NJL** and **DID + CCDM**, changing both
   sides of the pair at once. Depth is a runtime call — a converged table is the
   floor, the full headline treatment is not required. Whatever is skipped for
   runtime is printed, not silently dropped.

**The comparison ticket 05 promised is now OPTIONAL, and its target has moved.**
Ticket 05 held `notebooks/eos_tables_DD2vMIT/` — 32 tables, 42 figures from the
retired `DD2vMIT_general1oPT.ipynb` — because a replacement had to be measured
against it rather than asserted. Two things have since changed: **the user has
confirmed none of the 42 figures is published**, so nothing downstream depends on
reproducing them exactly; and the folder left the repo with `output_old/`, so its
current path is not known here.

So ticket 03's held-until condition is **discharged**: nothing is waiting on this
notebook to regenerate anything. If the user can point at where `output_old/`
went, an eyeball comparison of the DD2+vMIT figures is still worth ten minutes —
it is the cheapest end-to-end check that the engine gives the same physics it
gave before the refactor. If they cannot, say so and move on; do not hunt for it.

Resolved when the notebook executes end to end and the comparison is reported.

## Answer

**Shipped as sections 8-11 of `notebooks/hybrid_eos.py`**, commit `156384f`,
executed clean end to end: 26 code cells, 0 error outputs.

### 1. Figures — four families, and the panels ARE the eta selection

A construction is not a coordinate (section 5 of the notebook), so a figure
that wants Gibbs beside Maxwell is one file with a panel each, not two files
to line up by eye. Every family is drawn over `FIG_ETAS = (0.0, 0.5, 1.0)` on
a 60-point `n_B` grid at T = 0.

* **`pressure_window`** — P against n_B, with the coexistence window shaded
  from the `n_onset`/`n_offset` the engine *returned*, never recovered by
  scanning the curve for where chi left [0, 1]. The Maxwell limit is visible
  rather than asserted: the curve is flat across the span at eta = 1 and rises
  through it at eta = 0.
* **`quark_fraction`** — chi across the transition, flat outside the window by
  construction because the stitched table pins it at its pure-phase value.
* **`charges_by_phase`** — one panel per conserved charge, the hadronic and
  quark shares solid and their sum dashed. The C panel is the figure that
  earns the family: at eta = 0.5 the two phases carry charge of OPPOSITE SIGN
  while the sum is the fraction the mode fixed. A global sum cannot show that,
  and it is what eta controls.
* **`transition_curves`** — n_onset and n_offset against T over
  (0, 10, 20, 30) MeV, the pairing's phase diagram in the plane the windows
  live in. Twelve of twelve lines located a window. A line that could not
  would contribute no point rather than a fabricated one.

All styling is from `eos/general/figure_style.py` and nothing else — no
rcParams set in the notebook, no colour re-declared — through `paper_grid`,
`panel_label` and `save_figure`; the M-R bands come from
`eos/general/constraints` `overlay(ax, "M-R")`. §10 holds.

Three legibility fixes were needed and are worth recording, because two of
them are properties of the physics rather than of matplotlib: on M-R the three
constructions agree to within 0.1 km below M_max, so colour alone hid two
curves behind whichever was drawn last and a line style per eta was added; and
in the charge panels the automatic label corner landed on a curve.

### 2. The TOV pass — the gate returns a status, and no mass stands in for a failure

`.to_tov()` is the declared contract into `eos.astro.tov` — an
`EOSTable_for_TOV`, which lives in `general/`, the layer both may import.

`deliverable(core)` runs BEFORE any integration and returns `(ok, message,
cs2)`, modifying nothing. One decision inside it is not cosmetic: **P is
tested as non-decreasing, not as strictly rising.** A Maxwell window is an
exact plateau, so a strict test would reject precisely the construction that
is most clearly correct — which is why §8 words the invariant that way, and
this is the notebook where it bites.

`c_s^2` is the finite-difference `dP/deps` of the delivered table itself,
which is the quantity the structure solver will interpolate. It is *not* the
model's `eos_response`, which holds something fixed and answers a different
question. Framing it that way keeps this notebook out of the sound-speed
naming question entirely.

All three constructions PASS over 59 steps, max c_s^2 = 0.494 / 0.636 / 0.658.
Then M_max = 2.249 / 2.339 / 2.343 M_sun at R = 12.38 / 12.64 / 12.64 km, BPS
crust attached at n_B = 0.08 fm^-3, stable branch sliced at
`find_mmax_precise`. A construction that had not passed would simply have no
entry in `sequences` — the absence is visible in the printed gate line and no
number is invented for it.

### 3. The swap cell — both sides changed at once

DID + NJL and DID + CCDM, neither sharing a model with the headline. Both
converge to a table and its windows at T = 0:

* `did+njl` — n_onset = 1.1818, n_offset = 1.2875 fm^-3, 2 in-window rows.
* `did+ccdm` — n_onset = 0.7810, n_offset = 1.4000 fm^-3, 13 rows. The offset
  sits exactly on the grid ceiling, so the window is open at the top on this
  grid rather than closed at 1.4.

Depth was a runtime call and what was skipped is PRINTED before the results,
naming the four figure families, the boundary curves in T and the TOV pass, so
a reader cannot mistake the shallower treatment for a completed one.

### The comparison: the target was NOT lost, and the check is numeric

**This ticket's premise was stale.** The retired first-generation tables did
not leave with `output_old/`: they are tracked in the repository at
`eos_tables_DD2vMIT/` — 16 files plus 21 figures — and are in HEAD. Nothing had
to be hunted for.

Better still, no eyeball was needed. Each of those CSVs carries the run's full
provenance in a `# key = value` header: every coupling of the hadronic
parametrisation, the vMIT bag constant and quark masses, and every species
flag. The header keys map 1:1 onto `eos.dd2.Parameters`' field names, so
section 11 rebuilds the exact inputs and asks the present engine for the same
boundaries. That matters because the parametrisation is a CUSTOM set, not a
published one — `gamma_sigma = 10.686850` and `n_sat = 0.149077` differ from
`DD2Y`'s in the fourth digit, and `B4 = 170` against the shipped default 180 —
so guessing at it would have produced a meaningless comparison that looked
like a real one.

The engine reproduces the retired boundaries at T = 0 in beta equilibrium
across all three eta that run was built at, to under half a percent:

| eta | n_onset now | n_offset now | n_onset was | n_offset was | diff |
|---|---|---|---|---|---|
| 0.0 | 0.332477 | 1.078811 | 0.331029 | 1.078310 | 0.44% / 0.05% |
| 0.3 | 0.878653 | 0.993398 | 0.876717 | 0.997672 | 0.22% / 0.43% |
| 1.0 | 0.885372 | 0.964581 | 0.883485 | 0.966426 | 0.21% / 0.19% |

The boundaries of a first-order transition are the most sensitive thing the
engine computes — they depend on the hadronic sector, the quark sector and the
coupling between them all at once — so agreement at this level is a strong
end-to-end statement that the refactor preserved the physics, and a much
stronger one than comparing figures by eye would have been. Only the
boundaries are compared, because they are what the retired run recorded per
line in its `completeness.csv`.

The residual sub-percent difference is not chased here: it is the resolution
of the boundary-locating solve, not a disagreement about the physics.

### Paths

Every written path is anchored to the `ROOT` found in the notebook's first
cell — figures to `output/hybrid/`, tables to `output/tables/dd2vmit/`. Nothing
lands under `notebooks/output/`, confirmed after the run.

### How it was verified

`jupytext --to notebook --execute` under python3 3.14.2 (matplotlib 3.10.9) in
an isolated `git archive HEAD` copy, so no concurrent session's working-tree
edits could contribute: 26 code cells, 0 errors. The five figures were opened
and read, not merely confirmed to exist. The committed `.ipynb` is a second
clean execution in the repository itself, so its stored paths are the real
ones, and it round-trips to the `.py` identically. The full pytest suite was
not run: no library file was touched.

Status: resolved.
