# A coexistence locator for a phase held at fixed (Y_C, Y_S)

Type: build
Status: open
Blocked by: -
Parent: ../map.md

## Question

Split out of [ticket 83](83-enjl-branch-selection-physics.md), which measured
the window and argued the design but deliberately built neither. **Non-gating**:
nothing in the Acceptance criteria block of `docs/REFACTOR_PROMPTS.md` measures
it, and 83's guard means the gap now reports itself rather than shipping a
wrong table.

`eos.mixed.construction.enjl_coexistences` locates the transitions of the
BETA-EQUILIBRIUM branch pair only: it closes each phase with `neutral_phase`,
which solves mu_C so the phase neutralizes its own leptons. A leptonless phase
at a held (Y_C, Y_S) — what CLAUDE.md §3 says a mixed-phase construction
consumes, and the mode heavy-ion comparisons are made in — has no located
window, so `build_constructed_table` receives an empty list and, under 83's
change, correctly reports `deliverable = False`. This ticket makes it
deliverable.

## The target

Measured under ticket 83 on python.org 3.14.2 / numpy 2.3.5, `fixed_YC_YS`
Y_C = 0.5, Y_S = 0, `leptons=False`, T = 0, `Parameters.default()`, by cubic
splines through 0.01-spaced solves of both sweeps:

| | n_B [fm^-3] | P [MeV/fm^3] | g [MeV] | eps [MeV/fm^3] |
|---|---|---|---|---|
| broken   | 0.34945 | 22.9282 | 1006.4074 | 328.7573 |
| restored | 0.47500 | 22.9282 | 1006.4074 | 455.1121 |

Acceptance: the locator reproduces this window, `build_constructed_table` on
the same mode returns `deliverable = True` with a plateau across it, and the
eps crossing at 0.41774 fm^-3 — what min-eps picks — lies inside it.

Those numbers came off a DENSITY grid and are spline-interpolated, which is
exactly the accuracy `locate_maxwell` was written to avoid; they are the
target to within that, not a golden reference.

## What has to be decided

1. **The locator is new, not a second closure on the old one.**
   `locate_maxwell` bisects `gap(mu_B)`, one variable, and that is only
   possible because beta equilibrium with neutrality determines mu_C from
   mu_B — which makes the Gibbs free energy per baryon
   `g = (eps + P)/n_B = mu_B + Y_C mu_C + Y_S mu_S` reduce to mu_B, leaving
   equal-P as the only remaining condition. At a held (Y_C, Y_S) each phase
   carries its own (mu_C, mu_S), solved for its own fractions, `g != mu_B`,
   and coexistence needs equal P AND equal g: two conditions in two unknowns,
   the two branches' own mu_B. A 2-D root find.

   It still belongs in `eos/mixed/boundaries.py` beside the beta one, and for
   the same reason: getting a branch at a chosen potential means
   `enjl_branch_pair`, and a model has no vocabulary for "the restored
   branch". Locating from the model's own two sweeps instead would import
   nothing downstream and was rejected — it locates on a density grid, and the
   spline edges above are what that costs.

2. **`Coexistence` encodes the beta closure in its FIELD LIST.** `mu_B` is one
   field, documented as "equal across the two phases by construction -- that
   IS the Maxwell condition", and `plateau_row` writes `co.mu_B` onto every
   plateau row. Under the composition closure mu_B is not equal across the
   edges; g is. Settled while grilling 83, recorded here so it opens argued
   rather than open: **generalize the one carrier** — `P` and `g` the shared
   pair, `mu_B_lo`/`mu_B_hi` replacing `mu_B`, `mu_e_lo`/`mu_e_hi` nan under
   the composition closure the way `plateau_row` already nans `mu_e` on a
   plateau. A second dataclass would duplicate six fields; a carrier whose
   fields assert a condition only one closure satisfies is the same defect
   83 fixed in `build_constructed_table`'s promise. This touches the beta
   path, which is the whole reason it is not part of 83.

3. **Which fractions, and does eta mean the same thing?** The construction
   above holds (Y_C, Y_S) LOCALLY in each phase — the analogue of eta = 1,
   each phase separately neutral. The eta = 0 / Gibbs alternative lets the two
   phases carry different Y_C and Y_S subject to the global values, equates
   all three potentials as well as P, and gives a window whose pressure is not
   flat. Whether the eta argument spans both, or the composition closure is
   eta = 1 only with the rest deferred as it is for the beta path
   (`docs/DEFERRED.md`), is this ticket's.

4. **`docs/DEFERRED.md`.** The gap is not recorded there today. It sits
   alongside the two entries the beta construction already owns — the eta < 1
   table and the T > 0 locator — and belongs in the same place, written as
   what is missing rather than as what raises: nothing raises here, the
   delivered table reports `deliverable = False`. That entry is this ticket's,
   not [83](83-enjl-branch-selection-physics.md)'s.

5. **§11 text.** `eos/enjl/enjl.md` and `.tex` state the construction at
   §§ around the delivered table; a second closure means a second set of
   coexistence conditions written out, with the residual rows the 2-D solve
   assembles and its unknown vector. The `.tex` must still compile.

## Not in scope

The branch-selection question and the delivery gate. Both are
[ticket 83](83-enjl-branch-selection-physics.md), resolved: the raw
continuation maps a branch and owes no selection, min-eps selects the stable
PURE phase and is correct only outside a window, and
`ConstructedTable.deliverable` now reports when a window went unlocated.
