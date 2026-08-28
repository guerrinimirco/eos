# A coexistence locator for a phase held at fixed (Y_C, Y_S)

Type: build
Status: resolved
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


## Resolution

Built, `9bf61ec`. The window reproduces ticket 83's measurement to every digit
it reported: **n = [0.34945, 0.47500] fm^-3 at P = 22.9282 MeV/fm^3 and
g = 1006.4074 MeV**, with the eps crossing at 0.41774 inside it, and
`build_constructed_table` on `fixed_YC_YS` / Y_C = 0.5 / Y_S = 0 / leptonless
returns `deliverable = True` with a 13-row plateau flat to `ptp(P) = 0`. That
is the acceptance, met.

### 1. The locator — and the 2-D solve separates, which is why it is cheap

Built as three functions, `composition_phase` and
`locate_maxwell_composition` in `eos/mixed/boundaries.py` beside their beta
twins, `enjl_composition_coexistences` in `construction.py` beside
`enjl_coexistences`. The ticket's finding held: `locate_maxwell` cannot take
this closure, and coexistence here is equal P **and** equal g in the two
branches' own mu_B.

One finding the ticket did not have. **The 2-D system separates, and the
separation is what seeds it.** Gibbs-Duhem at fixed composition and T = 0 is
`n_B dg = dP` with `n_B > 0`, so g is monotone along a branch. Tabulating
`(mu_B, g, P)` on the mu_B grid for each branch and reading each branch's mu_B
and P at a COMMON g turns the seed into a one-dimensional sign change of
`P_lo - P_hi` in g — the same shape `locate_maxwell` has, recovered one level
down. That interpolated crossing is a grid-resolution answer, which is the
accuracy this module exists to beat, so it is the starting point and the 2-D
`root` polishes it. Cost: **7 s on a 5-point mu_B grid**, and the located
window is identical on a 6-point, a 5-point and a 60-point grid, which is the
evidence that the answer is not a grid artefact.

Two smaller things, both measured rather than assumed:

- **The acceptance test is the residual, not `sol.success`.** At Y_C = 0.5 the
  composition residual `n_C - Y_C n_B` is exactly zero at mu_C = 0 by u-d
  symmetry, so `hybr` starts AT round-off and returns status 5, "not making
  good progress", sitting on `x = [0.0]` with `fun = 2.8e-17`. Gating on the
  flag rejected the restored branch at exactly the coexistence potential and
  nowhere else. CLAUDE.md section 6 judges convergence on a residual norm, and
  that is the test that reads this situation correctly; `enjl.md`'s Numerics
  section already says the same thing about the gap solve.
- **mu_S is dropped from the unknowns where a held Y_S = 0 leaves its row
  reading 0 = 0.** Every strangeness carrier in this repository has S = +1 and
  at T = 0 there are no antiparticles, so `n_S = 0` forces each term to vanish
  separately, for any mu_S. This is `eos.enjl.solver.strangeness_row_is_empty`
  restated at the adapter boundary, and carrying the null column instead is
  the rank-deficiency ticket 72 measured.

### 2. The carrier is generalized, as ruled

`MaxwellPoint` and `Coexistence` now carry **P and g** — the pair coexistence
equates in EITHER closure, and the only two single-valued fields — with
`mu_B_lo`/`mu_B_hi` beside them and `mu_e_lo`/`mu_e_hi` nan where the phases
carry no leptons. `plateau_row` writes mu_B and mu_S only where the two edges
agree on them.

That equality is tested exactly, and it is not fragile: under the beta closure
mu_B is ONE number handed to both phases, a bitwise identity; under the
composition closure the two are independent unknowns, and where they do
coincide — a symmetric composition puts mu_C = mu_S = 0 on both sides, which is
g = mu_B again — the value written is true. **Both outcomes of the test are
true statements about the mixture**, which is what makes an exact comparison
the right one here.

Five readers followed: `enjl_coexistences` (now re-solves each edge at its own
mu_B), `check_maxwell_crossing`, `test_enjl_construction.py`,
`test_locate_maxwell.py`, `test_enjl_pair.py`. No notebook read the field.

### 3. eta: eta = 1 only, as for the beta path

Ruled as item 3's second option. The composition closure holds (Y_C, Y_S)
LOCALLY in each phase, which is the eta = 1 analogue. The eta = 0 / Gibbs
alternative is a **different delivered object**, not a refinement: the two
phases would carry different Y_C and Y_S subject to the global values, all
three potentials would be equated rather than P and g alone, and the window
would not be flat in pressure. Recorded in `docs/DEFERRED.md` on the entry the
beta path already owns, which now states the bound for both closures.

### 4. docs/DEFERRED.md

Two entries extended rather than one added, because the gaps are the same gaps:

- the eta = 1 entry now says "under either closure" and states why eta = 0
  differs MORE at held composition than in beta equilibrium;
- the T > 0 entry names the new functions and adds the two corrections T > 0
  owes this closure specifically: the s term in g (the change of variable is
  already made here, since g = mu_B is a beta-closure identity), and mu_S
  becoming determined once Fermi tails populate the strangeness carriers and
  their antiparticles.

### 5. §11 text

`eos/enjl/enjl.md` and `.tex` gain "A second closure: a phase held at fixed
(Y_C, Y_S)", stating (coexcomp), (gibbs), (coexPcomp), (coexgcomp), the
residual rows (coexrescomp) with their unknown vector, why mu_S is dropped, the
plateau's one change (mu_B joins the nan quantities) and (epscomp)
`eps = g^co n_B - P^co`. The `.tex` compiles, 16 pages, no undefined
references.

### Verification

    ENJL run_full_check: PASS, 20 checks (was 18)
      composition_crossing            max_err=9.48e-14
        broken+restored dP/P=9.5e-14 dg/g=8.6e-15 dEuler/g=8.1e-15
        n=[0.34945, 0.47500]
      composition_delivered_table     max_err=0.00e+00
        windows: deliverable=True (want True), min dP=+0.00e+00
        none:    deliverable=False (want False), P falls by 46.473 MeV/fm^3
                 between n_B = 0.4100 and 0.4200 fm^-3
        plateau 13 rows, ptp(P)=0.0e+00, eps residual to a line 1.7e-13

`composition_crossing` recomputes P and g from the phases at the located
potentials rather than reading back the record that asserted them, and checks
the Euler identity `g = mu_B + Y_C mu_C + Y_S mu_S` at both edges with each
side's own potentials — which is what says the located g is the thermodynamic
one and not the same numbers assembled twice.
`composition_delivered_table` runs the section 8 gate **in both directions**,
as ticket 83's did: without the empty-list row, "the construction delivers a
table" and "any table is delivered" are the same observation.

    pytest test/enjl test/mixed -q      391 passed, 0 failed (18m21s)
    python.org 3.14.2 / numpy 2.3.5 / scipy 1.17.0

A concurrent session was mid-refactor on `eos/zl` and `eos/mixed/adapters.py`
throughout; nothing here touches those files and the commit used explicit
pathspecs.

### Files

- `eos/mixed/boundaries.py`, `eos/mixed/construction.py`, `eos/mixed/__init__.py`
- `eos/enjl/table.py`, `eos/enjl/api.py`, `eos/enjl/verify/run_full_check.py`
- `eos/enjl/enjl.md`, `eos/enjl/enjl.tex`, `docs/DEFERRED.md`
- `test/mixed/test_locate_maxwell.py` (three cases: the measured window, that
  the two closures locate DIFFERENT windows, and no-crossing reported not
  invented), `test/enjl/test_enjl_construction.py`, `test/mixed/test_enjl_pair.py`
