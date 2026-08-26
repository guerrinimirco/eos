# ENJL's fixed_YC_YS continuation picks its chiral branch by warm start, not by physics

Type: grilling
Status: resolved
Assignee: session e393a47c
Blocked by: -
Parent: ../map.md

## Question

Found by [ticket 62](62-regenerate-baselines-py314.md) while regenerating the
baselines: `enjl` is the one model of thirteen whose 3.9 -> 3.14 difference is
**not** round-off, and the mechanism is a branch selection nothing in the model
adjudicates.

`fixed_YC_YS` at Y_C = 0.5, Y_S = 0, `leptons=False`, over the warm-started
density sweep:

| n_B [fm^-3] | 3.9: M_q.u | 3.9: P | 3.14: M_q.u | 3.14: P | eps lower on |
|---|---|---|---|---|---|
| 0.2667 | 268.99 | 7.52 | 268.99 | 7.52 | (identical) |
| 0.3000 | 260.23 | 12.45 | 49.02 | -41.10 | broken (3.9) |
| 0.3333 | 251.56 | 19.06 | 62.23 | -29.47 | broken (3.9) |
| 0.3667 | 242.84 | 27.58 | 5.50 | -25.71 | broken (3.9) |
| 0.4000 | 233.98 | 38.23 | 5.50 | -13.11 | broken (3.9) |
| 0.4333 | 224.91 | 51.18 | 5.50 | 1.49 | **restored (3.14)** |
| 0.4667 | 215.56 | 66.56 | 5.50 | 18.34 | **restored (3.14)** |
| 0.5000 | 5.50 | 37.62 | 5.50 | 37.62 | (identical) |

Six contiguous points, 454 moved keys, `converged = 1` on both sides, same
n_B, same n_C, same targets. The two stacks land in different basins of the
same gap equations and the continuation carries the choice forward until the
branches rejoin at n_B = 0.5.

**Neither answer is right across the window.** At T = 0 and fixed n_B the
stable root is the one with lower eps, and that crosses inside the window: the
broken branch is lower to n_B = 0.400, the restored branch from 0.433. So the
first-order chiral crossing sits near n_B ~ 0.41 fm^-3 and **both** baselines
ride a metastable branch past it — 3.9 too far up on the broken side, 3.14 too
far down on the restored side.

### What has to be decided

1. **Does the raw `enjl` branch owe branch selection at all?** CLAUDE.md §8
   says a raw model branch MAY violate monotonicity inside a first-order
   region and that a construction resolves it before a table reaches TOV. That
   permits a metastable branch. It does not obviously permit the branch being
   chosen by which BLAS the warm start ran on.
2. **If it does: located how?** Comparing eps at each point costs one extra
   solve per point from the other basin, which the warm start already has the
   seed for. Whether that belongs in `solver.py`, in `table.py`'s continuation,
   or in a Maxwell/Gibbs construction alongside `eos/mixed` is the design half.
3. **What does `converged` mean here?** Both roots return `converged = 1`
   truthfully — each IS a root. Whether a returned point should also carry
   which branch it sits on, so a caller can see a discontinuity rather than
   infer it, is the reporting half.
4. **`enjl.npz` stays on 3.9 until this is settled**, so `test_baseline[enjl]`
   is red on the canonical stack and is expected to be. Re-freezing it on 3.14
   would record the metastable-restored answer as ground truth and delete the
   evidence. Whichever way 1-3 go, the regeneration follows this ticket.

### Related

The map's Not-yet-specified entry on a `general/verify/` differential check
notes that an undetermined potential shows as a shift in the ratio of `S_i` or
`C_i`. This case is the **negative** control for that screen and shows it
working: nothing here is proportional to any charge — masses, densities, P,
eps and mu_S all move by O(1) fractions — which is exactly how the screen was
supposed to separate a moved potential from moved physics.

## Resolution

**The ticket's own diagnosis was wrong, and the correct one is smaller.** There
is no branch selection to decide here: `mu_S` was an unknown no row determined,
and the ill-conditioning that caused put one mode's residual within round-off
reach of the acceptance gate, behind which `solve` falls through to a starting
point on the other chiral branch.

### The measurement that replaced the premise

Same seed, same equations, `fixed_YC_YS` at Y_C = 0.5, Y_S = 0, leptonless,
n_B = 0.3000 fm^-3, warm-started from 0.2667:

| stack | M_u reached | scaled residual | nfev | gate 1e-10 |
|---|---|---|---|---|
| anaconda 3.9.7 / numpy 1.26.4 | 260.2337 | 1.59e-12 | 94 | pass |
| python.org 3.14.2 / numpy 2.3.5 | **260.2337** | **1.20e-10** | 39 | **FAIL** |

Both stacks reach **the same root in the same basin**. "The two stacks land in
different basins of the same gap equations" is not what happens.
`least_squares` terminates 55 iterations earlier on 3.14 and leaves the
residual a factor 75 higher, straddling a hard threshold. `solver.solve` then
walks its seed list in order and stops at the first that clears the gate — and
seed 2 is `_restored_branch(x0)`, on the other branch by construction. A 20%
near-miss and an outright divergence get the identical response, with no
comparison between them. The continuation carries the switch forward to
n_B = 0.5, where the branches rejoin.

So ticket 62's trigger WAS round-off, sitting at a cliff. What was O(1) was the
consequence, not the cause.

### Why only this mode could reach the gate

Residual margins over the same 34-point grid, python.org 3.14.2:

    beta_eq_neutrinoless        2.7e-15
    fixed_YC   Y_C=0.1  lep     2.8e-15
    fixed_YC   Y_C=0.5  nolep   7.2e-14
    fixed_YC_YS Y_C=0.5 Y_S=0   1.7e-11   <- three decades worse

`mu_S` is unknown 11 of this mode, and **no strange species is populated at any
density of the sweep** — `n_s = n_Lambda = 0` throughout, `mu_S` sitting at
exactly its seed 0.0000 along the whole broken branch because no gradient ever
moves it. Both strangeness carriers have S = +1 (`species.STRANGENESS`) and
nothing has S < 0, and at T = 0 there are no antiparticles, so
n_S = 0 forces every strange density to zero term by term and the row
`n_S - Y_S n_B` reads 0 = 0 for every `mu_S`. That is a null column in the
Jacobian; the least-squares termination tests fire early on the rank-deficient
problem and leave the residual of the WHOLE solve three decades high.

It is a T = 0 statement only. At T > 0 the Fermi tails populate both species
and their S = -1 antiparticles, the row acquires a gradient, and `mu_S` is
determined in the ordinary way.

### The fix

`eos/enjl/solver.py`: `strangeness_row_is_empty(spec, T)` states the theorem;
where it holds, the empty row is replaced by the pin `mu_S = 0` (scaled as a
potential, 100 MeV, not a density). Eleven unknowns and eleven rows as before —
nothing about the vector shape or the public API changes.

Zero is not an arbitrary choice: it is the value the solve already returned
along the branch, and it changes no density, no eps, no P and no s — only the
reported `mu_S` and, through `mu_i = B_i mu_B + C_i mu_C + S_i mu_S`, the
potentials of the two species that are ABSENT.

Measured after the fix, all four modes, 34 points each:

- every residual between **1.3e-16 and 4.5e-15** — four decades clear of the gate
- `fixed_YC_YS` solves **34 points, not 33**: n_B = 0.1667 was previously
  unreachable and the frozen 3.9 file has no entry for it
- the broken branch is followed continuously to n_B = 0.5000, and the flip at
  0.5333 is **genuine**: the warm start there fails at 1.06e-03, seven orders
  out, not a hair. A continuation changing branch where its own branch ends is
  what `table.py` says should happen.

`BetaPoint.seed` now records which starting point produced the accepted root —
`"warm"`, `"restored"` or `"cold"`. No behaviour change; it makes a branch
switch readable off the result instead of inferable two sessions later. Every
mode now reports `"cold"` at the first density and `"restored"` at exactly one,
its real transition.

**A policy that ranks seeds by residual would have made this worse, not
better**, and it is worth recording why: at n_B = 0.3000 the restored seed
returns 8.08e-16 against the warm start's 1.59e-12, so "take the best seed"
selects the restored branch on 3.9 as well. The flip would then have happened
on both stacks and nothing would ever have caught it.

### The regeneration, and what blesses it

Both stacks, the generator's own `case_enjl()`, 21278 keys each:

    py3.14 vs py3.9   0 keys differ outside numerically-zero quantities

The 45 keys that do differ are the isovector current `point.J_rho` (0.0 against
1e-9, at Y_C = 0.5 where isospin symmetry makes it vanish; the isoscalar
`J_omega` beside it is 6.9e+06) and the `mu_C`/`n_B^Q` slots of `x`, all below
2e-7 in absolute value. Exactly the two classes ticket 62 blessed for `ccdm`
and `njl`. **The property that was lost is measurably back.**

Against the superseded 3.9 file, of 21278 keys: 108 differ only in
numerically-zero quantities, 92 are the recovered n_B = 0.1667 point, none are
dropped, and **every real change is confined to
`fixed_YC_YS.Y_C0.5.Y_S0.nolep`** — the five other cases in the file are
untouched. Within that block exactly **four observables moved**, all at
n_B = 0.5000:

| quantity | frozen 3.9 | regenerated |
|---|---|---|
| M_q.u | 5.500000 | 205.890785 |
| eps | 480.649355 | 490.235836 |
| P | 37.618524 | 84.479985 |
| mu_b | 1036.535757 | 1149.431640 |

That is **the 3.9 golden carrying the same defect, one point of it**: it took
the same fall-through at n_B = 0.5000 that 3.14 took at 0.3000. The rest of the
diff is `mu_S` and the two absent species' potentials it projects into
(`kF.Lambda`, `nu.Lambda`) — the undetermined potential becoming determined,
which is the fix working.

    PYTHONPATH=. python3 -m pytest test/baseline/ -q       ->   16 passed
    PYTHONPATH=. python3 -m pytest test/enjl/ -q           ->  127 passed
    PYTHONPATH=. python3 -m pytest test/njl/ test/ccdm/ -q ->  134 passed
    PYTHONPATH=. python3 -m pytest test/mixed/ -q          ->  277 passed
    python3 -m eos.enjl.verify.run_full_check              ->  PASS, 18 checks

`test/mixed/` is the composite engine that consumes this model through
`adapters.enjl_branch_pair`, and it is green. That run is NOT cleanly
attributable — a concurrent session held eighteen files modified in the tree
throughout — but there is nothing to attribute: it passed. The attributable
statement about the coupling is ENJL's own `check_maxwell_crossing`, which
drives the branch pair through `eos.mixed.construction.locate_maxwell` and
reports dP/P = 6.4e-13, dmu_B/mu_B = 0.

All thirteen `.npz` reproduce, so the change is provably confined to `enjl`.
`test/baseline_py39/` and the user's hand copy at
`~/Desktop/Research/backups/baseline/` are untouched.

**`test_baseline[enjl]` is green.** The map's one deliberate red is gone, and
[ticket 25](25-acceptance.md) can report a clean suite with no carve-out.

### The countermeasure

`eos.enjl.verify.check_residual_margin` — every mode must clear `RESIDUAL_TOL`
by at least two decades, not merely pass it. Passing the gate is the wrong
test when the response to missing it is a root on another branch. Measured
worst is now 3.81e-15 against the 1e-12 margin.

**Demonstrated in both directions**, not argued: monkeypatching
`strangeness_row_is_empty` to return False re-opens the null column, and the
check returns `passed=False, max_error=2.76e-11`, naming
`fixed_YC_YS {'Y_C': 0.5, 'Y_S': 0.0} n_B=0.1200`. It would have gone red years
before any interpreter changed — and note it fires on the SYMPTOM, the residual
the ill-conditioning leaves behind, not on the null column itself. Catching the
column directly is [ticket 75](75-undetermined-potential-check.md)'s to decide.

### What ticket 62 got wrong, and it matters for ticket 75

`enjl` was run as the **negative control** for the `S_i`/`C_i` screen —
"nothing here is proportional to any charge" — while the same ticket recorded,
as a footnote, "the sibling in `mu_S` at Y_S = 0 (7.3e-04 MeV over three
densities), with every other entry of `x` bit-identical."

**Those are one finding.** `mu_S` is undetermined at Y_S = 0 for exactly the
structural reason `mu_3 = mu_C` is undetermined in the CFL pattern: a conserved
charge no populated species carries. The screen FIRED on `enjl` and its output
was read as noise. So the map's strongest claim for it — "fired verbatim in two
independently written models" — is three, not two, and the case held up as the
negative control is the one where the undetermined potential did the most
damage.

That sharpens what [ticket 75](75-undetermined-potential-check.md) should
check. An undetermined potential is not only a reporting nuisance that makes a
baseline key wander; **it is a conditioning hazard.** It degrades every residual
in the mode that carries it, and the first thing that degradation reaches is
whatever the solver does when a root misses its gate. A screen that only reads
ratios between two runs would have classified this correctly and still missed
that it was about to flip a branch.

### Already known, and papered over at the wrong layer

`test/baseline/generate_baseline.py`'s `row()` docstring has described this
phenomenon all along — "mu_S when no strange species is populated: n_S = 0
holds for a whole range of mu_S, the residual has no gradient in that
direction, and the solver stops wherever its path happens to end." The
repository knew. It responded by **excluding `mu_S` and the S-carrying species
potentials from the recorded baseline**, which hides the symptom at the
recording layer and leaves the ill-conditioning in the solver, where it went on
to select a chiral branch. `row()` is not changed here — it still protects
`did` and the other models at Y_S = 0, which have not had this fix — but the
gap between "do not record it" and "do not leave it undetermined" is the whole
of this ticket.

### Files

- `eos/enjl/solver.py` — `strangeness_row_is_empty`, the pinned row and its
  scale, `BetaPoint.seed` and the named seed list
- `eos/enjl/verify/run_full_check.py` — `check_residual_margin`, check 11
- `eos/enjl/enjl.md`, `eos/enjl/enjl.tex` — the modes table and residual row 11
  state the empty row and the pin (§11: every equation the code solves is
  written out). The `.tex` compiles.
- `test/baseline/enjl.npz` — regenerated on python.org 3.14.2 (untracked, §11)
- `test/baseline/generate_baseline.py` — the "do not regenerate" note replaced
  by what was actually wrong (untracked, §11)

### Not done here

The physics question the ticket posed — how a raw ENJL continuation should
choose its branch across a first-order transition — is real, is NOT what broke
the baseline, and is now [ticket 83](83-enjl-branch-selection-physics.md),
non-gating. One measurement for it, taken here: `build_constructed_table`'s
min-eps over the up and down sweeps does **not** immunise the delivered table
against this class of failure, because the up sweep is the thing that flips.
Before the fix it delivered eps = 295.173 at n_B = 0.3000 on 3.14 and 279.821
on 3.9.
