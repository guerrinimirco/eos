# How many NJL solves does one hybrid row actually cost?

Type: task
Status: closed
Assignee: guerrinimirco
Blocked by: 01
Parent: ../map.md

## Question

The mixed target (200-point hybrid table under 60 s) was sized from an
**unverified estimate**: ~50-150 NJL solves per row, giving ~10^4-10^5 per
table and a required ~3 ms per `thermo` call. If the true count is 10x either
way, the target moves by 10x and ticket 09's verdict changes with it.

`eos/mixed/solver.py:612` solves each row with
`scipy.optimize.root(residual, guess, method="hybr")`, and every residual
evaluation calls `p_Q.thermo(...)` at `solver.py:434` — a full NJL internal
solve. On top of that sit `boundaries.locate_window`'s scan-and-bisect and
`refine_window`.

### What to produce

Instrument `njl_phase`'s `thermo` with a call counter and run one hybrid table
(or a small grid, honestly extrapolated), attributing NJL solve calls across:

- `solver.solve`'s `root(hybr)` iterations — including how many are spent on
  the finite-difference Jacobian (n+1 residuals each, and `hybr` re-forms it
  more often than a Newton would)
- `boundaries.locate_window`'s scan and bisection
- `boundaries.refine_window`
- the retries in `solver.sweep`'s `step(...)` bisection when a step fails
- wing (pure-phase) rows outside the window

and the **distribution**: is the cost concentrated in a few hard rows near the
onset, or flat across the window? A few hard rows and a flat profile call for
completely different fixes.

Also record how many of those NJL calls are **wasted on losing patterns** — if
`njl_phase` is enumerating more than one pattern inside the mixed loop, that
multiplies straight through, and memory already records 355 s vs 16.6 s (a
21x factor) between three patterns and one at a single mixed point.

### Measured at ticket 02 — two premises above are wrong

- **The 21x pattern factor is not real.** One `eos.mixed.eos_point`, DID+NJL,
  eta = 0, T = 0, n_B = 1.000 fm^-3: three patterns **29.6 s**, held 2SC
  **9.2 s** — 3.2x, not 21x — and held CFL **109.5 s and NON-CONVERGENT**. The
  355 s figure the estimate rests on is 12x too high. Held 2SC also lands at
  chi = +0.3098 against the enumeration's +0.9745, so it is not the same state
  and the two are not interchangeable at any price.
- **The locator is the dominant cost, and it is the FIRST thing to attribute.**
  `eos.mixed.eos_table` with an enumerating njl adapter did not return in 24
  minutes on a four-point grid, while 20 standalone `eos_point` calls took
  2576 s. Scanning 20 densities located the window at 0.800 -> 1.200 fm^-3,
  which puts **52 of a 200-point grid inside it**: 52 x 29.6 s = 1539 s of
  rows, against a whole-build budget of 60 s. So even with the locator free,
  the rows alone need 26x.

The per-call budget this ticket must replace the ~3 ms estimate with is
therefore set by 52 rows, not 200.

## Gate

- NJL `thermo` call count per hybrid row, and per table, measured.
- The attribution across the five sources above.
- The **real** per-call budget for the 60 s target, replacing the ~3 ms
  estimate.
- A statement of whether the dominant cost is call COUNT or per-call COST,
  since that decides whether ticket 07 or ticket 06 is the mixed phase's lever.

---

## Resolution

**A hybrid row costs ~40 mixed residual evaluations = ~41 quark `Phase.thermo`
calls = ~123 NJL internal solves, and a whole 200-point table costs ~12,100
NJL internal solves in ~4,460 s. So the estimate this ticket was written to
check was RIGHT: 50-150 solves per row, ~10^4 per table. The target does not
move by 10x, and the per-call budget moves in the caller's favour — 15 ms per
`thermo` call, not the ~3 ms assumed.**

**The dominant cost is per-call COST, not call count — with one exception, and
it is the locator.**

### How it was measured

`.scratch/njl-speed/count_mixed.py` (raw output in `count_mixed_run.log`).
Counters monkey-patched from outside onto `eos.mixed.solver.{root, residual,
default_guess}`, the same three names in `eos.mixed.boundaries` (it imports
them directly), `eos.njl.thermodynamics.thermo_from_mu`,
`eos.njl.solver.solve_pattern`, and each `Phase.thermo` / `Phase.seed`.
**Nothing in `eos/` was edited**, so the tree stays landable (section 12).

Attribution is by call stack; the sweep's retry level is read off the innermost
`step` frame's `depth`. Jacobian columns are separated from trial steps by the
x-trace of each `root` call — MINPACK's `fdjac1` perturbs one coordinate at a
time from a common base, so a residual differing from the running base in
exactly one coordinate is a Jacobian column. `scipy`'s own `nfev` corroborates
every split.

Configuration is ticket 02's pinned benchmark: python.org **3.14.2, numpy
2.3.5, scipy 1.17.0**; `rg_njl1`, `csc=True`, `backend="fast"`, DID + NJL,
patterns `("unpaired", "2SC", "CFL")`, `beta_eq_neutrinoless`, T = 0, eta = 0,
target grid `linspace(0.08, 1.60, 200)`. The four unknowns are
`(mu_tilde_B_H, mu_B_Q, chi, mu_eG)`, so n = 4.

### The counts

| | resid | thermo/phase | `thermo_from_mu` | root calls | wall | cpu |
|---|---|---|---|---|---|---|
| one COLD row, n_B = 1.000 (chi +0.9745) | 47 | 48 | **147** | 1 | 38.8 s | 38.4 |
| 7 rows at the ONSET, 0.859→0.905 (chi 0.71→0.85) | 268 | 275 | **846** | 7 | 89.9 s | 89.9 |
| 7 rows DEEP, 1.004→1.050 (chi 0.977→0.991) | 289 | 296 | **912** | 8 | 201.4 s | 199.5 |
| `locate_window`, HINTED | 1825 | 1853 | **5700** | 47 | 3384.6 s | 3234.4 |
| quark wing, 6 rows | — | — | (njl's own solver) | — | 1.3 s | 1.3 |

**The counts are deterministic.** Three repeats of the cold row returned 47
residuals and 147 NJL solves every time; only the clock moved (41.5 / 38.8 /
37.0 s). The warm sweep, run twice, returned 289 / 912 both times at 233.6 s
and 201.4 s. **A count is quotable where a wall time is not** — which is worth
more to this map than either number.

Per row: **~40 residuals → ~41 quark `thermo` calls → ~123 NJL internal
solves** (each `thermo` enumerates 3 patterns; the extra thermo beyond the
residual count is `result_from_root`, and each `solve` also pays one
`Phase.seed` per phase = 3 more `thermo_from_mu` and ~4 `solve_pattern`).

### The five sources, attributed

Of the whole table's ~12,100 NJL internal solves and ~4,460 s:

| source | NJL solves | wall | share of wall |
|---|---|---|---|
| `locate_window` — `walk_to_crossing` | 3,003 | 1,404 s | 31% |
| `locate_window` — `scan` (the probes) | 1,650 | 1,331 s | 30% |
| `refine_window` (`solve_fixed_chi`) | 462 | 421 s | 9% |
| `locate_window` — `bisect` | 585 | 93 s | 2% |
| the 51 mixed rows inside the window | 6,273 | ~1,060 s | 24% |
| quark wing (47 rows × 3) | 141 | 10 s | <1% |
| hadronic wing (102 rows, DID) | 0 | negligible | — |

- **`root(hybr)` iterations.** 1,778 of the locator's 1,825 residuals are
  inside `root`; the rest are `solve`'s own `res_max` gate.
- **The finite-difference Jacobian is 9–24% of residuals, never more.** Every
  `root` call in every run formed the Jacobian **exactly once** — 4 columns,
  n = 4 — and Broyden-updated thereafter. The cold row: 42 trial steps against
  4 Jacobian columns, and `nfev = 46 = 1 + 4 + 41` confirms it.
- **`sweep`'s step bisection: 833 of the locator's 1,825 residuals (46%) sit at
  retry depth ≥ 1**, and every one of them is inside `walk_to_crossing`. Depths
  run to 6. In the table's own row sweeps, by contrast, retry depth is **0
  everywhere** — not one bisection in 14 rows.
- **`solve`'s own second guess fires.** The deep sweep took 8 `root` calls for
  7 rows: one row's warm start missed the 1e-10 residual gate and fell back to
  `default_guess`, doubling that row. That is invisible at the `sweep` level —
  it is not a bisection — and it is the only within-row variance seen.

### The distribution: flat in COUNT, 2x in COST, and NOT worst at the onset

Per-row residual counts are flat across the window — **38.3/row at the onset
against 40.3/row deep inside** — but the wall time is **12.8 s/row at the onset
against 28.8 s/row deep inside**, because the per-solve cost rises with chi:
101 ms per `thermo_from_mu` at the onset, 208 ms deep in, 579 ms in the
locator's wide-stride sweeps. So it is neither "a few hard rows near the onset"
nor flat: **the count is flat and the per-call cost is what varies**, and the
expensive end is the QUARK-dominated end, not the onset. The ticket's
hypothesis had the gradient backwards.

Spread over individual `root` calls: 25–51 residuals for table rows, 8–107 for
the locator's walk steps.

### The real per-call budget, replacing ~3 ms

For a 200-point hybrid table at these boundaries (51 rows inside the window,
47 quark-wing rows, 102 hadronic-wing rows):

- **~12,100 NJL internal solves → 60 s / 12,100 = 4.95 ms per `thermo_from_mu`.**
- **~3,944 quark `thermo` calls → 60 s / 3,944 = 15.2 ms per `thermo` call**,
  where one `thermo` call is a 3-pattern enumeration at fixed potentials.

Today those are 368 ms and 1,130 ms. **The gap is 74x on the whole build.**
Ticket 02's 26x was for the ROWS alone and stands (the rows are ~1,060 s here,
against 1,539 s there — the window turned out to be 51 rows, not 52, and the
onset rows are cheaper than the deep ones).

### Count or cost? COST — and the answer picks ticket 06, not ticket 07

**Per-call cost, decisively.** 12,100 solves is not a number that can be
argued down to fit 60 s at 368 ms each; it would have to fall to 163. The
product has to come from the per-call side, and the njl half of this map is
already aiming far past what is needed: **1 ms per enumerated njl point is the
same unit as a `thermo` call, and the budget is 15 ms.** If tickets 03/05/06
land anywhere near their target, the mixed 60 s follows with an order of
magnitude of margin and the mixed loop's call count never has to move.

So **ticket 06 (compile the Newton loop) is the mixed phase's lever.** Two
corrections for ticket 07:

- **Its stated lever is not there.** "`hybr` re-forms the Jacobian more often
  than a Newton would" and "the naive multiplier for ~5 unknowns is ~6x on the
  Jacobian-forming calls alone" are both wrong: `hybr` forms it **once per
  `root` call** and Broyden-updates, so the FD columns are 9–24% of residuals
  and **removing them entirely is a 1.1–1.3x ceiling.** What an analytic block
  might still buy is a shorter iteration path — 25 to 107 trial steps per solve
  is a lot for four unknowns — but that is now an unproven hypothesis, not a 6x,
  and ticket 07 has to measure it rather than assume it.
- **On this pairing BOTH adapters need a block.** `did_phase` carries no
  `jacobian_block` either — only `dd2_phase` and `vmit_phase` do — and
  `_jac_with_fallback` returns None if **any** phase lacks one. So
  `analytic_jac=True` is dead on DID+NJL until DID gets one too, or the pairing
  moves to DD2+NJL.

**The exception, and it is ticket 07's real prize: the locator.** It is 5,700
of the 12,100 NJL solves (47%) and 3,385 of the 4,460 s (76%) — and that is the
**cheapest form of it**: hinted at ticket 02's own scan range, which also
switches off the probe-refinement passes. `build_hybrid_table` calls
`locate_window` with **no hint**, and ticket 02 already measured that form not
returning in 24 minutes on a four-point grid. Inside it, `walk_to_crossing` is
53% of the residuals and 46% of them are the sweep's retry bisection — pure
call count, spent to produce two floating-point numbers. **That is a count
problem, it is the only one on the table, and it is worth more than every
other mixed-side change combined.**

### Numbers a later ticket should not have to re-measure

- Window on this pairing, exactly refined: **n_onset = 0.8534, n_offset =
  1.2450 fm^-3**, 51 of 200 target-grid densities inside, 15 probes.
- chi rises almost vertically off the onset: **+0.714 one grid step (0.0076)
  above n_onset**, reaching +0.9745 at n_B = 1.000. The mixed region is
  quark-dominated over nearly its whole width.
- A quark wing row costs **217 ms** (3 `solve_pattern`, warm-started); the
  whole 47-row upper wing is ~10 s, so wings are not a problem but are not
  free either.

### Correction, against ticket 03 (which landed concurrently)

This resolution named **ticket 06** as the mixed phase's lever. That was
written before [ticket 03](03-profile-one-cfl-solve.md) posted, and it is
wrong: 03 measured a fully compiled Newton loop at **1.05x** (1.3x if every
uncompiled numpy block became free), because a converged warm CFL solve is
already 77% jitted quadrature and 4.6% Python. Compilation is spent.

**The verdict itself is unchanged and is reinforced** — the mixed target is
bought on per-call COST, not call count. Only the route changes: not ticket 06
but [ticket 10](10-the-quadrature-itself.md) (the quadrature, 77% of a
converged solve) and [ticket 05](05-cheap-pre-screen.md) (the rescue ladder on
candidates that never converge).

**And the two tickets' numbers cross-check in a way that points straight at
05.** Ticket 03 measures a converged warm CFL solve at **80 ms/pt** in a
density sweep. Ticket 04 measures **208 ms** per `thermo_from_mu` deep in the
mixed window and **101 ms** at the onset — for the same single-pattern solve.
The mixed loop pays **~2.6x the njl table's converged cost per call**, and the
difference is not mixed-phase physics: it is that the mixed residual walks the
phase to potentials where a candidate collapses, and every such call pays the
same rescue ladder that is 89% of a CFL table and 92% of an enumerated one.
**So ticket 05 is worth roughly as much inside `eos/mixed` as it is in `eos/njl`,
and that was not visible from either ticket alone.**
