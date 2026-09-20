# Bound the rescue ladder below a branch

Type: prototype
Status: closed
Assignee: guerrinimirco
Blocked by: 05
Parent: ../map.md

## Question

The three-pattern list this ticket measures against is now the DECIDED
default, not a restriction: [Does `free` belong in the default
enumeration?](12-free-in-the-default.md) dropped `free` from
`DEFAULT_PATTERNS`. Nothing below changes; the numbers simply stopped being
conditional on a `patterns=` argument.

This is ticket 05's original target — abandoning work on a branch that is not
there — in the one place it actually lives, now that `free` is separated out
([The cheap pre-screen](05-cheap-pre-screen.md), [Does `free` belong in the
default enumeration?](12-free-in-the-default.md)).

With the three-pattern list, **13 of 600 candidates take 77.1% of the build**.
They are the `CFL` candidates below the CFL onset: each stalls its first
Newton run at ~9e-2, runs `attempt`'s MINPACK rungs, then the differenced
solve, and **converges onto the 2SC root** — a duplicate of a candidate that
already converged in its own layout, for 3.0 s.

**How much of that ladder can go, given that exactly one candidate in 200 is
rescued by it into a real CFL layout?**

### Measured at ticket 05 (three patterns, 600 candidates, 53.4 s, quiet
machine — shares, not absolute times)

| stage of `attempt` | n | s | % |
|---|---|---|---|
| B MINPACK + polish after the first Newton | 28 | 29.3 | 54.8% |
| A the first Newton (jac), all candidates | 600 | 12.1 | 22.6% |
| D the differenced rescue (no jac) | 13 | 11.2 | 21.0% |
| C the reinflate rescue | **0** | 0.0 | 0.0% |

| what the rungs below the first Newton bought | n | s | % |
|---|---|---|---|
| `CFL` -> converged as 2SC (duplicates) | 13 | 39.2 | 73.3% |
| `2SC` -> held its layout | 14 | 0.8 | 1.6% |
| `CFL` -> **held its layout** (the onset) | **1** | 0.6 | 1.1% |

### The two things this ticket must not break

- **The onset is load-bearing.** The single rescued CFL candidate is what puts
  a CFL seed into `_seeds`; without it CFL is cross-seeded at the next density
  too, stalls again, and the branch — 170 of 200 rows — is never found. Ticket
  05 measured that an abort threshold on the screen residual cannot tell that
  candidate from the 13 dead ones: they screen in the same 8-9e-2 band, with
  all three gaps alive in both.
- **A collapsed candidate must not seed forward.** `solve`'s `_seeds` filter
  already handles this and stays.

### The cheapest thing to try first

Rung D produced **13 duplicates and nothing else**. Dropping it for a
cross-seeded candidate returns `converged = False` at those 13 points — which
is a *truer* report than a CFL candidate that is silently a 2SC state, and
drops it from the ranking rather than into it. Measure: the ms/pt, the gate,
and whether the onset at n_B = 0.6583 still lands.

Then rung B, which is the larger 54.8% and is also where the onset is found —
so it is bounded, not removed.

## Gate

- ms/pt on the pinned benchmark, three patterns, before and after, on a
  machine whose load is stated (ticket 05's window ran 2.7x slow).
- The map's gate: `pattern_realised` unchanged at every density, P to 1e-8,
  worst point named.
- **The 2SC -> CFL onset still found at n_B = 0.6583 fm^-3**, and the CFL
  branch still 170 rows.
- A statement of what a caller now sees at the 13 densities where a CFL
  candidate has no root.

---

## Resolution

**Most of it goes — but not by either cut this ticket proposed, and not by
anything a residual or an evaluation budget can see. The separator is WHERE
THE SEED CAME FROM.** A candidate handed another pattern's state gets one
Newton and one `hybr`; if that does not converge it is reported non-converged
and dropped. No `lm`, no differenced repeat, no retry from cold. **2.02x on the
pinned benchmark, gate clean**, and it pays *more* off the benchmark than on
it — **2.70x** at T = 30 MeV and **4.66x** at `fixed_YC`, T = 30.

### The window

python.org 3.14.2 / numpy 2.3.5 / scipy 1.17.0, `rg_njl1`, `csc=True`,
`beta_eq_neutrinoless`, T = 0, `backend="fast"`, 200 densities over
0.5-1.55 fm^-3, and **no `patterns=` argument**: ticket 15's edit is in
`2536d2b`, so `DEFAULT_PATTERNS` IS the three-pattern list and this measures
the default.

**The machine was loaded and the load is the reason ratios are quoted, not
absolutes.** Another session held a four-way multiprocessing pool at ~88% CPU
each for the whole window; loadavg 14-24. Laptop on battery at 94%, but
`process_time` tracked `perf_counter` at 0.90-0.94 in every arm, so the 5%
duty-cycle throttling that ruins a battery measurement was not happening.
**The control is bracketed**: V0 at the start of the window 138.4 s, V0 at the
end 138.2 s -- **drift 1.00x wall, 1.03x cpu**. The V0 that the gates below
compare against is a third control taken immediately beside VX, at 147.9 s.

### Both cuts this ticket proposed were measured, and both failed

| variant | wall | vs V0 (147.9 s) |
|---|---|---|
| **VD** -- rung D removed, the ticket's "cheapest thing to try first" | 154.9 s | **0.95x, a LOSS** |
| **VL** -- stop when the iterate has left its layout | 149.0 s | 0.99x |
| **VX** -- the bounded ladder, below | **73.4 / 75.5 s** | **2.02x / 1.96x wall, 2.01-2.02x cpu** |

**VD is slower than doing nothing, and the reason is one line this ticket did
not look at.** Refusing rung D leaves `ok = False`, and `solve_pattern` then
does what it does with any failed warm start: `if not ok and warm and rescue`
retries the whole ladder from the cold guess. The saving is spent immediately,
at a small loss. The counter shows it directly -- **25 rung-D refusals for 13
candidates**, because the cold pass reaches rung D too.

**VL fired once in 600 candidates.** The hypothesis was that the 13 dead CFL
candidates could be recognised by their iterate having collapsed out of the
CFL layout. It has not: at every rung boundary all three gaps are still alive,
exactly as ticket 05 reported for the Newton stall. **The collapse happens
INSIDE rung D** -- the differenced solve is what walks the state to the 2SC
root. There is nothing to detect until the work is already done.

### What the census found, which picked the cut that worked

`proto13_rungs.py` spies the three `solve_system` sites in `attempt` plus
`newton_solve` and every `root` call, so each rung is priced separately
(171.8 s instrumented control, counts deterministic, wall +-30%):

| what | n | s | % |
|---|---|---|---|
| the 13 CFL candidates below the onset | 13 | **132.7** | **77.2%** |
| `lm`, entered by exactly those 13 and nobody else | 13 | 50.2 | 29.2% |
| rung D, which **converges all 13** onto the 2SC root | 13 | 36.1 | 21.0% |
| rung C, the reinflate rescue | **0** | 0.0 | 0.0% |
| the ONE CFL candidate that holds its layout, at n_B = 0.5686 | 1 | 2.3 | 1.3% |

**The onset is rescued by `hybr`, and never reaches `lm` or rung D.** That is
the whole finding: everything past `hybr` bought 13 duplicates and nothing
else. And the duplicates are duplicates in the strict sense -- at all 13
densities the 2SC candidate has already converged in its own layout at the
same free energy **to 1.9e-10 relative**, and the delivered row already reads
`2SC`.

**An evaluation cap cannot make this cut**, which is worth recording because
it is the obvious next idea and it is measured shut: the onset candidate uses
**77** `hybr` evaluations, MORE than any of the 13 dead ones (31 / 55 / 73,
min / median / max). Any `maxfev` that keeps the onset keeps all of them.
`LM_MAX_EVALUATIONS`'s argument -- bound what a rescue may SPEND rather than
guess whether one is possible -- is right, and here the budget is not the axis
that separates them.

### The rule that works

A candidate seeded by `seed_from(reference, ...)` is being asked a question:
*is there a root of this pattern near the state we already have?* A no is an
ordinary answer, and the enumeration is solving the pattern that owns the root
it collapses onto anyway. A candidate warm-started from **its own** converged
point one density down is a different claim: there was a root here a moment
ago, so a failure is evidence worth working against, and it keeps the whole
ladder including the cold retry `attempt`'s docstring justifies (the CFL point
at n_B = 1.2 that stops at 8e-9 from a continuation seed and reaches 3e-11
from the cold one -- a continuation seed, not a cross-seed).

`solve_pattern` cannot tell the two apart today, because both arrive as `x0`.
**Only `solve` knows**, and that is the one line the change turns on.

### Gate -- PASS

- **`pattern_realised` mismatches: 0** at all 200 densities.
- **worst |dP|/P = 7.009e-10** at n_B = 0.5633, 14x inside the 1e-8 gate. It
  is not a VX artifact: it is one of the 13, where the winner is now the 2SC
  candidate instead of its CFL duplicate, and the two roots agree to that.
- **The 2SC -> CFL onset is 0.6582914572865115 in both arms** -- identical to
  the last digit, not merely to the map's 0.6583.
- **CFL rows 170 in both.** 200/200 rows delivered in both.
- The un-gated `pattern` column differs at **4** densities, and differs
  BETTER: those are 4 of the 13 where the CFL duplicate was winning the tie by
  ~1e-10 and reporting `CFL` for a state that is 2SC. Ticket 05 flagged that
  column as unstable in the baseline; removing the duplicate is what makes it
  stable.

### What a caller sees at the 13 densities

**Nothing changes in the delivered table** -- not P, not the state, not even
the un-gated `pattern` column except at the 4 where it is corrected. The CFL
*candidate* is reported non-converged and dropped from the ranking instead of
being seated as a 2SC state under a CFL name, which is the truer report the
ticket asked for. At every one of the 13 the `unpaired` and `2SC` candidates
converge in their own layouts, so there is no density at which the enumeration
comes back empty.

### It is not a beta-eq / T = 0 trick

20 densities over 0.6-1.5 fm^-3, V0 against VX, **20/20 rows solved in both
arms of all four sweeps, 0 state mismatches in all four**:

| sweep | V0 | VX | | worst \|dP\|/P |
|---|---|---|---|---|
| `beta_eq_neutrinoless`, T = 0 | 14.7 s | 12.0 s | 1.23x | 0 (bit-identical) |
| `beta_eq_neutrinoless`, T = 30 | 43.0 s | 15.9 s | **2.70x** | 0 (bit-identical) |
| `fixed_YC` 0.4, leptons, T = 0 | 43.5 s | 35.1 s | 1.24x | 1.91e-10 |
| `fixed_YC` 0.4, leptons, T = 30 | 325.5 s | 69.9 s | **4.66x** | 4.50e-11 |

The rule is about the enumeration, not about a mode, and the measurement says
so. **This is free evidence for [Is it mode-agnostic?](08-is-it-mode-agnostic.md)**
-- not its whole job, which is the acceleration as a whole, but one lever
measured at two modes and two temperatures.

### What is left, measured -- and it is the same 13

The census re-taken under VX (71.8 s instrumented, against the 171.8 s
control):

| what | n | s | % |
|---|---|---|---|
| **the 13 rootless CFL candidates, still** | 13 | **39.3** | **54.7%** |
| `newton`, every candidate | 600 | 32.3 | 45.1% |
| `hybr`, the 28 that miss the Newton gate | 28 | 21.4 | 29.9% |
| `lm` | **0** | 0.0 | 0.0% |
| rung D | 13 refused | 0.00 | 0.0% |
| the onset, still found | 1 | 2.2 | 3.0% |

**Bounding the ladder takes the rootless candidates from 77% to 55% of the
build; it does not remove them.** All 39.3 s is now the single `hybr` they are
still allowed -- ~3 s each on a system with no root. There is no further cut
inside the ladder: `hybr` is what finds the onset, and the nfev distributions
overlap the wrong way.

**So the next lever is not the ladder, it is the PROPOSAL** -- not solving a
CFL candidate at a density where it has no root -- and that is the map's
`Not yet specified` continuation entry, which now has a price on it (55% of
the bounded build) and one hard constraint this ticket measured: the onset at
n_B = 0.5686 is found ONLY because the CFL candidate is proposed and
cross-seeded there, 13 failures in a row after it was first proposed. A
monitor that stops proposing a repeatedly-failing pattern loses the branch.
Nothing graduates.

### The change this specifies

Two files, no new module, and no behaviour change for any caller that does not
ask:

- `eos/general/solve.py`: `solve_system` gains `methods=('hybr', 'lm')` --
  which MINPACK rungs a caller is willing to pay for. Default unchanged.
  Documented beside `LM_MAX_EVALUATIONS`, as the same kind of bound on what a
  rescue may spend.
- `eos/njl/solver.py`: `solve_pattern` gains `cross_seeded=False`. When it is
  set, `attempt` passes `methods=('hybr',)`, skips rung D, and the cold retry
  gains `and not cross_seeded`. `solve` sets it from whether the seed came out
  of `seeds` or out of `seed_from`. Rung C is untouched -- it fires 0 times
  here, but `attempt`'s docstring records it finding the CFL ground state at
  T = 20 MeV.

Landing it is [Land the bounded rescue ladder](16-land-the-bounded-ladder.md).

### Artifacts (uncommitted, in `.scratch/njl-speed/`)

`proto13_rungs.py` the rung census (`PROTO13_VX=1` re-takes it under the
bound), `proto13_analyse.py`, `proto13_variant.py` (V0 / VD / VL / VX),
`proto13_gate.py`, `proto13_modes.py` the cross-mode probe, `proto13_run*.sh`
and their logs, and the JSON each arm wrote.
