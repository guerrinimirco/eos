# The window locator is 76% of a hybrid build. What is the cheapest honest way to find two numbers?

Type: prototype
Status: open
Blocked by: 04
Parent: ../map.md

## Question

Ticket 04 measured a 200-point DID+NJL hybrid build at **~12,100 NJL internal
solves / ~4,460 s**, of which `eos.mixed.boundaries.locate_window` is **5,700
solves (47%) and 3,385 s (76%)** — and that is the locator's CHEAPEST form,
hinted at ticket 02's own scan range, which also switches `max_refine` off.
`hybrid.build_hybrid_table` calls it with **no hint**, and ticket 02 measured
that form not returning in 24 minutes on a four-point grid.

Everything it spends produces **two floating-point numbers**: n_onset =
0.8534 and n_offset = 1.2450 fm^-3.

This is the map's ONLY genuine call-COUNT problem — ticket 04's verdict is that
everything else is per-call cost, which tickets 03/05/06 already own. It is
also the one that does not go away when the njl solve gets fast: at njl's 1 ms
destination the locator would still be ~6 s of a 60 s budget, and that is the
hinted form.

### Where the 5,700 solves go (ticket 04, measured)

| inside `locate_window` | NJL solves | wall | residuals |
|---|---|---|---|
| `walk_to_crossing` | 3,003 | 1,404 s | 971 (53%) |
| `scan` (the 12 probes) | 1,650 | 1,331 s | 529 (29%) |
| `bisect` | 585 | 93 s | 175 (10%) |
| `refine_window` / `solve_fixed_chi` | 462 | 421 s | 150 (8%) |

and **833 of the 1,825 residuals (46%) sit at `sweep` retry depth >= 1**, all
of them inside the walk, at depths running to 6. The table's own row sweeps
take **zero** bisections in 14 rows, so this is the walk's own pathology and
not the sweep's.

### What the shape of the answer probably is, and what must not be assumed

`walk_to_crossing` exists for a real reason, documented at
`boundaries.py:234`: below the onset the mixed system has no solution at all,
so the probe scan can come back with every chi > 0, `crossing` finds no sign
change, and a transition that plainly exists is reported as absent. It is a
correctness fix, and it is not to be deleted to make a number smaller.

But it walks in steps of `tol` = half a grid spacing (0.0038 here) with
MAX_WALK = 64, each step a full warm-started mixed solve with a bisection
ladder behind it. Candidates to weigh:

- **Walk in mu_B, not n_B.** The formal mixed solution ceasing to exist is what
  ends the walk downward; a potential-space step may cross the same boundary
  without the density-space stall that triggers the retry ladder.
- **Bracket first, then bisect.** The walk currently finds the crossing by
  marching; 46% of its residuals are the retry ladder on steps that miss.
  A geometric expansion downward to the first FAILED solve brackets the onset
  in O(log) steps, and `bisect` is already written and is the cheapest thing in
  the table (585 solves for two crossings).
- **`refine_window` is 421 s for 462 solves** — the most expensive solves in
  the locator, 911 ms each. It is an accelerator on top of the scan
  (`refine="bisect"` skips it). Is the exact chi-crossing worth 9% of a build,
  given the sweep then starts at a grid point anyway?
- **Reuse across a table.** `_locate_chained` already hints from the previous
  temperature. A T = 0 single-line build pays the unhinted price once and has
  nothing to inherit from — which is exactly the case measured.

### Two traps, both already paid for once

- **The hint is not free of physics.** `locate_window`'s own docstring warns
  that raising eta moves the window as well as narrowing it, so a hint from a
  different eta must be generous. Ticket 02 additionally measured that scanning
  with ONE PATTERN HELD puts the window somewhere the enumeration calls pure
  quark — a cheap scan that finds the wrong window is worse than an expensive
  one.
- **A missing boundary must stay nan.** The current code is careful that a
  crossing bracketed but not located returns nan rather than a plausible
  midpoint, "which is what actually happened". Any cheaper locator keeps that
  distinction or it trades 76% of the wall for boundaries that wander with the
  probe spacing.

## Gate

- The locator's NJL-solve count and wall time on ticket 02's pinned
  configuration, before and after, **unhinted as well as hinted** — the
  unhinted form is what `build_hybrid_table` actually calls and it has never
  been measured to completion.
- **The same window**: n_onset and n_offset agree with 0.8534 / 1.2450 to
  better than half a grid spacing, and `exists` is unchanged.
- The no-transition case still reports no transition, and a bracketed-but-
  unlocated boundary still comes back nan — exercised, not argued.
- A statement of what the locator costs once njl reaches ticket 09's target,
  since that is the number that decides whether 60 s is reachable at all.
