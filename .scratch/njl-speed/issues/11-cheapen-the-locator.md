# The window locator is 76% of a hybrid build. What is the cheapest honest way to find two numbers?

Type: prototype
Status: closed
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

---

## Resolution, 2026-09-22

**The cheapest honest cut was neither a new search nor a better step: it was
to stop paying for answers the locator already throws away.** Two edits, both
bit-identical in the window, land together with ticket 17 in **`66d713c`**:
a step DOWN in `walk_to_crossing` gets no `sweep` retry ladder, and
`njl_phase` declines `lm` while enumerating. **Hinted, the locator's NJL
work falls 2.35x (350,866 -> 149,000 internal evaluations, 5,700 -> 3,084
NJL solves); unhinted -- the form `build_hybrid_table` actually calls,
measured to completion here for the first time -- 1.48x (564,051 -> 381,179;
10,425 -> 8,451).** What is left is still more than half doomed work, and the
ticket says where.

### What the locator actually does on this pairing (the trace)

`t11_trace_locator.py` records every mixed solve the locator makes -- site,
`sweep` retry depth, density, outcome, chi, and its cost in mixed residuals,
NJL solves and NJL internal evaluations -- on top of ticket 04's counters and
ticket 17's census. Ticket 04's configuration exactly: DID + NJL `rg_njl1`,
csc, three patterns, backend `fast`, `beta_eq_neutrinoless`, T = 0, eta = 0,
grid `linspace(0.08, 1.60, 200)` (tol = 0.00382), python.org 3.14.2 / numpy
2.3.5 / scipy 1.17.0, HEAD `63a6dfa`. It reproduces ticket 04's hinted run to
the call (1825 residuals, 5700 NJL solves).

1. **The scan's probes below the onset fail, cold, at 17,000-38,000 NJL
   evaluations each, where a converged probe takes 2,900-8,400.** Below the
   onset this pairing's mixed system has no solution at all, so the probe is
   a cold `hybr` wandering 39-137 residuals before giving up, and every NJL
   candidate at those trial potentials that has no root grinds `lm`'s
   400-evaluation cap and the polish. Hinted: 3 probes, 84,367 evaluations
   (24%). Unhinted: 6 probes, 154,097 (27%). Their results are discarded.
2. **So `crossing(0)` finds no chi <= 0 probe and `walk_to_crossing` walks
   down** from the lowest mixed probe (hinted 0.859, 2 steps; unhinted 0.909,
   15 steps), every step converging at depth 0 in 24-79 residuals, until one
   fails.
3. **The failing step then pays `sweep`'s six-level retry ladder, which is
   bisecting toward a density that does not exist.** Its midpoints converge
   (0.853362 at chi +0.6931, 0.852407 at +0.6895, ... down to 0.851572), the
   target 0.851453 is re-tried from each and fails every time, each retry a
   full failing mixed solve of 71-146 residuals -- and **the walk returns the
   step's midpoint whatever the ladder did**, because `sweep` returns only the
   target and never the midpoints. Hinted: 833 of 1825 residuals, 140,546
   evaluations (40%). Unhinted: 698 residuals, 117,285 evaluations.
4. **`bisect` finds the offset** from a chi sign change the probes did bracket
   (5-6 steps, each re-solving its bracket's low end first).
5. **`refine_window`'s exact onset solve FAILS, every time, on both forms**
   (hinted 39,012 evaluations, unhinted 44,644), and unhinted the exact
   offset solve fails too, after **607 mixed residuals** and 66,226
   evaluations. These two are the most expensive single solves the locator
   makes.

### Two things ticket 04 recorded that are not so

- **"n_onset = 0.8534, exactly refined" -- it was never refined.** The chi = 0
  fixed-chi solve fails; 0.8533622658748288 is the walk's midpoint,
  0.5 * (0.855272 + 0.851453), resolved to tol. Only the hinted OFFSET is
  exact (1.245028, the refine converged); unhinted, neither boundary is.
- **The mixed branch here does not start at chi = 0.** Mixed solutions exist
  down to ~0.8516 at chi ~= 0.686 and then cease to exist; chi never runs
  continuously to 0 on this branch, which is why the chi = 0 solve has nothing
  near its seed to find. The reported onset is an EXISTENCE boundary resolved
  to tol, not a chi crossing, and both forms bracket the same one: hinted
  [0.851453, 0.855272], unhinted [0.847985, 0.851804]. Whether that
  discontinuity is physics (the quark phase switching pattern inside the
  Gibbs construction) or a second branch the continuation cannot follow is
  NOT this ticket's question, and nothing here changes it.

### The edit

`eos/mixed/boundaries.py`, `walk_to_crossing`: a downward step calls `sweep`
with `max_bisect=0`; an upward step keeps the ladder, because there "a failure
is only a failure" and the walk returns nan on one. The docstring carries the
argument and the measurement. **The argument is the walk's own**: going down
a failed step IS the boundary, and the return value on a failed step is the
midpoint unconditionally, so the ladder can change the answer only by
RESCUING a step -- a tol step that fails yet whose target exists, which the
walk's docstring already rules out ("the regime the mixed solve is reliable
in"). **Measured, not only argued**: every successful walk step on both forms
is at depth 0, and over the 18 DD2+vMIT configurations
`test/mixed/test_window_location.py` pins (`t11_walk_probe.py`), the walk
takes 4 steps, all at depth 0, none failing -- that pairing's below-onset
probes converge with chi < 0, so its onset is bracketed directly and the walk
is almost never used there.

Ticket 17's `lm` bound is the other half of the saving and is written up
there. Their shares, hinted, each measured alone as an outside patch on HEAD:
walk 350,866 -> 210,320 evaluations, `lm` -> 264,807, both (the landed tree)
-> 149,000.

### Gate -- PASS on counts; the clock half is not quotable

| `locate_window` | mixed residuals | NJL solves | NJL evals | window |
|---|---|---|---|---|
| hinted, HEAD `63a6dfa` | 1825 | 5700 | 350,866 | 0.8533622658748288 -> 1.2450284086872276 |
| hinted, landed | **975** | **3084** | **149,000** | identical to the last digit |
| unhinted, HEAD | 3356 | 10,425 | 564,051 | 0.8498949291914121 -> 1.2426704545454546 |
| unhinted, landed | **2710** | **8451** | **381,179** | identical to the last digit |

- **The same window**: bit-identical on both forms, `exists` True, 15 and 27
  probes unchanged. Against the gate's 0.8534 / 1.2450, the unhinted window
  differs by 0.0035 and 0.0024 -- both inside half a grid spacing (0.0038),
  before and after, because both are tol-resolution readings of the same
  boundaries from differently placed probes.
- **Every solve that converges is unchanged** in chi and in mixed-residual
  count; on the landed unhinted run the six doomed probes still fail, five of
  them along the identical mixed trajectory, the sixth (n_B = 0.08) along a
  longer one (91 residuals against 39) -- the one place the `lm` bound made a
  doomed solve dearer, +24% there, against -24% over the six.
- **No-transition / nan -- honestly, part exercised and part argued.** The
  edit touches no branch that returns nan or decides `exists`: off-grid,
  `MAX_WALK`, an upward failure and `bisect`'s unlocated midpoint all return
  exactly what they did, and the only step it changes is the failing downward
  one, whose value was always the midpoint. `no_transition` IS exercised on a
  real locate (`test_reason_separates_physics_from_a_failed_location`, vMIT
  B4 = 400) and passes. A bracketed-but-unlocated nan from `bisect` is
  exercised by NO test on a real solve, before or after -- only through
  synthetic `Window` objects. That gap predates this ticket and is recorded,
  not closed.
- **Tests on the landed tree**: `test/njl` 132, `test/mixed` 272,
  `test/baseline` 20, `test/test_imports.py` 221, all passed; details and the
  fingerprints in ticket 17.
- **Wall time: not quotable.** loadavg 70-430 all session (two BayEoS
  multiprocessing pools and a video call), battery until it died mid-run,
  cpu/wall 0.4-0.6. Indicative cpu only: hinted 3917 -> 1323 s, unhinted
  5664 -> 3130 s. **The unhinted form does return**: ticket 02's "did not
  return in 24 minutes" was `eos_table` on a four-point grid, a different
  call. A quiet AC window is still owed for the clock.

### What the locator costs at ticket 09's target

Priced two ways, since a doomed NJL solve costs ~3x a converged one here: at
1 ms per NJL solve (the unit this ticket used for its "~6 s"), and at 1 ms per
38.9 evaluations -- what a converged row solve takes (35,484 / 912).

| form | HEAD | landed |
|---|---|---|
| hinted | 5.7 s / 9.0 s | **3.1 s / 3.8 s** |
| unhinted | 10.4 s / 14.5 s | **8.5 s / 9.8 s** |

Beside ~6.3 s of rows (6,273 solves) and ~0.1 s of quark wing, **the whole
unhinted build lands near 15-16 s at the njl target against 60 s**. The
locator does not decide whether 60 s is reachable; the per-solve cost does.

### What is left, in the order it pays (landed unhinted, 381,179 evaluations)

1. **The scan's six doomed probes: 117,152 (31%).** The largest block and the
   only one that needs a design decision: nothing in a probe says it is below
   the onset until it has failed. Two candidates, neither tried: order the
   scan top-down and stop at the first failure below a converged probe
   (changes the scan's continuation, which its docstring defends at length),
   or screen a probe with the pure hadronic phase against the quark phase at
   its potentials (a Gibbs-condition test, valid at eta = 0 only).
2. **The two failed exact refines: 87,370 (23%).** On this pairing the onset
   refine looks for a chi = 0 root that the walk has just shown is not there,
   and the unhinted offset refine runs 607 residuals from a seed it cannot
   converge from. Skipping the onset refine when the walk ended on a FAILED
   step would be exact here; it needs the `Window` to carry how its onset was
   found, which is new plumbing, and a measurement on a pairing where the
   refine succeeds.
3. **The unhinted walk's 15 honest steps: 103,186 (27%).** Bracket-first
   (steps of tol, 2 tol, 4 tol, ... then bisect on existence) would take ~7
   solves for 15, but moves the answer within tol rather than keeping it.
   Separately, each step first RE-SOLVES its own start point (`sweep` over
   [r, n_next]) only so the ladder has a previous density -- 290 of the walk's
   857 residuals and 35,497 of its 103,186 evaluations, about a third. A
   downward step no longer has a ladder, so that re-solve now buys nothing
   there; dropping it changes the seed at round-off and was left out of this
   bit-identical landing.

### Artifacts (in `.scratch/njl-speed/`)

`t11_trace_locator.py` (the trace; `T11_VARIANT=walkdown0` / `nolm` patch an
arm from outside), `t11_walk_probe.py` / `.log` (the DD2+vMIT walk census),
`t11_{hinted,unhinted}_{head,walkdown0,landed}.log`, `t11_hinted_nolm.log`.
