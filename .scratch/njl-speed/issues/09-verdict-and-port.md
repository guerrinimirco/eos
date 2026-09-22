# The verdict: did exact solves reach 1 ms/pt and 60 s, and what ports?

Type: grilling
Status: open
Blocked by: 13, 16, 11, 17, 10, 18
Parent: ../map.md

## Question

The map's closing ticket. Three things to settle, with every number from a
closed ticket rather than re-derived.

### 1. The measurement

Assemble the product of the levers against the two targets:

| target | baseline (02) | after 05 | after 06 | after 07 | reached? |
|---|---|---|---|---|---|
| ms/pt, beta-eq T=0 CSC table | | | | n/a | 1 ms |
| s per 200-point hybrid table | | | | | 60 s |

State it plainly, including if it fell short. A 450-900x ask met at 200x is a
real result and a useful one; reporting it as success is not.

### 2. What the production port is

The prototype lives on `njl-speed` and is a spike. The port is a separate
effort and this ticket **specifies** it rather than doing it:

- which files change, and whether `backends/` stays deletable in the ported
  form (CLAUDE.md section 5)
- what goes in `eos/njl/verify/` and `test/njl/` — every new physics invariant
  needs a `verify/` entry, every new behaviour a test (section 12)
- whether `build_fast_table` and its `FAST_MODES` restriction survive, or are
  retired now that exact solves are cheap
- what `eos/njl/njl.tex` and `.md` owe: section 11 requires every equation the
  code solves to be written out, and a compiled Newton loop with re-inflation
  rescues is solver behaviour the document currently does not describe
- whether `analytic_jac` flips to default-on across `eos/mixed`
- a landing measurement per section 12: `test/run_clean_suite.sh` at a landing
  point, certificate path cited, **including any DISCARDs**

### 3. Which fog graduates

Judged on what was measured, not on appetite:

- **Continuation along the sweep** — graduates if 05's pre-screen left the
  enumeration still dominant.
- **The interpolated phase surface** — graduates if the 60 s mixed target was
  missed, or if the 1 s figure becomes the requirement. It is the only route to
  1 s and it is a different destination, so it graduates as a **new map**, not
  as tickets on this one.
- **C or Fortran via f2py** — graduates only if 06 measured numba short and said
  by how much.
- **The BayEoS `njl` registry entry** — now specifiable, because the per-theta
  table build time is finally a known number. Likely a new map.
- **Finite T and `fixed_YC` as delivered paths** — 08 measured them; making them
  production paths is a separate effort.

## Gate

- The table filled in from closed tickets, each number citing its ticket.
- A written port specification a hand-off session can execute without
  re-deriving anything.
- Each fog entry above ruled: graduated (to where), or left in fog (why).
- The map's Decisions-so-far complete, and the map marked closed.

---

## Part 1, 2026-09-22: the measurement, and the destination ruled

**Neither target was reached, and neither survives.** The njl table landed at
~254 ms/pt against 1 ms: 2.51x of a 637x ask. No lever still open takes it
below ~50 ms/pt, even at its ceiling. The hybrid build, re-priced on the
landed tree, is ~3,200-4,700 s against 60 s, i.e. 54-79x over. The rows
alone stay at least 8x over 60 s with every open lever at its ceiling. **Both
targets are retired**, and what the map measured replaces them.

The blockers are re-pointed from 05-08 to the tickets that delivered. 05
refuted the pre-screen and handed its lever to 12 and 13. 06 and 07 are still
open, with ceilings measured by 03 and 04. 08 has not run. Every number below
comes from a closed ticket. Where two are combined, the arithmetic is shown
and each input is cited.

### The verdict table

The columns are the tickets that landed something, in landing order (they
replace the question's 05/06/07 columns). A dash means the ticket cannot
reach that row, for the cited reason.

| target | baseline | 13/16: bounded ladder (`50b3b7f`) | 11/17: locator, `lm` (`66d713c`) | 18: unlocked seed (`a948b33`) | 10: quadrature (measured, NOT landed) | reached? |
|---|---|---|---|---|---|---|
| ms/pt, beta-eq T = 0 CSC table: **1 ms** | 637, three patterns, today's default ([02], [15]) | **2.51x -> ~254** ([16]) | -- (`eos/njl` solve path untouched, [17]) | 1.00x, rows bit-identical, cpu within 1% ([18]) | 1.52x on both single-pattern tables at 12/12 ([10] §5) | **no, ~254x short** |
| s per 200-point hybrid: **60 s** | ~4,460, hinted locator ([04]); unhinted never timed | -- (counts reproduce 04 to the call at `63a6dfa`, [11], [17]) | unhinted locator evals /1.48, hinted /2.35; rows bit-identical ([11], [17]) | -- (beta-eq never reaches it, [18]) | <= 1.67x per evaluation, converged CFL, T = 0 only ([10] §5) | **no, ~3,200-4,700 s, 54-79x over** (recomputed below) |

- The 637 row is ticket 02's three-pattern figure. It became the default
  when `free` left `DEFAULT_PATTERNS` ([12], [15]). 05 measured that change
  at 12.5x (9124 -> 730 ms/pt, loaded). Against the historical four-pattern
  default, 8,100 ms/pt ([02]), the default table is ~32x faster: the 12.5x of
  dropping `free`, times the ladder bound's 2.51x.
- The ~254 is 16's 2.51x carried onto 02's scale. 16's own quiet window read
  174.3 ms/pt for the default and 167.2 for three patterns. 05 measured one
  configuration 2.7x apart in two windows, so ratios transfer between windows
  and absolute times do not.

### njl, 1 ms/pt: retired

- **Landed: ~254 ms/pt**, 2.51x of a 637x ask ([16]). **The shortfall is
  254x.**
- **Projected, with 10 and 06 landed:**
  - 10 at its T = 0 rule (12/12) is 1.52x on both single-pattern tables
    ([10] §5): **~167 ms/pt**. At the T-agnostic 16/16 it is 1.28-1.29x,
    ~198 ms/pt.
  - 06 is priced at its hard ceiling, `gapless_momenta` made free. At 12/12
    that block is 35.2% of the CFL table and 16.7% of the 2SC table ([10] §6),
    which gives 1.54x and 1.20x: **~108-139 ms/pt**.
  - Together that is 3.8-5.9x of the 637x ask, and **the shortfall is still
    ~110-170x.**
- **Why no route reaches 1 ms: one converged CFL solve costs more than the
  whole budget.** On ticket 03's scale a converged CFL solve is ~48 ms at
  12/12. At 8/8 it is ~37 ms, the coarsest rule measured, which already
  leaves 2SC masses off by 2.2e-8 ([10] §2, §5). A table point in the CFL
  region pays at least that one solve before any rival is proposed. Nothing
  measured will cut the solve itself:
  - compilation is spent: Python is 4.6% of a converged solve, and compiling
    the whole loop buys 1.05x ([03]);
  - node count is spent: below ~10 nodes the rule stops paying, because
    `gapless_momenta` does not scale with it ([10] §6), and 12 nodes already
    fails at finite T ([10] §4);
  - the ladder has no cut left inside it ([13]).
- **One lever over 1.3x is not spent, and it does not change the ruling.**
  Continuation along the sweep removes the rootless candidates, still 54.7%
  of the build after the bound ([13]). Its ceiling is 2.2x. Stacked on the
  projection above it gives >= ~49 ms/pt, still ~50x short. That stack is
  generous, because the dead points are also where `gapless_momenta` costs
  most ([03]), so the two ceilings overlap.
- **Ruling: the 1 ms/pt target is retired.** The map's result replaces it:
  **2.51x on the pinned benchmark at the landed default**, rows bit-identical
  under 18, i.e. ~254 ms/pt on 02's scale and 174 ms/pt in 16's quiet window.
  **Another ~1.5x is measured and waits on
  [ticket 19](19-land-the-quadrature-rule.md).** For this model the
  exact-solve floor is **a few tens of ms per point**, set by one converged
  CFL solve and not by the solver around it. That is a real result. It is not
  the one the map was charted for.

### mixed, 60 s: recomputed, and it does not survive

Ticket 11's "~15-16 s" priced the landed counts at 1 ms per NJL solve, which
is njl's target and not what was reached. Here the same counts are re-priced
at the cost of one evaluation.

**Counts, landed** ([11], [17]):
- the unhinted locator, which is what `build_hybrid_table` calls: 381,179 NJL
  evaluations, 8,451 solves;
- the 51 rows: 6,273 solves, bit-identical. Nothing landed shaves them: `lm`
  is 1.7-4.2% of row work ([17]);
- the quark wing: 141 solves.

**Cost per evaluation.** Ticket 04's wall times (cpu within 5%) divided by
17's evaluation counts for the same runs; 17's instrument reproduces 04 to
the call. No commit since 04 changes the cost of one evaluation, only how
many there are: 16's bound reaches `njl_phase` only through seeds and wings,
and 11/17 cut counts.

| | evaluations ([17]) | wall ([04]) | ms per evaluation |
|---|---|---|---|
| 7 onset rows | 32,853 | 89.9 s | 2.74 |
| 7 deep rows | 35,484 | 201.4 s | 5.68 |
| `locate_window`, hinted | 350,866 | 3,384.6 s | 9.65 |

A locator evaluation costs more than a row evaluation, because it is mostly a
CFL candidate at a trial potential with no root. The landed cuts removed
mostly those, so the landed locator is priced between the deep-row rate and
its own old rate.

| block | landed count | at 1 ms per solve ([11]) | at the landed cost |
|---|---|---|---|
| 51 mixed rows | 6,273 solves | 6.3 s | ~1,060 s ([04], unchanged per [17]) |
| quark wing | 141 solves | 0.1 s | 10 s ([04]) |
| `locate_window`, unhinted | 381,179 evaluations | 8.5-9.8 s | 2,160-3,680 s |
| **200-point build** | | **15-16 s** | **~3,200-4,700 s: 54-79x over 60 s** |

- **Cross-check, not a measurement.** Ticket 11's indicative cpu for the
  landed unhinted locator is 3,130 s (at loadavg 70-430), inside the bracket.
- **What the map bought on the mixed side.** The same pricing puts the
  unhinted build before 11/17 (564,051 evaluations) at ~4,300-6,500 s. The map
  bought ~1.35x on the mixed build, all of it in the locator.
- **Every open lever at its ceiling:**
  - 10 at 1.67x per evaluation (its best multiple, T = 0 only): the build
    drops to ~1,900-2,800 s, still 32-47x over;
  - **the rows alone** (the locator free) are 1,060 s, 17.7x over;
  - the rows at 10's 1.67x are 635 s (10.6x over), and >= 488 s (8x over)
    with 07's 1.1-1.3x ceiling on top ([04]).

  No lever in this map, alone or combined, brings the rows alone under 60 s.
  **60 s does not survive.**

### The fog, ruled

- **The interpolated phase surface: GRADUATES, as a new map.** Its condition
  is met: the recomputed build misses 60 s by 54-79x, and the rows alone miss
  it by >= 8x at every ceiling. It is the only route left. It is also a
  different destination, because it gives up this map's first settled
  decision (exact solves only). So it is charted as its own map, not as
  tickets on this one. It inherits:
  - the 1 s aspiration from Out of scope;
  - ticket 08's warning. `build_fast_table`'s spline was 3-5% out in P at
    `fixed_YC` because it relied on dP/dmu_B = n_B, an identity only beta
    equilibrium has. A surface in all three potentials is mode-agnostic by
    construction, but that does not make it safe across a pattern switch,
    where the winner's P has a kink.

  Not charted here.
- **Continuation along the sweep: STAYS in fog.** 09's condition is met: the
  rootless candidates are still 54.7% of the build after the bound ([13]).
  The map's own precondition is not: a monitor stated and its miss priced.
  That miss is not a tail risk. Capture was 200 of 200 in [05], and a
  monitor that stops proposing a repeatedly failing pattern loses 170 of 200
  rows ([13]). It also cannot move either ruling above: <= 2.2x on njl, and
  nothing on the mixed rows, which enumerate at fixed potentials rather than
  along a sweep. It stays recorded, with its price, for whichever effort next
  owns njl table cost, most likely the BayEoS entry below.
- **C or Fortran via f2py: STAYS shut.** Its condition was numba measured
  short by 06. That has not happened: 06 is open and is being re-scoped
  ([ticket 20](20-relook-gapless-momenta.md)). What remains uncompiled is
  4.6% of a converged solve ([03]).
- **The BayEoS `njl` registry entry: GRADUATES, as a new map, downstream.**
  The number it waited for exists: a 200-point beta-eq T = 0 CSC table costs
  ~35 s per theta (16's quiet window) to ~51 s (02's scale), before 19. The
  registry lives in `bayeos`, which consumes `eos` (CLAUDE.md §1), so the map
  is charted there. Its first question is whether a sampler's call count can
  afford per-theta table builds at all. If it cannot, that map consumes the
  phase-surface map rather than standing beside it.
- **Finite T and `fixed_YC` as delivered paths: STAY in fog.** 08 has not
  run. Three closed tickets bear on it:
  - 13's bound pays more off the benchmark than on it: 2.70x at T = 30 and
    4.66x at `fixed_YC`, T = 30, gate clean;
  - 18 found and fixed a `fixed_YC`, T = 0 seed defect that beta equilibrium
    never meets;
  - 10's node rule depends on T.

  So the route is not beta-eq-only, and it is not uniformly mode-agnostic
  either. Making these production paths is a separate effort, after 08.
- **Whether `build_fast_table` survives (the map's sixth entry): it
  survives.** Its premise was exact solves reaching 1 ms/pt, and they did
  not. At 18 ms/pt it is still ~10-14x faster than the landed exact default,
  in the mode it is restricted to (`FAST_MODES`, beta equilibrium). What that
  means for the port is part 2's question.

### Filed, not done here

- [19: land the pairing quadrature rule](19-land-the-quadrature-rule.md),
  blocked by this ticket.
- [20: re-look at 06 now that `gapless_momenta` is the wall](20-relook-gapless-momenta.md),
  blocked by this ticket and 19.

Not done: part 2 (the port specification), charting the two graduating maps,
and closing this map.

[02]: 02-pin-the-benchmark.md
[03]: 03-profile-one-cfl-solve.md
[04]: 04-count-the-mixed-loop.md
[05]: 05-cheap-pre-screen.md
[10]: 10-the-quadrature-itself.md
[11]: 11-cheapen-the-locator.md
[12]: 12-free-in-the-default.md
[13]: 13-bound-the-rescue-ladder.md
[15]: 15-land-the-pattern-default.md
[16]: 16-land-the-bounded-ladder.md
[17]: 17-methods-bound-in-mixed.md
[18]: 18-bound-loses-gapless-cfl.md
