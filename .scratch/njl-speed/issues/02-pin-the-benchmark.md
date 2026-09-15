# Pin the benchmark and take the baseline

Type: task
Status: closed
Blocked by: 01
Parent: ../map.md

## Question

"1 ms per point" is unfalsifiable until one configuration is named and one
number is measured. What exactly is the benchmark, and what does it read today?

`notebooks/quark_timing.py` is already the harness — it times `njl.eos_table`
across parameter sweeps and already exposes `NJL_PATTERNS`, `NJL_BACKEND`,
`NJL_MODE`, `NJL_SELECTION`. It gets a **pinned benchmark configuration**
rather than a second harness.

### What the config must fix

- parameter set (`rg_njl1` unless there is a reason otherwise), species flags
  (`csc=True`), mode (`beta_eq_neutrinoless`), T = 0
- the n_B grid — **and its span**, which is not cosmetic: 0.5-1.55 stays above
  chiral restoration while 0.30-1.55 crosses it, and the fast-table error moves
  by an order of magnitude between them. Both are legitimate; the benchmark
  must say which it is.
- the pattern list, measured **three ways**: default enumeration (4 patterns —
  `("unpaired", "2SC", "CFL", "free")`), restricted to 3, and single-pattern
- n = 3 runs, median reported, nothing else running on the machine

### The stack question, which is not a formality

Every number this map inherits from memory was measured on **anaconda 3.9**.
CLAUDE.md section 12 names the **python.org 3.14** stack as the one the
baselines are frozen against, and records that the two disagree — scipy 1.13's
`root(..., method="hybr")` reports success while returning its seed unchanged
on one of eos's closures. A solver-speed map that measures on the stack whose
solver silently no-ops would be measuring nothing.

So: **benchmark on python.org 3.14**, and record interpreter + numpy + scipy
versions in the output. If 3.14 cannot run the notebook, say so and record what
does.

## Gate

- The config is committed in `notebooks/quark_timing.py` and is quotable by
  name from every later ticket.
- Baseline table numbers, ms/pt, median of 3, stack named: default enumeration,
  3 patterns, and each single pattern.
- A baseline **mixed** number: one 200-point hybrid table through
  `eos.mixed`, one T, with a real hadronic partner (DD2 or DID). If it cannot
  finish in a session, measure a small grid and state the extrapolation and its
  basis explicitly rather than quietly scaling.
- Both numbers restated against the map's targets: what multiple is needed.

## Resolution, 2026-09-10

**The config is committed as `54c7be9`** on `njl-speed`, as the last
cell of `notebooks/quark_timing.py`. It is self-contained and runs on its own
from the repository root with

    sed -n '/BENCH_BLOCK_BEGIN/,$p' notebooks/quark_timing.py \
        | NJL_BENCH=1 python3 -

Only the benchmark block was committed; the CSC-bag-mapping cell that shares
the same file belongs to another session and stays uncommitted in the working
tree.

`BENCH_SET` rg_njl1 unmodified, `BENCH_MODE` beta_eq_neutrinoless, `BENCH_T`
0, `BENCH_SPECIES` csc=True, `BENCH_BACKEND` "fast", `BENCH_NB` 200 densities
over 0.5-1.55 fm^-3, `BENCH_NB_CHIRAL` 200 over 0.30-1.55, `BENCH_REPEATS` 3,
`BENCH_PATTERN_SETS` the six pattern configurations, `BENCH_MIXED_SCAN` and
`BENCH_MIXED_NB_TARGET` for the mixed half. Stack is python.org 3.14.2 /
numpy 2.3.5 / scipy 1.17.0, printed with every result.

### THE BASELINE, python.org 3.14.2 / numpy 2.3.5 / scipy 1.17.0

n = 3, median, one warm-up call discarded. **Every cpu column below sits
within 2% of its wall column**, which is the quiet-machine check: three
earlier runs were discarded when it did not, at a ~5% duty cycle on battery
(`caffeinate -s` is valid only on AC power).

Table, `BENCH_NB` = 0.5 -> 1.55 fm^-3, 200 densities:

| patterns | ms/pt wall | ms/pt cpu | median s | runs | rows |
|---|---|---|---|---|---|
| default (4) | **8100.2** | 7966.9 | 1620.0 | 1588-1709 | 200 |
| three | **636.7** | 628.1 | 127.3 | 125.7-127.7 | 200 |
| unpaired | **1.4** | 1.4 | 0.3 | 0.3-0.3 | 200 |
| 2SC | **56.5** | 56.1 | 11.3 | 11.0-11.6 | 200 |
| CFL | **719.3** | 713.3 | 143.9 | 140.2-150.2 | **188** |
| free | **68.5** | 67.8 | 13.7 | 13.4-14.4 | 200 |

Table, `BENCH_NB_CHIRAL` = 0.30 -> 1.55 fm^-3, 200 densities, three patterns:
**836.5 ms/pt** wall / 831.4 cpu, 167.3 s, 200 rows.

Mixed, DID + NJL (unpaired, 2SC, CFL), eta = 0, T = 0:

- window scan **2575.8 s** wall / 2558.9 cpu for 20 densities; 12 of 20
  converged, 7 with chi in (0, 1); coexistence **0.800 -> 1.200 fm^-3**, which
  puts **52 of a 200-point grid inside the window**.
- one point at n_B = 1.000 fm^-3: **three patterns 29.6 s** (29.1-30.8,
  converged, chi = +0.9745); **held CFL 109.5 s** (107.8-110.7, **NOT
  converged**); **held 2SC 9.2 s** (8.7-9.8, converged, chi = +0.3098).

### AGAINST THE TARGETS

- **1 ms/pt.** default enumeration **8100x**, three patterns **637x**, CFL
  alone 719x, 2SC alone 56x. `unpaired` is ALREADY THERE at 1.4 ms/pt, on the
  exact-solve path with no interpolation — so the target is not absurd for this
  solver, and the whole gap is what the pairing sector costs on top.
- **60 s for a 200-point hybrid.** The mixed rows alone are 52 x 29.6 s = 1539
  s, i.e. **26x**. Adding the window locator at the scan's own rate (2576 s for
  20 densities) puts the whole build near 4100 s, i.e. **~68x** — so the map's
  inherited ~70x is right for the wrong reason: it assumed 355 s per point and
  no locator, and the truth is 29.6 s per point with a locator that dominates.

### FOUR THINGS THE NUMBERS SAY THAT THE MAP DID NOT

- **The enumeration costs 9.6x the sum of its parts.** The four singles sum to
  845.7 ms/pt; enumerating the same four costs 8100.2. Adding `free` to the
  three-pattern list adds **7463 ms/pt** — and `free` swept ALONE is one of the
  cheapest candidates at 68.5 ms/pt. So the cost is not in solving candidates,
  it is in what a candidate that collapses onto a rival's root does to the
  enumeration. This is ticket 05's retry ladder, quantified: it is 92% of the
  default table.
- **Fewer patterns is not a clean lever.** Three patterns (636.7) is CHEAPER
  than CFL alone (719.3): enumerating gives CFL seeds from its rivals, and a
  CFL-only sweep also burns time on 12 densities it never lands (188 of 200
  rows). Dropping a pattern can cost time and rows.
- **Holding a pattern in the mixed solve is neither faster nor safer here.**
  At n_B = 1.000 the enumeration converges in 29.6 s; held CFL takes 109.5 s
  and does not converge at all. Held 2SC converges in 9.2 s but at chi =
  +0.3098 against the enumeration's +0.9745 — a DIFFERENT state at the same
  density, which is the metastability the map already warns about, now with a
  number on it.
- **The stacks disagree, and 3.14 is not uniformly slower.** Against the
  anaconda 3.9 figures the map inherited: default 8100 against 6638 (worse),
  three 637 against 914 (better). A number that names no interpreter names
  nothing.

### Three things settled while pinning it, which the next session should not redo

- **The grid density is part of the benchmark.** The sweep is warm-started, so
  a coarser grid takes larger steps and reports a different ms/point for the
  same solver: 2SC measured 71.5 ms/pt at 20 densities and 46.0 at 200. 200 is
  pinned, which is also the grid the effort's inherited scale numbers used.
- **`eos.mixed.eos_table` cannot be the mixed baseline on this pairing.** With
  an enumerating njl adapter its window locator did not return in 24 minutes
  on a FOUR-point grid (DID+NJL, eta = 0, T = 0). The mixed half times one
  `eos.mixed.eos_point` instead, which is bounded, and which is what the map's
  own ~70x arithmetic is built from: table cost ~ window location + (points
  inside the window) x (cost per point). The locator's own cost is ticket 07's
  problem, and its unboundedness is a finding for that ticket.
- **The window must be scanned with the enumeration that is timed.** Scanning
  with one pattern held is much cheaper and was tried: held-2SC put the
  coexistence window at 1.0-1.2 fm^-3, and the enumeration returns chi =
  +1.0000 at 1.2 — pure quark. The timed point would have been a pure-phase
  solve wearing a mixed-phase label. The timed point now reports its own chi
  so a reader can see it was inside.

### One operational note

`notebooks/quark_timing.py` is jupytext-paired, and a save of the `.ipynb`
regenerated the `.py` and DELETED the benchmark cell mid-session. Run the
benchmark from the commit, not the working tree:

    git show 54c7be9:notebooks/quark_timing.py \
        | sed -n '/BENCH_BLOCK_BEGIN/,$p' > bench.py && NJL_BENCH=1 python3 -u bench.py

The whole run is about 2h15 of CPU, dominated by the default enumeration
(27 min) and the mixed window scan (43 min).
