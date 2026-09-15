# Where do the 443 ms of a CFL solve actually go?

Type: task
Status: closed
Assignee: guerrinimirco
Blocked by: 01
Parent: ../map.md

## Question

This is the measurement the rest of the map is guessing without. A warm-started
CFL `solve_pattern` costs ~443 ms with `backend="fast"` and the analytic
Jacobian. Memory records ~12 Newton steps (`NEWTON_STEP_FRACTION = 0.25` in
`eos/general/solve.py:158`, plus the line search), so **one residual-plus-
Jacobian evaluation is on the order of 15 ms.**

That is far too slow for what it nominally is — a Gauss-Legendre quadrature
over ~120 nodes for 9 colour-flavour modes, jitted. So the hypothesis is that
**most of the 443 ms is not the jitted kernel**: it is Python orchestration and
the pieces of `eos/general/pairing.py` that were never compiled — the spectrum
block assembly, `gapless_momenta`'s crossing hunt, the RG panel refinement, and
`pair_hessian`'s Daleckii-Krein perturbation theory.

**Confirm or refute that, with numbers.** Ticket 06 exists only if it holds.

### What to produce

- A per-function breakdown (cProfile plus targeted `perf_counter` around the
  suspected blocks) of ONE warm-started CFL solve at a density where CFL wins,
  attributing the time across: `modes_thermo` (the jitted kernel), the pairing
  spectrum, `gapless_momenta`, `pair_hessian`, `state_at`'s Python glue, the
  charge rows, and `newton_solve`'s own overhead.
- **Counts, not only times**: how many residual evaluations, how many Jacobian
  evaluations, how many line-search halvings, per solve.
- The same breakdown for a 2SC solve (63 ms) and an unpaired solve (6 ms), so
  the cost that scales with the gap structure is separable from the fixed cost.

### Two traps this ticket will hit

- **Profiler overhead lands unevenly on jitted code.** cProfile sees a numba
  call as one opaque entry and inflates the Python frames around it. Cross-check
  the top few attributions with plain `perf_counter` before believing the split.
- **`pair_hessian`'s cost depends on the RG panel rule.** With
  `max_panel_ratio=2` the nodes MOVE with the state; a profile taken with
  frozen nodes is a different computation, not a cheaper one.

## Gate

- A table attributing >90% of one CFL solve's wall time to named functions.
- The residual/Jacobian/line-search counts stated.
- A one-line verdict: **what fraction of the 443 ms could a fully compiled
  Newton loop plausibly remove?** That number sizes ticket 06.

## Resolution, 2026-09-11

**The hypothesis is REFUTED, and the ticket's own premise with it.** The
443 ms warm-started CFL solve does not exist: a warm-started CFL solve at a
density where CFL wins costs **47 ms**, and Python orchestration is **4.6%**
of it, not most of it. 77% of a converged CFL solve is ALREADY the jitted
kernels.

Stack: python.org **3.14.2 / numpy 2.3.5 / scipy 1.17.0 / numba 0.63.1**, on
AC power. Another session held four cores at ~90% throughout; cpu and wall
agreed to 1.5% on every run, and the three tables below land within 6% of
ticket 02's pinned baseline (CFL 763 against 719 ms/pt, 2SC 58.8 against
56.5, unpaired 1.4 against 1.4), so the SPLIT is trustworthy and the absolute
times carry that band.

Harness: `profile_cfl.py` / `profile_table.py` / `crosscheck.py` / `rgsplit.py`
(scratch, not committed — they monkeypatch 30 call sites and belong nowhere in
`eos/`). Self-time by a wrapper stack, so a nested call never double-counts;
attribution reached **99.9%** of the run, and the wrappers cost between -4%
and +6% of the uninstrumented median.

### One warm-started solve, n_B = 1.1490 fm^-3

CFL is the ground state at all 57 benchmark densities in 1.00-1.30 fm^-3
(f = 1483.6 against 2SC's 1562.6 and unpaired's 1753.8 MeV/fm^3 at the
profiled point). The seed is the converged point one density above, the
sweep direction.

| pattern | wall | residual evals | Jacobian evals | Newton steps | line-search halvings |
|---|---|---|---|---|---|
| CFL | 47.1 ms | 3 | 2 | 2 | 0 |
| 2SC | 36.3 ms | 9 | 5 | 5 | 3 |
| unpaired | 1.3 ms | 3 | 2 | 2 | 0 |

The counts are derived, not guessed: `newton_solve` evaluates the residual
once on entry and once per line-search trial, so residual evals = distinct
`state_at` calls minus the one final state `solve_pattern` takes after the
loop; steps = `residual_jacobian` calls; halvings = trials - steps. All three
patterns close exactly, `solve_system` ran once and `scipy.optimize.root`
zero times on each.

**The ticket's per-evaluation estimate was about right and its step count was
5x too high.** ~11 ms of residual-plus-Jacobian against the estimated 15;
2.3 Newton steps from a warm start, not 12.

Breakdown of the 47 ms CFL solve:

| | self | % |
|---|---|---|
| `_pair_pass` (jitted quadrature) | 20.5 ms | 45.4 |
| `_pair_hessian_pass` (jitted) | 14.2 ms | 31.4 |
| `gapless_momenta` (crossing hunt) | 6.5 ms | 14.5 |
| `_unpaired_reference_hessian` | 1.0 ms | 2.2 |
| `_unpaired_reference` | 0.6 ms | 1.4 |
| everything Python (state glue, RG algebra, charge rows, rows, scales, `newton_solve`, `point_from_state`, counterterm, cache) | 2.3 ms | 5.0 |

### The 200-point CFL table: 152.6 s, 763 ms/pt

**89% of it is twelve points that never converge.** They sit at
n_B = 0.500-0.558 fm^-3, below the CFL branch, and cost 136.4 s — 11.4 s
each, **762 state evaluations, 42 Jacobians, 4 MINPACK runs and 2
`newton_polish` calls per point**. The ladder is `attempt(x0)`'s
Newton -> hybr -> lm -> polish, then the `not ok` differenced repeat of all
three; the cold retry never fires because once the branch dies the next point
has no warm seed and `warm` is False.

The 188 points that DO converge cost 15.1 s — **80 ms/pt**, on Newton alone,
every one realising CFL.

| | whole table | the 188 converged |
|---|---|---|
| jitted (`_pair_pass` hot + RG vacuum, `_pair_hessian_pass`, mode integrals) | 69.4% | **77.3%** |
| numpy, uncompiled (`gapless_momenta`, the two unpaired references, `pair_nodes`) | 27.0% | 18.0% |
| Python orchestration | 3.4% | **4.6%** |
| MINPACK itself | 0.2% | 0 |

Per call, over the table: hot pass 5.37 ms (9953), RG vacuum pass 3.03 ms
(12872), Hessian pass 5.14 ms (2572), `gapless_momenta` 3.06 ms (10895).
`gapless_momenta` costs 1.09 ms at a converged state and 3 ms averaged over
the table — the dead points have far more zero crossings to hunt.

The gap structure separates cleanly: 2SC costs 58.8 ms/pt with the same shape
(`_pair_hessian_pass` 23.3%, `_pair_pass` 39.7%, `gapless_momenta` 11.1%) and
unpaired costs 1.4 ms/pt with **no pairing at all** — there the whole cost is
`state_at` glue 37.9%, `residual_jacobian` 21.8% and `newton_solve` 17.9%,
i.e. pure orchestration on 4.4 states and 2.4 Jacobians. **Everything above
the unpaired 1.4 ms/pt is the pairing quadrature.**

### Two things measured on the way

- **The RG split costs three quadrature passes per residual and the cache
  misses two thirds of the time.** Hot pass 288 nodes / 4.75 ms; each vacuum
  pass 192 nodes / 2.75 ms; `rg_pair_block` 11.75 ms cache-cold against
  6.15 ms cache-hot. Over the table `_vacuum_pair_block` took 19906 calls for
  12872 misses — a **35% hit rate**, because the key is (M, Delta) and both
  move every Newton step. So the vacuum subtraction alone is **25.6% of the
  whole build and 14.0% of a converged solve**.
- **The analytic Jacobian is not the expensive half.** One Hessian pass costs
  5.14 ms against the hot residual pass's 5.37, and a converged point takes
  2.3 of them against 4.3 residuals. A differenced Jacobian would cost ~13
  residual evaluations per step; this is the right trade and is not a target.

### The two traps

- **Profiler overhead.** Cross-checked with plain `perf_counter`, median of 7,
  at the converged CFL state: `pair_block` one hot pass 5.22 ms against the
  wrapper's 5.12; `gapless_breakpoints` 1.09 ms against 1.09; `pair_hessian`
  8.91 ms against 7.08 + 0.50 of reference + assembly. Agreement within 2%.
  cProfile was used for call counts only (1829 Python calls in one warm CFL
  solve — itself the evidence that orchestration is not the cost) and its
  times are not quoted.
- **The RG panel rule.** Everything above is the SHIPPED rule
  (`max_panel_ratio = RG_PANEL_RATIO`, `Delta` passed to `pair_nodes`), nodes
  moving with the state: 288 nodes at the profiled point. For the record, the
  default rule with no `Delta` gives 216 nodes and a 3.40 ms pass — recorded
  as the trap says, a DIFFERENT computation and not a cheaper one.

### The verdict that sizes ticket 06

**A fully compiled Newton loop removes 4.6% of a converged CFL solve** — the
orchestration, and that is all there is. 1.05x. Extending "compiled" to every
uncompiled numpy block (`gapless_momenta`, the two unpaired references,
`pair_nodes`) and assuming they became FREE gives **23%**, or 1.3x. Against a
763x gap, compilation is not the lever and ticket 06 has been re-scoped to say
so.

The two levers this profile does expose:

1. **The rescue ladder on dead branches — 89% of this single-pattern table**,
   the same disease the map already measured as 92% of the four-pattern
   enumeration. `rescue=False` already exists in `solve_pattern`. Ticket 05.
2. **The pairing quadrature — 77% of a converged solve, already jitted.**
   Getting faster there means FEWER or CHEAPER passes, not compiled ones: the
   RG triple-pass with its 35% cache hit rate, and the 288-node rule nobody
   has measured the 1e-13 gate against. New ticket 10.
