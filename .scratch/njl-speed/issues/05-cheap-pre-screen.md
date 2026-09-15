# The cheap pre-screen: rank every pattern loosely, polish only the winner

Type: prototype
Status: closed
Assignee: guerrinimirco
Blocked by: 02
Parent: ../map.md

## Question

The user's own lever, in its safe form. `eos/njl/solver.py:882`'s `solve`
currently solves **every** enumerated pattern to full convergence and then ranks
the converged candidates by f = eps - T s. The default enumeration is four
patterns (`("unpaired", "2SC", "CFL", "free")`), and most of that work is thrown
away.

**Does a loose-tolerance ranking pass, with an early abort on dead branches,
recover most of the enumeration cost without changing which pattern wins?**

### Why this is safe where following the winner is not

Nothing is skipped. Every pattern is still attempted; a losing one is
**abandoned early** rather than never tried. The ground-state choice is made on
the same evidence as today, only cheaper. That is why this is adopted and
continuation along the sweep is not (see the map's fog).

### What the design already knows

- **The discriminator is the residual, not the realised pattern.** A dead CFL
  node stalls at err = 2.0e-01; a hard-but-live 2SC node stalls at 1.5e-10.
  Nine orders apart, so the abort test is unambiguous. Reading "probe failed"
  as "branch ended" is the wrong test and cuts the 2SC branch at n_B = 1.13.
- **The first Newton run already knows.** Past the CFL onset the probe returns
  err = 2.0e-01 in 285 ms and the retry ladder then spends **21.2 s more**.
  That one node was 84% of a 200-point build.
- `solve_pattern(..., rescue=False)` already exists: one damped-Newton run, no
  hunt. It is the pre-screen's natural primitive.
- **A collapsed candidate must not seed forward.** `solve`'s `_seeds` already
  excludes candidates whose `pattern_realised != pattern`; the pre-screen must
  keep that, or a collapse propagates down the whole sweep.

### Open design choices this ticket settles

- The loose tolerance, and whether it is absolute or scaled by
  `residual_scales`.
- The abort threshold: a fixed number, or a margin against the best candidate
  so far.
- What happens when the loose ranking is **close** — two candidates within the
  loose tolerance of each other in f. Polishing both is the obvious answer and
  must not be quietly skipped: near a first-order boundary that is exactly where
  the ranking flips.

### Measured at ticket 02, on the pinned benchmark

The premise is not just confirmed, it is bigger than the ticket assumed.

- The four singles sum to **845.7 ms/pt**; enumerating the same four costs
  **8100.2**. The enumeration is **9.6x the sum of its parts**.
- Adding `free` to the three-pattern list adds **7463 ms/pt** (636.7 -> 8100.2),
  and `free` swept ALONE costs **68.5 ms/pt**. So `free` is not an expensive
  pattern; it is a pattern that collapses onto rivals' roots, and the retry
  ladder this ticket already names is what the 7463 ms buys. **92% of the
  default table is that ladder.**
- **Do not assume dropping a pattern saves its own cost.** Three patterns
  (636.7 ms/pt, 200 rows) is CHEAPER than CFL alone (719.3 ms/pt, 188 rows):
  the rivals seed CFL, and a CFL-only sweep loses 12 densities as well as time.
  A pre-screen that abandons a candidate early must not also deny the survivors
  that candidate's seed.

So the first thing to try is the abort on a DEAD branch (err ~ 2e-1), not a
loose tolerance on live ones: the dead branches are where the 7463 ms sits.

## Gate

- Measured ms/pt on the pinned benchmark (ticket 02), default enumeration,
  before and after.
- **The winning pattern is unchanged at every point of the benchmark grid**, and
  P agrees to 1e-8 (the map's gate). Report the worst point, not only the median.
- Behaviour at a boundary crossing is exercised: a grid straddling the
  2SC-to-CFL switch, with the pattern sequence printed either side.
- The `rescue=False` default path stays byte-identical for callers that do not
  opt in, or the change is explained.

## Resolution, 2026-09-11

**The lever does not exist, and the 92% is not the retry ladder.** Both halves
of this ticket's premise are refuted by the census below, and what replaces
them is not a solver tactic but a scoping question the map has to answer.

### Why a loose-tolerance ranking pass has nothing to bite on

`eos/general/solve.py`'s `solve_system` **already screens, for free**: handed a
Jacobian it runs `newton_solve` FIRST and returns the moment that succeeds
(solve.py:117). So `solve_pattern(..., rescue=False)` is not a cheaper probe
that a pre-screen could add — for every candidate that works it is exactly
what already runs, and there is no second, fuller solve to skip. Measured:
**587 of 800 candidates held the layout they were asked for — 572 of them
meeting the 1e-10 gate on that first Newton run alone — and they cost 32.0 s
between them, 1.8% of the build.** Polishing only the winner saves
nothing, because nothing else was polished.

And on the candidates that do NOT work, the residual does not discriminate.
The ticket's "nine orders apart" holds only for a candidate seeded in its OWN
layout. Cross-seeded from `seed_from`, a CFL candidate stalls at the same
8-9e-2 whether its root exists or not:

| n_B | CFL screen err | screen gaps (MeV) | what the ladder found |
|---|---|---|---|
| 0.500 | 8.9e-02 | (49.4, 86.2, 244.4) | collapsed to 2SC |
| 0.550 | 8.2e-02 | (41.9, 69.1, 252.5) | collapsed to 2SC |
| 0.600 | 9.8e-02 | (156.8, 161.8, 195.9) | **a real CFL root** |

Across the grid: `collapsed -> 2SC` screens at 8.2e-2 (p10) to 9.4e-2 (p90),
and the one CFL candidate that the ladder rescues into its own layout screens
inside that band. **An abort threshold that kills the dead 13 also kills the
CFL onset**, and with it 170 of 200 rows. The stalled iterate's gaps do not
separate them either (all three alive in both cases above).

### Where the time actually is: `free` is denied a warm start, structurally

Census of the pinned benchmark (ticket 02's config, `NJL_BENCH` block of
`notebooks/quark_timing.py` at `54c7be9`), 200 densities, 800 candidates,
instrumented: 1824.7 s wall / 1749.3 s cpu = **9123.7 ms/pt**.

| pattern | seed | n | total s | % of build | median ms | max ms |
|---|---|---|---|---|---|---|
| `free` | cross | 200 | 1677.8 | **91.9%** | 1137.2 | 57423.0 |
| `CFL` | cross | 14 | 117.0 | 6.4% | 7392.9 | 16710.8 |
| `CFL` | own-warm | 186 | 17.4 | 1.0% | 87.3 | 199.5 |
| `2SC` | own-warm | 199 | 12.1 | 0.7% | 46.2 | 280.2 |
| `unpaired` | own-warm | 199 | 0.4 | 0.0% | 1.6 | 10.4 |

`free` is cross-seeded at **all 200** densities, and not by luck — by
construction. `eos/general/pairing.py`'s `realised_pattern` returns one of the
eight names of `_REALISED`, and **`'free'` is not one of them**: it is a SEED,
not a state. So `solve`'s seed filter, `p.pattern_realised == p.pattern`
(solver.py:948), **cannot ever pass for the free candidate**. It is denied a
warm start in every sweep this library runs, and pays a hunt from the unpaired
state's potentials with an asymmetric gap seed at every density, forever.

That is the mechanism behind ticket 02's "adding `free` adds 7463 ms/pt". It
is not a retry ladder on a dead branch: `free` CONVERGES at 173 of 200
densities. It is a seeding defect.

### What `free` buys on this benchmark: nothing

- It realises **CFL at 141** densities and **2SC at 26** — pure duplicates of
  candidates that already converged in their own layouts. 649.2 s.
- It realises **uSC at 33**, of which **27 do not converge**. 1028.6 s.
- The 8 uSC states that do converge **never win**, and not narrowly:
  f(uSC) - f(winner) is **+29.6 to +41.5 MeV/fm^3**.
- The winner over the whole grid is `2SC` at 30 densities and `CFL` at 170.

Related, and it changes this ticket's own gate: the reported `pattern` column
is **not stable in the baseline**. `free` and `CFL` share a layout and land on
one root, so the `min(f)` tie flips on the last digit — the baseline reports
`pattern = 'free'` at 40 densities where `pattern_realised` is `CFL`. The gate
below is therefore judged on **`pattern_realised`**, the name that says what
the matter IS.

### Measured: four variants, one loaded window

All four in the same session at loadavg 7.4-10.2 (another session held four
cores at ~95% throughout), python.org **3.14.2 / numpy 2.3.5 / scipy 1.17.0**,
AC power, cpu within 6% of wall on every run. **The ratios are matched; the
absolute numbers carry the load.** (A later three-pattern run on a quiet
machine came in at 267.2 ms/pt where V3 measured 730.4 — same configuration,
2.7x apart. Absolute ms/pt from this window is not a pinned number.)

| variant | ms/pt | vs V0 | what `free` realises | gate |
|---|---|---|---|---|
| **V0** control, unmodified | 9123.7 | 1x | CFL 141 / uSC 33 / 2SC 26, 27 non-conv | reference |
| **V1** `free` carries its own seed | 760.5, 769.7 | **12.0x** | **2SC 200** — captured | PASS |
| **V2** V1 + a cold re-hunt every 10 | 4496.2 | 2.0x | CFL 148 / 2SC 30 / uSC 22, 15 non-conv | PASS |
| **V3** `free` dropped (three patterns) | 730.4 | **12.5x** | — | PASS |

Gate, all three against the V0 table: **200 of 200 rows, zero
`pattern_realised` mismatches, worst |dP|/P = 6.654e-10 at n_B = 0.916834**
(map gate 1e-8). The 2SC -> CFL crossing sits at **n_B = 0.6583 fm^-3** in V0
and in every variant, with `2SC` below and `CFL` above in each. `solve_pattern`
was not touched, so the `rescue=False` path is byte-identical.

**V1 is a trap.** Warm-starting `free` captures it completely: it realises 2SC
at all 200 densities and never probes again. The 12x is bought by turning the
probe into a no-op — which makes V1 strictly worse than V3, which does the
same thing honestly, 4% faster, and without leaving a dead candidate in the
enumeration. **V2 is the price of keeping the probe alive: 6x of the 12.5x.**

### What this ticket decides, and what it hands on

- **The pre-screen is not adopted.** There is nothing for it to skip.
- **The `free` candidate is the whole question**, and it is a physics-scope
  decision rather than a solver tactic: on one parameter set at T = 0 in beta
  equilibrium it costs 12.5x and changes no delivered row, but it is the only
  thing in the enumeration that can find uSC/dSC/sSC where those ARE the
  ground state. That decision is [Does `free` belong in the default
  enumeration?](12-free-in-the-default.md).
- **After `free`, the dead-branch ladder is what is left**, and it is the
  ticket's original target in the place it actually lives. Stage attribution
  of a three-pattern build (600 candidates, 53.4 s, quiet machine, shares
  only): **587 candidates take one `solve_system` call and 22.9% of the time;
  13 take two and 77.1%.** Those 13 are `CFL -> converged as 2SC` at 39.2 s
  (73.3%) — and exactly **one** candidate in 200 is rescued into a real CFL
  layout by the same ladder, at the onset. The differenced rescue (no
  Jacobian) fired 13 times for 11.2 s (21.0%) and produced **only duplicates**;
  the reinflate rescue never fired at all. See [Bound the rescue ladder below
  a branch](13-bound-the-rescue-ladder.md).

Harness: `.scratch/njl-speed/proto05_{instrument,variant,stages,analyse,
stages_analyse,gate}.py` and the `proto05_*.json` they wrote (scratch, not
committed — they monkeypatch `solve_pattern`, `solve_system`, `newton_solve`
and `seed_from`, and belong nowhere in `eos/`). No `eos/*.py` was modified.
