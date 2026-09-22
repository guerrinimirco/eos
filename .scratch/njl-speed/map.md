# Map: NJL colour-superconducting matter, fast enough for hybrid stars and inference

Label: `wayfinder:map`
Effort: `njl-speed`
Charted: 2026-09-08

## Destination

A **proving prototype** — not a production port — that demonstrates, on a pinned
and repeatable benchmark, that `eos.njl` with `SpeciesFlags(csc=True)` reaches:

- **~1 ms per n_B point** on a `beta_eq_neutrinoless`, T = 0 CSC table, every
  point a **converged solve** (no interpolation, no surrogate); and
- **a 200-point hybrid EoS with a complete mixed phase at one T in under 60 s**
  through `eos.mixed`.

and that the route taken is **mode-agnostic** — demonstrated, not assumed, by
measuring the same acceleration at `fixed_YC` with leptons and at T > 0 before
the map closes.

Reached when those three numbers are measured and the production port is
specified for a hand-off session. **The port itself is not this map's work.**

**Today, MEASURED** (ticket 02's pinned benchmark, python.org 3.14.2 / numpy
2.3.5 / scipy 1.17.0, `rg_njl1`, `csc=True`, beta-eq, T = 0, `backend="fast"`,
200 densities over 0.5-1.55 fm^-3, median of 3, cpu and wall agreeing to 2%):

| what | now | to 1 ms/pt |
|---|---|---|
| **historical** -- `eos_table`, the old 4-pattern default with `free` ([ticket 15](issues/15-land-the-pattern-default.md): `free` left `DEFAULT_PATTERNS` in `2536d2b`; the default is the 3-pattern row below) | ~~8100 ms/pt~~ | -- |
| `eos_table`, restricted to 3 patterns | 637 ms/pt | 637x |
| the same, the DEFAULT since `2536d2b`, with the bounded ladder ([ticket 16](issues/16-land-the-bounded-ladder.md), `50b3b7f`) | **2.51x** below it in one window (419 -> 167 ms/pt), i.e. ~254 ms/pt on this row's scale | ~250x |
| `eos_table`, one pattern: unpaired / 2SC / CFL / free | 1.4 / 56.5 / 719 / 68.5 ms/pt | — |
| of that CFL table (ticket 03): 12 dead points / 188 converged | 89% / 80 ms/pt | 80x on the rows |
| the converged solve at a 12-node pairing rule, T = 0 only ([ticket 10](issues/10-the-quadrature-itself.md), NOT landed) | 1.67x below it, i.e. ~48 ms/pt | ~48x |
| one `eos.mixed` point, DID+NJL, 3 patterns / held 2SC | 29.6 s / 9.2 s | — |
| the mixed window: 52 of 200 densities inside it | 52 x 29.6 s = 1539 s | 26x |
| `eos.mixed` window location, 20 densities | 2576 s | dominates |

and at ticket 04, the same configuration instrumented rather than timed
(counts deterministic to the call; wall +-30%, cpu quoted beside it):

| what | count | wall |
|---|---|---|
| one hybrid row | 40 residuals / 41 `thermo` / **123 NJL solves** | 12.8 s onset, 28.8 s deep |
| `locate_window`, HINTED (the cheap form) | **5,700 NJL solves** | 3,385 s |
| the same after [tickets 11](issues/11-cheapen-the-locator.md)/[17](issues/17-methods-bound-in-mixed.md) (`66d713c`), window bit-identical | **3,084 NJL solves**, NJL evals ÷2.35 | not quotable (loaded) |
| `locate_window`, UNHINTED (what `build_hybrid_table` calls), HEAD -> `66d713c` | **10,425 -> 8,451 NJL solves**, evals ÷1.48 | not quotable (loaded) |
| whole 200-point build (51 rows + wings + locator) | **~12,100 NJL solves** | ~4,460 s, i.e. **74x** |
| the per-call budget for 60 s | — | **15.2 ms / `thermo`, 4.95 ms / `thermo_from_mu`** |

**`unpaired` is already at 1 ms/pt**, on the exact-solve path, so the target is
not absurd for this solver: the entire gap is what the pairing sector costs on
top of it. The mixed target is ~26x on the rows and ~68x once the locator is
counted — the map's inherited ~70x was right for the wrong reason, since it
assumed 355 s per point and no locator.

**The lever the map expected is not the lever the numbers show.** "Solve fewer
patterns" is not multiplicative: three patterns (637 ms/pt) is CHEAPER than CFL
alone (719), because rivals seed each other. What IS multiplicative is the
handling of candidates that collapse — the four singles sum to 846 ms/pt while
enumerating the same four costs 8100, and adding `free` alone (68.5 swept by
itself) adds 7463 ms/pt to the enumeration.

**Ticket 05 found what that 7463 is, and it is not a retry ladder.** 91.9% of
the default build is the `free` candidate, which **can never carry a warm
start**: `realised_pattern` never returns `'free'`, so `solve`'s
`pattern_realised == pattern` seed filter cannot pass for it, and it hunts from
cold at every density of every sweep. It CONVERGES at 173 of 200 of them. The
92% was a seeding defect wearing the ladder's clothes — and the ladder is real
but smaller: 13 dead CFL candidates, 73% of what is left once `free` is out.

**Ticket 03 confirmed that inside a SINGLE pattern too**, where there is no
enumeration to blame: 89% of the CFL-only table is twelve points below the
branch, each spending 762 state evaluations on `attempt`'s Newton -> hybr ->
lm -> polish ladder and its differenced repeat. And it closed the other
question the map was guessing at — **the remaining 80 ms/pt of a converged
solve is 77% jitted quadrature and 4.6% Python**, so compilation is spent and
the only thing left to make cheaper is the quadrature itself (ticket 10).

## Notes

**Domain.** Nuclear/quark-matter equation-of-state library. `CLAUDE.md` at the
repo root is the specification and overrides defaults. Prior art this map sits
on: `eos/njl/backends/jacobian.py` (analytic Jacobian, 4-7x where CFL wins),
`eos/njl/table.py`'s `build_fast_table` (18 ms/pt but structurally beta-eq
only), `eos/general/pairing.py`'s `pair_hessian`.

**Skills every session should consult.** `mattpocock-skills:diagnosing-bugs`
for the profiling tickets; `mattpocock-skills:prototype` for 05, 06, 07;
`mattpocock-skills:grilling` + `domain-modeling` for 09.

### Settled while charting

- **Exact solves only.** Every delivered point is a converged solve. Speed
  comes from cheaper and fewer solves, never from interpolating between them.
  Interpolation and surrogates are recorded below, not pursued here.
- **The correctness gate is: same pattern, and P to 1e-8.** Not bit-identical
  (a different iteration path lands on the same root at a different last
  digit), and `test/baseline/`'s rtol = 1e-10 is preserved wherever it already
  holds. **Trap:** four `test/baseline/` entries (`ccdm`, `enjl`, `njl`,
  `zlvmit`) already fail on the anaconda 3.9 stack as 3.14 artifacts,
  pre-existing at HEAD. Check HEAD before chasing a baseline regression.
- **One branch, `njl-speed`.** Not `main`: this checkout is shared, and
  CLAUDE.md section 12's landing measurement is blocked by uncommitted
  `eos/*.py`.
- **One benchmark, in `notebooks/quark_timing.py`.** No second harness. Every
  ticket quotes that number, n = 3, median, with the interpreter and its numpy
  and scipy versions named. Timing on this laptop varies +-30% with concurrent
  work, so a single run is not a measurement. **Never wrap a measurement in
  `timeout`** — it is an x86_64 binary here and drags the interpreter under
  Rosetta, breaking numpy with an error naming a cause it does not have.
- **`eos/mixed` is in scope**, but its phase-adapter contract does **not** need
  to change: `Phase.jacobian_block` already exists and `njl_phase` simply does
  not implement it. See ticket 07.
- ~~**Pattern prediction is the cheap pre-screen and nothing else**: solve
  every pattern to loose tolerance, rank by f, polish only the winner.~~
  **REFUTED by ticket 05 and not adopted.** `eos/general/solve.py`'s
  `solve_system` already runs Newton first and returns the instant it
  succeeds, so a candidate that works is already screened for free — 587 of
  800 candidates, 1.8% of the build — and there is no fuller solve to skip.
  On candidates that do NOT work the screen residual does not discriminate: a
  cross-seeded CFL stalls at 8-9e-2 both where its root exists and where it
  collapses. **Ticket 13 closed the two remaining ways to guess**: the
  iterate's LAYOUT does not discriminate either (it is still CFL at every rung
  boundary; the collapse happens inside the differenced rescue), and neither
  does an evaluation budget (the one rescued candidate uses MORE `hybr`
  evaluations than any dead one). What discriminates is not a property of the
  solve at all -- it is where the seed came from.
  Following the winner along a sweep is still **not** adopted and is still in
  the fog below.
- **A timing number from this map states the machine's load.** Ticket 05
  measured the same three-pattern configuration at 730 ms/pt against a
  concurrent session and at 267 on a quiet machine — 2.7x apart, cpu tracking
  wall in both. Ratios are taken inside one window; absolutes are not
  comparable across them.

## Decisions so far

<!-- one line per closed ticket -->

- [The bounded ladder loses the gapless CFL ground state](issues/18-bound-loses-gapless-cfl.md):
  **neither the bound nor the analytic Jacobian is what fails -- the seed
  is, and on both backends.** A gapped CFL state at T = 0 is an insulator
  for the rotated charge Q~ (d mu_C = t, d mu_3 = -t, d mu_8 = -t/2, under
  which every CFL pair is neutral), so at a nonzero fixed Y_C, whose charge
  row is quark-only, the residual is EXACTLY flat along it until a
  Q~-charged pair unlocks; the scaled Jacobian at the cross seed is singular
  there to 1e-17, and the gapless root lies a finite distance out (the
  unlocking edge moves 69 -> 139 MeV over n_B = 0.8 -> 1.075). The exact
  Jacobian drops the direction and stalls in the gapped valley (the stall is
  the root's mirror image); any differenced or jittered one steps along it
  with the sign of its round-off. So **ticket 10's "rung D reaches the root"
  was a coin**, and so is the reference backend's 24/12 loss: the analytic
  Jacobian matches central differences to <= 9.4e-7 on the whole bounded
  path, which never becomes gapless, while a 1e-6 jitter of it flips the
  outcome. The ticket's candidates 4 (fix `_crossing_terms`) and 2 (rung D
  at T = 0) are refuted -- 2 would also give back the 2.51x, the pinned
  benchmark being T = 0 -- and fixed displacements along Q~ are not robust
  either. **Landed: `unlocked_seed`** moves a CFL seed on the plateau to just
  past its own unlocking edge (bisection on the state's `gapless` flag), for
  pattern CFL, T = 0, fixed Y_C != 0 only; beta equilibrium never reaches it.
  Gate PASS: gapless CFL at all 13 densities 0.775-1.075, P monotone, on both
  backends under 24/24, 24/12 and 32/32 (HEAD lost 13 / 0 / 7 fast and 0 / 7
  / -- reference), realised state identical to the known-good table at all
  45 densities, |dP|/P <= 6.7e-10; a both-backend regression test red on HEAD
  on both and green here. The pinned benchmark's rows are bit-identical
  to HEAD on both backends and its cpu within 1% (the window was contended,
  loadavg 13-30, so the absolute 2.51x is not re-measured, only kept).

- [The pairing quadrature itself](issues/10-the-quadrature-itself.md):
  **at T = 0 the shipped 24 nodes per panel buy nothing the gate can see; at
  T > 0 they are what the gate needs; and the RG vacuum half needs half of
  them at every T.** Arms H/V (hot pass / vacuum passes) 24/24 down to 8/8
  plus vacuum-only 24/16-24/8, on the pinned single-pattern tables, both
  backends: 2SC 200/200 and CFL 188/188 rows at every arm, 0 realised
  mismatches, worst |dP|/P 9.9e-10 (2SC, where the CONTROL is the outlier) and
  4.6e-9 (CFL onset, vacuum-only arms equally). Repeats bit-identical, and the
  rule changes no Newton count. **Converged CFL solve: 1.37 / 1.67 / 1.91 /
  2.16x at 16 / 12 / 10 / 8 nodes, 1.22x vacuum-only at 12**, n = 3
  interleaved, so ~48 ms/pt at 12 on ticket 03's 80. Three limits: (1) **12
  nodes fails the P gate at T = 20-30 MeV** (1.3e-7 .. 1.3e-6; 16 is 2e-9 in
  P and 3e-8 in s) because the thermal collars are hundreds of MeV wide; the
  vacuum passes are T = 0 by construction, so vacuum-at-12 holds at every T.
  (2) The Lambda_UV vacuum is the slow pass, and only in 2SC with M_u ~= 10
  MeV, a LAYOUT effect: its geometric panels stop at 47 MeV (breakpoints at
  the masses fix N = 8 in isolation, 88 nodes against 192). (3) A default
  change moves `test/baseline`'s njl keys past 1e-10 (vacuum-only 12: 9 of
  139, six of them quantities pinned only to solver resolution, which is a
  hygiene finding of its own), so landing is its own ticket with an `njl.npz`
  regeneration; `NODES_PER_PANEL` is shared with every unpaired integral and
  stays. **The pinned tables contain no gapless state**; at the documented one
  (`fixed_YC`, Y_C = 0.1, T = 0) a held CFL moves <= 6.3e-10 at all 34 gapless
  densities under every rule, and the enumeration's ROOT SELECTION there is a
  lottery any perturbation enters: **the shipped rule on HEAD fast loses
  gapless CFL at 13 densities** (a finer 32/32 loses 7, coarse ones keep it).
  That is [ticket 18](issues/18-bound-loses-gapless-cfl.md), a regression of
  `50b3b7f`'s bound: `428cd66` is right there, and the bound removed the one
  rung (the differenced repeat) that reaches the root from a cross seed.
  What is left on a converged solve at 12/12: still 65% jitted quadrature,
  of which the RG vacuum half is 34%. **Its Hessian alone was 19% at the
  shipped rule, a cost ticket 03 did not separate, hitting its cache 5% of
  the time.** And `gapless_momenta` does not scale with the rule: 17% of a
  converged solve and 35% of the CFL table at 12/12, scanning for crossings
  in states that have none.

- [Does the `methods` bound pay inside `eos/mixed`?](issues/17-methods-bound-in-mixed.md):
  **yes, and only where the build discards its work.** `lm` is 1.7-4.2% of
  the NJL work in a mixed ROW (so not where the row time goes -- the
  ticket's own trap, answered) and **18-24% of it in `locate_window`**,
  nearly all at trial potentials of mixed solves that fail either way. The
  discriminator is not ticket 13's seed origin (every candidate here is
  seeded from the per-solve constant) but **whether the candidate has
  rivals**: enumerating, a failed candidate is dropped and the max-P rival
  answers; held to one pattern it is the phase. Measured with an EXACT
  per-candidate counterfactual (the bounded call is `hybr` then the polish
  from its own iterate), 4 of 687 candidates `lm` rescued would be lost,
  every one inside a solve that fails anyway. **Landed in `66d713c`** for the
  enumeration only; window bit-identical, 15 rows bit-identical in every
  field on BOTH backends (ticket 16's trap checked, not inherited).

- [Cheapen the locator](issues/11-cheapen-the-locator.md): **landed with 17
  in `66d713c`, window bit-identical on both forms; NJL work ÷2.35 hinted,
  ÷1.48 unhinted** (the unhinted form measured to completion for the first
  time: 10,425 NJL solves, and it does return). The cut was `sweep`'s retry
  ladder on the walk's failing DOWNWARD step: six levels re-trying a
  density with no mixed solution from ever closer midpoints, whose result
  the walk discards (833 of 1825 residuals hinted). Two corrections to
  ticket 04: **the onset was never exactly refined** -- the chi = 0 solve
  fails on this pairing, 0.8534 is the walk's tol midpoint -- and **the
  mixed branch does not start at chi = 0**: it ends at chi ~= 0.69 near
  n_B = 0.8516, so the "onset" is an existence boundary (physics or a second
  branch: not decided). What is left is still ~57% doomed work unhinted: the
  scan's probes below the onset (31%, needs a design decision), two failed
  exact refines (23%), and the honest 15-step walk (27%, a third of it
  re-solves). At ticket 09's target the landed locator is **~3-4 s hinted,
  ~9-10 s unhinted**, so the whole build lands near 15-16 s against 60: the
  locator no longer decides whether 60 s is reachable. Clock not quotable
  this session (loadavg 70-430, battery died mid-run).

- [Land the bounded rescue ladder](issues/16-land-the-bounded-ladder.md):
  **landed in `50b3b7f`, 2.51x on the pinned benchmark** (437.6 -> 174.3 ms/pt
  on the default, 2.10x across chiral restoration, cpu within 5% of wall; the
  unreachable single-pattern rows drifted 0.84-1.17x in the same window), and
  bit-identical to ticket 13's VX on all 200 rows -- **but only after a
  departure from the edit as written: the bound applies only where there is
  an analytic Jacobian.** Written, the edit also bound `backend="reference"`,
  `eos_table`'s default, where a cross seed is left `hybrd` alone, and it
  **lost 30 of 30 CFL rows** over the pinned grid's first 60 densities: the
  CFL candidate stalls at 1e-6 where the branch begins (0.5686, found there
  by `lm` or the cold retry), then converges onto a **second CFL root ~1
  MeV/fm^3 higher**, seeds itself from it and tracks it, and the onset is
  never found. That is the map's capture hazard, created by a bound. The fix
  is `rescue=False`'s precedent (ignored without a Jacobian); restricted, the
  reference sweep is bit-identical to the parent. **All four test directories
  passed on the broken edit**, so a both-backend regression test now pins it
  (local; `test/` is gitignored). Two corrections handed on: `eos/mixed` IS
  reached, through `njl_phase`'s `cold_start`/`seed`/`wing_sweep`, whenever
  it runs `backend="fast"`; and [ticket 17](issues/17-methods-bound-in-mixed.md)
  now starts from a prior AGAINST bounding `lm` on a Jacobian-free path.

- [Land the pattern-default decision](issues/15-land-the-pattern-default.md):
  **closed; the default change changes no delivered ccdm point, but ticket
  12's premise does not transfer to ccdm.** Ten further stale sites fixed in
  `50b3b7f`, among them `table.py`'s "recommended fast restriction", the
  refuted retry-ladder account of `free`'s cost, and 6 / 63 / 443 / 6638
  ms/pt. ccdm probed cold at 12 points (T = 0/30/50, n_B = 1.3-2.5):
  - the three- and four-pattern defaults return the **identical state and
    f at all 12**;
  - at T = 30 `free` reaches uSC states at **3 of 12** where the NAMED `uSC`
    seed collapses, losing by 4.3-8.5 MeV/fm^3. That meets the ticket's
    reopen condition as written, and is handed to
    [ticket 14](issues/14-usc-dsc-in-the-default.md);
  - **the first winning asymmetric state**: uSC at T = 50, n_B = 1.3, by
    0.20 over 2SC;
  - a defect that predates this work: ccdm's enumeration **misses the CFL
    ground state at 3 of 12** (+4.3, +22.6, +6.0 MeV/fm^3), with or without
    `free`, because its cross-seeded CFL candidate collapses, keeps the name
    and competes, and `eos/ccdm/solver.py` has no layout filter at all.
    **It needs its own ticket.**

- [Bound the rescue ladder below a branch](issues/13-bound-the-rescue-ladder.md):
  **most of it goes, and the separator is WHERE THE SEED CAME FROM** -- not a
  residual, not an evaluation budget. A candidate handed another pattern's
  state gets one Newton and one `hybr`; a no is then an ordinary answer, and
  it is reported non-converged rather than walked down `lm`, the differenced
  repeat and the cold retry. **2.02x on the pinned benchmark**, gate clean:
  0 `pattern_realised` mismatches, worst |dP|/P **7.009e-10**, the onset
  **identical to the last digit** at 0.6582914572865115, 170 CFL rows. It pays
  MORE off the benchmark than on it -- **2.70x** at T = 30 and **4.66x** at
  `fixed_YC`, T = 30, 20/20 rows and 0 state mismatches in all four sweeps --
  which is free evidence for ticket 08. **Both cuts the ticket proposed
  failed, and their failures are the finding.** Dropping rung D alone is a
  **LOSS** (0.95x): `ok = False` fires `solve_pattern`'s cold retry and the
  whole ladder runs twice. A layout stop fires **once in 600 candidates** --
  the collapse happens INSIDE rung D, so there is nothing to detect until the
  work is done. And an evaluation cap is measured shut: the onset uses **77**
  `hybr` evaluations, MORE than any of the 13 dead ones (31/55/73). The census
  that picked the cut: the 13 rootless CFL candidates are **77.2%** of the
  build, `lm` is entered by exactly those 13 and rescues **none**, rung D
  converges all 13 onto the 2SC root -- duplicates of a 2SC candidate that
  already won, agreeing in f to **1.9e-10** -- and rung C fires **0** times.
  The onset is rescued by `hybr` and never reaches `lm` or D. Two things
  handed forward: bounding the ladder takes the rootless candidates from 77%
  to **54.7%** of the build but does not remove them, and there is no further
  cut inside the ladder, so the next lever is the PROPOSAL, which is the
  continuation fog below -- now priced, and constrained by the measurement
  that the onset is found only because CFL is still proposed after 13 failures
  in a row. Landing it is
  [ticket 16](issues/16-land-the-bounded-ladder.md); whether it reaches
  `eos/mixed` is [ticket 17](issues/17-methods-bound-in-mixed.md), and it does
  NOT come for free -- `njl_phase` goes through `thermo_from_mu`, a path with
  no Newton rung and no cross-seeds at all.

- [Does `free` belong in the default enumeration?](issues/12-free-in-the-default.md):
  **it leaves `DEFAULT_PATTERNS`, in both models**, and the argument is not the
  12.5x — `free` is a **dominated probe**. The ticket's own premise was wrong
  where it mattered: `uSC` and `dSC` ARE named patterns with their own seeds,
  and both are in `_REALISED`, so both **warm-start** where `free` structurally
  cannot. Measured cold at 12 points over T = 0/30/50, n_B = 0.8–2.0
  (`t12_probe.json`): in **11 of 12** every asymmetric state `free` reached was
  reached by `uSC` or `dSC` too, cheaper and converging where `free` did not;
  in the 1 remaining it found a uSC state **losing by +95.7 MeV/fm^3**; and it
  is *less reliable*, collapsing to 2SC at T = 30, n_B = 1.2 where both named
  seeds held their layouts. **No asymmetric state wins anywhere in the box** —
  but Gholami+ 2025's dSC melting region is NOT inside it, so that evidence is
  owed by ticket 14, not by this one. Baseline exposure is **two numbers at one
  density and they survive**: only `enumeration.n1.2.{f,Delta}` uses the default
  enumeration (the sweep half is `csc=False`, the paired half pattern-restricted)
  and it moves by **1.9e-13**, ~500x inside rtol = 1e-10. Section 4 does **not**
  bind this: `csc` is the sector and is untouched; this is a seed list. Caller
  guidance is **ask by the name of the state** —
  `patterns=("unpaired","2SC","CFL","uSC","dSC")` — with `free` still legal for
  the masks no pattern names (`sSC`, `usSC`, `dsSC`, unequal-gap). Two things
  handed forward: `_left_layout` drops a collapsed candidate only for
  `("2SC","CFL")`, so it must widen **before** uSC/dSC could join a default
  (ticket 14's precondition, not its discovery); and the map's 8100 ms/pt
  "default enumeration" row is now **historical** — the default is the
  three-pattern list.

- [The cheap pre-screen: rank every pattern loosely, polish only the winner](issues/05-cheap-pre-screen.md):
  **not adopted — there is nothing to skip**, and the 92% is a seeding defect,
  not a ladder. `free` is **91.9% of the default build** and is cross-seeded at
  all 200 densities because `realised_pattern` never returns `'free'`; it
  converges at 173 of them and changes **no delivered row** — it duplicates CFL
  (141) or 2SC (26), or reaches uSC (33, of which 27 do not converge and the 8
  that do lose by **+29.6 to +41.5 MeV/fm^3** in f). Dropping it is **12.5x**
  (9124 -> 730 ms/pt, same loaded window), gate clean: zero `pattern_realised`
  mismatches, worst |dP|/P **6.654e-10**, the 2SC -> CFL onset still at
  **n_B = 0.6583 fm^-3**. Two traps it leaves: **warm-starting `free` CAPTURES
  it** (200/200 densities as 2SC — a 12x bought by making the probe a no-op),
  and keeping the probe honest costs 6x of the 12.5x; and the reported
  `pattern` column is **not stable in the baseline** (`free` wins the tie at 40
  densities where the state is CFL), so this map's gate is judged on
  `pattern_realised`.

- [Where do the 443 ms of a CFL solve actually go?](issues/03-profile-one-cfl-solve.md):
  they do not exist. A warm-started CFL solve at a density where CFL wins is
  **47 ms**, 2.3 Newton steps, and **4.6% Python** — 77% of it is the jitted
  pairing quadrature and 10.5% is `gapless_momenta`. **A fully compiled Newton
  loop buys 1.05x**, or 1.3x if every uncompiled numpy block joined it and
  became free, so ticket 06 was re-scoped to the one block worth it. The
  bigger finding is the same one ticket 02 made about the enumeration, now in
  a SINGLE pattern: **89% of the 200-point CFL table is twelve points at
  0.500-0.558 fm^-3 that never converge**, 11.4 s and 762 state evaluations
  each on the rescue ladder; the 188 that do converge cost 80 ms/pt. Two
  measurements on the way: the RG triple-pass is 25.6% of the build with a
  **35% cache hit rate** (the key is (M, Delta) and both move every Newton
  step), and the analytic Jacobian is NOT the expensive half (one Hessian pass
  5.14 ms against the hot residual pass's 5.37).

- [How many NJL solves does one hybrid row actually cost?](issues/04-count-the-mixed-loop.md):
  a row is **~40 mixed residuals = ~41 quark `thermo` calls = ~123 NJL internal
  solves**; a 200-point table is **~12,100 solves in ~4,460 s**, so the
  inherited estimate was RIGHT and the target does not move. The real per-call
  budget is **15 ms per `thermo` call** (4.95 ms per `thermo_from_mu`), 5x
  looser than the ~3 ms assumed. **Cost, not count** — 12,100 solves cannot be
  argued down, and njl's own 1 ms/point destination is 15x better than the
  budget needs. (It first named ticket 06 as the route; ticket 03 landed
  concurrently and measured that at 1.05x, so the route is **tickets 10 and 05**
  — the verdict COST-not-count is unchanged and reinforced.) The two tickets
  cross-check: the mixed loop pays **208 ms per single-pattern solve against
  03's 80 ms converged**, so ~2.6x of every mixed call is the SAME rescue
  ladder ticket 05 owns — 05 is worth as much inside `eos/mixed` as in
  `eos/njl`, which neither ticket could see alone. Three things it
  overturns: the FD Jacobian is **9-24% of residuals** (`hybr` forms it ONCE
  per `root` call and Broyden-updates), so ticket 07's "~6x" is a **1.1-1.3x**
  ceiling; the cost is **flat in count and 2x in per-call cost across the
  window**, expensive at the QUARK end, not the onset; and `did_phase` carries
  no `jacobian_block` either, which disables `analytic_jac` for the whole pair.
  The one genuine COUNT problem is the **locator: 47% of the solves and 76% of
  the wall**, hinted — its unhinted form is what a table actually calls, and it
  is now [ticket 11](issues/11-cheapen-the-locator.md).
  Counts are deterministic to the call; only the clock moves.

- [Pin the benchmark and take the baseline](issues/02-pin-the-benchmark.md):
  the config is committed as **`54c7be9`**, the last cell of
  `notebooks/quark_timing.py`, self-contained and run from the COMMIT (a
  jupytext save deleted it from the working tree once). Numbers are the table
  above. Four things it settled beyond them: the enumeration costs 9.6x the sum
  of its parts and `free` is why; dropping a pattern can cost time AND rows;
  holding a pattern in the mixed solve is neither faster nor safer (held CFL
  109.5 s and non-convergent where the enumeration takes 29.6 s, held 2SC lands
  on a different chi); and **cpu time is printed beside wall time**, because
  three earlier runs were silently throttled to a 5% duty cycle on battery.
- [Land the uncommitted NJL fast-table work and open `njl-speed`](issues/01-land-the-fast-table-work.md):
  baseline is **`d6d9e7c`** on branch **`njl-speed`**; the NJL/mixed group
  (incl. `docs/DEFERRED.md`) landed as one commit and the J0614 group was left
  to its owning session. `test/njl` 130, `test/mixed` 272, `test/baseline` 20 —
  **all passed, zero failures**, on Python 3.14.2 / numpy 2.3.5 / scipy 1.17.0,
  so no HEAD control arm was needed. Two eos `constraints` files stay
  uncommitted (not ours), which blocks a full-suite landing measurement but no
  ticket here. Run the suites **one directory at a time** — `test/njl` and
  `test/mixed` both hold a `test_jacobian.py` and collide at collection.

## Not yet specified

- **Continuation along the sweep.** Enumerate fully once, then follow the
  winner, re-enumerating only when a monitor fires (a gap crossing zero, the
  free-energy margin over the runner-up narrowing). **Ticket 13 put a price
  and a hard constraint on this.** The price: with the rescue ladder bounded,
  the candidates that have no root are STILL **54.7%** of the build -- 13 CFL
  candidates below the onset, each paying the one `hybr` the bound leaves
  them, and there is no further cut inside the ladder because `hybr` is what
  finds the onset. Not proposing them is the only lever left. The constraint:
  that onset, at n_B = 0.5686, is reached ONLY because CFL is still proposed
  and cross-seeded there **after 13 consecutive failures**, so a monitor that
  stops proposing a repeatedly-failing pattern loses 170 of 200 rows. This is
  where the fewer-patterns lever actually pays in full — but it can silently seat a
  metastable branch, which this model has already done once (tables reported
  metastable 2SC where gapless CFL was the ground state). **Ticket 05 measured
  the capture directly and it is not a tail risk:** given its own seed, the
  `free` candidate sat on one root for **200 of 200** densities and never
  probed again, and the bounded version — a cold re-hunt every ten — cost 6x of
  the 12.5x it was protecting. So the monitor is the whole design, not a
  safeguard bolted to it, and nothing here graduates until one is stated and
  its miss is priced.
- **The interpolated phase surface.** Build P(mu_B, mu_C, mu_S) per parameter
  set and temperature, once per pattern, and let every downstream consumer —
  beta-eq, `fixed_YC`, the mixed Gibbs solve — root-find against derivatives of
  that one interpolant. This is the only route to the **1 s** mixed-phase
  aspiration (exact solves cannot reach it), and it is mode-agnostic by
  construction. Ruled out of the destination because exact solves were chosen
  first; graduate if ticket 09 finds the exact route short of the 60 s budget,
  or when the 1 s figure becomes the requirement.
- **Finite T and `fixed_YC` as delivered paths.** Ticket 08 *measures* them to
  prove mode-agnosticism; it does not make them production-ready. What a
  finite-T CSC table costs, and whether the entropy rows change the Newton
  basin, is a later effort.
- **The BayEoS `njl` registry entry.** `bayeos/registry/models.toml` carries
  `dd2`, `sfho`, `alphabag` and no `njl`. The figure of merit there is the
  **per-theta table build time** (nothing about NJL survives a change of theta
  — vacuum solve, bag constant and RG counterterm all move with it), so it is
  downstream of this map's number and cannot be specified until 09 lands.
- **Whether `build_fast_table` survives.** If exact solves reach 1 ms/pt, the
  18 ms/pt spline path and its `FAST_MODES` restriction may be dead weight —
  or may still be the right thing at 8 nodes. Decide after 09.
- **C or Fortran via f2py.** The Q10 escape hatch, deliberately unexercised:
  numba is already a dependency and already the `backends/` pattern. Opened
  only if ticket 06 measured numba short of what the loop needs — and ticket
  03 has since measured the whole uncompiled remainder at 4.6% of a converged
  solve, so there is almost nothing on the other side of this door. Left
  standing rather than ruled out, because it costs nothing to leave a door
  shut.

## Out of scope

- **A physics-prior branch predictor** (predicting 2SC-vs-CFL from mu_q, m_s
  and Delta by the textbook criterion). Ruled out on measured grounds: NJL's
  running M_s(mu) and Delta(mu) are exactly what a fixed-parameter criterion
  cannot represent, which is why fitting NJL with ABPR is 9-15% out in eps.
- **A surrogate across theta** (NN/GP emulator trained offline over parameter
  space). A different project; it changes what `eos` is, and CLAUDE.md section 6
  lists surrogates as future work rather than current design.
- **The 1 s hybrid-table aspiration, as a committed target.** Exact solves
  cannot reach it — it needs the interpolated phase surface, which is fog. The
  committed mixed target is 60 s.
