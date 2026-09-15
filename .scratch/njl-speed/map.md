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
| `eos_table`, default enumeration (4 patterns) | 8100 ms/pt | 8100x |
| `eos_table`, restricted to 3 patterns | 637 ms/pt | 637x |
| `eos_table`, one pattern: unpaired / 2SC / CFL / free | 1.4 / 56.5 / 719 / 68.5 ms/pt | — |
| of that CFL table (ticket 03): 12 dead points / 188 converged | 89% / 80 ms/pt | 80x on the rows |
| one `eos.mixed` point, DID+NJL, 3 patterns / held 2SC | 29.6 s / 9.2 s | — |
| the mixed window: 52 of 200 densities inside it | 52 x 29.6 s = 1539 s | 26x |
| `eos.mixed` window location, 20 densities | 2576 s | dominates |

and at ticket 04, the same configuration instrumented rather than timed
(counts deterministic to the call; wall +-30%, cpu quoted beside it):

| what | count | wall |
|---|---|---|
| one hybrid row | 40 residuals / 41 `thermo` / **123 NJL solves** | 12.8 s onset, 28.8 s deep |
| `locate_window`, HINTED (the cheap form) | **5,700 NJL solves** | 3,385 s |
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
  collapses. Following the winner along a sweep is still **not** adopted and
  is still in the fog below.
- **A timing number from this map states the machine's load.** Ticket 05
  measured the same three-pattern configuration at 730 ms/pt against a
  concurrent session and at 267 on a quiet machine — 2.7x apart, cpu tracking
  wall in both. Ratios are taken inside one window; absolutes are not
  comparable across them.

## Decisions so far

<!-- one line per closed ticket -->

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
  free-energy margin over the runner-up narrowing). This is where the
  fewer-patterns lever actually pays in full — but it can silently seat a
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
