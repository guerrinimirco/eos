# Land the bounded rescue ladder

Type: task
Status: closed
Blocked by: 13
Parent: ../map.md

## Question

Nothing to decide: [Bound the rescue ladder below a
branch](13-bound-the-rescue-ladder.md) decided it and measured it at **2.02x**
on the pinned benchmark, **2.70x** at T = 30 MeV and **4.66x** at `fixed_YC`,
T = 30, with the map's gate clean in all four. This ticket makes it real, so
that tickets 08, 09, 10 and 11 measure a solver that does not spend half its
time on candidates with no root.

### The edit, in two files

**`eos/general/solve.py`** -- `solve_system` gains one optional argument:

    def solve_system(residual, x0, scales_at, x0_fallback=None, tol=None,
                     jac=None, methods=('hybr', 'lm')):

the MINPACK rungs the caller is willing to pay for, in order. The default is
today's behaviour exactly. It is documented beside `LM_MAX_EVALUATIONS` and
makes the same argument in the same terms -- bound what a rescue may SPEND
rather than guess whether one is possible -- with ticket 13's measurement as
the evidence: `lm` was entered by exactly the 13 candidates with no root,
cost 29.2% of the build, and rescued none of them.

**`eos/njl/solver.py`** -- `solve_pattern` gains `cross_seeded=False`, and
`attempt` reads it in three places:

    x, err, ok = solve_system(rows, seed, unit_scales, tol=1.0e-13, jac=jac,
                              methods=('hybr',) if cross_seeded
                              else ('hybr', 'lm'))
    ...
    elif not ok and not cross_seeded:        # rung D
    ...
    if not ok and warm and rescue and not cross_seeded:   # the cold retry

and `solve` sets it where it already knows:

    cross = seed is None and reference is not None
    seed = seed_from(reference, par, spec, pattern) if cross else seed
    point = solve_pattern(..., x0=seed, cross_seeded=cross)

Rung C is **untouched**. It fires 0 times on the pinned benchmark, but
`attempt`'s docstring records it finding the CFL ground state at T = 20 MeV,
and ticket 13 did not measure that point.

### What the docstrings owe

`attempt`'s docstring currently describes a ladder every failing candidate
walks. It gains the distinction the change turns on, in the terms the
measurement made it: a candidate handed **another pattern's** state is being
asked whether a root of this pattern is near the state we already have, and a
no is an ordinary answer -- the pattern that owns the root it would collapse
onto is in the enumeration and finds it from its own seed. A candidate
warm-started from **its own** converged point one density down is the case the
cold retry was written for and keeps the whole ladder.

`solve`'s docstring gains one line: the seed's ORIGIN is now part of what it
hands `solve_pattern`, not just its value.

### Gate

- `test/njl`, `test/mixed`, `test/baseline` -- run **one directory at a time**
  (`test/njl` and `test/mixed` both hold a `test_jacobian.py` and collide at
  collection), on python.org 3.14.2 / numpy 2.3.5 / scipy 1.17.0. Four
  `test/baseline` entries fail on the anaconda 3.9 stack as 3.14 artifacts,
  pre-existing at HEAD -- check HEAD before chasing one.
- The pinned benchmark (`notebooks/quark_timing.py`, `54c7be9`): the ratio
  against a control taken in the same window, cpu printed beside wall.
- The map's gate re-run from `.scratch/njl-speed/proto13_gate.py`:
  `pattern_realised` at every density, P to 1e-8, the onset at
  n_B = 0.6583 and 170 CFL rows.

### What this does NOT reach

**`eos/mixed` is untouched by it, and the reason is worth writing down so
nobody expects the 2x there.** `njl_phase.thermo` does not call
`solve_pattern` at all -- it calls `eos.njl.thermodynamics.thermo_from_mu`,
which closes the model's internal system at fixed potentials through
`solve_system(..., tol=1e-13)` with **no Jacobian**: no Newton rung, no rung
C, no rung D, no cold retry. And its candidates are seeded from their OWN
previous root or from cold (`seeds.get(pattern)`), never cross-seeded, so
`cross_seeded` has nothing to key on there.

What the mixed path shares with what ticket 13 measured is only the `lm` rung,
paid by a pattern with no root at those potentials. Whether the `methods`
bound pays there is [Does the `methods` bound pay inside
`eos/mixed`?](17-methods-bound-in-mixed.md) -- a separate question with a
separate discriminator, and not something this ticket should claim.

---

## Resolution, 2026-09-21

**Landed in `50b3b7f`, with one departure from the edit as written, and the
departure is load-bearing: the bound applies only where there is an analytic
Jacobian.** As written, the edit also bound the reference backend, which is
`eos_table`'s default, and there it lost the CFL branch. On the fast path it
is exactly ticket 13's VX, bit for bit, and on the pinned benchmark it
measures **2.51x**.

### The departure, and the measurement that forced it

Ticket 13's gate, and this ticket's, ran `backend="fast"` only. But the
written edit binds both backends, and on the reference one `attempt` has no
Newton rung and no rung D, so a cross-seeded candidate is left with
`hybrd` and nothing else: no `lm`, no cold retry. Nobody had measured that
path.

`t16_reference_probe.py` covers the pinned grid's first 60 densities
(0.5 -> 0.817 fm^-3, same step), which span the 13 rootless CFL candidates
and the onset. It compared the reference backend, control `428cd66`, against
the edit as written:

- **30 of 30 CFL rows came back 2SC.** The 2SC -> CFL onset at 0.6583 was
  never found, and |dP|/P reached 0.85.
- The build got SLOWER (126.8 s against 111.1 s, cpu 121.0 against 105.5;
  the arms ran concurrently, so this is indicative only).

`t16_sweep_trace.py` recorded every candidate of that sweep, and it shows the
mechanism (`t16_trace_control.log`, `t16_trace_unbounded_reference.log`):

1. At **0.5686**, the CFL branch's first density, the control's cross-seeded
   CFL candidate converges in layout at f = 627.4562, reached by `lm` or the
   cold retry. Bounded, `hybrd` stalls at 1.03e-6 there and at the next six
   densities.
2. At **0.6055** the bounded candidate converges, but onto a **second CFL
   root**: f = 674.5474 against the control's 673.4439.
3. It then carries that root forward as its OWN seed, and from there it is
   warm-started, so it keeps the whole ladder and keeps tracking it. The
   branch it tracks sits 1 to 6 MeV/fm^3 above the true CFL branch, so it
   never undercuts 2SC.

This is the capture the map's continuation fog warns about, created by the
bound. A cold single point at the onset finds CFL in both arms, so the
failure exists only in a sweep.

**The fix is the precedent already in the same function.** `rescue=False` is
ignored when there is no analytic Jacobian, and `cross_seeded` now is too:
`bounded = cross_seeded and jac is not None`, and the three reads the ticket
specified take `bounded`. With it, the reference probe is **bit-identical to
the control on all 60 rows** (30 CFL rows), and the fast path is unchanged.
`attempt`'s and `solve_pattern`'s docstrings carry the measurement, and the
comment beside `LM_MAX_EVALUATIONS` says the 29.2% applies to a solve that
has a Newton rung first.

**The suite could not see it.** The four gate directories all PASSED on the
edit as written: njl 130, ccdm 60, mixed 272, baseline 20. Every csc test is
either a single point or a fast-backend sweep. There is now a regression
test, `test_a_cross_seeded_cfl_candidate_finds_the_branch_where_it_begins`
in `test/njl/test_pairing_patterns.py`, parametrized over both backends. It
calls `solve_pattern` on the cross-seeded CFL candidate at 0.5686. It was
RED on the edit as written (reference: `converged=False`, error 1.03e-6) and
is green on `50b3b7f`. `test/` is gitignored, so the test is local and not
in the commit.

### Gate -- PASS

- **The map's gate** (`proto13_gate.py V16 V0`, fast): 0 `pattern_realised`
  mismatches, worst |dP|/P **7.009e-10** at n_B = 0.563317, onset
  **0.6582914572865115**, identical to the last digit, **170** CFL rows, 0
  non-converged rows. The landed output is **bit-identical to
  `proto13_VX.json` on all 200 rows** (P, Delta, realised).
- **Tests on `50b3b7f`'s tree**, python.org 3.14.2 / numpy 2.3.5 / scipy
  1.17.0, one directory at a time, `eos/*.py` fingerprinted either side of
  each run and stable: `test/njl` **132**, `test/ccdm` **60**, `test/mixed`
  **272**, `test/baseline` **20**, all passed. These are the reachable
  suites:
  - njl is the change;
  - mixed reaches it through `njl_phase` (below);
  - ccdm and every other `solve_system` caller take the new argument's
    default, which builds the identical attempt list;
  - baseline pins every model at rtol = 1e-10.

  The first run, on the edit as written, is kept as
  `gate_tests_run1_unnarrowed.log` in the session scratchpad and is quoted
  above.
- **The pinned benchmark**, run from `54c7be9`'s block extracted verbatim
  (`git show 54c7be9:notebooks/quark_timing.py | sed ...`). Each arm is an
  isolated worktree, control `428cd66` against landed `50b3b7f`, one after
  the other, never concurrently. `t16_bench_tables.py` imports that
  `bench.py` and runs its table half with its own functions and print lines
  (`t16_bench_control.log`, `t16_bench_landed.log`). Median of 3, in ms/pt
  (cpu in brackets):

  | row | control `428cd66` | landed `50b3b7f` | ratio, wall / cpu |
  |---|---|---|---|
  | "default (4)" -- three since `2536d2b` | 437.6 (416.4) | 174.3 (172.7) | **2.51x / 2.41x** |
  | three | 418.9 (398.4) | 167.2 (166.5) | **2.51x / 2.39x** |
  | three, 0.30 -> 1.55, across chiral restoration | 591.9 (575.7) | 282.0 (281.2) | **2.10x / 2.05x** |
  | unpaired alone | 1.5 | 0.6 | a 0.3 s row, noise |
  | 2SC alone | 32.4 | 27.6 | 1.17x |
  | CFL alone (188 rows) | 442.7 | 460.1 | 0.96x |
  | `free` alone | 31.4 | 37.5 | 0.84x |

  **The window.** The machine was on AC and charged, and cpu tracked wall to
  within 5% in every row. Load was 3.1 to 6.2, all of it macOS daemons
  (BTLEServer at a steady ~1 core, WindowServer, fileproviderd); the only
  python was the arm being timed. The single-pattern rows **cannot** be
  reached by the change, since one pattern is never cross-seeded, so they
  are the window's own drift control: 0.84x to 1.17x on the 6 s rows and
  0.96x on the 90 s CFL row. The 2.5x is far outside that.

  **Against ticket 13's 2.02x.** That was measured at load 14-24 on battery,
  in a different window. Why the ratio is larger here is not measured. Both
  numbers are valid inside their own windows, and neither transfers to the
  other.

  **The mixed half of the block was not run**, for two reasons. The mixed
  point reaches this change only through `njl_phase`'s `seed`/`cold_start`,
  which is one njl `solve` per mixed solve, under 1 s of a 29.6 s point. And
  its window scan costs ~2600 s per arm. Whether the mixed engine gains is
  ticket 17's question. The "default (4)" row label is stale in the pinned
  block itself; it has timed the three-pattern default since `2536d2b`.

### What this ticket had wrong about `eos/mixed`

"`eos/mixed` is untouched by it" is true of `njl_phase.thermo`, the hot path,
and false of the adapter as a whole. `cold_start`, `seed` and `wing_sweep`
call `solve_beta_eq_neutrinoless` / `solve_fixed_yc` / ..., and those
enumerate and cross-seed. With `backend="fast"` the bound therefore reaches
the seed of every mixed solve and every pure-NJL wing. `njl_phase`'s own
default is `backend="reference"`, and there the bound is now off. `test/mixed`
passed either way.

### Handed to ticket 17

The Jacobian-free path is exactly the kind ticket 17 proposes to bound, and
the one measured here needed `lm`. Recorded in ticket 17 as a prior against,
not a verdict.

### Artifacts (in `.scratch/njl-speed/`)

- `proto13_V16.json`: the landed fast build.
- `t16_reference_probe.py`, `t16_reference_control.json`,
  `t16_reference_landed.json`. The last is the `50b3b7f` run; the
  as-written run's JSON was overwritten by it, and its evidence is the trace.
- `t16_sweep_trace.py`, `t16_trace_control.log`,
  `t16_trace_unbounded_reference.log`.
- `t16_bench_tables.py`, `t16_bench_control.log`, `t16_bench_landed.log`.
