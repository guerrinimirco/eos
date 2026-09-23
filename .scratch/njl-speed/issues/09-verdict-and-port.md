# The verdict: did exact solves reach 1 ms/pt and 60 s, and what ports?

Type: grilling
Status: closed
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

Part 2 below does the rest. The two graduating maps are named there, not
charted.

---

## Part 2, 2026-09-23: the port, the rulings, and the map closed

Part 1's rulings stand and are not revisited: both targets retired, the fog
ruled, `build_fast_table` survives. Part 2 owed four things: the port
specification, a ruling on every ticket still open, the landing measurement,
and the map marked closed.

### The port specification

**The port is a fast-forward, and it is held.** Every lever landed in
production form on `njl-speed`, with its documentation and local tests:
`2536d2b` (the three-pattern default), `50b3b7f` (the bounded ladder),
`66d713c` (the locator and `lm` cuts), `a948b33` (`unlocked_seed`) and
`a9a4aa1` (the vacuum rule). No code is left to port. `njl-speed` is 13
commits ahead of `main` and 0 behind, counting this one; the merge-base is
`d6d9e7c`.

- **Mechanics.** Take a CLEAN landing certificate on the SHA that main will
  move to. Then run `git fetch . njl-speed:main`. That form refuses anything
  but a fast-forward and needs no checkout, so the other sessions sharing this
  tree are not disturbed.
- **Held by ruling 7 below.** One blocker: the 188 MB blob described at the
  end of this list.

**What the map changed under `eos/`, by file:**

| file | commit | what | ticket |
|---|---|---|---|
| `eos/general/pairing.py` | `2536d2b` | `free` leaves `DEFAULT_PATTERNS` | [12], [15] |
| `eos/general/solve.py` | `50b3b7f` | `solve_system(methods=)`; default unchanged | [13], [16] |
| `eos/njl/solver.py` | `50b3b7f`, `a948b33` | `cross_seeded` bounds the ladder where there is a Jacobian; `unlocked_seed` | [16], [18] |
| `eos/njl/thermodynamics.py` | `66d713c`, `a9a4aa1` | `thermo_from_mu(methods=)`; `VACUUM_NODES_PER_PANEL = 12` | [17], [19] |
| `eos/njl/backends/jacobian.py` | `a9a4aa1` | reads the same vacuum constant | [19] |
| `eos/njl/api.py`, `table.py` | `50b3b7f`, `a9a4aa1` | docstrings; `pair_nodes_per_panel` means the in-medium pass | [15], [16], [19] |
| `eos/mixed/adapters.py` | `50b3b7f`, `66d713c` | enumerating `njl_phase` declines `lm` | [17] |
| `eos/mixed/boundaries.py` | `66d713c` | a failing downward walk step takes no retry ladder | [11] |
| `eos/ccdm/api.py`, `table.py` | `50b3b7f` | docstrings only | [15] |

The documents are `njl.tex`/`njl.md`, `ccdm.tex`/`ccdm.md` and
`docs/DEFERRED.md`. This commit adds `njl.tex`/`njl.md`, `mixed.tex`/`mixed.md`
and `DEFERRED.md` (the audit below).

**What the fast-forward also carries, which is not this map's.**
[Ticket 01][01] left the J0614 group "to its owning session", and `2536d2b`
then committed it on this branch:

- **`plot/data/samples/J0614_Miller.txt`, 188,310,363 bytes. This is the
  blocker.**
  - By the repository's own `.gitignore` convention, raw posterior samples are
    not tracked. They are re-fetched by `plot/fetch_samples.py`, which already
    carries this file's sha256. The file's `.gitignore` line was never added.
  - GitHub rejects any blob over 100 MB. Once the blob is in `main`'s history,
    `main` cannot be pushed, and a later `git rm --cached` does not help.
  - There are two ways out, and both are the user's call:
    - rewrite `njl-speed` from `2536d2b` without the blob before the
      fast-forward. Every SHA from `2536d2b` on changes, and this map cites
      them throughout, so the rewrite owes a SHA table in `map.md`;
    - fast-forward as it is, and never push `main` until both branches are
      rewritten.

  Either way, the `.gitignore` line belongs beside `J0614.dat`. It is not
  added here.
- `eos/dd2/nmp.py`, about 1,000 lines changed across `2536d2b` and `428cd66`
  (the latter is titled "improving speed njl"), and
  `eos/dd2/verify/run_full_check.py`. This is a DD2 inverse-map change that no
  ticket on this map reviewed. Only the landing certificate below covers it.
- The J0614 constraint data under `eos/general/constraints/`, `plot/`,
  `docs/csc_bag_mapping.md`, notebooks and `.scratch/pqm`.

**Decisions that change nothing in code:**

- **`backends/` stays deletable (CLAUDE.md §5). Measured, not argued.** An
  isolated worktree at `80e26d1` had `eos/njl/backends` removed. Run from
  inside it with `PYTHONPATH` on it, `eos.__file__` pointed at the worktree,
  and `NUMBA_OK` was False and `residual_jacobian` None. `import eos.njl`,
  `eos.mixed.adapters` and `eos.njl.verify.run_full_check` all load. The
  reference table (20 points over n_B = 0.6-1.5, `beta_eq_neutrinoless`,
  T = 0, `rg_njl1`, `csc=True`) came back **bit-identical to the main tree:
  20 rows, all 30 fields, 0 differences**, at 17.9 against 17.6 cpu s. It did
  not fail, so it was not repeated at `d6d9e7c`. The worktree is removed.
- `build_fast_table` and `FAST_MODES` stay (part 1).
- `analytic_jac` stays default-False at its four call sites ([07]).

### The audit: documents, `verify/`, tests

Each of the five landed behaviours was checked against CLAUDE.md §11 (it is
written out in `njl.tex` and `njl.md`) and §12 (a `verify/` entry where it is
a physics invariant, and a test). Tests are gitignored, and these were checked
in the local tree.

| behaviour | `.tex` / `.md` | `verify/` | test |
|---|---|---|---|
| three-pattern default | stated in both; **`njl.tex`'s RG paragraph still said "the free seed is in the enumeration"** (fixed here) | none owed: a seed list is not an invariant | **nothing pins `DEFAULT_PATTERNS`**; only `test_baseline`'s `enumeration.n1.2` covers it, indirectly |
| bounded ladder, fast backend only | **absent from `njl.tex`; `njl.md` §13.1 still described the unbounded rescue** (fixed here: `njl.tex` enumeration section, `njl.md` §12.4 and §13.1) | **owed**: backend parity is checked state by state (`state_at`), so a divergence of the solve PATH is invisible to it, and that is the hazard [16] met | `test_a_cross_seeded_cfl_candidate_finds_the_branch_where_it_begins`, both backends; **no unit test of `solve_system(methods=)`** |
| `unlocked_seed` | stated in both | **owed**: "a CFL root at T = 0 and Y_C != 0 is gapless" is a physics statement, since gapped CFL carries n_C = 0; `verify/` only constructs a gapless state, for Jacobian parity | `test_the_enumeration_finds_the_gapless_cfl_ground_state`, both backends |
| `VACUUM_NODES_PER_PANEL` | stated in both (the rule, the RG cost section, the API table) | none owed: Jacobian parity cannot resolve it ([19]), so a test pins it | `test_vacuum_passes_keep_their_own_rule_in_residual_and_jacobian` |
| locator step-down and adapter `lm` decline | **absent from both adapter sections and from `mixed.tex`/`mixed.md`** (fixed here) | none owed: the cut is bit-identical | **none**: [17] argued the held-pattern carve-out rather than testing it |

Two more things turned up:

- `mixed.tex`/`.md` called `MAX_WALK` a bound on "the bisection that backs"
  the scan. It bounds the WALK, and this commit's paragraph now describes the
  walk.
- `test_table_restriction_matches_the_point_solve`'s docstring still calls the
  three patterns "the documented fast one", which "drops only the asymmetric
  free seed". Those three patterns are now the default. The docstring is not
  edited here.

The fixed documents compile clean under `pdflatex` (`njl.tex`, `mixed.tex`).

### The rulings

1. **[06] closed, absorbed by [20].** Compiling the loop is 1.05x
   ([03](03-profile-one-cfl-solve.md)). The one surviving block, and 06's
   route and gate, live in 20.
2. **[07] closed, not pursued. `analytic_jac` stays default-False.** The
   ceiling is 1.1-1.3x ([04](04-count-the-mixed-loop.md)). `did_phase` has
   no block either. The 60 s target is retired.
3. **[14] closed into `docs/DEFERRED.md`**, entry "njl, ccdm: where the
   asymmetric pairing sector wins is not known". [08] found no uSC/dSC state on
   140 rows. The entry was stale on ccdm. It is corrected here with [15]'s
   measured win (uSC over 2SC by 0.20 MeV/fm^3 at T = 50, n_B = 1.3) and
   with the named `uSC` seed collapsing at T = 30.
4. **[20] handed on** to the BayEoS njl registry map. At the landed 24/12
   rule its ceiling is <= 1.39x on the CFL table and <= 1.19x on 2SC ([19]'s
   "Handed on").
5. **`test/baseline/njl.npz`: the six solver-resolution keys are dropped**,
   with the reason written in `generate_baseline.py`
   (`_flat_paired_potentials`). The rule is keyed on a criterion, not on key
   names: T = 0, not gapless, and realised 2SC (mu_3) or CFL (mu_C, mu_3,
   mu_8 along Q~), plus `.x`, which carries them.
   - Regenerated on python.org 3.14.2 / numpy 2.3.5 / scipy 1.17.0:
     **3790 -> 3784 keys, exactly the six gone, and all 3784 kept
     bit-identical** to [19]'s file (sha1 `0571bd07` -> `3924dbd1`).
     `test_baseline[njl]` passes.
   - Not a commit: `test/` is gitignored. The previous file is kept in this
     session's scratchpad.
6. **Audit gaps.** The document gaps are fixed in this commit: `njl.tex`,
   `njl.md`, `mixed.tex` and `mixed.md`. The `verify/` and test gaps are
   recorded below as owed, because they need `eos/*.py` or test edits, which
   are out of this session's scope.
7. **Merge: held.** `main` does not move. The blob above is the one blocker.
   **Push: no.**
8. From the landing, below: `test/njl/test_jacobian.py` renamed to
   `test/njl/test_analytic_jacobian.py`, content unchanged.

### Owed by the port, not done here

- `verify/`:
  - a solve-level backend-parity entry: same realised pattern and P to 1e-8
    over a warm sweep on both backends, which is where the bound can diverge;
  - an entry for "the CFL root at T = 0, Y_C != 0 is gapless".
- tests:
  - one pinning `DEFAULT_PATTERNS` (and that `free` is requestable and not
    enumerated);
  - one for `solve_system(methods=('hybr',))` declining `lm`;
  - one for 66d713c's two cuts;
  - the stale docstring above;
  - `bisect`'s bracketed-but-unlocated `nan`, which [11] recorded as untested
    on a real solve.
- `.gitignore`: `plot/data/samples/J0614_Miller.txt`.
- `test/run_clean_suite.sh` **fails open on a suite that cannot run** (below).
  Its verdict judges only the fingerprints, so a collection error is
  certified CLEAN. It needs a verdict that also requires a pytest summary
  line with a passed count.

### Loose ends, recorded and NOT acted on

- **ccdm's enumeration misses the CFL ground state at 3 of 12 points**
  ([15] item 2). T = 30, n_B = 1.3 returns uSC at +4.27 MeV/fm^3; T = 30,
  n_B = 1.6 returns usSC at +22.56; T = 50, n_B = 1.6 returns sSC at +6.04.
  A cross-seeded CFL candidate collapses, keeps its name and competes, because
  `eos/ccdm/solver.py` has no layout filter, no re-seed and no
  `realised_pattern`. It is a wrong ground state in shipped code, on the
  default call, at finite T. **Deferred by the user, still unfiled.**
- **The mixed branch's existence boundary is undecided** ([11]). The branch
  does not start at chi = 0: it ends at chi ~= 0.69 near n_B ~= 0.8516. So
  the "onset" is an existence boundary, and whether that is physics or a
  second branch was not decided.
- **The locator's probes below the onset** ([11]) are 31% of the unhinted
  locator's NJL work and wait on a design decision. Beside them sit two
  failed exact refines (23%) and the honest walk (27%, a third of it
  re-solves).
- **Continuation along the sweep stays in fog, with its price** ([13], [05]):
  - the price: rootless candidates are still 54.7% of the build after the
    bound, so the ceiling is <= 2.2x on njl and nothing on the mixed rows;
  - the constraint: the CFL onset is reached only because CFL is still
    proposed after 13 consecutive failures, so a monitor that stops proposing
    a failing pattern loses 170 of 200 rows;
  - capture was measured at 200 of 200, and a cold re-hunt every ten densities
    cost 6x of the 12.5x it protected.

  Nothing graduates until a monitor is stated and its miss is priced.

### Graduating maps: named, not charted

- **The interpolated phase surface** (a new map, in `eos`). Build
  P(mu_B, mu_C, mu_S) once per theta, T and pattern, and root-find every mode
  and the mixed Gibbs solve against that one interpolant: the only route to
  the 1 s hybrid aspiration. It starts from [08]'s warning (the beta-eq
  spline was 3-5% out in P at `fixed_YC`) and from the kink in the winner's P
  at a pattern switch.
- **The BayEoS `njl` registry entry** (a new map, charted in `bayeos`, which
  consumes `eos`). Add `njl` to `bayeos/registry/models.toml` at a per-theta
  cost of ~29-42 s for a 200-point beta-eq T = 0 CSC table (part 1's 35-51 s
  divided by [19]'s 1.207x). Its first question is whether a sampler can
  afford a table build per theta; if not, it consumes the phase-surface map.
  It inherits [20] and the continuation fog.

### The landing measurement

Two certificates, both from this work, and both cited (CLAUDE.md §12). The
stack for both is python.org CPython 3.14.2 / numpy 2.3.5 / scipy 1.17.0, on
AC power, at HEAD `80e26d1`, with no uncommitted `eos/*.py` and the `eos/`
fingerprint `cae94720` on both sides of each run.

- **`test/suite_certificates/20260923T120316.txt`: verdict CLEAN, and it is
  NOT a measurement.** pytest stopped at collection after 1 s with `1 error`,
  the `import file mismatch` between `test/njl/test_jacobian.py` and
  `test/mixed/test_jacobian.py` that [ticket 01][01] recorded. No test ran.
  The certifier judged only the fingerprints, so it certified a suite that
  could not run. That is a fail-open defect of `test/run_clean_suite.sh`,
  recorded above as owed. The file is committed with the others, because a
  certificate that exists is cited.
  - Why the full suite had not run since: `test/njl/test_jacobian.py` was
    created on 2026-09-04, and the last full run (`20260903T015007.txt`, a
    DISCARD) predates it.
  - `--import-mode=importlib` is not a way out: it breaks five modules that
    import sibling helpers by bare name (the dd2 and did TOV tests, three
    enjl tests).
  - With the user's go-ahead, the newer file was renamed to
    `test/njl/test_analytic_jacobian.py`, content unchanged, which makes
    `test_jacobian.py` the only such basename again. That is a local change:
    `test/` is gitignored.
- **`test/suite_certificates/20260923T120508.txt`: CLEAN, 1958 passed, 23
  skipped, 0 failed** (1981 collected), 12:05:08-12:30:02, 1492.9 s. This is
  the landing measurement for `80e26d1`. It ran with this ticket's regenerated
  `njl.npz` and the renamed test in place.

This commit changes no `eos/*.py`: only `.tex`/`.md` documents, ticket files
and certificates. So the measurement still describes the `eos/` tree at the
commit this ticket lands in. `main` was NOT moved (ruling 7).

[01]: 01-land-the-fast-table-work.md
[02]: 02-pin-the-benchmark.md
[03]: 03-profile-one-cfl-solve.md
[04]: 04-count-the-mixed-loop.md
[05]: 05-cheap-pre-screen.md
[06]: 06-compile-the-newton-loop.md
[07]: 07-njl-jacobian-block.md
[08]: 08-is-it-mode-agnostic.md
[10]: 10-the-quadrature-itself.md
[11]: 11-cheapen-the-locator.md
[12]: 12-free-in-the-default.md
[13]: 13-bound-the-rescue-ladder.md
[14]: 14-usc-dsc-in-the-default.md
[15]: 15-land-the-pattern-default.md
[16]: 16-land-the-bounded-ladder.md
[17]: 17-methods-bound-in-mixed.md
[18]: 18-bound-loses-gapless-cfl.md
[19]: 19-land-the-quadrature-rule.md
[20]: 20-relook-gapless-momenta.md
