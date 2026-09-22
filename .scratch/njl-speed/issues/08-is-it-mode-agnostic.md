# Is the acceleration mode-agnostic, or has it quietly become beta-eq-only?

Type: task
Status: closed
Blocked by: --
Parent: ../map.md

## Why the blockers are cleared, and why the question changed

This ticket was written to re-run tickets 05 and 06's speedups in other
modes. That premise is out of date: [05](05-cheap-pre-screen.md) refuted its
own lever (there is no loose pre-screen to measure) and
[06](06-compile-the-newton-loop.md) never ran (03 priced the compiled loop at
1.05x, and 20 re-opens it as a question). Neither is a precondition any more.
What has to be mode-agnostic is what actually LANDED on the njl table path
since the map's baseline `d6d9e7c`:

| commit | what | where it acts |
|---|---|---|
| `2536d2b` | `free` left `DEFAULT_PATTERNS` (default unpaired/2SC/CFL) | every mode |
| `50b3b7f` | bounded rescue ladder for a cross-seeded candidate | `backend="fast"` only (needs the analytic Jacobian) |
| `a948b33` | `unlocked_seed` | CFL, T = 0, fixed Y_C != 0 only |
| `a9a4aa1` | RG vacuum pairing passes on a 12-node rule | every T |

Evidence that already exists and is reused, not redone:
[ticket 13](13-bound-the-rescue-ladder.md) measured the bounded ladder ALONE
on `np.linspace(0.6, 1.5, 20)`, `backend="fast"`, 0 state mismatches in all
four sweeps (beta-eq T=0 1.23x, T=30 2.70x; fixed_YC 0.4 leptons T=0 1.24x,
T=30 4.66x; `proto13_modes.py`); [ticket 19](19-land-the-quadrature-rule.md)
measured the vacuum rule at T = 20/30 points (P within 4e-9), and a vacuum
pass is taken at mu* = 0, T = 0, so it cannot depend on mode or T.

The original failure this ticket guards against is unchanged:
`build_fast_table` looked ordinary at `fixed_YC` and was 3-5% out in P,
because its spline needs dP/dmu_B = n_B, which only beta equilibrium gives.

## Question

Is the WHOLE landed acceleration mode-agnostic? Still unmeasured:

- (a) the whole landed path per mode, not one lever;
- (b) `fixed_YC` with `leptons=False`, what `eos/mixed` needs per pure phase;
- (c) `beta_eq_neutrino_trapped`;
- (d) the accuracy risk of dropping `free`: at fixed Y_C an asymmetric state
  (uSC/dSC) could win, and the old default's `free` could have found it.
  Ticket 12 checked this at beta-eq only.

### Design

Two arms, both `backend="fast"`, `rg_njl1`, `SpeciesFlags(csc=True)`, on
ticket 13's grid `np.linspace(0.6, 1.5, 20)`:

- **base** -- `d6d9e7c`, whose default enumeration still has `free`, run
  from an isolated worktree outside the repo;
- **landed** -- HEAD, default enumeration.

Cases: beta_eq_neutrinoless T = 0 / 30; fixed_YC Y_C = 0.4 leptons=True
T = 0 / 30; the same with leptons=False; beta_eq_neutrino_trapped Y_Le = 0.4
T = 30. cpu (`process_time`) beside wall, median of n = 3, the arms
interleaved case by case in each repeat so they share one load window; one
untimed warm-up solve per process (numba compiles separately per tree);
loadavg with every run.

Accuracy per row: same `pattern_realised` and |dP|/P <= 1e-8. Every mismatch
is solved on HEAD with each single pattern alone and cold (unpaired, 2SC,
CFL, uSC, dSC, free), and the arm holding the lower f is named. A base-arm
loss (ticket 18's gapless-CFL lottery predates `d6d9e7c`'s fix) is not a
failure of the landed path; a landed-arm loss is.

### What a failure here looks like, and what it means

- **Speed holds, accuracy holds** -- the route is mode-agnostic, and the fog
  entry on finite-T and `fixed_YC` production paths stays a scheduling matter.
- **Speed shrinks in a mode** -- fixable; a new ticket names the fix.
- **Accuracy fails in a mode** -- the serious case; it goes to
  `docs/DEFERRED.md` (CLAUDE.md section 3) and ticket 09 says so plainly
  rather than shipping a second `FAST_MODES`.

## Gate

- The table, filled in: case | base cpu | landed cpu | ratio | rows solved
  (base/landed) | realised mismatches | worst |dP|/P | load.
- Every mismatch adjudicated by the single-pattern cold solves.
- An explicit verdict sentence: mode-agnostic, or not, and if not, which mode
  and whether speed or accuracy is what fails.
- A mode whose accuracy fails is recorded in `docs/DEFERRED.md`.

## Resolution, 2026-09-23

**The landed acceleration is mode-agnostic: it holds in all seven cases
(3.19x to 26.05x in cpu, never below the ~3x anchor), and so does accuracy
(140 of 140 rows solved in both arms, 0 realised mismatches, worst |dP|/P
8.9e-10).**

Stack: python 3.14.2 (python.org, `/Library/Frameworks/.../3.14/bin/python3`),
numpy 2.3.5, scipy 1.17.0. Arms: base `d6d9e7c` from an isolated worktree
(`DEFAULT_PATTERNS` with `free`), landed `071d712` (without). Every worker
printed its stack and asserted `eos.__file__` lies in its own tree. Both
`eos/*.py` fingerprints were unchanged across both runs (landed `3c3b4cc2`,
base `c3f0d353`), so no run is void. Timing is cpu, the median of n = 3. Each
repeat interleaves the arms case by case, and the per-repeat ratios are in
brackets. Repeat 0 is the budget run (`t08_rep0.*`, 745 s), and repeats 1-2
are `t08_reps12.*` (1588 s). The table is `t08_table.log`, built by
`t08_analyse.py`.

| case | base cpu | landed cpu | ratio | rows solved (base/landed) | realised mismatches | worst \|dP\|/P | load |
|---|---|---|---|---|---|---|---|
| 1 beta-eq T=0 | 95.5 s | 3.7 s | **26.05x** (27.89, 26.18, 26.05) | 20/20 | 0 | 8.9e-10 | 3-16 |
| 2 beta-eq T=30 | 37.3 s | 6.7 s | **5.55x** (6.09, 5.97, 4.39) | 20/20 | 0 | 5.2e-10 | 4-6 |
| 3 fixed_YC 0.4 leptons T=0 | 42.6 s | 12.6 s | **3.38x** (3.03, 3.13, 3.38) | 20/20 | 0 | 1.6e-10 | 4-6 |
| 4 fixed_YC 0.4 leptons T=30 | 179.5 s | 26.3 s | **6.83x** (6.86, 7.11, 6.53) | 20/20 | 0 | 2.1e-10 | 3-6 |
| 5 fixed_YC 0.4 no leptons T=0 | 41.4 s | 13.0 s | **3.19x** (3.16, 3.33, 3.19) | 20/20 | 0 | 1.8e-10 | 3-4 |
| 6 fixed_YC 0.4 no leptons T=30 | 188.1 s | 26.7 s | **7.04x** (7.14, 6.08, 7.00) | 20/20 | 0 | 2.3e-10 | 3-7 |
| 7 trapped Y_Le=0.4 T=30 | 88.2 s | 11.5 s | **7.68x** (7.22, 8.12, 7.68) | 20/20 | 0 | 4.5e-10 | 3-7 |

Per point (20 densities), landed runs at 185 / 335 / 630 / 1315 / 650 /
1335 / 575 ms/pt in case order.

**Machine load.** The window was 01:17-01:57 on 2026-09-23, on AC power.
Loadavg was 2.9-7 throughout, except the first reading of the first timed
run (15.8, from Spotlight and BTLE daemons, not an eos session), which had
decayed to 6.3 by the run's end. cpu/wall was 0.91-1.00 on every run. The
0.91 is one landed run of case 2, which is also that case's low per-repeat
ratio of 4.39.

**Adjudications: none were needed.** No row failed the gate, and no row was
solved by one arm and dropped by the other, so `t08_adjudicate.py` (the
single-pattern cold procedure) was not run. `t08_mismatches.json` is `[]`.

**(d), the risk of dropping `free`.** Base's `free` won the f-tie (the
reported `pattern` column) at 29 of 140 rows (5/1/6/4/6/4/3 by case). At
every one of them it realised the state the landed arm delivers, so it was
a duplicate, as ticket 05 found at beta-eq. No row in either arm, in any
case, is uSC/dSC or gapless. At Y_C = 0.4 the matter is 2SC at every density
0.6-1.5, at T = 0 and 30, with and without leptons. So `free` found no winner
the landed default lacks, anywhere on this grid. A NAMED uSC/dSC winning
where neither default proposes one is a different question, and it is
[ticket 14](14-usc-dsc-in-the-default.md)'s.

**Why the ratio spreads 3x-26x.** This is not a lever shrinking. Ticket 13
measured the bounded ladder ALONE at 1.23x (beta-eq T = 0) and 1.24x
(fixed_YC T = 0), the same in both modes, and the vacuum rule is mode-free by
construction. The spread is consistent with what `free` cost the base arm in
each mode: base beta-eq T = 0 is 4.8 s/pt, and base fixed_YC T = 0 is 2.1.
This was not decomposed further. The landed path itself is 3.4x dearer per
point at fixed_YC than at beta-eq, at T = 0 (630-650 against 185 ms/pt).
That is a cost the fog entry "finite T and `fixed_YC` as delivered paths"
inherits, not a failure of the acceleration.

**One base-arm oddity, for the record.** `d6d9e7c`'s rows differ between its
two worker processes in cases 2, 4 and 7 (T = 30), and are bit-identical
within each process. Where they differ: the `pattern` label flips on the
`free`/CFL/2SC tie, P moves <= 2.8e-10, and gaps sitting at numerical zero
move. `pattern_realised` never changes, and the gate holds against either
base process (worst 8.9e-10). Neither tree draws random numbers, and the
cause was not chased. It is a property of the base, which has `free`. The
landed arm is bit-identical across all six of its runs in every case.

No mode fails, so there is no `docs/DEFERRED.md` entry and no new ticket.
