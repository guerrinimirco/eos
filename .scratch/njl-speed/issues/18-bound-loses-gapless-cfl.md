# The bounded ladder loses the gapless CFL ground state at `fixed_YC`, T = 0

Type: bug
Status: closed
Blocked by: none
Parent: ../map.md

## What was found, by ticket 10

[Ticket 10](10-the-quadrature-itself.md) went looking for the gapless states a
coarse pairing quadrature would lose first. The pinned beta-eq tables have
none, so it used the gapless ground state this model is documented to have:
`fixed_YC`, Y_C = 0.1, `leptons=True`, T = 0, `rg_njl1`, `csc=True`, the
default enumeration, `backend="fast"`, 45 densities over 0.5-1.6 fm^-3
(`t10_gapless.py`). There, the **shipped** quadrature is the one that loses
the ground state. The coarser rules don't.

| tree | backend | 24/24 realises at 0.775-1.075 | P across 1.075 -> 1.100 |
|---|---|---|---|
| HEAD `e2729bd` | fast | **2SC, 13 densities** | 539.1 -> 473.5 MeV/fm^3, **non-monotone** |
| `428cd66` (before `50b3b7f`) | fast | gapless CFL, all 13 | monotone |
| HEAD `e2729bd` | reference | gapless CFL, all 13 | monotone |

The gapless CFL state has the lower f (= eps at T = 0) at every one of those
densities, by 0.6 MeV/fm^3 at 0.775 up to 37.7 at 1.075, so HEAD's fast table
delivers a **metastable 2SC** there. That is the failure the map has already
recorded once ("tables reported metastable 2SC where gapless CFL was the ground
state"), and the table shows the same signature: P non-monotone where the
branch is finally picked up.

## The mechanism, measured (`t10_onset_probe.py`, `t10_rung_probe.py`)

At the first density past the 2SC region the CFL candidate has no seed of its
own, so `solve` cross-seeds it from the converged unpaired candidate, and
since `50b3b7f` a cross seed with an analytic Jacobian gets `hybr` only:

- **bounded** (`cross_seeded=True`): `hybr` with the analytic Jacobian stalls
  at 4.6-4.7e-2 at n_B = 0.8, 0.9 and 1.0, at 24 and at 12 nodes alike;
- **full ladder**: `hybr` and `lm` with the analytic Jacobian stall the same
  way, and **rung D, the differenced repeat, converges to 2-3e-13 on the
  gapless CFL root** (eps 955.092 at 0.8, the value every arm that kept it
  delivers).

Ticket 13's census found rung D converging its 13 rootless CFL candidates
onto duplicates of 2SC, and that is why the bound removed it. Here it is the
**only** rung that reaches a real ground state. That premise held on the
beta-eq benchmark and fails here. Ticket 13's fixed_YC check (4.66x, 20/20
rows) was at T = 30 MeV, where there is no T = 0 occupation step and no
gapless state to lose.

Two further observations:

- **Which state a sweep delivers here is decided by the seeding path, not by
  the physics, and on BOTH backends.** The CFL layout has at least four roots
  at this onset: the gapless CFL ground state, a collapse onto 2SC, a
  collapse realising uSC, and a second, metastable CFL root, 19.4 MeV/fm^3 up
  at 0.925. A perturbation that moves no block output by more than ~1e-11
  picks among them:

  | backend | arm (hot/vacuum nodes) | gapless CFL lost at |
  |---|---|---|
  | fast | 24/24 (shipped; bit-identical on rerun) | 13 densities, 0.775-1.075 |
  | fast | 12/12 | 1 (0.775) |
  | fast | 32/32 (FINER than shipped) | 7, 0.775-0.925 |
  | fast | 16/16, 10/10, 8/8, 24/12, 24/8, 24/32, 24/48, 48/48 | none |
  | reference | 24/24, 12/12, 10/10 | none |
  | reference | 24/12 | 7: 2SC x6, uSC at 0.95, the metastable CFL root at 0.925 |

  So a fix that only restores `50b3b7f`'s ladder on the fast path gets back
  the SHIPPED table (`428cd66` is right at 24/24), but not a robust one: the
  reference path, which was never bounded, loses the same states under a
  vacuum-only change. The bound turns an occasional loss into a systematic
  one: from the unpaired cross seed it reaches the gapless root at 0 of 3
  densities, and the full ladder does at 3 of 3.
- **The analytic Jacobian fails where the differenced one succeeds**, from the
  same seed. A gapless state's zero crossings move with the unknowns, and
  the analytic Hessian sees them only through `_crossing_terms`. Whether the
  analytic Jacobian is poor along the path into a gapless state is worth one
  measurement before choosing a fix.

## Question

Make the gapless ground state the one a sweep delivers, robustly, without
giving back ticket 16's 2.51x on the pinned benchmark. Restoring the ladder
on the fast path restores the shipped table and nothing more (see the lottery
table above). Candidates, not yet measured:

- keep rung D for a cross seed when the candidate's own pattern has a known
  gapless branch (no: that is a guess at the answer);
- keep rung D for a cross seed only at T = 0 (the occupation step is what
  makes the analytic path fail);
- carry a CFL seed across the 2SC region so the candidate is never cross-seeded
  there (ticket 05's capture warning applies);
- fix the analytic Jacobian along the gapless path, if that is what fails.

## Gate

- The table above delivers gapless CFL at all of 0.775-1.075, P monotone, on
  BOTH backends and under the perturbations that flip it today: arms 24/24,
  24/12 and 32/32 of `t10_gapless.py`, which re-points the vacuum rule and
  changes nothing else. A fix that passes at 24/24 alone has not shown it is
  one; `428cd66` passes at 24/24 and the reference backend fails at 24/12.
- The pinned benchmark's 2.51x kept, with ticket 16's both-backend row diff.
- A both-backend regression test at one of these densities.

---

## Resolution, 2026-09-22

**The analytic Jacobian is not what fails, and neither is the bound. The
seed is.** A gapped CFL state at T = 0 is an insulator for the CFL rotated
charge Q~, so at a nonzero fixed Y_C the residual is EXACTLY flat along the Q~
potential until a Q~-charged pair unlocks, and the gapless root lies a finite
distance along that flat direction. No derivative method started on the
plateau knows which way to go: the exact Jacobian drops the direction, and a
differenced one steps along it with the sign of its round-off. That is the
whole lottery, on both backends. The fix moves a CFL seed on the plateau to
where it first unlocks (`eos.njl.solver.unlocked_seed`), and every arm of the
gate now delivers the ground state.

### The one measurement: the analytic Jacobian is right

`t18_jac_path.py`, fixed_YC Y_C = 0.1 leptons T = 0, rg_njl1, the cross
seed at n_B = 0.8, 0.9, 1.0 (24 nodes):

- **Part A.** At all 10-11 points where the bounded path (Newton + hybrj)
  asks for a Jacobian, the analytic one matches a 1e-4 central difference to
  1.5e-8 .. 9.4e-7 row-relative. The path **never enters a gapless state**:
  gaps stay 157-223 MeV, zero crossings at every point.
- **Part C.** On the straight line seed -> gapless root the states stay gapped
  until the root; the match is 3e-8 .. 9e-7 along it and 6e-6 AT the root,
  where the 1e-4 and 1e-6 differences disagree with each other by the same
  6e-6 -- the difference's resolution at the crossing, not the Jacobian's
  error (verify's own gapless-state figure is 5e-6).
- **Part B / the control (`t18_substitution.py`).** The bounded path with the
  analytic Jacobian REPLACED, same seed, reaching the gapless root:

  | Jacobian | 0.8 | 0.9 | 1.0 |
  |---|---|---|---|
  | analytic | no | no | no |
  | central 1e-3 / 1e-4 / 1e-5 / 1e-6 / 1e-7 | Y Y n Y n | n Y n Y Y | n n n n Y |
  | analytic x (1 + 1e-6 N) / (1 + 1e-3 N) | Y Y | Y Y | n n |

  A 1e-6 jitter of an accurate Jacobian flips the outcome, and the step size
  alternates yes/no. **Candidate 4 is refuted**: `_crossing_terms` is not the
  lever.

### The mechanism (`t18_first_step.py`, `t18_anatomy.py`, `t18_edge.py`)

- At the cross seed the scaled Jacobian's smallest singular value is
  **1e-17** of the largest, at 0.8, 0.9 and 1.0 alike, along
  (mu_3, mu_8, mu_C) proportional to (-2, -1, 2): d mu_C = t, d mu_3 = -t,
  d mu_8 = -t/2 shifts mode (f, a) by t (q_f - q_a), the rotated charge
  (u: 0,+1,+1; d, s: -1,0,0 over r,g,b), under which every CFL pair is
  neutral. The charge row at fixed_YC is quark-only (the leptons neutralise
  separately and do not feed back), so nothing gives the direction a slope.
  In beta equilibrium the electrons' dn/dmu_C does, which is why the pinned
  benchmark never meets this.
- With the exact Jacobian `lstsq` (rcond 1e-12) drops the direction and
  Newton stalls in the gapped valley (4.6-4.7e-2, colour_3 and charge rows);
  with any noisy one the direction gets sigma ~ 1e-7, the step is 1e8-1e9,
  the 25% clip leaves it PURELY along Q~ (d mu_C = +-439 MeV), and its random
  sign decides whether the state unlocks. Rung D "reaching the root 3 of 3"
  in the ticket, and the reference backend losing 7 densities at 24/12, are
  that coin.
- The bounded stall is the root's **mirror image**: Delta_1 < Delta_2 and
  mu_3 = +44 against the root's Delta_1 > Delta_2 and mu_3 = -20 at 0.8.
- Along +Q~ from the seed the scaled rows are identical to every printed
  digit on the plateau, then turn gapless at an edge that moves out with
  density: 60-80 MeV at 0.775-0.8, 80-100 at 0.9, 100-120 at 0.95, 120-140
  at 1.0-1.075; a second pair unlocks near 200-240.

**Candidate 2 is refuted with it**: its premise (the T = 0 occupation step
breaks the analytic path) does not hold -- the path is never gapless before
the root -- and the pinned benchmark is itself at T = 0, so rung D at T = 0
gives the 2.51x back. **Candidate 1** was a guess at the answer, and rung D is
itself a coin. What was built is **candidate 3's intent reached through the
physics**: never hand the CFL candidate a seed it cannot leave.

### Seeds that did not work, and the one that does

Scored over five draws of the bounded fast path (the analytic Jacobian and
four copies jittered at 1e-6), so a basin boundary shows as a split rather
than one lucky ticket (`t18_seeds.py`, `t18_qtilde.py`):

- the current cross seed: 0-2 of 5 per density, at 24 and 12 nodes;
- mu_C from the unpaired state, M_u = M_d, gaps nudged off the
  Delta_1 = Delta_2 plane to either side: 0-5 of 5 with no pattern across
  densities, and at 0.775 and 0.8 the two opposite nudges give draw-for-draw
  identical outcomes -- the seed's place in the gap plane is not what
  decides;
- a FIXED displacement t along Q~: 5/5 at some densities and failing at
  others on both backends (t = +100 is still on the plateau at 1.0, where the
  reference ladder misses; t = +300 collapses to 2SC at 0.8);
- **the seed moved to just past its own unlocking edge**, found by bisection
  on the state's `gapless` flag (`t18_edge_seed.py`): 5/5 fast and 3/3
  reference (seed jittered at 1e-9) at 0.8, 0.9, 1.0, 1.075. The edges come
  out identical on both backends (68.9 / 95.4 / 121.7 / 139.3 MeV), because
  they are a property of the state, not of the solver.

### The change

`eos/njl/solver.py`: `ROTATED_CHARGE`, `UNLOCKING_BISECTIONS = 8` and
`unlocked_seed`, called at the top of `solve_pattern`'s `attempt`, so every
seed handed to the ladder passes through it on both backends -- the warm or
cross seed and the cold retry. It acts only for pattern CFL, T = 0, a mode
that fixes a nonzero Y_C, and a seed state that is not already gapless; the
direction's sign is sign(Y_C) (the Q~ = +1 modes are u quarks). Otherwise it
returns its argument untouched. Cost where it acts: at most ten state
evaluations. The module docstring's seeding facts gain the third.

### Gate -- PASS

- **The gapless table** (`t10_gapless.py` as the ticket states it, 45
  densities over 0.5-1.6, the default enumeration, run on a frozen copy of
  the fixed tree; checked by `t18_gate_check.py`, which reproduces ticket
  10's lottery table exactly when fed ticket 10's runs):

  | backend | arm | HEAD lost | fixed lost | P from 0.775 up | realised vs control | worst dP/P |
  |---|---|---|---|---|---|---|
  | fast | 24/24 | 13 | **0** | monotone | 0 of 45 | 4.8e-10 |
  | fast | 24/12 | 0 | **0** | monotone | 0 of 45 | 6.4e-10 |
  | fast | 32/32 | 7 | **0** | monotone | 0 of 45 | 6.7e-10 |
  | reference | 24/24 | 0 | **0** | monotone | 0 of 45 | 6.6e-11 |
  | reference | 24/12 | 7 | **0** | monotone | 0 of 45 | 8.8e-11 |
  | reference | 32/32 | -- | **0** | monotone | 0 of 45 | 6.5e-11 |

  The control is HEAD's reference 24/24 table from ticket 10, which kept the
  state. Every arm: `{'2SC': 11, 'CFL': 34}`, gapless CFL at every density
  from 0.775.
- **The regression test**, `test_the_enumeration_finds_the_gapless_cfl_ground_state`
  in `test/njl/test_pairing_patterns.py`, beside ticket 16's, parametrized
  over both backends, at n_B = 1.0: the public `solve` (default enumeration)
  must deliver gapless CFL at f = 1267.804, and so must the cross-seeded CFL
  candidate from the cross seed moved 60 MeV along Q~ (still on the plateau,
  whose edge is 121.7 MeV there). **RED on HEAD on both backends** -- fast
  fails the `solve` check (2SC delivered), reference fails the plateau-seed
  check (not converged) -- and green on the fix (8.6 s), each arm run from
  inside its own tree copy with `eos.__file__` printed in the process that
  ran pytest. No single density is red on both backends through `solve`
  alone: HEAD's reference `solve` misses at 1.025 only of the 13, and HEAD's
  fast one wins its coin there. `test/` is gitignored, so the test is local.
- **The pinned benchmark**: **unchanged, and the rows are bit-identical to HEAD on both
  backends.** Run from `54c7be9`'s block extracted verbatim (`t18_bench.py`
  imports that `bench.py` and times with its own `bench_table`), one isolated
  tree per arm, the arms one after another and never concurrently; python.org
  3.14.2 / numpy 2.3.5 / scipy 1.17.0, on AC power, but CONTENDED: a foreign
  worker pool held loadavg at 13-30 throughout and cpu/wall ran 0.77-0.94.
  Median of 3, ms/pt, cpu in brackets:

  | row | `428cd66` | HEAD `e2729bd` | fixed | HEAD -> fixed, wall / cpu |
  |---|---|---|---|---|
  | default (4) -- three since `2536d2b` | 972.9 (753.2) | 449.4 (364.9) | 414.6 (362.3) | 1.08x / 1.01x |
  | three | 839.0 (732.3) | 462.1 (364.9) | 415.0 (360.9) | 1.11x / 1.01x |
  | three, 0.30 -> 1.55 | 1151.8 (1083.2) | 659.8 (580.2) | 651.9 (580.2) | 1.01x / 1.00x |

  The fixed tree's cpu is HEAD's to 1%, which is what it must be:
  `unlocked_seed` returns at its first test in beta equilibrium. **The
  absolute 2.51x is NOT re-measured by this window** -- `428cd66` -> HEAD
  itself reads 2.06x / 2.01x / 1.87x cpu here against ticket 16's 2.41x /
  2.39x / 2.05x on a quiet machine, the control arm having run under the
  heavier load -- so what this window shows is that the fix gives none of
  ticket 16's gain back, not a fresh figure for it. Row diffs
  (`t18_rowdiff.py`): the fast default table, 200 rows, is **bit-identical to
  HEAD** (P, Delta, realised) and against `428cd66` has 0 realised
  mismatches and worst |dP|/P 7.0e-10, ticket 16's own figure, 170 CFL rows;
  the reference backend over the pinned grid's first 60 densities
  (`t16_reference_probe.py`, the 13 rootless CFL candidates and the onset)
  is **bit-identical to HEAD and to ticket 16's landed run**, 30 CFL rows.
- **Suites**: python.org 3.14.2 / numpy 2.3.5 / scipy 1.17.0, on the working
  tree (HEAD `e2729bd` plus this change), one target at a time, every
  `eos/*.py` fingerprinted either side of each run and stable
  (`34a7609c...`): `test/njl` **134 passed** (ticket 16's 132 and the two
  new cases), `test/mixed` **272 passed** (50 min at loadavg ~25),
  `test/baseline` **20 passed**, `test/test_nonconvergence_return.py` +
  `test/test_imports.py` **233 passed**. These are the reachable suites:
  njl is the change; `eos/mixed` reaches it through `njl_phase`'s wing
  points (`solve_fixed_yc`); baseline pins njl and mixed at rtol = 1e-10 (its
  njl paired points are beta-eq and its fixed-Y_C sweeps `csc=False`, so no
  frozen key can reach `unlocked_seed`); the top-level pair imports
  `eos.njl.api` and checks the layering. No other package imports
  `eos.njl`. `eos/njl/njl.tex` and `njl.md` gain the third seeding fact
  (the tex compiles). Not committed.

### What this does not reach

- **T > 0** is left alone: the plateau is exactly flat only where the
  occupation is a step. At T > 0 the thermal tail gives Q~ a nonzero
  susceptibility with the physical sign; whether the analytic path follows
  it at small T is not measured.
- **Y_C < 0** takes the other sign by the physics stated above; not measured.
- **fixed_YC_YS** passes the same condition (it fixes C); the rotated charge
  is flat in its strangeness row too, but the mode is not measured.
- **`eos/mixed` reaches it**: `njl_phase`'s wing points call
  `solve_fixed_yc`, so a mixed build in a fixed-Y_C mode at T = 0 gets the
  moved seed for its CFL candidate -- the right physics for a pure-phase
  point, and inside the suites run above.
- A one-shot `eos_point` at fixed Y_C, T = 0 was a coin on both backends
  before this, including before `50b3b7f`; it is now deterministic at every
  density measured, which is the robustness the ticket asked for rather than
  the shipped table back.
