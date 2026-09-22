# Land the pairing quadrature rule that ticket 10 measured

Type: task
Status: closed
Blocked by: 09
Parent: ../map.md

## Why this is held

[Ticket 10](10-the-quadrature-itself.md) found the pairing block's 24 nodes
per panel are more than the gate needs. Landing a coarser rule moves
`test/baseline/njl.npz`, a golden reference (CLAUDE.md §12). A golden
baseline is changed on purpose, after the ruling, not along the way. So this
ticket waits for [ticket 09](09-verdict-and-port.md). Part 1 of 09 has
retired the 1 ms/pt target, so what this ticket buys is ~1.3-1.5x on an
exact-solve table, which matters for the BayEoS per-theta cost.

## What is measured, from ticket 10

- **The vacuum passes at 12 are safe at every T.** They are evaluated at
  T = 0 by construction, so their rule does not depend on T (§4). The
  table-verified result is 24/12 on both backends at T = 0, plus points at
  T = 20 and 30. It is worth **1.22x** on a converged CFL solve and 1.23x on
  the 2SC table (§5).
- **The hot pass at 16 is the T-agnostic candidate.** It is **1.37x** on a
  converged CFL solve and 1.28-1.29x on the single-pattern tables (§5). At
  T = 20-30 MeV it holds P to <= 1.9e-9 but s only to 2.6e-8 (§4). s is not
  in the gate (P to 1e-8), and it must be stated anyway.
- **The hot pass at 12 is 1.67x, but only at T = 0.** It fails the P gate at
  T = 20-30 (1.3e-7 .. 1.3e-6, §4). It can land only T-aware: inside
  `rg_pair_block`, or per call through the existing `pair_nodes_per_panel`.
- **16/12 was never run as an arm.** The two T-agnostic halves have not been
  measured together.

## Question

Which rule lands: 16 hot with 12 vacuum at every T, or a T-aware 12/12? Are
the ~1.3x and ~1.5x on the pinned tables still there when that rule is
measured as its own arm?

### Constraints, already paid for

- **`NODES_PER_PANEL` stays.** It is `eos/general/fermi_integrals.py`'s,
  every model's unpaired integrals use it, and ccdm builds its Gauss rule
  from it at import. Change only the rule the pairing block uses. ccdm's
  pairing (no RG split, finite T) was not measured by 10, so a change inside
  `eos/general/pairing.py`'s defaults reaches ccdm and owes ccdm a
  measurement. A change confined to `eos/njl` does not.
- **The residual and its Jacobian stay on one vacuum rule.** 10's harness
  enforced this by re-pointing the module globals both of them read, and the
  landed code must enforce it by construction.
- **The baseline moves.** With vacuum-only 12, 9 of njl's 139 keys land
  outside 1e-10 (§7):
  - six are pinned only to solver resolution: mu_3 ~= 0 in 2SC, and CFL's
    mu_C and -mu_3 taken separately;
  - three are physical, at 1.1-2.9e-10.

  With 12/12, 32 keys move, the T = 20 point's s, Y_C and n_e at ~1e-6.
  Regenerate **njl only**, in its own commit, with the deltas quoted
  (`test_baseline.py`'s rule 2). Whether the six solver-resolution keys stay
  frozen, or leave the way `row()` already drops mu_S, is a change to what the
  baseline pins. It is the user's call, not this ticket's.
- **The gapless table is part of the gate.** Before [ticket 18](18-bound-loses-gapless-cfl.md),
  root selection at `fixed_YC`, T = 0 was a lottery that a quadrature rule
  entered. 18's seed fixed it under 24/24, 24/12 and 32/32, but not under the
  rule landed here.
- njl.tex and njl.md state the quadrature (§11), so the node rule they state
  changes with it.

## Gate

- The pinned single-pattern tables, 2SC and CFL, on both backends: same
  pattern and P to 1e-8 against 24/24, worst point reported.
- Ticket 10's finite-T points (§4) at the landed rule: P to 1e-8, with s and
  M reported beside it.
- `t10_gapless.py` at the landed rule on both backends: gapless CFL at every
  density from 0.775, P monotone.
- The pinned benchmark, n = 3, interleaved against 24/24 in one window.
- `njl.npz` regenerated in its own commit, with the deltas quoted. The
  reachable suites (`test/njl`, `test/mixed`, `test/baseline`,
  `test/test_imports.py`), plus ccdm's if the change reaches it.

## Resolution, 2026-09-22

**Landed: the in-medium pass keeps 24, the two RG vacuum passes drop to 12, at
every T.** The rule is `eos/njl/thermodynamics.py`'s
`VACUUM_NODES_PER_PANEL = 12`, which `backends/jacobian.rg_pair_jacobian`
imports, so the residual and its Jacobian are on one vacuum rule by
construction rather than by a harness. `pair_nodes_per_panel` still means what
it meant -- the in-medium pass -- so an explicit 24 is still the default bit
for bit, and the caller who wants ticket 10's 12/12 passes 12.

**1.207x on the pinned benchmark** (cpu, n = 3 interleaved against HEAD's rule
in one window: 1.171, 1.209, 1.207), i.e. **~254 -> ~210 ms/pt** on the map's
scale. `pair_nodes_per_panel=12` is **1.562x** against HEAD and **1.31x** on
top of the landed default, in the same window.

Stack for everything below: python.org 3.14.2 / numpy 2.3.5 / scipy 1.17.0,
tree at `211dbed` plus this change, AC power. The machine was NOT quiet --
foreign 3.14 workers held loadavg 19-65 throughout, cpu/wall 0.71-0.91 -- so
only ratios inside the one interleaved window are quoted and the absolute
ms/pt are not comparable to tickets 02/03/16.

### Why this rule and not the other two

The question was three-way: 16 hot with 12 vacuum at every T, a T-aware 12/12,
or vacuum-only 12.

- **Vacuum-12 is the only cut that is T-independent BY CONSTRUCTION.** The
  vacuum passes sit at mu* = 0, T = 0 whatever the state's temperature, so
  they have no Fermi surface and no thermal collar and their rule cannot
  depend on T. Every other arm trades accuracy at some T for speed; this one
  is measured at the quadrature floor at every T tried (section "Gate").
- **Not 16 hot.** It holds P (<= 1.9e-9 at T = 20-30) but leaves **s at
  2.6e-8**, and s is a delivered quantity -- the simulation tables of
  CLAUDE.md section 6 are what the finite-T path is for. It also moves the
  T = 20 baseline point. What it buys over the landed rule was never measured
  as its own arm; 16/16 is 1.28-1.29x against 24/24 where the landed rule is
  1.21x, so the trade is ~0.1x for three decades of s.
- **Not a T-aware 12 inside `rg_pair_block`.** It is the fastest arm at T = 0
  and it is the caller's to ask for, not the library's to assume, for a
  measured reason: **12 in the medium fails at T = 1 MeV too, and the failure
  is not monotone in T** (`t19_smallT.log`, `Parameters.default()`, reference
  backend, 12 against 24 on the landed vacuum):

  | 2SC, n_B = 1.4 | T = 1 | 2 | 5 | 10 | 20 |
  |---|---|---|---|---|---|
  | dP/P | **1.9e-8** | 1.1e-10 | 7.9e-11 | 1.8e-9 | **1.3e-7** |
  | ds/s | 1.5e-6 | 5.9e-8 | 4.3e-10 | 2.6e-8 | 9.4e-7 |

  T = 1 MeV is already past the 1e-8 P gate, and s is out at every T tried.
  At the CFL point the two rules realise DIFFERENT layouts at T = 1 and 2
  (dSC, dsSC) -- but the 24-node arm moves between layouts there as well, so
  that point is knife-edge and is not charged to the coarse rule either way.
  A default keyed on `T == 0.0` would therefore hand a T = 0.5 MeV table a
  rule that no measurement supports, to buy 1.31x. The caller who IS at T = 0
  knows it and says so in the call.

### What a caller who wants another rule does

`pair_nodes_per_panel` is an argument all the way down from `eos_point`,
`eos_table` and `TableSpec`, and it sets the in-medium pass:

- **T = 0 and wants ticket 10's 12/12:** `pair_nodes_per_panel=12`. 1.31x on
  top of the landed default. Gated here on the pinned default table (0
  realised mismatches, worst |dP|/P 1.75e-9 at n_B = 0.7005, repeats
  bit-identical) and on the gapless `fixed_YC` table (below).
- **wants 16:** `pair_nodes_per_panel=16` gives 16/12.
- **wants HEAD's 24/24:** there is no argument for it, and no measured
  quantity distinguishes it -- vacuum 12 is at the floor of every delivered
  number (section 1 of ticket 10, and the gate below). The pre-change numbers
  are `a948b33`.

### Gate

All of it against a CONTROL ARM IN THE SAME TREE: both vacuum lookups
re-pointed to 24, which is HEAD's rule. Checked, not assumed -- the control's
200 default-table rows are **bit-identical to `a948b33`'s** (`t18_bench_fix.json`:
n_B, P, realised pattern and all three gaps).

| table, pinned grid, 200 densities | rows | realised mismatches | worst \|dP\|/P, where |
|---|---|---|---|
| default enumeration, fast | 200/200 | 0 | 1.75e-9 at 0.68995 (CFL) |
| 2SC held, fast | 200/200 | 0 | 9.00e-10 at 0.7796 (2SC) |
| CFL held, fast | 188/188 | 0 | 3.93e-9 at 0.5633 (CFL onset) |
| default enumeration, reference | 200/200 | 0 | 1.85e-10 at 0.500 (2SC) |
| 2SC held, reference | 200/200 | 0 | 7.16e-11 at 0.5791 (2SC) |
| CFL held, reference | 188/188 | 0 | 3.52e-9 at 1.4339 (CFL) |

The two worst points are the solve, not the rule, and both were checked rather
than assumed:

- 0.5633 is the CFL branch onset, where ticket 10 measured 2.9-4.6e-9 under
  EVERY arm including one that moves no block output past 1e-11.
- 1.4339 is deep CFL on the reference backend, and there a POINT solve under
  the landed rule agrees with fast-landed and with the reference control to
  <= 7.2e-11 from every seed, the control's own root included
  (`t19_ref_deep_cfl.log`). Only the warm-started reference sweep's row sits
  3.5e-9 out: M_u 27.383182701 against 27.383182775. It is where that sweep
  stops, and it is inside the gate.

**The gapless densities, by name** (`fixed_YC`, Y_C = 0.1, leptons, T = 0, the
DEFAULT enumeration, 45 densities over 0.5-1.6, both backends):

| arm | gapless CFL, 0.775 up | P falls | realised mismatches vs control | worst \|dP\|/P |
|---|---|---|---|---|
| landed, fast | 34 of 34 | none | 0 | 7.30e-10 |
| landed, reference | 34 of 34 | none | 0 | 6.76e-11 |
| `pair_nodes_per_panel=12`, fast | 34 of 34 | none | 0 | 4.26e-10 |

18's `unlocked_seed` is what makes this dull: ticket 10 lost 7 of these
densities on reference 24/12 and 1 (0.775) at 12/12, as a root-selection
lottery. Neither loss reproduces.

**Ticket 10 section 4's finite-T points at the landed rule**, against the
control, both backends -- P, s and M:

| point | fast: dP/P, ds/s, dM/M | reference: dP/P, ds/s, dM/M |
|---|---|---|
| 2SC, n_B = 1.4, T = 20 | 1.1e-10, 2.5e-14, 2.0e-11 | 1.9e-13, 3.4e-15, 7.9e-12 |
| CFL, n_B = 1.2, T = 30 | 1.2e-9, 2.3e-13, 3.5e-12 | 3.6e-11, 8.3e-14, 3.1e-12 |
| CFL, n_B = 1.2, T = 20 | 2.6e-10, 1.5e-12, 3.6e-12 | 4.3e-11, 1.4e-13, 6.1e-12 |

These reproduce ticket 10's monkeypatched 24/12 numbers digit for digit, which
is the check that the landed code does what the harness did.

**The benchmark**, bench.py as extracted from `54c7be9`, default enumeration,
fast backend, 200 densities, arms interleaved repeat by repeat:

| arm | cpu ms/pt (median of 3) | x vs control | per repeat |
|---|---|---|---|
| control = HEAD's 24/24 | 361.9 | 1.000 | — |
| **landed 24/12** | 301.2 | **1.207** | 1.171, 1.209, 1.207 |
| `pair_nodes_per_panel=12` | 229.9 | 1.562 | 1.539, 1.584, 1.562 |

Each arm's three repeats are bit-identical in every row. Wall agrees (1.202x,
1.538x medians) but one landed repeat ran at cpu/wall 0.71 and is why the cpu
column is the quoted one.

### The baseline, regenerated on purpose

`test/baseline/njl.npz` rebuilt with `generate_baseline.py njl` on the stack
above, after the rule was chosen. **3713 of 3790 keys are bit-identical** --
every unpaired sweep, every mode, the vacuum block -- and all 77 that move are
at the four paired points and the enumeration point. Nine are outside
rtol = atol = 1e-10, exactly the nine ticket 10 section 7 predicted:

**Six pinned only to solver resolution**, both of them directions the residual
is flat along at T = 0, so nothing in the equations places them:

| key | stored -> now | rel |
|---|---|---|
| `pattern.2SC.n1.4887.T0.mu_3` (and `.x` through it) | -3.565e-9 -> -7.528e-9 MeV | 1.1 |
| `pattern.CFL.n1.2.T0.mu_C`, `.mu_3`, `.mu_8` (and `.x`) | mu_C 0.5105040 -> 0.5104076 MeV | 1.9e-4 (mu_8: 7.2e-7) |

The CFL three move ALONG TICKET 18's ROTATED CHARGE, exactly:
d mu_C = -9.64e-5 MeV, d mu_3 = +9.64e-5 = -d mu_C, d mu_8 = +4.82e-5 =
-d mu_C/2. That is the Q~ plateau, which a gapped CFL state at T = 0 is an
insulator for; 2SC's mu_3 is the same statement for T_3, whose pairs are all
neutral. They would fail under any change to the Newton path.

**Three physical**, all at 2SC, n_B = 1.4887, T = 0, and all the vacuum
Lambda_UV pass's own truncation where M_u = 9.88 MeV sits in the lowest
geometric panel -- ticket 10 section 1's layout effect:

| key | stored -> now | rel |
|---|---|---|
| `M` (M_u) | 9.875640887 -> 9.875640884 | 2.9e-10 |
| `mu_C`, `mu_e` | -94.16574638 (+-) | 1.1e-10 |

The 68 that moved inside the tolerance are led by P at 9.6e-11 (CFL n1.2),
7.0e-11 (2SC n1.4887) and 1.9e-13 (2SC n1.4 T20), and the enumeration's f at
9.2e-12. The T = 20 point barely moves, which is the vacuum rule's
T-independence showing up in the baseline.

**It is NOT a commit, and could not be**: `test/` is gitignored (CLAUDE.md
section 11), so the regeneration is a working-tree change and this entry plus
`t19_baseline_diff.log` are its record. The pre-change file's sha1 is
`4968b43494d65e7e1399f27c61e5abc686bac129`, kept in this session's scratchpad
as `njl_before_t19.npz`. No other model's `.npz` was touched. Whether the six
solver-resolution keys should stay frozen at all, or leave the way `row()`
already drops mu_S, is still the user's call and is not made here.

### Suites

python.org 3.14.2 / numpy 2.3.5 / scipy 1.17.0, one directory at a time, on
this tree, with `eos.__file__` printed by the process that ran pytest and
`eos/*.py` fingerprinted either side of the whole sequence (unchanged):

    test/njl                              135 passed   (134 at a948b33; +1 is
                                                       the new vacuum-rule test)
    test/baseline                          20 passed
    test/test_imports.py                  221 passed
    test/test_nonconvergence_return.py     12 passed
    test/general/test_zero_pressure.py      9 passed
    test/mixed                            272 passed

**ccdm is not reached.** The change is inside `eos/njl`; `eos/general/pairing.py`'s
`NODES_PER_PANEL` and every unpaired integral are untouched, so ccdm's pairing
keeps the rule it had and owes no measurement.

The new test is `test/njl/test_pairing_patterns.py::test_vacuum_passes_keep_their_own_rule_in_residual_and_jacobian`:
it records the rule each vacuum lookup is handed during one solve with
`pair_nodes_per_panel=16` and asserts both halves saw
`VACUUM_NODES_PER_PANEL`. Shown red on purpose by putting the Jacobian's
vacuum on 24 while the residual's stayed at 12 -- the failure it exists for,
which `verify/`'s central-difference parity check cannot resolve (the two
rules differ by ~1e-11, the parity gate is 1e-5).

### Handed on

- **Ticket 20's shares are the ones to re-read at this rule**, not at 12/12:
  ticket 10 section 6's 24/12 column has `gapless_momenta` at 12.3% of a
  converged CFL solve, 28.2% of the CFL table and 15.9% of the 2SC table. The
  attribution was not re-taken here.
- njl.tex and njl.md state the rule (section 7.3, the new paragraph in 8.2,
  and the API table), and `eos_point`/`TableSpec` say what lowering the
  in-medium pass costs at T = 0 and at T > 0.
