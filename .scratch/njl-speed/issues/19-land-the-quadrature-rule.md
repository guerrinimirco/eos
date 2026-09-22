# Land the pairing quadrature rule that ticket 10 measured

Type: task
Status: open
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
