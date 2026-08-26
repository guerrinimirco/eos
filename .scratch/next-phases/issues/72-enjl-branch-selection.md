# ENJL's fixed_YC_YS continuation picks its chiral branch by warm start, not by physics

Type: grilling
Status: open
Blocked by: -
Parent: ../map.md

## Question

Found by [ticket 62](62-regenerate-baselines-py314.md) while regenerating the
baselines: `enjl` is the one model of thirteen whose 3.9 -> 3.14 difference is
**not** round-off, and the mechanism is a branch selection nothing in the model
adjudicates.

`fixed_YC_YS` at Y_C = 0.5, Y_S = 0, `leptons=False`, over the warm-started
density sweep:

| n_B [fm^-3] | 3.9: M_q.u | 3.9: P | 3.14: M_q.u | 3.14: P | eps lower on |
|---|---|---|---|---|---|
| 0.2667 | 268.99 | 7.52 | 268.99 | 7.52 | (identical) |
| 0.3000 | 260.23 | 12.45 | 49.02 | -41.10 | broken (3.9) |
| 0.3333 | 251.56 | 19.06 | 62.23 | -29.47 | broken (3.9) |
| 0.3667 | 242.84 | 27.58 | 5.50 | -25.71 | broken (3.9) |
| 0.4000 | 233.98 | 38.23 | 5.50 | -13.11 | broken (3.9) |
| 0.4333 | 224.91 | 51.18 | 5.50 | 1.49 | **restored (3.14)** |
| 0.4667 | 215.56 | 66.56 | 5.50 | 18.34 | **restored (3.14)** |
| 0.5000 | 5.50 | 37.62 | 5.50 | 37.62 | (identical) |

Six contiguous points, 454 moved keys, `converged = 1` on both sides, same
n_B, same n_C, same targets. The two stacks land in different basins of the
same gap equations and the continuation carries the choice forward until the
branches rejoin at n_B = 0.5.

**Neither answer is right across the window.** At T = 0 and fixed n_B the
stable root is the one with lower eps, and that crosses inside the window: the
broken branch is lower to n_B = 0.400, the restored branch from 0.433. So the
first-order chiral crossing sits near n_B ~ 0.41 fm^-3 and **both** baselines
ride a metastable branch past it — 3.9 too far up on the broken side, 3.14 too
far down on the restored side.

### What has to be decided

1. **Does the raw `enjl` branch owe branch selection at all?** CLAUDE.md §8
   says a raw model branch MAY violate monotonicity inside a first-order
   region and that a construction resolves it before a table reaches TOV. That
   permits a metastable branch. It does not obviously permit the branch being
   chosen by which BLAS the warm start ran on.
2. **If it does: located how?** Comparing eps at each point costs one extra
   solve per point from the other basin, which the warm start already has the
   seed for. Whether that belongs in `solver.py`, in `table.py`'s continuation,
   or in a Maxwell/Gibbs construction alongside `eos/mixed` is the design half.
3. **What does `converged` mean here?** Both roots return `converged = 1`
   truthfully — each IS a root. Whether a returned point should also carry
   which branch it sits on, so a caller can see a discontinuity rather than
   infer it, is the reporting half.
4. **`enjl.npz` stays on 3.9 until this is settled**, so `test_baseline[enjl]`
   is red on the canonical stack and is expected to be. Re-freezing it on 3.14
   would record the metastable-restored answer as ground truth and delete the
   evidence. Whichever way 1-3 go, the regeneration follows this ticket.

### Related

The map's Not-yet-specified entry on a `general/verify/` differential check
notes that an undetermined potential shows as a shift in the ratio of `S_i` or
`C_i`. This case is the **negative** control for that screen and shows it
working: nothing here is proportional to any charge — masses, densities, P,
eps and mu_S all move by O(1) fractions — which is exactly how the screen was
supposed to separate a moved potential from moved physics.
