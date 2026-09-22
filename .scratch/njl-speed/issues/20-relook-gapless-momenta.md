# Re-look at 06: `gapless_momenta` is the wall once the quadrature is cut

Type: prototype
Status: open
Blocked by: 09, 19
Parent: ../map.md

## Why 06 is re-opened as a question

[Ticket 06](06-compile-the-newton-loop.md) was scoped from ticket 03's
profile, at the shipped 24-node rule. There `gapless_momenta` was 10.5% of a
converged CFL solve and 21.9% of the CFL table, and 06 was told it was worth
at most 1.3x and not to grow.

[Ticket 10](10-the-quadrature-itself.md) §6 measured that the block's share
grows as the quadrature is cut. The block does not scale with the node rule:
it scans 48 momenta, refines, and runs Brent on every call, whatever the node
count.

| `gapless_momenta`, share of | 24/24 | 12/12 | 8/8 |
|---|---|---|---|
| a converged CFL solve | 10.3% | **17.0%** | 21.8% |
| the CFL table | 22.8% | **35.2%** | 42.8% |
| the 2SC table | 12.1% | 16.7% | 20.0% |

If the block became free at 12/12, the ceiling would be **1.20x** on a
converged solve, **1.54x** on the CFL table and 1.20x on the 2SC table. It is
the largest single block left on the table, and in the pinned grid it scans
for crossings in states that have none. There is no gapless state anywhere on
the grid (10 §3). The dead points below the CFL branch pay most: 3.06 ms per
call there, against 1.09 ms at a converged state (03).

These shares hold at 12/12, so this ticket's value depends on
[ticket 19](19-land-the-quadrature-rule.md) landing a coarser rule. At the
shipped rule, 06's original ceiling still stands.

## Question

Which is the lever: 06 as written, or a cheaper way to rule out a crossing?

1. **06 as written.** A jitted, allocation-free crossing search in the
   backend (numba first). `backends/` stays deletable (CLAUDE.md §5, §9), and
   the reference path keeps the NumPy scan.
2. **A cheaper no-crossing verdict.** Most calls end with no crossing found.
   A sufficient condition for "no branch of this block reaches zero on
   [k_lo, k_hi]", if one is cheap, returns before the scan. This changes the
   algorithm and not only its speed, so it goes in the REFERENCE path,
   `eos/general/pairing.py`, and both backends inherit it. A wrong
   no-crossing verdict loses a gapless state. That is exactly the failure
   [ticket 18](18-bound-loses-gapless-cfl.md) just fixed, and its table is
   the gate.

Measure both before building either. The first question is how much of the
block's cost falls in calls that find no crossing.

### Constraints

- `gapless_momenta` lives in `eos/general/pairing.py`. ccdm reaches it
  through `pair_block`, so a change there reaches ccdm and owes ccdm a
  measurement.
- Its docstring records two hazards already paid for: counting eigenvalues
  is robust where the smallest |lambda| is not, and a breakpoint that appears
  and disappears with a threshold makes the residual discontinuous. A
  shortcut must not reintroduce either.
- Breakpoints split the quadrature. They are not physical quantities, so the
  gate is that the solve converges and the numbers hold, not that the
  breakpoints match (06's own gate).

## Gate

- Ticket 10's attribution at the landed rule, before and after, with the
  share `gapless_momenta` gave up stated, not inferred from the total.
- The pinned single-pattern tables on both backends: same pattern and P to
  1e-8.
- `t10_gapless.py` on both backends: gapless CFL at every density from
  0.775, P monotone.
- If route 1 is taken: `backends/` shown to be deletable, and a parity check
  in `eos/njl/verify/`.
- **Measured against the ceilings above.** Falling short of them is the
  answer and settles the f2py fog entry. It is not a failure.
