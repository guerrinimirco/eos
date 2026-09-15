# Compile `gapless_momenta`, the only uncompiled block that costs anything

Type: prototype
Status: open
Blocked by: 03
Parent: ../map.md

## Question

**Re-scoped by [Where do the 443 ms of a CFL solve actually go?](03-profile-one-cfl-solve.md),
which refuted this ticket's premise.** It read: "the residual assembly, the
pairing spectrum, `gapless_momenta`, `pair_hessian`, the charge rows and
`newton_solve`'s step control and line search are all Python, executed ~30
times per solve". Measured, on a converged warm-started CFL solve:
`pair_hessian`'s bulk is jitted and is 38% of it, the quadrature is jitted and
is 39%, and **all the Python together is 4.6%** on 4.3 residual evaluations
and 2.3 Jacobians — not ~30 of anything.

So the ceiling is measured and it is small: **compiling the loop itself buys
1.05x. Compiling every uncompiled numpy block as well, and assuming they
became free, buys 1.3x.**

What survives is ONE block, and the ticket is now only about it:
**`gapless_momenta` is 10.5% of a converged CFL solve and 21.9% of the whole
CFL table** — 3.06 ms per call averaged over the table against 1.09 ms at a
converged state, because the dead low-density points have far more zero
crossings to hunt. It is a `np.linalg.eigvalsh` scan plus `brentq`, called
1.5-2 times per residual, and it is the only uncompiled thing in the profile
above 5%.

**Does a jitted crossing search remove that 10-22%, and at what accuracy?**
The three numpy blocks beside it — `_unpaired_reference` (2.0%),
`_unpaired_reference_hessian` (4.3%), `pair_nodes` (1.2%) — come along only
if they are cheap to take.

**This ticket is worth at most 1.3x and cannot be the map's lever.** Take it
for what it is, and do not let it grow: the levers are ticket 05 (the rescue
ladder, 89% of this table) and ticket 10 (the quadrature, 77% of a good
solve).

### Scope, bounded by CLAUDE.md

- It goes in `eos/njl/backends/` and `eos/general/pairing.py`'s existing jitted
  section. **`backends/` stays deletable** (section 5): with it gone the
  reference NumPy path must still produce the same numbers, more slowly.
  Section 9's reference/fast split is not negotiable and this ticket does not
  get to renegotiate it.
- numba first. C or Fortran via f2py is the escape hatch and opens **only** if
  numba measures short — that is a fog entry, not this ticket's licence.
- The parity check goes in `eos/njl/verify/` alongside `check_jacobian_parity`.

### The hard parts, named in advance

- **`gapless_momenta` is a root hunt inside the residual.** It reports each
  crossing twice (once per +- block) and needs deduping; that was a 2x
  overcount once already. A jitted version needs a bounded, allocation-free
  crossing search.
- **The RG panel rule moves the quadrature nodes with the state**
  (`max_panel_ratio = 2`). Fixed-shape arrays are what numba wants and moving
  node counts are what the physics wants; padding to a fixed maximum is the
  likely answer and it must be shown not to change numbers.
- **The step control and the line search stay in Python.** They were 0.6% of
  a converged solve; compiling them would drag `lstsq` at a singular Jacobian
  and both of `solve_pattern`'s rescues into numba for nothing. The two hard
  parts that used to live here — lstsq's minimum-norm step at a symmetric CFL
  root, and Newton's basin differing from hybrd's — are therefore NOT this
  ticket's problem any more, and they are the reason not to make them one.

## Gate

- ms/pt on the pinned benchmark, single-pattern CFL and 2SC, before and after,
  **and the same attribution ticket 03 took**, so the share `gapless_momenta`
  actually gave up is stated rather than inferred from the total.
- Same pattern and P to 1e-8 across the benchmark grid, worst point reported.
  A crossing search is a BREAKPOINT search, so the gate is that the solve
  still converges and the numbers hold — not that the breakpoints match.
- `backends/` demonstrably still deletable: the reference path run with the
  compiled module removed, numbers compared.
- A parity check landed in `eos/njl/verify/`.
- **Measured against 10.5% of a converged solve and 21.9% of the table.**
  Falling short of those is the answer, not a failure; what it opens is a
  decision about the f2py fog entry, and on these numbers f2py is very
  unlikely to be worth it.
