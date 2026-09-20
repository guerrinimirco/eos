# Land the bounded rescue ladder

Type: task
Status: open
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
