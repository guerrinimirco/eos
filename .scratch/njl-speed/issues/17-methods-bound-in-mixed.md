# Does the `methods` bound pay inside `eos/mixed`?

Type: prototype
Status: open
Blocked by: 16
Parent: ../map.md

## Question

[Bound the rescue ladder below a branch](13-bound-the-rescue-ladder.md)
measured `lm` at 29.2% of a 200-point njl build, entered by exactly the 13
candidates with no root and rescuing none of them, and
[16](16-land-the-bounded-ladder.md) gives `solve_system` a `methods` argument
to decline it. **Does the same rung cost the same nothing inside the mixed
loop, and can the same argument decline it there?**

### Why it is not the same question

`njl_phase.thermo` is a different code path and ticket 13 does not reach it:

- it calls `eos.njl.thermodynamics.thermo_from_mu`, not `solve_pattern`, so
  `attempt`'s ladder -- the Newton rung, rung C, rung D, the cold retry -- is
  not in it at all. What it runs is `solve_system(..., tol=1e-13)` with **no
  Jacobian**: `hybr`, then `lm`, then the polish;
- its candidates are seeded from their OWN previous root or from cold, never
  cross-seeded, so the discriminator ticket 13 found -- *where did this seed
  come from* -- does not exist here. A pattern with no root at those
  potentials looks exactly like one that simply has not been reached yet.

So the finding transfers but the rule does not, and the rule is the part that
has to be re-derived.

### What to measure first

The counts, not the clock -- [ticket 04](04-count-the-mixed-loop.md)'s
instrument already prices one hybrid row at ~41 `thermo` calls and ~123 NJL
internal solves. Of those solves, how many enter `lm`, what does it cost, and
how many does it rescue? If the answer matches ticket 13's (entered only by
rootless candidates, rescues none), the bound is `methods=('hybr',)` on the
adapter's call and the question becomes whether anything downstream of a
declined `lm` changes.

**Trap ticket 04 already left here:** the mixed cost is **flat in count and 2x
in per-call cost across the window**, expensive at the QUARK end rather than
at the onset. A rootless-candidate story predicts the opposite shape, so if
the `lm` share does not concentrate where the cost does, this is not where the
mixed time is going and the ticket should say so rather than shave it.

### Gate

- The mixed gate, not njl's: the located `window` (n_onset, n_offset), chi,
  and the per-phase charge decomposition unchanged; P to 1e-8.
- Counts before and after from ticket 04's instrument, wall quoted beside cpu
  with the machine's load stated.
