# Should `uSC` and `dSC` join the default enumeration?

Type: prototype
Status: open
Blocked by: 12, 13
Parent: ../map.md

## Question

[Does `free` belong in the default enumeration?](12-free-in-the-default.md)
removed the asymmetric probe on the grounds that `uSC` and `dSC` reach the
asymmetric states better under their own names. It did **not** put them in the
default, and that is the debt it left: today's decided default,
`("unpaired", "2SC", "CFL")`, enumerates **no asymmetric candidate at all**.

**Do `uSC` and `dSC` join `DEFAULT_PATTERNS`, and at what cost?**

### Why ticket 12 did not simply bundle this

Cold, the pair costs what `free` cost -- **+5.3 s against `free`'s +5.7 at
T = 0, and +1.5 s against +1.0 at T = 30** (`../t12_probe.json`). Trading one
for the other cold buys nothing. The entire case rests on the one structural
difference: `uSC` and `dSC` are in `_REALISED`, so a converged candidate that
holds its layout **seeds the next density**, which `free` could never do. That
warm-started cost is unmeasured, and at T = 0 `uSC` does not converge at all --
it burns the same rescue ladder [ticket 13](13-bound-the-rescue-ladder.md)
owns, which is why this is blocked on 13 rather than on 12 alone.

### A precondition, not a discovery

`_left_layout` (`eos/njl/solver.py:822`) drops a candidate that left its layout
only for `("2SC", "CFL")`. A `uSC` or `dSC` candidate that falls to 2SC is kept
and **competes** -- ticket 05's "the reported `pattern` column is not stable"
defect, with a new name on it. Ticket 12's cold table shows it in every T = 50
row and at T = 0, n_B = 0.8. **Widen that filter first**, then measure.

### The physics question ticket 12 deferred here

Twelve cold points over T = 0/30/50 and n_B = 0.8-2.0 found **no asymmetric
state winning anywhere**, and `dSC` converged as a real state losing to CFL by
57-114 MeV/fm^3. But Gholami, Hofmann & Buballa PRD 111, 014021 (2025) --
`njl.md` section 17.8, the paper `eos.njl` implements by default -- find the
RG-consistent phase diagram **melts CFL in a dSC pattern**. Melting is a
finite-T phenomenon and **that region is not inside the probed box.**

So this ticket owes the evidence ticket 12 declined to produce: a T-scan at
fixed n_B, carried until CFL loses, watching **what it melts into**. If it
melts into dSC, an asymmetric candidate belongs in any default that claims to
cover finite T, whatever it costs. Nothing here varies `G_D`/`eta_D` or `m_s`
either, which is the other axis uSC/dSC are argued on.

### Gate

- The melting pattern located: the (n_B, T) where CFL stops winning, and the
  state that replaces it, on the pinned parameter set.
- Warm-started ms/pt for `("unpaired", "2SC", "CFL", "uSC", "dSC")` against
  today's three, on the pinned benchmark, machine load stated.
- `_left_layout` widened, with the `pattern` column shown stable.
- A decision, with the same test ticket 12 used: does the candidate find
  anything the others do not, and can it be cached?

### Handed in from ticket 15 (2026-09-21): ccdm's box is not njl's

Ticket 15 ran ticket 12's probe on ccdm (`t15_ccdm_probe.json`: T = 0/30/50,
n_B = 1.3-2.5), and three things in it belong here:

- **An asymmetric state WINS**: at T = 50, n_B = 1.3, uSC beats 2SC by 0.20
  MeV/fm^3. That is the first win in this effort.
- **In ccdm the named `uSC` seed is the unreliable one.** At T = 30 it
  collapses (to unpaired or 2SC) at n_B = 1.3, 1.6 and 2.0, where `free`
  reaches a uSC state no named seed does. Those states lose by 4.3-8.5.
  "Does the candidate find anything the others do not" has a different answer
  per model.
- **ccdm's enumeration misses the CFL ground state** at three of those twelve
  points (+4.3, +22.6 and +6.0 MeV/fm^3), with or without `free`: the
  cross-seeded CFL candidate collapses, keeps its name and competes, and
  `eos/ccdm/solver.py` has no `_left_layout` at all. The precondition this
  ticket names ("widen that filter first") is, for ccdm, "WRITE that filter
  first". It is a correctness defect in its own right, and ticket 15 recorded
  it as needing its own ticket.
