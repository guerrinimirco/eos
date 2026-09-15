# Bound the rescue ladder below a branch

Type: prototype
Status: open
Blocked by: 05
Parent: ../map.md

## Question

The three-pattern list this ticket measures against is now the DECIDED
default, not a restriction: [Does `free` belong in the default
enumeration?](12-free-in-the-default.md) dropped `free` from
`DEFAULT_PATTERNS`. Nothing below changes; the numbers simply stopped being
conditional on a `patterns=` argument.

This is ticket 05's original target — abandoning work on a branch that is not
there — in the one place it actually lives, now that `free` is separated out
([The cheap pre-screen](05-cheap-pre-screen.md), [Does `free` belong in the
default enumeration?](12-free-in-the-default.md)).

With the three-pattern list, **13 of 600 candidates take 77.1% of the build**.
They are the `CFL` candidates below the CFL onset: each stalls its first
Newton run at ~9e-2, runs `attempt`'s MINPACK rungs, then the differenced
solve, and **converges onto the 2SC root** — a duplicate of a candidate that
already converged in its own layout, for 3.0 s.

**How much of that ladder can go, given that exactly one candidate in 200 is
rescued by it into a real CFL layout?**

### Measured at ticket 05 (three patterns, 600 candidates, 53.4 s, quiet
machine — shares, not absolute times)

| stage of `attempt` | n | s | % |
|---|---|---|---|
| B MINPACK + polish after the first Newton | 28 | 29.3 | 54.8% |
| A the first Newton (jac), all candidates | 600 | 12.1 | 22.6% |
| D the differenced rescue (no jac) | 13 | 11.2 | 21.0% |
| C the reinflate rescue | **0** | 0.0 | 0.0% |

| what the rungs below the first Newton bought | n | s | % |
|---|---|---|---|
| `CFL` -> converged as 2SC (duplicates) | 13 | 39.2 | 73.3% |
| `2SC` -> held its layout | 14 | 0.8 | 1.6% |
| `CFL` -> **held its layout** (the onset) | **1** | 0.6 | 1.1% |

### The two things this ticket must not break

- **The onset is load-bearing.** The single rescued CFL candidate is what puts
  a CFL seed into `_seeds`; without it CFL is cross-seeded at the next density
  too, stalls again, and the branch — 170 of 200 rows — is never found. Ticket
  05 measured that an abort threshold on the screen residual cannot tell that
  candidate from the 13 dead ones: they screen in the same 8-9e-2 band, with
  all three gaps alive in both.
- **A collapsed candidate must not seed forward.** `solve`'s `_seeds` filter
  already handles this and stays.

### The cheapest thing to try first

Rung D produced **13 duplicates and nothing else**. Dropping it for a
cross-seeded candidate returns `converged = False` at those 13 points — which
is a *truer* report than a CFL candidate that is silently a 2SC state, and
drops it from the ranking rather than into it. Measure: the ms/pt, the gate,
and whether the onset at n_B = 0.6583 still lands.

Then rung B, which is the larger 54.8% and is also where the onset is found —
so it is bounded, not removed.

## Gate

- ms/pt on the pinned benchmark, three patterns, before and after, on a
  machine whose load is stated (ticket 05's window ran 2.7x slow).
- The map's gate: `pattern_realised` unchanged at every density, P to 1e-8,
  worst point named.
- **The 2SC -> CFL onset still found at n_B = 0.6583 fm^-3**, and the CFL
  branch still 170 rows.
- A statement of what a caller now sees at the 13 densities where a CFL
  candidate has no root.
