# The two `nucleation` goldens that compare round-off

Type: grilling
Status: open
Assignee: session 5687bd50
Blocked by: (none — 24 is resolved)
Parent: ../map.md

## Question

Surfaced by [ticket 24](24-phase6-execute.md), whose gate required it be
reported rather than fixed. Two tests fail after the port, and **neither is
caused by an import or a call site**:

    FAILED nucleation/tests/test_composition.py::test_regression_solver_cases
    FAILED nucleation/tests/test_critical.py::test_energy_barrier_matches_golden

Ticket 24 verified the cause by an A/B that holds the ported `nucleation` code
fixed and swaps only the alphaBag kernel — the pre-refactor
`eos.alphabag.thermodynamics_quarks`, archived from `e44578a^`, against today's
`eos.alphabag.thermodynamics`:

    old kernel, ported nucleation   2 passed
    new kernel, ported nucleation   2 failed

The refactor changed the **floating-point association** of the quark block —
CLAUDE.md §2's shared `eos.general.basis.quark_charges` replacing the model's
inline charge sums, and a different route to the massive flavour's density —
by about one ulp:

    n_s   old 0.8170095995673604    new 0.8170095995673602
    n_B   old 0.8108295267682285    new 0.8108295267682284

Every physically nonzero quantity still matches its golden. What fails is
round-off, and in both cases because of what the golden asserts, not what the
code computes:

- `test_regression_solver_cases` — ten keys exceed `rel=1e-9`, and **all ten
  are quantities the CFL flavour lock forces to zero**: `mu_e ~ 3e-08`,
  `mu_C ~ 3e-10`, `Y_C ~ 4.5e-12`. The guard is `if abs(v) > 1e-12`, which
  admits a value that is zero by construction to a RELATIVE comparison.
- `test_energy_barrier_matches_golden` — `max|dW| = 3.027e-09` against an
  **absolute** bound of `1e-9`, on a `W(R)` curve reaching `-1.4875e+06` MeV.
  Relative deviation `2.0e-15`, about nine ulp; the bound is absolute on a
  quantity spanning six orders of magnitude.

**The decision this ticket owes.** Not "loosen the tolerance" — CLAUDE.md §12
forbids that, and the map's hard rules repeat it. The question is what the two
goldens should ASSERT, given that a golden frozen bit-for-bit against one
version of a shared arithmetic kernel is not a physics tripwire:

- for the near-zero CFL quantities: is the right assertion `abs(v) < eps`
  (they are zero by construction) rather than a relative match to a recorded
  round-off value? If so, what raises the `abs(v) > 1e-12` guard to?
- for `W(R)`: should the bound be relative to `max|W|`, or absolute but scaled
  to the curve? The conftest header calls the goldens "a TRIPWIRE, not a
  target" — this is asking what the tripwire is for.
- and: does regenerating either golden count as loosening? Regenerating hides
  the next real change the same way; deciding the assertion does not.

Whether the answer lands in `nucleation` or is also an `eos` matter (the ulp
shift is `eos`-side and every other consumer of `quark_charges` inherits it) is
part of the decision.

## Why it blocks Acceptance

[Ticket 23](23-phase6-respec.md) ruled `25 <- {21, 24}`, not 72, on the ground
that "nothing in [the Acceptance criteria block] reads a README". That argument
is right and it cuts the other way here: the block's **first line** is
`pytest eos/test/ and pytest nucleation/test/ fully green`, and these two
failures are exactly what stops that being true. So this ticket is wired
`25 <- 76` by the same reasoning that kept 72 out.

**A wrinkle for [ticket 25](25-acceptance.md), noted not resolved:** that
criterion names the path `nucleation/test/`, which only exists after ticket
72's move. The criterion therefore straddles the 24/72 split in a way ticket
23's ruling did not anticipate. Ticket 25's session decides whether to read the
line as "the suite, wherever it lives".
