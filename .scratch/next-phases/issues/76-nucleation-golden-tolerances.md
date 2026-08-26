# The two `nucleation` goldens that compare round-off

Type: grilling
Status: resolved
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


---

## Resolution

Both goldens asserted a quantity no assertion should have been reading. Neither
is fixed by changing a number, and neither change is a loosened tolerance under
the rule this ticket also settled.

### The rule, first

**A tolerance is LOOSENED when the assertion measures the same quantity with
more slack; it is CORRECTED when it measures a different, better-chosen
quantity.** CLAUDE.md §12 forbids the first absolutely. It does not forbid
replacing a relative comparison of a quantity that is zero by construction with
an absolute bound on that zero: `abs(mu_C) < eps` is not a slacker version of
`mu_C ~= -1.1369e-12`, it is a different and strictly stronger claim. This
wording is now on the map's hard-rules list, because the bare rule fails OPEN —
a session meeting a red golden either loosens it or blocks on a question already
answered here. It also forbids the one-character fix a tired session reaches
for: raising the `abs(v) > 1e-12` guard to `1e-7`.

### `test_regression_solver_cases`

The guard `if abs(v) > 1e-12` admitted quantities that are zero by construction
to a RELATIVE comparison against a recorded round-off value. Those values are
not physics: the CFL lock is not an algebraic identity here but a pair of
**residual rows** of a root find (`Qs.Y_C`, `Qs.Y_S - 1` in
`solve_saddlepoint_cfl`), because `m_s = 100 MeV` makes equal flavour densities
require unequal potentials. The golden therefore recorded where `robust_root`
happened to stop, and the assertion pinned the floating-point ASSOCIATION of the
quark block — which is exactly what the `general.basis.quark_charges` adoption
changed, by one ulp, while moving no physical quantity.

It now asserts the lock, absolutely, at **`LOCK_TOL = 1e-8`**. That number is not
chosen: it is `robust_root`'s own `atol`, the tightest claim the code makes about
these quantities. `eos/alphabag/verify/run_full_check.py:373` uses `1e-10` for
the same physics, and it was rejected precisely because it is `eos`'s solver's
gate rather than `nucleation`'s — asserting tighter than the solver promises is
how a test fails on a correct answer, the same trap one level down.

**The zero set is charge-mode dependent, which this ticket's framing missed.**
Measured over every CFL case, not only the failing ones:

| flavor/charge | `Y_C` | `Y_e` | `mu_C` | `mu_e` |
|---|---|---|---|---|
| saddlepoint/**lcn** | -7.1e-15 | +5.6e-13 | -1.2e-12 | +3.2e-08 |
| saddlepoint/**gcn** | +4.5e-12 | **+1.49e-01** | +3.1e-10 | **+3.17e+02** |
| saddlepoint/**gcn_coulomb** | +4.5e-12 | **+1.49e-01** | +3.1e-10 | **+3.17e+02** |

`mu_e` and `Y_e` are NOT zero under CFL. Under global neutrality the droplet is
charged and `mu_e = mu_e^H` is a real ~317 MeV electron sea; they vanish only
under LOCAL neutrality, where the droplet must be neutral alone and CFL already
is. A flat per-phase zero list would have asserted a physical electron sea to
zero. So:

- zero under `cfl`, every charge mode: **`Y_C`**, and **`mu_C`** — a LINEAR
  image of it (`mu_u - mu_d` at equal massless-flavour densities), so it
  inherits the gate once scaled by `mu_B`
- zero under `cfl` AND `lcn` only: **`Y_e`**
- **`mu_e` under `lcn` is dropped, not bounded.** It is a NONLINEAR image of
  `Y_e` (`n_e ~ mu_e T^2`), so no bound on it follows from the gate: a point
  converging to `Y_e` AT the gate would carry `mu_e` five orders above where it
  sits now and still be correct. `abs(Y_e) < 1e-8` is the physics claim; `mu_e`
  restates it through a temperature-dependent map for no gain.

Removing the `abs(v) > 1e-12` guard is clean: it was skipping only these CFL
crumbs and `mu_S = +0.0` exact in the unpaired cases, which passes under
`pytest.approx`'s default `abs=1e-12`. If that ever became a crumb the test
fires, and correctly — there is no solver gate it could legitimately drift
within.

### `test_energy_barrier_matches_golden`

`max|dW| < 1e-9` was absolute on a curve running from 0 at `R = 0` to 2.7e6 MeV.
At the top that bound is **4e-16 relative — below one ulp of a double**, so the
test asserted bit-identity with a rounding allowance and was never going to
survive any reassociation. It is now `max|dW| < 1e-12 * max|W|`. Pointwise
relative was rejected: the curve passes through zero at `R = 0`, and the
measured worst pointwise deviation (6.5e-14) sits at the small-`|W|` end where
relative error means nothing. `1e-12` is four orders tighter than the `rel=1e-8`
this suite already accepts on `W_c` two tests over.

### No `eos` diff

`eos/alphabag/verify/run_full_check.py:318` already asserts the CFL lock
absolutely (`abs(r.Y_C)`, `abs(r.Y_S - 1.0)`), and `:373` tightens neutrality to
`1e-10`. `eos`'s own tripwire for this physics was correct throughout and would
have caught a real break; the `nucleation` golden was the outlier, not the
proposal. The ulp shift is recorded here and belongs in the Stage 7 report as a
consequence of §2's shared-basis adoption, so a future session meeting it in
another `quark_charges` consumer recognises it. A sweep of the other consumers
(`njl`, `ccdm`, `vmit`, `abpr`, `test/baseline/*.npz`) was NOT run — the map's
"only the changes a ticket asks for" rule. Worth noting for that report:
`test/baseline/alphabag.npz` was regenerated by ticket 62 on py3.14, i.e. AFTER
the refactor, so it froze the new arithmetic and could not have caught this.

`test/golden/regression.json` is untouched. Regenerating a golden hides the next
real change the same way this one was hidden, and the recorded crumbs are a true
record of what the old kernel returned — the input to ticket 24's A/B.

### Measured

Canonical stack (python.org 3.14.2 / numpy 2.3.5 / scipy 1.17.0), worst case
over every golden case:

    Y_C          4.505e-12   against 1e-8      2,220x headroom
    mu_C/mu_B    2.480e-13   against 1e-8     40,326x
    Y_e (lcn)    5.609e-13   against 1e-8     17,829x
    W(R) rel     4.506e-15   against 1e-12       222x

The lock bound separates a locked phase from an unlocked one by **4.1e6**:
unpaired `|Y_C|` runs 0.041 .. 0.182. The `W(R)` headroom is 222x on 3.14
against 370x on 3.9 — the tighter of the two, and the number to compare against
if it ever moves again.

**Full `nucleation` suite, green on BOTH stacks:**

    python.org 3.14   72 passed in 6.44s
    anaconda 3.9.7    72 passed in 27.11s

3.14 needs `PYTHONPATH=<eos repo>` and `arch -arm64` — the shell runs under
Rosetta and 3.14's numpy is arm64-only, which presents as a numpy C-extension
`ImportError`, not as an architecture message pytest surfaces.

Landed as `37af659` in `nucleation`, two pathspecs. Ticket 72 committed as
`569296a` mid-session, so the suite had already moved to `test/`; the two files
were verified byte-identical to HEAD before editing.

**Ticket 25's noted wrinkle is dissolved, not decided.** The criterion names
`nucleation/test/`, and that path now exists. Nothing left for 25 to read the
line around.
