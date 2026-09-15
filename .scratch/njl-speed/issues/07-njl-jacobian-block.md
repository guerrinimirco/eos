# Give `njl_phase` a `jacobian_block`, and turn `analytic_jac` on

Type: prototype
Status: open
Blocked by: 04
Parent: ../map.md

## Question

`eos/mixed` already has everything needed for a true Newton on the mixed
residual, and a DD2+NJL hybrid uses none of it.

Found while charting:

- `Phase.jacobian_block(mu, mu_C, mu_S, T, state, th) -> ndarray` is part of
  the adapter contract (`eos/mixed/adapters.py:145`).
- `eos/mixed/backends/jacobian.py::mixed_jacobian` assembles the mixed Jacobian
  from the two phase blocks, and `solver.solve(..., analytic_jac=True)` wires it
  in with a finite-difference fallback for penalised trial points.
- **Only `dd2_phase` and `vmit_phase` advertise a block** (adapters.py:641 and
  :710). `njl_phase` (adapters.py:1144) passes nothing.
- `_jac_with_fallback` (solver.py:533) returns `None` if **any** phase in the
  pair lacks a block — so pairing NJL with DD2 silently disables the analytic
  Jacobian for the hadronic side too.
- **`analytic_jac` defaults to `False`** in `api.py`, `hybrid.py`, `table.py`
  and `solver.py`, so even a DD2+vMIT pair does not use it unless asked.

So the contract does not need to change (this corrects an assumption made while
charting). Two questions instead:

1. **What is NJL's block?** It is the charge-susceptibility matrix
   `dn_a/dmu_b` for a, b in (B, C, S) at fixed T, *including* the response of
   the internal unknowns (M_u, M_d, M_s, the gaps, Sigma_V) — the implicit
   function theorem through the gap equations, not a partial derivative at
   frozen state. `eos.general.pairing.pair_hessian` and
   `eos/njl/backends/jacobian.py::residual_jacobian` are the ingredients;
   whether the internal block can be reused directly or must be re-partitioned
   is the work.
2. **Should `analytic_jac` default to `True`** where every phase in the pair
   advertises a block? It is currently opt-in in four places, which is four
   places for it to be forgotten.

### What it is worth

`hybr` with finite differences costs n+1 residuals per Jacobian, and every
residual is two full phase solves. Ticket 04 measures the real multiplier;
the naive one for ~5 unknowns is ~6x on the Jacobian-forming calls alone.

### Traps

- Inactive-gap rows and columns of `pair_hessian` are **zero by construction**
  (the block is absent), and `dk/dDelta` at `Delta = 0` is **not** the pair
  susceptibility. Fine as solver input; wrong if read as physics.
- Verify the block by finite differences of `thermo` in (mu_B, mu_C, mu_S) —
  but only with the RG panel rule and **moving** nodes, or the reference is
  noise at 1e-3. Judge each scaled row against its own largest entry; a colour
  row at a symmetric root is identically zero.
- Turning the default on changes results only through the iteration path, so
  the map's gate applies: same pattern, P to 1e-8.

### Measured at ticket 02 — the held-pattern assumption has moved

DID + NJL, eta = 0, T = 0, n_B = 1.000 fm^-3, one `eos.mixed.eos_point`:
the **enumeration** converges in 29.6 s; **held CFL** takes 109.5 s and does
NOT converge; **held 2SC** converges in 9.2 s but at chi = +0.3098 against the
enumeration's +0.9745 — a different state at the same density. So "hold a
pattern to make the mixed solve cheap and stable" is not true on this pairing,
and a Jacobian block has to serve the ENUMERATING adapter to be worth having.

Also from ticket 02, and squarely this ticket's business: `eos.mixed.eos_table`
with an enumerating njl adapter did not return in 24 minutes on a FOUR-point
grid, while 20 separate `eos_point` calls took 2576 s. The window locator, not
the point solve, is what makes a hybrid table unaffordable.

## Gate

- `njl_phase` advertises a `jacobian_block`, checked against finite differences
  with the caveats above, worst scaled row reported.
- One hybrid row solved with `analytic_jac=True` and `False`: NJL `thermo` call
  count and wall time for each, plus the same converged answer.
- A ruling on the default, with the reason, and the four call sites made
  consistent with it.
- `test/mixed/` run and reported; check HEAD for the known pre-existing failures
  before attributing any to this change.

## Measured at ticket 04 — THIS TICKET'S STATED LEVER IS NOT THERE

Read [ticket 04](04-count-the-mixed-loop.md) before starting. Three findings
bind this ticket:

- **"`hybr` costs n+1 residuals per Jacobian, and the naive multiplier is ~6x"
  is wrong.** Instrumented over 47 `root` calls on this pairing, **every single
  one formed the Jacobian exactly once** (4 columns, n = 4) and Broyden-updated
  thereafter; `scipy`'s `nfev` confirms it (cold row: 46 = 1 + 4 + 41). The FD
  columns are **9-24% of residuals**, so removing them entirely is a
  **1.1-1.3x ceiling**, not 6x. `hybr` re-forms the Jacobian LESS often than a
  Newton would, not more.
- **What might still be there is the iteration path, and it is unproven.**
  25 to 107 trial steps per solve for four unknowns is a lot, and they are
  taken on a Jacobian that is up to 100 Broyden updates stale. An exact block
  could plausibly shorten that — but nothing measured says it does, and this
  ticket must now MEASURE that rather than assume the 6x. If the step count
  does not fall, the whole change is worth 1.1-1.3x.
- **`did_phase` carries no `jacobian_block` either.** Only `dd2_phase` and
  `vmit_phase` do, and `_jac_with_fallback` returns None if ANY phase in the
  pair lacks one. So giving `njl_phase` a block does nothing at all on the
  DID+NJL pairing every number in this map is taken on: either DID gets one
  too, or the demonstration moves to DD2+NJL. Decide which before building.

**And the locator is no longer this ticket's business.** The sentence at the
bottom of the charting notes — "the window locator, not the point solve, is
what makes a hybrid table unaffordable" — is confirmed (76% of a build) and has
become [ticket 11](11-cheapen-the-locator.md). This ticket is now the Jacobian
block and nothing else.
