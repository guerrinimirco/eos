# Make mu_S determined when the strange sector is empty

Type: grilling
Status: open
Parent: ../map.md

## Question

**The concern to settle first, because it shapes everything below.** When no
strange species is thermally populated, `mu_S` is not merely hard to pin — it is
**undefined**. The strangeness row `n_S = n_B Y_S` is satisfied identically for a
whole range of `mu_S`, the Jacobian is singular in that direction, and every
value in that range describes the *same physical state*: same `eps`, same `P`,
same `mu_B`, same densities. So "make it determined" cannot mean recovering a
hidden true value. It means **choosing a convention** and applying it
consistently. The ticket is worth doing anyway — a reported number that moves
with the solver path is worse than a declared convention — but nobody should
discover this halfway through the implementation.

Measured at [ticket 37](37-did-failures.md): in `did` at Y_C = 0.3, Y_S = 0,
T = 30 MeV the strange densities are ~1e-11 fm^-3, `n_S ~ 1e-16`, and `mu_S`
lands at −515 MeV (n_B = 0.32) and −532 MeV (n_B = 0.64). The looseness is
~500x larger at 0.32 than at 0.64. `docs/DEFERRED.md` records the same for
`sfho` and `vmit`, plus the sibling case **`mu_e` where no electrons are
present** at Y_C = 0, seen in `vmit`, `dd2`, `sfho`, `zl` and `alphabag`.

**Scope.** `fixed_YC_YS` is exposed by 11 models (`zl`, `sfho`, `dd2`, `did`,
`vmit`, `alphabag`, `njl`, `ccdm`, `abpr`, `enjl`, `mixed`). The strangeness row
is assembled per model — `eos/enjl/solver.py:397`, `eos/sfho/solver.py:445`,
`eos/vmit/solver.py:505,540`, `eos/mixed/solver.py:525` — so a per-model patch is
11 edits that will drift apart. §7's single-home rule points at shared machinery
in `general/` instead.

**Candidate conventions**, to be chosen before any code:

- **`mu_S = 0` when the strange sector is empty.** Declares "no strangeness
  driving force". Simple, defensible, and reported values become reproducible.
  Needs a threshold on what "empty" means, and that threshold is then a number in
  the physics.
- **Minimum-norm step in the singular direction** (pseudo-inverse or
  Levenberg–Marquardt damping). No threshold and no special case, and it
  regularises the `mu_e` sibling case for free — but the landing point still
  depends on the initial guess, so it makes `mu_S` *stable*, not *determined*,
  unless the guess is also fixed by convention.
- **Continuity in Y_S.** Define `mu_S` at Y_S = 0 as the limit from Y_S > 0.
  Physically the most satisfying and threshold-free, but it costs an extra solve
  per point and the limit may not be finite.

**This changes numbers.** Every model whose baseline contains a `fixed_YC_YS`
line at Y_S = 0, or a charge-neutral line at Y_C = 0 if the `mu_e` sibling is
taken too, needs re-pinning — in its own commit, quoting the before/after delta,
never a blanket regeneration (`test/baseline/test_baseline.py`'s own
instructions). §12 makes those baselines ground truth, so each delta must be
shown to be confined to the undetermined potentials and their `mu_eff_i`
descendants, exactly as ticket 37 showed for `did`: `Delta mu_eff_i = S_i x
Delta mu_S`, no nucleon, no density, no `eps` or `P`.

**Decide explicitly whether `mu_e` at Y_C = 0 is in scope.** It is the same
defect with a different conjugate density, it affects five models, and a
convention chosen for `mu_S` alone will look arbitrary beside it.

Once the convention is settled the implementation becomes a task ticket; this one
is the ruling.

## Measurement — the premise above is wrong: mu_S is determined, not free

`dn_S/dmu_S` at the `did` Y_S = 0 solutions, T = 30 MeV, Y_C = 0.3, computed
through `thermo_from_mu` at the solved fields:

| n_B | dn_S/dmu_S [fm^-3/MeV] | mu_S slack a 1e-10 residual admits | drift observed in the baseline |
|---|---|---|---|
| 0.32 | 7.8e-07 | 1.3e-04 MeV | **2.3e-05 MeV** |
| 0.64 | 5.5e-03 | 1.8e-08 MeV | **4.1e-08 MeV** |

**The observed drift is the convergence gate's slack**, to within a factor of
about two at both densities. The gradient is small but far from zero, so the
Jacobian is **ill-conditioned, not singular**, and `mu_S` has a true value the
solver simply is not required to reach.

Why it is non-zero, which the earlier reading missed: at T = 30 MeV the strange
sector is not empty, it is *cancelling*. The Xi densities come out **negative** —
net anti-Xi, since mu_S is strongly negative — so `n_S = 0` is reached by
Lambda's +5.97e-11 against Xi0's 2 x (-3.09e-11), not by nothing being there.
A cancellation has a gradient; an empty sector does not.

*Caveat:* the derivative holds the fields and `mu_tilde_B` fixed, so it is a
partial rather than the sensitivity along the constrained solution manifold.
The conclusion rests on the drift magnitudes matching the implied slack, not on
the derivative alone. Anyone implementing this should re-measure the constrained
sensitivity first.

### What this changes

**None of the three candidate conventions is the right fix**, and the ticket
title is wrong: nothing needs to be *made* determined. The fix is numerical —
**scale the strangeness residual row** so the convergence gate is judged on
whether `mu_S` is pinned rather than on whether `n_S` is small in absolute terms.
A row whose conjugate density responds at 1e-06 fm^-3/MeV cannot be gated at the
same absolute tolerance as one responding at 1e-01 and be expected to pin its
potential equally well.

That approach:

- needs **no convention** and no threshold on "empty", so nothing arbitrary
  enters the physics;
- changes no equilibrium state — only how tightly the solver is required to
  land, so `eps`, `P`, `mu_B` and every density are untouched;
- **generalises to the `mu_e` sibling** at Y_C = 0 for free, where
  `docs/DEFERRED.md` measures `dn_e/dmu_e ~ 4e-06 fm^-3 MeV^-1` — the same
  situation with a different conjugate density;
- is far cheaper than the per-model residual surgery the three conventions
  implied, since `residual_scales` machinery already exists in at least `enjl`.

**`docs/DEFERRED.md` needs correcting either way.** Its cross-cutting entry says
"the residual has no gradient in that direction and the Jacobian is singular
there". The gradient is 7.8e-07 at n_B = 0.32; the entry describes a degenerate
system where the truth is a badly scaled one, and that misreading is what made
three people reach for a convention.

## Constrained sensitivity — corrects the section above, and kills one option

The measurement above held the fields and `mu_tilde_B` fixed, which the ticket
flagged as a partial rather than the sensitivity along the solution manifold.
Measured properly — by solving at small non-zero `Y_S`, so every other unknown
relaxes and every other row stays satisfied — `did`, Y_C = 0.3, T = 30 MeV:

| n_B | Y_S | mu_S [MeV] | n_S | eps | P |
|---|---|---|---|---|---|
| 0.32 | 0 | **−515.324** | 1.6e-16 | 308.02317 | 18.352643 |
| 0.32 | 1e-6 | −261.439 | 3.2e-07 | 308.02322 | 18.352626 |
| 0.32 | 1e-5 | −192.360 | 3.2e-06 | 308.02363 | 18.352474 |
| 0.32 | 1e-4 | −123.268 | 3.2e-05 | 308.02780 | 18.350960 |

**Two corrections to what is written above.**

1. **The constrained `dn_S/dmu_S` is 1.26e-09 fm^-3/MeV at n_B = 0.32** (3.04e-09
   at 0.64), roughly 600x smaller than the partial derivative. So a 1e-10
   residual admits **0.079 MeV** of `mu_S`, not 1.3e-04. The observed baseline
   drift of 2.3e-05 MeV is therefore **far smaller than the gate permits** — the
   claim above that "the observed drift is the gate's slack" is wrong. The
   solver happens to land tightly; nothing requires it to. The quantity is
   looser than the baseline's rtol = 1e-10 implies by about three orders of
   magnitude, which strengthens rather than weakens the case for scaling that
   row.

2. **`mu_S` diverges logarithmically as Y_S -> 0+.** The steps are 69 MeV per
   decade of Y_S, which is `T ln 10` at T = 30 MeV — the Boltzmann tail. So
   `mu_S -> -inf`, the limit does not exist, and **the "continuity in Y_S"
   candidate convention is dead**: there is nothing to take a limit of. The
   finite −515.32 returned at Y_S = 0 is not on that branch at all; it comes
   from the particle/antiparticle cancellation (the Xi densities are negative,
   net anti-Xi), which is a different solution of the same row.

`eps` and `P` move in the fifth and sixth digit across the whole Y_S range,
confirming that none of this is physically consequential — which is exactly why
it should not be pinned at rtol = 1e-10 in a regression baseline.

### Where this leaves the three candidates

- **Continuity in Y_S** — dead, per above.
- **`mu_S = 0` when the sector is empty** — still viable, but note it would be a
  discontinuity of ~515 MeV against the Y_S -> 0+ branch, so the threshold is
  not cosmetic.
- **Minimum-norm / damped step** — still viable and now the most attractive: it
  needs no threshold and no special case, and with the true sensitivity at
  1e-09 the damping term dominates the singular direction cleanly.

**Recommended, and cheaper than all three:** scale the strangeness row by the
conjugate density's own responsiveness so the gate measures `mu_S`'s
determination, and — separately — **stop pinning `mu_eff_i` for S != 0 species
in `fixed_YC_YS` lines at Y_S = 0 in `test/baseline/`**. Those two together
close the failing test without touching physics. The second is not a tolerance
loosening in §12's sense: it removes a quantity that is not reproducible in
principle from a regression net, rather than widening a tolerance on one that is.
