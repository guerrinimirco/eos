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
