# `pqm.tex` and `pqm.md`: the paper-style description

Type: task
Status: open
Blocked by: 10
Parent: ../map.md

## Question

CLAUDE.md section 11: each model carries `eos/<model>/<model>.tex` (and `.md`),
a short paper-style description with bibliography. **"It is part of the model,
not optional documentation."** The test is that a physicist can reproduce the
model from the document without opening the source.

This model needs the document more than most, because it is a FITTED object and
a reader's first three questions are ones no source file answers: what was it
fitted TO, on what grid, and where does it stop being valid.

### What section 11 explicitly rules out, all of which first drafts do

- **Naming a term instead of defining it.** `P = (1/4pi^2) sum_t a_t X_t` is
  not a statement of the pressure until every `X_t` is given in closed form.
- **Leaving the ideal-gas integrals to a citation.** The Fermi and Bose
  integrals are shared code, but each document states them anyway: *"a
  paper-style description is self-contained, and a reader of one model's `.tex`
  must not have to open another's."* Here that means the massive Fermi gas
  behind `gs150/gs300/gs450` is written out, not cited to `general/`.
- **Omitting a quantity because nothing derives from it.** `s` and `n_s` are
  returned and must appear, with the identities they are computed through.

### What this document must contain

**The ansatz, every term in closed form:** `a0`, `a2`, `a2s`, `a4`, the
alpha_s-weighted logarithmic terms (ticket 02) with the two-loop running
coupling written out, the SATURATING `a6` with `M_V = 3 M_g/(2 sqrt 2)` and why
it saturates (for `P ~ mu^k`, `c_s^2 -> 1/(k-1)`, so a bare `mu^6/mu_star^2`
sends `P/P_free -> infinity` and `c_s^2 -> 1/5`), `aT2`, `aT4`, `pair`, and the
three massive strange gases with the exact Fermi integral behind them.

**The flavour rotation**, with the repo's sign conventions stated: `S = +1` per
s quark (opposite to PDG), `C` non-leptonic, `mu_C = mu_p - mu_n`.

**Every derivative in closed form** -- `n_u`, `n_d`, `n_s`, `n_B`, `n_C`,
`n_S`, `s` -- and the identities `eps = -P + sum_a mu_a n_a + T s` and
`f = eps - T s = -P + sum_a mu_a n_a`, said to hold IDENTICALLY and why.

**The gap law** `Delta = Delta_star (mu_B/mu_star)^sigma sqrt(1 - (T/T_c)^2)`,
`T_c = tc_coeff Delta(mu_B, 0)`, with the BCS citation and the measured
departure ticket 05 found.

**The conformal limit**: which terms die, which saturate, and the two rows that
impose it.

**Per pattern, which terms the model carries and which it drops**, with the
reason -- `pair` is absent from unpaired because it multiplies zero; `a2s` is
absent from CFL because the locked phase is evaluated at a common potential
where `sum_f mu_f^2 = 3 mu_s^2` and `a2s` IS `a2/3`, and left in the least
squares returns a cancelling pair of order 1e9.

**The mode table**, saying which rows close the system in each mode, and that
CFL has no solution in a mode demanding `Y_C != 0` or `Y_S != 1` because
locking fixes them.

**The branch selection**: largest `P` at fixed potentials, lowest `f` at fixed
`n_B`, and why those differ inside a first-order window.

### The four things that are specific to a fitted model

**1. The fit itself is part of the physics here and must be described**: the
grid in `(mu_B, mu_C, mu_S, T)`, the residual rows (P and the charge densities
and `s`, scaled by the grid maximum), the conformal rows, the Tikhonov ridge
and why a smooth coefficient map needs it, and the bounded linear least
squares. A reader must be able to REDO the fit, not just evaluate the result.

**2. The parameter surfaces**: quadratic in `(eta_D, G_V0/G_S)`, the box they
were fitted over, and the documented behaviour outside it (ticket 09).

**3. The honest statements, which a paper-style description is exactly the
place for:**

- **The leading-order condensation term does not survive.** Pinning `pair = 1`
  costs ~20% against 0.2%; left free it goes to zero. The pairing physics
  reaches the model through the per-pattern coefficient set and through
  `(Delta_star, sigma)` -- the textbook COEFFICIENT does not. Say this; do not
  present a condensation term that is multiplied by zero.
- **`a4 > 1`.** The RG-consistent NJL medium (`lambda_UV = 10`) is STIFFER than
  a free quark gas, so the perturbative bound `a4 = 1 - 2 alpha_s/pi <= 1` is
  the wrong prior and `a4` pegs at it. The coefficient is a parametrization of
  `eos.njl`, not a coupling.
- **Colour potentials `mu_3` and `mu_8` are absorbed, not represented.** *"This
  is stated here because it is a real approximation, not an omission"* -- with
  ticket 06's measured size of the absorption.
- **Gapless, g2SC, gCFL and crystalline phases are absent**, and where NJL's
  ground state is one of them this model is wrong by construction.

**4. The domain.** No branch reaches 1 n_sat: each terminates at 1.5-2.4 n_sat
at `mu_B` = 850-1150 MeV, below which the T = 0 quark phase does not exist
because the chiral condensate is still there. The reachable window is ~2-13
n_sat, the fit is measurably worst near the terminus, and the fitted parameter
box is bounded. **A fitted model without a stated domain is a trap**, and
stating it is cheaper than the paper that discovers it.

Bibliography goes in the shared `docs/eos.bib`.

**Docstrings and this document stand on their own** (section 13): state the
physics, name the equation, give the literature citation -- never a plan, a
phase, a milestone number, or a `docs/` working note. This is a public
repository and `.scratch/pqm/` is not in it, so nothing here may be referenced
from the `.tex`.

## Gate

- `eos/pqm/pqm.tex` and `eos/pqm/pqm.md` exist, with the bibliography in
  `docs/eos.bib`.
- **The section 11 test, applied:** a physicist can reproduce the model --
  evaluate it AND redo the fit -- without opening the source. The cheapest way
  to check it is to have a session that has not read the code implement
  `pressure()` from the document alone and compare.
- Every basis term in closed form; every derivative written out; `s` and `n_s`
  present with the identities they come through.
- The four honest statements above, each present, each with its number.
- The domain stated in `n_B`, in `mu_B`, in `T`, and in the fitted `(eta_D,
  G_V0/G_S)` box.
- No reference to `.scratch/`, to a ticket, or to a milestone anywhere in the
  document.
