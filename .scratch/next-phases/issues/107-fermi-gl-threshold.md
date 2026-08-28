# `solve_fermi_gl` returns a density three orders wrong and says it is accurate

Type: task
Status: open
Blocked by: -
Parent: ../map.md

## Question

Found by [ticket 64](64-general-verify-suite-missing.md), reported and not
fixed. `eos/general/fermi_integrals.py::solve_fermi_gl` falls back to the
analytic forms only below `T < 1e-4` MeV but breaks down around
`T/(mu - m) ~ 0.08`: at T = 0.5 MeV it silently returns a density **three
orders of magnitude wrong**, while its docstring claims "higher accuracy than
JEL".

**Checked this session: nothing computes with it.** The only importer outside
`eos/general/` is `general/verify/run_full_check.py`, which uses it as one of
the alternatives being validated against JEL. `njl` does NOT use it — that
model's integrals are its own cutoff-regularized ones, which §7 classifies as
model physics rather than an integral re-implementation. So no model number is
affected today, and the defect is a trap for the next caller rather than a
live wrong answer.

## Two ways, and the ruling is which

- **Raise the threshold** — a one-line change plus a docstring that states the
  domain instead of overclaiming. §7's spirit favours this: alternatives are
  "supplemented, never replaced", and a validated alternative to JEL has value
  precisely because the verify suite cross-checks it.
- **Delete it.** §7 protects JEL, not every alternative, and a routine no
  solve path calls whose docstring is false is a liability. The verify suite
  loses one of its five checks and says so.

Recommendation: raise the threshold. Deleting removes a cross-check that has
already earned its keep — it is what caught the split-panel entropy being 1.1%
off JEL — and the fix is smaller than the deletion.

## Gate

Whichever is chosen, `general/verify/run_full_check.py` still passes and its
docstring says what is checked against what. If the threshold moves, one test
at T = 0.5 MeV asserting agreement with JEL where the docstring now claims it.
No model number can move either way.
