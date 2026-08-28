# `solve_fermi_gl` returns a density three orders wrong and says it is accurate

Type: task
Status: resolved (2026-08-28)
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

---

## Ruling (2026-08-28): the threshold moves, and it moves to a NaN

**The recommendation was to raise the threshold, and it is raised — but not to
`solve_fermi_t0`, which is what "raise the threshold" would have meant read
literally.** The ticket proposed the one-line change without measuring what
there is to raise it *to*, and the measurement rules that half out. For a
nucleon gas (m = 939 MeV, g = 2), against JEL on the verify suite's own scaled
error:

| T / (mu - m) | Gauss-Laguerre | the T = 0 form |
|---|---|---|
| 0.10 | **6.9e-4** | n 1.4e-2, P 6.0e-2, s 100% |
| 0.08 | 4.3e-3 | " |
| 0.05 | 3.9e-2 | n 3.6e-3, P 1.6e-2, s 100% |
| 0.01 | 5.2e+1 | n 1.4e-4, P 6.4e-4, s 100% |

Two things kill the hand-off. First, **there is a gap**: GL leaves the suite's
2e-3 below T/(mu - m) ~ 0.1 and the T = 0 form does not enter it until ~0.02,
so a threshold anywhere between them returns a number no check would accept.
Second, and decisive at any threshold, **`solve_fermi_t0` returns s = 0
identically**. At T = 0.5 MeV the true entropy is 6.7e-3 fm^-3, not zero;
substituting the T = 0 form there replaces a density three orders wrong with
an entropy 100% wrong. That is the same defect the ticket was opened to remove,
wearing the fallback's name.

So the guard is on the degeneracy parameter, and outside the domain the
routine returns `(nan,) * 5`. §6's "non-convergence is a return value" is the
precedent and it is already this module's habit — `invert_fermi_density`
twenty lines below returns NaN for a target it cannot bracket rather than
raising or spinning. A NaN propagates visibly; a wrong number does not.

**What changed, two files:**

- `eos/general/fermi_integrals.py` — `GL_MIN_DEGENERACY = 0.1` declared beside
  the routine with the measured errors in its comment, the guard
  `if mu > m and T < GL_MIN_DEGENERACY * (mu - m): return (np.nan,) * 5`, and
  a docstring that states the domain instead of claiming "higher accuracy than
  JEL". The guard is on **mu > m** deliberately: `mu <= m` is the
  non-degenerate gas with no Fermi step to resolve, which is where the rule is
  at its best, and it is also the regime `test/dd2/test_dd2_m0.py:93` uses GL
  as its reference in (nu = 900 against m = 939). A guard written on T alone
  would have broken that test, and a guard written on `T / (mu - m)` without
  the `mu > m` limb would have divided by a negative.
- `eos/general/verify/run_full_check.py` — imports `GL_MIN_DEGENERACY` instead
  of re-declaring it. §7's single-home rule: the boundary is a property of the
  rule, so it belongs with the rule, and the suite that skips points outside
  the domain now reads the same constant the routine enforces.

**The T < 1e-4 fallback stays, and stays FIRST.** With the guard after it, a
degenerate gas at T < 1e-4 MeV still returns the T = 0 form exactly as before —
there s is genuinely negligible, not merely small — so the change touches only
the window 1e-4 < T < 0.1 (mu - m), which is precisely the broken one. No
existing call moves.

## Gate

- `general/verify/run_full_check.py` **PASS**, all six checks. `Fermi: JEL vs
  alts` reports `Gauss-Laguerre 3.1e-04 over 4` — unchanged, because the suite
  was already skipping the points the routine now refuses.
- New test, `test/general/test_fermi_integrals.py::
  test_gl_reports_its_domain_instead_of_a_wrong_number`, at the T = 0.5 MeV the
  ticket named, on **both** sides of the boundary from that one temperature:
  mu = m + 5 (T/(mu - m) = 0.1) agrees with JEL to 2e-3, mu = m + 500 returns
  NaN, and mu = m - 39 stays finite. That third assertion is the one that would
  catch a future guard rewritten on T.
- **No model number can move**, and this is measured rather than argued: the
  only importers outside `eos/general/` are the verify suite and one dd2 unit
  test, both green, and the ccdm baseline was instrumented with a spy on
  `solve_fermi_gl` that counted **0 calls**.

## Suite

**Anaconda Python 3.9.7 / numpy 1.26.4 / scipy 1.13.1** -- NOT the canonical
python.org 3.14.2 / numpy 2.3.5 / scipy 1.17.0 the ticket-99 counts were taken
on, so the numbers below are not comparable to that denominator:

- `test/general` + `test/dd2/test_dd2_m0.py` -- **154 passed, 0 failed**. That
  is the blast radius, not a sample of it: `solve_fermi_gl` has exactly two
  importers outside `eos/general/` and those are both of them.
- `python -m eos.general.verify.run_full_check` -- **PASS**, six of six.
- A whole-tree run reached 70% with 0 failures and was then stopped: it was on
  the wrong interpreter to produce a comparable count and was competing for CPU
  with a concurrent session's canonical run, which swept `test/baseline` and the
  first 58% of the tree with 0 failures on 3.14.2.

## A failure this session found and did NOT cause -- and it is the interpreter

`pytest test/baseline` fails on **`test_baseline[ccdm]`**, 108 quantities --
and the cause is the stack, not the tree. It PASSES on the canonical
python.org 3.14.2 / numpy 2.3.5 (a concurrent session's run cleared
`test/baseline` with 0 failures while this one was going) and fails on this
session's anaconda 3.9.7 / numpy 1.26.4. Two independent facts rule this
change out: the spy above counted **0 calls** to `solve_fermi_gl` through that
baseline, and `abpr` and `alphabag` pass on 3.9.7, so 3.9 is not globally
disqualified either. The worst key is `pattern.2SC.n1.5.T0.x` at 8.8e+03
relative -- a colour-superconducting pattern selection, exactly the
basin-sensitive quantity a scipy version moves -- alongside
`state.field_residual` at abs 1e-9 to 1e-5, a converged residual compared at
rtol = 1e-10.

**And `test/baseline_py39/` is not the answer to it.** The directory exists,
but `generate_baseline.path_for()` returns `HERE / f"{name}.npz"`
unconditionally, so nothing ever reads it -- and it is staler than the one
that IS read: fresh ccdm on 3.9.7 differs from `baseline_py39/ccdm.npz` in
**1256** of 6068 keys, against 108 for `baseline/ccdm.npz`.

Both halves belong in the map's fog, not in this ticket.
