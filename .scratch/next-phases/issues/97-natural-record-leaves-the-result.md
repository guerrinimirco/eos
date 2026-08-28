# The natural-units record leaves the public result, and its accessors take the fm names

Type: task
Status: open
Blocked by: [90](90-solver-signature-and-units-sweep.md)
Parent: ../map.md

## Question

Split out of [ticket 90](90-solver-signature-and-units-sweep.md) — its Work
item 5 in full, plus the part of item 3 that turned out to be the same act.
Ticket 90's gate is "**no value moves anywhere**"; this half moves and deletes
thousands of frozen keys, so it leaves rather than the gate being weakened.
That is the third time this ruling has needed the treatment: section 1 became
[89](89-dd2-honours-species-flags.md), section 4 became
[94](94-zl-solver-flags.md)/[95](95-vmit-solver-flags.md)/[96](96-alphabag-solver-flags.md),
and this is section 5.

**The §5 violation, as ticket 81 measured it.** `njl` and `ccdm` carry their
natural-units matter record on the public result as `.state`; `enjl` carries
its `EoSPoint` as `BetaPoint.point`. `n_B` on those records divides out to
exactly `hc3`, but **P, eps and s do not**, because the record is MATTER ONLY:
`njl .state.P / hc3 = 146.854334` against an outer `146.939710`. A caller who
spots the unit problem and corrects by `hc3` still gets a wrong answer,
silently, by 0.085376 MeV/fm^3. A rename cannot fix that; only removing the
record from the public surface can, which is why ticket 81 ruled for removal.

**Why item 3's remaining half belongs here.** Ticket 90 §3 renames the `_fm`
family — `n_C_fm`, `n_S_fm`, `eps_fm`, `P_fm`, `s_fm`. Every one of those is a
property ON these same records, and the natural-units field it converts holds
the bare name, so the rename is a swap: the field takes `_nat`, the accessor
takes the bare name. Measured against the frozen files:

    enjl.npz   21278 keys, 14976 nested under `.point`, 3042 of them the six
    njl.npz     6594 keys,  3255 nested under `.state`,  630 of them the six
    ccdm.npz    6068 keys,  3040 nested under `.state`,  456 of them the six

So the accessor rename alone changes what **4128 frozen keys mean** (the key
`state.P` holds a natural-units number today and an fm one after), and the
removal deletes **21271 keys** outright. Neither is a value that moved; both
are surface changes the baselines follow. Doing them together is what keeps
the result honest — the nested keys vanish, and no surviving key changes its
meaning. Done apart, the rename re-blesses 4128 keys under names that mean
something else, which is the one thing a regression baseline exists to catch.

Ticket 90 executed this on 2026-08-27, measured it, and **backed it out**;
`n_B_fm` -> `n_B` as a FUNCTION PARAMETER, and `BetaPoint.n_b_fm` -> `n_B` as
a result field, stayed there and are done.

## Work

1. `njl`/`ccdm`: `EoSPoint.state` -> `_state`, which the baseline flattener
   already skips (`generate_baseline.py:163` walks `vars()` and drops names
   beginning with `_`). `enjl`: `BetaPoint.point` -> `_point`, which also ends
   `result.point.point` at the public boundary.
2. Lift onto the outer point, in fm, what callers legitimately need: the
   colour densities `n_3`, `n_8` and the quark density `n_q` (13 sites across
   njl/ccdm). `euler_residual()` is dimensionless and stays reachable through
   the now explicitly internal `_state`, as ticket 81 ruled.
3. The one site that wants the matter-only pressure
   (`test/njl/test_equilibrium_modes.py:71`, charged against neutral) reads it
   through `_state` and says why in a comment — it is asserting a property of
   the matter block, not of the point.
4. `NJLState` / `CCDMState` / enjl's `EoSPoint`: the six natural-units fields
   take `_nat`, the six accessors take the bare names, and the dataclass
   docstring states the convention (bare = fm, `_nat` = natural, MeV
   quantities unsuffixed). This is `dd2/solver.py:159`'s existing `_nat`
   convention, applied where the ticket-90 sweep found the other half of it.
5. **The trap this sweep already sprang once.** Renaming a natural-units local
   out from under a function body silently changes units in whatever the sed
   did not reach. In `enjl/solver.py::default_guess` two seed expressions
   (`0.9 * n_B` and `0.5 * Y_Le * n_B`) were natural and became fm; the model
   still converged everywhere except `n_B = 5 fm^-3`, where it stopped at a
   scaled residual of 2.259e-09 against a 1e-10 bound —
   `test/enjl/test_enjl_beta_equilibrium.py::test_high_density_needs_a_widened_box`
   caught it and nothing else would have. Rename the FIELDS first, run, fix
   every AttributeError, and only then rename the accessors: that way every
   stale site raises instead of silently reading the other unit system.

## Gate

- **Regenerate `njl.npz`, `ccdm.npz` and `enjl.npz`, and measure first.** The
  removed key set must be **exactly** the nested `.state.*` / `.point.*` keys
  (3255, 3040 and 14976 respectively). Every surviving key BIT-identical —
  not rtol = 1e-10, bit-identical, because nothing in this ticket is arithmetic.
- No other `test/baseline/*.npz` moves at rtol = 1e-10, `mixed.npz` ASSERTED —
  `eos/mixed/adapters.py` reads the njl and ccdm records through exactly these
  accessors at `:1124` and `:1283`.
- Full suite green (§12); `njl`, `ccdm`, `enjl` and `mixed` `verify/` green.
- `notebooks/quark_eos.py` displays the internal record deliberately, as the
  model's stepwise walkthrough (~20 sites). It follows the record to `_state`
  and keeps working; its `.ipynb` pair moves with it.

## Finding this ticket owes the Stage 7 report

Deleting 21271 frozen keys is a real loss of regression coverage, and §12
calls `test/baseline/` ground truth. It is the price of the §5 fix and the
ticket takes it deliberately — but whether the internal records deserve a
baseline of their own, frozen through the internal path, is a question this
ticket raises and does not answer.

---

## Note from [ticket 91](91-leptons-default-and-drift-checks.md) (2026-08-28)

**This defect is now pinned by a test, and the pin is `strict`.** Ticket 91's
third check — the fm-based units band, parametrised over all ten models —
walks every float on `eos_point`'s result and asserts the energy and density
families sit in their fm bands. It finds exactly this ticket's three, and
exactly on the records this ticket removes:

    eos.njl.state.n_q        2.305052e+07     (fm^-3 band is |x| <= 1e3)
    eos.ccdm.state.n_q       2.305052e+07
    eos.enjl.point.n_s.s    -1.590857e+07

so the check reproduces ticket 81's finding from the outside, without knowing
which field to look at. The three are carried as
`pytest.mark.xfail(strict=True)` with this ticket's number as the reason.

**`strict=True` is the part that matters here.** A strict xfail FAILS when it
starts passing, so the day work item 1 lands — `.state` -> `_state`,
`BetaPoint.point` -> `_point` — `test/test_imports.py` goes red until the
three entries are deleted from `NATURAL_UNIT_RECORD`. The exemption cannot
outlive the defect, which is what an ordinary skip would have allowed.

Ticket 91's own gate did not include fixing this; the check was written to
detect and it does.
