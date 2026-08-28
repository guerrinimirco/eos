# `leptons=False` by default, and the checks that hold the line

Type: task
Status: resolved
Blocked by: -
Parent: ../map.md

## Question

Ruled by [ticket 81](81-second-default-solver-kwargs.md), sections 3 and 6.
Split out because it moves no baseline row and needs no regeneration — the
gate is a green suite and three new checks.

§3 names `leptons` as the orthogonal flag of `fixed_YC` and `fixed_YC_YS` but
never states its default, and nine models disagree: `False` in
dd2/sfho/did/alphabag, `True` in enjl/njl/ccdm/zl/vmit and in
`eos/mixed/solver.py:716`.

## Work

1. `leptons=False` everywhere it appears as a default. Ticket 65's rule is "off
   unless asked", and `leptons=True` is the one direction that silently ADDS a
   sector.
2. The legacy `TableSettings` layer, a THIRD place the same sectors carry a
   default: `zl/table.py:238-239` (`include_photons=True`,
   `include_leptons=True`) and `alphabag/table.py:283-286`
   (`include_photons`, `include_gluons`, `include_thermal_neutrinos` all True,
   `include_electrons=False`). Follow the flags object, as
   `zl/table.py:277` already does for `photons`.
3. Three checks in `test/test_imports.py`, beside
   `test_the_six_species_flags_all_default_to_off` — the precedent ticket 65
   set and this ticket's ruling names:
   - `leptons=` defaults to False in every model that has the argument.
   - No solver accepts BOTH a `SpeciesFlags` and a parallel `include_*` kwarg
     for a sector the flags object carries. This is what ticket 89 fixed in
     dd2 and ticket 90 removed from zl/vmit/alphabag; without the check
     nothing stops it coming back.
   - **The units band check**, parametrised over all ten models: call
     `eos_point`, walk every float field on the result, assert P and eps sit
     in an fm-plausible band and densities (including `n_s` and `s`) in
     theirs. Write it against MAGNITUDE BANDS, not values — a natural-unit
     field is off by `hc3 = 7.68e6`, six orders of magnitude, so the check
     needs no tolerance and can never become a second baseline to maintain.
     This is the check that found `njl`, `ccdm` and `enjl` in ticket 81.

## Gate

- No `test/baseline/*.npz` moves. All 16 `leptons=`/`include_electrons=` sites
  in `generate_baseline.py` pass the flag explicitly, so the default is not
  load-bearing for any frozen row. **If a row moves, the ruling was applied
  somewhere it was not meant to reach.**
- The band check must FAIL on `njl`/`ccdm`/`enjl` before ticket 90 lands and
  pass after — run it against `main` first to confirm it detects, rather than
  shipping a check that passes because it looks at nothing.
- Full suite green (§12).

## Resolution

Executed on python.org 3.14.2 (numpy 2.3.5, scipy 1.17.0), last of the flags
lane and after [94](94-zl-solver-flags.md), [95](95-vmit-solver-flags.md) and
[96](96-alphabag-solver-flags.md), so its three checks assert what those three
built rather than passing for the wrong reason.

### `leptons` is False everywhere — but the default is spelled `None`

The correction this ticket's work item 1 needed, and it is not cosmetic.
`resolve_leptons` **RAISES** on an explicit `leptons=False` in a
beta-equilibrium mode (§3: "beta equilibrium without the particles that define
it"). So `leptons=False` as the signature default of a mode-generic entry
point breaks every beta call in the model — which is exactly why the five
models that defaulted it on had spelled it `True`: on a beta mode `True` is
"redundantly made, accepted and ignored", and `False` is not.

The four models that were already off had the shape right all along:

    dd2, sfho, did, alphabag    eos_point(..., leptons=None, ...)
                               leptons = resolve_leptons(mode, leptons,
                                                         default=False)

`None` means "the caller did not name the flag", which is the only value that
can be resolved per mode. The flip therefore lands in two places, not one:
**`None` on every mode-generic surface** (the `eos_point`/`eos_table`/
`eos_response` entry points, the `TableSpec.leptons` fields, `mode_spec`,
`make_charge_spec`, the `solve_at` dispatchers) and **`False` on every
fixed-fraction-only surface** (`solve_fixed_yc`, `solve_fixed_yc_ys` in six
models, `general.modes.fixed_YC` and `fixed_YC_YS`), with every
`resolve_leptons(..., default=True)` becoming `default=False`. Nine models and
`eos/mixed` now agree.

**`ModeSpec.leptons` stays `True`, and that is not an exception to the rule.**
It is not a default a caller inherits: it is the spec's own statement about
the state it names, and `ModeSpec.__post_init__` already refuses `False`
wherever C is not FIXED — so flipping the field made `ModeSpec()`, the plain
beta-equilibrium spec, raise on construction. The caller meets the flag at
`fixed_YC(Y_C, leptons=...)`, and that is where it defaults False. A comment
on the field says so, because the next reader will try the same flip.

`eos/general/modes.py`'s `resolve_leptons` docstring loses the paragraph that
justified the divergence ("it differs between models on purpose") and states
the uniform rule and its reason instead.

### `eos/mixed/responses.py` was included; `standard_name` was not

Two judgement calls, both recorded rather than assumed:

- **Included.** `mixed/responses.py`'s four `leptons=True` defaults are §5's
  THIRD response-conditioning axis ("whether leptons re-neutralize against the
  held charge"), not §3's mode flag. They were flipped anyway: `eos/mixed/api.py`
  passes the value explicitly on every public path, so the change is inert
  there, and one uniform answer to "what does an unnamed `leptons` mean" is
  worth more than a second convention living in one file.
- **Excluded.** `eos/general/table_io.py`'s `standard_name(..., leptons=True)`
  labels a FILE. It moves no number, its argument describes the table the
  caller already built, and `test/general/test_table_io_names.py` pins the
  names it produces. Flipping it would rename files to say something about the
  default rather than about the table.
- **Excluded.** `eos/astro/gmode/verify/run_full_check.py`'s `dd2_frozen_cs2(...,
  leptons=True)` carries a documented physical reason in its own docstring
  ("the physically right choice for stellar matter"), which makes it a
  statement rather than an inherited default.

Also excluded, and for a sharper reason: the `leptons=` parameters of
`default_guess`, `warm_start` and `two_flavour_state` in `zl` and `vmit`. Those
say whether the unknown vector has a `mu_e` slot — a statement about the
residual's SHAPE, not about a sector — and the beta solvers call them without
naming it. Flipping those would not have changed a default, it would have
broken the solve.

### Work item 2: the legacy `TableSettings` layer, in four models not two

The ticket named `zl/table.py` and `alphabag/table.py`. The same layer exists in
`eos/vmit/compute_tables.py` and `eos/sfho/table.py`, carrying the same
sectors with the same `True`, and "a THIRD place the same sectors carry a
default" is a statement about the layer rather than about two models. All four
now FOLLOW the flags object:

    zl/table.py            include_photons, include_leptons          -> False
    vmit/compute_tables.py include_photons, include_leptons          -> False
    sfho/table.py          include_photons                           -> False
    alphabag/table.py      photons, gluons, electrons, thermal nu    -> False
                                                       (under ticket 96)

`docs/DEFERRED.md`'s alphaBag paragraph is rewritten around it.

### The three checks

In `test/test_imports.py`, beside `test_every_species_flag_defaults_off_or_raises`.

1. **`test_leptons_defaults_to_off_in_every_model_that_takes_it`** — BEHAVIOURAL,
   deliberately. The number that matters is often the `default=` argument of
   `resolve_leptons`, which no signature carries and `inspect` cannot see. The
   check compares the unnamed `fixed_YC` call against the two named ones:
   `P(unnamed) == P(leptons=False)` and `P(unnamed) != P(leptons=True)`. The
   second assertion is what keeps it from passing vacuously, which is also why
   it is asked at `Y_C = 0.3` rather than 0 — at zero non-leptonic charge there
   is nothing to neutralize and `leptons=True` adds an empty gas.
2. **`test_no_solver_takes_both_a_flags_object_and_a_parallel_sector_kwarg`** —
   static, over `solver`/`table`/`api`/`responses` in all ten models. A
   function whose signature contains `flags` or `species` may not also carry an
   `include_*` parameter or one named after a `SpeciesFlags` field. It fires
   only where a flags object is present, which is what exempts
   `eos.dd2.solver.solve_beta_eq` (the nucleon-only seed solve: `include_photons`
   there is the only switch, not a second one) and the layout helpers above,
   by the test rather than by name.
3. **`test_every_result_field_is_in_its_fm_based_band`** — parametrised over all
   ten models, walking every float on the result including nested blocks and
   mappings. Bands, not values: `hc^3 = 7.68e6`, so `|P|, |eps| <= 1e5
   MeV/fm^3`, `eps >= 1 MeV/fm^3`, `|n_i|, |s| <= 1e3 fm^-3` leave five decades
   of margin, need no tolerance, and cannot become a second baseline.

### The band check finds three models, and they are ticket 97's

Ticket 91's gate says the check must FAIL on njl/ccdm/enjl before the units
work lands and pass after. **The units work has not landed** — ticket 90 backed
it out and it is now [ticket 97](97-natural-record-leaves-the-result.md), still
open. So the check finds them, from the outside and without being told where to
look:

    eos.njl.state.n_q        2.305052e+07 fm^-3
    eos.ccdm.state.n_q       2.305052e+07
    eos.enjl.point.n_s.s    -1.590857e+07

which are precisely the natural-units records ticket 97 removes. The three are
carried as `pytest.mark.xfail(strict=True)` naming that ticket. **Strict** is
the point: a strict xfail FAILS when it starts passing, so the day 97 lands
this file goes red until the entries are deleted. The exemption cannot outlive
the defect.

### The gate: the checks were proved able to fail

A check nobody has seen fail is a check nobody has tested. Each was run against
a deliberately broken input and each detected it:

    leptons default   eos.vmit.eos_point wrapped to default leptons=True
                      -> "defaults leptons to ON: fixed_YC without the flag
                          gives P = 385.589 ..., with leptons=False 354.819"
    parallel kwarg    a function `(par, n_B, T, flags, include_photons=True)`
                      injected into eos.vmit.solver
                      -> "takes a flags object AND ['include_photons']"
    fm-based band     eos.vmit.eos_point wrapped to multiply P and eps by hc^3
                      -> "P_total = 2.683629e+09, which is not MeV/fm^3"

Script kept at the session scratchpad; it restores every patch it makes.

### Six call sites relied on the flipped default, and one of them was a baseline

The ticket's gate says "All 16 `leptons=`/`include_electrons=` sites in
`generate_baseline.py` pass the flag explicitly, so the default is not
load-bearing for any frozen row. **If a row moves, the ruling was applied
somewhere it was not meant to reach.**" A row moved, and the ruling was applied
exactly where it was meant to — the count was wrong. `case_vmit`'s
`fixed_YC_YS` call is a SEVENTEENTH site and names no flag:

    solve_fixed_yc_ys(par, n_B, 0.0, 1.0, 10.0, N)      # leptons: inherited

so the three `ycys.*` points moved by one electron gas (`P` by 4.992e-04,
`e` by 1.498e-03, `s` by 1.997e-04 MeV/fm^3 in absolute terms) — and the
baseline check reported them as "a physics change", correctly. The fix is to
NAME the flag at its old value rather than regenerate: `leptons=True` there
restores every frozen number and makes the call say what it does.
**`test/baseline/*.npz` is byte-identical, all fourteen, before and after this
ticket.**

The other five were the same shape, in tests and `verify/` suites of the models
whose default flipped:

    eos/zl/verify     `_states`'s "yc" state, and the mode-closure neutrality
                      check -- the FAIL was `mode closures max_err=3.00e-01`,
                      which is n_p - n_e at Y_C = 0.3 with no electrons
    eos/vmit/verify   the "yc" and "ycys" states, and their two closure checks
    eos/njl/verify    the "yc_ys" state
    test/general/test_modes.py   `fixed_YC_YS(0.5, 0.0)` asserting that
                      `neutrality` is among the charge conditions -- which is
                      the leptons' condition
    test/vmit/test_uniform_api.py  a seven-entry warm start, which is the
                      LEPTONIC unknown layout; the leptonless vector has six,
                      so the solve raised rather than failing an assertion

Every one is fixed by naming `leptons=True`, which is what those sites meant.
That is the flip working as intended: a default that was adding a sector
silently now has to be asked for, and the five places that were relying on it
say so.

The five verify/test failures and the one baseline move are the whole of what
the flip disturbed across the suite.

### Gate

    interpreter   python.org 3.14.2 / numpy 2.3.5 / scipy 1.17.0

    full suite    1826 passed, 23 skipped, 3 xfailed, 0 FAILED  (23:12)
    baselines     all 14 .npz BYTE-identical, before and after; nothing
                  regenerated, which is what this ticket promised

**The +13 against ticket 94's 1816 is accounted for exactly**, which matters
because a concurrent session held the tree during 94's run and its denominator
is not a clean control:

    +10   test_every_result_field_is_in_its_fm_based_band, one per model
     +1   test_leptons_defaults_to_off_in_every_model_that_takes_it
     +1   test_no_solver_takes_both_a_flags_object_and_a_parallel_sector_kwarg
     +1   test_the_photon_flag_reaches_the_solver (ticket 95, test/vmit)
     -3   the njl/ccdm/enjl band entries, which xfail rather than pass

1816 + 13 - 3 = 1826, and 1826 + 23 + 3 = 1852 collected. Zero failures on
both sides of the comparison, so "zero added failures" is unambiguous here in
a way it was not for 94.

This session ran [95](95-vmit-solver-flags.md), [96](96-alphabag-solver-flags.md)
and this ticket back to back at the user's request; the full suite was run once,
at the end, over all three. The per-ticket subsets were run and recorded
separately in each resolution.
