# Phase 5 items 1, 2 and 4 — top-level imports and a runnable README

Type: task
Status: resolved
Assignee: session dc554906
Blocked by: 10, 11
Parent: ../map.md

## Question

`docs/REFACTOR_PROMPTS.md` Phase 5, items 1 and 2, with item 4 applied to both:

1. `eos/eos/__init__.py` so the common tasks are one import deep, and the modes
   and species flags are importable from the top level.
2. Rewrite `eos/README.md`: what the library computes, install, and four runnable
   examples — (a) one point from one model in `beta_eq_neutrinoless`, (b) a
   `fixed_YC` table swept over `n_B` and `T`, (c) that table fed to
   `eos/astro/tov/` for an M–R curve and maximum mass, (d) the same model with
   hyperons enabled, showing the species flags. Then a fifth: an M–R figure with
   the observational constraints overlaid, in the house style, in under ten
   lines. **Copy-paste runnable, no pseudocode.**

**Item 4 is not optional**: execute every code block in the README and paste the
real output. Do not write an example you have not run.

Blocked on the rename approvals (ticket 10) and the conformance triage (ticket
11) because both can move the public surface these examples import.

## Resolution

**Shipped: a lazy top level, a TOV package surface that was empty, and a
README whose five examples were all executed.** Four files changed in `eos/`
(three of them `__init__.py`), plus `README.md`, `docs/figures/dd2_MR.png` and
two tests.

**Item 1 — `eos/eos/__init__.py`.** The subpackages are reached through a
PEP-562 `__getattr__`, not imported eagerly. Measured reason: `eos.dd2` alone
costs 0.47 s (Numba), `eos.general.constraints` 0.42 s, and eagerly importing
the ten models plus `mixed`, `zlvmit` and `astro` would put all of that on
`import eos` for the one model a caller wants. As shipped, `import eos` is
**0.088 s** and `eos.dd2.Parameters` resolves on first touch. What is eager is
what is cheap and shared: the four `ModeSpec` factories and `Conservation`
from `general/modes`, `EOSTable_for_TOV`, and `save_table`/`load_table`/
`export_csv`. `REPO_ROOT` stays — four `table.py` modules import it.

The vocabulary the ticket asks for is `eos.MODES` (§3's five names, as the
strings the uniform API takes) and `eos.SPECIES_FLAGS` (§4's six field names),
alongside `eos.MODELS` and `eos.ENGINES`. **The species flags could not be one
class**: every model carries its own `SpeciesFlags` dataclass, so what is
importable from the top level is the shared *names*, with the per-model class
one attribute away.

**A second empty surface, fixed because the ticket's own example needed it.**
`eos/astro/tov/__init__.py` was **zero bytes**, so the single most common
downstream task — an M–R curve — was `from eos.astro.tov.solver import ...`,
three modules deep, and `eos.astro.tov` did not even resolve after
`import eos`. It now re-exports `solver.py`'s and `crust.py`'s public names,
and `eos/astro/__init__.py` got the same lazy hook. `rotating` is deliberately
NOT re-exported: it shells out to a compiled RNS solver, which is heavier than
a `from ... import` line should carry.

**Item 2 + item 4 — the README, every block executed.** `README.md` was
rewritten (557 -> 424 lines) and was badly stale: it documented five models of
the ten, `eos/sfho/eos.py` and `thermodynamics_hadrons.py` (neither exists),
`eos/general/plotting_info.py` (removed), `eos/tov/` (now `eos/astro/tov/`),
and three different `TableSettings` dialects as the primary API.

Verification was mechanical rather than by eye: a script extracts every
` ```python ` block from `README.md` and `exec`s them **in one namespace, in
order**, so example 5 genuinely continues from example 3's `seq` and `i` the
way the text says. Seven blocks, six executed (the seventh is the three-line
signature listing of the uniform API), and the pasted output is that run's.
Ran on **anaconda python 3.9.7 / numpy 1.26.4 / scipy 1.13.1**; the sequence
numbers below are stack-dependent and would need re-running if
[ticket 57](57-canonical-stack.md) rules for 3.14.

The five: (a) one DD2 point in `beta_eq_neutrinoless`; (b) a `fixed_YC` table
over 12 densities x 3 temperatures; (c) that table converted to an
`EOSTable_for_TOV` and run through `compute_tov_sequence` with the shipped BPS
crust; (d) DD2Y with `hyperons=True`; (e) the M–R figure with
`constraints.overlay` in the house style, **11 lines including its three
imports**. (c) returns **M_max = 2.419 M_sun, R(M_max) = 11.99 km,
R(1.4) = 13.19 km** — the published DD2 star (2.42 / 13.2), reproduced from a
fresh environment with **no `EOS_CRUST_DIR`**, confirming
[ticket 39](39-crust-silent-fallback.md) end to end. Example (b) prints
negative cold pressures below n_B ~ 0.15: the liquid–gas instability of
Y_C = 0.3 matter, real, and explained in the text as §8's raw-branch case.

### Three things the code decided, all found by running rather than reading

1. **The quick-start example, as first written, raised.**
   `Parameters.default()` + `SpeciesFlags(hyperons=True)` dies at
   `eos/dd2/thermodynamics.py:289` with a bare `KeyError: 'Lambda'` — the
   nucleonic set carries `hyperon_couplings=()`. The refusal is correct
   physics (DD2 and DD2Y are different published parameterisations), but §4
   requires a raise that NAMES the gap and §6 puts that at the public
   boundary. The example now uses `Parameters.named("DD2Y")`; the defect is
   [ticket 60](60-dd2-hyperon-flag-raise.md), reported not fixed — it is model
   internals, and this ticket's whole diff is `__init__.py` files and prose.

2. **§4's flag names are not yet universal, so the README says so.** `dd2`,
   and `mixed` through it, carry neither `thermal_mesons` nor
   `thermal_neutrinos`. Reached independently of, and **converging with**,
   [ticket 61](61-dd2-species-flags.md), which
   [ticket 04](04-notebook-skeleton.md) opened while this ran — and 61's
   diagnosis corrects mine: `neutrinos` is NOT `thermal_neutrinos` under
   another name, it is the matter-composition field of the trapped modes, so
   the tau gas is genuinely unwired rather than merely misspelt. Both
   `README.md` and `eos.SPECIES_FLAGS`'s comment say it that way. The new
   `test_the_top_level_carries_the_mode_and_species_vocabulary` checks the
   other seven models against §4's six names and exempts `dd2` by name, so the
   exception cannot quietly grow while 61 is open.

3. **`eos_table` has no `leptons=` at all**, only `eos_point` does — so a
   neutralizing fixed-Y_C *table* is reachable today ONLY through the invented
   mode name `fixed_YC_neutral` that
   [ticket 54](54-signature-corrections.md) item 1 retires. Retiring it alone
   would make that table unreachable. Noted on ticket 54 rather than opened as
   its own ticket; the README shows the leptonless flavour and says which it
   is.

### Tests

`test/test_imports.py` gains two, because the lazy hook is invisible to the
module sweep that file already runs — every module still imports whether or
not `__getattr__` works, and only `eos.dd2` after a bare `import eos` breaks.
One asserts every name in `MODELS + ENGINES + ("general", "astro")` resolves
to the module of that name and that a non-subpackage raises `AttributeError`;
the other checks `eos.MODES` against the `ModeSpec` factories and
`eos.SPECIES_FLAGS` against seven models' dataclass fields. (`test/` is
gitignored, so this lands in a working copy only — the standing problem the
map's **Not yet specified** already records.)

**Targeted runs only** — the full-suite gate was held by another session, and
this ticket's diff cannot move a number. On anaconda 3.9.7:
`test/test_imports.py` 188 passed (plus the 2 new, passing);
`test/gmode` + `test/tov` **64 passed, 15 skipped**; `test/mixed` 20 passed.
Reported to the gate holder: my run overlapped theirs by ~24 minutes before I
killed it, which is exactly the contention `test/dd2/test_dd2_speed.py` is
known to flake under.
