# Phase 6, second half — the conformance pass on `nucleation`

Type: task
Status: resolved
Assignee: session 01b6c287
Blocked by: 24
Parent: ../map.md

## Question

The second half of [ticket 23](23-phase6-respec.md)'s corrected brief, split off
because it has a different gate from the port: the port is measured in minutes
against a test suite, this is measured by a notebook that takes hours and by
judgement.

**In scope of the map's destination, but NOT gating
[Acceptance](25-acceptance.md)** — see the Destination note. Ticket 25 needs
`nucleation` to import, run and respect §1, all of which ticket 24 delivers.

Four items. Phase 6's original list had six; two of them measure clean and one
has no meaning for a consumer package.

1. **The paper notebook**, `notebooks/2fam_PNS_nucleation.{py,ipynb}`. The port
   is mechanical — 4 import lines, and `AlphaBagTableSettings` /
   `compute_alphabag_table` -> `eos.alphabag.table.TableSettings` /
   `compute_table`, whose fields match one for one. **Then execute it.** It is
   what the paper reproduces from, so this ticket is not finished until it runs.
   Its own header warns the runs write to `output/paper/` and take hours; plan
   for that rather than discovering it.
2. **The test move.** `nucleation/nucleation/tests/` -> top-level
   `nucleation/test/`, **tracked, NOT gitignored**. `eos` hides its suite because
   it is private; `nucleation` is headed public, and publishing the repository
   behind a paper with no runnable tests is a real cost the layout parity does
   not justify. `make_fixture` moves with them and is invoked as
   `python test/make_fixture.py` — update the three docstring references in
   `conftest.py` and `make_fixture.py` that still say
   `python -m nucleation.tests.make_fixture`.
3. **`README.md`** to the standard of the new `eos` README, **with the examples
   actually run**.
4. **Dead code removed.**

### Measured clean — commission no sweep for these

- **Internal layering.** Already acyclic and layered: `barrier.py` at the bottom
  importing nothing internal, then `composition`/`critical`/`rates`/`tables`,
  then `analysis/`, then `analysis/figure/`. `eos`'s `general/` rule has no
  analogue to impose; it is satisfied in substance.
- **The docstring standard.** Nothing across 8,121 lines: no Phase/Stage/
  milestone reference, no TODO, no FIXME.

### Dropped

**"Apply the same API conventions."** `nucleation` is a consumer, not a model.
§5's uniform API is a contract for models, and imposing `eos_point`-shaped
signatures on a nucleation-rate sampler would be conformance theatre.

### Does not transfer from `eos`

**`nucleation`'s `output/` rule stands unchanged.** `.gitignore:32-38` already
ignores everything under `output/` except
`output/paper/{figures,figure_data,tables}` — 87 tracked files, the paper's own
figures and tables. That is §11's `output/public/` principle already correctly
specialised; flattening it to `eos`'s rule would untrack the paper's figures.
Do not "fix" it.

### Gate

The notebook executes to completion. `pytest` for `nucleation` still green at the
new paths, reported verbatim with interpreter and collected count. Then push to
`origin/paper-release`.

## Answer — all four items landed, pushed as `2b2b72f`

Gate met. The notebook executes to completion in PRODUCTION mode (39/39 code
cells, zero error outputs, `EXIT=0`), the suite is unchanged at the new paths,
and `origin/paper-release` is at `2b2b72f`.

    Python 3.14.2 / numpy 2.3.5 / scipy 1.17.0
    $ pytest test -rf
    collected 72 items
    FAILED test/test_composition.py::test_regression_solver_cases
    FAILED test/test_critical.py::test_energy_barrier_matches_golden
    2 failed, 70 passed in 4.80s

Identical to ticket 24's baseline, node id for node id modulo the path prefix,
so the move and the dead-code pass added nothing. Both survivors are
[ticket 76](76-nucleation-golden-tolerances.md)'s. No tolerance was touched.
Transcript: `output/_audit/nucleation_after_ticket72_py314.txt`.

### The premise that broke, and the isolated copy it forced

The first production run died on `ImportError: cannot import name
'create_custom_parametrization' from 'eos.sfho.nmp'`. **Not the port.** A
concurrent session holds `eos/sfho/{__init__,nmp,parameters,solver,table,
verify/run_full_check}.py` dirty and is mid-rename of that function to
`from_potential_depths`. It still exists at HEAD (`nmp.py:233`), which is what
this ticket measured against and ported to.

Every number here was therefore taken against an ISOLATED eos built with
`git archive HEAD`, per the map's concurrency note. The suite gives the same
2 failed / 70 passed against both the dirty tree and the isolated copy, which
is what proves the two failures are eos-intrinsic rather than the neighbour's.

**This is a live hazard for whoever lands that sfho rename**: `test/make_fixture.py:98`
imports `create_custom_parametrization`, and so does the paper notebook. Neither
is on the suite's import path (make_fixture imports it lazily inside `main`),
so the rename will land green and break both silently.

### 1. The notebook — mechanical, but five statements and two aliases

The brief said "4 import lines". It is five statements, and two of them need an
ALIAS rather than a rename, because the port creates two genuine collisions:

- `eos.sfho.table` and `eos.alphabag.table` BOTH export `TableSettings` and
  `compute_table`. One namespace cannot hold both, so the alphaBag pair is
  imported `as AlphaBagTableSettings` / `as compute_alphabag_table` — which
  keeps the two call sites saying which phase they build.
- `custom_params` is both the `nucleation.quark` constructor AND an sfho
  `TableSettings` field, used forty lines apart. Imported `as quark_params`;
  the ten `get_alphabag_custom` call sites take that name.

`EOSTable_for_TOV` comes from `eos.general.state`, not through
`eos.astro.tov.solver` which re-exports it — same choice ticket 24 made, and
for the reason the brief gave: it is the one target that changed LAYER.

**The masked break did NOT reach the notebook.** Ticket 24's `Y_L -> Y_Le` /
`mu_nu -> mu_nue` rename hits reads of eos-produced structures; the notebook's
`GRIDS['Y_L']` and `Y_L_values=` are its own key and a still-current sfho field
respectively. Checked, not assumed.

**The cached tables load.** The four hadronic `.dat` carry 18 columns, matching
`COLUMN_MAPS['trapped_neutrinos']`: the rename moved names, not positions.

### 2. The test move

17 files by `git mv`, every rename detected. `__init__.py` dropped — eos's
`test/` has none outside a reference-data dir. Nothing in `.gitignore` covers
`test/`, so it is tracked by default; the refusal of eos's gitignore needed no
edit. `[tool.setuptools.packages.find] include = ["nucleation*"]` already
excludes it, so packaging needed none either.

**The invocation references were SIX, not three.** `conftest.py` x3 and
`make_fixture.py` x1 as the brief said, plus `README.md:257` and
`docs/reproducing.md:193`. All now `python test/make_fixture.py`, and that
invocation was RUN, not merely rewritten.

### 3. README

Rewritten against the eos README's contract: examples runnable from a fresh
clone, real output beneath each, named stack. The three Quick-start blocks were
extracted back OUT of the finished README and executed; the pasted output
matches byte for byte. They read the committed 247 KB fixture, so they need no
table generation — the old block 1 opened on a `'...trapped....dat'`
placeholder and could never have run.

Two corrections the rewrite had to make rather than restate:

- **`make_fixture.py` no longer reproduces the committed fixture.** Verifying
  the invocation rewrote all 937 rows against eos HEAD — the same ~1 ulp basis
  change behind ticket 76. The committed file was restored bit-for-bit (sha
  `2b3224a0…`, suite back to 2/70) and the hazard is now documented where the
  "regenerable" claim lives. A golden's INPUT is ground truth too.
- **`docs/reproducing.md:16` told the reader to stop unless the suite is
  green.** It has not been green since the port. Corrected to name the expected
  `2 failed, 70 passed` and point at the explanation.

### 4. Dead code

pyflakes over `nucleation` + `test`: **24 findings -> 4**, and the four
survivors are all load-bearing and documented (`crossover_radius` and
`_HAVE_JOBLIB` are deliberate re-exports asserted by `test_imports.py:34-35`
and read by the notebook at line 151; two `joblib` presence probes).

The bulk was **fifteen private re-exports in `tables/__init__.py`** — traced
name by name, every consumer imports them from the submodule that defines them,
so nothing reached them through that door. Its comment claimed
"`critical.py` … import them by these exact names"; `critical.py` imports none
of them, and `grid.py:11` carried the same false claim about `_BASE_DATA_KEYS`.
Both corrected with the code they justified. Then six unused imports and one
unused local (`test_conditions.py:31` — the CALL is the check, so the binding
went and the call stayed, with a comment saying so).

`analysis/__init__.py`'s `_HAVE_JOBLIB` was NOT removed despite being flagged:
the notebook prints it.

### The smoke leg failed, and it is not this ticket's

The user asked for smoke-then-production. Smoke reached Figure 5 and raised
`ValueError: No objects to concatenate`.

**Pre-existing, smoke-only, and untouched by this diff.** Smoke mode scans a
single `alpha_s`, so `F8_SHOW = [1, 3]` clips at line 1935 to `[]`, and
`pd.concat([])` raises. The guard protects the INDEXING but not the empty
result — its own comment ("Clip rather than assert: fewer panels is still a
valid figure") did not anticipate clipping to zero. `WT_SHOW = [0, 2]` at 2870
clips to `[0]` and survives, so Figure 5 is the only casualty. Production's four
slices are unaffected, which is why the gate passed.

Per the map's hard rule this is Stage 7 material, not a diff. Flagged because
README and `docs/reproducing.md` both tell a reader to smoke-run FIRST, so the
path they recommend is broken as advertised.

### What the production run regenerated, and why none of it is committed

23 tracked files under `output/paper/` moved. Classified by byte diff before
anything was staged:

- **14 PDFs: timestamp only.** Identical once `/CreationDate (D:…)` is
  stripped. Exactly the brief's warning — 7 bytes each, the date digits. (Note
  for a future session: the stamp is `/CreationDate (D:…)` WITH a space; a
  regex for `/CreationDate\(` finds nothing and wrongly reports a real change.)
- **9 with real content: round-off, and the paper's own quantity did not
  move.** Every CSV keeps its shape and columns. Six of seven sit at 1e-15 to
  1e-11 relative. The outlier is `supp_viable_region_stars`: `R_1.4_km` max
  1.32e-04 km — **13 cm on 11.14 km** — and `M_max` max 2.97e-05 M_sun, both
  round-off amplified through TOV sequence interpolation. `sigma_crit_star` is
  **bit-identical across all 398 rows**.

All 23 were restored to HEAD. Regenerating the paper's committed figure data is
not one of this ticket's four items, and it is a publication decision rather
than a side effect of a conformance pass. **Open for the user**: the tracked
figures are now the pre-refactor ones, so they no longer match what the code
produces — invisibly so, but the choice is theirs, and it is one command.

### Confirmed still correct, not changed

`nucleation`'s `output/` gitignore stands (87 tracked files intact). The
README's `git clone github.com/guerrinimirco/eos.git` resolves — that remote
exists.

Status: resolved.
