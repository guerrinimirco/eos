# Map: eos next phases — notebooks, documents, conformance, Phases 5–6

Label: `wayfinder:map`
Effort: `next-phases`
Source: [docs/NEXT_PHASES_PROMPT.md](../../docs/NEXT_PHASES_PROMPT.md)

## Destination

`eos` on `main` and `nucleation` on `paper-release` both satisfy the Acceptance
criteria block of `docs/REFACTOR_PROMPTS.md`: **four** grouped usage notebooks
exist and execute end to end — the prompt's three plus `hybrid_eos`, added by
[ticket 05](issues/05-notebook-coverage.md) as this map's own addition — every
per-model document passes CLAUDE.md §11's test (a physicist reproduces the model
without opening the source), CLAUDE.md describes the repository as it actually
is, and Phase 5 and Phase 6 are done.

Reached when every ticket here is resolved and the Stage 7 report can be written
with real tool output behind every claim.

## Notes

**Domain.** Nuclear/quark-matter equation-of-state library. `CLAUDE.md` at the
repo root is the specification and overrides defaults; `docs/REFACTOR_PLAN.md`,
`docs/REFACTOR_PROMPTS.md` and `docs/DEFERRED.md` record what the refactor
settled and deferred.

**Order of work: conformance before notebooks — but only the code half.**
Settled after the audits. [Ticket 10](issues/10-rename-approvals.md) proposes
renaming `Parametrization`, `solve_octet` and `from_dd2_defaults`, all three of
which a notebook would import, so notebooks written first get rewritten; nothing
in the conformance work depends on the notebooks, so the risk runs one way only.
The **document** tickets (30–36) are NOT in that precedence: no notebook imports
a `.tex`, so they run in parallel with anything.

So: renames (10) and the code triage (11) → notebooks (12–19) → Phase 5/6, with
documents alongside throughout.

**This map carries execution, not only decisions.** Wayfinder's plan-don't-do
default is deliberately overridden here: tickets 12–25 build and run things.

**Standing preferences for this effort** (settled while charting):

- Work lands **directly on `eos` `main`** and **directly on `nucleation`
  `paper-release`**. No feature branches.
- **Both gates are lifted.** A ticket may `git commit` its own work without
  asking, and may delete or overwrite files once the ticket authorising it is
  resolved. Git history is the undo.
- Where `docs/NEXT_PHASES_PROMPT.md` and `CLAUDE.md` disagree, **neither wins by
  default** — each conflict is its own ticket.
- `wayfinder:research` tickets may be resolved by parallel subagents. Every other
  type is worked inline.
- Phase 6 is executed against **corrected premises**, not the stale text.

**Hard rules, inherited from the prompt, binding on every ticket:**

- Only the changes a ticket asks for. A defect noticed and not asked about goes
  in the Stage 7 report, never in a diff.
- Golden references are ground truth (§12): `test/baseline/` at rtol = 1e-10, the
  DD2 golden SNM point and published NMP/TOV values, the CompOSE HS(DD2) slices,
  the ENJL author tables.
- **Never loosen a tolerance to make a test pass.**
- **No new dependencies.** stdlib, numpy/scipy/matplotlib, numba, jupytext.
  `cProfile`/`pstats`/`time.perf_counter` cover the timing work.
- Notebooks and benchmarks are written against the public API (`eos_point`,
  `eos_table`, `eos_response`, the `progress` callback), never by instrumenting
  solver internals. **Deep solver code never prints** (§5).
- Non-convergence is a return value, not an exception (§6).
- Every ticket reports failures **added** against `output/_audit/pytest_before.txt`
  and never silently fixes or deletes a pre-existing one.
- **Commit with explicit pathspecs. Never `git add -A`, never `git commit -a`.**
  The rule for concurrent sessions, and it earned itself: for part of this
  session a 24 GB `output_old/` sat untracked in the tree, because
  `.gitignore:37` ignores `output/` and a rename walked out from under it. The
  user has since removed the folder — the repo is 646 MB with 0 untracked files
  — so the hazard is gone and the rule stands on the concurrency alone.
- **An `output/_audit/` file names its interpreter in its FILENAME**, not only
  in prose: `_py39` (anaconda 3.9.7 / numpy 1.26.4 / scipy 1.13.1) or `_py314`
  (python.org 3.14.2 / numpy 2.3.5 / scipy 1.17.0). Per
  [ticket 47](issues/47-dd2-nmp-inversion.md) the stack IS the difference between
  0 and 14 failures, so a directory listing that cannot say which stack produced
  a before-image invites a future session to diff across the two and read the
  interpreter as a regression. **The nine files predating this rule are all
  `_py314`** and are not renamed; new ones carry the suffix.
- **Report the collected count with the failure count.** The denominator moves:
  ticket 20 added +2 tests to `test/test_imports.py`, so collection is **1665**,
  not the 1663 this map's earlier numbers imply.

**Skills every session should consult.** `grilling` and `domain-modeling` for the
decision tickets; `research` for the audit tickets; `prototype` for ticket 04.

**Facts established while charting.**

- `main` is the line: 64 commits ahead of `origin/enjl-finite-T`, 120 ahead of
  `origin/phase2-mixed`, and contains `5950f92`. Nothing to merge.
- `nucleation` lives at `/Users/mircoguerrini/Desktop/Research/Python_codes/nucleation`,
  branch `paper-release`, **and has a remote**:
  `github.com/guerrinimirco/metastability-nucleation`. 8.1k lines across 38 files;
  tests at `nucleation/nucleation/tests/`.
- All twelve model documents (`.md` + `.tex`) exist, plus `eos/astro/tov/tov.tex`.
- `eos/dd2/notebook_api.py` still exists (25.7 kB) and is imported only by
  `notebooks/DD2_usage.{py,ipynb}` — both of which Stage 0 removes.
- `eos/general/constraints/__init__.py:403` defines `overlay(ax, plane, ...)`.
- `docs/STRUCTURE.md` **now exists** (commit `f479845`,
  [ticket 21](issues/21-phase5-structure.md)); CLAUDE.md §10 and §11 both
  reference it, and §10's worked figure example lives in its §12.

## Suite status

**[Ticket 57](issues/57-canonical-stack.md) is RESOLVED, and the text below has
not caught up.** The ruling is **python.org 3.14 is canonical** and
`test/baseline/` regenerates on it, under two conditions; until that
regeneration lands, 57 itself restates the rule that every failure count names
its interpreter and its collected count. Read the blocks below as history.

**Concurrency is now the binding constraint on a full-suite number, not the
stack.** Tickets 60/61 met their gates while another session held eight
`eos/*/api.py` files and a new `test/test_nonconvergence_return.py` modified in
the working tree (ticket 49). A full-suite run in the repo therefore measures
BOTH sessions and has no before-image; the way through is an isolated copy built
with `git archive HEAD` plus one snapshot of the gitignored `test/`, run beside a
second copy carrying only HEAD. Two costs, both measured: an isolated copy has
**no numba cache**, so kernels recompile and `test/mixed` runs for hours; and it
produces **six `test/abpr` round-trip failures the real repo does not**, which
without the HEAD copy beside it read as the ticket's own regressions. Collection
in such a copy is **1677**, not 1665 — ticket 49's file adds 12.


**CORRECTED by [ticket 47](issues/47-dd2-nmp-inversion.md): the 14 are a
STACK artifact, not a code state.** This machine carries two Python stacks —
anaconda `python` (3.9.7 / numpy 1.26.4 / scipy 1.13.1) and python.org 3.14
(3.14.2 / numpy 2.3.5 / scipy 1.17.0). Every `test/baseline/*.npz` was made on
the first; every file in `output/_audit/` was made on the second. **The fourteen
failing node ids pass on anaconda in one invocation — `14 passed in 182.66s` —
and fail on 3.14.** So the two "root causes" below are one cause, and it is not
in `eos/`. `pyproject.toml:5` admits both stacks and picks neither; the ruling
is [ticket 57](issues/57-canonical-stack.md). **Until 57 is ruled, every failure
count in this block and in `output/_audit/` is a measurement of an interpreter,
not of the code — including "0 added failures" claims checked against it.**
Report which interpreter you ran with.

**Superseded again by [ticket 56](issues/56-baseline-empty-sector-gate.md).**
Current measured state, both stacks, current tree:

    anaconda 3.9.7   1650 passed, 15 skipped, 0 failed   GREEN
                     output/_audit/pytest_after_ticket56_py39.txt
    python.org 3.14  12 failed, 1638 passed, 15 skipped
                     output/_audit/pytest_after_ticket56_py314.txt

Collection is **1665**, not 1663 — ticket 20 added two import tests. On 3.14 the
count fell 14 -> 12 because ticket 56 dropped the undetermined potentials that
`sfho` and `vmit` were failing on; the `^E ` diff against
`pytest_after_ticket45.txt` is **pure deletion, 19 lines, nothing added or
changed**, so that is 0 added failures. The remaining 12 on 3.14 are all
[ticket 57](issues/57-canonical-stack.md).

**`pytest_after_ticket45.txt` is no longer a valid before-image for
`test_baseline[*]`**: four `.npz` were regenerated after it, and `test/` is
gitignored so the key set it was measured against cannot be reconstructed.
Compare baseline rows against `pytest_after_ticket56_py314.txt` instead.

The block below is kept as written, and was accurate FOR THE 3.14 STACK before
ticket 56.

**14 failed, 1634 passed, 15 skipped at HEAD.** Expect 14, not 0. All fourteen
are **pre-existing and not physics regressions** — verified twice, independently,
by running them against a detached worktree at HEAD carrying the pre-rename
`eos/`, where the same tests fail with byte-identical messages. Two causes:

- **6 — dd2's NMP inversion misses its targets**, now
  [ticket 47](issues/47-dd2-nmp-inversion.md). Diagnosed as never having worked:
  bit-identical at all 13 commits since `eos/dd2/nmp.py` was created. Two of the
  three shapes are a tolerance asserting below the documented stencil noise; the
  third is real — the default 5x5 closure lands in the spurious basin
  `nmp.py:70` describes, Q_sat 117.49 against 169.00.
- **8 — `test/baseline/`, and NOT all one cause.** `ccdm`'s `field_residual`
  and the tov sequences are round-off drift. `sfho`'s `mu_S` and `vmit`'s
  `n_e` are [ticket 56](issues/56-baseline-empty-sector-gate.md): the generator's
  absolutely-scaled gates. `vmit`'s is a genuine flakiness band (`n_e` at
  Y_C = 0 straddling 1e-12 at 1.7e-13 against a stored 3.0e-12). `sfho`'s
  additionally has a solver story — its three `fixed_YC_YS` rows all target
  Y_S = 0 and close to n_S = 2.4921e-09, 3.9248e-16, 3.0107e-16, so the
  n = 0.16 row converged SEVEN ORDERS less tightly than its own siblings and
  that, not a different physical case, is what put it the wrong side of a gate
  that classified the other two correctly. It belongs with "scaling the
  strangeness residual" below as much as with the generator.

**A ticket reporting "0 added failures" means 14, unchanged.** Compare against
`output/_audit/pytest_after_ticket45.txt` — a full run on `main` at
`66b1051`, which is the current pair with `pytest_before_ticket45.txt` and
supersedes `pytest_before_with_crust.txt` (that file predates the 14 and reads
`1 failed, 1648 passed`). The two ticket-45 files have a **byte-identical set
of assertion messages**, so a diff of their `^E ` lines is the cheapest
added-failure check there is. Both causes are Stage 7 report material, not
diffs.

The crust no longer needs `EOS_CRUST_DIR`: [ticket 39](issues/39-crust-silent-fallback.md)
shipped the tables in `eos/astro/tov/data/`, so a bare run gets 15 skipped and
no crust failure.

**Do not run the full suite concurrently with another session** —
`test/dd2/test_dd2_speed.py` is a timing test and goes flaky under CPU
contention. Coordinate with whichever session holds the suite gate.

*Superseded:* the earlier "1648 passed, 15 skipped, 0 failed" measured after
Stage 0 and recorded in `output/_audit/pytest_after_stage0.txt`. That count had
fallen from 1660 because [ticket 03](issues/03-stage0-removals.md) deleted
`test/dd2/test_notebook_api.py` and its 12 tests with the module they
smoke-test. It was accurate when written and is not any more.

The session began at 4 failures on a bare checkout, against an inherited note
claiming 12. None of the four was a physics defect: three were the BPS crust
table not being on the search path, and the fourth was the regression baseline
freezing an undetermined `mu_S` under six derived names. Compare later work
against this file, not against the earlier `pytest_before*.txt`.

## Decisions so far

<!-- one line per closed ticket: gist + link -->

- [dd2 cannot take two of §4's six species flags](issues/61-dd2-species-flags.md):
  **closed by giving dd2 the names — all ten models now take `SpeciesFlags(**six)`
  and none answers §4's vocabulary with a `TypeError`.** The meson half is a pure
  rename (`include_pseudoscalars` -> `thermal_mesons`, `include_thermal_vectors` ->
  `thermal_vectors`), reading §4's *"pi, K (and optionally the vector nonet)"* as
  one sector plus an option rather than two halves; no alias, because an alias
  leaves two spellings of one boolean for `dataclasses.replace` to pick between.
  The tau-gas half is the name WITHOUT the sector: `thermal_neutrinos` exists and
  RAISES `NotImplementedError`, which §4 permits and which the ticket blessed —
  the complaint was the missing name, not the missing physics. dd2's own
  `neutrinos` (the trapped modes' matter-composition electron neutrino) is
  untouched and now sits next to it with comments pointing each at the other.
  **The prose sites were FOUR, not the three the two-way gate names**:
  `README.md`, `eos/__init__.py`, the test docstring — and `docs/DEFERRED.md`,
  whose dd2 section carried the same claim as a deferral and which nothing would
  have fired on. That is a fourth instance of the Not-yet-specified pattern
  below, found the same way as the first three: by accident. `exempt` in
  `test/test_imports.py` is now `{}` (kept as an empty dict, so a future
  exemption has to be argued for in writing); the inversion the ticket demanded
  was already in place, so `enjl` and `abpr` stopped being skipped for nothing.
  Defaults measured and NOT changed as instructed — the divergence is
  **three-way** (`muons` 5/5, `photons` 6/4, `thermal_neutrinos` True in
  `alphabag` alone), graduated to [ticket 65](issues/65-species-flag-defaults.md),
  which does not block the notebooks because the knobs cell passes all six
  explicitly. AST check for `{thermal_mesons, thermal_vectors,
  thermal_neutrinos}` clean before and after; no positional `SpeciesFlags`
  construction anywhere, so reordering fields is safe; no `test/baseline/*.npz`
  stores a key named after either old flag (all 13 checked — `table_io.asdict`
  is the path by which a field rename could have moved a stored key).

- [dd2 raises a bare KeyError when hyperons are asked of a nucleonic set](issues/60-dd2-hyperon-flag-raise.md):
  **fixed with one guard, and the exception TYPE is the whole fix.**
  `build_baryon_specs` has exactly one caller in the repository, so the guard
  goes there and no sibling caller keeps the old behaviour; it tests
  `b.name not in hyp`, so a partial coupling map is caught like an empty one.
  It raises **`NotImplementedError`, not `ValueError`** — measured, not
  reasoned: with `ValueError` (which is what `sfho`'s equivalent guard uses)
  `eos_point` returned `ok=False` instead of raising, because
  `eos/dd2/api.py:106-108` re-raises `NotImplementedError` and converts
  `ValueError` into a non-converged status. A malformed call reported as a
  failed solve is worse than the bare `KeyError` the ticket opened on: a sampler
  scores it and moves on. `sfho` has the same defect one refactor away and is
  saved only by calling its guard outside `api.py`'s try — Stage 7 material, not
  changed here. **`deltas` has no equivalent hole**, measured rather than
  assumed: `x_Delta_*` default to 1.0 as a real coupling choice, and
  `deltas=True` on the nucleonic set converges, so no guard was added where the
  model can genuinely compute.

- [The baseline's empty-sector gate is absolute where the physics is relative](issues/56-baseline-empty-sector-gate.md):
  **fixed — the gates now read fractions, and the suite is GREEN on the stack
  that made the baselines** (1650 passed, 15 skipped, 0 failed). `Y_E_EMPTY =
  1e-8` sits in an 81,220x gap; `Y_S_EMPTY = 1e-6` is deliberately permissive
  because no threshold in Y_S separates free from imposed — the `mu_S != 0` test
  does that, as the ticket reasoned. Both are strictly more permissive than the
  old absolute gate, so the change can only drop a key. **The ticket
  under-predicted the blast radius: 34 keys, not 13** — `did` lost 18 (all
  exactly `0.0`) because the first gate pops `mu_S` BEFORE the second gate judges
  whether it was free, which the ticket's "did survives" argument had not
  accounted for. Every survivor bit-identical by *exact equality*, zero added,
  nine files not regenerated at all. On 3.14: **0 added failures, 2 cleared**
  (`sfho`, `vmit`), `^E ` diff pure deletion. A second, more principled gate
  change was rejected as out of scope — it would RESTORE 31 keys in `mixed`/`njl`.
  **None of this is in git** (§11 gitignores `test/`): the first instance of
  losing DATA §12 calls ground truth rather than recoverable logic, which is now
  the sharpest argument for [ticket 21](issues/21-phase5-structure.md).

- [dd2's NMP inversion misses its targets](issues/47-dd2-nmp-inversion.md):
  **a report, and the ticket's premise was false — the stack DID move.** Two
  Python stacks live on this machine and `python` resolves to the one that did
  NOT run the audits; all 14 failures (ticket 47's six AND
  [ticket 56](issues/56-baseline-empty-sector-gate.md)'s eight) pass on
  anaconda 3.9 and fail on python.org 3.14. The earlier determinism checks were
  sound but varied every axis except that one, and the install-date check could
  not discriminate because BOTH stacks predate every `.npz`. The physics ruling
  asked for is **No**: the published DD2 couplings are not a root of the 5x5
  closure — they miss its own cross-constraint by 2.200718e-03, so four rows
  vanish there and the cross row does not, and reaching the true root moves
  `b_sigma`/`c_sigma` (Q_sat's coefficients) to predict Q_sat 117.49. scipy 1.13
  only appeared to confirm the docstring by returning the seed after ZERO
  iterations, which `ISO_GATE = 2e-2` waves through as ok=True. Corrected in
  `5644ed0` (docstrings only, no number moves). The `abs=0.2` Q_sat tolerances
  are confirmed below their noise floor and deliberately **not** touched (§12).
  Stack ruling graduated to [ticket 57](issues/57-canonical-stack.md).

- [Where do the models deviate from §13's names, order and docstrings?](issues/07-naming-sweep.md):
  56 failing `thermodynamics.py` docstrings, 74 name deviations (58 public), 6
  files in the wrong reading order, 8 docstrings citing a plan or phase, 9
  equation-hiding comprehensions. `vmit` was never converted and `DEFERRED.md:320`
  wrongly calls it done; `dd2` is second worst; `zl`/`abpr`/`enjl` pass.
  `nucleation` touches none of the renames. Full report:
  [naming-sweep.md](research/naming-sweep.md).

- [Do the twelve model documents pass §11's reproduce-without-source test?](issues/06-document-audit.md):
  24 documents audited (not 25 — `tov.md` exists too). **Only `zl.tex` and
  `alphabag.tex` pass outright**, and **the `.tex` carries more than its `.md` in
  all twelve pairs**. Furthest from passing: `ccdm.md` and `did.md` (3/16),
  `njl.md` (4/16), and the `vmit` pair — the only model where both files fail.
  `mixed.tex` does not compile. Full report:
  [document-audit.md](research/document-audit.md).

- [Where does the repository actually stand against CLAUDE.md?](issues/08-conformance-table.md):
  136 cells — 87 Pass, 24 Fail, 25 Ambiguous. Worst units `dd2` and `mixed`.
  Sorted 22 (a) code-is-wrong / 11 (b) document-should-change / 12 (c) deferred.
  Headline failures: **`dd2` accepts `photons=False` and returns a photon gas
  anyway** ([ticket 28](issues/28-photons-silent-ignore.md)), the `progress`
  dict's `fracs` drops fixed fractions in dd2 and sfho, and `astro/gmode` imports
  `eos.dd2.solver`. `ccdm` appears nowhere in CLAUDE.md. Full report:
  [conformance-table.md](research/conformance-table.md).

- [Keep both .md and .tex per model, or drop the .tex?](issues/09-tex-or-md.md):
  **keep both, carrying the SAME information, each written natively for its
  format.** §11 stands unchanged and so does the compiling-`.tex` acceptance
  criterion. Cost accepted: 24 documents to §11 standard, not 12 — graduated to
  tickets 30–36.
- [dd2 and mixed accept photons=False and return a photon gas anyway](issues/28-photons-silent-ignore.md):
  confirmed and **fixed** — dd2 now reads `species.photons` at all five dispatch
  points, matching the five models that already did. Exactly one photon gas was
  leaking (0.355 % on P at T = 30 MeV, growing as T⁴). **No golden reference was
  affected**: the baseline generator never asked for `photons=False`. Check left
  at `test/dd2/test_photons_flag.py`. `mixed` split to
  [ticket 29](issues/29-mixed-species-flags.md) — it has no `species.py` at all.
  Verified against a fresh post-fix run: **0 added failures**.

- [Establish the pre-existing failure baseline on main](issues/01-pytest-baseline.md):
  **4 failed, 1634 passed, 26 skipped**, saved to `output/_audit/pytest_before.txt`.
  Three of the four are `did` ([ticket 37](issues/37-did-failures.md)); the fourth
  is a dd2 TOV radius ([ticket 38](issues/38-dd2-tov-radius.md)). Supersedes both
  the inherited 12-failure count (measured on `enjl-finite-T`) and a first
  8-failure run that was contaminated by a source edit mid-suite. **With
  `EOS_CRUST_DIR` set the suite is 1 failed / 1648 passed / 15 skipped** — the
  crust table also unskips eleven tests that were silently not running. Later
  tickets compare against `output/_audit/pytest_before_with_crust.txt`.

- [dd2's TOV pipeline gives R = 12.33 km against an asserted 13.2 ± 0.4](issues/38-dd2-tov-radius.md):
  **not a defect — the BPS crust table is absent and the test helper silently
  falls back to no crust.** `have_crust("BPS")` is False because the search path
  is only `<repo>/data/crust`; the file is at
  `/Users/mircoguerrini/Desktop/Research/Crust/BPST0.dat`. With `EOS_CRUST_DIR`
  set, dd2's and did's M–R tests all pass. The silent downgrade is
  [ticket 39](issues/39-crust-silent-fallback.md).

- [Fix the 13 factual defects the document audit found](issues/27-document-defects.md):
  **the acceptance criterion "every model has a .tex that compiles" was failing on
  FIVE documents, not the one the audit reported** — LaTeX halts at the first
  error, so each masked the next. All twelve now compile. Nine of the thirteen
  factual defects fixed, each verified against the code first (notably `sfho`'s
  field sources were missing `(hbar c)^3`, and `enjl`'s modes table claimed T = 0
  for a solver that converges at T = 10 MeV). Four are omissions rather than
  errors and were re-assigned to the pairs being rewritten (tickets 30, 32, 35, 36).

- [Does zl get an nmp.py, or is its absence recorded as deferred?](issues/26-zl-nmp.md):
  **forward map added, inverse raises.** The five published values reproduce and
  are now pinned in `verify/`; they had been quoted in a docstring that nothing
  checked. The symmetry energy had to be the `beta -> 0` curvature rather than
  `did`'s full-step estimator, which carries `beta^4` contamination — a real
  difference, now documented in both models. `invert_nmp` raises: six couplings
  against five NMPs, no published closure condition.

- [Three grouped notebooks, or one per model?](issues/02-notebook-grouping.md):
  **grouped**, and §11 is amended. Stage 1's figures 1–5 each overlay all four
  hadronic models on one axis, which nine per-model notebooks cannot do without
  importing each other or sharing a helper module §11 forbids.

- [A missing crust table silently becomes a 0.9 km physics error](issues/39-crust-silent-fallback.md):
  the tables were never large — `BPST0.dat` is 4.8 kB — so they now ship in
  `eos/astro/tov/data/` and a fresh clone has a crust with no environment set up.
  The two TOV helpers skip with a message instead of silently dropping the crust.
  Caveat: `test/` is gitignored, so only the data move is in git.

- [Make mu_S determined when the strange sector is empty](issues/40-determine-mu-s.md):
  the cause was neither the ticket's nor DEFERRED's reading. **The baseline
  generator already drops `mu_S` where `n_S = 0` but keeps `mu_i`/`mu_eff_i`,
  which carry it linearly through `S_i`** — so the free number stayed frozen
  under six other names. Exclusion completed, `did` and `sfho` regenerated,
  `dd2` untouched. `mu_S` also diverges as `T ln 10` per decade of `Y_S`, so the
  "continuity" candidate was dead. Solver-side residual scaling is still open
  but no longer needed for a green suite.

- [tov.md and tov.tex to §11 standard](issues/34-tov-documents.md): the pair
  described how RNS is *driven* and never what it solves. Both now carry the KEH
  metric, the first integral and the specific enthalpy that is the only place the
  EoS enters; the eighteen fields of `RotatingResult`; and the numerical
  parameters that bound a computed mass. `Komatsu1989` and
  `CookShapiroTeukolsky1994` added to `docs/eos.bib` — both documents leaned on a
  formulation neither cited. Also corrected a staleness this session's own crust
  commit introduced.

- [What goes from notebooks/, and what is lost with it](issues/03-stage0-removals.md):
  **fifteen files removed, one 46 MB folder held, one defect found.** The five
  `_usage` pairs carried zero stored outputs, so nothing reachable was lost;
  `notebook_api.py` went as a three-part removal (module + its test + the now-empty
  `_EXEMPT_FILES` entry), closing **three** stale `DEFERRED.md` references and the
  last model-to-model import edge. `notebooks/eos_tables_DD2vMIT/` is **held, not
  deleted**: it is gitignored with 0 files tracked, so the map's lifted deletion
  gate does not cover it and no replacement is guaranteed until
  [ticket 05](issues/05-notebook-coverage.md) rules on `mixed`. Separately, commit
  `d9f8eec` **broke the JSON of all three notebooks it touched** — two were being
  deleted anyway, the third is on the KEEP list, split to
  [ticket 41](issues/41-corrupt-notebooks.md).

- [Approve or reject the proposed public renames](issues/10-rename-approvals.md):
  **the gate is passed** — 46 of 58 approved, 3 frozen, 1 deferred, 5 split out as
  not-renames, 3 ruled keep. The fact that made it cheap: **not one of the 58
  touches a `nucleation` call site**, so Phase 6 is not exposed and the whole
  ~550-site radius is `eos/` + `test/` (over half of it in gitignored files).
  Application graduated to per-package tickets [42](issues/42-rename-internal.md),
  [43](issues/43-rename-vmit.md), [44](issues/44-rename-dd2.md),
  [45](issues/45-rename-sfho.md), with the five behaviour changes at
  [46](issues/46-api-changes.md). Frozen: vmit's three legacy table symbols, whose
  only consumer is the out-of-scope and currently-unopenable ZLvMIT notebook.
  Also corrected `DEFERRED.md`'s vmit entry, which claimed DONE while naming 2 of
  ~23 unconverted names.

- [did.md and did.tex to §11 standard](issues/31-did-documents.md): the .md was a
  README at 3/16 and is now the specification (168 lines → 885); the .tex failed
  C2 alone and completely. **Neither file had ever carried a parameter table** —
  two numbers against 29 stored fields — and the omega/phi couplings are DERIVED,
  so the vertex strengths behind them existed only in the source. Both now carry
  all of it, with g_8^S = 9.178769 / g_8^N = 9.326009 computed from the code.
  Two claims the code corrected: `F_M(1) = 0.988` is not one number for "the
  vector shape" (omega/phi 0.98829, rho 0.96488), and the quoted `g_phiN = -5.20`
  is the value at saturation, not the stored g^{S(0)}_phiN = -5.262966. No .bib
  key needed — every new citation was already there. Compiles clean, 13 pages.

- [Apply the approved renames — eos/mixed and eos/did](issues/42-rename-internal.md):
  9 renames, 16 files, **0 added failures and `test/baseline/` unmoved at
  rtol = 1e-10** — a rename that changes a number is not a rename. But the
  rehearsal earned itself: **the renames were not mechanical and the failure was
  silent.** `mixed/api.py` had a local `def solve(...)` adapter beside the imported
  `solve_mixed`; renaming one onto the other made it call itself, and since
  `RecursionError` subclasses `RuntimeError` the existing `except` swallowed it
  into a returned "did not converge" — 12 tests red with no traceback naming the
  cause. The pattern is systematic: this repo already used §13's vocabulary for
  LOCAL adapters, which is what the public names are being renamed TO. An AST
  check now rides on tickets 43–45 and has already found the next one,
  `vmit/table.py:188`, before any code moved.

- [njl.md and njl.tex to §11 standard](issues/32-njl-documents.md): the audit's
  **largest single §11 violation** is closed — 720 lines of equations that
  contained zero parameter values now carry all three tiers with the vacuum
  observable each is fitted to, so the reproduction table's numbers are
  checkable from the document. The `.md` went from 166-line summary to the same
  document in plain text. Three claims the code overturned while writing:
  the shipped pairing quadrature is **24** nodes per panel, not the 100 of the
  accuracy benchmark; `n_ref = 0.48 fm^-3` is a *quark* density, i.e.
  `n_B = n_sat`; and the mixed adapter does **not** put the winning pattern in
  the returned block's `fields` — it comes back as the warm-start key, and
  `njl_phase`'s own docstring says otherwise (reported, not fixed). The
  carried-in `n_s` collision is discharged with its reason: `eps - 3P = M rho_s`
  holds only for the medium pieces and only with `P_k4`, so the trace identity
  is used nowhere and `n_s` really is the strange-quark density. `.tex` compiles
  clean in two passes, `eos.bib` untouched.

- [ccdm.md and ccdm.tex to §11 standard](issues/30-ccdm-documents.md): the joint
  worst `.md` (3/16) is now the same document as the `.tex`, in plain text — `U`
  and `V` in closed form, the five integrals at both temperatures, the residual
  row by row, every parameter with its number. The carried-in label collision is
  discharged by giving the *modes* a new label: rows are `R_1..R_4`, the
  specification's closure rows are **`M1..M5`** in both files. **Four defects
  the code overturned**: `rho_s = +dOmega/dM*` (the `.tex` had the sign its own
  per-mode identity contradicts), `gapless` is `min E < 1e-3 max|Delta|` not
  `min E < 0`, the enumeration ranks by **`f = eps - Ts` at fixed density** and
  by `Omega` only at fixed potential, and the thermal-neutrino sector (3
  free-streaming / 2 trapped) `solver.py` adds to `P`, `eps`, `s` was in neither
  file. `ccdm.tex` now cites `docs/eos.bib` like the other eleven; doing so
  surfaced **two latent bib defects** — `Steiner2002` and `deCarvalho2010` carry
  bare underscores in `note` fields and halt pdflatex — now escaped, plus
  `ParticleDataGroup2024` appended for the tier-1 vacuum constants. 22 pages, no
  undefined reference or citation.

- [Apply the approved renames — eos/vmit](issues/43-rename-vmit.md): 23 renames,
  24 files, **0 added failures and `test/baseline/` unmoved**. Both predicted
  collisions were real and **one was not the one the ticket named**:
  `table.py`'s local `warm_start` adapter was (it is `seed` now), but so were
  four local `default_guess = ...` bindings in `solver.py` that the AST check
  cannot see — it compares imported against defined names, and a local binding
  inside a function body is neither. `_GUESS_KIND` died with the rename: once
  `warm_start`/`default_guess` read the §3 mode name, its translation table had
  nothing to translate. Merging four cold guesses into one was proved
  **bit-identical** over every mode x density x temperature x Y_C x lepton flag
  before it was trusted. vMIT is the sixth model to expose `thermo_from_mu`, so
  `mixed/adapters.py` aliases it beside the five it already aliases.

- **The map's "1648 passed, 15 skipped, 0 failed" line above is STALE.** The
  suite is **14 failed, 1634 passed, 15 skipped** at HEAD, and all 14 predate
  ticket 43 — verified against a worktree at HEAD carrying the pre-rename
  `eos/`, where the same 14 fail with byte-identical messages. Two root causes,
  now [ticket 47](issues/47-dd2-nmp-inversion.md): **dd2's NMP inversion misses
  its targets** — 3 `test/dd2` + 3 `test/tov` failures AND the `dd2` baseline's
  `nmp.K_sat`/`Q_sat`/`K_sym` drift, one function, seven tests. Diagnosed and
  **it is not a regression**: bit-identical at all 13 commits since
  `eos/dd2/nmp.py` was created, `compute_nmp` deterministic and independent of
  numba and threads, and numpy/scipy/python predate every `.npz`. Two of the
  three shapes are a tolerance asserting below the documented stencil noise;
  the third is real — the default 5x5 closure lands in the spurious basin
  `nmp.py:70` describes, Q_sat 117.49 against 169.00. Also
  **`test/baseline/` drift — but not all one cause** (`ccdm`'s
  `field_residual` and the tov sequences are round-off; `sfho`'s `mu_S` and
  vmit's `n_e` are [ticket 56](issues/56-baseline-empty-sector-gate.md), and sfho's
  is a convergence outlier rather than drift — see the Suite status block; and
  vmit's
  `n_e` at Y_C = 0 straddling the 1e-12 gate at 1.7e-13 against a stored
  3.0e-12). Both are Stage 7 report material, not diffs.

- [ZLvMIT_hybrid.ipynb is corrupt JSON and cannot be opened](issues/41-corrupt-notebooks.md):
  repaired forward rather than reverted — 48 cells, 29 code, 0 stored outputs,
  the same shape `d9f8eec^` had. The first defect was not a string swap at all:
  the commit SPLIT one import into three because `load_crust_table` had moved to
  `crust.py` and `EOSTable_for_TOV` to `general/state.py`, and emitted the three
  with real newlines inside one JSON string. `add_crust` now comes from
  `eos.astro.tov.crust`, where it is defined, rather than through `solver.py`'s
  re-export. `notebooks/zlvmit_test.ipynb`, the third notebook `d9f8eec`
  touched, was checked and is valid, so the KEEP list is entirely loadable.

- [alphabag, abpr, enjl and mixed document pairs to §11](issues/36-quark-engine-documents.md):
  eight documents, all four `.tex` compiling clean, no `.py` touched. **The
  largest single gap was `enjl.md`'s five naming-not-defining failures
  compounding**: with the quantum-number table absent, `J_rho`, `Sigma^R_b` and
  three residual rows could not be evaluated from the page at all. **`n_s` is
  settled and the answer is not §11's identity** — `eps - 3P = M n_s` holds
  identically per species for ENJL's MEDIUM terms (verified to round-off) and
  fails once the Dirac sea is subtracted, since `eps^vac != M n_s^vac`; the
  other three have no scalar density at all. `mixed` gained the five missing
  `PhaseThermo` fields (`mu_dot_n` among them — what its own 1e-8 Euler check
  consumes), eight named-but-never-given constants, the `Window` reason labels
  and the API surface; its `.md`'s asserted row order was wrong, and ticket 29's
  unconditional photon gas is now in both files. `abpr`'s parameter tables gave
  wrong code names in the one column a reader would copy from. **The name ticket
  10 deferred here is ruled**: the §5 adapter surface is `thermo_from_mu`
  everywhere, a lower layer that also takes the fields is `thermo_from_fields`;
  applied by tickets [44](issues/44-rename-dd2.md), [45](issues/45-rename-sfho.md)
  and the new [48](issues/48-rename-did-surface.md), not here.

- [Sort every failing conformance row into fix-code, fix-CLAUDE.md, or defer](issues/11-conformance-triage.md):
  all 41 rows ruled — **23 (a), 12 (b), 10 (c)** — but the first result was that
  **six had already been discharged** by tickets 03, 26, 28 and 43 since the audit
  was written, and finding 6 was not dd2's bug under another name: `eos/mixed` has
  no `species.py` at all, which is why it is [ticket 29](issues/29-mixed-species-flags.md)
  and not a one-liner. The (a) work is cut **by what gate the change needs, not by
  model or by section** — what decides whether two fixes share a session is
  whether they can move a number: [49](issues/49-nonconvergence-return.md) the
  seven §6 boundary raises, [50](issues/50-mechanical-fixes.md) the eight fixes
  that move nothing, [51](issues/51-verify-invariants.md) the four missing
  `verify/` invariants, [52](issues/52-general-t0-integrals.md) the `general/` T=0
  promotion under a golden-reference gate,
  [53](issues/53-gmode-contract.md) the gmode composition contract,
  [54](issues/54-signature-corrections.md) the public-signature corrections. The
  (b) rows are one CLAUDE.md diff on [22](issues/22-phase5-claudemd.md); the (c)
  rows are [55](issues/55-deferred-ledger.md). Four rulings the code decided:
  **`general/fermi_integrals.py` exports no public T = 0 entry point at all**, so
  dd2 re-deriving the Fermi gas is a missing door in `general/` and the fix lands
  there, not in dd2; §4's own wording settles the `thermal_neutrinos`-plus-trapped
  split five models disagree on (the three that succeed are right); `Y_p` stays in
  dd2's signature because it is a freeze target, not a condition; and a failed
  `eos_response` returns the full dict with `converged=False` and NaN, so no
  caller needs a second code path. Six of the 23 (a) fixes can move a number and
  each names its §12 check.

- [Phase 5 item 5 — apply the CLAUDE.md diff](issues/22-phase5-claudemd.md): all
  twelve (b)-class rows applied plus the two carried from tickets 02 and 09 —
  `CLAUDE.md` +97/−17, `docs/REFACTOR_PROMPTS.md` +12, **no `eos/`, `test/` or
  `docs/DEFERRED.md` file touched**. `ccdm` now appears in the specification at
  all; §1's model list had three of the ten. Three rows came out differently
  than the triage anticipated: **§3's `cfl` could not be a table row alone** —
  §3 opens "Every model exposes the same modes", so adding a fifth implies every
  model owes it, and the entry carries the paragraph saying why a locked phase
  is a statement about which phase the model describes rather than a choice of
  equilibrium condition (which then justifies §5's single-mode `mode` default,
  so the two rows cross-link). **§4's `thermal_neutrinos` ruling needed no new
  rule** — it follows from §4's existing "flavors NOT tracked in the matter
  composition", so it is written as the consequence five models read two ways.
  And **the §10 rcParams criterion took two attempts, the first wrong in the
  finding's own way**: the audit's `rcParams\s*\[` misses `rcParams.update(base)`,
  one of the 22 real assignments, and widening to `[\[.]` re-hits the prose
  sentence at `zlvmit/plot_results.py:184`. Shipped as
  `grep -rnE --include='*.py' 'rcParams\s*(\[|\.update)'`, verified across both
  repositories: 22 hits, one file. **The (c)-class half stayed at
  [ticket 55](issues/55-deferred-ledger.md)** rather than moving with this ticket
  as its Question intended — `docs/DEFERRED.md` is a file the rename tickets
  edit, and a concurrent session was live in this checkout applying
  [ticket 44](issues/44-rename-dd2.md); `CLAUDE.md` was taken instead precisely
  because it is disjoint from every rename.

- **The empty-sector question is now sharp and ticketed.** What ticket 37 left as
  "whether an underdetermined potential belongs in a frozen baseline at all" is
  [ticket 56](issues/56-baseline-empty-sector-gate.md): ticket 40's exclusion
  MECHANISM is right and its GATE is wrong, testing an absolute `n_S < 1e-12`
  where emptiness is a statement about `n_S / n_B`. Measured blast radius is
  **one row in one model** — `sfho.npz ycys.n0.16`, `n_S = 2.49e-09`
  (Y_S = 1.6e-08, nine orders above the gate) with a free `mu_S = 8.4496` frozen
  at rtol = 1e-10. `did`'s twenty tiny-`n_S` rows and all twelve of `dd2`'s carry
  `mu_S = 0` exactly — imposed, not solved — so the generator's second gate keeps
  them correctly and the fix cannot cost them. **Measuring both gates across all
  nine baselines then changed the ticket twice.** The lepton side has a clean
  gap — empty tops out at `Y_e = 6.9e-11`, populated starts at `5.6e-06`, nothing
  between — and the absolute gate sits INSIDE the empty cluster, not in the gap,
  which is why vmit flakes: three rows sit above it by less than a factor of 70,
  and vmit's two are its "2 quantities no longer produced". The strange side has
  **no such gap** — sfho's one free row at `Y_S = 1.56e-08` is bracketed by did's
  imposed rows at 5.5e-09 and 2.2e-08, so no magnitude threshold separates them
  and two of the ticket's three original options are dead there. What rescues it
  is that the first gate need not discriminate at all: the second gate is exact,
  since an imposed `mu_S` is identically zero. And the root cause is upstream of
  both — sfho's `ycys` at n = 0.16 closed the strangeness row to 2.49e-09 where
  its own siblings at 0.32 and 0.64 reached 3.9e-16 and 3.0e-16, **seven orders
  tighter, same model and mode** — which ties this to the map's open
  "Scaling the strangeness residual" question rather than to the gate alone.

- [Apply the approved renames — eos/dd2](issues/44-rename-dd2.md): 17 renames +
  1 fold across 74 files, **0 added failures, `test/baseline/` unmoved, every §12
  golden reference intact** — dd2/verify's SNM(0.16) golden point at 1.40e-05,
  CompOSE HS(DD2) at 2.83e-05, backend parity at 4.40e-14, and ticket 47's NMP
  floor reproducing bit-identically, which is the one area already known
  unstable. **The ticket's proposed merge of the two warm starts was wrong and
  the ticket was right to ask**: `beta_warm_start` returns a fixed 4-vector for
  the nucleon-only solver, `octet_warm_start` a variable-length one carrying
  omega0, phi0, mu_S and mu_nue — so the octet path took the §13 name `warm_start`
  and the reduced path became `nucleon_warm_start`, with `default_beta_guess`
  following it so the pair stays a pair. `sweep_beta_eq_octet` DID fold, being a
  pure pass-through. **A third collision shape surfaced, invisible to tickets 42
  and 43's check**: once two models take the same §3 names,
  `test/mixed/test_hybrid_modes.py` imports `solve_fixed_yc` from BOTH dd2 and
  vmit at module level and the second silently wins — it raised `TypeError` only
  because the signatures differ. The extended checker is at
  `shadowcheck.py`; [ticket 45](issues/45-rename-sfho.md) makes sfho the third
  model onto those words and must sweep the whole tree, not just its own names.

- [Write the ten (c)-class conformance rows into docs/DEFERRED.md](issues/55-deferred-ledger.md):
  all ten written plus the correction, `docs/DEFERRED.md` +206/-2 with **no
  `eos/` or `test/` file touched**, so the suite could not move. Three rows had
  moved since the audit and the code decided each: gmode's fourth import site
  now names `Parameters` and `sweep`, not the `Parametrization` and
  `sweep_beta_eq_octet` [ticket 44](issues/44-rename-dd2.md) retired; the
  downward-deferral row is **ten sites, not eleven**, because dd2's
  `responses as _fd` is gone and its two remaining deferrals are the
  analytic-Jacobian branch, which is a deliberate optional-backend deferral
  rather than drift; and **the correction the ticket asks for had already come
  true** — ticket 44 made "every model's parameter dataclass is `Parameters`"
  true of all ten (`grep -c '^class Parameters' eos/*/parameters.py` → ten), so
  the sentence is corrected to state the fact with its history rather than
  deleted as false, and sfho's still-unconverted `get_sfho*` constructors are
  named as [ticket 45](issues/45-rename-sfho.md)'s half. Row 4 gained a
  measurement: the one unbounded loop expands its bracket GEOMETRICALLY, so a
  hang needs a non-finite `n_target`, which is why the fix is a counter
  returning non-convergence rather than a large bound. Noticed and not fixed:
  `### astro/tov`'s crust-path bullet is stale, closed by
  [ticket 39](issues/39-crust-silent-fallback.md) — Stage 7 material.

- [vmit.md and vmit.tex to §11 standard](issues/33-vmit-documents.md): the only
  pair where both files failed is closed — `.md` 82 → 707 lines, `.tex`
  265 → 992, compiling clean at 10 pages, no `.py` touched. **The ordering
  defect was not one wrong statement but two half-right ones**: there are TWO
  unknown layouts (`mu_e` in slot four in the beta modes, appended last in the
  fixed-fraction ones) and each file had generalised the one the other got
  wrong — which is why neither read as broken. Both now carry the layout table
  and every residual row by row, including that **`R6` and `R7` swap between
  the two beta modes**. Four gaps beyond the audit's list: the finite-T
  integrals were **missing their `(hbar c)^3`** (the sfho defect class again),
  the totals equation **omitted leptons and photons entirely**, there were no
  `T = 0` forms in a model whose strange onset IS a `T = 0` threshold, and
  nothing on returns, table keys, API or cold starts. Three claims the code
  decided: `B/(hbar c)^3 = 136.63`, the light flavours are ultra-relativistic
  to 5e-4 rather than 3e-4, and **`verify/` runs EIGHT checks, not the seven
  its own docstring enumerates** (reported, not fixed). `n_s` discharged with
  its reason — no scalar sector, so the trace identity has no meaning and
  `n_s` is the strange-quark number density. Ticket 27's carried-in
  `eos_response` claim was already gone, taken by `2844a9a`. `eos.bib`
  untouched; the `.md`'s own reference list was corrected against it.

- [zl, sfho and dd2 document pairs to §11 standard](issues/35-hadronic-documents.md):
  all three pairs done — six files, three `.tex` compiling clean (7, 8 and 8
  pages), no `.py` touched. Both "the closed forms are in `<model>.tex`
  Eq. (T0)" defects closed, and dd2's T = 0 forms **verified against
  `kinetic_thermo` rather than transcribed** — it returns natural units, so the
  fm-based forms differ from it by exactly `(hbar c)^3`, agreeing to 7e-16 once
  reconciled. **A defect in the file the audit passed 14/14**: `zl.tex` gave one
  neutrality row where the code has TWO SIGNS — `n_C - n_e` in the beta modes,
  `n_e - n_C` in `fixed_YC` — the same shape as vMIT's `R6`/`R7` swap, so worth
  expecting in the remaining models. dd2's two Partial cells were one cause, the
  thermal meson sector having no Bose thermodynamics at all; both files now carry
  it with the eighteen species and the condensation refusal, plus the
  `eos_response` set, the masses, the reduced nucleon-only system and the
  phase-adapter residual. **Two sfho claims the code overturned, one of them
  mine**: the trapped mode does NOT double-count `nu_e` — `solver.py` refuses
  the combination — and the factor where the gas IS added is a measured
  3.000000. **The refusal's disposition then took a second correction**, from
  the session holding ticket 45: it is a DEFECT
  [ticket 54](issues/54-signature-corrections.md) deletes, not a ledgered gap.
  `CLAUDE.md:176` already forbids the raise (landed by
  [ticket 22](issues/22-phase5-claudemd.md)), and
  [ticket 11](issues/11-conformance-triage.md):135 rules the row (b) + (a), so
  the `DEFERRED.md` pointer my first fix added had no target and never should
  have. Corrected in `463278a`. **The sfho half was written
  against [ticket 45](issues/45-rename-sfho.md)'s uncommitted tree** by
  arrangement with the session holding it, every name verified independently and
  re-checked at commit. Its correction also saved the dd2 half: **ticket 44
  never applied the `thermo_at_potentials -> thermo_from_mu` rename it carried**,
  so dd2 has no `thermo_from_mu` at all and both pairs name what dd2 actually
  has, flagging it for [ticket 48](issues/48-rename-did-surface.md).

- [Apply the approved renames — eos/sfho](issues/45-rename-sfho.md): 8 renames,
  15 files, **0 added failures, `test/baseline/` unmoved, sfho's `verify/` green
  on all eight invariants**. The keys `named()` takes are the five strings each
  set ALREADY carried in its `name` field, so `Parameters.named(p.name)`
  round-trips and no stored string moves; `default()` is `SFHo_Nucleonic`.
  **A third spelling of the same five existed** — `table.py`'s legacy
  `TableSettings` short strings — and is now an alias table deferring to
  `named`, so there is one registry. `PUBLISHED_SETS` holds **builders, not
  instances**, which is §6 rather than style: a `Parameters` carries mutable
  coupling maps and the builders mutate them, so a module-level dict of
  instances would be the global mutable state §6 forbids — the old
  `get_all_parametrizations()` escaped it only by rebuilding on every call.
  **The collision check earned itself a fourth time, on a collision the rename
  INTRODUCED**: converting `test/mixed/test_phase_pairs.py:111` to
  `Parameters` put a function-local sfho import under the module-level
  `from eos.dd2 import Parameters`. Shape 3 across the whole tree is clean —
  sfho already carried the §3 mode names, so it did not become a third package
  converging on those words. **Two things reported, not fixed**: dd2 never took
  its half of ticket 36's ruling (ticket 44 carried the instruction and its 19
  renames omit it), now on [ticket 48](issues/48-rename-did-surface.md); and
  **`nucleation` cannot import `eos` today** — five modules gone, two names
  gone, every one a Phase 3/4 MOVE rather than a Phase 5 rename, now on
  [ticket 23](issues/23-phase6-respec.md).

- [Which models and subsystems get a notebook at all](issues/05-notebook-coverage.md):
  **four notebooks, not three.** `mixed` gets the fourth, `notebooks/hybrid_eos`
  (tickets [58](issues/58-hybrid-skeleton.md), [59](issues/59-hybrid-figures.md)),
  headline pairing DD2+vMIT with DID+NJL and DID+CCDM as the swap cell, ending on
  a TOV pass through `to_tov()`. `abpr` is a companion panel inside `quark_eos`
  against `alphabag` at CFL, not a fifth peer in the knobs cell. `astro/tov` gets
  no notebook — it is already exercised in two — `astro/gmode` gets nothing and is
  a named gap, and `zlvmit` stays out of scope **without** touching
  [ticket 41](issues/41-corrupt-notebooks.md). Two of the ticket's premises were
  wrong: **`abpr` already has `eos_table`** (`api.py:146`; it lacks only
  `table.py`, because nothing in it iterates — which also settles ticket 07's row
  as "the physics has no such part"), and `astro/*` has no `eos_table` at all.
  `notebooks/eos_tables_DD2vMIT/` was **moved, not deleted**, to
  `eos_tables_DD2vMIT/`, TRACKED at the repository root and in HEAD (the
  `output_old/` premise was wrong), so ticket 59 can compare
  against the published figures instead of asserting a replacement.

- [The shared notebook skeleton: knobs cell, gap handling, table naming](issues/04-notebook-skeleton.md):
  delivered as [research/notebook_skeleton.py](research/notebook_skeleton.py),
  self-check passing. **The ticket's premise that a knobs cell can hold §4's six
  flags is false**: `SpeciesFlags(**six)` raises `TypeError` on **dd2**, which
  lacks `thermal_mesons` (split into `include_pseudoscalars` +
  `include_thermal_vectors`) and `thermal_neutrinos` (genuinely unwired — its
  `neutrinos` field is the matter-composition one), and no conformance row
  covered it. Now [ticket 61](issues/61-dd2-species-flags.md), blocking 12, 15, 18
  and 58. The other two shapes: the gap pattern keeps **three** failure modes
  distinct — a refusal (`NotImplementedError`/`ValueError`, caught), a
  non-convergence (§6 return value, no `except` sees it, tested via `.ok`) and a
  `TypeError` (the notebook's own bug, deliberately not caught); and table names
  are `standard_name()` in `eos/general/table_io.py`, landing under ticket 12,
  writing to `output/tables/<model>/` (§11's per-model split, over the ticket's
  flat `output/tables/`). `use_nmp_inversion` ships off — zl refuses by design and
  dd2's inversion is ticket 47 — so **12 is now also blocked by 47**.

- [Phase 5 items 1, 2 and 4 — top-level imports and a runnable README](issues/20-phase5-api-readme.md):
  `import eos` is now **0.088 s** and lazy (PEP 562), because eager would put
  `eos.dd2`'s 0.47 s of Numba and `constraints`' 0.42 s on every caller;
  `eos.MODES`, `eos.SPECIES_FLAGS`, `eos.MODELS`, the `ModeSpec` factories,
  `EOSTable_for_TOV` and the table I/O are the eager top-level surface.
  **The species flags could not be one class** — every model has its own
  dataclass — so the top level carries the shared NAMES. **A second empty
  surface turned up and is fixed**: `eos/astro/tov/__init__.py` was zero bytes,
  so the commonest downstream task was three modules deep; it now re-exports
  `solver.py` and `crust.py` (not `rotating`, which shells out to a compiled
  solver). `README.md` rewritten 557 -> 424 lines — it documented five models
  of ten, three retired module paths and three `TableSettings` dialects.
  **Every block executed, mechanically**: a script `exec`s all seven in ONE
  namespace in order, so example 5 really does continue from example 3.
  (c) gives **M_max = 2.419, R(M_max) = 11.99 km, R(1.4) = 13.19 km** — the
  published DD2 star, from a bare environment with **no `EOS_CRUST_DIR`**,
  which is [ticket 39](issues/39-crust-silent-fallback.md) confirmed end to
  end. Anaconda 3.9.7 stack; those numbers are stack-dependent if
  [57](issues/57-canonical-stack.md) rules for 3.14. **Three findings, all
  from running rather than reading**: the quick-start example as first written
  raised a bare `KeyError: 'Lambda'` — `Parameters.default()` with
  `hyperons=True` — which is §4's raise without §4's message, now
  [ticket 60](issues/60-dd2-hyperon-flag-raise.md); dd2's missing flag names,
  reached independently of and converging with
  [ticket 61](issues/61-dd2-species-flags.md), whose diagnosis corrects mine
  (`neutrinos` is the matter-composition field, not `thermal_neutrinos`
  misspelt); and **`eos_table` has no `leptons=` at all**, so retiring
  `fixed_YC_neutral` per [ticket 54](issues/54-signature-corrections.md) item 1
  without adding the flag there would make the neutral fixed-Y_C table
  unreachable — noted on 54. Two tests added to `test/test_imports.py`, since
  the lazy hook is invisible to a module sweep. Targeted runs only (the suite
  gate was held): imports 188 + 2, `gmode`+`tov` 64 passed / 15 skipped,
  `mixed` 20 passed.

- [Phase 5 item 3 — write docs/STRUCTURE.md](issues/21-phase5-structure.md):
  shipped, commit `f479845` — **`docs/STRUCTURE.md`, 1132 lines**, plus
  `docs/figures/structure_dd2_vmit.png`. Thirteen sections; **§3 is a
  quantity -> module -> function index** in four tables, which is the section
  Acceptance's under-a-minute test actually lands on. Ticket 09's ruling
  applied as ruled: **both** the `.md` and the `.tex` are linked for all
  thirteen documents, because they carry the same information natively for
  each format; all four notebooks linked, `.ipynb` and `.py`. **14 python
  blocks, all 14 executed** in ONE namespace in order — ticket 20's method —
  so §12's figure really does continue from §11.3's table; the two shell
  blocks are real `verify/` runs. The one non-`python` fence is the four-line
  `try/except ImportError` quoted from `eos/dd2/solver.py`, demoted after the
  extractor caught it, which is the argument for running the extractor. The
  §10 worked figure goes through `figure_style.py` ALONE — `paper_grid`,
  `panel_label`, `apply_style`, `save_figure`, `STANDARD_COLORS`. Worked
  example is the **DD2 + vMIT hybrid end to end** (pairing -> point ->
  rows+windows -> TOV -> response -> figure), deliberately not the README's
  pure DD2: `M_max = 2.254` against pure DD2's `2.419`, identical `R(1.4)`,
  the transition invisible at canonical mass. All 48 relative links and all
  13 TOC anchors check. **Ticket 11's carried-in half was already done** by
  [ticket 64](issues/64-general-verify-suite-missing.md); its suite output is
  pasted into §8. **One defect, reported not fixed**:
  `eos/mixed/api.py:eos_response` does not forward `phases=`/`muons=` to its
  central point, so the general two-`Phase` calling form returns
  `converged=False` and nan everywhere while the front door returns
  `chi=0.373300` — Stage 7. Blocks re-verified after a second session went
  live on 30 files: every physics number byte-identical, only the ms timings
  move. python.org 3.14.2; full suite deliberately not run.

- [Rename did's and dd2's phase-adapter surface to thermo_from_mu](issues/48-rename-did-surface.md):
  done, commit `2891715`. did's lower layer -> `thermo_from_fields` FIRST, then
  both surfaces -> `thermo_from_mu`. **All ten models now spell the surface the
  same way**; sfho and did are the only two with a lower layer. **The ticket's
  collision prediction was wrong and the real one is 3x bigger**: sfho's and
  did's imports in `eos/mixed/adapters.py` are ALIASED and never collide; the
  actual colliders are three BARE function-local imports it never names —
  `enjl:426`, `njl:1081`, `ccdm:1231`, shape 2 not shape 3, proved in a
  scratchpad sandbox before touching the repo. Aliasing dd2's to `_dd2_at_mu`
  clears all three. **`\b` is not supported by BSD sed** — the first rename pass
  was a silent no-op at exit 0, caught only by grepping after instead of
  trusting; tickets 42-45 all describe `sed` without flagging this. Working BSD
  spelling is the boundary CLASS, `s/[[:<:]]NAME[[:>:]]/NEW/g`. **The exposure
  was then checked and is GREEN**: all 11 old names from 42-45 grepped clean
  across `eos/` (only cosmetic hit `sfho/parameters.py:662`, a printed label,
  45's territory). Real trap, cost this ticket one pass, bit no earlier ticket. Five
  document passages that the rename made FALSE were corrected (`dd2.md`,
  `sfho.md`, `sfho.tex`, `mixed.md`, `mixed.tex` each said dd2 was "the
  outstanding one"). Evidence: **904 probe values bit-identical by exact `==`**
  (deep walk of the whole returned structure, not a hand-listed field set),
  full suite **byte-identical** to `pytest_after_ticket56_py314.txt` — same 12
  node ids, `diff` of the 121 `^E ` lines EMPTY, 0 added / 0 cleared — and
  did/dd2/mixed `verify` all PASS. Run on **3.14**, stated as a comparability
  choice, not a vote on [57](issues/57-canonical-stack.md). Baseline gate met in
  56's wording: no SURVIVING value moved, and `test/baseline/` never written to.
  First suite run was killed and discarded for spanning a re-indentation edit,
  the same contamination [45](issues/45-rename-sfho.md) paid for.
  `test/` edits (6 sites in did's tests, 4 in dd2's) are gitignored and live
  only in the working copy.

- [Which Python/numpy/scipy stack is canonical](issues/57-canonical-stack.md):
  **python.org 3.14** — 3.9 is end-of-life and both `pyproject.toml` files say
  `requires-python = ">=3.9"` while `nucleation` heads for a public remote, so
  pinning 3.9 would ship a library needing a dead interpreter. The cause was
  measured, not attributed to version numbers: the two stacks compute on
  **different BLAS** — anaconda on OpenBLAS 0.3.23, 3.14 on Apple Accelerate —
  plus scipy 1.13→1.17 solver internals and numpy 1.26→2.3. Each is ~1e-16 at the
  operation, but an iterative solve at a 1e-10 residual lets a last-bit difference
  flip its stopping iteration, so the answer moves at the scale of the gate —
  which is why only `test/baseline/` fails. Execution graduated to
  [ticket 62](issues/62-regenerate-baselines-py314.md), with a stop condition:
  any difference larger than round-off halts the regeneration rather than being
  absorbed into a new ground truth.

- [Non-convergence escapes as an exception at seven public boundaries](issues/49-nonconvergence-return.md):
  **eleven sites, not seven** — all now return the same dict the converged path
  returns, with `converged=False`, NaN in every quantity and the `reason` the
  exception carried, built by one new `general/tabulate.unconverged_response`.
  The ticket's SnB half was two sites (njl, ccdm), not seven: the other five
  already caught. Its four "already compliant" models were the bigger half —
  `sfho`, `did`, `njl` and `ccdm` all leaked a `RuntimeError` out of
  `eos_response`, each with a `responses.py` docstring asserting the api layer
  caught it, which it did not. `dd2` unverified by the session constraint (its
  route is the analytic Jacobian, not a stencil). 1055 collected, 0 failed
  across eleven suites on python.org 3.14.2; verify PASS 9/9; 0 added failures,
  and the intersection with the 12 known 3.14 failures is empty.
  Two findings graduated rather than fixed:
  [ticket 63](issues/63-verify-causality-nan-silent-pass.md) (the NaN this fix
  lets five verify checks absorb) and
  [ticket 64](issues/64-general-verify-suite-missing.md) (`general/` has no
  `verify/`).

- [notebooks/hadronic_eos — skeleton, knobs, modes and parametrisation](issues/12-hadronic-skeleton.md):
  **shipped and executing.** `notebooks/hadronic_eos.py` paired to `.ipynb`
  (`e81e034`), and the code this ticket owed the other three notebooks —
  `standard_name` / `table_path` in `eos/general/table_io.py` (`407c984`),
  reproducing all three of ticket 04's names byte-for-byte. `jupytext --to
  notebook --execute` completes with 0 error outputs, verified BOTH on the live
  tree and in an isolated `git archive HEAD` copy, because two other sessions
  hold `eos/*/api.py` and `general/fermi_integrals.py`; the two runs differ only
  in round-off (sfho Q_sat by 9e-4). 323 collected / 323 passed on
  `test/general` + `test/test_imports.py` in the HEAD copy, python.org 3.14.2.
  Three adaptations of the spine, each forced by this ticket's own deliverable
  list and none of them a re-decision of the three shapes: `mode` -> `modes` with
  `conditions(mode)` (a section per mode needs more than one), species flags
  built INSIDE the section (`SpeciesFlags(**six)` is itself a refusal site, so
  the knobs cell would die before the reporting pattern could run), and a
  four-line path bootstrap (`eos` is installed in neither stack).
  **`leptons` governs `eos_point` only**: `eos_table` takes it in zl and did and
  `TypeError`s in sfho and dd2, and passing it to two of four would be exactly
  the per-model translation table ticket 04 blocked this ticket to avoid — so it
  waited on [ticket 54](issues/54-signature-corrections.md) item 1, **which
  landed in `4434768` and the follow-through with it (`16181e6`)**: the `build`
  closure names the flag under the same §3 rule as the points. That rule also
  sidesteps a divergence 54 did not level — on a beta-eq mode `sfho` and `dd2`
  raise for `leptons=True` where `zl` and `did` accept it. Re-verified against
  HEAD carrying 54: 0 error outputs, 327 collected / 327 passed.
  **Four real unrecorded gaps, none fixed**: did reports meson condensation as
  `residual 5.684e-15 above the gate` when its gate is 1e-10 (the cause is
  `condensation >= 1.0` at `did/solver.py:574`, and `did/api.py:99` prints the
  residual either way); zl's `EoSPoint` spells the totals `P_total/e_total/s_total`
  against `P/eps/s` in the other three (the `rows_from_result` schema, by
  contrast, is uniform — which is why every grid goes through rows); `did`'s
  `compute_nmp` returns its own key names; and the inverse map has no shared
  calling convention (`dd2.invert_nmp(nmp)` positional, `sfho.invert_nmp(**nmp)`
  keyword), with `sfho.from_nmp` raising `RuntimeError` where `invert_nmp`
  returns a status.
  The first of the four is now [ticket 66](issues/66-did-condensation-message.md);
  the other three are naming rows and are Stage 7 report material.
  Environment: `nbconvert 7.17.1` installed into python.org 3.14 — jupytext's
  executor, which this ticket's done-when names; `pyproject.toml` untouched. And
  `timeout python3` runs under Rosetta and dies on numpy's arm64 extensions.

- [general/ owes a public T = 0 entry point, and one loop is unbounded](issues/52-general-t0-integrals.md):
  the door is open and the loop is bounded — `_compute_exact_T0` promoted to
  **`solve_fermi_t0`**, the third solver of the `solve_fermi_*` family (same
  jitted body, so its three existing callers are a rename apart and no number in
  the repository moves), and `invert_fermi_density`'s two bracket loops now stop
  at `_MAX_BRACKET_STEPS = 200` and **return NaN** rather than raising (§6).
  `ffae9db`, gated in an isolated copy beside a HEAD control on anaconda 3.9.7
  because the live checkout had a concurrent session editing `dd2/api.py`
  mid-run: **137 passed in both**, `test/general` plus the dd2, zl, vmit and enjl
  baselines at rtol = 1e-10. **dd2's half was implemented, measured and
  reverted**: of 4692 stored dd2 quantities 3434 come out bit-identical and every
  EoS quantity moves under 5.9e-15 relative, but the finite-difference NMP map
  amplifies last-bit noise by 1e6–1e8 — `nmp.Q_sat` moves 3.6e-4 (0.061 MeV),
  `K_sat` and `K_sym` ~2e-8 — so `test_baseline[dd2]` and one knife-edge
  inversion test fail while the golden SNM(0.16) point and CompOSE HS(DD2) do not
  move at all. No reformulation makes it bit-exact (fastmath, `mu` against a
  round-tripped `EF`, and an `hc3` round trip), so the remaining half is a ruling
  rather than a task: [ticket 67](issues/67-dd2-t0-adoption.md), blocked by
  [62](issues/62-regenerate-baselines-py314.md). Two ledger corrections fell out:
  the loop cannot hang even on a non-finite target (`n_hi` overflows to `inf` and
  the comparison goes False — the worst case is a ~1700-iteration spin), and its
  UPPER bracket cannot be exhausted by any finite target, since it opens at twice
  the T = 0 estimate.

- [The public-signature corrections §5 and §3 require](issues/54-signature-corrections.md):
  **all five rows landed, 0 added failures, `test/baseline/` unmoved.**
  `leptons` is a named argument on all three entry points in dd2, sfho and did,
  and **`fixed_YC_neutral` / `fixed_YC_YS_neutral` no longer exist anywhere in
  the repository** — the flag rides beside the mode (`takes_leptons` in `MODES`,
  a `leptons` field on both `TableSpec`s), so ticket 20's condition is met on its
  own terms and **ticket 12's one-line change is now unblocked**. `mode` lost its
  default at nine sites in njl/ccdm/enjl (abpr keeps `"cfl"`); an AST sweep finds
  0 call sites left without one. `zl.thermo_from_n` takes `(n_n, n_p, T, params)`
  and the §5 purity grep over `eos/*/thermodynamics.py` now returns no
  non-docstring hit. `TC_COEFF` is `Parameters.tc_coeff`, threaded through the six
  CFL gap functions so the override reaches T_c.
  **Item 5 needed the COUNT, not just the raise.** Deleting sfho's and did's
  refusal alone would have shipped the double-count it warned about: the three
  conformant models multiply by `3 - (1 if the mode holds Y_Le)`, so adopting
  their answer means adopting their count. Both models now do. `sfho.md`/`.tex`
  already described this resolution as due — they only had to catch up.
  **Measured as an isolated PAIR**, because ticket 52 held three files in the
  live tree for most of the session (since landed, `ffae9db`/`f1484b6`): control
  = HEAD `407c984`, mine = HEAD + this ticket's 22 files. **3 failed / 1291
  passed** against **3 failed / 1297 passed**, failure sets identical node id by
  node id, the three being ticket 47's dd2 NMP stack artifact; `test/baseline/`
  6 failed / 10 passed in both. The +6 are this ticket's own new tests. Eight
  `verify/` suites green. python.org 3.14.2 / numpy 2.3.5 / scipy 1.17.0,
  collection **1692**.
  **Bit-identity proved, not assumed**, for all three number-touching items:
  `leptons=True` reproduces `fixed_YC_neutral` 12/12 hex rows in dd2 AND sfho;
  zl's `thermo_from_n`/`nuclear_matter`/`compute_nmp` 52/52; alphabag's whole
  CFL sector 33/33 on the `tc_coeff` default.
  **The `_check` guard caught a live caller on its first run** —
  `test/dd2/test_photons_flag.py` was passing `fixed=dict(Y_C=0.1, leptons=True)`,
  the exact smuggling ticket 20 predicted, forced because `eos_table` had nowhere
  else to take it.
  **Two findings NOT fixed**, per the hard rule: njl and ccdm carry item 1's
  defect too (the audit read `njl:122`, which is `eos_table`, and both models'
  `eos_table` IS conformant), and `leptons=True` on a beta mode still gets three
  different answers across six models. Both are
  [ticket 68](issues/68-njl-ccdm-leptons-condition.md).

- [`did` reports meson condensation as a residual below its own gate](issues/66-did-condensation-message.md): the condensation rejection got its
  own words, in `eos/general/thermal_mesons.py` so it exists once. The
  root-cause sweep found the defect in **three** sites across **two** models —
  `sfho/responses.py` had it too, and `sfho` was the ticket's model to copy.
  Only `did` and `sfho` clear `converged` for a second reason at all. The
  condensation onset is a curve near 0.302 fm^-3 that moves with T, the
  species flags and the couplings, not the `n_B <= 0.2` domain the ticket
  guessed at.

- [notebooks/hadronic_eos — the six figure families and the TOV pass](issues/13-hadronic-figures.md):
  **all six shipped and executing** (`77c2976`). The whole notebook — sections 1
  to 7 — runs with **0 error outputs**, 56 cells / 30 code cells, ~45 s, verified
  in an isolated `git archive HEAD` copy; 327 collected / 327 passed on
  `test/general` + `test/test_imports.py` there, python.org 3.14.2 /
  matplotlib 3.10.9. Twelve files in `output/hadronic/`.
  **The panels ARE the sector selection** — one panel per entry of `FIG_SECTORS`,
  all four hyperon/Delta combinations exercised, two shipped selected — and the
  sector carries the parameter set it needs, which section 5 had already
  established it must. §8's gate differences the delivered table before
  integration and returns a status; all seven (model, sector) tables PASS, so
  nothing had to be held back and nothing was repaired. M_max 1.99–2.42 M_sun.
  `truncate_to_stable_branch` is **not** used: it re-orders to six columns and
  drops `Lambda`, so the notebook slices at `find_mmax_precise`'s index instead.
  **One library fix was forced** (`dfe9695`): every negative decade on every log
  axis in the house style rendered as a hollow box — matplotlib emits log tick
  labels as `$\mathdefault{...}$`, mathtext turns a hyphen into U+2212, and it
  resolves `\mathdefault` through CMU Serif, which has no U+2212 glyph.
  `mathtext.fallback`, `mathtext.default`, `mathtext.fontset` and a
  `font.family` fallback list all fail to reach that path; ordinary mathtext
  renders, so `figure_style.log_decades` was added beside the existing
  protection. §10 forbids rcParams in a notebook, so the single home was the only
  lawful place.
  **One real unrecorded gap**: `zl` and `dd2` return the sound speed as
  `cs2_eq` — a name for the FREEZE — where `sfho` and `did` return
  `cs2_isothermal`. At T = 0 it is the same number; at T > 0 `cs2_eq` is exactly
  the bare name §5 forbids. Not fixed: a public return-key rename across two
  models.
  The `table_path` relative-root bug is closed for this notebook at all four
  write sites; the isolated run created no `notebooks/output/`.

- [njl and ccdm smuggle `leptons` through `**conditions` too](issues/68-njl-ccdm-leptons-condition.md):
  both models take the flag as a named argument on all three entry points and
  nothing in `eos/` reads it out of a condition bag any more. The effective
  default was **`True`** in both — not the `False` sfho/dd2/did use — and was
  kept, so no number moved. The beta-mode half is now
  [ticket 70](issues/70-leptons-on-a-beta-mode.md), with njl/ccdm's answer
  measured as the OPPOSITE of sfho/dd2's on both values rather than the
  "whatever `solve` does" 54 recorded.

- [notebooks/enjl — skeleton, knobs, figures and the author-table reproduction](issues/18-enjl-notebook.md):
  **shipped as `notebooks/enjl_eos.py`** (the ticket wrote `notebooks/enjl`; its
  three siblings are `hadronic_eos`, `quark_eos`, `hybrid_eos`, so all four now
  spell the same thing). 44 cells, **0 error outputs**, in a `git archive HEAD`
  copy at `77c2976` with the kernel started in `notebooks/`; `test/enjl` +
  `test/test_imports.py` **317 collected, 317 passed**; python.org 3.14.2. No
  library code touched.
  **The author's tables are reproduced and the residual printed for all five
  sets**: `P` median 6.9e-07 to 1.7e-06, and the max is 9.94e-06 at `n_B` = 0.10
  — the LOWEST density, where `P` = 0.46 MeV/fm^3 and the ratio is an absolute
  agreement of ~5e-6 divided by it. Two sets exceed 1e-04, on their last one or
  two rows only, and those rows are the approach to that set's own coexistence
  endpoint, where the author is already following a construction and we are
  still following a branch. Both numbers are printed rather than a window tuned
  to hide the effect. The notebook **parses the `.dat` itself and imports
  nothing out of `test/`**, which `docs/DEFERRED.md` requires of the replacement
  enjl notebook.
  **The branch pair is `direction="up"/"down"`** and only 5-6 of the 24-27
  overlapping densities are two distinct states; above the transition the two
  continuations converge on one root, so the first draft reported the sign of a
  1e-12 difference as "which branch is stable". A `1e-8` tolerance now separates
  them. `enjl_branch_pair` is stated in one line to belong to
  [ticket 58](issues/58-hybrid-skeleton.md); `eos.mixed` is never imported.
  **The `table_path` root bug is confirmed by counterexample**, not assumed:
  with the kernel in `notebooks/` the default returns
  `<root>/notebooks/output/tables/...`, and with `root=ROOT/"output"/"tables"`
  everything landed at the repository root. Tickets 15 and 58 owe the same
  argument.
  **A concurrent session moved HEAD mid-verification**: `fs.log_decades` was
  committed by another session (`dfe9695`) BETWEEN the `git archive` and the
  `AttributeError` it caused. The archived SHA must be recorded with any
  archive-copy result. Side effect: `quark_eos.py:690`'s two-line
  `set_yticks`/`set_yticklabels` workaround for the same glyph hazard can now
  collapse into one `fs.log_decades` call.

- [Five verify causality checks absorb a NaN and report PASS](issues/63-verify-causality-nan-silent-pass.md):
  five local `np.isfinite` guards, **not** a shared helper — every verify suite
  declares its own `CheckResult` and is meant to be read standalone. Measured
  first: **three** of the five absorbed the NaN and even printed a clean c_s^2
  range over a list containing it; `did` and `njl` already failed, but by
  accident of NaN comparison semantics and without naming the density. All five
  now FAIL with the density. 121 checks across **eleven** suites still pass.

- [notebooks/quark_eos — the benchmark section](issues/17-quark-benchmark.md):
  **shipped as section 9**, commits `d0445fb` + `f2dee22`; verified by
  `jupytext --execute` in an archive copy of the COMMITTED tree at `d0445fb`
  (53 cells, 25 code, **0 error outputs**), `test/general test/test_imports.py
  test/{vmit,alphabag,njl,ccdm,abpr}` **809 collected, 809 passed** in a second
  archive copy, python.org 3.14.2, no library file touched.
  **The finding is a three-order-of-magnitude spread**: `alphabag`/`vmit` solve a
  cold point in well under a millisecond, `njl` in tens of milliseconds, `ccdm`
  in ~0.1 s. The section costs ~10 minutes to execute and essentially all of it
  is `ccdm`, which the notebook says up front.
  **`ccdm`'s warm column is 10-13 s/pt against a ~0.1 s cold point** — two orders
  above its own cold cost, where every other model's two numbers sit within a
  factor of three. Nothing solved got slower: the line pays for the 7 points it
  never solves, each retried through `MAX_BISECT = 6` halved steps with a full
  candidate enumeration per retry, and those attempts are in `elapsed_s` but not
  in `n_solved`. This is exactly why the ticket asked for cold and warm
  separately.
  **The misses are in different places and the densities say so**: `njl` misses
  1 of 24 at the TOP (3.0 fm^-3, its cutoff domain), `ccdm` misses 7 in an
  INTERIOR band (0.178-0.948, its sub-onset region) — identically in both
  configurations. Reported not diagnosed: `ccdm`'s first density 0.05 fm^-3
  *solves* while the band above it does not, so the window is interior on both
  sides and whatever root lives down there is worth a look.
  **No quark model ships a `backends/`** — checked in executed output for all
  five, not asserted — which is also the cause of the profile's ~50 residual
  evaluations per point (finite-difference Jacobian). The `abpr`-vs-`alphabag`
  CFL timing (0.170 vs 3.358 ms, ~20x) is shown beside it and explicitly labelled
  a pair of MODELS, not of backends.
  **The `table_path` root bug is confirmed a third time** (after tickets 18 and
  58) and worked around with `root=str(ROOT / "output" / "tables")`. Section 4's
  save is left on the relative default on purpose: ticket 15 ruled the fix
  belongs in `table_io.py` uniformly, and one notebook does not get to overturn
  that. Noticed alongside: `standard_name`'s `_span` uses `%.1f`, so a grid
  starting at 0.05 renders `nB0.1` and two grids differing only below 0.1 fm^-3
  would collide — ticket 04's to fix.

- [notebooks/hybrid_eos — skeleton, knobs, the pairing choice and the tables](issues/58-hybrid-skeleton.md):
  **recovered, not written here.** The session that did the work marked the
  ticket resolved and stopped without committing; both files sat untracked for
  two hours. Committed AS FOUND in `bbf07f9` after executing the `.py` in an
  archive copy of HEAD (12 code cells, **0 errors**), so the rescue is
  distinguishable from a rewrite, and the answer reconstructed in `9de1b98`
  and marked as such. The done-when holds: DD2 + vMIT converges, 11 in-window
  rows, windows carried into the file through `save_table(windows=...)`. Two
  claims of the original outcome did NOT survive re-measurement — `hadronic_eos`
  does not still write under `notebooks/output/` (ticket 13 closed it), and the
  one table that had landed there came from a draft run predating the `root=`
  fix by five minutes.

- [notebooks/hybrid_eos — figures, the TOV pass, and the swap cell](issues/59-hybrid-figures.md):
  **shipped as sections 8-11**, commit `156384f`, 26 code cells **0 errors**.
  Four figure families with the panels as the *eta* selection; the §8 gate
  returning a status before integration (P tested **non-decreasing**, since a
  Maxwell window is an exact plateau and a strict test would reject the most
  clearly correct construction) then M_max = 2.249/2.339/2.343 M_sun; DID+NJL
  and DID+CCDM swapped with the skipped depth printed.
  **This ticket's premise was stale and the comparison is now numeric.** The
  retired tables never left: `eos_tables_DD2vMIT/` is tracked at the repository
  root, in HEAD. Each CSV carries the run's full provenance in a
  `# key = value` header whose keys map 1:1 onto `eos.dd2.Parameters` fields —
  a CUSTOM parametrisation, not a published one — so the inputs are rebuilt
  exactly and **the engine reproduces the retired transition boundaries to
  under 0.5% at all three eta**. That is a far stronger end-to-end check than
  the eyeball the ticket settled for, and it discharges ticket 05's comparison.

- [notebooks/enjl — step-by-step treatment and benchmarks](issues/19-enjl-stepwise.md):
  **shipped as sections 9 and 10 of `notebooks/enjl_eos.py`**, commits `5aae00b`
  + `f8a0c79`; verified by `jupytext --execute` in an archive copy of the
  COMMITTED tree at `f8a0c79` with the kernel in `notebooks/` (76 cells, 38
  code, **0 error outputs**, 1 min 52 s), `test/enjl test/test_imports.py`
  **319 collected, 319 passed** in a second archive copy, python.org 3.14.2, no
  library file touched.
  **Ticket 16's five steps: ENJL has three, answers one with a different object
  and has nothing for one.** No diquark channel in the functional, so no gap to
  map and no pattern to select — proved by listing the parameter and species
  fields and scanning them, whose one hit is `deltas`, the Delta(1232) flag, a
  false positive printed rather than filtered. `M_u` and `chi` are mapped over
  `(n_B, T)` where `Delta` would be, and the chiral **bracket** is reported per
  temperature rather than an interpolated crossing, because `M_u` falls 200 MeV
  between adjacent grid densities.
  **A branch pair is not a pairing pattern** — two roots of one set of
  stationarity conditions, not two ansaetze — and at fixed `(mu_B, T)` the
  criterion is the higher `P`. That is a DIFFERENT criterion from section 5's
  `eps` at fixed `n_B`, and the two land in **adjacent grid intervals in every
  set**: the gap between them is the coexistence window, where neither pure
  branch is stable. Ticket 18's trap recurred here in a new dress: a naive
  crossing hunt printed **22 "Maxwell conditions" for one set** from
  interpolation wiggles above the transition, fixed by restricting to section
  5's distinct-state densities. Under it, `mu_B` is **not monotone** along a
  branch through the transition (the swallowtail), so nothing may be sorted by
  it; the cell prints which branches walk backwards.
  **The benchmark carries an axis its two siblings do not: the branch.** `"up"`
  and `"down"` differ by ~20x in wall clock (0.24-0.40 s against 3-10 s for the
  same 24-point line) and every non-converged point in the whole section is on a
  `"down"` line. **Cold vs warm is a difference in kind for a continuation
  model**: the cold start fails in a BAND between 0.6 and 1.0 fm^-3 — sharper
  than the docstring's "around 0.5" — and converges again above it. Bottleneck:
  **84,833 residual evaluations for 24 densities** under
  `approx_derivative`; `eos/enjl` ships no `backends/`, which is checked, and
  that IS the reason.
  Findings reported not fixed: `eos/enjl/api.py:106` documents
  `eos.enjl.solver.UNKNOWNS`, which does not exist (it is `BASE_UNKNOWNS`) —
  belongs with [ticket 54](issues/54-signature-corrections.md); and `fq0.5_B1`
  at fixed `Y_C` on the `"down"` branch misses an INTERIOR density band.

- [notebooks/quark_eos — the NJL and CCDM step-by-step section](issues/16-quark-stepwise.md):
  **shipped as section 10**, commit `2d9d7e8`; verified by `jupytext --execute`
  in an archive copy of the COMMITTED tree (86 cells, 43 code, **0 error
  outputs**, ~14 min, of which section 10 is ~5), targeted tests **811
  collected, 811 passed** in a second archive copy — 809 in tickets 15/17, +2
  from another session, **0 added** — python.org 3.14.2, no library file
  touched. Five new figures, and the two colormaps it needs are BUILT from
  `figure_style`'s palette rather than picked from matplotlib's.
  **In neutral matter CFL wins wherever a gap survives**, at fixed `mu_B` and at
  fixed `n_B` alike, in both models. **`P` does not rank phases at fixed
  density and the printed table shows it failing to**: `njl`'s 2SC branch has
  the higher pressure while CFL has the lower free energy and is the ground
  state — the branches sit at different `mu_B` at the same `n_B`.
  **A three-root trap was hit and fixed as physics, not tuning**: a
  density-only warm start put `ccdm`'s 2SC row at T = 40 on the trivial gap root
  and carried the zero across the whole line, printing a gapless band between
  two gapped ones. The sweep now continues in temperature as well, which also
  takes `njl`'s 2SC line from 6/8 to 8/8 points and 13 s to 5.7 s.
  **The document comparison is IN the notebook (10.3c), not only in the answer**,
  against `njl_csc_implementation.md` section 6's own neutral solve at
  `mu_B = 1500` MeV. Every entry of both rows matches to the document's printed
  precision **except the 2SC light constituent masses** (doc 11.96/7.65, code
  9.72/8.90) — and `Delta_3`, `mu_8`, `n_B` and `P` all agreeing to four digits
  is what says the document's two entries are wrong, not the code's. Three more
  disagreements reported, code deciding each: the CFL neutral solve is flagged
  "not tightly converged (residual 13)" in that document's sections 6.3 and 11
  and **is converged now** (residual 4e-16, 41 of 42 grid cells);
  `ccdm_implementation.md` section 6.5 says gapless states must be excluded from
  the Omega minimization and **both solvers rank them** (`njl/solver.py:601`,
  `ccdm/solver.py:678`), which is live at the probe point where `njl`'s 2SC state
  is gapless; and the document puts leptons inside `Omega` where the code keeps
  them out of the phase, same totals, different boundary.
  Reported not fixed: `njl`'s `eos_response` cannot reach n_B = 2.2 fm^-3 for
  `unpaired`/`2SC` (stencil neighbour at 2.2022 does not converge), and `ccdm`'s
  returns `branch_changed=True` there even with the pattern restricted.

- [The eight conformance fixes that move no number](issues/50-mechanical-fixes.md):
  **seven landed as one commit `5c75584`, the eighth cannot be committed** —
  `ccdm` was missing from `test/test_imports.py MODEL_PACKAGES` and is now in it
  and passing, but `test/` is gitignored, so that fix exists only in the working
  tree. The grouping held: `test/baseline/` at rtol = 1e-10 shows zero movement,
  `alphabag` and `mixed` included, so items 6-7's charge-map dedup was arithmetic
  identity as predicted. Three rulings the code forced. `fracs` is fixed by
  passing the builders' own dict and **`combos` is deliberately left narrow**,
  because it feeds `rows_from_result` and widening it would move table content
  rather than progress. `_mode_kwargs` **cannot move alone** — `MODES` and
  `MODE_FRACTIONS` moved to `solver.py` with it — and the dd2 constructors could
  not either, being public names with callers in `mixed/scan.py` and five test
  files including `generate_baseline.py`. And **`abpr` gets the docstring, not
  the vectorisation**: array-in/array-out is genuinely reachable there and
  nowhere else, but it is a change to `solve_cfl`'s signature, so it needs a gate
  on numbers moving, which is what this ticket was defined by not having. Gate on
  **python.org 3.14.2**, in an isolated `git archive HEAD` copy with two sessions
  live: baseline 16 collected / 6 failed, targeted 1083 collected / 3 failed,
  **all nine pre-existing and named in `output/_audit/`, zero added**.

- [`eos.enjl.api` documents a name the module does not export](issues/71-enjl-unknowns-docstring.md):
  fixed in `4d0ab58`, and **the grep the ticket asked for found two more**. The
  named defect was worse than a wrong order — `unknown_slots(spec)` returns 10
  slots for `beta_eq_neutrinoless` and `fixed_YC` but **11** for `fixed_YC_YS`
  (+mu_S) and for the trapped mode (+mu_nue), so a guessed vector is the wrong
  LENGTH, not merely mis-ordered. Resolving every backticked `eos.*` name in
  every public docstring under `eos/` by import: 17 unresolvable, **13 of them
  CompOSE data filenames** (`eos.nb`, `eos.t`, `eos.thermo`, `eos.yq`) that only
  look like modules, and four real — the target, plus
  `eos.enjl.verify.check_entropy_limit` sixty lines away in the same package
  (real function, wrong module), `eos.vmit.solver_table` (never existed; it is
  `eos.vmit.eos_table`), and `eos.astro.tov.solver.load_crust_table` (it is in
  `.crust`). The first three fixed; **the fourth left alone and reported**,
  astro not being one of the nine models the ticket opened. A second pass over
  BARE backticked identifiers in every `api.py` found nothing: the dotted form is
  where this class of defect lives. Suite 165 collected / 164 passed; the one
  failure is ticket 69's rename half-landed across the shared gitignored `test/`
  tree, green when that session's `eos/vmit/api.py` is overlaid.

- [`cs2_eq` names the freeze where §5 requires the thermal variable](issues/69-cs2-eq-naming.md):
  **closed by renaming the key to `cs2_isothermal` in `zl`, `vmit`, `alphabag`
  and `dd2`; a grep for `cs2_eq` over the ten models now returns nothing.** The
  freeze is not lost — §5's three axes are properties of the CALL, and the
  composition axis is the `frozen=` argument, still there; the key names the one
  axis a caller cannot recover from its own arguments. Swept through each
  `verify/`, the `.tex` AND `.md` documents, `test/`, and both notebooks that
  carried a two-key reader, which lose it. Three docstrings gain §5's named gap:
  `zl`, `vmit` and `alphabag` compute no adiabatic speed, having no `C_P` to
  form it with. **Two dd2 documents were stating something false** and are
  corrected rather than renamed — they called the `composition` freeze's speed
  "the adiabatic one", but `responses.py` holds T on both stencil points, so it
  is isothermal at frozen composition; the rename made that visible rather than
  causing it. **Three sites deliberately NOT renamed**, each for a physics
  reason rather than scope: `mixed` and `gmode` are one surface sharing
  `sound_speed_eq`/`sound_speed_frozen`, so half a rename is worse than none
  ([53](issues/53-gmode-contract.md), noted there); `dd2`'s `TableResult.cs2_eq`
  is isothermal on a `T` axis and ADIABATIC on an `SnB` axis, so it can be
  renamed to neither; and `dd2`'s `cs2_ad` is a DIFFERENT quantity from the
  `cs2_adiabatic` four models already return
  ([73](issues/73-dd2-remaining-cs2-names.md)). The ticket's own "grep returns
  nothing over `eos/`" condition is therefore not met and should not be.
  Gated as an isolated HEAD-vs-HEAD+patch **pair, twice** — a concurrent session
  moved HEAD mid-ticket, and its first build was contaminated by a swept-up
  `_mode_kwargs` import hunk that added two false failures until stripped.
  Baseline and all four verify suites identical on both sides at both HEADs;
  **no number moved, no failure added**. The dd2 half of the rename landed
  inside `5c75584`, a concurrent session's commit, not in `5a4a6cc`.

## Not yet specified

In scope, not yet sharp enough to ticket:

- **Nothing in this repository notices when a stated limitation stops being
  true.** Three instances surfaced in one afternoon, each found by accident
  while someone was doing something else, and each is a comment that outlived
  the behaviour it described:

  - `eos/dd2/nmp.py` claimed a 5x5 round trip "returns the published couplings
    unchanged". True only while SciPy 1.13 declined to iterate; the published
    couplings are not a root of the closure at all
    ([ticket 47](issues/47-dd2-nmp-inversion.md)).
  - `eos/__init__.py`'s species comment said "dd2 is exempt" where the code
    implemented no exemption — just an inclusion list of seven, which silently
    skipped `enjl` and `abpr` too ([ticket 61](issues/61-dd2-species-flags.md)).
  - `test/test_imports.py`'s species check could only fail in the PESSIMISTIC
    direction. Closing 61 would have made dd2 conform and sailed straight
    through, leaving the README and the `#:` comment describing a gap that no
    longer existed.
  - `docs/DEFERRED.md`'s dd2 section carried the SAME claim as a deferral, and
    the two-way gate built for 61 does not name it — the gate lists three prose
    sites and there were four. Found while closing 61, by grepping for the old
    flag names rather than by anything firing. §11 calls that file the tracked
    ledger of per-model gaps, which makes it the one place a closed gap left
    open is most load-bearing, and it is the site with the least protection.

  The common shape: we are well drilled at asserting nothing got worse, and
  have almost nothing that fires when a documented gap is CLOSED. Every
  `DEFERRED.md` entry, every "not yet implemented" raise and every §4 gap
  disposition is a claim of this kind, and none of them is checked. The
  two-way exemption gate eos-88 built for 61 is the first instance of the
  countermeasure — the test goes red on purpose when the limitation lifts, and
  its message names the prose that must move with it.

  Unsharp in two ways, which is why this is fog: whether the general form is a
  `verify/` check, a `DEFERRED.md` convention, or a §12 sentence; and whether
  it can be mechanised at all, since "this raise still raises" is testable
  while "this prose still describes the code" is the document-verification
  problem already in the entry below. Both need
  `eos/general/verify/`, which does not exist —
  [ticket 21](issues/21-phase5-structure.md) is where that lands.

- **The model documents are unverified in two ways, and both were caught by
  chance rather than by a check.** Found while tickets 35 and 45 ran
  concurrently, one on each side of the same files.

  *Signs.* `zl.tex` PASSED the document audit 14/14 and still had the
  neutrality row backwards in one of three modes — the code writes
  `n_C - n_e` in both beta modes and `n_e - n_C` in `fixed_YC`, and the
  document stated one row. `vmit`'s R6/R7 swap between its two beta modes is
  the same shape. Two of two checked were wrong, so the remaining ten pairs'
  residual-row signs should be assumed unchecked rather than clean.

  *Dispositions.* A document says of each gap whether it is deferred, a
  defect, or by design — and that label is a claim about `CLAUDE.md` and the
  triage rulings, not about the code. `sfho.md`/`.tex` shipped
  `thermal_neutrinos` + trapped as a ledgered §4 gap pointing at
  `docs/DEFERRED.md`, when [ticket 11](issues/11-conformance-triage.md):135
  had ruled the row (b)+(a), [ticket 22](issues/22-phase5-claudemd.md) had
  ALREADY landed §4's "a model must not raise on the combination", and
  [ticket 54](issues/54-signature-corrections.md) deletes the raise. The
  pointer named a `DEFERRED.md` entry that does not exist. Corrected in
  `463278a`. The structural cause is that the document tickets run against a
  specification other tickets are concurrently editing, so verifying a
  document against the CODE is only half the check.

  Whether this is one sweep, one ticket per pair, or a check added to the
  `verify/` suites is not decidable until someone measures how many of the
  twelve are affected.

- **Several real fixes now live outside version control.** `.gitignore:75`
  excludes `/test/` entirely (§11), so ticket 39's helper skip, ticket 40's
  completed baseline exclusion and now [ticket 56](issues/56-baseline-empty-sector-gate.md)'s
  gate correction exist only in a working copy — anyone reconstructing `test/`
  reintroduces all three bugs. Whether some of that logic
  belongs in `eos/` where it would be tracked, or whether the layout rule should
  bend, is worth settling during [ticket 21](issues/21-phase5-structure.md).
  **Decided, for now (user, this session): §11 stands unchanged and
  `test/baseline/` is copied OUTSIDE the repo by hand.** Tracking `test/` was
  weighed — 44 MB total, of which 13 MB is the irreplaceable `.npz` and ~25 MB
  regenerable `zlvmit` fixtures — and rejected for now because §11 says `test/`
  is not published and un-ignoring it would publish the suite with the repo.
  **Two caveats the interim measure carries:** a hand copy is only as fresh as
  the last time it was taken, and [ticket 57](issues/57-canonical-stack.md)
  regenerates every `.npz` — so the copy is worth taking AFTER that lands, not
  before, or it preserves the superseded set. The underlying question — whether
  some of this belongs in `eos/` where it would be tracked — is still
  [ticket 21](issues/21-phase5-structure.md)'s.

    **Ticket 56 escalates this from lost logic to lost DATA**: besides the gate
  fix, its four regenerated `.npz` files exist only in this working copy. Lose
  the checkout and the absolute-vs-relative gate bug returns *and* brings the 34
  keys back with it — and `test/baseline/` is §12 ground truth, so what is
  unrecoverable here is a golden reference, not a convenience.
  A third shape, from [ticket 20](issues/20-phase5-api-readme.md): its two
  lazy-import tests are the ONLY check that `import eos` then `eos.dd2` works,
  and the sweep already in `test_imports.py` cannot substitute for them —
  every module imports whether or not the `__getattr__` hook does. So the
  public surface [ticket 23](issues/23-phase6-respec.md) ports `nucleation`
  onto is guarded by a file a fresh clone does not have.
- **Scaling the strangeness residual.** Now has a second, sharper witness:
  [ticket 56](issues/56-baseline-empty-sector-gate.md) measured sfho's three
  `fixed_YC_YS` rows closing the same residual to 2.49e-09, 3.92e-16 and
  3.01e-16 — seven orders of spread across one density sweep of one model, which
  is what put the n = 0.16 row on the wrong side of the baseline's gate.
  Ticket 40 measured that a 1e-10 gate admits 0.079 MeV of `mu_S` — three orders of magnitude looser than the
  baseline's tolerance implied — and the `mu_e` sibling at Y_C = 0 shares it.
  That is solver conditioning, not a baseline question, and it is no longer
  urgent now the suite is green. What scaling to use, and whether it touches
  `general/` or each solver, is unsharp.

  **A grep-able signature for the failure, found by ticket 56 and free.** In the
  cleared sfho failure the shifts fell into **two values with the multiplet
  degeneracy of S_i**: 1.867e-06 for FOUR species (Lambda, Sigma+, Sigma0,
  Sigma-, all S = 1) and 3.734e-06 for TWO (Xi0, Xi-, S = 2). Three Sigmas
  landing on Lambda's number to the digit is what makes this a theorem rather
  than a ratio that happened to be 2 — the shift is `S_i`. §2 sets S = +1 per s
  quark, so S_Lambda = 1 and S_Xi = 2, and `mu_i = B_i mu_B + C_i mu_C + S_i
  mu_S + ...` carries ONE undetermined `Delta mu_S` into every strange species
  in exact proportion to its strangeness. So the ratio is the diagnostic:
  **shifts in the strangeness ratios 1 : 2 mean an undetermined potential moved;
  anything else means the physics moved**, and the two need opposite responses.
  It generalises past sfho to any model with a strange sector, and past `mu_S`
  to `mu_C` and `mu_B` by the same algebra — a `C_i`-proportional shift is the
  `mu_e`-at-Y_C = 0 sibling named above, which this retro-explains: that case was
  written down before anyone knew what it was.

  **And it is an argument for §2 nobody had made.** The signature exists only
  because species potentials are DERIVED by projection through B_i, C_i, S_i.
  A model carrying its own ad-hoc species potentials would show the identical
  failure as unstructured drift across a dozen quantities, with no ratio to
  read and no way to tell an undetermined potential from a moved number. §2's
  single-home rule for the basis maps is what makes the diagnosis possible at
  all, which is a stronger case for it than "declared once, imported
  everywhere".

  **Better as a check than as a diagnostic, and it has a home that does not exist
  yet.** Any two species differing only in strangeness must move in the ratio of
  their `S` when an undetermined potential shifts — so this could run directly
  and fail FIRST, rather than being applied after something goes red. The home is
  the `general/verify/` suite §5 requires and [ticket 11](issues/11-conformance-triage.md)
  row 31 ruled `general/` earns: **`eos/general/verify/` does not exist today**
  (checked), so this is candidate content for whatever
  [ticket 21](issues/21-phase5-structure.md) creates. What stays unsharp is the
  form — a single-point identity (`mu_i` equals its projection through B_i, C_i,
  S_i) and a two-run differential check are different tests, and only the second
  is what was actually observed.
- **Whether other tests silently degrade on missing data.** Ticket 39 fixed the
  two TOV helpers and generalised their guard to every `CRUST_FILES` name, but
  nothing has swept the rest of `test/` for the same pattern — an absent input
  turned into a wrong number rather than a skip. §10 promises the constraints
  module fails with a fetch message; whether it does, and whether anything else
  does not, is unmeasured.

## Out of scope

- **Creating or pushing any git remote.** `docs/NEXT_PHASES_PROMPT.md` Stage 7
  forbids it and `nucleation` already has one. Publication is the user's act, not
  this map's.
- **The `zlvmit` legacy pair.** CLAUDE.md §5 exempts `eos/zlvmit/` from the
  uniform API, and Stage 0 keeps both `ZLvMIT_hybrid.ipynb` and
  `zlvmit_test.ipynb`. Kept for published results, not brought into conformance.

- [eos/general/ has no verify/, which CLAUDE.md §5 states it has](issues/64-general-verify-suite-missing.md):
  **shipped** — `eos/general/verify/{__init__,run_full_check}.py`, five checks,
  all PASS, and every one of them proved able to FAIL on a deliberately broken
  `general/` (12-row table in the ticket). Section 5's three named checks became
  five because the meson gas earns its own and the integrals split Fermi/Bose.
  **One of the three was already done and is deliberately NOT repeated**:
  `test/general/test_fermi_gauss.py::test_agrees_with_jel` IS the split-panel
  validated against JEL, so the suite took the two alternatives nothing checked
  — Gauss-Laguerre and the entire Bose family — and says so in its docstring.
  **The T -> 0 check is a T^2 ladder, not a magnitude test**: a wrong CONSTANT in
  a closed form passes any loose magnitude tolerance and dies on the ratio, and a
  2e-5 offset was caught by the ratio alone. JEL is not the finite-T side there —
  its own 1e-4 error floors the ladder and hides that signature.
  **Finding, reported not fixed**: `solve_fermi_gl` falls back to the analytic
  forms only below `T < 1e-4` MeV but breaks down around `T/(mu-m) ~ 0.08` — at
  T = 0.5 MeV it silently returns a density **three orders of magnitude wrong**,
  while its docstring claims "higher accuracy than JEL". No solve path calls it,
  so no model number is affected; raising the threshold is a one-line change
  somebody should own. Second finding: the split-panel entropy is 1.1% off JEL at
  (m = 939, mu = 960, T = 1) MeV, not a k_max artifact.
  Gates: `test/test_imports.py test/general` **329/329 passed**; `test/baseline`
  6 failed / 10 passed, **byte-identical to `pytest_after_ticket61_baseline_py314.txt`,
  zero added**; python.org 3.14.2.

- [Four missing verify/ invariants, and who owes the delivery gate](issues/51-verify-invariants.md):
  **all five shipped**, nothing outside `eos/*/verify/` touched, no number moved.
  dd2 gains free energy + rearrangement + the delivery gate, `mixed` free energy,
  `ccdm` causality and monotonicity, and `njl`'s causality check now runs by
  default (`--sound` inverted to `--no-sound`; cost measured at **0.6 s**, so it
  is not `slow` and the ledger gains nothing).
  **The delivery gate is stated against n_B, not against the row order, and that
  is the whole content of it**: both `build_core_table`s end in
  `order = np.argsort(P)`, so `np.diff(P) >= 0` is monotone BY CONSTRUCTION and
  cannot fail. The sort does not repair the density column — proved by injecting
  a softening and re-sorting exactly as the builder does, which passes a
  `diff(P)` test and fails `diff(nB)`. `enjl`'s implementation does not have the
  problem because a constructed table is already ordered by density; the two
  builders that sort do.
  dd2's rearrangement check is ccdm's, copied not reinvented, with DD2's assembly
  substituted: both identities hold to 1.9e-16 and **Sigma^R carries 4.1% of eps**
  at n_B = 0.5, so neither passes by being small. All thirteen new checks proved
  able to FAIL (table in the ticket, including a **1 ppm** break caught in mixed
  and the ccdm nan guard from ticket 63).
  Gates: **twelve verify entry points, 134 checks, 0 FAIL** (121/eleven before
  this ticket and 64); `test/{dd2,did,ccdm,njl,mixed,general}
  test/test_imports.py` **1032 collected, 1029 passed, 3 failed** — the same
  three dd2 NMP-inversion node ids as `pytest_after_ticket61_dd2_py314.txt`,
  **zero added**, ticket 47's stack artifact; `test/baseline` 6 failed / 10
  passed, identical to the same before-image. python.org 3.14.2.
  Finding, reported not fixed: dd2's verify module docstring still lists a "TOV
  cross-check" as item 3 that the suite does not run — the list was already
  wrong and this ticket renumbered around it rather than editing a line it was
  not asked to touch.
