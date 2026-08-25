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
- `docs/STRUCTURE.md` does not exist; CLAUDE.md §10 and §11 both reference it.

## Suite status

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

The block below is kept as written, and is accurate FOR THE 3.14 STACK.

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
  `output_old/eos_tables_DD2vMIT_from_notebooks/`, so ticket 59 can compare
  against the published figures instead of asserting a replacement.

## Not yet specified

In scope, not yet sharp enough to ticket:

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

- **`output_old/` is 24 GB and is not gitignored.** `.gitignore:37` ignores
  `output/`; the rename to `output_old/` walked out from under it, so every
  session now sees 24 GB of untracked files in `git status`. Whether the folder
  is ignored, pruned or moved off the repo is the user's call and was not part
  of [ticket 05](issues/05-notebook-coverage.md).
- **`output/public/` curation.** §11 makes it the one tracked output folder. The
  new notebooks will produce tables with standardised names; which of them belong
  in the tracked folder is not decidable until the tables exist.
- **`docs/DEFERRED.md` updates.** The three notebooks each report the gaps they
  hit. Whether those are new entries, corrections to existing ones, or closures
  is not knowable before the notebooks run.
- **What the 56 failing docstrings cost to fix.** Ticket 07 lists them but rewriting
  a docstring into the model document's notation requires that document to be
  settled first — so this waits on the document audit and the `.tex`/`.md` ruling.
  Whether it is one ticket per model or one sweep is not decidable yet. Now also
  entangled with the renames: tickets 42–45 rewrite the very signatures those
  docstrings describe, so the sweep is cheapest immediately AFTER them, not before.
- **The 6 mis-ordered files and the 9 dense comprehensions.** Reordering
  `eos/dd2/solver.py` and `eos/sfho/thermodynamics.py` is the serious half and
  touches solver internals, so it may need its own gate; the rest is cosmetic.
  The rename ruling has now landed (ticket 10), and both serious files are ones
  tickets 44 and 45 rewrite anyway — so the open question narrows to whether the
  reorder rides along with those renames or is gated separately from them.
- **Several real fixes now live outside version control.** `.gitignore:75`
  excludes `/test/` entirely (§11), so ticket 39's helper skip, ticket 40's
  completed baseline exclusion and now [ticket 56](issues/56-baseline-empty-sector-gate.md)'s
  gate correction exist only in a working copy — anyone reconstructing `test/`
  reintroduces all three bugs. Whether some of that logic
  belongs in `eos/` where it would be tracked, or whether the layout rule should
  bend, is worth settling during [ticket 21](issues/21-phase5-structure.md).
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
