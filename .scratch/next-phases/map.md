# Map: eos next phases — notebooks, documents, conformance, Phases 5–6

Label: `wayfinder:map`
Effort: `next-phases`
Source: [docs/NEXT_PHASES_PROMPT.md](../../docs/NEXT_PHASES_PROMPT.md)

## Destination

`eos` on `main` and `nucleation` on `paper-release` both satisfy the Acceptance
criteria block of `docs/REFACTOR_PROMPTS.md`: three grouped usage notebooks exist
and execute end to end, every per-model document passes CLAUDE.md §11's test (a
physicist reproduces the model without opening the source), CLAUDE.md describes
the repository as it actually is, and Phase 5 and Phase 6 are done.

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

## Decisions so far

<!-- one line per closed ticket: gist + link -->

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

## Not yet specified

In scope, not yet sharp enough to ticket:

- **Notebooks for what the three do not cover.** `mixed`, `zlvmit`, `astro/tov`
  (including the RNS rotating backend) and `astro/gmode` have no usage notebook,
  and `docs/REFACTOR_PLAN.md:110` once planned per-model notebooks for several of
  them. Ticket 05 sharpens this; whatever survives becomes tickets.
- **Golden-reference re-verification after code fixes.** Any (a)-class fix from
  the conformance triage may move numbers. Whether that needs its own
  verification pass, and against which of the §12 references, depends on what the
  triage actually rules.
- **`output/public/` curation.** §11 makes it the one tracked output folder. The
  new notebooks will produce tables with standardised names; which of them belong
  in the tracked folder is not decidable until the tables exist.
- **`docs/DEFERRED.md` updates.** The three notebooks each report the gaps they
  hit. Whether those are new entries, corrections to existing ones, or closures
  is not knowable before the notebooks run.
- **What the 56 failing docstrings cost to fix.** Ticket 07 lists them but rewriting
  a docstring into the model document's notation requires that document to be
  settled first — so this waits on the document audit and the `.tex`/`.md` ruling.
  Whether it is one ticket per model or one sweep is not decidable yet.
- **The 6 mis-ordered files and the 9 dense comprehensions.** Reordering
  `eos/dd2/solver.py` and `eos/sfho/thermodynamics.py` is the serious half and
  touches solver internals, so it may need its own gate; the rest is cosmetic.
  Which of the two it is depends on the rename ruling landing first.
- **The 21 remaining (a)-class fixes.** Ticket 08 located each with file:line and
  several are one-liners, but they cannot be cut into tickets until
  [ticket 11](issues/11-conformance-triage.md) rules which are fixes and which are
  CLAUDE.md corrections. Whether they group by model or by section is a question
  for after that ruling.
- **Whether an underdetermined potential belongs in a frozen baseline at all.**
  [Ticket 37](issues/37-did-failures.md) has now measured `did`'s case exactly —
  one `Delta mu_S` of 2.32e-05 MeV propagating into each hyperon as `S_i x
  Delta mu_S`, nothing physical moved — so the general question is live:
  `docs/DEFERRED.md` records that `mu_S` at Y_S = 0 and `mu_e` at Y_C = 0 land on
  round-off, and `test/baseline/` pins them anyway at rtol = 1e-10 with a 1e-10
  absolute floor. Whether the floor should exclude such quantities by name, and
  which models' baselines carry them, is the design question. Blocked on ticket
  37's ruling.
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
