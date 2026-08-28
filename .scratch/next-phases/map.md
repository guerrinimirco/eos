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

**What "Phase 6 is done" means, amended by
[ticket 23](issues/23-phase6-respec.md):** Phase 6 splits into a **port**
([ticket 24](issues/24-phase6-execute.md)) and a **conformance pass**
([ticket 80](issues/80-phase6-conformance.md)), and **Acceptance gates on the
port only**. The Acceptance criteria block can check that `nucleation` imports
`eos`, that its suite runs, and that §1 holds in both directions — nothing in it
reads a README. Ticket 80 is in scope of this map and **not gating**: gating the
Stage 7 report on a rewrite no criterion measures would hold it hostage.
**[Ticket 76](issues/76-nucleation-golden-tolerances.md) IS gating**, and by the
same test: the port left two `nucleation` goldens comparing round-off, and the
criteria block's first line is "pytest ... fully green".

**THE STAGE 7 REPORT IS WRITTEN** ([ticket 25](issues/25-acceptance.md),
2026-08-27), with real tool output behind every claim. **All eleven Acceptance
criteria now pass.** The eleventh — "its mode coverage matches what CLAUDE.md
claims" — failed on exactly one name in exactly one model, `dd2`'s `fixed_YS`,
and [ticket 98](issues/98-fixed-ys-undeclared-mode.md) **ruled it on
2026-08-27**: not a mode, demoted to an internal `ModeSpec` label. Re-probing
10 models x 7 names through `eos_point` now returns `{}` where it returned
`{'dd2': ['fixed_YS']}`, and the result is held by
`test_imports.py::test_every_model_exposes_only_section_3_modes` rather than
by prose. **That was this map's last gating criterion.** The remaining open
tickets are non-gating by their own text — among them
[ticket 99](issues/99-quark-ea-at-zero-pressure.md), charted 2026-08-27 on the
user's request: E/A at P = 0 for two- and three-flavour quark matter, the
Bodmer-Witten window, whose two-flavour arm 98 has now unblocked (route 2, a
species flag, with its category ruled and inherited).

Reached when every ticket here is resolved and the Stage 7 report can be written
with real tool output behind every claim. **Ticket 80 is RESOLVED** (`2b2b72f`,
pushed to `origin/paper-release`), so the carve-out it was granted here — the one
ticket that might still be open when the report is written — is spent and no
longer applies to anything. (These three lines said "ticket 72" until the
`72 -> 80` renumber's stragglers were swept up by
[ticket 72](issues/72-enjl-branch-selection.md), the DIFFERENT ticket that now
carries the number; read literally they claimed this map's enjl finding was
already closed.)

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
- **Never loosen a tolerance to make a test pass.** Refined by
  [ticket 76](issues/76-nucleation-golden-tolerances.md): a tolerance is
  LOOSENED when the assertion measures the same quantity with more slack,
  and CORRECTED when it measures a different, better-chosen quantity. The
  rule forbids the first absolutely. It does not forbid replacing a relative
  comparison of a quantity that is zero by construction with an absolute
  bound on that zero — that is a different and strictly stronger claim.
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

**CURRENT, measured by [ticket 94](issues/94-zl-solver-flags.md) on
2026-08-28, as a BEFORE/AFTER PAIR in one session:**

    before  1812 passed, 23 skipped, 0 failed  (1835 collected, 21:04)
            output/_audit/pytest_before_ticket94_py314.txt
    after   1816 passed, 23 skipped, 0 failed  (1839 collected, 30:10)
            output/_audit/pytest_after_ticket94_py314.txt

**Zero added failures, and zero failures full stop on both sides** — which is
the one comparison a concurrent session cannot corrupt. The before-image
reproduces ticket 102's 1835/1812/23 exactly, which closes that block's
off-by-one worry from the other side: 1835 was right.

**But this pair is NOT a clean control, and the +4 is not this ticket's.**
Another session wrote `eos/enjl/*`, `eos/mixed/{boundaries,construction,
__init__}.py`, `docs/DEFERRED.md` and three issue files between 16:13 and
16:29 — inside the before-run (16:05-16:26) and across the edits. The +4
splits **3 theirs / 1 mine**, checkable from mtimes:
`test/enjl/test_enjl_construction.py` (16:23:56),
`test/mixed/test_enjl_pair.py` (16:24:07),
`test/mixed/test_locate_maxwell.py` (16:26:10) against
`test/zl/test_zl_modes.py` (16:29:39, exactly one added test). The two diffs
touch **disjoint files**, verified per-set with `git diff --stat`. The
runtime 21:04 -> 30:10 is contention between the two sessions.

So a session quoting 1839 should say which tree it means. What carried ticket
94's own claim was the attributable subset, disjoint from the other session's
files: `test/baseline` 20 passed, `test/zl` + `test/mixed/test_phase_pairs.py`
61 passed, `zl/verify` PASS 10/10.

**Previously, [ticket 102](issues/102-retire-phi-field-flag.md) on
2026-08-28:**

    python.org 3.14.2  1812 passed, 23 skipped, 0 failed  (1835 collected, 34:45)

Whole tree in one run, and **this is a POST-change collection with no
pre-change control** — `test/` is gitignored, so this session could not
snapshot the tree it started from and cannot close the denominator arithmetic.
What 102 itself contributes is known exactly: **+1**, the drift check
`test_phi_sector_is_off_exactly_when_its_coupling_is_zero`. Nothing was
subtracted — DID's `test_phi_field_cannot_be_switched_off` became
`test_phi_sector_has_no_off_switch` and dd2's `test_phi_field_presence` kept
its name, both 1-for-1.

**An off-by-one worth someone checking.** [Ticket 93](issues/93-dd2-nmp-inversion-noop.md)
recorded 1835 collected / 1812 passed / 23 skipped earlier the same day. If
that was measured on a tree already carrying tickets 104-107, then 102's +1
should read 1836 here, not 1835 — so either 93's number predates one of those
tickets or one of them is +0 where it was recorded as +1. **Nothing in this
ticket's gate rests on it** (102's own claim is "no number moves", carried by
the unmoved `dd2.npz`/`mixed.npz` and 0 failed), but the next session to quote
a denominator should measure a control rather than chain onto either figure.

**A run that lied, recorded because the failure mode is cheap to repeat.** The
first attempt reported **exit code 0 with its output truncated at 54% and no
summary line**, having printed two `F`s at 31%. The 29-33% band is
`test_dd2_m9` -> `test/did/test_couplings`, which holds the timing-sensitive
`test_dd2_speed.py`, and a concurrent session was running its own suite (it
regenerated `test/baseline/vmit.npz` at 14:48). Re-running the same tree with
the shell owning the log file: that band clean, 0 failed. **An exit code is not
a result — only the summary line is**, and `pytest -q` into a captured pipe can
lose the line that carries it.

**Previously, [ticket 98](issues/98-fixed-ys-undeclared-mode.md), 2026-08-27:**

    python.org 3.14  1738 passed, 20 skipped, 0 failed  (1758 collected, 18:13)

Two runs covering every test (`test/dd2` and `test/test_imports.py` in both,
green in both): `--ignore=test/baseline` gave 1718 passed / 20 skipped / 0
failed, and `test/baseline test/dd2 test/test_imports.py` gave 431 passed / 0
failed with all 13 baselines at rtol = 1e-10.
**1757 -> 1758 is ticket 98's one added test** and nothing else — the gate
`test_every_model_exposes_only_section_3_modes`. Quote the denominator with the
count or an added test is indistinguishable from a fixed failure.

**Previously, [ticket 25](issues/25-acceptance.md), same day:**

    python.org 3.14  1737 passed, 20 skipped, 0 failed  (1757 collected, 20:26)
                     output/_audit/pytest_ticket25_py314.txt
    nucleation         72 passed,  0 skipped, 0 failed  (72 collected, 4.31s)
                     output/_audit/nucleation_ticket25_py314.txt

Collection is **1757**, not 1696 — +61 since ticket 74. Against
`pytest_after_ticket74_py314.txt` that is **0 added failures**, and the
arithmetic closes with nothing left over: passed +57, skipped +5, failed -1,
and 57 + 5 - 1 = 61. The 20 skips are all availability guards (RNS binary, BPS
crust, CompOSE slices) plus one physics-conditional. **`eos` is NOT
pip-installed on 3.14** — both numbers above need
`PYTHONPATH=<eos>:<nucleation>`, which is the standing trap, not a repo defect.

**[Ticket 57](issues/57-canonical-stack.md)'s ruling is EXECUTED.**
python.org 3.14 is canonical; [ticket 62](issues/62-regenerate-baselines-py314.md)
regenerated twelve of the thirteen `.npz` on it and pinned the stack in both
repositories (`requires-python = ">=3.11"`, `numpy>=2.0`, `scipy>=1.17`, the
tested stack recorded in `pyproject.toml` and the README).

[Ticket 74](issues/74-py314-non-baseline-failures.md) then re-derived the six
non-baseline tolerances and premises on that stack.

    python.org 3.14  1 failed, 1680 passed, 15 skipped  (1696 collected)
                     output/_audit/pytest_after_ticket74_py314.txt

down from 12 -> 7 -> 1, and this after-number is HARD: the collected count is
1696 both before and after, the same denominator as ticket 62's after-image, so
it is a clean comparison rather than 62's soft one.

**That last survivor was `test_baseline[enjl]`, and it is now GREEN.**
[Ticket 72](issues/72-enjl-branch-selection.md) found that the model did not
pick different roots on the two stacks at all: both reached the SAME root, and
one of them missed the 1e-10 acceptance gate by 20% (1.20e-10 against
1.59e-12), behind which `solver.solve` falls through to a starting point on
the other chiral branch. The mode could reach the gate only because a held
Y_S = 0 left `mu_S` as an unknown no row determined. With that row pinned,
every ENJL mode sits four decades clear of the gate, the sweep is identical on
both interpreters, and `enjl.npz` is regenerated on 3.14 like the other twelve.

    test/baseline/  16 passed          all thirteen models, both stacks agree
    test/enjl/     127 passed
    test/njl/ test/ccdm/  134 passed   the sibling CFL undetermined potentials
    test/mixed/    277 passed          the engine that consumes the branch pair

So **"0 added failures" now means 0**, and the map has no deliberate red left.
The interpreter and collected count still travel with every number, because the
two stacks still both exist on this machine. The blocks below are history.

Ticket 74 took the six OFF the stack dependence rather than moving them onto
the other side of it: all six pass on anaconda 3.9.7 as well (`14 passed`).
Its sample was chosen for being seed-limited on BOTH stacks for exactly that
reason.

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

- **[Ticket 95 — `vmit.solver` takes `flags`, and `include_photons` goes](issues/95-vmit-solver-flags.md)**
  (resolved 2026-08-28). Second of the three serial solver-flag tickets.
  **108 of 1119 `vmit.npz` keys moved, every one one photon gas at T > 0** —
  36 points x (P, eps, s), zero composition keys, zero at T = 0, residue
  against `frozen - gamma` EXACT at all 108. **The before-image is the
  post-[ticket 100](issues/100-vmit-point-Y_S-never-assigned.md) state and
  that is measured, not assumed**: the same run with `photons=True` moves 0 of
  1119 against the on-disk file, whose 42 `.Y_S` keys are all non-zero, so
  100's regeneration is inside the frozen file and outside this delta. That
  control is stronger than 94's — it isolates the flag rather than only the
  generator's determinism. **The signature is njl's and ccdm's,
  `(par, n_B, [fraction], T, flags)`**, not zl's: vMIT is a quark model and
  §13's "read one, read the next" points at its own neighbours, which order
  `T` before the flags. **`two_flavour` went into the flags object too** — it
  was a parallel kwarg for a sector `SpeciesFlags` already carried, which is
  what [91](issues/91-leptons-default-and-drift-checks.md)'s second drift
  check forbids; value-neutral, both spellings defaulted False. `vmit_phase`
  got `photons=False` and NO `flags=` parameter
  ([109](issues/109-flagless-mixed-adapters.md) owns that). Thirteen other
  `.npz` BYTE-identical. **One call site reached only through an `as`-alias**
  (`vmit_trapped` in `test/mixed/test_hybrid_modes.py`) survived the grep and
  failed in the test run — a source enumeration here has to grep the aliases.
  `eos/zlvmit/mixed_phase_eos.py:2460-2464` is DEAD the same way 94 found zl's
  to be, and is left alone for the same reason.

- **[Ticket 94 — `zl.solver` takes `flags`, and `include_photons` goes](issues/94-zl-solver-flags.md)**
  (resolved 2026-08-28). First of the three serial solver-flag tickets
  ([95](issues/95-vmit-solver-flags.md), [96](issues/96-alphabag-solver-flags.md),
  then [91](issues/91-leptons-default-and-drift-checks.md)). **144 of 1356
  `zl.npz` keys moved, and every one is one photon gas at T > 0** — 48 points
  x (P, eps, s), zero composition keys, zero of the 216 T = 0 keys, and the
  residue against `frozen - gamma` is **0.000e+00, exact**, which is stronger
  than ticket 89's 0.89 ulp because the gas is not added rather than added
  differently. The control ran first: **0 of 1356 moved with the edits held
  back**, so the null hypothesis is measured at zero. Twelve other `.npz`
  BYTE-identical; `mixed.npz` and `zlvmit.npz` asserted, not assumed.
  **The signature is sfho's and did's — `(par, n_B, [fraction], flags, T)`** —
  not the literal "after `par`" the ticket wrote, which would have been a
  fourth argument order against §13. `leptons` keeps `True`; ticket 91 owns
  the flip. **Two call sites the work list did not have**:
  `eos/zlvmit/mixed_phase_eos.py:2386-2390` has been DEAD since ticket 90
  (pre-90 argument order inside a bare `except: pass`) and is left alone
  because repairing it can move `zlvmit.npz` — now
  [ticket 110](issues/110-zlvmit-dead-warm-start-calls.md); and **`zl_phase`,
  `vmit_phase` and `alphabag_phase` take no flags object at all**, so none can
  obey `eos/mixed/species.py`'s "a wing carries the caller's own `photons`".
  zl's now carries `photons=False`, which agrees with the mixture at its
  default. **Whether those three adapters grow a `flags=` parameter is ONE
  ruling, not three** —
  [ticket 109](issues/109-flagless-mixed-adapters.md), blocked by 95 and 96. Ticket 81's named coverage gap is
  closed for zl by both a flags-passing baseline case and
  `test_the_photon_flag_reaches_the_solver`.

- **[Ticket 100 — `eos.vmit.EoSPoint.Y_S` is never assigned, and the baseline froze the zero](issues/100-vmit-point-Y_S-never-assigned.md)**
  (resolved 2026-08-28). **Fixed and regenerated, in that order, and the sweep
  came first.** Every model's point, every mode, each cached fraction against
  `eos.general.basis` on that point's own densities: **on `Y_S`, `vmit` alone**
  — the ticket's claim held — and the fix is four `result.Y_S = q_thermo.Y_S`,
  routed through the map `table.py` was already using. **`vmit.npz`: exactly 39
  keys moved, all of them `.Y_S`, all from 0.0**, and the tree reproduced the
  stored file with 0 keys moving beforehand, so every delta is this fix's.
  `beta.T0.n0.45.Y_S` now reads 0.8402368306 beside the `Y_s = 0.8402` it used
  to contradict. **The sweep found a SECOND shape on `Y_L`** — `zl`, `vmit`,
  `alphabag`, all outside the trapped mode — which is
  [ticket 108](issues/108-cached-lepton-fraction-three-models.md) and not a
  wider diff here, because the three that leave it zero DOCUMENT it as the
  trapped mode's input while `njl`/`ccdm`/`did` measure it: a ruling, not a
  fix. New cross-model `test/test_cached_fractions.py`, verified red before
  green (`Obtained: 0.0, Expected: 0.8510384763983517`), 21 passed / 3 skipped
  on pre-existing non-convergence. **Ticket 99's retracted finding was still
  live in the source**: `general/zero_pressure.py` named vMIT's set as the one
  that comes apart — measured now, its surface is Y_S = 0.8379, not
  two-flavour — corrected there and in `vmit.md`/`.tex`, whose returned-fields
  table called `Y_S` one of "the fractions the mode fixed".

- **[Ticket 99 — E/A at P = 0 for two- and three-flavour quark matter](issues/99-quark-ea-at-zero-pressure.md)**
  (resolved 2026-08-27). **Both numbers ship, through one name in five models.**
  `zero_pressure_point(par, species)` in `vmit`, `alphabag`, `njl`, `ccdm` and
  `abpr`, over `eos/general/zero_pressure.py`'s `locate_zero_pressure`, which
  takes the state as a CALLABLE and so imports no model (§1). The flag 98
  ruled is **`two_flavour`**, named for the restriction and not the sector
  because §4 forces a two-valued flag to default False and `strange_quarks`
  would then have made `SpeciesFlags()` mean two-flavour matter — moving every
  quark number in the repository. **Ruled here for NJL/CCDM: two-flavour matter
  empties the s FERMI SEA and keeps the s CONDENSATE**, so `phi_s`/`zeta` still
  feed the light-quark masses through the 't Hooft determinant; dropping them
  would change the model, not the flavour content. `fixed_YC_YS` refuses the
  flag in all four (98's null column, stated as behaviour); `abpr` refuses it
  outright — no NaN. The **Fog is answered: both, split by job** — the number
  is public API, the identity `E/A = mu_B + Y_S mu_S` is the `verify/`
  invariant, the Bodmer–Witten verdict is a reported `below_iron` field
  asserted nowhere. **The golden reproduces**: 831.5839 MeV through the
  bracketed locator against `abpr.mu_from_P`'s closed form, agreeing to 5.5e-16;
  every identity is ≥ 3 decades inside the 1e-12 gate. The two `mu_B`
  conventions are now a TEST rather than a measurement — `alphabag/verify`
  locates the **CFL** surface, where `mu_S = 40.68 MeV` and `E/A = mu_B` alone
  would be wrong by 41 MeV, the only such case in the package.
  **Two of the ticket's own findings turned over under the build**: `njl` DOES
  have a surface (0.3824 fm^-3, E/A = 1102.02) — the scan for the lowest RISING
  crossing finds what a bare bracket missed — and `vmit`'s surface is NOT
  two-flavour (Y_S = 0.8379, not 0.0000), the zero having come from a field
  three of vMIT's four solvers never assign, now
  [ticket 100](issues/100-vmit-point-Y_S-never-assigned.md). `ccdm` has no
  locatable surface: pre-existing non-convergence, reported and not hidden.
  **1754 passed / 0 failed** (+36 tests), **all 20 baselines at rtol = 1e-10 —
  no number moves.** **BayEoS pinged** (`OPEN_QUESTIONS.md` Q7, a second
  UPSTREAM RULING block): both gates come off hold, with the call spelled out,
  the `Y_S`-is-measured-not-requested trap named, `abpr`'s refusal to catch,
  and `E_per_A_iron` to import rather than re-declare. Committed `4ceb442`.

- **[Ticket 98 — `fixed_YS` is a mode the code has and §3 does not declare](issues/98-fixed-ys-undeclared-mode.md)**
  (resolved). **It is not a mode.** Arm (b) — demoted to an internal `ModeSpec`
  label, unreachable by name — and arm (a) refused *because of* arm (c), the
  route [ticket 99](issues/99-quark-ea-at-zero-pressure.md) added. **The two
  `fixed_YS`es are different jobs wearing one word.** On `dd2` with hyperons on,
  Y_S is a free fraction with strange baryons to carry it and `mu_S` determined.
  On a quark model at Y_S = 0 — the only use ever requested, by BayEoS and then
  by the user — it is a sector REMOVED: `n_S = 0` holds over a range of `mu_S`,
  which is [ticket 75](issues/75-undetermined-potential-check.md)'s null column
  and [ticket 72](issues/72-enjl-branch-selection.md)'s priced receipt
  (residual within round-off of a 1e-10 gate, seed fallthrough, round-off
  choosing a chiral branch). §4 already refuses it in words — "if a sector is
  off, its flag is False" — so **two-flavour quark matter is
  `beta_eq_neutrinoless` with the strange flag False**, and route (c) is not a
  rival to (a) but the correct spelling of the job (a) would have been misused
  for. What (a) would have bought: a capability on 1 model of 11 with 0 baseline
  keys, 0 notebooks, 0 `verify/` entries, 0 document mentions and 0 tests naming
  it, for ten model audits and ten `DEFERRED.md` entries. What (b) cost: two
  dict keys. **(c) has no ten-model cost either** — the quark flavour content is
  physics only quark models have, so the flag joins §4's
  `phi_field`/`gluons`/`csc` class, five models, not a seventh mandatory name.
  **The flag's category is ruled here and binds 99**, on the user's constraint
  that `strange=False` is meaningless under CFL: it is `alphabag.gluons`'s shape
  exactly — two legal values in unpaired and 2SC, RAISES under CFL pairing,
  `abpr` refusing it outright — which §4's `gluons` paragraph already licenses,
  so no new §4 category and **no CLAUDE.md amendment at all**; a ruling needing
  one would have been arm (a). **Gate**:
  `test_imports.py::test_every_model_exposes_only_section_3_modes`, two halves —
  every model's registry is a SUBSET of `eos.MODES` ("same set" is
  unsatisfiable: `cfl` is only on `alphabag`/`abpr`, and `abpr` has nothing
  else), and a §3 name a model lacks raises with the mode NAMED, §3's own
  sentence and the half that stops a model passing the subset test by exposing
  nothing. Half 2 passes across all ten today with no behaviour change; half 1
  was mutation-tested red-then-green. It closes the direction
  `test_the_top_level_carries_the_mode_and_species_vocabulary` could not see:
  that one asks whether every `eos.MODES` name is buildable, never whether a
  model exposes a name outside it. **The two comment defects took different
  edits, as the ticket demanded**: `general/modes.py` was a plain miscount
  ("four" where §3 has five) and is rewritten to say why `cfl` is not a
  `ModeSpec` and that the two structural labels are not modes;
  `dd2/solver.py`'s false "Mirrors `eos.mixed.MODE_FRACTIONS`" is made true **by
  deleting the key**, and its "four" was never wrong. Three further sites the
  ticket had not found: `dd2/api.py`'s "DD2 also offers the extras in
  eos.dd2.MODES" (there are none), `solve_hadronic`'s mode list, and
  `docs/STRUCTURE.md:397`, where `fixed_YS` was printed inside the block
  demonstrating good refusal messages — regenerated from the code, not
  hand-edited. Also renamed `test_fixed_YS_counts_thermal_kaons`, which passes
  `charge_mode="fixed"` and has always solved `fixed_YC_YS`: this ticket's own
  subject, prose outliving behaviour, in the test directory. **No number
  moves** — names were removed, not equations; no `.npz` held a `fixed_YS` key.
  **BayEoS pinged**; its `two_flavour_stable: skipped` comes off hold when 99
  ships the flag, its `OPEN_QUESTIONS` must not be rewritten to request a mode,
  and its recommended `m_s -> 1e4 MeV` workaround is killed in the same note as
  the same defect one layer down. **Suite: 1758 collected, 1738 passed, 20
  skipped, 0 failed** on the canonical python.org 3.14.2 stack, across two runs
  covering everything; all 13 baselines reproduce at rtol = 1e-10. 1757 -> 1758 is this ticket's one added test and
  nothing else.

- **[Ticket 85 — the CLAUDE.md sentences the post-22 rulings owe](issues/85-claudemd-sentences-owed.md)**
  (resolved). **Six sentences landed, two deliberately did not.** Landed: §1's
  astro carve-out now names `mixed/hybrid.py` ALONE (with its
  `docs/DEFERRED.md:145` echo, the fourth site the ticket-61 gate missed, swept
  in the same edit); §2's Naming block gains `_nat`; §3 states that
  `leptons=False` on a beta mode RAISES while `leptons=True` is accepted and
  ignored; §4 states that a flag's category is judged over the modes the model
  HAS, so a mode may refuse a sector without becoming ticket 82's forbidden
  third category; §5's front-door clause is replaced by "in the first position
  of every public entry point ... `adapters.default_pair(par, flags,
  vmit_params)`, a call rather than a privileged position"; §5's composite-engine
  file list gains `species.py`. **Every one was re-read against the code before
  it was written**, and one grep would have lied: `grep -rn "astro\.tov" eos/`
  hits ten model `api.py` files, all of them DOCSTRINGS saying "the result feeds
  `eos.astro.tov`" — read as imports they would have manufactured a §1 violation
  in ten models. The real import is `eos/mixed/hybrid.py:237-238` and nothing
  else. **Not landed, and this is the ruling**: ticket 81 owed THREE sentences,
  not the one 90 shipped. `leptons=` defaults to False waits on
  [ticket 91](issues/91-leptons-default-and-drift-checks.md) (nine models still
  disagree) and §5's units sentence naming `s` waits on
  [ticket 97](issues/97-natural-record-leaves-the-result.md) (the `_fm`
  accessors are still on the public result in njl and enjl) — both open, and a
  sentence that becomes an invariant may not outrun its ruling, which is this
  ticket's own rule turned on itself. **85 does not stay open for them**: it
  succeeded ticket 22 in the vehicle role and a collector that never closes is
  the failure it was created to fix; 91 and 97 land their own sentences, as 82
  did. Ticket 82's two-category rule was verified PRESENT in §4 and not
  duplicated. Gate is greps, not a suite run — no file under `eos/` changed. All
  ten tickets citing ticket 22 are resolved, so nothing still names a retired
  vehicle. Noticed for Stage 7: `docs/STRUCTURE.md` cites CLAUDE.md §§ by stale
  numbers ("§5's six names" is §4, "the §4 name lowercased" is §3).

- **[Ticket 90 — one solver signature, one unit system at the boundary](issues/90-solver-signature-and-units-sweep.md)**
  (resolved). Ticket 81 §§2 and 5 were meant to be the no-value-moves half of
  that ruling; **two more halves of it turned out to move frozen values and
  were split out rather than the gate being weakened**, the treatment
  [ticket 89](issues/89-dd2-honours-species-flags.md) was given. §4 became one
  ticket per model — [94](issues/94-zl-solver-flags.md),
  [95](issues/95-vmit-solver-flags.md), [96](issues/96-alphabag-solver-flags.md)
  — because deleting `include_photons`/`include_gluons`/`include_thermal_neutrinos`
  into the flags moves every T > 0 row in three baselines, the generators naming
  none of the three. §5 and the record half of §3 became
  [97](issues/97-natural-record-leaves-the-result.md), split *after* being
  written and measured: the `_fm` names are ACCESSORS on the natural-units
  records the baselines freeze, so the rename changes what **4128 frozen keys
  mean** and the removal deletes **21271** (enjl 14976 nested under `.point`,
  njl 3255 and ccdm 3040 under `.state`). All four are blocked by this ticket;
  96 was also blocked by [92](issues/92-cfl-gluon-term.md), which has since
  ruled.
  **What executed moves nothing.** `par` is now first and required in all ten
  models — the seven that lacked it gained it, and the
  `if par is None: par = Parameters.default()` reach was DELETED from thirteen
  zl/vmit/alphabag entry points, which is the §6 violation the signature was
  hiding rather than a convenience. `params` -> `par`; `n_B_fm` -> `n_B` over 87
  sites with `_nat` on every displaced natural-units variable, now stated in
  `docs/STRUCTURE.md` §5 as the repository rule. **161 call sites moved, 139 of
  them by an AST rewrite** that relocates the argument's exact source slice, so
  multi-line calls move correctly. `dd2/solver.py`'s local is `n_B_solved`, not
  `n_B`: it is the EVALUATED density against the function's TARGET `n_B`, which
  differ at T > 0.
  **One silent-unit trap, and only one test caught it**: two seed expressions in
  `enjl`'s `default_guess` were natural units and became fm; every enjl test
  passed but `test_high_density_needs_a_widened_box`, which stopped at a scaled
  residual of 2.259e-09 against a 1e-10 bound. Confirmed against an isolated
  reverted control. Ticket 97 is told to rename fields first and accessors
  second for exactly this reason.
  Gate: `test/baseline` **20 passed, UNREGENERATED** except `enjl.npz`, whose
  **234 renamed keys are BIT-identical and whose 21044 surviving keys are
  bit-identical, 0 moved, 0 added or lost**; twelve `verify/` entry points,
  **136 checks, 0 FAIL**; full suite **1737 passed, 20 skipped, 0 failed**,
  py314. Four findings reported not fixed, including a baseline generator whose
  breakage is invisible to a partial run.

- **[Ticket 92 — the CFL phase refuses the free gluon gas](issues/92-cfl-gluon-term.md)**
  (resolved). `alphabag`'s `cfl` mode raises on `gluons=True`, and the ruling
  reached a second flag the ticket had not named: `thermal_neutrinos` was being
  **silently dropped** by `table.solve_at`'s `cfl` arm, §4's "never a silent
  no-op" live at the same five lines. **Three of the ticket's own premises were
  wrong.** `abpr` is not the independent second opinion it was taken for — it
  refuses photons, gluons AND thermal neutrinos with the same PRIMARY reason,
  "identically zero at T = 0, the only temperature this model has", and Meissner
  is a "besides"; the proof its T = 0 reasons do not bind alphaBag is that
  alphaBag's `cfl` carries PHOTONS at T > 0 and nobody calls that a
  contradiction. The physics that DOES decide it is the one the ticket never
  stated: locking leaves a single unbroken U(1)_Qtilde, so of the NINE gauge
  bosons exactly one stays massless — **the rotated photon, which is why
  `photons` stays and `gluons` goes** — and the free gas is the wrong COUNT
  (2 x 8 = 16) as well as the wrong dispersion. And candidate 1 is not "add a
  Meissner mass": the phase's light bosons are the superfluid phonon and the
  pseudo-Goldstone octet, a DIFFERENT sector, deferred as such. **"It is small"
  was never the argument**: 0 exactly at T = 0, but +0.119 MeV/fm^3 and 1.4–3.0%
  of the entropy at T = 30, and 5% of P and 10% of the entropy at T = 60,
  n_B = 0.45. **0 baseline keys, 0 figures, 0 verify values** — the six frozen
  `cfl.*` rows are T = 0, both notebook CFL builds are T = 0 and never named the
  flag, and every existing `solve_cfl` caller passes `False` or nothing. So
  `include_gluons` keeps its place and merely defaults `False` and raises on
  `True`: zero call-site churn, a message rather than a `TypeError`, and one
  fewer pass-through for [ticket 90](issues/90-solver-signature-and-units-sweep.md),
  which is **unblocked**. **Ticket 82's two-category rule is untouched, and that
  is the sentence to carry forward: 82's rule is about the FLAG, this refusal is
  about the PHASE** — `gluons` keeps two legal values over the modes the model
  has, so it stays a default, and the drift check needs no exemption because it
  iterates DEFAULTS. One real cost, not glossed: the legacy `TableSettings` shim
  has ONE sector switch set for both phases, so a legacy CFL table now raises
  unless both flags are named, and the published first-generation CFL tables
  contain the gluon term and can no longer be reproduced at T > 0 — recorded in
  `docs/DEFERRED.md`, and the shim's one caller (a test) names them. The §4
  sentence owed goes to [ticket 85](issues/85-claudemd-sentences-owed.md) item 5;
  `CLAUDE.md` untouched here. Gate is an isolated control/change PAIR from clean
  HEAD, both arms given the SAME `test/` so the control fails exactly the new
  nodes and nothing else: **control 21 failed / 1455 passed, change 19 failed /
  1457 passed** — **0 added failures and 2 fewer**, the two being this ticket's
  own refusal tests, red by design in the control. Every other failing node id
  is identical across the arms: `test_baseline[enjl]` plus 18 `test/enjl/`
  nodes, which are `test/` (gitignored, so working-tree) running ahead of the
  clean-HEAD `eos/enjl` a concurrent session holds uncommitted — measured, not
  inferred: `test/enjl` against the WORKING tree is **127 passed, 0 failed**.
  `test_baseline[alphabag]` green in the change arm; **`test_baseline[ccdm]`
  was the apparatus lying again** — red in the baseline-only run, green in both
  arms of the full one, ticket 82's concurrent-suites pattern.
  `test/mixed` excluded from the pair — it imports
  `sound_speed_frozen_pure`, which exists only in a concurrent session's
  uncommitted tree — and `eos/mixed` reaches only alphaBag's UNPAIRED
  `thermo_from_mu`, so the exclusion is outside the blast radius by
  construction.

- **[Ticket 89 — `dd2.solver.solve` honours `flags.photons`](issues/89-dd2-honours-species-flags.md)**
  (resolved, `992fd9c`). Ticket 81 §1 executed, and kept alone because it is the
  only commit in that ruling that moves frozen values. `include_photons` deleted
  from `solve` and its four wrappers and from `sweep`; `solve_beta_eq` keeps its
  own, having no flags object. **162 of dd2.npz's 4692 keys moved** — 54 points
  x P, eps, s — every other key BIT-identical, **0 of 2454 T = 0 and NMP keys**,
  composition unmoved, and every moved key moved by exactly one photon gas to
  under 1 ulp of the total. `mixed.npz` ASSERTED unmoved, 2497 keys, 0 moved.
  **Ticket 81 §1's own arithmetic is wrong and should not be requoted**: it
  predicted "456 of 976 rows"; the file holds 4692 keys and 162 moved, and the
  per-temperature split is not a constant multiple either. Its substantive
  clauses all hold. Work item 3 also under-counted the callers: `_dd2_frozen_block`
  is not a seed (its P/eps ARE the frozen sound speed), `verify/compose.py` adds
  its own photon gas against a CompOSE golden, and `backends/responses_jac.py`
  has two more — all three given `replace(flags, photons=False)`. One site now
  follows the caller's flag on purpose: `dd2_phase.wing_sweep`, whose rows reach
  `build_hybrid_table` with no mixture layer above them to add the radiation, and
  which until now had photons even at `SpeciesFlags(photons=False)`.
  **0 added failures** — test/dd2 + test/mixed + test/baseline, **492 collected,
  492 passed, 0 failed**, py314.

- **[Ticket 79 — three routes to a parameter set, in every model](issues/79-parametrization-surface.md)**
  (resolved, `ff7888e`). `named()` added to `zl`, `vmit`, `alphabag`, `abpr` as
  a ONE-ENTRY MAP in dd2/did's shape, keyed on the set's own `name` field — the
  docstring-only option lost to the same argument gap 2 turns on, that an absent
  attribute is an `AttributeError` a caller cannot interpret. `did`'s inverse
  map is REFUSED, not written: `invert_nmp`/`from_nmp` exist and raise, ticket
  26's zl pattern, and the session found a SECOND reason the ticket did not
  anticipate — DID's two inequivalent symmetry energies (S, S_2, 2.72 MeV apart
  at saturation) leave even the LIST to impose undetermined. dd2/did document
  `dataclasses.replace(default(), ...)` as the new-set route; vmit's replacement
  sentence is paid. A uniform routes block in all ten `.md` and all ten `.tex`,
  all of which compile. **0 added failures** — 892 collected, 2 failed, 890
  passed, IDENTICAL in an isolated HEAD control copy and a work copy, py314.
  `test_baseline[ccdm]` is newly red and is **ticket 65's**: 20
  `state.field_residual` keys, absolute 1e-6, compared at rtol 1e-10 against a
  stored zero — ticket 76's shape exactly.

- **[Ticket 65 — §4's six flag defaults unify on all-False](issues/65-species-flag-defaults.md)**
  (resolved). Nine models default every §4 name to `False`; `enjl` is the one
  exemption (it fixes every flag and raises on any move). **3108 of 53763
  baseline keys moved, all explained**, and a control run — defaults reverted,
  everything regenerated — moved **0 of 53763**, so the generator is
  deterministic on this stack and the flip is the sole cause. Movers: `did`
  (292) and `sfho` (163) photons, `njl` (1411) and `ccdm` (1242) muons, with
  every `nolep` case and every T=0 beta-eq case at exactly 0. Five suite
  failures, none papered over: two dd2 tests that inherited `muons=True` (one
  commented "# muons on" while relying on the default), an njl test that had
  pinned a `'CFL'`/`'free'` label decided at 1e-16, and two vMIT tests that
  caught the library disagreeing with itself. **Ticket 29 must adopt all-False**;
  `mixed` already inherits it via dd2's flags. Two findings raised, not decided:
  [ticket 81](issues/81-second-default-solver-kwargs.md) and
  [ticket 82](issues/82-alphabag-gluons-default.md). 1 failed / 1696 passed /
  1712 collected on 3.14.2, zero added failures.

- [The five items on the rename list that are not renames](issues/46-api-changes.md):
  **all five applied, 0 added failures, `test/baseline/` unmoved at
  rtol = 1e-10** (`95d4052`). Two of the ruling's four "settled by measurement"
  premises were FALSE and both were re-measured rather than argued.
  **sfho's isentropic solvers are not a second copy of the shared outer solve**:
  they carry the entropy axis as a ROW OF THEIR OWN RESIDUAL, T as a 7th
  unknown, and `general.tabulate.temperature_at_entropy` disagrees with them by
  up to **5.4e-8 in T** at the three densities `sfho.npz` freezes -- three to
  four orders above the gate, so the ruling's literal instruction would have
  moved a frozen number. Folded as the ticket's own question words instead
  (`SnB=` on the two mode solvers), which is bit-identical because `_system`
  already routed both axes. **`find_mixed_window` and `locate_window` are not
  one job either** -- one returns the list of mixed points, the other a
  `Window` of boundaries, and the first's docstring says so; it was a one-line
  alias for `sweep(..., mixed_only=True)` with zero callers, so it is deleted
  and `locate_window` is untouched. **A third shape of the ticket-42 trap, and
  the AST check cannot see this one either**: every `get_vmit_custom` caller
  already binds `Parameters` to **DD2's** dataclass, so the mechanical
  substitution would have handed a hadronic parameter set to a quark phase --
  the check flags names a ticket INTRODUCES, and `Parameters` is instead the
  name a deletion makes callers reach for. All five sites alias
  `VMITParameters`, as `eos/mixed/adapters.py:55` already did. The other three
  landed as ruled: `get_vmit_custom` deleted (23 occurrences, not 31 -- the
  ruling counted its own `.scratch`), `from_potential_depths` /
  `from_coupling_ratios` renamed in place, `build_hybrid_table` over 29
  occurrences in 14 files. **`docs/REFACTOR_PLAN.md:66` deliberately keeps the
  old name**: it records what the refactor deleted, and the name was the old
  one then. python.org **3.14.2**, collected **417** on
  `test/baseline test/mixed test/sfho test/vmit test/tov`, **1 failed / 401
  passed / 15 skipped before AND after**, same node id, `^E ` diff over 22
  lines EMPTY; sfho, vmit and mixed `verify/` green. **New git trap, measured:
  `git commit -- <pathspec>` commits the WORKING TREE at those paths and
  discards a filtered index** -- it swept a concurrent session's four
  `DEFERRED.md` hunks into the commit, since amended. Stage, then
  `git commit --amend` with no pathspec.
  [Ticket 79](issues/79-parametrization-surface.md) is unblocked.

<!-- one line per closed ticket: gist + link -->

- [Should the bare solver `include_*` kwargs follow §4's flags to False?](issues/81-second-default-solver-kwargs.md):
  **the premise was wrong in four ways and the ruling is not a default at all.**
  `dd2.solver.solve` ACCEPTS a `SpeciesFlags` and never reads `.photons` —
  measured, `SpeciesFlags(photons=False)` and `(photons=True)` give
  `P = 36.84136685` alike, exactly 1.000000 photon gases above the honest
  answer. `include_electrons` is not one of these kwargs: it is §3's
  `leptons=` under an `include_*` name, already ruled by ticket 70.
  The rule: **a solver that accepts a flags object must honour it and carry no
  parallel kwarg; one that lacks a flags object grows one** — then there is no
  second default to adjudicate. Because `flags` becomes REQUIRED, the call site
  decides the numbers and `zl`/`vmit`/`alphabag` move **zero** rows, which
  candidate 1 could not offer; `dd2` moves **456 of 976** because its generator
  flags already say `photons=False` while its frozen numbers contain photons.
  Renaming `n_B_fm` exposed a §5 violation the ticket never suspected:
  `njl`/`ccdm`/`enjl` return a natural-units record on the public result whose
  `n_B` divides out to exactly `hc3` but whose **P, eps and s do not** — it is
  matter-only, so `njl .state.P / hc3 = 146.854334` against an outer
  `146.939710`, and correcting by `hc3` still gives a wrong answer.
  Execution split by what must be re-measured, three ways and then six:
  [89](issues/89-dd2-honours-species-flags.md) (the only commit moving frozen
  values, as it was then), [90](issues/90-solver-signature-and-units-sweep.md),
  [91](issues/91-leptons-default-and-drift-checks.md) — and on 2026-08-27
  ticket 90 shed its §4, whose "no value moves" premise ticket 82 had
  falsified, into one ticket per affected model:
  [94](issues/94-zl-solver-flags.md), [95](issues/95-vmit-solver-flags.md),
  [96](issues/96-alphabag-solver-flags.md), all three blocked by 90 and 96
  also by [92](issues/92-cfl-gluon-term.md). Three sentences owed to
  [ticket 85](issues/85-claudemd-sentences-owed.md); ticket 82 becomes decisive
  rather than half an answer.

- [Should `alphabag.gluons` default False like the six?](issues/82-alphabag-gluons-default.md):
  **yes, and the rule is bigger than the flag.** §4 binds a model's OWN flags
  too, by a test that needs no list: **a flag with two legal values is a
  DEFAULT and is False, whatever its name; a flag with only one legal value
  RAISES on the other and is a STATEMENT. No third category.** That third
  category was where both defects lived, and it had exactly two members in ten
  models: `alphabag.gluons` and `dd2.phi_field`, both flipped to `False`.
  `sfho`/`did`'s `phi_field` and `abpr`'s `gluons` already raise and are
  untouched; `njl`/`ccdm`'s `csc` was already False. **Zero baseline keys, zero
  golden references, zero test failures** — established from the source before
  the first edit: `dd2` reads `phi_field` only as `phi_field and hyperons` so
  it is inert nucleonically, `case_dd2` already names it explicitly, and
  `case_alphabag` calls raw solvers the flag never reaches. Gate through the
  public surface instead: every alphaBag delta equals **minus `P_gluon(T)` to
  machine precision**, T = 0 unmoved. The widened drift check
  (`test_every_species_flag_defaults_off_or_raises`) **retired ticket 65's
  `enjl` exemption** — self-invalidating as designed, because under the
  two-category rule `enjl` is not an exemption, it IS the second category.
  Three findings: README example 1 carried a **wrong captured number since
  ticket 65** (recaptured, verified); ticket 90's "no value moves" premise is
  **false for alphaBag** (recorded there); the CFL phase's free gluon gas
  contradicts alphaBag's own document -> [ticket 92](issues/92-cfl-gluon-term.md).
  Gate is an isolated control/change PAIR from clean HEAD: the first full run
  had to be thrown away because a concurrent session was mid-edit on
  [ticket 67](issues/67-dd2-t0-adoption.md) in `dd2/thermodynamics.py`. Change
  arm run alone: **1734 passed, 0 failed**. The pair caught one real failure —
  `test_thermal_meson_feedback`'s helper named `hyperons=True` and INHERITED
  `phi_field`, while the Jacobian test hand-builds its unknown vector with
  `phi0` in it, so a default was deciding that vector's length (6 rows -> 5,
  `IndexError`). Flag named; 11 passed in BOTH arms, which is the proof it
  restores rather than patches. It also caught the apparatus lying twice:
  `test_baseline[ccdm]` went red in both arms and passes solo — concurrent
  suites, not a failure. `hadronic_eos` notebook sites deferred: a concurrent
  session held those files.

- [dd2 cannot adopt the shared T = 0 door without re-freezing three NMP entries](issues/67-dd2-t0-adoption.md):
  **it adopts, and the title's premise is false — zero `.npz` acts.** dd2 was
  the last model carrying its own T = 0 Fermi integrals; `kinetic_thermo` now
  calls `general.fermi_integrals.solve_fermi_t0(mu_eff, m, g, False)` and four
  closed forms delete, closing §7's finding 24. The re-freeze the title demands
  is forced only by a gate that **asserts below the stencil**. An `h`-sweep of
  `compute_nmp` over [5e-5, 3e-3], control vs adoption, put every shifted key
  under its own noise floor: `Q_sat` 5.1e-4 against 1.5e-3, `K_sat` 1.9e-8
  against 3.1e-6, `K_sym` 3.1e-9 against 7.7e-6 — and found a FOURTH key on the
  same floor, `L_sym`, passing `rtol = 1e-10` today by luck of magnitude. So the
  answer is a **per-key gate, not a re-freeze**: `MEASURED_RTOL` partitions the
  baseline by h-exact vs h-sensitive (1e-10 / 1e-5 / 3e-3), a partition the
  sweep draws with no judgement call. Dropping the keys instead would let a
  genuine 10% `K_sat` regression pass; 1e-5 still catches that with four orders
  to spare. **A cost the ticket never anticipated**: dd2's EoS quantities are
  stable to 5.9e-15 but a TOV sequence integrated over that table moves 1.24e-07
  — and perturbing the table by 1e-15 moves it 1.22e-07, at 1e-12 1.26e-07, a
  PLATEAU rather than growth, which identifies adaptive-step placement rather
  than propagated error. Same ruling, same reason. **`test_dd2_m8` was asserting
  a lottery**: at DD2's OWN NMPs a relative 1e-14 nudge of the target flips the
  5x5 between converging to 6.7e-11 and returning the published seed
  bit-for-bit, with `ok = True` either way because ISO_GATE admits the seed's
  own 2.2e-3 violation. Re-targeting is NOT available — 48 configurations over
  eight targets, not one holds its verdict across its own six — so the test now
  asserts the 5x5 default closure at K_sat = 220, stable across all twelve
  configurations measured. That lottery is larger than this ticket and spun out
  to [ticket 93](issues/93-invert-nmp-basin-lottery.md), cross-referenced to
  [ticket 47](issues/47-dd2-nmp-inversion.md), which found the same floor from
  the other side. **What this ticket got wrong twice, both by generalising a
  measurement past what it covered**: "the cost is not real" (four tests failed,
  three outside dd2) and "re-target the knife edge" (no such point exists).
  Landing note: `test/` is gitignored, so decisions 2 and 3 are landed-in-tree
  only and never became commits — the `MEASURED_RTOL` dict was instead shown
  green on HEAD *without* the adoption (`test/baseline` 20 passed, solo), which
  is the independence the ruling claimed for it.
  Gate, python.org 3.14.2 on `3781907`, each run solo: **1734 passed, 20
  skipped, 0 failed** WITH the adoption, byte-identical to the same suite
  without it, so the adoption changes no test outcome. dd2 `run_full_check`
  PASS, golden SNM(0.16) `1.40e-05` and CompOSE HS(DD2) `2.83e-05` both
  UNMOVED; `eos/general/verify` PASS.

- [Phase 6, second half — the conformance pass on nucleation](issues/80-phase6-conformance.md):
  **all four items landed and pushed (`2b2b72f` on `origin/paper-release`); the
  notebook EXECUTES in production mode, 39/39 code cells, zero error outputs.**
  Suite unchanged at the new tracked path `test/`: **2 failed, 70 passed, 72
  collected** on python.org 3.14.2 — identical to ticket 24's baseline node id
  for node id, both survivors [ticket 76](issues/76-nucleation-golden-tolerances.md)'s,
  no tolerance touched. **The brief's "4 import lines" is five statements and
  two ALIASES**, because the port creates two real collisions: `sfho.table` and
  `alphabag.table` both export `TableSettings`/`compute_table`, and
  `custom_params` is both the `nucleation.quark` constructor and an sfho
  `TableSettings` field. The make_fixture references were **six, not three**.
  pyflakes 24 -> 4, the bulk being fifteen private re-exports in
  `tables/__init__.py` that nothing imported through that door — the comment
  claiming `critical.py` needed `_BASE_DATA_KEYS` was false, and `critical.py`
  imports none of them. README examples run from a fresh clone off the committed
  fixture and were extracted back out and executed to verify. **Two things
  measured that the brief did not predict**: `make_fixture.py` no longer
  reproduces the committed fixture (all 937 rows move; restored bit-for-bit,
  hazard documented), and the smoke leg CANNOT complete — `F8_SHOW = [1,3]`
  clips to `[]` on the single-alpha smoke grid and `pd.concat([])` raises,
  pre-existing and untouched by the diff, but both READMEs tell a reader to
  smoke-run first. **Nothing under `output/paper/` committed**: 23 tracked files
  moved, 14 PDFs timestamp-only and 9 real, whose largest move is 13 cm on an
  11.14 km R_1.4 with `sigma_crit_star` bit-identical across all 398 rows; all
  restored to HEAD. **A concurrent eos session forced an isolated eos**
  (`git archive HEAD`): it holds `eos/sfho/*` dirty mid-rename of
  `create_custom_parametrization` -> `from_potential_depths`, which will break
  `test/make_fixture.py:98` and the paper notebook SILENTLY, since neither is on
  the suite's import path.

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
  `alphabag` alone), graduated to [ticket 65](issues/65-species-flag-defaults.md)
  (**now resolved: all-False**),
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

- [Regenerate test/baseline/ on the canonical stack, and pin it](issues/62-regenerate-baselines-py314.md):
  **Twelve of thirteen regenerated on python.org 3.14.2; `enjl` stopped the
  regeneration and became [ticket 72](issues/72-enjl-branch-selection.md).**
  630 of 53763 keys moved. Seven models are bit-identical across the stacks;
  `tov` and `zlvmit` drift at 1e-8; `ccdm` and `njl` move only in residual
  norms, numerically-zero quantities, and **`mu_3 = mu_C` in the CFL pattern**
  — where §3 says the locking leaves no free charge fraction — with `mu_8`
  moving by **exactly half** in both models, its coefficient in the projection.
  **The `C_i` fingerprint the Not-yet-specified section predicted fired
  verbatim, in two independently written models**, which is the strongest
  evidence yet that it belongs in `eos/general/verify/` as a check rather than
  a diagnostic. **Three, in fact**: [ticket 72](issues/72-enjl-branch-selection.md)
  later showed that `enjl` — held up here as the NEGATIVE control — was a
  positive hit whose undetermined `mu_S` at Y_S = 0 is what flipped its branch.
  The screen fired on it and this ticket read the output as noise. `dd2` moved in **three keys only** — `nmp.Q_sat` by 0.351 MeV,
  `K_sat`, `K_sym` — with all 4689 other keys bit-identical, which isolates it
  to the finite-difference stencil of [ticket 47](issues/47-dd2-nmp-inversion.md)
  rather than the physics; that blessing is the one in this regeneration NOT
  resting on the round-off screen, and is flagged as such.
  **The stop condition was discharged before anything moved**: the user's hand
  copy was found at `~/Desktop/Research/backups/baseline/` (2026-08-25 18:46),
  byte-identical to the 3.9 set; nothing was discarded, the 3.9 files are at
  `test/baseline_py39/`, and the verified 3.14 set was added **beside** the old
  one as `backups/baseline_py314/` with a README naming which is which.
  Stack pinned in both repos; `matplotlib` deliberately left unpinned because
  `figure_style.py:337` carries a working 3.4 fallback. All five README
  examples reproduce **bit-identically** on 3.14, so no printed digit changed.
- [ENJL's fixed_YC_YS continuation picks its chiral branch by warm start](issues/72-enjl-branch-selection.md):
  **the ticket's own diagnosis was wrong, and the real one is smaller. The last
  deliberate red is gone.** Both stacks reach the SAME root from the warm start
  (M_u = 260.2337); what differs is its residual — 1.59e-12 on 3.9 against
  1.20e-10 on 3.14, straddling the 1e-10 gate — and `solver.solve` answers a
  missed gate by falling through to `_restored_branch`, a seed on the other
  chiral branch, with no comparison against the near-miss it discarded. So
  ticket 62's trigger WAS round-off; the O(1) move was the consequence.
  **The mode could reach the gate only because `mu_S` was an unknown no row
  determined**: no species of this model carries S < 0 and at T = 0 there are no
  antiparticles, so Y_S = 0 forces every strange density to zero and the row
  reads 0 = 0 — a null Jacobian column that left this one mode three decades
  worse (1.7e-11) than its siblings (2.7e-15). Pinning it puts all four modes at
  1e-16…1e-13, recovers a density the old code dropped (34 points, not 33), and
  leaves exactly one branch change per mode — a real one, where the warm start
  fails by seven orders rather than by 20%. `BetaPoint.seed` now names which
  starting point produced a point. **A "take the best seed" policy would have
  made it worse**: the restored seed scores 8.08e-16 against the warm start's
  1.59e-12, so it would have flipped 3.9 too and nothing would ever have caught
  it. Regenerated on 3.14 and **identical on both stacks** outside
  numerically-zero quantities; against the superseded 3.9 file exactly FOUR
  observables moved, all at n_B = 0.5000, where that file had taken the same
  fall-through one point early. `test/baseline/` **16 passed**, `test/enjl/`
  **127 passed**, ENJL verify PASS with a new check 11
  (`check_residual_margin`, two decades of margin required, which would have
  gone red at 1.7e-11 years before any interpreter changed).
  **It also corrects ticket 62's classification, and that matters for
  [ticket 75](issues/75-undetermined-potential-check.md)**: `enjl` was run as
  the screen's NEGATIVE control while the same ticket recorded, as a footnote,
  "the sibling in `mu_S` at Y_S = 0". Those are one finding — the screen fired
  and its output was read as noise — so the fingerprint has three witnesses,
  not two, and the sharper lesson is that an undetermined potential is a
  **conditioning hazard**, not only a wandering baseline key. The physics
  question the ticket opened with is [ticket 83](issues/83-enjl-branch-selection-physics.md),
  non-gating.
- [How a raw ENJL continuation chooses its branch across a transition](issues/83-enjl-branch-selection-physics.md):
  **RESOLVED.** It does not choose one — it maps one, and after ticket 72 the up
  and down sweeps are two correct complementary branches over the whole grid.
  The choice belongs to the assembler, and the assembler's rule was wrong:
  **min-eps is not a construction.** Keeping the lower-eps row selects the
  stable PURE phase — correctly, and in every closure, since at T = 0 the free
  energy IS eps and the two roots carry identical charges, which settles the
  ticket's item 4 in the affirmative — but the minimum of two CONVEX eps(n_B)
  curves is CONCAVE at their crossing, so mu_B jumps down and P = mu_B n_B - eps
  falls with it. **min-eps cannot deliver a monotone table across a crossing,
  for any parameter set, in any mode.** The plateau is not an improvement on it;
  it removes a defect min-eps manufactures.
  **And the defect was in beta equilibrium, not only in the leptonless
  heavy-ion mode the ticket asked about**: on the notebook's own grid and call,
  `fq0.5_B1` and `fq0.7_B1` deliver min dP = -34.6 and -24.5 MeV/fm^3 with
  c_s^2 down to -0.227, and only `fq1.0_B1` is clean. Nothing caught it because
  the notebook plotted the clean set under markdown asserting the general claim,
  and `check_delivered_table` ran a failing set WITH its windows so the plateau
  covered the hole; `ConstructedTable.cs2` would have shown it and had no
  reader. Note `EOSTable_for_TOV` disclaims the check and `eos.astro.tov` does
  not perform it, so this model's `verify/` is the only §8 enforcement on the
  path — which is §8's own rule, the gate belonging to whoever builds the table.
  Fixed as a STATUS, not a raise (§6: a P-drop is a physics outcome a sampler
  must be able to score, and the empty-list path is correct for `fq1.0_B1`):
  `ConstructedTable.deliverable` / `.defect`, with `check_delivered_table`
  READING the predicate rather than recomputing it and grown to three cases
  demonstrated in both directions — windows/True, none/False naming n_B = 0.50,
  and `fq1.0_B1`/none/True as the **negative control**, without which "the gate
  fires" and "the gate fires whenever no window was passed" are the same
  observation. That is precisely the mistake ticket 62 made with `enjl`. ENJL
  verify PASS, 18 checks. Four documents corrected, the notebook's prose among
  them. The window the leptonless mode needs was MEASURED and not built —
  [ticket 88](issues/88-fixed-composition-coexistence.md).
- [A coexistence locator for a phase held at fixed (Y_C, Y_S)](issues/88-fixed-composition-coexistence.md):
  **RESOLVED** (`9bf61ec`, 2026-08-28), non-gating. Built, and it reproduces
  83's spline measurement to every digit that ticket reported:
  **[0.34945, 0.47500] fm^-3 at P = 22.9282 MeV/fm^3 and g = 1006.4074 MeV**,
  the eps crossing at 0.41774 inside it, `build_constructed_table` on the
  leptonless held-(Y_C, Y_S) mode now `deliverable = True` with a 13-row
  plateau flat to ptp(P) = 0. Both findings it opened with held.
  `composition_phase` + `locate_maxwell_composition` +
  `enjl_composition_coexistences` sit beside their beta twins; the carrier was
  GENERALIZED, not duplicated — `MaxwellPoint` and `Coexistence` now carry
  **P and g**, the pair coexistence equates in either closure and the only two
  single-valued fields, with `mu_B_lo`/`mu_B_hi` beside them, and `plateau_row`
  writes mu_B and mu_S only where the two edges agree (a bitwise identity under
  the beta closure, since there mu_B is one number handed to both phases).
  **The 2-D solve separates, which the ticket did not know**: Gibbs-Duhem at
  fixed composition gives n_B dg = dP, so g is monotone along a branch, and
  reading each branch's mu_B and P at a common g turns the SEED into the same
  one-dimensional sign change `locate_maxwell` bisects — 7 s on a 5-point
  grid, with the same window on 5, 6 and 60 points. Two traps recorded: the
  composition residual is exactly zero at mu_C = 0 by u-d symmetry, so `hybr`
  returns "not making good progress" on an exact answer and the gate must be
  the residual norm (section 6), not the success flag; and mu_S is dropped
  where a held Y_S = 0 at T = 0 leaves its row reading 0 = 0, which is ticket
  72's rank deficiency restated at the adapter boundary. eta = 1 only, as for
  the beta path — eta = 0 at held composition is a DIFFERENT delivered object
  (different Y_C, Y_S per phase, all three potentials equated, a window not
  flat in P), recorded on the two `docs/DEFERRED.md` entries the beta
  construction already owns rather than as a third. §11 text in `enjl.md` and
  `.tex` (compiles, 16 pages). ENJL verify PASS, **20 checks** (was 18), the
  section 8 gate demonstrated in both directions as 83's was; test/enjl +
  test/mixed 391 passed, 0 failed.
- [The six non-baseline failures on 3.14](issues/74-py314-non-baseline-failures.md):
  OPEN. The rest of ticket 57's cost list, none of it a `.npz` — Q_sat's
  `abs=0.2` re-derived from a noise floor measured on 3.14, `test_dd2_m8`'s
  (240, 300) premise re-measured, three tov robustness cases given a sample the
  6x6 closure can reach, and the DD2 published NMP/TOV values and CompOSE
  HS(DD2) slices re-checked — that last having no failing test attached, which
  is what makes it the easiest to skip.

- [Re-specify Phase 6 against nucleation as it actually is](issues/23-phase6-respec.md):
  **the corrected brief is written and agreed** — fifteen decisions across four
  grilling rounds, frontier empty. **Three of the ticket's own premises were
  false.** The alphabag rename question has no subject: `compute_alphabag_*` /
  `compute_cfl_*` are GONE from `eos`, which already carries §13's vocabulary, so
  nucleation simply follows and nothing widens. The line-count row is not drift —
  40 files / 11,620 lines in the tree, 8,121 in the package, both right for
  different denominators. And the job is not "a port across the refactor's module
  layout": measured target by target it is **one mechanical pass plus exactly one
  structural change**, because six of the seven broken targets have in-place
  successors of identical signature.
  Re-measured against `eos` at HEAD after ~57 tickets including four renames: the
  breakage is **exactly the seven, and nothing new** — so ticket 07's finding now
  holds EMPIRICALLY against a much-changed `eos`. `EOSTable_for_TOV` is the one
  that did not follow its neighbours into `astro/`; it went to `general/`,
  because it is the contract surface both layers may import.
  The structural change: the total-thermo assembly has no successor, and
  **nucleation keeps its own saddlepoint solver**, assembling from five pieces
  that are all already public (`thermo_from_mu`, `gluon_thermo`, and
  `electron_thermo` / `photon_thermo` / `neutrino_thermo`, the last three in a
  module nucleation already imports successfully). **No new `eos` code.**
  **Phase 6 splits**: [24](issues/24-phase6-execute.md) is the port, gated on a
  before-image taken FIRST (the suite cannot have been green since Phase 3, so
  "0 added failures" has nothing to measure against) with the rule that surviving
  failures are reported, not fixed; [80](issues/80-phase6-conformance.md) is the
  conformance pass, blocked by 24 and NOT gating Acceptance.
  **Two eos rules deliberately do not transfer.** `nucleation/test/` is tracked,
  not gitignored — `eos` hides its suite because it is private, nucleation is
  headed public and would ship a paper's repository with no runnable tests. And
  nucleation's `output/` gitignore stands: it already tracks
  `output/paper/{figures,figure_data,tables}` (87 files), which IS §11's
  `output/public/` principle correctly specialised; flattening it would untrack
  the paper's figures.
  **Two conformance items measured clean** and are commissioned no sweep:
  internal layering (already acyclic, `barrier.py` at the bottom) and the
  docstring standard (nothing in 8,121 lines). "Apply the same API conventions"
  is dropped outright — nucleation is a consumer, not a model.
  **The no-push rule is lifted** (its premise, "no remote exists", is false):
  push after 24, again after 72. **The dirty tree is resolved, and the brief's own
  instruction for half of it was wrong**: the 16 "regenerated" paper PDFs differ
  by 3-6 bytes each, all inside a `/CreationDate` stamp, zero content change — so
  they were DISCARDED, not committed, and a paper repository was spared sixteen
  binary blobs recording a timestamp. `docs/nucleation_physics.md` and
  `docs/reproducing.md` were restored; nucleation's tree is clean and its HEAD is
  now `cad424b`, ticket 62 having pinned the stack there too.
  Noted for Stage 7, not fixed: `alphabag`'s `solve_*` take `params=None` and a
  boolean flag-bag rather than `SpeciesFlags` — and that non-conformance is
  precisely why two of the seven targets are signature-compatible name swaps.

- [Execute Phase 6 — the port](issues/24-phase6-execute.md): **`nucleation`
  imports `eos` again**, landed as `32ef8c4` and **pushed to
  `origin/paper-release`** (`33c1e61..32ef8c4`). 17 files, +281/-63; **no file
  under `eos/` was edited**, so the brief's "no new `eos` code" held end to end.
  The port ran ITERATIVELY, as the before-image forced — one root cause masked
  every other break, so the evidence is the sequence, not a diff:

      run 0  before          module walk  38 of 38 FAIL   pytest  0 collected
      run 1  imports fixed   module walk   0 of 39 fail   pytest  21F 15E 36P
      run 2  Y_Le / mu_nue                                pytest   7F 12E 53P
      run 3  Y_u, Y_d, Y_s                                pytest   2F  0E 70P

  **Runs 2 and 3 existed only because run 1 cleared the mask.** `eos.sfho.table`
  had renamed the trapped table's outer axis `Y_L -> Y_Le` and its potential
  `mu_nu -> mu_nue`; only the READS of eos-produced tables were renamed, never
  nucleation's own keys. The structural change is `nucleation/quark.py`: a
  `DropletThermo` and ONE assembly function (the old unpaired and CFL builders
  differed only in their phase block) over `thermo_from_mu` /
  `cfl_thermo_from_mu`, `gluon_thermo` and the lepton and photon gases, plus
  `custom_params` for the seven `get_alphabag_custom` sites.
  **The brief's "exactly five added fields" was nine.** `e_total` is read at
  four sites, and `Y_u`/`Y_d`/`Y_s` through `tables/grid.py:_BASE_DATA_KEYS` —
  a list of key STRINGS fed to `getattr`, which no attribute-access grep can
  see. That is the lesson, not the fields.
  §1 holds both ways: forward `import nucleation` -> OK, pulling in only
  `eos.alphabag` and `eos.general`; reverse
  `test_eos_never_imports_nucleation` passes (194 in the file).
  **Two failures survive and were REPORTED, not fixed, with no tolerance
  touched** — an A/B swapping only the alphaBag kernel under the ported code
  (old: 2 passed, new: 2 failed) traces them to a **~1 ulp** change in the quark
  block's floating-point association, §2's shared basis maps replacing inline
  charge sums. They are now [ticket 76](issues/76-nucleation-golden-tolerances.md),
  which **blocks Acceptance** — by ticket 23's own argument, since the criteria
  block's first line is "pytest ... fully green".
  Reported, not fixed: **`eos` is not installed on the canonical stack** —
  `nucleation/pyproject.toml` declares it, `pip list` on 3.14 does not have it,
  and `import eos` works only via `PYTHONPATH`. The before-image was taken the
  same way, so the numbers are comparable, but "nucleation depends on eos" is
  true in the source and not yet in the environment.
  Transcripts: `output/_audit/nucleation_after_ticket24_py314.txt`.

- [The six non-baseline failures on 3.14](issues/74-py314-non-baseline-failures.md):
  **all six fixed, nothing loosened, and none was a regression** — the suite is
  `1 failed, 1680 passed, 15 skipped` (1696 collected), the survivor being
  ticket 72's deliberate `enjl` red. The six reduce to ONE function,
  `dd2/nmp.py::invert_nmp`, and **`eos/` was not edited at all**: the whole diff
  is four tests.
  **Ticket 57's framing was half wrong and that is the finding.** It called
  `test_api.py:127`/`:143` tolerances asserting below a noise floor; both are
  false PREMISES, and widening either would have pinned a number produced by a
  solve THAT NEVER RAN. `:127` is ticket 47's Q1 applied — DD2's published point
  is not a root of its own 5x5 closure, so the honest prediction is Q_sat =
  117.5, and 3.9 passed only because scipy 1.13's `hybr` stalled and returned
  the seed. **`:143` is not a stack artifact at all**: at DD2's own NMPs the
  6x6 returns the seed BIT-IDENTICALLY on both stacks (`status=5`, 48 restarts
  never beat 2.201e-03), because the `Q_sat` row moves 7.1e-04 under hybr's own
  1.49e-08 probe against a 1.5e-03 base residual — half its Jacobian column is
  stencil noise. 3.9 passed by coincidence.
  **The measured floor, which ticket 67 was waiting for: Q_sat carries 0.25 MeV
  of stencil excursion at the shipped h = 1e-4** (h swept in both maps together
  per `nmp.py:85`; plateau [2e-4, 1e-3] spread 0.088), so two evaluations differ
  by ~0.5 MeV. Underneath it sits an h-STABLE **-0.207 MeV** offset between the
  published table's 6-decimal coefficients and the re-derived ones — real, not
  noise. `test_dd2_m1.py:73` had pinned Q_sat at `abs=0.5` all along and
  survived the stack move, so `abs=0.2` was the outlier of the repo's own two
  tolerances on one quantity. **h was NOT moved**: that needs both maps together
  and a `dd2.npz` re-freeze this ticket does not authorise.
  `(K_sat, Q_sat) = (220, 300)` replaces both the m8 target and the tov sample —
  the only candidate **seed-limited on BOTH stacks** (x895 on 3.14, x219 on
  3.9), verified to keep every tov premise (core still undercuts BPS, 0.308 vs
  0.406 at n_B = 0.080). Item 4 discharged and unmoved: golden SNM 1.40e-05,
  CompOSE HS(DD2) 2.83e-05, published NS point and M_max >= 2 all PASS.
  Evidence: `output/_audit/nmp_noise_floor_ticket74_py314.txt`.
  **Reported, not fixed: `invert_nmp` returns `ok=True` on a solve that never
  ran** — ISO_GATE admits the 2.2e-03 stall, so a caller asking for Q_sat =
  169.0 is handed couplings whose Q_sat is 168.65 with no signal. Library
  contract, so Stage 7 report; the test now asserts the solve left the seed.

- **The seven grilling tickets, all ruled in one session** — [73](issues/73-dd2-remaining-cs2-names.md),
  [70](issues/70-leptons-on-a-beta-mode.md), [65](issues/65-species-flag-defaults.md),
  [53](issues/53-gmode-contract.md), [75](issues/75-undetermined-potential-check.md),
  [29](issues/29-mixed-species-flags.md), [46](issues/46-api-changes.md). Nine
  decisions across three rounds, frontier empty. Six are re-typed `task` and stay
  open for execution; 53 is resolved because its deliverable WAS the design.
  **Most of them were settled by measurement rather than preference.**
  **The literature settled two at once.** Zhao & Lattimer (arXiv:2204.03037)
  Eq. (1): `nu_g^2 = g^2 (1/c_e^2 - 1/c_s^2) e^(nu-lambda)` — **the g-mode IS the
  difference between the equilibrium and frozen sound speeds**, so with one alone
  it is identically zero. That is why gmode imports `dd2.solver`. It also showed
  `dd2`'s `cs2_ad` is **not a mistake but Zhao's own notation** (`c_s` = frozen
  composition), clashing with CompOSE/Typel where "adiabatic" means fixed
  entropy. §5's structure resolves it without picking a winner: composition rides
  on `frozen=`, thermal is the key name — so dd2 returns BOTH `cs2_isothermal`
  and `cs2_adiabatic` at each freeze, one multiplication away since it already
  computes `C_V` and `C_P`.
  **The g-mode contract shrank and its blocker moved.** Payload is the two sound
  speeds, not a composition derivative; T = 0 only, which collapses the thermal
  axis and NOT the composition axis (Zhao's operative clause is "without varying
  chemical composition", not the zero temperature). And the blocker was never
  `C_P`: **`frozen='composition'` is implemented in `dd2` ALONE** — six models
  expose only `equilibrium`, and `njl`/`ccdm`/`enjl` expose none. So the contract
  ends the §1 breach without making gmode general, which is still worth doing:
  it makes gmode's DD2-only-ness visible and per-model instead of hidden in an
  import. Execution [77](issues/77-gmode-contract-build.md), the nine-model gap
  [78](issues/78-composition-freeze-nine-models.md). **The count in this entry
  is superseded by ticket 78's live measurement**: `njl` and `ccdm` do carry
  `equilibrium`, so eight models expose only that and `enjl` alone exposes
  none — and `mixed`, uncounted here, is an eleventh unit carrying `chi`.
  **§4's flag defaults unify on all-False**, because the defaults are
  load-bearing and not cosmetic: 66 bare `SpeciesFlags()` calls, 13 entry points
  building one when `species=None`, 148 calls passing only `hyperons=`. It moves
  numbers and needs a baseline regeneration — done NOW while ticket 62's
  machinery is warm. `mixed` gains the flag set and delegates the per-phase
  sectors to its two phases; measured, there is **no double-counting risk**,
  because `adapters.py` already passes `include_photons=False` and the mixture
  adds photons once — the flag makes that correct by construction.
  **`leptons=False` raises on a beta mode; `leptons=True` is accepted and
  ignored** — njl/ccdm's reading made the rule for all six, because in a beta
  mode the leptons are constitutive, not unimplemented.
  **Four of ticket 46's five items answered themselves**: `get_vmit_custom` is a
  pure alias (`Parameters` carries identical defaults); the shared `SnB` outer
  solve ALREADY exists at `general/tabulate.py:78` and sfho just does not use it;
  `find_mixed_window` and `locate_window` have identical signatures; and
  `create_custom_parametrization` is already correctly in `nmp.py`.
  **One new requirement from the user**, now [ticket 79](issues/79-parametrization-surface.md):
  every model must offer a published set by name, an arbitrary new set, and — for
  hadronic models — one built from NMPs. Measured, it is not met: four models
  have no `named()`, `did` is hadronic with NO inverse map, and `dd2`/`did`
  cannot be built field-by-field.
- [The two `nucleation` goldens that compare round-off](issues/76-nucleation-golden-tolerances.md):
  both goldens asserted things no assertion should have read, and neither is
  fixed by a tolerance. `test_regression_solver_cases` compared, RELATIVELY,
  quantities the CFL flavour lock forces to zero — where the golden records
  only where the root find stopped, so it pinned floating-point ASSOCIATION and
  duly broke on the one-ulp `quark_charges` reassociation. It now asserts the
  lock itself, absolutely, at `robust_root`'s own `atol = 1e-8` — the tightest
  claim the code makes, so it cannot re-arm on a correctly converged point.
  **The zero set is charge-mode dependent**, which the ticket's framing missed:
  `Y_C` and `mu_C` are zero under `cfl` in every mode, but `Y_e`/`mu_e` only
  under LOCAL neutrality — under `gcn` the droplet is charged and
  `mu_e = mu_e^H` is a real ~317 MeV electron sea. `mu_e` under `lcn` is
  dropped rather than bounded: it is a NONLINEAR image of `Y_e` (`n_e ~ mu_e
  T^2`), so no bound on it follows from the gate. `test_energy_barrier` bounded
  `max|dW|` absolutely at `1e-9` on a curve reaching 2.7e6 MeV — 4e-16 of it,
  BELOW ONE ULP, so it asserted bit-identity; now relative to the barrier
  height at `1e-12`. **No `eos` diff**: `eos/alphabag/verify` already asserts
  the CFL lock absolutely, so `eos`'s own tripwire for this was correct all
  along and the `nucleation` golden was the outlier. **`regression.json` is
  untouched** — regenerating a golden hides the next real change the same way.
  Measured on the canonical stack: lock residuals `Y_C` 4.51e-12, `mu_C/mu_B`
  2.48e-13, `Y_e` 5.61e-13 against 1e-8 (2200x, 40000x, 17800x headroom);
  `W(R)` 4.51e-15 against 1e-12 (222x). The lock bound separates a locked phase
  from an unlocked one by 4.1e6. **`nucleation` is now fully green on BOTH
  stacks — `72 passed` on python.org 3.14 and on anaconda 3.9.7** — so the
  Acceptance criteria block's first line is satisfied. Ticket 25's noted
  wrinkle is dissolved rather than decided: [ticket 80](issues/80-phase6-conformance.md)
  landed as `569296a` mid-session, so the path the criterion names,
  `nucleation/test/`, now exists.

- [Build the T = 0 g-mode composition contract and drop the DD2 import](issues/77-gmode-contract-build.md):
  built, committed as `cddcf91`. `eos/general/sound_speeds.py` carries
  `EOSTable_for_gmode`, which SUBCLASSES `EOSTable_for_TOV` and adds
  `cs2_equilibrium` + `cs2_frozen` — a g-mode table IS a structure table, so one
  object serves both layers. `import eos.astro.gmode` now pulls in **no model
  package**, gated in `test_imports.py` twice: an AST sweep of `eos/astro/*` and
  a fresh-interpreter runtime check that catches the function-local import the
  AST cannot see. Both were confirmed to bite on a planted breach. **Every dd2
  g-mode number is bit-identical** (g1 = 149.565 Hz, f = 2064.516 Hz); the only
  movement anywhere is the Urca rate at 1.3e-5, which is `M_PI` now coming from
  `general/particles` as §7 requires.
  **The verify/ carve-out is ruled to EXTEND to an astro suite importing a
  model, and not to the reverse** — the directions are not symmetric, since a
  model importing `astro/` is the cycle the rule exists to prevent while an
  astro suite importing a model creates none, and the carve-out's own
  justification is a claim about suites rather than about which layer they sit
  in. **`cs2_eq`/`cs2_ad` are deliberately NOT renamed inside gmode**: the
  star's second slot holds the DYNAMICAL speed once `at_frequency` folds in a
  rate, so "frozen" would be false there, while the table's column is always the
  strict limit. Ticket 53's "mixed and gmode are one vocabulary" premise is
  severed by this ticket, so `mixed.eos_response`'s keys are now independent.
  **`cs2_frozen_point`/`cs2_frozen_along` were deleted** — no caller anywhere.
  **Unplanned finding, and ticket 73 does NOT close it:** `dd2`'s
  `frozen='composition'` speed is LEPTONLESS, so dd2 cannot fill the contract
  through `eos_response` either — differencing it against the with-leptons
  equilibrium speed compares two different fluids, a leading-order error in
  `N^2`. Ticket 73 split that speed on the THERMAL axis (isothermal vs adiabatic
  frozen); the missing axis is §5's THIRD one, `leptons=`, and it is missing in
  all ten models rather than nine. Not fixed here because `eos/dd2/` was under
  concurrent edit; the working producer sits in
  `gmode/verify/run_full_check.py` as `dd2_table`/`dd2_frozen_cs2`, one copy,
  ledgered, and [ticket 78](issues/78-composition-freeze-nine-models.md) is
  annotated with it. **The spelling proposed here is superseded**: ticket 78
  ruled that the third axis CANNOT be `leptons=`, which already means the §3
  mode flag on `eos_response` in seven models and the response axis in
  `mixed`. It becomes `reneutralize=`.

- **[`vmit_params` is threaded through the engine's internals](issues/84-vmit-params-in-the-plumbing.md)**,
  raised by the user asking why `hybrid_table` names one quark model. The engine
  DOES work for all models — `phases=(Phase, Phase)` is the general route and
  nine adapters ship — and CLAUDE.md:277 explicitly blesses
  `(par, flags, vmit_params)` as the DD2+vMIT front door. **The defect is that
  the slot did not stay at the front door**: 264 sites, with `solver.py` 16,
  `boundaries.py` 13, `table.py` 11, `responses.py` 9. §5's "couples phases only
  through this surface" is true of the documentation and not of the code. Now
  ticket 81, and it should be ruled consistently with
  [29](issues/29-mixed-species-flags.md), whose `species.py` removes the
  `muons=` kwarg sitting beside it in the same four signatures.

- **[`eos/mixed/scan.py` is removed](issues/87-remove-mixed-scan.md)**, on the
  user's ruling in ticket 84: 626 lines of declared DD2+vMIT study, plus its
  271-line test file, and with them the last module-level `dd2`/`vmit` imports
  outside the engine's adapter layer. **One function survived and relocated** —
  `build_parametrization` is now a free function in `eos/dd2/nmp.py`, where §5
  says an NMP-inverting constructor belongs, exported from `eos.dd2` and
  deliberately NOT re-exported from `eos.mixed`. Two duplicated constants and
  two silently-ignored keyword arguments died with the move. The three `vmit`
  documents that cited the scan carry a replacement statement rather than a
  gap: there is no scan driver, and a sweep is a caller-side loop over
  `Parameters`, one `eos.mixed.eos_table` call per sample. §8's delivery gate
  was never in the scan — `mixed/verify` implements it independently.
  `docs/DEFERRED.md:145` echoes CLAUDE.md §1's mention and is owed to
  [ticket 85](issues/85-claudemd-sentences-owed.md) with it.

- **[`leptons=` on a beta-equilibrium mode: one rule, in nine models](issues/70-leptons-on-a-beta-mode.md)**:
  **`leptons=False` raises; `leptons=True` is accepted and ignored as
  redundant**, written once as `eos.general.modes.resolve_leptons` and called by
  every unit that turns a mode name into a spec. In a beta mode the leptons are
  not an unimplemented sector but a CONSTITUTIVE one — the condition IS
  mu_C + mu_e = 0 — so §4's "an unimplemented flag raises" does not reach the
  `True` case. **The census was short by three**: nine models carry the row, not
  six — `vmit` and `alphabag` were silently accepting `False` beside `zl` and
  `did`, `enjl` was already correct beside `njl` and `ccdm` — and `eos/mixed`
  with them. `njl`'s and `ccdm`'s `eos_table` still accepted the refused call
  after `eos_point` was fixed, because `skip_errors` swallowed the raise inside
  the sweep; all 27 public entry points now refuse it. `leptons=None` is the
  caller not naming the flag, and is needed because ticket 68 ruled the
  per-model fixed-fraction default deliberate. **No number moved**: 204 cases
  compared bit-exact, 32 lines changed and every one a beta mode with an
  EXPLICIT flag, 0 with it unset. Gate 1424 collected, 0 added failures,
  `test/baseline/` unmoved. `eos/enjl/solver.py` keeps its own copy, that file
  being staged by a concurrent session for ticket 72.

- **[The undetermined-potential screen, all three limbs](issues/75-undetermined-potential-check.md)**:
  `basis.projection_residual` (one state), `basis.undetermined_potential` (two
  runs, the exact charge ratio) and `solve.undetermined_unknowns` (one run, the
  null Jacobian column) — the ruling's two plus the conditioning limb ticket
  72's amendment added, answered **directly** rather than through the residual
  symptom. **The differential's second run is the stored baseline**, chosen on
  availability: backends exist for two models, a second stack needs a second
  interpreter, a perturbed seed asks a different question. It ANNOTATES an
  already-red `test_baseline` rather than asserting, since a charge-proportional
  shift is legitimate. **Two readings, because the first version was blind to
  `ccdm` and `njl`** — the two models the screen's whole record comes from carry
  `mu_3`/`mu_8` flat, with no quantum number to divide by — caught by firing it
  at a real red ccdm and getting silence. Every limb proved to fail on a broken
  input and fired on real baselines, reproducing ticket 62's `mu_8=+0.500`.
  **Found and not fixed: `did`'s `fixed_YC_YS` at T = 0 carries `mu_S` as a null
  column**, at Y_S = 0 and 0.05 alike, live again at T = 20 — ticket 72's defect
  in the model its amendment names, now measured. Gate 1407 collected, 0 added
  failures, eleven `verify/` suites PASS.

- [Nine models cannot compute a frozen-composition response](issues/78-composition-freeze-nine-models.md):
  the gap and the ORDER are recorded in `docs/DEFERRED.md`; no freeze implemented
  and no per-model tickets spawned (see Out of scope). **The ticket's own
  measurement was wrong and is corrected**: `njl` and `ccdm` have carried
  `equilibrium` since the commits that introduced them, so `enjl` alone has an
  empty menu, and `mixed` — absent from every earlier count — is the eleventh
  unit with a third spelling, `chi`. `composition` exists in **one of eleven**.
  **The substance is why it is nine separate jobs**: a freeze needs the model at
  prescribed species densities (§13's `thermo_from_n`), unreachable by re-tuning
  `(mu_B, mu_C, mu_S)` because three potentials cannot hold eight fractions. That
  block exists in `zl`, `vmit`, `enjl`; `dd2` has only the nucleonic special case,
  so **dd2 cannot freeze with hyperons on either**. Order: dd2 → zl, vmit → abpr
  (a RULING: CFL locks the composition, so its frozen speed IS the equilibrium one
  and an ABPR g-mode is zero for a physical reason) → sfho, did (must write the
  block; did's beta-rearrangement channel drops out by construction) → alphabag →
  njl, ccdm (pairing means the composition is not free, and the pattern-under-
  freeze question comes first) → enjl (`equilibrium` first) → **mixed last BY
  CONSTRUCTION**, since the species live in its phases behind the adapter contract.
  **Also ruled, on ticket 77's hand-off**: §5's third axis cannot be spelled
  `leptons=` — that keyword already means the §3 mode flag in seven models and the
  response axis in `mixed`, opposite senses. `leptons=` keeps the mode meaning;
  the response axis becomes `reneutralize=` (default True), so `mixed` is the one
  surface that renames. Consequence: with ticket 70's rule live, the leptonless
  probe MUST go through `reneutralize=False`.
  Gate: docs only, no `eos/` file touched, no test can move; the concurrent
  session's `DEFERRED.md` edit checked intact before and after.

- **[Ticket 29 — eos/mixed has no species flags, so its photon gas cannot be
  turned off](issues/29-mixed-species-flags.md)** (resolved, `8bb546c`).
  `eos/mixed/species.py` exists, carrying §4's six names, all `False` per
  ticket 65. The engine stopped BORROWING DD2's flag class: `api.py` no longer
  imports `eos.dd2.species`, and no docstring says `dd2 SpeciesFlags`. The
  per-phase/phase-common split the ruling described turned out to be already
  built — every adapter passed `include_photons=False`, the mixture added
  `photon_thermo(T)` once — so the flag made a hardcoded `False` correct by
  construction: switching `photons` on at T = 20 MeV moves P, eps and s by
  exactly `photon_thermo(T)` (3.6e-15 on P), and at T = 0 it is bit-for-bit
  inert. `thermal_neutrinos` is carried and RAISES, as in `dd2`.
  The `muons=None` kwarg went from all four entry points, and the internal
  chain that carried it now threads the FLAG OBJECT as `species=` — so
  `photons` needed no plumbing of its own and the next §4 name will not
  either; leaf helpers that take a bool keep it. `mixture_flags` reads the six
  off any §4-conforming object, which is what lets the DD2+vMIT front door
  hand one `dd2.SpeciesFlags` to both the phase and the mixture, and why no
  existing call site grew an argument.
  **Left where [ticket 86](issues/86-mixed-phase-pair-primary.md) can
  finish it**: the front door's default flags moved from `api.py` into
  `adapters.default_flags()`, which 86 deletes with the rest of the front
  door. `__init__.py` and `docs/DEFERRED.md` were deliberately NOT touched —
  both carried another session's uncommitted work; the `thermal_neutrinos`
  DEFERRED row is owed and is this ticket's one loose end (it IS stated in
  `mixed.md`/`.tex`).
  Gates, python.org 3.14.2 (numpy 2.3.5, scipy 1.17.0), never under `timeout`,
  as an **isolated-copy pair** — the live tree carried a concurrent session
  throughout, and HEAD moved mid-ticket (tickets 87 and 75 landed), so the
  pair was rebuilt on the new HEAD. `test/mixed` + `test/baseline`: **282
  collected, 281 passed / 1 failed** against control **274 passed / 1
  failed**, the +7 this ticket's new tests. The one failure is
  `test_baseline[ccdm]`, identical in the control — the same isolation
  artifact ticket 69's entry above already names. `test_baseline[mixed]`
  passes at rtol = 1e-10 (the mixed baseline is T = 0; the photon term is
  finite-T), `eos/mixed/verify/run_full_check.py` **PASS**, all ten `[ok ]`.
  §11: `mixed.md` and `mixed.tex` said the opposite in three places and now
  state the photon treatment, the split, the corrected signatures and the
  `thermal_neutrinos` refusal.

- [Make the Phase pair the parameter argument in `eos/mixed`](issues/86-mixed-phase-pair-primary.md):
  **ticket 84's ruling, executed.** `phases=(Phase, Phase)` is positional #1 on
  `eos_point`, `eos_table`, `hybrid_table` and `eos_response`, and
  `(par, flags, vmit_params)` is ABSENT, not deprecated — so the composite
  engine finally reads as §5's `eos_point(par, mode, species, **conditions)`
  with the pair in `par`'s place. `vmit_params` in `eos/` outside the
  §1-exempt `zlvmit`: **84 -> 0** in the plumbing; the eight survivors are the
  ruling's own `default_pair(par, flags, vmit_params)` call form, in that
  function's signature and in six sentences quoting it.
  `adapters.default_flags()` deleted (it absorbed into `default_pair(par,
  flags=None, ...)`); `hybrid_table`'s narrowed trapped-mode guard deleted (the
  DD2 adapter's `_dd2_wing_kwargs` already raises, and still before any solve);
  `__init__.py:51`'s `from eos.dd2 import ...` — which was in the DOCSTRING,
  not an import, which is why 29 could leave it — rewritten, and the package
  now exports its own `SpeciesFlags`/`mixture_flags`, which ticket 29 created
  but never put on the surface.
  `hadronic_qn`/`hadronic_charges` moved to `general/basis.py` with
  `active_baryons`; `charges_from_densities` does NOT do the same job (it sums
  every non-lepton species, the other sums only the flags' active baryons, and
  `dd2/table.py` needs the narrow one because a meson gas rides in the same
  dict), so nothing merged and both now name each other.
  `responses.py`'s `from eos.dd2.solver import warm_start` and its
  `flags.phi_field and flags.hyperons` read both lived in
  `sound_speed_frozen_hadronic`, which with `sound_speed_frozen_quark` was two
  model-specific spellings of one idea; both retire into one
  `sound_speed_frozen_pure(phase, th, ...)` over `Phase.frozen_thermo` — the
  surface 84 said had been there all along.
  One behaviour the sweep nearly lost, caught by its own test and fixed rather
  than reverted: `hybrid_table`'s trapped-mode guard sat OUTSIDE the
  `try/except` that turns non-convergence into a status, and the adapter raise
  that replaces it happens INSIDE it — so a malformed trapped call stopped
  raising and started returning `ok=False`, which §6 forbids. The three-line
  wing pre-flight is now `hybrid.validate_wings(phases, spec, T)`, called from
  `build_hybrid_table` as before and from `hybrid_table` before its try.
  Gates on python.org 3.14.2 (numpy 2.3.5, scipy 1.17.0), in an isolated copy:
  `test/mixed` + `test/baseline` **279 passed / 2 failed**, `test/mixed`
  entirely green across all 27 files and the two failures traced to
  `eos/njl/solver.py:solve` moving to `par` first (their ticket-90 sweep)
  against `generate_baseline.py`'s untouched old-order call — `git diff` on
  that file mentions njl/ccdm zero times. `test/baseline`
  `mixed.npz` and `tov.npz` **unmoved** at their frozen tolerances (a signature
  change moves no number, and the baseline generator had to gain an explicit
  `species=MixedFlags(muons=True)` or it WOULD have — the front door read that
  flag off the hadronic `flags`); `eos/mixed/verify/run_full_check.py` **PASS**,
  all ten `[ok ]`, TOV M_max = 2.340 at 12.64 km; DD2+alphaBag, ZL+vMIT,
  DID+NJL, DID+CCDM and the ENJL branch pair all run end to end through the new
  primary signature.
  The gate ran in an isolated copy because the live tree carried a concurrent
  session that **reverted `eos/mixed/adapters.py` outright** mid-ticket
  (`default_flags()` back, `default_pair` stripped of its new default) — and
  because nothing imported the dead function, the suite still passed and only a
  grep saw it. That session also moved `eos.vmit.solver`'s four mode solvers to
  `par` as positional #1, which briefly broke `eos/mixed/adapters.py`'s own
  `_vmit_wing_solve` and left five DIRECT `from eos.vmit.solver import ...`
  call sites in `test/mixed` needing the new signature; those are swept here,
  since otherwise this ticket's gate line cannot be measured. The njl/ccdm half
  of the same shift lands in `test/baseline/generate_baseline.py`'s own cases
  and is left to them — this ticket's baseline line names `mixed.npz`.
  Two model imports stay inside `eos/mixed` and are named rather than swept:
  `backends/jacobian.py` (the accelerated flavour of the two shipped adapters,
  reached only through `Phase.jacobian_block`; §5 makes `backends/` deletable
  and §9 keeps it off the reference path, so moving those kernels into
  `adapters.py` would break both rules to satisfy a summary of this one) and
  `verify/run_full_check.py` (§1's `verify/` carve-out).
  Swept: all 27 `test/mixed` files, `test/baseline/generate_baseline.py`,
  `test/test_nonconvergence_return.py`, `test/tov/test_solver_fast_robustness.py`,
  `test/dd2/test_table_rows.py`, `notebooks/hybrid_eos.{py,ipynb}` (whose
  section 3 stops being "two calling forms" and becomes one signature with
  DD2+vMIT beside SFHo+NJL — and which was ALREADY BROKEN before this ticket,
  still passing the `muons=` kwarg ticket 29 removed), `docs/STRUCTURE.md` §11,
  `mixed.{md,tex}`, `eos/vmit/parameters.py`, `eos/vmit/vmit.md`, `README.md`,
  `eos/__init__.py`, `docs/DEFERRED.md`.
  **`../nucleation` on `paper-release` needed nothing, measured not assumed**:
  a grep over `*.py`/`*.ipynb`/`*.md` and packaging — notebook cells and lazy
  imports included, the fog's named blind spot — finds `eos` used there only
  through `eos.sfho.table`. `eos.mixed` is not imported in that repo at all.
  CLAUDE.md untouched: the §5 sentences are written out in
  [ticket 85](issues/85-claudemd-sentences-owed.md) item 3, marked SHIPPED,
  with the §7 note (no sentence owed for the `hadronic_qn` move — §2 already
  required it) and the "ticket 81" renumber straggler corrected to 84.

- [Run the Acceptance criteria block and write the Stage 7 report](issues/25-acceptance.md):
  **ten of eleven criteria pass with real tool output behind each; the eleventh
  fails on one name.** eos **1737 passed, 20 skipped, 0 failed (1757 collected,
  20:26)**; nucleation **72 passed, 0 failed (72 collected)**; python.org 3.14.2.
  **0 added failures against `pytest_after_ticket74_py314.txt`**, and the
  arithmetic closes exactly: collection +61 = passed +57, skipped +5, failed -1.
  All 13 baselines reproduce at rtol = 1e-10 (`20 passed`); 13/13 `.tex` compile
  under `-halt-on-error`; the rcParams grep hits exactly one file; 454 tracked
  files, none over 5 MB; no dependency added (the four optional imports all
  predate the map by weeks); §1 holds both ways.
  **The failure is criterion 4, and it is [ticket 98](issues/98-fixed-ys-undeclared-mode.md):**
  probing all 60 model x mode pairs through `eos_point` returns
  `{'dd2': ['fixed_YS']}` as the only name reachable that no specification
  declares — and `docs/STRUCTURE.md:397` prints it, inside the block
  demonstrating good refusal messages, while §4's mode table lists five.
  **The 500-set script needed ±95%, not ±25%**, before the sample left the
  convergent region: 4 samples / 2000 evaluations, 11 `ok=False`, **0 raised,
  0 hung**, slowest task 0.19 s. Its first draft was rejected by
  `Parameters.__post_init__` for drawing `a` and `d` free when `couplings.py`
  derives both — the validator was right and the script was wrong.
  **25 document examples run in order**: 18 byte-identical, 4 differ, and every
  difference is stale prose or a wall-clock field — no physics number in any
  document is wrong. STRUCTURE.md §12's worked figure regenerates
  byte-identically; `docs/figures/dd2_MR.png` does **not** (D4), which puts the
  map's tracked-figures fog item in `eos` as well as `nucleation`.
  Nine defects reported not fixed (D1–D9), incl. a library loader that prints
  with no off switch, `np.trapz` deprecated under an unbounded `numpy>=2.0` pin,
  and `nucleation` `37af659` unpushed. Logs:
  `output/_audit/{pytest,nucleation,baseline}_ticket25_py314.txt`,
  `output/_audit/{inference_stress_500,mode_species_coverage,doc_examples_*}*`.


- [`zl.invert_nmp`: the closed form exists and was verified](issues/104-zl-analytic-inversion.md):
  **built, and it makes zl the one model here whose inversion is algebra.** No
  seed, no basin, no restart count -- which is the property
  [ticket 93](issues/93-invert-nmp-basin-lottery.md) and
  [ticket 105](issues/105-dd2-isoscalar-conditioning.md) are about the ABSENCE
  of. The ticket's closed form was re-derived rather than transcribed and
  reproduces digit for digit (`a0=-96.6555 b0=58.8619 gamma=1.39854
  a1=-25.1985 b1=7.1850`), round-tripping to n_sat 1.2e-14, E_sat 5.6e-13,
  E_sym 5.4e-07 and K_sat/L_sym to 1e-2 -- the forward map's own stencil.
  **The premise the ticket left unstated is `n0 := n_sat`**: the form is exact
  only because the functional's reference density is SET to the requested
  saturation density, so saturation is imposed at u = 1 rather than found,
  which is also why inverting the published NMPs returns gamma to 3e-5 and
  a0/b0 to 0.3 % rather than exactly. The two convention traps were the whole
  risk and both are now measured, not asserted: on the shipped set the
  quadratic E_sym/L_sym read **30.848 / 41.270** against the full-step
  **31.561 / 42.718**, so the familiar target {31.6, 43} IS the shipped set in
  Constantinou's convention and was never an independent target -- exactly the
  0.87 MeV that moves a1 from -26.06 to -25.19. The rest-mass trap's FIRST
  test was wrong (couplings are not mass-invariant; the free gas depends on m)
  and is pinned instead by the one thing that is exact: feed E_sat = -16 and
  the functional must bind at -16, not m - 16. Both ZL facts confirmed —
  `Q_sat,V = 3(gamma-2) K_sat,V` exactly, so Q_sat is refused as the sixth
  datum, and it is tested where it IS exact (nothing isovector reaches gamma
  or b0, so the predicted Q_sat is **bit-identical** under moved E_sym, L_sym,
  gamma1) rather than against a third-derivative stencil that only converges
  as h^2; and K_sym imposition, which needs no root find either since gamma1
  falls out linearly. Beyond the ticket: an `InversionStatus` (§6 -- a target
  saturating outside `N_SAT_BRACKET` is a return value), and the free choice
  named exactly once (gamma1 XOR a dict `K_sym`, never both, no default). **A
  downstream site the ticket did not list**: `notebooks/hadronic_eos.py` called
  `invert_nmp(**target)` relying on the `NotImplementedError`, which under the
  new signature is a `TypeError` `run()` does not catch -- fixed, with the
  markdown bullet that told the reader zl "cannot be built *from* a set of
  them". Gate is the blast radius, not the tree: `test/zl` + all 12 baselines +
  imports/routes/nonconvergence, **329 passed, 0 failed**, `zl.npz` unmoved, a
  concurrent session holding dd2/general/mixed/CLAUDE.md making a whole-tree
  run unattributable. `HADRONIC["zl"]` flips False -> True in
  `test_parameter_routes`; `docs/STRUCTURE.md:292` needed no edit because it
  already listed zl under "nuclear-matter parameters, inverse" at HEAD, which
  was FALSE while the function raised -- a standing claim made true rather
  than corrected. `docs/DEFERRED.md` carried no zl entry to retire,
  only DID's "the way `eos.zl` does" cross-reference, corrected in place.

- [`solve_fermi_gl` returns a density three orders wrong and says it is accurate](issues/107-fermi-gl-threshold.md)
  (2026-08-28): **the threshold moves, but to a NaN, not to `solve_fermi_t0`.**
  The ticket recommended raising the fallback threshold; measuring what there
  is to raise it *to* rules that out twice over. There is a GAP — the 30-node
  Gauss-Laguerre rule leaves the suite's 2e-3 below T/(mu - m) ~ 0.1 (6.9e-4 at
  0.1, 4.3e-3 at 0.08, 5.2e+1 at 0.01) and the T = 0 form does not enter it
  until ~0.02 — and, decisive at any threshold, **`solve_fermi_t0` returns
  s = 0 identically**, so substituting it at the T = 0.5 MeV the ticket named
  trades a density three orders wrong for an entropy 100% wrong, which is the
  same defect wearing the fallback's name. So the guard is on the degeneracy
  parameter and outside the domain the routine returns `(nan,) * 5`, following
  §6 and this module's own habit — `invert_fermi_density` twenty lines below
  already returns NaN for a target it cannot bracket. The guard's `mu > m` limb
  is load-bearing, not defensive: `mu <= m` is the non-degenerate gas the rule
  is BEST at, and it is the regime `test/dd2/test_dd2_m0.py:93` uses GL as its
  reference in, so a guard written on T alone would have broken that test.
  `GL_MIN_DEGENERACY` now lives with the routine and the verify suite imports
  it instead of re-declaring it (§7). The T < 1e-4 fallback stays and stays
  FIRST, so only the broken window 1e-4 < T < 0.1 (mu - m) changes. **No model
  number can move, measured not argued**: the only importers outside
  `eos/general/` are the verify suite and one dd2 test, both green, and a spy
  on `solve_fermi_gl` counted **0 calls** through the ccdm baseline.

- [`invert_nmp` returns ok=True when the solver never left the seed](issues/93-invert-nmp-basin-lottery.md):
  **the defect was not the verdict — the stall SUPPRESSED THE RESTARTS that
  would have solved it.** `_restart_loop` fired on `best_res >= ISO_GATE`, and a
  stall carries the published couplings' own 2.201e-03 cross-row violation,
  which sits UNDER the 2e-2 gate; so the 32 restarts never ran. They were never
  needed to be many — at DD2's own NMPs the **FIRST** jittered restart drives
  the 5x5 to 6.8e-08 and recovers K_sat to 1e-4 MeV. Feeding one condition,
  `_stalled` (the seed returned BIT FOR BIT on a residual above
  `STALL_RES = 1e-5`), into both the restart trigger and `ok` therefore turns a
  silent wrong answer into a **correct** answer, not merely into the reported
  failure §6 demanded. The 5x5 lottery is gone: seven target perturbations over
  eps in [0, 1e-8] all converge, 2.4e-10 .. 9.1e-08, `coupling_shift` 3.948e-02
  every time. **ISO_GATE stays 2e-2 and the ticket's own remedy is refuted by
  measurement**: a moved and ACCURATE 5x5 solve lands at 1.944e-03 (K_sat to
  0.0095 MeV) against the stall's 2.201e-03, so stalled and converged residuals
  OVERLAP and no threshold on the residual separates them at any value —
  the certificate had to stop being a gate reading. (`root()`'s own `success`
  flag was tried and rejected: it reports the stall, but also reports status 5
  at K_sat = 200 and 260, which land at 8.5e-08 with K_sat correct.) Restart
  coverage is a **`verify/` entry**, `_check_restarts_extend_the_basin`, 0/9 ->
  4/9 cells at 0 vs 32 restarts, ~10 s — asserting that the restarts CHANGE the
  answer, since the keep-the-best loop is monotone by construction. **Decision 4
  measured, not argued**: before the fix `from_nmp` handed back
  `gamma_sigma == 10.686681` bit for bit and `build_parametrization` said
  `stage='ok'`; both route through `invert_nmp` and inherit the fix. **Two dd2
  tests were asserting the defect** — `test_roundtrip_recovers_couplings`
  compared the published couplings with themselves, a premise `nmp.py`'s
  docstring has always denied, and both it and `test_idempotent` routed to the
  **6x6** because a whole `compute_nmp` dict carries `Q_sat`. Corrected onto the
  six imposed keys, where the 5x5 is idempotent to 3.8e-08. **What is handed to
  [ticket 105](issues/105-dd2-isoscalar-conditioning.md) is sharper than before**:
  the 6x6 now converges instead of stalling — to 1.408e-02, under the gate by a
  factor 1.4, `ok=True` — while imposing Q_sat only to **1.585 MeV**, saturating
  (64 and 128 restarts find nothing better). The "amplified noise" framing
  SPLITS the two closures rather than covering both: the noise is the whole
  story in the 6x6 and no part of it in the 5x5, whose floor is 2e-10 .. 1e-07.
  Also surfaced and handed to [ticket 103](issues/103-nmp-closures-four-models.md):
  a failed inversion is a `None` in dd2 and a `RuntimeError` in zl, and dd2's
  None reaches `solver.py` as an AttributeError.
  Gate: dd2 `run_full_check` **PASS** with the new check, golden SNM(0.16)
  `1.40e-05` and CompOSE HS(DD2) `2.83e-05` both UNMOVED; `test/dd2` 211 passed;
  full suite on python.org 3.14.2 **1812 passed, 23 skipped, 0 failed**
  (1835 collected, 46:30, exit 0).


- [Retire `phi_field`; the hidden-strange vector is controlled by its coupling](issues/102-retire-phi-field-flag.md)
  (2026-08-28): **executed, and DID's half of it dissolved rather than moved.**
  The user's ruling stood; the ticket's plan for DID did not. It said "prefer
  the refusal — the §4 statement just moves from a boolean to the coupling",
  but DID's ratios `g_phi/g_8 = -tan(theta) - c_i(z, alpha)` have **no common
  zero**, measured before anything was deleted: ideal mixing `z = 1/sqrt6`
  kills only the nucleon's, `tan_theta = 0` only Lambda's and Sigma's. So a
  `tan_theta == 0` guard would refuse a setting that is not "phi off" and say
  something false, and an all-zero-column guard would be unreachable at every
  parameter set. **User chose prose-only** (2026-08-28): DID states the sector
  is structural in `species.py`/`did.md` plus a test asserting the
  no-common-zero property over four settings. The other two moved as planned —
  `dd2` gained `Parameters.has_phi_coupling` (the `x_phi` column) and
  `from_hyperon_potentials(x_phi=None)`, a float override chosen over a
  `phi=True/False` keyword because §6 wants a knob a sampler can vary
  continuously, not the retired boolean one file over; `sfho` was **pure
  deletion**, the flag having had no reader anywhere — it existed only to raise
  on False, while `SFHo_2fam` against `SFHo_2fam_phi` was already the coupling
  switch. **Two sites the ticket's list did not carry**: `notebooks/hybrid_eos`
  rebuilds a retired run from a provenance header that records
  `flags.phi_field`, so deleting the constructor line alone would have silently
  rebuilt phi-OFF runs with the phi ON — the key now zeroes the `x_phi` column
  instead; and `docs/STRUCTURE.md:427` used `phi_field` as its worked example,
  in the paragraph directly above the "coupling happens to be zero" sentence
  the ruling had to be reconciled with. **CLAUDE.md §4 gained the owed
  paragraph** and it names that tension: there a number vanishes with nothing
  saying so, here the coupling IS the statement, documented as the sector's
  switch. Gate met — `grep phi_field eos/` empty, `dd2.npz`/`mixed.npz`
  unmoved, drift check
  `test_phi_sector_is_off_exactly_when_its_coupling_is_zero` beside
  `test_every_species_flag_defaults_off_or_raises`, **1812 passed, 23 skipped,
  0 failed**. No number could move and the mechanism says why: every
  `phi_field=False` in the tree sat beside `hyperons=False`, every
  `phi_field=True` beside a DD2Y par. **A first run lied**: exit code 0, output
  truncated at 54%, two `F`s at 31% in the band holding `test_dd2_speed.py`,
  all of it contention with the concurrent session that regenerated
  `vmit.npz` — an exit code on a truncated pytest log is not a pass, only the
  summary line is.

- [DD2's isoscalar closure carried DD's constraint, not DD2's](issues/105-dd2-isoscalar-conditioning.md)
  (2026-08-28): **the ticket's two analytic proposals were both refuted by
  measurement, and the literature check moved the ground under the question.**
  Typel, PRC 71, 064301 (2005) §IV imposes f''_sigma(1) = f''_omega(1) on the
  rational couplings *"in order to reduce the number of free parameters"* —
  parameter economy, stated, so the "sigma and omega run together" gloss
  invented a motivation — and counts **eight** parameters for DD; Typel et al.,
  PRC 81, 015803 (2010) states only f_i(1) = 1 and f_i''(0) = 0 and counts
  **ten**. The difference of one IS the constraint, and the tables agree:
  f''_sigma(1) − f''_omega(1) is **−6.0e-08 for DD and 2.201e-03 for DD2**. The
  2.200718e-3 this map has quoted for months was never a fit imperfection, it
  was the constraint's absence, and `invert_nmp` had been closing DD2 with DD's
  condition. **The user ruled it out entirely** (2026-08-28): six isoscalar
  couplings free, isovector fixed by E_sym and L_sym.
  **Proposal 1 is refuted and the refutation is invariant**: the
  reparametrisation is block-diagonal, and principal angles between the
  sigma-shape and omega-shape planes (0.000° and 12.62°) cannot be changed by
  that class of map. The zero is forced because **E_sat and m*/m are
  shape-BLIND** at fixed n_sat — four shape knobs answer to three rows, shape
  singular values [77.0, 7.16, 1.94, **1.8e-10**], an exact rank deficiency of
  one. So the diagnosis is four knobs in a three-dimensional response, not
  "four columns are one direction"; running the reparametrisation moves cond
  2623 → **3082, worse**. **Proposal 2 is the only remedy and it is deferred**:
  the best closure that imposes Q_sat conditions at 259 and 259 × 1.5e-3 = 0.39
  relative coupling error, so no conditioning rescues a stencil Q_sat and
  proposal 1 buys 3.8× where 1e9 is needed →
  [ticket 111](issues/111-dd2-analytic-nmp-derivatives.md), on the user's Q4
  ruling that all relations writable analytically should be.
  **The shipped closure is now four rows with `b_sigma` and `c_omega` pinned**
  (cond **128**), which **refutes this ticket's own fallback ordering**: one pin
  per meson beats holding the omega shape whole (354) or the sigma shape whole
  (305), because what should stay free is the least collinear surviving pair.
  A sixth NMP is refused — Z_sat is the only candidate, nobody quotes it, its
  fourth difference spans **4.8e+04 on a value of 4547**, and the closure
  conditions at **550340**. The payoff beyond the ruling: **the published DD2
  couplings are a root of their own inverse map again** — the round trip
  returns them to **1.1e-05** where the old closure landed 3.9% away, and the
  predicted Q_sat goes **117.5 → 168.5** against a forward 168.9, so ticket 93's
  withdrawn `test_roundtrip_recovers_couplings` is true again and for a reason.
  A `Q_sat` key now selects nothing (`impose_Q_sat` defaults to a plain False).
  **Reported not fixed, and handed to 111**: ISO_GATE = 2e-2 was widened for
  the cross row and for Q_sat's stencil and both reasons are retired — on a
  105-cell (K_sat, m*/m) grid the 101 passing cells split 95 below 1e-5 and 6
  in [1e-3, 2e-2] with nothing between, so six are certified without being
  roots; 93's "cannot be tightened" rested on the premise this ticket removed.
  **`CLAUDE.md` §5 stated the cross-constraint as specification** and was
  corrected, gaining the general sentence: a closure condition belongs to the
  parametrization that imposed it, checked against the paper that fitted THAT
  set. Gate: dd2 `run_full_check` PASS, golden SNM(0.16) 1.40e-05 and CompOSE
  HS(DD2) 2.83e-05 unmoved, `test/dd2` 211 passed; no published number moved
  (every forward path is untouched — what moved is what the INVERSE returns).

## Not yet specified

Two patches graduated on 2026-08-27 out of
[ticket 99](issues/99-quark-ea-at-zero-pressure.md) and are now
[ticket 100](issues/100-vmit-point-Y_S-never-assigned.md) (a cached conserved
fraction its solvers never fill, frozen into a baseline) and
[ticket 101](issues/101-pressure-and-energy-field-names.md) (`P`/`eps` against
`P_total`/`e_total`, six models against four). Ticket 100's own sweep then
graduated a third on 2026-08-28:
[ticket 108](issues/108-cached-lepton-fraction-three-models.md) — `Y_L` is the
trapped mode's condition in three models and the measured lepton fraction in
three others, and the zeros are a defect or correct depending which.

In scope, not yet sharp enough to ticket:

- **`test/baseline_py39/` is a snapshot nothing selects and nothing reproduces.**
  Found while gating [ticket 107](issues/107-fermi-gl-threshold.md).
  `generate_baseline.path_for()` returns `HERE / f"{name}.npz"` unconditionally,
  so `test/baseline_py39/` is never read by `test_baseline` on any interpreter —
  and it is staler than the directory that IS read: fresh ccdm on Python 3.9.7 /
  numpy 1.26.4 differs from `baseline_py39/ccdm.npz` in **1256** of 6068 keys,
  against 108 for `baseline/ccdm.npz`. Either it is dead weight to delete or
  `path_for` is meant to branch on `sys.version_info` and never did; which one
  has not been read, and the answer decides whether §12's "frozen at
  rtol = 1e-10" is claimed for one interpreter or two.

- **The baselines are interpreter-bound and §12 does not say so.** Same
  gate: `test_baseline[ccdm]` fails on Python 3.9.7 / numpy 1.26.4 (108
  quantities, worst `pattern.2SC.n1.5.T0.x` at 8.8e+03 relative, a CSC pattern
  selection, plus `state.field_residual` at abs 1e-9 to 1e-5) while passing on
  the canonical python.org 3.14.2 / numpy 2.3.5. It is NOT a regression — a spy
  counted 0 calls to `solve_fermi_gl` through that baseline, and a concurrent
  canonical run cleared `test/baseline` with 0 failures. `abpr` and `alphabag`
  pass on 3.9.7, so the split is per-model, not global. Whether an rtol = 1e-10
  freeze is meaningful across interpreters at all, or whether §12 should name
  the stack the way a suite count does, is the question — and it is the same
  question the `baseline_py39/` patch above asks from the other end.

- *(A patch graduated on 2026-08-27: the mode list divergence is now
  [ticket 98](issues/98-fixed-ys-undeclared-mode.md), surfaced by a BayEoS
  design session reading `eos` as a downstream consumer. `fixed_YS` is a
  sixth mode reachable through dd2's public `eos_point` that CLAUDE.md §3
  does not declare, and the shared `ModeSpec` can build a seventh.)*

- **`active_baryons` is a second instance of the finding ticket 84 named, and
  nobody has counted the rest.** Ticket 86 moved `hadronic_qn` /
  `hadronic_charges` out of `eos/dd2/species.py` into `general/basis.py` under
  §7's single-home rule: general-purpose functions that read only the shared
  `Particle` objects and a flags object, sitting in a model package because
  that is where the first caller happened to be. `active_baryons` travelled
  with them because `hadronic_qn` is written in terms of it — and
  `eos/did/species.py` and `eos/sfho/species.py` still carry their own,
  byte-identical copies, which ticket 86 left alone because its charter was
  `eos/mixed`. That is three copies of one function and a fourth in
  `general/`. The open question is not those three: it is whether a sweep of
  every model's `species.py` (and `parameters.py`, and the small helpers
  around them) finds more of the same shape, and what the test is for "this
  belongs in `general/`" that does not also drag genuinely per-model physics
  up a layer. Sharpen after someone has actually counted.

- **A rename landing green is not a rename landing safe.** A concurrent session
  is renaming `eos.sfho.create_custom_parametrization` -> `from_potential_depths`
  (uncommitted at the time [ticket 80](issues/80-phase6-conformance.md) ran).
  Two `nucleation` consumers import the old name — `test/make_fixture.py:98` and
  the paper notebook — and **neither is on the suite's import path**:
  make_fixture imports it lazily inside `main()`, and a notebook is not
  collected at all. So the rename will show a green nucleation suite and break
  both silently, exactly the shape ticket 24 hit when 38 of 38 modules could not
  import while the suite reported nothing. The question is not this one rename;
  it is that the cross-repo call-site check has a blind spot wherever an import
  is lazy or lives in a notebook, and no gate covers it. Related to the
  stated-limitation rot below, but the opposite direction: not a comment
  outliving behaviour, a CONSUMER outliving a name.

- **Smoke mode cannot complete, and both documents tell a reader to run it
  first.** `notebooks/2fam_PNS_nucleation.py:1935` clips `F8_SHOW = [1, 3]`
  against a single-`alpha_s` smoke grid, gets `[]`, and `pd.concat([])` raises
  at Figure 5. Pre-existing and production-safe; found by
  [ticket 80](issues/80-phase6-conformance.md) and left undiffed under the
  only-what-the-ticket-asks rule. The sharp question is not the one-line fix but
  whether a "prove every cell runs before you commit hours" path deserves a gate
  of its own, given that this is the second smoke-only shape bug the same cell
  has carried (its comment records the first, `F8_SHOW = [0,1,2,3]` on a 2x2).

- **The paper's tracked figures no longer match what the code produces.**
  Ticket 80's production run regenerated 23 tracked files under `output/paper/`
  and restored every one: 14 PDFs differed only inside `/CreationDate`, and the
  9 real changes are round-off — largest 13 cm on an 11.14 km `R_1.4`, with
  `sigma_crit_star` bit-identical across all 398 rows. Restoring was right for a
  conformance ticket, but it leaves the tracked figures as the pre-refactor
  ones. Whether to re-commit the regenerated set is a publication decision for
  the user, and it wants deciding before the repository goes public rather than
  discovering later that a reader's rerun does not match the committed CSVs.

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
  **DONE (ticket 62).** The copy was taken after the regeneration, and
  additively: `~/Desktop/Research/backups/baseline/` still holds the 3.9 set,
  `backups/baseline_py314/` holds the verified canonical set, and a
  `backups/README.txt` says which is which — so the superseded set is preserved
  and labelled rather than left looking current. The 3.9 files are also on disk
  at `test/baseline_py39/`. What remains true is the general point: a hand copy
  is only as fresh as the last time it was taken. The underlying question — whether
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

  **GRADUATED to [ticket 75](issues/75-undetermined-potential-check.md).** Both
  reasons it was fog have cleared: `eos/general/verify/` now exists
  ([ticket 64](issues/64-general-verify-suite-missing.md)), and the screen has a
  second witness taken PREDICTIVELY rather than after the fact —
  [ticket 62](issues/62-regenerate-baselines-py314.md) ran it forward over 53763
  keys and it separated them correctly in both directions, firing on `ccdm` and
  `njl`'s CFL `mu_3 = mu_C` (with `mu_8` at exactly half). What stays open is the
  form, and that is now ticket 75's question rather than fog.
  **Amended by [ticket 72](issues/72-enjl-branch-selection.md):** the third case
  was read backwards here. The screen DID fire on `enjl` — `mu_S` at Y_S = 0,
  where no populated species carries S — and ticket 62 filed that as a footnote
  while classifying the O(1) move beside it as unrelated physics. It was the
  same finding: the undetermined potential is a null Jacobian column, and the
  three decades of residual it costs are what let round-off pick the chiral
  branch. Three witnesses, not two, and the third says the check must catch a
  CONDITIONING hazard and not only a wandering key.
- **Whether other tests silently degrade on missing data.** Ticket 39 fixed the
  two TOV helpers and generalised their guard to every `CRUST_FILES` name, but
  nothing has swept the rest of `test/` for the same pattern — an absent input
  turned into a wrong number rather than a skip. §10 promises the constraints
  module fails with a fetch message; whether it does, and whether anything else
  does not, is unmeasured.

## Out of scope

- **Implementing the nine missing composition freezes.** Ruled by
  [ticket 78](issues/78-composition-freeze-nine-models.md), which decided the
  order and stopped there. Nothing in the Acceptance criteria block measures a
  response freeze, and `docs/DEFERRED.md` is the repository's own tracked ledger
  for a per-model gap carried past the refactor — so the order lives there, not
  as nine tickets on this map. Returns only if the destination is redrawn.

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

- [dd2's two remaining `cs2` names, neither of which is a rename](issues/73-dd2-remaining-cs2-names.md):
  **executed as ruled; `cs2_ad` and `TableResult.cs2_eq` are gone from
  `eos/dd2`.** `eos_response` now returns `cs2_isothermal` AND `cs2_adiabatic`
  at BOTH freezes — composition on the `frozen=` argument, thermal in the key —
  and `TableResult` carries two fields with **exactly one populated**, chosen by
  `spec._temp_key` (`T` -> isothermal, `SnB` -> adiabatic), the other `None`.
  Five models now derive the pair through C_P/C_V the one way.
  **The equilibrium freeze cost one multiplication, as the ruling predicted;
  the composition freeze did not, and the ruling did not claim it would.** dd2's
  `C_V`/`C_P` are taken along the BETA-EQ sequence, so they are the wrong ratio
  for a frozen-`Y_p` speed: `responses.py` gained `_frozen_derivs`, one central
  stencil along the fixed-`Y_p` sequence, and builds both heat capacities from it.
  **The function names had to move with the keys** —
  `responses.sound_speed_adiabatic` WAS Zhao's `c_s` (frozen composition at
  fixed T), so it split into `sound_speed_isothermal_frozen` (same stencil
  byte-for-byte, number unmoved) and `sound_speed_adiabatic_frozen`. Its three
  outside importers are all tests; **no `eos/mixed` or `eos/astro` source file
  imports it**, so neither package was touched.
  **The four notebooks needed no reader change** — the ruling expected a two-key
  reader, but ticket 69 had already collapsed them to one `cs2_isothermal` read;
  only a stale sentence naming `did` as the sole model returning the pair was
  corrected, in the `.py` and its `.ipynb`.
  One gap opened and RECORDED, not left silent: `Gamma` still stands on
  `cs2_isothermal`, so at T > 0 it is the isothermal index — restanding it would
  move a number, which a naming ruling does not authorise (`docs/DEFERRED.md`).
  Both documents state the clash as physics now, `dd2.tex` printing Zhao &
  Lattimer Eq. (1) and citing it against `TypelCompOSE2015`.
  Gates, python.org 3.14.2, never under `timeout`: a concurrent session's
  in-flight `eos/astro/gmode/sound_speeds.py` breaks `test/gmode` at COLLECTION
  on the live tree, so the suite ran as an **isolated-copy pair** with that file
  reverted on both sides — **1671 passed / 11 failed** against control **1670 /
  11**, the **same eleven failures name for name**, the +1 this ticket's new
  test. Eight are the reverted gmode file meeting that session's edited tests;
  `test_baseline[ccdm]` fails in both copies and PASSES live, so it is an
  isolation artifact too. Live `test/baseline`: **1 failed (enjl), 15 passed**,
  identical to the before-image. `test/dd2` **207 passed**, dd2 `run_full_check`
  **PASS**. **No baseline number moved, no tolerance touched.**
