# The five items on the rename list that are not renames

Type: task
Status: resolved
Assignee: session eff950ed
Blocked by: 10
Parent: ../map.md

## Question

Ticket 10 approved the renames and split these out: each changes a signature or
deletes behaviour, so a naming gate does not authorise them. Each needs its own
ruling before any code moves.

1. **Delete `get_vmit_custom()`** — 6 `eos/` + 12 `test/` + 13 notebook sites.
   What replaces those callers? If the answer is `Parameters(...)` directly,
   say so and the deletion is mechanical; if it carries defaults nobody has
   written down, deleting it loses them.

2. **`solve_isentropic_beta_eq` / `solve_isentropic_trapped` fold into `SnB=`**
   on the mode solvers. CLAUDE.md §3 already blesses the shape ("wherever a
   temperature axis is accepted, entropy per baryon `SnB` is accepted in its
   place (an outer 1-D solve for T)"), so the question is not whether but
   whether sfho's outer solve is the shared one or its own.

3. **`find_mixed_window` merges into `locate_window`** — a merge, not a rename.
   Confirm they are one job; §5 makes the located window part of the mixed
   result, so the merged name has to serve both callers.

4. **`get_sfho_general(...)` and `create_custom_parametrization(...)` become
   `from_*` constructors.** §5 is explicit that an NMP-inverting constructor is
   a FREE FUNCTION in `nmp.py`, not a classmethod on the parameter dataclass —
   "putting it there forces a deferred import, which is the cycle announcing
   itself". So the ruling is which of the two is an NMP inversion (that one goes
   to `nmp.py`) and which is a plain alternate constructor.

5. **`build_mixed_eos_table` needs a name distinguishing it from `build_table`
   by job** — nobody has proposed one. §5 says a mixed table is "rows +
   windows", so the distinction to name is probably which of those two the
   caller gets.

Resolved when each of the five is ruled and the approved ones applied, with the
added-failure count reported.

## Ruling

All five ruled, four of them **settled by measurement rather than preference**.

1. **Delete `get_vmit_custom`.** `eos/vmit/parameters.py:71` defaults are
   `m_u=5.0, m_d=7.0, m_s=150.0, a=0.2, B4=180.0` — and `Parameters` carries
   **identical** defaults. So `get_vmit_custom(B4=170.0, a=0.15)` IS
   `Parameters(B4=170.0, a=0.15)`: a pure alias carrying no undocumented
   physics. Mechanical, 31 sites. The replacement sentence in vmit's document
   is owed by [ticket 79](79-parametrization-surface.md).
2. **Fold the isentropic solvers into `SnB=`.** `eos/general/tabulate.py:78
   temperature_at_entropy(...)` ALREADY IS the shared outer 1-D solve, with
   `TEMPERATURE_AXES = ("T", "SnB")` declared beside it; sfho's
   `solve_isentropic_beta_eq/_trapped` (`solver.py:735,761`) are a private
   second copy. §3's sentence is implemented — sfho just does not use it. Fold
   into the shared one.
3. **Merge `find_mixed_window` into `locate_window`.** `boundaries.py:107` and
   `solver.py:792` have **identical signatures**
   `(par, flags, n_B_grid, eta, spec, vmit_params=None, T=0.0, ...)`. One job,
   two names, two modules. `boundaries.py` wins — it also holds
   `locate_windows`.
4. **Confirmatory, no module moves.** `create_custom_parametrization` is already
   in `eos/sfho/nmp.py:233` (it is the NMP inversion, correctly placed by §5);
   `get_sfho_general` is in `parameters.py:568` and is the plain alternate
   constructor. Rename each to a `from_*` form in place.
5. **`build_mixed_eos_table` -> `build_hybrid_table`.** `hybrid.py:118` stitches
   hadronic + mixed + quark into one branch; `table.py:323 build_table` is
   §13's grid driver and keeps the vocabulary word. The name says the job.

Only item 5 is a preference; the other four are settled by what the code
already contains.

Open for execution.

## Resolution

**All five applied; 0 added failures; `test/baseline/` unmoved at rtol = 1e-10.**
Committed `95d4052`, 19 files.

**Two of the four "settled by measurement" premises are FALSE, and both were
measured rather than argued.** The ruling was executed in the shape its own
`## Question` asks for, not the shape its `## Ruling` names.

- **Item 2's premise is wrong: sfho's isentropic solvers are not a second copy
  of `temperature_at_entropy`.** They add T as a 7th (or 8th) unknown with
  `s/n_B - SnB` as a residual row (`solver.py` `System.isentropic`,
  `unknown_names`, `residual`) -- a coupled Newton solve, where the shared
  function is an outer `brentq` at `xtol = 1e-5`. The two disagree, at the
  three densities `test/baseline/sfho.npz` freezes:

        n_B     T (coupled)      T (outer)     rel
        0.16   11.4115136397  11.4115141115  4.13e-08
        0.32   19.3770133836  19.3770133941  5.39e-10
        0.64   31.8777857611  31.8777840272  5.44e-08

  Three to four orders above the 1e-10 gate, so routing sfho through the
  shared solve WOULD have moved a frozen number -- the outcome the ticket says
  to stop and report rather than absorb. Executed instead as the question's own
  words, "fold into `SnB=` on the mode solvers": the two wrappers are gone and
  `solve_beta_eq_neutrinoless` / `solve_beta_eq_neutrino_trapped` take
  `SnB=` beside `T`. Bit-identical by construction -- `_system` already routed
  both axes -- and the baseline confirms it.

- **Item 3's premise is wrong: `find_mixed_window` and `locate_window` are not
  one job and their signatures are not identical.** `locate_window` returns a
  `Window` (the two boundaries, found by bisecting the chi crossings) and takes
  eleven more keyword arguments; `find_mixed_window` returned the LIST of mixed
  points on the grid. `find_mixed_window`'s own docstring said so. Nothing can
  merge into `locate_window` without changing its return type, and there was
  nothing to merge: it was a one-line alias for `sweep(..., mixed_only=True)`
  with zero callers in `eos/`, `test/` or `notebooks/`. **Deleted, item-1
  style. `boundaries.py:locate_window` is untouched.**

**One collision, not predicted, and structurally invisible to the AST check.**
All four `get_vmit_custom` call sites already bind `Parameters` to **DD2's**
dataclass (`test/mixed/test_muons_and_mesons.py:26`, `test_hyperons.py:22`,
`test_window_location.py:24`, `test/tov/test_solver_fast_robustness.py:84,186`,
and `eos/mixed/scan.py:600` aliases DD2's as `P`). A bare `Parameters(...)`
substitution would have silently built a hadronic parameter set and handed it
to a quark phase. Every site now imports `Parameters as VMITParameters`,
which is what `eos/mixed/adapters.py:55` already does. The check cannot see
this class: it flags names the ticket INTRODUCES, and `Parameters` is not new
here -- it is the name the deletion makes callers reach for. **That is the
third distinct shape of the ticket-42 trap**, after 42's local adapter and 43's
function-local binding: a *cross-model* collision on a §13 vocabulary name that
ten models share. The check itself ran clean, both shapes, before and after.

**The other three, as ruled.**

1. `get_vmit_custom` deleted -- defaults identical to `Parameters`', confirmed.
   23 name occurrences in the repository (6 `eos/`, 12 `test/`, 2 notebook,
   3 doc), not 31; the ruling's count included its own `.scratch` files. The
   `vmit.md` sentence now names `Parameters(...)`; the replacement prose is
   still [ticket 79](79-parametrization-surface.md)'s.
4. Confirmatory and confirmed. `from_potential_depths` in `nmp.py`,
   `from_coupling_ratios` in `parameters.py`. No module moves.
   `docs/REFACTOR_PLAN.md:66` was deliberately NOT renamed: it records what the
   refactor DELETED, and the two implementations carried the old name then.
5. `build_mixed_eos_table` -> `build_hybrid_table`, 29 occurrences over 14
   files; `build_table` keeps §13's vocabulary word. Continuation lines
   re-aligned -- the new name is five columns shorter.

**Measurement.** python.org **3.14.2** / numpy 2.3.5 / scipy 1.17.0.
`test/baseline test/mixed test/sfho test/vmit test/tov`, **collected 417**:

    before  1 failed, 401 passed, 15 skipped   output/_audit/pytest_before_ticket46_py314.txt
    after   1 failed, 401 passed, 15 skipped   output/_audit/pytest_after_ticket46_py314.txt

Same node id both times -- `test_baseline[enjl]`, the survivor ticket 74 leaves
red on purpose -- and the `^E ` diff over its 22 assertion lines is EMPTY:
**0 added, 0 cleared.** `test_baseline[sfho]`, `[vmit]`, `[dd2]`, `[mixed]` and
`[zlvmit]` all pass, which is `test/baseline/` unmoved at rtol = 1e-10.
The `sfho`, `vmit` and `mixed` `verify/` suites all pass.

**Staging note.** Another session held 4 of `docs/DEFERRED.md`'s 7 hunks
(gmode, dd2 responses) in the working tree. `git commit -- <pathspec>` commits
the WORKING TREE at those paths and ignores a filtered index, so the first
attempt swept their work into this commit; amended, and their four hunks are
back in the tree uncommitted. **A filtered index does not survive a pathspec
commit** -- stage, then `git commit --amend` with no pathspec.

**Two defects found, not fixed** (the hard rule):

- `eos/sfho/sfho.md` and `sfho.tex` never mention the entropy axis at all --
  not the row, not the extra unknown, not `SnB`. A §11 gap that predates this
  ticket and is [ticket 35](35-sfho-documents.md)'s territory.
- `notebooks/zlvmit_test.ipynb:20` and `ZLvMIT_hybrid.ipynb:66` import
  `VMITParams`, a name ticket 43 removed, so both were already broken before
  this ticket touched anything. Left as found; the map rules the second out of
  scope and [ticket 41](41-corrupt-notebooks.md) records it unopenable.
