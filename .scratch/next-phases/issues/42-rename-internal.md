# Apply the approved renames — eos/mixed and eos/did

Type: task
Status: resolved
Blocked by: 10
Parent: ../map.md

## Question

The rehearsal. Both packages are fully internal: `nucleation` imports neither,
and `eos.mixed` is imported only by `eos/` and `test/`. Smallest blast radius on
the list, so this is where the mechanical procedure is proven before it runs at
dd2 and vmit scale.

**`eos/mixed` — 11 renames, all §13 rule 1 (the name repeats the package):**

    MixedResult          -> Result
    solve_mixed          -> solve
    sweep_mixed          -> sweep
    solve_mixed_at_entropy -> solve_at_entropy
    MixedTableSpec       -> TableSpec
    build_mixed_table    -> build_table
    MixedEoSTable        -> EoSTable
    MixedWindow          -> Window

NOT in this ticket: `find_mixed_window` -> merge into `locate_window`, and
`build_mixed_eos_table` -> a name distinguishing it from `build_table` by job.
Both are [ticket 46](46-api-changes.md) — a merge and an unnamed rename are not
mechanical.

**`eos/did` — 1 rename, §13 rule 3 (a name says what it takes and returns):**

    evaluate             -> baryon_kinetics

matching `dd2/thermodynamics.py:388`. `evaluate` says nothing about either side.

Resolved when both packages are renamed, `test/test_imports.py` passes, the
`verify/` suites run, and the added-failure count against
`output/_audit/pytest_final.txt` is reported. Note `test/baseline/` pins these
models at rtol = 1e-10 — a rename must move NO number, so a non-zero diff there
means the rename was not mechanical.

## Answer

**Applied. 9 renames across 16 source files, 1648 passed / 15 skipped / 0 failed,
0 added failures, and `test/baseline/` unmoved at rtol = 1e-10 — so the rename
changed no number, which is the whole test of a rename.** Saved to
`output/_audit/pytest_after_ticket42.txt`.

    eos/mixed   MixedResult -> Result           solve_mixed -> solve
                sweep_mixed -> sweep            solve_mixed_at_entropy -> solve_at_entropy
                MixedTableSpec -> TableSpec     build_mixed_table -> build_table
                MixedEoSTable -> EoSTable       MixedWindow -> Window
    eos/did     evaluate -> baryon_kinetics

`evaluate` was rewritten only under `did` paths: it is a generic word elsewhere
and a repo-wide substitution would have hit unrelated code. All 13 `did` sites
were the one function at `thermodynamics.py:215`.

Five files outside the two packages changed, all docstring cross-references
(`astro/gmode/sound_speeds.py`, `dd2/solver.py`, `dd2/table.py`,
`general/table_io.py`, `sfho/table.py`). That is a small piece of evidence for
§13's premise: these names were repo-wide vocabulary, not local labels.

Left untouched for [ticket 46](46-api-changes.md), as ruled: `find_mixed_window`
(4 sites) and `build_mixed_eos_table` (27 sites). Both are merges or
renames-without-a-proposed-name, not mechanical.

### What the rehearsal was for

**The renames were not mechanical, and the failure was silent.** `eos/mixed/api.py`
imported `solve_mixed` and separately defined a nested adapter
`def solve(temperature)`. Renaming the public name onto the local one made the
function call itself, and because `RecursionError` subclasses `RuntimeError`, the
surrounding `except (RuntimeError, ValueError)` caught it and returned
`PointResult(False, "did not converge")`. **Twelve tests failed with nothing in
any traceback naming the cause** — a rename had quietly converted a working
solver into a non-converging one, exactly the failure mode a green baseline is
supposed to catch and nearly didn't.

Fixed by moving the LOCAL name (`solve` -> `point_at`), not the public one: §13
fixes the vocabulary, so the adapter is what yields.

The pattern is systematic rather than unlucky — this codebase already used §13's
vocabulary (`solve`, `warm_start`, `sweep`) for local adapters, which is what the
public names are being renamed TO. An AST check for it is recorded on
[43](43-rename-vmit.md), [44](44-rename-dd2.md) and [45](45-rename-sfho.md), and
run predictively it already found the next one before any code moved:
`eos/vmit/table.py:188` has `def warm_start(point): return result_to_guess(...)`,
which ticket 43's `result_to_guess -> warm_start` would break identically.
`dd2` and `sfho` are clean for their planned renames.
