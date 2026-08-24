# Apply the approved renames — eos/mixed and eos/did

Type: task
Status: open
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
