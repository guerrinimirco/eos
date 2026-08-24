# Establish the pre-existing failure baseline on main

Type: task
Status: resolved
Parent: ../map.md

## Question

Run `pytest test/ -q --tb=no` on `main` and save the failure list verbatim to
`output/_audit/pytest_before.txt`. Memory records roughly a dozen failures on the
`enjl-finite-T` line; `main` is 64 commits ahead of it, so that count is not
transferable and must be measured here.

Resolved when the file exists, and the answer records: the exact count, the test
ids grouped by model, and which failures look like the known
`docs/DEFERRED.md` entries versus which are unaccounted for.

Every later ticket reports failures *added* against this file. Nothing here is
fixed — this ticket only measures.

## Answer

`python -m pytest test/ -q --tb=no` on `main`, in a clean process with no edits
during the run, saved to `output/_audit/pytest_before.txt`:

    4 failed, 1634 passed, 26 skipped, 2 warnings

    FAILED test/baseline/test_baseline.py::test_baseline[did]
    FAILED test/dd2/test_dd2_m4_tov.py::test_tov_dd2_nucleonic_pipeline
    FAILED test/did/test_did_tov.py::test_mass_radius_matches_the_paper[DID-flags0]
    FAILED test/did/test_did_tov.py::test_mass_radius_matches_the_paper[DIDY-flags1]

**Three of the four are `did`.** Grouped by likely cause, they become
[ticket 37](37-did-failures.md) and [ticket 38](38-dd2-tov-radius.md).

**This supersedes two earlier counts, both wrong.** The memory note carried in
from the `enjl-finite-T` line said 12 failures; `main` is 64 commits ahead and
the set is different. And a first run of this suite reported **8**, adding
`test_baseline` failures for `ccdm`, `njl`, `sfho` and `vmit` — those were an
artefact of that run spanning a source edit in `eos/dd2/`, which forced Numba
kernel recompilation mid-suite. `test/baseline/test_baseline.py`'s own docstring
predicts exactly that: the kernels are built with `fastmath`, so recompilation
moves the last digits. A standalone re-run of all five generators in a fresh
process confirms it — `ccdm` (6068 keys), `njl` (6594), `sfho` (3111) and `vmit`
(1121) reproduce with **zero** mismatches and no key-set change.

**Method note for every later ticket: never edit source while the suite runs.**
A run that spans an edit is not a usable baseline. The suite takes 36 minutes
uncontended (1h42 when competing with other background jobs).

Status: resolved.

## Addendum — the real baseline needs the crust data

Re-run with `EOS_CRUST_DIR=/Users/mircoguerrini/Desktop/Research/Crust`, saved to
`output/_audit/pytest_before_with_crust.txt`:

    1 failed, 1648 passed, 15 skipped, 2 warnings in 42:15

    FAILED test/baseline/test_baseline.py::test_baseline[did]

**One failure on the whole repository**, and it is the underdetermined `mu_S`
that [ticket 40](40-determine-mu-s.md) rules on. The three others were the
missing BPS crust ([ticket 38](38-dd2-tov-radius.md),
[ticket 39](39-crust-silent-fallback.md)).

The counts reconcile exactly: 1634 + 11 crust-dependent tests that were being
skipped + 3 recovered failures = 1648, with skips dropping 26 -> 15. So the
crust table does not only fix three assertions, it **unskips eleven tests** that
were silently not running.

**Which file is the baseline?** `pytest_before.txt` (4 failures) is the honest
record of a bare checkout; `pytest_before_with_crust.txt` (1 failure) is the
record of a configured one. Later tickets should compare against the **crust**
file and state that they set the variable — otherwise three known-good tests read
as added failures. Where the crust tables should live so this needs no
environment variable at all is [ticket 39](39-crust-silent-fallback.md).

Only `.tex` and `.md` files changed during this run; no Python was touched, so
it does not repeat the contamination described above.
