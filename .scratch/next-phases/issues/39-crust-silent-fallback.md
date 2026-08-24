# A missing crust table silently becomes a 0.9 km physics error

Type: grilling
Status: resolved
Parent: ../map.md

## Question

`test/dd2/dd2_tov_sequence.py` and `test/did/did_tov_sequence.py` both open with

    if crust == "BPS" and not have_crust("BPS"):
        crust = "No"

and then run a TOV sequence whose result is asserted against published values
that **include** a crust. When the table is absent, the assertion does not say
"data missing" — it reports a star 0.64 to 0.87 km too small, which reads as a
physics regression in the model. That is what three of the four failures on
`main` were ([ticket 37](37-did-failures.md), [ticket 38](38-dd2-tov-radius.md)),
and it cost a full diagnosis to tell apart from real drift.

`eos/astro/tov/crust.py` is **not** at fault and should not change:
`crust_path` raises `MissingCrustData` naming the file, every directory searched
and the `EOS_CRUST_DIR` override — precisely the §10 standard of failing with a
message saying how to fetch the data. `have_crust`'s own docstring even warns
that "falling back to no crust is not free — it moves M_max by about 1%". The
callers took the fallback anyway, and for `R_1.4` the cost is far larger than 1%.

The decision has two parts.

**1. How should a crust-dependent test behave when the table is absent?**

- **Skip** (`pytest.skip("BPS crust table not found; set EOS_CRUST_DIR")`) — the
  assertion is meaningless without the data, and a skip says so. Matches how
  `test/baseline/test_baseline.py` already handles its missing `.npz` files.
- **Fail loudly** with the `MissingCrustData` message — the test suite is
  supposed to be green on a configured machine, so silence hides a broken setup.
- **Keep the fallback but assert crust-free reference values** — needs a second
  set of published numbers nobody has.

**2. Where should the crust table live so this stops recurring?**

Today `crust_search_path()` yields only `<repo>/data/crust`, which does not exist
in this checkout, while `BPST0.dat` sits at
`/Users/mircoguerrini/Desktop/Research/Crust/`. Options: create
`<repo>/data/crust` and put (or symlink) the tables there, so a plain `pytest`
works with no environment set up; or document `EOS_CRUST_DIR` as required and
have the suite say so. Note `CRUST_FILES` also names three CompOSE tables
(`SFHO_Compose/eos.thermo.ns` and two others) with the same exposure.

This belongs in `docs/STRUCTURE.md` ([ticket 21](21-phase5-structure.md)) and in
the README's run instructions either way — a fresh clone currently produces three
failures that look like physics.

## Answer

Both halves settled, in the order they mattered.

**2. Where the tables live — in the package.** The premise that put them outside
it does not hold: `BPST0.dat` is **4.8 kB**, and all three available tables come
to 1.1 MB, well inside the 5 MB bar. They now sit in `eos/astro/tov/data/`,
which is the **second** search root — after `$EOS_CRUST_DIR` so an explicit
override still wins, before `<repo>/data/crust` so existing checkouts keep
working. `pyproject.toml` ships them under `package-data`, following the pattern
already used for the constraint contours, because every name in `CRUST_FILES` is
a runtime option of `solver.py`'s `add_crust_table` — an installed wheel without
them would lose the crust silently.

A fresh clone now runs the crusted TOV path with no environment set up, and the
suite stops skipping eleven crust-dependent tests it was quietly passing over.
Commit `7ff8627`.

**1. How a crust-dependent test behaves when the table is absent — it skips.**
The fallback in both helpers is replaced by `pytest.skip` naming the missing
table and where the tables ship. Verified both ways: 6 passed with the data
present, 2 skipped with the message when it is hidden. This matches how
`test/baseline/test_baseline.py` already handles its missing `.npz` files, and
it is the option the ticket recommended — the assertion is meaningless without
the data, so a skip says so where a silent downgrade said "your model is wrong
by 0.9 km".

The generalised form is worth keeping: the guard is now
`if crust != "No" and not have_crust(crust)`, so it covers the three CompOSE
tables too, not just `"BPS"`.

**One caveat.** `test/` is gitignored by §11 ("kept locally, gitignored, not
published"), so the two helper edits are **not** in git — only the data move and
the search-path change are. Anyone reconstructing `test/` from scratch can
reintroduce the silent fallback. Whether that is acceptable, or whether the
guard belongs in `eos/astro/tov` where it would be tracked, is worth a thought
during [ticket 21](21-phase5-structure.md) when the run instructions get written.

Status: resolved.
