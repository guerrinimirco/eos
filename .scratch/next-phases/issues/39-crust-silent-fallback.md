# A missing crust table silently becomes a 0.9 km physics error

Type: grilling
Status: open
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
