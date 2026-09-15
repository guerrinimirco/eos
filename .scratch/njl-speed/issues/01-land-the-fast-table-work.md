# Land the uncommitted NJL fast-table work and open `njl-speed`

Type: task
Status: closed
Blocked by: -
Parent: ../map.md

## Question

Nothing to decide about physics. This map's every later number is a comparison
against a baseline, and there is no baseline while the tree is dirty.

`git status` carries **two unrelated bodies of work** in `eos/` and they must
not land as one commit:

    eos/njl/{__init__,api,solver,table}.py   the fast-table / solve_nodes work
    eos/mixed/adapters.py                     njl_phase seeding + wing_sweep
    notebooks/quark_timing.{py,ipynb}         the timing harness edits
    ---- unrelated, J0614 observational-constraint work ----
    eos/general/constraints/{__init__,build}.py
    eos/general/constraints/data/J0614_*
    notebooks/hybrid_stars.{py,ipynb}         untracked
    plot/                                     untracked, several files

The first group is the 2026-09-07 work recorded in memory as shipped but
uncommitted: `TableSpec.solve_nodes`, `build_fast_table`, `branch_ladder`,
`solve_pattern(..., rescue=False)`, and the `njl_phase` seeding fix whose
`seed_cacheable` rule is load-bearing. It is what every measurement in this map
starts from.

## Gate

- The NJL/mixed group is committed on `main` as its own commit, and the J0614
  group is **left alone** (it is someone else's in-flight work — check whether
  another session owns it before touching anything).
- Branch `njl-speed` exists off that commit and its SHA is recorded here.
- The tests the change can reach are run and reported with the interpreter
  named (CLAUDE.md section 12): `test/njl/`, `test/mixed/`, `test/baseline/`.
  Expect the four known 3.14-artifact baseline failures and the three
  pre-existing njl/mixed failures recorded in memory; **verify each against
  HEAD before calling it pre-existing**, do not take the list on trust.
- No `eos/*.py` left uncommitted on `njl-speed` at the end of the session.

---

## Resolution (2026-09-08)

**Baseline SHA: `d6d9e7c`** (`feat(njl): fast table mode, non-rescuing pattern
solve, njl_phase seeding`), on `main`, parent `a7ddbc7`. Branch **`njl-speed`**
is open at that SHA and is where every later ticket works.

**What landed** — eight files, one commit, the group the ticket names plus
`docs/DEFERRED.md`:

    docs/DEFERRED.md                          njl `solve_nodes` deferral entry
    eos/njl/{__init__,api,solver,table}.py    solve_nodes / build_fast_table /
                                              Branch / rescue=False
    eos/mixed/adapters.py                     njl_phase seeding + wing sweep
    notebooks/quark_timing.{py,ipynb}         the timing harness

`docs/DEFERRED.md` was not in the ticket's file list but belongs to this group:
its entire diff is the `TableSpec.solve_nodes` entry explaining why the fast
table is `beta_eq_neutrinoless`-only. Committed with `git commit --only <paths>`,
never a bare `git commit`, because the checkout is shared.

**What was left alone.** The J0614 observational-constraint group — `eos/general/
constraints/{__init__,build}.py`, its three data files, `plot/*`, and untracked
`notebooks/hybrid_stars.{py,ipynb}`. It is another session's live work:
`notebooks/hybrid_stars.py` was written 2026-09-08 14:41, and several `claude`
processes were running in this checkout at the time.

**The last gate line is NOT met, and cannot be.** "No `eos/*.py` left
uncommitted on `njl-speed`" collides with "leave the J0614 group alone", which
the same ticket requires: that group holds two `eos/*.py` files
(`eos/general/constraints/{__init__,build}.py`). They are not mine to commit.
Nothing on the NJL path imports `eos.general.constraints`, so this does not
touch any measurement this map takes; it does mean a CLAUDE.md section 12
LANDING MEASUREMENT (full suite) is blocked until that session lands its own
work. No ticket in this map needs one.

**Tests — Python 3.14.2** (`/Library/Frameworks/Python.framework/Versions/3.14`),
numpy 2.3.5, scipy 1.17.0. Run against the working tree that became `d6d9e7c`:

| suite | result |
|---|---|
| `test/njl` | **130 passed**, 0 failed, 120.56 s |
| `test/mixed` | **272 passed**, 0 failed, 1515.29 s |
| `test/baseline` | **20 passed**, 0 failed, 75.99 s |

Reachable set per CLAUDE.md section 12: the change touches `eos/njl/`'s solver
and table and `eos/mixed/adapters.py`, so it reaches `test/njl/` and
`test/mixed/`; `test/baseline/` is reachable because `njl.npz` and `mixed.npz`
are frozen against exactly these code paths. It reaches nothing else — no other
model imports njl or the mixed adapters.

**No HEAD control arm was run, and none was needed.** The gate anticipated four
3.14-artifact baseline failures and three pre-existing njl/mixed failures. There
were **zero failures**, so there was nothing to attribute and the archive-pair
comparison was skipped (the HEAD copy was staged before the numbers came in, and
discarded). The memory entry those expected failures come from names the
**anaconda 3.9** stack; the baselines are 3.14 artifacts, which is exactly why
they are green here. That is a confirmation of the recorded trap, not a
contradiction of it.

**Fact later tickets depend on: the suites must be invoked one directory at a
time.** `python3.14 -m pytest test/njl test/mixed` aborts at collection with
`import file mismatch` — `test/njl/test_jacobian.py` and
`test/mixed/test_jacobian.py` share a basename and neither directory has an
`__init__.py`. Pre-existing, unrelated to this work, not fixed here (the fix is
someone's `__init__.py` or `--import-mode=importlib` decision, and it is not on
this map's route). Loop the directories instead:

    for d in njl mixed baseline; do python3.14 -m pytest test/$d -q; done

**No fog graduated.** The ticket decided nothing about physics or route; it only
established the baseline. Tickets 02, 03 and 04 are unblocked.
