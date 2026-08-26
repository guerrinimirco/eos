# Remove `eos/mixed/scan.py`

Type: task
Status: resolved.
Assignee: session 3671f414
Blocked by: 84
Parent: ../map.md

## Question

The user's ruling in [ticket 84](84-vmit-params-in-the-plumbing.md): *"scan is
just a code that help us find parametrizations useful. We can remove it,
rethink it. In future I will have a bayesian code which do it better so probably
we can just remove it."*

626 lines whose own first line reads "Where in parameter space does a DD2 + vMIT
hybrid equation of state exist?" — a declared DD2+vMIT study inside a composite
engine that is meant to be general, and the reason `mixed` imports `dd2.nmp` and
`dd2.solver` at module level.

§6 lists Bayesian inference as use case 3, so a scan is aligned with the
library's purpose. This deletes THIS scan, not the ambition.

### One function must survive, and it relocates

**`build_parametrization(nmp, flags, ...)` moves to `eos/dd2/nmp.py`.** Not a
concession to the deletion — a correction owed anyway: §5 says an NMP-inverting
constructor "is therefore a free function in `nmp.py`, not a classmethod on the
parameter dataclass". It has been living in the composite engine, one layer
further from home than the anti-pattern §5 names.

**Four sites depend on it**, in `test/tov/test_solver_fast_robustness.py:87,94,
188,196` — among the tov tests [ticket 74](74-py314-non-baseline-failures.md)
JUST repaired. Do not break them.

### The rest of the tail, measured

    eos/mixed/__init__.py:124-126,164-165   six re-exported names
    test/mixed/test_scan.py                 271 lines, goes with it
    test/tov/test_rotating.py:373           a `dd2_scan` fixture
    CLAUDE.md:36-37                         names `mixed/scan.py` in the §1
                                            astro carve-out  -> ticket 85
    eos/vmit/parameters.py:66               cites `eos.mixed.scan`
    eos/vmit/vmit.tex:160, vmit.md:102      cite it in the DOCUMENTS (§11)
    eos/mixed/charges.py:109                a comment

The three `vmit` citations are §11 documents describing what the code does. They
do not simply lose a sentence: each says `eos.mixed.scan` is what moves over
(B4, a, m_s), so each needs a replacement statement of how a vMIT parameter
sweep is actually done now.

### The §8 gate is NOT lost

`scan.py`'s `eos_is_physical` is a SECOND implementation used only by the scan.
`eos/mixed/verify/run_full_check.py:231 _check_causality` implements §8's
delivery gate independently and stays.

### Gate

`grep -rn "scan_parameters\|scan_point\|scan_hadronic\|eos.mixed.scan" eos test
docs` returns nothing outside this ticket's own deletions; `test/tov` still
green including the four `build_parametrization` sites; `test/baseline/`
unmoved. CLAUDE.md §1's mention goes via
[ticket 85](85-claudemd-sentences-owed.md), not here.

---

## Resolution

`eos/mixed/scan.py` (626 lines) and `test/mixed/test_scan.py` (271 lines) are
deleted. `eos/mixed` no longer imports `eos.dd2` or `eos.vmit` from `scan.py`,
and its `__all__` lost the ten scan names.

### `build_parametrization` landed in `eos/dd2/nmp.py`

Together with `SECTOR_KEYS` and the private `_split_sample`, appended below
`from_delta_potential` — the composed constructor reads after the two sector
constructors it composes. Exported from `eos.dd2`; **not** re-exported from
`eos.mixed`, because putting a `dd2` name back on the composite engine's
surface is the privilege [ticket 84](84-vmit-params-in-the-plumbing.md) ruled
against. The four dependent sites now read
`from eos.dd2 import (..., build_parametrization)`.

Two things changed on the way, both deletions of duplication:

- **`DEFAULT_HYPERON_POTENTIALS` and `DEFAULT_U_DELTA` did not move.** They
  restated `from_hyperon_potentials`'s own `(-30, +30, -18)` and
  `from_delta_potential`'s `-50` — one file away, in the same module. Keys
  absent from a sample are now simply not passed, so the constructors apply
  their published defaults. Same numbers, one fewer thing to drift.
- **`x_wD` / `x_rD` are gone from the signature.** They were never forwarded
  to `_split_sample`, so `build_parametrization(nmp, flags, x_wD=1.2)`
  silently did nothing; the ratios ride in the sample dict, which is what
  every call site already does. A `TypeError` beats a silent no-op.

Three tests moved to `test/dd2/test_dd2_m8.py` (the NMP-inverter file), since
the function they cover survives: sectors attach, the two failure stages
report apart, and the sector potentials ride in the sample.

### The three documents got replacement statements

Each said `eos.mixed.scan` moves over (B4, a, m_s); each now says what is
true instead — the library ships no scan driver, model parameters are
arguments (§6), so a sweep is a caller-side loop building one `Parameters`
per sample and calling `eos.mixed.eos_table(..., vmit_params=...)` once on
each. Written in `eos/vmit/parameters.py:66`, `vmit.tex:160`, `vmit.md:102`.
The `.tex` sentence also named `get_vmit_custom(...)`, which no longer
exists anywhere in the repo; it now reads `Parameters(B4=..., a=..., m_s=...)`
like the `.md` already did.

### The §8 gate

Confirmed as the ticket states: `eos/mixed/verify/run_full_check.py:231`
`_check_causality` builds the hybrid table and asserts P monotone and
0 <= c_s^2 <= 1 on its own. Nothing of §8 left with `eos_is_physical`.

### Two entries of the measured tail needed nothing, and one is owed elsewhere

- **`test/tov/test_rotating.py:373` is a false positive.** Its `dd2_scan`
  fixture is `rot.rratio_scan` — an RNS axis-ratio scan. Name collision only;
  untouched.
- **`eos/mixed/charges.py:109` and `test/mixed/test_chargespec_pickles.py:3`**
  cited the scan as the reason pickling matters. The §6 requirement outlives
  its example, so both now state the requirement without naming a driver.
- **`docs/DEFERRED.md:145` names `eos/mixed/scan.py`** in the same §1 astro
  carve-out sentence as `CLAUDE.md:36-37`, which this ticket routes to
  [ticket 85](85-claudemd-sentences-owed.md). Left for 85 to sweep with its
  twin — the file also has uncommitted edits from the concurrent session.
  `docs/REFACTOR_PLAN.md:150` and the frozen logs in `output/_audit/` are
  historical records of runs that happened; they are not updated.

### Gate

python.org **3.14.2** (`python3`, no `timeout`), run in an isolated copy —
`git archive HEAD` plus a snapshot of the gitignored `test/` plus only this
ticket's seven changed files and the two deletions — so the concurrent
session's edits across `eos/general` and the models' api/table/solver are not
in it. Targeted, not the full suite:

    test/mixed + test/dd2/test_dd2_m8.py    267 collected, 267 passed  9:15
    test/tov                                 42 collected,  27 passed,
                                                            15 skipped  0:28
    test/tov/test_solver_fast_robustness.py  11 collected,  11 passed  0:25

The last was run separately and verbose to prove the four
`build_parametrization` sites actually executed rather than skipping:
`test_crusted_hybrid_star_agrees_with_the_reference[0.3]` and `[0.6]` both
PASSED. `test/baseline/` is unmoved and untouched.

The ticket's grep now returns only the three lines named above (the RNS
fixture, CLAUDE.md §1 -> ticket 85, and the two historical records).
