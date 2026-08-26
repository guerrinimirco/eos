# Execute Phase 6 — the port: make `nucleation` import `eos` again

Type: task
Status: resolved
Assignee: session cf726299
Blocked by: 23
Parent: ../map.md

## Question

The first half of [ticket 23](23-phase6-respec.md)'s corrected brief. Read that
brief before starting: it carries the measured mapping target by target, and
three of the premises the old Phase 6 text rests on are false.

On `nucleation` branch `paper-release`, at
`/Users/mircoguerrini/Desktop/Research/Python_codes/nucleation`. **Confirm the
path with the user before writing anything outside `eos/`** — the prompt's
Stage 7 requires it and the map does not lift that particular gate.

### Before anything: the tree, then the before-image

**DONE, and one of the two instructions was wrong.** Recorded here because the
correction matters more than the action.

Ticket 23's brief said to commit the 16 modified paper PDFs, on the grounds that
"a figure regeneration is a real artifact". **There was no regeneration.**
Measured byte by byte against HEAD, all sixteen differ by **3 to 6 bytes each,
every one inside a `/CreationDate` stamp**, same matplotlib 3.10.9, zero content
change. Committing them would have put sixteen binary blobs into a paper
repository's history to record a timestamp. They were **discarded**, not
committed.

`docs/nucleation_physics.md` (28.9 kB) and `docs/reproducing.md` (8.5 kB) were
**restored** — the four live references stand, including
`nucleation/tables/quantum.py:7` citing "section 7" of the first.

`nucleation`'s tree is now **clean**, and its HEAD has moved to `cad424b`
(ticket 62's session pinned the stack there too: `requires-python >= 3.11`,
numpy >= 2.0, scipy >= 1.17). **Take the before-image against `cad424b`, not
`33c1e61`** — run the suite against `eos` HEAD and record it verbatim as
`nucleation_before_py314.txt`. It will be mostly collection errors:
`nucleation` cannot import `eos` and has not been able to since Phase 3. That
wreck is the instrument this ticket is measured with.

**TAKEN — and it does not enumerate anything, which changes how the port runs.**
`output/_audit/nucleation_before_py314.txt`, python.org 3.14.2 / numpy 2.3.5 /
scipy 1.17.0, nucleation at `cad424b` against eos at `d509edb`:

    bare pytest        cannot load conftest; 0 tests collected
    module walk        0 of 27 nucleation modules import

**All 27 fail on the same first error** — `No module named 'eos.alphabag.eos'` —
because everything routes through `nucleation/__init__.py`, which imports
`composition`, which is the first broken import. The single root cause masks
every other break behind it.

So the gate cannot be "this list goes to zero": there is no list yet. **The port
is iterative** — fix, re-measure, repeat, and the real breakage is only visible
as each mask is removed. Re-run the module walk after every site and keep the
successive counts; that sequence, not a single diff, is this ticket's evidence.

### The port

Seven import targets, every one with an in-place successor; the mapping table is
in the brief. Six are mechanical. One is not:

**The total-thermo assembly.** `compute_alphabag_total_thermo_from_mu` and its
CFL twin have no successor. `nucleation` **keeps its own saddlepoint solver** and
assembles the total itself from five already-public pieces —
`alphabag.thermo_from_mu` / `cfl_thermo_from_mu`, `alphabag.gluon_thermo`, and
`general.thermodynamics_leptons.{electron_thermo, photon_thermo,
neutrino_thermo}`. The three `include_*` booleans become three `if`s. **No new
`eos` code.** Do not route this through `alphabag.eos_point`: that would hand
`eos` the 4-vector solve that is the reason `nucleation` is a package.

**Measured, so the port need not rediscover it.** `thermo_from_mu` returns
`MatterThermo(n_u, n_d, n_s, n_B, n_C, n_S, T, mu_u, mu_d, mu_s, P, e, s, f,
Y_C, Y_S, mu_B, mu_C, mu_S)`. Across `composition.py`, `critical.py`,
`barrier.py` and `tables/`, what callers actually read off the OLD total result
is exactly:

    n_B  n_C  Y_C  Y_S  mu_B  mu_C  mu_S     <- MatterThermo already carries these
    P_total  Y_e  Y_nu  mu_e  mu_nu          <- the five the assembly must add

So the assembly is a small `nucleation` dataclass wrapping `MatterThermo` plus
those five. Nothing else is consumed, and `P_total` is the only one with physics
in it — get the terms and signs right there and the rest is bookkeeping.

**The parameter helper.** `get_alphabag_custom(alpha=, B4=, m_s=)` has no
successor, but `Parameters` carries exactly those fields. One small helper in
`nucleation` wrapping `dataclasses.replace(Parameters.default(), ...)` — not
seven inline copies, not a new `eos` constructor.

### Out of scope, deliberately

- **The paper notebook.** Its port is mechanical (4 import lines, 2 name swaps)
  but its gate is execution, and its own header says the runs take hours.
  [Ticket 72](72-phase6-conformance.md).
- **The test move.** This ticket's gate is a node-id-to-node-id comparison;
  moving the files underneath it destroys the only instrument it has. Ticket 72.
- Anything in ticket 72's list.

### Gate

- Every failure caused by an import or a call site goes to **zero**.
- **Every failure that survives is REPORTED, not fixed.** A survivor belongs to
  ticket 72 or to a ticket of its own. Do not let this ticket be conscripted
  into fixing pre-existing breakage it did not cause.
- Report `pytest` for `nucleation` verbatim, with the interpreter and the
  collected count, against the before-image.
- Re-run the §1 dependency check **in both directions** and report both. The
  reverse direction is already gated by
  `eos/test/test_imports.py:41 test_eos_never_imports_nucleation` and passes
  today; the forward direction is what this ticket restores.

### After the gate

**Push to `origin/paper-release`.** The no-push rule is lifted: the premise that
produced it ("nucleation has no git remote") is false —
`github.com/guerrinimirco/metastability-nucleation` exists. A public repo that
imports is worth more than a polished one that does not, and the tests are
tracked, so this push is what makes them publicly runnable.


## Answer — the port is done, and `nucleation` imports `eos` again

Landed on `paper-release` as `32ef8c4` and **pushed to
`origin/paper-release`** (`33c1e61..32ef8c4`). 17 files, +281/-63, one of them
new. **No file under `eos/` was edited** — the brief's "no new `eos` code" held
end to end.

### The successive counts, which are this ticket's evidence

The before-image enumerated nothing: one root cause masked every other break.
So the port ran iteratively, re-measuring after each site. Instrument: a module
walk that imports every `nucleation` module **in its own fresh interpreter**,
so a module pulled in as a side effect of an earlier failed import is not
miscounted as importing on its own. That reproduces the before-image exactly.

    run 0  before        module walk  38 of 38 FAIL   pytest  0 collected
    run 1  imports fixed module walk   0 of 39 fail   pytest  21 failed, 15 errors, 36 passed
    run 2  Y_Le / mu_nue                              pytest   7 failed, 12 errors, 53 passed
    run 3  Y_u, Y_d, Y_s                              pytest   2 failed,  0 errors, 70 passed

Runs 2 and 3 are the point of the iteration: **neither break was visible until
the one above it was cleared.**

### The seven targets

Six mechanical, exactly as the brief measured. `EOSTable_for_TOV` was taken
from `eos.general.state` rather than through `eos.astro.tov.solver`, which
re-exports it — the brief flagged it as the one target that changed LAYER, and
importing it from the layer it actually lives in says so.

The seventh, the total-thermo assembly, became **`nucleation/quark.py`**: a
`DropletThermo` dataclass and two builders, assembled from
`alphabag.thermo_from_mu` / `cfl_thermo_from_mu`, `alphabag.gluon_thermo` and
`general.thermodynamics_leptons.{electron_thermo, photon_thermo,
neutrino_thermo}`, with the three `include_*` booleans as three `if`s. The two
old builders differed only in their phase block, so there is **one** assembly
function and two thin wrappers keeping the call shape `composition.py` already
had. `alphabag.eos_point` was not used, as ruled. The same module carries
`custom_params(alpha, B4, m_s)` for the seven `get_alphabag_custom` sites.

The brief measured the assembly's added fields as five (`P_total`, `Y_e`,
`Y_nu`, `mu_e`, `mu_nu`). **It is nine.** `e_total` is read at four sites
(`rates.py:100`, `tables/thermal.py:358,361,364`), and `Y_u`, `Y_d`, `Y_s` are
read through `tables/grid.py:_BASE_DATA_KEYS` — a list of key STRINGS fed to
`getattr`, which is why an attribute-access grep could not see them. `s_total`
and `f_total` are carried too: nothing reads them, but a total-thermo block
that reports P and eps and not s is not a thermodynamic state, and they are two
lines. The count is the lesson, not the fields: a `getattr(obj, key)` over a
name list is invisible to the measurement the brief used.

### The two masked breaks

`eos.sfho.table` renamed the trapped-neutrino table's outer axis `Y_L -> Y_Le`
and its potential `mu_nu -> mu_nue` (`COLUMN_MAPS`/`GRID_AXES`, §3's condition
names). Seven sites read those keys off an eos-produced table. **Only the reads
of eos structures were renamed**; `nucleation`'s own `Y_L`/`mu_nu` keys, which
are internally consistent, were left alone — renaming them would be conformance
work, not a call-site fix.

### The gate

**Every import- and call-site-caused failure is at zero.** 38 of 38 modules
failing to import -> 0 of 39. 0 tests collectable -> 72 collected, 70 passing.

**Two survive, and both are REPORTED, not fixed. No tolerance was touched.**

    python.org 3.14.2 / numpy 2.3.5 / scipy 1.17.0
    PYTHONPATH=<eos>  MPLBACKEND=Agg
    pytest nucleation/tests -q      2 failed, 70 passed in 1.87s   (72 collected)

    FAILED nucleation/tests/test_composition.py::test_regression_solver_cases
    FAILED nucleation/tests/test_critical.py::test_energy_barrier_matches_golden

Both were verified NOT to be the port's, by an A/B that holds the ported code
fixed and swaps only the alphaBag kernel — the pre-refactor
`thermodynamics_quarks`, archived from `e44578a^`, against today's
`thermodynamics`:

    old kernel, ported nucleation   2 passed
    new kernel, ported nucleation   2 failed

The cause is a **~1 ulp change in the quark block's floating-point
association** — §2's shared `eos.general.basis.quark_charges` replacing the
model's inline charge sums, plus a different route to the massive flavour's
density:

    n_s   old 0.8170095995673604    new 0.8170095995673602
    n_B   old 0.8108295267682285    new 0.8108295267682284

Every physically nonzero quantity still matches its golden. What fails is
round-off:

- `test_regression_solver_cases` — ten keys exceed `rel=1e-9`, **all ten
  quantities the CFL flavour lock forces to zero**: `mu_e ~ 3e-08`,
  `mu_C ~ 3e-10`, `Y_C ~ 4.5e-12`. The test's guard is `if abs(v) > 1e-12`,
  which admits a value that is zero by construction to a relative comparison.
- `test_energy_barrier_matches_golden` — `max|dW| = 3.027e-09` against an
  **absolute** bound of `1e-9`, on a `W(R)` curve reaching `-1.4875e+06` MeV.
  Relative deviation `2.0e-15`, about nine ulp.

Both are test-premise defects — a near-zero quantity compared relatively, a
large one compared absolutely — and fixing either means re-deciding what the
golden asserts. **That is a ticket of its own, not this one and not 72's list**
(72 is notebook, test move, README, dead code); see the map's Not-yet-specified.

### §1, both directions

    forward   import nucleation -> OK; eos subpackages pulled in:
              ['eos.alphabag', 'eos.general']
    reverse   eos/test/test_imports.py::test_eos_never_imports_nucleation
              1 passed        (whole file: 194 passed)

Full transcripts: `output/_audit/nucleation_after_ticket24_py314.txt`.

### Reported, not fixed — Stage 7 material

**`eos` is not installed on the canonical stack.** `nucleation/pyproject.toml`
declares `eos` a dependency, but `pip list` on python.org 3.14 shows only
`nucleation` (editable); `import eos` succeeds solely via `PYTHONPATH` or cwd.
The before-image was taken the same way, so every number here is comparable —
but "nucleation depends on eos" is true in the source and not yet true in the
environment. It is an install, not a code change, and it is not this ticket's.

**`pyflakes` over the package reports 20 unused imports**, all pre-existing
(`tables/__init__.py` re-exporting private names, `types.SimpleNamespace` in
`qstar.py`, `joblib` in `replay.py`). `test_no_undefined_names` passes: none is
an undefined name. Dead-code removal is ticket 72's item 4.

### Out of scope, untouched as instructed

`notebooks/2fam_PNS_nucleation.py` still imports the old paths (4 import lines,
2 name swaps) and `nucleation/tests/` has not moved. Both ticket 72.

Status: resolved.
