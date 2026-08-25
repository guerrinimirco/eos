# dd2 cannot take two of §4's six species flags

Type: task
Status: resolved
Assignee: session e5e1c4c9 (also the other of 60/61)
Blocked by: 44
Parent: ../map.md

## Question

Found by [ticket 04](04-notebook-skeleton.md) while building the shared knobs
cell, and covered by **no row** of ticket 08's audit or
[ticket 11](11-conformance-triage.md)'s triage — §3-ii covers
`thermal_neutrinos` + trapped *behaviour*, not the missing *names*.

`SpeciesFlags(**{six §4 flags})` raises **`TypeError` on dd2** and succeeds on
the other nine models. dd2 is missing `thermal_mesons` and `thermal_neutrinos`:

    dd2  extra = neutrinos, phi_field, sigma_star,
                 include_pseudoscalars, include_thermal_vectors

Two different problems wearing one finding:

1. **`thermal_mesons` is present but split** into `include_pseudoscalars` and
   `include_thermal_vectors`. §4 names one flag — "pi, K (and optionally the
   vector nonet)" — so the split is arguably a refinement, but §4's name does
   not exist and §13's "the same job carries the same name in every model" is
   the rule that binds. Decide: one `thermal_mesons` flag with the vector nonet
   behind a second, clearly-secondary name, or `thermal_mesons` as the alias the
   other nine models spell.
2. **`thermal_neutrinos` is genuinely absent.** dd2's `neutrinos` is NOT a
   misspelling of it — the field's own comment reads "only trapped / fixed-Y_Le
   modes", i.e. the matter-composition neutrinos. So the tau gas is unwired.
   §4 is explicit that setting a flag a model does not implement **raises**, and
   that a NotImplementedError is never a silent no-op — so a raise is an
   acceptable resolution here and wiring the sector is not required. What is not
   acceptable is the flag not existing, because then the caller gets a
   `TypeError` that reads as their own bug.

Also worth measuring while in the file: the **defaults** diverge across models —
`photons` True in six and False in four, `muons` True in four and False in six,
`alphabag` shipping `thermal_neutrinos=True`. §4's "no sector is enabled or
disabled implicitly" is only honoured if a caller passes all six every time.
Whether the defaults should be unified is a separate ruling; report, do not
change, unless the answer is obvious.

**Tickets [12](12-hadronic-skeleton.md), [15](15-quark-notebook.md),
[18](18-enjl-notebook.md) and [58](58-hybrid-skeleton.md) are blocked by this.**
The alternative — a per-model translation table in the knobs cell — would be
copied into four notebooks.

Blocked by [44](44-rename-dd2.md) so it does not collide with the dd2 renames in
the same files, and run tickets 42/43's AST check here: both proved that a
rename onto a name this repo already uses for a local adapter fails silently.

Resolved when `SpeciesFlags(**six)` constructs for all ten models, or raises
with a §4-compliant message, and `test/baseline/` is unmoved.

**Reporting.** Name the interpreter **in the filename** of whatever lands in
`output/_audit/` — `_py39` or `_py314` — not only in the prose. Per
[ticket 47](47-dd2-nmp-inversion.md) the stack is the whole difference between
0 and 14 failures, and the directory already holds files from both with nothing
in the name to tell them apart, so a listing alone can hand the next session the
wrong before-image. Report the **collected count** alongside the failure count
too: ticket 20's +2 tests in `test/test_imports.py` moved it to 1665.

**Neighbour, not duplicate:** [ticket 60](60-dd2-hyperon-flag-raise.md) is a bare
`KeyError` when `hyperons=True` meets a nucleonic parameter set — the same file
family, a different defect (a refusal with the wrong exception, against a flag
that does not exist at all). Whoever runs second should read the other first.

**Two documentation sites change when this lands**, both written by
[ticket 20](20-phase5-api-readme.md) and both stating the gap as it is TODAY,
so neither is wrong now and both go stale the moment dd2's flags move:

- `README.md`, the "Species flags" section — the paragraph beginning "One
  model does not yet carry two of the six", which names
  `include_pseudoscalars`, `include_thermal_vectors`, and `neutrinos` as the
  matter-composition field rather than the tau gas.
- `eos/__init__.py`, the `#:` comment above `SPECIES_FLAGS` (commit
  `7c5b7a9`), which says the same thing in the same words and ends "Passing
  the six shared names to dd2 raises TypeError, which reads as the caller's
  bug."

The second is deliberately a commit of its own so it can be reverted or
rewritten without touching the lazy-import work in `fe68f20`. If this ticket
resolves by giving dd2 the two names, both paragraphs are deleted rather than
edited; if it resolves by a raise that names the gap (which §4 permits for
`thermal_neutrinos`, since the tau gas is unwired rather than misnamed), both
are rewritten to say THAT, and `README.md`'s claim that the six names are
carried by every model becomes true without qualification.

Also: `test/test_imports.py::test_the_top_level_carries_the_mode_and_species_vocabulary`
checks §4's six names against seven models' dataclass fields and **exempts
`dd2` by name**, with this ticket cited in its docstring. Drop the exemption
when this lands — that is the check that keeps the answer from drifting back.
(`test/` is gitignored, so it is in a working copy only; see the map's
untracked-guards bullet.)


## Verified, and the exemption is not dd2-specific

Checked rather than taken on relay. `test/test_imports.py:252` reads

    for model in ("sfho", "zl", "did", "vmit", "alphabag", "njl", "ccdm"):

— an **inclusion list of seven**, not an exemption of one. Three models are
absent: `dd2` (which genuinely fails), and **`enjl` and `abpr`, both of which
PASS the six-flag check today** and are being skipped for nothing. Measured with
`dataclasses.fields`: neither is missing any of §4's six.

So dropping dd2's exemption when this closes is necessary and not sufficient.
**The list must be inverted**: iterate every model and exempt by explicitly named
exception, because as written a model added tomorrow is exempt by default and
the check meant to prevent drift is what permits it. That is the same defect the
docstring's `dd2` sentence describes, one level up.

Three prose sites carry the claim and all three change when this lands
(eos-88, `bcb9d56`): `README.md`'s species-flags paragraph, `eos/__init__.py`'s
`SPECIES_FLAGS` `#:` comment (`7c5b7a9`), and this test's docstring. The edit
differs by resolution — giving dd2 the two names DELETES those paragraphs, a
raise that names the gap REWRITES them.

**The exemption is a TWO-WAY gate, and this paragraph is the durable copy of
that fact.** The check exempting `dd2` does not merely skip it: it asserts
that dd2 IS still missing the names, so the moment this ticket gives dd2 the
two flags the test goes red rather than green, and its message names what has
to move with it. Closing this ticket therefore requires **three edits in one
change** — delete the `dd2` entry from the exemption dict, delete `README.md`'s
"One model does not yet carry two of the six" paragraph, and delete the
corresponding `#:` block above `SPECIES_FLAGS` in `eos/__init__.py`. Do them
together or the suite stays red, which is the point.

Recorded here rather than left to the failing assertion because **`test/` is
gitignored**: a fresh clone, a `git worktree`, or another machine has no such
test, so the assertion cannot reach whoever works this ticket there. The
ticket body is the only artifact that survives the checkout. (Where `test/`
does exist, the enforcement is real and will catch a partial close.)

## The check is two-way, and that must be read HERE

`test/test_imports.py`'s rebuilt check fires in **both** directions: it fails if
a model loses a §4 flag, and it fails if **dd2 GAINS** the two names — naming the
three prose sites that must move with it (`README.md`, `eos/__init__.py`'s
`SPECIES_FLAGS` comment, the test's own docstring). Verified to fire by standing
`sfho` in for a fixed dd2, not assumed. 190 passed on anaconda 3.9.7.

Written here rather than left for whoever closes this ticket to meet as a red
test, because **`test/` is gitignored (§11)**: a fresh clone has no such test, so
the failing-test channel does not survive and this ticket body is the only
durable carrier. So: closing this ticket makes that test go red ON PURPOSE, and
the fix is to update the three sites, never to relax the check.


## Resolution

**Both halves fixed, by different means, because they are different defects.**
`SpeciesFlags(**six)` now constructs on all ten models, and dd2 is no longer
the model that answers §4's vocabulary with a `TypeError`.

### 1. `thermal_mesons` — the split is kept, the NAMES are §4's

Of the ticket's two options, "one `thermal_mesons` flag with the vector nonet
behind a second, clearly-secondary name" is what landed, because §4 writes the
sector as *"pi, K (and optionally the vector nonet)"* — the parenthesis says the
nonet is an option on top of the gas, not half of it. So:

    include_pseudoscalars   ->  thermal_mesons
    include_thermal_vectors ->  thermal_vectors

Pure rename, no alias and no second source of truth: an alias would have left
two spellings of one boolean and `dataclasses.replace` reaching for whichever
the caller happened to use. `thermal_vectors` keeps its independent meaning —
vectors-without-pseudoscalars is still expressible, as it was — and the comment
at the field quotes §4's phrase so the secondary status is visible at the
declaration.

**What did NOT get renamed, deliberately.** `eos/general/thermal_mesons.py` and
dd2's thin wrappers around it take `include_pseudoscalars=` /
`include_thermal_vectors=` as *function keyword arguments*, and `did` and `sfho`
call them under those names too. Those are the shared gas's signature, not
dd2's flag vocabulary, and renaming them would have been a `general/` change
this ticket was not asked for. `MatterCtx`'s two fields keep the general/ names
for the same reason — they exist to be forwarded to that signature. The four
sites that read the FLAG are the ones that moved: `dd2/solver.py:767-771`,
`dd2/thermodynamics.py:357-358`, and `mixed/adapters.py:222-223, 581-582` (both
`dataclasses.replace` calls, which fail loudly on an unknown field name — so a
missed one could not have been silent).

### 2. `thermal_neutrinos` — the name is added, the sector RAISES

Added as a real field that raises `NotImplementedError` when set, which §4
explicitly permits for an unwired sector and which the ticket blessed. The
message says what the sector is and, in the same breath, that it is **not**
dd2's `neutrinos`:

```
SpeciesFlags: thermal_neutrinos -- the neutrino flavours a mode does not track,
carried as mu = 0 gases -- is not wired in dd2. It is NOT `neutrinos`, which is
the matter-composition electron neutrino of the trapped modes
```

`neutrinos` is untouched and stays: it is dd2's own physics, a different sector
with its own flag, and the two now sit adjacent in the dataclass with comments
pointing at each other. Wiring the tau gas was considered and rejected as
outside this ticket — it is new physics with its own numbers, and §4's
complaint was the missing NAME, not the missing sector.

Note the shape this leaves: `SpeciesFlags(**six)` with everything False
constructs everywhere, and with everything True dd2 raises
`NotImplementedError` exactly as **eight of the other nine** do. Measured:

| | all six False | all six True |
|---|---|---|
| dd2, sfho, zl, vmit, alphabag, abpr, njl, ccdm | constructs | NotImplementedError |
| did | constructs | constructs |
| enjl | NotImplementedError (it fixes every flag) | NotImplementedError |

No `TypeError` anywhere. That is the ticket's resolution criterion.

### 3. The inclusion list was already inverted; dropping dd2 was the rest of it

`test/test_imports.py`'s check had been rebuilt into
`for model in eos.MODELS` minus a named `exempt` dict, so `enjl` and `abpr` are
no longer skipped for nothing and a model added tomorrow is covered by default.
What this ticket owed was the last entry: **`exempt` is now `{}`**. The dict is
kept rather than deleted, because an empty dict is the statement "no model is
exempt" and is where a future exemption would have to be argued for in writing.

The two-way half fired exactly as this ticket's body predicted it would: the
`assert missing` branch turns red the moment dd2 gains the names, and the fix
was the three prose sites, not the check. All three moved in the same change:

- **`README.md`** — "One model does not yet carry two of the six" is gone,
  replaced by a paragraph saying all ten carry all six and that carrying a name
  is not wiring the sector.
- **`eos/__init__.py`** — the `#:` block above `SPECIES_FLAGS`, same rewrite.
- **`test/test_imports.py`** — the docstring now records that the exemption set
  is empty and *why* the check is about the name rather than the sector.

`test/test_imports.py`: **190 passed** on anaconda 3.9.7.

### 4. A fourth prose site the ticket did not name

`docs/DEFERRED.md`'s dd2 section carried the same claim as a deferral
("Species-flag naming: the spec calls the meson switch `thermal_mesons`; dd2's
`SpeciesFlags` carries the finer pair..."). §11 calls that file the tracked
ledger of per-model gaps, so leaving a closed gap open in it is the failure mode
`c195392` already named — nothing notices when a stated limitation stops being
true. Rewritten rather than deleted: the naming half is closed, the
**`thermal_neutrinos` sector remains unwired** and that is now what the entry
says.

### 5. The AST check ran, clean both directions

Tickets 42/43's cross-module shadowing check, run before and after against
`{thermal_mesons, thermal_vectors, thermal_neutrinos}` over `eos/**` and
`test/**`: **0 hits**. A companion grep for module-level rebinding of those
three names found only continuation lines of keyword arguments in other models,
none in `dd2` or `mixed`. The renamed things here are dataclass FIELDS, which
cannot shadow an import the way 42's `solve` did, so the check was cheap
insurance rather than the live risk it was there.

Two collision risks that the AST check does not cover were checked by hand and
are clear: nothing constructs `SpeciesFlags` positionally anywhere in `eos/` or
`test/` (so reordering the fields is safe), and no `test/baseline/*.npz` stores
a key named after either old flag (checked all 13 — `eos/general/table_io.py`
`asdict`s dataclasses into table metadata, which is the path by which a field
rename could have moved a stored key).

### 6. Defaults: measured, reported, NOT changed — and graduated to a ticket

As instructed. The divergence is real and is three-way, not the two the ticket
recalled: `muons` True in five models and False in five, `photons` True in six
and False in four, `thermal_neutrinos` True in `alphabag` alone. The answer is
not obvious — unifying on all-False is the cleanest reading of §4 and would
strip the photon gas from every T > 0 call that relies on the default, moving
`test/baseline/` numbers §12 calls ground truth. So it is
[ticket 62](62-species-flag-defaults.md), with the full measured table, and it
does **not** block the notebooks: the knobs cell passes all six explicitly.

### 7. The gate

**Interpreter: python.org 3.14.2 / numpy 2.3.5 / scipy 1.17.0** — the stack
[ticket 57](57-canonical-stack.md) ruled canonical. (57 is RESOLVED; the map's
"still unruled" note is stale.) Also run on anaconda 3.9.7 where noted.

**`test/baseline/` is unmoved — 6 failed, 10 passed, 16 collected, and the six
node ids are byte-for-byte the six in `output/_audit/pytest_after_ticket48_py314.txt`:**

    ccdm, dd2, enjl, njl, tov, zlvmit          0 added, 0 cleared

`diff` of the sorted `FAILED` node-id lists is EMPTY. Recorded as
`output/_audit/pytest_after_ticket61_baseline_py314.txt`.

**`test/test_imports.py`: 190 passed** on anaconda 3.9.7, with `exempt = {}`.

**`test/dd2`: 3 failed, 203 passed, 206 collected** — and the three node ids
are `test_api.py::test_inversion_without_Q_sat_predicts_it`,
`test_api.py::test_inversion_with_Q_sat_still_imposes_it` and
`test_dd2_m8.py::test_restarts_recover_a_seed_limited_inversion`, the three
[ticket 47](47-dd2-nmp-inversion.md) NMP-inversion failures the before-image
lists, none of them on the flag or hyperon path. `diff` of the sorted node-id
lists EMPTY: **0 added, 0 cleared**. Recorded as
`output/_audit/pytest_after_ticket61_dd2_py314.txt`.

**Both verify suites PASS** with no number moved: `eos/dd2` (golden SNM point
1.40e-05, HVH 1.50e-15, backend parity 8.88e-16, CompOSE HS(DD2) 2.83e-05) and
`eos/mixed` (Euler/HVH 9.10e-15, Gibbs/Maxwell, backend parity 3.77e-15,
**TOV M_max = 2.340, R = 12.64 km**, unchanged).

**A matched full-suite pair was run and is the reason the above is trustworthy.**
The tree was being edited by a concurrent session throughout (eight `api.py`
files plus a new `test/test_nonconvergence_return.py`, ticket 49), so the gate
was run in two isolated copies built with `git archive HEAD` from ONE snapshot
of the gitignored `test/`, differing only by this ticket's seven files. Both
collected **1677** (not 1665 — ticket 49's new file adds 12) and produced
**identical failure sets at every index through 677 tests**. That control was
necessary: an isolated copy shows six `test/abpr` round-trip failures that the
real repo does not, and without the HEAD copy beside it they would have read as
this ticket's regressions. Neither full run was carried to completion — in a
cold copy the numba kernels recompile and `test/mixed` alone runs for hours
under three-way CPU contention — so the completed gates above are the targeted
ones, and that is stated rather than papered over.

**Not run to completion, and named as such:** `test/mixed` (277 tests). Its
cover is `eos/mixed/verify/run_full_check.py` PASSING, which exercises both
renamed `dataclasses.replace` call sites end to end through nine invariants
including the TOV mass — and `replace` raises on an unknown field name, so a
missed rename there could not have been silent.
