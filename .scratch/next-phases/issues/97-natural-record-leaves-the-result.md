# The natural-units record leaves the public result, and its accessors take the fm names

Type: task
Status: resolved (2026-08-29), §12 full-suite line OUTSTANDING
Assignee: session dc4b25ab
Blocked by: [90](90-solver-signature-and-units-sweep.md) — RESOLVED
Parent: ../map.md

## Question

Split out of [ticket 90](90-solver-signature-and-units-sweep.md) — its Work
item 5 in full, plus the part of item 3 that turned out to be the same act.
Ticket 90's gate is "**no value moves anywhere**"; this half moves and deletes
thousands of frozen keys, so it leaves rather than the gate being weakened.
That is the third time this ruling has needed the treatment: section 1 became
[89](89-dd2-honours-species-flags.md), section 4 became
[94](94-zl-solver-flags.md)/[95](95-vmit-solver-flags.md)/[96](96-alphabag-solver-flags.md),
and this is section 5.

**The §5 violation, as ticket 81 measured it.** `njl` and `ccdm` carry their
natural-units matter record on the public result as `.state`; `enjl` carries
its `EoSPoint` as `BetaPoint.point`. `n_B` on those records divides out to
exactly `hc3`, but **P, eps and s do not**, because the record is MATTER ONLY:
`njl .state.P / hc3 = 146.854334` against an outer `146.939710`. A caller who
spots the unit problem and corrects by `hc3` still gets a wrong answer,
silently, by 0.085376 MeV/fm^3. A rename cannot fix that; only removing the
record from the public surface can, which is why ticket 81 ruled for removal.

**Why item 3's remaining half belongs here.** Ticket 90 §3 renames the `_fm`
family — `n_C_fm`, `n_S_fm`, `eps_fm`, `P_fm`, `s_fm`. Every one of those is a
property ON these same records, and the natural-units field it converts holds
the bare name, so the rename is a swap: the field takes `_nat`, the accessor
takes the bare name. Measured against the frozen files:

    enjl.npz   21278 keys, 14976 nested under `.point`, 3042 of them the six
    njl.npz     6594 keys,  3255 nested under `.state`,  630 of them the six
    ccdm.npz    6068 keys,  3040 nested under `.state`,  456 of them the six

So the accessor rename alone changes what **4128 frozen keys mean** (the key
`state.P` holds a natural-units number today and an fm one after), and the
removal deletes **21271 keys** outright. Neither is a value that moved; both
are surface changes the baselines follow. Doing them together is what keeps
the result honest — the nested keys vanish, and no surviving key changes its
meaning. Done apart, the rename re-blesses 4128 keys under names that mean
something else, which is the one thing a regression baseline exists to catch.

Ticket 90 executed this on 2026-08-27, measured it, and **backed it out**;
`n_B_fm` -> `n_B` as a FUNCTION PARAMETER, and `BetaPoint.n_b_fm` -> `n_B` as
a result field, stayed there and are done.

## Work

1. `njl`/`ccdm`: `EoSPoint.state` -> `_state`, which the baseline flattener
   already skips (`generate_baseline.py:163` walks `vars()` and drops names
   beginning with `_`). `enjl`: `BetaPoint.point` -> `_point`, which also ends
   `result.point.point` at the public boundary.
2. Lift onto the outer point, in fm, what callers legitimately need: the
   colour densities `n_3`, `n_8` and the quark density `n_q` (13 sites across
   njl/ccdm). `euler_residual()` is dimensionless and stays reachable through
   the now explicitly internal `_state`, as ticket 81 ruled.
3. The one site that wants the matter-only pressure
   (`test/njl/test_equilibrium_modes.py:71`, charged against neutral) reads it
   through `_state` and says why in a comment — it is asserting a property of
   the matter block, not of the point.
4. `NJLState` / `CCDMState` / enjl's `EoSPoint`: the six natural-units fields
   take `_nat`, the six accessors take the bare names, and the dataclass
   docstring states the convention (bare = fm, `_nat` = natural, MeV
   quantities unsuffixed). This is `dd2/solver.py:159`'s existing `_nat`
   convention, applied where the ticket-90 sweep found the other half of it.
5. **The trap this sweep already sprang once.** Renaming a natural-units local
   out from under a function body silently changes units in whatever the sed
   did not reach. In `enjl/solver.py::default_guess` two seed expressions
   (`0.9 * n_B` and `0.5 * Y_Le * n_B`) were natural and became fm; the model
   still converged everywhere except `n_B = 5 fm^-3`, where it stopped at a
   scaled residual of 2.259e-09 against a 1e-10 bound —
   `test/enjl/test_enjl_beta_equilibrium.py::test_high_density_needs_a_widened_box`
   caught it and nothing else would have. Rename the FIELDS first, run, fix
   every AttributeError, and only then rename the accessors: that way every
   stale site raises instead of silently reading the other unit system.

## Gate

- **Regenerate `njl.npz`, `ccdm.npz` and `enjl.npz`, and measure first.** The
  removed key set must be **exactly** the nested `.state.*` / `.point.*` keys
  (3255, 3040 and 14976 respectively). Every surviving key BIT-identical —
  not rtol = 1e-10, bit-identical, because nothing in this ticket is arithmetic.
- No other `test/baseline/*.npz` moves at rtol = 1e-10, `mixed.npz` ASSERTED —
  `eos/mixed/adapters.py` reads the njl and ccdm records through exactly these
  accessors at `:1124` and `:1283`.
- Full suite green (§12); `njl`, `ccdm`, `enjl` and `mixed` `verify/` green.
- `notebooks/quark_eos.py` displays the internal record deliberately, as the
  model's stepwise walkthrough (~20 sites). It follows the record to `_state`
  and keeps working; its `.ipynb` pair moves with it.

## Finding this ticket owes the Stage 7 report

Deleting 21271 frozen keys is a real loss of regression coverage, and §12
calls `test/baseline/` ground truth. It is the price of the §5 fix and the
ticket takes it deliberately — but whether the internal records deserve a
baseline of their own, frozen through the internal path, is a question this
ticket raises and does not answer.

---

## Note from [ticket 91](91-leptons-default-and-drift-checks.md) (2026-08-28)

**This defect is now pinned by a test, and the pin is `strict`.** Ticket 91's
third check — the fm-based units band, parametrised over all ten models —
walks every float on `eos_point`'s result and asserts the energy and density
families sit in their fm bands. It finds exactly this ticket's three, and
exactly on the records this ticket removes:

    eos.njl.state.n_q        2.305052e+07     (fm^-3 band is |x| <= 1e3)
    eos.ccdm.state.n_q       2.305052e+07
    eos.enjl.point.n_s.s    -1.590857e+07

so the check reproduces ticket 81's finding from the outside, without knowing
which field to look at. The three are carried as
`pytest.mark.xfail(strict=True)` with this ticket's number as the reason.

**`strict=True` is the part that matters here.** A strict xfail FAILS when it
starts passing, so the day work item 1 lands — `.state` -> `_state`,
`BetaPoint.point` -> `_point` — `test/test_imports.py` goes red until the
three entries are deleted from `NATURAL_UNIT_RECORD`. The exemption cannot
outlive the defect, which is what an ordinary skip would have allowed.

Ticket 91's own gate did not include fixing this; the check was written to
detect and it does.


---

## Resolution (2026-08-29)

All five work items landed. The natural-units record is off the public result
in all three models, its accessors carry the fm names, and the fields behind
them carry `_nat`.

### What changed

1. **`njl`/`ccdm`: `EoSPoint.state` -> `_state`; `enjl`: `BetaPoint.point` ->
   `_point`.** The second also ends `result.point.point` at the boundary.
2. **`n_3`, `n_8`, `n_q` lifted onto the njl/ccdm point in fm**, so the colour
   densities a caller legitimately wants no longer require reaching into the
   record. `euler_residual()` is dimensionless and stays reachable through
   `_state`, as ticket 81 ruled.
3. **`test/njl/test_equilibrium_modes.py`** reads the matter-only pressure
   through `_state` and says why: the claim is about the matter block, and the
   alternative — a public matter-only `P` beside `P_total` — would put a second
   pressure on an fm boundary and invite the confusion the assertion guards.
4. **The six/three fields take `_nat`, the accessors take the bare names**, and
   the convention is stated where the accessors live.
5. **The two-phase order was followed and it earned itself** (below).

### The finding this ticket did not expect: `_` is not one line, it was two

Work item 1 rests on `_`-prefixing making a record internal, and cites the
baseline flattener, which does skip `_` names (`generate_baseline.py:163`).
**Ticket 91's units-band walker did not.** So the moment `.state` became
`_state`, `test_every_result_field_is_in_its_fm_based_band` still descended
into it and still failed on `eos.ccdm._state.n_q = 2.305052e+07` — the strict
xfail would have gone on being satisfied by a defect that was fixed, which is
the exact failure mode `strict=True` exists to prevent, wearing the opposite
face.

`_float_fields` now skips `_`-prefixed names, with the reason written down:
section 5 binds the PUBLIC boundary, an underscore is how this repository says
a field is not on it, and the flattener and the band check must draw that line
in the same place or one of them is wrong about what "public" means. **All ten
models now pass the band check with no exemptions** (215 passed), which is what
the strict xfail was built to force.

### The two-phase rename caught eleven sites

Item 5 said: rename the FIELDS first, run, fix every AttributeError, and only
then rename the accessors, so a stale site raises instead of silently reading
the other unit system. It found, in order: 7 in njl/ccdm
(`test_equilibrium_modes`, `test_ccdm_pairing`, `test_confinement`, four in
`test_thermodynamic_consistency`), then `eos/enjl/solver.py:651`, then
`eos/mixed/adapters.py:476` and `test_enjl_construction.py:144`, then
`test/enjl/test_enjl_fixed_composition.py:170`. **Every one raised.** The last
is the sharpest: it is the ENJL author-table comparison, a section 12 golden
reference, and it read `pt.eps / hc3`. Under a one-phase rename that line would
have kept running and compared an already-fm number to the author's column,
off by hc^3 — a golden test failing loudly is fine, but the one-phase order
could equally have produced the mirror case of a silently-passing wrong bound.

### Where this deviates from the Work list, deliberately

Item 2 expected the record's colour-density readers (13 sites) to move onto the
lifted fm fields. **Most did not, and moving them would have been wrong**: the
verify suites and pairing tests scale those densities by unit-dependent
quantities. `eos/njl/verify/run_full_check.py:402` divides by
`max(abs(n_q), 1.0)` — a floor that never binds at 1e7 MeV^3 and always binds
at ~1.5 fm^-3, so the same expression means a different thing in the two
systems; `test/ccdm/test_ccdm_pairing.py:70` divides by
`(mu_B/3)**3/pi**2`, a MeV^3 scale; `test/njl/test_pairing_patterns.py:119`
bounds n_3 at an absolute 1e-6 in MeV^3. Converting those silently rescales
what they assert, which is the defect this ticket exists to remove, not a step
toward removing it. **They read `_state` and assert on the matter block on
purpose.** The lifted fm fields serve the public surface — the notebook, the
adapters, any caller — which is what item 2's "callers legitimately need" was
about.

### CLAUDE.md §5, the sentence ticket 85 parked here

Landed, and wider than the one sentence owed. The units line now names **s and
n_s in fm^-3** — the two the shorter list omitted, and precisely where natural
units survived longest — and states the convention this ticket establishes: a
model keeping its own natural-units working record holds it under a **leading
underscore**, which is the same line the baseline flattener and the units-band
check both draw.

### Documents (§11)

`njl.md`, `njl.tex` and `ccdm.md` described a public `state` field and did not
list the colour densities. Their returned-field tables now name `_state` as
internal and in natural units, and carry `n_3`, `n_8`, `n_q` in fm.
`eos/enjl/thermodynamics.py`'s two prose references to the deleted `_fm`
property family were fixed by a concurrent session in `4f6f453` — a real miss
on my side: the rename sweep matched `_fm` as an ATTRIBUTE (`\.X_fm\b`) and
prose mentions carry no dot.

### Gate

- **Baselines regenerated and audited against a pre-image** taken before
  regeneration (01:07). Compared with `np.array_equal`, not a tolerance:

  | model | keys | removed | added | survivors NOT bit-identical |
  |---|---|---|---|---|
  | `njl` | 6594 -> 3654 | **3255**, all nested `.state.` | 315, all lifted | **0** |
  | `ccdm` | 6068 -> 3256 | **3040**, all nested `.state.` | 228, all lifted | **0** |
  | `enjl` | 21278 -> 6302 | **14976**, all nested `.point.` | 0 | **0** |

  The three removal counts are exactly the three this ticket predicted before
  any of it ran. Nothing removed was un-nested; nothing added was anything but
  `n_3`/`n_8`/`n_q`; **no surviving key moved by a single bit.** The additions
  are item 2's lift and are not in the ticket's gate wording, which anticipated
  removals only.

  **Independently reproduced.** A concurrent session (eos-b1) ran its own audit
  against the same pre-image from separate code and obtained the same figures.
  It had reached me arguing the gate had a hole — that regenerating in place
  makes `test_baseline` self-comparing and proves nothing about bit-identity —
  which is correct reasoning and would have held had no pre-image been kept.
- Targeted suites, each on a tree only this session was editing:
  `test/njl test/ccdm test/test_imports.py` **349 passed**;
  `test/enjl test/mixed` **391 passed**; `test/test_imports.py` alone
  **215 passed, 0 xfailed**.
- `notebooks/quark_eos.py` follows the record to `_state` and its `.ipynb` pair
  is synced through jupytext (3 stale sites, now 0).
- **OUTSTANDING: the §12 full-suite line.** Not skipped and not claimed. A run
  started 01:08 spanned another session's writes to `eos/dd2/nmp.py`,
  `eos/sfho/parameters.py` and `eos/enjl/thermodynamics.py` at 01:13 and
  01:19-01:21, and was concurrent with a second session's
  `pytest test/dd2 test/baseline/test_baseline.py` — the pair this map records
  as falsely reddening `test_baseline[ccdm]` in BOTH arms. It was killed rather
  than reported, both because it could not be a measurement and because it was
  degrading the other session's baseline comparison. Three or more sessions are
  live on the NMP tickets (111/113/115/116), so the quiet window is a property
  of the tree and no session can declare it. `test/run_clean_suite.sh` (written
  by eos-b1) refuses to start unless no pytest is running and `eos/*.py` has
  been untouched for two minutes, fingerprints every `eos/*.py` around the run,
  and writes `test/suite_certificates/<timestamp>.txt` carrying the window
  bounds, the before/after fingerprint, the CLEAN or DISCARD verdict, the HEAD
  SHA and the interpreter with its numpy/scipy versions. **The full-suite pass
  goes through it, and this line is not satisfied until a CLEAN certificate
  exists** — a path to cite rather than a session's word for it. The
  interpreter field is not decoration: this machine has two stacks that
  disagree, six of thirteen models cross the rtol = 1e-10 gate between them,
  and a bare "N passed" naming no stack has already cost this map a session.

  Two process findings worth keeping. **Killing a compromised run is something
  you can do FOR someone else**: the concurrent-suite trap is recorded here as
  something that produces a false red, but not as something a session can
  choose to stop producing — mine was worthless to me AND harmful to
  c79d1b37's baseline arm, and those are separable reasons, the second
  sufficient on its own. And **the obvious guard for "is a suite running" is
  wrong**: `pgrep -f pytest` matches every session's WAITER loop
  (`until ! pgrep -f "pytest test/"`) as well as the real suites — measured, 6
  matches of which 1 was a suite — so it refuses forever and names the waiting
  session's own waiter as the reason. Match the python process instead:
  `ps -Ao pid=,comm=,args= | awk '$2 ~ /[Pp]ython/ && /pytest/'`.

### Where the work landed, which is not where it looks

**Commit `3fb1c8b`, titled "implement analytic nuclear-matter derivatives and
refactor NMP verification suite".** That is ticket 111's title. It carries all
of ticket 97's three models, `CLAUDE.md`, `eos/mixed/adapters.py` and
`notebooks/quark_eos.py` alongside ticket 111's `dd2/nmp.py` rewrite, because a
concurrent session staged the shared working tree. `git log --grep 97` finds
nothing. This is the third recorded instance of the same structural trap in
this checkout (`5c75584` took ticket 69's dd2 rename; `a4fdfbd` took a
ticket-82 map entry). History is not being rewritten with other sessions live;
the SHA is named here instead, which is the agreed remedy. `4f6f453` is the
enjl docstring fix. `notebooks/quark_eos.ipynb` and the three documents above
remain uncommitted at the time of writing.

### The finding this ticket owes the Stage 7 report

**21271 frozen keys were deleted** and section 12 calls `test/baseline/` ground
truth. The loss is deliberate and is the price of the section 5 fix, but the
question the ticket raised stands unanswered: whether the internal records
deserve a baseline of their own, frozen through the internal path. What can be
said now that could not be said before is that the loss is **precisely** the
nested blocks and nothing else, and that the surviving 12659 keys are
bit-identical — so what was lost is coverage of the internal record, not
coverage of any public quantity.
