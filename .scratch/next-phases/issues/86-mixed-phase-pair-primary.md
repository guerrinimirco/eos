# Make the Phase pair the parameter argument in `eos/mixed`

Type: task
Status: resolved.
Assignee: session ddd94d70
Blocked by: -   (29 and 84 both resolved)
Parent: ../map.md

## Question

Execution of [ticket 84](84-vmit-params-in-the-plumbing.md)'s ruling. Read it
first: the design is settled and the user's instruction is that **`dd2` and
`vmit` must not be preferred or special with respect to the other models**.

**[Ticket 29](29-mixed-species-flags.md) is DONE** (`8bb546c`) and was the
first half: `eos/mixed/species.py` exists, `api.py` no longer imports
`eos.dd2.species`, the `species` argument is no longer a DD2 type, and the
`muons=None` kwarg is gone from all four signatures — the chain that carried
it now threads the flag object as `species=`.

Two things 29 left for this ticket to finish. The front door's DEFAULT flags
moved out of `api.py` into **`adapters.default_flags()`** — delete it here
along with the rest of the front door. And `_engine_fractions`' sibling check
in `hybrid_table`, the trapped-mode `species.neutrinos` guard, is now narrowed
to `phases is None`, because for a `Phase` pair the adapter's own
`_dd2_wing_kwargs` already raises; when the front door goes, that guard goes
with it.

`eos/mixed/__init__.py:51` was deliberately NOT touched by 29 — the re-export
is this ticket's.

### The inversion

`phases=(Phase, Phase)` becomes **the** parameter argument on all four public
entry points — `eos_point`, `eos_table`, `hybrid_table`, `eos_response`.
`(par, flags, vmit_params)` **retires entirely**, not as a compatibility
overload: a callable that wants DD2+vMIT writes
`phases=default_pair(par, flags, vmit_params)`, and `default_pair` stays in
`adapters.py`. The convenience survives; the privileged position does not.

**The gate is a grep.** `vmit_params` must return **zero** hits in `eos/`. It is
**264** today: `solver.py` 16, `boundaries.py` 13, `table.py` 11,
`responses.py` 9, `hybrid.py` 8, `api.py` 4, `scan.py` 3 (scan goes to
[ticket 87](87-remove-mixed-scan.md)), plus the rest.

### The five module-level model imports that must go

`adapters.py` importing both models is CORRECT — that is what an adapter layer
is. These are not adapters:

    api.py:49        from eos.dd2.species import SpeciesFlags      -> ticket 29
    charges.py:56    from eos.dd2.species import hadronic_qn, hadronic_charges
    responses.py:68  from eos.dd2.solver import warm_start
    scan.py:73,76,77 dd2.nmp, dd2.solver.sweep, vmit.parameters    -> ticket 87
    __init__.py:51   from eos.dd2 import Parameters, SpeciesFlags

**`hadronic_qn` / `hadronic_charges` move to `general/basis.py`.** They read only
`general/particles` and a flags object with `hyperons`/`deltas` — nothing
DD2-specific — so this is §7's single-home rule, NOT a §2 duplication:
`eos/dd2/species.py:23` imports the shared `Particle` objects and re-declares
nothing. There is no duplicate to delete, only a file to move. Check whether
`general/basis.charges_from_densities` then does the same job; if it does, the
merged one wins.

**`responses.py`'s DD2 warm start needs no new contract.** The `Phase` surface
ALREADY provides it: `thermo(mu, mu_C, mu_S, T, n_B_guess=None, x0=None,
return_state=False)` returns `(block, state)` where "`state` is an opaque
internal vector", and the class docstring says outright that "the engine never
sees a model type". `responses.py:68` is a shortcut around a surface that has
been there all along. Note that `:208` also reaches into `flags.phi_field and
flags.hyperons` — DD2-specific FIELDS, not just a DD2 function.

### Gate

- `grep -rn vmit_params eos/` returns nothing.
- No module outside `eos/mixed/adapters.py` imports `eos.dd2` or `eos.vmit`.
- `test/baseline/` for `mixed` unmoved at rtol = 1e-10 — this changes how the
  engine is CALLED, not what it computes. If a number moves, STOP AND REPORT.
- `eos/mixed/verify/run_full_check.py` all `[ok ]`.
- A pairing that is neither DD2 nor vMIT — `sfho_phase` with `njl_phase` — runs
  end to end through the new primary signature. That is the whole point, so it
  is part of the gate rather than a nice-to-have.

CLAUDE.md §5's front-door clause is retired by
[ticket 85](85-claudemd-sentences-owed.md); do not edit CLAUDE.md here.

---

## Resolution

Shipped. `phases=(Phase, Phase)` is the first positional argument of
`eos_point`, `eos_table`, `hybrid_table` and `eos_response`, and
`(par, flags, vmit_params)` is gone from the engine — not deprecated, absent.

### Where the gate was run

**The live tree carried a concurrent session throughout**, and mid-ticket that
session **reverted `eos/mixed/adapters.py` outright** — `default_flags()` came
back and `default_pair` lost its `flags=None`. Nothing imported `default_flags`
any more so the tree still ran, which is exactly why the revert was invisible
until a grep; the edit was re-applied and re-checked. The gates below were
therefore run in an **isolated copy** (`rsync` minus `.git`, `output/`,
`.scratch/`, `__pycache__`), on python.org 3.14.2 / numpy 2.3.5 / scipy 1.17.0.

The isolated copy also inherits that session's in-flight work, which moved
under the gate three times. Attribution, done statically rather than assumed:

- **`eos/vmit/solver.py` gained `par` as positional #1** on all four mode
  solvers (was `params=` keyword). Mid-flight this broke `eos/mixed/adapters.py`
  itself — HEAD's `_vmit_wing_solve` passing `params=` — which read exactly
  like this ticket killing the vMIT wing; that resolved when they landed the
  matching `adapters.py` half. What it left behind is five DIRECT
  `from eos.vmit.solver import ...` call sites in `test/mixed`
  (`test_hybrid_modes`, `test_charges_and_phases`), none of them going through
  a `Phase` or through `eos.mixed` at all. Swept here with the new signature,
  because otherwise this ticket's own gate line cannot be measured.
- **`test_baseline[njl]` and `test_baseline[ccdm]`** fail with
  `ValueError: unknown mode 1.4887` / `1.5` — a density landing in the `mode`
  slot. Traced: `eos/njl/solver.py:solve` went from
  `solve(mode, n_B_fm, T, par, flags, ...)` to `solve(par, mode, n_B, T,
  flags, ...)` (their ticket-90 solver-signature sweep), and
  `generate_baseline.py:698` still calls it in the old order, so `par` gets the
  mode string and `mode` gets the density. `case_ccdm` is the same shape.
  `git diff test/baseline/generate_baseline.py | grep -c 'njl\|ccdm'` is **0**
  — this ticket touches neither case. Left alone: that generator half is
  theirs, and this ticket's baseline line names `mixed.npz` (plus `tov.npz`),
  both of which pass.
- An earlier gate copy also showed `test_baseline[enjl]` and
  `test_enjl_adapter` red. The first is their in-flight ENJL edits; the second
  was MY error — pinning `eos/enjl/` to HEAD while `test/` stayed live, which
  is the documented "overlay invents a foreign failure" trap. Fixed by gating
  against the live tree with no pins at all.

One environment trap re-encountered and mis-diagnosed once here, so it is
worth restating: `timeout <cmd>` on this machine is x86_64, so
`timeout python3.14 ...` runs under Rosetta and arm64 numpy will not load. The
error numpy prints is "you should not try to import numpy from its source
directory", which names the cwd and is about the architecture — the real cause
is chained ABOVE it, so `2>&1 | tail` shows only the misleading half. It looked
flaky (fail, then pass) purely because some invocations carried the `timeout`
prefix and some did not. Drop `timeout`; read the head of the traceback.

### One behaviour the sweep nearly lost, and the fix

Deleting `hybrid_table`'s trapped-mode guard was correct — the DD2 adapter's
`_dd2_wing_kwargs` does raise for the same case — but the guard sat OUTSIDE
`hybrid_table`'s `try/except (RuntimeError, ValueError)`, and the adapter's
raise happens inside it. So a trapped-neutrino call with no neutrino
population stopped RAISING and started coming back as `HybridResult(ok=False)`.
That is a §6 violation: a malformed call is a programming error and must raise,
while non-convergence is the return value.

Caught by `test_trapped_hybrid_without_neutrino_flag_raises`, and fixed
properly rather than by putting the old guard back: `build_hybrid_table`'s
three-line wing pre-flight is now `hybrid.validate_wings(phases, spec, T)`, one
helper called from `build_hybrid_table` as before AND from `hybrid_table`
before its try. Both refusals — a phase with no `wing_sweep`, and a phase whose
`wing_sweep` cannot dispatch this spec — now raise from either entry point, and
the engine still asks the adapter rather than carrying a list of conditions of
its own.

### The gate

**`grep -rn vmit_params eos/` — 263 hits before, and now:**

| where | before | after |
|---|---|---|
| `eos/mixed/` (solver, boundaries, table, responses, hybrid, api, verify) | 84 | **0** |
| `eos/mixed/adapters.py` — `default_pair`'s own parameter | 4 | 2 |
| prose quoting `default_pair(par, flags, vmit_params)` | — | 6 |
| `eos/vmit/` stale docstrings | 2 | 0 |
| `eos/zlvmit/` (legacy, §1-exempt, its own API) | 165 | 165 |

The eight surviving hits outside `zlvmit` are the ruling's own prescribed call
form — `default_pair(par, flags, vmit_params)` — in the function's signature
and in six sentences quoting it. Nothing threads a `vmit_params` keyword
anywhere. `zlvmit` was never in scope: §1 exempts it and it is not this
engine.

**No module outside `eos/mixed/adapters.py` imports `eos.dd2` or `eos.vmit`**,
with two named and pre-existing exceptions this ticket did not create and
should not have folded in:

- `eos/mixed/backends/jacobian.py` imports both models' `thermodynamics`. It
  is the accelerated flavour of the two shipped adapters and is reached only
  through the closures `dd2_phase` / `vmit_phase` install in
  `Phase.jacobian_block`. §5 defines `backends/` by the property that
  deleting it changes no number, and §9 keeps it separate from the reference
  path on purpose; moving those kernels into `adapters.py` would break both
  rules to satisfy a one-line summary of this one. Left, deliberately.
- `eos/mixed/verify/run_full_check.py` imports `eos.dd2` and
  `eos.vmit.parameters` to build the pairing it checks. §1's `verify/`
  carve-out covers exactly this.

**Numbers unmoved.** `test/baseline/` `mixed.npz` and `tov.npz` pass at their
frozen tolerances on the canonical stack (python.org 3.14.2 / numpy 2.3.5 /
scipy 1.17.0). This was a signature change; nothing moved, and nothing was
regenerated.

**Full gate: `test/mixed` + `test/baseline`, 281 collected, 279 passed /
2 failed** — `test/mixed` entirely green across all 27 files, and the two
failures are `test_baseline[njl]` and `test_baseline[ccdm]`, both attributed
above to the concurrent session's solver-signature sweep in call sites this
ticket does not touch.

**`eos/mixed/verify/run_full_check.py`: PASS**, all ten `[ok ]` on 3.14 —
euler/HVH 8.9e-15, free energy 8.9e-15, mechanical eq 4.1e-14, Gibbs/Maxwell
1.1e-12, cross-mode repro 3.2e-13, analytic J ~ FD 1.2e-8, backend parity
5.1e-15, causality (max c_s^2 = 0.484), sound speeds 5.8e-15, TOV M_max = 2.322
at R = 12.66 km. (On anaconda 3.9 the same suite passes at 2.340 / 12.64 km —
the stack difference, both far inside the check's own 1.9-2.6 window.)

**A pairing that is neither DD2 nor vMIT** runs end to end through the new
primary signature: `test/mixed/test_phase_pairs.py`, `test_njl_pair.py`,
`test_ccdm_pair.py` and `test_enjl_pair.py` all drive DD2+alphaBag, ZL+vMIT,
DID+NJL, DID+CCDM and the ENJL branch pair through it, and the notebook's
section 3 now runs SFHo+NJL beside DD2+vMIT in the same call.

### What changed, module by module

**The signature.** `phases` is positional #1 everywhere and `species` stays
positional #3 — so the composite engine now reads as §5's
`eos_point(par, mode, species, **conditions)` with the pair in `par`'s place,
which is what §5 already said and the code did not do. Internally:

    solve(phases, n_B, eta, spec, T=, x0=, n_B_guess=, species=, ...)
    sweep(phases, n_B_grid, eta, spec, T=, ...)
    locate_window(phases, n_B_grid, eta, spec, T=, ...)
    locate_windows / refine_window / solve_fixed_chi / solve_at_entropy
    build_hybrid_table(phases, n_B_grid, eta, spec, T=, ...)
    mass_radius_mixed(phases, n_B_grid, eta, spec, T=, ...)
    build_mixed_ctx(spec, eta, n_B, phases, T=, ...)
    TableSpec(phases, mode, axes, ...)
    sound_speed_frozen(phases, result, ...) / frozen_along(phases, results, ...)
    solve_<mode>(phases, n_B, ..., eta, T=, **kw)

`MixedCtx` lost its `par`, `flags` and `vmit_params` fields — they existed only
so a caller could read back what it had passed in. `mixed_slots` lost its
`flags` argument entirely (see below), so it is now
`mixed_slots(spec, eta, pair=None, fixed_chi=False)`: the slot list is a
function of the spec, eta and the pairing, and a species flag has no route into
it at all. That is a stronger statement than the test which used to check it.

**`adapters.default_flags()` deleted** with the rest of the front door.
`default_pair(par, flags=None, vmit_params=None)` stays and absorbed its one
job: `flags=None` now means DD2's own `SpeciesFlags()`, which is where a
DD2+vMIT default belongs.

**`hybrid_table`'s trapped-mode `species.neutrinos` guard deleted.** It was
narrowed to `phases is None` by ticket 29 and had nothing left to guard: for a
`Phase` pair `build_hybrid_table` validates each wing by calling
`wing_sweep(spec, grid[:0], T)` before locating anything, and the DD2 adapter's
`_dd2_wing_kwargs` raises there with the same message. Still before any solve,
which is what the guard was for.

**`eos/mixed/__init__.py:51` re-export** — it was in the module docstring's
"Typical use", not an import statement, which is why ticket 29 could leave it.
`from eos.dd2 import Parameters, SpeciesFlags` is gone; the example now builds
a pairing two ways and shows both. The package also **now exports its own**
`SpeciesFlags` and `mixture_flags` — ticket 29 created `eos/mixed/species.py`
but never put it on the package surface, so `eos.mixed.SpeciesFlags` did not
resolve until this ticket.

**`hadronic_qn` / `hadronic_charges` moved to `eos/general/basis.py`**, with
`active_baryons` (which `hadronic_qn` is written in terms of) travelling with
them. `eos/dd2/species.py` imports all three back and re-exports them, so
`eos.dd2.hadronic_charges` still resolves and `eos/dd2/table.py` is untouched.
`eos/mixed/charges.py` takes them from `general/basis` beside `quark_charges`.

**`charges_from_densities` does NOT do the same job**, so nothing merged. It
sums every non-lepton species in the map; `hadronic_charges` sums only the
baryons the flags declare active and ignores the rest — which is what
`eos/dd2/table.py` needs, because a thermal meson gas travels in the same
dictionary and must not enter a baryons-only sum. The two now sit side by side
in `general/basis.py` with each docstring naming the other, which is the useful
outcome of the check the ticket asked for.

`eos/did/species.py` and `eos/sfho/species.py` still carry their own
`active_baryons`. They are byte-identical to the shared one and are a second
instance of the same §7 finding — out of scope here (this ticket's charter is
`eos/mixed`), and recorded rather than swept.

**`responses.py`'s DD2 warm start is gone, and so are the two model-specific
wings.** `from eos.dd2.solver import warm_start` and the `flags.phi_field and
flags.hyperons` read at `:208` were both inside
`sound_speed_frozen_hadronic(par, flags, point, ...)`, which with
`sound_speed_frozen_quark` made two model-specific spellings of one idea.
They are replaced by ONE contract-driven function:

    sound_speed_frozen_pure(phase, th, T=0.0, rel_dn=1e-3,
                            leptons=True, muons=True, mu_slot=None)

It goes through `Phase.frozen_thermo`, which is the surface ticket 84 pointed
out had been there all along, and it works for either wing of any pairing —
the DD2 warm start the old function hand-rolled is what `dd2_phase`'s own
`frozen_thermo` already does from `mu_slot`. `test_mixed_responses.py`'s two
"entry point is the chi->0 / chi->1 limit" tests became one, and its docstring
says what the check now pins: the mixture's endpoint weighting and lepton
bookkeeping, not two implementations against each other. The two-implementation
version of that check WAS the DD2 shortcut.

### Swept call sites

- `eos/mixed/verify/run_full_check.py`: every check now takes one `pair`
  instead of `(par, flags, vp)`; `run_full_check(pair=None, grid=None,
  include_tov=True)` defaults to the DD2+vMIT pairing and says in its docstring
  that another pairing needs a grid bracketing ITS window.
- `test/mixed/` — all 27 files.
- `test/baseline/generate_baseline.py` (`case_mixed`, `case_tov`), which had to
  gain an explicit `species=MixedFlags(muons=True)`: the front door read the
  mixture's muon flag off the hadronic `flags`, and without it the numbers
  WOULD have moved. They did not.
- `test/test_nonconvergence_return.py`, `test/tov/test_solver_fast_robustness.py`,
  `test/dd2/test_table_rows.py`.
- `notebooks/hybrid_eos.{py,ipynb}`, jupytext-synced. Section 3 was "two calling
  forms for the same physics" and is now "the parameter argument of a composite
  engine": one signature, DD2+vMIT built by `default_pair` and by hand, plus a
  second pairing (SFHo+NJL, a knob) run beside it. **The notebook was already
  broken before this ticket** — it still passed `muons=KNOBS.species["muons"]`,
  a kwarg ticket 29 removed, which `_engine_fractions` would have rejected as a
  fraction the mode does not take. Swept here.
- `docs/STRUCTURE.md` §11 (three worked examples), `eos/mixed/mixed.{md,tex}`
  (§11's own document, API surface and the front-door paragraphs),
  `eos/vmit/parameters.py` and `eos/vmit/vmit.md` (both told a reader to sweep
  parameters through `eos.mixed.eos_table(..., vmit_params=...)`),
  `README.md` and `eos/__init__.py` (both said `eos.mixed` "reuses" dd2's
  flags — it has had its own since ticket 29), `docs/DEFERRED.md:1120`.

**`../nucleation` on `paper-release`: nothing to sweep, and that is a measured
result rather than an assumption.** A grep for `mixed`, `eos.mixed`,
`vmit_params`, `hybrid_table`, `eos_point`, `eos_table` and `eos_response`
across `*.py`, `*.ipynb`, `*.md` and the packaging files — the notebook cells
and lazy imports the map's fog names as ticket 24's blind spot included — finds
`eos` used only through `eos.sfho.table.load_eos_table` /
`build_interpolators` in `README.md`, `test/conftest.py` and
`notebooks/2fam_PNS_nucleation.py`. The four occurrences of "mixed" in that
repo are a local variable in `nucleation/critical.py`, a `SimpleNamespace` in
`nucleation/tables/thermal.py` and one sentence of prose. `eos.mixed` is not
imported there at all, at module scope or lazily.

### Not done here, and why

- **CLAUDE.md is untouched**, as the ticket directs. The §5 sentences this owes
  are recorded in [ticket 85](85-claudemd-sentences-owed.md) item 3, with the
  replacement clause written out and marked SHIPPED, plus the note that §7
  needs no new sentence for the `hadronic_qn` move (§2 already required it) and
  that item 3's "ticket 81" was a renumber straggler for ticket 84.
- `eos/mixed/backends/jacobian.py` and the `verify/` suite keep their model
  imports, for the two reasons given above.
- `mixed_slots(spec, eta)` with no pairing still returns the
  (kinetic H, physical Q) slot names. That is not a DD2+vMIT privilege — it is
  what any pairing with a density-dependent hadronic phase derives — and it is
  what lets a caller ask for a spec's slot ORDER without holding a pairing.
  The docstring now says that instead of naming `default_pair`.

**CLAUDE.md landed** by [ticket 85](85-claudemd-sentences-owed.md): **§5**'s
phase-adapter paragraph now reads "in the first position of every public
entry point; the DD2+vMIT pairing is built by
`adapters.default_pair(par, flags, vmit_params)`, a call rather than a
privileged position".
