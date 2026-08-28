# `alphabag.solver` takes `flags`, and all three `include_*` sectors go

Type: task
Status: resolved
Blocked by: [90](90-solver-signature-and-units-sweep.md)
            ([92](92-cfl-gluon-term.md) is RESOLVED — see below)
Parent: ../map.md

## Question

Split out of [ticket 90](90-solver-signature-and-units-sweep.md). This is the
model whose measurement broke ticket 90's premise, recorded there by
[ticket 82](82-alphabag-gluons-default.md).

`eos/alphabag/solver.py` carries **three** sector kwargs, all defaulting
`True` — `include_photons`, `include_gluons`, `include_thermal_neutrinos` — on
`solve_beta_eq_neutrinoless:396`, `solve_beta_eq_neutrino_trapped:473`,
`solve_fixed_yc:557`, `solve_fixed_yc_ys:646` and on the two inner solves at
`:233` and `:325`, with `solve_cfl:758` carrying photons and gluons.
`eos/alphabag/species.py:52-54` already carries all three as fields, every one
`False`. `eos/alphabag/table.py:59-79` is the only translator. Roughly 40
lines of the solver are pass-through of the three names.

**Why values move**, measured through `eos_point` at n_B = 0.8 for the gluon
sector alone ([ticket 82](82-alphabag-gluons-default.md)):

    beta.T0     unchanged — every thermal sector vanishes at T = 0
    beta.T10    P  -1.465838e-03 MeV/fm^3
    beta.T30    P  -1.187329e-01 MeV/fm^3

`test/baseline/generate_baseline.py:531` (`case_alphabag`) calls the raw
solvers and names none of the three, so it picks up all three §4 defaults at
once: `photons` and `thermal_neutrinos` since
[ticket 65](65-species-flag-defaults.md), `gluons` since ticket 82.

**[Ticket 92](92-cfl-gluon-term.md) has RULED, and this ticket was waiting on
it.** `solve_cfl` raises on `gluons=True` and the `cfl` arm of
`table.solve_at` raises on `thermal_neutrinos`. So for the paired phase
section 1 below re-points two existing `NotImplementedError`s at the flags
object rather than translating two kwargs; there is no gluon term in `cfl` to
read a flag for. The `cfl.*` baseline rows are all at T = 0 and move under
neither answer, which ticket 92 re-measured.

## Work

1. `flags: SpeciesFlags` required, after `par` (ticket 90 has already put
   `par` first and spelled it `par`). Every solver reads `flags.photons`,
   `flags.gluons` and `flags.thermal_neutrinos`; the three `include_*` names
   are deleted from the four mode solvers, the two inner solves, and
   `solve_cfl` — about 40 lines of pass-through go with them.
2. `solve_cfl`'s and `table.solve_at`'s refusals read the flags object
   instead of the kwargs, keeping ticket 92's ruling exactly: `gluons=True`
   raises in `cfl`, `thermal_neutrinos=True` raises in `cfl`, and neither is
   silently dropped (§4).
3. `include_electrons` -> `leptons`, a SEPARATE named argument (§5,
   [ticket 70](70-leptons-on-a-beta-mode.md)), never a species flag. Note it
   already defaults `False` here, unlike zl/vmit.
4. `eos/alphabag/table.py:59-79` stops translating and passes the flags
   through. `TableSettings.include_thermal_neutrinos:286` and its two readers
   at `:347` and `:419` go the same way.
5. The 32 call sites outside `solver.py` name their flags. Enumerate from the
   source — [ticket 89](89-dd2-honours-species-flags.md) found three sites its
   own list had missed, and one of them was a frozen block whose P and eps ARE
   the frozen sound speed.
6. Regenerate `test/baseline/alphabag.npz` under ticket 65's
   measure-then-regenerate gate.

## Gate

- **Measure BEFORE regenerating**: re-evaluate `case_alphabag()` against the
  frozen file, key by key at rtol = 1e-10.
- **Every moved key is P, eps or s at T > 0, and the move decomposes into the
  three sectors named** — photon gas + gluon gas + thermal neutrino gases,
  each evaluated on its own at the same (T, alpha_s) and summed. Ticket 82
  gated exactly this way and got machine precision; anything left over is not
  a sector deletion.
- **Zero keys move at T = 0** — including every `cfl.*` row, which is why
  ticket 92's answer cannot move a frozen number.
- No density, no potential, no Y_C, no Y_S moves: none of the three sectors
  carries a conserved charge.
- No other `test/baseline/*.npz` moves at rtol = 1e-10.
- Full suite green, zero added failures against
  `output/_audit/pytest_before.txt`; `alphabag` `verify/` green.
- README example 1 quotes a captured alphaBag number and was already wrong
  once for this reason (ticket 82, recaptured). Recapture it if it moves.


---

## Answer from [ticket 92](92-cfl-gluon-term.md) (2026-08-27, resolved)

**`solve_cfl` RAISES on the gluon sector**, the recommendation this ticket
anticipated, and the ruling reached one flag more than that. Both refusals are
already written and shipped, so work item 2 is a re-point rather than a
decision:

- `solve_cfl` (`solver.py`) defaults `include_gluons=False` and raises
  `NotImplementedError` on `True`. The reason is the phase, not the
  temperature: locking leaves a single unbroken `U(1)_Qtilde`, so of the nine
  gauge bosons only the rotated photon stays massless and all eight gluons are
  Meissner-massive. **`include_photons` is unaffected** and stays a real
  sector — the rotated photon is a free massless gas of two polarizations, the
  same `photon_thermo(T)` the unpaired phase carries.
- `table.solve_at`'s `cfl` arm raises on `species.thermal_neutrinos`, which it
  had been **dropping silently**. The physics gap (the paired phase has never
  carried the gas; closing it moves published CFL tables) stays deferred; only
  the reporting moved.

**What this changes for the work items here.**

- Item 1: `solve_cfl` reads `flags.photons` and raises on `flags.gluons`. The
  kwarg is kept for now precisely so this ticket can delete it in one place
  with the reason already beside it.
- Item 2: settled, both sectors, nothing left to decide.
- Item 4: `table.py`'s `cfl` arm keeps its two raises when it stops
  translating — they move from the kwarg to the flag, they do not disappear.
- Item 5: **the legacy `TableSettings` shim now needs both flags named on its
  `cfl` path**, because it carries one sector switch set for both phases.
  `test/alphabag/test_alphabag_api.py` already names them; a legacy CFL table
  is therefore no longer bit-identical to the published first-generation one,
  which is recorded in `docs/DEFERRED.md`.

**The category question this ticket will meet, answered.** A per-mode refusal
is NOT ticket 82's abolished third category: 82's rule is about the FLAG,
judged over the modes the model has, and this refusal is about the PHASE.
`alphabag.gluons` keeps two legal values in the three unpaired modes and stays
a default; `test_the_gluon_flag_is_still_a_default_in_the_unpaired_modes` pins
it. The drift check `test_every_species_flag_defaults_off_or_raises` iterates
DEFAULTS and needs no exemption. The §4 sentence is owed to
[ticket 85](85-claudemd-sentences-owed.md) item 5.

---

## Note from [ticket 94](94-zl-solver-flags.md) (2026-08-28)

This model's `eos/mixed` adapter takes **no flags object** — it is
`(params=None)` — so its `wing_sweep` cannot carry the caller's own `photons`
the way `eos/mixed/species.py` says a wing must. Ticket 94 hit the same thing
in `zl_phase` and gave it `photons=False` rather than invent an API three
times. **Do the same here and do NOT add a `flags=` parameter**: that is
[ticket 109](109-flagless-mixed-adapters.md), which is blocked by this ticket
and is one ruling covering all three adapters.

Two more findings from 94 that apply directly:

- **The signature is sfho's and did's**, `(par, n_B, [fraction], flags, T)`,
  not the literal "after `par`" these tickets were written with — that would
  be a fourth argument order against §13. Do not rename `initial_guess` to
  `x0` or give `T` a default; ticket 90 left both alone everywhere.
- **`leptons` keeps whatever default it has.**
  [Ticket 91](91-leptons-default-and-drift-checks.md) owns the flip to False,
  and moving it here moves rows the measure-then-regenerate gate does not
  allow for.

## Resolution

Executed on python.org 3.14.2 (numpy 2.3.5, scipy 1.17.0). `eos/alphabag/solver.py`
takes a required `flags: SpeciesFlags` at every entry point — the four mode
solvers, `solve_cfl`, and the two inner `*_point_from_mu` builders — and the
three `include_*` names are gone from all seven, about forty lines of
pass-through with them. `include_electrons` is `leptons`; `table.py` stops
translating and passes the flags object through.

### The signature is njl's, ccdm's and now vmit's

The same finding [ticket 95](95-vmit-solver-flags.md) recorded and for the same
reason: the quark models order `(par, n_B, [fraction], T, flags, ...)`, with `T`
before the flags, where the hadronic three order it the other way. alphaBag's
`T` was already the third positional, so every call site gains an argument
rather than reordering. `solve_cfl` is `(par, n_B, T, Delta0, flags,
initial_guess=None)`.

### The three cfl refusals moved INTO `solve_cfl`

Work item 2 said "`solve_cfl`'s and `table.solve_at`'s refusals read the flags
object instead of the kwargs". Read literally that leaves `solve_cfl` — which
now receives the flags object itself — silently ignoring `thermal_neutrinos`
and `two_flavour`, because [ticket 92](92-cfl-gluon-term.md) put those two
refusals in `table.solve_at` where only the table path could see them. A direct
`solve_cfl` caller would then get the no-op §4 forbids, through the very object
that was supposed to end it.

All three now raise inside `solve_cfl`, with ticket 92's messages verbatim, and
`table.solve_at`'s `cfl` arm keeps a comment pointing at them rather than a
second copy. **Nothing about ticket 92's ruling changed** — same three sectors,
same reasons, same `NotImplementedError` — only the place, and the place is now
where the flag arrives. It is also less code: one refusal apiece instead of two.

### `two_flavour` went into the flags object, as it did in vmit

Same shape, same reason: `SpeciesFlags.two_flavour` existed AND every solver
carried a parallel `two_flavour: bool = False` kwarg for the same sector, which
is what [ticket 91](91-leptons-default-and-drift-checks.md)'s second drift
check forbids. Value-neutral — both spellings defaulted False.

### The measurement, taken BEFORE regenerating

`alphabag.npz` has **1158 keys**.

    control, the three old kwarg defaults       0 moved of 1158
    moved at rtol = 1e-10                     120   (30 points x 4 fields)
    moved keys that are not P/eps/s/f           0
    moved keys at T = 0                         0
    moved `cfl.*` keys                          0 of 12
    decomposes into gamma + gluons + nu_th    EXACT at 92 of 120
    worst relative residue                    2.013e-16

The control is the delta's own null: the same new code run with
`SpeciesFlags(photons=True, gluons=True, thermal_neutrinos=True)` for the
unpaired modes and `SpeciesFlags(photons=True)` for `cfl` — the exact sector
configuration the seven deleted kwarg defaults gave — reproduces the frozen
file at 0 of 1158.

**The moved field list is four names, not three**: `P_total`, `e_total`,
`s_total` AND `f_total`. `f = eps - T s` is a combination of two of them and
moves by `de - T ds` of the same three gases; it is not a fourth quantity and
not a composition key. It is also why 28 of the 120 are not bit-exact — a
derived difference cannot be — and why the other 28 land at 2.0e-16 relative,
which is one machine epsilon: unlike zl and vmit, where deleting the LAST
addition leaves the preceding sum bit-for-bit, here three gases are added into
a running total and removing them re-associates the sum.

**Ticket 82's numbers reproduce to every digit quoted.** The gluon component
alone, at the shipped `alpha`:

    T =  0    -0.000000e+00 MeV/fm^3      (ticket 82: unchanged)
    T = 10    -1.465838e-03 MeV/fm^3      (ticket 82: -1.465838e-03)
    T = 30    -1.187329e-01 MeV/fm^3      (ticket 82: -1.187329e-01)

so the delta measured here contains ticket 82's measured delta exactly, plus
the photon and thermal-neutrino gases ticket 65 had already defaulted off.

**Zero `cfl.*` keys moved**, as ticket 92 predicted: all twelve are at T = 0,
where no thermal sector contributes at all, so the CFL ruling could not have
moved a frozen number under either answer.

md5 over all fourteen `test/baseline/*.npz` before and after: **only
`alphabag.npz` differs**; the other thirteen are BYTE-identical. Key-by-key
diff at `output/_audit/baseline_diff_ticket96_py314.txt`.

### The legacy `TableSettings` layer

Work item 4 and [ticket 91](91-leptons-default-and-drift-checks.md) item 2.
All four sector fields — `include_photons`, `include_gluons`,
`include_electrons`, `include_thermal_neutrinos` — now default `False`, which
is what `SpeciesFlags` defaults them to, so the shim FOLLOWS the flags object
instead of being a third place the same switch is thrown. `docs/DEFERRED.md`'s
paragraph is rewritten: the shim no longer raises on a default CFL table, it
returns one without the sectors, and the reason it still cannot reproduce the
first-generation tables is ticket 92's ruling rather than the defaults.

### Call sites: what the enumeration found

- **`eos/mixed/adapters.py`'s `alphabag_phase` takes no flags object**, the
  third of the three ticket 94 named. Given a bare `SpeciesFlags()` throughout
  — not just `photons=False`: the gluon gas and the thermal neutrino gases are
  phase-common thermal sectors in exactly the same sense, and this phase has no
  caller flags to follow either. **No `flags=` parameter added**;
  [ticket 109](109-flagless-mixed-adapters.md) owns that for all three.
  `eos/mixed/species.py` now names all three adapters.
- **`test/abpr/test_abpr_alphabag_limit.py` is the §1 carve-out and needed
  eight edits.** It is the file CLAUDE.md §1 exempts by name — "abpr checks
  itself against the CFL phase of alphabag" — and it calls
  `alphabag.solve_cfl` directly. A sweep over `eos/` and `test/alphabag/` does
  not see it. Given `SpeciesFlags()`: the comparison is at T = 0 on both sides,
  so no thermal sector is in either.
- **One test shadowed its own module import.**
  `test_the_paired_phase_refuses_the_free_gluon_gas` had a function-local
  `from eos.alphabag import SpeciesFlags, eos_point` after a line that now uses
  the module-level `SpeciesFlags`, which Python reads as an unbound local. The
  local imports are gone. Not a physics defect, but the second time in this
  session that a call site was invisible to grep and visible only to the run.

### The README needs no recapture

The gate's last line says "README example 1 quotes a captured alphaBag number".
It no longer does: all five README examples are `dd2`, `mixed` and the M–R
sequence, and the only alphaBag mentions are the model table and two prose
sentences about `gluons` being a default. Ticket 82's recapture is what moved
it. Nothing to redo; the line is stale and is recorded here rather than acted
on.

### Gate

    interpreter   python.org 3.14.2 / numpy 2.3.5 / scipy 1.17.0

    test/alphabag + test/abpr   314 passed, 0 failed
    alphabag/verify             PASS, all eleven checks
    other .npz                  13 of 13 BYTE-identical

