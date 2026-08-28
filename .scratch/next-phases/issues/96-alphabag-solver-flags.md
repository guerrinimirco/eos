# `alphabag.solver` takes `flags`, and all three `include_*` sectors go

Type: task
Status: open
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
