# `zl.solver` takes `flags`, and `include_photons` goes

Type: task
Status: open
Blocked by: [90](90-solver-signature-and-units-sweep.md)
Parent: ../map.md

## Question

Split out of [ticket 90](90-solver-signature-and-units-sweep.md), which was
ruled by [ticket 81](81-second-default-solver-kwargs.md) §4 and gated on
"**no value moves anywhere**". That gate is false for this model, so the work
leaves ticket 90 rather than the gate being weakened — the split
[ticket 89](89-dd2-honours-species-flags.md) was given, applied to the three
models ticket 81 wrongly believed would move zero rows.

`eos/zl/solver.py` carries `include_photons: bool = True` on
`solve_beta_eq_neutrinoless`, `solve_fixed_yc`,
`solve_beta_eq_neutrino_trapped` and on `_finish`, and takes no
`SpeciesFlags` at all. `eos/zl/species.py:32` already carries
`photons: bool = False`, and `eos/zl/table.py:46` is the only thing that
translates one into the other — the ticket-89 shape exactly, one layer lower:
a solver's own callers get a photon gas whatever the flags say.

**Why values move.** `test/baseline/generate_baseline.py:468` (`case_zl`)
calls the raw solvers and names `params=` and `include_electrons=` only. Once
`solve` reads `flags.photons`, the generator picks up the §4 default, which is
`False` since [ticket 65](65-species-flag-defaults.md). Ticket 81 predicted
"zl moves **zero** rows"; it moves every row at T > 0.

`photons` is the ONLY live flag here. `SpeciesFlags.__post_init__` raises on
`hyperons`, `deltas`, `thermal_mesons`, `muons` and `thermal_neutrinos`, so
there is no second sector for the solver to honour or to forget.

## Work

1. `flags: SpeciesFlags` becomes a required argument of every solver in
   `eos/zl/solver.py`, placed after `par` (ticket 90 has already put `par`
   first and required, and spelled it `par`). `solve` and `_finish` read
   `flags.photons`; `include_photons` is deleted from both and from every
   wrapper.
2. `include_electrons` -> `leptons`, a SEPARATE named argument — §5, and
   already ruled by [ticket 70](70-leptons-on-a-beta-mode.md). It is not a
   species flag and never enters `SpeciesFlags`.
3. `eos/zl/table.py:46` stops translating and passes the flags through.
4. The 19 call sites outside `solver.py` name their flags. A site whose P,
   eps and s are DISCARDED (a warm-start seed, a potentials-only extraction)
   takes `photons=False` and says so in a comment — ticket 89 found three such
   sites its own work list had missed, so enumerate them from the source
   before editing rather than trusting this count.
5. Regenerate `test/baseline/zl.npz` under
   [ticket 65](65-species-flag-defaults.md)'s measure-then-regenerate gate.

## Gate

- **Measure BEFORE regenerating.** Re-evaluate `case_zl()` against the frozen
  file and diff key by key at rtol = 1e-10. Record moved, unmoved, and the
  residue.
- **Every moved key is P, eps or s at T > 0, and moves by exactly one photon
  gas.** Photons carry no conserved charge, so every density, every potential,
  every Y_C and Y_S is unchanged. A moved composition key is a bug, not a
  regeneration.
- **Zero keys move at T = 0.** `TEMPERATURES = (0.0, 10.0, 30.0)`, so the
  T = 0 third of the file is the control the same run carries with it.
- No other `test/baseline/*.npz` moves at rtol = 1e-10 — `mixed.npz`
  ASSERTED, not assumed, if any zl code is on its path.
- Full suite green, zero added failures against
  `output/_audit/pytest_before.txt`; `zl` `verify/` green.
