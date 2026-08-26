# `leptons=False` by default, and the checks that hold the line

Type: task
Status: open
Blocked by: -
Parent: ../map.md

## Question

Ruled by [ticket 81](81-second-default-solver-kwargs.md), sections 3 and 6.
Split out because it moves no baseline row and needs no regeneration — the
gate is a green suite and three new checks.

§3 names `leptons` as the orthogonal flag of `fixed_YC` and `fixed_YC_YS` but
never states its default, and nine models disagree: `False` in
dd2/sfho/did/alphabag, `True` in enjl/njl/ccdm/zl/vmit and in
`eos/mixed/solver.py:716`.

## Work

1. `leptons=False` everywhere it appears as a default. Ticket 65's rule is "off
   unless asked", and `leptons=True` is the one direction that silently ADDS a
   sector.
2. The legacy `TableSettings` layer, a THIRD place the same sectors carry a
   default: `zl/table.py:238-239` (`include_photons=True`,
   `include_leptons=True`) and `alphabag/table.py:283-286`
   (`include_photons`, `include_gluons`, `include_thermal_neutrinos` all True,
   `include_electrons=False`). Follow the flags object, as
   `zl/table.py:277` already does for `photons`.
3. Three checks in `test/test_imports.py`, beside
   `test_the_six_species_flags_all_default_to_off` — the precedent ticket 65
   set and this ticket's ruling names:
   - `leptons=` defaults to False in every model that has the argument.
   - No solver accepts BOTH a `SpeciesFlags` and a parallel `include_*` kwarg
     for a sector the flags object carries. This is what ticket 89 fixed in
     dd2 and ticket 90 removed from zl/vmit/alphabag; without the check
     nothing stops it coming back.
   - **The units band check**, parametrised over all ten models: call
     `eos_point`, walk every float field on the result, assert P and eps sit
     in an fm-plausible band and densities (including `n_s` and `s`) in
     theirs. Write it against MAGNITUDE BANDS, not values — a natural-unit
     field is off by `hc3 = 7.68e6`, six orders of magnitude, so the check
     needs no tolerance and can never become a second baseline to maintain.
     This is the check that found `njl`, `ccdm` and `enjl` in ticket 81.

## Gate

- No `test/baseline/*.npz` moves. All 16 `leptons=`/`include_electrons=` sites
  in `generate_baseline.py` pass the flag explicitly, so the default is not
  load-bearing for any frozen row. **If a row moves, the ruling was applied
  somewhere it was not meant to reach.**
- The band check must FAIL on `njl`/`ccdm`/`enjl` before ticket 90 lands and
  pass after — run it against `main` first to confirm it detects, rather than
  shipping a check that passes because it looks at nothing.
- Full suite green (§12).
