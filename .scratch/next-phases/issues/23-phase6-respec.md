# Re-specify Phase 6 against nucleation as it actually is

Type: grilling
Status: open
Blocked by: 20, 22
Parent: ../map.md

## Question

`docs/REFACTOR_PROMPTS.md` Phase 6 is written on premises that no longer hold.
Stage 7 orders it executed verbatim; this map executes it against corrected
premises instead, and this ticket produces them.

Known drift, measured while charting:

| Phase 6 says | Actually |
|---|---|
| "nucleation has no git remote… do not create or push a GitHub repo" | remote exists: `github.com/guerrinimirco/metastability-nucleation` |
| "11.6k lines across 40 files — one pass should do" | 8.1k lines across 38 files |
| (branch unstated) | on `paper-release`; work lands there directly |

Unchanged and still in force: fix every import and call site broken by the `eos`
changes; verify both directions of the §1 dependency rule (`nucleation` depends
on `eos`, `eos` never imports `nucleation`) including after the Phase 3 figure
move; apply the same treatment to `nucleation`'s own code (the `general/` rule,
the same API conventions, the same docstring standard, dead code removed); move
`nucleation/nucleation/tests/` to a top-level `nucleation/test/` and gitignore it,
matching `eos`; improve `nucleation/README.md` to the standard of the new `eos`
README with examples actually run.

**The breakage is much wider than one module, and it is all pre-existing.**
Measured against `eos` at HEAD by importing every target `nucleation` names —
**five of its `eos` modules do not exist and two more are missing the name it
asks for**:

| `nucleation` imports | today |
|---|---|
| `eos.tov.solver` (5 files + a notebook) | gone — it is `eos.astro.tov.solver` |
| `eos.alphabag.eos` | gone — §5 forbids the module name `eos.py` outright |
| `eos.alphabag.thermodynamics_quarks` (2 files) | gone — §5 forbids the sector suffix in a one-sector model package |
| `eos.alphabag.compute_tables` | gone |
| `eos.sfho.compute_tables` (3 files + a notebook) | gone — it is `eos.sfho.table` |
| `eos.alphabag.parameters.get_alphabag_custom` (4 files) | module ok, name gone |
| `eos.sfho.parameters.create_custom_parametrization` (2 files) | module ok, name moved to `eos.sfho.nmp` (§5 puts an NMP-inverting constructor there) |

Everything `nucleation` takes from `eos.general` still resolves — constants,
lepton thermodynamics, `figure_style`, and both `constraints` and the older
`observational_constraints` path. So the damage is entirely in the model and
astro packages, and every one of those breaks is a Phase 3/4 module MOVE, not a
Phase 5 rename: **`nucleation` cannot import `eos` today, before this map
touches anything.**

Two consequences for the brief:

- The Phase 6 pass is not "fix what Phase 5 broke" — it is a port across the
  refactor's module layout, and it must be scoped as one. `nucleation`'s own
  test suite cannot have been green since Phase 3.
- [Ticket 07](07-naming-sweep.md)'s finding that `nucleation` touches none of
  the 58 proposed renames still holds and is still what keeps Phase 5 cheap.
  But `nucleation/composition.py:45-51` imports four
  `compute_alphabag_*_thermo_from_mu` / `compute_cfl_*_thermo_from_mu` names,
  which break §13 rules 1 and 3 (`compute_` prefix, package name repeated) and
  are NOT among the 58. Whether Phase 6 renames them — and so whether
  `alphabag` gets the same treatment `vmit`, `dd2` and `sfho` got — is a
  decision this brief owes. `eos/alphabag/thermodynamics.py:345` already
  defines `thermo_from_mu`, so the package holds both spellings.

Resolved when the corrected Phase 6 brief is written out and the user has agreed
to it. **Creating or pushing a remote stays out of scope** regardless.
