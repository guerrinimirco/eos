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

**One breakage already measured** ([ticket 07](07-naming-sweep.md)): `nucleation`
imports `eos.tov.solver` in five files, and `eos/eos/tov/` does not exist — it is
`eos/astro/tov`. Those imports are broken today, before any Phase 5 change. The
same ticket verified that `nucleation` touches none of the 58 proposed `eos`
renames, so the import path is the drift that matters, not the vocabulary.

Resolved when the corrected Phase 6 brief is written out and the user has agreed
to it. **Creating or pushing a remote stays out of scope** regardless.
