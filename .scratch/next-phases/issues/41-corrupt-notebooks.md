# ZLvMIT_hybrid.ipynb is corrupt JSON and cannot be opened

Type: task
Status: resolved
Parent: ../map.md

## Question

Commit `d9f8eec` ("refactor(astro/tov): crust attachment gets its own module")
mechanically rewrote `eos.tov` -> `eos.astro.tov` across `notebooks/` and broke
the JSON of all three `.ipynb` files it touched. `d9f8eec^` is valid JSON in
every case; `d9f8eec` is not. `nbformat.read` refuses all three, so Jupyter
cannot open them.

Two of the three were removal candidates and are gone under
[ticket 03](03-stage0-removals.md). The third is on the KEEP list:

    notebooks/ZLvMIT_hybrid.ipynb   112 kB, tracked, 29 code cells, 0 outputs

CLAUDE.md §11 keeps `zlvmit` for its published results, and a file nobody can
open is not being kept.

**The damage.** The rename replaced a single JSON source line with text split
across real newlines without re-escaping, leaving an unterminated string:

    line 1640-1642   "from eos.general.state import EOSTable_for_TOV
                     from eos.astro.tov.solver import add_crust\n"
    line 2482        a source-array element missing its trailing comma

Restore the string termination and the comma. The notebook carries no stored
outputs, so `d9f8eec^` plus a correctly-applied rename is an equally valid
route and loses nothing.

Resolved when `nbformat.read("notebooks/ZLvMIT_hybrid.ipynb", as_version=4)`
succeeds, the imports still name `eos.astro.tov`, and the suite shows 0 added
failures against `output/_audit/pytest_final.txt`.

**Carry to the Stage 7 report** ([ticket 25](25-acceptance.md)): a mechanical
rename shipped three broken notebooks and nothing in the repository checks that
an `.ipynb` is loadable, so the breakage survived five days and four commits
undetected. Whether a validity check belongs in the suite is a question for
[ticket 21](21-phase5-structure.md), not this ticket.

## Answer

**Repaired in place; `nbformat.read` succeeds.** 48 cells, 29 code cells, 0
stored outputs — the same shape `d9f8eec^` had, so nothing was lost by fixing
forward rather than reverting.

Both defects were exactly as the ticket described, and both came from the same
mechanical rewrite. The first was not a string swap at all: `d9f8eec^` read

    from eos.astro.tov.solver import load_crust_table, EOSTable_for_TOV, add_crust

and the commit split it into three imports because `load_crust_table` had moved
to `crust.py` and `EOSTable_for_TOV` to `general/state.py` — a semantic rewrite
emitted with real newlines inside one JSON string element. Restored as two
properly escaped elements. The second was a missing trailing comma on
`from eos.astro.tov.solver import compute_tov_sequence, generate_ec_logspace`.

**One thing the ticket did not ask for, taken because it is the correctly-applied
rename rather than a change of behaviour**: `add_crust` now comes from
`eos.astro.tov.crust`, where it is defined, instead of
`eos.astro.tov.solver`, which merely re-exports it at `solver.py:20`. The broken
text named the re-export; both resolve, and all five names in the two repaired
cells were checked to import.

**0 added failures, structurally**: `grep -rn --include="*.py" "ipynb\|notebooks/" test/ eos/` returns
nothing, so no test and no package module reads a notebook, and the diff touches
one `.ipynb` and no importable code. A full-suite run was not the evidence here —
a parallel session is editing this checkout, which makes one worthless anyway.

**`notebooks/zlvmit_test.ipynb` was checked too and is valid JSON** — `d9f8eec`
touched three notebooks, and this is the one of the three that survived, so the
KEEP list is now entirely loadable.

Carried to the Stage 7 report unchanged: nothing in the repository checks that an
`.ipynb` is loadable, so this survived five days and four commits.

Status: resolved.
