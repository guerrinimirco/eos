# ZLvMIT_hybrid.ipynb is corrupt JSON and cannot be opened

Type: task
Status: open
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
