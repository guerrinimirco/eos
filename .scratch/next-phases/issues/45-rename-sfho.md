# Apply the approved renames — eos/sfho

Type: task
Status: open
Blocked by: 10, 42
Parent: ../map.md

## Question

~60 call sites. sfho has no `Parameters.default()` / `.named()` pair at all —
it carries five `get_sfho*` free functions instead, which is the same job under
five names none of which is §13's.

**Rule 2 (7):**

    add Parameters.default() and Parameters.named(name)     (40 eos / 15 test)
    get_sfho_nucleonic       -> Parameters.named("nucleonic")
    get_sfhoy_fortin         -> Parameters.named(...)
    get_sfhoy_star_fortin    -> Parameters.named(...)
    get_sfho_2fam_phi        -> Parameters.named(...)
    get_sfho_2fam            -> Parameters.named(...)
    get_all_parametrizations() -> PUBLISHED_SETS

Settle the exact `named()` keys while doing it: they become public API and the
model document has to state them.

NOT in this ticket — [ticket 46](46-api-changes.md): `get_sfho_general(...)` and
`create_custom_parametrization(...)` becoming `from_*` constructors, and the
isentropic solvers folding into `SnB=`.

Resolved when sfho is renamed and the added-failure count is reported.

## Warning from ticket 42 (the rehearsal), binding on this ticket

**A rename onto a §13 vocabulary name can collide with a local adapter already
using that name, and the failure is SILENT.** Found the hard way in
[ticket 42](42-rename-internal.md): `eos/mixed/api.py` imported `solve_mixed`
and separately defined a nested `def solve(temperature)` adapter. Renaming
`solve_mixed` -> `solve` made that function call itself — and because
`RecursionError` subclasses `RuntimeError`, the surrounding
`except (RuntimeError, ValueError)` swallowed it into a returned
"did not converge" rather than a crash. Twelve tests failed with no traceback
pointing at the cause.

The pattern is systematic, not bad luck: this codebase already used §13's
vocabulary for *local adapters* (`solve`, `warm_start`, `sweep`), which is
exactly what the public names are being renamed TO.

**Run this before renaming, and again after:**

    python3 - <<'PY'
    import ast, pathlib
    NEW = {...the names this ticket introduces...}
    for f in list(pathlib.Path("eos").rglob("*.py")) + list(pathlib.Path("test").rglob("*.py")):
        try: tree = ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError: continue
        imported = {(a.asname or a.name) for n in ast.walk(tree)
                    if isinstance(n, ast.ImportFrom) for a in n.names}
        defined  = {n.name for n in ast.walk(tree)
                    if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
        if (imported & defined) & NEW: print(f, sorted((imported & defined) & NEW))
    PY

A hit means one of the two has to be renamed. In ticket 42 the local adapter
was the one that moved (`solve` -> `point_at`), since the public vocabulary name
is the one §13 fixes.

## Do NOT run 43, 44 and 45 concurrently

Measured, not assumed. The three rename sets touch overlapping files, so two
sessions in one checkout will corrupt each other:

    44 (dd2) n 43 (vmit)   15 files   incl. eos/mixed/adapters.py, hybrid.py,
                                      responses.py, solver.py, table.py and
                                      9 test/mixed files
    44 (dd2) n 45 (sfho)    3 files   incl. eos/sfho/parameters.py
    43 (vmit) n 45 (sfho)   2 files   incl. eos/sfho/solver.py

All three also rewrite `test/baseline/generate_baseline.py`. Run them one at a
time. The document tickets (30, 31, 32, 33, 35, 36) ARE disjoint from these and
from each other, and need no pytest run, so they parallelise freely — their only
shared file is `docs/eos.bib`, which is append-only.
