# Apply the approved renames — eos/vmit

Type: task
Status: open
Blocked by: 10, 42
Parent: ../map.md

## Question

The worst package on the list. Ticket 07: **`eos/vmit` was never converted at
all below `Parameters`** — every §13 vocabulary name in it is wrong — while
`docs/DEFERRED.md:320` calls the conversion DONE. Fix that ledger line in the
same change.

**Rule 3, drop `compute_` where it carries nothing (14):**

    compute_quark_thermo              -> kinetic_thermo
    compute_quark_density             -> quark_density
    compute_vector_field              -> vector_field
    compute_vector_pressure           -> vector_pressure
    compute_vector_energy             -> vector_energy
    compute_bag_pressure              -> bag_pressure
    compute_bag_energy                -> bag_energy
    compute_mu_effective              -> effective_potential
    compute_effective_mu_quarks       -> effective_potentials
    compute_mu_physical               -> physical_potentials
    compute_quark_densities_for_solver -> effective_state
    compute_vmit_thermo_from_mu_n     -> thermo_from_mu_n
    compute_quark_matter_thermo_from_n -> thermo_from_n
    compute_quark_matter_thermo_from_mu -> thermo_from_mu

**Rule 1, the name repeats the package (2):**

    VMITEOSResult -> EoSPoint          (the record eight other models already use)
    VMITThermo    -> MatterThermo

**Rule 2, the shared vocabulary (7):**

    get_vmit_default()            -> Parameters.default()
    solve_vmit_beta_eq            -> solve_beta_eq_neutrinoless
    solve_vmit_fixed_yc           -> solve_fixed_yc
    solve_vmit_fixed_yc_ys        -> solve_fixed_yc_ys
    solve_vmit_trapped_neutrinos  -> solve_beta_eq_neutrino_trapped
    result_to_guess               -> warm_start
    get_default_guess_{beta_eq,fixed_yc,fixed_yc_ys,trapped_neutrinos}
                                  -> default_guess(mode, ...)

**FROZEN, do not rename** (ticket 10 Q4): `VMITTableSettings`,
`compute_vmit_table`, `save_vmit_results` in `compute_tables.py`. Their only
consumer is `notebooks/ZLvMIT_hybrid.ipynb`, which the map rules out of scope and
[ticket 41](41-corrupt-notebooks.md) records as unopenable.

NOT in this ticket: deleting `get_vmit_custom()` — [ticket 46](46-api-changes.md).

Resolved when vmit is renamed, `DEFERRED.md:320` no longer claims it was already
done, and the added-failure count is reported. `test/baseline/` must not move.

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

**Already located in vmit, by running that check predictively:**

    eos/vmit/table.py:188   def warm_start(point):
                                return result_to_guess(point, guess_kind, ...)

`result_to_guess` -> `warm_start` turns this into the identical silent
recursion. Rename the local adapter first, or inline it. The same file also
binds a local `solve` at :192, so re-run the check after the solver renames
land too.

Checked and CLEAN: `eos/dd2` and `eos/sfho` have no such collision for their
planned renames.

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
