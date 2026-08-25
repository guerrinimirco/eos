# Apply the approved renames — eos/sfho

Type: task
Status: resolved
Assignee: session 32a0f093
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

## Added by ticket 36 (quark-engine documents)

**The phase-adapter surface is named, and this package is one of the three that
must follow.** [Ticket 10](10-rename-approvals.md) deferred
`thermo_at_potentials` vs `thermo_from_mu` to `mixed.tex`, which has now ruled:

- the §5 contract surface — `(baryon potential, mu_C, mu_S, T) -> PhaseThermo`,
  solving the phase's own self-consistency — is **`thermo_from_mu`**, in every
  model;
- a lower evaluation layer that additionally takes the solved mean fields is
  **`thermo_from_fields`**, because the name should say what the function takes,
  and that is the distinction between the two layers.

7 of the 10 models already spell the surface `thermo_from_mu`. Apply the pair of
renames here, with the AST collision check tickets 43-45 already carry: this is
exactly the shape that bit ticket 42 (`mixed/api.py`'s local `solve`) and
ticket 43 (`vmit/table.py`'s local `warm_start`) — a rename ONTO a name the
package already uses at another layer.

`sfho` carries BOTH: `thermo_at_potentials` on the surface and
`thermo_from_mu(mu_B, mu_C, mu_S, sigma, omega, rho, phi, T, ...)` beneath it
(`thermodynamics.py:421`). Rename the lower one to `thermo_from_fields` FIRST,
or the second rename lands on an occupied name.

## Resolution

**8 renames across 15 files; **14 failed, 1634 passed, 15 skipped** — the identical failure set, every assertion message byte-identical; `test/baseline/` unmoved; sfho's
`verify/` suite green on all eight invariants.**

    thermo_from_mu        -> thermo_from_fields   (the lower layer, moved FIRST)
    thermo_at_potentials  -> thermo_from_mu       (the section-5 adapter surface)
    get_sfho_nucleonic    -> Parameters.default()
    get_sfhoy_fortin      -> Parameters.named("SFHoY_Fortin")
    get_sfhoy_star_fortin -> Parameters.named("SFHoY*_Fortin")
    get_sfho_2fam_phi     -> Parameters.named("SFHo_2fam_phi")
    get_sfho_2fam         -> Parameters.named("SFHo_2fam")
    get_all_parametrizations() -> PUBLISHED_SETS

### The `named()` keys, and why they are the `name` field

The ticket required settling them because they become public API. They are the
five strings each set ALREADY carried in its `name` field, unchanged, so
`Parameters.named(p.name)` round-trips and no stored string moves:
`SFHo_Nucleonic`, `SFHoY_Fortin`, `SFHoY*_Fortin`, `SFHo_2fam_phi`,
`SFHo_2fam`. `default()` is `SFHo_Nucleonic` — the nucleon-only CompOSE table,
which is what `nmp.py` reports the published NMPs against and what
`test/baseline` is frozen at, matching dd2's `default()` being its nucleon set.

**A third spelling of the same five existed and is now an alias table, not a
registry.** `table.py:461` carried `{'sfho', 'sfhoy', 'sfhoy_star', '2fam_phi',
'2fam'}` mapped to the five builders — the legacy `TableSettings` string. It
now maps those short strings to the published names and defers to
`Parameters.named`, so there is one registry and one set of keys.

**`PUBLISHED_SETS` holds BUILDERS, not instances**, and that is a section-6
requirement rather than a style choice: a `Parameters` carries mutable
`couplings_map` dicts, and the five builders mutate them (`_two_family` calls
`_two_family_phi` and zeroes `g_phi`). A module-level dict of instances would
be exactly the global mutable state section 6 forbids, and the old
`get_all_parametrizations()` avoided it only by rebuilding on every call.
Builders keep that property while making the registry a constant. Exported from
`eos/sfho/__init__.py` beside `Parameters`, matching `enjl` and `njl`.

### The collision check earned itself a fourth time

Shape 2, and introduced BY the rename rather than found in it.
`test/mixed/test_phase_pairs.py:111` had a function-local
`from eos.sfho.parameters import get_sfho_2fam_phi`; converting it to
`Parameters` put it directly under the module-level
`from eos.dd2 import Parameters` at :18. Scoped, so it would never have raised
and never have been wrong — but it is the same shape that broke tickets 42, 43
and 44, and the file's own next line already aliases
(`SpeciesFlags as SFHoFlags`). Aliased to `SFHoParameters`. The tree is back to
the two pre-existing hits ticket 44 flagged (`test/mixed/test_scan.py:212,225`).

**Shape 3 across the whole tree is clean**, which was the specific worry ticket
44 left: sfho already carried the section-3 mode names before this ticket, so it
did not become a third package converging on those words and no
`solve_fixed_yc`-style double import appeared.

### `_get_base_sfho` deliberately not renamed

It repeats its package and keeps a `get_` that carries nothing, so rules 1 and
3 both bite. It is not in the ticket's list, it is private, and its one
cross-module caller is `nmp.py:294` inside `create_custom_parametrization` —
which [ticket 46](46-api-changes.md) turns into a `from_*` constructor and moves.
Renaming it here would be churn that ticket 46 rewrites. Reported, not fixed.

### Evidence

- **Bit-identical, 13 probe points**: three parameter sets x
  {beta_eq_neutrinoless, fixed_YC} x {T = 0, 20 MeV}, plus `fixed_YC_YS` and a
  direct call on the renamed adapter surface. Same floats to the last digit
  against the pre-rename tree. A rename that changes a number is not a rename.
- **`eos/sfho/verify/run_full_check.py` PASS**, all eight: Euler/HVH 9.11e-10,
  published NMPs 2.01e-03, E_sym curvature 4.69e-04, causality and monotone P
  0.00e+00, CompOSE HS(SFHo) 8.37e-04, Jacobian vs FD 5.00e-08, chi_ab 1.52e-05,
  NMP forward/inverse 8.54e-08.
- **`test_baseline[sfho]`'s failure body is byte-identical** to the before-run —
  same 13 quantities, same `ycys.n0.16.matter.mu_S` at rel. 2.210e-07. That row
  is [ticket 56](56-baseline-empty-sector-gate.md)'s and this ticket did not
  touch it.
- Before: `output/_audit/pytest_before_ticket45.txt`, after:
  `output/_audit/pytest_after_ticket45.txt`.

### One measurement thrown away

The first post-rename suite run was killed at 17% and discarded: I had edited
`eos/sfho/__init__.py` and re-indented `thermodynamics.py` after it started, so
it spanned a source edit — the same contamination the map records for an earlier
run. The reported numbers come from a run on a frozen tree, with the second edit
being whitespace and a docstring only. Cheaper to pay twenty minutes than to
report a number I would have had to qualify.

### Reported, not fixed

- **dd2 never took its half of ticket 36's ruling.** Ticket 44 carried the same
  "Added by ticket 36" instruction and its 19 renames do not include it:
  `eos/dd2/thermodynamics.py:571` is still `thermo_at_potentials`, with no
  `thermo_from_mu` anywhere in `eos/dd2`. Confirmed independently by the session
  working ticket 35. Widened [ticket 48](48-rename-did-surface.md) to carry dd2
  beside `did`, with `mixed/adapters.py:52`'s BARE module-level import called
  out — once dd2's surface is `thermo_from_mu`, that file holds a module-level
  `thermo_from_mu` beside two function-local aliased imports of the same name,
  which is shape 3.
- **`nucleation` cannot import `eos` today.** Measured by importing every target
  it names: five modules gone, two more missing the name. All Phase 3/4 module
  MOVES, none of them a Phase 5 rename. Written into
  [ticket 23](23-phase6-respec.md), which had recorded one of the seven.
- `eos/sfho/parameters.py`'s module docstring still opens `sfho_parameters.py`,
  a filename that has not existed since the reshape.
- `eos/vmit/verify/run_full_check.py` runs eight checks and its module docstring
  enumerates seven — "bag / vector signs" is real, passes, and is missing from
  the numbered list. Found by the session working ticket 33.
