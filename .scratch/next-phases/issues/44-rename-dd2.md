# Apply the approved renames — eos/dd2

Type: task
Status: resolved
Assignee: session cb0ab980 (also 43, 47)
Blocked by: 10, 42
Parent: ../map.md

## Question

The largest blast radius on the list, ~240 call sites, over half of them in
`test/`. Ticket 07 calls dd2 the second-worst package.

**Rule 2, the parameter object (3):**

    Parametrization                    -> Parameters       (52 eos / 144 test / 43 nb)
    Parametrization.from_dd2_defaults() -> Parameters.default()
    Parametrization.from_dd2y_defaults() -> Parameters.named("DD2Y")

`from_dd2_defaults` says "dd2" twice in `eos.dd2.Parametrization.from_dd2_defaults`.

**Rule 2, the solvers named after the §3 modes rather than after `octet` (6):**

    solve_octet          -> solve                            (35 eos / 66 test / 2 nb)
    solve_beta_eq_octet  -> solve_beta_eq_neutrinoless
    solve_fixed_yc_octet -> solve_fixed_yc
    solve_yl_octet       -> solve_beta_eq_neutrino_trapped
    sweep_octet          -> sweep
    sweep_beta_eq_octet  -> fold into sweep

**Rule 2, the warm start (2):**

    beta_warm_start  -> warm_start
    octet_warm_start -> warm_start

Two functions collapsing to one name: confirm they are the same job before
merging, and if they are not, the second keeps a name saying how it differs.

**RULED KEEP** (ticket 10 Q2): `solve_composition`, `solve_snm` and their `_t0`
twins are NOT renamed. They are not §3 modes — symmetric matter at saturation is
what `nmp.py` needs, not a mode a caller selects — so rule 2 does not bind them.

Already done: `notebook_api.py` deleted under [ticket 03](03-stage0-removals.md).

Resolved when dd2 is renamed and the added-failure count is reported. The §12
golden references bind hardest here: the DD2 golden SNM point at
n_B = 0.16 fm^-3, the published NMP/TOV values pinned in `dd2/verify` and
`test/dd2/`, and the CompOSE HS(DD2) slices. A rename moves NO number.

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

`dd2` has only `thermo_at_potentials` (`thermodynamics.py:571`); the rename is
one name, with no lower layer to re-spell.


## Resolution

**17 renames + 1 fold across 74 files; 0 added failures; `test/baseline/`
unmoved; every §12 golden reference intact.**

Suite: **14 failed, 1634 passed, 15 skipped** (20:04). The failure set is
BYTE-IDENTICAL to the pre-44 run — same 14 tests, same messages. All 14 are
the pre-existing set: 8 `test/baseline` and 6 from
[ticket 47](47-dd2-nmp-inversion.md).

Golden references (§12 binds hardest here, and a rename moves no number):

- `eos.dd2.verify` PASS — SNM(0.16) golden point at 1.40e-05, CompOSE HS(DD2)
  at 2.83e-05, backend parity at 4.40e-14.
- Ticket 47's NMP floor reproduces bit-identically: predicted-Q_sat offset
  51.5090, (240,300) not restart-recovered, (250,100,30) at isoscalar residual
  8.12e-02. Renaming `compute_nmp`/`invert_nmp` and everything around them
  moved not one digit of the one area that was already unstable.

### What was renamed

Beyond the ticket's list, on rulings taken during the work:

    Parametrization              -> Parameters
    .from_dd2_defaults()         -> Parameters.default()
    .from_dd2y_defaults()        -> Parameters.named("DD2Y")
    solve_octet                  -> solve
    solve_beta_eq_octet          -> solve_beta_eq_neutrinoless
    solve_fixed_yc_octet         -> solve_fixed_yc
    solve_yl_octet               -> solve_beta_eq_neutrino_trapped
    sweep_octet                  -> sweep
    sweep_beta_eq_octet          -> FOLDED into sweep
    solve_octet_at_entropy       -> solve_at_entropy
    octet_warm_start             -> warm_start
    beta_warm_start              -> nucleon_warm_start
    default_beta_guess           -> default_nucleon_guess
    default_octet_guess          -> default_guess
    assemble_octet               -> assemble
    _octet_x0                    -> _x0
    octet_residual               -> residual
    octet_jacobian               -> residual_jacobian
    octet_unknowns               -> n_unknowns

**The two warm starts did NOT merge, and the ticket was right to ask.** They
are different jobs: `beta_warm_start` returns a fixed 4-vector
[sigma, rho0, mu_eff_n, mu_C] for the nucleon-only `solve_beta_eq`;
`octet_warm_start` returns a variable-length vector through `_octet_x0`
carrying omega0, phi0, mu_S and mu_nue. The octet solver is dd2's main solver,
so it takes the §13 name; the reduced path became `nucleon_warm_start`, and
its sibling `default_beta_guess` became `default_nucleon_guess` so the pair
stays a pair.

**`sweep_beta_eq_octet` folded cleanly** because it was a pure pass-through:
its whole body was `return sweep_octet(...)` on that function's own default
arguments. Its docstring's one unique sentence (that the defaults ARE the
beta-equilibrium case) moved into `sweep`.

**The `octet` survey in the ticket's list was incomplete, and so was mine.**
The ticket named 6 solvers + 2 warm starts. A first sweep found 4 more
(`solve_octet_at_entropy`, `assemble_octet`, `default_octet_guess`,
`_octet_x0`). That sweep used a regex requiring a character BEFORE "octet", so
it missed the three names that start with it: `octet_residual` (20 sites),
`octet_jacobian` (18), `octet_unknowns` (5). All three were renamed to the
names `sfho` and `did` already use for the same jobs. The genuine physics uses
of the word — `HYPERONS_OCTET`, `BARYONS_OCTET`, the baryon octet in prose and
test names — are untouched, which is the correct line: `octet` is physics, not
a package name.

### The collision the ticket's own check cannot see

**The AST check ticket 42 prescribed missed a real, suite-breaking collision.
So did the extended version, on its first pass.** Three shapes exist and only
the first is checked:

1. **module import ∩ module def** — ticket 42's `mixed/api.py` bug.
   Found live here: `eos/mixed/solver.py` imported dd2's `solve_beta_eq_octet`
   at :66 and defines its OWN `solve_beta_eq_neutrinoless` at :706, so the
   rename put an import directly under a same-named def. The import turned out
   to be **dead** — one occurrence, never called — so it was deleted rather
   than aliased.
2. **an imported name rebound INSIDE a function** — nested def, assignment,
   for-target, `with ... as`, `except ... as`, or a parameter name. This is
   what bit [ticket 43](43-rename-vmit.md), whose four
   `default_guess = get_default_guess_<mode>(...)` locals the prescribed check
   cannot see because a local binding is neither an import nor a module-level
   def.
3. **the same name imported TWICE at module level from different modules.**
   The second silently wins.

Shape 3 broke three tests and is the one that matters going forward:

    test/mixed/test_hybrid_modes.py
      from eos.dd2.solver  import solve_fixed_yc, solve_beta_eq_neutrino_trapped
      from eos.vmit.solver import solve_fixed_yc, solve_beta_eq_neutrino_trapped

Ticket 43 gave vMIT the §3 mode names. Ticket 44 gave dd2 the SAME §3 mode
names. **This collision could not exist until both had landed**, which is why
neither ticket's own check would ever have found it, however carefully run.
It surfaced as `TypeError: got multiple values for argument 'T'` only because
the two signatures differ — with matching signatures it would have been silent
and wrong. Fixed by importing each under its phase (`dd2_fixed_yc` /
`vmit_fixed_yc`); a whole-tree shape-3 sweep found no other instance.

Also aliased three bare cross-model imports in `eos/mixed/adapters.py`
(`_vmit_beta`, `_vmit_yc`, `_vmit_yc_ys`, `_vmit_trapped`, `_did_beta`), where
five other models were already aliased by hand — that file had discovered this
hazard before anyone named it.

**The structural point, which outlives this ticket:** §13 makes every model
converge on one vocabulary, so every file importing two models becomes a
collision site. That is a cost of the rule, not a defect in it, but it means
**ticket 45 will be the third model onto the same words** and every
sfho-plus-anything importer is a candidate. Run shape 3 across the WHOLE tree
after 45, not just against sfho's new names: the collision is between two
OTHER packages as seen from a third file.

The three-shape checker is at
`scratchpad/shadowcheck.py`; it was regression-tested against ticket 43's four
known misses (catches all four) and against aliasing (`import X as f` beside a
bare `X` is correctly NOT reported). Known noise: a shape-2 hit on a function
*parameter* name is a real shadow but usually deliberate — read it, don't fix
it. The dangerous shape-2 hit is an assignment whose target is the imported
callable being called.

Four collisions remain flagged and all four are pre-existing, confirmed
against HEAD: two same-module redundant re-imports, one local `def
residual(mu_C)` in `mixed/boundaries.py`, one local `Parameters` re-import in
`test/mixed/test_scan.py`.

### Not done here

`test/` is gitignored (`.gitignore:75`), so the `test_hybrid_modes.py` fix
lives only in the working copy — the same hazard the map records for tickets
39, 40 and 56. Anyone reconstructing `test/` reintroduces the collision.
