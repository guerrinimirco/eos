# Apply the approved renames — eos/vmit

Type: task
Status: resolved
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


## Resolution

**23 renames across 24 files; 0 added failures; `test/baseline/` unmoved.**

All three groups applied. Rule 3's fourteen `compute_*` names lost the prefix,
the two records that repeated the package are `EoSPoint` and `MatterThermo`
(vmit defines its own, as `zl`, `alphabag`, `njl`, `ccdm` and `did` do, rather
than taking `eos.general.state`'s), and the seven vocabulary names landed:
`Parameters.default()`, the four §3-named solvers, `warm_start(point, mode)`
and `default_guess(mode, ...)`. The three frozen `compute_tables.py` symbols
were not touched. `DEFERRED.md`'s vmit entry now reads DONE and says what was
done; the stale "vmit and sfho have not" clause in the `dd2` entry above it
is corrected to name sfho alone.

**The two collisions the ticket predicted were both real, and one was NOT the
one the ticket named.**

- `eos/vmit/table.py:188` — as recorded: the local `def warm_start(point)`
  adapter would have called itself. It became `def seed(point)`, matching
  `eos/zl/table.py:165` and `eos/did/table.py:158`, which already use that
  name for the same one-argument closure.
- **`eos/vmit/solver.py`, four sites, not in the ticket**: each solver bound a
  LOCAL `default_guess = get_default_guess_<mode>(...)` before using it, so
  introducing a module-level `default_guess` would have made every one of them
  an `UnboundLocalError`. Loud rather than silent — unlike ticket 42's case —
  but the same shape. The locals are now `x0_default`.

The AST check found neither predictively, because it compares *imported*
against *defined* names and both of these are local bindings inside a function
body. It ran clean before and after; what it is good for is the cross-module
case, and it did confirm that.

**`_GUESS_KIND` went with the rename.** `table.py` carried a four-entry table
translating the §3 mode names into `result_to_guess`'s private strings
(`beta_eq`, `trapped_neutrinos`, ...). Once `warm_start` and `default_guess`
read the mode name itself, the table had nothing to translate.

**Bit-identity was proved, not assumed.** Merging four cold-guess functions
into one is the only part of this change that could move a number, so the four
originals were reconstructed and compared against the merged
`default_guess` across every mode x {6 densities} x {4 temperatures} x
{3 Y_C} x {leptons on/off}: **every returned array is bit-identical**. The
solvers therefore enter at exactly the same x0 and follow exactly the same
path.

**Five files needed an alias, for a reason worth recording.** vMIT's
`thermo_from_mu` / `thermo_from_n` / `thermo_from_mu_n` are the sixth model to
expose that vocabulary, and `eos/mixed/adapters.py` imports the same three
names from `enjl`, `njl`, `ccdm`, `zl` and `alphabag` inside individual
functions. A module-level unaliased vMIT import would have been shadowed
function by function -- working, but exactly the ambiguity ticket 42 was bitten
by. vMIT's are `_vmit_from_mu` / `_vmit_from_n` / `_vmit_from_mu_n` there,
beside the existing `_zl_from_mu` and `_ab_from_mu`; `eos/zlvmit` takes
`vmit_thermo_from_mu_n` beside its existing `zl_thermo_from_mu_n`, the
convention `DEFERRED.md` already describes for that package. The two dead
aliases in `mixed_phase_eos.py` (`get_vmit_guess`, `vmit_result_to_guess`,
imported and never called) were repointed rather than deleted -- deleting is
ticket 46's kind of decision, not a rename's.

`get_vmit_custom()` was left alone, as the ticket directs.

## Suite

**14 failed, 1634 passed, 15 skipped** (`pytest test/ -q`, 20:37). **All 14
are pre-existing**, verified by running the same tests against a detached
worktree at HEAD with the pre-rename `eos/` and the test files reverted to the
old call sites: the same 14 fail there, with byte-identical assertion messages
and identical mismatch magnitudes.

    8  test/baseline/  ccdm dd2 enjl njl sfho tov vmit zlvmit
    3  test/dd2/       test_api.py x2, test_dd2_m8.py
    3  test/tov/       test_solver_fast_robustness.py

Two root causes, neither in this diff:

- **dd2's NMP inversion no longer converges** (`inversion_failed`, isoscalar
  residual 8.12e-02 against a 2e-02 floor). This is the 6 `test/dd2` +
  `test/tov` failures AND the `dd2` baseline's `nmp.K_sat` / `nmp.Q_sat` /
  `nmp.K_sym` drift -- one defect, seven tests. Present at HEAD.
- **Baseline drift at the 1e-7..1e-10 level** in quantities the generator's own
  docstring calls round-off: `ccdm`'s `field_residual`, `sfho`'s `mu_S` at
  Y_S = 0, the tov sequences. vmit's is the sharpest illustration: `n_e` at
  Y_C = 0 sits at 1.7e-13 where the stored `.npz` (dated 10 Aug) has 3.0e-12,
  straddling the generator's 1e-12 gate, so `mu_e` is dropped from the fresh
  run and reported as "no longer produced". Both numbers are zero to any
  physics.

Neither is touched here -- they are Stage 7 report material. **The map's
"1648 passed, 15 skipped, 0 failed" line is stale.**
