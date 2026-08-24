# Where do the models deviate from §13's names, order and docstrings?

Type: research
Status: resolved
Parent: ../map.md

## Question

Stage 5, read-only. Across every model:

- **Docstrings in `thermodynamics.py`.** Every function must state the explicit
  quantity it returns — the formula, in the notation of the model's document,
  with units. `"""Kinetic pressure of one species."""` fails; the closed-form
  expression with its integral passes. List every failing docstring, file:line.
- **Names (§13).** A name never repeats its package; the same job carries the
  same name in every model (`kinetic_thermo`, `mean_fields`, `thermo_from_mu`,
  `thermo_from_n`, `assemble`, `residual`, `default_guess`, `warm_start`,
  `solve_<mode>`, `build_table`, `compute_nmp`/`invert_nmp`, `Parameters`,
  `Parameters.default()`, `Parameters.named()`); a name says what the function
  takes and returns. Report every deviation with file and line. **Propose renames
  as a list — rename nothing.**
- **Order.** Functions ordered by the physics, not alphabetically and not by call
  depth: `thermodynamics.py` reads single species → mean fields → per-species
  loop → sums; `solver.py` reads guesses → residual → solve → modes → sweep.
  Report files where the reading order is wrong. Do not reorder.
- **Self-contained docstrings.** A docstring may not reference a plan, a phase, a
  milestone number or a `docs/` working note. Grep and list them.
- **Physics visible over clever.** Flag dense comprehensions and nested
  expressions where a named intermediate would show the equation.

Check these suspects explicitly: `eos/vmit/compute_tables.py` (name is not in the
§5 layout), `eos/dd2/notebook_api.py` (§11 forbids it), `eos/abpr/` (no
`table.py` — genuine absence of that physics, or a gap?), `eos/zl/` (no
`nmp.py` — the prompt says state that rather than fake it).

Write to `.scratch/next-phases/research/naming-sweep.md`.

## Answer

Full report: [naming-sweep.md](../research/naming-sweep.md). Read-only — nothing
renamed, reordered or edited.

| category | findings |
|---|---|
| `thermodynamics.py` docstrings failing the formula test | **56** functions (+6 dataclasses restating their own name) |
| §13 name deviations | **74**, of which **58 are PUBLIC**, plus 4 §5 file-name deviations |
| wrong reading order | **6** files, 2 of them serious |
| plan / phase / milestone / working-note references in docstrings | **8** |
| dense comprehensions hiding an equation | **9** |

**Nine models are close to the standard; two are not.** `eos/vmit` was never
converted below `Parameters` — every §13 vocabulary name in it is wrong, and
`docs/DEFERRED.md:320` calls that conversion "DONE", so the ledger is stale.
`eos/dd2` is second: `Parametrization` rather than `Parameters`, six solver entry
points named after `octet` rather than the §3 modes, and the worst docstrings in
the repo — including §13's own failing example verbatim at
`eos/dd2/thermodynamics.py:79` and `:88`. `zl`, `abpr` and `enjl`
thermodynamics pass every check and are the rewrite target.

Serious order faults: `eos/dd2/solver.py` (four solves before any guess, arranged
as two sector blocks so guesses→residual→solve is read twice) and
`eos/sfho/thermodynamics.py` (the per-species loop precedes the mean fields, and
no single-species function exists — sfho is the only model with no
`kinetic_thermo`).

**58 public renames need approval**, grouped: dd2 13 (~240 sites, over half in
`test/`), vmit 26 (~250 sites), sfho 9 (~60 sites), mixed 11, did 1, plus one
cross-model decision — `thermo_at_potentials` (dd2, sfho, did) against
`thermo_from_mu` (seven models), where dd2 and did carry both at two layers, so
the question is what the *upper* layer is called, not a blind rename. These go to
the [rename-approval ticket](10-rename-approvals.md), now unblocked.

Verified: `nucleation` on `paper-release` touches none of these — grep for
`get_sfho|Parametrization|solve_vmit|get_vmit` across its tree returns nothing.

**The four suspects:**

1. **`eos/vmit/compute_tables.py`** — a 203-line legacy settings-object shim over
   `table.build_table`; nothing is solved in it. Imported *only* by
   `notebooks/ZLvMIT_hybrid.ipynb`, which the map puts out of scope; nothing in
   `eos/` or `test/` imports it. `DEFERRED.md` already rules it the one
   deliberate exception, so **keep the module**. But that ruling does not cover
   its three *symbols* — `VMITTableSettings`, `compute_vmit_table`,
   `save_vmit_results` each repeat the package (§13 rule 1) and zl's equivalents
   were converted. Rename the symbols or record them as frozen.
2. **`eos/dd2/notebook_api.py`** — forbidden by §11 and dead inside the package.
   Complete importer list: `notebooks/DD2_usage.py:76` and
   `notebooks/DD2_usage.ipynb:79`. Not in `__all__`; no `eos/` module and no test
   imports it. It also prints on self-check (`:578`) and imports `astro`, a §1
   violation. **Delete with the notebook** — [ticket 03](03-stage0-removals.md)
   already owns it, no blocker.
3. **`eos/abpr` has no `table.py`** — a real but mild §5 gap. abpr *does* produce
   an `eos_table` (`api.py:146`) with the full §5 progress dictionary, plus
   `TableResult:112` and `cfl_row:127` — the trio every other model keeps in
   `table.py`. §5: "where it has one of these parts, that part has this name."
   What abpr genuinely lacks is only the warm-started sweep, and its docstring
   argues that absence correctly (closed-form density inverse). Move ~90 lines to
   `eos/abpr/table.py`; public API unchanged. Secondary: `response_at_mu` sits in
   `solver.py:350`, which §5 reserves for equilibrium conditions and their solves.
4. **`eos/zl` has no `nmp.py`** — **a real gap, not an absence of physics**, and
   it contradicts the assumption written into
   [ticket 12](12-hadronic-skeleton.md). Graduated to
   [ticket 26](26-zl-nmp.md).

Status: resolved.
