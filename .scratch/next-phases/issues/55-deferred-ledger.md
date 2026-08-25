# Write the ten (c)-class conformance rows into docs/DEFERRED.md

Type: task
Status: resolved
Blocked by: 11
Parent: ../map.md

## Question

[Ticket 11](11-conformance-triage.md) ruled ten rows genuinely deferred. §3 and
§5 both require that a gap left open is *recorded*, and the audit's own note is
that `docs/DEFERRED.md` is unusually thorough — these are the ones missing from
it. Each entry carries the reasoning and, where it was measured, the measurement;
evidence is in [conformance-table.md](../research/conformance-table.md).

1. **`astro/gmode` imports model internals** (finding 2). The short-term entry
   while [ticket 53](53-gmode-contract.md) designs the contract. Recorded nowhere
   today — `grep -n -i gmode docs/DEFERRED.md` returns only the unrelated
   "expensive, not hung" line at `:101`. Name the four import sites and say the
   contract is being designed, not that the breach is accepted.

2. **Eleven needless *downward* deferred imports** (finding 10b), where no cycle
   exists and a top-level import would work: `sfho/nmp.py:66,67,138`,
   `did/nmp.py:162`, `sfho/thermodynamics.py:583`, `njl/thermodynamics.py:479`,
   `ccdm/thermodynamics.py:557`, and `api.py -> responses.py` in `sfho:193`,
   `did:145`, `njl:174`, `ccdm:197`, `dd2:177`. Style drift, not cycles.

3. **Three physics-bearing module constants with no override path**
   (finding 17b): `mixed/adapters.py:314,319 _CHIRAL_SPLIT = 50.0` and
   `_DECONFINED_BARYON_FRACTION = 1.0e-4`, which decide which *phase label* a
   point gets; `astro/tov/rns_backend.py:94-95 RNS_RHO_SURFACE`, `RNS_P_SURFACE`,
   which mirror the C source; and `zlvmit/hybrid_table_generator.py:46 B4`, in an
   exempt package. Deferred rather than fixed because each is a threshold whose
   change moves a classification, not a coupling an inference run varies.

4. **`general/fermi_integrals.py:519,524` is the one unbounded loop**
   (finding 20). §6 requires a bounded iteration count. The fix rides
   [ticket 52](52-general-t0-integrals.md); the ledger line records that it was
   the only one, and that geometric growth made a hang practically unreachable.

5. **`sfho` and `zl` parameter dataclasses are not frozen** (finding 21). Both
   pickle, so multiprocessing works today. `sfho/parameters.py:68` carries
   `a_coeffs: np.ndarray` (`:152`) and a `couplings_map` written after
   construction at `:373,389,400,425,441,452,480`; `zl/parameters.py:27` is
   benign (all `str`/`float`) but unhashable, so it cannot key the read-only
   caches §6 permits. Deferred because the sfho fix is a builder pattern that has
   to become `replace()` or a real constructor, not a decorator change.
   `eos/enjl` shows the target shape: frozen dataclass with `_vacuum_cache` keyed
   on it (`enjl/thermodynamics.py:404-430`).

6. **`eos/astro/tov` has no `verify/` at all** (finding 30) — the unit holding
   the integrator, tidal deformability, crust handling and the RNS backend. §12
   pins "the DD2 published NMP/**TOV** values" as golden, and **`test/` is
   gitignored**, so a fresh clone has no way to check TOV. The two ledgered
   astro/tov gaps (the ~2 % two-backend Lambda disagreement, the non-monotone
   table fragility) have no suite to run from.

7. **`eos/abpr` refuses all four §3 modes and nothing records it**
   (finding 32). `abpr/solver.py:47-49 MODE_FRACTIONS = {"cfl": ()}`,
   `:54-73 MODE_REFUSALS`, `:84` raises. §3 requires the gap be recorded. The
   refusal messages are good and worth quoting — the `fixed_YC_YS` one explains
   that locked matter has `Y_C = 0`, `Y_S = +1` identically, so the
   symmetric-matter slice the mode exists for is not a state it can reach. The
   `abpr` section (`:1011-1045`) records four other limits but never this.

8. **`eos/mixed`'s `Y_Lmu` refusal** (finding 33). `mixed/api.py:81-84`. Nine
   other models have a ledger bullet for the identical gap; `mixed` does not.

9. **`VMITTableSettings`** (finding 35). `vmit/compute_tables.py:41`, plus
   `compute_tables.py` being a second table driver beside `vmit/table.py` and
   outside §5's template. [Ticket 43](43-rename-vmit.md) renamed the other two
   `VMIT*` classes; this one is **frozen deliberately** by
   [ticket 10](10-rename-approvals.md) because its only consumer is the
   out-of-scope ZLvMIT notebook. Record that it is frozen and why, so the next
   reader does not think it was missed.

10. **`output/public/` does not exist** (finding 37). `.gitignore:35-39` reserves
    it correctly (`output/`, `!output/public/`, `!output/public/**`) and §11
    describes it as an existing tracked folder. `ls output/` shows nine entries,
    none of them `public/`. A ledger line now; the `mkdir` and the curation wait
    on the notebooks producing tables.

**Also correct one existing entry:** `docs/DEFERRED.md:328` asserts "so every
model's parameter dataclass is `Parameters`". `eos/dd2/parameters.py:35` is
`class Parametrization` and the file contains no `Parameters` at all. The vmit
half of that sentence was corrected by ticket 10; the dd2 half is
[ticket 44](44-rename-dd2.md)'s to complete, so this ticket fixes the sentence
and points at 44 rather than pre-announcing the rename as done.

Documentation only. No `eos/` or `test/` file is touched, so the suite cannot
move; report that it did not.

## Resolution

**All ten rows written, plus the correction — `docs/DEFERRED.md` +206/−2, and no
`eos/` or `test/` file touched, so the suite cannot have moved and did not.**
Placed by shape rather than by row number: six are cross-cutting `###` entries
appended to that section, four are bullets in the per-unit section they belong to
(`general`, `mixed`, `abpr`, `vmit`, and `astro/tov`).

**Every row was re-measured against the code before it was written, and three had
moved since the audit.**

- **Row 1's fourth import site is not the one the ticket names.** `eos/astro/
  gmode/verify/run_full_check.py:39-41` now reads `from eos.dd2 import Parameters,
  SpeciesFlags` / `responses.sound_speed_eq` / `solver.sweep,
  solve_beta_eq_neutrinoless` — the audit's `Parametrization` and
  `sweep_beta_eq_octet` are both names [ticket 44](44-rename-dd2.md) retired. The
  breach is unchanged in substance and the entry states the current symbols.

- **Row 2 is ten sites, not eleven.** `eos/dd2/api.py`'s `responses as _fd` is
  gone: dd2's `api.py` now defers only `backends/responses_jac` (`:166`) and
  `responses` (`:178`), both inside the branch that selects the analytic
  Jacobian — an optional-backend deferral, which is not the drift this row is
  about. The other ten are live at the lines the audit gave, with `sfho/
  parameters.py`'s two shifted by one.

- **The correction the ticket asks for had already come true.** `DEFERRED.md:328`
  asserted "every model's parameter dataclass is `Parameters`" while
  `eos/dd2/parameters.py` held `class Parametrization`. Ticket 44 has since
  landed, so the sentence is now TRUE of all ten models —
  `grep -c '^class Parameters' eos/*/parameters.py` returns ten hits, verified.
  It is corrected to say so, with the history (sfho and vmit first, dd2 last) and
  a checkable grep, rather than being deleted as false or left implying dd2 was
  never the exception. The same paragraph's "sfho has not" is left standing and
  sharpened: sfho's CLASS converted, its five `get_sfho*` CONSTRUCTORS have not,
  and that half is [ticket 45](45-rename-sfho.md)'s.

**One row gained a measurement it did not have.** Row 4 called
`general/fermi_integrals.py:518,524` the one unbounded loop; the entry now also
records WHY it has never been observed to bite — the bracket expansion is
geometric (`mu_hi *= 1.5`, `mu_lo *= 0.5`), so it clears any representable target
in a few dozen iterations, and the shape that would actually spin is a non-finite
`n_target`. That is what makes a counter returning non-convergence the right fix
rather than a large bound.

**Noticed and NOT fixed, per the map's hard rule.** `### astro/tov` still carries
"Crust table paths are absolute and machine-specific. A missing crust file
currently degrades to no crust" — closed by [ticket 39](39-crust-silent-fallback.md),
which shipped the tables in `eos/astro/tov/data/` and made the helpers skip. Stage
7 report material, not this ticket's diff.
