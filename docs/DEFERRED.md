# DEFERRED — known gaps, per model

The tracked ledger CLAUDE.md refers to: modes a model does not support, physics
not yet wired, and behaviour that is understood but not yet fixed. A gap
recorded here is a decision; a gap not recorded here is a bug.

Each entry says what the gap is, how it shows up, and what closing it would
take. Entries are removed when closed, not marked "done".

---

## Cross-cutting

### A potential is only pinned as tightly as its conjugate density responds

**Models:** sfho and vmit (observed), and any model exposing `fixed_YC_YS` or
a charge-neutral mode at Y_C = 0.

Two cases, one cause. A chemical potential is fixed by the residual only
through the density it conjugates, so where that density is zero the solver
has little or nothing to go on.

In `fixed_YC_YS` with Y_S = 0 — symmetric nuclear matter, the heavy-ion slice —
no strange species is thermally populated at the densities and temperatures
tested (Lambda ~ 1e-16 fm^-3, Xi ~ 1e-32 fm^-3, n_S = 0 exactly). The
strangeness constraint n_S = n_B Y_S is then satisfied for a whole range of
mu_S: the residual has no gradient in that direction and the Jacobian is
singular there. The solver converges — every other quantity is determined and
reproducible to the last digit — but reports whichever mu_S its path happened
to reach. Recompiling the Numba integral kernels moves it by ~10 MeV.

Seen at n_B = 0.16, 0.32, 0.64 fm^-3, T = 10 MeV, SFHo-Y (Fortin) with the
full baryon octet. `eps`, `P`, `mu_B` and every density are unaffected.

**mu_e where no electrons are present** — a charge-neutral phase at Y_C = 0.
Here mu_e = 0 IS the answer, so it is determined rather than free, but only
weakly: dn_e/dmu_e is of order T^2/(hbar c)^3, about 4e-6 fm^-3 MeV^-1 at
T = 10 MeV, so a residual gate at 1e-10 on the density leaves mu_e loose at
the 1e-5 MeV level. Its landing point is round-off. Seen in vmit, dd2, sfho,
zl and alphabag, in every `fixed_YC` slice at Y_C = 0.

Closing either means deciding what the API should say when a conserved charge
is carried by no populated species. Options: report the potential as undefined
(NaN) with a status flag; pin it by convention (mu = 0) and document that; or
have the mode raise. Until then `test/baseline` does not freeze mu_S where n_S
is zero, or mu_e where n_e is zero, because there is nothing there to freeze.

### The pure-Python integral fallback is not a bit-exact reference

`general/fermi_integrals.py` and `general/bose_integrals.py` define their
kernels under `@njit(fastmath=True, cache=True)`, with a pure-Python fallback
used when Numba is absent (or `NUMBA_DISABLE_JIT=1`). The two paths agree to
about 1e-7 relative, not to machine precision, because `fastmath` lets the
compiler reassociate floating-point operations.

That is expected rather than wrong, but it means the fallback is not the
"reference flavour" in the sense of CLAUDE.md §9 — it is a second
implementation with its own error, and no parity test currently pins the two
together. Worth deciding during the `general/` refactor whether to add a
parity check at a documented tolerance, or to drop `fastmath` on the kernels
where the speed gain does not justify it.

### ASY-EOS band columns are headed "low"/"up" but are stiff/soft edges

`plot/data/samples/ASYEOS_2016_Esym.txt` and the CSV derived from it carry the
header `rho_fm3 Esym_low_MeV Esym_up_MeV`, but the two curves cross at
saturation: below n_0 the "low" column is the larger of the two, above it the
smaller. That is the physics — the constraint bounds the SLOPE of E_sym, so
the band is pinned where E_sym is already known and fans out either side — but
the column names invite exactly the wrong fix, which is to sort them.

Drawing is unaffected (`fill_between` fills between two curves in any order)
and both the crossing and the pivot are pinned by tests. What is left is to
rename the columns to something like `Esym_stiff` / `Esym_soft` in SOURCES.md
and the converter, so the file stops implying an ordering it does not have.

### The CompOSE reader cycle is CLOSED

`eos/sfho/compose_loader.py` imported `EOSTable_for_TOV` from
`eos.astro.tov.solver` while that module imported `SFHOComposeLookup` back
from it -- a genuine cycle, worked around with a lazy import inside the
function. `eos/dd2/verify/compose.py` imported the same module, so a model
depended on another model.

Done as designed, in the `astro/tov` session where the crust path has test
cover. `eos/sfho/compose_loader.py` is now `eos/general/compose.py`; it
imports nothing else in the repository, and `to_crust_table` -- which had to
know about `EOSTable_for_TOV` -- is `slice_arrays`, returning plain
(P, epsilon, n_B). `eos.astro.tov.solver` wraps those into the table, at a
normal top-level import, because astro consumes what general produces. The
class is `ComposeLookup`: nothing in it was ever SFHo-specific, SFHo is just
the table this repository points it at, and a name saying otherwise inside
`general/` was misleading.

The `dd2 -> sfho` edge is gone with it; `dd2/verify/compose.py` and
`sfho/verify/compose.py` both read from `general/`. The last dd2 -> sfho
import was `dd2/notebook_api.py` reaching for `eos.sfho.table`; that file has
now been deleted, so no model-to-model edge remains.

### astro/gmode is expensive, not hung

Worth recording because it was misdiagnosed once in this repository's own
notes. `test/gmode` does not hang. It is slow, and the cost is the
crust-attached background. Measured:

    no crust data, -m "not slow"      69 s      29 passed, 5 skipped
    crust configured, -m "not slow"   17.5 min  34 passed
    one `slow` test on its own        3.6 min

`pyproject.toml` already declares the convention --
`markers = ["slow: long-running (excluded with -m 'not slow')"]` -- so a gate
should use it. A run that looks stuck after a dozen dots is a `slow` test
doing a complex frequency scan, and a run that looked stuck BEFORE the
astro/tov session was the crust being found through a hardcoded path and the
full background being built.

The five crust-dependent tests now skip when no crust table is configured,
rather than erroring, for the same reason the rotating ones do: a fresh clone
carries no external data and must still produce a green suite.

### The astro imports are resolved: the rule is tightened, `eos/mixed` named

CLAUDE.md section 1 said no model imports `astro/`, and five files did. They
are gone, and the rule now says what it means rather than leaving the reader
to guess whether a `verify/` suite counted.

What each of them actually was, which is what made the answer easy: two
separable things in one file. `build_core_table` (dd2 and did alike) turns a
cold beta-equilibrium sweep into an `EOSTable_for_TOV` and imports NOTHING
from astro -- it is the model's half of the contract, and that record lives in
`eos.general.state` precisely so both layers may hold it. `mass_radius` runs
the sequence. The first moved to the model's own `table.py`; the second moved
to `test/<model>/`:

    eos/dd2/verify/tov.py   ->  build_core_table, N_TRANSITION in
                                eos/dd2/table.py
                                mass_radius in test/dd2/dd2_tov_sequence.py,
                                the M_max >= 2 check now
                                test/dd2/test_dd2_m10.py
    eos/did/verify/tov.py   ->  the same split, and the Table VIII comparison
                                (DID 2.245 / 11.99, DIDY 2.196 / 11.99) is now
                                test/did/test_did_tov.py, marked `slow`
    eos/dd2/notebook_api.py     imported astro; deleted, so the edge is gone

`eos/mixed/hybrid.py` KEEPS its import, as the one named exception in
section 1. `mixed` is a composite engine, not a model: it sits directly below
`astro/` in the layering order and couples to nothing else downstream, and
`hybrid.py`'s result columns ARE M_max and R_1.4 -- taking TOV out of it
removes the point of the hybrid table rather than fixing a layering problem.
(`eos/mixed/scan.py` stood beside it here until the file was removed; the
exception now names one file.)

`test/test_imports.py` enforces the tightened rule with
`test_no_model_imports_astro`, which has no `verify/` carve-out. The blanket
`verify/` exemption survives only for the model-to-model half (enjl's verify
suite reaches into `eos.mixed` to check its own coexistences), and the stale
`sfho/compose_loader.py` entry is gone with the file it named.

What the models lost: `eos.dd2.verify.run_full_check` no longer reports
`TOV M_max>=2` and `eos.did.verify.run_all` no longer takes `include_tov`.
Both checks still run, as tests. That is the cost of the rule and it was taken
deliberately: `test/` is gitignored and unpublished, so the DID
paper-reproduction check is no longer inside the distributed package.

### Three notebook import lines are dead, one per rename that passed them by

RESOLVED by deletion, not by fixing. The renames of this session updated every
call site in `eos/` and `test/` and deliberately did not touch the notebooks;
three import lines were left naming modules that no longer exist
(`notebooks/DD2vMIT_general1oPT` reaching for `eos.vmit.eos` and
`eos.dd2.verify.tov`, `notebooks/DID_usage` for `eos.did.verify.tov`). Both
notebooks have since been removed under Stage 0, so no dead import survives.
The replacement notebooks are written against the public API and cannot
reproduce this class of breakage: they import `eos_point` / `eos_table` /
`eos_response`, never a model's internals.

### One notebook still sets rcParams

RESOLVED by deletion. `notebooks/ENJL_usage.py` set `figure.dpi` and
`figure.figsize` directly rather than going through
`eos.general.figure_style`; the file has since been removed under Stage 0.
Every module in `eos/` and in `nucleation/` goes through `figure_style`, and
CLAUDE.md section 10 binds the replacement notebooks to the same rule.

### The shared records are adopted by dd2 and sfho; the other models are not

`eos/general/state.py` holds the records every model is meant to hand back --
`PhaseThermo` (matter only: the model's own fields under its own names, EVERY
active species' density, mu_i, mu_eff_i and m_eff_i per species, the conserved
charges, P, eps, s), `LeptonThermo` (all four potentials explicit, so a
transparent muon family is a visible assumption rather than a hidden
`mu_mu = -mu_C`), and `EoSPoint` (what a mode returns: the conditions, the
nested `matter` and `leptons` blocks, and the TOTALS on top). `eos/general/
modes.py` holds `ModeSpec`, the mode as one choice per conserved charge, whose
`name` property is where the section 3 mode name comes from.

Where this stands: dd2 and sfho return `EoSPoint` with `matter` a
`PhaseThermo` and `leptons` a `LeptonThermo`, and `eos/mixed` consumes
`PhaseThermo` across the phase-adapter surface. Still on flat records of their
own: zl, vmit, alphabag, abpr, enjl, did and njl. Each conversion changes what
`eos_point` returns and so lands in its own commit with its baseline re-run --
except that, as with dd2 and sfho, two models whose records are the same shape
should convert together rather than leave two conventions standing.

What the dd2/sfho conversion cost and what it did not: every one of the 2784
dd2 baseline quantities and 2117 sfho quantities that survives the rename is
BIT-IDENTICAL, and the key names are what moved (`sigma` ->
`matter.fields.sigma`, `n_e` -> `leptons.densities.e-`, and so on). Two
quantities left the baseline rather than moving, both benign:

- dd2's `mu_S` at 90 points. `generate_baseline.row` drops mu_S where no
  strange species is populated, and dd2's flat record carried no `n_S` field
  for the rule to read, so the rule had never fired for dd2. All 102 stored
  values were exactly 0.0.
- sfho's `Y_Le` at 67 points, where the mode does not hold a lepton fraction.
  The flat record carried a default 0.0; `conditions` carries the fraction
  only when the mode fixes it. All 67 were exactly 0.0, and the three genuine
  ones (Y_Le = 0.4) survive as `conditions.Y_Le`.

`Y_C`, `Y_S` and the Euler residual are no longer stored fields: the first two
are properties of `PhaseThermo` over the recorded `n_C`, `n_S` and `n_B`, and
`hvh_rel` is `EoSPoint.euler_residual()`. sfho's `P_hadrons` / `P_leptons` /
`P_photons` split is `matter.P`, `leptons.P` and the remainder; nothing read
it. `LeptonThermo` still has no muon family in sfho -- `mu-` is present and
zero, which is what the model implements.

What each model still on a flat record has to supply is decided by ONE
question -- what its internal self-consistent solution is -- because the rest
of the state is (chemical potentials, T) in every model:

    dd2, sfho          meson mean fields sigma, omega0, rho0, (phi0);
                       n_i derived
    enjl               constituent masses M_u, M_d, M_s from the gap
                       equation; n_i derived
    vmit               the vector field V = a hbar c sum_q n_q
    zl                 the interaction potentials mu_Hv(n_p, n_n)
    alphabag, abpr     none -- everything is explicit in mu

vmit and zl are the two that look different and are not: both carry densities
in the *unknown vector* because that keeps the residual polynomial rather than
nesting Fermi integrals inside it. That is a conditioning choice of the
solver, not a statement about the state -- physically vmit's state is still
(mu_q, V, T). So `fields` holds whatever the model solves for, under the
model's own names, and is empty where there is nothing to solve.

Three things the records fix that the remaining models get wrong in the same
way, so they are worth doing together rather than one model at a time. dd2 and
sfho are converted on all three and are the worked examples:

- **nucleons are privileged over hyperons.** Every model's kinetics computes
  mu_eff and m* for each active species and then keeps only n and p. Neither
  is recoverable from the record afterwards (they need the fields and the
  per-species coupling ratios), so g-modes and the response functions recompute
  them. `m_eff` is even singular although m*_i differs per species -- in DD2Y
  at n_B = 0.6 the neutron sits at 192 MeV while Lambda is at 652 and Xi- at
  1083, so the single number was the neutron's and the rest were discarded.
- **the mode is welded into the state.** dd2 carried eight mode fields on its
  context and branched on strings inside the residual; the other models do the
  same with their own vocabularies, which is why no two of them accept a mode
  the same way.
- **the non-leptonic charge is called `mu_Q` in some models and `mu_C` in
  others.** Section 2 says C. Storing mu_C rather than recovering it as
  mu_p - mu_n is also the numerically better choice, and measurably so: those
  are two ~1300 MeV numbers differing by ~100 MeV, so the subtraction costs
  about two digits, and removing it from dd2's warm start took its backend
  parity from 3.0e-14 to 8.9e-16.

---

### The module names are standardised, and most models have not been renamed

CLAUDE.md §5 fixes one name per role — `parameters.py`, `species.py`,
`thermodynamics.py`, `solver.py`, `table.py`, `api.py`, `verify/` — with the
names mandatory and their existence conditional. Two renames are done
(`dd2/parametrization.py` → `parameters.py`, `vmit/thermodynamics_quarks.py` →
`thermodynamics.py`). The rest are outstanding, each belonging in its model's
own session where the baseline and `test_imports.py` are already being run:

    dd2       DONE: notebook_api.py deleted, with its smoke test and the
              `_EXEMPT_FILES` entry in test_imports.py that recorded it as a
              violation. Its only importer was notebooks/DD2_usage, removed in
              the same change. The records are the shared `eos.general.state`
              ones (above). The rest of dd2's layout is now CLAUDE.md §5: physics/{thermo, fields,
              mesons} and the non-residual half of physics/octet became
              thermodynamics.py; octet_residual, assemble_octet and
              physics/residual.py became solver.py; physics/ became backends/
              {jacobian, kernel_numba, responses_jac}; coefficients.py became
              responses.py (kept apart from responses_jac.py because §9 makes
              them the reference and fast flavors of one thing, and backends/
              has to stay deletable); nmp.py + nmp_inverter.py merged; xp.py
              deleted.
    sfho      DONE: eos.py -> solver.py, compute_tables.py -> table.py,
              nuclear_saturation_properties.py -> nmp.py,
              thermodynamics_hadrons.py -> thermodynamics.py, and species.py
              and api.py added. `result_to_guess(result, eq_type)` is
              `warm_start(result, spec)`, reading the mode declaration rather
              than a string. `SFHoParams` is `Parameters`, and the records
              are the shared `eos.general.state` ones (above). What is left is
              `get_sfho_default()`, which carries the model name where
              section 13 asks for `Parameters.default()`.
    vmit      DONE. The file moves were already done (eos.py -> solver.py) and
              `VMITParams` was already `Parameters`; the ~23 names below
              `Parameters` have now followed. The fourteen `compute_*` names
              that carried nothing lost the prefix (`compute_quark_thermo` ->
              `kinetic_thermo`, `compute_bag_pressure`/`_energy` ->
              `bag_pressure`/`bag_energy`, `compute_quark_matter_thermo_from_mu`
              -> `thermo_from_mu`, and so on); the two records that repeated
              the package are `EoSPoint` and `MatterThermo`;
              `get_vmit_default()` is `Parameters.default()`; the four
              `solve_vmit_*` entry points carry their section 3 mode names;
              and `result_to_guess` / `get_default_guess_*` are
              `warm_start(point, mode)` / `default_guess(mode, ...)`, both
              reading the mode name rather than a private string -- which
              retired `table.py`'s `_GUESS_KIND` translation table with them.
              `eos/zlvmit` moved with them, importing `Parameters as
              VMITParams` and `thermo_from_mu_n as vmit_thermo_from_mu_n`
              against the ZL names in the same signatures; `eos/mixed`
              likewise aliases the three vMIT `thermo_from_*` names, since
              five models now expose that vocabulary into one file. Three
              legacy table symbols in `compute_tables.py`
              (`VMITTableSettings`, `compute_vmit_table`, `save_vmit_results`)
              are deliberately frozen: their only consumer is the ZLvMIT
              notebook. `get_vmit_custom()` is DELETED: the frozen
              dataclass carries the identical defaults, so it was a pure
              alias for `Parameters(...)`. zl, abpr and vmit have converted;
              sfho has not.
    zl        DONE: eos.py -> solver.py, compute_tables.py -> table.py,
              thermodynamics_nucleons.py -> thermodynamics.py, and species.py
              and api.py added. `ZLParams` is `Parameters` and
              `get_zl_default()` is `Parameters.default()` -- the first model
              to take the section 13 name. Every one of the ten models has
              since followed -- sfho and vmit first, dd2 last, where the class
              was `Parametrization` until the dd2 rename took it -- so the
              claim "every model's parameter dataclass is `Parameters`" is now
              true of all ten and is checkable in one line:
              `grep -n '^class Parameters' eos/*/parameters.py` returns ten
              hits. What sfho still owes is the CONSTRUCTOR half, not the
              class: its five published sets are still `get_sfho*` free
              functions rather than `Parameters.default()` /
              `Parameters.named()`.
              `compute_zl_thermo_from_mu` is `thermo_from_mu`,
              `compute_nucleon_thermo` is `kinetic_thermo`,
              `compute_V_interaction` / `compute_P_interaction` are
              `interaction_energy` / `interaction_pressure`, and
              `compute_mu_pHv` / `compute_mu_nHv` are one function,
              `interaction_potentials`, returning the pair. `result_to_guess`
              is `warm_start(point, mode)` and `get_default_guess_*` is
              `default_guess(mode, ...)`, both reading the mode name rather
              than a private string. `eos/zlvmit` and the two ZLvMIT notebooks
              moved with them; the legacy code imports `Parameters as
              ZLParams` and `thermo_from_mu_n as zl_thermo_from_mu_n`, which
              is disambiguation against the vMIT names in the same
              signatures, not a compatibility alias.
    alphabag  DONE: eos.py -> solver.py, compute_tables.py -> table.py,
              thermodynamics_quarks.py -> thermodynamics.py, and species.py,
              api.py and verify/ added. `AlphaBagParams` is `Parameters` and
              `get_alphabag_default()` is `Parameters.default()`;
              `get_alphabag_custom(...)` is gone, since the frozen dataclass
              IS that constructor. The `_alpha` suffix came off the massless
              closed forms (`n_massless_alpha` -> `n_massless`), which named
              the package rather than the quantity, and `compute_` came off
              everything: `compute_alphabag_thermo_from_mu` ->
              `thermo_from_mu`, `compute_cfl_thermo_from_mu` ->
              `cfl_thermo_from_mu`, `compute_quark_thermo` ->
              `kinetic_thermo`, `n_quark_alpha` -> `quark_density`,
              `compute_bag_pressure`/`_energy` -> `bag_pressure`/`bag_energy`,
              `gap_cfl`/`dgap_dT_cfl` -> `cfl_gap`/`cfl_dgap_dT`,
              `P_cfl_correction` and siblings -> `cfl_P_correction` and
              siblings. `compute_alphabag_total_thermo_from_mu` is
              `point_from_mu` and its CFL twin `cfl_point_from_mu`;
              `AlphaBagEOSResult`/`CFLEOSResult` are `EoSPoint`/`CFLPoint`;
              `solve_alphabag_*` are `solve_beta_eq_neutrinoless`,
              `solve_fixed_yc` and `solve_fixed_yc_ys`, joined by the new
              `solve_beta_eq_neutrino_trapped`; `AlphaBagTableSettings` /
              `compute_alphabag_table` / `save_alphabag_results` are
              `TableSettings` / `compute_table` / `save_results`. Deleted with
              no caller anywhere: `B_eff_cfl`, the `compute_u/d/s_thermo`
              aliases, the module-level `settings` block that ran on import,
              and the whole `.dat` LOADER stack (`load_eos_table`,
              `load_eos_tables_multi`, `build_interpolators`, `EOSTableData`,
              `results_to_arrays`, `COLUMN_MAPS`, `GRID_AXES`) -- the study
              that reads tables back reads the sfho ones.

              NUCLEATION IS NOT UPDATED (Phase 6). Its suite is already red
              from the sfho renames, so nothing there would have caught a
              missed edit; the exact map it needs is below.
    abpr      DONE: the one file eos.py became parameters.py, species.py,
              thermodynamics.py, solver.py and api.py, plus verify/, and every
              name lost its package: `ABPRParams` is `Parameters` (with
              `ms` -> `m_s` and `Delta` -> `Delta0`, the names its two sibling
              quark models already use), `pressure_abpr` /
              `baryon_density_abpr` / `energy_density_abpr` are `pressure` /
              `baryon_density` / `energy_density`, and `mu_from_nB_abpr` /
              `mu_from_P_abpr` / `mu_from_epsilon_abpr` are `mu_from_nB` /
              `mu_from_P` / `mu_from_eps`. `ABPREOSResult` is `CFLPoint`, the
              name eos/alphabag gives the same record. `Parameters.B` now
              returns MeV^4 rather than MeV/fm^3, matching alphabag and vmit;
              the single division by (hbar c)^3 moved to
              `thermodynamics.pressure`, so no number changed.
              `get_abpr_default()` is `Parameters.default()` and
              `get_abpr_custom()` is gone -- the frozen dataclass IS that
              constructor, and its signature defaults were a SECOND set of
              defaults (Delta = 100, B4 = 145) disagreeing with the
              dataclass's own (80, 135) in the same file.
              Deleted with no caller anywhere: `generate_abpr_tables`,
              `_write_table`, `_write_combined_table` and the `__main__`
              parameter-scan block -- a .dat writer with prints in it, which
              `eos.general.table_io` and `api.eos_table(rows=True)` replace.
              Every call site is inside eos (nucleation does not import abpr)
              and moved in the same commit: README section 6 and
              `notebooks/mass distribution.ipynb`.
    enjl      DONE: `uniform.py` and `eos_beta.py` are gone, but not as the
              merge the plan called for -- CLAUDE.md section 5 splits them
              differently, and along the line the section draws.
              `uniform.solve_point` takes DENSITIES and solves only the
              model's own gap equation, so it is `thermo_from_n` and it went
              to `thermodynamics.py` with the gap, the baryon masses, the
              effective scalar densities, the mean fields, the vacuum and the
              charge sums; `eos_beta`'s residual and solve are `solver.py`;
              the continuation is `table.py`; and `api.py`, `species.py`'s
              flags and `verify/` are new. `ENJLParams` is `Parameters` with
              `default()` and `named()`, `get_enjl_default()` is gone,
              `ENJLEoSPoint` is `EoSPoint`, `solve_beta_point` is
              `solve_beta_eq_neutrinoless`, `beta_eos_table(grid, ...)` is
              `build_table(TableSpec(...))`, `_continuation_state` is
              `warm_start`, `_baryon_masses` and
              `_effective_scalar_densities` lost their underscores (both were
              imported by the notebook and the plot script, so they were
              public whatever they were called), and `_evaluate` /
              `_residual` are `state_at` / `residual`. The four `*_t0`
              free-gas functions are gone: `kinetic_thermo(nu, m, g, Lambda)`
              takes the medium part from `eos.general.fermi_integrals` and
              adds the model's own Lambda vacuum terms, which is the whole
              of CLAUDE.md section 7 here and the seam finite temperature
              needs. `make_species` and its `Species` dataclass are gone too;
              the quantum numbers are plain tables in `species.py`.
              `P_kin_t0` and the old `kinetic_thermo` had no caller anywhere
              and were deleted -- the second was also wrong, returning
              eps = 0 below threshold where every live call site returned the
              negative vacuum term.

Function names go with them, per the §13 vocabulary: no name repeats its
package (`compute_zl_thermo_from_mu` -> `thermo_from_mu`), and the same job
carries the same name everywhere (`get_<model>_default` -> `Parameters.default()`,
`get_default_guess_*` -> `default_guess(mode, ...)`, `result_to_guess` /
`*_warm_start` -> `warm_start(point)`, `compute_<model>_table` ->
`build_table(spec)`). These are public names, so each model's renames land in
its own commit with every call site fixed alongside — no aliases, since two
names for one thing is what the rule removes.

`dd2/parameters.py` carried three function-level imports, two of them
commented "local import breaks the cycle", because three constructors reach up
to `solver.py` from the bottom layer. `from_nmp` is now a free function in
`nmp.py` and its import is gone. Two remain: `from_hyperon_potentials` and
`from_delta_potential`, which solve symmetric matter at saturation to invert a
single-particle potential U_Y or U_Delta into a scalar coupling ratio.

They are the same kind of object as `from_nmp` — an inverse map from a
physical observable to a coupling, which is why they sit above `solver.py` —
so the natural home is `nmp.py`, with that module documented as the maps
between couplings and the quantities they are fitted to rather than as
nuclear-matter parameters alone. What is not yet decided is whether the
module keeps the name `nmp.py` once it holds potential inversions too. About
ninety call sites use these as classmethods, so the move belongs with the
`Parametrization` -> `Parameters` rename, which touches the same lines.

`vmit/eos.py` is now `vmit/solver.py`, and every importer moved with it except
the notebooks -- see the notebook entry below.

`vmit/compute_tables.py` is the one deliberate exception to the scheme: it is
the first-generation settings-object interface, kept because the ZLvMIT
notebook drives vMIT through it, and it now sits beside `table.py` as a shim
over the shared driver rather than being renamed to it.

**The exception covers its three symbols too**, which it previously did not
say: `VMITTableSettings`, `compute_vmit_table` and `save_vmit_results` each
repeat the package and so break section 13 rule 1, independently of where the
module lives -- the parallel `zl` names were converted to `TableSettings` /
`compute_table` / `save_results` and these were not. They are FROZEN rather
than renamed: their only consumer is `notebooks/ZLvMIT_hybrid.ipynb`, which
CLAUDE.md section 5 exempts as legacy and which the current map rules out of
scope, so renaming them buys nothing and touches a notebook kept for published
results. The previous state -- a documented exception on the file, an
undocumented one on the symbols -- was the actual defect.

`eos/mixed` is a composite engine and takes the shorter list of CLAUDE.md §5 —
`adapters.py`, `api.py`, `responses.py`, `verify/`, `mixed.tex` — plus the
subpackage-free model-shaped modules its solve needs: `charges.py`,
`thermodynamics.py` (mode-blind, per the §5 grep test), `solver.py`,
`boundaries.py`, `hybrid.py`, `table.py`, and `backends/` for the analytic
Jacobian.

#### The alphabag map Phase 6 applies to `nucleation`, verbatim

Five files import `eos.alphabag` directly, plus the jupytext-paired notebook.
Nothing else in either repository does. Old -> new, complete:

    MODULES
    eos.alphabag.eos                      -> eos.alphabag.solver
    eos.alphabag.thermodynamics_quarks    -> eos.alphabag.thermodynamics
    eos.alphabag.compute_tables           -> eos.alphabag.table

    NAMES
    AlphaBagParams                        -> Parameters
    get_alphabag_default()                -> Parameters.default()
    get_alphabag_custom(alpha=, B4=, m_s=)
                                          -> Parameters(alpha=, B4=, m_s=)
                                             keyword for keyword; the `name`
                                             keyword survives too
    compute_alphabag_thermo_from_mu       -> thermo_from_mu
    compute_cfl_thermo_from_mu            -> cfl_thermo_from_mu
    compute_alphabag_total_thermo_from_mu -> point_from_mu
    compute_cfl_total_thermo_from_mu      -> cfl_point_from_mu
    solve_alphabag_beta_eq                -> solve_beta_eq_neutrinoless
    solve_alphabag_fixed_yc               -> solve_fixed_yc
    solve_alphabag_fixed_yc_ys            -> solve_fixed_yc_ys
    solve_cfl                             -> solve_cfl      (unchanged)
    T_critical                            -> T_critical     (unchanged)
    AlphaBagTableSettings                 -> TableSettings
    compute_alphabag_table                -> compute_table

    SIGNATURES: unchanged, every one of them, positional and keyword alike.
    RESULT FIELDS: unchanged. `AlphaBagEOSResult` is `EoSPoint` and
    `CFLEOSResult` is `CFLPoint`, but neither is constructed by name outside
    `eos`, and every field keeps its name.

The five files and what each one needs:

    nucleation/composition.py       both module lines; four names
                                    (compute_alphabag_total_thermo_from_mu,
                                    compute_cfl_total_thermo_from_mu,
                                    compute_alphabag_thermo_from_mu,
                                    compute_cfl_thermo_from_mu)
    nucleation/conditions.py        one module line (T_critical, name
                                    unchanged)
    nucleation/analysis/filters.py  two module lines; get_alphabag_custom and
                                    two solve_alphabag_* names
    nucleation/analysis/droplet.py  one module line; get_alphabag_custom
    nucleation/analysis/scan.py     one module line; get_alphabag_custom
    notebooks/2fam_PNS_nucleation.{py,ipynb}
                                    three module lines; get_alphabag_custom,
                                    AlphaBagTableSettings,
                                    compute_alphabag_table, T_critical --
                                    BOTH halves of the jupytext pair.

`eos.alphabag` re-exports every new name, so each import line may equally
become `from eos.alphabag import ...`.

---

### `docs/STRUCTURE.md` does not exist yet

CLAUDE.md §10 sends a new figure to the `figure_style` module docstring **and**
to a worked figure example in `docs/STRUCTURE.md`, and §11 lists the file as
part of the layout. Only the module docstring is written. The document belongs
with the notebook rework in Phase 5, which is where the worked example comes
from.

---

### The README's directory tree predates the refactor

`README.md` still lists `general/plotting_info.py` and a one-file `tov/`, both
of which are gone, and carries no `dd2/`, `mixed/` or `enjl/` at all. It is a rewrite rather than a patch, and Phase 5 does it
alongside the notebook rework; individual lines are corrected only where a
rename in this phase would otherwise leave them newly wrong.

---

### The freeze selector of `eos_response` is a fixed menu, not a selection

CLAUDE.md §5 requires the freeze to be selectable case by case. Today each
model takes a single string from a short hard-coded list — dd2
`('equilibrium', 'composition')`, vmit `('equilibrium',)`, mixed
`('equilibrium', 'chi')` — so a combination nobody anticipated cannot be
asked for. The target is a *set* of held quantities, with the named freezes as
presets that expand to one:

    "equilibrium"  frozenset()                    nothing held
    "fast"         {every species Y_i} | {"chi"}  no reaction has time
    "slow"         {"Y_C", "chi"}                 all chemical equilibria but beta
    "conserved"    {"Y_C", "Y_S", "chi"}          strong imposed, both weak frozen

so that `frozen={"Y_C", "Y_S"}` (chi free), `frozen={"chi"}` and
`frozen=set(species)` are all reachable. The names come from Constantinou,
Guerrini et al., arXiv:2506.20418 §IV, whose fast and slow limits are taken at
fixed (y_i, chi) and fixed (Y_e, chi) respectively — the {y_i} there are
PARTICLE fractions, and the conserved-charge description appears only in the
slow limit, where imposing every equilibrium but beta collapses them to Y_C.

A second, orthogonal axis is missing entirely: the thermal condition. Every
`eos_response` in the repository differentiates at fixed T, while the adiabatic
sound speed of the CompOSE manual and of that paper is taken at fixed entropy
per baryon. At T = 0 they agree; at T = 50 MeV they do not.

The NAMING half of that is now closed across the ten models: every one returns
`cs2_isothermal`, and `did`, `njl`, `ccdm`, `sfho` and `dd2` return
`cs2_adiabatic` beside it, derived through C_P/C_V. What remains is that `zl`,
`vmit` and `alphabag` return no adiabatic speed at all — they carry no C_P to
form it with. Two surfaces also still spell the pair for the composition axis
rather than the thermal one: `mixed.eos_response` returns `cs2_eq`/`cs2_frozen`,
and `astro/gmode` takes `cs2_eq`/`cs2_ad` as argument and background-field
names. Those two share the `sound_speed_eq` / `sound_speed_frozen` vocabulary of
`mixed/responses.py` and are one rename, not two.

That the word itself is contested is the reason the pair is carried on two
axes rather than one. In Zhao & Lattimer (arXiv:2204.03037) Eq. (1) the
"adiabatic sound speed" `c_s` means FROZEN COMPOSITION and the g-mode is the
difference between it and the equilibrium `c_e`; in the CompOSE manual
"adiabatic" means FIXED ENTROPY. The repository serves both, so the
composition is said by the `frozen=` argument and the thermal condition by the
key name, and no model returns a word doing both jobs.

---

### The response functions are three finite-difference stencils, not one derivation

C_V, C_P, Gamma and c_s^2 are all second derivatives of the same free energy
per baryon F(T, n_B, Y_C), and the CompOSE manual (arXiv:2203.03209 §3.6)
derives every one of them from d2F/dT2, d2F/dn_B dT and d2F/dn_B^2. The code
instead takes a separate central difference per quantity at a relative step of
1e-3, which is the least accurate step in the response path. Constantinou,
Guerrini et al. arXiv:2506.20418 Eq. (76)-(77) closes the loop:
(dP/dn_B)_S = (C_P/C_V) (dP/dn_B)_T, so the adiabatic sound speed follows
algebraically from the isothermal one and the heat-capacity ratio — one
stencil instead of two. Worth doing when the freeze selector is built, since
both touch the same code.

### The ideal pion gas leaves its domain just above saturation

`eos/general/thermal_mesons.py` is now the single implementation, with physical
masses and the isospin partners split, and both dd2 and sfho go through it.
What is NOT implemented is what happens when a meson's effective potential
reaches its mass: the species condenses and the ideal-gas expressions stop
describing it. `solve_bose_jel` caps mu at m rather than diverging, so nothing
blows up -- a caller that does not look simply receives a saturated gas where
a condensate belongs.

Every entry point therefore reports `condensation`, the largest |mu*_j| / m_j
over the active species, and BOTH models refuse such a point -- sfho by
setting `converged=False`, dd2 by raising, each matching how that model
already reports a bad state. `eos_point` says which, in both. That is the
agreed interim behaviour, an error rather than a wrong number, until a
condensate is written.

What a condensate needs, when it is written: mu*_j pinned AT m_j as an
equation rather than derived, the condensed density n_cond,j as a new unknown
of the solve, and its contribution added as eps = m_j n_cond,j with P = 0 and
s = 0 -- the p = 0 state carries charge and energy but neither pressure nor
entropy. The thermal part is already right: capping mu at m returns the
critical density of the excited states, which is what it should be.

WHY IT HAPPENS SO EARLY, which is the part worth understanding before anyone
concludes the code is wrong. In beta equilibrium the pion potential is

    mu*_pi- = -mu_C + Gamma_rhoN rho0 = mu_e + Gamma_rhoN rho0

and the rho term is NEGATIVE in neutron-rich matter, so it SUPPRESSES
condensation. What drives it is mu_e alone (DD2, npe(mu) matter, T = 10 MeV):

      n_B     mu_e   Gamma_rho rho0   mu*_pi-   /m_pi
     0.10   102.25          -21.72     80.53    0.577
     0.20   136.56          -20.63    115.93    0.831
     0.25   149.94          -17.80    132.14    0.947
     0.30   161.94          -14.78    147.15    1.054
     0.60   213.56           -3.42    210.14    1.506

mu_e crosses m_pi = 139.57 MeV at about n_B = 0.27 fm^-3. That is the textbook
s-wave criterion for pi- condensation, mu_e >= m_pi*, and it is met just above
saturation in every beta-equilibrium nucleonic model. The arithmetic here is
right.

What is missing is the repulsive s-wave piN interaction, which raises the
in-medium pi- energy and is precisely what suppresses s-wave pion condensation
in realistic matter. An IDEAL pion gas shifted only by the vector mean fields
has no such term, so it condenses spuriously early. The domain where the gas
as implemented is valid is the one it was written for -- heavy-ion and
early-protoneutron-star conditions, high T and low mu_B, where mu_e is small
(Lavagno) -- and NOT cold beta-equilibrium neutron-star matter above about
0.25 fm^-3.

So the refusal is not a nuisance to be worked around: it marks a real boundary
of the model. Two things would move it, and they are different work. An
in-medium pion self-energy (an s-wave optical potential) pushes the threshold
up to where it physically belongs and is what makes the gas usable in
beta-equilibrium matter at all. A condensate handles what happens past
whatever threshold survives. The first is the more valuable.

NEGATIVELY CHARGED BARYONS SUPPRESS IT, and strongly enough to matter. The
condition is driven by mu_e, and Sigma- and Delta- take over the job of
neutralising the protons, so the electrons are no longer needed and mu_e
collapses. DD2/DD2Y, beta equilibrium, T = 10 MeV, mu*_pi- / m_pi:

      n_B    nucleons   +hyperons   +hyperons+Deltas
     0.25       0.947       0.947              0.942
     0.30       1.054       1.052              0.975   <- peak with Deltas
     0.40       1.239       1.120              0.860
     0.60       1.506       0.992              0.608
     1.00       1.840       0.592              0.212

With nucleons alone the gas condenses from n_B ~ 0.28 upward and never
recovers. With hyperons it condenses over a WINDOW, roughly 0.29 to 0.58, and
comes back out above it as mu_e turns over. With the Deltas open as well the
ratio peaks at 0.975 and the condition is never met at all.

So the sector that makes the ideal gas usable in beta-equilibrium matter is the
one that was physically motivated anyway. Two caveats: 0.975 is a thin margin,
so it is parametrization-dependent -- the Delta coupling ratios move it -- and
this is T = 10 MeV, with higher T lowering mu_e further and making it safer.
A caller wanting the thermal gas in cold neutron-star matter should open the
Deltas and check `condensation` rather than assume.

How far it reaches today. In SFHo, over beta equilibrium and fixed
Y_C = 0.05 / 0.5, nucleons and hyperons, |mu*|/m reaches 3.11 (pi+), 3.52 (K+)
and 2.62 (K0) by n_B = 1.2 fm^-3. In DD2 BETA EQUILIBRIUM the ratio is above 1
for every n_B >= 0.3 at T <= 40 MeV and only falls back below it above T ~ 80.

A claim that used to stand here -- that there is therefore NO state both inside
the DD2+vMIT coexistence window and outside condensation -- was WRONG, and is
withdrawn. It was measured through a broken seed (below), which refused on the
BETA-EQUILIBRIUM gas rather than on the mixed phase's own; the mixed phase sits
at different potentials and its gas is markedly less critical. Solvable states
inside the window, eta = 0.5:

      T     n_B     chi    |mu*|/m
     60    0.50   0.480      0.914
     60    0.60   0.967      0.756
     70    0.40   0.418      0.755
     70    0.50   0.954      0.627
     80    0.40   0.825      0.491

And ETA decides it, more strongly than temperature does. At eta = 0 (Gibbs)
only GLOBAL neutrality is imposed, the hadronic phase stays positively charged,
mu_C is far less negative than in beta equilibrium, and mu*_pi- = -mu_C +
Gamma_rhoN rho0 stays below m_pi. At n_B = 0.7, T = 20 MeV -- deep in the
region the old code refused outright -- eta = 0 gives chi = 0.342 and
|mu*|/m = 0.907, inside the window and inside the model, while eta = 0.5 and
eta = 1.0 at the same point are condensed and are refused. That is the same
mechanism by which Sigma- and Delta- suppress condensation in a pure hadronic
phase, with the quark phase playing the negative-charge carrier.

So the hybrid engine IS usable with a thermal meson gas: at eta = 0 across the
transition, and at any eta above about T = 60 MeV.

Every consumer is now guarded. `PhaseThermo` carries `condensation` -- the
shared record and `eos/mixed`'s own -- both adapters fill it (a quark phase has
no meson gas and reports 0), and `solve_mixed` refuses a condensed phase the
way dd2 does. What that uncovered: `eos/mixed` could not run with the gas AT
ALL. Both `solvers/point.default_guess` and `adapters.hadronic_seed` built
their starting configuration from a full beta-equilibrium dd2 solve carrying
the meson flags, and dd2 raises on a condensed gas -- so every gas-enabled
mixed call died in its seed, and the condensation check downstream never ran.
The gas sources none of the four field equations and adds no unknown to either
vector being seeded, so both seeds now switch it off: a seed must not fail for
a reason that has nothing to do with seeding.

### astro/gmode is DD2-only until nine models implement a composition freeze

RESOLVED as a section 1 breach; what remains is a per-model capability gap.

`eos/astro/gmode` used to import model internals at four sites, one of them
`rates.py:85` at top level, so `import eos.astro.gmode` pulled DD2 in whether
or not the caller wanted a DD2 background. The contract that replaced them is
`eos.general.sound_speeds.EOSTable_for_gmode`: the `(P, eps, n_B)` table plus
`cs2_equilibrium` and `cs2_frozen` on the same rows, living beside
`EOSTable_for_TOV` in the layer both `astro/` and the models may import. A
model produces one; the mode solver consumes one; `test_imports.py` gates it
both as source text and at runtime in a fresh interpreter.

The payload is the two SOUND SPEEDS, not a composition derivative: Zhao and
Lattimer (arXiv:2204.03037) Eq. (1),
`nu_g^2 = g^2 (1/c_e^2 - 1/c_s^2) e^(nu-lambda)`, needs only those two per
point, and `eos_response` already computes both. T = 0 only -- Zhao's operative
clause is "without varying chemical composition", not the zero temperature, so
T = 0 collapses the thermal axis and leaves the composition axis intact.

**What is still deferred.** `frozen='composition'` is implemented in `eos.dd2`
alone: eight models expose only `equilibrium`, and `enjl` exposes no freeze at
all. (This paragraph read "six models ... and `njl`, `ccdm` and `enjl` expose no
freeze at all" until the measurement in the section below was taken against the
tree rather than restated: njl and ccdm have carried `equilibrium` since the
commits that introduced them.) So nine models cannot compute the second sound speed
under any conditioning and cannot fill the contract; one that cannot raises
saying so, which is section 3's own answer to a partly-filled surface. This
converts "gmode is DD2-only by accident, hidden inside
`from eos.dd2.solver import solve_composition`" into "gmode is DD2-only until
nine models implement one freeze": visible, per-model and ticketable.

Also deferred: the DD2 producer itself. `dd2.eos_response(frozen='composition')`
returns the LEPTONLESS frozen speed, which is the right probe of the nucleonic
sector but the wrong half of a g-mode -- differencing it against the
with-leptons equilibrium speed compares two different fluids, and the lepton
term is a sizeable fraction of the whole buoyancy signal. The with-leptons
producer belongs in `eos.dd2`, reached through section 5's third response axis
(`leptons=`). It currently sits in
`eos/astro/gmode/verify/run_full_check.py` as `dd2_table` / `dd2_frozen_cs2`,
which is where the tests import it from, so there is one copy rather than two.

**The verify/ carve-out, ruled.** Section 1's `verify/` exemption is written
for the model-to-model half of the rule; it extends to an ASTRO suite importing
a model, and not to the reverse. The two directions are not symmetric. A model
importing `astro/` is the cycle the rule exists to prevent and has no carve-out
anywhere. An astro suite importing a model creates no cycle -- `astro/` already
sits above the models -- and the carve-out's own justification, that a suite is
not on the path an inference sampler imports, is a statement about suites and
not about which layer they sit in. `gmode/verify/run_full_check.py` uses it
narrowly: five of its eight checks are model-free and `include_dd2=False` turns
the rest off.

### The composition freeze: what each model owes, and the order it lands in

`RESPONSE_FREEZES` measured across the ten models and the composite engine,
against the tree rather than restated from an audit:

    dd2                              ('equilibrium', 'composition')
    mixed                            ('equilibrium', 'chi')
    sfho did zl vmit alphabag
      abpr njl ccdm                  ('equilibrium',)
    enjl                             ()

`composition` therefore exists in exactly ONE of eleven units, and `mixed` --
absent from every earlier count -- carries a third spelling that is neither.

**Why this is not one job done nine times.** A composition freeze needs the
model evaluated at PRESCRIBED SPECIES DENSITIES with no equilibrium condition
imposed: section 13's `thermo_from_n`, the direction that INVERTS the Fermi
integrals rather than solving them forward. It cannot be reached by re-tuning
(mu_B, mu_C, mu_S), because three conserved potentials cannot hold eight
species fractions -- so any model with more than nucleons needs the
density-side block outright, and cannot borrow the potential-side one it
already has. Who has that block today:

    zl     thermo_from_n(n_n, n_p, T)          nucleonic, complete
    vmit   thermo_from_n(n_u, n_d, n_s, T)     three flavours, complete
    enjl   thermo_from_n(n, ...)               all eight species, complete
    dd2    solve_composition(n_n, n_p, T)      NUCLEONIC ONLY
    sfho did alphabag abpr njl ccdm mixed      absent

Note what the fourth line says about the one working implementation:
`solve_composition` is a single `brentq` on the sigma gap over n and p, so with
`hyperons=True` or `deltas=True` **dd2 cannot freeze either**. What exists is
the nucleonic freeze, not the freeze.

**What each model's freeze costs, in its own physics.** They are not alike, and
assuming they were is what made this look like one ticket:

- `dd2` -- has it, nucleonic. Owes the with-leptons variant (below) and the
  extension past n, p: with hyperons the sigma gap is unchanged in form but the
  freeze needs all eight densities as inputs, i.e. `thermo_from_n` after all.
- `zl` -- cheapest of the eleven. The interaction is a closed-form function of
  two densities, so at fixed (n_n, n_p) NO fixed point is solved at all: the
  potentials are read off and the Fermi integrals inverted once.
  `thermo_from_n` already does it. This is wiring, not physics.
- `vmit` -- cheapest in the quark sector, and for the same reason: one
  algebraic mean field, no scalar sector, three independent Fermi inversions.
  `thermo_from_n` already does it.
- `sfho` -- the first real cost. Its four field equations are NONLINEAR in the
  fields (the c3 omega^4 self-coupling and the omega-rho mixing A(omega, rho)),
  so at prescribed densities they are a coupled four-dimensional root find, not
  dd2's single scalar gap. The block has to be written; `thermo_from_mu`
  (`thermodynamics.py:556`) solves the same four equations from the other side
  and is the shape to mirror.
- `did` -- dd2's shape plus a physics choice that is DID's alone. Its couplings
  depend on n_B *and* on the isospin asymmetry beta = sum_i tau_3i n_i / n_B,
  so the model carries TWO rearrangement self-energies. Along a frozen-
  composition sequence beta is constant by construction, so the beta channel
  contributes nothing to the frozen derivative while the n_B channel still
  does. That is defensible -- beta IS a composition variable -- but it makes
  DID's frozen sound speed structurally not the same object as the other RMFs',
  and it must be chosen deliberately rather than inherited from whichever model
  gets ported first.
- `alphabag` -- flavours decouple entirely in the unpaired sector
  (`quark_density(mu_q, T, m_q, alpha)` per flavour), so the freeze is three
  independent one-dimensional inversions. But of the ALPHA_S-CORRECTED density,
  which is not the free Fermi form, so `general.invert_fermi_density` does not
  serve and the model owes its own inverter. Per section 7 that is model
  physics, not an integral re-implementation.
- `abpr` -- **nothing to freeze, and this is a ruling rather than work.** The
  model is `cfl` and nothing else, and colour-flavour locking sets
  n_u = n_d = n_s identically, hence Y_C = 0 and Y_S = +1 with no fraction
  free (section 3 says exactly this). A composition freeze is therefore
  DEGENERATE: the frozen sound speed IS the equilibrium one, and a g-mode built
  on an ABPR background is identically zero for a physical reason, not a
  missing feature. What abpr owes is that sentence in its docstring and a
  `composition` entry that returns the equilibrium number, not a new solve.
- `njl`, `ccdm` -- the most expensive pair, for two reasons that compound.
  First, the freeze must re-solve the model's own self-consistency at each
  stencil point at prescribed flavour densities: the chiral gap equations for
  `njl`, those plus the dielectric field for `ccdm`. Their integrals are
  cutoff-regularised, so again the shared inverter does not serve. Second,
  **pairing means the composition is not independently free.** In CFL the
  locking is abpr's above and the freeze is degenerate; in 2SC two flavours of
  three are tied. Worse, both models SELECT their pattern (and ccdm its branch)
  by comparing pressures, and under a frozen composition that is not the same
  competition -- so the freeze must declare whether the pattern is held WITH
  the composition or re-selected at every stencil point. That declaration is
  the real cost, and it is the same question already open for enjl's branch
  pair.
- `enjl` -- two steps behind on the surface and one step ahead on the physics.
  It has NO `eos_response` at all, so `equilibrium` comes first and is the
  larger job. But its `thermo_from_n` is the most complete in the repository
  (all eight species, the gap equation for M_u, M_d, M_s solved inside the
  density inversion) and is the direction the paper's own Figs. 1-3 are
  evaluated in, so once the surface exists the composition freeze is cheap.
- `mixed` -- **last, by construction rather than by priority.** Its `chi` freeze
  already holds the quark volume fraction and with it each phase's Y_C and Y_S,
  with the species re-equilibrating inside them. Section 5's `fast` preset
  additionally holds every Y_i, and the engine HAS no species of its own: the
  per-species freeze lives in the phase and is reached through the
  phase-adapter contract. So mixed can hold {Y_i} | {chi} only once BOTH phases
  of a pairing can hold their own {Y_i}, which is why it cannot be ordered
  earlier no matter how much its consumers want it.

**The order.** Driven by the live consumer -- the g-mode contract, which needs
the two sound speeds on the same rows -- and, after that, by taking the models
whose block already exists before the ones whose does not:

    1. dd2        move the with-leptons producer home and add the third axis;
                  this is what makes the ONE existing freeze usable for a
                  g-mode at all, and its test is already waiting
    2. zl, vmit   the two models whose `thermo_from_n` exists; wiring, and they
                  prove the axis generalises across a hadronic AND a quark
                  model before anything expensive is built
    3. abpr       the ruling above: degenerate, no new solve
    4. sfho, did  the two RMFs that must WRITE the density-side block; did's
                  two-rearrangement choice made deliberately, not inherited
    5. alphabag   its own alpha_s density inverter
    6. njl, ccdm  the pattern-under-freeze question decided FIRST, then the
                  gap-coupled block
    7. enjl       `equilibrium` first (it has none), then composition, which is
                  cheap once the surface exists
    8. mixed      last by construction: needs both phases of a pairing to hold
                  their own species fractions first

Steps 1-3 are the ones that pay immediately: after them a g-mode has three
backgrounds (a DD-RMF, a nucleonic model with no fields, a bag model) and the
abpr zero is explained rather than missing.

**The third axis, and a name collision to resolve first.** Section 5's third
conditioning axis -- whether leptons re-neutralize against the held charge --
is what a g-mode needs and is missing everywhere but `mixed`. It cannot simply
be spelled `leptons=`, because that keyword is ALREADY TAKEN on `eos_response`
and means something else:

    sfho zl did vmit alphabag njl ccdm   leptons= is the section 3 MODE flag,
                                         routed into mode_spec(mode, fracs,
                                         leptons) exactly as on eos_point
    mixed                                leptons= is the RESPONSE axis, routed
                                         into sound_speed_frozen, deciding
                                         whether the perturbed mixture is
                                         re-neutralised
    dd2 abpr enjl                        no such argument

One keyword, two jobs, already shipped in opposite senses -- section 13's "one
name per job", and the reason this spelling is decided here rather than in nine
per-model tickets, where it would drift on contact.

**Ruled: `leptons=` keeps its section 3 mode meaning on `eos_response`, and the
response axis takes a different name.** Three reasons, none of them a majority
count. The mode flag is the SAME argument `eos_point` and `eos_table` take, and
a uniform API (section 5) cannot have one entry point read a keyword one way
and its sibling another. The rule for it across the modes was just settled and
landed in nine models, so re-pointing the name would reopen a closed question
for a second one. And the two are genuinely orthogonal -- `leptons=` says
whether the STATE has leptons, the third axis says whether the PERTURBATION
re-neutralises -- so a caller can want both at once, which one keyword cannot
express.

The proposed name is `reneutralize=`, a bool, defaulting to True (the physical
choice for stellar matter, and what `mixed` already defaults to). A g-mode's
two speeds are then

    eos_response(par, mode, species, frozen='equilibrium',  n_B=...)
    eos_response(par, mode, species, frozen='composition', n_B=..., Y_p=...,
                 reneutralize=True)

and the leptonless nucleonic probe is the same call with
`reneutralize=False`, which is what `dd2.responses.sound_speed_adiabatic_frozen`
returns today. `mixed` is the one surface that changes, one rename against
nine models left alone.

Note that under this ruling the collision has a concrete casualty to watch:
`leptons=False` on a beta-equilibrium mode now RAISES, so the leptonless frozen
probe MUST be reached through `reneutralize=False` and not by turning the mode's
leptons off. The two would have meant the same thing to a caller reading only
the argument name, and one of them is an error.

**What is recorded here and what is not.** This entry is the measurement, the
per-model cost and the order. It implements no freeze; each step above is its
own piece of work, and the per-model entries later in this file
already carry the model-local half of the statement: the `eos_response` bullet
under each of `### dd2`, `### did`, `### sfho`, `### zl`, `### vmit`,
`### alphabag`, `### mixed`. They are cited by SECTION NAME and not by line,
because inserting this entry moved every one of the line numbers the ticket
that ordered it quoted.

### Eleven deferred imports are downward, where no cycle exists

Function-local imports are how this repository breaks a genuine cycle -- section
5's layer order makes `nmp.py` -> `solver.py` one, for instance. These eleven are
not that: each imports a module BELOW it in the same package, where a top-level
import would work. Style drift, cheap to fix, and recorded so the next reader
does not mistake them for cycles that must stay deferred:

    eos/sfho/nmp.py:66,67          solver.solve_fixed_yc, species.SpeciesFlags
    eos/sfho/nmp.py:138            solver.solve_fixed_yc again
    eos/did/nmp.py:162             solver.solve_beta_eq_neutrinoless
    eos/sfho/thermodynamics.py:583 species.active_baryons
    eos/njl/thermodynamics.py:479  species.pattern_seed
    eos/ccdm/thermodynamics.py:557 species.branch_seed, species.pattern_seed
    eos/sfho/api.py:193            responses          (as `_fd`)
    eos/did/api.py:145             responses          (as `_fd`)
    eos/njl/api.py:174             responses          (as `_fd`)
    eos/ccdm/api.py:197            responses          (as `_fd`)

The `api.py -> responses.py` shape is four of the eleven and is the same edge
four times: section 5 puts `responses.py` below `api.py`, so it is a plain
top-level import in every one of them. The `dd2` sibling this list was first
written against no longer exists -- `eos/dd2/api.py` now defers only
`backends/responses_jac` (`:166`) and `responses` (`:178`), both inside the
branch that selects an analytic Jacobian, which is a deliberate optional-backend
deferral rather than drift.

### Three physics-bearing module constants have no override path

CLAUDE.md section 6: model parameters are arguments, never module-level
constants. These four numbers are, and each one moves a result:

    eos/mixed/adapters.py:315   _CHIRAL_SPLIT = 50.0            MeV
    eos/mixed/adapters.py:320   _DECONFINED_BARYON_FRACTION = 1.0e-4
    eos/astro/tov/rns_backend.py:94-95
                                RNS_RHO_SURFACE = 7.8           g/cm^3
                                RNS_P_SURFACE = 1.01e8          dyn/cm^2
    eos/zlvmit/hybrid_table_generator.py:46
                                B4 = 165.0                      MeV

Deferred rather than fixed, and for a reason that is the same in all three
cases: each is a THRESHOLD whose change moves a classification, not a coupling
an inference run varies. `_CHIRAL_SPLIT` and `_DECONFINED_BARYON_FRACTION`
decide which phase LABEL a converged point is given (`adapters.py:331,335`), not
what the point is; the two RNS surface values mirror hardwired numbers in the C
source and changing them makes the backend disagree with the code it wraps; and
`B4` is in `eos/zlvmit/`, which section 5 exempts wholesale. Promoting a
threshold to an argument is worth doing when something needs to vary it, and
nothing does yet.

### sfho and zl parameter dataclasses are not frozen

Section 6 asks for no global mutable state and permits read-only caches keyed by
immutable parameters. Both dataclasses pickle, so multiprocessing and MPI work
today; neither is hashable, so neither can key such a cache.

`eos/zl/parameters.py:28` is the benign half -- eight `str`/`float` fields, so
`@dataclass(frozen=True)` is a one-word change whenever anything wants it.

`eos/sfho/parameters.py:69` is the reason this is deferred and not done.
It carries `a_coeffs: np.ndarray` (`:152`), which is unhashable whatever the
decorator says, and a `couplings_map: Dict[str, Dict[str, float]]` (`:160`)
that is WRITTEN AFTER CONSTRUCTION by every published set --
`:373,389,400,425,441,452,480`. Freezing it is therefore not a decorator change
but a rewrite of five constructors into either `dataclasses.replace()` or a real
`__init__` that takes the couplings. `eos/enjl` shows the target shape already:
a frozen dataclass with a `_vacuum_cache` keyed on it
(`enjl/thermodynamics.py:404-430`).

### `output/public/` does not exist yet

CLAUDE.md section 11 describes it as the existing tracked subfolder of an
otherwise-gitignored `output/`. `.gitignore:36-38` reserves it correctly --
`output/`, then `!output/public/` and `!output/public/**` -- but `ls output/`
returns nine entries and none of them is `public/`. Nothing is broken: an
un-ignore rule for a path that does not exist is inert.

Not created here because an empty tracked folder is not a deliverable and git
will not carry one. The `mkdir` belongs with the first table that deserves to be
in it, which waits on the replacement notebooks: they produce tables under
standardised names, and WHICH of those tables are worth publishing is a curation
decision that cannot be made before the tables exist.


## Per model

### dd2
- `nmp.invert_nmp(impose_Q_sat=True)` — the Q_sat-imposing isoscalar closure —
  converges but does not IMPOSE what it names. Ticket 93 removed the half of
  this entry that was a solver defect; ticket 105 (2026-08-28) measured the
  cause and named the remedy, and what is deferred is now that remedy alone.

  **The cause, measured.** Q_sat is a third finite difference of a solved
  quantity: it spans 2.48 MeV over h in [5e-5, 5e-4], a relative floor near
  1.5e-03. The five-row closure that imposes it conditions at 259 at the
  published DD2 point, so a solve inherits 259 x 1.5e-03 = 0.39 of relative
  coupling error. No choice of pinned coefficient rescues that — 259 is the
  best of the four (c_omega 259, b_omega 354, c_sigma 703, b_sigma 4191) — and
  no reparametrisation does either, since the collinearity is a rank statement
  rather than a coordinate one: E_sat and m*/m at fixed n_sat are blind to the
  shape coefficients, so four shape knobs answer to three rows and the shape
  block has an exact rank deficiency of one.

  **The remedy, and it is the whole of what remains deferred.** Take the
  derivative analytically instead of by stencil. K_sat, a second difference,
  spans only 5.2e-04 MeV over the same h range, so the default closure is
  unaffected and does not wait on this. With the floor gone the amplification
  is harmless at any of the four pins, and imposing Q_sat becomes legitimate.
  Forward and inverse must go analytic TOGETHER — the finite-difference bias
  currently cancels on a round trip only because both difference identically.

  **Retired by ticket 105, no longer part of this entry.** The old text
  blamed a cross-constraint f_sigma''(1) = f_omega''(1) that the DD2 table
  obeys only to 2.200718e-03. That constraint is DD's, not DD2's (Typel, PRC
  71, 064301 (2005) Sec. IV imposes it and counts eight parameters; Typel et
  al., PRC 81, 015803 (2010) states only f_i(1) = 1 and f_i''(0) = 0 and
  counts ten), and it has been removed from the inverse map. With it gone the
  published couplings are a root of the default closure and the round trip
  recovers them to 1.1e-05.

  **A defect this surfaced and did not fix**, for the ticket that measures it:
  ISO_GATE = 2e-2 was set wide to clear the cross row's 2.2e-03 and Q_sat's
  stencil, and neither reason survives in the default closure. On a 105-cell
  (K_sat, m*/m) grid with restarts on, the 101 passing cells split 95 below
  1e-5 and 6 in [1e-3, 2e-2] with nothing in between, so six are certified
  without being roots. Choosing the tighter value moves `ok` for real targets
  and needs a scan on more than two axes.

  What depends on it today: nothing in the library — every shipped path uses
  the default closure, and a "Q_sat" key in the target dict no longer selects
  anything.
- `table.hadronic_row` emits a Y_C and Y_S that are BARYONS ONLY, while the
  `EoSPoint` it flattens carries the totals. It recomputes them itself:

      _, n_C, n_S = hadronic_charges(flags, p.composition_map)

  and `composition_map` holds baryons alone, so a thermal meson gas is dropped.
  Its own docstring says the row is "keyed exactly the way
  `eos.mixed.composition_row` keys a mixed point, so a pure-hadronic table and
  a hybrid table concatenate without renaming anything" — but the mixed side
  uses the totals, so the same column name carries different physics on the
  two sides, differing by 10–20 percent at T = 40 MeV with pions. This is the
  defect class already fixed once in the hadronic frozen wing (b83d162, then
  `sound_speed_frozen_pure`),
  in the sibling that was not checked then, and it contradicts CLAUDE.md §2.

  The fix is one line — `Y_C=p.Y_C, Y_S=p.Y_S` — but it CHANGES TABLE COLUMNS
  at T > 0, so it is a deliberate physics-changing fix: its own commit, the
  before/after quoted, and the affected baseline entries regenerated. It must
  not be folded into a refactor, which is why it is still here.
- Deleting `backends/` leaves every equation-of-state baseline BIT-IDENTICAL
  at rtol = 1e-10 — that is CLAUDE.md §5's property, measured — but moves the
  TOV sequences by 4.8e-07 relative and M_max by 2.5e-08 (6e-08 Msun on a
  2.4 Msun star). The jitted T = 0 kernel and the NumPy one evaluate the same
  closed form to machine precision, yet the two paths converge to roots
  differing in the last bits, and the adaptive integrator amplifies that by
  ~1e7 — the same amplification measured in the backend-parity entry under
  astro/tov. Nothing to fix in `backends/`; recorded so the next person to run
  the deletion check knows the tov baseline is expected to move and by how
  much.
- `susceptibilities` exists only in the analytic-Jacobian flavor. §9 says the
  fast flavor is validated against a reference, and for chi_ab there is none:
  the sound speeds and heat capacities have their finite-difference twins in
  `responses.py`, chi_ab does not. Two consequences. It is unvalidated except
  against its own symmetry and a hand-rolled grand-canonical difference in
  `test/dd2/test_dd2_m10_jac.py`; and `eos_response(frozen='equilibrium')`
  raises without `backends/` rather than degrading to a slower path, which is
  the one place §5's deletability is a feature gap rather than a speed cost.
  `thermo_from_mu` now makes the reference cheap to write — perturb
  mu_B, mu_C, mu_S, re-solve, read (n_B, n_C, n_S) — but writing it is new
  physics, so it waits for the response-function session.

  sfho has since found a cheaper reference that needs no new physics at all,
  and dd2 should adopt it: chi_ab is the INVERSE of dmu_a/dn_b, and
  `fixed_YC_YS` already computes that direction — impose (n_B, n_C, n_S),
  read back (mu_B, mu_C, mu_S), and require chi (dmu/dn) = I. It agrees to
  1.6e-05 at n_B = 0.8 fm^-3 with hyperons, and is pinned in sfho's
  `verify/run_full_check.py`. It has to be taken where the conjugate density
  is populated: at n_B = 0.16 with hyperons at T = 10 MeV, n_S is 2.5e-07
  fm^-3 and the numerical dmu_S/dn_S is meaningless (the flat-mu_S entry at
  the top of this file).
- The two models return chi_ab in DIFFERENT UNITS. dd2's is natural
  (MeV^2), sfho's is fm-based (fm^-3 MeV^-1). §5 makes fm-based the rule at
  every public boundary, so dd2's is the one to change; it is left alone here
  because it is dd2's number and this was an sfho session. Same quantity, same
  physics — a units convention, not a discrepancy.
- `eos_response` implements the freezes `equilibrium` (beta_eq_neutrinoless
  only: both sound speeds, C_V, C_P, chi_ab) and `composition` (nucleonic
  Y_p: both sound speeds and Gamma). Not yet wired: frozen conserved
  fractions (Y_C, Y_S fixed with species re-equilibrating), the leptonic
  re-neutralization variants, the thermal index through the API, and
  `equilibrium` for the other modes. All raise naming this file.
- `Gamma` under `frozen='composition'` is built on `cs2_isothermal`, so it is
  the ISOTHERMAL index; the fixed-entropy one is larger by the same C_P/C_V
  that separates the two sound speeds, and at T = 0 they coincide. The
  material to form it is already there — `responses._frozen_derivs` returns
  the heat capacities along the frozen-Y_p sequence — so this is a second key
  rather than new physics, left out because changing what the existing key
  means at T > 0 was not what the naming pass was authorised to do.
- The muon lepton family is not tracked in the trapped mode:
  `beta_eq_neutrino_trapped` takes (n_B, Y_Le, T) and Y_Lmu raises.
- `fixed_YC_YS` with neutralizing leptons (`leptons=True`) is not wired; the
  flag applies to `fixed_YC` only.
- The `thermal_neutrinos` sector is not wired. The FLAG is carried, so the six
  §4 names construct a dd2 `SpeciesFlags` the way they construct every other
  model's, but setting it True raises `NotImplementedError`: the neutrino
  flavours a mode does not track are not carried as mu = 0 gases here. dd2's
  own `neutrinos` is a different sector — the matter-composition electron
  neutrino of the trapped modes — and stays.
  (The naming half of this entry is closed: `include_pseudoscalars` and
  `include_thermal_vectors` are now `thermal_mesons` and the secondary
  `thermal_vectors`, §4's "pi, K (and optionally the vector nonet)".)

### did
- The low-density nuclear-statistical-equilibrium sector is not implemented.
  Section III of arXiv:2511.15646 embeds 8244 nuclei from AME20/FRDM12 as an
  excluded-volume van der Waals gas inside the RMF sea, and it is what the
  paper's crust and its Table VIII radii are built on. `eos/did` is the
  uniform-matter part alone; below saturation a caller attaches a crust table
  through `eos/astro/tov`, as every other model in this repository does. The
  consequence is measured rather than assumed: with BPS attached,
  R_1.4 = 12.07 km against the published 11.99 km, and M_max agrees to
  0.002 M_sun. Closing it means a cluster sector — a new module and a new set
  of degrees of freedom, not a wiring job — and it belongs with whatever
  session brings NSE to the whole repository.
- The inverse nuclear-matter map is not implemented and is not published for
  this functional form. `nmp.compute_nmp` is the forward direction only: the
  couplings come from a Bayesian analysis over 18 observables (hyperon
  potentials in two media, saturation properties, chi-EFT and heavy-ion
  pressures), and the nuclear-matter parameters are what that fit PREDICTS.
  An inversion would have to choose which 15 of those observables to impose,
  which is a modelling decision the paper does not make. DID also carries two
  inequivalent symmetry energies (S and S_2, differing by 2.72 MeV at
  saturation), so even the LIST to impose is undetermined: {n_0, B, K, S, L}
  and {n_0, B, K, S_2, L_2} are different inversions with different answers.
  The refusal is NAMED rather than absent: `nmp.invert_nmp` and `nmp.from_nmp`
  exist, are exported from `eos.did`, and raise `NotImplementedError` giving
  the reason and the two ways to close it. An absent attribute is an
  `AttributeError` a caller cannot interpret. (`eos.zl` used to refuse for the
  analogous reason and no longer does: its inverse is closed form, and the
  free choice six couplings leave against five NMPs is named by the caller
  rather than chosen by the function. DID cannot follow, because for DID it is
  the LIST to impose that is undetermined, not one member of a family.)
- `eos_response` implements the `equilibrium` freeze only (c_s^2 isothermal
  and adiabatic, C_V, C_P, Gamma_th, in every mode). The composition freezes
  — held species fractions, held Y_C with the species re-equilibrating — and
  the susceptibility matrix chi_ab are not wired and raise naming this file.
  For DID a frozen-composition derivative has a wrinkle the other models do
  not have: holding the composition holds beta, so the couplings stop moving
  too, and whether that is what "frozen" should mean is a physics choice to
  make deliberately rather than inherit.
- The muon lepton family is not tracked as a conserved charge:
  `beta_eq_neutrino_trapped` takes (n_B, Y_Le, T) and a Y_Lmu raises. The muon
  SPECIES is wired and selectable, and it matters: it moves the hyperon onsets
  up by about 0.03 fm^-3 relative to the paper's electron-only sector.
- The thermal kaon effective potentials carry the omega and rho shifts but not
  the phi. The arithmetic is inherited from `eos/dd2`, where g_phiN = 0 makes
  the omission exact; in DID the SU(3) vector sector gives the nucleon
  g_phiN = -5.2, so the kaon's strange quark sees a field the potential does
  not know about. Fixing it means deciding what the additive-quark shift is
  for a hidden-strange vector, which is model physics rather than a missing
  term, and it should be decided once for every model that carries a phi.
- The Delta(1232) quartet is an EXTENSION, not part of arXiv:2511.15646:
  there is no published DID Delta coupling table, so the ratios default to
  universal coupling and `nmp.delta_ratios_from_potential` offers the
  U_Delta inversion. Any Delta result from this model is this
  implementation's, and did.tex says so.
- DDBY, the comparison parameterisation of the paper's Table V, is not
  shipped. Its four sigma couplings are published there but the base DDB model
  they attach to is not (Ref. 92 of arXiv:2511.15646 carries it), so a DDBY
  set here would be four numbers on top of a parameterisation this repository
  does not have. It is a comparison curve, not part of DID, and adding it
  means adding DDB.
- The delta (a_0(980)) and f_0(980) mesons are absent by the model's own
  design, not by omission here, and there is no species flag for them. Adding
  either changes the model rather than switching on a sector.
- S_2, L_2 and K_sym2 are extracted as coefficients of a beta^4-truncated
  expansion (one Richardson step from beta = -1 and -0.5), and agree with the
  published values to about one percent, against two parts in a thousand for
  everything else in Table VI. The gap is a definition, not an error: the true
  beta -> 0 curvature is nearer 32.9 MeV, the published S_2 is 32.44, and
  which one a given extraction returns depends on how much beta^6 behaviour it
  absorbs. A small-step estimator cannot decide it either — at beta = 0.05 the
  signal is 0.08 MeV on a binding energy of 900 MeV. Closing it means asking
  the authors what their fit window was.

### sfho
- The muon lepton family is not wired: it appears in no residual, no
  neutrality row and no total. `include_muons=True` now raises (CLAUDE.md §4)
  where it used to be accepted and ignored.
- The NMP inversion is written, closed as {g_sigma_N, g_omega_N, g2, g3}
  against {n_sat, E_sat, K_sat, m*/m} in the isoscalar sector and
  **(g_rho_N, b1)** against {E_sym, L_sym} in the isovector one. What remains
  open is smaller and is recorded here rather than in the module, which states
  the closure it uses and why.

  Q_sat and K_sym are PREDICTIONS, as in dd2. Unlike dd2 there is no option to
  impose Q_sat instead: doing so would need a fifth isoscalar knob, and the
  natural candidate is c3, which is currently held at its published value
  because it is a high-density vector parameter that saturation says little
  about. Worth adding if a target set ever carries Q_sat.

  The hyperon and Delta sectors are NOT refitted by the inversion. Their
  couplings are stored as absolute values derived from ratios against the
  NUCLEON couplings, which the inversion has just changed, so a parameter set
  inverted from a base carrying hyperons keeps hyperon couplings that no
  longer correspond to the potential depths they were built from.
  `from_potential_depths` has to be re-run on the result to hold
  U_Lambda, U_Sigma, U_Xi. The docstring says so; folding it in automatically
  would mean deciding whether the depths or the ratios are the thing held,
  and that is the caller's physics.

  The closure has a second branch, and it is refused rather than returned.
  E_sym's potential term is n g_rho^2 / [8 (m_rho^2 + 2A)] with A = g_rho^2 f,
  so it SATURATES at n/(16 f) as g_rho grows: a runaway (g_rho_N, b1) can
  reproduce a target set exactly. A target at low m*/m and low L_sym
  (0.60, 20 MeV) has ONLY that root — a 19x23 seed scan finds no other — and
  it fits every NMP to 1e-9 at 2A/m_rho^2 = +108.9, against +0.37 for
  published SFHo. `invert_nmp` therefore checks |2A| < m_rho^2 at saturation
  after converging and reports ok=False when the fit lands there. Every
  accepted fit from L_sym = 40 to 140 sits inside [-0.40, +0.69]. The limit is
  the assumption the model form is written under — A is a correction to the
  rho mass term, not a replacement for it — but it is a single number, and
  where exactly the physical branch ends has not been mapped.
- The mean fields are `sigma, omega, rho, phi` here and `sigma, omega0, rho0,
  phi0` in dd2, in `EoSPoint` and in `PhaseThermo.fields` alike. One name per
  job (section 13), so one of the two spellings has to go; sfho's is also the
  `.dat` column header the published 2fam PNS tables carry, so the rename is a
  file-format change and belongs with the table-I/O unification rather than
  with a record swap.
- `eos_response` implements the `equilibrium` freeze only — `cs2_isothermal`,
  and at T > 0 also `cs2_adiabatic`, `C_V`, `C_P`, `Gamma_th` and the
  susceptibility matrix `chi`. Everything but chi is a finite difference along
  re-solved sequences in `sfho/responses.py`; chi comes off the analytic
  Jacobian. Not wired, and raising: every freeze that holds a composition. A
  per-species freeze needs the Y_i in the residual, which SFHo does not carry
  (dd2 reaches its `composition` freeze through `solve_composition(n_n, n_p)`,
  which has no SFHo counterpart), and holding the conserved fractions with the
  species free is the fixed_YC / fixed_YC_YS modes differentiated at fixed
  fraction — cheap to add, not yet asked for.

  The three-stencil objection in the cross-cutting section applies to this
  implementation as written: C_V, C_P and the two sound speeds are separate
  central differences rather than one derivation of F(T, n_B, Y_C). What it
  does NOT repeat is the naming defect — the returned sound speeds say which
  thermal condition they were taken at, and `cs2_adiabatic` is derived from
  `cs2_isothermal` through C_P/C_V rather than by a second stencil.
- An isentropic fixed-Y_C solve with neutralizing leptons raises: the
  electrons follow from n_C only after the solve, so they are missing from
  the entropy row that fixes T. Wiring it means putting mu_e in the unknown
  vector for that mode.
- The `.dat` writer and reader (`save_results`, `load_eos_table`,
  `build_interpolators`, `COLUMN_MAPS`, `GRID_AXES`) are a per-equilibrium
  column layout of `table.py`'s own, rather than `eos.general.table_io`. It is
  kept deliberately: the published 2fam PNS nucleation tables were written in
  that format and are read back through it, so unifying it changes files on
  disk and belongs with the nucleation propagation, not with a refactor.
  `TableSettings` / `compute_table` are kept for the same reason and are now a
  thin adapter onto `build_table` rather than a second sweep -- the shape
  `vmit/compute_tables.py` already has. `load_eos_table`'s `mu_nue` column used
  to be reconstructed as `mu_e + mu_nu` from a result field the solvers never
  set -- wrong relation (it is `mu_e + mu_C`) on a column of zeros. Tables
  written before that fix carry zeros there.
- `TableSettings.Y_L_values` keeps its name because zl and vmit spell it the
  same way; the §2 rename to `Y_Le_values` lands once, across all three.

### zl
- `eos_response` implements the freeze `equilibrium` only, and computes it by
  central differences along the mode's own sequence (c_s^2 = dP/deps at fixed
  T, C_V = (T/n_B) ds/dT at fixed n_B) because ZL has no analytic Jacobian in
  this repository. Frozen composition, frozen conserved fractions and the
  leptonic re-neutralization variants all raise naming this file. An analytic
  Jacobian is easy here -- the interaction is a closed-form function of two
  densities, so d(mu_Hv_i)/dn_j is elementary -- and would give the
  susceptibility matrix chi_ab as well.
- The muon lepton family is not wired: `SpeciesFlags(muons=True)` raises, and
  `beta_eq_neutrino_trapped` takes (n_B, Y_Le, T) only.
- `thermal_neutrinos` -- flavours not tracked in the composition, carried as
  mu = 0 gases -- is not wired and raises.
- `eos_point` takes the entropy-per-baryon axis; `TableSpec` does not, and
  raises for `axes={'SnB': ...}` -- the same gap vmit has, and it closes the
  same way.
- `mu_S` is reported as 0.0 rather than as undefined. The model has no strange
  degree of freedom at all, so unlike the flat-mu_S entry at the top of this
  file this is a genuine convention rather than a weakly determined number;
  `fixed_YC_YS` raises rather than accepting a Y_S it would have to ignore.
  Whether the API should say "this charge does not exist here" differently
  from "this potential is undetermined" is the same open question.

### vmit
- `eos_response` implements the freeze `equilibrium` only, and computes it by
  central differences along the mode's own sequence (c_s^2 = dP/deps at fixed
  T, C_V = (T/n_B) ds/dT at fixed n_B) because vMIT has no analytic Jacobian
  in this repository. Frozen composition, frozen conserved fractions and the
  leptonic re-neutralization variants all raise naming this file. An analytic
  Jacobian is straightforward here -- the model has one algebraic mean field
  and no scalar sector -- and would give the susceptibility matrix chi_ab as
  well.
- The muon lepton family is not wired: `SpeciesFlags(muons=True)` raises, and
  `beta_eq_neutrino_trapped` takes (n_B, Y_Le, T) only.
- `thermal_neutrinos` -- flavours not tracked in the composition, carried as
  mu = 0 gases -- is not wired and raises.
- `eos_point` takes the entropy-per-baryon axis; `TableSpec` does not, and
  raises for `axes={'SnB': ...}`. The outer solve exists
  (`eos.general.tabulate.temperature_at_entropy`); wiring it into the table
  driver is what is left.
- The flavour densities are not constrained positive. At exotic fixed
  fractions (Y_C well above 1, say) the equations have solutions with net
  ANTI-down and anti-strange densities, and the solver returns them as
  converged. They are genuine states of the model at finite temperature, not
  solver failures, but nothing in the API says so; a scan over fractions
  should either filter them or the result should carry a flag.

- `compute_tables.py` is a SECOND table driver beside `vmit/table.py`, outside
  section 5's template, and its `VMITTableSettings` (`:41`) still repeats its
  package the way section 13 rule 1 forbids. It is **frozen deliberately, not
  missed.** The rename pass took the other two `VMIT*` symbols and stopped at
  this one because its only consumer is `notebooks/ZLvMIT_hybrid.ipynb`, which
  is legacy code kept for published results and out of scope for conformance --
  renaming a symbol whose sole caller is a notebook nobody is converting buys a
  cleaner name and a broken notebook. The two siblings in the same file,
  `compute_vmit_table` and `save_vmit_results`, are frozen for the same reason
  and by the same ruling. `get_vmit_custom()` is no longer among them: it is
  deleted, its defaults having been identical to the frozen dataclass's own.
  Recorded so the next reader does not re-derive the omission as a bug.

### alphabag
- `eos_response` implements the freeze `equilibrium` only, and computes it by
  central differences along the mode's own sequence (c_s^2 = dP/deps at fixed
  T, C_V = (T/n_B) ds/dT at fixed n_B) because alphaBag has no analytic
  Jacobian in this repository. Frozen composition, frozen conserved fractions
  and the leptonic re-neutralization variants all raise naming this file. An
  analytic Jacobian is straightforward here -- the potential is explicit in
  mu, so dn_q/dmu_q is one derivative of a closed form plus one Fermi
  integral -- and would give the susceptibility matrix chi_ab as well.
- The muon lepton family is not wired: `SpeciesFlags(muons=True)` raises, and
  `beta_eq_neutrino_trapped` takes (n_B, Y_Le, T) only.
- `eos_point` takes the entropy-per-baryon axis; `TableSpec` does not, and
  raises for `axes={'SnB': ...}` -- the same gap vmit and zl have, and it
  closes the same way.
- The paired phase carries NO thermal neutrino gas, where every unpaired
  solver adds three flavours (two when the electron neutrino is trapped). The
  asymmetry is inherited: the first-generation CFL table builder never passed
  its own `include_thermal_neutrinos` down to `solve_cfl`, so the CFL branch
  of every table written for the 2fam PNS nucleation study is missing that
  gas while the unpaired branch has it. It is preserved deliberately rather
  than fixed here, because closing it changes the published CFL tables and
  the two branches are compared against each other in that study. Closing it
  means passing `species.thermal_neutrinos` through `table.solve_at`'s `cfl`
  arm and regenerating those tables together.

  What is no longer deferred is the SILENCE. `table.solve_at` used to accept
  `thermal_neutrinos=True` in the `cfl` mode and drop it on the floor, which
  section 4 forbids -- a sector a model does not implement raises, and a
  `NotImplementedError` is never turned into a silent no-op. The `cfl` arm
  now raises naming this entry. The physics gap above is unchanged; only the
  way it is reported is.

- **The colour-flavour-locked phase has no light bosonic sector at all**, and
  what it should have is not what the code used to add. Locking breaks
  `SU(3)_c x U(1)_Q` to a single unbroken `U(1)_Qtilde`, so exactly one of the
  nine gauge bosons stays massless -- the rotated photon, which `photons=True`
  supplies at leading order -- and all eight gluons acquire Meissner masses.
  The light degrees of freedom the phase genuinely carries are the superfluid
  phonon and the pseudo-Goldstone meson octet, neither of which is
  implemented. Until the CFL branch was ruled on, `gluons=True` added the
  UNPAIRED phase's free gas of `g_g = 2 x 8 = 16` massless bosons instead:
  the wrong count and the wrong dispersion, worth +0.119 MeV/fm^3 in P and
  1.4-3.0% of the phase's entropy at T = 30 MeV, and +1.900 MeV/fm^3 and
  5-11% of the entropy at T = 60. That term is gone and `gluons=True` now
  raises in the `cfl` mode. Closing the gap properly means a phonon
  dispersion (`v = 1/sqrt(3)` at leading order) and the meson octet's masses
  -- new physics with its own literature, not a mass added to the gluon gas,
  which is why it is a deferred sector rather than a corrected one.

- The legacy `TableSettings` shim defaults `include_gluons=True` and
  `include_thermal_neutrinos=True`, so `compute_table(TableSettings(
  phase='cfl'))` now raises where it used to return a table. That is the
  ruling above reaching the shim, and it is correct: the first-generation
  CFL tables of the 2fam PNS study DO contain the free gluon gas, so the shim
  can no longer reproduce them at `T > 0` without naming
  `include_gluons=False` (and `include_thermal_neutrinos=False`, which the
  `cfl` arm used to drop in silence) and accepting the difference. The
  defaults are left as they are because they are the unpaired path's legacy
  defaults, which are unaffected; the shim's only caller in this repository
  is `test/alphabag/test_alphabag_api.py`, where the two flags are now named.
- The flavour densities are not constrained positive -- the same gap vmit
  has, from the same cause: at exotic fixed fractions the equations have
  solutions with net anti-down and anti-strange densities and the solver
  returns them as converged.

### abpr
- The three inverse maps are closed forms rather than root finds, so this
  model reports a residual it can always compute exactly; what it cannot
  report is a failure mode that does not exist. The one status it does return
  is `converged = False` for a target outside the phase (a pressure below -B,
  an energy density below the bag), and that is a property of the request, not
  of a solve that might have gone better from another start.
- `eos_response` returns `cs2_isothermal` and nothing else. The heat
  capacities C_V and C_P and the thermal and adiabatic indices are not defined
  at T = 0; the susceptibilities chi_ab = dn_a/dmu_b are singular, flavour
  locking leaving n_C and n_S with no potential to respond to. Closing this
  would mean giving the model a temperature, which is `eos/alphabag`.
- Finite temperature is absent by construction, not deferred: the four terms
  of the ABPR pressure are a T = 0 expression, and the finite-T CFL phase --
  with its BCS gap Delta(T), its thermal quarks and its entropy correction --
  is the `cfl` mode of `eos/alphabag`. `T > 0` raises pointing there.
- The strange quark mass is carried to O(m_s^2) only, so this model and the
  CFL phase of `eos/alphabag` differ by the m_s^4 term of that expansion.
  Measured at the shipped set, at three equal potentials, the pressure gap
  runs from -5.694 MeV/fm^3 at mu = 350 MeV to -7.796 at mu = 800, which is
  the analytic term to within 0.7% and 0.1% respectively; at matched n_B the
  disagreement is 7.9e-2 in P and 1.3e-2 in eps at n_B = 0.3 fm^-3, falling to
  2.8e-3 and 1.1e-3 by 3 fm^-3. This is not a gap to close -- adding the term
  to abpr would make it eos/alphabag, and adding abpr's O(m_s^2) term to
  eos/alphabag would count the strange mass twice -- but a study needing the
  strange mass better than a percent at the lowest densities should use the
  other model. `verify/run_full_check.py` asserts the relation.
- There is no published single ABPR parameter set, so `Parameters.default()`
  is a choice within the range a hybrid study scans rather than a fit to
  anything. It is the set `test/baseline` is frozen at, and at it the P = 0
  surface has E/A = 831.58 MeV -- absolutely stable strange quark matter.
- `eos_table` takes only `axes={'nB': ..., 'T': [0.0]}`: there is one
  temperature and no fraction to sweep, so a table has exactly one line. The
  entropy axis is accepted only at SnB = 0, since s = 0 identically.

- **The model refuses all four of section 3's equilibrium modes, and the refusal
  is physics rather than missing implementation.** `abpr/solver.py:47-49` sets
  `MODE_FRACTIONS = {"cfl": ()}`; `:54-73` gives the four refusal messages; `:84`
  raises `NotImplementedError` naming the reason. Section 3 requires that a mode
  a model cannot support be recorded here, and this section already lists four
  other limits without listing the largest one.

  The reason is one sentence repeated four ways: colour-flavour locking has
  already fixed the composition those modes exist to determine.
  `beta_eq_neutrinoless` would fix mu_C through mu_C + mu_e = 0, but locking
  leaves mu_C = 0 with no electrons to equilibrate against, so the condition has
  no free variable; `beta_eq_neutrino_trapped` adds that the phase carries no
  leptons of any family, so there is no Y_Le to fix; `fixed_YC` asks for a Y_C
  the phase does not have, since Y_C = 0 identically at every density and for
  every parameter set, and Y_C = 0 IS the `cfl` mode; and `fixed_YC_YS` has both
  fractions as OUTPUTS of the closure, so in particular the symmetric-matter
  slice Y_C = 0.5, Y_S = 0 that the mode exists for is not a state of deconfined
  locked matter at all.

  So this is a gap that will not be closed: it is what makes `cfl` the model's
  only mode, and what makes `abpr` the one model section 5 lets default its
  `mode` argument.

### zlvmit
- `get_default_guess` calls the ZL fixed-fraction and trapped solvers
  with T and the fraction transposed -- `solve_pure_H_fixed_yc(n_B, T, Y_C,
  ...)` where the signature is `(n_B, Y_C, T, ...)`, and the same in the
  trapped branch. The call sits inside a `try` whose `except` falls back to
  the analytic guess, and it only ever produces a SEED, so it cannot change a
  converged answer; found while renaming the ZL entry points and left alone,
  because zlvmit is kept for its published results and even a seed change
  moves the last bits of a baseline row.
- Styling is DONE: `eos/general/figure_style.py` is now the only source of
  rcParams, colours and figure geometry anywhere in `eos/zlvmit`.
  `plot_results.setup_matplotlib_style` is a wrapper that passes this study's
  25/20 pt text and DejaVu-first font order into `set_paper_style` rather than
  setting anything itself -- that font order is deliberate and must not be
  "fixed" to prefer CMU, because these are published figures and some CMU
  Serif installs are partial. The usage example in `table_reader.py` was the
  last holdout (figsize, dpi and a literal 'b-') and now writes into
  `output/zlvmit/` instead of the working directory.
- What is deliberately NOT done, and is not a gap: the uniform API. CLAUDE.md
  section 1 exempts zlvmit from it, and new hybrid work goes through
  `eos/mixed`. The module keeps its own solvers, its own table stack and its
  own `.dat` reader. It carries no `.tex`, no `verify/` and no
  `eos_point`/`eos_table`, and none of those are owed.

### enjl
- **Finite temperature is implemented; what is NOT is the CONSTRUCTION above
  T = 0.** All four modes solve at any T >= 0, entropy per baryon is accepted
  wherever a temperature is, and photons and thermal_neutrinos are selectable
  mu = 0 sectors. `build_constructed_table` and `eos.mixed.construction`
  (`enjl_coexistences`, `enjl_composition_coexistences`, `enjl_phase`,
  `locate_maxwell`, `locate_maxwell_composition`, `neutral_phase`,
  `composition_phase`) still raise for T != 0, and closing that is its own
  session:

    - locating a coexistence at T > 0 equates the GIBBS FREE ENERGIES of the
      two branches, not P and mu_B alone, so the entropy enters the
      coexistence bookkeeping. The composition closure already equates g
      rather than mu_B at T = 0 -- it has to, since g = mu_B is a beta-closure
      identity -- so what T > 0 adds there is the s term in g, not the change
      of variable;
    - `composition_phase` drops mu_S from its unknowns where a held Y_S = 0
      leaves the strangeness row reading 0 = 0, which is a T = 0 statement
      (every carrier has S = +1 and there are no antiparticles at T = 0). At
      T > 0 the Fermi tails populate the carriers and their antiparticles, the
      row acquires a gradient, and mu_S must be solved for -- the same
      correction `eos.enjl.solver.strangeness_row_is_empty` owes;
    - the plateau's lever rule then averages s across the window as well —
      `table.plateau_row` already levers `s` and derives `S_per_B` from the
      averaged value, so that half is in place;
    - and the eta = 1 lepton bookkeeping, which at T > 0 has positrons in it.

  Two smaller things left open with it. `eos_table` takes ONE thermal value
  per call (a T axis or an SnB axis of one value): that is not a temperature
  limitation but the rule the fraction axis already followed — a table here
  is a density continuation, and that is what carries the sweep — but a
  caller wanting a T grid has to loop, and `TableResult` would need a per-line
  shape to do otherwise. And `neutralizing_leptons` returns zeros for
  n_C <= 0; at T > 0 a phase with n_C < 0 would be neutralized by a net
  positron gas at mu_e < 0, which no mode of this model currently asks for.

  Three measurements from the session that built it, worth keeping:

  1. **Forward is free, inverse is not.** `state_at` and `thermo_from_mu` go
     nu -> n and cost no more at T > 0 than the Fermi integral itself.
     `thermo_from_n` goes n -> nu and is the only caller that inverts, through
     `general/fermi_integrals.invert_fermi_density` (round trip 1e-13, ~30 us,
     temperature-independent). That inversion sits INSIDE the gap iteration for
     the six strongly interacting species — nu depends on the mass and the
     masses are the gap unknowns — so a `thermo_from_n` call costs about 4 ms
     at T > 0 against nothing at T = 0. The leptons have fixed masses and stay
     outside it.
  2. **The JEL fit does not converge back to the exact T = 0 closed form.**
     `solve_fermi_jel` switches branches at T != 0 and the answer steps
     discontinuously, then STAYS at that offset as T falls: +6.9e-6 in n (u at
     nu = 400, M = 5.5), -6.3e-5 (s at nu = 500, M = 140.7), -3.0e-6 (neutron
     at nu = 1000), same in eps, flat from T = 1e-3 MeV down. A T -> 0
     continuity check therefore has a floor near 1e-4 relative;
     `verify/check_entropy_limit` states it rather than chasing it. The T = 0
     branch is kept EXACT for this reason: routing everything through the fit
     for smoothness would move every frozen number in the repository
     (CLAUDE.md section 12) to buy 1e-5 of cosmetic continuity.
  3. **`x ** 2` is not `x * x`.** They differ in the last bit for about one
     argument in a thousand on this platform's libm. Writing the seam with
     `kF * kF + m * m` where the old expression was `kF ** 2 + m ** 2` moved
     a fixed_YC sweep off a point at n_B = 0.533 fm^-3 — the deconfinement
     onset, a knife edge — nine densities after the last identical one.
     Anything claiming to hold `test/baseline` bit-for-bit must preserve the
     EXPRESSION, not merely the value.
- Cold starts stop converging around 0.5 fm^-3. The beta-equilibrium table is
  built by continuation (`table.build_table`), and the "up" and "down" sweeps
  differ where more than one branch exists — that difference is the branch
  structure, and choosing between branches needs a Maxwell construction that
  a single sweep cannot do. `test/baseline` freezes BOTH sweeps for exactly
  that reason.
- (The entry saying three of the four modes raise is REMOVED: all four are
  closed. A mode is now a declaration -- `eos.general.modes.ModeSpec`, the
  mechanism dd2 and sfho already use -- read by one residual assembly, so
  `beta_eq_neutrinoless` keeps exactly its ten slots and its rows while
  `fixed_YC`, `fixed_YC_YS` and `beta_eq_neutrino_trapped` add a potential and
  a row each. `thermo_from_mu(mu_B, mu_C, mu_S, T)` was added with them: it is
  the phase-adapter surface, nine unknowns including the phase's own n_B, no
  leptons and no neutrality.)
- `beta_eq_neutrino_trapped` carries the electron neutrinos as a massless
  left-handed gas, g = 1, in the lepton-number row and in eps, P and s. It
  does NOT carry a muon-neutrino family: `Y_Lmu` is not accepted, so the muon
  family stays transparent (mu_mu = mu_e - mu_nue). Adding it is one more
  unknown and one more row, and nothing has asked for it.
- Cold starts stop converging around 0.5 fm^-3 in EVERY mode, not just beta
  equilibrium; a table warm-starts, which is how the model is used. Measured
  on a 0.2-0.8 fm^-3 grid: all four modes converge cold at 0.2 and 0.4 and
  none at 0.6 or 0.8, while a warm-started sweep from 0.10 reaches 1.20 in all
  four.
- **The cap in Eq. (6) costs a little thermodynamic consistency, measured.**
  `effective_scalar_densities` caps nbar^s_q at zero from above, which is
  right — a positive value is a condensate of the wrong sign — but where the
  cap binds, nbar^s_q stops responding to the densities and eps stops being
  stationary with respect to that flavour's constituent mass. That
  stationarity is what makes mu_i = d eps/d n_i hold. It costs nothing when
  NO light flavour is capped, and nothing when BOTH are (the determinant term
  then vanishes in both light channels, M_u = M_d = m_q0 exactly, and the
  state sits in a flat region). It bites only when exactly one is: at
  f_q = 0.5, B = 0, n_B = 0.8 fm^-3 the identity misses by 6.9e-2 MeV on
  mu_Lambda = 1419 MeV, 4.8e-5 relative — below the 0.05-0.20 MeV at which the
  engine is validated against the author's tables, which is why it never
  showed up there. `verify/run_full_check.py` gates the smooth states at
  1e-3 MeV and the capped ones at 1e-1 MeV rather than absorbing the
  difference into one loose bound. Closing it means changing how Eq. (6) is
  regularized — a smooth cutoff, or carrying the cap as an explicit
  constraint in the stationarity condition — which is a physics decision.
- `eos_response` is not implemented and raises naming both reasons, one of
  which is now closed: T > 0 is implemented, so C_V, C_P, the thermal index
  and the isothermal/adiabatic distinction are no longer blocked by it. What
  remains is that c_s^2 and chi_ab = dn_a/dmu_b need
  the branch the derivative is taken along to be settled, and above the
  model's first first-order transition more than one branch satisfies the
  equilibrium conditions at the same density — differentiating along whichever
  one a continuation reached would return a number whose meaning depends on
  the direction the table was swept in. It unblocks with the Maxwell rule,
  not before.
- `SpeciesFlags` here is fixed rather than chosen: the model's species set is
  (p, n, Lambda, u, d, s, e, mu), so `hyperons` and `muons` are True and
  `deltas` and `thermal_mesons` False, and moving any of those four raises.
  `photons` and `thermal_neutrinos` are NO LONGER on that list: both are
  implemented, default False, and are the caller's.
  `deltas` and `thermal_mesons` are genuinely absent from the model — in particular
  sigma, omega and rho are auxiliary fields eliminated in favour of g^2/m^2,
  so there is no meson mass to put in a thermal gas. But `hyperons=False` is
  merely unimplemented: switching the Lambda off is dropping it from the
  species sums and the residual, a few lines, and no caller has wanted it.
  Note also that `hyperons=True` here means the Lambda alone; Sigma and Xi are
  not in the model, since the paper does not carry them.
- **The figure script does not run from a fresh clone.**
  `plot/enjl_paper_figures.py` does `sys.path.insert` into `test/enjl` and
  imports `PARAMETER_SETS`, `load_reference`, `solved_rows`, `bad_rows` and
  `baryon_potential` from `test/enjl/reference`. `test/` is gitignored, so a
  fresh clone has neither the loader nor the five `.dat` files, and the script
  fails at import. (`notebooks/ENJL_usage.py` shared this and was removed under
  Stage 0; the replacement enjl notebook must not reintroduce the reach into
  `test/`, which is exactly what this entry has to settle first.) Fixing it means deciding where the author's Maple
  output lives when it is not in `test/`: the loader is ~200 lines and the
  tables are 770 kB, so tracking them is cheap by the 5 MB rule, but whether
  a third party's data ships in this repository at all is not a decision the
  code can make. It is a Phase 5 item (public API, fresh clone), not a model
  one, and it is the only thing standing between `eos/enjl` and being usable
  by someone who is not the author.
- `plot/enjl_paper_figures.py` sits in a top-level `plot/` that CLAUDE.md
  section 11 does not list at all. Its styling is already correct — it imports
  `eos.general.figure_style` — so this is a location question and nothing
  else. The natural home is `notebooks/`, since section 11 says notebooks
  carry their own plotting code and this script is a notebook that never
  became one; but `plot/` also holds `plot/data/samples/`, the observational
  data the constraint overlays read, so the directory cannot simply be
  deleted and the move belongs with whatever decides that data's home.
- `docs/enjl/verify_reference_tables.py` stays in `docs/enjl/`, beside the
  document it produced. It is NOT a candidate for `eos/enjl/verify/`: it
  checks the author's TABLES rather than the model, and its whole value is
  that it depends on nothing in `eos/` — folding it into `run_full_check.py`
  would make it agree with the engine by construction and destroy the only
  independent oracle there is. It shares the fresh-clone problem above, since
  it reads the same `.dat` files.
- `notebooks/ENJL_usage.py` set `figure.dpi` and `figure.figsize` directly;
  removed under Stage 0, so this is closed (see the cross-cutting entry above).
- **The `fq0.7_B0` deconfined branch is not a root of this residual, and the
  window at n_b = 5.5757/5.6010 therefore cannot be constructed.** Measured,
  not inferred. Two causes, the second fatal:

  1. At B = 0 the deconfined *seed* cannot steer. `enjl_branch_seed`
     distinguishes a deconfined start from a restored one by the n_B^Q slot
     alone (quark fraction 1.0 against 0.2), and n_B^Q enters the residual in
     exactly two places -- `baryon_masses` as `B n_B^Q` and `Sigma^R_q` as
     (1/3) B sum n^s_i -- both multiplied by B, which is zero for this set. All
     six seeds (three densities x two branches) at mu_B = 6348.7562,
     mu_C = -100 MeV return the identical root: n_B = 5.5207 fm^-3, baryon
     fraction 7.71e-2, M_u = 5.50 MeV, g_omega omega = 1978.84 MeV.
  2. There is no deconfined root there to reach. Solving the quark-only
     subsystem at the author's own potentials (mu_B = 6348.7562,
     mu_C = -7.8747 MeV) converges to 3.4e-14 and reproduces her high
     endpoint -- n_B = 5.6004 against her 5.60098 (1.0e-4 relative),
     n_u/n_d/n_s = 5.6037/5.7339/5.4637 against 5.6009/5.7306/5.4713. But at
     that state the baryon effective potentials are nu_p = 520.76,
     nu_n = 517.76, nu_Lambda = 154.34 MeV against masses M_p = M_n = 16.50
     and M_Lambda = 151.70 MeV (the author's own printed Mp/Mn/ML -- Eq. (4)
     agrees). Every baryon is above threshold, so the full residual
     repopulates them.

  The author's table says the same from her side: at n_b = 5.60098 she reports
  mu_n = 5852.3091 MeV while mu_b = `munr` = 6348.7562 MeV, a 496.4 MeV gap,
  with n_n = 8.69e-6 fm^-3. Her baryons are not in chemical equilibrium with
  her quarks at that row; mu_n floats free once the baryon density collapses.
  This repository imposes mu_i = B_i mu_B + C_i mu_C + S_i mu_S on every
  species (CLAUDE.md section 2), so mu_n = mu_B identically and the state is
  not representable. Reproducing it needs either an explicit occupation
  restriction excluding baryons from the deconfined phase -- which
  `docs/enjl/PHASE_TRANSITION_DESIGN.md` section 1e establishes the author's
  worksheet does NOT implement -- or breaking mu_n = mu_B, which section 2
  forbids. Pinned by `test/enjl/test_enjl_construction.py`
  `test_deconfined_branch_absent_at_B_zero`, which fails if a baryon-free root
  ever appears.

  Separately and independently: the neutralizing mu_C at both endpoints of
  that window is -33.23 and -7.87 MeV (the rows carry `mue` = 0.511, the
  electron mass, i.e. no leptons -- the matter self-neutralizes), and
  `eos.mixed.boundaries.MU_C_SCAN` covers [-300, -20]. Even with a branch that
  existed, `neutral_phase` could not neutralize it without a wider scan. The
  three constructible windows all sit at mu_C in [-213, -169] and are
  unaffected, which is why the constant is left alone.
- The construction delivers eta = 1 only, **under either closure**.
  `eos.enjl.table.build_constructed_table` raises for any other eta. That
  bound is the same for the beta-equilibrium branch pair
  (`eos.mixed.construction.enjl_coexistences`) and for a phase held at a
  non-leptonic (Y_C, Y_S) (`enjl_composition_coexistences`), and the eta = 0 /
  Gibbs alternative differs more at held composition than it does in beta
  equilibrium: there the two phases may carry DIFFERENT Y_C and Y_S subject
  to the global values, all three potentials are equated rather than P and g
  alone, and the resulting window is not flat in pressure -- so it is a
  different delivered object, not a refinement of this one. An
  eta < 1 delivered table needs the mixed system solved at every density
  inside the window rather than a lever rule across it, seeded from the
  eta = 1 point; the solve itself works (measured at fq0.7_B1: chi = 0.4814 at
  eta = 0.5, n_B = 0.49 fm^-3), so what is missing is the swept table around
  it, not the physics. f(eta) is monotone decreasing in eta at every density
  measured -- f(0) - f(1) = -0.029, -0.038 and -0.031 MeV/fm^3 at
  n_B = 0.470, 0.490 and 0.510 fm^-3, about -6e-5 relative -- with no interior
  extremum, as `docs/enjl/PHASE_TRANSITION_DESIGN.md` section 5 argues for the
  endpoints. No minimizer is shipped and none should be: at interior eta both
  lepton populations exist with weights eta and 1 - eta and enter eps
  additively, so f there is not variational.
- The located windows sit systematically ABOVE the author's, by +1.0e-4 to
  +2.0e-4 in mu_B and +4.2e-4 to +8.7e-4 in P, on all three constructible
  transitions. Traced, and not a bug in the locator: at the author's own
  coexistence potentials our matter pressure is below hers on both branches
  and by a DIFFERENT amount on each (3.3e-5, 2.9e-5 and 1.49e-4 relative for
  fq1.0_B0, fq1.0_B1, fq0.7_B1), leaving P_lo - P_hi = +0.0061, +0.0060 and
  +0.0104 MeV/fm^3 where the Maxwell condition needs zero. Dividing that by
  n_lo - n_hi = -0.0222, -0.0390 and -0.0857 fm^-3 predicts mu_B shifts of
  +1.99e-4, +1.09e-4 and +1.04e-4, against +1.981e-4, +1.077e-4 and +1.030e-4
  located -- so the whole bias is that one gap amplified 11-45x by the narrow
  window. The reference's own internal consistency is the same size: the
  author's columns fail her own Euler relation P + E - sum_i mu_i n_i by
  1.7e-4 to 2.3e-4 peak-to-peak relative to P over n_b = 0.3-0.9 fm^-3.
  Ruled out by arithmetic rather than left open: `locate_maxwell`'s
  xtol = 1e-8 MeV is seven orders below the 0.12-0.28 MeV shift, and
  `thermo_from_mu`'s 1e-10 scaled residual gate is further still. The muon
  treatment is not it either -- the author populates muons at mu_mu = mu_e
  exactly, as `charged_leptons` does at mu_nue = 0, and switching them off
  moves P by 1.1-3.3 MeV/fm^3, two to three orders ABOVE the gap.

### njl

- **RG-consistent regularization is not implemented; `lambda_UV != 1` raises.**
  The medium integral is not a spectator in this model. At T = 0 and unpaired
  it is self-limiting at k_F and cutoff-free while k_F < Lambda, but that
  protection disappears at finite T and in ANY colour-superconducting phase,
  where the Fermi surface is smeared. Two consequences are measured rather
  than argued: with everything cut at Lambda the density SATURATES at
  n_B = Lambda^3/pi^2 = 2.881 fm^-3 and freezes (checked by
  `verify/run_full_check.py`); and the gap dies at mu ~ 1.13 Lambda, so a
  sharp-cutoff three-flavour CSC calculation is quantitatively safe only for
  mu << Lambda/2, which is BELOW deconfinement onset. There is effectively no
  window where lambda = 1 is trustworthy for CSC, and it is shipped for code
  validation against published sharp-cutoff results, not for production.

  The fix is RG consistency: integrate the medium term to Lambda_UV >> Lambda,
  keep the vacuum integral at Lambda, and subtract a counterterm cancelling
  the medium divergence, which is logarithmic, exists only when mu != 0 AND
  Delta != 0, and does not scale with the quark masses:
  Gamma_med/V_4 ~= -(2/pi^2) mubar^2 Delta^2 ln Lambda_UV (the coefficient is
  confirmed to 0.08% in docs/njl_csc_implementation.md section 7.2). The
  parameter `lambda_UV` and `Lambda_medium` exist and are threaded through;
  what is missing is the counterterm's closed form, which
  docs/njl_csc_implementation.md documents as unwritten -- it says of the
  three published schemes that the massive one is rejected by its own authors
  and that massless and minimal both have closed forms, without stating
  either. Writing one from Gholami, Hofmann & Buballa (arXiv:2408.06704) is
  the work. Until then any lambda != 1 raises `NotImplementedError`, because a
  lambda-dependent answer would be worse than an exception.

  Two things follow and are recorded here rather than worked around. The
  published Kunkel set (`Parameters.named("kunkel")`) ships its COUPLINGS
  (eta_D = 1.45, eta_V = 0.7) at lambda = 1 rather than their lambda ~ 10, and
  the two are not independent -- RG-consistent CFL gaps run almost 90% above
  sharp-cutoff ones -- so it is a strong-coupling point, not a reproduction of
  that paper. And the conformal asymptotics of the vector sector cannot be
  EXHIBITED at lambda = 1 at all: c_s^2 -> max(1 - alpha, 1/3) is a statement
  about n_B -> infinity and this regularization has no densities above
  2.881 fm^-3. `verify/check_sound_speed` therefore asserts causality and
  monotonicity only, and says so.

- **The paired scalar density disagrees with the specification's 2SC light
  masses, and the thermodynamic identity decides in favour of the code.**
  At the solved 2SC point (mu_B = 1500 MeV, T = 0, eta_D = 0.75)
  docs/njl_csc_implementation.md section 6 reports M_u, M_d = (11.96, 7.65)
  MeV; this implementation gives (9.73, 8.90). Everything else in that row
  agrees to the reported precision: Delta_3 = 95.50, mu_3 = 0,
  mu_8 = -2.46 MeV, M_s = 243.13, mu_C = -62.3, n_B = 1.4887 fm^-3,
  P = 324.7 MeV/fm^3.

  The difference is the SIGN of the hole amplitudes in the paired scalar
  density delta_rho_s: whether the Hellmann-Feynman derivative of the BdG
  eigenvalue with respect to M_f carries (|top|^2 - |bot|^2) or
  (|top|^2 + |bot|^2). Reproducing the specification's masses requires the
  latter, and it was reproduced exactly with it -- so this is a convention
  difference, not a coincidence.

  It was settled by n_B = dP/dmu_B along the neutral solution, which is the
  statement that the gap equation IS stationarity of Omega rather than merely
  a plausible self-consistency. As implemented it holds to 3e-7, the
  finite-difference floor; with the other sign it fails by 2.6e-4, three
  orders of magnitude worse. `test/njl/test_pairing_patterns.py` and
  `verify/check_density_derivative` both pin this, and the light masses are
  deliberately left ungated in `verify/check_anchor` with the reason in its
  docstring.

  Why it shows up only in M_u and M_d: near chiral restoration M - m is the
  small difference of two large numbers, so a percent-level change in one
  scalar density moves the light constituent masses by twenty percent and
  moves nothing else measurably. Closing this means an independent third
  calculation, or the specification's author confirming which sign
  `verify_njl_csc.py` used.

- **The 't Hooft--diquark cross-term is omitted and eta_D absorbs it.** The
  determinant term expanded in the presence of diquark condensates generates a
  term coupling |Delta_eta|^2 to the quark-antiquark condensates. Ruester et
  al. state it explicitly and neglect it; SRP absorb it into their diquark
  coupling; Kunkel et al. do not include it. Baym et al. (arXiv:1707.04966) DO
  carry it, with a coupling K' ~= K from the Fierz transformation, and their
  form is stationarity-consistent (the apparent K'/2 versus K'/4 mismatch
  between Omega_cond and the two channels is required, not a typo). So a
  coefficient is available if it is wanted. It is not implemented here, which
  means eta_D = G_D/G_S must be READ as an effective coupling that has
  absorbed it -- any paper using this code should say so. Adding it is
  Baym's Eq. set, not an invention; what it would change is that it raises
  M_i and REDUCES the effective pairing strength H -> H - (K'/4) sigma_i, with
  sigma_i negative in the broken phase, so it encourages coexistence of the
  chiral and diquark condensates.

- **The trapped muon lepton family is not a conserved charge.**
  `beta_eq_neutrino_trapped` takes (n_B, Y_Le, T) only; passing Y_Lmu raises.
  The muon SPECIES is fully available through `SpeciesFlags(muons=True)` and
  is populated in beta equilibrium above mu_e = m_mu. What is missing is the
  second lepton-family constraint and the mu_numu unknown that goes with it.
  This is the same gap alphabag has, and closing it is one more row and one
  more unknown in `_charge_rows` plus the muon-neutrino block in
  `lepton_block`.

- **`eos_response` implements only the `equilibrium` freeze.** The composition
  freezes of CLAUDE.md section 5 -- held Y_i, held Y_C -- and the one this
  model adds, held Delta, are not wired, and neither is the susceptibility
  matrix chi_ab. The gap freeze is the interesting one: holding Delta against
  its own gap equation while the density is perturbed is what a sound wave
  faster than the pairing-relaxation timescale would see, and there is no
  reason to expect it to equal the equilibrium value in a gapped phase.
  Everything returned is a finite difference along a re-solved sequence, which
  is the reference flavour; there is no accelerated flavour and none is owed.

- **A cold start stops converging around n_B = 2.1 fm^-3.** `default_guess`
  builds mu_B from the free massless relation, which knows nothing about a
  cutoff and therefore cannot see mu_B running away as the density approaches
  the saturation ceiling at 2.881 fm^-3. A warm-started sweep reaches
  2.82 fm^-3 without trouble (`verify/check_saturation_density` does exactly
  that), so this bites only a caller asking for a single high-density point
  cold. The fix is a guess that inverts the CUT density integral rather than
  the free one.

- **The mixed adapter runs a full internal solve per call, and may not cache
  its seed.** Unlike vMIT or alphaBag, whose thermodynamics is explicit in mu,
  `njl_phase.thermo` closes a Newton system (three masses, the free gaps, the
  two colour potentials, Sigma_V) at every call: about 0.15 s unpaired and
  several seconds paired. The mixed residual calls the adapter once per
  evaluation and is finite-differenced, so a window search is thousands of
  calls. `seed_cacheable=False` makes it worse and is nonetheless correct --
  the seed chooses the chiral branch, so caching it would change which state
  is found, not how fast (the ENJL rule).

  Consequences today: `test/mixed/test_njl_pair.py` locates its window once,
  on a coarse grid with `refine="bisect"`, and pins the finite-temperature
  behaviour with a single mixed solve rather than a second window search; and
  a CSC window is not attempted at all. What would fix it is a cheap analytic
  `jacobian_block` for this phase (the `Phase` contract already has the slot,
  and it is optional by design), or a seed cache keyed by the BRANCH rather
  than forbidden outright -- which needs a way for the adapter to say which
  chiral branch a seed belongs to. Performance work is profile-driven and
  comes after correctness (CLAUDE.md section 6); this entry is the profile.

- **The pairing sector is the reference flavour only; there is no fast one.**
  A paired point diagonalises a batch of 18x18 matrices at every quadrature
  node and every residual evaluation, which makes a four-pattern enumeration
  at one (n_B, T) take a few seconds against a few tenths for the unpaired
  model. That is the honest cost of the general path, and the 2SC closed form
  of `eos.general.pairing.twosc_dispersion` is already there as the obvious
  fast path for the one pattern that has one -- it is used as a unit test of
  the general path but is not yet wired as a production shortcut. Doing so is
  a `backends/` job by CLAUDE.md section 9, and it comes after correctness.

- **The dilaton / colour-dielectric graft is not implemented.**
  docs/njl_csc_implementation.md section 10 works out four graft points for
  coupling a colour-dielectric field to the NJL four-fermion interaction and
  finds that exactly one of them works (Variant II / Graft D: chi dresses the
  medium term only, with a D2 chain-rule source, B_g, p = 1, q in {0, 1}). It
  also records that there is NO published model doing this -- every dielectric
  quark-matter paper pairs the dielectric with a linear sigma model instead --
  so it would be new work, and that it is untested for transition order,
  pairing coexistence and finite temperature. Parked with that section cited.

- **The CFL branch is the least robust of the enumerated patterns.** It is
  electrically neutral without electrons, so its seed puts mu_C at zero
  (docs/njl_csc_implementation.md section 6.3 records that an electron-bearing
  seed converges to a spurious point with an 11% flavour-density spread), and
  with that seed it converges cleanly at the densities tested here and wins on
  free energy at eta_D = 0.75. What is not done is a Ginzburg-Landau-informed
  seed, which is what the specification asks for and what would make the
  branch reliable across a whole table rather than at the points checked.

### ccdm
- **A note on where this model's integrals came from.** `eos.ccdm` does not
  implement its own Fermi integrals: the split-panel Gauss-Legendre ideal gas
  it shares with `eos.njl` was lifted into `eos.general.fermi_gauss` when this
  model was written, and the pattern declarations into `eos.general.pairing`,
  so that neither model carries a copy (CLAUDE.md section 7). Both moves are
  bit-exact no-ops for NJL -- `test/baseline/njl.npz` reproduces at
  rtol = 1e-10 across them. `test/general/test_fermi_gauss.py` is the
  validation against JEL that section 7 requires of an alternative
  implementation.

- **The `fixed_YC_YS` mode at Y_S = 0 does not converge**, and the cause is
  the cross-cutting degeneracy of this document's first entry rather than
  anything specific to this model: once M*_s rises above mu*_s the strange
  density is identically zero at T = 0, the strangeness row is satisfied for a
  whole RANGE of mu_S, and that column of the Jacobian vanishes. The solve
  stalls on the threshold with mu_S wherever its path reached. `eos.njl` shows
  the same thing at n_B = 2.0 fm^-3. Seeding mu_S on the strangeness-
  SUPPRESSING side (`solver.default_guess`) shortens the path to the
  degeneracy but cannot close it; closing it means deciding what the API says
  when a conserved charge is carried by no populated species, which is the
  decision that cross-cutting entry is waiting on. Non-convergence is a return
  value, so a sampler can score the point and move on.

- **`ccdm_phase` RAISES below the quark onset, and a mixed solve whose path
  wanders there stalls.** The confined branch is excluded from the adapter's
  default enumeration because its pressure is identically zero -- it is the
  vacuum, and in a hybrid construction the hadronic phase is what occupies
  that side of the transition. The consequence is that below the onset the
  adapter has nothing to return and raises, so the mixed engine meets an
  exception rather than a smooth boundary; since the mixed residual is
  finite-differenced, the Jacobian is then undefined and the solve reports
  "not making good progress".

  Measured on DID+CCDM at T = 30 MeV, beta equilibrium: the CCDM phase
  converges at no enumerated branch for mu_B below about 1400 MeV, and the
  mixed solve consequently fails for n_B in roughly 1.3-1.8 fm^-3 while
  succeeding at 0.9, 1.1 and 2.0. The WINDOW still locates correctly
  (0.711 -> 2.189 fm^-3 at T = 30 against 0.711 -> 2.261 at T = 0), because
  the probe only needs points where the solve succeeds. At T = 0 the same
  band solves, which is why `test/mixed/test_ccdm_pair.py` can use the window
  midpoint there and picks points either side of the band at T = 30.

  Restricting to a single branch does NOT fix it -- tried, and it does not --
  so this is the domain boundary itself, not the branch-choice discontinuity
  it first looked like.

  Closing it means deciding what a quark adapter should return below its own
  onset. Returning the confined vacuum (n_B = 0, P = 0) would make the
  adapter total and let the engine push back up, at the risk of the solver
  settling on spurious chi -> 0 states; raising, as now, keeps every returned
  block a real deconfined state. The other quark adapters here never face the
  choice because they have no confinement and so no domain boundary. This is
  the first adapter that does, and the decision wants more thought than a
  session that was adding a model.

- **The onset DENSITY is not located by this model, only by `eos.mixed`.**
  The deconfined branch's pressure crosses zero very close to where the branch
  itself terminates -- the crossing moves from about 1.34 to 1.38 fm^-3
  between B_g^(1/4) = 150 and 190 MeV -- so a grid scan for the crossing finds
  one parameter point's onset and falls off the end of the branch for the
  other. `verify/check_glue_scale_stiffens` therefore tests the robust
  statement of the same physics (a larger B_g costs pressure at fixed density)
  and section 6.5 of the specification says the same thing: locate transitions
  by root-finding on pressure differences, never by an argmin over a scan
  grid. That root-finding lives in `eos.mixed.boundaries`.

- **The dilaton gradient and finite-size terms are absent.** This is a bulk,
  homogeneous mean field, so surface tension, curvature and the finite-size
  physics of a strangelet or of a mixed-phase droplet are outside it. The
  Lagrangian carries (d phi)^2 and the model does not; anything that needs a
  droplet needs it added.

- **The de Carvalho contact coupling h(phi) is NOT used for G_D**, and that is
  a decision rather than an omission (section 10 of
  docs/ccdm_implementation.md). Their construction is sound and is *why*
  L_pair is a legitimate leading term rather than an ad hoc addition -- write
  the confining field as chi_bar + delta chi, integrate out delta chi at
  Gaussian order, and a contact interaction of range 1/M_chi falls out, an
  inverse glue mass the model already carries. Translated into this model's
  variables the coupling costs no new parameter (phi_0 cancels identically).
  But what it predicts is h/G_D^Fierz ~ 3e-4 in the deconfined branch and
  ~1e4 in the confined one: negligible where the quarks are light and enormous
  where they are heavy, which removes pairing from the deconfined phase where
  a star's core lives and would condense a diquark in the confining vacuum.
  It carries q = 4 at p = 1, violating the q <= p criterion. What is shipped
  instead is the mildest dielectric dressing, G_D -> G_D/chi^q with
  q in {0, 1} as a declared discrete choice.

- **g_s, gbar_omega and n_c are not pinned by the specification.** g_q = 3.0
  IS -- its section 10 table quotes M*_(u,d) = 826 MeV at phi_bar = 0.90 and
  1531 MeV at 0.95 in the confined branch, and both invert to 3.00 -- and
  B_g^(1/4) = 150 with m_sigma = 550 are its stated baseline. The other three
  are shipped at documented mid-prior values (3.0, 4.0, 1.0 fm^-3) and
  `parameters.py` says so rather than presenting them as measurements. G_D was
  calibrated in this session to the specification's 20-150 MeV gap window.
  A real calibration against neutron-star data is the work.

  Note one consequence of the shipped n_c = 1.0 fm^-3: the vector interaction
  energy W = (1/2) m_omega^2 omega_0^2 PEAKS at n_B = n_c and falls beyond it,
  so Sigma_V changes sign there and is -118 MeV at n_B = 1.5 fm^-3, putting
  -532 MeV/fm^3 into P. That is a property of the specification's coupling
  form, not of the implementation, but it means the shipped point sits right
  on the turnover and a calibration should move n_c rather than inherit it.

- **`eos_response` implements only the `equilibrium` freeze.** The composition
  freezes (held Y_i, held Y_C, held Delta, held fields) and the susceptibility
  matrix chi_ab = dn_a/dmu_b are not wired; holding a composition needs the
  species fractions carried through the solve as constraints, and holding the
  gaps or the fields needs them fixed against their own equations. Asking for
  one raises. `eos_response` DOES return `branch_changed`, because this model
  has a first-order transition and there is no way to see from a sound speed
  alone that its stencil straddled one.

- **The muon lepton family is not a conserved charge here.**
  `beta_eq_neutrino_trapped` takes (n_B, Y_Le, T) only; Y_Lmu raises. The muon
  SPECIES is available through `SpeciesFlags(muons=True)`, and in this model's
  deconfined matter it is never populated anyway: three-flavour quark matter is
  nearly charge neutral by itself, so mu_e comes out around 47 MeV at
  n_B = 1.5 fm^-3 against m_mu = 105.7. The same gap as in `eos.njl`.

- **`ccdm_phase` has no `frozen_thermo`**, so the frozen-composition responses
  raise for any pairing that includes it -- the same gap as `njl_phase` and
  `alphabag_phase`, and for the same reason: the model exposes no
  thermo-at-given-densities surface.

- **No `backends/`.** The reference NumPy path is the only one. A paired solve
  diagonalises an 18x18 matrix at every quadrature node and the branch times
  in `test/mixed/test_ccdm_pair.py` show where that lands; if paired hybrid
  tables are actually wanted, that is the thing to profile first. Nothing has
  been profiled yet, so nothing is written (CLAUDE.md section 6: performance
  work comes after correctness).

### general
- Most of the thermal meson gas is missing from `general/particles.py`, so it
  cannot be summed as species. `pi+`, `pi-`, `pi0`, `K+`, `K-`, `K0` and `eta`
  are registered; `K0bar`, `eta'` and the whole vector nonet (`rho+-0`,
  `omega`, `phi`, `K*`) are not. Because `PhaseThermo.assemble` derives n_C and
  n_S from the species table, a gas listed in `densities` would land in the
  charge totals automatically — which is exactly what §2 wants and what the
  baryons-only bug class keeps coming from. Until the entries exist, the gas
  reaches the totals through `assemble`'s `extra_charges` argument instead: a
  second summation, correct today because `thermal_meson_charges` is
  validated, but a second path all the same. Adding the entries retires the
  argument. Watch the sign convention when doing it: S = +1 per s quark, so
  K+ = u sbar has S = -1 and K0bar = dbar s has S = +1. The check that the
  entries are right is that summing C_i n_i over the gas reproduces
  `thermal_meson_charges`.

- `fermi_integrals.py:518,524` holds the ONE unbounded loop in the package.
  Section 6 requires every solver to have a bounded iteration count, and a sweep
  of `eos/` found exactly these two `while` loops with no counter: the bracket
  expansion inside the inverse `n -> mu` solve, which multiplies `mu_hi` by 1.5
  until `n_hi >= n_target` and halves `mu_lo` until `n_lo <= n_target` before
  handing the bracket to `brentq`. Recorded rather than left silent, and recorded
  with its measurement: the growth is GEOMETRIC, so from any finite start the
  bracket clears any representable target in a few dozen iterations and a hang is
  practically unreachable -- which is why this has never been observed to bite. A
  non-finite or NaN `n_target` is the shape that would spin, and the caller-facing
  fix is a counter that returns non-convergence rather than a bound that happens
  to be large. The change rides with the T = 0 entry-point work on the same file.

### mixed
- Capability gaps of the shipped pairings, each a loud NotImplementedError
  naming the phase, never a silent skip: `alphabag_phase` has no
  `frozen_thermo` (alphabag exposes no thermo-at-given-densities surface, so
  the frozen-composition sound speed is undefined for any pairing that
  includes it); `enjl_branch_pair` has neither `wing_sweep` nor
  `frozen_thermo` (its from_n surface needs branch-consistent seeding, which
  belongs with the coming construction driver); `sfho_phase`/`zl_phase`
  wings cold-start per point (add their models' own warm_start if a wing
  shows holes). abpr is structurally excluded as a mixed-phase quark
  adapter: a single common potential, so its charge block is rank one and
  (mu_C, mu_S) have no independent meaning.
- The zl+vmit pairing's offset refinement can fail on coarse probe scans (a
  mid-window continuation gap between n_B ~ 1.1 and 1.9 at the shipped
  parameters): the mixed solve converges on either side but the probe steps
  are too large in between, so `locate_window` reports the honest
  `offset_unbracketed` instead of a wrong number. Needs pairing-specific
  probe tuning (a denser scan or a smaller first-step) when zl+vmit tables
  are actually wanted.
- The analytic Jacobian (`backends/jacobian.py`) covers only the normal slot
  layout, where chi is the unknown. The fixed-chi layout of
  `mixed_slots(..., fixed_chi=True)` — chi imposed, n_B unknown, used by
  `boundaries.solve_fixed_chi` to land on a phase boundary in one solve —
  runs on the solver's numeric Jacobian. The missing column is cheap (the
  n_B column is constant: -1/n_scale in the density row, -Y_X/n_scale in
  each conservation row, zero elsewhere, replacing the chi column), but a
  boundary location is a couple of solves per isotherm, so nothing profiles
  as needing it yet.
- (The entry claiming the hadronic phase adapter treated the thermal meson gas
  as a spectator in the charge and strangeness bookkeeping was WRONG, and is
  removed. `hadronic_phase` takes n_C and n_S from `assemble_octet`'s totals,
  which count the gas, and the mixed residual uses those totals throughout, so
  the engine has always agreed with dd2 and with CLAUDE.md §2. A comment in
  `phases.py` asserted the opposite and has been corrected; a test now pins
  the behaviour. Nothing moved.)
- `eos_response` implements two of the freezes CLAUDE.md §5 names:
  `'equilibrium'` (nothing held) and `'chi'` (the quark volume fraction held,
  and with it each phase's Y_C and Y_S). Two are not wired and raise:
  frozen per-species composition (all Y_i held, which for a hadronic phase
  with hyperons or Deltas is strictly stronger than holding Y_C and Y_S), and
  frozen conserved fractions with chi left free — which in a Maxwell window
  simply returns to the plateau, so it is only meaningful at eta < 1. The
  susceptibility matrix chi_ab = dn_a/dmu_b is not computed for the mixture
  either; the natural definition has to say whether it is taken at fixed chi.
- `eos_response(frozen='equilibrium')` returns nan outside the coexistence
  window. The mixed system still has a root there (chi runs negative or past
  one) but it is an analytic continuation, not the state — at eta = 1 it sits
  on the pressure plateau at every density. The physically continuous answer
  is the pure phase's own `eos_response`, and stitching the three into one
  curve is left to the caller; the engine does not dispatch across the
  boundary because that would mean importing model internals past the
  phase-adapter surface.
- `mixed_slots` activates the local and global lepton populations at exactly
  eta > 0 and eta < 1. Within about 1e-3 of an endpoint the just-activated
  population carries almost no weight, its potential is a near spectator and
  the Jacobian is near-singular, so a cold start can stall there. Interior eta
  on practical grids is fine; a near-endpoint eta needs a warm start from the
  eta = 0 or eta = 1 solution.

- The muon lepton family is not wired into the trapped mode: `mixed/api.py:81-84`
  raises `NotImplementedError` on a `Y_Lmu` condition, so
  `beta_eq_neutrino_trapped` takes (n_B, Y_Le, T) only. Recorded because it is
  the identical gap nine models already carry a ledger bullet for, and `mixed`
  was the one unit missing it -- not a different limitation, and not evidence
  that the engine is further along than the models it couples.

### astro/tov
- The fast backend returns a silently wrong tidal deformability when the table
  it is handed is not monotone in pressure. On the eta = 1 hybrid table, whose
  Maxwell plateau produced a few dozen round-off inversions of order
  1e-13 MeV/fm^3, it gave Lambda = 14 for a 0.94 Msun star where the scipy
  reference gave 2.8e6 — while M and R still agreed to 1e-4, so nothing else
  flagged it. CLAUDE.md §6 says non-convergence is a return value: meeting a
  non-monotone table is exactly that case and must come back as a status, not
  as a number. The cause has been removed upstream — `build_hybrid_table`
  now enforces §8 before the table is delivered — but the backend is still
  fragile to any other caller that hands it one.
- Crust table paths are absolute and machine-specific. A missing crust file
  currently degrades to no crust, which shifts M_max by ~1%; it must instead
  be an explicit argument with an informative error.
- The two TOV backends differ by about 2% in the tidal deformability on a
  hybrid EoS, and that difference converges with the resolution of the EoS
  table rather than with the central-density grid: doubling the density grid
  from 220 to 440 points takes the gap from 6.3% to 2.2% at eta = 1 and leaves
  M_max agreeing to 4e-4 Msun. Neither backend is wrong, but 2% is large for
  two integrators on the same table, and where the remaining difference comes
  from has not been chased down — the interpolation of eps(P) near the
  transition is the first suspect. `test/mixed/test_tov_backend_parity.py`
  pins the measured numbers.

- **There is no `eos/astro/tov/verify/` at all.** `ls eos/astro/tov/` returns
  `crust.py`, `rns_backend.py`, `rotating.py`, `solver.py`, `solver_fast.py`,
  `data/`, and the document pair -- no suite. Every other unit that carries
  physics has one, and this is the unit holding the integrator, the tidal
  deformability, the crust attachment and the RNS backend.

  What makes it more than a template gap: section 12 pins "the DD2 published
  NMP/**TOV** values" as golden ground truth, and `test/` is gitignored, so a
  fresh clone of the published package has NO way to check that its TOV
  integration is right. The two gaps already ledgered just above -- the ~2 %
  two-backend disagreement in Lambda, and the fast backend's silent wrong answer
  on a non-monotone table -- both name numbers that were measured in `test/`, and
  neither has a suite inside the package to be re-run from.

  What the suite owes, if the section 8 gate is read for this unit: the delivered
  table check (P non-decreasing in n_B, 0 <= c_s^2 <= 1) before integration,
  backend parity between `solver.py` and `solver_fast.py`, the DD2 published
  M_max and R_1.4, and a crust-attached against crust-free comparison now that
  the tables ship in `eos/astro/tov/data/`.
