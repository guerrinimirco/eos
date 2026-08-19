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

### The CompOSE reader creates a cycle, and moving it is not just a move

`eos/sfho/compose_loader.py` imports `EOSTable_for_TOV` from `eos.tov.solver`,
and `eos.tov.solver` imports `SFHOComposeLookup` back from it — a genuine
import cycle, currently worked around with a lazy import inside the function
and a comment saying so. On top of that, `eos/dd2/verify/compose.py` imports
the same module, so a model depends on another model.

Both are violations of the layering in CLAUDE.md section 1, and the plan's
remedy — move it to `general/compose.py` — cannot be applied literally,
because `general/` may import nothing else in the repository and the reader
currently returns an `EOSTable_for_TOV`.

The shape the move has to take: `general/compose.py` reads a CompOSE table and
returns plain arrays (P, eps, n_B); the crust-table wrapper `to_crust_table`
moves to `astro/tov`, which is the layer entitled to know about
`EOSTable_for_TOV`. That resolves the cycle in the right direction — general
produces data, astro consumes it — and drops the `dd2 -> sfho` edge. It should
be done in the `astro/tov` session, where the crust path has test cover: a
silent fall back to no crust shifts M_max by about 1%.

### One notebook still sets rcParams

`notebooks/ENJL_usage.py` sets `figure.dpi` and `figure.figsize` directly. It
is a per-notebook display preference rather than house style, and the file is
jupytext-paired, so changing it means changing the .ipynb in the same edit.
Left for the notebook rework, which rewrites both halves anyway. Every module
in `eos/` and in `nucleation/` now goes through
`eos.general.figure_style`.

---

### `state.EoSPoint` and `LeptonThermo` are written but adopted by nobody

`eos/general/state.py` holds the records every model is meant to hand back —
`PhaseThermo` (matter only: the model's own fields under its own names, EVERY
active species' density, mu_i, mu_eff_i and m_eff_i per species, the conserved
charges, P, eps, s), `LeptonThermo` (all four potentials explicit, so a
transparent muon family is a visible assumption rather than a hidden
`mu_mu = -mu_C`), and `EoSPoint` (what a mode returns). `eos/general/modes.py`
holds `ModeSpec`, the mode as one choice per conserved charge.

Where this actually stands: `PhaseThermo` is adopted by dd2 and sfho, whose
`thermodynamics.py` both return it, and `ModeSpec` by dd2, sfho and mixed.
`EoSPoint` and `LeptonThermo` are adopted by NOBODY — dd2's solver returns a
flat `EoSPoint` of its own, and sfho now returns the same flat record, field
for field, because two models returning one shape is worth more than one model
leading the way to a third. (`eos/mixed`'s second `PhaseThermo` is gone: the
engine and its adapters now consume `eos.general.state.PhaseThermo` directly,
and dd2's block crosses the phase-adapter surface without re-packaging.)
zl is a third: its solver returns a flat
`EoSPoint` too, carrying (n_p, n_n) rather than a composition map, because a
two-species model has nothing to iterate over. It is a smaller record than
dd2's and sfho's, not a different convention, and it converges with them when
the shared record is adopted.

Converting to the shared `EoSPoint` is therefore ONE commit across dd2 and
sfho together, not a per-model step: doing it in either alone puts two record
shapes in the repository at the same time, which is the single thing section
13 exists to prevent. What it buys is the nesting — `point.matter` as a
`PhaseThermo` and `point.leptons` as a `LeptonThermo`, with the totals on top —
and it costs both models' baselines a key rename. What it must keep is
`converged` / `error` on the record: sfho reports non-convergence as a return
value at every layer, and its table sweep reads the flag point by point, while
dd2 raises and wraps at `api.py`. `LeptonThermo` also has no home in dd2 or
sfho until the muon family is wired.

The rest of this entry is the design the flat records already implement.

What each model has to supply is decided by ONE question — what its internal
self-consistent solution is — because the rest of the state is (chemical
potentials, T) in every model:

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
solver, not a statement about the state — physically vmit's state is still
(mu_q, V, T). So `fields` holds whatever the model solves for, under the
model's own names, and is empty where there is nothing to solve.

Three things the records fix that the remaining models get wrong in the same
way, so they are worth doing together rather than one model at a time. dd2 is
converted on all three and is the worked example:

- **nucleons are privileged over hyperons.** Every model's kinetics computes
  mu_eff and m* for each active species and then keeps only n and p. Neither
  is recoverable from the record afterwards (they need the fields and the
  per-species coupling ratios), so g-modes and the response functions recompute
  them. `m_eff` is even singular although m*_i differs per species — in DD2Y
  at n_B = 0.6 the neutron sits at 192 MeV while Lambda is at 652 and Xi- at
  1083, so the single number was the neutron's and the rest were discarded.
- **the mode is welded into the state.** dd2 carried eight mode fields on its
  context and branched on strings inside the residual; the other models do the
  same with their own vocabularies, which is why no two of them accept a mode
  the same way.
- **the non-leptonic charge is called `mu_Q` in some models and `mu_C` in
  others.** §2 says C. dd2 is converted; the rest are not. Storing mu_C rather
  than recovering it as mu_p - mu_n is also the numerically better choice, and
  measurably so: those are two ~1300 MeV numbers differing by ~100 MeV, so the
  subtraction costs about two digits, and removing it from dd2's warm start
  took its backend parity from 3.0e-14 to 8.9e-16.

Each model's conversion lands in its own commit with its baseline re-run,
since the records change what `eos_point` returns.

---

### The module names are standardised, and most models have not been renamed

CLAUDE.md §5 fixes one name per role — `parameters.py`, `species.py`,
`thermodynamics.py`, `solver.py`, `table.py`, `api.py`, `verify/` — with the
names mandatory and their existence conditional. Two renames are done
(`dd2/parametrization.py` → `parameters.py`, `vmit/thermodynamics_quarks.py` →
`thermodynamics.py`). The rest are outstanding, each belonging in its model's
own session where the baseline and `test_imports.py` are already being run:

    dd2       delete notebook_api.py -- the last one outstanding. The rest of
              dd2's layout is now CLAUDE.md §5: physics/{thermo, fields,
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
              than a string. What is left is the records (above) and the
              parameter dataclass name, `SFHoParams` -> `Parameters`.
    vmit      eos.py -> solver.py
    zl        DONE: eos.py -> solver.py, compute_tables.py -> table.py,
              thermodynamics_nucleons.py -> thermodynamics.py, and species.py
              and api.py added. `ZLParams` is `Parameters` and
              `get_zl_default()` is `Parameters.default()` -- the first model
              to take the section 13 name, so sfho's `SFHoParams` and vmit's
              `VMITParams` are now the two that differ.
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

`vmit/eos.py` is held back only because `notebooks/DD2vMIT_general1oPT`
imports from it directly and that notebook has uncommitted work in it; the
rename and the notebook's import line have to move together.

`vmit/compute_tables.py` is the one deliberate exception to the scheme: it is
the first-generation settings-object interface, kept because the ZLvMIT
notebook drives vMIT through it, and it now sits beside `table.py` as a shim
over the shared driver rather than being renamed to it.

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
per baryon. At T = 0 they agree; at T = 50 MeV they do not. Returned names
should say which — `cs2_isothermal` against `cs2_adiabatic`, never a bare
`cs2` whose meaning depends on the arguments.

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

## Per model

### dd2
- `table.hadronic_row` emits a Y_C and Y_S that are BARYONS ONLY, while the
  `EoSPoint` it flattens carries the totals. It recomputes them itself:

      _, n_C, n_S = hadronic_charges(flags, p.composition_map)

  and `composition_map` holds baryons alone, so a thermal meson gas is dropped.
  Its own docstring says the row is "keyed exactly the way
  `eos.mixed.composition_row` keys a mixed point, so a pure-hadronic table and
  a hybrid table concatenate without renaming anything" — but the mixed side
  uses the totals, so the same column name carries different physics on the
  two sides, differing by 10–20 percent at T = 40 MeV with pions. This is the
  defect class already fixed once in `sound_speed_frozen_hadronic` (b83d162),
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
  `thermo_at_potentials` now makes the reference cheap to write — perturb
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
  only: c_s^2, C_V, C_P, chi_ab) and `composition` (nucleonic Y_p: adiabatic
  c_s^2 and Gamma). Not yet wired: frozen conserved fractions (Y_C, Y_S fixed
  with species re-equilibrating), the leptonic re-neutralization variants,
  the thermal index through the API, and `equilibrium` for the other modes.
  All raise naming this file.
- The muon lepton family is not tracked in the trapped mode:
  `beta_eq_neutrino_trapped` takes (n_B, Y_Le, T) and Y_Lmu raises.
- `fixed_YC_YS` with neutralizing leptons (`leptons=True`) is not wired; the
  flag applies to `fixed_YC` only.
- Species-flag naming: the spec calls the meson switch `thermal_mesons`;
  dd2's `SpeciesFlags` carries the finer `include_pseudoscalars` /
  `include_thermal_vectors` pair (and `neutrinos` for the trapped mode, where
  the spec name `thermal_neutrinos` means the untracked mu = 0 gas, which dd2
  does not implement). Unifying the names across models is deferred until the
  other models reach the spec API, so it lands as one rename, not five.

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
  `create_custom_parametrization` has to be re-run on the result to hold
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

### zlvmit
- `get_default_guess` calls the ZL fixed-fraction and trapped solvers
  with T and the fraction transposed -- `solve_pure_H_fixed_yc(n_B, T, Y_C,
  ...)` where the signature is `(n_B, Y_C, T, ...)`, and the same in the
  trapped branch. The call sits inside a `try` whose `except` falls back to
  the analytic guess, and it only ever produces a SEED, so it cannot change a
  converged answer; found while renaming the ZL entry points and left alone,
  because zlvmit is kept for its published results and even a seed change
  moves the last bits of a baseline row.

### enjl
- **Finite temperature is implemented; what is NOT is the CONSTRUCTION above
  T = 0.** All four modes solve at any T >= 0, entropy per baryon is accepted
  wherever a temperature is, and photons and thermal_neutrinos are selectable
  mu = 0 sectors. `build_constructed_table` and `eos.mixed.construction`
  (`enjl_coexistences`, `enjl_phase`, `locate_maxwell`, `neutral_phase`) still
  raise for T != 0, and closing that is its own session:

    - locating a coexistence at T > 0 equates the GIBBS FREE ENERGIES of the
      two branches, not P and mu_B alone, so the entropy enters the
      coexistence bookkeeping;
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
- The neutralizing leptons of a held-Y_C mode are found by a local bracketed
  1-D solve, `solver.neutralizing_leptons`. That is the THIRD copy of this
  idea in the repository -- `dd2/physics/octet.py::_yc_neutralizing_leptons`
  is the first and `eos/mixed` imports it out of dd2's kernels -- and the
  cross-cutting entry above already says it belongs in
  `general/thermodynamics_leptons.py`. Whoever moves it should absorb all
  three; this one is written against the shared Fermi integrals, so the move
  is a deletion here rather than a rewrite.
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
- **The notebook and the figure script do not run from a fresh clone.**
  `notebooks/ENJL_usage.py` (and its jupytext-paired .ipynb) and
  `plot/enjl_paper_figures.py` both `sys.path.insert` into `test/enjl` and
  import `PARAMETER_SETS`, `load_reference`, `solved_rows`, `bad_rows` and
  `baryon_potential` from `test/enjl/reference`. `test/` is gitignored, so a
  fresh clone has neither the loader nor the five `.dat` files, and both
  scripts fail at import. Fixing it means deciding where the author's Maple
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
- `notebooks/ENJL_usage.py` still sets `figure.dpi` and `figure.figsize`
  directly (see the cross-cutting entry above); it is a per-notebook display
  preference and is left for the notebook rework.
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
- The construction delivers eta = 1 only.
  `eos.enjl.table.build_constructed_table` raises for any other eta. An
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
- `_yc_neutralizing_leptons` — the electron/muon gas at the chemical potential
  that neutralises a given charge density — lives in `dd2/physics/octet.py`
  but is fully model-independent: its arguments are (target charge, m_e, m_mu,
  include_muons, T) and no model parameter enters. CLAUDE.md §7 makes lepton
  thermodynamics a `general/` responsibility, and `eos/mixed/coefficients.py`
  already imports it out of dd2's kernels to build its own lepton block. It
  belongs in `general/thermodynamics_leptons.py`.

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

### astro/tov
- The fast backend returns a silently wrong tidal deformability when the table
  it is handed is not monotone in pressure. On the eta = 1 hybrid table, whose
  Maxwell plateau produced a few dozen round-off inversions of order
  1e-13 MeV/fm^3, it gave Lambda = 14 for a 0.94 Msun star where the scipy
  reference gave 2.8e6 — while M and R still agreed to 1e-4, so nothing else
  flagged it. CLAUDE.md §6 says non-convergence is a return value: meeting a
  non-monotone table is exactly that case and must come back as a status, not
  as a number. The cause has been removed upstream — `build_mixed_eos_table`
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
