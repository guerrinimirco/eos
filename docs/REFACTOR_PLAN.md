# REFACTOR_PLAN — target structure, per-model gaps, estimates, risks

Companion to CLAUDE.md (the specification). This file says how the repository
gets from its current state to the one CLAUDE.md describes: the target tree
for both repos, the API gap and work estimate per model, and the ranked list
of what could silently change a physics number. Working document — delete
before publication.

## 1. Target tree — eos repo

    eos/                          repo root
    ├── eos/                      the package — DIRECTORY NAME KEPT (`import eos`)
    │   ├── general/
    │   │   ├── physics_constants.py
    │   │   ├── particles.py
    │   │   ├── basis.py          NEW — (mu_B,mu_C,mu_S) <-> (mu_u,mu_d,mu_s) <-> species
    │   │   │                     maps; collects the copies now in mixed/, vmit/, zl/
    │   │   ├── fermi_integrals.py
    │   │   ├── bose_integrals.py
    │   │   ├── thermodynamics_leptons.py
    │   │   ├── thermal_mesons.py NEW — meson-gas machinery moved up from
    │   │   │                     dd2/physics/mesons.py, generalized so SFHo supplies its
    │   │   │                     couplings (ONE arXiv:1210.0400 implementation)
    │   │   ├── compose.py        MOVED from sfho/compose_loader.py — generic CompOSE
    │   │   │                     reader (kills the dd2->sfho import and the duplicate in
    │   │   │                     compare_with_compose)
    │   │   ├── tabulate.py       NEW — the ONE grid driver (mode grid x warm start x
    │   │   │                     progress callback x writer) replacing four copied
    │   │   │                     compute_tables.py stacks
    │   │   ├── table_io.py
    │   │   ├── figure_style.py
    │   │   └── constraints/      observational constraints: ONE folder, ONE module family
    │   │       ├── __init__.py   overlay API per plane (M-R, M-Lambda, P-n_B, eps-n_B),
    │   │       │                 68/95 contours AND continuous-gradient rendering,
    │   │       │                 list_available()
    │   │       ├── build.py      MOVED from plot/compute_contours.py + the fetch_gw*
    │   │       │                 downloaders — producing contours lives with the data
    │   │       └── data/         contour CSVs + band files (+ samples/, Phase-2 decision)
    │   ├── dd2/  sfho/  zl/  vmit/  alphabag/  abpr/  enjl/   one subpackage per model
    │   ├── mixed/                composite engine + explicit adapters.py (adapter contract)
    │   ├── zlvmit/               legacy, kept as own code; hygiene only
    │   └── astro/
    │       ├── tov/              MOVED from eos/tov: solver.py (reference),
    │       │                     crust.py, rotating.py, rns_backend.py,
    │       │                     backends/ (the fast integrator, §9)
    │       └── gmode/            MOVED from eos/gmode, unchanged internally
    ├── notebooks/                one usage notebook per model (.ipynb <-> .py, jupytext);
    │                             dd2/notebook_api.py DELETED, its recipes inlined
    ├── output/                   gitignored EXCEPT output/public/ (tracked, curated)
    ├── test/                     local, gitignored (+ test/baseline/ from Phase 1)
    └── docs/                     STRUCTURE.md, DEFERRED.md, eos.bib, model .tex sources

Deleted outright, one line of justification each:
- eos/sfhoalphabag/ — zero callers anywhere (grep across eos, test, notebooks,
  nucleation); the 2fam paper uses sfho + alphabag pure phases directly.
- eos/general/plotting_info.py — deprecation shim, no importers left.
- eos/dd2/notebook_api.py — ruling: notebooks use library functions, no
  notebook-API modules.
- plot/plot_tidal_contours.py, plot/plot_component_tidal.py — superseded by
  the constraints overlay API.
- sfho/compare_with_compose.py — its CompOSE reader is a duplicate; the
  comparison logic moves to sfho/verify/compose.py (dd2 pattern).
- one of the two create_custom_parametrization implementations in sfho
  (parameters.py keeps it; nuclear_saturation_properties.py's copy goes).
- build/, eos.egg-info/, .DS_Store, top-level eos_tables_DD2vMIT/ and
  sfho_tables_output/ — generated artifacts; anything worth keeping goes to
  output/public/.
- DD2_OPEN_QUESTIONS.md — mined for its pinned conventions (E2b meson
  feedback, A1 mass convention, tau_3 rule, E7-E11 numerics notes ->
  DEFERRED.md), then deleted.

## 2. Target tree — nucleation repo (Phase 6)

    nucleation/
    ├── nucleation/               package unchanged in structure (already clean)
    ├── test/                     MOVED from nucleation/nucleation/tests/, gitignored
    ├── notebooks/
    └── output/                   (already has the tracked-paper-subset pattern)

Changes: imports follow eos.tov -> eos.astro.tov; basis-map/TOV-column helpers
that duplicate eos/general/basis.py switch to importing it; README brought to
the new eos standard. The core -> tables -> analysis -> figure layering stays.

## 3. API gap and work estimate per model

The table below is the ORIGINAL Phase-0 estimate and is kept as the record of
what each model was judged to need. It is not a status board: `general`, `dd2`,
`vmit` and `mixed` are done (and their rows describe work already delivered —
including one item, the mixed "meson-neutrality fix", that turned out not to
exist; see §4.1). What remains open per model is `docs/DEFERRED.md`, and the
retrofit those four still need is under §5 below.

| model    | today                                            | to reach spec | estimate |
|----------|--------------------------------------------------|---------------|----------|
| general  | styling/constraints/table_io in place            | basis.py; thermal_mesons.py (generalize dd2's); compose.py move; tabulate.py; constraints: gradient rendering + P-n_B/eps-n_B planes + build/fetch merged in | 2-3 days |
| dd2      | solve_hadronic(mode,...) + TableSpec/build_table — already spec-shaped | thin eos_point/eos_table/eos_response wrappers with status returns; nu_* -> mu_eff_* public rename; NMP inverter rework (K_sat free, Q_sat predicted); meson module moves up; .tex | 2-3 days (+1 day .tex) |
| sfho     | 6 per-mode solvers, converged flags, own table stack | fix eta-meson energy bug (thermodynamics_hadrons.py:584); muons flag -> raise; spec API + statuses; compute_nmp/invert_nmp (NEW physics: inversion for the nonlinear-RMF functional); dedupe (reader, custom-param builder, table loader); meson gas via general/; .tex | ~1 week |
| vmit     | 4 solvers, loose 0.01 sum-of-squares gate        | spec API, residual-norm gate, tables via tabulate.py, .tex | 1 day |
| zl       | 3 solvers, loose gate                            | same treatment; fixed_YC_YS raises (no strangeness); .tex | 1 day |
| alphabag | 3 solvers + solve_cfl, own table stack           | spec API (CFL as phase selector), gate, dedupe loader, .tex | 1-2 days |
| abpr     | analytic fns + table generator                   | standalone spec API (non-beta / T>0 modes raise), .tex | an afternoon |
| enjl     | T=0 uniform + beta (branch G3 open)              | finite-T extension (cutoff-regularized finite-T integrals — new numerics), G3 branch rule, spec API, .tex | 1-2 weeks (largest physics item) |
| mixed    | solve_mixed/tables/scan, exception-based         | adapters.py contract; meson-neutrality fix in the hadronic adapter (count meson n_C/n_S, matching DD2); spec API incl. windows-in-result; response functions w/ freeze spec; ChargeSpec pickle fix (plain dict); .tex | 3-4 days |
| zlvmit   | legacy                                           | de-style plot_results (import figure_style), keep as own code | an afternoon |
| astro/tov| works, absolute /Users paths                     | move; crust.py split; crust/RNS paths configurable (arg/env, informative errors); RNS cleanup + fixes + useful additions; .tex/.md | 2-3 days |
| astro/gmode | clean, new                                    | move; relocate docs/gmode/gmode_theory.tex | half a day |
| notebooks| DD2, ENJL, DD2vMIT exist + legacy ZLvMIT         | rework DD2 (no notebook_api); new usage notebooks for sfho, zl, vmit, alphabag, abpr, tov+rotation, gmode | ~1 week total across Phases 4-5 |

Phase 4 order: general -> dd2 -> vmit -> mixed -> sfho -> zl -> alphabag ->
abpr -> enjl -> zlvmit (hygiene) -> astro/tov -> astro/gmode.

The §5 layout and the §13 naming vocabulary were settled after general, dd2,
vmit and mixed were already done, so those four are RETROFITTED before the
order resumes at sfho: dd2 first (it is the template sfho is reshaped to),
then vmit and mixed, which are small. Building six more models against a
shape that is about to change would cost more than the retrofit, and would
leave two conventions live in the repository at the same time — which is the
single thing §13 exists to prevent.

Recorded for Phase 5: the README states this repository is a rewrite of the
Mathematica and Python code written across the author's master's thesis and
PhD, carried out with the help of Claude Code.

## 4. Risk list — ranked by "could silently change a physics number"

1. Planned physics-changing fixes vs the Phase-1 baseline. Two intentional
   changes WILL move numbers: the sfho eta-energy bug fix, and the zl
   convergence-gate tightening (it reclassifies rows). The vmit gate
   tightening is done and moved nothing — the roots it now accepts are seven
   orders tighter, not different. The third originally listed here, a "mixed
   meson-neutrality fix", was DISPROVED and must not be attempted: the
   hadronic adapter always counted the thermal gas in n_C and n_S, and only a
   comment claimed otherwise. Protocol: the Phase-1 baseline freezes today's
   behaviour; each fix regenerates the affected baseline entries in its own
   commit, with the before/after delta quoted in the commit body. Everything
   else reproduces at rtol 1e-10.
2. nu_* -> mu_eff_* rename inside dd2 kernels touches the analytic Jacobian
   and the Numba kernel — the highest-value code in the repo. Pure rename,
   zero numeric change; guarded by the golden SNM point, the backend-parity
   gate, the M-suite, and the baseline.
3. Meson machinery move (dd2/physics/mesons -> general/thermal_mesons) and
   sfho rerouting. The E2b charge feedback and the kaon omega/rho shifts must
   reproduce exactly. Guard: baseline + test_dd2_m7 +
   test_thermal_meson_feedback + a new sfho meson test.
4. NMP inverter rework changes what from_nmp returns for the same inputs. Old
   closure kept available; new one validated by forward-map round trips; the
   (K_sat, Q_sat) ridge note in mixed/scan docs updated.
5. Table-writer unification changes on-disk column layouts read by the ZLvMIT
   notebook and nucleation's test fixture. Guard: keep legacy readers where
   needed; nucleation golden/regression.json is the tripwire.
6. eos/tov -> eos/astro/tov breaks imports in nucleation, gmode, notebooks.
   Mechanical; caught by both import sweeps; eos side fixed in the same
   commit, nucleation in Phase 6.
7. Crust/RNS path configurability: a silent fallback to crust='No' shifts
   M_max at the ~1% level. Results must state which crust was used; the
   baseline includes a crust-less TOV row.
8. Contour regeneration from raw samples can differ from shipped CSVs (KDE
   subsampling). Fixed RNG seed retained; regenerated CSVs byte-compared
   before replacing.
9. Golden values and the reference/fast split are load-bearing constraints,
   not risks to manage around: golden values are never edited (the A1
   average-mass convention stays the kernel default or they fail); no fast
   path is changed without its parity test; no tolerance is loosened, ever.

## 5. Internal structure per model (proposed)

Standard shape (CLAUDE.md §5), mandatory names, conditional existence:

    parameters.py  species.py  thermodynamics.py  solver.py  table.py  api.py
    verify/  <model>.tex+.md
    couplings.py*  nmp.py*  responses.py*  backends/*

The two rules that decide what goes where — *thermodynamics computes from the
state, solver finds the state*, and *backends/ is deletable* — are stated with
their tests in CLAUDE.md §5.

- dd2 is the template and was retrofitted FIRST, because sfho is reshaped to
  it. DONE: `physics/` is `backends/{jacobian, kernel_numba, responses_jac}`;
  `physics/{thermo, fields, mesons}` and the non-residual half of `octet` are
  `thermodynamics.py`; `octet_residual`, `assemble_octet` and
  `beta_eq_residual` are in `solver.py`; `coefficients.py` is `responses.py`;
  `nmp.py` + `nmp_inverter.py` are merged with `from_nmp` a free function
  there rather than a classmethod on Parametrization (which needed a deferred
  import to break the cycle); `xp.py` is gone.

  The mode block (`charge_mode`, `Y_C`, `strange_mode`, `Y_S`, `lepton_mode`,
  `Y_L`, `yc_leptons`) came off `OctetCtx`, which dissolved into
  `thermodynamics.MatterCtx`; `octet_residual(x, ctx, spec)` now takes the
  `ModeSpec` as an argument. `solve_octet` keeps its keywords — they are the
  public vocabulary and appear in notebooks — and turns them into a spec in
  one place, `solver.mode_spec`.

  `EoSPoint` took the hard break the records were built for: the nucleon
  fields `n_n, n_p, m_eff, mu_eff_n, mu_eff_p, mu_n, mu_p, mu_L` are gone, and
  every active baryon is carried the same way through `composition`,
  `mu_eff_i` and `m_eff_i`, read by `.n(name)`, `.mu_eff(name)`,
  `.m_eff(name)`. `.mu(name)` derives mu_i = B_i mu_B + C_i mu_C + S_i mu_S
  from section 2's basis rather than storing it. No aliases were left behind,
  so a missed call site raises AttributeError instead of reading a stale
  scalar.

  `backends/` being deletable is now a measured fact rather than a claim:
  nothing in it is re-exported from `eos/dd2/__init__.py`, and both
  `solver.py` and `thermodynamics.py` import it under `try/except ImportError`
  so a missing directory lands on the NumPy reference path. Removing it leaves
  every EoS baseline bit-identical at rtol = 1e-10; see DEFERRED for the one
  quantity that does move (the TOV sequence, by 5e-07) and why.

  STILL OUTSTANDING: `notebook_api.py` goes, and
  `_yc_neutralizing_leptons` — model-independent, now
  `thermodynamics.neutralizing_leptons` with its alias dropped at all three
  consumers — moves to `general/thermodynamics_leptons.py`.
- sfho: reshaped to the dd2 pattern — nuclear_saturation_properties.py
  becomes nmp.py; comparison logic becomes verify/compose.py; meson gas
  through general/thermal_mesons.py. No couplings.py: SFHo's g_i are
  constants and its density dependence is nonlinear self-interaction terms,
  which are thermodynamics.
- zl / vmit / alphabag: lose their private table stacks to general/tabulate.py
  and gain a minimal verify/run_full_check.py each.
- abpr: solver.py + api.py.
- enjl: eos_beta.py + uniform.py merge into solver.py; thermodynamics.py grows
  finite-T; gains verify/.
- mixed: has adapters.py, api.py and responses.py (same role, same name as
  dd2's). DONE: the electron lepton fraction is `Y_Le` and its potential
  `mu_nue` everywhere in dd2 and mixed — CLAUDE.md §2's names — and both
  `api.py` translation layers are deleted. `Y_L` was only unambiguous while
  the muon family was untrackable, and §3 already allows `Y_Lmu`.

  ALSO DONE: `ChargeSpec` is a `ModeSpec` plus a `Locality` per charge, so §3's
  four modes are declared once, in `general/`. `Regime` survives as the
  composition of the two — not held is NOT_CONSERVED, held is GLOBAL or LOCAL —
  and is a property, so the ~25 sites that read `spec.C is Regime.GLOBAL` did
  not move. What went with the mode are the `targets` field, its validation,
  the `yc_leptons` flag and the pickling dance: all four now live once, on
  `ModeSpec`. `B` GLOBAL stopped being a checked invariant and became a
  structural one — it is a property with no field behind it, so there is
  nothing to set wrongly.
- astro/tov: crust handling split out of solver.py into crust.py — not
  because solver.py is long (§13 forbids that reason) but because stitching a
  crust EoS onto a core table is separable physics from integrating the TOV
  equations, and the crust is what a caller chooses per run. The fast
  integrator moves to `backends/` for the same reason a model's does (§9),
  since the two are pinned against each other by a parity test. Rotating/RNS
  stay inside tov (shared EOSTable_for_TOV type, crust plumbing, and the
  static cross-check that validates RNS).
- astro/gmode: unchanged.

### 5a. Two structural decisions that the model re-cut depends on

**The mode declaration is shared, in `general/`.** A mode is one binary choice
per conserved charge: either its fraction is imposed and its potential is an
unknown (FIXED), or its potential is set by an equilibrium relation and the
fraction comes out (EQUILIBRATED). The field equations are never
mode-dependent. So the three families a solver needs — field equations,
conserved-charge relations, equilibrium relations — are each written ONCE, and
the residual, the unknown vector and the analytic Jacobian are assembled by
reading the declaration. No per-mode residual functions anywhere.

`eos/mixed` already works this way and is the proof: its `ChargeSpec` /
`Regime` drive `mixed_slots`, `mixed_residual` and `mixed_jacobian`, and its
four named modes carry no per-mode code. dd2 does the same thing by branching
inline on `ctx.charge_mode` / `has_muS` / `has_muL`, which is what welded the
mode block onto `OctetCtx`.

The declaration therefore moves to `general/`, where §7 already puts the
conserved-charge machinery, and every model reads it. `eos/mixed` keeps its
own on top: a two-phase system refines FIXED into GLOBAL (conserved on the
volume average) and LOCAL (conserved inside each phase), and EQUILIBRATED is
mixed's NOT_CONSERVED. Open when writing it: whether mixed's `ChargeSpec`
contains the shared declaration or subclasses it, and whether the shared name
is `ChargeSpec` (mixed's, already public) or something else.

**A model's thermodynamics owns its own field solve.** Thermodynamics has two
layers, and both belong to the model:

    evaluation      at GIVEN fields and potentials -- no solve. What the
                    residual calls on every iteration.
    self-consistent at given CHARGE POTENTIALS, solving the model's own
                    fields (and, for a phase of a mixture, its own density).
                    A solve, but of nothing mode-dependent.

`eos/mixed/adapters.py:hadronic_phase` is the second layer for DD2, written
outside dd2 with its own residual, seed and gate -- a second implementation of
DD2's field solve that can drift from the first. dd2/thermodynamics.py gains
`thermo_at_potentials(par, flags, mu_tilde_B, mu_C, mu_S, T, x0)` and the
hadronic adapter becomes a thin call that maps the result into `PhaseThermo`.
The adapter contract then IS the thermodynamics/solver boundary rather than
merely resembling it. The mixed hot path runs through here, so the warm start
(`x0`, and the per-solve constant seed) has to survive the move intact.

**State bases.** The per-species kernel is primitive; the conserved-charge
entry is that composed with §2's basis map from `general/basis.py`. Both are
public because they answer different physics: species potentials assume
nothing and are what a frozen-composition response function needs, charge
potentials assume strong equilibrium and are the normal case. `thermo_from_n`
inverts per species and is thermodynamics too.

Function names are the §13 vocabulary — `thermo_from_mu`, `thermo_from_n`,
`kinetic_thermo`, `assemble`, `residual`, `default_guess`, `warm_start`,
`solve_<mode>`, `build_table`, `Parameters.default()` — and no name repeats its
package (`compute_zl_thermo_from_mu` -> `thermo_from_mu`). These are public
names: each model's renames land in that model's own commit with every call
site fixed alongside, no aliases. `nucleation`'s imports follow in Phase 6.
