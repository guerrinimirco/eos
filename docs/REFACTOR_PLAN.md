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
    │       │                     solver_fast.py, crust.py (split out of solver.py),
    │       │                     rotating.py, rns_backend.py
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

Recorded for Phase 5: the README states this repository is a rewrite of the
Mathematica and Python code written across the author's master's thesis and
PhD, carried out with the help of Claude Code.

## 4. Risk list — ranked by "could silently change a physics number"

1. Planned physics-changing fixes vs the Phase-1 baseline. Three intentional
   changes WILL move numbers: the sfho eta-energy bug fix, the zl/vmit
   convergence-gate tightening (reclassifies rows), and the mixed
   meson-neutrality fix (T>0 with thermal mesons). Protocol: the Phase-1
   baseline freezes today's behaviour; each fix regenerates the affected
   baseline entries in its own commit, with the before/after delta quoted in
   the commit body. Everything else reproduces at rtol 1e-10.
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

Standard shape (CLAUDE.md §5): parameters.py / thermodynamics*.py / solver.py
/ api.py (the three spec entry points) / verify/ / <model>.tex+.md.

- dd2: keeps physics/ (residual, jacobian, kernel_numba, octet);
  coefficients.py + coefficients_jac.py merge into responses.py (FD flavor =
  reference, Jacobian flavor = fast).
- sfho: reshaped to the dd2 pattern — nuclear_saturation_properties.py
  becomes nmp.py + nmp_inverter.py; comparison logic becomes
  verify/compose.py; meson gas through general/thermal_mesons.py.
- zl / vmit / alphabag: lose their private table stacks to general/tabulate.py
  and gain a minimal verify/run_full_check.py each.
- abpr: one eos.py + api.py.
- enjl: keeps its layout; thermodynamics.py grows finite-T; gains verify/.
- mixed: gains adapters.py; coefficients stay (they are the response-function
  reference for the freeze-spec API).
- astro/tov: crust handling split out of the 1288-line solver.py into
  crust.py; rotating/RNS stay inside tov (shared EOSTable_for_TOV type, crust
  plumbing, and the static cross-check that validates RNS).
- astro/gmode: unchanged.
