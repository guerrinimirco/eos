# eos + nucleation cleanup — phased prompts

Working notes, not part of the library. Delete before publishing.

Run one phase per Claude Code session, started from
`~/Desktop/Research/Python_codes` so both repos are in scope. Each phase ends
with a commit and a green test suite. Do not start phase N+1 until phase N is
committed.

Use the same model for every phase so the style stays consistent. Use a
1M-context variant for Phase 0 — the whole `eos/eos/` package (~39k lines)
fits in context, which is what makes a real audit possible instead of a
guessed one.

Environment trap, worth pasting into any phase that runs code: a stale `eos`
in site-packages shadows the repo when a script is run by path. Run
`pip install -e .` from the repo, or run things as `python -m`, and check
`eos.__file__` points into the working tree before trusting a test result.

| Phase | What | Interactive? |
|---|---|---|
| 0 | Read everything → agree conventions → agree structure | **yes, heavily** |
| 1 | Freeze a numerical baseline | no |
| 2 | Repo hygiene + the approved moves | one decision |
| 3 | One figure file + observational constraints | no |
| 4 | Per model: write the .tex, then refactor | per-model check-ins |
| 5 | Public API, README, STRUCTURE.md | no |
| 6 | Propagate to nucleation | no |

---

## Phase 0 — Read, then decide together

> Use plan mode. Three steps, with a hard stop between each. Nothing is
> written until step 2, and no code changes at all in this phase.

```
This is a discussion, not a task. You will stop twice and wait for me.

STEP 1 — READ. Read all of `eos/` — every subpackage of eos/eos/, the tests,
the notebooks, CLAUDE.md, DD2_OPEN_QUESTIONS.md — and then all of
`nucleation/`. Then report back, concisely:

  a) INVENTORY. One row per subpackage of eos/eos/: lines, public entry
     points, physics implemented, which eos modules it imports, and whether
     nucleation/ or the notebooks import it.

  b) MODE COVERAGE. Rows = models. Columns = the four modes listed under
     step 2 below, plus the leptons on/off flag. Each cell: implemented /
     partial / absent / physically meaningless for this model. This matrix
     tells me how big the "uniform API" job actually is — it is the single
     most useful thing you can give me in this phase.

  c) SPECIES COVERAGE. Rows = models, columns = nucleons, hyperons, deltas,
     thermal mesons, photons. Same cell vocabulary. Note where a model has
     the physics under a different flag name.

  d) DUPLICATION. Every place a model implements its own Fermi or Bose
     integral, lepton thermodynamics, particle property, constant, or figure
     styling instead of using eos/general/. For each hit classify it:
     legitimate call into general/, a local re-implementation that must go,
     or a genuinely different physical quantity that only looks similar.
     Check, do not guess — unifying things that were never duplicates is how
     physics silently breaks.

  e) DEAD CODE. Functions, modules and files with no caller anywhere in eos,
     nucleation, the notebooks or the tests. Give the evidence.

  f) INFERENCE READINESS. These models will eventually be driven by a Bayesian
     sampler and by ML surrogates, so per model report: are the model
     parameters arguments or module-level constants? What happens on
     non-convergence — a status flag, an exception, or an unbounded loop? Is
     there global mutable state or module-level caching? Does the model object
     pickle? Roughly how long does one eos_point call take? This tells me
     whether inference is a small adaptation or a rewrite.

  g) OPEN QUESTIONS. A numbered list of every convention or design choice you
     cannot settle from the code — places where two models disagree, where a
     name is used for two different quantities, or where the physics is
     genuinely ambiguous. Do NOT pick a convention silently to keep moving.
     This list is what I answer before you write anything.

Then STOP. Do not write CLAUDE.md yet.

---

STEP 2 — CLAUDE.md, after I have answered your questions.

Rewrite eos/CLAUDE.md as the SPECIFICATION the repo will be refactored to
match. Present tense, as if the repo already complies. Nothing marked
"planned" or "TODO".

Keep from the existing file only the physics invariants — the dependency
direction rule (eos never imports nucleation), the units rules, the _ref/_fast
contract, the thermodynamic-consistency checks (Euler relation, f = eps - Ts,
Sigma^R enters mu and P but never eps, HVH, 0 <= c_s^2 <= 1) — and the rule
that docstrings are self-contained with no references to plans, milestones or
docs/ files. Everything else in the current CLAUDE.md is yours to restructure,
shorten, reorganise or drop; it describes a repo that is about to change.

Write in the following. This is dictated — record it faithfully, do not
paraphrase the physics or invent extra conventions.

CONSERVED CHARGES. The general basis is B, C, S, e, mu, nu_e, nu_mu:
  B   baryon number.
  C   electric charge of strongly-interacting matter ONLY — baryons, quarks
      and charged mesons. Leptons are excluded from C.
  S   strangeness, S = +1 per s quark. The s quark has S = +1, Lambda has
      S = +1, Xi has S = +2. This is the OPPOSITE of the PDG convention and
      is used consistently throughout. Never silently flip it.
  e, mu, nu_e, nu_mu   individual lepton species.
A reduced basis B, C, S, L_e, L_mu may be used instead, where L_e and L_mu are
the lepton family numbers. Legitimate because total electric neutrality ties
the lepton content to C.

Y_X = n_X / n_B for every charge: Y_C, Y_S, Y_Le, Y_Lmu. Always relative to
n_B, never to total particle number.

Y_C is NON-LEPTONIC. Total electric neutrality (n_C = n_e + n_mu) is a
separate, additional condition that a mode may or may not impose. Conflating
the two is the most common error in this domain and the API must make it
impossible to do by accident.

Chemical potentials: mu_i = B_i mu_B + C_i mu_C + S_i mu_S + lepton terms.
Species potentials are derived, never independent unknowns. Solvers use
kinetic potentials nu_i = mu_i - Sigma0_i for the reasons already documented.

MODES. Every model exposes the same modes; a mode fixes the independent
variables:
  BETA_NEUTRINOLESS  (n_B, T)
      Beta equilibrium, neutrinos free-streaming (mu_nu = 0), charge neutral.
  BETA_TRAPPED       (n_B, Y_Le, Y_Lmu, T)
      Beta equilibrium, trapped neutrinos. Muon family optional: with it
      disabled the mode takes (n_B, Y_Le, T).
  FIXED_YC           (n_B, Y_C, T)
      Fixed non-leptonic charge fraction. This produces simulation tables.
  FIXED_YC_YS        (n_B, Y_C, Y_S, T)
      Fixed charge and strangeness. Y_C = 0.5, Y_S = 0 is symmetric nuclear
      matter, for heavy-ion comparisons.

One orthogonal flag, applying to FIXED_YC and FIXED_YC_YS:
  leptons=True   electrons — and muons if enabled — are added to enforce
                 total electric neutrality, contributing to eps, P and s.
  leptons=False  strongly-interacting matter only; the result is electrically
                 charged. This is what a mixed-phase construction needs for
                 each pure phase before imposing GLOBAL neutrality.

SPECIES FLAGS. Nucleons (n, p) always present. Every other degree of freedom
is an explicit named boolean, with identical names across all models:
  hyperons        Lambda, Sigma, Xi
  deltas          Delta(1232)
  thermal_mesons  pi, K. These carry C and S and therefore enter the charge
                  and strangeness bookkeeping, not only eps, P and s.
  photons         contribute to eps, P and s only; carry no conserved charge.
No sector is enabled or disabled implicitly because "its coupling happens to
be zero" — if a sector is off, its flag is False. Setting a flag a model does
not implement RAISES; never turn a NotImplementedError into a silent no-op.

UNIFORM MODEL API. Every model exposes the same two entry points with the
same signature:
    eos_point(mode, species, **conditions)  -> quantities at one point 
    eos_table(mode, species, grid)          -> tabulated EoS over a grid
`conditions` are the independent variables of the mode, named exactly n_B, T,
Y_C, Y_S, Y_Le, Y_Lmu. Every public boundary is fm-based: n in fm^-3, T and mu
in MeV, eps and P in MeV/fm^3. Natural units stay inside the physics modules.
What eos_table() returns is directly consumable by eos/astro/tov/ and by the
plotting code with no per-model adapter.

WHAT THE LIBRARY IS FOR. The API is shaped by its downstream uses, and a
design choice that makes any of them awkward is the wrong choice:
  1. Generating tables for astrophysical simulations.
  2. Figures of thermodynamic quantities, and TOV / stellar structure.
  3. Bayesian inference over model parameters — future, but designed for now.
  4. Machine-learning surrogates and emulators — future.
  5. Downstream physics packages; nucleation is the first.

Four requirements follow from 3 and 4, and they are not optional:

  MODEL PARAMETERS ARE ARGUMENTS, never module-level constants or globals.
  Inference varies couplings, nuclear-matter parameters and B across millions
  of calls; a parameter that can only be changed by editing a source file
  makes inference impossible. Published parameter sets are named defaults,
  not hardcoded values.

  NON-CONVERGENCE IS A RETURN VALUE, not an exception and never a hang. A
  sampler walks into unphysical parameter space constantly and must be able to
  score that point and move on. Every solver reports a convergence status the
  caller can test, and has a bounded iteration count. This is the most common
  reason an inference pipeline built on an EoS code fails.

  NO GLOBAL MUTABLE STATE. Same inputs, same outputs. Model objects are
  picklable so multiprocessing and MPI work. Two models with different
  parameters coexist in one process without interfering.

  ARRAY IN, ARRAY OUT wherever the physics allows. eos_table() takes grids and
  returns arrays. A table must not be a bare Python loop over eos_point() that
  the caller could have written — unless the solver genuinely needs the
  previous point as a warm start, in which case say so in the docstring.

Differentiability is NOT a requirement. Do not reopen the JAX question — the
hand-coded analytic Jacobian is the shipped design and that was a physics
decision, not an unfinished task. But do not make a future autodiff or
surrogate path structurally impossible either.

INTEGRALS. All Fermi and Bose integrals, at T = 0 and finite T, come from
eos/general/. No model implements its own. JEL is the validated
implementation and remains a SELECTABLE option; faster alternatives may be
added alongside and are validated against it, never replacing it.

GENERAL/ IS THE SINGLE HOME for particle properties, physical constants,
lepton thermodynamics, table I/O, figure styling and observational
constraints. Declared once, imported everywhere. No second rcParams setter, no
re-declared constant or particle mass in a submodule.

FIGURES. One module, eos/general/figure_style.py, is the only place that sets
matplotlib styling, colours, or figure geometry. Its house style is the style
of the 2fam_PNS_nucleation paper figures. Observational constraints have a
one-call overlay API.

LAYOUT.
    eos/          all models, one subpackage each, plus general/
    notebooks/    Jupyter notebooks
    output/       generated tables and plots, in per-model or per-study
                  subfolders. Gitignored.
    test/         the test suite. Gitignored: kept locally, not published.
    docs/         documents
Each model carries eos/<model>/<model>.tex and <model>.md, a short paper-style description
with bibliography. It is part of the model, not optional documentation.

READABILITY. The intended reader is a physicist, not a software engineer.
Code is judged on whether the physics is visible in it. Clever, compact or
heavily abstracted code is a defect here even when correct. Prefer an explicit
loop to a dense comprehension, a named intermediate to a nested expression,
and adding a module to growing one past ~600 lines.

Then STOP again.

---

STEP 3 — PROPOSED STRUCTURE, after I have approved CLAUDE.md.

Write eos/docs/REFACTOR_PLAN.md: the target tree for both repos, one line of
justification per move, plus

  - the API gap per model: current point/table signature beside the spec one,
    and what must change;
  - the work estimate per model, in the honest units of "this is an afternoon"
    / "this is a week", especially for modes the coverage matrix marked absent;
  - a RISK LIST, ranked: what could silently change a physics number and what
    check would catch it. Call out anything touching the _ref/_fast split, the
    analytic Jacobians, or the golden values in test/dd2/.

Keep the package directory named eos/eos/. Renaming it breaks `import eos` in
nucleation and in every notebook, for no gain.

Also the structure internal of each model should be evaluated and a new version proposed. 

Change no code in this phase.
```

**Read the plan yourself before Phase 1.** This is the step not to delegate —
it is where you decide what the repo becomes.

---

## Phase 1 — Freeze a numerical baseline

```
Before any restructuring, build the regression net that proves the refactor
changes no physics.

Write eos/test/baseline/generate_baseline.py. For every model in eos/eos/,
evaluate its main solver on a small fixed grid and save to
eos/test/baseline/<model>.npz. The grid covers, per model, every mode the
audit found implemented, both leptons=True and leptons=False where
applicable, and at least one point near each phase transition or particle
threshold the model has. Include a TOV sequence — M-R plus maximum mass — for
at least dd2, vmit and the mixed-phase code.

Then eos/test/baseline/test_baseline.py: one test per model regenerating those
points and asserting agreement with the stored .npz at rtol=1e-10. The
tolerance is deliberately tight — it checks that a refactor is a no-op, not
that the physics is right.

Keep the whole suite under two minutes. Commit the .npz files.

test/ is gitignored, so these live locally only. Note that in a comment at the
top of the generator, and tell me how to regenerate from scratch if I lose them.

Run it, confirm green against current code, commit.
```

---

## Phase 2 — Repo hygiene and the approved moves

```
Apply the structural moves from eos/docs/REFACTOR_PLAN.md and clean both repos
for publication. Touch no physics in this phase — moves, deletions and imports
only.

1. eos is already published at github.com/guerrinimirco/eos.git with ~185 MB
   of observational sample data tracked in plot/data/samples/ — a 61 MB
   J0740.txt, a 44 MB GW190425 .h5, and 24/19 MB files. Do NOT rewrite git
   history without asking me. Propose options — git-lfs, a GitHub release
   asset plus a download script, or trimming the samples to the columns
   actually used — with tradeoffs, and wait for my choice. Note this
   interacts with Phase 3: the constraint overlay needs this data to be
   reachable by a fresh clone.

2. notebooks/ZLvMIT_hybrid.ipynb is 25 MB and DD2vMIT_general1oPT.ipynb 3 MB,
   from embedded output. Strip outputs from all notebooks. Add an nbstripout
   pre-commit hook, or a one-line script target if that means a new
   dependency — ask before adding one.

3. Rewrite both .gitignore files. Globs, not hand-listed paths — the current
   eos one has a notebook path and ~60 individual test/zlvmit/*.dat lines.
   Add test/.

4. Move to the target layout: eos/, notebooks/, output/, test/, docs/.
   Reorganise output/ into per-model or per-study subfolders. Delete build/,
   eos.egg-info/, all __pycache__/, .DS_Store, and generated plot and table
   directories at top level — checking first whether anything imports from
   them. plot/data/samples/ is INPUT data, not output; handle under item 1.

5. Each notebook keeps its .ipynb only. The paired .py exports (DD2_usage.py,
   ENJL_usage.py, DD2vMIT_general1oPT.py, 2fam_PNS_nucleation.py) are jupytext
   artifacts — tell me which of each pair is newer before deleting either. .py paired should be keeped but gitignored maybe.

Run the full test suite and the Phase 1 baseline. Commit.
```

---

## Phase 3 — One figure file, and usable observational constraints

```
Make eos/general/figure_style.py the only module in either repo that sets
matplotlib styling, and make the observational constraints a one-call overlay.

Right now SIX places set rcParams, colours or figure geometry:
    eos/general/figure_style.py
    eos/general/plotting_info.py
    eos/general/observational_constraints.py
    eos/sfho/compare_with_compose.py
    eos/zlvmit/plot_results.py
    nucleation/analysis/figure/  (4 files, 528 lines)

1. The house style is the style of the 2fam_PNS_nucleation paper figures.
   That code currently lives in nucleation/analysis/figure/ — DOWNSTREAM of
   eos. It must MOVE UP into eos/general/figure_style.py, and nucleation then
   imports it back down. It cannot be imported sideways: eos must never import
   nucleation, and this is precisely the violation CLAUDE.md records as having
   already happened once with the M-R constraint overlay.

2. Fold plotting_info.py into figure_style.py and delete it. Strip the
   styling out of the other four call sites and have them import. After this,
   the rcParams grep in the Acceptance criteria block returns hits in exactly
   one file (see there for why the pattern, not the bare word).

3. figure_style.py must expose, at minimum: the rcParams setter, the shared
   colour palette, figure-size presets for one-column and two-column journal
   figures, and whatever mark/annotation helpers the 2fam figures use. One
   module — if it exceeds ~600 lines, split by function (style vs marks) and
   say why.

4. Observational and experimental constraints: M-R, M-Tidal, FOPI,
   Danielewicz, chiral EFT, and whatever else the repo already has data for.
   Note these do NOT all live in the same plane — M-R and M-Lambda are stellar,
   FOPI and Danielewicz are P vs n_B/n_0, chiral EFT is a low-density band. So
   the API is one call to overlay the constraints appropriate to a given plane
   onto an axis, one call to list what is available, and a documented way to
   add a new one. You choose how the plane is identified; make adding a
   constraint a data entry, not a new code path.
   It must work from a fresh clone — coordinate with whatever data decision
   came out of Phase 2 item 1, and fail with a message telling me how to fetch
   the data if it is absent, never a bare FileNotFoundError.

5. Reproduce one existing figure from the 2fam paper and one existing eos
   figure through the new module, and show me both. Byte-identical is not
   required; visually equivalent is.

Run both test suites and the Phase 1 baseline. Commit.
```

---

## Phase 4 — Per model: write the .tex, then refactor

Run once per model, in dependency order: `general` → `dd2`, `vmit`, `sfho`, `alphabag` , 
`zl` → `mixed`, `zlvmit`, `sfhoalphabag`, `dd2vMIT` → `tov`, `gmode`. Do not
batch them.

```
Working from eos/docs/REFACTOR_PLAN.md and eos/CLAUDE.md, handle ONE model:
<MODEL>. Touch no other model this session.

STEP 1 — write eos/eos/<MODEL>/<MODEL>.tex BEFORE changing any code.
A paper-style description, roughly 3-6 pages: the Lagrangian or thermodynamic
potential, the parameter set with values and the reference it is fitted to,
the field equations, the equilibrium conditions, and how each supported mode
closes the system. Shared bibliography in eos/docs/eos.bib. If pdflatex is
available, compile it and fix the errors.

This is a comprehension check, not paperwork. If you cannot state the field
equations from the code, you do not understand the model well enough to
refactor it — stop and tell me which part is unclear rather than guessing.

STEP 2 — refactor:
- Route every Fermi/Bose integral (T=0 and finite T), lepton thermodynamic
  quantity, particle property and constant through eos/general/. Delete local
  re-implementations. If a local version disagrees NUMERICALLY with general/,
  STOP and tell me — that is a physics discrepancy, not a refactor.
- Bring the public API to the spec: eos_point() and eos_table(), spec mode
  names, spec condition names, fm-based units at the boundary.
- Implement the modes the coverage matrix marked absent, where the audit said
  they are meaningful for this model. If that is large for this model, tell me
  the size before starting rather than half-implementing it.
- Delete the dead code the plan identified.
- Every public function gets a docstring stating the physics, naming the
  equation, citing the literature. Self-contained: no references to this plan,
  to phases, or to docs/ files.
- Preserve the _ref/_fast split. _ref stays readable and stays the reference.
- Faster, more robust or easier to use is always welcome and you do not need
  to ask — on three conditions: the Phase 1 baseline still reproduces at
  rtol=1e-10, _ref stays the readable reference and is not bypassed, and you
  show me a before/after timing rather than asserting an improvement.
- Prefer deleting to adding, and a shared function in general/ to an
  inheritance hierarchy. You may introduce an abstraction if it genuinely
  earns its place — twelve models sharing one API is a case where a small base
  class might — but justify it in one line in the commit body, and remember
  the reader is a physicist: the physics must stay visible in the concrete
  model, not disappear up into a parent class.

STEP 3 — run eos/test/ in full and the Phase 1 baseline. Report verbatim. If a
baseline value moved, revert and tell me what moved and by how much. Do not
adjust the tolerance.

Commit as "refactor(<model>): ..." with the deletions listed in the body.
```

---

## Phase 5 — Public API and documentation

```
The refactor is done. Make eos usable by someone who is not me.

1. eos/eos/__init__.py so the common tasks are one import deep, and the modes
   and species flags are importable from the top level.

2. Rewrite eos/README.md: what the library computes, install, and four
   runnable examples — (a) one point from one model in BETA_NEUTRINOLESS,
   (b) a FIXED_YC table swept over n_B and T, (c) that table fed to eos/astro/tov/
   for an M-R curve and maximum mass, (d) the same model with hyperons
   enabled, showing the species flags. Then a fifth: an M-R figure with the
   observational constraints overlaid, in the house style, in under ten lines.
   Copy-paste runnable, no pseudocode.

3. eos/docs/STRUCTURE.md, aimed at a physicist who has never seen the repo:
   the module map, the mode and species tables, the charge conventions (Y_C
   non-leptonic, S = +1 per s quark), the units, the _ref/_fast contract, how
   to add a new model, and one worked end-to-end example longer than the
   README ones. Link each model to its .tex.

4. Execute every code block in both documents and paste the real output. Do
   not write an example you have not run.

5. Re-read CLAUDE.md against the repo as it now is. It was written in Phase 0
   as a target; correct anything the refactor settled differently. Show me a
   diff before removing anything.
```

---

## Phase 6 — Propagate to nucleation

```
eos has been restructured. Update nucleation to match.

- Fix every import and call site broken by the eos changes.
- nucleation depends on eos; eos must never import nucleation. Verify both
  directions still hold, including after the Phase 3 figure move.
- Apply the same treatment to nucleation's own code: the general/ rule, the
  same API conventions, the same docstring standard, dead code removed. It is
  11.6k lines across 40 files — one pass should do; tell me if you disagree.
- Move nucleation/nucleation/tests/ to a top-level nucleation/test/ and
  gitignore it, matching eos.
- Improve nucleation/README.md to the standard of the new eos README, with
  examples you have actually run.
- nucleation has no git remote. Get it publication-ready but do not create or
  push a GitHub repo — I will do that.

Run both test suites and the eos baseline. Report verbatim.
```

---

## Acceptance criteria (paste into any phase)

```
Done means all of:
- pytest eos/test/ and pytest nucleation/test/ fully green.
- The Phase 1 numerical baseline reproduces at rtol=1e-10.
- Every model has a .tex that compiles.
- Every model implements eos_point() and eos_table() with the spec signature, and its
  mode and species coverage matches what CLAUDE.md claims.
- No Fermi or Bose integral implemented outside eos/general/.
- Model parameters are arguments everywhere; no solver raises or hangs on
  non-convergence; model objects pickle. Show this with one script that
  evaluates a model at 500 random parameter sets across a multiprocessing
  Pool, counts the non-converged ones, and finishes.
- `grep -rnE --include='*.py' 'rcParams\s*(\[|\.update)' eos/ nucleation/`
  hits exactly one file, `eos/eos/general/figure_style.py`. The pattern matches
  a subscript assignment or an `.update(...)` call — the two ways rcParams is
  actually set — rather than the bare word, and `--include='*.py'` keeps prose
  out. A plain `grep -rn "rcParams"` fails on a repository that satisfies §10
  in full: it also hits comments stating that a file does *not* set rcParams
  (`eos/zlvmit/plot_results.py`, `eos/zlvmit/table_reader.py`), so it cannot
  tell the rule from a sentence about the rule.
- Every README and STRUCTURE.md example executed, real output pasted.
- No file over 5 MB newly tracked in git.
- No new third-party dependency without asking me first.
- A physicist can find the function computing a given quantity in under a
  minute from STRUCTURE.md.
```
