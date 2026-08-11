# CLAUDE.md — conventions for the `eos` repository

Durable project conventions, stated in the present tense: this is what the
repository looks like and how it behaves. These are not suggestions — they are
the invariants the test suite encodes. If a change appears to require breaking
one, stop and ask rather than working around it.

---

## 1. Dependency direction and layering (non-negotiable)

**`eos` never imports `nucleation`** (or any other downstream project). `eos`
is the library; `nucleation` and friends are its consumers and declare it as a
dependency. An import in that direction is a cycle: it makes `pip install eos`
alone insufficient to use `eos`. `test/test_imports.py` enforces the rule.

Inside the package the layers are strict:

    general/  →  models  →  composite engines  →  astro/

- `general/` imports nothing else in the repo.
- A **model** (`dd2`, `sfho`, `zl`, `vmit`, `alphabag`, `abpr`, `enjl`)
  imports only `general/`. **No model imports another model.**
- A **composite engine** (`mixed/`) couples one hadronic and one quark phase
  through the *phase-adapter contract* (§5); it imports the models it couples
  only through that surface.
- `astro/` (`tov` — including the rotating/RNS backend — and `gmode`)
  consumes tables and arrays produced by models and engines; it never imports
  model internals, and no model imports `astro/`.
- `eos/zlvmit/` is first-generation legacy code kept for its published
  results. It follows these conventions where cheap and is exempt from the
  uniform API; new hybrid work goes through `eos/mixed`.

Corollary: shared figure code belongs in `eos/general/figure_style.py`, the
one home for publication styling. Do not re-declare `STANDARD_COLORS` or write
a second rcParams setter in a submodule — import them.

## 2. Conserved charges, bases and sign conventions (non-negotiable)

The general basis is **B, C, S, e, mu, nu_e, nu_mu**:

- **B** — baryon number.
- **C** — electric charge of strongly-interacting matter ONLY: baryons, quarks
  and charged mesons. **Leptons are excluded from C.**
- **S** — strangeness, **S = +1 per s quark**: the s quark has S = +1, Lambda
  has S = +1, Xi has S = +2. This is the OPPOSITE of the PDG convention and is
  used consistently throughout. Never silently flip it.
- **e, mu, nu_e, nu_mu** — individual lepton species.

A reduced basis **B, C, S, L_e, L_mu** may be used instead, where L_e and
L_mu are the lepton family numbers. This is legitimate because total electric
neutrality ties the lepton content to C.

**Fractions.** Y_X = n_X / n_B for every charge: Y_C, Y_S, Y_Le, Y_Lmu.
Always relative to n_B, never to total particle number.

**Y_C is NON-leptonic.** Total electric neutrality (n_C = n_e + n_mu) is a
separate, additional condition that a mode may or may not impose. Conflating
the two is the most common error in this domain and the API makes it
impossible to do by accident.

**Thermal mesons carry C and S.** A thermal pi/K gas contributes its net
charge and strangeness to n_C and n_S (and hence to neutrality, fixed-Y_C and
fixed-Y_S conditions), not only to eps, P and s — the treatment of
arXiv:1210.0400. Y_C and Y_S are the TOTAL non-leptonic fractions including
the meson gas, never baryons-only.

**Chemical potentials** follow the conserved-charge decomposition

    mu_i = B_i mu_B + C_i mu_C + S_i mu_S + (lepton terms)

Species potentials are *derived*, never independent unknowns. The sign of
mu_C is fixed by **mu_C = mu_p − mu_n**, so beta equilibrium reads
**mu_C + mu_e = 0**.

**Basis changes are declared once.** The maps between the conserved-charge
basis and the species bases — (mu_B, mu_C, mu_S) ↔ (mu_u, mu_d, mu_s),
(mu_B, mu_C, mu_S) → mu_p, mu_n, mu_Lambda, ..., and the density sums
(n_B, n_C, n_S) from species densities — live in `general/` and are imported
by every model. No model carries its own copy of these algebraic maps.

**Naming.** The effective (kinetic) chemical potential is `mu_eff_i`
(mu_eff_i = mu_i − Sigma0_i, i.e. mu minus the vector/rearrangement
self-energies); the effective (Dirac) mass is `m_eff_i`. The compact symbols
nu_i, m*_i belong in docstring and .tex mathematics, defined where used.
Solver unknown vectors use the *effective* potentials rather than mu_B —
mathematically equivalent (public results always report the full mu_i), and
the recommended default because the rearrangement term and the large vector
shift cancel out of the iteration and the effective potentials vary smoothly
along a density sweep, which is what makes warm starts work.

## 3. Modes

Every model exposes the same modes; a mode fixes the independent variables:

| mode                        | independent variables      | meaning |
|-----------------------------|----------------------------|---------|
| `beta_eq_neutrinoless`      | (n_B, T)                   | beta equilibrium, free-streaming neutrinos (mu_nu = 0), charge neutral |
| `beta_eq_neutrino_trapped`  | (n_B, Y_Le, [Y_Lmu], T)    | beta equilibrium with trapped neutrinos; muon family optional — without it the mode takes (n_B, Y_Le, T) |
| `fixed_YC`                  | (n_B, Y_C, T)              | fixed non-leptonic charge fraction — the simulation-table mode |
| `fixed_YC_YS`               | (n_B, Y_C, Y_S, T)         | fixed charge and strangeness; Y_C = 0.5, Y_S = 0 is symmetric nuclear matter, for heavy-ion comparisons |

One orthogonal flag applies to `fixed_YC` and `fixed_YC_YS`:

- `leptons=True` — electrons (and muons if enabled) are added to enforce
  total electric neutrality, contributing to eps, P and s.
- `leptons=False` — strongly-interacting matter only; the result is
  electrically charged. This is what a mixed-phase construction needs for
  each pure phase before imposing GLOBAL neutrality.

Wherever a temperature axis is accepted, entropy per baryon `SnB` is accepted
in its place (an outer 1-D solve for T).

A mode a model cannot support — physically meaningless (fixed_YC_YS for
nucleonic ZL) or not yet implemented (finite T where a model is T=0) —
**raises** with a message saying which; the gap is recorded in
`docs/DEFERRED.md`. Nothing is ever silently skipped.

## 4. Species flags

Nucleons (n, p) are always present. Every other degree of freedom is an
explicit named boolean, with identical names across all models:

- `hyperons`          — Lambda, Sigma, Xi
- `deltas`            — Delta(1232)
- `muons`             — the muon lepton family (selectable everywhere; models
                        that have not wired it yet raise, they do not ignore it)
- `thermal_mesons`    — pi, K (and optionally the vector nonet). These carry
                        C and S and therefore enter the charge and
                        strangeness bookkeeping, not only eps, P and s.
- `thermal_neutrinos` — neutrino flavors NOT tracked in the matter
                        composition (e.g. the tau family, or all flavors in an
                        untrapped hot gas), included as thermal mu = 0 gases:
                        they contribute to eps, P and s only.
- `photons`           — contribute to eps, P and s only; carry no conserved
                        charge.

No sector is enabled or disabled implicitly because "its coupling happens to
be zero" — if a sector is off, its flag is False. Setting a flag a model does
not implement RAISES; a NotImplementedError is never turned into a silent
no-op.

## 5. Uniform model API

Every model exposes the same entry points with the same signatures:

    eos_point(mode, species, **conditions)     -> quantities at one point
    eos_table(mode, species, grid)             -> tabulated EoS over a grid
    eos_response(mode, species, frozen=...,
                 **conditions)                 -> second derivatives and
                                                  response functions

`conditions` are the independent variables of the mode, named exactly
**n_B, T, Y_C, Y_S, Y_Le, Y_Lmu**. Every public boundary is fm-based:
n in fm^-3, T and mu in MeV, eps and P in MeV/fm^3. Natural units stay inside
the physics modules and never leak across a module boundary.

What `eos_table()` returns is directly consumable by `eos/astro/tov` and by
the plotting code with no per-model adapter. Along the stiff axis (density)
tables are warm-started: each solved point seeds the next, with the
continuation tactics (bisected steps through onsets) documented where used.

**Progress reporting.** Every table builder accepts an optional `progress`
callback, with the same shape across models: invoked once per completed line
(or axis combination) with a small dict — the axis values, points
solved/skipped, elapsed time. Default is silent; passing a callback (or
`verbose=True` for the built-in printer) turns it on. Deep solver code never
prints.

**Response functions.** `eos_response` computes the second-derivative
quantities of the CompOSE manual (Typel et al.): heat capacities C_V and C_P,
equilibrium and frozen sound speeds, adiabatic and thermal indices, and the
susceptibility matrix chi_ab = dn_a/dmu_b for a,b in (B, C, S). Because a
second derivative is only defined once one says WHAT IS HELD FIXED — and that
choice encodes which reactions are faster than the perturbation timescale —
the freeze is an explicit argument, selectable case by case: full equilibrium
(everything re-equilibrates), frozen per-species composition (all Y_i fixed),
frozen conserved fractions (Y_C, Y_S fixed, species re-equilibrate within
them), and, in a mixed phase, frozen quark volume fraction chi; each with
leptonic re-neutralization on or off. The choices a function implements are
named in its docstring, never implied.

**Composite engines return more than a point EoS.** The mixed-phase API also
reports the transition observables: the phase boundaries (n_onset, n_offset
per temperature and fraction combination), the quark volume fraction chi, and
the per-phase decomposition of every conserved charge. A mixed table is
"rows + windows", and the windows are part of the result, not a by-product.

**Phase-adapter contract.** `eos/mixed` couples phases only through this
surface: an adapter maps (baryon potential, mu_C, mu_S, T) to a `PhaseThermo`
block — densities, n_B/n_C/n_S, P, eps, s, and the conserved-charge
potentials — solving the phase's own internal self-consistency (fields,
densities) at those fixed potentials, with an optional opaque state for warm
starts. DD2 (hadronic) and vMIT (quark) provide the shipped adapters; a new
pairing is a new adapter, not a new engine.

**One internal shape.** Every model is laid out the same way, so a physicist
who has read one can navigate all of them, and a new model is added by
supplying its equations in the same shape. **The names are mandatory; the
existence is conditional.** A model does not carry an empty module to satisfy
the template — a single-file model is fine — but where it has one of these
parts, that part has this name:

    parameters.py       the parameter dataclass + named published sets
    species.py          the SpeciesFlags: which degrees of freedom are active
    thermodynamics.py   the model's kinetic/mean-field kernels
    solver.py           the equilibrium solves, one per mode family
    table.py            the grid driver: warm-started sweep + progress callback
    api.py              eos_point / eos_table / eos_response (§5)
    verify/             the model's physics-invariant checks
    <model>.tex, .md    the paper-style description (§11)

and, only where the physics has that part:

    nmp.py,             forward and inverse nuclear-matter-parameter maps
    nmp_inverter.py
    responses.py        second-derivative quantities, when they outgrow api.py
    couplings.py        a non-trivial coupling functional
    physics/            when the kernels genuinely split — residual, analytic
                        Jacobian, jitted backend (§9)

Not `eos.py`: since `api.py` holds `eos_point` and `eos_table`, a module named
`eos.py` that is not the eos API misleads, and `eos.<model>.eos` was never a
good import line. Not `thermodynamics_<sector>.py` either, unless a model
genuinely carries two sectors — the package name already says which it is.

A composite engine (§5) is not a model and does not take this list. It carries
`adapters.py`, `api.py`, `verify/` and its own `<name>.tex`, plus whatever
subpackages its solve needs.

Differences between models come from the physics (an RMF has field
equations, a bag model does not), never from style drift.

**Nuclear-matter parameters.** Models with a nuclear sector (`dd2`, `sfho`)
expose the forward map (couplings → NMPs, `compute_nmp`) and the inverse
(NMPs → couplings, `invert_nmp` / `from_nmp`). The inversion imposes
{n_sat, E_sat, m*/m, K_sat, E_sym, L_sym}: E_sym and L_sym close the
isovector sector; the isoscalar sector is closed by the model's structural
conditions (for DD2, the cross-constraint f''_sigma(1) = f''_omega(1) plus
one shape coefficient pinned at its published value). Higher derivatives not
imposed (Q_sat, K_sym) are *reported as predictions*, with imposing Q_sat
available as an option.

## 6. What the library is for

The API is shaped by its downstream uses, and a design choice that makes any
of them awkward is the wrong choice:

1. Generating tables for astrophysical simulations.
2. Figures of thermodynamic quantities, and TOV / stellar structure.
3. Bayesian inference over model parameters — future, but designed for now.
4. Machine-learning surrogates and emulators — future.
5. Downstream physics packages; `nucleation` is the first.

Four requirements follow from 3 and 4, and they are not optional:

- **MODEL PARAMETERS ARE ARGUMENTS**, never module-level constants or
  globals. Inference varies couplings, nuclear-matter parameters and B across
  millions of calls; a parameter that can only be changed by editing a source
  file makes inference impossible. Published parameter sets are named
  defaults, not hardcoded values.
- **NON-CONVERGENCE IS A RETURN VALUE** at every public boundary, not an
  exception and never a hang. A sampler walks into unphysical parameter space
  constantly and must be able to score that point and move on. Every result
  carries a convergence status the caller can test, judged on a residual
  norm, and every solver has a bounded iteration count. (Internal layers may
  raise; the public entry points catch and report.)
- **NO GLOBAL MUTABLE STATE.** Same inputs, same outputs. Model objects are
  picklable so multiprocessing and MPI work. Two models with different
  parameters coexist in one process without interfering. Read-only caches
  keyed by immutable parameters are allowed.
- **ARRAY IN, ARRAY OUT** wherever the physics allows. `eos_table()` takes
  grids and returns arrays. A table must not be a bare Python loop over
  `eos_point()` that the caller could have written — unless the solver
  genuinely needs the previous point as a warm start, in which case the
  docstring says so.

**On derivatives and autodiff.** The derivatives the solvers need are
hand-derived analytic Jacobians, written and tested — that is the shipped
design, chosen after an automatic-differentiation (JAX) port was tried and
rejected (the integral cores and T = 0 thresholds do not trace well). The
code is therefore not required to be auto-differentiable; but no design
choice may make a future autodiff or surrogate path structurally impossible.

Performance work (faster kernels, dedicated fast variants for Bayesian use)
is welcome but profile-driven and comes after correctness; the reference
implementations are never sacrificed to it.

## 7. Integrals and the single home rule

All Fermi and Bose integrals, at T = 0 and finite T, come from
`eos/general/`. No model implements its own. **JEL is the validated
implementation and is never removed**; the integral implementations may be
improved (accuracy, speed) and alternatives added alongside, each validated
against JEL — supplemented, never replaced. (Analytic expressions that are
genuinely different physics — perturbative-QCD-corrected gases,
cutoff-regularized NJL integrals — are model physics, not integral
re-implementations, and live with their model.)

`general/` is likewise the single home for: particle properties and quantum
numbers, physical constants, the conserved-charge basis maps (§2), lepton and
photon thermodynamics, the thermal meson gas machinery (models supply their
couplings/effective potentials), table I/O, figure styling, and observational
constraints. Declared once, imported everywhere. A model overrides a shared
value only through its own parameter object (e.g. the masses in its params
dataclass), never by re-declaring the shared constant.

## 8. Thermodynamic consistency — the invariants that must hold

Checked by the `verify/` suites; any new physics satisfies them. They are the
fastest way to catch a wrong implementation:

- Euler relation, per phase: eps + P = T·s + sum_i mu_i n_i  (to ~1e-8 rel.)
- Free energy: f = eps − T·s, and f = −P + sum_i mu_i n_i
- Rearrangement: Sigma^R enters mu and P, never eps.
- Any EoS table DELIVERED to a structure solver has P non-decreasing in n_B
  and 0 <= c_s^2 <= 1. A raw model branch MAY violate this inside a
  first-order transition region (mechanical instability is real physics and
  branch mapping must be able to represent it); the violation is resolved by
  a construction — Maxwell, Gibbs, or the eta-mixed phase — before the table
  reaches TOV, and the check runs before integration, returning a status
  rather than a meaningless mass.

## 9. The reference/fast split

Where a solver exists in two flavors, the pattern is preserved:

- the **reference** flavor — plain NumPy/SciPy, readable, straightforwardly
  correct. This is what correctness is judged against and it is never
  bypassed or removed.
- the **fast** flavor — Numba-jitted and/or analytic-Jacobian accelerated,
  selected by a backend argument, validated against the reference by
  backend-parity checks in `verify/`.

## 10. Figures and observational constraints

`eos/general/figure_style.py` is the ONLY module in this repo (and its
downstream projects) that sets matplotlib styling, colours or figure
geometry. Its house style is the 2fam_PNS_nucleation paper style. **How to
use it is documented in the module docstring (paper vs. notebook style, the
palettes, the panel-grid helpers) and in a worked figure example in
`docs/STRUCTURE.md`** — a new figure starts from those two places, not from a
copied cell. It handles the CMU-Serif missing-minus-sign glyph (ASCII minus +
mathtext fallback); that protection is never removed.

Observational and experimental constraints live in ONE module family
(`eos/general/constraints/`) with ONE data folder: M-R, M-Lambda, P vs n_B
(FOPI, Danielewicz), eps vs n_B / chiral-EFT bands, and whatever is added
later. These do not all live in the same plane, so the API is: one call to
overlay the constraints of a given plane onto an axis, one call to list what
is available, and adding a constraint is a data entry, not a new code path.
Contours render either as 68%/95% credible regions or as a continuous
posterior-density gradient. The code that PRODUCES contours from raw
posterior samples lives in the same module family, so new data becomes an
overlay without hunting for a script. It works from a fresh clone and fails
with a message saying how to fetch missing data — never a bare
FileNotFoundError.

## 11. Layout

    eos/            the package (directory name `eos/eos/` — never renamed)
      general/      shared infrastructure (§7, §10)
      dd2/ sfho/ zl/ vmit/ alphabag/ abpr/ enjl/     one subpackage per model
      mixed/        the composite hadron-quark engine (phase adapters, §5)
      zlvmit/       legacy first-generation hybrid (kept, exempt from §5)
      astro/
        tov/        stellar structure: TOV, tidal, crust, rotating (RNS)
        gmode/      composition g-modes
    notebooks/      one usage notebook per model: .ipynb paired to .py via
                    jupytext; notebooks call library functions and contain
                    their own plotting code — there are no *notebook_api*
                    modules
    output/         generated tables and plots, per-model/per-study
                    subfolders. Gitignored, EXCEPT `output/public/`, the
                    curated tracked subfolder for tables meant to be shared
                    on GitHub.
    test/           the test suite; kept locally, gitignored, not published
    docs/           documents, incl. STRUCTURE.md and DEFERRED.md (the
                    tracked ledger of per-model gaps), shared eos.bib

Each model carries `eos/<model>/<model>.tex` (and `.md`): a short
paper-style description with bibliography — Lagrangian or thermodynamic
potential, parameters and the reference they fit, field equations,
equilibrium conditions, how each mode closes the system. It is part of the
model, not optional documentation.

## 12. Testing

- Tests live in `test/<model>/`, named after the physics they check
  (`test_fixed_yc.py`).
- New physics gets a test in the same style AND an entry in the model's
  `verify/` suite where it is a physics invariant rather than a unit
  behaviour.
- The full suite must pass before any commit that touches solver internals.
- Do not loosen a numerical tolerance to make a test pass. If a tolerance
  genuinely needs to change, say why in the test.
- **Golden reference values are ground truth**: a new implementation that
  disagrees with them is wrong until proven otherwise. They are: the DD2
  golden SNM point at n_B = 0.16 fm^-3 and the DD2 published NMP/TOV values
  (pinned in `dd2/verify` and `test/dd2/`), the CompOSE HS(DD2) comparison
  slices, the ENJL author tables in `test/enjl/reference/`, and the
  per-model regression baselines in `test/baseline/` (every model, frozen at
  rtol = 1e-10 before the refactor began).

## 13. Readability

The intended reader is a physicist, not a software engineer. Code is judged
on whether the physics is visible in it. Clever, compact or heavily
abstracted code is a defect here even when correct. Prefer an explicit loop
to a dense comprehension and a named intermediate to a nested expression.

Split a file when it holds two separable pieces of physics, never because it
passed a line count: the layout of §5 is the target, and a model earns a
module beyond it only where the physics genuinely has a separate part — an
analytic Jacobian, a jitted backend, an inverse map. Two models that do the
same job have the same files. A line count is a property of the text and says
nothing about where the physics separates; splitting on one drives models
apart, which costs more than the length ever did.

**Docstrings stand on their own.** This is a public repository, so a comment
may not depend on a document that is not in it. State the physics, name the
equation, give the literature citation — never a plan, a phase, a milestone
number, or a `docs/` working note.
