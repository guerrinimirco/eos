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
- A **model** (`dd2`, `sfho`, `zl`, `did`, `vmit`, `alphabag`, `abpr`, `enjl`,
  `njl`, `ccdm`) imports only `general/`. **No model imports another model.**
  The one carve-out is a model's `verify/` suite, below.
- A **composite engine** (`mixed/`) couples one hadronic and one quark phase
  through the *phase-adapter contract* (§5); it imports the models it couples
  only through that surface.
- `astro/` (`tov` — including the rotating/RNS backend — and `gmode`)
  consumes tables and arrays produced by models and engines; it never imports
  model internals, and **no model imports `astro/`** — not in its solver, not
  in its `table.py`, and not in its `verify/` suite. A model's side of the
  contract is producing an `EOSTable_for_TOV` (which lives in `general/`, the
  layer both may import); running a sequence over one is astro's side, so a
  model's M–R check is a test, in `test/<model>/`, never a `verify/` entry.
  The **composite engine is the one exception**, and a named one: `mixed/`
  builds hybrid stars and its result columns ARE M_max and R_1.4, so
  `mixed/hybrid.py` imports `eos.astro.tov`. That is one file, not a
  subpackage-wide licence. The engine
  sits directly below `astro/` in the order above and couples to nothing else
  downstream; a model does not get the same latitude.
- **A `verify/` suite may reach sideways.** The model-to-model half of the
  rule binds importable model code, not the invariant suites: a `verify/`
  entry checks END-TO-END invariants, and some of those genuinely span two
  models — `abpr` checks itself against the CFL phase of `alphabag`, `enjl`
  checks its branch pair through `eos/mixed`. A suite is not on the path an
  inference sampler imports, which is what the layering rule protects, so the
  carve-out costs nothing it was defending. It is exactly that narrow: the
  suite may import another model or the composite engine; **nothing else in
  the package may**, and the astro half of the rule has no such carve-out.
  `test/test_imports.py` encodes the exemption rather than dropping the check.
- `eos/zlvmit/` is first-generation legacy code kept for its published
  results. It is exempt from the uniform API (§5), from §11's per-model
  document and from §12's test requirement — it is kept for results already
  published, not brought into conformance, and `test/baseline/` freezing a
  `zlvmit.npz` is the whole of what pins it. It follows these conventions
  where cheap; new hybrid work goes through `eos/mixed`.

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

One narrow exception: a model whose species list spans BOTH baryons and quarks
(ENJL is the only one) may keep a local quantum-number table, because its
species set is not the set any single shared table was written for. The table
is a transcription, never a second convention — the model's `verify/` suite
cross-checks every entry against `general/basis`, and that cross-check is what
buys the exception. A model with an ordinary species list does not get it.

**Naming.** The effective (kinetic) chemical potential is `mu_eff_i`
(mu_eff_i = mu_i − Sigma0_i, i.e. mu minus the vector/rearrangement
self-energies); the effective (Dirac) mass is `m_eff_i`. The compact symbols
nu_i, m*_i belong in docstring and .tex mathematics, defined where used.
Solver unknown vectors use the *effective* potentials rather than mu_B —
mathematically equivalent (public results always report the full mu_i), and
the recommended default because the rearrangement term and the large vector
shift cancel out of the iteration and the effective potentials vary smoothly
along a density sweep, which is what makes warm starts work.

A variable at the fm-based boundary of §5 carries **no unit suffix**; its
natural-units twin inside a physics module carries **`_nat`**. That suffix is
the one place the two unit systems are named apart, and it is why a bare
`n_B` never has to be read twice to learn which system it is in.

## 3. Modes

Every model exposes the same modes; a mode fixes the independent variables:

| mode                        | independent variables      | meaning |
|-----------------------------|----------------------------|---------|
| `beta_eq_neutrinoless`      | (n_B, T)                   | beta equilibrium, free-streaming neutrinos (mu_nu = 0), charge neutral |
| `beta_eq_neutrino_trapped`  | (n_B, Y_Le, [Y_Lmu], T)    | beta equilibrium with trapped neutrinos; muon family optional — without it the mode takes (n_B, Y_Le, T) |
| `fixed_YC`                  | (n_B, Y_C, T)              | fixed non-leptonic charge fraction — the simulation-table mode |
| `fixed_YC_YS`               | (n_B, Y_C, Y_S, T)         | fixed charge and strangeness; Y_C = 0.5, Y_S = 0 is symmetric nuclear matter, for heavy-ion comparisons |
| `cfl`                       | (n_B, T)                   | colour-flavour-locked quark matter; the locking fixes Y_C = 0 and Y_S = +1 identically, so no fraction is free to name |

`cfl` is the one mode that is not available to every model, because it is not a
choice of equilibrium condition but a statement about which phase the model
describes: a locked phase HAS no free charge or strangeness fraction. Only the
models whose physics is that phase expose it (`alphabag`, `abpr`), and for
`abpr` it is the only mode there is — which is why §5 lets that model, alone,
default its `mode` argument.

One orthogonal flag applies to `fixed_YC` and `fixed_YC_YS`:

- `leptons=True` — electrons (and muons if enabled) are added to enforce
  total electric neutrality, contributing to eps, P and s.
- `leptons=False` — strongly-interacting matter only; the result is
  electrically charged. This is what a mixed-phase construction needs for
  each pure phase before imposing GLOBAL neutrality.

**On a beta-equilibrium mode the flag is not a choice**: there the leptons are
constitutive, not optional. `leptons=True` is a true statement redundantly
made and is accepted and ignored; `leptons=False` asks for beta equilibrium
without the particles that define it and **RAISES**. §4's "a flag a model does
not implement RAISES" does not govern the `True` case, because nothing is
unimplemented — raising on it would punish exactly the caller writing one
uniform call across modes and models.

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

**`thermal_neutrinos` is meaningful alongside `beta_eq_neutrino_trapped`**, and
a model must not raise on the combination. The flag is defined by what it does
NOT cover — flavors absent from the matter composition — so under trapping,
where the e and mu families ARE tracked, it means the tau family, which is
free-streaming and carries no lepton number the mode constrains. The two are
orthogonal by construction: the mode says which families are trapped, the flag
adds the ones that are not.

No sector is enabled or disabled implicitly because "its coupling happens to
be zero" — if a sector is off, its flag is False. Setting a flag a model does
not implement RAISES; a NotImplementedError is never turned into a silent
no-op.

**Defaults, including a model's own flags.** Every flag above defaults to
False, so `SpeciesFlags()` means one thing in every model and no call inherits
a sector it did not name. A model may add flags for physics only it has
(`gluons`, `csc`, dd2's matter-composition `neutrinos`), and those follow the
same rule: **a flag with two legal values is a DEFAULT and is False; a flag
with only one legal value RAISES on the other and is a STATEMENT about the
model.** There is no third category — nothing defaults to True and quietly
accepts False, because that is the same implicit switch-on this section
forbids, wearing a model-specific name. `enjl` is the single exemption and is
the second kind throughout: it fixes every flag and raises on any move.

**A sector the model ALREADY carries a coupling for gets no flag: it is
controlled by that coupling.** Setting the coupling to zero is the same
statement, made where every other model number is made (§6 — parameters are
arguments, so a sampler can vary it continuously), and a boolean beside it
would be a second way to say one thing, reachable only by editing the call.
This is not the implicit switch-off the paragraph above forbids: there the
sector is off because a number *happens* to vanish and nothing says so; here
the coupling IS the statement, named and documented as the sector's switch.
The hidden-strange vector phi is the worked case — `dd2` and `sfho` read the
hyperon `x_phi` column, which is SU(6) times a free factor per multiplet
(`y_phi_Lambda = y_phi_Sigma = y_phi_Xi = 0` builds a set without the sector,
and is what `sfho`'s `SFHo_2fam` is), `sfho` reads `g_phi_N` besides, and
`did` derives `g_phi` from `(g~_omegaN, z)` by a map with no zero, which is how
that model states the sector is structural. None of the three carries a flag
for it.

**A flag's category is a property of the flag, judged over the modes the model
has** — which is why a mode may refuse a sector its physics does not contain
without creating that forbidden third category. `alphabag.gluons` keeps two
legal values in the unpaired modes and is a default there, and raises in `cfl`
because a colour-flavour-locked phase has no free gluon gas: locking leaves a
single unbroken U(1), so of the nine gauge bosons exactly one stays massless —
the rotated photon, which is why `photons` stays and `gluons` goes. That is
the same statement `abpr` makes by refusing the flag outright; `abpr` IS that
phase and nothing else, so for it the phase's statement and the flag's
category coincide. This is not a carve-out from the rule above but §3's
sentence about `cfl` — "not a choice of equilibrium condition but a statement
about which phase the model describes" — applied one sector at a time, and a
refusal is still a raise: dropping the sector silently is the no-op this
section already forbids.

## 5. Uniform model API

Every model exposes the same entry points with the same signatures:

    eos_point(par, mode, species, **conditions)  -> quantities at one point
    eos_table(par, mode, species, axes)          -> tabulated EoS over a grid
    eos_response(par, mode, species, frozen=...,
                 **conditions)                   -> second derivatives and
                                                    response functions

`par` comes first and is never optional: model parameters are arguments (§6),
so there is no entry point that reaches for a default set on the caller's
behalf. **`mode` is likewise required**, and for the same reason: defaulting it
picks a physics condition on the caller's behalf. The single exception is a
model that has exactly one mode — `abpr`, which is `cfl` and nothing else (§3) —
where there is no choice to make and no default to get wrong.

`conditions` are the independent variables of the mode, named exactly
**n_B, T, Y_C, Y_S, Y_Le, Y_Lmu**. A *freeze target* may also appear as a named
argument of `eos_response` — `Y_p=` on a model whose `composition` freeze holds
the proton fraction, say. That is not a condition and is not covered by the list
above: a condition names an independent variable of the mode, a freeze target
names what a second derivative holds fixed (below), and calling one by the
other's name would be the more misleading of the two. The `leptons` flag is
NEITHER, and is an explicit named argument, never smuggled through
`**conditions` — it is orthogonal to the mode (§3) and routing it through the
condition bag has only ever produced mode names §3 does not define.
Every public boundary is fm-based:
n in fm^-3, T and mu in MeV, eps and P in MeV/fm^3, and the entropy density s
and scalar density n_s in fm^-3 — the two the shorter list used to omit, which
is exactly where natural units survived longest. Natural units stay inside the
physics modules and never leak across a module boundary. A model that keeps its
own natural-units working record holds it under a **leading underscore**
(`_state`, `_point`): the underscore is what says the record is not on the
boundary, and it is the same line the baseline flattener and the units-band
check both draw.

What `eos_table()` returns is directly consumable by `eos/astro/tov` and by
the plotting code with no per-model adapter. Along the stiff axis (density)
tables are warm-started: each solved point seeds the next, with the
continuation tactics (bisected steps through onsets) documented where used.

**Progress reporting.** Every table builder accepts an optional `progress`
callback, invoked once per completed line — one temperature and one
combination of the fractions the mode fixes — with the SAME dictionary in
every model, so one printer serves them all:

    {mode, line, n_lines, temp_key, temp, fracs, n_solved, n_requested,
     elapsed_s}

`fracs` carries every fraction the line was solved at, swept or fixed. An
engine with more to report adds keys alongside these (the mixed builder adds
`eta` and the located `window`); it does not rename them. Default is silent;
passing a callback, or `verbose=True` for the built-in printer, turns it on.
Deep solver code never prints.

**Response functions.** `eos_response` computes the second-derivative
quantities of the CompOSE manual (Typel et al.): heat capacities C_V and C_P,
equilibrium and frozen sound speeds, adiabatic and thermal indices, and the
susceptibility matrix chi_ab = dn_a/dmu_b for a,b in (B, C, S). Because a
second derivative is only defined once one says WHAT IS HELD FIXED — and that
choice encodes which reactions are faster than the perturbation timescale —
the conditioning is explicit, and it has THREE independent axes:

- **what composition is held.** A *set* of quantity names, not a choice from a
  menu: any of the species fractions Y_i, the conserved fractions Y_C and Y_S,
  and in a mixed phase the quark volume fraction chi. Named freezes are
  presets that expand to a set — `equilibrium` holds nothing, `fast` holds
  every Y_i and chi, `slow` holds Y_C and chi — and a caller may always pass
  the set instead, so a combination nobody anticipated (chi free at fixed
  Y_C, say) is reachable without new code.
- **what thermal variable is held**: T (isothermal) or entropy per baryon
  (adiabatic). These differ at T > 0 by the factor C_P/C_V, so a returned name
  says which — `cs2_isothermal` against `cs2_adiabatic`, never a bare `cs2`
  whose meaning depends on the arguments.
- **whether leptons re-neutralize** against the held charge.

The combinations a function implements are named in its docstring, never
implied, and one it does not implement raises saying so.

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
starts. A pairing is two declared `Phase` objects, each closing over its own
model's parameters — for the composite engine the Phase pair IS the
parameter argument (which is how §6's "parameters are arguments" reads
there), in the first position of every public entry point; the DD2+vMIT
pairing is built by `adapters.default_pair(par, flags, vmit_params)`, a call
rather than a privileged position. Whether a phase's slot carries the kinetic
or the physical baryon potential is a declared property of the phase, never an
engine assumption. Shipped adapters: DD2, SFHo, ZL, DID (hadronic), vMIT,
alphaBag, NJL, CCDM (quark), and the ENJL branch pair (two branches of one
functional); a new pairing is a new adapter, not a new engine.

**One internal shape.** Every model is laid out the same way, so a physicist
who has read one can navigate all of them, and a new model is added by
supplying its equations in the same shape. **The names are mandatory; the
existence is conditional.** A model does not carry an empty module to satisfy
the template — a single-file model is fine — but where it has one of these
parts, that part has this name:

    parameters.py       the parameter dataclass + named published sets
    species.py          the SpeciesFlags, and the model's quantum numbers
    thermodynamics.py   quantities computed FROM the state (see below)
    solver.py           the equilibrium conditions and the solves that close
                        them, one per mode family
    table.py            the grid driver: warm-started sweep + progress callback
    api.py              eos_point / eos_table / eos_response
    verify/             the model's physics-invariant checks
    <model>.tex, .md    the paper-style description (§11)

and, only where the physics has that part:

    couplings.py        a coupling that is a FUNCTION of the state, e.g. the
                        density-dependent Gamma_i(n_B) of a DD-RMF. A model
                        whose couplings are constants has none: the numbers
                        go in parameters.py.
    nmp.py              the nuclear-matter-parameter map, forward AND inverse
    responses.py        second-derivative quantities, when they outgrow api.py
    backends/           the SAME equations written more than once — the
                        analytic Jacobian, the jitted kernel (§9)

Not `eos.py`: since `api.py` holds `eos_point` and `eos_table`, a module named
`eos.py` that is not the eos API misleads, and `eos.<model>.eos` was never a
good import line. Not `thermodynamics_<sector>.py` either **unless the model
genuinely carries two or more sectors, in which case every one of them takes
the suffix**: a package holding exactly one suffixed file is wrong, because
the suffix only restates the package name. This rule is about a MODEL package,
where the package name already says which physics it is. It does not bind
`general/`, which is nobody's model and holds the shared thermodynamics of
several sectors at once — `general/thermodynamics_leptons.py` names its sector
because `eos.general.thermodynamics` alone would say nothing.

**thermodynamics.py computes quantities from the state; solver.py finds the
state.** `thermodynamics.py` takes chemical potentials, fields, T, the
parameters and the species flags, and returns densities, P, eps, s and their
sums — including any self-consistency internal to the model (the mean fields,
a bag model's flavour densities). `solver.py` takes n_B, T and a mode's
conditions and finds the potentials and fields that satisfy them. The test is
that **thermodynamics.py never knows which mode it is in**: grep it for
`beta`, `Y_C`, `neutral` or `trapped` and find nothing. This is the same
boundary as the phase-adapter contract above, seen from inside a model — which
is why `eos/mixed` can consume the thermodynamics half of a model and nothing
else.

**backends/ is deletable.** Remove it and the model still gives the same
numbers, only slower: the reference path is `thermodynamics.py` + `solver.py`
and is complete on its own. That is §9's split stated as a directory, and it
is what keeps the readable implementation readable — a physicist checking the
equations never has to walk past a jitted kernel.

A composite engine (§5) is not a model and does not take this list. It carries
`adapters.py`, `species.py`, `api.py`, `responses.py`, `verify/` and its own
`<name>.tex`, plus whatever subpackages its solve needs. `species.py` is on
that list because §4 binds an engine exactly as it binds a model: the engine
carries the six names, delegating the per-phase ones to the two `Phase`
objects and consuming the phase-common ones (photons, thermal neutrinos) once
at the mixture level, so an adapter's hardcoded `photons=False` is correct by
construction rather than by accident.

**`general/` carries a `verify/` too.** It is not a model either, but it is the
single home of the Fermi and Bose integrals (§7), the conserved-charge basis
maps (§2) and the thermal meson gas — the pieces every model's correctness
rests on, and the ones a wrong result is hardest to trace back to. Its suite
checks those shared pieces against each other: JEL against the alternatives §7
requires be validated against it, the basis maps against the species tables,
the T = 0 limits against the finite-T forms as T -> 0.

**Layer order inside a model.** Imports run one way only:

    couplings.py → parameters.py → thermodynamics.py → solver.py → nmp.py
                                                                 → table.py → api.py

`parameters.py` is at the bottom and imports nothing above `couplings.py`;
`nmp.py` is near the top because computing nuclear-matter parameters requires
solving symmetric matter at saturation. A constructor that inverts NMPs is
therefore a free function in `nmp.py`, not a classmethod on the parameter
dataclass — putting it there forces a deferred import, which is the cycle
announcing itself.

Differences between models come from the physics (an RMF has field
equations, a bag model does not), never from style drift.

**Parameter or coupling?** A parameter takes no arguments; a coupling is a
function of the state. `Gamma_sigma(n_sat)` is a parameter — a stored number,
what an inference run varies. `Gamma_sigma(n_B)` is a coupling — evaluated at
every density, never stored. So `couplings.py` holds the functional *form*
(pure mathematics, no model numbers in it), `parameters.py` holds the numbers
that pin it down, and the evaluation lives on the parameter object. Coupling
*ratios* (`x_sigma_Lambda`, `x_omega_Delta`) are parameters even though they
multiply a density-dependent coupling. A mean field is neither: it is a
dynamical variable and belongs to `thermodynamics.py`.

**Nuclear-matter parameters.** Models with a nuclear sector (`dd2`, `sfho`,
`did`, `zl`) expose the forward map (couplings → NMPs, `compute_nmp`) and the inverse
(NMPs → couplings, `invert_nmp` / `from_nmp`) — both in `nmp.py`, since they
are two directions of one map and share the NMP list, its ordering and the
residual. The inversion imposes
{n_sat, E_sat, m*/m, K_sat, E_sym, L_sym}: E_sym and L_sym close the
isovector sector; the isoscalar sector is closed by the model's structural
conditions (for DD2, two shape coefficients — b_sigma and c_omega — pinned at
their published values, because E_sat and m*/m at fixed n_sat are blind to
the shape and only P and K_sat constrain it, so four shape coefficients
answer to two rows). Higher derivatives not imposed (Q_sat, K_sym) are
*reported as predictions*, with imposing Q_sat available as an option.

**A closure condition belongs to the parametrization that imposed it.** DD2's
inverse map used to carry the cross-constraint f''_sigma(1) = f''_omega(1);
that condition is the DD parametrization's, and DD2's own fit dropped it —
Typel, PRC 71, 064301 (2005) §IV imposes it and counts eight independent
parameters, while Typel et al., PRC 81, 015803 (2010) states only
f_i(1) = 1 and f_i''(0) = 0 and counts ten. Before a structural
condition is written into an inverse map, it is checked against the paper
that fitted THAT parameter set, not against the model family.

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

  **The gate belongs to whoever builds the table.** A unit with a `table.py`
  that produces a table a structure solver can consume owes this check in its
  own `verify/`; a model that builds no such table does not, and its absence
  there is correct rather than an omission. That is the test to apply — does
  this unit hand a table onward — not whether the check appears in some other
  suite.

## 9. The reference/fast split

Where a solver exists in two flavors, the pattern is preserved:

- the **reference** flavor — plain NumPy/SciPy, readable, straightforwardly
  correct. This is what correctness is judged against and it is never
  bypassed or removed.
- the **fast** flavor — Numba-jitted and/or analytic-Jacobian accelerated,
  selected by a backend argument, validated against the reference by
  backend-parity checks in `verify/`. It lives in the model's `backends/`,
  which §5 defines by the property that deleting it changes no number.

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
      dd2/ sfho/ zl/ did/ vmit/ alphabag/ abpr/ enjl/ njl/ ccdm/
                    one subpackage per model
      mixed/        the composite hadron-quark engine (phase adapters, §5)
      zlvmit/       legacy first-generation hybrid (kept, exempt from §5)
      astro/
        tov/        stellar structure: TOV, tidal, crust, rotating (RNS)
        gmode/      composition g-modes
    notebooks/      usage notebooks, GROUPED by physics rather than one per
                    model: .ipynb paired to .py via jupytext; notebooks call
                    library functions and contain their own plotting code —
                    there are no *notebook_api* modules. Grouped because the
                    figures that matter overlay several models on one axis,
                    and per-model notebooks cannot do that without importing
                    each other or sharing a helper module this section forbids.
    output/         generated tables and plots, per-model/per-study
                    subfolders. Gitignored, EXCEPT `output/public/`, the
                    curated tracked subfolder for tables meant to be shared
                    on GitHub.
    test/           the test suite; kept locally, gitignored, not published —
                    except `run_clean_suite.sh`, its own test and
                    `suite_certificates/`, which ARE tracked: §12's landing
                    measurement is only a gate if its mechanism and its
                    evidence can be cited from a commit
    docs/           documents, incl. STRUCTURE.md and DEFERRED.md (the
                    tracked ledger of per-model gaps), shared eos.bib

Each model carries `eos/<model>/<model>.tex` (and `.md`): a short
paper-style description with bibliography — Lagrangian or thermodynamic
potential, parameters and the reference they fit, field equations,
equilibrium conditions, how each mode closes the system. It is part of the
model, not optional documentation.

**Every equation the code solves, and every quantity it returns, is written
out.** The test is that a physicist can reproduce the model from the document
without opening the source. Three things this rules out, all of which the
first drafts did:

- naming a term instead of defining it. `eps = sum_i eps_kin_i + ...` is not
  a statement of the energy density until `eps_kin_i` is given in closed form.
- leaving the ideal-gas integrals to a citation. The Fermi and Bose integrals
  are shared code (§7) but each document states them anyway: a paper-style
  description is self-contained, and a reader of one model's `.tex` must not
  have to open another's. Duplication in prose is not the duplication §7
  forbids in code.
- omitting a quantity because nothing derives from it. `s` and `n_s` are
  returned by every model and must appear, including the identities they are
  computed through rather than integrated — `n_s = (eps - 3P)/m*` from the
  trace of the energy-momentum tensor, `s = (eps + P - sum_i mu_i n_i)/T`.

So the document states, explicitly: the residual — every row, in the order
the solver assembles them, with the unknown vector; the field or gap
equations; the single-species thermodynamics at T = 0 and T > 0; every
model-specific contribution to P, eps and s, with the terms that differ
between P and eps called out; and the assembly of the totals. Where a mode
changes the rows, the table of modes says which rows.

## 12. Testing

- Tests live in `test/<model>/`, named after the physics they check
  (`test_fixed_yc.py`).
- New physics gets a test in the same style AND an entry in the model's
  `verify/` suite where it is a physics invariant rather than a unit
  behaviour.
- **Before a commit, every test the change can reach passes** — the model's own
  `test/<model>/`, `test/baseline/`, and the suites that import the names the
  change touches. A change to solver internals reaches past its own model and
  the set widens to match. Naming a set is a claim about blast radius, so what
  ran and why those are the reachable suites is stated with the result.
- **The full suite is a LANDING MEASUREMENT, not a commit gate.** It runs for
  ~20 minutes and this checkout is shared, so what it measures is a TREE and
  not a change: a run spanning any `eos/*.py` edit is not a measurement even
  when the edit is whitespace, and the invalidation is discovered twenty
  minutes after the number is believed. Measured, on 2026-08-29: a window that
  opened on "no pytest running and no `eos/*.py` written for two minutes" was
  shut again twenty-nine seconds later. A timer measures a PAUSE, not a quiet
  tree, so the full suite is taken at a LANDING POINT — no uncommitted
  `eos/*.py`, a state someone CHOOSES to enter and which therefore cannot
  flicker — and it is a property of a SHA rather than of anyone's commit.
  It goes through `test/run_clean_suite.sh`, which refuses to start against a
  live tree, fingerprints every `eos/*.py` either side of the run, and writes
  `test/suite_certificates/<timestamp>.txt` carrying the count, the verdict
  CLEAN or DISCARD, the HEAD SHA and the interpreter with its numpy and scipy
  versions. **A count naming no interpreter names nothing**: this machine has
  two Python stacks that disagree, and the baselines below are frozen against
  one of them.
- **A DISCARD is cheap, and stays cheap.** Twenty minutes of CPU and nothing
  else, so nobody holds `eos/*.py` still to protect someone else's run: the
  certificate exists precisely so an invalidated run is cheap to DETECT rather
  than expensive to BELIEVE, and a session sitting on its hands has paid for
  the mechanism twice. There is no cap on re-running and exactly one
  obligation instead — **a result claiming a full-suite number cites its
  certificate path, and cites every other certificate the same work produced,
  DISCARDs included.** Re-running until one comes back CLEAN is the old sin in
  the mechanism's own clothes, and harder to see because each certificate is
  individually honest; a later CLEAN does not erase an earlier DISCARD that
  went unmentioned.
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

**Names.** Three rules, so that a physicist who has read one model can read
the next without a translation table:

1. **A name never repeats its package.** `eos.zl.compute_zl_thermo_from_mu`
   says "zl" twice; it is `eos.zl.thermo_from_mu`. Same disease as a
   `thermodynamics_quarks.py` inside a quark model.
2. **The same job carries the same name in every model.** The list below is
   the vocabulary; a model uses these names or explains in its docstring why
   its physics is a different job.
3. **A name says what the function takes and returns, not that it computes.**
   In a file called `thermodynamics.py` everything computes; `thermo_from_mu`
   against `thermo_from_n` tells the reader which variables they are handing
   over. Drop `compute_` where it carries nothing.

    Parameters                the parameter dataclass, this name in EVERY
                              model — `VMITParams` says "vmit" twice in
                              `eos.vmit.VMITParams`, and two models with one
                              job carry one name
    Parameters.default()      the published parameter set
    Parameters.named(name)    another published set
    kinetic_thermo(...)       one species as an ideal gas
    mean_fields(...)          the model's mean fields at the current state
    thermo_from_mu(...)       the block at given chemical potentials
    thermo_from_n(...)        the block at given densities
    assemble(...)             the sums: n_B, n_C, n_S, P, eps, s, sum mu_i n_i
    residual(x, ...)          the equations that must vanish
    default_guess(mode, ...)  the cold start
    warm_start(point)         the seed taken from the previous point
    solve_<mode>(...)         one equilibrium solve, with <mode> the §3 mode
                              name lowercased: solve_beta_eq_neutrinoless,
                              solve_fixed_yc, ... — so there is no second list
                              of names to drift out of step with §3
    build_table(spec, ...)    the warm-started grid
    compute_nmp / invert_nmp  the nuclear-matter-parameter map, both directions
    eos_point / eos_table / eos_response     the uniform API (§5)
    verify/run_full_check.py  the model's invariant suite, one entry point

**Order functions by the physics, not alphabetically and not by call depth.**
`thermodynamics.py` reads single species → mean fields → the per-species loop
→ the sums; `solver.py` reads guesses → residual → the solve → the modes →
the sweep. The same reading order in every model is most of what makes the
second model quick to read.

**Docstrings stand on their own.** This is a public repository, so a comment
may not depend on a document that is not in it. State the physics, name the
equation, give the literature citation — never a plan, a phase, a milestone
number, or a `docs/` working note.
