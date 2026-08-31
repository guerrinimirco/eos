# API — every entry point, every argument, every returned field

The reference companion to `README.md` (what the library is, five worked
examples) and `docs/STRUCTURE.md` (where a quantity is computed, and why the
layout is what it is). This document answers a narrower question: **what can I
pass, and what comes back?**

Read `README.md` first if you have never called the library. Read
`docs/STRUCTURE.md` when you want to know where in the source a number is made.
Read this when you are writing a call and want the complete list of options.

Everything below is checked against the shipped signatures. Where a model
deviates from the uniform surface, the deviation is stated with its reason —
they are physics, not drift.

---

## 0. The three contracts, before any signature

Three properties hold at every public boundary. Calls are written against them,
so they come first.

**Units are fm-based.** `n` in fm^-3, `T` and `mu` in MeV, `eps` and `P` in
MeV/fm^3, entropy density `s` and scalar density `n_s` in fm^-3, `S_per_B` in
k_B per baryon. Natural units never cross a module boundary; a variable that
carries them inside a physics module is suffixed `_nat` and is not public.

**Non-convergence is a return value, never an exception.** A solve that fails
comes back with a status you test:

```python
result = dd2.eos_point(par, "beta_eq_neutrinoless", flags, n_B=0.4, T=0.0)
if result.ok:
    ...                       # result.point is an EoSPoint
else:
    print(result.message)     # scored and moved past, not raised
```

A sampler walks into unphysical parameter space constantly; it must be able to
score a point and continue. Internal layers may raise, the public entry points
catch and report.

**A refusal IS an exception, and a different statement.** Asking a model for a
sector it does not have, a mode it does not implement, or a flag it has not
wired raises `NotImplementedError` or `ValueError` naming which. That is the
contract working, not a defect, and it must not be collapsed with
non-convergence:

```python
try:
    species = zl.SpeciesFlags(hyperons=True)
except (NotImplementedError, ValueError) as err:
    print(err)   # "hyperons has no coupling in the Zhao-Lattimer functional..."
```

Never catch broadly. A `TypeError` is your own bug — an unexpected keyword —
and a bare `except` files it under "the model does not support that", where
nobody will find it.

**Model parameters are arguments.** There are no module-level constants to
edit, no global state, no hidden defaults reached for on your behalf. `par` is
the first argument of every entry point and is never optional; `mode` is
required too, except in the one model that has exactly one mode (`abpr`).

---

## 1. Getting a model

```python
from eos import dd2, sfho, zl, did          # hadronic
from eos import vmit, alphabag, abpr, njl, ccdm   # quark
from eos import enjl                        # both sectors, one functional
from eos import mixed                       # the composite hadron-quark engine
```

Each model package exposes the same names. Three are the API of §3–§5, the rest
are what you build the call out of:

| name | what it is |
|---|---|
| `Parameters` | the parameter dataclass; `Parameters.default()`, `Parameters.named(name)` |
| `SpeciesFlags` | the degrees of freedom, one named boolean each |
| `eos_point` | quantities at one point |
| `eos_table` | a tabulated EoS over a grid |
| `eos_response` | second derivatives and response functions |
| `rows_from_result` | a `TableResult` flattened to the long format |
| `build_table` | the grid driver `eos_table` wraps |

`rows_from_result` is re-exported at package level by `dd2`, `sfho`, `zl`,
`did`, `vmit` and `alphabag`. For `njl` and `ccdm` it lives one level down, in
`eos.<model>.table`; `enjl` returns its own row shapes (`beta_row`,
`plateau_row`) and `abpr` has no table module at all.

### Parameters

```python
par = dd2.Parameters.default()          # the published set the model is named for
par = dd2.Parameters.named("DD2Y")      # another published set
par = dataclasses.replace(par, x_Delta_sigma=0.8)   # frozen dataclass: replace, not assign
```

The published sets, per model:

| model | `default()` | `named(...)` |
|---|---|---|
| `dd2` | DD2 (nucleonic) | `DD2`, `DD2Y` |
| `sfho` | SFHo_Nucleonic | `SFHo_Nucleonic`, `SFHoY_Fortin`, `SFHoY*_Fortin`, `SFHo_2fam_phi`, `SFHo_2fam` (also `sfho.PUBLISHED_SETS`) |
| `zl` | ZL | `ZL_Constantinou` |
| `did` | DID | `DID`, `DIDY` |
| `vmit` | vMIT default | `vMIT_default` |
| `alphabag` | alphaBag default | `alphabag_default` |
| `abpr` | ABPR default | `abpr_default` |
| `enjl` | first published set | `fq0.5_B0`, `fq0.5_B1`, `fq0.7_B0`, `fq0.7_B1`, `fq1.0_B0`, `fq1.0_B1` |
| `njl` | RKH | `rkh`, `kunkel`, `gluon_exchange` |
| `ccdm` | baseline | `baseline`, `novector`, `dressed`, `stiff` |

**A parameter set and a species flag are not independent.** `DD2` and `DD2Y`
are two published fits, not one set read through two flag settings: asking a
nucleonic set for hyperons raises rather than inventing couplings nobody
published. `SFHoY_Fortin` carries hyperons and refuses Deltas; `SFHo_2fam`
carries both. `did` fitted its hyperon couplings with the rest, so `DID` and
`DIDY` are the same numbers and the flag alone selects the sector.

### SpeciesFlags

Nucleons (or the model's base quark flavours) are always present. Everything
else is an explicit named boolean, **defaulting to False in every model**, so
`SpeciesFlags()` means the same thing everywhere and no call inherits a sector
it did not name.

The six names every model carries:

| flag | sector |
|---|---|
| `hyperons` | Lambda, Sigma, Xi |
| `deltas` | Delta(1232) |
| `muons` | the muon lepton family (electrons are always on) |
| `thermal_mesons` | pi, K (and optionally the vector nonet) — these carry C and S, so they enter the charge and strangeness bookkeeping, not only eps, P and s |
| `thermal_neutrinos` | neutrino flavours NOT tracked in the composition (the tau family, or all flavours in an untrapped hot gas): eps, P, s only |
| `photons` | radiation; no conserved charge, matters only at T > 0 |

Model-specific flags, for physics only that model has:

| model | extra flags |
|---|---|
| `dd2` | `neutrinos` (the matter-composition nu_e of the trapped modes — NOT `thermal_neutrinos`), `sigma_star`, `thermal_vectors` |
| `vmit`, `alphabag`, `njl`, `ccdm` | `two_flavour` |
| `alphabag`, `abpr` | `gluons` |
| `njl`, `ccdm` | `csc` (colour superconductivity) |

Two rules govern all of them. **A flag with two legal values is a default and
is False.** **A flag with only one legal value raises on the other and is a
statement about the model** — `enjl` fixes every flag and raises on any move;
`alphabag.gluons` is a default in the unpaired modes and raises in `cfl`,
because a colour-flavour-locked phase leaves one massless gauge boson and it is
the rotated photon.

A sector the model already carries a coupling for gets no flag — the coupling
is the switch, set to zero where the sector is absent. The hidden-strange
vector phi is the worked case: `dd2` and `sfho` read `x_phi` columns, `did`
derives `g_phi` structurally, and none of the three carries a `phi` boolean.

Flags are frozen dataclasses. Flip one with `dataclasses.replace(species,
hyperons=True)`, and build them per model — `dd2.SpeciesFlags` and
`sfho.SpeciesFlags` are different classes and are not interchangeable.

---

## 2. Modes, and the conditions they take

A mode fixes which variables are independent. The names are identical in every
model.

| mode | conditions | meaning |
|---|---|---|
| `beta_eq_neutrinoless` | `n_B`, `T` | beta equilibrium, free-streaming neutrinos (mu_nu = 0), charge neutral |
| `beta_eq_neutrino_trapped` | `n_B`, `Y_Le`, `[Y_Lmu]`, `T` | beta equilibrium with trapped neutrinos; the muon family is optional |
| `fixed_YC` | `n_B`, `Y_C`, `T` | fixed non-leptonic charge fraction — the simulation-table mode |
| `fixed_YC_YS` | `n_B`, `Y_C`, `Y_S`, `T` | fixed charge and strangeness; `Y_C = 0.5, Y_S = 0` is symmetric nuclear matter |
| `cfl` | `n_B`, `T`, and in `alphabag` the gap `Delta0` | colour-flavour-locked quark matter; the locking fixes Y_C = 0 and Y_S = +1 identically, so no fraction is free to name. `alphabag` takes the CFL gap `Delta0` [MeV] as the mode's own condition; `abpr` carries it in the parameters |

Condition names are exactly `n_B, T, Y_C, Y_S, Y_Le, Y_Lmu`. Wherever a
temperature is accepted, entropy per baryon `SnB=` is accepted in its place (an
outer 1-D solve for T).

**`Y_C` is the charge fraction of strongly-interacting matter only.** Leptons
are excluded from it. Total electric neutrality is a separate, additional
condition — that is what the `leptons` flag imposes.

**`leptons` is orthogonal to the mode and is a named argument, never a
condition.**

- `leptons=True` — electrons (and muons if that family is on) are added to
  enforce total electric neutrality, contributing to eps, P and s.
- `leptons=False` — strongly-interacting matter only; the result is
  electrically charged. This is what a mixed-phase construction needs for each
  pure phase before imposing GLOBAL neutrality.
- **On a beta-equilibrium mode it is not a choice.** `True` is a redundant
  truth and is accepted and ignored; `False` asks for beta equilibrium without
  the particles that define it and RAISES.

Which model has which mode:

| model | `beta_eq_neutrinoless` | `beta_eq_neutrino_trapped` | `fixed_YC` | `fixed_YC_YS` | `cfl` |
|---|---|---|---|---|---|
| `dd2`, `sfho`, `did` | yes | yes | yes | yes | — |
| `zl` | yes | yes | yes | raises: n_S = 0 identically, so `fixed_YC` at Y_C = 0.5 IS symmetric matter | — |
| `vmit`, `njl`, `ccdm`, `enjl` | yes | yes | yes | yes | — |
| `alphabag` | yes | yes | yes | yes | yes, with `Delta0=` |
| `abpr` | — | — | — | — | yes, and it is the only mode |

`cfl` is not a choice of equilibrium condition but a statement about which
phase the model describes, which is why only the models whose physics IS that
phase expose it. A mode a model cannot support raises with a message saying
which; the gap is recorded in `docs/DEFERRED.md`, never silently skipped.

---

## 3. `eos_point` — one point

```python
eos_point(par, mode, species, n_B, T=None, SnB=None, leptons=None, x0=None,
          **conditions)
```

| argument | meaning |
|---|---|
| `par` | the parameter object. Required, first, never defaulted. |
| `mode` | one of §2. Required (except `abpr`, where it defaults to `'cfl'`). |
| `species` | a `SpeciesFlags` of that model. |
| `n_B` | baryon density [fm^-3]. |
| `T` | temperature [MeV]. |
| `SnB` | entropy per baryon [k_B/baryon], in place of `T`. Pass one, not both. |
| `leptons` | see §2. Named for the fixed-fraction modes, left unsaid for beta equilibrium. |
| `x0` | a warm start — the unknown vector from a nearby solved point. `None` takes the model's cold start. |
| `**conditions` | the mode's fractions: `Y_C`, `Y_S`, `Y_Le`, `Y_Lmu`. |

Model-specific extras:

| model | extra argument |
|---|---|
| `dd2` | `analytic_jac=True` — the analytic Jacobian of `backends/`. `False` selects the finite-difference reference; the numbers are the same, the speed is not. |
| `njl`, `ccdm` | `patterns=` — the pairing patterns to try in a colour-superconducting solve. |
| `ccdm` | `branches=` — which solution branches to follow. |

### What comes back

A `PointResult` with three fields:

| field | meaning |
|---|---|
| `ok` | did it converge (test this — non-convergence is a return value) |
| `message` | why not, when it did not |
| `point` | an `EoSPoint`, when it did |

An `EoSPoint` carries:

| field | meaning |
|---|---|
| `mode`, `conditions` | what was asked for |
| `n_B`, `T` | where |
| `P`, `eps`, `s` | pressure, energy density [MeV/fm^3], entropy density [fm^-3] |
| `matter` | the strongly-interacting block: densities, `m_eff_i`, `mu_eff_i`, the mean fields, `mu_dot_n` |
| `leptons` | the lepton block, or `None` when `leptons=False` |
| `converged`, `error` | the same status, at point level |

and two methods worth knowing:

- `point.euler_residual()` — `(eps + P - T s - sum_i mu_i n_i) / eps`, the
  identity of the thermodynamic consistency checks as a number to test rather
  than an assertion to trip. It should sit near 1e-8 or below.
- `point.entropy_per_baryon`, `point.free_energy_density` — derived, named.

```python
result = sfho.eos_point(par, "fixed_YC", species, n_B=0.4, T=10.0,
                        Y_C=0.1, leptons=True)
if result.ok:
    p = result.point
    print(p.P, p.eps, p.s, p.matter.m_eff_i["n"], p.euler_residual())
```

---

## 4. `eos_table` — a grid

```python
eos_table(par, mode, species, axes, fixed=None, leptons=None,
          skip_errors=True, rows=False, progress=None, verbose=False)
```

| argument | meaning |
|---|---|
| `axes` | the grid, as a dict, and **the only place the fractions go**. `{"nB": array, "T": array}` — or `"SnB"` in place of `"T"` — plus one array per fraction the mode fixes: `{"Y_C": np.array([0.1])}`. Unlike `eos_point`, `eos_table` takes NO `**conditions`: a fraction held at one value is an axis of length one, and passing `Y_C=0.1` here is a `TypeError`. |
| `fixed` | quantities held at a value across the whole grid, where the model supports it. |
| `leptons` | as §2. |
| `skip_errors` | `True` (default) drops a non-converged point from its line and keeps going; `False` stops at it. A row count below the requested count is the table saying which points it could not solve — never a silent truncation. |
| `rows` | `True` returns the long format directly instead of a `TableResult`. Not available in `dd2`; `rows_from_result` is the route all models share. |
| `progress` | a callback, invoked once per completed line (see below). |
| `verbose` | `True` uses the built-in printer. Default is silent; deep solver code never prints. |

The density axis is **warm-started inside the library** — each solved point
seeds the next, with bisected steps through onsets — so a table is not a loop
over `eos_point` that you could have written yourself.

`enjl` deviates, and its extras are the branch structure of that functional:
`direction='up'|'down'` (which way the continuation runs), `coexistences=`,
`eta=`, `x0=`.

### The progress callback

Called once per completed line — one temperature and one combination of the
fractions the mode fixes — with the SAME dictionary in every model, so one
printer serves them all:

```python
{ "mode", "line", "n_lines", "temp_key", "temp", "fracs",
  "n_solved", "n_requested", "elapsed_s" }
```

`fracs` carries every fraction the line was solved at, swept or fixed. An
engine with more to report adds keys alongside these — the mixed builder adds
`eta` and the located `window` — it does not rename them.

```python
lines = []
table = dd2.eos_table(par, "fixed_YC", species, axes, progress=lines.append)
print(lines[-1]["n_solved"], "of", lines[-1]["n_requested"],
      "in", lines[-1]["elapsed_s"], "s")
```

### What comes back

A `TableResult` — the solver's own structure, not the shape you plot from:

| field | meaning |
|---|---|
| `nB` | the density axis as asked for |
| `temp_key` | `"T"` or `"SnB"` |
| `temp_values` | the thermal axis |
| `combos` | `(temperature, fractions)` per line, in order |
| `points` | the lines; `points[i]` pairs with `combos[i]`, each entry an `EoSPoint` |
| `spec` | the `TableSpec` the build ran from |

`rows_from_result(table)` flattens it into the long format everything
downstream reads: a plain list of dicts, one per **converged** point, with the
same column names in every model.

```python
rows = dd2.rows_from_result(table)
cold = [row for row in rows if row["T"] == 0.0]
n_B = np.array([row["n_B"] for row in cold])
P = np.array([row["P"] for row in cold])
```

The columns present in every row: `n_B`, `T` (or `SnB`), `P`, `eps`, `s`,
`S_per_B`, `mu_B`, `mu_S`, `mu_e`, `Y_C`, `Y_S`, `chi`, `phase`, plus one
`Y_<species>` per particle the flags turned on — `Y_n`, `Y_p`, `Y_e`,
`Y_Lambda`, `Y_Delta++` for a hadronic model, `Y_u`, `Y_d`, `Y_s` for a quark
one — and `mu_nue` under a trapped mode. Which `Y_i` appear is the species
selection; the rest do not move between models.

Use `rows` for anything you plot, save, or hand to a structure solver. Reach
into `table.points` only for something a row does not carry — `euler_residual`,
the mean fields, the raw `matter` block.

---

## 5. `eos_response` — second derivatives

```python
eos_response(par, mode, species, frozen='equilibrium', n_B=None, T=0.0,
             **conditions)
```

A second derivative is only defined once you say **what is held fixed**, and
that choice encodes which reactions are faster than the perturbation
timescale. So the conditioning is explicit and has three independent axes:

1. **what composition is held** — `frozen=`. **What ships today is narrower
   than the design**, and a preset a model has not wired raises with the list
   it does have rather than guessing:

   | model | implemented freezes |
   |---|---|
   | `dd2` | `'equilibrium'`, `'composition'` (needs the freeze target `Y_p=`) |
   | `sfho`, `zl`, `did`, `vmit`, `alphabag`, `abpr`, `njl`, `ccdm` | `'equilibrium'` |
   | `mixed` | `'equilibrium'`, `'chi'` |
   | `enjl` | none — `eos_response` raises; the second derivatives are not wired for this model |

   The design allows the freeze to be a SET of quantity names, so a
   combination nobody anticipated is reachable without new code; the presets
   named in `CLAUDE.md` (`'fast'` holding every `Y_i` and chi, `'slow'` holding
   `Y_C` and chi) are not wired in any model yet. Ask for one and you get a
   `NotImplementedError` listing what that model implements.
2. **what thermal variable is held** — T (isothermal) or entropy per baryon
   (adiabatic). These differ at T > 0 by C_P/C_V, so the RETURNED NAME says
   which: `cs2_isothermal` against `cs2_adiabatic`, never a bare `cs2`.
3. **whether leptons re-neutralize** against the held charge — `leptons=`.

A freeze target may appear as a named argument: `Y_p=` on a model whose
`composition` freeze holds the proton fraction. That is not a condition and
does not travel in `**conditions`.

Returned keys, a plain dict: `C_V`, `C_P`, `cs2_isothermal`,
`cs2_adiabatic` (where the model computes it), and `chi` — the susceptibility
matrix chi_ab = dn_a/dmu_b for a, b in (B, C, S). Some models also return
`converged`; test it with `out.get("converged", True)` rather than assuming it
is there. The combinations a model implements are named in its docstring; one
it does not implement raises saying so.

Per-model extras: `rel_step=` / `rel_dn=` / `dT=` (the finite-difference
stencils, `zl`, `did`, `vmit`, `alphabag`, `njl`, `ccdm`), `patterns=`,
`branches=` (`njl`, `ccdm`).

---

## 6. The nuclear-matter-parameter maps

Models with a nuclear sector expose the forward map (couplings -> NMPs) and,
where a published closure exists, the inverse. Both live in `eos.<model>.nmp`.

| model | `compute_nmp` | `invert_nmp` / `from_nmp` | sector attachment |
|---|---|---|---|
| `dd2` | yes | yes | `from_hyperon_potentials`, `from_delta_potential`, and `build_parametrization` which runs all three |
| `sfho` | yes | yes | `from_potential_depths` (hyperon depths + the three Delta ratios) |
| `zl` | yes | yes, in CLOSED FORM — no seed, no basin. Six couplings against five NMPs leaves one free choice, so the caller names it — `invert_nmp(nmp, gamma1=..., par_base=None)`, or a sixth datum `K_sym` inside `nmp` | — |
| `did` | yes | raises: the DID inversion has no published closure; the paper's NMPs are predictions | — |

`compute_nmp(par)` returns a dict: `n_sat`, `E_sat`, `m_eff_ratio`, `K_sat`,
`E_sym`, `L_sym`, and `Q_sat`, `K_sym` as **predictions** — the closure does
not impose them. (`did` returns its own key names: `n_0`, `B`, `K`, `Q`, `M`,
`S_2`, `L_2`, ... Read them against that model's document.)

`invert_nmp` imposes `{n_sat, E_sat, m_eff_ratio, K_sat, E_sym, L_sym}` and
returns `(Parameters, InversionStatus)`. Non-convergence is a return value:
`status.ok` is False and the parameters are `None`. Only a hard infeasibility
raises — a target outside the physical window, where there is nothing to find.

Calling conventions differ and are not interchangeable:

```python
# dd2: NMPs as one dict; sector keys may ride in the same dict
par, stage, message = dd2_nmp.build_parametrization(
    dict(NMP, x_Delta_omega=1.0, x_Delta_rho=1.0), flags,
    hyperon_potentials=dict(U_Lambda=-30.0, U_Sigma=30.0, U_Xi=-18.0),
    U_Delta=-50.0)
# stage: 'ok' | 'inversion_failed' (no DD-RMF realisation of those NMPs)
#              | 'sectors_failed' (there is one, but the sector inversion missed)

# sfho: NMPs expanded as keywords, then the sectors on the inverted base
base, status = sfho_nmp.invert_nmp(**NMP)
par = sfho_nmp.from_potential_depths(U_Lambda_N=-30.0, U_Sigma_N=30.0,
                                     U_Xi_N=-18.0, base=base,
                                     x_Delta_sigma=1.15, x_Delta_omega=1.0,
                                     x_Delta_rho=1.0)

# zl: the free isovector choice is named
par, inversion = zl_nmp.invert_nmp(target_without_m_eff, gamma1=2.45)
```

**Each sector is inverted ON the inverted nucleon base**, so it adapts to the
couplings the NMP inversion has just moved rather than assuming the published
ones. dd2 takes its Delta sector as a DEPTH (`U_Delta`, inverted into
x_Delta_sigma); sfho inverts no Delta depth, so all three Delta couplings are
named as ratios. `invert_nmp` on an sfho base that already carries hyperons
requires `hold_hyperons='ratios'|'depths'` — the ratios and the depths cannot
both survive an inversion, and which one does is physics.

What is legitimate to ask for is bounded by the functional. m\*/m is the one to
watch: DD2 sits at 0.5625, SFHo at 0.76, and a target far outside a model's
neighbourhood either fails to invert or inverts into a parametrisation whose
scalar mass collapses before the top of your density grid.

---

## 7. The composite engine — `eos.mixed`

`mixed` couples one hadronic and one quark phase through the phase-adapter
contract. It is not a model and its first argument is not a `Parameters`: it
is a **pair of `Phase` objects**, each closing over its own model's parameters.

```python
from eos import mixed
phases = mixed.default_pair(par, flags, vmit_params)     # DD2 + vMIT
result = mixed.eos_point(phases, "beta_eq_neutrinoless", species,
                         n_B=0.6, T=0.0, eta=0.0)
```

```python
eos_point(phases, mode, species=None, n_B=None, T=None, SnB=None, eta=0.0,
          leptons=None, x0=None, analytic_jac=False, check_consistency=True,
          **conditions)
eos_table(phases, mode, species=None, axes=None, eta=0.0, fixed=None,
          leptons=None, window_only=True, analytic_jac=False, refine='exact',
          progress=None, verbose=False)
eos_response(phases, mode, species=None, frozen='equilibrium', n_B=None,
             T=0.0, eta=0.0, leptons=None, rel_dn=0.001, **conditions)
```

| argument | meaning |
|---|---|
| `eta` | the mixed-phase surface parameter: `0.0` is Gibbs (global neutrality), large eta approaches Maxwell (local). |
| `window_only` | solve only inside the located coexistence window. |
| `refine` | how the phase boundaries are located. |
| `check_consistency` | verify the adapter blocks against each other at the solved potentials. |

**A mixed result reports more than a point EoS**: the transition observables
are part of the result, not a by-product — the phase boundaries (`n_onset`,
`n_offset` per temperature and fraction combination), the quark volume fraction
`chi`, and the per-phase decomposition of every conserved charge. A mixed table
is "rows + windows".

Shipped adapters: `dd2_phase`, `did_phase`, `alphabag_phase`, `ccdm_phase`,
`composition_phase`, `enjl_branch_pair`, and `default_pair` for DD2+vMIT. A new
pairing is a new adapter, not a new engine.

---

## 8. Stellar structure — `eos.astro.tov`

A model's side of the contract is producing an `EOSTable_for_TOV`; running a
sequence over one is astro's side.

```python
from eos.general.state import EOSTable_for_TOV
from eos.astro.tov import compute_tov_sequence, find_mmax_precise

core = EOSTable_for_TOV(P=..., epsilon=..., nB=...)   # MeV/fm^3, MeV/fm^3, fm^-3
sequence = compute_tov_sequence(core, e_c, add_crust_table="BPS",
                                n_transition=0.08, verbose=False,
                                backend="fast")
index, _, m_max = find_mmax_precise(sequence)
stable = sequence[:index + 1]
```

```python
compute_tov_sequence(eos_input, e_c_vec, add_crust_table='No',
                     add_crust_mode='attach', n_transition=None, delta_n=0.01,
                     custom_crust_path=None, crust_YL=None, crust_S=None,
                     compute_baryonic_mass=True, compute_tidal=True,
                     output_file=None, eos_columns=(0, 1, 2), skip_header=0,
                     verbose=True, backend='scipy', tov_parallel=True)
```

| argument | meaning |
|---|---|
| `eos_input` | an `EOSTable_for_TOV`, or a path to a table file (then `eos_columns` and `skip_header` apply). |
| `e_c_vec` | central energy densities [MeV/fm^3]. `generate_ec_logspace(e_min, e_max, n)` builds one. |
| `add_crust_table` | `'No'`, `'BPS'`, `'compose_sfho_nYCT'`, `'compose_sfho_nT0_beta'`, `'compose_sfho_nYLS_trap'`, `'personalized'`. |
| `add_crust_mode` | `'attach'` or `'interpolate'`; `n_transition` is where, `delta_n` the interpolation width. |
| `compute_baryonic_mass`, `compute_tidal` | which columns to fill. |
| `backend` | `'scipy'` (default) is the adaptive DOP853 REFERENCE — robust through strong phase transitions and edge cases, and what correctness is judged against. `'fast'` is the numba Dormand-Prince kernel, ~100x quicker on a sequence, for sweeps and inference. Same column layout either way. |
| `tov_parallel` | fast backend only: numba `prange` over the sequence. Set `False` when the call is already inside a parallel map. |

Returns an array of one row per star, eight columns:

```
(eps_c, n_c, P_c, R, M, M_b, k2, Lambda)
    0      1    2   3  4   5    6    7
```

so radius is column 3 and gravitational mass column 4.

- `find_mmax_precise(results, precision=1e-3)` -> `(index, e_c, M_max)`.
- `truncate_to_stable_branch(results, ...)` -> the stable branch, but re-ordered
  to six columns and without `Lambda`; slice `[:index + 1]` yourself if you
  need the tidal column.

**Two gates worth running before you believe a mass.** A table delivered to a
structure solver must have P non-decreasing in n_B and 0 <= c_s^2 <= 1 — a raw
model branch may legitimately violate both inside a first-order transition, and
the violation is resolved by a construction before the table reaches TOV. And a
maximum is only a maximum if the sequence **turned over**: when the heaviest
star is the last one computed, the table ran out before the star did and the
number is a lower bound on M_max, not M_max.

---

## 9. Shared infrastructure — `eos.general`

Declared once, imported everywhere. A model overrides a shared value only
through its own parameter object, never by re-declaring the constant.

### Tables on disk — `eos.general.table_io`

```python
save_table(rows, path, meta=None, windows=None)      # -> the written path
load_table(path)                                     # -> (columns, meta, windows)
export_csv(...)
standard_name(model, mode, conditions, axes, species, leptons=True, eta=None, ext='h5')
table_path(model, name, root='output/tables')
```

`load_table` returns `(columns, meta, windows)`: `columns` is `{name: array}`
(wrap it in a `pandas.DataFrame` if you want one), `meta` is the flat metadata
dict, `windows` is `{axis key tuple: (n_onset, n_offset)}` and is empty unless
a mixed table stored some.

**`standard_name` takes the species as a DICT, not a `SpeciesFlags`** — it asks
each flag by name, so pass `dataclasses.asdict(species)` (or the dict you built
the flags from). Everything else there takes the flags object.

`standard_name` builds the filename out of the run itself — model, mode, the
mode's fractions, both axes, the sectors that are on, `nolep` when the
neutralizing leptons are off — so two runs cannot collide silently. The full
metadata still goes inside the file through `save_table(meta=...)`.

```python
name = standard_name("dd2", "fixed_YC", {"Y_C": 0.1}, axes,
                     dataclasses.asdict(species), leptons=True)
save_table(rows, table_path("dd2", name, root=str(ROOT / "output" / "tables")),
           meta={"model": "dd2", "mode": "fixed_YC", "parameters": par})
# dd2_fixed_YC_YC0.100_T0.0-10.0x2_nB0.1-1.2x40_mu.h5
```

### Observational constraints — `eos.general.constraints`

```python
overlay(ax, plane, *, only=None, style='contours', inline_labels=False,
        show_mass_bands=True, zorder=0)
list_available()
```

One call overlays the constraints of a plane onto an axis. The planes:
`'M-R'`, `'M-Lambda'`, `'Mchirp-Lambdatilde'`, `'P-n'`, `'E-n'`, `'Esym-n'`.
`style='contours'` draws 68%/95% credible regions; the alternative is a
continuous posterior-density gradient. Adding a constraint is a data entry, not
a new code path. Missing data fails with a message saying how to fetch it.

### Figures — `eos.general.figure_style`

The ONLY module in this repository (and its downstream projects) that sets
matplotlib styling, colours or figure geometry. Do not re-declare
`STANDARD_COLORS` or write a second rcParams setter.

| name | what it is |
|---|---|
| `LABELS` | axis labels by key: `nB`, `P`, `epsilon`, `s`, `T`, `Y_C`, `Y_i`, `mu_B`, `mu_C`, `mu_S`, `sigma`, `omega`, `rho`, `phi`, ... |
| `OKAB_CAT`, `OKAB`, `COLORS_SEQ`, `STANDARD_COLORS`, `T_COLORS` | the palettes; `get_T_color(T)` for a temperature |
| `PARTICLE_STYLES`, `particle_style(name)` | colour by particle, linestyle by multiplet — nucleons solid, hyperons dashed, Deltas dash-dot, leptons dotted, so a panel reads in black and white |
| `paper_grid(shape, mode=...)` | the panel grid in the house style |
| `panel_label`, `add_panel_labels`, `log_decades` | panel furniture |
| `save_figure(fig, path)` | writes the figure in the house formats |
| `set_paper_style`, `set_global_style`, `apply_style` | the styling entry points |

How to use it is documented in the module docstring (paper vs notebook style,
the palettes, the panel-grid helpers) and in the worked figure in
`docs/STRUCTURE.md` §12. A new figure starts from those two places, not from a
copied cell. The CMU-Serif missing-minus-sign protection lives there and is
never removed.

### States — `eos.general.state`

`EoSPoint`, `PhaseThermo`, `LeptonThermo`, and `EOSTable_for_TOV(P, epsilon,
nB)` — the contract object between a model and `eos.astro`, which lives in
`general/` precisely because both layers may import it.

### Integrals and constants

All Fermi and Bose integrals, at T = 0 and finite T, come from `eos.general` —
`fermi_integrals`, `bose_integrals`. JEL is the validated implementation and is
never removed; alternatives are added alongside and validated against it. No
model implements its own. `general/` is likewise the single home for particle
properties and quantum numbers (`particles`), physical constants
(`physics_constants`), the conserved-charge basis maps (`basis`), lepton and
photon thermodynamics, and the thermal meson gas.

---

## 10. Recipes

**One point.**

```python
from eos import dd2
par = dd2.Parameters.default()
species = dd2.SpeciesFlags(muons=True)
result = dd2.eos_point(par, "beta_eq_neutrinoless", species, n_B=0.4, T=0.0)
print(result.point.P if result.ok else result.message)
```

**A table, saved.**

```python
import dataclasses
import numpy as np
from eos.general.table_io import save_table, standard_name, table_path

# The mode's fractions are AXES, not keyword arguments: eos_table takes a grid,
# and a fraction held at one value is an axis of length one.
axes = {"nB": np.geomspace(0.05, 1.2, 150), "T": np.array([0.0, 10.0, 30.0]),
        "Y_C": np.array([0.1])}
table = dd2.eos_table(par, "fixed_YC", species, axes, leptons=True,
                      verbose=True)
rows = dd2.rows_from_result(table)
save_table(rows, table_path("dd2", standard_name("dd2", "fixed_YC", {"Y_C": 0.1},
                                                 axes, dataclasses.asdict(species))),
           meta={"model": "dd2", "mode": "fixed_YC", "parameters": par})
```

**A parametrisation built from nuclear-matter parameters, then a star.**

```python
from eos.dd2 import nmp as dd2_nmp
from eos.astro.tov import compute_tov_sequence, find_mmax_precise
from eos.general.state import EOSTable_for_TOV

flags = dd2.SpeciesFlags(hyperons=True, deltas=True)
NMP = dict(n_sat=0.153, E_sat=-16.1, m_eff_ratio=0.70, K_sat=242.0,
           E_sym=31.6, L_sym=50.0)
par, stage, message = dd2_nmp.build_parametrization(
    dict(NMP, x_Delta_omega=1.0, x_Delta_rho=1.0), flags,
    hyperon_potentials=dict(U_Lambda=-30.0, U_Sigma=30.0, U_Xi=-18.0),
    U_Delta=-50.0)

rows = dd2.rows_from_result(dd2.eos_table(
    par, "beta_eq_neutrinoless", flags,
    {"nB": np.geomspace(0.05, 1.6, 150), "T": np.array([0.0])}))
core = EOSTable_for_TOV(P=np.array([r["P"] for r in rows]),
                        epsilon=np.array([r["eps"] for r in rows]),
                        nB=np.array([r["n_B"] for r in rows]))
e_c = np.geomspace(250.0, 0.95 * float(core.epsilon.max()), 25)
sequence = compute_tov_sequence(core, e_c, add_crust_table="BPS",
                                n_transition=0.08, verbose=False,
                                backend="fast")
index, _, m_max = find_mmax_precise(sequence)
```

**Response functions, said out loud.**

```python
out = dd2.eos_response(par, "beta_eq_neutrinoless", species,
                       frozen="equilibrium", n_B=0.4, T=10.0)
out["cs2_isothermal"], out["cs2_adiabatic"], out["C_V"], out["C_P"], out["chi"]
```

**Sweeping parameters** — the shape inference takes. Parameters are arguments,
there is no global state, and model objects are picklable, so this parallelises
without interference:

```python
for K_sat in (220.0, 240.0, 260.0):
    par, status = sfho_nmp.invert_nmp(**dict(NMP, K_sat=K_sat))
    if not status.ok:
        continue                     # score it and move on; nothing raised
    ...
```

---

## Where the rules live

`CLAUDE.md` states the conventions this API is built on — the layering, the
charge basis and sign conventions, the mode and flag definitions, the
reference/fast split, the testing rules. `docs/STRUCTURE.md` maps the source.
`docs/DEFERRED.md` is the tracked ledger of per-model gaps: what a model does
not implement yet, and why. When this document and the code disagree, the code
is right and this document is a bug.
