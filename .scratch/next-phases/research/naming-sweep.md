# Naming, order and docstring sweep — every model package

Ticket: [07-naming-sweep](../issues/07-naming-sweep.md) · Parent: [map](../map.md)
Standard: `CLAUDE.md` §5 (mandatory internal shape) and §13 (readability, names, order).
Scope: `eos/{zl,sfho,dd2,did,vmit,alphabag,njl,ccdm,abpr,enjl,mixed}`.
**Read-only. Nothing was renamed, reordered or edited.**

## Method

Every public and private top-level definition in the eleven packages was
extracted by `ast` (name, line, signature, docstring) — 1066 definitions across
92 modules — and judged against §5's mandated file list, §13's three name rules
and the §13 vocabulary. Order was read off the file positions. The
self-contained-docstring sweep and the cleverness sweep are greps over the same
tree. `docs/DEFERRED.md` was read afterwards, and where it already records a
finding that is noted — in two places it records a state the code does not have.

## Verdict in one line

Nine models are close to the standard and two are not. **`eos/vmit` has not been
converted at all** below `Parameters` — every §13 vocabulary name in it is
wrong, and `docs/DEFERRED.md:320` calls it "DONE". **`eos/dd2` carries the
second-largest gap**: `Parametrization` instead of `Parameters`, six solver
entry points named after `octet` rather than after the §3 modes, and the worst
docstrings in `thermodynamics.py` in the repository — including the two the
ticket names as its failing example, verbatim.

`eos/zl/thermodynamics.py`, `eos/abpr/thermodynamics.py` and
`eos/enjl/thermodynamics.py` pass every docstring test and are the reference the
rest should be written against.

## Counts

| category | findings |
|---|---|
| 1. Docstrings in `thermodynamics.py` failing the formula test | **56** functions (+ 6 dataclasses whose docstring restates their name) |
| 2. Name deviations from §13 | **74** (of which **58 PUBLIC**), plus **4** file-name deviations from §5 |
| 3. Files whose reading order is wrong | **6** (2 serious: `dd2/solver.py`, `sfho/thermodynamics.py`) |
| 4a. Docstrings referencing a plan/phase/milestone/working note | **8** |
| 4b. Dense comprehensions / nested expressions hiding an equation | **9** |

---

# 1. DOCSTRINGS — `thermodynamics.py`

The test: the docstring states the explicit quantity returned, as a formula, in
the notation of the model's document, with units. A restatement of the function
name fails. Deferring the integral to another module by naming it fails — §11
makes each model's description self-contained, and the same reasoning binds the
code (`eos/zl/thermodynamics.py:52` and `eos/enjl/thermodynamics.py:110` both
call into `eos.general.fermi_integrals` **and write the integrals out anyway**;
that is the passing shape).

## 1.1 `eos/dd2/thermodynamics.py` — 18 failures

The ticket's example is here twice, word for word.

| file:line | function | current docstring | why it fails |
|---|---|---|---|
| `eos/dd2/thermodynamics.py:61` | `kF_from_n` | `Fermi momentum [MeV] from number density n [MeV^3], degeneracy g.` | restates the signature; `n = g kF^3/(6 pi^2)` never written (cf. `eos/enjl/thermodynamics.py:67`, which does) |
| `eos/dd2/thermodynamics.py:66` | `number_density_t0` | `n [MeV^3].` | names the symbol only |
| `eos/dd2/thermodynamics.py:71` | `scalar_density_t0` | `n_s [MeV^3].` | names the symbol only |
| `eos/dd2/thermodynamics.py:79` | `eps_kin_t0` | `Kinetic energy density [MeV^4].` | **the ticket's failing example** |
| `eos/dd2/thermodynamics.py:88` | `P_kin_t0` | `Kinetic pressure [MeV^4].` | **the ticket's failing example** |
| `eos/dd2/thermodynamics.py:97` | `kinetic_thermo` | `Full kinetic thermodynamics of one fermion species. … Returns (n, P, eps, s, n_s) in natural units. T = 0 uses the exact closed forms; T > 0 comes from the Johns-Ellis-Lattimer integrals in eos.general.fermi_integrals` | names the quantities and the source module; no integral is written |
| `eos/dd2/thermodynamics.py:129` | `vector_fields` | `omega_0 and rho_0 [MeV] from the algebraic vector field equations.` | the field equations are named, not stated |
| `eos/dd2/thermodynamics.py:141` | `rearrangement` | `Rearrangement self-energy Sigma^R [MeV], identical for all baryons.` | Sigma^R is nine arguments' worth of algebra and none of it appears |
| `eos/dd2/thermodynamics.py:150` | `_field_sources` | `The meson-field sources sum_i x_i Gamma_i (n_i or n_s,i) [MeV^3].` | "(n_i or n_s,i)" leaves which source takes which; the four sums are not written separately |
| `eos/dd2/thermodynamics.py:167` | `field_eps_P` | `Meson mean-field contributions (eps_field, P_field) [MeV^4]. The scalar enters P with a minus sign, the vectors with a plus.` | states signs, not the expressions (cf. `eos/sfho/thermodynamics.py:283`, which writes both in full) |
| `eos/dd2/thermodynamics.py:189` | `lambda_omega_ratio` | `x_omega^Lambda entering the kaon omega-shift (SU(6) 2/3 by default).` | value not written |
| `eos/dd2/thermodynamics.py:215` | `meson_families` | `(mu_eff, mass, Q, S, g) per thermal meson species at DD2's potentials.` | lists the tuple's slots and nothing else |
| `eos/dd2/thermodynamics.py:224` | `thermal_meson_charges` | `(n_C, n_S) of the gas [fm^-3]. Zero unless a thermal-meson flag is on…` | the sums over the gas are not written |
| `eos/dd2/thermodynamics.py:241` | `thermal_meson_thermo` | `Full gas thermodynamics at (n_B [fm^-3], T [MeV]) on DD2's mean field. Returns the dict of eos.general.thermal_mesons.thermal_meson_thermo…` | defers wholly to another module |
| `eos/dd2/thermodynamics.py:334` | `build_matter_ctx` | `The context for matter at (n_B [fm^-3], T [MeV]) with these species.` | restates the name |
| `eos/dd2/thermodynamics.py:388` | `baryon_kinetics` | `Per-species (mu_eff, m*, n, n_s, eps, P, s) at the current fields.` | lists the returns |
| `eos/dd2/thermodynamics.py:413` | `meson_charges_nat` | `(n_C, n_S) of the thermal meson gas in NATURAL units [MeV^3].` | restates the name plus a unit |
| `eos/dd2/thermodynamics.py:426` | `assemble` | `The matter state at these fields and potentials, as a PhaseThermo.` + four paragraphs of correct prose about what is and is not included | **the sums it exists to perform — P, eps, s, n_B, n_C, n_S — are nowhere written.** This is the one function whose formula matters most |

Passing in this file: `effective_masses:353`, `effective_potentials:370`,
`meson_potentials:195`, `self_consistency_residual:507`,
`thermo_at_potentials:571`, `_sigma_ceiling:652`, `_cold_start:671`.

## 1.2 `eos/vmit/thermodynamics.py` — 6 failures

| file:line | function | current docstring |
|---|---|---|
| `eos/vmit/thermodynamics.py:88` | `compute_quark_thermo` | `Kinetic n, P, e, s of one flavour at effective potential mu_eff.` |
| `eos/vmit/thermodynamics.py:102` | `compute_quark_density` | `Number density of one flavour at effective potential mu_eff.` |
| `eos/vmit/thermodynamics.py:250` | `compute_quark_densities_for_solver` | `Effective potentials at the given mean field, and the densities they imply — the inner step of every solver in eos.vmit.solver.` (states the condition `n_*_calc == n_*`; the densities themselves are never written) |
| `eos/vmit/thermodynamics.py:289` | `compute_vmit_thermo_from_mu_n` | `Assemble a full quark-matter state from potentials and the mean field.` — the assembly `P = sum_q P_q + P_V - B`, `eps = sum_q eps_q + eps_V + B` is not written |
| `eos/vmit/thermodynamics.py:352` | `compute_quark_matter_thermo_from_n` | `Densities fix the mean field directly, so this needs no root find: invert the Fermi integrals…, add V back…, and assemble.` — procedure, not quantity |
| `eos/vmit/thermodynamics.py:401` | `compute_quark_matter_thermo_from_mu` | same shape |

The eight vector/bag/potential helpers (`:113 :124 :131 :140 :145 :153 :160 :279`)
all give the closed form and pass. The failures are exactly the functions that
assemble.

## 1.3 `eos/sfho/thermodynamics.py` — 3 functions + 3 records

| file:line | function | current docstring |
|---|---|---|
| `eos/sfho/thermodynamics.py:100` | `baryon_thermo` | `Compute thermodynamic quantities for all hadron species.` + a numbered procedure. M\* and mu\* are given in full; step 3 is `Evaluates Fermi integrals for (n, P, e, s, n_s)` and no integral appears |
| `eos/sfho/thermodynamics.py:405` | `thermal_meson_thermo` | `A thin call into eos.general.thermal_mesons, which is the ONE implementation… See that module for the physics` — explicitly sends the reader elsewhere |
| `eos/sfho/thermodynamics.py:512` | `get_residual_vector` | `Compute residual vector for self-consistent field solver. This function is designed to be used with scipy.optimize.fsolve or similar.` — the rows are not named, let alone written, though `field_residuals:221` above it writes all four |
| `eos/sfho/thermodynamics.py:40` | `HadronState` (record) | `Thermodynamic state for a single hadron species.` |
| `eos/sfho/thermodynamics.py:61` | `HadronState.__repr__` | **no docstring** |
| `eos/sfho/thermodynamics.py:66` | `HadronThermoResult` (record) | `Complete thermodynamic result for all hadrons.` |

Separately: this file is the only one in the eleven that writes its mathematics
in **Unicode Greek and subscripts** (`M*_j = m_j - g_σj × σ`, `fm⁻³`, `ℏc`,
`Σⱼ`) rather than the ASCII every other model uses. Not a §13 rule, but it is
style drift of the kind §13 exists to stop, and it defeats grepping for `sigma`.

## 1.4 `eos/did/thermodynamics.py` — 5 failures

| file:line | function | current docstring |
|---|---|---|
| `eos/did/thermodynamics.py:215` | `evaluate` | `One pass over the baryons at the given state -> Matter.` — the sums it accumulates (both rearrangement self-energies among them) are named as "paper Eqs. 10 and 11" and not written |
| `eos/did/thermodynamics.py:321` | `thermal_meson_thermo` | `The gas at DID's potentials, or an all-zero block when it is off. Returns the dictionary of eos.general.thermal_mesons…` |
| `eos/did/thermodynamics.py:358` | `thermo_from_mu` | `The matter block at given potentials and fields, as a PhaseThermo.` — correct and detailed on what is included and on the Sigma^t cancellation, but P, eps and s are not written |
| `eos/did/thermodynamics.py:443` | `_pack` | **no docstring** |
| `eos/did/thermodynamics.py:448` | `_unpack_fields` | **no docstring** |

## 1.5 `eos/ccdm/thermodynamics.py` — 1 function + 6 properties

| file:line | function | current docstring |
|---|---|---|
| `eos/ccdm/thermodynamics.py:240` | `mode_thermo` | `One colour-flavour mode's medium integrals [natural units].` — the integrals are the point of the function and are not there |
| `:329 :333 :337 :341 :345 :349` | `CCDMState.{n_B_fm,P_fm,eps_fm,s_fm,n_C_fm,n_S_fm}` | **no docstring** ×6 — each is a unit conversion of a returned quantity |

## 1.6 `eos/njl/thermodynamics.py` — 6 properties

`:306 :310 :314 :318 :322 :326` — `NJLState.{n_B_fm,P_fm,eps_fm,s_fm,n_C_fm,n_S_fm}`,
**no docstring** ×6. Everything else in this file passes; `state_at:336` and
`internal_residual:517` are among the best in the repository.

## 1.7 `eos/enjl/thermodynamics.py` — 3 properties

`:524 :528 :532` — `EoSPoint.{n_b_fm,eps_fm,P_fm}`, **no docstring** ×3.
Every function in the file passes.

## 1.8 `eos/alphabag/thermodynamics.py` — 1 function + 3 records

| file:line | name | current docstring |
|---|---|---|
| `eos/alphabag/thermodynamics.py:215` | `fermi_thermo` | `(n, P, eps, s) of an UNCORRECTED Fermi gas of mass m, degeneracy 6. From eos.general.fermi_integrals, which evaluates the integrals through the Johns-Ellis-Lattimer analytic approximation` — names the source, writes no integral. The one gap in an otherwise exemplary file |
| `:62 :72 :103` | `QuarkThermo`, `MatterThermo`, `CFLThermo` (records) | each restates its own name |

## 1.9 `eos/mixed/thermodynamics.py` — 1 failure

`eos/mixed/thermodynamics.py:71` `assemble` — `The totals of the mixture:
(P, eps, s, sum_i mu_i n_i).` then `Weighted as the module docstring states`.
The chi/eta weighting *is* the quantity this function returns and it is one
pointer away. The module docstring is in the same file, so this is the mildest
failure in the sweep.

## 1.10 Clean

`eos/zl/thermodynamics.py` (12/12), `eos/abpr/thermodynamics.py` (7/7). Both
write every quantity in closed form with units, cite the paper, and call out
where P and eps differ. `eos/enjl/thermodynamics.py` is clean on functions.
**These three are the target for the rewrites above.**

---

# 2. NAMES

§13's three rules: (1) a name never repeats its package; (2) the same job
carries the same name in every model; (3) a name says what it takes and returns,
not that it computes.

PUBLIC = reachable from the package `__init__` (`__all__`) or from `api.py`, so
renaming breaks callers. Blast radii were counted across `eos/`, `test/` and
`notebooks/`.

## 2.1 `eos/vmit` — 28 deviations, 26 PUBLIC

Every §13 vocabulary name in this package is wrong. `docs/DEFERRED.md:320-324`
records vmit as "DONE" with only `get_vmit_default()` / `get_vmit_custom()`
outstanding; **that is not the state of the code**, and the ledger needs
correcting whether or not the renames happen.

| file:line | current | proposed | rule | vis. |
|---|---|---|---|---|
| `eos/vmit/parameters.py:60` | `get_vmit_default()` | `Parameters.default()` | 1,2 | PUBLIC (55 in eos, 27 in test) |
| `eos/vmit/parameters.py:71` | `get_vmit_custom(...)` | delete — the dataclass is that constructor (as abpr and alphabag did) | 1,2 | PUBLIC (6/12/13) |
| `eos/vmit/thermodynamics.py:60` | `VMITThermo` | `MatterThermo` (zl, alphabag) | 1,2 | PUBLIC |
| `eos/vmit/thermodynamics.py:88` | `compute_quark_thermo` | `kinetic_thermo` | 2,3 | PUBLIC |
| `eos/vmit/thermodynamics.py:102` | `compute_quark_density` | `quark_density` (alphabag) | 3 | PUBLIC |
| `eos/vmit/thermodynamics.py:113` | `compute_vector_field` | `vector_field` (ccdm) | 3 | PUBLIC |
| `eos/vmit/thermodynamics.py:124` | `compute_vector_pressure` | `vector_pressure` | 3 | PUBLIC |
| `eos/vmit/thermodynamics.py:131` | `compute_vector_energy` | `vector_energy` (njl) | 3 | PUBLIC |
| `eos/vmit/thermodynamics.py:140` | `compute_bag_pressure` | `bag_pressure` (alphabag) | 3 | PUBLIC |
| `eos/vmit/thermodynamics.py:145` | `compute_bag_energy` | `bag_energy` (alphabag) | 3 | PUBLIC |
| `eos/vmit/thermodynamics.py:153` | `compute_mu_effective` | `effective_potential` | 3 | PUBLIC |
| `eos/vmit/thermodynamics.py:160` | `compute_effective_mu_quarks` | `effective_potentials` (zl, dd2, did) | 2,3 | PUBLIC |
| `eos/vmit/thermodynamics.py:250` | `compute_quark_densities_for_solver` | `effective_state` (zl's name for the same object) | 2,3 | PUBLIC |
| `eos/vmit/thermodynamics.py:279` | `compute_mu_physical` | `physical_potentials` | 3 | PUBLIC |
| `eos/vmit/thermodynamics.py:289` | `compute_vmit_thermo_from_mu_n` | `thermo_from_mu_n` (zl) | 1,2,3 | PUBLIC |
| `eos/vmit/thermodynamics.py:352` | `compute_quark_matter_thermo_from_n` | `thermo_from_n` | 2,3 | PUBLIC (8/2) |
| `eos/vmit/thermodynamics.py:401` | `compute_quark_matter_thermo_from_mu` | `thermo_from_mu` | 2,3 | PUBLIC |
| `eos/vmit/solver.py:54` | `VMITEOSResult` | `EoSPoint` (every other model) | 1,2 | PUBLIC |
| `eos/vmit/solver.py:107` | `get_default_guess_beta_eq` | `default_guess(mode, ...)` | 2,3 | PUBLIC |
| `eos/vmit/solver.py:138` | `get_default_guess_fixed_yc` | ″ | 2,3 | PUBLIC |
| `eos/vmit/solver.py:178` | `get_default_guess_fixed_yc_ys` | ″ | 2,3 | PUBLIC |
| `eos/vmit/solver.py:226` | `get_default_guess_trapped_neutrinos` | ″ | 2,3 | PUBLIC |
| `eos/vmit/solver.py:238` | `solve_vmit_beta_eq` | `solve_beta_eq_neutrinoless` | 1,2 | PUBLIC (39/43/12) |
| `eos/vmit/solver.py:337` | `solve_vmit_fixed_yc` | `solve_fixed_yc` | 1,2 | PUBLIC |
| `eos/vmit/solver.py:466` | `solve_vmit_fixed_yc_ys` | `solve_fixed_yc_ys` | 1,2 | PUBLIC |
| `eos/vmit/solver.py:590` | `solve_vmit_trapped_neutrinos` | `solve_beta_eq_neutrino_trapped` | 1,2 | PUBLIC |
| `eos/vmit/solver.py:682` | `result_to_guess` | `warm_start(point)` | 2 | PUBLIC |
| `eos/vmit/compute_tables.py:76,114,41` | `compute_vmit_table`, `save_vmit_results`, `VMITTableSettings` | see suspect 1 below | 1,2 | PUBLIC (notebook) |

## 2.2 `eos/dd2` — 16 deviations, 13 PUBLIC

| file:line | current | proposed | rule | vis. |
|---|---|---|---|---|
| `eos/dd2/parameters.py:35` | `Parametrization` | `Parameters` — §13 says this name in **every** model | 2 | PUBLIC (52 eos / **144 test** / 43 notebooks) |
| `eos/dd2/parameters.py:153` | `Parametrization.from_dd2_defaults()` | `Parameters.default()` — and `eos.dd2.…from_dd2_defaults` says dd2 twice | 1,2 | PUBLIC |
| `eos/dd2/parameters.py:199` | `Parametrization.from_dd2y_defaults()` | `Parameters.named("DD2Y")` | 1,2 | PUBLIC |
| `eos/dd2/solver.py:201` | `beta_warm_start` | `warm_start(point)` | 2 | PUBLIC |
| `eos/dd2/solver.py:208` | `default_beta_guess` | `default_guess(mode, ...)` | 2 | private |
| `eos/dd2/solver.py:389` | `octet_warm_start` | `warm_start(point, ...)` — two warm starts, neither with the vocabulary name | 2 | PUBLIC |
| `eos/dd2/solver.py:403` | `default_octet_guess` | `default_guess(mode, ...)` | 2 | private |
| `eos/dd2/solver.py:457` | `octet_unknowns` | `unknown_names(sys)` (sfho, did) | 2 | private |
| `eos/dd2/solver.py:491` | `octet_residual` | `residual(x, ctx, spec)` | 2 | private |
| `eos/dd2/solver.py:554` | `assemble_octet` | `assemble(x, ctx, spec)` | 2 | private |
| `eos/dd2/solver.py:707` | `solve_octet` | `solve(sys, x0)` (sfho, did) | 2 | PUBLIC (35/66/2) |
| `eos/dd2/solver.py:849` | `solve_beta_eq_octet` | `solve_beta_eq_neutrinoless` — dd2 has **no** function with any §3 mode name | 2 | PUBLIC |
| `eos/dd2/solver.py:888` | `solve_fixed_yc_octet` | `solve_fixed_yc` | 2 | PUBLIC |
| `eos/dd2/solver.py:911` | `solve_yl_octet` | `solve_beta_eq_neutrino_trapped` | 2 | PUBLIC |
| `eos/dd2/solver.py:927` | `sweep_beta_eq_octet` | fold into `sweep(...)`; `_octet` is a sector, not a mode | 2 | PUBLIC |
| `eos/dd2/solver.py:952` | `sweep_octet` | `sweep(...)` | 2 | PUBLIC |
| `eos/dd2/backends/jacobian.py:105` | `octet_jacobian` | `residual_jacobian` (sfho's name for the same job) | 2 | private |

`solve_beta_eq:285`, `solve_composition:101`, `solve_snm:190` and their `_t0`
twins are a second, nucleon-only entry-point family with no counterpart in any
other model. Not a §13 violation on its face, but it is why `_octet` had to be
appended to everything else, and no other model needs the suffix.

## 2.3 `eos/sfho` — 12 deviations, 9 PUBLIC

| file:line | current | proposed | rule | vis. |
|---|---|---|---|---|
| `eos/sfho/parameters.py` | `Parameters` has **no `default()` / `named()`** | add both; the eight `get_sfho*` functions become the published sets they wrap | 2 | PUBLIC (40 eos / 15 test) |
| `eos/sfho/parameters.py:338` | `get_sfho_nucleonic()` | `Parameters.named("nucleonic")` | 1,2,3 | PUBLIC |
| `eos/sfho/parameters.py:349` | `get_sfhoy_fortin()` | `Parameters.named("SFHoY_Fortin")` | 1,2,3 | PUBLIC |
| `eos/sfho/parameters.py:405` | `get_sfhoy_star_fortin()` | `Parameters.named(...)` | 1,2,3 | PUBLIC |
| `eos/sfho/parameters.py:457` | `get_sfho_2fam_phi()` | `Parameters.named(...)` | 1,2,3 | PUBLIC |
| `eos/sfho/parameters.py:523` | `get_sfho_2fam()` | `Parameters.named(...)` | 1,2,3 | PUBLIC |
| `eos/sfho/parameters.py:549` | `get_sfho_general(...)` | a `from_*` constructor | 1,3 | PUBLIC |
| `eos/sfho/parameters.py:667` | `get_all_parametrizations()` | `PUBLISHED_SETS` (njl, ccdm, enjl all carry that constant) | 2,3 | PUBLIC |
| `eos/sfho/nmp.py:235` | `create_custom_parametrization(...)` | a `from_*` constructor in `nmp.py` | 3 | PUBLIC |
| `eos/sfho/solver.py:737` | `solve_isentropic_beta_eq` | not a §3 mode — §3 says `SnB` is accepted *in place of* T, so this is `solve_beta_eq_neutrinoless(..., SnB=)` | 2 | PUBLIC |
| `eos/sfho/solver.py:763` | `solve_isentropic_trapped` | ″ | 2 | PUBLIC |
| `eos/sfho/thermodynamics.py:100` | `baryon_thermo` | `baryon_kinetics` (dd2) — and sfho has no single-species `kinetic_thermo` at all | 2 | private |
| `eos/sfho/thermodynamics.py:283` | `meson_field_thermo` | `field_eps_P` (dd2, did) | 2 | private |
| `eos/sfho/thermodynamics.py:512` | `get_residual_vector` | `self_consistency_residual` (dd2, did) | 2,3 | private |

`print_nmp_summary` (`eos/sfho/nmp.py:744`) and `print_params_summary`
(`eos/sfho/parameters.py:641`) print from inside the model. §5 forbids printing
in deep solver code; these are diagnostics rather than solver internals, so it
is a judgement call — but no other model has one.

## 2.4 `eos/mixed` — 13 deviations, all rule 1 (name repeats package), 11 PUBLIC

`eos.mixed.build_mixed_table` says "mixed" twice, exactly the disease §13 rule 1
names. This runs through the whole engine.

| file:line | current | proposed | vis. |
|---|---|---|---|
| `eos/mixed/solver.py:86` | `MixedResult` | `Result` / `EoSPoint` | PUBLIC |
| `eos/mixed/solver.py:224` | `mixed_slots` | `slots` | private |
| `eos/mixed/solver.py:300` | `MixedCtx` | `Ctx` | private |
| `eos/mixed/solver.py:353` | `build_mixed_ctx` | `build_ctx` | private |
| `eos/mixed/solver.py:583` | `solve_mixed` | `solve` (every other model) | PUBLIC |
| `eos/mixed/solver.py:738` | `sweep_mixed` | `sweep` | PUBLIC |
| `eos/mixed/solver.py:793` | `find_mixed_window` | duplicate of `boundaries.locate_window`; pick one | PUBLIC |
| `eos/mixed/table.py:133` | `solve_mixed_at_entropy` | `solve_at_entropy` (enjl's name) | PUBLIC |
| `eos/mixed/table.py:182` | `MixedTableSpec` | `TableSpec` (every other model) | PUBLIC |
| `eos/mixed/table.py:323` | `build_mixed_table` | `build_table` (§13 vocabulary) | PUBLIC |
| `eos/mixed/hybrid.py:51` | `MixedEoSTable` | `EoSTable` | PUBLIC |
| `eos/mixed/hybrid.py:118` | `build_mixed_eos_table` | second table builder beside `build_table`; the two need distinguishing by job, not by adjective | PUBLIC |
| `eos/mixed/boundaries.py:46` | `MixedWindow` | `Window` | PUBLIC |
| `eos/mixed/backends/jacobian.py:197` | `mixed_jacobian` | `residual_jacobian` (sfho) | private |

## 2.5 Cross-model vocabulary splits (same job, two names)

These are rule-2 violations that no single package owns.

| job | names in use | files |
|---|---|---|
| the phase-adapter surface: potentials in, block out | **`thermo_at_potentials`** (dd2, sfho, did) vs **`thermo_from_mu`** (zl, vmit, alphabag, njl, ccdm, abpr, enjl) | `dd2/thermodynamics.py:571`, `sfho/thermodynamics.py:556`, `did/thermodynamics.py:542` vs seven others. §13's vocabulary lists `thermo_from_mu`; **7–3 in its favour**. dd2/did additionally have a *separate* `thermo_from_mu` at a lower layer (`did/thermodynamics.py:358`), so this is a real two-layer distinction that needs a name for each, not a drift — but the split must be declared, and the top layer should not be the one carrying a name §13 does not list |
| the unknown-vector name list | `unknown_names` (sfho:313, did:222) vs `unknown_slots` (njl:128, ccdm:138, enjl:137) vs `octet_unknowns` (dd2:457) | three names, one job |
| the per-species loop | `baryon_kinetics` (dd2:388), `baryon_thermo` (sfho:100), `evaluate` (did:215) | `evaluate` also fails rule 3 — it says nothing about what it takes or returns. PUBLIC in did |
| the flat table row | `hadronic_row` (dd2:70, sfho:125, did:59), `quark_row` (vmit:81, alphabag:90, njl:69, ccdm:79), `nucleon_row` (zl:64), `beta_row`/`plateau_row` (enjl:151,208), `cfl_row` (abpr:127), `composition_row` (mixed:74) | zl's `nucleon_row` is the outlier among the hadronic models; the rest track their sector honestly |
| the solved-point record | `EoSPoint` (zl, sfho, did, alphabag, njl, ccdm, enjl, dd2) vs `VMITEOSResult` (vmit:54), `CFLPoint` (alphabag:130, abpr:107), `BetaPoint` (enjl:446), `MixedResult` (mixed:86) | `CFLPoint` is a genuinely different record; `VMITEOSResult` and `MixedResult` are not |
| the point constructor | `point_from_mu` (alphabag:229, abpr:290) vs `point_from_state` (njl:435, ccdm:483) | two names, and the signatures differ in kind (mu vs state), so this may be correct — worth one line in whichever `.tex` covers it |
| the legacy settings-object driver | `TableSettings` + `compute_table` + `save_results` present in `zl/table.py:215,250,292`, `sfho/table.py:378,524,609`, `alphabag/table.py:250,310,375`, and as `VMITTableSettings`/`compute_vmit_table`/`save_vmit_results` in `vmit/compute_tables.py` | four models carry a first-generation driver beside `build_table`; three of them are inside `table.py` (§5-conformant) and vmit's is not |

## 2.6 FILE names against §5

| package | finding |
|---|---|
| `eos/dd2` | **`notebook_api.py`** — §11 forbids `*notebook_api*` modules outright. See suspect 2 |
| `eos/vmit` | **`compute_tables.py`** — not in the §5 list. See suspect 1 |
| `eos/abpr` | **no `table.py`** although the grid driver exists. See suspect 3 |
| `eos/zl` | **no `nmp.py`** although the model has a nuclear sector. See suspect 4 |
| `eos/abpr` | no `responses.py`; `response_at_mu` sits in `solver.py:350`. §5 makes `responses.py` conditional ("when they outgrow api.py"), so this passes — but a response function in `solver.py` inverts §5's layer order, since `solver.py` is meant to hold the equilibrium conditions and their solves |
| `eos/mixed` | `construction.py` and `scan.py` are not in the §5 composite list nor in `docs/DEFERRED.md`'s enumeration of mixed's modules. §5 allows "whatever subpackages its solve needs", so they pass; the ledger is simply out of date |
| all eleven | `verify/run_full_check.py` present in every package ✓; `parameters.py`, `species.py`, `thermodynamics.py`, `solver.py`, `api.py` present everywhere they are required ✓ |

---

# 3. ORDER

§13: `thermodynamics.py` reads single species → mean fields → per-species loop →
sums; `solver.py` reads guesses → residual → solve → modes → sweep.

## 3.1 Serious

**`eos/dd2/solver.py` — ordered by sector, then by call depth; not by the physics.**

Actual order: `_nucleon_mu_effs:87` → **`solve_composition:101`, `solve_composition_t0:184`,
`solve_snm:190`, `solve_snm_t0:196`** → `beta_warm_start:201` → `default_beta_guess:208`
→ `BetaCtx:218` → `make_beta_ctx:239` → `beta_eq_nucleon_mu_eff:250` →
`beta_eq_residual:259` → `solve_beta_eq:285` → `solve_beta_eq_t0:365` →
`_octet_x0:375` → `octet_warm_start:389` → `default_octet_guess:403` →
`mode_spec:418` → `octet_unknowns:457` → `_unpack:477` → `octet_residual:491` →
`assemble_octet:554` → `_residual_and_jacobian:684` → `solve_octet:707` →
`solve_beta_eq_octet:849` → `solve_hadronic:863` → `solve_fixed_yc_octet:888` →
`solve_yl_octet:911` → `sweep_beta_eq_octet:927` → `sweep_octet:952`.

Two problems. **Four solves appear before any guess** — the file opens on
`solve_composition`. And the file is two sequential blocks (nucleon-only, then
octet), each internally guesses→residual→solve, so the reader meets *guesses,
residual, solve* twice. That is organisation by sector, which §13 rules out as
firmly as alphabetical: "the same reading order in every model is most of what
makes the second model quick to read."

**`eos/sfho/thermodynamics.py` — the per-species loop precedes the mean fields, and there is no single-species function.**

Actual order: `HadronState:40` → `HadronThermoResult:66` → **`baryon_thermo:100`
(per-species loop)** → **`field_residuals:221` (fields)** → `meson_field_thermo:283`
→ `meson_potentials:366` → `thermal_meson_thermo:405` → `thermo_from_mu:421`
(sums) → **`get_residual_vector:512`** → `thermo_at_potentials:556`.

The §13 order is single species → mean fields → loop → sums; sfho reads loop →
fields → sums, with `get_residual_vector` — a residual — landing *after* the sums.
The single-species step is not a function at all: it is inlined inside
`baryon_thermo`, which is why sfho is the only model with no `kinetic_thermo`.

## 3.2 Minor

| file | finding |
|---|---|
| `eos/zl/solver.py` | `warm_start:465` is **last**, after all four modes. §13 puts guesses first; every other model that has both puts `default_guess` and `warm_start` adjacent (njl:156/231, ccdm:166/261, enjl:168/241, mixed:123/171, did:260/320) |
| `eos/sfho/solver.py` | same: `warm_start:824` is the last definition, 567 lines after `default_guess:257` |
| `eos/vmit/solver.py` | same: `result_to_guess:682` (the warm start) is last |
| `eos/alphabag/solver.py` | same: `warm_start:862` is last, and `solve_cfl:758` sits after `_neutralizing_mu_e:730` |
| `eos/ccdm/thermodynamics.py` | the single-species function `mode_thermo:240` comes **after** the fields (`dielectric:134` … `bag_constant:217`). Everything else is in order; njl, its sibling, has the same layout, so the two are at least consistent with each other |
| `eos/did/thermodynamics.py` | the two guesses `field_estimate:481` and `cold_start:517` sit after `self_consistency_residual:453`. dd2 does the same (`_cold_start:671` after `thermo_at_potentials:571`), so the two are consistent; neither matches the solver ordering rule, which does not formally bind `thermodynamics.py` |

## 3.3 In order

`eos/{zl,dd2,did,vmit,alphabag,njl,abpr,enjl,mixed}/thermodynamics.py` and
`eos/{did,njl,ccdm,abpr,enjl,mixed}/solver.py`. `eos/did/solver.py`,
`eos/njl/solver.py` and `eos/mixed/solver.py` are textbook: guesses → residual →
solve → modes → sweep, with nothing out of place.

---

# 4. SELF-CONTAINED DOCSTRINGS AND CLEVERNESS

## 4a. References to a plan, a phase, a milestone or a working note — 8

§13: "State the physics, name the equation, give the literature citation — never
a plan, a phase, a milestone number, or a `docs/` working note."

| file:line | text |
|---|---|
| `eos/dd2/parameters.py:15` | `Every construction is validated in __post_init__ (the M0 ingest gate)` — milestone |
| `eos/dd2/parameters.py:284` | `"""Standalone ingest check, mirrors the M0 gate."""` — milestone |
| `eos/dd2/backends/kernel_numba.py:11` | `Correctness is guarded by backend parity (kernel_numba vs octet.py) in the M9 gate.` — milestone **and** a file that no longer exists (`eos/dd2/octet.py` is gone) |
| `eos/dd2/backends/kernel_numba.py:9` | `which Numba cannot trace (report D3)` — a working note not in the repository |
| `eos/dd2/verify/run_full_check.py:14` | `Backend parity (eos_ref vs eos_fast) is the M9 check, added with that backend.` — milestone |
| `eos/mixed/backends/jacobian.py:35` | `is what the Phase-1 autodiff attempt foundered on` — phase |
| `eos/mixed/boundaries.py:335`, `:390` | `(docs/SALVAGE.md section 1)`, `(the best predictor across the window, docs/SALVAGE.md section 2)` — `docs/` working note |
| `eos/mixed/table.py:241` | `first-generation hybrids used, docs/SALVAGE.md section 3` — `docs/` working note |

**Not counted, and why.** Twenty-odd docstrings cite `docs/DEFERRED.md`
(`eos/zl/species.py:50`, `eos/njl/api.py:62`, `eos/mixed/adapters.py:988`, …).
§11 makes `DEFERRED.md` a *tracked ledger of per-model gaps* and §3 requires
gaps to be recorded there, so a raise message pointing at it is sanctioned by
the specification rather than forbidden by it — it is not a working note. Three
docstrings cite `docs/njl_csc_implementation.md` and `docs/ccdm_implementation.md`
(`eos/njl/__init__.py:24`, `eos/njl/thermodynamics.py:45`,
`eos/njl/parameters.py:126`); those files *are* in the repository, but they are
implementation specifications rather than the model's own `.tex`, and §11 says
a physicist must be able to reproduce the model from `njl.tex` alone. Flagged as
a judgement call for the deciding ticket, not counted as a violation.

## 4b. Dense comprehensions and nested expressions — 9

Ranked by how much of the physics a named intermediate would recover.

| file:line | expression | what a name would show |
|---|---|---|
| `eos/njl/thermodynamics.py:388` | `[sum(modes[j].rho_s for j in range(N_MODES) if FLAVOUR_OF_MODE[j] == i) for i in range(3)]` (146 chars, nested) | this is `rho_s,f = sum over the colour modes of flavour f` — the per-flavour scalar density, the source of the gap equation. A `rho_s_per_flavour` loop would say so |
| `eos/ccdm/thermodynamics.py:408` | the same expression, 120 chars | ditto |
| `eos/enjl/thermodynamics.py:386` | `(sum(VALENCE[b][qi] * (M_q[q] - m0[q]) for qi, q in enumerate(QUARKS)) * d_alpha * n_s_b[b] for b in BARYONS)` (nested genexp inside a `sum`) | this is the third term of `Sigma^R_b`, `alpha_S'(n_B) sum_i [sum_q N^q_i (M_q - m_q0)] n^s_i` — written out in the docstring at `:346` and unrecognisable in the code |
| `eos/njl/solver.py:578` | `{} if x0 is None else dict(x0) if isinstance(x0, dict) else {patterns[0]: x0}` (nested ternary) | three-way seed dispatch as one expression |
| `eos/ccdm/solver.py:654` | the same nested ternary | ditto |
| `eos/enjl/solver.py:555` | `(all(abs(a - b) <= 1e-9 * max(1.0, abs(b)) for a, b in zip(seed, other)) for other in already)` | "have we already tried this seed" — a `seed_already_tried` helper |
| `eos/dd2/verify/compose.py:154` | `[[float(x) for x in ln.split()] for ln in lines[1:] if len(ln.split()) >= 10]` | parsing, not physics — low priority, but `ln.split()` runs twice per line |
| `eos/sfho/species.py:109` | `(abs(par.get_coupling(b.name, meson)) for b in group for meson in ("sigma", "omega", "rho"))` | two generators, no name for "the couplings of this multiplet" |
| `eos/mixed/scan.py:592` | `[dict(zip(keys, (float(v) for v in combo))) for combo in product(*grids)]` | nested; grid expansion |

Repeated three-line boilerplate that is not "clever" but is duplicated verbatim:
`njl/table.py:147,151,155` and `ccdm/table.py:166,170,174` are the same six
one-line comprehension properties (`P`, `eps`, `nB_solved`) in two files.

---

# The four suspects

## Suspect 1 — `eos/vmit/compute_tables.py`

**What it is.** A 203-line compatibility shim, not a solver. Its own module
docstring (`eos/vmit/compute_tables.py:1-13`) says so: "`VMITTableSettings` +
`compute_vmit_table` are the first-generation interface … Nothing is solved
here. The sweep, the warm start and the timing are `eos.general.tabulate` by way
of `eos.vmit.table`; this module only translates names and reshapes the result."
It maps four legacy equilibrium strings onto the §3 mode names
(`_LEGACY_MODES`, `:24`) and the legacy fraction-axis names onto spec names
(`_LEGACY_FRACTIONS`, `:32`), then calls `eos.vmit.table.build_table`.

**Who uses it.** Exactly one consumer, `notebooks/ZLvMIT_hybrid.ipynb`
(lines 75, 275, 292, 543, 560, 2356, 2370). **Nothing inside `eos/` imports it**
and no test does. That notebook is explicitly Out of scope in
[map.md](../map.md) ("The `zlvmit` legacy pair … Stage 0 keeps both
`ZLvMIT_hybrid.ipynb` and `zlvmit_test.ipynb`").

**Verdict: a deliberate, already-documented exception — leave it, but the name is
still wrong on two counts.** `docs/DEFERRED.md` records it: "`vmit/compute_tables.py`
is the one deliberate exception to the scheme: it is the first-generation
settings-object interface, kept because the ZLvMIT notebook drives vMIT through
it, and it now sits beside `table.py` as a shim over the shared driver rather
than being renamed to it." That reasoning holds — merging it into `table.py`
would put a legacy name-translation table inside the §5 grid driver, and zl,
sfho and alphabag all carry their equivalent *inside* `table.py`, which is worse,
not better.

What is **not** justified by that ruling is the three symbol names:
`VMITTableSettings`, `compute_vmit_table` and `save_vmit_results` each repeat the
package (`eos.vmit.compute_vmit_table`), which is §13 rule 1 and independent of
where the module lives. The parallel `zl` names were converted
(`compute_table`, `save_results`, `TableSettings`) and the vmit ones were not.

**Proposed:** keep the module and its `.py` name; rename the three symbols to
`TableSettings` / `compute_table` / `save_results` (PUBLIC — the ZLvMIT notebook
imports two of them by name), **or** record in `DEFERRED.md` that the legacy
names are frozen with the legacy notebook. Either is defensible; the current
state — a documented exception on the file, an undocumented one on the symbols —
is not.

## Suspect 2 — `eos/dd2/notebook_api.py`

**Confirmed forbidden and confirmed dead inside the package.** §11: "notebooks
call library functions and contain their own plotting code — there are no
`*notebook_api*` modules."

**What still imports it, exhaustively:**

| importer | line |
|---|---|
| `notebooks/DD2_usage.py` | `:76` — `from eos.dd2 import notebook_api as api` |
| `notebooks/DD2_usage.ipynb` | `:79` — the same line, and prose at `:14` and `:77` |

That is the whole list. **No module in `eos/` imports it, no test imports it,
and it is not in `eos/dd2/__init__.py`'s `__all__`.** The map's charting fact
holds ("imported only by `notebooks/DD2_usage.{py,ipynb}` — both of which Stage 0
removes"), and ticket 03 already owns the deletion.

Two further reasons it cannot simply be kept: it is the only module in `eos/`
that self-tests by printing (`:578`, `print("notebook_api self-check OK")`), and
`docs/DEFERRED.md:144` records that it "still imports astro, and dies with the
file" — a model module importing `astro/`, which §1 forbids outright.

**Verdict: delete with the notebook, as ticket 03 schedules. No blocker found.**

## Suspect 3 — `eos/abpr` has no `table.py`

**A real §5 gap, not an absence of physics — but the mildest one in the sweep.**

abpr **can** produce an `eos_table`: `eos/abpr/api.py:146`, a full grid driver
with the §5 progress-callback dictionary (`{mode, line, n_lines, temp_key, temp,
fracs, n_solved, n_requested, elapsed_s}` at `:189`), a `rows=True` path, and the
shared `print_progress`. Beside it sit `TableResult` (`:112`) and `cfl_row`
(`:127`) — the two records every other model keeps in `table.py`.

So the part exists, and §5 is explicit: "**The names are mandatory; the existence
is conditional.** A model does not carry an empty module to satisfy the template
— a single-file model is fine — but where it has one of these parts, that part
has this name."

What abpr genuinely lacks is only the *warm-started sweep*, and its docstring
already argues that absence correctly: "There is no warm start and no bisected
continuation here, and their absence is the physics rather than a gap: the
density inverse is a closed form (`eos.abpr.solver.mu_from_nB`), so no point
needs its neighbour and the grid is evaluated by array arithmetic." That is
right, and it is not an argument for keeping the driver in `api.py` — njl, ccdm
and enjl all have `table.py` files of 217, 237 and 404 lines carrying exactly
this trio.

**Proposed:** move `TableResult`, `cfl_row` and the body of `eos_table` to a new
`eos/abpr/table.py` as `TableResult` / `cfl_row` / `build_table(spec)`, leaving
`api.py:eos_table` the thin wrapper it is in every other model. Public API
unchanged (`eos.abpr.eos_table`, `eos.abpr.TableResult`, `eos.abpr.cfl_row` all
stay importable from the package root). Roughly 90 lines move.

Secondary: `response_at_mu` (`eos/abpr/solver.py:350`) is a response function
living in `solver.py`. §5 makes `responses.py` conditional, so a two-function
response sector may stay in `api.py` — but `solver.py` is the wrong home either
way, since §5 defines it as "the equilibrium conditions and the solves that
close them."

## Suspect 4 — `eos/zl` has no `nmp.py`

**Confirmed absent — and this is a real gap, not an absence of physics.**

ZL is a nucleonic density functional with a full nuclear sector. Six of its eight
parameters exist to set nuclear-matter parameters; `eos/zl/parameters.py:1-10`
says so ("six numbers set the six lowest nuclear-matter parameters almost
independently"), and `eos/zl/thermodynamics.py:84-98` derives the symmetry-energy
structure explicitly.

And the NMPs are **quoted in the code with no code behind them**.
`eos/zl/parameters.py:64-65`:

> `n_sat = 0.15951 fm^-3, E_sat = -16.00, K_sat = 250.2, E_sym = 30.85, L_sym = 41.26 MeV (measured from the code at T = 0; see zl.tex).`

A grep of `eos/zl/` for `compute_nmp`, `invert_nmp`, `esym`, `energy_per_baryon`
or any saturation solve returns **nothing**. Those six numbers were computed once
by something that is not in the package and pasted into a docstring. Nothing in
`verify/run_full_check.py` reproduces them, so if a parameter changed the
docstring would go silently stale — which is precisely the failure mode §12's
golden-reference rule exists to prevent, and `eos/dd2/verify` pins its published
NMPs for exactly this reason.

Against that: **§5 names only `dd2` and `sfho`** as the models that expose the
map ("Models with a nuclear sector (`dd2`, `sfho`) expose the forward map …"), so
the letter of the specification does not require `zl/nmp.py`. But `eos/did/nmp.py`
exists (224 lines, `compute_nmp:145`) and did is not in that list either, so the
list reads as illustrative rather than exhaustive — and did is the precedent that
settles it. Note also that §5 puts `nmp.py` at the *top* of the import order
because "computing nuclear-matter parameters requires solving symmetric matter at
saturation", which zl can do today via `thermo_from_n(n_B, Y_C=0.5, T=0)`
(`eos/zl/thermodynamics.py:374`).

`docs/DEFERRED.md` records nothing about a missing zl NMP map. Its zl section
(`:935` onward) covers the response-function and `SnB`-axis gaps only.

**Verdict: a real gap.** Either (a) add `eos/zl/nmp.py` with `compute_nmp` —
the forward map only; the inverse is a separate question, since ZL's six
parameters map almost one-to-one onto the NMPs and inverting may be trivial or
may not — and pin the six published numbers in `eos/zl/verify/run_full_check.py`;
or (b) record in `docs/DEFERRED.md` that zl has no NMP map and that the numbers
in `parameters.py:64` are external. **(a) is the right answer** — the docstring
already claims the numbers were "measured from the code", and no code measures
them.

---

# Appendix — the PUBLIC rename list, for human approval

58 public renames. Grouped by package, ordered by blast radius. Counts are
call-site occurrences in `eos/` / `test/` / `notebooks/`.

**`eos/dd2` (13)** — total blast radius ≈ 240 sites, over half of them in `test/`

1. `Parametrization` → `Parameters` (52/144/43)
2. `Parametrization.from_dd2_defaults()` → `Parameters.default()`
3. `Parametrization.from_dd2y_defaults()` → `Parameters.named("DD2Y")`
4. `solve_octet` → `solve` (35/66/2)
5. `solve_beta_eq_octet` → `solve_beta_eq_neutrinoless`
6. `solve_fixed_yc_octet` → `solve_fixed_yc`
7. `solve_yl_octet` → `solve_beta_eq_neutrino_trapped`
8. `sweep_octet` → `sweep`
9. `sweep_beta_eq_octet` → fold into `sweep`
10. `beta_warm_start` → `warm_start`
11. `octet_warm_start` → `warm_start`
12. `solve_composition` / `solve_snm` and their `_t0` twins — keep, but decide whether they survive the `_octet` unsuffixing
13. `notebook_api.py` — delete (ticket 03)

**`eos/vmit` (26)** — total blast radius ≈ 250 sites

14. `get_vmit_default()` → `Parameters.default()` (55/27/0)
15. `get_vmit_custom()` → delete (6/12/13)
16. `solve_vmit_beta_eq` → `solve_beta_eq_neutrinoless` (39/43/12 across the family)
17. `solve_vmit_fixed_yc` → `solve_fixed_yc`
18. `solve_vmit_fixed_yc_ys` → `solve_fixed_yc_ys`
19. `solve_vmit_trapped_neutrinos` → `solve_beta_eq_neutrino_trapped`
20. `VMITEOSResult` → `EoSPoint`
21. `VMITThermo` → `MatterThermo`
22. `result_to_guess` → `warm_start`
23–26. `get_default_guess_{beta_eq,fixed_yc,fixed_yc_ys,trapped_neutrinos}` → `default_guess(mode, ...)`
27. `compute_quark_thermo` → `kinetic_thermo`
28. `compute_quark_density` → `quark_density`
29. `compute_vector_field` → `vector_field`
30. `compute_vector_pressure` → `vector_pressure`
31. `compute_vector_energy` → `vector_energy`
32. `compute_bag_pressure` → `bag_pressure`
33. `compute_bag_energy` → `bag_energy`
34. `compute_mu_effective` → `effective_potential`
35. `compute_effective_mu_quarks` → `effective_potentials`
36. `compute_mu_physical` → `physical_potentials`
37. `compute_quark_densities_for_solver` → `effective_state`
38. `compute_vmit_thermo_from_mu_n` → `thermo_from_mu_n`
39. `compute_quark_matter_thermo_from_n` → `thermo_from_n` (8/2/0)
40. `compute_quark_matter_thermo_from_mu` → `thermo_from_mu`
41–43. `VMITTableSettings` / `compute_vmit_table` / `save_vmit_results` → `TableSettings` / `compute_table` / `save_results` (ZLvMIT notebook — see suspect 1)

**`eos/sfho` (9)** — total blast radius ≈ 60 sites

44. add `Parameters.default()` and `Parameters.named(name)` (40/15/0 for the `get_sfho*` family)
45–49. `get_sfho_nucleonic` / `get_sfhoy_fortin` / `get_sfhoy_star_fortin` / `get_sfho_2fam_phi` / `get_sfho_2fam` → `Parameters.named(...)`
50. `get_sfho_general(...)` → a `from_*` constructor
51. `get_all_parametrizations()` → `PUBLISHED_SETS`
52. `create_custom_parametrization(...)` → a `from_*` constructor in `nmp.py`
53. `solve_isentropic_beta_eq` / `solve_isentropic_trapped` → `SnB=` on the mode solvers

**`eos/mixed` (11)** — internal to the repo; `nucleation` does not import `eos.mixed`

54. `MixedResult` → `Result`
55. `solve_mixed` → `solve`
56. `sweep_mixed` → `sweep`
57. `find_mixed_window` → merge into `locate_window`
58. `solve_mixed_at_entropy` → `solve_at_entropy`
59. `MixedTableSpec` → `TableSpec`
60. `build_mixed_table` → `build_table`
61. `MixedEoSTable` → `EoSTable`
62. `build_mixed_eos_table` → a name distinguishing it from `build_table` by job
63. `MixedWindow` → `Window`

**`eos/did` (1)**

64. `evaluate` → `baryon_kinetics`

**Cross-model (1 decision, then up to 3 renames)**

65. `thermo_at_potentials` (dd2, sfho, did) vs `thermo_from_mu` (seven models) — §13's vocabulary lists `thermo_from_mu`; dd2 and did carry both at two layers, so the decision is what to call the *upper* layer, not whether to rename blindly

## Downstream

`nucleation` (branch `paper-release`) imports `eos.general` (13), `eos.alphabag`
(11), `eos.sfho` (6) and `eos.tov` (5). A grep of the whole `nucleation` tree for
`get_sfho`, `Parametrization`, `solve_vmit` and `get_vmit` returns **nothing**:
none of the 58 public renames above touches a `nucleation` call site.

Noticed while checking that, and **not** part of this ticket: `nucleation` imports
`eos.tov.solver` at `nucleation/analysis/{config,filters,replay,stellar}.py` and
`notebooks/2fam_PNS_nucleation.py`. `eos/eos/tov/` does not exist — §11 puts the
module at `eos/astro/tov`. Those five imports are already broken. For the Stage 7
report.
