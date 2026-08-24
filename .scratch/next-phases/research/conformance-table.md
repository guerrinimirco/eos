# Where does the repository actually stand against CLAUDE.md?

Ticket: [`08-conformance-table.md`](../issues/08-conformance-table.md) · Parent: [`map.md`](../map.md)
Repo state: `main` at `136c57c`, working tree clean. Read-only audit — nothing in
`eos/`, `test/`, `docs/` or `CLAUDE.md` was modified.

`CLAUDE.md` (576 lines) and `docs/DEFERRED.md` (1690 lines) were read end to end.
Every claim below carries `file:line` evidence or pasted tool output.

**Recorded as instructed:** `docs/STRUCTURE.md` does not exist.
`CLAUDE.md` §10 (line 424, "a worked figure example in `docs/STRUCTURE.md`") and
§11 (line 460, "docs/, incl. STRUCTURE.md") both reference it. It is already in
the ledger at `docs/DEFERRED.md:529` and belongs to ticket 21. **Not written here.**

---

## 1. The conformance table

One row per unit, one column per checkable claim.
**P** = Pass · **F** = Fail · **A** = Ambiguous · **–** = N/A.

| unit | §1 layering | §2 conventions | §3 modes | §4 flags | §5 layout/API | §6 inference | §7 integrals | §8 verify | §10 styling | §11/§12 docs+tests |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `zl`          | P | P | P | P | **F** | **A** | P | P | P | P |
| `sfho`        | P | **A** | P | P | **F** | **F** | P | **A** | P | P |
| `dd2`         | **F** | **F** | P | **F** | **F** | P | **A** | **F** | **A** | **F** |
| `did`         | P | P | P | P | **A** | P | P | P | P | P |
| `vmit`        | P | P | P | P | **A** | P | P | P | P | P |
| `alphabag`    | P | **A** | P | **A** | P | **F** | P | P | P | P |
| `njl`         | P | P | P | P | **A** | **F** | P | **A** | P | P |
| `ccdm`        | P | P | P | P | **A** | **F** | P | **F** | P | P |
| `abpr`        | **A** | P | **F** | P | **A** | **F** | P | P | P | P |
| `enjl`        | **A** | **A** | P | P | **A** | P | P | P | P | P |
| `mixed`       | P | **A** | **F** | **F** | **F** | **F** | P | **A** | P | P |
| `zlvmit`      | – | P | – | – | – | **F** | P | – | P | **F** |
| `astro/tov`   | P | – | – | – | – | **A** | P | **F** | P | **A** |
| `astro/gmode` | **F** | P | – | – | – | **F** | P | – | P | P |
| `general`     | P | P | P | – | **A** | **A** | P | **A** | P | **A** |

**Totals: 24 Fail, 25 Ambiguous.**
150 cells; 14 are `–` (not applicable), leaving 136 scored: **87 Pass, 24 Fail,
25 Ambiguous**. Per unit (F/A): `dd2` 6/2 · `mixed` 4/2 · `sfho` 2/2 ·
`ccdm` 2/1 · `abpr` 2/2 · `zlvmit` 2/0 · `astro/gmode` 2/0 · `zl` 1/1 ·
`alphabag` 1/2 · `njl` 1/2 · `astro/tov` 1/2 · `did` 0/1 · `vmit` 0/1 ·
`enjl` 0/3 · `general` 0/4.

Cross-cutting facts that sit outside any one row:

- `docs/STRUCTURE.md` **does not exist** (ledgered, ticket 21).
- `output/public/` **does not exist** — `.gitignore:37-39` reserves it
  (`output/`, `!output/public/`, `!output/public/**`) and §11 line 458 calls it
  "the curated tracked subfolder", but the directory has never been created.
- `CLAUDE.md`'s own model enumerations are stale: §1 line 22 omits `did`, `njl`,
  `ccdm`; §11 line 449 omits `njl`, `ccdm`. `ccdm` appears **nowhere** in
  `CLAUDE.md` (`grep -n "ccdm" CLAUDE.md` → no output).
- §5 line 231's adapter list ("DD2, SFHo, ZL, DID, vMIT, alphaBag, ENJL branch
  pair") omits the shipped `njl_phase` (`eos/mixed/adapters.py:1051`) and
  `ccdm_phase` (`:1189`).

---

## 2. §1 layering — verified by import graph, not by eye

Script: `importgraph.py` (written to the session scratchpad, **not** the repo).
It walks every `.py` under `eos/` with `ast`, resolves relative and absolute
imports to dotted targets, buckets each edge by source and destination unit, and
prints the violations. Real output:

```
========================================================================
A. eos -> nucleation (must be empty)
========================================================================
  (none)

========================================================================
B. model -> another model (must be empty)
========================================================================
  [abpr] eos/abpr/verify/run_full_check.py:47  ->  eos.alphabag.parameters.Parameters
  [abpr] eos/abpr/verify/run_full_check.py:48  ->  eos.alphabag.thermodynamics.cfl_thermo_from_mu
  [dd2]  eos/dd2/notebook_api.py:484           ->  eos.sfho.table.compute_table
  [dd2]  eos/dd2/notebook_api.py:484           ->  eos.sfho.table.TableSettings

========================================================================
C. model -> astro (must be empty; mixed/ is the ONE named exception)
========================================================================
  [dd2] eos/dd2/notebook_api.py:34  ->  eos.astro.tov.crust.have_crust
  [dd2] eos/dd2/notebook_api.py:35  ->  eos.astro.tov.solver.compute_tov_sequence
  [dd2] eos/dd2/notebook_api.py:35  ->  eos.astro.tov.solver.find_mmax_precise
  [dd2] eos/dd2/notebook_api.py:35  ->  eos.astro.tov.solver.generate_ec_logspace

========================================================================
D. mixed -> astro (allowed only in hybrid.py / scan.py per CLAUDE.md §1)
========================================================================
  eos/mixed/hybrid.py:235  ->  eos.astro.tov.crust.have_crust
  eos/mixed/hybrid.py:236  ->  eos.astro.tov.solver.compute_tov_sequence
  eos/mixed/scan.py:188    ->  eos.astro.tov.crust.have_crust
  eos/mixed/scan.py:189    ->  eos.astro.tov.solver.compute_tov_sequence

========================================================================
E. general -> anything else in eos (general must import nothing in repo)
========================================================================
  (none)

========================================================================
F. astro -> model internals (astro must not import model internals)
========================================================================
  eos/astro/gmode/rates.py:85                 ->  eos.dd2.solver.solve_composition
  eos/astro/gmode/sound_speeds.py:94          ->  eos.mixed.responses.sound_speed_eq
  eos/astro/gmode/sound_speeds.py:94          ->  eos.mixed.responses.sound_speed_frozen
  eos/astro/gmode/sound_speeds.py:149         ->  eos.dd2.solver.solve_composition
  eos/astro/gmode/verify/run_full_check.py:39 ->  eos.dd2.Parametrization
  eos/astro/gmode/verify/run_full_check.py:40 ->  eos.dd2.responses.sound_speed_eq
  eos/astro/gmode/verify/run_full_check.py:41 ->  eos.dd2.solver.sweep_beta_eq_octet

========================================================================
H2. model/engine -> mixed (a model must not import the composite engine)
========================================================================
  [enjl] eos/enjl/verify/run_full_check.py:817  ->  eos.mixed.construction.enjl_coexistences
  [enjl] eos/enjl/verify/run_full_check.py:834  ->  eos.mixed.adapters.enjl_branch_pair
  [enjl] eos/enjl/verify/run_full_check.py:835  ->  eos.mixed.boundaries.total_pressure

========================================================================
I. per-unit outbound summary (unit -> set of units it imports)
========================================================================
  abpr       -> alphabag, general
  alphabag   -> general
  astro      -> dd2, general, mixed
  ccdm       -> general
  dd2        -> astro, general, sfho
  did        -> general
  enjl       -> general, mixed
  general    -> (nothing outside itself)
  mixed      -> alphabag, astro, ccdm, dd2, did, enjl, general, njl, sfho, vmit, zl
  njl        -> general
  sfho       -> general
  vmit       -> general
  zl         -> general
  zlvmit     -> general, vmit, zl

scanned 169 .py files under eos/, 2489 intra-repo import edges
```

Reading of each bucket:

- **A — `eos` never imports `nucleation`: clean**, and `test/test_imports.py:44
  test_eos_never_imports_nucleation` enforces it with a `sys.meta_path` guard.
- **C, F — two genuine breaches.** `eos/dd2/notebook_api.py` imports `astro/`
  (finding 1) and `eos/astro/gmode` imports model internals (finding 2).
- **B, H2 — three `verify/` reach-arounds.** `test/test_imports.py:88-104`
  deliberately exempts `verify/` from the model-to-model half of the rule; the
  document contains no such carve-out (finding 3).
- **D — the named exception, honoured exactly.** `mixed/hybrid.py` and
  `mixed/scan.py` are the only two files, which is precisely what §1 line 30 and
  `docs/DEFERRED.md:122-158` say. Six of the seven other model→astro edges the
  ledger names as removed are indeed gone.
- **E — `general/` imports nothing else in the repo.** The one previously
  ledgered cycle (`sfho/compose_loader.py` ↔ `astro/tov/solver`) is closed;
  `eos/general/compose.py` now holds it.
- `test/test_imports.py:108 MODEL_PACKAGES` omits `ccdm`, so `eos.ccdm` is
  outside the automated layering gate (finding 4). Its graph row is clean today.

---

## 3. §2 conventions

**(a) S = +1 per s quark — PASS in all 15 units, no PDG leak anywhere.**
Only three places in the repository write a strangeness number down:

```
eos/general/particles.py:59:   strangeness: Optional[float] = None  # S (s-quark = +1)
eos/general/particles.py:203:  strangeness=+1.0, ...   # Lambda
eos/general/particles.py:229:  strangeness=+2.0, ...   # Xi0
eos/general/particles.py:235:  strangeness=+2.0, ...   # Xi-
eos/general/particles.py:355:  strangeness=+1.0, ...   # s quark
eos/general/particles.py:298:  strangeness=-1.0, ...   # K+ = u sbar  (correct: ANTI-s)
eos/general/particles.py:402:  Our convention:   Q = I3 + (B - S)/2  where s-quark has S = +1
eos/general/pairing.py:109:    STRANGENESS = np.array([0.0, 0.0, 1.0])
eos/enjl/species.py:57:        STRANGENESS = {"p": 0.0, "n": 0.0, "Lambda": 1.0, "u": 0.0, "d": 0.0, "s": 1.0, ...}
```

Every other unit derives S from one of these. `abpr` is the sharpest test —
`eos/abpr/verify/run_full_check.py:165` asserts CFL has `Y_C = 0` and
`Y_S = +1`, which under PDG would be −1. Correct.

**(b) Y_C non-leptonic — PASS in all 15.** `eos/general/basis.py:71-76` skips
leptons structurally (`if particle.is_lepton: continue`), and every model keeps
neutrality as a separate residual row. `eos/enjl/solver.py:391,394` is the
clearest statement of the distinction: the fixed-Y_C row excludes leptons, the
neutrality row includes them.

*One recorded exception, and it is the only §2 code failure:*
`eos/dd2/table.py hadronic_row` emits a Y_C and Y_S that are **baryons only**,
dropping a thermal meson gas that `eos/mixed` counts under the same column name
— a 10–20 % difference at T = 40 MeV. Ledgered at `docs/DEFERRED.md:722-739`,
which itself says "it contradicts CLAUDE.md §2".

**(c) mu_C = mu_p − mu_n, beta equilibrium mu_C + mu_e = 0 — PASS in all 15.**

```
eos/general/basis.py:112:  mu_p = mu_B + mu_C and mu_n = mu_B, hence mu_C = mu_p - mu_n.
eos/general/basis.py:135:  mu_C = mu_u - mu_d
eos/general/modes.py:245:  return mu_nue - mu_C
eos/vmit/solver.py:286:    eq6 = mu_u + mu_e - mu_d          # literally mu_C + mu_e
eos/zl/solver.py:222:      r4 = mu_n - mu_p - mu_e           # = -(mu_C + mu_e)
eos/dd2/solver.py:341:     beta_res = -base.matter.mu_C - mu_e
eos/mixed/boundaries.py:565: Beta equilibrium fixes mu_e = -mu_C (CLAUDE.md section 2: mu_C + mu_e = 0)
```

Zero occurrences of `mu_C - mu_e`; zero occurrences of `mu_n - mu_p` assigned to
`mu_C`. The kaon potentials, where a mu_S sign flips easily, also check out
(`eos/sfho/thermodynamics.py:400` `mu_K_plus = mu_C - mu_S - ...`, consistent
with K+ carrying C=+1, S=−1).

**(d) Basis maps imported from `general/` — four local re-derivations.**
`eos/general/basis.py` offers `charges_of`, `charges_from_densities`,
`quark_charges`, `species_potential`, `quark_potentials`,
`charge_potentials_from_quarks`, `baryon_potentials`. Findings 12–14 below.

---

## 4. §3 modes

**I** implemented · **R** accepted but raises · **–** absent.

| unit | beta_eq_neutrinoless | beta_eq_neutrino_trapped | fixed_YC | fixed_YC_YS | SnB for T | gaps in DEFERRED? |
|---|:-:|:-:|:-:|:-:|:-:|---|
| `zl`       | I `solver.py:210` | I `:387` | I `:282` | **R** `:376` | point ✔ / **table R** `table.py:118` | ✔ both |
| `sfho`     | I `:634` | I `:714` | I `:658` | I `:687` | ✔ / ✔ (**R** for isentropic fixed_YC+leptons `:581`) | ✔ |
| `dd2`      | I | I `:911` | I `:888` | I (**R** with `leptons=True`, `api.py:56`) | ✔ / ✔ | ✔ |
| `did`      | I `:590` | I `:600` | I `:612` | I `:626` | ✔ / ✔ | ✔ |
| `vmit`     | I `:238` | I `:590` | I `:337` | I `:466` | ✔ / **table R** `table.py:140` | ✔ |
| `alphabag` | I `:393` | I `:471` | I `:556` | I `:645` | ✔ / **table R** `table.py:148` | ✔ |
| `njl`      | I `:609` | I `:615` | I `:622` | I `:629` | ✔ / ✔ | ✔ |
| `ccdm`     | I `:685` | I `:695` | I `:705` | I `:718` | ✔ / ✔ | ✔ |
| `abpr`     | **R** | **R** | **R** | **R** | only SnB = 0 | **✗ NOT recorded** |
| `enjl`     | I `:651` | I `:663` | I `:675` | I `:683` | ✔ (one thermal value/call) | ✔ |
| `mixed`    | I `:706` | I `:711` | I `:717` | I `:727` | ✔ / ✔ | Y_Lmu **✗ NOT recorded** |
| `zlvmit`   | own solvers | own | own | absent | own | exemption ✔ |
| `general`  | declares all four `ModeSpec` factories, `modes.py:145-179` | | | | `tabulate.temperature_at_entropy` | — |

`Y_Lmu` (the optional muon-family axis of `beta_eq_neutrino_trapped`) is
implemented in **zero** units; `eos/general/modes.py:155-167` builds the
`L_mu=FIXED` spec and every consumer refuses it. Nine of ten refusals are
ledgered; `mixed/api.py:81-84` is not.

A fifth mode name, `cfl`, exists outside the §3 set
(`eos/alphabag/solver.py:61`, `eos/abpr/solver.py:48`) and is declared nowhere.

Every refusal is a real raise with a message, never a silent skip — `abpr`'s
`MODE_REFUSALS` (`eos/abpr/solver.py:54-73`) is the best-written of them, one
physics reason per mode. Its ledger entry is what is missing, not its manners.

---

## 5. §4 species flags — the silent-ignore hunt

Method: parse each `species.py` with `ast` for the `SpeciesFlags` fields, then
count word-boundary reads of each field name **outside** `species.py` across the
whole model package. A field with zero outside reads is either (i) validated by
`__post_init__` and correctly refused, or (ii) silently ignored — the failure to
hunt. Every zero-read field was then checked against its `__post_init__`.

| unit | hyperons | deltas | muons | thermal_mesons | thermal_neutrinos | photons | extra |
|---|:-:|:-:|:-:|:-:|:-:|:-:|---|
| `zl`       | R | R | R | R | R | I | — |
| `sfho`     | I | I | R | I | I | I | `phi_field` R-if-False |
| `dd2`      | I | I | I | **absent** | **absent** | **🔴 IGNORED** | `neutrinos` I, `phi_field` I, `sigma_star` R, `include_pseudoscalars` I, `include_thermal_vectors` I |
| `did`      | I | I | I | I | I | I | `phi_field` R-if-False |
| `vmit`     | R | R | R | R | R | I | — |
| `alphabag` | R | R | R | R | **I, dropped on `cfl`** | I | `gluons` I |
| `njl`      | R | R | I | R | I | I | `csc` I |
| `ccdm`     | R | R | I | R | I | I | `csc` I |
| `abpr`     | R | R | R | R | R | R | `gluons` R |
| `enjl`     | R (pinned True) | R | R (pinned True) | R | I | I | — |
| `mixed`    | I via dd2 | I via dd2 | I | absent | absent | **🔴 IGNORED** | reuses `eos.dd2.species.SpeciesFlags` |
| `zlvmit`   | no `SpeciesFlags` at all | | | | | | exempt |

Every `R` is a real `NotImplementedError` from `__post_init__` naming the
physics — e.g.

```
eos/zl/species.py:42-48   "hyperons/deltas/thermal_mesons has no coupling in the
                           Zhao-Lattimer functional ... the sector is absent from
                           the model, not merely unimplemented"
eos/vmit/species.py:43    "... is a hadronic sector and has no meaning in
                           deconfined quark matter"
eos/abpr/species.py       _WHY_OFF, one reason per flag
```

That half of §4 is in excellent shape. The two red cells are findings 5 and 6.

---

## 6. §5 layout and API

**`par` first and never optional — PASS, 33/33 entry points.**
`grep -rn "def eos_\(point\|table\|response\)(par=" eos/*/api.py` returns nothing.

```
eos/dd2/api.py:62:  def eos_point(par, mode, species, n_B, T=None, SnB=None, x0=None, analytic_jac=True, **conditions)
eos/dd2/api.py:113: def eos_table(par, mode, species, axes, fixed=None, skip_errors=True, progress=None, verbose=False)
eos/dd2/api.py:142: def eos_response(par, mode, species, frozen="equilibrium", n_B=None, T=0.0, Y_p=None, **conditions)
eos/njl/api.py:73:  def eos_point(par, mode="beta_eq_neutrinoless", species=None, n_B=None, ...)
eos/abpr/api.py:73: def eos_point(par, mode="cfl", species=None, n_B=None, T=0.0, SnB=None, **conditions)
```

Condition names are exactly `n_B, T, Y_C, Y_S, Y_Le, Y_Lmu` everywhere, with one
leak (`Y_p`, `eos/dd2/api.py:142`) and one smuggled non-condition (`leptons`
popped from `**conditions` at `eos/sfho/api.py:56` and `eos/dd2/api.py:53`).
`mode` acquired a default in four models (findings 15–16).

**Progress callback — one shared home, one real defect.**
`eos/general/tabulate.py:230-238` builds the canonical dict and is used by `zl`,
`did`, `vmit`, `alphabag`, `njl`, `ccdm`, `enjl`, `mixed`, `abpr`. `dd2` and
`sfho` build their own with the same key set. `mixed` adds `eta`/`window` and
`njl`/`ccdm` add `pattern`/`branch`, all of which §5 line 185 explicitly permits.
**No unit renames or drops a required key.** But `fracs` loses the *fixed*
fractions in `dd2` and `sfho` (finding 7).

**`thermodynamics.py` free of `beta`/`Y_C`/`neutral`/`trapped` — real grep output**
(`grep -n -iE "beta|Y_C|neutral|trapped" eos/<model>/thermodynamics.py`):

```
### mixed
  (no matches)

### abpr
185:    -- the phase is electrically neutral by construction, with no leptons, and

### sfho
372:    list, the conjugation mu*_{j-} = -mu*_{j+}, the neutral strangeless mesons
563:    `thermo_from_mu`. No leptons, no neutrality, no held fraction - those are

### dd2
7:`solver.py`. Grep this file for `beta`, `Y_C`, `neutral` or `trapped` and find
436:    that neutrality and the fixed-Y_C / fixed-Y_S conditions are stated in
516:    There is no charge, strangeness or neutrality row: the potentials are
578:    leptons, no neutrality, no held fraction -- those are conditions on a
614:            # builds it from a full beta-equilibrium solve) pays for it only

### njl
9:satisfy. IT NEVER KNOWS WHICH EQUILIBRIUM MODE IT IS IN; imposing beta
14:paired, the two colour potentials that make it colour-neutral. That is the
16:neutrality is a structural property of a colour-superconducting phase, not a
266:    `n_3` and `n_8` are the colour densities that colour neutrality sets to
522:        colour neutrality           n_3 = 0 ,  n_8 = 0        (paired only)
542:    Closes the model's own internal system -- masses, gaps, colour neutrality

### ccdm
11,15,16,275,512,578   (the identical colour-neutrality set)

### enjl
8:which equilibrium mode it is in; imposing beta equilibrium or a charge
370:    `n_B` is passed rather than summed from `n` because the beta-equilibrium
726:    returns the block. No leptons, no neutrality, no held fraction: those are

### vmit
78:    Y_C: float = 0.0       # Charge fraction n_C/n_B
338:    Y_C = n_C / n_B
345:        Y_C=Y_C, Y_S=Y_S,

### alphabag
95,107,129,398,566,575   (the identical output-field set)

### did
~50 hits, every one the DID isovector field `beta = sum_i tau_3i n_i / n_B`
130:    beta: float          # dimensionless
475:            fields.beta - matter.n_3 / fields.n_B,     # a residual row

### zl
276:    Y_C: float = 0.0       # non-leptonic charge fraction
305:    mu_C = mu_p - mu_n, so beta equilibrium reads mu_C + mu_e = 0.
331:        Y_p=..., Y_C=n_C / n_B, Y_S=n_S / n_B,
374:def thermo_from_n(n_B: float, Y_C: float, T: float,          <- GENUINE
378:    With the composition fixed, no fixed point is needed: n_p = Y_C n_B and
386:    n_p = Y_C * n_B                                          <- GENUINE
387:    n_n = (1 - Y_C) * n_B
```

Verdict: `mixed` is literally clean. `abpr`, `sfho`, `dd2`, `njl`, `ccdm`, `enjl`
hit only docstrings and comments, several of which quote the rule at itself.
`did`'s ~50 `beta` hits are a *mean field* the solver solves for, and
`did/thermodynamics.py:7` says so, deliberately omitting `beta` from the grep it
quotes. `vmit` and `alphabag` hit an **output** field (`Y_C` is reported, never
consumed) — the grep over-triggers on a legitimate return value. **`zl` is the
one substantive breach** (finding 8).

**Layer order — no upward *top-level* import in any model.** `couplings.py` has
zero intra-package imports everywhere it exists (`dd2`, `did`, `njl`, `ccdm`);
no `parameters.py` imports `thermodynamics`/`solver` at top level; no
`thermodynamics.py` imports `solver`. Four deferred (function-local) imports are
cycles announcing themselves — findings 9–10.

**`backends/` deletable** — three packages have one (`dd2`, `sfho`, `mixed`).
Every reference-path import is guarded:

```
eos/dd2/thermodynamics.py:38-44   try: from eos.dd2.backends.kernel_numba import ...
                                  except ImportError:
                                      # `backends/` is optional: CLAUDE.md section 5 defines it by
                                      # the property that deleting it changes no number ...
eos/dd2/solver.py:67-78           same shape
eos/sfho/solver.py:79-85          same shape
eos/mixed/*                       lazy, inside functions only
```

Selection is an `analytic_jac=` boolean (`dd2` default True; `sfho`, `mixed`
default False), not the `backend=` string §9 suggests. **One unguarded import
breaks the property** (finding 11).

**File layout vs the §5 template:**

| model | parameters | species | thermo | solver | table | api | verify/ | .tex | .md | couplings | nmp | responses | backends |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| zl | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | – | – | – |
| sfho | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ |
| dd2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| did | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – |
| vmit | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | – | – | – |
| alphabag | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | – | – | – |
| njl | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | – |
| ccdm | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | – |
| abpr | ✓ | ✓ | ✓ | ✓ | **✗** | ✓ | ✓ | ✓ | ✓ | – | – | – | – |
| enjl | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | – | – | – |
| mixed | – | – | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | – | ✓ | ✓ |
| zlvmit | – | – | – | – | – | – | **✗** | **✗** | **✗** | – | – | – | – |
| astro/tov | – | – | – | – | – | – | **✗** | ✓ | ✓ | – | – | – | – |
| astro/gmode | – | – | – | – | – | – | ✓ | ✓ | ✓ | – | – | – | – |

`find eos -name "eos.py"` → nothing. **PASS.**
`find eos -name "thermodynamics_*.py"` → `eos/general/thermodynamics_leptons.py`,
exactly the "package holding exactly one suffixed file" §5 line 256 calls wrong
(though `general/` is not a model, so the rule applies only by analogy).
Two extra non-template modules: `eos/dd2/notebook_api.py` (forbidden by name,
§11 line 457) and `eos/vmit/compute_tables.py` (a second table driver beside
`vmit/table.py`).

---

## 7. §6 — parameters, non-convergence, global state, pickling, arrays

**Parameters are arguments.** Published named sets are correctly done
(`njl/parameters.py:159 PUBLISHED_SETS`, `ccdm:258`, `enjl:163`,
`sfho/nmp.py:738 PUBLISHED_NMP`, `dd2/couplings.py:70 SU6_HYPERON`). Almost all
module-level constants are tolerances and scales, which are allowed. Five are
physics numbers with no override path — finding 17.

**Non-convergence is a return value.** `eos_point` is compliant in all eleven
models: each returns `PointResult(ok, message, point)` and each re-raises only
`NotImplementedError` — `eos/zl/api.py:117-120` states the reasoning ("an
unwired request must never be a status"). `eos/enjl/verify/run_full_check.py:769
check_non_convergence_is_returned` even pins it as an invariant. `eos_table` is
compliant everywhere via `eos/general/tabulate.py:210-219` with
`skip_errors=True` the default. **Two classes of breach** — findings 18–19.

**Never a hang.** `grep -rn "while True" eos/` → **zero matches**. Every `while`
is bounded, with one exception (`eos/general/fermi_integrals.py:519,524` — a
geometric bracket-expansion loop with no counter and no cap; finding 20).

**No global mutable state / picklable.** Zero `global` statements in code.
`@dataclass(frozen=True)` on nine of eleven parameter classes; `zl` and `sfho`
are plain (finding 21). The one module-level cache,
`eos/enjl/thermodynamics.py:404 _vacuum_cache`, is keyed on the frozen
`Parameters` object and argues the case correctly in its docstring — exactly the
"read-only caches keyed by immutable parameters" §6 allows.

Measured, by actually pickling:

```
zl        Parameters frozen=False  default() OK  pickle_roundtrip_eq=True
sfho      Parameters frozen=False  default() -> AttributeError: no attribute 'default'
dd2       no Parameters class; has: ['Parametrization']
did       Parameters frozen=True   default() OK  pickle_roundtrip_eq=True
vmit      Parameters frozen=True   default() -> AttributeError: no attribute 'default'
alphabag  Parameters frozen=True   default() OK  pickle_roundtrip_eq=True
njl       Parameters frozen=True   default() OK  pickle_roundtrip_eq=True
ccdm      Parameters frozen=True   default() OK  pickle_roundtrip_eq=True
abpr      Parameters frozen=True   default() OK  pickle_roundtrip_eq=True
enjl      Parameters frozen=True   default() OK  pickle_roundtrip_eq=True
```

Everything pickles. The naming gaps (`Parametrization`; missing `.default()`) are
ledgered at `docs/DEFERRED.md:290-340`, **except** that the ledger asserts at
line 328 "every model's parameter dataclass is `Parameters`", which
`eos/dd2/parameters.py:35 class Parametrization` contradicts (finding 22).

**Array in / array out.** All ten grid drivers are genuine warm-started sweeps
sharing `general/tabulate.sweep_lines`, and every one says so in its `eos_table`
docstring — the §6 escape hatch, used correctly. `abpr` is the exception
(finding 23).

---

## 8. §7 — Fermi and Bose integrals

`eos/general/fermi_integrals.py` (797 lines) and `eos/general/bose_integrals.py`
(469 lines) both exist and **JEL is present and primary** in each
(`fermi_integrals.py:6` "JEL is the validated one and is never removed";
`:49 aJEL = 0.433`; `bose_integrals.py:35 aBJEL = 1.040`). Alternatives sit
alongside, validated against JEL, exactly as §7 requires.

Scipy quadrature outside `general/`:

```
$ grep -rn -iE "quad\(|integrate\.quad|dblquad|simpson|trapz|trapezoid|leggauss|laguerre" \
      eos/{zl,sfho,dd2,did,vmit,alphabag,njl,ccdm,abpr,enjl,mixed,zlvmit,astro}
eos/astro/tov/solver.py:395:     M_b_MeV = m_nucleon_MeV * np.trapz(integrand, dx=dr_fm)
eos/astro/tov/rns_backend.py:78: from scipy.integrate import cumulative_trapezoid
eos/astro/tov/rns_backend.py:231: h_fine = RNS_C_CGS**2 * cumulative_trapezoid(...)
```

All three are *radial* integrals along a star, not Fermi/Bose. The
occupation-number grep (`1/(np.exp`, `logaddexp`, `expit`) returns nothing
outside `eos/general/`.

Every model either imports `solve_fermi_jel` / `kinetic_thermo` /
`invert_fermi_density` from `general/`, or writes analytic physics CLAUDE.md
explicitly allows: `alphabag`'s pQCD-corrected gas
(`thermodynamics.py:141-190`, cited to Freedman & McLerran 1977 and added *on
top of* an exact `solve_fermi_jel` call at `:222`), `abpr`'s closed-form CFL
potential, and the cutoff-regularized Dirac-sea subtractions of `njl`
(`thermodynamics.py:86-114`) and `enjl` (`:81-107`). `enjl` documents the
distinction best (`thermodynamics.py:113-116`). **One borderline case** —
finding 24.

Informational: `njl/thermodynamics.py:86-114` and `enjl/thermodynamics.py:81-107`
are the same two sea integrals written twice in different algebraic
arrangements. §1 forbids fixing that by an import between models; if a third
NJL-family model lands, `general/` is the home.

---

## 9. §8 — what each `verify/` suite actually checks

✅ present · ❌ absent · – not applicable.

| unit | Euler | free energy | Σ^R in μ,P not ε | P non-decreasing | 0≤cs²≤1 | backend parity |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| `zl` | ✅ `:101` | ✅ `:112` | – | ❌ | ✅ `:239` | – |
| `sfho` | ✅ `:124` | ❌ | – | ✅ `:234` | ✅ `:237` | ✅ `:289` (Jacobian only) |
| `dd2` | ⚠️ inside `_check_identities` `:70` | ❌ | **❌ (dd2 HAS Σ^R)** | ❌ | ✅ `:80` | ✅ `:108` |
| `did` | ✅ `:129` | ✅ `:139` | ✅ `:155` | ❌ | ✅ `:375` | – |
| `vmit` | ✅ `:104` | ✅ `:111` | – | ❌ | ✅ `:219` | – |
| `alphabag` | ✅ `:121` | ✅ `:140` | – | ❌ | ✅ `:398` | – |
| `njl` | ✅ `:290` | ✅ `:300` | – | ❌ | ⚠️ `:545` **off by default** (`:602 if include_sound:`) | – |
| `ccdm` | ✅ `:313` | ✅ `:327` | ✅ `:263` (best in repo) | ❌ | **❌ none at all** | – |
| `abpr` | ✅ `:97` | ✅ `:117` | – | ❌ | ✅ `:211` | – |
| `enjl` | ✅ `:158` | ✅ `:193` | ✅ `:286` | ✅ `:889` | ✅ `:893` | – |
| `mixed` | ✅ `:106` | ❌ | ❌ | ✅ `:191` | ✅ `:191` | ✅ `:173` |
| `astro/gmode` | – | – | – | – | – | – |
| `astro/tov` | **no `verify/` at all** | | | | | |
| `zlvmit` | **no `verify/` at all** (exempt) | | | | | |
| `general` | **no `verify/`** (7 files in `test/general/`) | | | | | |

`eos/ccdm/verify/run_full_check.py:263 check_rearrangement_placement` is the
model implementation of the §8 rearrangement invariant: it quotes CLAUDE.md,
asserts `P - (S_P - U - V + W) == Sigma_R * n_q`, **and** asserts the term is
non-trivial (`size > 1.0e-6`, `:297`) so the check cannot pass vacuously.
`eos/enjl/verify/run_full_check.py:866 check_delivered_table` is the model
implementation of the §8 delivery gate. Both are worth copying.

---

## 10. §11/§12 — documents, `verify/run_full_check.py`, tests

`verify/run_full_check.py` present: `zl`, `sfho`, `dd2`, `did`, `vmit`,
`alphabag`, `njl`, `ccdm`, `abpr`, `enjl`, `mixed`, `astro/gmode`.
Absent: `astro/tov`, `zlvmit`, `general`. No unit has a `verify/` directory
without a `run_full_check.py`.

`<model>.tex` and `<model>.md` present for all eleven models plus `astro/tov`
and `astro/gmode`. Absent only for `zlvmit` and `general`.

```
test/mixed  27   test/dd2  23   test/general 7   test/enjl 6
test/did     5   test/njl   5   test/ccdm    5   test/abpr 4
test/sfho    4   test/gmode 4   test/alphabag 3  test/vmit 3
test/zl      2   test/tov   2   test/zlvmit  0 (61 .dat golden tables, no test_*.py)
```

Plus `test/baseline/` (§12 golden ledger: `test_baseline.py`,
`generate_baseline.py`, 15 `.npz` frozen at rtol = 1e-10), `test/test_imports.py`
and `test/test_interpolator_edge_snap.py`. As expected, `test/` flattens
`astro/`: `test/tov/` and `test/gmode/`, no `test/astro/`.

---

## 11. §10 — styling

The ticket's two required commands, real output:

```
$ grep -rn "rcParams" eos/
eos/general/figure_style.py:20:Calling one after the other overwrites the other's rcParams -- pick one per
eos/general/figure_style.py:195:    """Publication rcParams: CMU Serif + Computer-Modern math (matches a LaTeX
eos/general/figure_style.py:208:      rc        : optional dict of extra rcParams to override on top, e.g.
eos/general/figure_style.py:242:    mpl.rcParams.update(base)
eos/general/figure_style.py:277:      rc       : optional dict of extra rcParams (e.g. line/axes widths),
eos/general/figure_style.py:410:    """Notebook rcParams: CMU Serif at 12-14 pt, 150 dpi, ASCII minus.
eos/general/figure_style.py:416:    plt.rcParams['font.family'] = FONTS['family']
eos/general/figure_style.py:417:    plt.rcParams['font.serif'] = FONTS['serif']
eos/general/figure_style.py:420-446:  (23 further assignments, all in this file)
eos/general/figure_style.py:593-595:  (three self-test assertions)
eos/zlvmit/plot_results.py:184:    sets rcParams. Only what is peculiar to these figures is passed in: the
eos/zlvmit/table_reader.py:703:    # (`eos.general.figure_style`); nothing here sets rcParams of its own.
```

```
$ grep -rn "STANDARD_COLORS" eos/
eos/general/figure_style.py:46:STANDARD_COLORS = {                       <- the ONE declaration
eos/general/figure_style.py:80-140:   (internal uses: TEMPERATURE_COLORS, PARTICLE_COLORS)
eos/general/figure_style.py:598:      assert len(set(map(tuple, STANDARD_COLORS.values()))) == len(STANDARD_COLORS)
eos/general/constraints/__init__.py:46:from eos.general.figure_style import STANDARD_COLORS
eos/general/constraints/__init__.py:85-143:  (19 uses of the imported dict)
eos/zlvmit/plot_results.py:74:from eos.general.figure_style import STANDARD_COLORS
eos/zlvmit/plot_results.py:76-82:  STANDARD_RED = STANDARD_COLORS['Red']  ... (aliases, not a redeclaration)
eos/zlvmit/table_reader.py:700:    from eos.general.figure_style import STANDARD_COLORS, set_paper_style
eos/zlvmit/table_reader.py:739:        ax.plot(n_B_array, P_array, color=STANDARD_COLORS['Blue'])
```

```
$ grep -rn "plt\.style\|matplotlib\.style\|mpl\.style" eos/
(nothing)
$ grep -rln "import matplotlib\|from matplotlib" eos/
eos/general/constraints/__init__.py
eos/general/constraints/build.py
eos/general/figure_style.py
eos/dd2/notebook_api.py          <- the only non-general, non-zlvmit importer
eos/zlvmit/plot_results.py
eos/zlvmit/table_reader.py
```

**Verdict.** `STANDARD_COLORS` is declared exactly once
(`figure_style.py:46`) and everywhere else imported — full Pass.
The `rcParams` acceptance criterion ("hits exactly one file") **fails on the
literal grep** (three files) but **passes in substance**: every one of the ~30
assignments is in `eos/general/figure_style.py`, and the two `zlvmit` hits are
prose saying the file does *not* set rcParams. `eos/zlvmit/table_reader.py:703`
is the comment "nothing here sets rcParams of its own". I record this as a Pass
with the literal-grep caveat noted, not as a violation — but the criterion as
written in the ticket is not met verbatim.

One live rcParams setter does remain in the repository, **outside `eos/`**:
`notebooks/ENJL_usage.py` sets `figure.dpi` and `figure.figsize`, ledgered at
`docs/DEFERRED.md:194-203` and belonging to the notebook rework.

---

## 12. Findings

Every Fail and every Ambiguous cell, numbered, with evidence and a read.
Classes: **(a)** the code is wrong and should be fixed · **(b)** CLAUDE.md
describes a target the refactor settled differently and the document should
change · **(c)** genuinely deferred, belongs in `docs/DEFERRED.md`.
*My read is input to a human decision, not the decision.*

**Index by class.**

- **(a) fix the code — 22:** 1 (delete `notebook_api.py`) · 2 (gmode → model
  internals) · 4 (`ccdm` missing from the layering gate) · **5 (dd2 `photons`
  silently ignored)** · **6 (mixed `photons` silently ignored)** ·
  **7 (`fracs` drops fixed fractions)** · 8 (`zl.thermo_from_n` takes `Y_C`) ·
  9 (dd2 parameter classmethods force a deferred solver import) · 10 (dd2
  solver → table cycle) · 11 (`mixed/backends` not deletable) · 12 (alphabag
  re-derives `quark_charges` ×5) · 13 (second `quark_charges` in mixed) ·
  16b (`leptons` smuggled through `**conditions`) · 17a (`TC_COEFF`, gmode
  weak constants) · 18 (`SnB` raises in njl/ccdm) · 19 (`eos_response` raises
  in five units) · 21 (sfho/zl parameters not frozen) · 23 (abpr docstring
  claims array arithmetic) · 24 (dd2 re-derives the T = 0 Fermi gas) ·
  25 (dd2 verify: no free-energy, no rearrangement) · 26 (mixed verify: same) ·
  27 (ccdm verify: no causality at all) · 28 (njl causality off by default) ·
  38 (matplotlib inside `eos/dd2`).
- **(b) change the document — 11:** 3 (`verify/` carve-out is real and unwritten) ·
  14 (a mixed-species model may keep a local QN table?) · 15 (may `mode` carry a
  default?) · 16a (`Y_p` as a freeze target in a signature) · 29 (who owes the
  P-monotonicity delivery gate) · 31 (does `general/` earn a `verify/`) ·
  34 (`zlvmit`'s exemption: API only, or documents and tests too) ·
  36 (`thermodynamics_<sector>.py` rule scoped to models?) ·
  **39 (CLAUDE.md's model lists are stale; `ccdm` appears nowhere)** ·
  41 (the §10 `rcParams` grep cannot tell an assignment from a sentence) ·
  plus the two §3 minors (the undeclared `cfl` mode; `thermal_neutrinos` +
  trapped raising in two models and succeeding in three).
- **(c) ledger it in `docs/DEFERRED.md` — 12:** 2 (short-term entry while the
  gmode contract is decided) · 10b (the needless downward deferrals) ·
  17b (`_CHIRAL_SPLIT`, RNS surface constants, legacy `B4`) · 20 (the one
  unbounded loop) · 22 (the ledger's own `Parameters` sentence is wrong for
  dd2) · 30 (`astro/tov` has no `verify/`) · **32 (abpr refuses all four modes,
  unrecorded)** · 33 (mixed's `Y_Lmu` refusal, unrecorded) · 35 (`VMIT*` class
  names) · 37 (`output/public/` does not exist) · 40 (`zl` has no `nmp.py`) ·
  and the `alphabag` `cfl` thermal-neutrino drop, already filed.

### §1 layering

**1. `eos/dd2/notebook_api.py` imports `astro/` and `sfho` — the last live §1 breach in importable code.** → **(a)**
`eos/dd2/notebook_api.py:34-35` (`eos.astro.tov.crust`, `eos.astro.tov.solver`)
and `:484` (`eos.sfho.table`). Both are the *only* remaining edges in their
buckets. Ledgered twice (`docs/DEFERRED.md:97`, `:144`, `:299`) with the same
disposition: "it dies with the file". `test/test_imports.py:100-104` carries an
explicit `_EXEMPT_FILES` entry for it, calling it "a pre-existing violation in
importable code". §11 line 457 forbids the file by name. **Delete the file; both
edges go with it.** This is not a judgement call — every document in the repo
already agrees, it just has not been done.

**2. `eos/astro/gmode` imports model internals.** → **(a)** with a **(b)** tail.
```
eos/astro/gmode/rates.py:85            from eos.dd2.solver import solve_composition
eos/astro/gmode/sound_speeds.py:94     from eos.mixed.responses import sound_speed_eq, sound_speed_frozen
eos/astro/gmode/sound_speeds.py:149    from eos.dd2.solver import solve_composition   (function-local)
eos/astro/gmode/verify/run_full_check.py:39-41   eos.dd2.Parametrization, eos.dd2.responses, eos.dd2.solver
```
§1 line 33: "`astro/` … consumes tables and arrays produced by models and
engines; **it never imports model internals**." `solve_composition` is a solver
internal, and `rates.py:85` is a top-level import, so `import eos.astro.gmode`
pulls DD2 in. The gmode composition-g-mode calculation genuinely needs
d(composition)/dn_B, which no table carries — so the fix is either a declared
contract in `general/` (the way `EOSTable_for_TOV` works for TOV) or an
explicit statement in §1 that gmode couples to a model. **Not recorded anywhere
in `docs/DEFERRED.md`** — `grep -n -i gmode docs/DEFERRED.md` returns only the
cross-cutting "expensive, not hung" entry at line 101. Whichever way it is
resolved, the gap needs a ledger entry today. **(c)** in the short term.

**3. Three `verify/` suites reach sideways into another model or into `mixed`.** → **(b)**
```
eos/abpr/verify/run_full_check.py:47-48   eos.alphabag.parameters, eos.alphabag.thermodynamics
eos/enjl/verify/run_full_check.py:817     eos.mixed.construction.enjl_coexistences
eos/enjl/verify/run_full_check.py:834-835 eos.mixed.adapters, eos.mixed.boundaries
```
The refactor settled this deliberately: `test/test_imports.py:88-97` documents a
`verify/` carve-out from the model-to-model half of §1 ("it checks END-TO-END
invariants … doing that requires reaching sideways by construction. It is also
not on the path a sampler imports, which is what the layering rule protects"),
and `docs/DEFERRED.md:155-157` confirms it, naming enjl. **CLAUDE.md §1 does not
contain the carve-out.** It should — the reasoning is sound and the astro half
of the rule was tightened in the same session precisely because it had been
ambiguous. Note `abpr`'s edge is not named in the ledger even though enjl's is;
if the carve-out is written into §1, both are covered.

**4. `test/test_imports.py` does not gate `eos.ccdm`.** → **(a)**
`test/test_imports.py:107-109`:
```python
MODEL_PACKAGES = ("dd2", "sfho", "zl", "did", "vmit", "alphabag", "abpr",
                  "enjl", "njl")
```
`ccdm` is missing, so neither `test_a_model_imports_only_general` nor
`test_no_model_imports_astro` runs against it. Its import graph is clean today —
this is a gap in the gate, not a live violation. One-word fix, and it is
downstream of finding 25 (CLAUDE.md's own model list omits `ccdm` too).

### §4 species flags

**5. `dd2.SpeciesFlags.photons` is accepted and never read — a silently ignored sector.** → **(a)**
`eos/dd2/species.py:26`:
```python
    photons: bool = True                # radiation (matters only at T>0)
```
`grep -rn "photons" eos/dd2 eos/mixed` returns 44 hits; **not one of them reads
`flags.photons` / `species.photons`.** The photon gas is switched by a separate
keyword that is never fed from the flag:
```
eos/dd2/solver.py:709   def solve_octet(par, n_B, flags, ..., include_photons=True, ...)
eos/dd2/solver.py:757       if include_photons and T > 0.0:
eos/dd2/api.py:102          point = solve_octet(par, n_B, species, T=T, x0=x0,
                                                analytic_jac=analytic_jac, **kwargs)
eos/dd2/table.py:106        return solve_octet(par, n_B, flags, T=T, x0=x0, include_photons=True, ...)
eos/dd2/table.py:243        return solve_octet(spec.parametrization, float(n), flags,
                                               T=float(tv), x0=x0, **mode_kw)
```
`_mode_kwargs` never produces `include_photons`, so the default `True` wins.
`eos.dd2.eos_point(par, ..., species=SpeciesFlags(photons=False), T=30)` and
`eos_table` with the same flags **both return a state with a photon gas in eps,
P and s**. §4 line 143: "No sector is enabled or disabled implicitly … Setting a
flag a model does not implement RAISES; a `NotImplementedError` is never turned
into a silent no-op." Not recorded in `docs/DEFERRED.md` (the dd2 species bullet
at `:786-792` covers only the *naming* of `thermal_mesons`/`thermal_neutrinos`).
The fix is one line at each call site. **This is the single loudest finding in
the audit** — it is the exact failure mode §4 exists to prevent, in the
flagship hadronic model.

**6. `eos/mixed` silently ignores `photons` for the same reason.** → **(a)**
`eos/mixed/thermodynamics.py:84`:
```python
    ph = photon_thermo(T) if T > 0.0 else None
```
Unconditional on `flags`. `mixed` reuses dd2's dataclass
(`eos/mixed/api.py:48 from eos.dd2.species import SpeciesFlags`) and reads
`flags.muons`, `flags.neutrinos`, `flags.sigma_star`, `flags.hyperons`,
`flags.deltas`, `flags.phi_field` and both meson-gas flags — never
`flags.photons`. Same fix, same class. Not in the ledger.

**Also under §4, not a Fail:** `alphabag`'s `thermal_neutrinos` is threaded
into four of five solver arms and **dropped on the `cfl` arm**
(`eos/alphabag/table.py:80-84`). It is a silent drop rather than a raise, but it
is deliberate, commented at the call site, and ledgered (`docs/DEFERRED.md`
alphabag section: "The paired phase carries NO thermal neutrino gas … preserved
deliberately rather than fixed here, because closing it changes the published
CFL tables"). **(c)**, already correctly filed — marked Ambiguous only because
a reader of §4 alone would expect a raise.

### §5 API and layout

**7. The `progress` dict's `fracs` drops the *fixed* fractions in `dd2` and `sfho`.** → **(a)**
§5 line 184: "`fracs` carries every fraction the line was solved at, **swept or
fixed**." Both models compute the full set and then discard half of it:
```
eos/dd2/table.py:234-236    fracs = dict(spec.fixed)
                            fracs.update(zip(frac_keys, (float(c) for c in combo)))
eos/dd2/table.py:250        combos.append((float(tv), dict(zip(frac_keys, map(float, combo)))))
eos/dd2/table.py:273                          temp=float(tv), fracs=combos[-1][1],
```
identical at `eos/sfho/table.py:296` and `:333`. So
`eos_table(par, "fixed_YC_YS", axes={'nB':…, 'T':…, 'Y_C':…}, fixed={'Y_S': 0.0})`
reports `fracs={'Y_C': …}` and loses `Y_S`. The shared driver gets it right —
`eos/general/tabulate.py:58 conditions = dict(fixed or {})` seeds each line from
`fixed`, so `split_conditions` returns both. **The one outright key-content
violation of the progress contract**, and it defeats §5's stated purpose ("the
SAME dictionary in every model, so one printer serves them all"). Not ledgered.
Two-line fix: pass `fracs`, not `combos[-1][1]`.

**8. `eos/zl/thermodynamics.py:374 thermo_from_n(n_B, Y_C, T, params)` takes a mode's held fraction.** → **(a)**, arguably **(b)**
```
374: def thermo_from_n(n_B: float, Y_C: float, T: float,
386:     n_p = Y_C * n_B
387:     n_n = (1 - Y_C) * n_B
```
The only non-docstring hit in the whole §5 purity grep. §5 line 262 says
`thermodynamics.py` "takes chemical potentials, fields, T, the parameters and
the species flags"; this takes the `fixed_YC` closure. It is exported publicly
(`eos/zl/__init__.py:24,45`) and consumed by the composite engine at
`eos/mixed/adapters.py:913`, so the engine is reaching through the
phase-adapter surface into a mode-aware function.

*The counter-argument, and why this is worth a human decision:* `(n_B, Y_C)` is
just a re-parameterization of `(n_n, n_p)`, and §5's own vocabulary names
`thermo_from_n(...)` "the block at given densities". Changing the signature to
`thermo_from_n(n_n, n_p, T, params)` would satisfy both the letter and the
spirit at the cost of one adapter line. If instead the repository decides that
naming a composition by its charge fraction is not mode-awareness, then §5's
grep test is the thing that needs qualifying — **(b)**. Either way it should not
stay as it is, because today the grep test the document publishes gives a false
positive that a reader cannot distinguish from a real one.

**9. `eos/dd2/parameters.py` carries the exact anti-pattern §5 names, comment included.** → **(a)**
```
eos/dd2/parameters.py:220   def from_hyperon_potentials(...)          # classmethod
eos/dd2/parameters.py:235       from eos.dd2.solver import solve_snm  # local import breaks the cycle
eos/dd2/parameters.py:254   def from_delta_potential(...)             # classmethod
eos/dd2/parameters.py:268       from eos.dd2.solver import solve_snm  # local import breaks the cycle
```
§5 lines 300-303: "A constructor that inverts NMPs is therefore a free function
in `nmp.py`, not a classmethod on the parameter dataclass — putting it there
forces a deferred import, **which is the cycle announcing itself**." The code has
the classmethod, the deferred import, and a comment that says the cycle out
loud. `eos/dd2/nmp.py:266 invert_nmp` and `:437 from_nmp` are already correctly
placed free functions, so the destination exists. Not ledgered.

**10. `eos/dd2/solver.py:880` is a genuine upward import (solver → table).** → **(a)**
```
880:    from eos.dd2.table import _mode_kwargs
```
`eos/dd2/table.py:21` imports `solve_octet` from `solver.py`, so this is a real
cycle deferred to hide it. `_mode_kwargs` (mode name → argument set) is solver
vocabulary; `did` puts the same thing in `solver.py` and imports it upward
(`eos/did/api.py:24`), which is the right shape. Two smaller siblings in the
same package: `eos/dd2/table.py:335 from eos.dd2 import Parametrization` (a
submodule reaching back through its own package `__init__`) and
`eos/dd2/nmp.py:406,420 from eos.dd2.nmp import esym` (a module importing
itself, no-op). Not ledgered.

Lower-severity deferrals of *downward* imports, where no cycle exists and a
top-level import would work: `eos/sfho/nmp.py:66,67,138`,
`eos/did/nmp.py:162`, `eos/sfho/thermodynamics.py:583`,
`eos/njl/thermodynamics.py:479`, `eos/ccdm/thermodynamics.py:557`, and
`api.py → responses.py` in `sfho:193`, `did:145`, `njl:174`, `ccdm:197`,
`dd2:177`. Style drift, not cycles. **(c)** at most.

**11. `eos/mixed/backends/` is not deletable.** → **(a)**
```
eos/mixed/verify/run_full_check.py:44:from eos.mixed.backends.jacobian import mixed_jacobian
```
The only unconditional module-scope backend import in the repository. §5 line
283: "**`backends/` is deletable.** Remove it and the model still gives the same
numbers, only slower." Delete `eos/mixed/backends/` and the mixed verify suite
fails at import. `dd2` and `sfho` get this right by deferring the same import
inside functions (`eos/dd2/verify/run_full_check.py:97`,
`eos/sfho/verify/run_full_check.py:304,395`). One-line fix. Not ledgered.

Related but ledgered: `eos/sfho/api.py:171-174` documents that `backends/` is
"the only route" to the susceptibilities, and `docs/DEFERRED.md:753-757` says
the same for dd2 — "`eos_response(frozen='equilibrium')` raises without
`backends/` rather than degrading to a slower path, which is the one place §5's
deletability is a feature gap rather than a speed cost." Correctly filed **(c)**.

### §2 basis maps

**12. `eos/alphabag/solver.py` re-derives `quark_charges` five times, in literal fractions.** → **(a)**
```
eos/alphabag/solver.py:441:  n_C = (2.0/3.0)*n_u - (1.0/3.0)*n_d - (1.0/3.0)*n_s
eos/alphabag/solver.py:521:  (identical)
eos/alphabag/solver.py:613:  (identical)
eos/alphabag/solver.py:700:  (identical)
eos/alphabag/solver.py:745:  (identical)
```
plus `n_B_calc = (n_u + n_d + n_s) / 3.0` at `:440,:520,:612,:699`. §2 line 106:
"Basis changes are declared once … No model carries its own copy of these
algebraic maps." The model contradicts itself — its own
`eos/alphabag/thermodynamics.py:36` already imports `quark_charges` from
`eos.general.basis`, and `eos/alphabag/verify/run_full_check.py:21` claims "no
local copy of the map", which holds for the *reported* charges but not for the
residual rows. No sign risk (the numbers are correct), but it is five places a
convention change would have to be chased. Not ledgered.

**13. `eos/mixed/charges.py:153-169` defines a second `quark_charges`.** → **(a)**, low
```
153: QUARK_QN = {q.name: (q.baryon_no, q.charge, q.strangeness) for q in (Up, Down, Strange)}
157: def quark_charges(n_u, n_d, n_s):
```
alongside the one `eos/mixed/adapters.py:50` already imports from
`eos.general.basis`. Built from the shared `Particle` objects so it cannot drift
in sign, but the engine now has two functions of that name in scope. Not ledgered.

**14. `eos/enjl` carries its own quantum-number tables and density sums.** → **(b)** or **(c)**
```
eos/enjl/species.py:38   BARYON_NUMBER = {...}
eos/enjl/species.py:51   CHARGE = {...}
eos/enjl/species.py:57   STRANGENESS = {...}
eos/enjl/thermodynamics.py:473-476   n_B/n_C/n_S written out longhand
```
All signs correct, and `eos/enjl/verify/run_full_check.py:67` cross-checks
against `eos.general.basis`, which is the mitigation. Partially justified —
ENJL is the only model with baryons *and* quarks in one species list — but
`general/basis.charges_from_densities` handles exactly that (it looks each name
up in `general/particles.py` and skips leptons; all eight ENJL species are
registered there). Either import it, or record in §7/§2 that a model with a
mixed species list may keep a local table. Same class, milder: `sfho` writes the
species-potential map inline (`eos/sfho/thermodynamics.py:171,481-482`,
`eos/sfho/backends/jacobian.py:197`) where `dd2` and `did` import
`eos.general.basis.species_potential` (`eos/dd2/solver.py:53`,
`eos/did/solver.py:58`). Not ledgered — `grep -n "species_potential\|general.basis"
docs/DEFERRED.md` returns nothing.

### §5 signatures

**15. `mode` acquired a default in four models.** → **(b)**
`njl/api.py:73,121,154` and `ccdm/api.py:81,137,171` and `enjl/api.py:78,135,232`
default to `"beta_eq_neutrinoless"`; `abpr/api.py:73,146,203` to `"cfl"`. §5
line 156 shows `mode` as a required positional, and the same reasoning that makes
`par` non-optional applies — a default mode is a physics choice made on the
caller's behalf. `abpr` is defensible (one mode exists). Either §5 states that
`mode` may carry a default, or the four models drop it.

**16. `Y_p` in a signature, and `leptons` smuggled through `**conditions`.** → **(a)** for the second, **(b)** for the first
```
eos/dd2/api.py:142:  def eos_response(par, mode, species, frozen="equilibrium", n_B=None, T=0.0, Y_p=None, **conditions)
eos/sfho/api.py:56:      leptons = conditions.pop("leptons", None)
eos/dd2/api.py:53:       leptons = conditions.pop("leptons", None)
eos/did/api.py:57:       (whitelists "leptons" as an allowed extra)
```
§5 line 167 fixes the condition names at `n_B, T, Y_C, Y_S, Y_Le, Y_Lmu`. `Y_p`
is a species fraction and the only such leak in a signature — it is what dd2's
`composition` freeze holds, so §5 may simply need to say that a freeze target
may appear as a named argument. `leptons` is different: §3 defines it as an
orthogonal *flag*, and six models make it an explicit named argument
(`zl/api.py:66`, `vmit:66`, `alphabag:71`, `njl:122`, `enjl:79`, `mixed:98`).
`sfho`/`dd2`/`did` route it through the conditions bag, where it then mutates
`mode` into a name §3 does not define (`fixed_YC_neutral`,
`eos/dd2/table.py:54`). That is drift, and the majority shape is the right one.

### §6

**17. Five module-level physics constants with no override path.** → **(a)** for two, **(c)** for the rest
```
eos/alphabag/thermodynamics.py:50   TC_COEFF = 0.57 * 2**(1.0/3.0)      -> :410 T_critical(Delta0)
eos/astro/gmode/rates.py:90-97      G2_FERMI = 1.1e-22 ; G_A = 1.26 ; F_PI_NN = 1.0 ; M_PI = 139.57039
eos/mixed/adapters.py:314,319       _CHIRAL_SPLIT = 50.0 ; _DECONFINED_BARYON_FRACTION = 1.0e-4
eos/zlvmit/hybrid_table_generator.py:46   B4 = 165.0     # bag constant, legacy
eos/astro/tov/rns_backend.py:94-95  RNS_RHO_SURFACE = 7.8 ; RNS_P_SURFACE = 1.01e8
```
§6 line 340: "**MODEL PARAMETERS ARE ARGUMENTS** … a parameter that can only be
changed by editing a source file makes inference impossible." `TC_COEFF` is the
CFL critical-temperature coefficient and is not a field of
`eos/alphabag/parameters.py:37-61`; `T_critical` takes no override — an
inference run over CFL pairing cannot vary it. **(a)**. The gmode weak-coupling
constants are the same shape, and `M_PI` additionally duplicates a mass that
belongs in `eos/general/particles.py` per §7 — **(a)**, and it rides along with
finding 2. `_CHIRAL_SPLIT` / `_DECONFINED_BARYON_FRACTION` decide which *phase
label* a point gets — physics-bearing thresholds — **(c)**. The RNS surface
conditions mirror the C source and the legacy `B4` is in an exempt package —
**(c)**.

Not violations, listed so the next auditor does not re-flag them: `dd2/thermodynamics.py:54
_X_OMEGA_LAMBDA` (documented fallback, overridden by `par.hyperon_couplings` at
`:191-192` and a default argument at `:331`), `sfho/nmp.py:44 N_SAT`,
`did/couplings.py:45-58`, `mixed/scan.py:93,98`, and every `RESIDUAL_TOL` /
`FIELD_SCALE` / `MASS_DAMPING` / `GAP_TOL` — those are numerics.

**18. `eos_point(..., SnB=...)` raises in `njl` and `ccdm`.** → **(a)**
`eos/general/tabulate.py:102-103` raises when an entropy target is unreachable:
```python
        raise RuntimeError(f"entropy per baryon {target} is unreachable below T = {T_cap} ...")
```
`zl/api.py:113`, `vmit:113`, `alphabag:119`, `mixed:160` and `dd2:97` all call it
**inside** the `try`. `eos/njl/api.py:109` and `eos/ccdm/api.py:122` call it bare:
```python
        T = temperature_at_entropy(entropy_at, SnB)
```
So an unreachable isentrope escapes a public entry point as a `RuntimeError`.
§6 line 348: "**NON-CONVERGENCE IS A RETURN VALUE** at every public boundary."
Two-line fix (move the call inside the existing `try`). Not ledgered.

**19. `eos_response` raises on non-convergence in five units.** → **(a)**
```
eos/zl/api.py:199-201        raise RuntimeError(f"eos_response could not solve its stencil point ...")
eos/vmit/api.py:199          (same)
eos/alphabag/api.py:207-209  (same)
eos/mixed/api.py:345-346     raise RuntimeError(f"eos_response could not solve its central point ...")
eos/mixed/api.py:374-376     (again, for the stencil points)
eos/abpr/api.py:233-234      mu, converged = mu_from_nB(n_B, par)
                             if not converged:
                                 raise RuntimeError(f"eos_response could not invert n_B = {n_B}")
```
`eos_response` is one of the three §5 entry points, so §6's rule covers it.
The `abpr` case is the starkest: the status is explicitly in hand and is
converted into an exception. `sfho`, `dd2`, `did`, `njl`, `ccdm` return dicts and
do not have this construct, so the compliant shape already exists in the
repository. Not ledgered — and note `eos/abpr` DEFERRED (`:1012-1017`) asserts
the opposite ("what it cannot report is a failure mode that does not exist"),
which the code at `:233` contradicts.

**20. One unbounded loop.** → **(c)**
```
eos/general/fermi_integrals.py:519   while n_hi < n_target:   (mu_hi *= 1.5)
eos/general/fermi_integrals.py:524   while n_lo > n_target:
```
No iteration counter, no cap. Every other `while` in the repository is bounded
(`grep -rn "while True" eos/` → zero matches; `general/tabulate.py:98` capped at
`T_cap=400.0`; `general/thermodynamics_leptons.py:343-345` raises past `mu_max`;
`general/solve.py:72-83` "Three attempts at most: a parameter scan must always
get an answer back"). Geometric growth makes a hang practically unreachable, but
§6 line 353 says "every solver has a bounded iteration count", so this is a
ledger line and a two-line fix.

**21. `zl` and `sfho` parameter dataclasses are not frozen.** → **(a)** for `sfho`, **(c)** for `zl`
```
eos/zl/parameters.py:27      @dataclass          (plain; all fields str/float)
eos/sfho/parameters.py:68    @dataclass          (plain)
eos/sfho/parameters.py:152   a_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(7))
eos/sfho/parameters.py:373   p.couplings_map['lambda'] = {...}       # mutated post-construction
eos/sfho/parameters.py:425, :480   (same)
```
Both pickle (measured above), so multiprocessing works. But `sfho` carries
mutable arrays and a `couplings_map` dict mutated after construction — two SFHo
parameter objects sharing that reference would interfere, which is exactly what
§6 line 356 ("Two models with different parameters coexist in one process
without interfering") rules out. `zl`'s is benign in practice but leaves it
unhashable, so it cannot key the read-only caches §6 permits. `enjl` shows the
target shape: frozen dataclass, `_vacuum_cache` keyed on it
(`eos/enjl/thermodynamics.py:404-430`).

**22. `docs/DEFERRED.md:328` asserts something the code contradicts.** → **(c)**
The ledger says "so every model's parameter dataclass is `Parameters`".
`eos/dd2/parameters.py:35` is `class Parametrization`, and `eos/dd2/parameters.py`
contains no `Parameters` at all. The dd2 ledger entry (`:299-305`) lists only
"delete notebook_api.py -- the last one outstanding", so the rename is neither
done nor recorded. `sfho` and `vmit` additionally still lack `Parameters.default()`
(measured above) — that half *is* recorded at `:317-323`. Fix the ledger
sentence, add the dd2 rename to the dd2 entry.

**23. `abpr`'s `eos_table` docstring claims array arithmetic it does not do.** → **(a)**, low
```
eos/abpr/api.py:154-158   "... the density inverse is a closed form ..., so no point
                           needs its neighbour and the grid is evaluated by array arithmetic."
eos/abpr/api.py:185       points = [solve_cfl(float(n), par, T=T) for n in nB]
```
A bare Python loop, one scalar at a time. The physics justification for having no
warm start is sound; the array claim is not. ABPR is the one model where genuine
array-in/array-out is achievable (§6 line 361) and is not done. Either vectorize
or correct the docstring.

### §7

**24. `eos/dd2/thermodynamics.py:66-94` re-derives the T = 0 Fermi gas.** → **(a)**, or **(b)** on the shared module
```
eos/dd2/thermodynamics.py:66  def number_density_t0(kF, g): return g * kF ** 3 / (6.0 * _PI2)
eos/dd2/thermodynamics.py:71  def scalar_density_t0(kF, ms, g): ... (kF*EF - ms**2*np.log((kF+EF)/ms))
eos/dd2/thermodynamics.py:79  def eps_kin_t0(kF, ms, g)
eos/dd2/thermodynamics.py:88  def P_kin_t0(kF, ms, g)
```
§7 line 380: "All Fermi and Bose integrals, **at T = 0 and finite T**, come from
`eos/general/`. No model implements its own." These four are the exact T = 0
ideal Fermi gas, duplicating `eos/general/fermi_integrals._compute_exact_T0`
(`:220-260`). The formulas are correct and no numeric discrepancy was found —
the reason they exist is that the shared module keeps its T = 0 closed forms
**private** and exports no public T = 0 entry point. **The right fix is
therefore in `general/`, not in `dd2`:** promote the T = 0 forms to a public
name and have dd2 import them. Not ledgered.

Not violations, confirmed and listed so they are not re-flagged: `alphabag`'s
pQCD-corrected gas (§7 line 387 names this case explicitly), `abpr`'s CFL
potential, and the cutoff-regularized sea integrals of `njl` and `enjl`.

### §8

**25. `dd2`'s verify suite checks neither the free energy nor the rearrangement placement — and dd2 carries Σ^R.** → **(a)**
`eos/dd2/verify/run_full_check.py` registers five checks (`:134-156`):
golden points, thermo identities, responses, coeff analytic~FD, backend parity,
CompOSE. Euler appears only inside `_check_identities` (`:70-78`, via
`p.euler_residual()`); there is no `f = eps - T s` check and **no rearrangement
check**, although `eos/dd2/thermodynamics.py:492 Sigma_R=Sig_R` and
`eos/dd2/solver.py:174,398,666` show DD2 has the term. §8 line 400 makes
"Rearrangement: Σ^R enters mu and P, never eps" an invariant, and it is exactly
the invariant that catches a wrong density-dependent RMF. `eos/did` and
`eos/ccdm` both implement it — `eos/ccdm/verify/run_full_check.py:263-297` is the
model version, asserting both the identity and that the term is non-trivial.
**Copy it into dd2.** Not ledgered.

**26. `eos/mixed`'s verify suite checks neither the free energy nor the rearrangement.** → **(a)**, lower priority
`eos/mixed/verify/run_full_check.py` has Euler/HVH (`:106`), causality +
monotonicity (`:187-194`) and backend parity (`:173`), but no free-energy
identity. It couples DD2, which carries Σ^R. Same fix as 25, downstream of it.

**27. `eos/ccdm`'s verify suite has no causality or monotonicity check at all.** → **(a)**
`grep -n "cs2\|sound\|monoton" eos/ccdm/verify/run_full_check.py` → nothing.
Every other model checks `0 ≤ cs² ≤ 1` somewhere. CCDM is a colour-superconducting
model where a wrong gap contribution shows up first in the sound speed. Ironic
next to its best-in-repo rearrangement check.

**28. `eos/njl`'s causality check is disabled by default.** → **(a)**, low
```
eos/njl/verify/run_full_check.py:545   def check_sound_speed(...)
eos/njl/verify/run_full_check.py:602   if include_sound:
eos/njl/verify/run_full_check.py:612   (gated on the --sound CLI flag)
```
`run_all()` with no arguments runs no causality check. §12 line 495 says new
physics gets a `verify/` entry "where it is a physics invariant"; an invariant
that does not run by default is not one. If the cost is the reason, say so in
the ledger and mark it `slow` the way `pyproject.toml`'s marker convention
already allows.

**29. `P` non-decreasing in `n_B` is checked in only 3 of 12 verify suites.** → **(b)**
Present: `sfho:234`, `enjl:889`, `mixed:191`. Absent: everywhere else.
§8 line 402 scopes the check precisely — "Any EoS table **DELIVERED to a
structure solver**" — and eight of the twelve models never build one, so their
absence is correct by the letter. But the document does not say which suites
owe the check, and `eos/dd2` and `eos/did` **do** build core tables
(`eos/dd2/table.py:316 build_core_table`, `eos/did/table.py:203`) and do not
check them. `eos/enjl/verify/run_full_check.py:866 check_delivered_table` is the
model implementation. Either §8 names the delivery gate as belonging to whoever
builds a table, or dd2/did adopt enjl's check. Related, ledgered:
`docs/DEFERRED.md` astro/tov entry — "the fast backend returns a silently wrong
tidal deformability when the table it is handed is not monotone … CLAUDE.md §6
says non-convergence is a return value: meeting a non-monotone table is exactly
that case and must come back as a status".

**30. `eos/astro/tov` has no `verify/` at all.** → **(c)**
The unit carrying the TOV integrator, tidal deformability, crust handling and
the RNS rotating backend has `tov.tex`, `tov.md` and two files in `test/tov/`,
but no invariant suite. §12 line 511 pins "the DD2 published NMP/**TOV** values"
as golden. The two-backend disagreement (~2 % in Λ) and the non-monotone-table
fragility are both ledgered under astro/tov but there is no suite to run them
from. `test/` is gitignored, so a fresh clone has no way to check TOV at all.

**31. `eos/general` has no `verify/`.** → **(b)**
Seven files in `test/general/` cover it, and §5's `verify/` list is written for
models. But `general/` is the single home of the Fermi/Bose integrals (§7), the
basis maps (§2) and the meson gas — the things every model's correctness rests
on — and the JEL-vs-fallback tolerance question is already ledgered
(`docs/DEFERRED.md:48-62`, "no parity test currently pins the two together").
Worth deciding whether `general/` earns one.

### §3

**32. `eos/abpr` refuses all four §3 modes and the ledger has no entry for it.** → **(c)**
```
eos/abpr/solver.py:47-49   MODE_FRACTIONS = {"cfl": ()}
eos/abpr/solver.py:54-73   MODE_REFUSALS  = {one physics reason per mode}
eos/abpr/solver.py:84      raise NotImplementedError(f"eos.abpr does not support mode {mode!r}: {...}")
```
§3 line 137: "A mode a model cannot support — physically meaningless … or not
yet implemented — **raises** with a message saying which; **the gap is recorded
in `docs/DEFERRED.md`**." The raises and the messages are excellent — the
`fixed_YC_YS` refusal explains that "the locked phase has Y_C = 0 and Y_S = +1
identically … so in particular the symmetric-matter slice this mode exists for
is not a state of deconfined locked matter". The `abpr` ledger section
(`:1011-1045`) records the T = 0 limit, the one-line table, the `eos_response`
limits and the parameter set, but never that the model implements none of the
four. Pure ledger work; nothing to fix in code.

**33. `eos/mixed`'s `Y_Lmu` refusal is the one of ten that is unrecorded.** → **(c)**
```
eos/mixed/api.py:81-84   raise NotImplementedError("the mixed engine tracks one trapped lepton
                                                   family; beta_eq_neutrino_trapped takes (n_B, Y_Le, T) only")
```
`zl`, `vmit`, `alphabag`, `sfho`, `dd2`, `did`, `njl`, `ccdm`, `enjl` each have a
ledger bullet for the identical gap; `mixed` does not. One line.

Two smaller §3 items, both **(b)**: the fifth mode name `cfl`
(`eos/alphabag/solver.py:61`, `eos/abpr/solver.py:48`) is declared nowhere in §3
or the ledger; and `thermal_neutrinos` combined with the trapped mode **raises**
in `sfho` (`:576`) and `did` (`:213`) but **succeeds** in `njl` (`:275`),
`ccdm` (`:307`) and `enjl` (`:224-236`) — the same call, five models, two
answers, and the uniform API says nothing about which is right.

### §11 / §12 / §10 / documents

**34. `eos/zlvmit` has no `.tex`, no `.md`, no `verify/`, and zero tests.** → **(b)**
`test/zlvmit/` exists but holds 61 `.dat` golden tables and no `test_*.py`.
§1 line 44 exempts `zlvmit` from "the uniform API"; it does not exempt it from
§11's document requirement or §12's test requirement, and `test/baseline/` does
carry a `zlvmit.npz`. Either §1's exemption is widened explicitly to cover
documents and tests, or the gap is ledgered. The map already places `zlvmit` out
of scope for this effort, so this is a document question, not a work item.

**35. `eos/vmit`'s public result type repeats its package, and it is the only model that does.** → **(c)**
```
eos/vmit/solver.py:54          class VMITEOSResult:
eos/vmit/thermodynamics.py:60  class VMITThermo:
eos/vmit/compute_tables.py:41  class VMITTableSettings:
eos/vmit/api.py:26             from eos.vmit.solver import VMITEOSResult
```
§13 rule 1: "A name never repeats its package." Every sibling uses the bare name
(`EoSPoint` in `zl`, `dd2`, `njl`, `ccdm`, `did`). Because `api.py:26` re-exports
it, `eos_point` returns a differently-named type here than in the other ten
models — so this is an API-uniformity break, not only cosmetics. Milder
instances: `NJLState` (`eos/njl/thermodynamics.py:252`), `CCDMState`
(`eos/ccdm/thermodynamics.py:263`). Ledgered in the general "module names are
standardised, and most models have not been renamed" entry
(`docs/DEFERRED.md:290-340`), which names `vmit` as DONE for the module renames
but does not list these three class names. Also `eos/vmit/compute_tables.py` is
a second table driver beside `eos/vmit/table.py` — not in the §5 template.

**36. `eos/general/thermodynamics_leptons.py` is the single suffixed file §5 calls wrong.** → **(b)**
```
$ find eos -name "thermodynamics_*.py"
eos/general/thermodynamics_leptons.py
```
§5 lines 254-257 forbid "a package holding exactly one suffixed file". `general/`
is not a model, so the rule applies only by analogy, and the suffix here does not
restate the package name (`eos.general.thermodynamics_leptons` reads fine). But
the repository has zero other `thermodynamics_*` files, so either §5 scopes the
rule to models explicitly, or the file becomes `leptons.py`.

**37. `output/public/` does not exist.** → **(c)**
`.gitignore:35-39` reserves it correctly:
```
# output/ is the one home for these; output/public/ is the curated subset
output/
!output/public/
!output/public/**
```
`ls output/` shows nine entries, none of them `public/`. §11 line 458 describes
it as an existing tracked folder. The map already says the curation is not
decidable before the notebooks produce tables, so this is a ledger line now and a
`mkdir` later.

**38. `eos/dd2/notebook_api.py` imports matplotlib inside `eos/`.** → **(a)**
`eos/dd2/notebook_api.py:565`. It does not set `rcParams`, so §10's substantive
rule holds, but §11 line 457 forbids the module by name and §10 line 420 makes
`figure_style.py` "the ONLY module in this repo … that sets matplotlib styling,
colours or figure geometry". Rides along with finding 1: delete the file.

**39. `CLAUDE.md`'s own model enumerations are stale, and `ccdm` is absent entirely.** → **(b)**
```
$ grep -n "ccdm" CLAUDE.md
(no output)

CLAUDE.md:22    - A **model** (`dd2`, `sfho`, `zl`, `vmit`, `alphabag`, `abpr`, `enjl`)
CLAUDE.md:449         dd2/ sfho/ zl/ did/ vmit/ alphabag/ abpr/ enjl/  one subpackage per model
CLAUDE.md:231   ... Shipped adapters: DD2, SFHo, ZL, DID (hadronic), vMIT,
CLAUDE.md:317   **Nuclear-matter parameters.** Models with a nuclear sector (`dd2`, `sfho`)
```
§1 omits `did`, `njl`, `ccdm`; §11 omits `njl`, `ccdm`; §5's adapter list omits
the shipped `njl_phase` (`eos/mixed/adapters.py:1051`) and `ccdm_phase` (`:1189`);
§5's nuclear-sector list omits `did`, which has `eos/did/nmp.py`. Both `njl` and
`ccdm` are complete models with `verify/` suites, `.tex`/`.md`, five test files
each and shipped mixed-phase adapters. This is the document lagging the code, and
it is upstream of finding 4 (`test_imports.py` inherits the same stale list).
**The cheapest and highest-value document fix in the audit.**

**40. `eos/zl` has a nuclear sector but no `nmp.py`.** → **(c)**, low
§5 line 317 names only `dd2` and `sfho` as owing the NMP map, and `did`
volunteered one. ZL is a nucleonic model whose forward NMP map is not exposed —
a consistency gap rather than a rule breach, worth a ledger line so the next
person does not think it was overlooked.

**41. The §10 acceptance criterion is not met verbatim.** → **(b)**
`grep -rn "rcParams" eos/` hits **three** files, not one. Two of the three
(`eos/zlvmit/plot_results.py:184`, `eos/zlvmit/table_reader.py:703`) are prose
comments stating that the file does *not* set rcParams. Every one of the ~30
actual assignments is in `eos/general/figure_style.py`. The rule is satisfied;
the grep-based test of it is not, because a grep cannot tell an assignment from a
sentence about assignments. If the criterion is to be an automated gate it needs
to be `grep -rn "rcParams\s*\[" eos/` or an AST check.

---

## 13. What is already ledgered, and correctly

Recorded here so the triage does not re-open settled questions.
`docs/DEFERRED.md` is unusually thorough — most of what a naive audit would flag
is already in it, with the reasoning and often the measurement:

- The astro-import tightening and the `eos/mixed` exception, with a file-by-file
  account of what moved where (`:122-158`).
- The flat-`mu_S` / flat-`mu_e` weak-determination class, with measured
  sensitivities (`:14-46`).
- The JEL-vs-pure-Python fallback tolerance (`:48-62`).
- `docs/STRUCTURE.md` (`:529-537`) and the stale README tree (`:539-546`).
- The `eos_response` freeze selector being a fixed menu rather than a set
  (`:548-576`) and the three-stencil response derivation (`:578-589`).
- The `dd2` `hadronic_row` baryons-only Y_C/Y_S §2 contradiction, with the
  one-line fix, the reason it is not folded into a refactor, and the size of the
  effect (`:722-739`).
- `backends/` deletability measured on dd2: baselines bit-identical at
  rtol = 1e-10, TOV sequences moving by 4.8e-07 (`:741-751`).
- Per-model mode gaps for all ten models, `Y_Lmu` nine times over, the
  SnB-table gap in zl/vmit/alphabag, `dd2`'s species-flag naming (`:786-792`).
- The `mixed` capability gaps, each "a loud NotImplementedError naming the
  phase, never a silent skip" (`:1609-1667`).
- The `astro/tov` fast-backend tidal fragility, the machine-specific crust paths
  and the 2 % two-backend gap (`:1668-1690`).
- The `enjl` notebook and figure script not running from a fresh clone
  (`:1185-1198`) and `plot/` not being in §11's layout (`:1199+`).

---

## 14. Method and reproducibility

- Import graph: `ast`-based walk of all 169 `.py` files under `eos/`, 2489
  intra-repo edges, bucketed by rule. Script kept in the session scratchpad, not
  committed.
- Species-flag scan: `ast` extraction of every `SpeciesFlags` field, then
  word-boundary read counts across each package excluding `species.py`, then a
  manual `__post_init__` check on every zero-read field.
- Pickling and `frozen=` status: measured by importing each
  `eos.<model>.parameters` and round-tripping `Parameters.default()`.
- Every grep in this report was run against the working tree at `136c57c` and
  its output pasted, not paraphrased.
- Four parallel read-only sub-audits (modes+flags, API+layout, conventions+
  integrals, verify+tests+§6) were run and their headline claims re-verified
  first-hand before entering the table — findings 5, 7 and 24 in particular.
