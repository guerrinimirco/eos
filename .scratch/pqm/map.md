# Map: a parametrized grand potential for colour-superconducting quark matter

Label: `wayfinder:map`
Effort: `pqm`
Charted: 2026-09-11

## Destination

A new `eos` model subpackage, **`eos/pqm`** -- "parametrized quark matter" -- a
closed-form grand potential

    Omega(mu_B, mu_C, mu_S, T; theta)

for deconfined quark matter with and without colour superconductivity, whose
coefficients are fitted to `eos.njl`, which

- **never assumes composition.** `Y_C` and `Y_S` are free; the pairing pattern
  and the flavour content come OUT of minimizing Omega at the given potentials,
  exactly as NJL does. This is the requirement that rules out `eos/abpr` and
  every CFL-assuming bag model, and it is what a mixed phase needs: each pure
  phase evaluated at imposed, non-neutral, non-symmetric potentials.
- is **thermodynamically consistent by construction**: `n_a = -dOmega/dmu_a` and
  `s = -dOmega/dT` in CLOSED FORM from one scalar, so CLAUDE.md section 8's
  Euler relation and free-energy identities hold IDENTICALLY rather than to a
  fit error.
- is fast enough for Bayesian sampling and for `(n_B, Y_C, T)` simulation
  tables, and
- is usable as an `eos/mixed` phase adapter.

Reached when the four acceptance benchmarks below are measured at target, the
package is shipped in the section 5 layout with its `.tex`, and `pqm_phase` is
paired against a hadronic model in `eos/mixed`.

**This map is the GENERALIZATION of work that already exists**, untracked, on
this branch. It is not a build from zero. See "Where this starts" below.

### The acceptance benchmarks

All against `eos.njl` at matched parameters, `n_B` = 1-15 n_sat with **3-10
n_sat weighted heaviest**, the error normalized by `max|P|` over the window (a
pointwise relative error diverges where `P` crosses zero -- that is the whole of
the 20000% figures an earlier pass produced,
`notebooks/csc_bag_map.py:1084-1090`).

| benchmark | target | serves |
|---|---|---|
| T = 0, beta equilibrium | median `\|dP\|/max\|P\|` <= **1%**, max <= 3%; **c_s^2 within 10%** | inference |
| T = 0, Y_C = 0.5, Y_S = 0, no leptons | median <= **2%**, max <= 5% | inference, heavy ions |
| T = 0, Y_C = 0, Y_S = 0, no leptons | median <= **2%**, max <= 5% | inference, mixed phase |
| T = 0-100 MeV, Y_C = 0.01-0.5, mu_S = 0 | median <= **3%** in P; **s within 10%** | merger and supernova tables |
| **phase agreement, EVERY mode** | **>= 90%** | all of them |
| Euler + free-energy identities | <= **1e-10**, structural | section 8 |
| M_max, R_1.4 through `eos.astro.tov` | within **2%** | the application test |

The phase-agreement row is a **blocking gate**, decided while charting: a model
that inverts the branch ordering is useless however small its pressure
residual, which is acceptance criterion 4 of `docs/csc_bag_mapping.md:464-467`
arriving as a gate rather than as prose. It is also the weakest thing measured
so far -- see the table under "Where this starts".

### Where this starts

**MEASURED already, T = 0, on the dense 900-1800 MeV grid** (median
`|dP|/max|P|` over 2-15 n_sat, four gluon-exchange NJL sets):

| mode | median error | phase agreement |
|---|---|---|
| beta equilibrium | **0.3-0.7%** (3 of 4 sets), 2.4% for eta_D=0.75/G_V0=0.75 | **58-78%** |
| `fixed_YC` | 0.8-4.0% | 65-95% |
| Y_C = 0.5, Y_S = 0 | 2.0-3.9% | 88-100% |
| Y_C = 0, Y_S = 0 | 3.3-4.0% | -- |
| the (eta_D, G_V0) map, leave-one-out over 15 NJL points | quadratic **0.9%**, kernel ridge 1.9%, bilinear 4.1% | 98% |

**So the pressure is close to target and the PHASE ORDERING is not.** That is
the shape of this map: milestone 1 closes three structural gaps in the ansatz,
milestone 2 attacks the ordering, milestone 3 ships the package.

**A warning about those numbers.** They live only in a session memory. The
paired `.ipynb` has every output cleared, and the headline in
`docs/csc_bag_mapping.md:601-611` (0.2% max / 0.1% median) was taken on the
ABANDONED `mu_B` = 1500-2600 MeV grid, which is 5-38 n_sat -- almost entirely
ABOVE where stars live, and which the notebook's own comment
(`notebooks/csc_bag_map.py:261-262`) calls "how a fit comes out excellent and
useless". **No accuracy number for the grid the code currently ships exists in
any file.** Ticket 03 fixes that, and until it lands every number in the table
above is a claim without a citation.

## Notes

**Domain.** Nuclear/quark-matter equation-of-state library. `CLAUDE.md` at the
repo root is the specification and overrides defaults. The sections this effort
lives under: section 1 (layering), 2 (conserved charges, **S = +1 per s quark**,
`Y_C` NON-leptonic, `mu_C = mu_p - mu_n`), 3 (modes), 4 (species flags), 5
(uniform API, phase-adapter contract, mandatory internal layout), 6 (parameters
are arguments, non-convergence is a return value, no global state, array in /
array out), 7 (single-home integrals), 8 (thermodynamic invariants), 9
(reference/fast split), 11 (per-model `.tex`), 12 (testing), 13 (names).

**Prior art this map sits on, all UNTRACKED on branch `njl-speed`:**

- `docs/csc_bag_mapping.md` (662 lines) -- the derivation, the conventions, the
  traps, and acceptance criteria 1-5.
- `notebooks/csc_bag_map.py` (1257 lines, paired `.ipynb` with outputs cleared)
  -- the working prototype: the power-series basis, the bounded linear fit, the
  conformal row, the three modes, the (eta_D, G_V0) surface, the benchmark.
- `notebooks/csc_dataset.py` (307 lines) -- a Sobol sampler for a wide
  `(mu_B, mu_C, mu_S, T)` box. **Never run**; `output/csc_ml/` does not exist.
- `output/csc_map_cache/` -- 181 cached NJL samples (135 samples + 46 `ref_*`
  benchmark tables), 2.6 MB, mtimes 2026-09-09 to 2026-09-10.

**Skills a session should consult.** `mattpocock-skills:prototype` for tickets
02, 05, 07, 08; `mattpocock-skills:grilling` + `domain-modeling` for 06;
`mattpocock-skills:diagnosing-bugs` for 07.

### Settled while charting

- **The approach is a parametrized analytic Omega fitted per pattern.** The
  four alternatives were weighed and lose on measured numbers:

  - **Tabulate and interpolate.** `notebooks/csc_dataset.py:15-22` measured the
    killer: a product grid of 100 mu_B x 50 mu_C x 20 mu_S x 10 T is 1e6 solves
    per (parameter set, pattern), and at the measured 10 / 150 / 600 ms per
    solve that is **180 CPU-years** for 25 sets. Inference also needs theta as
    AXES of the table, making it six-dimensional. Decisive.
  - **A neural-network Omega.** Already measured against the closed form on the
    same job: leave-one-out over the (eta_D, G_V0) map gives quadratic 0.9%,
    kernel ridge 1.9%, bilinear 4.1% -- **a kernel method OVERFITS at this data
    volume**, and it got worse with denser data while the quadratic got better.
    A network would also cost a dependency, the analytic `mu(P)` / `mu(n_B)`
    inversion the `eos/mixed` inner loop needs, and parameters with names.
    Rejected -- but CLAUDE.md section 6 says nothing may STRUCTURALLY block a
    future surrogate, and nothing here does: the route stays open in the fog.
  - **Keep speeding up `eos.njl`.** That is a different map,
    `.scratch/njl-speed/`, which rules the surrogate route out of its OWN scope
    (`map.md:249-257`). Its measured ceiling: a converged CFL solve is **80
    ms/pt, 77% jitted quadrature, 4.6% Python**, and a fully compiled Newton
    loop buys **1.05x**. 80 ms to microseconds is not reachable. NJL stays the
    ground truth; the two efforts are complementary, not rivals.
  - **A hybrid** -- adopted in exactly one narrow sense, already in the
    prototype: **Delta is TAKEN from NJL, not fitted.**

- **`arXiv:2301.10765` does not help here, and the reason is instructive.**
  Gaertlein, Ivanytskyi, Sagun & Blaschke fit a colour-superconducting
  density-functional EoS to **ABPR** -- and `eos/abpr` already implements that
  target form. `eos/abpr/thermodynamics.py:198` calls
  `quark_charges(n_B, n_B, n_B)`: `n_C = 0` and `n_S = n_B` IDENTICALLY. It
  cannot leave the CFL line. The paper confirms the term structure is adequate
  ON A LINE and is a warning that it is a line. Not a template.

- **Fit Omega, never P and n separately.** Every basis term is elementary
  except the massive strange Fermi gases, and THEIR `d/dmu_s` and `d/dT` are
  the `n` and `s` that `eos.general.fermi_integrals.solve_fermi_jel` already
  returns alongside `P`. So closed-form derivatives need **no new integral**,
  and the chain rule runs through the constant charge matrix of
  `eos/general/basis.py` (`quark_potentials`), imported rather than re-declared
  (section 2: "basis changes are declared once"). Then
  `eps = -P + sum_a mu_a n_a + T s` and `f = eps - T s` hold by CONSTRUCTION.

- **Delta is eliminated on-shell per pattern, not retained as a variational
  unknown.** `Delta = Delta_star (mu_B/mu_star)^sigma * sqrt(1 - (T/T_c)^2)`,
  with `(Delta_star, sigma)` read from the NJL gaps by a log-log fit. Two
  measurements decide it: left free the gap DRIFTS (a 2SC gap of 372 MeV
  against the NJL 220), and the leading-order condensation coefficient `pair`
  **goes to zero anyway** -- pinning it at its textbook value 1 costs ~20% in
  beta-equilibrium P against 0.2%. So the stationarity argument that would
  favour a variational Delta buys nothing, while an inner minimization per
  evaluation would cost the microsecond-per-point target and introduce local
  minima. The power law is trivially differentiable, so the chain rule through
  Delta is carried exactly and Omega stays ONE differentiable object.

- **The vector axis is `G_V0_over_GS`, NOT `eta_V`.** `eta_V` is read only by
  `vector_form="constant"` (`eos/njl/parameters.py:73`), and that form is
  **unmappable**: measured unpaired rms in P of 2.1% / 9.8% / 13.8% at
  eta_V = 0 / 0.5 / 1, against 3.8% for `gluon_exchange` with every parameter
  inside its window. A constant vector coupling adds `~G_V n_q^2 ~ mu^6` and
  the basis absorbs it only by leaving the physical window. So the continuous
  Bayesian axes are **(eta_D, G_V0_over_GS)**, carried by a QUADRATIC surface
  per coefficient per pattern, and `vector_form="constant"` becomes a
  `docs/DEFERRED.md` entry. `notebooks/csc_dataset.py:107-113` builds its grid
  with `constant` and contradicts this; ticket 04 fixes it.

- **A small Tikhonov ridge is REQUIRED, not cosmetic.** `a4`, `a4l` and `a6`
  are nearly collinear over any finite mu range, so the least squares pins
  their COMBINATION and not each one: the coefficients jump 20-40% between
  neighbouring NJL points while the EoS they predict is good to 2.5%. `RIDGE =
  1e-3` picks the minimum-norm solution out of the valley, which is a smooth
  function of the data -- and a smooth map is the entire object of the
  exercise. The per-coefficient deviation is therefore the WRONG test; the
  right one is whether an EoS built from PREDICTED coefficients reproduces
  `eos.njl` (`notebooks/csc_bag_map.py:822-828`).

- **Branch selection depends on the mode, and the authority is
  `eos/njl/solver.py:36-53`.** At fixed `(mu_B, mu_C, mu_S, T)` -- what an
  `eos/mixed` adapter does -- the stable phase is the one of **largest P**. At
  fixed `n_B` and in beta equilibrium the ranking is by **`f = eps - T s`**.
  These CANNOT agree inside a first-order transition, because there no pure
  phase is the ground state at fixed `n_B` -- a mixture is. Measured on
  `rg_njl1`, beta-eq, T = 0: equal-`f` at `n_B = 0.653 fm^-3`, while equal-P at
  equal `mu_B` puts the window at **0.583 -> 0.736 fm^-3** (3.6 -> 4.6 n_0,
  which is the MUSES module's and Kunkel et al.'s published answer). So a
  fixed-`n_B` switch is NOT the transition and the jump in P across it is not
  physics. `eos_table` therefore returns per-branch tables PLUS the fixed-mu
  window, and the first-order jump is resolved by a construction before any
  table reaches TOV -- section 8's gate runs on the CONSTRUCTED table, and it
  belongs to `pqm`'s `verify/` because `pqm` builds tables a structure solver
  consumes.

- **The gapless caveat is the leading suspect for the phase disagreement.**
  `EoSPoint`'s docstring (`eos/njl/solver.py:474-476`) warns that comparing
  candidates by Omega ACROSS a gapless branch is not valid. The ansatz has no
  gapless branch at all, so where NJL's ground state is gapless the surrogate
  is wrong BY CONSTRUCTION. `docs/DEFERRED.md` records that a fixed-Y_C = 0.1
  table once reported metastable 2SC at 98 of 100 points because the gapless
  CFL that minimised `f` could not reach the gate. Ticket 07 measures how much
  of the 22-42% this explains before ticket 08 tries to fix it.

- **The name is `pqm`.** `cscbag` was rejected: the shipped basis is a POWER
  SERIES, not a bag model -- the bag form is the sub-basis
  `(a0, a2s, a4, pair)`, measured and superseded. `csc` was rejected because
  the model must carry the unpaired branch too, and that is the one it fits
  best.

- **The fitting code is NOT model code.** It consumes `eos.njl` and produces an
  `eos.pqm.Parameters`, so it stays in `notebooks/`. That is what keeps section
  1 intact: **`eos/pqm` never imports `eos.njl`.** Two prototype imports must be
  dropped on the way in -- `notebooks/csc_bag_map.py:95-96` pulls
  `eos.alphabag.thermodynamics.fermi_thermo` and `eos.mixed.njl_phase`; the
  massive-gas terms take `solve_fermi_jel` from `eos/general/fermi_integrals.py`
  directly instead.

- **Milestone 1 proves accuracy in the notebook BEFORE `eos/pqm` is written.**
  Decided while charting. The basis is still changing in three ways (closed-form
  derivatives, the conformal fix, finite T), and a section 5 subpackage whose
  `parameters.py` has to be reworked twice is worse than a notebook that moves.

- **One branch, `njl-speed`, and no commits from a map session.** This checkout
  is shared and a second agent is in it: never a bare `git commit`, never the
  full test suite as part of this work. Timing on this laptop varies +-30% with
  concurrent work and drops to a ~5% duty cycle on battery, so **a single run is
  not a measurement**; every timing ticket states n, the median, the
  interpreter, and its numpy and scipy versions, and prints cpu beside wall.

- **The "300x" speed claim is not citable and must be re-pinned.** It rests on
  a 175 ms `eos.njl` baseline with no recorded provenance, inconsistent with
  every pinned timing in `.scratch/njl-speed/issues/02-pin-the-benchmark.md`.
  The honest statement today: the model is **0.5-1.0 ms/pt for a full
  equilibrium solve** (the root finds included) and **~10 microseconds for a
  bare pressure evaluation**; the NJL comparator is **56 ms** (2SC alone),
  **719 ms** (CFL alone) or **8100 ms** (the default four-pattern enumeration).
  So the ratio is 50x, 700x or 8000x depending on what is being replaced, and
  a claim that does not say which is not a claim.

## Decisions so far

<!-- one line per closed ticket; the detail lives in the ticket, not here -->

- **01 closed.** `n_a` and `s` are closed form from one Omega; the identities
  hold at 1e-16 and the FD gate falls as h^2 in every pattern. Two structural
  finds: the gap power law must run on the LOCKED potential mu_B + mu_S or CFL
  loses n_S = n_B off the sampled line (`pair` then collapses to 1.7e-8, at no
  cost in rms), and the massive-gas threshold cut was a T > 0 discontinuity.
  `a4` has NO ceiling -- measured 2.67 / 3.13 / 2.81, so a 2 would peg like
  the 1 did; the conformal row is what bounds the mu^4 sum.

## Not yet specified

- **The uSC and dSC patterns.** `eos.njl` enumerates them and over 75 scanned
  parameter points they won exactly twice, metastable both times. Whether they
  need their own coefficient set, or whether refusing them is the right
  statement, is not answerable until ticket 07 says what the phase
  disagreement is made of.

- **Representing a gapless branch at all.** If ticket 07 finds the
  disagreement is mostly gapless states, the question becomes whether the
  ansatz can carry a gapless correction (a term that switches on when a
  quasiparticle branch crosses zero) or whether the honest answer is a
  `docs/DEFERRED.md` entry and a stated domain. Both are live; neither is
  specifiable now.

- **Whether `m_s` needs a density dependence near the onset.** Measured: the
  ansatz is at its best deep inside the quark phase and at its worst near the
  terminus -- which is precisely where a hybrid star puts its quark core,
  because the NJL constituent masses are still running there (`M_s` runs ~460
  to ~200 MeV across these branches). The three fixed-mass Fermi gases are the
  current answer and they only partly work (`n_S` rms 9.5% -> 7.6%). Whether
  more masses, a running mass, or a stated lower bound on the domain is right
  waits on ticket 06.

- **Varying the tier-1 NJL parameters** (Lambda, G_S, K, the current masses).
  The surface is fitted at one vacuum tier; varying them needs a refit, and
  whether that is a second surface axis or a re-run is not decidable until the
  first surface is measured with T data (ticket 09).

- **An ML surrogate for the coefficient map, if the quadratic stops
  sufficing.** Measured today at 0.9% leave-one-out and improving with denser
  data, so it does not need replacing -- but if the finite-T refit degrades it,
  the question returns. Nothing in the design blocks it.

- **Whether `pqm` should also carry a `frozen_thermo` for the mixed engine's
  frozen response.** Densities are closed form, so it is cheap; whether the
  `chi` freeze needs it is a question for after ticket 11.

## Out of scope

- **Speeding up `eos.njl` itself.** That is `.scratch/njl-speed/`, an active
  map with its own destination. This map CONSUMES njl at whatever speed it has
  and pays the data-generation cost once.

- **Crystalline (LOFF) quark matter.** `eos.njl` does not carry it, so there is
  no ground truth to fit, and the ansatz could not represent it either. The
  correct treatment is to say so.

- **The hadronic side of any hybrid star.** `pqm` is a quark phase; the
  pairing against DD2/SFHo/DID goes through the existing `eos/mixed` contract
  unchanged.

- **Committing or landing the untracked prototype files.** `docs/`,
  `notebooks/csc_*` and `.scratch/njl-speed/` are untracked on a shared
  checkout and a second session owns that decision. This map reads them and
  adds only `.scratch/pqm/`.

- **Changing the `eos/mixed` phase-adapter contract.** `pqm_phase` fits the
  existing `Phase` / `PhaseThermo` surface as it stands -- it needs LESS of it
  than `njl_phase` does, not more, because it has no internal solve.
