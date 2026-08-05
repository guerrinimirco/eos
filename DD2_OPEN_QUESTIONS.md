# DD2 engine — open questions & pinned-convention decisions

Running log of every physics choice the report/prompt left open that I had to
pin to make progress, plus data we don't have and methodology points worth a
decision. Grouped by theme; each item notes **what I did**, **why**, and **what
would change if we decide differently**. Nothing here blocks the current build —
these are for review.

Status: **M0–M10 complete + hardened** (Phase 1 done). 123 tests green,
run_full_check 7/7 (golden, identities, coefficients, coeff-analytic-vs-FD,
backend parity, CompOSE, TOV). DD2Y reproduces to <0.1% eps, M_max=2.038.

### RESOLVED in report v11 + follow-ups (2026-07-20)
- **Δ calibration (B5/B6)**: no canonical DD2Δ table; default universal
  x_Δσ=x_Δω=x_Δρ=1, plus `from_delta_potential(U_Delta∈[-100,-50], x_wD, x_rD)`
  inverting x_Δσ. M5 validated against a chosen (U_Δ, x_Δω) point.
- **Numba (M9 "fast")**: `kernel_numba.py` jits the T=0 residual+Jacobian
  (machine-identical to NumPy, ~1.9× on a hyperon sweep); `analytic_jac=True`
  routes T=0 through it, T>0 stays NumPy-analytic (JEL doesn't trace).
- **Coefficients from J (M10)**: `coefficients_jac.py` — susceptibilities χ_ab
  (pure J, symmetric, matches grand-canonical FD 2e-3), c_s² (J tangent, matches
  FD 1e-6), C_V/C_P per baryon (C_P≥C_V; C_V~1% vs FD, T>0 JEL floor).
- **Fixed-Y_C flavor 2b** (neutralizing e+μ) added earlier (report v9).

Deferred (confirmed): **full JAX** (revisit only if Numba profiles as the
bottleneck); **σ*** (guarded slot — DD2Yσ* fails 2 M_⊙, not needed for NS EoS).
On hold: **μ-family ν-trapping** (electron-family Y_L only; user confirming need).

### RESOLVED in report v8 (2026-07-19)
- **B1** exact DD2Y R-couplings hardcoded in `from_dd2y_defaults()` (Fortin 2017
  Table 1: R_σ 0.62/0.48/0.32, SU(6) vectors). eps 4–8% → 0.02–0.09%.
- **B2** U_Ξ = −18 (DD2Y), default fixed; the inversion path is now
  `from_hyperon_potentials()` and regenerates R_σ to ~1%.
- **B3** Marques hyperon masses (Λ=1115.683, Σ=1190, Ξ⁻=1321.68, Ξ⁰=1314.83),
  carried in the coupling row.
- **B4** φ density-dependent, inheriting f_ω: Γ_φY(n_B)=x_φ·Γ_ωN(n_B), with the
  matching ∂Γ_φY/∂n rearrangement term.
- **A2/A3/A4/C1/C3/D1/E2** confirmed as-built (Δ τ₃=±3,±1; m_φ=1019.45;
  σ*=975 off-by-default; CompOSE only in uniform region n_B≳0.2 via both the
  general-purpose fixed-Y_q and cold-NS gates; residual-norm acceptance; thermal
  vectors off by default). **B5/B6** (DD2Δ ratios, scalar collapse) held as a
  separate follow-up per the user (Δ at x_Δ=1.0 for now).

---

## A. Conventions I pinned (report left them open) — please confirm

### A1. Nucleon mass: average vs physical  *(load-bearing)*
- **Did:** kernel uses the **average** nucleon mass `(m_n+m_p)/2` by default
  (`Parametrization.nucleon_mass_mode='average'`); a `'physical'` mode uses
  `m_n≠m_p`.
- **Why:** the report §2.7 golden points (and `dd2_reference_validation.py`) only
  reproduce with the average mass. Physical mass shifts μ_p by ~0.6 MeV and fails
  the 1e-4 golden gate. CompOSE (HS(DD2)) uses physical, so the comparison path
  switches to `'physical'`.
- **Open:** is "average for golden / physical for CompOSE" the intended split, or
  should the production EoS always be physical (and the golden points are just a
  convention artifact of the oracle)?

### A2. Δ isobar τ₃ normalization  *(load-bearing for Δ onset)*
- **Did:** `t3(Δ) = 2·I₃ = {+3,+1,−1,−3}` for {Δ⁺⁺,Δ⁺,Δ⁰,Δ⁻}, and
  `x_ρΔ = g_ρΔ/g_ρN` multiplies these — consistent with the nucleon τ₃=±1 rule
  the report states in §1.4.
- **Why:** the report explicitly deferred "the τ₃ normalization for the quartet"
  to M5 but states the rule `t3 = 2 I₃`; applying it uniformly gives ±3,±1.
- **Open:** some DD2Δ works normalize the ρΔ coupling with I₃ (±3/2,±1/2) and
  absorb factors into g_ρΔ. This changes the Δ⁻ onset density. Which does the
  DD2Δ / R(DD2YΔ) CompOSE model use? (Affects M5 onset comparison.)

### A3. φ meson mass (mean-field)
- **Did:** `m_φ = 1019.45 MeV` (your pick, PDG φ(1020)).
- **Open:** confirm; the report's §A.4 thermal-meson table lists 1020. Negligible
  numerically.

### A4. σ* (hidden-strange scalar) mass
- **Did:** not yet used (`sigma_star` flag off, guarded to raise). When wired,
  the report suggests m_σ* ≈ 975 MeV (f₀(980)).
- **Open:** confirm mass and whether we ever turn σ* on (ΛΛ physics; report keeps
  it a pluggable slot, default off).

---

## B. Coupling data we don't have (the DD2Y / DD2Δ reproduction gap)

### B1. Exact DD2Y hyperon R-couplings  *(this is the main one)*
- **Did:** `from_dd2y_defaults()` derives hyperon couplings from the report's
  generic recipe: SU(6) vectors + scalar couplings inverted from
  U_Λ=−30, U_Σ=+30, U_Ξ=−14 MeV.
- **Result:** a DD2Y-**class** model. Λ onset 2.2 n_sat ✓, M_max 2.09 ≥ 2 ✓, but
  hyperonic pressure differs **4–8% (median), up to ~22% at Y_q=0.3** from the
  DD2Y general-purpose CompOSE table (measured in fixed-composition mode, so
  **not** a lepton-cancellation artifact — it's the couplings).
- **Need:** Marques, Oertel, Hempel, Novak (PRC 96, 045806, 2017) Table 1 gives
  R_σY, R_ωY, R_φY, R_ρY per hyperon. The report says these exist but doesn't
  reproduce them. **CompOSE does not ship the parametrization** (only the tables),
  so we likely need the paper. With those numbers it's a one-line change to
  `from_dd2y_defaults()` and we'd expect <0.1%.
- **Question for us:** do we want bit-exact DD2Y, or is a DD2Y-class model
  (correct onsets, M_max, structure) sufficient for the project's purpose?

### B2. Hyperon potentials — exact values
- **Did:** U_Λ=−30, U_Σ=+30, U_Ξ=−14 (report §2.4 "typical inputs").
- **Open:** are these Marques' exact values? U_Ξ in particular is debated
  (−14 vs −18 in the literature); it shifts the Ξ onset and softening.

### B3. Hyperon masses
- **Did:** PDG values from `particles.py` (Λ=1115.68, Σ=1189/1193/1197,
  Ξ=1314.86/1321.71).
- **Open:** DD2Y/Fortin sometimes use rounded masses (1116, 1189, 1193, 1197,
  1315, 1321). Sub-MeV effect on onsets; confirm which set.

### B4. φ coupling: constant vs density-dependent
- **Did:** constant `Γ_φY = (SU6 factor)·Γ_ωN(n_sat)` (DD2Y default,
  `phi_density_dependent=False` per report). Δ has S=0 → no φ.
- **Open:** confirm DD2Y uses constant φ (not f_ω-inherited). The report offers
  both behind a switch; I implemented the constant default and left the
  density-dependent variant unimplemented (would reuse f_ω).

### B5. Exact DD2Δ coupling ratios
- **Did:** `x_Δσ = x_Δω = x_Δρ = 1.0` default (within the report's stated range
  x_Δσ~1.0–1.3, x_Δω~1.0), all configurable.
- **Open:** the CompOSE `R(DD2YΔ)` entries are named "1.2-1.1", "1.1-1.1",
  "1.2-1.3" — presumably (x_ωΔ, x_σΔ) or similar. Which pair, and which ordering?
  This sets the Δ onset and whether the scalar collapse (B6) is hit.

### B6. Δ scalar collapse at high density
- **Observed:** with x_Δσ=1.0, the abundant Δ's drive m*→0 (scalar collapse) at
  ~6.5 n_sat — a real DD-RMF feature, flagged as a feasibility boundary (the EoS
  is truncated there, so a Δ TOV M_max isn't a clean gravitational turning point).
- **Open:** is this expected for the DD2Δ we're targeting, or do the intended
  x_Δ ratios keep m*>0 to higher density? Answering B5 resolves this.

---

## C. CompOSE comparison methodology

### C1. HS(DD2) low-density cluster tail
- **Did:** compare only at n_B ≳ 0.2 fm⁻³ (uniform-matter region); below that the
  HS statistical model has light nuclei/clusters my uniform-matter engine doesn't.
- **Open:** confirm we only ever validate against CompOSE in the uniform region
  (i.e. the crust/cluster regime is out of scope for the DD2 *uniform* engine and
  handled separately by the TOV crust attachment).

### C2. `eos.thermo.ns` column interpretation
- **Did:** parsed the cold-NS 1D table with the standard CompOSE column layout
  (Q1=P/n_B, …, Q7=e/(n_B m_n)−1, trailing = Y_q). Numbers come out physical.
- **Open:** confirm the trailing columns; the DD2Y `.ns` had two extras
  (Y_q ≈ 0.05–0.46 and a second ~0.99 column) I read as (Y_q, something).

### C3. Which CompOSE table is the DD2Y gate?
- **Did:** used both the general-purpose 3D (n_B,T,Y_q) table (fixed-Y_C mode) and
  the cold-NS 1D `.ns` (β-eq). Both show the B1 coupling gap.
- **Open:** is the general-purpose table (fixed composition) the intended M4/M6
  validation target, or the cold-NS one, or both?

---

## D. Numerical / method choices

### D1. Solver acceptance on residual norm
- **Did:** accept a solve when `‖residual‖_∞ ≤ 1e-10`, regardless of scipy's
  `success` flag (hybr reports "no progress" at converged round-off roots).
- **Open:** fine? It's the physically correct criterion; just flagging it.

### D2. Warm-start / continuation robustness
- **Did:** step-bisection continuation through onsets (report §3.4), plus a
  `stop_at_boundary` mode that returns the valid prefix at a feasibility boundary
  (Δ collapse) instead of crashing.
- **Open:** the report's fuller ladder (tangent predictor, default-guess restart)
  isn't all built yet — current bisection suffices through hyperon/Δ onsets.

### D3. T=0 threshold kinks & the JAX port (M9)
- **Note:** at T=0 a species right at threshold has a `(ν²−m*²)^{3/2}` kink
  (zero-slope density), which stresses finite-difference Jacobians. Fine so far
  with hybr; **flagging for M9** — autodiff through the kink and the `|μ|≤m`
  Bose clamp may need the report's authorized Numba+analytic-Jacobian fallback.

---

## E. Scope decisions for the remaining milestones

### E1. M6 remainder — Y_L / neutrino-trapped + entropy axis
- **Deferred:** fixed-Y_L (neutrino-trapped) needs the neutrino sector, currently
  guarded off. The entropy-per-baryon axis (outer T-solve) and the `TableSpec`
  driver aren't built.
- **Question:** priority of trapped-neutrino matter vs the rest of M7–M10?

### E2. M7 thermal meson gas — vector double-count
- **Plan:** pseudoscalar nonet reuses the existing repo implementation; the
  thermal **vector** nonet is new. The report says keep `include_thermal_vectors`
  **off by default for cold NS** (the thermal ρ,ω,φ quanta partly double-count the
  mean fields). Confirm default-off is what we want.

### E3. M8 NMP inverter
- **Plan:** the §2.5 cascade (saturation point → curvatures → isovector →
  hyperons), with feasibility flags (§2.6). `from_nmp` currently raises
  NotImplementedError.
- **Question:** which NMP set is the primary input, and do we want the generalized
  (rational) Γ_ρ(n) form to make K_sym an independent knob (report §2.3)? Current
  Γ_ρ is single-exponential, so K_sym is *predicted*, not free.

### E4. M9 eos_fast (JAX)
- **Big effort.** The `xp` namespace is in place. Question: is the JAX backend a
  priority, or is `eos_ref` (NumPy/scipy) sufficient for now? The report authorizes
  a Numba + analytic-Jacobian fallback if JAX-tracing the JEL/Bose core is
  impractical (likely, per D3).

### E5. M10 coefficients
- **Plan:** both speed-of-sound flavors (equilibrium `dP/dε` along the sequence,
  and adiabatic/frozen), C_V, C_P, susceptibilities, thermal index — from the
  equilibrium Jacobian. Straightforward once M9's derivative engine exists (or via
  finite-difference in `eos_ref`).

### E6. Mixed-phase solver nesting (deferred, measured)
- **What it is:** the outer mixed root calls a *nested* inner DD2 root through
  `hadronic_phase`, roughly 15 times per converged point (measured: 1619 inner
  solves and 26 654 `_hadronic_residual` calls for a 105-point window sweep).
- **Consequence:** the analytic mixed Jacobian is a net *loss* — 11.2 vs
  9.5 ms/pt — because `_hadronic_block` and `_fd_quark_kappa` finite-difference
  *through* that inner solve, so the "analytic" Jacobian is partly numeric and
  costs more than it saves. `MixedTableSpec.analytic_jac` therefore stays
  `False` by default, unlike the DD2 one.
- **The fix, when it is worth doing:** fold the hadronic field/density unknowns
  into the outer unknown vector, so there is one MINPACK solve per point. Est.
  a further 2-3x, and it is what would make the analytic Jacobian pay.
- **Why deferred:** it restructures `eos/mixed/solvers/phases.py` and the
  equilibrium layer, which is a much larger and riskier change than the jitted
  `meson_sources_t0` inner loop that was taken instead (measured 1.9x).

### E7. TOV backend difference across a Maxwell jump (known, bounded)
- **What it is:** at eta = 1 the `fast` (Numba) and `scipy` TOV backends give
  M_max differing by 4.1e-3 Msun (~0.2%). At eta = 0 they agree to 1.4e-4.
- **Not a resolution artifact:** the gap is flat from n_ec = 100 to 400, so it
  is a systematic in how the two treat the density discontinuity, not central-
  density discretization. R(M_max) agrees to ~0.002 km either way.
- **Status:** `backend="fast"` is the shipped default (it is ~600x faster and
  is what makes a Bayesian scan affordable); `backend="scipy"` remains the
  reference. Pinned by `test/mixed/test_tov_backend_parity.py`.
- **Open:** whether the 0.2% matters depends on the inference's M_max
  likelihood width. Revisit if a Maxwell-dominated posterior is the target.

### E8. What "frozen" means in the mixed-phase sound speed (convention, pinned)
- **The choice:** `eos/mixed/coefficients.py:sound_speed_frozen` holds **chi**
  fixed, compresses both phases by the same factor, and holds each phase's
  **Y_C and Y_S** fixed while re-neutralising with leptons.
- **Why chi is the part that matters:** freezing only the charge fractions
  would let the solve readjust chi and slide back along a Maxwell plateau, so
  c_ad would collapse with c_eq and the two definitions would carry no extra
  information. With chi frozen, c_eq -> 0 at eta = 1 while c_ad stays ~0.55.
- **Where it is weaker than a full freeze:** with hyperons or Deltas active the
  individual species still re-equilibrate among themselves at fixed total Y_C
  and Y_S. For nucleonic matter the two coincide, and the chi = 0, leptons-off
  limit reproduces `eos.dd2.coefficients.sound_speed_adiabatic` to 1e-13
  (pinned by `test/mixed/test_coefficients.py`).
- **Leptons are not a detail:** including them changes c_ad by several percent.
  They are on by default (stellar matter is neutral); `leptons=False` exists so
  the DD2 nucleonic limit is directly comparable.
- **Open:** a strictly frozen per-species composition would need a "hadronic
  thermo at given densities" entry point that `eos/dd2` does not have
  (`solve_composition` is nucleons only). Add it if a hyperonic frozen sound
  speed is ever needed quantitatively.

### E9. K_sat cannot be moved at fixed Q_sat — and this is the long-standing M8 failure
- **Measured:** at the DD2 Q_sat (168.7 MeV) the isoscalar inversion converges
  ONLY at the DD2 K_sat. The residual grows smoothly away from it — 2.6e-3 at
  K_sat = 243, 2.7e-2 at 240, 5.0e-2 at 250, 0.21 at 220 — against a 2e-2 gate.
  L_sym by contrast is free across 30-100 MeV (isovector residual ~1e-9).
- **This is the cause of `test/dd2/test_dd2_m8.py::test_perturbed_nmp_solves`,**
  which has been failing for some time. It asks for `K_sat=250, L_sym=60` and
  calls it "a nearby feasible NMP set"; the K_sat half of that premise is what
  is wrong. The test has NOT been loosened — the tolerance is not the problem.
- **Why:** K_sat and Q_sat are tied by the cross-constraint that closes the
  6x6 isoscalar system, so they are not independent knobs. A joint (K_sat,
  Q_sat) scan does find a feasible ridge — (230, 100) inverts, as does
  (243, 169) — but it is narrow, so a rectangular grid mostly misses it.
- **Consequence for the Bayesian goal:** a prior that varies K_sat
  independently will reject nearly every sample. Either sample along the
  (K_sat, Q_sat) ridge, or treat Q_sat as determined by K_sat via a 1-D root
  solve on the cross-constraint rather than as a free parameter.
- **Open:** is the 2e-2 gate on the isoscalar residual the right threshold, and
  would a better-conditioned isoscalar solve widen the feasible region? Both
  are worth checking before the inference run.

### E10. Spurious chi=0 crossings with hyperons + Deltas at low bag constant
- **What happens:** `locate_window` reports an onset where the mixed branch it
  converges to sits at a LOWER pressure than the hadronic branch at the same
  density — so it is not the favoured state. Measured with hyperons + Deltas,
  L_sym = 85, B^1/4 = 165 MeV: at the claimed onset chi jumps 0 -> 0.517 and P
  falls 36.3 -> 7.8 MeV/fm^3.
- **Consequence:** `build_mixed_eos_table` stitches a table that steps DOWN in
  pressure, c_s^2 goes to -4, and TOV integrates it to maximum masses in the
  hundreds of solar masses without raising.
- **Mitigation shipped:** `eos/mixed/scan.py:eos_is_physical` checks P is
  non-decreasing and 0 <= c_s^2 <= 1 before any TOV call, and the scan reports
  `eos_unphysical` instead of a mass. On one representative grid this rejected
  21 of 27 apparent transitions. Maxwell plateaus (dP = 0) still pass.
- **Open:** the real fix is in the window locator — it should reject a chi
  crossing whose branch is not pressure-favoured over the pure phase, rather
  than leaving every downstream consumer to check. Not attempted here because
  it changes solver behaviour that the whole `test/mixed` suite is pinned to.

### E11. chi runs BACKWARDS in density for vMIT a >= 0.25 fm^2

- **Where:** `eos/mixed`, quark vector coupling `a` at or above ~0.25 fm^2.
- **Symptom:** the mixed solve converges and the mixed branch is energetically
  favoured (lower eps than pure hadronic at the same n_B), but chi *decreases*
  with density instead of rising. Measured at B^1/4 = 150 MeV, a = 0.25,
  m_s = 150, hyperons + Deltas: chi = 0.586 / 0.477 / 0.208 at
  n_B = 0.4 / 0.6 / 0.8 fm^-3, against eps_mix - eps_had = -9.0 / -6.9 / -1.1
  MeV/fm^3. At a = 0.20 the same parameters give chi = 0.870 / 1.013 / 1.265,
  which is the expected direction.
- **Consequence:** `MixedWindow.exists` requires n_offset > n_onset, so these
  samples are reported `no_window`. A scan over a therefore shows a hard edge
  at a = 0.20 that is really "no *ordered* window", not "no transition". This
  was checked against the obvious alternative explanation — that the onset had
  simply moved above the grid — and it is not that: a corner scan over
  B^1/4 in [120, 200] MeV at a in {0.25, 0.30, 0.40} returned `no_window` for
  all 576 samples, so lowering the bag constant does not recover an ordered
  window.
- **Pinned:** the empirical viable range is quoted as 0.05 <= a <= 0.20 fm^2,
  with the a >= 0.25 edge flagged as an unexplained solver/physics boundary
  rather than an established bound on the vMIT vector coupling.
- **Open:** whether the retreating-chi branch is a genuine feature of vMIT at
  strong vector repulsion or the same branch-selection pathology as [E10] seen
  from the other side. Both show the solver settling on a chi that the pure
  phases do not support; E10's mitigation (`eos_is_physical`) does not catch
  this one, because these tables are rejected earlier, at the window stage.

---

## G. Extended NJL engine (`eos/enjl`, Xia 2024 PRD 110 014022)

### G1. The effective scalar density is capped at zero — pinned, not in the paper

- **Where:** `eos/enjl/uniform.py::_effective_scalar_densities`.
- **The issue:** Eq. (6) reads nbar^s_q = n^s_q + alpha_S sum_i N^q_i n^s_i.
  The quark term is negative (vacuum-subtracted) but the baryon cluster term is
  positive and grows with baryon density, so above chiral restoration the sum
  turns positive: at n_b = 10 fm^-3 in the f_q = 0.7, B = 1 set it reaches
  +2.5 fm^-3 for the u flavor. A positive nbar^s_q is a scalar condensate of
  the wrong sign and drives M_q below its current mass m_q0.
- **Pinned:** nbar^s_q is capped at zero from above, so M_q >= m_q0 always.
- **Evidence this is what the author's own implementation does**, all from the
  reference tables in `test/enjl/reference/`, which are that implementation's
  output: (i) the `Sigmaq` columns are written as exactly 0 on precisely the
  rows where the uncapped expression would be positive, while `Sigmas` on the
  same rows is reproduced by the uncapped formula to 1e-5 fm^-3; (ii) with the
  cap, the computed nbar^s_q matches the `Sigmaq` columns over *all* solved
  rows of all five files to <= 9.5e-4 fm^-3; (iii) M_s on those rows follows
  from m_s0 - 4 G_S nbar^s_s with the u and d contributions to the 't Hooft
  term switched off by the cap, which reproduces the printed M_s exactly
  (386.887 at n_b = 10); (iv) the E_0 offset extracted from Eq. (13) stays
  density-independent, which it would not if the condensate energy used
  uncapped values.
- **Consequence if we decide differently:** without the cap the uniform solver
  does not merely lose accuracy, it fails — the gap equation acquires spurious
  negative-mass roots and `solve_point` either raises or returns M_q < 0.
- **Open:** whether the paper intends the cap as a constraint on the field
  equations or whether Eq. (6) is meant to be read with a cluster term that
  cannot exceed the quark term. The two agree everywhere the tables can
  distinguish them, so nothing downstream depends on the answer.

### G2. Where the chiral knee falls, to one grid step

- **Where:** `Beta_fq0.7_B0.dat` at n_b = 0.63 and 0.64 fm^-3 only.
- **Symptom:** the reference tables pin M_d to m_d0 = 5.5 MeV on those two
  rows while this solver's nbar^s_d is still marginally negative, giving
  M_d = 6.30 and 5.77. Through the 't Hooft term this also moves M_u
  (8.79 against the table's 10.42 at n_b = 0.63), and through Eq. (4) it moves
  M_Lambda, which is where the residual shows up: 0.26 MeV in mu_Lambda.
- **Pinned:** reported rather than tuned away. Excluding those two rows the
  file agrees to 0.018 MeV over its whole range, which is the same figure an
  independent rebuild of the mean field from the table's own columns reaches —
  so the difference is the placement of the knee, not the mean field.
  `test/enjl/test_enjl_fixed_composition.py` asserts both numbers.
- **Open:** whether the reference run applied a tolerance when testing
  nbar^s_d <= 0 that this solver does not. Sub-MeV, and confined to the two
  grid points either side of chiral restoration in one parameter set.

### G3. Branch selection above the transitions — UNRESOLVED, blocks Figs. 4-8

- **Where:** `eos/enjl/eos_beta.py::beta_eos_table`, above the first-order
  transitions of each parameter set.
- **The situation:** the local equations (23)-(24) have **three** solution
  branches, all self-consistent, all charge-neutral to 1e-15, all satisfying
  the gap equation: (i) chirally broken hadronic; (ii) chirally restored but
  still confined — the quarkyonic branch, baryons *and* quarks; (iii) fully
  deconfined quark matter, n_p = n_n = 0, n_u = n_d = n_s. Continuation follows
  whichever branch it starts on straight past a transition, indefinitely.
- **Two selection rules were tried and neither reproduces all five tables:**

  | rule | fq1.0_B0 | fq1.0_B1 | fq0.7_B0 | fq0.7_B1 | fq0.5_B1 |
  |---|---|---|---|---|---|
  | upward continuation | 1 | 7 | **183** | 1 | 16 |
  | lowest eps at fixed n_b | 0 | 0 | 18 | **79** | 0 |

  (rows out of ~250 where mu_b differs from the table by more than 0.5 MeV.)
- **The specific contradiction:** for f_q = 0.7, B = 1 GeV/fm^3 branch (iii)
  exists from n_b ~ 2 fm^-3 up and has *both* lower eps at fixed n_b and higher
  P at fixed mu_b than the branch the reference table follows — at n_b = 7.2 it
  gives eps = 32545 against the table's 44204 MeV/fm^3. Either criterion
  therefore selects it. But Paper 1 states explicitly that full deconfinement
  to n_b^Q/n_b = 1 occurs for (0.5, 0), (0.5, 1) and (0.7, 0) and *not* for
  (0.7, 1), (1, 0), (1, 1), and the reference table agrees with the paper. So
  branch (iii) is reachable in this implementation and is not in the author's.
- **What is confirmed working:** the Maxwell machinery itself. Constructing the
  crossing of branches (i) and (ii) for f_q = 0.7, B = 1 on a 0.1 fm^-3 grid
  gives mu_b = 1164.9 MeV, P = 69.33 MeV/fm^3 against the table's recorded
  chiral transition at 1168.4748 and 69.6419 — 0.3% on both, on a grid far too
  coarse to do better. So the transition-finding is right and only the
  admissibility of branch (iii) is in question.
- **Open, and the next thing to settle:** what excludes branch (iii) in the
  author's treatment. Candidates: an additional constraint on n_b^Q that this
  implementation does not impose; a restriction that baryons cannot be removed
  by hand once bound; or the possibility that the author's solver simply never
  reaches it from a continuation and the branch is physical but not explored.
  Until this is decided, `beta_eos_table` deliberately does *not* choose:
  it maps one branch per call, with `direction` selecting which.

### G4. Finite-temperature decisions — not yet taken

Thermal antibaryons, thermal antiquarks and whether the paper's "quarks are
restricted to the lowest energy states" remark has finite-T content are all
open. They are decisions for the finite-temperature work and are recorded here
when that starts, not before.

---

## F. Things that are settled (no action needed) — for the record
- τ₃=±1 nucleon convention: verified (E_sym=31.67, ρ₀ sign correct).
- DD2 nucleonic sector: exact — TOV M_max 2.419 vs pub 2.42, R_1.4 13.19 vs 13.2;
  CompOSE nucleonic <1e-4; NMPs to 0.18 MeV.
- Σ^R in μ and P but never ε; HVH ≤ 1e-11 through all onsets.
- Rearrangement with density-dependent hyperon/Δ couplings: consistent (HVH holds).
- Fixed-Y_C mode == solve_composition to 1e-8 (mode physics is the gated kernel).
