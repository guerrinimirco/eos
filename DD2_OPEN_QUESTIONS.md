# DD2 engine — open questions & pinned-convention decisions

Running log of every physics choice the report/prompt left open that I had to
pin to make progress, plus data we don't have and methodology points worth a
decision. Grouped by theme; each item notes **what I did**, **why**, and **what
would change if we decide differently**. Nothing here blocks the current build —
these are for review.

Status: M0–M8 + M10 complete, M6 partial (fixed-Y_C/Y_S), 93 tests green,
run_full_check passes all 5 checks. **Report v8 resolved B1–B4** (exact DD2Y
R-couplings, density-dependent φ, Marques masses, U_Ξ=−18): DD2Y now reproduces
to <0.1% eps and M_max=2.038 (was a DD2Y-*class* 2.09). Remaining: M6 remainder
(Y_L/entropy-axis/TableSpec) then M9 (Numba+analytic-Jacobian, JAX deferred).

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

---

## F. Things that are settled (no action needed) — for the record
- τ₃=±1 nucleon convention: verified (E_sym=31.67, ρ₀ sign correct).
- DD2 nucleonic sector: exact — TOV M_max 2.419 vs pub 2.42, R_1.4 13.19 vs 13.2;
  CompOSE nucleonic <1e-4; NMPs to 0.18 MeV.
- Σ^R in μ and P but never ε; HVH ≤ 1e-11 through all onsets.
- Rearrangement with density-dependent hyperon/Δ couplings: consistent (HVH holds).
- Fixed-Y_C mode == solve_composition to 1e-8 (mode physics is the gated kernel).
