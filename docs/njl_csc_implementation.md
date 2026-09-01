# Three-flavour NJL with colour superconductivity — implementation specification

Companion to `ccdm_implementation.md`. Same units, same index conventions, same
standard of evidence: **every number in this document is printed by
`verify_njl_csc.py`**, which exits `ALL CHECKS PASSED` (106 assertions). Where the
literature could not supply a coefficient, this document says so instead of
inventing one.

The register is deliberate. This is not a review — it tells a coder what to
compute, in what order, and flags the specific mistakes that return a
plausible-looking wrong answer.

---

## 1. Conventions (inherited — do not re-derive)

| quantity | convention |
|---|---|
| units | ħ = c = k_B = 1, MeV throughout |
| densities | MeV³; P, ε, Ω in MeV⁴; convert only at output |
| ħc | 197.3269804 MeV fm |
| flavour | f ∈ {u, d, s}; colour a ∈ {r, g, b}; nine modes j = (f,a), **flavour-major**: j = 3·i_f + i_a |
| charges | q_u = +2/3, q_d = q_s = −1/3 |
| strangeness | s_s = **+1** per s quark — **opposite to the PDG sign**, used consistently. Never flip it silently. |
| spin degeneracy | g = 2 per mode |
| colour generators | (T₃)_{r,g,b} = (+½, −½, 0); (T₈)_{r,g,b} = (+⅓, +⅓, −⅔) |

μ of a mode:

```
μ_{f,a} = μ_B/3 + q_f μ_C + s_f μ_S + (T₃)_a μ₃ + (T₈)_a μ₈ ,     μ_e = −μ_C
```

### 1.1 Generator normalisation — a real conversion factor

Three normalisations are in circulation, and mixing them corrupts μ₈ by factors
of order 1.15–1.7. T₃ = diag(½, −½, 0) is universal; only T₈ varies.

| convention | T₈ | used by | μ₈ conversion |
|---|---|---|---|
| **this document** | diag(1, 1, −2)/3 = λ₈/√3 | — | — |
| **halved Gell-Mann** | √3 T₈ = diag(½, ½, −1), i.e. λ₈/2 | **Rüster *et al.*, Pagliara–Schaffner-Bielich, Kunkel *et al.*** | μ₈^theirs = (2/√3) μ₈^ours = 1.1547 μ₈^ours |
| **full Gell-Mann** | λ₈ = diag(1, 1, −2)/√3 | Buballa, Steiner–Reddy–Prakash, Blaschke | μ₈^ours = √3 μ₈^theirs |

All three were read off the sources directly. Note that the largest group —
Rüster, PSB and Kunkel — share **one** convention, written identically as
`√3 T₈ = diag(½, ½, −1)`; an earlier version of this document wrongly filed PSB
alongside our own normalisation.

Worked consequences, because they bite when checking against a paper:

- SRP's CFL result μ₈ = −(1/(2√3)) m_s²/μ becomes **μ₈ = −(1/2) m_s²/μ** here. At m_s = 300, μ = 450 MeV: −57.7 MeV in their convention, **−100.0 MeV** in ours.
- Our solved 2SC point has μ₈ = −2.46 MeV; the same physical state is **μ₈ = −2.84 MeV** in the Rüster/PSB/Kunkel convention.

### 1.2 Two naming traps in the source literature

- **SRP's `G_D` is the 't Hooft coupling, not the diquark coupling.** Their
  diquark coupling is `G_DIQ`.
- **`Δ₂` means the *ud* gap** in the Rüster/Buballa index convention (η = 3 pairs
  *u*–*d*), not the second-largest gap.

---

## 2. Lagrangian and what is actually kept

```
L = q̄(i∂̸ − m̂)q
  + G_S Σ_{a=0}^{8} [ (q̄ τ_a q)² + (q̄ i γ₅ τ_a q)² ]                    scalar / pseudoscalar
  − K { det_f [ q̄(1+γ₅)q ] + det_f [ q̄(1−γ₅)q ] }                       't Hooft determinant
  + G_D Σ_{η} [ (q̄ i γ₅ ε_η λ_η^A C q̄ᵗ)(qᵗ C i γ₅ ε_η λ_η^A q) ]        diquark, colour 3̄, J^P = 0⁺
  − G_V (q̄ γ^μ q)²                                                       vector (§11)
```

Mean-field masses **including the determinant cross-terms** — these are the
coefficients that get dropped or mis-signed most often:

```
M_u = m_u − 4 G_S φ_u + 2 K φ_d φ_s
M_d = m_d − 4 G_S φ_d + 2 K φ_u φ_s
M_s = m_s − 4 G_S φ_s + 2 K φ_u φ_d
```

with φ_f = ⟨q̄_f q_f⟩ (negative), and per flavour summed over 3 colours × 2 spins.

### 2.0 Δ is an auxiliary field for the diquark bilinear, exactly as σ is for q̄q

All three mean fields come from **one** Hubbard–Stratonovich step applied to the
four-fermion terms — none is more fundamental than the others. Ivanytskyi writes
the resulting Euler–Lagrange equations side by side (his Eqs. VII–IX), which is
the clearest statement of the parallel in the literature:

```
σ_a  = −2 G_S s_a          s_a  = q̄ τ_a q                (quark–antiquark)
ω_μ  = −2 G_V j_μ          j_μ  = q̄ γ_μ q                (vector)
Δ_ab = −6 G_D d_ab         d_ab = qᵗ C γ₅ ε q            (diquark)
```

(His factor 6 rather than 2 follows from the conventional factor 3 on his diquark
term, §12.4; in this document's normalisation the pairing cost is Δ²/(4G_D).)
Each field is proportional to the bilinear it decouples, carries its own
quadratic cost in Ω, and is fixed by stationarity — so the gap equation
∂Ω/∂Δ_η = 0 and the mass equation ∂Ω/∂M_f = 0 are the same kind of statement.

**Where the analogy breaks — all four cost you if carried too far:**

1. **The condensation-energy normalisations differ by a factor 2.** In the scalar sector 2G_S Σφ_f² = Σ(M_f − m_f)²/(8G_S) (exact at K = 0, verified to 3×10⁻¹⁴); the pairing cost is Σ_η Δ_η²/(4G_D). The 1/(8G_S) versus 1/(4G_D) is channel and Fierz counting, not a typo. **Consequence: η_D = G_D/G_S = 1 does not mean "equally strong channels."**
2. **σ enters the BdG diagonal; Δ is strictly off-diagonal.** The mass shifts quasiparticle energies, the gap *mixes particles with holes* — which is why pairing requires the doubled 18×18 basis (§5.3) and the mass alone does not. The gap matrix has identically zero diagonal.
3. **[G, M] ≠ 0 at unequal masses** (§5.3). A mass matrix is always simultaneously diagonalisable with the kinetic term, so the σ sector always has a closed-form dispersion; the Δ sector does not for a general pairing pattern.
4. **Δ vanishes identically in vacuum; σ does not** (M_u − m_u = 362.15 MeV at the RKH point). Δ is complex, colour-antitriplet, B = 2/3 — it breaks U(1)_B and colour, so it is a superfluid order parameter and, being gauge-variant, not a gauge-invariant order parameter at all. σ is real, colour-singlet, and breaks only chiral symmetry.

The practical payoff is point 2: **the gap kernel carries branch signs where the
mass gap equation does not.** Writing it as Δ/|E| by analogy with the mass
equation is wrong by a factor 12 in the gapless window — see §5.6.

### 2.1 The determinant–diquark cross-term: omitted by most, but Baym writes it

The 't Hooft term, expanded in the presence of diquark condensates, generates a
term coupling |Δ_η|² to the quark–antiquark condensates — structurally
∝ Σ_α σ_α |Δ_α|².

**Most sources omit it.** Rüster *et al.* state it explicitly — that the
interaction "gives also rise to mixed contributions containing both diquark and
quark-antiquark condensates", and that they neglect such terms for simplicity;
SRP absorb it into their diquark coupling; Kunkel *et al.* do not include it.

**Baym *et al.* do include it**, with a coupling K′ and the expectation
K′ ≃ K from the Fierz transformation connecting the corresponding vertices:

```
L⁽⁶⁾_σd = K′ ( tr[(d_R† d_L) φ] + h.c. ) ,          K′ > 0

M_i = m_i − 4G σ_i + K |ε_ijk| σ_j σ_k + (K′/4) |d_i|²
Δ_k = −2 d_k ( H − (K′/4) σ_i )
Ω_cond = Σ_i [ 2G σ_i² + (H − (K′/2) σ_i) |d_i|² ] − 4K σ₁σ₂σ₃ − g_V n_q²
```

(their G = our G_S, their H = our G_D.) The structure is
**stationarity-consistent**: treating Ω_single's mean-field derivatives as
∂Ω/∂M_j = σ_j and ∂Ω/∂Δ_k = d_k, all six residuals of the σ- and d-sector
stationarity conditions vanish identically. The apparent mismatch — K′/2 in
Ω_cond versus K′/4 in both M_i and Δ_k — is **required**, not a typo: the |d|²
term appears once in Ω_cond but is differentiated in two channels.

So a coefficient *is* available if you want the cross-term: adopt Baym's K′ with
K′ ≃ K, and note that it raises M_i and *reduces* the effective pairing strength
H → H − (K′/4)σ_i, with σ_i negative in the broken phase — i.e. it encourages
coexistence of the chiral and diquark condensates.

**This document does not implement it.** With it omitted, **η_D = G_D/G_S must be
read as an effective coupling that has absorbed it**, and any paper should say so.
Do not invent a coefficient by analogy with the mass cross-terms — use Baym's.

### 2.2 Sign conventions in the sources are not uniform

Two hazards, both verified rather than taken on trust:

- **SRP eq. (6) carries the opposite sign** on the condensate terms to
  Buballa/Rüster/Blaschke, and to their own eq. (7). Their fermionic term is
  fine — their 72-eigenvalue form with prefactor −½ and integrand
  λ/2 + T ln(1+e^{−λ/T}) is *exactly* equal to Rüster's 18-eigenvalue form,
  because Σλ_i/2 cancels over the ± pairs (verified: Σw = −2.3×10⁻¹³, and
  Σ|w|/2 equals the sum of the 9 non-negative branches to 2×10⁻¹⁶).
- **Buballa uses two opposite vector-sign conventions in the same review**
  (§2.3: μ̃ = μ − 2G_V n with −(μ−μ̃)²/4G_V; §4.4: μ̃ = μ + 2G_v n with
  +(μ̃−μ)²/4G_v). Each is internally consistent; mixing them is silent.

### 2.3 The field-energy identity holds only at K = 0

A convenient check appears in the literature:
2 G_S Σ_f φ_f² = Σ_f (M_f − m_f)²/(8 G_S). **It is exact only when K = 0**
(verified to 4×10⁻¹⁶ at K = 0), and fails by 34% with the determinant term on —
the ratio is 0.6616 for the RKH set. Use it as a unit test of the scalar sector
with K switched off, never as a check on the full model.

---

## 3. Base medium integrals

One mode, g = 2, in `njl_core.py`. All five quantities come from one quadrature pass:

```
N  = (g/2π²) ∫ dk k²        (f⁺ − f⁻)          number density
Rs = (g/2π²) ∫ dk k² (M/E)  (f⁺ + f⁻)          scalar density
E  = (g/2π²) ∫ dk k²  E     (f⁺ + f⁻)          energy density
S  = (g/2π²) ∫ dk k² [entropy per mode]
P  = (g/2π²) ∫ dk k²  T[ln(1+e^{−(E−μ)/T}) + ln(1+e^{−(E+μ)/T})]
```

Antiparticles **subtract** in N and **add** in Rs, E, P.

Verified identities (all four, at five (M, μ, T, cutoff) combinations spanning
T = 0.5–50 MeV, cut and uncut): N = ∂P/∂μ to 3×10⁻⁹, Rs = −∂P/∂M to 9×10⁻¹⁰,
S = ∂P/∂T to 1×10⁻⁸, and Euler ε = −P + μN + TS to machine precision.

### 3.1 Quadrature: panels, not more nodes

Gauss–Legendre on one panel over [0, Λ] cannot resolve the Fermi step. Build
breakpoints at k_F − 25T, **k_F**, k_F + 25T and integrate each panel separately.

**Impose the cutoff as the panel upper limit *before* building panels.** Filtering
breakpoints out afterwards can delete the Fermi-surface break and silently revert
to single-panel accuracy — this produced 10⁻¹ errors at T = 30 MeV during
development.

### 3.2 The cutoff surface term — P must come from the logarithm form

The two standard pressure forms,

```
P_log = (g/2π²) ∫₀^Λ dk k² T[ln(1+e^{−(E−μ)/T}) + ln(1+e^{−(E+μ)/T})]
P_k⁴  = (g/6π²) ∫₀^Λ dk k⁴/E (f⁺ + f⁻)
```

are equal only up to the surface term generated by integrating by parts, which
**does not vanish when the integral is cut**:

```
P_log − P_k⁴ = (g/6π²) Λ³ · T[ln(1+e^{−(E_Λ−μ)/T}) + ln(1+e^{−(E_Λ+μ)/T})]
```

Verified to 6×10⁻¹⁴ against that closed form. It is not small:

| M | μ | T | P_log | surface term | fraction of P |
|---|---|---|---|---|---|
| 100 | 500 | 20 | 4.846×10⁸ | 5.858×10⁵ | 0.1% |
| 40 | 590 | 30 | 1.041×10⁹ | 1.088×10⁸ | 10.5% |
| 140 | 700 | 5 | 1.661×10⁹ | 6.025×10⁸ | 36.3% |
| 140 | 700 | 50 | 1.676×10⁹ | 6.683×10⁸ | 39.9% |

**Use `P_log` whenever the medium integral is cut.** At T = 0 with k_F < Λ the
two agree because the integrand vanishes at the upper limit; that is exactly why
the error hides until finite T. `njl_core.integrals` returns `P` (log form),
plus `P_k4` and `P_surface` as diagnostics.

---

## 4. Vacuum and parameters

RKH set (Rehberg–Klevansky–Hüfner, PRC 53 (1996) 410) — the standard
three-flavour NJL fit, used throughout:

| parameter | value |
|---|---|
| Λ | 602.3 MeV |
| G_S Λ² | 1.835 |
| K Λ⁵ | 12.36 |
| m_u = m_d | 5.5 MeV |
| m_s | 140.7 MeV |

Vacuum outputs, all reproduced from the gap equation with no fitting:

| quantity | computed | published |
|---|---|---|
| M_u | 367.648 MeV | 367.7 |
| M_s | 549.479 MeV | 549.5 |
| (−φ_u)^{1/3} | 241.946 MeV | 241.9 |
| (−φ_s)^{1/3} | 257.688 MeV | 257.7 |
| f_π (quark loop) | 92.391 MeV | 92.4 |

f_π uses f_π² = 2 M² I₂(M, Λ) with I₂ = (N_c/4π²)[arcsinh(Λ/M) − Λ/√(Λ²+M²)].
It is the **vacuum-fit diagnostic** used in §12 to test the dielectric grafts.

**Effective bag constant.** Ω at fixed M evaluated at the current masses minus at
the broken-phase masses:

```
B_eff = (228.93 MeV)⁴ = 357.49 MeV/fm³
```

Reported because the CCDM document quotes B_g + B_χ, and this is the NJL
counterpart — but note it is a *derived* quantity here, not an input.

### 4.1 Solver: damped fixed point on the masses, not a root-finder on φ

`fsolve` on the condensates diverged during development, returning masses that
*increase* with density. Iterate instead:

```
M ← M + λ (M_new[φ(M)] − M) ,  λ ≈ 0.3
```

converging on (M_f, n_q) jointly. Robust across the whole μ_B range at every
graft and vector variant tested.

---

## 5. The pairing sector

### 5.1 The gap matrix, and why multiplicities must be derived

```
G_{(f a),(g b)} = Σ_η Δ_η ε^{a b η} ε_{f g η}
```

Its eigenvalue multiplicities are a **derived** property — never assign them by
hand:

| pattern | Δ | spectrum of G (Δ₀ = 60) |
|---|---|---|
| unpaired | (0,0,0) | 0 (×9) |
| 2SC | (0,0,Δ) | −60 (×2), 0 (×5), +60 (×2) |
| CFL | (Δ,Δ,Δ) | −60 (×5), +60 (×3), +120 (×1) |
| uSC | (0,Δ,Δ) | ±84.85 (×1 each), ±60 (×2 each), 0 (×3) |
| dSC | (Δ,0,Δ) | same as uSC |

With independent gaps the ±√2 Δ eigenvalue generalises to ±√(Δ₂²+Δ₃²): verified
at (Δ₂, Δ₃) = (40, 70) → 80.6226 MeV.

### 5.2 The commutator obstruction

At **unequal masses** the gap matrix and the mass matrix do not commute:
‖[G, M]‖_F = 7.43×10⁴ at M = (40, 45, 480), versus exactly 0 at equal masses.
**Consequence: there is no closed-form dispersion for a general pattern at
unequal masses.** Diagonalise the Bogoliubov–de Gennes matrix numerically.

### 5.3 BdG construction and structure checks

At each k, for particles and antiparticles separately:

```
ξ_j = E_{f(j)} ∓ μ*_j ,   H = [[diag(ξ), G], [G, −diag(ξ)]]   (18×18)
```

The nine non-negative eigenvalues are the quasiparticle energies. Verified
structure:

- particle–hole symmetry: max|w_i + w_{18−i}| = 4×10⁻¹⁰
- Σ w_i = −2.3×10⁻¹³ (the ±pairs cancel)
- Σ|w_i|/2 = Σ(9 non-negative branches) to 2×10⁻¹⁶
- trace identity Σw² = 2 tr ξ² + 2 tr G² to 10⁻¹³

**2SC has a closed form** (both masses may differ):

```
E^± = √( (Ē − μ̄)² + Δ² ) ± [ ½(E_d − E_u) − δμ ]
Ē = ½(E_u+E_d),  μ̄ = ½(μ_ur+μ_dg),  δμ = ½(μ_dg − μ_ur)
```

Verified against the full BdG spectrum over 200 random configurations of masses,
chemical potentials, gap and momentum: **max deviation 4.5×10⁻¹³ MeV**. Use it as
a unit test of the general path, and as a fast path for 2SC production runs.

### 5.4 δΩ_pair: use the correction form

Write the pairing contribution as a **difference** from the unpaired spectrum:

```
δΩ_pair = −(1/2π²) Σ_{r=±} ∫ dk k² Σ_{a=1..9} [ φ(E_a) − φ(|ξ_a|) ]
φ(x) = x                                     at T = 0
φ(x) = x + 2T ln(1 + e^{−x/T})               at T > 0
```

This **vanishes identically at Δ = 0** — verified as exactly 0.0 in both the BdG
and 2SC-closed-form paths, not merely small. That property is what makes the
unpaired phase a clean limit of the same code, and it is worth an assertion in
production.

BdG and closed-form agree to 10⁻¹² for the 2SC pattern at Δ = 20, 60, 100 MeV.

**BCS scaling** in the clean limit: −δΩ_pair / [μ²Δ²(ln(2Λ/Δ) − ½)] → 2/π².
Verified: the ratio × π² is 1.9955, 1.9948, 1.9941, 1.9933 at Δ = 2, 5, 10, 20 MeV
(→ 2).

**Antiparticle branches are not optional.** At T = 0 they do not vanish and they
grow with the cutoff: at M* = 60, μ* = 450, Δ = 60 MeV they contribute 8.8% of
the particle piece at Λ = 600 MeV and **17.1% at Λ = 1000 MeV**.

### 5.5 Pairing quadrature: split at every mode Fermi momentum

The |ξ_j| subtraction **kinks at each of the nine k_F,j = √(μ*_j² − M_f²)**. One
panel cannot resolve nine kinks. Splitting there is dramatically better than
adding nodes:

| scheme | nodes/panel | panels | total nodes | rel. error |
|---|---|---|---|---|
| single panel | 100 | 1 | 100 | 1.1×10⁻⁴ |
| single panel | 800 | 1 | 800 | 2.4×10⁻⁷ |
| **split at k_F,j** | **100** | **7** | **700** | **3.2×10⁻¹⁴** |

At T > 0 add the thermal collar k_F,j ± 25T as well.

### 5.6 The gap equation kernel — the sign structure matters

The gap equation is ∂Ω/∂Δ_η = 0. The kernel is **not** Δ/|E|:

```
WRONG:    ∂|E^e|/∂Δ = Δ / |E^e|
RIGHT:    ∂|E^e|/∂Δ = sign(E^e) · Δ / base ,   base = √((Ē−μ̄)² + Δ²)
```

They agree when every branch is positive (verified to 0.0 at zero mismatch). With
a mismatch they do not, because in the **gapless window** E^− < 0 and the two
branches *cancel* — the BCS blocking region. Against finite differences of
δΩ_pair at μ_u = 400, μ_d = 500 MeV:

| Δ | finite difference | Δ/\|E\| form | error | sign form | error |
|---|---|---|---|---|---|
| 40 | −5.602×10⁶ | −7.268×10⁷ | **×12.0** | −5.601×10⁶ | 2×10⁻⁴ |
| 60 | −9.889×10⁶ | −1.705×10⁷ | ×1.7 | −9.889×10⁶ | 2×10⁻⁹ |
| 80 | −1.142×10⁷ | −1.475×10⁷ | ×1.3 | −1.142×10⁷ | 1×10⁻⁹ |

The wrong form makes 1/|E^−| blow up, and the gap *grows* spuriously with the
mismatch. This is the single most consequential coding error in this document.

For the general pattern use **Hellmann–Feynman** on the BdG matrix rather than
finite differences:

```
∂E_a/∂Δ_η = v_a† ∂H/∂Δ_η v_a ,   ∂H/∂Δ_η = [[0, ∂G/∂Δ_η], [∂G/∂Δ_η, 0]]
```

Verified against finite differences over all nine branches: max deviation
8.6×10⁻¹⁰. The same construction gives n_j and ρ_s,f analytically (verified to
6×10⁻⁸), which is what makes the neutral solve tractable — a fully
finite-differenced version was both ~40× slower and ill-conditioned enough to
fail convergence.

### 5.7 The gap equation has three roots — do not use a fixed bracket

With a mismatch, R(Δ) = Δ/(2G_D) − kernel has **three** roots: Δ = 0, a barrier
maximum, and the physical BCS root. `brentq` on a fixed bracket silently returns
the wrong one or fails. **Scan Δ, then bracket each sign change.** At μ* = 450,
η_D = 0.75:

| δμ [MeV] | roots of R(Δ) [MeV] | Ω at BCS root |
|---|---|---|
| 0 | 92.71 | −1.665×10⁸ |
| 50 | 32.37, 92.71 | −6.368×10⁷ |
| 60 | 52.48, 92.71 | −1.832×10⁷ |
| 65 | 60.11, 92.71 | +7.496×10⁶ |

**Clogston–Chandrasekhar limit** recovered: Δ₀ = 92.708 MeV predicts
δμ_c = Δ₀/√2 = 65.554 MeV; the measured free-energy crossover is **63.589 MeV**,
a ratio of 0.970. The 3% deficit is the finite cutoff — the analytic result is a
weak-coupling statement.

### 5.8 Paired modes: densities and entropy need the derivative, not the formula

Once Δ ≠ 0, `n_j` is **not** the unpaired Fermi integral. Take
n_j = −∂Ω/∂μ_j. At μ_B = 1400, T = 20, Δ₃ = 80 MeV the unpaired formula is wrong
in **both directions**:

| mode | unpaired formula | −∂Ω/∂μ_j | error if formula used |
|---|---|---|---|
| (u,r), (u,g) | 2.494×10⁶ | 3.163×10⁶ | **−21.1%** |
| (d,r), (d,g) | 3.710×10⁶ | 3.325×10⁶ | **+11.6%** |
| all six others | — | — | 0.00% |

Entropy is worse. Paired/unpaired entropy ratio at M* = 60, μ* = 450, Δ = 60:

| T [MeV] | s_paired / s_unpaired |
|---|---|
| 5 | 2.26×10⁻⁴ |
| 10 | 3.71×10⁻² |
| 20 | 3.28×10⁻¹ |
| 50 | 8.03×10⁻¹ |

**Four orders of magnitude at T = 5 MeV.** A merger EoS that uses unpaired
entropy in a gapped phase is not approximately wrong, it is qualitatively wrong.

### 5.9 Finite T is non-analytic at T = 0

In a fully gapped phase *every* Taylor coefficient of the paired contribution
vanishes at T = 0 (the behaviour is e^{−Δ/T}). A Taylor-expansion branch in T
must be **bypassed** for gapped phases, not corrected. The closed asymptotic form
P_th ∝ μ̄²T²√(Δ/T) e^{−Δ/T} is valid only while Δ ≥ δμ/2, i.e. not in a gapless
window.

---

## 6. Neutrality and pattern selection

Solve simultaneously: three gap equations for M_f, the free gaps Δ_η, and

```
n_C − n_l = 0      electric neutrality (leptons: e, μ with g = 2)
n_3 = 0 ,  n_8 = 0  colour neutrality
n_3 = Σ_i (n_{i,r} − n_{i,g}) ,   n_8 = Σ_i (n_{i,r} + n_{i,g} − 2 n_{i,b})
```

**In the unpaired phase n_8 vanishes identically at μ₈ = 0 and μ₈ is
unconstrained** — verified: n₃ responds only to μ₃, n₈ only to μ₈, and both are
exactly 0 at μ₃ = μ₈ = 0. Do not let the solver hunt for μ₈ in an unpaired
region; fix it to zero.

Solved neutral points at μ_B = 1500 MeV, T = 0, η_D = 0.75, no vector coupling:

| pattern | M (u,d,s) [MeV] | μ_C | μ₃ | μ₈ | Δ₃ | n_B [fm⁻³] | P [MeV/fm³] | residual | Euler |
|---|---|---|---|---|---|---|---|---|---|
| unpaired | (9.84, 8.55, 265.59) | −34.20 | 0.000 | 0.00 | — | 1.4319 | 302.12 | 1×10⁻¹³ | 1×10⁻¹⁶ |
| 2SC | (11.96, 7.65, 243.13) | −62.27 | −0.000 | −2.46 | 95.50 | 1.4887 | 324.75 | 2×10⁻¹² | 2×10⁻¹⁶ |

μ₃ = 0 in 2SC as it must be (the pattern is symmetric under r ↔ g), and 2SC
lowers Ω — the pressure rises from 302 to 325 MeV/fm³.

### 6.1 Assembly of Ω and ε — the audit that catches everything

```
C     = 2 G_S Σ_f φ_f² − 4 K φ_u φ_d φ_s                     condensate cost
D     = Σ_η Δ_η² / (4 G_D)                                   pairing cost
W     = G_V(n_q) n_q²      Σ_V = dW/dn_q                     vector (§11)

Ω   = −Σ_j P_med,j − Σ_f ε_sea,f + C − (Σ_V n_q − W) − P_lep + δΩ_pair + D
ε   =  Σ_j ε_med,j − Σ_f ε_sea,f + C + W + ε_lep + ε_pair
ε_pair = δΩ_pair + D + T s_pair + Σ_j μ*_j (δn_pair)_j
```

Both carry the **same** vacuum constant, so Ω_vac = ε_vac exactly and Euler holds
after subtracting it from each. Three assembly bugs were caught this way during
development, each of which produced a plausible EoS:

1. a sign error in ε (Euler off by O(1));
2. the pairing cost and δΩ_pair omitted from both Ω and ε (Euler off by 7.7×10⁻³ — small enough to look like quadrature);
3. `s_pair` omitted (fails only at T > 0).

**Audit at every solved point**: n_B = −∂Ω/∂μ_B by finite difference, and
ε = −P + Ts + Σ_i μ_i n_i. Both hold to machine precision above.

### 6.2 A solver trap in the vector sector

For G_V > 0 the stationary point in μ̃ is a **maximum** of Ω, not a minimum. δΩ/δμ̃ = 0
is a constraint to **root-find**; a joint minimiser over (σ_f, Δ_η, ω₀) will diverge.

### 6.3 CFL closure differs

CFL is electrically neutral **without electrons** (equal flavour densities), so
seed μ_C near 0 and disable leptons for that branch. With the electron-bearing
seed the solve converges to a spurious point (residual 13, flavour-density spread
11%). This branch is reported as not yet tightly converged — it needs the
Ginzburg–Landau-informed seed, and that is flagged rather than papered over.

---

## 7. Regularization — the sharp cutoff is not safe here

**The medium integral is not a spectator.** At T = 0, unpaired, it is
self-limiting at k_F and cutoff-free while k_F < Λ. That protection disappears at
finite T and in **any** CSC phase, where the Fermi surface is smeared.

Two prescriptions were tested, and both fail at high density in specific ways:

- **cut everything at Λ**: the density saturates at n_B = 3Λ³/(3π²) = 2.881 fm⁻³ and freezes;
- **cut only the Dirac sea**: no saturation, but the medium integral is then unregularized where the model has no content.

### 7.1 The sharp cutoff destroys the gap

With η_D = 1, M* = 5.5 MeV, Λ = 602.3 MeV:

| μ* [MeV] | μ*/Λ | Δ [MeV] |
|---|---|---|
| 300 | 0.498 | 109.65 |
| 500 | 0.830 | 149.45 |
| 600 | 0.996 | 119.12 |
| 650 | 1.079 | 75.88 |
| **680** | **1.129** | **0.00 — gap gone** |

This reproduces the published failure point (μ ≈ 680 MeV = 1.13 Λ) exactly.
Beyond it the loop contributions are cut off entirely and the code silently
returns a **free gas**. Published convergence studies find artifacts already for
1 < λ < 5 (λ = Λ_UV/Λ), i.e. well below the vacuum cutoff, with RG-consistent CFL
gaps almost 90% higher than at λ = 1, and pressure underestimated by ~10% at
μ/Λ = ½ and ~30% at μ/Λ = 1.

**Practical verdict**: a sharp-cutoff three-flavour CSC calculation is
quantitatively safe only for μ ≪ Λ/2, which is *below* deconfinement onset —
there is effectively no window where it is trustworthy. Use it for code
validation against published sharp-cutoff results, not for production.

### 7.2 The RG-consistent fix, and the divergence it removes

Integrate the medium integral to a large scale Λ_UV ≫ Λ, keep the divergent
vacuum integral at Λ, and subtract a counterterm cancelling the medium
divergence. That divergence is **logarithmic**, exists only when μ ≠ 0 **and**
Δ ≠ 0, and does not scale with quark masses:

```
Γ_med / V₄  ≃  −(2/π²) μ̄² Δ² ln Λ_UV
```

Verified by differentiating the vacuum-subtracted δΩ_pair with respect to ln Λ_UV
at μ = 450, Δ = 60 MeV:

| Λ_UV [MeV] | d Γ_med / d ln Λ | −(2/π²) μ²Δ² | ratio |
|---|---|---|---|
| 2000 | −1.5539×10⁸ | −1.4773×10⁸ | 1.0519 |
| 4000 | −1.4957×10⁸ | −1.4773×10⁸ | 1.0125 |
| 8000 | −1.4822×10⁸ | −1.4773×10⁸ | 1.0034 |
| 16000 | −1.4784×10⁸ | −1.4773×10⁸ | 1.0008 |

The coefficient is confirmed to 0.08% and scales correctly in both μ and Δ
(−0.2033, −0.2038, −0.2033 versus −2/π² = −0.2026 at three (μ, Δ) points).
**Subtract the vacuum piece before testing this** — the total δΩ_pair carries the
vacuum divergence too and the ratio then runs away (20.8, 80.0, 317.1).

Implement λ = Λ_UV/Λ as a parameter: λ = 1 reduces exactly to conventional cutoff
regularization; λ ≈ 10 is Λ-independent to <1%. Of the three published
counterterm schemes, the **massive** one is rejected by its own authors (it
inverts the gap ordering to Δ₃ < Δ₁ = Δ₂ and predicts the wrong melting pattern);
massless and minimal both have closed forms.

---

## 8. Assembly order

1. Fix the parameter set; solve the vacuum gap equation; store Ω_vac and ε_vac (they must agree — assert it).
2. Choose pattern → which Δ_η are free.
3. At each (μ_B, T): damped fixed point on (M_f, n_q), inner Newton/scan on Δ_η, outer solve for (μ_C, μ₃, μ₈).
4. Build the pairing panels from the *current* μ*_j and M_f — they move as the solve converges.
5. Assemble Ω and ε per §6.1; subtract the vacuum constants.
6. **Audit**: n_B = −∂Ω/∂μ_B, Euler, n_C − n_l = 0, n₃ = n₈ = 0, δΩ_pair(Δ=0) = 0.
7. Compare Ω across patterns at fixed (μ_B, T, neutrality) to select the phase.

Assertions worth keeping in production: δΩ_pair vanishes at Δ = 0; Σw = 0 for the
BdG matrix; Euler to 10⁻⁸; and every gap root re-bracketed by scan rather than
inherited.

---

## 9. Question 1 — should G_V depend on density to restore c_s² → 1/3?

**Yes, and the required scaling is fixed, not free.**

### 9.1 Why constant G_V fails

With chiral symmetry restored the scalar channel dies (M → m → 0) and the
asymptotics is set entirely by the vector term. The vector density is the free one
at the shifted potential, so the self-consistency relation is **cubic**:

```
μ = μ' + (2 G_V N_c N_f / 3π²) μ'³        ⟹     μ' ∝ G_V^{−1/3} μ^{1/3}
```

The shifted potential grows only as the cube root of the bare one. The interaction
energy density W = G_V n² then grows like n², faster than the kinetic n^{4/3}, and
the sound speed runs away from 1/3.

### 9.2 The exponent that fixes it

Write ε = Σ_i C_i n^{p_i}. Each term contributes P_i = C_i(p_i − 1)n^{p_i}, so

```
c_s² = Σ_i C_i p_i (p_i−1) n^{p_i−1} / Σ_i C_i p_i n^{p_i−1}
```

Free massless quarks give p = 4/3 → c_s² = 1/3. With G_V ∝ n^{−α} the vector term
has p_V = 2 − α, hence:

```
c_s²(n → ∞) = max(1 − α, 1/3)
```

Verified against the solver at μ_B = 4 GeV:

| α | c_s²(μ_B = 2 GeV) | 3 GeV | 4 GeV | asymptote |
|---|---|---|---|---|
| 0 (const G_V) | 0.3770 | 0.5704 | **0.6303** | 1 |
| 1/3 | 0.3445 | 0.3784 | 0.3908 | 2/3 |
| **2/3** | 0.3204 | 0.3325 | **0.3332** | **1/3** |
| 1 | 0.3206 | 0.3317 | 0.3326 | 1/3 |
| G_V = 0 | 0.3221 | 0.3325 | 0.3332 | 1/3 |

### 9.3 α = 2/3 is marginal and exact at every density

At α = 2/3 the interaction pressure equals exactly one third of the interaction
energy density — **not asymptotically, identically**:

```
P_int / ε_int = 1 − α = 1/3      (verified to 10⁻⁹ at every density tested)
```

So α = 2/3 is the unique choice for which the vector term is conformal on its own.
For α > 2/3 the vector term dies faster than the kinetic one and the free result
takes over; for α < 2/3 it dominates and c_s² → 1 − α > 1/3.

### 9.4 The rearrangement term is mandatory

If G_V depends on n, the vector self-energy is the **derivative of the energy
density**, not 2G_V n:

```
W(n) = G_V(n) n²      Σ_V = dW/dn = (2 − α) G_V(n) n       μ*_f = μ_f − Σ_V
Σ_V / (2 G_V n) = (2 − α)/2
```

Omitting the rearrangement piece breaks thermodynamics measurably:

| α | Σ_V correct | naive 2G_V n | ratio | P error |
|---|---|---|---|---|
| 1/3 | 36.22 MeV | 43.47 MeV | 0.8333 | **5.19%** |
| 2/3 | 12.43 MeV | 18.65 MeV | 0.6667 | **4.85%** |

With the correct Σ_V, n = dP/dμ holds to 10⁻⁸ at every density. Note the contrast
with a **field-dependent** coupling: a coupling that depends on a mean field needs
no rearrangement term, because it enters through that field's own equation of
motion. A density-dependent coupling does.

### 9.5 A physically motivated form that lands on 2/3 by itself

The gluon-exchange-motivated form (from the nonlocal-NJL literature)

```
G_V(n_q) = G_V⁰ / [ 1 + 8 k_F² / (9 M_g²) ] ,     k_F = (π² n_q / 2)^{1/3}
```

with a non-perturbative gluon mass M_g ≈ 500 MeV has effective exponent
α_eff = −d ln G_V/d ln n_q → 2/3 **without tuning**:

| n_q [MeV³] | G_V/G_S | α_eff |
|---|---|---|
| 10⁶ | 0.4533 | 0.0623 |
| 10⁸ | 0.1553 | 0.4596 |
| 10⁹ | 0.0442 | 0.6077 |
| 10¹⁰ | 0.0102 | 0.6530 |

In the solver it gives a peak c_s² = 0.362 at μ_B ≈ 1.5 GeV (n_B ≈ 0.8 fm⁻³) and
settles to 0.343 at 4 GeV — a sound-speed peak followed by conformal approach,
which is the qualitatively desired shape for a compact-star EoS. **This is the
recommended choice**: it is not a fitted interpolation, and it reaches the
marginal exponent as a consequence of its own structure.

### 9.6 What the gap contributes

Pairing does not change the asymptotics. With η_V = 0 the approach is from
**above**: c_s² → 1/3 + (2/9)Δ²/μ², which is 2.2×10⁻³ at μ = 500 MeV and
2.5×10⁻⁴ at 1.5 GeV — subleading, dying as 1/μ². With η_V ≠ 0 and constant G_V
the sound speed runs to 1 (Zel'dovich behaviour) regardless of the gap.

![Sound speed versus chemical potential for five vector-coupling choices (left), and the asymptotic value as a function of the exponent α with solver points overlaid (right)]({{artifact:78d58e48-3895-451a-a311-0832ea8f18ee}})

---

## 10. Question 2 — can the colour dielectric with dilaton potential supply confinement here?

**Yes, but only one of four graft points works, and the reason the others fail is
structural rather than numerical.**

### 10.1 There is no literature to lean on

An explicit search found **no published model coupling a colour-dielectric or
dilaton field to an NJL four-fermion interaction.** Every dielectric quark-matter
paper pairs the dielectric with a *linear sigma model* — explicit σ, π fields with
a Mexican-hat potential — never with a contact four-quark term whose coupling is
fixed by a vacuum gap equation. Searches for "dielectric NJL", "dilaton NJL quark
matter", "Gribov–Zwanziger quark matter", "confining NJL scalar field" returned
either nothing or papers that use CDM and NJL as two *separate* alternative
equations of state and compare them.

The nearest existing neighbours are: an infrared-regulator route (confinement in
the regulator, so **no order parameter** and no first-order transition), a
density-dependent-cutoff route, and confining density-functional models with
density-dependent couplings. **This graft is new work**, and the document should
say so.

### 10.2 The structural tension

In the CCDM the constituent mass is M* = (g_q σ + m_f)/χ with the σ sector fixed
by f_π, and χ → 0 in the confining vacuum drives M* → ∞. In NJL the constituent
mass comes from the **self-consistent gap equation**, with G_S and Λ fixed by
vacuum observables. The dielectric convention places χ = 1 at the *perturbative*
point — which is precisely where the NJL vacuum data would have to be fitted, and
the confining vacuum sits at χ → 0.

Worse: the manoeuvre that makes the dielectric harmless in the CCDM — discarding
the sea term — is **unavailable in NJL, where the sea *is* the condensate**.

### 10.2a The full construction: dilaton term plus refitted parameters

The obvious objection to §10.3 below is that those grafts hold G_S, K and Λ at
their RKH values. The honest version of the proposal is: **add the dilaton term to
the Lagrangian, divide the whole mass operator by χ^p, solve χ's own field
equation, and refit every parameter** — accepting that new ones (at minimum B_g)
are needed. That construction was built and tested. Two results, one encouraging
and one fatal.

**The refit succeeds, and is in fact exactly degenerate.** Using the CCDM's own
forms — Φ ≡ φ̄⁴ = ⟨G²⟩_med/⟨G²⟩_vac, χ = (1−Φ)^p, U = B_g[Φ(lnΦ−1)+1] — and
holding the *dressed* masses M*_u = 367.6481, M*_s = 549.4794 MeV at their
physical values, the gap equations are **linear** in (G_S, K) and solve exactly:

| χ_vac | G_SΛ² | KΛ⁵ | M_u (field) | M*_u (dressed) | f_π | (−φ_u)^⅓ |
|---|---|---|---|---|---|---|
| 1.00 | 1.8350 | 12.360 | 367.65 | 367.648 | 92.391 | 241.95 |
| 0.90 | 1.4224 | 16.886 | 330.88 | 367.648 | 92.391 | 241.95 |
| 0.80 | 1.0097 | 21.412 | 294.12 | 367.648 | 92.391 | 241.95 |
| 0.70 | 0.5971 | 25.937 | 257.35 | 367.648 | 92.391 | 241.95 |
| 0.60 | 0.1844 | 30.463 | 220.59 | 367.648 | 92.391 | 241.95 |
| 0.55 | **−0.0219** | 32.726 | 202.21 | 367.648 | 92.391 | 241.95 |

f_π and the condensate are **invariant to 0.00×10⁰** across the whole column — χ_vac
is simply absorbed into (G_S, K). So the vacuum fit is *never* the obstruction, and
§10.3's "fit destroyed" verdicts describe only the fixed-coupling case.

Two caveats on the refit itself. It requires **χ_vac > 0.5553**; below that the
physical M* demands G_S < 0, i.e. no chiral symmetry breaking and no NJL mechanism
left. And (f_π, condensate) alone admit **two** roots, (M*, Λ) = (368.00, 602.18)
and (489.16, 570.13) MeV — the pair does not pin the constituent mass; m_π and m_K
break that degeneracy in the RKH fit.

**The fatal problem is the Dirac sea, and it is a divergence.** With χ^{−p}
multiplying the mass *inside* the sea, the dilaton field equation reads

```
∂Ω/∂Φ = U′(Φ) + Σ_f ρ_sea,f · ∂M*_f/∂Φ = B_g lnΦ + S/(1−Φ) ,   S = 1.9816×10¹⁰ MeV⁵
```

Near the confining vacuum Φ = 1 − ε, the restoring term U′ → 0 **linearly** in ε
while the sea term diverges as S/ε:

| ε = 1−Φ | \|U′\| | S/ε | ratio |
|---|---|---|---|
| 10⁻¹ | 4.1156×10⁸ | 1.9816×10¹¹ | 2.077×10⁻³ |
| 10⁻² | 3.9259×10⁷ | 1.9816×10¹² | 1.981×10⁻⁵ |
| 10⁻³ | 3.9082×10⁶ | 1.9816×10¹³ | 1.972×10⁻⁷ |
| 10⁻⁴ | 3.9064×10⁵ | 1.9816×10¹⁴ | 1.971×10⁻⁹ |

The ratio scales as B_g ε²/S → 0, so ∂Ω/∂Φ stays **positive** for every ε: the
equation is pushed *away* from χ = 0 and **has no confining root**. Balancing it at
ε = 10⁻³ would need B_g^{1/4} = 11 865 MeV — a ~12 GeV gluon-condensate scale
against a physical (250–400 MeV). **This is not a parameter problem. It is a
divergence, and no refit or additional parameter cures it**: driving M* → ∞ over
the filled sea costs infinite energy, whereas in the CCDM the sea is absent.

### 10.2b What does work: keep χ out of the sea

Dress the medium term only. Then at μ = 0 the medium contribution vanishes, the
vacuum dilaton equation is just U′(Φ) = B_g lnΦ = 0, and its root is **Φ = 1,
χ = 0 — the CCDM confining vacuum, recovered exactly.** At finite density the
medium term competes with U′ and a deconfined root appears:

| μ_B [MeV] | Φ | χ | M*_u [MeV] |
|---|---|---|---|
| 2300 | 0.5015 | 0.4985 | 737.5 |
| 2500 | 0.5472 | 0.4528 | 812.0 |
| 2800 | 0.5999 | 0.4001 | 919.0 |

with the onset of a χ > 0 root at **μ_B = 1824.4 MeV (μ_q = 608.1 MeV)** for
B_g = (250 MeV)⁴. That onset scales with B_g and is therefore tunable — the extra
parameter anticipated — but **the structure is not negotiable: χ must not multiply
the sea.** This is the same conclusion Graft D reaches below, arrived at from the
field-equation side rather than the fixed-coupling side, and it is why Graft D is
the recommendation.

### 10.3 The four graft points at fixed couplings

**Graft A — dielectric on the mass** (M_f → M_f/χ^p, gap equation fed by the
dressed mass). The condensate blows up and f_π is destroyed:

| χ | M*_u [MeV] | f_π [MeV] | f_π/92.4 |
|---|---|---|---|
| 1.0 | 367.6 | 92.39 | 1.00 |
| 0.8 | 577.4 | 98.03 | 1.06 |
| 0.5 | 1105.4 | 88.86 | 0.96 |
| 0.1 | 6041.9 | 42.61 | 0.46 |

**Graft B — dielectric on the coupling** (G_S → G_S/χ, K → K/χ^{2.5}), vacuum
re-solved. Diverges in the confining vacuum — exactly the wrong direction:

| χ | M_u [MeV] | M_s [MeV] | (−φ_u)^{1/3} | M_u/M_u(ref) |
|---|---|---|---|---|
| 1.0 | 367.65 | 549.48 | 241.95 | 1.000 |
| 0.8 | 657.09 | 808.55 | 263.08 | 1.787 |
| 0.5 | 1676.05 | 1811.56 | 277.35 | 4.559 |
| 0.3 | 4560.38 | 4695.11 | 280.30 | 12.404 |

**Graft C — dielectric on the cutoff** (Λ → Λχ, G_S fixed). Collapses chiral
symmetry breaking almost immediately: a 10% reduction in χ takes M_u from 368 to
129 MeV, and by χ = 0.8 the vacuum is essentially restored (M_u = 25.6 MeV,
condensate down from 242 to 96 MeV).

| χ | Λ_eff | M_u [MeV] | M_s [MeV] | (−φ_u)^{1/3} |
|---|---|---|---|---|
| 1.0 | 602.3 | 367.65 | 549.48 | 241.95 |
| 0.9 | 542.1 | 129.32 | 371.40 | 173.62 |
| 0.8 | 481.8 | 25.62 | 285.13 | 96.41 |
| 0.6 | 361.4 | 9.46 | 199.04 | 57.20 |

**Graft D — dielectric on the medium term only.** The vacuum has no medium term,
so the fit is **exact at every χ by construction** — verified: M_u unchanged to
5×10⁻¹⁴ at χ = 0.4. And it reproduces the CCDM pinning mechanism, at μ_B = 1300 MeV:

| χ | M_u(sea) | M_u(medium) | n_B [fm⁻³] | quarks? |
|---|---|---|---|---|
| 1.00 | 16.71 | 16.7 | 0.7138 | yes |
| 0.90 | 11.06 | 12.3 | 0.7145 | yes |
| 0.80 | 367.65 | 459.6 | 0.0000 | **NO — pinned** |
| 0.60 | 367.65 | 612.7 | 0.0000 | **NO — pinned** |
| 0.40 | 367.65 | 919.1 | 0.0000 | **NO — pinned** |

Below a dielectric threshold the medium mass exceeds μ*, no quarks appear, and the
sea mass snaps back to its vacuum value — the confining branch. This is the CCDM
M* ≥ μ* test, recovered inside NJL without touching the vacuum fit.

### 10.4 Graft D must carry the chain rule

There are two ways to write graft D, and only one is thermodynamically consistent.
The scalar source in the gap equation is ∂Ω/∂M, and if the medium term depends on
M through M_med = M/χ^p then the chain rule contributes a factor χ^{−p}:

```
D1 (naive):        source = ρ_s,med
D2 (variational):  source = ρ_s,med / χ^p        ← correct
```

Testing n_q = −∂Ω/∂μ_B:

| variant | χ | n_B (sum) | −∂Ω/∂μ_B | rel. error |
|---|---|---|---|---|
| D1 | 1.00 | 5.4841×10⁶ | 5.4841×10⁶ | 1.4×10⁻⁷ |
| D1 | 0.95 | 5.4854×10⁶ | 5.4877×10⁶ | **4.1×10⁻⁴** |
| D1 | 0.90 | 5.4865×10⁶ | 5.4904×10⁶ | **7.0×10⁻⁴** |
| D2 | 0.95 | 5.4875×10⁶ | 5.4875×10⁶ | 1.5×10⁻⁷ |
| D2 | 0.90 | 5.4897×10⁶ | 5.4897×10⁶ | 1.5×10⁻⁷ |

D1 is wrong at every χ ≠ 1 and *looks* right at χ = 1, which is how it would
survive a naive unit test.

### 10.5 Recommended construction

```
χ(φ) = [1 − φ̄⁴]^p            (CCDM convention: χ = 1 perturbative, χ → 0 confining)
M_med,f = M_f / χ             dielectric dresses the MEDIUM mass only
gap eq. source = ρ_s,med / χ + (untouched sea term)
U(φ)                          the CCDM dilaton potential, unchanged
```

**Notation guard.** Here χ *already contains* the exponent p, exactly as in the
CCDM document. In §§10.3–10.4 the scanned variable χ was the *base* dielectric
(1 − φ̄⁴), so the `/χ^p` written there equals the `/χ` written here — do not apply
the exponent twice. An earlier draft of this section carried the doubled exponent
(`χ = [1−φ̄⁴]^p` *and* `M/χ^p`), which is harmless only at the locked baseline
p = 1.

with the dilaton treated as a dynamical field whose equation of motion follows
from ∂Ω/∂φ = 0 — which is also why a field-dependent coupling needs no
rearrangement term (§9.4).

For the **diquark** coupling the project's CCDM work already reached a criterion:
G_D → G_D/χ^q with q ∈ {0, 1} subject to q ≤ p. That carries over unchanged, since
it concerns the coupling rather than the sea. Note also the published bound on
η_D = G_D/G_S beyond which the *vacuum itself* would pair — check any chosen η_D
against it.

**Honest limitations.** The dielectric graft is verified here for (i) preservation
of the vacuum fit, (ii) thermodynamic consistency, (iii) reproduction of the
pinning mechanism. It has **not** been verified for the order of the deconfinement
transition, coexistence with pairing, or finite-T behaviour. Those are the next
tests, and this document does not claim them.

### 10.6 The dilaton–NJL model, written out for coding

Everything a code needs, collected from §§10.2a–10.5 into one place. Two variants
exist; only the second is implementable.

**Variant I — Lagrangian-level dressing (the naive construction; do not use).**
Add the dilaton to the NJL Lagrangian of §2 and divide the quark mass operator by
the dielectric function:

```
L = q̄ i∂̸ q − Σ_f (m_f/χ) q̄_f q_f
  + G_S Σ_a [ (q̄ τ_a q)² + (q̄ i γ₅ τ_a q)² ]
  − K { det_f [ q̄(1+γ₅)q ] + det_f [ q̄(1−γ₅)q ] }
  + G_D Σ_η |q̄ i γ₅ ε_η λ_η^A C q̄ᵗ|²
  − G_V (q̄ γ^μ q)²
  + ½ ∂_μ φ ∂^μ φ − U(φ)

χ(φ) = [1 − φ̄⁴]^p ,   φ̄ = φ/φ₀ ,   U(φ) = B_g [ φ̄⁴ (ln φ̄⁴ − 1) + 1 ]
```

so that the *full* constituent mass is dressed, M*_f = M_f/χ, sea included. This
is the construction "add a dilaton field and divide the quark mass terms by the
dielectric function" taken literally, and it is theoretically well-defined as a
Lagrangian — but §10.2a shows its mean-field thermodynamics is **non-confining by
a divergence**: the dilaton field equation ∂Ω/∂Φ = B_g ln Φ + S/(1−Φ) (with S the
sea source) has no root near the confining vacuum Φ → 1, because driving
M* → ∞ over the *filled Dirac sea* costs infinite energy. No refit cures it
(the refit itself works and is exactly degenerate down to χ_vac ≈ 0.5553, but the
confining root never appears). Keep this variant only as a documented dead end.

**Variant II — grand-potential-level dressing (Graft D; the model to implement).**
The dilaton and its potential enter as in Variant I, but χ dresses only the
*medium* part of every quark loop; the Dirac-sea (vacuum) part keeps the bare
M_f. Working in the solve variable Φ ≡ φ̄⁴ (§10.2a; χ = (1−Φ)^p):

```
Effective medium mass       M_med,f = M_f / χ
Dressed dispersions         E_med,f(k) = √(k² + M_med,f²)      (medium and pairing integrals)
Sea (unchanged)             ε_sea,f, ρ_s,sea,f evaluated at M_f with cutoff Λ

Condensates (per flavour)   φ_f = −3 s_sea(M_f, Λ) + R_s,med(M_med,f, μ*, T) / χ
                            [the 1/χ is the §10.4 chain rule — D2, not D1]

Mass equations              M_f = m_f − 4 G_S φ_f + 2 K φ_g φ_h      (f,g,h cyclic)

Dilaton equation            ∂Ω/∂Φ = B_g ln Φ + [p/(1−Φ)] Σ_f M_med,f ρ_s,med,f = 0
                            [ρ_s,med,f evaluated at M_med,f; verified form of §10.2a–b]

Diquark coupling            G_D → G_D/χ^q ,   q ∈ {0, 1} ,  q ≤ p    (declared discrete choice)
Pairing sector              §§5–6 unchanged, with M_med,f in place of M_f
                            and pairing cost Σ_η Δ_η² χ^q/(4 G_D)

Grand potential             Ω = Ω(§6.1) with medium pieces at M_med,f, plus U(φ) added
                            (vacuum: U = 0 at Φ = 1, so the vacuum constants are unchanged)
```

Properties verified in §10.2b/§10.3–10.4: the vacuum dilaton equation reduces to
B_g ln Φ = 0 with root Φ = 1 (χ = 0, the CCDM confining vacuum, exact); the RKH
vacuum fit is untouched at every χ; a deconfined χ > 0 root appears at finite
density with onset μ_B = 1824.4 MeV for B_g = (250 MeV)⁴ and p = 1 (the onset
scales with B_g); and thermodynamic consistency n_q = −∂Ω/∂μ_B holds only with
the chain-rule source (D2). At T = 0 apply the CCDM pinning test: if
M_med,f ≥ μ*_j the mode's medium integrals are identically zero — that *is* the
confinement mechanism.

New parameters relative to the plain NJL: **B_g** (continuous), **p** (locked at
1 as baseline), **q ∈ {0,1}** (discrete). φ₀ and m_φ do not enter the bulk EoS at
fixed B_g (they price gradients and the glueball mass; φ₀ = 4√B_g/m_φ).

---

## 11. What is verified and what is not

**Verified** (106 assertions, `verify_njl_csc.py`): the Δ-as-auxiliary-field structure and its four failure modes; base integrals and their four
derivative identities; T = 0 closed forms; Dirac-sea closed forms; the cutoff
surface term; the RKH vacuum against published values including f_π; B_eff; gap
matrix multiplicities for five patterns; the commutator obstruction; the 2SC
closed form against BdG over 200 random configurations; BdG structure identities;
δΩ_pair vanishing at Δ = 0; BCS logarithm scaling; the pairing quadrature study;
the gap-kernel sign structure; the Clogston limit; Hellmann–Feynman derivatives;
paired-mode density and entropy errors; colour-neutrality structure; generator
normalisation; the sharp-cutoff gap collapse; the medium log divergence
coefficient; the vector conformality exponent and rearrangement term; the
gluon-exchange effective exponent; and all four dielectric grafts.

**Not verified, and flagged as such**: the CFL neutral solve is not tightly
converged (residual 13) and needs a better seed; the 't Hooft–diquark cross-term
has no published coefficient and is absorbed into η_D; the dielectric graft is untested
for transition order, pairing coexistence, and finite T. Equation and section
numbers taken from arXiv preprints may differ from journal versions.

## 12. Relation to the four reference treatments

Read against the sources directly (arXiv LaTeX, not PDF text). Summary: **this
document's core is algebraically identical to Kunkel and to PSB, is a subset of
Baym, and shares almost nothing with Ivanytskyi beyond the coupling names.**

| | Baym *et al.* 2018 | Kunkel *et al.* 2026 | Pagliara–Schaffner-Bielich 2008 | Ivanytskyi 2024 |
|---|---|---|---|---|
| writes a Lagrangian | yes | yes | **no** (cites Rüster) | yes |
| locality | local | local | local | **nonlocal** (Gaussian formfactor) |
| 't Hooft determinant | yes | yes | yes | **absent** |
| σ–Δ cross-term | **yes, K′ ≃ K** | no | no | no |
| flavour masses | m_u, m_d, m_s | m_u, m_d, m_s | m_u, m_d, m_s | **degenerate m** |
| vector term | −g_V(q̄γq)² | −G_V(q̄γq)² | added, ω₀ form | −G_V j·j |
| regularization | sharp cutoff Λ | **RG-consistent** | sharp cutoff Λ | formfactor (UV-finite) |
| Ω normalization | vacuum-subtracted | — | **absolute + bag constant B** | vacuum-subtracted |
| T₈ | full Gell-Mann λ₈ | halved λ₈/2 | halved λ₈/2 | n/a (common μ) |
| **this document** | subset (no K′) | **identical core** | **identical algebra** | different model |

### 12.1 Kunkel *et al.* — the closest match

Term-for-term identical to §2: same four interaction terms with the same signs,
the same mass formula M_α = m_α − 4G_Sσ_α + 2Kσ_βσ_γ, the same gap definition
and the same (ds, us, ud) → (Δ₁, Δ₂, Δ₃) indexing, and the **identical RKH
parameter set** (Λ′ = 602.3 MeV, G_SΛ′² = 1.835, KΛ′⁵ = 12.36, m_ud = 5.5,
m_s = 140.7 MeV).

Three points of comparison, none of them a disagreement any more:

1. **RG-consistent regularization** (the "massless" scheme). IMPLEMENTED — see
   §7.2 — and now the default at λ = 10, with their couplings shipped as
   `Parameters.named("rg_njl1")`: G_D = 1.45 G_S, G_V = 0.7 G_S.
2. **Neutrinos and lepton-family numbers.** Correcting what this section used
   to say: they do NOT carry a muon lepton family. Their §II.4 includes muons
   in the neutrino-transparent EoS but "do not include muons or muon neutrinos
   in the neutrino-trapped beta-equilibrium EoS for simplicity", and they
   describe the evolution in terms of fixed n_B, Y_Le and s alone. That is
   exactly `beta_eq_neutrino_trapped(n_B, Y_Le, T)` with `muons=False`, which
   this model has; the Y_Lmu gap in `docs/DEFERRED.md` is real but is not a
   gap against this paper.
3. **T₈ differs by √3, not 2/√3** (§1.1). Gholami *et al.* Eq. 14 writes
   μ̂ = (μδ + μ_Q Q)δ_ab + [μ₃(λ₃)_ab + μ₈(λ₈)_ab]δ_αβ with the FULL Gell-Mann
   λ₈, and Kunkel *et al.* inherit it through the shared module. With our
   T₈ = λ₈/√3 that gives μ₈^ours = √3 μ₈^theirs. The halved λ₈/2 convention is
   Rüster's and Pagliara–Schaffner-Bielich's, not theirs.

### 12.2 Pagliara–Schaffner-Bielich — same algebra, different bookkeeping

They write no Lagrangian, adopting Rüster's potential and adding an isoscalar
vector term. Their T = 0 pressure is

```
p = (1/2π²) Σ_{i=1}^{18} ∫₀^Λ dk k²|ε_i| + 4K σ_uσ_dσ_s − (1/4G_D) Σ_c |Δ_c|²
    − 2G_S Σ_α σ_α² + ω₀²/(4G_V) + p_e
ω₀ = 2G_V ⟨n_u + n_d + n_s⟩ ,   μ_{u,d,s} → μ_{u,d,s} − ω₀
```

Three things verified against my assembly:

- **The vector sector is algebraically identical at constant G_V.** Their shift ω₀ = 2G_V n_q equals my Σ_V = dW/dn, and their pressure term ω₀²/(4G_V) equals my (Σ_V n − W) — both confirmed symbolically. The two forms diverge **only** when G_V depends on density, where Σ_V = (2−α)G_V n picks up exactly the rearrangement factor (2−α)/2 that §9.4 shows is mandatory. PSB's form is the α = 0 special case.
- **The sign convention matches mine term by term**: my −C expands to −2G_S Σφ² + 4K∏φ and my −D to −Σ Δ²/(4G_D), reproducing their pressure exactly. PSB therefore agree with Buballa and Rüster, and are **opposite to SRP eq. (6)** — independent confirmation of the inconsistency flagged in §2.2.
- **Their bag constant is a different quantity from mine.** They use the absolute +|ε_i| form, so their pressure is defined only up to an additive B fixed by requiring p − B = 0 at μ = 0, giving B₀ = (425.4 MeV)⁴. I reproduced that from their own stated formula B₀ = B₀^ref + Σ_f (3/π²)∫₀^Λ dk k²√(k²+m_f²) with B₀^ref = (217.6 MeV)⁴: **B₀^(1/4) = 425.444 MeV, a relative error of 1.0×10⁻⁴ (0.010%)** on the quarter-power, equivalently 4.1×10⁻⁴ (0.041%) on B₀ itself. My B_eff = (228.93 MeV)⁴ is a *difference* of two grand potentials (restored minus broken vacuum), and is an output rather than an input. **These two numbers should not be compared** — they are not the same object.

Smaller differences: their electron term p_e = μ_e⁴/(12π²) is massless and has no
muon sector (verified identical to my massless g = 2 integral; my m_e = 0.511 MeV
shifts it by 8×10⁻⁵ at μ_e = 100 MeV, and I add muons above μ_e = 105.66 MeV).
They fit the same four vacuum observables (m_π = 135.0, m_K = 497.7,
m_η′ = 957.8, f_π = 92.4 MeV). Beyond the quark sector they add a Maxwell
construction to a relativistic mean-field hadronic phase, a new prescription for
the bag constant (B_* — requiring chiral restoration to coincide with the
hadron–quark transition), TOV solutions, and the finding of two sequential
transitions, hadronic → 2SC → CFL. None of that has a counterpart here: this
document has no hadronic sector.

### 12.3 Baym *et al.* — a superset

Same local three-flavour NJL with scalar, diquark, vector and determinant terms
(their G = G_S, H = G_D, g_V = G_V; the vector sign matches mine). Their
μ̂ = μ_q − 2g_V n_q + μ₈λ₈ + μ_Q Q is my constant-G_V case with full Gell-Mann
λ₈. They use the same vacuum subtraction, Ω_q = Ω^bare(μ,T) − Ω^bare(0,0), and
note the 72×72 propagator has 4-fold degenerate eigenvalues (2 spin × 2
Nambu–Gorkov) — the 18 distinct branches I diagonalise.

**The one term they have that I do not is the σ–Δ cross-term** (§2.1), which is
why this document is a subset rather than an equal.

### 12.4 Ivanytskyi — a different model that reaches the same conclusion

Shares the coupling names (η_V = G_V/G_S, η_D = G_D/G_S) and little else:

```
L = q̄(i∂̸ − m + μγ₀)q + G_S Σ_a s_a s_a − G_V j_μ j^μ + 3G_D Σ_{a,b=2,5,7} d⁺_ab d_ab
s_a(x) = ∫dz g(z) q̄(x + z/2) τ_a q(x − z/2) ,     g_k = exp(−k²/Λ²)
```

Nonlocal (smeared currents), **no determinant term at all**, degenerate current
masses so that a single μ enforces colour and electric neutrality automatically,
and a conventional factor 3 on the diquark term giving Δ*Δ/(12G_D) in Ω where I
have Δ²/(4G_D).

Most important for §9: **his conformal restoration is not a density-dependent
G_V.** The vector mean field *saturates*
(ω_∞ = −d G_V Λ³/(4π^{3/2}) = const) and the **formfactor vanishes**, giving

```
c_s² ≃ (1/3)[ 1 − ω_∞ μ ∂/∂μ (g_μ/μ) − m²/μ² ]  →  1/3
```

In my language a formfactor that kills the vector contribution faster than n^{4/3}
is the α > 2/3 branch of c_s² → max(1−α, 1/3) — **same conclusion, different
mechanism.** His model-independent asymptotic relation δ = 1/3 − c_s² is a useful
cross-check: it holds to 1.2×10⁻¹⁰ in my α = 2/3 model at k_F = 3 GeV, and only
to 3×10⁻² (α = 0) and 1.5×10⁻² (α = 1/3) away from marginality — exactly as he
states, an asymptotic identity.

He also flags a tension worth carrying: the confining density-functional approach
reaches c_s² from *below* while pQCD approaches from the opposite side. Mine
reaches 1/3 **from above** at G_V = 0 with a gap (1/3 + (2/9)Δ²/μ², §9.6), and is
*exactly* 1/3 at every density for α = 2/3 — neither above nor below.

---

### 12.5 Provenance of the literature statements

The equations and conventions in §1–§2, §7 and §12 were extracted from arXiv
LaTeX source, not from PDF text, so factors and signs are transcribed rather than
reconstructed. Three items carry explicit caveats:

- The Rüster *et al.* cross-term attribution in §2.1 was checked against the
  source directly: the sentence stating that the 't Hooft term generates mixed
  contributions ∝ Σ_α σ_α |Δ_α|², and that they neglect such terms, appears
  verbatim in the paragraph following their Ω equation.
- Two numeric constants in the accompanying literature extraction
  (`lit_modern.json`) are slightly misstated at the last digit — a BCS T_c/Δ₀
  ratio given as 0.72 where the stated formula yields 0.714291, and a
  Ginzburg–Landau coefficient given as 12.5145 where 32π²/(21ζ(3)) = 12.511385.
  **Neither value is used anywhere in this document**; the melting-pattern
  statement in §7.2 rests on the qualitative uSC-versus-dSC ordering only.
- Section and equation numbers quoted from preprints should be re-checked against
  journal versions before they appear in a paper.

---

## 13. Parameters — what is fixed and what the Bayesian analysis samples

The model has three tiers. Only tier 3 enters the sampler; tiers 1–2 are frozen
before any dense-matter run.

**Tier 1 — fixed by vacuum physics (never sampled).** The RKH set of §4:

| parameter | value | fixed by |
|---|---|---|
| Λ | 602.3 MeV | vacuum fit (with tier-1 partners) |
| G_S Λ² | 1.835 | m_π, f_π |
| K Λ⁵ | 12.36 | m_η′ |
| m_u = m_d | 5.5 MeV | m_π |
| m_s | 140.7 MeV | m_K |

Re-sampling any of these breaks the vacuum phenomenology the model is anchored
to; they move only if the whole vacuum fit is redone (§10.2a shows how the fit
responds — and that (f_π, condensate) alone leave a two-root degeneracy, so m_π
and m_K must stay in any refit).

**Tier 2 — structural choices (discrete, declared per run, not sampled).**
Regularization scheme (sharp cutoff λ = 1 versus RG-consistent λ = Λ_UV/Λ ≈ 10,
§7.2 — production CSC work needs the latter); pairing pattern enumeration list
(§6); lepton content (e, μ; neutrinos if trapping); the vector-coupling form
(constant vs the gluon-exchange form of §9.5); dilaton graft on/off; q ∈ {0, 1}
and p = 1 in the graft.

**Tier 3 — free parameters, the Bayesian vector.**

| parameter | role | natural range / prior support | notes |
|---|---|---|---|
| η_D = G_D/G_S | diquark strength | 0.5 – 1.5 | Fierz gives 0.75; Kunkel use 1.45 (RG-consistent); check the vacuum-pairing bound |
| η_V = G_V/G_S | vector repulsion | 0 – 1 | constant-G_V variant; Kunkel use 0.7 |
| G_V⁰/G_S, M_g | vector repulsion | 0 – 1; 400 – 800 MeV | density-dependent variant (§9.5); replaces η_V |
| α | vector decay exponent | 0 – 1 | only if the power-law form of §9.2 is used instead of §9.5; α = 2/3 is the conformal point |
| B_g^{1/4} | dilaton graft: bag scale | 150 – 400 MeV | sets deconfinement onset (§10.6); only with the graft on |

Everything else in the EoS pipeline — μ₃, μ₈, μ_C, the Δ_η, M_f, φ — is an
*internal unknown* solved at each (μ_B, T), never a sampled parameter. Λ_UV (or
λ) is a convergence control, not physics: results must be shown λ-independent,
not marginalized over λ.

Practical note for the likelihood: at fixed tier-3 point the EoS is deterministic,
so the sampler cost is one full (μ_B, T)-table per point. The solver of §8 must
signal non-convergence (§9.3 of the CCDM companion applies verbatim) — a silent
fallback poisons the posterior.

---

## 14. References (arXiv)

Core NJL + CSC formalism:

- Nambu, Jona-Lasinio — *Dynamical model of elementary particles I, II*, Phys. Rev. 122 (1961) 345; 124 (1961) 246. (pre-arXiv)
- Klevansky — *The Nambu–Jona-Lasinio model of QCD*, Rev. Mod. Phys. 64 (1992) 649. (pre-arXiv)
- Rehberg, Klevansky, Hüfner — *Hadronization in the SU(3) NJL model*, Phys. Rev. C 53 (1996) 410 — **arXiv:hep-ph/9506436** (the RKH parameter set of §4)
- Buballa — *NJL-model analysis of dense quark matter*, Phys. Rept. 407 (2005) 205 — **arXiv:hep-ph/0402234**
- Steiner, Reddy, Prakash — *Color-neutral superconducting quark matter*, PRD 66 (2002) 094007 — **arXiv:hep-ph/0205201**
- Rüster, Werth, Buballa, Shovkovy, Rischke — *Phase diagram of neutral quark matter*, PRD 72 (2005) 034004 — **arXiv:hep-ph/0503184**
- Blaschke, Fredriksson, Grigorian, Öztaş, Sandin — *Phase diagram of three-flavor quark matter under compact star constraints*, PRD 72 (2005) 065020 — **arXiv:hep-ph/0503194**
- Alford, Schmitt, Rajagopal, Schäfer — *Color superconductivity in dense quark matter*, RMP 80 (2008) 1455 — **arXiv:0709.4635**

The four reference treatments of §12:

- Baym, Hatsuda, Kojo, Powell, Song, Takatsuka — *From hadrons to quarks in neutron stars*, Rept. Prog. Phys. 81 (2018) 056902 — **arXiv:1707.04966**
- Kunkel, Rather, et al. — *The petit four of color-superconducting phases in proto-neutron star evolution* — **arXiv:2607.11537**
- Pagliara, Schaffner-Bielich — *Stability of CFL cores in hybrid stars*, PRD 77 (2008) 063004 — **arXiv:0711.1119**
- Ivanytskyi — *nonlocal NJL with CFL*, PRD 111 (2025) 034004 — **arXiv:2409.05859**

Regularization / RG consistency (§7):

- Braun, Leonhardt, Pospiech — *RG consistency and low-energy effective theories* — **arXiv:1806.04432**
- Gholami, Hofmann, Buballa — *RG-consistent treatment of CSC in the NJL model* — **arXiv:2408.06704**
- Gholami et al. — *Astrophysical constraints on CSC phases (RG-consistent NJL)* — **arXiv:2411.04064**
- *Finite-temperature framework for CSC quark matter* — **arXiv:2512.16720**
- *Renormalizing the quark–meson–diquark model* — **arXiv:2505.22542**
- Geißel et al. — *CSC under neutron-star conditions* — **arXiv:2504.03834**
- Schmitt — *Phases and properties of color superconductors* — **arXiv:2511.07319**

Conformal limit and vector channel (§9):

- Ivanytskyi, Blaschke — *density functional with confinement and CSC* — **arXiv:2204.03611**; *conformal limit with density-dependent couplings* — **arXiv:2209.02050**; *early deconfinement, asymptotically conformal CSC* — **arXiv:2211.12730**
- Shukla, Lo — *asymptotic c_s², NJL-class failure* — **arXiv:2507.06741**
- Benghi Pinto — *vector repulsion and the sound-speed peak* — **arXiv:2208.06911**
- Azeredo, Pasqualotto, Lopes, Duarte, Farias — *conformal-limit violation, medium separation scheme* — **arXiv:2602.05796**

Confinement grafts and neighbours (§10):

- Drago, Fiolhais, Tambini — *Quark matter in the chiral colour-dielectric model* — **arXiv:hep-ph/9503462**
- Ghosh, Phatak — *Three-flavour quark matter in the chiral colour-dielectric model* — **arXiv:nucl-th/9509017**
- de Carvalho, Malheiro, et al. — *Color superconductivity and confinement in the chromodielectric model*, Nucl. Phys. B Proc. Suppl. 199 (2010) 308, doi:10.1016/j.nuclphysbps.2010.02.049. (no arXiv located)
- Baldo, Burgio, Castorina, Plumari, Zappalà — *NJL with density-dependent cutoff* — **arXiv:hep-ph/0607343** (also the proceedings **arXiv:0710.5388**)
- Casalbuoni, Gatto, Nardulli, Ruggieri — *running NJL coupling, μ-dependent cutoff* — **arXiv:hep-ph/0302077**
- Lawley, Bentz, Thomas — *NJL with simulated confinement (proper-time IR cutoff)* — **arXiv:nucl-th/0409073**, **arXiv:nucl-th/0602014**

arXiv IDs above were carried over from the project's literature-extraction files
(`lit_core.json`, `lit_modern.json`, `lit_conformal.json`, `refs_csc.bib`), where
each source's LaTeX/PDF was actually fetched; titles for the newest preprints are
as recorded there and should be re-checked against the journal versions.
