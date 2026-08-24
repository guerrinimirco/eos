# Ticket 06 — do the model documents pass CLAUDE.md §11's reproduce-without-source test?

Type: research (read-only). Parent: [../map.md](../map.md). Ticket: [../issues/06-document-audit.md](../issues/06-document-audit.md)

Nothing in `eos/` or `docs/` was changed. Every defect below is reported, none is fixed.

## Scope note

The ticket says 25 documents. There are **24**: eleven model pairs (`zl`, `sfho`, `dd2`,
`did`, `vmit`, `alphabag`, `njl`, `ccdm`, `abpr`, `enjl`, `mixed`) plus **both**
`eos/astro/tov/tov.md` and `eos/astro/tov/tov.tex` — the ticket named only the `.tex`,
but `tov.md` exists and is audited here on the same rubric.

## The rubric

One column per checkable claim in the ticket, graded against §11's test — *a physicist
reproduces the model from the document without opening the source*:

- **C1** — Lagrangian or grand potential stated in closed form (not named)
- **C2** — every parameter, with its value, and the reference it is fitted to
- **C3** — the field / gap equations
- **C4** — the residual row by row **in the order `solver.py` assembles it**, with the
  unknown vector (graded by extracting the true row order from source and comparing)
- **C5** — single-species thermodynamics at T = 0 **and** T > 0, written out, not cited
- **C6** — every returned quantity, including `s` and `n_s`, with the identities
  `n_s = (eps − 3P)/m*` and `s = (eps + P − Σ_i mu_i n_i)/T`
- **C7** — the terms that differ between `P` and `eps`
- **C8** — which rows each mode changes

`n/a` means the physics genuinely has no such part (no scalar field → no gap equation;
no scalar density → no `n_s`; a closed-form inversion → no residual vector). Each `n/a`
is justified in the per-document paragraph.

## The table

| document | C1 L/Ω | C2 params+ref | C3 field/gap | C4 residual+order | C5 T=0 & T>0 | C6 returns, s & n_s | C7 P vs eps | C8 mode rows | score |
|---|---|---|---|---|---|---|---|---|---|
| `zl/zl.md` | Partial | **Fail** | n/a | Partial | **Fail** | **Fail** | Pass | Pass | 6/14 |
| `zl/zl.tex` | Pass | Pass | n/a | Pass | Pass | Pass | Pass | Pass | **14/14** |
| `sfho/sfho.md` | Partial | **Fail** | Pass | Pass | Partial | Partial | Partial | Partial | 9/16 |
| `sfho/sfho.tex` | Pass | Partial | Pass | Pass | Pass | Pass | Pass | Pass | 15/16 |
| `dd2/dd2.md` | **Fail** | **Fail** | Pass | Pass | Partial | Partial | Pass | Partial | 9/16 |
| `dd2/dd2.tex` | Pass | Pass | Pass | Pass | Partial | Partial | Pass | Pass | 14/16 |
| `did/did.md` | **Fail** | **Fail** | **Fail** | Partial | **Fail** | **Fail** | Partial | Partial | **3/16** |
| `did/did.tex` | Pass | **Fail** | Pass | Pass | Partial | Partial | Pass | Pass | 12/16 |
| `vmit/vmit.md` | Pass | **Fail** | Pass | **Fail** | **Fail** | **Fail** | Pass | Partial | 7/16 |
| `vmit/vmit.tex` | Pass | **Fail** | Pass | **Fail** | Partial | Partial | Pass | Partial | 9/16 |
| `alphabag/alphabag.md` | Pass | Pass | Pass | Partial | Partial | Partial | Pass | Pass | 13/16 |
| `alphabag/alphabag.tex` | Pass | Pass | Pass | Pass | Pass | Pass | Pass | Pass | **16/16** |
| `njl/njl.md` | Pass | Partial | Partial | **Fail** | **Fail** | **Fail** | **Fail** | **Fail** | **4/16** |
| `njl/njl.tex` | Pass | **Fail** | Pass | Pass | Partial | Partial | Pass | Pass | 12/16 |
| `ccdm/ccdm.md` | **Fail** | Partial | **Fail** | **Fail** | **Fail** | **Fail** | Partial | Partial | **3/16** |
| `ccdm/ccdm.tex` | Pass | Partial | Pass | Pass | Pass | Partial | Pass | Pass | 14/16 |
| `abpr/abpr.md` | Pass | Partial | Pass | n/a | Partial | Partial | Pass | Pass | 11/14 |
| `abpr/abpr.tex` | Pass | Partial | Pass | n/a | Pass | Pass | Pass | Pass | 13/14 |
| `enjl/enjl.md` | Partial | Partial | Partial | Pass | Partial | Partial | Pass | Partial | 10/16 |
| `enjl/enjl.tex` | Pass | Pass | Pass | Partial | Pass | Partial | Pass | Pass | 14/16 |
| `mixed/mixed.md` | Partial | Partial | Partial | Partial | n/a | Partial | Pass | Pass | 9/14 |
| `mixed/mixed.tex` | Pass | Partial | Partial | Pass | n/a | Partial | Pass | Pass | 11/14 |
| `astro/tov/tov.md` | Partial | Partial | Partial | n/a | n/a | Partial | n/a | Partial | 5/10 |
| `astro/tov/tov.tex` | Partial | Partial | Partial | n/a | n/a | Partial | n/a | Partial | 5/10 |

Score = Pass 2, Partial 1, Fail 0, over the applicable columns. **Two documents pass
outright: `zl.tex` and `alphabag.tex`.** `sfho.tex` and `abpr.tex` miss by one cell each.

## Per document — exactly what is missing

### `eos/zl/zl.md` — 6/14
Seven of the eight fitted numbers are named without values (`n0 = 0.16`, `a0 = −96.64`,
`b0 = 58.85`, `gamma = 1.40`, `a1 = −26.06`, `b1 = 7.34`, `gamma1 = 2.45`, from
`parameters.py:52-59`); only the nucleon masses carry a number. The entire single-species
Fermi gas is delegated by the exact sentence §11 forbids — "evaluated at `mu_eff_i`
through the JEL integrals in `eos.general.fermi_integrals`" — so neither the T > 0
integrals nor the T = 0 closed forms in `k_F` appear. `s_kin` is never defined (only
`s_int = 0`), there is no photon or lepton block, no assembly of the totals, and no
`f = eps − Ts` although `f` is returned. `mu_Hv_i` is left as `dV/dn_i` rather than
differentiated. C3 is `n/a` and correctly declared: "There is no scalar field, so no gap
equation and no effective mass"; `n_s` is `n/a` for the same reason.

### `eos/zl/zl.tex` — 14/14, the reference document of the repository
Nothing §11 enumerates is missing. Eq. (4) gives `V` in closed form, the parameter table
carries all eight values with the Constantinou 2021/2023/2025 provenance and the caveat
that `n0` is not `n_sat`, Eqs. (5)–(8) write the Fermi integrals with
`f(x) = [1+e^{x/T}]^{-1}` *before* naming JEL as the evaluator (the correct pattern),
Eq. (12) gives the T = 0 closed forms, Eq. (15) gives `x = [mu_p, mu_n, mu_e, n_p, n_n]`
with r1…r5 in `solver.py`'s order, and each mode paragraph says which rows drop.
**One factual defect** (report only): r5 in the `fixed_YC` branch is written `n_p − n_e`
where `solver.py:334` returns `n_e − state.n_C` — same root, transcribed with the opposite
sign. Also not stated: the electron/neutrino gas is deferred to
`eos.general.thermodynamics_leptons` without repeating `m_e` or the degeneracies, and
`eos_response`'s outputs are undocumented.

### `eos/sfho/sfho.md` — 9/16
**Not one coupling value appears**: no `g_sigma_N`, `g_omega_N`, `g_rho_N`, `g2`, `g3`,
`c3`, `c4`, `a_1..a_6`, `b_1..b_3`, no meson masses, no baryon masses. The model cannot be
reproduced from this file at any level. C5 fails on its own terms: the T = 0 forms are
handed to the sibling — "the closed forms are in `sfho.tex` Eq. (T0)" — which is precisely
a document that does not stand alone. `eps_mf`/`P_mf` are named ("with `P_mf`, `eps_mf`
the mean-field terms above") where "above" gave only the omega terms, so the reader never
learns the `3c4/4` vs `c4/4` rho-quartic asymmetry or the `−V(sigma)` vs `+V(sigma)` sign
flip — which is why C7 is Partial. There is no mode→rows table; the mapping must be
inverted out of the row conditionals. C3 and C4 are its strengths: all four field
equations with `hc^3` on the sources (which the `.tex` gets *wrong*), and R1…R9 matching
`solver.py:430-457` row for row including the 30 MeV field scale.

### `eos/sfho/sfho.tex` — 15/16
The only §11 gap is C2: the published SFHoY scalar coupling ratios (`x_sigma` = 0.854315
Lambda, 0.586611 Sigma, 0.512754 Xi in `parameters.py`) are referred to as "the published
SFHoY values" without numbers, and no non-nucleonic baryon mass or `m_e` appears anywhere.
Table I's thirteen nucleonic parameters were recomputed from `parameters.py` and agree to
all quoted digits. **Two defects** (report only): Eq. (5) writes the field sources without
the `(hbar c)^3` that both the code and the document's own R1–R4 carry, so the
field-equation section as printed is dimensionally wrong; and the three-flavour massless
`mu = 0` neutrino gas that `assemble` adds to P, eps and s
(`N_NEUTRINO_FLAVOURS = 3.0`, `solver.py:523-527`) appears in **neither** SFHo document.
`eos_response`'s return set is documented in the `.md` and not here.

### `eos/dd2/dd2.md` — 9/16
No Lagrangian (the model is named: "DD-RMF of Typel et al., PRC 81, 015803 (2010)"), and
**not one numerical coefficient** — no `n_sat`, no `Gamma_i(n_sat)`, no `a/b/c/d`, no
masses; the coupling *forms* are given but nothing pins them down. `Sigma^R` is used in
`mu_eff_i`, in `P` and in `U_Y` and is **never defined** — §11's naming-not-defining
prohibition, applied to the term that makes the model density-dependent. T = 0 is again
delegated to the sibling ("the closed forms are in `dd2.tex` Eq. (T0)") and the Bose
integrals to `eos/general/bose_integrals`. No mode table. C4 is a genuine Pass: R1…R8 in
`octet_residual`'s order with the same conditionals and scaling.

### `eos/dd2/dd2.tex` — 14/16
C5 and C6 are Partial for one shared reason: the **thermal meson sector has no Bose
thermodynamics anywhere in the document.** §gas gives the three `mu^*_j` and then says
"Its P, eps, s and Σ_j mu*_j n_j join the totals" — no `n_j`, `P_j`, `eps_j`, `s_j`, no
meson masses, no degeneracies. Also absent: the hyperon/Delta/lepton masses as numbers;
the whole `eos_response` set (`C_V`, `C_P`, equilibrium and frozen `c_s^2`, `Gamma`,
`chi_ab`) that `api.py:142-160` returns; the `condensation` diagnostic and the hard
refusal at `|mu*|/m ≥ 1` (`solver.py:792`); the four rows of the reduced nucleon-only beta
system (named, `x = [sigma, rho_0, mu_eff_n, mu_C]`, rows not given); and the entire
phase-adapter residual (`self_consistency_residual`, unknowns `[sigma, omega, rho, (phi),
n_B]`, no charge row) that `eos/mixed` consumes — which `did.tex` *does* document.

### `eos/did/did.md` — 3/16, joint worst
This is a README, not a specification. No Lagrangian, no grand potential, no field
equations, no `m*_i`. **Exactly two parameter numbers exist in the file** (`n_0` and
`g_phiN = −5.20`) against 27 fields in `Parameters.default()`. There is no kinetic section
at all: the strings `eps`, `P^kin` and `k_F` do not occur, so `n_i`, `n^s_i`, `eps_kin`,
`P_kin`, `s_i` are absent at both temperatures, as are both §11 identities, the totals,
the charge sums, leptons, photons and the meson gas. The residual rows exist only as
English names — "Rows: four field equations, the beta definition, the Sigma^t
definition…" — which is why C4 is Partial rather than Pass: the *order* is right and the
unknown vector is exact and complete, but no row is written as an equation. `Sigma^r` and
`Sigma^t` in closed form and the P/eps rearrangement rule are the only physics that
survives.

### `eos/did/did.tex` — 12/16
Equationally this is one of the strongest documents in the repo — Lagrangian *and* grand
potential with the log integral, all four field equations, both rearrangement terms with
the chain rule producing the `(tau_3i − beta)` weight, R1…R11 in `residual()`'s exact
order including the subtlety that R11 must see the post-hoc neutralizing leptons, and the
best C8 of any document (a mode table with an explicit "extra rows" column). C2 fails
alone and completely: **there is no parameter table anywhere** —
`grep '\begin{tabular}'` finds only the mode table and the NMP comparison. Four numbers
appear in the whole document against 27 stored fields; the reader is pointed at "the
transcribed Table II parameters". Also missing: the Bose thermodynamics of the pi/K sector
(only the three `mu^*_j`), the numerical spin degeneracies `d_i`, the value of
`FIELD_SCALE`, and the `eos_response` definitions.

### `eos/vmit/vmit.md` — 7/16
`grep` for `180`, `0.2`, `150` over the file returns **nothing**: `B4`, `a`, `m_u`, `m_d`,
`m_s` are named with units and no values, and no reference is given — nor the fact that
`get_vmit_default` calls the set "a starting point, not a published fit". C5 fails on the
forbidden sentence ("through the JEL integrals in `eos.general.fermi_integrals`"). C4
fails outright: the rows never appear as a residual, and the single unknown vector given
at L68 matches `fixed_YC` but **contradicts** `solve_vmit_beta_eq` and
`solve_vmit_trapped_neutrinos`, where `mu_e` and `mu_nu` occupy slots 4–5, not the tail.
`s` is never given a formula; the lepton and photon terms that `solver.py:317-325` adds to
all three totals are asserted in prose. **Stale claim** (report only): L81 says
`eos_response` is "not implemented" — `api.py:167-212` implements it and
`docs/DEFERRED.md:552` records its freeze list. `n_s` is genuinely `n/a`: no scalar density
exists anywhere in `eos/vmit`.

### `eos/vmit/vmit.tex` — 9/16
The Lagrangian, the vector field equation `m_V^2 V^0 = g_V Σ_q n_q → a (hbar c) Σ_q n_q`,
and the T > 0 quark Fermi integrals with `s_q = (eps_q + P_q − mu_eff_q n_q)/T` are all
present and closed-form. Everything else fails with the `.md`: **no parameter table and no
number at all** (`B^{1/4} = 180` MeV, `a = 0.2` fm², `m_{u,d,s} = 5/7/150` MeV are
nowhere); **no T = 0 limit** — no `k_F = sqrt(mu_eff^2 − m^2)`, no closed forms, and the
document then says "These integrals are *not* implemented here: they come from
`eos.general.fermi_integrals`"; **no enumerated residual for any mode** and the same wrong
unknown ordering for the two beta modes; the totals equation omits the electron, neutrino
and photon terms the code adds; and no returned-field list against `VMITEOSResult`'s 30
fields. Neither document states that `leptons=True` adds a seventh row.

### `eos/alphabag/alphabag.md` — 13/16
Strong for a `.md`, and the parameter values *are* here (`m_u = m_d = 0`, `m_s = 150`,
`alpha = 0.3`, `B4 = 165`, `B = 96.466 MeV/fm^3`) with the Fischer et al. provenance. Three
Partials: the massive flavour is named, not defined ("the exact Fermi gas from
`eos.general.fermi_integrals` (JEL)") with no T = 0 `k_F` form and no
`s_F = (eps_F + P_F − mu n_F)/T`; the mode table's row listing happens to match the
solver's order for all five solvers but never says that it *is* the assembly order and
leaves the rows unnumbered; and there is no returned-field list — `EoSPoint` also carries
`converged/error`, `Y_L`, `mu_B/C/S`, `n_e`, `n_nu`, `f_total` and five `Y_*`, and the
photon formula `P_gamma = pi^2 T^4/45(hc)^3` never appears.

### `eos/alphabag/alphabag.tex` — 16/16, the second passing document
Every row of every mode numbered and matching source order, with the row scales; the
massive-flavour Fermi integrals at T > 0 *and* their T = 0 closed forms with the
`m^4 ln[(k_F+E_F)/m]` terms and the `n_F = 0` for `mu ≤ m` branch; the massless limit
explicitly reduced at T = 0; a field-by-field returns table matching `EoSPoint`/`CFLPoint`;
the lepton/photon/thermal-neutrino totals with the three-vs-two flavour counting rule that
`point_from_mu` implements; and an entropy paragraph that exists for exactly §11's stated
reason ("worth a word because nothing in the residual derives from it and it is therefore
easy to leave unstated"). `n_s` is `n/a` and grounded (masses are parameters, no gap
equation). Only non-§11 gap: `eos_response` is never mentioned, though `api.py:175`
returns `cs2_eq` and `C_V`.

### `eos/njl/njl.md` — 4/16
The Lagrangian is written out and tier-1 parameters carry values with the Rehberg–
Klevansky–Hüfner reference — and then the document stops being a specification. No unknown
vector, no residual rows, no integral of any kind at either temperature, no `Omega`, no
`eps`, no `s`, no scalar density, neither §11 identity, no P/eps difference (there is no P
or eps expression to compare), and "**Modes.** All four of CLAUDE.md §3" as the entire
treatment of C8. The mass gap equation it does print is unevaluable because `phi_u` is
never defined; the diquark gap equation is alluded to ("Hellmann–Feynman on the BdG
matrix") and never written. Missing numerics: the shipped `eta_D = 0.75`,
`G_V0_over_GS = 0.5`, `M_g = 500.0`, `alpha`, `n_ref`.

### `eos/njl/njl.tex` — 12/16
Equationally excellent — Lagrangian and grand potential, the mass gap with the 't Hooft
determinant cross-terms, the cutoff-regularized medium integrals with `P_log`, the
closed-form Dirac sea, the 18×18 BdG problem and all four Hellmann–Feynman kernels, the
unknown vector and rows matching `unknown_slots`/`residual`/`_charge_rows` slot for slot,
and a modes table with a "rows replaced" column. **C2 is a total failure and it is the
single largest §11 violation in the audit: there is no parameter section and not one
parameter value in 720 lines.** `grep` for `602`, `1.835`, `12.36`, `140.7` returns zero
hits (verified directly). The "What the implementation reproduces" table quotes *outputs*
(`M_u = 367.648`, `f_pi = 92.391`) against `\cite{Rehberg1996}` — the reference is cited
for the results, never for the inputs — so a reader of this document cannot check a single
number in it. Second: leptons, muons, trapped neutrinos and photons are delegated by
literal source-file citation ("all from `eos/general/thermodynamics_leptons.py`"), which is
worse than a paper citation because the document's own integrals are cut at `Lambda` and
therefore wrong for a lepton. Third: T = 0 is a limiting *rule*, not closed forms. Fourth:
`n_s = (eps − 3P)/m*` is absent, the ~40 returned fields are never enumerated, and neither
document discloses that the field the code calls `n_s` is the **strange-quark density**,
not the scalar density §11 means.

### `eos/ccdm/ccdm.md` — 3/16, joint worst
No Lagrangian and no grand potential. `B_eff = [U(0) − U(phi_0)] + [V(0,0) − V(f_pi,
zeta_0)]` **names `U` and `V` without ever defining either** — the two potentials the whole
model is built from — which is §11's first prohibition at the center of the document. `R_2`
and `R_3` are absent, only the fragment `dU/dPhi = B_g ln Phi` survives, and `Sigma_V =
g_omega omega_0 + Sigma_R` is asserted with none of `Sigma_R`, `g_omega(n_B)` or `omega_0`
given. No integral at either temperature. No unknown vector and no ordered residual. `s`,
the scalar density and both identities are absent. Tier-1 parameters are listed *by name*
("f_pi, m_pi, f_K, m_K, the current masses, m_zeta, m_phi, m_omega") with no numbers.
**Label collision** (report only): its `R1..R5` denote *modes* while `ccdm.tex`'s `R1..R4`
denote *residual rows*, and the same file uses `R_4` in both senses three paragraphs apart.

### `eos/ccdm/ccdm.tex` — 14/16
Full Lagrangian, `U(Phi)` and `V(sigma,zeta)` with both derivatives, the boxed `Omega`, all
five ideal-gas integrals at T > 0 *and* their T = 0 closed forms (including `R_s` with the
`ln` term), the `k_max` prescription as a formula, photons as `pi^2 T^4/45`, and an unknown
vector and row order identical to `unknown_slots`/`residual` — including the vector row
sitting *before* the gaps, which is where CCDM genuinely differs from NJL and which the
document gets right. C2 Partial: the shipped `g_s = 3.0`, `gbar_omega = 4.0`,
`n_c = 1.0`, `Lambda = 600.0` and `q` are given only as prior ranges, with "the shipped
values are mid-prior rather than measurements; the code says so" — which is the one place
it tells the reader to open the source. C6 Partial: `n_s = (eps − 3P)/M*` is absent (the
document argues past it, correctly, at the single-mode Euler relation), and the ~45 fields
`EoSPoint` returns — `branch`, `gapless`, `beyond_cutoff`, `phi_bar`, `chi`, `omega_0`,
`Sigma_R`, `f_total`, the `Y_*` set — are never enumerated.

### `eos/abpr/abpr.md` — 11/14
`P(mu)` is given term by term with `A` and `C` defined, all four parameter values appear,
and the document is honest about provenance ("There is no published single ABPR set").
C4 is `n/a` and correctly handled: **there is no residual vector in the source** — the
solve is Cardano on a depressed cubic followed by `point_from_mu` — and the document states
the closure and the stable inversion form `mu = u − p/(3u)` instead. C5 Partial: T > 0 is
genuinely `n/a` (`check_temperature` raises, and the document says so), but **the
single-flavour thermodynamics the whole expansion is built from is absent** — no `k_F`, no
massive closed forms, and the massless `n = mu^3/(pi^2 (hc)^3)`, `P = mu^4/(4 pi^2 (hc)^3)`
that the `3 a4 mu^4/(4 pi^2)` term is three copies of is never written. No returned-field
list against `CFLPoint`. **Defect** (report only): the code names `ms` and `Delta` are
wrong — `parameters.py:59-62` declares `m_s` and `Delta0`.

### `eos/abpr/abpr.tex` — 13/14
Misses only C2, and for the same code-name error as the `.md` (`\texttt{ms}`,
`\texttt{Delta}` for `m_s`, `Delta0`) in a table that claims to give them. Otherwise it is
a model of §11 compliance, including a section titled "The single-flavour thermodynamics
the expansion comes from" that opens by *quoting §11's own reason*: "Equation (expansion)
is quoted above rather than cited, because a paper-style description must be
self-contained." All three inverse maps in closed form with the `D < 0` trigonometric
Cardano branch and the Descartes-rule argument for the physical root; a field-by-field
`CFLPoint` returns table with the equation per field; `s = 0` argued rather than omitted
("a value the code returns, not a quantity it declines to compute"); and a refusal table
matching `MODE_REFUSALS` reason for reason.

### `eos/enjl/enjl.md` — 10/16
Fails §11's naming-not-defining rule in five places: `alpha_S(n_B)`, `Gamma_omega(n_B)` and
`Gamma_rho(n_B)` are used throughout and **never written** as `a e^{−n_B/n} + b`, so the
nine coefficients listed at L96-98 parametrize an equation the reader never sees and the
primed couplings in `Sigma^R` are derivatives of undefined functions; `E0` is subtracted in
the energy density and quoted as a number and never defined as that expression at zero
density; the photon and thermal-neutrino sectors are named with no formulas; and the
quantum-number table is absent, leaving `N^q_i`, `tau_i`, `q_i`, `C_i`, `S_i`, `B_i`
unvalued so that `J_rho`, `Sigma^R_b` and residual rows 4, 10, 11 cannot be evaluated. Also
missing: the T = 0 closed form for `P^med`, the definition `E(k) = sqrt(k^2+M^2)`, an
explicit `f_+`, `m_e`/`m_mu`, the total entropy assembly, `n_s = (eps − 3P)/m*`, and any
list of what a point returns. **Defect** (report only): the modes table gives `(n_B, T=0)`
as the independent variables for **all four** modes, contradicting the document's own
"All four are closed, at any T ≥ 0" and `check_temperature`. Its one advantage over the
`.tex` is real: C4 is a clean Pass, with the correct fifth unknown `mu_C`.

### `eos/enjl/enjl.tex` — 14/16
The most complete large document in the repo: T = 0 and T > 0 integrals written out with
`f_∓` and `E(k)` defined and an explicit statement that they are restated "because a
paper-style description must be self-contained"; all 23 parameters with values and the
RKH/Xia references; the gap and field equations; closed-form `alpha_S`, `Gamma_omega`,
`Gamma_rho` and `eps_0`; the residual rows *and their scales* matching `state_at` exactly;
and a returns table covering every `EoSPoint` field. Two things cost it: **Eq. (unknowns)
prints `mu_e` as the fifth component where `solver.BASE_UNKNOWNS` carries `mu_C`** —
verified directly against `solver.py:133` — and the very next sentence contradicts its own
equation, so the one place a reader would copy the vector from is wrong, and wrong in a way
that only surfaces in `fixed_YC`, where `mu_e` is not an unknown at all. And
`n_s = (eps − 3P)/m*` appears nowhere. Minor: `kF` is listed twice in the returns table,
and `chi = n_B^Q/n_B`, a delivered `beta_row` column, is defined in neither document.
On finite T: `docs/DEFERRED.md:1070` confirms all four modes solve at any `T ≥ 0`, so T > 0
is in scope and cannot be graded `n/a`; what is deferred is the *construction* above T = 0.

### `eos/mixed/mixed.md` — 9/14
Three residual rows are **named, not written** — "then by regime the average charge,
strangeness and lepton-number conditions" — with no `(1−chi)n_C^H + chi n_C^Q = Y_C n_B`,
no `Y_S` row, no eta-weighted `Y_Le` row. And the one row order it does assert is **wrong**:
it places the eta-shifted charge-matching row after S and L_e, where `solver.py:512-521`
emits it immediately after the C-average row. The `PhaseThermo` block is understated by
five fields (`mu_dot_n`, `Sigma_R`, `mu_eff_i`, `m_eff_i`, `fields`), of which `mu_dot_n`
is what the document's own 1e-8 HVH check consumes. C5 is `n/a` — the engine consumes
phases through the adapter and forms no scalar density — and the lepton single-species
block *is* given in closed form including `s = (eps + P − mu n)/T`. **Defect** (report
only): `mu_mu = mu_e` contradicts `thermodynamics.py:64`'s `mu_mu = mu_e − mu_nue` in the
trapped mode.

### `eos/mixed/mixed.tex` — 11/14
Every residual row as a numbered equation in `residual()`'s exact order, the unknown vector
in slot order, the mode table with each row tagged by regime, and the P-vs-eps asymmetry
stated with its reason (P read off one phase because the mechanical row has made the two
equal; eps volume-averaged). C2/C3/C6 Partial: no pointer to the sibling parameter
documents that carry the phases' numbers; `mu_scale = 100.0 MeV`, `n_scale = max(n_B,
0.01)`, `_P_ROUNDOFF = 1e-12`, `n_probe`, `max_refine`, `max_bisect`, `MAX_WALK` all named
qualitatively and never given; the `PhaseThermo` block understated by the same five fields
and the adapter call signature given only in prose; and the `eos_point`/`eos_table`/
`eos_response` surface is never named, `C_V` appears nowhere, and `MixedWindow`'s `probes`,
`onset_state`/`offset_state` and four `reason` labels are absent. **Defects** (report
only): line 524 uses `\tmuB`, which is never defined (only `\mutB` is, line 12) — verified,
**the document as committed will not compile past that equation**; the same `mu_mu = mu_e`
error as the `.md`; and the refusal of fixed `Y_C` together with fixed `Y_Le`
(`solver.py:260-263`) is unmentioned though two other refusals are.

### `eos/astro/tov/tov.tex` — 5/10
The static layer is well covered — TOV in physical and integrated form, the tidal Riccati
with `F`, `Q` and the `ΔY` jump, the crust blends, the returns table matching
`_results_to_array` — and C4/C5/C7 are legitimately `n/a` (no algebraic residual, no
species, no model P/eps decomposition; the document says so: "This layer consumes tables
and produces observables"). **The rotating half is a citation, not a formulation**: "Rotating
models are not computed here. They are computed by RNS" is the entire treatment — no metric
ansatz, no KEH self-consistent-field equations, no Cook–Shapiro–Teukolsky modification, and
neither Komatsu 1989 nor CST 1994 is in `docs/eos.bib`. The RNS table's defining relation
`dh = dp/(e+p)` is never given although the document requires its column to round-trip, and
the entire `RotatingResult` return set (`M_0`, `R_e`, `r_e`, `r_p`, `Omega`, `freq`,
`Omega_K`, `freq_K`, `J`, `I`, `T_over_W`, `Z_p`, `Z_f`, `Z_b`, `accuracy`, `converged`) is
undocumented. Also absent: `backend='fast'` — a *different* integrator (RK45, rtol 1e-6,
atol 1e-8, uniform log10 P grid, `M_b` as a fourth ODE variable) that the document never
mentions; `r_max = 15.0`; the `P_tol = 1e-4` plateau rule; the crust defaults; and the fact
that the surface `ΔY` is applied to **every** star, not only self-bound ones.

### `eos/astro/tov/tov.md` — 5/10
Same three structural gaps as the `.tex` (rotating formulation, RNS enthalpy and file
layout, `RotatingResult`), plus its own: **`k2` is cited rather than written** — "then k2
from the surface value Y(R) and compactness C (Hinderer 2008 Eq. 23)" — so the three-term
denominator `D` of `_love_number` is nowhere; `m̂(r̂_min)` is named ("the uniform-core m̂")
and not written; the `interpolate` tanh blend and the `maxwell` `eps(P)` blend are
described in words only; and every citation the `.tex` carries is absent (BPS named without
Baym–Pethick–Sutherland, CompOSE without Typel, and Tolman / Oppenheimer–Volkoff /
Damour–Nagar uncited). It has one thing the `.tex` lacks: the module map is the **only**
place in either document that `solver_fast.py` is mentioned at all — and it is mischaracterized
there as "the jitted variant of the same integration", which it is not.

## `.md` vs `.tex` — which carries more (input to ticket 09)

**The `.tex` carries more in eleven of the twelve pairs.** The exception is none: even
where the `.md` wins a column, it wins one, and the `.tex` wins the document.

| pair | more coverage | what the loser uniquely holds |
|---|---|---|
| zl | **.tex**, decisively | `.md`: the "Not implemented" ledger only |
| sfho | **.tex** | `.md`: the `hc^3` on the field sources (the .tex is wrong here), the source definitions inline, the `eos_response` return set |
| dd2 | **.tex** | `.md`: names Bose statistics at all (the .tex never mentions them) |
| did | **.tex**, overwhelmingly | `.md`: that DID and DIDY are one parameter set differing only by a flag; the `SpeciesFlags` roster; `responses.py`'s outputs — the only mention in either file |
| vmit | **.tex** (but both fail) | `.md`: the per-mode condition table; a "Not implemented" line that is **wrong** |
| alphabag | **.tex** | `.md`: the "Not implemented" ledger; `eos_response` is absent from the .tex |
| njl | **.tex** for equations, **.md** for numbers | `.md`: **every parameter value** — the .tex has none |
| ccdm | **.tex**, on all eight columns | `.md`: usage snippet, verify entry point, one number (`n_B ≈ 1.35 fm^-3`) |
| abpr | **.tex** | `.md`: the file-layout map, which is what documents the absence of `table.py` |
| enjl | **.tex** | `.md`: the **correct** fifth unknown `mu_C`; the named parameter-set API |
| mixed | **.tex** | `.md`: the "Not implemented" list incl. the live `fixed Y_C + Y_Le` refusal; the `ChargeSpec = ModeSpec + Locality` rule; DID among the pairings |
| tov | **.tex** | `.md`: the module map, the only mention of `solver_fast.py` |

**Two pairs block a clean "drop the .tex" decision outright**, because the `.md` explicitly
delegates to the `.tex` for content §11 requires: `sfho.md` says "the closed forms are in
`sfho.tex` Eq. (T0)" and `dd2.md` says "the closed forms are in `dd2.tex` Eq. (T0)". A
document that names its sibling as its own completion cannot be the surviving one.

**A "drop the .md" decision is cheaper but not free.** Before any `.md` is dropped these
must move into the corresponding `.tex`: SFHo's `hc^3` field-source correction and
`eos_response` set; DD2's mention of Bose statistics; DID's DID/DIDY identity and
`responses.py` outputs; **NJL's entire parameter tier list** (the `.tex` has zero
parameter values); ENJL's correct `mu_C` unknown; mixed's "Not implemented" list including
the live `fixed Y_C + Y_Le` refusal; ABPR's layout map; TOV's `solver_fast.py`; and the
per-model "Not implemented" ledgers, which no `.tex` carries.

## Cross-cutting findings

1. **The Bose integrals are the most-violated clause.** §11 names them explicitly. `dd2`
   (both files) and `did` (both files) give the three meson effective potentials and no
   Bose thermodynamics at all; `sfho.tex` is the only DD-RMF document that writes them.
2. **`n_s = (eps − 3P)/m*` is missing wherever it applies but is not a DD-RMF.** Present in
   `sfho`, `dd2`, `did`. **Absent** in `njl.tex`, `ccdm.tex` and both ENJL documents, all of
   which compute a scalar density. Genuinely `n/a` in `zl`, `vmit`, `alphabag`, `abpr`,
   `mixed`, `tov` — none forms a scalar density — and each of those says so except
   `vmit.md`.
3. **`eos_response` is undocumented in almost every document.** The uniform API's third
   entry point (CLAUDE.md §5) has its return set written out nowhere except `sfho.md` and
   `did.md`. Under a strict reading of "every quantity it returns is written out", this
   alone keeps most documents off a full C6 Pass.
4. **Two shipped models are absent from CLAUDE.md's own layout.** `njl` and `ccdm` appear
   in neither §1's model list nor §11's directory list (verified by grep); `did` appears in
   §11 but not §1. §5's shipped-adapter list also omits the NJL adapter that `njl.tex`
   documents as existing. This is a CLAUDE.md defect, not a document defect, and belongs to
   ticket 22.
5. **Two `.md` files name an external document as the governing authority** —
   `njl.md` ("the authority wherever the two differ" is `docs/njl_csc_implementation.md`)
   and `ccdm.md` similarly. That is structurally in tension with §11: the reader must now
   reconcile three documents.

## Factual defects found (reported, not fixed — per the map's hard rules)

- `zl.tex` — r5 in the `fixed_YC` branch has the opposite sign to `solver.py:334`.
- `sfho.tex` — Eq. (5) field sources lack the `(hbar c)^3` the code and its own R1–R4 carry.
- `sfho` (both) — the three-flavour `mu = 0` thermal-neutrino gas added to P, eps, s
  (`solver.py:523-527`) is in neither document.
- `vmit.md:81` — claims `eos_response` is not implemented; `api.py:167` implements it.
- `vmit` (both) — the unknown-vector ordering given contradicts `solve_vmit_beta_eq` and
  `solve_vmit_trapped_neutrinos`.
- `abpr` (both) — parameter table gives code names `ms`/`Delta` for `m_s`/`Delta0`.
- `ccdm.md` — `R1..R5` label modes while `ccdm.tex`'s `R1..R4` label residual rows, and the
  `.md` uses `R_4` in both senses.
- `njl` (both) — the returned field named `n_s` is the strange-quark density, not a scalar
  density; neither document says so.
- `enjl.tex` — Eq. (unknowns) prints `mu_e` where `BASE_UNKNOWNS` carries `mu_C`, and the
  following sentence contradicts the equation.
- `enjl.md` — the modes table gives `T = 0` for all four modes; all four solve at `T ≥ 0`.
- `mixed` (both) — `mu_mu = mu_e` where `thermodynamics.py:64` uses `mu_mu = mu_e − mu_nue`.
- `mixed.tex:524` — undefined macro `\tmuB` (only `\mutB` is defined, line 12); **the
  document does not compile past that equation.**
- `tov.md` — `solver_fast.py` described as "the jitted variant of the same integration";
  it is a different integrator with different tolerances and a different `M_b` algorithm.
