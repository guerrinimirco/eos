# DID parameter provenance — what fixed each number, and what a sampler may move

Research output for ticket 113 (`issues/113-did-parameter-provenance.md`).
Question: per coupling, which quantity fixed it, what KIND of constraint that
is, and whether a Bayesian inference run may vary it.

**Primary source.** G. Frohaug, K. Maslov, V. Dexheimer, J. Grefa, J. Jahan,
C. Ratti and T. E. Restrepo, *Relativistic mean-field model with density- and
isospin-density-dependent couplings*, arXiv:2511.15646v1 [nucl-th], 19 Nov 2025
(`Frohaug2025` in `docs/eos.bib`). All section, table and equation numbers below
are that paper's own labels, read from the PDF. The repository side is
`eos/did/parameters.py`, `couplings.py`, `nmp.py`, `did.md`, `did.tex`.

Reproduction is NOT in scope here: ticket 103 already established that
`eos/did/verify/run_full_check.py::check_nuclear_matter_parameters` reproduces
Table VI at 0.84x tolerance.

---

## 1. The headline: no astrophysical observation is in the likelihood

This is the single most important provenance fact and it is easy to get wrong,
because the abstract and Sec. I both name NICER and GW170817.

Sec. IV: "was fitted using the PyMultiNest package [154] with 2000 live points
to **18 observables** described below, with likelihoods modeled as independent
normally distributed variables (least-squares minimization of z-scores). The
maximum likelihood estimate (MLE) was then found using a minimizer initialized
at the greatest-likelihood posterior samples."

Sec. IV C enumerates those 18. **Not one of them is an astrophysical
observation.** NICER and GW170817 enter the paper in exactly two other places:

- **as a discrete post-hoc selection** on one parameter. Sec. IV A: "we
  performed separate fits for five discrete values
  b_omega in {0.60, 0.65, 0.70, 0.75, 0.80}, selecting the best value by
  requiring consistency with neutron star tidal deformability constraints from
  GW170817 [156]." So `b_omega = 0.80` IS set by an astrophysical observation —
  but by choosing among five completed fits, not through the likelihood.
- **as a check on the result**, Sec. V D and Fig. 9 (mass–radius against
  PSR J0030+0451, PSR J0740+6620, HESS J1731−347; Lambda_A–Lambda_B against the
  GW170817 posterior).

The conclusion (Sec. VII) restates the calibration set and again lists only
saturation observables, chi-EFT NM pressure, HIC ISM pressure and the nine
hyperon potentials.

## 2. The 18 observables, verbatim

Sec. IV C, "The 18 observables used to constrain the model parameters are:".
The count checks: 9 + 4 + 1 + 2 + 2 = 18.

### 2a. Hyperon single-particle potentials — 9 observables

"The U_Y at 3-momentum |k| = 0 in ISM and NM, calculated in the BHF approach
with baryon–baryon interactions obtained by HALQCD collaboration [125] at
near-physical quark masses (m_pi = 146 MeV, m_K = 525 MeV). This comprises
three iso-multiplet averages in ISM and six individual hyperon species in NM,
with statistical uncertainties ~ ±2 MeV."

Values are Table IV ("Hyperon potentials in ISM and NM at n_B = n_0 vs. results
from [125]. Blank cells in the ISM columns are the same across each
iso-multiplet."), HALQCD column, in MeV:

| # | observable | HALQCD value | kind |
|---|---|---|---|
| 1 | U_Lambda (ISM) | −28.15 ± 2.02 | theory: LQCD (HAL QCD) + BHF |
| 2 | U_Sigma (ISM, iso-multiplet average) | +14.62 ± 1.82 | theory: LQCD + BHF |
| 3 | U_Xi (ISM, iso-multiplet average) | −3.60 ± 2.14 | theory: LQCD + BHF |
| 4 | U_Lambda (NM) | −25.42 ± 1.78 | theory: LQCD + BHF |
| 5 | U_Sigma+ (NM) | +8.24 ± 3.68 | theory: LQCD + BHF |
| 6 | U_Sigma0 (NM) | +15.73 ± 1.70 | theory: LQCD + BHF |
| 7 | U_Sigma− (NM) | +24.86 ± 1.39 | theory: LQCD + BHF |
| 8 | U_Xi0 (NM) | −12.19 ± 1.46 | theory: LQCD + BHF |
| 9 | U_Xi− (NM) | +5.79 ± 2.59 | theory: LQCD + BHF |

Ref. [125] is Kohno et al., *Hyperon single-particle potentials in symmetric and
neutron matter from HAL QCD baryon-baryon interactions*, Phys. Rev. C **110**,
054001 (2024) — already in `docs/eos.bib` as `Kohno2024`.

These are NOT laboratory data. They are a many-body (BHF) calculation on top of
lattice-QCD baryon–baryon potentials. That matters for a sampler: their
"uncertainty" is the statistical error of a calculation, not of a measurement.

### 2b–2e. Everything else — 9 observables

Values are Table III ("List of datapoints, other than hyperon potentials, used
as evidence in the Bayesian analysis"), "Empirical" column:

| # | observable | value | ref. | kind |
|---|---|---|---|---|
| 10 | n_0 [fm^-3] | 0.150 ± 0.010 | [130] | laboratory: PREX parity-violating electron scattering, 208Pb radius |
| 11 | B [MeV] | −15.6 ± 0.6 | [103] | laboratory: nuclear-matter saturation |
| 12 | K [MeV] | 240 ± 20 | [103] | laboratory: giant monopole resonance of 208Pb and 90Zr |
| 13 | S_2 [MeV] | 32.0 ± 1.1 | [131] | laboratory: "fitted to various nuclear data" |
| 14 | M(0.11 fm^-3) [MeV] | 1100 ± 70 | [157] | laboratory: finite-nucleus crossover density |
| 15 | P_NM(0.08 fm^-3) [MeV/fm^3] | 0.472 ± 0.036 | [18] | theory: chi-EFT, N3LO + 3N |
| 16 | P_NM(0.16 fm^-3) [MeV/fm^3] | 2.898 ± 0.404 | [18] | theory: chi-EFT, N3LO + 3N |
| 17 | P_ISM(0.32 fm^-3) [MeV/fm^3] | 19.0 ± 14.3 | [132] | laboratory: heavy-ion-collision data, via a Bayesian EoS analysis |
| 18 | P_ISM(0.56 fm^-3) [MeV/fm^3] | 106.8 ± 22.0 | [132] | laboratory: heavy-ion-collision data, via a Bayesian EoS analysis |

The paper's own wording for each group (Sec. IV C):

- **ISM saturation properties (4 observables).** "The binding energy per nucleon
  B = −15.6 ± 0.6 MeV [103], the incompressibility from GMR of 208Pb and 90Zr
  with K = 240 ± 20 MeV [103], the quadratic symmetry energy at saturation
  S_2 = 32.0 ± 1.1 MeV from Ref. [131] (fitted to various nuclear data), and the
  saturation density n_0 = 0.150 ± 0.010 fm^-3 from Ref. [130] (estimated from
  radius measurements of 208Pb with parity-violating electron scattering)."
- **Finite nucleus crossover density (1 observable).** Eq. (4.2),
  M = 3 n_B d/dn_B [ 9n^2 d^2 B/dn^2 + 18 P/n ], "evaluated at the crossover
  density of 0.11 fm^-3 ~ 0.7 n_0, the average density in the outer extent of
  atomic nuclei where EoSs fitted to finite nuclei agree most strongly, with
  M = 1100 ± 70 MeV [157]."
- **NM pressure (2 observables).** "Ref. [18] used an N3LO + 3N formulation to
  compute the pressure of NM ... over (0.05–0.34) fm^-3 ~ (0.3–2.1) n_0. We
  selected two keypoints to ensure well-behaved low-density matter:
  n_B in {0.08, 0.16} fm^-3."
- **Dense ISM pressure (2 observables).** "we used the pressure series from
  Ref. [132] for ISM (excluding electrons). There, a Bayesian analysis of HIC
  data employed a flexible EoS parameterization ... We selected keypoints at
  n_B in {2, 3.5} n_0 = {0.32, 0.56} fm^-3."

Sec. IV C closes: "DID is not fitted to finite nuclei, and we did not verify how
accurately it describes their measured masses and shapes."

**Kind tally over the 18.** Seven laboratory data: n_0, B, K, S_2, M, and the
two P_ISM heavy-ion points. Eleven theoretical: the nine LQCD+BHF hyperon
potentials and the two chi-EFT P_NM points. **Zero astrophysical.**

## 3. The two exclusion heuristics (Sec. IV B) — hard priors, not observables

A sampler reusing this model should reuse these; they are the paper's own
rejection rules and they are separate from the likelihood.

1. "For any choice of parameters, the saturation density n_0 was calibrated so
   that P(n_0) = 0 in ISM without leptons at T = 0. Some EoS candidates were
   discarded because this could not be achieved. Specifically, we ruled out EoSs
   if n_0 < 0.01 fm^-3 or n_0 > 0.30 fm^-3."
2. "Since the DD couplings in DD-RMFs are not Lorentz-invariant [68], we needed
   to introduce the second exclusion constraint 0 < c_s^2 < 1 at selected
   keypoints. We chose to evaluate this for ISM and NM at c_omega n_0, where
   there is a peak in the speed of sound due to the large positive second
   density derivative of g_omegaN in the transition zone."

Consequence a sampler must not miss: **n_0 is not a free parameter and is not a
constant.** It is recomputed for every parameter point from P(n_0) = 0, and it
then appears inside x = n_B/n_0 in every coupling. The shipped
`n_0 = 0.15880045 fm^-3` is the value at the MLE point (Table III), not a
fixed input.

## 4. Ruling 1 — the hyperon neutron-matter branches are a MODELLING CHOICE

The ticket asks whether the proportionality tying `g^{N(0)}_{iY}` to the nucleon
ratio is a fit result or a modelling choice. **It is a modelling choice**, made
before the fit and used to reduce the parameter count. Two statements settle it,
both in Sec. IV A, both inside the passage that counts the free parameters.

Opening of Sec. IV A: "We constrained all hyperon couplings to fixed ratios
relative to nucleon couplings, and similarly fixed the ratios
g^N_{MN}/g^S_{MN}. As follows from Eq. (2.5), this can be implemented by
adjusting only the saturation values g^{S,N(0)}_{MN}."

And, in the bullet "**Hyperon sigma couplings (3 parameters)**": "The parameters
g^{S(0)}_{sigma Y} for Y in {Lambda, Sigma, Xi}. For convenience of fitting, we
varied these constants directly, not as ratios to g^{S(0)}_{sigma N}. **The
g^{N(0)}_{sigma Y}'s are in the same proportions to g^{N(0)}_{sigma N} as their
ISM counterparts.**"

The sentence is the reason the hyperon sigma sector costs three parameters and
not six. It is not reported as a posterior finding anywhere: it is not in
Table II (which lists only `g^{S(0)}_{sigma Lambda/Sigma/Xi}`,
`g^{S(0)}_{rho Sigma}`, `g^{S(0)}_{rho Xi}` among the hyperon rows), and no
figure shows a posterior over it. `eos/did/parameters.py:79` (`_branch_pair`)
transcribes exactly this sentence, so the repository has it right.

**What this decides.** A hyperon potential measured in symmetric matter
constrains only the S branch, as the ticket says. But the six NM hyperon
potentials WERE fitted (observables 4–9 above), and they were fitted through the
sectors that remain free in neutron matter: the nucleon N branches
`g^{N(0)}_{sigma N}`, `g~^{N(0)}_{omega N}`, `g^{N(0)}_{rho N}`, the SU(3)
number z, and the two isovector hyperon vertices `g^{S(0)}_{rho Sigma}` and
`g^{S(0)}_{rho Xi}`. Sec. VII says so directly: "For the isovector mesons, we
treat hyperon couplings as fit parameters in order to fit the splitting between
Sigma^{±,0}, Xi^{−,0} single-particle potentials in neutron matter."

So for a sampler: a non-symmetric U_Y is legitimate INPUT DATA (the paper used
six such values), but `g^{N(0)}_{sigma Y}` is **not a dial** — it is derived
from `g^{S(0)}_{sigma Y}` and the nucleon ratio. Freeing it is a change of
model, not a change of parameters, and it would need a new justification because
the sigma sector has no SU(3) relation to fall back on (Sec. II B: "Instead of
attempting an SU(3) extension to the scalar sector, we treat g_{sigma i}/
g_{sigma N} as free parameters").

## 5. Ruling 2 — the observable is S_2, the quadratic (beta^2) coefficient

Unambiguous, and it confirms ticket 103's ruling.

- Sec. IV C, ISM saturation properties: "the **quadratic symmetry energy at
  saturation S_2 = 32.0 ± 1.1 MeV** from Ref. [131] (fitted to various nuclear
  data)".
- Table III lists the row as `S_2`, empirical 32.0 ± 1.1, ref. [131], MLE 32.44.
- Sec. VII repeats it: "saturation point observables (n_0 = 0.150 ± 0.010 fm^-3
  [130], B = −15.6 ± 0.6 MeV, K = 240 ± 20 MeV [103], **S_2 = 32.0 ± 1.1 MeV**
  [131])".

The full ISM-to-PNM difference `S` was **not** an observable. It appears only in
Table VI, as the derived row `S − S_2 = −2.72` compared against `1.2 ± 1.5`
[159] — Table VI's caption calls its right-hand column "estimates from
experiment or chi-EFT", i.e. a post-hoc comparison, not evidence. The same is
true of `L_2`, `L − L_2`, `K_sym2`, `K_sym − K_sym2`, `Q` and `X_p^eq(n_0)`:
none is in Table III, so none was fitted. **L in particular was not an
observable**; the paper constrains it only indirectly, noting in the NM-pressure
bullet that P_NM is "related to the slope of symmetry energy
L = 3 n_0 dS/dn_B".

## 6. The 15 fitted parameters, with the paper's prior ranges

Sec. IV A: "With these constraints, there are 15 free parameters fitted to the
18 observables described below", grouped as 3 + 3 + 2 + 2 + 3 + 2 = 15.
Table II gives the ranges (uniform priors) and the MLE.

| parameter | prior range | MLE | posterior 68% C.L. |
|---|---|---|---|
| g^{S(0)}_sigmaN | 6.00 – 11.00 | 8.94873669 | 8.263 +0.713 −0.735 |
| g^{N(0)}_sigmaN | 6.00 – 11.00 | 8.89241948 | 8.094 +0.695 −0.706 |
| a_sigma | 0.00 – 1.00 | 0.16394393 | 0.189 +0.041 −0.032 |
| g^{S(0)}_sigmaLambda | 5.00 – 11.00 | 7.51077621 | 6.203 +0.887 −0.741 |
| g^{S(0)}_sigmaSigma | 3.00 – 9.00 | 6.26418057 | 4.770 +0.999 −0.809 |
| g^{S(0)}_sigmaXi | 1.00 – 7.00 | 6.53781517 | 4.616 +1.309 −1.373 |
| g~^{S(0)}_omegaN | 7.00 – 14.00 | 10.82857726 | 9.703 +1.165 −1.255 |
| g~^{N(0)}_omegaN | 7.00 – 14.00 | 11.00228164 | 9.698 +1.087 −1.133 |
| a_omega | 0.00 – 1.00 | 0.15313180 | 0.172 +0.048 −0.032 |
| z | 0.00 – 2/sqrt(6) | 0.07720445 | 0.194 +0.135 −0.121 |
| g^{S(0)}_rhoN | 0.00 – 6.00 | 3.23020263 | 3.563 +0.400 −0.320 |
| g^{N(0)}_rhoN | 0.00 – 6.00 | 2.59340047 | 2.663 +0.644 −0.541 |
| a_rho | 0.00 – 4.00 | 0.39223762 | 0.212 +0.149 −0.130 |
| g^{S(0)}_rhoSigma | 0.00 – 6.00 | 0.00545444 | 0.787 +0.819 −0.553 |
| g^{S(0)}_rhoXi | 0.00 – 6.00 | 1.11415631 | 2.096 +0.820 −0.729 |

Table II also carries `b_omega = 0.80000000` and `b_rho = 0.40000000` with "—"
in the range column, i.e. present but not sampled — 17 rows, 15 free. These
are the prior ranges a downstream inference should start from.

Note the MLE sits far from the posterior median in several directions
(g^{S(0)}_sigmaXi: MLE 6.54 vs 4.62 +1.31 −1.37). The paper flags the general
hazard: "While DD-RMF posteriors can contain spurious correlations, the optimal
EoS remains reliable if the functional form is well-chosen [155]."

## 7. The parameters fixed a priori, and why

Sec. IV A, in order:

- **c_omega = c_rho = 3.5, d_omega = d_rho = 1.8** (in units of n_0): "In order
  to qualitatively reproduce the density-dependent coupling behavior from
  Ref. [136] and ensure smooth speed of sound profiles reminiscent of those in
  Ref. [132], we have set the transition zone parameters for vector mesons to
  c_{omega,rho} = 3.5 and d_{omega,rho} = 1.8 (in units of n_0) **exempt from
  the Bayesian analysis**." Kind: modelling choice, shaped against a DBHF
  calculation and an HIC-fitted c_s^2 profile.
- **c_sigma = infinity** (and d_sigma thereby irrelevant): "For the sigma meson,
  we did not impose high-density flattening because it tends to introduce the
  instability c_s^2 < 0, which is not associated with appearance of new degrees
  of freedom and we ruled out as spurious. ... To utilize this phenomenological
  freedom, we fixed c_sigma = infinity, which makes d_sigma irrelevant." Kind:
  modelling choice, imposed for stability.
- **b_omega = 0.80**: chosen from the five-value grid by GW170817 tidal
  deformability (Sec. 1 above). Kind: **astrophysical observation**, applied as
  a discrete selection between completed fits.
- **b_rho = 0.40**: "We also set b_rho = 0.40 a priori because low b_rho delays
  hyperon onset that helps to solve both the M_max hyperon puzzle and NS cooling
  part of the hyperon puzzle within this class of models." Kind: modelling
  choice, astrophysically motivated but not fitted to any observation.
- **alpha = F/(D+F) = 1**: Sec. II B gives the SU(6) values
  "theta = arctan(1/sqrt(2)) ~ 35.3° (where phi becomes a pure s-sbar state),
  alpha = 1, and z = 1/sqrt(6)", and the allowed ranges "alpha in [0,1] and
  z in [0, 2/sqrt(6)] [140]", then: "**To simplify the Bayesian analysis, we
  choose to vary z alone, while keeping alpha = 1.**" Kind: theoretical
  symmetry choice (the SU(6) ideal value), held fixed for convenience — so it is
  a legal dial in [0,1] that this fit chose not to turn.
- **tan(theta) = 1/sqrt(2)**: "The vector mixing angle is well-constrained by
  the meson masses as very close to the SU(6) ideal case, so we use ideal mixing
  [141]." Kind: theoretical symmetry, pinned by measured meson masses (PDG).
  Sec. VII repeats "we fix the parameters tan theta = 1/sqrt(2) (the omega–phi
  mixing angle)".
- **e = 1/3**: Sec. II A, under Eq. (2.4): "The tanh(x/e) multiplier also ensures
  that g_Mi is isospin-independent at zero density (necessary to ensure that
  Sigma_t is well-behaved as n_B -> 0), **and we set e = 1/3**." Kind: modelling
  choice; the tanh factor itself is a regularity requirement, the value 1/3 is
  a choice. The paper gives no fit or datum for it.
- **g_rhoLambda = 0**: Sec. IV A, bullet "Hyperon rho couplings (2 parameters)":
  "(g_{rho Lambda} is always zero, since the Lambda has zero isospin)". Kind:
  theoretical identity. Never varied.

## 8. Masses

Sec. II B: "The masses of the mesons are those from DD2Y; they are irrelevant
(except in finite systems and **for calibrating omega and phi couplings with
SU(3)**) because the couplings exclusively appear in density-dependent RMFs as
g_Mi/m_M^2. For the baryons, we use the Particle Data Group mass values [141].
All vacuum masses used for the particles included in the model are shown in
Table I."

Table I: sigma 550., omega 783., phi 1020., rho 763., e− 0.511; p 938.272,
n 939.565, Lambda 1115.683, Sigma+ 1189.37, Sigma0 1192.642, Sigma− 1197.449,
Xi0 1314.86, Xi− 1321.71 (all MeV).

The parenthesis matters for the provenance table: m_omega and m_phi are NOT
inert in this implementation, because the aggregate inversion
g_8 = g~_omegaN / sqrt(A_omega^2 + (m_omega/m_phi)^2 A_phi^2) uses their ratio.
Changing them changes every vector coupling.

## 9. Two paper-internal wrinkles worth recording

1. **15 vs 17.** Sec. IV A says "there are 15 free parameters fitted to the 18
   observables"; Sec. VII says "perform a **17-parameter** Bayesian analysis".
   Table II has 17 rows, of which b_omega and b_rho carry "—" for their range.
   The 15 is the operative number — it is the one with the itemised breakdown —
   and 17 is the row count of Table II. `did.md` and `did.tex` already say
   fifteen, which is right.
2. **K quoted twice, differently.** The likelihood used K = 240 ± 20 MeV [103]
   (Sec. IV C and Table III). Table VI's comparison column instead shows
   K = 230 ± 40 MeV [131]. Two different references; only the first is evidence.

## 10. What the repository had right, and the one thing it had wrong

Everything the ticket listed as already-established checks out against the
paper:

- fifteen fitted numbers — Sec. IV A, and the fifteen `did.md` marks "fitted"
  are exactly the fifteen rows of Table II that carry a prior range;
- `c_sigma = infinity`, `c_M = 3.5`, `d_M = 1.8`, `b_omega = 0.80`,
  `b_rho = 0.40`, `e = 1/3` fixed a priori — Sec. IV A and Sec. II A;
- `alpha = 1`, `tan(theta) = 1/sqrt(2)` as ideal SU(3)/SU(6) mixing — Sec. II B,
  Sec. VII;
- `g_rhoLambda = 0` from I_Lambda = 0 — Sec. IV A;
- the hyperon N-branch tying as an assumption — Sec. IV A (Ruling 1 above);
- the Deltas absent from the paper — Table I ("List of particles included in the
  model") has no Delta, and no section mentions one. The `x_{i Delta}` are
  wholly this repository's extension, as `did.md` already says;
- `n_0` calibrated by P(n_0) = 0 in ISM at T = 0 — Sec. IV B (with the extra
  detail "without leptons", and the [0.01, 0.30] fm^-3 rejection window);
- omega/phi derived from `g~_omegaN` and z — Sec. IV A and Eq. (2.6);
- meson masses from DD2Y, Table I.

**Nothing was found wrong.** One thing looks wrong and is not, so it is recorded
here to stop the next reader from "fixing" it. The repository cites the paper's
equations by a GLOBAL SEQUENTIAL count, while arXiv:2511.15646v1 prints
section-prefixed labels. The two agree:

| repository | paper's printed label | what it is |
|---|---|---|
| Eq. 4 (`couplings.py:16`, `:44`) | (2.4) | the isospin blend, with e = 1/3 |
| Eq. 5 (`couplings.py:21`, `:66`) | (2.5) | the shared density shape F_M(x) |
| Eq. 6 (`couplings.py:151`, `did.md:158`, `did.tex:243`) | (2.6) | the SU(3) vector relations |
| Eqs. 10–11 (`couplings.py:34`) | (2.10)–(2.11) | Sigma^r and Sigma^t |
| Eq. 12 (`nmp.py:249`) | (2.12) | the single-particle potential U_i |
| Eq. 52 (`couplings.py:188`, `parameters.py:87`, `:113`, `did.md:172`, `:861`) | (4.1) | the aggregated omega–phi strength |
| Eq. 53 (`nmp.py:127`) | (4.2) | the crossover derivative M |

The arithmetic confirms it: Sec. II runs (2.1)–(2.19) = 1–19, Sec. III runs
(3.1)–(3.32) = 20–51, so Sec. IV's two equations are 52 and 53 — consecutive,
exactly as the repository has them. "Eq. 52" is therefore correct in the
repository's own scheme, and no citation was changed by this ticket. (The scheme
does cost a reader holding the PDF a manual count; unifying it on the printed
labels would be a separate documentation sweep across four `.py` files and both
documents, out of scope here.)

The subagent audit of `parameters.py`, `couplings.py`, `nmp.py` and `species.py`
confirmed all twelve claims, with two locational corrections to the ticket's own
wording, neither of them a repository error:

- the branch-tying quote is in the `Parameters` class docstring at
  `parameters.py:77-82`, not in `_branch_pair`'s docstring; `_branch_pair`
  itself is at `parameters.py:156-166`. The source attributes the rule to
  "Section IV.1", which is Sec. IV A in the published labelling.
- `e = 1/3` is not a `Parameters` field. It is the module constant
  `couplings.E_ISOSPIN`, threaded as a default keyword through `blend`,
  `dblend_dx`, `dblend_dbeta` and `coupling`, and `Parameters.couplings_at`
  never passes it — so it is not reachable from a parameter set at all. That is
  the right encoding for a number nobody may sample, and the provenance table
  says "held fixed" for it.

## 11. The provenance table as delivered

The gate table added to `eos/did/did.md` and `eos/did/did.tex` has four columns
— parameter, what fixed it, kind of constraint, and whether an inference may
vary it — with these kinds:

- *theoretical identity* — cannot be varied at all (g_rhoLambda = 0);
- *theoretical symmetry* — a symmetry choice, legally variable but not by this
  fit (alpha, tan theta);
- *modelling choice* — fixed a priori for shape, stability or phenomenology
  (c_M, d_M, c_sigma, b_rho, e, the branch tying, the Delta ratios);
- *Bayesian posterior* — the 15 sampled by MultiNest over the 18 observables;
- *astrophysical observation* — b_omega alone, and by discrete selection;
- *derived* — n_0 (recomputed per point), the omega/phi vertices, the hyperon
  N branches;
- *measured particle property* — baryon and lepton masses (PDG), meson masses
  (DD2Y / PDG).

Delivered as:

- `eos/did/did.md`, new subsection "Provenance: what fixed each number" inside
  "The parameter set", between the masses/degeneracies table and "Three routes
  to a parameter set". It carries the provenance table, the eighteen
  observables with their values and kinds, the two exclusion heuristics, and the
  two rulings.
- `eos/did/did.tex`, new `\subsection{Provenance: what fixed each number}`
  (`\label{sec:provenance}`) in the same position, with
  `Table~\ref{tab:provenance}` and `Table~\ref{tab:evidence}` and the same two
  rulings as `\paragraph`s. The existing parameter table gained
  `\label{tab:params}` and a forward pointer, because the new text references
  it. Both tables use `booktabs`, already in the preamble; no package was added.
  `pdflatex` runs clean (exit 0, no errors, no undefined references on the
  second pass, 15 pages) and introduces no overfull box beyond the two the file
  already had at HEAD.

No `.py` file was touched, and no test was run.
