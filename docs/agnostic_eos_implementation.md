# Agnostic equations of state: what they are, and how they would fit `eos`

A short literature review of the model-agnostic dense-matter parametrizations in
current use, and a concrete proposal for adding them to this repository.

Written 2026-08-28. Citations are given with arXiv numbers; check them against
`docs/eos.bib` before any of them goes into a `.tex`.

---

## 0. The one thing to settle first

Every model already in `eos/` is a **microphysical** model: it has a species
list, conserved charges, and a solver that finds chemical potentials. That is
what makes §3's modes and §4's species flags meaningful.

Almost every agnostic model is a **barotrope**: a curve `P(eps)` or `P(n_B)` and
nothing else. It has no species, no `Y_C`, no `mu_C`, no leptons — not because
the author left them out, but because the whole point of the parametrization is
to be agnostic about *what the matter is made of* while spanning the space of
curves that a microphysical model could have produced.

So the answer to "can they simulate hyperons, quarks, deltas, colour
superconductivity?" splits in two, and the split is the most important thing in
this document:

| | composition? | can it *be* hyperons/quarks? |
|---|---|---|
| **Barotropic** (piecewise polytrope, spectral, sound-speed, GP) | none | it can reproduce the **thermodynamic footprint** a new degree of freedom leaves — a softening, a `c_s^2` plateau, a jump in `eps` at fixed `P` — but it cannot tell you *which* one, and cannot report a single fraction |
| **Metamodel** (Margueron et al.) | full nucleonic (n, p, e, mu) | yes for nucleons, and it extends to hyperons/deltas exactly as an RMF does, because it is a real density functional |

That is the whole answer, and §5 below spells out the consequences for the
uniform API.

---

## 1. The families

### 1.1 Piecewise polytropes

**Read, Lackey, Owen & Friedman**, PRD 79, 124032 (2009), arXiv:0812.2163.

Split the density axis into segments; on each one

    P(n) = K_i n^{Gamma_i},   eps(n) from the first law d(eps/n)/d(1/n) = -P

with `K_i` fixed by continuity of `P` at each joint. The canonical form is four
parameters: `{p_1, Gamma_1, Gamma_2, Gamma_3}` above a fixed crust, with the
segment boundaries pinned at `10^{14.7}` and `10^{15}` g/cm^3. Read et al.
showed 3–4 parameters reproduce ~34 tabulated EoSs to a few percent in radius.

- **Why it is used:** trivial to evaluate and to invert; the standard prior in
  early GW EoS inference; `Hebeler, Lattimer, Pethick & Schwenk`, ApJ 773, 11
  (2013), arXiv:1303.4662, made it the standard way to extend chiral-EFT bands.
- **Known defect:** `c_s^2 = dP/deps` is **discontinuous** at every joint. That
  is unphysical, and it biases anything sensitive to `c_s^2` (mode frequencies,
  sound-speed inference).
- **Fix:** *generalized* piecewise polytropes — **O'Boyle, Markakis,
  Stergioulas & Read**, PRD 102, 083027 (2020), arXiv:2008.03342 — an ansatz
  continuous in `P`, `eps` **and** `c_s`, at the cost of one extra parameter per
  segment. If this repo implements one polytropic form, it should be this one.

### 1.2 Spectral representation

**Lindblom**, PRD 82, 103011 (2010), arXiv:1009.0738; **Lindblom & Indik**, PRD
86, 084003 (2012) and PRD 89, 064003 (2014).

Expand the *logarithm of the adiabatic index* in the logarithm of the pressure:

    ln Gamma(p) = sum_{k=0}^{N-1} gamma_k [ln(p/p_0)]^k

and integrate the first law to get `eps(p)` and `n_B(p)`. Four `gamma_k` fit
realistic EoSs to ~0.5% — better accuracy per parameter than piecewise
polytropes, and `c_s^2` is smooth by construction.

- **Why it is used:** the LIGO/Virgo default alongside piecewise polytropes;
  fewer parameters for the same fidelity.
- **Known defect:** causality (`c_s <= 1`) is not built in — it has to be
  imposed as a prior cut, which distorts the sampled space. Lindblom's *causal*
  spectral representation (PRD 97, 123019 (2018), arXiv:1804.04072) fixes this
  by expanding a variable whose range enforces `0 <= c_s^2 <= 1` identically.

### 1.3 Sound-speed parametrizations

Parametrize `c_s^2` directly, then integrate. This is now the dominant family,
because `c_s^2(n)` is the quantity that actually carries the physics signal
(conformal limit, phase transitions) and because causality is a box constraint
on the parameters rather than a derived condition.

Variants, all in current use:

- **CSS — constant speed of sound.** **Alford, Han & Prakash**, PRD 88, 083013
  (2013), arXiv:1302.4732. Three parameters: a transition density `n_trans`, an
  energy-density jump `Delta eps`, and a constant `c_s^2` on the high-density
  branch (`P = c_s^2 (eps - eps_0)`). This is the **minimal first-order phase
  transition model** and the reference tool for classifying hybrid stars
  (absent / connected / disconnected twin branches). Small, exactly solvable,
  and the single most useful agnostic model for the phase-transition questions
  in this repo.
- **Random segments in `n_B`.** **Tews, Carlson, Gandolfi & Reddy**, ApJ 860,
  149 (2018), arXiv:1801.01923: piecewise-linear `c_s^2(n)` above a chiral-EFT
  band, with segment-wise random draws. Established that `c_s^2` must exceed
  the conformal value 1/3 to support 2 M_sun (the point **Bedaque & Steiner**,
  PRL 114, 031103 (2015), arXiv:1408.5116, made first).
- **Gaussians.** **Greif, Raaijmakers, Hebeler, Schwenk & Watts**, MNRAS 485,
  5363 (2019), arXiv:1812.08188: `c_s^2(n)` as a smooth logistic background plus
  Gaussian bumps. A bump is a soft, tunable stand-in for a phase transition.
- **Piecewise-linear in `mu_B`.** **Annala, Gorda, Kurkela, Nättilä &
  Vuorinen**, Nature Physics 16, 907 (2020), arXiv:1903.09121. Interpolates
  between chiral EFT at low density and perturbative QCD at very high density;
  a near-vertical `c_s^2` segment reproduces a first-order transition. This is
  the framework behind the "quark-matter cores in massive stars" claims.

### 1.4 Metamodel (meta-modelling, MM)

**Margueron, Hoffmann Casali & Gulminelli**, PRC 97, 025805 and 025806 (2018),
arXiv:1708.06894 and arXiv:1708.06895.

The odd one out, and the one that matters most for this repository. It is a
**nucleonic density functional whose parameters ARE the nuclear-matter
parameters**. Write the energy per particle as a Taylor series in

    x = (n_B - n_sat) / (3 n_sat)

separately for the isoscalar and isovector channels,

    e_sat(x) = E_sat + (1/2) K_sat x^2 + (1/6) Q_sat x^3 + (1/24) Z_sat x^4 + ...
    e_sym(x) = E_sym + L_sym x + (1/2) K_sym x^2 + (1/6) Q_sym x^3 + ...

add a relativistic-Fermi-gas kinetic term with density-dependent effective
masses controlled by `kappa_sat` and `kappa_sym`, and multiply the correction by
`u_alpha(x) = 1 - (-3x)^{N+1-alpha} exp(-b n_B/n_sat)` so the expansion goes to
the right limit at `n_B -> 0`.

- **Why it is used:** it is agnostic *within* the nucleonic hypothesis. It spans
  the space of existing Skyrme/RMF functionals to sub-percent accuracy while
  being parametrized in exactly the quantities nuclear experiment constrains.
  It is the bridge between `eos/dd2`-style microphysics and the barotropic
  families, and the natural prior for inference that wants to use nuclear data.
- **Composition:** full. Protons, neutrons, electrons, muons, beta equilibrium,
  `Y_p(n_B)`, symmetry energy, effective masses, finite `T` through the kinetic
  term. Every §3 mode is meaningful for it.
- **Extensions:** hyperons and deltas are added the same way they are added to
  an RMF — a new species with its own coupling ratios — and the "nucleonic
  hypothesis" framing (**Somasundaram, Margueron et al.**, arXiv:2109.09675) is
  precisely about testing when that hypothesis breaks.
- **In this repo it would have a `nmp.py` where the forward map is nearly the
  identity** — the parameters *are* `{n_sat, E_sat, K_sat, Q_sat, E_sym, L_sym,
  K_sym, ...}`. That makes it the natural cross-check for `dd2/nmp.py` and
  `sfho/nmp.py`: same NMPs in, compare the EoS out.

### 1.5 Non-parametric: Gaussian processes and neural networks

**Landry & Essick**, PRD 99, 084049 (2019), arXiv:1811.12529; **Essick, Landry &
Holz**, PRD 101, 063007 (2020); **Legred, Chatziioannou, Essick, Han & Landry**,
PRD 106, 023019 (2022), arXiv:2106.05313.

A Gaussian process over an auxiliary variable (typically
`phi = ln(c^2/c_s^2 - 1)`, whose range enforces causality and stability
identically), conditioned on a set of tabulated EoSs to set the mean and
correlation length. Drawing from the GP gives an EoS. Modified kernels can be
made to produce phase-transition-like features (arXiv:2302.07978), and the
framework now runs unified from the crust to pQCD densities (arXiv:2505.13691).
Deep-network variants exist (arXiv:2305.03323).

**This is not a model — it is a prior over models.** It has no parameter
dataclass in the sense of §6; its "parameters" are hyperparameters of a
stochastic process plus a random seed. It belongs to whatever does the
inference, not to `eos/`. What `eos/` would owe it is the ability to *consume* a
drawn `c_s^2(n_B)` curve — which is exactly the barotrope machinery of §4 below.

### 1.6 Finite temperature, agnostically

Barotropic models are cold by construction. Two prescriptions are in use:

- **The `Gamma`-law thermal index:** `P_th = (Gamma_th - 1) eps_th`, with
  `Gamma_th ~ 1.5–2` constant. Universal in merger simulations, and *known to be
  wrong* where new degrees of freedom appear — the thermal index dips at
  intermediate density once hyperons are abundant (arXiv:2211.04855).
- **`M*` framework:** **Raithel, Özel & Psaltis**, ApJ 875, 12 (2019),
  arXiv:1902.10735. Adds a degenerate thermal contribution with a
  density-dependent effective mass, reproducing the true `Gamma_th(n)` shape far
  better than a constant. This is the one to implement if finite-`T` agnostic
  tables are wanted.

The metamodel needs neither: its finite-`T` behaviour follows from the kinetic
term, like any density functional.

---

## 2. What each family can and cannot represent

Answering the hyperons/quarks/deltas/CSC question concretely.

| Feature | PP / spectral / `c_s^2` / GP | CSS | Metamodel |
|---|---|---|---|
| species fractions (`Y_p`, `Y_Lambda`, `Y_s`) | **no** | no | yes (nucleonic; extensible) |
| beta equilibrium as a *solved* condition | no — asserted, not solved | no | yes |
| `fixed_YC`, `fixed_YC_YS` (simulation tables) | **no** | no | `fixed_YC` yes; `fixed_YC_YS` no (no strangeness) |
| neutrino trapping, `Y_Le` | no | no | yes |
| first-order transition | as a `c_s^2 -> 0` plateau / `eps` jump; unlabelled | **yes, explicitly** — that is the model | via an added quark branch, i.e. `eos/mixed` |
| hyperons / deltas | only as a softening | no | yes, as new species with coupling ratios |
| colour superconductivity | **no** — a CFL gap changes `n_S`, `c_s^2` and the pairing pressure in ways a `P(eps)` curve cannot label | no | no |
| composition g-mode (`astro/gmode`) | **`N^2 = 0` identically** — frozen and equilibrium speeds coincide when there is no composition | 0 | yes |
| `eos_response` susceptibilities `chi_ab` | no (`chi_BB` only) | no | yes |
| finite `T` | only via a bolted-on prescription (§1.6) | same | native |

The two rows in bold are the ones to remember: **a barotropic agnostic model
cannot report a composition, and therefore cannot support a composition g-mode,
a fixed-`Y_C` simulation table, or any `chi_ab` beyond `chi_BB`.** It is not a
gap to be filled later; it is what "agnostic" means.

What barotropes *are* good for, and what nothing else in the repo does: spanning
the space of curves consistent with data, so that "how much does this conclusion
depend on the model?" becomes an answerable question. That is exactly the use
case §6.3 (Bayesian inference) is being designed for.

---

## 3. Proposal

Two packages, not five, and one shared module. In repository order:

### 3.1 `eos/general/barotrope.py` — the only genuinely new machinery

Every barotropic family reduces to the same three steps, so they are written
once, in the layer both models and `astro/` may import (§1, §7):

1. **Integrate the first law.** Given `c_s^2(n_B)` (or `P(n_B)`, or
   `Gamma(P)`), integrate

       d eps / d n_B = (eps + P)/n_B,    dP/d n_B = c_s^2 * d eps/d n_B

   from a matching point to get `eps(n_B)`, `P(n_B)` and `mu_B = (eps+P)/n_B`.
   `scipy.integrate.solve_ivp` on a log-density grid; nothing exotic.
2. **Splice a low-density branch.** Match to a crust table, or to a chiral-EFT
   band, or to any table this repo already produces — `EOSTable_for_TOV` in
   `general/state.py` is already the right input type, so `eos/dd2` output can
   be the low-density half of an agnostic high-density extension.
3. **Gate.** `P` non-decreasing and `0 <= c_s^2 <= 1`, per §8, at the point the
   table is delivered.

Everything else in this proposal is parameters feeding step 1.

### 3.2 `eos/agnostic/` — the barotropic families, one package

`Parameters` (§13's mandatory name) is a small hierarchy sharing one method,
`cs2(n_B)`, plus a named low-density branch and an optional `Transition`:

    parameters.py   PiecewisePolytrope   {p_1, Gamma_1..3} or the generalized form
                    Spectral             {gamma_0..3, p_0}
                    SoundSpeed           segment / Gaussian / linear-in-mu variants
                    Transition           {n_trans, delta_eps, cs2_high}  -- CSS
                    each with .default() reproducing the published fit
    species.py      SpeciesFlags where every flag has ONE legal value and False
                    is the only one -- §4's "a flag with only one legal value
                    RAISES on the other and is a STATEMENT about the model",
                    the `enjl` precedent applied to a model with no species
    solver.py       calls general/barotrope.py; check_mode() refusing every mode
                    but `beta_eq_neutrinoless`, with a message saying why
    table.py        the density sweep (no warm start needed -- it is a quadrature)
    api.py          eos_point / eos_table / eos_response
    verify/         first-law consistency (integrate P -> eps, differentiate
                    back), causality, the CSS twin-branch classification against
                    Alford-Han-Prakash Fig. 2, and the §8 delivery gate
    agnostic.tex/.md

Design points worth arguing about before coding:

- **One package, not three.** Piecewise polytrope, spectral and sound-speed are
  the same job — a function of density integrated through the first law — and
  §13 says two models with one job carry one name. The cost is that
  `Parameters` becomes three dataclasses in one module rather than one. If you
  would rather keep §5 strictly (`eos/pwpoly/`, `eos/spectral/`, `eos/csound/`,
  each with its own `Parameters`), the shared module of §3.1 makes the
  duplication small — but it is three `api.py` files that differ only in which
  parameter class they import, which is the drift §13 is trying to prevent.
- **The mode refusals are the honest part.** This model does not *solve* beta
  equilibrium; it asserts a curve that is claimed to be the result of one.
  `solve_beta_eq_neutrinoless` returning a point with no lepton fractions and no
  `mu_C` is truthful; a `fixed_YC` that silently ignores `Y_C` would not be.
  Every other mode raises, and `docs/DEFERRED.md` records it as *structural*,
  not as an unfinished item.
- **It cannot be an `eos/mixed` phase.** The phase-adapter contract (§5) needs
  `(mu_B, mu_C, mu_S, T) -> PhaseThermo`; a barotrope has no `mu_C` or `mu_S`.
  A phase transition inside an agnostic model is therefore the CSS `Transition`
  field above (a Maxwell construction in one variable), never `eos/mixed`. Worth
  stating in the `.tex` so nobody tries.
- **Finite `T` is deferred**, with the Raithel `M*` prescription named in
  `DEFERRED.md` as the intended route if it is ever wanted.

### 3.3 `eos/meta/` — the metamodel, a full §5 model

This one earns the complete layout, because it is a real density functional
with a composition:

    parameters.py   {n_sat, E_sat, K_sat, Q_sat, Z_sat, E_sym, L_sym, K_sym,
                     Q_sym, Z_sym, kappa_sat, kappa_sym, b}, plus named sets
                     reproducing published functionals (SLy4, DD2, ...)
    species.py      nucleons always; muons a real flag; hyperons/deltas as the
                    natural extension (start with them raising, per §4)
    thermodynamics.py  e_sat(x) + delta^2 e_sym(x) + the Fermi-gas kinetic term
                       with the effective masses; leptons from
                       general/thermodynamics_leptons.py
    solver.py       beta_eq_neutrinoless, beta_eq_neutrino_trapped, fixed_YC;
                    fixed_YC_YS raises (no strangeness without hyperons)
    nmp.py          the forward map is nearly the identity and the inverse map
                    exactly is -- which makes this the cleanest possible cross-
                    check on dd2/nmp.py and sfho/nmp.py
    table.py, api.py, responses.py, verify/, meta.tex/.md

The `nmp.py` point is the strongest argument for building this one: it turns
"do our NMP inversions agree?" into a two-line test, and it gives the repo a
model whose parameters are directly the quantities nuclear experiment measures.

### 3.4 Gaussian processes: not now

A GP is a prior, not a model (§1.5). If inference moves into this repo, the
draw is a `c_s^2(n_B)` array and `general/barotrope.py` already consumes it.
Nothing to build until then.

---

## 4. Suggested order

1. `general/barotrope.py` + its `verify` entry (first law round-trip). Small,
   and everything else depends on it.
2. `eos/agnostic` with **CSS only**. Three parameters, closed form, immediately
   useful for the phase-transition questions the repo already asks, and it
   validates the whole shape — mode refusals, empty `SpeciesFlags`, the §8 gate
   — on the smallest possible model.
3. Add `SoundSpeed` (segments and Gaussians) and `Spectral` to the same package.
   Piecewise polytrope last, and in the generalized (continuous-`c_s`) form
   only; the original is worth having only for reproducing published GW priors.
4. `eos/meta`, as a full model, once the NMP cross-check is worth the effort.

Steps 1–2 are a couple of days' work. Step 3 is mostly parameters. Step 4 is a
model-sized job comparable to `did` or `ccdm`.

---

## Sources

Piecewise polytropes: arXiv:0812.2163, arXiv:1303.4662, arXiv:2008.03342.
Spectral: arXiv:1009.0738, arXiv:1804.04072.
Sound speed: arXiv:1302.4732, arXiv:1408.5116, arXiv:1801.01923,
arXiv:1812.08188, arXiv:1903.09121.
Metamodel: arXiv:1708.06894, arXiv:1708.06895, arXiv:2109.09675.
Non-parametric: arXiv:1811.12529, arXiv:2106.05313, arXiv:2302.07978,
arXiv:2505.13691, arXiv:2305.03323.
Finite T: arXiv:1902.10735, arXiv:2211.04855.
Reviews touching all of the above: arXiv:2201.06791 (implicit correlations
between parametric models), arXiv:2208.03026 (sound-speed inference),
arXiv:2407.11153 (dense matter review), arXiv:2507.03232 (semiparametric,
nuclear-physics informed).
