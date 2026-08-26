# vMIT — the MIT bag model with a repulsive vector interaction

`vmit.tex` is the same description written for LaTeX, with the bibliography;
this file carries the same physics in plain text. Either one alone is enough to
reproduce the model. Where a document and the source differ, **the source
decides**.

**Model.** Three quark flavours in a bag of constant energy density `B`,
interacting through a flavour-blind isoscalar-vector field. Chodos et al.,
PRD 9, 3471 (1974) for the bag; the vector term in the form used by Gomes
et al., ApJ 877, 139 (2019) and Constantinou et al., PRD 104, 123032 (2021)
and PRD 107, 074013 (2023):

    L = sum_q qbar [ gamma_mu (i d^mu - g_V V^mu) - m_q ] q
        - 1/4 V_munu V^munu + 1/2 m_V^2 V_mu V^mu - B

with `V_munu = d_mu V_nu - d_nu V_mu`. The quark masses are the *current*
masses and are parameters: unlike an NJL model there is no scalar condensate
and no gap equation, so `m_q` is an input and not a solved quantity. One
coupling `g_V` serves all three flavours.

In uniform matter at rest only `V^0` survives and its equation of motion is
algebraic — the only self-consistency in the model:

    m_V^2 V^0 = g_V sum_q n_q
      =>   V = g_V V^0 = (g_V^2/m_V^2) sum_q n_q = a hbar c (n_u + n_d + n_s)

The coupling is carried as the single combination `a = g_V^2/m_V^2` in fm^2;
with the densities in fm^-3 the product `a sum_q n_q` is an inverse length, and
`hbar c = 197.3269804 MeV fm` converts it to MeV. `g_V` and `m_V` are **not
separately identifiable at mean-field level** — only the ratio enters `V` and
everything downstream — so the code never carries them apart, and an inference
run over the quark sector varies `a`.

Because `V` couples to quark *number* and is flavour blind, it shifts all three
potentials equally. What enters the Fermi integrals is the effective (kinetic)
potential

    mu_eff_q = mu_q - V

and the mean-field problem is the fixed point `n_q = n_q(mu_eff_q, T, m_q)`
with `V` as above. Note what `mu_eff_q` does NOT contain: there is **no
rearrangement self-energy**, because `a` is a constant rather than a function
of the density. A density-dependent coupling would add one; here `Sigma^R = 0`
identically.

Since the same `V` is subtracted from all three, the differences are untouched:

    mu_eff_u - mu_eff_d = mu_u - mu_d = mu_C
    mu_eff_s - mu_eff_d = mu_s - mu_d = mu_S

so the isospin and strangeness structure of a state lives entirely in the
physical potentials and the vector field carries none of it. Only the baryon
direction is shifted, `mu_B_eff = mu_B - 3V`.

## Conventions

Every public boundary is fm-based: `n` in fm^-3, `T` and `mu` in MeV, `P` and
`eps` in MeV/fm^3. Natural units stay inside the physics modules; the factor
`(hbar c)^3` is what carries MeV^3 to fm^-3 and MeV^4 to MeV/fm^3, and it
appears explicitly in every integral below.

`C` is the electric charge of strongly-interacting matter ONLY — leptons are
excluded and enter through the separate condition of total electric neutrality.
**Strangeness is S = +1 per s quark**, the opposite of the PDG sign and the
repository's convention throughout. Fractions are always relative to `n_B`.

## Parameters

Four numbers set the equation of state. There is no scalar sector and no
density dependence, which is what makes the model cheap enough to scan and why
`B` and `a` are the two axes a hybrid parameter study moves along. They live in
a frozen `Parameters` dataclass and are ALWAYS an argument to every entry
point, never module state.

| symbol | code name | value | meaning |
|---|---|---|---|
| — | `name` | `"vMIT_default"` | label carried into table headers and legends |
| `m_u` | `m_u` | 5.0 MeV | current up-quark mass |
| `m_d` | `m_d` | 7.0 MeV | current down-quark mass |
| `m_s` | `m_s` | 150.0 MeV | current strange-quark mass |
| `a` | `a` | 0.2 fm^2 | vector coupling `g_V^2/m_V^2` |
| `B^(1/4)` | `B4` | 180.0 MeV | bag constant, as its fourth root |

`Parameters.default()` returns exactly this set. `B` is stored as its fourth
root because that is the form it is quoted in; the property `Parameters.B`
returns `B = (B^(1/4))^4` in MeV^4, and the single place that divides by
`(hbar c)^3` to reach MeV/fm^3 is the bag term. At `B^(1/4) = 180 MeV`,
`B/(hbar c)^3 = 136.63 MeV/fm^3`.

**What these numbers are fitted to, and what they are not.** The masses are
current masses of the order of the Particle Data Group values (2024); because
there is no chiral condensate to dress them they enter the Fermi integrals
directly, and `m_s` is the only one large enough to matter — at `m_u = 5 MeV`
and `mu_eff_u >~ 300 MeV` the light flavours are ultra-relativistic to better
than 5e-4 in `n`. The pair `(B^(1/4), a)` is **not a published fit**. vMIT's
parameters are what a hybrid study scans, and which pair is right depends on
the hadronic model the quark phase is paired with: `B` sets where deconfinement
becomes energetically possible, `a` sets how stiff the quark branch is above
it, which is how a hybrid star reaches two solar masses. The values above are
the working set of this repository and the set `test/baseline` is frozen at;
`Parameters(B4=..., a=..., m_s=...)` returns a parametrization with any
subset changed and the rest left at the table. There is no scan driver in the
library: parameters are arguments, so a sweep over `(B4, a, m_s)` is a loop in
the caller that builds one parametrization per sample and calls the mixed
engine (`eos.mixed.eos_table(..., vmit_params=...)`) once on each. The literature values these sit
near are `B^(1/4) ~ 180 MeV` and `a` in 0–0.3 fm^2, as used by Gomes et al. and
Constantinou et al., who write the same vector term as `G_V` or `g_V^2/m_V^2`.

**Limits, for orientation.** `a = 0` removes the vector term and leaves the
original bag model of Chodos et al.; `B^(1/4) = 0` removes the bag. Both are
reachable, and one of them is a `verify/` check: with `a = 0` and `B = 0` the
solved state must equal the Fermi integrals evaluated at the physical
potentials themselves, asserted at 1e-10.

**Three routes to a parameter set.** CLAUDE.md section 6 makes model
parameters arguments, so all three have to exist. *By name:*
`Parameters.default()` is the working set above, and
`Parameters.named('vMIT_default')` takes it by name. vMIT ships exactly one
set, so the map has a single entry; it exists so that a caller sweeping
parameter sets need not know which models happen to have more than one. *A new
set:* every field carries a default, so `Parameters(B4=..., a=...)` names only
what changes; the dataclass is frozen, so `dataclasses.replace` is how a set
already in hand is modified, and there is no setter and no mutating helper.
*From nuclear-matter parameters:* no route, and none is missing -- vMIT has no
nuclear sector, so there is no `nmp.py` and nothing to invert.

## Single-flavour thermodynamics

Each flavour is a free Fermi gas of mass `m_q` and degeneracy

    g = 2 (spin) x 3 (colour) = 6

evaluated at `mu_eff_q` with antiparticles included. The code takes `g` from
the shared particle table rather than writing 6.

### Finite temperature

With `f(x) = [1 + exp(x/T)]^-1` and `E_k = sqrt(k^2 + m_q^2)`:

    n_q   = g/(2 pi^2 (hbar c)^3) INT_0^inf dk k^2 [ f(E_k - mu_eff_q)
                                                   - f(E_k + mu_eff_q) ]

    P_q   = g/(6 pi^2 (hbar c)^3) INT_0^inf dk (k^4/E_k) [ f(E_k - mu_eff_q)
                                                        + f(E_k + mu_eff_q) ]

    eps_q = g/(2 pi^2 (hbar c)^3) INT_0^inf dk k^2 E_k [ f(E_k - mu_eff_q)
                                                      + f(E_k + mu_eff_q) ]

`n_q` is the NET density, particles minus antiparticles — that is what is
conserved, what sources `V`, and what enters the charges below.

The entropy density is not integrated separately. It comes from the three above
through the single-gas Euler identity

    s_q = (eps_q + P_q - mu_eff_q n_q) / T

which is exact rather than a convenience: it equals, term by term, the entropy
integral

    s_q = g/(2 pi^2 (hbar c)^3) INT_0^inf dk k^2
            sum_(+-) [ (x_+-/T) f(x_+-) + ln(1 + exp(-x_+-/T)) ],
          x_+- = E_k -+ mu_eff_q

Note the identity is written with `mu_eff_q`, not `mu_q`: it is the identity of
the KINETIC gas, which is the gas the integrals describe. The step back to the
physical potentials is taken once, for the totals, under "the Euler relation".

These integrals are NOT implemented in this subpackage. They come from
`eos.general.fermi_integrals`, which evaluates them through the
Johns–Ellis–Lattimer analytic approximation (1996), uniformly valid from the
degenerate to the non-degenerate limit and exact at T = 0. They are written out
here anyway: a description that leaves its ideal-gas integrals to a citation is
not one a reader can reproduce the model from.

### Zero temperature

The antiparticle terms vanish, the occupations become step functions, and the
flavour is filled to

    k_F,q = sqrt(mu_eff_q^2 - m_q^2)   if mu_eff_q > m_q, else 0
    E_F,q = sqrt(k_F,q^2 + m_q^2) = mu_eff_q

and the three close in elementary functions:

    n_q   = g k_F^3 / (6 pi^2 (hbar c)^3)

    eps_q = g/(16 pi^2 (hbar c)^3) [ k_F (2 k_F^2 + m_q^2) E_F
                                     - m_q^4 ln((k_F + E_F)/m_q) ]

    P_q   = g/(48 pi^2 (hbar c)^3) [ k_F (2 k_F^2 - 3 m_q^2) E_F
                                     + 3 m_q^4 ln((k_F + E_F)/m_q) ]

    s_q   = 0

These satisfy `eps_q + P_q = mu_eff_q n_q`, which is the identity above at
`s_q = 0`; the `T -> 0` limit of the finite-T forms is the same, and
`eos.general.verify` asserts the agreement.

The threshold is what makes the strange flavour appear at a finite density
rather than at zero: with `m_s = 150 MeV` a state with `mu_eff_s < 150 MeV` has
`n_s = 0` exactly at T = 0, and the onset is where `mu_eff_s` crosses `m_s`. At
T > 0 there is no threshold — the exponential tail populates the flavour at any
`mu_eff_s` — which is why the cold start below treats the strange fraction as a
function of both T and `n_B`.

### The scalar density, and why it is not used

The shared integral routine also returns a scalar density
`rho_s,q = (eps_q - 3 P_q)/m_q`, the trace identity of the energy-momentum
tensor for one gas. This subpackage discards it, and the reason is structural
rather than an oversight: a scalar density is what a scalar field couples to,
and vMIT has no scalar sector. The masses in the Lagrangian are current masses,
not Dirac effective masses; nothing varies with `rho_s` and there is no gap
equation for it to source.

This matters for reading the code and the tables, because **`n_s` in this
package is NOT a scalar density**: it is the strange-quark number density, the
`s` of `u, d, s`. The repository-wide identity `n_s = (eps - 3P)/m*` has no
meaning here — there is no `m*` — and the symbol is free to mean the flavour
density, which is what the charges consume.

## The vector field, the bag, and the totals

Beyond the kinetic sum, two terms:

    P_V = eps_V = 1/2 a hbar c (n_u + n_d + n_s)^2
    P_B = -B/(hbar c)^3          eps_B = +B/(hbar c)^3

so the strongly-interacting sector totals

    P_matter   = sum_q P_q + P_V + P_B
    eps_matter = sum_q eps_q + eps_V + eps_B
    s_matter   = sum_q s_q

**The two terms that differ between P and eps differ in opposite ways, and this
is the whole structure of the model.** The vector field enters both with the
SAME sign, `eps_V = +P_V`: it is a repulsion, it raises the pressure at fixed
density, it stiffens the quark branch. The bag enters with OPPOSITE signs,
`eps_B = -P_B = B/(hbar c)^3`: it costs energy to make a bag, and it holds the
pressure negative — the matter unbound — until the kinetic and vector terms
have paid for it, which is what deconfinement at a finite density means here.
Neither contributes to `s`: `V` is a mean field with no entropy of its own and
`B` is a constant. `verify/` asserts `eps_B = -P_B` and `eps_V = +P_V` at
1e-14.

### Leptons, photons and what a point reports

The above is the matter sector alone. A solved point adds, from
`eos.general.thermodynamics_leptons`, whichever of these the mode and the flags
call for:

- **electrons** (with positrons) at `mu_e`, a Fermi gas of mass
  `m_e = 0.511 MeV` and degeneracy 2, through the same integrals. Present
  wherever the mode has a lepton condition: both beta-equilibrium modes always,
  and the two fixed-fraction modes when `leptons=True`.
- **electron neutrinos** (with antineutrinos) at `mu_nue`, massless, degeneracy
  1, present only in `beta_eq_neutrino_trapped`.
- **photons**, when `photons=True`: massless bosons at `mu = 0`, degeneracy 2,

      P_gamma   = (pi^2/45) T^4 / (hbar c)^3
      eps_gamma = 3 P_gamma
      s_gamma   = (4 pi^2/45) T^3 / (hbar c)^3
      n_gamma   = (2 zeta(3)/pi^2) T^3 / (hbar c)^3

  vanishing identically at T = 0.

None of these carries B, C or S — `n_C` is non-leptonic by definition — so they
enter `P`, `eps` and `s`, and the neutrality condition, and nothing else. The
totals a point returns are

    P   = P_matter   + P_e   + P_nue   + P_gamma
    eps = eps_matter + eps_e + eps_nue + eps_gamma
    s   = s_matter   + s_e   + s_nue   + s_gamma

with the neutrino terms present only in the trapped mode and the photon terms
only when the flag is on, and with `f = eps - T s = -P + sum_i mu_i n_i`, the
second equality being one of the `verify/` checks.

### The Euler relation

    eps + P = T s + sum_i mu_i n_i

holds **identically**, not merely numerically, and the demonstration is worth
writing because it is what the vector term has to satisfy to be a mean field
rather than an added constant. For the matter sector: the `+-B` terms cancel
between `eps` and `P`; the kinetic sector satisfies the single-gas identity
summed, `sum_q (eps_q + P_q) = T s + sum_q mu_eff_q n_q`; and what is left over
is `2 P_V`, which is

    2 P_V = a hbar c (sum_q n_q)^2 = V sum_q n_q = sum_q (mu_q - mu_eff_q) n_q

exactly the shift from the effective potentials to the physical ones. The
lepton and photon sectors each satisfy the relation on their own, so they add
without spoiling it. `verify/` asserts it at 1e-8 relative in every mode;
because the cancellation is exact, what that check actually tests is the Fermi
integrals and the assembly, with nothing left to hide an error in.

## Conserved charges

The quark quantum numbers:

| | B_q | C_q | S_q |
|---|---|---|---|
| u | 1/3 | +2/3 | 0 |
| d | 1/3 | -1/3 | 0 |
| s | 1/3 | -1/3 | +1 |

from which

    n_B = (n_u + n_d + n_s)/3
    n_C = (2 n_u - n_d - n_s)/3
    n_S = n_s

and `Y_C = n_C/n_B`, `Y_S = n_S/n_B`. Species potentials are projections of the
conserved ones, `mu_q = B_q mu_B + C_q mu_C + S_q mu_S`, a square system for
three flavours, which inverts to

    mu_B = mu_u + 2 mu_d
    mu_C = mu_u - mu_d
    mu_S = mu_s - mu_d

The plus sign on `mu_S` in `mu_s = mu_B/3 - mu_C/3 + mu_S` is the `S = +1`
convention; under the PDG sign it would be a minus. The sign of `mu_C` is fixed
by `mu_C = mu_p - mu_n` in the hadronic sector, and is reproduced here through
`mu_p - mu_n = (2 mu_u + mu_d) - (mu_u + 2 mu_d) = mu_u - mu_d`; this is why
beta equilibrium reads `mu_C + mu_e = 0` in both sectors.

Neither map is written out in this subpackage: both come from
`eos.general.basis`, built from the quantum-number table above, so a quark
phase and a hadronic phase cannot drift apart in a hybrid construction.

## Equilibrium modes and their residuals

A mode fixes the independent variables and supplies the conditions that close
the system. Every mode carries the same first four rows — the three flavour
self-consistencies and the density — and differs only in what comes after.

### The unknown vectors

**There are two layouts, not one, and the difference is where `mu_e` sits.** In
the beta-equilibrium modes the lepton potential travels with the other
potentials, in slot four, ahead of the densities. In the fixed-fraction modes
it is appended AFTER the densities, because it is present only conditionally
and the layout must stay the same length whether or not it is there. Writing
either layout as though it were universal — as earlier drafts of this document
and of `vmit.tex` did, in opposite directions — makes every residual below
index the wrong slot.

| mode | `leptons` | unknown vector `x` | dim |
|---|---|---|---|
| `beta_eq_neutrinoless` | — | `(mu_u, mu_d, mu_s, mu_e, n_u, n_d, n_s)` | 7 |
| `beta_eq_neutrino_trapped` | — | `(mu_u, mu_d, mu_s, mu_e, mu_nue, n_u, n_d, n_s)` | 8 |
| `fixed_YC` | True | `(mu_u, mu_d, mu_s, n_u, n_d, n_s, mu_e)` | 7 |
| `fixed_YC` | False | `(mu_u, mu_d, mu_s, n_u, n_d, n_s)` | 6 |
| `fixed_YC_YS` | True | `(mu_u, mu_d, mu_s, n_u, n_d, n_s, mu_e)` | 7 |
| `fixed_YC_YS` | False | `(mu_u, mu_d, mu_s, n_u, n_d, n_s)` | 6 |

In the beta-equilibrium modes leptons are always present — they are what the
equilibrium is about — so `leptons` does not apply.

Carrying the densities as unknowns, rather than substituting
`V = a hbar c sum_q n_q` and solving for the potentials alone, is deliberate:
it makes the residual POLYNOMIAL in the mean field instead of nesting the Fermi
integrals inside it. The cost is three extra unknowns; the return is a smooth
system a hybrid method closes in a few iterations.

### The rows

Throughout, `n_q^calc = n_q(mu_eff_q, T, m_q)` is the density the effective
potentials produce, while `n_q` without a superscript is the corresponding
component of `x`, which is what sources `V`. The solver's job is to make the
two agree. Likewise `n_B(n_u,n_d,n_s)`, `n_C` and `n_S` are the charge maps
evaluated on the `x`-components, not on the computed ones, and `n_e(mu_e)`,
`n_nue(mu_nue)` are the lepton gases above.

**`beta_eq_neutrinoless`**, conditions `(n_B, T)` — seven rows, in the order
assembled:

    R1 = n_u^calc - n_u
    R2 = n_d^calc - n_d
    R3 = n_s^calc - n_s
    R4 = n_B(n_u,n_d,n_s) - n_B
    R5 = n_C(n_u,n_d,n_s) - n_e(mu_e)
    R6 = mu_u + mu_e - mu_d
    R7 = mu_d - mu_s

`R6` is `mu_C + mu_e = 0` written in flavour potentials — the weak process
`d <-> u + e + nubar_e` with `mu_nue = 0` — and `R7` is `mu_S = 0`, the
strangeness-changing process `s <-> d`. `R5` is electric neutrality of the
TOTAL system, a separate statement from `n_C` itself, imposed here because this
mode says so.

**`beta_eq_neutrino_trapped`**, conditions `(n_B, Y_Le, T)` — eight rows:

    R1 = n_u^calc - n_u
    R2 = n_d^calc - n_d
    R3 = n_s^calc - n_s
    R4 = n_B(n_u,n_d,n_s) - n_B
    R5 = n_C(n_u,n_d,n_s) - n_e(mu_e)
    R6 = mu_d - mu_s
    R7 = mu_u + mu_e - mu_d - mu_nue
    R8 = (n_e(mu_e) + n_nue(mu_nue))/n_B - Y_Le

**`R6` and `R7` swap places relative to `beta_eq_neutrinoless`:** here the
strangeness equality comes before the beta condition. The rows are the same
physics in both modes and the ordering is immaterial to the solution, but it is
not immaterial to a reader matching this document against the residual, or to
anyone writing a Jacobian by hand, so it is stated as the code has it. `R7` is
`mu_C + mu_e - mu_nue = 0`: the neutrino potential is retained instead of being
set to zero, and `R8` fixes the electron-family lepton number in its place. The
muon family is not tracked, so `Y_Lmu` is refused rather than ignored.

**`fixed_YC`**, conditions `(n_B, Y_C, T)` — with `leptons=True`, seven rows:

    R1 = n_u^calc - n_u
    R2 = n_d^calc - n_d
    R3 = n_s^calc - n_s
    R4 = n_B(n_u,n_d,n_s) - n_B
    R5 = n_C(n_u,n_d,n_s) - Y_C n_B
    R6 = mu_d - mu_s
    R7 = n_e(mu_e) - n_C(n_u,n_d,n_s)

With `leptons=False` the vector loses `mu_e` and the residual loses `R7`: six
rows, `R1`–`R6`, unchanged. There is no beta-equilibrium row — `Y_C` has
replaced it — but strangeness stays equilibrated at `mu_S = 0` through `R6`,
which is what makes this a three-flavour state at a prescribed charge rather
than a two-parameter family.

**`fixed_YC_YS`**, conditions `(n_B, Y_C, Y_S, T)` — with `leptons=True`, seven
rows:

    R1 = n_u^calc - n_u
    R2 = n_d^calc - n_d
    R3 = n_s^calc - n_s
    R4 = n_B(n_u,n_d,n_s) - n_B
    R5 = n_C(n_u,n_d,n_s) - Y_C n_B
    R6 = n_S(n_u,n_d,n_s) - Y_S n_B
    R7 = n_e(mu_e) - n_C(n_u,n_d,n_s)

With `leptons=False`, six rows, `R1`–`R6`. **The `mu_S = 0` row is gone**,
replaced by `R6`: once the strangeness fraction is imposed, `mu_S` is an
OUTPUT — whatever potential the demanded amount of strangeness costs — and
asking for both would over-determine the system. This is the only mode in which
`mu_S != 0` is expected, and it is what makes `Y_C = 0.5`, `Y_S = 0` the
symmetric-nuclear-matter slice a heavy-ion comparison wants.

### The `leptons` flag

Orthogonal to the mode, and an explicit argument, never one of the conditions.
With `leptons=True` an electron gas is added at whatever `mu_e` makes the total
system neutral — which is exactly what the row `R7 = n_e - n_C` says — and it
contributes to `eps`, `P` and `s`. With `leptons=False` the result is charged
quark matter with no leptons at all, which is what a mixed-phase construction
needs for each pure phase before imposing GLOBAL neutrality. In the two
beta-equilibrium modes the flag has no meaning and is ignored.

Photons are an independent flag, contribute to `eps`, `P` and `s` only, and
carry no conserved charge. Wherever a temperature is accepted at a POINT,
entropy per baryon `s/n_B` is accepted in its place through an outer 1-D solve
for T; the table driver does not yet take an entropy axis.

## Numerics

### Judging a solution

Convergence is judged on the residual vector, never on the root finder's own
termination report — these are different questions, and a solver that reports
success has told you its iteration stopped. The residual is made dimensionless
first, each row divided by the scale of the quantity it balances:

    error = max_j |R_j| / sigma_j

    sigma_j = n_B                       if R_j balances a density or a charge
            = max(|mu_B|, 1 MeV)        if R_j equates two potentials
            = 1                         if R_j is already dimensionless

with `mu_B = mu_u + 2 mu_d` at the current iterate, and a floor of 1 MeV so a
pathological iterate passing through `mu_B = 0` cannot divide by zero. A state
is accepted when `error < 1e-10`, one number for the whole repository, so
"converged" means the same thing in every model. Per mode:

    beta_eq_neutrinoless       (n_B, n_B, n_B, n_B, n_B, mu_B, mu_B)
    beta_eq_neutrino_trapped   (n_B, n_B, n_B, n_B, n_B, mu_B, mu_B, 1)
    fixed_YC       (leptons)   (n_B, n_B, n_B, n_B, n_B, mu_B, n_B)
    fixed_YC_YS    (leptons)   (n_B, n_B, n_B, n_B, n_B, n_B,  n_B)

— the trapped mode's `R8` is already a ratio, and every row of `fixed_YC_YS` is
a density. Without leptons each drops its last entry with its last row.

This matters because the rows carry mixed units — densities of order
1e-1 fm^-3, fractions of order unity, potential equalities of order 1e3 MeV —
so a gate on the raw norm is dominated by whichever row happens to be largest
and accepts states that satisfy the others only loosely.

### The solve

Three bounded attempts, in order: Powell's hybrid method from the supplied
start; Levenberg-Marquardt from the same start if the first does not reach the
gate; and, when the caller passed a warm start, one more hybrid attempt from
the mode's own cold guess, since a warm start carried across a threshold can
land outside the basin. The best attempt is returned with its scaled residual
and its status, whether or not the gate was met.

**Non-convergence is a return value, never an exception.** The model is used
inside parameter scans that walk into regions where no positive-density
solution exists, and a scan must be able to score such a point and move on. A
malformed CALL is a different thing — an unknown mode, a fraction the mode does
not take, a sector the model does not implement — and raises before any solve.

### Cold starts

Where the composition is unknown, the densities are estimated first and the
potentials follow. For the two beta-equilibrium modes,

    n_u = n_d = n_B
    n_s = n_B * min(0.9, max(0.01, T/100 MeV + n_B/0.5 fm^-3))

the strange fraction rising with both temperature and density, because the T=0
threshold is what suppresses it and both loosen it. The potentials are then
`mu_q = sqrt(k_F^2 + m_q^2) + V` at the Fermi momentum of those densities,

    k_F(n) = hbar c (6 pi^2 n / g)^(1/3)

with `V` evaluated on the estimate, and `mu_e = max(0, mu_d - mu_u)`, which is
the beta-equilibrium estimate `-mu_C`. The trapped mode adds `mu_nue = 10 MeV`.

Where the mode FIXES the composition, the densities are not estimated but
inverted from the constraints. At fixed `Y_C` and `Y_S`,

    n_s = Y_S n_B,   n_u = (1 + Y_C) n_B,   n_d = 3 n_B - n_u - n_s

which solves `R4`, `R5` and `R6` of `fixed_YC_YS` exactly and leaves only the
vector self-consistency for the solver to close. At fixed `Y_C` alone the
strangeness is not determined by the constraints, and the guess takes

    n_s = 0.3 n_B
    n_u = max(n_B + Y_C n_B + n_s/3, 0.3 n_B)
    n_d = max(n_B - Y_C n_B/2,       0.3 n_B)

the floors keeping a flavour from being seeded at or below zero. Where the
layout carries `mu_e` it is seeded at `sqrt(k_Fe^2 + m_e^2)` with
`k_Fe = hbar c (3 pi^2 n_e)^(1/3)` and `n_e = Y_C n_B`, the neutralizing
electron density.

A guess is only valid within its own mode, since the layouts differ.

### Warm starts and the density sweep

Along a density sweep each solved point seeds the next, in the layout of its
own mode: the potentials and flavour densities of the previous solution, read
back into that mode's order. They are the right continuation variables because
they vary smoothly with `n_B` — including across the strange quark's onset,
where the composition changes fastest — and because the large vector shift is
common to all three potentials and therefore cancels out of the step.

## The API surface

Three entry points, the same three every model exposes, with the parameters
always first and the mode always required:

- `eos_point(par, mode, species, n_B=, T= | SnB=, leptons=, **conditions)` —
  one state. Returns a `PointResult` carrying `ok`, a `message`, and the
  `point` when `ok`. Exactly one of `T` and `SnB`; `SnB` adds an outer solve
  for T. `n_B <= 0` RAISES rather than returning a status: the equations have
  roots there — a net-antiquark state solves them — so the domain is stated
  rather than discovered.
- `eos_table(par, mode, species, axes, fixed=, leptons=, progress=, verbose=)`
  — a solved grid over `{n_B} x {T} x` the fraction axes the mode fixes, with
  the density axis warm-started. `skip_errors=True`, the default, drops points
  the solver could not reach instead of aborting the table. `progress` is
  called once per completed line with `{mode, line, n_lines, temp_key, temp,
  fracs, n_solved, n_requested, elapsed_s}`, the same dictionary in every
  model; `verbose=True` installs the built-in printer. Deep solver code never
  prints.
- `eos_response(par, mode, species, frozen='equilibrium', n_B=, T=, leptons=,
  rel_step=, **conditions)` — second derivatives.

### What `eos_response` implements

Only `frozen='equilibrium'`: everything re-equilibrates under the perturbation,
so the derivatives are taken along the mode's own sequence. Both are central
differences over a relative step `rel_step` (default 1e-3) in the variable
differentiated, because vMIT has no analytic Jacobian in this repository:

    cs2_isothermal = (dP/deps)_T
           ~ [P(n_B+d) - P(n_B-d)] / [eps(n_B+d) - eps(n_B-d)]

    C_V    = (T/n_B) (ds/dT)_n_B
           ~ (T/n_B) [s(T+dT) - s(T-dT)] / (2 dT)

returned under the keys `cs2_isothermal` and `C_V`, the latter only at T > 0
where it is defined. The adiabatic speed, larger by `C_P/C_V` at T > 0, is not
computed by this model: `C_P` is not among the returned quantities, so there is
no factor to form it with. Every other freeze of the specification — frozen per-species
composition, frozen conserved fractions, the leptonic re-neutralization
variants — and the susceptibility matrix `chi_ab = dn_a/dmu_b` raise
`NotImplementedError` naming the gap.

## What a solved point returns

A solve returns an `EoSPoint` carrying every quantity below. Nothing is omitted
because nothing downstream consumes it.

| field | symbol | unit / meaning |
|---|---|---|
| `converged` | — | the status a caller must test first |
| `error` | — | largest scaled residual, dimensionless |
| `n_B` | n_B | fm^-3, the condition |
| `T` | T | MeV, the condition |
| `Y_C`, `Y_S`, `Y_L` | Y_C, Y_S, Y_Le | the fractions the mode fixed |
| `mu_u`, `mu_d`, `mu_s` | mu_u, mu_d, mu_s | MeV, *physical* potentials (not `mu_eff_q`) |
| `mu_e`, `mu_nu` | mu_e, mu_nue | MeV |
| `mu_B`, `mu_C`, `mu_S` | mu_B, mu_C, mu_S | MeV, from the basis map |
| `n_u`, `n_d`, `n_s` | n_u, n_d, n_s | fm^-3, net flavour densities |
| `n_e`, `n_nu` | n_e, n_nue | fm^-3 |
| `P_total` | P | MeV/fm^3, the grand total |
| `e_total` | eps | MeV/fm^3 |
| `s_total` | s | fm^-3 |
| `Y_u`, `Y_d`, `Y_s`, `Y_e`, `Y_nu` | n_i/n_B | per-species fractions |

When `converged` is False every other field holds the best iterate reached,
which is not a physical state.

Two of these are worth stating separately because CLAUDE.md §11 singles them
out.

**`s`** is `s_total`, assembled as the sum of the sectors' entropy densities.
Each sector's own `s` comes through the single-gas Euler identity at its own
potential; the model does NOT compute `s` from the TOTAL Euler relation, which
is why that relation is available as an independent `verify/` check rather than
being true by construction.

**`n_s`** is the strange-quark number density, not a scalar density. vMIT has
no scalar field and no effective mass, so `n_s = (eps - 3P)/m*` is not the
definition here and is used nowhere in the package.

### A table row

`eos_table` flattens each solved point into a row keyed exactly the way a
hadronic table is keyed, so a pure-quark table and a hadronic one concatenate
without renaming: `n_B`, `T`, `chi`, `phase`, `P`, `eps`, `s`, `S_per_B`,
`mu_B`, `mu_C`, `mu_S`, `mu_e`, `Y_C`, `Y_S`, `Y_u`, `Y_d`, `Y_s`, `Y_e`, plus
`Y_nue` and `mu_nue` in the trapped mode. Here `chi = 1` and `phase = "Q"` say
the matter is entirely deconfined. `Y_C` in the row is the charge the solved
state turned out to have, computed from the charge map — not the value that was
requested, which is what makes the column meaningful in every mode.

## The `verify/` suite

Eight physics invariants, each returning a structured pass/fail with the
largest error it saw:

1. **Euler relation** `eps + P = T s + sum_q mu_q n_q`, in every mode, at 1e-8
   relative.
2. **Free energy** `f = eps - T s = -P + sum_i mu_i n_i`, at 1e-8.
3. **Vector self-consistency**: `V` equals `a hbar c sum_q n_q` on the solved
   densities, and `n_q(mu_q - V, T, m_q)` reproduces the density the solver
   settled on, at 1e-8.
4. **Bag and vector signs**: `eps_B = -P_B = B/(hbar c)^3` and `eps_V = +P_V`,
   at 1e-14.
5. **Mode closures**: each mode's own conditions at its own solution —
   `mu_C + mu_e`, `mu_S`, `n_C - n_e` for `beta_eq_neutrinoless`;
   `n_C - Y_C n_B` and `mu_S` for `fixed_YC`; `n_C - Y_C n_B` and
   `n_S - Y_S n_B` for `fixed_YC_YS`; `mu_C + mu_e - mu_nue` and the lepton
   fraction for the trapped mode — at 1e-8.
6. **Free-gas limit**: with `a = 0` and `B = 0` the solved state equals the
   Fermi integrals at the physical potentials themselves, at 1e-10. This is the
   check that the interaction terms are additions to a correct kinetic sector
   rather than a correction hiding one.
7. **Residual gate**: every state the suite solved is inside 1e-10.
8. **Causality**: `0 <= c_s^2 <= 1` along a cold beta-equilibrium sequence.

The suite's density grid stays above the bag's unbinding point, where quark
matter exists at all: below it there is no positive-pressure solution to check.

Not in this list, and deliberately: the monotonicity and causality gate a table
owes before it reaches a structure solver. vMIT's `table.py` produces rows for
a hybrid construction, where a pure quark branch is not the delivered table;
the gate belongs to whoever assembles the delivered one, which for a hybrid
star is `eos/mixed`.

## Not implemented (see docs/DEFERRED.md)

Each raises, naming itself; none is silently ignored.

- **The muon lepton family.** `SpeciesFlags(muons=True)` raises, and
  `beta_eq_neutrino_trapped` takes `(n_B, Y_Le, T)` only — a `Y_Lmu` condition
  is refused at the API boundary.
- **`thermal_neutrinos`**, the flavours not tracked in the composition carried
  as `mu = 0` gases, is not wired and raises.
- **The hadronic sector flags** `hyperons`, `deltas`, `thermal_mesons` raise,
  and will continue to: they have no meaning in a deconfined phase, where
  strangeness enters through the s quark. A refusal for physics, not a gap.
- **The entropy axis of a table.** `eos_point` takes `SnB`; `TableSpec` raises
  for `axes={'SnB': ...}`. The outer solve exists in `eos.general.tabulate`;
  wiring it into the table driver is what is left.
- **Every response freeze but `equilibrium`**, and the susceptibility matrix
  `chi_ab`. An analytic Jacobian is straightforward for this model — one
  algebraic mean field, no scalar sector — and would give `chi_ab` with it.
- **Positivity of the flavour densities is not imposed.** At exotic fixed
  fractions (`Y_C` well above 1, say) the equations have solutions with net
  ANTI-down and anti-strange densities, and the solver returns them as
  converged. They are genuine states of the model at finite temperature rather
  than solver failures, but nothing in the API says so.

## References

- A. Chodos, R. L. Jaffe, K. Johnson, C. B. Thorn, V. F. Weisskopf,
  *New extended model of hadrons*, Phys. Rev. D **9**, 3471 (1974).
- R. O. Gomes, P. Char, S. Schramm,
  *Constraining strangeness in dense matter with GW170817*,
  Astrophys. J. **877**, 139 (2019).
- C. Constantinou, S. Han, P. Jaikumar, M. Prakash,
  *g modes of neutron stars with hadron-to-quark crossover transitions*,
  Phys. Rev. D **104**, 123032 (2021).
- C. Constantinou, S. Han, P. Jaikumar, M. Prakash,
  *Framework for phase transitions between the Maxwell and Gibbs
  constructions*, Phys. Rev. D **107**, 074013 (2023).
- P. M. Johns, P. J. Ellis, J. M. Lattimer,
  *Numerical approximation to the thermodynamic integrals*,
  Astrophys. J. **473**, 1020 (1996).
- Particle Data Group, *Review of Particle Physics* (2024).
