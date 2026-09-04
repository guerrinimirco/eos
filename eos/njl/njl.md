# NJL — three-flavour Nambu–Jona-Lasinio quark matter, with colour superconductivity

`njl.tex` is the same description written for LaTeX, with the bibliography;
this file carries the same physics in Markdown, with the equations in math
mode. Either one alone is enough to reproduce the model. The implementation
specification both follow is `docs/njl_csc_implementation.md`, which is the
authority wherever it and a document differ; where a document and the source
differ, **the source decides**.

---

## 0. Orientation: reading this model from a bag model

If the quark models you already know are `eos.vmit` and `eos.alphabag`, this
section is the map. Everything below is the same physics written out; nothing
here is a shortcut past it.

A bag model *declares* the things that NJL *derives*. That is the whole
difference, and it is why NJL costs a root find where vMIT costs a formula.

| quantity | `vmit` / `alphabag` | `njl` |
|---|---|---|
| quark masses | parameters $m_u, m_d, m_s$, constant | $M_u, M_d, M_s$ **solved** from a gap equation at every point; they fall from $\sim\!368$ MeV to $\sim\!m_f$ across chiral restoration |
| bag constant | an input $B$ | **derived**: $B_\mathrm{eff}$ is a vacuum pressure difference, reported, never set |
| pairing gap | `alphabag`: an input $\Delta_0$, with a BCS $\Delta(T)$ imposed | $\Delta_1,\Delta_2,\Delta_3$ **solved** from three gap equations |
| pairing pattern | `alphabag`: `cfl` is a *mode* the caller picks | an **outcome**: every candidate is solved and the free energy picks the winner |
| colour neutrality | not present (CFL is neutral by construction) | two extra unknowns $\mu_3,\mu_8$ and two extra rows $n_3 = n_8 = 0$ |
| momentum integrals | to $\infty$ | **cut** at $\Lambda$ (vacuum) and $\Lambda_\mathrm{UV}$ (medium) |
| Dirac sea | absent — renormalised into $B$ | explicit, in closed form, and subtracted at the vacuum |
| vector repulsion | `vmit`: $V = a\,\hbar c \sum_q n_q$, one constant $a$ | $\Sigma_V = \mathrm{d}W/\mathrm{d}n_q$, with $G_V$ optionally a function of the density |
| unknowns per point | 2–4 | 5 (unpaired, no vector) to 11 (CFL, trapped, with vector) |
| cost per point | microseconds | milliseconds (unpaired) to ~0.1 s (paired, RG-consistent) |

Two consequences worth internalising before reading further.

**There is no “quark gas plus a correction”.** In `alphabag` the CFL phase is
the unpaired gas plus an analytic $\Delta^2\mu^2$ term. Here the paired phase
is a *different spectrum*: the gap mixes particles with holes, the
quasiparticle energies come out of a matrix diagonalisation at every momentum,
and the pairing piece of $\Omega$ is the difference between that spectrum and
the unpaired one. The `alphabag` form is the weak-coupling limit of it.

**A “bag constant” in the NJL literature is one of three different numbers.**
See §16; the numbers differ by well-defined terms and comparing them blindly
is a 20–25 % error.

---

## 1. The model

A contact four-fermion theory of the light quarks: the scalar channel that
breaks chiral symmetry, the 't Hooft determinant that ties the three flavours
together, a colour-antitriplet diquark channel that condenses into a colour
superconductor, and a vector channel whose repulsion sets the high-density
stiffness.

$$
\begin{aligned}
\mathcal{L} =\;& \bar q\,(i\,\partial\!\!\!/ - \hat m)\,q \\
&+ G_S \sum_{a=0}^{8}\Big[(\bar q\,\tau_a\,q)^2 + (\bar q\,i\gamma_5\tau_a\,q)^2\Big] \\
&- K\Big\{\det\nolimits_f\big[\bar q\,(1+\gamma_5)\,q\big] + \det\nolimits_f\big[\bar q\,(1-\gamma_5)\,q\big]\Big\} \\
&+ G_D \sum_{\eta}\big(\bar q\, i\gamma_5\,\epsilon_\eta\,\lambda_\eta\, C\,\bar q^{\,t}\big)\big(q^{t}\,C\,i\gamma_5\,\epsilon_\eta\,\lambda_\eta\, q\big) \\
&- G_V\,(\bar q\,\gamma^\mu q)^2 .
\end{aligned}
$$

$\tau_a$ are the flavour Gell-Mann matrices with $\tau_0=\sqrt{2/3}\,\mathbb{1}$,
$\epsilon_\eta$ and $\lambda_\eta$ the antisymmetric flavour and colour
generators, $C = i\gamma^2\gamma^0$ the charge-conjugation matrix.

References: Rehberg, Klevansky & Hüfner, PRC **53**, 410 (1996) for the
parameter set; Rüster *et al.*, PRD **72**, 034004 (2005) for the neutral
three-flavour pairing sector and for the quasiparticle spectrum of §7;
Gholami, Hofmann & Buballa, PRD **111**, 014021 (2025) for the RG-consistent
regularization of §8; Alford, Schmitt, Rajagopal & Schäfer, RMP **80**, 1455
(2008) for the review.

**Almost nothing is an input.** The constituent masses come out of the gap
equation, the effective bag constant is a *derived* vacuum pressure
difference, and which colour-superconducting pattern the matter is in is an
*outcome* chosen by free energy, not a declaration.

The scalar, vector and diquark mean fields all arise from ONE
Hubbard–Stratonovich step; none is more fundamental. Two places where the
analogy nevertheless breaks matter for the implementation.

1. The condensation-energy normalisations differ by a factor two — the scalar
   cost is $\sum_f (M_f-m_f)^2/(8G_S)$ and the pairing cost
   $\sum_\eta \Delta_\eta^2/(4G_D)$ — so $\eta_D = G_D/G_S = 1$ does **not**
   mean “equally strong channels”.
2. $\sigma$ enters the quasiparticle diagonal while $\Delta$ is strictly
   off-diagonal: the gap matrix has identically zero diagonal, mixes particles
   with holes, and therefore needs the doubled Dirac basis of §7. That is also
   why the gap kernel carries branch signs where the mass equation does not.

**What is omitted.** The 't Hooft term expanded in the presence of diquark
condensates generates a cross-term
$\propto \sum_\alpha \sigma_\alpha |\Delta_\alpha|^2$, with coupling $K'$.
Most treatments omit it; Baym *et al.*, Rept. Prog. Phys. **81**, 056902
(2018) write it down and then set $K'=0$ in
every published result. It is **not** implemented here, so $\eta_D$ must be
read as an effective coupling that has absorbed it. Nothing here invents a
coefficient by analogy with the mass cross-terms. See §17.

---

## 2. Conventions

Natural units inside the physics modules, $\hbar=c=k_B=1$ with MeV throughout:
momenta, masses and potentials in MeV, densities in MeV³, and $\Omega$, $P$,
$\varepsilon$ in MeV⁴. Every public boundary is fm-based — $n$ in fm⁻³, $T$
and $\mu$ in MeV, $P$ and $\varepsilon$ in MeV/fm³ — converted through
$(\hbar c)^3$ with $\hbar c = 197.3269804$ MeV fm.

Nine colour-flavour modes $j=(f,a)$, $f\in\{u,d,s\}$, $a\in\{r,g,b\}$, indexed
**flavour-major**, $j = 3\,i_f + i_a$. The spin degeneracy of one mode is
$g=2$; the Dirac sea of one flavour carries $g_\mathrm{sea} = 2N_c = 6$, since
the vacuum is not resolved by colour.

Charges $q_u = +2/3$, $q_d = q_s = -1/3$, and **strangeness $S=+1$ per $s$
quark**, the opposite of the PDG sign and this repository's convention
throughout. $C$ is the charge of strongly-interacting matter ONLY; the leptons
are excluded from it and enter through the separate condition of total
electric neutrality.

Colour generators:

$$
T_3 = \mathrm{diag}\Big(\tfrac12,\,-\tfrac12,\,0\Big),
\qquad
T_8 = \mathrm{diag}\Big(\tfrac13,\,\tfrac13,\,-\tfrac23\Big) = \frac{\lambda_8}{\sqrt3}.
$$

**Three normalisations of $T_8$ are in circulation** and mixing them corrupts
$\mu_8$ by factors of 1.15 to 1.7:

- the halved Gell-Mann form $\sqrt3\,T_8 = \mathrm{diag}(\tfrac12,\tfrac12,-1)$,
  for which $\mu_8^\mathrm{theirs} = (2/\sqrt3)\,\mu_8^\mathrm{ours} = 1.1547\,\mu_8^\mathrm{ours}$
  — Rüster *et al.*, Pagliara & Schaffner-Bielich, and Kunkel *et al.*'s Eq. (7)
  **as written**;
- the full $\lambda_8$, for which $\mu_8^\mathrm{ours} = \sqrt3\,\mu_8^\mathrm{theirs}$
  — Buballa, Steiner–Reddy–Prakash, Baym *et al.*, Gholami *et al.*, and the
  MUSES NJL module those papers are computed with;
- this repository's $T_8 = \mathrm{diag}(1,1,-2)/3$.

The RG-consistent line of papers therefore **splits**: Kunkel *et al.* print
the halved convention and compute in the full one, because their published
module is Gholami *et al.*'s. A comparison against either paper's CODE, or
against Gholami's equations, uses $\sqrt3$; only a comparison against the
symbols printed in Kunkel *et al.* uses $2/\sqrt3$.

As a worked consequence, the CFL result $\mu_8 = -\tfrac{1}{2\sqrt3}\,m_s^2/\mu$
of Steiner *et al.* reads $\mu_8 = -\tfrac12\,m_s^2/\mu$ here: at $m_s = 300$,
$\mu = 450$ MeV that is $-57.7$ MeV in their convention and $-100.0$ MeV in
ours.

The mode potentials, with all five conserved-charge potentials:

$$
\mu_{(f,a)} = \frac{\mu_B}{3} + q_f\,\mu_C + s_f\,\mu_S + (T_3)_a\,\mu_3 + (T_8)_a\,\mu_8 ,
\qquad
\mu_e = \mu_{\nu_e} - \mu_C ,
$$

the last of which is beta equilibrium in the sign convention
$\mu_C = \mu_p - \mu_n$ used across this repository.

**Two naming traps in the sources.** The $G_D$ of Steiner, Reddy & Prakash is
the 't Hooft coupling, not the diquark one (theirs is $G_\mathrm{DIQ}$); and
$\Delta_2$ in the Rüster/Buballa index convention means the *us* gap, not the
second largest. Our index convention is theirs: $\Delta_1$ pairs $d$ with $s$,
$\Delta_2$ pairs $u$ with $s$, $\Delta_3$ pairs $u$ with $d$.

---

## 3. Parameters

Every function takes the parameter set as its first argument and none reaches
for a default on the caller's behalf: the set is one frozen record, hashable
and safely shared between processes. Three tiers, and the tiers say which
numbers an inference run may move.

### Tier 1 — the vacuum fit, never sampled

The Rehberg–Klevansky–Hüfner set, fitted to $m_\pi = 135.0$, $f_\pi = 92.4$,
$m_K = 497.7$, $m_{\eta'} = 957.8$ MeV:

| field | value | what it buys |
|---|---|---|
| `Lambda` | $\Lambda = 602.3$ MeV | the fit as a whole |
| `GS_Lambda2` | $G_S\Lambda^2 = 1.835 \Rightarrow G_S = 5.0584\times10^{-6}\ \mathrm{MeV^{-2}}$ | $m_\pi$, $f_\pi$ |
| `K_Lambda5` | $K\Lambda^5 = 12.36 \Rightarrow K = 1.5594\times10^{-13}\ \mathrm{MeV^{-5}}$ | $m_{\eta'}$ |
| `m_u`, `m_d` | 5.5 MeV | $m_\pi$ |
| `m_s` | 140.7 MeV | $m_K$ |

The *dimensionless* combinations are what is stored, because they are what the
fit determines; $G_S = 1.835/\Lambda^2$ and $K = 12.36/\Lambda^5$ are derived
properties and every equation below takes them in those units. Re-sampling one
of these five breaks the vacuum phenomenology the model is anchored to — and
$(f_\pi,\phi)$ alone leave a two-root degeneracy, so $m_\pi$ and $m_K$ must
stay in any refit.

### Tier 2 — structural, declared per run

These change the equations, not a number in them.

| field | default | legal values |
|---|---|---|
| `vector_form` | `"constant"` | `"constant"`, `"power_law"`, `"gluon_exchange"` |
| `alpha` | $2/3$ | any real $\ge 0$; only read by `power_law` |
| `n_ref` | 0.48 fm⁻³ | the reference **quark** density of `power_law`, i.e. $n_B = n_\mathrm{sat}$ |
| `lambda_UV` | 10.0 | $\lambda = \Lambda_\mathrm{UV}/\Lambda \ge 1$; $\lambda = 1$ is the sharp cutoff. $\lambda < 1$ **raises** |

The lepton content is structural in the same sense and is carried by the
species flags (§5).

### Tier 3 — the sampled vector

| field | default | range | meaning |
|---|---|---|---|
| `eta_D` | 0.75 | 0.5–1.5 | $\eta_D = G_D/G_S$; 0.75 is the Fierz value |
| `eta_V` | 0.0 | 0–1 | $\eta_V = G_V/G_S$, for `constant` and `power_law` |
| `G_V0_over_GS` | 0.5 | 0–1 | vector strength of `gluon_exchange` only |
| `M_g` | 500 MeV | 400–800 MeV | the gluon mass of that form |

$\eta_D = 1$ does not mean equally strong channels (see the factor two in §1),
and $\eta_D$ also has to absorb the omitted 't Hooft–diquark cross-term, so it
is an effective coupling and a paper using it should say so.

### Derived properties

$G_S = \texttt{GS\_Lambda2}/\Lambda^2$, $K = \texttt{K\_Lambda5}/\Lambda^5$,
$G_D = \eta_D G_S$, `current_masses` $= (m_u, m_d, m_s)$, and
`Lambda_medium` $= \lambda\Lambda = \Lambda_\mathrm{UV}$.

### Published sets

`Parameters.named(name)` takes one of

| name | what it is |
|---|---|
| `"rkh"` | nothing changed — the shipped default, and the set every sharp-cutoff number below was produced at |
| `"rg_njl1"` | $\eta_D = 1.45$, $\eta_V = 0.7$ |
| `"kunkel"` | an alias of `"rg_njl1"` |
| `"gluon_exchange"` | the `gluon_exchange` form, $G_{V0}/G_S = 0.5$, $M_g = 500$ MeV |

An unknown name raises `KeyError` listing the four.

`rg_njl1` is Gholami *et al.*'s “parameter set 1” — the soft one of their three
— and the set Kunkel, Rather *et al.* [arXiv:2607.11537] use for their
proto-neutron-star study. With the default $\lambda = 10$ the couplings AND
the regularization are theirs, which is what makes this a reproduction of that
model rather than a borrowing of its numbers: the two were never independent,
since RG-consistent gaps run well above sharp-cutoff ones at the same
coupling. `kunkel` is kept as an alias because that is what the set was called
before the regularization caught up with it.

**The constants the shared sectors carry**, not fitted here:
$m_e = 0.510999$ MeV and $m_\mu = 105.6584$ MeV with $g_e = g_\mu = 2$;
neutrinos massless with $g_\nu = 1$ per flavour, particles and antiparticles
summed; photons massless with $g_\gamma = 2$.

**Three routes to a parameter set.** CLAUDE.md §6 makes model parameters
arguments, so all three have to exist. *By name:* `Parameters.default()` is
the shipped RKH set, and `Parameters.named(name)` takes any of the four
published sets. *A new set:* every field carries a default, so
`Parameters(eta_D=..., eta_V=...)` names only what changes; the dataclass is
frozen, so `dataclasses.replace` is how a set already in hand is modified.
*From nuclear-matter parameters:* no route, and none is missing — NJL has no
nuclear sector, so there is no `nmp.py` and nothing to invert; its parameters
are fixed by vacuum data instead.

---

## 4. The integrals, against vMIT and alphaBag

This is the section a reader of `eos.vmit` or `eos.alphabag` should read
first. The physics is the same Fermi gas; four things are different, and each
one is a place a bag-model instinct gives the wrong answer.

### 4.1 One mode as a cut Fermi gas

One mode of mass $M$ at effective potential $\mu^\ast$ and temperature $T$,
with $E = \sqrt{k^2+M^2}$, $f^\pm = [1+e^{(E\mp\mu^\ast)/T}]^{-1}$ and
$x_\pm = E\mp\mu^\ast$:

$$
\begin{aligned}
n      &= \frac{g}{2\pi^2}\int_0^{\Lambda}\!\mathrm{d}k\;k^2\,\big(f^+ - f^-\big), \\
\rho_s &= \frac{g}{2\pi^2}\int_0^{\Lambda}\!\mathrm{d}k\;k^2\,\frac{M}{E}\,\big(f^+ + f^-\big), \\
\varepsilon &= \frac{g}{2\pi^2}\int_0^{\Lambda}\!\mathrm{d}k\;k^2\,E\,\big(f^+ + f^-\big), \\
s &= \frac{g}{2\pi^2}\int_0^{\Lambda}\!\mathrm{d}k\;k^2 \sum_{\pm}\Big[\frac{x_\pm}{T}f^\pm + \ln\!\big(1+e^{-x_\pm/T}\big)\Big], \\
P_{\log} &= \frac{g}{2\pi^2}\int_0^{\Lambda}\!\mathrm{d}k\;k^2\,T\Big[\ln\!\big(1+e^{-(E-\mu^\ast)/T}\big) + \ln\!\big(1+e^{-(E+\mu^\ast)/T}\big)\Big].
\end{aligned}
$$

Antiparticles **subtract** in $n$ and **add** in $\rho_s$, $\varepsilon$ and
$P$.

**Difference 1 — the upper limit is finite.** $\Lambda$ here is the cutoff on
the MEDIUM integral, which is $\Lambda_\mathrm{UV}$ and not the vacuum's
$\Lambda$ whenever $\lambda>1$ (§8). `eos.vmit` and `eos.alphabag` integrate to
infinity, because a bag model has no cutoff to regularise: the four-fermion
interaction is what needs one, and they have none. The cut is not decoration —
it is what makes $\phi_f$ finite, and it is what §8 exists to repair.

**Difference 2 — the mass is a solved quantity.** $M$ above is $M_f$ from the
gap equation of §6, not a parameter. Its density dependence is why the NJL
energy density is not a sum of independent flavour integrals: moving $\mu_u$
moves $M_s$.

At $T=0$ the occupations become step functions, $f^+ = \theta(\mu^\ast - E)$
and $f^- = 0$ for $\mu^\ast>0$, and with

$$
k_F = \min\!\Big(\sqrt{\mu^{\ast2}-M^2},\;\Lambda\Big),
\qquad E_F = \sqrt{k_F^2+M^2}
$$

every one of them integrates in closed form:

$$
\begin{aligned}
n &= \frac{g}{6\pi^2}k_F^3, \\
\rho_s &= \frac{g}{2\pi^2}\cdot\frac{M}{2}\Big[k_F E_F - M^2\,\mathrm{arcsinh}\frac{k_F}{M}\Big], \\
\varepsilon &= \frac{g}{2\pi^2}\cdot\frac18\Big[k_F(2k_F^2+M^2)E_F - M^4\,\mathrm{arcsinh}\frac{k_F}{M}\Big], \\
P_{\log} &= \mu^\ast n - \varepsilon, \qquad s = 0,
\end{aligned}
$$

and **all five vanish exactly** when $|\mu^\ast| \le M$: a mode too heavy for
its own potential is not in the medium, which is a statement rather than an
optimisation. These are the Dirac-sea integrals of §6 with $\Lambda$ replaced
by $k_F$ and $g_\mathrm{sea}$ by $g$ — the same integral over a different
interval. **The $\min$ is where a cut theory differs from an uncut one**, and
`vmit`'s closed forms are exactly these with the $\min$ removed.

**Difference 3 — the pressure must come from the logarithm.** The second
standard pressure form,

$$
P_{k^4} = \frac{g}{6\pi^2}\int_0^{\Lambda}\!\mathrm{d}k\;\frac{k^4}{E}\big(f^+ + f^-\big),
$$

follows from $P_{\log}$ by parts, and the boundary term does NOT vanish when
the integral is cut:

$$
P_{\log} - P_{k^4} = \frac{g}{6\pi^2}\,\Lambda^3\,T\Big[\ln\!\big(1+e^{-(E_\Lambda-\mu^\ast)/T}\big) + \ln\!\big(1+e^{-(E_\Lambda+\mu^\ast)/T}\big)\Big],
\quad E_\Lambda = \sqrt{\Lambda^2+M^2}.
$$

It is not small: 0.1 % of $P$ at $(M,\mu^\ast,T) = (100,500,20)$ MeV, 10.5 %
at $(40,590,30)$, 36.3 % at $(140,700,5)$ and 39.9 % at $(140,700,50)$.
**Every assembly here uses $P_{\log}$.** In `vmit`/`alphabag` the two forms
agree identically, because there the upper limit is infinite and the surface
term vanishes; here they agree only at $T=0$ with $k_F<\Lambda$, which is
exactly why the error would hide until a table is built at finite temperature.

**Difference 4 — the quadrature has to be told where the Fermi surface is.**
Gauss–Legendre on one panel over $[0,\Lambda]$ cannot resolve the Fermi step.
Breakpoints go at each Fermi momentum
$k_{F,j} = \sqrt{\mu^{\ast2}_j - M_{f(j)}^2}$ and, at $T>0$, at
$k_{F,j}\pm 25\,T$; each panel is integrated separately, and **the cutoff is
imposed as the panel upper limit before the panels are built**, because
filtering breakpoints afterwards can delete the Fermi-surface break and revert
silently to single-panel accuracy. `vmit` and `alphabag` sidestep this
entirely by using the JEL expansion, which is a closed form and has no
quadrature to break.

### 4.2 The three models side by side

| | `vmit` | `alphabag` | `njl` |
|---|---|---|---|
| degrees of freedom | 3 flavours, $g=6$ | 3 flavours, $g=6$ (+ gluons) | 9 colour-flavour modes, $g=2$ |
| mass in the integral | $m_q$ (parameter) | $m_q$ (parameter) | $M_f$ (solved) |
| upper limit | $\infty$ | $\infty$ | $\Lambda_\mathrm{UV}$ |
| evaluated by | JEL expansion | JEL + massless closed forms | panelled Gauss–Legendre |
| $P$ from | JEL | closed forms | $P_{\log}$, never $P_{k^4}$ |
| Dirac sea | none | none | $-\sum_f \varepsilon_{\mathrm{sea},f}$, closed form |
| condensate cost | none | none | $\mathcal{C} = 2G_S\sum\phi_f^2 - 4K\phi_u\phi_d\phi_s$ |
| pairing | none | $+\big(\mu_u^2+\mu_d^2+\mu_s^2\big)\Delta^2/\pi^2$, $\Delta$ imposed | $\delta\Omega_\mathrm{pair}$ from a 36-state spectrum, $\Delta$ solved |
| perturbative QCD | none | $\alpha_s$ multiplying the free-gas terms | none (the four-fermion couplings are the interaction) |
| vector | $P_V = \varepsilon_V = \tfrac12 a\hbar c\,n_q^2$ | none | $W = G_V(n_q)n_q^2$, $\Sigma_V = \mathrm{d}W/\mathrm{d}n_q$ |
| bag | $\pm B/(\hbar c)^3$ | $\pm B/(\hbar c)^3$ | $-\Omega_\mathrm{vac}$, derived |
| colour potentials | — | — | $\mu_3,\mu_8$, solved from $n_3 = n_8 = 0$ |

The Euler relation $\varepsilon + P = T s + \sum_i \mu_i n_i$ holds in all
three, and in all three it is an audit rather than a definition.

### 4.3 Where the alphaBag CFL term comes from

`alphabag` adds $\delta P = (\mu_u^2+\mu_d^2+\mu_s^2)\Delta^2/(\pi^2 (\hbar c)^3)$
— Alford, Braby, Paris & Reddy, ApJ **629**, 969 (2005). That is the standard
CFL condensation energy: nine gapped modes at a common $\mu$, counted, with
$\Delta$ an **input**.

Its counterpart here is not one term but the PAIR
$\delta\Omega_\mathrm{pair} + \mathcal{D}$ of §10 — the change in the
spectrum, plus the $\sum_\eta\Delta_\eta^2/(4G_D)$ cost of making the
condensate — evaluated where it is stationary in $\Delta$, which is the gap
equation. At that point the two scale the same way, $\sim \mu^2\Delta^2$,
and they differ in three ways that matter at $\Delta \sim 100$ MeV: the gap
is **solved** rather than declared, the nine modes are **not degenerate**
(the $s$ mass breaks the locking, so the three gaps are three numbers), and
the paired **densities and entropy** are not obtained by differentiating a
$\Delta^2\mu^2$ term but by the Hellmann–Feynman pass of §7.4 — the largest
single departure, worth tens of percent (trap 3).

---

## 5. Species flags

`eos.njl.SpeciesFlags` carries the repository's six names plus two of its own.
Every one defaults to `False`.

| flag | default | legal | meaning |
|---|---|---|---|
| `csc` | `False` | both | the colour-superconducting sector. Off: no gaps, no spectrum to diagonalise, $\mu_3=\mu_8=0$ identically, and the only pattern is `unpaired` |
| `muons` | `False` | both | the muon species, in leptonic equilibrium at $\mu_\mu = \mu_e - \mu_{\nu_e}$ |
| `thermal_neutrinos` | `False` | both | neutrino flavours NOT tracked in the composition, as $\mu=0$ gases: $\varepsilon$, $P$, $s$ only |
| `photons` | `False` | both | blackbody photons; $\varepsilon$, $P$, $s$ only, no conserved charge |
| `two_flavour` | `False` | both in `unpaired`/`2SC`; **raises** in `CFL`/`uSC`/`dSC`/`free` | switches the strange **Fermi sea** off (§19) |
| `hyperons` | `False` | `False` only | **raises**: this is a quark model; there are no baryons in it to be strange |
| `deltas` | `False` | `False` only | **raises**: no baryon resonances |
| `thermal_mesons` | `False` | `False` only | **raises**: the mesons of this Lagrangian are the auxiliary fields of the four-fermion terms, eliminated in favour of $G_S$, $K$, $G_D$; they carry no independent thermal population at mean field |

Setting a flag this model does not implement raises `NotImplementedError`
naming the reason; it is never quietly ignored.

`thermal_neutrinos` is meaningful alongside `beta_eq_neutrino_trapped` and
does not raise there: the flag covers the flavours absent from the matter
composition, which under trapping means the $\tau$ family.

---

## 6. The Dirac sea, the condensates and the vacuum

The vacuum integrals of one flavour, closed form, $g_\mathrm{sea} = 6$,
$R = \sqrt{\Lambda^2+M^2}$:

$$
\begin{aligned}
\rho_s^\mathrm{vac}(M) &= \frac{g_\mathrm{sea}}{2\pi^2}\cdot\frac{M}{2}\Big[\Lambda R - M^2\,\mathrm{arcsinh}\frac{\Lambda}{M}\Big], \\
\varepsilon_\mathrm{sea}(M) &= \frac{g_\mathrm{sea}}{2\pi^2}\cdot\frac18\Big[\Lambda(2\Lambda^2+M^2)R - M^4\,\mathrm{arcsinh}\frac{\Lambda}{M}\Big],
\end{aligned}
$$

and $\mathrm{d}\varepsilon_\mathrm{sea}/\mathrm{d}M = \rho_s^\mathrm{vac}$
identically. The condensate of one flavour is the sea's scalar density with
the medium's own occupation subtracted, summed over the three colours of that
flavour:

$$
\phi_f = -\rho_s^\mathrm{vac}(M_f) + \sum_a \rho_{s,(f,a)} + \delta\rho_{s,f},
$$

with $\delta\rho_{s,f}$ the pairing correction of §7. The constituent masses,
including the determinant cross-terms — the coefficients most often dropped or
mis-signed:

$$
\begin{aligned}
M_u &= m_u - 4G_S\,\phi_u + 2K\,\phi_d\phi_s, \\
M_d &= m_d - 4G_S\,\phi_d + 2K\,\phi_u\phi_s, \\
M_s &= m_s - 4G_S\,\phi_s + 2K\,\phi_u\phi_d.
\end{aligned}
$$

$\phi_f = \langle\bar q_f q_f\rangle$ is negative in the broken phase. The
convenient identity $2G_S\sum_f\phi_f^2 = \sum_f (M_f-m_f)^2/(8G_S)$ holds
ONLY at $K=0$, where it is exact to $4\times10^{-16}$; with the determinant on
it fails by 34 %. The condensate cost entering both $\Omega$ and $\varepsilon$
is

$$
\mathcal{C} = 2G_S\sum_f \phi_f^2 - 4K\,\phi_u\phi_d\phi_s .
$$

**The vacuum solution.** At $\mu = T = 0$ the mass and condensate equations
close on themselves and are solved by a **damped fixed point** on the masses,

$$
M \leftarrow M + \lambda_\mathrm{damp}\big(M_\mathrm{new}[\phi(M)] - M\big),
\qquad \lambda_\mathrm{damp} = 0.3,
$$

to a step below $10^{-12}$ MeV in at most 4000 iterations, and NOT by a root
finder on the condensates, which diverges and returns masses that increase
with density. The vacuum constant is

$$
\Omega_\mathrm{vac} = \varepsilon_\mathrm{vac} = -\sum_f \varepsilon_\mathrm{sea}(M_f^\mathrm{vac}) + \mathcal{C}^\mathrm{vac},
$$

and the equality of the two is what makes the Euler relation survive the
subtraction. It is asserted at construction.

**Vacuum diagnostics.** The pion decay constant from the quark loop,

$$
f_\pi^2 = 2M^2 I_2,
\qquad
I_2 = \frac{N_c}{4\pi^2}\Big[\mathrm{arcsinh}\frac{\Lambda}{M} - \frac{\Lambda}{\sqrt{\Lambda^2+M^2}}\Big],
$$

and the effective bag constant, $\Omega$ at fixed masses evaluated at the
current masses minus at the broken-phase ones,

$$
B_\mathrm{eff} = \Omega\big[M_f = m_f\big] - \Omega\big[M_f = M_f^\mathrm{vac}\big].
$$

$B_\mathrm{eff}$ is a **derived** quantity here, not an input the way a bag
constant is in `vmit` or `alphabag`. §16 gives its published counterparts and
the terms by which they differ.

---

## 7. The pairing sector

The gap matrix, the 36-state Dirac-basis spectrum, the pairing correction to
$\Omega$ and the Hellmann–Feynman kernels are NOT in this package: they are
`eos/general/pairing.py`, shared with the chiral colour-dielectric model,
because the pairing sector of the two is the same sector. The cut medium
integrals ARE here, under the carve-out for cutoff-regularized NJL integrals,
which are model physics.

### 7.1 The gap matrix

$$
\mathcal{G}_{(fa),(gb)} = \sum_\eta \Delta_\eta\,\epsilon^{ab\eta}\,\epsilon_{fg\eta}
= \sum_\eta \Delta_\eta\,(B_\eta)_{(fa),(gb)},
$$

so $\Delta_1$ pairs $d$ with $s$, $\Delta_2$ pairs $u$ with $s$, and
$\Delta_3$ pairs $u$ with $d$ — the 2SC gap. The matrix is symmetric with
identically zero diagonal. Its eigenvalue multiplicities are a *derived*
property of the pattern and are never assigned by hand; at $\Delta_0 = 60$ MeV
the spectrum of $\mathcal{G}$ is

| pattern | mask $(\Delta_1,\Delta_2,\Delta_3)$ | spectrum of $\mathcal{G}$ [MeV] |
|---|---|---|
| `unpaired` | $(0,0,0)$ | $0\ (\times 9)$ |
| `2SC` | $(0,0,D)$ | $-60\ (\times2)$, $0\ (\times5)$, $+60\ (\times2)$ |
| `CFL` | $(D,D,D)$ | $-60\ (\times5)$, $+60\ (\times3)$, $+120$ |
| `uSC` | $(0,D,D)$ | $\pm84.85$, $\pm60\ (\times2)$, $0\ (\times3)$ |
| `dSC` | $(D,0,D)$ | as `uSC` |

With independent gaps the $\pm\sqrt2\,\Delta_0$ eigenvalue generalises to
$\pm\sqrt{\Delta_2^2+\Delta_3^2}$.

### 7.2 The quasiparticle spectrum

The mean-field inverse propagator is diagonalised in the **full Dirac basis**:
four components per colour-flavour mode — two particle, two antiparticle — so
36 states at each momentum, with the gap mixing them. This is Appendix A of
Rüster *et al.*, PRD **72**, 034004 (2005), and it is what the published MUSES
NJL module diagonalises.

The 36 states block-diagonalise. Six of the nine modes pair off pairwise —
$(d_r,u_g)$ through $\Delta_3$, $(s_r,u_b)$ through $\Delta_2$, $(s_g,d_b)$
through $\Delta_1$ — and each such pair $(a,b)$ gives two $4\times4$ blocks,
one per sign $\varsigma = \pm1$:

$$
\mathcal{M}^{(\varsigma)}_{ab}(k) =
\begin{pmatrix}
-\varsigma\mu^\ast_a + \varsigma M_a & k & 0 & -\Delta_\eta \\
k & -\varsigma\mu^\ast_a - \varsigma M_a & \Delta_\eta & 0 \\
0 & \Delta_\eta & \varsigma\mu^\ast_b + \varsigma M_b & k \\
-\Delta_\eta & 0 & k & \varsigma\mu^\ast_b - \varsigma M_b
\end{pmatrix}.
$$

The remaining triple $(u_r,d_g,s_b)$ is coupled by all three gaps at once and
gives one $12\times12$ block: slot $m$ (mode $j$) carries rows
$4m+(0,1,2,3)$ with diagonal
$(-\mu^\ast_j - M_j,\; -\mu^\ast_j + M_j,\; \mu^\ast_j - M_j,\; \mu^\ast_j + M_j)$
and $k$ at $(4m, 4m{+}1)$ and $(4m{+}2, 4m{+}3)$, and the gap $\Delta_\eta$
joining slots $A$ and $B$ enters symmetrically at

$$
(4A{+}0,\,4B{+}3) = -\Delta_\eta,\quad
(4A{+}1,\,4B{+}2) = +\Delta_\eta,\quad
(4A{+}2,\,4B{+}1) = +\Delta_\eta,\quad
(4A{+}3,\,4B{+}0) = -\Delta_\eta,
$$

with $\Delta_3$ joining slots $(0,1)$, $\Delta_2$ slots $(0,2)$ and $\Delta_1$
slots $(1,2)$. Every matrix is real symmetric, its spectrum comes in $\pm$
pairs, and $\Omega$ wants the positive half — taken as **half the sum over all
36 of $|\lambda_a|$**, so a branch crossing zero inside a gapless window needs
no bookkeeping: its partner crosses back, and
$\mathrm{d}|\lambda|/\mathrm{d}x = \mathrm{sgn}(\lambda)\,\mathrm{d}\lambda/\mathrm{d}x$
carries the branch sign.

A mode **no nonzero gap touches** is left out of the blocks altogether, and
out of the unpaired reference with it. Its four Dirac components are then
exactly the unpaired ones and it contributes nothing — but as a difference it
would be zero minus an ill-conditioned number, since an ungapped branch
crosses zero at its own Fermi surface with its partner at $-0$ in the same
block. Leaving it out makes the zero exact; carrying it stalls the equilibrium
solve at $10^{-8}$.

**Why not the on-shell reduction.** Solving the free Dirac problem first,
fixing $E_f = \sqrt{k^2+M_f^2}$ and pairing the on-shell modes in an
$18\times18$ Bogoliubov–de Gennes problem
$\big[\begin{smallmatrix}\mathrm{diag}(\xi^r) & \mathcal{G}\\ \mathcal{G} & -\mathrm{diag}(\xi^r)\end{smallmatrix}\big]$
with $\xi^r_j = E_{f(j)} - r\mu^\ast_j$, is the familiar construction, and for
one gap it gives

$$
E^\pm = \sqrt{(\bar E - \bar\mu)^2 + \Delta^2} \pm \Big[\frac{E_d - E_u}{2} - \delta\mu\Big],
\quad
\bar E = \frac{E_u+E_d}{2},\;
\bar\mu = \frac{\mu^\ast_{ur}+\mu^\ast_{dg}}{2},\;
\delta\mu = \frac{\mu^\ast_{dg}-\mu^\ast_{ur}}{2}.
$$

It drops the particle–antiparticle mixing the gap induces, and what controls
that is the **mass mismatch of the pair**: at equal paired masses the closed
form reproduces the exact spectrum to $10^{-13}$ MeV whatever the mass is, and
the two part company as the masses differ — 3.4 MeV at $M = (5.5,300)$ MeV,
$\Delta = 80$ MeV, $k = 320$ MeV. So the reduction is harmless for 2SC, which
pairs $u$ with $d$, and not harmless for CFL, which pairs both with $s$: at
parameter set 1 it moves the CFL branches by up to 11 MeV, the pressure by
6–16 %, and the 2SC $\to$ CFL transition density by 9 %. $E^-$ may be
**negative**: that is the gapless window, the BCS blocking region, not an
error.

### 7.3 The pairing potential

Written as a *correction* — a difference from the unpaired spectrum:

$$
\delta\Omega_\mathrm{pair} = -\frac{1}{2\pi^2}\int_0^{\Lambda}\!\mathrm{d}k\;k^2
\Big[\tfrac12\sum_{a=1}^{36}\varphi(|\lambda_a|) - \sum_{r=\pm}\sum_{j\in\mathcal{C}}\varphi(|\xi^r_j|)\Big],
$$
$$
\varphi(x) = x + 2T\ln\!\big(1+e^{-x/T}\big),
\qquad \varphi(x) = x \text{ at } T = 0 .
$$

This vanishes **identically** at $\Delta = 0$, which is what makes the
unpaired phase a clean limit of the same code, and in the clean weak-coupling
limit it obeys the BCS logarithm,

$$
-\delta\Omega_\mathrm{pair} \longrightarrow \frac{2}{\pi^2}\,\mu^2\Delta^2\Big[\ln\frac{2\Lambda}{\Delta} - \frac12\Big],
$$

which is the `alphabag` $\delta P$ of §4.3 with the logarithm still in it.
$\mathcal{C}$ is the set of modes the blocks cover. Both signs of $r$ are
summed in the reference: the antiparticle branches contribute 8.8 % of the
particle piece at $\Lambda = 600$ MeV and 17.1 % at 1000 MeV. The $|\xi_j|$
subtraction kinks at each of the nine $k_{F,j}$, so the pairing quadrature is
split there (and at $k_{F,j}\pm 25T$) exactly as in §4.1. It is the
*splitting* that buys the accuracy, not the node count: at 100 nodes per panel
the relative error is $3\times10^{-14}$ where a single panel of 800 nodes
reaches $2\times10^{-7}$. The shipped rule is 24 Gauss–Legendre nodes per
panel, overridable per call.

### 7.4 The gap equations, and everything else the same pass returns

Since $\varphi'(x) = \tanh(x/2T)$, differentiating by Hellmann–Feynman on
$H^r$ gives, with eigenvectors $|V_a\rangle$ in the doubled basis and $P_j$
the projector on mode $j$:

$$
\frac{\Delta_\eta}{2G_D} - \frac{1}{2\pi^2}\sum_r\int\!\mathrm{d}k\,k^2 \sum_a
\Big\langle V^r_a \Big| \begin{pmatrix}0 & B_\eta\\ B_\eta & 0\end{pmatrix} \Big| V^r_a \Big\rangle
\tanh\frac{E^r_a}{2T} \;=\; 0,
$$

$$
\delta n_j = \frac{1}{2\pi^2}\sum_r\int\!\mathrm{d}k\,k^2
\Big[\sum_a \tanh\frac{E^r_a}{2T}\frac{\partial E^r_a}{\partial\mu_j} + r\tanh\frac{\xi^r_j}{2T}\Big],
\qquad \frac{\partial H^r}{\partial \mu_j} = -r\,(P_j \oplus -P_j),
$$

$$
\delta\rho_{s,f} = -\frac{1}{2\pi^2}\sum_r\int\!\mathrm{d}k\,k^2
\Big[\sum_a \tanh\frac{E^r_a}{2T}\frac{\partial E^r_a}{\partial M_f} - \sum_{j\in f}\frac{M_f}{E_j}\tanh\frac{\xi^r_j}{2T}\Big],
$$
$$
\frac{\partial H^r}{\partial M_f} = \mathrm{diag}(d)\oplus-\mathrm{diag}(d), \quad d_j = \frac{M_f}{E_j}\delta_{f(j),f},
$$

$$
\delta s = \frac{1}{2\pi^2}\sum_r\int\!\mathrm{d}k\,k^2\sum_a\big[\psi(E^r_a) - \psi(|\xi^r_a|)\big],
\qquad \psi(x) = 2\ln\!\big(1+e^{-x/T}\big) + \frac{x}{T}\Big[1-\tanh\frac{x}{2T}\Big].
$$

All four come from ONE quadrature pass and one batched diagonalisation;
computing them separately would diagonalise the same blocks five times, and
finite-differencing them instead was measured 40× slower and ill-conditioned
enough to lose convergence.

**The kernel is not $\Delta/|E|$** (trap 2). That form — obtained by
differentiating $|E|$ as though every branch were positive — is wrong by a
factor 12.0 at $\Delta = 40$ MeV, 1.7 at 60 and 1.3 at 80 for $\mu_u = 400$,
$\mu_d = 500$ MeV, and it makes the gap *grow* with the mismatch, the opposite
of the physics. It agrees with the true kernel only where every branch is
positive.

**Paired densities and entropy are not the unpaired integrals** (trap 3). At
$\mu_B = 1400$ MeV, $T = 20$ MeV, $\Delta_3 = 80$ MeV the unpaired density
formula is wrong by $-21.1\,\%$ on the paired $u$ modes and $+11.6\,\%$ on the
paired $d$ modes. The entropy is worse: in a fully gapped phase the ratio of
paired to unpaired entropy is $2\times10^{-4}$ at $T = 5$ MeV. This is the
single largest departure from the `alphabag` treatment, where the pairing term
is added to $P$ and differentiated analytically.

**The gap equation has three roots** (trap 4). With a mismatch,
$R(\Delta) = \Delta/2G_D - \mathrm{kernel}$ vanishes at $\Delta = 0$, at a
barrier maximum, and at the physical BCS root, so a fixed bracket returns
whichever it happens to contain. At $\mu^\ast = 450$ MeV, $\eta_D = 0.75$ the
roots are 92.71 MeV at zero mismatch and $(32.4, 92.71)$, $(52.5, 92.71)$,
$(59.9, 92.71)$ at $\delta\mu = 50, 60, 65$ MeV; the free-energy crossover sits
at $\delta\mu_c = 63.59$ MeV, a fraction 0.970 of the weak-coupling
Clogston–Chandrasekhar value $\Delta_0/\sqrt2$, the 3 % deficit being the
finite cutoff.

---

## 8. Regularization: the vacuum at $\Lambda$, the medium at $\Lambda_\mathrm{UV}$

A sharp cutoff on EVERY momentum integral is what the model was fitted with,
and it is unusable at the densities this model is applied to. With
$\Lambda = 602.3$ MeV and quark potentials of 400–500 MeV the medium reaches
the cutoff: the 2SC gap peaks at $\mu^\ast = 0.83\,\Lambda$ and is identically
zero by $\mu^\ast = 1.13\,\Lambda$, where the loop contributions are cut off
entirely and the model silently returns a free gas. Measured here at
$\eta_D = 1$:

| $\mu^\ast/\Lambda$ | 0.664 | 0.830 | 0.996 | 1.129 | 1.328 | 1.494 |
|---|---|---|---|---|---|---|
| $\Delta_3$ [MeV] | 141.06 | 149.48 | 113.33 | 0.00 | 0.00 | 0.00 |

The fix is renormalization-group consistency (Gholami, Hofmann & Buballa, PRD
**111**, 014021 (2025) [arXiv:2408.06704]): require that the answer not depend
on the scale the theory is initialized at,

$$
\lim_{\Lambda_\mathrm{UV}\to\infty}\;\Lambda_\mathrm{UV}\,\frac{\partial\Gamma}{\partial\Lambda_\mathrm{UV}} = 0 .
$$

The vacuum keeps the cutoff it was FITTED at and only the medium runs:

$$
\Omega = \mathcal{V}(\chi) - \frac{1}{2\pi^2}\Big[\int_0^{\Lambda}\!\mathrm{d}p\,p^2\,A_\mathrm{vac}(\chi)
+ \int_0^{\Lambda_\mathrm{UV}}\!\mathrm{d}p\,p^2\big(A(\mu,T,\chi) - A_\mathrm{vac}(\chi)\big)\Big],
$$

with $A_\mathrm{vac}(\chi) = A(\mu=0,T=0,\chi)$ the vacuum integrand AT THE
SAME CONDENSATES — **gaps included**, which is the half that is easy to drop.
The $\Delta$-dependent Dirac sea is a vacuum quantity and keeps $\Lambda$;
leaving it in the medium remainder makes that remainder diverge
QUADRATICALLY in $\Lambda_\mathrm{UV}$ instead of logarithmically, and no
counterterm of the form below can cancel that.

$\lambda = \Lambda_\mathrm{UV}/\Lambda$ is the one parameter that says which
scheme is in use. It is 10 by default and $\lambda = 1$ is the conventional
sharp-cutoff model, exactly and not approximately — see the counterterm's
$\lambda = 1$ limit below.

### 8.1 The medium divergence, and the counterterm that removes it

In an UNPAIRED phase the medium remainder converges and nothing more is
needed: at $T = 0$ the occupations are step functions cut at $k_F$ and the
cutoff never binds, so the unpaired numbers are independent of $\lambda$ to
the last bit. This is the “vacuum/medium separation” that works in ordinary
NJL and fails under pairing, because there the cancellation that makes
$f_\mathrm{vac} = 2\sqrt{M^2+p^2}$ no longer happens and the remainder carries
a logarithm. Summing the six Cooper pairs $(i,j)$ with mean potential
$\bar\mu_{ij} = (\mu^\ast_i + \mu^\ast_j)/2$,

$$
\Omega_\mathrm{med} \;\sim\; -\frac{1}{\pi^2}\sum_{(ij)} \bar\mu_{ij}^2\,\Delta_\eta^2\,\ln\Lambda_\mathrm{UV}.
$$

The counterterm is the **massless scheme** of Gholami *et al.* (their
Eq. C7), which sets $M = 0$ in the renormalization factors — and is therefore
the scheme with a closed form, the one Kunkel *et al.* use, and the one
implemented here:

$$
\delta\Omega_\mathrm{ct} = \frac{1}{\pi^2}\sum_{(ij)}\bar\mu_{ij}^2\,\Delta_\eta^2\;g(\Delta_\eta),
$$

$$
g(\Delta) = \frac{\Lambda}{\sqrt{\Lambda^2+\Delta^2}}
- \frac{\Lambda_\mathrm{UV}}{\sqrt{\Lambda_\mathrm{UV}^2+\Delta^2}}
+ \ln\frac{\Lambda_\mathrm{UV}+\sqrt{\Lambda_\mathrm{UV}^2+\Delta^2}}{\Lambda+\sqrt{\Lambda^2+\Delta^2}} .
$$

The six pairs are the two colour choices per gap:

| gap | pairs |
|---|---|
| $\Delta_1$ | $(d_g,s_b)$, $(d_b,s_g)$ |
| $\Delta_2$ | $(u_r,s_b)$, $(u_b,s_r)$ |
| $\Delta_3$ | $(u_r,d_g)$, $(u_g,d_r)$ |

Within a pair the colour potential $\mu_3$ cancels, so the two pairs of one
gap share a $\bar\mu$ and the sum may equally be written over the three gaps
with a coefficient $2/\pi^2$, which is how Eq. C7 states it.

**Two limits.** $g \to \ln(\Lambda_\mathrm{UV}/\Lambda)$ for large
$\Lambda_\mathrm{UV}$, which is the logarithm the divergence needs cancelled;
and $g = 0$ IDENTICALLY at $\Lambda_\mathrm{UV} = \Lambda$. The second is why
one parameter carries the scheme and no boolean stands beside it (CLAUDE.md
§4): at $\lambda = 1$ the counterterm is not small, it is zero, and the
sharp-cutoff model is recovered bit for bit.

**It is not only a term in $\Omega$.** $\delta\Omega_\mathrm{ct}$ depends on
the potentials, so it carries a density, and on the gaps, so it enters the gap
equations:

$$
\delta n_j = -\frac{\partial\,\delta\Omega_\mathrm{ct}}{\partial\mu^\ast_j}
= -\frac{1}{\pi^2}\sum_{(ij)\ni j}\bar\mu_{ij}\,\Delta_\eta^2\,g(\Delta_\eta),
$$

$$
\frac{\mathrm{d}}{\mathrm{d}\Delta}\Big[\Delta^2 g\Big]
= 3\Delta\Big(\frac{\Lambda}{A} - \frac{\Lambda_\mathrm{UV}}{B}\Big)
+ 2\Delta\ln\frac{\Lambda_\mathrm{UV}+B}{\Lambda+A}
+ \Delta^3\Big(\frac{\Lambda_\mathrm{UV}}{B^3} - \frac{\Lambda}{A^3}\Big),
$$

with $A = \sqrt{\Lambda^2+\Delta^2}$, $B = \sqrt{\Lambda_\mathrm{UV}^2+\Delta^2}$.
So the gap row of the residual is
$\Delta_\eta/(2G_D) - \mathrm{kernel}_\eta + \partial(\delta\Omega_\mathrm{ct})/\partial\Delta_\eta$.
Adding the counterterm to $\Omega$ alone leaves Euler violated and the gaps
solving the wrong equation, and both failures look like a plausible equation
of state. It carries no explicit $T$ and no mass, so it contributes to neither
$s$ nor the scalar density.

### 8.2 What it costs, and one numerical trap

Three quadrature passes instead of one — the medium at $\Lambda_\mathrm{UV}$,
and the vacuum block at $\Lambda_\mathrm{UV}$ and at $\Lambda$ — over a
momentum range ten times wider. At $\lambda = 1$ the two vacuum blocks are the
same integral and the single pass is taken directly.

**The panels have to follow the cutoff.** The pairing quadrature breaks at
each Fermi momentum, which at $\lambda = 1$ covers the whole interval; at
$\lambda = 10$ it leaves everything between the highest $k_F$ and 6023 MeV in
ONE panel, across which no number of Gauss nodes resolves a $1/p$ tail. That
mis-integrates the pairing potential by $4.0\times10^{-7}$ relative — small
enough to look like round-off and large enough to make a warm-started table
and a cold point solve disagree past their convergence gate. Geometric panels
at $\Lambda_\mathrm{UV}/2, \Lambda_\mathrm{UV}/4, \dots$ down to the highest
Fermi momentum bring it to $2.9\times10^{-13}$ at the same node count.

### 8.3 What it buys

The gaps rise monotonically instead of collapsing — at $\eta_D = 1$ the table
above becomes 166.71, 214.51, 251.50, 276.16, 306.85, 327.65 MeV — and the
density ceiling $\Lambda^3/\pi^2 = 2.881$ fm⁻³, which is a regularization
artifact rather than physics, moves to $\Lambda_\mathrm{UV}^3/\pi^2$ and stops
binding. The residual $\lambda$-dependence is the $O(1/p^3)$ tail the
asymptotic expansion drops: doubling $\lambda$ from 10 to 20 moves $\Omega$
and the gaps by $1.4\times10^{-3}$ (2SC) and $1.9\times10^{-3}$ (CFL), which
matches the $<1\,\%$ Gholami *et al.* report for the same doubling.

---

## 9. The vector sector

With $n_q = \sum_j n_j$ the total quark density, the vector interaction energy
and self-energy are

$$
W(n_q) = G_V(n_q)\,n_q^2,
\qquad
\Sigma_V = \frac{\mathrm{d}W}{\mathrm{d}n_q} = \big(2 - \alpha_\mathrm{eff}\big)G_V(n_q)\,n_q,
$$
$$
\alpha_\mathrm{eff} = -\frac{\mathrm{d}\ln G_V}{\mathrm{d}\ln n_q},
\qquad
\mu^\ast_j = \mu_j - \Sigma_V .
$$

Three forms are implemented, selected by the tier-2 choice:

| `vector_form` | $G_V$ | $\alpha_\mathrm{eff}$ |
|---|---|---|
| `"constant"` | $\eta_V G_S$ | $0$ |
| `"power_law"` | $\eta_V G_S\,(n_\mathrm{ref}/n_q)^{\alpha}$ | $\alpha$ |
| `"gluon_exchange"` | $\dfrac{(G_{V0}/G_S)\,G_S}{1+u}$, $u = \dfrac{8k_F^2}{9M_g^2}$, $k_F = \big(\tfrac{\pi^2 n_q}{2}\big)^{1/3}$ | $\dfrac{2}{3}\dfrac{u}{1+u}$ |

The power law's $\eta_V G_S$ is the coupling *at* $n_\mathrm{ref}$, so the
constant form is the $\alpha = 0$ member of the same family; $n_\mathrm{ref}$
is declared in fm⁻³ and multiplied by $(\hbar c)^3$ before it meets $n_q$. The
gluon-exchange form takes its strength from its own $G_{V0}/G_S$ rather than
from $\eta_V$, because a run using it is not a run at a constant coupling of
the same size.

**Why the density dependence exists.** With chiral symmetry restored the
scalar channel dies and the high-density behaviour is set entirely by the
vector term. At constant $G_V$ the interaction energy grows like $n^2$ against
the kinetic $n^{4/3}$ and the sound speed runs away to 1 (Zel'dovich). Writing
$\varepsilon = \sum_i C_i n^{p_i}$, each term contributes
$P_i = C_i(p_i-1)n^{p_i}$, so

$$
c_s^2 = \frac{\sum_i C_i p_i (p_i-1) n^{p_i-1}}{\sum_i C_i p_i n^{p_i-1}},
\qquad
c_s^2(n\to\infty) = \max\big(1-\alpha,\;\tfrac13\big) \text{ for } G_V \sim n^{-\alpha},
$$

since the vector term then has $p_V = 2-\alpha$ against the free-quark $4/3$.
$\alpha = 2/3$ is the marginal exponent, and it is marginal *identically*, not
asymptotically: there the vector term's own pressure is exactly one third of
its own energy density at every density. The gluon-exchange form reaches
$\alpha_\mathrm{eff} = 2/3$ as a consequence of its own structure with no
tuning — 0.062, 0.460, 0.608, 0.653 at $n_q = 10^6, 10^8, 10^9, 10^{10}$ MeV³
— which is why it is the recommended variant. Pairing does not change the
asymptotics: $c_s^2 \to \tfrac13 + \tfrac29\Delta^2/\mu^2$, which dies as
$1/\mu^2$.

Ivanytskyi [arXiv:2409.05859] reaches the same conclusion from the other
direction, and the agreement is worth stating: in a *nonlocal* NJL a local
vector channel gives $\omega_\infty \to -\mu + (3\pi^2\mu/G_V d)^{1/3}$,
$p = \mu^2/4G_V$ and $c_s^2 = 1$ — the identical Zel'dovich runaway — and it
is cured by letting the interaction fall off with momentum. Our
density-dependent $G_V$ and his form factor are two ways of writing the same
suppression; §17 compares them.

**The rearrangement term is mandatory.** Once $G_V$ depends on the density,
$\Sigma_V$ is $\mathrm{d}W/\mathrm{d}n_q$ and not $2G_V n_q$; the ratio is
0.833 at $\alpha = 1/3$ and 0.667 at $2/3$, and using the naive form shifts
$P$ by about 5 % and breaks $n = \mathrm{d}P/\mathrm{d}\mu$ at the first
digit. (A coupling that depends on a mean *field* needs no such term, because
it enters through that field's own equation of motion. A density-dependent one
does.) `vmit`'s $V = a\hbar c\,n_q$ is the $\alpha_\mathrm{eff} = 0$ case with
$a\hbar c = 2G_V$, and needs no rearrangement for that reason.

---

## 10. The totals

With $\mathcal{D} = \sum_\eta \Delta_\eta^2/(4G_D)$ the pairing cost and
$\mathcal{C}$ the condensate cost, the matter sector alone — no leptons, no
photons — assembles as

$$
\begin{aligned}
\Omega &= -\sum_j P_{\mathrm{med},j} - \sum_f \varepsilon_{\mathrm{sea},f} + \mathcal{C}
 - \big(\Sigma_V n_q - W\big) + \delta\Omega_\mathrm{pair} + \mathcal{D}
 + \delta\Omega_\mathrm{ct} - \Omega_\mathrm{vac}, \\
\varepsilon &= \sum_j \varepsilon_{\mathrm{med},j} - \sum_f \varepsilon_{\mathrm{sea},f} + \mathcal{C} + W
 + \varepsilon_\mathrm{pair} + \varepsilon_\mathrm{ct} - \varepsilon_\mathrm{vac}, \\
\varepsilon_\mathrm{pair} &= \delta\Omega_\mathrm{pair} + \mathcal{D} + T\,\delta s + \sum_j \mu^\ast_j\,\delta n_j, \\
\varepsilon_\mathrm{ct} &= \delta\Omega_\mathrm{ct} + \sum_j \mu^\ast_j\,\delta n_j^\mathrm{ct}, \\
s &= \sum_j s_{\mathrm{med},j} + \delta s, \qquad P = -\Omega .
\end{aligned}
$$

Both $\Omega$ and $\varepsilon$ carry the *same* vacuum constant, so
$\Omega_\mathrm{vac} = \varepsilon_\mathrm{vac}$ exactly and Euler survives the
subtraction. The conserved-charge sums are

$$
n_j = n_{\mathrm{med},j} + \delta n_j + \delta n_j^\mathrm{ct},
\qquad
n_q = \sum_j n_j, \quad n_B = \frac{n_q}{3},
$$
$$
n_C = \sum_j q_{f(j)} n_j,
\qquad
n_S = \sum_j s_{f(j)} n_j,
$$

and the colour densities that neutrality sets to zero,

$$
n_3 = \sum_f \big(n_{(f,r)} - n_{(f,g)}\big),
\qquad
n_8 = \sum_f \big(n_{(f,r)} + n_{(f,g)} - 2 n_{(f,b)}\big),
$$

these being the generator densities up to the constants $1/2$ and $1/3$; a row
that must vanish does not care about its normalisation.

**Euler.** The assembly gives

$$
\varepsilon + P = T s + \sum_j \mu_j n_j
$$

with the *physical* mode potentials, since
$\sum_j \mu^\ast_j n_j + \Sigma_V n_q = \sum_j \mu_j n_j$. This is audited at
every solved point. Three assembly bugs found during development each produced
a plausible equation of state and each was caught here: a sign error in
$\varepsilon$ ($O(1)$); the pairing cost and $\delta\Omega_\mathrm{pair}$
dropped from both sums ($7.7\times10^{-3}$, small enough to pass for
quadrature); and $\delta s$ dropped, which fails only at $T>0$.

---

## 11. Leptons, photons and the untracked neutrinos

The leptons are added to the totals **after** the matter and are not part of
the phase: they feel no field the quarks feel, carry no colour, and a phase
does not own them — which is why $\Omega$, $\varepsilon$ and $s$ above contain
none of them and why the phase-adapter block does not either. **Their
integrals are not cut**: the cutoff regularises the four-fermion interaction,
and a free lepton has none, so its momentum integral runs to infinity.

One charged lepton of mass $m_l$ and degeneracy $g_l = 2$ at potential
$\mu_l$:

$$
\begin{aligned}
n_l &= \frac{g_l}{2\pi^2}\int_0^{\infty}\!\mathrm{d}k\,k^2\,(f^+ - f^-), \\
\varepsilon_l &= \frac{g_l}{2\pi^2}\int_0^{\infty}\!\mathrm{d}k\,k^2\,E\,(f^+ + f^-), \\
P_l &= \frac{g_l}{2\pi^2}\int_0^{\infty}\!\mathrm{d}k\,k^2\,T\Big[\ln\big(1+e^{-(E-\mu_l)/T}\big)+\ln\big(1+e^{-(E+\mu_l)/T}\big)\Big], \\
s_l &= \frac{\varepsilon_l + P_l - \mu_l n_l}{T},
\end{aligned}
$$

the same integrands as the medium ones with the upper limit at infinity, so
their $T=0$ limit is the closed forms of §4.1 with
$k_F = \sqrt{\mu_l^2 - m_l^2}$ and no $\min$. They are evaluated by the JEL
expansion of `eos/general`, the one home the Fermi integrals of this
repository have; the forms above are what it approximates and what any
alternative implementation is checked against.

For a **massless** species — every neutrino here — they close in elementary
functions, with $g_\nu = 1$ per flavour and antineutrinos included:

$$
n_\nu = \frac{g_\nu}{6\pi^2}\big(\mu_\nu^3 + \pi^2\mu_\nu T^2\big),
\qquad
P_\nu = \frac{g_\nu}{24\pi^2}\Big(\mu_\nu^4 + 2\pi^2\mu_\nu^2 T^2 + \frac{7\pi^4}{15}T^4\Big),
$$
$$
\varepsilon_\nu = 3P_\nu,
\qquad
s_\nu = \frac{g_\nu}{6}\Big(\mu_\nu^2 T + \frac{7\pi^2}{15}T^3\Big),
$$

and the photon gas, massless with $g_\gamma = 2$ and $\mu = 0$, is
Stefan–Boltzmann:

$$
P_\gamma = \frac{\pi^2}{45}T^4,\quad
\varepsilon_\gamma = 3P_\gamma,\quad
s_\gamma = \frac{4\pi^2}{45}T^3,\quad
n_\gamma = \frac{2\zeta(3)}{\pi^2}T^3 .
$$

The neutrino form at $\mu_\nu = 0$ gives $P_\nu/P_\gamma = 7/8$, which is the
check that the degeneracy is the one stated.

**Which species are present, and at which potential.** Electrons always, at
$\mu_e = \mu_{\nu_e} - \mu_C$ in a beta-equilibrium mode; muons wherever the
`muons` flag allows them, at $\mu_\mu = \mu_e - \mu_{\nu_e}$, which is
muon-decay equilibrium with a transparent muon family. The electron neutrino
appears only when it is trapped, $\mu_{\nu_e}\neq0$; free streaming means
$\mu_{\nu_e} = 0$, and then it carries neither lepton number nor pressure,
which is what free streaming is. The `thermal_neutrinos` flag adds the
flavours NOT tracked in the composition as $\mu = 0$ gases: three when the
electron neutrino free-streams, two when it is trapped, since the trapped
flavour is already counted at its own potential. Photons follow the flag of
the same name.

In a fixed-fraction mode with `leptons=True` none of this applies: there
$\mu_C$ is an unknown fixed by the charge condition, so the leptons are solved
**after** the matter, from the one condition $n_e + n_\mu = n_C$ at a single
potential with $\mu_\mu = \mu_e$ (no neutrinos: a fixed-$Y_C$ table is not a
trapped one). Where the matter turns out *negatively* charged, $n_C \le 0$,
there is nothing for electrons to neutralize and the lepton blocks are empty
with $\mu_e = 0$; positrons are not added. With `leptons=False` the result is
charged matter, which is what a mixed-phase construction needs per pure phase.

---

## 12. The solve

### 12.1 The unknown vector

In this order:

$$
x = \Big(\;\underbrace{M_u, M_d, M_s}_{\text{always}},\;
\underbrace{\{\Delta_\eta\}}_{\text{one per gap the PATTERN makes free}},\;
\underbrace{\mu_3, \mu_8}_{\text{if the pattern pairs at all}},\;
\underbrace{\Sigma_V}_{\text{if } G_V \neq 0},\;
\underbrace{\mu_B, \mu_C}_{\text{always}},\;
\underbrace{\mu_S}_{\text{iff the mode holds } Y_S},\;
\underbrace{\mu_{\nu_e}}_{\text{iff the mode holds } Y_{L_e}}\Big).
$$

### 12.2 The rows, in the order the solver assembles them

| # | row | when |
|---|---|---|
| 1–3 | $M_f - \big[m_f - 4G_S\phi_f + 2K\phi_g\phi_h\big] = 0$ | always, $f = u,d,s$ |
| then | $\dfrac{\Delta_\eta}{2G_D} - \mathrm{kernel}_\eta + \dfrac{\partial\,\delta\Omega_\mathrm{ct}}{\partial\Delta_\eta} = 0$ | one per free gap |
| then | $n_3 = 0$, $n_8 = 0$ | if the pattern pairs |
| then | $\Sigma_V - \mathrm{d}W/\mathrm{d}n_q = 0$ | if $G_V \neq 0$ |
| then | $n_B - n_B^\mathrm{target} = 0$ | always |
| then | $n_C - n_e - n_\mu = 0$ **or** $n_C - Y_C n_B = 0$ | neutrality, or the held charge |
| then | $n_S - Y_S n_B = 0$ | if $Y_S$ held |
| then | $(n_e + n_{\nu_e}) - Y_{L_e} n_B = 0$ | if $Y_{L_e}$ held |

Each row is divided by the scale of the quantity it balances — $\mu_B$ for a
potential, $n_B$ for a density, $(\mu_B/3)^3/\pi^2$ for a gap or colour row —
and the state is accepted when the largest scaled component is below
$10^{-10}$ (`eos.general.solve.RESIDUAL_TOL`).

**The residual handed to the root finder is already scaled**, not merely
judged scaled afterwards: the raw rows span twenty orders of magnitude (mass
rows in MeV against gap and colour rows in MeV³) and a root finder terminates
on its own view of the residual. Without that, every row but the largest is
driven only as far as the largest one needs, and a paired solve reports
$10^{-8}$ where it can reach $10^{-16}$. Note in particular that a gap row is
a **density**, since $\Delta_\eta/2G_D$ carries MeV³, and judging it against a
potential is four orders of magnitude too strict.

### 12.3 The modes

| mode | independent variables | extra unknowns | rows replaced |
|---|---|---|---|
| `beta_eq_neutrinoless` | $(n_B, T)$ | — | neutrality; $\mu_S = 0$ |
| `beta_eq_neutrino_trapped` | $(n_B, Y_{L_e}, T)$ | $\mu_{\nu_e}$ | neutrality + $Y_{L_e}$ |
| `fixed_YC` | $(n_B, Y_C, T)$ | — | $n_C = Y_C n_B$; $\mu_S = 0$ |
| `fixed_YC_YS` | $(n_B, Y_C, Y_S, T)$ | $\mu_S$ | $n_C = Y_C n_B$, $n_S = Y_S n_B$ |

All four close at any temperature. Wherever a temperature is accepted, entropy
per baryon may be given instead, as an outer one-dimensional solve for $T$.
`leptons=True/False` applies to the two fixed-fraction modes; in beta
equilibrium `True` is accepted and ignored and `False` raises, since the
leptons are what the equilibrium is about. The muon lepton *family* as a
conserved charge is not implemented — $Y_{L_\mu}$ raises (see DEFERRED) —
while the muon *species* is available through the flag. Kunkel *et al.* make
the same simplification for the same reason (§17).

### 12.4 The pattern is not a mode

Which condensates survive is decided by free energy, not declared. Every
enumerated candidate — `unpaired`, `2SC`, `CFL`, and one asymmetric `free`
seed that can land on `uSC`, `dSC` or an unequal-gap state — is solved to
self-consistency, and the converged one with the lowest $f = \varepsilon - Ts$
is returned; at fixed $\mu_B$ (which is what the phase adapter does) the
criterion is the largest $P$ instead, and the two agree. A candidate that did
not converge is dropped, not substituted. Every point reports the winner, the
three gaps, $\mu_3$, $\mu_8$ and whether the state is gapless.

Two seeding facts. CFL is electrically neutral **without** electrons, so its
seed puts $\mu_C$ at zero; seeded with an electron-bearing potential the solve
reaches a spurious point with an 11 % flavour-density spread. And in an
unpaired region $\mu_8$ is unconstrained — $n_8$ vanishes identically at
$\mu_8 = 0$ (trap 5) — so it is pinned there rather than solved for. A warm
start is keyed by pattern, because the pattern decides the vector's *layout*;
a density sweep carries the winning pattern's seed and lets the others start
cold, which is also what keeps the enumeration honest.

The cold-start gap seed is one rule, $\Delta_\mathrm{seed} = \max(0.35\,\mu_q, 50)$
MeV, set ABOVE the physical gap rather than below it: the gap equation has a
trivial root at $\Delta = 0$ as well as the physical one, the residual between
them is nearly flat, and a Newton step off a flat residual overshoots onto
zero — where a silently zero gap reads as an unpaired phase rather than as a
failure.

---

## 13. The API

Every entry point takes `par` first and never reaches for a default parameter
set. Non-convergence is a **return value**; a malformed *call* still raises,
because that is a programming error a sampler would otherwise re-make a
million times.

### 13.1 `eos_point`

```python
eos_point(par, mode, species=None, n_B=None, T=None, SnB=None,
          leptons=None, x0=None, patterns=None, backend="reference",
          pair_nodes_per_panel=None, **conditions) -> PointResult
```

| argument | type | legal values | default | meaning |
|---|---|---|---|---|
| `par` | `Parameters` | any frozen set (§3) | **required** | the model parameters |
| `mode` | `str` | `"beta_eq_neutrinoless"`, `"beta_eq_neutrino_trapped"`, `"fixed_YC"`, `"fixed_YC_YS"` | **required** | anything else raises `ValueError` listing the four |
| `species` | `SpeciesFlags` | §5 | `SpeciesFlags()` — unpaired, no muons, no photons | the active degrees of freedom |
| `n_B` | `float` | $>0$, fm⁻³ | **required in practice** | baryon density |
| `T` | `float` | $\ge 0$, MeV | — | temperature. Exactly one of `T`/`SnB` |
| `SnB` | `float` | $>0$, dimensionless | — | entropy per baryon; solved for $T$ by an outer 1-D bracket over $[0.2, 400]$ MeV. An unreachable target comes back as `ok=False`, not a raise |
| `leptons` | `bool` or `None` | `True`, `False`, `None` | `None` $\to$ `False` | fixed-fraction modes only. On a beta mode `True` is ignored, `False` raises |
| `x0` | array, or `{pattern: array}`, or `None` | a converged `point.x`, or `warm_start(point)` | `None` (cold) | warm start. A bare vector seeds the FIRST pattern tried; a mapping seeds each named pattern and leaves the rest cold |
| `patterns` | tuple of `str`, or `None` | any of `"unpaired"`, `"2SC"`, `"uSC"`, `"dSC"`, `"CFL"`, `"free"` | `None` | restricts the enumeration. `None` means `("unpaired","2SC","CFL","free")` with `csc=True` and `("unpaired",)` with it off |
| `backend` | `str` | `"reference"`, `"fast"` | `"reference"` | which flavour of the nine medium integrals. `"fast"` needs numba and raises rather than falling back |
| `pair_nodes_per_panel` | `int` or `None` | $\ge 1$ | `None` $\to$ 24 | Gauss–Legendre nodes per panel of the PAIRING quadrature |
| `**conditions` | | `Y_C`, `Y_S`, `Y_Le` as the mode requires | — | a missing one raises; an extra one raises; `Y_Lmu` raises `NotImplementedError`; `leptons` in here raises `TypeError` |

Returns `PointResult(ok: bool, message: str, point: EoSPoint)`. **`point` is
present even when `ok` is False** — it is the best iterate reached, which is
not a physical state. `message` names the pattern that converged, and says so
explicitly when the *realised* pattern differs from the *requested layout*.

**A pattern restriction is not a guarantee.** A pattern declares which gaps
are free, and a free gap may come out zero, so `patterns=("CFL",)` can return
a 2SC state. Read `point.pattern_realised`, which names the state; read
`point.pattern`, which names the layout, only when re-seeding.

**What `backend="fast"` is worth**, measured on a colour-superconducting solve
at $n_B = 1.2$ fm⁻³ (CPython 3.14.2, numpy 2.3.5): 8.5× at $T = 0$ and 7.1× at
$T = 30$ MeV. Three quarters of an unaccelerated CSC solve is the spectral
diagonalisation at each quadrature node, which is what the compiled pass
replaces. The two flavours sum the modes in different orders, so they agree to
round-off rather than bit for bit, which is why the default is the reference
one and `test/baseline` is frozen against it.

**What `pair_nodes_per_panel` is worth**, on a 9-point `csc=True` table over
$n_B = 1.0$–1.4 fm⁻³ at $T = 0$, `backend="fast"`,
`patterns=("unpaired","2SC","CFL")`: 11.5 s at the shipped 24 nodes, 6.2 s at
16, 3.6 s at 12, with $P$ moving by $3\times10^{-10}$ and $4\times10^{-10}$
relative. Both are ABOVE the $10^{-10}$ the `test/baseline` entries are frozen
at, so lowering it is a deliberate act by a caller who has decided what
accuracy the answer needs.

**What `backend="fast"` also selects: the analytic Jacobian.** With
`backends/jacobian.py` present, the fast backend hands the root finder the
hand-derived Jacobian of the residual instead of letting MINPACK difference
it, and `eos.general.solve.solve_system` then runs a damped Newton attempt
first, falling back to MINPACK's hybrid and Levenberg–Marquardt methods (now
with the same Jacobian) from the original seed when Newton stalls. The
Jacobian is the second derivatives of the three blocks the residual is built
from — the nine cut Fermi gases differentiated under the integral, the pairing
correction by second-order perturbation theory of the quasiparticle spectrum
(`eos.general.pairing.pair_hessian`, with the $T = 0$ Fermi-surface terms of a
gapless state put in at the zero crossings), and the counterterm in closed
form — chained to the unknowns through the linear map of §2. The leptons are
the one differenced column. `verify/` checks it against a central difference
of the residual on every mode (worst $8\times10^{-7}$ on the scaled rows).

Measured on `rg_njl1` (CPython 3.14.2): one CFL Jacobian costs 5–8 ms at
$T = 0$ and 8–14 ms at $T = 50$ MeV against a 2–4 ms residual; a warm-started
CFL step converges in three Newton steps (8e-3 → 2e-5 → 1e-10 → 1e-12) where
MINPACK's forward differences took 70–140 residuals, and a warm 2SC step in
two. Per solve, warm-started, the fast backend is 4–6× faster than the same
backend differencing its residual; a cold CFL start gains 1–3×. Newton has a
basin of its own, and from the unpaired seed at the 2SC → CFL switch it can
converge onto the 2SC root of the CFL layout where the differenced hybrid
method found the CFL root, so `solve_pattern` owes two rescues: a solve that
converged but left its layout is re-seeded with the gaps reset and solved
again (three or four Newton steps either way), and a solve that failed is
followed by the differenced solve from the same seed. Below the CFL onset
every CFL candidate collapses and pays the first rescue, which is why a
table whose CFL layout never wins gains least. Every 200-point table built
both ways agrees to $10^{-8}$ relative in $P$ with the same realised pattern
at every point; the per-table factors are in `output/njl_tables/summary.txt`.

### 13.2 `eos_table`

```python
eos_table(par, mode, species=None, axes=None, fixed=None, leptons=None,
          skip_errors=True, rows=False, progress=None, verbose=False,
          backend="reference", patterns=None,
          pair_nodes_per_panel=None) -> TableResult | list[dict]
```

| argument | legal values | default | meaning |
|---|---|---|---|
| `axes` | `dict` with `"nB"` (required) and exactly one of `"T"` / `"SnB"`, plus optionally any fraction the mode fixes (`"Y_C"`, `"Y_S"`, `"Y_Le"`) as a further axis | `{}` | missing `"nB"`, or zero or two temperature axes, raises |
| `fixed` | `dict` of scalars | `{}` | the fractions the mode needs and the axes do not sweep |
| `skip_errors` | `bool` | `True` | drop non-converged points from their line rather than aborting |
| `rows` | `bool` | `False` | `False` returns a `TableResult`; `True` returns long-format dicts |
| `progress` | callable or `None` | `None` | invoked once per completed line |
| `verbose` | `bool` | `False` | installs the shared one-line printer as that callback |

The remaining arguments are `eos_point`'s and mean the same. A pattern
restriction is validated at `TableSpec` construction, not inside the sweep,
because `skip_errors` would otherwise swallow the error and return an empty
table.

The density axis is warm-started, bisecting a missed step back towards the
last solved point up to 6 times — which is what carries it across the two
thresholds a cold NJL density axis has, the strange quark's onset and the
pairing onset.

`progress` receives the repository's dictionary, the same in every model,
plus one key this model adds:

```
{mode, line, n_lines, temp_key, temp, fracs, n_solved, n_requested,
 elapsed_s, pattern}
```

`TableResult` carries `spec`, `nB` (the requested axis), `lines` (one dict of
conditions per line), `points` (`points[i_line][i_nB]`, shorter than `nB`
where points were skipped) and the convenience properties `P`, `eps`,
`nB_solved`, each a list of arrays, one per line.

**Table row keys** (`rows=True`, or `quark_row(point)`):

`n_B`, `T`, `chi` ($=1$), `phase` ($=$ `"Q"`), `P`, `eps`, `s`, `S_per_B`,
`mu_B`, `mu_C`, `mu_S`, `mu_e`, `Y_C`, `Y_S`, `Y_u`, `Y_d`, `Y_s`, `Y_e`,
`Y_mu-`, `M_u`, `M_d`, `M_s`, `pattern`, `pattern_realised`, `gapless`,
`Delta_1`, `Delta_2`, `Delta_3`, `mu_3`, `mu_8`, plus `Y_nue` and `mu_nue`
where the mode traps, plus whatever the line's conditions were. `chi` and
`phase` exist so that a quark table and a hadronic one concatenate without
renaming.

**Measured cost**, on a 9-point `csc=True` table over $n_B = 1.0$–1.4 fm⁻³ at
$T = 0$: 63.9 s with the defaults, 7.5 s with `backend="fast"`, 3.7 s with
`backend="fast"` and `patterns=("unpaired","2SC","CFL")` — 17.5×, agreeing
with the default table to $1.2\times10^{-10}$ relative in $P$. Neither
argument changes the equations. Per point on a warm-started 250-point sweep:
1.0 ms unpaired/fast, 6.3 ms unpaired/reference, 3.3 ms 2SC at $\lambda = 1$,
114 ms 2SC at $\lambda = 10$, 701 ms for the three-pattern enumeration.

### 13.3 `eos_response`

```python
eos_response(par, mode, species=None, frozen="equilibrium", n_B=None, T=0.0,
             leptons=None, rel_dn=1e-3, dT=0.05, patterns=None,
             backend="reference", **conditions) -> dict
```

| argument | legal values | default |
|---|---|---|
| `frozen` | `"equilibrium"` only (`RESPONSE_FREEZES`) | `"equilibrium"` |
| `rel_dn` | relative density step of the central difference | $10^{-3}$ |
| `dT` | temperature step [MeV] | 0.05 |

Anything else for `frozen` raises `NotImplementedError` — holding the species
fractions, holding $Y_C$, holding the gaps, and the susceptibility matrix
$\chi_{ab} = \partial n_a/\partial\mu_b$ are all recorded in
`docs/DEFERRED.md`.

Returns a plain `dict`. Always present:

| key | at $T = 0$ | at $T > 0$ |
|---|---|---|
| `cs2_isothermal` | the sound speed | $(\partial P/\partial n_B)_T/(\partial\varepsilon/\partial n_B)_T$ |
| `cs2_adiabatic` | equal to the isothermal one | larger by $C_P/C_V$ |
| `C_V` | absent | $T(\partial\sigma/\partial T)_{n_B}$ |
| `C_P` | absent | see below |
| `Gamma_th` | absent | thermal index |
| `converged` | `bool` | |
| `reason` | `str` | |

With $\sigma = s/n_B$ the entropy PER BARYON (CompOSE manual,
arXiv:2203.03209 §3.6):

$$
c_{s,\mathrm{isothermal}}^2 = \frac{(\partial P/\partial n_B)_T}{(\partial\varepsilon/\partial n_B)_T},
\qquad
C_V = T\Big(\frac{\partial\sigma}{\partial T}\Big)_{n_B},
$$
$$
C_P = T\Big[\Big(\frac{\partial\sigma}{\partial T}\Big)_{n_B}
- \Big(\frac{\partial P}{\partial T}\Big)_{n_B}\frac{(\partial\sigma/\partial n_B)_T}{(\partial P/\partial n_B)_T}\Big],
\qquad
c_{s,\mathrm{adiabatic}}^2 = \frac{C_P}{C_V}\,c_{s,\mathrm{isothermal}}^2,
$$
$$
\Gamma_\mathrm{th} = 1 + \frac{P(n_B,T) - P(n_B,0)}{\varepsilon(n_B,T)-\varepsilon(n_B,0)} .
$$

Both sound speeds are named for the thermal variable they hold and never
returned as a bare `cs2`. `Gamma_th` returns `nan` rather than a signed
nonsense where the cold reference is not below the hot one. A stencil point
the solver cannot reach is NOT an exception: the same dict comes back with
`converged=False` and `nan` in every quantity.

`frozen="equilibrium"` means **nothing is held**: the composition
re-equilibrates under the perturbation, and so does the pairing pattern —
every neighbour of the stencil re-runs the enumeration, so a central
difference straddling a pattern boundary returns the chord, not the tangent.
Pass `patterns=("2SC",)` to differentiate *within* one pattern. In a fully
gapped phase $C_V$ is exponentially small at low $T$, because the paired
entropy is: that suppression is real physics, and it is what makes a colour
superconductor cool differently from unpaired quark matter.

### 13.4 `zero_pressure_point`

```python
zero_pressure_point(par, species=None, n_lo=0.02, n_hi=1.5,
                    n_scan=60) -> ZeroPressurePoint
```

Locates the self-bound surface — $P = 0$, $T = 0$, `beta_eq_neutrinoless` —
and returns `ok`, `n_B`, `E_per_A`, `mu_B`, `Y_S`, `mu_S`, `identity_error`,
`P`, `two_flavour`, `below_iron`. See §19.

### 13.5 The layers below the API

These are public, and are what `eos.mixed`, `verify/` and a notebook drawing a
branch use.

| callable | returns | what it is for |
|---|---|---|
| `solve(par, mode, n_B, T, flags, x0, patterns, vac, backend, pair_nodes_per_panel, **fractions)` | `EoSPoint` | the enumeration; `eos_point` without the argument validation |
| `solve_pattern(...)` | `EoSPoint` | ONE mode in ONE declared pattern; the pattern is not chosen here |
| `solve_beta_eq_neutrinoless`, `solve_beta_eq_neutrino_trapped`, `solve_fixed_yc`, `solve_fixed_yc_ys` | `EoSPoint` | the named-mode spellings of `solve` |
| `warm_start(point)` | `{pattern: array}` | the seed a sweep carries |
| `patterns_for(flags, patterns)` | tuple | the enumeration, validated against the flags |
| `thermo_from_mu(par, mu_B, mu_C, mu_S, T, pattern, x0, vac, return_state, backend)` | `(NJLState, ok, err[, x])` | the phase-adapter surface (§15) |
| `state_at(par, M, Delta, Sigma_V, mu_B, mu_C, mu_S, mu_3, mu_8, T, ...)` | `NJLState` | one state EVALUATED; no equilibrium condition imposed |
| `vacuum_solution(par)` | `Vacuum(M, phi, Omega, eps, f_pi)` | the chirally broken vacuum, memoized on `par` |
| `bag_constant(par, vac=None)` | `float` [MeV⁴] | $B_\mathrm{eff}$ of §6 |
| `vector_coupling`, `vector_energy`, `vector_self_energy`, `effective_exponent` | `float` | §9, at a given $n_q$ [MeV³] |
| `counterterm_shape(Delta, Lambda, Lambda_UV)`, `counterterm(par, Delta, mu_star)` | | §8.1 |

---

## 14. What a solved point returns

`EoSPoint`, fm-based throughout: densities in fm⁻³, potentials and masses in
MeV, $P$, $\varepsilon$ and $f$ in MeV/fm³, $s$ in fm⁻³.

| field | meaning |
|---|---|
| `converged`, `error` | the status, and the largest SCALED row. **Test it first**: when `converged` is False every other field is the best iterate reached, not a state |
| `mode` | which of the four equilibria was closed |
| `n_B`, `T` | the independent variables |
| `Y_C` | $n_C/n_B$, the non-leptonic charge fraction — imposed in a fixed-fraction mode, an outcome in a beta one |
| `Y_S` | $n_S/n_B$ |
| `Y_Le` | $(n_e + n_{\nu_e})/n_B$, the electron-family lepton fraction — defined in every mode, not only the one that holds it |
| `pattern` | which candidate LAYOUT won; what `x` has to be unpacked by |
| `pattern_realised` | the pattern the solved gaps ARE; where the two differ this is the one that names the physics |
| `gapless` | whether a quasiparticle branch has reached zero. A gapless state is physical, but comparing candidates by $\Omega$ across one is not, so it is reported rather than silently ranked |
| `Delta` | $(\Delta_1,\Delta_2,\Delta_3)$ [MeV], zero in the channels the winner leaves unpaired |
| `M` | $(M_u, M_d, M_s)$ [MeV] |
| `mu_B`, `mu_C`, `mu_S` | the conserved-charge potentials |
| `mu_3`, `mu_8` | the colour potentials (both pinned to zero in an unpaired region) |
| `mu_e`, `mu_nu` | the lepton potentials |
| `n_u`, `n_d`, `n_s` | the three FLAVOUR densities, $n_f = \sum_a n_{(f,a)}$ |
| `n_e`, `n_mu`, `n_nu` | the lepton densities |
| `n_3`, `n_8` | the COLOUR densities, whose vanishing is what makes the state colour-neutral |
| `n_q` | the total quark density, summed over flavour and colour |
| `Y_u`, `Y_d`, `Y_s`, `Y_e`, `Y_nu` | the same divided by $n_B$ |
| `P`, `eps`, `s` | the totals: matter + leptons + thermal gases |
| `f` | $\varepsilon - Ts$, and what the enumeration ranks by |
| `_state` | the matter block, **INTERNAL**, natural units [MeVⁿ]. `_state.euler_residual()` is dimensionless |
| `x` | the converged unknown vector — which is what a warm start is |

**`n_s` is the strange-quark density, not a scalar density.** Every other
model in this repository returns a field called `n_s` meaning the *scalar*
density, computed through $n_s = (\varepsilon - 3P)/m^\ast$. **Here it is the
number density of $s$ quarks**, the third entry of $(n_u, n_d, n_s)$, and the
collision is a real one: a caller moving from a hadronic model to this one and
reading `n_s` as a scalar density gets a plausible number that means something
else. This model's scalar densities are per flavour, are called $\rho_{s,f}$,
and are not returned on the point at all — they enter through the condensates
$\phi_f$, which are.

The trace identity itself does not survive here, and not for want of care:
$\varepsilon - 3P = M\rho_s$ holds mode by mode for the *medium* pieces alone
and with $P_{k^4}$ rather than $P_{\log}$, and the assembled $\varepsilon$ and
$P$ additionally carry the Dirac sea, the condensate cost $\mathcal{C}$, the
vector terms and the pairing correction, each entering the two with different
weight. So $\rho_{s,f}$ is integrated from the medium integral and corrected
by the Hellmann–Feynman pass, and the trace identity is not used as a
definition anywhere in this model.

**$s$ likewise is integrated, not divided.** The identity
$s = (\varepsilon + P - \sum_j \mu_j n_j)/T$ is Euler rearranged and holds
exactly, but it is used as an *audit* rather than as the definition. $s$ comes
from the entropy integrand plus the pairing correction $\delta s$, because the
identity is a difference of three numbers of order $10^9$ divided by $T$, and
in a cold nearly-degenerate gas — or in a fully gapped phase, where $s$ is
genuinely $e^{-\Delta/T}$ small — the cancellation eats every significant
digit.

---

## 15. The phase-adapter surface

`eos/mixed` consumes this model through one function: given
$(\mu_B, \mu_C, \mu_S, T)$ it returns the phase block, having closed the
model's own internal system — masses, gaps, $\Sigma_V$ **and the two colour
potentials**. Colour neutrality is internal because $\mu_3$ and $\mu_8$ are
not conserved charges of the mixed system: no hadronic phase carries them and
there is nothing across the interface for them to equilibrate with. The slot
carries the **physical** baryon potential, and the seed is not cacheable: the
seed chooses the root, so caching it would change physics rather than speed.

The rows it drives to zero are those of §12.2 **without** the density row and
without the mode's charge rows, since the three potentials are given rather
than solved for:

$$
x_\mathrm{int} = \Big(M_u, M_d, M_s,\; \{\Delta_\eta\}_\mathrm{free},\;
\underbrace{\mu_3, \mu_8}_{\text{if paired}},\;
\underbrace{\Sigma_V}_{\text{if } G_V \neq 0}\Big),
$$

with the three mass equations, one gap equation per free gap, $n_3 = n_8 = 0$
where the pattern pairs, and $\Sigma_V = \mathrm{d}W/\mathrm{d}n_q$ where
there is a vector coupling — in that order, scaled the same way.

The block handed back carries $T$, $\mu_B$, $\mu_C$, $\mu_S$; the three
flavour densities; the flavour potentials
$\mu_f = \mu_B/3 + q_f\mu_C + s_f\mu_S$ and their effective partners
$\mu_f - \Sigma_V$; the effective masses $M_f$; $n_B$, $n_C$, $n_S$; $P$,
$\varepsilon$, $s$ and $\sum_j \mu_j n_j$; and, as declared fields, the three
masses, the three gaps, $\mu_3$, $\mu_8$ and $\Sigma_V$. The adapter
enumerates the patterns at every call and keeps the one with the largest $P$,
which at fixed potentials is the stable one; the *label* of that winner comes
back as the key of the warm-start mapping the adapter returns alongside the
block. No lepton enters it. There is no thermo-at-given-*densities* surface,
so a mixed-phase response that would need one raises.

---

## 16. What the implementation reproduces

### 16.1 The RKH vacuum

| quantity | computed | published (Rehberg *et al.*) |
|---|---|---|
| $M_u$ (vacuum) | 367.648 MeV | 367.7 |
| $M_s$ (vacuum) | 549.479 MeV | 549.5 |
| $(-\phi_u)^{1/3}$ | 241.946 MeV | 241.9 |
| $(-\phi_s)^{1/3}$ | 257.688 MeV | 257.7 |
| $f_\pi$ (quark loop) | 92.391 MeV | 92.4 |

The tier-1 parameters that produce them are in §3, so every number here is
checkable from this document alone.

### 16.2 A second published vacuum, as an independent check

The Hatsuda–Kunihiro set ($\Lambda = 631.4$ MeV, $G_S\Lambda^2 = 1.835$,
$K\Lambda^5 = 9.29$, $m_{u,d} = 5.5$, $m_s = 135.7$ MeV) is not shipped as a
named set, but the same code run at it gives $M_{u,d} = 335.5$ MeV and
$M_s = 528.1$ MeV against the 336 and 528 quoted by Baym *et al.*, Table I and
§IV H. Two independent vacuum fits reproduced by one gap equation is a
stronger statement than one.

### 16.3 The three published “bag constants”, and how they differ

This is the single most-mixed-up number in the NJL compact-star literature.
Three quantities are all called $B$; all three are reported here, and they
differ by well-defined terms.

$$
\begin{aligned}
B_\mathrm{eff} &= \Omega[M_f = m_f] - \Omega[M_f = M_f^\mathrm{vac}]
 && \texttt{bag\_constant(par)} \\
B_\mathrm{sea} &= B_\mathrm{eff} - \mathcal{C}(m_f)
 && \text{sea difference alone} \\
B_0 &= -\Omega_\mathrm{vac} = P_\mathrm{NJL}(\mu = T = 0)
 && \text{the vacuum-subtraction constant itself}
\end{aligned}
$$

where $\mathcal{C}(m_f)$ is the condensate cost of §6 evaluated at the
*current* masses.

| set | $B_\mathrm{eff}^{1/4}$ | $B_\mathrm{sea}^{1/4}$ | $B_0^{1/4}$ | published |
|---|---|---|---|---|
| RKH | 228.93 MeV | **217.59 MeV** | **425.44 MeV** | 217.6 MeV (Schertler *et al.*, via Pagliara & Schaffner-Bielich); $B_0 = 425.4$ MeV (Pagliara & Schaffner-Bielich footnote) |
| HK | 230.00 MeV | **218.28 MeV** | 444.50 MeV | $(218\ \mathrm{MeV})^4 = 296$ MeV/fm³ (Baym *et al.*, Eq. 57) |

So: the numbers quoted as “the NJL bag constant” by Baym *et al.* and by
Pagliara & Schaffner-Bielich are $B_\mathrm{sea}$, and this implementation
reproduces both to the digits they print. `bag_constant()` returns
$B_\mathrm{eff}$, which is larger by $\mathcal{C}(m_f)$ — 22 % in MeV⁴, 5 % in
MeV. $B_\mathrm{eff}^{1/4} = 228.93$ MeV $= 357.49$ MeV/fm³ is what the
colour-dielectric companion model quotes its own $B_g + B_\chi$ against.

**None of these three is a free parameter.** A bag model's $B$ is; here all
three are outputs of the vacuum fit, and $B_0$ in particular is nothing but
the vacuum subtraction $\Omega \to \Omega - \Omega_\mathrm{vac}$ written as an
additive constant. Papers that *add* a further bag constant by hand (Kunkel
*et al.*'s $B = 10$ MeV/fm³, Pagliara & Schaffner-Bielich's $B^\ast$) are
shifting the deconfinement onset relative to a hadronic model, not fixing the
NJL normalisation; see §17.

### 16.4 Solved points

The two neutral solved points at $\mu_B = 1500$ MeV, $T = 0$, $\eta_D = 0.75$,
sharp cutoff ($\lambda = 1$):

```
unpaired   M = (9.84, 8.55, 265.59) MeV,  mu_C = -34.20 MeV,
           n_B = 1.4319 fm^-3,  P = 302.12 MeV/fm^3
2SC        Delta_3 = 95.50 MeV, mu_3 = 0, mu_8 = -2.46 MeV,
           M_s = 243.13 MeV, mu_C = -62.27 MeV,
           n_B = 1.4887 fm^-3, P = 324.75 MeV/fm^3
```

Other anchors: the 2SC gap at $\mu^\ast = 450$ MeV, $\eta_D = 0.75$ is 92.71
MeV; the Clogston ratio $\delta\mu_c/(\Delta_0/\sqrt2)$ is 0.970 against the
weak-coupling 1.

At `Parameters.named("rg_njl1")` with the default $\lambda = 10$ the
2SC $\to$ CFL transition at $T = 0$ lands on the $3.7\,n_0 \to 4.7\,n_0$
density jump Kunkel *et al.* report.

**One deliberate discrepancy.** The specification reports
$M_u, M_d = (11.96, 7.65)$ MeV for the 2SC point and this implementation gives
$(9.73, 8.90)$. The difference is the sign of the hole amplitudes in the
paired scalar density $\delta\rho_{s,f}$, and $n_B = \mathrm{d}P/\mathrm{d}\mu_B$
along the neutral solution decides it: it holds to the finite-difference floor
with the sign used here and fails by $2.6\times10^{-4}$ with the other. Near
chiral restoration $M - m$ is the small difference of two large numbers, which
is why a percent-level change in one scalar density moves the light masses by
20 % and moves $M_s$, $\Delta_3$, $\mu_8$, $\mu_C$, $n_B$ and $P$ not at all.
Recorded in `docs/DEFERRED.md`.

---

## 17. This model against the published NJLs

Every entry below is the same Lagrangian of §1, or a stated subset of it. The
differences are in four places, and they are worth naming before the table:
**(i)** which channels are switched on, **(ii)** how the ultraviolet is
regularized, **(iii)** how the vacuum constant is fixed and whether a further
bag constant is added, **(iv)** whether the pairing pattern and the neutrality
conditions are solved or declared.

### 17.1 Master table

| | flavours | $G_S$, $K$ | diquark | vector | $K'$ | cutoff | pattern | colour neutrality | this repo's counterpart |
|---|---|---|---|---|---|---|---|---|---|
| **`eos.njl`** | 3 | RKH, $\Lambda=602.3$ | $\eta_D$ free, default 0.75 | 3 forms, $\alpha_\mathrm{eff}$ 0–2/3 | **omitted** | RG-consistent, $\lambda = 10$; $\lambda = 1$ reproduces sharp cutoff | solved (enumeration) | solved, $n_3 = n_8 = 0$ | — |
| Rehberg *et al.* 1996 | 3 | **the source** | none | none | — | sharp $\Lambda$ | — | — | `Parameters.default()` |
| Rüster *et al.* 2005 | 3 | RKH | $\eta_D$ varied around the Fierz $3/4$ | none | — | sharp $\Lambda$ | solved | solved | `csc=True`, `lambda_UV=1` |
| Pagliara & Schaffner-Bielich 2008 | 3 | RKH, identical | $G_D = 1.0, 1.2\,G_S$ | $G_V = 0, 0.2\,G_S$ | — | sharp $\Lambda$ | solved | solved | `eta_D=1.0/1.2`, `eta_V=0/0.2`, `lambda_UV=1` |
| Logoteta *et al.* 2012 | 3 | RKH, identical | **none** | **none** | — | sharp $\Lambda$ | — (unpaired) | — | `csc=False`, `eta_V=0`, `lambda_UV=1` |
| Lugones 2016 (review) | 3 | model-dependent | pattern-dependent | — | — | — | — | — | bulk only; no finite-size terms here |
| Baym *et al.* 2018 (QHC18) | 3 | HK ($\Lambda = 631.4$) | $H/G = 1.4$–1.6 | $g_V/G = 0.5$–1.0, **constant** | written, then **set to 0** | sharp $\Lambda$ | solved | solved, $\lambda_3,\lambda_8$ | `lambda_UV=1`, HK numbers, `vector_form="constant"` |
| Gholami *et al.* 2025 | 3 | RKH | $\eta_D = 1$; sets 1–3 | $\eta_V$ | — | **RG-consistent**, massless scheme, $\lambda = 10$ | solved | solved, $\lambda_8$ | **the shipped default** |
| Ivanytskyi 2025 | 3, mass-degenerate | nonlocal, Gaussian $g_k$ | $\eta_D = 0.27$–$0.44$ (bound $<0.765$) | $\eta_V = 0.55$–$1.21$, **nonlocal** | — | **no cutoff** (form factor) | CFL only, by argument | automatic (degenerate) | no counterpart; see §17.7 |
| Kunkel *et al.* 2026 | 3 | RKH | $\eta_D = 1.45$ | $\eta_V = 0.7$ | — | RG-consistent, massless | solved | solved | `Parameters.named("rg_njl1")` |
| MUSES NJL module | 3 | RKH | configurable | configurable | — | RG-consistent, 4 `RG_scheme` spellings | one phase per run | solved | our `analytic` $\equiv$ its `analytic` |

### 17.2 Rehberg, Klevansky & Hüfner, PRC 53, 410 (1996)

The parameter set is theirs and is reproduced digit for digit (§16.1). Their
model has no diquark and no vector channel: it is `SpeciesFlags(csc=False)`
with `eta_V = 0`. Their mass equation
$M_i = m_i - 4G_S\phi_i + 2K\phi_j\phi_k$ is ours unchanged.

### 17.3 Rüster, Werth, Buballa, Shovkovy & Rischke, PRD 72, 034004 (2005)

The neutral three-flavour pairing sector, and the source of two things this
implementation takes literally: the **Appendix A quasiparticle spectrum**
(§7.2), and the gap-index convention $\Delta_{1,2,3} = ds, us, ud$. Their
$\mu_8$ is the halved Gell-Mann one, $\mu_8^\mathrm{theirs} = 1.1547\,\mu_8^\mathrm{ours}$.
`lambda_UV=1` with `csc=True` is their calculation. Expected differences: they
work at fixed $\mu$ and $T$ and map a phase diagram, where we solve at fixed
$n_B$ and rank by free energy — the same criterion, read on the conjugate
axis.

### 17.4 Pagliara & Schaffner-Bielich, PRD 77, 063004 (2008) [arXiv:0711.1119]

**The closest published match to `eos.njl` at `lambda_UV=1`.** Same RKH
parameters, same Rüster dispersion relations, same six gap equations and same
three neutrality conditions $n_Q = n_3 = n_8 = 0$, plus a vector term with
$\omega_0 = 2G_V\langle\psi^\dagger\psi\rangle$ shifting $\mu\to\mu-\omega_0$
— which is our `vector_form="constant"` with $\Sigma_V = 2G_V n_q$.

Their pressure, their Eq. (7),

$$
p = \frac{1}{2\pi^2}\sum_{i=1}^{18}\int_0^\Lambda\!\mathrm{d}k\,k^2|\epsilon_i|
+ 4K\sigma_u\sigma_d\sigma_s - \frac{1}{4G_D}\sum_c|\Delta_c|^2
- 2G_S\sum_\alpha\sigma_\alpha^2 + \frac{\omega_0^2}{4G_V} + p_e ,
$$

is our $P = -\Omega$ of §10 term by term: their $-\mathcal{D}$, their
$-\mathcal{C}$ (the $-2G_S\sum\sigma^2 + 4K\sigma\sigma\sigma$ pair), their
$\omega_0^2/4G_V = -(\Sigma_V n_q - W)$ at constant $G_V$, and their
$\sum_i|\epsilon_i|$ integral is our spectrum sum without the unpaired
subtraction — they take the absolute spectrum, we take the difference from the
unpaired reference and add the unpaired integrals back. The two are
algebraically the same and numerically not: the difference form is what makes
$\delta\Omega_\mathrm{pair}$ vanish *identically* at $\Delta = 0$.

**We reproduce their two bag constants exactly**, $B_0 = (425.44\ \mathrm{MeV})^4$
against their 425.4 and $B_0^\mathrm{ref} = (217.59\ \mathrm{MeV})^4$ against
their 217.6 (§16.3).

*Expected differences.* (1) Their $B^\ast$ — the alternative bag constant that
makes deconfinement coincide with chiral restoration — is a **hadronic
matching choice**, not an NJL quantity, and has no counterpart here: it is the
job of `eos.mixed` or of a Maxwell construction the caller performs. (2) They
use only 18 eigenvalues, i.e. the on-shell BdG reduction; §7.2 measures what
that costs in CFL. (3) They find $\Delta_\mathrm{CFL}\sim160$ MeV at
$\mu = 500$ MeV for $G_D = 1.2G_S$ under a sharp cutoff; at
`lambda_UV=1, eta_D=1.2` we are in the same regime, and at the default
$\lambda = 10$ the gap is substantially larger — that is §8, not a
disagreement.

### 17.5 Logoteta, Bombaci, Providência & Vidaña, PRD 85, 023003 (2012) [arXiv:1203.4159]

The nucleation paper. Same RKH parameter set (their Eq. 15 and the numbers
below it; note their “$K\Lambda^2 = 12.36$” is a typo for $K\Lambda^5$), same
gap equation, **no colour superconductivity and no vector channel** — they say
so explicitly. So the counterpart is
`SpeciesFlags(csc=False)`, `eta_V = 0`, `lambda_UV = 1`.

Their $B_\mathrm{eff}$ deserves care, because it is not a constant and not our
`bag_constant()`. They write

$$
\varepsilon = \sum_i \frac{3}{\pi^2}\int_0^{k_{F_i}}\!\mathrm{d}k\,k^2\sqrt{m_i^2+k^2} + B_\mathrm{eff},
\qquad
B_\mathrm{eff} = B_0 - B,
$$
$$
B = \sum_i\Big[\frac{3}{\pi^2}\int_0^\Lambda\!\mathrm{d}k\,k^2\sqrt{m_i^2+k^2} - 2G\langle\bar q_i q_i\rangle^2\Big] + 4K\langle\bar u u\rangle\langle\bar d d\rangle\langle\bar s s\rangle .
$$

Substituting $B = \sum_f\varepsilon_\mathrm{sea}(M_f) - \mathcal{C}$ shows
that **their $B_\mathrm{eff}$ is exactly our vacuum-subtracted sea-plus-condensate
combination**,

$$
B_\mathrm{eff}^\mathrm{Logoteta} = \Big[-\sum_f\varepsilon_\mathrm{sea}(M_f) + \mathcal{C}\Big] - \varepsilon_\mathrm{vac},
$$

which is the pair of terms our $\varepsilon$ of §10 carries. It is therefore
*density-dependent* — they plot it running from 90 to 195 MeV/fm³ — and it
coincides with our `bag_constant()` only in the chirally restored limit
$M_f \to m_f$. Reading their Fig. 3 as a constant, or comparing its value at
the transition point with our 357.49 MeV/fm³, compares two different
quantities.

*Expected differences.* Their $Q^\ast$ phase is the *flavour-frozen* transient
of nucleation — quark matter at the strangeness content of the hadronic phase
it nucleated out of — which is a `fixed_YC_YS` call here, not a
beta-equilibrium one. And their conclusion that the NJL model makes the quark
star branch unreachable is a statement about the sharp-cutoff, unpaired,
$G_V = 0$ corner of the parameter space; the RG-consistent, paired, vector
corner this model defaults to is a different one.

### 17.6 Lugones, EPJA 52, 53 (2016) [DOI 10.1140/epja/i2016-16053-x]

A review of deconfinement nucleation: quark drops, finite-size effects,
surface and curvature terms, and the growth of a nucleated drop. **None of the
finite-size physics is in `eos.njl`, by design** — this is a bulk homogeneous
equation of state, so surface tension, curvature energy and the colour-Debye
screening of a finite drop are the caller's, and the model supplies the bulk
$\Omega$ they are corrections to. The relevant `eos.njl` entry point for a
nucleation study is `thermo_from_mu` (the bulk phase at given potentials,
§15), together with a `fixed_YC_YS` point for the flavour-conserving transient
phase.

### 17.7 Baym, Hatsuda, Kojo, Powell, Song & Takatsuka, RPP 81, 056902 (2018) [arXiv:1707.04966]

The QHC18 review, and the paper that puts the $K'$ cross-term on the record.
Their mean-field structure is ours with two additions:

$$
M_i = m_i - 4G\sigma_i + K|\epsilon_{ijk}|\sigma_j\sigma_k + \frac{K'}{4}|d_i|^2,
\qquad
\Delta_k = -2d_k\Big(H - \frac{K'}{4}\sigma_i\Big).
$$

Their $K|\epsilon_{ijk}|\sigma_j\sigma_k$ is our $2K\phi_j\phi_k$ once the
double sum over $j,k$ is written out — the same term, not a factor-2
disagreement. The genuinely new pieces are the $K'$ terms, and **they set
$K' = 0$ in every published QHC18 result**, on the argument that its effect is
absorbed into $g_V$ and $H$. That is precisely the position `eos.njl` takes,
and it is why $\eta_D$ here must be read as an effective coupling (§1).

Other differences, all expected:

- **Parameter set.** They use HK ($\Lambda = 631.4$ MeV, $K\Lambda^5 = 9.29$,
  $m_s = 135.7$ MeV), we ship RKH. §16.2 shows the same code reproduces their
  vacuum masses at their set.
- **Coupling range.** They need $H/G \simeq 1.4$–1.6 and $g_V/G = 0.5$–1.0 for
  a $2M_\odot$ star. Our default $\eta_D = 0.75$ is the Fierz value, and
  `rg_njl1` sits at 1.45 — inside their window.
- **Regularization.** Sharp cutoff throughout, i.e. `lambda_UV=1`. They are
  explicit that the NJL model is used only for $n_B \gtrsim 5n_0$ and
  interpolated to APR below, which is a *different* way of avoiding the
  artifacts §8 removes: they stay away from the region where the cutoff bites,
  we push the cutoff away.
- **Not an EoS on its own.** QHC18 is a polynomial interpolation in $\mu_B$
  between APR and NJL. `eos.njl` returns the NJL branch itself; the
  interpolation, if wanted, is the caller's or `eos.mixed`'s.
- **Vector form.** Constant $g_V$ only, $\hat\mu = \mu_q - 2g_V n_q + \dots$,
  which is our `vector_form="constant"`. Their §V A's scaling argument —
  a term $\alpha n^\gamma$ in $\varepsilon$ gives $(\gamma-1)\alpha n^\gamma$
  in $P$ — is the same algebra §9 uses to derive
  $c_s^2 \to \max(1-\alpha,1/3)$; they stop before drawing the
  density-dependent conclusion because they never go to asymptotic density.
- **Bag constant.** Their Eq. (29) is our $B_\mathrm{eff}$ definition and their
  quoted $(218\ \mathrm{MeV})^4$ is our $B_\mathrm{sea}$; see §16.3.
- **$\mu_8$ convention.** $\lambda_8$, so $\mu_8^\mathrm{ours} = \sqrt3\,\mu_8^\mathrm{theirs}$.

### 17.8 Gholami, Hofmann & Buballa, PRD 111, 014021 (2025) [arXiv:2408.06704]

**The model `eos.njl` implements by default.** RG consistency, the medium
divergence $\sum \bar\mu^2\Delta^2\ln\Lambda$, the three counterterm schemes,
and the choice of the massless one. Their Eq. (C7) is our
`counterterm_shape`, with their per-gap coefficient $2/\pi^2$ equal to our
per-pair $1/\pi^2$ summed over the two colour pairs. Their $\lambda \gtrsim 5$
convergence and their $\lambda = 10$ choice are ours; their $<1\,\%$ change
between $\lambda = 10$ and 20 is our measured $1.4$–$1.9\times10^{-3}$.

Two of their conclusions are load-bearing here and are adopted rather than
re-derived: the **massive scheme must not be used** (it inverts the gap
ordering to $\Delta_3 < \Delta_1 = \Delta_2$ and predicts the wrong melting
pattern), and the RG-consistent phase diagram melts CFL in a **dSC** pattern
where the sharp-cutoff one melts it in a uSC pattern. The `free` seed in our
enumeration exists so that a `CFL`-layout solve can fall onto exactly such an
asymmetric state.

Two things their prose does not state and that measurement forced (both in
`docs/DEFERRED.md` and §8):

1. the vacuum piece of the split is $A_\mathrm{vac}(\chi)$ at the SAME
   condensates, **gaps included**; subtracting only the $\Delta$-independent
   Dirac sea leaves the remainder diverging quadratically;
2. the pairing quadrature panels must follow $\Lambda_\mathrm{UV}$, or the
   $1/p$ tail is mis-integrated by $4\times10^{-7}$.

### 17.9 Ivanytskyi, PRD 111, 034004 (2025) [arXiv:2409.05859]

A **nonlocal** three-flavour NJL: the contact currents are smeared by a
Gaussian form factor $g_k = \exp(-k^2/\Lambda^2)$ with $\Lambda = 564$ MeV,
which makes every momentum integral converge and removes the cutoff from the
model entirely. This is the *other* published cure for the disease §8 treats,
and the two are worth holding side by side:

| | `eos.njl` (RG-consistent) | Ivanytskyi (nonlocal) |
|---|---|---|
| how the UV is tamed | vacuum at $\Lambda$, medium at $\Lambda_\mathrm{UV} = 10\Lambda$, plus a counterterm | a form factor that falls faster than $k^{-2}$ |
| vacuum fit | $m_\pi, f_\pi, m_K, m_{\eta'}$ | $M_{0,k=0} = 400$ MeV, $\langle\bar f f\rangle_0 = -(250\ \mathrm{MeV})^3$ |
| 't Hooft term | present | **absent** |
| quark masses | $m_u = m_d = 5.5$, $m_s = 140.7$ MeV | **degenerate**, $m = 3.5$ MeV (“CFLL”) |
| colour/electric neutrality | two colour potentials solved, $\mu_C$ from neutrality | automatic: degenerate masses $\Rightarrow$ equal densities $\Rightarrow$ neutral at one common $\mu$ |
| pattern | enumerated | CFL by construction, with a kinematic argument that $M_s$ never unlocks it above $\mu_B \simeq 1000$ MeV |
| $c_s^2$ asymptote | $\max(1-\alpha_\mathrm{eff}, 1/3)$, reached from above | $1/3$, reached from below |

**Where the two agree, and it matters.** He shows that a *local* vector
channel gives $\omega_\infty \to -\mu + (3\pi^2\mu/G_V d)^{1/3}$, hence
$p = \mu^2/4G_V$, $c_s^2 = 1$ and $\delta = -2/3$ — exactly the Zel'dovich
runaway §9 identifies for constant $G_V$ — and that a local *diquark* channel
gives a divergent gap and $c_s^2 = 1/5$. Our density-dependent $G_V$ and his
momentum-dependent form factor are two ways of suppressing the same
interaction at large momentum, and they reach the same conformal limit. The
difference is where the suppression is written: he puts it in the vertex, we
put it in the coupling's density dependence, which keeps the model local and
keeps the RKH vacuum fit intact.

*Expected differences.* No `eos.njl` parameter set reproduces his numbers: the
Lagrangian is not the same one (no determinant term, degenerate masses,
nonlocal vertices), so this is a comparison of *behaviour*, not of values.

### 17.10 Kunkel, Rather, Gholami, Hofmann & Schaffner-Bielich [arXiv:2607.11537]

Proto-neutron-star evolution with colour superconductivity, and the paper
`Parameters.named("kunkel")` is named for. Their Eq. (1) is our §1 Lagrangian
including the term ordering; their Eq. (3),
$M_\alpha = m_\alpha - 4G_S\sigma_\alpha + 2K\sigma_\beta\sigma_\gamma$, is
our mass equation verbatim; their $\Delta_{1,2,3}$ are the $ds$, $us$, $ud$
channels as ours are. Regularization: sharp $\Lambda' = 602.3$ MeV for the
vacuum, RG-consistent **massless scheme** in medium, couplings
$G_D = 1.45\,G_S$ and $G_V = 0.7\,G_S$. That is `Parameters.named("rg_njl1")`
at the default $\lambda = 10$, and it is the same model.

Their lepton treatment matches ours in an unusual detail: they include muons
in the neutrino-transparent equation of state and **do not** fix a muon lepton
fraction in the trapped one, “for simplicity”. That is exactly the
`beta_eq_neutrino_trapped` we implement, taking $(n_B, Y_{L_e}, T)$ only, with
$Y_{L_\mu}$ raising (§12.3, `docs/DEFERRED.md`) — a shared deferral rather
than a divergence.

*Expected differences.*

- **They add a bag constant by hand**, $B = 10$ MeV/fm³, with
  $P \to P - B$ and $\varepsilon \to \varepsilon + B$, chosen so that a 2SC
  phase exists at $T = 0$ when matched to DD2. That is a matching knob on the
  *hybrid* construction, not part of the quark EoS, and `eos.njl` has no
  field for it: a caller wanting their hybrid curve applies the shift outside
  the model (or, better, uses `eos.mixed`).
- **They work at fixed $(T, \mu_B, \mu_Q)$** and interpolate to fixed
  $Y_{L_e}$ and fixed $s/n_B$ afterwards; `eos.njl` solves at fixed
  $(n_B, Y_{L_e}, T\ \text{or}\ S/B)$ directly. The states are the same; the
  independent variables are not, which is why their mixed 2SC/CFL region is an
  interpolation artifact of the procedure and ours is an enumeration outcome.
- **Their printed $\mu_8$ convention is the halved one** while their code
  (Gholami's) is the full $\lambda_8$; see §2. Compare against the code, not
  the symbol.
- **Their 2SC–CFL mixed phase** is constructed by linear volume-fraction
  interpolation of $s/n_B$ between the two phases. `eos.njl` returns the
  single lowest-free-energy phase at each point and does not interpolate; a
  two-quark-phase mixture is a construction for the caller.

Reproduced from their paper: the $T = 0$ 2SC $\to$ CFL density jump,
$3.7\,n_0 \to 4.7\,n_0$.

### 17.11 The MUSES NJL module (Zenodo 10.5281/zenodo.18249033)

The authors' own code, and what Gholami *et al.* and Kunkel *et al.* are
computed with; it ships in the Calliope release of the MUSES Calculation
Engine. `eos.njl` has been checked against it directly. Four traps in that
comparison, all recorded here because they are what makes a mismatch look like
a physics disagreement when it is not:

1. **`RG_scheme` defaults to `minimal`, and ours is `analytic`.** The minimal
   scheme keeps only the leading logarithm $g\to\ln(\Lambda_\mathrm{UV}/\Lambda)$;
   that agrees with the closed form to 0.01 % at $\Delta = 10$ MeV, is 5 % off
   at $\Delta = 250$ MeV, and at $\eta_D = 1.45$ moves the 2SC window away
   entirely. A comparison must select `analytic`.
2. **Its `Omega` includes the leptons**, ours does not (the phase does not own
   them, §11).
3. **Its `n` is the quark density**, $n_B = n/3$.
4. **Its `mu_8` is the full-$\lambda_8$ convention**, so
   $\mu_8^\mathrm{ours} = \sqrt3\,\mu_8^\mathrm{theirs}$; and its shipped
   config's couplings are $\eta_D = 1.5$, $\eta_V = 0.8$, not the paper's 1.45
   and 0.7.

That comparison is also what found the on-shell-reduction defect of §7.2: our
2SC sector always matched to $10^{-4}$ and our CFL pressure was 6–16 % high,
because the error is controlled by the paired pair's mass mismatch — zero for
$u$–$d$, MeV for anything with $s$ — so it hid in 2SC and broke CFL.

---

## 18. The five traps

Each of these returns a plausible-looking wrong answer; each is stated in full
where it belongs above.

1. **$P$ must come from the logarithm form when the integral is cut** (§4.1) —
   the surface term does not vanish at a finite cutoff, and at $T = 0$ below
   the cutoff the two forms agree, which is how the error hides.
2. **The gap kernel is not $\Delta/|E|$** (§7.4) — it is Hellmann–Feynman on
   the matrix, which carries the branch sign for free.
3. **Paired densities and entropy are not the unpaired Fermi integrals**
   (§7.4).
4. **The gap equation has three roots** under any mismatch (§7.4) — scan, then
   bracket each sign change.
5. **$\mu_8$ is unconstrained in an unpaired region** (§12.4), where $n_8$
   vanishes identically at $\mu_8 = 0$. It is pinned there, never solved for.

Two more that belong to the regularization rather than to the equations: the
vacuum block of the RG split must be taken at the same **gaps** (§8), and the
pairing panels must follow $\Lambda_\mathrm{UV}$ (§8.2).

---

## 19. The self-bound surface, and the two-flavour arm

### 19.1 What is computed

A parametrization whose pressure crosses zero at finite density describes
**self-bound** matter: the phase ends there, with no crust below it. The
quantity reported at that endpoint is the energy per baryon,

$$
E/A = \frac{\varepsilon}{n_B}\quad\text{at}\quad P(n_B) = 0,\; T = 0 \qquad [\mathrm{MeV}],
$$

which is what a lump of this matter at rest weighs per baryon. The entry point
is `zero_pressure_point(par, species)`, and it returns a `ZeroPressurePoint`
carrying `n_B`, `E_per_A`, `mu_B`, `Y_S`, `mu_S`, the identity residual below,
the pressure actually reached, the flavour content requested, and whether
`E_per_A` fell below the 930.4 MeV of iron.

### 19.2 The identity that makes the read self-checking

At $T = 0$ the Euler relation is $\varepsilon + P = \sum_i \mu_i n_i$, so at
$P = 0$ the energy per baryon IS the Gibbs energy per baryon. Expanding the
species potentials in the conserved-charge basis and using beta equilibrium
($\mu_C + \mu_e = 0$) with total electric neutrality ($n_C = n_e$), the charge
and lepton terms cancel exactly:

$$
\sum_i \mu_i n_i = \mu_B n_B + \mu_C n_C + \mu_S n_S + \mu_e n_e = \mu_B n_B + \mu_S n_S ,
$$

and therefore

$$
\boxed{\;E/A = \mu_B + Y_S\,\mu_S\;},\qquad Y_S = n_S/n_B. \tag{$\ast$}
$$

**The full form is the one to use.** $E/A = \mu_B$ is the special case
$Y_S\mu_S = 0$, which holds in every beta-equilibrium mode because strangeness
self-equilibrates there and $\mu_S = 0$. It does NOT hold in a colour-flavour
locked phase, where the condensate pairs the three flavours at equal densities
and unequal masses force unequal potentials: on the CFL surface of
`eos.alphabag` at $\Delta_0 = 100$ MeV, $\mu_S = 40.68$ MeV, and $\mu_B$ alone
gives 895.87 MeV where $E/A$ is 936.55 MeV. $(\ast)$ is checked at every
located root; a root that misses it is a root of something other than $P$.

### 19.3 How the root is found

`eos.general.zero_pressure.locate_zero_pressure` samples $P(n_B)$ on a grid,
takes the LOWEST density at which $P$ rises through zero, and refines it by
Brent's method. The scan is not a convenience: $P(n_B)$ can cross zero more
than once, and a crossing where $P$ FALLS is the top of a mechanically
unstable region rather than a surface. It takes the state as a callable, so it
holds no model and lives in `general/`; a density where the solve does not
converge thins the scan rather than aborting it, and a set with no surface at
all comes back as a status, never as an exception.

### 19.4 The Bodmer–Witten window

| arm | condition | what it says |
|---|---|---|
| three-flavour | $E/A < 930.4$ MeV | strange quark matter is absolutely stable |
| two-flavour | $E/A > 930.4$ MeV | ordinary nuclei are not already decaying into it |

A set failing either is excluded. **Both facts are REPORTED and neither is
asserted**: whether a set sits in the window is a property of the set, so
`below_iron` is a field on the result and no `verify/` entry fails on it. Note
that the same `below_iron = True` reads in opposite directions on the two arms.

**The content requested and the content found can differ, and the result
carries both.** A three-flavour request returns whatever strangeness the
equilibrium actually populated: a set whose surface sits below the $s$ quark's
threshold returns $Y_S = 0$ and the two-flavour number from the three-flavour
call. Read the content off `Y_S`, never off `two_flavour`.

### 19.5 The `two_flavour` flag

Two-flavour quark matter is `beta_eq_neutrinoless` with the strange sector
switched off — which is what it physically is — and **not** `fixed_YC_YS` at
$Y_S = 0$. The distinction is not stylistic. With no populated species
carrying strangeness, $n_S = 0$ holds for a whole range of $\mu_S$, so the row
$n_S = Y_S n_B$ leaves $\mu_S$ undetermined and its Jacobian column null; a
solve then converges on round-off. Switching the sector off removes the
flavour from the unknown vector instead, and CLAUDE.md §4 states the rule
directly: *no sector is enabled or disabled implicitly because its coupling
happens to be zero*. Accordingly `eos.njl` **raises** on `fixed_YC_YS` with
the flag on.

With the flag on:

- $n_s = 0$, $Y_S = 0$ and $\mu_S = 0$ identically, exactly rather than
  approximately;
- $\mu_B = \mu_u + 2\mu_d$ and $\mu_C = \mu_u - \mu_d$ are unchanged, neither
  reading $\mu_s$, so $(\ast)$ collapses to $E/A = \mu_B$;
- $\mu_s$ is reported as the weak-equilibrium value $\mu_d$ — the relation
  $s \leftrightarrow d$ still holds, there is simply nothing populated at it.

**The $s$ condensate stays; only the $s$ Fermi sea is emptied.** The three
strange colour-flavour modes contribute nothing to the medium — no density, no
scalar density, no pressure, energy or entropy — but $\phi_s = \langle\bar s s\rangle$
is still solved from its own field equation, and it still feeds the
light-quark masses through the 't Hooft determinant, $2K\phi_d\phi_s$ in $M_u$
and $2K\phi_u\phi_s$ in $M_d$. That is the physics of two-flavour quark
matter: the strange Fermi sea is empty, while the strange condensate of the
QCD vacuum is not. Dropping the strange field from the equations instead would
move $M_u$, $M_d$ and the subtracted vacuum constant — it would change the
MODEL, not the flavour content asked of it.

**Pairing patterns.** A diquark containing an $s$ quark is not a state
two-flavour matter has. With the flag on, $\Delta_1$ ($ds$) and $\Delta_2$
($us$) have nothing to pair, so `CFL`, `uSC`, `dSC` and `free` leave the
default enumeration, and an explicitly requested one raises. The patterns that
survive are `unpaired` and `2SC`.

---

## 20. Layout

```
parameters.py      Parameters: the RKH set and the three coupling tiers
couplings.py       G_V as a function of the state, and its rearrangement
species.py         SpeciesFlags, the quantum numbers, the gap patterns
thermodynamics.py  quantities computed FROM the state, and the internal
                   solve at fixed conserved-charge potentials
solver.py          the equilibrium conditions and the pattern enumeration
table.py           the warm-started density sweep + progress callback
api.py             eos_point / eos_table / eos_response / zero_pressure_point
responses.py       second derivatives, by re-solved finite differences
backends/          the same medium integrals, jitted, and the analytic
                   Jacobian of the residual; deleting it changes no number,
                   only the time they take
verify/            the physics invariants
```

The pairing machinery itself is NOT here: the gap matrix, the quasiparticle
spectrum, the pairing correction and the Hellmann–Feynman kernels are
`eos/general/pairing.py`, shared with the colour-dielectric model.

Tests: `test/njl/` for the model, `test/general/test_pairing.py` for the
shared pairing machinery.

---

## 21. Not implemented (see `docs/DEFERRED.md`)

The 't Hooft–diquark cross-term $K'$; the trapped muon lepton family as a
conserved charge ($Y_{L_\mu}$); the composition and gap freezes of
`eos_response`, and the susceptibility matrix $\chi_{ab}$; and the
dilaton/colour-dielectric graft, which the specification marks as unverified
for transition order, pairing coexistence and finite temperature.

---

## 22. References

- P. Rehberg, S. P. Klevansky, J. Hüfner, *Phys. Rev. C* **53**, 410 (1996),
  arXiv:hep-ph/9506436 — the parameter set and the vacuum fit.
- M. Buballa, *Phys. Rept.* **407**, 205 (2005), arXiv:hep-ph/0402234 — the
  review.
- A. W. Steiner, S. Reddy, M. Prakash, *Phys. Rev. D* **66**, 094007 (2002),
  arXiv:hep-ph/0205201 — colour neutrality and the CFL $\mu_8$.
- S. B. Rüster, V. Werth, M. Buballa, I. A. Shovkovy, D. H. Rischke,
  *Phys. Rev. D* **72**, 034004 (2005), arXiv:hep-ph/0503184 — the neutral
  three-flavour pairing sector, and Appendix A, the spectrum this model
  diagonalises.
- M. Alford, M. Braby, M. Paris, S. Reddy, *Astrophys. J.* **629**, 969
  (2005), nucl-th/0411016 — the $\Delta^2\mu^2$ CFL term `eos.alphabag` uses.
- G. Pagliara, J. Schaffner-Bielich, *Phys. Rev. D* **77**, 063004 (2008),
  arXiv:0711.1119 — the closest published sharp-cutoff match, and the two bag
  constants of §16.3.
- M. G. Alford, A. Schmitt, K. Rajagopal, T. Schäfer, *Rev. Mod. Phys.* **80**,
  1455 (2008), arXiv:0709.4635 — the review of colour superconductivity.
- D. Logoteta, I. Bombaci, C. Providência, I. Vidaña, *Phys. Rev. D* **85**,
  023003 (2012), arXiv:1203.4159 — unpaired NJL and the chromodielectric model
  in quark-matter nucleation; the density-dependent $B_\mathrm{eff}$ of §17.5.
- G. Lugones, *Eur. Phys. J. A* **52**, 53 (2016),
  doi:10.1140/epja/i2016-16053-x — the nucleation review; finite-size physics
  this model does not carry.
- G. Baym, T. Hatsuda, T. Kojo, P. D. Powell, Y. Song, T. Takatsuka,
  *Rept. Prog. Phys.* **81**, 056902 (2018), arXiv:1707.04966 — QHC18, and the
  't Hooft–diquark cross-term $K'$ (set to zero there as here).
- H. Gholami, M. Hofmann, M. Buballa, *Phys. Rev. D* **111**, 014021 (2025),
  arXiv:2408.06704 — RG-consistent regularization; the model this
  implementation defaults to.
- O. Ivanytskyi, *Phys. Rev. D* **111**, 034004 (2025), arXiv:2409.05859 — the
  nonlocal alternative, and the local-vector/local-diquark pathologies of §9.
- S. Kunkel, I. A. Rather, H. Gholami, M. Hofmann, J. Schaffner-Bielich,
  arXiv:2607.11537 — the `rg_njl1`/`kunkel` couplings, proto-neutron stars.
- M. Hofmann, H. Gholami, C. Pelicer, Y. Yang, the MUSES NJL module,
  Zenodo 10.5281/zenodo.18249033 — the authors' own code.
- S. Typel, M. Oertel, T. Klähn *et al.*, the CompOSE manual,
  *Eur. Phys. J. A* **58**, 221 (2022), arXiv:2203.03209 — the response
  functions.

All bibliography keys are in `docs/eos.bib`.
