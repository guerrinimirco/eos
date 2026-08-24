# CCDM quark-matter EoS — numerical implementation specification

Chiral colour-dielectric model for deconfined $u,d,s$ matter, written for direct coding.
Everything below is checked in `verify_impl.py`; nothing appears here that is not verified
there. Companion to the model summary (`summary.md`), which argues the physics — this
document only states what to compute.

**Units.** $\hbar=c=k_B=1$. Masses, chemical potentials, $T$ in MeV; densities in MeV³;
$P,\varepsilon,\Omega$ in MeV⁴. Convert at the end with $\hbar c=197.3269804$ MeV fm:
$n\,[\mathrm{fm^{-3}}]=n\,[\mathrm{MeV^3}]/(\hbar c)^3$, and the same factor cubed for
$P,\varepsilon$ to MeV fm⁻³.

**Index conventions.** Flavour $f\in\{u,d,s\}$; colour $a\in\{r,g,b\}$; the nine
colour–flavour modes are labelled $j=(f,a)$. Electric charges $q_u=+\tfrac23$,
$q_d=q_s=-\tfrac13$; strangeness $s_u=s_d=0$, $s_s=+1$ (one unit of $S$ per $s$ quark).

---

## 1. The Lagrangian

Four boson fields — dilaton $\varphi$, light scalar $\sigma$, pion triplet
$\boldsymbol\pi$, strange scalar $\zeta$ — plus the vector meson $\omega_\mu$ at L1 and
the diquark interaction at L3:

$$
\mathcal L \;=\; \sum_{f} \bar q_f\left[\,i\gamma^\mu\partial_\mu - M^*_f\,\right]q_f
\;+\;\tfrac12\partial_\mu\varphi\,\partial^\mu\varphi - U(\varphi)
\;+\;\tfrac12\partial_\mu\sigma\,\partial^\mu\sigma
\;+\;\tfrac12\partial_\mu\boldsymbol\pi\!\cdot\!\partial^\mu\boldsymbol\pi
\;+\;\tfrac12\partial_\mu\zeta\,\partial^\mu\zeta - V(\sigma,\boldsymbol\pi,\zeta)
\;+\;\mathcal L_{\rm vec}\;+\;\mathcal L_{\rm pair}
$$

$$
\mathcal L_{\rm vec} = -\,g_\omega\,\bar q\gamma^\mu q\,\omega_\mu
   -\tfrac14 F_{\mu\nu}F^{\mu\nu} + \tfrac12 m_\omega^2\,\omega_\mu\omega^\mu
\qquad[\mathrm{L1}]
$$

$$
\mathcal L_{\rm pair} = G_D\sum_{\eta=1}^{3}
   \bigl|\,\bar q\,i\gamma_5 C\,\epsilon^{ab\eta}\epsilon_{ij\eta}\,\bar q^{\,T}\bigr|^2 ,
\qquad \eta=1,2,3 \leftrightarrow (ds),(us),(ud)
\qquad[\mathrm{L3}]
$$

The quark mass operator carries the dielectric function in the denominator. In the pion
channel the light term is $g_q(\sigma + i\gamma_5\boldsymbol\tau\!\cdot\!\boldsymbol\pi)/\chi$;
at mean field $\boldsymbol\pi\to0$ by parity.

**$\chi$ is a function, not a field.** It has no kinetic term, no potential and no field
equation — it is an abbreviation for a function of $\varphi$, with the exponent $p$ inside:

$$
\boxed{\;\chi(\varphi) = \bigl[1-\bar\varphi^{\,4}\bigr]^{\,p},
\qquad \bar\varphi \equiv \varphi/\varphi_0\;}
$$

**Why the fourth power?** Because $\bar\varphi^{\,4}$, not $\bar\varphi$, is the gluon
condensate. The dilaton is a canonically normalized scalar, mass dimension 1, while
$\langle G^A_{\mu\nu}G^{A\mu\nu}\rangle$ has dimension 4 — so the unique power of $\varphi$
that can represent the condensate is the fourth. The scale anomaly makes this sharp: the
combination that the anomaly fixes,
$4U-\varphi\,\partial U/\partial\varphi$, comes out as a **pure power** of $\bar\varphi$ only
for exponent 4,

$$
4U - \varphi\frac{\partial U}{\partial\varphi} = 4B_g\bigl(1-\bar\varphi^{\,4}\bigr)
$$

whereas the same construction with $\bar\varphi^{2}$, $\bar\varphi^{3}$, $\bar\varphi^{5}$ or
$\bar\varphi^{6}$ leaves a residual $\bar\varphi^{\,n}\ln\bar\varphi$ (verified symbolically).
So $\bar\varphi^{\,4}$ is the condensate variable, and

$$
1-\bar\varphi^{\,4} \;=\; 1-\frac{\langle G^2\rangle_{\rm med}}{\langle G^2\rangle_{\rm vac}}
$$

says the dielectric's deviation from transparency is **linear in the gluon condensate** — a
leading-order response ansatz, and the only candidate with that property. Physically: the
medium becomes transparent to colour exactly in proportion to how much condensate has
melted. It also fixes the two endpoints correctly, $\chi\to0$ at $\bar\varphi=1$ (confinement,
$M^*\to\infty$) and $\chi\to1$ at $\bar\varphi=0$ (perturbative, $M^*\to m_f$), with
$\mathrm{d}\chi/\mathrm{d}\bar\varphi\to0$ at the perturbative end, so the dielectric
correction switches off at the same $\mathcal{O}(\bar\varphi^{4})$ rate as the anomaly itself.

The exponent $p$ outside is a separate matter: $p$ and the function are meaningful only as a
pair (only $\chi^{\,p}$ enters $M^*$), which is why they are written as one object with
$p=1$ as the locked baseline. Squaring the bracket instead would silently double the
confining-end exponent.

**Why not just define the field to be the condensate?** A fair question, since
$\bar\varphi^{\,4}$ is what carries physical meaning. Write
$\Phi \equiv \langle G^2\rangle_{\rm med}/\langle G^2\rangle_{\rm vac} = \bar\varphi^{\,4}$.
The potential and the map both get *simpler*,

$$
U = B_g\bigl[\Phi(\ln\Phi - 1) + 1\bigr],
\qquad
\chi = (1-\Phi)^p
$$

— the power vanishes, and at $p=1$ the dielectric becomes exactly linear in the field with
$\mathrm{d}\chi/\mathrm{d}\Phi=-1$, a constant. So as a **solve variable** $\Phi$ is
genuinely better, and there is one concrete reason it is better that is worth stating:

> $R_1$ has a **spurious root at $\bar\varphi=0$**. Both terms of
> $R_1 = U'(\bar\varphi) - (\chi'/\chi)\sum_f M^*_f\rho_{s,f}$ vanish as $\bar\varphi^{\,3}$
> there — the first as $\bar\varphi^{\,3}\ln\bar\varphi$, the second as $\bar\varphi^{\,3}$ —
> so $R_1(0)=0$ for *any* scalar density. It is an artifact of the parametrization: the
> Jacobian $\mathrm{d}\Phi/\mathrm{d}\bar\varphi=4\bar\varphi^{\,3}$ vanishes, not the physics.
> A Newton solve on $R_1$ in $\bar\varphi$ landed on it from **3 of 5** starting points (one
> even ran negative), while the same equation in $\Phi$ — where
> $\mathrm{d}U/\mathrm{d}\Phi = B_g\ln\Phi\to-\infty$ and no such root exists — converged to
> the genuine root from **all 4** starts.

But $\Phi$ **cannot be the Lagrangian field**, for two reasons. Its kinetic term is not
canonical: $\tfrac12(\partial\varphi)^2 = \tfrac{\varphi_0^2}{32}\Phi^{-3/2}(\partial\Phi)^2$,
whose coefficient diverges at the perturbative point $\Phi\to0$. And $[\Phi]=4$, so
$(\partial\Phi)^2$ has mass dimension 10 — there is no renormalizable Lagrangian in it. The
glueball mass would also stop being $U''$: with no canonical kinetic term,
$\mathrm{d}^2U/\mathrm{d}\Phi^2 = B_g/\Phi$ has dimension $-4$, not $2$.

**So: $\varphi$ is the field, $\Phi=\bar\varphi^{\,4}$ is the better solve variable.** Keep the
Lagrangian in $\varphi$ — that is what makes the kinetic term canonical, $m_\varphi$ meaningful
and the theory renormalizable — and let the *solver* work in $\Phi$, recovering
$\bar\varphi=\Phi^{1/4}$ at the end. Since $\Phi\mapsto\bar\varphi$ is monotone on $[0,1]$
the two are equivalent as parametrizations, and nothing in §§3–9 changes except which
variable the root-finder steps in. If you keep $\bar\varphi$, guard the spurious root
explicitly: reject any converged state with $\bar\varphi<10^{-3}$ unless $\Omega$ there is
genuinely lower than the interior root's.

There is no gluon term. A mean field $\langle A^A_\mu\rangle\neq0$ would break colour and
there are no dynamical gluons at this order, so $-\tfrac14\chi G^2$ would contribute
nothing to $\Omega$; it is omitted rather than written and dropped.

---

## 2. The mean-field Lagrangian

At homogeneous mean field every boson field is replaced by its expectation value and all
gradient terms vanish:

$$
\varphi\to\langle\varphi\rangle,\quad
\sigma\to\langle\sigma\rangle,\quad
\zeta\to\langle\zeta\rangle,\quad
\boldsymbol\pi\to0,\quad
\omega_\mu\to\delta_{\mu0}\,\omega_0 .
$$

From here on $\varphi,\sigma,\zeta,\omega_0$ **denote those expectation values** — no
overbar. The overbar is reserved for two things only: Dirac conjugation $\bar q$, and the
reduced dilaton $\bar\varphi\equiv\varphi/\varphi_0$, which runs from $1$ in the physical
vacuum to $0$ perturbatively.

$$
\boxed{\;
\mathcal L_{\rm MF} = \sum_{f}\bar q_f\left[\,i\gamma^\mu\partial_\mu
   - g_\omega(n_B)\gamma^0\omega_0 - M^*_f\,\right]q_f
\;+\;\mathcal L^{\rm MF}_{\rm pair}
\;-\;\Bigl[\,U(\varphi) + V(\sigma,\zeta) - \tfrac12 m_\omega^2\omega_0^2\,\Bigr]\;}
$$

Without $\mathcal L^{\rm MF}_{\rm pair}$ this is a free Dirac Lagrangian for three
quasiparticle flavours plus a constant, the bracket being the field energy that plays the
role of the bag constant. That is the L0–L2 case, $\Delta_\eta=0$.

### 2.1 The diquark term at mean field

The diquark condensate is a mean field like any other, and it belongs here — the
dispersions of §§5–6 are *derived* from this term, not postulated. Define the three
condensates

$$
s_\eta \;\equiv\; \bigl\langle\, q^{T}\,C\gamma_5\,
   \epsilon^{ab\eta}\epsilon_{ij\eta}\,q \,\bigr\rangle
\qquad (\eta=1,2,3),
$$

each a Lorentz scalar carrying one unit of colour antitriplet and flavour antitriplet.
Substituting $\bar q i\gamma_5 C\bar q^{T}\to s_\eta^*$ in $\mathcal L_{\rm pair}$ and
keeping terms to quadratic order in the fluctuation gives

$$
\boxed{\;
\mathcal L^{\rm MF}_{\rm pair} = \sum_{\eta=1}^{3}
 \left[\;\frac{\Delta_\eta^*\,\mathcal O_\eta + \Delta_\eta\,\mathcal O_\eta^\dagger}{2}
   \;-\;\frac{|\Delta_\eta|^2}{4G_D}\;\right],
\qquad
\mathcal O_\eta \equiv q^{T}C\gamma_5\,\epsilon^{ab\eta}\epsilon_{ij\eta}\,q\;}
$$

$$
\boxed{\;\Delta_\eta \;=\; 2\,G_D\,s_\eta\;}
$$

Three things follow, all verified:

- Substituting $\Delta_\eta=2G_Ds_\eta$ back reproduces $G_D|s_\eta|^2$ **exactly** — the
  mean-field treatment is exact for an auxiliary field appearing quadratically, so nothing
  is approximated at this step beyond the mean field itself.
- $\Omega$ carries $-\mathcal L$, so the pairing contribution is
  $\Omega_\eta = -\Delta_\eta s_\eta + |\Delta_\eta|^2/4G_D$. Its stationarity
  $\partial\Omega/\partial\Delta_\eta=0$ **is** $\Delta_\eta=2G_Ds_\eta$: the gap equation
  and the condensate definition are the same statement. This is why §5.4 solves
  $\partial\Omega/\partial\Delta=0$ rather than imposing a separate self-consistency loop.
- At the solution $\Omega_\eta=-G_D|s_\eta|^2<0$: condensation lowers the grand potential,
  as it must.

The term is **bilinear in $q$ but not diagonal in it** — it couples $q$ to $q^T$, so the
quadratic form is no longer a Dirac operator on $q$ alone. Writing it in the doubled basis
$\Psi=(q,\bar q^{T})^{T}$ makes the quadratic form a matrix whose eigenvalues are the
$E^{(j)}_\Delta$ of §§5–6. **That diagonalization is the only place the doubled basis is
needed**; everything downstream uses the eigenvalues, so a code never manipulates the
$18\times18$ object explicitly — it builds the $9\times9$ gap matrix of §6, diagonalizes
it, and uses the resulting $\Delta^{(j)}$ in scalar dispersions.

Choosing all $\Delta_\eta$ real is a convention (the phases are unobservable in a
homogeneous condensate), so $|\Delta_\eta|^2\to\Delta_\eta^2$ throughout.

### 2.2 Explicit mean-field definitions

$$
M^*_u = \frac{g_q\sigma + m_u}{\chi(\varphi)},\qquad
M^*_d = \frac{g_q\sigma + m_d}{\chi(\varphi)},\qquad
M^*_s = \frac{g_s\zeta + m_s}{\chi(\varphi)}
$$

$$
U(\varphi) = B_g\left[\,\bar\varphi^{\,4}\bigl(\ln\bar\varphi^{\,4}-1\bigr)+1\,\right],
\qquad
U'(\bar\varphi) \equiv \frac{\partial U}{\partial\bar\varphi}
   = 16\,B_g\,\bar\varphi^{\,3}\ln\bar\varphi
$$

$$
V(\sigma,\zeta) = \frac{\lambda}{4}\bigl(\sigma^2-v^2\bigr)^2
  + \frac{\lambda_\zeta}{4}\bigl(\zeta^2-v_\zeta^2\bigr)^2
  - \epsilon_\sigma\sigma - \epsilon_\zeta\zeta + C_0
$$

$$
\frac{\partial V}{\partial\sigma} = \lambda\sigma(\sigma^2-v^2)-\epsilon_\sigma,
\qquad
\frac{\partial V}{\partial\zeta} = \lambda_\zeta\zeta(\zeta^2-v_\zeta^2)-\epsilon_\zeta
$$

$$
\chi'(\bar\varphi) = -4p\,\bar\varphi^{\,3}\bigl(1-\bar\varphi^{\,4}\bigr)^{p-1},
\qquad
\frac{\chi'(\bar\varphi)}{\chi(\bar\varphi)} = \frac{-4p\,\bar\varphi^{\,3}}{1-\bar\varphi^{\,4}}
$$

$$
g_\omega(n_B) = \frac{\bar g_\omega}{1+(n_B/n_c)^2},
\qquad
\frac{\partial g_\omega}{\partial n_B} = -\frac{2\,\bar g_\omega\,n_B/n_c^2}{\bigl[1+(n_B/n_c)^2\bigr]^2},
\qquad
\Sigma_R = \frac{\partial g_\omega}{\partial n_B}\,\omega_0\,n_B
$$

### 2.3 Chemical potentials

Conserved-charge basis. Colour potentials $\mu_3,\mu_8$ enter only at L3:

$$
\mu_{f,a} = \tfrac13\mu_B + q_f\,\mu_C + s_f\,\mu_S + (T_3)_a\,\mu_3 + (T_8)_a\,\mu_8
$$

$$
(T_3)_{r,g,b} = \bigl(+\tfrac12,\,-\tfrac12,\,0\bigr),
\qquad
(T_8)_{r,g,b} = \bigl(+\tfrac13,\,+\tfrac13,\,-\tfrac23\bigr)
$$

Both generators are traceless, so colour drops out of $n_B$. The shifted potential fed to
every momentum integral is

$$
\boxed{\;\mu^*_{f,a} = \mu_{f,a} - g_\omega(n_B)\,\omega_0 - \Sigma_R\;}
$$

**Sign warning.** $\mu_C$ is the *charge* chemical potential and $\mu_e = -\mu_C$. Writing
$+q_f\mu_e$ in place of $+q_f\mu_C$ flips the $u$–$d$ ordering: at $\mu_B=1200$,
$\mu_C=-53.15$ MeV the correct values are $\mu_u=364.6 < \mu_d=417.7$ MeV ($d$ excess,
electrons present), whereas $+q_f\mu_e$ gives $\mu_u=435.4 > \mu_d=382.3$ MeV — wrong.

Leptons are free Fermi gases evaluated with the **same §3 integrals**, with
$\mu_\ell = \mu_{L_\ell} - \mu_C$; in cold catalysed matter $\mu_{L_\ell}=0$ so
$\mu_e = \mu_\mu = -\mu_C$. Their masses and degeneracies:

| species | mass [MeV] | $g$ | present when |
|---|---|---|---|
| $e^-$ | $0.510999$ | 2 | always (unless the row has no leptons) |
| $\mu^-$ | $105.658$ | 2 | $\mu_\mu > m_\mu$; otherwise its density is zero |
| $\nu_e,\nu_\mu$ | $0$ | **1** | only when the row traps neutrinos (R3) |

Neutrinos carry $g=1$, not 2 — they are left-handed only, and no antineutrino term appears
because the trapped species is fixed by the lepton-number constraint. The muon threshold is
a genuine discontinuity in $\partial n/\partial\mu$: test $\mu_\mu>m_\mu$ rather than
letting the integral return a tiny spurious value.

---

## 3. Base thermodynamic integrals

One colour–flavour mode, spin degeneracy $g=2$. With
$E_k=\sqrt{k^2+M^{*2}}$ and

$$
f^+_k = \left[1+e^{(E_k-\mu^*)/T}\right]^{-1},
\qquad
f^-_k = \left[1+e^{(E_k+\mu^*)/T}\right]^{-1}
$$

$$
\mathcal N(M^*,\mu^*,T) = \frac{g}{2\pi^2}\int_0^\infty\!\!dk\;k^2\bigl[f^+_k-f^-_k\bigr]
\qquad\text{(number: antiparticles *subtract*)}
$$

$$
\mathcal R_s(M^*,\mu^*,T) = \frac{g}{2\pi^2}\int_0^\infty\!\!dk\;k^2\,\frac{M^*}{E_k}
   \bigl[f^+_k+f^-_k\bigr]
\qquad\text{(scalar: antiparticles *add*)}
$$

$$
\mathcal P(M^*,\mu^*,T) = \frac{g}{6\pi^2}\int_0^\infty\!\!dk\;\frac{k^4}{E_k}
   \bigl[f^+_k+f^-_k\bigr]
$$

$$
\mathcal E(M^*,\mu^*,T) = \frac{g}{2\pi^2}\int_0^\infty\!\!dk\;k^2 E_k
   \bigl[f^+_k+f^-_k\bigr]
$$

$$
\mathcal S(M^*,\mu^*,T) = -\frac{g}{2\pi^2}\int_0^\infty\!\!dk\;k^2
   \sum_{\pm}\Bigl[f^\pm_k\ln f^\pm_k + (1-f^\pm_k)\ln(1-f^\pm_k)\Bigr]
$$

These five satisfy, per mode and to machine precision,

$$
\mathcal N = \frac{\partial\mathcal P}{\partial\mu^*},\qquad
\mathcal R_s = -\frac{\partial\mathcal P}{\partial M^*},\qquad
\mathcal S = \frac{\partial\mathcal P}{\partial T},\qquad
\mathcal E = -\mathcal P + \mu^*\mathcal N + T\,\mathcal S
$$

**Assert all four in the code.** A wrong prefactor in any one cannot survive them. Note
the Euler relation contains $T\mathcal S$ and **not** $M^*\mathcal R_s$: the scalar density
is a response, not a conserved charge.

### 3.1 Quadrature — the detail that decides whether this works

A single Gauss–Legendre panel over $[0,k_{\max}]$ **fails at low $T$**. The occupation is
a step of width $\sim T$ inside a range of $\sim1500$ MeV. Measured on $\mathcal P$ at
$M^*=100$, $\mu^*=500$, $T=0.5$ MeV, against the $T=0$ closed form:

| nodes (one panel) | deviation |
|---|---|
| 900 | $9.4\times10^{-4}$ |
| 2000 | $4.3\times10^{-5}$ |
| 5000 | $2.2\times10^{-5}$ |
| 12000 | $2.2\times10^{-5}$ |

Two effects are superposed and must not be confused. The **converged floor**
$2.2\times10^{-5}$ is the genuine $O(T^2)$ difference between $T=0.5$ MeV and the $T=0$
closed form — real physics, and the accuracy an implementation should expect. Everything
above that floor is **quadrature error**, and it grows as $T$ falls: halving $T$ raises it
instead of quartering it, which is the diagnostic that separates the two.

Fix: split the integral at the Fermi surface. With $k_F=\sqrt{\mu^{*2}-M^{*2}}$ when
$\mu^*>M^*$, integrate over the panels

$$
\bigl[0,\;k_F-25T\bigr],\quad \bigl[k_F-25T,\;k_F+25T\bigr],\quad \bigl[k_F+25T,\;k_{\max}\bigr]
$$

with $k_{\max}=\max(\mu^*,M^*)+45T+12M^*+200$ MeV. Measured head-to-head on the same
$\mathcal P$:

| scheme | nodes/panel | panels | total | deviation |
|---|---|---|---|---|
| single panel | 900 | 1 | 900 | $9.4\times10^{-4}$ |
| single panel | 2000 | 1 | 2000 | $4.3\times10^{-5}$ |
| single panel | 5000 | 1 | 5000 | $2.18\times10^{-5}$ |
| split at $k_F$ | 100 | 3 | 300 | $2.18\times10^{-5}$ |
| split at $k_F$ | 400 | 3 | 1200 | $2.18\times10^{-5}$ |

The split scheme is **already at the $O(T^2)$ floor with 100 nodes per panel** — 300 total
against 5000 for a single panel, a factor $\sim17$ fewer evaluations for the same accuracy,
and it is insensitive to further refinement. 400 per panel is a comfortable default. Note
also that single-panel accuracy is **not monotone in $N$** (400 nodes is worse than 300 —
$1.3\times10^{-2}$ against $1.4\times10^{-3}$), because whether a node happens to land near
the Fermi step is accidental; that non-monotonicity is itself a symptom of the unresolved
step, and it disappears once the integral is split.

### 3.2 $T=0$ closed forms

Use these below a switch temperature ($T\lesssim1$ MeV). If $\mu^*\le M^*$ **every one of
them is exactly zero** — return zero, do not integrate. Otherwise, with
$k_F=\sqrt{\mu^{*2}-M^{*2}}$:

$$
\mathcal P = \frac{g}{48\pi^2}\Bigl[\,2k_F^3\mu^*
   - 3M^{*2}\Bigl(\mu^*k_F - M^{*2}\ln\frac{k_F+\mu^*}{M^*}\Bigr)\Bigr]
$$

$$
\mathcal N = \frac{g}{6\pi^2}\,k_F^3,
\qquad
\mathcal R_s = \frac{g}{4\pi^2}M^*\Bigl(k_F\mu^* - M^{*2}\ln\frac{k_F+\mu^*}{M^*}\Bigr)
$$

$$
\mathcal E = \frac{g}{16\pi^2}\Bigl[\,2k_F^3\mu^*
   + M^{*2}\Bigl(\mu^*k_F - M^{*2}\ln\frac{k_F+\mu^*}{M^*}\Bigr)\Bigr],
\qquad \mathcal S = 0
$$

---

## 4. Phase A — unpaired quark matter (L0–L2)

Colour-symmetric: $\mu_3=\mu_8=0$, so $\mu^*_{f,r}=\mu^*_{f,g}=\mu^*_{f,b}\equiv\mu^*_f$
and each flavour carries $N_c=3$ identical modes.

$$
n_f = 3\,\mathcal N(M^*_f,\mu^*_f,T),
\qquad
\rho_{s,f} = 3\,\mathcal R_s(M^*_f,\mu^*_f,T)
$$

$$
n_B = \tfrac13\sum_f n_f,
\qquad
n_C = \sum_f q_f n_f,
\qquad
n_S = \sum_f s_f n_f = n_s
$$

### 4.1 The grand potential

$$
\boxed{\;
\Omega = U(\varphi) + V(\sigma,\zeta)
 - 3\sum_f \mathcal P(M^*_f,\mu^*_f,T)
 - \tfrac12 m_\omega^2\omega_0^2
 - \sum_\ell \mathcal P_\ell - \mathcal P_\gamma \;}
$$

with $\mathcal P_\gamma=\pi^2T^4/45$ (photons; drop at $T=0$) and $\mathcal P_\ell$ the
lepton gases from §3 with $g=2$ and $M^*=m_\ell$ (neutrinos $g=1$, $m=0$).

### 4.2 The system to solve

Unknowns $\vec X = (\bar\varphi,\ \sigma,\ \zeta,\ \omega_0)$ plus whatever the closure row
adds (§7). Residuals, all $=0$:

$$
R_1 = U'(\bar\varphi) - \frac{\chi'(\bar\varphi)}{\chi(\bar\varphi)}
      \sum_f M^*_f\,\rho_{s,f}
    \;=\; U'(\bar\varphi) + \frac{4p\,\bar\varphi^{\,3}}{1-\bar\varphi^{\,4}}
      \sum_f M^*_f\,\rho_{s,f}
$$

$$
R_2 = \frac{\partial V}{\partial\sigma} + \frac{g_q}{\chi(\bar\varphi)}
      \bigl(\rho_{s,u}+\rho_{s,d}\bigr)
$$

$$
R_3 = \frac{\partial V}{\partial\zeta} + \frac{g_s}{\chi(\bar\varphi)}\,\rho_{s,s}
$$

$$
R_4 = m_\omega^2\,\omega_0 - g_\omega(n_B)\,n_q,
\qquad n_q \equiv 3n_B
$$

**The vector source is the quark number density $n_q=3n_B$, not $n_B$** — the coupling in
$\mathcal L_{\rm vec}$ is to $\bar q\gamma^\mu q$. Using $n_B$ understates $\omega_0$ by a
factor 3 and the repulsive energy by 9.

$R_1$–$R_3$ are **minima** of $\Omega$, with the boundary minimizer $\sigma=0$ admitted —
that is how chiral restoration appears. $R_4$ is a stationary **maximum** in $\omega_0$, as
always for a repulsive vector field: solve it as a fixed point, never by minimizing.

**Branch enumeration, not iteration.** Below the deconfinement onset two chiral branches
coexist at fixed $\varphi$ (broken, $\sigma\simeq f_\pi$, quarks too heavy to appear; and
restored, $\sigma=0$, quarks present). A single alternating loop two-cycles between them
and exits with a *mixed* state — $\sigma$ from one branch, $\omega_0$ from the other —
which reads as a spuriously deep minimum at zero quark density. Solve each branch to
self-consistency separately, then compare by $\Omega$.

### 4.3 Outputs

$$
P = -\Omega,
\qquad
n_i = -\frac{\partial\Omega}{\partial\mu_i}\ \ (i=B,C,S),
\qquad
s = -\frac{\partial\Omega}{\partial T}
$$

Evaluated at the stationary point these become explicit — the implicit field dependence
drops out by $\partial\Omega/\partial(\text{field})=0$:

$$
P = 3\sum_f \mathcal P(M^*_f,\mu^*_f,T) + \sum_\ell \mathcal P_\ell + \mathcal P_\gamma
    - U(\varphi) - V(\sigma,\zeta) + \tfrac12 m_\omega^2\omega_0^2
$$

$$
s = 3\sum_f \mathcal S(M^*_f,\mu^*_f,T) + \sum_\ell \mathcal S_\ell
    + \tfrac{4}{45}\pi^2T^3
$$

$$
\varepsilon = 3\sum_f \mathcal E(M^*_f,\mu^*_f,T) + \sum_\ell \mathcal E_\ell
    + \mathcal E_\gamma + U(\varphi) + V(\sigma,\zeta) - \tfrac12 m_\omega^2\omega_0^2
$$

with $\mathcal E_\gamma = 3\mathcal P_\gamma$. Note the **signs**: $U+V$ enter $P$
negatively and $\varepsilon$ positively; the $\omega_0$ term does the opposite. Getting
either backwards is the most common assembly error, and the audit below catches it.

**Mandatory audit at every solved point:**

$$
\varepsilon = -P + T s + \sum_i \mu_i n_i
$$

with $i$ running over the *independent* potentials of the closure row (and lepton terms
where present). Also check $P=-\Omega$ from the two assemblies independently.

Sound speed: $c_s^2 = dP/d\varepsilon$ **along the solved branch**. Never finite-difference
across a first-order transition — take one-sided derivatives on each branch and leave the
transition as a gap.

### 4.4 Effective bag constant

Not an input. At the physical vacuum ($\bar\varphi=1$, $\sigma=f_\pi$, $\zeta=\zeta_0$) the
field energy is zero by construction; at the perturbative point ($\bar\varphi\to0$,
$\sigma\to0$, $\zeta\to0$) it is

$$
B_{\rm eff} = \underbrace{U(0)-U(\varphi_0)}_{B_g}
 \;+\;\underbrace{V(0,0)-V(f_\pi,\zeta_0)}_{B_\chi}
$$

For $B_g^{1/4}=150$, $m_\sigma=550$, $m_\zeta=980$ MeV: $B_g=(150)^4$,
$B_\chi=(230)^4$, $B_{\rm eff}=(240\ \mathrm{MeV})^4 = 429$ MeV fm⁻³. **The chiral sector
supplies the larger part** — $B_g$ is not the whole bag constant.

$U(0)=B_g$ requires the limit $\bar\varphi^4\ln\bar\varphi^4\to0$ to be special-cased,
otherwise the glue potential returns NaN at the perturbative point.

---

## 5. Phase B — 2SC with $m_u \neq m_d$ (L3)

### 5.1 Which modes pair, and what replaces what

The nine colour–flavour modes split into two disjoint sets. With $\eta=3$, i.e. $(ud)$
pairing, the antisymmetric tensor $\epsilon^{ab3}\epsilon_{ij3}$ pairs $u$ and $d$ of
colours $r,g$ only:

| set | modes | count | treatment |
|---|---|---|---|
| paired | $(u,r),(u,g),(d,r),(d,g)$ | 4 | quasiparticle dispersion $E^\pm_\Delta$ |
| unpaired | $(u,b),(d,b),(s,r),(s,g),(s,b)$ | 5 | ordinary Fermi integrals of §3 |

**Answer to the bookkeeping question, explicitly: the pairing term *replaces* the four
paired modes' Fermi integrals — it does not add to them. The five unpaired modes keep the
§3 integrals unchanged. Every mode is counted exactly once.**

Two equivalent ways to write that. They differ by a $\Delta$-independent constant, and
only one is safe to code.

**Form (i) — replacement (what the literature writes).** Sum §3 integrals over the five
unpaired modes and the quasiparticle expression over the four paired ones:

$$
\Omega_{\rm quark} = -\!\!\sum_{j\in\rm unpaired}\!\!\mathcal P\bigl(M^*_j,\mu^*_j,T\bigr)
 \;+\;\Omega^{\rm dir}_{\rm pair}\;+\;\frac{\Delta^2}{4G_D}
$$

$$
\Omega^{\rm dir}_{\rm pair} = -\frac{2}{2\pi^2}\int_0^\Lambda\!\!dk\,k^2
 \sum_{e=\pm}\left[\bigl|E^e_\Delta(k)\bigr|
   + 2T\ln\Bigl(1+e^{-|E^e_\Delta(k)|/T}\Bigr)\right]
$$

The outer factor 2 counts the two colour pairings ($r$–$g$ and $g$–$r$) at fixed $\eta$;
together with the two branches $e=\pm$ this covers all four paired modes.

**That factor 2 is valid only because $\mu_3=0$ in 2SC.** The two pairings share $\bar\mu$
always, but their mismatches differ by exactly $\mu_3$. In the 2SC pattern the $\{r,g\}$
pair enters symmetrically, so $n_3=0$ is solved by $\mu_3=0$ (verified: $n_3$ changes sign
through zero at $\mu_3=0$, $n_3=\pm7.5\times10^4$ MeV³ at $\mu_3=\mp10$ MeV) and the two
pairings are degenerate. For **uSC/dSC, where $\mu_3\neq0$, the shortcut fails** and each
colour pairing must be summed separately with its own $\delta\mu$. Code the general
two-pairing sum, not the factor 2, if you intend to enumerate those patterns.

**Form (ii) — correction (recommended).** Treat all nine modes as unpaired, then add a
correction that vanishes identically at $\Delta=0$:

$$
\boxed{\;
\Omega_{\rm quark} = -\!\!\sum_{j=1}^{9}\!\mathcal P\bigl(M^*_j,\mu^*_j,T\bigr)
 \;+\;\delta\Omega_{\rm pair}\;+\;\frac{\Delta^2}{4G_D}\;}
$$

$$
\delta\Omega_{\rm pair} = -\frac{1}{\pi^2}\int_0^\Lambda\!\!dk\,k^2
 \sum_{r=\pm}\sum_{e=\pm}\Bigl\{\bigl(|E^{e,r}_\Delta| - |E^{e,r}_0|\bigr)
   + 2T\Bigl[\ln\bigl(1+e^{-|E^{e,r}_\Delta|/T}\bigr)
            - \ln\bigl(1+e^{-|E^{e,r}_0|/T}\bigr)\Bigr]\Bigr\}
$$

where $E^{e,r}_0 \equiv E^{e,r}_\Delta\big|_{\Delta=0}$, the index $e=\pm$ labels the two
branches of §5.2, and **$r=\pm$ labels particle and antiparticle**: the $r=-$ terms are
obtained by $\bar\mu\to-\bar\mu$, $\delta\mu\to-\delta\mu$ in §5.2.

**Do not drop the antiparticle branches.** They are not a finite-$T$ refinement. Measured
at $M^*=60$, $\mu^*=450$ MeV, $\Lambda=600$ MeV: the antiparticle correction is
$-1.48\times10^{7}$ MeV⁴ against $-1.45\times10^{8}$ for the particle branches — **10% of
the pairing term**, same sign, and it does **not** vanish at $T=0$. It also grows with the
cutoff ($-5.2\times10^{7}$ at $\Lambda=1000$ MeV), so omitting it is a $\Lambda$-dependent
error in $\Omega$ that will not cancel anywhere.

**Why form (ii).** The difference between the two is exactly $\Delta$-independent —
verified to $1.7\times10^{-16}$ across $\Delta = 0$–$80$ MeV — so both give the *same gap
equation*. But that constant is **not** the four modes' $-\sum\mathcal P$: at $M^*=60$,
$\mu^*\simeq450$ MeV, $T=20$ MeV, $\Lambda=600$ MeV it is $-1.4264\times10^{9}$ MeV⁴
against $-1.3982\times10^{9}$, a difference of $-2.82\times10^{7}$ MeV⁴ — **2% of the term,
and $\Lambda$-dependent**. It is the Dirac-sea piece the $|E|$ integral carries and the
Fermi integrals do not. Using form (i) without subtracting its own $\Delta=0$ value
silently adds that constant to $\Omega$, shifting $P$ and $\varepsilon$ by a
cutoff-dependent amount while leaving $\Delta$ untouched — so the gap looks right and the
EoS is wrong. Form (ii) cannot make that error: at $\Delta=0$ it reproduces the plain
nine-mode sum to $0$ relative error.

If you prefer form (i), the fix is one line: use
$\Omega^{\rm dir}_{\rm pair}(\Delta) - \Omega^{\rm dir}_{\rm pair}(0)$ and keep the four
modes in the unpaired sum, which is form (ii).

### 5.2 The dispersion

With $E_{k,f}=\sqrt{k^2+M_f^{*2}}$ and $\mu^*$ from §2.3 including $\mu_3,\mu_8$,

$$
\bar\mu = \tfrac12\bigl(\mu^*_{u,r}+\mu^*_{d,g}\bigr),
\qquad
\delta\mu = \tfrac12\bigl(\mu^*_{d,g}-\mu^*_{u,r}\bigr)
$$

$$
\boxed{\;
E^\pm_\Delta(k) = \sqrt{\Bigl(\tfrac{E_{k,u}+E_{k,d}}{2}-\bar\mu\Bigr)^2 + \Delta^2}
 \;\pm\;\Bigl(\tfrac{E_{k,d}-E_{k,u}}{2} - \delta\mu\Bigr)\;}
$$

Verified: at equal masses and potentials this collapses to BCS $\sqrt{\xi^2+\Delta^2}$ with
degenerate branches; at $\Delta\to0$ it returns the unpaired
$\{|E_{k,u}-\mu^*_u|,\,|E_{k,d}-\mu^*_d|\}$; and it turns gapless exactly at
$\delta\mu=\Delta$ — the Clogston–Chandrasekhar threshold, recovered rather than imposed.

**This closed form is exact for 2SC, and only for 2SC.** Checked against the full
Bogoliubov–de Gennes problem of §6.3 over twenty combinations of $(M^*_u,M^*_d,\mu^*_u,
\mu^*_d,\Delta,k)$: agreement to $1.1\times10^{-13}$ MeV. The reason is structural — with
only $\Delta_3\neq0$ the gap matrix is **block-diagonal in $2\times2$ blocks**, each gapped
mode coupling to exactly one partner ($ru$–$gd$ and $rd$–$gu$), so the two-body reduction is
the exact block diagonalization. That property fails for uSC, dSC and CFL, where a gap
eigenvector can mix three modes; those patterns need §6.3. If you only ever want 2SC, this
section is self-contained and you can skip the matrix machinery entirely.

Use $|E^e_\Delta|$ in the integrands. The $-$ branch goes negative in the gapless window,
and the absolute value is what the quasiparticle expression requires.

### 5.3 Densities — paired modes are not given by the unpaired formulas

$$
n_j = -\frac{\partial\Omega_{\rm quark}}{\partial\mu_j},
\qquad
\rho_{s,j} = -\frac{\partial\Omega_{\rm quark}}{\partial M^*_j}
\qquad\text{for every mode } j
$$

For the five **unpaired** modes this reduces to §3: $n_j=\mathcal N(M^*_j,\mu^*_j,T)$,
$\rho_{s,j}=\mathcal R_s(M^*_j,\mu^*_j,T)$, and these are **independent of $\Delta$** —
verified: $n_{(s,b)}$ does not move between $\Delta=0$ and $80$ MeV.

For the four **paired** modes it does not. Pairing redistributes occupation around the
Fermi surface, so their densities must come from the derivative of the full
$\Omega_{\rm quark}$ — analytically or by finite difference. Verified at $M^*=60$,
$\mu^*=450$ MeV, $T=20$ MeV: $n_{(u,r)}=3.056\times10^6$ MeV³ at $\Delta=0$ (equal to the
unpaired formula, as it must be) and $3.517\times10^6$ at $\Delta=80$ MeV — a **15%
increase**. Substituting the unpaired formula for a paired mode is a 15% density error at a
realistic gap, and it breaks neutrality and the Euler audit simultaneously.

Then aggregate as usual:

$$
n_f = \sum_{a}n_{(f,a)},\qquad
n_B = \tfrac13\sum_f n_f,\qquad
n_C = \sum_f q_f n_f,\qquad
n_S = n_s
$$

and $\rho_{s,f}=\sum_a \rho_{s,(f,a)}$ feeds the field equations $R_1$–$R_3$ of §4.2
unchanged — those equations do not know about pairing except through $\rho_{s,f}$.

### 5.4 The full grand potential and the residuals

$$
\Omega = U(\varphi) + V(\sigma,\zeta) - \tfrac12 m_\omega^2\omega_0^2
 \;+\;\Omega_{\rm quark}
 \;-\;\sum_\ell\mathcal P_\ell - \mathcal P_\gamma
$$

Unknowns $\vec X = (\bar\varphi,\sigma,\zeta,\omega_0,\Delta,\mu_3,\mu_8)$. Residuals
$R_1$–$R_4$ exactly as in §4.2, plus:

$$
R_\Delta = \frac{\partial\Omega}{\partial\Delta}
 = \frac{\Delta}{2G_D} + \frac{\partial\,\delta\Omega_{\rm pair}}{\partial\Delta} = 0
$$

**The gap derivative is analytic — do not finite-difference $\Omega$.** For each branch,

$$
\frac{\partial}{\partial\Delta}
 \left[E_\Delta + 2T\ln\bigl(1+e^{-E_\Delta/T}\bigr)\right]
 = \frac{\Delta}{E_\Delta}\tanh\!\frac{E_\Delta}{2T}
$$

exactly (verified to $4\times10^{-17}$ over $E,\Delta,T$), so

$$
R_\Delta = \frac{\Delta}{2G_D}
 - \frac{1}{\pi^2}\int_0^\Lambda\!\!dk\,k^2 \sum_{r,e}
   \frac{\Delta}{E^{e,r}_\Delta}\,\tanh\!\frac{E^{e,r}_\Delta}{2T}\;=\;0
$$

At $T\to0$ the $\tanh\to1$ and this becomes the standard BCS kernel
$\int dk\,k^2\Delta/E_\Delta$. Note that dividing by $\Delta$ gives the familiar form whose
trivial root $\Delta=0$ has been divided out — keep the undivided form so both roots remain
visible.

$$
R_{n_3} = \sum_f\bigl(n_{(f,r)}-n_{(f,g)}\bigr) = 0,
\qquad
R_{n_8} = \sum_f\bigl(n_{(f,r)}+n_{(f,g)}-2n_{(f,b)}\bigr) = 0
$$

$\Delta=0$ is always a root of $R_\Delta$. **Test both roots and compare by $\Omega$** —
the paired solution is only realized when it lowers the grand potential, and near onset
there is also a barrier-maximum root that must be discarded.

**Where the coupling goes.** $\Delta$ sits in the dispersion; $G_D$ appears *only* in the
$\Delta^2/4G_D$ cost. Putting the coupling inside the dispersion (a gap matrix
$\hat\Delta=G_D\Delta[\ldots]$) while keeping the $\Delta^2/4G_D$ cost double-counts it:
the gap equation becomes $\Delta = 2G_D^2 I'(G_D\Delta)$, so $\Delta\sim G_D^2$ instead of
the correct $\Delta\sim G_D$ at weak coupling.

**Colour neutrality is not decorative.** In the *unpaired* phase $n_8$ vanishes identically
for any $\mu_8$ (verified: $n_8(\mu_8{=}0)=0$ exactly, $n_8(30\,\mathrm{MeV})\neq0$), so
$\mu_8$ is unconstrained there and the paired phase's $\mu_8$ **cannot be inherited** from
an unpaired solution — it must be solved within the pattern. It is that term which roughly
doubles the naive mismatch and reproduces the standard gapless criterion.

### 5.5 Outputs and audits

$$
P = -\Omega,\qquad
s = -\frac{\partial\Omega}{\partial T},\qquad
\varepsilon = -P + Ts + \sum_i\mu_i n_i
$$

**Paired-mode entropy is not the §3 formula either — and the error is far larger than for
the densities.** Gapped quasiparticles freeze out, so the entropy must be built from the
quasiparticle occupations $n^{e,r}_k=[1+e^{E^{e,r}_\Delta/T}]^{-1}$,

$$
s_{\rm paired} = -\frac{1}{\pi^2}\int_0^\Lambda\!\!dk\,k^2\sum_{r,e}
 \Bigl[n^{e,r}_k\ln n^{e,r}_k + \bigl(1-n^{e,r}_k\bigr)\ln\bigl(1-n^{e,r}_k\bigr)\Bigr]
$$

or equivalently from $-\partial\Omega/\partial T$. Measured at $M^*=60$, $\mu^*=450$,
$\Delta=60$ MeV: the paired entropy is $2.3\times10^{-4}$ of the unpaired value at $T=5$
MeV and $0.80$ of it at $T=50$ MeV. Substituting the unpaired formula is a
**four-orders-of-magnitude error in $s$ at low $T$**, and it propagates straight into
$\varepsilon$.

**$T=0$ must be branched, not approached.** Every thermal term above divides by $T$. Use
the $T=0$ forms below a switch temperature (§3.2 for the unpaired modes; for the paired
ones drop the $\ln$ terms and set $s=0$, since $\tanh(E/2T)\to1$ and
$n^{e,r}_k\to\theta(-E^{e,r}_\Delta)$). Compute the thermal logarithm as
`2*T*logaddexp(0, -|E|/T)`, never as `2*T*log(1+exp(-|E|/T))`, which overflows for
$|E|/T\gtrsim700$.

Assert at every solved point:

1. $\Delta\to0$ returns §4 **mode by mode** — verified for all nine modes.
2. The condensation energy scales as $\Delta^2\mu^2$ with the BCS logarithm: in the clean
   limit ($M^*\to0$, $T\to0$, $\mu=450$, $\Lambda=600$ MeV),
   $-\delta\Omega_{\rm pair}\big/\bigl[\mu^2\Delta^2(\ln\tfrac{2\Lambda}{\Delta}-\tfrac12)\bigr]
   \to 2/\pi^2$ — measured $1.92, 1.91, 1.89, 1.87$ times $1/\pi^2$ at
   $\Delta=2,5,10,20$ MeV, converging to $2$ as $\Delta\to0$. A wrong mode count or
   prefactor shows up here immediately.
3. $\delta\Omega_{\rm pair}(\Delta{=}0)=0$ exactly — not approximately.
4. The Euler relation, with $n_i$ from the derivatives of the full $\Omega$.

## 6. Phase C — the general pairing problem (L3)

### 6.1 The gap matrix, and the patterns derived from it

Three gaps $\Delta_1,\Delta_2,\Delta_3$ for $(ds),(us),(ud)$. Build the $9\times9$ matrix on
the colour–flavour modes $j=(a,i)$ directly from the antisymmetric tensors,

$$
\mathcal{G}_{(a i),(b j)} \;=\; \sum_{\eta=1}^{3}\Delta_\eta\,
  \epsilon^{ab\eta}\,\epsilon_{ij\eta}
$$

It is symmetric. Diagonalizing it gives the multiplicities — **derived, never assigned**:

| pattern | gaps | eigenvalues of $\mathcal{G}$ | gapped / ungapped |
|---|---|---|---|
| unpaired | all zero | $0$ (×9) | 0 / 9 |
| 2SC | $\Delta_3$ only | $\pm\Delta$ (×2 each), $0$ (×5) | 4 / 5 |
| uSC | $\Delta_2,\Delta_3$ | $\pm\Delta$ (×2 each), $\pm\sqrt2\Delta$ (×1 each), $0$ (×3) | 6 / 3 |
| dSC | $\Delta_1,\Delta_3$ | same as uSC | 6 / 3 |
| CFL | all equal | $\Delta$ (×3), $-\Delta$ (×5), $2\Delta$ (×1) | 9 / 0 |

The CFL row is the octet-plus-singlet spectrum: eight modes at $|\Delta|$ and one at
$2\Delta$. With **independent** gaps the uSC $\sqrt2\Delta$ becomes
$\sqrt{\Delta_2^2+\Delta_3^2}$, and dSC likewise with $\Delta_1,\Delta_3$ — so the $\sqrt2$
is an artifact of setting the two gaps equal, not a structural constant.

**Is $\Delta$ the same for every $(f,a)$? No — and in two distinct senses.**

*Across modes, within a pattern: no.* Even with a single nonzero $\Delta_\eta$, the
quasiparticle gaps are the eigenvalues of $\mathcal{G}$, and those differ mode by mode. In
2SC four modes are gapped and five are not. In CFL the eight octet modes sit at $|\Delta|$
but the singlet sits at $2\Delta$ — a factor two, from the $\epsilon$ contraction alone.
In uSC/dSC four modes sit at $\Delta$ and two at $\sqrt{\Delta_2^2+\Delta_3^2}$. So "the
gap" is never one number across the nine modes; it is one number *per channel* $\eta$,
producing a spectrum.

*Across channels: only at the symmetric point.* Setting $\Delta_1=\Delta_2=\Delta_3$ is what
*defines* the CFL ansatz, but whether the solution respects it is a question for the solver.
Minimizing $\Omega$ over all three gaps freely, at $\mu^*=450$ MeV, $T=20$ MeV,
$G_D=0.75G_S$:

| $M^*_u,M^*_d$ | $M^*_s$ | $\Delta_1$ $(ds)$ | $\Delta_2$ $(us)$ | $\Delta_3$ $(ud)$ |
|---|---|---|---|---|
| 60 | 60 | 70.7 | 70.7 | 70.7 |
| 60 | 150 | 69.4 | 69.4 | 71.3 |
| 60 | 200 | 66.8 | 66.8 | 72.7 |
| 60 | 300 | **0.0** | **0.0** | 89.2 |

At exactly equal masses the three gaps come out equal to better than 1% — an **output**, not
an input, and there the single CFL $\Delta$ is legitimate. The equality then degrades with
$M^*_s$ and **collapses** by $M^*_s\simeq300$ MeV: the two strange channels switch off
entirely and the free solve lands on 2SC. Note also that $\Delta_1=\Delta_2$ throughout,
which is the residual $u\leftrightarrow d$ symmetry at $M^*_u=M^*_d$ — break that and those
two separate too.

Two consequences for a code. First, **do not hardwire $\Delta_1=\Delta_2=\Delta_3$**: solve
the three gap equations and let the pattern emerge, or you will report CFL in a region where
the strange channels are dead. Second, the collapse at $M^*_s\simeq300$ MeV is abrupt rather
than gradual, so a continuation in $M^*_s$ needs seeding from both sides.

### 6.2 Why the eigenvalues alone are not enough

It is tempting to diagonalize $\mathcal{G}$, take each eigenvalue $\Delta^{(j)}$, and write
$E^{(j)}=\sqrt{(\xi^{(j)})^2+(\Delta^{(j)})^2}$. **That is wrong as soon as the quark masses
differ**, for a reason that is easy to state and easy to miss:

$$
\bigl[\,\mathcal{G},\ \mathrm{diag}(M^*_f)\,\bigr] \neq 0
\qquad\text{whenever } M^*_u,M^*_d,M^*_s \text{ are not all equal}
$$

(the commutator vanishes identically at equal masses). The gap matrix and the mass matrix
share no eigenbasis, so there is no basis in which both the kinetic and the pairing parts
are simultaneously diagonal, and the problem does not factorize into independent scalar
dispersions. Quantified at $M^*=(60,65,300)$, $\mu^*=450$ MeV, $\Delta=60$ MeV: the
eigenvalue prescription misses branches by up to **33.6 MeV in uSC and 27.1 MeV in CFL** —
comparable to $\Delta$ itself, and it puts the lowest CFL branch at $60.1$ MeV where the
correct value is $33.0$ MeV, which is the difference between a gapped and a nearly gapless
phase.

A second obstruction blocks the §5.2 two-body construction as a general substitute: the
$\sqrt2$ eigenvector of uSC mixes **three** modes ($ru$, $gd$, $bs$), not two. There is no
assignment of uSC/dSC modes into pairs, so no per-pair formula can exist for them.

### 6.3 The Bogoliubov–de Gennes problem — what to actually code

At each momentum $k$, assemble and diagonalize

$$
\boxed{\;
H_{\rm BdG}(k) = \begin{pmatrix} \xi(k) & \mathcal{G} \\[2pt] \mathcal{G} & -\xi(k)\end{pmatrix},
\qquad
\xi(k) = \mathrm{diag}\bigl(E_{k,f(j)} - \mu^*_j\bigr)_{j=1}^{9}\;}
$$

an $18\times18$ real symmetric matrix whose spectrum is particle–hole symmetric. The nine
quasiparticle energies $E^{(j)}_\Delta(k)$ are the **non-negative half of the signed
spectrum** — take `sort(eigvalsh(H))[9:]`, not the nine largest absolute values (that
mistake duplicates branches and silently drops the low ones).

Antiparticles: repeat with $\mu^*_j\to-\mu^*_j$, giving nine more branches per $k$. Two
$18\times18$ symmetric eigendecompositions per momentum node — negligible cost against the
quadrature itself.

Validated four ways: at $\Delta_\eta=0$ it returns $|E_{k,f}-\mu^*_j|$ for all nine modes
exactly; at equal masses it reproduces the CFL octet-plus-singlet and the uSC
$3+4(\Delta)+2(\sqrt2\Delta)$ structure to $10^{-8}$; the $18$ eigenvalues come in $\pm$
pairs to $10^{-13}$; and $\sum_i w_i^2 = 2\,\mathrm{tr}\,\xi^2 + 2\,\mathrm{tr}\,\mathcal{G}^2$
to $6\times10^{-16}$ — a cheap runtime assertion worth keeping.

The grand potential is then §5.1's correction form with the mode sum running over the BdG
branches:

$$
\boxed{\;
\Omega_{\rm quark} = -\!\!\sum_{j=1}^{9}\!\mathcal P\bigl(M^*_j,\mu^*_j,T\bigr)
 \;+\;\delta\Omega_{\rm pair}\;+\;\sum_{\eta=1}^{3}\frac{\Delta_\eta^2}{4G_D}\;}
$$

$$
\delta\Omega_{\rm pair} = -\frac{1}{2\pi^2}\int_0^\Lambda\!\!dk\,k^2
 \sum_{r=\pm}\sum_{j=1}^{9}\Bigl\{\bigl(E^{(j),r}_\Delta - \bigl|\xi^{(j),r}\bigr|\bigr)
   + 2T\Bigl[\ln\bigl(1+e^{-E^{(j),r}_\Delta/T}\bigr)
            - \ln\bigl(1+e^{-|\xi^{(j),r}|/T}\bigr)\Bigr]\Bigr\}
$$

with $\xi^{(j),r}$ the same expression at $\Delta_\eta=0$ — i.e. the sorted
$|E_{k,f}-r\mu^*_j|$. Every brace vanishes identically at $\Delta_\eta=0$, so
$\delta\Omega_{\rm pair}=0$ there and the whole expression reduces to §4 mode by mode; the
$\Lambda$-independence argument of §5.1 carries over unchanged.

### 6.4 The three gap equations, written out

**Yes — with unequal gaps all three must be solved simultaneously.** There is one residual
per channel, not one gap. Write the gap matrix as a linear combination,

$$
\mathcal{G}(\Delta_1,\Delta_2,\Delta_3) = \sum_{\eta=1}^{3}\Delta_\eta\,\mathcal{B}_\eta,
\qquad
\bigl(\mathcal{B}_\eta\bigr)_{(ai),(bj)} = \epsilon^{ab\eta}\epsilon_{ij\eta}
$$

(exact, since $\mathcal{G}$ is linear in the gaps). Each $\mathcal{B}_\eta$ has exactly four
nonzero entries and couples exactly two mode pairs:

| $\eta$ | channel | $\mathcal{B}_\eta$ couples |
|---|---|---|
| 1 | $(ds)$ | $gd$–$bs$, $gs$–$bd$ |
| 2 | $(us)$ | $ru$–$bs$, $rs$–$bu$ |
| 3 | $(ud)$ | $ru$–$gd$, $rd$–$gu$ |

Then the three residuals are, with $\mathcal{V}^{(j),r}$ the eigenvector of
$H_{\rm BdG}$ belonging to $E^{(j),r}_\Delta$,

$$
\boxed{\;
R_{\Delta_\eta} \;=\; \frac{\Delta_\eta}{2G_D}
 \;-\;\frac{1}{2\pi^2}\int_0^\Lambda\!\!dk\,k^2
 \sum_{r=\pm}\sum_{j=1}^{9}
 \Bigl\langle \mathcal{V}^{(j),r}\Bigl|\,
   \begin{pmatrix} 0 & \mathcal{B}_\eta \\ \mathcal{B}_\eta & 0\end{pmatrix}
 \Bigr|\mathcal{V}^{(j),r}\Bigr\rangle\,
 \tanh\frac{E^{(j),r}_\Delta}{2T}
 \;=\;0
\qquad \eta=1,2,3\;}
$$

The matrix element is the Hellmann–Feynman derivative
$\partial E^{(j),r}_\Delta/\partial\Delta_\eta$, and $\partial H_{\rm BdG}/\partial\Delta_\eta$
is just $\mathcal{B}_\eta$ in the off-diagonal blocks — a **constant** matrix, assembled once
at startup, not per iteration. The $\tanh$ carries the thermal factor exactly as in §5.4, and
$\to1$ at $T\to0$.

Verified against finite differences of $\Omega$ at $M^*=(60,65,300)$, $\mu^*=450$ MeV,
$T=20$ MeV, $(\Delta_1,\Delta_2,\Delta_3)=(25,40,70)$ MeV: agreement to
$6\times10^{-7}$ relative in every channel. So the gap equations are analytic — no
differencing of $\Omega$ anywhere.

**The unknown vector at L3 is therefore**

$$
\vec X = \bigl(\bar\varphi,\ \sigma,\ \zeta,\ \omega_0,\
 \Delta_1,\ \Delta_2,\ \Delta_3,\ \mu_3,\ \mu_8\bigr)
$$

nine unknowns against nine residuals: $R_1$–$R_4$ from §4.2, $R_{\Delta_1}$–$R_{\Delta_3}$
above, and $R_{n_3}$, $R_{n_8}$ from colour neutrality — plus whatever the closure row of §7
adds on top. **Do not reduce this to a single $\Delta$**: the §6.1 table shows the three gaps
splitting and then two of them vanishing as $M^*_s$ grows, so a one-gap solve reports CFL in
a region where the strange channels are dead.

Practical notes. $\Delta_\eta=0$ is always a root of $R_{\Delta_\eta}$, so seed both zero and
$\sim50$ MeV in each channel and compare by $\Omega$; the named patterns of §6.1 are just
particular roots of this system, distinguished by which gaps come out zero. Request
eigenvectors from the diagonalization, not only eigenvalues. And the Hellmann–Feynman form
is valid branch by branch only when the branch is non-degenerate — at an exact degeneracy use
the subspace-projected derivative, or perturb the degeneracy numerically by $10^{-6}$ MeV,
which is well inside the tolerances of §9.4.

### 6.5 Pattern selection is an output

Solve each candidate — unpaired, 2SC, uSC, dSC, CFL — to self-consistency under the same
neutrality conditions, then take the lowest $\Omega$. In practice the general solve with all
three $\Delta_\eta$ free subsumes the list: each named pattern is a stationary point of the
same functional, distinguished by which gaps vanish. Enumerating them explicitly is still
worth doing, because the free solve finds one root and the pattern seeds find the others.

Do not include gapless states in the minimization — they are not stationary points of the
same functional. Instead flag *enumeration-invalid* when a converged pattern turns out
gapless ($\min_k E^{(j)}_\Delta<0$ for some branch) and report it, rather than silently
comparing incomparable states.

At $T=0$ the gap-equation root alone does not reveal the transition — it is independent of
the mismatch. Only $\Omega$ crossing zero does. Locate transitions by root-finding on
$\Omega$ differences, never by an argmin over a scan grid.

**Cutoff.** $\Lambda\simeq600$ MeV, applied to the pairing integral only. The unpaired
Fermi-gas integrals terminate at their own $k_F$, so they are untouched while $k_F<\Lambda$
— that is why a sharp cutoff is preferred. It does imply a validity ceiling
$\mu_{\max}=\sqrt{\Lambda^2+M^{*2}}$ for the pairing sector; declare it rather than exceed
it. $\Lambda$ and $G_D$ are nearly interchangeable: over the standard range
$G_D/G_S=0.5$–$1.0$ the gap moves by a factor $\sim3$, so the uncertainty lives in $G_D$.

## 7. Closure conditions

$\Omega$ is the same function in every row. Only the constraints differ.

| Row | Fixed | Solve for | Conditions |
|---|---|---|---|
| R1 cold star | $\mu_B$, $T=0$ | $\mu_C$ | $n_C^{\rm tot}=0$; weak eq., $\mu_L=0$; $\mu_S=0$ |
| R2 merger/CCSN | $n_B$, $Y_e$, $T$ | $\mu_B,\mu_C$ | $n_e=Y_e n_B$ ($\mu_e$ independent); $n_C=n_e$; **no** weak eq.; $\mu_S=0$; no $\mu$, no $\nu$ |
| R3 proto-NS | $\mu_B$, $Y_{L_e}$, $T$ | $\mu_C,\mu_{L_e},\mu_{L_\mu}$ | $n_C^{\rm tot}=0$; $n_{L_e}=Y_{L_e}n_B$; $Y_{L_\mu}=0$; $\mu_S=0$ |
| R4 heavy-ion | $n_B$, $Y_C$, $Y_S$, $T$ | $\mu_C,\mu_S$ | $n_C/n_B=Y_C$; $Y_S=0$ (i.e. $n_S=0$); no leptons |
| R5 symmetric | $n_B$, $T$ | $\mu_C,\mu_S$ | $Y_C=\tfrac12$; $Y_S=0$; no leptons |
| R6 pQCD corner | $\mu_B\gtrsim2.4$ GeV | — | equal $\mu_f$; $c_s^2\to\tfrac13$ (matching condition, not solved) |

Total electric neutrality is $n_C^{\rm tot} \equiv n_C - n_e - n_\mu = 0$, where $n_C$ is
the **quark-only** charge density. Fractions are per baryon: $Y_C=n_C/n_B$,
$Y_S=n_S/n_B$; these differ by a factor 3 from the per-quark $n_s/(3n_B)$ often plotted.

Weak equilibrium, when a row imposes it, is $\mu_S=0$ (nonleptonic $s\leftrightarrow d$)
and $\mu_e=-\mu_C$ (from $d\to u+e+\bar\nu$). **These are per-row closures, never
identities** — R2 in particular is locally neutral and *not* weak-equilibrated, and
hardwiring weak equilibrium into $\Omega$ makes such matter unrepresentable.

---

## 8. Parameters

**Fixed by vacuum data — not free.** $f_\pi=93$, $m_\pi=138$, $f_K=113$, $m_K=496$,
$m_{u,d}=5$, $m_s=95$, $m_\zeta=980$, $m_\varphi=1600$ (lattice scalar glueball),
$m_\omega=783$ MeV. Derived once at startup:

$$
\sigma_0=f_\pi,\qquad
\zeta_0=\sqrt2 f_K-\frac{f_\pi}{\sqrt2},\qquad
\epsilon_\sigma=f_\pi m_\pi^2,\qquad
\epsilon_\zeta=\sqrt2 f_K m_K^2-\frac{f_\pi}{\sqrt2}m_\pi^2,\qquad
\varphi_0=\frac{4\sqrt{B_g}}{m_\varphi}
$$

$$
\lambda=\frac{m_\sigma^2-m_\pi^2}{2f_\pi^2},\qquad
v^2=f_\pi^2-\frac{m_\pi^2}{\lambda},\qquad
\lambda_\zeta=\frac{m_\zeta^2-\epsilon_\zeta/\zeta_0}{2\zeta_0^2},\qquad
v_\zeta^2=\zeta_0^2-\frac{\epsilon_\zeta}{\lambda_\zeta\zeta_0}
$$

and $C_0$ from $V(\sigma_0,\zeta_0)=0$. Numerically, at $m_\sigma=550$, $m_\zeta=980$ MeV:
$\zeta_0=94.05$ MeV, $\lambda=16.39$, $v=86.53$ MeV, $\lambda_\zeta=31.41$,
$v_\zeta^2=-4039$ MeV², $C_0=2.435\times10^{9}$ MeV⁴.

$v$ and $v_\zeta$ are **constants, not fields** — the Mexican-hat radii before explicit
breaking, and not observables. The linear terms shift the true minima to $\sigma_0,\zeta_0$.

**$v_\zeta^2$ is negative** at the baseline $m_\zeta$: the strange quartic is convex,
explicit breaking dominating, so the strange sector does not break chirally on its own in
this truncation. The sign flips between $m_\zeta=1100$ and $1150$ MeV. **Never assume
$v_\zeta^2>0$.**

**Free and dense-matter-relevant.** $B_g$, $g_q$, $g_s$, $m_\sigma$ (450–700), $\bar
g_\omega$, $n_c$, and at L3 $G_D$, $\Lambda$. $m_\omega$ is a normalization convention
(only $g_\omega/m_\omega$ enters); $m_\varphi$ cancels from the bulk EoS at fixed $B_g$,
pricing gradients and the glueball spectrum instead.

### 8.1 The Bayesian parameter vector, explicitly

The sampler sees exactly this vector — everything else is either derived at
startup (§8), a discrete structural choice declared per run, or an internal
unknown solved at each $(\mu_B,T)$ point ($\bar\varphi,\sigma,\zeta,\omega_0,
\Delta_\eta,\mu_3,\mu_8,\mu_C,\dots$ are *never* sampled):

| # | parameter | units | role | suggested prior support |
|---|---|---|---|---|
| 1 | $B_g^{1/4}$ | MeV | glue bag scale; sets $\varphi_0$ and the deconfinement onset | 120–250 |
| 2 | $g_q$ | — | light-quark coupling; $M^*_{u,d}$ in the confined branch | 3–6 |
| 3 | $g_s$ | — | strange coupling; $M^*_s$ | 3–8 |
| 4 | $m_\sigma$ | MeV | light scalar mass; fixes $\lambda$, $v$ | 450–700 |
| 5 | $\bar g_\omega$ | — | vector repulsion at $n_B\to0$ | 0–12 |
| 6 | $n_c$ | fm⁻³ | vector-coupling decay scale | 0.3–3 |
| 7 | $G_D$ | MeV⁻² | diquark coupling (L3 only) | such that $\Delta\sim$ 20–150 MeV at $\mu_q\simeq450$ MeV |
| 8 | $\Lambda$ | MeV | pairing-integral cutoff (L3 only) | 550–800; nearly degenerate with $G_D$ (§6.5) |

Discrete choices (declared, not sampled): $p=1$ (dielectric exponent, locked);
$q\in\{0,1\}$ (dielectric dressing of $G_D$, §10); the closure row (§7); the
pairing-pattern list (§6). If both $G_D$ and $\Lambda$ are sampled the posterior
will show the §6.5 degeneracy — consider fixing $\Lambda=600$ MeV and sampling
$G_D$ alone unless the data can break it.

Fixed by vacuum data (tier above the sampler): $f_\pi, m_\pi, f_K, m_K, m_{u,d},
m_s, m_\zeta, m_\varphi, m_\omega$ and everything derived from them in §8.

---

## 9. Assembly order, solver recipe, and what to assert

### 9.1 Startup (once)

Derive the §8 constants. Assert $V(\sigma_0,\zeta_0)=0$, $U(\varphi_0)=0$, and that the
curvatures reproduce $m_\sigma$ and $m_\zeta$ to $<0.5$ MeV. Do **not** assume
$v_\zeta^2>0$.

### 9.2 The nested structure

Three levels, outermost first. Flattening them into one big root-find is possible but
fragile; the nesting below is what converges.

1. **Closure loop** (outermost): solve the row's constraints of §7 for
   $(\mu_C,\mu_S,\dots)$ — a 1–3 dimensional root-find.
2. **Field loop**: for given chemical potentials, solve $R_1$–$R_4$ (+ $R_\Delta$,
   $R_{n_3}$, $R_{n_8}$ at L3) for $(\bar\varphi,\sigma,\zeta,\omega_0,\Delta,\mu_3,\mu_8)$
   — **per branch**, see §9.3.
3. **Vector fixed point** (innermost): $\omega_0$ depends on $n_B$, which depends on
   $\omega_0$ through $\mu^*$. Iterate $\omega_0^{(k+1)} = g_\omega(n_B^{(k)})n_q^{(k)}/m_\omega^2$
   to a relative tolerance of $10^{-10}$; it converges monotonically because $g_\omega$
   decreases with $n_B$. Never minimize in $\omega_0$ — §4.2.

### 9.3 Branch enumeration, not iteration

At each $(\mu,T)$ point solve **every** branch to self-consistency and compare by $\Omega$:

| branch | seed |
|---|---|
| confined | $\bar\varphi=1-10^{-6}$, $\sigma=f_\pi$, $\zeta=\zeta_0$ |
| deconfined, chirally restored | $\bar\varphi=0.4$, $\sigma=0$, $\zeta=0$ |
| deconfined, partially restored | $\bar\varphi=0.4$, $\sigma=0.5f_\pi$, $\zeta=0.7\zeta_0$ |

plus, at L3, each pairing pattern of §6 with $\Delta$ seeded at $0$ and at $\sim50$ MeV.
A solver that alternates between updating $\sigma$ and $\omega_0$ will two-cycle between the
first two branches below the onset and exit with a **mixed** state ($\sigma$ from one,
$\omega_0$ from the other), which reads as a spuriously deep minimum at zero quark density.
Assert that the returned $\omega_0$ is the one the returned densities source.

A branch that fails to converge must be **reported as missing, never replaced** by a
neighbouring point — silently substituting a converged neighbour is how a fake phase
boundary appears.

### 9.4 Numerical defaults

| quantity | value | note |
|---|---|---|
| quadrature nodes | 400 per panel, 3 panels | §3.1; 100 already suffices |
| $T$ switch to $T=0$ forms | $T<1$ MeV | below this the closed forms are more accurate than any grid |
| $\bar\varphi$ guard | $[0,\,1-10^{-9}]$ | $\chi\to0$ at the endpoint; $M^*$ overflows past it |
| solve variable for the glue | $\Phi=\bar\varphi^{\,4}$ | §1: removes the spurious $R_1$ root at $\bar\varphi=0$; recover $\bar\varphi=\Phi^{1/4}$ |
| field-equation tolerance | $|R_i|/{\rm scale}<10^{-8}$ | scale each residual by its own natural size, e.g. $R_1$ by $B_g/\varphi_0$ |
| vector fixed point | $10^{-10}$ relative | innermost, cheap |
| gap tolerance | $|\Delta^{(k+1)}-\Delta^{(k)}|<10^{-6}$ MeV | plus $|R_\Delta|$ scaled by $\Delta/2G_D$ |
| $\Lambda$ | 600 MeV | pairing integrals only |
| finite-difference step for $n_i$ | $h=0.02$–$0.05$ MeV | central differences; the audits below detect a bad $h$ |

Work in MeV throughout and convert only at output. Densities in MeV³ are $O(10^6)$ and
pressures $O(10^8)$ — well inside double precision, so there is no need to rescale.

### 9.5 Order of evaluation inside one field-equation residual

1. Guard $\bar\varphi$; compute $\chi$, $\chi'/\chi$.
2. Compute $M^*_f$ from the current $\sigma,\zeta$.
3. Inner fixed point for $\omega_0$; form $\mu^*_{f,a}$ (§2.3).
4. **Test $M^*_f$ against $\mu^*_{f,a}$ at $T=0$: if $M^*\ge\mu^*$ that mode's integrals
   are identically zero — return zero, do not integrate.** This is not a numerical nuisance
   to be smoothed away; it is the confinement mechanism, and smoothing it destroys the
   pinning that makes the deconfinement transition first order.
5. Base integrals per mode with the split-panel quadrature (§3.1).
6. At L3: build the $9\times9$ gap matrix, diagonalize, form the pair dispersions (§6),
   evaluate $\delta\Omega_{\rm pair}$ and its analytic $\Delta$-derivative.
7. Assemble $\Omega$ and the residuals.

### 9.6 Assertions — run these, not just once

At every solved point:

$$
P = -\Omega \quad\text{from both assemblies independently};\qquad
\varepsilon = -P + Ts + \sum_i\mu_i n_i
$$

$$
n_i \stackrel{!}{=} -\frac{\partial\Omega}{\partial\mu_i}\Big|_{\rm finite\ diff}
\quad\text{against the summed mode densities}
$$

At L3 additionally: $\delta\Omega_{\rm pair}(\Delta{=}0)=0$ **exactly**; $n_3=n_8=0$ to the
same tolerance as the other residuals; and the pattern's gapless flag.

Reduction chain, as one-off tests:

$$
G_D\to0 \Rightarrow \text{L2},\qquad
\bar g_\omega\to0 \Rightarrow \text{L1}\to\text{L0},\qquad
\Delta\to0 \Rightarrow \text{§4 mode by mode}
$$

These catch sign and bookkeeping errors nothing else will. Finally, $c_s^2=dP/d\varepsilon$
**along a branch only** — never finite-difference across a first-order transition; take
one-sided derivatives on each side and leave the transition as a gap.

## 10. On the de Carvalho pairing mechanism

Their construction is real and worth stating precisely, but it does not substitute for
$G_D$ in this application. What they do: write the confining field as
$\chi=\bar\chi+\delta\chi$, expand, and integrate out $\delta\chi$ at Gaussian order —
exact, since $\delta\chi$ couples linearly to $\bar\Psi\Psi$ and appears quadratically in
its own mass term. The result is a contact interaction $h(\bar\Psi\Psi)^2$ with

$$
h = \frac12\left[\frac{G'(\bar\chi)}{M_\chi}\right]^2,
\qquad
M_\chi^2 = m_\chi^2 - G''(\bar\chi)\langle\bar\Psi\Psi\rangle,
\qquad
G(\chi) = -\frac{g f_\pi}{\chi}
$$

Reproduced at their parameters ($g=0.023$ GeV, $m_\chi=1.7$ GeV): $h = 6.6$, $2.6$, $1.0$
GeV⁻² at $\rho=0.5,1,2$ fm⁻³, inside their Fig. 2 range, with $M_\chi^2\simeq3m_\chi^2$ as
they state.

Translated into this model's variables — replacing their vertex derivative by
$\partial M^*/\partial\varphi$ and using $m_\varphi^2=16B_g/\varphi_0^2$ — the coupling is

$$
h(\varphi) = \frac{M^{*2}\,p^2\,\bar\varphi^{\,6}}{2B_g\bigl(1-\bar\varphi^{\,4}\bigr)^2}
$$

and **$\varphi_0$ cancels identically** (verified symbolically), so it costs no new
parameter. That is the appealing part. The problem is what it predicts:

| branch | $\bar\varphi$ | $M^*_{u,d}$ | $h/G_D^{\rm Fierz}$ |
|---|---|---|---|
| deconfined ($\sigma=0$) | 0.57 | 5.6 MeV | $3.5\times10^{-4}$ |
| deconfined ($\sigma=0$) | 0.77 | 7.7 MeV | $7.7\times10^{-3}$ |
| confined ($\sigma=f_\pi$) | 0.90 | 826 MeV | $8.0\times10^{2}$ |
| confined ($\sigma=f_\pi$) | 0.95 | 1531 MeV | $1.3\times10^{4}$ |

The coupling is negligible where the quarks are light and enormous where they are heavy —
which is exactly what de Carvalho report ("no gap in the chiral restored phase; in phase I,
where $\chi$ is small and the quarks are massive, the quarks do pair"), and it is the
wrong way round for a compact-star EoS: it removes pairing from the deconfined phase,
where a star's core lives, and would condense a diquark in the confining vacuum. Their own
paper flags the same region as outside its validity — the gap blows up at low density,
where the pairing energy becomes comparable to the total.

**What is taken from it.** Eq. (10) is why $\mathcal L_{\rm pair}$ is a legitimate leading
term rather than an *ad hoc* addition, and it gives "zero range" a physical meaning: the
range is $1/M_\chi$, an inverse glue mass the model already carries. **What is not taken:**
$h$ as a replacement for $G_D$. The defensible middle course is the mildest dielectric
dressing, $G_D\to G_D/\chi^{q}$ with $q=1$ — the exponent a gluon-exchange origin gives,
and the largest that leaves the pairing channel confined (the criterion is $q\le p$,
because the critical coupling for vacuum diquark condensation grows only linearly in
$M^*\propto\chi^{-p}$). Their $h$ carries $q=4$ at $p=1$, which violates it. Carry
$q\in\{0,1\}$ as a declared discrete choice at calibration.

---

## 11. References (arXiv)

Colour-dielectric / dilaton foundations (mostly pre-arXiv; DOIs from the project
bibliography `refs_csc.bib`):

- Friedberg, Lee — *Fermion-field nontopological solitons I, II*, PRD 15 (1977) 1694; PRD 16 (1977) 1096; *QCD and the soliton model of hadrons*, PRD 18 (1978) 2623.
- Nielsen, Patkós — *Effective dielectric theory from QCD*, Nucl. Phys. B 195 (1982) 137.
- Mack — *Dielectric lattice gauge theory*, Nucl. Phys. B 235 (1984) 197.
- Schechter — *Effective Lagrangian with two color-singlet gluon fields*, PRD 21 (1980) 3393.
- Migdal, Shifman — *Dilaton effective Lagrangian in gluodynamics*, PLB 114 (1982) 445.
- Pirner — *The color dielectric model of QCD*, Prog. Part. Nucl. Phys. 29 (1992) 33.
- Birse — *Soliton models for nuclear physics*, Prog. Part. Nucl. Phys. 25 (1990) 1.
- Wilets — *Nontopological solitons* (World Scientific, 1989).

CCDM quark matter:

- Drago, Fiolhais, Tambini — *Quark matter in the chiral colour-dielectric model* — **arXiv:hep-ph/9503462**
- Ghosh, Phatak — *Study of quark matter in the chiral colour dielectric model*, J. Phys. G 18 (1992) 755; *Three-flavour quark matter*, PRC 52 (1995) 2195 — **arXiv:nucl-th/9509017**
- Alberico, Drago, Ratti — *Strangelet stability, MIT bag vs CDM* — **arXiv:hep-ph/0110091**
- Maieron, Baldo, Burgio, Schulze — *Hybrid stars, CDM vs MIT bag* — **arXiv:nucl-th/0404089**
- de Carvalho, Malheiro, et al. — *Color superconductivity and confinement in the chromodielectric model*, Nucl. Phys. B Proc. Suppl. 199 (2010) 308, doi:10.1016/j.nuclphysbps.2010.02.049 (no arXiv located; §10 works from the project copy `decarvalho.pdf`)

Pairing sector (shared with the NJL companion):

- Buballa — *NJL-model analysis of dense quark matter* — **arXiv:hep-ph/0402234**
- Alford, Schmitt, Rajagopal, Schäfer — *Color superconductivity in dense quark matter* — **arXiv:0709.4635**
- Steiner, Reddy, Prakash — *Color-neutral superconducting quark matter* — **arXiv:hep-ph/0205201**
- Rüster et al. — *Phase diagram of neutral quark matter* — **arXiv:hep-ph/0503184**
- Kunkel, Rather, et al. — *CSC phases in proto-neutron star evolution* — **arXiv:2607.11537**
- Pagliara, Schaffner-Bielich — *Stability of CFL cores in hybrid stars* — **arXiv:0711.1119**
- Lavagno, Pagliara — *Gapless CFL phase in quark and hybrid stars* — **arXiv:nucl-th/0504066**
- Logoteta, Bombaci, Providência, et al. — *Chiral model approach to quark matter nucleation* — **arXiv:1203.4159**

Context (confining density functionals, conformal limit):

- Ivanytskyi, Blaschke — *Density functional approach to quark matter with confinement and CSC* — **arXiv:2204.03611**; **arXiv:2209.02050**; **arXiv:2211.12730**
- Baym, Hatsuda, Kojo, Powell, Song, Takatsuka — *From hadrons to quarks in neutron stars* — **arXiv:1707.04966**

arXiv IDs were carried over from the project's literature-extraction files
(`refs_csc.bib`, `lit_conformal.json`), where the sources were actually fetched.
