# CCDM — chiral colour-dielectric quark matter, with colour superconductivity

A mean-field model of deconfined `u, d, s` matter in which confinement and
chiral symmetry breaking are two faces of one mechanism. A dilaton field
carries the gluon condensate; the dielectric function built from it measures
the medium's transparency to colour and *divides* the quark masses, so that as
the condensate reaches its vacuum value the effective masses diverge and the
quarks leave the medium entirely.

This file states every equation the code solves and every quantity it returns.
`ccdm.tex` is the same document typeset, compiled against `../../docs/eos.bib`;
neither defers to the other. The implementation specification is
`docs/ccdm_implementation.md` and is the authority wherever it and this
document differ, with the two exceptions under
[Two corrections to the specification](#two-corrections-to-the-specification),
where the specification contradicts the thermodynamic audit it itself mandates.

Lineage: the soliton and dielectric Lagrangians of Friedberg and Lee (PRD 15
(1977) 1694; PRD 18 (1978) 2623), Nielsen and Patkos (NPB 195 (1982) 137) and
Mack (NPB 235 (1984) 197), reviewed by Birse (PPNP 25 (1990) 1) and Pirner
(PPNP 29 (1992) 33), in the three-flavour chiral form of Drago, Fiolhais and
Tambini (NPA 588 (1995) 801, hep-ph/9503462) and Ghosh and Phatak (PRC 52
(1995) 2195, nucl-th/9509017); applied to strangelets by Alberico, Drago and
Ratti (NPA 706 (2002) 143) and to hybrid stars by Maieron et al. (PRD 70
(2004) 043010). The pairing sector follows Buballa (Phys. Rept. 407 (2005)
205), Alford et al. (RMP 80 (2008) 1455) and Ruester et al. (PRD 72 (2005)
034004), with the neutrality treatment of Steiner, Reddy and Prakash (PRD 66
(2002) 094007).


## Conventions

Natural units `hbar = c = k_B = 1` inside the physics; masses, chemical
potentials and the dilaton in MeV, densities in MeV^3, `Omega`, `P` and `eps`
in MeV^4. The public boundary is fm-based — `n` in fm^-3, `T` and every `mu` in
MeV, `eps` and `P` in MeV/fm^3, `s` in fm^-3 — with
`hc = 197.3269804 MeV fm` and `(hc)^3 = 7.6835057e6 MeV^3 fm^3` applied once,
at that boundary.

Flavour `f in {u,d,s}`, colour `a in {r,g,b}`, and the nine colour-flavour
modes `j = (f,a)` indexed flavour-major, `j = 3 i_f + i_a`. Charges
`q_u = +2/3`, `q_d = q_s = -1/3`. **Strangeness is S = +1 per s quark**, the
opposite of the PDG sign, used consistently throughout this repository. `C` is
the electric charge of strongly interacting matter only: the leptons are
excluded from it and enter through the separate condition of total electric
neutrality. Fractions are per baryon, `Y_C = n_C/n_B` and `Y_S = n_S/n_B`.

Colour generators:

    (T_3)_(r,g,b) = (+1/2, -1/2, 0)
    (T_8)_(r,g,b) = (+1/3, +1/3, -2/3)      i.e. T_8 = lambda_8/sqrt(3)

Three normalisations are in circulation in the colour-superconductivity
literature and mixing them corrupts `mu_8` by factors between 1.15 and 1.7;
the conversions are in `eos.general.pairing`.


## The Lagrangian

Four boson fields — the dilaton `phi`, the light scalar `sigma`, the pion
triplet `pi` and the strange scalar `zeta` — plus the vector meson `omega_mu`
and the diquark interaction:

    L = sum_f qbar_f [ i gamma^mu d_mu - M*_f ] q_f
        + (1/2)(d phi)^2   - U(phi)
        + (1/2)(d sigma)^2 + (1/2)(d pi)^2 + (1/2)(d zeta)^2
        - V(sigma, pi, zeta)
        + L_vec + L_pair

    L_vec  = -g_omega qbar gamma^mu q omega_mu
             - (1/4) F_munu F^munu + (1/2) m_omega^2 omega_mu omega^mu

    L_pair = G_D sum_(eta=1..3) | qbar i gamma_5 C eps^(ab eta) eps_(ij eta)
                                   qbar^T |^2

with `eta = 1,2,3` pairing `(ds), (us), (ud)`. `U` is the potential the scale
anomaly fixes (Schechter, PRD 21 (1980) 3393; Migdal and Shifman, PLB 114
(1982) 445); it is written out below.

### The dielectric, and why the fourth power

The dielectric function is **not** a field: no kinetic term, no potential, no
field equation. It is an abbreviation, and it appears only in the quark mass
operator:

    chi(phi) = [1 - phi_bar^4]^p ,   phi_bar = phi/phi_0 ,   p = 1

The fourth power is fixed. `phi_bar^4`, not `phi_bar`, is the gluon
condensate: the dilaton is a canonically normalised scalar of mass dimension 1
while `<G^A_munu G^A^munu>` has dimension 4. The scale anomaly makes it sharp —
the combination it fixes comes out a pure power only for exponent 4,

    4 U - phi dU/dphi = 4 B_g (1 - phi_bar^4)

whereas `phi_bar^2`, `^3`, `^5` or `^6` leave a residual `phi_bar^n ln phi_bar`.
So

    1 - phi_bar^4 = 1 - <G^2>_med / <G^2>_vac

and the medium becomes transparent to colour exactly in proportion to how much
condensate has melted. Both endpoints are then correct by construction:
`chi -> 0` at `phi_bar = 1` (confinement, `M* -> infinity`) and `chi -> 1` at
`phi_bar = 0` (perturbative, `M* -> m_f`). Only `chi^p` enters `M*`, so `p` and
the bracket are meaningful only as a pair, and the code locks `p = 1`.

### The solve variable

Define

    Phi = phi_bar^4 = <G^2>_med / <G^2>_vac

in which the potential and the dielectric are simply

    U        = B_g [ Phi (ln Phi - 1) + 1 ]
    chi      = (1 - Phi)^p
    dU/dPhi  = B_g ln Phi

`U(1) = 0` at the physical vacuum and `U(0) = B_g` at the perturbative point,
which is what makes `B_g` the glue part of the effective bag constant. The
limit `Phi ln Phi -> 0` is taken explicitly at `Phi = 0`; without it the
perturbative point returns NaN.

**`phi` is the field; `Phi` is the solve variable.** `Phi` cannot be the
Lagrangian field: its kinetic term
`(1/2)(d phi)^2 = (phi_0^2/32) Phi^(-3/2) (d Phi)^2` is not canonical and
diverges at `Phi -> 0`, `[Phi] = 4` makes `(d Phi)^2` dimension ten so no
renormalisable Lagrangian exists in it, and the glueball mass would stop being
`U''`. But as a solve variable it is strictly better, for one concrete reason:
written in `phi_bar` the dilaton residual has a **spurious root at
phi_bar = 0**, where both of its terms vanish as `phi_bar^3` — the first as
`phi_bar^3 ln phi_bar`, the second as `phi_bar^3` — so it is satisfied there
for *any* scalar density. It is an artefact of the parametrisation: the
Jacobian `dPhi/dphi_bar = 4 phi_bar^3` vanishes, not the physics. In `Phi`,
where `dU/dPhi = B_g ln Phi -> -infinity`, no such root exists. A Newton solve
in `phi_bar` landed on it from three of five starting points.

`Phi` is evaluated only inside

    Phi in [1e-14, 1 - 1e-13]

a clamp and not a physical bound. A Newton step may propose a dilaton outside
`[0,1]`, where the residual is NaN and the solve dies with no information;
evaluating at the edge instead leaves `dU/dPhi` strongly signed, which pushes
the next step back inside. The ceiling is not `1 - 1e-9`: `Phi = 1` exactly is
where the *confined* branch's solution genuinely sits (with no quarks `R_1`
reduces to `B_g ln Phi`, whose only root is `Phi = 1`), so the ceiling must be
close enough that `|ln Phi| = 1e-13` falls inside the residual gate, and far
enough that `M* = 2.8e15 MeV` stays an ordinary double. `1 - 1e-13` is both.

### Mean fields

At homogeneous mean field every boson is replaced by its expectation value,
`pi -> 0` by parity, and `omega_mu -> delta_(mu 0) omega_0`:

    L_MF = sum_f qbar_f [ i gamma^mu d_mu - g_omega(n_B) gamma^0 omega_0
                          - M*_f ] q_f
           + L_pair^MF
           - [ U(phi) + V(sigma, zeta) - (1/2) m_omega^2 omega_0^2 ]

with

    M*_u = (g_q sigma + m_u)/chi
    M*_d = (g_q sigma + m_d)/chi
    M*_s = (g_s zeta  + m_s)/chi

and the chiral potential

    V(sigma, zeta) = (lambda/4)(sigma^2 - v^2)^2
                     + (lambda_z/4)(zeta^2 - v_zeta^2)^2
                     - eps_sigma sigma - eps_zeta zeta + C_0

    dV/dsigma = lambda sigma (sigma^2 - v^2) - eps_sigma
    dV/dzeta  = lambda_z zeta (zeta^2 - v_zeta^2) - eps_zeta

`v` and `v_zeta` are constants, not fields: the Mexican-hat radii before
explicit breaking, and not observables. The linear terms shift the true minima
to `sigma_0`, `zeta_0`. A state whose effective masses are not all positive is
marked invalid and dropped by the enumeration rather than ranked: the Mexican
hat admits its reflected minimum, `sigma < 0`, which solves the same equation
and is not physical matter.

### The diquark condensate

The three condensates
`s_eta = <q^T C gamma_5 eps^(ab eta) eps_(ij eta) q>` give, to quadratic order
in the fluctuation,

    L_pair^MF = sum_eta [ (Delta_eta* O_eta + Delta_eta O_eta^dag)/2
                          - |Delta_eta|^2/(4 G_D) ]
    Delta_eta = 2 G_D s_eta

The mean-field treatment is exact for an auxiliary field appearing
quadratically, so substituting `Delta_eta = 2 G_D s_eta` back reproduces
`G_D |s_eta|^2` exactly; the gap equation `dOmega/dDelta_eta = 0` *is* the
condensate definition; and at the solution `Omega_eta = -G_D |s_eta|^2 < 0`, so
condensation lowers the grand potential as it must. All `Delta_eta` are taken
real, a convention since the phases are unobservable in a homogeneous
condensate.

**The sign of each gap is a gauge.** `Omega` is invariant under flipping any
subset of the three `Delta_eta` and each gap kernel flips with its own gap, so
`-Delta` is a root whenever `Delta` is. What is *reported* is the magnitude;
the signed values stay on the internal state and in the unknown vector, which
is what a warm start is built from.


## Chemical potentials

In the conserved-charge basis, with the colour potentials present only where
pairing is,

    mu_(f,a) = mu_B/3 + q_f mu_C + s_f mu_S + (T_3)_a mu_3 + (T_8)_a mu_8

Both colour generators are traceless, so colour drops out of `n_B`. The
potential fed to every momentum integral is shifted by the vector self-energy:

    mu*_(f,a) = mu_(f,a) - Sigma_V ,   Sigma_V = g_omega(n_B) omega_0 + Sigma_R

`mu_C` is the *charge* chemical potential and `mu_e = -mu_C`, so beta
equilibrium reads `mu_C + mu_e = 0`. Writing `+q_f mu_e` in place of
`+q_f mu_C` flips the u-d ordering: at `mu_B = 1200`, `mu_C = -53.15` MeV the
correct values are `mu_u = 364.6 < mu_d = 417.7` MeV, whereas the wrong sign
gives `mu_u = 435.4 > mu_d = 382.3` MeV.

### The lepton sector

Leptons feel no field the quarks feel, carry no colour and no `C` in the sense
above, and are free Fermi gases of `eos.general.thermodynamics_leptons`. Masses
`m_e = 0.5109989` MeV, `m_mu = 105.6584` MeV, degeneracy `g = 2` with
antiparticles included; trapped neutrinos are massless with `g = 1`, not 2 —
they are left-handed only. Their thermodynamics is the same ideal gas as
[The ideal-gas integrals](#the-ideal-gas-integrals) at `M* -> m_l`,
`mu* -> mu_l` and that degeneracy.

The potentials follow from the reactions that are fast:

    mu_e  = mu_nue - mu_C            (n + nu_e <-> p + e)
    mu_mu = mu_e - mu_nue            (mu- <-> e- nu_e_bar nu_mu)

with `mu_nue = 0` where the neutrinos free-stream — which is what
free-streaming *means* here: no lepton number and no pressure of their own.
Only the electron family is ever trapped; the muon neutrinos escape, which is
why `mu_mu = mu_e - mu_nue` rather than `mu_e`.

In a mode that *holds* `Y_C`, `mu_e` is not given by that relation at all. With
`leptons=True` the neutralizing leptons are solved **after** the matter, from
the single condition

    n_e(mu_e, T) + n_mu(mu_e - 0, T) = n_C

because nothing about them feeds back into the quark sector. They are therefore
not a row of the residual. With `leptons=False` there are no leptons at all and
the phase is electrically charged — which is what a mixed-phase construction
needs of each pure phase before imposing *global* neutrality.


## The vector coupling and its rearrangement

The vector coupling is a function of the state, not a parameter:

    g_omega(n_B)      = gbar_omega / [1 + (n_B/n_c)^2]
    dg_omega/dn_B     = -2 gbar_omega (n_B/n_c^2) / [1 + (n_B/n_c)^2]^2

with `n_c` stored in fm^-3 — a density a reader holds in fm — and converted to
MeV^3 where `n_B` is, so a caller never has to remember which side of the
boundary it is on.

A repulsion that dies off at high density is what keeps the sound speed away
from the causal limit without a hand-placed ceiling: the vector energy grows as
`g_omega^2 n_q^2`, so a coupling falling as `n_B^-2` turns it into a term that
stops growing at all. The derivative is negative everywhere, which is also why
the innermost fixed point on `omega_0` converges monotonically.
`gbar_omega = 0` switches the whole sector off exactly — no `omega_0`, no
`Sigma_R`, and no `Sigma_V` row in the unknown vector.

**The source is the quark number density `n_q = 3 n_B`**, because the coupling
in `L_vec` is to `qbar gamma^mu q`; using `n_B` understates `omega_0` by three
and the repulsive energy by nine:

    omega_0 = g_omega(n_B) n_q / m_omega^2

Because `g_omega` depends on the density, the shift of `mu` is the derivative
of the interaction energy `W = (1/2) m_omega^2 omega_0^2` with respect to the
*quark* density, not merely `g_omega omega_0`:

    Sigma_V = dW/dn_q = g_omega omega_0 + Sigma_R
    Sigma_R = (dg_omega/dn_B) omega_0 n_B

(the chain rule through `n_B = n_q/3` is what turns the naive
`(dg/dn_q) n_q` into the `n_B` written here). **Sigma_R enters mu and P and
never eps.** Note that `W` peaks at `n_B = n_c` and falls beyond it, so
`Sigma_R` — and with it `Sigma_V` — changes sign there; that is a property of
the coupling form, not of the implementation.

The diquark coupling may be dressed by the dielectric,

    G_D -> G_D / chi^q ,   q in {0, 1}

`q = 1` being the exponent a gluon-exchange origin gives and the largest that
leaves the pairing channel confined, the criterion being `q <= p` because the
critical coupling for vacuum diquark condensation grows only linearly in
`M* ~ chi^-p`.


## The ideal-gas integrals

Every momentum integral in this model is a relativistic ideal gas of one
colour-flavour mode with spin degeneracy `g = 2`. With
`E_k = sqrt(k^2 + M*^2)` and

    f^+- = [ 1 + e^((E_k -+ mu*)/T) ]^-1

the five integrals are

    N(M*,mu*,T)   = (g/2pi^2) int_0^kmax dk k^2 [ f^+ - f^- ]
                                                  (antiparticles SUBTRACT)
    R_s(M*,mu*,T) = (g/2pi^2) int_0^kmax dk k^2 (M*/E_k) [ f^+ + f^- ]
                                                  (antiparticles ADD)
    P(M*,mu*,T)   = (g/2pi^2) int_0^kmax dk k^2 T
                      sum_+- ln( 1 + e^(-(E_k -+ mu*)/T) )
    E(M*,mu*,T)   = (g/2pi^2) int_0^kmax dk k^2 E_k [ f^+ + f^- ]
    S(M*,mu*,T)   = (g/2pi^2) int_0^kmax dk k^2
                      sum_+- [ ((E_k -+ mu*)/T) f^+-
                               + ln(1 + e^(-(E_k -+ mu*)/T)) ]

They satisfy, per mode and to machine precision,

    N = dP/dmu* ,   R_s = -dP/dM* ,   S = dP/dT ,
    E = -P + mu* N + T S

Note that the single-mode Euler relation carries `T S` and **not** `M* R_s`:
the scalar density is a response, not a conserved charge.

`S` is *integrated*, not obtained from `S = (E + P - mu* N)/T`. The two are
equal identically — the identity holds integrand by integrand, so it even
survives the cutoff — but the Euler route is a difference of three numbers of
order 1e9 divided by `T`, and in a cold nearly degenerate gas, where `S` is
genuinely 1e-8 of them, the cancellation eats every significant digit.
Integrating costs one more array in the same quadrature pass and leaves the
identity available as a *check* rather than spending it as a definition.

`P` is the **logarithm** form above and not the `k^4/E` form
`(g/6pi^2) int dk (k^4/E_k)(f^+ + f^-)`. The two differ by the boundary term of
the integration by parts,

    P_log - P_k4 = (g/6pi^2) kmax^3 T sum_+- ln(1 + e^(-(E_kmax -+ mu*)/T))

which does not vanish when the integral is cut and is not small: 0.1% of `P` at
`(M*, mu*, T) = (100, 500, 20)` MeV, 10.5% at `(40, 590, 30)` and 39.9% at
`(140, 700, 50)`. At `T = 0` with `k_F < kmax` the two agree, which is exactly
why the error would hide until a table was built at finite temperature.

**The medium integrals are unregularised.** `kmax` is a numerical ceiling at
which the integrand has died,

    kmax = max(|mu*|, M*) + 45 T + 12 M* + 200 MeV

and not a parameter; the `12 M*` term covers the antiparticle tail of a heavy
confined mode, whose integrand decays on the scale of `M*` rather than of `T`.
The cutoff `Lambda` applies to the **pairing** integral alone. That is why a
sharp cutoff is admissible here, and it implies the validity ceiling
`mu_ceiling = sqrt(Lambda^2 + m_s^2)` for the pairing sector, which the code
declares rather than exceeds.

### Quadrature, and confinement

The integrals are evaluated by **panel-split Gauss-Legendre** quadrature with
breakpoints at the Fermi momentum `k_F = sqrt(mu*^2 - M*^2)` and at
`k_F +- 25 T`. This is not a refinement: a single panel cannot resolve a step
of width `~T` inside a range of `~1500` MeV, and single-panel accuracy is not
even monotone in the node count, because whether a node lands near the Fermi
step is accidental. At `T = 0.5` MeV the split scheme reaches with 100 nodes
per panel what a single panel needs 5000 to reach.

**At T = 0 a mode with `M* >= mu*` contributes identically zero** — exactly
0.0, not a small number. That is the confinement mechanism itself: as
`phi_bar -> 1` the dielectric closes, `M*` diverges, and the quarks leave the
medium. Smoothing the threshold destroys the pinning that makes deconfinement
first order here rather than a crossover. At `T > 0` the same statement is a
threshold rather than an identity, applied where `M* - |mu*| > 60 T`: the
occupation there is `e^-60 ~ 1e-26`, so it is exact well inside double
precision, and it exists because the confined branch drives `M*` to 1e15 MeV,
where integrating is not wrong but pointless.

At `T = 0` and `mu* > M*` the closed forms are, with
`L = ln((k_F + mu*)/M*)`:

    N   = (g/6pi^2) k_F^3
    R_s = (g/4pi^2) M* ( k_F mu* - M*^2 L )
    S   = 0
    P   = (g/48pi^2) [ 2 k_F^3 mu* - 3 M*^2 ( mu* k_F - M*^2 L ) ]
    E   = (g/16pi^2) [ 2 k_F^3 mu* +   M*^2 ( mu* k_F - M*^2 L ) ]

The code reaches them as the `T -> 0` limit of the same quadrature — the
occupations become step functions and the upper limit becomes `k_F` — rather
than by a separate branch, so the two agree by construction.

The photon gas, where the `photons` flag enables it, is the massless `mu = 0`
boson with `g = 2`:

    P_gamma = (pi^2/45) T^4 ,  eps_gamma = 3 P_gamma ,
    s_gamma = (4 pi^2/45) T^3

carrying no conserved charge.


## The pairing sector

The pairing machinery is `eos.general.pairing`, shared with `eos.njl` because
the pairing sector of the two is the same sector. It is stated here in full so
that this document is self-contained.

### The gap matrix and the BdG problem

Build the 9x9 gap matrix on the colour-flavour modes directly from the
antisymmetric tensors,

    G = sum_eta Delta_eta B_eta ,
    (B_eta)_((a i),(b j)) = eps^(ab eta) eps_(ij eta)

Its eigenvalue multiplicities are *derived*, never assigned: unpaired `0 (x9)`;
2SC `+-Delta (x2), 0 (x5)`; uSC/dSC
`+-Delta (x2), +-sqrt(Delta_a^2 + Delta_b^2), 0 (x3)`; CFL
`Delta (x3), -Delta (x5), 2 Delta`.

**The eigenvalues alone are not enough.** As soon as the quark masses differ,
`[G, diag(M*_f)] != 0`, so the gap matrix and the mass matrix share no
eigenbasis and the problem does not factorise into independent scalar
dispersions. At `M* = (60, 65, 300)`, `mu* = 450`, `Delta = 60` MeV the
eigenvalue prescription misses branches by up to 33.6 MeV in uSC and 27.1 in
CFL. So at each momentum the code assembles and diagonalises

    H_BdG(k) = [[ xi(k),  G     ],
                [ G,     -xi(k) ]]
    xi^(j),r(k) = E_(k, f(j)) - r mu*_j ,   xi = diag(xi^(j),r)_(j=1..9)

an 18x18 real symmetric matrix, once for particles (`r = +1`) and once for
antiparticles (`r = -1`). The nine quasiparticle energies `E^(j),r_Delta(k)`
are the **non-negative half of the signed spectrum**, `sort(eigvalsh)[9:]`, and
not the nine largest in modulus: the two agree in value, but only the first is
smooth through a gapless window, and that smoothness is what makes the
Hellmann-Feynman derivatives below carry the correct branch sign. Write
`V^(j),r(k)` for the corresponding 18-component eigenvector and split it as
`V = (V_top, V_bot)` into its two 9-component halves; the matrix is real, so
`|V_top,(j b)|^2` is simply its square.

### The pairing potential, as a correction

    Omega_quark = - sum_(j=1..9) P(M*_j, mu*_j, T)
                  + dOmega_pair
                  + sum_eta Delta_eta^2/(4 G_D)

    dOmega_pair = -(1/2pi^2) int_0^Lambda dk k^2 sum_(r=+-) sum_(j=1..9)
                     [ varphi(E^(j),r_Delta) - varphi(|xi^(j),r|) ]

    varphi(x)         = x + 2 T ln(1 + e^(-x/T))
    varphi'(x)        = tanh(x/2T)
    dvarphi/dT (x)    = 2 ln(1 + e^(-x/T)) + (2x/T)/(e^(x/T) + 1)

Written as a *correction* — a difference from the unpaired spectrum — every
bracket vanishes identically at `Delta_eta = 0`, so `dOmega_pair = 0` there
exactly and the whole expression reduces to the unpaired one mode by mode. The
alternative "replacement" form, in which the paired modes' Fermi integrals are
dropped and the `|E|` integral put in their place, differs by a
`Delta`-independent but `Lambda`-dependent constant of about 2% of the term:
the gap comes out right and the equation of state is wrong. In the clean
weak-coupling limit it obeys the BCS logarithm,
`-dOmega_pair / [ mu^2 Delta^2 (ln(2 Lambda/Delta) - 1/2) ] -> 2/pi^2`,
approached from below as `Delta` falls.

**The antiparticle branches are not optional.** At `T = 0` they contribute 8.8%
of the pairing potential at `Lambda = 600` MeV and 17.1% at 1000 MeV, so
omitting them is a cutoff-dependent error that cancels nowhere. Every logarithm
is computed as `T logaddexp(0, -x/T)` and every occupation through `tanh`,
never as `ln(1 + e^(-x/T))` or `1/(1 + e^(x/T))` directly, which overflow for
`x/T >~ 700` — a `T = 1` MeV point with a 300 MeV branch reaches that
immediately.

### The three gap equations

There is one residual per channel, not one gap:

    R_Delta_eta = Delta_eta/(2 G_D)
                  - (1/2pi^2) int_0^Lambda dk k^2 sum_(r=+-) sum_(j=1..9)
                      < V^(j),r | [[0, B_eta],[B_eta, 0]] | V^(j),r >
                      tanh( E^(j),r_Delta / 2T )
                = 0

The matrix element is the Hellmann-Feynman derivative
`dE^(j),r_Delta/dDelta_eta`, and `dH_BdG/dDelta_eta` is the constant matrix
`B_eta` in the off-diagonal blocks, assembled once.

**This is never the `Delta/|E|` form**, which is obtained by differentiating
`|E|` as though every branch were positive. Against finite differences of
`dOmega_pair` at `mu_u = 400`, `mu_d = 500` MeV that form is wrong by a factor
12.0 at `Delta = 40` MeV, 1.7 at 60 and 1.3 at 80, and it makes the gap *grow*
with the Fermi-surface mismatch — the opposite of the physics.

### Paired densities, scalar densities and entropy

For the modes the gap touches, the densities are **not** the unpaired Fermi
integrals: pairing redistributes occupation around the Fermi surface. Every
paired quantity is again a *correction*, added to the unpaired sum without a
second code path,

    n_j        = N_j + dn_j
    rho_s,f    = sum_(j in f) R_s,j + drho_s,f
    s_quark    = sum_j S_j + ds_pair

and each correction is the Hellmann-Feynman derivative of `dOmega_pair`, taken
on `H_BdG` in the same quadrature pass. With
`W_(j,b),r = |V_top,(j b)^r|^2 - |V_bot,(j b)^r|^2`:

    dn_j     = -d(dOmega_pair)/dmu_j
             = (1/2pi^2) int_0^Lambda dk k^2 sum_(r=+-)
                 [ -r sum_(b=1..9) tanh(E^(b),r_Delta/2T) W_(j,b),r
                   + r tanh(xi^(j),r / 2T) ]

    drho_s,f = -d(dOmega_pair)/dM*_f
             = -(1/2pi^2) int_0^Lambda dk k^2 sum_(r=+-) sum_(j in f)
                 (M*_f/E_(k,f))
                 [ sum_(b=1..9) tanh(E^(b),r_Delta/2T) W_(j,b),r
                   - tanh(xi^(j),r / 2T) ]

    ds_pair  = -d(dOmega_pair)/dT
             = (1/2pi^2) int_0^Lambda dk k^2 sum_(r=+-) sum_(j=1..9)
                 [ dvarphi/dT (E^(j),r_Delta) - dvarphi/dT (|xi^(j),r|) ]

Note the sign of `drho_s`: the scalar density is `rho_s = +dOmega/dM*`
(equivalently `-dP/dM*`, which is the per-mode identity above), so the
correction carries the opposite sign to `dn_j`. *Substituting the unpaired
formula for either is a 15% density error at a realistic gap and breaks
neutrality and Euler simultaneously.* Using the unpaired entropy in a gapped
phase is a four-orders-of-magnitude error at low `T`: the suppression is a
property of the *spectrum*, and at matched Fermi surfaces the entropy ratio is
2.0e-4 at `T = 5` MeV and `Delta = 60` MeV, while at `M*_s = 300` MeV the
mismatch pushes the lowest branch from 60 to 11.9 MeV and the ratio only to
0.20.

`T = 0` is a branch of `varphi`, not a limit of it: in a fully gapped phase
every Taylor coefficient in `T` vanishes, so `varphi(x) -> x`,
`varphi'(x) -> sign(x)` and `dvarphi/dT -> 0` are substituted rather than
approached.

A state is flagged **gapless** when the smallest quasiparticle energy on the
quadrature grid has collapsed relative to the gap scale,

    min_(k,j,r) E^(j),r_Delta  <  1e-3 max_eta |Delta_eta|

A gapless solution is a real physical state, but it is not a stationary point
of the same functional as a fully gapped one, so comparing `Omega` across one
is not valid — which is why it is *reported* rather than silently ranked.

Colour neutrality is imposed **within** the pattern:

    n_3 = sum_f ( n_(f,r) - n_(f,g) )           = 0
    n_8 = sum_f ( n_(f,r) + n_(f,g) - 2 n_(f,b) ) = 0

These are the generator densities up to the constant factors 1/2 and 1/3 of
`T_3` and `T_8`; a row that must vanish does not care about its own
normalisation, and this is the form the literature states. In an *unpaired*
region `n_3` and `n_8` vanish identically at `mu_3 = mu_8 = 0` whatever else
the state does, so the colour potentials are unconstrained there and are pinned
rather than solved for; a paired phase's `mu_8` cannot be inherited from an
unpaired solution.


## Assembly

Write the mode sums `S_P = sum_j P_j`, `S_E = sum_j E_j`, `S_S = sum_j S_j`,
the pairing cost `D = sum_eta Delta_eta^2/(4 G_D)`, and the vector field energy
`W = (1/2) m_omega^2 omega_0^2`. The **matter block** — quarks and fields only,
which is what a phase owns and what `eos.mixed` consumes — is

    Omega_matter = U(Phi) + V(sigma, zeta) - W - Sigma_R n_q
                   - S_P + dOmega_pair + D

    eps_matter   = S_E + U + V + W + eps_pair

with `P_matter = -Omega_matter` and

    eps_pair   = dOmega_pair + D + T ds_pair + sum_j mu*_j dn_j
    s_matter   = S_S + ds_pair
    sum_j mu_j n_j                (the PHYSICAL mode potentials, not mu*)

The conserved-charge sums are

    n_f = sum_a n_(f,a)
    n_B = (1/3) sum_f n_f = n_q/3
    n_C = sum_f q_f n_f
    n_S = n_s

**The signs.** `U + V` enter `eps` positively and `P` negatively; the vector
field energy `W` enters *both* positively; the rearrangement term
`Sigma_R n_q` enters `P` only. No vacuum subtraction appears anywhere, because
`U` and `V` both vanish at the physical vacuum by construction.

The audit that must hold at every solved point, on the matter block, and does
to machine precision — measured `-7.0e-16` relative at `n_B = 1.5` fm^-3,
`T = 0`, beta equilibrium — is

    eps_matter + P_matter = T s_matter + sum_j mu_j n_j

together with `f = eps - T s = -P + sum_j mu_j n_j` and `P = -Omega` computed
from the two assemblies independently.

### The totals

The sectors shared by the system rather than owned by the phase — leptons,
photons, and the neutrino flavours not tracked in the composition — are added
on top of the matter block:

    P   = P_matter   + sum_l P_l   + [gamma] P_gamma     + [nu] N_nu P_nu(0,T)
    eps = eps_matter + sum_l eps_l + [gamma] 3 P_gamma   + [nu] N_nu eps_nu(0,T)
    s   = s_matter   + sum_l s_l   + [gamma] s_gamma     + [nu] N_nu s_nu(0,T)
    f   = eps - T s

`[gamma]` and `[nu]` are the `photons` and `thermal_neutrinos` flags, and the
sum over `l` runs over the electrons, the muons where `muons` is set, and the
trapped electron neutrinos where `mu_nue != 0`; every one of those is the ideal
gas above. `N_nu` is the number of *untracked* massless `mu = 0` neutrino
flavours,

    N_nu = 3   free-streaming (mu_nue = 0)
    N_nu = 2   trapped

two in the trapped case because the electron flavour is then already counted at
its own potential. These carry no conserved charge and enter `P`, `eps` and `s`
only.

### The scalar density, and why it is not a trace identity

The model returns per-flavour scalar densities `rho_s,f`, and they are the
integral `R_s` plus the pairing correction `drho_s,f`, computed directly.
**The identity `n_s = (eps - 3P)/m*` does not apply here** and is not used: it
holds for a single ideal Fermi gas, whereas `eps` and `P` above carry `U`, `V`,
`W`, `Sigma_R n_q` and the pairing terms, and there are three distinct `M*_f`
rather than one `m*` to divide by. What replaces it as the audit is the Euler
relation above together with the per-mode relations `N = dP/dmu*`,
`R_s = -dP/dM*`, `S = dP/dT` — which is the stronger statement: `rho_s,f` is
checked as `-dP/dM*_f`, at every flavour, rather than as one combination of
totals. The quantity called `n_S` in this model is *strangeness*, not a scalar
density; the collision of names is why this paragraph exists.

### Two corrections to the specification

`docs/ccdm_implementation.md` is the authority for this model, and two places
in its assembly do not survive the audit its own section 9.6 mandates.

  - its section 4.3 writes `eps` with `-(1/2) m_omega^2 omega_0^2`. **The sign
    must be plus.** A repulsive vector interaction adds energy density: the
    Hamiltonian density of the mean vector field is
    `g_omega omega_0 n_q - (1/2) m_omega^2 omega_0^2 = +(1/2) m_omega^2
    omega_0^2` once the field equation is used. This is the standard
    density-dependent mean-field result — the scalar potentials enter `eps`
    positively and `P` negatively, the vector term enters *both* positively;

  - its section 4.1 carries `Sigma_R` in `mu*` but omits the compensating
    `-Sigma_R n_q` from `Omega`. Without it `n = -dOmega/dmu` and Euler both
    fail as soon as `g_omega` depends on the density. The rearrangement term
    enters `mu` and `P` and **never** `eps`.

With either error present the Euler residual is of order percent while every
other quantity still looks like a reasonable equation of state; with both
corrections it is ~1e-16.


## The residual

The unknown vector, in the order the solver assembles it, is

    X = ( Phi, sigma, zeta, [Sigma_V], [Delta_eta]_(eta free), [mu_3, mu_8],
          mu_B, mu_C, [mu_S], [mu_nue] )

the bracketed entries being present only where the model or the mode has them:
`Sigma_V` where `gbar_omega != 0`, the gaps as the pairing pattern declares,
the colour potentials whenever the pattern pairs at all, `mu_S` iff the mode
holds `Y_S`, and `mu_nue` iff it holds `Y_Le`. Where `mu_S` is *not* an unknown
it is zero, because strangeness-changing weak reactions are fast on any
astrophysical timescale.

The rows, in the same order:

    R_1  dilaton       B_g ln Phi + (p/(1 - Phi)) sum_f M*_f rho_s,f
    R_2  light scalar  dV/dsigma + (g_q/chi) (rho_s,u + rho_s,d)
    R_3  strange       dV/dzeta  + (g_s/chi) rho_s,s
    R_4  vector        Sigma_V - [ g_omega(n_B) omega_0 + Sigma_R ]
    R_Delta_eta        Delta_eta/(2 G_D) - kernel_eta      one per free gap
    R_n3, R_n8         n_3 , n_8                           colour neutrality
    R_nB               n_B - n_B_target                    the density
                       + the mode's charge rows

and the charge rows are, in this order and only where the mode has them:

    C held           n_C - Y_C n_B
    C equilibrated   n_C - [ n_e(mu_e,T) + n_mu(mu_mu,T) ]
    S held           n_S - Y_S n_B
    L_e held         n_e(mu_e,T) + n_nue(mu_nue,T) - Y_Le n_B

with `mu_e`, `mu_mu` from the lepton relations above. Exactly one of the two
`C` rows is present. In a fixed-fraction mode the neutralizing leptons are
**not** a row: they are solved after the matter, from the charge the matter
turned out to carry.

`R_1` is `dOmega/dPhi`, obtained from `dOmega/dphi_bar` by dividing out the
Jacobian `4 phi_bar^3`; that division is exactly what removes the spurious
root. `R_1`–`R_3` are minima of `Omega`, with the boundary minimiser
`sigma -> 0` admitted — that is how chiral restoration appears.

**`R_4` carries `Sigma_V`, not `omega_0`.** The vector field is circular:
`omega_0` sets `mu*`, which sets `n_B`, which sets `omega_0`. Carried as the
total shift `Sigma_V` everything downstream is explicit — `mu*` from
`Sigma_V`, the densities from `mu*`, `omega_0` and `Sigma_R` from the densities
— and `R_4` is the single statement that the returned field is the one the
returned densities source. It is also this repository's declared convention for
a density-dependent coupling: the unknown vector uses the effective potentials,
so the rearrangement and the large vector shift cancel out of the iteration.

### Row scales and the convergence gate

Each row is divided by the scale of the quantity it balances *before* it
reaches the root finder, so the vector handed to it is dimensionless:

| row | scale | what it is |
|---|---|---|
| `R_1` | `B_g` | an energy density |
| `R_2`, `R_3` | `\|eps_sigma\|`, `\|eps_zeta\|` | scalar densities, against what they balance in vacuum |
| `R_4` | `max(\|mu_B\|,1)/3` | a chemical potential |
| `R_Delta_eta`, `R_n3`, `R_n8` | `max[(mu_B/3)^3/pi^2, 1]` | densities in MeV^3, against the quark scale |
| `R_nB`, charge rows | `max(n_B, 1e-3)` | baryon densities in fm^-3 |

Unscaled, the rows span twenty-two orders of magnitude — the dilaton row is an
energy density in MeV^4 against a charge row in fm^-3 — and the norm would
report whichever carries the largest units.

The solve is Powell's hybrid method, then Levenberg-Marquardt, then (when a
warm start was given) one more hybrid attempt from the mode's own cold guess,
each with a bounded iteration count; the root finder's own tolerance is 1e-13,
and `converged` is the separate repository-wide gate

    max_i |R_i| / scale_i  <  1e-10

so that "converged" means one thing in every model here. **Non-convergence is a
return value**, never an exception and never a hang: the returned point carries
the best iterate reached and a status the caller tests.

### The cold start

With `n_q = max(3 n_B (hc)^3, 1)` MeV^3 and
`mu_q = max[ ((1/2) pi^2 n_q)^(1/3), 50 ]` MeV — the free massless relation at
one flavour per colour, floored so a vanishing density does not give a
vanishing potential — the default guess is

    (Phi, sigma, zeta) = the branch seed (table below)
    Sigma_V            = 0
    Delta_eta          = pattern seed x max(0.1 mu_q, 20)
    (mu_3, mu_8)       = (0, -0.02 mu_q)
    mu_B               = 3 mu_q
    mu_C               = 0 in CFL, else -0.05 mu_q
    mu_S               = -0.6 mu_q [ 1 - min(Y_S, 1) ]
    mu_nue             = 0.3 mu_q

Three of these are choices rather than conveniences. `Sigma_V = 0` because
`R_4` is nearly linear in `omega_0`, so the first Newton step lands close and a
nonzero seed only risks starting on the far side of the density the coupling is
evaluated at. `mu_C = 0` in CFL because CFL matter is electrically neutral
*without* electrons, and seeded with an electron-bearing potential a CFL solve
converges to a spurious point. `mu_S` is seeded on the sign that *suppresses*
strangeness: with `S = +1` per s quark the s modes sit at
`mu*_s = mu_B/3 - mu_C/3 + mu_S - Sigma_V`, so a negative `mu_S` is what pushes
them below their own effective mass, and seeded at zero the strange sector
starts fully populated.

That seed does **not** make `Y_S = 0` converge and is not meant to. Once `M*_s`
rises above `mu*_s` the strange density is identically zero, the strangeness
row is satisfied for a whole *range* of `mu_S`, and its column of the Jacobian
vanishes, so the solve stalls on the threshold with `mu_S` wherever its path
reached. That is the general statement that a potential is only pinned as
tightly as its conjugate density responds, and it is recorded in
`docs/DEFERRED.md`.

A warm start replaces this seed with a converged neighbour's unknown vector.
The fields are **not** carried across a branch — a branch *is* a choice of
field root, so seeding the restored branch from the confined one's fields hands
it the very root it is meant to be an alternative to — and a seed belongs to
the layout it was solved in, since a 2SC vector has one gap in it and a CFL
vector three.


## The internal system: the phase-adapter surface

`eos.mixed` couples this model to a hadronic one through one function, which
maps `(mu_B, mu_C, mu_S, T)` to a thermodynamic block. That function solves the
model's own *internal* self-consistency at those fixed potentials and nothing
else: the unknowns are

    X_int = ( Phi, sigma, zeta, [Sigma_V], [Delta_eta]_(eta free),
              [mu_3, mu_8] )

and the rows are `R_1`, `R_2`, `R_3`, `[R_4]`, the free gap equations and,
where the pattern pairs, the two colour-neutrality rows — that is, the residual
above without the density row and without any charge row. The scales are the
same, with `mu_B` the given potential.

Colour neutrality stays *inside*: it is a structural property of a
colour-superconducting phase rather than a condition a caller chooses, so the
engine never learns that `mu_3`, `mu_8` or a dilaton exist. The branch and the
pattern are **declared** to this function, not discovered by it; choosing
between them is a comparison across separate self-consistent solves and belongs
to the caller.


## Branches and patterns: what is enumerated

Neither the chiral/dielectric **branch** nor the pairing **pattern** is a mode:
both are decided by which candidate minimises the free energy.

Below the deconfinement onset two chiral branches coexist at fixed dilaton — a
confined one, where `sigma ~ f_pi`, the dielectric is nearly opaque and the
quarks are too heavy to appear at all, and a restored one, where `sigma` has
collapsed and the quarks are present. **A solver that alternates between
updating `sigma` and `omega_0` two-cycles between them** and exits with a mixed
state — `sigma` from one branch, `omega_0` from the other — which reads as a
spuriously deep minimum at zero quark density. So each branch is seeded
separately, solved to self-consistency and compared; a branch that fails to
converge is reported missing, never replaced by a neighbour, because
substituting a converged neighbouring point is how a fake phase boundary
appears.

| branch | seed `(Phi, sigma/sigma_0, zeta/zeta_0)` | at fixed `mu` | at fixed `n_B` |
|---|---|---|---|
| `confined` | `(1 - 1e-6, 1, 1)` | yes | **no** |
| `restored` | `(0.4^4, 0, 0)` | yes | yes |
| `partial`  | `(0.4^4, 0.5, 0.7)` | yes | yes |

The fractions are of the *vacuum* condensates, so a parameter point with a
different `m_sigma` or `f_K` is seeded to its own vacuum rather than to the
shipped one. The confined seed sits just inside the `Phi` guard: starting
exactly at `Phi = 1` overflows `M*` rather than approaching it. The partially
restored seed exists because the transition is first order and the intermediate
root is a real one over a window of densities, not because the other two
sometimes fail.

The confined branch is enumerated at fixed *potential*, where it is what the
deconfined branch must beat — it carries no quarks, so its pressure is exactly
zero, and the onset is where the deconfined pressure crosses it — and *not* at
fixed density: with the dielectric closed `n_B = 0` identically, so no nonzero
density row can be met and the Jacobian's field columns all vanish.

**Which potential decides depends on what is held.** At fixed potential the
winner is the largest `P`, i.e. the smallest `Omega`; at fixed density the
winner is the smallest `f = eps - T s`, which is the right potential there. The
code uses each in its own place, and a candidate that did not converge is
dropped rather than substituted.

| pattern | free gaps | seed `(Delta_1, Delta_2, Delta_3)/Delta_scale` |
|---|---|---|
| `unpaired` | none | `(0, 0, 0)` |
| `2SC` | `Delta_3` | `(0, 0, 1)` |
| `uSC` | `Delta_2, Delta_3` | `(0, 0.6, 1)` |
| `dSC` | `Delta_1, Delta_3` | `(0.6, 0, 1)` |
| `CFL` | all three | `(1, 1, 1)` |
| `free` | all three | `(0.3, 0.6, 1)` |

`eta = 1,2,3` pair `(ds), (us), (ud)`. The default enumeration at `csc=True` is
`unpaired, 2SC, CFL, free`; with `csc=False` only `unpaired` exists. A pattern
is a declaration of which `Delta_eta` are unknowns, and it adds no code. The
gap equation has three roots at any Fermi-surface mismatch (zero, a barrier
maximum, and the physical BCS root), so which root a solve lands on is decided
by the seed; enumerating the seeds is what makes the answer an answer rather
than one solve repeated. `free` is the asymmetric seed that lets a CFL-layout
solve fall to something that is not CFL. Gapless states are *flagged* rather
than ranked.

The candidate set of a fixed-density solve is the *product* of the two
enumerations, because which pattern survives depends on the strange quark's
effective mass, which is a property of the branch.

**The first-order transition is real physics.** Between the onset and the point
where the deconfined branch turns around, `dP/dn_B < 0`: the mechanically
unstable side, which CLAUDE.md section 8 admits in a raw branch and which a
construction (Maxwell, Gibbs, or the eta-mixed phase of `eos.mixed`) removes
before any table reaches a structure solver. Sound speeds must be taken
one-sided on each branch, never across the transition. At the shipped parameter
point the deconfined pressure crosses zero near `n_B ~ 1.35` fm^-3; below the
onset there is no deconfined root at fixed density at all, and a solve there
returns a status rather than a fabricated state. The low-density half of a
hybrid equation of state comes from a hadronic model through `ccdm_phase`.


## Species flags

| flag | default | what it does |
|---|---|---|
| `csc` | `False` | the three gaps become unknowns, the pairing correction enters `Omega`, `eps`, `s` and every density, and `mu_3`, `mu_8` are solved from colour neutrality within the pattern |
| `muons` | `True` | the muon family, at `mu_mu = mu_e - mu_nue` |
| `thermal_neutrinos` | `False` | the untracked flavours as `mu = 0` gases; `P`, `eps`, `s` only |
| `photons` | `False` | the blackbody gas above; `P`, `eps`, `s` only |
| `hyperons` | `False` | fixed: there are no baryons here to be strange |
| `deltas` | `False` | fixed: no baryon resonances in a quark model |
| `thermal_mesons` | `False` | fixed: `sigma`, `pi`, `zeta` and `phi` *are* the mean fields that give the quarks their masses |

The three fixed flags **raise** if set otherwise; they are never quietly
ignored. Giving the scalars a thermal population would double-count the
condensate they already are, and the mesonic fluctuations that would populate
them are beyond mean field.


## Modes

The closure rows of the specification map onto the repository's four modes.
**The specification labels them R1..R5; they are relabelled M1..M5 here**,
because `R_1`..`R_4` are the residual rows above and one document cannot carry
two meanings of one symbol.

| | mode | variables | rows added to `R_1`..`R_4` and `R_nB` |
|---|---|---|---|
| M1 | `beta_eq_neutrinoless` | `(n_B, T)` | `n_C = n_e + n_mu`. `mu_S = 0`, `mu_e = -mu_C`, no `mu_nue` |
| M3 | `beta_eq_neutrino_trapped` | `(n_B, Y_Le, T)` | `n_C = n_e + n_mu` and `n_e + n_nue = Y_Le n_B`; `mu_nue` joins the unknowns; `nu` with `g = 1` |
| M2 | `fixed_YC`, `leptons=True` | `(n_B, Y_C, T)` | `n_C = Y_C n_B`; the leptons then neutralize *after* the matter. **No** weak equilibrium |
| M4 | `fixed_YC_YS`, `leptons=False` | `(n_B, Y_C, Y_S, T)` | `n_C = Y_C n_B` and `n_S = Y_S n_B`; `mu_S` joins the unknowns; no leptons at all |
| M5 | `fixed_YC_YS` at `Y_C = 1/2`, `Y_S = 0` | `(n_B, T)` | as M4: symmetric matter is a parameter choice, not a fifth mode |

M2 is the one worth stating twice: it imposes total electric neutrality
*without* imposing beta equilibrium, which is what merger and core-collapse
matter is on a dynamical timescale. **Weak equilibrium is a per-row closure,
never an identity built into `Omega`**; hardwiring it would make such matter
unrepresentable. `Y_C` and `Y_S` are per baryon, which differ by a factor three
from the per-quark `n_s/(3 n_B)` often plotted.

`leptons=False` is refused in the two beta modes, which are *defined* by their
leptons. The muon lepton family is not available as a conserved charge:
`beta_eq_neutrino_trapped` takes `(n_B, Y_Le, T)` and a `Y_Lmu` raises
(`docs/DEFERRED.md`); the muon *species* is available through the flag.

Wherever a temperature is accepted, entropy per baryon is accepted in its
place, through an outer one-dimensional solve for `T` of
`s/n_B = (s/n_B)_target`, since `s` is not a variable the residual carries.


## Parameters

Three tiers, and the split is what makes the model usable in an inference run:
tier 1 is fixed by vacuum data and never sampled, tier 2 is structural and
declared per run, tier 3 is the Bayesian vector. Everything is an argument;
nothing is module state.

### Tier 1: fixed by vacuum data

| symbol | value | what it is |
|---|---|---|
| `f_pi` | 93 MeV | pion decay constant |
| `m_pi` | 138 MeV | pion mass (isospin average) |
| `f_K` | 113 MeV | kaon decay constant |
| `m_K` | 496 MeV | kaon mass |
| `m_u`, `m_d` | 5 MeV | current masses |
| `m_s` | 95 MeV | current mass |
| `m_zeta` | 980 MeV | the strange scalar, `f_0(980)` |
| `m_phi` | 1600 MeV | the scalar glueball |
| `m_omega` | 783 MeV | the vector meson |

The masses and decay constants are the standard vacuum values (Particle Data
Group, PRD 110 (2024) 030001); `m_phi = 1600` MeV is the specification's choice
of glueball scale, following Drago, Fiolhais and Tambini.

Everything below follows in closed form, so the scalar sector has no free
normalisation left once tier 1 and `(m_sigma, B_g)` are chosen:

    sigma_0    = f_pi
    zeta_0     = sqrt2 f_K - f_pi/sqrt2
    phi_0      = 4 sqrt(B_g)/m_phi
    eps_sigma  = f_pi m_pi^2
    eps_zeta   = sqrt2 f_K m_K^2 - (f_pi/sqrt2) m_pi^2
    lambda     = (m_sigma^2 - m_pi^2)/(2 f_pi^2)
    v^2        = f_pi^2 - m_pi^2/lambda
    lambda_z   = (m_zeta^2 - eps_zeta/zeta_0)/(2 zeta_0^2)
    v_zeta^2   = zeta_0^2 - eps_zeta/(lambda_z zeta_0)
    C_0        from V(sigma_0, zeta_0) = 0

Numerically at the shipped `m_sigma = 550` MeV and `B_g^(1/4) = 150` MeV:

    sigma_0   = 93 MeV          zeta_0    = 94.0452 MeV
    eps_sigma = 1.77109e6 MeV^3 eps_zeta  = 3.80625e7 MeV^3
    lambda    = 16.3866         v^2       = 7486.83 MeV^2  (v = 86.5265 MeV)
    lambda_z  = 31.4135         v_zeta^2  = -4039.30 MeV^2
    C_0       = 2.43517e9 MeV^4 phi_0     = 56.25 MeV
    B_g       = 5.0625e8 MeV^4

**`v_zeta^2` is negative** at the baseline `m_zeta`: the strange quartic is
convex, explicit breaking dominating, so the strange sector does not break
chirally on its own in this truncation. The sign flips between
`m_zeta = 1100` and 1150 MeV. Never assume it is positive and never write
`v_zeta` as a square root — it appears only inside the bracket of `V`, squared
off.

### Tier 2: structural

`p = 1`, **locked** — `chi` and `p` are meaningful only as the pair `chi^p`,
and squaring the bracket silently doubles the confining-end exponent; any other
value raises. `q in {0,1}`, shipped at `q = 0`: the dielectric dressing
`G_D -> G_D/chi^q`, declared per run rather than sampled, with `q = 1` the
gluon-exchange exponent and the largest that `q <= p` admits.

### Tier 3: the sampled vector, and the shipped values

| symbol | shipped | prior support | status |
|---|---|---|---|
| `B_g^(1/4)` | 150 MeV | 120–250 MeV | the glue bag scale; sets `phi_0` and the onset |
| `g_q` | 3.0 | 3–6 | **pinned**: the specification's section 10 table quotes `M*_(u,d) = 826` MeV at `phi_bar = 0.90` and 1531 MeV at 0.95 in the confined branch, and both invert to `g_q = 3.00` |
| `g_s` | 3.0 | 3–8 | *not* pinned; 3.0 is the flavour-symmetric choice `g_s = g_q` |
| `m_sigma` | 550 MeV | 450–700 MeV | fixes `lambda` and `v` |
| `gbar_omega` | 4.0 | 0–12 | *not* pinned; mid-prior. 0 switches the vector sector off (the L1 -> L0 reduction) |
| `n_c` | 1.0 fm^-3 | 0.3–3 fm^-3 | *not* pinned; mid-prior |
| `G_D` | 5e-6 MeV^-2 | (with `Lambda`) | calibrated here, not quoted: at this value the gap sits inside the specification's 20–150 MeV window at `mu_q ~ 450` MeV — 30.0 MeV in 2SC and 119.4 MeV in CFL at `mu_B = 1450`, `mu_C = -30` MeV, `T = 0`. Below ~4.5e-6 the 2SC gap equation has no root but the trivial one there |
| `Lambda` | 600 MeV | 550–800 MeV | the **pairing** cutoff only; nearly degenerate with `G_D` |

Three of these eight are calibration knobs at mid-prior values rather than
measurements, and the code says so rather than dressing them otherwise.
`m_omega` is a normalisation convention, only `g_omega/m_omega` entering;
`m_phi` cancels from the bulk equation of state at fixed `B_g`, pricing
gradients and the glueball spectrum instead.

`Lambda` implies a declared validity ceiling for the pairing sector,

    mu_ceiling = sqrt(Lambda^2 + m_s^2) = 607.47 MeV

above which the paired Fermi surface has left the region the sharp cutoff
describes. A point past it is *flagged and returned*, not refused: the unpaired
thermodynamics is untouched by it, and a sampler must be able to score the
point.

### The published sets

| name | what it is |
|---|---|
| `baseline` | the shipped default above; `Parameters.default()` |
| `novector` | `gbar_omega = 0`: the L1 -> L0 reduction |
| `dressed` | `q = 1`: the gluon-exchange dressing of `G_D` |
| `stiff` | `B_g^(1/4) = 190` MeV: a heavier glue scale, later onset |

The four levels the specification names are what the flags and a zero coupling
select, not separate models: L0 quarks in the dielectric with no vector and no
pairing, L1 adds the density-dependent repulsion (`gbar_omega != 0`), L2 is L1
at finite temperature, L3 adds colour superconductivity (`csc=True`).

### The effective bag constant

Not an input. At the physical vacuum the field energy is zero by construction;
at the perturbative point it is

    B_eff = [ U(0) - U(phi_0) ] + [ V(0,0) - V(f_pi, zeta_0) ] = B_g + B_chi

with `U` and `V` the closed forms written out above, so nothing here is named
without being defined. At `B_g^(1/4) = 150`, `m_sigma = 550`, `m_zeta = 980`
MeV this gives `B_chi^(1/4) = 229.89` MeV and
`B_eff = (239.66 MeV)^4 = 429.39 MeV/fm^3`. **The chiral sector supplies the
larger part** — quoting `B_g` alone as "the bag constant" of this model is
wrong by a factor of six in energy density.


**Three routes to a parameter set.** CLAUDE.md section 6 makes model
parameters arguments, so all three have to exist. *By name:*
`Parameters.default()` is the baseline set, and `Parameters.named(name)` takes
any of the four published sets -- `'baseline'`, `'dressed'`, `'novector'`,
`'stiff'` -- an unknown name raising `KeyError` that lists them. *A new set:*
every field carries a default, so `Parameters(g_q=..., B_g4=...)` names only
what changes; the dataclass is frozen, so `dataclasses.replace` is how a set
already in hand is modified. *From nuclear-matter parameters:* no route, and
none is missing -- CCDM has no nuclear sector, so there is no `nmp.py` and
nothing to invert; the tier-1 parameters are fixed by vacuum data instead.

## What is returned

### One point

A point carries a convergence status first — `ok`, a `message`, and the `point`
itself, which is present even when `ok` is false so a caller can see where the
solve got to. The point carries:

| group | fields | units |
|---|---|---|
| status | `converged`, `error` (the scaled residual), `mode` | |
| state point | `n_B`, `T`, `Y_C`, `Y_S`, `Y_L` | fm^-3, MeV, — |
| the answer | `branch`, `pattern`, `Delta` = `(\|Delta_eta\|)`, `gapless`, `beyond_cutoff` | MeV |
| fields | `phi_bar` = `Phi^(1/4)`, `chi`, `sigma`, `zeta`, `omega_0`, `Sigma_R` | —, MeV |
| masses | `M_star` = `(M*_u, M*_d, M*_s)` | MeV |
| potentials | `mu_B`, `mu_C`, `mu_S`, `mu_3`, `mu_8`, `mu_e`, `mu_nu` | MeV |
| densities | `n_u`, `n_d`, `n_s`, `n_e`, `n_mu`, `n_nu` | fm^-3 |
| totals | `P_total`, `e_total`, `f_total` = `eps - T s`; `s_total` | MeV/fm^3; fm^-3 |
| fractions | `Y_u`, `Y_d`, `Y_s`, `Y_e`, `Y_nu` | — |
| internals | `state` (the matter block), `x` (the unknown vector, the warm start) | |

Five of those are part of the *answer* rather than diagnostics. `branch` and
`pattern` say which candidate won; `Delta` are the gap magnitudes, zero where
the pattern does not pair; `gapless` says the gapless test fired, so ranking by
`Omega` across this point is not valid; `beyond_cutoff` says
`max_j |mu*_j| > mu_ceiling`.

The matter block on `state` carries, in natural units, everything of the
assembly before the leptons: `T`, `Phi`, `phi_bar`, `chi`, `sigma`, `zeta`,
`omega_0`, `Sigma_V`, `Sigma_R`, `M*_f`, `Delta_eta`, the five
conserved-charge potentials, the nine `mu_j` and `mu*_j`, `rho_s,f`, the nine
`n_j`, the three `n_f`, `n_q`, `n_B`, `n_C`, `n_S`, `n_3`, `n_8`, `U`, `V`,
`Omega`, `P`, `eps`, `s`, `sum_j mu_j n_j`, the residual arrays
`(R_1..R_4)` and `R_Delta_eta`, the branch and pattern it was solved in,
`gapless`, `valid` (all `M*_f > 0`), `dOmega_pair` and `D`; plus the fm-based
conversions of `n_B`, `n_C`, `n_S`, `P`, `eps`, `s` and the Euler residual as a
callable audit.

### A table

A table is a set of *lines* — one per temperature (or entropy per baryon) and
per combination of the fractions the mode fixes — each swept along the baryon
density with a warm start, the winning candidate's seed carried forward and the
others left cold, so a candidate seeded only from itself can never fail to be
displaced. A missed density step is bisected back towards the last solved
point, at most six times; the thresholds that earn it are the strange quark's
onset, the pairing onset, and the branch change, where the fields move
discontinuously. A non-converged point is dropped from its line rather than
aborting the table, which is what the sub-onset densities need.

Each row carries `n_B`, `T`, `chi` = 1 and `phase` = `'Q'` (the shared
mixed-phase columns: this matter is entirely deconfined), `P`, `eps`, `s`,
`S_per_B`, `mu_B`, `mu_C`, `mu_S`, `mu_e`, `Y_C`, `Y_S`, `Y_u`, `Y_d`, `Y_s`,
`Y_e`, `Y_mu-`, `M_u`, `M_d`, `M_s`, `branch`, `pattern`, `gapless`,
`beyond_cutoff`, `phi_bar`, `chi_diel`, `sigma`, `zeta`, `omega_0`, `Sigma_R`,
`Delta_1`, `Delta_2`, `Delta_3`, `mu_3`, `mu_8`, and where the neutrinos are
trapped `Y_nue` and `mu_nue`. **`chi` in a table row is the mixed-phase quark
volume fraction**, the column name every quark and hadronic table shares; this
model's dielectric is `chi_diel`, and the collision is why it carries a suffix.

Progress is reported through the repository's shared callback, once per
completed line, with the dictionary

    {mode, line, n_lines, temp_key, temp, fracs, n_solved, n_requested,
     elapsed_s}

and two keys added here, `branch` and `pattern` — which phase a line ended in
is the thing a reader of this table wants first. Deep solver code never prints.

### Response functions

The quantities are those of the CompOSE manual (Typel et al., arXiv:2203.03209,
section 3.6). They are taken by central differences along re-solved sequences:
every neighbour of the stencil re-runs the whole equilibrium *and both
enumerations*. Only one conditioning is implemented, `frozen='equilibrium'` —
nothing is held, the composition and both enumerations re-equilibrate — and any
other raises. With `sig = s/n_B` the entropy per baryon, and every partial
derivative taken along the mode's own sequence:

    cs2_isothermal = (dP/dn_B)_T / (deps/dn_B)_T
    C_V            = T (dsig/dT)_n_B
    C_P            = T [ (dsig/dT)_n_B
                         - (dP/dT)_n_B (dsig/dn_B)_T / (dP/dn_B)_T ]
    cs2_adiabatic  = (C_P/C_V) cs2_isothermal
    Gamma_th       = 1 + [ P(n_B,T) - P(n_B,0) ] / [ eps(n_B,T) - eps(n_B,0) ]

Both sound speeds are named for the thermal variable they hold, never as a bare
`cs2` whose meaning would depend on the arguments. At `T = 0` the ratio
`C_P/C_V` is 1 by construction and the two coincide, so the cold limit needs no
special case; `C_V`, `C_P` and `Gamma_th` are returned at `T > 0` only.
`Gamma_th` returns NaN where the hot and cold states are on different
candidates, since the difference is then not a thermal one at all.

A sixth quantity is returned, and it is a warning rather than a number:
`branch_changed` says whether the density stencil straddled a branch or pattern
change, in which case every value above is a *chord across a first-order jump*
rather than a tangent and the derivative should be retaken one-sided by
restricting the enumeration. There is no way to see that from the returned
number alone, which is why it is returned rather than left to the caller to
suspect. In a fully gapped phase `C_V` is exponentially small at low `T` — the
paired entropy goes as `e^(-Delta/T)` — and that suppression is physics, not a
numerical failure: it is what makes a colour superconductor cool differently
from unpaired quark matter.

`cs2_isothermal` *may come out negative* on a raw branch between the onset and
the turnaround, because `dP/dn_B` is genuinely negative there. That is the
mechanically unstable side of the first-order transition, and what removes it
is a construction, applied before any table reaches a structure solver.

The composition freezes of the repository's response contract — held `Y_i`,
held `Y_C`, held `Delta_eta`, held fields — and the susceptibility matrix
`chi_ab = dn_a/dmu_b` are **not** implemented here and are recorded in
`docs/DEFERRED.md`: holding a composition needs the species fractions carried
through the solve as constraints, and holding the gaps or the fields needs them
fixed against their own equations.


## What is not in the model

The dilaton gradient and finite-size terms (this is a bulk, homogeneous mean
field). The de Carvalho contact coupling `h(phi)` (de Carvalho, Malheiro et
al., NPB Proc. Suppl. 199 (2010) 308) as a replacement for `G_D`: their
construction is sound — integrating out the confining field's fluctuation at
Gaussian order gives a contact interaction, which is why `L_pair` is a
legitimate leading term rather than an ad hoc addition, and it gives "zero
range" the physical meaning `1/M_chi` — but what it predicts is a coupling
negligible where the quarks are light and enormous where they are heavy, which
removes pairing from the deconfined phase where a star's core lives and would
condense a diquark in the confining vacuum. It carries `q = 4` at `p = 1`,
violating `q <= p`. What is taken from it is the argument, not the number.

The muon lepton family as a conserved charge, and every response conditioning
but `equilibrium`. Both are in `docs/DEFERRED.md`.


## Usage

    from eos.ccdm import Parameters, SpeciesFlags, eos_point, eos_table

    par = Parameters.default()
    r = eos_point(par, "beta_eq_neutrinoless", SpeciesFlags(csc=True),
                  n_B=1.5, T=0.0)
    r.ok, r.point.branch, r.point.pattern, r.point.Delta

    python -m eos.ccdm.verify.run_full_check
