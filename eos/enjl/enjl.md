# ENJL — the extended Nambu-Jona-Lasinio model of baryonic and quark matter

Baryons and quarks from **one** functional, as in Xia, PRD 110, 014022 (2024)
[arXiv:2405.02946]. A baryon is a three-quark cluster whose mass is built from
the same constituent quark masses that the NJL gap equation determines, so the
chiral, quarkyonic and deconfinement transitions all come out of a single mean
field rather than out of two models joined at a boundary. The independent
unknowns of that mean field are the three constituent masses `M_u`, `M_d`,
`M_s`; everything else — the baryon masses, the vector fields, the
rearrangement terms — follows algebraically. The model carries
density-dependent couplings `alpha_S(n_B)`, `Gamma_omega(n_B)` and
`Gamma_rho(n_B)`, and therefore rearrangement self-energies, and it carries the
NJL three-momentum cut-off `Lambda`, which enters as a temperature-independent
vacuum subtraction on the quark sector alone.

This file states every equation the code solves and every quantity it returns.
`enjl.tex` is the same document typeset, compiled against `../../docs/eos.bib`;
neither defers to the other. The reference is a `T = 0` paper; the
implementation is closed at any non-negative temperature, with entropy per
baryon accepted wherever a temperature is, and the numerics section says what
temperature costs and what is still open above the first first-order
transition.


## Conventions

Natural units inside the physics — densities in MeV^3, masses and potentials in
MeV, `eps` and `P` in MeV^4 — with `(hc)^3 = 7.6838e6 MeV^3 fm^-3` converting
to the fm^-3 and MeV/fm^3 used at every public boundary.

**Strangeness is S = +1 per s quark**, the opposite of the PDG sign, so
`Lambda` and `s` both carry `+1`. `C` is the electric charge of *strongly
interacting* matter only: the leptons are excluded from it, and total electric
neutrality is a separate condition imposed where a mode calls for it.
Fractions are per baryon, `Y_C = n_C/n_B` and `Y_S = n_S/n_B`.


## Degrees of freedom

Eight species, in three families, with the quantum numbers of
`eos.enjl.species`:

| species | `g` | `B` | `C` | `S` | `tau_3` | valence content `(N^u, N^d, N^s)` |
|---|---|---|---|---|---|---|
| `p`       | 2 | 1   | +1   | 0  | +1 | (2,1,0) |
| `n`       | 2 | 1   | 0    | 0  | -1 | (1,2,0) |
| `Lambda`  | 2 | 1   | 0    | +1 | 0  | (1,1,1) |
| `u`       | 6 | 1/3 | +2/3 | 0  | +1 | (1,0,0) |
| `d`       | 6 | 1/3 | -1/3 | 0  | -1 | (0,1,0) |
| `s`       | 6 | 1/3 | -1/3 | +1 | 0  | (0,0,1) |
| `e-`      | 2 | 0   | —    | 0  | 0  | — |
| `mu-`     | 2 | 0   | —    | 0  | 0  | — |

The quark degeneracy `g = 6` is three colours times two spins; the baryon and
lepton degeneracy `g = 2` is spin alone. The leptons carry physical electric
charge `q_e = q_mu = -1` but are excluded from `C` by the convention above.

The last column, `N^q_i`, is the number of valence quarks of flavour `q` in
baryon `i`, and it is the object through which the whole model is built: it
appears in the baryon mass (baryonmass), in the effective scalar density
(nbar), in the omega source (Jomega), and in the rearrangement term (SigmaRb).

A model-wide rescaling factor `f_i` multiplies each species' coupling to the
vector fields: `f_p = f_n = 1` by construction, `f_Lambda` is fitted to the
`Lambda` potential depth, and the three quarks share one value `f_q` which is a
parameter of the study. Leptons do not couple, `f_e = f_mu = 0`.


## Single-species thermodynamics, and the cut-off

Each species is a free Fermi gas of degeneracy `g` and effective mass `M` at
kinetic (effective) chemical potential `nu`, and it is the **only** place
temperature enters. At `T = 0` the gas is filled to a sharp Fermi momentum
`k_F = sqrt(nu^2 - M^2)` and every integral has a closed form; at `T > 0` there
is no sharp surface, `k_F` is not defined, and `nu` is the primary variable.

With `x = k_F/M` and `nu = sqrt(k_F^2 + M^2)`:

    n^med(k_F,M,g)   = g k_F^3/(6 pi^2)                            (nmed)
    n^s,med(k_F,M,g) = (g M^3/(4 pi^2)) [ x sqrt(x^2+1) - asinh x ]
                                                                 (nsmed)
    eps^med(k_F,M,g) = (g M^4/(16 pi^2))
                       [ x(2x^2+1) sqrt(x^2+1) - asinh x ]      (epsmed)
    P^med(k_F,M,g)   = nu n^med - eps^med
                     = (g M^4/(48 pi^2))
                       [ x(2x^2-3) sqrt(x^2+1) + 3 asinh x ]      (Pmed)
    s                = 0        every species, at T = 0               (s)

Equation (epsmed) includes the rest mass, and `eps + P = nu n` holds
identically for one species by (Pmed). Below threshold, `nu <= M`, all four
vanish. These are the standard `T = 0` Fermi integrals and are *not* written a
second time inside this model: the code calls
`eos.general.fermi_integrals.solve_fermi_jel` at `T = 0`, which routes to the
exact closed forms above. They are restated here because a paper-style
description must be self-contained.

### Finite temperature

At `T > 0` the occupation is Fermi-Dirac rather than a step, and antiparticles
are thermally populated. With `E(k) = sqrt(k^2 + M^2)` and

    f_∓(k) = 1 / ( 1 + exp[ (E(k) ∓ nu)/T ] )                 (fermidirac)

the occupation of particles (`f_-`) and antiparticles (`f_+`), the four medium
integrals become

    n^med(nu,M,g,T)   = (g/(2 pi^2)) int_0^inf dk k^2 [ f_- - f_+ ]  (nmedT)
    n^s,med(nu,M,g,T) = (g/(2 pi^2)) int_0^inf dk k^2 (M/E(k))
                                                     [ f_- + f_+ ] (nsmedT)
    eps^med(nu,M,g,T) = (g/(2 pi^2)) int_0^inf dk k^2 E(k)
                                                     [ f_- + f_+ ](epsmedT)
    P^med(nu,M,g,T)   = (g/(6 pi^2)) int_0^inf dk (k^4/E(k))
                                                     [ f_- + f_+ ]  (PmedT)

and the entropy density of the species follows from the Euler relation for one
free gas,

    s(nu,M,g,T) = ( eps^med + P^med - nu n^med ) / T                  (sT)

which is how the code obtains it, rather than by integrating
`-[f ln f + (1-f) ln(1-f)]` directly. Note the sign structure: the *number*
density is the difference of the two occupations, because an antiparticle
carries the opposite charge, while `n^s`, `eps` and `P` are sums. Setting
`T -> 0` at fixed `nu > M` returns (nmed)-(s) exactly, with
`f_- -> theta(k_F - k)` and `f_+ -> 0`.

The cut-off subtraction below is unchanged at `T > 0`: `n^s,vac` and `eps^vac`
depend on `(M, g, Lambda)` alone. The Dirac sea is a property of the vacuum,
and `Lambda` regularizes it there and not in the medium, so there is no
temperature in either term to carry.

**How the integrals are evaluated, and one thing to know about it.** Equations
(nmedT)-(PmedT) are not written a second time inside this model: they come from
`eos.general.fermi_integrals.solve_fermi_jel`, the single home for the Fermi
integrals of this repository, which uses the rational approximation of Johns,
Ellis and Lattimer (ApJ 473 (1996) 1020), accurate to about 1e-4 relative. That
routine evaluates the *exact* closed forms at `T = 0` and the fit at every
`T != 0`, and **the fit does not converge back to the closed form**: the answer
steps discontinuously the moment `T != 0` and then *stays* at that offset as
`T` falls. Flat from `T = 1e-3` MeV downward, the step is

    dn/n = +6.9e-6   u at nu = 400, M = 5.5 MeV
         = -6.3e-5   s at nu = 500, M = 140.7 MeV
         = -3.0e-6   n at nu = 1000 MeV                        (jelfloor)

with the same numbers in `eps`. Two consequences, both deliberate. First, "take
`T` small and compare against the `T = 0` answer" is not a validation route
below about 1e-4 relative, and the continuity check in `verify/` states that
floor rather than chasing it. Second, `T = 0` is kept as an exact special case
rather than routed through the fit for smoothness: doing so would move every
number this model has ever published, including the golden values of
CLAUDE.md §12, to buy a 1e-5 cosmetic gain. The entropy itself carries no such
offset — it is zero in both branches at `T = 0` — so (sT) vanishes smoothly.

### The cut-off, and the one way it enters

The NJL interaction is not renormalizable and is regularized by a
three-momentum cut-off `Lambda` applied to the *vacuum* (Dirac sea) integrals
of the quark sector. Baryons and leptons carry no cut-off. Writing
`y = Lambda/M`, the quark scalar density and energy density are the medium
terms *minus* a `k_F`-independent vacuum term,

    n^s(k_F,M,g,Lambda) = n^s,med(k_F,M,g) - n^s,vac(M,g,Lambda)
    n^s,vac(M,g,Lambda) = (g M^3/(4 pi^2)) [ y sqrt(y^2+1) - asinh y ]
                                                                 (nsvac)
    eps(k_F,M,g,Lambda) = eps^med(k_F,M,g) - eps^vac(M,g,Lambda)
    eps^vac(M,g,Lambda) = (g M^4/(16 pi^2))
                          [ y(2y^2+1) sqrt(y^2+1) - asinh y ]   (epsvac)

and the number density (nmed) is untouched, the Dirac sea carrying no net
baryon number. Two properties of this split matter and are easy to lose:

- **The medium integrals are not cut off.** Only the vacuum subtraction carries
  `Lambda`. This is not a detail: the quark kinetic potential `nu_q` exceeds
  `Lambda = 602.3` MeV above `n_B ~ 3 fm^-3`, so a cut applied to the medium
  integral would truncate the physical Fermi sea and change the equation of
  state exactly where it is being used.
- **The vacuum terms depend only on `(M, g, Lambda)`.** They are additive
  constants at fixed constituent mass, independent of `k_F` and — this is what
  made the finite-temperature extension a one-function change — independent of
  `T`. In the code they are the model's own analytic terms, added outside the
  call into `eos.general`.

Because `n^s,vac > 0`, the quark scalar density is *negative* in vacuum: at the
shipped parameters and the vacuum masses below,
`n^s_u(k_F = 0) = n^s_d(k_F = 0) = -1.8433 fm^-3` and
`n^s_s(k_F = 0) = -2.2270 fm^-3`. That negative value *is* the chiral
condensate, up to a positive factor, and it is what makes the constituent
masses large in vacuum through the gap equation.


## Parameters

Every number below is an argument (`eos.enjl.Parameters`, a frozen dataclass),
never a module-level constant. Two groups: the NJL set proper, which is the RKH
parametrization of Rehberg, Klevansky and Hüfner, PRC 53, 410 (1996), and the
density-dependent structural functions fitted by Xia (his Table I).

| symbol | code | shipped value | meaning |
|---|---|---|---|
| `Lambda` | `Lambda` | 602.3 MeV | three-momentum cut-off |
| `m_u0`, `m_d0` | `m_u0`, `m_d0` | 5.5 MeV | current light-quark masses |
| `m_s0` | `m_s0` | 140.7 MeV | current strange-quark mass |
| `G_S` | `GS` | `1.835/Lambda^2` MeV^-2 | scalar coupling, `G_S = g_sigma^2/(4 m_sigma^2)` |
| `K` | `K` | `12.36/Lambda^5` MeV^-5 | 't Hooft determinant coupling |
| `a_S, b_S, n_S` | `aS, bS, nS` | 0.4413715, 0.4076285, 0.16 fm^-3 | `alpha_S(n_B)` |
| `a_V, b_V, n_V` | `aV, bV, nV` | 3.566049, 1.062771, 0.214 fm^-3 | `Gamma_omega(n_B)` |
| `a_TV, b_TV, n_TV` | `aTV, bTV, nTV` | 0.5014459, 0.0117601, 0.1 fm^-3 | `Gamma_rho(n_B)` |
| `f_Lambda` | `f_Lambda` | 1.0626 | fixed by `U_Lambda(n_sat) = -30` MeV |
| `f_q` | `f_q` | 0.5 | quark vector rescaling; the study uses 0.5, 0.7, 1.0 |
| `B` | `B_GeV_fm3` | 1.0 GeV/fm^3 | Pauli-blocking strength, (baryonmass) |
| `m_e`, `m_mu` | `m_e`, `m_mu` | 0.511, 105.66 MeV | lepton masses |
| `m_sigma, m_omega, m_rho` | `m_sigma, m_omega, m_rho` | 630, 1e5, 769 MeV | bare meson masses, inert here |

`Parameters.default()` is `(f_q = 0.5, B = 1)`, the set of the paper's
Figs. 4-6 and the one `test/baseline` is frozen at;
`Parameters.named("fq1.0_B0")` and its five siblings are the study's other
combinations, named as the author's own tables are.

The three density-dependent functions, with `n_B` the total baryon density, are

    alpha_S(n_B)      = a_S exp(-n_B/n_S) + b_S                  (alphaS)
    Gamma_omega(n_B)  = 4 G_S [ a_V exp(-n_B/n_V) + b_V ]        (Gomega)
    Gamma_rho(n_B)    = 9 * 4 G_S [ a_TV exp(-n_B/n_TV) + b_TV ]   (Grho)

where `Gamma_omega = g_omega^2/m_omega^2` and `Gamma_rho = g_rho^2/m_rho^2`:
only the ratio `g^2/m^2` is determined by the model, which is why the bare
meson masses in the table are inert. They are carried because the Thomas-Fermi
treatment of Xia, Maruyama, Yasutake and Tatsumi (2024) needs them to fix the
interaction ranges once gradient terms are switched on. Its `m_omega = 1e5` MeV
is not a typo for 105: that work deliberately adopts a very large omega mass to
suppress density fluctuations in its Thomas-Fermi solve, and says so.

Their derivatives, needed by the rearrangement terms, are read off
analytically:

    alpha_S'(n_B)     = -(a_S/n_S) exp(-n_B/n_S)
    Gamma_omega'(n_B) = -4 G_S (a_V/n_V) exp(-n_B/n_V)
    Gamma_rho'(n_B)   = -36 G_S (a_TV/n_TV) exp(-n_B/n_TV)

**The factor 9 in `Gamma_rho`** is one the printed Eq. (22) of the paper does
not carry. It is required by the isospin source used here,
`J_rho = sum_i f_i tau_i n_i` with `tau_p = +1`, `tau_n = -1`, and it is
confirmed twice, independently:

- the published symmetry energies are reproduced with it and not without it —
  `E_sym(0.1) = 25.50` and `E_sym(n_sat) = 31.55` MeV measured, against 25.5
  and 31.5 published, where the literal Eq. (22) gives 13.3 and 20.2;
- on the nucleonic rows of the author's own tables the isospin splitting of the
  printed potentials gives
  `g_rho rho = (1/2)[(E_F^n - E_F^p) - (mu_n - mu_p)]`, and dividing by
  `J_rho = n_p - n_n` returns exactly `9.0000 x` Eq. (22) at every one of those
  densities, with no fit involved.

The omega channel needs no such factor, because its source (Jomega) already
carries `N_i = 3` for the baryons. The constant lives in the code as
`eos.enjl.parameters.RHO_FACTOR`. It is not a bug to be "fixed" back.


**Three routes to a parameter set.** CLAUDE.md section 6 makes model
parameters arguments, so all three have to exist. *By name:*
`Parameters.default()` is the set of Figs. 4-6, and `Parameters.named(name)`
takes any of the six published (f_q, B) combinations -- `'fq0.5_B0'`,
`'fq0.5_B1'`, `'fq0.7_B0'`, `'fq0.7_B1'`, `'fq1.0_B0'`, `'fq1.0_B1'` -- named
after the author's own tables, an unknown name raising `KeyError` that lists
them. *A new set:* every field carries a default, so
`Parameters(f_q=..., GS=...)` names only what changes; the dataclass is frozen,
so `dataclasses.replace` is how a set already in hand is modified. *From
nuclear-matter parameters:* no route. ENJL spans both baryons and quarks and
has no `nmp.py`: its couplings are fixed by vacuum data and by the author's
tables, not by a saturation-property list.

## The mean field

### Effective scalar densities and the gap equation

The scalar source of the gap equation is not the quark scalar density alone. A
baryon is a cluster of three quarks, so its scalar density contributes to the
condensate of each of its valence flavours, weighted by `alpha_S`:

    nbar^s_q = min( n^s_q(k_F^q, M_q, 6, Lambda)
                    + alpha_S(n_B) sum_{i=p,n,Lambda} N^q_i n^s_i , 0 )
                                                                  (nbar)

with `n^s_q` from (nsvac) (cut-off included) and `n^s_i` from (nsmed) at
`Lambda = 0`. The constituent masses then satisfy the three-flavour NJL gap
equation with the 't Hooft determinant term,

    M_u = m_u0 - 4 G_S nbar^s_u + 2 K nbar^s_d nbar^s_s            (gap)

and cyclically for `d` and `s`. Two statements about (nbar)-(gap) are
properties of *this implementation* and not of the printed equations, and both
change numbers:

- **The determinant term is written over the other two flavours.** The paper
  writes it as `2 K nbar^s_u nbar^s_d nbar^s_s / nbar^s_q`, which is the same
  number wherever `nbar^s_q != 0` but is `0/0` at chiral restoration, exactly
  where a solver needs it. Equation (gap) is that expression with the removable
  singularity removed.
- **`nbar^s_q` is capped at zero from above.** It is the condensate up to a
  positive factor (`g_sigma sigma_q = 4 G_S nbar^s_q`), so it is negative in
  vacuum and rises towards zero as chiral symmetry is restored. A positive
  value would be a condensate of the wrong sign and would drive `M_q` *below*
  its current mass. The cap binds because the cluster term of (nbar) is
  positive and grows with baryon density: in symmetric nucleonic matter at
  `n_B = 10 fm^-3` the uncapped expression returns `+0.105 fm^-3` for the `u`
  and `d` flavours, long after the light condensates have vanished, while the
  `s` flavour — which the nucleons do not feed — is still at `-2.072 fm^-3` and
  is not capped. Once `nbar^s_q = 0`, (gap) returns `M_q = m_q0` exactly and
  that flavour decouples from the determinant term of the other two.

The gap equation is the model's only genuine self-consistency: the three `M_q`
are the unknowns, and everything in the rest of this section is algebraic once
they are known.

### Baryon masses

A baryon's mass is the sum of its valence constituent masses, interpolated by
`alpha_S` between the current and constituent values, plus a Pauli-blocking
term proportional to the baryon density carried by *deconfined* quarks,

    M_i = sum_q N^q_i [ m_q0 + alpha_S(n_B) (M_q - m_q0) ] + B n_B^Q
    n_B^Q = (n_u + n_d + n_s)/3                              (baryonmass)

The second term is what makes baryons unbind: as quarks are liberated, `n_B^Q`
grows, every baryon mass is pushed up, and past some density the baryons
dissolve — the Mott (deconfinement) transition. `B = 0` switches that mechanism
off, and the two shipped values `B = 0` and `1 GeV/fm^3` bracket the study. In
vacuum (`alpha_S(0) = 0.849`, `n_B^Q = 0`) (baryonmass) gives `M_N = 938.89`
and `M_Lambda = 1113.68` MeV, against the published 938.9 and 1113.7.

### Vector fields

Only the products `g_omega omega_0` and `g_rho rho_0` enter, and each is its
coupling times its source,

    J_omega = sum_i f_i N_i n_i
            = 3 (n_p + n_n + f_Lambda n_Lambda)
              + f_q (n_u + n_d + n_s),      N_i = sum_q N^q_i    (Jomega)
    J_rho   = sum_i f_i tau_i n_i
            = (n_p - n_n) + f_q (n_u - n_d)                        (Jrho)
    g_omega omega = Gamma_omega(n_B) J_omega
    g_rho rho     = Gamma_rho(n_B) J_rho                         (fields)

`N_i = 3` for every baryon and `N_q = 1` for every quark: the omega couples to
quark number, and a baryon couples three times as strongly because it contains
three quarks. Leptons do not appear, `f_e = f_mu = 0`.

### Rearrangement

Because `alpha_S`, `Gamma_omega` and `Gamma_rho` depend on `n_B`, the variation
of the energy density with respect to a density picks up terms from the
couplings themselves. These are the rearrangement self-energies, and they enter
the chemical potentials and hence the pressure, **never** the energy density:

    Sigma^R_b = (1/2) Gamma_omega'(n_B) J_omega^2
              + (1/2) Gamma_rho'(n_B) J_rho^2
              + alpha_S'(n_B) sum_{i=p,n,Lambda}
                [ sum_q N^q_i (M_q - m_q0) ] n^s_i              (SigmaRb)
    Sigma^R_q = (1/3) B sum_{i=p,n,Lambda} n^s_i
              + (1/3) Sigma^R_b                                 (SigmaRq)

The asymmetry of (SigmaRq) is worth reading twice: the Pauli-blocking term
`B n_B^Q` of (baryonmass) raises *baryon* masses, but `n_B^Q` is a *quark*
density, so differentiating with respect to a quark density acts back on the
quark potential — and the factor 1/3 is `d n_B^Q/d n_q`. The second term of
(SigmaRq) is `(1/3) Sigma^R_b` for the same reason: a quark carries one third
of a baryon's worth of `n_B`, which is what the density-dependent couplings
respond to.

That `Sigma^R` is placed correctly is not asserted — it is measured. The
statement that fixes it is the thermodynamic definition of the chemical
potential,

    mu_i = ( d eps / d n_i )_{n_{j != i}}                          (HVH)

which holds to 1.8e-6 MeV — the floor of the central difference — over
baryonic, hyperonic, mixed and quark-dominated compositions at
`n_B = 0.16-3 fm^-3` and three parameter sets, on potentials of order 1e3 MeV.
Drop either rearrangement term and (HVH) fails by tens of MeV.

**The one exception, and it is the cap.** (HVH) holds because the gap equation
makes `eps` stationary with respect to each `M_q`. Where the cap of (nbar)
binds, `nbar^s_q` stops responding to the densities and that stationarity is
lost in the capped channel. It costs nothing in either of the two symmetric
situations — with no light flavour capped nothing is clamped, and with *both*
`u` and `d` capped the determinant term vanishes in both light channels,
`M_u = M_d = m_q0` exactly, and the state sits in a flat region — so the only
states affected are those with exactly one light flavour at the cap. One such
state is reached in the checks: `f_q = 0.5`, `B = 0` at `n_B = 0.8 fm^-3`,
where (HVH) misses by 6.9e-2 MeV on `mu_Lambda = 1419` MeV, i.e. 4.8e-5
relative. That is below the 0.05-0.20 MeV at which the engine is validated
against the author's own tables, so it does not show up there; it is recorded
because it is a property of the cap rather than of the arithmetic, and removing
it means changing how (nbar) is regularized, which is a physics decision and
not a refactor.

### Chemical potentials

With `nu_i = sqrt(k_{F,i}^2 + M_i^2)` the kinetic (effective) potential,

    mu_i = nu_i + f_i ( 3 g_omega omega + tau_i g_rho rho ) + Sigma^R_b
                                              i = p, n, Lambda     (mub)
    mu_q = nu_q + f_q (   g_omega omega + tau_q g_rho rho ) + Sigma^R_q
                                              q = u, d, s          (muq)
    mu_l = nu_l = sqrt(k_{F,l}^2 + m_l^2)     l = e, mu             (mul)

The factor 3 on `g_omega omega` in (mub) and its absence in (muq) is the same
three-valence-quark counting as in (Jomega); the rho term carries no such
factor because `tau_i` already distinguishes the two nucleons.


## Energy density, pressure, entropy

The energy density sums the single-species kinetic terms — with the cut-off
subtraction on the quarks and without it on baryons and leptons — and adds the
condensate and vector terms of the interaction:

    eps = sum_i eps(nu_i, M_i, g_i, Lambda_i, T)
          + 2 G_S sum_q (nbar^s_q)^2
          - 4 K nbar^s_u nbar^s_d nbar^s_s
          + (1/2) Gamma_omega J_omega^2
          + (1/2) Gamma_rho J_rho^2
          + eps_gamma + eps_nu
          - E0                                                     (eps)

with `Lambda_i = Lambda` for `i` in {u,d,s} and `Lambda_i = 0` otherwise, and
`eps_gamma`, `eps_nu` the optional thermal sectors below, zero unless asked for
and identically zero at `T = 0`. **No rearrangement term appears in (eps)**,
and that is the invariant a density-dependent mean field lives or dies by:
`Sigma^R` is the derivative of the couplings, so it belongs to `mu` and to `P`,
and putting it in `eps` would count it twice (CLAUDE.md §8; checked by
`verify/`).

The entropy density is the sum of the single-species entropies of (sT), and of
the thermal sectors where they are on:

    s = sum_i s(nu_i, M_i, g_i, T) + s_gamma + s_nu                (stot)

**and nothing else in the model carries entropy.** The Dirac-sea subtraction is
a vacuum term, and the condensate and vector terms of (eps) are functions of
the densities alone, so neither has a temperature derivative at fixed
composition. At `T = 0` every term of (stot) is exactly zero, by (s) — not to a
tolerance, but identically, which is what keeps the frozen `T = 0` numbers of
CLAUDE.md §12 bit-for-bit across the finite-temperature extension.

The pressure is the Euler relation,

    P = T s + sum_i mu_i n_i - eps                                    (P)

and it is how the code computes `P` — not by summing (PmedT) over species. The
two agree, but only once the interaction and rearrangement contributions to the
pressure are added by hand, and (P) adds them automatically through the
`Sigma^R` inside every `mu_i`. It also hands the `mu = 0` thermal sectors their
own pressure without their appearing in it, since `eps_X + P_X = T s_X` for
those. The free energy density is `f = eps - T s = -P + sum_i mu_i n_i`, and
reduces to `f = eps` at `T = 0`.

### The thermal sectors

Two sectors carry no conserved charge, so they enter `eps`, `P` and `s` and *no
equation of the solve*. Both are off by default.

`photons`: a blackbody gas at `mu = 0`, `g = 2`,

    P_gamma   = (pi^2/45) T^4/(hc)^3
    eps_gamma = 3 P_gamma
    s_gamma   = (4 pi^2/45) T^3/(hc)^3                            (photon)

`thermal_neutrinos`: the neutrino flavours the mode does *not* track, each as a
massless `mu = 0` gas with `g = 1` and its antiparticle, so that per flavour

    eps_nu,1 = (7/8) (pi^2/15) T^4/(hc)^3
    P_nu,1   = eps_nu,1 / 3
    s_nu,1   = (4/3) eps_nu,1 / T                              (thermalnu)

and `eps_nu = N_nu eps_nu,1`. **The count `N_nu` is a property of the mode.**
This model carries the electron neutrino in the composition, and only where a
mode holds `Y_Le`; the muon family is transparent here
(`mu_mu = mu_e - mu_nue`) and the tau family is never carried. So `N_nu = 3` in
`beta_eq_neutrinoless` and in the fixed-fraction modes, and `N_nu = 2` in
`beta_eq_neutrino_trapped`, where `nu_e` is already an unknown.

### The vacuum constant E0

Equation (eps) without `E0` does not vanish in vacuum: the cut-off quark sea
carries a large negative energy. `E0` is (eps) evaluated at zero density, so
that `eps(vacuum) = 0`,

    E0 = sum_q eps(0, M_q^vac, 6, Lambda)
         + 2 G_S sum_q (n^s,vac_q)^2
         - 4 K prod_q n^s,vac_q                                    (eps0)

with `M_q^vac` the solution of (nbar)-(gap) at zero density (no baryons, so no
cluster term). It is a property of the vacuum and depends only on
`(Lambda, m_q0, G_S, K)` — not on `f_q`, not on `B`, not on density, and not on
temperature. Measured: `M_u^vac = M_d^vac = 367.6483` MeV,
`M_s^vac = 549.4792` MeV, and

    E0 = -4263.8455 MeV/fm^3

That all five of the author's tables, spanning three values of `f_q` and two of
`B`, return the same constant to 8e-7 relative is itself a check on (eps0).


## Conserved charges

The conserved-charge densities are the species sums of the quantum numbers
above, with leptons excluded from all three (`eos.general.basis`):

    n_B = n_p + n_n + n_Lambda + (n_u + n_d + n_s)/3                 (nB)
    n_C = n_p + (2/3) n_u - (1/3)(n_d + n_s)                        (nCdef)
    n_S = n_Lambda + n_s                                            (nSdef)

and `Y_C = n_C/n_B`, `Y_S = n_S/n_B`. Equation (nSdef) carries the repository's
sign, `S = +1` per `s` quark, so `Y_S >= 0` in strange matter here where the
PDG convention would give `Y_S <= 0`.

The reference does not introduce `mu_C` and `mu_S`: it writes the equilibrium
condition directly in the physical charge `q_i` and the electron potential,
(beta) below. The translation is exact and worth stating, because it is what an
adapter into a composite engine would use. Applying
`mu_i = B_i mu_B + C_i mu_C + S_i mu_S` to `p` and `n` gives
`mu_C = mu_p - mu_n`, and comparing with (beta) gives

    mu_B = mu_b,    mu_C = -mu_e,    mu_S = 0                     (basis)

So this repository's beta-equilibrium condition `mu_C + mu_e = 0` is (beta)
read in the conserved-charge basis, and `mu_S = 0` is the statement that
strangeness is not conserved on the timescale of weak equilibrium —
`mu_Lambda = mu_n` and `mu_s = mu_d` are both consequences of it, not extra
assumptions. Equivalently, in the deconfined regime the inverse map
`mu_B = mu_u + 2 mu_d` holds; that identity is what the author's tables print
in their `munr` column, and it is the quantity a phase construction must equate
across a boundary.


## The closure, and the four modes

**Fixed composition: not a mode.** The lowest-level entry point,
`thermodynamics.thermo_from_n`, is given *all eight* species densities and
returns the state. It solves (nbar)-(gap) for `(M_u, M_d, M_s)` — the model's
internal self-consistency — and evaluates everything above. Nothing about the
composition is determined by it, so it is not one of the repository's modes and
is not named like one: it is the "block at given densities" of the shared
vocabulary, the counterpart of `thermo_from_mu` in the models that solve at
fixed potentials. Figures 1-3 of the paper — constituent masses in symmetric
matter, `E/A` of symmetric and pure neutron matter, the `Lambda` potential
depth — are all evaluated this way, and they carry no branch ambiguity because
the composition is imposed.

**The four modes, as one declaration.** All four modes of this repository are
closed here, at any `T >= 0`, with `s/n_B` accepted in place of `T` throughout.
And they are not four solvers: a mode is a *declaration*
(`eos.general.modes.ModeSpec`), one binary choice per conserved charge, and the
residual assembly reads it. For each charge, either its fraction is imposed and
its conjugate potential is an unknown, or its potential is set by an
equilibrium relation and the fraction comes out.

| mode | independent variables | extra unknowns | the charge row |
|---|---|---|---|
| `beta_eq_neutrinoless` | `(n_B, T)` | — | `sum_i q_i n_i = 0` |
| `beta_eq_neutrino_trapped` | `(n_B, Y_Le, T)` | `mu_nue` | `sum_i q_i n_i = 0` |
| `fixed_YC` | `(n_B, Y_C, T)` | — | `n_C = Y_C n_B` |
| `fixed_YC_YS` | `(n_B, Y_C, Y_S, T)` | `mu_S` | `n_C = Y_C n_B` |

The trapped mode adds the row `(n_e + n_nue)/n_B = Y_Le`, and `fixed_YC_YS` the
row `n_S = Y_S n_B`. In the two fixed-fraction modes the `leptons` flag decides
whether the neutralizing electrons and muons are added; either way they enter
*no equation*, because `n_C` is already pinned by the charge row, so they are
computed afterwards from `n_e + n_mu = n_C` and contribute to `eps`, `P` and
`s` alone. With `leptons=False` the result is electrically charged strongly
interacting matter, which is what a mixed-phase construction needs for each
pure phase before imposing global neutrality.

Species potentials are the conserved-charge projection

    mu_i = B_i mu_B + C_i mu_C + S_i mu_S                     (projection)

over the six strongly interacting species, with the lepton potentials given by
the weak-equilibrium relations `mu_C + mu_e = mu_nue` and
`mu_mu = mu_e - mu_nue` wherever `C` is equilibrated. In the neutrinoless mode
`mu_nue = 0` and `mu_S = 0`, and (projection) reduces to the paper's Eq. (23).

Weak equilibrium with free-streaming neutrinos makes every species potential a
projection of two numbers, the baryon potential `mu_b` and the electron
potential `mu_e`,

    mu_i = B_i mu_b - q_i mu_e                                     (beta)

for all eight species — `q_i` being the physical electric charge, leptons
included. Total electric neutrality closes the system,

    sum_i q_i n_i = 0                                           (neutral)

where the sum *does* run over leptons. Equations (beta) and (neutral) are the
conditions to keep apart: the first is equilibrium, the second is neutrality,
and `n_C` is the non-leptonic charge that neither of them fixes directly. The
muon family is carried, not merely declared: `mu-` populates from
`mu_e = m_mu = 105.66` MeV upward, which at the shipped parameters is between
`n_B = 0.1` and `0.2 fm^-3`, and by `n_B = 1.2 fm^-3` carries
`n_mu = 0.0204 fm^-3` against `n_e = 0.0338 fm^-3`.


## The residual, row by row

The unknown vector has ten components,

    x = ( M_u, M_d, M_s, mu_b, mu_C, n_B^Q,
          g_omega omega, g_rho rho, Sigma^R_b, Sigma^R_q )
        (+) ( mu_S )   iff Y_S is held
        (+) ( mu_nue ) iff Y_Le is held                        (unknowns)

ten always, plus one potential per held fraction. **`mu_C` is not in that
second group**: it is a potential of the matter whichever mode is asked, and
what the declaration changes is the row that closes it. The vector is longer
than the three-unknown gap solve of the fixed-composition entry point for one
reason: at fixed composition the fields and the rearrangement terms are *known*
from the densities, whereas here the densities are what is being solved for.
Carrying `g_omega omega`, `g_rho rho`, `Sigma^R_b` and `Sigma^R_q` as unknowns
with their defining equations as residual rows replaces a nested fixed-point
iteration by four more rows of one outer solve, and — the reason it is done
this way — guarantees that every quantity entering any residual comes from one
and the same state. `n_B^Q` is carried as an unknown for the same reason: it
enters the baryon masses through (baryonmass) *before* the quark densities that
define it are known.

Given `x` and the target density `n_B^target`, the state is built forwards: the
couplings `alpha_S`, `Gamma_omega`, `Gamma_rho` and their derivatives are
evaluated at `n_B^target`; the baryon masses from (baryonmass); then for every
species, in turn,

    nu_i = mu_i - Delta_i
    Delta_i = f_i (3 g_omega omega + tau_i g_rho rho) + Sigma^R_b
                                                    i = p, n, Lambda
            = f_q (  g_omega omega + tau_i g_rho rho) + Sigma^R_q
                                                    i = u, d, s
            = 0                                     i = e, mu     (forward)

with `mu_i` from (projection), and then `n_i`, `n^s_i`, `J_omega`, `J_rho` and
the two `Sigma^R` expressions. The residuals, in the order the code assembles
them, are the nine that every mode carries followed by the charge row and one
row per held fraction:

| row | residual | scale it is divided by | what it imposes |
|---|---|---|---|
| 1-3 | `M_q - [ m_q0 - 4 G_S nbar^s_q + 2 K nbar^s_q' nbar^s_q'' ]` | 100 MeV | the gap equation, (gap) |
| 4 | `sum_i B_i n_i - n_B^target` | `n_B^target` | baryon number, (nB) |
| 5 | `n_B^Q - (n_u + n_d + n_s)/3` | `n_B^target` | the quark baryon density, (baryonmass) |
| 6 | `g_omega omega - Gamma_omega(n_B^target) J_omega` | `3 Gamma_omega n_B^target` | the omega field, (fields) |
| 7 | `g_rho rho - Gamma_rho(n_B^target) J_rho` | `Gamma_rho n_B^target` | the rho field, (fields) |
| 8 | `Sigma^R_b - Sigma^R_b[x]` | 3000 MeV | baryon rearrangement, (SigmaRb) |
| 9 | `Sigma^R_q - Sigma^R_q[x]` | 1000 MeV | quark rearrangement, (SigmaRq) |
| 10 | `sum_i q_i n_i` **or** `n_C - Y_C n_B^target` | `n_B^target` | neutrality, (neutral), *or* the held `Y_C` |
| 11 | `n_S - Y_S n_B^target` | `n_B^target` | the held `Y_S` (`fixed_YC_YS` only) |
| 12 | `n_e + n_nue - Y_Le n_B^target` | `n_B^target` | the held `Y_Le` (trapped mode only) |

Rows 1-9 are present in every mode; rows 10-12 are what the declaration
selects, and the count of unknowns follows them exactly. The trapped neutrinos
are massless and left-handed, `g = 1`, so `n_nue = mu_nue^3/(6 pi^2)`.

Row 4 is what licenses evaluating the density-dependent couplings at
`n_B^target` rather than at the state's own baryon density: at the solution the
two are equal, and using the target makes `alpha_S`, `Gamma_omega`,
`Gamma_rho` constants of the residual rather than functions of the unknowns,
which removes their derivatives from the Jacobian. The third column is the
scale each row is divided by before the convergence gate is applied: the rows
carry mixed units — MeV, MeV^3 — and a norm of the raw vector would be
dominated by whichever row happens to be largest (`eos.general.solve`). Once
solved, the composition is handed back to `thermo_from_n`, seeded with the
constituent masses just found, so the reported state is produced by exactly one
code path whichever entry point was called.

**What temperature changes in the residual: no row, and one map.** *Not a
single row of the table above changes at `T > 0`*, and neither does the unknown
vector. The reason is that the residual is written in the *potentials*: rows
1-3 equate constituent masses, rows 4-9 equate fields and rearrangement terms
to their definitions, and the last rows equate charge densities to what the
mode demands. Every one of those statements is temperature-independent as an
*equation*.

What changes is the map from `nu_i` to the densities that fill those rows. At
`T = 0`, (forward) continues `k_{F,i} = sqrt(nu_i^2 - M_i^2)` for `nu_i > M_i`
and 0 otherwise, and then `n_i` and `n^s_i` come from the closed forms
(nmed)-(nsmed). At `T > 0` there is no `k_{F,i}`: `n_i` and `n^s_i` come from
(nmedT)-(nsmedT) evaluated at `nu_i` directly, and a `nu_i` below `M_i` is a
thermally populated state rather than an empty one. This is the *forward*
direction, potentials to densities, so it costs no more than the integral
itself.

**The one direction that inverts, and where it sits.** `thermo_from_n` goes the
other way: the densities are given and the `nu_i` they correspond to are
wanted. At `T = 0` that is algebra, `k_{F,i}` from (nmed) inverted and then
`nu_i = sqrt(k_{F,i}^2 + M_i^2)`. At `T > 0` it is a genuine numerical
inversion of (nmedT), done by the shared
`eos.general.fermi_integrals.invert_fermi_density`.

Where that inversion sits is the structural part. `k_{F,i}` depends only on
`(n_i, g_i)`, so at `T = 0` it is computed once, *outside* the gap iteration.
Its `T > 0` counterpart `nu_i` depends on the *mass*, and the three constituent
masses are exactly what the gap equation solves for — so all six strongly
interacting species must be re-inverted at every gap iteration. The two leptons
have fixed masses and stay outside it. Measured, the inversion costs about
30 us against 0.1 us for the closed form, so a `thermo_from_n` call at `T > 0`
costs of order `6 x 20 x 30 us ~ 4 ms` where the `T = 0` call costs nothing
extra. That asymmetry — forward free, inverse not — is why the phase-adapter
surface below, which a mixed-phase construction consumes, is the forward one.

**Entropy per baryon in place of temperature.** Wherever a temperature is
accepted, `s/n_B` is accepted in its place. It is an outer one-dimensional
solve for `T`: an isentrope is an isotherm whose temperature is found, per
density, from `s(n_B,T)/n_B = (s/n_B)^target`, which is monotone increasing in
`T` at fixed density and so a well-posed bracketed root. Every evaluation of
that bracket is a full solve of the residual above. `s/n_B = 0` is answered
directly at `T = 0`: it is the only entropy the exact `T = 0` branch reaches,
and no positive bracket contains it.

**The block at given potentials.** One further entry point is neither a mode
nor a fixed composition: `thermo_from_mu(mu_B, mu_C, mu_S, T)`, which solves
the model's own self-consistency — the gap equation, the vector fields, the
rearrangement terms *and the phase's own baryon density* — and returns the
block. No leptons, no neutrality, no held fraction: those are conditions on a
*system*, and this describes matter. It is the surface this repository's
phase-adapter contract is written in, and a mixed-phase construction consumes
exactly this, once per phase.

Nine unknowns — `M_u`, `M_d`, `M_s`, `n_B`, `n_B^Q`, `g_omega omega`,
`g_rho rho`, `Sigma^R_b`, `Sigma^R_q` — against the nine rows above with the
baryon-number row read the other way round: the mode solve imposes `n_B` and
finds `mu_B`; this imposes `mu_B` and finds `n_B`. Which *branch* is returned
is decided by the starting point, since above a first-order transition the same
potentials admit more than one solution — so a caller that differentiates this
function numerically must make its seed a deterministic function of the
arguments, or the returned Jacobian is not the derivative of anything.

**Temperature.** Every mode is closed at any `T >= 0`, and `s/n_B` is accepted
in place of `T` throughout; a negative `T` raises. What still refuses a
temperature is the *construction* below, not the model. **Species.** The
composition is fixed by the model: `hyperons` and `muons` are on and fixed; the
octet beyond `Lambda`, the `Delta` quartet and a thermal meson gas are absent,
and switching one on raises rather than being silently ignored. `photons` and
`thermal_neutrinos` are the two the caller *does* choose — implemented,
(photon)-(thermalnu), off by default and identically zero at `T = 0`. A thermal
meson gas is not merely unimplemented but inapplicable: sigma, omega and rho
here are auxiliary fields eliminated in favour of `g^2/m^2`, and
(Gomega)-(Grho) leave the bare masses undetermined for exactly that reason.


## What a solved point returns

`thermo_from_n` returns an `EoSPoint`, in natural units (densities in MeV^3,
masses and potentials in MeV, `eps` and `P` in MeV^4) with fm-based properties
for the public boundary. Every field, and where it comes from:

| field | symbol | from |
|---|---|---|
| `n` | `n_i`, all eight species | the request |
| `M_q` | `M_u, M_d, M_s` | the gap equation, (gap) |
| `M_b` | `M_p, M_n, M_Lambda` | (baryonmass) |
| `nu` | `nu_i`, all eight — the PRIMARY kinetic quantity | (forward) |
| `kF` | `k_{F,i}`, the `T = 0` derived view of `nu_i` | `sqrt(nu_i^2 - M_i^2)` |
| `n_s` | `n^s_i`, all six strongly interacting species | (nsmed), (nsmedT), (nsvac) |
| `nbar_s` | `nbar^s_q` | (nbar) |
| `alpha_S`, `Gw`, `Gr` | `alpha_S, Gamma_omega, Gamma_rho` | (alphaS)-(Grho) at `n_B` |
| `J_omega`, `J_rho` | `J_omega, J_rho` | (Jomega)-(Jrho) |
| `gomega_omega`, `grho_rho` | `g_omega omega, g_rho rho` | (fields) |
| `SigmaR_b`, `SigmaR_q` | `Sigma^R_b, Sigma^R_q` | (SigmaRb)-(SigmaRq) |
| `mu` | `mu_i`, all eight | (mub)-(mul) |
| `eps`, `P` | `eps`, `P` | (eps), (P) |
| `s` | `s`, zero at `T = 0` | (stot) |
| `T` | the temperature it was solved at | — |
| `n_b`, `n_bQ` | `n_B`, `n_B^Q` | (nB), (baryonmass) |
| `n_C`, `n_S` | `n_C`, `n_S` | (nCdef)-(nSdef) |
| `EperB` | `eps/n_B - 938.9` MeV | the paper's Fig. 2 ordinate |

A table row (`table.beta_row`) is that state flattened and made fm-based:
`n_B`, `T`, `P`, `eps`, `s`, `s/n_B`, the four potentials `mu_B`, `mu_C`,
`mu_S`, `mu_e`, the fractions `Y_C` and `Y_S`, the eight species densities, the
three constituent and three baryon masses, and

    chi = n_B^Q / n_B                                               (chi)

the fraction of the baryon density carried by deconfined quarks — the `fq`
column of the author's tables. It plays here the part the quark volume fraction
plays in a two-engine mixed phase, and carries the same name for that reason,
but it is not a volume fraction: the two phases of this model occupy the same
volume.

**The scalar density, and why it is not the trace identity.** `n^s_i` is
returned for all six strongly interacting species and is *integrated*,
(nsmed), (nsmedT) and (nsvac), not obtained from the trace of the
energy-momentum tensor. The trace form

    eps^med - 3 P^med = M n^s,med                                 (trace)

does hold here, identically and per species, for the *medium* terms: it follows
from (epsmed)-(Pmed) and (nsmed) term by term, and is verified to round-off for
a baryon and for a quark with the cut-off switched off. It fails for the quarks
as they are actually used, and for the totals, and for two separate reasons.
The Dirac-sea subtraction removes `eps^vac` from `eps` and `n^s,vac` from `n^s`
but nothing from `P`, and `eps^vac != M n^s,vac`; and the total `eps` and `P`
of (eps) and (P) carry condensate, vector, rearrangement and thermal terms that
(trace) knows nothing about. So `n_s = (eps - 3P)/m*` is not a route to `n^s`
in this model and is not used as one; where a reader of another model's
document meets that identity, this is where it stops.

A mode solve returns a `BetaPoint`, which carries the same record plus the
density it was asked at, the mode declaration it was asked under, the
conserved-charge potentials `mu_B`, `mu_C` and `mu_S` it solved for, the lepton
potential `mu_e`, and the converged unknown vector `x` — which is what
warm-starts the next density of a sweep, and what makes the continuation follow
one branch.

`eos_response` is **not implemented** for this model and raises naming both
reasons. The second-derivative quantities of the CompOSE list divide in two:
the heat capacities, the thermal index and the isothermal/adiabatic distinction
need `T > 0`, and `c_s^2` and the susceptibilities `chi_ab = dn_a/dmu_b` need a
statement about which branch the derivative is taken along, which the branch
structure below has not yet settled — above the model's first first-order
transition more than one branch satisfies the equilibrium conditions at the
same density, and differentiating along the branch a continuation happened to
reach would return a number whose meaning depends on the direction the table
was swept in. (The raise message states the first half as "this is a `T = 0`
model", which the modes are not: they are closed at any `T >= 0`. What is
`T = 0`-only is the construction, and the response gap is the branch statement,
not the temperature.)


## The construction: from branches to a delivered EoS

A table swept by `build_table` is a *continuation*: it follows one branch and
keeps following it past a first-order transition, into the metastable region
beyond, so it may violate `dP/dn_B >= 0`. That is real physics — mechanical
instability inside a coexistence region — and mapping it is what a branch sweep
is for. What a structure solver must be handed is the other object, the stable
equation of state, and `build_constructed_table` assembles it.

**The coexistence conditions.** Two branches `alpha` (low density) and `beta`
(high density) of the same functional coexist where they share a baryon
chemical potential and a pressure, each phase being separately charge-neutral
with its own lepton population — the `eta = 1`, or Maxwell, construction:

    mu_B^alpha = mu_B^beta = mu_B^co                            (coexmu)
    P^alpha + P_l^alpha = P^beta + P_l^beta = P^co               (coexP)
    n_C^gamma = n_e^gamma + n_mu^gamma,
    mu_C^gamma + mu_e^gamma = 0,   gamma in {alpha, beta}  (coexneutral)

The lepton pressure belongs in (coexP) because at `eta = 1` the leptons live
inside the structures whose pressures are equated; dropping it moves `mu_B^co`
by tens of MeV. The two phases carry *different* `mu_e`, and that difference is
precisely what `eta = 1` means: neither borrows charge from the other.
(coexneutral) is one equation in the one unknown `mu_C^gamma` per phase, solved
first; then (coexP) is one equation in the one remaining unknown `mu_B^co`.

The two edge densities follow from the solved endpoints,

    n_lo = n_B^alpha(mu_B^co),   n_hi = n_B^beta(mu_B^co)     (coexedges)

and between them no single phase is stable.

**The plateau.** For `n_lo <= n_B <= n_hi` the system separates into the two
phases at fixed `mu_B^co` and `P^co`. With `f` the volume fraction occupied by
the high-density phase, baryon number gives the lever rule

    n_B = (1-f) n_lo + f n_hi   =>   f = (n_B - n_lo)/(n_hi - n_lo)
                                                                (lever)

and every density averages with the same weights,

    n_i = (1-f) n_i^alpha + f n_i^beta
    eps = (1-f) eps^alpha + f eps^beta
    P   = P^co                                                (plateau)

with the entropy density levering like any other density. The conserved
fractions are formed from the *averaged* densities, `Y_C = n_C/n_B` and
`Y_S = n_S/n_B`, not by averaging the two phases' own fractions, which are
relative to their own `n_B`. The effective masses `M_q` and `M_i` are
properties of one phase and have no value in the mixture; the implementation
returns them as `nan` across the plateau rather than averaging them into a
state that exists nowhere. `mu_C` and `mu_e` are returned as `nan` for the same
reason — at `eta = 1` the two phases genuinely differ in them.

**The energy density needs no further solve.** On a neutral phase at `T = 0`
the Euler relation collapses. Writing `sum_i mu_i n_i` over matter and leptons
and using (coexneutral),

    sum_i mu_i n_i = mu_B n_B + mu_C n_C + mu_S n_S + mu_e (n_e + n_mu)
                   = mu_B n_B + mu_S n_S                  (eulerneutral)

because `mu_C n_C + mu_e (n_e + n_mu) = -mu_e n_C + mu_e n_C = 0`. In beta
equilibrium without strangeness conservation `mu_S = 0`, so

    eps = mu_B n_B - P                                     (epsneutral)

Applied at `mu_B^co` and `P^co` this makes `eps` *linear* in `n_B` across the
plateau automatically, which is what (plateau) asserts — the two statements
agree identically, and the slope of the plateau in the `(n_B, eps)` plane *is*
`mu_B^co`. It also means `c_s^2 = dP/deps = 0` there exactly.

**Which branch outside a window.** Outside every window the delivered point is
the branch that minimises the energy density at that `n_B`: at `T = 0` and
fixed `n_B`, in beta equilibrium with neutrality, the stable state is the one
of lowest `eps`. Both continuations are swept, upward from the low-density
chirally broken side and downward from the top of the grid, and the comparison
is made point by point. No branch bookkeeping enters: where only one branch
exists the comparison is trivial, and where both do it is the physical
criterion.

**Where this lives.** Locating a transition needs both branches at once, which
is a two-phase problem, so it lives in the composite engine
(`eos.mixed.construction.enjl_coexistences`, over
`eos.mixed.boundaries.locate_maxwell`) rather than in the model: a model does
not import a composite engine. The located windows are handed to
`eos.enjl.table.build_constructed_table` as an argument, and the assembly of
(lever)-(epsneutral) is pure ENJL. A parameter set with three branches admits
three pairings and the driver sweeps the declared list of them, merging what it
finds; which pairings are realised is a property of the parameters, not of the
engine.

Only `eta = 1` is delivered. At `0 < eta < 1` both a local and a global lepton
population exist, with weights `eta` and `1 - eta`, and the plateau of
(plateau) is replaced by a solved mixed system at every density; `f(eta)` is
measured to fall monotonically from `eta = 1` to `eta = 0`, by 3e-2 MeV/fm^3 at
the `f_q = 0.7`, `B = 1` chiral transition, with no interior extremum.


## Numerics

**The gap solve.** Three unknowns `(M_u, M_d, M_s)`, Powell's hybrid method,
accepted on the residual of (gap) at 1e-12 MeV, which on masses of 5-550 MeV is
1e-14 relative or better. The solver's own success flag is *not* the gate: at
this tolerance it routinely reports "not making good progress" on a converged
root, because it is being asked for more precision than the residual can
supply. The residual decides. The cold start is the vacuum solution
`(367.6, 367.6, 549.5)` MeV, which is a poor guess once the light condensates
have collapsed — a fixed-composition request in the chirally restored region
can fail from it and must be seeded from a neighbouring point.

**The beta-equilibrium solve.** Ten unknowns, (unknowns), bounded least squares
(`scipy.optimize.least_squares`) on the scaled residuals above, accepted at the
repository's common gate `eos.general.solve.RESIDUAL_TOL = 1e-10`. The bounds
are not decoration: at the shipped parameters `mu_b = 6.24` GeV and
`g_omega omega = 1.68` GeV already at `n_B = 3.8 fm^-3`, and both keep growing
roughly linearly, so a box calibrated at saturation would exclude the solution
entirely above a few times `n_sat`; it widens with density instead, while the
constituent masses have a genuine density-independent ceiling (their vacuum
values) and `n_B^Q` a genuine one (`n_B` itself). Starting points are tried in
order of decreasing plausibility: the previous point of a sweep first, then the
same state with the light condensates switched off — which is where the
chirally restored branch sits, and is what carries a sweep *across* a chiral
transition, since a first-order transition puts the next root several hundred
MeV away in the quark masses. Two parameter-free cold starts, one nucleonic and
one deconfined, are available only when there is no previous point. On the
shipped grid the achieved scaled residual is at most 8.8e-13, median 4.0e-15.

**Continuation, and the open branch question.** A table is a continuation, not
a phase diagram. Each point is warm-started from its neighbour, so the sequence
follows one branch of the model and keeps following it past a first-order
transition into the metastable region beyond. Cold starts are allowed only
until the branch is established; permitting them mid-sweep lets the sequence
hop between branches from one density to the next, which shows up as an
equation of state that oscillates rather than one that has a transition in it.
The direction of the sweep therefore selects the branch: `direction="up"` from
the low-density chirally broken side, or `"down"` from a deconfined guess at
the top of the grid. Where only one branch exists the two agree; where several
do, they differ, and *that difference is the branch structure*.

Choosing between branches is a Maxwell construction — at fixed `mu_b`, take the
branch of larger `P` — and it cannot be done from a single sweep, because it
needs both branches at once. Three transitions can each be first order or
continuous depending on `(f_q, B)`: the *quarkyonic* onset (McLerran and
Pisarski, NPA 796 (2007) 83), where quasi-free quarks appear alongside baryons;
the *chiral* transition, where `nbar^s_q -> 0` and `M_q -> m_q0`; and
*deconfinement*, where (baryonmass)'s Pauli-blocking term unbinds the baryons.
Two of the author's own tables retain a step with `dP/dn_B < 0` rather than the
coexistence plateau that would replace it, which is the same statement seen
from the other side: a raw branch may violate mechanical stability, and a
construction — not the branch map — is what resolves it before a table reaches
a structure solver (CLAUDE.md §8). The baryon potential to equate across such a
boundary is `mu_B` of (basis), which in the deconfined phase is
`mu_u + 2 mu_d` and *not* the vanishing neutron's own `mu_n`.

**Where finite temperature entered, and where it did not.** Nowhere except the
single-species integrals, as the structure of the model predicted. The
couplings are functions of `n_B`; the gap equation is the same equation with
`T`-dependent scalar densities; the rearrangement terms have the same
expressions; `E0` is the same number; and *not one row* of the residual table
changed, because those rows are written in the potentials. Only (nmed)-(s)
became (nmedT)-(sT) — and the vacuum terms (nsvac)-(epsvac) did not change even
there, being `T`-independent by construction. The pressure is the full Euler
form (P).

**What the construction still refuses.** The branch *map* — a warm-started
continuation — runs at any temperature. Replacing a first-order window by its
plateau does not. Locating a coexistence at `T > 0` means equating the Gibbs
free energies of the two branches rather than `P` and `mu_B` alone, so the
entropy enters the coexistence bookkeeping, and the lever rule of (plateau)
then averages it across the window as well. Both
`eos.enjl.table.build_constructed_table` and `eos.mixed.construction` raise
above `T = 0` and say so.


## What the implementation reproduces, measured

**Against the published numbers.** Symmetric nuclear matter is solved at fixed
composition and the saturation properties read off by central differences; the
`Lambda` potential is `U_Lambda = mu_Lambda - M_Lambda^vac` at vanishing
`Lambda` density.

| quantity | this implementation | Xia (2024) | |
|---|---|---|---|
| `n_sat` | 0.158297 fm^-3 | 0.158 | Table II |
| `E/A` at `n_sat` | -16.010 MeV | -16.0 | Table II |
| `K_sat` | 234.20 MeV | 234.5 | Table II |
| `E_sym(n_sat)` | 31.549 MeV | 31.5 | Table II |
| `L_sym` | 42.35 MeV | 42.4 | Table II |
| `E_sym(0.1)` | 25.500 MeV | 25.5 | Sec. II |
| `E/n_B` at 0.1 | 924.79 MeV | 924.8 | Sec. II |
| `E/n_B` at 0.158 | 922.89 MeV | 922.9 | Sec. II |
| `M_N` at 0.16 | 519.23 MeV | 519.2 | Sec. II |
| `M_N`, `M_Lambda` in vacuum | 938.89, 1113.68 MeV | 938.9, 1113.7 | Sec. II |
| `M_u = M_d`, `M_s` in vacuum | 367.648, 549.479 MeV | 367.6, 549.5 | Sec. II |
| `alpha_S(0)`, `alpha_S(n_S)` | 0.849000, 0.570000 | 0.849, 0.57000 | Table I |
| `U_Lambda(n_sat)` | -30.020 MeV | -30 (fitted) | Sec. II |

**Against the author's own tables.** Five beta-equilibrium tables produced by
the author's Maple worksheet, at
`(f_q, B) in {(1.0,0), (1.0,1), (0.7,0), (0.7,1), (0.5,1)}` and
`n_B = 0.01-10 fm^-3`, are the tightest available constraint: they come from
the code that made the paper's figures and so pin the model far beyond the two
or three significant figures the paper prints. Reading them correctly requires
three things that are not in their column headers, and each of them costs
hundreds of MeV if missed. The scalar-density column named `nsq` is the
*medium* part of (nsvac), while the column named `Sigmaq` is the full
`nbar^s_q` of (nbar) — so the gap equation must be fed `Sigmaq`. The baryon
potential is the column `munr`, not `mun`: once baryons dissolve the two part
company, `munr` continuing as `mu_u + 2 mu_d`, (basis). And (beta) holds only
for species that are *present*; a species below threshold has a stale printed
potential, and `mu_e` must be recovered as `mu_d - mu_u` where no electrons
remain. Read that way, and with the interpolated mixed-phase rows of the
`f_q = 0.5` file and nine non-converged rows of the `f_q = 0.7`, `B = 0` file
excluded, the engine reproduces the tables' own identities — the gap equation
to 2e-5 relative, (P) to 1e-8 - 1e-4 relative depending on the file, and the
mean-field rebuild of every printed `mu_i` to between 0.005 and 0.20 MeV on
potentials of 1000-2500 MeV. These comparisons and the per-file tolerances they
need live in `test/enjl`.

**Against `eos/general`.** The free-gas parts of (nmed)-(Pmed) are the shared
`T = 0` Fermi integrals of `eos.general.fermi_integrals`, and the model calls
them rather than carrying a second copy. Before that routing was made, the
local closed forms and the shared ones were compared over `g` in {2, 6}, `M` in
{5.5, 140.7, 200, 300, 367.6, 500, 549.5, 938.9, 939, 1115.7} MeV and `k_F` in
{1, 10, 50, 200, 500, 1200, 2200, 5000} MeV, at `Lambda = 0`. Worst relative
deviation by `k_F`:

| `k_F` [MeV] | `n` | `eps` | `n^s` | `P` |
|---|---|---|---|---|
| 1 | 3.5e-10 | 6.5e-8 | 2.6e-7 | 4.1e-1 |
| 10 | 1.8e-12 | 2.5e-11 | 9.3e-11 | 1.3e-6 |
| 50 | 1.4e-13 | 3.3e-13 | 1.2e-12 | 5.7e-10 |
| 200 | 8.5e-15 | 1.8e-14 | 3.1e-14 | 9.7e-13 |
| >= 500 | 1.4e-15 | 1.4e-15 | 3.3e-15 | 1.4e-14 |

The two are the same analytic expression, so the table is a floating-point
statement and not a physics one. It is quoted rather than asserted because the
`k_F = 1` MeV entry looks alarming and is not: there `x = k_F/M ~ 1e-3` and the
pressure is a cancellation of two nearly equal terms in both forms. At that
point the asymptotic `P -> g k_F^5/(30 pi^2 M)` gives 7.880e-13 MeV/fm^3, this
model's `nu n - eps` gives 7.875e-13 and the shared closed form gives
4.671e-13; the shared form is the one losing digits, and the species carrying
it has a density of 4e-9 fm^-3. Everywhere a species is populated enough to
matter, the agreement is 1e-12 or better, and routing the model through the
shared integrals moved no frozen quantity of `test/baseline` by more than
1.5e-12 relative, against a gate of 1e-10.

**Internal invariants.** `verify/run_full_check.py` asserts, over a grid of
compositions, densities, modes and parameter sets: the Euler relation and
`f = eps - Ts = -P + sum_i mu_i n_i`; `mu_i = deps/dn_i`, (HVH), which is the
statement that `Sigma^R` is in `mu` and `P` and not in `eps`, gated at 1e-3 MeV
where the functional is smooth and at 1e-1 MeV at the cap, whose exception is
named rather than absorbed; the gap equation satisfied at the returned state,
cap included; the charge densities (nB)-(nSdef) against `eos.general.basis`;
(beta) for every species that is present and (neutral) to 9e-13 MeV; that each
fixed-fraction mode returns the fraction it was asked for, to 7e-15; that
`fixed_YC_YS` at `Y_C = 0.5`, `Y_S = 0` with the leptons off IS nucleonic
symmetric matter wherever its own solution is quark-free, to 9e-12; that the
trapped mode hits `Y_Le` and satisfies `mu_C + mu_e = mu_nue`; that
`thermo_from_mu` returns, cold, the state whose potentials it was given, to
2e-13, together with the conserved-charge reading (basis); `0 <= c_s^2 <= 1` on
the stable stretch of a continuation, which is LOCATED by walking the sweep
rather than assumed, since a raw branch may legitimately violate
`dP/dn_B >= 0`; the vacuum masses and `E0`, including that `E0` is identical
across parameter sets; that each mode, temperature, species flag and response
function that is said to raise, does; and that a request the solver cannot
reach comes back as a status rather than an exception.


## Not implemented

Recorded in `docs/DEFERRED.md` with what closing each would take: the
CONSTRUCTION at `T > 0` — locating a coexistence there means equating Gibbs
free energies rather than `P` and `mu_B` alone, so the entropy enters the
coexistence bookkeeping and the plateau's lever rule; the construction at
`eta < 1`; `eos_response`; and the exception to (HVH) at the cap of (nbar).


## References

- Z. Xia, Phys. Rev. D 110, 014022 (2024) [arXiv:2405.02946] — the model.
- P. Rehberg, S. P. Klevansky and J. Hüfner, Phys. Rev. C 53, 410 (1996) — the
  RKH NJL parametrization.
- Y. Nambu and G. Jona-Lasinio, Phys. Rev. 122, 345 (1961) — the NJL model.
- G. 't Hooft, Phys. Rev. D 14, 3432 (1976) — the determinant interaction.
- L. McLerran and R. D. Pisarski, Nucl. Phys. A 796, 83 (2007) — quarkyonic
  matter.
- J. M. Lattimer et al. (Johns, Ellis, Lattimer), Astrophys. J. 473, 1020
  (1996) — the analytic Fermi integrals.
