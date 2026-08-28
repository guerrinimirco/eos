# NJL — three-flavour Nambu–Jona-Lasinio quark matter, with colour superconductivity

`njl.tex` is the same description written for LaTeX, with the bibliography;
this file carries the same physics in plain text. Either one alone is enough to
reproduce the model. The implementation specification both follow is
`docs/njl_csc_implementation.md`, which is the authority wherever it and a
document differ; where a document and the source differ, **the source decides**.

**Model.** A contact four-fermion theory of the light quarks: the scalar
channel that breaks chiral symmetry, the 't Hooft determinant that ties the
three flavours together, a colour-antitriplet diquark channel that condenses
into a colour superconductor, and a vector channel whose repulsion sets the
high-density stiffness. Rehberg, Klevansky & Hüfner, PRC 53, 410 (1996) for
the parameter set; Rüster et al., PRD 72, 034004 (2005) for the neutral
three-flavour pairing sector; Alford, Schmitt, Rajagopal & Schäfer, RMP 80,
1455 (2008) for the review.

    L = qbar(i d_slash - m_hat) q
      + G_S sum_a [(qbar tau_a q)^2 + (qbar i g5 tau_a q)^2]
      - K {det_f[qbar(1+g5)q] + det_f[qbar(1-g5)q]}
      + G_D sum_eta (qbar i g5 eps_eta lam_eta C qbar^t)(q^t C i g5 eps_eta lam_eta q)
      - G_V (qbar gamma^mu q)^2

Almost nothing is an input. The constituent masses come out of the gap
equation, the effective bag constant is a *derived* vacuum pressure difference,
and which colour-superconducting pattern the matter is in is an *outcome*
chosen by free energy, not a declaration.

The scalar, vector and diquark mean fields all arise from ONE
Hubbard–Stratonovich step; none is more fundamental. Two places where the
analogy nevertheless breaks matter for the implementation. First, the
condensation-energy normalisations differ by a factor two — the scalar cost is
`sum_f (M_f - m_f)^2/(8 G_S)` and the pairing cost `sum_eta Delta_eta^2/(4 G_D)`
— so `eta_D = G_D/G_S = 1` does NOT mean "equally strong channels". Second,
sigma enters the quasiparticle diagonal while Delta is strictly off-diagonal:
the gap matrix has identically zero diagonal, mixes particles with holes, and
therefore needs the doubled basis of the Bogoliubov–de Gennes problem below.
That is also why the gap kernel carries branch signs where the mass equation
does not.

**What is omitted.** The 't Hooft term expanded in the presence of diquark
condensates generates a cross-term proportional to `sum_alpha sigma_alpha
|Delta_alpha|^2`. Most treatments omit it; Baym et al., Rept. Prog. Phys. 81,
056902 (2018) include it with `K' ~ K`. It is NOT implemented here, so `eta_D`
must be read as an effective coupling that has absorbed it. Nothing here
invents a coefficient by analogy with the mass cross-terms.

## Conventions

Natural units inside the physics modules, `hbar = c = k_B = 1` with MeV
throughout: momenta, masses and potentials in MeV, densities in MeV^3, and
`Omega`, `P`, `eps` in MeV^4. Every public boundary is fm-based — `n` in fm^-3,
`T` and `mu` in MeV, `P` and `eps` in MeV/fm^3 — converted through
`(hc)^3` with `hc = 197.3269804` MeV fm.

Nine colour-flavour modes `j = (f, a)`, `f in (u, d, s)`, `a in (r, g, b)`,
indexed **flavour-major**, `j = 3 i_f + i_a`. The spin degeneracy of one mode
is `g = 2`; the Dirac sea of one flavour carries `g_sea = 2 N_c = 6`, since the
vacuum is not resolved by colour.

Charges `q_u = +2/3`, `q_d = q_s = -1/3`, and **strangeness S = +1 per s
quark**, the opposite of the PDG sign and the repository's convention
throughout. `C` is the charge of strongly-interacting matter ONLY; the leptons
are excluded from it and enter through the separate condition of total electric
neutrality.

Colour generators:

    (T_3)_(r,g,b) = (+1/2, -1/2, 0)
    (T_8)_(r,g,b) = (1/3, 1/3, -2/3) = lambda_8/sqrt(3)

**Three normalisations of T_8 are in circulation** and mixing them corrupts
`mu_8` by factors of 1.15 to 1.7. Rüster et al., Pagliara & Schaffner-Bielich
and Kunkel et al. use the halved Gell-Mann form
`sqrt(3) T_8 = diag(1/2, 1/2, -1)`, for which
`mu_8^theirs = (2/sqrt3) mu_8^ours = 1.1547 mu_8^ours`; Buballa and Steiner,
Reddy & Prakash use the full `lambda_8`, for which
`mu_8^ours = sqrt(3) mu_8^theirs`. As a worked consequence, the CFL result
`mu_8 = -(1/2 sqrt3) m_s^2/mu` of Steiner et al. reads `mu_8 = -(1/2) m_s^2/mu`
here: at `m_s = 300`, `mu = 450` MeV that is -57.7 MeV in their convention and
-100.0 MeV in ours.

The mode potentials, with all five conserved-charge potentials:

    mu_(f,a) = mu_B/3 + q_f mu_C + s_f mu_S + (T_3)_a mu_3 + (T_8)_a mu_8
    mu_e     = mu_nue - mu_C

the last of which is beta equilibrium in the sign convention
`mu_C = mu_p - mu_n` used across this repository.

**Two naming traps in the sources.** The `G_D` of Steiner, Reddy & Prakash is
the 't Hooft coupling, not the diquark one (theirs is `G_DIQ`); and `Delta_2`
in the Rüster/Buballa index convention means the *ud* gap, not the second
largest.

## Parameters

Every function takes the parameter set as its first argument and none reaches
for a default on the caller's behalf: the set is one frozen record, hashable
and safely shared between processes. Three tiers, and the tiers say which
numbers an inference run may move.

**Tier 1 — the vacuum fit, never sampled.** The Rehberg–Klevansky–Hüfner set,
fitted to `m_pi = 135.0`, `f_pi = 92.4`, `m_K = 497.7`, `m_eta' = 957.8` MeV:

    Lambda      = 602.3 MeV                   the fit as a whole
    G_S Lambda^2 = 1.835     -> G_S = 5.0584e-6  MeV^-2     m_pi, f_pi
    K Lambda^5   = 12.36     -> K   = 1.5594e-13 MeV^-5     m_eta'
    m_u = m_d   = 5.5 MeV                     m_pi
    m_s         = 140.7 MeV                   m_K

The *dimensionless* combinations are what is stored, because they are what the
fit determines; `G_S = 1.835/Lambda^2` and `K = 12.36/Lambda^5` are derived and
every equation below takes them in those units. Re-sampling one of these five
breaks the vacuum phenomenology the model is anchored to — and `(f_pi, phi)`
alone leave a two-root degeneracy, so `m_pi` and `m_K` must stay in any refit.
Every vacuum number in "What the implementation reproduces" is this set with no
further tuning.

**Tier 2 — structural, declared per run.** These change the equations, not a
number in them:

    vector form  "constant"     one of constant / power_law / gluon_exchange
    alpha        2/3            the power-law exponent, power_law only
    n_ref        0.48 fm^-3     that form's reference QUARK density, i.e. n_B = n_sat
    lambda       1              = Lambda_UV/Lambda; anything else RAISES

The lepton content is structural in the same sense and is carried by the
species flags: `csc` (the pairing sector — with it off there are no gaps, no
Bogoliubov–de Gennes problem, and `mu_3 = mu_8 = 0` identically), `muons` (on
by default), `thermal_neutrinos` and `photons` (both off). The three flags this
model does not have — `hyperons`, `deltas`, `thermal_mesons` — **raise** rather
than being ignored: there are no baryons here to be strange or resonant, and
the mesons of the Lagrangian are the auxiliary fields of the four-fermion
terms, eliminated in favour of `G_S`, `K` and `G_D`, so they carry no
independent thermal population at mean field.

**Tier 3 — the sampled vector.**

    eta_D = G_D/G_S    0.75      range 0.5-1.5       diquark strength; 0.75 is Fierz
    eta_V = G_V/G_S    0.0       range 0-1           vector repulsion, constant/power_law
    G_V0/G_S           0.5       range 0-1           vector repulsion, gluon_exchange
    M_g                500 MeV   range 400-800 MeV   the gluon mass of that form

`eta_D = 1` does not mean equally strong channels (see the factor two above),
and `eta_D` also has to absorb the omitted 't Hooft–diquark cross-term, so it
is an effective coupling and a paper using it should say so.

**Published sets.**

    rkh              nothing changed — the shipped default, and the set every
                     number below was produced at
    kunkel           eta_D = 1.45, eta_V = 0.7
    gluon_exchange   the gluon_exchange form, G_V0/G_S = 0.5, M_g = 500 MeV

The `kunkel` *couplings* are those of Kunkel, Rather et al.
[arXiv:2607.11537]; the regularization is not. Their calculation is
RG-consistent at `lambda ~ 10` and this one is at `lambda = 1`, because the
counterterm that makes `lambda > 1` finite is not implemented. The two are not
independent — RG-consistent gaps run almost 90% above sharp-cutoff ones — so
use it as a strong-coupling point, not as a reproduction of that paper.

**The constants the shared sectors carry**, not fitted here:
`m_e = 0.510999` MeV and `m_mu = 105.6584` MeV with `g_e = g_mu = 2`;
neutrinos massless with `g_nu = 1` per flavour, particles and antiparticles
summed; photons massless with `g_gamma = 2`.

**Three routes to a parameter set.** CLAUDE.md section 6 makes model
parameters arguments, so all three have to exist. *By name:*
`Parameters.default()` is the shipped RKH set, and `Parameters.named(name)`
takes any of the three published sets -- `'rkh'`, `'kunkel'`,
`'gluon_exchange'` -- an unknown name raising `KeyError` that lists them.
*A new set:* every field carries a default, so `Parameters(G_D=..., G_V=...)`
names only what changes; the dataclass is frozen, so `dataclasses.replace` is
how a set already in hand is modified. *From nuclear-matter parameters:* no
route, and none is missing -- NJL has no nuclear sector, so there is no
`nmp.py` and nothing to invert; its parameters are fixed by vacuum data
instead.

## The cut medium integrals

One mode of mass `M` at effective potential `mu*` and temperature `T`,
integrated to the cutoff, with `E = sqrt(k^2 + M^2)`,
`f^+- = [1 + exp((E -+ mu*)/T)]^-1` and `x_+- = E -+ mu*`:

    n     = (g/2 pi^2) int_0^Lambda dk k^2       (f^+ - f^-)
    rho_s = (g/2 pi^2) int_0^Lambda dk k^2 (M/E) (f^+ + f^-)
    eps   = (g/2 pi^2) int_0^Lambda dk k^2  E    (f^+ + f^-)
    s     = (g/2 pi^2) int_0^Lambda dk k^2 sum_+- [ (x_+-/T) f^+- + ln(1 + e^(-x_+-/T)) ]
    P_log = (g/2 pi^2) int_0^Lambda dk k^2 T [ ln(1 + e^(-(E-mu*)/T))
                                             + ln(1 + e^(-(E+mu*)/T)) ]

Antiparticles **subtract** in `n` and **add** in `rho_s`, `eps` and `P`.

**At T = 0** the occupations become step functions, `f^+ = theta(mu* - E)` and
`f^- = 0` for `mu* > 0`, and with

    k_F = min( sqrt(mu*^2 - M^2), Lambda ),   E_F = sqrt(k_F^2 + M^2)

every one of them integrates in closed form:

    n     = (g/6 pi^2) k_F^3
    rho_s = (g/2 pi^2) (M/2) [ k_F E_F - M^2 arcsinh(k_F/M) ]
    eps   = (g/2 pi^2) (1/8) [ k_F (2 k_F^2 + M^2) E_F - M^4 arcsinh(k_F/M) ]
    P_log = mu* n - eps
    s     = 0

and **all five vanish exactly** when `|mu*| <= M`: a mode too heavy for its own
potential is not in the medium, which is a statement rather than an
optimisation. These are the Dirac-sea integrals below with `Lambda` replaced by
`k_F` and `g_sea` by `g` — the same integral over a different interval. The
`min` is where a cut theory differs from an uncut one, and
`P_log = mu* n - eps` is the T = 0 statement of `Omega = eps - mu n` for one
mode.

**The surface term** (trap 1). The second standard pressure form,

    P_k4 = (g/6 pi^2) int_0^Lambda dk (k^4/E) (f^+ + f^-)

follows from `P_log` by parts, and the boundary term does NOT vanish when the
integral is cut:

    P_log - P_k4 = (g/6 pi^2) Lambda^3 T [ ln(1 + e^(-(E_L - mu*)/T))
                                         + ln(1 + e^(-(E_L + mu*)/T)) ]
    E_L = sqrt(Lambda^2 + M^2)

It is not small: 0.1% of `P` at `(M, mu*, T) = (100, 500, 20)` MeV, 10.5% at
`(40, 590, 30)`, 36.3% at `(140, 700, 5)` and 39.9% at `(140, 700, 50)`.
**Every assembly here uses P_log.** At T = 0 with `k_F < Lambda` the two agree,
which is exactly why the error would hide until a table is built at finite
temperature.

**Quadrature.** Gauss–Legendre on one panel over `[0, Lambda]` cannot resolve
the Fermi step. Breakpoints go at each Fermi momentum
`k_F,j = sqrt(mu*_j^2 - M_f(j)^2)` and, at T > 0, at `k_F,j +- 25 T`, and each
panel is integrated separately; **the cutoff is imposed as the panel upper
limit before the panels are built**, because filtering breakpoints afterwards
can delete the Fermi-surface break and revert silently to single-panel
accuracy.

## The Dirac sea, the condensates and the vacuum

The vacuum integrals of one flavour, closed form, `g_sea = 6`,
`R = sqrt(Lambda^2 + M^2)`:

    rho_s_vac(M) = (g_sea/2 pi^2) (M/2) [ Lambda R - M^2 arcsinh(Lambda/M) ]
    eps_sea(M)   = (g_sea/2 pi^2) (1/8) [ Lambda (2 Lambda^2 + M^2) R
                                        - M^4 arcsinh(Lambda/M) ]

and `d eps_sea/dM = rho_s_vac` identically. The condensate of one flavour is
the sea's scalar density with the medium's own occupation subtracted, summed
over the three colours of that flavour:

    phi_f = -rho_s_vac(M_f) + sum_a rho_s,(f,a) + drho_s,f

with `drho_s,f` the pairing correction below. The constituent masses, including
the determinant cross-terms — the coefficients most often dropped or mis-signed:

    M_u = m_u - 4 G_S phi_u + 2 K phi_d phi_s
    M_d = m_d - 4 G_S phi_d + 2 K phi_u phi_s
    M_s = m_s - 4 G_S phi_s + 2 K phi_u phi_d

`phi_f = <qbar_f q_f>` is negative in the broken phase. The convenient identity
`2 G_S sum_f phi_f^2 = sum_f (M_f - m_f)^2/(8 G_S)` holds ONLY at `K = 0`,
where it is exact to 4e-16; with the determinant on it fails by 34%. The
condensate cost entering both `Omega` and `eps` is

    C = 2 G_S sum_f phi_f^2 - 4 K phi_u phi_d phi_s

**The vacuum solution.** At `mu = T = 0` the mass and condensate equations
close on themselves and are solved by a **damped fixed point** on the masses,

    M <- M + lam (M_new[phi(M)] - M),   lam = 0.3

to a step below 1e-12 MeV in at most 4000 iterations, and NOT by a root finder
on the condensates, which diverges and returns masses that increase with
density. The vacuum constant is

    Omega_vac = eps_vac = -sum_f eps_sea(M_f^vac) + C^vac

and the equality of the two is what makes the Euler relation survive the
subtraction.

**Vacuum diagnostics.** The pion decay constant from the quark loop,

    f_pi^2 = 2 M^2 I_2 ,
    I_2 = (N_c/4 pi^2) [ arcsinh(Lambda/M) - Lambda/sqrt(Lambda^2 + M^2) ]

and the effective bag constant, `Omega` at fixed masses evaluated at the
current masses minus at the broken-phase ones,

    B_eff = Omega[M_f = m_f] - Omega[M_f = M_f^vac]

`B_eff` is a **derived** quantity here, not an input the way a bag constant is
in a bag model.

## The pairing sector

The gap matrix, the 18x18 Bogoliubov–de Gennes problem, the pairing correction
to `Omega` and the Hellmann–Feynman kernels are NOT in this package: they are
`eos/general/pairing.py`, shared with the chiral colour-dielectric model,
because the pairing sector of the two is the same sector. The cut medium
integrals ARE here, under the carve-out for cutoff-regularized NJL integrals,
which are model physics.

**The gap matrix.**

    G_(fa),(gb) = sum_eta Delta_eta eps^(ab eta) eps_(fg eta)
                = sum_eta Delta_eta (B_eta)_(fa),(gb)

so `Delta_1` pairs d with s, `Delta_2` pairs u with s, and `Delta_3` pairs u
with d — the 2SC gap. The matrix is symmetric with identically zero diagonal.
Its eigenvalue multiplicities are a *derived* property of the pattern and are
never assigned by hand; at `Delta_0 = 60` MeV the spectrum of `G` is

    unpaired  (0,0,0)              0 (x9)
    2SC       (0,0,D)              -60 (x2), 0 (x5), +60 (x2)
    CFL       (D,D,D)              -60 (x5), +60 (x3), +120
    uSC       (0,D,D)              +-84.85, +-60 (x2), 0 (x3)
    dSC       (D,0,D)              as uSC

With independent gaps the `+- sqrt(2) Delta_0` eigenvalue generalises to
`+- sqrt(Delta_2^2 + Delta_3^2)`.

**The Bogoliubov–de Gennes problem.** At each momentum `k`, for particles
(`r = +`) and antiparticles (`r = -`) separately, with
`xi^r_j = E_f(j) - r mu*_j`:

    H^r(k) = [[ diag(xi^r),  G          ],
              [ G,          -diag(xi^r) ]]        an 18x18 real matrix

whose spectrum comes in `+-` pairs. The nine quasiparticle energies are the
**non-negative half of the signed spectrum**, `E_a = sort(eig H)[9:]` — not the
nine largest in modulus. The two prescriptions agree in value, but only the
first is smooth through a gapless window, where a branch crosses zero and its
partner crosses back, and that smoothness is what makes the Hellmann–Feynman
derivatives carry the correct branch sign with no sign bookkeeping.

At unequal masses `[G, M] != 0` — the Frobenius norm is 7.4e4 at
`M = (40, 45, 480)` MeV against exactly zero at equal masses — so there is no
closed-form dispersion for a general pattern and the matrix is diagonalised
numerically. The 2SC pattern is the exception:

    E^+- = sqrt((Ebar - mubar)^2 + Delta^2) +- [ (E_d - E_u)/2 - dmu ]
    Ebar = (E_u + E_d)/2 ,  mubar = (mu*_ur + mu*_dg)/2 ,
    dmu  = (mu*_dg - mu*_ur)/2

which reproduces the numerical spectrum to 1e-11 MeV over random
configurations and serves both as a fast path and as the unit test of the
general one. `E^-` may be **negative**: that is the gapless window, the BCS
blocking region, not an error.

**The pairing potential**, written as a *correction* — a difference from the
unpaired spectrum:

    dOmega_pair = -(1/2 pi^2) sum_r int_0^Lambda dk k^2 sum_(a=1..9)
                     [ varphi(E^r_a) - varphi(|xi^r_a|) ]
    varphi(x) = x + 2 T ln(1 + e^(-x/T)) ,   varphi(x) = x at T = 0

This vanishes **identically** at `Delta = 0`, which is what makes the unpaired
phase a clean limit of the same code, and in the clean weak-coupling limit it
obeys the BCS logarithm,
`-dOmega_pair -> (2/pi^2) mu^2 Delta^2 [ln(2 Lambda/Delta) - 1/2]`. Both signs
of `r` are summed: the antiparticle branches contribute 8.8% of the particle
piece at `Lambda = 600` MeV and 17.1% at 1000 MeV. The `|xi_j|` subtraction
kinks at each of the nine `k_F,j`, so the pairing quadrature is split there
(and at `k_F,j +- 25 T`) exactly as above. It is the *splitting* that buys the
accuracy, not the node count: at 100 nodes per panel the relative error is
3e-14 where a single panel of 800 nodes reaches 2e-7. The shipped rule is 24
Gauss–Legendre nodes per panel, overridable per call.

**The gap equations, and everything else the same pass returns.** Since
`varphi'(x) = tanh(x/2T)`, differentiating by Hellmann–Feynman on `H^r` gives,
with eigenvectors `|V_a>` in the doubled basis and `P_j` the projector on mode
`j`:

    Delta_eta/(2 G_D)
      - (1/2 pi^2) sum_r int dk k^2 sum_a <V^r_a| [[0, B_eta],[B_eta, 0]] |V^r_a>
        tanh(E^r_a/2T)  =  0

    dn_j = (1/2 pi^2) sum_r int dk k^2 [ sum_a tanh(E^r_a/2T) dE^r_a/dmu_j
                                       + r tanh(xi^r_j/2T) ] ,
           dH^r/dmu_j = -r (P_j (+) -P_j)

    drho_s,f = -(1/2 pi^2) sum_r int dk k^2 [ sum_a tanh(E^r_a/2T) dE^r_a/dM_f
                                            - sum_(j in f) (M_f/E_j) tanh(xi^r_j/2T) ] ,
           dH^r/dM_f = diag(d) (+) -diag(d),  d_j = (M_f/E_j) delta_(f(j),f)

    ds = (1/2 pi^2) sum_r int dk k^2 sum_a [ psi(E^r_a) - psi(|xi^r_a|) ] ,
         psi(x) = 2 ln(1 + e^(-x/T)) + (x/T) [1 - tanh(x/2T)]

All four come from ONE quadrature pass and one batched diagonalisation;
computing them separately would diagonalise the same 18x18 matrices five times,
and finite-differencing them instead was measured 40x slower and
ill-conditioned enough to lose convergence.

**The kernel is not `Delta/|E|`** (trap 2). That form — obtained by
differentiating `|E|` as though every branch were positive — is wrong by a
factor 12.0 at `Delta = 40` MeV, 1.7 at 60 and 1.3 at 80 for `mu_u = 400`,
`mu_d = 500` MeV, and it makes the gap *grow* with the mismatch, the opposite
of the physics. It agrees with the true kernel only where every branch is
positive.

**Paired densities and entropy are not the unpaired integrals** (trap 3). At
`mu_B = 1400` MeV, `T = 20` MeV, `Delta_3 = 80` MeV the unpaired density
formula is wrong by -21.1% on the paired u modes and +11.6% on the paired d
modes. The entropy is worse: in a fully gapped phase the ratio of paired to
unpaired entropy is 2e-4 at `T = 5` MeV.

**The gap equation has three roots** (trap 4). With a mismatch,
`R(Delta) = Delta/2G_D - kernel` vanishes at `Delta = 0`, at a barrier maximum,
and at the physical BCS root, so a fixed bracket returns whichever it happens
to contain. At `mu* = 450` MeV, `eta_D = 0.75` the roots are 92.71 MeV at zero
mismatch and `(32.4, 92.71)`, `(52.5, 92.71)`, `(59.9, 92.71)` at
`dmu = 50, 60, 65` MeV; the free-energy crossover sits at `dmu_c = 63.59` MeV,
a fraction 0.970 of the weak-coupling Clogston–Chandrasekhar value
`Delta_0/sqrt(2)`, the 3% deficit being the finite cutoff.

## The vector sector

With `n_q = sum_j n_j` the total quark density, the vector interaction energy
and self-energy are

    W(n_q)  = G_V(n_q) n_q^2
    Sigma_V = dW/dn_q = (2 - alpha_eff) G_V(n_q) n_q
    alpha_eff = -d ln G_V/d ln n_q
    mu*_j   = mu_j - Sigma_V

Three forms are implemented, selected by the tier-2 choice:

    constant        G_V = eta_V G_S                          alpha_eff = 0
    power_law       G_V = eta_V G_S (n_ref/n_q)^alpha        alpha_eff = alpha
    gluon_exchange  G_V = (G_V0/G_S) G_S / (1 + u) ,         alpha_eff = (2/3) u/(1+u)
                    u = 8 k_F^2/(9 M_g^2), k_F = (pi^2 n_q/2)^(1/3)

The power law's `eta_V G_S` is the coupling *at* `n_ref`, so the constant form
is the `alpha = 0` member of the same family; `n_ref` is declared in fm^-3 and
multiplied by `(hc)^3` before it meets `n_q`. The gluon-exchange form takes its
strength from its own `G_V0/G_S` rather than from `eta_V`, because a run using
it is not a run at a constant coupling of the same size.

**Why the density dependence exists.** With chiral symmetry restored the scalar
channel dies and the high-density behaviour is set entirely by the vector term.
At constant `G_V` the interaction energy grows like `n^2` against the kinetic
`n^(4/3)` and the sound speed runs away to 1 (Zel'dovich). Writing
`eps = sum_i C_i n^p_i`, each term contributes `P_i = C_i (p_i - 1) n^p_i`, so

    c_s^2 = sum_i C_i p_i (p_i - 1) n^(p_i - 1) / sum_i C_i p_i n^(p_i - 1)
    c_s^2(n -> inf) = max(1 - alpha, 1/3)     for G_V ~ n^(-alpha)

since the vector term then has `p_V = 2 - alpha` against the free-quark 4/3.
`alpha = 2/3` is the marginal exponent, and it is marginal *identically*, not
asymptotically: there the vector term's own pressure is exactly one third of
its own energy density at every density. The gluon-exchange form reaches
`alpha_eff = 2/3` as a consequence of its own structure with no tuning —
0.062, 0.460, 0.608, 0.653 at `n_q = 1e6, 1e8, 1e9, 1e10` MeV^3 — which is why
it is the recommended variant. Pairing does not change the asymptotics:
`c_s^2 -> 1/3 + (2/9) Delta^2/mu^2`, which dies as `1/mu^2`.

**The rearrangement term is mandatory.** Once `G_V` depends on the density,
`Sigma_V` is `dW/dn_q` and not `2 G_V n_q`; the ratio is 0.833 at
`alpha = 1/3` and 0.667 at 2/3, and using the naive form shifts `P` by about 5%
and breaks `n = dP/dmu` at the first digit. (A coupling that depends on a mean
*field* needs no such term, because it enters through that field's own equation
of motion. A density-dependent one does.)

## The totals

With `D = sum_eta Delta_eta^2/(4 G_D)` the pairing cost and `C` the condensate
cost, the matter sector alone — no leptons, no photons — assembles as

    Omega = -sum_j P_med,j - sum_f eps_sea,f + C - (Sigma_V n_q - W)
            + dOmega_pair + D - Omega_vac
    eps   =  sum_j eps_med,j - sum_f eps_sea,f + C + W + eps_pair - eps_vac
    eps_pair = dOmega_pair + D + T ds + sum_j mu*_j dn_j
    s     = sum_j s_med,j + ds
    P     = -Omega

and both carry the *same* vacuum constant, so `Omega_vac = eps_vac` exactly and
Euler survives the subtraction. The conserved-charge sums are

    n_j = n_med,j + dn_j
    n_q = sum_j n_j ,  n_B = n_q/3 ,
    n_C = sum_j q_f(j) n_j ,  n_S = sum_j s_f(j) n_j

and the colour densities that neutrality sets to zero,

    n_3 = sum_f ( n_(f,r) - n_(f,g) )
    n_8 = sum_f ( n_(f,r) + n_(f,g) - 2 n_(f,b) )

these being the generator densities up to the constants 1/2 and 1/3; a row that
must vanish does not care about its normalisation.

**Euler.** The assembly gives

    eps + P = T s + sum_j mu_j n_j

with the *physical* mode potentials, since
`sum_j mu*_j n_j + Sigma_V n_q = sum_j mu_j n_j`. This is audited at every
solved point. Three assembly bugs found during development each produced a
plausible equation of state and each was caught here: a sign error in `eps`
(O(1)); the pairing cost and `dOmega_pair` dropped from both sums (7.7e-3,
small enough to pass for quadrature); and `ds` dropped, which fails only at
T > 0.

## Leptons, photons and the untracked neutrinos

The leptons are added to the totals **after** the matter and are not part of
the phase: they feel no field the quarks feel, carry no colour, and a phase
does not own them — which is why `Omega`, `eps` and `s` above contain none of
them and why the phase-adapter block does not either. **Their integrals are not
cut**: the cutoff regularises the four-fermion interaction, and a free lepton
has none, so its momentum integral runs to infinity.

One charged lepton of mass `m_l` and degeneracy `g_l = 2` at potential `mu_l`:

    n_l   = (g_l/2 pi^2) int_0^inf dk k^2       (f^+ - f^-)
    eps_l = (g_l/2 pi^2) int_0^inf dk k^2  E    (f^+ + f^-)
    P_l   = (g_l/2 pi^2) int_0^inf dk k^2 T [ ln(1 + e^(-(E-mu_l)/T))
                                            + ln(1 + e^(-(E+mu_l)/T)) ]
    s_l   = (eps_l + P_l - mu_l n_l)/T

the same integrands as the medium ones with the upper limit at infinity, so
their T = 0 limit is the closed forms above with `k_F = sqrt(mu_l^2 - m_l^2)`
and no `min`. They are evaluated by the JEL expansion of `eos/general`, the one
home the Fermi integrals of this repository have; the forms above are what it
approximates and what any alternative implementation is checked against.

For a **massless** species — every neutrino here — they close in elementary
functions, with `g_nu = 1` per flavour and antineutrinos included:

    n_nu   = (g_nu/6 pi^2) ( mu_nu^3 + pi^2 mu_nu T^2 )
    P_nu   = (g_nu/24 pi^2) ( mu_nu^4 + 2 pi^2 mu_nu^2 T^2 + (7 pi^4/15) T^4 )
    eps_nu = 3 P_nu
    s_nu   = (g_nu/6) ( mu_nu^2 T + (7 pi^2/15) T^3 )

and the photon gas, massless with `g_gamma = 2` and `mu = 0`, is
Stefan–Boltzmann:

    P_gamma = (pi^2/45) T^4 ,  eps_gamma = 3 P_gamma ,
    s_gamma = (4 pi^2/45) T^3 ,  n_gamma = (2 zeta(3)/pi^2) T^3

The neutrino form at `mu_nu = 0` gives `P_nu/P_gamma = 7/8`, which is the check
that the degeneracy is the one stated.

**Which species are present, and at which potential.** Electrons always, at
`mu_e = mu_nue - mu_C` in a beta-equilibrium mode; muons wherever the `muons`
flag allows them, at `mu_mu = mu_e - mu_nue`, which is muon-decay equilibrium
with a transparent muon family. The electron neutrino appears only when it is
trapped, `mu_nue != 0`; free streaming means `mu_nue = 0`, and then it carries
neither lepton number nor pressure, which is what free streaming is. The
`thermal_neutrinos` flag adds the flavours NOT tracked in the composition as
`mu = 0` gases: three when the electron neutrino free-streams, two when it is
trapped, since the trapped flavour is already counted at its own potential.
Photons follow the flag of the same name.

In a fixed-fraction mode with `leptons=True` none of this applies: there
`mu_C` is an unknown fixed by the charge condition, so the leptons are solved
**after** the matter, from the one condition `n_e + n_mu = n_C` at a single
potential with `mu_mu = mu_e` (no neutrinos: a fixed-`Y_C` table is not a
trapped one). Where the matter turns out *negatively* charged, `n_C <= 0`,
there is nothing for electrons to neutralize and the lepton blocks are empty
with `mu_e = 0`; positrons are not added. With `leptons=False` the result is
charged matter, which is what a mixed-phase construction needs per pure phase.

## The solve

**The unknown vector**, in this order:

    x = ( M_u, M_d, M_s,                          always
          {Delta_eta} for each gap the PATTERN makes free,
          mu_3, mu_8,                             if the pattern pairs at all
          Sigma_V,                                if G_V != 0
          mu_B, mu_C,                             always
          mu_S,                                   iff the mode holds Y_S
          mu_nue )                                iff the mode holds Y_Le

**The rows, in the order the solver assembles them:**

    1..3   M_f - [ m_f - 4 G_S phi_f + 2 K phi_g phi_h ] = 0      f = u, d, s
    then   Delta_eta/(2 G_D) - kernel_eta = 0                     one per free gap
    then   n_3 = 0 ,  n_8 = 0                                     if the pattern pairs
    then   Sigma_V - dW/dn_q = 0                                  if G_V != 0
    then   n_B - n_B^target = 0                                   always
    then   n_C - n_e - n_mu = 0   or   n_C - Y_C n_B = 0          neutrality, or the held charge
    then   n_S - Y_S n_B = 0                                      if Y_S held
    then   (n_e + n_nue) - Y_Le n_B = 0                           if Y_Le held

Each row is divided by the scale of the quantity it balances — `mu_B` for a
potential, `n_B` for a density, `(mu_B/3)^3/pi^2` for a gap or colour row — and
the state is accepted when the largest scaled component is below 1e-10.
**The residual handed to the root finder is already scaled**, not merely judged
scaled afterwards: the raw rows span twenty orders of magnitude (mass rows in
MeV against gap and colour rows in MeV^3) and a root finder terminates on its
own view of the residual. Without that, every row but the largest is driven
only as far as the largest one needs, and a paired solve reports 1e-8 where it
can reach 1e-16. Note in particular that a gap row is a **density**, since
`Delta_eta/2 G_D` carries MeV^3, and judging it against a potential is four
orders of magnitude too strict.

**The modes.**

    mode                       variables         extra unknowns  rows replaced
    beta_eq_neutrinoless       (n_B, T)          —               neutrality; mu_S = 0
    beta_eq_neutrino_trapped   (n_B, Y_Le, T)    mu_nue          neutrality + Y_Le
    fixed_YC                   (n_B, Y_C, T)     —               n_C = Y_C n_B; mu_S = 0
    fixed_YC_YS                (n_B, Y_C, Y_S, T) mu_S           n_C = Y_C n_B, n_S = Y_S n_B

All four close at any temperature. Wherever a temperature is accepted, entropy
per baryon may be given instead, as an outer one-dimensional solve for `T`.
`leptons=True/False` applies to the two fixed-fraction modes; in beta
equilibrium it is meaningless and raises, since the leptons are what the
equilibrium is about. The muon lepton *family* as a conserved charge is not
implemented — `Y_Lmu` raises (see DEFERRED) — while the muon *species* is
available through the flag.

**The pattern is not a mode.** Which condensates survive is decided by free
energy, not declared. Every enumerated candidate — unpaired, 2SC, CFL, and one
asymmetric free seed that can land on uSC, dSC or an unequal-gap state — is
solved to self-consistency, and the converged one with the lowest
`f = eps - T s` is returned; at fixed `mu_B` (which is what the phase adapter
does) the criterion is the largest `P` instead, and the two agree. A candidate
that did not converge is dropped, not substituted. Every point reports the
winner, the three gaps, `mu_3`, `mu_8` and whether the state is gapless.

Two seeding facts. CFL is electrically neutral **without** electrons, so its
seed puts `mu_C` at zero; seeded with an electron-bearing potential the solve
reaches a spurious point with an 11% flavour-density spread. And in an unpaired
region `mu_8` is unconstrained — `n_8` vanishes identically at `mu_8 = 0`
(trap 5) — so it is pinned there rather than solved for. A warm start is keyed
by pattern, because the pattern decides the vector's *layout*; a density sweep
carries the winning pattern's seed and lets the others start cold, which is
also what keeps the enumeration honest.

## What a solved point returns

Fm-based throughout: densities in fm^-3, potentials and masses in MeV, `P`,
`eps` and `f` in MeV/fm^3, `s` in fm^-3.

    converged, error     the status, and the largest SCALED row above. Test it
                         first: when converged is false every other field is
                         the best iterate reached, not a state
    mode                 which of the four equilibria was closed
    n_B, T               the independent variables
    Y_C = n_C/n_B        the non-leptonic charge fraction — imposed in a
    Y_S = n_S/n_B        fixed-fraction mode, an outcome in a beta one
    Y_Le = (n_e + n_nue)/n_B     the electron-family lepton fraction
    pattern, gapless     which candidate won, and whether a quasiparticle
                         branch has reached zero — both part of the answer
    Delta_1,2,3          the gaps, zero in the channels the winner leaves unpaired
    M_u, M_d, M_s        the constituent masses
    mu_B, mu_C, mu_S     the conserved-charge potentials
    mu_3, mu_8           the colour potentials
    mu_e, mu_nu          the lepton potentials
    n_u, n_d, n_s        the three FLAVOUR densities, n_f = sum_a n_(f,a)
    n_e, n_mu, n_nu      the lepton densities
    Y_u, Y_d, Y_s,       the same divided by n_B
    Y_e, Y_nu
    P, eps, s            the totals: matter + leptons + thermal gases
    f = eps - T s        the free-energy density, and what the enumeration ranks by
    state, x             the matter block in natural units, and the converged
                         unknown vector — which is what a warm start is

**`n_s` is the strange-quark density, not a scalar density.** Every other model
in this repository returns a field called `n_s` meaning the *scalar* density,
computed through `n_s = (eps - 3P)/m*`. **Here it is the number density of s
quarks**, the third entry of `(n_u, n_d, n_s)`, and the collision is a real
one: a caller moving from a hadronic model to this one and reading `n_s` as a
scalar density gets a plausible number that means something else. This model's
scalar densities are per flavour, are called `rho_s,f`, and are not returned on
the point at all — they enter through the condensates `phi_f`, which are.

The trace identity itself does not survive here, and not for want of care:
`eps - 3P = M rho_s` holds mode by mode for the *medium* pieces alone and with
`P_k4` rather than `P_log`, and the assembled `eps` and `P` additionally carry
the Dirac sea, the condensate cost `C`, the vector terms and the pairing
correction, each entering the two with different weight. So `rho_s,f` is
integrated from the medium integral and corrected by the Hellmann–Feynman pass,
and the trace identity is not used as a definition anywhere in this model.

**`s` likewise is integrated, not divided.** The identity

    s = (eps + P - sum_j mu_j n_j)/T

is Euler rearranged and holds exactly, but it is used as an *audit* rather than
as the definition. `s` comes from the entropy integrand plus the pairing
correction `ds`, because the identity is a difference of three numbers of order
1e9 divided by `T`, and in a cold nearly-degenerate gas — or in a fully gapped
phase, where `s` is genuinely `e^(-Delta/T)` small — the cancellation eats every
significant digit. Integrating costs one array in a pass already being made and
leaves Euler available as the check it is used as.

**The table row.** `eos_table` sweeps the density axis with a warm start,
bisecting a missed step back towards the last solved point — which is what
carries it across the two thresholds a cold NJL density axis has, the strange
quark's onset and the pairing onset. Its rows carry `n_B`, `T`, `P`, `eps`,
`s`, `S_per_B`, `mu_B`, `mu_C`, `mu_S`, `mu_e`, `Y_C`, `Y_S`, `Y_u`, `Y_d`,
`Y_s`, `Y_e`, `Y_mu-`, `M_u`, `M_d`, `M_s`, `pattern`, `gapless`, `Delta_1..3`,
`mu_3`, `mu_8`, plus `Y_nue` and `mu_nue` where the mode traps; and two columns
so that a quark table and a hadronic one concatenate without renaming,
`chi = 1` and `phase = "Q"`, saying the matter is entirely deconfined. The
progress callback is the repository's — `{mode, line, n_lines, temp_key, temp,
fracs, n_solved, n_requested, elapsed_s}` — plus one key this model adds,
`pattern`, because which phase a line ended in is the first thing a reader of a
colour-superconducting table wants to know.

**The response functions.** A second derivative is defined only once one says
what is held fixed, so `eos_response` names its conditioning. *One freeze is
implemented*, `equilibrium`: nothing is held, the composition re-equilibrates
under the perturbation and so does the pairing pattern — every neighbour of the
stencil re-runs the enumeration. A caller wanting the derivative *within* one
pattern restricts the enumeration instead. Anything else — holding the species
fractions, holding `Y_C`, holding the gaps, or the susceptibility matrix
`chi_ab = dn_a/dmu_b` — raises saying so, and is recorded in
`docs/DEFERRED.md`. Everything is a central difference along a re-solved
sequence of the mode's own equilibrium, with `sigma = s/n_B` the entropy PER
BARYON (CompOSE manual, arXiv:2203.03209):

    cs2_isothermal = (dP/dn_B)_T / (deps/dn_B)_T
    C_V            = T (dsigma/dT)_n_B
    C_P            = T [ (dsigma/dT)_n_B
                         - (dP/dT)_n_B (dsigma/dn_B)_T / (dP/dn_B)_T ]
    cs2_adiabatic  = (C_P/C_V) cs2_isothermal
    Gamma_th       = 1 + [P(n_B,T) - P(n_B,0)] / [eps(n_B,T) - eps(n_B,0)]

`C_P` is the usual Jacobian rotation, holding the pressure by letting the
density move with the temperature. Both sound speeds are named for the thermal
variable they hold and never returned as a bare `cs2`; at T = 0 the ratio
`C_P/C_V` is one by construction and the two coincide, so only the two sound
speeds come back there. `Gamma_th` returns `nan` rather than a signed nonsense
where the cold reference is not below the hot one. In a fully gapped phase
`C_V` is exponentially small at low T, because the paired entropy is: that
suppression is real physics, and it is what makes a colour superconductor cool
differently from unpaired quark matter.

## The phase-adapter surface

`eos/mixed` consumes this model through one function: given
`(mu_B, mu_C, mu_S, T)` it returns the phase block, having closed the model's
own internal system — masses, gaps, `Sigma_V` **and the two colour
potentials**. Colour neutrality is internal because `mu_3` and `mu_8` are not
conserved charges of the mixed system: no hadronic phase carries them and there
is nothing across the interface for them to equilibrate with. The slot carries
the **physical** baryon potential, and the seed is not cacheable: the seed
chooses the root, so caching it would change physics rather than speed.

The rows it drives to zero are the ones above **without** the density row and
without the mode's charge rows, since the three potentials are given rather
than solved for:

    x_int = ( M_u, M_d, M_s, {Delta_eta}_free,
              mu_3, mu_8      if paired,
              Sigma_V         if G_V != 0 )

with the three mass equations, one gap equation per free gap, `n_3 = n_8 = 0`
where the pattern pairs, and `Sigma_V = dW/dn_q` where there is a vector
coupling — in that order, scaled the same way. The block handed back carries
`T`, `mu_B`, `mu_C`, `mu_S`; the three flavour densities; the flavour
potentials `mu_f = mu_B/3 + q_f mu_C + s_f mu_S` and their effective partners
`mu_f - Sigma_V`; the effective masses `M_f`; `n_B`, `n_C`, `n_S`; `P`, `eps`,
`s` and `sum_j mu_j n_j`; and, as declared fields, the three masses, the three
gaps, `mu_3`, `mu_8` and `Sigma_V`. The adapter enumerates the patterns at
every call and keeps the one with the largest `P`, which at fixed potentials is
the stable one; the *label* of that winner is not one of the declared fields —
it comes back as the key of the warm-start mapping the adapter returns
alongside the block, and the gaps in the block are what say whether and how the
phase paired. No lepton enters it. There is no thermo-at-given-*densities* surface, so a
mixed-phase response that would need one raises.

## The five traps

Each of these returns a plausible-looking wrong answer; each is stated in full
where it belongs above.

1. **P must come from the logarithm form when the integral is cut** — the
   surface term does not vanish at a finite cutoff, and at T = 0 below the
   cutoff the two forms agree, which is how the error hides.
2. **The gap kernel is not `Delta/|E|`** — it is Hellmann–Feynman on the BdG
   matrix, which carries the branch sign for free.
3. **Paired densities and entropy are not the unpaired Fermi integrals.**
4. **The gap equation has three roots** under any mismatch — scan, then bracket
   each sign change.
5. **`mu_8` is unconstrained in an unpaired region**, where `n_8` vanishes
   identically at `mu_8 = 0`. It is pinned there, never solved for.

## What the implementation reproduces

| quantity | computed | published |
|---|---|---|
| M_u (vacuum) | 367.648 MeV | 367.7 |
| M_s (vacuum) | 549.479 MeV | 549.5 |
| (-phi_u)^(1/3) | 241.946 MeV | 241.9 |
| (-phi_s)^(1/3) | 257.688 MeV | 257.7 |
| f_pi (quark loop) | 92.391 MeV | 92.4 |
| B_eff^(1/4) | 228.93 MeV | 357.49 MeV/fm^3 |
| 2SC gap at mu* = 450 MeV, eta_D = 0.75 | 92.71 MeV | — |
| Clogston ratio dmu_c/(Delta_0/sqrt2) | 0.970 | 1 (weak coupling) |

The published column is Rehberg, Klevansky & Hüfner for the vacuum rows; the
tier-1 parameters that produce them are in the table above, so every number
here is checkable from this document alone.

The two neutral solved points at `mu_B = 1500` MeV, `T = 0`, `eta_D = 0.75`:

    unpaired   M = (9.84, 8.55, 265.59) MeV, mu_C = -34.20 MeV,
               n_B = 1.4319 fm^-3, P = 302.12 MeV/fm^3
    2SC        Delta_3 = 95.50 MeV, mu_3 = 0, mu_8 = -2.46 MeV,
               M_s = 243.13 MeV, mu_C = -62.27 MeV,
               n_B = 1.4887 fm^-3, P = 324.75 MeV/fm^3

**One deliberate discrepancy.** The specification reports
`M_u, M_d = (11.96, 7.65)` MeV for the 2SC point and this implementation gives
`(9.73, 8.90)`. The difference is the sign of the hole amplitudes in the paired
scalar density `drho_s,f`, and `n_B = dP/dmu_B` along the neutral solution
decides it: it holds to the finite-difference floor with the sign used here and
fails by 2.6e-4 with the other. Near chiral restoration `M - m` is the small
difference of two large numbers, which is why a percent-level change in one
scalar density moves the light masses by 20% and moves `M_s`, `Delta_3`,
`mu_8`, `mu_C`, `n_B` and `P` not at all. Recorded in `docs/DEFERRED.md`.

## Layout

    parameters.py      Parameters: the RKH set and the three coupling tiers
    couplings.py       G_V as a function of the state, and its rearrangement
    species.py         SpeciesFlags, the quantum numbers, the gap patterns
    thermodynamics.py  quantities computed FROM the state, and the internal
                       solve at fixed conserved-charge potentials
    solver.py          the equilibrium conditions and the pattern enumeration
    table.py           the warm-started density sweep + progress callback
    api.py             eos_point / eos_table / eos_response
    responses.py       second derivatives, by re-solved finite differences
    verify/            the physics invariants

Tests: `test/njl/` for the model, `test/general/test_pairing.py` for the shared
pairing machinery.

## Not implemented (see docs/DEFERRED.md)

RG-consistent regularization at `lambda > 1` — the parameter exists and any
value but 1 raises rather than returning a divergent number, since the
counterterm cancelling the medium's logarithmic divergence
`-(2/pi^2) mubar^2 Delta^2 ln Lambda_UV` is not written; the 't Hooft–diquark
cross-term `K'`; the trapped muon lepton family as a conserved charge; the
composition and gap freezes of `eos_response`, and the susceptibility matrix;
and the dilaton/colour-dielectric graft, which the specification marks as
unverified for transition order, pairing coexistence and finite temperature.

## The self-bound surface, and the two-flavour arm

### What is computed

A parametrization whose pressure crosses zero at finite density describes
**self-bound** matter: the phase ends there, with no crust below it. The
quantity reported at that endpoint is the energy per baryon,

    E/A = eps / n_B      at   P(n_B) = 0,  T = 0      [MeV]

which is what a lump of this matter at rest weighs per baryon. The entry point
is `zero_pressure_point(par, species)`, and it returns a `ZeroPressurePoint`
carrying `n_B`, `E_per_A`, `mu_B`, `Y_S`, `mu_S`, the identity residual below,
the pressure actually reached, the flavour content requested, and whether
`E_per_A` fell below the 930.4 MeV of iron.

### The identity that makes the read self-checking

At T = 0 the Euler relation is

    eps + P = sum_i mu_i n_i ,

so at P = 0 the energy per baryon IS the Gibbs energy per baryon. Expanding
the species potentials in the conserved-charge basis, mu_i = B_i mu_B +
C_i mu_C + S_i mu_S, and using beta equilibrium (mu_C + mu_e = 0) together with
total electric neutrality (n_C = n_e), the charge term and the lepton term
cancel exactly:

    sum_i mu_i n_i = mu_B n_B + mu_C n_C + mu_S n_S + mu_e n_e
                   = mu_B n_B + mu_S n_S          (since mu_C n_C + mu_e n_e = 0)

and therefore

    E/A = mu_B + Y_S mu_S ,      Y_S = n_S / n_B .            (*)

**The full form is the one to use.** `E/A = mu_B` is the special case
Y_S mu_S = 0, which holds in every beta-equilibrium mode because strangeness
self-equilibrates there and mu_S = 0. It does NOT hold in a colour-flavour
locked phase, where the condensate pairs the three flavours at equal densities
and unequal masses force unequal potentials: on the CFL surface of `eos.alphabag`
at Delta_0 = 100 MeV, mu_S = 40.68 MeV, and mu_B alone gives 895.87 MeV where
E/A is 936.55 MeV. There are not two conventions here, only one identity and
the cases in which a term of it drops out.

`(*)` is checked at every located root; a root that misses it is a root of
something other than P.

### How the root is found

`eos.general.zero_pressure.locate_zero_pressure` samples P(n_B) on a grid,
takes the LOWEST density at which P rises through zero, and refines it by
Brent's method. The scan is not a convenience: P(n_B) can cross zero more than
once, and a crossing where P FALLS is the top of a mechanically unstable
region rather than a surface. It takes the state as a callable, so it holds no
model and lives in `general/`; a density where the solve does not converge
thins the scan rather than aborting it, and a set with no surface at all comes
back as a status, never as an exception.

### The Bodmer-Witten window

The pair of numbers is a two-sided gate on a parameter set:

| arm | condition | what it says |
|---|---|---|
| three-flavour | `E/A < 930.4 MeV` | strange quark matter is absolutely stable |
| two-flavour | `E/A > 930.4 MeV` | ordinary nuclei are not already decaying into it |

A set failing either is excluded. **Both facts are REPORTED and neither is
asserted**: whether a set sits in the window is a property of the set, so
`below_iron` is a field on the result and no `verify/` entry fails on it. Note
that the same `below_iron = True` reads in opposite directions on the two arms.

**The content requested and the content found can differ, and the result
carries both.** A three-flavour request returns whatever strangeness the
equilibrium actually populated: a set whose surface sits below the s quark's
threshold returns `Y_S = 0` and the two-flavour number from the three-flavour
call. Read the content off `Y_S`, never off `two_flavour`.

### The `two_flavour` flag

Two-flavour quark matter is `beta_eq_neutrinoless` with the strange sector
switched off — which is what it physically is — and not `fixed_YC_YS` at
Y_S = 0. The distinction is not stylistic. With no populated species carrying
strangeness, n_S = 0 holds for a whole range of mu_S, so the row
n_S = Y_S n_B leaves mu_S undetermined and its Jacobian column null; a solve
then converges on round-off. Switching the sector off removes the flavour from
the unknown vector instead, and CLAUDE.md section 4 states the rule directly:
*no sector is enabled or disabled implicitly because its coupling happens to
be zero — if a sector is off, its flag is False.*

Accordingly ``eos.njl`` **raises** on `fixed_YC_YS` with the flag on.

The flag defaults to `False`, meaning the u, d, s matter this model has always
solved, so `SpeciesFlags()` moves no existing number. It is named for the
restriction rather than for the sector because a sector-voiced
`strange_quarks` would have had to default `True` to keep that, which section 4
forbids.

With the flag on:

- n_s = 0, Y_S = 0 and mu_S = 0 identically, exactly rather than approximately;
- mu_B = mu_u + 2 mu_d and mu_C = mu_u - mu_d are unchanged, neither reading
  mu_s, so `(*)` collapses to E/A = mu_B with both strange terms vanishing;
- mu_s is reported as the weak-equilibrium value mu_d — the relation
  s <-> d still holds, there is simply nothing populated at it.

### What stays when the strange flavour goes

**The s condensate stays; only the s Fermi sea is emptied.** The three strange
colour-flavour modes contribute nothing to the medium — no density, no scalar
density, no pressure, energy or entropy — but `phi_s = <sbar s>` is still solved
from its own field equation, and it still feeds the light-quark masses through
the 't Hooft determinant term, 2 K phi_d phi_s in M_u and 2 K phi_u phi_s in M_d. That is the physics of two-flavour quark matter: the strange Fermi
sea is empty, while the strange condensate of the QCD vacuum is not.

Dropping the strange field from the equations instead would move M_u, M_d and
the subtracted vacuum constant with them — it would change the MODEL, not the
flavour content asked of it — so the flag does not touch them.

### Pairing patterns

A diquark containing an s quark is not a state two-flavour matter has. With
the flag on, `Delta_1` (d-s) and `Delta_2` (u-s) therefore have nothing to
pair, so the patterns that carry them — `CFL`, `uSC`, `dSC`, `free` — leave
the default enumeration, and an explicitly requested one raises. The patterns
that survive are `unpaired` and `2SC`, the u-d condensate. This is the same
split the `csc` flag already makes: a candidate that loses on free energy is
dropped, a call that asks for a state the flags forbid is refused.

## References

- P. Rehberg, S. P. Klevansky, J. Hüfner, Phys. Rev. C **53**, 410 (1996),
  arXiv:hep-ph/9506436 — the parameter set and the vacuum fit.
- M. Buballa, Phys. Rept. **407**, 205 (2005), arXiv:hep-ph/0402234 — the review.
- S. B. Rüster, V. Werth, M. Buballa, I. A. Shovkovy, D. H. Rischke,
  Phys. Rev. D **72**, 034004 (2005), arXiv:hep-ph/0503184 — the neutral
  three-flavour pairing sector.
- A. W. Steiner, S. Reddy, M. Prakash, Phys. Rev. D **66**, 094007 (2002),
  arXiv:hep-ph/0205201 — colour neutrality and the CFL `mu_8`.
- M. G. Alford, A. Schmitt, K. Rajagopal, T. Schäfer, Rev. Mod. Phys. **80**,
  1455 (2008), arXiv:0709.4635 — the review of colour superconductivity.
- G. Baym, T. Hatsuda, T. Kojo, P. D. Powell, Y. Song, T. Takatsuka,
  Rept. Prog. Phys. **81**, 056902 (2018), arXiv:1707.04966 — the
  't Hooft–diquark cross-term.
- G. Pagliara, J. Schaffner-Bielich, Phys. Rev. D **77**, 063004 (2008),
  arXiv:0711.1119.
- S. Kunkel, I. A. Rather et al., arXiv:2607.11537 — the `kunkel` couplings.
- H. Gholami, M. Hofmann, M. Buballa, arXiv:2408.06704 — RG-consistent
  regularization.
- S. Typel, M. Oertel, T. Klähn et al., the CompOSE manual,
  Eur. Phys. J. A **58**, 221 (2022), arXiv:2203.03209 — the response
  functions.

All bibliography keys are in `docs/eos.bib`.
