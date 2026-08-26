# DID / DIDY — a relativistic mean field with density- AND isospin-density-dependent couplings

The same description in LaTeX, with the bibliography, is `did.tex` (compiled
against `../../docs/eos.bib`). This file carries the same information in plain
text.

`eos/did` implements the model of Frohaug, Maslov, Dexheimer, Grefa, Jahan,
Ratti & Restrepo, arXiv:2511.15646: a density-dependent RMF in which every
baryon-meson coupling depends not only on n_B but also on the isospin asymmetry

    beta = sum_i tau_3i n_i / n_B.

That second dependence is the model's reason for existing — it is what lets the
hyperon single-particle potentials be reproduced in NEUTRON matter as well as in
symmetric matter, as the HAL QCD-based Brueckner-Hartree-Fock calculations of
Kohno et al. require — and it is what makes the thermodynamics different from an
ordinary DD-RMF: there are TWO rearrangement self-energies, one per state
variable the couplings depend on.

## Conventions

Three, fixed once, two of them differing from the paper's, none ever silently
flipped.

**Isospin.** `tau_3i = 2 I_3i`, normalised to +/-1 for the nucleons:

    tau_3(p)  = +1   tau_3(n)  = -1
    tau_3(Sigma+/-) = +/-2      tau_3(Sigma0) = tau_3(Lambda) = 0
    tau_3(Xi0) = +1  tau_3(Xi-) = -1
    tau_3(Delta) = +3, +1, -1, -3

This is the paper's normalisation and it enters twice: in the rho coupling
`g_rho i tau_3i rho` and in beta itself. It is NOT
`eos.general.particles.Particle.t3`, the DD2 rho-coupling convention in which
the Sigma factor of two is carried by a coupling ratio `x_rho Sigma = 2`; DID
fits `g_rho Sigma` and `g_rho Xi` independently and has no ratio to absorb it.

**Isospin asymmetry.**

    beta = (1/n_B) sum_{i in B} tau_3i n_i,      n_B = sum_{i in B} n_i

summed over EVERY active baryon, so beta = 0 in isospin-symmetric matter (ISM)
and -1 in pure neutron matter (NM). For nucleonic matter beta = 2 Y_C - 1 with
Y_C the proton fraction. |beta| > 1 is reachable in Sigma-rich matter and the
couplings simply extrapolate there.

**Strangeness.** `S = +1 per s quark`, so S(Lambda) = +1 and S(Xi) = +2 — the
opposite of the paper's PDG sign, and the repository-wide convention. It cancels
out of every mode with mu_S = 0 and is carried by `eos.general.basis` elsewhere.
The electric charge C counts strongly interacting matter ONLY: leptons are
excluded, and total neutrality `n_C = n_e + n_mu` is a separate condition a mode
may or may not impose.

**Units** on every public boundary are fm-based: n in fm^-3, T and mu in MeV,
eps and P in MeV/fm^3. The one internal exception is the meson field equations,
where a density must be written in MeV^3 to divide by a squared meson mass; the
factor `(hbar c)^3` appears there and nowhere else.

## The Lagrangian and the mean fields

The baryon octet (optionally plus the Delta(1232) quartet) coupled to sigma,
omega, phi and rho:

    L = sum_i psibar_i ( i gamma_mu d^mu - m_i + g_sigma i sigma
                         - g_omega i gamma_mu omega^mu
                         - g_phi i   gamma_mu phi^mu
                         - g_rho i   gamma_mu tau_i . rho^mu ) psi_i
        + 1/2 (d_mu sigma d^mu sigma - m_sigma^2 sigma^2)
        - 1/4 omega_munu omega^munu + 1/2 m_omega^2 omega_mu omega^mu
        + 1/2 m_phi^2 phi_mu phi^mu
        - 1/4 rho_munu . rho^munu   + 1/2 m_rho^2 rho_mu . rho^mu

There is no delta (a0(980)) meson and no f0(980)/sigma*: both omissions are
deliberate in the paper. The nucleon effective-mass splitting the delta would
produce is poorly constrained; the Lambda-Lambda force the f0(980) carries is
much weaker than the Lambda-N one (adding it to DD2Y lowered M_max from 2.04 to
1.87 M_sun, Marques et al. 2017). `SpeciesFlags` carries no flag for either.

In mean field the meson fields are replaced by uniform, static expectation
values (rho is the third isospin component):

    sigma = (1/m_sigma^2) sum_i g_sigma i n^s_i
    omega = (1/m_omega^2) sum_i g_omega i n_i
    phi   = (1/m_phi^2)   sum_i g_phi i   n_i
    rho   = (1/m_rho^2)   sum_i g_rho i tau_3i n_i

The phi is NOT a hyperon-only field here: the SU(3) sector below gives the
NUCLEON `g_phiN = -5.20` at saturation for the published set, so the phi is
active at every composition and `SpeciesFlags(phi_field=False)` RAISES rather
than switching a sector off.

## The couplings

### Density and isospin dependence

Each coupling interpolates between a branch fitted in symmetric matter and one
fitted in neutron matter:

    g_Mi(n_B, beta) = [1 - w] g^S_Mi(n_B) + w g^N_Mi(n_B)
    w(x, beta)      = beta^2 tanh(x/e),      x = n_B/n_0,   e = 1/3 fixed

The `tanh(x/e)` factor is not decoration: it makes the couplings
isospin-INDEPENDENT at zero density, which is what keeps Sigma^t — which carries
a 1/n_B prefactor — finite as n_B -> 0.

Both branches carry the same shape in density, so a vertex is two numbers and a
meson is one function:

    g^{S,N}_Mi(n_B) = g^{S,N(0)}_Mi F_M(x)

    F_M(x) = E_M(x) (1 - t_M)/2 + b_M (1 + t_M)/2
    E_M(x) = exp[1 - ((x+1)/2)^(2 a_M)]
    t_M(x) = tanh[(x - c_M)/d_M]

`g^{S,N(0)}_Mi` is the vertex strength near saturation, `a_M` sets the
low-density decay, `b_M` is the high-density plateau in units of `g^(0)`, and
`c_M`, `d_M` are the centre and width of the switch between them, in units of
n_0. The published set takes `c_sigma = infinity`, which removes the plateau
from the scalar sector (flattening it drives c_s^2 < 0) and makes `b_sigma` and
`d_sigma` irrelevant; the code branches on `c_M = inf` rather than feeding it to
tanh, where `inf - inf` would be nan.

Both derivatives are analytic and both are needed:

    dg_Mi/dn_B  = (1/n_0) { [(1-w) g^{S(0)} + w g^{N(0)}] F'_M(x)
                            + (g^{N(0)} - g^{S(0)}) (dw/dx) F_M(x) }
    dg_Mi/dbeta = (g^{N(0)} - g^{S(0)}) (dw/dbeta) F_M(x)

    F'_M(x) = E'_M(x) (1 - t_M)/2 + (b_M - E_M(x)) t'_M(x)/2
    E'_M(x) = -a_M ((x+1)/2)^(2 a_M - 1) E_M(x)
    t'_M(x) = (1 - t_M^2)/d_M
    dw/dx    = (beta^2/e) [1 - tanh^2(x/e)]
    dw/dbeta = 2 beta tanh(x/e)

At beta = 0 both `w` and `dw/dbeta` vanish, so `dg/dbeta = 0`, `Sigma^t = 0` and
`g_Mi = g^{S(0)}_Mi F_M(x)`: in symmetric matter the model IS an ordinary
DD-RMF. That is a check in `verify/run_full_check.py`, not a remark.

### The SU(3) vector sector

The omega and phi couplings of the whole octet follow from the octet coupling
g_8, the singlet-to-octet ratio z = g_1/g_8, the mixing angle theta and
alpha = F/(D+F). With c_i the coefficient of each baryon,

    g^(0)_omega i / g_8 = 1 - c_i tan(theta)
    g^(0)_phi i   / g_8 = -tan(theta) - c_i

    c_N      = (z/sqrt3)(1 - 4 alpha)
    c_Lambda = (2z/sqrt3)(1 - alpha)
    c_Sigma  = -(2z/sqrt3)(1 - alpha)
    c_Xi     = (z/sqrt3)(1 + 2 alpha)

The model fixes ideal mixing, `tan(theta) = 1/sqrt2`, and `alpha = 1`, leaving z
as the only fitted parameter of this sector. With alpha = 1 both c_Lambda and
c_Sigma vanish, which is why the Lambda and Sigma vector couplings coincide.

**One correction to the paper's Eq. (6).** That equation prints the Xi line of
the phi sector with `2z/sqrt3` where the pairing above has
`c_Xi = (z/sqrt3)(1 + 2 alpha)`. The paired form used here is fixed by the SU(6)
limit: at `z = 1/sqrt6`, `alpha = 1` and ideal mixing it gives

    g_omegaLambda = g_omegaSigma = (2/3) g_omegaN,   g_omegaXi = (1/3) g_omegaN
    g_phiN = 0,   g_phiLambda = g_phiSigma = -(sqrt2/3) g_omegaN,
    g_phiXi = -(2 sqrt2/3) g_omegaN

the textbook SU(6) ratios; the printed form gives `g_phiXi = -sqrt2 g_omegaN`
and breaks all four. `verify/run_full_check.py` asserts the SU(6) limit.

**The aggregated strength.** What the Bayesian analysis varies is not `g_omegaN`
but the combination nucleonic matter feels, since both vectors enter the energy
as g^2/m^2 times the density (paper Eq. 52):

    g~^{S,N(0)}_omegaN = g^{S,N(0)}_omegaN
                         sqrt(1 + [(g_phiN/m_phi)/(g_omegaN/m_omega)]^2)

With `g_omegaN = g_8 A_omega` and `g_phiN = g_8 A_phi` this inverts in closed
form:

    g_8 = g~_omegaN / sqrt(A_omega^2 + (m_omega/m_phi)^2 A_phi^2)

so the implementation stores `g~_omegaN` and z and derives the two couplings —
storing them separately would make the fitted quantity a function of two stored
ones and let them drift apart.

### Scalar and isovector hyperon couplings, and the branch tying

The scalar sector is NOT related by SU(3): `g^{S(0)}_sigma Y` for
Y in {Lambda, Sigma, Xi} are fitted directly to the HAL QCD-based hyperon
potentials, as are `g^{S(0)}_rho Sigma` and `g^{S(0)}_rho Xi`, with
`g_rho Lambda = 0` identically (the Lambda carries no isospin). The
neutron-matter branch of each hyperon vertex is tied to the symmetric one by the
nucleon ratio of the same meson,

    g^{N(0)}_MY = g^{S(0)}_MY  g^{N(0)}_MN / g^{S(0)}_MN

so a hyperon carries one fitted number per meson rather than two. A vertex whose
symmetric value vanishes stays zero in both branches.

### Delta(1232) isobars: an extension beyond the paper

arXiv:2511.15646 has no Delta isobars. This implementation adds the quartet with
the ratio scheme `eos/dd2` uses,

    g^{S,N(0)}_M Delta = x_M Delta g^{S,N(0)}_MN,  M in {sigma, omega, rho},
    g_phi Delta = 0

so the Delta inherits both the density and the isospin dependence of the nucleon
vertex. The default is universal coupling, x = 1 for all three;
`nmp.delta_ratios_from_potential` instead fixes `x_sigma Delta` from a chosen
Delta potential in ISM at saturation, where rho = 0 and Sigma^t = 0 so that U_i
collapses to one linear equation,

    U_Delta = -x_sigma Delta g_sigmaN sigma + x_omega Delta g_omegaN omega
              + Sigma^r

refused outside the literature range U_Delta in [-100, -50] MeV. There is no
published DID Delta table, so this is a choice this implementation offers, not a
result of the paper.

## The parameter set

Every number here is an ARGUMENT: `Parameters` is passed into every entry point
and nothing reads a module-level constant, so a Bayesian run varies these across
millions of calls without editing a source file. `Parameters.default()` is the
maximum-likelihood set of Table II, transcribed digit for digit, with n_0 from
Table III and the meson masses from Table I. DID and DIDY are the SAME set —
what distinguishes DIDY is `SpeciesFlags(hyperons=True)` — so `named("DID")` and
`named("DIDY")` return the same object and say so.

    saturation    n_0 = 0.15880045 fm^-3     Table III; fixes P(n_0) = 0 in ISM

    meson masses  m_sigma = 550.0 MeV        Table I (the DD2Y values)
                  m_omega = 783.0
                  m_phi   = 1020.0
                  m_rho   = 763.0

    sigma         g^{S(0)}_sigmaN  = 8.94873669     fitted
                  g^{N(0)}_sigmaN  = 8.89241948     fitted
                  a_sigma          = 0.16394393     fitted
                  c_sigma          = infinity       fixed a priori: no plateau
                  b_sigma, d_sigma = 0.0, 1.8       irrelevant while c = inf
                  g^{S(0)}_sigmaLambda = 7.51077621 fitted to U_Lambda
                  g^{S(0)}_sigmaSigma  = 6.26418057 fitted to U_Sigma
                  g^{S(0)}_sigmaXi     = 6.53781517 fitted to U_Xi

    omega, phi    g~^{S(0)}_omegaN = 10.82857726    fitted
                  g~^{N(0)}_omegaN = 11.00228164    fitted
                  z = g_1/g_8      = 0.07720445     fitted
                  a_omega          = 0.15313180     fitted
                  b_omega          = 0.80           fixed a priori
                  c_omega, d_omega = 3.5, 1.8       fixed a priori
                  alpha            = 1              fixed
                  tan(theta)       = 1/sqrt2        ideal mixing, fixed

    rho           g^{S(0)}_rhoN    = 3.23020263     fitted
                  g^{N(0)}_rhoN    = 2.59340047     fitted
                  a_rho            = 0.39223762     fitted
                  b_rho            = 0.40           fixed a priori
                  c_rho, d_rho     = 3.5, 1.8       fixed a priori
                  g^{S(0)}_rhoSigma = 0.00545444    fitted
                  g^{S(0)}_rhoXi    = 1.11415631    fitted
                  g_rhoLambda       = 0             identically, I_Lambda = 0

    Delta         x_sigmaDelta, x_omegaDelta, x_rhoDelta = 1, 1, 1
                  the extension above; not in arXiv:2511.15646

    blend         e = 1/3                           fixed

Fifteen numbers are the fit (marked "fitted"). The vector transition zones
c_M = 3.5, d_M = 1.8 in units of n_0, the plateaus b_omega = 0.80 and
b_rho = 0.40, and c_sigma = infinity were fixed a priori rather than sampled.

The omega and phi couplings are not stored: they are derived from `g~_omegaN`
and z through the SU(3) relations, and the hyperon sigma and rho vertices carry
one fitted number each with the neutron branch following from the tying rule.
The vertex strengths `g^(0)` that result, with `g_8^S = 9.178769` and
`g_8^N = 9.326009`:

    multiplet   sigma S / N          omega S / N            phi S / N              rho S / N
    N           8.948737 / 8.892419  10.046675 / 10.207836  -5.262966 / -5.347391  3.230203 / 2.593400
    Lambda      7.510776 / 7.463509   9.178769 /  9.326009  -6.490370 / -6.594484  0        / 0
    Sigma       6.264181 / 6.224758   9.178769 /  9.326009  -6.490370 / -6.594484  0.005454 / 0.004379
    Xi          6.537815 / 6.496671   8.310864 /  8.444181  -7.717774 / -7.841577  1.114156 / 0.894512
    Delta       8.948737 / 8.892419  10.046675 / 10.207836   0        /  0         3.230203 / 2.593400

These are the `g^(0)`; the shape factor F_M(x) multiplies them.
`F_sigma(1) = 1` exactly, `F_omega(1) = F_phi(1) = 0.98829` and
`F_rho(1) = 0.96488`, so in symmetric matter at saturation the couplings are
`g_sigmaN = 8.9487`, `g_omegaN = 9.9291`, `g_phiN = -5.2014`,
`g_rhoN = 3.1168`.

**Masses and degeneracies.** Baryon and lepton masses are not model parameters:
they are the PDG values held once in `eos.general.particles` and shared by every
model. `d_i` is the spin degeneracy 2J+1 that multiplies every integral below.

    i           m_i [MeV]   C_i   S_i   tau_3i   d_i   multiplet
    p            938.2721    +1     0     +1      2    N
    n            939.5654     0     0     -1      2    N
    Lambda      1115.683      0    +1      0      2    Lambda
    Sigma+      1189.370     +1    +1     +2      2    Sigma
    Sigma0      1192.642      0    +1      0      2    Sigma
    Sigma-      1197.449     -1    +1     -2      2    Sigma
    Xi0         1314.860      0    +2     +1      2    Xi
    Xi-         1321.710     -1    +2     -1      2    Xi
    Delta++     1232.0       +2     0     +3      4    Delta
    Delta+      1232.0       +1     0     +1      4    Delta
    Delta0      1232.0        0     0     -1      4    Delta
    Delta-      1232.0       -1     0     -3      4    Delta
    e-             0.510999   -      0     -      2    -
    mu-          105.6584     -      0     -      2    -
    nu             0          -      0     -      1    -

Leptons carry no strong charge C. The neutrino degeneracy is 1 because only one
helicity state exists; antineutrinos come from the antiparticle branch of the
integral.

**Three routes to a parameter set.** CLAUDE.md section 6 makes model
parameters arguments, so all three have to exist -- and one of the three is
refused here rather than written. *By name:* `Parameters.default()` is the
maximum-likelihood set above, and `Parameters.named('DID')` / `named('DIDY')`
take it by name; the two are the same numbers, DIDY being
`SpeciesFlags(hyperons=True)` rather than a second parameterisation. *A new
set:* `dataclasses.replace(Parameters.default(), a_sigma=...)`. Twenty-nine of
the thirty-four fields carry no default, so bare field-by-field construction
means supplying all twenty-nine, and `with_deltas` is the constructor for the
Delta extension. *From nuclear-matter parameters:* **no route.**
`nmp.invert_nmp` and `nmp.from_nmp` exist and raise `NotImplementedError`
naming the reason: DID's couplings are the maximum-likelihood point of a
Bayesian analysis over 18 observables rather than the solution of a fixed list
of saturation data, and the model carries two inequivalent symmetry energies
(S and S_2, differing by 2.72 MeV at saturation), so the list to impose is
itself undetermined. They raise rather than being absent because an
`AttributeError` is a gap a caller cannot interpret. `nmp.compute_nmp` is the
forward direction and is complete.

## Thermodynamics

### The grand potential and the two rearrangement terms

    Omega/V = 1/2 (m_sigma^2 sigma^2 - m_omega^2 omega^2
                   - m_phi^2 phi^2 - m_rho^2 rho^2)
              - n_B Sigma^r
              - sum_i (tau_3i - beta) n_i Sigma^t
              - sum_i (d_i T / 2pi^2) INT_0^inf k^2 dk
                  ( ln[1 + e^{-(E*_ki - nu_i)/T}]
                  + ln[1 + e^{-(E*_ki + nu_i)/T}] )

with `E*_ki = sqrt(k^2 + m*_i^2)` and `P = -Omega/V`. The two rearrangement
self-energies are what make this consistent when the couplings depend on the
state:

    Sigma^r = sum_i [ -(dg_sigma i/dn_B) sigma n^s_i
                      + (dg_omega i/dn_B) omega n_i
                      + (dg_phi i/dn_B)   phi   n_i
                      + (dg_rho i/dn_B)   tau_3i rho n_i ]

    Sigma^t = (1/n_B) sum_i [ -(dg_sigma i/dbeta) sigma n^s_i
                              + (dg_omega i/dbeta) omega n_i
                              + (dg_phi i/dbeta)   phi   n_i
                              + (dg_rho i/dbeta)   tau_3i rho n_i ]

Sigma^r is the familiar density rearrangement term; Sigma^t is its isospin
counterpart and is specific to this model. Both follow from requiring
`mu_i = deps/dn_i` at T = 0, once the chain rule

    d/dn_i = d/dn_B + [(tau_3i - beta)/n_B] d/dbeta

is applied to the couplings — the whole content of the paper's Appendix A, and
the reason Sigma^t appears weighted by `(tau_3i - beta)` rather than uniformly.

**Two properties that are checked, not assumed.** First, both terms enter mu_i
and P and NEITHER enters eps. Second, the Sigma^t term of Omega vanishes
identically at the self-consistent beta,

    sum_i (tau_3i - beta) n_i = n_B beta - beta n_B = 0

and so does its contribution to `sum_i mu_i n_i`. Sigma^t therefore shifts the
individual chemical potentials — which is the point, it is what splits the Sigma
and Xi single-particle potentials in neutron matter — while leaving P, eps and
`sum_i mu_i n_i` untouched.

### Effective masses, effective potentials, single-particle potentials

    m*_i  = m_i - g_sigma i sigma

    mu_i  = nu_i + Sigma^v_i,
    Sigma^v_i = g_omega i omega + g_phi i phi + g_rho i tau_3i rho
                + Sigma^r + (tau_3i - beta) Sigma^t

Because the species potentials are derived from the conserved charges,
`mu_i = mu_B + C_i mu_C + S_i mu_S`, this inverts to the form the solver
iterates on:

    nu_i = mu~_B + C_i mu_C + S_i mu_S
           - g_omega i omega - g_phi i phi - g_rho i tau_3i rho
           - (tau_3i - beta) Sigma^t,          mu~_B = mu_B - Sigma^r

`mu~_B` is the KINETIC baryon potential: Sigma^r is common to every species, so
carrying it outside the iteration removes its density circularity, and `mu~_B`
varies smoothly along a density sweep, which is what makes warm starts work.
Sigma^t cannot be absorbed the same way — it is weighted by `(tau_3i - beta)`
and so differs per species — and stays inside as an unknown of the state.

The single-particle potential, the energy gained by adding a baryon at k = 0 to
the medium, is

    U_i = Sigma^v_i - Sigma^s_i
        = -g_sigma i sigma + g_omega i omega + g_phi i phi + g_rho i tau_3i rho
          + Sigma^r + (tau_3i - beta) Sigma^t

evaluated for a TEST particle at the medium's fields, so U_Y at n_0 is
meaningful before any hyperon has appeared — which is what the hyperon couplings
were fitted to, in ISM and in NM.

### One species as an ideal Fermi gas

Every species is an ideal Fermi gas at its own (nu_i, m*_i), with antiparticles.
With

    f_i(k)    = 1/(e^{(E*_ki - nu_i)/T} + 1)
    fbar_i(k) = 1/(e^{(E*_ki + nu_i)/T} + 1)

the quantities the model needs are

    n_i       = (d_i/2pi^2)  INT_0^inf k^2 dk (f_i - fbar_i)
    n^s_i     = (d_i/2pi^2)  INT_0^inf k^2 dk (m*_i/E*_ki)(f_i + fbar_i)
    eps^kin_i = (d_i/2pi^2)  INT_0^inf k^2 dk E*_ki (f_i + fbar_i)
    P^kin_i   = (d_i/6pi^2)  INT_0^inf dk (k^4/E*_ki)(f_i + fbar_i)
    s_i       = -(d_i/2pi^2) INT_0^inf k^2 dk
                  [ f_i ln f_i + (1-f_i) ln(1-f_i)
                  + fbar_i ln fbar_i + (1-fbar_i) ln(1-fbar_i) ]

At T = 0 the antiparticle term vanishes, f_i becomes a step at the Fermi
momentum `k_Fi = sqrt(nu_i^2 - m*_i^2)` (the species is absent where
nu_i <= m*_i), and the integrals are elementary, with
`E*_Fi = sqrt(k_Fi^2 + m*_i^2)`:

    n_i       = d_i k_Fi^3 / (6 pi^2)
    n^s_i     = (d_i m*_i / 4pi^2)
                [ k_Fi E*_Fi - m*_i^2 ln((k_Fi + E*_Fi)/m*_i) ]
    eps^kin_i = (d_i/16pi^2)
                [ k_Fi E*_Fi (2 k_Fi^2 + m*_i^2)
                  - m*_i^4 ln((k_Fi + E*_Fi)/m*_i) ]
    P^kin_i   = (d_i/48pi^2)
                [ k_Fi E*_Fi (2 k_Fi^2 - 3 m*_i^2)
                  + 3 m*_i^4 ln((k_Fi + E*_Fi)/m*_i) ]
    s_i       = 0

At T > 0 the integrals are evaluated with the Johns-Ellis-Lattimer
approximation in `eos.general.fermi_integrals`, the repository's single home for
them; the model implements none of its own. Two identities are used rather than
integrated, and are how `n^s_i` and `s_i` are actually obtained:

    n^s_i = (eps^kin_i - 3 P^kin_i)/m*_i        the trace of T^munu
    s_i   = (eps^kin_i + P^kin_i - nu_i n_i)/T

### The totals

The matter sector — baryons plus any thermal meson gas — assembles as

    eps^matter = sum_i eps^kin_i
                 + 1/2 (m_sigma^2 sigma^2 + m_omega^2 omega^2
                        + m_phi^2 phi^2 + m_rho^2 rho^2)
                 + eps^gas

    P^matter   = sum_i P^kin_i
                 + 1/2 (-m_sigma^2 sigma^2 + m_omega^2 omega^2
                        + m_phi^2 phi^2 + m_rho^2 rho^2)
                 + n_B Sigma^r + P^gas

    s^matter   = sum_i s_i + s^gas

    sum_i mu_i n_i = mu_B n_B + mu_C n_C + mu_S n_S + sum_j mu*_j n_j |_gas

**What differs between P and eps.** The scalar field enters P with a MINUS sign
and the vectors with a plus, while all four enter eps with a plus; `n_B Sigma^r`
enters P and NOT eps; the Sigma^t term is absent from both by the cancellation
above. The conserved-charge densities are

    n_B = sum_i n_i
    n_C = sum_i C_i n_i + n_C^gas
    n_S = sum_i S_i n_i + n_S^gas

summed through the quantum numbers of `eos.general.particles`, and the fractions
reported are `Y_C = n_C/n_B` and `Y_S = n_S/n_B` — the TOTAL non-leptonic
fractions, meson gas included, which is what the fixed-fraction conditions are
stated in terms of.

The state satisfies the Euler (Hugenholtz-Van Hove) identity

    eps + P = T s + sum_i mu_i n_i

to ~1e-14 relative at every solved point, the sharpest test that both
rearrangement terms are in the right places.

### The thermal meson gas, written out

An ideal Bose gas of one-body excitations riding on the mean fields (Lavagno
2010). What is DID's is the arithmetic of the three effective potentials:

    mu*_pi+ = mu_C - g_rhoN rho
    mu*_K+  = mu_C - mu_S - (g_omegaN - g_omegaLambda) omega - 1/2 g_rhoN rho
    mu*_K0  =      - mu_S - (g_omegaN - g_omegaLambda) omega + 1/2 g_rhoN rho

The kaon's omega shift is `(g_omegaN - g_omegaLambda)`: under the additive quark
picture the kaon couples through its one light quark and the Lambda coupling
supplies the strange-sector piece. No rearrangement term enters any mu*_j — the
gas carries no baryon number and no tau_3-weighted source, so it is a spectator
to Sigma^r and Sigma^t. Particle and antiparticle are SEPARATE species with
conjugated potentials; strangeless neutral mesons sit at mu* = 0:

    j        mu*_j        m_j [MeV]   C_j   S_j   g_j
    pi+      mu*_pi+       139.57039   +1     0    1
    pi-     -mu*_pi+       139.57039   -1     0    1
    pi0      0             134.9768     0     0    1
    K+       mu*_K+        493.677     +1    -1    1
    K-      -mu*_K+        493.677     -1    +1    1
    K0       mu*_K0        497.611      0    -1    1
    K0bar   -mu*_K0        497.611      0    +1    1
    eta      0             547.862      0     0    1
    eta'     0             957.78       0     0    1

Strangeness follows S = +1 per s quark, so K+ = u sbar carries S = -1. The
isospin partners carry their SEPARATE physical masses: the 4.6 MeV
pi+- / pi0 splitting alone moves the pi0 density by some 17% at T = 30 MeV. The
vector nonet is not wired in DID — its `thermal_mesons` flag is the pseudoscalar
gas.

With `E_kj = sqrt(k^2 + m_j^2)` and `b_j(k) = 1/(e^{(E_kj - mu*_j)/T} - 1)`:

    n_j       = (g_j/2pi^2) INT_0^inf k^2 dk b_j(k)
    eps^gas_j = (g_j/2pi^2) INT_0^inf k^2 dk E_kj b_j(k)
    P^gas_j   = (g_j/6pi^2) INT_0^inf dk (k^4/E_kj) b_j(k)
    s^gas_j   = (eps^gas_j + P^gas_j - mu*_j n_j)/T

evaluated with the Bose branch of the JEL approximation in
`eos.general.bose_integrals`. The gas contributes

    P^gas = sum_j P^gas_j       eps^gas = sum_j eps^gas_j
    s^gas = sum_j s^gas_j       n_C^gas = sum_j C_j n_j
    n_S^gas = sum_j S_j n_j     and sum_j mu*_j n_j to the Euler sum

The charge and strangeness sums are the point: they enter n_C and n_S, and hence
the neutrality and fixed-fraction rows, not only eps, P and s. Every solved
point reports

    condensation = max_j |mu*_j| / m_j

and a point at which it reaches 1 is returned as NOT converged: the species
Bose-condenses there and the ideal-gas expressions stop describing it.
Condensates are not implemented.

### Leptons, photons and thermal neutrinos, written out

None feel the strong mean fields, and all come from `eos.general`.

**Leptons** are the SAME ideal Fermi gas as above with `m*_i -> m_i` and
`nu_i -> mu_i`: `m_e = 0.510999` MeV and `m_mu = 105.6584` MeV at d = 2,
massless neutrinos at d = 1, all with antiparticles, and the same T = 0 closed
forms under the same substitution. In a beta-equilibrium mode

    mu_e  = mu_nue - mu_C          (mu_C + mu_e = 0 when neutrinos free-stream)
    mu_mu = mu_e - mu_nue + mu_numu

with `mu_numu = 0`: only the electron family is ever trapped. In a fixed-Y_C
mode with `leptons=True` they are populated AFTER the solve at the single
potential that makes `n_e + n_mu = n_C`. The paper carries electrons only; the
muon family is an option of this implementation and moves the hyperon onsets up
by about 0.03 fm^-3.

**Photons**, g = 2 polarisations at mu = 0, carrying no conserved charge:

    P_gamma = pi^2 T^4 / (45 (hbar c)^3) = eps_gamma/3
    s_gamma = 4 pi^2 T^3 / (45 (hbar c)^3) = 4 P_gamma / T
    n_gamma = 2 zeta(3) T^3 / (pi^2 (hbar c)^3)

**Thermal neutrinos** are the flavours the composition does NOT track: three
mu = 0 massless Fermi gases at d = 1 each, contributing to eps, P and s only.
Requesting them together with a trapped electron family raises — nu_e would be
counted twice.

## The solve

### The unknown vector

    x = [sigma, omega, rho, phi, beta, Sigma^t, mu~_B, mu_C,
         (mu_S), (mu_nue), (T)]

Six entries are present in every mode: the four mean fields, beta and Sigma^t.
The last two are what distinguishes this model's solve from an ordinary DD-RMF's
— they cannot be evaluated inside a residual, because the couplings depend on
beta and Sigma^t shifts the very potentials whose densities define both, so each
is carried as an unknown with its defining equation as a row, which is what
keeps the residual an explicit function of x. mu_C is an unknown in every mode;
what a mode changes is the row that closes it. mu_S and mu_nue join where
strangeness is held or the electron family is trapped, and T joins where an
entropy per baryon is imposed in its place.

### The rows

In the order `residual()` assembles them, with `n_B^target` the density asked
for:

    R1  sigma - (1/m_sigma^2) sum_i g_sigma i n^s_i        = 0
    R2  omega - (1/m_omega^2) sum_i g_omega i n_i          = 0
    R3  rho   - (1/m_rho^2)   sum_i g_rho i tau_3i n_i     = 0
    R4  phi   - (1/m_phi^2)   sum_i g_phi i n_i            = 0
    R5  beta - (1/n_B^target) sum_i tau_3i n_i             = 0
    R6  Sigma^t - (1/n_B^target) sum_i [ -(dg_sigma i/dbeta) sigma n^s_i
                                         + (dg_omega i/dbeta) omega n_i
                                         + (dg_phi i/dbeta)   phi   n_i
                                         + (dg_rho i/dbeta)   tau_3i rho n_i ]
                                                           = 0
    R7  (1/n_B^target) sum_i n_i - 1                       = 0
    R8  (n_C - n_e - n_mu)/n_B^target = 0      where C equilibrates (neutrality)
        n_C/n_B^target - Y_C           = 0      where C is held
    R9  n_S/n_B^target - Y_S           = 0      iff S is held
    R10 (n_e + n_nue)/n_B^target - Y_Le = 0     iff the e family is trapped
    R11 s^total/n_B^target - S/A        = 0     iff an entropy per baryon is
                                                imposed

The four field rows are present in every mode: a mean field does not know what
is being held fixed. Each row is divided by the scale of the quantity it
balances — a field gap by the typical field 30 MeV, a density row by
n_B^target — so that one tolerance means the same thing for all of them; the
state is accepted when the largest scaled row is below 1e-10, the one gate the
whole repository uses. Non-convergence is a value carried on the returned
record, never an exception and never an unbounded loop.

The entropy row deserves one remark. In a fixed-Y_C mode with neutralizing
leptons, the leptons enter no field equation and are populated after the solve —
but they carry entropy, so R11 must see them, and it does. Without that, the
temperature returned is the one at which the MATTER alone carries the requested
entropy, which is a different state.

### The modes

    mode                        independent vars    extra unknowns  extra rows
    beta_eq_neutrinoless        (n_B, T)            -               R8 neutrality
    beta_eq_neutrino_trapped    (n_B, Y_Le, T)      mu_nue          R8 neutrality, R10
    fixed_YC                    (n_B, Y_C, T)       -               R8 held Y_C
    fixed_YC_YS                 (n_B, Y_C, Y_S, T)  mu_S            R8 held Y_C, R9

Strangeness equilibrates (mu_S = 0) except where it is held; neutrinos
free-stream (mu_nue = 0) except where the electron family is trapped. The
fixed-fraction modes carry a `leptons` flag: with it, neutralizing leptons make
the total electrically neutral; without it the matter is charged, which is what
a mixed-phase construction needs per pure phase. Anywhere T appears, an entropy
per baryon may be given instead, which adds T to the unknown vector and R11 to
the rows.

`fixed_YC_YS` RAISES without a strange degree of freedom: with nucleons only
n_S = 0 identically, and R9 is satisfied at Y_S = 0 for any mu_S and
unsatisfiable otherwise, so the potential returned would be wherever the
solver's path happened to end.

### What one solved point carries

Every quantity the model returns, so nothing below has to be re-derived by a
consumer that has only the densities.

- **The state asked for:** n_B and T (the T solved for, where an entropy per
  baryon was imposed).
- **The model's own variables:** sigma, omega, rho, phi [MeV], beta, and both
  rearrangement self-energies Sigma^r, Sigma^t [MeV].
- **Potentials:** mu_B, mu_C = mu_p - mu_n, mu_S, mu_e, mu_nue, and per species
  the effective potential nu_i and effective mass m*_i. The full mu_i is not
  stored: it is `mu_B + C_i mu_C + S_i mu_S` and is derived on demand, so the
  charge basis cannot drift out of step with the species potentials. In a
  leptonless fixed-Y_C solve mu_e is reported from `mu_e = -mu_C` as a
  diagnostic; there is no lepton sector there.
- **Thermodynamics:** eps, P, s — totals, including the leptons, photons and any
  thermal gas the flags enabled — with P split into its hadronic, leptonic and
  photonic parts, and the free energy density f = eps - T s.
- **Composition:** n_i for every active baryon, n_e, n_mu, n_nue, and the
  conserved-charge densities n_C, n_S with the fractions Y_C, Y_S, Y_Le, Y_e and
  S/A = s/n_B. n_C and n_S are TOTAL and non-leptonic: baryons plus the meson
  gas.
- **Scalar densities** n^s_i, through the trace identity rather than integrated,
  and summed into the sigma source.
- **Status:** `converged` (the scaled-residual gate), `error` (the largest
  scaled row), the Hugenholtz-Van Hove residual
  `(eps + P - T s - sum_i mu_i n_i)/eps` as a standing diagnostic, and the gas
  condensation ratio.

### Tables and continuation

A table is a set of lines — one per temperature (or entropy per baryon) and per
combination of the fractions the mode fixes — each swept along the baryon
density with a warm start, each solved point seeding the next. That loop is
`eos.general.tabulate`, shared with the other models; what this subpackage
supplies is which solve a mode name means, what part of a solved point becomes
the next guess, and how a point flattens into a row. A missed step is bisected
back towards the last solved point, up to six times: at T = 0 a hyperon
threshold crossed inside one grid interval leaves the previous answer outside
the new basin, and halving the interval gives the continuation a seed on the far
side.

## Nuclear matter

Nuclear-matter parameters are computed as PREDICTIONS of the couplings: the
model's parameters are the output of a Bayesian analysis over 18 observables,
not of an inversion of a fixed list of saturation properties, so only the
forward map is implemented. The binding energy per baryon is

    B(n_B, beta) = (eps - sum_i m_i n_i)/n_B

with the rest mass following the COMPOSITION — subtracting a single average
nucleon mass instead would leave a term `-1/2 (m_n - m_p) beta`, larger than the
quadratic symmetry term below beta ~ 0.04. With x = n_B/n_0 and derivatives at
x = 1:

    K = 9 d^2B/dx^2      Q = 27 d^3B/dx^3
    M(n_B) = 3 n_B d/dn_B [ 9 n_B^2 d^2B/dn_B^2 + 18 P/n_B ]

the last evaluated at 0.11 fm^-3, the mean density in the outer region of a
heavy nucleus. Two symmetry energies are reported, and they differ by more here
than in a conventional DD-RMF because the isospin dependence of the couplings
makes B genuinely non-quadratic in beta:

    B(beta) = B(0) + S_2 beta^2 + S_4 beta^4 + O(beta^6)
    S = B(-1) - B(0) ~ S_2 + S_4

with slopes `L_2 = 3 dS_2/dx`, `L = 3 dS/dx` and curvatures
`K_sym2 = 9 d^2S_2/dx^2`, `K_sym = 9 d^2S/dx^2`. S_2 is extracted by one
Richardson step on the NEUTRON-RICH side,

    S_2 = [4 f(-1/2) - f(-1)]/3,     f(beta) = [B(beta) - B(0)]/beta^2

proton-rich matter empties the neutron sector and leaves mu_C unpinned, and a
small-beta estimator is numerical noise: B(beta) - B(0) at beta = 0.05 is
0.08 MeV against a binding energy of 900 MeV.

    quantity          this implementation   paper (Table VI)
    n_0 [fm^-3]       0.158800              0.158800    imposed by P(n_0) = 0 in ISM
    B [MeV]           -15.402               -15.40
    K [MeV]           227.07                227.06
    Q [MeV]           -608.10               -608.09
    M(0.11) [MeV]     1122.69               1122.72
    S_2 [MeV]         32.12                 32.44       truncation-defined, see above
    S [MeV]           29.71                 29.72
    L [MeV]           59.85                 59.95
    K_sym [MeV]       -97.33                -97.32
    X_p^eq(n_0)       0.0336                0.0336

Asserted by `verify/run_full_check.py`.

## Response functions

`eos_response` returns the second-derivative quantities of the CompOSE manual. A
second derivative is only defined once one says what is held fixed, so the
conditioning is explicit and has three axes (CLAUDE.md §5): what composition is
held, which thermal variable is held, and whether leptons re-neutralize. DID
implements `frozen='equilibrium'` — NOTHING is held, the composition
re-equilibrates under the perturbation — and raises for any other freeze.
Whether the leptons re-neutralize is the mode's own choice and is inherited, not
a separate argument.

Every derivative is a central difference along the same mode's sequence with a
full re-solve at each neighbour: the reference flavour of CLAUDE.md §9, and DID
ships no accelerated one. With `sigma = s/n_B` the entropy PER BARYON — per
baryon rather than per volume because C_P is taken at fixed pressure, where the
volume moves — and steps `dn_B = 1e-3 n_B`, `dT = 0.05` MeV:

    (dP/dn_B)_T   (deps/dn_B)_T   (dsigma/dn_B)_T   (dP/dT)_n   (dsigma/dT)_n

the last two zero at T = 0, where there is no cold side to difference against.
From them

    cs2_isothermal = (dP/deps)_T = (dP/dn_B)_T / (deps/dn_B)_T

    C_V = T (dsigma/dT)_n
    C_P = T (dsigma/dT)_P
        = T [ (dsigma/dT)_n - (dP/dT)_n (dsigma/dn_B)_T / (dP/dn_B)_T ]

    cs2_adiabatic = (C_P/C_V) cs2_isothermal
    Gamma_th      = 1 + (P - P_cold)/(eps - eps_cold)

both heat capacities per baryon and therefore dimensionless. C_P is the usual
Jacobian rotation, with every partial taken along the same re-solved sequence.
The cold reference of Gamma_th is the SAME mode at T = 0 and the same n_B, which
is what a simulation's thermal-pressure prescription is calibrated against; it
is returned as nan where `eps - eps_cold <= 0`. At T = 0 the ratio C_P/C_V is
unity by construction and the two sound speeds coincide, so the cold limit needs
no special case and only c_s^2 is returned. Both are named for the thermal
variable they hold: there is no bare `cs2` whose meaning would depend on the
arguments.

Not implemented, and recorded in `docs/DEFERRED.md`: the composition freezes
(held Y_i, held Y_C) and the susceptibility matrix
`chi_ab = dn_a/dmu_b` for a, b in (B, C, S). A freeze DID does not carry raises
rather than silently falling back to `equilibrium`.

## What the implementation reproduces

Every published number of arXiv:2511.15646 that does not require its
nuclear-statistical-equilibrium crust, from the transcribed Table II parameters:

| quantity | this implementation | paper |
|---|---|---|
| P(n_0) in ISM | 2e-7 | 0 (the calibration condition) |
| P_NM(0.08), P_NM(0.16) | 0.4569, 3.2334 | 0.4569, 3.233 |
| P_ISM(0.32), P_ISM(0.56) | 12.107, 109.019 | 12.11, 109.0 |
| B, K, Q | -15.402, 227.07, -608.10 | -15.40, 227.06, -608.09 |
| S, L, K_sym | 29.71, 59.85, -97.33 | 29.72, 59.95, -97.32 |
| X_p at n_0 | 0.0336 | 0.0336 |
| U_Y, ISM and NM (12 values) | all within 0.01 MeV | Table IV |
| hyperon onsets (DIDY) | 0.470, 0.578, 0.978, no Xi0 | Table VII |
| c_s^2 peak | 0.706 at 0.70 fm^-3 | ~0.71 at ~0.66 (Fig. 8) |
| M_max DID / DIDY | 2.243 / 2.196 | 2.245 / 2.196 |
| R_1.4 | 12.07 km (BPS crust) | 11.99 km (their NSE crust) |

The hyperon potentials are the single strongest check: they exercise the SU(3)
vector unpacking, the fitted scalar and isovector hyperon couplings, BOTH
rearrangement terms and the tau_3 = 2 I_3 normalisation at once. The Sigma
splitting in neutron matter, `U_Sigma- - U_Sigma+ = 17.9` MeV, is carried almost
entirely by Sigma^t — the rho contribution is negligible, `g_rhoSigma = 0.0055`.
The onsets are the paper's own lepton sector (electrons only) and the inverted
hierarchy Sigma- before Lambda before Xi- is the model's headline result. The
R_1.4 difference is the crust: the paper uses its own NSE crust below saturation
and this comparison attaches BPS.

`python -m eos.did.verify.run_full_check` asserts all of it except the last two
rows; add `--tov` for those (it costs ~40 s).

## The phase-adapter surface

For the hybrid engine `eos/mixed`, DID is presented through the phase-adapter
contract: a map from `(mu~_B, mu_C, mu_S, T)` to a solved block, with the
phase's own density an unknown — in a mixture only the volume average is
prescribed, never each phase's own. That solve carries the seven unknowns

    [sigma, omega, rho, phi, beta, Sigma^t, n_B]

against the four field equations, the definitions of beta and Sigma^t, and
`n_B = sum_i n_i`. There is no charge, strangeness or neutrality row: the
potentials are inputs, which is what makes this thermodynamics rather than a
mode.

The slot carries the KINETIC potential `mu~_B = mu_B - Sigma^r`, as DD2's does
and for the same reason. The second rearrangement term is not absorbed that way
and stays inside the phase, which the contract allows: what crosses it is the
conserved-charge decomposition `mu_i = mu_B + C_i mu_C + S_i mu_S`, and mu_B is
restored at assembly.

Two properties of this surface are physics rather than implementation. First,
the map `mu~_B -> n_B` is not one-to-one below saturation, where the isotherm is
mechanically unstable; the seed chooses the root, and the adapter's seed is a
PURE function of its arguments so that a finite-difference Jacobian of the mixed
residual stays the derivative of something. Second, a mixed-phase construction
needs each pure phase at fixed Y_C WITHOUT leptons, which is the
`leptons=False` flavour above.

## Species flags

`hyperons`, `deltas`, `muons`, `thermal_mesons`, `thermal_neutrinos`, `photons`,
`phi_field`. Nucleons are always present. Setting a flag DID does not implement
RAISES; a NotImplementedError is never turned into a silent no-op —
`phi_field=False` is the one DID refuses outright, because the SU(3) sector
gives the nucleon a phi coupling and dropping the field would change the model,
not switch off a sector.

## Usage

```python
from eos.did import Parameters, SpeciesFlags, eos_point, eos_table

par   = Parameters.default()
didy  = SpeciesFlags(hyperons=True, muons=False)

r = eos_point(par, "beta_eq_neutrinoless", didy, n_B=0.66, T=0.0)
r.ok, r.point.P, r.point.Y("Lambda")

table = eos_table(par, "fixed_YC", didy,
                  axes={"nB": np.linspace(0.05, 1.2, 120),
                        "T": [0.0, 10.0, 30.0], "Y_C": [0.1, 0.3, 0.5]},
                  leptons=True, verbose=True)
```

## Layout

    couplings.py       the functional form: F_M(x), the blend, SU(3), no numbers
    parameters.py      the numbers, the multiplet table, couplings_at(n_B, beta)
    species.py         SpeciesFlags and the active baryon list
    thermodynamics.py  kinetics, fields, both Sigma terms, assembly, and
                       thermo_from_mu (the phase-adapter surface)
    solver.py          the residual, the modes, the warm start
    nmp.py             the forward nuclear-matter map + the Delta inversion
    table.py           the grid driver over eos.general.tabulate
    responses.py       c_s^2 (isothermal and adiabatic), C_V, C_P, Gamma_th
    api.py             eos_point / eos_table / eos_response
    verify/            the physics invariants, and the TOV cross-check

## What is not implemented

Recorded in `docs/DEFERRED.md` and repeated here so the document stands alone:
the nuclear-statistical-equilibrium description of nuclear clusters below
saturation (Section III of the paper), for which the repository attaches a crust
table instead; the inverse nuclear-matter map, which is not published for this
functional form; the delta and f0(980) mesons, deliberately absent from the
model; a trapped MUON family; the composition freezes and chi_ab of
`eos_response`; and the phi contribution to the thermal kaon effective
potentials, which is inherited from a model in which g_phiN = 0 and is not one
here.
