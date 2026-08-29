# DD2 — density-dependent relativistic mean field

`dd2.tex` is the same description written for LaTeX, with the bibliography;
this file carries the same physics in plain text. Either one alone is enough to
reproduce the model. Where a document and the source differ, **the source
decides**.

**Model.** DD-RMF of Typel et al., PRC 81, 015803 (2010): nucleons (plus,
optionally, the hyperon octet and the Delta quartet) exchange sigma, omega,
rho and — for strange baryons — the hidden-strange phi. The Lagrangian:

    L = sum_i psibar_i [ gamma_mu ( i d^mu - Gamma_omega_i omega^mu
                                    - Gamma_rho_i t3_i rho^mu
                                    - Gamma_phi_i phi^mu )
                         - ( m_i - Gamma_sigma_i sigma ) ] psi_i
      + 1/2 ( d_mu sigma d^mu sigma - m_sigma^2 sigma^2 )
      - 1/4 W_munu W^munu + 1/2 m_omega^2 omega_mu omega^mu
      - 1/4 R_munu R^munu + 1/2 m_rho^2   rho_mu   rho^mu
      - 1/4 F_munu F^munu + 1/2 m_phi^2   phi_mu   phi^mu

with `t3_i` the third isospin component in the **`tau_3 = +-1` convention**
(`t3_p = +1`, `t3_n = -1`), so the published DD2 rho coupling is used as
tabulated. Leptons (e, mu) and, in trapped modes, electron neutrinos enter as
free Dirac gases; photons as a free Bose gas.

The couplings depend on the total baryon density:

    Gamma_i(n_B) = Gamma_i(n_sat) f_i(x),  x = n_B/n_sat
    f_{sigma,omega}(x) = a (1 + b(x+d)^2)/(1 + c(x+d)^2)      (rational)
    f_rho(x)           = exp[-a_rho (x-1)]                     (exponential)

with `f_i(1) = 1` and `f_i''(0) = 0` making `a_i` and `d_i` dependent.

**Parameters.** The published DD2 set, `Parameters.default()`, all fields of a
frozen dataclass and arguments everywhere:

| symbol | code | value | | symbol | code | value |
|---|---|---|---|---|---|---|
| `n_sat` | `n_sat` | 0.149065 fm^-3 | | `Gamma_sigma(n_sat)` | `gamma_sigma` | 10.686681 |
| `m_n` | `m_n` | 939.56536 MeV | | `a_sigma` | `a_sigma` | 1.357630 |
| `m_p` | `m_p` | 938.27203 MeV | | `b_sigma` | `b_sigma` | 0.634442 |
| `m_sigma` | `m_sigma` | 546.212459 MeV | | `c_sigma` | `c_sigma` | 1.005358 |
| `m_omega` | `m_omega` | 783.0 MeV | | `d_sigma` | `d_sigma` | 0.575810 |
| `m_rho` | `m_rho` | 763.0 MeV | | `Gamma_omega(n_sat)` | `gamma_omega` | 13.342362 |
| `m_phi` | `m_phi` | 1019.45 MeV | | `a_omega` | `a_omega` | 1.369718 |
| `U_Lambda` | `U_Lambda` | -30.0 MeV | | `b_omega` | `b_omega` | 0.496475 |
| `U_Sigma` | `U_Sigma` | +30.0 MeV | | `c_omega` | `c_omega` | 0.817753 |
| `U_Xi` | `U_Xi` | -18.0 MeV | | `d_omega` | `d_omega` | 0.638452 |
| `x_Delta_sigma/omega/rho` | | 1.0 (universal) | | `Gamma_rho(n_sat)` | `gamma_rho` | 3.626940 |
| | | | | `a_rho` | `a_rho` | 0.518903 |

The reference is Typel et al., PRC 81, 015803 (2010); the density-dependent
form is Typel & Wolter, NPA 656, 331 (1999). `nucleon_mass_mode='average'`
selects which nucleon mass the dimensionless residual scales against.

**Three routes to a parameter set.** CLAUDE.md section 6 makes model
parameters arguments, so all three have to exist. *By name:*
`Parameters.default()` is the published DD2 table above, and
`Parameters.named('DD2')` / `named('DD2Y')` take a published set by name, an
unknown name raising `KeyError` that lists what there is. *A new set:*
`dataclasses.replace(Parameters.default(), gamma_rho=...)`, or
`Parameters.from_microscopic(...)`, which derives the omitted shape
coefficients from the internal constraints and validates any that are
supplied. Eighteen of the twenty-seven fields carry no default, so bare
field-by-field construction means supplying all eighteen -- deliberate for a
DD-RMF, where a coupling is meaningless without the four shape coefficients
that go with it. *From nuclear-matter parameters:* `nmp.from_nmp` and
`nmp.invert_nmp`, imposing `{n_sat, E_sat, m*/m, K_sat, E_sym, L_sym}`.

**Masses**, from the shared particle table rather than from `Parameters` (MeV,
with the degeneracy `g_i`):

    n     939.56536  g=2      Xi-      1321.71   g=2
    p     938.27203  g=2      Xi0      1314.86   g=2
    Lambda 1115.683  g=2      Delta    1232.00   g=4  (spin 3/2)
    Sigma- 1197.449  g=2      e-          0.510999 g=2
    Sigma0 1192.642  g=2      mu-       105.6584  g=2
    Sigma+ 1189.370  g=2

**The rearrangement self-energy.** The density dependence produces `Sigma^R`,
the same for every baryon, because `Gamma_i` depends on the TOTAL `n_B`:

    Sigma^R = (dGamma_omega/dn_B) omega0 sum_i x_omega_i n_i
            + (dGamma_rho/dn_B)   rho0   sum_i x_rho_i t3_i n_i
            + (dGamma_omega/dn_B) phi0   sum_i x_phi_i n_i
            - (dGamma_sigma/dn_B) sigma  sum_i x_sigma_i ns_i

It enters the chemical potentials and the pressure and **never the energy
density** — that is what makes the model thermodynamically consistent, and it
is the term §11 forbids naming without defining. Every assembled state is
checked against the Hugenholtz–Van Hove identity
`eps + P - T s = sum_i mu_i n_i` at 1e-8.

**Extensions.** Hyperons: the vector ratios are SU(6) times a free factor,
`x_MY = y_MY * SU(6)`, nine of them over M = omega, rho, phi and
Y = Lambda, Sigma, Xi, each an ordinary parameter defaulting to 1 (= SU(6),
which is DD2Y). Nine and not three because a published set may break omega and
phi while leaving rho alone — SFHoY does exactly that — and `y_phi_*`
multiplies a NEGATIVE ratio, so a factor above one makes g_phiY more negative;
`y_phi_* = 0` in every multiplet is a set with no phi sector, which is how that
sector is switched off. Scalar ratios are either the published DD2Y values
(Marques et al. 2017; Fortin et al. 2017) or inverted from the potentials
U_Lambda, U_Sigma, U_Xi in saturated symmetric matter — and that inversion runs
AFTER the rescaling, since U_Y holds the scalar and vector couplings together,
so a rescaled x_omegaY changes the x_sigmaY that reproduces the same depth.
The phi inherits the omega density dependence. Deltas take no SU(6) factors:
x_Delta_sigma, x_Delta_omega, x_Delta_rho are free variables directly,
defaulting to universal coupling, or x_Delta_sigma comes from U_Delta. Thermal pi/K (and
optionally the vector nonet) as Bose gases whose effective potentials are
shifted by the same vector mean fields (Lavagno 2010; arXiv:1210.0400); the
gas contributes charge and strangeness to the equilibrium constraints, no
baryon number, and no field sources.

**Field equations.** Algebraic in the sources, so the fields are eliminated
against them:

    m_sigma^2 sigma = sum_i Gamma_sigma_i ns_i
    m_omega^2 omega0 = sum_i Gamma_omega_i n_i
    m_rho^2   rho0   = sum_i Gamma_rho_i t3_i n_i
    m_phi^2   phi0   = sum_i Gamma_phi_i n_i

with `m*_i = m_i - Gamma_sigma_i sigma` and
`mu_eff_i = mu_i - Gamma_omega_i omega0 - Gamma_rho_i t3_i rho0
            - Gamma_phi_i phi0 - Sigma^R`.

**One species as an ideal gas.** Each baryon is a Fermi gas of mass `m*_i` at
`mu_eff_i`, degeneracy g_i (2 for the octet, 4 for the Delta), antiparticles
included. With `E = sqrt(k^2 + m*_i^2)` and
`f± = 1/(1 + exp((E ∓ mu_eff_i)/T))`:

    n_i       = g_i/(2 pi^2 hc^3) ∫dk k^2       (f+ - f-)
    eps_kin_i = g_i/(2 pi^2 hc^3) ∫dk k^2 E     (f+ + f-)
    P_kin_i   = g_i/(6 pi^2 hc^3) ∫dk k^4 / E   (f+ + f-)

The other two are NOT integrated — they come from the trace of the
energy-momentum tensor and the one-species Euler relation:

    ns_i = (eps_kin_i - 3 P_kin_i) / m*_i
    s_i  = (eps_kin_i + P_kin_i - mu_eff_i n_i) / T

which matters: an error in eps_kin_i or P_kin_i propagates into ns_i, and ns_i
sources the sigma field, so it does not stay confined to the totals.

At T = 0 the integrals are elementary. With
`kF = sqrt(mu_eff_i^2 - m*_i^2)` (everything vanishing when
`|mu_eff_i| <= m*_i`) and `L = ln[(kF + |mu_eff_i|)/m*_i]`:

    n_i       = sgn(mu_eff_i) g_i kF^3 / (6 pi^2 hc^3)

    ns_i      = g_i m*_i / (4 pi^2 hc^3) * ( kF |mu_eff_i| - m*_i^2 L )

    P_kin_i   = g_i/(48 pi^2 hc^3) [ (2 kF^3 - 3 m*_i^2 kF) |mu_eff_i|
                                     + 3 m*_i^4 L ]

    eps_kin_i = g_i/(16 pi^2 hc^3) [ (2 kF^3 + m*_i^2 kF) |mu_eff_i|
                                     - m*_i^4 L ]

    s_i       = 0

and that branch is also Numba-compiled. At T > 0 the integrals are the
Johns-Ellis-Lattimer approximants from `eos/general/fermi_integrals` (~1e-4
accurate), with a Gauss-Laguerre quadrature there as the accuracy reference.

**The thermal meson gas is the same expressions with Bose statistics.** Each
species is an ideal Bose gas of mass `m_j`, degeneracy `g_j` and effective
potential `mu*_j`, its ANTIPARTICLE carried as a separate species at `-mu*_j`
rather than through an antiparticle term inside the integral. With
`b(x) = 1/(exp(x/T) - 1)`:

    n_j   = g_j/(2 pi^2 hc^3) INT dk k^2       b(E_k - mu*_j)
    P_j   = g_j/(6 pi^2 hc^3) INT dk k^4/E_k   b(E_k - mu*_j)
    eps_j = g_j/(2 pi^2 hc^3) INT dk k^2 E_k   b(E_k - mu*_j)
    s_j   = (eps_j + P_j - mu*_j n_j) / T

— the one-species Euler relation again, at the EFFECTIVE potential, which is
why the gas joins the HVH sum through `sum_j mu*_j n_j` and not through a
physical potential. From `eos/general/bose_integrals`. There is no T = 0
branch: the gas is defined only at T > 0 and returns zero below it.

The active species (mass in MeV, charge `Q_j`, strangeness `S_j` under
`S = +1` per s quark, degeneracy `g_j`):

    pi+   139.57039  +1   0  1     rho+     775.26  +1   0  3
    pi-   139.57039  -1   0  1     rho-     775.26  -1   0  3
    pi0   134.9768    0   0  1     rho0     775.26   0   0  3
    K+    493.677    +1  -1  1     omega    782.66   0   0  3
    K-    493.677    -1  +1  1     K*+      891.67  +1  -1  3
    K0    497.611     0  -1  1     K*-      891.67  -1  +1  3
    K0bar 497.611     0  +1  1     K*0      891.67   0  -1  3
    eta   547.862     0   0  1     K*0bar   891.67   0  +1  3
    eta'  957.78      0   0  1     phi     1019.461  0   0  3

The left block is `thermal_mesons`; the right is the optional vector nonet,
which reuses the same three potentials because the shift depends on the quark
content and vector and pseudoscalar partners share it. `g = 1` for a
pseudoscalar, `g = 3` for a vector.

**Condensation is refused, not approximated.** These expressions describe the
gas only while `|mu*_j| < m_j`. At `|mu*_j| = m_j` the species condenses: the
particles beyond the critical density go into the `k = 0` state, carrying
charge and `eps = m_j n_cond` with NO pressure and NO entropy, and `n_cond`
becomes a new unknown rather than a function of `mu*_j`. None of that is
implemented, so the package computes

    condensation = max_j |mu*_j| / m_j

and **refuses any state with `condensation >= 1`**, raising rather than
returning it. Not defensive tidiness: the underlying integral routine CAPS
`mu*` at `m`, so past the threshold the gas silently stops absorbing charge
and the entropy it reports drifts — turning negative by `|mu*|/m ~ 3`. A
returned state there would be wrong rather than approximate. `eos.sfho`
refuses the same condition, there as a status because its solvers report
rather than raise.

**The totals.**

    eps = sum_i eps_kin_i + (1/2)(m_s^2 s^2 + m_w^2 w^2 + m_r^2 r^2 + m_p^2 p^2)
          + eps_lep + eps_gamma + eps_mes
    P   = sum_i P_kin_i  + (1/2)(-m_s^2 s^2 + m_w^2 w^2 + m_r^2 r^2 + m_p^2 p^2)
          + n_B Sigma^R + P_lep + P_gamma + P_mes
    s   = sum_i s_i                    + s_lep + s_gamma + s_mes
    n_B = sum_i B_i n_i    n_C = sum_i Q_i n_i + n_C_mes
                           n_S = sum_i S_i n_i + n_S_mes

The mean fields carry no entropy, so `s` has no field term. The only
asymmetries between eps and P are the SIGN of the sigma mass term and the
rearrangement term `n_B Sigma^R` — the two places a mean-field model is most
easily got wrong. Photons: `P = pi^2 T^4/(45 hc^3)`, `eps = 3P`,
`s = 4 pi^2 T^3/(45 hc^3)`. The HVH sum takes baryons and leptons at their FULL
potentials and the meson gas at its EFFECTIVE ones.

**Solving.** One residual system for all modes over
`x = [sigma, omega0, rho0, (phi0), mu_B - Sigma^R, mu_C, (mu_S), (mu_nue)]`;
species potentials follow `mu_i = B_i mu_B + Q_i mu_C + S_i mu_S`, and the
solver works in the effective potentials `mu_eff_i = mu_i - Sigma0_i`, which
vary smoothly along density sweeps (that is what makes warm starts work). The
rows, in the order they are assembled, each divided by m_N or n_B so all are
dimensionless and O(1):

    R1..R4  field - source/m_M^2, for sigma, omega0, rho0, (phi0)
    R5      (sum_i B_i n_i - n_B) / n_B                            always
    R6      (n_C - n_e - n_mu)/n_B  (C equilibrated)  |
            (n_C - Y_C n_B)/n_B     (Y_C imposed)
    R7      (n_S - Y_S n_B) / n_B                        iff Y_S imposed
    R8      (n_e + n_nue - Y_Le n_B) / n_B   iff the electron family is trapped

n_C and n_S are the TOTALS, gas included; the gas carries no baryon number so
it is absent from R5. Per mode:

| mode | independent variables | closing constraints | potentials |
|---|---|---|---|
| `beta_eq_neutrinoless` | `(n_B, T)` | `sum_i Q_i n_i + n_C_mes - n_e - n_mu = 0` | `mu_S = mu_nue = 0` |
| `beta_eq_neutrino_trapped` | `(n_B, Y_Le, T)` | neutrality; `(n_e + n_nue)/n_B = Y_Le` | `mu_S = 0`; `mu_nue` unknown |
| `fixed_YC` | `(n_B, Y_C, T)` | `(sum_i Q_i n_i + n_C_mes)/n_B = Y_C` | `mu_S = mu_nue = 0` |
| `fixed_YC_YS` | `(n_B, Y_C, Y_S, T)` | the `Y_C` row; `(sum_i S_i n_i + n_S_mes)/n_B = Y_S` | `mu_S` unknown; `mu_nue = 0` |

In `fixed_YC` the flag `leptons` selects between the strongly-interacting
slice alone (electrically charged; what a mixed-phase construction consumes)
and the same slice plus neutralizing leptons (`mu_mu = mu_e`,
`n_e + n_mu = Y_C n_B`) — the leptons do not source the fields, so they close
post hoc through a single 1-D root. A temperature axis may be replaced by
entropy per baryon (outer 1-D solve for T, monotone at fixed `n_B` so the
bracket is well posed).

**Two more systems, neither of which is a mode.**

*The reduced nucleon-only beta system* has its own unknown vector, because
eliminating `omega0` collapses two unknowns into none:
`x = [sigma, rho0, mu_eff_n, mu_C]`, four unknowns and four rows. `omega0` is
eliminated algebraically at the target density — which works only while every
species shares the nucleon couplings, and is exactly why the octet system
cannot do it — and
`mu_eff_p = mu_eff_n + mu_C - (t3_p - t3_n) Gamma_rho rho0
= mu_eff_n + mu_C - 2 Gamma_rho rho0` under `tau_3 = +-1`. The rows, in the
order assembled and each already dimensionless:

    Rbar1 = [ sigma - Gamma_sigma (ns_n + ns_p) / m_sigma^2 ] / mbar
    Rbar2 = [ rho0  - Gamma_rho (t3_n n_n + t3_p n_p) / m_rho^2 ] / mbar
    Rbar3 = (n_n + n_p)/n_B - 1
    Rbar4 = (n_p - n_e - n_mu)/n_B

with `mbar` the mean nucleon mass, `mu_e = -mu_C` and `mu_mu = mu_e`. If
either effective mass goes non-positive the residual returns `[1e6, 0, 0, 0]`,
which pushes the root finder back inside the physical domain instead of
letting it evaluate a Fermi integral at an imaginary mass.

*The phase-adapter residual* is what `eos/mixed` consumes across the §5
contract. Given `(mu_tilde_B, mu_C, mu_S, T)` — the potentials as INPUTS — it
solves for

    x = [sigma, omega0, rho0, (phi0), n_B]

the field equations plus one baryon-density self-consistency row. **The
density is an unknown here**, unlike in every mode above, because for one
phase of a mixture only the AVERAGE density is prescribed; and because DD2's
couplings are density-dependent, `Gamma_i(n_B)` is re-evaluated at the current
`n_B` on every iteration, which is what makes this a coupled system rather
than a field solve at fixed couplings. **There is no charge, strangeness or
neutrality row**, and that absence is the content of the contract: the
potentials are given, so nothing here chooses a composition, and that is what
makes this thermodynamics rather than a mode. Imposing neutrality on one phase
of a Gibbs construction would be wrong in any case — global neutrality is the
engine's condition, not the phase's.

The entry point is `thermo_from_mu(par, flags, mu_tilde_B, mu_C, mu_S,
T, ...)` — the section 5 phase-adapter surface, which all ten models spell the
same way.

**NMPs.** Forward map at the model's own saturation: E_sat, m*/m, K_sat,
Q_sat, E_sym, L_sym (and K_sym). Inverse map imposes
{n_sat, E_sat, m*/m, K_sat, E_sym, L_sym}; the isoscalar sector closes by
pinning TWO shape coefficients, `b_sigma` and `c_omega`, at their published
values, and Q_sat / K_sym come back as predictions. Every derivative is
analytic — the sigma gap equation differentiated implicitly twice and
substituted into the closed-form E/A and E_sym, so nothing here is a finite
difference of a solved quantity. The inverter retries from jittered seeds
before declaring a target unrepresentable.

**There is no cross-constraint, and this is a correction.** Earlier versions
closed the sector with `f_sigma''(1) = f_omega''(1)`. That condition is the DD
parametrization's, not DD2's: Typel, *Phys. Rev. C* **71**, 064301 (2005),
Sec. IV imposes it alongside `f_i(1) = 1` and `f_i''(0) = 0` and counts eight
independent parameters, while Typel *et al.*, *Phys. Rev. C* **81**, 015803
(2010) — the DD2 paper — states only the latter two and counts ten. The
difference of one is exactly this constraint, and the published tables agree:
`f_sigma''(1) - f_omega''(1)` is -6.0e-08 for DD and 2.200718e-03 for DD2. The
constraint bound the INVERSE MAP ONLY — `Parameters.__post_init__` validates
`f_i(1) = 1` and `d_i = 1/sqrt(3 c_i)` and never checked it, so no forward
path ever saw it — and it is now gone from the inverse map too.

Two coefficients are pinned rather than one because E_sat and m*/m at fixed
n_sat are blind to the shape: only P and K_sat carry shape information among
the four rows, so four shape coefficients answer to two rows. `b_sigma` and
`c_omega` is the best of the six pairs by condition number (128, against 305
for holding the sigma shape whole and 354 for the omega shape), because what
should be left free is the least collinear surviving pair.

With the constraint removed the published couplings ARE a root of the closure:
all four rows vanish at the published table and a round trip through
`compute_nmp` returns it to 1.1e-05, where the old closure reached a root 3.9%
away. `InversionStatus` still reports `coupling_shift` — the max relative
distance from the seed — because "converged" and "recovered the published
couplings" remain different statements away from DD2's own point. A solve that
returns its seed unmoved on a nonzero residual is a Powell hybrid giving up on
its first step, not an answer, and comes back `ok=False`.

Imposing Q_sat instead of one pin is available and is now usable. It was not
while Q_sat was a third finite difference spanning 2.48 MeV over h in
[5e-5, 5e-4]: the five-row system conditions at 259, so the recovered
couplings inherited 259 x 1.5e-3 of relative error, and at DD2's own point
that closure reached only max|residual| = 1.4e-2, imposing Q_sat to 1.6 MeV.
With the derivative taken by hand the floor is gone and the amplification has
nothing to amplify — the same closure at the same point reaches 1.5e-12 and
imposes Q_sat to 1e-10 MeV, and perturbed targets over dK_sat in [-20, +10]
and dQ_sat in [-30, +100] MeV come back at the same order. The default
closure still stops at K_sat, because it imposes the nuclear-matter
parameters that are actually quoted and predicts the rest.

**What `eos_response` returns.** A second derivative is only defined once one
says what is held fixed, so the conditioning is an explicit argument. Two
freezes are implemented; every other combination raises
`NotImplementedError` naming itself.

`frozen='equilibrium'` — everything re-equilibrates. Wired for
`beta_eq_neutrinoless` only, and computed from the ANALYTIC Jacobian rather
than by differencing solved points:

    cs2_isothermal   c_s^2 = (dP/deps)_T along the mode's own sequence  always
    cs2_adiabatic    the same at fixed entropy per baryon,             always
                     = (C_P/C_V) cs2_isothermal
    chi      chi_ab = dn_a/dmu_b for a,b in (B, C, S)          always
    C_V      (T/n_B)(ds/dT)_n_B                                T > 0 only
    C_P      (T/n_B)(ds/dT)_P                                  T > 0 only

`C_V` and `C_P` are ABSENT at T = 0 rather than returned as zero: they are not
defined there, and a zero would be indistinguishable from a computed one.
`cs2_adiabatic` is present at T = 0 all the same, where the ratio C_P/C_V is 1
by construction and the two speeds coincide.

`frozen='composition'` — every particle fraction held fixed, i.e. reactions
slow compared with the perturbation. This is the freeze that takes a target:
the proton fraction `Y_p` is a named argument of `eos_response`, and it is a
**freeze target, not one of the mode's conditions** — which is why it is not
among `n_B, T, Y_C, Y_S, Y_Le`. It returns the same two sound speeds at that
proton fraction, by finite difference along the frozen-`Y_p` sequence, plus
the index `Gamma` = (eps + P)/P * `cs2_isothermal`, for nucleonic matter.

Raising: `frozen='equilibrium'` in any mode but `beta_eq_neutrinoless`; the
frozen conserved fractions; the leptonic re-neutralization variants.

The conditioning of a second derivative has two axes here, and each is carried
by its own thing, so no word does double duty. The COMPOSITION axis is the
`frozen=` argument: `frozen='equilibrium'` differentiates along the mode's own
sequence, `frozen='composition'` at frozen particle fractions. The THERMAL
axis is the key name: `cs2_isothermal` holds `T`, `cs2_adiabatic` holds the
entropy per baryon, and nothing here is called a bare `cs2`. Both freezes
return both speeds — four numbers, two keys, two arguments.

The word itself is why this matters. In asteroseismology — Zhao & Lattimer,
arXiv:2204.03037, Eq. (1) — "the adiabatic sound speed" `c_s` means FROZEN
COMPOSITION, and the g-mode frequency is the difference between it and the
equilibrium `c_e`. In the CompOSE manual (Typel et al.) "adiabatic" means
FIXED ENTROPY. This library serves both literatures, so the word is never
used unqualified: the composition is said by the argument and the entropy by
the key.

One gap remains, recorded in `docs/DEFERRED.md`: `Gamma` is built on
`cs2_isothermal`, so at T > 0 it is the isothermal index and the fixed-entropy
one is larger by the same `C_P/C_V`.

**The API surface.** `eos_point(par, mode, species, n_B=, T= | SnB=, ...)`,
`eos_table(par, mode, species, axes, ..., progress=, verbose=)` and
`eos_response(par, mode, species, frozen=, n_B=, T=, Y_p=, **conditions)`.
`progress` is called once per completed line with `{mode, line, n_lines,
temp_key, temp, fracs, n_solved, n_requested, elapsed_s}`, the same dictionary
in every model; deep solver code never prints. Non-convergence is a return
value at the public boundary, never an exception.

**Backends.** Reference: NumPy/SciPy, finite-difference Jacobian — the
correctness oracle. Fast: hand-derived analytic Jacobian, Numba-compiled at
T = 0, held to the reference by backend-parity gates.
