# DD2 — density-dependent relativistic mean field

The full description, with equations and bibliography, is `dd2.tex` (compiled
against `../../docs/eos.bib`). This file is the plain-text summary.

**Model.** DD-RMF of Typel et al., PRC 81, 015803 (2010): nucleons (plus,
optionally, the hyperon octet and the Delta quartet) exchange sigma, omega,
rho and — for strange baryons — the hidden-strange phi, with couplings that
depend on the total baryon density:

    Gamma_i(n_B) = Gamma_i(n_sat) f_i(x),  x = n_B/n_sat
    f_{sigma,omega}(x) = a (1 + b(x+d)^2)/(1 + c(x+d)^2)      (rational)
    f_rho(x)           = exp[-a_rho (x-1)]                     (exponential)

with `f_i(1) = 1` and `f_i''(0) = 0` making `a_i` and `d_i` dependent. The
density dependence produces the rearrangement self-energy `Sigma^R`, the same
for every baryon, which enters chemical potentials and pressure but never the
energy density. Every assembled state is checked against the
Hugenholtz–Van Hove identity `eps + P - T s = sum_i mu_i n_i` at 1e-8.

**Extensions.** Hyperons: SU(6) vector ratios; scalar ratios either the
published DD2Y values (Marques et al. 2017; Fortin et al. 2017) or inverted
from the potentials U_Lambda, U_Sigma, U_Xi in saturated symmetric matter.
The phi inherits the omega density dependence. Deltas: ratio couplings,
default universal, or x_Delta_sigma from U_Delta. Thermal pi/K (and
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
sources the sigma field, so it does not stay confined to the totals. At T = 0
everything is elementary (`kF = sqrt(mu_eff^2 - m*^2)`, n ∝ kF^3, s = 0); the
closed forms are in `dd2.tex` Eq. (T0), and that branch is also Numba-compiled.
At T > 0 the integrals are the Johns-Ellis-Lattimer approximants from
`eos/general/fermi_integrals` (~1e-4 accurate), with a Gauss-Laguerre
quadrature there as the accuracy reference. The thermal mesons are the same
expressions with Bose statistics, from `eos/general/bose_integrals`.

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
it is absent from R5. Modes: `beta_eq_neutrinoless`,
`beta_eq_neutrino_trapped` (electron family trapped), `fixed_YC` (leptons
on/off), `fixed_YC_YS`. A temperature axis may be replaced by entropy per
baryon (outer 1-D solve for T).

**NMPs.** Forward map at the model's own saturation: E_sat, m*/m, K_sat,
Q_sat, E_sym, L_sym (and K_sym). Inverse map imposes
{n_sat, E_sat, m*/m, K_sat, E_sym, L_sym}; the isoscalar sector closes with
the cross-constraint `f_sigma''(1) = f_omega''(1)` plus one shape coefficient
pinned at its published value, and Q_sat / K_sym come back as predictions
(imposing Q_sat instead remains an option). Q_sat rides on a third finite
difference — forward and inverse use the identical stencil so the bias
cancels on round trips — and the inverter retries from jittered seeds before
declaring a target unrepresentable.

**Backends.** Reference: NumPy/SciPy, finite-difference Jacobian — the
correctness oracle. Fast: hand-derived analytic Jacobian, Numba-compiled at
T = 0, held to the reference by backend-parity gates.
