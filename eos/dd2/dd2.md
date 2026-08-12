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

**Solving.** One residual system for all modes over
`x = [sigma, omega0, rho0, (phi0), mu_B - Sigma^R, mu_C, (mu_S), (mu_nue)]`;
species potentials follow `mu_i = B_i mu_B + Q_i mu_C + S_i mu_S`, and the
solver works in the effective potentials `mu_eff_i = mu_i - Sigma0_i`, which
vary smoothly along density sweeps (that is what makes warm starts work).
Modes: `beta_eq_neutrinoless`, `beta_eq_neutrino_trapped` (electron family
trapped), `fixed_YC` (leptons on/off), `fixed_YC_YS`. A temperature axis may
be replaced by entropy per baryon (outer 1-D solve for T).

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
