# SFHo — nonlinear relativistic mean field

The full description, with equations and bibliography, is `sfho.tex` (compiled
against `../../docs/eos.bib`). This file is the plain-text summary.

**Model.** Nonlinear RMF of Steiner, Hempel & Fischer, ApJ 774 (2013) 17:
nucleons (plus, optionally, the hyperon octet and the Delta quartet) exchange
sigma, omega, rho and — for strange baryons — the hidden-strange phi, with
CONSTANT couplings. The density dependence lives in the self-interactions
instead:

    U(sigma) = (g2/3) sigma^3 + (g3/4) sigma^4          Boguta-Bodmer
    (c3/4) omega^4,  (c4/4) rho^4                        quartic vector
    A(sigma,omega) rho^2,  A = g_rhoN^2 [sum_i a_i sigma^i + sum_j b_j omega^2j]

The last is the isoscalar-isovector cross coupling of Steiner, Prakash,
Lattimer & Ellis, Phys. Rept. 411 (2005) — six a_i and three b_j — which is
what makes L_sym adjustable at fixed E_sym. It generalises the
Horowitz-Piekarewicz Lambda_v omega^2 rho^2 term, which is b_1 alone. A is
separable in sigma and omega, so d2A/dsigma domega = 0.

Isospin is the tau_3/2 convention here (I_3 = ±1/2 for nucleons), NOT DD2's
tau_3 = ±1: the two absorb a factor of two into g_rho, and neither model may
be read with the other's coupling table.

Because the couplings are constants there is NO rearrangement self-energy:
Sigma^R = 0, stated rather than omitted. That is the one structural difference
from DD2; everything else — module layout, names, modes, records — is the
same by design.

**The energy density has two terms the pressure does not.** Eliminating the
omega source through its own field equation gives
`e_omega = (1/2) m_omega^2 omega^2 + (3 c3/4) omega^4 + omega (dA/domega) rho^2`
against `(1/2) m_omega^2 omega^2 + (c3/4) omega^4` in P, and likewise
`3A rho^2` against `A rho^2` when only b_1 is kept. Both were missing once and
only the Hugenholtz-Van Hove identity found it: the solver converged perfectly
on an eps wrong by up to 1.8 percent in asymmetric matter, which put E_sym at
18.7 MeV instead of 31.6 and made L_sym negative. Every assembled state is now
checked against `eps + P - T s = sum_i mu_i n_i` at 1e-8.

**Extensions.** Hyperons: SU(6) vector ratios with the SFHoY enhancement
y = 1.5 (Lambda, Sigma) and 1.875 (Xi) of Fortin, Oertel & Providência, PASA
35 (2018) e044; scalar ratios either the published SFHoY values or inverted
from the potentials U_Lambda = -30, U_Sigma = +30, U_Xi = -14 MeV in saturated
symmetric matter. Deltas: universal vector coupling by default, or
x_Delta_sigma from U_Delta. Thermal pi/K/eta as Bose gases whose effective
potentials are shifted by the same vector mean fields (Lavagno 2010); the gas
contributes charge and strangeness to the equilibrium constraints, no baryon
number, and no field sources. Bose condensation is REFUSED, not approximated:
every entry point reports `condensation = max_j |mu*_j|/m_j` and a state at or
past 1 comes back with `converged = False`.

**Solving.** One residual system for all modes over
`x = [sigma, omega, rho, phi, mu_B, mu_C, (mu_S), (mu_nue), (T)]`; species
potentials follow `mu_i = B_i mu_B + C_i mu_C + S_i mu_S`. All four fields are
always unknowns — a mean field does not know what is being held fixed — and
mu_C is an unknown in every mode; the mode changes only the row that closes
it. Modes: `beta_eq_neutrinoless`, `beta_eq_neutrino_trapped` (electron family
trapped), `fixed_YC` (leptons on/off), `fixed_YC_YS`. A temperature axis may
be replaced by entropy per baryon, with T joining the unknown vector and
`s/n_B = S/A` joining the rows. Non-convergence is a return value at every
layer, never an exception.

The muon lepton family is not tracked at all; requesting it raises.

**NMPs.** Forward map at the model's own saturation (the P = 0 root of
symmetric matter): n_sat, E_sat, m*/m, K_sat, Q_sat, E_sym, L_sym, K_sym — the
same key set `eos.dd2.compute_nmp` returns. It reproduces the published SFHo
values (0.158 fm^-3, -16.2, 31.6, 47.1 MeV) to better than 0.3 percent. E_sym
uses the closed form `k_F^2/(6 E_F*) + n_B g_rho^2 / [8 (m_rho^2 + 2A)]`, and
the verify suite compares it against the delta^2 curvature of E/A, which is an
independent route through eps. The INVERSE map is not written: its isoscalar
half is the classical Boguta-Bodmer inversion, but its isovector half needs a
closure — two conditions {E_sym, L_sym} face g_rho_N plus nine shape
coefficients — and that choice decides how E_sym behaves above saturation.

**Backends.** Reference: NumPy/SciPy with MINPACK's own forward-difference
Jacobian — the correctness oracle, and the default. `backends/jacobian.py`
carries a hand-derived analytic Jacobian of the same residual, agreeing with a
central difference of it to better than 1e-7 in every mode; it is NOT on by
default because it is not faster here (it cuts residual evaluations by about a
third and costs more per evaluation than it saves). What it is for is the
susceptibility matrix chi_ab = dn_a/dmu_b, which no finite difference inside
the model can give — the solver never varies the three potentials
independently — and which is checked against the inverse map dmu_a/dn_b that
the `fixed_YC_YS` mode supplies.

**Responses.** `eos_response(frozen='equilibrium')` returns `cs2_isothermal`,
and at T > 0 also `cs2_adiabatic`, `C_V`, `C_P` and `Gamma_th`, all by finite
differences along re-solved sequences, plus `chi` from the Jacobian. The two
sound speeds are named for the thermal condition they are taken at because at
T > 0 they are different numbers.
