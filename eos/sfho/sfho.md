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
x_Delta_sigma from U_Delta. The thermal pseudoscalar nonet (pi, K, eta, eta')
enters as Bose gases whose effective potentials are shifted by the same vector
mean fields (Lavagno 2010); the gas contributes charge and strangeness to the
equilibrium constraints, no baryon number, and no field sources. Bose
condensation is REFUSED, not approximated: every entry point reports
`condensation = max_j |mu*_j|/m_j` and a state at or past 1 comes back with
`converged = False`.

**Field equations.** Solved for (sigma, omega, rho, phi) together with the
potentials — three of the four are nonlinear in the fields, so none can be
eliminated algebraically the way DD2's can:

    m_sigma^2 sigma + g2 sigma^2 + g3 sigma^3 - (dA/dsigma) rho^2 = hc^3 S_sigma
    m_omega^2 omega + c3 omega^3     + (dA/domega) rho^2          = hc^3 S_omega
    m_rho^2   rho   + c4 rho^3       + 2 A rho                    = hc^3 S_rho
    m_phi^2   phi                                                 = hc^3 S_phi

    S_sigma = sum_i g_sigma_i ns_i     S_omega = sum_i g_omega_i n_i
    S_rho   = sum_i g_rho_i I3_i n_i   S_phi   = sum_i g_phi_i n_i

with `m*_i = m_i - g_sigma_i sigma` and
`mu_eff_i = mu_i - g_omega_i omega - g_rho_i I3_i rho - g_phi_i phi`.

**One species as an ideal gas.** Each baryon is a Fermi gas of mass `m*_i` at
`mu_eff_i`, degeneracy g_i (2 for the octet, 4 for the Delta), antiparticles
included. With `E = sqrt(k^2 + m*_i^2)` and
`f± = 1/(1 + exp((E ∓ mu_eff_i)/T))`:

    n_i   = g_i/(2 pi^2 hc^3) ∫dk k^2       (f+ - f-)
    eps_i = g_i/(2 pi^2 hc^3) ∫dk k^2 E     (f+ + f-)
    P_i   = g_i/(6 pi^2 hc^3) ∫dk k^4 / E   (f+ + f-)

The other two are NOT integrated — they come from the trace of the
energy-momentum tensor and the one-species Euler relation:

    ns_i = (eps_i - 3 P_i) / m*_i          s_i = (eps_i + P_i - mu_eff_i n_i)/T

which matters: an error in eps_i or P_i propagates into ns_i, and ns_i sources
the sigma field, so it does not stay confined to the totals. At T = 0, with
`kF = sqrt(mu_eff^2 - m*^2)` and `L = ln((kF + |mu_eff|)/m*)`, everything is
elementary (n ∝ kF^3, s = 0); the closed forms are in `sfho.tex` Eq. (T0). At
T > 0 the integrals are the Johns-Ellis-Lattimer approximants from
`eos/general/fermi_integrals` (~1e-4 accurate), with a Gauss-Laguerre
quadrature there as the accuracy reference.

The thermal mesons are the same expressions with Bose statistics,
`b± = 1/(exp((E ∓ mu*_j)/T) - 1)`, g_j = 1, at their PHYSICAL masses — pi±
139.570, pi0 134.977, K± 493.677, K0/K0bar 497.611, eta 547.862, eta' 957.780
MeV — from `eos/general/bose_integrals`. Nine species: the charged and neutral
partners are kept apart rather than averaged, since they carry different
charges and averaging would misplace n_C_mes as well as the population.

**The totals.**

    eps = sum_i eps_i + eps_mf + eps_mes        (+ leptons, photons)
    P   = sum_i P_i   + P_mf   + P_mes          (+ leptons, photons)
    s   = sum_i s_i             + s_mes         (+ leptons, photons)
    n_B = sum_i B_i n_i     n_C = sum_i C_i n_i + n_C_mes
                            n_S = sum_i S_i n_i + n_S_mes

with P_mf, eps_mf the mean-field terms above. Photons:
`P = pi^2 T^4/(45 hc^3)`, `eps = 3P`, `s = 4 pi^2 T^3/(45 hc^3)`. The Euler sum
reported with the state takes baryons at their FULL potentials and the meson
gas at its EFFECTIVE ones:
`sum_i mu_i n_i + sum_j mu*_j n_j + mu_e n_e + mu_nue n_nue`.

**Solving.** One residual system for all modes over
`x = [sigma, omega, rho, phi, mu_B, mu_C, (mu_S), (mu_nue), (T)]`; species
potentials follow `mu_i = B_i mu_B + C_i mu_C + S_i mu_S`. The rows, in the
order they are assembled:

    R1..R4  the four field equations, each divided by m_M^2 x 30 MeV so the
            row is dimensionless and O(1) rather than O(1e7) MeV^3 — without
            that scaling they dominate the norm and the charge rows never
            converge
    R5      sum_i B_i n_i - n_B                                    always
    R6      n_C - n_e   (C equilibrated)  |  n_C - Y_C n_B  (Y_C imposed)
    R7      n_S - Y_S n_B                                iff Y_S imposed
    R8      (n_e + n_nue)/n_B - Y_Le      iff the electron family is trapped
    R9      s/n_B - S/A                   iff an entropy per baryon replaces T

All four fields are always unknowns — a mean field does not know what is being
held fixed — and mu_C is an unknown in every mode; the mode changes only the
row that closes it. Modes: `beta_eq_neutrinoless`,
`beta_eq_neutrino_trapped` (electron family trapped), `fixed_YC` (leptons
on/off), `fixed_YC_YS`. Non-convergence is a return value at every layer,
never an exception.

The muon lepton family is not tracked at all; requesting it raises.

**NMPs.** Forward map at the model's own saturation (the P = 0 root of
symmetric matter): n_sat, E_sat, m*/m, K_sat, Q_sat, E_sym, L_sym, K_sym — the
same key set `eos.dd2.compute_nmp` returns. It reproduces the published SFHo
values (0.158 fm^-3, -16.2, 31.6, 47.1 MeV) to better than 0.3 percent. E_sym
uses the closed form `k_F^2/(6 E_F*) + n_B g_rho^2 / [8 (m_rho^2 + 2A)]`, and
the verify suite compares it against the delta^2 curvature of E/A, which is an
independent route through eps.

Inverse map: `invert_nmp` / `from_nmp`. The inversion is TRIANGULAR, because in
symmetric matter the rho field and A rho^2 drop out of every equation and the
isoscalar sector never sees the isovector couplings. Isoscalar half is the
classical Boguta-Bodmer inversion, {g_sigma_N, g_omega_N, g2, g3} against
{P(n_sat) = 0, E_sat, m*/m, K_sat} at fixed m_sigma, m_omega, c3 — four
against four, no structural closure needed — solved in the reduced (b, c) the
published table states, since b ~ 7e-3 and c ~ -4e-3 scale far better than
g2 ~ 3e3 MeV and g3 ~ -12.

Isovector half needs a closure: {E_sym, L_sym} face g_rho_N plus NINE shape
coefficients of A, so exactly two are freed. The choice is **(g_rho_N, b1)**,
for three measured reasons: best conditioned of the candidates (2x2 Jacobian
in log-knobs, cond 3.40 against 3.53 for an overall scale on f and 11.60 for
a1); widest reach (L_sym from -6 to 146 MeV at E_sym ~ 31.5, against [-34, 59]
and [27, 69]); and b1 IS the Horowitz-Piekarewicz Lambda_v omega^2 rho^2
coupling, so an inverted set stays comparable to published ones. Note that an
overall scale on f, which looks least invasive because it preserves the SHAPE
of A, is the MOST invasive to the physics: scaling A changes how 2A competes
with m_rho^2 as density rises, so at L_sym = 70 it puts E_sym(3 n_sat) at 55.8
MeV against b1's 52.2 and a1's 49.6.

Q_sat and K_sym come back as predictions. Two failure modes are RETURN VALUES,
not exceptions: an isoscalar or isovector solve that misses its gate, and a
fit that lands on the runaway cross-coupling branch — E_sym's potential term
saturates as g_rho grows, so a physically absurd (g_rho_N, b1) can reproduce a
target exactly, and |2A| < m_rho^2 at saturation is checked afterwards to
refuse it. Hyperon and Delta couplings are NOT refitted; they ride along from
the base set and no longer match the potential depths they were built from.

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
