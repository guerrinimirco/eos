# alphaBag — the perturbatively corrected bag model, unpaired and CFL

The full description, with equations and bibliography, is `alphabag.tex`
(compiled against `../../docs/eos.bib`). This file is the plain-text summary.

**Model.** Deconfined `u, d, s` quark matter inside a bag, with the leading
perturbative QCD correction carried as one constant coupling `alpha_s`
multiplying the free-gas pressure, in the arrangement of Fischer et al., ApJS
194, 39 (2011). Light flavours are massless, the strange quark is massive and
gets the exact Fermi gas. Thermal gluons are an optional sector with their own
correction factor. There is no vector field (that is `eos/vmit`) and no gap
equation: the potential is explicit in `mu`, so nothing but the equilibrium
conditions has to be solved.

    P     = sum_q P_q(mu_q, T, m_q, alpha) + P_g(T, alpha) - B/(hc)^3
    eps   = sum_q eps_q + eps_g + B/(hc)^3
    s     = sum_q s_q + s_g

The bag enters `eps` and `P` with opposite signs and neither `s` nor any
`n_q`, so it cancels out of `eps + P` and the Euler relation
`eps + P = T s + sum_q mu_q n_q` holds with no bag term in it.

**The correction** is two multiplicative factors, plus one for the gluons:

    c_q = 1 - 2 alpha/pi         c_T = 1 - 50 alpha/(21 pi)
    c_g = 1 - 15 alpha/(4 pi)

At `alpha = 0.3`: 0.8090, 0.7726, 0.6419. `alpha` does not run — it is a
parameter of the set, which is what lets an inference run vary it.

**Massless flavour** (degeneracy `g = 6`, antiquarks included):

    P_0 = [ (7/60) pi^2 T^4 c_T + (T^2 mu^2/2 + mu^4/(4 pi^2)) c_q ] / (hc)^3
    n_0 = ( mu T^2 + mu^3/pi^2 ) c_q / (hc)^3
    s_0 = [ (7/15) pi^2 T^3 c_T + T mu^2 c_q ] / (hc)^3
    eps_0 = 3 P_0

`n_0 = dP_0/dmu` and `s_0 = dP_0/dT` hold identically, so the corrected
massless gas satisfies Euler on its own.

**Massive flavour**: the exact Fermi gas from `eos.general.fermi_integrals`
(JEL), plus the *massless* correction evaluated at the same `mu`:

    X(mu,T,m,alpha) = X_Fermi(mu,T,m) + [X_0(mu,T,alpha) - X_0(mu,T,0)]

for X in {n, P, eps, s}. That is a prescription, not an expansion of the
massive result — the true O(alpha_s) term differs at relative order
`m^2/mu^2`, about 12% of a 19% correction for `m_s = 150` MeV at
`mu_s ~ 440` MeV. What it does guarantee is the exact massless limit and a
correction that is itself a consistent set, so Euler survives.

**Gluons**: `g = 16` massless bosons at `mu = 0`,

    P_g = (8 pi^2/45) T^4 c_g/(hc)^3,  eps_g = 3 P_g,
    s_g = (32 pi^2/45) T^3 c_g/(hc)^3, n_g = 0

`gluons` is this model's own sector flag; no other model in the repository has
one.

**Parameters.** `m_u = m_d = 0`, `m_s = 150` MeV, `alpha = 0.3`,
`B4 = 165` MeV. `B = B4^4 = 7.412e8 MeV^4 = 96.466 MeV/fm^3` is a derived
property, never stored, so a set cannot carry a `B` and a `B4` that disagree.
The CFL gap `Delta0` is deliberately NOT a parameter field: it selects a phase
rather than tuning one, and is passed per call.

Cold beta-equilibrated matter reaches `P = 0` at `n_B = 0.40309 fm^-3` with
`eps/n_B = 1046.17` MeV, so the shipped set is not absolutely stable strange
matter. Pairing lowers it: CFL at `Delta0 = 100` MeV has its surface at
`n_B = 0.36286 fm^-3`, `eps/n_B = 936.55` MeV.

**CFL phase.** A second phase, not a further correction. The gap is imposed,
not solved:

    Delta(T) = Delta0 sqrt(1 - T^2/T_c^2)  for T < T_c, else 0
    T_c      = 0.57 * 2^(1/3) * Delta0 = 0.71815 Delta0
    dDelta/dT = -Delta0 T / (T_c^2 sqrt(1 - T^2/T_c^2))

and the pairing term of Alford, Braby, Paris & Reddy, ApJ 629, 969 (2005) is
added per flavour so the three potentials need not be equal:

    P_CFL = sum_q P_q + (Delta^2/pi^2) sum_q mu_q^2 / (hc)^3 - B/(hc)^3
    n_q   = n_q + 2 mu_q Delta^2/(pi^2 (hc)^3)                (= dP/dmu_q)
    s     = sum_q s_q + 2 Delta (dDelta/dT) sum_q mu_q^2/(pi^2 (hc)^3)  (= dP/dT)
    f     = -P + sum_q mu_q n_q,   eps = f + T s

The `-3 m_s^2 mu^2/(4 pi^2)` term those references carry alongside the pairing
term is NOT added: it is the leading expansion of the massive strange Fermi
gas, which `P_s` already contains exactly, and adding both would count it
twice. Gluons are not inside the CFL potential (Meissner-massive), but remain
available as a flag at the solver level.

**Charges.** `n_B = (n_u+n_d+n_s)/3`, `n_C = (2n_u-n_d-n_s)/3`, `n_S = n_s`
(S = +1 per s quark, this repository's convention); `mu_B = mu_u + 2 mu_d`,
`mu_C = mu_u - mu_d`, `mu_S = mu_s - mu_d`. The maps are
`eos.general.basis`, not a local copy, and a test pins them bit-for-bit.

**Modes.** Unknowns are chemical potentials only — the potential is explicit
in `mu`, so no field equation and no density enters the unknown vector.

| mode | unknowns | rows |
|------|----------|------|
| `beta_eq_neutrinoless`     | `mu_u, mu_d, mu_s, mu_e`         | `n_B`, `n_C = n_e`, `mu_C + mu_e = 0`, `mu_S = 0` |
| `beta_eq_neutrino_trapped` | `mu_u, mu_d, mu_s, mu_e, mu_nue` | the same, with `mu_C + mu_e - mu_nue = 0` and `(n_e+n_nue)/n_B = Y_Le` |
| `fixed_YC`                 | `mu_u, mu_d, mu_s`               | `n_B`, `Y_C`, `mu_S = 0` |
| `fixed_YC_YS`              | `mu_u, mu_d, mu_s`               | `n_B`, `Y_C`, `Y_S` |
| CFL (a phase, not a mode)  | `mu_u, mu_d, mu_s`               | `n_u = n_d = n_s = n_B` (flavour locking) |

`leptons=True/False` applies to the two fixed-fraction modes: with leptons a
neutralizing electron gas is added AFTER the solve, by inverting
`n_e(mu_e,T) = n_C`, because the quark sector does not respond to `mu_e` at
fixed `Y_C`; without them the phase is charged, which is what a mixed phase
needs per pure phase. Photons, gluons and thermal neutrinos are separate
flags. Entropy per baryon may replace `T`, through an outer 1-D solve.

Two things about the CFL closure that are easy to expect wrongly: it is
electrically neutral by construction (`n_C = 0` identically, no electrons),
and it is NOT in strangeness equilibrium — equal densities at unequal masses
need unequal potentials, so `mu_S != 0` and `eps/n_B = mu_B + mu_S` at
`P = 0`, not `mu_B`.

**Numerics.** Three to five equations, Powell hybrid then
Levenberg-Marquardt, with one further cold-start attempt when a warm start was
supplied and failed. Convergence is judged on a dimensionless residual —
density rows divided by `n_B`, potential equalities by `mu_B`, fraction rows
already dimensionless — gated on the largest scaled component at 1e-10.
Non-convergence is a return value, not an exception.

**Not implemented** (see `docs/DEFERRED.md`): muons, hyperons, deltas, thermal
mesons, the entropy-per-baryon table axis, and the freezes of `eos_response`
beyond `equilibrium`.
