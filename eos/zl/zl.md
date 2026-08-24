# ZL — the Zhao-Lattimer nucleonic density functional

`zl.tex` is the same description typeset, with the bibliography compiled
against `../../docs/eos.bib`. Both files carry the same physics: this one is not
a summary of that one, and neither defers to the other for an equation.

**Model.** Protons and neutrons as free Fermi gases at their vacuum mass, plus
a density-dependent interaction energy density. Zhao & Lattimer, PRD 102,
023021 (2020); the parameter set is the one used by Constantinou et al., PRD
104, 123032 (2021), PRD 107, 074013 (2023) and arXiv:2506.20418 (2025):

    V(n_p, n_n) = 4 n_p n_n [a0/n0 + b0/n0 u^(gamma-1)]
                + (n_n - n_p)^2 [a1/n0 + b1/n0 u^(gamma1-1)],   u = n_B/n0

There is no scalar field, so no gap equation and no effective mass: the whole
self-consistency of the model is between the densities and the interaction
potentials they generate,

    mu_Hv_i = dV/dn_i     mu_eff_i = mu_i - mu_Hv_i     n_i = n_i(mu_eff_i, T, m_i)

Per baryon, with delta = (n_n-n_p)/n_B,

    V/n_B = (1-delta^2) [a0 u + b0 u^gamma] + delta^2 [a1 u + b1 u^gamma1]

so BOTH brackets enter the symmetry energy: the a0/b0 term is a proton-neutron
cross interaction, not an isoscalar one, and its potential part is
`(a1-a0) u + b1 u^gamma1 - b0 u^gamma`. Reading `a1+b1` as the potential
symmetry energy is the standard way to misread the functional, and gives the
wrong sign.

**Parameters.** `n0` (a reference density of the functional, not the
saturation density it predicts), `a0, b0, gamma`, `a1, b1, gamma1`, and the
two nucleon masses, both 939.5 MeV — the kinetic term has no isospin
splitting. All are fields of `Parameters` and are arguments everywhere.

**Single-nucleon thermodynamics.** Each species is a free Fermi gas of mass
`m_i` and degeneracy `g = 2` (spin), evaluated at its effective potential
`mu_eff_i`, antiparticles included. With `f(x) = 1/(1 + exp(x/T))` and
`E_k = sqrt(k^2 + m_i^2)`:

    n_i   = g/(2 pi^2 (hc)^3) int_0^inf dk k^2   [f(E_k - mu_eff_i) - f(E_k + mu_eff_i)]
    P_i   = g/(6 pi^2 (hc)^3) int_0^inf dk k^4/E_k [f(E_k - mu_eff_i) + f(E_k + mu_eff_i)]
    eps_i = g/(2 pi^2 (hc)^3) int_0^inf dk k^2 E_k [f(E_k - mu_eff_i) + f(E_k + mu_eff_i)]

The entropy density is NOT integrated separately. It comes from the Euler
relation of the free gas,

    s_i = (eps_i + P_i - mu_eff_i n_i) / T

which is exact for an ideal gas at the EFFECTIVE potential, and is the reason
the interaction never enters `s`.

At T = 0 the antiparticle terms vanish, the occupations become step functions at
`k_F_i = sqrt(mu_eff_i^2 - m_i^2)` (with `n_i = 0` when `mu_eff_i <= m_i`), and
these close in elementary functions, with `E_F_i = sqrt(k_F_i^2 + m_i^2) = mu_eff_i`:

    n_i   = g k_F_i^3 / (6 pi^2 (hc)^3)
    eps_i = g/(16 pi^2 (hc)^3) [ k_F_i (2 k_F_i^2 + m_i^2) E_F_i
                                 - m_i^4 ln((k_F_i + E_F_i)/m_i) ]
    P_i   = g/(48 pi^2 (hc)^3) [ k_F_i (2 k_F_i^2 - 3 m_i^2) E_F_i
                                 + 3 m_i^4 ln((k_F_i + E_F_i)/m_i) ]
    s_i   = 0

These integrals are not implemented in this subpackage: they come from
`eos.general.fermi_integrals`, which evaluates them through the
Johns-Ellis-Lattimer analytic approximation — uniformly valid from the
degenerate to the non-degenerate limit and exact at T = 0. The same module
supplies the inverse `mu_eff_i(n_i, T, m_i)` used by the density-first entry
point. They are written out here anyway, because a paper-style description is
self-contained.

**There is no scalar density `n_s`.** Every model with an effective mass returns
one through `n_s = (eps - 3P)/m*`; ZL has no scalar field and no `m*`, so the
quantity is not defined and is not returned. That is an absence of the physics,
not an omission.

**The totals.** Summed over the active species, plus leptons and photons where
a mode includes them:

    eps = sum_i eps_i + V              P = sum_i P_i + P_int
    s   = sum_i s_i                    (V carries no T)

and the Euler relation `eps + P = T s + sum_i mu_i n_i` holds identically at the
physical potentials, which is what `verify/` checks.

**The interaction** adds

    eps_int = V                                       (the functional itself)
    P_int   = sum_i n_i mu_Hv_i - V
            = 4 n_p n_n [a0/n0 + gamma b0/n0 u^(gamma-1)]
            + (n_n-n_p)^2 [a1/n0 + gamma1 b1/n0 u^(gamma1-1)]
    s_int   = 0                                       (V carries no T)

The power-law pieces of `P_int` pick up the factors `gamma`, `gamma1` and the
linear pieces do not; that identity is the numerical content of "V is a
functional of the densities and nothing else", and `verify/` checks it. There
is no rearrangement term to place separately: `mu_Hv_i` is the full derivative,
so the Euler relation `eps + P = T s + sum_i mu_i n_i` holds identically at the
physical potentials.

**Charges.** `n_B = n_p + n_n`, `n_C = n_p`, `n_S = 0`; `mu_B = mu_n`,
`mu_C = mu_p - mu_n`, and `mu_S` is reported as zero by convention — no
equation of any mode responds to it.

**Modes.** Every mode imposes the two self-consistency equations
`n_i(mu_eff_i, T, m_i) = n_i`. Then:

| mode | unknowns | conditions added |
|------|----------|------------------|
| `beta_eq_neutrinoless`     | `mu_p, mu_n, mu_e, n_p, n_n`         | `n_p+n_n = n_B`, `mu_C + mu_e = 0`, `n_p = n_e` |
| `beta_eq_neutrino_trapped` | `mu_p, mu_n, mu_e, mu_nue, n_p, n_n` | the same, with `mu_C + mu_e - mu_nue = 0` and `(n_e+n_nue)/n_B = Y_Le` |
| `fixed_YC`                 | `mu_p, mu_n` (`, mu_e`)              | `n_p, n_n` are known from `Y_C`; with leptons, `n_e = n_p` |
| `fixed_YC_YS`              | —                                    | RAISES: `n_S = 0` identically, so the mode is meaningless here |

`leptons=True/False` applies to `fixed_YC`: without leptons the result is
charged nucleonic matter, which is what a mixed phase needs per pure phase
before global neutrality is imposed. Photons are a separate flag. Entropy per
baryon may replace `T`, through an outer 1-D solve.

**Nuclear-matter parameters.** `nmp.compute_nmp(par)` is the forward map, at
T = 0. Saturation is the `P = 0` root of symmetric matter and is NOT `n0`, the
functional's reference density. Every value is a prediction — ZL imposes no
saturation condition:

    n_sat = 0.15951 fm^-3   E_sat = -15.99648   K_sat = 250.174
    E_sym = 30.84803        L_sym =  41.27034
    Q_sat = -352.9          K_sym =  -88.5 +/- 0.2      (all MeV)

The first five are the published set of Constantinou et al. and are pinned in
`verify/run_full_check.py` at the published precision. `Q_sat` and `K_sym` are
reported but not pinned: they are not in the published set, and a check cannot
assert a number nobody published.

`K_sym` is quoted to one decimal on purpose. It is a SECOND density derivative
of a quantity that is itself a second derivative in `beta`, and the estimate
drifts with the step — -88.47, -88.42, -88.46, -88.60 for `h/n_sat` = 0.05,
0.02, 0.01, 0.005 — as round-off starts to dominate before truncation has
finished falling. So it is determined to roughly +/- 0.2, and any four-digit
value for it, in this document or elsewhere, claims a precision the finite
difference does not have. `L_sym`, one derivative lower, converges cleanly
(41.307, 41.270, 41.265, 41.264 over the same steps).

`S(n)` is the curvature at `beta = 0`, `S = (1/2) d^2(E/A)/d beta^2`, which is
the definition the published numbers use. Note that `eos.did.nmp` estimates the
same coefficient with a full step to pure neutron matter and a Richardson
correction; that route gives 30.776 and 41.124 here, carrying `beta^4`
contamination the published values do not include. The difference is real, not
numerical: DID needs the full step because its `E/A` difference at small
asymmetry sits in noise, and ZL's does not.

**`invert_nmp` raises.** ZL has six couplings — `a0, b0, gamma, a1, b1, gamma1`
— against the five NMPs of the standard list, so the inversion has a
one-parameter family of solutions. DD2 closes its isoscalar sector with a
structural cross-constraint; nothing in Constantinou et al. singles out a member
of ZL's family. Closing it needs either a sixth imposed datum (`Q_sat`, or an
effective mass) or one coupling held fixed, and until one is chosen the function
raises saying so rather than returning an arbitrary member.

**Numerics.** Two to six equations, Powell hybrid then Levenberg-Marquardt,
with one further cold-start attempt when a warm start was supplied and failed.
Convergence is judged on a dimensionless residual — density and charge rows
divided by `n_B`, potential equalities by `mu_B` — gated on the largest scaled
component at 1e-10. Non-convergence is a return value, not an exception.

**Not implemented** (see `docs/DEFERRED.md`): muons, hyperons, deltas, thermal
mesons, thermal neutrinos, `fixed_YC_YS`, the NMP inversion (above), and the
freezes of `eos_response` beyond `equilibrium`.
