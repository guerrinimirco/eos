# ZL — the Zhao-Lattimer nucleonic density functional

The full description, with equations and bibliography, is `zl.tex` (compiled
against `../../docs/eos.bib`). This file is the plain-text summary.

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

**Thermodynamics.** Each nucleon is a free Fermi gas of degeneracy `g = 2`,
with antiparticles, evaluated at `mu_eff_i` through the JEL integrals in
`eos.general.fermi_integrals`. The interaction adds

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

**Nuclear-matter parameters**, measured from the code at T = 0 (saturation is
the `P = 0` root of symmetric matter, and is not `n0`):

    n_sat = 0.15951 fm^-3   E_sat = -16.00   K_sat = 250.2   Q_sat = -352.8
    E_sym = 30.85           L_sym = 41.26    K_sym = -88.86      (all MeV)

No inverse map is implemented: the parameters sit close enough to these
quantities that a fit is six one-dimensional relations rather than new physics.

**Numerics.** Two to six equations, Powell hybrid then Levenberg-Marquardt,
with one further cold-start attempt when a warm start was supplied and failed.
Convergence is judged on a dimensionless residual — density and charge rows
divided by `n_B`, potential equalities by `mu_B` — gated on the largest scaled
component at 1e-10. Non-convergence is a return value, not an exception.

**Not implemented** (see `docs/DEFERRED.md`): muons, hyperons, deltas, thermal
mesons, thermal neutrinos, `fixed_YC_YS`, and the freezes of `eos_response`
beyond `equilibrium`.
