# vMIT — MIT bag model with a repulsive vector interaction

The full description, with equations and bibliography, is `vmit.tex` (compiled
against `../../docs/eos.bib`). This file is the plain-text summary.

**Model.** Three quark flavours in a bag of constant energy density `B`,
interacting through an isoscalar-vector field. Chodos et al., PRD 9, 3471
(1974) for the bag; the vector term in the form used by Gomes et al., ApJ 877,
139 (2019) and Constantinou et al., PRD 104, 123032 (2021) and PRD 107, 074013
(2023):

    L = sum_q qbar [ gamma_mu (i d^mu - g_V V^mu) - m_q ] q
        - 1/4 V_munu V^munu + 1/2 m_V^2 V_mu V^mu - B

Quark masses are the *current* masses and are parameters: there is no scalar
condensate and no gap equation, so the only mean field is the vector one, and
it is algebraic:

    V = (g_V^2/m_V^2) sum_q n_q = a hbar c (n_u + n_d + n_s)

The two couplings are not separately identifiable at mean-field level, so the
code carries only the combination `a = g_V^2/m_V^2` in fm^2. Being flavour
blind, `V` shifts all three potentials equally: `mu_eff_q = mu_q - V` is what
enters the Fermi integrals.

**Parameters.** `B4 = B^(1/4)` in MeV (stored as the fourth root, the form it
is quoted in), `a` in fm^2, and the three current masses `m_u`, `m_d`, `m_s`.
All are arguments — `Parameters` — never module constants.

**Thermodynamics.** Each flavour is a free Fermi gas of mass `m_q` and
degeneracy `g = 2 spin x 3 colour = 6`, with antiparticles, evaluated at
`mu_eff_q` through the JEL integrals in `eos.general.fermi_integrals`. The
vector field and the bag add

    P_V = eps_V = 1/2 a hbar c (sum_q n_q)^2
    P_B = -B/(hbar c)^3        eps_B = +B/(hbar c)^3

A vector field enters `P` and `eps` with the SAME sign; the bag with opposite
signs, which is what makes the pressure negative below deconfinement. The
Euler relation `eps + P = T s + sum_q mu_q n_q` then holds identically — the
bag terms cancel and `2 P_V` is exactly the shift `sum_q (mu_q - mu_eff_q) n_q`
— so the `verify/` check at 1e-8 tests the integrals and the assembly with no
cancellation left to hide an error.

**Charges.** `n_B = (n_u+n_d+n_s)/3`, `n_C = (2n_u-n_d-n_s)/3`, `n_S = n_s`,
and `mu_B = mu_u + 2 mu_d`, `mu_C = mu_u - mu_d`, `mu_S = mu_s - mu_d`. None
of these is written out here: they come from `eos.general.basis`, built from
the quantum-number table, with `S = +1` per s quark (opposite to PDG) and `C`
excluding leptons.

**Modes.** Every mode enforces the three self-consistency equations
`n_q(mu_eff_q, T, m_q) = n_q` and fixes `n_B`. Then:

| mode | conditions added |
|------|------------------|
| `beta_eq_neutrinoless`     | `mu_C + mu_e = 0`, `mu_S = 0`, `n_C = n_e` |
| `beta_eq_neutrino_trapped` | `mu_C + mu_e - mu_nue = 0`, `mu_S = 0`, `n_C = n_e`, `(n_e+n_nue)/n_B = Y_Le` |
| `fixed_YC`                 | `n_C = Y_C n_B`, `mu_S = 0` |
| `fixed_YC_YS`              | `n_C = Y_C n_B`, `n_S = Y_S n_B` (`mu_S` becomes an output) |

`leptons=True/False` applies to the two fixed-fraction modes: with leptons an
electron gas is added at the `mu_e` that makes the total system neutral
(`n_e = n_C`); without them the result is charged quark matter, which is what
a mixed phase needs per pure phase before imposing global neutrality. Photons
are a separate flag. Entropy per baryon may replace `T` anywhere, through an
outer 1-D solve.

**Numerics.** Unknowns are `(mu_u, mu_d, mu_s, n_u, n_d, n_s)` plus `mu_e`
where a lepton condition is present and `mu_nue` in the trapped mode — six to
eight, solved with a Powell hybrid method. Keeping `n_q` as unknowns rather
than substituting the mean field is what keeps the residual polynomial in `V`
instead of nesting the Fermi integrals inside it.

Convergence is judged on a dimensionless residual: density and charge
equations divided by `n_B`, potential equalities by `mu_B`, gated on the
largest scaled component at 1e-10. The equations carry mixed units (densities
~1e-1 fm^-3, potentials ~1e3 MeV), so a gate on the raw sum of squares is
dominated by whichever equation is largest and accepts states satisfying the
others only loosely. Non-convergence is a return value, not an exception.

**Not implemented** (see `docs/DEFERRED.md`): muons, `eos_response`.
