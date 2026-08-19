# DID / DIDY — density- AND isospin-density-dependent couplings

The full description, with equations and bibliography, is `did.tex` (compiled
against `../../docs/eos.bib`). This file is the plain-text summary.

**Model.** A relativistic mean field of the DD-RMF family in which every
baryon-meson coupling depends on the isospin asymmetry as well as on the
density. Frohaug, Maslov, Dexheimer, Grefa, Jahan, Ratti & Restrepo,
arXiv:2511.15646. Baryon octet coupled to sigma, omega, phi and rho; no delta
meson and no f0(980), both deliberately.

    g_Mi(n_B, beta) = [1 - w] g^S_Mi(n_B) + w g^N_Mi(n_B),   w = beta^2 tanh(x/e)

    g^{S,N}_Mi(n_B) = g^{S,N(0)}_Mi F_M(x)

    F_M(x) = exp[1 - ((x+1)/2)^(2 a_M)] (1 - t)/2 + b_M (1 + t)/2,
             t = tanh[(x - c_M)/d_M],   x = n_B/n_0,   e = 1/3

with beta = sum_i tau_3i n_i / n_B, so beta = 0 in symmetric matter and -1 in
pure neutron matter. `g^S` is the branch fitted in symmetric matter, `g^N` the
one fitted in neutron matter, and the `tanh(x/e)` factor makes the couplings
isospin-independent at zero density (which is what keeps Sigma^t finite there).
The shape `F_M` belongs to the MESON, the two strengths to the VERTEX; that
factorisation is the paper's own and is why one shape serves every baryon.

**Why the isospin dependence exists.** Lambda is an isoscalar, so in an
ordinary DD-RMF its potential cannot depend on the asymmetry through the rho.
HAL QCD-based Brueckner calculations say `U_Lambda` becomes LESS attractive in
neutron matter (-28.15 in ISM, -25.42 in NM); a conventional model gives the
opposite sign of the trend. Making `g_sigma` and `g_omega` depend on beta fixes
it, and the later hyperon onsets that follow are what keep `M_max` above 2
solar masses with the full octet.

**Conventions** (both stated in `parameters.py`, neither ever silently
flipped):

- `tau_3 = 2 I_3`, normalised to +/-1 for nucleons, so Sigma+/- carry +/-2 and
  Xi0/Xi- carry +/-1, Delta carries +/-3, +/-1. This is NOT
  `eos.general.particles.Particle.t3`, which is the DD2 rho-coupling
  convention where the Sigma factor of two sits in the coupling ratio instead.
  DID fits `g_rho Sigma` and `g_rho Xi` independently and so has no ratio to
  absorb it.
- `S = +1 per s quark` (the repository convention), the opposite of the
  paper's PDG sign. It cancels wherever mu_S = 0 and is carried by
  `eos.general.basis` elsewhere.

**Parameters.** `Parameters.default()` is the published maximum-likelihood set
(Table II, transcribed digit for digit; `n_0 = 0.15880045` from Table III, the
density at which P = 0 in symmetric matter). DID and DIDY are the SAME
parameter set — what distinguishes DIDY is `SpeciesFlags(hyperons=True)`.
`named("DID")` and `named("DIDY")` both return it, and say so.

The omega and phi couplings are DERIVED, not stored: the fit varies the
aggregated strength

    g~_omegaN = g_omegaN sqrt(1 + [(g_phiN/m_phi)/(g_omegaN/m_omega)]^2)

and the SU(3) ratio z (with alpha = 1 and ideal mixing), and
`couplings.g8_from_aggregate` inverts that in closed form. Note that DID gives
the NUCLEON a phi coupling (g_phiN = -5.20 at the published set), so the phi
field is active at every composition; `SpeciesFlags(phi_field=False)` raises.

One correction to the paper: its Eq. (6) prints the g_phiXi coefficient with a
factor 2 that breaks all four SU(6) limits (it would give
`g_phiXi = -sqrt2 g_omegaN` instead of `-2 sqrt2/3 g_omegaN`). The paired form
is used here and the SU(6) limit is asserted in `verify/`.

**Two rearrangement terms.** Because the couplings depend on two state
variables, there are two:

    Sigma^r = sum_i [ -dg_sigma_i/dn_B sigma n^s_i + dg_omega_i/dn_B omega n_i
                      + dg_phi_i/dn_B phi n_i + dg_rho_i/dn_B tau_3i rho n_i ]

    Sigma^t = (1/n_B) sum_i [ the same with d/dbeta ]

    mu_i = nu_i + g_omega_i omega + g_phi_i phi + g_rho_i tau_3i rho
                + Sigma^r + (tau_3i - beta) Sigma^t

Both enter mu and P, neither enters eps. The `(tau_3i - beta)` weight comes
from the chain rule `d/dn_i = d/dn_B + [(tau_3i - beta)/n_B] d/dbeta`, and it
has a consequence worth knowing: `sum_i (tau_3i - beta) n_i = 0`, so Sigma^t
cancels identically out of P, eps and `sum_i mu_i n_i`. It shifts the
individual chemical potentials and nothing else — which is exactly what splits
the Sigma and Xi single-particle potentials in neutron matter (the Sigma
splitting there, 17.9 MeV, is carried almost entirely by Sigma^t: g_rho Sigma
is 0.0055).

**The unknown vector** is larger than an ordinary DD-RMF's for the same reason:

    x = [sigma, omega, rho, phi, beta, Sigma^t, mu~_B, mu_C,
         (mu_S), (mu_nue), (T)]

beta and Sigma^t are solved for, each with its own defining row, because the
couplings depend on beta and Sigma^t shifts the potentials whose densities
define both. `mu~_B = mu_B - Sigma^r` is the kinetic baryon potential. Rows:
four field equations, the beta definition, the Sigma^t definition, baryon
number, the charge row (neutrality or held Y_C), then mu_S / mu_nue / entropy
rows as the mode demands.

**Modes** (all four of CLAUDE.md §3, plus an entropy-per-baryon axis anywhere
a temperature is taken):

    beta_eq_neutrinoless        (n_B, T)
    beta_eq_neutrino_trapped    (n_B, Y_Le, T)
    fixed_YC                    (n_B, Y_C, T)   leptons=True/False
    fixed_YC_YS                 (n_B, Y_C, Y_S, T)   needs hyperons

**Species flags.** `hyperons`, `deltas`, `muons`, `thermal_mesons`,
`thermal_neutrinos`, `photons`, `phi_field`. The paper carries electrons only;
muons are an option here and move the hyperon onsets up by ~0.03 fm^-3. Deltas
are an extension (ratio scheme on the nucleon vertex, both branches), not in
the paper.

**Usage.**

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

**What it reproduces** (all from the transcribed Table II, none of it fitted
here):

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

`python -m eos.did.verify.run_full_check` asserts all of it except the last
two rows; add `--tov` for those (it costs ~40 s).

**Layout.**

    couplings.py       the functional form: F_M(x), the blend, SU(3), no numbers
    parameters.py      the numbers, the multiplet table, couplings_at(n_B, beta)
    species.py         SpeciesFlags and the active baryon list
    thermodynamics.py  kinetics, fields, both Sigma terms, assembly, and
                       thermo_at_potentials (the phase-adapter surface)
    solver.py          the residual, the modes, the warm start
    nmp.py             the forward nuclear-matter map + the Delta inversion
    table.py           the grid driver over eos.general.tabulate
    responses.py       c_s^2 (isothermal and adiabatic), C_V, C_P, Gamma_th
    api.py             eos_point / eos_table / eos_response
    verify/            the physics invariants, and the TOV cross-check

**Not implemented** (see `docs/DEFERRED.md`): the nuclear-statistical-
equilibrium cluster description below saturation (Section III of the paper —
`eos/tov` attaches a crust instead), the inverse NMP map, a trapped muon
family, and the phi contribution to the thermal kaon potentials.
