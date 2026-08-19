# NJL — three-flavour Nambu–Jona-Lasinio quark matter, with colour superconductivity

The full description, with equations and bibliography, is `njl.tex`. The
implementation specification this follows is `docs/njl_csc_implementation.md`,
which is the authority wherever the two differ. This file is the plain-text
summary.

**Model.** A contact four-fermion theory of the light quarks: the scalar
channel that breaks chiral symmetry, the 't Hooft determinant that ties the
three flavours together, a colour-antitriplet diquark channel that condenses
into a colour superconductor, and a vector channel whose repulsion sets the
high-density stiffness. Rehberg, Klevansky & Hüfner, PRC 53, 410 (1996) for
the parameter set; Rüster et al., PRD 72, 034004 (2005) for the neutral
three-flavour pairing sector; Alford, Schmitt, Rajagopal & Schäfer, RMP 80,
1455 (2008) for the review.

    L = qbar(i d_slash - m_hat) q
      + G_S sum_a [(qbar tau_a q)^2 + (qbar i g5 tau_a q)^2]
      - K {det_f[qbar(1+g5)q] + det_f[qbar(1-g5)q]}
      + G_D sum_eta (qbar i g5 eps_eta lam_eta C qbar^t)(q^t C i g5 eps_eta lam_eta q)
      - G_V (qbar gamma^mu q)^2

Almost nothing is an input. The constituent masses come out of the gap
equation

    M_u = m_u - 4 G_S phi_u + 2 K phi_d phi_s     (and cyclic)

the effective bag constant is a *derived* vacuum pressure difference,
B_eff^(1/4) = 228.93 MeV = 357.49 MeV/fm^3, and which colour-superconducting
pattern the matter is in is an *outcome* chosen by free energy, not a
declaration.

**Conventions.** Nine colour-flavour modes j = (f,a), flavour-major.
Strangeness S = +1 per s quark (the repository's sign, opposite to the PDG).
C is the charge of strongly-interacting matter only. The colour generators are
T_3 = diag(1/2, -1/2, 0) and **T_8 = diag(1, 1, -2)/3 = lambda_8/sqrt(3)** —
three normalisations of T_8 are in circulation and mixing them corrupts mu_8
by 15–70%; Rüster, Pagliara–Schaffner-Bielich and Kunkel use the halved
Gell-Mann form, for which mu_8^theirs = 1.1547 mu_8^ours.

**What is shared, and why.** The gap matrix, the 18x18 Bogoliubov–de Gennes
problem, the pairing correction to Omega and the Hellmann–Feynman kernels are
NOT in this package: they are `eos/general/pairing.py`, shared with the
chiral colour-dielectric model, because the pairing sector of the two is the
same sector (CLAUDE.md §7). The cut medium integrals ARE here, under §7's
carve-out for cutoff-regularized NJL integrals, which are model physics.

**Modes.** All four of CLAUDE.md §3, at any temperature, plus the entropy axis.
`leptons=True/False` on the fixed-fraction modes. Species flags: `csc`,
`muons`, `thermal_neutrinos`, `photons`; a flag this model does not have
(`hyperons`, `deltas`, `thermal_mesons`) raises rather than being ignored.

**Parameter tiers** (CLAUDE.md §6: parameters are arguments, never globals):

- tier 1, fixed by vacuum physics, never sampled: Lambda = 602.3 MeV,
  G_S Lambda^2 = 1.835, K Lambda^5 = 12.36, m_u = m_d = 5.5, m_s = 140.7 MeV;
- tier 2, structural, declared per run: the vector-coupling *form*, the
  regularization scale lambda = Lambda_UV/Lambda, the lepton content;
- tier 3, the Bayesian vector: eta_D = G_D/G_S, eta_V = G_V/G_S, or
  (G_V0/G_S, M_g) for the density-dependent vector variant.

Published sets: `rkh` (the default), `kunkel` (eta_D = 1.45, eta_V = 0.7 —
their *couplings*, at lambda = 1 rather than their RG-consistent scheme; see
DEFERRED), `gluon_exchange` (the recommended vector variant).

## The five traps

Each of these returns a plausible-looking wrong answer.

1. **P must come from the logarithm form when the integral is cut.** The two
   standard pressure integrals differ by a surface term that does not vanish
   at a finite cutoff: 0.1% of P at (M, mu, T) = (100, 500, 20) MeV and 39.9%
   at (140, 700, 50). At T = 0 below the cutoff they agree, which is exactly
   how the error hides until a table is built at finite temperature.

2. **The gap kernel is not Delta/|E|.** Differentiating |E| as though every
   branch were positive is wrong by a factor 12.0 at Delta = 40 MeV in the
   gapless window, and makes the gap *grow* with the Fermi-surface mismatch —
   the opposite of the physics. The kernel is Hellmann–Feynman on the BdG
   matrix, which carries the branch sign for free.

3. **Paired densities and entropy are not the unpaired Fermi integrals.** At a
   2SC point the unpaired density formula is out by -21% on the paired u modes
   and +12% on the paired d modes; in a fully gapped phase the unpaired
   entropy is four orders of magnitude too large at T = 5 MeV.

4. **The gap equation has three roots** under any mismatch — zero, a barrier
   maximum and the physical BCS root — so a fixed bracket silently returns the
   wrong one. Scan, then bracket each sign change.

5. **mu_8 is unconstrained in an unpaired region**, where n_8 vanishes
   identically at mu_8 = 0. It is pinned there, never solved for.

## What the implementation reproduces

| quantity | computed | published |
|---|---|---|
| M_u (vacuum) | 367.648 MeV | 367.7 |
| M_s (vacuum) | 549.479 MeV | 549.5 |
| (-phi_u)^(1/3) | 241.946 MeV | 241.9 |
| (-phi_s)^(1/3) | 257.688 MeV | 257.7 |
| f_pi (quark loop) | 92.391 MeV | 92.4 |
| B_eff^(1/4) | 228.93 MeV | 357.49 MeV/fm^3 |
| 2SC gap at mu* = 450 MeV, eta_D = 0.75 | 92.71 MeV | — |
| Clogston ratio dmu_c/(Delta_0/sqrt2) | 0.970 | 1 (weak coupling) |

and the two neutral solved points at mu_B = 1500 MeV, T = 0, eta_D = 0.75:

    unpaired   M = (9.84, 8.55, 265.59) MeV, mu_C = -34.20,
               n_B = 1.4319 fm^-3, P = 302.12 MeV/fm^3
    2SC        Delta_3 = 95.50, mu_3 = 0, mu_8 = -2.46, M_s = 243.13 MeV,
               mu_C = -62.27, n_B = 1.4887 fm^-3, P = 324.75 MeV/fm^3

**One deliberate discrepancy.** The specification reports M_u, M_d =
(11.96, 7.65) MeV for the 2SC point and this implementation gives
(9.73, 8.90). The difference is the sign of the hole amplitudes in the paired
scalar density, and `n_B = dP/dmu_B` along the neutral solution decides it: it
holds to the finite-difference floor with the sign used here and fails by
2.6e-4 with the other. Near chiral restoration M - m is the small difference of
two large numbers, which is why a percent-level change in one scalar density
moves the light masses by 20% and moves M_s, Delta_3, mu_8, mu_C, n_B and P
not at all. Recorded in `docs/DEFERRED.md`.

## Why the vector coupling depends on the density

With chiral symmetry restored the scalar channel dies and the high-density
behaviour is set entirely by the vector term. At constant G_V the interaction
energy grows like n^2 against the kinetic n^(4/3) and c_s^2 runs away to 1
(Zel'dovich). Writing eps = sum_i C_i n^p_i gives

    c_s^2(n -> inf) = max(1 - alpha, 1/3)     for G_V ~ n^(-alpha)

so alpha = 2/3 is the marginal exponent — and it is marginal *identically*,
not asymptotically: there the vector term's own pressure is exactly one third
of its own energy density at every density. The gluon-exchange form

    G_V(n_q) = G_V0 / [1 + 8 k_F^2/(9 M_g^2)],   k_F = (pi^2 n_q/2)^(1/3)

reaches alpha_eff = 2/3 with no tuning (0.062, 0.460, 0.608, 0.653 at
n_q = 1e6, 1e8, 1e9, 1e10 MeV^3), which is why it is the recommended variant.
Once G_V depends on n the rearrangement term is mandatory: Sigma_V = dW/dn =
(2 - alpha) G_V n, not 2 G_V n — a 5% error in P otherwise.

## Layout

    parameters.py      Parameters: the RKH set and the three coupling tiers
    couplings.py       G_V as a function of the state, and its rearrangement
    species.py         SpeciesFlags, the quantum numbers, the gap patterns
    thermodynamics.py  quantities computed FROM the state, and the internal
                       solve at fixed conserved-charge potentials
    solver.py          the equilibrium conditions and the pattern enumeration
    table.py           the warm-started density sweep + progress callback
    api.py             eos_point / eos_table / eos_response
    responses.py       second derivatives, by re-solved finite differences
    verify/            the physics invariants

Tests: `test/njl/` for the model, `test/general/test_pairing.py` for the shared
pairing machinery.

## Not implemented (see docs/DEFERRED.md)

RG-consistent regularization at lambda > 1 (the parameter exists and any value
but 1 raises rather than returning a divergent number); the 't Hooft–diquark
cross-term K'; the trapped muon lepton family as a conserved charge; the
composition and gap freezes of `eos_response`; the dilaton/colour-dielectric
graft.
