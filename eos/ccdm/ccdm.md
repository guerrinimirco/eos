# CCDM — chiral colour-dielectric quark matter, with colour superconductivity

The full description, with every equation and the bibliography, is `ccdm.tex`.
The implementation specification this follows is
`docs/ccdm_implementation.md`, which is the authority wherever the two differ
— with the two exceptions recorded under "Two corrections" below, where the
specification contradicts the thermodynamic audit it itself mandates. This
file is the plain-text summary.

**Model.** A mean-field theory of deconfined u, d, s matter in which
confinement and chiral symmetry breaking are two faces of one mechanism. A
dilaton field carries the gluon condensate; the dielectric function built from
it measures how transparent the medium is to colour, and it sits in the
*denominator* of the quark masses:

    chi = (1 - phi_bar^4)^p ,   p = 1 ,   phi_bar = phi/phi_0

    M*_u,d = (g_q sigma + m_u,d)/chi ,     M*_s = (g_s zeta + m_s)/chi

So as the condensate reaches its vacuum value (phi_bar -> 1) the medium goes
opaque, chi -> 0, the effective masses diverge, and the quarks leave the
medium entirely. At the perturbative point (phi_bar -> 0) chi -> 1 and the
masses return to the current ones.

**Why the fourth power.** phi_bar^4, not phi_bar, is the gluon condensate: the
dilaton is a canonically normalised scalar of mass dimension 1 while
<G^A_munu G^A^munu> has dimension 4. The scale anomaly makes it sharp — the
combination it fixes, 4U - phi dU/dphi, comes out a pure power of phi_bar only
for exponent 4, and any other power leaves a residual phi_bar^n ln phi_bar. So
1 - phi_bar^4 = 1 - <G^2>_med/<G^2>_vac, and the deviation from transparency is
linear in the condensate.

**Confinement is a pinning, not a suppression.** At T = 0 a mode with
M* >= mu* contributes *identically zero* — exactly 0.0, not a small number.
That is the mechanism, and it is what makes deconfinement first order here
rather than a crossover. Smoothing the threshold produces a plausible-looking
equation of state with the transition destroyed.

**The bag constant is derived, not input.** Both potentials vanish at the
physical vacuum by construction, so no vacuum subtraction appears anywhere.
At the perturbative point the field energy is

    B_eff = [U(0) - U(phi_0)] + [V(0,0) - V(f_pi, zeta_0)] = B_g + B_chi

and at the shipped set B_g^(1/4) = 150 MeV against B_chi^(1/4) = 229.9 MeV,
giving B_eff = (239.7 MeV)^4 = 429.4 MeV/fm^3. **The chiral sector supplies
the larger part** — quoting B_g alone as "the bag constant" is wrong by a
factor of six in energy density.

**Conventions.** Nine colour-flavour modes j = (f,a), flavour-major.
Strangeness S = +1 per s quark (the repository's sign, opposite to the PDG).
C is the charge of strongly-interacting matter only, and total electric
neutrality is a separate condition. The colour generators are
T_3 = diag(1/2, -1/2, 0) and **T_8 = diag(1, 1, -2)/3 = lambda_8/sqrt(3)** —
three normalisations are in circulation and mixing them corrupts mu_8 by
15–70%; the conversions are in `eos.general.pairing`.

**The solve variable is Phi = phi_bar^4, not phi_bar.** Written in phi_bar the
dilaton residual has a spurious root at phi_bar = 0, where both of its terms
vanish as phi_bar^3, so it is satisfied there for *any* scalar density — an
artefact of the parametrisation, since the Jacobian dPhi/dphi_bar = 4 phi_bar^3
vanishes and not the physics. A Newton solve in phi_bar landed on it from three
of five starting points. In Phi the same equation reads dU/dPhi = B_g ln Phi,
which runs to -infinity there, and no such root exists.

**The vector unknown is Sigma_V, not omega_0.** The field is circular: omega_0
sets mu*, which sets n_B, which sets omega_0. Carried as the total shift
Sigma_V = g_omega omega_0 + Sigma_R everything downstream is explicit, and the
R_4 row is the single statement that the returned field is the one the returned
densities source. The vector source is the *quark* density n_q = 3 n_B: using
n_B understates omega_0 by three and the repulsive energy by nine.

**Two enumerations, not one.** Neither is a mode; both are decided by free
energy.

  - the chiral/dielectric *branch* — confined (the vacuum: n_B = 0, P = 0
    exactly), deconfined-restored, deconfined-partially-restored. A solver
    that alternates between updating sigma and omega_0 two-cycles between the
    first two and exits with a mixed state, which reads as a spuriously deep
    minimum at zero quark density. Each is seeded separately, solved to
    self-consistency and compared by Omega; a branch that fails is reported
    missing, never replaced by a neighbour.
  - the pairing *pattern* — unpaired, 2SC, uSC, dSC, CFL, and one asymmetric
    free seed. The gap equation has three roots at any Fermi-surface mismatch
    (zero, a barrier maximum, the physical BCS root), so the seed decides.

The candidate set at fixed density is their *product*: which pattern survives
depends on the strange quark's effective mass, which is a property of the
branch. The confined branch is enumerated at fixed *potential* only — with the
dielectric closed n_B = 0 identically, so it cannot meet a nonzero density row.

**Where the model is defined.** Below the deconfinement onset there is no
deconfined root at fixed density at all: the quarks are not in the medium.
That comes back as a status, not an exception. At the shipped parameter point
the deconfined pressure crosses zero near n_B ≈ 1.35 fm^-3, and between the
onset and where the branch turns around dP/dn_B < 0 — the mechanically
unstable side of a first-order transition, which CLAUDE.md section 8 admits in
a raw branch and which a construction (Maxwell, Gibbs, or the eta-mixed phase
of `eos.mixed`) removes before any table reaches a structure solver. The
low-density half of a hybrid equation of state comes from a hadronic model
through `ccdm_phase`.

**Two corrections to the specification.** Both are forced by the Euler audit
its own section 9.6 mandates, and both are written out in `ccdm.tex` beside
its forms:

  - its section 4.3 writes eps with -(1/2) m_omega^2 omega_0^2. **The sign
    must be plus.** A repulsive vector interaction adds energy density: the
    Hamiltonian density of the mean vector field is
    g_omega omega_0 n_q - (1/2) m_omega^2 omega_0^2 = +(1/2) m_omega^2
    omega_0^2 once the field equation is used. This is the standard
    density-dependent mean-field result — the scalar potentials enter eps
    positively and P negatively, the vector term enters *both* positively;
  - its section 4.1 carries Sigma_R in mu* but omits the compensating
    -Sigma_R n_q from Omega. Without it n = -dOmega/dmu and Euler both fail as
    soon as g_omega depends on the density. The rearrangement term enters mu
    and P and **never** eps (CLAUDE.md section 8).

With either error present the Euler residual is of order percent while every
other quantity still looks like a reasonable equation of state; with both
corrections it is ~1e-16.

**Pairing.** The gap matrix, the 18x18 Bogoliubov–de Gennes problem, the
pairing correction to Omega, the Hellmann–Feynman gap kernels and the paired
densities and entropy are `eos.general.pairing`, shared with `eos.njl` because
the pairing sector of the two is the same sector. Four things there return
plausible wrong answers if done the obvious way: the pairing potential must be
written as a *correction* so it vanishes identically at Delta = 0; the gap
kernel is not Delta/|E| (wrong by a factor 12 in the gapless window); paired
densities and entropies are not the unpaired Fermi integrals; and the
antiparticle branches are not optional (8.8% of the pairing potential at
T = 0, Lambda = 600 MeV). The eigenvalues of the gap matrix alone are not
enough either, because it does not commute with the mass matrix once the quark
masses differ.

**The sign of each gap is a gauge.** Omega is invariant under flipping any
subset of the three Delta_eta and each kernel flips with its own gap, so
-Delta is a root whenever Delta is. What is reported is the magnitude.

**Modes.** All four of CLAUDE.md section 3, at any temperature, and the
specification's closure rows map onto them one for one:

    R1 cold star     beta_eq_neutrinoless          mu_S = 0, mu_e = -mu_C
    R3 proto-NS      beta_eq_neutrino_trapped      + Y_Le held, nu with g = 1
    R2 merger/CCSN   fixed_YC, leptons=True        LOCALLY NEUTRAL AND NOT
                                                   WEAK-EQUILIBRATED
    R4 heavy-ion     fixed_YC_YS, leptons=False
    R5 symmetric     fixed_YC_YS at Y_C = 1/2, Y_S = 0

R2 is worth stating twice: weak equilibrium is a per-row closure, never an
identity built into Omega, because hardwiring it would make merger and
supernova matter unrepresentable. Fractions are per baryon, so Y_S = n_S/n_B
differs by a factor three from the per-quark n_s/(3 n_B) often plotted.

**Parameters.** Tier 1 is fixed by vacuum data (f_pi, m_pi, f_K, m_K, the
current masses, m_zeta, m_phi, m_omega) and everything in the derived block
follows in closed form. Tier 2 is structural: p = 1 locked, q in {0,1} for the
dielectric dressing of G_D. Tier 3 is the Bayesian vector: B_g^(1/4), g_q,
g_s, m_sigma, gbar_omega, n_c, and at L3 G_D and Lambda.

`g_q = 3.0` is *pinned* by the specification's section 10 table
(M*_u,d = 826 MeV at phi_bar = 0.90 and 1531 at 0.95, confined — both invert
to 3.00). `g_s`, `gbar_omega` and `n_c` are calibration knobs at documented
mid-prior values and the code says so rather than dressing them as
measurements. `G_D = 5e-6 MeV^-2` was calibrated here to put the gap inside
the specification's 20–150 MeV window at mu_q ≈ 450 MeV.

**v_zeta^2 is negative** at the baseline m_zeta = 980 MeV: the strange quartic
is convex, explicit breaking dominating, so the strange sector does not break
chirally on its own in this truncation. The sign flips between m_zeta = 1100
and 1150 MeV. Never assume it is positive; never write v_zeta as a square
root.

**Not in the model.** Dilaton gradient and finite-size terms (this is a bulk
homogeneous mean field). The de Carvalho contact coupling h(phi) as a
replacement for G_D: their construction is sound and is *why* the pairing term
is a legitimate leading term rather than an ad hoc addition, but what it
predicts is a coupling negligible where the quarks are light and enormous
where they are heavy — the wrong way round for a compact-star core, and it
carries q = 4 at p = 1, violating q <= p. What is taken from it is the
argument, not the number. See `docs/DEFERRED.md`.

**Usage.**

    from eos.ccdm import Parameters, SpeciesFlags, eos_point, eos_table

    par = Parameters.default()
    r = eos_point(par, "beta_eq_neutrinoless", SpeciesFlags(csc=True),
                  n_B=1.5, T=0.0)
    r.ok, r.point.branch, r.point.pattern, r.point.Delta

    python -m eos.ccdm.verify.run_full_check
