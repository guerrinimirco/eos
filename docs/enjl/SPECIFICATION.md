# `eos/enjl/` — the finite-temperature extension, and what comes after

The T = 0 model is implemented and described in `eos/enjl/enjl.tex`, which
states every equation the code solves, the ten-unknown beta-equilibrium
residual row by row, the modes that raise and why, and what the
implementation reproduces measured against the paper and against the author's
own tables. **That document, not this one, is the specification of what
exists.** What is left here is the design of what does not exist yet.

Sources:

* **Paper 1** — C.-J. Xia, Phys. Rev. D **110**, 014022 (2024),
  arXiv:2405.02946. Uniform matter, beta equilibrium, TOV. *All equation
  numbers in this document refer to Paper 1 unless stated otherwise.*
* **Paper 2** — C.-J. Xia, T. Maruyama, N. Yasutake, T. Tatsumi,
  arXiv:2409.12489. Same Lagrangian with gradient terms, Thomas-Fermi
  approximation, mixed-phase geometrical structures.
* **Reference tables** — `test/enjl/reference/`, the author's own Maple output.
  Their column semantics and the identities they satisfy are documented in
  `docs/enjl/REFERENCE_TABLES.md`, which was written from a numerical audit and
  **must be read before touching the solver**. Several columns do not mean what
  their names suggest.
* **Audit script** — `docs/enjl/verify_reference_tables.py`. An independent
  rebuild of the mean field from the published parameters, depending on nothing
  in `eos/`. If it and `eos/enjl/` disagree, one is wrong.

## What is done

The T = 0 reproduction is complete except for the branch rule. Milestones E0-E3
and E5 of the original plan are green (`test/enjl`, 89 tests): the reference
loader, the fixed-composition solve against all five tables, the
beta-equilibrium solver against the composition, and the paper-level physics
checks. `eos/enjl/verify/run_full_check.py` carries the invariants.

**E4, the Maxwell construction and the branch rule, is NOT done**, and it is the
one open item at T = 0. Locate the first-order transitions by maximizing P at
fixed mu_b. *Gate:* the four coexistence windows of `REFERENCE_TABLES.md` §5
reproduced, P to 0.1%, both edge densities to 1%. Remember that mu_b is the
`munr` column, which in the quark phase is mu_u + 2 mu_d.

E6, TOV, follows E4: Paper 1 Fig. 8 gives M-R curves for all six (f_q, B)
combinations, and Paper 1 stitches a DD-LZ1 crust below the core-crust
transition by Maxwell construction — a crustless star is ~0.9 km smaller at
1.4 Msun, the same trap already documented for the DD2 TOV test.

## 3. Extending to finite temperature

This is the part where `eos/general/` does the work, and where the T = 0 code
must not be duplicated.

### 3.1 The whole extension is one function, and the seam is already cut

Every T-dependence in the model enters through the single-species kinetic block
`(n, P, ε, s, n^s)`. Nothing else changes form: the couplings are functions of
n_b, the gap equation is the same equation with T-dependent scalar densities,
the rearrangement terms have the same expressions, E₀ is the same number. So
the extension is:

> **Add a `T` argument to `eos.enjl.thermodynamics.kinetic_thermo` and pass it
> through to `eos/general/fermi_integrals.py` for the medium part, keeping the
> vacuum term analytic.**

`kinetic_thermo(nu, m, g, Lambda)` already has that shape: it calls
`solve_fermi_jel(nu, 0.0, m, g, include_antiparticles=False)` for the medium
part, converts with hc3, and subtracts `vacuum_energy(m, g, Lambda)` and
`vacuum_scalar_density(m, g, Lambda)` — both of which are T-independent by
construction. It is the only function in the model that touches an integral,
and every caller goes through it. So the change is the signature and the two
arguments passed on:

```python
def kinetic_thermo(nu, m, g, Lambda=0.0, T=0.0, antiparticles=False):
    n, P, eps, s, n_s = solve_fermi_jel(nu, T, m, g,
                                        include_antiparticles=antiparticles)
    return (n * hc3, P * hc3, eps * hc3 - vacuum_energy(m, g, Lambda),
            s * hc3, n_s * hc3 - vacuum_scalar_density(m, g, Lambda))
```

The caller-side work is larger than that and is the real cost: at T > 0 the
Fermi surface is not sharp, so `thermo_from_n`'s density → k_F → ν path has to
become a density → ν inversion (`eos.general.fermi_integrals`
`invert_fermi_density`), `solver.state_at`'s `_fermi_momentum` clamp has to
become a clamp on ν, and the whole T = 0 branch has to stay reachable and
bit-identical.

### 3.2 What `eos/general` gives you, and its edges

`solve_fermi_jel(mu, T, m, g, include_antiparticles=True)` returns
`(n, P, e, s, ns)` in **fm-based units** — n and s in fm⁻³, P and e in
MeV/fm³. `CLAUDE.md` §3 forbids leaking those across a module boundary: convert
with `hc3` inside `thermodynamics.py` and let every call site stay in natural
units.

Verified behaviour, checked against the exact T = 0 forms before writing this:

* `T < 1e-4` routes to `_compute_exact_T0`, and over a grid of
  M ∈ {5.5, 140.7, 300, 550, 939} MeV, g ∈ {2, 6},
  k_F ∈ {50, 200, 500, 1200, 2200} MeV at T ≤ 10⁻⁵ MeV its output agrees with
  `number_density_t0`, `eps_kin_t0` and `scalar_density_t0` (Λ = 0) to
  worst-case relative deviations of 2×10⁻¹⁵ (n), 2×10⁻¹³ (ε), 9×10⁻¹³ (n^s)
  and 4×10⁻¹⁰ (P, formed as νn − ε and so the most cancellation-prone).
  **So the T → 0 limit of the new path reproduces the old path to round-off**
  once the vacuum term is added. That is gate T1, and 10⁻⁸ relative is a safe
  threshold for it.
* `e` includes the rest mass, the same convention as Eq. (13). No offset.
* `ns` is computed from the trace identity (ε − 3P)/m, exact for a free gas.
* `m < 1e-5` routes to the ultra-relativistic limit and returns `ns = 0`. Quark
  masses bottom out at m_q0 = 5.5 MeV so this branch is never reached here, but
  do not pass a massless species through expecting a scalar density.
* Accuracy is ~10⁻⁴ (JEL's design target). `solve_fermi_gl` is the slower,
  more accurate alternative and `CLAUDE.md` §6 requires JEL to remain
  selectable — add alternatives, never replace it.
* `invert_fermi_density(n_target, T, m, g)` inverts n → µ by Brent, useful if a
  fixed-composition entry point at T > 0 is wanted.

### 3.3 Three physics decisions that are yours to make, not to assume

None of these is settled by Paper 1, which is a T = 0 paper. `CLAUDE.md` §10
says ask rather than pick silently — so each gets an explicit entry in
`docs/DEFERRED.md` with the choice made and the reasoning, and if the
reasoning is thin, ask.

1. **Thermal antibaryons.** Paper 1 states that the baryonic Dirac sea does not
   exist, "since quarks no longer form clusters in the Dirac sea due to Pauli
   blocking". A thermal antibaryon population is a different object from the
   Dirac sea — it is a real excitation at high T — but a model in which baryons
   exist only as bound quark clusters gives no obvious licence for antibaryon
   clusters. Default: `antiparticles=False` for p, n, Λ, on the grounds that it
   is the choice continuous with the paper's own statement. Flag it; it matters
   above T ≈ 50 MeV.
2. **Thermal antiquarks.** Here the answer is cleaner: the Dirac sea is handled
   explicitly by the Λ subtraction, which is the T = 0 filled negative-energy
   states; the JEL antiparticle branch is the *thermal* antiparticle population
   on top of it, and the two do not double count. Default:
   `antiparticles=True` for u, d, s. Leptons likewise.
3. **The quarkyonic restriction.** Paper 1 §II says that in the quarkyonic
   phase "only baryons can be exited to higher energy states while quarks are
   restricted to the lowest energy states as they are still confined". At T = 0
   this is what filling from the bottom means and there is nothing to do. At
   T > 0 an ordinary Fermi-Dirac distribution *does* excite quarks, so the
   statement is either (a) a T = 0 kinematic remark with no finite-T content,
   or (b) a genuine constraint that would need the quark occupation modified.
   Default: (a). This is the most consequential of the three and the one most
   worth asking about.

There is also one thing that is **not** a decision: **do not add a thermal
meson gas.** In this model σ, ω, ρ are auxiliary fields eliminated in favour of
g²/m², and Paper 1 leaves the meson masses undetermined precisely because only
the ratios enter. `eos/general/bose_integrals.py` exists and is the right tool
in a model with dynamical mesons; this is not one. (Paper 2 does assign
m_σ = 630, m_ω = 10⁵, m_ρ = 769 MeV, but only because its gradient terms need
the interaction ranges — and note that its m_ω is *deliberately unphysical*,
chosen large to suppress density fluctuations in the Thomas-Fermi solve. That is
not a particle mass you could put in a thermal gas even if you wanted to.)

### 3.4 Consequences to get right

**Pressure.** Eq. (19) is the T = 0 special case of the Euler relation. At
T > 0 use

```
P = T s + Σ_i µ_i n_i − E
```

with s summed over the kinetic blocks. This is exact for a density-dependent-
coupling mean field **provided Σ^R appears in µ_i and never in E** — the
invariant of `CLAUDE.md` §7 that makes DD-RMF thermodynamically consistent.
Getting it wrong yields an EoS that looks plausible and violates
Hugenholtz-Van Hove.

**Free energy.** `f = E − T s = −P + Σ_i µ_i n_i`. Both forms must agree; that
is a cheap and sharp consistency test.

**The gap equation gains a thermal channel.** nbar^s_q now falls with T at
fixed density, so chiral restoration happens partly thermally and the
transition lines in the (n_b, T) plane are not vertical. Expect the T = 0
first-order transitions to weaken and end at critical points as T rises. That
is physics, not a bug — but it means the sign of a converged root can change
across a temperature line, so the solver needs the same
continuation/warm-start discipline in T as in n_b.

**Warm starting.** Follow the DD2 pattern: sweep n_b at fixed T seeding each
point from the previous, and seed each temperature line from the previous line.
The quark masses are the natural continuation variables; they move smoothly
where µ-based unknowns do not.

**Entropy per baryon as an axis.** For anything astrophysical, a fixed-S/A
table is more useful than fixed T. `eos/mixed/tables/generate.py` already
implements the outer one-dimensional solve for T at given S/n_b; consume that
pattern rather than writing a second one.

### 3.5 Milestones for finite temperature

**T1 — the kinetic block.** Add `T` and `antiparticles` to
`thermodynamics.kinetic_thermo`, keep the vacuum term analytic, and pin the
reference/fast split: the exact T = 0 branch is the `_ref` against which the
JEL branch is judged.
*Gate:* for a grid of (ν, M, g, Λ) covering baryons and quarks, the JEL branch
at T = 10⁻⁶ MeV reproduces the exact T = 0 branch to 10⁻⁸ relative in n, P, ε
and n^s. Also assert the vacuum term is T-independent, by construction and by
test.

**T2 — the T = 0 limit of the whole engine.** Route `solve_point` and
`solve_beta_point` through the T-capable block with `T = 0.0`.
*Gate:* every E2 and E3 number unchanged, bit-for-bit where the T = 0 branch is
taken. A regression here means the plumbing changed the physics.

**T3 — thermodynamic consistency at T > 0.** No reference data exists, so the
gate is internal identities on a (n_b, T) grid spanning
n_b ∈ [0.05, 5] fm⁻³, T ∈ [0, 100] MeV:
Euler `E + P = T s + Σ µ_i n_i`; `f = E − Ts = −P + Σ µ_i n_i`;
`s → 0` as `T → 0`; `s ≥ 0` everywhere; `∂P/∂T|_n ≥ 0`; `∂s/∂T|_n ≥ 0`;
0 ≤ c_s² ≤ 1 along isentropes; and Maxwell relation
`∂s/∂n_b|_T = −∂µ_b/∂T|_n` by finite difference.
*Gate:* all of the above to 10⁻⁶ relative, or a stated and justified looser
tolerance where the JEL 10⁻⁴ accuracy floor binds.

**T4 — limits and sanity.** High-T ideal-gas limit for the quark sector;
S/A table generation; the (n_b, T) chiral transition line and its critical
point.
*Gate:* the phase boundary is continuous and single-valued in T; the critical
point is located to within the grid spacing and reported.

## 4. Later: mixed phases with the general framework

Deferred, and deliberately not specified here. The eventual target is Paper 2's
Thomas-Fermi/Wigner-Seitz treatment, and the connection to what already exists
in this repository is:

* `eos/mixed/` solves two-phase coexistence for **two different engines** (DD2
  hadronic + vMIT quark) with a continuous local/global charge-neutrality
  parameter η. Its `PhaseThermo` adapter interface is engine-agnostic by
  design, so an ENJL adapter is the natural entry point — but note that ENJL is
  *one* engine describing both phases, so "hadronic phase" and "quark phase"
  become two branches of the same functional rather than two models. Whether
  the η machinery applies unchanged is an open question, not a given.
* Paper 2 goes further than η: it solves the Klein-Gordon equations with
  gradient terms in a WS cell and gets droplet/rod/slab/tube/bubble structures
  and surface tensions directly. That needs the meson masses of Paper 2
  Table I, which Paper 1 leaves free — **m_σ = 630 MeV, m_ρ = 769 MeV, and
  m_ω = 10⁵ MeV.** That last one is not a typo and not a physical ω mass: Paper 2
  states it adopted "a rather large ω meson mass" to prevent density
  fluctuations in the Thomas-Fermi approximation, and notes that m_ω could be
  reduced to more reasonable values with a better treatment. Anyone reading the
  table alone will mistake 10⁵ for 105 — a factor of a thousand in the ω
  interaction range — so carry the caveat with the number. A spatial solver for
  the Klein-Gordon equations does not currently exist anywhere in this
  repository either.

Do not start either of these until §2 and §3 are green. When the time comes,
they get their own specification.
