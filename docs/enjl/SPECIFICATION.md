# `eos/enjl/` — specification and milestone plan

The extended NJL model of Xia, reproduced at T = 0 and then extended to finite
temperature on top of `eos/general/`.

Sources:

* **Paper 1** — C.-J. Xia, Phys. Rev. D **110**, 014022 (2024),
  arXiv:2405.02946. Uniform matter, beta equilibrium, TOV. *All equation
  numbers in this document refer to Paper 1 unless stated otherwise.*
* **Paper 2** — C.-J. Xia, T. Maruyama, N. Yasutake, T. Tatsumi,
  arXiv:2409.12489. Same Lagrangian with gradient terms, Thomas-Fermi
  approximation, mixed-phase geometrical structures. Relevant here only for the
  meson masses it fixes and as the target of a later phase.
* **Reference tables** — `test/enjl/reference/`, the author's own Maple output.
  Their column semantics and the identities they satisfy are documented in
  `docs/enjl/REFERENCE_TABLES.md`, which was written from a numerical audit and
  **must be read before touching the solver**. Several columns do not mean what
  their names suggest.
* **Audit script** — `docs/enjl/verify_reference_tables.py`. An independent
  rebuild of the mean field from the published parameters, depending on nothing
  in `eos/`. If it and `eos/enjl/` disagree, one is wrong.

---

## 0. What already exists, and the one thing that is broken

`eos/enjl/` is partially written and its ten P1 tests pass:

| module | state |
|---|---|
| `parameters.py` | complete. RKH set, Table I couplings, `RHO_FACTOR = 9.0`, `B_nat`. |
| `species.py` | complete. Quantum numbers, valence content, degeneracies. |
| `thermodynamics.py` | **T = 0 only.** Exact Eqs. (11)-(13) with the Λ cut-off. |
| `uniform.py` | complete. Fixed-composition mean-field solve, vacuum solution, E₀. |
| `eos_beta.py` | **does not import.** See below. |
| `star.py`, `tfa/` | do not exist; `__init__.py` advertises them. |

**`eos/enjl/eos_beta.py` raises `NameError` on import**: line 73 annotates
`pt: ENJLEoSPoint` but the import at line 35 pulls only
`GAP_TOL, _baryon_masses, _f_of, _m0_of, _N_BARYON, solve_point`. Add
`ENJLEoSPoint` to that import list. Nothing else in the module is wrong as far
as static reading goes, but it has never been executed, so treat every number
it produces as unverified until it is checked against the reference tables.

The docstring of `eos/enjl/__init__.py` promises `star.py` and `tfa/`. Either
build them or trim the docstring; a module list that lies is worse than a short
one.

## 1. The T = 0 model

### 1.1 Degrees of freedom

Baryons p, n, Λ (g = 2, B = 1); quarks u, d, s (g = 6, B = 1/3); leptons e, µ
(g = 2, B = 0). Repo sign conventions apply in full (`CLAUDE.md` §2): **S = +1
per s-quark**, and **C is the non-leptonic charge**. Paper 1 uses the physical
electric charge q_i directly in Eqs. (23)-(24) and never introduces a separate
C; when this engine is later coupled to `eos/mixed/`, the translation to
(B, C, S) is the adapter's job, not the model's.

### 1.2 The self-consistent system

Baryons are clusters of three quarks, so the *only* independent unknowns of the
mean field are the three quark masses. Everything else is algebraic:

```
nbar^s_q = n^s_q + α_S Σ_{i=p,n,Λ} N^q_i n^s_i                        (6)
M_q      = m_q0 − 4 G_S nbar^s_q + 2K nbar^s_u nbar^s_d nbar^s_s / nbar^s_q   (5),(8)
M_i      = Σ_q N^q_i [m_q0 + α_S (M_q − m_q0)] + B n_b^Q              (4)
```

`g_σ σ_q = 4 G_S nbar^s_q` is Eq. (5) with Eq. (8) substituted, and
`G_S = g_σ²/4m_σ²`. Verified against the tables to 2×10⁻⁵ relative.

Vector sector, Eqs. (9)-(10), written as couplings rather than fields since only
g²/m² is determined:

```
J_ω = Σ_i f_i n_i Σ_q N^q_i = 3(n_p + n_n + f_Λ n_Λ) + f_q (n_u + n_d + n_s)
J_ρ = Σ_i f_i τ_i n_i       = (n_p − n_n) + f_q (n_u − n_d)
g_ω ω = Γ_ω(n_b) J_ω,   g_ρ ρ = Γ_ρ(n_b) J_ρ
```

with τ_p = +1, τ_n = −1, τ_Λ = 0, τ_u = +1, τ_d = −1, τ_s = 0 and

```
Γ_ω = 4 G_S [a_V e^{−n_b/n_V} + b_V]                                  (21)
Γ_ρ = 9 × 4 G_S [a_TV e^{−n_b/n_TV} + b_TV]                           (22)
α_S = a_S e^{−n_b/n_S} + b_S                                          (20)
```

**The factor 9 in Γ_ρ is not in the printed Eq. (22).** It is required, it is
already in `parameters.py` as `RHO_FACTOR`, and it is now confirmed twice over:
by the symmetry energies S(0.1) = 25.5 and S(0.158) = 31.5 MeV, and — without
any fitting — by reading g_ρρ straight off the reference tables' isospin
splitting, which gives exactly 9.0000× Eq. (22) at every nucleonic density
(`REFERENCE_TABLES.md` §4c). Do not "fix" it back to the paper's literal form.

Rearrangement terms, Eqs. (17)-(18), unchanged in form:

```
Σ^R_b = Σ_i f_i [ ω n_i Σ_q N^q_i dg_ω/dn_b + ρ τ_i n_i dg_ρ/dn_b ]
      + Σ_{i=n,p,Λ} [ (dα_S/dn_b) Σ_q N^q_i (M_q − m_q0) ] n^s_i
Σ^R_q = (1/3) B Σ_{i=n,p,Λ} n^s_i + (1/3) Σ^R_b
```

In the g²/m² form the first line is
`½ (dΓ_ω/dn_b) J_ω² + ½ (dΓ_ρ/dn_b) J_ρ²`. Note the asymmetry of Σ^R_q: the
Pauli-blocking term B n_b^Q of Eq. (4) shifts the *quark* potential through
Σ^R_q, because n_b^Q is a quark density and the derivative of a baryon mass
with respect to it acts back on the quarks.

Chemical potentials, Eqs. (14)-(16) — note the factor 3 on ω for baryons
(three valence quarks) and its absence for quarks:

```
µ_b = √(ν_b² + M_b²) + f_b (3 g_ω ω + g_ρ ρ τ_b) + Σ^R_b               (14)
µ_q = √(ν_q² + M_q²) + f_q (g_ω ω + g_ρ ρ τ_q)   + Σ^R_q               (15)
µ_l = √(ν_l² + M_l²)                                                    (16)
```

Energy density Eq. (13) and pressure Eq. (19):

```
E = Σ_i ε_kin(ν_i, M_i, g_i)  −  Σ_{q} ε_vac(M_q, Λ)
  + 2 G_S Σ_q (nbar^s_q)²  −  4K nbar^s_u nbar^s_d nbar^s_s
  + ½ Γ_ω J_ω² + ½ Γ_ρ J_ρ²  −  E₀
P = Σ_i µ_i n_i − E
```

E₀ is fixed once, from the vacuum gap solution, so that E = 0 in vacuum:
`vacuum_energy_density()` gives **4263.845 MeV/fm³**, matching the Maple
worksheet's hard-coded constant and the tables' offset to 4×10⁻⁷.
**E₀ is a property of the vacuum and never changes** — not with density, not
with parameters other than (Λ, m_q0, G_S, K), and not with temperature.

### 1.3 The cut-off enters exactly one way

Only quarks carry Λ, and it enters only as a **temperature-independent additive
vacuum term**. Writing y = Λ/M:

```
n^s(ν, M, g, Λ) = n^s_med(ν, M, g)  −  (g M³/4π²) [ y√(y²+1) − arcsinh y ]
ε   (ν, M, g, Λ) = ε_med  (ν, M, g)  −  (g M⁴/16π²)[ y(2y²+1)√(y²+1) − arcsinh y ]
```

The medium terms are the x-terms of Eqs. (12)-(13) and are **not** cut off —
Paper 1 applies the cut-off to the vacuum subtraction only, which matters
because ν_q exceeds Λ above n_b ≈ 3 fm⁻³. Do not introduce a cut on the medium
integral.

This additive split is the hinge of the whole finite-temperature extension
(§3), and the reference tables already display it: their `nsq` column is the
medium part and their `Sigmaq` column is medium + vacuum + cluster.

### 1.4 Beta equilibrium

```
µ_i = B_i µ_b − q_i µ_e     (23)        Σ_i q_i n_i = 0     (24)
```

Two points that cost time if missed, both established in
`REFERENCE_TABLES.md`:

* Eq. (23) holds only for species that are *present*. A species below threshold
  has no equilibrium potential and the tables print a stale value for it.
* µ_b in the tables is the `munr` column, not `mun`. Once baryons dissolve,
  µ_b = µ_u + 2µ_d while `mun` is the vanishing neutron's own potential.

### 1.5 First-order transitions at T = 0

Paper 1 restricts itself to Maxwell construction: at fixed µ_b, maximize P.
Three transitions can each be first order or continuous depending on (f_q, B):

* **quarkyonic** — quasi-free quarks appear alongside baryons;
* **chiral** — nbar^s_q → 0 and M_q → m_q0;
* **deconfinement** — baryons become unbound and dissolve (Mott transition).

The reference tables hand you four verified coexistence windows to test any
construction against (`REFERENCE_TABLES.md` §5). Paper 1 §III also notes that
Maxwell is the wrong construction when the surface tension is small — that is
what Paper 2 and, later, `eos/mixed/` address.

## 2. Milestones for the T = 0 reproduction

Each has a gate. The gate must be green before the next milestone starts, and
gate tolerances come from `REFERENCE_TABLES.md`, not from taste.

**E0 — unbreak and audit.**
Fix the `ENJLEoSPoint` import in `eos_beta.py`. Run
`python docs/enjl/verify_reference_tables.py` and read its output.
*Gate:* `python -m pytest test/enjl -q` still green (10 passed); the audit runs
clean; you can state in one sentence each what `Sigmaq`, `nsq`, `munr` and
`mue` mean.

**E1 — reference loader in the test suite.**
`test/enjl/reference/` already provides `load_reference`, `solved_rows`,
`baryon_potential`, `electron_potential`, `present`. Write
`test/enjl/test_enjl_reference.py` that walks all five files and checks the
model's *own consistency* using table columns as input — the gap equation, the
scalar-density identities, Eq. (19), Eq. (23) — i.e. the identities the audit
script checks, but through `eos/enjl/` functions rather than reimplemented
ones.
*Gate:* the **per-file** identity table of `REFERENCE_TABLES.md` §4, reached
through `eos.enjl.thermodynamics` and `eos.enjl.uniform` calls rather than
reimplemented arithmetic. Parametrize the test over the five files and carry a
per-file tolerance dict; a single global tolerance is wrong, because the files
differ by up to five orders of magnitude on the same identity. Apply the row
exclusions of §4b-bis and §6 through `solved_rows()` and the stale-`nB`
detector. This is the milestone that proves the T = 0 kernel is right,
independently of any solver.

**E2 — fixed-composition solve against the tables.**
For each table row, feed `solve_point` the row's own densities and compare
outputs. This tests `uniform.py` end to end without any root-finding on
composition.
*Gate:* µ_i to **0.05 MeV** on `Beta_fq1.0_B0.dat`, `Beta_fq1.0_B1.dat` and
`Beta_fq0.7_B1.dat`; M_q to 2.5×10⁻⁵ relative; P to 5.2×10⁻⁴ relative.
`Beta_fq0.5_B1.dat` needs 0.25 MeV on µ_i unless restricted to n_b < 1 fm⁻³
(where 0.001 MeV holds), and `Beta_fq0.7_B0.dat` needs the nine bad rows of
REFERENCE_TABLES.md §4b-bis excluded — **use the programmatic stale-`nB`
detector given there, not a hard-coded density window, and name the reason in a
comment.** Reach for the per-file table in §4 rather than one global number:
agreement varies by five orders of magnitude between files, and a single
tolerance is either vacuous or wrong.

**E3 — beta-equilibrium solver.**
`solve_beta_point` / `beta_eos_table` must reproduce the composition, not just
consume it. Continuation in density; the existing `least_squares` formulation
with the ten unknowns (M_u, M_d, M_s, µ_b, µ_e, n_b^Q, g_w, g_r, Σ^R_b, Σ^R_q)
is a reasonable starting point but has never been run.
*Gate:* densities of every present species to 1% and µ_b to 0.5 MeV against
`Beta_fq1.0_B0.dat` and `Beta_fq0.7_B1.dat` over the full grid; onset densities
of d, s and Λ to within one grid step. Report — do not silently pass — any
density where the solve lands on a different branch than the table.

**E4 — Maxwell construction and the transition windows.**
Locate first-order transitions by maximizing P at fixed µ_b.
*Gate:* the four coexistence windows of `REFERENCE_TABLES.md` §5 reproduced:
P to 0.1%, both edge densities to 1%. Remember that µ_b in the quark phase is
µ_u + 2µ_d.

**E5 — paper-level physics checks.**
The existing `test_enjl_p1.py` covers vacuum masses, Table II, U_Λ(n₀) and the
factor 9. Add the remaining published numbers: onset densities n_onset(Λ) = 0.72
(SNM) and 0.44 fm⁻³ (PNM); the quarkyonic-onset statement that d quarks appear
at n_b = n₀/2 for f_q = 0.5, B = 0 and at 0.17 fm⁻³ for B = 1 GeV/fm³;
Λ-onset 0.51 and 0.64 fm⁻³ for B = 0 and 1 at f_q = 0.5.
*Gate:* each to the precision the paper quotes, no tighter.

**E6 — TOV.**
Feed E3/E4 tables to `eos/tov/`. Paper 1 Fig. 8 gives M-R curves for all six
(f_q, B) combinations; the qualitative statements are that (0.7, 1), (1, 0) and
(1, 1) satisfy the astrophysical constraints while (0.5, 0) gives radii that are
too small.
*Gate:* M_max and R(1.4 M☉) consistent with Fig. 8 to the readability of the
figure (~0.05 M☉, ~0.3 km). Note that Paper 1 stitches a DD-LZ1 crust below the
core-crust transition by Maxwell construction — a crustless star is ~0.9 km
smaller at 1.4 M☉, which is the same trap already documented for the DD2 TOV
test.

## 3. Extending to finite temperature

This is the part where `eos/general/` does the work, and where the T = 0 code
must not be duplicated.

### 3.1 The whole extension is one function

Every T-dependence in the model enters through the single-species kinetic block
`(n, P, ε, s, n^s)`. Nothing else in §1 changes form: the couplings are
functions of n_b, the gap equation is the same equation with T-dependent scalar
densities, the rearrangement terms have the same expressions, E₀ is the same
number. So the extension is:

> **Add a `T` argument to one function in `eos/enjl/thermodynamics.py` and
> route it to `eos/general/fermi_integrals.py` for the medium part, keeping the
> vacuum term analytic.**

The template already exists and should be followed rather than reinvented:
`eos/dd2/physics/thermo.py::kinetic_thermo` is exactly this pattern for the
nucleonic sector — T = 0 closed forms in one branch, `solve_fermi_jel` with
hc3 conversion in the other. The ENJL version differs only by the additive
vacuum term.

```python
def kinetic_thermo(nu, m, g, T=0.0, Lambda=0.0, antiparticles=False):
    """(n, P, eps, s, ns) in natural units: MeV^3, MeV^4, MeV^4, MeV^3, MeV^3.

    The Lambda cut-off contributes a T-INDEPENDENT vacuum term (spec 1.3),
    added analytically outside the finite-T medium integral.
    """
    if T == 0.0:
        ...  # existing exact branch, unchanged
    from eos.general.fermi_integrals import solve_fermi_jel
    n, P, eps, s, ns = solve_fermi_jel(nu, T, m, g,
                                       include_antiparticles=antiparticles)
    n, P, eps, s, ns = n * hc3, P * hc3, eps * hc3, s * hc3, ns * hc3
    if Lambda > 0.0:
        eps += _eps_vacuum(m, g, Lambda)     # negative
        ns  += _ns_vacuum(m, g, Lambda)      # negative
    return n, P, eps, s, ns
```

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
`DD2_OPEN_QUESTIONS.md` with the choice made and the reasoning, and if the
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
