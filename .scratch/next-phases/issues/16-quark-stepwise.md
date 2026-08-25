# notebooks/quark_eos — the NJL and CCDM step-by-step section

Type: task
Status: resolved
Blocked by: 15
Parent: ../map.md

## Question

Stage 2, second half, for `njl` and `ccdm` specifically. One cell per step, each
**printing the quantity it just computed**:

1. the model **without colour superconductivity**: parameters, the gap/field
   equations, the grand potential, and the thermodynamic quantities at one
   `(n_B, T)` point — each labelled with its symbol and units
2. the same point with pairing on, one pairing pattern at a time
3. **unpaired vs 2SC vs CFL** compared: grand potential per phase at fixed
   `(mu_B, T)`, which phase is favoured where, `P` and `eps` vs `n_B` per phase
4. `Delta(n_B, T)` as a 2-D map per pairing pattern, plus `Delta` vs `n_B` at
   fixed `T` and vs `T` at fixed `n_B`
5. the quantities that go with it: quark and electron fractions, `c_s²`, the
   phase boundary in the `(mu_B, T)` plane

`docs/njl_csc_implementation.md` and `docs/ccdm_implementation.md` are the
references for what these models implement. **Where the notebook and those
documents disagree, the code decides and the disagreement is reported** in the
answer.

## Answer

**Shipped: section 10 of [notebooks/quark_eos.py](../../../notebooks/quark_eos.py)**,
commit `2d9d7e8`. Five steps, one cell per step per model where the physics is
model-specific, every cell printing the quantity it just computed with its
symbol and its units. **No library file was touched**, and nothing below reads a
solver internal that is not exported at package level.

Verified by `jupytext --to notebook --execute` in an isolated `git archive HEAD`
copy of the **committed** tree at `2d9d7e8`: **86 cells, 43 code cells, 0 error
outputs**, ~14 minutes end to end, of which section 10 is about five. Interpreter
**python.org 3.14.2**, numpy 2.3.5, scipy 1.17.0, matplotlib 3.10.9. Targeted
tests in a second archive copy plus a snapshot of the gitignored `test/`:
`test/general test/test_imports.py test/vmit test/alphabag test/njl test/ccdm
test/abpr` = **811 collected, 811 passed, 0 failed** in 60 s. (Tickets 15 and 17
reported 809 in the same set; the +2 arrived from another session's work between
then and now, and **0 failures are added by this ticket**.) The full suite was
not run.

Five new figures in `output/quark/` (`.png` + `.pdf`): `csc_phases_vs_nB`,
`csc_gap_maps`, `csc_gap_cuts`, `csc_composition_cs2`, `csc_phase_boundary`.
All styling from `eos/general/figure_style.py`; the two colormaps the section
needs are BUILT OUT OF its palette (a white-to-`OKAB['blue']` ramp for the gap
maps, a `ListedColormap` of the pattern colours for the boundary) rather than
picked from matplotlib's, so the colour decision stays in the one module that
owns it.

### The five steps

1. **Unpaired** (`csc=False`), one cell per model, at n_B = 1.6 fm^-3, T = 0.
   For `njl`: the parameters, the three gap equations
   `M_f = m_f - 4 G_S phi_f + 2 K phi_g phi_h` with the condensates, the
   constituent masses and `max|R_mass| = 5.8e-13` MeV printed so the equations
   are seen to be satisfied rather than asserted; then `Omega`, `P`, `eps`, `s`,
   `f`, every potential, every density and fraction, the derived
   `B_eff^(1/4) = 228.93` MeV and the Euler residual (`2.4e-16`). For `ccdm`:
   the same shape over its four field equations, the dilaton, the dielectric
   `chi`, the two condensates, `omega_0`, `Sigma_R`, the dressed masses, the
   glue and chiral potentials `U`, `V`, and its own `B_eff^(1/4) = 239.66` MeV.
2. **Pairing on, one pattern at a time**, `patterns=('2SC',)` and so on — a
   restriction on the enumeration, which the cell says and which is how a branch
   that is not the ground state gets drawn at all. Prints the three gaps, `mu_3`
   and `mu_8` (in the `T_8 = diag(1,1,-2)/3` normalisation, with the conversion
   pointed at), `gapless`, the pairing correction `delta_Omega` and the
   condensation cost separately, and the Euler residual, which holds to machine
   precision paired as well as unpaired.
3. **The three compared, in both holdings, because they answer different
   questions.** At fixed `(mu_B, T)` by `Omega`, with **electric neutrality
   imposed per pattern** (`mu_e = -mu_C`, `mu_C` from `n_C = n_e`, `mu_S = 0`);
   at fixed `n_B` by `f = eps - T s`, which is what the enumeration itself
   ranks. Plus `P` and `eps` vs `n_B` per phase.
4. **`Delta(n_B, T)`** as a 2-D map per pattern per model over 8 densities x 4
   temperatures, printed as a table and drawn on one shared colour scale, plus
   the two cuts: `Delta` vs `n_B` at T = 0 and `Delta` vs `T` at n_B = 1.6.
5. **What goes with it**: the four fractions per phase, `c_s^2` per phase from
   `eos_response(frozen='equilibrium', patterns=(p,))` — taken *within* a branch,
   since across the enumeration it would be a chord over a first-order jump —
   and the favoured pattern over the `(mu_B, T)` plane, read off the same
   neutrality map as step 3.

### What the section establishes

- **In neutral matter CFL wins wherever a gap survives**, in both models, at
  fixed `mu_B` and at fixed `n_B` alike. At n_B = 1.6 fm^-3: `njl`
  `f - f_unpaired` = -24.3 (2SC) and **-46.5** (CFL) MeV/fm^3; `ccdm` -47.2 and
  **-92.4**.
- **`P` does not rank the phases at fixed density, and the printed table shows
  it failing to.** `njl`'s 2SC branch carries the higher pressure of the two
  paired ones (373.5 against 371.3 MeV/fm^3) while CFL carries the lower free
  energy and is the ground state. The branches sit at different `mu_B` at the
  same `n_B`, so the two orderings are allowed to differ — ranking by pressure
  is the fixed-`mu_B` statement, not the fixed-`n_B` one. This is the one place
  a reader could most easily draw the wrong conclusion, and the notebook says so
  where the numbers are.
- **Pairing stiffens both models at fixed density**: higher `P`, lower `eps`,
  and `c_s^2` ordered unpaired < 2SC < CFL at every density measured
  (`njl` at 1.8 fm^-3: 0.270, 0.316, 0.371; `ccdm`: 0.325, 0.397, 0.497).
- **CFL is neutral without electrons**, and the section meets that fact from
  both sides: `Y_u = Y_d = Y_s = 1.00000` exactly with `Y_e ~ 1e-16`, and the
  neutrality solve of step 3 finding its root at `mu_C = 0` where no bracket
  contains a sign change. That case is checked first and named as physics, not
  patched as a solver inconvenience.
- **The gaps melt where BCS says they should.** A printed check against
  `T_c ~ 0.567 Delta(0)`, which is not fitted here: `njl` 2SC/CFL predict
  T_c = 54.1/44.6 MeV and the map is gapped at T = 40 and bare at 60; `ccdm`
  predicts 81.1/67.6 and the map is still gapped at 60, where the grid ends.
  Four for four.
- **The paired region has two edges and they are not independent**: bounded
  above in `mu_B`, where the Fermi surface runs out towards the cutoff, and
  above in `T`, where it melts — and the high-`mu_B` edge moves DOWN as `T`
  rises. `njl` loses its condensate entirely by T = 60 MeV; `ccdm`, whose gaps
  are half again as large, still has one.

### The seeding fix, which is physics and not tuning

The first working draft printed a **gapless band at T = 40 MeV in `ccdm`'s 2SC
map**, between two gapped rows, and an `njl` CFL row at T = 40 identical to its
own 2SC row to five digits. Neither is real. The gap equation has three roots at
any Fermi-surface mismatch — zero, a barrier maximum, and the physical BCS root
— so a cold-started Newton solve returns whichever its seed was nearest,
silently, and a density-warm-started row then carries that root across the whole
line.

The sweep therefore continues in **both** directions: each point is seeded from
the same density one temperature down where that exists, and from its density
neighbour otherwise. With it, all four maps are monotone in `T`, `njl`'s CFL row
separates from its 2SC row, `ccdm`'s T = 40 row reads 125-134 MeV, and the
`njl` 2SC line is also 8/8 rather than 6/8 and 5.7 s rather than 13. The
notebook states the hazard where the sweep is defined, since a reader copying
the sweep is exactly who needs to know.

### Where the notebook and the documents disagree — the code decides

**A number-for-number check is IN the notebook, not only in this answer**
(section 10.3c), because `docs/njl_csc_implementation.md` section 6 prints a
neutral solve at `mu_B = 1500` MeV, T = 0, `eta_D = 0.75`, no vector coupling —
which is exactly the shipped default set — for the unpaired and 2SC patterns.
Executed output:

| pattern | quantity | document | code | difference |
|---|---|---:|---:|---:|
| unpaired | M_u, M_d, M_s | 9.84, 8.55, 265.59 | 9.8436, 8.5535, 265.5848 | < 0.006 |
| unpaired | mu_C, n_B, P | -34.20, 1.4319, 302.12 | -34.2013, 1.4319, 302.1166 | < 0.004 |
| 2SC | M_s, mu_C, mu_8 | 243.13, -62.27, -2.46 | 243.1559, -62.2475, -2.4623 | < 0.03 |
| 2SC | Delta_3, n_B, P | 95.50, 1.4887, 324.75 | 95.4996, 1.4888, 324.7480 | < 0.002 |
| **2SC** | **M_u** | **11.96** | **9.7239** | **-2.24** |
| **2SC** | **M_d** | **7.65** | **8.8977** | **+1.25** |

1. **The 2SC light constituent masses are the one disagreement, and everything
   downstream of them agrees.** Every other entry in both rows matches the
   document's printed precision, including `Delta_3` to four decimals and `P` to
   0.002 MeV/fm^3 — which could not happen if the masses really differed by
   2.2 MeV. The document has the u-d splitting *growing* from 1.29 MeV
   (unpaired) to 4.31 MeV in 2SC; the code has it *shrinking* to 0.83 MeV, which
   is the direction pairing u with d implies. The code decides, and the check
   stays in the notebook so the next reader sees it rather than rediscovers it.
2. **`docs/njl_csc_implementation.md` section 6.3 and section 11 both flag the
   CFL neutral solve as "not tightly converged (residual 13)", needing a
   Ginzburg-Landau-informed seed. It is converged now.** At `mu_B = 1500`,
   T = 0 the code returns residual `4e-16`, Euler `2e-16`, `mu_C = 0` exactly
   and `Delta = (76.6, 76.6, ...)` MeV; over the section's whole `(mu_B, T)`
   grid CFL converges in 41 of 42 cells per model and is the favoured phase in
   most of them. The document's caveat is stale, and section 10 leans on the
   converged answer.
3. **`docs/ccdm_implementation.md` section 6.5: "Do not include gapless states
   in the minimization... flag *enumeration-invalid*... rather than silently
   comparing incomparable states." Both models rank them.** `njl/solver.py:601`
   and `ccdm/solver.py:678` are `min(converged, key=f_total)` over every
   converged candidate, gapless or not; `gapless` is carried on the point and
   reported, never used to exclude. This is live at the section's own probe
   point: `njl`'s 2SC state at n_B = 1.6 fm^-3 **is gapless**, and so is the
   2SC row of the document's own section-6 table. Section 10.2 prints the flag
   for every pattern and says what it means for the comparison; the behaviour is
   reported, not changed.
4. **The lepton bookkeeping differs from `njl` section 6.1's assembly.** The
   document writes `Omega = ... - P_lep + ...` and `eps = ... + eps_lep`, i.e.
   leptons inside the phase potential. The code keeps them out —
   `state_at`'s "the lepton sector is NOT in either: a phase does not own the
   leptons" — and adds them in `solver.py`. Same totals, different boundary, and
   the code's is the one the phase-adapter contract needs. Section 10.1 prints
   `Omega` and `P (matter)` beside the full `P` so the difference is visible
   rather than implied.
5. **The enumerated pattern list differs.** `ccdm` section 6.5 says to solve
   "unpaired, 2SC, uSC, dSC, CFL"; `DEFAULT_PATTERNS` is
   `("unpaired", "2SC", "CFL", "free")` — uSC and dSC are declared in
   `general/pairing.PATTERNS` but reached through the asymmetric `free` seed
   instead of enumerated by name. Section 10 works the three the ticket names
   and states the table of what each pattern makes free.
6. **Agreements worth recording**, since a check that passes is also a result:
   `ccdm`'s parameter table (section 8, row 7) asks for a `G_D` giving
   `Delta ~ 20-150` MeV at `mu_q ~ 450` MeV, and the shipped `G_D = 5e-6`
   MeV^-2 gives 102-146 MeV over the section's grid; the `njl` unpaired row
   above matches its document to five digits; and `delta_Omega` is exactly 0 in
   every `unpaired` row printed, which is section 5.4's requirement that the
   correction form be used rather than a difference of two large potentials.

### Choices made, and stated in the notebook where a reader would ask

- **The fixed-`(mu_B, T)` comparison imposes neutrality.** Without it the
  question is decided by a charge the phase is not paying for — CFL is neutral
  with no electrons, so at `mu_C = 0` it is a different system rather than a
  different phase. This is the one place the section leaves
  `eos_point`/`eos_table`/`eos_response`: it uses `thermo_from_mu`, the
  phase-adapter surface both models export at package level and the one
  `eos/mixed` consumes, which is the right entry point for a fixed-potential
  question.
- **The neutrality root is located to 0.5 MeV, and that is not a loosened
  tolerance.** `dOmega/dmu_C = -n_C + n_e = 0` at neutrality, so an error in
  `mu_C` costs `Omega` only at second order. Measured: tightening to 1e-3 MeV
  changes no printed digit of the table and costs 1.7x the time (422 s against
  243 s for the map).
- **`ccdm`'s chiral branch is declared `'restored'` in the fixed-mu map rather
  than enumerated**, because measured on this grid the branch enumeration has
  nothing to decide: above `mu_B ~ 1550` MeV the `restored` and `partial` seeds
  converge to the same root to every printed digit, and below it neither
  converges. Stated in the notebook with the reason.
- **A tie between candidates whose gaps have closed is reported as `unpaired`.**
  Above the gap's endpoint all three restricted solves return the same state and
  their potentials agree to every digit, so `min` was picking a name by float
  noise — the boundary map flipped a cell between `CFL` and `unpaired` between
  two runs. It now reports the name that says there is no condensate.
- **`Delta` is reported as `max_eta |Delta_eta|`.** The gap equation is odd in
  `Delta_eta`, so the sign is a phase convention and `thermo_from_mu` returns
  either one; a magnitude is the only thing that compares across patterns.
- **`cs2` is read with the existing two-key reader** (`CS2_KEYS` from section
  6.3) and renamed nowhere — ticket 69 is open and is not this ticket's. Both
  models return `cs2_isothermal`, and the panel is labelled with the key that
  was actually returned.
- **Grids.** n_B in [1.0, 2.4] fm^-3 and T in {0, 20, 40, 60} MeV, well above
  the knobs cell's grid, because these gaps do not open at all below n_B ~ 0.7
  fm^-3; `mu_B` in [1450, 2050] MeV for the same reason. The probe point is
  taken FROM the grid (`CSC_N_B_GRID[3]`) rather than written again, so no
  float-keyed lookup can miss silently.

### Reported, not fixed

- **`njl`'s `eos_response` fails at n_B = 2.2 fm^-3 for the `unpaired` and
  `2SC` patterns** — "the response stencil needs a converged neighbour and the
  solve at n_B = 2.2022 fm^-3 did not converge (residual 3.1e-01)". CFL solves
  there. Non-convergence is a return value and the notebook prints it as one.
- **`ccdm`'s `eos_response` sets `branch_changed=True` at n_B = 2.2 fm^-3** for
  the unpaired and 2SC patterns even with the pattern restricted, i.e. the
  density stencil straddled a *branch* change. The notebook prints the flag
  beside the number, which is what the field exists for; the numbers so flagged
  are chords, not tangents.
- **`njl` CFL does not converge at n_B = 1.0 fm^-3, T = 0** (residual 2.6e-02),
  nor does `ccdm` CFL, whose message names the reason as physics: below the
  deconfinement onset there is no deconfined root at fixed density. One cell of
  each CFL map is therefore grey, which the map's colormap distinguishes from
  white — white is a gap that has closed, grey is a point the solver could not
  reach, and printing them the same colour would merge two different statements.
- **Ticket 68 stands unchanged**: `njl` and `ccdm` still take `leptons` through
  the condition bag rather than as a named argument. Section 10 never passes
  `leptons` — beta equilibrium is the only mode it uses — so it does not touch
  the gap either way.

### Scope

`notebooks/quark_eos.py` and its paired `.ipynb` (committed without stored
outputs, `docs/strip_notebook_outputs.py`), plus this file and the map. Two
one-line cross-references in the existing text were updated to point at section
10 instead of at "a separate ticket"; the knobs cell and the three-way reporting
pattern are untouched, as the ticket required. `notebooks/enjl_eos.*` and
`notebooks/hybrid_eos.*` were not touched.

Status: resolved.
