# DEFERRED — known gaps, per model

The tracked ledger CLAUDE.md refers to: modes a model does not support, physics
not yet wired, and behaviour that is understood but not yet fixed. A gap
recorded here is a decision; a gap not recorded here is a bug.

Each entry says what the gap is, how it shows up, and what closing it would
take. Entries are removed when closed, not marked "done".

---

## Cross-cutting

### A potential is only pinned as tightly as its conjugate density responds

**Models:** sfho and vmit (observed), and any model exposing `fixed_YC_YS` or
a charge-neutral mode at Y_C = 0.

Two cases, one cause. A chemical potential is fixed by the residual only
through the density it conjugates, so where that density is zero the solver
has little or nothing to go on.

In `fixed_YC_YS` with Y_S = 0 — symmetric nuclear matter, the heavy-ion slice —
no strange species is thermally populated at the densities and temperatures
tested (Lambda ~ 1e-16 fm^-3, Xi ~ 1e-32 fm^-3, n_S = 0 exactly). The
strangeness constraint n_S = n_B Y_S is then satisfied for a whole range of
mu_S: the residual has no gradient in that direction and the Jacobian is
singular there. The solver converges — every other quantity is determined and
reproducible to the last digit — but reports whichever mu_S its path happened
to reach. Recompiling the Numba integral kernels moves it by ~10 MeV.

Seen at n_B = 0.16, 0.32, 0.64 fm^-3, T = 10 MeV, SFHo-Y (Fortin) with the
full baryon octet. `eps`, `P`, `mu_B` and every density are unaffected.

**mu_e where no electrons are present** — a charge-neutral phase at Y_C = 0.
Here mu_e = 0 IS the answer, so it is determined rather than free, but only
weakly: dn_e/dmu_e is of order T^2/(hbar c)^3, about 4e-6 fm^-3 MeV^-1 at
T = 10 MeV, so a residual gate at 1e-10 on the density leaves mu_e loose at
the 1e-5 MeV level. Its landing point is round-off. Seen in vmit, dd2, sfho,
zl and alphabag, in every `fixed_YC` slice at Y_C = 0.

Closing either means deciding what the API should say when a conserved charge
is carried by no populated species. Options: report the potential as undefined
(NaN) with a status flag; pin it by convention (mu = 0) and document that; or
have the mode raise. Until then `test/baseline` does not freeze mu_S where n_S
is zero, or mu_e where n_e is zero, because there is nothing there to freeze.

### The pure-Python integral fallback is not a bit-exact reference

`general/fermi_integrals.py` and `general/bose_integrals.py` define their
kernels under `@njit(fastmath=True, cache=True)`, with a pure-Python fallback
used when Numba is absent (or `NUMBA_DISABLE_JIT=1`). The two paths agree to
about 1e-7 relative, not to machine precision, because `fastmath` lets the
compiler reassociate floating-point operations.

That is expected rather than wrong, but it means the fallback is not the
"reference flavour" in the sense of CLAUDE.md §9 — it is a second
implementation with its own error, and no parity test currently pins the two
together. Worth deciding during the `general/` refactor whether to add a
parity check at a documented tolerance, or to drop `fastmath` on the kernels
where the speed gain does not justify it.

### ASY-EOS band columns are headed "low"/"up" but are stiff/soft edges

`plot/data/samples/ASYEOS_2016_Esym.txt` and the CSV derived from it carry the
header `rho_fm3 Esym_low_MeV Esym_up_MeV`, but the two curves cross at
saturation: below n_0 the "low" column is the larger of the two, above it the
smaller. That is the physics — the constraint bounds the SLOPE of E_sym, so
the band is pinned where E_sym is already known and fans out either side — but
the column names invite exactly the wrong fix, which is to sort them.

Drawing is unaffected (`fill_between` fills between two curves in any order)
and both the crossing and the pivot are pinned by tests. What is left is to
rename the columns to something like `Esym_stiff` / `Esym_soft` in SOURCES.md
and the converter, so the file stops implying an ordering it does not have.

### The CompOSE reader creates a cycle, and moving it is not just a move

`eos/sfho/compose_loader.py` imports `EOSTable_for_TOV` from `eos.tov.solver`,
and `eos.tov.solver` imports `SFHOComposeLookup` back from it — a genuine
import cycle, currently worked around with a lazy import inside the function
and a comment saying so. On top of that, `eos/dd2/verify/compose.py` imports
the same module, so a model depends on another model.

Both are violations of the layering in CLAUDE.md section 1, and the plan's
remedy — move it to `general/compose.py` — cannot be applied literally,
because `general/` may import nothing else in the repository and the reader
currently returns an `EOSTable_for_TOV`.

The shape the move has to take: `general/compose.py` reads a CompOSE table and
returns plain arrays (P, eps, n_B); the crust-table wrapper `to_crust_table`
moves to `astro/tov`, which is the layer entitled to know about
`EOSTable_for_TOV`. That resolves the cycle in the right direction — general
produces data, astro consumes it — and drops the `dd2 -> sfho` edge. It should
be done in the `astro/tov` session, where the crust path has test cover: a
silent fall back to no crust shifts M_max by about 1%.

### One notebook still sets rcParams

`notebooks/ENJL_usage.py` sets `figure.dpi` and `figure.figsize` directly. It
is a per-notebook display preference rather than house style, and the file is
jupytext-paired, so changing it means changing the .ipynb in the same edit.
Left for the notebook rework, which rewrites both halves anyway. Every module
in `eos/` and in `nucleation/` now goes through
`eos.general.figure_style`.

---

### Only dd2 returns the shared state records; six models still return their own

`eos/general/state.py` holds the records every model is meant to hand back —
`PhaseThermo` (matter only: the model's own fields under its own names, EVERY
active species' density, mu_i, mu_eff_i and m_eff_i per species, the conserved
charges, P, eps, s), `LeptonThermo` (all four potentials explicit, so a
transparent muon family is a visible assumption rather than a hidden
`mu_mu = -mu_C`), and `EoSPoint` (what a mode returns). `eos/general/modes.py`
holds `ModeSpec`, the mode as one choice per conserved charge. dd2's
thermodynamics returns `PhaseThermo` and its solver takes a `ModeSpec`. The
others do not, and `eos/mixed` still carries a second `PhaseThermo` of its own.

What each model has to supply is decided by ONE question — what its internal
self-consistent solution is — because the rest of the state is (chemical
potentials, T) in every model:

    dd2, sfho          meson mean fields sigma, omega0, rho0, (phi0);
                       n_i derived
    enjl               constituent masses M_u, M_d, M_s from the gap
                       equation; n_i derived
    vmit               the vector field V = a hbar c sum_q n_q
    zl                 the interaction potentials mu_Hv(n_p, n_n)
    alphabag, abpr     none -- everything is explicit in mu

vmit and zl are the two that look different and are not: both carry densities
in the *unknown vector* because that keeps the residual polynomial rather than
nesting Fermi integrals inside it. That is a conditioning choice of the
solver, not a statement about the state — physically vmit's state is still
(mu_q, V, T). So `fields` holds whatever the model solves for, under the
model's own names, and is empty where there is nothing to solve.

Three things the records fix that the remaining models get wrong in the same
way, so they are worth doing together rather than one model at a time. dd2 is
converted on all three and is the worked example:

- **nucleons are privileged over hyperons.** Every model's kinetics computes
  mu_eff and m* for each active species and then keeps only n and p. Neither
  is recoverable from the record afterwards (they need the fields and the
  per-species coupling ratios), so g-modes and the response functions recompute
  them. `m_eff` is even singular although m*_i differs per species — in DD2Y
  at n_B = 0.6 the neutron sits at 192 MeV while Lambda is at 652 and Xi- at
  1083, so the single number was the neutron's and the rest were discarded.
- **the mode is welded into the state.** dd2 carried eight mode fields on its
  context and branched on strings inside the residual; the other models do the
  same with their own vocabularies, which is why no two of them accept a mode
  the same way.
- **the non-leptonic charge is called `mu_Q` in some models and `mu_C` in
  others.** §2 says C. dd2 is converted; the rest are not. Storing mu_C rather
  than recovering it as mu_p - mu_n is also the numerically better choice, and
  measurably so: those are two ~1300 MeV numbers differing by ~100 MeV, so the
  subtraction costs about two digits, and removing it from dd2's warm start
  took its backend parity from 3.0e-14 to 8.9e-16.

Each model's conversion lands in its own commit with its baseline re-run,
since the records change what `eos_point` returns.

---

### The module names are standardised, and most models have not been renamed

CLAUDE.md §5 fixes one name per role — `parameters.py`, `species.py`,
`thermodynamics.py`, `solver.py`, `table.py`, `api.py`, `verify/` — with the
names mandatory and their existence conditional. Two renames are done
(`dd2/parametrization.py` → `parameters.py`, `vmit/thermodynamics_quarks.py` →
`thermodynamics.py`). The rest are outstanding, each belonging in its model's
own session where the baseline and `test_imports.py` are already being run:

    dd2       delete notebook_api.py -- the last one outstanding. The rest of
              dd2's layout is now CLAUDE.md §5: physics/{thermo, fields,
              mesons} and the non-residual half of physics/octet became
              thermodynamics.py; octet_residual, assemble_octet and
              physics/residual.py became solver.py; physics/ became backends/
              {jacobian, kernel_numba, responses_jac}; coefficients.py became
              responses.py (kept apart from responses_jac.py because §9 makes
              them the reference and fast flavors of one thing, and backends/
              has to stay deletable); nmp.py + nmp_inverter.py merged; xp.py
              deleted.
    sfho      eos.py -> solver.py, compute_tables.py -> table.py,
              nuclear_saturation_properties.py -> nmp.py,
              thermodynamics_hadrons.py -> thermodynamics.py
    vmit      eos.py -> solver.py
    zl        eos.py -> solver.py, compute_tables.py -> table.py,
              thermodynamics_nucleons.py -> thermodynamics.py
    alphabag  eos.py -> solver.py, compute_tables.py -> table.py,
              thermodynamics_quarks.py -> thermodynamics.py
    abpr      eos.py -> solver.py
    enjl      eos_beta.py + uniform.py -> solver.py (a merge, not a rename)

Function names go with them, per the §13 vocabulary: no name repeats its
package (`compute_zl_thermo_from_mu` -> `thermo_from_mu`), and the same job
carries the same name everywhere (`get_<model>_default` -> `Parameters.default()`,
`get_default_guess_*` -> `default_guess(mode, ...)`, `result_to_guess` /
`*_warm_start` -> `warm_start(point)`, `compute_<model>_table` ->
`build_table(spec)`). These are public names, so each model's renames land in
its own commit with every call site fixed alongside — no aliases, since two
names for one thing is what the rule removes.

`dd2/parameters.py` carried three function-level imports, two of them
commented "local import breaks the cycle", because three constructors reach up
to `solver.py` from the bottom layer. `from_nmp` is now a free function in
`nmp.py` and its import is gone. Two remain: `from_hyperon_potentials` and
`from_delta_potential`, which solve symmetric matter at saturation to invert a
single-particle potential U_Y or U_Delta into a scalar coupling ratio.

They are the same kind of object as `from_nmp` — an inverse map from a
physical observable to a coupling, which is why they sit above `solver.py` —
so the natural home is `nmp.py`, with that module documented as the maps
between couplings and the quantities they are fitted to rather than as
nuclear-matter parameters alone. What is not yet decided is whether the
module keeps the name `nmp.py` once it holds potential inversions too. About
ninety call sites use these as classmethods, so the move belongs with the
`Parametrization` -> `Parameters` rename, which touches the same lines.

`vmit/eos.py` is held back only because `notebooks/DD2vMIT_general1oPT`
imports from it directly and that notebook has uncommitted work in it; the
rename and the notebook's import line have to move together.

`vmit/compute_tables.py` is the one deliberate exception to the scheme: it is
the first-generation settings-object interface, kept because the ZLvMIT
notebook drives vMIT through it, and it now sits beside `table.py` as a shim
over the shared driver rather than being renamed to it.

`eos/mixed` is a composite engine and takes the shorter list of CLAUDE.md §5 —
`adapters.py`, `api.py`, `verify/`, `mixed.tex` — all of which it now has. The
one name still out of step is internal: `ChargeSpec`, `MODE_FRACTIONS` and the
table axes call the electron lepton fraction `Y_L` where the spec calls it
`Y_Le`; the public boundary translates and refuses the internal spelling, but
the rename itself is outstanding.

---

### `docs/STRUCTURE.md` does not exist yet

CLAUDE.md §10 sends a new figure to the `figure_style` module docstring **and**
to a worked figure example in `docs/STRUCTURE.md`, and §11 lists the file as
part of the layout. Only the module docstring is written. The document belongs
with the notebook rework in Phase 5, which is where the worked example comes
from.

---

### The README's directory tree predates the refactor

`README.md` still lists `sfhoalphabag/`, `general/plotting_info.py` and a
one-file `tov/`, all of which are gone, and carries no `dd2/`, `mixed/` or
`enjl/` at all. It is a rewrite rather than a patch, and Phase 5 does it
alongside the notebook rework; individual lines are corrected only where a
rename in this phase would otherwise leave them newly wrong.

---

### The freeze selector of `eos_response` is a fixed menu, not a selection

CLAUDE.md §5 requires the freeze to be selectable case by case. Today each
model takes a single string from a short hard-coded list — dd2
`('equilibrium', 'composition')`, vmit `('equilibrium',)`, mixed
`('equilibrium', 'chi')` — so a combination nobody anticipated cannot be
asked for. The target is a *set* of held quantities, with the named freezes as
presets that expand to one:

    "equilibrium"  frozenset()                    nothing held
    "fast"         {every species Y_i} | {"chi"}  no reaction has time
    "slow"         {"Y_C", "chi"}                 all chemical equilibria but beta
    "conserved"    {"Y_C", "Y_S", "chi"}          strong imposed, both weak frozen

so that `frozen={"Y_C", "Y_S"}` (chi free), `frozen={"chi"}` and
`frozen=set(species)` are all reachable. The names come from Constantinou,
Guerrini et al., arXiv:2506.20418 §IV, whose fast and slow limits are taken at
fixed (y_i, chi) and fixed (Y_e, chi) respectively — the {y_i} there are
PARTICLE fractions, and the conserved-charge description appears only in the
slow limit, where imposing every equilibrium but beta collapses them to Y_C.

A second, orthogonal axis is missing entirely: the thermal condition. Every
`eos_response` in the repository differentiates at fixed T, while the adiabatic
sound speed of the CompOSE manual and of that paper is taken at fixed entropy
per baryon. At T = 0 they agree; at T = 50 MeV they do not. Returned names
should say which — `cs2_isothermal` against `cs2_adiabatic`, never a bare
`cs2` whose meaning depends on the arguments.

---

### The response functions are three finite-difference stencils, not one derivation

C_V, C_P, Gamma and c_s^2 are all second derivatives of the same free energy
per baryon F(T, n_B, Y_C), and the CompOSE manual (arXiv:2203.03209 §3.6)
derives every one of them from d2F/dT2, d2F/dn_B dT and d2F/dn_B^2. The code
instead takes a separate central difference per quantity at a relative step of
1e-3, which is the least accurate step in the response path. Constantinou,
Guerrini et al. arXiv:2506.20418 Eq. (76)-(77) closes the loop:
(dP/dn_B)_S = (C_P/C_V) (dP/dn_B)_T, so the adiabatic sound speed follows
algebraically from the isothermal one and the heat-capacity ratio — one
stencil instead of two. Worth doing when the freeze selector is built, since
both touch the same code.

## Per model

### dd2
- `table.hadronic_row` emits a Y_C and Y_S that are BARYONS ONLY, while the
  `EoSPoint` it flattens carries the totals. It recomputes them itself:

      _, n_C, n_S = hadronic_charges(flags, p.composition_map)

  and `composition_map` holds baryons alone, so a thermal meson gas is dropped.
  Its own docstring says the row is "keyed exactly the way
  `eos.mixed.composition_row` keys a mixed point, so a pure-hadronic table and
  a hybrid table concatenate without renaming anything" — but the mixed side
  uses the totals, so the same column name carries different physics on the
  two sides, differing by 10–20 percent at T = 40 MeV with pions. This is the
  defect class already fixed once in `sound_speed_frozen_hadronic` (b83d162),
  in the sibling that was not checked then, and it contradicts CLAUDE.md §2.

  The fix is one line — `Y_C=p.Y_C, Y_S=p.Y_S` — but it CHANGES TABLE COLUMNS
  at T > 0, so it is a deliberate physics-changing fix: its own commit, the
  before/after quoted, and the affected baseline entries regenerated. It must
  not be folded into a refactor, which is why it is still here.
- Deleting `backends/` leaves every equation-of-state baseline BIT-IDENTICAL
  at rtol = 1e-10 — that is CLAUDE.md §5's property, measured — but moves the
  TOV sequences by 4.8e-07 relative and M_max by 2.5e-08 (6e-08 Msun on a
  2.4 Msun star). The jitted T = 0 kernel and the NumPy one evaluate the same
  closed form to machine precision, yet the two paths converge to roots
  differing in the last bits, and the adaptive integrator amplifies that by
  ~1e7 — the same amplification measured in the backend-parity entry under
  astro/tov. Nothing to fix in `backends/`; recorded so the next person to run
  the deletion check knows the tov baseline is expected to move and by how
  much.
- `susceptibilities` exists only in the analytic-Jacobian flavor. §9 says the
  fast flavor is validated against a reference, and for chi_ab there is none:
  the sound speeds and heat capacities have their finite-difference twins in
  `responses.py`, chi_ab does not. Two consequences. It is unvalidated except
  against its own symmetry and a hand-rolled grand-canonical difference in
  `test/dd2/test_dd2_m10_jac.py`; and `eos_response(frozen='equilibrium')`
  raises without `backends/` rather than degrading to a slower path, which is
  the one place §5's deletability is a feature gap rather than a speed cost.
  `thermo_at_potentials` now makes the reference cheap to write — perturb
  mu_B, mu_C, mu_S, re-solve, read (n_B, n_C, n_S) — but writing it is new
  physics, so it waits for the response-function session.
- `eos_response` implements the freezes `equilibrium` (beta_eq_neutrinoless
  only: c_s^2, C_V, C_P, chi_ab) and `composition` (nucleonic Y_p: adiabatic
  c_s^2 and Gamma). Not yet wired: frozen conserved fractions (Y_C, Y_S fixed
  with species re-equilibrating), the leptonic re-neutralization variants,
  the thermal index through the API, and `equilibrium` for the other modes.
  All raise naming this file.
- The muon lepton family is not tracked in the trapped mode:
  `beta_eq_neutrino_trapped` takes (n_B, Y_Le, T) and Y_Lmu raises.
- `fixed_YC_YS` with neutralizing leptons (`leptons=True`) is not wired; the
  flag applies to `fixed_YC` only.
- Species-flag naming: the spec calls the meson switch `thermal_mesons`;
  dd2's `SpeciesFlags` carries the finer `include_pseudoscalars` /
  `include_thermal_vectors` pair (and `neutrinos` for the trapped mode, where
  the spec name `thermal_neutrinos` means the untracked mu = 0 gas, which dd2
  does not implement). Unifying the names across models is deferred until the
  other models reach the spec API, so it lands as one rename, not five.

### sfho
- Eta-meson energy density is dropped when `include_pseudoscalar_mesons=True`
  at T > 0 (`thermodynamics_hadrons.py`, in the total-energy accumulation).
  Known, deliberate to fix later: the fix changes numbers, so it is made in
  its own commit against a regenerated baseline with the delta quoted.
- `include_muons` is accepted but the muon sector is not wired everywhere the
  spec requires; per CLAUDE.md §4 an unimplemented flag must raise rather than
  be ignored.
- No `compute_nmp` / `invert_nmp`. dd2 has both; sfho needs the same forward
  and inverse nuclear-matter-parameter maps.

### zl
- Convergence is judged on a sum of squares against a loose 0.01 gate rather
  than a residual norm. Tightening it reclassifies rows near the edges of the
  tables, so it is a baseline-moving change. (vmit had the same gate; it now
  uses a scaled residual norm at 1e-10, and `eos/vmit/eos.py` shows the
  shape zl's should take.)
- `fixed_YC_YS` is physically meaningless (no strangeness in the model) and
  must raise rather than silently ignore Y_S.

### vmit
- `eos_response` implements the freeze `equilibrium` only, and computes it by
  central differences along the mode's own sequence (c_s^2 = dP/deps at fixed
  T, C_V = (T/n_B) ds/dT at fixed n_B) because vMIT has no analytic Jacobian
  in this repository. Frozen composition, frozen conserved fractions and the
  leptonic re-neutralization variants all raise naming this file. An analytic
  Jacobian is straightforward here -- the model has one algebraic mean field
  and no scalar sector -- and would give the susceptibility matrix chi_ab as
  well.
- The muon lepton family is not wired: `SpeciesFlags(muons=True)` raises, and
  `beta_eq_neutrino_trapped` takes (n_B, Y_Le, T) only.
- `thermal_neutrinos` -- flavours not tracked in the composition, carried as
  mu = 0 gases -- is not wired and raises.
- `eos_point` takes the entropy-per-baryon axis; `TableSpec` does not, and
  raises for `axes={'SnB': ...}`. The outer solve exists
  (`eos.general.tabulate.temperature_at_entropy`); wiring it into the table
  driver is what is left.
- The flavour densities are not constrained positive. At exotic fixed
  fractions (Y_C well above 1, say) the equations have solutions with net
  ANTI-down and anti-strange densities, and the solver returns them as
  converged. They are genuine states of the model at finite temperature, not
  solver failures, but nothing in the API says so; a scan over fractions
  should either filter them or the result should carry a flag.

### zl, vmit, alphabag
- `eos/general/tabulate.py` is the shared line-grid driver (warm-started
  density sweep, skipping, progress callback). vmit uses it; zl and alphabag
  still carry their own copies of the same loop in `compute_tables.py`, and
  dd2's `table.py` has a richer version with bisected continuation through
  onsets. When the second and third consumers arrive, the continuation tactics
  should move into the shared driver too rather than being reimplemented.

### enjl
- Finite temperature is not implemented; the model is T = 0 only.
- Cold starts stop converging around 0.5 fm^-3. The beta-equilibrium table is
  built by continuation (`beta_eos_table`), and the "up" and "down" sweeps
  differ where more than one branch exists — that difference is the branch
  structure, and choosing between branches needs a Maxwell construction that
  a single sweep cannot do.

### general
- Most of the thermal meson gas is missing from `general/particles.py`, so it
  cannot be summed as species. `pi+`, `pi-`, `pi0`, `K+`, `K-`, `K0` and `eta`
  are registered; `K0bar`, `eta'` and the whole vector nonet (`rho+-0`,
  `omega`, `phi`, `K*`) are not. Because `PhaseThermo.assemble` derives n_C and
  n_S from the species table, a gas listed in `densities` would land in the
  charge totals automatically — which is exactly what §2 wants and what the
  baryons-only bug class keeps coming from. Until the entries exist, the gas
  reaches the totals through `assemble`'s `extra_charges` argument instead: a
  second summation, correct today because `thermal_meson_charges` is
  validated, but a second path all the same. Adding the entries retires the
  argument. Watch the sign convention when doing it: S = +1 per s quark, so
  K+ = u sbar has S = -1 and K0bar = dbar s has S = +1. The check that the
  entries are right is that summing C_i n_i over the gas reproduces
  `thermal_meson_charges`.
- `_yc_neutralizing_leptons` — the electron/muon gas at the chemical potential
  that neutralises a given charge density — lives in `dd2/physics/octet.py`
  but is fully model-independent: its arguments are (target charge, m_e, m_mu,
  include_muons, T) and no model parameter enters. CLAUDE.md §7 makes lepton
  thermodynamics a `general/` responsibility, and `eos/mixed/coefficients.py`
  already imports it out of dd2's kernels to build its own lepton block. It
  belongs in `general/thermodynamics_leptons.py`.

### mixed
- (The entry claiming the hadronic phase adapter treated the thermal meson gas
  as a spectator in the charge and strangeness bookkeeping was WRONG, and is
  removed. `hadronic_phase` takes n_C and n_S from `assemble_octet`'s totals,
  which count the gas, and the mixed residual uses those totals throughout, so
  the engine has always agreed with dd2 and with CLAUDE.md §2. A comment in
  `phases.py` asserted the opposite and has been corrected; a test now pins
  the behaviour. Nothing moved.)
- `eos_response` implements two of the freezes CLAUDE.md §5 names:
  `'equilibrium'` (nothing held) and `'chi'` (the quark volume fraction held,
  and with it each phase's Y_C and Y_S). Two are not wired and raise:
  frozen per-species composition (all Y_i held, which for a hadronic phase
  with hyperons or Deltas is strictly stronger than holding Y_C and Y_S), and
  frozen conserved fractions with chi left free — which in a Maxwell window
  simply returns to the plateau, so it is only meaningful at eta < 1. The
  susceptibility matrix chi_ab = dn_a/dmu_b is not computed for the mixture
  either; the natural definition has to say whether it is taken at fixed chi.
- `eos_response(frozen='equilibrium')` returns nan outside the coexistence
  window. The mixed system still has a root there (chi runs negative or past
  one) but it is an analytic continuation, not the state — at eta = 1 it sits
  on the pressure plateau at every density. The physically continuous answer
  is the pure phase's own `eos_response`, and stitching the three into one
  curve is left to the caller; the engine does not dispatch across the
  boundary because that would mean importing model internals past the
  phase-adapter surface.
- The public boundary takes `Y_Le` (the §5 condition name) while the engine's
  own `ChargeSpec`, `MODE_FRACTIONS` and table axes call the same quantity
  `Y_L`. `eos/mixed/api.py` translates in one place and refuses the internal
  name, so the two never coexist at one boundary, but the internal rename is
  still outstanding and belongs with the next change that touches
  `equilibrium/charges.py`.
- `mixed_slots` activates the local and global lepton populations at exactly
  eta > 0 and eta < 1. Within about 1e-3 of an endpoint the just-activated
  population carries almost no weight, its potential is a near spectator and
  the Jacobian is near-singular, so a cold start can stall there. Interior eta
  on practical grids is fine; a near-endpoint eta needs a warm start from the
  eta = 0 or eta = 1 solution.

### astro/tov
- The fast backend returns a silently wrong tidal deformability when the table
  it is handed is not monotone in pressure. On the eta = 1 hybrid table, whose
  Maxwell plateau produced a few dozen round-off inversions of order
  1e-13 MeV/fm^3, it gave Lambda = 14 for a 0.94 Msun star where the scipy
  reference gave 2.8e6 — while M and R still agreed to 1e-4, so nothing else
  flagged it. CLAUDE.md §6 says non-convergence is a return value: meeting a
  non-monotone table is exactly that case and must come back as a status, not
  as a number. The cause has been removed upstream — `build_mixed_eos_table`
  now enforces §8 before the table is delivered — but the backend is still
  fragile to any other caller that hands it one.
- Crust table paths are absolute and machine-specific. A missing crust file
  currently degrades to no crust, which shifts M_max by ~1%; it must instead
  be an explicit argument with an informative error.
- The two TOV backends differ by about 2% in the tidal deformability on a
  hybrid EoS, and that difference converges with the resolution of the EoS
  table rather than with the central-density grid: doubling the density grid
  from 220 to 440 points takes the gap from 6.3% to 2.2% at eta = 1 and leaves
  M_max agreeing to 4e-4 Msun. Neither backend is wrong, but 2% is large for
  two integrators on the same table, and where the remaining difference comes
  from has not been chased down — the interpolation of eps(P) near the
  transition is the first suspect. `test/mixed/test_tov_backend_parity.py`
  pins the measured numbers.
