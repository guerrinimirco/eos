# SALVAGE — what was worth keeping from deleted code

When a module is deleted, the code goes but the ideas in it should not have to
be rediscovered. This file records the approaches worth reusing, where they
came from, and what in the current repository they would improve. It is not a
changelog: a technique lands here only if it is genuinely better than, or
absent from, what replaced it.

---

## `eos/sfhoalphabag/` — first-generation SFHo + AlphaBag hybrid

Deleted with zero importers anywhere in `eos`, `nucleation`, the notebooks or
`test`; superseded by `eos/mixed`, which couples phases through the
phase-adapter contract instead of importing two models' internals. 2895 lines
in `mixed_phase_eos.py` (the solver) and `hybrid_table_generator.py` (the run
orchestrator).

### 1.-3. Boundary location by the fixed-chi solve — LANDED

The three boundary-search ideas this section carried are now the shipped
implementation in `eos/mixed/boundaries.py` and `eos/mixed/table.py`:

- **impose chi, solve for n_B** — `mixed_slots(..., fixed_chi=True)` swaps the
  one slot and `solve_fixed_chi` lands on a boundary in one solve, exactly,
  with no grid resolution in the answer. As predicted here, it went in as one
  more slot choice on the derived unknown vector, not as a second solver
  family. The scan of `locate_window` stayed as the cold-start finder that
  seeds it (`refine_window`): the scan decides which root, the exact solve
  decides where.
- **seed the offset from the converged onset** — `refine_window` does exactly
  that, with the scan's own offset estimate in the density slot instead of the
  old fitted 2.5 factor.
- **march the boundary search along temperature** — `table._march_boundaries`
  extrapolates the last two converged boundary vectors linearly in T
  (converged-only history, so a failed isotherm cannot poison it) and replaces
  the per-isotherm scan with two warm-started solves.

### 4. Caching pure-phase tables between runs (still open)

`save_pure_table` / `load_pure_table` pickled the pure hadronic and pure quark
tables to disk, keyed by phase, equilibrium mode, quark parameters and the n_B
grid, so a re-run at a new eta did not re-solve either pure phase. A hybrid scan
over several eta values recomputes both pure phases every time today. Note the
key has to include everything the table depends on — the old key did not include
the SFHo parametrization, so switching hadronic model while keeping the filename
would have loaded the wrong cache.

### 5. Nothing else

The rest was structure the current engine already has in better form: the eta
parametrisation with its three unknown-vector layouts (11 / 12 / 13 unknowns for
eta = 0, eta = 1, and 0 < eta < 1) is `eos/mixed`'s `mixed_slots`; the
`EquilibriumMode` enum with two members is `ModeSpec` with four; the per-mode
solver quartet is exactly the per-mode duplication CLAUDE.md section 5a exists
to remove. `include_gluons` is AlphaBag's own sector flag and lives with that
model.

---

## `eos/sfho/compare_with_compose.py` — first-generation CompOSE comparison

Deleted with zero importers anywhere in `eos`, `nucleation`, the notebooks or
`test`. 880 lines: a second CompOSE reader (`compose_loader.py` has its own,
self-contained one), a loader for pre-computed `.dat` tables out of `output/`,
two six-panel matplotlib comparison figures, and a `run_comparison` driver.
The comparison logic is now `eos/sfho/verify/compose.py`, in the shape
`eos/dd2/verify/compose.py` already had.

### 1. Interpolate the CompOSE temperature, do not snap to it

**Kept, and it is the one thing in the file that changed a number.** The
CompOSE T grid is logarithmic and coarse — the points either side of 10 MeV
are about 8.9 and 11.2 — so selecting the nearest temperature compares the
engine at one T against the table at another, and the offset shows up as an
apparent disagreement of order a percent that is not a disagreement at all.
`get_compose_slice(..., interpolate_T=True)` linearly interpolated every
quantity between the bracketing grid points, snapping only within 0.01 MeV of
a grid point to avoid interpolating noise where it was not needed. That is now
`verify/compose.slice_at`. Y_q is still snapped, deliberately, and the snapped
value is reported back so the engine is run at the same fraction: a 0.01
mismatch in Y_q moves the pressure of neutron-rich matter by more than the
agreement being measured.

### 2. What a CompOSE table does NOT contain

`add_neutrino_to_compose` added three mu = 0 neutrino flavours to a CompOSE
slice before comparing. Not carried over as code — the engine's
`thermal_neutrinos` flag is off by default and both sides then exclude them,
so adding to one side alone would break agreement rather than improve it — but
the underlying point is worth keeping and is now stated in the module
docstring: the general-purpose tables carry baryons, electrons/positrons at
net n_e = Y_q n_B, photons and NUCLEI, and neither muons nor neutrinos. That
list is what decides the comparison conditions (`fixed_YC` with
`leptons=True`, photons on, muons off, densities above cluster dissolution),
and getting it wrong is the easiest way to measure bookkeeping instead of
physics.

### 3. Nothing else

The plotting is a notebook's job (CLAUDE.md §11: notebooks carry their own
plotting code), the second reader is the duplicate the refactor exists to
remove, and the `.dat` table loader duplicates `table.load_eos_table`.
