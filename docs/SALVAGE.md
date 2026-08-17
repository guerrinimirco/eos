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

### 1. Locate a phase boundary by imposing chi and solving for n_B

**The best idea in it.** The mixed system is normally solved at a given n_B,
returning the quark volume fraction chi. To find where the mixed phase begins
and ends, that has to be inverted, and there are two ways:

- **Scan and bisect** — what `eos/mixed/solvers/sweep.py:locate_window` does:
  probe the density grid, read chi as a regime indicator, bisect the chi = 0
  and chi = 1 crossings to half a grid spacing. Its own docstring prices this
  at "a couple of dozen solves", and it can miss a window thinner than the
  coarse probe spacing (hence `max_refine`).

- **Swap the unknown** — what `solve_eta0_fixed_chi_beta` and its three
  siblings did: put n_B IN the unknown vector and impose chi instead.

      normal:      unknowns [sigma, omega, rho, phi, mu_B_H, mu_C_H,
                             mu_u, mu_d, mu_s, mu_eG, chi]      given n_B
      fixed-chi:   unknowns [sigma, omega, rho, phi, mu_B_H, mu_C_H,
                             mu_u, mu_d, mu_s, mu_eG, n_B]      given chi

  Every equation is unchanged — the four field equations, beta equilibrium in
  each sector, mu_S_Q = 0, baryon conservation (1-chi) n_B_H + chi n_B_Q = n_B,
  global neutrality, mu_B_H = mu_B_Q, and P_H = P_Q. Only which symbol is
  solved for moves. chi = 0 then returns n_onset and chi = 1 returns n_offset,
  each in ONE solve, exactly, with no grid resolution in the answer at all.

  This is the same trick as CLAUDE.md section 3's modes — a mode is a choice of
  which variable is imposed and which is unknown — applied to chi rather than
  to a conserved charge. `eos/mixed` already assembles its residual from a
  declaration (`ChargeSpec`), so the natural form there is to let the
  declaration carry "chi imposed, n_B unknown" as one more slot choice, rather
  than to write a second family of solvers as the old code did (four functions:
  eta0/eta1 x beta/fixed_YC).

  Worth doing when the window locator is next touched. Keep the scan as the
  cold-start finder — the fixed-chi solve needs a starting n_B and can walk to
  the wrong root without one — and use it to seed the exact solve.

### 2. Seed the offset from the converged onset

`find_phase_boundaries_single` took the converged chi = 0 solution, copied its
unknown vector, and multiplied the n_B entry by 2.5 to seed the chi = 1 solve.
Crude, and the 2.5 is fitted to nothing, but it is the right shape: the two
boundaries of one transition are far more alike than either is to a cold start.
The same relation holds in `eos/mixed`, where onset and offset are currently
found independently along the probe sweep.

### 3. Continuation along temperature, not only along density

`extrapolate_guess(history, T_target, fallback)` kept a list of (T, unknowns)
pairs and linearly extrapolated the next seed in T, falling back to a copy when
fewer than two points were available. `find_all_boundaries` then marched the
whole T grid this way, so each temperature's boundary search started from the
previous temperature's answer.

The repository warm-starts along density everywhere (CLAUDE.md section 5) and
along temperature nowhere; `eos/mixed`'s window locator accepts a `hint` span
from a neighbouring temperature but not a seed vector. For a boundary search,
where the expensive part is the solve rather than the sweep, marching T with an
extrapolated seed is the bigger win of the two.

### 4. Caching pure-phase tables between runs

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
