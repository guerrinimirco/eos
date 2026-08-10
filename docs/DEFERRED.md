# DEFERRED — known gaps, per model

The tracked ledger CLAUDE.md refers to: modes a model does not support, physics
not yet wired, and behaviour that is understood but not yet fixed. A gap
recorded here is a decision; a gap not recorded here is a bug.

Each entry says what the gap is, how it shows up, and what closing it would
take. Entries are removed when closed, not marked "done".

---

## Cross-cutting

### mu_S is undetermined when no strange species is populated

**Models:** sfho (observed), and any model exposing `fixed_YC_YS`.

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

Closing it means deciding what the API should say when a conserved charge is
carried by no populated species. Options: report mu_S as undefined (NaN) with
a status flag; pin it by convention (mu_S = 0) and document that; or have the
mode raise. Until then `test/baseline` does not freeze mu_S where n_S is zero,
because there is nothing there to freeze.

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

## Per model

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

### zl, vmit
- Convergence is judged on a sum of squares against a loose 0.01 gate rather
  than a residual norm. Tightening it reclassifies rows near the edges of the
  tables, so it is a baseline-moving change.

### zl
- `fixed_YC_YS` is physically meaningless (no strangeness in the model) and
  must raise rather than silently ignore Y_S.

### enjl
- Finite temperature is not implemented; the model is T = 0 only.
- Cold starts stop converging around 0.5 fm^-3. The beta-equilibrium table is
  built by continuation (`beta_eos_table`), and the "up" and "down" sweeps
  differ where more than one branch exists — that difference is the branch
  structure, and choosing between branches needs a Maxwell construction that
  a single sweep cannot do.

### mixed
- The hadronic phase adapter treats the thermal meson gas as a spectator in
  the charge and strangeness bookkeeping, while dd2 counts it. CLAUDE.md §2
  fixes the convention: mesons carry C and S. Fixing the adapter moves numbers
  at T > 0 with `thermal_mesons=True`.

### astro/tov
- Crust table paths are absolute and machine-specific. A missing crust file
  currently degrades to no crust, which shifts M_max by ~1%; it must instead
  be an explicit argument with an informative error.
