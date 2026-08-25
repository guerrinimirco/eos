# notebooks/hybrid_eos — skeleton, knobs, the pairing choice and the tables

Type: task
Status: open
Blocked by: 04, 05
Parent: ../map.md

## Question

The fourth notebook, added by [ticket 05](05-notebook-coverage.md) and not by
`docs/NEXT_PHASES_PROMPT.md`. Subject: hybrid constructions — how a caller picks
the two phases, sets both parameter sets, and gets a table out.

Copy ticket 04's spine, then add what a composite engine has and a model does
not:

1. **The knobs cell gains a pairing choice.** `eos/mixed/adapters.py` ships
   `dd2_phase`, `sfho_phase`, `did_phase`, `zl_phase` (hadronic) and
   `vmit_phase`, `alphabag_phase`, `njl_phase`, `ccdm_phase` (quark). A `Phase`
   pair IS the parameter argument (§5), so the cell selects two adapters and two
   parameter sets, not one. The plain `(par, flags, vmit_params)` signature stays
   the DD2+vMIT front door and the notebook should show both forms.
2. **`eta` is a scalar per call, not an axis** — `mixed/api.py` says so
   explicitly, because it changes the shape of the unknown vector. eta = 0 is
   Gibbs, eta = 1 Maxwell. The knobs cell must not present it as a grid.
3. **`eos_table` returns `(rows, windows)`.** The phase boundaries are part of
   the result, not a by-product (§5). Tables saved under ticket 04's naming
   scheme must carry the windows, not drop them.
4. **The headline pairing is DD2 + vMIT**, run end to end. It is what
   `output_old/eos_tables_DD2vMIT_from_notebooks/` holds — 32 tables, 42 figures
   — so this notebook is a checkable replacement for a retired one, and the
   comparison is [ticket 59](59-hybrid-figures.md)'s to make.
5. **ENJL is out of this notebook.** `enjl_branch_pair` is an `eos/mixed`
   adapter but the physics is ENJL's; [ticket 18](18-enjl-notebook.md) owns it.
   State the boundary in one line here and in that notebook.

`hybrid_table` (`mixed/api.py:227`) is the stitched hadronic + mixed + quark
core at one equilibrium and returns a `HybridResult` — test `.ok`, §6.

Resolved when the notebook builds a converged DD2+vMIT table and saves it.
