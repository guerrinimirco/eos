# The shared notebook skeleton: knobs cell, gap handling, table naming

Type: prototype
Status: open
Blocked by: 02
Parent: ../map.md

## Question

All three notebooks share one spine. Build it once, concretely, as a throwaway to
react to — not prose about what it might look like.

Three things to pin down:

1. **The knobs cell** (first executable cell, everything selectable from it):
   model subset; mode (§3) with its own conditions (`Y_C`, `Y_S`, `Y_Le`,
   `Y_Lmu`) and the `leptons` flag; the `n_B` grid and either a `T` grid or an
   entropy-per-baryon `SnB` grid; the six species flags (§4); and the
   parametrisation per model — published sets via `Parameters.default()` /
   `Parameters.named(...)`, plus, for models carrying `nmp.py`, a set built by
   inverting `{n_sat, E_sat, m*/m, K_sat, E_sym, L_sym}` with `Q_sat`/`K_sym`
   reported as predictions.

2. **The unsupported-combination pattern.** A mode, flag or parametrisation a
   model does not support raises with a message saying which (§3). The notebook
   catches that at the top of a section, prints the message, and continues with
   the models that do support it. It never presents a gap as a result. Settle
   the exact shape of that try/except block and its printed form.

3. **The table-save convention.** Stage 1 asks that every table produced show how
   to save it to `output/tables/` under a standardised, automatic name. Settle
   the naming scheme (model, mode, fractions, grid, species flags) and the
   one-call helper form — noting that §11 forbids a helper module beside the
   notebook, so it lives in the notebook or in `eos/general/` table I/O.

Deliver as a runnable `.py` fragment. Resolved when the user has reacted to the
concrete artifact and the three shapes are fixed; tickets 12–19 then copy it.
