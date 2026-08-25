# notebooks/hybrid_eos — skeleton, knobs, the pairing choice and the tables

Type: task
Status: resolved
Blocked by: 04, 05, 61
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

## Answer

**Reconstructed after the fact.** The session that did this work marked the
ticket resolved at 21:44 and stopped without committing either file, leaving
`notebooks/hybrid_eos.py` and its paired `.ipynb` untracked. It did leave a
`## Outcome` section — not this repository's `## Answer` heading, and with no
closing status line — and every claim in it was re-measured and holds; it is
restated below in this document's usual shape. What follows was written by a
later session reading the recovered code and its executed outputs, not by the
session that wrote them; treat the *rationale* accordingly, the *numbers* were
verified independently.

### What the notebook is

`notebooks/hybrid_eos.py` + paired `.ipynb`, 12 code cells, executed clean end
to end. It is driven entirely through `eos.mixed`'s public surface —
`eos_point`, `eos_table`, `hybrid_table` and `eos.mixed.adapters` — with no
helper module beside it, per §11.

Seven sections: the knobs; the three distinct ways a call can end; the two
calling forms; the sixteen shipped pairings; `eta` one scalar call at a time;
a table as rows plus windows, saved; and the stitched `hybrid_table` with the
handoff into `eos.astro.tov`.

### Against the five items of the question

* **(1) The knobs cell names two adapters and two parameter sets.** `Knobs`
  carries `hadronic`/`quark` adapter names beside `hadronic_parameters`/
  `quark_parameters`, and `phase_pair()` builds the `Phase` pair by closing
  each factory over its own model's `Parameters` — which is how "model
  parameters are arguments" reads for a composite engine. Both calling forms
  run and agree bit-for-bit at n_B = 0.75 fm^-3: chi = +0.4726370426,
  P = 257.0946821684, eps = 906.3334419744, and the notebook prints the
  `==` comparison rather than a tolerance. `zl` is special-cased in the
  factory call because it is written in (n_p, n_n) and takes no species flags.
* **(2) `eta` is a scalar field**, with a separate `eta_examples` tuple that is
  a list of separate calls; `Knobs.axes()` carries a comment saying `eta` is
  deliberately absent from it. The Maxwell plateau is shown rather than
  asserted: across n_B = 0.70 and 0.75 inside the window, dP = -1.023e-12
  MeV/fm^3 at eta = 1 against +3.057e+01 at eta = 0.
* **(3) `eos_table` is unpacked as `(rows, windows)`** and the windows go into
  the file through `save_table(windows=...)`; the automatic name carries eta.
* **(4) The headline DD2 + vMIT runs end to end** and is saved to
  `output/tables/dd2vmit/dd2vmit_beta_eq_neutrinoless_eta0.50_T0.0-20.0x2_nB0.1-1.4x27_ph.h5`
  — 11 in-window rows, windows located on both lines (T = 0:
  n_onset = 0.6309, n_offset = 0.9241 fm^-3; T = 20: 0.6069, 0.8862).
  `hybrid_table` runs all three modes with P non-decreasing in every one, and
  `.to_tov()` returns an `EOSTable_for_TOV`.
* **(5) The ENJL boundary is stated** in the closing section: `enjl_branch_pair`
  is an `eos/mixed` adapter but pairs two branches of one functional, so it
  belongs to [ticket 18](18-enjl-notebook.md)'s notebook.

### Three things the executed output makes visible

* **A non-convergence is a return value.** All sixteen pairings go through the
  one engine at one density; `dd2+njl` fails there and is *printed* as "did not
  converge" with its residual (1.55e-01 against tol 1e-10), never raised and
  never folded into the "unsupported" bucket. `run()` catches only
  `NotImplementedError` and `ValueError`; `TypeError` is deliberately left to
  propagate, so a wrong keyword is this notebook's bug and not filed as an
  engine gap.
* **chi outside [0, 1] is an answer, not a failure.** At one fixed density most
  pairings are not in coexistence, and the notebook reports `point.phase`
  ('H' / 'mix' / 'Q') beside every chi so the reader can tell a converged pure
  phase from a mixed one. `sfho+vmit` at chi = -1.61 is the hadronic phase
  saying its window is elsewhere.
* **The per-phase charge partition closes.** On the headline rows,
  max |Y_B_H + Y_B_Q - 1| = 2.887e-15, max |Y_C_H + Y_C_Q - Y_C| = 1.388e-17,
  max |Y_S_H + Y_S_Q - Y_S| = 0.

### The `table_path` root

`table_path`'s folder is relative to the working directory and a notebook's is
`notebooks/`, so the save passes `root=ROOT / "output" / "tables"` with `ROOT`
the repository root already located for the import. This is the same trap
[ticket 17](17-quark-benchmark.md) and [ticket 18](18-enjl-notebook.md)
recorded. A draft run at 21:38, before that fix landed at 21:43, had left one
table under `notebooks/output/tables/dd2vmit/`; the shipped cell does not
produce it, and the stray tree was removed. The original `## Outcome` also
claimed `notebooks/hadronic_eos.py` still had the latent issue — it does not;
[ticket 13](13-hadronic-figures.md) closed it there, and nothing of hadronic's
was found under `notebooks/output/`.

### How it was verified

Executed with `jupytext --to notebook --execute` under python3 3.14.2 in an
isolated `git archive HEAD` copy (HEAD = 3d09b4c), so no concurrent session's
working-tree edits could contribute: 0 errors across 12 code cells. The
committed `.ipynb`'s stored outputs match that run line for line, differing
only in the ROOT-anchored save path. Committed as found in bbf07f9, without
edit or restructuring, so the recovery is distinguishable from a rewrite.

Figures, the TOV pass and the swap cell are [ticket 59](59-hybrid-figures.md)'s.

Status: resolved.
