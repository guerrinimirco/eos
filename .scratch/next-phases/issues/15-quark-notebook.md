# notebooks/quark_eos — skeleton, knobs and figures

Type: task
Status: open
Blocked by: 04, 05, 61
Parent: ../map.md

## Question

Stage 2. Same shape as `hadronic_eos` — same knobs cell, same jupytext pairing,
same figure rules — for `vmit`, `alphabag`, `njl`, `ccdm`, plus whatever ticket 05
ruled about `abpr`. **Say in the intro whether `abpr` belongs here and why**, per
that ruling.

Quark-model figures use the pure-quark planes that make sense: `P` vs `n_B`,
`eps` vs `P`, `c_s²` (named `cs2_isothermal` or `cs2_adiabatic`, never bare), and
flavour composition `Y_u`, `Y_d`, `Y_s`, `Y_e`.

**Where a bare quark model gives no stable star, say so** instead of plotting an
empty M–R panel.

Figures to `output/quark/`. Done when the notebook executes clean and every
figure file exists.

## Added by ticket 05

**`abpr` is in, as a companion panel — not a fifth peer in the knobs cell.**
One figure against `alphabag` at CFL and T = 0, showing the O(m_s^4) difference
`abpr/verify/run_full_check.py` already measures; the two are driven as a matched
pair through alpha_s = pi/2 (1 - a4). It is CFL-only and T = 0-only, so as a peer
it would trip ticket 04's unsupported-combination pattern on nearly every cell.

`abpr` has no `table.py` but **does** have `eos_table` (`abpr/api.py:146`) —
nothing in the model iterates, so §5's conditional-existence clause is satisfied
and there is no gap to work around.
