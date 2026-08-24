# notebooks/quark_eos — skeleton, knobs and figures

Type: task
Status: open
Blocked by: 04, 05
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
