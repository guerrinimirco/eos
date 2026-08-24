# notebooks/quark_eos — the benchmark section

Type: task
Status: open
Blocked by: 15
Parent: ../map.md

## Question

Stage 2 benchmarks, identical in shape to ticket 14 but for the quark models:
cold vs warm `eos_point` timings reported separately, wall time for one full
`n_B` line, non-converged counts and their `n_B` values, `cProfile` top ~15 with
a one-line reading, a cross-model summary table, and reference vs fast backend
side by side where a model ships one (§9).

Built on the `progress` callback (§5). No timing hooks in library code.

Done when the table carries real numbers, and the `.ipynb` is committed without
stored outputs (`docs/strip_notebook_outputs.py`).
