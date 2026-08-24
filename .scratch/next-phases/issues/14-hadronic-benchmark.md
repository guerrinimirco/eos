# notebooks/hadronic_eos — the benchmark section

Type: task
Status: open
Blocked by: 12
Parent: ../map.md

## Question

Stage 1, benchmarks. Built on the existing `progress` callback (§5), whose
dictionary already carries `n_solved`, `n_requested` and `elapsed_s` — **do not
add timing hooks to library code.** Per model and configuration report:

- time for a single `eos_point`, **cold start and warm-started inside a sweep
  reported separately** — they are different numbers
- wall time for one full `n_B` line at fixed `T` and fixed fractions
- non-converged points: the count, and the `n_B` values where they fall
  (non-convergence is a return value, §6 — the benchmark counts them, it does not
  crash on them)
- bottlenecks: `cProfile` on one representative line, top ~15 by cumulative time,
  plus a one-line reading of what dominates (integrals, Jacobian, field solve)
- a summary table across models, and — where a model ships one — reference vs
  fast backend side by side (§9)

Each table shows how to save it to `output/tables/` under the ticket-04 naming
convention.

Done when the benchmark table is populated with real numbers, and the `.ipynb` is
committed **without stored outputs** (`docs/strip_notebook_outputs.py`).
