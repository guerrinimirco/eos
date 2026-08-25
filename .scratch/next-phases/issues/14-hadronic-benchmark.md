# notebooks/hadronic_eos — the benchmark section

Type: task
Status: resolved
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

## Answer

Section 6 of [notebooks/hadronic_eos.py](../../../notebooks/hadronic_eos.py),
commit 4d02856. Every timing is `timeit`/`time` around a public call or a field
of the `progress` dict; no library file was touched.

Real numbers, python.org 3.14, 64-point line spanning n_B = 0.002-3.0 fm^-3
(deliberately wider than a production table, so the non-convergence counter
reports something):

| model | mode | T | cold ms | warm ms/pt | line s | solved |
|---|---|---:|---:|---:|---:|---|
| zl | beta_eq_neutrinoless | 0 | 0.345 | 0.260 | 0.017 | 64/64 |
| sfho | beta_eq_neutrinoless | 0 | 1.309 | 3.701 | 0.155 | 42/64 |
| dd2 | beta_eq_neutrinoless | 0 | 0.601 | 0.083 | 0.005 | 64/64 |
| did | beta_eq_neutrinoless | 0 | 1.100 | 1.240 | 0.079 | 64/64 |
| zl | fixed_YC | 10 | 0.132 | 0.413 | 0.026 | 64/64 |
| sfho | fixed_YC | 10 | 2.191 | 3.562 | 0.150 | 42/64 |
| dd2 | fixed_YC | 10 | 1.250 | 0.290 | 0.019 | 64/64 |
| did | fixed_YC | 10 | 1.753 | 1.070 | 0.069 | 64/64 |

`warm` is `elapsed_s / n_solved` from the callback, so a line with misses pays
for the attempts in the numerator and not the denominator - which is why
`sfho`'s warm column sits above its cold one.

**Non-convergence.** Only `sfho`, and the same 22 points in both
configurations: every requested density from n_B = 2.001 fm^-3 up. `zl`, `dd2`
and `did` solve all 64. Counted and reported, never raised (section 6).

**Bottleneck.** `dd2` beta equilibrium at T = 10 MeV: 0.023 s of 0.030 s under
MINPACK `hybrj`, spent in `residual` -> `kinetic_thermo` -> `solve_fermi_jel`,
which is also the largest entry by internal time. The analytic Jacobian costs
about what the residual it differentiates costs.

A trap the notebook documents: a T = 0 line profiled in a fresh process reports
Numba compiling the T = 0 kernel (`llvmlite`, `install_registry`,
`marshal.loads` at the top), not physics. The profile cell runs after the
benchmark cells for that reason.

**Backends.** `dd2` is the only one of the four whose fast backend is reachable
from the public API (`eos_point(..., analytic_jac=)`): 1.291 ms reference
against 1.341 ms fast per cold point - the exact Jacobian is not a speed win at
one point, and the table path takes it by default anyway. `sfho` ships one and
leaves it off for the reason its own docstring gives; `zl` and `did` ship no
`backends/`.

**Saved** to `output/tables/hadronic/hadronic_benchmark_T0.0-10.0x2_nB0.0-3.0x64_ph.h5`
via `standard_name`/`table_path`, the model slot carrying the study because the
table spans four models.

Open question for the user, not fixed here: `sfho` failing every point above
n_B = 2 fm^-3 in both a beta-equilibrium and a fixed-Y_C sweep is a solver
limit worth a ticket of its own.
