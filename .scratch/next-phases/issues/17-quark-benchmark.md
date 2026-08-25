# notebooks/quark_eos — the benchmark section

Type: task
Status: resolved
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

## Answer

**Shipped: section 9 of [notebooks/quark_eos.py](../../../notebooks/quark_eos.py)**,
commits d0445fb (the section) and f2dee22 (one prose fix, below). Every timing
is `timeit`/`time` around a public call or a field of the `progress` dict; **no
library file was touched by this ticket**, and no solver internal is read.

Verified by `jupytext --to notebook --execute` in an isolated `git archive HEAD`
copy of the **committed** tree at d0445fb: **53 cells, 25 code cells, 0 error
outputs**, exit 0. Interpreter **python.org 3.14.2**. The isolated copy is why
the claim is about the committed tree rather than the working tree, where a
concurrent session is editing `eos/njl/api.py` and `eos/ccdm/api.py`.

### The numbers

24-point line spanning n_B = 0.05-3.0 fm^-3, deliberately wider than a
production table so the non-convergence counter reports something:

| model | mode | T | cold ms | warm ms/pt | line s | solved |
|---|---|---:|---:|---:|---:|---|
| vmit | beta_eq_neutrinoless | 0 | 0.129 | 0.260 | 0.006 | 24/24 |
| alphabag | beta_eq_neutrinoless | 0 | 0.082 | 0.115 | 0.003 | 24/24 |
| njl | beta_eq_neutrinoless | 0 | 42.2 | 122.7 | 2.82 | 23/24 |
| ccdm | beta_eq_neutrinoless | 0 | 93.5 | 10853.7 | 184.5 | 17/24 |
| vmit | fixed_YC | 10 | 0.807 | 0.367 | 0.009 | 24/24 |
| alphabag | fixed_YC | 10 | 0.071 | 0.106 | 0.003 | 24/24 |
| njl | fixed_YC | 10 | 43.9 | 125.0 | 2.88 | 23/24 |
| ccdm | fixed_YC | 10 | 106.2 | 13336.2 | 226.7 | 17/24 |

**The four span three orders of magnitude**, which is the finding. A closed-form
bag model solves a point in under a tenth of a millisecond; `njl` pays tens of
milliseconds for a gap equation re-solved at every residual call; `ccdm` pays
more again for enumerating the chiral/dielectric branch and the pairing pattern
on top of it. Section 9 costs about ten minutes to execute and essentially all
of it is `ccdm` — stated in the notebook, since a reader should not discover it
by waiting.

**`warm` is not a per-solved-point cost for `ccdm`, and that is why cold and
warm are reported separately.** Its cold point is around a tenth of a second
while its warm figure is ten seconds and more — two orders above its own cold
cost, where every other model's two numbers sit within a factor of three. The
solved points did not get slower: the line spends its wall clock on the points
it never solves, each retried through up to `MAX_BISECT = 6` halved steps with a
full candidate enumeration per retry. Those attempts are in `elapsed_s` and not
in `n_solved`. Honest arithmetic for anyone budgeting a table, which does pay
for the attempts.

### Non-convergence: counted, never raised, and the densities matter

    [vmit     both configs]  0 of 24
    [alphabag both configs]  0 of 24
    [njl      both configs]  1 of 24  at n_B = 3.000
    [ccdm     both configs]  7 of 24  at n_B = 0.178 ... 0.948

The two that miss, miss in different places, which is why the section prints
densities and not only counts. **`njl` misses at the top** — 3.0 fm^-3, the last
grid point, the far end of a cutoff-regularized model's domain. **`ccdm` misses
an interior band**, 0.178-0.948, identically in both configurations: below its
deconfinement onset there is no deconfined phase, which its own `table.py`
states as physics.

**One observation reported, not diagnosed.** `ccdm`'s first density, 0.05
fm^-3, *solves*, while the band immediately above it does not — the miss window
is interior on both sides. Whatever root the solver finds down there is worth a
look before anyone leans on it. The notebook says exactly that and no more; a
mechanism would be a guess.

### Bottleneck

`njl`, beta equilibrium, T = 0. Cumulative time runs
`solve_system` -> `solver.residual` -> `_state` -> `thermodynamics.state_at`:
the NJL state itself — constituent masses and the cutoff-regularized integrals
— rebuilt on every residual call. The call counts are the other half: about
**1600 residual evaluations for ~30 attempted points, some 50 per point**
(52 `scipy.optimize.root` calls for 31 attempts). That is what a
**finite-difference** Jacobian over this unknown vector costs. So the gap
against the bag models is two compounding factors, not one: a residual that is
itself far more expensive, evaluated many more times per solve.

No Numba hazard here, unlike the hadronic notebook's T = 0 line — none of these
models ships a jitted kernel, so a cold profile does not report a compiler.

### Backends (section 9): there is nothing to compare, and that is checked

    [vmit    ] backends/ absent   backend switch on eos_point: none
    [alphabag] backends/ absent   backend switch on eos_point: none
    [njl     ] backends/ absent   backend switch on eos_point: none
    [ccdm    ] backends/ absent   backend switch on eos_point: none
    [abpr    ] backends/ absent   backend switch on eos_point: none

Executed output, not prose: no `eos/<model>/backends/` and no backend switch in
any `eos_point` signature. `vmit` and `alphabag` say the same thing in their own
`eos_response` docstrings ("no analytic Jacobian in this repository"). Every
number above is therefore a reference number needing no second column — and it
is also the direct cause of the ~50 residual evaluations per point above.

The nearest thing the quark side ships is a **pair of models**, not of backends,
and the section labels it so: `abpr` evaluates CFL in closed form where
`alphabag` root-finds it. **0.170 ms against 3.358 ms** for the same 24-point
line, about twentyfold, with `abpr` solving 24 rows where `alphabag` solves 23.
Explicitly not counted as a backend-parity check.

### The ticket-04 naming, and the `table_path` root bug

    output/tables/quark/quark_benchmark_T0.0-10.0x2_nB0.1-3.0x24_ph.h5

Written through `standard_name` / `table_path` with the model slot carrying the
study, since the table spans four models. **`root=str(ROOT / "output" /
"tables")` is passed and is not decoration**: `table_path`'s default root is the
relative `output/tables`, so under `jupytext --execute` — which runs with cwd at
`notebooks/` — the file lands in `notebooks/output/tables/`. Confirmed by the
executed run writing to the repository's `output/tables/quark/` with the
argument and to `notebooks/output/` without it.

**Section 4's existing save is deliberately left on the default relative root.**
[Ticket 15](15-quark-notebook.md) ruled that the fix belongs in
`eos/general/table_io.py`, uniformly for all four notebooks rather than forked
into one, and overturning that from inside a single notebook is not this
ticket's call. The divergence is stated in the notebook where it would otherwise
puzzle a reader.

Cosmetic, noticed while reading the name: `_span` formats with `%.1f`, so a grid
starting at 0.05 renders as `nB0.1`. Two grids differing only below 0.1 fm^-3
would collide. Not fixed here — `standard_name` is ticket 04's.

### One prose fix after the verification run (f2dee22)

The first draft quoted absolute millisecond ranges read off a single execution.
A second run on a quieter machine moved every one of them by ~25% while
reproducing the ordering, both miss sets and every ratio exactly. The prose now
states the ratios, which are what the section actually claims and what any run
reproduces. The tables in this answer are from the committed-tree run.

### Tests

Targeted, in a second isolated HEAD copy at d0445fb with a snapshot of the
gitignored `test/`: `test/general test/test_imports.py test/vmit test/alphabag
test/njl test/ccdm test/abpr` = **809 collected, 809 passed, 0 failed** in 43 s,
python.org 3.14.2. Identical to [ticket 15](15-quark-notebook.md)'s count, which
is the expected result of a notebook-only change: **0 added failures**. The full
suite was not run (concurrency). Run in the isolated copy rather than in place
because the working tree carries another session's edits to `eos/njl/api.py` and
`eos/ccdm/api.py`, which a run here would have attributed to this ticket.

### Scope

`notebooks/quark_eos.py` only, plus this file. Ticket 16's step-by-step NJL/CCDM
section was not started and no other notebook was touched.

Status: resolved.
