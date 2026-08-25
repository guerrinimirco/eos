# notebooks/enjl — step-by-step treatment and benchmarks

Type: task
Status: resolved
Assignee: session bc56a22b
Blocked by: 18
Parent: ../map.md

## Question

Stage 3, second half. Cover the same step-by-step treatment as ticket 16 **where
the physics has the corresponding part, and say plainly where it does not** — an
ENJL branch pair is not a pairing pattern, and the section must not pretend
otherwise.

Then the benchmark section, identical in shape to tickets 14 and 17: cold vs warm
`eos_point`, wall time per `n_B` line, non-converged counts and locations,
`cProfile` top ~15 with a reading, summary table, reference vs fast backend where
one ships (§9). Built on the `progress` callback; no timing hooks in library code.

Done when the benchmark table carries real numbers and the `.ipynb` is committed
without stored outputs.

## Answer

**Shipped: sections 9 and 10 of
[notebooks/enjl_eos.py](../../../notebooks/enjl_eos.py)** paired to `.ipynb`,
commits `5aae00b` (the two sections) and `f8a0c79` (three prose corrections
against the executed profile, below). **No library file was touched by this
ticket** and no solver internal is read: every timing is `timeit`/`time` around
a public call or a field of the `progress` dict, and the step-by-step section
drives `eos_point` / `eos_table` and their result objects alone.

Verified in an isolated `git archive HEAD` copy at **`f8a0c79`**, kernel started
in `notebooks/`: `jupytext --to notebook --execute` gives **76 cells, 38 code
cells, 0 error outputs**, exit 0, **1 min 52 s**. Interpreter **python.org
3.14.2**, numpy 2.3.5, scipy 1.17.0, matplotlib 3.10.9, h5py 3.16.0,
jupytext 1.19.4. `test/enjl/reference/*.dat` copied into the archive copy by
hand, as ticket 18 had to: `test/` is gitignored so `git archive` does not carry
it. Targeted tests in a **second** isolated copy with a snapshot of `test/`:
`test/enjl` + `test/test_imports.py` = **319 collected, 319 passed, 0 failed**
in 277 s. The full suite was not run (concurrency). The live tree was not used
for either claim — other sessions were committing into `eos/*/verify` and
`notebooks/hybrid_eos.py` throughout.

### Ticket 16's five steps, and which of them ENJL has

Read, then answered one at a time rather than forced into a parallel. The
notebook opens section 9 with this table and the cells below it are the
evidence for each row:

| step, in `quark_eos` | ENJL |
|---|---|
| 1. no pairing: parameters, gap equations, potential, one point | **has it** — and there is no second configuration to contrast it with, because there is only one |
| 2. the same point with pairing on, one pattern at a time | **has nothing here.** No diquark channel in the functional |
| 3. unpaired vs 2SC vs CFL at fixed `(mu_B, T)` | **a different object, the same question**: the branch pair, picked between by the same criterion |
| 4. `Delta(n_B, T)` mapped per pattern | **no gap to map.** `M_u` (chiral) and `chi` (deconfined fraction) are mapped in its place |
| 5. fractions, `c_s^2`, phase boundary in `(mu_B, T)` | **has all three** |

**Step 2 is evidence, not assertion.** The cell lists the parameter dataclass
and the species flags in full and scans them for `delta`, `gap`, `diquark`,
`pair`, `csc`, `2sc`, `cfl`. It finds exactly one name, `deltas`, and that false
positive is printed rather than filtered: it is the Delta(1232) resonance flag
every model carries, which is precisely the confusion the step exists to
prevent. The other half was already on the page — `cfl` is not one of this
model's modes and section 4 prints the refusal by name.

**A branch pair is not a pairing pattern**, stated where it is easiest to lose:
a pairing pattern is a different *ansatz*, a condensate the Lagrangian does not
carry until it is put there, so unpaired/2SC/CFL are three sets of equations;
the two branches are two *roots of one* set. The arithmetic that picks the
winner maps across, the objects it picks between do not.

### Step 3 found a real result, and a trap on the way to it

At fixed `(mu_B, T)` the favoured state is the one of lowest `Omega/V = -P`, so
the higher pressure wins — **a different criterion from section 5's**, which
compares `eps` at fixed `n_B`. Both are right about their own question, and the
two land in **adjacent grid intervals, in every set**:

| set | transition bracket, `P` at fixed `mu_B` | bracket, `eps` at fixed `n_B` (section 5) |
|---|---|---|
| `fq0.5_B1` | 0.300 – 0.350 | 0.350 – 0.400 |
| `fq0.7_B1` | 0.400 – 0.450 | 0.450 – 0.500 |
| `fq1.0_B1` | 0.600 – 0.650 | 0.650 – 0.700 |

The gap between them is the coexistence window: inside it neither pure branch is
the stable state, so asking which has the lower `eps` there compares two
metastable states, while equal `P` at equal `mu_B` and `T` *is* the coexistence
condition. The notebook says this rather than presenting one bracket as the
answer.

**The trap, and it is ticket 18's trap again.** The first draft interpolated
both branches onto a common `mu_B` grid and hunted sign changes, and printed
**twenty-two "Maxwell conditions" for one set**: above the transition the two
continuations have found one root — identical `mu_B` and identical `P` to the
last digit — and every interpolation wiggle became a transition. The fix is the
one section 5 already carries: compare **only at the densities where section 5
found two distinct states**, under its own tolerance. Each set then reports one
crossing.

A second trap under it: **`mu_B` is not monotone along a branch through the
transition** — the swallowtail — so sorting either curve by `mu_B` reorders
physical points. The comparison instead interpolates the `"down"` branch at each
`"up"` row's `mu_B`, and the cell **prints which branches walk backwards**: on
this grid `fq0.5_B1` and `fq0.7_B1` do, `fq1.0_B1` does not. Figure panel (a) is
held to the window where the branches differ, because over the full grid it is
one curve drawn twice.

### Step 4: what stands where the gap would be

`M_u` and `chi` over a 31 x 9 grid in `(n_B, T)`, `T` = 0 to 80 MeV, on the
`"up"` branch, both slices ticket 16 asks for, and the chiral **bracket** per
temperature rather than an interpolated crossing — `M_u` falls by two hundred
MeV between one grid density and the next, and interpolating across a
discontinuity would invent a density the model does not have.

    [T=  0.0 MeV] M_u  165.6 ->   5.5 MeV between n_B = 0.450 and 0.500, mu_B = 1096.6 -> 1086.9
    [T= 80.0 MeV] M_u  199.6 ->   5.5 MeV between n_B = 0.400 and 0.450, mu_B =  973.2 ->  912.5

That bracket read in the `(mu_B, T)` plane is step 5's phase boundary, and it
moves ~180 MeV in `mu_B` over 80 MeV of temperature.

The map takes the **lowest-`f_q`** set, not the last one the older figures use,
and the reason is measured rather than asserted: section 4's own `chi` column
falls 0.19 / 0.09 / 0.02 at the top of the density window as `f_q` rises, so it
is the only set where *both* order parameters move along this branch.

**`c_s^2` is named for the line it is taken along.** `eos_response` refuses for
this model (section 7 prints why), so the curves are finite differences down one
named branch: figure 8.2(b)'s `T = 0` curve keeps its `cs2_adiabatic` label, and
the new `T > 0` curves are **`cs2_isothermal`**, because at finite `T` the
temperature is what is held and the two differ by `C_P/C_V`. Nothing was
renamed — [ticket 69](69-cs2-eq-naming.md) is not this ticket's — and the
notebook has no `cs2_eq`/`cs2_isothermal` reader to carry, since it never calls
`eos_response`.

Section 9.1 also lands two checks the ticket did not ask for and the section
needs: the accepted **scaled residual** (2.3e-15 at the shown point) and the
**Euler relation** of §8, `eps + P` against `T s + sum_i mu_i n_i`, agreeing to
2.2e-16.

### The benchmarks — the shape of tickets 14 and 17, plus one axis

24-point line spanning n_B = 0.05-2.0 fm^-3, three parameter sets (the axis this
notebook sweeps, per ticket 18), and **the branch as a configuration axis**,
which the two sibling notebooks do not have: `direction` changes every number in
a table and changes the cost by more than an order of magnitude, so `"up"` and
`"down"` are separate rows rather than an average.

| set | mode | T | branch | cold ms | warm ms/pt | line s | solved |
|---|---|---:|---|---:|---:|---:|---|
| fq0.5_B1 | beta_eq_neutrinoless | 0 | up | 24.6 | 9.8 | 0.24 | 24/24 |
| fq0.7_B1 | beta_eq_neutrinoless | 0 | up | 53.3* | 12.3 | 0.30 | 24/24 |
| fq1.0_B1 | beta_eq_neutrinoless | 0 | up | 20.4 | 13.9 | 0.33 | 24/24 |
| fq0.5_B1 | beta_eq_neutrinoless | 0 | down | 26.4 | 278.2 | 6.12 | 22/24 |
| fq0.7_B1 | beta_eq_neutrinoless | 0 | down | 54.1* | 142.1 | 2.99 | 21/24 |
| fq1.0_B1 | beta_eq_neutrinoless | 0 | down | 22.3 | 350.7 | 6.66 | 19/24 |
| fq0.5_B1 | fixed_YC | 10 | up | 34.1 | 13.1 | 0.31 | 24/24 |
| fq0.7_B1 | fixed_YC | 10 | up | 28.1 | 13.8 | 0.33 | 24/24 |
| fq1.0_B1 | fixed_YC | 10 | up | 32.2 | 16.4 | 0.40 | 24/24 |
| fq0.5_B1 | fixed_YC | 10 | down | 32.4 | 301.7 | 4.22 | 14/24 |
| fq0.7_B1 | fixed_YC | 10 | down | 25.4 | 216.7 | 4.55 | 21/24 |
| fq1.0_B1 | fixed_YC | 10 | down | 28.7 | 509.1 | 9.67 | 19/24 |

`*` marks a cold point that **did not converge** — timed anyway, because a
sampler pays for a failure too. The `"up"` and `"down"` lines solve the same
equations over the same densities and differ by a factor of twenty in wall
clock; nothing about the solved points got slower, the `"down"` line spends its
clock on the points it never solves, and those attempts are in `elapsed_s` and
not in `n_solved`. That is why `warm` sits above `cold` there by a factor of ten
and why the two are reported apart.

**Cold against warm is a difference in kind here, not only in number**, and
10.1 measures it density by density:

    [fq0.5_B1 ] 0.05:ok 0.10:ok 0.20:ok 0.40:ok 0.60:--  0.80:--  1.00:ok 1.50:ok 2.00:ok
    [fq0.7_B1 ] 0.05:ok 0.10:ok 0.20:ok 0.40:ok 0.60:--  0.80:--  1.00:--  1.50:ok 2.00:ok
    [fq1.0_B1 ] 0.05:ok 0.10:ok 0.20:ok 0.40:ok 0.60:ok 0.80:--  1.00:ok 1.50:ok 2.00:ok

**A band, not a ceiling** — sharper than the docstring's "stop converging around
0.5 fm^-3". The cold starts fail between 0.6 and 1.0 fm^-3, set by set, and
converge again above it where the restored branch is the only root left. The
densities a cold start cannot reach are the ones at which the model has more
than one state and no continuation to say which, which is the same physics as
section 5. The cost column does not split cleanly by outcome either: a `--`
costs what it takes to try every start and give up (60-80 ms), and a converged
point that only the last start reaches costs the same or more (1.1 s at n_B =
2.0 for `fq0.5_B1`) — so the slowest entry in a row can be an `ok`.

### Non-convergence: counted, located, never raised

    [up   lines, both modes, all sets]  0 of 24
    [down lines, beta_eq]               2, 3, 5 of 24 — at the bottom, 0.050 upward
    [down lines, fixed_YC]              3, 5 of 24 at the bottom; 10 of 24 for fq0.5_B1

Every miss is on a `"down"` line and the `"up"` lines miss nothing. Most are the
bottom of the line: a continuation started from the deconfined side has no state
to continue to there — the branch does not become unfavourable, it stops
existing. **One line is different**: `fq0.5_B1` at fixed `Y_C` misses an
*interior* band (0.389-0.983) as well as the bottom two points. Printed, not
diagnosed — a mechanism from here would be a guess, and it is the same shape of
observation ticket 17 recorded for `ccdm`.

### Bottleneck

`fq1.0_B1`, beta equilibrium, T = 0, the `"down"` branch. Cumulative time runs
`least_squares` -> `trf_bounds` -> `approx_derivative` -> the model's
`residual` -> `state_at` -> `thermodynamics`: **84,833 residual evaluations for
24 requested densities**, about 3,500 per attempted point, from 7,689 Jacobian
builds at eleven residuals each — ten unknowns plus the base point. A
**finite-difference Jacobian** is the whole finding, and 10.4 is its direct
consequence.

Two deviations from the sibling notebooks' profile cell, both deliberate and
both stated in the notebook: **20 entries rather than 15** (the first fifteen by
cumulative time are all SciPy's and the model's own frames start below them),
and **a second listing sorted by internal time** (the chain the cumulative list
ends on continues past its cut; the internal-time list is what names the leaf).
The second listing is also what corrected a claim in the first draft: `state_at`
is the largest internal-time entry *belonging to this repository*, not the
largest outright — SciPy's own trust-region bookkeeping sits above it.

No Numba hazard here: `eos/enjl` ships no jitted kernel, so a cold profile does
not report a compiler the way a `T = 0` hadronic line does.

### Backends (§9): nothing to compare, and it is checked

    [enjl] backends/ absent   backend switch on eos_point: none
    [enjl] backends/ absent   backend switch on eos_table: none

Executed output, not prose. Every number above is therefore a reference number
needing no second column — and it is the direct cause of the ~3,500 residual
evaluations per point. The nearest thing this model ships to a second flavour is
the **branch pair**, and the cell says why that is not one: `"up"` and `"down"`
are two different states, not two implementations of one calculation, and their
timings differ because they solve different physics.

### The `table_path` root

`root=str(ROOT / "output" / "tables")` — section 3's `TABLE_ROOT`, reused rather
than redeclared. Confirmed by the archive run with the kernel in `notebooks/`:
tables and figures landed at the repository root and `notebooks/output/` does
not exist in the copy. The benchmark file is

    output/tables/enjl/enjl_benchmark_T0.0-10.0x4_nB0.1-2.0x24_hyp+mu.h5

with the model slot carrying the study, since the table spans three parameter
sets and four configurations. The `nB0.1` for a grid starting at 0.05 is
`standard_name`'s `%.1f`, the cosmetic collision hazard
[ticket 17](17-quark-benchmark.md) already recorded; not fixed here.

### Two things found, neither fixed here

* **`eos/enjl/api.py:106` names `eos.enjl.solver.UNKNOWNS`, which does not
  exist.** The module has `BASE_UNKNOWNS` and `unknown_slots(spec)`. It is the
  documented contract for `x0`'s ordering, so the docstring points a caller at a
  name they cannot import. Found while deciding how to label the unknown vector
  in 9.1 — the notebook prints the ten unknowns by name instead of importing
  anything. Belongs with [ticket 54](54-signature-corrections.md).
* **`fq0.5_B1` at fixed `Y_C`, `"down"` branch, misses an interior density
  band.** Above.

### Scope

`notebooks/enjl_eos.py` + `.ipynb` and this file. `notebooks/quark_eos.py` and
`notebooks/hybrid_eos.py` were read and not touched; ticket 16's own
step-by-step section for `njl`/`ccdm` remains unstarted.

Status: resolved.
