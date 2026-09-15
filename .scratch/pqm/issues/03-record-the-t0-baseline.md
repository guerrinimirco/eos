# Re-measure the T = 0 baseline and put it in a FILE

Type: task
Status: open
Blocked by: 01, 02
Parent: ../map.md

## Question

**There is no accuracy number in this repository for the grid the prototype
currently ships.** That is not a suspicion, it is checkable three ways:

1. `docs/csc_bag_mapping.md:601-611` gives the headline -- max 0.2%, median
   0.1% in P, 92-96% phase agreement, "180-320x" -- and states the slice it was
   taken on: **`n_B = 1.55-3.20 fm^-3`**. That is ~10-20 n_sat, from the OLD
   grid `FIT_MU_B = linspace(1500, 2600, 10)`.
2. The notebook has since abandoned that window, and says why in its own
   comment (`notebooks/csc_bag_map.py:261-262`): *"mu_B = 1500-2600 MeV is
   5-38 n_sat: almost entirely ABOVE the region of interest, and anchoring the
   coefficients there is how a fit comes out excellent and useless."* It now
   ships `GRID_MU_B = linspace(900, 1800, 28)`.
3. The paired `.ipynb` has **every output cleared** (`execution_count: None` on
   all 20 cells), so the re-measurement, if it was run, left no record.

Numbers for the current grid exist only in a session's memory. That is exactly
the failure mode CLAUDE.md section 12 legislates against for the test suite --
a number nobody can cite is not a number -- and this map's whole "Where this
starts" table currently rests on it.

**Also unimplemented: three of the five acceptance criteria the docs
themselves set** (`docs/csc_bag_mapping.md:454-476`):

- **Criterion 2, the sound speed.** `sound_speed()` exists
  (`csc_bag_map.py:986-989`, `np.gradient(P, eps)`) and `cs_m - cs_t` is
  PLOTTED (`:1163-1165`), but **no tolerance is ever computed or asserted** --
  despite the docs calling it *"the derivative test and it is much sharper than
  the pressure test; a fit can match P to a part in 1e3 and get c_s^2 wrong by
  a factor."* After ticket 01 the model's `c_s^2` is available analytically
  rather than from `np.gradient`, which makes this sharper still.
- **Criterion 4, branch ordering.** A transition density reproduced to a few
  percent. Not implemented -- and it is milestone 2's whole subject, so this
  ticket only needs to RECORD the current phase-agreement percentages, not fix
  them.
- **Criterion 5, M_max and R_1.4 through `eos.astro.tov`.** Not implemented;
  deferred to ticket 13.

### What to do

Re-run the benchmark on the shipped grid, with the ticket-01 closed-form
derivatives and the ticket-02 term set, and **write the table into
`docs/csc_bag_mapping.md`** -- a new dated appendix, not a replacement of
Appendix A/B, which record the earlier passes and should stay as history.

Four modes x four gluon-exchange NJL parameter sets, the error normalized by
`max|P|` over the window (a pointwise relative error diverges where P crosses
zero -- `csc_bag_map.py:1084-1090`), weighting 3-10 n_sat heaviest:

| slice | report |
|---|---|
| beta equilibrium, T = 0 | median and max `\|dP\|/max\|P\|`; **`c_s^2` max relative deviation**; phase agreement % |
| `fixed_YC` with leptons, T = 0 | the same three |
| `Y_C = 0.5, Y_S = 0`, no leptons, T = 0 | the same three |
| `Y_C = 0, Y_S = 0`, no leptons, T = 0 | the same three |

plus, per pattern, the fit rms in `P`, `n_B`, `n_C`, `n_S` and the fitted
coefficient vector.

**Re-pin the speed claim while here**, because the current one is not citable:
the 175 ms `eos.njl` baseline behind "180-320x" has no recorded provenance and
is inconsistent with every pinned timing in
`.scratch/njl-speed/issues/02-pin-the-benchmark.md:84-91` (2SC alone 56.5
ms/pt, CFL alone 719.3, the default four-pattern enumeration 8100.2). Worse,
`njl_reference` is DISK-CACHED (`csc_bag_map.py:964-983`), so on any re-run its
timer reads a pickle, not a solve.

Report instead, separately:

- **the bare `pressure()` evaluation**, microseconds, which is what a mixed
  adapter's inner loop pays;
- **one full equilibrium solve** in each mode, which includes `scan_root`'s
  121-point scan plus `brentq`, or the 2-D `root` for `fixed_YC_YS`;
- **the `eos.njl` comparator, named by configuration** -- which patterns,
  which backend, fixed-`n_B` or fixed-mu -- so the ratio says what it is a
  ratio of.

n = 3, median, cpu printed beside wall, interpreter with its numpy and scipy
versions named. **Never wrap a timing in `timeout`** -- it is an x86_64 binary
on this machine and drags the interpreter under Rosetta, breaking numpy with
an error naming a cause it does not have. And nothing is timed on battery: the
duty cycle drops to ~5%.

## Gate

- The four-slice table above, filled in, **written into
  `docs/csc_bag_mapping.md`** as a dated appendix, with the interpreter, numpy
  and scipy versions named.
- A `c_s^2` number per slice, computed and compared against the 10% criterion
  -- pass or fail, but stated. This is the first time it exists.
- The speed claim restated as three separate numbers with their configurations
  named, replacing the uncitable "180-320x".
- The map's "Where this starts" table updated to cite this appendix instead of
  a memory.
- If any slice is now WORSE than the remembered numbers, that is reported, not
  smoothed -- tickets 01 and 02 changed the basis and a regression is
  information.
