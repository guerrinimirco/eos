# The six non-baseline failures on 3.14: four re-measurements ticket 57 named

Type: task
Status: resolved
Assignee: session 7cc34465
Blocked by: 62
Parent: ../map.md

## Question

[Ticket 62](62-regenerate-baselines-py314.md) executed the `.npz` half of
[ticket 57](57-canonical-stack.md)'s ruling and took the suite from 12 failures
to 7 on python.org 3.14.2. One of the survivors is `test_baseline[enjl]`, which
is [ticket 72](72-enjl-branch-selection.md) and red on purpose. **The other six
are the rest of ticket 57's cost list, and none of them is a golden reference
file** — they are tolerances and test premises that must be re-derived on the
canonical stack:

    FAILED test/dd2/test_api.py::test_inversion_without_Q_sat_predicts_it
    FAILED test/dd2/test_api.py::test_inversion_with_Q_sat_still_imposes_it
    FAILED test/dd2/test_dd2_m8.py::test_restarts_recover_a_seed_limited_inversion
    FAILED test/tov/test_solver_fast_robustness.py::test_attached_crust_leaves_pressure_monotone
    FAILED test/tov/test_solver_fast_robustness.py::test_crusted_hybrid_star_agrees_with_the_reference[0.3]
    FAILED test/tov/test_solver_fast_robustness.py::test_crusted_hybrid_star_agrees_with_the_reference[0.6]

Ticket 57 listed exactly these, quoted verbatim:

1. **`test_api.py:127` and `:143`'s `abs=0.2` on Q_sat** are re-derived from a
   noise floor MEASURED on 3.14, not loosened to fit — ticket 47 Q3, and
   `nmp.py:85`'s requirement that `h` move in the forward and inverse maps
   together. §12 forbids loosening a tolerance to make a test pass, so this is
   a measurement task before it is an edit.
2. **`test_dd2_m8.py::test_restarts_recover_a_seed_limited_inversion`'s
   `(K_sat, Q_sat) = (240, 300)` premise** is re-measured: the seed-limited /
   infeasible distinction it guards is exactly what changed between scipy 1.13
   and 1.17.
3. **`test/tov/test_solver_fast_robustness.py`'s three cases** get a sample the
   6x6 closure can actually reach on 3.14 — today it returns `None` at
   isoscalar residual 8.12e-02.
4. **The DD2 published NMP/TOV values and the CompOSE HS(DD2) slices are
   re-checked**, since those are §12 ground truth independent of any `.npz` and
   nothing in ticket 62 touched them.

Item 4 is the one with no failing test attached, which is why it is the easiest
to skip and the most load-bearing: `dd2`'s `nmp.Q_sat` moved 0.351 MeV between
the stacks, and whether the published values still verify on the canonical
stack is a separate question from whether the frozen `.npz` reproduces.

### Why this is not ticket 62's diff

Ticket 62's stop condition was about `.npz` files. These six are tests, and
three of the four items require deriving a number (a noise floor, a reachable
sample) before anything is edited — which is measurement work, not a
regeneration. Ticket 57's "green on 3.14" is met when this ticket and
[ticket 72](72-enjl-branch-selection.md) are both closed, not before.

### Prerequisite

Needs a quiet tree for the noise-floor measurement in item 1: an h-sweep whose
answer is a tolerance cannot be taken while another session is editing
`eos/dd2/`.

## Resolution

**All six fixed, no tolerance loosened to fit, and none of them was a
regression.** The six reduce to ONE function — `eos/dd2/nmp.py::invert_nmp` —
and to two distinct defects in the tests, not in the library. **`eos/` is
untouched by this ticket**: the entire diff is under `test/`.

### The measurement that reframed the ticket

Ticket 57 and this ticket's item 1 both describe `test_api.py:127`/`:143` as
tolerances asserting below a noise floor. **Only `:127`'s sibling is a
tolerance question at all; both assertions are false PREMISES**, and the
distinction matters because widening either one would have pinned a number
produced by a solve that never ran.

**`:127`, the 5x5 closure.** Ticket 47 Q1 already ruled that the default 5x5
closure cannot return DD2's own basin: the published couplings miss the
closure's own cross-constraint `f''_sigma(1) = f''_omega(1)` by 2.200718e-03,
so the published point is a stationary point of the residual norm and not a
zero of it. Confirmed here on the canonical stack — the 5x5 converges to
isoscalar residual **6.686e-11** and predicts Q_sat = **117.494** against the
forward map's 169.003, with the six imposed NMPs round-tripping to 3.41e-05.
The test asserted `predictions["Q_sat"] == approx(nmp["Q_sat"], abs=0.2)`,
which passed on 3.9 only because scipy 1.13's `hybr` stalled and handed the
seed back. It is a solver artifact encoded as a requirement, exactly as ticket
47 said.

**`:143`, the 6x6 closure — and this one is NOT a stack artifact.** At DD2's
own NMPs the 6x6 does not converge on EITHER stack:

    scipy 1.17.0   nfev=25  status=5  max|res| 2.201e-03  moved 0.00e+00
    scipy 1.13.1   nfev=23  status=5  max|res| 2.201e-03  moved 0.00e+00

`hybr` returns the seed **bit-identically** — all six couplings unchanged to
the last bit — reporting "The iteration is not making good progress". 48
jittered restarts never beat 2.201e-03, the published cross-constraint
violation. The cause is measured, and it is the stencil: probing each unknown
by `hybr`'s own 1.49e-08 relative step,

    row            typical |dR|      base |R|
    P              2.563e-07         1.735e-04
    E_sat          5.684e-13         2.458e-05
    m_ratio        0.000e+00         7.813e-06
    K_sat*1e-2     1.137e-07         1.871e-04
    Q_sat*1e-2     7.119e-04         1.475e-03     <- half the row is noise
    cross          5.022e-09         2.201e-03

The `Q_sat` row's Jacobian column is ~50% finite-difference noise, so hybr
cannot descend. The 5x5 omits that row (`want_Q=False`) and converges to
6.7e-11 — which is why the two closures behave so differently. **The test
passed on 3.9 by coincidence**: there the forward map returned 168.65 and the
reconstruction path also returned 168.51, so two unrelated numbers happened to
agree within 0.2.

### The noise floor, measured on the canonical stack

`h` swept in the forward AND inverse maps together, per `nmp.py:85`:

    h        Q_sat      | h        Q_sat
    5e-5     166.2932   | 4e-4     168.7705
    8e-5     168.7020   | 5e-4     168.7619
    1e-4     169.0034   | 7e-4     168.7423
    1.5e-4   168.8047   | 1e-3     168.6947
    2e-4     168.7631   | 1.5e-3   168.5795
    3e-4     168.7828   | 2e-3     168.4180

    plateau h in [2e-4, 1e-3]:  mean 168.7525   spread 0.0881
    shipped h = 1e-4:           169.0034        +0.2508 off the plateau mean
    full band 5e-5..2e-3:                       spread 2.7102

So the shipped `h = 1e-4` carries **0.25 MeV** of stencil excursion, two and a
half times the ~0.1 MeV `nmp.py`'s docstring estimates, and two independent
evaluations of Q_sat differ by up to ~0.5 MeV. **`abs=0.2` asserted below that
floor.** A second, h-STABLE effect sits underneath it: the published table
stores `n_sat`, `a_sigma`, `d_sigma`, `a_omega`, `d_omega` to six decimals
while `from_microscopic` re-derives them, and that reconstruction offset is
**-0.207 MeV**, flat to three digits across h in [3e-4, 2e-3]. It is a real
difference between two parametrizations, not noise, and no tolerance should
have been hiding it.

**Independent corroboration that 0.5 is the honest scale**: `test_dd2_m1.py:73`
already pins Q_sat against the published 168.6 at `abs=0.5` and passes on the
canonical stack at 169.003 (diff 0.403). That tolerance was written before any
of this and survived the stack move; `test_api.py`'s `abs=0.2` was the outlier
of the repository's own two tolerances on the same quantity.

### The re-measured sample, one target serving items 2 and 3

Scanning the (K_sat, Q_sat) plane on 3.14 for a target that is SEED-LIMITED —
missing the gate from the published seed, converging from a jittered one —
and cross-checking every candidate on 3.9:

    target        3.14 r=0 -> r=32              3.9 r=0 -> r=32
    (240, 300)    6.1e-01 -> 5.78e-02  MISS     1.3e+00 -> 1.82e-04  ok
    (220, 300)    6.1e-01 -> 6.82e-04  ok       1.3e+00 -> 6.02e-03  ok
    (230, 300)    8.2e-01 -> 4.93e-03  ok       1.3e+00 -> 2.73e-02  MISS
    (220, 200)    3.1e-01 -> 2.60e-03  ok       3.1e-01 -> 2.26e-02  MISS
    (250, 100)    4.5e-01 -> 8.12e-02  MISS     1.6e-01 -> 8.50e-03  ok

**`(K_sat, Q_sat) = (220, 300)` is the only candidate seed-limited on BOTH
stacks**, dropping x895 on 3.14 (30x inside ISO_GATE) and x219 on 3.9. It
serves item 2 and item 3 at once, and it takes both tests OFF the stack
dependence rather than moving them onto the other side of it. The RNG stream
was verified identical across numpy 1.26.4 and 2.3.5, so the same 32 jittered
seeds are tried on both stacks and the difference is genuinely convergence,
not seed luck.

For the tov sample, every premise the module rests on was re-verified with the
new NMPs: `build_parametrization` returns `ok`, the mixed table still
`has_transition`, and **the core still undercuts the BPS crust at the join** —
0.308 against 0.406 MeV/fm^3 at n_B = 0.080 fm^-3, where the old sample gave
0.225. The docstring's quoted number was updated with it.

### Item 4 — the ground truth with no failing test attached

Discharged on the canonical stack, and unmoved:

    eos/dd2/verify/run_full_check.py     PASS
      golden points      1.40e-05   SNM(0.16)
      CompOSE HS(DD2)    2.83e-05   nucleonic T=1 Yq=0.5
      backend parity     4.40e-14
    test/dd2/test_dd2_m1.py  test_nmp_reproduction     PASS (Q_sat at abs=0.5)
    test/dd2/test_dd2_m4_tov.py  published NS point    PASS (M_max ~ 2.42,
                                                             R_1.4 ~ 13.2 km)
    test/dd2/test_dd2_m10.py     M_max >= 2 gate       PASS

The golden and CompOSE errors are the SAME numbers ticket 67 recorded from the
3.9 run (1.40e-05 / 2.83e-05), so the §12 ground truth independent of any
`.npz` does not move between the stacks. `dd2`'s `nmp.Q_sat` moving 0.351 MeV
is confined to the finite-difference map, exactly as ticket 62 concluded from
the other 4689 keys being bit-identical.

### What changed

Four tests, all under `test/`. **No file under `eos/` was edited, no `.npz`
regenerated, no golden reference touched, and `h` was NOT moved** — moving it
requires the forward and inverse maps together (`nmp.py:85`) and therefore a
`dd2.npz` re-freeze, which this ticket does not authorise and whose vehicle
has departed (see ticket 67).

- `test/dd2/test_api.py::test_inversion_without_Q_sat_predicts_it` — asserts
  the closure's SELF-consistency (`status.predictions` against a forward map of
  the couplings it returned) instead of agreement with DD2's own Q_sat.
  `abs=0.5` on Q_sat, derived from the 0.25 MeV stencil excursion measured
  above; `abs=1e-2` on K_sym, a second difference measured at 7.9e-07 (3.14)
  and 5.6e-04 (3.9).
- `test/dd2/test_api.py::test_inversion_with_Q_sat_still_imposes_it` — moved to
  (220, 300) where the 6x6 actually converges, and **now asserts the solve left
  the seed**, which is the guard that was missing and would have caught the
  stall. Tolerance is `ISO_GATE / 1e-2` = 2 MeV: the gate's OWN budget, since
  Q_sat enters the residual scaled by 1e-2. Measured error 0.353 (3.14), 0.539
  (3.9) — a quarter of the budget.
- `test/dd2/test_dd2_m8.py::test_restarts_recover_a_seed_limited_inversion` —
  target (240, 300) -> (220, 300), and a second assertion that the residual
  falls by at least x100. That is ticket 67's point 3 discharged: the verdict
  is now the orders-of-magnitude drop, not the knife-edge `ok`.
- `test/tov/test_solver_fast_robustness.py` — the soft Delta-rich sample's
  (K_sat, Q_sat) 250/100 -> 220/300, in both places it is written, with the
  docstring's crust-join pressure updated 0.225 -> 0.308 to match.

### Reported, not fixed

**`invert_nmp` reports `ok=True` on a solve that never ran.** `ISO_GATE = 2e-2`
admits the 2.201e-03 stall, so at DD2's own NMPs the 6x6 returns the published
seed with a success status. Ticket 47 noted the gate "cannot by itself
distinguish a converged solve from one that never moved"; this ticket measured
what that costs — a caller asking for Q_sat = 169.0 is handed couplings whose
Q_sat is 168.65, with no indication the imposition did not happen. Fixing it
means changing `invert_nmp`'s contract (a convergence flag distinct from the
gate, or a seed-displacement check), which is library code this ticket does not
ask for. The new assertion in `test_inversion_with_Q_sat_still_imposes_it` is
the guard at the test level; the API-level fix is for the Stage 7 report.

Second, smaller: `ISO_GATE = 2e-2` on a row scaled by 1e-2 means the closure's
advertised Q_sat can be wrong by 2 MeV while reporting success — a budget an
order of magnitude above the 0.25 MeV stencil floor. Whether the gate should be
per-row rather than a max-norm is the same question ticket 67 point 4 asks from
the baseline side.

### For [ticket 67](67-dd2-t0-adoption.md), which is unblocked by this

The honest floor it was waiting for: **Q_sat at the shipped h = 1e-4 carries
0.25 MeV of stencil excursion, and two independent evaluations differ by up to
~0.5 MeV.** Ticket 67's T = 0 adoption moves it by 0.061 MeV — **a quarter of
the floor, and an eighth of the two-evaluation spread.** The stack change moved
it by 0.351 MeV, also inside the two-evaluation spread. So neither shift is
distinguishable from stencil noise, and 67's question 4 ("does `nmp.Q_sat`
belong in a frozen rtol = 1e-10 baseline at all?") now has a measured answer
from three independent witnesses rather than two: no.

Its point 3 is discharged rather than merely unblocked — `test_dd2_m8` no longer
asserts a knife-edge `ok`, and its target is seed-limited on both stacks.

### Suite

Both runs python.org 3.14.2, collected counts included per the map's rule, and
compared against [ticket 62](62-regenerate-baselines-py314.md)'s after-image:

    62's after  output/_audit/pytest_after_ticket62_py314.txt
                7 failed, 1674 passed, 15 skipped   (1696 collected, 21:05)
    before      output/_audit/pytest_before_ticket74_py314.txt
                6 failed, 8 passed                  (the six node ids only)
    after       output/_audit/pytest_after_ticket74_py314.txt
                1 failed, 1680 passed, 15 skipped   (1696 collected, 20:19)

**Six cleared, 0 added, and the denominator is unchanged at 1696** — the same
collected count as ticket 62's after-image, so this is a clean comparison
rather than 62's soft one. The one survivor is `test_baseline[enjl]`, red on
purpose and owned by [ticket 72](72-enjl-branch-selection.md).

The six were also re-run on the anaconda 3.9.7 discriminator stack: **14 passed**.
They were 3.9-only before this ticket and are now green on both, which is a
stronger result than "green on the canonical stack" and the reason the sample
was chosen for two-stack seed-limitation rather than for 3.14 alone.

Measurement transcripts, both stacks, in
`output/_audit/nmp_noise_floor_ticket74_py314.txt`.

### The caveat this ticket cannot fix

**Every file this ticket changed is gitignored.** `.gitignore:75` ignores
`/test/` and `.gitignore:37` ignores `output/`, so the four test edits and both
audit transcripts live outside version control and the commit carries only this
ticket and the map. That is the hazard the map already records under "Several
real fixes now live outside version control", and it is the same one ticket 47
hit when it could not walk `test/` across history. The re-derived tolerances
here are exactly the kind of measured, hard-to-reconstruct number that hazard
loses. Flagged for the Stage 7 report; changing what is tracked is not this
ticket's to do.
