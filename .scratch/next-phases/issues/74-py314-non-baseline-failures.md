# The six non-baseline failures on 3.14: four re-measurements ticket 57 named

Type: task
Status: open
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
