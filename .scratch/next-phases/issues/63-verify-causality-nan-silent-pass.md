# Five verify causality checks absorb a NaN and report PASS

Type: task
Status: open
Blocked by: 49
Parent: ../map.md

## Question

Introduced by [ticket 49](49-nonconvergence-return.md), and named in its
resolution rather than fixed there.

Five `verify/run_full_check.py` causality checks sweep a density grid and
accumulate the worst violation as

    worst = max(worst, max(0.0 - cs2, cs2 - 1.0, 0.0))

    eos/zl/verify/run_full_check.py:245
    eos/vmit/verify/run_full_check.py:223
    eos/alphabag/verify/run_full_check.py:403
    eos/did/verify/run_full_check.py:387
    eos/njl/verify/run_full_check.py:563

Before ticket 49 a grid point the solver could not reach RAISED out of
`eos_response`, and the check died loudly. It now returns `cs2 = nan`, and NaN
loses every comparison, so `max` propagates the incumbent and the point is
absorbed: the check reports PASS over a grid it did not actually evaluate.

That is section 4's "nothing is ever silently skipped" and section 8's "the
check runs before integration, returning a status rather than a meaningless
mass", both violated in the same line. It is the price of the section 6 fix
and it has to be paid somewhere; the right place is the check, which is what
knows a NaN in c_s^2 is not a passing value.

**Nothing is masked today.** All nine verify suites pass on the current
parameters (measured in ticket 49), which is itself the proof that no grid
point currently fails to converge. This is a hole in the safety net, not a
live wrong answer — which is why it is a ticket and not a stop.

The obvious shape is one `np.isfinite` assertion per site, failing the check
with the density that produced the NaN. Whether the five want a shared helper
in `general/` (they are five copies of one loop) or five local guards is the
call to make; `min(values)` / `max(values)` in the same functions' messages
have the same NaN exposure and should be looked at in the same pass.

Resolved when a deliberately non-convergent grid point makes each of the five
checks FAIL naming the density, and all nine verify suites still pass on the
published parameters.
