# Five verify causality checks absorb a NaN and report PASS

Type: task
Status: resolved
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

## Resolution

**Five local guards, not a shared helper** — the call the ticket left open.
Every verify suite in this repository declares its **own** `CheckResult`
(ten of them do), because a suite is a standalone script a physicist runs and
reads on its own; a shared causality helper in `general/` would be the first
thing to break that, and `general/` has no `verify/` home to put it in yet
([ticket 64](64-general-verify-suite-missing.md)). The guard is four lines per
site and reads in place.

**The ticket's premise was right about three of the five and wrong about two,
and the line numbers had drifted.** Measured before touching anything, by
wrapping each suite's `eos_response` and poisoning the SECOND grid point with
a nan:

    zl        passed=True   c_s^2 in [0.464, 0.935]        <- absorbed
    vmit      passed=True   c_s^2 in [0.407, 0.490]        <- absorbed
    alphabag  passed=True   c_s^2 in [0.311, 0.322]        <- absorbed
    did       passed=False  max c_s^2 = nan at n_B = 0.25
    njl       passed=False  c_s^2 = 0.2359, nan at n_B = 1.5, 2.0

`zl`, `vmit` and `alphabag` carry the `worst = max(worst, max(0.0 - cs2,
cs2 - 1.0, 0.0))` line the ticket quotes and absorb the nan exactly as
described — note that their MESSAGE hides it too: `min(values)`/`max(values)`
returned a clean-looking range over a list containing a nan, so the report
positively asserted a range it had not measured. `did` (`check_causality`,
now line 391) and `njl` (`check_sound_speed`, now 564) do not carry that line
at all; they fail already, but only by accident of nan losing every
comparison, and they report the nan rather than the density. All five are the
same defect at bottom — the check has no opinion about a non-finite value —
and all five now say so.

**After**, same injection:

    zl        passed=False  c_s^2 is not finite at n_B = 0.800 fm^-3: ...
    vmit      passed=False  c_s^2 is not finite at n_B = 0.800 fm^-3: ...
    alphabag  passed=False  c_s^2 is not finite at n_B = 0.800 fm^-3: ...
    did       passed=False  c_s^2 is not finite at n_B = 0.25 fm^-3: ...
    njl       passed=False  c_s^2 is not finite at n_B = 2.0 fm^-3: ...

each with `max_error = inf` and the sentence "the response did not converge
there, so this grid was not evaluated".

**The `min`/`max` message exposure is closed by construction, not by a second
guard.** The three loop guards return BEFORE appending, so `values` can never
hold a non-finite entry and the range in the message is always over what was
actually evaluated; `did`'s array guard runs before `cs2.max()` and
`cs2.argmax()`, and `njl`'s before `max(values)`. Each docstring says so, so
the next reader does not re-derive it.

**The sweep is complete for this class.** Only these five checks consume a
number out of `eos_response` in any `verify/` suite. The two other call sites
— `abpr/verify:294` and `enjl/verify:757` — are refusal checks that assert a
RAISE and never read a value.

### Gate

python.org **3.14.2** (`python3`). **All eleven `verify/` suites pass on the
published parameters**, every check `[ok ]`, zero non-ok lines: dd2 6, sfho 8,
zl 10, did 13, vmit 8, alphabag 10, abpr 9, njl 15, ccdm 16, enjl 17,
mixed 9 — 121 checks. (The ticket says nine; there are eleven now, and all of
them were run.)

**0 added failures**, isolated PAIR from `git archive HEAD` (`71239a4`) plus a
snapshot of the gitignored `test/`, collection **318** over `test/zl test/vmit
test/alphabag test/did test/njl test/baseline`:

    control (HEAD)          6 failed, 312 passed   2:55
    mine    (HEAD + t63)    6 failed, 312 passed   2:43

Identical failure sets — the same six pre-existing `test/baseline` failures
(`ccdm`, `dd2`, `enjl`, `njl`, `tov`, `zlvmit`). `test/baseline/` cannot move
here in any case: the diff is confined to `verify/` suites, which no model
imports.
