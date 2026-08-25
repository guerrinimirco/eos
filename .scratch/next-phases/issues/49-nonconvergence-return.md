# Non-convergence escapes as an exception at seven public boundaries

Type: task
Status: resolved
Assignee: mirco (session)
Blocked by: 11
Parent: ../map.md

## Question

CLAUDE.md §6 calls this non-negotiable: "**NON-CONVERGENCE IS A RETURN VALUE** at
every public boundary, not an exception and never a hang." Seven live sites break
it, all inside the three §5 entry points.

**The `SnB=` path (finding 18).** `eos/general/tabulate.py:102` raises when an
entropy target is unreachable. `zl/api.py:113`, `vmit/api.py:113`,
`alphabag/api.py:119`, `mixed/api.py:160` and `dd2/api.py:97` all call
`temperature_at_entropy` **inside** their `try`; `eos/njl/api.py:109` and
`eos/ccdm/api.py:122` call it bare. So an unreachable isentrope escapes
`eos_point` as a `RuntimeError`. Two-line fix each.

**`eos_response` (finding 19).**

    eos/zl/api.py:199          raise RuntimeError("eos_response could not solve its stencil point ...")
    eos/vmit/api.py:199        (same)
    eos/alphabag/api.py:207    (same)
    eos/mixed/api.py:345       central point
    eos/mixed/api.py:374       stencil points
    eos/abpr/api.py:234        raise RuntimeError(f"eos_response could not invert n_B = {n_B}")

`abpr` is the starkest: `mu_from_nB` returns `(mu, converged)` and the status is
converted straight into an exception. `sfho`, `dd2`, `did`, `njl` and `ccdm`
already return dicts, so the compliant shape exists in-repo to copy.

**The return shape, ruled by [ticket 11](11-conformance-triage.md):** the SAME
dict shape the converged path returns, with `converged=False` and **NaN in every
quantity** — not a minimal `{converged: False, reason: ...}`. A caller writing
`result["cs2_adiabatic"]` into an array column must not need a second code path
for the failure case, and NaN propagates to a plot honestly. Carry a `reason`
string alongside; do not drop the message the exception used to carry.

Resolved when all seven return rather than raise, a check exists that a
deliberately unconvergeable point comes back with `converged=False`, and the
added-failure count against `output/_audit/pytest_before_with_crust.txt` is
reported. **No converged number may move** — this touches failure paths only.

---

## Resolution

Done. Every named site returns rather than raises, and the shared failure dict
is one function in `general/`. Commit and evidence below.

**The ticket named seven sites; there are eleven.** Four models it lists as
already compliant are not, and the count is the main finding here.

### The SnB half is two sites, not seven

The ticket's prose reads as if all seven `temperature_at_entropy` callers leak.
They do not. `zl:113`, `vmit:113`, `alphabag:119`, `mixed:160` and `dd2:97` call
it inside a `try` that already catches `(RuntimeError, ValueError)` and returns
`PointResult(False, str(err))` — checked in each file, not inferred. Only
`njl:109` and `ccdm:122` called it bare, and those two are fixed. The ticket's
own sentence "call it **inside** their `try`" was right; the sentence after it
generalised past its evidence.

### The four "already compliant" models were the bigger half

The ticket says `sfho`, `dd2`, `did`, `njl` and `ccdm` "already return dicts, so
the compliant shape exists in-repo to copy". They return dicts on the CONVERGED
path only. `sfho`, `did`, `njl` and `ccdm` all let a `RuntimeError` straight out
of `eos_response` when a stencil neighbour fails, from

    eos/sfho/responses.py:40    eos/njl/responses.py:48
    eos/did/responses.py:35     eos/ccdm/responses.py:52

and each of those four docstrings SAYS the api layer catches it — "An internal
layer may raise (CLAUDE.md section 6); `api.eos_response` [...]". It did not.
Confirmed live rather than by reading:

    did.eos_response(par, "beta_eq_neutrinoless", flags, n_B=1e-8, T=0.0)
      -> RuntimeError: the response stencil needs a converged neighbour and
         the solve at n_B=1.001e-08 fm^-3, T=0

Fixed all four. This is the ponytail root-cause call, not scope creep: leaving
them would have made the shape NEWLY inconsistent — `zl.eos_response` returning
where `sfho.eos_response` raises is worse than both raising — and section 5's
uniform API is the thing the ticket is enforcing.

A real physical case now returns instead of raising: `ccdm.eos_response` below
the deconfinement onset (n_B = 0.3, 0.5, 0.8 fm^-3) has no root at fixed
density and comes back `converged=False` with the reason, where it used to
throw. That is exactly the sampler case section 6 is written for.

### dd2 is unverified, by the session constraint

`eos/dd2/api.py:eos_response` was off-limits. It takes the analytic-Jacobian
route (`backends/responses_jac.py`) rather than a stencil, so it has no
equivalent of the raise fixed in the other nine, and it was NOT one of the
ticket's seven. Whether that Jacobian path can raise on non-convergence is
unchecked. Flagged, not reached across.

### The shape

`eos/general/tabulate.py` gains `unconverged_response(reason, quantities)` —
NaN in every named quantity, plus `converged=False` and `reason`. It sits beside
`accepted()`, which is the other status helper every model already imports from
there, so this is not a new home. The converged path in every touched model now
sets `converged=True` and `reason="converged"`, because "the SAME dict shape"
is only true if the status key exists on both paths.

Two places carry a deliberate non-NaN:

- `mixed`, stencil failure: `chi` and `phase` come from the CENTRE point, which
  converged, so they are kept. Only the derivatives are NaN.
- `mixed`, `phase` on a central-point failure: `None`, not NaN. It is a label
  ('H' / 'mix' / 'Q'), and which regime the point is in is precisely what did
  not get determined.

And one place where `converged=True` sits beside a NaN, which is not a
contradiction and is documented in the docstring: outside the coexistence window
`mixed.eos_response` returns `cs2_eq = nan` because the mixed root there is an
analytic continuation, not the state. That NaN is physics; the section 6 NaN is
a status. The `reason` string distinguishes them.

### Evidence

**Interpreter**: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3`
— 3.14.2, numpy 2.3.5, scipy 1.17.0. Ticket 57's ruling, and `python`, not
`python3`, is the anaconda 3.9.7 that ruling rejects.

**Targeted pytest, per suite** (the full suite was NOT run — another session is
live on dd2 and `test/dd2/test_dd2_speed.py` goes flaky under contention):

    test/zl         35 passed      test/ccdm       52 passed
    test/vmit       38 passed      test/sfho       39 passed
    test/alphabag   61 passed      test/did        83 passed
    test/abpr      248 passed      test/general   128 passed
    test/njl        82 passed      test/mixed     277 passed
    test/test_nonconvergence_return.py             12 passed

**1055 collected, 0 failed.**

**Added-failure count against `output/_audit/pytest_before_with_crust.txt`: 0
for the suites run**, and the comparison is clean rather than approximate: all
12 known 3.14 failures in the most recent full image
(`pytest_after_ticket48_py314.txt`) sit in `test/baseline`, `test/dd2` and
`test/tov` — the intersection with the eleven suites above is EMPTY. So a green
targeted run is a zero-added-failure claim for everything this ticket touched,
and says nothing either way about dd2/tov/baseline.

**`test/baseline/` is untouched, by construction.** `test_baseline` asserts NO
ADDED KEYS ("`{model}: {n} quantities are new`"), so adding `converged` and
`reason` would trip it if a baseline case called `eos_response`. Grepped
`generate_baseline.py`: it contains no `eos_response` and no `SnB`. Neither
changed path is on its route. `test/baseline/` was never written to.

**verify PASS, nine of nine**: zl, vmit, alphabag, abpr, njl, ccdm, sfho, did,
mixed. The tenth, `eos.general.verify.run_full_check`, DOES NOT EXIST — see
below.

### One measurement was retaken, on the committed tree

The other session committed `eos/dd2/species.py`, `eos/dd2/thermodynamics.py`,
`eos/dd2/solver.py` and `eos/mixed/adapters.py` (`03ee45b`, 19:31) INSIDE this
ticket's measurement window, and `eos/mixed` imports `eos.dd2.species`. The
first `test/mixed` figure was therefore taken against their uncommitted tree —
the contamination ticket 45 paid for, arriving from the other direction.

Retaken on the committed HEAD after both commits here: **289 passed in 347s**,
`test/mixed` (277) plus this ticket's own check (12), zero failed. That is the
number reported above. The two changes compose: nothing in the section 6 fix
interacts with the six-flag `SpeciesFlags` work.

The other nine suites do not import `eos.dd2` and are unaffected either way.

### A ticket-number collision, for whoever reads this next

`62-regenerate-baselines-py314.md` (graduated by ticket 57) and
`62-species-flag-defaults.md` (created by the other session in `42cc7f5`) both
exist. Not renumbered from here — the second is another session's live work and
renaming it under them is how two sessions lose a ticket. This ticket's own
additions are 63 and 64, which are clear.

### Three tests changed, and why that is not loosening a tolerance

    test/abpr/test_abpr_modes.py:132   assert set(out) == {"cs2_isothermal"}
    test/sfho/test_sfho_responses.py:81  == {6 names}
    test/sfho/test_sfho_responses.py:100 == {"cs2_isothermal", "chi"}

Three exact-key assertions, and the ticket-11 ruling changes that key set on
purpose. No number moved and no tolerance was touched; each was rewritten as
`set(r) - {"converged", "reason"} == {...}` so it still asserts what it was
written to assert — which SECOND DERIVATIVES exist at that T. The abpr one
carries a comment saying the status keys are not second derivatives. These were
the only three failures in the whole targeted run, and they are the whole of
the "before" side of the numbers above.

### The check

`test/test_nonconvergence_return.py`, 12 cases: eight models parametrised over
"force the internal layer to fail, assert the same dict comes back with
converged=False and NaN in every quantity", both `mixed` branches separately
(central point and stencil), and the two SnB paths against a genuinely
unreachable target (`SnB=1e4`), which needs no forcing.

The response failures are FORCED at the internal layer rather than hunted for
at some extreme density, deliberately: a test that depends on a particular n_B
being unreachable stops testing anything the day the solver improves. The two
entropy cases do not need that, so they do not use it.

`test/` is gitignored (`.gitignore:75`), so this file lives ONLY in the working
copy — the hazard the map already records for tickets 39, 40, 45, 48 and 56.

### Two findings for new tickets, NOT fixed here

**1. A verify-suite silent-pass this ticket introduced.** Five causality checks
do

    worst = max(worst, max(0.0 - cs2, cs2 - 1.0, 0.0))

over a density grid (`zl/verify:245`, `vmit/verify:223`, `alphabag/verify:403`,
`did/verify:387`, `njl/verify:563`). A grid point that used to RAISE now returns
NaN, and NaN loses every comparison in `max`, so it would be absorbed and the
check would report PASS. Nothing is masked today — all nine suites pass, which
proves no grid point currently fails — but this is a new hole in the safety net
and it is the "nothing is ever silently skipped" clause of section 4. One
`np.isfinite` assertion per site would close it. Deliberately not widened into
here.

**2. `eos/general/verify/` does not exist.** CLAUDE.md section 5 states it in
the present tense — "**`general/` carries a `verify/` too**" — and names what it
checks: JEL against the alternatives section 7 requires be validated against it,
the basis maps against the species tables, the T = 0 limits against the finite-T
forms. `ls eos/general/` shows no `verify`, and
`python3 -m eos.general.verify.run_full_check` is a `ModuleNotFoundError`. This
is pre-existing, predates this ticket, and is NOT covered by
[ticket 51](51-verify-invariants.md), whose four missing invariants are all in
models. Found while running this ticket's verify sweep.
