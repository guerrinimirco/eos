# Two NMP surface fixes the closure ruling authorises: dd2's `from_nmp` raises,
# and every `PUBLISHED_NMP` gains its full-precision twin

Type: task
Status: resolved (2026-08-29)
Blocked by: -   (103 ruled both)
Parent: ../map.md

## Question

Two one-file consequences of [ticket 103](103-nmp-closures-four-models.md)
(2026-08-29). Neither is a physics change.

**1. `dd2.nmp.from_nmp` returns `None` where `sfho` and `zl` raise.** All three
carry the same docstring sentence for the convenience face -- "a caller asking
only for parameters has nowhere to put a failure" -- and two of them act on it.
[Ticket 93](93-invert-nmp-basin-lottery.md) recorded what the third produces:
the `None` travels until `solver.py` raises `'NoneType' object has no attribute
'kernel_masses'`, a section 6 non-convergence arriving as an AttributeError two
layers down. **Ruled: dd2 conforms.** `invert_nmp` stays the section-6
boundary and returns `(Parameters, InversionStatus)`; `from_nmp` raises.
`build_parametrization`, which already tests `status.ok` itself, must keep
behaving as it does.

**2. Every model's published NMP quote gains its full-precision twin.** 103
measured what the rounded quotes cost when the goal is recovering the published
COUPLINGS:

    dd2    1.6e-4 from the literature quote     (DD2 prints 4-6 digits)
    sfho   2.8e-2 from the quote
           1.3e-3 with an exact m*/m and everything else still rounded
           0      at full precision
    zl     7.2e-3, and it does not improve with precision

SFHo's whole factor of 22 is one two-digit entry: the paper prints
m*/m = 0.76 where the value is 0.761564. So each `PUBLISHED_NMP` keeps the
paper's quote -- which is what a reader comparing against the paper's table
needs -- and gains the forward-mapped values at full precision beside it,
which is what an inference starting "around" a published set needs. Both named,
neither a gate: ZL's 7.2e-3 is a property of its published fit (the couplings
saturate 0.3% below their own `n0`, so they are not a root of any closure) and
no gate could improve it.

## Gate

`dd2.from_nmp` raises on `ok=False` and its docstring says so; the three
models' convenience faces read identically. Each of dd2, sfho and zl exposes
both quotes under names that say which is which, with the measured recovery
distance recorded beside them. `test/<model>` green in all three; no baseline
moved -- both changes are on the inverse path and on module constants.

---

## Resolution (2026-08-29)

Both halves shipped. Neither is a physics change and no baseline moved.

### 1. `dd2.from_nmp` raises

`eos/dd2/nmp.py`. The three convenience faces now carry the identical two
lines:

    par, status = invert_nmp(...)
    if not status.ok:
        raise RuntimeError(f"NMP inversion failed: {status.message}")

`invert_nmp` is untouched and remains the section-6 boundary returning
`(Parameters, InversionStatus)`. `build_parametrization` **behaves exactly as
it did** — it still returns `(None, "inversion_failed", message)` on a soft
miss — but it now reaches that through `invert_nmp` directly rather than
through `from_nmp(..., return_status=True)`, because the face it used to lean
on is the one that no longer answers a failure with a value. One comment on
the line says so, since the call reads as a downgrade otherwise.

`docs/STRUCTURE.md` already said "`from_nmp` ... raises" of all three models;
it is now true rather than aspirational, so no document changed.

### 2. The regression test

`test/dd2/test_dd2_m8.py::test_from_nmp_raises_where_invert_nmp_reports`, the
sibling of sfho's test of the same name.

Finding the target was the work. The two feasibility tests already in that
file (`m_eff_ratio=0.25`, `E_sym=12.0`) raise `ValueError` and **never
returned None** — hard infeasibility was always an exception, so neither one
exercises the regression. What returned None is a SOFT miss: both target
values inside their physical windows, and the isoscalar root not found after
all 32 restarts. Scanned for one; **`K_sat = 400 MeV` at `m*/m = 0.85`** on
the published set is the shipped case, `ok=False` at isoscalar residual
1.97e-01, and it costs 0.3 s. The test asserts `not status.ok` from
`invert_nmp` FIRST, so if a future closure ever reaches that target the test
says "pick another one" instead of silently ceasing to test anything.

### 3. `PUBLISHED_NMP` + `PUBLISHED_NMP_EXACT`, in all three

Only sfho had a constant at all. dd2 had none, and **zl's quote lived only
inside `verify/run_full_check.py`** as five literals in a `want` dict. Each
model now carries both dicts, exported from its `__init__`:

    PUBLISHED_NMP         the digits the paper prints
    PUBLISHED_NMP_EXACT   compute_nmp(Parameters.default()) on the published
                          couplings, frozen so reading it costs no saturation
                          solve

zl's is five keys, not six: no scalar field, so no `m*/m`. `did` is excluded
throughout — it has no published closure, so there is no inverse for either
dict to seed.

`zl/verify` now READS `PUBLISHED_NMP` instead of restating it, keeping its
tolerances local. That was the one place the five numbers were duplicated.

### 4. The recovery distances, re-measured rather than transcribed

103's numbers were quoted without their metric, and re-deriving them landed
somewhere else, so the metric is now written beside each number: **the worst
RELATIVE distance between the couplings `invert_nmp` returns and the published
ones, over that model's free couplings**, named per model in the comment.

    dd2   from the quote  8.5e-05      over the eight free couplings
          at full precision  7.6e-05
    sfho  from the quote  3.7e-02      over (g_sigma_N, g_omega_N, g2, g3,
          m*/m exact, rest rounded  1.5e-03            g_rho_N)
          at full precision  0
    zl    from the quote  7.4e-03      over (a0, b0, gamma, a1, b1)
          at full precision  7.2e-03

Same structure as 103 measured, one row sharper and one row different:

- **New: dd2's rounding costs nothing measurable.** 8.5e-05 and 7.6e-05 are
  both the isoscalar solve's own convergence floor — DD2 prints enough digits
  that its rounding falls below it. 103 reported 1.6e-04 for the quote and
  drew the same conclusion; the discrepancy is the coupling set the max runs
  over, which is exactly what the metric sentence now fixes.
- **sfho reproduces exactly, cause and all**: the whole factor of 25 is the
  one two-digit entry, `m*/m = 0.76` against 0.761564, and full precision
  returns 0 because SFHo's published set IS a root of this closure.
- **zl's 7.2e-03 matches 103 to the digit** and does not improve with
  precision, because the published couplings saturate ~0.3% below their own
  `n0` and are a root of no closure. Reported, not gated, as 103 ruled: there
  is no gate that could improve it.

### 5. What guards the frozen twin

`test/test_parameter_routes.py::test_published_nmp_comes_in_both_precisions`,
parametrized over the three. It asserts `PUBLISHED_NMP_EXACT` equals
`compute_nmp(Parameters.default())` to rel 1e-12 — a frozen literal claiming
to be a call is precisely the thing that rots — and that the quote is a
ROUNDING of it rather than a different number, at rel 1e-2. That bound is
loose because the printed precision runs from six digits (dd2's `n_sat`) down
to two (sfho's `m*/m`, 2.1e-3), and the loosest entry sets it; it is a
transcription check, and what each rounding COSTS is measured beside the
constants instead.

### Gate

    test/baseline                   284 passed, NO baseline moved
    test/dd2 test/sfho test/zl test/mixed
      test_parameter_routes test_imports   green
    dd2/verify    11/11        sfho/verify  9/9        zl/verify  10/10
    FULL SUITE    1833 passed, 23 skipped, 0 failed   (30:37)

Measured on python.org 3.14 with `PYTHONPATH=.` (eos is not pip-installed
there). The full-suite run overlapped a concurrent session's own suite on the
same cores, which is what the 30 minutes are — 0 failures is the number that
matters, and the working tree it ran against carries that session's
uncommitted `eos/sfho/*` and `eos/dd2/{couplings,parameters}.py` edits as well
as this ticket's. The per-suite lines above were run before those edits landed
and agree.

### Not committed here

The working tree carries a **concurrent session's** uncommitted edits to
`eos/sfho/nmp.py`, `eos/sfho/parameters.py`, `eos/dd2/couplings.py` and
`eos/dd2/parameters.py` — 400+ lines that are not this ticket's. This
ticket's files are `eos/dd2/nmp.py`, `eos/sfho/nmp.py` (the last hunk only),
`eos/zl/nmp.py`, the three `__init__.py`, `eos/zl/verify/run_full_check.py`,
`test/dd2/test_dd2_m8.py`, `test/test_parameter_routes.py` and this file.
Whoever commits stages those by name; a bare `git commit -a` here would take
the other session's work with it, which is the shared-tree trap this map has
now recorded four instances of.

Status: resolved (2026-08-29).
