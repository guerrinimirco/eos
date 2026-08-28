# Two NMP surface fixes the closure ruling authorises: dd2's `from_nmp` raises,
# and every `PUBLISHED_NMP` gains its full-precision twin

Type: task
Status: open
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
