# dd2's two remaining `cs2` names, neither of which is a rename

Type: grilling
Status: open
Blocked by: 69
Parent: ../map.md

## Question

[Ticket 69](69-cs2-eq-naming.md) renamed the `eos_response` equilibrium key to
`cs2_isothermal` in the four models that spelled it `cs2_eq`. Two `cs2` names
in `dd2` survive it, and neither is a rename — each needs a decision first.

**1. `eos/dd2/api.py`, `frozen='composition'` returns `cs2_ad`.**
`eos/dd2/responses.py:35-44` holds `T` on both stencil points and perturbs
`n_B` at fixed `Y_p`, so this is the *isothermal speed at frozen composition*.
`_ad` therefore misnames its thermal axis in exactly the way `cs2_eq` did.

But it cannot simply become `cs2_adiabatic`: that key already exists in `did`,
`njl`, `ccdm` and `sfho`, where it means `(C_P/C_V) * cs2_isothermal` — the
FIXED-ENTROPY speed. Two different quantities under one key is worse than the
misnomer.

Applying ticket 69's own principle gives `cs2_isothermal`, with the
composition axis carried by `frozen='composition'` — so `dd2` would return
`cs2_isothermal` from both freezes, distinguished by the argument. That is
consistent with §5 and reads oddly at first glance, which is the decision.

No caller reads `cs2_ad` today: the only site is the return dict itself.

**2. `eos/dd2/table.py`, `TableResult.cs2_eq`.**
`_cs2_along` is `np.gradient(P, eps)` along one line of the table, and the
line's thermal condition is whichever axis the spec was built on — `T` or
`SnB`. So the field is **isothermal on a `T` axis and ADIABATIC on an `SnB`
axis**, and can be renamed to neither. This is §5's "bare `cs2` whose meaning
depends on the arguments" in its purest form, and the fix is a decision about
the field, not a name: split it in two, carry the thermal key alongside it, or
name it for the axis it was built on.

`test/dd2/test_dd2_m6_remainder.py:86` and `eos/dd2/table.py:365`'s demo read
it. `eos/dd2/backends/responses_jac.py:145`'s print label follows whichever way
part 1 goes.

Nothing here changes a number; both are naming decisions with a physics
argument underneath, which is why this is a grilling rather than a task.
