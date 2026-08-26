# dd2's two remaining `cs2` names, neither of which is a rename

Type: task
Status: resolved
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

## Ruling

Agreed with the user, informed by **Zhao & Lattimer, arXiv:2204.03037** Eq. (1):

    nu_g^2 = g^2 (1/c_e^2 - 1/c_s^2) e^(nu-lambda)

with `c_e = sqrt(dp/deps)` the EQUILIBRIUM sound speed and
`c_s = sqrt(gamma p / (mu n_B))` the ADIABATIC one. **The g-mode IS the
difference between the two**; given only one, `nu_g` is identically zero.

**The finding that reframes this ticket: `cs2_ad` is not a mistake.** It is
Zhao & Lattimer's notation — `c_s`, "the adiabatic sound speed", meaning
*composition frozen*. The clash is between two literatures this repository
serves at once: in asteroseismology "adiabatic" means frozen composition; in
the CompOSE manual (Typel et al.) it means fixed entropy. The word cannot carry
either meaning unqualified here.

**1. Resolved by §5's own structure, not by picking a winner.** The COMPOSITION
axis rides on the `frozen=` argument; the THERMAL axis is the key name. So dd2
returns `cs2_isothermal` AND `cs2_adiabatic` at each `frozen=` setting — four
numbers, two keys, two arguments, no word doing double duty. `cs2_ad` goes.

**Both, not one**, at the user's request and because it costs one
multiplication: dd2 already computes `C_V` and `C_P`, and `eos/dd2/api.py:161`
already states the missing speed is "larger by `C_P/C_V` at T > 0". Five models
(`did`, `njl`, `ccdm`, `sfho`, and dd2 after this) then return both.
`zl`, `vmit` and `alphabag` compute only `C_V` and so can offer only the
isothermal one — their docstrings say which, per §5. At T = 0 the pair
coincides.

**2. `TableResult.cs2_eq` is `np.gradient(P, eps)` — literally Zhao's `c_e`.**
Only its THERMAL axis is ambiguous (isothermal on a `T` axis, adiabatic on an
`SnB` axis). **Two fields, `cs2_isothermal` and `cs2_adiabatic`, exactly one
populated per table**, chosen by the axis. `TableResult` already carries
`temp_key`, so the consumer knows which to expect and the name never lies. One
always-`None` field is cheaper than a consumer branching on an axis, and it
matches the two-key shape every model's response dict already has.

Callers to sweep: `test/dd2/test_dd2_m6_remainder.py:86`,
`eos/dd2/table.py:365`'s demo, `eos/dd2/backends/responses_jac.py:145`'s label.

Open for execution.

## Resolution

**Both parts of the ruling are executed. `cs2_ad` and `TableResult.cs2_eq` are
gone from `eos/dd2`; a grep for either over `eos/` now returns only
`eos/mixed` and `eos/astro/gmode`, whose surface is
[ticket 53](53-gmode-contract.md)'s.**

### 1. `eos_response` returns both speeds at both freezes

`eos/dd2/api.py` now returns `cs2_isothermal` AND `cs2_adiabatic` from
`frozen='equilibrium'` and from `frozen='composition'` alike — four numbers,
two keys, two arguments. The composition axis is the `frozen=` argument, the
thermal axis is the key.

- **equilibrium.** One multiplication, exactly as the ruling predicted: the
  branch already computed `C_V` and `C_P` at T > 0, so
  `cs2_adiabatic = (C_P/C_V) * cs2_isothermal` is a line in `api.py` and no new
  backend function. At T = 0 the branch does not compute the heat capacities at
  all (they are undefined there and dd2 has always omitted them rather than
  return zero), so `cs2_adiabatic` is set equal to `cs2_isothermal` — the
  C_P/C_V -> 1 limit, stated in the code.
- **composition.** This one was NOT one multiplication, and the ticket did not
  say it would be: dd2's `C_V`/`C_P` are taken along the **beta-eq** sequence
  in `backends/responses_jac.py`, so they are the wrong heat capacities to form
  the frozen-`Y_p` ratio with. `eos/dd2/responses.py` gained `_frozen_derivs`,
  a single central-difference stencil along the fixed-`Y_p` sequence returning
  `dP/dn`, `deps/dn`, `dsigma/dn`, `dP/dT`, `dsigma/dT` (sigma = s/n_B, per
  baryon because that is what C_P needs when the volume changes at fixed P).
  `sound_speed_adiabatic_frozen` builds `C_V` and `C_P` from it and returns
  `(C_P/C_V) * cs2_iso`. This is the same shape `did`, `njl`, `ccdm` and `sfho`
  already use, so five models now derive the pair the one way.

**The function names moved with the keys.** `responses.sound_speed_adiabatic`
was Zhao's `c_s` — frozen composition, taken at fixed T — and leaving that name
on it while a sibling returned the fixed-entropy speed would have been the
ticket's own defect one layer down. It split into
`sound_speed_isothermal_frozen` (the old body, byte-for-byte the same stencil,
so the number is unmoved) and `sound_speed_adiabatic_frozen`. Both are exported
from `eos.dd2`. Three callers outside `eos/dd2/` imported the old name, all
tests, all updated: `test/mixed/test_mixed_responses.py`,
`test/gmode/test_sound_speeds.py`, `test/dd2/test_dd2_m10.py`. No `eos/mixed`
or `eos/astro` source file imports it, so neither package was touched.

**One gap opened and is recorded, not silently left.** `Gamma` under
`frozen='composition'` is still built on `cs2_isothermal`, so at T > 0 it is
the isothermal index. Changing which speed it stands on would move a number at
T > 0, which is not what a naming ruling authorises; `docs/DEFERRED.md`'s dd2
section now says so and notes that `_frozen_derivs` already has the material.

### 2. `TableResult` carries two fields, exactly one populated

`cs2_eq` became `cs2_isothermal` and `cs2_adiabatic`, and `build_table` fills
whichever one the spec's temperature axis names — `T` -> isothermal,
`SnB` -> adiabatic — leaving the other `None`. The branch is an explicit
`if/else` on `spec._temp_key` next to the construction, not a computed keyword,
because a physicist reading `table.py` should see which axis fills which field
without resolving a dict (§13).

Callers swept, all four named in the ruling and one more:

- `test/dd2/test_dd2_m6_remainder.py` — a `T` axis; now reads
  `res.cs2_isothermal` and additionally asserts `res.cs2_adiabatic is None`,
  which is what pins "exactly one".
- `eos/dd2/table.py`'s `__main__` demo — an `SnB` axis, so it prints
  `max cs2_adiabatic=`. Run: `S=1.0: T range 9.6-20.5 MeV, max
  cs2_adiabatic=0.458`, `S=2.0: 20.6-45.1 MeV, 0.453`.
- `eos/dd2/backends/responses_jac.py`'s `__main__` label — `cs2_eq=` ->
  `cs2_isothermal=`, part 1's answer. Run: `n_B=0.16: cs2_isothermal=0.0599`,
  `0.4: 0.2598`, `0.8: 0.4775`.
- `build_table`'s docstring and `TableSpec.want_coeffs`' comment both said
  "equilibrium c_s^2", the freeze word this ticket exists to remove.

**The four notebooks needed no reader change.** The ruling expected a two-key
reader; ticket 69 had already collapsed them to a single `cs2_isothermal` read.
`notebooks/hadronic_eos` reads dd2 only through
`eos_response(frozen='equilibrium')`, which still returns that key; the other
three never call a dd2 response. One line of prose there was made true again —
it said `did` alone returns `cs2_adiabatic` beside it, and now `dd2` and `sfho`
do too — and the `.ipynb` pair was patched to match (one line each).

### Documents

`eos/dd2/dd2.md` and `dd2.tex` both gained the `cs2_adiabatic` row, the
`frozen='composition'` sentence, and a rewritten "isothermal against adiabatic"
paragraph that states the ruling's finding as physics: the .tex prints Zhao &
Lattimer Eq. (1) and cites `ZhaoLattimer2022` (already in `docs/eos.bib`)
against `TypelCompOSE2015`, so a reader of the document meets the clash rather
than inheriting it. The cross-cutting `docs/DEFERRED.md` entry now lists dd2
among the five models returning the pair and `zl`/`vmit`/`alphabag` as the
three that carry no C_P to form it with.

### Gate

**python.org CPython 3.14.2** (`/Library/Frameworks/Python.framework/Versions/3.14/bin/python3`), never under `timeout`.

A concurrent session was mid-edit on `eos/astro/gmode/sound_speeds.py` and had
added an untracked `eos/general/sound_speeds.py`; on the live tree that
combination breaks `test/gmode` at COLLECTION, which would have made a full-suite
number meaningless. The suite was therefore gated in an **isolated-copy pair**,
both copies with that session's two files reverted to HEAD, differing only in
whether this ticket's changes are present:

    mine     1697 collected   1671 passed, 11 failed, 15 skipped   38m45s
    control  1696 collected   1670 passed, 11 failed, 15 skipped   34m11s

**The eleven failures are the SAME eleven, name for name, in both.** Eight are
`test/gmode/test_reaction_rates.py` (`NameError`), which is the reverted gmode
file meeting that session's already-edited tests — an artifact of the isolation,
not of either tree. Two are `test_baseline[enjl]` and `test_baseline[ccdm]`;
`enjl` is the known pre-existing one, and `ccdm` fails in BOTH isolated copies
while passing on the live tree, so it too is an isolation artifact rather than
anything this ticket did. The +1 passed on the left is this ticket's new test.

`test/baseline` on the live tree after the change: **1 failed (enjl), 15
passed** — identical to the before-image taken at the start of the session.
**No baseline number moved and no tolerance was touched.** The rename could not
move one; the new adiabatic key is a new quantity and no baseline reads
`eos_response`.

`test/dd2`: **207 passed** (was 206 + the new test). `eos.dd2.verify.run_full_check`:
**PASS**, all nine checks, `responses` and `coeff analytic~FD` among them.

The new quantity is pinned by `test/dd2/test_api.py::test_both_freezes_return_both_sound_speeds`,
which is the ruling's own verification written out: at BOTH freezes the two keys
are present and coincide at T = 0 (rel 1e-12); at T = 15 MeV the equilibrium
freeze's `cs2_adiabatic` equals `(C_P/C_V) * cs2_isothermal` from the returned
heat capacities (rel 1e-12), and the composition freeze's equals the ratio built
independently from `_frozen_derivs` (rel 1e-8, a finite-difference comparison);
both are strictly greater than the isothermal one, which is C_P > C_V.

Nothing under `eos/*/species.py` (ticket 65) or `eos/mixed/` (tickets 46, 29)
was touched.
