# `fixed_YS` is a mode the code has and §3 does not declare

Type: grilling
Status: resolved
Blocked by: -
Parent: ../map.md

## Question

Surfaced by a BayEoS design session while [ticket 85](85-claudemd-sentences-owed.md)
was landing, and it is that ticket's own failure mode seen from the other side:
not prose outliving behaviour, but **behaviour the prose never had a word for**.

CLAUDE.md §3 opens "Every model exposes the same modes" and tables five:
`beta_eq_neutrinoless`, `beta_eq_neutrino_trapped`, `fixed_YC`, `fixed_YC_YS`,
`cfl`. There is a sixth in the code.

- `eos/dd2/solver.py:875` — `MODES` carries
  `"fixed_YS": dict(charge_mode="neutral", strange_mode="fixed")`, and
  `:885` gives it the fraction tuple `("Y_S",)`.
- It is **reachable from the public API**, not an internal arm:
  `eos/dd2/api.py:71` documents `mode` as "A key of `eos.dd2.MODES`", and
  `eos/dd2/table.py:107` validates against the same dict. So
  `eos_point(par, "fixed_YS", species, n_B=..., Y_S=...)` is a supported call
  on dd2 that §3 does not define.
- `eos/general/modes.py:138` builds the name in the SHARED `ModeSpec`:
  `f"{base}_fixed_YS" if self.is_fixed("S") else base`, which yields both
  `fixed_YS` and `beta_eq_neutrino_trapped_fixed_YS`. Its docstring at `:124`
  calls these "the two combinations the named factories below do not have a
  word for" — the divergence is **declared in a docstring and nowhere in the
  specification**.
- `eos/vmit/table.py:33` — `MODE_FRACTIONS` has four entries and no
  `fixed_YS`, so an unsupported call there dies as
  `ValueError: unknown mode`, not as §3's "raises with a message saying
  which".

So dd2 exposes six modes, vmit four, and §3 says five and "the same modes".

**The same docstring is stale twice over**: `general/modes.py:124` says "The
four names of CLAUDE.md section 3". §3 has five — `cfl` was added and this
line was not.

### What has to be decided

Whether `fixed_YS` is a mode.

- **If yes**, §3's table gains a sixth row (n_B, Y_S, T — strangeness held,
  charge equilibrated), and §3's existing rule applies unchanged: a model that
  cannot support it RAISES saying which, and the gap goes in
  `docs/DEFERRED.md`. That is nine models to audit, not one. The
  `beta_eq_neutrino_trapped_fixed_YS` combination `ModeSpec` can also build
  needs the same ruling in the same breath, or the next session finds a
  seventh.
- **If no**, it is an internal solver arm and must stop being reachable by
  name through `eos_point` — the public `mode` argument stops being "a key of
  `MODES`" and becomes the §3 list, with `MODES` holding whatever the solver
  needs behind it.

Either way `general/modes.py:124` gets its count fixed.

### A second site, and it is better evidence than a miscount

Found by a peer session spot-checking this ticket, reported as "the same
off-by-one in a different file". It is **not** the same defect, and the
difference is the point. `eos/dd2/solver.py:882-883` reads:

    #: Mirrors `eos.mixed.MODE_FRACTIONS`, which names the four modes both
    #: engines share.

"Four" is CORRECT about `eos/mixed`: `eos/mixed/table.py:53` genuinely has
four entries. What is false is **"Mirrors"** — the dict directly beneath that
comment has five, and the fifth is `fixed_YS`. So this is not a stale count
that nobody updated; it is a claimed invariant between two engines that the
code one line below breaks, and it breaks it on exactly the mode this ticket
is about. `general/modes.py:124`'s "four names of CLAUDE.md section 3" IS a
plain miscount (§3 has five, `cfl` was added and the line was not) and the two
should not be fixed with the same edit or described in the same sentence.

Which way this one resolves follows the ruling above rather than preceding it:
if `fixed_YS` is a §3 mode, `eos/mixed` is the engine that is missing it and
the comment is describing a real gap in the wrong tone; if it is an internal
arm, dd2's public `MODE_FRACTIONS` should not carry it and the mirror becomes
true by deletion.

### The blast radius differs by a factor of ten between the two arms

Counted, not assumed, after a peer session raised it: the package holds
**eleven** `MODE_FRACTIONS` dicts — `abpr`, `alphabag`, `ccdm`, `dd2`, `did`,
`enjl`, `njl`, `zl` solvers plus `mixed`, `sfho`, `vmit` tables — and
`"fixed_YS"` appears as a key in **exactly one of them, dd2**
(`solver.py:885`, and in dd2's other public dict `MODES` at `:875`; the two
are consistent with each other).

This cuts against reading it as a §3 mode the rest of the package merely has
not caught up to. A §3 mode is one every model either exposes or explicitly
raises on with a message saying which; ten of eleven do neither — the key is
simply absent, so a caller falls through to `ValueError: unknown mode` listing
what that model does have. Declaring it in §3 makes the fix **ten models
wide**; demoting it makes the fix one comment and one key.

And the absence is not a physics limitation. `sfho/table.py:75` and
`did/solver.py:81` both carry `fixed_YC_YS`, so both already hold Y_S fixed —
they have the strangeness machinery and lack only the charge-EQUILIBRATED
variant. `sfho` is the other DD-RMF, the model closest to dd2 in physics and
in shape, and it does not have the key. (`zl` lacking `fixed_YC_YS` altogether
is the one case §3 already explains: "physically meaningless (fixed_YC_YS for
nucleonic ZL)". Spec and code agree there.)

So the one-model footprint, plus `general/modes.py` being able to build the
name structurally, reads as the second arm: an internal capability of the
shared `ModeSpec` that leaked into a single model's public surface. **That is
evidence for the ruling, not the ruling** — a mode wanted by a downstream
consumer on models that do not have it is exactly how a real gap would also
look from here.

### One input, weighted as an input and not as a decision

The BayEoS session (a §6 use case 5 consumer — a downstream physics package)
wants `fixed_YS` **on the quark models**, for the two-flavour Bodmer–Witten
gate: Y_S = 0, neutral, beta-equilibrated, so mu_u + mu_e = mu_d. It reports
the user said they would add it. That is a reason the mode is wanted; it is
not a reason it is a §3 mode rather than a `vmit`/`alphabag` capability, and
the two questions should not be run together.

The same session also records that it **always passes `leptons=` explicitly**
and never relies on a model default, so it is insensitive to
[ticket 91](91-leptons-default-and-drift-checks.md)'s ruling and should not be
weighted there.

**The same request has since arrived from the user directly**, as
[ticket 99](99-quark-ea-at-zero-pressure.md): E/A at P = 0 for two- AND
three-flavour quark matter. It adds one cost to this ticket's ledger that the
BayEoS input did not carry — holding Y_S = 0 where no populated species carries
S is exactly what [ticket 75](75-undetermined-potential-check.md)'s screen fires
on, and [ticket 72](72-enjl-branch-selection.md) measured the price. Ticket 99
also names a route this ticket's two arms do not cover: a SPECIES FLAG that
switches the s quark off, which reaches two-flavour matter through
`beta_eq_neutrinoless` and needs no sixth mode at all. Weigh it here; 99's
two-flavour arm is blocked on the answer.

**Whoever rules this owes a ping downstream.** BayEoS is holding its
two-flavour gate at `skipped` — logging `two_flavour_stable: skipped` rather
than passing silently — and its `OPEN_QUESTIONS` entry points at this ticket
instead of proposing a fix. It is waiting on the ruling, not guessing at it,
which is the behaviour that deserves to be told when the answer lands. It is
in the sibling checkout
(`/Users/mircoguerrini/Desktop/Research/Python_codes/BayEoS`), design-only,
and does not write to this tree.

## Gate

Whatever is ruled, the check is that the §3 list and the set of mode names a
caller can actually reach through `eos_point` are the SAME SET, in every
model, enforced somewhere rather than asserted in prose —
`test/test_imports.py` already holds this shape of drift check
(`test_the_six_species_flags_all_default_to_off`).

No number should move under either ruling: this is about which names are
callable, not what they compute. If a baseline moves, the ruling was applied
somewhere it was not meant to reach.

## Resolution

**`fixed_YS` is not a mode.** Arm (b): it is an internal `ModeSpec` label and
stops being reachable by name. Arm (a) is refused, and the reason is arm (c) —
the route [ticket 99](99-quark-ea-at-zero-pressure.md) added, which the two
original arms did not cover.

Ruled 2026-08-27 with the user. Gate form and the CFL constraint below were put
to them as the frontier round and both came back as recommended.

### The two `fixed_YS`es are different jobs wearing one word

This is the whole ruling, and the ticket's own evidence assembles it.

On `dd2` with hyperons on, Y_S is a genuinely free fraction: strange baryons
exist to carry it, `n_S` responds to `mu_S`, and the Jacobian column is
populated. `test/dd2/test_table_rows.py` said so in a comment all along —
"below the hyperon threshold there are no strange baryons to carry it and
[the mode] has no solution, which is physics, not a solver failure."

On a quark model at Y_S = 0 — **the only use anyone has actually asked for**,
from BayEoS and then from the user directly as ticket 99 — you are not fixing a
fraction. You are removing a sector. No populated species carries S, `n_S = 0`
holds for a whole range of `mu_S`, and the residual has no gradient in that
direction. That is [ticket 75](75-undetermined-potential-check.md)'s screen
firing, and [ticket 72](72-enjl-branch-selection.md) is the priced receipt:
`mu_S` undetermined at Y_S = 0 put one mode's residual within round-off reach
of a 1e-10 acceptance gate, `solve` fell through its seed list, and round-off
chose a chiral branch. The map has diagnosed this hazard twice; declaring a
mode whose only requested use walks into it would be the third.

CLAUDE.md §4 already refuses the (a) route in so many words:

> No sector is enabled or disabled implicitly because "its coupling happens to
> be zero" — if a sector is off, its flag is False.

Reaching two-flavour matter by asking for Y_S = 0 is disabling a sector through
a fraction that happens to vanish. **Route (c) is therefore not a competitor to
(a) for the same job; it is the correct spelling of the job (a) was about to be
misused for.** Two-flavour quark matter is `beta_eq_neutrinoless` with the
strange sector's flag False, which is what it physically is, and `mu_S` leaves
the unknown vector instead of sitting in it as a null column.

### What is left of the mode once (c) takes the demand

A hadronic capability on one model of eleven with **no consumer**: 5 source
hits in the package, 0 baseline keys, 0 notebooks, 0 `verify/` entries, 0
document mentions, and 0 tests naming it (the one that did,
`test_fixed_YS_counts_thermal_kaons`, passes `charge_mode="fixed"` and has been
solving `fixed_YC_YS` under the wrong name; renamed here). Declaring it in §3
costs ten models an audit and a `DEFERRED.md` entry each. Demoting it costs two
dict keys.

### (c)'s ten-model cost, which it turns out not to have

Weighed and dismissed. §4's six names are mandatory package-wide, so a seventh
would be ten models wide — but the quark sector's flavour content is **physics
only quark models have**, which puts the flag in §4's stated `phi_field` /
`gluons` / `csc` class, not in the mandatory six. Five models, not ten.

### The flag's category, ruled here and binding on ticket 99

The user's constraint: `strange=False` is meaningless under CFL, legal at most
for 2SC and unpaired. **That is `alphabag.gluons`'s shape exactly**, and §4
already carries the case:

> `alphabag.gluons` keeps two legal values in the unpaired modes and is a
> default there, and raises in `cfl` because a colour-flavour-locked phase has
> no free gluon gas. […] That is the same statement `abpr` makes by refusing
> the flag outright.

Swap "no free gluon gas" for "no free strangeness fraction" — §3's "the locking
fixes Y_C = 0 and Y_S = +1 identically" — and the sentence transfers verbatim.
So:

- **two legal values in the unpaired and 2SC modes**, defaulting False per §4;
- **RAISES under CFL pairing**, in `alphabag`, `njl` and `ccdm`, which each
  carry both regimes;
- **`abpr` refuses it outright**, as it does `gluons`: `cfl` is its only mode.

No new §4 category is created — this is §3's sentence about `cfl` ("not a
choice of equilibrium condition but a statement about which phase the model
describes") applied one sector at a time, which §4 already sanctions. **Ticket
99 inherits this and does not re-litigate it**; 99 still owns the flag's NAME,
because a §13 name has to fit the entry point it serves.

### The gate

`test/test_imports.py::test_every_model_exposes_only_section_3_modes`, two
halves because they are two defects:

1. every model's public mode registry is a **subset** of `eos.MODES`. Subset,
   not equal set: the ticket's "SAME SET" is literally unsatisfiable — `cfl`
   lives only on `alphabag` and `abpr`, and `abpr` has nothing else.
2. a §3 name a model lacks **raises with the mode named in the message**, §3's
   own sentence, and the half that keeps (1) honest — without it a model could
   satisfy the subset test by exposing nothing at all.

Half (2) passes today across all ten models with no behaviour change: every
`unknown mode {mode!r}; expected one of […]` message already names the mode,
and `abpr`'s `NotImplementedError` does too. It pins that rather than moving
it. Half (1) was mutation-tested: re-adding the `fixed_YS` key turns it red at
the subset assertion, removing it turns it green.

The complementary direction was already covered —
`test_the_top_level_carries_the_mode_and_species_vocabulary` asserts every name
in `eos.MODES` is buildable by a factory. It could not see a model exposing a
name that is *not* in `eos.MODES`, which is the hole `fixed_YS` sat in for as
long as it did.

The flag half of the gate needs nothing new:
`test_every_species_flag_defaults_off_or_raises` already enforces §4's
default-or-statement rule, so the strange flag is checked the day it is added.

### The two comment defects, fixed differently as the ticket required

- **`eos/general/modes.py`** — a plain miscount, "The four names of CLAUDE.md
  section 3" where §3 has five. Now "Four of the five names", with `cfl`'s
  absence explained (it is not a `ModeSpec`), and the docstring states outright
  that `fixed_YS` and `beta_eq_neutrino_trapped_fixed_YS` are **internal
  labels, not modes**, with the §4 reason and a pointer to the gate that keeps
  them from leaking back into an API.
- **`eos/dd2/solver.py`** — a false claimed invariant, `Mirrors
  eos.mixed.MODE_FRACTIONS`, broken one line below by the fifth key. Fixed by
  **deleting the key**, exactly as the ticket predicted: dd2 now has the same
  four entries as `eos/mixed`, and the comment is true as written. "Four" was
  never the defect there and is unchanged.

### Files

- `eos/dd2/solver.py` — `fixed_YS` out of `MODES` and `MODE_FRACTIONS` and out
  of the roster comment. `dd2/table.py` and `dd2/api.py` import these, so the
  public surface closes with one edit. The `solve` arm
  (`charge_mode="neutral", strange_mode="fixed"`) is untouched and still
  reachable by kwargs — it is a combination of existing switches, so removing
  it would mean ADDING a guard.
- `eos/dd2/api.py` — the module docstring's "DD2 also offers the extras in
  eos.dd2.MODES" deleted; there are no extras.
- `eos/general/modes.py` — `ModeSpec.name`'s docstring, above.
- `test/test_imports.py` — the gate.
- `test/dd2/test_thermal_meson_feedback.py` — `test_fixed_YS_counts_thermal_kaons`
  renamed to `test_fixed_YC_YS_counts_thermal_kaons`, which is the mode its
  arguments have always selected. Prose outliving behaviour, this ticket's own
  subject, in the test directory.
- `test/dd2/test_table_rows.py` — the stale mode name in the grid-choice comment.

### No number moves

By construction: this ticket removed names, not equations. `test_every_mode_builds_rows`
iterates `MODE_FRACTIONS`, so it now covers four modes instead of five and
asserts the same thing about each. No baseline `.npz` holds a `fixed_YS` key —
checked across all thirteen before the edit.

### CLAUDE.md is not amended

§3 stays at five modes, which is the ruling. §4 needs no new sentence: the
strange flag lands in the `phi_field`/`gluons`/`csc` class §4 already defines,
and its CFL refusal is the `alphabag.gluons` paragraph already written. A
ruling that required editing the specification would have been arm (a).

### Downstream

BayEoS pinged (see the map's Decisions-so-far). Its `two_flavour_stable:
skipped` can come off hold once ticket 99 ships the flag; the ruling it was
waiting on is that the gate is reached through `beta_eq_neutrinoless` plus a
species flag, never through a `fixed_YS` mode, so its `OPEN_QUESTIONS` entry
should not be rewritten to request one.

### Suite

Two runs covering every test (`test/dd2` and `test/test_imports.py` ran in
both, green in both), on **python.org 3.14.2 / numpy 2.3.5 / scipy 1.17.0** --
the canonical stack ruled by [ticket 57](57-canonical-stack.md), and the one all
13 baselines were regenerated on after [ticket 72](72-enjl-branch-selection.md):

- `pytest test/ --ignore=test/baseline` — **1718 passed, 20 skipped, 0 failed**
  in 16:42, exit 0.
- `pytest test/baseline test/dd2 test/test_imports.py` — **431 passed, 0
  failed** in 1:31. All 13 baselines reproduce at rtol = 1e-10, which is the
  "no number moves" gate.

**1758 collected**, against 1757 at [ticket 25](25-acceptance.md): this ticket
added exactly one test and moved nothing else. 1738 passed + 20 skipped = 1758.
