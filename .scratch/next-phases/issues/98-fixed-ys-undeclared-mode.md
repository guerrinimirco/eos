# `fixed_YS` is a mode the code has and §3 does not declare

Type: grilling
Status: open
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
