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

## Gate

Whatever is ruled, the check is that the §3 list and the set of mode names a
caller can actually reach through `eos_point` are the SAME SET, in every
model, enforced somewhere rather than asserted in prose —
`test/test_imports.py` already holds this shape of drift check
(`test_the_six_species_flags_all_default_to_off`).

No number should move under either ruling: this is about which names are
callable, not what they compute. If a baseline moves, the ruling was applied
somewhere it was not meant to reach.
