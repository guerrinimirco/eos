# Retire `phi_field`; the hidden-strange vector is controlled by its coupling

Type: task
Status: open
Blocked by: -
Parent: ../map.md

## Question

**Ruled by the user (2026-08-27):** retire the flag and control the sector by
setting the phi coupling to zero in the parameters. This ticket executes that
ruling; it does not reopen it.

The reasoning behind it is §4's own. A sector switch that duplicates a
coupling already in `Parameters` gives two ways to say the same thing and one
of them is not an inference knob: a sampler varies couplings (§6), so a
sector reachable only through a boolean is a sector inference cannot turn
down continuously. `g_phi = 0` is the same statement, it lives where every
other model number lives, and it removes a flag from §4 rather than adding a
category to it.

[Ticket 82](82-alphabag-gluons-default.md) is the immediate predecessor and
this is its natural end: 82 sorted every model-private flag into "default and
False" or "statement and raises", and `phi_field` was the flag that made both
lists — `False` in `dd2`, `True`-and-raises in `sfho` and `did`. A parameter
has no such split to make.

### The three models do not carry the coupling the same way, and one of them
### may not be able to take this ruling as stated

This is the ticket's real content; the flag deletion itself is mechanical.

- **`sfho`** has the field literally: `parameters.py:129`, `g_phi_N = 0.0`
  ("Usually 0 for nucleons"), with the hyperon phi couplings built from
  `SU6_RATIOS` as `-sqrt(2)/3 * g_omega_N`. Zeroing is a direct assignment,
  and the nucleon coupling is ALREADY zero — so `phi_field=True` in a
  nucleonic SFHo run currently solves a field whose source vanishes, which is
  a second argument for the ruling.
- **`dd2`** has no scalar `g_phi`. The coupling lives per hyperon, as the
  `x_phi = g_phiY/g_omegaN` column of the `hyperon_couplings` rows
  (`parameters.py:60-66`). "Set `g_phi = 0`" means zeroing that column in
  every row, so the ruling needs a named way to do it that is not editing a
  tuple by hand — the constructor is `from_hyperon_potentials`, which builds
  the rows from `SU6_HYPERON`, so the switch belongs there.
- **`did` cannot do it by assignment at all, and this needs a decision.** DID
  stores neither `g_omega` nor `g_phi`: both are DERIVED from the aggregated
  strength `g~_omegaN` and the SU(3) ratio `z` through
  `couplings.g8_from_aggregate` (Eq. 52), because that combination is what the
  Bayesian analysis varies. Forcing `g_phi = 0` is therefore a statement about
  `z` and the mixing angle `tan_theta`, not a parameter assignment, and it
  contradicts the ideal-mixing values the model ships (`ALPHA_IDEAL`,
  `TAN_THETA_IDEAL`). DID's flag today RAISES on False and says "not a DID
  configuration" — which under the ruling has to become either a parameter
  refusal with the same message or a structural change nobody asked for.
  **Decide this before deleting DID's flag**, and prefer the refusal: the
  §4 statement survives, it just moves from a boolean to the coupling.

### Sites

`eos/dd2/species.py:51`, `eos/sfho/species.py:55`, `eos/did/species.py:45`
(and the two `__post_init__` raises); readers at `eos/dd2/solver.py:415,1016`,
`thermodynamics.py:328`, `table.py:208,318`,
`backends/responses_jac.py:42,114,145`, `verify/run_full_check.py:269`,
`eos/mixed/adapters.py:229` and `eos/mixed/species.py:17`; and the prose at
`eos/__init__.py:73,95-98`, which names `phi_field` as an example of a
model-private flag in the two-category rule.

**One CLAUDE.md sentence is owed and it is §4's**, whose parenthetical
"(`phi_field`, `gluons`, `csc`)" names the retired flag as its first example.
It needs a replacement example and, better, one added line: a sector already
carried by a coupling is controlled by that coupling, not by a second boolean.

## Gate

- `grep -rn phi_field eos/` returns nothing outside `docs/DEFERRED.md` history.
- **No number moves.** dd2 reads the flag only as `phi_field and hyperons`, so
  a nucleonic run cannot see this; a DD2Y run with the shipped rows keeps
  every `x_phi` it has. `test/baseline/dd2.npz` and `mixed.npz` unmoved,
  full suite green, 0 added failures.
- One new check, beside `test_every_species_flag_defaults_off_or_raises`: the
  phi sector is off exactly when its coupling is zero, asserted through
  `eos_point` on dd2 and sfho — the drift check that stops the boolean
  returning under another name.
