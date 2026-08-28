# Retire `phi_field`; the hidden-strange vector is controlled by its coupling

Type: task
Status: resolved (2026-08-28)
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


## Ruling executed (2026-08-28): the sector moved to the coupling, and DID's refusal dissolved

The flag is gone from all three models. `grep -rn phi_field eos/` returns
nothing. **1812 passed, 23 skipped, 0 failed** on the full tree; `dd2.npz` and
`mixed.npz` untouched (mtimes predate the session).

### DID: the ticket's preferred option does not exist, and this was confirmed before any deletion

The ticket said "prefer the refusal: the §4 statement survives, it just moves
from a boolean to the coupling." **There is nothing to refuse.** DID's phi
ratios are `g_phi/g_8 = -tan(theta) - c_i(z, alpha)`, one per multiplet, and
they have no common zero — measured, not argued:

| `(z, tan_theta)` | N | Lambda | Sigma | Xi |
|---|---|---|---|---|
| published `z = 0.0772` | -0.573 | -0.707 | -0.707 | -0.841 |
| `tan_theta = 0` | +0.134 | 0 | 0 | -0.134 |
| `z = 1/sqrt6` (SU(6)) | ~0 | -0.707 | -0.707 | -1.414 |
| `z = 0` | -0.707 | -0.707 | -0.707 | -0.707 |

Ideal mixing kills only the nucleon's; zero mixing angle only Lambda's and
Sigma's. So a `tan_theta == 0` guard would refuse a setting that is NOT "phi
off" and its message would be false, and an all-zero-column guard would be a
branch unreachable at every parameter set. **The user chose prose-only**
(2026-08-28): the flag goes, no guard replaces it, and the statement lives in
`did/species.py`, `did/did.md` and a test that asserts the no-common-zero
property directly (`test_phi_sector_has_no_off_switch`, four settings).

This is the honest reading of §4 anyway. DID's phi was never a *sector* that a
boolean or a coupling could switch — it is structural, and the old flag's
`NotImplementedError` was the model saying so through the only channel it had.

### The three models carry it three ways, as the ticket said

- **`dd2`** — the `x_phi = g_phiY/g_omegaN` column of `hyperon_couplings`. New
  `Parameters.has_phi_coupling` property (`any(row[5] != 0)`), and the six
  readers that said `flags.phi_field and flags.hyperons` now say
  `flags.hyperons and par.has_phi_coupling`. The named route is
  `from_hyperon_potentials(..., x_phi=None)`: `None` keeps the SU(6) column, a
  float replaces it in every row, so `x_phi=0.0` builds a hyperonic set with no
  phi and a study wanting a *scaled* phi gets that free. Chosen over a
  `phi=True/False` keyword, which is the retired boolean moved one file over
  and gives §6's sampler nothing continuous to vary.
- **`sfho`** — pure deletion. The flag had **no reader anywhere**: it existed
  only to raise on False. The coupling was already the switch and already
  exercised, as `SFHo_2fam` (g_phi = 0 for every hyperon) against
  `SFHo_2fam_phi`. The field stays in the unknown vector and solves to zero, so
  no solver branch depended on the boolean — which is the second argument for
  the ruling, beside the ticket's observation that `g_phi_N = 0.0` already made
  a nucleonic `phi_field=True` run solve a field with no source.
- **`did`** — above.

### Two sites the ticket's list did not carry, both real

- **`notebooks/hybrid_eos.py` (+ `.ipynb`)** rebuilds a retired hybrid run from
  the `# key = value` header the retired notebook wrote, and that header records
  `flags.phi_field`. Deleting the constructor line alone would have silently
  rebuilt a phi-OFF retired run with the phi ON — a wrong number, quietly. The
  header key is now translated into the parameters: `flags.phi_field != "True"`
  zeroes the `x_phi` column of `retired_par`. This is the migration the ruling
  implies for every stored provenance block, and the notebook is where one
  exists.
- **`docs/STRUCTURE.md:427`** named `phi_field` as its worked example of a
  model-private flag, in the paragraph immediately above "No sector is enabled
  or disabled implicitly because its coupling happens to be zero" — the exact
  sentence the ruling has to be reconciled with.

Prose also updated: `eos/__init__.py` (both the `SPECIES_FLAGS` preamble and
the two-category rule, which cited `dd2.phi_field` / `sfho.phi_field` /
`did.phi_field` as its three worked examples), `eos/mixed/species.py`,
`eos/did/did.md` (two places).

### The CLAUDE.md §4 sentence, and the tension it had to resolve

§4's parenthetical lost its first example (now `gluons`, `csc`, dd2's
matter-composition `neutrinos`) and gained a paragraph. The paragraph had to
answer §4's own opening line — "No sector is enabled or disabled implicitly
because *its coupling happens to be zero*" — which reads as forbidding exactly
this ruling. It does not, and the distinction is named explicitly: **there a
number vanishes and nothing says so; here the coupling IS the statement, named
and documented as the sector's switch.** The paragraph ends on the phi as the
worked case across all three models, so §4 keeps a concrete example where it
had one.

### Gate

- `grep -rn phi_field eos/` → **nothing**. The only survivors repo-wide are
  `test/` (the drift check's own message, and DID's `not hasattr` assertion)
  and `notebooks/hybrid_eos.py`, where it is a *retired header key* being read,
  not a flag.
- **No number moves, and the mechanism says why**: every `phi_field=False` in
  the tree was paired with `hyperons=False`, where dd2's `phi_field and
  hyperons` was already False; every `phi_field=True` was paired with a DD2Y
  par, whose SU(6) `x_phi` column is nonzero. `test/baseline/dd2.npz` and
  `mixed.npz` unmoved.
- **The drift check** is `test/test_imports.py::
  test_phi_sector_is_off_exactly_when_its_coupling_is_zero`, beside
  `test_every_species_flag_defaults_off_or_raises`. Two limbs: no model's
  `SpeciesFlags` carries the name (dd2, sfho, did), and through the public
  `eos_point` the field is present exactly when the coupling is — dd2 with the
  `x_phi` column zeroed loses `phi0` and softens (`on.P > off.P`), sfho's
  `SFHo_2fam` has `phi = 0.0` where `SFHo_2fam_phi` has `-1.95`, and
  `from_hyperon_potentials(x_phi=0.0)` is asserted to build a phi-free set.
- `test/dd2/test_dd2_m4.py::test_phi_field_presence` kept its name and its
  physics (φ repulsion stiffens) but now switches by zeroing the coupling.

## Suite

**1812 passed, 23 skipped, 0 failed**, 34:45, python.org 3.14.2, 1835
collected — a POST-change count with no pre-change control, since `test/` is
gitignored and this session took no snapshot. 102's own contribution is exactly
**+1** (the drift check); the two converted tests are 1-for-1. Ticket 93
recorded 1835 earlier the same day, which would make this 1836 — flagged on the
map rather than reconciled here, because nothing in this gate rests on the
denominator: the claim is "no number moves", and that is carried by the unmoved
`dd2.npz`/`mixed.npz` and by 0 failed.

**A first run lied and it is worth recording how.** Run 1 reported exit code 0
with its output truncated at 54% and no summary line, having shown two `F`s at
31%. Both are artefacts of running against a concurrent session (which
regenerated `test/baseline/vmit.npz` at 14:48): the 29–33% band is
`test_dd2_m9` → `test_couplings`, containing the timing-sensitive
`test_dd2_speed.py`. Re-running the same tree with the shell owning the log
file: that band clean, 0 failed. **An exit code of 0 on a truncated pytest log
is not a pass** — the summary line is the only thing that counts.
