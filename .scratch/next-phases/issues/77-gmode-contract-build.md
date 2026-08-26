# Build the T = 0 g-mode composition contract and drop the DD2 import

Type: task
Status: open
Blocked by: 53
Parent: ../map.md

## Question

Execution of [ticket 53](53-gmode-contract.md)'s ruling. Read it first — the
design is settled and grounded in Zhao & Lattimer, arXiv:2204.03037 Eq. (1).

A `general/` table carrying **the two sound speeds along the sequence** —
`c_e = sqrt(dp/deps)` and the frozen-composition `c_s` — beside
`EOSTable_for_TOV`, the layer both `astro/` and the models may import. Models
produce it; `gmode` consumes it; **no model internal is imported**.

**T = 0 only.** At T = 0 the thermal axis collapses and the composition axis
does not, so the payload is exactly two numbers per point with no thermal-axis
naming problem. Finite T when it is useful.

Removes the last live §1 breach: `gmode/rates.py:85` and
`gmode/sound_speeds.py:94,149` import `eos.dd2.solver` and `eos.mixed.responses`
at module level, so `import eos.astro.gmode` pulls DD2 in today.

`gmode/verify/run_full_check.py:39-41` reaches into dd2 as well. §1's `verify/`
carve-out is written for the model-to-model half of the rule; **whether it
extends to an astro suite reaching down into a model is a separate call** and
this ticket must make it explicitly rather than assume it.

Only `dd2` can fill the contract today ([ticket 78](78-composition-freeze-nine-models.md)).
A model that cannot raises saying so — §3's own answer to a partly-filled
surface. So this ticket does not make `gmode` general; it makes gmode's
DD2-only-ness **visible and per-model** instead of hidden in an import.

**Rides along, same files** (§6, §7): `gmode/rates.py:90-97`'s `G2_FERMI`,
`G_A`, `F_PI_NN`, `M_PI` are module constants with no override path, and `M_PI`
duplicates a mass `general/particles.py` already declares. Mass from there;
weak couplings become arguments.

Done when `import eos.astro.gmode` pulls in no model package, `test_imports.py`
gates it, and the dd2 g-mode numbers are unchanged.
