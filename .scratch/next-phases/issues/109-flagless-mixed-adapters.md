# Should `zl_phase`, `vmit_phase` and `alphabag_phase` take a `flags`?

Type: grilling
Status: open
Blocked by: [95](95-vmit-solver-flags.md), [96](96-alphabag-solver-flags.md)
Parent: ../map.md

## Question

Surfaced by [ticket 94](94-zl-solver-flags.md), which hit it in one adapter and
declined to invent an API three times.

`eos/mixed/species.py` states the rule: photons and thermal neutrinos are
PHASE-COMMON and counted once at the mixture level, "which is why every shipped
adapter hands the phase it wraps a flag object with `photons=False`. ... The one
exception is a phase's `wing_sweep`, whose rows are stitched into the hybrid
table as they stand, with no mixture layer above them to add the radiation:
those carry the caller's own `photons`."

**Three adapters cannot obey the exception, because they are handed no flags at
all.** `dd2_phase(par, flags)` and `sfho_phase(par, flags)` take one;
`zl_phase(params=None)`, `vmit_phase(params=None)` and
`alphabag_phase(params=None)` take parameters and nothing else. So their wings
have no caller `photons` to carry, and each has had to pick a value:

    dd2_phase       wing follows `flags.photons`          the rule, obeyed
    sfho_phase      takes flags; check what its wing does
    zl_phase        `photons=False` since ticket 94       agrees with the
                                                          mixture at its default
    vmit_phase      `include_photons=True` default        radiation
    alphabag_phase  `include_photons=True` default        unconditionally

The two `True` rows are ticket 95's and 96's to move and will land on the same
choice zl faced. **That is why this is one ruling and not three**: whichever way
it goes, the three adapters should agree, and a hybrid table at T > 0 whose
wings and window disagree about the radiation is the defect underneath all of
it.

**The candidates.**

1. **Give all three a `flags=None` parameter**, defaulting to that model's
   all-False `SpeciesFlags`. The wing then follows the caller, `species.py`'s
   sentence becomes true again, and `zl_phase()` / `vmit_phase()` keep working.
2. **Rule the exception dead**: a wing carries `photons=False` in EVERY adapter,
   including `dd2_phase`, and `eos.mixed.hybrid` adds the radiation once when it
   stitches. Fewest places for the two to disagree, and it moves dd2's wing —
   [ticket 89](89-dd2-honours-species-flags.md) deliberately put it the other
   way and its reasoning has to be answered, not skipped.
3. **Leave it per-adapter and document why.** Costs nothing and keeps a rule
   `species.py` states but the code only half-keeps.

Whichever is chosen, say whether a hybrid table's wing and window must agree
about the radiation at T > 0, because that is the physics question under the
API one. `test/mixed/test_hybrid_modes.py` is what asserts the dd2 answer today.

**Not a baseline blocker.** `mixed.npz` is T = 0 throughout and moves under no
answer here.
