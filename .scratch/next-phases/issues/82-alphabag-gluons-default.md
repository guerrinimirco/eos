# Should `alphabag.gluons` default False like the six?

Type: grilling
Status: open
Blocked by: -
Parent: ../map.md

## Question

Raised by [ticket 65](65-species-flag-defaults.md) and deliberately left to a
ruling, because `gluons` is **not** one of §4's six names and that ticket was
scoped to the six.

After 65, `eos/alphabag/species.py` reads:

    photons: bool = False
    gluons: bool = True
    thermal_neutrinos: bool = False

All three are mu = 0 thermal boson gases that carry no conserved charge and
contribute to eps, P and s alone, all three vanish at T = 0 — and one of them
is now on by default while its two neighbours are off. `SpeciesFlags()` in
`alphabag` therefore gives a thermal gluon gas and no photons.

**The case for False.** §4's rule is about sectors, not about which list a
flag's name appears on: "no sector is enabled or disabled implicitly ... if a
sector is off, its flag is False". A default of True switches the sector on
for every caller who did not name it, which is the same defect ticket 65
closed for the six.

**The case for True.** `gluons` is alphabag's own physics, not shared
vocabulary, and §4's six are the only names the repository promises behave
identically across models. A bag model's gluon gas is arguably part of what
the model *is* — closer to `enjl`'s fixed `hyperons=True` than to a
convenience — in which case the honest form may be `enjl`'s: fix it and raise
on any move, rather than leave it a silent default.

**What it would cost.** `alphabag` moved 0 baseline keys under ticket 65
because the generator calls its raw solvers (see
[ticket 81](81-second-default-solver-kwargs.md)), so this is measurable
cheaply through `eos_point` / `eos_table` before anything is regenerated.

`njl` and `ccdm`'s `csc`, and `dd2`/`sfho`/`did`'s `phi_field`, are the same
class of question and should be answered by the same ruling: `csc: bool =
False` is already off, and `phi_field=False` **raises** in `sfho` and `did`,
which is the fixed-by-the-model shape rather than a default.
