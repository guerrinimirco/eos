# Should §4's six flags carry the same default in every model?

Type: grilling
Status: open
Blocked by: -
Parent: ../map.md

## Question

Graduated from [ticket 61](61-dd2-species-flags.md), which was asked to measure
this and explicitly told to report rather than change it. All ten models now
carry all six §4 names; the **defaults** behind those names do not agree.
Measured with `dataclasses.fields`, on `main` with ticket 61 applied:

| model | hyperons | deltas | muons | thermal_mesons | thermal_neutrinos | photons | extra flags |
|---|---|---|---|---|---|---|---|
| `dd2`      | False | False | **True**  | False | False | **True**  | neutrinos, phi_field, sigma_star, thermal_vectors |
| `sfho`     | False | False | False | False | False | **True**  | phi_field |
| `zl`       | False | False | False | False | False | **True**  | — |
| `did`      | False | False | **True**  | False | False | **True**  | phi_field |
| `vmit`     | False | False | False | False | False | **True**  | — |
| `alphabag` | False | False | False | False | **True**  | **True**  | gluons |
| `abpr`     | False | False | False | False | False | False | gluons |
| `enjl`     | **True**  | False | **True**  | False | False | False | — |
| `njl`      | False | False | **True**  | False | False | False | csc |
| `ccdm`     | False | False | **True**  | False | False | False | csc |

Three axes disagree: `muons` (True in five, False in five), `photons` (True in
six, False in four), and `thermal_neutrinos` (True in `alphabag` alone).
`hyperons=True` in `enjl` is a fourth, but a different kind — that model fixes
every flag and raises on any move, so its default is a statement about the
model rather than a convenience.

**The rule at stake.** §4: "No sector is enabled or disabled implicitly because
'its coupling happens to be zero' — if a sector is off, its flag is False."
Read strictly that governs the *coupling*, not the default; read as intent, a
caller who writes `SpeciesFlags()` and gets photons in one model and not the
next has had a sector switched on implicitly. The rule is only unambiguously
honoured today if every caller passes all six every time, which is what the
notebooks' shared knobs cell will in fact do.

**Three candidate rulings**, and this is the decision the ticket wants:

1. **Unify on all-False.** Every sector is off unless asked for; `SpeciesFlags()`
   is the same object everywhere. Cleanest against §4, and it MOVES NUMBERS —
   `photons=True` is the current default in six models, so every T > 0 call that
   relies on the default loses the photon gas. Every `.npz` in `test/baseline/`
   built through a default would move, which §12 makes ground truth. The blast
   radius has to be measured before this is chosen, not after.
2. **Unify on the physically-usual set** (`photons=True`, `muons=True`, the rest
   False). Moves fewer numbers but still moves some, and it is a convenience
   argument dressed as a physics one.
3. **Rule that defaults are deliberately per-model** and say so in `README.md`
   and `eos/__init__.py`, on the ground that a bag model with no lepton sector
   should not advertise `muons=True`. Costs no numbers; requires the prose to
   stop implying the six behave identically.

Whichever is chosen, the answer belongs in the same three prose sites ticket 61
rewrote, and a check in `test/test_imports.py` alongside the six-name check is
what keeps it from drifting back — that half of 61 is the precedent.

**Not a notebook blocker.** The knobs cell passes all six explicitly, so
tickets 12, 15, 18 and 58 do not wait on this.
