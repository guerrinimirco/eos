# Should §4's six flags carry the same default in every model?

Type: task
Status: resolved
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

## Ruling

Agreed with the user: **unify on all-False**, sequenced as a behaviour change.

**The measurement that decides it.** The defaults are load-bearing, not
cosmetic: **66 bare `SpeciesFlags()` calls**, **13 entry points** in `njl`,
`did` and `ccdm` that construct one when `species=None`, and **148 calls
passing only `hyperons=`** and inheriting the rest. So a physicist writing
`SpeciesFlags(hyperons=True)` gets a photon gas in `dd2` and none in `abpr`.
That is §4's "no sector is enabled or disabled implicitly" failing in practice,
not in theory.

All-False is the only default under which `SpeciesFlags()` means one thing
everywhere, and "off unless asked" is the only rule that cannot silently ADD
physics. Option 2 (the physically-usual set) moves numbers for a convenience —
the cost without the principle.

**This moves numbers and needs its own gate**: `photons=True` is the current
default in six models, so every T > 0 call reaching a default loses the photon
gas, and every `.npz` built through one moves. §12 makes those ground truth.
**Do it now**: [ticket 62](62-regenerate-baselines-py314.md) has just
regenerated all thirteen baselines, so the machinery and the judgement behind
it are warm. Measure the blast radius first, regenerate second.

`enjl`'s `hyperons=True` is a different kind of default — that model fixes every
flag and raises on any move, so its default states the model rather than a
convenience. It is exempt, and its docstring says why.

Open for execution.

## Resolution

**Unified on all-False, as ruled.** Nine models now default every one of §4's
six names to `False`; `enjl` is exempt and says why. `SpeciesFlags()` means one
thing in every model.

| model | flipped |
|---|---|
| `dd2`, `did` | `muons`, `photons` |
| `sfho`, `zl`, `vmit` | `photons` |
| `alphabag` | `photons`, `thermal_neutrinos` |
| `njl`, `ccdm` | `muons` |
| `abpr` | already all-False |
| `enjl` | **exempt** — fixes every flag and raises on any move |

A model's OWN flags are untouched: `phi_field`, `gluons`, `csc` are that
model's physics and default where the model says.

### The blast radius, measured before regenerating

**3108 of 53763 baseline keys moved, in four models. Every one is explained,
and a control run proves the flip is the sole cause.**

The control is the part worth keeping. The eight `species.py` edits were
reverted, all twelve baselines regenerated, and compared against the
pre-change set: **0 of 53763 keys moved.** So the generator is bit-for-bit
deterministic on this stack — no round-off drift, no concurrent-session
contamination — and the whole delta is attributable to the default and
nothing else. Ticket 62 had to argue its deltas were round-off; here the
null hypothesis is measured at exactly zero.

| model | moved | why |
|---|---|---|
| `did` | 292 | `P`, `P_photons`, `eps`, `s`, at T=10 and T=30 **only**. The photon gas is additive and enters no equation |
| `sfho` | 163 | same P/eps/s at T>0; plus the `isentropic` block, where losing the photon entropy makes the outer 1-D solve land on a **higher** T (+0.013, +0.054, +0.189 MeV at n=0.16/0.32/0.64) and the whole state follows. Outside `isentropic`, **zero** non-P/eps/s keys moved. `mu_mu` mirrors `mu_e` bit-for-bit — it is the shared lepton block, not a muon sector |
| `njl` | 1411 | the muon sector, and only it: every `nolep` case moved **0**, and `beta_eq_neutrinoless.lep.T0` moved **0** because mu_e ≈ 91 MeV < m_mu there. Moves are where `n_mu` -> 0: the T>0 thermal tail, and `fixed_YC` with leptons where mu_e = 259–362 MeV genuinely populates muons |
| `ccdm` | 1242 | same muon signature, same `nolep` = 0 and `T0` = 0 |
| the other eight | 0 | bit-identical |

Two residues, both cleared rather than waved past:

- **`ccdm pattern.2SC.n1.5.T0`, 4 keys.** All residuals or colour densities
  pinned at zero by colour neutrality: `n_3` = −7.5e-09 -> **exactly 0**,
  `n_8` = −4.4e-08 -> **exactly 0**, against an `n_C` scale of 2.6e4. Twelve
  orders below the physical scale, and the new values are the cleaner ones.
  `P_total` did not move; `n_mu` = 0 both sides.
- **`njl enumeration.n1.2.Delta`, a sign flip.** Δ₁ 75.29562952 -> −75.29562952
  with |Δ| unchanged and `f_total` differing in the last bit (rel 3e-16).
  `n_mu` = 0 and mu_e = 0.511 MeV **on both sides**, so no muon physics is
  involved at all. `'CFL'` and `'free'` declare the same mask
  `(True, True, True)` and differ only in seeding; both converge on this root,
  and Ω depends on Δ through |Δ|², so the component's sign is a phase
  convention. This is the coin-flip `eos/general/pairing.py` already documents:
  *"the order only decides which of two exactly degenerate answers is
  reported."*

### The five suite failures the change caused, and what each was

Not one was papered over; each was a real thing the flip exposed.

1. **`test_dd2_m6_remainder.py::test_yc_2b_electrons_and_muons`** — its flags
   said `SpeciesFlags(hyperons=True, phi_field=True)  # muons on`. The comment
   claimed the sector; the default supplied it. Now written `muons=True`. This
   is the ticket's "148 calls passing only some flags and inheriting the rest"
   in the flesh.
2. **`test_dd2_m4.py::test_octet_reduces_to_nucleon`** — compares the flag path
   against the gated `solve_beta_eq`, which carries its own
   `include_muons=True`. The two sides were at different lepton content. Flags
   now say `muons=True` to match, with a comment saying why.
3. **`test_pairing_patterns.py::test_the_enumeration_picks_the_lowest_free_energy`**
   — asserted `chosen.pattern == "CFL"`, i.e. pinned the coin-flip above. It
   duly flipped when an unrelated default moved the sixteenth digit. Rewritten
   to assert the **state** — `f_total` is the minimum and `|Delta|` matches the
   forced-CFL gaps — and to accept either of the two labels that reach that
   root. Strictly stronger than what it replaced.
4–5. **`test/vmit/test_uniform_api.py` and `test_tables.py`** — see below;
   these two are a finding, not a test defect.

### Finding: a second default for the same sector, one layer down

The two vMIT failures are the library disagreeing with itself. `eos_point` with
no `species` builds `SpeciesFlags()` (photons now off) while the bare
`solve_beta_eq_neutrinoless` defaults `include_photons=True`; the test compared
them and saw ΔP = 2.85e-04 MeV/fm³, exactly the photon pressure at T = 10.

`zl`, `vmit` and `alphabag` take `include_photons` / `include_gluons` /
`include_thermal_neutrinos` as bare solver kwargs defaulting **True**, and
`dd2`'s `solver.solve` carries `include_photons=True` and `include_muons=True`.
`SpeciesFlags` reaches them only through `api.py` / `table.py`. So §4's "a
sector that is off is off because its flag says so" is now honoured at the
dataclass and violated one layer below it, and the two defaults disagree.

**Reported, not changed** — following this map's own precedent, where
[ticket 61](61-dd2-species-flags.md) measured the divergent defaults, was told
to report rather than change them, and graduated this ticket. Flipping those
kwargs is a second behaviour change with its own blast radius (it would move
`zl`, `vmit` and `alphabag` baselines, which this one did not), and that is a
ruling to make, not to assume. The two tests were made explicit on **both**
sides so neither inherits any default, which is robust whichever way the
follow-up goes.

**A coverage gap comes with it.** `zl`, `vmit` and `alphabag` moved 0 baseline
keys *because the generator calls their raw solvers and never touches
`SpeciesFlags`*. Their public default did move: `alphabag.eos_point` at T = 30
went 156.3823 -> 156.2985 MeV/fm³. So `test/baseline/` does not exercise the
`SpeciesFlags` -> solver wiring for those three models at all.

### Finding: `alphabag.gluons` still defaults True

Out of scope by the letter of the ruling — `gluons` is not one of §4's six —
but after this change `alphabag.SpeciesFlags()` gives no photons and no thermal
neutrinos while still giving a thermal gluon gas, and all three are mu = 0
thermal boson gases contributing to eps, P and s alone. The diff leaves
`gluons: bool = True` sitting between two freshly-Falsed lines. Not changed;
raised for a ruling.

### The drift check

`test/test_imports.py` gains `test_the_six_species_flags_all_default_to_off`,
beside the six-name check, with the same shape: iterate `eos.MODELS` minus a
named `exempt` dict, and assert **two ways** — a model in `exempt` that stops
being exempt turns the check red, so the exemption cannot go stale silently.
`enjl` is the single entry. Verified non-vacuous: restoring `dd2.photons = True`
turns it red (`assert not ['photons']`), restoring False turns it green.

### Prose

- **`README.md`** — a paragraph saying all six default False and to ask for
  what the physics needs. All five examples now pass flags explicitly; because
  each was given the values that used to be the defaults, **every captured
  output reproduces bit-identically** (example 1's six lines, and example 3's
  `M_max = 2.419 M_sun`, `R(M_max) = 11.99 km`, `R(1.4) = 13.19 km`).
- **`eos/__init__.py`** — the `#:` block above `SPECIES_FLAGS`, same content,
  stating it is a behaviour change and naming the `enjl` exemption.
- **`docs/DEFERRED.md`** — checked, carries no claim about defaults. Untouched.
- Notebooks — no bare `SpeciesFlags()` anywhere; the knobs cell passes all six.

### For ticket 29

**`eos/mixed` must default all six of §4's names to `False`.** It needs no edit
today: it has no `species.py` and reuses `dd2`'s flags, so it inherited the
change. When 29 gives it its own, that is the row to copy, and
`test_the_six_species_flags_all_default_to_off` will cover it as soon as
`mixed` is in the iterated set.

### The gate

**Interpreter: CPython 3.14.2 / numpy 2.3.5 / scipy 1.17.0 / numba 0.63.1.**

    before  output/_audit/pytest_before_ticket65_1failed_1695passed_1711collected_py314.txt
            1 failed, 1695 passed, 15 skipped   (1711 collected, 18:32)
    after   output/_audit/pytest_after_ticket65_1failed_1696passed_1712collected_py314.txt
            1 failed, 1696 passed, 15 skipped   (1712 collected, 21:32)

**Zero added failures.** The one failure is the same node id before and after,
`test_baseline[enjl]`, red by design since ticket 62 and not touched here —
`enjl.npz` is still the CPython 3.9 file and was excluded from the
regeneration. The denominator moves 1711 -> 1712 by exactly the new drift
check. Key-by-key baseline diff kept at
`output/_audit/baseline_diff_ticket65_3108keys_py314.txt`.

Ran alone, as instructed. The working tree carried one pre-existing
modification throughout, `docs/DEFERRED.md`, which is ticket 46's rename
`build_mixed_eos_table` -> `build_hybrid_table` reaching a doc reference the
commit `95d4052` missed. It is correct against the code, is not this ticket's,
and was left alone; every write here used an explicit pathspec.
