# The shared notebook skeleton: knobs cell, gap handling, table naming

Type: prototype
Status: resolved
Blocked by: 02
Parent: ../map.md

## Question

All three notebooks share one spine. Build it once, concretely, as a throwaway to
react to — not prose about what it might look like.

Three things to pin down:

1. **The knobs cell** (first executable cell, everything selectable from it):
   model subset; mode (§3) with its own conditions (`Y_C`, `Y_S`, `Y_Le`,
   `Y_Lmu`) and the `leptons` flag; the `n_B` grid and either a `T` grid or an
   entropy-per-baryon `SnB` grid; the six species flags (§4); and the
   parametrisation per model — published sets via `Parameters.default()` /
   `Parameters.named(...)`, plus, for models carrying `nmp.py`, a set built by
   inverting `{n_sat, E_sat, m*/m, K_sat, E_sym, L_sym}` with `Q_sat`/`K_sym`
   reported as predictions.

2. **The unsupported-combination pattern.** A mode, flag or parametrisation a
   model does not support raises with a message saying which (§3). The notebook
   catches that at the top of a section, prints the message, and continues with
   the models that do support it. It never presents a gap as a result. Settle
   the exact shape of that try/except block and its printed form.

3. **The table-save convention.** Stage 1 asks that every table produced show how
   to save it to `output/tables/` under a standardised, automatic name. Settle
   the naming scheme (model, mode, fractions, grid, species flags) and the
   one-call helper form — noting that §11 forbids a helper module beside the
   notebook, so it lives in the notebook or in `eos/general/` table I/O.

Deliver as a runnable `.py` fragment. Resolved when the user has reacted to the
concrete artifact and the three shapes are fixed; tickets 12–19 then copy it.

## Answer

**The artifact is [research/notebook_skeleton.py](../research/notebook_skeleton.py)**
— runnable, self-check passing (`python3 notebook_skeleton.py`). Tickets 12, 15,
18 and [58](58-hybrid-skeleton.md) copy it; nothing imports it (§11 forbids a
helper module beside a notebook), with the single exception named in part 3.

### The finding that changes the ticket: §4's six flags are not constructible

Measured across all ten models with `dataclasses.fields`:

| model | missing from §4's six | extra |
|---|---|---|
| **`dd2`** | **`thermal_mesons`, `thermal_neutrinos`** | `neutrinos`, `phi_field`, `sigma_star`, `include_pseudoscalars`, `include_thermal_vectors` |
| `sfho`, `did` | — | `phi_field` |
| `alphabag`, `abpr` | — | `gluons` |
| `njl`, `ccdm` | — | `csc` |
| `zl`, `vmit`, `enjl` | — | — |

`SpeciesFlags(**six)` raises **`TypeError` on dd2** and works on the other nine.
The defaults diverge too: `photons` is True in six models and False in four,
`muons` True in four and False in six, and `alphabag` ships
`thermal_neutrinos=True`. §4's "no sector is enabled or disabled implicitly" only
holds if the knobs cell passes all six every time — which is exactly what dd2
refuses.

**No conformance row covers this.** Ticket 08's audit and
[ticket 11](11-conformance-triage.md)'s triage were both checked: §3-ii covers
`thermal_neutrinos` + trapped *behaviour*, not the missing *names*. And dd2's
`neutrinos` is **not** a misspelled `thermal_neutrinos` — its own comment reads
"only trapped / fixed-Y_Le modes", i.e. matter-composition neutrinos. So dd2 is
missing the tau-gas sector outright, while `thermal_mesons` is present but split
into `include_pseudoscalars` + `include_thermal_vectors`.

That is a rename **plus a gap to wire or refuse**, so it is
[ticket 61](61-dd2-species-flags.md) and not a line of this one. Tickets 12, 15,
18 and 58 are blocked by it: a knobs cell carrying a per-model translation table
would be copied into four notebooks, and §13's "two models that do the same job
have the same files" is the rule that costs the most to break.

### 1. The knobs cell

One dataclass, everything selectable from it, and two things it refuses to do:

- **`leptons` is a field of its own, never an entry in `conditions()`.** §5 makes
  it neither a condition nor a freeze target, and routing it through the bag is
  what produced the invented mode name `fixed_YC_neutral`
  ([ticket 54](54-signature-corrections.md) item 1).
- **`conditions()` returns only the fractions the mode takes.** Set `Y_S` while
  in `fixed_YC` and it is dropped rather than silently accepted:
  `Knobs(mode="fixed_YC", Y_C=0.1, Y_S=0.4).conditions() == {"Y_C": 0.1}`.

`use_nmp_inversion` ships **off**. `zl.invert_nmp` refuses by design
([ticket 26](26-zl-nmp.md): six couplings against five NMPs) and dd2's inversion
is [ticket 47](47-dd2-nmp-inversion.md), unruled — so **ticket 12 is also blocked
by 47**, or its end-to-end pass would depend on an open ticket's outcome. The
knobs cell still offers the knob and the notebook prints zl's refusal: a printed
refusal is §3's contract working, not a gap.

`eta` is deliberately **absent**. For `hybrid_eos` it is a scalar per call and
not an axis (`mixed/api.py` says so: it changes the shape of the unknown vector),
so ticket 58 adds it as its own knob rather than letting anyone grid it.

### 2. The unsupported-combination pattern: three shapes, kept three

The ticket assumed one failure mode. There are three, and conflating them is the
"presents a gap as a result" failure the ticket exists to prevent.
`run_section(name, call, **kwargs)` returns `(status, payload)`:

- **`"unsupported"`** — `NotImplementedError` or `ValueError`, the two a refusal
  uses. Caught, printed with the model's own message, section continues.
- **`"unconverged"`** — §6 makes non-convergence a **return value**, so no
  `except` clause ever sees it; it is found by testing `.ok`. Reported in its own
  words, because calling it "not supported" would be a lie about the physics.
- **`TypeError` is not caught at all** — an unexpected keyword is the notebook's
  own bug (precisely what the dd2 finding above produces today), and a broad
  `except Exception` would file it under "this model does not support that" where
  nobody would ever see it. The self-check asserts it escapes.

Printed form, one indented line per skip, under a section header carrying the
mode and conditions so a skipped model is visible in the executed output:

    === hadronic — mode=fixed_YC {'Y_C': 0.1} leptons=True ===
      [dd2] not supported: finite T not wired
      [sfho] did not converge: no root bracketed

### 3. The table-save convention

**`standard_name()` goes in `eos/general/table_io.py`, beside `save_table`** —
which already exists and already takes `windows=`, so ticket 58's mixed tables
need nothing new. §7 makes `general/` the single home for table I/O and §11's ban
is on a helper module *beside the notebook*; this is neither beside one nor new,
and four notebooks would otherwise carry four copies of the same string-building.
**It lands under [ticket 12](12-hadronic-skeleton.md)**, not here — the fragment
is the decision, the code move belongs with the first notebook that needs it.

Order: model, mode, the mode's fractions, `eta` if a composite engine, the
thermal axis, the density axis, the sectors that are ON, `nolep` only when
leptons are off. Everything that changes a number is in the name, so two runs
cannot collide silently; the complete metadata still goes inside the file through
`save_table(meta=...)`.

    dd2_fixed_YC_YC0.100_T0.0-30.0x4_nB0.1-1.2x64_mu+ph.h5
    vmit_beta_eq_neutrinoless_T0.0x1_nB0.1-1.5x32_ph_nolep.h5
    dd2vmit_fixed_YC_YC0.100_eta0.30_T0.0-30.0x4_nB0.1-1.2x64_mu+ph.h5

`bare` is the literal token when no sector is on, rather than an empty gap that
would make the name ambiguous. Length (~55 chars) is accepted deliberately: a
folder of these is self-describing months later, which is the whole point.

**Tables land in `output/tables/<model>/`, created on demand.** This ticket said
`output/tables/` and §11 says `output/` holds per-model/per-study subfolders —
a prompt/CLAUDE.md conflict, ruled toward §11. `output/` no longer exists (the
user renamed it to `output_old/`), so the tree is created fresh and clean.

Status: resolved.
