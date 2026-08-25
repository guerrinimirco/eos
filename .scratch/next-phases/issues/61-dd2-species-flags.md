# dd2 cannot take two of §4's six species flags

Type: task
Status: open
Blocked by: 44
Parent: ../map.md

## Question

Found by [ticket 04](04-notebook-skeleton.md) while building the shared knobs
cell, and covered by **no row** of ticket 08's audit or
[ticket 11](11-conformance-triage.md)'s triage — §3-ii covers
`thermal_neutrinos` + trapped *behaviour*, not the missing *names*.

`SpeciesFlags(**{six §4 flags})` raises **`TypeError` on dd2** and succeeds on
the other nine models. dd2 is missing `thermal_mesons` and `thermal_neutrinos`:

    dd2  extra = neutrinos, phi_field, sigma_star,
                 include_pseudoscalars, include_thermal_vectors

Two different problems wearing one finding:

1. **`thermal_mesons` is present but split** into `include_pseudoscalars` and
   `include_thermal_vectors`. §4 names one flag — "pi, K (and optionally the
   vector nonet)" — so the split is arguably a refinement, but §4's name does
   not exist and §13's "the same job carries the same name in every model" is
   the rule that binds. Decide: one `thermal_mesons` flag with the vector nonet
   behind a second, clearly-secondary name, or `thermal_mesons` as the alias the
   other nine models spell.
2. **`thermal_neutrinos` is genuinely absent.** dd2's `neutrinos` is NOT a
   misspelling of it — the field's own comment reads "only trapped / fixed-Y_Le
   modes", i.e. the matter-composition neutrinos. So the tau gas is unwired.
   §4 is explicit that setting a flag a model does not implement **raises**, and
   that a NotImplementedError is never a silent no-op — so a raise is an
   acceptable resolution here and wiring the sector is not required. What is not
   acceptable is the flag not existing, because then the caller gets a
   `TypeError` that reads as their own bug.

Also worth measuring while in the file: the **defaults** diverge across models —
`photons` True in six and False in four, `muons` True in four and False in six,
`alphabag` shipping `thermal_neutrinos=True`. §4's "no sector is enabled or
disabled implicitly" is only honoured if a caller passes all six every time.
Whether the defaults should be unified is a separate ruling; report, do not
change, unless the answer is obvious.

**Tickets [12](12-hadronic-skeleton.md), [15](15-quark-notebook.md),
[18](18-enjl-notebook.md) and [58](58-hybrid-skeleton.md) are blocked by this.**
The alternative — a per-model translation table in the knobs cell — would be
copied into four notebooks.

Blocked by [44](44-rename-dd2.md) so it does not collide with the dd2 renames in
the same files, and run tickets 42/43's AST check here: both proved that a
rename onto a name this repo already uses for a local adapter fails silently.

Resolved when `SpeciesFlags(**six)` constructs for all ten models, or raises
with a §4-compliant message. Report added failures with the interpreter named
([ticket 57](57-canonical-stack.md)) and check `test/baseline/` is unmoved.

**Neighbour, not duplicate:** [ticket 60](60-dd2-hyperon-flag-raise.md) is a bare
`KeyError` when `hyperons=True` meets a nucleonic parameter set — the same file
family, a different defect (a refusal with the wrong exception, against a flag
that does not exist at all). Whoever runs second should read the other first.
