# Three grouped notebooks, or one per model?

Type: grilling
Status: open
Parent: ../map.md

## Question

`CLAUDE.md` §11 states: `notebooks/` holds "one usage notebook per model".
`docs/NEXT_PHASES_PROMPT.md` Stages 1–3 ask for **three grouped notebooks**
(`hadronic_eos` covering zl/sfho/dd2/did, `quark_eos` covering
vmit/alphabag/njl/ccdm, `enjl`) covering nine models.

These cannot both stand. The decision shapes the whole notebook half of this map
and one row of the conformance triage.

Weigh: a grouped notebook is where cross-model overlay figures (Stage 1 figures
1–5 put all four models on one axis) are natural, and one knobs cell serves four
models; per-model notebooks are what §11 promises a reader and what
`docs/REFACTOR_PLAN.md:110` planned.

Resolve to one of: grouped (and §11 is amended by ticket 22), per-model (and the
prompt's Stages 1–3 are re-cut), or a hybrid with the split stated explicitly.
