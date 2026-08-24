# Three grouped notebooks, or one per model?

Type: grilling
Status: resolved
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

## Answer

**Three grouped notebooks, as the prompt asks; CLAUDE.md §11 is amended to
match.**

What settles it is the figure list, which did not exist when §11 was written.
Stage 1's figures 1–5 each put all four hadronic models on one axis — `P` vs
`n_B` in beta equilibrium, the symmetric-matter slice against the heavy-ion
constraints, M–R, M–Λ, and `c_s^2`. A cross-model overlay is the point of those
panels, and nine per-model notebooks cannot produce them without one of the nine
importing the other eight, or a shared helper module §11 also forbids. The knobs
cell would additionally be duplicated nine times.

So `notebooks/` holds `hadronic_eos`, `quark_eos` and `enjl` (plus the two
zlvmit legacy notebooks §11 already exempts), and [ticket 22](22-phase5-claudemd.md)
changes §11's "one usage notebook per model" line to say one notebook per
*family*, with the reason.

This does **not** license a helper module beside them: §11's other clause holds,
and each notebook still calls library functions and carries its own plotting
code. What ticket 04 settles is a shared *shape* copied into three files, not a
shared import.

Status: resolved.
