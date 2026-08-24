# Phase 5 item 5 — apply the CLAUDE.md diff

Type: task
Status: open
Blocked by: 02, 09, 11
Parent: ../map.md

## Question

`docs/REFACTOR_PROMPTS.md` Phase 5 item 5: re-read `CLAUDE.md` against the repo as
it now is and correct anything the refactor settled differently. It was written
in Phase 0 as a target.

This ticket **applies** what three earlier tickets ruled — it does not decide
anything itself:

- ticket 02 — §11's "one usage notebook per model" line
- ticket 09 — §11's mandated `.tex`, and the matching line in the Acceptance
  criteria block of `docs/REFACTOR_PROMPTS.md` if the `.tex` goes
- ticket 11 — every (b)-class row from the conformance triage

[Ticket 08](08-conformance-table.md) produced 11 (b)-class rows for this diff.
The largest: **`ccdm` appears nowhere in CLAUDE.md**, and `njl` only ever inside
the string "enjl" — §1's model list omits `did`/`njl`/`ccdm`, §11's omits
`njl`/`ccdm`, and §5's adapter list omits the shipped `njl_phase` and
`ccdm_phase`. The `verify/` carve-out from the model-to-model import rule is real
(`test/test_imports.py:88-97` plus a DEFERRED entry) but unstated in §1.

One of the 11 lands outside CLAUDE.md: **the §10 acceptance criterion**
`grep -rn "rcParams" eos/ nucleation/` *"hits exactly one file"* now hits three,
but two of those are prose saying the file does **not** set rcParams. The rule
passes in substance — all ~30 assignments live in `general/figure_style.py` — so
it is the grep-as-gate that needs rewording, in the Acceptance criteria block of
`docs/REFACTOR_PROMPTS.md` rather than here.

One correction found independently by [ticket 06](06-document-audit.md) and owed
nothing to the tickets above: **`njl` and `ccdm` are absent from CLAUDE.md's own
§1 and §11 model lists**, though both ship as full models with documents, tests
and `verify/` suites. That is a plain omission in the specification, not a
settled-differently row, so it lands here regardless of how the triage rules.

Show the diff before removing anything. The (c)-class entries from ticket 11 land
in `docs/DEFERRED.md` under this ticket too, so the two documents move together.
