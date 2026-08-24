# Phase 5 item 3 — write docs/STRUCTURE.md

Type: task
Status: open
Blocked by: 20, 14, 17, 19
Parent: ../map.md

## Question

`docs/REFACTOR_PROMPTS.md` Phase 5 item 3. `docs/STRUCTURE.md` does not exist
today, and `CLAUDE.md` §10 and §11 both reference it — §10 requires that a worked
figure example live there.

Aimed at a physicist who has never seen the repo: the module map; the mode and
species tables; the charge conventions (`Y_C` non-leptonic, `S = +1` per s
quark); the units (n in fm^-3, T and mu in MeV, eps and P in MeV/fm^3); the
reference/fast contract (§9); how to add a new model; and one worked end-to-end
example longer than the README ones. **Link each model to its document** — to the
`.tex` or the `.md` per ticket 09's ruling.

Two additions on top of the Phase 5 text:

- **Execute every code block and paste the real output.** Same standard as the
  README.
- **Link the three notebooks from Stages 1–3 as the worked examples** — which is
  why this is blocked on all three benchmark tickets.

Acceptance asks that a physicist find the function computing a given quantity in
under a minute from this document. Judge the draft against that.
