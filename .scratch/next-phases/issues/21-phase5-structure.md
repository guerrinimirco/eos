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

## Carried in from ticket 11

[Ticket 11](11-conformance-triage.md) ruled finding 31: **`general/` earns a
`verify/` suite.** It is the single home of the Fermi/Bose integrals (§7), the
conserved-charge basis maps (§2) and the thermal meson gas — what every model's
correctness rests on — and the JEL-vs-pure-Python fallback parity gap is already
ledgered as untested (`docs/DEFERRED.md:48-62`, "no parity test currently pins
the two together"). Seven files in `test/general/` cover it, but `test/` is
gitignored, so a fresh clone can check none of it.

§5's `verify/` list is extended by [ticket 22](22-phase5-claudemd.md); building
the suite is this ticket's, alongside `docs/STRUCTURE.md`. The JEL parity check is
the obvious first entry — §7 makes JEL the validated implementation that is never
removed and requires every alternative to be validated against it.
