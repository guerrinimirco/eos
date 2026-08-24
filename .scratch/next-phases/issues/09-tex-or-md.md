# Keep both .md and .tex per model, or drop the .tex?

Type: grilling
Status: resolved
Blocked by: 06
Parent: ../map.md

## Question

The user wants to consider **keeping only the `.md`** — a short literature
review, the physical motivation, the Lagrangian, the grand potential, every
thermodynamic quantity explicitly, the field equations — and dropping the `.tex`.

This is a `CLAUDE.md` change, not a file deletion: §11 currently mandates the
`.tex` ("Each model carries `eos/<model>/<model>.tex` (and `.md`)"), and the
Acceptance criteria in `docs/REFACTOR_PROMPTS.md` require *"Every model has a
.tex that compiles."* Both would have to move.

Give a recommendation with the cost either way, informed by what ticket 06 found
about which format actually carries the content today: a `.tex` that compiles is
the publishable artifact and the bibliography lives there; an `.md` renders on
GitHub and is what a reader of a public repo opens first. Duplication across the
two is the maintenance cost.

**[Ticket 06](06-document-audit.md) has now measured this, and it points against
dropping the `.tex`:**

- **The `.tex` carries more in all twelve pairs, without exception.** The only two
  documents that pass §11 outright are `zl.tex` and `alphabag.tex`; no `.md`
  passes. Dropping the `.tex` means promoting the weaker file in every case.
- **Dropping the `.tex` is outright blocked for two models.** `sfho.md` and
  `dd2.md` each say "the closed forms are in `<model>.tex` Eq. (T0)". A document
  that names its sibling as its own completion cannot be the survivor without
  first absorbing it.
- **Dropping the `.md` is cheaper but not free.** These live *only* in the `.md`
  and would have to migrate first: NJL's entire parameter list (its `.tex` has
  720 lines and zero parameter values), ENJL's *correct* `mu_C` unknown (the
  `.tex` prints `mu_e` there), SFHo's `hc³` field-source correction (the `.tex`
  is dimensionally wrong without it), DID's DID/DIDY identity and `responses.py`
  outputs, mixed's live `fixed Y_C + Y_Le` refusal, ABPR's layout map, TOV's only
  mention of `solver_fast.py`, and every "Not implemented" ledger.

So the real choice is not "which file do we delete" but **which file absorbs the
other**, and the audit says the `.tex` is the cheaper direction — at the cost of
losing GitHub-rendered documentation, which is what a reader of a public repo
opens first. A third option the audit makes viable: keep the `.tex` as the
specification and cut the `.md` down to a genuine short literature review plus a
pointer, dropping the pretence that it is self-contained.

Note that `mixed.tex` does not currently compile ([ticket 27](27-document-defects.md)),
so "every model has a .tex that compiles" fails acceptance today either way.

Resolved when the user has ruled. Fixing an incomplete `.md` in place needs no
permission and is not this ticket. Deleting any `.tex` happens only under the
ruling recorded here.

## Answer

**Ruled: keep both. The `.md` and the `.tex` carry the SAME information — each
written natively for its format.** Neither is a pointer to the other, neither is
a subset. LaTeX mathematics in the `.tex`, Markdown-with-mathtext in the `.md`;
same equations, same parameters, same residual rows, same returned quantities.

So the `.tex` is **not** dropped, §11 stands unchanged, and the Acceptance
criterion "every model has a .tex that compiles" stands — which makes
`mixed.tex`'s undefined `\tmuB` macro a live blocker
([ticket 27](27-document-defects.md), now unblocked).

The cost is explicit and accepted: **24 documents to bring to §11 standard, not
12.** [Ticket 06](06-document-audit.md) measured where each one stands; only
`zl.tex` (14/14) and `alphabag.tex` (16/16) pass today. The completion work
graduates out of the map's fog into [tickets 30–36](../map.md).

What this rules out: the "short literature review plus a pointer" shape the
ticket floated for the `.md`. A `.md` that says "the closed forms are in
`<model>.tex`" — which `sfho.md` and `dd2.md` both do today — is now a defect to
fix, not a division of labour.

Status: resolved.
