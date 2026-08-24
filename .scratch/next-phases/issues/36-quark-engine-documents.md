# alphabag, abpr, enjl and mixed document pairs to §11 standard

Type: task
Status: open
Blocked by: 09
Parent: ../map.md

## Question

Four pairs, all partial, none catastrophic.

| pair | .md | .tex |
|---|---|---|
| alphabag | 13/16 | **16/16 — passes** |
| abpr | 11/14 | 13/14 |
| enjl | 10/16 | 14/16 |
| mixed | 9/14 | 11/14 |

`alphabag.tex` is one of only two documents in the repo that pass §11 outright
and is the template for the rest.

Items living **only** in the `.md`, to survive the merge: ENJL's **correct**
`mu_C` unknown (the `.tex` prints `mu_e` there — the code decides, so the `.md`
is right and the `.tex` is the defect), `mixed`'s live `fixed Y_C + Y_Le`
refusal, and ABPR's layout map.

For `mixed`, two extra obligations: the phase-adapter contract and the
transition observables §5 requires a composite engine to return (`n_onset`,
`n_offset`, `chi`, the per-phase charge decomposition), and **the photon
treatment** — [ticket 29](29-mixed-species-flags.md) found the engine carries
photons unconditionally at T > 0 and neither document says so.

`mixed.tex`'s undefined `\tmuB` macro is [ticket 27](27-document-defects.md)'s,
but it is in the equations this ticket rewrites — coordinate with it.

Bring both files to CLAUDE.md §11's test — a physicist reproduces the
model from the document without opening the source — with the `.md` and the
`.tex` carrying the **same information**, each written natively for its format
([ticket 09](09-tex-or-md.md)).

§11 requires, explicitly, in both files: the Lagrangian or grand potential; every
parameter and the reference it is fitted to; the field or gap equations; the
residual **row by row in the order the solver assembles it**, with the unknown
vector; the single-species thermodynamics at `T = 0` and `T > 0` **written out
rather than cited** — the Fermi and Bose integrals are shared code but each
document states them anyway; **every** returned quantity, including `s` and `n_s`
and the identities they come through (`n_s = (eps − 3P)/m*`,
`s = (eps + P − Σ_i mu_i n_i)/T`); the terms that differ between `P` and `eps`;
and which rows each mode changes.

Per-document gaps are named in
[document-audit.md](../research/document-audit.md). The code decides wherever a
document and the source disagree; report any disagreement rather than silently
following one.

**Carried in from [ticket 27](27-document-defects.md)**: both `abpr`
documents give the *code* names `ms` and `Delta` in the parameter table where
the mathematics wants `m_s` and `Delta_0`. §13's rule that a name says what it
is applies to the documents too — a reader reproducing the model from the page
should not have to know the identifier.
