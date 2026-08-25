# zl, sfho and dd2 document pairs to §11 standard

Type: task
Status: claimed
Blocked by: 09
Parent: ../map.md

## Question

Three pairs where the `.tex` is strong and the `.md` lags, so the work is mostly
bringing the `.md` to parity — then closing the remaining `.tex` cells.

| pair | .md | .tex |
|---|---|---|
| zl | 6/14 | **14/14 — passes** |
| sfho | 9/16 | 15/16 |
| dd2 | 9/16 | 14/16 |

**`sfho.md` and `dd2.md` each say "the closed forms are in `<model>.tex`
Eq. (T0)".** Under [ticket 09](09-tex-or-md.md)'s ruling that is now a defect:
each file carries the same information, so the closed forms are written out in
both.

Two items live **only** in the `.md` and must survive the merge, not be
overwritten by the stronger `.tex`: SFHo's `hc³` field-source correction (the
`.tex` is dimensionally wrong without it) and DID-style "Not implemented"
ledgers. `zl.tex` passes outright and is the model for what the others should
look like.

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

**Carried in from [ticket 27](27-document-defects.md)**: the three-flavour
`mu = 0` thermal-neutrino gas that `eos/sfho/solver.py:523-527` adds to `P`,
`eps` and `s` appears in **neither** sfho document. It is an omission rather
than an error, which is why ticket 27 left it here — but §11 requires every
contribution to the totals to be written out, so both files gain it.
