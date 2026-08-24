# ccdm.md and ccdm.tex to §11 standard

Type: task
Status: open
Blocked by: 09
Parent: ../map.md

## Question

`ccdm.md` scored **3/16** — the joint worst. It has no Lagrangian and no grand
potential, and `B_eff = [U(0)−U(φ₀)] + [V(0,0)−V(f_π,ζ₀)]` names `U` and `V`
without ever defining either: §11's first prohibition, at the centre of the
document. No integrals at either temperature, no unknown vector, no ordered
residual, no `s`, no scalar density; tier-1 parameters listed by name with no
numbers. `ccdm.tex` is at 14/16 and is the content source for most of it.

`docs/ccdm_implementation.md` is the reference for what the model implements.

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

**Carried in from [ticket 27](27-document-defects.md)**: `ccdm.md` uses
`R1..R5` to label **modes** while `ccdm.tex` uses `R1..R4` to label **residual
rows**, and `ccdm.md` then uses `R_4` in both senses. Ticket 27 did not fix it
because the fix is to choose one labelling across the pair, which is this
ticket's job. §11 wants the residual rows numbered in solver order; give the
modes a different label entirely.
