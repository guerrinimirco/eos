# tov.md and tov.tex to §11 standard

Type: task
Status: open
Blocked by: 09
Parent: ../map.md

## Question

Both score **5/10** — the only pair where neither file leads.

The largest gap: both **treat the rotating case as a citation rather than a
formulation**, though the RNS backend ships. Neither Komatsu 1989 nor CST 1994 is
in `docs/eos.bib`, so the citations do not even resolve. §11's test is that a
physicist reproduces what the code does; today the rotating solver cannot be
reproduced from either document.

`astro/tov` is not a model and has no residual or single-species
thermodynamics, so those columns are genuinely N/A — but the structure equations,
the tidal deformability, the crust treatment and the rotating formulation are all
in scope and must be written out. `tov.md` carries the only mention of
`solver_fast.py` in either file.

Note that `astro/tov` also has no `verify/` suite — that is a (c)-class
DEFERRED item from [ticket 08](08-conformance-table.md), not this ticket.

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
