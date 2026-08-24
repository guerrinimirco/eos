# njl.md and njl.tex to §11 standard

Type: task
Status: open
Blocked by: 09
Parent: ../map.md

## Question

`njl.md` scored **4/16**: Lagrangian and tier-1 parameters, then it stops. No
residual, no integral at either temperature, no `Ω`/`eps`/`s`, no P-vs-eps
difference (there is no P or eps expression), and "**Modes.** All four of
CLAUDE.md §3" as the entire mode-rows column. The `phi_u` in its printed gap
equation is never defined.

`njl.tex` is at 12/16 but carries the audit's worst single column: **720 lines
containing zero parameter values** — no `Λ`, `G_S`, `K` or current masses. Its
reproduction table cites Rehberg 1996 for the *outputs* and never for the inputs,
so no number in it can be checked from the document. The parameter list currently
exists only in the `.md`.

`docs/njl_csc_implementation.md` is the reference for what the model implements,
including the colour-superconducting pairing patterns.

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

**Carried in from [ticket 27](27-document-defects.md)**: the returned field
named `n_s` is the **strange-quark density**, not a scalar density, and neither
document says so. Every other model in the repo returns `n_s` as the scalar
density via `n_s = (eps - 3P)/m*`, so the collision is actively misleading and
must be called out where the returned quantities are listed.
