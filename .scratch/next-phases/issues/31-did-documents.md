# did.md and did.tex to §11 standard

Type: task
Status: open
Blocked by: 09
Parent: ../map.md

## Question

`did.md` scored **3/16** — a README, not a specification. Two parameter numbers
against 27 fields in `Parameters.default()`. **No kinetic section at all**: the
strings `eps`, `P^kin` and `k_F` do not occur, so both §11 identities, the
totals, leptons, photons and the meson gas are all absent. Residual rows exist
only as English names.

`did.tex` is at 12/16 but fails the parameters-and-reference column outright, so
neither file currently carries the parameter table — it has to be written from
`eos/did/parameters.py` and the paper. Note the two known paper traps recorded
for this model: the Eq. (6) typo and `tau_3 = 2 I_3`.

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
