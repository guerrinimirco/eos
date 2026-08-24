# Do the twelve model documents pass §11's reproduce-without-source test?

Type: research
Status: resolved
Parent: ../map.md

## Question

Stage 4. Read every `eos/<model>/<model>.md` and `eos/<model>/<model>.tex` for
`zl`, `sfho`, `dd2`, `did`, `vmit`, `alphabag`, `njl`, `ccdm`, `abpr`, `enjl`,
`mixed`, plus `eos/astro/tov/tov.tex`, against §11's test: **a physicist
reproduces the model from the document without opening the source.**

Report as one table, one row per document, one column per checkable claim:

- states the Lagrangian or grand potential
- every parameter, and the reference they are fitted to
- the field / gap equations
- the residual **row by row in the order the solver assembles it**, with the
  unknown vector
- the single-species thermodynamics at `T = 0` and `T > 0` **written out**, not
  cited — §11 forbids leaving the Fermi and Bose integrals to a citation even
  though they are shared code
- **every** returned quantity explicitly, including `s` and `n_s` and the
  identities they come through (`n_s = (eps − 3P)/m*`,
  `s = (eps + P − Σ_i mu_i n_i)/T`)
- the terms that differ between `P` and `eps`
- which rows each mode changes

Name what is missing, per document. Read-only: fix nothing, propose nothing.
Write the table to `.scratch/next-phases/research/document-audit.md`.

## Answer

Full report: [document-audit.md](../research/document-audit.md), 464 lines.
Read-only — nothing in `eos/` or `docs/` touched.

**Scope correction: 24 documents, not 25.** `eos/astro/tov/tov.md` exists
alongside `tov.tex`; the ticket named only the `.tex`. Both were audited.

**Result: two documents pass outright — `zl.tex` (14/14) and `alphabag.tex`
(16/16).** `sfho.tex` and `abpr.tex` miss by one cell. Full per-column table in
the report.

**The `.tex` is stronger than its `.md` in all twelve pairs**, without exception.
That is the audit's most consequential finding and it points the opposite way
from the hoped-for "keep only the `.md`" — see [ticket 09](09-tex-or-md.md).

**Furthest from passing:**

1. **`ccdm.md` (3/16)** — no Lagrangian, no grand potential.
   `B_eff = [U(0)−U(φ₀)] + [V(0,0)−V(f_π,ζ₀)]` names `U` and `V` without ever
   defining either — §11's first prohibition, sitting at the centre of the
   document. No integrals at either temperature, no unknown vector, no ordered
   residual, no `s`, no scalar density; tier-1 parameters listed by name with no
   numbers.
2. **`did.md` (3/16)** — a README, not a specification. Two parameter numbers
   against 27 fields in `Parameters.default()`. No kinetic section at all: the
   strings `eps`, `P^kin` and `k_F` do not occur, so both §11 identities, the
   totals, leptons, photons and the meson gas are all absent. Residual rows exist
   only as English names.
3. **`njl.md` (4/16)** — Lagrangian and tier-1 parameters, then it stops. No
   residual, no integral at either temperature, no `Ω`/`eps`/`s`, no P-vs-eps
   difference (there is no P or eps expression), and "**Modes.** All four of
   CLAUDE.md §3" as the whole of the mode-rows column. The `phi_u` in its printed
   gap equation is never defined.
4. **The `vmit` pair (7/16 and 9/16)** — the only model where **both** files
   fail. `grep` for `180`, `0.2`, `150` returns zero hits in either: not one
   parameter value, and no reference. Neither enumerates a residual, and the
   unknown ordering both give **contradicts** `solve_vmit_beta_eq` and
   `solve_vmit_trapped_neutrinos`. `vmit.md:81` also carries a stale claim that
   `eos_response` is unimplemented — `api.py:167` implements it.

Worst `.tex` on a single column is **`njl.tex`**: 720 lines containing **zero
parameter values** — no `Λ`, `G_S`, `K` or current masses. Its reproduction table
cites Rehberg 1996 for the *outputs* and never for the inputs, so no number in it
can be checked from the document.

**13 factual defects** reported and not fixed — graduated to
[ticket 27](27-document-defects.md). The one that bites hardest:
**`mixed.tex:524` uses an undefined macro `\tmuB`, so the document does not
compile past that equation** — which the Acceptance criteria block currently
requires of every model.

`njl` and `ccdm` are absent from CLAUDE.md's own §1 and §11 model lists — carried
into [ticket 22](22-phase5-claudemd.md).

Status: resolved.
