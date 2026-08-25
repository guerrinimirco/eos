# njl.md and njl.tex to §11 standard

Type: task
Status: resolved
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

---

## Resolution

**Both files rewritten to §11; `njl.tex` compiles clean.** Committed as
`323edf8`, `eos/njl/njl.tex` and `eos/njl/njl.md` only. `docs/eos.bib`
needed **no change**: `njl.tex` carries its own `thebibliography`, so the one
reference added there for the response functions (`TypelCompOSE2015`, the
CompOSE manual) is a new `\bibitem` in the `.tex` — the key was already in
`eos.bib`, as were the other nine.

### What each file gained

`njl.tex`, against the audit's four gaps:

- **C2, the largest single §11 violation in the audit** — a `\section{The
  parameters}` with all three tiers: the RKH five with the vacuum observable
  each is fitted to (`m_pi`, `f_pi`, `m_K`, `m_eta'`) and the derived
  `G_S = 5.0584e-6 MeV^-2`, `K = 1.5594e-13 MeV^-5`; the four structural
  fields; the four sampled ones with ranges; the three published sets; the
  species flags and which three raise; and the lepton/photon constants the
  shared sectors carry. Every number in the reproduction table is now checkable
  from the document alone.
- **The lepton delegation** — the paragraph that pointed at
  `eos/general/thermodynamics_leptons.py` is now a subsection with the charged
  lepton integrals, the closed massless-neutrino forms (checkable:
  `P_nu/P_gamma = 7/8` at `mu_nu = 0`) and Stefan–Boltzmann, plus which species
  are at which potential and what happens in a fixed-`Y_C` mode. Stated
  **uncut**: the cutoff regularises the four-fermion interaction and a free
  lepton has none.
- **T = 0 as closed forms**, not a limiting rule — and they are the Dirac-sea
  integrals with `Lambda -> k_F`, `g_sea -> g`, which is worth saying because
  it is the same integral over a different interval.
- **A `\section{What a solved point returns}`** — all ~40 fields, the table row
  keys, the progress dictionary, and `eos_response`'s five outputs with their
  definitions and the one implemented freeze.

`njl.md` was 166 lines of summary and is now the same document in plain text:
conventions, Lagrangian, parameters, both temperature limits of the medium
integrals, sea/condensates/vacuum, the whole pairing sector including the four
Hellmann–Feynman kernels, the vector sector, the totals with Euler, leptons,
the unknown vector and the eight row groups in assembly order, the modes table,
the pattern enumeration, the returned quantities, the adapter, the reproduction
table and the deferred list. The five-trap list survives as a five-line index
into the sections that now state each one in full.

### The carried-in defect, discharged

`n_s` **is the strange-quark density**, the third entry of
`(n_u, n_d, n_s)` in `solver.py:465`, and both documents now say so where the
returned quantities are listed. The reason it cannot be the scalar density is
stated rather than asserted: `eps - 3P = M rho_s` holds mode by mode for the
*medium* pieces and with `P_k4`, not `P_log`, and the assembled `eps` and `P`
additionally carry the sea, the condensate cost, the vector terms and the
pairing correction with different weights. So `rho_s,f` is integrated and the
trace identity is used nowhere.

### Three places the code corrected the draft

Verified against source before writing, per the ticket:

1. **24, not 100.** `NODES_PER_PANEL = 24` in `eos/general/fermi_integrals.py`.
   The `.tex` read as though 100 were the shipped rule; 100/800 is the accuracy
   benchmark for the *splitting*, and both files now separate the two.
2. **The adapter does not return the pattern label as a field.**
   `eos/mixed/adapters.py:1098-1101` builds `fields` from the masses, gaps,
   `mu_3`, `mu_8` and `Sigma_V` — no pattern. The winner comes back as the key
   of the warm-start mapping. **`njl_phase`'s own docstring says otherwise**
   ("the winner's label rides on the returned block's `fields`") — reported,
   not fixed; ticket 32 is documents.
3. **`n_ref` is a QUARK density.** `0.48 fm^-3` is `n_B = n_sat`, not
   `n_B = 3 n_sat`; the draft's "3 n_sat" was ambiguous and is now explicit.

Also corrected while there, both pre-existing: the reproduction table's two
solved-point rows overflowed the page (245pt and 221pt overfull), now full-width
paragraph cells.

### Gate

`pdflatex` twice: **no errors, no undefined references, no overfull boxes**,
13 pages. The full pytest suite was not run — another session owns it — and
nothing outside `eos/njl/*.md` and `eos/njl/*.tex` was touched.

Status: **resolved**.
