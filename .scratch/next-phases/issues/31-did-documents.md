# did.md and did.tex to §11 standard

Type: task
Status: resolved
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

## Resolution

Both files brought to CLAUDE.md §11, verified against the source before writing.

**did.md** was rewritten end to end: Lagrangian, mean-field equations, both
coupling functionals with their analytic derivatives, the SU(3) vector sector
with the Eq. (6) correction and its SU(6) test, the grand potential, both
rearrangement self-energies and the chain rule that weights Sigma^t, the
single-species Fermi gas at T = 0 and T > 0 written out, the Bose gas, the
leptons and photons, R1–R11 in the solver's own order, the modes table, the
NMP map, the responses, the adapter surface and the gaps. 168 lines → 885.

**did.tex** gained the parameter content it lacked entirely (C2) plus the four
omissions ticket 27 re-assigned here: the pi/K Bose thermodynamics, the
numerical degeneracies d_i, the value of FIELD_SCALE, and the eos_response
definitions. Also a "what one solved point carries" list closing C6.

**The complete parameter content, first stated here.** 29 fields in
`Parameters.default()` against the two numbers the old .md carried. Fifteen are
the fit; the vector transition zones (c = 3.5, d = 1.8), the plateaus
b_omega = 0.80 and b_rho = 0.40 and c_sigma = inf were fixed a priori. The
omega and phi couplings are DERIVED and were never written down anywhere:
computed from the source, g_8^S = 9.178769 and g_8^N = 9.326009, and the full
5×4×2 vertex table is now in both files.

**Two corrections the code settled.**

- Both documents said `F_M(1) = 0.988` for "the vector shape". There is no one
  vector shape: `a_rho` differs from `a_omega`, so F_omega(1) = F_phi(1) =
  0.98829 while F_rho(1) = 0.96488. Now stated separately, with the resulting
  saturation couplings g_sigmaN = 8.9487, g_omegaN = 9.9291, g_phiN = -5.2014,
  g_rhoN = 3.1168.
- `g_phiN = -5.20` (both files, unqualified) is the coupling AT SATURATION, not
  a stored parameter: the stored strength is g^{S(0)}_phiN = -5.262966. Both
  numbers now appear, each labelled.

**No .bib key was needed.** Every reference the new sections cite —
`TypelCompOSE2015` and `Constantinou2025` for the responses,
`JohnsEllisLattimer1996` and `Lavagno2010` for the gases — was already in
`docs/eos.bib`. It was re-read immediately before checking and not modified.

Gate: `did.tex` compiles clean at 13 pages with no undefined citation or
reference (latexmk, bibliography resolved). Commit `bb67689`; nothing outside
`eos/did/did.{md,tex}` was touched.
