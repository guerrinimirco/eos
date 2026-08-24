# tov.md and tov.tex to §11 standard

Type: task
Status: resolved
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

## Progress — the rotation gap closed, the pair not yet at 5/10 -> full

Two of the audit's four failing columns are now closed in **both** files.

**C1 / C3 — the formulation.** Both documents described how RNS is *driven* and
never what it solves. They now carry the Komatsu–Eriguchi–Hachisu
quasi-isotropic metric in the Cook–Shapiro–Teukolsky form RNS implements, the
four potentials with which three are elliptic, the statement that rotation is
uniform (`Omega` is one number) and that `omega(r,theta)` is the frame-dragging
rate rather than the same quantity, the relativistic first integral

    H + (1/2)(gamma + rho) - (1/2) ln(1 - v^2) = const,
    v = (Omega - omega) r sin(theta) e^(-rho),

and the specific enthalpy `H(P) = int dP'/(eps + P')` that carries it. That last
is the only place the EoS enters, which is why the RNS table has an enthalpy
column and why `_specific_enthalpy` integrates on 16003 points uniform in ln P —
so the interface details the documents already had now read as consequences
rather than trivia.

`Komatsu1989` and `CookShapiroTeukolsky1994` added to `docs/eos.bib`: both
documents leaned on a formulation neither cited, and neither entry existed.

**C6 — returned quantities.** Both files listed the static returns and said
nothing about `RotatingResult`, which is a different object with eighteen fields
in RNS's units. Now tabulated in both, with the two easy misreadings called out:
`Omega_K` is the mass-shedding rate *of the model in hand*, not the sequence's
Kepler frequency; and `I` is NaN rather than zero at `r_ratio = 1` because it
comes from `J/Omega` and both vanish.

`tov.tex` still compiles.

## Answer

**C2 — numerical parameters.** Now stated in both files: `DOP853`, `rtol` 1e-10
and `atol` 1e-12 on the reference path, `r_min` 0.001 km, `r_max` 15.0 km,
`p_surface` 1e-10 MeV/fm^3, `P_tol` 1e-4 for identifying a first-order jump, and
`n_grid` 2000 for the fast path's uniform ln P cells. Two carry more than a
table row and both files say why: the reference tolerances are deliberately
tighter than any EoS fed to them, so the *integration* is never the limiting
error — which is what lets the reference path be what `solver_fast.py` is
validated against (§9) — and `n_grid` is a resolution rather than a tolerance,
so a first-order transition narrower than one cell is smoothed. That is the one
place the two paths can legitimately disagree, and it now lives in the document
rather than being discovered when they do.

**C8 — which rows each mode changes: N/A, and now justified rather than
silent.** `astro/tov` is not a model, has no §3 modes and no residual, so the
column has nothing to describe. Its nearest equivalent is the crust-attachment
choice, which does change what is integrated — and both files now enumerate the
four resolvable names (`BPS` and the three CompOSE tables) plus the `No` and
`personalized` paths, where before only the `.tex` listed them.

**One defect found and fixed that this ticket did not anticipate**, and it was
mine: commit `7ff8627` moved the crust tables into the package, which made both
documents' "large external data, neither shipped with the package nor tracked
in git" false and their search path incomplete — and I did not update them in
the same commit. Exactly the code/document drift the audit exists to catch,
committed by the audit's own reader. Corrected in `ad3669c`.

`tov.tex` compiles throughout.

Status: resolved.
