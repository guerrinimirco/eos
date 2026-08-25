# vmit.md and vmit.tex to §11 standard

Type: task
Status: open
Blocked by: 09
Parent: ../map.md

## Question

**The only model where both files fail** — 7/16 and 9/16.

`grep` for `180`, `0.2`, `150` returns zero hits in either file: not one
parameter value, and no reference. Neither enumerates a residual, and **the
unknown ordering both give contradicts** `solve_vmit_beta_eq` and
`solve_vmit_trapped_neutrinos` — an order stated and wrong, which §11 treats as
worse than absent.

`vmit.md:81` also claims `eos_response` is unimplemented; `eos/vmit/api.py:167`
implements it. That stale claim is [ticket 27](27-document-defects.md)'s, but it
is the same paragraph this ticket rewrites, so take it here.

Because neither file is a usable source, this pair is written from the code and
the literature rather than by porting one into the other.

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

## Claim released 2026-08-25

Marked `claimed` in commit 359f76f with **no assignee**, and no session has held
it since: `ListAgents` shows no holder, and the documents are untouched. Measured
rather than assumed — the pairs finished under tickets 30/31/32/36 run 860–1205
lines in the `.md`, and these are all still at pre-refactor summary length:

    vmit.md   82    vmit.tex  265 (Aug 10, untouched)
    zl.md    167    zl.tex    485
    sfho.md  188    sfho.tex  685
    dd2.md   117    dd2.tex   530

So the work was never started, and the stale claim was hiding it from the
frontier query. Back to `open`. Re-claim before starting.
