# vmit.md and vmit.tex to §11 standard

Type: task
Status: resolved
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

## Resolution

**Both files rewritten from the code and the literature, as the ticket
directed — `vmit.md` 82 → 707 lines, `vmit.tex` 265 → 992, carrying the same
information. `.tex` compiles clean in three passes, 10 pages, no undefined
citation or reference. No `.py` touched, so the suite cannot have moved.**

**The ordering defect is worse than "stated and wrong", and that is what made
it invisible.** There are TWO unknown layouts, and each file had generalised a
different one:

    beta_eq_neutrinoless       (mu_u, mu_d, mu_s, mu_e, n_u, n_d, n_s)
    beta_eq_neutrino_trapped   (mu_u, mu_d, mu_s, mu_e, mu_nue, n_u, n_d, n_s)
    fixed_YC / fixed_YC_YS     (mu_u, mu_d, mu_s, n_u, n_d, n_s, mu_e)

`vmit.md` said "`mu_e` in the fourth slot, before the densities, not appended
after them" — true of the beta modes, false of the fixed-fraction ones.
`vmit.tex` said the vector is `(mu_u, mu_d, mu_s, n_u, n_d, n_s)` "extended by
`mu_e` where a lepton condition is present" — true of the fixed-fraction modes,
false of the beta ones. **Each file was right about the half the other was
wrong about**, which is why neither reads as obviously broken and why the
disagreement survived. Both now carry the layout table and every residual row
by row in solver order.

**A second ordering fact neither file could have carried, because neither
enumerated the rows:** `R6` and `R7` SWAP between the two beta modes.
`beta_eq_neutrinoless` assembles the beta condition then the strangeness one;
`beta_eq_neutrino_trapped` assembles them the other way round. Immaterial to
the solution, not immaterial to anyone hand-writing a Jacobian, so it is stated
as the code has it.

**Four gaps beyond the ones the audit named.**

- **The finite-T integrals were missing their `(hbar c)^3`.** `vmit.tex`'s
  Eqs. for `n_q`, `P_q`, `eps_q` had the `g/2pi^2` prefactor and no conversion,
  so they return MeV^3 where the package's every boundary is fm-based. Same
  defect class as the sfho field sources ticket 27 fixed.
- **The totals equation omitted the leptons and photons entirely** — it summed
  the quark sector and stopped, in a model where both beta modes carry an
  electron gas by construction.
- **No `T = 0` forms at all**, in a model whose strange-quark ONSET is a `T = 0`
  threshold.
- **Nothing on what a solved point returns**, the table row keys, the API
  surface, the cold starts or the verify suite.

**Three claims measured against the code rather than asserted.**

- `B/(hbar c)^3 = 136.63 MeV/fm^3` at `B4 = 180` — computed, not estimated.
- The light flavours are ultra-relativistic to **5e-4**, not the 3e-4 first
  drafted: at `mu_eff = 300 MeV` the `m = 5 MeV` gas differs from the massless
  one by 4.17e-4 in `n`, at 400 MeV by 2.34e-4.
- The `verify/` suite runs **EIGHT** checks, not the seven its own module
  docstring enumerates — `bag / vector signs` is real, passes at 0.00e+00, and
  is absent from the docstring's numbered list. Documents say eight. The
  docstring is a code defect, reported not fixed.

**The carried-in ticket 27 item was already discharged.** `vmit.md:81`'s claim
that `eos_response` is unimplemented is not in the file that arrived — the
current line 81 is the "Not implemented: muons" ledger. Commit `2844a9a`
("seven places a document disagreed with the code") took it. Nothing to do; the
rewritten pair states what `eos_response` does implement (`frozen='equilibrium'`
only, both derivatives by central differences) and what raises.

**`n_s` is discharged with its reason, and the answer is not §11's identity.**
vMIT has no scalar sector — current masses, no gap equation, no `m*` — so
`n_s = (eps - 3P)/m*` has no meaning here and the shared integrals' scalar
density is computed and discarded. `n_s` in this package is the strange-quark
NUMBER density, which is what the charge maps consume. Stated explicitly in
both files, since the symbol collides with the repository-wide one.

`docs/eos.bib` untouched — every citation was already there. The `.md`'s
reference list was corrected against it: the first draft had `Gomes2019` as
"Gomes, Dexheimer, Han, Schramm" (it is Gomes, Char, Schramm) and
`Constantinou2021` as "Constantinou, Zhao, Han, Prakash" (it is Constantinou,
Han, Jaikumar, Prakash).

Verify suite re-run after the rewrite: **PASS, all eight**, worst 3.52e-11.
