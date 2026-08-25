# alphabag, abpr, enjl and mixed document pairs to §11 standard

Type: task
Status: resolved
Blocked by: 09
Parent: ../map.md

## Question

Four pairs, all partial, none catastrophic.

| pair | .md | .tex |
|---|---|---|
| alphabag | 13/16 | **16/16 — passes** |
| abpr | 11/14 | 13/14 |
| enjl | 10/16 | 14/16 |
| mixed | 9/14 | 11/14 |

`alphabag.tex` is one of only two documents in the repo that pass §11 outright
and is the template for the rest.

Items living **only** in the `.md`, to survive the merge: ENJL's **correct**
`mu_C` unknown (the `.tex` prints `mu_e` there — the code decides, so the `.md`
is right and the `.tex` is the defect), `mixed`'s live `fixed Y_C + Y_Le`
refusal, and ABPR's layout map.

For `mixed`, two extra obligations: the phase-adapter contract and the
transition observables §5 requires a composite engine to return (`n_onset`,
`n_offset`, `chi`, the per-phase charge decomposition), and **the photon
treatment** — [ticket 29](29-mixed-species-flags.md) found the engine carries
photons unconditionally at T > 0 and neither document says so.

`mixed.tex`'s undefined `\tmuB` macro is [ticket 27](27-document-defects.md)'s,
but it is in the equations this ticket rewrites — coordinate with it.

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

**Carried in from [ticket 27](27-document-defects.md)**: both `abpr`
documents give the *code* names `ms` and `Delta` in the parameter table where
the mathematics wants `m_s` and `Delta_0`. §13's rule that a name says what it
is applies to the documents too — a reader reproducing the model from the page
should not have to know the identifier.

## Added by ticket 10 (rename approvals)

**Name the phase-adapter surface while writing `mixed.tex`.** Ticket 07 found the
same job under two names: `thermo_at_potentials` (dd2, sfho, did) against
`thermo_from_mu` (the other seven models). §13's vocabulary lists
`thermo_from_mu`, 7–3 in its favour — but dd2, sfho and did are not drifting,
they carry BOTH, at two layers: `thermo_from_mu` low, `thermo_at_potentials` on
the phase-adapter surface (`did/thermodynamics.py:358` against `:542`).

[Ticket 10](10-rename-approvals.md) deferred it here rather than renaming
blindly, because the upper layer wants a name that names the §5 contract —
"(baryon potential, mu_C, mu_S, T) -> a `PhaseThermo` block" — and picking that
name while `mixed.tex` is still being written risks naming it twice. Settle the
name in the document, then rename the three models to match.

## Answer

**Eight documents, four pairs, all four `.md` files rewritten to carry what
their `.tex` carries. All four `.tex` compile clean with no undefined citation
or reference** (alphabag 8 pages, abpr 7, mixed 9, enjl 15), and no `.py` file
was touched, so the suite is unmoved by construction.

**alphabag** (13/16, 16/16). The `.md` gained the massive-flavour Fermi
integrals at `T > 0` and their `T = 0` closed forms, the returned-field tables
against `EoSPoint` and `CFLPoint`, the photon formula, and the residual rows
stated as *the order `solver.py` assembles them* rather than as a coincidence.
Both files gained the API surface — the one gap in the repository's only other
passing document — including `eos_response`'s single freeze and its two
quantities, which `api.py:175` has returned all along. One code correction:
`CFLPoint` drops `n_e` and `n_nu` but **keeps** `mu_e`, `mu_nu`, `Y_e` and
`Y_nu` at zero, where its own docstring says `Y_e` is absent.

**abpr** (11/14, 13/14). The carried-in defect was in both files and was worse
than a style slip: the parameter tables gave the code names as `ms` and
`Delta`, and `parameters.py:57-61` declares `m_s` and `Delta0`, so the one
column a reader would copy from was wrong. Corrected, with the symbol column
now `m_s` / `Delta_0` and a line saying why the equations still write `Delta`.
The `.md` gained the single-flavour thermodynamics the whole expansion is built
from (`k_F`, the massive `T = 0` closed forms, the massless limit the
`3 a4 mu^4/4 pi^2` term is three copies of), the `CFLPoint` returns table, the
closed-form inverse maps with the Descartes-rule argument, and the measured
alphabag comparison. `eos_response`'s paragraph now names the key it returns,
`cs2_isothermal`, rather than a bare `c_s^2`.

**mixed** (9/14, 11/14) — the pair with the extra obligations, and both were
real. The `PhaseThermo` block is now written out field by field in both files;
the five that were missing include `mu_dot_n`, which is what the engine's own
1e-8 Euler check consumes, so the document had been describing a contract that
could not have supported its own headline invariant. The numerical constants
that were named qualitatively are now given: `n_scale = max(n_B, 0.01)`,
`mu_scale = 100.0` MeV, the mechanical row's `max(|P^H_eff|, |P^Q_eff|, 1.0)`,
`_P_ROUNDOFF = 1e-12`, `n_probe = 12`, `max_refine = 2`, `max_bisect = 6`,
`MAX_WALK = 64`. Added: the `Window` record with its four `reason` labels, the
`Result` record, the `eos_point`/`eos_table`/`eos_response` surface with both
implemented freezes and the `nan`-outside-the-window rule that makes `cs2_eq`
and `C_V` meaningful, the third mode refusal (`L_e` GLOBAL with `C` conserved,
`solver.py:260-263`) and the two phase-level ones, and the pointer to the
sibling parameter documents.

**Two things the code overturned in `mixed`.** The `.md`'s asserted row order
was wrong in the way the audit predicted: `solver.py:512-521` emits the
eta-shifted charge-matching row **immediately after** the C-average row, before
`S` and `L_e`. And [ticket 29](29-mixed-species-flags.md)'s finding is now in
both documents: `thermodynamics.py:84` adds a photon gas whenever `T > 0`
without consulting any flag — the engine has no `SpeciesFlags` of its own — so
a mixed point at finite temperature always carries photons and a caller cannot
turn them off. Stated as the known gap it is, in both files and in both "not
implemented" lists.

**enjl** (10/16, 14/16) — the largest `.md` gap in this ticket. It failed §11's
naming-not-defining rule in five places at once, and the failure compounded:
`alpha_S`, `Gamma_omega` and `Gamma_rho` were used throughout with only their
nine fitted coefficients listed, so the primed couplings in `Sigma^R` were
derivatives of undefined functions; `E0` was a number with no expression; the
thermal sectors were named with no formulas; and **the quantum-number table was
absent**, leaving `N^q_i`, `tau_i`, `q_i`, `C_i`, `S_i` and `B_i` unvalued, so
`J_rho`, `Sigma^R_b` and residual rows 4, 10 and 11 could not be evaluated from
the page at all. All written out, with the closed forms and their analytic
derivatives, plus the `T = 0` closed form for `P`, `E(k)`, an explicit `f_+`,
the lepton masses, the total entropy assembly, the forward map
`nu_i = mu_i - Delta_i`, and the returns tables.

**The `n_s` question is settled, and the answer is not §11's identity.** For
this model `eps - 3P = M n_s` holds **identically and per species for the
medium terms** — it follows from the closed forms term by term, and was
verified to round-off here for a baryon and for a quark with the cut-off off
(ratio 1.00000000 both times). It **fails** for the quarks as they are actually
used (ratio 3.897), for two independent reasons: the Dirac-sea subtraction
removes `eps^vac` from `eps` and `n_s^vac` from `n_s` but nothing from `P`, and
`eps^vac != M n_s^vac`; and the totals carry condensate, vector, rearrangement
and thermal terms the trace form knows nothing about. So ENJL integrates `n_s`
and the trace identity is not a route to it — now stated in both files, with
the same discharge for `alphabag`, `abpr` and `mixed`, none of which has a
scalar density at all.

Also defined `chi = n_B^Q/n_B` — a delivered `beta_row` column that appeared in
neither document — and removed the duplicated `kF` row from `enjl.tex`'s
returns table.

**The name ticket 10 deferred here is settled.** The §5 phase-adapter surface
is **`thermo_from_mu`**, in every model, and a lower evaluation layer that
additionally takes the solved mean fields is **`thermo_from_fields`**. The
reasoning is in `mixed.tex`: 7 of the 10 models already spell the surface that
way and §13's vocabulary lists it, but the deciding argument is that the three
holdouts are not drifting — `dd2`, `sfho` and `did` carry TWO layers, and the
lower one takes the fields, which is exactly what a name should say. Naming it
`thermo_from_fields` removes the one-job-two-names split instead of freezing
it. **The renames themselves were deliberately not done here**: `dd2` and
`sfho` are tickets [44](44-rename-dd2.md) and [45](45-rename-sfho.md), which
now carry the ruling and the collision warning; `did` had no ticket, since
[42](42-rename-internal.md) closed before the name was settled, so it is new
ticket [48](48-rename-did-surface.md). Doing them here would also have collided
with the session working `eos/dd2` in this checkout.

**One code defect reported and not fixed** (map hard rule): `eos.enjl.api`'s
`eos_response` raise message says "this is a T = 0 model". The *modes* are not
— all four close at any `T >= 0`, which `DEFERRED.md:1070` confirms and the
`.tex` has said since the finite-T work. What is `T = 0`-only is the
construction, and the real response gap is the branch statement. Both documents
now say so; the message is Stage 7 report material.

**0 added failures, structurally.** `git diff --name-only` over this ticket's
commits returns eight documents and nothing else — no `.py` file was touched —
and a full-suite run would have been worthless anyway with a parallel session
editing `eos/dd2` in this checkout.

Status: resolved.
