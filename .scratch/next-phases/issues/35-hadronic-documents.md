# zl, sfho and dd2 document pairs to §11 standard

Type: task
Status: resolved
Blocked by: 09
Parent: ../map.md

## Question

Three pairs where the `.tex` is strong and the `.md` lags, so the work is mostly
bringing the `.md` to parity — then closing the remaining `.tex` cells.

| pair | .md | .tex |
|---|---|---|
| zl | 6/14 | **14/14 — passes** |
| sfho | 9/16 | 15/16 |
| dd2 | 9/16 | 14/16 |

**`sfho.md` and `dd2.md` each say "the closed forms are in `<model>.tex`
Eq. (T0)".** Under [ticket 09](09-tex-or-md.md)'s ruling that is now a defect:
each file carries the same information, so the closed forms are written out in
both.

Two items live **only** in the `.md` and must survive the merge, not be
overwritten by the stronger `.tex`: SFHo's `hc³` field-source correction (the
`.tex` is dimensionally wrong without it) and DID-style "Not implemented"
ledgers. `zl.tex` passes outright and is the model for what the others should
look like.

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

**Carried in from [ticket 27](27-document-defects.md)**: the three-flavour
`mu = 0` thermal-neutrino gas that `eos/sfho/solver.py:523-527` adds to `P`,
`eps` and `s` appears in **neither** sfho document. It is an omission rather
than an error, which is why ticket 27 left it here — but §11 requires every
contribution to the totals to be written out, so both files gain it.

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

## Added by ticket 45 (the sfho renames), binding on the `sfho` pair

`sfho`'s five `get_sfho*` free functions are gone. The published sets are now
reached through `Parameters.default()` and `Parameters.named(name)`, and the
keys are public API the document has to state — that was the ticket's own
condition for settling them. They are the `name` field each set already
carried, so `Parameters.named(p.name)` round-trips:

    SFHo_Nucleonic   nucleons only, the CompOSE SFHo table; what default()
                     returns, what nmp.py reports the published NMPs against,
                     and what test/baseline is frozen at.
    SFHoY_Fortin     + hyperons, scaled vector couplings (y = 1.5), scalar
                     couplings from U_Y (Fortin et al. 2017).
    SFHoY*_Fortin    + hyperons, SU(6) vector couplings (same reference).
    SFHo_2fam_phi    + hyperons and Deltas, SU(6) vectors, hyperons coupled
                     to phi.
    SFHo_2fam        as SFHo_2fam_phi with g_phi = 0 for every strange baryon.

`eos.sfho.PUBLISHED_SETS` maps those keys to builders (not instances — a
`Parameters` holds mutable coupling maps, so a shared instance would be the
global mutable state §6 forbids).

Two more names the pair must not describe by their old spelling: the §5
phase-adapter surface is **`thermo_from_mu`** (was `thermo_at_potentials`) and
the layer beneath it, which additionally takes the solved mean fields, is
**`thermo_from_fields`** (was `thermo_from_mu`). `dd2`'s half of that same
ruling has NOT landed — its surface is still `thermo_at_potentials`, now on
[ticket 48](48-rename-did-surface.md) — so the `dd2` pair should describe the
function it actually has today, not sfho's new spelling.

## Resolution

**All three pairs done, six files, no `.py` touched, all three `.tex` compiling
clean.**

    zl.md    167 -> 336     zl.tex   485 -> 628   (7 pages)
    dd2.md   117 -> 337     dd2.tex  530 -> 761   (8 pages)
    sfho.md  188 -> 313     sfho.tex 685 -> 800   (8 pages)

Commits `421996b` (zl), `ced5431` (dd2), `38ddf91` (sfho).

**Both "the closed forms are in `<model>.tex` Eq. (T0)" defects are closed**, and
the dd2 forms were verified against `kinetic_thermo` rather than transcribed:
it returns NATURAL units, so the fm-based forms both documents state differ from
it by exactly `(hbar c)^3` — agreement to 7e-16 on all four quantities once
reconciled.

**A defect in the file the audit passed 14/14.** `zl.tex`'s neutrality row is
written `n_e - n_p` and presented as one shared row `r_5`. The code has **two
signs**: `n_C - n_e` in both beta-equilibrium modes, `n_e - n_C` in `fixed_YC`.
The root is unchanged; the residual is not, and §11 asks for the rows as
assembled. Both files now give each mode its own. Note this is the same shape
as vMIT's ([ticket 33](33-vmit-documents.md)), where the two beta modes also
swap `R6`/`R7` — a pattern worth expecting in the remaining models.

**dd2's two Partial `.tex` cells were one cause.** The thermal meson sector had
no Bose thermodynamics anywhere in the document — three `mu*_j` and then "its
P, eps, s join the totals". Both files now carry `n_j`/`P_j`/`eps_j`/`s_j`, the
eighteen species with masses, charges, strangeness and degeneracies, and the
condensation diagnostic with its refusal at `|mu*|/m >= 1` and the reason
capping `mu*` returns a WRONG state rather than an approximate one. Also added
to both, all absent: the `eos_response` set, the hyperon/Delta/lepton masses as
numbers, the four rows of the reduced nucleon-only beta system, and the
phase-adapter residual `did.tex` documented and `dd2.tex` did not.

**Two claims the code overturned in sfho, one of them mine.** I wrote that the
trapped mode plus `thermal_neutrinos` double-counts `nu_e`; it does not —
`solver.py` REFUSES the combination and names that exact reason. Measured
instead of inferred, and the factor where the gas IS added is exactly
3.000000 of the single-flavour gas.

Second — and this one I got wrong twice before it was right, caught by the
session holding [ticket 45](45-rename-sfho.md): **the refusal is a DEFECT, not a
deferred design choice.** `CLAUDE.md:176` already says `thermal_neutrinos` is
meaningful alongside `beta_eq_neutrino_trapped` and that a model MUST NOT RAISE
on the combination — landed by [ticket 22](22-phase5-claudemd.md) — so my first
correction, which called it a ledgered gap and pointed at `docs/DEFERRED.md`,
had a document arguing against the current specification. The pointer also had
no target: `grep` finds no such entry, correctly, because
[ticket 11](11-conformance-triage.md):135 rules the row **(b) + (a)** and not
(c), which is why [ticket 55](55-deferred-ledger.md) was never asked to ledger
it. [Ticket 54](54-signature-corrections.md) item 5 deletes the raise in `sfho`
and `did`; the three that succeed (`njl`, `ccdm`, `enjl`) are the conformant
ones. Fixed in `463278a`: both files name it as a defect due for removal and
say what changes when it goes — the trapped mode gains the two remaining
flavours, not three.

**Carried-in ticket 27 item discharged**: the three-flavour gas is now in both
sfho files. **Both `.md`-only items survived the merge**: SFHo's `hc^3` field
sources and the "not implemented" ledgers are intact.

## The sfho half was written against another session's uncommitted tree

Worth recording because it is a precedent, not an accident. Session `32a0f093`
held [ticket 45](45-rename-sfho.md) live in this checkout while this ticket
ran, and 45 renames the very symbols an sfho document must state as public API.
It messaged the final names; I verified all of them against the live tree
myself — `PUBLISHED_SETS` exported from both the package and
`parameters.py`, the five `named()` keys, `default().name`, and
`thermo_from_mu`'s full signature — re-checked immediately before committing,
and told that session the documents go stale if any of it moves.

**Its correction changed the dd2 half too, and saved a wrong document.**
Ticket 44 carried the `thermo_at_potentials -> thermo_from_mu` instruction and
never applied it: `eos/dd2/thermodynamics.py:571` is still
`thermo_at_potentials` and there is no `thermo_from_mu` in the package
(verified independently). The dd2 pair therefore names the function dd2
actually has and flags it as the outstanding one of the three; sfho's pair says
the same from its side. That rename is [ticket 48](48-rename-did-surface.md)'s,
now widened to cover dd2.

## Reported, not fixed

`eos/vmit/verify/run_full_check.py` runs eight checks and its module docstring
enumerates seven — `bag / vector signs` is real, passes, and is missing from the
numbered list. Stage 7 material. (Found under ticket 33; session `32a0f093` is
also carrying it in its own resolution so it survives either session ending.)
