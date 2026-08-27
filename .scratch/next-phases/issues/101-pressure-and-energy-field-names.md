# `P`/`eps` against `P_total`/`e_total`: one job, two names, ten models

Type: grilling
Status: open
Blocked by: -
Parent: ../map.md

## Question

Recorded as finding 1 of [ticket 99](99-quark-ea-at-zero-pressure.md) and
deferred out of it. A solved point's pressure and energy density are called

    P, eps            dd2, sfho, did, enjl
    P_total, e_total  zl, vmit, alphabag, abpr, njl, ccdm

Six models against four, for the same two quantities. CLAUDE.md section 13:
"The same job carries the same name in every model." There is no physics in
the split -- it is the hadronic models against the quark ones, which is where
the two first-generation lineages met.

### Ticket 99 routed around it rather than paying it

99 needed to read P off a point in five models from one shared locator. It
avoided the name entirely: `eos.general.zero_pressure.locate_zero_pressure`
takes the STATE AS A CALLABLE, `point_at(n_B) -> (P, E_per_A, mu_B, Y_S,
mu_S)`, and each model's `zero_pressure_point` supplies it in a three-line
adapter that names its own fields. That was the right call there for a second
reason -- a callable is what keeps the locator above every model in the
layering (section 1) -- so the divergence cost that ticket nothing, and it
remains unpaid.

It will not always be free. Any future caller that wants to read a solved
point generically -- a table writer, a response-function driver, a sampler's
scoring loop -- pays it again, and pays it as a per-model branch rather than
as an adapter.

### What has to be decided

- Which name wins. `eps` matches the symbol every document uses and pairs with
  `P`; `e_total` says "the total, leptons and bag included" against a
  per-sector `e`, which is a real distinction in the quark models where
  `thermo_from_mu` returns a sector block called `e`.
- Whether the losing name survives as an alias or is removed. Every baseline
  `.npz` key, every notebook and `eos/mixed`'s row assembly read these fields,
  so a removal is wide and an alias is a second home section 7 argues against.
- Whether `s_total` / `s` and `f_total` / `f` move with them; they have the
  same split.
- **Whether any number moves.** A pure rename should move none, which makes
  the baselines the check rather than the obstacle -- but the baseline `.npz`
  files store COLUMN NAMES, so a rename that reaches `table.py` renames keys
  and the comparison has to be taught the mapping or the files regenerated.

## Gate

- One name per quantity across all ten models, or a stated reason the split is
  physics.
- No number moves; a key that is renamed is named in the resolution.
