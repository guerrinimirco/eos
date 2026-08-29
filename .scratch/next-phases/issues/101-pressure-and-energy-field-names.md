# `P`/`eps` against `P_total`/`e_total`: one job, two names, ten models

Type: grilling
Status: resolved (2026-08-29)
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

## Resolution

**`P`, `eps`, `s`, `f` win. `P_total`, `e_total`, `s_total`, `f_total` are gone
from every point object in the package, with no alias left behind.** Seven
dataclasses in six models moved: `zl.EoSPoint`, `vmit.EoSPoint`,
`alphabag.EoSPoint`, `alphabag.CFLPoint`, `abpr.CFLPoint`, `njl.EoSPoint`,
`ccdm.EoSPoint`.

### The case for `_total`, and why it loses

`_total` has one real argument and it is not weak: the quark models' sector
blocks in `thermodynamics.py` DO carry a plain `P`, `e`, `s`, `f`, and a total
that is matter + leptons + photons + bag is genuinely a different quantity from
one sector's contribution. That is a distinction worth a name.

It loses on three counts, each checked rather than asserted.

- **The distinction is between dataclasses, not within one.** No point object
  in the package carries both a sector `e` and a total `e_total` -- checked by
  walking `dataclasses.fields` of every `_total`-carrying class in the six
  models: not one of them holds `P`, `eps`, `s`, `f` or `e` as well. `EoSPoint`
  has `e_total` and no `e`; `QuarkThermo` has `e` and no `e_total`. So at every
  site where either name is read, the type already says which it is, and the
  suffix disambiguates nothing. CLAUDE.md section 13 rule 3 names exactly this
  disease -- a name that restates its container, the same fault as a
  `thermodynamics_quarks.py` inside a quark model.
- **The engine already ruled, and both lineages already meet in it.**
  `eos/mixed/adapters.py` reads BOTH spellings: `p.P, p.eps` on the dd2/sfho/
  did/enjl branches, `p.P_total, p.e_total` on the zl/vmit/alphabag/njl/ccdm
  ones. That is the per-model branch this ticket predicted a generic caller
  would pay, already paid, in the one file section 5 designates as the coupling
  surface. And every adapter converges on the bare names on the way out --
  `P=th.P, eps=th.e` into `PhaseThermo` -- while `MixedPoint` is `P`/`eps`/`s`
  and `HybridBranch` is `P`/`eps`. The composite engine had already chosen; the
  four models were the holdouts.
- **Four more translations exist purely to cross the split**, and all four
  collapse: `notebooks/quark_eos.py` and `notebooks/hadronic_eos.py` each carry
  a `thermo(point)` helper that branches on `hasattr(point, "P")` with a
  docstring explaining the divergence; `test/test_imports.py` carries
  `_pressure_of(point)` looping over `("P", "P_total")`; and downstream,
  `nucleation/tables/qstar.py:139-142` aliases `result['P'] = result['P_total']`
  and three more -- the consumer voting for the bare names at its own boundary.

`eps` over `e` because it is the symbol every model document already uses and
it pairs with `P`; the sector blocks keep their `e`, which now reads as the
per-sector name it always was.

### What moved, and what deliberately did not

**Renamed** (fields and every read of them):

- the seven point dataclasses above, and their reads across the six models'
  `solver.py`, `api.py`, `table.py`, `responses.py`, `compute_tables.py` and
  `verify/run_full_check.py`;
- `eos/mixed/adapters.py`, five reads on the quark branches;
- `test/` -- 22 files, plus the two collapsed helpers;
- `notebooks/quark_eos.{py,ipynb}` and `notebooks/hadronic_eos.{py,ipynb}`,
  jupytext-synced, both `thermo()` helpers deleted;
- `nucleation/analysis/filters.py` lines 59, 82, 129 -- the ONLY three places
  in `nucleation` that read an `eos` point (see the sweep below);
- the five model documents that tabulate the field names
  (`zl.md`, `vmit.md`, `alphabag.md`, `abpr.md`, `ccdm.md`) and the worked
  example at `docs/STRUCTURE.md:352`.

**Not renamed, on purpose:**

- **Written table COLUMN HEADERS stay `P_total`, `e_total`, `s_total`,
  `f_total`.** That is a different surface with a different owner, and it is
  already an open fog patch on this map ("a written table's column headers are
  a format nobody has ruled on", left standing by ticket 108). `sfho/table.py`
  proves the two are separable: a bare-name model already writes `_total`
  headers through the translation dict at line 619, which is untouched. Every
  `.dat` this repository and `nucleation` have ever written still reads.
- **Local accumulators keep their names.** `P_total = quark.P + thermo_e.P` in
  `alphabag/solver.py`, and the same shape in `sfho`, `did`, `vmit` and
  `eos/mixed/thermodynamics.py`, are function-local sums that end up assigned
  into a field named `P`. Section 13 governs the vocabulary a reader crosses
  between models, not a local in a summation.
- **`eos/zlvmit/` and the two `ZLvMIT` notebooks.** Legacy, exempt by section 1;
  their `P_total` are their own dict keys, not a model point's fields.
- **`nucleation`'s own `QuarkResult`** (`nucleation/quark.py`) keeps
  `P_total`/`e_total`/`s_total`/`f_total`. It is nucleation's type, built by
  summing `eos`'s SECTOR blocks (`quark.P + thermo_e.P`), and this ticket's
  gate is one name across the ten `eos` models. Renaming it is nucleation's
  call.

### No number moved, and the baselines prove it

The baseline flattener walks `dataclasses.fields`, so the field names ARE the
store keys. **1191 keys were renamed across five `.npz` files** -- `alphabag`
168, `ccdm` 305, `njl` 421, `vmit` 126, `zl` 171 -- plus two hand-written keys
in `generate_baseline.py` (`enumeration.n1.2.f_total` -> `.f`, and `n1.5`).
`abpr.npz`, `dd2.npz`, `did.npz`, `enjl.npz`, `mixed.npz`, `sfho.npz`,
`tov.npz` and `zlvmit.npz` hold no such key and were not rewritten.

The rename was done ON THE STORED FILES -- load, re-key, `savez_compressed` --
never by regenerating. **Nothing was re-solved, so the section 12 freeze at
rtol = 1e-10 is untouched**, and that is checked rather than claimed: every
value in every rewritten file was compared with `np.array_equal(...,
equal_nan=True)` against a copy of the pre-rename file under its old name.
**1191 keys renamed, 0 value mismatches.** Teaching `test_baseline` a name
mapping was rejected -- that is the alias this ticket asked about, in the one
place section 7's single-home argument bites hardest.

### The gate

Certificate for the tree this change sits on: **`test/suite_certificates/
20260829T103554.txt`** -- HEAD `286da5f`, CPython 3.14.2 / numpy 2.3.5 /
scipy 1.17.0, verdict CLEAN, `1850 passed, 23 skipped, 0 failed` (19:08), taken
by [ticket 118](118-first-landing-measurement.md). That measures the SHA this
work started from; **it does not measure this change**, and no new certificate
was taken, because `eos/*.py` is dirty here by construction -- a landing
measurement is a property of a SHA and this change has not landed. What ran
against the working tree, on the same interpreter:

- `pytest test/zl test/vmit test/alphabag test/abpr test/njl test/ccdm
  test/mixed test/baseline test/test_imports.py` -- **1039 passed, 0 failed,
  0 skipped** (8:13). All thirteen `test/baseline` cases pass on the re-keyed
  store, which is the check that the 1191 renamed keys still find their values.
- the six models' `verify/run_full_check.py`: **zl, vmit, alphabag, abpr, njl
  PASS**. **ccdm FAILs one check, `reduction chain`, and it is NOT this
  change** -- a HEAD control (`git archive HEAD eos` into a scratch tree, run
  there) reproduces it with a byte-identical message and the same
  `max_err = 7.64e+08`.
- `docs/STRUCTURE.md`'s worked `abpr` example, run as written: `P = 224.2725`,
  `eps = 833.8153` MeV/fm^3.

That set is the blast radius: the six models whose fields moved, `test/mixed`
because `adapters.py` reads them, `test/baseline` because the store keys moved,
and `test/test_imports.py` because its dual-spelling helper was deleted. `dd2`,
`sfho`, `did` and `enjl` were not touched -- their fields were already `P`/`eps`.

### The consumer sweep, and the two things it found

Swept `notebooks/` and `../nucleation` on `paper-release`, `.py` and `.ipynb`,
including function-local imports. After the rename, `grep` for
`\.P_total|\.e_total|\.s_total|\.f_total|P_total=|...` across both repositories
returns **nothing** outside `zlvmit`.

**`nucleation/analysis/filters.py` is the only file in `nucleation` that reads
an `eos` point.** Everything else imports `eos.general.*`, `eos.<model>.
thermodynamics` (sector blocks, `P`/`e`/`s` -- untouched), `eos.<model>.table`
(rows and `.dat` columns -- untouched), `eos.<model>.parameters`, or
`eos.astro.tov` + `EOSTable_for_TOV` (untouched). Verified by importing all 25
`nucleation` modules: **25/25 import.**

Two pre-existing defects surfaced, neither caused here, both left for a ticket:

1. **`nucleation/analysis/filters.py` cannot run against `eos` today.** Its
   three solver calls pass `include_photons=`, `include_gluons=`,
   `include_electrons=`, kwargs that no longer exist -- `solve_cfl` is now
   `(par, n_B, T, Delta0, flags, initial_guess=None)`. Calling
   `cfl_eos_at_params` raises `TypeError` immediately. The file was ALREADY
   dirty in `nucleation`'s working tree when this session opened it (a
   par-first/exception-narrowing edit in progress, uncommitted), so the fix is
   somebody's live work and was not taken over here; this ticket's contribution
   to that file is exactly the three renamed lines. **This is the ticket-24
   blind spot with a live instance in it.**
2. **`nucleation`'s own suite cannot collect**: all 9 test modules die on
   `ImportError: Error importing numpy` under pytest (numpy imports fine
   outside it). **Identical under a HEAD-`eos` control**, so it is an
   environment defect in that checkout, not a consequence of this rename.

Because that suite cannot run, the three renamed `nucleation` lines were gated
directly instead: `solve_cfl(par, 0.8, 0.0, 80.0, SpeciesFlags())` returns
`P = 161.7844`, `eps = 845.8648`, `s = 0`, `f = 845.8648`, and
`solve_beta_eq_neutrinoless` returns `P = 147.2606`, `eps = 890.3843` -- the
four field names the file now reads, all present and finite.
