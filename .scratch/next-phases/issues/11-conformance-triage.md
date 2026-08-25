# Sort every failing conformance row into fix-code, fix-CLAUDE.md, or defer

Type: grilling
Status: resolved
Assignee: mirco (session)
Blocked by: 02, 08, 09
Parent: ../map.md

## Question

Ticket 08 is resolved: 24 Fail and 25 Ambiguous cells over 136, already sorted
into 22 (a) / 11 (b) / 12 (c) with file:line evidence in
[conformance-table.md](../research/conformance-table.md). This ticket is the
human ruling on that sort, not a re-derivation of it.

**One (a)-class finding is already carried out of this pile**: the
`photons=False` silent-ignore in dd2 and mixed, which changes numbers and so gets
its own gate at [ticket 28](28-photons-silent-ignore.md). The other 21 stay here.

**The auditor's own note, worth weighing before ruling:** `docs/DEFERRED.md` is
unusually thorough — most of what a naive audit would flag is already recorded
there with reasoning and measurements. That is why the (c) pile is only 12
entries, and why the real work is the (a) fixes, several of which are one-liners.

Take ticket 08's table and put every row that **fails or is ambiguous** to the
user, grouped as:

- **(a)** the code is wrong and should be fixed
- **(b)** `CLAUDE.md` describes a target the refactor settled differently, and the
  document should change
- **(c)** genuinely deferred, and belongs in `docs/DEFERRED.md`

Two rows are already known to be live and are decided elsewhere — carry their
rulings in rather than re-litigating: §11's "one usage notebook per model"
(ticket 02) and §11's mandated `.tex` (ticket 09).

[Ticket 07](07-naming-sweep.md) surfaced §5 layout rows the conformance table may
not repeat — carry them in as findings to be triaged, not re-derived:

- `eos/abpr` has no `table.py` though it carries `TableResult`, `cfl_row` and an
  `eos_table` with the full progress dictionary (`api.py:112,127,146`), and
  `response_at_mu` sits in `solver.py:350` rather than a `responses.py`.
- `eos/vmit/compute_tables.py` is kept by an existing `DEFERRED.md` ruling, but
  its three package-repeating symbol names are not covered by that ruling.
- `docs/DEFERRED.md:320` records the vmit §13 conversion as "DONE"; it is not.
  A stale ledger entry is itself a (c)-class row.

Nothing is edited here. This ticket produces the ruling; ticket 22 applies the
(b)-class edits as a `CLAUDE.md` diff, the (a)-class fixes become their own work,
and the (c)-class entries land in `docs/DEFERRED.md`.

Any (a)-class fix that could move a number must say which §12 golden reference it
is checked against.

## Answer

Every Fail and Ambiguous row is ruled. **Six were already discharged** between the
audit and this session and are recorded as such rather than re-ruled; one was
re-scoped. Of the 35 live rows: **23 (a)**, **12 (b)**, **10 (c)** — several rows
split across two classes, which is why the totals exceed 35.

### Discharged since the audit — no ruling needed

| # | audit finding | state at HEAD |
|---|---|---|
| 1, 38 | `dd2/notebook_api.py` imports `astro/`, `sfho`, matplotlib | file deleted by [ticket 03](03-stage0-removals.md); import buckets C and F(dd2) are now empty |
| 5 | dd2 `photons=False` silently ignored | fixed by [ticket 28](28-photons-silent-ignore.md) |
| 40 | `eos/zl` has no `nmp.py` | forward map added by [ticket 26](26-zl-nmp.md), inverse raises |
| 35 (2 of 3) | `VMITEOSResult`, `VMITThermo` | renamed by [ticket 43](43-rename-vmit.md) |
| 22 (half) | `DEFERRED.md:328` "every model's dataclass is `Parameters`" | vmit half corrected by [ticket 10](10-rename-approvals.md) |

**Re-scoped, not discharged:** finding 6 (mixed `photons`) is not dd2's bug with a
different file name — `eos/mixed` has no `species.py` at all, so there is no flag
to ignore. It is a design question and lives at [ticket 29](29-mixed-species-flags.md).

**Carried in, not re-litigated:** §11's one-notebook-per-model is **(b)**, §11
already amended ([ticket 02](02-notebook-grouping.md)); §11's mandated `.tex`
**stands unchanged** ([ticket 09](09-tex-or-md.md)).

Every remaining row was re-verified against the working tree before ruling —
`fracs` still drops fixed fractions at `dd2/table.py:275` and `sfho/table.py:333`,
`mixed/verify` still imports `backends/` at module scope, `dd2/solver.py:880`
still imports upward from `table.py`, `ccdm` is still absent from
`test/test_imports.py:76 MODEL_PACKAGES`, `ccdm/verify` still has zero hits for
`cs2|sound|monoton`, and all seven `raise RuntimeError` sites in `eos_response`
and the `SnB` path are present.

### The ruling

| # | row | class | goes to |
|---|---|:-:|---|
| 2 | `astro/gmode` imports `dd2.solver` and `mixed.responses` | **(c)** now + **(a)** | ledger + [53](53-gmode-contract.md) |
| 3 | three `verify/` suites reach sideways | **(b)** | [22](22-phase5-claudemd.md) |
| 4 | `ccdm` absent from the layering gate | **(a)** | [50](50-mechanical-fixes.md) |
| 7 | `fracs` drops the *fixed* fractions, dd2 + sfho | **(a)** | [50](50-mechanical-fixes.md) |
| 8 | `zl.thermo_from_n(n_B, Y_C, ...)` takes a mode's fraction | **(a)** | [54](54-signature-corrections.md) |
| 9 | dd2 parameter classmethods force a deferred solver import | **(a)** | [50](50-mechanical-fixes.md) |
| 10 | `dd2/solver.py:880` imports upward from `table.py` | **(a)** | [50](50-mechanical-fixes.md) |
| 10b | needless *downward* deferred imports, 11 sites | **(c)** | [55](55-deferred-ledger.md) |
| 11 | `mixed/backends/` is not deletable | **(a)** | [50](50-mechanical-fixes.md) |
| 12 | `alphabag/solver.py` re-derives `quark_charges` ×5 | **(a)** | [50](50-mechanical-fixes.md) |
| 13 | second `quark_charges` in `mixed/charges.py` | **(a)** | [50](50-mechanical-fixes.md) |
| 14 | `enjl` keeps local quantum-number tables | **(b)** | [22](22-phase5-claudemd.md) |
| 14b | `sfho` writes the species-potential map inline | **(a)** | [45](45-rename-sfho.md) |
| 15 | `mode` defaults in njl, ccdm, enjl | **(a)** | [54](54-signature-corrections.md) |
| 15b | `mode` defaults in abpr (one mode exists) | **(b)** | [22](22-phase5-claudemd.md) |
| 16a | `Y_p` in `dd2.eos_response`'s signature | **(b)** | [22](22-phase5-claudemd.md) |
| 16b | `leptons` smuggled through `**conditions`, 3 models | **(a)** | [54](54-signature-corrections.md) |
| 17a | `TC_COEFF` has no override path | **(a)** | [54](54-signature-corrections.md) |
| 17a' | gmode weak constants, `M_PI` duplicating `general/particles.py` | **(a)** | [53](53-gmode-contract.md) |
| 17b | `_CHIRAL_SPLIT`, RNS surface constants, legacy `B4` | **(c)** | [55](55-deferred-ledger.md) |
| 18 | `SnB=` raises out of `eos_point` in njl, ccdm | **(a)** | [49](49-nonconvergence-return.md) |
| 19 | `eos_response` raises in zl, vmit, alphabag, mixed, abpr | **(a)** | [49](49-nonconvergence-return.md) |
| 20 | the one unbounded loop, `general/fermi_integrals.py:519` | **(c)** + fix | [52](52-general-t0-integrals.md) |
| 21 | `sfho`, `zl` parameter dataclasses not frozen | **(c)** | [55](55-deferred-ledger.md) |
| 22b | `DEFERRED.md` dd2 `Parametrization` half | — | [44](44-rename-dd2.md) |
| 23 | `abpr`'s docstring claims array arithmetic it does not do | **(a)** | [50](50-mechanical-fixes.md) |
| 24 | dd2 re-derives the T = 0 Fermi gas | **(a)**, in `general/` | [52](52-general-t0-integrals.md) |
| 25 | dd2 verify: no free energy, no rearrangement | **(a)** | [51](51-verify-invariants.md) |
| 26 | mixed verify: same | **(a)** | [51](51-verify-invariants.md) |
| 27 | ccdm verify: no causality check at all | **(a)** | [51](51-verify-invariants.md) |
| 28 | njl causality gated behind `--sound` | **(a)** | [51](51-verify-invariants.md) |
| 29 | who owes the P-monotonicity delivery gate | **(b)** + **(a)** | [22](22-phase5-claudemd.md) + [51](51-verify-invariants.md) |
| 30 | `astro/tov` has no `verify/` | **(c)** | [55](55-deferred-ledger.md) |
| 31 | does `general/` earn a `verify/` — **ruled yes** | **(b)** | [22](22-phase5-claudemd.md) + [21](21-phase5-structure.md) |
| 32 | `abpr` refuses all four modes, unrecorded | **(c)** | [55](55-deferred-ledger.md) |
| 33 | `mixed`'s `Y_Lmu` refusal, unrecorded | **(c)** | [55](55-deferred-ledger.md) |
| 34 | `zlvmit`'s exemption: API only, or documents and tests too | **(b)** | [22](22-phase5-claudemd.md) |
| 35b | `VMITTableSettings`, frozen by ticket 10 | **(c)** | [55](55-deferred-ledger.md) |
| 36 | `general/thermodynamics_leptons.py`, the one suffixed file | **(b)** | [22](22-phase5-claudemd.md) |
| 37 | `output/public/` does not exist | **(c)** | [55](55-deferred-ledger.md) |
| 39 | CLAUDE.md's model lists are stale; `ccdm` appears nowhere | **(b)** | [22](22-phase5-claudemd.md) |
| 41 | the §10 `rcParams` grep cannot tell an assignment from a sentence | **(b)** | [22](22-phase5-claudemd.md) |
| §3-i | the fifth mode name `cfl` is declared nowhere | **(b)** | [22](22-phase5-claudemd.md) |
| §3-ii | `thermal_neutrinos` + trapped: five models, two answers | **(b)** + **(a)** | [22](22-phase5-claudemd.md) + [54](54-signature-corrections.md) |

### The six rulings that needed argument

**1. The gmode breach is ledgered today and contracted separately.** Amending §1
to name gmode as a second exception was the cheap answer and is the wrong one:
the astro half of §1 was tightened *because* this ambiguity existed, so blessing
it re-creates the ambiguity under a new name — and it would make gmode DD2-only
by specification, when the physics need (d(composition)/dn_B, which no
`EOSTable_for_TOV` carries) is general. So: a `DEFERRED.md` entry today, and a
composition-derivative contract in `general/` as [ticket 53](53-gmode-contract.md).

**2. The four §5 signature rows split three-to-one toward the code.** `leptons`
through `**conditions` is drift — six models already make it an explicit named
argument, and routing it through the bag mutates the mode name into
`fixed_YC_neutral`, which §3 does not define. A `mode` default is a physics choice
made on the caller's behalf, the same reason `par` is non-optional. `zl`'s
`thermo_from_n(n_B, Y_C, ...)` becomes `(n_n, n_p, ...)` — not because the
parameterisation is wrong, but because it is the only hit in §5's published purity
grep, so it makes that grep return a false positive a reader cannot tell from a
real one. Only `Y_p` goes the other way: it is a *freeze target*, not a condition,
and renaming it to a §5 condition name would be a lie.

**3. A failed `eos_response` returns the full dict shape with `converged=False`
and NaN in every quantity** — not a minimal error object. A caller writing
`result["cs2_adiabatic"]` into an array column must not need a second code path
for the failure case, and NaN propagates to a plot honestly. `DEFERRED.md`
already asks for this same shape for the astro/tov non-monotone-table case.

**4. The §8 verify gaps are closed now, not in Phase 5.** They are pure
additions — no number moves — and finding 27 is a real hole: CCDM is a
colour-superconducting model with **no causality check at all**, where a wrong gap
shows up in the sound speed first. Finding 25 is the same shape in the flagship
DD-RMF: dd2 carries Σ^R and checks neither `f = eps − Ts` nor the rearrangement
placement, which is the one invariant that catches a wrong density-dependent RMF.
`ccdm/verify/run_full_check.py:263-297` is the model implementation to copy.
`astro/tov` having no `verify/` at all is ledgered rather than fixed here — with
`test/` gitignored, a fresh clone has no way to check TOV, which is a bigger
question than this triage.

**5. Finding 24's fix lands in `general/`, not in `dd2`.** dd2 did not ignore §7
so much as find no door: `general/fermi_integrals.py` exports **no public T = 0
entry point** — everything at `:220 _compute_exact_T0` is private and the public
names (`solve_fermi_jel`, `Fermi_Numerical`, `kinetic_thermo`) are finite-T. So
the T = 0 closed forms are promoted to a public name and dd2's four functions go.

**6. Inference-readiness is closed where it is one field and deferred where it is
a rewrite.** `TC_COEFF = 0.57 * 2**(1/3)` moves into `alphabag/parameters.py`
because an inference run over CFL pairing genuinely cannot vary it today. Freezing
`sfho`'s dataclass is a bigger job than it looks — the `couplings_map` is written
after construction at seven sites in `parameters.py`, a builder pattern that has
to become `replace()` or a real constructor — and both models pickle, so
multiprocessing works today. Ledgered.

### Golden references, per §12

Six of the 23 (a)-class fixes could move a number, and each names its check:

- **24** (T = 0 promotion, `general/`) — **DD2 golden SNM point at
  n_B = 0.16 fm⁻³** and `test/baseline/dd2` at rtol = 1e-10. The closed forms are
  already algebraically identical and the audit found no discrepancy, so anything
  beyond round-off means the promotion is wrong and gets reverted.
- **12, 13** (`quark_charges` dedup) — `test/baseline/` for `alphabag` and `mixed`.
- **8** (`thermo_from_n` signature) — `test/baseline/` for `zl` and `mixed`,
  since `mixed/adapters.py:913` consumes it.
- **17a** (`TC_COEFF` into the parameter object) — `test/baseline/alphabag`; the
  default must reproduce `0.57 * 2**(1/3)` exactly.
- **14b** (sfho's inline species-potential map) — `test/baseline/sfho`, and it
  rides [ticket 45](45-rename-sfho.md), which already carries that gate.

The other seventeen touch signatures, docstrings, import placement, failure paths
or `verify/` additions only, and move no converged number. Every ticket reports
failures **added** against `output/_audit/pytest_before_with_crust.txt`.

### Cut of the (a)-class work

Not by model and not by CLAUDE.md section, but **by what gate the change needs** —
because the thing that decides whether two fixes can share a session is whether
they can move a number, not which file they live in:

- [49](49-nonconvergence-return.md) — §6 non-convergence at every public boundary
  (18, 19). One behaviour shape applied to seven sites; needs the ruling above
  applied uniformly or it drifts again.
- [50](50-mechanical-fixes.md) — the eight fixes that move no number (4, 7, 9, 10,
  11, 12, 13, 23). One session, one commit.
- [51](51-verify-invariants.md) — the four missing `verify/` invariants (25, 26,
  27, 28) plus dd2/did adopting the delivery gate (29). Pure additions.
- [52](52-general-t0-integrals.md) — `general/` gets a public T = 0 entry point and
  a bounded loop (24, 20). **Touches every model's numbers**, so it gets its own
  golden-reference gate.
- [53](53-gmode-contract.md) — the composition contract and the gmode constants
  (2, 17a').
- [54](54-signature-corrections.md) — the public-signature corrections (8, 15,
  16b, 17a, §3-ii). Blocked on [44](44-rename-dd2.md) and [45](45-rename-sfho.md)
  so it does not collide with the renames already in flight.

The (b) rows all land in [ticket 22](22-phase5-claudemd.md) as one CLAUDE.md diff;
the (c) rows land in [ticket 55](55-deferred-ledger.md) as `docs/DEFERRED.md`
entries. Nothing was edited by this ticket.
