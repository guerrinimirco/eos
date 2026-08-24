# Where does the repository actually stand against CLAUDE.md?

Type: research
Status: resolved
Parent: ../map.md

## Question

Stage 6, read-only. Read `CLAUDE.md` end to end against the repository as it now
stands and produce **one table, model by model** — `zl`, `sfho`, `dd2`, `did`,
`vmit`, `alphabag`, `njl`, `ccdm`, `abpr`, `enjl`, `mixed`, `zlvmit`,
`astro/tov`, `astro/gmode`, `general` — one column per checkable claim:

- **§1 layering** — no model imports another model; no model imports `astro/`,
  including its `table.py` and `verify/`; `mixed/` importing `eos.astro.tov` is
  the one named exception. **Verify by import graph, not by eye.**
- **§2 conventions** — `S = +1` per s quark; `Y_C` non-leptonic;
  `mu_C = mu_p − mu_n`; basis maps imported from `general/`, not re-derived.
- **§3 modes** — which of the four each model supports, which raise, and whether
  each gap is recorded in `docs/DEFERRED.md`.
- **§4 species flags** — which are implemented, which raise, that none is
  silently ignored.
- **§5 layout and API** — `eos_point`/`eos_table`/`eos_response` with the spec
  signature, `par` first and never optional; the `progress` dictionary identical
  across models; `thermodynamics.py` free of `beta`/`Y_C`/`neutral`/`trapped`
  (grep it); the layer order inside the model; `backends/` deletable.
- **§6** — parameters are arguments; non-convergence is a return value; no global
  mutable state; model objects pickle; array in / array out.
- **§7** — no Fermi or Bose integral implemented outside `eos/general/`.
- **§8** — the invariants each `verify/` suite actually checks.
- **§10** — `figure_style.py` the only styling module; no second
  `STANDARD_COLORS`; `grep -rn "rcParams" eos/` hits exactly one file.
- **§11/§12** — `verify/run_full_check.py` present; tests in `test/<model>/`.

Record that `docs/STRUCTURE.md` does not exist — it is Phase 5 item 3 and belongs
to ticket 21, not here. **Do not edit `CLAUDE.md`.** Write to
`.scratch/next-phases/research/conformance-table.md`.

## Answer

Full report: [conformance-table.md](../research/conformance-table.md), 1340 lines.
Read-only confirmed — `git diff --stat` empty.

**136 scored cells: 87 Pass, 24 Fail, 25 Ambiguous, 14 N/A.** Worst units:
`dd2` (6F/2A) and `mixed` (4F/2A). Full per-section table in the report.

**(a) the code is wrong — 22 findings.** The three that matter most:

1. **`dd2`'s `SpeciesFlags.photons` is accepted and never read.**
   `dd2/species.py:26` declares it; `solver.py:757` and `table.py:243` take a
   separate `include_photons` kwarg defaulting True. So `photons=False` **silently
   returns a photon gas** — the exact silent-ignore §4 exists to forbid, in the
   flagship model. `mixed` repeats it at `thermodynamics.py:84`. Graduated to
   [ticket 28](28-photons-silent-ignore.md) rather than buried in the triage,
   because it changes numbers.
2. **The `progress` dictionary's `fracs` drops the *fixed* fractions** in `dd2`
   (`table.py:250`) and `sfho` (`table.py:296`), contradicting §5's "swept or
   fixed" verbatim. One printer no longer serves them all.
3. **`astro/gmode` imports model internals** — `rates.py:85`,
   `from eos.dd2.solver import solve_composition`. §1 forbids it and `gmode` is
   not the named exception.

Then: `eos_response` raises on non-convergence in five units (§6 breach —
`abpr/api.py:234` converts a `converged` flag into a `RuntimeError`);
`zl/thermodynamics.py:374` signature; dd2's parameter classmethods forcing the
deferred solver import §5 names verbatim, comment included;
`mixed/verify/run_full_check.py:44` making `backends/` non-deletable; alphabag
re-deriving `quark_charges` five times; dd2 re-deriving the T = 0 Fermi gas; and
three `verify/` suites missing invariants (dd2 has `Sigma_R` but no rearrangement
check, ccdm has no causality check at all, njl's is off by default).

**(b) CLAUDE.md should change — 11 findings.** Largest: **`ccdm` appears nowhere
in CLAUDE.md**, and `njl` only ever inside the string "enjl". §1's model list
omits `did`/`njl`/`ccdm`; §11's omits `njl`/`ccdm`; §5's adapter list omits the
shipped `njl_phase` and `ccdm_phase`. The `verify/` carve-out from the
model-to-model import rule is real (`test/test_imports.py:88-97` plus a DEFERRED
entry) but absent from §1. Also open: may `mode` carry a default; who owes the
P-monotonicity delivery gate; the scope of zlvmit's exemption; and the §10
acceptance criterion — `grep -rn "rcParams" eos/` hits **three** files, but two
are prose saying the file does *not* set rcParams, so the rule passes in
substance (all ~30 assignments live in `general/figure_style.py`) while the
grep-as-gate does not.

**(c) belongs in `docs/DEFERRED.md` — 12 findings.** `abpr` refuses all four §3
modes with good in-code reasons and no ledger entry; `mixed`'s `Y_Lmu` refusal is
the one of ten unrecorded; `astro/tov` has no `verify/`; `output/public/` does not
exist; one unbounded loop at `general/fermi_integrals.py:519`; and
`DEFERRED.md:328`'s own claim that "every model's parameter dataclass is
`Parameters`" is false — dd2's is `Parametrization`, which ticket 07 confirms
independently.

**`docs/STRUCTURE.md` does not exist** — already ledgered at `DEFERRED.md:529`.
Not written; it is [ticket 21](21-phase5-structure.md).

**Auditor's note worth carrying into triage:** `docs/DEFERRED.md` is unusually
thorough. Most of what a naive audit would flag is already recorded there with
reasoning and measurements, which is why the (c) pile is small. The real work is
the 22 (a)-class fixes, several of them one-liners.

Status: resolved.
