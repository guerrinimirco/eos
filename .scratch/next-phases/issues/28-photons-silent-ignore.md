# dd2 and mixed accept photons=False and return a photon gas anyway

Type: task
Status: resolved
Parent: ../map.md

## Question

[Ticket 08](08-conformance-table.md) found that `dd2` declares
`SpeciesFlags.photons` at `eos/dd2/species.py:26`, but the solver and table
driver take a **separate** `include_photons` keyword that defaults to `True`
(`eos/dd2/solver.py:757`, `eos/dd2/table.py:243`). The flag is accepted and never
read. `eos/mixed/thermodynamics.py:84` repeats the same pattern.

So `SpeciesFlags(photons=False)` silently returns results **with** a photon
contribution to `eps`, `P` and `s`. CLAUDE.md §4 forbids exactly this: *"No
sector is enabled or disabled implicitly… Setting a flag a model does not
implement RAISES; a NotImplementedError is never turned into a silent no-op."*

This is separated from the [conformance triage](11-conformance-triage.md), which
owns the other 21 (a)-class findings, for two reasons: it is a correctness bug
rather than a layout question, and **it changes numbers** — so it is the one
finding that must be weighed against §12's golden references before anything
moves.

What resolving it requires:

- Decide whether `photons` and `include_photons` unify onto the flag (§4's
  reading) or whether the kwarg is the real interface and the flag should raise.
- Establish which of the §12 golden references were computed with a photon gas
  present. The DD2 golden SNM point, the published NMP/TOV values, the CompOSE
  HS(DD2) slices and `test/baseline/` at rtol = 1e-10 are ground truth: if a
  baseline was generated with `photons=False` and silently got photons, the
  baseline itself encodes the bug and re-pinning it is part of the fix, with the
  reason stated.
- Check whether any figure, table or published result in `output/` or in
  `nucleation` was produced through this path.

**Do not loosen a tolerance to make a test pass.** If a baseline must move, say
why in the test, per §12.

## Measurement (2026-08-24) — bug confirmed, baselines clean

Static: `flags.photons` is read **nowhere** in `eos/dd2/`. Every other model that
declares the flag reads it — `zl/table.py:45`, `sfho/solver.py:455,517,816`,
`did/solver.py:385`, `njl/solver.py:272`, `ccdm/solver.py:304`,
`vmit/table.py:58`, `alphabag/table.py:58`, `enjl/solver.py:599`. dd2 is the only
model that does not. (`abpr` also never reads it, but correctly: it is T = 0
only, its flag defaults `False`, and `species.py:32` documents the refusal.)

Numerically, `fixed_YC` at `n_B = 0.16`, `Y_C = 0.1`, `leptons=True`, through
`eos_point` with `SpeciesFlags(photons=False)` against `solve_octet(...,
include_photons=False)`:

| T [MeV] | leaked ΔP | as multiple of P_γ | rel. err on P | rel. err on eps | rel. err on s |
|---|---|---|---|---|---|
| 0  | 0 | — | 0 | 0 | 0 |
| 10 | 2.854e-4 | **1.0000** | 0.008 % | 0.0006 % | 0.09 % |
| 30 | 2.312e-2 | **1.0000** | 0.355 % | 0.0441 % | 0.96 % |

Exactly one photon gas leaks in, and `photons=True` and `photons=False` return
**bit-identical** results at every temperature. It grows as T⁴.

**No golden reference is affected, and nothing needs re-pinning.**
`test/baseline/generate_baseline.py` `case_dd2` never constructs
`SpeciesFlags(photons=False)` — it takes the `photons=True` default and calls
`solve_octet` without `include_photons`, which also defaults `True`. Flag and
kwarg agree, so the baseline records what it intends. The dd2 tests likewise pass
`include_photons=False` as an explicit kwarg (≈20 call sites) and so bypass the
flag entirely. The DD2 golden SNM point is at T = 0, where the leak is zero.

**So the bug is latent, not retroactive.** It bites only a caller who goes
through the public API with `photons=False` — which is precisely what the three
notebooks about to be written will do, and what §4 promises works.

**Two corrections to the audits:**

1. **`mixed` is a different finding.** `eos/mixed/` has no `species.py` at all, so
   there is no flag being ignored — `thermodynamics.py:84` calls
   `photon_thermo(T)` unconditionally at T > 0 and the engine offers **no way to
   turn photons off**. That is a §4 *missing flag*, not a silent-ignore, and it
   needs a different fix. [Ticket 08](08-conformance-table.md) framed the two as
   the same bug.
2. **A third dd2 path is also hardcoded**: `eos/dd2/table.py:106` passes
   `include_photons=True` literally, so the entropy-per-baryon table can never
   turn photons off even via the kwarg.

**Also found, and owed to [ticket 03](03-stage0-removals.md):**
`test/dd2/test_notebook_api.py:13` does `from eos.dd2 import notebook_api as api`.
[Ticket 07](07-naming-sweep.md) reported that no test imports it — that is wrong.
Deleting `notebook_api.py` must delete that test file with it, or the suite
breaks.

## Ruling and fix (2026-08-24)

**Ruled:** wire dd2 the way the other five models already work — `table.py` /
`api.py` read `species.photons` and pass it down as `include_photons`. The kwarg
stays as internal plumbing; it is not a competing public interface.

A correction to the framing that produced the first ruling: `include_photons` is
**101 sites in 18 files** across `eos/` plus **39 in 14 files** in `test/`, not
the ~20 dd2 sites first reported, and in `zl`, `vmit`, `alphabag`, `sfho` and
`enjl` it is the internal plumbing that already reads the flag correctly
(`zl/table.py:45`, `vmit/table.py:58`, `alphabag/table.py:58`,
`sfho/solver.py:816`, `enjl/solver.py:599`). Deleting it would rewrite five
working models plus `nucleation/composition.py`, which passes it through its own
public API, for no correctness gain. dd2 was the one model missing the wiring.

**`mixed` split out** to [ticket 29](29-mixed-species-flags.md): it has no
`species.py`, so it is a missing-flag design question, not this bug.

**Applied** — four dispatch points in `eos/dd2/table.py` and one in
`eos/dd2/api.py`:

- `solve_octet_at_entropy` stopped hardcoding `include_photons=True`
  (`table.py:106`)
- the per-point T branch of `solve_at` (`table.py:244`)
- the fast one-sweep-per-line path (`table.py:255`)
- `build_core_table`, which hardcoded `include_photons=False` (`table.py:325`) —
  a no-op at T = 0, where photons vanish, but it contradicted the flag
- `eos_point`'s `solve_octet` branch (`api.py:102`). The entropy branch needs
  nothing: `solve_octet_at_entropy` reads the flag itself.

**Verified**: switching the flag now moves P, eps and s by *exactly* the mu = 0
photon gas, through both `eos_point` and `eos_table`, and by nothing at T = 0.

**Check left behind**: `test/dd2/test_photons_flag.py`, 4 tests. It asserts the
difference equals `photon_thermo(T)` to `rel=1e-12` rather than merely "differs",
so a flag that reached the field equations would fail it.

## Verification

`pytest test/dd2/ test/baseline/ test/mixed/ -q`, run in a fresh process **after**
the fix, gives exactly two failures:

    FAILED test/dd2/test_dd2_m4_tov.py::test_tov_dd2_nucleonic_pipeline
    FAILED test/baseline/test_baseline.py::test_baseline[did]

Both pre-existing and both unrelated:

- `test_dd2_m4_tov` calls `sweep_beta_eq_octet(..., include_photons=False)`
  explicitly at line 55 — the kwarg, which this change does not touch — and fails
  on a radius assertion at line 35 (`12.33` km against `13.2 ± 0.4`), at T = 0
  where photons vanish regardless. Verified by running the file alone.
- `test_baseline[did]` is a `did` mismatch; this change touched only `dd2`.

Every other dd2, baseline and mixed test passes, including `test_baseline[dd2]`
— which confirms the fix moved no baselined number, as predicted: the generator
never asked for `photons=False`.

**Added failures: 0.**

Status: resolved.
