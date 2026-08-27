# One solver signature, one unit system at the boundary

Type: task
Status: resolved
Blocked by: -
Parent: ../map.md

## Question

Ruled by [ticket 81](81-second-default-solver-kwargs.md), sections 2 and 5.
Split out because **no value moves** in any of it: the whole gate is a green
suite plus unmoved baselines. The one regeneration it carries renames keys and
leaves every number identical.

Sections 4 and 5 were here too, and are not any more. The note at the foot of
this ticket showed section 4's premise is false — deleting the sector kwargs
into the flags moves frozen rows in three models — so it left rather than the
gate being weakened, one ticket per model:
[94](94-zl-solver-flags.md), [95](95-vmit-solver-flags.md),
[96](96-alphabag-solver-flags.md). Executing what remained then showed the
same of section 5, and of the half of section 2 entangled with it:
[ticket 97](97-natural-record-leaves-the-result.md). This ticket blocks all
four.

§13 exists so "a physicist who has read one model can read the next without a
translation table". `solver.py` currently needs one. Three argument orders:
`(par, n_B, flags, ...)` in dd2/sfho/did, `(n_B, T, params=None, ...)` in
zl/vmit/alphabag, `(n_B_fm, Y_C, T=0.0, par=None, flags=None, ...)` in
njl/ccdm/enjl. `par` is required in two models, optional in seven, and spelled
`params` in three.

## Work

**Signatures**, every model's `solver.py`:

1. `par` first and required. `params=` -> `par=` in zl/vmit/alphabag.
2. `n_B_fm` -> `n_B` (87 sites). Natural-units working variables displaced by
   the rename take `_nat`, the convention `dd2/solver.py:159` already uses.
   Four functions hold both names at once and need care:
   `enjl/solver.py:186,520`, `enjl/verify/run_full_check.py:130`, and
   `dd2/solver.py:101` where the sense is REVERSED (`n_B` is the fm one).
3. The rest of the `_fm` family: `n_C_fm` (12), `n_S_fm` (8), `eps_fm` (7),
   `P_fm` (7), `s_fm` (6), and `n_b_fm` — which is a result field, and whose
   lowercase `b` also violates §2's B-for-baryon convention.
4. **Not here.** Giving `zl`/`vmit`/`alphabag` a required
   `flags: SpeciesFlags` and deleting the `include_*` sector kwargs into it
   MOVES frozen values, for the reason recorded in the note below. It is now
   one ticket per model: [94](94-zl-solver-flags.md),
   [95](95-vmit-solver-flags.md), [96](96-alphabag-solver-flags.md), each
   blocked by this one. `include_electrons` -> `leptons` goes with them, not
   because it moves a value but because it is the same signature, and a
   model's signature should change once.

**Units at the boundary** — §5 is already the rule; three models break it:

5. **Not here either**, and neither is the half of item 3 that turned out to
   be the same act. Both are [ticket 97](97-natural-record-leaves-the-result.md).
   The `_fm` family named in item 3 is a set of PROPERTIES on the natural-units
   records item 5 removes, so renaming them swaps a field and an accessor on a
   record the baselines freeze: **4128 frozen keys change meaning** and item 5
   deletes **21271** outright. Measured, on the frozen files:

       enjl.npz  21278 keys, 14976 nested under `.point`, 3042 of them the six
       njl.npz    6594 keys,  3255 nested under `.state`,  630 of them the six
       ccdm.npz   6068 keys,  3040 nested under `.state`,  456 of them the six

   What stays here is the half that moves nothing: `n_B_fm` as a FUNCTION
   PARAMETER (the 87 sites, which is the §13 signature problem this ticket is
   named for) and `BetaPoint.n_b_fm` -> `n_B` as a result field, whose 234
   keys are a rename with identical values and are gated below.

## Gate

- **No value moves anywhere.** Every `test/baseline/*.npz` unmoved at
  rtol = 1e-10, `enjl.npz` excepted below.
- `test/baseline/enjl.npz`: the `n_b_fm` -> `n_B` rename changes **234 keys**.
  Verify by comparing old and new arrays key-for-key after mapping the name —
  every value identical, or it is not a rename.
- Full suite green (§12). `verify/` green for every model touched.
- `zl`, `vmit` and `alphabag` baselines stay green WITHOUT regeneration —
  and here that is easy, because this ticket leaves their sector kwargs
  exactly where they are. `params=` -> `par=` is a keyword rename; the
  generator's 6 call sites are updated and reproduce the frozen rows bit for
  bit. A moved row in any of the three means the rename touched something
  that was not a name.

---

## Note from [ticket 82](82-alphabag-gluons-default.md) (2026-08-26)

**"No value moves" does not hold for alphaBag.** Section 2's deletion of
`include_photons` / `include_gluons` / `include_thermal_neutrinos` into the
flags moves `test/baseline/case_alphabag`, because that generator calls the
raw solvers and names none of the three; once they read the flags it picks up
the defaults, and all three alphaBag defaults are now `False` — `photons` and
`thermal_neutrinos` since [ticket 65](65-species-flag-defaults.md), `gluons`
since [ticket 82](82-alphabag-gluons-default.md).

Measured through `eos_point` at n_B = 0.8, for the gluon sector alone:

    beta.T0     unchanged — every thermal sector vanishes at T = 0
    beta.T10    P  -1.465838e-03 MeV/fm^3
    beta.T30    P  -1.187329e-01 MeV/fm^3

`zl` and `vmit` carry the same shape for `include_photons`. So this ticket
needs a baseline regeneration and its own key-by-key diff after all, or the
three affected models split out the way [ticket 89](89-dd2-honours-species-flags.md)
was — which was the right instinct applied to the wrong model.

**Ruled (2026-08-27):** split, not weakened — twice, the second time by
measurement taken while executing. Section 4 is now
[94](94-zl-solver-flags.md), [95](95-vmit-solver-flags.md) and
[96](96-alphabag-solver-flags.md), each blocked by this ticket and each
carrying its own measure-then-regenerate gate; 96 is additionally blocked by
[ticket 92](92-cfl-gluon-term.md), which decides whether `solve_cfl` keeps a
gluon term at all. Section 5 and the record half of section 3 are
[ticket 97](97-natural-record-leaves-the-result.md), split after the `_fm`
rename was written, measured against the frozen files and backed out: the
`_fm` names are accessors on records the baselines freeze, so the rename
changes what 4128 keys mean and the removal deletes 21271. What is left here
moves no value, which is the gate this ticket was given and can now keep.



---

## Note from [ticket 92](92-cfl-gluon-term.md) (2026-08-27)

**Resolved, and the answer is on [ticket 96](96-alphabag-solver-flags.md)**,
which is the split that carries alphaBag. In short: `solve_cfl` raises on
`gluons=True` and the `cfl` arm of `table.solve_at` raises on
`thermal_neutrinos`, so for the paired phase section 2 re-points two existing
`NotImplementedError`s rather than translating two kwargs. The alphaBag
baseline note above is unchanged and was re-measured: the six `cfl.*` rows are
at T = 0 and move under no answer, and `test_baseline[alphabag]` is green
under ticket 92 in an isolated change arm.

---

## Resolution (2026-08-27)

Executed on python.org 3.14.2 (numpy 2.3.5, scipy 1.17.0). What is here now is
items 1, 2 and the parameter half of 3; items 4 and 5 are the four tickets
above, split rather than the gate weakened.

### `par` first and required, in seven models

`dd2`, `sfho` and `did` already had it. The other seven now do:

- `zl`, `vmit`, `alphabag`: `params` -> `par` throughout `solver.py`, `par`
  moved to first position, and **the `if par is None: par = Parameters.default()`
  reach deleted from all thirteen entry points** — that block was the §6
  violation the signature was hiding, not a convenience.
- `abpr`: `solve_cfl(n_B, par=None, T=0)` -> `solve_cfl(par, n_B, T=0)`.
- `njl`, `ccdm`: `solve(mode, n_B, T, par=None, flags=None, ...)` ->
  `solve(par, mode, n_B, T=0.0, flags=None, ...)`, and the same on
  `solve_pattern` / `solve_candidate` and the four mode wrappers. Both models
  carried a hand-written `raise TypeError("... needs par ...")`; both are
  deleted, because a required positional argument raises that itself.
- `enjl`: `solve(mode, n_B, par=None, ...)` -> `solve(par, mode, n_B, ...)`,
  and `solve_at_entropy` with it.

**139 call sites moved by an AST rewrite**, not a regex: the argument's exact
source slice is located from the parse tree and re-inserted before the first
positional, so multi-line calls move correctly. 22 more were hand-edited where
the parameter was already positional (the three `table.py` shims, `eos/mixed/
adapters.py`'s zl/vmit/alphabag/njl/ccdm wing points, and a dozen test calls).

### `n_B_fm` -> `n_B`, and the `_nat` convention

87 sites. The natural-units working variables displaced by the rename take
`_nat` — `dd2/solver.py:159`'s existing convention, now stated in
`docs/STRUCTURE.md` §5 as the rule: **a bare name is fm-based, `_nat` is the
only place the two systems are named apart.** In `enjl/solver.py` that meant
the whole residual family (`state_at`, `residual`, `residual_scales`,
`_scaled_residual`), whose `n_B` was MeV^3 while every other `n_B` in the file
was fm^-3.

`dd2/solver.py`'s local `n_B_fm` is **not** renamed to `n_B`: it is the
EVALUATED density against the function's `n_B` TARGET, which differ at T > 0 by
the inversion tolerance, so it is `n_B_solved` — `eos.enjl.table`'s spelling of
the same distinction. `_fm` would have said nothing once a bare name is fm.

`BetaPoint.n_b_fm` -> `n_B`, closing §2's lowercase-`b`.

### The one silent-unit trap this sprang, and what caught it

`enjl/solver.py::default_guess` held **two seed expressions in natural units**
that the rename did not reach — `0.9 * n_B` (the deconfined seed's quark
fraction) and `0.5 * Y_Le * n_B` (the trapped seed's neutrino density). Both
silently became fm. Every enjl test still passed except one:
`test_high_density_needs_a_widened_box` at n_B = 5 fm^-3, which stopped at a
scaled residual of **2.259e-09 against a 1e-10 bound**. Verified against an
isolated control copy with `eos/{enjl,njl,ccdm,mixed}` reverted, where the same
test passes. Nothing else in the suite would have found it; it is why
[ticket 97](97-natural-record-leaves-the-result.md) is told to rename fields
first and accessors second.

### Gate

    test/baseline           20 passed, 0 failed — UNREGENERATED except enjl
    enjl.npz                21278 keys: 234 renamed n_b_fm -> n_B, every one
                            BIT-identical; 21044 surviving keys, 21044
                            bit-identical, 0 moved; 0 keys added or lost
    verify/                 twelve entry points, 136 checks, 0 FAIL
    full suite              1737 passed, 20 skipped, 0 failed  (1319 s)

**Zero added failures** against `output/_audit/pytest_before.txt`, and the
prediction of the gate above held exactly: the 234 keys are a rename and
nothing else, and no other baseline moved at rtol = 1e-10 without being
regenerated at all.

### Findings, reported not fixed

- **`test/baseline/generate_baseline.py`'s njl and ccdm cases call `solve`
  positionally and the suite hides it.** Four such calls broke on the new
  signature and `test/baseline` PASSED when run alone; only the full suite
  showed `ValueError: unknown mode`. The cases run under `build_table` for the
  modes and reach the raw `solve` only for the pairing patterns, so a partial
  run never touches them. Fixed here because the ticket broke them, but the
  shape — a baseline case whose failure needs the full suite to appear — is
  worth a gate of its own.
- **`TableSpec.params` is still `params`** in `zl`, `vmit` and `alphabag`,
  while their solvers and `api.py` now say `par`. Item 1 is scoped to
  `solver.py` and this ticket did not widen it; the field is public
  (`build_table(TableSpec(params=...))`), so renaming it is a caller-visible
  change that wants its own ticket.
- **`enjl/thermodynamics.py` names its natural-units record `EoSPoint`**,
  which is also the name of the SHARED public record in
  `eos.general.state` that njl, ccdm, zl and vmit return. Two different jobs,
  one name, in one package — §13 rule 2. [Ticket 97](97-natural-record-leaves-the-result.md)
  touches this record and is the natural place to fix it.
- **`njl`/`ccdm` `solve` still take `flags=None` and build a default
  `SpeciesFlags()`** when it is absent. That is §4's implicit switch-on wearing
  a different hat, and it is the same shape ticket 81 §1 ruled on for `dd2`.
  Not this ticket's; not measured here.
