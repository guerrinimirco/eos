# Approve or reject the proposed public renames

Type: grilling
Status: resolved
Blocked by: 07
Parent: ../map.md

## Question

Ticket 07 produces a list of §13 deviations and proposes renames. This ticket is
the approval gate: **no public name is touched until the user has ruled on the
list.**

Take them one at a time where they differ in kind: a private helper renamed is
cheap; a public entry point renamed breaks `nucleation` call sites (Phase 6) and
any notebook already written against it.

Named suspects the list will contain: `eos/vmit/compute_tables.py` (not a §5
layout name), and whatever `Parameters` subclasses still repeat their package
(§13's example is `eos.vmit.VMITParams` saying "vmit" twice).

Resolved when every proposed rename is approved, rejected or deferred, and the
approved ones are applied with the test suite green and the added-failure count
against `output/_audit/pytest_before.txt` reported.

## Answer

**The gate is passed. 46 of the 58 approved, 3 frozen, 1 deferred, 5 split out as
not-renames, and 3 ruled keep.** Application graduated to
[42](42-rename-internal.md), [43](43-rename-vmit.md), [44](44-rename-dd2.md),
[45](45-rename-sfho.md) and [46](46-api-changes.md) — 550 call sites is more than
one session, and a single commit that size cannot be reviewed or bisected.

**The fact that made this cheap:** a grep of the whole `nucleation` tree for
`get_sfho`, `Parametrization`, `solve_vmit` and `get_vmit` returns nothing.
**Not one of the 58 touches a downstream call site**, so Phase 6 is not exposed
and the blast radius is entirely `eos/` + `test/` — and `test/` is gitignored,
so over half the churn is not even in version control.

### Approved (46), applied per package, smallest first

- **Rule 3 — drop `compute_` where it carries nothing (15).** vmit's fourteen,
  plus did's `evaluate` -> `baryon_kinetics`.
- **Rule 1 — a name never repeats its package (13).** vmit's `VMITEOSResult` ->
  `EoSPoint` and `VMITThermo` -> `MatterThermo`; the eleven `eos.mixed` names
  saying "mixed" twice.
- **Rule 2 — the shared vocabulary (18).** dd2's `Parametrization` ->
  `Parameters` with `.default()` / `.named("DD2Y")`, its six solver entry points
  renamed off `octet` onto the §3 mode names, and its two warm starts; sfho
  gaining the same constructor pair in place of five `get_sfho*` functions, with
  `get_all_parametrizations()` -> `PUBLISHED_SETS`; vmit's `get_vmit_default()`,
  four `solve_vmit_*`, `result_to_guess` and four `get_default_guess_*`.

Landed **per package, one commit each**, `eos/mixed` + `eos/did` first as the
rehearsal (fully internal, smallest radius), suite green between each. Rule 2
lands **before** [ticket 04](04-notebook-skeleton.md), since the skeleton would
otherwise hardcode names about to change — which is the precedence the map's
Notes already set.

### Ruled KEEP (3)

`solve_composition`, `solve_snm` and their `_t0` twins keep their names. They are
not §3 modes — symmetric matter at saturation is what `nmp.py` needs, not a mode
a caller selects — so rule 2 does not bind them, and unsuffixing `_octet` does
not reach them.

### FROZEN (3)

`VMITTableSettings`, `compute_vmit_table`, `save_vmit_results`. Their sole
consumer is `notebooks/ZLvMIT_hybrid.ipynb`, which CLAUDE.md §5 exempts as legacy
and the map rules out of scope — and which [ticket 41](41-corrupt-notebooks.md)
records as unopenable corrupt JSON. Renaming symbols whose only caller cannot be
opened buys nothing. `docs/DEFERRED.md` documented the exception on the *file*
and said nothing about the *symbols*; that gap was the actual defect and is now
closed.

### DEFERRED (1)

`thermo_at_potentials` vs `thermo_from_mu`. Not drift: dd2, sfho and did carry
both, at two layers, so the question is what to call the UPPER one. It wants a
name that names the §5 phase-adapter contract, so it is settled while `mixed.tex`
is written — recorded on [ticket 36](36-quark-engine-documents.md).

### SPLIT OUT — not renames (5)

Deleting `get_vmit_custom()`, folding the isentropic solvers into `SnB=`, merging
`find_mixed_window` into `locate_window`, turning two sfho constructors into
`from_*` (where §5 forces the NMP-inverting one into `nmp.py` as a free function,
not a classmethod), and naming `build_mixed_eos_table` by job. Each changes a
signature or deletes behaviour; approving them through a naming gate would
smuggle behaviour changes past review. [Ticket 46](46-api-changes.md).

### Also corrected here

`docs/DEFERRED.md`'s vmit entry claimed **DONE** while naming only two unconverted
functions. There are ~23. The entry now says PARTIAL and enumerates them, since a
ledger that undercounts by an order of magnitude is worse than no ledger.
