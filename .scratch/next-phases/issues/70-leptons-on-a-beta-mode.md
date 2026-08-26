# `leptons=` on a beta-equilibrium mode: six models, three answers

Type: task
Status: resolved
Blocked by: —
Parent: ../map.md

## Question

Split out of [ticket 68](68-njl-ccdm-leptons-condition.md), which fixed the
mechanical half of the row and declined this one, as
[ticket 54](54-signature-corrections.md) had before it. Both declined for the
same reason: converging the six is a BEHAVIOUR change, and which way to
converge them is not mechanical.

Now that `leptons` is a named argument everywhere, the six models with both a
beta mode and the flag answer `leptons=` on a beta mode in three ways —
measured on the tree at ticket 68's commit, not restated from the audit:

    sfho, dd2     RAISE on leptons=True (their own pre-54 guards)
    zl, did       SILENTLY IGNORE both values; zl's docstring says so in as
                  many words -- "Ignored by the beta-equilibrium modes, where
                  leptons are what the equilibrium is about"
    njl, ccdm     ACCEPT leptons=True, RAISE on leptons=False
                  ("leptons=False has no meaning in beta equilibrium, which is
                  defined by the leptons")

Note that njl/ccdm's answer is the OPPOSITE of sfho/dd2's on both values, which
is the sharpest form of the problem: the same call is a hard error in one model
and the only accepted spelling in another.

§4 says "a flag a model does not implement RAISES; a NotImplementedError is
never turned into a silent no-op", which argues for raising. The
counter-argument is that in a beta mode the leptons are not *unimplemented*,
they are constitutive — so `leptons=True` is redundant rather than
unsupported, and `leptons=False` is the only genuinely contradictory call,
which is exactly the reading njl and ccdm already ship.

A third reading nobody ships: ignore `leptons=True` (redundant, harmless) and
raise on `leptons=False` (contradictory) — with the docstring saying so — which
is njl/ccdm's behaviour generalised, and would make `zl`/`did`'s silent
acceptance of `False` the only thing that changes.

Decided ONCE for all six, not per model. Whatever wins, §4 or §3 gains the
sentence that makes it the rule rather than the majority, via
[ticket 85](85-claudemd-sentences-owed.md), and the four models that do not already
match it change.

Resolved when the reading is chosen with its reasoning recorded, the six models
agree, `test/baseline/` has not moved (no beta-mode NUMBER changes under any
reading -- only whether the call is accepted), and the added-failure count is
reported with its interpreter and collected count.

## Ruling

Agreed with the user: **`leptons=False` on a beta-equilibrium mode RAISES;
`leptons=True` is accepted and ignored as redundant.** All six models converge
on that — njl and ccdm's behaviour made the rule.

Reasoning. §4's "a flag a model does not implement RAISES" governs
*unimplemented sectors*, and in a beta mode the leptons are not unimplemented,
they are **constitutive**. So `leptons=True` is a true statement redundantly
made, and raising on it punishes precisely the caller writing uniform code
across models — which is what a notebook knobs cell does, and why
`notebooks/hadronic_eos.py` currently withholds the flag on beta modes rather
than passing it uniformly (ticket 12). `leptons=False` is genuinely
contradictory: it asks for beta equilibrium without the particles that define
it, and silent acceptance loses that error.

Changes: `sfho` and `dd2` stop raising on `True`; `zl` and `did` stop silently
accepting `False`. `njl` and `ccdm` are already correct and are the reference.
§3 gains the sentence via [ticket 85](85-claudemd-sentences-owed.md).

No beta-mode NUMBER changes under this reading — only whether the call is
accepted — so `test/baseline/` must not move.

Open for execution.

## Resolution

**Landed as `77f8962`, 0 added failures, `test/baseline/` unmoved.** The rule
is written ONCE, as `eos.general.modes.resolve_leptons`, and called by every
unit that turns a mode name into a spec.

### The census was short by three

Measured on the tree at `e88736a` rather than restated from ticket 68's audit:
**nine** models carry both a beta mode and the flag, not six. `vmit` and
`alphabag` were silently accepting `False` alongside `zl` and `did`, and `enjl`
was already correct alongside `njl` and `ccdm`.

    before   raise on True, accept False   dd2, sfho
             silently accept both          zl, did, vmit, alphabag
             accept True, raise False      enjl, njl, ccdm
    after    accept True, raise False      all nine, and eos/mixed

Fixing only the six the ticket named would have left the identical defect in
two models and the composite engine, so all nine changed and `eos/mixed`'s
`make_charge_spec` with them.

### Three entry points per model, not one

`njl`'s and `ccdm`'s `eos_table` **accepted** `leptons=False` on a beta mode
after `eos_point` had been fixed, because the refusal was raised inside the
sweep, where `skip_errors=True` dropped it as a failed point — a §4 silent
no-op reached through the very flag this ticket is about, and the same
`table.py` line ticket 68 recorded as "found and NOT fixed". Both now resolve
at `TableSpec` construction, outside the sweep. Verified rather than assumed:
all **27** public entry points across the nine models refuse it, plus
`eos.mixed.make_charge_spec`; the surface probe reports `LEAKS: none`.

Two more sites the first pass missed, both defaults one layer below the public
API: `dd2.solver.solve_hadronic` and `alphabag.api.eos_table`. Both turned up
in the gate as added failures, which is what the paired run is for.

### `leptons=None`, and why it is needed

`dd2`, `sfho`, `did` and `alphabag` default the flag **off**; `zl`, `vmit`,
`njl`, `ccdm`, `enjl` and `mixed` default it **on**, and ticket 68 ruled that
divergence deliberate. Without a third state an omitted argument in the first
four is indistinguishable from an explicit `False`, so every default beta call
would raise. `None` is therefore the only spelling that refuses the
contradictory call without changing what `fixed_YC` means by default. The two
legacy `Settings` adapters (`sfho`, `alphabag`, `zl`) pass `None` where the
mode has no such flag, for the same reason.

`dd2`'s pre-existing "`fixed_YC_YS` with leptons is not wired" raise is kept:
that one IS §4's unimplemented sector, and folding it into the new rule would
have turned it into a silent drop.

### No number moved, measured

204 cases — nine models x four modes x two temperatures x {flag unset, True,
False} — compared bit-exact as hex floats (`P`, `eps`, `s`, `mu_B`, `mu_C`)
between `git archive e88736a` and the same tree plus this change:

- **32 lines changed**, every one a beta-equilibrium mode with an EXPLICIT
  flag: 24 `leptons=False` now refused, 8 `leptons=True` (dd2, sfho) now
  accepted.
- **0 changed with the flag unset.** The default path is bit-identical.
- **0 cases where both sides returned numbers and the numbers differ.**

### Gate

python.org **3.14.2** (`python3`), **1424** collected over `test/baseline
test/general test/{dd2,sfho,zl,did,vmit,alphabag,njl,ccdm,enjl,mixed}
test/test_imports.py test/test_nonconvergence_return.py
test/test_parameter_routes.py`, run as an isolated **pair** from
`git archive e88736a` plus a snapshot of the gitignored `test/`:

    control (e88736a)         1419 passed, 5 skipped   18:56
    mine    (+ ticket 70)     1419 passed, 5 skipped   18:58

**0 added failures**, and `test/baseline/` did not move.

### Done differently, and said so

`eos/enjl/solver.py` keeps its own copy of the rule rather than calling the
shared one. Its behaviour already matches the ruling — it is one of the three
reference models — and a concurrent session has that file staged for
[ticket 72](72-enjl-branch-selection.md); sharing the implementation there is
tidiness, and it is not worth the collision. The refactor is one import and
five deleted lines whenever that work lands.
