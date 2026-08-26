# `leptons=` on a beta-equilibrium mode: six models, three answers

Type: task
Status: open
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
