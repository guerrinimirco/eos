# njl and ccdm smuggle `leptons` through `**conditions` too

Type: task
Status: resolved
Blocked by: —
Parent: ../map.md

## Question

[Ticket 54](54-signature-corrections.md) item 1 fixed finding 16b in the three
models the conformance audit named — `sfho`, `dd2`, `did` — and while doing so
found **the same defect in two more**. The audit missed them because it read
`njl/api.py:122`, which is `eos_table`, and `eos_table` in both models already
takes the flag as a named argument. `eos_point` and `eos_response` do not.

    eos/njl/api.py:68     extra = [k for k in conditions
                                   if k not in MODE_FRACTIONS[mode]
                                   and k != "leptons"]
    eos/ccdm/api.py:76    (the identical line)

so `leptons` rides in the condition bag and is forwarded to `solve` through
`**conditions`, exactly as `sfho/dd2/did` did before 54. §5 fixes the condition
names at `n_B, T, Y_C, Y_S, Y_Le, Y_Lmu` and makes `leptons` an explicit named
argument, "never smuggled through `**conditions`" — the sentence 54 quoted.

Neither model has an invented mode name to retire with it, so this is smaller
than 54 item 1: `eos_point` and `eos_response` gain `leptons=<the current
effective default>`, the `k != "leptons"` carve-out goes, and the docstring line
"plus leptons=True/False" (`njl:97`, `ccdm:110`) moves out of the `conditions`
paragraph into its own. `eos_table` already has it and does not move.

**Check the effective default before choosing one.** `eos_table` in both models
defaults `leptons=True`; what `eos_point` does today is whatever `solve` does
when the key is absent, which is not necessarily the same. Match today's
behaviour and say so, or the change moves numbers where it claims not to.

Not in this ticket: giving the beta-equilibrium modes a raise for `leptons=True`.
That is the second half of the same row and it is not settled — see below.

## The related row 54 deliberately left alone

After 54 the five models answer `leptons=True` on a *beta* mode three ways:

- `sfho`, `dd2` — **raise** (their own pre-54 guards, preserved verbatim).
- `zl`, `did` — **silently ignore**, and `zl`'s docstring says so explicitly:
  "Ignored by the beta-equilibrium modes, where leptons are what the
  equilibrium is about."
- `njl`, `ccdm` — whatever `solve` does with the forwarded key.

§4's "a flag a model does not implement RAISES; a NotImplementedError is never
turned into a silent no-op" argues for raising everywhere. The counter-argument
is that in a beta mode the leptons are not *unimplemented*, they are constitutive
— so `leptons=True` is redundant rather than unsupported, and `leptons=False` is
the only genuinely contradictory call. 54 changed nobody's behaviour here because
converging them is a behaviour change item 1 did not authorise. Deciding which
reading wins is a `grilling` question, not a mechanical one, and it should be
decided once for all six models rather than per model.

Resolved when njl's and ccdm's three entry points take `leptons` as a named
argument, no entry point in the repository reads it out of `**conditions`, and
the added-failure count is reported with its interpreter and collected count.
`test/baseline/` must not move.

## Resolution

**Landed, 0 added failures, `test/baseline/` unmoved.** `njl` and `ccdm` take
`leptons` as a named argument on all three entry points; the `k != "leptons"`
carve-out is gone from both `_check` functions, and a repository-wide grep for
`!= "leptons"`, `conditions.pop("leptons"` and `conditions["leptons"]` over
`eos/` returns **nothing**.

**The effective default was measured before it was chosen, as the ticket
required: it is `True`.** `njl/solver.py:563` and `ccdm/solver.py:640` both do
`fractions.pop("leptons", True)`, so an absent key already meant neutralizing
leptons, and `eos_table` in both models already defaulted `leptons=True`. The
named argument therefore defaults to `True` in `eos_point` and `eos_response`
— which is NOT the `leptons=False` that `sfho`, `dd2` and `did` default to, and
the difference is deliberate: matching the siblings would have moved numbers
this ticket says it does not move. Proved rather than asserted — `fixed_YC` at
n_B = 0.6, T = 10, Y_C = 0.3 gives `P_total` bit-identical between the old
implicit path and `leptons=True` (`0x1.555bec7c649b3p+6`), and `leptons=False`
gives a different number (`0x1.2601e34970886p+6`), so the flag is live and the
default is the old behaviour.

The docstring line "plus leptons=True/False" left the `conditions` paragraph
for its own `leptons : bool` entry in both models, and `eos_response` gained a
sentence saying the flag holds through every point of the stencil.
`eos_table` was already correct and did not move.

**One deviation from 54's pattern, stated because it is one.** 54's `_check`
guard — `TypeError("leptons is a flag, not a condition")` — is carried over
verbatim, but in these two models it is currently **unreachable from the public
API**: with `leptons` a named parameter, Python binds a `leptons=` keyword to
the parameter and it can never land in `**conditions`, and unlike `sfho`/`dd2`
neither model routes `TableSpec.fixed` through `_check`. It is kept because
`_check` is the single validation point for a public call and the sibling
models read identically there; it is three lines, and it is honest about being
belt-and-braces rather than the live catch it was in 54.

### The AST check, both shapes, before and after

Clean. **75** functions in `eos/` take `leptons` as a parameter and **7**
rebind the name locally, and the two sets do not intersect — **0 collisions**.
The seven are the same seven 54 found (`ccdm/solver.py:640`,
`dd2/solver.py:352,828`, `did/solver.py:665`, `njl/solver.py:563`,
`sfho/solver.py:536`, `sfho/table.py:508`), all in functions that do not take
the argument. No call site in `eos/`, `test/` or `notebooks/` passes `leptons`
to `njl` or `ccdm` `eos_point`/`eos_response` at all, so nothing had to change
with them.

### Gate

python.org **3.14.2** (`python3`), collection **150** over `test/njl test/ccdm
test/baseline`, measured as an isolated PAIR from `git archive HEAD`
(`95db690`) plus a snapshot of the gitignored `test/`:

    control (HEAD)          6 failed, 144 passed   2:56
    mine    (HEAD + t68)    6 failed, 144 passed   3:08

Identical failure sets — the same six pre-existing `test/baseline` failures
(`ccdm`, `dd2`, `enjl`, `njl`, `tov`, `zlvmit`). Because `njl` and `ccdm` are
themselves two of the six, the failure DETAIL was diffed too, not just the node
ids: the 28 changed `njl` quantities and the `ccdm` list are byte-identical
between control and mine, so this ticket moved nothing inside an
already-failing baseline either. `njl`'s `verify/` reports **15/15 `[ok ]`**
and `ccdm`'s **16/16 `[ok ]`**.

### Found and NOT fixed, per the map's hard rule

`njl/table.py:56` and `ccdm/table.py:66` build the solve's fractions as
`{k: v for k, v in conditions.items() if k in MODE_FRACTIONS[mode]}`, so
`eos_table(..., fixed={"leptons": False})` is **silently dropped** rather than
either honoured or refused — the named `leptons=` argument beside it is what
actually decides. It is not the smuggling this ticket retires (nothing reads
the flag out of the bag; the bag's copy is discarded), but it is a §4 silent
no-op on the same name, and it is one line in each model.

### The related row, now [ticket 70](70-leptons-on-a-beta-mode.md)

Measured on the current tree rather than restated: `njl` and `ccdm` **accept**
`leptons=True` on a beta mode and **raise** on `leptons=False`
(`njl/solver.py:115`, `ccdm/solver.py` likewise: "leptons=False has no meaning
in beta equilibrium, which is defined by the leptons"). So the six models
answer in three ways, and the third one is not what 54 recorded as "whatever
`solve` does":

    sfho, dd2     raise on leptons=True
    zl, did       silently ignore both
    njl, ccdm     accept True, raise on False

Deciding which reading wins is the grilling question 54 declined and this
ticket declined; it is now its own ticket rather than a paragraph on a resolved
one.
