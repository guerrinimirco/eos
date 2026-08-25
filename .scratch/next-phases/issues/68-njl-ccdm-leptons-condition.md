# njl and ccdm smuggle `leptons` through `**conditions` too

Type: task
Status: open
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
