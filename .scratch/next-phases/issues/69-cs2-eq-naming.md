# `cs2_eq` names the freeze where §5 requires the thermal variable

Type: task
Status: resolved
Assignee: mirco (session)
Parent: ../map.md

## Question

Four models return the equilibrium sound speed under the key **`cs2_eq`**:
`zl` and `dd2` ([ticket 13](13-hadronic-figures.md)), `vmit` and `alphabag`
([ticket 15](15-quark-notebook.md) item 2). The other six — `sfho`, `did`,
`njl`, `ccdm`, `abpr` and `mixed`'s hadronic side — return
`cs2_isothermal`, and `did`/`njl`/`ccdm` return `cs2_adiabatic` beside it.

`cs2_eq` names the *composition* axis of §5's conditioning (nothing held, the
composition re-equilibrates) and leaves the *thermal* axis unsaid, although the
derivative is taken at fixed `T`. §5 requires the returned name say which
thermal condition it was taken at, "never a bare `cs2` whose meaning depends on
the arguments" — and at `T > 0` the isothermal and adiabatic speeds differ by
`C_P/C_V`, so `cs2_eq` is exactly that bare name with a freeze word in front of
it. At `T = 0` the two coincide, which is why every notebook that reads the key
has been able to plot it and label it correctly anyway.

The freeze is not lost by the rename: it is the `frozen=` argument the caller
passed, and §5's three axes are conditioning, not return-key components.

So: rename the key to `cs2_isothermal` in the four, keep `cs2_adiabatic`
wherever the model can compute it (and say in its docstring where it cannot),
and sweep the callers — each model's `verify/`, `responses.py` docstrings, the
`.tex`/`.md` documents, `test/`, and the four notebooks, which currently all
carry a two-key reader for exactly this reason.

Cross-check against `eos/mixed` and `eos/astro/gmode` before renaming:
[ticket 49](49-nonconvergence-return.md) records `mixed.eos_response` returning
`cs2_eq = nan` outside the mixed window, and
[ticket 53](53-gmode-contract.md) asks whether `cs2_equilibrium` /
`cs2_frozen` is the g-mode surface — a third spelling, and this ticket should
not settle its own four while leaving that one open.

Done when a grep for `cs2_eq` over `eos/` returns nothing outside a changelog,
every `verify/` suite still passes, and `test/baseline/` is unmoved (a key
rename moves no number).

## Answer

**Renamed in all four.** `zl`, `vmit` and `alphabag` in commit `5a4a6cc`; the
`dd2` half (`eos/dd2/api.py`) landed inside `5c75584`, a concurrent session's
conformance commit that staged this file with its own work. Nothing was lost
and no history was rewritten, but the dd2 rename is not findable from this
ticket's commit, which is why it is recorded here and in the audit file.

A grep for `cs2_eq` over the ten models now returns nothing.

**The freeze is not lost, and that is the objection worth answering.** §5's
conditioning has three axes — what composition is held, what thermal variable
is held, whether leptons re-neutralize. Those are properties of the CALL, not
components of the return key. The composition axis is the `frozen=` argument
the caller passed; it is still there, still `frozen='equilibrium'`. What the
key names is the one axis the caller cannot recover from its own arguments.

**Swept:** each model's `api.py` and `verify/run_full_check.py`, the `.tex` AND
`.md` documents, the `test/` readers, and both notebooks that carried a two-key
reader for exactly this split — `hadronic_eos` §6 and `quark_eos` §6.3 and §8,
which now read the single key and have lost the reader, `CS2_KEYS`, and the
`sound_key`-threaded axis label that existed only to report which spelling came
back. `enjl_eos` and `hybrid_eos` never read the key and are untouched.

**Docstring gaps named, as asked.** `zl`, `vmit` and `alphabag` compute no
adiabatic speed: `C_P` is not among their returned quantities, so there is no
factor to form it with. Said in each docstring and each document.
`did`/`njl`/`ccdm`/`sfho` keep the `cs2_adiabatic` they already return.

**Two documents were stating something false, and are corrected rather than
renamed.** `dd2.md` and `dd2.tex` both said the `composition` freeze's speed is
"the adiabatic one", differing from the equilibrium one by `C_P/C_V`.
`eos/dd2/responses.py:35-44` holds `T` on both stencil points, so it is
*isothermal at frozen composition*. Both dd2 freezes hold T; what separates
them is the composition axis. The rename made this visible — it was false
before it.

### The AST check (tickets 42, 43, 54), both shapes, before and after

- **shape 1** — a name both imported and defined in one file: **none**, either
  side of the change.
- **shape 2** — a local rebind of the name being INTRODUCED: `cs2_isothermal`
  is bound locally **nowhere** in `eos/`, `test/` or `notebooks/`, so the
  rename lands on no existing binding and cannot fail silently.
- `cs2_eq` binds as an *identifier* in `eos/astro/gmode/*` (35 sites) and
  `eos/dd2/table.py`. Neither is the `eos_response` key; both are out of scope
  below, and the check is what made that boundary visible rather than assumed.
- Incidental, worth knowing: `notebooks/enjl_eos.py:891` binds `cs2_adiabatic`
  as a local variable — the exact shape-2 trap for any future rename onto that
  name in that file.

### Gate

Run as an isolated HEAD-vs-HEAD+patch **pair**, because a concurrent session
was editing the checkout throughout and a working-tree run would have credited
this rename with its failures. Run **twice**, because that session moved HEAD
mid-ticket. Full numbers in `output/_audit/pytest_after_ticket69_py314.txt`.

Against `ffbe55c`: baseline 6 failed / 10 passed on BOTH sides, same six.
Against `5c75584`: baseline 1 failed / 15 passed on BOTH sides, same one.
Both HEADs: all four verify suites exit 0 with zero failures on both sides;
the targeted key-readers 2 failed / 92 passed on both sides, the same two
pre-existing NMP-inversion failures.

**Failures added: none. No number moved, no tolerance touched.** The baseline
improvement between the two gates is `5c75584`'s, not this ticket's.

The first `mine` build was itself contaminated — `git diff HEAD` had swept up
the other session's `_mode_kwargs` import hunk in `eos/dd2/api.py`, which
failed against HEAD's `solver.py` and added two failures. Stripping that hunk
and rebuilding is what turned a false alarm into a clean pair; recorded because
the next person gating beside a live session will hit it.

`test/baseline/*.npz` was confirmed unmoved, as the ticket said it would be.
Ticket 62 is not ordered against this one.

### Out of scope, and why

- **`eos/mixed` (`cs2_eq`/`cs2_frozen`) and `eos/astro/gmode`
  (`cs2_eq`/`cs2_ad`).** Not four models but ONE surface: `gmode` imports
  `sound_speed_eq` and `sound_speed_frozen` from `eos/mixed/responses.py`, so
  the eq/frozen pair is shared vocabulary and renaming half of it is worse than
  renaming none. [Ticket 53](53-gmode-contract.md) owns that surface and is
  open; this ticket said it should not settle its own four while leaving that
  one open, and the honest reading is that it cannot — 53 is a `grilling`
  ticket and its answer is a decision, not a sweep. Noted on 53.
- **`eos/dd2/table.py`'s `TableResult.cs2_eq`.** It is `np.gradient(P, eps)`
  along one line, and the line's thermal condition is whichever axis the spec
  was built on: **isothermal on a `T` axis, ADIABATIC on an `SnB` axis**. It
  can therefore be renamed to neither, and it is the same §5 defect with a
  different correct answer. [Ticket 72](72-dd2-remaining-cs2-names.md).
- **`eos/dd2/api.py`'s `cs2_ad`** under `frozen='composition'`. Taken at fixed
  T, so `_ad` misnames its thermal axis exactly as `cs2_eq` did — but it is a
  DIFFERENT quantity from `did`/`njl`/`ccdm`'s `cs2_adiabatic` (frozen
  composition, not fixed entropy), so renaming it to that name would collide
  two quantities under one key. Needs a decision, not a sweep.
  [Ticket 72](72-dd2-remaining-cs2-names.md).

The ticket's Done condition — "a grep for `cs2_eq` over `eos/` returns
nothing" — is therefore **not** met and should not be: three of its remaining
sites are one open decision (53) and one new one (72), and the fourth
(`dd2/backends/responses_jac.py`'s demo print label) follows whichever way 72
goes.
