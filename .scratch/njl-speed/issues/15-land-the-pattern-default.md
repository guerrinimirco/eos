# Land the pattern-default decision

Type: task
Status: closed
Assignee: guerrinimirco
Blocked by: 12
Parent: ../map.md

## Question

Nothing to decide: [Does `free` belong in the default
enumeration?](12-free-in-the-default.md) decided it. This ticket makes the
decision real, so that tickets 08, 09 and 11 measure the **default** rather
than a `patterns=` argument each of them has to remember to pass.

### The edit

    eos/general/pairing.py:1650
    -  DEFAULT_PATTERNS = ("unpaired", "2SC", "CFL", "free")
    +  DEFAULT_PATTERNS = ("unpaired", "2SC", "CFL")

`free` stays in `PATTERNS` and stays requestable. Nothing raises.

### The four documentation sites, and one tracked entry

1. `eos/general/pairing.py:1631-1650` -- the comment block above `PATTERNS`
   already says what `free` is FOR; it gains why it is not enumerated by
   default (it cannot be cached, and the named asymmetric seeds dominate it).
   Plus the `DEFAULT_PATTERNS` docstring.
2. `eos/njl/njl.md:1147` and `:1198`, `eos/njl/njl.tex:1242`.
3. `eos/ccdm/ccdm.md:921` and its `.tex`.
4. `eos/njl/njl.md` section 17.8 -- the Gholami passage that makes the physics
   claim gains the sentence that the asymmetric sector is reached by `uSC` and
   `dSC` under their own names.

Plus **one `docs/DEFERRED.md` entry for the open physics question only** --
where the asymmetric sector wins -- because `.scratch/njl-speed/` is untracked
and that question would otherwise die with this map. Not for the default
change, which is stated where the default is stated.

No new constant. `table.py`'s `NOT_A_BRANCH` is not the precedent: that one
makes a name RAISE.

### The measurement this task owes

`ccdm` has never been measured here, and ticket 12 bound the default for both
models on a structural argument. Confirm it: `ccdm` with `csc=True`, a handful
of densities, `free`'s realised pattern and `f` against the named candidates.
If `ccdm` shows `free` finding something `uSC`/`dSC` do not, ticket 12's
"one default, both models" reopens.

### Gate

- The edit and the five documentation changes landed on `njl-speed`.
- The `ccdm` confirmation, with its numbers.
- CLAUDE.md section 12's reachable set, run **one directory at a time**
  (`test/njl` and `test/mixed` both hold a `test_jacobian.py` and collide at
  collection, per ticket 01): `test/njl`, `test/ccdm`, `test/mixed`,
  `test/baseline` -- with the interpreter and its numpy and scipy versions
  named, and what ran and why those are the reachable suites stated with the
  result.
- `test/baseline`'s `enumeration.n1.2.{f,Delta}` still passing at rtol = 1e-10
  (ticket 12 measured the change at 1.9e-13, ~500x inside it).

---

## Resolution, 2026-09-21

**Landed, and the ccdm confirmation holds on what a caller receives -- but it
does not hold on ticket 12's structural premise, and it turned up a ccdm
defect that predates this effort.**

The edit and the five sites this ticket named were already in `2536d2b`
(`DEFAULT_PATTERNS` and its comment block in `eos/general/pairing.py`,
`njl.md`/`njl.tex`, `ccdm.md`/`ccdm.tex`, `njl.md` section 17.8, and the
`docs/DEFERRED.md` entry "njl, ccdm: where the asymmetric pairing sector wins
is not known"). What this close adds is the sites the ticket did not list,
the ccdm measurement, and the gate.

### Ten more sites, landed in `50b3b7f`

Every one still described the four-pattern default or ticket 05's refuted
account of what `free` costs:

- `eos/njl/table.py`, `TableSpec` `patterns`: called `('unpaired', '2SC',
  'CFL')` "the recommended fast restriction" and said `free` "burns its whole
  retry ladder -- measured at 90% of one paired point's cost". Now: it is the
  default; the asymmetric states are asked for by name; `free`'s cost is that
  `realised_pattern` never returns it, so it never carries its own seed and
  starts from another pattern's state at every density (ticket 05's 12.5x,
  stated as taken with the full rescue ladder).
- the same docstring's 9-point measurement: "with the defaults" is now "with
  'free' enumerated".
- `eos/njl/table.py`, `solve_nodes`: "the shipped defaults cost 6638" is now
  "the enumeration with 'free' in it cost 6638".
- `eos/njl/table.py`, the fast-table comment: 6 / 63 / 443 / 6638 ms/pt are
  replaced by the pinned 1.4 / 56.5 / 719 (ticket 02), CFL's 89% dead rows and
  80 ms/pt converged (ticket 03), and the three-pattern enumeration at 637
  ms/pt with the full ladder, half that bounded (ticket 16).
- `eos/njl/api.py`, `eos_table` `solve_nodes`: "6638 with the shipped
  defaults".
- `eos/njl/api.py` and `eos/ccdm/api.py`, `eos_point` `patterns`: both said
  the default "enumerates unpaired, 2SC, CFL and one asymmetric free seed".
- `eos/ccdm/table.py`, `TableSpec` `patterns`: the same recommended-restriction
  and retry-ladder story. Rewritten to what holds for BOTH models: ccdm's sweep
  carries only the winner's seed, so njl's "free never carries its own seed" is
  not ccdm's story, and the docstring does not claim it.
- `eos/njl/solver.py`, the module docstring: "`solve` enumerates seeds --
  unpaired, 2SC, CFL, and one asymmetric free seed".
- `eos/mixed/adapters.py`, `njl_phase`'s `seed`: "2230 ms over the default
  patterns" was measured with `free` in them; it now says so.

### The ccdm confirmation

`t15_ccdm_probe.py` is ticket 12's `t12_probe.py` pointed at ccdm:
`Parameters.default()`, `csc=True`, beta-eq, `backend="fast"`, each candidate
ALONE and COLD, at T = 0 / 30 / 50 MeV and n_B = 1.3 / 1.6 / 2.0 / 2.5 fm^-3.
That is ccdm's deconfined side; its tables start at 1.3. f in MeV/fm^3, python.org
3.14.2 / numpy 2.3.5 / scipy 1.17.0 (`t15_ccdm_probe.json`):

| T | n_B | lowest f (state) | `free` reaches | named candidate at the same state? | `free` vs lowest |
|---|---|---|---|---|---|
| 0 | 1.3 | 1770.922 (CFL) | CFL | CFL | 0 |
| 0 | 1.6 | 2188.918 (CFL) | CFL | CFL | 0 |
| 0 | 2.0 | 2800.954 (CFL) | 2SC | 2SC, and uSC collapsed there | +41.34 |
| 0 | 2.5 | 3655.347 (CFL) | CFL | CFL | 0 |
| 30 | 1.3 | 1762.237 (CFL) | **uSC** | **none** -- `uSC` collapsed to unpaired | +4.27 |
| 30 | 1.6 | 2180.765 (CFL) | **uSC** | **none** -- `uSC` collapsed to 2SC | +7.39 |
| 30 | 2.0 | 2787.162 (CFL) | **uSC** | **none** -- `uSC` collapsed to 2SC | +8.47 |
| 30 | 2.5 | 3619.676 (CFL) | CFL | CFL | 0 |
| 50 | 1.3 | **1720.398 (uSC)** | uSC | uSC, and CFL's candidate | 0 |
| 50 | 1.6 | 2135.462 (CFL) | CFL | CFL | 0 |
| 50 | 2.0 | 2728.835 (CFL) | uSC | uSC | +0.67 |
| 50 | 2.5 | 3537.568 (CFL) | CFL | CFL | 0 |

Summed over the twelve, cold: `free` 27.6 s, `uSC` 15.0, `dSC` 7.5, CFL 10.0.

**The reopen condition is met as written.** At three of twelve points, all at
T = 30, `free` reaches an asymmetric state neither `uSC` nor `dSC` reaches,
because the named `uSC` seed collapses there. In ccdm it is the NAMED seed that
is the less reliable probe at T = 30, the reverse of what ticket 12 found in
njl, so ticket 12's premise that the named seeds dominate `free` does not
carry over to ccdm.

**What a caller receives does not move.** All three of those states lose, by
4.3 to 8.5 MeV/fm^3. The three-pattern and four-pattern defaults, run as
enumerations at all twelve points, return the identical state and f to the
last printed digit (`t15_ccdm_enumeration.json`). So "one default, both
models" stands on the delivered rows, and nothing is reverted. The premise
that fails is recorded here and handed to ticket 14, which owns the
asymmetric sector.

### Two findings on the way, neither this ticket's to fix

1. **An asymmetric state wins in ccdm.** At T = 50, n_B = 1.3, uSC beats 2SC
   by 0.20 MeV/fm^3 (1720.398 against 1720.601), and the default enumeration
   reaches it, because the CFL candidate lands there. That is the first winning
   asymmetric state this effort has seen; njl's twelve-point box had none. It
   is evidence for the `docs/DEFERRED.md` entry and for ticket 14.
2. **ccdm's enumeration misses the ground state, and it did before this
   effort.** Both defaults, three patterns and four, return:
   - T = 30, n_B = 1.3: uSC, **+4.27** MeV/fm^3 above the CFL root;
   - T = 30, n_B = 1.6: usSC, **+22.56**;
   - T = 50, n_B = 1.6: sSC, **+6.04**.

   In each case the CFL root is the one the CFL candidate reaches from its own
   cold seed. The mechanism: cross-seeded from the unpaired state, the CFL
   candidate collapses to a lower-symmetry state, keeps the 'CFL' name, and
   competes. `eos/ccdm/solver.py` has no `_left_layout`, no re-seed and no
   `realised_pattern` at all. `free` never protected against it, and ticket 15
   did not cause it. **It needs its own ticket.** It is a wrong ground state
   in shipped code, at finite T, on the default call.

### Gate -- PASS

- The edit and the five named sites in `2536d2b`; the ten further sites in
  `50b3b7f`.
- The ccdm confirmation, above.
- CLAUDE.md section 12's reachable set, python.org 3.14.2 / numpy 2.3.5 /
  scipy 1.17.0, one directory at a time, `eos/*.py` fingerprinted either side
  of each run and stable: `test/njl` 132, `test/ccdm` 60, `test/mixed` 272,
  `test/baseline` 20, all passed. These are the reachable suites because the
  default is read by both models' enumerations and by `njl_phase`, and
  baseline pins every model. This is the same run as ticket 16's gate, taken
  on `50b3b7f`'s tree.
- `test_baseline[njl]`, which holds `enumeration.n1.2.{f,Delta}`, passes at
  rtol = 1e-10.

Artifacts: `t15_ccdm_probe.py`, `t15_ccdm_probe.json`,
`t15_ccdm_enumeration.json`.
