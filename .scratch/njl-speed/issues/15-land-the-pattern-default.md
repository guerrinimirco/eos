# Land the pattern-default decision

Type: task
Status: open
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
