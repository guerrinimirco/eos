# Regenerate test/baseline/ on the canonical stack, and pin it

Type: task
Status: open
Blocked by: 57
Parent: ../map.md

## Question

Execution of [ticket 57](57-canonical-stack.md)'s ruling: **python.org 3.14 is
canonical.** Two pieces, and the first has a stop condition.

### 1. Regenerate the thirteen `.npz` on 3.14

Today they were made on anaconda 3.9.7 / numpy 1.26.4 / scipy 1.13.1 against
**OpenBLAS 0.3.23**; the canonical stack is 3.14.2 / numpy 2.3.5 / scipy 1.17.0
against **Apple Accelerate**. That is why 12 tests fail on 3.14 and 0 on 3.9.

**Keep the 3.9 files until the new ones are verified against them.** Every
difference must be round-off at the `rtol = 1e-10` gate. **Anything larger stops
the regeneration and is reported** — §12 makes these ground truth, and a
regeneration that quietly absorbs a real change destroys the only thing that
would have caught it.

Cheap first screen, from the map's Not-yet-specified section: shifts in the
**strangeness ratios 1 : 2** (four S = 1 species moving together, two S = 2
species at exactly twice) mean an undetermined potential moved; anything else
means the physics moved. The same algebra through `C_i` catches the `mu_e`-at-
Y_C = 0 sibling.

The user is keeping a hand copy of `test/baseline/` outside the repo (§11 stands,
`test/` is not tracked). **Confirm that copy exists and post-dates this
regeneration** before the old files are discarded — a copy of the superseded set
is worse than none, because it looks like a backup.

### 2. Pin the stack

`pyproject.toml:5` reads `requires-python = ">=3.9"` with unpinned numpy/scipy,
in **both** `eos` and `nucleation` — it admits both stacks and picks neither. On
this machine `python` is anaconda 3.9.7 and `python3` is python.org 3.14.2: two
major versions behind two near-identical command names on one PATH, which is how
this map already lost a full suite run.

Raise `requires-python`, record the tested numpy/scipy, and say in the README
which interpreter the suite runs with. Python 3.9 is end-of-life (October 2025)
and `nucleation` is headed for a public remote, so `>=3.9` would ship a library
requiring a dead interpreter.

### Not urgent

Nothing on the notebook critical path waits on this. It changes what a failure
count MEANS, not what any ticket may do — so it runs in parallel, and until it
lands the map's rule holds: report the interpreter and the collected count with
every failure count.

Resolved when the suite is green on 3.14 with every difference shown to be
round-off, and the stack is pinned in both repositories.
