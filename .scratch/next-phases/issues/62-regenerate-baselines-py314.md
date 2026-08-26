# Regenerate test/baseline/ on the canonical stack, and pin it

Type: task
Status: resolved
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

## Resolution

**Twelve of thirteen regenerated on python.org 3.14.2. `enjl` stopped the
regeneration and is a finding, now [ticket 72](72-enjl-branch-selection.md).**

### The stop condition, discharged first

The user's hand copy was found at
`/Users/mircoguerrini/Desktop/Research/backups/baseline/`, taken 2026-08-25
18:46, all thirteen `.npz` **byte-identical** to the live 3.9 set. Nothing was
discarded regardless: the 3.9 files were copied to `test/baseline_py39/` before
the generator ran and are still there. The verified 3.14 set was then added
beside the old one as `backups/baseline_py314/`, with a `backups/README.txt`
naming which directory is which stack — additive, so the superseded set is
preserved and labelled rather than left looking like a current backup.

### The screen, and what it found

Every key of every model compared old against new at the suite's own gate
(rtol = atol = 1e-10). 630 of 53763 keys moved, in six models:

| model | moved | verdict |
|---|---|---|
| abpr alphabag did mixed sfho vmit zl | 0 | bit-identical across stacks |
| tov | 6 | max rel 1.7e-08 — ODE + root-find round-off |
| zlvmit | 33 | max rel 1.16e-08, one density |
| ccdm | 106 | 79 residual-norm keys, 17 numerically-zero, 10 undetermined-potential |
| njl | 28 | 3 residual-norm, 15 numerically-zero, 10 undetermined-potential |
| dd2 | 3 | `nmp.Q_sat`, `K_sat`, `K_sym` only |
| **enjl** | **454** | **branch flip — STOPPED** |

**The `C_i` fingerprint fired, exactly as specified.** In both `ccdm` and `njl`
the only value-carrying shifts are in the **CFL** pattern, where §3 says the
locking leaves no free charge fraction, so `mu_3 = mu_C` is undetermined. The
shifts are `5.572e-05` (ccdm) and `1.291e-05` (njl), carried identically by
`mu_3`, `mu_C`, `x`, `state.mu_modes` and `state.mu_star` — and `mu_8` moves by
**exactly half** in both models, its coefficient in the projection. One
undetermined potential, propagated through the basis map, in two independently
written models. Every determined quantity — P, eps, gaps, densities — is
bit-identical. `enjl` also shows the sibling in `mu_S` at Y_S = 0
(7.3e-04 MeV over three densities, against the 0.079 MeV that ticket 40
measured a 1e-10 gate to admit), with every other entry of `x` bit-identical.

**One blessing does not rest on the round-off screen, and should stay visible.**
`dd2`'s `nmp.Q_sat` moved 0.351 MeV (168.65 -> 169.00), which is not round-off.
It is blessed on [ticket 47](47-dd2-nmp-inversion.md)'s diagnosis — a third
finite difference whose documented noise floor is >=0.1 MeV and whose h-sweep
spans 2.7 MeV — corroborated here by all 4689 other `dd2` keys being
bit-identical, which isolates the move to the stencil rather than the physics.
Q4 of ticket 47 ("does Q_sat belong in a frozen rtol=1e-10 baseline? No") is
unchanged by this, and `test_api.py`'s `abs=0.2` was NOT touched.

### Why enjl stopped it

`fixed_YC_YS` at Y_C = 0.5, Y_S = 0, `leptons=False` selects a **different root
of the gap equations** on the two stacks over six contiguous densities,
n_B = 0.300 to 0.467 fm^-3: 3.9 stays chirally broken (M_q 260 -> 216 MeV, no
quarks, P = +12.4 to +66.6) while 3.14 enters the restored branch six points
early (M_q -> 5.5 MeV, n_u = n_d ~ 0.08 fm^-3, P = -41.1 to +18.3). Both report
`converged = 1`; both are roots. Nothing scales with any conserved charge, so
the fingerprint says physics, not a potential.

And **neither branch is right across the window**: at T = 0 and fixed n_B the
stable root is the lower-eps one, and that crosses inside the window — broken
is lower to n_B = 0.400, restored from 0.433. So the first-order chiral
crossing sits near 0.41 fm^-3 and both baselines ride a metastable branch past
it. `enjl.npz` is therefore still the 3.9 file, `test_baseline[enjl]` is red on
the canonical stack **on purpose**, and the reason is written into the
generator's module docstring so a future session cannot re-bless it by reflex.

### Part 2, the pin

- `eos/pyproject.toml`: `requires-python = ">=3.11"`, `numpy>=2.0`,
  `scipy>=1.17`, with the tested stack recorded in a comment. The scipy floor
  is a correctness floor, not a preference: ticket 47 measured 1.13's `hybr`
  returning its seed unchanged while reporting success.
- `nucleation/pyproject.toml`: the same floor, same reasons.
- `eos/README.md`: badge 3.9+ -> 3.11+, the install line, a new **"The stack the
  suite is run with"** block naming CPython 3.14.2 and `python3 -m pytest`
  rather than whichever `python` a shell resolves, and the Examples provenance
  sentence.
- `nucleation/README.md`: Python >= 3.11.
- `matplotlib` was deliberately left unpinned. A first draft pinned `>=3.5`
  citing `paper_grid`; `figure_style.py:337` carries an explicit 3.4 fallback
  for `layout='constrained'`, so the floor would have orphaned working code.

**All five README examples reproduce bit-identically on 3.14**, so not one
printed digit changed — only the sentence saying which stack printed them.

### Suite

Both runs python.org 3.14.2, collected counts included per the map's rule:

    before  output/_audit/pytest_before_ticket62_py314.txt
            12 failed, 1667 passed, 15 skipped   (1694 collected, 22:29)
    after   output/_audit/pytest_after_ticket62_py314.txt
            7 failed, 1674 passed, 15 skipped    (1696 collected, 21:05)

The before-run's twelve are the same node ids as
`pytest_after_ticket56_py314.txt` — 0 added failures, denominator up 29.
Five baseline failures cleared (ccdm, dd2, njl, tov, zlvmit); `enjl` stays red
by design.

**The after-number is not a clean measurement and must not be quoted as one.**
A second session was committing to `main` throughout: `5c75584` at 10:40 (28
files under `eos/`) and `5a4a6cc` at 11:49, the latter roughly four minutes
into a run that started at 11:45, and collection moved 1694 -> 1696 mid-session.
What IS clean is the baseline verification itself, re-run against HEAD at
12:10 after those commits landed: **1 failed, 15 passed** — the twelve
regenerated files reproduce against the current tree, so the regeneration is
sound irrespective of the concurrency.

### Not resolved here, and not this ticket's diff

The six survivors are the rest of [ticket 57](57-canonical-stack.md)'s cost
list, none of which is a `.npz`: `test_api.py:127`/`:143`'s `abs=0.2` on Q_sat
re-derived from a noise floor measured on 3.14; `test_dd2_m8.py`'s
(K_sat, Q_sat) = (240, 300) premise re-measured; and
`test/tov/test_solver_fast_robustness.py`'s three cases given a sample the 6x6
closure can reach on 3.14. Ticket 57's "green on 3.14" is therefore NOT met by
this ticket alone, and saying so is the point — those four items need their own
ticket.

### Reported, not fixed

`generate_baseline.py`'s documented invocation does not work: `eos` is
installed in neither site-packages, so the script's own
`python test/baseline/generate_baseline.py` dies at `import eos` — only the
tests import, via `conftest.py`'s path insert. Used `PYTHONPATH=.` and recorded
that in the docstring. Separately, README example 2's captured output shows
`[1/3] fixed_YC T=0` where the progress printer now emits
`[1/3] fixed_YC T=0 Y_C=0.3`; that is drift from another ticket, not the stack,
and is left for the Stage 7 report.
