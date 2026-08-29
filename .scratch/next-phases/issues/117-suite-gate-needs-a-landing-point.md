# A "full suite green" gate assumes a tree that holds still, and this one does not

Type: grilling
Status: resolved (2026-08-29)
Blocked by: -
Parent: ../map.md

## Question

Graduated out of the map's fog on 2026-08-29 by
[ticket 97](97-natural-record-leaves-the-result.md), whose §12 line it now
blocks — so this is load-bearing rather than theoretical.

**§12 says the full suite must pass before any commit that touches solver
internals.** That sentence assumes the suite can be RUN, which assumes the tree
holds still for the ~20 minutes it takes. During active work it does not.

Measured, not asserted. A waiter armed on "no pytest running AND no `eos/*.py`
written for two minutes" fired legitimately and was re-checked 29 seconds later:

    01:30:28  WINDOW OPEN: no pytest, newest eos/*.py 129s old
    01:30:57  REFUSED: eos/*.py written 7s ago -- eos/dd2/nmp.py

**The window was open for twenty-nine seconds.** A suite started on that ping
would have been invalidated half a minute in and the invalidation discovered
twenty minutes later. Tickets 115 and 116 were mid-flight with five uncommitted
`eos/*.py` between them; while work is uncommitted the window reopens at every
lull and shuts on the next keystroke.

**So the condition is wrong, not the implementation.** A two-minute timer
measures a PAUSE — someone reading a failure, someone thinking. Fishing for a
20-minute gap inside that is a lottery where each losing ticket costs 20
minutes to discover it lost.

The decision is what §12's gate should MEAN:

- **(a) Run it at a declared landing point.** The condition becomes "no
  uncommitted `eos/*.py`" — the tickets have LANDED. That is a state someone
  CHOOSES to enter rather than a gap to fall into, and it cannot flicker. Costs
  a coordination step: somebody has to declare and hold it.
- **(b) Per-ticket subsets, full suite periodic.** Each ticket gates on the
  subsets its change can reach (which is what ticket 97 actually has: 349 + 391
  + a baseline audit, each on a tree only its own session was editing), and the
  full suite becomes a periodic check owned by nobody in particular. Cheaper
  per ticket, and it gives up the §12 sentence as literally written.

Note that ticket 97 already demonstrates (b) working: its evidence is three
clean subset runs and a bit-identity audit, all of which held, while its one
attempt at the full suite was killed as unmeasurable. The question is whether
that is an acceptable gate or a gap being normalised.

## What the answer must also settle

- **§12's wording.** If (a), the section should say the gate is run at a
  landing point and what declares one. If (b), it should say which subsets a
  ticket owes and how often the full suite runs — otherwise "the full suite
  must pass before any commit" stays in the text while nobody does it, which is
  worse than either arm.
- **Where the tooling lives.** `test/run_clean_suite.sh` is the mechanism, and
  `test/` is gitignored (§11), so it has no history, no review, and no ticket
  requiring its use.
- **What checks the certifier.** It had three defects in its first twenty
  minutes, two failing OPEN — a live-pytest guard matching every session's
  WAITER loop, and a tree-quietness guard using `find -newermt` where `find` is
  a shim to `bfs` that rejects the relative timestamp, prints to stderr and
  EXITS 0. A mechanism trusted because it prints CLEAN is trusted for exactly
  the reason the raw counts were. **Partly answered on 2026-08-29**:
  `test/test_run_clean_suite.sh` now exists, 11 cases — a quiet tree certifies
  CLEAN with the interpreter and HEAD on the certificate, a tree written DURING
  the run certifies DISCARD and exits non-zero, a freshly-written tree is
  REFUSED and the suite does not run at all, and `--check` reports without
  starting anything. Confirmed 11 passed from a second session's shell. Its
  case 3 is a regression test for a TRAP rather than for a line of code, which
  is why it constructs its own fresh file: any check against a quiet tree
  passes, and only a deliberately-fresh file separates the good guard from the
  blind one.

  **ANSWERED on 2026-08-29, and reproducibly.** `SCRIPT` is now a seam
  (`SCRIPT=${SCRIPT:-test/run_clean_suite.sh}`), so the mutation check is two
  commands from the repo root rather than something one session did once:

      CONTROL   sh test/test_run_clean_suite.sh                    -> 11 passed, 0 failed
      MUTANT    SCRIPT=test/.mutant.sh sh test/test_run_clean_suite.sh -> 9 passed, 2 failed

  The two failures are case 3's assertions, and the mutant's output is the
  pathology verbatim: the suite runs on a tree written moments earlier and
  prints "CLEAN: the count above is a measurement" over it. Verified by a
  SECOND session writing its OWN fail-open mutant rather than rerunning the
  author's — same result, which is what makes it a property of the test rather
  than of one person's mutant.

  **The mutant must live inside `test/`.** `run_clean_suite.sh` does
  `cd "$(dirname "$0")/.."` to find the repo, so a mutant in `/tmp` resolves
  the repo root to `/` and the script operates on the root filesystem and dies
  — `mkdir: test: Read-only file system`, or a `git rev-parse` failure, or an
  arithmetic error, depending which line it reaches first. Running from the
  repo root is necessary and NOT sufficient. Two sessions hit this
  independently before the cause was found, both reading it as a rig problem of
  their own making.

## The hazard the certificate CREATES, which the gate must also settle

A certificate moves the failure mode; it does not remove one. Before it, the
risk was **believing an invalidated run**. After it, the risk is **re-running
until one comes back CLEAN** — which is the same sin wearing the mechanism's
own clothes, and harder to see because every individual certificate is honest.
Two properties follow, and whichever arm wins should state them:

- **A DISCARD is reported, not retried into silence.** The verdict of the run
  you took is the verdict, and a later CLEAN does not erase an earlier DISCARD
  that was never mentioned.
- **A DISCARD is CHEAP and must stay cheap.** Twenty minutes of CPU and nothing
  else. So nobody should block real work to protect someone's measurement:
  holding `eos/*.py` still for twenty minutes to keep a run valid inverts which
  of the two matters. The certificate exists precisely so that an invalidated
  run is cheap to DETECT rather than expensive to BELIEVE — that is the whole
  trade, and a session sitting on its hands has paid for the mechanism twice.

## The distinction worth keeping whichever way it goes

**The guard reduces the chance of WASTING a run; the certificate removes the
chance of BELIEVING one. Only the second is a guarantee.** No local check can
promise a 20-minute window exists, so the two-minute threshold is a floor
against the obvious cases, never a predictor.


## Resolution (2026-08-29)

**Neither (a) nor (b): the sentence was welding together two measurements with
different SUBJECTS, and that is why it could not be satisfied.** "Did this
change break anything" is a property of a CHANGE — its blast radius is
knowable, and the reachable subsets measure it better than the full suite does,
being faster, targeted, and run on a tree the session controls. "What is the
suite's state" is a property of a SHA. §12 attached the second to the first's
event ("before any commit"), so the gate could only be met by a tree nobody
owns holding still for twenty minutes.

So the ruling is the split:

- **the per-commit gate is (b)** — every test the change can reach, with what
  ran and why those are the reachable suites stated alongside the result;
- **the full suite is (a), demoted from a gate to a LANDING MEASUREMENT** bound
  to a SHA, taken at a landing point through `run_clean_suite.sh`, and cited by
  path.

(a) could not have been the per-commit gate on its own terms anyway: "no
uncommitted `eos/*.py`" is a property of the TREE, so the moment you commit and
the next session starts typing, your successor's window is gone — every commit
would gate on everyone else's idleness.

**The obvious third arm is closed by the record, not by preference.** "Run it
somewhere the tree cannot move" fails twice over: a `git worktree` has no suite
(`test/` is gitignored), and the `git archive` + `cp -R test` copy has no numba
cache — `test/mixed` runs for HOURS where the repo takes minutes, and copying
`eos/**/__pycache__` in did not help — while also inventing six
`test/abpr/test_abpr_inverses.py` failures that read as yours. Isolation is
affordable for subsets and unaffordable for the full suite.

### The ticket's phenomenon fired inside the ticket

`git status --porcelain -- 'eos/*.py'` was **empty** when this session opened,
and carried **ten modified files** (dd2, sfho and zl `nmp.py` among them) by
the time the first round of questions was written — minutes later. A landing
point existed, was not declared, and was gone before anyone could spend it.
That is the twenty-nine-second window again at a different scale, and it is why
**the landing measurement was NOT taken here**: no certificate is claimed by
this resolution, and none is fabricated.

### What checks the certifier: answered from the other end

The store the mechanism writes to had exactly one file in it, and **the
fail-open MUTANT wrote it.** `test/suite_certificates/20260829T013708.txt`,
timestamp matching commit `1bf091b`'s mutation check:

    HEAD        7deeb06
    interpreter CPython 3.14.2, numpy 2.3.5, scipy 1.17.0
    verdict     CLEAN -- no eos/*.py changed during the run

    ran anyway

Real HEAD, real interpreter, verdict CLEAN, and the rig's stub string where the
pytest count belongs. `TREE` and `SUITE` were seams; the certificate PATH was
not, so every rig case reaching past the guards wrote into the real evidence
store — and case 3 only reaches past them when the script is broken, which is
why the first certificate this repository ever held was produced by a
deliberately-defective certifier. Case 1 removes its own certificate; nothing
removed that one. **An evidence store its own test can write to is not
evidence**, and this is the same family as 1c/1d: not a check that fails open,
but a check whose EXHAUST is indistinguishable from the thing it certifies.

Fixed rather than sniffed for: `CERT_DIR` is now the third seam
(`CERT_DIR=${CERT_DIR:-test/suite_certificates}`), the rig exports it to its
own `$TMP`, and the mutant's certificate is deleted. Teaching the store to
recognise real pytest output would have been a fourth thing that can fail open,
in a repository whose recorded failure class is exactly that.

Re-measured after the change, both arms sequentially, and the store checked
afterwards:

    CONTROL   sh test/test_run_clean_suite.sh                          -> 11 passed, 0 failed
    MUTANT    SCRIPT=test/.mutant.sh sh test/test_run_clean_suite.sh   ->  9 passed, 2 failed
    test/suite_certificates/  after both arms                          -> EMPTY

Same two case-3 failures as before, so the seam did not weaken the rig; the
empty store is the new property.

### Where the tooling lives

**Tracked in place, by negating the ignore** — not moved. Moving is what breaks
it: `run_clean_suite.sh` finds the repo with `cd "$(dirname "$0")/.."`, so a
copy outside `test/` resolves the root to `/` and dies there, which is the trap
two sessions hit independently. And git cannot re-include a file whose parent
DIRECTORY is excluded, so `/test/` had to become `/test/*` first:

    /test/*
    !/test/run_clean_suite.sh
    !/test/test_run_clean_suite.sh
    !/test/suite_certificates/

Verified: `git check-ignore -v` still names `/test/*` for
`test/dd2/test_dd2_m8.py` and `test/baseline/dd2.npz`, and
`git status --porcelain -uall -- test/` lists exactly the two scripts. §11's
layout line now carries the exception.

### The retry hazard: disclosure, no cap

Rationing re-runs prices a DISCARD as expensive, which is the inversion the
ticket names — the certificate exists so an invalidated run is cheap to DETECT
rather than expensive to BELIEVE, and a session sitting on its hands has paid
for the mechanism twice. The sin is not re-running; it is re-running
UNMENTIONED. So: no cap, and one obligation — a result claiming a full-suite
number cites its certificate path AND every other certificate the same work
produced, DISCARDs included. Both properties are in §12 verbatim.

### What landed

- **CLAUDE.md §12**: the one-line sentence is gone, replaced by three bullets —
  the reachable-subset commit gate, the landing measurement (with the
  twenty-nine-second measurement quoted, the certificate's fields named, and
  "a count naming no interpreter names nothing"), and the DISCARD/disclosure
  rule. **§11**'s `test/` line carries the three tracked exceptions.
- `test/run_clean_suite.sh`: `CERT_DIR` seam, header updated to three seams
  with the mutant-certificate finding recorded in it.
- `test/test_run_clean_suite.sh`: exports `CERT_DIR` to its own `$TMP`.
- `test/suite_certificates/20260829T013708.txt`: deleted.
- `.gitignore`: `/test/` -> `/test/*` plus three negations.

### Ticket 97's §12 line, retroactively

**Satisfied, and it always was under the ruling.** 97's per-commit gate is the
tests its change could reach, and it ran and reported them: `test/njl test/ccdm
...` 349 passed, `test/enjl test/mixed` 391 passed, `test/test_imports.py` 215
passed / 0 xfailed, plus the bit-identity baseline audit
(`np.array_equal(equal_nan=True)` against a pre-image, zero surviving keys
changed by a single bit, independently reproduced by a concurrent session).
That IS the gate now. The full-suite run it killed was never 97's to owe: it is
a property of a SHA, it belongs to a landing point, and killing a compromised
run was the correct act rather than a shortfall. 97's OUTSTANDING marker is
cleared with no new run required.

### Still owed by somebody, and it is not a decision

No real certificate exists yet — the mechanism has never certified an actual
20-minute suite. Raised as [ticket 118](118-first-landing-measurement.md).
