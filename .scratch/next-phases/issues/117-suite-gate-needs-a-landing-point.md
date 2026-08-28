# A "full suite green" gate assumes a tree that holds still, and this one does not

Type: grilling
Status: open
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
