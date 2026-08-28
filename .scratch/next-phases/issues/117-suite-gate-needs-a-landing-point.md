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
  the reason the raw counts were. The obvious check — dirty the tree mid-run,
  assert DISCARD — does not exist.

## The distinction worth keeping whichever way it goes

**The guard reduces the chance of WASTING a run; the certificate removes the
chance of BELIEVING one. Only the second is a guarantee.** No local check can
promise a 20-minute window exists, so the two-minute threshold is a floor
against the obvious cases, never a predictor.
