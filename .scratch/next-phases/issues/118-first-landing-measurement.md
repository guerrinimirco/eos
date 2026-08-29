# The landing measurement has never actually been taken

Type: task
Status: resolved (2026-08-29)
Blocked by: -   (117 ruled the gate; this is the first run under it)
Parent: ../map.md

## Question

Nothing to decide. [Ticket 117](117-suite-gate-needs-a-landing-point.md) ruled
that the full suite is a landing measurement bound to a SHA, taken through
`test/run_clean_suite.sh` and cited by certificate path. **No such certificate
exists.** The only file the store ever held was written by the fail-open mutant
and has been deleted, so the mechanism has certified nothing yet and the
repository has no full-suite number that names its interpreter and its HEAD.

The work is: wait for a landing point — `git status --porcelain -- 'eos/*.py'`
empty, which is a state to DECLARE and hold, not a gap to fall into — then

    sh test/run_clean_suite.sh

and commit the certificate. `--check` polls the window without spending twenty
minutes to discover it was open.

Three things this ticket is NOT allowed to do, all of them §12's rules applied
to the first case:

- **Fabricate or borrow.** If it comes back DISCARD, the DISCARD is the result
  and it is reported. A later CLEAN does not erase it.
- **Freeze the tree to protect the run.** A DISCARD is twenty minutes of CPU;
  another session's work is worth more. Take the run when the tree is quiet
  because the work landed, never by asking anyone to stop.
- **Report a bare count.** The certificate carries the interpreter and the HEAD
  SHA because this machine has two Python stacks that disagree and six of
  thirteen models cross the rtol = 1e-10 gate between them.

Note the expected result is NOT green on both stacks: `test/baseline/*.npz` are
Python 3.14 artifacts now, so anaconda 3.9.7 reds four of them and that is not
a regression. The certificate naming its interpreter is what makes the number
readable at all.

## Resolution

**The measurement exists.** Certificate:

    test/suite_certificates/20260829T103554.txt

    opened      2026-08-29T10:35:54       (see the timestamp note below)
    closed      2026-08-29T10:55:03
    HEAD        286da5f
    interpreter CPython 3.14.2, numpy 2.3.5, scipy 1.17.0
    eos/ before d2a05904b9897e572e98cd146dfb58679a0e3c22
    eos/ after  d2a05904b9897e572e98cd146dfb58679a0e3c22
    verdict     CLEAN -- no eos/*.py changed during the run

    1850 passed, 23 skipped, 0 failed  in 1148.61s (19:08)

**This is the only certificate this work produced. There were no DISCARDs**,
so the obligation to cite every one of them is discharged by citing this one.
The store held nothing before it; it now holds exactly this file.

### The landing point had to be CREATED, and by a commit

The ticket assumed 112 and 114 were in flight. They were not: both were
**resolved at ~02:38**, the newest `eos/*.py` write was 7.5 hours old and no
pytest was running, but their thirteen `eos/*.py` files were still
**uncommitted** -- and 114's own resolution had deferred the commit
("whoever commits stages those by name"). So the window was open and the
landing point did not exist, which is exactly the state 117 defined as not
measurable: a certificate naming HEAD `b0e5d53` while measuring thirteen
modified files is a number that is not a property of that SHA.

Nobody was asked to stop, because nobody was working -- the tree was quiet
because the work had FINISHED, not because anyone held still. What was missing
was the commit, and that is an authorization, so it was put to the human and
taken on their instruction: **`286da5f`**, the 112 + 114 set staged by name
(no `git commit -a`; the shared-tree trap this map has recorded four times).

**One commit, not two, and deliberately.** 114's file list and 112's interleave
inside `eos/sfho/nmp.py` -- 114 owns its last hunk, 112 the rest -- so any
split leaves an intermediate tree where `eos/sfho/parameters.py` carries the
`y_phi_*` fields that `eos/sfho/nmp.py` does not yet read. A broken
intermediate commit is worse evidence than a shared one, and the commit
message says so rather than hiding it.

### What the count is, and is not

1850 against ticket 114's 1833 and ticket 94's 1816: the +17 is 112's and
114's own new tests, landed in `286da5f`. The number is a property of that
SHA on **python.org 3.14.2**, the canonical stack ruled by
[ticket 57](57-canonical-stack.md).

**anaconda 3.9.7 was NOT run.** Four `test/baseline/*.npz` red there because
those are 3.14 artifacts since [ticket 62](62-regenerate-baselines-py314.md),
which is a known non-regression -- so a second twenty-minute run would have
bought a number already documented. Read this certificate as measuring one
stack, which is why it names it.

### A defect in the certifier, found by using it

The first real certificate carried `opened 10:55:03` -- **the CLOSE time under
an `opened` label**, disagreeing by nineteen minutes with its own filename
`20260829T103554.txt`. The script built the filename from `date` at the top and
then called `date` AGAIN inside the heredoc, which runs after the suite. That is
the fourth defect in this certifier and the third that made it lie in the
direction of looking fine.

Fixed in `test/run_clean_suite.sh`: the open time is stamped **once**, into
`OPENED`, and the filename is derived from it, so the two cannot drift.
`closed` is now a field of its own. `sh test/test_run_clean_suite.sh` ->
**11 passed, 0 failed**.

**The certificate above was NOT edited to match.** An evidence store is not
something you go back and correct; its filename already carries the true open
time, and this note is where the discrepancy is recorded.
