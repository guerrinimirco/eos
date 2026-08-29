# The landing measurement has never actually been taken

Type: task
Status: open
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
