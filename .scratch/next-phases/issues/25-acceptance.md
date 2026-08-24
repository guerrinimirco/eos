# Run the Acceptance criteria block and write the Stage 7 report

Type: task
Status: open
Blocked by: 01, 21, 24
Parent: ../map.md

## Question

The Acceptance criteria block of `docs/REFACTOR_PROMPTS.md`, every line, with the
tool output behind each:

- `pytest eos/test/` and `pytest nucleation/test/` fully green
- the Phase 1 numerical baseline reproduces at **rtol = 1e-10**
- every model has a `.tex` that compiles — **unless ticket 09 removed that
  criterion**, in which case the replacement criterion is checked instead
- every model implements `eos_point()` and `eos_table()` with the spec signature,
  and its mode and species coverage matches what `CLAUDE.md` claims
- no Fermi or Bose integral implemented outside `eos/general/`
- model parameters are arguments everywhere; no solver raises or hangs on
  non-convergence; model objects pickle. **Show this with one script that
  evaluates a model at 500 random parameter sets across a multiprocessing Pool,
  counts the non-converged ones, and finishes.**
- `grep -rn "rcParams" eos/ nucleation/` hits exactly one file
- every README and STRUCTURE.md example executed, real output pasted
- no file over 5 MB newly tracked in git
- no new third-party dependency

Then the Stage 7 report: both suites verbatim, the baseline reproduction, the
**added-failure count against `output/_audit/pytest_before.txt`**, the list of
files created, changed and deleted, and every question from tickets 03, 05, 09,
10, 11 and 23 that is still open.
