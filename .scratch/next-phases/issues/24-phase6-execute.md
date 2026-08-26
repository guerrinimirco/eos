# Execute Phase 6 — the port: make `nucleation` import `eos` again

Type: task
Status: in progress
Assignee: session 9a857509
Blocked by: 23
Parent: ../map.md

## Question

The first half of [ticket 23](23-phase6-respec.md)'s corrected brief. Read that
brief before starting: it carries the measured mapping target by target, and
three of the premises the old Phase 6 text rests on are false.

On `nucleation` branch `paper-release`, at
`/Users/mircoguerrini/Desktop/Research/Python_codes/nucleation`. **Confirm the
path with the user before writing anything outside `eos/`** — the prompt's
Stage 7 requires it and the map does not lift that particular gate.

### Before anything: the tree, then the before-image

**DONE, and one of the two instructions was wrong.** Recorded here because the
correction matters more than the action.

Ticket 23's brief said to commit the 16 modified paper PDFs, on the grounds that
"a figure regeneration is a real artifact". **There was no regeneration.**
Measured byte by byte against HEAD, all sixteen differ by **3 to 6 bytes each,
every one inside a `/CreationDate` stamp**, same matplotlib 3.10.9, zero content
change. Committing them would have put sixteen binary blobs into a paper
repository's history to record a timestamp. They were **discarded**, not
committed.

`docs/nucleation_physics.md` (28.9 kB) and `docs/reproducing.md` (8.5 kB) were
**restored** — the four live references stand, including
`nucleation/tables/quantum.py:7` citing "section 7" of the first.

`nucleation`'s tree is now **clean**, and its HEAD has moved to `cad424b`
(ticket 62's session pinned the stack there too: `requires-python >= 3.11`,
numpy >= 2.0, scipy >= 1.17). **Take the before-image against `cad424b`, not
`33c1e61`** — run the suite against `eos` HEAD and record it verbatim as
`nucleation_before_py314.txt`. It will be mostly collection errors:
`nucleation` cannot import `eos` and has not been able to since Phase 3. That
wreck is the instrument this ticket is measured with.

### The port

Seven import targets, every one with an in-place successor; the mapping table is
in the brief. Six are mechanical. One is not:

**The total-thermo assembly.** `compute_alphabag_total_thermo_from_mu` and its
CFL twin have no successor. `nucleation` **keeps its own saddlepoint solver** and
assembles the total itself from five already-public pieces —
`alphabag.thermo_from_mu` / `cfl_thermo_from_mu`, `alphabag.gluon_thermo`, and
`general.thermodynamics_leptons.{electron_thermo, photon_thermo,
neutrino_thermo}`. The three `include_*` booleans become three `if`s. **No new
`eos` code.** Do not route this through `alphabag.eos_point`: that would hand
`eos` the 4-vector solve that is the reason `nucleation` is a package.

**The parameter helper.** `get_alphabag_custom(alpha=, B4=, m_s=)` has no
successor, but `Parameters` carries exactly those fields. One small helper in
`nucleation` wrapping `dataclasses.replace(Parameters.default(), ...)` — not
seven inline copies, not a new `eos` constructor.

### Out of scope, deliberately

- **The paper notebook.** Its port is mechanical (4 import lines, 2 name swaps)
  but its gate is execution, and its own header says the runs take hours.
  [Ticket 72](72-phase6-conformance.md).
- **The test move.** This ticket's gate is a node-id-to-node-id comparison;
  moving the files underneath it destroys the only instrument it has. Ticket 72.
- Anything in ticket 72's list.

### Gate

- Every failure caused by an import or a call site goes to **zero**.
- **Every failure that survives is REPORTED, not fixed.** A survivor belongs to
  ticket 72 or to a ticket of its own. Do not let this ticket be conscripted
  into fixing pre-existing breakage it did not cause.
- Report `pytest` for `nucleation` verbatim, with the interpreter and the
  collected count, against the before-image.
- Re-run the §1 dependency check **in both directions** and report both. The
  reverse direction is already gated by
  `eos/test/test_imports.py:41 test_eos_never_imports_nucleation` and passes
  today; the forward direction is what this ticket restores.

### After the gate

**Push to `origin/paper-release`.** The no-push rule is lifted: the premise that
produced it ("nucleation has no git remote") is false —
`github.com/guerrinimirco/metastability-nucleation` exists. A public repo that
imports is worth more than a polished one that does not, and the tests are
tracked, so this push is what makes them publicly runnable.
