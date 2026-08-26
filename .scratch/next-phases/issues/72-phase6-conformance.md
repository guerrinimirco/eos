# Phase 6, second half — the conformance pass on `nucleation`

Type: task
Status: open
Blocked by: 24
Parent: ../map.md

## Question

The second half of [ticket 23](23-phase6-respec.md)'s corrected brief, split off
because it has a different gate from the port: the port is measured in minutes
against a test suite, this is measured by a notebook that takes hours and by
judgement.

**In scope of the map's destination, but NOT gating
[Acceptance](25-acceptance.md)** — see the Destination note. Ticket 25 needs
`nucleation` to import, run and respect §1, all of which ticket 24 delivers.

Four items. Phase 6's original list had six; two of them measure clean and one
has no meaning for a consumer package.

1. **The paper notebook**, `notebooks/2fam_PNS_nucleation.{py,ipynb}`. The port
   is mechanical — 4 import lines, and `AlphaBagTableSettings` /
   `compute_alphabag_table` -> `eos.alphabag.table.TableSettings` /
   `compute_table`, whose fields match one for one. **Then execute it.** It is
   what the paper reproduces from, so this ticket is not finished until it runs.
   Its own header warns the runs write to `output/paper/` and take hours; plan
   for that rather than discovering it.
2. **The test move.** `nucleation/nucleation/tests/` -> top-level
   `nucleation/test/`, **tracked, NOT gitignored**. `eos` hides its suite because
   it is private; `nucleation` is headed public, and publishing the repository
   behind a paper with no runnable tests is a real cost the layout parity does
   not justify. `make_fixture` moves with them and is invoked as
   `python test/make_fixture.py` — update the three docstring references in
   `conftest.py` and `make_fixture.py` that still say
   `python -m nucleation.tests.make_fixture`.
3. **`README.md`** to the standard of the new `eos` README, **with the examples
   actually run**.
4. **Dead code removed.**

### Measured clean — commission no sweep for these

- **Internal layering.** Already acyclic and layered: `barrier.py` at the bottom
  importing nothing internal, then `composition`/`critical`/`rates`/`tables`,
  then `analysis/`, then `analysis/figure/`. `eos`'s `general/` rule has no
  analogue to impose; it is satisfied in substance.
- **The docstring standard.** Nothing across 8,121 lines: no Phase/Stage/
  milestone reference, no TODO, no FIXME.

### Dropped

**"Apply the same API conventions."** `nucleation` is a consumer, not a model.
§5's uniform API is a contract for models, and imposing `eos_point`-shaped
signatures on a nucleation-rate sampler would be conformance theatre.

### Does not transfer from `eos`

**`nucleation`'s `output/` rule stands unchanged.** `.gitignore:32-38` already
ignores everything under `output/` except
`output/paper/{figures,figure_data,tables}` — 87 tracked files, the paper's own
figures and tables. That is §11's `output/public/` principle already correctly
specialised; flattening it to `eos`'s rule would untrack the paper's figures.
Do not "fix" it.

### Gate

The notebook executes to completion. `pytest` for `nucleation` still green at the
new paths, reported verbatim with interpreter and collected count. Then push to
`origin/paper-release`.
