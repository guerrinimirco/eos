# Re-specify Phase 6 against nucleation as it actually is

Type: grilling
Status: resolved
Assignee: session 9a857509
Blocked by: 20, 22
Parent: ../map.md

## Question

`docs/REFACTOR_PROMPTS.md` Phase 6 is written on premises that no longer hold.
Stage 7 orders it executed verbatim; this map executes it against corrected
premises instead, and this ticket produces them.

Known drift, measured while charting:

| Phase 6 says | Actually |
|---|---|
| "nucleation has no git remote… do not create or push a GitHub repo" | remote exists: `github.com/guerrinimirco/metastability-nucleation` |
| "11.6k lines across 40 files — one pass should do" | 8.1k lines across 38 files |
| (branch unstated) | on `paper-release`; work lands there directly |

Unchanged and still in force: fix every import and call site broken by the `eos`
changes; verify both directions of the §1 dependency rule (`nucleation` depends
on `eos`, `eos` never imports `nucleation`) including after the Phase 3 figure
move; apply the same treatment to `nucleation`'s own code (the `general/` rule,
the same API conventions, the same docstring standard, dead code removed); move
`nucleation/nucleation/tests/` to a top-level `nucleation/test/` and gitignore it,
matching `eos`; improve `nucleation/README.md` to the standard of the new `eos`
README with examples actually run.

**The breakage is much wider than one module, and it is all pre-existing.**
Measured against `eos` at HEAD by importing every target `nucleation` names —
**five of its `eos` modules do not exist and two more are missing the name it
asks for**:

| `nucleation` imports | today |
|---|---|
| `eos.tov.solver` (5 files + a notebook) | gone — it is `eos.astro.tov.solver` |
| `eos.alphabag.eos` | gone — §5 forbids the module name `eos.py` outright |
| `eos.alphabag.thermodynamics_quarks` (2 files) | gone — §5 forbids the sector suffix in a one-sector model package |
| `eos.alphabag.compute_tables` | gone |
| `eos.sfho.compute_tables` (3 files + a notebook) | gone — it is `eos.sfho.table` |
| `eos.alphabag.parameters.get_alphabag_custom` (4 files) | module ok, name gone |
| `eos.sfho.parameters.create_custom_parametrization` (2 files) | module ok, name moved to `eos.sfho.nmp` (§5 puts an NMP-inverting constructor there) |

Everything `nucleation` takes from `eos.general` still resolves — constants,
lepton thermodynamics, `figure_style`, and both `constraints` and the older
`observational_constraints` path. So the damage is entirely in the model and
astro packages, and every one of those breaks is a Phase 3/4 module MOVE, not a
Phase 5 rename: **`nucleation` cannot import `eos` today, before this map
touches anything.**

Two consequences for the brief:

- The Phase 6 pass is not "fix what Phase 5 broke" — it is a port across the
  refactor's module layout, and it must be scoped as one. `nucleation`'s own
  test suite cannot have been green since Phase 3.
- [Ticket 07](07-naming-sweep.md)'s finding that `nucleation` touches none of
  the 58 proposed renames still holds and is still what keeps Phase 5 cheap.
  But `nucleation/composition.py:45-51` imports four
  `compute_alphabag_*_thermo_from_mu` / `compute_cfl_*_thermo_from_mu` names,
  which break §13 rules 1 and 3 (`compute_` prefix, package name repeated) and
  are NOT among the 58. Whether Phase 6 renames them — and so whether
  `alphabag` gets the same treatment `vmit`, `dd2` and `sfho` got — is a
  decision this brief owes. `eos/alphabag/thermodynamics.py:345` already
  defines `thermo_from_mu`, so the package holds both spellings.

Resolved when the corrected Phase 6 brief is written out and the user has agreed
to it. **Creating or pushing a remote stays out of scope** regardless.

## Answer — the corrected Phase 6 brief

Agreed with the user across four grilling rounds. Fifteen decisions, every one
put and answered; the frontier is empty.

### What changed under the ticket while it sat open

Three of this ticket's own premises are false as written, and the brief corrects
them rather than inheriting them.

1. **The alphabag rename question has no subject.** The ticket asks whether
   Phase 6 renames `compute_alphabag_*_thermo_from_mu` / `compute_cfl_*`, noting
   "the package holds both spellings". It does not. Those spellings are **gone
   from `eos` entirely**; `alphabag` already carries §13's vocabulary
   (`thermo_from_mu`, `cfl_thermo_from_mu`, `solve_beta_eq_neutrinoless`,
   `solve_fixed_yc_ys`, `solve_cfl`, `T_critical`). There is no scope-widening
   choice to make — `nucleation` follows, full stop.
2. **The line-count row is not drift.** The ticket calls Phase 6 wrong for
   saying "11.6k lines across 40 files" against a measured "8.1k across 38".
   Measured now: **40 `.py` files and 11,620 lines in the tree; 8,121 lines in
   the package alone.** Both are right, for different denominators. The row
   comes out.
3. **The port is not "a port across the refactor's module layout".** Measured
   target by target, it is **one mechanical pass plus exactly one structural
   change**. Every broken target has an in-place successor of identical
   signature; only the total-thermo assembly genuinely changed shape.

So the corrected premises are **two**, not three: the remote exists
(`github.com/guerrinimirco/metastability-nucleation`), and work lands on
`paper-release` directly.

### What is measured, not assumed

**The breakage is exactly the seven this ticket recorded — and nothing more.**
Re-measured against `eos` at HEAD, after ~57 tickets landed including four
rename tickets: 5 modules gone, 2 names gone, no new breaks. Ticket 07's
finding that `nucleation` touches none of the 58 renames now holds
**empirically against a much-changed `eos`**, which is a stronger statement than
it could make when it was written.

    nucleation imports                          ->  today
    eos.tov.solver         compute_tov_sequence     eos.astro.tov.solver
                           generate_ec_logspace     eos.astro.tov.solver
                           truncate_to_stable_branch eos.astro.tov.solver
                           EOSTable_for_TOV         eos.general.state   <- different layer
    eos.sfho.compute_tables  all four names         eos.sfho.table      (all present)
    eos.sfho.parameters      create_custom_...      eos.sfho.nmp
    eos.alphabag.thermodynamics_quarks  T_critical  eos.alphabag.thermodynamics
                           compute_alphabag_thermo_from_mu  thermo_from_mu   (same signature)
                           compute_cfl_thermo_from_mu       cfl_thermo_from_mu (same signature)
    eos.alphabag.eos       solve_alphabag_beta_eq   solve_beta_eq_neutrinoless (compatible)
                           solve_alphabag_fixed_yc_ys solve_fixed_yc_ys        (compatible)
                           compute_*_total_thermo_from_mu   NO SUCCESSOR -- see below
    eos.alphabag.compute_tables  AlphaBagTableSettings  eos.alphabag.table.TableSettings
                           compute_alphabag_table         eos.alphabag.table.compute_table
    eos.alphabag.parameters  get_alphabag_custom      NO SUCCESSOR -- see below

`EOSTable_for_TOV` is the one that is **not** the same move as its neighbours:
it did not follow `tov` into `astro/`, it went to `general/`, because it is the
contract surface both layers may import (§1).

**Everything from `eos.general` already resolves** — `figure_style`,
`observational_constraints`, `physics_constants`, `thermodynamics_leptons`.

**The other direction of §1 is verified and gated.** `eos` imports `nucleation`
nowhere, and `test/test_imports.py:41 test_eos_never_imports_nucleation`
enforces it, with a docstring recording that `eos.dd2.notebook_api` once did.

**Two conformance items measure clean, and the brief commissions no sweep for
them.** `nucleation`'s internal layering is already acyclic and layered —
`barrier.py` at the bottom importing nothing internal, then
`composition`/`critical`/`rates`/`tables`, then `analysis/`, then
`analysis/figure/`. And the docstring standard finds **nothing** across 8,121
lines: no Phase/Stage/milestone reference, no TODO, no FIXME.

### The one structural change (decision 1)

`compute_alphabag_total_thermo_from_mu(mu_u, mu_d, mu_s, mu_e, T, params,
include_photons=, include_gluons=, include_thermal_neutrinos=, mu_nu=)` has no
successor: the boolean flag-bag became `SpeciesFlags` and the total assembly
moved behind `eos_point`.

**Ruled: `nucleation` keeps its own solver and assembles the total itself**, from
five pieces that are all already public and need **no new `eos` code**:

    eos.alphabag.thermo_from_mu / cfl_thermo_from_mu   quarks + bag
    eos.alphabag.gluon_thermo(T, alpha)                gluons
    eos.general.thermodynamics_leptons.electron_thermo electrons  <- already imported
    eos.general.thermodynamics_leptons.photon_thermo   photons
    eos.general.thermodynamics_leptons.neutrino_thermo thermal neutrinos

The three `include_*` booleans become three `if`s. Rejected: calling
`alphabag.eos_point`, which would hand `eos` the 4-vector saddlepoint solve that
is the reason `nucleation` exists as a package; and asking `eos` for a restored
total-assembly entry point, which is the phase-adapter contract rebuilt from the
wrong side.

`get_alphabag_custom(alpha=, B4=, m_s=)` has no successor either, but
`Parameters` carries exactly those three fields (`name, m_u, m_d, m_s, alpha,
B4, tc_coeff`). **Ruled: one small helper in `nucleation`** wrapping
`dataclasses.replace(Parameters.default(), ...)`, not seven inline copies and
not a new `eos` constructor — the set is *custom*, and §13 reserves `named()`
for published sets.

### The split, and the gates

**Phase 6 becomes two tickets.** The port and the conformance pass have
different gates, and every comparable ticket on this map got split for that
reason.

**[Ticket 24](24-phase6-execute.md) — the port.** Fix the seven targets across
the library and its tests; make the one structural change; add the parameter
helper. **Not the notebook.** Gate:

- **Take the before-image first.** `nucleation`'s suite cannot have been green
  since Phase 3 — it cannot import `eos` — so "0 added failures" has nothing to
  measure against. Run the suite at `33c1e61` against `eos` HEAD and record it
  verbatim as `nucleation_before_py314.txt`. It will be mostly collection
  errors. That is the point.
- **Then: every failure caused by an import or a call site goes to zero, and
  every failure that survives is reported, not fixed.** A survivor belongs to
  the conformance ticket or to one of its own.
- Re-run the §1 check in both directions and report both.
- **Do not move the tests in this ticket.** The gate is a node-id-to-node-id
  comparison; moving the files underneath it destroys the only instrument it
  has.

**[Ticket 72](72-phase6-conformance.md) — the conformance pass**, blocked by 24.
Exactly four items, since two of the original six measure clean and one has no
meaning here:

1. the paper notebook — port (mechanical: 4 import lines, 2 name swaps) **and
   execute**. Its own header says the runs take hours; that gate, not the edit,
   is why it is here.
2. the test move: `nucleation/nucleation/tests/` -> top-level `nucleation/test/`,
   **tracked, not gitignored**, with `make_fixture` invoked as
   `python test/make_fixture.py` and its three docstring references updated.
3. `README.md` to the standard of the new `eos` README, examples actually run.
4. dead code removed.

**"Apply the same API conventions" is dropped.** `nucleation` is a consumer, not
a model; §5's uniform API is a contract for models, and imposing
`eos_point`-shaped signatures on a nucleation-rate sampler would be conformance
theatre.

### Two rules that do NOT transfer from `eos`

- **`test/` is tracked, not gitignored.** `eos` hides its suite because it is
  private and unpublished. `nucleation` is on `paper-release` with a live remote
  and is headed public: gitignoring its 16 test files would publish the
  repository backing a paper with no runnable tests. Take the *move* for layout
  parity; refuse the *gitignore*. The parity that matters is where tests live,
  not whether they are published, and the two repos have opposite and
  well-founded answers to the publication question.
- **`nucleation`'s `output/` rule stands unchanged.** `.gitignore:32-38` already
  ignores everything under `output/` *except*
  `output/paper/{figures,figure_data,tables}` — **87 tracked files**, the
  paper's own figures and tables. That is §11's `output/public/` principle,
  already correctly specialised. Flattening it to `eos`'s rule would untrack the
  paper's figures. The brief says so explicitly so a later session does not
  "fix" it into conformance.

### Git hygiene, before ticket 24 opens

`nucleation`'s tree is dirty on top of `33c1e61`: **16 regenerated paper PDFs**
under `output/paper/figures/`, and **two deleted docs**.

- Commit the PDFs — a figure regeneration is a real artifact.
- **Restore `docs/nucleation_physics.md` (28.9 kB, the formalism) and
  `docs/reproducing.md` (8.5 kB).** They have four live references —
  `README.md:267,271` as links, `nucleation/tables/quantum.py:7` citing
  "section 7" of the first from a source docstring, and the paper notebook — and
  `docs/` is now empty. A source docstring citing a section of a deleted
  document is §13's "docstrings stand on their own" broken in the worst
  direction. If the deletion was deliberate (a rewrite), the conformance ticket
  carries "rewrite the two docs" as an explicit item with the four references
  updated; it does not leave a hole.

### Pushing

**The no-push rule is lifted** — the premise that produced it ("no remote
exists") is false. Push **after ticket 24**, and again after 72. A public repo
that imports is worth more than a polished one that does not, and today's remote
carries a `paper-release` branch that cannot import its own dependency. The
first push is also what makes the tests publicly runnable (§ tests tracked).

### Acceptance

**[Ticket 25](25-acceptance.md) waits on the port only** — `25 <- {21, 24}`, not
72. The Acceptance criteria block can check that `nucleation` imports `eos`,
that its suite runs, and that §1 holds both ways; nothing in it reads a README.
Gating the Stage 7 report on a rewrite no criterion measures would hold it
hostage. The map's Destination is amended to say so, and 72 is recorded as
in-scope-but-not-gating.

### Noted, not fixed — Stage 7 report material

`eos/alphabag/solver.py`'s `solve_beta_eq_neutrinoless` and `solve_fixed_yc_ys`
take `params=None` and a boolean flag-bag (`include_photons`, `include_gluons`,
`include_thermal_neutrinos`, `include_electrons`) rather than `SpeciesFlags`.
That is §5's "`par` comes first and is never optional" and §4's vocabulary, both
still open in `alphabag`. It is an `eos` matter, not a Phase 6 one — and it is
load-bearing for this brief in the good direction: it is *why* two of the seven
targets are signature-compatible name swaps.

Status: resolved.
