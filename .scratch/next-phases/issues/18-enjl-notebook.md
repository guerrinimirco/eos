# notebooks/enjl — skeleton, knobs, figures and the author-table reproduction

Type: task
Status: resolved
Assignee: session 2ee58c30
Blocked by: 04, 61
Parent: ../map.md

## Question

Stage 3. Same shape again, for `enjl`, **including its branch pair** — §5 records
the ENJL branch pair as two branches of one functional, and the notebook must
show both.

`docs/enjl/` and `test/enjl/reference/` hold the author tables. **Show the
notebook reproducing at least one of them, with the residual printed.** Those
tables are §12 golden references: code that disagrees with them is wrong.

Finite-`T` ENJL is an open item. Check `docs/DEFERRED.md` and **let the notebook
report the gap rather than work around it** — the §3 raise is caught at the top
of the section, its message printed, and the notebook continues.

Figures to `output/enjl/`. Done when the notebook executes clean, every figure
file exists, and the reproduced author table's residual is printed.

## Added by ticket 05

**The ENJL branch pair belongs to THIS notebook, not `hybrid_eos`.**
`enjl_branch_pair` lives in `eos/mixed/adapters.py` and §5 lists it among the
shipped adapters, so this notebook and [ticket 58](58-hybrid-skeleton.md) overlap
on exactly that object. The two branches are two branches of one functional, not
two models being coupled — the physics is ENJL's and `eos/mixed` is the machinery
it is expressed through. State the boundary in one line here; ticket 58 states it
there.

## Answer

**Shipped: [notebooks/enjl_eos.py](../../../notebooks/enjl_eos.py) paired to
`.ipynb` (`2e68b46`, `f692028`).** The ticket writes the name `notebooks/enjl`;
its three siblings are `hadronic_eos`, `quark_eos` and `hybrid_eos`, so it is
**`enjl_eos`** and the four now spell the same thing.

Verified in an isolated `git archive HEAD` copy (HEAD = `77c2976`), with the
kernel started in `notebooks/` so the bootstrap and the output roots are
measured where a reader would actually run them: `jupytext --to notebook
--execute` gives **44 cells, 22 code cells, 0 error outputs**, ~70 s. The live
tree was not used for the claim — three other sessions were committing into it
during this one, one of them mid-verification (below). Interpreter:
**python.org 3.14.2**, numpy 2.3.5, scipy 1.17.0, matplotlib 3.10.9,
h5py 3.16.0, jupytext 1.19.4. Targeted tests in the same isolated copy:
`test/enjl` + `test/test_imports.py` = **317 collected, 317 passed**. The full
suite was not run (concurrency), and no library code was touched by this ticket.

`test/enjl/reference/*.dat` was copied into the archive copy by hand: `test/`
is gitignored, so `git archive` does not carry it, and the ticket's golden
references are data rather than another session's code.

### The spine, and the three things this model moved

Copied from the shipped `hadronic_eos.py`, not from the prototype: the four-line
path bootstrap, `Knobs` with `conditions(mode)` dropping what the mode does not
take, species flags built INSIDE the section, `run()`'s three-way
`ok`/`unsupported`/`unconverged` with `TypeError` left to escape. Three
adaptations, each forced by the model rather than chosen:

1. **The axis swept is the PARAMETER SET, not the model.** There is one model
   here and six published `(f_q, B)` sets, so `Knobs.sets` replaces
   `Knobs.models` and `parameters_for` takes a set name. Everything else in the
   spine is unchanged, including the printed-header shape.
2. **`axes(mode, thermal_value)` takes the value.** `eos.enjl.eos_table` refuses
   a thermal axis or a fraction axis with more than one entry — a table here is
   a density continuation and a second value restarts it, which the refusal says
   in those words. So the method takes one value and the cells loop. The
   notebook asks for the two-temperature axis anyway, in section 7, and prints
   the refusal.
3. **`direction` is a knob.** It is the branch pair (below) and it changes every
   number in a table, so it is in the automatic file name too.

`leptons=` is named for the fixed-fraction modes and left unsaid for beta
equilibrium, under §3, exactly as ticket 12 settled. That rule is not cosmetic
here: `eos.enjl` **raises** on `leptons=False` in beta equilibrium — *"has no
meaning in beta equilibrium, which is defined by the leptons"* — and the
notebook prints that refusal rather than avoiding the call.

### The golden references, with the residual printed

Section 6 solves our `beta_eq_neutrinoless` continuation at the author's OWN
densities (nothing interpolated on either side) and prints max, median and the
density of the worst row, per column, for **all five** reference sets:

| set | P max | P median | eps max | mu_B max |
|---|---|---|---|---|
| `fq0.5_B1` | 9.94e-06 | 1.66e-06 | 1.16e-07 | 1.46e-07 |
| `fq0.7_B0` | 9.94e-06 | 6.92e-07 | 1.17e-07 | 1.42e-07 |
| `fq0.7_B1` | 2.68e-04 | 1.42e-06 | 1.46e-05 | 3.20e-05 |
| `fq1.0_B0` | 5.11e-04 | 1.29e-06 | 2.56e-07 | 5.89e-05 |
| `fq1.0_B1` | 9.94e-06 | 1.07e-06 | 1.43e-07 | 1.78e-07 |

Two things measured rather than assumed:

* **The 9.94e-06 ceiling on `P` is at `n_B` = 0.10 in every set that has one,**
  the LOWEST density of the window, where `P` is 0.46 MeV/fm^3 and the residual
  is an absolute agreement of ~5e-6 MeV/fm^3 divided by it. It falls
  monotonically to ~5e-07 by `n_B` = 0.3, which is why the median is printed
  beside the max and why the figure draws the whole curve.
* **The two sets above 1e-04 are above it on their last one or two rows only,**
  and those rows are the approach to that set's own coexistence endpoint —
  `fq1.0_B0` is 5.11e-04 at 0.410 and 7.00e-05 at 0.400 against 9.94e-06 at
  every density below; `fq0.7_B1` is 2.68e-04 at 0.420 and 9.94e-06 below 0.40.
  There the author's table is already following a construction and ours is
  following a branch, so the comparison stops being a measurement of the
  implementation. Both numbers are printed — full window and below 0.40 — so
  the reader sees the effect rather than a window tuned to hide it.

The notebook **parses the `.dat` itself in eight lines and imports nothing out
of `test/`**, which `docs/DEFERRED.md` explicitly requires of the replacement
enjl notebook (`plot/enjl_paper_figures.py` reaches into `test/enjl` and fails
at import from a fresh clone). Absent tables are reported as a message and the
notebook continues; the four column traps the loader documents are handled and
named — `E` is already vacuum-subtracted, `munr` and not `mun` is mu_B, a blank
`munr` marks the 203 interpolated plateau rows of `fq0.5_B1`, and the off-grid
densities are the author's coexistence endpoints.

### The branch pair

Shown through the model's own surface, `direction="up"` / `"down"` — two
self-consistent states of ONE thermodynamic potential at the same potentials,
which is what §5 means by two branches of one functional. Measured on the three
swept sets: 24-27 densities carry both continuations, of which only **5-6 are
two DISTINCT states**; above the transition the two converge on one root.

That mattered. The first draft named a winner from the sign of `eps_up -
eps_down` at every overlapping density and printed `delta = +0.0000 stable:
down` / `-0.0000 stable: up` alternating on round-off — reporting 1e-12 as
physics. A `SAME_BRANCH = 1e-8` tolerance now separates the two cases and only
the genuinely distinct densities are listed, with the same filter applied to the
`branch` label of the constructed table (which flips on the same noise). For
`fq1.0_B1`: up wins at 0.450-0.650, down at 0.700, and
`build_constructed_table` with an empty window list picks exactly that.

**The `enjl_branch_pair` adapter of `eos/mixed/adapters.py` belongs to
[ticket 58](58-hybrid-skeleton.md)'s `hybrid_eos`, not here** — stated in one
line in section 5. This notebook never imports `eos.mixed`: locating a
coexistence needs both branches at once and is a composite engine's job, so
`build_constructed_table` takes the located windows as an ARGUMENT and the
notebook passes `[]`, which still returns the stable branch.

### The finite-T gap, reported and not worked around

`docs/DEFERRED.md`'s enjl section says finite temperature IS implemented and
what is NOT is the CONSTRUCTION above T = 0. Section 7 shows all three faces:

* the warm branch at T = 20 MeV with `photons` and `thermal_neutrinos` on
  **works** — 31/31 rows on all three sets, `S/B` from 2.33 down to 0.55;
* `build_constructed_table` at T = 20 raises, and the notebook prints its
  message in full (Gibbs free energies rather than P and mu_B alone, the
  entropy in the lever rule);
* `eos_response` raises for this model at any T, and that message is printed
  too.

`SpeciesFlags` is a refusal site four ways here — `hyperons`, `deltas`,
`muons` and `thermal_mesons` are FIXED by the model and moving any of them
raises — and the section-2 probe prints all four reasons and identifies
`photons` and `thermal_neutrinos` as the caller's. This is the strongest case in
the repository for ticket 12's finding that flags cannot be built in a knobs
cell.

### The `table_path` root, confirmed by counterexample

`root=str(ROOT / "output" / "tables")` is passed, using the bootstrap's `ROOT`.
The bug is real and was measured in the isolated copy with the kernel in
`notebooks/`: `table_path("enjl", "x.h5")` on its default returns
`<root>/notebooks/output/tables/enjl/x.h5`. With the argument, the executed run
wrote `output/tables/enjl/…h5` and `output/enjl/*.{png,pdf}` at the repository
root and left `notebooks/output/` non-existent. Figures are anchored the same
way, `FIG_DIR = ROOT / "output" / "enjl"`.

Three figures, all styling from `eos.general.figure_style`:
`enjl_branch_pair`, `enjl_masses_and_cs2`, `enjl_composition_and_residual`
(png + pdf each). The sound speed is a finite difference `dP/d(eps)` down one
named branch and is labelled **`cs2_adiabatic`**, never a bare `cs2`, with the
panel text saying that at T = 0 along a beta-equilibrium line it is the same
derivative the isothermal one would be — `eos_response` cannot be asked, and
its refusal is printed two cells earlier.

### One collision with a concurrent session, resolved in our favour

Both log axes call **`figure_style.log_decades`**, which did not exist when the
first verification ran and produced an `AttributeError` in the archive copy. It
was not a stale checkout: another session committed `dfe9695`
*"fix(figure_style): log-axis exponent minus renders, via log_decades"*
**between** the `git archive` and the failure. So the fix was to re-archive, not
to write the two-line `set_yticks`/`set_yticklabels` workaround
`quark_eos.py:690` still carries — that notebook can now collapse those two
lines into one call, which is a line for whoever next opens it. The hazard is
worth recording: **`git archive HEAD` is a moving target when other sessions are
committing**, so the archived SHA has to be recorded with the result, and this
one is `77c2976`.

Status: resolved.
