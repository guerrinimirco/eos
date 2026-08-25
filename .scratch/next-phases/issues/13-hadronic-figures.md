# notebooks/hadronic_eos — the six figure families and the TOV pass

Type: task
Status: resolved
Assignee: session 638a3170
Blocked by: 12
Parent: ../map.md

## Question

Stage 1, figures. All styling from `eos/general/figure_style.py` and nothing else
(§10); overlays from `eos/general/constraints/` via
`overlay(ax, plane, ...)` (`eos/general/constraints/__init__.py:403`). **Every
panel selectable for with/without hyperons and with/without deltas.**

1. `P` vs `n_B` in beta equilibrium, all models overlaid
2. `P` vs `n_B` at `Y_C = 0.5`, `Y_S = 0`, `leptons=False` — symmetric nuclear
   matter — against the heavy-ion constraints (FOPI, Danielewicz)
3. mass–radius against the M–R constraints
4. mass–tidal-deformability against the M–Λ constraints
5. speed of sound squared vs `n_B` — the label says **which one it is**
   (`cs2_isothermal` vs `cs2_adiabatic`, §5), never a bare `c_s^2`
6. composition: particle fractions `Y_i` vs `n_B`, one panel per model, species
   colours from `figure_style.particle_style`

Structure work goes through `eos.astro.tov`. The P-monotonicity and
`0 ≤ c_s² ≤ 1` check runs **before** integration and reports status rather than a
meaningless mass (§8).

Figures written to `output/hadronic/`. Done when every figure file exists and the
notebook executes clean.

## Answer

**Section 7 of [notebooks/hadronic_eos.py](../../../notebooks/hadronic_eos.py),
commit `77c2976`, plus one library fix in
[eos/general/figure_style.py](../../../eos/general/figure_style.py) (`dfe9695`).**

`jupytext --to notebook --execute` runs the whole notebook — sections 1 to 7 —
with **no traceback**, verified on the live tree and again in an isolated
`git archive HEAD` copy, since three other sessions hold `eos/*/api.py` and the
other notebooks. **56 cells, 30 code cells, 0 error outputs**, ~45 s.
Interpreter **python.org 3.14.2**, numpy 2.3.5, scipy 1.17.0, **matplotlib
3.10.9**. Targeted tests in the same isolated copy: `test/general` +
`test/test_imports.py` = **327 collected, 327 passed, 0 failed** — unchanged from
ticket 12's post-54 count. `python3 eos/general/figure_style.py` self-check ok.
The full suite was not run.

### The panels ARE the sector selection

`FIG_SECTORS` is a tuple of (name, flag overrides); a figure gets one panel per
entry, so "with and without hyperons" is one file with two panels rather than
two files to line up by eye. All four combinations were exercised while building
it and all four work; the notebook ships with `nucleonic` and `hyperons`
selected and the other two one uncomment away.

The sector is not free of the parametrisation, which section 5 had already
established: `SECTOR_SETS` carries `SFHoY_Fortin` / `SFHo_2fam` / `DD2Y` where a
sector needs them, `did` needs none, and `zl` refuses both sectors at flag
construction and is reported once per figure rather than once per point.

### Six families, and the numbers behind them

Grid `n_B = 0.05 … 1.2` fm^-3, 60 points, `T = 0`.

| model | sector | beta-eq rows | gate | M_max [M_sun] | R(M_max) [km] | max c_s^2 (gate) |
|---|---|---|---|---|---|---|
| zl | nucleonic | 60/60 | PASS | 2.283 | 11.39 | 0.866 |
| sfho | nucleonic | 60/60 | PASS | 2.059 | 10.30 | 0.806 |
| dd2 | nucleonic | 60/60 | PASS | 2.424 | 11.97 | 0.806 |
| did | nucleonic | 60/60 | PASS | 2.244 | 10.99 | 0.770 |
| sfho | hyperons | 60/60 | PASS | 1.991 | 10.35 | 0.724 |
| dd2 | hyperons | 60/60 | PASS | 2.035 | 11.39 | 0.578 |
| did | hyperons | 60/60 | PASS | 2.196 | 10.90 | 0.709 |

Twelve files in `output/hadronic/` (six families, `.png` + `.pdf` through
`figure_style.save_figure`): `pressure_beta_eq`, `pressure_snm`, `mass_radius`,
`mass_lambda`, `cs2_isothermal`, `composition`.

**The gate runs before integration and returns a status.** `deliverable(core)`
differences the delivered table itself — the quantity the solver will
interpolate — and reports `P` falling or `c_s^2` outside [0, 1] with the density
where it first happens. All seven tables PASS here, so no branch had to be held
back; the HOLD path prints the reason and simply does not integrate, and nothing
is repaired. A first-order transition violating it would be real physics and is
reported as such, not smoothed.

**Structure goes through `eos.astro.tov`** with the BPS crust at
`n_B = 0.08` fm^-3. `truncate_to_stable_branch` is NOT used: it re-orders to six
columns and **drops `k2` and `Lambda`**, which family 4 needs, so the notebook
takes `find_mmax_precise`'s index and slices — one line, and it keeps the full
eight-column layout.

### The sound-speed label, and a naming divergence worth a ticket

The panel is labelled `c_{s,isothermal}^2`, never a bare `c_s^2`. The curves are
at `T = 0`, where the isothermal and adiabatic speeds coincide, and the markdown
says so — the label names what was computed, which is the §5 rule.

**Real and unrecorded:** the four models do not spell the key alike.
`sfho` returns `cs2_isothermal`, `did` returns `cs2_isothermal` and
`cs2_adiabatic`, and **`zl` and `dd2` return `cs2_eq`** — a name for the *freeze*
(nothing held) rather than for the thermal variable §5 requires be named. At
`T = 0` it is the same number; at `T > 0` `cs2_eq` is exactly the bare name §5
forbids, since which of the two it is depends on the arguments. The notebook
takes whichever key the model returns and prints which it took. Not fixed here —
it is a public return-key rename across two models.

### The library fix this ticket forced

Every log axis in the repository's house style rendered its negative decades as
hollow boxes. `set_paper_style` and `set_global_style` already force an ASCII
minus, but matplotlib emits its own log tick labels as mathtext
(`$\mathdefault{10^{-4}}$`), mathtext turns a hyphen into U+2212, and it resolves
`\mathdefault` through the TEXT font — CMU Serif, which has no U+2212 glyph.
`mathtext.fallback`, `mathtext.default`, `mathtext.fontset` and a `font.family`
fallback list were all tried and none reaches that path. Labelling the decades as
ordinary mathtext does render, which is why `fm$^{-3}$` in an axis name was
always fine. So `figure_style.log_decades(ax, axis='y')` was added beside the
existing protection, with an assertion in the module self-check. §10 forbids
setting rcParams in a notebook, and this is the single home for styling, so it
was the only lawful place to put it. It is a pre-existing defect, not one this
ticket introduced.

### Three things the executed output makes visible

- **Symmetric matter is the same curve with hyperons on and off.** At
  `Y_C = 0.5, Y_S = 0` the strangeness condition forbids a net hyperon
  population, so `sfho`, `dd2` and `did` return byte-identical pressures in both
  panels. The panels are not redundant — Deltas carry no strangeness and would
  differ — but the physics is worth reading off the figure.
- **`zl` refuses `fixed_YC_YS`, and the refusal is the reason it does not need
  it**: the functional is written in `n_p` and `n_n` alone, so `n_S = 0`
  identically and `fixed_YC` at `Y_C = 0.5` IS symmetric matter there. The
  notebook carries one mode name per model with that reason stated, rather than
  dropping the only model whose refusal is informative.
- **`did`'s non-monotonic `c_s^2`** — the paper's signature feature, a peak near
  0.71 at `n_B ≈ 0.66` fm^-3 — comes out of the figure at exactly the value and
  place `eos/did/verify` pins it to, without the figure knowing about the check.

Two smaller ones: `zl`'s `eos_response` fails to converge at 1 of 25 densities
(the curve is 24 points, and the count line is the report); and `dd2` with
`deltas=True` drops 3 of 60 densities to scalar collapse, which is
`stop_at_boundary` working, seen while exercising the sector but not in the
shipped selection.

### The path bug, closed for this notebook

`table_io.table_path`'s `root` is relative, so under `jupytext --execute` from
`notebooks/` output lands in `notebooks/output/`. All three write sites in this
notebook — the section 3 example, the section 4 table, the section 6 benchmark —
now pass `root=str(ROOT / "output" / "tables")`, and `FIG_DIR` is
`ROOT / "output" / "hadronic"`. Verified: the isolated run wrote twelve figures
and both tables under the archive root and **created no `notebooks/output/`**.
The `notebooks/output/tables/dd2vmit/` in the live tree is the hybrid notebook's,
another session's, and still has the bug.

Status: resolved.
