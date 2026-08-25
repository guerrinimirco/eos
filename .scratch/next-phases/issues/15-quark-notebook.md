# notebooks/quark_eos — skeleton, knobs and figures

Type: task
Status: resolved
Blocked by: 04, 05, 61
Parent: ../map.md

## Question

Stage 2. Same shape as `hadronic_eos` — same knobs cell, same jupytext pairing,
same figure rules — for `vmit`, `alphabag`, `njl`, `ccdm`, plus whatever ticket 05
ruled about `abpr`. **Say in the intro whether `abpr` belongs here and why**, per
that ruling.

Quark-model figures use the pure-quark planes that make sense: `P` vs `n_B`,
`eps` vs `P`, `c_s²` (named `cs2_isothermal` or `cs2_adiabatic`, never bare), and
flavour composition `Y_u`, `Y_d`, `Y_s`, `Y_e`.

**Where a bare quark model gives no stable star, say so** instead of plotting an
empty M–R panel.

Figures to `output/quark/`. Done when the notebook executes clean and every
figure file exists.

## Added by ticket 05

**`abpr` is in, as a companion panel — not a fifth peer in the knobs cell.**
One figure against `alphabag` at CFL and T = 0, showing the O(m_s^4) difference
`abpr/verify/run_full_check.py` already measures; the two are driven as a matched
pair through alpha_s = pi/2 (1 - a4). It is CFL-only and T = 0-only, so as a peer
it would trip ticket 04's unsupported-combination pattern on nearly every cell.

`abpr` has no `table.py` but **does** have `eos_table` (`abpr/api.py:146`) —
nothing in the model iterates, so §5's conditional-existence clause is satisfied
and there is no gap to work around.

## Answer

**Shipped: [notebooks/quark_eos.py](../../../notebooks/quark_eos.py) paired to
`.ipynb`.** `jupytext --to notebook --execute` (via `nbconvert`) runs it to
completion with **no traceback**, verified in an isolated `git archive HEAD`
copy at `1a8c55a` so the claim is about the committed tree rather than the
concurrent session editing `notebooks/hybrid_eos.*` beside it. **36 cells, 18
code cells, 0 error outputs**, ~2.5 minutes end to end. Interpreter:
**python.org 3.14.2, numpy 2.3.5, scipy 1.17.0, matplotlib 3.10.9**.

Six figure files in `output/quark/` (`.png` + `.pdf`):
`quark_P_nB_and_eps_P`, `quark_cs2_and_composition`, `abpr_vs_alphabag_cfl`.

Targeted tests, in the same isolated HEAD copy plus a snapshot of the
gitignored `test/`: `test/general test/test_imports.py test/vmit
test/alphabag test/njl test/ccdm test/abpr` = **809 collected, 809 passed, 0
failed**. No library file was touched by this ticket, so **0 added failures**.
The full suite was not run (concurrency).

### The spine, copied from `hadronic_eos` and not from the prototype

Knobs cell, three-way gap reporting (`run`/`header`), `standard_name` /
`table_path` from `eos/general/table_io.py`, flags built inside the section
rather than in the knobs cell, the path bootstrap, and §3's rule that
`leptons=` is named for the fixed-fraction modes and left unsaid for beta
equilibrium. Two quark-specific additions:

- `lepton_kwargs(mode)` is the §3 rule as one named function rather than an
  inline `if` repeated at three call sites. It also covers `cfl`, where the
  flag is meaningless for a second reason: the locked phase is neutral by
  construction with no electrons at all.
- `cfl` is in `KNOBS.modes`. Only `alphabag` has it; `vmit`, `njl` and `ccdm`
  raise `ValueError("unknown mode 'cfl'")` and the notebook reports all three
  as refusals. That is §3's "cfl is not available to every model" visible in
  executed output rather than asserted in prose.

### `abpr`: in, as ticket 05 ruled, and the intro says why

Section 7, a companion panel against `alphabag`'s CFL phase at T = 0, matched
through `alpha_s = pi/2 (1 - a4)` with the same `m_s`, `B4` and `Delta0`. The
intro states the ruling and its reason (CFL-only and T = 0-only, so as a fifth
peer it would refuse nearly every cell). Both sides go through `eos_table`;
their row schemas are identical, so no adapter was needed.

**One thing the panel deliberately does NOT plot.** `abpr/verify` measures the
difference against the analytic `O(m_s^4)` term to better than 1% — but at
equal *quark chemical potentials*. A table is density-driven, and the two
models reach a given `n_B` at different `mu`, so the difference picks up
`n dmu` on top of the expansion term: measured against the analytic curve at
`mu = mu_B/3` the ratio runs 0.49–0.55, not 1. Interpolating alphaBag onto
abpr's `mu_B` grid is worse (3.4–7.7), because alphaBag's CFL potentials are
unequal at equal densities while ABPR assumes one common `mu`. So the panel
shows the density-driven difference — **−4.15% to −0.60% across the grid, ABPR
softer everywhere** — and points at the verify suite for the closed statement.
Reproducing that statement in the notebook would need `cfl_thermo_from_mu`,
which is an internal.

### Where a bare quark model gives no stable star (section 8)

Answered with a TOV sequence rather than an empty M–R panel. The §8 gate (P
monotone, `0 <= cs2 <= 1`) runs **before** integration. Three distinct outcomes
on the shipped grid, and the section keeps them apart:

    [vmit]     M_max = 1.574 M_sun at R = 7.79 km (23 of 25 on the stable branch)
    [njl]      M_max = 1.232 M_sun at R = 7.98 km (22 of 25)
    [alphabag] still rising at the last density of the knobs grid (M = 1.428,
               R = 7.87): no maximum located — widen n_B, not a verdict
    [ccdm]     only 3 rows above P = 0 on this grid: no surface bracketed,
               no sequence integrated

Every mass is far under 2 M_sun, which is the point: a bare deconfined phase is
the quark *half* of a construction, and the M–R figure worth drawing belongs to
`hybrid_eos`.

### Ticket 68 confirmed, and NOT fixed here

Measured on the shipped tree:

    njl.eos_point   (par, mode, species=None, n_B, T, SnB, x0, patterns, **conditions)
    ccdm.eos_point  (par, mode, species=None, n_B, T, SnB, x0, branches, patterns, **conditions)

Neither takes `leptons` as a named argument; `vmit`, `alphabag` and `abpr` do.
`eos_response` is the same in both. So the notebook's `leptons=KNOBS.leptons`
reaches `njl` and `ccdm` **through the condition bag**, via the `k != "leptons"`
carve-out at `njl/api.py:68` and `ccdm/api.py:76`. It works today — which is
why the notebook is uniform across all four and needs no per-model translation
table — but it works by the mechanism §5 forbids. Reported, not fixed.

### Six more divergences the notebook had to cross or name

1. **`alphabag`'s `cfl` takes `Delta0` as a mode fraction.**
   `MODE_FRACTIONS["cfl"] == ("Delta0",)`, so the gap is a *condition* that may
   be swept as an axis; in `abpr` the same gap is a *parameter* on the
   parameter object. §3 says the locking leaves no free fraction to name. The
   knobs cell carries `Delta0` beside the fractions and a markdown cell states
   the divergence; section 7 crosses the two spellings, which is the only place
   in the notebook that has to.
2. **`cs2` is spelled two ways.** `njl` and `ccdm` return `cs2_isothermal` /
   `cs2_adiabatic` — named for the *thermal* variable, which is the axis §5
   names. `vmit` and `alphabag` return **`cs2_eq`** — named for the
   *composition* axis, leaving the thermal variable unsaid although the
   derivative is taken at fixed T. `abpr` returns `cs2_isothermal`. The figure
   reads whichever key is present, is drawn at T = 0 where the two coincide,
   and is labelled for what was computed. This is a §5 naming gap in two models,
   not a physics gap.
3. **`rows_from_result` is not exported at package level in `njl`, `ccdm` or
   `abpr`** (it lives in their `table.py`; `abpr` has `cfl_row` in `api.py`),
   where `vmit` and `alphabag` export it. The notebook sidesteps it entirely by
   using `eos_table(..., rows=True)`, which all five accept and which is the
   better spelling anyway.
4. **`RESPONSE_FREEZES` is not exported at package level in `njl` or `ccdm`**,
   where the other three export it.
5. **`leptons` defaults differ**: `True` in `vmit`, `njl`, `ccdm`; **`False` in
   `alphabag`**. Harmless here because the notebook names the flag wherever it
   means anything, but a caller who does not would get a different physical
   system from `alphabag` than from its three siblings.
6. **Sectors the six-flag knob cannot reach.** `alphabag` ships `gluons=True`
   and `njl`/`ccdm` ship `csc=False`; §4's six names reach none of them, so they
   run at their own defaults throughout. A second printed table in section 2
   lists each model's own flags and the value they are left at, rather than
   leaving a reader to discover it from a number that does not add up. The
   pairing sector is [ticket 16](16-quark-stepwise.md)'s subject.

### One defect inherited from the shipped spine, reported not fixed

`table_path(model, name, root="output/tables")` takes a **relative** root, so a
kernel started in `notebooks/` writes its tables to
`notebooks/output/tables/<model>/` rather than to the repository's `output/`.
`hadronic_eos` has the same behaviour and it is why `notebooks/output/` exists
in the tree. The figures are unaffected — `FIG_DIR` is built from the bootstrap
`ROOT` and is absolute. Left alone deliberately: the fix belongs either in
`table_io.py` or in ticket 12's file, uniformly for all four notebooks, and
this ticket copies the spine rather than forking it.

Status: resolved.
