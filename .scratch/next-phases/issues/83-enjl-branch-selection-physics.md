# How a raw ENJL continuation should choose its branch across a transition

Type: grilling
Status: resolved
Blocked by: -
Parent: ../map.md

## Question

Split out of [ticket 72](72-enjl-branch-selection.md), which opened as this
question and turned out to be a different, smaller defect. **Non-gating**: the
suite is green without it, and nothing in the Acceptance criteria block of
`docs/REFACTOR_PROMPTS.md` measures it.

Ticket 72 removed the mechanism by which round-off chose a chiral branch. It
did not answer what the model should do where the branches genuinely overlap,
and that question survives intact:

1. **Where does a first-order chiral crossing sit, and does anything locate
   it?** For `fixed_YC_YS` at Y_C = 0.5, Y_S = 0, leptonless, the broken branch
   has lower eps to n_B = 0.4000 (380.632 against 382.793) and the restored one
   from 0.4333 (414.178 against 416.046), so the crossing is near
   0.41 fm^-3 — inside a window where the raw continuation follows whichever
   branch it started on, to 0.5000. CLAUDE.md §8 permits exactly that of a raw
   branch, and says a construction resolves it before a table reaches TOV.
   Whether one exists for THIS mode is the question.

2. **`eos.mixed.construction.enjl_coexistences` does not cover it.** It solves
   `mu_C` for neutrality with muons, so it locates transitions of the
   beta-equilibrium branch pair only. A leptonless held-(Y_C, Y_S) phase — the
   thing §3 says a mixed-phase construction consumes — has no located window
   and therefore reaches `build_constructed_table` with an empty
   `coexistences` list.

3. **Min-eps over the two sweeps is not a construction, and does not by itself
   make the delivered table robust.** Measured under ticket 72: before its fix,
   `build_constructed_table` delivered eps = 295.173 at n_B = 0.3000 on
   python.org 3.14 and 279.821 on anaconda 3.9 — because the up sweep is the
   thing that flipped, so the pair of sweeps held one branch and a fragment,
   not two branches. The min-eps rule is only as good as the two continuations
   fed to it. It is also, strictly, the "no mixing allowed" answer: between the
   spinodals the equilibrium state is a MIXTURE by the lever rule, not either
   pure branch.

4. **Does `build_constructed_table`'s stability rule hold outside beta
   equilibrium?** Its docstring argues min-eps from "at T = 0 and fixed n_B, in
   beta equilibrium with neutrality, the stable state is the one that minimizes
   eps". At fixed (n_B, Y_C, Y_S) and T = 0 the same conclusion follows —
   F = eps and the two roots carry identical conserved charges — but the
   document states the narrower premise, and §11 says a document states what
   the code does.

### Not in scope

The solver defect. That was ticket 72: `mu_S` carried as an unknown its rows
did not determine, the residual left three decades high, and `solver.solve`
answering a missed gate with a root on the other branch. Fixed, regenerated and
green; `BetaPoint.seed` now names which starting point produced a point, which
is the instrument this ticket will want.

## Resolution

**A raw ENJL continuation does not choose its branch — it maps one.** After
[ticket 72](72-enjl-branch-selection.md) the up and down sweeps are two
correct, complementary branches over the whole grid, and adding a per-point
energy comparison inside `solver.py` or `table.py` would destroy that: each
sweep would follow whichever branch is momentarily lower, neither would be a
branch any more, and the common tangent below would become uncomputable. The
choice belongs to the assembler, and the assembler's rule was wrong.

### min-eps is not a construction, and the failure is structural

Outside a coexistence window, keeping the lower-eps row selects the stable
PURE phase — correctly, and in every closure: at T = 0 the free energy IS eps
and the two roots carry identical conserved charges, so item 4's premise holds
at a held (Y_C, Y_S) exactly as it holds in beta equilibrium. What it does not
buy is a deliverable table.

The minimum of two CONVEX `eps(n_B)` curves is CONCAVE at their crossing. If
the branches cross at n_x, then `(eps_hi - eps_lo)' > 0` there, so the kept
slope `mu_B = deps/dn_B` jumps DOWN across it, and `P = mu_B n_B - eps` falls
with it. min-eps cannot deliver a monotone table across a crossing for any
parameter set, in any mode. The plateau is not an improvement on min-eps; it
is what removes a defect min-eps manufactures.

### Measured, python.org 3.14.2 / numpy 2.3.5

`fixed_YC_YS`, Y_C = 0.5, Y_S = 0, `leptons=False`, T = 0, `Parameters.default()`.
The two sweeps now hold two branches over the whole grid — up follows broken
from 0.24 to 0.52, down follows restored back down to 0.24 at negative
pressure. At fixed composition Euler gives the Gibbs free energy per baryon
`g = (eps + P)/n_B = mu_B + Y_C mu_C + Y_S mu_S`, checked against the solved
rows to 8.8e-12, and coexistence is equal P and equal g:

| | n_B [fm^-3] | P [MeV/fm^3] | g [MeV] | eps [MeV/fm^3] |
|---|---|---|---|---|
| broken   | 0.34945 | 22.9282 | 1006.4074 | 328.7573 |
| restored | 0.47500 | 22.9282 | 1006.4074 | 455.1121 |

Width 0.12555 fm^-3. The eps crossing — what min-eps picks — is at
**0.41774 fm^-3, inside the window**, as it must be. Delivered by min-eps
instead: `min dP = -37.52 MeV/fm^3`, `c_s^2 in [-0.402, 0.657]`, and
`P = -1.601 MeV/fm^3` at n_B = 0.4267 with `mu_B` falling 1041.3 -> 952.0
across that step.

### The defect is not confined to the heavy-ion mode

`build_constructed_table(TableSpec(nB=linspace(0.10, 1.60, 31), par=par), [])`
— `beta_eq_neutrinoless`, T = 0, leptons on, the notebook's own call:

| set | rows | min dP [MeV/fm^3] | c_s^2 | P drops at |
|---|---|---|---|---|
| fq0.5_B1 | 31 | **-34.597** | [-0.227, 0.900] | n_B = 0.40 |
| fq0.7_B1 | 31 | **-24.534** | [-0.007, 0.912] | n_B = 0.50 |
| fq1.0_B1 | 31 | +1.187 | [+0.025, 0.914] | — |

Two of the three published sets, in the mode that reaches TOV. So this binds
CLAUDE.md §8 directly, not by analogy to the P vs n_B constraint plane the
leptonless mode is compared on.

**Why nothing caught it.** `notebooks/enjl_eos.py` ran `KNOBS.sets[-1]` —
`fq1.0_B1`, the one clean set — under markdown asserting that an empty window
list "still returns the stable EoS". `check_delivered_table` ran `fq0.7_B1`,
the set that fails, but WITH its located windows, so the plateau covered the
hole. Neither had ever run the combination that breaks. `ConstructedTable.cs2`
would have shown it and has no reader.

`eos.general.state.EOSTable_for_TOV` disclaims the check in so many words —
"the check belongs where the table is DELIVERED to a structure solver, not
where it is built" — and `eos.astro.tov` does not perform it, so this model's
`verify/` is the only §8 enforcement anywhere on the path. §8 agrees: the gate
belongs to whoever builds the table.

### The change

- `eos/enjl/table.py`: `DELIVERY_TOL`, and `ConstructedTable.defect` /
  `.deliverable` — both §8 conditions at the tolerances `check_delivered_table`
  already used, as a property of the table rather than arithmetic in the check.
  A returned status, not a raise: a P-drop is a physics outcome of the
  parameter set, which is the case §6's "non-convergence is a return value"
  protects, and the empty-list path is CORRECT for `fq1.0_B1` — removing it
  would delete a working object to fix a broken one.
- `eos/enjl/verify/run_full_check.py`: `check_delivered_table` READS the
  predicate rather than recomputing it (a green check and a False flag must not
  be able to coexist) and grows to three cases, demonstrated in both directions
  as [ticket 72](72-enjl-branch-selection.md)'s countermeasure was:

      fq0.7_B1 + windows  deliverable=True   min dP=+0.00e+00   c_s^2 in [+1.014e-01, 0.8806]
      fq0.7_B1 + none     deliverable=False  min dP=-2.45e+01   P falls by 24.534 between 0.4500 and 0.5000
      fq1.0_B1 + none     deliverable=True   min dP=+5.27e+00   c_s^2 in [+1.065e-01, 0.8850]

  The third is the negative control: without it, "the gate fires" and "the gate
  fires whenever no window was passed" are the same observation. That is the
  mistake ticket 62 made with `enjl` and ticket 72's postmortem named.
- Four documents corrected — `build_constructed_table`'s premise and its
  `coexistences` parameter, `eos.enjl.api.eos_table`'s "an empty list is legal
  and gives the raw stable branch", `table.py`'s module header (which stated
  the Maxwell condition as equal P and mu_B, true only where g = mu_B), and
  `notebooks/enjl_eos.py` §7's markdown, which asserted the general form of the
  falsehood in prose while the cell below ran the one set for which it is true.
  The notebook cell now prints `deliverable`.
- `rows=True` documented as dropping the status, as it already drops `windows`.

### What this does NOT build, and why

The fixed-composition locator: [ticket 88](88-fixed-composition-coexistence.md).
83 asks how the branch should be chosen and the answer is "by a construction,
and here is the one this mode needs" — not "and here it is, built". The map's
hard rule points the same way.

One finding for it, taken here: **`locate_maxwell` will not take a second
closure.** It bisects `gap(mu_B)`, one variable, and that works because beta
equilibrium with neutrality determines mu_C from mu_B, which makes `g = mu_B`
and leaves equal-P as the only remaining condition. At a held (Y_C, Y_S) the
collapse is gone and coexistence needs equal P AND equal g — two conditions in
two unknowns, the two branches' own mu_B, which differ across the plateau
edges. `Coexistence.mu_B` is a single field documented as "equal across the two
phases by construction", and `plateau_row` writes it onto every plateau row, so
the carrier encodes the beta closure in its field list and not only in its
prose.

### Files

- `eos/enjl/table.py`, `eos/enjl/api.py`, `eos/enjl/verify/run_full_check.py`
- `notebooks/enjl_eos.py`, `notebooks/enjl_eos.ipynb`
