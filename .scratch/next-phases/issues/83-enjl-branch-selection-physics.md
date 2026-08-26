# How a raw ENJL continuation should choose its branch across a transition

Type: grilling
Status: open
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
