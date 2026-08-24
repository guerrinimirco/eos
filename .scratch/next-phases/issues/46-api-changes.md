# The five items on the rename list that are not renames

Type: grilling
Status: open
Blocked by: 10
Parent: ../map.md

## Question

Ticket 10 approved the renames and split these out: each changes a signature or
deletes behaviour, so a naming gate does not authorise them. Each needs its own
ruling before any code moves.

1. **Delete `get_vmit_custom()`** — 6 `eos/` + 12 `test/` + 13 notebook sites.
   What replaces those callers? If the answer is `Parameters(...)` directly,
   say so and the deletion is mechanical; if it carries defaults nobody has
   written down, deleting it loses them.

2. **`solve_isentropic_beta_eq` / `solve_isentropic_trapped` fold into `SnB=`**
   on the mode solvers. CLAUDE.md §3 already blesses the shape ("wherever a
   temperature axis is accepted, entropy per baryon `SnB` is accepted in its
   place (an outer 1-D solve for T)"), so the question is not whether but
   whether sfho's outer solve is the shared one or its own.

3. **`find_mixed_window` merges into `locate_window`** — a merge, not a rename.
   Confirm they are one job; §5 makes the located window part of the mixed
   result, so the merged name has to serve both callers.

4. **`get_sfho_general(...)` and `create_custom_parametrization(...)` become
   `from_*` constructors.** §5 is explicit that an NMP-inverting constructor is
   a FREE FUNCTION in `nmp.py`, not a classmethod on the parameter dataclass —
   "putting it there forces a deferred import, which is the cycle announcing
   itself". So the ruling is which of the two is an NMP inversion (that one goes
   to `nmp.py`) and which is a plain alternate constructor.

5. **`build_mixed_eos_table` needs a name distinguishing it from `build_table`
   by job** — nobody has proposed one. §5 says a mixed table is "rows +
   windows", so the distinction to name is probably which of those two the
   caller gets.

Resolved when each of the five is ruled and the approved ones applied, with the
added-failure count reported.
