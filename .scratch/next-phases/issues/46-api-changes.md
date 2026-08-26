# The five items on the rename list that are not renames

Type: task
Status: open
Assignee: session eff950ed
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

## Ruling

All five ruled, four of them **settled by measurement rather than preference**.

1. **Delete `get_vmit_custom`.** `eos/vmit/parameters.py:71` defaults are
   `m_u=5.0, m_d=7.0, m_s=150.0, a=0.2, B4=180.0` — and `Parameters` carries
   **identical** defaults. So `get_vmit_custom(B4=170.0, a=0.15)` IS
   `Parameters(B4=170.0, a=0.15)`: a pure alias carrying no undocumented
   physics. Mechanical, 31 sites. The replacement sentence in vmit's document
   is owed by [ticket 79](79-parametrization-surface.md).
2. **Fold the isentropic solvers into `SnB=`.** `eos/general/tabulate.py:78
   temperature_at_entropy(...)` ALREADY IS the shared outer 1-D solve, with
   `TEMPERATURE_AXES = ("T", "SnB")` declared beside it; sfho's
   `solve_isentropic_beta_eq/_trapped` (`solver.py:735,761`) are a private
   second copy. §3's sentence is implemented — sfho just does not use it. Fold
   into the shared one.
3. **Merge `find_mixed_window` into `locate_window`.** `boundaries.py:107` and
   `solver.py:792` have **identical signatures**
   `(par, flags, n_B_grid, eta, spec, vmit_params=None, T=0.0, ...)`. One job,
   two names, two modules. `boundaries.py` wins — it also holds
   `locate_windows`.
4. **Confirmatory, no module moves.** `create_custom_parametrization` is already
   in `eos/sfho/nmp.py:233` (it is the NMP inversion, correctly placed by §5);
   `get_sfho_general` is in `parameters.py:568` and is the plain alternate
   constructor. Rename each to a `from_*` form in place.
5. **`build_mixed_eos_table` -> `build_hybrid_table`.** `hybrid.py:118` stitches
   hadronic + mixed + quark into one branch; `table.py:323 build_table` is
   §13's grid driver and keeps the vocabulary word. The name says the job.

Only item 5 is a preference; the other four are settled by what the code
already contains.

Open for execution.
