# Claude Code kickoff prompt — Phase 2

Paste the block below into Claude Code with the repo root as cwd.

---

## The prompt

> We are starting Phase 2 of this project: a mixed-phase equation-of-state
> engine coupling the existing DD2 hadronic model to the existing vMIT quark
> model through a first-order phase transition, with a continuous local/global
> charge-neutrality parameter η.
>
> Read these three documents in order before writing any code:
>
> 1. `CLAUDE.md` — repository conventions. These are invariants, not
>    suggestions. Pay particular attention to the sign conventions (S = +1 per
>    s-quark; C excludes leptons), the reference/fast split, and the note on
>    autodiff.
> 2. `docs/phase2/STEP0_AUDIT.md` — which equilibrium conditions have been
>    verified against the thesis derivation, and the one place where the physics
>    admits two readings.
> 3. `docs/phase2/SPECIFICATION_AND_PLAN.md` — the physics specification,
>    architecture, and the P0–P9 milestone plan with validation gates.
>
> Then work through the milestones in order, starting with **P0**. Do not skip
> ahead: each milestone has a validation gate and the gate must be green before
> the next one starts.
>
> Ground rules for this work:
>
> - `eos/dd2/`, `eos/vmit/`, `eos/general/`, and `eos/tov/` are the validated
>   baseline. Consume them as libraries. Do not refactor, rename, or "improve"
>   them. If you believe a change there is genuinely required, stop and ask.
> - `eos/zlvmit/` is a first-generation implementation. Read it for behaviour
>   and for its bracketing heuristics, but do not copy its structure — its
>   per-mode branch duplication is what this rewrite exists to eliminate. It
>   stays in the tree as a regression oracle.
> - The central architectural requirement is in §1.5 of the specification:
>   assemble the unknown vector and residual list from the per-charge regime
>   assignment. The four named modes must be four *configurations* of one
>   solver, and an unnamed combination must work without new code. If you find
>   yourself writing a fourth near-duplicate residual function, stop — the
>   design has gone wrong.
> - New code goes in `eos/mixed/` and `test/mixed/`. Follow the existing
>   docstring style: state the physics the module implements and cite the
>   specification section.
> - Run the existing test suite before and after each milestone. It must stay
>   green — there are 93 test functions across 17 files in `test/dd2/`, and none
>   of them should change.
> - When the physics is genuinely ambiguous and the audit document does not
>   already resolve it, ask rather than picking a convention silently. §5 of the
>   specification lists the known cases.
>
> Start by reading the three documents and `eos/dd2/solver.py` (note that
> `solve_octet` already carries the `charge_mode`/`strange_mode`/`lepton_mode`
> pattern — Phase 2 extends this idea, it does not reinvent it). Then tell me
> your plan for P0 before implementing it.

---

## Notes for the human

**Work on a branch.** `git checkout -b phase2-mixed` before starting, so the
diff is reviewable and `main` keeps the validated Phase-1 state.

**The `CLAUDE.md` file is read automatically** by Claude Code on every
invocation in this repo — you do not need to re-paste conventions in later
prompts. If a convention gets re-litigated in conversation, add the resolution
to `CLAUDE.md` so it persists.

**After each milestone**, the useful loop is: review the diff yourself, then
bring the branch back to Claude Science for numerical validation — running the
suite, checking the thermodynamic identities, verifying the analytic Jacobian
against finite differences, confirming the η endpoint limits, and pushing tables
through the TOV solver. That is the division of labour: Claude Code implements,
this session checks the physics.

**Baseline test state** as of writing: `python -m pytest test/ -q` gives
**123 passed, 10 skipped, 2 failed**. Both failures are *environment*, not code,
and Claude Code should not try to "fix" them:

- `test_dd2_m4_tov.py::test_tov_dd2_nucleonic_pipeline` — asserts
  `R_1.4 ≈ 13.2 km` but gets `12.33 km`. Cause: `CRUST_PATHS["BPS"]` points at
  `~/Desktop/Research/Crust/BPST0.dat`, outside the repo. When that file is
  unreachable, `eos/dd2/verify/tov.py::mass_radius` silently falls back to
  `crust="No"`, and a crustless star is ~0.9 km smaller at 1.4 M☉. `M_max`
  passes because the crust barely affects it. On a machine where the crust file
  is present this test passes. **Do not** loosen the tolerance or change the
  fallback — if you see this failure, check the crust path first.
- `test_notebook_api.py::test_tov_plots` — `ModuleNotFoundError: nucleation`.
  `eos/dd2/notebook_api.py:308` imports `nucleation.analysis.figure` for
  observational contours; that is an external package, not part of this repo.

Treat "123 passed" as the green baseline. If a Phase-2 change drops that number,
that is a real regression.

**One open physics item** is flagged in `STEP0_AUDIT.md` §4 and specification
§1.7: the extension of the η split to trapped neutrinos is not derived anywhere
in the thesis. The specification states an assumption (neutrinos are purely
global) with its rationale, which is enough for Mode B to be implemented — but
it is an assumption, and if the eventual physics case matters you may want to
settle it before P5 rather than after.
