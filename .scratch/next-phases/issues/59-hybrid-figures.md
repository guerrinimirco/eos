# notebooks/hybrid_eos — figures, the TOV pass, and the swap cell

Type: task
Status: open
Blocked by: 58
Parent: ../map.md

## Question

Three parts.

1. **Figures** for the headline DD2 + vMIT construction: P vs n_B with the
   window marked, the quark volume fraction chi across the transition, the
   per-phase decomposition of each conserved charge, and the transition curves
   n_onset / n_offset. These are the composite engine's own observables (§5) and
   are why `mixed` earned a notebook.

2. **The TOV pass.** `HybridResult.table.to_tov()` is the declared contract into
   `eos.astro.tov`, and `mixed/hybrid.py` and `mixed/scan.py` already import it —
   the one §1 exception. End on M–R. Run §8's gate (P non-decreasing,
   0 <= c_s^2 <= 1) BEFORE integrating and report its status, never a mass
   computed past a failed gate.

3. **The swap cell.** Re-run with **DID + NJL** and **DID + CCDM**, changing both
   sides of the pair at once. Depth is a runtime call — a converged table is the
   floor, the full headline treatment is not required. Whatever is skipped for
   runtime is printed, not silently dropped.

**The comparison ticket 05 promised is now OPTIONAL, and its target has moved.**
Ticket 05 held `notebooks/eos_tables_DD2vMIT/` — 32 tables, 42 figures from the
retired `DD2vMIT_general1oPT.ipynb` — because a replacement had to be measured
against it rather than asserted. Two things have since changed: **the user has
confirmed none of the 42 figures is published**, so nothing downstream depends on
reproducing them exactly; and the folder left the repo with `output_old/`, so its
current path is not known here.

So ticket 03's held-until condition is **discharged**: nothing is waiting on this
notebook to regenerate anything. If the user can point at where `output_old/`
went, an eyeball comparison of the DD2+vMIT figures is still worth ten minutes —
it is the cheapest end-to-end check that the engine gives the same physics it
gave before the refactor. If they cannot, say so and move on; do not hunt for it.

Resolved when the notebook executes end to end and the comparison is reported.
