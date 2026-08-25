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

**The comparison ticket 05 promised.** `output_old/eos_tables_DD2vMIT_from_notebooks/`
holds the 32 tables and 42 figures the retired `DD2vMIT_general1oPT.ipynb`
produced, at least some of which are published. Compare the regenerated DD2+vMIT
figures against the held ones and say in the answer whether they agree — that is
what discharges ticket 03's held-until condition as a measurement rather than an
assertion. Do not delete the held folder under this ticket; report and let the
user rule.

Resolved when the notebook executes end to end and the comparison is reported.
