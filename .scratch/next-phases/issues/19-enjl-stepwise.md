# notebooks/enjl — step-by-step treatment and benchmarks

Type: task
Status: open
Blocked by: 18
Parent: ../map.md

## Question

Stage 3, second half. Cover the same step-by-step treatment as ticket 16 **where
the physics has the corresponding part, and say plainly where it does not** — an
ENJL branch pair is not a pairing pattern, and the section must not pretend
otherwise.

Then the benchmark section, identical in shape to tickets 14 and 17: cold vs warm
`eos_point`, wall time per `n_B` line, non-converged counts and locations,
`cProfile` top ~15 with a reading, summary table, reference vs fast backend where
one ships (§9). Built on the `progress` callback; no timing hooks in library code.

Done when the benchmark table carries real numbers and the `.ipynb` is committed
without stored outputs.
