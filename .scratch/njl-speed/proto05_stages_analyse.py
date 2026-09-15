"""PROTOTYPE (wayfinder ticket 05) -- throwaway. Reads proto05_stages.json."""
import json
from collections import defaultdict

import numpy as np

D = json.load(open(".scratch/njl-speed/proto05_stages.json"))
recs = D["records"]
print(D["stack"])
print(f"table: {D['n_rows']} rows, wall {D['wall_s']:.1f}s cpu {D['cpu_s']:.1f}s "
      f"= {1e3 * D['wall_s'] / D['n_rows']:.1f} ms/pt, {len(recs)} candidates\n")
total = sum(r["wall_s"] for r in recs)

# `attempt` calls solve_system at most three times: #1 the jac rung, then
# EITHER #2 the reinflate rescue (jac, after a converged-but-collapsed #1)
# OR #2 the differenced rescue (no jac, after a failed #1). solve_pattern may
# then run the whole thing again from the cold guess.
print("=== where a candidate's wall time goes ===")
buckets = defaultdict(float)
counts = defaultdict(int)
for r in recs:
    first = True
    for s in r["systems"]:
        if first:
            buckets["A  first Newton (jac)"] += s["newton_s"]
            buckets["B  MINPACK+polish after it"] += s["wall_s"] - s["newton_s"]
            counts["A  first Newton (jac)"] += 1
            counts["B  MINPACK+polish after it"] += 0 if s["ok"] and s["newton_err"] is not None and s["newton_err"] <= 1e-10 else 1
            first = False
        elif s["jac"]:
            buckets["C  reinflate rescue (jac)"] += s["wall_s"]
            counts["C  reinflate rescue (jac)"] += 1
        else:
            buckets["D  differenced rescue (no jac)"] += s["wall_s"]
            counts["D  differenced rescue (no jac)"] += 1
    acc = sum(s["wall_s"] for s in r["systems"])
    buckets["E  everything else in solve_pattern"] += r["wall_s"] - acc
for k in sorted(buckets, key=lambda k: -buckets[k]):
    print(f"  {k:<34} {buckets[k]:8.1f}s {100*buckets[k]/total:5.1f}%  "
          f"n={counts[k]}")
print(f"  {'TOTAL':<34} {total:8.1f}s\n")

# The retry-from-cold: solve_pattern runs `attempt` twice when the warm one
# failed. Count candidates with more than one "rung #1".
print("=== the ladder, by how far down it went ===")
depth = defaultdict(lambda: [0, 0.0])
for r in recs:
    n = len(r["systems"])
    key = f"{n} solve_system call(s)"
    depth[key][0] += 1
    depth[key][1] += r["wall_s"]
for k in sorted(depth):
    print(f"  {k:<26} n={depth[k][0]:4d}  {depth[k][1]:8.1f}s "
          f"{100*depth[k][1]/total:5.1f}%")
print()

# What each rung below the first BOUGHT: did the candidate end up converged
# and holding its own layout?
print("=== what the rungs below the first Newton bought ===")
out = defaultdict(lambda: [0, 0.0])
for r in recs:
    if not r["systems"]:
        continue
    s0 = r["systems"][0]
    screened_ok = s0["newton_err"] is not None and s0["newton_err"] <= 1e-10
    if screened_ok:
        continue                     # the Newton alone answered; no ladder ran
    held = r["converged"] and r["realised"] == r["pattern"]
    tag = (f"{r['pattern']:>8} -> "
           + ("HELD its layout" if held
              else f"converged as {r['realised']}" if r["converged"]
              else "did not converge"))
    out[tag][0] += 1
    out[tag][1] += r["wall_s"] - s0["newton_s"]
for k in sorted(out, key=lambda k: -out[k][1]):
    print(f"  {k:<44} n={out[k][0]:4d}  {out[k][1]:8.1f}s "
          f"{100*out[k][1]/total:5.1f}%")
