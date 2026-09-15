"""PROTOTYPE (wayfinder ticket 05) -- throwaway. The map's correctness gate:
same pattern at every point, and P to 1e-8. Judged on `pattern_realised`, not
on `pattern`: the census showed the LAYOUT column is not stable even in the
baseline (free and CFL land on one root and the tie flips on the last digit
of f), so the name that says what the matter IS is the one to gate on."""
import json
import sys

import numpy as np

base = json.load(open(".scratch/njl-speed/proto05_baseline.json"))["winners"]
name = sys.argv[1]
new = json.load(open(f".scratch/njl-speed/proto05_{name}.json"))["winners"]

print(f"{name} vs baseline: {len(base)} vs {len(new)} rows")
b = {round(r["n_B"], 6): r for r in base}
n = {round(r["n_B"], 6): r for r in new}
missing = sorted(set(b) - set(n))
extra = sorted(set(n) - set(b))
if missing:
    print(f"  MISSING {len(missing)} densities: {missing[:8]}")
if extra:
    print(f"  EXTRA {len(extra)} densities: {extra[:8]}")

bad_state, worst = [], (0.0, None)
for k in sorted(set(b) & set(n)):
    if b[k]["realised"] != n[k]["realised"]:
        bad_state.append((k, b[k]["realised"], n[k]["realised"]))
    rel = abs(n[k]["P"] - b[k]["P"]) / max(abs(b[k]["P"]), 1e-30)
    if rel > worst[0]:
        worst = (rel, k)
print(f"  realised-state mismatches: {len(bad_state)}")
for k, x, y in bad_state[:12]:
    print(f"     n_B={k:.4f}  baseline {x} -> {name} {y}")
print(f"  worst |dP|/P = {worst[0]:.3e} at n_B={worst[1]}  "
      f"(gate 1e-8: {'PASS' if worst[0] <= 1e-8 and not bad_state else 'FAIL'})")

# the layout column, reported for completeness -- it is NOT the gate
lay = sum(1 for k in set(b) & set(n) if b[k]["pattern"] != n[k]["pattern"])
print(f"  layout-column (`pattern`) differences: {lay}  [not gated; see above]")
