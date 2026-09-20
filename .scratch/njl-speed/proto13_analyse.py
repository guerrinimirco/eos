"""PROTOTYPE (wayfinder ticket 13) -- throwaway. Reads proto13_rungs.json.

Three questions ticket 05's stage census could not answer:
  1. which rung rescues the ONE CFL candidate that holds its layout at the
     onset, and what does that rescue SPEND;
  2. what the 13 dead CFL candidates spend in the same rung;
  3. whether an evaluation cap separates them.
"""
import json
from collections import Counter, defaultdict

import sys
d = json.load(open(".scratch/njl-speed/proto13_rungs"
                 + ("_VX" if len(sys.argv) > 1 else "") + ".json"))
calls = d["calls"]
print(f"{d['stack']}  loadavg {d['loadavg']}")
print(f"{d['n_rows']} rows, wall {d['wall_s']:.1f}s cpu {d['cpu_s']:.1f}s, "
      f"{1e3 * d['wall_s'] / d['n_rows']:.1f} ms/pt, {len(calls)} candidates\n")

# ---- stage shares (ticket 05's table, re-taken) -------------------------
tot = sum(c["wall_s"] for c in calls)
stage_s, stage_n = defaultdict(float), Counter()
for c in calls:
    for rung in c["rungs"]:
        for st in rung["stages"]:
            key = (rung["rung"], st["stage"])
            stage_s[key] += st["wall_s"]
            stage_n[key] += 1
print(f"candidate wall total {tot:.1f}s")
print(f"{'rung':<5}{'stage':<10}{'n':>6}{'s':>9}{'%':>8}")
for key in sorted(stage_s, key=lambda k: -stage_s[k]):
    print(f"{key[0]:<5}{key[1]:<10}{stage_n[key]:>6}{stage_s[key]:>9.1f}"
          f"{100 * stage_s[key] / tot:>7.1f}%")

# ---- what the ladder below the first Newton bought ----------------------
def entered(c, rung):
    return any(r["rung"] == rung for r in c["rungs"])

def newton_ok(c):
    for r in c["rungs"]:
        if r["rung"] == "AB":
            return any(s["stage"] == "newton" and s["ok"] for s in r["stages"])
    return False

past = [c for c in calls if not newton_ok(c)]
print(f"\ncandidates whose first Newton did NOT reach the gate: {len(past)}"
      f"  ({sum(c['wall_s'] for c in past):.1f}s, "
      f"{100 * sum(c['wall_s'] for c in past) / tot:.1f}%)")
buckets = Counter()
bwall = defaultdict(float)
for c in past:
    if c["converged"]:
        tag = (f"{c['pattern']} -> held its layout" if c["realised"] == c["pattern"]
               else f"{c['pattern']} -> converged as {c['realised']}")
    else:
        tag = f"{c['pattern']} -> did not converge"
    buckets[tag] += 1
    bwall[tag] += c["wall_s"]
print(f"{'outcome':<38}{'n':>5}{'s':>9}{'%':>8}")
for tag in sorted(bwall, key=lambda t: -bwall[t]):
    print(f"{tag:<38}{buckets[tag]:>5}{bwall[tag]:>9.1f}"
          f"{100 * bwall[tag] / tot:>7.1f}%")

# ---- the hybr evaluation budget, split by outcome -----------------------
print("\nMINPACK 'hybr' rung (rung B), evaluations by outcome:")
print(f"{'outcome':<38}{'n':>5}{'nfev min/med/max':>22}{'s':>9}")
rows = defaultdict(list)
for c in past:
    fev = [st["nfev"] for r in c["rungs"] if r["rung"] in ("AB", "C")
           for st in r["stages"] if st["stage"] == "hybr"]
    if not fev:
        continue
    if c["converged"]:
        tag = (f"{c['pattern']} -> held its layout" if c["realised"] == c["pattern"]
               else f"{c['pattern']} -> converged as {c['realised']}")
    else:
        tag = f"{c['pattern']} -> did not converge"
    rows[tag].append((sum(fev), c["wall_s"], c["n_B"]))
for tag in sorted(rows, key=lambda t: -sum(r[1] for r in rows[t])):
    v = sorted(r[0] for r in rows[tag])
    s = sum(r[1] for r in rows[tag])
    print(f"{tag:<38}{len(v):>5}"
          f"{v[0]:>8}{v[len(v) // 2]:>7}{v[-1]:>7}{s:>9.1f}")
    if "held its layout" in tag and "CFL" in tag:
        for n, w, nb in sorted(rows[tag], key=lambda r: r[2]):
            print(f"        n_B={nb:.4f}  {n} evaluations, {w:.2f}s")

# ---- rung D: what it rescued -------------------------------------------
dd = [c for c in calls if entered(c, "D")]
print(f"\nrung D entered by {len(dd)} candidates "
      f"({sum(c['wall_s'] for c in dd):.1f}s):")
for c in sorted(dd, key=lambda c: c["n_B"]):
    dwall = sum(r["wall_s"] for r in c["rungs"] if r["rung"] == "D")
    dok = any(r["ok"] for r in c["rungs"] if r["rung"] == "D")
    print(f"   n_B={c['n_B']:.4f} {c['pattern']:<9} warm={c['warm']!s:<5} "
          f"D {dwall:6.2f}s ok={dok!s:<5} -> converged={c['converged']!s:<5} "
          f"realised={c['realised']}")
cc = [c for c in calls if entered(c, "C")]
print(f"rung C entered by {len(cc)} candidates")
