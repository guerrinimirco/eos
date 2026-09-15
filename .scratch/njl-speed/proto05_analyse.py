"""PROTOTYPE (wayfinder ticket 05) -- throwaway. Reads proto05_baseline.json."""
import json
from collections import defaultdict

import numpy as np

D = json.load(open(".scratch/njl-speed/proto05_baseline.json"))
recs = D["records"]
print(D["stack"])
print(f"table: {D['n_rows']} rows, wall {D['wall_s']:.1f}s cpu {D['cpu_s']:.1f}s "
      f"= {1e3 * D['wall_s'] / D['n_rows']:.1f} ms/pt "
      f"({len(recs)} candidates)\n")

total = sum(r["wall_s"] for r in recs)

# ---- 1. where the time is, by pattern and by whether the candidate had a
#         seed of its OWN pattern --------------------------------------------
print("=== cost census: pattern x seed provenance ===")
print(f"{'pattern':>9} {'seed':>12} {'n':>5} {'total s':>9} {'%':>6} "
      f"{'median ms':>10} {'max ms':>9}")
cells = defaultdict(list)
for r in recs:
    seed = "cross" if r["cross"] else ("own-warm" if r["warm"] else "cold")
    cells[(r["pattern"], seed)].append(r["wall_s"])
for key in sorted(cells, key=lambda k: -sum(cells[k])):
    v = np.array(cells[key])
    print(f"{key[0]:>9} {key[1]:>12} {len(v):5d} {v.sum():9.1f} "
          f"{100 * v.sum() / total:5.1f}% {1e3 * np.median(v):10.1f} "
          f"{1e3 * v.max():9.1f}")
print()

# ---- 2. the screen: does the first Newton run separate useful from not? -----
def useful(r):
    """A candidate is USEFUL if it converged AND held the layout it was asked
    for -- that is exactly the test `solve`'s `_seeds` already applies, and a
    candidate failing it either lost or duplicated a rival."""
    return r["converged"] and r["realised"] == r["pattern"]

print("=== screen error vs. what the full ladder produced ===")
print(f"{'outcome':>34} {'n':>5} {'screen med':>11} {'screen p10':>11} "
      f"{'screen p90':>11} {'cost s':>8}")
groups = defaultdict(list)
for r in recs:
    if r["pattern"] == "free":
        tag = "free -> " + r["realised"]
    elif useful(r):
        tag = "held its layout"
    elif r["converged"]:
        tag = f"collapsed -> {r['realised']}"
    else:
        tag = "did not converge"
    groups[tag].append(r)
for tag in sorted(groups, key=lambda t: -sum(x["wall_s"] for x in groups[t])):
    g = groups[tag]
    s = np.array([x["newton"][0][0] if x["newton"] else np.nan for x in g])
    print(f"{tag:>34} {len(g):5d} {np.nanmedian(s):11.1e} "
          f"{np.nanpercentile(s, 10):11.1e} {np.nanpercentile(s, 90):11.1e} "
          f"{sum(x['wall_s'] for x in g):8.1f}")
print()

# ---- 3. per density: the winner, the margin, and who duplicated whom --------
by_nB = defaultdict(list)
for r in recs:
    by_nB[round(r["n_B"], 9)].append(r)

dup_cost = 0.0
dup_n = 0
uniq_realised_from_free = []
tie_wins = 0
for nB in sorted(by_nB):
    cand = by_nB[nB]
    conv = [c for c in cand if c["converged"]]
    if not conv:
        continue
    win = min(conv, key=lambda c: c["f"])
    if win["pattern"] != win["realised"]:
        tie_wins += 1
    seen = {}
    for c in cand:
        if not c["converged"]:
            continue
        key = c["realised"]
        if key in seen:
            dup_cost += c["wall_s"]
            dup_n += 1
        else:
            seen[key] = c["pattern"]
    if "free" in [c["pattern"] for c in conv]:
        f = [c for c in conv if c["pattern"] == "free"][0]
        others = {c["realised"] for c in conv if c["pattern"] != "free"}
        if f["realised"] not in others:
            uniq_realised_from_free.append((nB, f["realised"]))

print(f"candidates whose realised state DUPLICATES an earlier candidate at the "
      f"same density: {dup_n} of {len(recs)}, {dup_cost:.1f}s "
      f"({100 * dup_cost / total:.1f}% of solve time)")
print(f"densities where the winner's LAYOUT is not the state it realises: "
      f"{tie_wins}")
print(f"densities where `free` found a state no other candidate did: "
      f"{len(uniq_realised_from_free)}")
for nB, rp in uniq_realised_from_free[:20]:
    print(f"    n_B={nB:.4f} -> {rp}")
print()

# ---- 4. the branch map -----------------------------------------------------
print("=== realised branches along the sweep (winner) ===")
w = D["winners"]
prev = None
for row in w:
    tag = f"{row['pattern']}/{row['realised']}"
    if tag != prev:
        print(f"  n_B={row['n_B']:.4f}  {tag}")
        prev = tag
