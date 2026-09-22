"""Ticket 18: row diff between two arms' dumped tables -- t18_bench.py's fast
default table ("rows") or t16_reference_probe.py's reference table
("winners"). Reports bit-identity, realised mismatches and worst |dP|/P.

    python3 t18_rowdiff.py A.json B.json
"""
import json
import sys

a, b = (json.load(open(p)) for p in sys.argv[1:3])
key = "rows" if "rows" in a else "winners"
ra = {round(r["n_B"], 9): r for r in a[key]}
rb = {round(r["n_B"], 9): r for r in b[key]}
common = sorted(set(ra) & set(rb))
identical = all(ra[n] == rb[n] for n in common) and len(ra) == len(rb)
mism = [n for n in common if ra[n]["realised"] != rb[n]["realised"]]
dP = max(abs(ra[n]["P"] - rb[n]["P"]) / abs(ra[n]["P"]) for n in common)
cfl = sum(r["realised"] == "CFL" for r in rb.values())
print(f"{sys.argv[1]} vs {sys.argv[2]} ({key}): rows {len(ra)} / {len(rb)}, "
      f"only-in-one {len(set(ra) ^ set(rb))}, bit-identical {identical}, "
      f"realised mismatches {len(mism)}, worst |dP|/P {dP:.1e}, CFL rows {cfl}")
