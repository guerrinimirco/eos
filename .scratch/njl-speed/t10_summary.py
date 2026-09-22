"""Median ms/pt per arm of a t10_tables.py `time` run, with the converged-point
cost, cpu beside wall, ratios to the 24/24 arm, and whether each arm's repeats
delivered bit-identical rows (the determinism check).

    python3 t10_summary.py RUN.json
"""
import json
import sys

import numpy as np

d = json.load(open(sys.argv[1]))
n = 200
base = None
print(f"{sys.argv[1]}: {d['pattern']} {d['backend']!r}  {d['stack']}  HEAD {d['head']}")
print(f"{'arm':7s} {'wall ms/pt':>11s} {'cpu ms/pt':>10s} {'runs (wall)':>22s}"
      f" {'conv wall':>10s} {'conv cpu':>9s} {'x cpu':>6s} {'x conv cpu':>10s} rows  repeats identical")
for arm, reps in d["results"].items():
    wall = np.median([r["wall"] for r in reps]) * 1e3 / n
    cpu = np.median([r["cpu"] for r in reps]) * 1e3 / n
    cw = np.median([r["conv_wall"] / r["n_conv"] for r in reps]) * 1e3
    cc = np.median([r["conv_cpu"] / r["n_conv"] for r in reps]) * 1e3
    if base is None:
        base = (cpu, cc)
    same = all(r["rows"] == reps[0]["rows"] for r in reps)
    runs = "-".join(f"{r['wall'] * 1e3 / n:.1f}" for r in reps)
    print(f"{arm:7s} {wall:11.1f} {cpu:10.1f} {runs:>22s} {cw:10.1f} {cc:9.1f}"
          f" {base[0] / cpu:6.2f} {base[1] / cc:10.2f} {len(reps[0]['rows']):4d}  {same}")
