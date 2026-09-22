"""Ticket 18's gate over t10_gapless.py outputs.

Per arm: (a) gapless CFL delivered at every density of 0.775-1.075; (b) P
non-decreasing over the rows from 0.775 up (the 0.75 -> 0.775 step is the
2SC -> CFL switch of a fixed-n_B sweep, which ticket 10's good arms carry too);
(c) against the control -- HEAD's reference 24/24 table from ticket 10, which
kept the gapless state -- the realised state at all 45 densities and the worst
|dP|/P.

    python3 t18_gate_check.py RUN.json...
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WINDOW = np.round(np.arange(0.775, 1.0751, 0.025), 3)
control = {round(r["n_B"], 3): r for r in json.load(open(
    os.path.join(HERE, "t10_gapless_ref.json")))["results"]["24/24"][0]["rows"]}

for path in sys.argv[1:]:
    run = json.load(open(path))
    for arm, reps in run["results"].items():
        rows = {round(r["n_B"], 3): r for r in reps[0]["rows"]}
        lost = [n for n in WINDOW
                if not (n in rows and rows[n]["pattern_realised"] == "CFL"
                        and rows[n]["gapless"])]
        upper = sorted(n for n in rows if n >= 0.775)
        P = np.array([rows[n]["P"] for n in upper])
        falls = [(upper[i], upper[i + 1]) for i in range(len(upper) - 1)
                 if P[i + 1] < P[i]]
        common = sorted(set(rows) & set(control))
        mism = [n for n in common if rows[n]["pattern_realised"]
                != control[n]["pattern_realised"]]
        dP = max(abs(rows[n]["P"] - control[n]["P"]) / abs(control[n]["P"])
                 for n in common)
        ok = not lost and not falls and not mism and len(common) == 45
        print(f"{run['backend']:9s} {arm:6s} {'PASS' if ok else 'FAIL'}  "
              f"rows {len(rows)}  gapless CFL lost at {len(lost)} of "
              f"{len(WINDOW)} {lost}  P falls from 0.775 up {falls}  "
              f"realised mismatches vs control {len(mism)} {mism}  "
              f"worst |dP|/P {dP:.1e}")
