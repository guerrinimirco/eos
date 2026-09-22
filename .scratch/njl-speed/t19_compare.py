"""Ticket 19's gate over t19.py outputs: an arm's rows against the control's.

Same grid densities delivered, the same `pattern_realised` and `gapless` flag
at every one, and P to 1e-8 relative with the worst point named. P is carried
to its exact grid density along the control's dP/dn_B first (ticket 10's
t10_gate.py rule): the solve delivers n_B only to its own tolerance.

    python3 t19_compare.py CONTROL.json ARM.json [--gapless]

--gapless adds the fixed_YC checks by name: gapless CFL at every density from
0.775 up, and P non-decreasing over those rows.
"""
import json
import sys

import numpy as np

ctl_run, arm_run = (json.load(open(p)) for p in sys.argv[1:3])
grid = np.array(ctl_run["grid"])


def key(n_B):
    return round(float(grid[np.argmin(np.abs(grid - n_B))]), 6)


def rows_of(run, rep=None):
    rows = run["rows"] if rep is None else rep["rows"]
    return {key(r["n_B"]): r for r in rows}


def gate(ctl, arm, label):
    common = sorted(set(ctl) & set(arm))
    n = np.array(common)
    P = np.array([ctl[k]["P"] for k in common])
    slope = dict(zip(common, np.gradient(P, n))) if len(common) > 1 else {}
    worst, where = 0.0, None
    for k in common:
        pa = arm[k]["P"] + slope.get(k, 0.0) * (ctl[k]["n_B"] - arm[k]["n_B"])
        d = abs(pa - ctl[k]["P"]) / abs(ctl[k]["P"])
        if d > worst:
            worst, where = d, k
    realised = [k for k in common
                if arm[k]["pattern_realised"] != ctl[k]["pattern_realised"]]
    flag = [k for k in common if arm[k]["gapless"] != ctl[k]["gapless"]]
    missing, extra = sorted(set(ctl) - set(arm)), sorted(set(arm) - set(ctl))
    ok = (not realised and not flag and not missing and not extra
          and worst < 1e-8)
    print(f"{label}: {'PASS' if ok else 'FAIL'}  rows {len(arm)}/{len(ctl)}  "
          f"realised mismatches {len(realised)} {realised[:6]}  gapless-flag "
          f"mismatches {len(flag)}  missing {missing[:6]} extra {extra[:6]}  "
          f"worst |dP|/P {worst:.2e} at n_B = {where} "
          f"({ctl[where]['pattern_realised'] if where else '-'})")
    return ok


if arm_run.get("what") == "bench":
    ctl = rows_of(None, ctl_run["results"]["control"][0])
    for arm, reps in arm_run["results"].items():
        for i, rep in enumerate(reps):
            gate(ctl, rows_of(None, rep), f"{arm} repeat {i}")
else:
    ctl, arm = rows_of(ctl_run), rows_of(arm_run)
    gate(ctl, arm, f"{arm_run.get('pattern', arm_run['what'])} "
         f"{arm_run['backend']} {arm_run['arm']} vs {ctl_run['arm']}")
    if "--gapless" in sys.argv:
        for label, rows in (("control", ctl), (arm_run["arm"], arm)):
            upper = sorted(k for k in rows if k >= 0.775 - 1e-9)
            lost = [k for k in upper if not (rows[k]["pattern_realised"] == "CFL"
                                             and rows[k]["gapless"])]
            P = [rows[k]["P"] for k in upper]
            falls = [(upper[i], upper[i + 1]) for i in range(len(upper) - 1)
                     if P[i + 1] < P[i]]
            print(f"  {label:8s} from 0.775 up: {len(upper)} densities, gapless "
                  f"CFL at {len(upper) - len(lost)}, lost at {lost}; P falls "
                  f"{falls}")
