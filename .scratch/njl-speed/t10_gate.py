"""Ticket 10's gate: each arm's delivered rows against a control arm's.

Same densities delivered, the same `pattern_realised` at every one, P to 1e-8
relative (worst point named), and the control's GAPLESS densities checked by
name -- the flag, the realised pattern and dP/P at each.

    python3 t10_gate.py RUN.json [CONTROL.json] [--control-arm 24/24]

With one file the control is that file's control arm, first repeat; with two,
the second file's (a cross-backend diff: e.g. fast rows against reference).
"""
import json
import sys

import numpy as np

GRID = np.linspace(0.5, 1.55, 200)          # bench.BENCH_NB


def key(n_B):
    """The grid density a row was asked for: the solved n_B carries the
    solve's own residual, so it is matched to the nearest grid point."""
    return round(float(GRID[np.argmin(np.abs(GRID - n_B))]), 6)

args = [a for a in sys.argv[1:] if not a.startswith("--")]
ctl_arm = "24/24"
if "--control-arm" in sys.argv:
    ctl_arm = sys.argv[sys.argv.index("--control-arm") + 1]
    args = [a for a in args if a != ctl_arm]
run = json.load(open(args[0]))
if "grid" in run:
    GRID = np.array(run["grid"])
ctl_file = json.load(open(args[1])) if len(args) > 1 else run
control = {key(r["n_B"]): r for r in ctl_file["results"][ctl_arm][0]["rows"]}
gapless = sorted(n for n, r in control.items() if r["gapless"])
# dP/dn_B along the control, to carry every P to its exact grid density: the
# solve delivers n_B only to its own tolerance, and that alone moves P
_n = np.array(sorted(control))
_P = np.array([control[n]["P"] for n in _n])
DPDN = dict(zip(_n, np.gradient(_P, _n)))


def at_grid(n, r):
    return r["P"] + DPDN[n] * (n - r["n_B"])
print(f"{args[0]}: {run['pattern']} backend {run['backend']!r}; control "
      f"{ctl_arm} of {args[1] if len(args) > 1 else 'the same run'} "
      f"({ctl_file['backend']!r}), {len(control)} rows, gapless at "
      f"{len(gapless)}: {[round(n, 4) for n in gapless]}")

for arm, reps in run["results"].items():
    for i, rep in enumerate(reps):
        rows = {key(r["n_B"]): r for r in rep["rows"]}
        missing = sorted(set(control) - set(rows))
        extra = sorted(set(rows) - set(control))
        common = sorted(set(control) & set(rows))
        pat = [n for n in common
               if rows[n]["pattern_realised"] != control[n]["pattern_realised"]]
        gl = [n for n in common if rows[n]["gapless"] != control[n]["gapless"]]
        dP = [(abs(rows[n]["P"] - control[n]["P"]) / abs(control[n]["P"]), n)
              for n in common]
        worst, at = max(dP) if dP else (0.0, None)
        dPg = max(((abs(at_grid(n, rows[n]) - at_grid(n, control[n]))
                    / abs(control[n]["P"]), n) for n in common),
                  default=(0.0, None))
        dM = max(((abs(rows[n][k] - control[n][k]) / abs(control[n][k]), n)
                  for n in common for k in ("M_u", "M_d", "M_s")),
                 default=(0.0, None))
        dD = max((abs(rows[n][k] - control[n][k])
                  / max(abs(control[n][k]), 1.0)
                  for n in common for k in ("Delta_1", "Delta_2", "Delta_3")),
                 default=0.0)
        verdict = ("PASS" if not missing and not extra and not pat
                   and worst < 1e-8 else "FAIL")
        print(f"  {arm:6s} rep {i}: {verdict}  rows {len(rows)} "
              f"(missing {len(missing)}, extra {len(extra)})  realised "
              f"mismatches {len(pat)}  gapless-flag mismatches {len(gl)}  "
              f"worst |dP|/P {worst:.3e} at n_B={at if at is None else round(at, 4)}"
              f"  worst dDelta {dD:.1e}")
        if common:
            print(f"      at the grid density: worst |dP|/P {dPg[0]:.1e} at "
                  f"{dPg[1]:.4f};  worst |dM|/M {dM[0]:.1e} at {dM[1]:.4f}")
        if missing:
            print(f"      missing: {[round(n, 4) for n in missing][:12]}")
        if pat:
            print("      realised mismatches: "
                  + ", ".join(f"{n:.4f} {control[n]['pattern_realised']}->"
                              f"{rows[n]['pattern_realised']}" for n in pat))
        if gapless and i == 0:
            named = []
            for n in gapless:
                if n not in rows:
                    named.append(f"{n:.4f} MISSING")
                    continue
                r = rows[n]
                named.append(f"{n:.4f} {r['pattern_realised']}"
                             f"{' gapless' if r['gapless'] else ' GAPPED'} "
                             f"{abs(r['P'] - control[n]['P']) / abs(control[n]['P']):.1e}")
            print("      gapless by name: " + "; ".join(named))
