"""Ticket 08: who wins each mismatch t08_analyse.py wrote out.

Every mismatched row is solved on HEAD with each single pattern ALONE and
COLD (no x0), backend="fast": unpaired, 2SC, CFL, uSC, dSC, free. The arm
whose row holds the lower f = eps - T s (the potential `solve` ranks by at
fixed density) is the winner; a landed-arm loss is a failure of the landed
path, a base-arm loss is not.

    PYTHONPATH=<repo> python3 t08_adjudicate.py MISMATCHES.json OUT.json

NOT RUN for ticket 08: t08_analyse.py found zero rows to adjudicate
(t08_mismatches.json is []). Kept as the procedure the gate names.
"""
import json
import os
import sys

import numpy as np

import eos
from eos import njl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t08 import CASES, REPO                    # noqa: E402

SINGLES = ("unpaired", "2SC", "CFL", "uSC", "dSC", "free")


def f_of(row):
    return row["eps"] - row["T"] * row["s"] if row else None


def main(src, out):
    import scipy
    assert os.path.realpath(eos.__file__).startswith(REPO + "/"), eos.__file__
    print(f"python {sys.version.split()[0]}, numpy {np.__version__}, scipy "
          f"{scipy.__version__}, eos from {eos.__file__}", flush=True)
    par = njl.Parameters.named("rg_njl1")
    res = []
    for m in json.load(open(src)):
        label, mode, T, fixed, leptons = CASES[m["case"]]
        fb, fl = f_of(m["base"]), f_of(m["landed"])
        singles = {}
        for p in SINGLES:
            r = njl.eos_point(par, mode, njl.SpeciesFlags(csc=True),
                              n_B=m["n_B"], T=T, leptons=leptons,
                              patterns=(p,), backend="fast", **fixed)
            pt = r.point
            singles[p] = {"ok": bool(r.ok), "realised": pt and pt.pattern_realised,
                          "f": pt and float(pt.f), "P": pt and float(pt.P)}
        conv = {p: s for p, s in singles.items() if s["ok"]}
        best = min(conv, key=lambda p: conv[p]["f"]) if conv else None
        if fb is None or fl is None:
            verdict = f"{'base' if fb is None else 'landed'} dropped the row"
        else:
            verdict = ("landed lower" if fl < fb else "base lower"
                       if fb < fl else "tie") + f" by {abs(fb - fl):.3e} MeV/fm^3"
        res.append({**m, "f_base": fb, "f_landed": fl, "singles": singles,
                    "best_single": best, "verdict": verdict})
        print(f"{label} n_B {m['n_B']:.4f} [{m['why']}] base "
              f"{m['base'] and m['base']['pattern_realised']} f={fb}  landed "
              f"{m['landed'] and m['landed']['pattern_realised']} f={fl}  -> "
              f"{verdict}", flush=True)
        for p, s in singles.items():
            print(f"    {p:8s} ok {s['ok']!s:5s} realised {s['realised']!s:9s} "
                  f"f {s['f']}", flush=True)
        print(f"    best single: {best}"
              + (f" f={conv[best]['f']}" if best else ""), flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
