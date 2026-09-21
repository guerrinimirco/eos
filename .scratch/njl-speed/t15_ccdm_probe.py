"""Ticket 15's owed measurement: does `free` find anything in eos.ccdm that
the named asymmetric seeds `uSC`/`dSC` do not? Ticket 12's t12_probe.py,
pointed at ccdm: each candidate alone and COLD, 12 points over T = 0/30/50,
on the deconfined side of the shipped set (tables there start at 1.3 fm^-3).

    python3 .scratch/njl-speed/t15_ccdm_probe.py .scratch/njl-speed/t15_ccdm_probe.json
"""
import json
import sys
import time

import numpy as np
import scipy

from eos.ccdm import Parameters, SpeciesFlags, solve
from eos.general.pairing import realised_pattern

par = Parameters.default()
paired = SpeciesFlags(csc=True)
print(f"python {sys.version.split()[0]}, numpy {np.__version__}, "
      f"scipy {scipy.__version__}", flush=True)

rows = []
for T in (0.0, 30.0, 50.0):
    for n_B in (1.3, 1.6, 2.0, 2.5):
        rec = {"T": T, "n_B": n_B, "cand": {}}
        for p in ("unpaired", "2SC", "CFL", "uSC", "dSC", "free"):
            t0 = time.perf_counter()
            try:
                q = solve(par, "beta_eq_neutrinoless", n_B, T, paired,
                          patterns=(p,), backend="fast")
                rec["cand"][p] = dict(conv=bool(q.converged),
                                      branch=q.branch,
                                      realised=realised_pattern(q.Delta),
                                      f=float(q.f),
                                      D=[float(d) for d in q.Delta],
                                      s=time.perf_counter() - t0)
            except Exception as exc:
                rec["cand"][p] = dict(conv=False, realised="ERROR",
                                      err=str(exc)[:120],
                                      s=time.perf_counter() - t0)
        rows.append(rec)
        print(json.dumps(rec), flush=True)

with open(sys.argv[1], "w") as fh:
    json.dump({"finiteT": rows}, fh, indent=1)
print("DONE", flush=True)
