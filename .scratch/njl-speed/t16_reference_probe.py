"""Ticket 16: the bound on the REFERENCE backend, which ticket 13 never ran.

With no Jacobian a cross-seeded candidate gets `hybr` alone and no cold retry
(`attempt` has no Newton rung and no differenced repeat there), so the edit
changes this path too. The pinned grid's first 60 densities (0.5 -> 0.817
fm^-3, the same step) span the 13 rootless CFL candidates and the 2SC -> CFL
onset at 0.6583. The arm is chosen by PYTHONPATH.

    PYTHONPATH=<tree> python3 .scratch/njl-speed/t16_reference_probe.py OUT.json
"""
import json
import os
import sys
import time

import numpy as np
import scipy

import eos
from eos import njl

NB = np.linspace(0.5, 1.55, 200)[:60]
par = njl.Parameters.named("rg_njl1")
w0, c0 = time.perf_counter(), time.process_time()
result = njl.eos_table(par, "beta_eq_neutrinoless", njl.SpeciesFlags(csc=True),
                       {"nB": NB, "T": np.array([0.0])}, backend="reference")
wall, cpu = time.perf_counter() - w0, time.process_time() - c0
rows = njl.table.rows_from_result(result)
stack = (f"python {sys.version.split()[0]}, numpy {np.__version__}, "
         f"scipy {scipy.__version__}")
print(f"{eos.__file__}\n{stack}  loadavg {os.getloadavg()}\n"
      f"{len(rows)} rows  wall {wall:.1f}s cpu {cpu:.1f}s", flush=True)
out = [{"n_B": float(r["n_B"]), "P": float(r["P"]), "realised":
        r["pattern_realised"], "pattern": r["pattern"]} for r in rows]
with open(sys.argv[1], "w") as fh:
    json.dump({"eos": eos.__file__, "stack": stack, "wall_s": wall,
               "cpu_s": cpu, "winners": out}, fh)
