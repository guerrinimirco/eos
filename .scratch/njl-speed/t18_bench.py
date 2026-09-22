"""Ticket 18: the pinned benchmark's table rows that carry ticket 16's 2.51x,
timed with bench.py as extracted verbatim from 54c7be9 (imported with
NJL_BENCH unset, so its own body does not fire), plus one untimed build of the
default table whose rows are dumped for the fast-backend row diff. The arm is
chosen by PYTHONPATH; eos.__file__ is printed so it is on the record.

    PYTHONPATH=<tree> python3 t18_bench.py LABEL OUT.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                            # noqa: E402

import eos                                    # noqa: E402
import bench as B                             # noqa: E402
from eos import njl                           # noqa: E402

label, out = sys.argv[1], sys.argv[2]
print(f"=== ticket 18 benchmark, arm {label} ===\n{B.bench_stack()}\n"
      f"eos from {eos.__file__}\nloadavg at start {os.getloadavg()}", flush=True)
timings = {}
for name, patterns, grid in (("default (4)", None, B.BENCH_NB),
                             ("three", B.BENCH_PATTERNS_MAIN, B.BENCH_NB),
                             ("three, chiral", B.BENCH_PATTERNS_MAIN,
                              B.BENCH_NB_CHIRAL)):
    wall, cpu, runs, n_rows = B.bench_table(patterns, grid)
    timings[name] = {"wall": wall, "cpu": cpu, "runs": runs, "rows": n_rows}
    print(f"  {name:14s} {wall * 1e3 / len(grid):8.1f} ms/pt wall  "
          f"{cpu * 1e3 / len(grid):8.1f} cpu  {wall:7.1f} s median (runs "
          f"{min(runs):.1f}-{max(runs):.1f})  {n_rows} rows  load "
          f"{os.getloadavg()[0]:.1f}", flush=True)

result = njl.eos_table(njl.Parameters.named(B.BENCH_SET), B.BENCH_MODE,
                       B.BENCH_SPECIES, {"nB": B.BENCH_NB,
                                         "T": np.array([B.BENCH_T])},
                       backend=B.BENCH_BACKEND)
rows = [{"n_B": float(r["n_B"]), "P": float(r["P"]),
         "realised": r["pattern_realised"], "pattern": r["pattern"],
         "Delta": [float(r[f"Delta_{k}"]) for k in (1, 2, 3)]}
        for r in njl.table.rows_from_result(result)]
with open(out, "w") as fh:
    json.dump({"label": label, "eos": eos.__file__, "stack": B.bench_stack(),
               "timings": timings, "rows": rows}, fh)
print(f"  dumped {len(rows)} default-table rows to {out}", flush=True)
