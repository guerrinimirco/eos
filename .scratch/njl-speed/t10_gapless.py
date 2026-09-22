"""Ticket 10: the node rules where a coarse rule fails first -- GAPLESS states.

The pinned beta-eq tables deliver no gapless row in either single-pattern
sweep, so the gapless check is taken where this model's gapless ground state
is documented: fixed_YC, Y_C = 0.1, leptons, T = 0 -- the table that once
reported metastable 2SC at 98 of 100 points because the T = 0 occupation step
of a gapless CFL state sat inside a Gauss panel. It runs the DEFAULT
enumeration, so losing the gapless CFL state shows up as a realised-pattern
flip to 2SC, which is exactly the failure the ticket names.

Same arms and same vacuum re-pointing as t10_tables.py; output in its format,
with the grid stored, so t10_gate.py reads it.

    PYTHONPATH=. python3 t10_gapless.py BACKEND N_LO N_HI N_PTS OUT.json ARM...
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                            # noqa: E402

import t10_tables as T                        # noqa: E402  (vacuum re-pointing)
from eos import njl                           # noqa: E402

backend, lo, hi, npts, out = sys.argv[1:6]
arms = sys.argv[6:]
grid = np.linspace(float(lo), float(hi), int(npts))
par = njl.Parameters.named("rg_njl1")
flags = njl.SpeciesFlags(csc=True)
#: T10_PATTERNS=CFL holds one pattern, so every point is own-seeded and
#: the node rule is compared on a FIXED path; unset, the default enumeration
patterns = (tuple(os.environ["T10_PATTERNS"].split(","))
            if os.environ.get("T10_PATTERNS") else None)
print(f"eos from {T.eos.__file__}")
print(f"=== ticket 10 gapless: fixed_YC Y_C=0.1 leptons T=0, patterns {patterns}, "
      f"backend {backend!r}, n_B {lo}-{hi} x {npts}\n{T.B.bench_stack()}\n"
      f"loadavg {os.getloadavg()}", flush=True)
res = {}
for arm in arms:
    H, V = (int(x) for x in arm.split("/"))
    T.set_vacuum(V)
    w0, c0 = time.perf_counter(), time.process_time()
    result = njl.eos_table(par, "fixed_YC", flags,
                           {"nB": grid, "T": np.array([0.0])},
                           fixed={"Y_C": 0.1}, leptons=True, backend=backend,
                           patterns=patterns,
                           pair_nodes_per_panel=H)
    wall, cpu = time.perf_counter() - w0, time.process_time() - c0
    rows = njl.table.rows_from_result(result)
    out_rows = [{k: (bool(r[k]) if k == "gapless" else
                     (r[k] if isinstance(r[k], str) else float(r[k])))
                 for k in ("n_B", "P", "eps", "mu_B", "pattern_realised",
                           "gapless", "Delta_1", "Delta_2", "Delta_3",
                           "M_u", "M_d", "M_s")} for r in rows]
    res[arm] = [{"wall": wall, "cpu": cpu, "rows": out_rows}]
    gl = [r["n_B"] for r in out_rows if r["gapless"]]
    pats = {}
    for r in out_rows:
        pats[r["pattern_realised"]] = pats.get(r["pattern_realised"], 0) + 1
    print(f"  arm {arm:6s} {wall * 1e3 / len(grid):8.1f} ms/pt wall "
          f"{cpu * 1e3 / len(grid):8.1f} cpu  rows {len(rows)}  {pats}  "
          f"gapless at {len(gl)}: {[round(n, 4) for n in gl]}", flush=True)
with open(out, "w") as fh:
    json.dump({"what": "gapless", "pattern": ",".join(patterns) if patterns else "default", "backend": backend,
               "grid": grid.tolist(), "stack": T.B.bench_stack(),
               "head": os.popen("git rev-parse --short HEAD").read().strip(),
               "results": res}, fh)
