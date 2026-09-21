"""The TABLE half of the pinned njl-speed benchmark, run from bench.py as
extracted verbatim from 54c7be9 (imported with NJL_BENCH unset, so its own
`if BENCH_RUN:` body does not fire; the loop below is that body's table half,
line for line). The arm is chosen by PYTHONPATH; eos.__file__ is printed so
the arm is on the record.

    PYTHONPATH=<worktree> python3 bench_tables.py <label>
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eos                                    # noqa: E402
import bench as B                             # noqa: E402

print(f"=== njl-speed benchmark, TABLE half, arm {sys.argv[1]} ===\n"
      f"{B.bench_stack()}\neos from {eos.__file__}\n"
      f"loadavg at start {os.getloadavg()}")
print(f"set {B.BENCH_SET}, {B.BENCH_MODE}, T = {B.BENCH_T} MeV, "
      f"csc={B.BENCH_SPECIES.csc}, backend {B.BENCH_BACKEND!r}, "
      f"median of {B.BENCH_REPEATS}\n")

print(f"--- table, n_B = {B.BENCH_NB[0]:.2f} -> {B.BENCH_NB[-1]:.2f} fm^-3, "
      f"{len(B.BENCH_NB)} points (above chiral restoration) ---")
for name, patterns in B.BENCH_PATTERN_SETS.items():
    wall, cpu, runs, n_rows = B.bench_table(patterns, B.BENCH_NB)
    spread = f"{min(runs):.1f}-{max(runs):.1f}"
    print(f"  {name:12s} {wall * 1e3 / len(B.BENCH_NB):9.1f} ms/pt wall   "
          f"{cpu * 1e3 / len(B.BENCH_NB):9.1f} cpu   "
          f"{wall:8.1f} s median (runs {spread})   {n_rows} rows   "
          f"load {os.getloadavg()[0]:.1f}", flush=True)

print(f"\n--- table, n_B = {B.BENCH_NB_CHIRAL[0]:.2f} -> "
      f"{B.BENCH_NB_CHIRAL[-1]:.2f} fm^-3, {len(B.BENCH_NB_CHIRAL)} points "
      f"(crosses chiral restoration) ---")
wall, cpu, runs, n_rows = B.bench_table(B.BENCH_PATTERNS_MAIN,
                                        B.BENCH_NB_CHIRAL)
print(f"  {'three':12s} {wall * 1e3 / len(B.BENCH_NB_CHIRAL):9.1f} ms/pt wall"
      f"   {cpu * 1e3 / len(B.BENCH_NB_CHIRAL):9.1f} cpu   "
      f"{wall:8.1f} s median   {n_rows} rows   "
      f"load {os.getloadavg()[0]:.1f}", flush=True)
