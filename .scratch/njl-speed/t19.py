"""Ticket 19: the landed pairing rule (hot 24 / vacuum 12) against HEAD's 24/24.

ARMS, all in ONE tree (the landed one), so the ratio isolates the rule:

    control  both vacuum lookups re-pointed to 24 nodes by setting the ONE
             constant in the two modules that read it -- HEAD's rule, and
             checked bit-identical to HEAD's rows (t18_bench_fix.json)
    landed   the tree as it is: hot 24 / vacuum 12
    hot12    landed + pair_nodes_per_panel=12: 12/12, the per-call T = 0 option

    PYTHONPATH=. python3 t19.py table   ARM PATTERNS BACKEND OUT.json
    PYTHONPATH=. python3 t19.py gapless ARM BACKEND OUT.json
    PYTHONPATH=. python3 t19.py finiteT ARM BACKEND OUT.json
    PYTHONPATH=. python3 t19.py bench   REPEATS OUT.json ARM...

PATTERNS is default | 2SC | CFL. `bench` is bench.py's pinned default table
(extracted verbatim from 54c7be9), the arms INTERLEAVED repeat by repeat so
they share one load window, cpu beside wall and the loadavg at each run.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                            # noqa: E402

import eos                                    # noqa: E402
import bench as B                             # noqa: E402
from eos import njl                           # noqa: E402
import eos.njl.thermodynamics as th           # noqa: E402
import eos.njl.backends.jacobian as jac       # noqa: E402

LANDED_VACUUM = th.VACUUM_NODES_PER_PANEL
KEYS = ("n_B", "P", "eps", "s", "mu_B", "pattern_realised", "gapless",
        "Delta_1", "Delta_2", "Delta_3", "M_u", "M_d", "M_s")


def set_arm(arm):
    """Point the rule at `arm`; return the pair_nodes_per_panel it passes."""
    vac = 24 if arm == "control" else LANDED_VACUUM
    th.VACUUM_NODES_PER_PANEL = vac
    jac.VACUUM_NODES_PER_PANEL = vac
    return 12 if arm == "hot12" else None


def dump(rows):
    out = []
    for r in rows:
        d = {}
        for k in KEYS:
            v = r.get(k)
            if k == "gapless":
                d[k] = bool(v)
            elif isinstance(v, str) or v is None:
                d[k] = v
            else:
                d[k] = float(v)
        out.append(d)
    return out


def header(what):
    print(f"=== ticket 19 {what} ===\n{B.bench_stack()}\neos from "
          f"{eos.__file__}\nHEAD {os.popen('git rev-parse --short HEAD').read().strip()}"
          f"  loadavg {tuple(round(x, 1) for x in os.getloadavg())}", flush=True)


def table(arm, patterns, backend, out):
    header(f"table {arm} {patterns} {backend}")
    nodes = set_arm(arm)
    pats = None if patterns == "default" else (patterns,)
    par = njl.Parameters.named(B.BENCH_SET)
    w0, c0 = time.perf_counter(), time.process_time()
    result = njl.eos_table(par, B.BENCH_MODE, B.BENCH_SPECIES,
                           {"nB": B.BENCH_NB, "T": np.array([B.BENCH_T])},
                           patterns=pats, backend=backend,
                           pair_nodes_per_panel=nodes)
    wall, cpu = time.perf_counter() - w0, time.process_time() - c0
    rows = dump(njl.table.rows_from_result(result))
    print(f"  {len(rows)} rows  wall {wall:.1f} s  cpu {cpu:.1f} s  "
          f"loadavg {os.getloadavg()[0]:.1f}", flush=True)
    json.dump({"what": "table", "arm": arm, "pattern": patterns,
               "backend": backend, "grid": B.BENCH_NB.tolist(),
               "eos": eos.__file__, "stack": B.bench_stack(),
               "wall": wall, "cpu": cpu, "rows": rows}, open(out, "w"))


GAPLESS_GRID = np.linspace(0.5, 1.6, 45)       # t10_gapless.py's


def gapless(arm, backend, out):
    header(f"gapless {arm} {backend}: fixed_YC Y_C=0.1 leptons T=0, default "
           f"enumeration")
    nodes = set_arm(arm)
    w0, c0 = time.perf_counter(), time.process_time()
    result = njl.eos_table(njl.Parameters.named("rg_njl1"), "fixed_YC",
                           njl.SpeciesFlags(csc=True),
                           {"nB": GAPLESS_GRID, "T": np.array([0.0])},
                           fixed={"Y_C": 0.1}, leptons=True, backend=backend,
                           pair_nodes_per_panel=nodes)
    wall, cpu = time.perf_counter() - w0, time.process_time() - c0
    rows = dump(njl.table.rows_from_result(result))
    gl = [round(r["n_B"], 4) for r in rows if r["gapless"]]
    print(f"  {len(rows)} rows  wall {wall:.1f} s  cpu {cpu:.1f} s  gapless "
          f"at {len(gl)}: {gl}", flush=True)
    json.dump({"what": "gapless", "arm": arm, "backend": backend,
               "grid": GAPLESS_GRID.tolist(), "eos": eos.__file__,
               "stack": B.bench_stack(), "wall": wall, "cpu": cpu,
               "rows": rows}, open(out, "w"))


#: ticket 10 section 4's points: Parameters.default(), one pattern held
FINITE_T = (("2SC", 1.4, 20.0), ("CFL", 1.2, 30.0), ("CFL", 1.2, 20.0))


def finite_t(arm, backend, out):
    header(f"finiteT {arm} {backend}")
    nodes = set_arm(arm)
    par = njl.Parameters.default()
    res = []
    for pattern, n_B, T in FINITE_T:
        r = njl.eos_point(par, "beta_eq_neutrinoless", njl.SpeciesFlags(csc=True),
                          n_B=n_B, T=T, patterns=(pattern,), backend=backend,
                          pair_nodes_per_panel=nodes)
        p = r.point
        res.append({"pattern": pattern, "n_B": n_B, "T": T, "ok": bool(r.ok),
                    "realised": p.pattern_realised, "P": float(p.P),
                    "s": float(p.s), "eps": float(p.eps),
                    "M": [float(x) for x in p.M],
                    "Delta": [float(x) for x in p.Delta]})
        print(f"  {pattern} n_B={n_B} T={T}: ok {r.ok} realised "
              f"{p.pattern_realised} P {p.P!r} s {p.s!r}", flush=True)
    json.dump({"what": "finiteT", "arm": arm, "backend": backend,
               "eos": eos.__file__, "stack": B.bench_stack(), "points": res},
              open(out, "w"))


def bench(repeats, out, arms):
    header(f"bench, default table, arms {arms} interleaved x {repeats}")
    par = njl.Parameters.named(B.BENCH_SET)
    axes = {"nB": B.BENCH_NB, "T": np.array([B.BENCH_T])}
    # the warm-up bench_table takes, discarded: the jit compile is not a
    # per-point cost
    njl.eos_table(par, B.BENCH_MODE, B.BENCH_SPECIES,
                  {"nB": B.BENCH_NB[:2], "T": np.array([B.BENCH_T])},
                  backend=B.BENCH_BACKEND)
    res = {arm: [] for arm in arms}
    n = len(B.BENCH_NB)
    for rep in range(repeats):
        for arm in arms:
            nodes = set_arm(arm)
            load0 = os.getloadavg()[0]
            w0, c0 = time.perf_counter(), time.process_time()
            result = njl.eos_table(par, B.BENCH_MODE, B.BENCH_SPECIES, axes,
                                   backend=B.BENCH_BACKEND,
                                   pair_nodes_per_panel=nodes)
            wall, cpu = time.perf_counter() - w0, time.process_time() - c0
            load1 = os.getloadavg()[0]
            rows = dump(njl.table.rows_from_result(result))
            res[arm].append({"wall": wall, "cpu": cpu, "load": [load0, load1],
                             "rows": rows})
            print(f"  rep {rep} {arm:8s} {wall * 1e3 / n:8.1f} ms/pt wall "
                  f"{cpu * 1e3 / n:8.1f} cpu  cpu/wall {cpu / wall:.2f}  "
                  f"rows {len(rows)}  loadavg {load0:.1f} -> {load1:.1f}",
                  flush=True)
    json.dump({"what": "bench", "grid": B.BENCH_NB.tolist(),
               "eos": eos.__file__, "stack": B.bench_stack(),
               "results": res}, open(out, "w"))
    base = res[arms[0]]
    for arm in arms:
        cpu = [r["cpu"] for r in res[arm]]
        ratio = [b["cpu"] / r["cpu"] for b, r in zip(base, res[arm])]
        print(f"  {arm:8s} median cpu {np.median(cpu) * 1e3 / n:8.1f} ms/pt  "
              f"x{np.median(ratio):.3f} vs {arms[0]} (per repeat "
              f"{', '.join(f'{x:.3f}' for x in ratio)})", flush=True)


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "table":
        table(*sys.argv[2:6])
    elif what == "gapless":
        gapless(*sys.argv[2:5])
    elif what == "finiteT":
        finite_t(*sys.argv[2:5])
    elif what == "bench":
        bench(int(sys.argv[2]), sys.argv[3], sys.argv[4:])
