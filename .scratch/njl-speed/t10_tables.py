"""Ticket 10: the pinned single-pattern tables at several pairing node rules.

An ARM is "H/V": H Gauss-Legendre nodes per panel for the hot pass (the
`pair_nodes_per_panel` argument, threaded from `eos_table`), V for the two RG
vacuum passes. V is set by re-pointing the module globals `rg_pair_block` and
`rg_pair_jacobian` look the vacuum blocks up through, so the residual and its
Jacobian always see the SAME vacuum rule. "24/24" is the shipped rule.

Configuration is the pinned benchmark's (bench.py, extracted verbatim from
54c7be9): rg_njl1, beta_eq_neutrinoless, T = 0, csc=True, BENCH_NB, one
pattern held.

    PYTHONPATH=. python3 t10_tables.py time PATTERN BACKEND REPEATS OUT.json ARM...
    PYTHONPATH=. python3 t10_tables.py attr PATTERN BACKEND 1 OUT.json ARM...

`time` runs the arms INTERLEAVED, repeat by repeat, so every arm sees the same
load window and only ratios inside the run are quoted. `attr` runs each arm
once under ticket 03's self-time wrapper stack (wrappers cost a few percent,
so its clock is not the benchmark's).
"""
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                            # noqa: E402

import eos                                    # noqa: E402
import bench as B                             # noqa: E402
from eos import njl                           # noqa: E402
import eos.general.pairing as pairing         # noqa: E402
import eos.njl.thermodynamics as th           # noqa: E402
import eos.njl.backends.jacobian as jac       # noqa: E402
import eos.njl.table as table                 # noqa: E402

VAC_BLOCK = th._vacuum_pair_block             # the lru-cached originals
VAC_HESS = jac._vacuum_pair_hessian


def set_vacuum(V):
    """Point both vacuum lookups at rule V (None: whatever the caller asks)."""
    if V is None:
        th._vacuum_pair_block = VAC_BLOCK
        jac._vacuum_pair_hessian = VAC_HESS
        return
    th._vacuum_pair_block = (
        lambda Mb, Db, k_max, nodes, backend: VAC_BLOCK(Mb, Db, k_max, V,
                                                        backend))
    jac._vacuum_pair_hessian = (
        lambda Mb, Db, k_max, nodes: VAC_HESS(Mb, Db, k_max, V))


# --- per-solve clock: which densities converged, and what each cost -------
SOLVES = []
_solve_at = table.solve_at


CONVERGED_SELF = defaultdict(float)     # attribution over converged solves


def timed_solve_at(par, mode, n_B, *a, **k):
    before = dict(SELF)
    w0, c0 = time.perf_counter(), time.process_time()
    p = _solve_at(par, mode, n_B, *a, **k)
    wall = time.perf_counter() - w0
    ok = bool(p is not None and p.converged)
    SOLVES.append((float(n_B), ok, wall, time.process_time() - c0))
    if ok:
        CONVERGED_SELF["_total"] += wall
        for name, t in SELF.items():
            CONVERGED_SELF[name] += t - before.get(name, 0.0)
    return p


table.solve_at = timed_solve_at


# --- ticket 03's attribution: self time by a wrapper stack ------------------
STACK = []
SELF = defaultdict(float)
CALLS = defaultdict(int)
KERNELS = ("pair_pass", "hess_pass", "unpaired_ref", "unpaired_ref_hess",
           "panel_nodes", "gapless_momenta", "crossing_terms")


def wrap(mod, attr, name):
    f = getattr(mod, attr)

    def w(*a, **k):
        label = name
        if name in KERNELS and any(s[0].startswith("VAC") for s in STACK):
            label = name + "[vac]"
        t0 = time.perf_counter()
        STACK.append([label, 0.0])
        try:
            return f(*a, **k)
        finally:
            dt = time.perf_counter() - t0
            _, child = STACK.pop()
            SELF[label] += dt - child
            CALLS[label] += 1
            if STACK:
                STACK[-1][1] += dt
    setattr(mod, attr, w)


def instrument():
    wrap(pairing, "_pair_pass", "pair_pass")
    wrap(pairing, "_pair_hessian_pass", "hess_pass")
    wrap(pairing, "_unpaired_reference", "unpaired_ref")
    wrap(pairing, "_unpaired_reference_hessian", "unpaired_ref_hess")
    wrap(pairing, "gapless_momenta", "gapless_momenta")
    wrap(pairing, "_crossing_terms", "crossing_terms")
    wrap(pairing, "panel_nodes", "panel_nodes")
    wrap(jac, "panel_nodes", "panel_nodes")
    wrap(th, "modes_thermo", "modes_thermo")
    wrap(jac, "modes_jacobian", "modes_jacobian")


GROUPS = {
    "jitted": ("pair_pass", "pair_pass[vac]", "hess_pass", "hess_pass[vac]",
               "modes_thermo", "modes_jacobian"),
    "numpy": ("gapless_momenta", "unpaired_ref", "unpaired_ref[vac]",
              "unpaired_ref_hess", "unpaired_ref_hess[vac]", "panel_nodes",
              "panel_nodes[vac]", "crossing_terms"),
}


ATTRIBUTE = []


def run_table(pattern, backend, arm):
    H, V = (int(x) for x in arm.split("/"))
    set_vacuum(V)
    if ATTRIBUTE:
        # the vacuum lookups were just re-pointed, so mark them again: every
        # kernel called beneath them is the vacuum half's, not the hot pass's
        wrap(th, "_vacuum_pair_block", "VAC_block")
        wrap(jac, "_vacuum_pair_hessian", "VAC_hess")
    par = njl.Parameters.named(B.BENCH_SET)
    SOLVES.clear()
    w0, c0 = time.perf_counter(), time.process_time()
    result = njl.eos_table(par, B.BENCH_MODE, B.BENCH_SPECIES,
                           {"nB": B.BENCH_NB, "T": np.array([B.BENCH_T])},
                           patterns=(pattern,), backend=backend,
                           pair_nodes_per_panel=H)
    wall, cpu = time.perf_counter() - w0, time.process_time() - c0
    rows = njl.table.rows_from_result(result)
    out_rows = [{k: (float(r[k]) if isinstance(r[k], (float, np.floating))
                     else r[k])
                 for k in ("n_B", "P", "eps", "mu_B", "pattern_realised",
                           "gapless", "Delta_1", "Delta_2", "Delta_3",
                           "M_u", "M_d", "M_s")} for r in rows]
    for r in out_rows:
        r["gapless"] = bool(r["gapless"])
    conv = [s for s in SOLVES if s[1]]
    return {"wall": wall, "cpu": cpu, "rows": out_rows,
            "n_solve_at": len(SOLVES), "n_conv": len(conv),
            "conv_wall": sum(s[2] for s in conv),
            "conv_cpu": sum(s[3] for s in conv),
            "solves": list(SOLVES)}


def main():
    what, pattern, backend, repeats, out = sys.argv[1:6]
    arms = sys.argv[6:]
    repeats = int(repeats)
    print(f"=== ticket 10 {what}: {pattern}, backend {backend!r}, arms {arms}"
          f"\n{B.bench_stack()}\neos from {eos.__file__}\n"
          f"loadavg at start {os.getloadavg()}", flush=True)
    # warm-up, discarded: the jit compile is not a per-point cost
    set_vacuum(None)
    njl.eos_table(njl.Parameters.named(B.BENCH_SET), B.BENCH_MODE,
                  B.BENCH_SPECIES, {"nB": B.BENCH_NB[:2],
                                    "T": np.array([B.BENCH_T])},
                  patterns=(pattern,), backend=backend)
    res = {arm: [] for arm in arms}
    attr = {}
    if what == "attr":
        instrument()
        ATTRIBUTE.append(True)
    for rep in range(repeats):
        for arm in arms:
            if what == "attr":
                SELF.clear()
                CALLS.clear()
                CONVERGED_SELF.clear()
                VAC_BLOCK.cache_clear()
                VAC_HESS.cache_clear()
            r = run_table(pattern, backend, arm)
            res[arm].append(r)
            n = len(B.BENCH_NB)
            print(f"  rep {rep} arm {arm:6s} {r['wall'] * 1e3 / n:8.1f} ms/pt "
                  f"wall {r['cpu'] * 1e3 / n:8.1f} cpu  rows {len(r['rows'])}"
                  f"  converged solves {r['n_conv']}/{r['n_solve_at']} at "
                  f"{r['conv_wall'] * 1e3 / max(r['n_conv'], 1):7.1f} ms "
                  f"wall / {r['conv_cpu'] * 1e3 / max(r['n_conv'], 1):7.1f}"
                  f" cpu  load {os.getloadavg()[0]:.1f}", flush=True)
            if what == "attr":
                total = r["wall"]
                named = sum(SELF.values())
                a = {"total": total, "self": dict(SELF),
                     "calls": dict(CALLS),
                     "vac_block_cache": VAC_BLOCK.cache_info()._asdict(),
                     "vac_hess_cache": VAC_HESS.cache_info()._asdict()}
                for g, names in GROUPS.items():
                    a[g] = sum(SELF.get(x, 0.0) for x in names)
                a["python"] = total - a["jitted"] - a["numpy"]
                conv = dict(CONVERGED_SELF)
                a["converged"] = conv
                for g, names in GROUPS.items():
                    a["converged_" + g] = sum(conv.get(x, 0.0) for x in names)
                a["converged_python"] = (conv.get("_total", 0.0)
                                         - a["converged_jitted"]
                                         - a["converged_numpy"])
                attr[arm] = a
                print(f"    attribution: jitted {a['jitted'] / total:6.1%}  "
                      f"numpy {a['numpy'] / total:6.1%}  python "
                      f"{a['python'] / total:6.1%}   named {named / total:.1%}")
                for name in sorted(SELF, key=SELF.get, reverse=True):
                    print(f"      {name:24s} {SELF[name] / total:6.1%}  "
                          f"{SELF[name] * 1e3 / max(CALLS[name], 1):7.3f} "
                          f"ms/call  x{CALLS[name]}")
                print(f"      vacuum block cache {a['vac_block_cache']}\n"
                      f"      vacuum hessian cache {a['vac_hess_cache']}",
                      flush=True)
    with open(out, "w") as fh:
        json.dump({"what": what, "pattern": pattern, "backend": backend,
                   "stack": B.bench_stack(), "eos": eos.__file__,
                   "head": os.popen("git rev-parse --short HEAD").read().strip(),
                   "results": res, "attribution": attr}, fh)


if __name__ == "__main__":
    main()
