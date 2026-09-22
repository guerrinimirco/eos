"""Ticket 08: is the landed njl acceleration mode-agnostic?

ARMS, both backend="fast", rg_njl1, SpeciesFlags(csc=True), default enumeration:

    base    d6d9e7c (the map's baseline; its default enumeration has 'free'),
            run from an isolated worktree with PYTHONPATH pointing at it
    landed  HEAD of the main repo

Each arm is ONE long-lived worker process, so numba compiles once per tree and
the warm-up is discarded once per process (every case is warmed on the grid's
first two densities, because numba compiles lazily per code path). The driver
feeds the two workers case by case, alternating arms, so both arms of a case
share one load window; only one worker solves at any moment.

    python3 t08.py drive REPEATS OUT.json         # spawns both workers
    python3 t08.py worker ARM TREE                # internal

Every worker prints its stack and asserts eos.__file__ lies in TREE.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

PY = "/Library/Frameworks/Python.framework/Versions/3.14/bin/python3"
REPO = "/Users/mircoguerrini/Desktop/Research/Python_codes/eos"
BASE = "/Users/mircoguerrini/Desktop/Research/Python_codes/eos-t08-base"
TREES = {"base": BASE, "landed": REPO}

NB = np.linspace(0.6, 1.5, 20)                  # ticket 13's grid
#: (label, mode, T, fixed, leptons)
CASES = [
    ("1 beta-eq T=0", "beta_eq_neutrinoless", 0.0, {}, None),
    ("2 beta-eq T=30", "beta_eq_neutrinoless", 30.0, {}, None),
    ("3 fixed_YC 0.4 lep T=0", "fixed_YC", 0.0, {"Y_C": 0.4}, True),
    ("4 fixed_YC 0.4 lep T=30", "fixed_YC", 30.0, {"Y_C": 0.4}, True),
    ("5 fixed_YC 0.4 nolep T=0", "fixed_YC", 0.0, {"Y_C": 0.4}, False),
    ("6 fixed_YC 0.4 nolep T=30", "fixed_YC", 30.0, {"Y_C": 0.4}, False),
    ("7 trapped Y_Le=0.4 T=30", "beta_eq_neutrino_trapped", 30.0,
     {"Y_Le": 0.4}, None),
]


def plain(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, str) or v is None:
        return v
    return float(v)


def run_case(njl, par, case, grid):
    _, mode, T, fixed, leptons = case
    result = njl.eos_table(par, mode, njl.SpeciesFlags(csc=True),
                           {"nB": grid, "T": np.array([T])}, fixed=fixed,
                           leptons=leptons, backend="fast")
    return [{k: plain(v) for k, v in row.items()}
            for row in njl.table.rows_from_result(result)]


def worker(arm, tree):
    sys.path = [p for p in sys.path if p not in ("", ".")]
    import scipy
    import eos
    from eos import njl
    from eos.general.pairing import DEFAULT_PATTERNS
    head = subprocess.run(["git", "-C", tree, "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    stack = (f"python {sys.version.split()[0]} ({sys.executable}), numpy "
             f"{np.__version__}, scipy {scipy.__version__}")
    assert os.path.realpath(eos.__file__).startswith(os.path.realpath(tree) + "/"), \
        f"{arm}: eos from {eos.__file__}, not {tree}"
    print(f"[{arm}] {stack}\n[{arm}] eos from {eos.__file__} at {head}; "
          f"DEFAULT_PATTERNS {DEFAULT_PATTERNS}", file=sys.stderr, flush=True)
    par = njl.Parameters.named("rg_njl1")
    w0 = time.perf_counter()
    for case in CASES:                          # discarded: numba compile
        try:
            run_case(njl, par, case, NB[:2])
        except Exception as exc:
            print(f"[{arm}] warm-up {case[0]} RAISED {exc!r:.120}",
                  file=sys.stderr, flush=True)
    print(json.dumps({"ready": arm, "stack": stack, "eos": eos.__file__,
                      "head": head, "patterns": list(DEFAULT_PATTERNS),
                      "warmup_s": time.perf_counter() - w0}), flush=True)
    for line in sys.stdin:
        i = int(line)
        load0 = os.getloadavg()
        w0, c0 = time.perf_counter(), time.process_time()
        try:
            rows, err = run_case(njl, par, CASES[i], NB), None
        except Exception as exc:
            rows, err = [], repr(exc)[:300]
        wall, cpu = time.perf_counter() - w0, time.process_time() - c0
        print(json.dumps({"case": i, "wall": wall, "cpu": cpu, "load": [
            load0[0], os.getloadavg()[0]], "error": err, "rows": rows}),
            flush=True)


def fingerprint(tree):
    return subprocess.run(
        "find eos -name '*.py' -type f | sort | xargs shasum | shasum",
        shell=True, cwd=tree, capture_output=True, text=True).stdout.split()[0]


def drive(repeats, out):
    here = os.path.abspath(__file__)
    procs = {}
    for arm, tree in TREES.items():
        env = dict(os.environ, PYTHONPATH=tree)
        procs[arm] = subprocess.Popen([PY, here, "worker", arm, tree], cwd=tree,
                                      env=env, stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, text=True)
    record = {"grid": NB.tolist(), "cases": [c[0] for c in CASES],
              "fingerprint_before": {a: fingerprint(t) for a, t in TREES.items()},
              "workers": {}, "runs": []}
    for arm, p in procs.items():
        record["workers"][arm] = json.loads(p.stdout.readline())
        print(f"{arm} ready, warm-up {record['workers'][arm]['warmup_s']:.0f} s",
              flush=True)
    t_start = time.perf_counter()
    for rep in range(repeats):
        for i, case in enumerate(CASES):
            for arm in ("base", "landed"):
                procs[arm].stdin.write(f"{i}\n")
                procs[arm].stdin.flush()
                r = json.loads(procs[arm].stdout.readline())
                r.update(rep=rep, arm=arm)
                record["runs"].append(r)
                print(f"rep {rep} {case[0]:<27} {arm:6s} cpu {r['cpu']:7.1f} s "
                      f"wall {r['wall']:7.1f} s  rows {len(r['rows']):2d}  load "
                      f"{r['load'][0]:.1f}->{r['load'][1]:.1f}"
                      + (f"  ERROR {r['error']}" if r["error"] else ""),
                      flush=True)
                json.dump(record, open(out, "w"))
    for p in procs.values():
        p.stdin.close()
        p.wait()
    record["fingerprint_after"] = {a: fingerprint(t) for a, t in TREES.items()}
    record["elapsed_s"] = time.perf_counter() - t_start
    json.dump(record, open(out, "w"))
    same = record["fingerprint_before"] == record["fingerprint_after"]
    print(f"fingerprints {'UNCHANGED' if same else 'CHANGED -- landed runs VOID'}"
          f" {record['fingerprint_after']}; elapsed {record['elapsed_s']:.0f} s",
          flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "worker":
        worker(sys.argv[2], sys.argv[3])
    else:
        drive(int(sys.argv[2]), sys.argv[3])
