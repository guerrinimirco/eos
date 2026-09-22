"""Ticket 19: which njl.npz keys the landed rule moves, and by how much.

Runs `generate_baseline.case_njl` on the tree (not saved over njl.npz), and
compares every key as `test_baseline.test_baseline` does (rtol = atol = 1e-10,
equal_nan), printing EVERY mismatch with its relative and absolute change and
the file's own `classify_drift` reading. The fresh store is kept beside this
script so the regeneration can be checked against it bit for bit.

    PYTHONPATH=.:test/baseline python3 .scratch/njl-speed/t19_baseline_diff.py OUT.npz
"""
import os
import sys

import numpy as np

import eos
import generate_baseline as G
import test_baseline as TB

out = sys.argv[1]
print(f"eos from {eos.__file__}\nHEAD {os.popen('git rev-parse --short HEAD').read().strip()}",
      flush=True)
stored = np.load(G.path_for("njl"))
fresh = G.CASES["njl"]()
np.savez_compressed(out, **fresh)
assert set(stored.files) == set(fresh), "key sets differ"
bad = []
for key in stored.files:
    want = stored[key]
    got = np.asarray(fresh[key], dtype=float)
    if not np.allclose(got, want, rtol=TB.RTOL, atol=TB.ATOL, equal_nan=True):
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = float(np.nanmax(np.abs((got - want) / np.where(want == 0, np.nan, want))))
        bad.append((key, rel, float(np.nanmax(np.abs(got - want))),
                    want.tolist(), got.tolist()))
moved = sum(1 for k in stored.files
            if not np.array_equal(np.asarray(fresh[k], dtype=float), stored[k], equal_nan=True))
print(f"{len(stored.files)} keys; {moved} not bit-identical; {len(bad)} outside "
      f"rtol = atol = 1e-10")
for key, rel, ab, want, got in sorted(bad, key=lambda b: -b[1]):
    print(f"  {key}: rel {rel:.2e}  abs {ab:.2e}   stored {want}  now {got}")
print("classify_drift:")
for line in TB.classify_drift(stored, fresh):
    print("  " + line)
