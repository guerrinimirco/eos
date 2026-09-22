"""Ticket 10: would a coarser njl pairing default keep test/baseline green?

The four paired points `generate_baseline.case_njl` freezes, flattened by its
own `row`, on the reference backend (as frozen), at the shipped 24 nodes and
at a coarser rule; each key compared as test_baseline compares it (rtol = atol
= 1e-10), and against the stored njl.npz too so the 24-node arm is its own
control.

    PYTHONPATH=.:test/baseline:.scratch/njl-speed python3 t10_baseline_precheck.py
"""
import numpy as np

import generate_baseline as G
import t10_tables as T10
from eos.njl import Parameters, SpeciesFlags, solve

par = Parameters.default()
paired = SpeciesFlags(csc=True)
stored = np.load("test/baseline/njl.npz")
ARMS = ((None, None), (12, None), (10, None), (None, 12), (None, 10))
for nodes, vac in ARMS:
    T10.set_vacuum(vac)
    store = {}
    for pattern, n_B, T in (("unpaired", 1.4887, 0.0), ("2SC", 1.4887, 0.0),
                            ("CFL", 1.2, 0.0), ("2SC", 1.4, 20.0)):
        p = solve(par, "beta_eq_neutrinoless", n_B, T, paired,
                  patterns=(pattern,), pair_nodes_per_panel=nodes)
        tag = f"pattern.{pattern}.n{n_B:g}.T{T:g}"
        G.row(store, tag, p)
        store[f"{tag}.Delta"] = np.asarray(p.Delta, dtype=float)
    fails = []
    for key, got in store.items():
        want = stored[key]
        if not np.allclose(got, want, rtol=1e-10, atol=1e-10, equal_nan=True):
            rel = float(np.max(np.abs(np.asarray(got) - want)
                               / np.maximum(np.abs(want), 1e-300)))
            fails.append((rel, key))
    fails.sort(reverse=True)
    print(f"hot {nodes or 24} / vacuum {vac or nodes or 24}: {len(fails)} of {len(store)} keys outside "
          f"rtol=atol=1e-10 against njl.npz" + "".join(
              f"\n    {k}: {r:.1e}" for r, k in fails[:60]), flush=True)
