"""PROTOTYPE (wayfinder ticket 13) -- throwaway.

The pinned benchmark is beta-eq at T = 0. The layout stop is a rule about the
ENUMERATION, not about that mode, so the risk it carries is a candidate that
passes THROUGH a collapsed layout on its way to its own root and would have
been rescued by the rungs the stop removes. The census saw `lm` rescue nothing
at T = 0; this asks the same question where the gaps are melting and where
leptons re-neutralize.

Four small sweeps, V0 against VX, gated on the realised state and on P.

    python3 .scratch/njl-speed/proto13_modes.py
"""
import json
import os
import sys
import time

import numpy as np

from eos import njl
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import proto13_variant as V          # the bounded ladder lives there

NB = np.linspace(0.6, 1.5, 20)
CASES = [
    ("beta_eq_neutrinoless", 0.0, {}),
    ("beta_eq_neutrinoless", 30.0, {}),
    ("fixed_YC", 0.0, {"Y_C": 0.4, "leptons": True}),
    ("fixed_YC", 30.0, {"Y_C": 0.4, "leptons": True}),
]


def sweep(par, mode, T, extra):
    axes = {"nB": NB, "T": np.array([T])}
    axes.update({k: np.array([v]) for k, v in extra.items()
                 if k.startswith("Y_")})
    kw = {k: v for k, v in extra.items() if not k.startswith("Y_")}
    t0 = time.perf_counter()
    result = njl.eos_table(par, mode, njl.SpeciesFlags(csc=True), axes,
                           backend="fast", **kw)
    rows = njl.table.rows_from_result(result)
    return (time.perf_counter() - t0,
            [{"n_B": float(r["n_B"]), "P": float(r["P"]),
              "realised": r["pattern_realised"]} for r in rows])


def main():
    par = njl.Parameters.named("rg_njl1")
    out = {}
    for arm in ("V0", "VX"):
        if arm == "VX":
            V.install("VX")
        for mode, T, extra in CASES:
            key = f"{mode}_T{T:g}" + ("".join(f"_{k}{v}" for k, v in extra.items()
                                              if k.startswith("Y_")))
            try:
                wall, rows = sweep(par, mode, T, extra)
            except Exception as exc:            # a mode njl refuses is a fact
                out[(arm, key)] = ("RAISED", repr(exc)[:120])
                print(f"{arm:3} {key:<38} RAISED {exc!r:.90}", flush=True)
                continue
            out[(arm, key)] = (wall, rows)
            print(f"{arm:3} {key:<38} {wall:6.1f}s  "
                  f"{len(rows)}/{len(NB)} rows solved", flush=True)

    print("\ngate: V0 vs VX")
    for _, key in [k for k in out if k[0] == "V0"]:
        a, b = out[("V0", key)], out[("VX", key)]
        if a[0] == "RAISED" or b[0] == "RAISED":
            print(f"  {key:<38} {'both raised' if a[0] == b[0] else 'DIFFER'}")
            continue
        ra = {round(r["n_B"], 6): r for r in a[1]}
        rb = {round(r["n_B"], 6): r for r in b[1]}
        bad = [k for k in set(ra) & set(rb) if ra[k]["realised"] != rb[k]["realised"]]
        conv = sorted(set(ra) ^ set(rb))   # rows one arm solved and the other dropped
        shared = set(ra) & set(rb)
        worst = max((abs(rb[k]["P"] - ra[k]["P"]) / max(abs(ra[k]["P"]), 1e-30),
                     k) for k in shared)
        print(f"  {key:<38} {a[0]:6.1f}s -> {b[0]:6.1f}s "
              f"({a[0] / b[0]:.2f}x)  state-mismatch {len(bad)}  "
              f"row-count diffs {len(conv)}  worst |dP|/P {worst[0]:.2e}  "
              f"{'PASS' if not bad and not conv and worst[0] <= 1e-8 else 'FAIL'}")
        if bad:
            print(f"     states differ at {sorted(bad)[:8]}")


if __name__ == "__main__":
    main()
