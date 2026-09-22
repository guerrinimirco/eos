"""Ticket 17's gate on the mixed rows: HEAD against `lm` declined, row by row.

Ticket 04's pinned pairing (DID + NJL rg_njl1, csc, backend 'fast', three
patterns, beta_eq_neutrinoless, T = 0, eta = 0), the same 7 onset rows and 7
deep rows ticket 04 counted, plus its one cold row -- solved with no counters,
and every quantity the mixed gate names written out: P, eps, chi, the winning
quark pattern, and each phase's (n_B, n_C, n_S, P) and potentials.

    T17_ARM=head|nolm python3 t17_gate.py out.json
    python3 t17_gate.py --compare a.json b.json
"""
import json
import os
import sys
import time

import numpy as np


def arm_patch():
    if os.environ.get("T17_ARM") != "nolm":
        return
    import eos.njl.thermodynamics as NT
    real = NT.solve_system

    def solve_system(*a, **kw):
        if sys._getframe(1).f_code.co_name == "thermo_from_mu":
            kw["methods"] = ("hybr",)
        return real(*a, **kw)
    NT.solve_system = solve_system


def rows():
    from eos import njl
    from eos.did.parameters import Parameters as DIDParameters
    from eos.did.species import SpeciesFlags as DIDFlags
    from eos.general.pairing import realised_pattern
    from eos.mixed.adapters import did_phase, njl_phase
    from eos.mixed.charges import beta_eq_neutrinoless
    import eos.mixed.solver as S

    phases = (did_phase(DIDParameters.default(), DIDFlags()),
              njl_phase(njl.Parameters.named("rg_njl1"),
                        njl.SpeciesFlags(csc=True),
                        patterns=("unpaired", "2SC", "CFL"),
                        backend=os.environ.get("T17_BACKEND", "fast")))
    grid = np.linspace(0.08, 1.60, 200)
    onset = grid[grid >= 0.855][:7]
    deep = grid[grid >= 1.000][:7]
    cs = beta_eq_neutrinoless()

    def record(tag, r):
        f = r.th_Q.fields
        return {"tag": tag, "n_B": r.n_B, "P": r.P, "eps": r.eps, "chi": r.chi,
                "pattern": realised_pattern((f["Delta_1"], f["Delta_2"],
                                             f["Delta_3"])),
                "potentials": {k: float(v) for k, v in r.potentials.items()},
                "H": [r.th_H.n_B, r.th_H.n_C, r.th_H.n_S, r.th_H.P],
                "Q": [r.th_Q.n_B, r.th_Q.n_C, r.th_Q.n_S, r.th_Q.P]}

    out = []
    t0, c0 = time.perf_counter(), time.process_time()
    out.append(record("cold", S.solve(phases, 1.000, 0.0, cs, T=0.0)))
    for tag, g in (("onset", onset), ("deep", deep)):
        for r in S.sweep(phases, g, 0.0, cs, T=0.0):
            out.append(record(tag, r))
    print(f"{len(out)} rows  wall {time.perf_counter() - t0:.1f} s  "
          f"cpu {time.process_time() - c0:.1f} s", flush=True)
    return out


def compare(a_path, b_path):
    a, b = json.load(open(a_path)), json.load(open(b_path))
    assert len(a) == len(b), (len(a), len(b))
    worst, bitwise = 0.0, 0
    for ra, rb in zip(a, b):
        assert ra["tag"] == rb["tag"] and ra["n_B"] == rb["n_B"]
        assert ra["pattern"] == rb["pattern"], (ra["n_B"], ra["pattern"],
                                                rb["pattern"])
        dP = abs(ra["P"] - rb["P"]) / abs(ra["P"])
        worst = max(worst, dP)
        same = ra == rb
        bitwise += same
        print(f"  {ra['tag']:5s} n_B {ra['n_B']:.6f} {ra['pattern']:8s} "
              f"chi {ra['chi']:+.6f}/{rb['chi']:+.6f}  |dP|/P {dP:.1e}  "
              f"{'bit-identical' if same else 'DIFFERS'}")
    print(f"{len(a)} rows, {bitwise} bit-identical in every field, "
          f"worst |dP|/P {worst:.2e}, same pattern everywhere")


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        arm_patch()
        import scipy
        import eos
        print(f"arm {os.environ.get('T17_ARM', 'head')}  backend "
              f"{os.environ.get('T17_BACKEND', 'fast')}  eos from "
              f"{os.path.dirname(eos.__file__)}  python "
              f"{sys.version.split()[0]} numpy {np.__version__} scipy "
              f"{scipy.__version__}", flush=True)
        json.dump(rows(), open(sys.argv[1], "w"), indent=1)
