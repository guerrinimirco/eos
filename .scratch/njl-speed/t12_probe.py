"""Ticket 12 facts: (a) baseline exposure at n_B=1.2 T=0, (b) does `free`
reach uSC/dSC at T>0, which is what it exists for (Gholami+ 2025 melting)."""
import sys, time, json
import numpy as np
from eos.njl import Parameters, SpeciesFlags, solve

par = Parameters.named("rg_njl1")
paired = SpeciesFlags(csc=True)
out = {}

# (a) the ONE baseline entry that uses the default enumeration
t0 = time.perf_counter()
four = solve(par, "beta_eq_neutrinoless", 1.2, 0.0, paired)
t_four = time.perf_counter() - t0
t0 = time.perf_counter()
three = solve(par, "beta_eq_neutrinoless", 1.2, 0.0, paired,
              patterns=("unpaired", "2SC", "CFL"))
t_three = time.perf_counter() - t0
out["baseline_n1.2"] = dict(
    four_pattern=four.pattern, four_realised=four.pattern_realised,
    three_pattern=three.pattern, three_realised=three.pattern_realised,
    f4=float(four.f), f3=float(three.f),
    rel_df=abs(float(four.f) - float(three.f)) / abs(float(four.f)),
    D4=[float(d) for d in four.Delta], D3=[float(d) for d in three.Delta],
    rel_dD=[abs(a - b) / max(abs(a), 1e-30)
            for a, b in zip(four.Delta, three.Delta)],
    P4=float(four.P), P3=float(three.P),
    rel_dP=abs(float(four.P) - float(three.P)) / abs(float(four.P)),
    s_four=t_four, s_three=t_three)
print(json.dumps(out["baseline_n1.2"], indent=1), flush=True)

# (b) what `free` REALISES at finite T, cold, per candidate
rows = []
for T in (0.0, 30.0, 50.0):
    for n_B in (0.8, 1.2, 1.6, 2.0):
        rec = {"T": T, "n_B": n_B, "cand": {}}
        for p in ("unpaired", "2SC", "CFL", "uSC", "dSC", "free"):
            t0 = time.perf_counter()
            try:
                q = solve(par, "beta_eq_neutrinoless", n_B, T, paired,
                          patterns=(p,))
                rec["cand"][p] = dict(conv=bool(q.converged),
                                      realised=q.pattern_realised,
                                      f=float(q.f),
                                      D=[float(d) for d in q.Delta],
                                      s=time.perf_counter() - t0)
            except Exception as exc:
                rec["cand"][p] = dict(conv=False, realised="ERROR",
                                      err=str(exc)[:120],
                                      s=time.perf_counter() - t0)
        rows.append(rec)
        print(json.dumps(rec), flush=True)
out["finiteT"] = rows
with open(sys.argv[1], "w") as fh:
    json.dump(out, fh, indent=1)
print("DONE", flush=True)
