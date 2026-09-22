"""Ticket 10: the CFL candidate at the first gapless-CFL density of the
fixed_YC, Y_C = 0.1, leptons, T = 0 table, seeded the way `solve` seeds it
when it has no CFL seed of its own -- from the converged unpaired candidate --
with the ladder bounded (cross_seeded=True, what `solve` does since 50b3b7f)
and in full (cross_seeded=False), at the shipped rule and at 12 nodes.

    PYTHONPATH=. python3 t10_onset_probe.py
"""
import time

from eos import njl
from eos.njl.solver import mode_spec, seed_from, solve_pattern

par = njl.Parameters.named("rg_njl1")
flags = njl.SpeciesFlags(csc=True)
spec = mode_spec("fixed_YC", leptons=True, Y_C=0.1)
for n_B in (0.800, 0.900, 1.000):
    for nodes in (24, 12):
        ref = solve_pattern(par, "fixed_YC", n_B, 0.0, flags, "unpaired",
                            spec=spec, backend="fast",
                            pair_nodes_per_panel=nodes)
        seed = seed_from(ref, par, spec, "CFL")
        for bounded in (True, False):
            t0 = time.perf_counter()
            p = solve_pattern(par, "fixed_YC", n_B, 0.0, flags, "CFL",
                              spec=spec, x0=seed, backend="fast",
                              pair_nodes_per_panel=nodes,
                              cross_seeded=bounded)
            print(f"n_B {n_B:.3f} nodes {nodes:2d} "
                  f"{'bounded' if bounded else 'full   '}: converged "
                  f"{p.converged!s:5s} err {p.error:.1e} realised "
                  f"{p.pattern_realised:4s} gapless {p.gapless!s:5s} "
                  f"eps {p.eps:10.3f}  {time.perf_counter() - t0:5.1f} s",
                  flush=True)
