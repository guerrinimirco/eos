"""Ticket 18: the cross seed, the bounded stall and the gapless root side by
side, with every scaled residual row at the stall.

    PYTHONPATH=. python3 .scratch/njl-speed/t18_anatomy.py
"""
import numpy as np

from eos import njl
from eos.njl.solver import (mode_spec, residual, residual_scales, seed_from,
                            solve_pattern, unknown_slots, _unpack)
from eos.njl.thermodynamics import vacuum_solution

par = njl.Parameters.named("rg_njl1")
flags = njl.SpeciesFlags(csc=True)
spec = mode_spec("fixed_YC", leptons=True, Y_C=0.1)
vac = vacuum_solution(par)
names = unknown_slots(par, spec, "CFL")
np.set_printoptions(linewidth=200, precision=3, suppress=True)
print("      ", "  ".join(f"{n:>8s}" for n in names))
for n_B in (0.8, 1.0):
    ref = solve_pattern(par, "fixed_YC", n_B, 0.0, flags, "unpaired",
                        spec=spec, backend="fast")
    two = solve_pattern(par, "fixed_YC", n_B, 0.0, flags, "2SC", spec=spec,
                        backend="fast")
    seed = seed_from(ref, par, spec, "CFL")
    kw = dict(spec=spec, x0=seed, backend="fast")
    stall = solve_pattern(par, "fixed_YC", n_B, 0.0, flags, "CFL",
                          cross_seeded=True, **kw)
    root = solve_pattern(par, "fixed_YC", n_B, 0.0, flags, "CFL", **kw)
    print(f"n_B {n_B}: unpaired f {ref.f:.3f}, 2SC f {two.f:.3f} "
          f"(conv {two.converged}), gapless CFL f {root.f:.3f}")
    print(f"  unpaired mu_C {_unpack(ref.x, par, spec, 'unpaired')[6]:.3f}; "
          f"2SC x {np.asarray(two.x)}")
    for label, x in (("seed", seed), ("stall", np.asarray(stall.x)),
                     ("root", np.asarray(root.x))):
        print(f"  {label:5s}", "  ".join(f"{v:8.2f}" for v in x))
    x = np.asarray(stall.x)
    mu_B = _unpack(x, par, spec, "CFL")[5]
    r = np.array(residual(x, par, flags, spec, "CFL", n_B, 0.0, vac,
                          backend="fast"))
    s = np.array(residual_scales(par, spec, "CFL", n_B, abs(mu_B), T=0.0))
    print("  stall scaled rows", r / s)
