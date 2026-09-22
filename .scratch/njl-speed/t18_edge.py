"""Ticket 18: where along the rotated charge Q~ does the CFL cross seed
unlock? State evaluations only: t -> gapless?, crossings, scaled residual.

    PYTHONPATH=. python3 .scratch/njl-speed/t18_edge.py
"""
import numpy as np

from eos import njl
from eos.general.pairing import gapless_breakpoints
from eos.njl.solver import (_state, _unpack, mode_spec, residual,
                            residual_scales, seed_from, solve_pattern,
                            unknown_slots)
from eos.njl.thermodynamics import vacuum_solution

par = njl.Parameters.named("rg_njl1")
flags = njl.SpeciesFlags(csc=True)
spec = mode_spec("fixed_YC", leptons=True, Y_C=0.1)
vac = vacuum_solution(par)
names = unknown_slots(par, spec, "CFL")
ix = {n: i for i, n in enumerate(names)}
for n_B in (0.775, 0.8, 0.9, 0.95, 1.0, 1.075):
    ref = solve_pattern(par, "fixed_YC", n_B, 0.0, flags, "unpaired",
                        spec=spec, backend="fast")
    seed = seed_from(ref, par, spec, "CFL")
    out = []
    for t in range(0, 401, 20):
        x = seed.copy()
        x[ix["mu_C"]] += t
        x[ix["mu_3"]] -= t
        x[ix["mu_8"]] -= 0.5 * t
        st = _state(x, par, spec, "CFL", 0.0, vac, False, "fast", None)
        r = np.array(residual(x, par, flags, spec, "CFL", n_B, 0.0, vac,
                              backend="fast"))
        s = np.array(residual_scales(par, spec, "CFL", n_B,
                                     abs(_unpack(x, par, spec, "CFL")[5]),
                                     T=0.0))
        out.append(f"{t}:{'G' if st.gapless else '-'}{np.max(np.abs(r / s)):.2f}"
                   f"/q{r[-1] / s[-1]:+.3f}")
    print(f"n_B {n_B}: gap seed {seed[ix['Delta_1']]:.0f}  " + " ".join(out),
          flush=True)
