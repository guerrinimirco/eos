"""Ticket 18: is the lottery decided by an ill-conditioned FIRST step?

Captures `solve_system`'s own scaled rows and Jacobian closures for the
cross-seeded CFL candidate, then at the seed: the singular values of the
scaled Jacobian, and the first Newton step (lstsq, rcond as shipped, then the
25% clip) under the analytic Jacobian and jittered copies -- its
antisymmetric components (Delta_1 - Delta_2, mu_3) and the final outcome.

    PYTHONPATH=. python3 .scratch/njl-speed/t18_first_step.py
"""
import numpy as np

import eos.general.solve as G
import eos.njl.solver as S
from eos import njl
from eos.njl.solver import mode_spec, seed_from, solve_pattern, unknown_slots

par = njl.Parameters.named("rg_njl1")
flags = njl.SpeciesFlags(csc=True)
spec = mode_spec("fixed_YC", leptons=True, Y_C=0.1)
names = unknown_slots(par, spec, "CFL")
ix = {n: i for i, n in enumerate(names)}
np.set_printoptions(linewidth=200, precision=2)
CAP = {}
_newton = G.newton_solve


def capture(residual, jac, x0, scales_at, **kw):
    CAP.setdefault("closures", (residual, jac))
    return _newton(residual, jac, x0, scales_at, **kw)


for n_B in (0.8, 0.9, 1.0):
    ref = solve_pattern(par, "fixed_YC", n_B, 0.0, flags, "unpaired",
                        spec=spec, backend="fast")
    seed = seed_from(ref, par, spec, "CFL")
    CAP.clear()
    G.newton_solve = capture
    solve_pattern(par, "fixed_YC", n_B, 0.0, flags, "CFL", spec=spec, x0=seed,
                  backend="fast", cross_seeded=True)
    G.newton_solve = _newton
    rows, jac = CAP["closures"]
    r = np.asarray(rows(seed))
    J = np.asarray(jac(seed))
    sv = np.linalg.svd(J, compute_uv=False)
    print(f"\nn_B {n_B}: scaled rows at seed {r}")
    print(f"  singular values / max: {sv / sv[0]}")
    U, s, Vt = np.linalg.svd(J)
    weak = Vt[-1]
    print("  weakest right vector:", " ".join(
        f"{n}={v:+.2f}" for n, v in zip(names, weak) if abs(v) > 0.05))
    rng = np.random.default_rng(0)
    for k in range(6):
        Jk = J if k == 0 else J * (1 + 1e-6 * rng.standard_normal(J.shape))
        step = np.linalg.lstsq(Jk, -r, rcond=G.NEWTON_RCOND)[0]
        cap = G.NEWTON_STEP_FRACTION * max(np.max(np.abs(seed)), 1.0)
        damp = min(1.0, cap / np.max(np.abs(step)))
        print(f"  {'analytic' if k == 0 else f'jit{k}':8s} |step| "
              f"{np.max(np.abs(step)):9.3g}  clip {damp:.3g}  clipped "
              f"dDelta1-dDelta2 {damp * (step[ix['Delta_1']] - step[ix['Delta_2']]):+8.2f}"
              f"  dmu_3 {damp * step[ix['mu_3']]:+8.2f}  dmu_C "
              f"{damp * step[ix['mu_C']]:+8.2f}  dmu_B {damp * step[ix['mu_B']]:+8.2f}")
