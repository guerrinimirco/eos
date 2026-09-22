"""Ticket 11: does `sweep`'s retry ladder ever RESCUE a walk step?

`walk_to_crossing` steps one `tol` at a time through
`sweep(phases, [r.n_B, n_next], ...)`, which carries `sweep`'s default
max_bisect=6 ladder. On a step that fails, the walk returns the midpoint of
(r, n_next) going down and nan going up -- whatever the ladder did on the way.
So the ladder can change the walk's answer ONLY by rescuing a step (the
target converges at retry depth >= 1). This records, for every walk step, the
direction, whether the target was reached, the deepest retry used, and the
solves spent, over the configurations `test/mixed/test_window_location.py`
pins (DD2 Y+Delta + vMIT, cheap), plus the DID+NJL and DID+CCDM windows of
`test_njl_pair.py` / `test_ccdm_pair.py` if asked for.

Nothing in `eos/` is edited.
"""
import os
import sys
import time

import numpy as np

import eos.mixed.boundaries as B
import eos.mixed.solver as S

STEPS = []          # one dict per walk step
_depth_seen = []    # retry depths of the solves inside the current walk step


def install():
    real_solve, real_sweep = S.solve, B.sweep

    def solve(*a, **kw):
        f = sys._getframe(1)
        depth = 0
        while f is not None:
            if f.f_code.co_name == "step" and "depth" in f.f_locals:
                depth = max(depth, int(f.f_locals["depth"]))
            f = f.f_back
        ok = False
        try:
            r = real_solve(*a, **kw)
            ok = True
            return r
        finally:
            if _depth_seen:
                _depth_seen[-1].append((depth, ok))

    def sweep(phases, grid, *a, **kw):
        caller = sys._getframe(1).f_code.co_name
        if caller != "walk_to_crossing":
            return real_sweep(phases, grid, *a, **kw)
        _depth_seen.append([])
        t0 = time.process_time()
        out = real_sweep(phases, grid, *a, **kw)
        seen = _depth_seen.pop()
        n_prev, n_next = float(grid[0]), float(grid[-1])
        reached = bool(out) and abs(out[-1].n_B - n_next) <= 1e-12
        STEPS.append({
            "dir": "down" if n_next < n_prev else "up", "n_next": n_next,
            "reached": reached, "max_depth": max(d for d, _ in seen),
            "solves": len(seen), "retry_solves": sum(d > 0 for d, _ in seen),
            "cpu": time.process_time() - t0})
        return out

    S.solve = solve
    B.sweep = sweep


def summary(tag):
    rescued = [s for s in STEPS if s["reached"] and s["max_depth"] > 0]
    failed = [s for s in STEPS if not s["reached"]]
    print(f"  [{tag}] walk steps {len(STEPS)}: "
          f"down {sum(s['dir'] == 'down' for s in STEPS)}, "
          f"up {sum(s['dir'] == 'up' for s in STEPS)}; "
          f"RESCUED by the ladder {len(rescued)} "
          f"({', '.join(s['dir'] + '@' + format(s['n_next'], '.4f') for s in rescued)}); "
          f"failed {len(failed)} spending "
          f"{sum(s['retry_solves'] for s in failed)} retry solves of "
          f"{sum(s['solves'] for s in STEPS)} walk solves, "
          f"cpu in failed steps {sum(s['cpu'] for s in failed):.1f} of "
          f"{sum(s['cpu'] for s in STEPS):.1f} s", flush=True)


def dd2_vmit_cases():
    """The configurations `test_window_location.py` exercises."""
    from eos.dd2 import (SpeciesFlags, from_delta_potential,
                         from_hyperon_potentials, from_nmp)
    from eos.mixed import beta_eq_neutrinoless
    from eos.mixed.adapters import default_pair
    from eos.vmit.parameters import Parameters as VMITParameters
    flags = SpeciesFlags(hyperons=True, deltas=True, muons=True, photons=True)
    grid = np.linspace(0.0149077, 1.788924, 300)
    par = from_delta_potential(U_Delta=-100.0, x_Delta_omega=1.2,
                               base=from_hyperon_potentials(
                                   U_Lambda=-30.0, U_Sigma=30.0, U_Xi=-18.0))
    pair = default_pair(par, flags, VMITParameters(B4=180.0, a=0.15,
                                                   m_s=150.0))
    spec = beta_eq_neutrinoless()
    for eta, T in [(0.3, 0.0), (0.3, 25.0), (0.6, 0.0), (1.0, 0.0),
                   (0.3, 15.0), (0.3, 20.0), (0.3, 30.0)]:
        yield f"dd2vmit eta={eta} T={T}", lambda eta=eta, T=T: \
            B.locate_window(pair, grid, eta, spec, T=T)
    for n_probe in (8, 12, 20):
        for n_lo in (0.03, 0.05, 0.08):
            g = np.linspace(n_lo, 1.788924, 300)
            yield f"dd2vmit probes={n_probe} n_lo={n_lo}", \
                lambda g=g, n_probe=n_probe: B.locate_window(
                    pair, g, 0.3, spec, T=0.0, n_probe=n_probe)
    small = np.linspace(0.05, 1.6, 80)
    vm = VMITParameters(B4=170.0, a=0.20, m_s=150.0)
    for U_L in (-25.0, -30.0):
        p = from_nmp(dict(n_sat=0.149077, E_sat=-16.02, m_eff_ratio=0.5625,
                          K_sat=290.0, Q_sat=300.0, E_sym=31.67, L_sym=50.0))
        p = from_hyperon_potentials(U_Lambda=U_L, U_Sigma=30.0, U_Xi=-10.0,
                                    base=p)
        p = from_delta_potential(U_Delta=-50.0, x_Delta_omega=1.20,
                                 x_Delta_rho=1.00, base=p)
        yield f"dd2vmit dropped-probe U_L={U_L}", \
            lambda p=p: B.locate_window(default_pair(p, flags, vm), small, 0.0,
                                        spec, T=0.0)


if __name__ == "__main__":
    install()
    print(sys.version.split()[0], flush=True)
    grand = []
    for tag, fn in dd2_vmit_cases():
        STEPS.clear()
        w = fn()
        print(f"{tag}: window {w.n_onset:.5f} -> {w.n_offset:.5f} "
              f"({w.reason})")
        summary(tag)
        grand += STEPS
    STEPS[:] = grand
    summary("ALL dd2+vmit")
