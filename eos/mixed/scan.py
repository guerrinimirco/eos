"""
mixed/scan.py
=============
Where in parameter space does a DD2 + vMIT hybrid equation of state exist?

*Public API* (re-exported from `eos.mixed`): `scan_parameters`, `scan_point`.

A hybrid star needs two things that are not guaranteed for an arbitrary set of
parameters, and this module answers both, in the order that fails cheapest:

1. **Is the hadronic model representable at all?** The nuclear-matter
   parameters {n_sat, E_sat, m*/m, K_sat, Q_sat, E_sym, L_sym} are inverted to
   DD2 couplings by `eos.dd2.invert_nmp`. Not every combination has a solution
   — Q_sat in particular can be inconsistent with the cross-constraint that
   closes the isoscalar system — and the inversion reports that rather than
   returning a silently wrong parametrization.

2. **Does a mixed phase exist?** The quark volume fraction chi comes out of the
   mixed solve and is not clamped, so the question is whether it crosses both 0
   and 1 on the density grid. `locate_window` finds those two crossings; when
   it reports no window the honest answer is that these parameters have no
   complete transition, not that the solver failed.

Whether a hybrid star exists is a physics question about (the NMPs, B^1/4, the
vector coupling a, m_s), not something a solver can decide. Both hyperons and
Delta isobars soften the hadronic branch, which moves the transition and can
remove it entirely, so a scan over the hadronic sector alone will find regions
with no window and that is the correct result.

Failures are recorded as rows, never raised: a scan that aborts on the first
pathological sample is useless for mapping a boundary, and the pathological
samples are exactly the boundary one is trying to map.

Units are fm-based on every boundary, as everywhere else in `eos`: densities
fm^-3, B^1/4 and m_s in MeV, the vector coupling a in fm^2.
"""
import time

import numpy as np

from eos.dd2.parametrization import Parametrization
from eos.vmit.parameters import get_vmit_custom
from eos.mixed.equilibrium.charges import beta_eq_neutrinoless
from eos.mixed.solvers.sweep import locate_window

#: The nuclear-matter parameters `eos.dd2.invert_nmp` consumes, in the order a
#: report reads best. Every key is required — the inversion has no defaults.
NMP_KEYS = ("n_sat", "E_sat", "m_eff_ratio", "K_sat", "Q_sat",
            "E_sym", "L_sym")

#: The vMIT parameters worth varying. The quark masses m_u and m_d are held at
#: their defaults: at these densities the equation of state is insensitive to
#: them, and varying them buys nothing but scan dimensions.
VMIT_KEYS = ("B4", "a", "m_s")


def scan_point(nmp, vmit, flags, n_B_grid, eta=0.0, T=0.0, spec=None,
               analytic_jac=False, tov=False, tov_parallel=True):
    """One (nmp, vmit) sample: invert, then look for a transition window.

    nmp   : dict with the `NMP_KEYS`
    vmit  : dict with any of the `VMIT_KEYS`; the rest take vMIT defaults
    flags : `SpeciesFlags` — note that hyperons need a parametrization carrying
            hyperon couplings, which `from_nmp` does not attach
    spec  : `ChargeSpec`; defaults to neutrino-transparent beta equilibrium,
            the cold-star condition

    tov   : also build the core equation of state and run the mass-radius
            sequence, adding M_max, R_Mmax, R_1p4 and cs2_max to the row. Only
            attempted when a window exists. Costs ~1-2 s per sample on top of
            the ~0.5 s the window search takes, which is affordable only
            because the TOV integration defaults to the Numba backend.
    tov_parallel : passed through to the TOV sequence. Set False when the scan
            itself is running under `n_jobs > 1`, so the process pool and the
            integrator's own threads do not oversubscribe the cores —
            `scan_parameters` does this for you.

    Returns one flat dict row. `status` is 'ok' when every requested check
    passes, 'inversion_failed' when the NMPs are not representable, 'no_window'
    when they are but chi never completes its crossing, 'tov_failed' when the
    star sequence did not integrate, and 'error: ...' when the solve raised —
    which is itself a finding, so it is recorded, not raised.
    """
    missing = [k for k in NMP_KEYS if k not in nmp]
    if missing:
        raise ValueError(f"nmp is missing {missing}; needs all of {NMP_KEYS}")

    row = {k: float(nmp[k]) for k in NMP_KEYS}
    vmit_full = {k: float(vmit[k]) for k in VMIT_KEYS if k in vmit}
    params = get_vmit_custom(**vmit_full)
    row.update(B4=params.B4, a=params.a, m_s=params.m_s, eta=float(eta),
               T=float(T))
    row.update(inversion_ok=0.0, window_exists=0.0,
               n_onset=np.nan, n_offset=np.nan, seconds=0.0, status="")
    if tov:
        row.update(M_max=np.nan, R_Mmax=np.nan, R_1p4=np.nan, cs2_max=np.nan)

    t0 = time.time()
    try:
        par, status = Parametrization.from_nmp(dict(nmp), return_status=True)
        row["inversion_ok"] = float(bool(status.ok))
        if not status.ok:
            row["status"] = "inversion_failed"
            return _timed(row, t0)

        grid = np.asarray(n_B_grid, float)
        cs = spec if spec is not None else beta_eq_neutrinoless()
        window = locate_window(par, flags, grid, float(eta), cs,
                               vmit_params=params, T=float(T),
                               analytic_jac=analytic_jac)
        row["window_exists"] = float(bool(window.exists))
        row["n_onset"] = float(window.n_onset)
        row["n_offset"] = float(window.n_offset)
        row["status"] = "ok" if window.exists else "no_window"

        if tov and window.exists:
            row.update(_tov_columns(par, flags, grid, eta, cs, params, T,
                                    window, tov_parallel))
            if not np.isfinite(row["M_max"]):
                row["status"] = "tov_failed"
    except Exception as exc:                    # a finding, not a crash
        row["status"] = f"error: {type(exc).__name__}: {exc}"[:200]
    return _timed(row, t0)


def _tov_columns(par, flags, grid, eta, spec, params, T, window, tov_parallel):
    """M_max, R(M_max), R(1.4) and max c_s^2 for one parameter sample.

    The window is reused rather than re-located — `scan_point` has already paid
    for it. A star sequence that does not integrate gives nan columns and a
    'tov_failed' status, never an exception.
    """
    from eos.mixed.tables.core_eos import build_mixed_eos_table, mass_radius_mixed
    from eos.mixed.coefficients import sound_speed_eq

    table = build_mixed_eos_table(par, flags, grid, float(eta), spec,
                                  vmit_params=params, T=float(T),
                                  window=window)
    cs2 = sound_speed_eq(table.P, table.eps)
    out = {"cs2_max": float(np.nanmax(cs2)) if cs2.size else np.nan}
    try:
        res = mass_radius_mixed(par, flags, grid, float(eta), spec,
                                vmit_params=params, T=float(T), table=table,
                                n_ec=120, tov_parallel=tov_parallel)
    except Exception:
        out.update(M_max=np.nan, R_Mmax=np.nan, R_1p4=np.nan)
        return out
    out.update(M_max=float(res["M_max"]), R_Mmax=float(res["R_Mmax"]),
               R_1p4=float(res["R_1p4"]))
    return out


def _timed(row, t0):
    row["seconds"] = time.time() - t0
    return row


def scan_parameters(nmp_samples, vmit_samples, flags, n_B_grid, eta=0.0,
                    T=0.0, spec=None, n_jobs=1, progress=None, tov=False):
    """Scan the product of `nmp_samples` and `vmit_samples`.

    Returns long-format rows — one per (nmp, vmit) pair — ready for
    `eos.general.table_io.save_table` / `export_csv`.

    tov=True adds M_max, R(M_max), R(1.4) and max c_s^2 per sample, so the scan
    answers not just "is there a transition" but "is the resulting star
    astrophysically viable" — which is what choosing parameters actually needs.

    n_jobs > 1 spreads the samples over processes with joblib, which is where
    the parallelism belongs: the samples are independent and each one is a long
    serial solve, so there is nothing to gain from threading inside a solve.
    The TOV integrator's own `prange` is switched off in that case so the two
    layers of parallelism do not oversubscribe the cores. Pass n_jobs=1 (the
    default) to keep it serial and debuggable.

    `progress` is an optional callable invoked with each finished row, so a
    notebook can print without this module importing anything to print with.
    """
    pairs = [(nmp, vmit) for nmp in nmp_samples for vmit in vmit_samples]

    def one(pair):
        return scan_point(pair[0], pair[1], flags, n_B_grid, eta=eta, T=T,
                          spec=spec, tov=tov, tov_parallel=(n_jobs == 1))

    if n_jobs == 1:
        rows = []
        for pair in pairs:
            row = one(pair)
            rows.append(row)
            if progress is not None:
                progress(row)
        return rows

    from joblib import Parallel, delayed
    rows = Parallel(n_jobs=n_jobs)(delayed(one)(p) for p in pairs)
    if progress is not None:
        for row in rows:
            progress(row)
    return list(rows)


def grid_samples(**axes):
    """Cartesian product of named axes as a list of dicts.

    Convenience for building `nmp_samples` / `vmit_samples`:

        grid_samples(B4=[150, 165, 180], a=[0.0, 0.2])

    gives six dicts. A scalar axis is treated as a one-element one, so the
    parameters being held fixed read the same way as the ones being varied.
    """
    from itertools import product
    keys = list(axes)
    grids = [np.atleast_1d(axes[k]) for k in keys]
    return [dict(zip(keys, (float(v) for v in combo)))
            for combo in product(*grids)]


if __name__ == "__main__":
    # Smallest runnable check: a representable sample finds a window, and an
    # unrepresentable one is reported rather than raising.
    from eos.dd2 import SpeciesFlags, compute_nmp, Parametrization as P

    flags = SpeciesFlags(hyperons=False, phi_field=False, muons=False)
    grid = np.linspace(0.05, 1.6, 120)
    good = compute_nmp(P.from_dd2_defaults())
    bad = dict(good, K_sat=1.0e4)               # not representable in DD-RMF

    rows = scan_parameters([good, bad], grid_samples(B4=180.0), flags, grid,
                           tov=True)
    assert len(rows) == 2, rows
    assert rows[0]["status"] == "ok" and rows[0]["window_exists"] == 1.0, rows[0]
    assert rows[1]["status"] != "ok", rows[1]
    assert all(isinstance(r["seconds"], float) for r in rows)
    assert np.isfinite(rows[0]["M_max"]) and rows[0]["M_max"] > 1.5, rows[0]
    assert np.isnan(rows[1]["M_max"]), rows[1]
    print(f"DD2 defaults : window [{rows[0]['n_onset']:.4f}, "
          f"{rows[0]['n_offset']:.4f}] fm^-3, M_max={rows[0]['M_max']:.3f} "
          f"Msun, R(1.4)={rows[0]['R_1p4']:.2f} km, "
          f"max c_s^2={rows[0]['cs2_max']:.3f}")
    print(f"K_sat=1e4    : {rows[1]['status']}")
    print("scan self-check OK")
