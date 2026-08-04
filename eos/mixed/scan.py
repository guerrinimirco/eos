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
               analytic_jac=False):
    """One (nmp, vmit) sample: invert, then look for a transition window.

    nmp   : dict with the `NMP_KEYS`
    vmit  : dict with any of the `VMIT_KEYS`; the rest take vMIT defaults
    flags : `SpeciesFlags` — note that hyperons need a parametrization carrying
            hyperon couplings, which `from_nmp` does not attach
    spec  : `ChargeSpec`; defaults to neutrino-transparent beta equilibrium,
            the cold-star condition

    Returns one flat dict row. `status` is 'ok' when both checks pass,
    'inversion_failed' when the NMPs are not representable, 'no_window' when
    they are but chi never completes its crossing, and 'error: ...' when the
    solve raised — which is itself a finding, so it is recorded, not raised.
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

    t0 = time.time()
    try:
        par, status = Parametrization.from_nmp(dict(nmp), return_status=True)
        row["inversion_ok"] = float(bool(status.ok))
        if not status.ok:
            row["status"] = "inversion_failed"
            return _timed(row, t0)

        window = locate_window(par, flags, np.asarray(n_B_grid, float),
                               float(eta),
                               spec if spec is not None else beta_eq_neutrinoless(),
                               vmit_params=params, T=float(T),
                               analytic_jac=analytic_jac)
        row["window_exists"] = float(bool(window.exists))
        row["n_onset"] = float(window.n_onset)
        row["n_offset"] = float(window.n_offset)
        row["status"] = "ok" if window.exists else "no_window"
    except Exception as exc:                    # a finding, not a crash
        row["status"] = f"error: {type(exc).__name__}: {exc}"[:200]
    return _timed(row, t0)


def _timed(row, t0):
    row["seconds"] = time.time() - t0
    return row


def scan_parameters(nmp_samples, vmit_samples, flags, n_B_grid, eta=0.0,
                    T=0.0, spec=None, n_jobs=1, progress=None):
    """Scan the product of `nmp_samples` and `vmit_samples`.

    Returns long-format rows — one per (nmp, vmit) pair — ready for
    `eos.general.table_io.save_table` / `export_csv`.

    n_jobs > 1 spreads the samples over processes with joblib, which is where
    the parallelism belongs: the samples are independent and each one is a long
    serial solve, so there is nothing to gain from threading inside a solve.
    Pass n_jobs=1 (the default) to keep it serial and debuggable.

    `progress` is an optional callable invoked with each finished row, so a
    notebook can print without this module importing anything to print with.
    """
    pairs = [(nmp, vmit) for nmp in nmp_samples for vmit in vmit_samples]

    def one(pair):
        return scan_point(pair[0], pair[1], flags, n_B_grid, eta=eta, T=T,
                          spec=spec)

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

    rows = scan_parameters([good, bad], grid_samples(B4=180.0), flags, grid)
    assert len(rows) == 2, rows
    assert rows[0]["status"] == "ok" and rows[0]["window_exists"] == 1.0, rows[0]
    assert rows[1]["status"] != "ok", rows[1]
    assert all(isinstance(r["seconds"], float) for r in rows)
    print(f"DD2 defaults : window [{rows[0]['n_onset']:.4f}, "
          f"{rows[0]['n_offset']:.4f}] fm^-3")
    print(f"K_sat=1e4    : {rows[1]['status']}")
    print("scan self-check OK")
