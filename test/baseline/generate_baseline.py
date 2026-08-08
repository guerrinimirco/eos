"""Freeze the numerical output of every model, so a refactor can be proved
to be a no-op.

Each case below evaluates a model's main solver on a small fixed grid and
returns a flat dict of numbers. `main()` writes one `<model>.npz` per model;
`test_baseline.py` re-evaluates the same cases and demands agreement at
rtol = 1e-10. The tolerance is deliberately tight: it checks that the code
still computes exactly what it computed before, NOT that the physics is right
(the `verify/` suites and the golden values in test/dd2 do that).

The grids cover, per model, every mode that model implements, both leptons on
and leptons off where the mode has that flag, and points placed either side of
the thresholds each model has (muon onset, hyperon onset, the quark
transition). TOV sequences are included for dd2, vMIT and the mixed phase.

REGENERATING FROM SCRATCH
    test/ is gitignored, so these .npz files live only on this machine and are
    not recoverable from GitHub. If they are lost, check out the commit whose
    behaviour you want to freeze and run

        python -m pytest test/baseline --collect-only   # sanity: imports work
        python test/baseline/generate_baseline.py

    which rewrites every .npz from the code as it stands in the working tree.
    Regenerating against changed code silently re-blesses whatever it now
    computes, so only do it deliberately: on a known-good commit, or when a
    fix is *meant* to move numbers, in which case regenerate only the affected
    model and quote the before/after delta in the commit message.

    Single model:  python test/baseline/generate_baseline.py dd2 sfho

ENVIRONMENT
    A stale copy of `eos` in site-packages will shadow the working tree and
    silently baseline the wrong code. `_assert_working_tree()` refuses to run
    in that situation; the fix is `python -m pip install -e .` from the repo
    root (note `python -m pip`, not a bare `pip`, which may belong to a
    different interpreter).
"""
from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent


def _assert_working_tree():
    """Refuse to run against an installed copy of `eos` other than this one."""
    import eos
    found = pathlib.Path(eos.__file__).resolve()
    expected = REPO / "eos" / "__init__.py"
    if found != expected:
        raise RuntimeError(
            f"`import eos` resolves to {found}, not the working tree at "
            f"{expected}. A stale install is shadowing the repo; run "
            f"`python -m pip install -e .` from {REPO}."
        )


_assert_working_tree()


# --------------------------------------------------------------------------
# Turning a solver result into a flat dict of numbers
# --------------------------------------------------------------------------

def flatten(obj):
    """Collect every number reachable from `obj` into {name: value}.

    Walks dataclass fields and dicts recursively so that a nested block (the
    per-phase `PhaseThermo` of a mixed point, a `composition` dict, the
    `m_eff` map) is captured without naming its fields here. Strings and None
    are skipped: convergence *messages* are prose and drift harmlessly, while
    the `converged` flag itself is a bool and is kept as a number, because a
    refactor that quietly reclassifies a point as unconverged must fail.
    """
    out = {}

    def add(name, value):
        key = name.lstrip(".")
        # np.bool_ is deliberately listed: it subclasses neither `bool` nor
        # `np.number`, so leaving it out silently discards every `converged`
        # flag the solvers return.
        if isinstance(value, (bool, np.bool_, int, float, np.number)):
            out[key] = float(value)
        elif isinstance(value, np.ndarray):
            if value.dtype.kind in "fiub":
                out[key] = value.astype(float)
        elif isinstance(value, dict):
            for k, v in value.items():
                add(f"{name}.{k}", v)
        elif isinstance(value, (list, tuple)):
            if value and all(isinstance(v, (bool, np.bool_, int, float,
                                            np.number)) for v in value):
                out[key] = np.asarray(value, dtype=float)
        elif hasattr(value, "__dict__") and not isinstance(value, type):
            for k, v in vars(value).items():
                if not k.startswith("_"):
                    add(f"{name}.{k}", v)

    add("", obj)
    return out


def row(store, tag, obj):
    """Record every number of one solved point under a tag naming the case."""
    for key, value in flatten(obj).items():
        store[f"{tag}.{key}"] = value


# --------------------------------------------------------------------------
# The density grids: saturation, thresholds, and the high-density end
# --------------------------------------------------------------------------

N_HADRONIC = np.array([0.04, 0.08, 0.12, 0.16, 0.24, 0.32, 0.48, 0.64, 0.80])
N_QUARK = np.array([0.30, 0.45, 0.60, 0.80, 1.00, 1.30])
TEMPERATURES = (0.0, 10.0, 30.0)


# --------------------------------------------------------------------------
# dd2 — density-dependent RMF
# --------------------------------------------------------------------------

def case_dd2():
    from eos.dd2 import Parametrization, SpeciesFlags
    from eos.dd2.solver import solve_octet
    from eos.dd2.nmp import compute_nmp

    par = Parametrization.from_dd2_defaults()
    par_y = Parametrization.from_hyperon_potentials(
        U_Lambda=-30.0, U_Sigma=30.0, U_Xi=-18.0)
    par_yd = Parametrization.from_delta_potential(
        U_Delta=-50.0, x_wD=1.0, x_rD=1.0, base=par_y)

    nucleons = SpeciesFlags(hyperons=False, deltas=False, muons=True)
    hyperons = SpeciesFlags(hyperons=True, deltas=False, muons=True, phi_field=True)
    full = SpeciesFlags(hyperons=True, deltas=True, muons=True, phi_field=True)

    store = {}

    # Beta equilibrium, neutrino-transparent: the standard neutron-star mode.
    # The grid crosses the muon onset near 0.11 fm^-3 and, with hyperons on,
    # the Lambda onset near 2-3 n_sat.
    for T in TEMPERATURES:
        for n_B in N_HADRONIC:
            row(store, f"beta.nuc.T{T:g}.n{n_B:g}",
                solve_octet(par, n_B, nucleons, T=T))
    for n_B in N_HADRONIC[3:]:
        row(store, f"beta.hyp.T0.n{n_B:g}",
            solve_octet(par_y, n_B, hyperons, T=0.0))
        row(store, f"beta.hypdelta.T0.n{n_B:g}",
            solve_octet(par_yd, n_B, full, T=0.0))

    # Fixed charge fraction, with and without the neutralizing leptons. The
    # leptonless flavour is what a mixed-phase construction consumes.
    for Y_C in (0.1, 0.3, 0.5):
        for T in (0.0, 30.0):
            for n_B in (0.08, 0.16, 0.32, 0.64):
                row(store, f"yc.lep.YC{Y_C:g}.T{T:g}.n{n_B:g}",
                    solve_octet(par, n_B, nucleons, T=T, charge_mode="fixed",
                                Y_C=Y_C, yc_leptons=True))
                row(store, f"yc.nolep.YC{Y_C:g}.T{T:g}.n{n_B:g}",
                    solve_octet(par, n_B, nucleons, T=T, charge_mode="fixed",
                                Y_C=Y_C, yc_leptons=False))

    # Symmetric nuclear matter at fixed strangeness — the heavy-ion slice.
    for n_B in (0.08, 0.16, 0.32):
        row(store, f"ycys.snm.n{n_B:g}",
            solve_octet(par_y, n_B, hyperons, T=0.0, charge_mode="fixed",
                        Y_C=0.5, strange_mode="fixed", Y_S=0.0,
                        yc_leptons=False))

    # Trapped neutrinos at fixed lepton fraction.
    for Y_L in (0.2, 0.4):
        for n_B in (0.16, 0.32, 0.64):
            row(store, f"trapped.YL{Y_L:g}.n{n_B:g}",
                solve_octet(par, n_B, nucleons, T=10.0,
                            lepton_mode="trapped", Y_L=Y_L))

    # Nuclear-matter parameters: the forward map couplings -> NMPs.
    row(store, "nmp", compute_nmp(par))
    return store


# --------------------------------------------------------------------------
# sfho — nonlinear RMF
# --------------------------------------------------------------------------

def case_sfho():
    from eos.sfho.eos import (
        solve_sfho_beta_eq, solve_sfho_fixed_yc, solve_sfho_fixed_yc_ys,
        solve_sfho_trapped_neutrinos, solve_sfho_isentropic_beta_eq,
        BARYONS_N, BARYONS_NY,
    )
    from eos.sfho.parameters import get_sfho_nucleonic, get_sfhoy_fortin

    nuc = get_sfho_nucleonic()
    hyp = get_sfhoy_fortin()
    store = {}

    for T in TEMPERATURES:
        for n_B in N_HADRONIC:
            row(store, f"beta.nuc.T{T:g}.n{n_B:g}",
                solve_sfho_beta_eq(n_B, T, nuc, BARYONS_N, include_muons=True))
    for n_B in N_HADRONIC[3:]:
        row(store, f"beta.hyp.T0.n{n_B:g}",
            solve_sfho_beta_eq(n_B, 0.0, hyp, BARYONS_NY, include_muons=True))

    for Y_C in (0.1, 0.3, 0.5):
        for n_B in (0.08, 0.16, 0.32, 0.64):
            row(store, f"yc.lep.YC{Y_C:g}.n{n_B:g}",
                solve_sfho_fixed_yc(n_B, Y_C, 10.0, nuc, BARYONS_N,
                                    include_electrons=True))
            row(store, f"yc.nolep.YC{Y_C:g}.n{n_B:g}",
                solve_sfho_fixed_yc(n_B, Y_C, 10.0, nuc, BARYONS_N,
                                    include_electrons=False))

    for n_B in (0.16, 0.32, 0.64):
        row(store, f"ycys.n{n_B:g}",
            solve_sfho_fixed_yc_ys(n_B, 0.5, 0.0, 10.0, hyp, BARYONS_NY))
        row(store, f"trapped.n{n_B:g}",
            solve_sfho_trapped_neutrinos(n_B, 0.4, 10.0, nuc, BARYONS_N))
        row(store, f"isentropic.n{n_B:g}",
            solve_sfho_isentropic_beta_eq(n_B, 1.0, nuc, BARYONS_N))

    # The thermal pseudoscalar-meson gas. This branch carries a known bug in
    # the eta-meson energy density; the case is here so that fixing it shows
    # up as an explicit, quotable change rather than passing unnoticed.
    for n_B in (0.16, 0.32):
        row(store, f"mesons.n{n_B:g}",
            solve_sfho_beta_eq(n_B, 30.0, nuc, BARYONS_N,
                               include_pseudoscalar_mesons=True))
    return store


# --------------------------------------------------------------------------
# zl — Zhao-Lattimer nucleonic
# --------------------------------------------------------------------------

def case_zl():
    from eos.zl.eos import (
        solve_zl_beta_eq, solve_zl_fixed_yc, solve_zl_trapped_neutrinos)
    from eos.zl.parameters import get_zl_default

    par = get_zl_default()
    store = {}
    for T in TEMPERATURES:
        for n_B in N_HADRONIC:
            row(store, f"beta.T{T:g}.n{n_B:g}",
                solve_zl_beta_eq(n_B, T, params=par))
    for Y_C in (0.1, 0.3, 0.5):
        for n_B in (0.08, 0.16, 0.32, 0.64):
            row(store, f"yc.lep.YC{Y_C:g}.n{n_B:g}",
                solve_zl_fixed_yc(n_B, Y_C, 10.0, params=par,
                                  include_electrons=True))
            row(store, f"yc.nolep.YC{Y_C:g}.n{n_B:g}",
                solve_zl_fixed_yc(n_B, Y_C, 10.0, params=par,
                                  include_electrons=False))
    for Y_L in (0.2, 0.4):
        for n_B in (0.16, 0.32, 0.64):
            row(store, f"trapped.YL{Y_L:g}.n{n_B:g}",
                solve_zl_trapped_neutrinos(n_B, Y_L, 10.0, params=par))
    return store


# --------------------------------------------------------------------------
# vmit — vector-interaction MIT bag
# --------------------------------------------------------------------------

def case_vmit():
    from eos.vmit.eos import (
        solve_vmit_beta_eq, solve_vmit_fixed_yc, solve_vmit_fixed_yc_ys,
        solve_vmit_trapped_neutrinos)
    from eos.vmit.parameters import get_vmit_default

    par = get_vmit_default()
    store = {}
    for T in TEMPERATURES:
        for n_B in N_QUARK:
            row(store, f"beta.T{T:g}.n{n_B:g}",
                solve_vmit_beta_eq(n_B, T, params=par))
    for Y_C in (0.0, 0.3, 0.5):
        for n_B in (0.45, 0.80, 1.30):
            row(store, f"yc.lep.YC{Y_C:g}.n{n_B:g}",
                solve_vmit_fixed_yc(n_B, Y_C, 10.0, params=par,
                                    include_electrons=True))
            row(store, f"yc.nolep.YC{Y_C:g}.n{n_B:g}",
                solve_vmit_fixed_yc(n_B, Y_C, 10.0, params=par,
                                    include_electrons=False))
    for n_B in (0.45, 0.80, 1.30):
        row(store, f"ycys.n{n_B:g}",
            solve_vmit_fixed_yc_ys(n_B, 0.0, 1.0, 10.0, params=par))
        row(store, f"trapped.n{n_B:g}",
            solve_vmit_trapped_neutrinos(n_B, 0.4, 10.0, params=par))
    return store


# --------------------------------------------------------------------------
# alphabag — bag model with alpha_s corrections, plus the CFL branch
# --------------------------------------------------------------------------

def case_alphabag():
    from eos.alphabag.eos import (
        solve_alphabag_beta_eq, solve_alphabag_fixed_yc,
        solve_alphabag_fixed_yc_ys, solve_cfl)
    from eos.alphabag.parameters import get_alphabag_default

    par = get_alphabag_default()
    store = {}
    for T in TEMPERATURES:
        for n_B in N_QUARK:
            row(store, f"beta.T{T:g}.n{n_B:g}",
                solve_alphabag_beta_eq(n_B, T, params=par))
    for Y_C in (0.0, 0.3):
        for n_B in (0.45, 0.80, 1.30):
            row(store, f"yc.lep.YC{Y_C:g}.n{n_B:g}",
                solve_alphabag_fixed_yc(n_B, Y_C, 10.0, params=par,
                                        include_electrons=True))
            row(store, f"yc.nolep.YC{Y_C:g}.n{n_B:g}",
                solve_alphabag_fixed_yc(n_B, Y_C, 10.0, params=par,
                                        include_electrons=False))
    for n_B in (0.45, 0.80, 1.30):
        row(store, f"ycys.n{n_B:g}",
            solve_alphabag_fixed_yc_ys(n_B, 0.0, 1.0, 10.0, params=par))
        # Colour-flavour-locked branch, at two pairing gaps.
        for Delta0 in (50.0, 100.0):
            row(store, f"cfl.D{Delta0:g}.n{n_B:g}",
                solve_cfl(n_B, 0.0, Delta0, params=par))
    return store


# --------------------------------------------------------------------------
# abpr — analytic CFL parametrization at T = 0
# --------------------------------------------------------------------------

def case_abpr():
    from eos.abpr.eos import (
        get_abpr_default, pressure_abpr, baryon_density_abpr,
        energy_density_abpr, mu_from_nB_abpr, mu_from_P_abpr,
        mu_from_epsilon_abpr)

    par = get_abpr_default()
    store = {}
    mu_grid = np.array([1000.0, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0])
    store["mu_grid"] = mu_grid
    store["P"] = np.array([pressure_abpr(mu, par) for mu in mu_grid])
    store["n_B"] = np.array([baryon_density_abpr(mu, par) for mu in mu_grid])
    store["eps"] = np.array([energy_density_abpr(mu, par) for mu in mu_grid])

    # The three inversions, each on its own natural variable.
    for n_B in (0.4, 0.8, 1.2):
        row(store, f"mu_from_nB.n{n_B:g}", {"value": mu_from_nB_abpr(n_B, par)[0]})
    for P in (50.0, 200.0, 500.0):
        row(store, f"mu_from_P.P{P:g}", {"value": mu_from_P_abpr(P, par)[0]})
    for eps in (500.0, 1500.0):
        row(store, f"mu_from_eps.e{eps:g}",
            {"value": mu_from_epsilon_abpr(eps, par)[0]})
    return store


# --------------------------------------------------------------------------
# enjl — extended NJL, T = 0
# --------------------------------------------------------------------------

def case_enjl():
    """Both branches of the beta-equilibrium continuation.

    Points are warm-started from their neighbour: solved cold, this model
    stops converging around 0.5 fm^-3. "up" follows the chirally broken
    branch from low density, "down" walks back from a deconfined guess, and
    where both exist the difference between them IS the branch structure —
    so freezing only one would hide half the physics.
    """
    from eos.enjl.eos_beta import beta_eos_table
    from eos.enjl.parameters import get_enjl_default

    par = get_enjl_default()
    grid = np.linspace(0.10, 1.20, 34)
    store = {}
    for direction in ("up", "down"):
        points, P, eps = beta_eos_table(grid, par=par, direction=direction)
        store[f"{direction}.P"] = np.asarray(P, dtype=float)
        store[f"{direction}.eps"] = np.asarray(eps, dtype=float)
        for p in points:
            row(store, f"{direction}.n{p.n_b_fm:.6f}", p)
    return store


# --------------------------------------------------------------------------
# mixed — the eta-interpolated DD2 + vMIT hybrid engine
# --------------------------------------------------------------------------

def case_mixed():
    from eos.dd2 import Parametrization, SpeciesFlags
    from eos.mixed import (
        beta_eq_neutrinoless, fixed_YC, locate_window, sweep_mixed, solve_mixed)
    from eos.vmit.parameters import get_vmit_default

    par = Parametrization.from_dd2_defaults()
    flags = SpeciesFlags(hyperons=False, deltas=False, muons=True)
    vmit = get_vmit_default()
    grid = np.linspace(0.05, 1.5, 80)
    store = {}

    # eta interpolates Gibbs (0) to Maxwell (1). The window boundaries are
    # part of the result, not a by-product, so they are frozen too.
    for eta in (0.0, 0.5, 1.0):
        spec = beta_eq_neutrinoless()
        window = locate_window(par, flags, grid, eta, spec,
                               vmit_params=vmit, T=0.0)
        store[f"window.eta{eta:g}.exists"] = float(window.exists)
        store[f"window.eta{eta:g}.n_onset"] = float(window.n_onset)
        store[f"window.eta{eta:g}.n_offset"] = float(window.n_offset)

        inside = grid[(grid >= window.n_onset) & (grid <= window.n_offset)]
        for r in sweep_mixed(par, flags, inside[::3], eta, spec,
                             vmit_params=vmit, T=0.0):
            row(store, f"sweep.eta{eta:g}.n{r.n_B:.6f}", r)

    # A fixed-charge slice, leptons on and off.
    for leptons in (True, False):
        spec = fixed_YC(0.3, leptons=leptons)
        tag = "lep" if leptons else "nolep"
        for n_B in (0.4, 0.6, 0.8):
            row(store, f"yc.{tag}.n{n_B:g}",
                solve_mixed(par, flags, n_B, 0.0, spec,
                            vmit_params=vmit, T=0.0))
    return store


# --------------------------------------------------------------------------
# zlvmit — the first-generation ZL + vMIT hybrid, kept for published results
# --------------------------------------------------------------------------

def case_zlvmit():
    from eos.zlvmit.mixed_phase_eos import (
        solve_eta0_beta, solve_eta1_beta, solve_etaX_beta)
    from eos.zl.parameters import get_zl_default
    from eos.vmit.parameters import get_vmit_default

    zl, vmit = get_zl_default(), get_vmit_default()
    store = {}
    for n_B in (0.4, 0.6, 0.8):
        row(store, f"eta0.n{n_B:g}",
            solve_eta0_beta(n_B, 0.0, zl_params=zl, vmit_params=vmit))
        row(store, f"eta1.n{n_B:g}",
            solve_eta1_beta(n_B, 0.0, zl_params=zl, vmit_params=vmit))
        row(store, f"etaX.n{n_B:g}",
            solve_etaX_beta(n_B, 0.0, 0.5, zl_params=zl, vmit_params=vmit))
    return store


# --------------------------------------------------------------------------
# TOV — mass-radius sequences and the maximum mass
# --------------------------------------------------------------------------

def _tov_from_points(points, tag, store):
    """Run a TOV sequence on a (P, eps, n_B) table built from solved points.

    No crust is attached: a crust file lives outside the package and its
    absence would otherwise silently change M_max at the ~1% level. The
    crust-less sequence is the reproducible thing to freeze.
    """
    from dataclasses import fields

    from eos.tov.solver import compute_tov_sequence, EOSTable_for_TOV, TOVResult

    n_B = np.array([p[0] for p in points])
    eps = np.array([p[1] for p in points])
    P = np.array([p[2] for p in points])
    order = np.argsort(n_B)
    table = EOSTable_for_TOV(P=P[order], epsilon=eps[order], nB=n_B[order])

    e_c = np.geomspace(300.0, 0.98 * eps.max(), 16)
    seq = np.asarray(
        compute_tov_sequence(table, e_c, add_crust_table="No",
                             compute_tidal=True, verbose=False,
                             backend="scipy", tov_parallel=False),
        dtype=float)
    store[f"{tag}.sequence"] = seq

    # The sequence columns follow the TOVResult field order
    # (e_c, n_c, P_c, R, M, M_b, k2, Lambda); take the index from the
    # dataclass rather than hardcoding it, so a new column cannot silently
    # turn M_max into some other quantity.
    mass_col = [f.name for f in fields(TOVResult)].index("M")
    store[f"{tag}.M_max"] = float(np.nanmax(seq[:, mass_col]))


def case_tov():
    from eos.dd2 import Parametrization, SpeciesFlags
    from eos.dd2.solver import solve_octet
    from eos.vmit.eos import solve_vmit_beta_eq
    from eos.vmit.parameters import get_vmit_default
    from eos.mixed import beta_eq_neutrinoless
    from eos.mixed.tables.core_eos import build_mixed_eos_table

    store = {}

    par = Parametrization.from_dd2_defaults()
    flags = SpeciesFlags(hyperons=False, deltas=False, muons=True)
    grid = np.linspace(0.05, 1.3, 60)

    pts = []
    for n_B in grid:
        p = solve_octet(par, n_B, flags, T=0.0)
        pts.append((n_B, p.eps, p.P))
    _tov_from_points(pts, "dd2", store)

    vmit = get_vmit_default()
    pts = []
    for n_B in np.linspace(0.25, 1.5, 50):
        r = solve_vmit_beta_eq(n_B, 0.0, params=vmit)
        if r.converged:
            pts.append((n_B, r.e_total, r.P_total))
    _tov_from_points(pts, "vmit", store)

    # Hybrid star: the stitched hadronic + mixed + quark core table.
    tab = build_mixed_eos_table(par, flags, grid, 0.0, beta_eq_neutrinoless(),
                                vmit_params=vmit, T=0.0)
    pts = list(zip(np.asarray(tab.n_B, dtype=float),
                   np.asarray(tab.eps, dtype=float),
                   np.asarray(tab.P, dtype=float)))
    _tov_from_points(pts, "mixed", store)
    return store


# --------------------------------------------------------------------------

CASES = {
    "dd2": case_dd2,
    "sfho": case_sfho,
    "zl": case_zl,
    "vmit": case_vmit,
    "alphabag": case_alphabag,
    "abpr": case_abpr,
    "enjl": case_enjl,
    "mixed": case_mixed,
    "zlvmit": case_zlvmit,
    "tov": case_tov,
}


def path_for(name):
    return HERE / f"{name}.npz"


def main(names=None):
    names = names or list(CASES)
    total = 0.0
    for name in names:
        t0 = time.time()
        data = CASES[name]()
        np.savez_compressed(path_for(name), **data)
        dt = time.time() - t0
        total += dt
        print(f"{name:10s} {len(data):5d} values  {dt:6.2f}s")
    print(f"{'total':10s} {'':5s}          {total:6.2f}s")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
