"""
notebook_api.py
====================
Plotting / analysis layer for the DD2 engine, driven by ``notebooks/DD2_usage.ipynb``.

The notebook is deliberately thin: every non-trivial routine lives here as a
function that returns a Matplotlib ``Figure`` (so it is testable and re-runnable).
Each function takes a ``Parametrization`` as its first argument, so the whole set
can be regenerated for a second parametrization (e.g. one built from NMPs) just by
passing a different ``par``.

Conventions (report §3.x): every figure is built with ``fig, ax = plt.subplots()``
and drawn through ``ax.*`` — never bare ``plt.*``. Cold/T=0 unless the plot is
about temperature. Nucleonic NS-structure plots use
``SpeciesFlags(hyperons=False, phi_field=False)``; composition plots use the full
octet where the point is the hyperon onsets.
"""
import os
import time

import numpy as np
import matplotlib.pyplot as plt

from eos import REPO_ROOT
from eos.dd2 import (
    Parametrization, SpeciesFlags, compute_nmp,
    solve_beta_eq_octet, sweep_beta_eq_octet, solve_composition, solve_snm,
    solve_octet_at_entropy, octet_warm_start,
    sound_speed_eq, sound_speed_adiabatic, heat_capacity_V,
)
from eos.dd2.verify.tov import build_core_table, N_TRANSITION
from eos.tov.solver import (
    compute_tov_sequence, find_mmax_precise, generate_ec_logspace, CRUST_PATHS,
)

#: Nucleonic degrees of freedom (npe-mu): the NS-structure default.
NUCLEONIC = SpeciesFlags(hyperons=False, phi_field=False)
#: Full baryon octet with the hidden-strange vector (composition plots).
OCTET = SpeciesFlags(hyperons=True, phi_field=True)

#: Shipped observational data (figs 8, 10).
_DATA = os.path.join(str(REPO_ROOT), "plot", "data")
CONTOUR_DIR = os.path.join(_DATA, "contours")
CHIRAL_EFT = os.path.join(_DATA, "samples", "chiral_eft.txt")

#: DD2 reference NMPs (Typel et al. 2010), for the fig-11 comparison table.
DD2_NMP_REFERENCE = {
    "n_sat": 0.149065, "E_sat": -16.02, "K_sat": 242.7, "Q_sat": 169.0,
    "E_sym": 31.67, "L_sym": 55.03, "m_eff_ratio": 0.5625,
}
_NMP_LABELS = {
    "n_sat": "n_sat [fm^-3]", "E_sat": "E_sat [MeV]", "K_sat": "K_sat [MeV]",
    "Q_sat": "Q_sat [MeV]", "E_sym": "E_sym [MeV]", "L_sym": "L_sym [MeV]",
    "m_eff_ratio": "m*/m",
}


def default_grid(n_lo=0.06, n_hi=1.2, n=60):
    """Geometric β-eq density grid [fm^-3] (log-spaced: the stiff axis)."""
    return np.geomspace(n_lo, n_hi, n)


# =============================================================================
# 1. P vs n_B — β-equilibrium
# =============================================================================
def plot_p_vs_nb(par, flags=NUCLEONIC, grid=None, T=0.0):
    """P(n_B) along β-equilibrium (``sweep_beta_eq_octet``), log-y."""
    grid = default_grid() if grid is None else np.asarray(grid, float)
    pts = sweep_beta_eq_octet(par, grid, flags, T=T, include_photons=False)
    nB = np.array([p.n_B for p in pts])
    P = np.array([p.P for p in pts])
    fig, ax = plt.subplots()
    ax.semilogy(nB, P, "-", lw=2)
    ax.set_xlabel(r"$n_B$ [fm$^{-3}$]")
    ax.set_ylabel(r"$P$ [MeV fm$^{-3}$]")
    ax.set_title(r"DD2 $\beta$-equilibrium pressure")
    ax.grid(True, which="both", alpha=0.3)
    return fig


# =============================================================================
# 2. Composition Y_i vs n_B
# =============================================================================
def plot_composition(par, flags=NUCLEONIC, grid=None, T=0.0, y_floor=1e-4):
    """
    Particle fractions Y_i = n_i/n_B along β-equilibrium. Shows every baryon
    that rises above ``y_floor`` plus the leptons (so hyperon onsets are visible
    with the octet on, nucleons+leptons with it off). Pass ``flags=OCTET`` with a
    DD2Y parametrization (``Parametrization.from_dd2y_defaults()``) to see the
    hyperon onsets — the plain DD2 par carries no hyperon couplings.
    """
    grid = default_grid() if grid is None else np.asarray(grid, float)
    pts = sweep_beta_eq_octet(par, grid, flags, T=T, include_photons=False)
    nB = np.array([p.n_B for p in pts])

    # collect all baryon species that ever appear, plus leptons
    series = {}
    for name in {n for p in pts for n, _ in p.composition}:
        y = np.array([p.Y(name) for p in pts])
        if y.max() > y_floor:
            series[name] = y
    for lep, attr in (("e", "Y_e"), ("mu", "Y_mu")):
        y = np.array([getattr(p, attr) for p in pts])
        if y.max() > y_floor:
            series[lep] = y

    fig, ax = plt.subplots()
    for name, y in sorted(series.items()):
        ax.plot(nB, np.clip(y, y_floor, None), label=name, lw=1.8)
    ax.set_yscale("log")
    ax.set_ylim(y_floor, 1.5)
    ax.set_xlabel(r"$n_B$ [fm$^{-3}$]")
    ax.set_ylabel(r"$Y_i = n_i / n_B$")
    ax.set_title("DD2 β-equilibrium composition")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    return fig


# =============================================================================
# 3. Isentropic T vs n_B
# =============================================================================
def plot_isentropic_T(par, flags=NUCLEONIC, S_values=(1.0, 2.0, 4.0), grid=None):
    """Temperature along constant entropy-per-baryon S=s/n_B paths (§3.6)."""
    grid = default_grid(n=30) if grid is None else np.asarray(grid, float)
    has_phi = flags.phi_field and flags.hyperons
    fig, ax = plt.subplots()
    for S in S_values:
        Ts, x0 = [], None
        for n in grid:
            p = solve_octet_at_entropy(par, float(n), float(S), flags, x0=x0)
            Ts.append(p.T)
            x0 = octet_warm_start(p, has_phi)
        ax.plot(grid, Ts, "-", lw=1.8, label=f"S = {S:g}")
    ax.set_xlabel(r"$n_B$ [fm$^{-3}$]")
    ax.set_ylabel(r"$T$ [MeV]")
    ax.set_title("DD2 isentropic temperature")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


# =============================================================================
# 4. c_s^2 vs n_B — adiabatic (frozen) and equilibrium
# =============================================================================
def plot_sound_speed(par, flags=NUCLEONIC, grid=None, T=0.0):
    """
    Frozen (``sound_speed_adiabatic``, at the local β-eq composition) and
    equilibrium (``sound_speed_eq``) c_s^2 along β-equilibrium, with the causal
    limit c_s^2 = 1 marked.
    """
    grid = default_grid(n=40) if grid is None else np.asarray(grid, float)
    nB, cad, ceq = [], [], []
    for n in grid:
        Y_p = solve_beta_eq_octet(par, float(n), flags, T=T,
                                  include_photons=False).Y_p
        nB.append(n)
        cad.append(sound_speed_adiabatic(par, float(n), Y_p, T=T))
        ceq.append(sound_speed_eq(par, float(n), flags, T=T))
    fig, ax = plt.subplots()
    ax.plot(nB, cad, "-", lw=1.8, label=r"frozen $c_{s,\mathrm{ad}}^2$")
    ax.plot(nB, ceq, "--", lw=1.8, label=r"equilibrium $c_{s,\mathrm{eq}}^2$")
    ax.axhline(1.0, color="k", ls=":", lw=1, label="causal limit")
    ax.set_xlabel(r"$n_B$ [fm$^{-3}$]")
    ax.set_ylabel(r"$c_s^2 / c^2$")
    ax.set_title("DD2 speed of sound")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


# =============================================================================
# 5. C_V and C_P vs n_B at fixed T > 0
# =============================================================================
def _heat_capacity_P(par, n_B, flags, T, cv_vol, rel_dn=1e-3, dT=1e-2):
    """
    Per-baryon C_P from the Mayer relation at the frozen (local β-eq)
    composition — coefficients.py ships only C_V, so C_P is assembled here:

        c_P - c_V = T (dP/dT)_n^2 / (n^2 (dP/dn)_T)   (per baryon, fixed Y_p)

    ``cv_vol`` is the volumetric heat_capacity_V; both are returned per baryon.
    # ponytail: frozen-composition Mayer relation; swap for an analytic C_P if
    # coefficients_jac ever grows one.
    """
    Y_p = solve_beta_eq_octet(par, n_B, flags, T=T, include_photons=False).Y_p
    P = lambda n, t: solve_composition(par, (1 - Y_p) * n, Y_p * n, T=t).P
    dP_dT = (P(n_B, T + dT) - P(n_B, T - dT)) / (2 * dT)
    dP_dn = (P(n_B * (1 + rel_dn), T) - P(n_B * (1 - rel_dn), T)) \
        / (2 * rel_dn * n_B)
    cv = cv_vol / n_B
    cp = cv + T * dP_dT ** 2 / (n_B ** 2 * dP_dn)
    return cv, cp


def plot_heat_capacity(par, flags=NUCLEONIC, grid=None, T=10.0):
    """Per-baryon C_V (``heat_capacity_V``) and C_P vs n_B at fixed T>0."""
    grid = default_grid(n=30) if grid is None else np.asarray(grid, float)
    nB, cv, cp = [], [], []
    for n in grid:
        cvv = heat_capacity_V(par, float(n), flags, T=T)
        cV, cP = _heat_capacity_P(par, float(n), flags, T, cvv)
        nB.append(n); cv.append(cV); cp.append(cP)
    fig, ax = plt.subplots()
    ax.plot(nB, cv, "-", lw=1.8, label=r"$C_V / n_B$")
    ax.plot(nB, cp, "--", lw=1.8, label=r"$C_P / n_B$")
    ax.set_xlabel(r"$n_B$ [fm$^{-3}$]")
    ax.set_ylabel(r"heat capacity per baryon [$k_B$]")
    ax.set_title(f"DD2 heat capacities at T = {T:g} MeV")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


# =============================================================================
# 6-8. TOV: M-R, Λ-M, constraints
# =============================================================================
def compute_tov(par, flags=NUCLEONIC, e_c_min=150.0, e_c_max=3000.0, n_ec=180):
    """
    Cold β-eq core + BPS crust TOV sequence with tidal deformability. Returns a
    dict with the stable branch (M, R, Lambda), M_max, R_1.4, Lambda_1.4, and
    the raw ``results`` array. Shared by figs 6-8 so the notebook solves once
    per parametrization.
    """
    core = build_core_table(par, flags)
    crust = "BPS" if os.path.isfile(CRUST_PATHS.get("BPS", "")) else "No"
    e_c_vec = generate_ec_logspace(e_c_min, e_c_max, n_ec)
    res = compute_tov_sequence(
        core, e_c_vec, add_crust_table=crust, add_crust_mode="attach",
        n_transition=(N_TRANSITION if crust != "No" else None),
        compute_baryonic_mass=False, compute_tidal=True,
        verbose=False, backend="scipy",
    )
    idx, _, M_max = find_mmax_precise(res)
    R = res[:idx + 1, 3]
    M = res[:idx + 1, 4]
    Lam = res[:idx + 1, -1]           # tidal Λ is the last column
    stable = M[-1] >= 1.4 > M[0]
    R_1p4 = float(np.interp(1.4, M, R)) if stable else float("nan")
    L_1p4 = float(np.interp(1.4, M, Lam)) if stable else float("nan")
    return dict(results=res, idx_max=idx, M=M, R=R, Lambda=Lam,
                M_max=float(M_max), R_Mmax=float(res[idx, 3]),
                R_1p4=R_1p4, Lambda_1p4=L_1p4, crust=crust)


def plot_mass_radius(par, flags=NUCLEONIC, tov=None):
    """M-R curve, annotating M_max and R_1.4."""
    tov = compute_tov(par, flags) if tov is None else tov
    fig, ax = plt.subplots()
    ax.plot(tov["R"], tov["M"], "-", lw=2)
    ax.plot(tov["R_Mmax"], tov["M_max"], "o", color="C3")
    ax.annotate(f"$M_{{max}}$ = {tov['M_max']:.2f} $M_\\odot$",
                (tov["R_Mmax"], tov["M_max"]),
                textcoords="offset points", xytext=(8, -4), fontsize=9)
    if np.isfinite(tov["R_1p4"]):
        ax.plot(tov["R_1p4"], 1.4, "s", color="C2")
        ax.annotate(f"$R_{{1.4}}$ = {tov['R_1p4']:.2f} km",
                    (tov["R_1p4"], 1.4),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)
    ax.set_xlabel(r"$R$ [km]")
    ax.set_ylabel(r"$M$ [$M_\odot$]")
    ax.set_title("DD2 mass-radius")
    ax.grid(True, alpha=0.3)
    return fig


def plot_lambda_mass(par, flags=NUCLEONIC, tov=None):
    """Tidal deformability Λ vs mass along the stable branch."""
    tov = compute_tov(par, flags) if tov is None else tov
    fig, ax = plt.subplots()
    ax.semilogy(tov["M"], tov["Lambda"], "-", lw=2)
    if np.isfinite(tov["Lambda_1p4"]):
        ax.plot(1.4, tov["Lambda_1p4"], "s", color="C2")
        ax.annotate(f"$\\Lambda_{{1.4}}$ = {tov['Lambda_1p4']:.0f}",
                    (1.4, tov["Lambda_1p4"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)
    ax.set_xlabel(r"$M$ [$M_\odot$]")
    ax.set_ylabel(r"$\Lambda$")
    ax.set_title("DD2 tidal deformability")
    ax.grid(True, which="both", alpha=0.3)
    return fig


def plot_mr_with_constraints(par, flags=NUCLEONIC, tov=None,
                             contour_dir=CONTOUR_DIR):
    """
    M-R curve over the observational constraints. ``add_observational_constraints``
    lives in the user's figure utilities (imported lazily below so this module
    loads even before that package is on the path); it reads the contour CSVs in
    ``contour_dir`` (J0030, J0740, HESS, GW170817, GW190425) and silently skips
    any that are missing.
    """
    from nucleation.analysis.figure import add_observational_constraints
    tov = compute_tov(par, flags) if tov is None else tov
    fig, ax = plt.subplots()
    add_observational_constraints(ax, contour_dir)
    ax.plot(tov["R"], tov["M"], "-", lw=2, color="k", label="DD2")
    ax.set_xlim(9, 16)
    ax.set_ylim(0, 2.6)
    ax.set_xlabel(r"$R$ [km]")
    ax.set_ylabel(r"$M$ [$M_\odot$]")
    ax.set_title("DD2 vs observations")
    ax.legend(loc="lower left", fontsize=8)
    return fig


# =============================================================================
# 9. P vs n_B — symmetric nuclear matter
# =============================================================================
def plot_p_vs_nb_snm(par, grid=None, T=0.0):
    """P(n_B) of symmetric nuclear matter (Y_p=0.5), log-y."""
    grid = default_grid() if grid is None else np.asarray(grid, float)
    P = np.array([solve_snm(par, float(n), T=T).P for n in grid])
    # SNM pressure is negative below saturation; log-y needs the positive branch.
    pos = P > 0
    fig, ax = plt.subplots()
    ax.semilogy(grid[pos], P[pos], "-", lw=2)
    ax.set_xlabel(r"$n_B$ [fm$^{-3}$]")
    ax.set_ylabel(r"$P$ [MeV fm$^{-3}$]")
    ax.set_title("DD2 symmetric nuclear matter pressure")
    ax.grid(True, which="both", alpha=0.3)
    return fig


# =============================================================================
# 10. Chiral-EFT comparison in pure neutron matter
# =============================================================================
def plot_pnm_chiral(par, grid=None):
    """
    Pure neutron matter energy per particle E/N vs the chiral-EFT band shipped
    in ``chiral_eft.txt``. E/N = eps/n_B - m_N (kernel average mass).
    """
    band = np.loadtxt(CHIRAL_EFT)          # rho, E_low, E_up
    rho, e_lo, e_hi = band[:, 0], band[:, 1], band[:, 2]
    grid = rho if grid is None else np.asarray(grid, float)
    EN = np.array([solve_composition(par, float(n), 0.0).eps / n - par.m_nucleon
                   for n in grid])
    fig, ax = plt.subplots()
    ax.fill_between(rho, e_lo, e_hi, alpha=0.3, color="C0",
                    label="chiral EFT")
    ax.plot(grid, EN, "-", lw=2, color="C3", label="DD2 PNM")
    ax.set_xlabel(r"$n_B$ [fm$^{-3}$]")
    ax.set_ylabel(r"$E/N$ [MeV]")
    ax.set_title("DD2 pure neutron matter vs chiral EFT")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


# =============================================================================
# 11. Compute NMPs
# =============================================================================
def nmp_comparison(par):
    """
    NMPs of ``par`` next to the DD2 reference (report §2.1). Returns a dict
    ``{key: (computed, reference, abs_diff)}`` over n_sat, E_sat, K_sat, Q_sat,
    E_sym, L_sym, m*/m — all finite floats.
    """
    got = compute_nmp(par)
    out = {}
    for key, ref in DD2_NMP_REFERENCE.items():
        val = float(got[key])
        out[key] = (val, float(ref), abs(val - ref))
    return out


def format_nmp_comparison(par):
    """Human-readable table string for the fig-11 notebook cell."""
    cmp = nmp_comparison(par)
    lines = [f"{'quantity':<16}{'computed':>14}{'DD2 ref':>12}{'|Δ|':>12}"]
    for key, (val, ref, d) in cmp.items():
        lines.append(f"{_NMP_LABELS[key]:<16}{val:>14.4f}{ref:>12.4f}{d:>12.4f}")
    return "\n".join(lines)


# =============================================================================
# NMP-built parametrization (second pass)
# =============================================================================
def near_dd2_nmp():
    """
    An NMP set near DD2 but not equal: L_sym nudged 55 -> 70 MeV (a visibly
    stiffer symmetry energy). The isovector sector inverts exactly, so the two
    passes differ cleanly in Y_p, R_1.4 and PNM.
    """
    nmp = dict(compute_nmp(Parametrization.from_dd2_defaults()))
    nmp["L_sym"] = 70.0
    return nmp


def build_nmp_par(nmp=None):
    """
    ``Parametrization.from_nmp`` on the near-DD2 set. Returns (par, status);
    the caller checks ``status.ok`` before plotting.
    """
    nmp = near_dd2_nmp() if nmp is None else nmp
    return Parametrization.from_nmp(nmp, return_status=True)


# =============================================================================
# Speed test: SFHo vs DD2 on a matched β-eq sweep
# =============================================================================
def benchmark_dd2_vs_sfho(par, grid=None, flags=NUCLEONIC):
    """
    Time the DD2 fast path (``sweep_beta_eq_octet(..., analytic_jac=True)``, first
    call discarded for Numba compile) against the SFHo table generator on the
    same β-eq density grid. Returns ms/point for each and the ratio.
    """
    from eos.sfho.compute_tables import compute_table, TableSettings
    grid = default_grid(n=100) if grid is None else np.asarray(grid, float)
    n = len(grid)

    # DD2: discard the first (compile) call, then time.
    sweep_beta_eq_octet(par, grid[:5], flags, include_photons=False,
                        analytic_jac=True)
    t0 = time.perf_counter()
    sweep_beta_eq_octet(par, grid, flags, include_photons=False,
                        analytic_jac=True)
    dd2_ms = 1e3 * (time.perf_counter() - t0) / n

    # SFHo on the same grid (nucleonic β-eq, T=0).
    settings = TableSettings(
        parametrization="sfho", particle_content="nucleons",
        equilibrium="beta_eq", n_B_values=grid, T_values=[0.0],
        include_photons=False, include_muons=flags.muons,
        print_results=False, print_timing=False, print_errors=False,
        save_to_file=False,
    )
    t0 = time.perf_counter()
    compute_table(settings)
    sfho_ms = 1e3 * (time.perf_counter() - t0) / n

    return dict(n_points=n, dd2_ms_per_pt=dd2_ms, sfho_ms_per_pt=sfho_ms,
                ratio=sfho_ms / dd2_ms)
