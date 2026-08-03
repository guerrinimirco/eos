"""
mixed/notebook_api.py
=====================
Plotting / analysis helpers for the DD2+vMIT eta-mixed-phase engine, so
`notebooks/mixed_usage.py` stays thin (same split as eos/dd2/notebook_api.py).
Every figure is built as ``fig, ax = plt.subplots()`` and drawn through ``ax.*``.

A *case* is a ``(label, eta, spec)`` triple -- a named equilibrium configuration
(mode + neutrality parameter). Figures overlay a list of cases so the same
quantity can be compared across modes / eta on one axis, which is the whole
point of the engine (spec §1.5). Thermodynamic derivatives (C_V, C_P, c_ad^2,
susceptibilities) are finite-differenced on the equilibrium solve -- the
analytic blocks are P7 (deferred); see the ponytail notes.

Units: fm-based throughout (n_B [fm^-3], P/eps [MeV/fm^3], T/mu [MeV]).
"""
import numpy as np
import matplotlib.pyplot as plt

from eos.mixed import mode_A
from eos.mixed.solver import solve_mixed
from eos.mixed.continuation import sweep_mixed
from eos.mixed.table import build_mixed_eos_table, mass_radius_mixed


def default_grid(n_lo=0.10, n_hi=1.20, n=56):
    """Default n_B grid [fm^-3] spanning crust-edge to deep core."""
    return np.round(np.linspace(n_lo, n_hi, n), 4)


def _sweep(par, flags, grid, eta, spec, vp, T):
    """Warm-started sweep -> arrays (n_B, P, eps, s, chi, mu_B). Non-convergent
    points are dropped, so lengths track the convergent subset."""
    rs = sweep_mixed(par, flags, grid, eta, spec, vmit_params=vp, T=T)
    cols = dict(n_B="n_B", P="P", eps="eps", s="s", chi="chi", mu_B="mu_B")
    out = {k: np.array([getattr(r, a) for r in rs]) for k, a in cols.items()}
    out["results"] = rs
    return out


# =============================================================================
# 1-2. P(n_B) and chi(n_B) across cases
# =============================================================================
def plot_p_vs_nb(par, flags, grid, cases, vp=None, T=0.0):
    """Total pressure vs n_B for each case (mode / eta overlaid)."""
    fig, ax = plt.subplots()
    for label, eta, spec in cases:
        d = _sweep(par, flags, grid, eta, spec, vp, T)
        ax.plot(d["n_B"], d["P"], "-", lw=1.8, label=label)
    ax.set_xlabel(r"$n_B$ [fm$^{-3}$]")
    ax.set_ylabel(r"$P$ [MeV/fm$^3$]")
    ax.set_title(f"DD2+vMIT pressure (T = {T:g} MeV)")
    ax.legend(); ax.grid(True, alpha=0.3)
    return fig


def plot_chi_vs_nb(par, flags, grid, cases, vp=None, T=0.0):
    """Quark volume fraction chi vs n_B: 0 -> 1 across the mixed window; the
    width in n_B shows how Gibbs (eta=0) spreads it vs Maxwell (eta=1)."""
    fig, ax = plt.subplots()
    for label, eta, spec in cases:
        d = _sweep(par, flags, grid, eta, spec, vp, T)
        m = (d["chi"] > 1e-6) & (d["chi"] < 1 - 1e-6)
        ax.plot(d["n_B"][m], d["chi"][m], "-", lw=1.8, label=label)
    ax.set_xlabel(r"$n_B$ [fm$^{-3}$]")
    ax.set_ylabel(r"quark fraction $\chi$")
    ax.set_title(f"Mixed-phase quark fraction (T = {T:g} MeV)")
    ax.legend(); ax.grid(True, alpha=0.3)
    return fig


# =============================================================================
# 3. Composition Y_i(n_B) for one case (both phases)
# =============================================================================
def plot_composition(par, flags, grid, eta, spec, vp=None, T=0.0, y_floor=1e-3):
    """Global (volume-weighted) number fraction of every populated species vs
    n_B for one case. Hadrons solid, quarks dashed; leptons omitted."""
    from eos.mixed.table import composition_table
    rows = composition_table(par, flags, grid, eta, spec, vmit_params=vp, T=T)
    nB = np.array([r["n_B"] for r in rows])
    species = sorted({k[2:] for r in rows for k in r if k.startswith("Y_")
                      and k not in ("Y_C", "Y_S")})
    quarks = {"u", "d", "s"}
    fig, ax = plt.subplots()
    for sp in species:
        y = np.array([r.get(f"Y_{sp}", 0.0) for r in rows])
        if np.nanmax(y) < y_floor:
            continue
        ls = "--" if sp in quarks else "-"
        ax.plot(nB, y, ls, lw=1.6, label=sp)
    ax.set_yscale("log"); ax.set_ylim(y_floor, 2.0)
    ax.set_xlabel(r"$n_B$ [fm$^{-3}$]")
    ax.set_ylabel(r"$Y_i = n_i / n_B$ (volume-weighted)")
    ax.set_title(f"Composition (T = {T:g} MeV)")
    ax.legend(ncol=2, fontsize=8); ax.grid(True, alpha=0.3)
    return fig


# =============================================================================
# 4. Phase boundaries: onset / offset n_B vs eta
# =============================================================================
def phase_boundaries(par, flags, grid, spec, etas, vp=None, T=0.0):
    """(onset, offset) n_B of the mixed window vs eta (via build_mixed_eos_table's
    boundary location). Returns (etas, onset[], offset[])."""
    on, off = [], []
    for eta in etas:
        t = build_mixed_eos_table(par, flags, grid, float(eta), spec,
                                  vmit_params=vp, T=T)
        on.append(t.onset); off.append(t.offset)
    return np.asarray(etas, float), np.array(on), np.array(off)


def plot_phase_boundaries(par, flags, grid, spec, etas, vp=None, T=0.0):
    """Mixed-window onset/offset density vs eta -- Gibbs (eta=0) is widest, the
    window narrows toward the Maxwell (eta=1) density jump."""
    e, on, off = phase_boundaries(par, flags, grid, spec, etas, vp, T)
    fig, ax = plt.subplots()
    ax.plot(e, on, "o-", lw=1.8, label="onset ($\\chi\\to0$)")
    ax.plot(e, off, "s-", lw=1.8, label="offset ($\\chi\\to1$)")
    ax.fill_between(e, on, off, alpha=0.15)
    ax.set_xlabel(r"$\eta$ (local-neutrality fraction)")
    ax.set_ylabel(r"$n_B$ [fm$^{-3}$]")
    ax.set_title(f"Mixed-phase boundaries (T = {T:g} MeV)")
    ax.legend(); ax.grid(True, alpha=0.3)
    return fig


# =============================================================================
# 5. Sound speed: equilibrium c_s^2 across cases
# =============================================================================
def plot_sound_speed(par, flags, grid, cases, vp=None, T=0.0):
    """Equilibrium c_s^2 = dP/deps along each case (the mixed phase softens it;
    a Maxwell plateau drives it to 0)."""
    fig, ax = plt.subplots()
    for label, eta, spec in cases:
        d = _sweep(par, flags, grid, eta, spec, vp, T)
        cs2 = np.gradient(d["P"], d["eps"])
        ax.plot(d["n_B"], cs2, "-", lw=1.8, label=label)
    ax.axhline(1.0, color="k", ls=":", lw=1, label="causal limit")
    ax.set_xlabel(r"$n_B$ [fm$^{-3}$]")
    ax.set_ylabel(r"$c_s^2 / c^2$")
    ax.set_title(f"Equilibrium sound speed (T = {T:g} MeV)")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(-0.02, 1.05)
    return fig


# =============================================================================
# 6. Thermal derivatives: C_V, C_P, c_ad^2 (finite-difference, T>0)
# =============================================================================
def _thermo_derivs(par, flags, n_B, eta, spec, vp, T, dT=0.5, rel_dn=2e-3):
    """Per-baryon C_V, C_P and adiabatic c_ad^2 at (n_B, T) by central FD on the
    EQUILIBRIUM solve (composition re-equilibrates at each stencil point):

        c_V = (1/n_B)(deps/dT)_n
        c_P = c_V + T (dP/dT)_n^2 / (n^2 (dP/dn)_T)   [Mayer relation]
        c_ad^2 = (c_P/c_V) (dP/deps)_T

    # ponytail: equilibrium (not frozen) heat capacities via FD re-solves; swap
    # for analytic coefficients once P7 ships the mixed Jacobian blocks.
    """
    def sol(n, t):
        return solve_mixed(par, flags, n, eta, spec, vmit_params=vp, T=t,
                           check_consistency=False)
    cp_, cm = sol(n_B, T + dT), sol(n_B, T - dT)
    np_, nm = sol(n_B * (1 + rel_dn), T), sol(n_B * (1 - rel_dn), T)
    deps_dT = (cp_.eps - cm.eps) / (2 * dT)
    dP_dT = (cp_.P - cm.P) / (2 * dT)
    dP_dn = (np_.P - nm.P) / (2 * rel_dn * n_B)
    deps_dn = (np_.eps - nm.eps) / (2 * rel_dn * n_B)
    cv = deps_dT / n_B
    cp = cv + T * dP_dT ** 2 / (n_B ** 2 * dP_dn)
    cad2 = (cp / cv) * (dP_dn / deps_dn)
    return cv, cp, cad2


def thermal_profile(par, flags, grid, eta, spec, vp=None, T=10.0):
    """(n_B, c_V, c_P, c_ad^2) along one case at fixed T>0 (skips FD failures)."""
    nB, cv, cp, cad = [], [], [], []
    for n in grid:
        try:
            a, b, c = _thermo_derivs(par, flags, float(n), eta, spec, vp, T)
        except (RuntimeError, ValueError, ZeroDivisionError):
            continue
        nB.append(n); cv.append(a); cp.append(b); cad.append(c)
    return (np.array(nB), np.array(cv), np.array(cp), np.array(cad))


def plot_heat_capacity(par, flags, grid, eta, spec, vp=None, T=10.0):
    """Per-baryon C_V and C_P vs n_B at fixed T>0 (equilibrium, FD)."""
    nB, cv, cp, _ = thermal_profile(par, flags, grid, eta, spec, vp, T)
    fig, ax = plt.subplots()
    ax.plot(nB, cv, "-", lw=1.8, label=r"$C_V / n_B$")
    ax.plot(nB, cp, "--", lw=1.8, label=r"$C_P / n_B$")
    ax.set_xlabel(r"$n_B$ [fm$^{-3}$]")
    ax.set_ylabel(r"heat capacity per baryon [$k_B$]")
    ax.set_title(f"Mixed-phase heat capacities (T = {T:g} MeV)")
    ax.legend(); ax.grid(True, alpha=0.3)
    return fig


def plot_adiabatic_cs2(par, flags, grid, eta, spec, vp=None, T=10.0):
    """Adiabatic c_ad^2 = (dP/deps)_s vs n_B at fixed T>0 (FD)."""
    nB, _, _, cad = thermal_profile(par, flags, grid, eta, spec, vp, T)
    fig, ax = plt.subplots()
    ax.plot(nB, cad, "-", lw=1.8)
    ax.axhline(1.0, color="k", ls=":", lw=1, label="causal limit")
    ax.set_xlabel(r"$n_B$ [fm$^{-3}$]")
    ax.set_ylabel(r"$c_{\mathrm{ad}}^2 / c^2$")
    ax.set_title(f"Adiabatic sound speed (T = {T:g} MeV)")
    ax.legend(); ax.grid(True, alpha=0.3)
    return fig


# =============================================================================
# 7. Susceptibility: baryon-number chi_B = dn_B/dmu_B along a case
# =============================================================================
def plot_susceptibility(par, flags, grid, cases, vp=None, T=0.0):
    """Baryon-number susceptibility chi_B = dn_B/dmu_B along each case, from the
    solved mu_B(n_B) sequence (its dip through the mixed phase is the softening).
    # ponytail: headline chi_B by FD on the sequence; the full chi_ab matrix
    # (via coefficients_jac + a quark block) is P7."""
    fig, ax = plt.subplots()
    for label, eta, spec in cases:
        d = _sweep(par, flags, grid, eta, spec, vp, T)
        order = np.argsort(d["mu_B"])
        mu, n = d["mu_B"][order], d["n_B"][order]
        chi_B = np.gradient(n, mu)
        ax.plot(n, chi_B, "-", lw=1.8, label=label)
    ax.set_xlabel(r"$n_B$ [fm$^{-3}$]")
    ax.set_ylabel(r"$\chi_B = \mathrm{d}n_B/\mathrm{d}\mu_B$ [fm$^{-3}$/MeV]")
    ax.set_title(f"Baryon-number susceptibility (T = {T:g} MeV)")
    ax.legend(); ax.grid(True, alpha=0.3)
    return fig


# =============================================================================
# 8-10. TOV: M-R and Lambda-M across cases
# =============================================================================
def compute_tov(par, flags, grid, eta, spec, vp=None, T=0.0, n_ec=160,
                e_c_min=150.0, e_c_max=3000.0, backend="scipy"):
    """Mixed-EoS TOV sequence (crust + core) -> dict with the stable branch
    (M, R, Lambda), M_max, R_1.4, Lambda_1.4. Wraps table.mass_radius_mixed and
    trims to the stable branch."""
    from eos.tov.solver import find_mmax_precise
    mr = mass_radius_mixed(par, flags, grid, eta, spec, vmit_params=vp, T=T,
                           n_ec=n_ec, e_c_min=e_c_min, e_c_max=e_c_max,
                           compute_tidal=True, backend=backend)
    res = mr["results"]
    idx, _, M_max = find_mmax_precise(res)
    M, R, Lam = res[:idx + 1, 4], res[:idx + 1, 3], res[:idx + 1, 6]
    stable = M[-1] >= 1.4 > M[0]
    L_1p4 = float(np.interp(1.4, M, Lam)) if stable else float("nan")
    return dict(M=M, R=R, Lambda=Lam, M_max=float(M_max),
                R_Mmax=mr["R_Mmax"], R_1p4=mr["R_1p4"], Lambda_1p4=L_1p4,
                table=mr["table"], results=res)


def plot_mass_radius(cases_tov):
    """M-R for each (label -> tov dict) in cases_tov."""
    fig, ax = plt.subplots()
    for label, tov in cases_tov.items():
        ax.plot(tov["R"], tov["M"], "-", lw=2, label=f"{label} "
                f"($M_{{max}}$={tov['M_max']:.2f})")
        ax.plot(tov["R_Mmax"], tov["M_max"], "o", ms=5)
    ax.set_xlabel(r"$R$ [km]"); ax.set_ylabel(r"$M$ [$M_\odot$]")
    ax.set_title("DD2+vMIT mass-radius"); ax.legend(); ax.grid(True, alpha=0.3)
    return fig


def plot_lambda_mass(cases_tov):
    """Tidal Lambda vs M for each case (the Maxwell density jump's tidal
    correction is applied automatically inside eos/tov)."""
    fig, ax = plt.subplots()
    for label, tov in cases_tov.items():
        ax.semilogy(tov["M"], tov["Lambda"], "-", lw=2, label=label)
    ax.set_xlabel(r"$M$ [$M_\odot$]"); ax.set_ylabel(r"$\Lambda$")
    ax.set_title("DD2+vMIT tidal deformability")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    return fig
