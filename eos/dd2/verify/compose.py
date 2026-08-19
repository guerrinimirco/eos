"""
verify/compose.py
====================
CompOSE table comparison for the DD2 engine.

Nucleonic reference: HS(DD2) (CompOSE ids 17/18, Hempel & Schaffner-Bielich
statistical model with DD2 interactions; Typel et al. 2010). Content of the
"with electrons" table: n, p (+antibaryons), e-, e+, photons, and NUCLEI
(excluded-volume NSE) — no muons. The uniform-matter engine is therefore
compared on native (T, Y_q) grid points at densities above the
cluster-dissolution region, with muons off, electrons at fixed net
n_e = Y_q n_B, and photons on.

Known convention difference: the engine's golden-point kernel uses the
average nucleon mass, HS(DD2) uses the physical m_n != m_p. This cancels in
P, enters eps only via the rest-mass term (n_n - n_p)(m_n - m_p)/2, and
shifts mu_B by ~(m_n - m_bar). compare_slice reports both raw and
rest-mass-adjusted eps errors so the physics agreement is visible behind
the bookkeeping offset.
"""
import os

import numpy as np

from eos.general.particles import Electron
from eos.general.physics_constants import hc3
from eos.general.thermodynamics_leptons import photon_thermo
from eos.general.fermi_integrals import invert_fermi_density
from eos.general.compose import read_compose_data
from eos.dd2.thermodynamics import kinetic_thermo
from eos.dd2.solver import solve_composition

#: Local CompOSE data locations (downloaded from compose.obspm.fr).
COMPOSE_ROOT = os.path.expanduser("~/Desktop/Research/Compose")
DD2_COMPOSE = os.path.join(COMPOSE_ROOT, "DD2_Compose")        # id 18, npe
DD2_NOE_COMPOSE = os.path.join(COMPOSE_ROOT, "DD2_noe_Compose")  # id 17, np
DD2Y_COMPOSE = os.path.join(COMPOSE_ROOT, "DD2Y_Compose")      # id 104, DD2Y
DD2Y_NS = os.path.join(DD2Y_COMPOSE, "eos.thermo.ns")         # cold beta-eq 1D

_CACHE = {}


def load_table(compose_dir, name="DD2"):
    """Cached CompOSE table load (reuses the SFHo reader — same format)."""
    if compose_dir not in _CACHE:
        _CACHE[compose_dir] = read_compose_data(compose_dir, name=name,
                                                verbose=False)
    return _CACHE[compose_dir]


def _nearest(grid, x):
    i = int(np.argmin(np.abs(np.asarray(grid) - x)))
    return i, float(grid[i])


def engine_point(par, n_B, Y_q, T, include_electrons=True, flags=None):
    """
    Uniform matter at fixed (n_B, Y_q, T) matching the CompOSE general-purpose
    table content: baryons + electrons(+positrons) at net n_e = Y_q n_B +
    photons; no muons. flags=None (or flags.hyperons=False) solves nucleons
    only (HS(DD2)); with flags.hyperons the full octet is solved in fixed-Y_C
    mode (DD2Y general-purpose table).

    Returns dict with P, eps [MeV/fm^3], s_per_B, mu_B, mu_C [MeV].
    Uses the "physical" nucleon-mass convention (the CompOSE one).
    """
    from dataclasses import replace as _dc_replace
    if par.nucleon_mass_mode != "physical":
        par = _dc_replace(par, nucleon_mass_mode="physical")
    if flags is not None and flags.hyperons:
        from eos.dd2.solver import solve_fixed_yc_octet
        base = solve_fixed_yc_octet(par, n_B, Y_q, flags, T=T,
                                    include_photons=False)
    else:
        base = solve_composition(par, (1.0 - Y_q) * n_B, Y_q * n_B, T=T)
    P, eps, s = base.P, base.eps, base.s
    if include_electrons:
        n_e = Y_q * n_B
        if T > 0.0:
            mu_e = invert_fermi_density(n_e, T, Electron.mass, 2.0)
        else:
            kF = (3.0 * np.pi ** 2 * n_e * hc3) ** (1.0 / 3.0)
            mu_e = float(np.sqrt(kF ** 2 + Electron.mass ** 2))
        ne_nat, Pe, ee, se, _ = kinetic_thermo(mu_e, Electron.mass, 2.0, T)
        P += Pe / hc3
        eps += ee / hc3
        s += se / hc3
    if T > 0.0:
        ph = photon_thermo(T)
        P, eps, s = P + ph.P, eps + ph.e, s + ph.s
    return dict(P=P, eps=eps, s_per_B=s / n_B,
                mu_B=base.mu_B, mu_C=base.mu_C)


def compare_slice(par, compose_dir=DD2_COMPOSE, T=5.0, Y_q=0.5,
                  nB_min=0.14, nB_max=0.6, include_electrons=True,
                  flags=None, name="DD2"):
    """
    Compare the engine against one native-(T, Y_q) CompOSE slice over
    n_B in [nB_min, nB_max] (uniform-matter region). flags=None compares the
    nucleonic HS(DD2) table; pass a hyperonic SpeciesFlags to compare the DD2Y
    general-purpose table in fixed-Y_C mode.

    Returns dict with the slice actually used and per-quantity max relative
    errors over the slice: P, eps, s, mu_B.
    """
    data = load_table(compose_dir, name=name)
    iT, T_use = _nearest(data.T_values, T)
    iY, Y_use = _nearest(data.Y_C_values, Y_q)
    sel = np.where((data.n_B_values >= nB_min)
                   & (data.n_B_values <= nB_max))[0]

    rows = []
    for i in sel:
        nB = float(data.n_B_values[i])
        tab_P = data.P[iT, i, iY]
        tab_e = data.e[iT, i, iY]
        tab_s = data.s[iT, i, iY]
        tab_muB = data.mu_B[iT, i, iY]
        if not np.isfinite(tab_P):
            continue
        eng = engine_point(par, nB, Y_use, T_use,
                           include_electrons=include_electrons, flags=flags)
        rows.append(dict(
            n_B=nB,
            err_P=abs(eng["P"] / tab_P - 1.0),
            err_eps=abs(eng["eps"] / tab_e - 1.0),
            err_s=abs(eng["s_per_B"] / tab_s - 1.0) if tab_s > 0 else 0.0,
            err_muB=abs(eng["mu_B"] / tab_muB - 1.0),
        ))
    if not rows:
        raise RuntimeError(
            f"no finite CompOSE rows in [{nB_min}, {nB_max}] at "
            f"T={T_use}, Y_q={Y_use}")
    out = dict(T=T_use, Y_q=Y_use, n_points=len(rows),
               n_B_range=(rows[0]["n_B"], rows[-1]["n_B"]))
    for key in ("err_P", "err_eps", "err_s", "err_muB"):
        out[f"max_{key}"] = max(r[key] for r in rows)
    out["rows"] = rows
    return out


def load_ns_table(ns_path=DD2Y_NS):
    """
    Cold catalyzed (beta-eq, T=0) 1D CompOSE NS table (eos.thermo.ns +
    eos.nb.ns). Returns (n_B [fm^-3], P [MeV/fm^3], eps [MeV/fm^3],
    mu_B [MeV], Y_q) sorted by n_B.
    """
    directory = os.path.dirname(ns_path)
    nb = np.array([float(x) for x in
                   open(os.path.join(directory, "eos.nb.ns")).read().split()[2:]])
    lines = open(ns_path).read().splitlines()
    m_n = float(lines[0].split()[0])
    rows = np.array([[float(x) for x in ln.split()]
                     for ln in lines[1:] if len(ln.split()) >= 10])
    i_nb = rows[:, 1].astype(int) - 1
    n_B = nb[i_nb]
    # CompOSE eos.thermo columns (1-based, after the 3 index cols):
    # Q1=p/nB, Q2=s/nB, Q3=muB/mn-1, ..., Q7=e/(nB mn)-1; trailing extra = Y_q.
    P = rows[:, 3] * n_B
    eps = (rows[:, 9] + 1.0) * m_n * n_B
    mu_B = (rows[:, 5] + 1.0) * m_n
    Y_q = rows[:, -2]
    order = np.argsort(n_B)
    return (n_B[order], P[order], eps[order], mu_B[order], Y_q[order])


def compare_ns(points, ns_path=DD2Y_NS, nB_min=0.2, nB_max=1.2):
    """
    Compare a solved cold beta-eq sweep (list of EoSPoint) against the DD2Y
    cold-NS CompOSE table over [nB_min, nB_max]. Returns max/median relative
    errors on P, eps, mu_B. eps and mu_B are the robust indicators; P is a
    small difference of large numbers and amplifies any composition/coupling
    difference.
    """
    n_B, P, eps, mu_B, _ = load_ns_table(ns_path)
    m = (n_B >= nB_min) & (n_B <= nB_max)
    myN = np.array([p.n_B for p in points])
    myP = np.array([p.P for p in points])
    myE = np.array([p.eps for p in points])
    myM = np.array([p.mu_B for p in points])
    eP = np.abs(np.interp(n_B[m], myN, myP) / P[m] - 1.0)
    eE = np.abs(np.interp(n_B[m], myN, myE) / eps[m] - 1.0)
    eM = np.abs(np.interp(n_B[m], myN, myM) / mu_B[m] - 1.0)
    return dict(n_points=int(m.sum()),
                max_err_P=float(eP.max()), med_err_P=float(np.median(eP)),
                max_err_eps=float(eE.max()), max_err_muB=float(eM.max()))
