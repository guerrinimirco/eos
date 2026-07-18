"""
verify/compose.py
====================
CompOSE table comparison for the DD2 engine (report §3.7 check 1).

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
from eos.sfho.compose_loader import read_compose_data
from eos.dd2.physics.thermo import kinetic_thermo
from eos.dd2.solver import solve_composition

#: Local CompOSE data locations (downloaded from compose.obspm.fr).
COMPOSE_ROOT = os.path.expanduser("~/Desktop/Research/Compose")
DD2_COMPOSE = os.path.join(COMPOSE_ROOT, "DD2_Compose")        # id 18, npe
DD2_NOE_COMPOSE = os.path.join(COMPOSE_ROOT, "DD2_noe_Compose")  # id 17, np
DD2Y_COMPOSE = os.path.join(COMPOSE_ROOT, "DD2Y_Compose")      # id 104, DD2Y

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


def engine_point(par, n_B, Y_q, T, include_electrons=True):
    """
    Uniform npe(+photon) matter at fixed (n_B, Y_q, T) matching the HS(DD2)
    table content: baryons + electrons(+positrons) at net n_e = Y_q n_B +
    photons; no muons.

    Returns dict with P, eps [MeV/fm^3], s_per_B, mu_B, mu_C [MeV].
    Uses the "physical" nucleon-mass convention (the HS(DD2) one).
    """
    from dataclasses import replace as _dc_replace
    if par.nucleon_mass_mode != "physical":
        par = _dc_replace(par, nucleon_mass_mode="physical")
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
                mu_B=base.mu_n, mu_C=base.mu_p - base.mu_n)


def compare_slice(par, compose_dir=DD2_COMPOSE, T=5.0, Y_q=0.5,
                  nB_min=0.14, nB_max=0.6, include_electrons=True):
    """
    Compare the engine against one native-(T, Y_q) CompOSE slice over
    n_B in [nB_min, nB_max] (uniform-matter region).

    Returns dict with the slice actually used and per-quantity max relative
    errors over the slice: P, eps, s, mu_B.
    """
    data = load_table(compose_dir)
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
                           include_electrons=include_electrons)
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
