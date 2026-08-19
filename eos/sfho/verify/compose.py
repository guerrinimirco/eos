"""
verify/compose.py
=================
CompOSE table comparison for the SFHo engine.

Reference tables: HS(SFHo) and HS(SFHoY) (compose.obspm.fr; Steiner, Hempel &
Fischer, ApJ 774 (2013) 17, with the Hempel & Schaffner-Bielich statistical
model supplying the sub-saturation nuclei). Content of the general-purpose
tables: baryons and antibaryons, electrons and positrons at net n_e = Y_q n_B,
photons, and NUCLEI — no muons and no neutrinos.

That content is what the comparison has to match, and it decides the
conditions:

- the engine is run in `fixed_YC` with `leptons=True`, which is exactly the
  CompOSE (n_B, T, Y_q) convention: Y_q is the non-leptonic charge fraction and
  the electrons are whatever neutralises it;
- photons on, muons off, no thermal neutrinos;
- densities ABOVE the cluster-dissolution region only. Below about
  0.1 fm^-3 at T = 10 MeV the table is a gas of nuclei and this engine is
  uniform matter, so they are describing different states and a comparison
  there measures nothing.

Neutron-rich slices are the point. An error in the isovector sector vanishes
identically at Y_q = 0.5, which is how the missing omega(dA/domega) rho^2 term
in the energy density survived for so long; Y_q = 0.1 is where it shows.

Temperature is INTERPOLATED rather than snapped. The CompOSE T grid is
logarithmic and coarse — the points either side of 10 MeV are about 8.9 and
11.2 — so nearest-neighbour selection compares the engine at one temperature
against the table at another, which shows up as an apparent error of a percent
or so that is really just a temperature offset.

References:
- Typel, Oertel, Klahn et al., CompOSE manual, arXiv:2203.03209
- Steiner, Hempel, Fischer, ApJ 774 (2013) 17
- Fortin, Oertel, Providencia, PASA 35 (2018) e044
"""
import os

import numpy as np

from eos.general.compose import read_compose_data
from eos.sfho.solver import solve_fixed_yc

#: Local CompOSE data locations (downloaded from compose.obspm.fr).
COMPOSE_ROOT = os.path.expanduser("~/Desktop/Research/Compose")
SFHO_COMPOSE = os.path.join(COMPOSE_ROOT, "SFHO_Compose")     # nucleonic
SFHOY_COMPOSE = os.path.join(COMPOSE_ROOT, "SFHOY_Compose")   # with hyperons

_CACHE = {}


def available(compose_dir=SFHO_COMPOSE):
    """True when the table is on this machine.

    The tables are ~100 MB downloads and are not shipped with the repository,
    so every entry point here is optional rather than assumed.
    """
    return os.path.isfile(os.path.join(compose_dir, "eos.thermo"))


def load_table(compose_dir=SFHO_COMPOSE, name="SFHO"):
    """Cached CompOSE table load, as the 3-D (T, n_B, Y_q) grid."""
    if compose_dir not in _CACHE:
        _CACHE[compose_dir] = read_compose_data(compose_dir, name=name,
                                                verbose=False)
    return _CACHE[compose_dir]


def slice_at(data, Y_C, T):
    """One (T, Y_q) slice of the table, over its native n_B grid.

    Y_q is snapped to its nearest grid value and reported back, because the
    engine must then be run at the SAME fraction — a 0.01 mismatch in Y_q moves
    the pressure of neutron-rich matter by more than the agreement being
    measured. T is linearly interpolated between the bracketing grid points,
    for the reason in the module docstring, and falls back to the nearest point
    outside the grid.

    Returns a dict of arrays over n_B, plus the (T, Y_C) actually used.
    """
    iY = int(np.argmin(np.abs(data.Y_C_values - Y_C)))
    Y_use = float(data.Y_C_values[iY])
    grid = data.T_values

    def at(index):
        return {key: getattr(data, key)[index, :, iY]
                for key in ("P", "e", "s", "f", "mu_B", "mu_C", "mu_L")}

    if grid.min() < T < grid.max():
        hi = int(np.searchsorted(grid, T, side="right"))
        lo = hi - 1
        w = (T - float(grid[lo])) / float(grid[hi] - grid[lo])
        below, above = at(lo), at(hi)
        out = {}
        for key in below:
            out[key] = (1.0 - w) * below[key] + w * above[key]
        T_use = float(T)
    else:
        iT = int(np.argmin(np.abs(grid - T)))
        out = at(iT)
        T_use = float(grid[iT])

    out["n_B"] = data.n_B_values
    out["T"] = T_use
    out["Y_C"] = Y_use
    return out


def engine_point(par, n_B, Y_C, T, flags):
    """Uniform matter at (n_B, Y_q, T) with the CompOSE table's content.

    `fixed_YC` with neutralizing electrons and photons on: baryons at the
    imposed non-leptonic charge fraction, electrons and positrons at net
    n_e = n_C, blackbody photons. Muons and neutrinos stay off because the
    table has neither.

    Returns a dict with P, eps [MeV/fm^3], s_per_B, mu_B, mu_C [MeV], and the
    `converged` flag of the solve.
    """
    point = solve_fixed_yc(par, n_B, Y_C, flags, T=T, leptons=True)
    return dict(P=point.P, eps=point.eps, s_per_B=point.entropy_per_baryon,
                mu_B=point.mu_B, mu_C=point.mu_C, converged=point.converged)


def compare_slice(par, flags, compose_dir=SFHO_COMPOSE, name="SFHO",
                  T=10.0, Y_C=0.1, nB_min=0.2, nB_max=0.6):
    """Compare the engine against one (T, Y_q) CompOSE slice over a density range.

    Returns a dict with the slice actually used, the per-quantity maximum
    relative errors over it (P, eps, s, mu_B) and the per-density rows behind
    them, so a caller can see where the worst point sits rather than only how
    bad it is.
    """
    data = load_table(compose_dir, name=name)
    sl = slice_at(data, Y_C, T)
    keep = np.where((sl["n_B"] >= nB_min) & (sl["n_B"] <= nB_max))[0]

    rows = []
    for i in keep:
        n_B = float(sl["n_B"][i])
        P_ref, eps_ref = float(sl["P"][i]), float(sl["e"][i])
        s_ref, muB_ref = float(sl["s"][i]), float(sl["mu_B"][i])
        if not (np.isfinite(P_ref) and np.isfinite(eps_ref)):
            continue
        eng = engine_point(par, n_B, sl["Y_C"], sl["T"], flags)
        if not eng["converged"]:
            continue
        rows.append(dict(
            n_B=n_B,
            err_P=abs(eng["P"] / P_ref - 1.0),
            err_eps=abs(eng["eps"] / eps_ref - 1.0),
            err_s=abs(eng["s_per_B"] / s_ref - 1.0) if s_ref > 0 else 0.0,
            err_muB=abs(eng["mu_B"] / muB_ref - 1.0),
        ))
    if not rows:
        raise RuntimeError(
            f"no comparable CompOSE rows in [{nB_min}, {nB_max}] fm^-3 at "
            f"T={sl['T']:g} MeV, Y_q={sl['Y_C']:g}")

    out = dict(T=sl["T"], Y_C=sl["Y_C"], n_points=len(rows),
               n_B_range=(rows[0]["n_B"], rows[-1]["n_B"]), rows=rows)
    for key in ("err_P", "err_eps", "err_s", "err_muB"):
        out[f"max_{key}"] = max(r[key] for r in rows)
    return out
