"""Reading a CompOSE equation-of-state table: the 3-axis (n_B, Y_C, T) grid.

CompOSE (Typel et al., the CompOSE manual) distributes a tabulated EoS as a
set of plain files -- ``eos.nb``, ``eos.t``, ``eos.yq`` for the three axes and
``eos.thermo`` for the quantities on them. This module reads that layout once,
caches it, and interpolates bilinearly in (T, Y_C) along the native n_B grid.
Nothing here is specific to one tabulated model; SFHo is simply the table this
repository is usually pointed at.

It lives in ``general/`` because it PRODUCES data and imports nothing else in
the repository (CLAUDE.md section 1). In particular it does not know about
``EOSTable_for_TOV``: a slice comes back as plain arrays, and the layer
entitled to wrap those into a structure-solver table is ``eos.astro.tov``,
which consumes them. Reading it the other way round is what made this module
and ``eos.astro.tov.solver`` import each other.

Quantities exposed per (T, Y_C) slice:
    n_B [fm^-3], P [MeV/fm^3], epsilon [MeV/fm^3], mu_B [MeV], mu_C [MeV],
    mu_L [MeV], s [per baryon], f [MeV/fm^3]
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np



@dataclass
class CompOSEData:
    """Container for the 3-D CompOSE EOS grid (n_T × n_nB × n_Yq)."""

    name: str
    m_N: float              # neutron mass [MeV] used for normalization
    n_B_values: np.ndarray  # fm⁻³
    T_values: np.ndarray    # MeV
    Y_C_values: np.ndarray  # charge fraction
    P: np.ndarray           # MeV/fm³
    s: np.ndarray           # per baryon
    e: np.ndarray           # MeV/fm³
    f: np.ndarray           # MeV/fm³
    mu_B: np.ndarray        # MeV
    mu_C: np.ndarray        # MeV
    mu_L: np.ndarray        # MeV


def read_compose_data(compose_dir: str, name: str = "CompOSE",
                      verbose: bool = True) -> CompOSEData:
    """Read CompOSE EOS files (``eos.nb``, ``eos.t``, ``eos.yq``, ``eos.thermo``).

    Self-contained reader — does not depend on any other SFHO module so it
    works even if sibling modules are broken or under refactor.

    Quantities returned are in physical units (MeV, fm) following the standard
    CompOSE convention with normalization by the neutron mass.
    """

    def _read_grid_file(filename: str) -> np.ndarray:
        with open(os.path.join(compose_dir, filename), "r") as f:
            lines = f.readlines()
        # CompOSE grid files start with two header lines (flag, N_points).
        return np.array([float(line) for line in lines[2:] if line.strip()])

    n_B_values = _read_grid_file("eos.nb")
    T_values = _read_grid_file("eos.t")
    Y_C_values = _read_grid_file("eos.yq")

    n_nB, n_T, n_Yq = len(n_B_values), len(T_values), len(Y_C_values)

    if verbose:
        print(f"Reading CompOSE data from {compose_dir}")
        print(f"  n_B: {n_nB} points, range [{n_B_values.min():.2e}, {n_B_values.max():.2e}] fm^-3")
        print(f"  T:   {n_T} points, range [{T_values.min():.2f}, {T_values.max():.2f}] MeV")
        print(f"  Y_C: {n_Yq} points, range [{Y_C_values.min():.3f}, {Y_C_values.max():.3f}]")

    thermo_path = os.path.join(compose_dir, "eos.thermo")
    with open(thermo_path, "r") as f:
        header = f.readline().split()
        m_N_neutron = float(header[0])
        rows = []
        for line in f:
            parts = line.split()
            if len(parts) >= 10:
                rows.append([float(x) for x in parts])

    data = np.array(rows)
    if verbose:
        print(f"  m_N (neutron) = {m_N_neutron:.5f} MeV")
        print(f"  Thermo data: {len(data)} rows")

    shape = (n_T, n_nB, n_Yq)
    P = np.full(shape, np.nan)
    s = np.full(shape, np.nan)
    e = np.full(shape, np.nan)
    f_arr = np.full(shape, np.nan)
    mu_B = np.full(shape, np.nan)
    mu_C = np.full(shape, np.nan)
    mu_L = np.full(shape, np.nan)

    # eos.thermo columns:
    # i_T, i_nB, i_Yq, P/nB, s/nB, μB/mN-1, μC/mN, μL/mN, f/mN/nB-1, ε/mN/nB-1
    # Indices in CompOSE files are 1-based.
    for row in data:
        i_T = int(row[0]) - 1
        i_nB = int(row[1]) - 1
        i_Yq = int(row[2]) - 1
        if not (0 <= i_T < n_T and 0 <= i_nB < n_nB and 0 <= i_Yq < n_Yq):
            continue
        nB = n_B_values[i_nB]
        P[i_T, i_nB, i_Yq] = row[3] * nB
        s[i_T, i_nB, i_Yq] = row[4]
        mu_B[i_T, i_nB, i_Yq] = (row[5] + 1) * m_N_neutron
        mu_C[i_T, i_nB, i_Yq] = row[6] * m_N_neutron
        mu_L[i_T, i_nB, i_Yq] = row[7] * m_N_neutron
        f_arr[i_T, i_nB, i_Yq] = (row[8] + 1) * m_N_neutron * nB
        e[i_T, i_nB, i_Yq] = (row[9] + 1) * m_N_neutron * nB

    return CompOSEData(
        name=name, m_N=m_N_neutron,
        n_B_values=n_B_values, T_values=T_values, Y_C_values=Y_C_values,
        P=P, s=s, e=e, f=f_arr, mu_B=mu_B, mu_C=mu_C, mu_L=mu_L,
    )


class ComposeLookup:
    """Cached lookup over the full SFHO CompOSE (n_B, Y_C, T) grid.

    Parameters
    ----------
    compose_dir : str
        Directory containing ``eos.nb``, ``eos.t``, ``eos.yq``, ``eos.thermo``.

    Notes
    -----
    The load is eager (happens in ``__init__``) and the full 3-D arrays are kept
    in memory (~few hundred MB at most for SFHO).
    """

    _QUANTITIES = ("P", "e", "s", "f", "mu_B", "mu_C", "mu_L")

    def __init__(self, compose_dir: str, name: str = "CompOSE"):
        self.compose_dir = compose_dir
        self.data: CompOSEData = read_compose_data(compose_dir, name=name)

    # ---- public API ----------------------------------------------------

    @property
    def n_B(self) -> np.ndarray:
        """Native n_B grid [fm⁻³] (length 308 for SFHO)."""
        return self.data.n_B_values

    @property
    def T_grid(self) -> np.ndarray:
        """Native T grid [MeV]."""
        return self.data.T_values

    @property
    def Y_C_grid(self) -> np.ndarray:
        """Native Y_C grid."""
        return self.data.Y_C_values

    def get_slice(self, T: float, Y_C: float) -> Dict[str, np.ndarray]:
        """Bilinear (T, Y_C) interpolation along the native n_B grid.

        Returns
        -------
        dict
            Keys ``n_B, P, epsilon, mu_B, mu_C, mu_L, s, f`` (1-D arrays), plus
            scalar ``T`` and ``Y_C`` echoing the request.

        Notes
        -----
        T and Y_C outside the native grid are clamped to the endpoints (no
        extrapolation). The native arrays contain ``NaN`` where CompOSE has no
        data for a given (i_T, i_nB, i_Yq); the bilinear blend will propagate
        those ``NaN`` cells — callers should handle them (or trim by ``n_B``).
        """
        T_arr = self.T_grid
        Y_arr = self.Y_C_grid

        # Locate bracketing indices (and weights) on each axis, clamped.
        iT_lo, iT_hi, wT = _bracket(T_arr, T)
        iY_lo, iY_hi, wY = _bracket(Y_arr, Y_C)

        slice_out: Dict[str, np.ndarray] = {"n_B": self.n_B}
        for q in self._QUANTITIES:
            arr3d = getattr(self.data, q)
            v_ll = arr3d[iT_lo, :, iY_lo]
            v_lh = arr3d[iT_lo, :, iY_hi]
            v_hl = arr3d[iT_hi, :, iY_lo]
            v_hh = arr3d[iT_hi, :, iY_hi]
            v = (
                (1 - wT) * (1 - wY) * v_ll
                + (1 - wT) * wY * v_lh
                + wT * (1 - wY) * v_hl
                + wT * wY * v_hh
            )
            # rename "e" to the more readable "epsilon" in the output dict
            key = "epsilon" if q == "e" else q
            slice_out[key] = v

        slice_out["T"] = float(T)
        slice_out["Y_C"] = float(Y_C)
        return slice_out

    def slice_arrays(
        self,
        T: float,
        Y_C: float,
        n_B_max: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(P, epsilon, n_B) of the (T, Y_C) slice, finite rows only.

        Plain arrays, deliberately: this module produces data and does not
        know what will be built from it. `eos.astro.tov.solver` is what turns
        these into an ``EOSTable_for_TOV`` for crust attachment.

        Parameters
        ----------
        T, Y_C : float
            Slice coordinates.
        n_B_max : float, optional
            If given, drop rows with ``n_B > n_B_max``, so the result covers
            only the subnuclear regime -- which is what crust attachment
            wants. If ``None``, the full native n_B range comes back.

        Notes
        -----
        Rows where P, epsilon or n_B are not finite are dropped.
        """
        s = self.get_slice(T, Y_C)
        n_B = s["n_B"]
        P = s["P"]
        eps = s["epsilon"]

        mask = np.isfinite(P) & np.isfinite(eps) & np.isfinite(n_B)
        if n_B_max is not None:
            mask &= n_B <= n_B_max

        return P[mask], eps[mask], n_B[mask]


# ---- helpers -----------------------------------------------------------

def _bracket(grid: np.ndarray, x: float) -> tuple[int, int, float]:
    """Find indices (lo, hi) bracketing ``x`` in a sorted grid plus weight.

    Returns ``(i_lo, i_hi, w)`` such that
    ``value ≈ (1 - w) * grid[i_lo] + w * grid[i_hi]``. ``x`` outside the grid
    is clamped (``w = 0`` or ``1``, ``i_lo == i_hi``).
    """
    n = len(grid)
    if x <= grid[0]:
        return 0, 0, 0.0
    if x >= grid[-1]:
        return n - 1, n - 1, 0.0
    i_hi = int(np.searchsorted(grid, x, side="right"))
    i_lo = i_hi - 1
    span = grid[i_hi] - grid[i_lo]
    w = 0.0 if span == 0 else float((x - grid[i_lo]) / span)
    return i_lo, i_hi, w
