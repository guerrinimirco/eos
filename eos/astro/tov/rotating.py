"""
rotating.py
===========
Uniformly rotating, axisymmetric relativistic stellar models from a tabulated
equation of state: Keplerian (mass-shedding) limits, models at a prescribed
rotation frequency, angular momentum or mass, and constant-frequency or
constant-J sequences.

The stationary, rigidly rotating configurations are computed by the
Komatsu-Eriguchi-Hachisu (1989, MNRAS 237, 355) self-consistent-field method
with the modifications of Cook, Shapiro & Teukolsky (1994, ApJ 422, 227), as
implemented in RNS (Stergioulas & Friedman 1995, ApJ 444, 306). This module
drives that solver; `eos.astro.tov.rns_backend` owns its file format and command
line. Non-rotating models come from `eos.astro.tov.solver` and are untouched here.

*Public API*: :class:`RotatingResult`, :func:`prepare_rotating_eos`,
:func:`kepler_model`, :func:`rratio_scan`, :func:`rotating_model`,
:func:`kepler_sequence`, :func:`rotating_grid`, :func:`static_cross_check`.

Why every target is reached by inverting an axis-ratio scan
-----------------------------------------------------------
The solver offers two kinds of task. One fixes the axis ratio r_p/r_e and
relaxes the metric and fluid fields onto it. The others fix a physical
quantity -- angular velocity, angular momentum, gravitational or rest mass --
by wrapping that same field relaxation in a second, secant-like outer loop.

Only the first kind is reliable. The outer loop of the second kind is where
convergence is lost, and no choice of tolerance or relaxation factor fixes it
because the inner solution it is differencing is itself only converged to that
tolerance.

So this module never asks for a physical target directly. It scans r_p/r_e
between the Keplerian limit and unity, where every model converges, and
inverts the resulting curve in Python. Omega, M, M_0 and J are all monotone in
the axis ratio at fixed central density, so a monotone (PCHIP) interpolant plus
a bracketed root find is well posed. One scan answers every target at once:
asking for a frequency *and* an angular momentum *and* a mass costs one scan,
not three root finds.

The one place this is genuinely ill conditioned is near mass shedding, where
Omega saturates -- it changes by under 1% over the last 5% of axis ratio. A
frequency requested within about a percent of the Keplerian value therefore
does not determine r_p/r_e sharply. The masses and radii remain well
determined; only the axis ratio is loose, and the result carries a note saying
so.

Units
-----
Public arguments and results are in the units used throughout this repository:
energy density MeV/fm^3, pressure MeV/fm^3, baryon density fm^-3, mass M_sun,
radius km, frequency Hz, angular velocity rad/s. Angular momentum is the
dimensionless cJ/(G M_sun^2) that the solver reports. The conversion to the
solver's CGS conventions happens in `eos.astro.tov.rns_backend` and does not leak
across this boundary.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

from eos.general.physics_constants import MEV_FM3_TO_G_CM3
from eos.general.state import EOSTable_for_TOV
from eos.astro.tov.crust import add_crust, load_crust_table
from eos.astro.tov.solver import compute_tov_sequence
from eos.astro.tov import rns_backend
from eos.astro.tov.rns_backend import (
    MAX_EOS_ROWS,
    RNS_RHO_SURFACE,
    have_rns,
    run_rns,
    write_rns_eos,
)

__all__ = [
    "RotatingResult",
    "prepare_rotating_eos",
    "kepler_model",
    "rratio_scan",
    "rotating_model",
    "kepler_sequence",
    "rotating_grid",
    "turning_point",
    "static_cross_check",
    "KEPLER_COLUMNS",
    "GRID_COLUMNS",
    "STATIC_CHECK_COLUMNS",
    "have_rns",
]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class RotatingResult:
    """
    One uniformly rotating equilibrium model.

    A non-converged model carries NaN in every physical field, ``converged =
    False`` and a `note` explaining why, rather than raising: a parameter sweep
    must survive individual bad points.
    """

    e_c: float                    # Central energy density [MeV/fm^3]
    r_ratio: float                # Axis ratio r_p/r_e (1 = non-rotating)
    M: float                      # Gravitational mass [M_sun]
    M_0: float                    # Rest mass [M_sun], with the solver's own
                                  #   baryon mass 1.66e-24 g
    R_e: float                    # Circumferential equatorial radius [km]
    r_e: float                    # Coordinate equatorial radius [km]
    r_p: float                    # Coordinate polar radius [km] = r_e * r_ratio
    Omega: float                  # Angular velocity [rad/s]
    freq: float                   # Spin frequency [Hz]
    Omega_K: float                # Mass-shedding angular velocity at the
                                  #   equator for this model [rad/s]
    freq_K: float                 # ... as a frequency [Hz]
    J: float                      # Angular momentum, dimensionless cJ/(G M_sun^2)
    I: float                      # Moment of inertia [10^45 g cm^2]; NaN when
                                  #   non-rotating, where it is not defined
    T_over_W: float               # Rotational to gravitational binding energy
    Phi_2: float                  # Mass quadrupole moment [10^42 g cm^2]
    Z_p: float                    # Polar redshift
    Z_f: float                    # Forward equatorial redshift
    Z_b: float                    # Backward equatorial redshift
    converged: bool = True
    accuracy: float = float("nan")   # Tolerance that produced this model
    backend: str = "rns"
    note: str = ""

    @classmethod
    def failed(cls, e_c: float, r_ratio: float = float("nan"),
               note: str = "", backend: str = "rns") -> "RotatingResult":
        """A NaN-filled placeholder for a model that could not be computed."""
        nan = float("nan")
        return cls(e_c=e_c, r_ratio=r_ratio, M=nan, M_0=nan, R_e=nan, r_e=nan,
                   r_p=nan, Omega=nan, freq=nan, Omega_K=nan, freq_K=nan, J=nan,
                   I=nan, T_over_W=nan, Phi_2=nan, Z_p=nan, Z_f=nan, Z_b=nan,
                   converged=False, backend=backend, note=note)

    def as_dict(self) -> Dict[str, Union[float, bool, str]]:
        """Flat dict, for `eos.general.table_io.save_table` / `export_csv`."""
        return asdict(self)


def _from_rns(raw: Dict[str, float], backend: str = "rns") -> RotatingResult:
    """Build a :class:`RotatingResult` from a parsed solver report."""
    r_ratio = raw.get("r_ratio", float("nan"))
    r_e = raw.get("r_e", float("nan"))
    return RotatingResult(
        e_c=raw["e_c"] / MEV_FM3_TO_G_CM3,
        r_ratio=r_ratio,
        M=raw.get("M", float("nan")),
        M_0=raw.get("M_0", float("nan")),
        R_e=raw.get("R_e", float("nan")),
        r_e=r_e,
        r_p=r_e * r_ratio,
        Omega=raw.get("Omega", float("nan")),
        freq=raw.get("freq", float("nan")),
        Omega_K=raw.get("Omega_K", float("nan")),
        freq_K=raw.get("freq_K", float("nan")),
        J=raw.get("J", float("nan")),
        I=raw.get("I", float("nan")),
        T_over_W=raw.get("T_over_W", float("nan")),
        Phi_2=raw.get("Phi_2", float("nan")),
        Z_p=raw.get("Z_p", float("nan")),
        Z_f=raw.get("Z_f", float("nan")),
        Z_b=raw.get("Z_b", float("nan")),
        converged=True,
        accuracy=raw.get("accuracy", float("nan")),
        backend=backend,
    )


# Physical quantities that :func:`rotating_model` can solve for, and the
# attribute of :class:`RotatingResult` each one reads. All are monotonically
# increasing as the axis ratio decreases, at fixed central density.
_TARGETS = {
    "freq": "freq",
    "Omega": "Omega",
    "J": "J",
    "M": "M",
    "M_0": "M_0",
}


# ---------------------------------------------------------------------------
# EOS preparation
# ---------------------------------------------------------------------------

# Written tables live for the life of the interpreter, keyed by table content,
# so a sweep over central densities converts and writes the EOS once. The
# directory object is held so its finalizer removes the files at exit.
_TABLE_CACHE: Dict[Tuple, str] = {}
_TABLE_DIR: List[tempfile.TemporaryDirectory] = []


def _session_dir() -> str:
    if not _TABLE_DIR:
        _TABLE_DIR.append(tempfile.TemporaryDirectory(prefix="eos_rotating_"))
    return _TABLE_DIR[0].name


def _extend_to_surface(eos: EOSTable_for_TOV,
                       crust: EOSTable_for_TOV) -> EOSTable_for_TOV:
    """
    Prepend the crust rows that lie below the table's own lowest point.

    :func:`eos.astro.tov.solver.add_crust` discards rows below n_B = 1e-8 fm^-3,
    which keeps the merged table well conditioned for the TOV integrator but
    truncates the BPS crust at ~3e7 g/cm^3. The rotating solver instead
    extrapolates its 4-point interpolation below the first tabulated point to
    find the surface it has pinned at 7.8 g/cm^3, and extrapolating a log-log
    Lagrange polynomial over seven decades does not give anything usable. The
    outermost crust carries no appreciable mass, but the table has to be there.
    """
    mask = ((crust.epsilon < float(np.min(eos.epsilon)))
            & (crust.P < float(np.min(eos.P)))
            & np.isfinite(crust.P) & np.isfinite(crust.epsilon)
            & np.isfinite(crust.nB)
            & (crust.P > 0.0) & (crust.epsilon > 0.0) & (crust.nB > 0.0))
    if not mask.any():
        return eos
    return EOSTable_for_TOV(
        P=np.concatenate([crust.P[mask], eos.P]),
        epsilon=np.concatenate([crust.epsilon[mask], eos.epsilon]),
        nB=np.concatenate([crust.nB[mask], eos.nB]),
    )


def prepare_rotating_eos(eos: Union[str, EOSTable_for_TOV],
                         crust: str = "BPS",
                         crust_mode: str = "attach",
                         n_points: int = MAX_EOS_ROWS,
                         path: Optional[str] = None,
                         **crust_kwargs) -> str:
    """
    Convert an EOS table to the rotating solver's format, attaching a crust.

    The solver pins the stellar surface at 7.8 g/cm^3, so a table that stops at
    the crust-core transition never matches its surface and no model converges.
    A crust is therefore attached with :func:`eos.astro.tov.solver.add_crust` unless
    the table already reaches that density. `BPST0.dat` bottoms out at
    7.86 g/cm^3, which is exactly what is needed.

    Parameters
    ----------
    eos : str or EOSTable_for_TOV
        A table, or a path already in the solver's own format (returned
        unchanged), or a path to a three-column `P, epsilon, nB` file.
    crust : str
        Any name accepted by :func:`eos.astro.tov.solver.add_crust`, or ``'No'`` to
        attach nothing. Ignored when the table already reaches the surface.
    crust_mode : str
        ``'attach'``, ``'interpolate'`` or ``'maxwell'``.
    n_points : int
        Rows to write, capped at 200 by the solver's fixed-size arrays.
    path : str, optional
        Where to write. Defaults to a cached file in a session temporary
        directory, reused for identical tables.
    **crust_kwargs
        Forwarded to :func:`eos.astro.tov.solver.add_crust`.

    Returns
    -------
    str
        Path to the written table.
    """
    if isinstance(eos, str):
        if not os.path.isfile(eos):
            raise FileNotFoundError(f"No such EOS file: {eos!r}")
        with open(eos) as fh:
            first = fh.readline().split()
        # A solver-format table opens with a bare row count.
        if len(first) == 1 and first[0].isdigit():
            return os.path.abspath(eos)
        eos = EOSTable_for_TOV.from_file(eos)

    rho_min = float(np.min(eos.epsilon)) * MEV_FM3_TO_G_CM3
    if rho_min > RNS_RHO_SURFACE and crust != "No":
        crust_table = load_crust_table(
            crust,
            custom_path=crust_kwargs.get("custom_crust_path"),
            YL=crust_kwargs.get("crust_YL"),
            S=crust_kwargs.get("crust_S"),
            T=crust_kwargs.get("crust_T"),
            Y_C=crust_kwargs.get("crust_Y_C"),
        )
        if crust_mode == "attach" and crust_kwargs.get("n_transition") is None:
            # add_crust's 'attach' mode has no default transition density, so
            # join at the top of the crust table -- the same convention as
            # eos/dd2/verify/tov.py, whose N_TRANSITION = 0.08 is where BPS
            # ends.
            crust_kwargs["n_transition"] = float(np.max(crust_table.nB))
        eos = add_crust(eos, crust_name=crust, mode=crust_mode, **crust_kwargs)
        eos = _extend_to_surface(eos, crust_table)

    columns = rns_backend.rns_columns(eos, n_points=n_points)
    if path is not None:
        return write_rns_eos(eos, path, n_points=n_points)

    key = (n_points,) + tuple(arr.tobytes() for arr in columns)
    cached = _TABLE_CACHE.get(key)
    if cached is not None and os.path.isfile(cached):
        return cached
    out = os.path.join(_session_dir(), f"eos_{len(_TABLE_CACHE):04d}.rns")
    write_rns_eos(eos, out, n_points=n_points)
    _TABLE_CACHE[key] = out
    return out


def _e_c_cgs(e_c: float) -> float:
    """Central energy density MeV/fm^3 -> g/cm^3, the solver's input unit."""
    return float(e_c) * MEV_FM3_TO_G_CM3


# ---------------------------------------------------------------------------
# Parallel helper
# ---------------------------------------------------------------------------

def _pmap(func, items, parallel: bool, n_jobs: int = -1):
    """
    Map `func` over `items`, in processes when asked and joblib is available.

    Every model is a separate solver process, so this is clean process
    parallelism. Set `parallel=False` when this call is already inside a
    parallel map -- the same convention as the `tov_parallel` flag of
    :func:`eos.astro.tov.solver.compute_tov_sequence`.
    """
    items = list(items)
    if not parallel or len(items) < 2:
        return [func(x) for x in items]
    try:
        from joblib import Parallel, delayed
    except ImportError:
        return [func(x) for x in items]
    return Parallel(n_jobs=n_jobs)(delayed(func)(x) for x in items)


# ---------------------------------------------------------------------------
# Single models
# ---------------------------------------------------------------------------

def kepler_model(eos_path: str, e_c: float, *,
                 omega_tol: float = 1e-3,
                 **run_kwargs) -> RotatingResult:
    """
    Compute the Keplerian (mass-shedding) model at a given central density.

    This is the fastest rotating configuration that exists at this central
    density: at the mass-shedding limit the fluid angular velocity equals the
    orbital angular velocity of a particle at the equator, so the solver's own
    Omega and Omega_p must agree. That equality is checked here, because a
    Keplerian search that stopped early produces a plausible-looking model that
    is simply not at the limit, with no other symptom.

    Parameters
    ----------
    eos_path : str
        Table from :func:`prepare_rotating_eos`.
    e_c : float
        Central energy density [MeV/fm^3].
    omega_tol : float
        Allowed relative mismatch between Omega and Omega_K.
    **run_kwargs
        Forwarded to :func:`eos.astro.tov.rns_backend.run_rns` (`timeout`,
        `accuracy`, `cf`, `rns_binary`, `ladder`).

    Returns
    -------
    RotatingResult
    """
    raw = run_rns(eos_path, "kepler", _e_c_cgs(e_c), **run_kwargs)
    if raw is None:
        return RotatingResult.failed(
            e_c, note="Keplerian model did not converge at any setting")
    res = _from_rns(raw)
    if np.isfinite(res.Omega) and np.isfinite(res.Omega_K) and res.Omega_K > 0:
        mismatch = abs(res.Omega / res.Omega_K - 1.0)
        if mismatch > omega_tol:
            res.converged = False
            res.note = (f"not at mass shedding: |Omega/Omega_K - 1| = "
                        f"{mismatch:.2e} > {omega_tol:.0e}")
    return res


def _model_at(eos_path: str, e_c: float, r_ratio: float,
              run_kwargs: Dict) -> RotatingResult:
    """One fixed-axis-ratio model. Module level so joblib can pickle it."""
    raw = run_rns(eos_path, "model", _e_c_cgs(e_c), r_ratio=r_ratio,
                  **run_kwargs)
    if raw is None:
        return RotatingResult.failed(
            e_c, r_ratio, note=f"no convergence at r_p/r_e = {r_ratio:.4f}")
    return _from_rns(raw)


def rratio_scan(eos_path: str, e_c: float, *,
                n: int = 20,
                kepler: Optional[RotatingResult] = None,
                parallel: bool = True,
                n_jobs: int = -1,
                **run_kwargs) -> List[RotatingResult]:
    """
    Scan axis ratio from the Keplerian limit to the non-rotating star.

    This is the workhorse every other entry point is built on. It is the part
    of the solver that converges reliably; see the module docstring.

    Parameters
    ----------
    eos_path : str
        Table from :func:`prepare_rotating_eos`.
    e_c : float
        Central energy density [MeV/fm^3].
    n : int
        Number of axis ratios, including both endpoints.
    kepler : RotatingResult, optional
        A Keplerian model already computed at this `e_c`, to save recomputing
        it. Supplies the lower end of the scan.
    parallel : bool
        Run the fixed-ratio models in parallel processes. Set False when
        already inside a parallel map.
    n_jobs : int
        Worker count for joblib; -1 uses every core.
    **run_kwargs
        Forwarded to :func:`eos.astro.tov.rns_backend.run_rns`.

    Returns
    -------
    list of RotatingResult
        Ascending in axis ratio, so descending in rotation rate. The first
        entry is the Keplerian model itself -- it is used rather than a
        fixed-ratio run at the same ratio, which is marginal there. The last
        entry is the non-rotating star. Non-converged points are kept in place
        with ``converged = False`` so the caller can see what was lost.
    """
    if n < 4:
        raise ValueError(f"n must be at least 4 to interpolate, got {n}.")

    kep = kepler if kepler is not None else kepler_model(
        eos_path, e_c, **run_kwargs)
    if not kep.converged or not np.isfinite(kep.r_ratio):
        return [RotatingResult.failed(
            e_c, note="no Keplerian model, cannot bound the scan: " + kep.note)]

    grid = np.linspace(kep.r_ratio, 1.0, n)
    rest = _pmap(lambda r: _model_at(eos_path, e_c, r, run_kwargs),
                 grid[1:], parallel, n_jobs)
    return [kep] + list(rest)


# ---------------------------------------------------------------------------
# Inversion
# ---------------------------------------------------------------------------

def _invert_scan(scan: Sequence[RotatingResult], quantity: str,
                 target: float) -> Tuple[Optional[float], str]:
    """
    Find the axis ratio at which `quantity` equals `target`.

    Returns ``(r_ratio, note)``; `r_ratio` is None when the target lies outside
    the range this central density can produce.
    """
    good = [s for s in scan if s.converged and np.isfinite(getattr(s, quantity))]
    if len(good) < 4:
        return None, f"only {len(good)} converged scan points, need 4"

    r = np.array([s.r_ratio for s in good])
    q = np.array([getattr(s, quantity) for s in good])
    order = np.argsort(r)
    r, q = r[order], q[order]
    keep = np.concatenate(([True], np.diff(r) > 0))
    r, q = r[keep], q[keep]
    if r.size < 4:
        return None, "scan axis ratios are degenerate"

    q_fast, q_slow = q[0], q[-1]      # Keplerian end, non-rotating end
    if target > q_fast:
        return None, (f"{quantity} = {target:.6g} exceeds the Keplerian value "
                      f"{q_fast:.6g} at this central density")
    if target < q_slow:
        return None, (f"{quantity} = {target:.6g} is below the non-rotating "
                      f"value {q_slow:.6g} at this central density")

    spline = PchipInterpolator(r, q)

    def residual(x: float) -> float:
        return float(spline(x)) - target

    end_fast, end_slow = residual(r[0]), residual(r[-1])
    if end_fast * end_slow > 0.0:
        # The range check above passed, so the target is bracketed in exact
        # arithmetic; a positive product here means it coincides with an
        # endpoint to within round-off. Asking for the non-rotating star by
        # requesting zero frequency is the ordinary way to land here.
        root = r[0] if abs(end_fast) <= abs(end_slow) else r[-1]
    else:
        root = brentq(residual, r[0], r[-1], xtol=1e-10)

    note = ""
    span = abs(q_fast - q_slow)
    if span > 0 and (q_fast - target) / span < 0.01:
        note = ("within 1% of the mass-shedding limit, where Omega saturates: "
                "the axis ratio is poorly determined, masses and radii are not")
    return float(root), note


def rotating_model(eos: Union[str, EOSTable_for_TOV], e_c: float, *,
                   freq: Optional[float] = None,
                   Omega: Optional[float] = None,
                   J: Optional[float] = None,
                   M: Optional[float] = None,
                   M_0: Optional[float] = None,
                   r_ratio: Optional[float] = None,
                   kepler: bool = False,
                   n_scan: int = 20,
                   scan: Optional[Sequence[RotatingResult]] = None,
                   backend: str = "rns",
                   parallel: bool = True,
                   n_jobs: int = -1,
                   crust: str = "BPS",
                   **run_kwargs) -> RotatingResult:
    """
    Compute one rotating model at a prescribed target.

    Exactly one of `freq`, `Omega`, `J`, `M`, `M_0`, `r_ratio` or `kepler` must
    be given. A `r_ratio` or `kepler` request runs the solver directly. Any
    other target is reached by scanning the axis ratio and inverting, as
    explained in the module docstring, followed by one confirming run at the
    solved ratio -- so the numbers returned are always the solver's own, never
    an interpolation.

    Parameters
    ----------
    eos : str or EOSTable_for_TOV
        Passed through :func:`prepare_rotating_eos`.
    e_c : float
        Central energy density [MeV/fm^3].
    freq : float, optional
        Target spin frequency [Hz].
    Omega : float, optional
        Target angular velocity [rad/s].
    J : float, optional
        Target angular momentum, dimensionless cJ/(G M_sun^2).
    M, M_0 : float, optional
        Target gravitational or rest mass [M_sun].
    r_ratio : float, optional
        Target axis ratio r_p/r_e in (0, 1].
    kepler : bool
        Compute the mass-shedding model.
    n_scan : int
        Axis ratios used when a scan is needed.
    scan : sequence of RotatingResult, optional
        A scan already computed at this `e_c` and EOS, reused instead of
        recomputing. This is how several targets share one scan.
    backend : {'rns'}
        Rotating-star code to use.
    parallel, n_jobs : bool, int
        Passed to :func:`rratio_scan`.
    crust : str
        Passed to :func:`prepare_rotating_eos`.
    **run_kwargs
        Forwarded to :func:`eos.astro.tov.rns_backend.run_rns`.

    Returns
    -------
    RotatingResult
    """
    if backend != "rns":
        raise NotImplementedError(
            f"backend={backend!r} is not wired up; only 'rns' is available.")

    given = {k: v for k, v in
             (("freq", freq), ("Omega", Omega), ("J", J), ("M", M),
              ("M_0", M_0), ("r_ratio", r_ratio))
             if v is not None}
    if kepler:
        given["kepler"] = True
    if len(given) != 1:
        raise ValueError(
            "Specify exactly one of freq, Omega, J, M, M_0, r_ratio, kepler; "
            f"got {sorted(given) or 'none'}."
        )

    eos_path = prepare_rotating_eos(eos, crust=crust)

    if kepler:
        return kepler_model(eos_path, e_c, **run_kwargs)
    if r_ratio is not None:
        return _model_at(eos_path, e_c, r_ratio, run_kwargs)

    (quantity, target), = given.items()
    if scan is None:
        scan = rratio_scan(eos_path, e_c, n=n_scan, parallel=parallel,
                           n_jobs=n_jobs, **run_kwargs)

    root, note = _invert_scan(scan, _TARGETS[quantity], target)
    if root is None:
        return RotatingResult.failed(e_c, note=note)

    res = _model_at(eos_path, e_c, root, run_kwargs)
    if note:
        res.note = note
    return res


# ---------------------------------------------------------------------------
# Sequences and grids
# ---------------------------------------------------------------------------

KEPLER_COLUMNS: Tuple[str, ...] = (
    "e_c", "M", "M_0", "R_e", "Omega", "freq", "J", "I", "T_over_W", "r_ratio",
)

GRID_COLUMNS: Tuple[str, ...] = (
    "e_c", "r_ratio", "M", "M_0", "R_e", "Omega", "freq", "J", "I",
    "T_over_W", "freq_K",
)

STATIC_CHECK_COLUMNS: Tuple[str, ...] = (
    "e_c", "M_rot", "R_rot", "M_tov", "R_tov", "dM_rel", "dR_rel",
)


def _as_array(results: Sequence[RotatingResult],
              columns: Sequence[str]) -> np.ndarray:
    """Stack results into a float array with the given column order."""
    return np.array([[getattr(r, c) for c in columns] for r in results],
                    dtype=float)


def kepler_sequence(eos: Union[str, EOSTable_for_TOV],
                    e_c_grid: Sequence[float], *,
                    parallel: bool = True,
                    n_jobs: int = -1,
                    crust: str = "BPS",
                    **run_kwargs) -> np.ndarray:
    """
    Keplerian limit as a function of central density.

    Parameters
    ----------
    eos : str or EOSTable_for_TOV
        Passed through :func:`prepare_rotating_eos`.
    e_c_grid : sequence of float
        Central energy densities [MeV/fm^3].
    parallel, n_jobs : bool, int
        Parallelise over central densities.
    crust : str
        Passed to :func:`prepare_rotating_eos`.
    **run_kwargs
        Forwarded to :func:`eos.astro.tov.rns_backend.run_rns`.

    Returns
    -------
    ndarray, shape (len(e_c_grid), 10)
        Columns :data:`KEPLER_COLUMNS`. Rows for central densities where the
        limit could not be found are NaN except for `e_c`.
    """
    eos_path = prepare_rotating_eos(eos, crust=crust)
    results = _pmap(lambda ec: kepler_model(eos_path, ec, **run_kwargs),
                    e_c_grid, parallel, n_jobs)
    return _as_array(results, KEPLER_COLUMNS)


def rotating_grid(eos: Union[str, EOSTable_for_TOV],
                  e_c_grid: Sequence[float], *,
                  freq_grid: Optional[Sequence[float]] = None,
                  J_grid: Optional[Sequence[float]] = None,
                  M_grid: Optional[Sequence[float]] = None,
                  M_0_grid: Optional[Sequence[float]] = None,
                  r_ratio_grid: Optional[Sequence[float]] = None,
                  n_scan: int = 20,
                  parallel: bool = True,
                  n_jobs: int = -1,
                  crust: str = "BPS",
                  **run_kwargs) -> np.ndarray:
    """
    Constant-frequency, constant-J, constant-mass, constant-baryonic-mass or
    constant-axis-ratio sequences over a grid of central densities.

    One axis-ratio scan is run per central density and every requested target
    is read off it by monotone interpolation. Unlike :func:`rotating_model`,
    the returned values are therefore interpolated between converged models
    rather than each being a converged model in its own right. With the default
    `n_scan` that costs well under a part in a thousand, and it is what makes a
    full grid affordable; call :func:`rotating_model` for a point that has to
    be exact.

    Parameters
    ----------
    eos : str or EOSTable_for_TOV
        Passed through :func:`prepare_rotating_eos`.
    e_c_grid : sequence of float
        Central energy densities [MeV/fm^3].
    freq_grid : sequence of float, optional
        Spin frequencies [Hz].
    J_grid : sequence of float, optional
        Angular momenta, dimensionless cJ/(G M_sun^2).
    M_grid : sequence of float, optional
        Gravitational masses [M_sun].
    M_0_grid : sequence of float, optional
        Rest (baryonic) masses [M_sun]. This is the sequence a star actually
        evolves along as it spins down, since baryon number is conserved while
        the gravitational mass is not; a constant-M_0 curve that turns over is
        what makes a supramassive star collapse.
    r_ratio_grid : sequence of float, optional
        Axis ratios; these need no inversion and are interpolated directly.
    n_scan : int
        Axis ratios per central density.
    parallel, n_jobs : bool, int
        Parallelise over central densities. The inner scans then run serially
        so the cores are not oversubscribed.
    crust : str
        Passed to :func:`prepare_rotating_eos`.
    **run_kwargs
        Forwarded to :func:`eos.astro.tov.rns_backend.run_rns`.

    Returns
    -------
    ndarray, shape (len(e_c_grid) * n_targets, 11)
        Columns :data:`GRID_COLUMNS`. A target beyond the Keplerian limit of
        its central density gives a NaN row rather than an error, so a grid
        that overhangs the limit stays rectangular and plottable.
    """
    requested = [(k, v) for k, v in (("freq", freq_grid), ("J", J_grid),
                                     ("M", M_grid), ("M_0", M_0_grid),
                                     ("r_ratio", r_ratio_grid))
                 if v is not None]
    if len(requested) != 1:
        raise ValueError(
            "Give exactly one of freq_grid, J_grid, M_grid, M_0_grid, "
            f"r_ratio_grid; got {[k for k, _ in requested] or 'none'}."
        )
    (quantity, targets), = requested
    targets = np.asarray(targets, dtype=float)

    eos_path = prepare_rotating_eos(eos, crust=crust)
    inner_parallel = not parallel   # never nest process pools

    def one_e_c(e_c: float) -> np.ndarray:
        scan = rratio_scan(eos_path, e_c, n=n_scan, parallel=inner_parallel,
                           n_jobs=n_jobs, **run_kwargs)
        good = [s for s in scan if s.converged]
        rows = np.full((targets.size, len(GRID_COLUMNS)), np.nan)
        rows[:, 0] = e_c
        if len(good) < 4:
            return rows

        order = np.argsort([s.r_ratio for s in good])
        r_sorted = np.array([good[i].r_ratio for i in order])
        keep = np.concatenate(([True], np.diff(r_sorted) > 0))
        r = r_sorted[keep]
        if r.size < 4:
            return rows

        # Per column, because they are not all defined everywhere on the scan:
        # the moment of inertia is reported as undefined for the non-rotating
        # endpoint, where the star has no rotation to have inertia about.
        splines = {}
        for col in GRID_COLUMNS[2:]:
            y = np.array([getattr(good[i], col) for i in order])[keep]
            finite = np.isfinite(y)
            if finite.sum() >= 4:
                splines[col] = PchipInterpolator(r[finite], y[finite])

        for k, target in enumerate(targets):
            if quantity == "r_ratio":
                root = target if r[0] <= target <= r[-1] else None
            else:
                root, _ = _invert_scan(scan, _TARGETS[quantity], target)
            if root is None:
                continue
            rows[k, 1] = root
            for j, col in enumerate(GRID_COLUMNS[2:], start=2):
                if col in splines:
                    rows[k, j] = float(splines[col](root))
        return rows

    blocks = _pmap(one_e_c, e_c_grid, parallel, n_jobs)
    return np.vstack(blocks)


def turning_point(x: Sequence[float], M: Sequence[float],
                  precision: float = 1e-3
                  ) -> Tuple[np.ndarray, float, float]:
    """
    Locate the secular-instability point of a **constant-J** sequence.

    Friedman, Ipser & Sorkin (1988, ApJ 325, 722) proved that along a sequence
    of uniformly rotating equilibria with fixed angular momentum, the point
    where the gravitational mass is stationary in central density marks the
    onset of secular instability to axisymmetric perturbations: models on the
    low-density side of that turning point are stable, models beyond it are
    not. Because dM = Omega dJ + mu dM_0 at fixed entropy, the turning points of
    M and of the rest mass M_0 along the same sequence coincide, so either
    column may be passed.

    Two limitations of the theorem, both worth knowing before quoting a number:
    it is *sufficient* for instability but not necessary, and the true neutral
    point sits marginally on the stable side of the turning point -- by a few
    tenths of a percent in central density for uniform rotation (Takami,
    Rezzolla & Yoshida 2011, MNRAS 416, L1). And it applies to sequences of
    **constant J** (or constant M_0), *not* constant angular velocity: a
    constant-frequency sequence has a mass maximum too, and it is not a
    stability boundary.

    The **first** maximum is taken, not the largest. With a first-order phase
    transition the mass can turn over, dip and rise again into a second stable
    (twin) branch whose peak may be the higher of the two; the first branch
    still becomes unstable at its own turning point. A maximum counts only if
    the mass later falls at least `precision` below it, which is what keeps
    solver noise on a nearly flat sequence from being read as a turning point.

    Parameters
    ----------
    x : sequence of float
        Sequence parameter, ascending: central energy density [MeV/fm^3] or
        central baryon density [fm^-3]. The two are monotonically related, so
        the turning point is the same model either way.
    M : sequence of float
        Gravitational (or rest) mass [M_sun] along the sequence, same length as
        `x`. Non-converged entries may be NaN and are skipped.
    precision : float
        Mass drop [M_sun] required after a maximum for it to count.

    Returns
    -------
    stable : ndarray of bool
        Aligned with the input: True for a converged model at or below the
        turning point. Non-converged (NaN) entries are False.
    x_crit, M_crit : float
        Sequence parameter and mass at the turning point, refined by monotone
        interpolation between the bracketing grid points. Both NaN when the
        sequence has no turning point on this grid, in which case every
        converged model is marked stable -- extend the grid to higher density
        rather than concluding that none of them destabilises.
    """
    x = np.asarray(x, dtype=float)
    M = np.asarray(M, dtype=float)
    if x.shape != M.shape:
        raise ValueError(f"x and M must have the same shape, got {x.shape} "
                         f"and {M.shape}.")

    finite = np.isfinite(x) & np.isfinite(M)
    order = np.argsort(x[finite])
    xf, Mf = x[finite][order], M[finite][order]

    idx = None
    for i in range(1, xf.size - 1):
        if (Mf[i] >= Mf[i - 1] and Mf[i] >= Mf[i + 1]
                and Mf[i] - Mf[i + 1:].min() > precision):
            idx = i
            break

    if idx is None:
        return finite.copy(), float("nan"), float("nan")

    # Refine on the three bracketing points only, by the vertex of the parabola
    # through them. A monotone (PCHIP) interpolant cannot help here: it is flat
    # by construction at an extremum node, so its maximum is that node and the
    # answer would never beat the grid spacing. A wider window is no better --
    # it reaches the rise of a twin branch and climbs out of the maximum it was
    # sent to find.
    coeff = np.polyfit(xf[idx - 1:idx + 2], Mf[idx - 1:idx + 2], 2)
    if coeff[0] < 0.0:
        x_crit = float(np.clip(-coeff[1] / (2.0 * coeff[0]),
                               xf[idx - 1], xf[idx + 1]))
    else:
        x_crit = float(xf[idx])      # degenerate triple, keep the grid point
    M_crit = float(np.polyval(coeff, x_crit))
    return finite & (x <= x_crit), x_crit, M_crit


def static_cross_check(eos: Union[str, EOSTable_for_TOV],
                       e_c_grid: Sequence[float], *,
                       crust: str = "BPS",
                       parallel: bool = True,
                       n_jobs: int = -1,
                       **run_kwargs) -> np.ndarray:
    """
    Compare non-rotating models from the rotating code against the TOV solver.

    Both codes are given the same EOS, the rotating one at unit axis ratio. The
    two should agree but will not agree exactly: the rotating code pins its own
    stellar surface at 7.8 g/cm^3 and works from a table thinned to at most 200
    rows on a spectral-like grid, while :func:`eos.astro.tov.solver.compute_tov_sequence`
    integrates the supplied table directly. A disagreement of a few tenths of a
    percent is the expected level; a systematically larger one means the table
    conversion is wrong.

    Parameters
    ----------
    eos : str or EOSTable_for_TOV
        Passed through :func:`prepare_rotating_eos` and, unconverted, to the
        TOV solver.
    e_c_grid : sequence of float
        Central energy densities [MeV/fm^3].
    crust : str
        Crust applied to both codes, so the comparison is like for like.
    parallel, n_jobs : bool, int
        Parallelise the rotating-code models.
    **run_kwargs
        Forwarded to :func:`eos.astro.tov.rns_backend.run_rns`.

    Returns
    -------
    ndarray, shape (len(e_c_grid), 7)
        Columns :data:`STATIC_CHECK_COLUMNS`.
    """
    if isinstance(eos, str):
        eos = EOSTable_for_TOV.from_file(eos)
    eos_path = prepare_rotating_eos(eos, crust=crust)

    rot = _pmap(lambda ec: _model_at(eos_path, ec, 1.0, run_kwargs),
                e_c_grid, parallel, n_jobs)

    n_transition = (None if crust == "No"
                    else float(np.max(load_crust_table(crust).nB)))
    tov = compute_tov_sequence(
        eos, np.asarray(e_c_grid, dtype=float),
        add_crust_table=crust, add_crust_mode="attach",
        n_transition=n_transition,
        compute_baryonic_mass=False, compute_tidal=False, verbose=False,
    )
    M_tov, R_tov = tov[:, 4], tov[:, 3]

    out = np.full((len(rot), len(STATIC_CHECK_COLUMNS)), np.nan)
    out[:, 0] = np.asarray(e_c_grid, dtype=float)
    out[:, 1] = [r.M for r in rot]
    out[:, 2] = [r.R_e for r in rot]
    out[:, 3] = M_tov
    out[:, 4] = R_tov
    out[:, 5] = np.abs(out[:, 1] / out[:, 3] - 1.0)
    out[:, 6] = np.abs(out[:, 2] / out[:, 4] - 1.0)
    return out
