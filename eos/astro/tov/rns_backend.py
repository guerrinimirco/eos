"""
rns_backend.py
==============
File-format and process layer for the RNS code (Stergioulas & Friedman 1995,
ApJ 444, 306), which constructs uniformly rotating, axisymmetric relativistic
stellar models by the Komatsu-Eriguchi-Hachisu (1989) self-consistent-field
method with the Cook, Shapiro & Teukolsky (1994) modifications.

This module owns everything that is specific to RNS's on-disk format and its
command line. It knows nothing about sequences, root-finding or physics
targets; that is `eos.astro.tov.rotating`.

*Public API*: :func:`find_rns_binary`, :func:`write_rns_eos`,
:func:`parse_rns_output`, :func:`run_rns`.

Tabulated EOS format
--------------------
RNS reads a header line holding the number of rows, then that many rows of
four CGS columns::

    n_tab
    rho [g/cm^3]   p [dyn/cm^2]   h*c^2 [cm^2/s^2]   n_0 [cm^-3]

`rho` is the total energy density divided by c^2, `p` the pressure, `n_0` the
baryon number density, and `h` the specific relativistic enthalpy defined by

    dh = dp / (e + p),        h(p_surface) = 1/c^2,

i.e. the file column is ``c^2 * integral_{p_1}^{p} dp'/(e(p') + p')``, floored
at the literal value 1.0 in the first row. RNS inverts this table both ways
(`p_at_h` and `e_at_p`); if the three columns are not mutually consistent
through the relation above, the field iteration will not settle and there is no
other symptom. That is why the enthalpy is integrated here rather than
approximated.

Unit traps
----------
These differ between the command line, the table and the printed report, and
are the usual source of silent factor errors:

- ``-e`` takes the central energy density in **g/cm^3**, while the report
  prints ``e_c`` in **10^15 g/cm^3**.
- ``-o`` takes the angular velocity in **10^4 s^-1**, not rad/s, and the
  report prints ``Omega`` in the same units. (This module never uses ``-o``;
  see `eos.astro.tov.rotating` for why.)
- ``-j`` takes the **dimensionless** angular momentum cJ/(G M_sun^2).
- ``M_0`` is a rest mass built with RNS's own fixed baryon mass
  m_B = 1.66e-24 g, 0.03% below this repository's ``m_nucleon_MeV = 931.494``.
  It is reported as RNS computes it, unrescaled.

Hard limits taken from the RNS sources
--------------------------------------
- The table arrays are dimensioned ``[201]`` and are filled without a bound
  check, so **at most 200 rows** may be written. More silently corrupts memory.
- Interpolation is 4-point, so at least 5 rows are required.
- The stellar surface is pinned at rho = 7.8 g/cm^3, p = 1.01e8 dyn/cm^2, so
  the table must extend down to at least that density or the surface is never
  matched and the model does not converge.
- The EOS path is read with ``sscanf(argv[i+1], "%s", file_name)`` into a
  ``char file_name[80]``. A path of 80 characters or more overruns it and
  overwrites the globals that follow, after which the run diverges or hangs
  with no diagnostic. A temporary directory alone is enough to exceed this on
  macOS. :func:`run_rns` therefore always invokes the solver from the directory
  holding the table and passes a short basename, so the path length is bounded
  no matter where the caller put the file.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator

from eos.general.physics_constants import (
    MEV_FM3_TO_DYNE_CM2,
    MEV_FM3_TO_G_CM3,
)
from eos.astro.tov.solver import EOSTable_for_TOV

# ---------------------------------------------------------------------------
# Limits and constants fixed by the RNS sources
# ---------------------------------------------------------------------------

MAX_EOS_ROWS = 200        # arrays are double[201], 1-indexed, unchecked
MIN_EOS_ROWS = 5          # 4-point Lagrange interpolation
RNS_RHO_SURFACE = 7.8     # g/cm^3, the hardwired stellar surface
RNS_P_SURFACE = 1.01e8    # dyn/cm^2
MAX_PATH_CHARS = 79       # char file_name[80], filled by sscanf("%s")

# How far above the solver's surface density a table may start. The check
# exists to catch a missing crust, which is wrong by seven decades, not to
# demand an exact match: BPS bottoms out at 7.86 g/cm^3, 0.8% above the
# surface, and the outermost crust carries no appreciable mass.
SURFACE_TOLERANCE = 10.0

# Short, fixed name the EOS table is linked to inside the run directory, so the
# path handed to the solver can never approach MAX_PATH_CHARS.
_LINK_NAME = "eos.rns"

# RNS's own value of c. Used only for the enthalpy column, which RNS divides by
# exactly this constant on read, so writing it with the same value makes the
# round trip exact instead of accurate to 3e-6.
RNS_C_CGS = 2.9979e10

# Fallback location of the build that ships with this project's working tree.
_RNS_DEFAULT_PATH = (
    "/Users/mircoguerrini/Desktop/Research/rns-main-official/source/rns.v1.1d/rns"
)

# (accuracy, relaxation factor) attempted in order. The first entry is RNS's
# own default and reproduces the values in the code's `examples.test`; the rest
# are the fallbacks that the hand-written scan scripts arrived at empirically.
# Tightening before loosening is deliberate: most non-convergence here is an
# iteration oscillating about the solution, which a smaller step fixes and a
# larger tolerance only hides.
RETRY_LADDER: Tuple[Tuple[float, float], ...] = (
    (1e-5, 1.0),
    (1e-6, 0.8),
    (1e-7, 0.8),
    (1e-4, 0.6),
)


# ---------------------------------------------------------------------------
# Binary discovery
# ---------------------------------------------------------------------------

def find_rns_binary(path: Optional[str] = None) -> str:
    """
    Locate the `rns` executable.

    Resolution order: explicit argument, ``$RNS_BIN``, ``rns`` on ``PATH``,
    then the default build location.

    Parameters
    ----------
    path : str, optional
        Explicit path, checked first.

    Returns
    -------
    str
        Absolute path to an executable `rns`.

    Raises
    ------
    FileNotFoundError
        If no candidate is executable, with the searched locations listed.
    """
    candidates = [path, os.environ.get("RNS_BIN"), shutil.which("rns"),
                  _RNS_DEFAULT_PATH]
    for cand in candidates:
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return os.path.abspath(cand)
    raise FileNotFoundError(
        "Could not find an executable 'rns'. Set $RNS_BIN, put it on PATH, or "
        "pass rns_binary=... . Searched: "
        + ", ".join(repr(c) for c in candidates if c)
    )


def have_rns(path: Optional[str] = None) -> bool:
    """True if an `rns` executable can be found. For test skip conditions."""
    try:
        find_rns_binary(path)
        return True
    except FileNotFoundError:
        return False


# ---------------------------------------------------------------------------
# EOS table writing
# ---------------------------------------------------------------------------

def _enforce_strict_increase(x: np.ndarray, rel: float = 1e-10) -> np.ndarray:
    """
    Nudge a non-decreasing array to be strictly increasing.

    RNS interpolates log(p) against log(h) and log(e) in both directions, which
    divides by differences of the abscissa. A Maxwell construction gives a
    genuinely flat pressure plateau across the mixed phase, and that divides by
    zero. Tilting the plateau by a relative `rel` per step preserves the
    physics far below RNS's 1e-5 working accuracy while keeping every
    interpolation well posed.
    """
    out = np.array(x, dtype=float)
    for i in range(1, out.size):
        floor = out[i - 1] * (1.0 + rel)
        if out[i] <= floor:
            out[i] = floor
    return out


def _specific_enthalpy(P: np.ndarray, eps: np.ndarray,
                       n_fine: int = 16003) -> np.ndarray:
    """
    Integrate ``h = c^2 * integral dp/(e+p)`` on a refined grid.

    Both P and eps are given in the same (arbitrary) unit; the ratio is
    dimensionless and the result carries the c^2 that RNS divides out on read.

    The integration runs on `n_fine` points uniform in ln p, with e(p)
    reconstructed by monotone log-log interpolation -- the same strategy as
    RNS's own converter HnG.c, which uses 16003 divisions for exactly this
    reason. Integrating on the table's own points instead is not merely less
    accurate, it is wrong at the percent level: the integrand 1/(e+p) falls by
    orders of magnitude between adjacent rows of a log-spaced table, which is
    the regime where the trapezoidal rule fails worst. A 1% error in h shifts
    the resulting stellar masses by ~1%.

    The first row is set to the literal 1.0 that RNS expects, matching HnG.c
    (which sets h[1] = 1/c^2) and rns.c's `enthalpy_min`.
    """
    lnP = np.log(P)
    fine = np.linspace(lnP[0], lnP[-1], n_fine)
    P_fine = np.exp(fine)
    eps_fine = np.exp(PchipInterpolator(lnP, np.log(eps))(fine))
    # d/d(ln p) of the integral is p/(e+p)
    h_fine = RNS_C_CGS**2 * cumulative_trapezoid(
        P_fine / (eps_fine + P_fine), fine, initial=0.0
    )
    h = np.interp(lnP, fine, h_fine)
    h[0] = 1.0
    return _enforce_strict_increase(h)


def rns_columns(eos: EOSTable_for_TOV, n_points: int = MAX_EOS_ROWS
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert an EOS table to RNS's four CGS columns.

    The specific enthalpy is integrated on the **full** input table before any
    thinning, so resampling never degrades the ``dh = dp/(e+p)`` relation that
    RNS depends on.

    Thinning, when needed, is log-uniform in energy density. That is the
    distribution of the reference tables shipped with RNS (`eosC` carries 5-14
    points per decade of rho, and is far from uniform in p), and it keeps
    resolution in the 10^14-10^16 g/cm^3 range where the star actually lives.
    Energy density is also the only one of the two that stays monotone across a
    first-order phase transition.

    Parameters
    ----------
    eos : EOSTable_for_TOV
        Table with `P` and `epsilon` in MeV/fm^3 and `nB` in fm^-3, ascending
        in density. Must reach down to the RNS surface, 7.8 g/cm^3.
    n_points : int
        Target number of rows, capped at 200.

    Returns
    -------
    rho, p, h, n0 : ndarray
        Energy density [g/cm^3], pressure [dyn/cm^2], specific enthalpy times
        c^2 [cm^2/s^2], and baryon number density [cm^-3].
    """
    if n_points > MAX_EOS_ROWS:
        raise ValueError(
            f"RNS stores the EOS in double[{MAX_EOS_ROWS + 1}] arrays and reads "
            f"them without a bound check, so at most {MAX_EOS_ROWS} rows may be "
            f"written; got n_points={n_points}."
        )

    P = np.asarray(eos.P, dtype=float)
    eps = np.asarray(eos.epsilon, dtype=float)
    nB = np.asarray(eos.nB, dtype=float)
    if not (P.shape == eps.shape == nB.shape):
        raise ValueError("P, epsilon and nB must have the same length.")

    keep = np.isfinite(P) & np.isfinite(eps) & np.isfinite(nB)
    keep &= (P > 0.0) & (eps > 0.0) & (nB > 0.0)
    P, eps, nB = P[keep], eps[keep], nB[keep]
    if P.size < MIN_EOS_ROWS:
        raise ValueError(
            f"Need at least {MIN_EOS_ROWS} positive, finite rows for RNS's "
            f"4-point interpolation; got {P.size}."
        )

    order = np.argsort(eps)
    P, eps, nB = P[order], eps[order], nB[order]
    eps = _enforce_strict_increase(eps)
    P = _enforce_strict_increase(P)

    rho_min = eps[0] * MEV_FM3_TO_G_CM3
    if rho_min > RNS_RHO_SURFACE * SURFACE_TOLERANCE:
        raise ValueError(
            f"The table starts at {rho_min:.3g} g/cm^3 but RNS pins the stellar "
            f"surface at {RNS_RHO_SURFACE} g/cm^3; without a crust reaching that "
            "density the surface is never matched and no model converges. "
            "Attach one first, e.g. eos.astro.tov.solver.add_crust(table, 'BPS')."
        )

    h = _specific_enthalpy(P, eps)

    if P.size > n_points:
        log_eps = np.log(eps)
        target = np.linspace(log_eps[0], log_eps[-1], n_points)
        eps_out = np.exp(target)
        eps_out[0], eps_out[-1] = eps[0], eps[-1]
        P = np.exp(PchipInterpolator(log_eps, np.log(P))(target))
        h = np.exp(PchipInterpolator(log_eps, np.log(h))(target))
        nB = np.exp(PchipInterpolator(log_eps, np.log(nB))(target))
        eps = eps_out
        P = _enforce_strict_increase(P)
        h = _enforce_strict_increase(h)

    return (eps * MEV_FM3_TO_G_CM3,
            P * MEV_FM3_TO_DYNE_CM2,
            h,
            nB * 1.0e39)


def write_rns_eos(eos: EOSTable_for_TOV, path: str,
                  n_points: int = MAX_EOS_ROWS) -> str:
    """
    Write an EOS table in RNS's tabulated format.

    See the module docstring for the format and :func:`rns_columns` for the
    conversion and thinning rules.

    Parameters
    ----------
    eos : EOSTable_for_TOV
        Table in the repository's fm-based units.
    path : str
        Destination file.
    n_points : int
        Target number of rows, capped at 200.

    Returns
    -------
    str
        The absolute path written.
    """
    rho, p, h, n0 = rns_columns(eos, n_points=n_points)
    with open(path, "w") as fh:
        fh.write(f"{rho.size}\n")
        for row in zip(rho, p, h, n0):
            fh.write("%.10e %.10e %.15e %.15e\n" % row)
    return os.path.abspath(path)


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

# print_table() in rns.c emits `printf("  %6.5e  <label>\n", value)`, one
# quantity per line, with unconverged or undefined entries printed as "---".
_ROW_RE = re.compile(r"^\s*(-?\d[\d.eE+\-]*|---)\s+(\S+)")

# First whitespace-delimited token of the label -> our key. Trailing unit
# parentheses are dropped by the regex.
_LABELS: Dict[str, str] = {
    "e_c": "e_c_1e15",            # 10^15 g/cm^3
    "M": "M",                     # M_sun
    "M_0": "M_0",                 # M_sun
    "R_e": "R_e",                 # km
    "Omega": "Omega_1e4",         # 10^4 s^-1
    "Omega_p": "Omega_p_1e4",     # 10^4 s^-1, mass-shedding limit
    "T/W": "T_over_W",
    "cJ/GM_sun^2": "J",           # dimensionless cJ/(G M_sun^2)
    "I": "I",                     # 10^45 g cm^2
    "Phi_2": "Phi_2",             # 10^42 g cm^2, mass quadrupole
    "h+": "h_plus",               # km
    "h-": "h_minus",              # km
    "Z_p": "Z_p",
    "Z_f": "Z_f",
    "Z_b": "Z_b",
    "omega_c/Omega": "omega_c_over_Omega",
    "r_e": "r_e",                 # km, coordinate equatorial radius
    "r_p/r_e": "r_ratio",
}


def parse_rns_output(text: str) -> Dict[str, float]:
    """
    Parse the report `rns` prints for a single model.

    Values RNS prints as ``---`` become NaN. Unrecognised lines are ignored,
    so the banner and any convergence monitoring are harmless.

    Parameters
    ----------
    text : str
        Captured stdout of one `rns` run with ``-p 1`` (the default).

    Returns
    -------
    dict
        Keys as in `_LABELS`, plus the derived SI/CGS forms ``e_c`` [g/cm^3],
        ``Omega`` and ``Omega_p`` [rad/s] and ``freq``, ``freq_K`` [Hz].
    """
    out: Dict[str, float] = {}
    for line in text.splitlines():
        m = _ROW_RE.match(line)
        if not m:
            continue
        raw, label = m.group(1), m.group(2)
        key = _LABELS.get(label)
        if key is None or key in out:
            continue  # keep the first occurrence; ignore anything unexpected
        out[key] = float("nan") if raw == "---" else float(raw)

    if "e_c_1e15" in out:
        out["e_c"] = out["e_c_1e15"] * 1.0e15
    if "Omega_1e4" in out:
        out["Omega"] = out["Omega_1e4"] * 1.0e4
        out["freq"] = out["Omega"] / (2.0 * np.pi)
    if "Omega_p_1e4" in out:
        out["Omega_K"] = out["Omega_p_1e4"] * 1.0e4
        out["freq_K"] = out["Omega_K"] / (2.0 * np.pi)
    return out


def _looks_converged(res: Dict[str, float]) -> bool:
    """A model is usable if RNS printed a positive, finite mass and radius."""
    for key in ("M", "R_e"):
        val = res.get(key)
        if val is None or not np.isfinite(val) or val <= 0.0:
            return False
    return True


# ---------------------------------------------------------------------------
# Process invocation
# ---------------------------------------------------------------------------

def run_rns(eos_path: str, task: str, e_c: float, *,
            r_ratio: Optional[float] = None,
            rns_binary: Optional[str] = None,
            accuracy: Optional[float] = None,
            cf: Optional[float] = None,
            timeout: float = 30.0,
            ladder: Optional[Sequence[Tuple[float, float]]] = None,
            ) -> Optional[Dict[str, float]]:
    """
    Run one `rns` model, retrying down a ladder of solver settings.

    Only the ``model`` and ``kepler`` tasks should be used. RNS's ``omega``,
    ``jmoment``, ``gmass`` and ``rmass`` tasks wrap the field iteration in a
    second, secant-like loop over the requested quantity, and that outer loop is
    where the code loses convergence; `eos.astro.tov.rotating` reaches those targets
    by inverting a scan instead.

    Parameters
    ----------
    eos_path : str
        Table written by :func:`write_rns_eos`.
    task : {'model', 'kepler', 'static'}
        ``model`` needs `r_ratio`; ``kepler`` and ``static`` do not.
    e_c : float
        Central energy density in **g/cm^3**.
    r_ratio : float, optional
        Axis ratio r_p/r_e in (0, 1]. Required for ``model``.
    rns_binary : str, optional
        Passed to :func:`find_rns_binary`.
    accuracy, cf : float, optional
        Convergence tolerance and relaxation factor for the first attempt. If
        given they replace the first rung of the ladder; the remaining rungs are
        still tried on failure.
    timeout : float
        Seconds allowed per attempt. RNS can loop indefinitely on a model it
        cannot resolve; this is what bounds it, and it replaces the external
        `timeout`/`gtimeout` binary the shell scripts relied on.
    ladder : sequence of (accuracy, cf), optional
        Overrides :data:`RETRY_LADDER`.

    Returns
    -------
    dict or None
        Parsed model, with the settings that produced it under ``accuracy`` and
        ``cf``, or None if every rung failed. Never raises on non-convergence:
        a sweep must survive a bad point.

    Raises
    ------
    ValueError
        For a malformed request (unknown task, missing or out-of-range
        `r_ratio`).
    RuntimeError
        If RNS cannot open the EOS file, which is a caller error rather than a
        convergence failure.
    """
    if task not in ("model", "kepler", "static"):
        raise ValueError(
            f"task must be 'model', 'kepler' or 'static', got {task!r}. The "
            "fixed-quantity tasks are deliberately unsupported; see the "
            "function docstring."
        )
    if task == "model":
        if r_ratio is None:
            raise ValueError("task='model' requires r_ratio.")
        if not (0.0 < r_ratio <= 1.0):
            raise ValueError(f"r_ratio must be in (0, 1], got {r_ratio}.")

    binary = find_rns_binary(rns_binary)
    eos_path = os.path.abspath(eos_path)

    rungs = list(ladder if ladder is not None else RETRY_LADDER)
    if accuracy is not None or cf is not None:
        acc0, cf0 = rungs[0]
        rungs[0] = (accuracy if accuracy is not None else acc0,
                    cf if cf is not None else cf0)

    with tempfile.TemporaryDirectory(prefix="rns_") as workdir:
        # The solver reads the path into a char[80] with sscanf("%s"), so it is
        # linked under a short name here and referenced relatively from the run
        # directory. Passing the absolute path directly overflows that buffer
        # for any table under a macOS temporary directory.
        link = os.path.join(workdir, _LINK_NAME)
        try:
            os.symlink(eos_path, link)
        except OSError:
            shutil.copyfile(eos_path, link)

        for acc, relax in rungs:
            cmd = [binary, "-q", "tab", "-f", _LINK_NAME, "-t", task,
                   "-e", "%.10e" % e_c, "-a", "%.3e" % acc,
                   "-c", "%.4f" % relax, "-d", "0"]
            if task == "model":
                cmd += ["-r", "%.10f" % r_ratio]
            try:
                proc = subprocess.run(cmd, cwd=workdir, capture_output=True,
                                      text=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                continue

            if "cannot open file" in proc.stdout:
                raise RuntimeError(
                    f"rns could not open the EOS file {eos_path!r}."
                )

            res = parse_rns_output(proc.stdout)
            if _looks_converged(res):
                res["accuracy"] = acc
                res["cf"] = relax
                return res

    return None
