"""Attaching a crust to a core table, and the tables that supply one.

A model computes the dense core. Below roughly n_B ~ 0.08 fm^-3 the matter is
no longer uniform -- it is nuclei in a lattice, then a neutron drip regime, then
an outer crust -- and that regime comes from a separate, tabulated calculation.
This module is where the two are joined into the single (P, epsilon, n_B) table
a structure solver integrates.

It is separated from `solver.py` because it is separate physics: nothing here
integrates anything, and nothing in the TOV equations knows where the table's
low-density end came from. What IS delicate is the join. The crust and the core
are independent calculations and disagree on P at the same n_B -- for one
parametrization BPS gives 0.406 MeV/fm^3 at n_B = 0.080 fm^-3 where the core
gives 0.225 -- so splitting on density alone inverts P at the seam. An inverted
P makes epsilon(P) double-valued, drives c_s^2 negative and diverges the
integration. The three join modes below differ precisely in how they avoid
that.

Crust tables are large external data, neither shipped with the package nor
tracked in git; `crust_path` resolves them and says how to supply them when it
cannot.

Units are the repository's fm-based public ones throughout: P and epsilon in
MeV/fm^3, n_B in fm^-3.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.interpolate import PchipInterpolator

from eos import REPO_ROOT
from eos.general.compose import ComposeLookup
from eos.general.physics_constants import MEV_FM3_TO_KM2_INV
from eos.general.state import EOSTable_for_TOV


class MissingCrustData(FileNotFoundError):
    """A named crust table could not be found, with where to put it."""


#: Filename of each named crust table, RELATIVE to a data root. These are
#: large external tables -- a BPS crust and slices of the SFHO CompOSE grid --
#: so they are not shipped with the package and not tracked in git; what is
#: recorded here is their names and where they are looked for.
CRUST_FILES = {
    'BPS': 'BPST0.dat',
    'compose_sfho_nYCT': 'SFHO_Compose/eos.thermo.ns',
    'compose_sfho_nT0_beta': 'SFHO_compose_betaeq_T0.dat',
    'compose_sfho_nYLS_trap': 'SFHO_compose_betaeq_S.dat',
}

#: Environment variable naming where to look. It is a SEARCH PATH, several
#: directories separated by os.pathsep like $PATH, because these tables do not
#: naturally live together: a working setup commonly keeps the crust files in
#: one directory and the CompOSE tree in another.
CRUST_DIR_ENV = "EOS_CRUST_DIR"


def crust_search_path():
    """Directories searched for crust tables, in order.

    ``$EOS_CRUST_DIR`` first, so a caller can always override; then the tables
    shipped beside this module in ``eos/astro/tov/data``; then
    ``<repo>/data/crust``, kept for checkouts that put them there. Nothing here
    reads the filesystem; it is the list of places `crust_path` will try.

    The shipped tables are small enough to live in the package (BPST0.dat is
    under 5 kB), so a fresh clone runs the crusted TOV path with no environment
    set up. That matters because the callers that fall back to no crust move
    R_1.4 by most of a kilometre when they do.
    """
    roots = []
    env = os.environ.get(CRUST_DIR_ENV, "")
    for entry in env.split(os.pathsep):
        if entry:
            roots.append(Path(entry).expanduser())
    roots.append(Path(__file__).resolve().parent / "data")
    roots.append(REPO_ROOT / "data" / "crust")
    return roots


def crust_path(crust_name: str, custom_path: Optional[str] = None) -> Path:
    """Absolute path of a named crust table, or a message saying how to supply it.

    Resolution order: the explicit ``custom_path``, then each directory of
    `crust_search_path`. Raises `MissingCrustData` naming the file, every
    directory tried and the environment variable that adds one -- never a bare
    FileNotFoundError, because the caller's next question is always "where do
    I put it".
    """
    if custom_path is not None:
        given = Path(custom_path).expanduser()
        if not given.is_file():
            raise MissingCrustData(f"crust file not found: {given}")
        return given

    if crust_name not in CRUST_FILES:
        raise ValueError(f"unknown crust {crust_name!r}; "
                         f"known: {sorted(CRUST_FILES)}")

    relative = CRUST_FILES[crust_name]
    roots = crust_search_path()
    for root in roots:
        candidate = root / relative
        if candidate.is_file():
            return candidate
    raise MissingCrustData(
        f"crust table {crust_name!r} needs the file {relative!r}, which was "
        f"not found in any of:\n"
        + "".join(f"    {root}\n" for root in roots)
        + f"These tables are large external data and are neither shipped with "
        f"the package nor tracked in git. Point {CRUST_DIR_ENV} at the "
        f"directory holding them (several may be given, separated by "
        f"{os.pathsep!r}), or pass an explicit path.")


def have_crust(crust_name: str) -> bool:
    """Whether a named crust table can be found. For callers that fall back.

    Use this rather than testing a path yourself. Falling back to no crust is
    not free -- it moves M_max by about 1% -- so the decision deserves to be
    visible at the call site.
    """
    try:
        crust_path(crust_name)
        return True
    except (MissingCrustData, ValueError):
        return False


def load_crust_table(crust_name: str, custom_path: Optional[str] = None,
                     YL: Optional[float] = None, S: Optional[float] = None,
                     T: Optional[float] = None, Y_C: Optional[float] = None) -> EOSTable_for_TOV:
    """
    Load a crust EOS table.

    Args:
        crust_name: 'BPS', 'compose_sfho_nYCT', 'compose_sfho_nT0_beta',
                    'compose_sfho_nYLS_trap', or 'personalized'
        custom_path: Path to custom crust file (required if personalized)
        YL: Lepton fraction (required for 'compose_sfho_nYLS_trap')
        S: Entropy per baryon (required for 'compose_sfho_nYLS_trap')
        T: Temperature [MeV] (required for 'compose_sfho_nYCT')
        Y_C: Charge fraction (required for 'compose_sfho_nYCT')

    Returns:
        EOSTable_for_TOV with crust data
    """
    if crust_name == 'personalized':
        if custom_path is None:
            raise ValueError("Must provide custom_path for personalized crust")
        filepath = custom_path
        # Assume columns: P, epsilon, nB
        return EOSTable_for_TOV.from_file(filepath, columns=(0, 1, 2))

    elif crust_name == 'BPS':
        # BPS file format: P [km^-2], epsilon [km^-2], nB [fm⁻³]
        # Convert to MeV/fm³
        filepath = str(crust_path('BPS'))
        data = np.loadtxt(filepath)
        P_geo = data[:, 0]       # km^-2
        e_geo = data[:, 1]       # km^-2
        nB = data[:, 2]          # fm⁻³

        # Convert to MeV/fm³
        P_mev = P_geo / MEV_FM3_TO_KM2_INV      # km^-2 → MeV/fm³
        e_mev = e_geo / MEV_FM3_TO_KM2_INV         # km^-2 → MeV/fm³

        return EOSTable_for_TOV(P=P_mev, epsilon=e_mev, nB=nB)

    elif crust_name == 'compose_sfho_nYCT':
        # Compose format — full 3D table (n_B, Y_C, T); needs T and Y_C
        if T is None or Y_C is None:
            raise ValueError("Must provide T and Y_C for 'compose_sfho_nYCT' crust")
        filepath = str(crust_path('compose_sfho_nYCT'))
        return _load_compose_crust(filepath, T=T, Y_C=Y_C)

    elif crust_name == 'compose_sfho_nT0_beta':
        # SFHO beta-equilibrium at T=0 - columns: P [MeV/fm³], epsilon [MeV/fm³], nB [fm⁻³]
        filepath = str(crust_path('compose_sfho_nT0_beta'))
        return EOSTable_for_TOV.from_file(filepath, columns=(0, 1, 2))

    elif crust_name == 'compose_sfho_nYLS_trap':
        # SFHO with trapped neutrinos - columns: YL, S, P, epsilon, nB
        if YL is None or S is None:
            raise ValueError("Must provide YL and S for 'compose_sfho_nYLS_trap' crust")
        filepath = str(crust_path('compose_sfho_nYLS_trap'))
        data = np.loadtxt(filepath)
        # Filter rows matching YL and S (with tolerance for float comparison)
        tol = 1e-6
        mask = (np.abs(data[:, 0] - YL) < tol) & (np.abs(data[:, 1] - S) < tol)
        if not np.any(mask):
            raise ValueError(f"No data found for YL={YL}, S={S} in {filepath}")
        filtered = data[mask]
        return EOSTable_for_TOV(P=filtered[:, 2], epsilon=filtered[:, 3], nB=filtered[:, 4])

    else:
        raise ValueError(f"Unknown crust: {crust_name}. Use 'BPS', 'compose_sfho_nYCT', "
                         f"'compose_sfho_nT0_beta', 'compose_sfho_nYLS_trap', or 'personalized'")


_SFHO_LOOKUP_CACHE: dict = {}


def _load_compose_crust(filepath: str, T: float = 0.0, Y_C: float = 0.5) -> EOSTable_for_TOV:
    """Load a (T, Y_C) slice of the SFHO CompOSE table as a crust.

    The full 3-D CompOSE grid is loaded once per directory and cached, so
    repeated calls for different (T, Y_C) reuse the same in-memory tables.

    Args:
        filepath: Path to ``eos.thermo.ns`` (or any file inside the CompOSE
            directory). The directory is inferred from ``filepath``.
        T: Temperature [MeV].
        Y_C: Charge fraction.

    Returns:
        EOSTable_for_TOV with the subnuclear (n_B ≤ 0.16) slice at (T, Y_C).
    """
    compose_dir = str(Path(filepath).parent)
    if compose_dir not in _SFHO_LOOKUP_CACHE:
        _SFHO_LOOKUP_CACHE[compose_dir] = ComposeLookup(compose_dir)
    # Trim to subnuclear range so the crust <-> core blend in add_crust() stays
    # on the side that's actually "crust".
    #
    # The reader hands back plain arrays and this layer wraps them: astro
    # consumes what general produces, which is the direction CLAUDE.md
    # section 1 fixes. It used to be a lazy import inside this function,
    # because the two modules imported each other.
    P, eps, n_B = _SFHO_LOOKUP_CACHE[compose_dir].slice_arrays(
        T, Y_C, n_B_max=0.16)
    return EOSTable_for_TOV(P=P, epsilon=eps, nB=n_B)


def add_crust(
    eos_table: EOSTable_for_TOV,
    crust_name: str = 'No',
    mode: str = 'attach',
    n_transition: Optional[float] = None,
    delta_n: float = 0.01,
    delta_P: float = 0.0,
    custom_crust_path: Optional[str] = None,
    crust_YL: Optional[float] = None,
    crust_S: Optional[float] = None,
    crust_T: Optional[float] = None,
    crust_Y_C: Optional[float] = None,
    save_merged: bool = False,
    output_dir: Optional[str] = None,
    input_filename: Optional[str] = None,
    verbose: bool = False,
) -> EOSTable_for_TOV:
    """
    Add crust to EOS table.

    Args:
        eos_table: High-density EOS table
        crust_name: 'No', 'BPS', 'compose_sfho_nYCT', 'compose_sfho_nT0_beta',
                    'compose_sfho_nYLS_trap', or 'personalized'
        mode: 'attach', 'interpolate', or 'maxwell'
        n_transition: Transition density [fm⁻³] (if None, use crust max)
        delta_n: Width of interpolation region [fm⁻³] (for 'interpolate' mode)
        delta_P: Pressure smoothing width [MeV/fm³] (for 'maxwell' mode)
                 If delta_P=0, sharp Maxwell construction; if delta_P>0, smooth crossover
        custom_crust_path: Path to custom crust file
        crust_YL: Lepton fraction for 'compose_sfho_nYLS_trap' crust
        crust_S: Entropy per baryon for 'compose_sfho_nYLS_trap' crust
        crust_T: Temperature [MeV] for 'compose_sfho_nYCT' crust
        crust_Y_C: Charge fraction for 'compose_sfho_nYCT' crust
        save_merged: Whether to save merged table
        output_dir: Directory for output file
        input_filename: Base name for output file
        verbose: Print transition information

    Returns:
        Merged EOSTable_for_TOV
    """
    if crust_name == 'No':
        return eos_table

    # Load crust
    crust = load_crust_table(crust_name, custom_crust_path,
                             YL=crust_YL, S=crust_S,
                             T=crust_T, Y_C=crust_Y_C)


    if mode == 'attach':
        merged = _attach_crust(eos_table, crust, n_transition)
    elif mode == 'interpolate':
        merged = _interpolate_crust(eos_table, crust, n_transition, delta_n)
    elif mode == 'maxwell':
        merged = _interpolate_crust_maxwell(eos_table, crust, delta_P, verbose)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'attach', 'interpolate', or 'maxwell'")
    
    # Save if requested
    if save_merged and output_dir is not None:
        base = input_filename or "eos"
        base = os.path.splitext(os.path.basename(base))[0]
        outfile = os.path.join(output_dir, f"{base}_withcrust_{crust_name}_{mode}.dat")
        _save_eos_table(merged, outfile)
        print(f"Saved merged EOS to: {outfile}")
    
    return merged


def _drop_nonfinite(table: EOSTable_for_TOV, n_B_min: float = 1e-8) -> EOSTable_for_TOV:
    """Drop rows with non-finite P/ε/n_B, n_B ≤ 0, or n_B below ``n_B_min``.

    The ``n_B_min`` cut prevents the lowest CompOSE rows (n_B ~ 10⁻¹²) from
    inflating μ_B = (P + ε)/n_B to ~10¹² MeV, which can overflow during PCHIP
    polynomial fitting and trigger
    ``ValueError: 'y' must contain only finite values``.

    The returned table is sorted by n_B so PchipInterpolator gets a strictly
    increasing x.
    """
    m = (np.isfinite(table.P) & np.isfinite(table.epsilon)
         & np.isfinite(table.nB) & (table.nB >= n_B_min))
    P, eps, nB = table.P[m], table.epsilon[m], table.nB[m]
    order = np.argsort(nB)
    return EOSTable_for_TOV(P=P[order], epsilon=eps[order], nB=nB[order])


def _attach_crust(eos: EOSTable_for_TOV, crust: EOSTable_for_TOV, n_transition: float) -> EOSTable_for_TOV:
    """Attach a crust below `n_transition`, keeping P monotone across the join.

    Splitting on density alone is not enough. The crust and the core are
    independent calculations that do not agree on P at the same n_B — BPS gives
    P = 0.406 MeV/fm^3 at n_B = 0.080 fm^-3 where DD2 gives 0.225 — so a plain
    concatenation at the transition density steps DOWN in pressure at the join.
    An EOS table whose P decreases with n_B is not an EOS: eps(P) is then
    double-valued, the implied cs^2 is negative there, and a TOV integration
    crossing it diverges. The Numba backend used to turn that into confident
    nonsense; it now returns NaN, which is honest but still not a star.

    Crust points at or above the core's own starting pressure are therefore
    dropped, which joins the two tables at the crossing of their P(n_B) curves
    rather than at a density picked in advance. `n_transition` still sets the
    upper bound on how much crust is considered.
    """
    crust = _drop_nonfinite(crust)
    eos = _drop_nonfinite(eos)
    # Use crust below transition, EOS above
    crust_mask = crust.nB <= n_transition # generate a np list of bools
    eos_mask = eos.nB > n_transition # generate a np list of bools

    P_core = eos.P[eos_mask]
    if P_core.size:
        # Keep only crust points the core branch does not already undercut.
        crust_mask = crust_mask & (crust.P < P_core.min())

    P = np.concatenate([crust.P[crust_mask], P_core])
    epsilon = np.concatenate([crust.epsilon[crust_mask], eos.epsilon[eos_mask]])
    nB = np.concatenate([crust.nB[crust_mask], eos.nB[eos_mask]])

    return EOSTable_for_TOV(P=P, epsilon=epsilon, nB=nB)


def _interpolate_crust(eos: EOSTable_for_TOV, crust: EOSTable_for_TOV, n_transition: float,
                       delta_n: float) -> EOSTable_for_TOV:
    """
    Smooth tanh interpolation between crust and EOS using μB.

    Interpolates P and μB = (P + ε) / n_B, then computes ε = μB * n_B - P.
    This ensures thermodynamic consistency in the transition region.
    """
    crust = _drop_nonfinite(crust)
    eos = _drop_nonfinite(eos)
    if len(crust.nB) < 2 or len(eos.nB) < 2:
        raise ValueError(
            f"Cannot blend: crust has {len(crust.nB)} finite rows, "
            f"EOS has {len(eos.nB)} finite rows (need ≥2 each)."
        )

    n_low = n_transition - delta_n
    n_high = n_transition + delta_n

    # Compute baryon chemical potential μB = (P + ε) / n_B
    muB_crust = (crust.P + crust.epsilon) / crust.nB
    muB_eos = (eos.P + eos.epsilon) / eos.nB

    # Create interpolators for P and μB
    crust_P_interp = PchipInterpolator(crust.nB, crust.P, extrapolate=True)
    crust_muB_interp = PchipInterpolator(crust.nB, muB_crust, extrapolate=True)
    eos_P_interp = PchipInterpolator(eos.nB, eos.P, extrapolate=True)
    eos_muB_interp = PchipInterpolator(eos.nB, muB_eos, extrapolate=True)

    # Unified nB grid with dense sampling in transition region
    nB_crust_below = crust.nB[crust.nB < n_low]
    nB_eos_above = eos.nB[eos.nB > n_high]
    nB_transition = np.linspace(n_low, n_high, 50)
    unified_nB = np.concatenate([nB_crust_below, nB_transition, nB_eos_above])
    unified_nB = np.unique(unified_nB)  # Remove duplicates and sort

    # Blending function: f(n) = 0.5 * (1 + tanh((n - n_transition) / (delta_n/2)))
    def blend(n):
        return 0.5 * (1.0 + np.tanh((n - n_transition) / (delta_n / 2.0)))

    # Interpolate P and μB with blending
    P_merged = np.zeros_like(unified_nB)
    muB_merged = np.zeros_like(unified_nB)

    for i, n in enumerate(unified_nB):
        if n <= n_low:
            P_merged[i] = crust_P_interp(n)
            muB_merged[i] = crust_muB_interp(n)
        elif n >= n_high:
            P_merged[i] = eos_P_interp(n)
            muB_merged[i] = eos_muB_interp(n)
        else:
            f = blend(n)
            P_merged[i] = (1 - f) * crust_P_interp(n) + f * eos_P_interp(n)
            muB_merged[i] = (1 - f) * crust_muB_interp(n) + f * eos_muB_interp(n)

    # Compute ε from μB: ε = μB * n_B - P
    e_merged = muB_merged * unified_nB - P_merged

    return EOSTable_for_TOV(P=P_merged, epsilon=e_merged, nB=unified_nB)


def _interpolate_crust_maxwell(eos: EOSTable_for_TOV, crust: EOSTable_for_TOV,
                                delta_P: float = 0.0, verbose: bool = False) -> EOSTable_for_TOV:
    """
    Maxwell-style interpolation between crust and EOS in pressure space.

    Finds P_trans where μ_crust(P) = μ_eos(P), then interpolates:
    ε(P) = ½[1 - tanh((P - P_trans)/δP)] ε_crust(P) + ½[1 + tanh((P - P_trans)/δP)] ε_eos(P)

    Args:
        eos: High-density EOS table
        crust: Crust EOS table
        delta_P: Pressure smoothing width [MeV/fm³].
                 If 0, sharp Maxwell construction; if >0, smooth crossover.
        verbose: Print transition information

    Returns:
        Merged EOSTable_for_TOV with smooth (or sharp) transition
    """
    crust = _drop_nonfinite(crust)
    eos = _drop_nonfinite(eos)
    if len(crust.nB) < 2 or len(eos.nB) < 2:
        raise ValueError(
            f"Cannot blend: crust has {len(crust.nB)} finite rows, "
            f"EOS has {len(eos.nB)} finite rows (need ≥2 each)."
        )

    # Compute μB = (P + ε) / n_B for both phases
    muB_crust = (crust.P + crust.epsilon) / crust.nB
    muB_eos = (eos.P + eos.epsilon) / eos.nB

    # Create interpolators: μB(P) for both phases
    # Sort by pressure for interpolation
    idx_crust = np.argsort(crust.P)
    idx_eos = np.argsort(eos.P)

    P_crust_sorted = crust.P[idx_crust]
    muB_crust_sorted = muB_crust[idx_crust]
    e_crust_sorted = crust.epsilon[idx_crust]
    nB_crust_sorted = crust.nB[idx_crust]

    P_eos_sorted = eos.P[idx_eos]
    muB_eos_sorted = muB_eos[idx_eos]
    e_eos_sorted = eos.epsilon[idx_eos]
    nB_eos_sorted = eos.nB[idx_eos]

    # Interpolators for μB(P), ε(P), nB(P)
    muB_crust_of_P = PchipInterpolator(P_crust_sorted, muB_crust_sorted, extrapolate=True)
    muB_eos_of_P = PchipInterpolator(P_eos_sorted, muB_eos_sorted, extrapolate=True)
    e_crust_of_P = PchipInterpolator(P_crust_sorted, e_crust_sorted, extrapolate=True)
    e_eos_of_P = PchipInterpolator(P_eos_sorted, e_eos_sorted, extrapolate=True)
    nB_crust_of_P = PchipInterpolator(P_crust_sorted, nB_crust_sorted, extrapolate=True)
    nB_eos_of_P = PchipInterpolator(P_eos_sorted, nB_eos_sorted, extrapolate=True)

    # Find P_trans where μ_crust(P) = μ_eos(P)
    # Use root finder with initial guess at typical crust-core transition (~0.08 fm⁻³)
    from scipy.optimize import root

    def delta_mu(P):
        return float(muB_crust_of_P(P) - muB_eos_of_P(P))

    # Initial guess: P at n_B ~ 0.08 fm⁻³ (typical crust-core boundary)
    n_guess = 0.08  # fm⁻³
    # Find closest point in crust table to get P guess
    idx_guess = np.argmin(np.abs(nB_crust_sorted - n_guess))
    P_guess = P_crust_sorted[idx_guess]

    # Find root using scipy.optimize.root
    try:
        sol = root(delta_mu, P_guess)
        if sol.success:
            P_trans = float(sol.x[0])
        else:
            # Fallback: use the guess
            P_trans = P_guess
            if verbose:
                print(f"  Warning: Root finder did not converge, using P_guess = {P_trans:.4f} MeV/fm³")
    except Exception as e:
        P_trans = P_guess
        if verbose:
            print(f"  Warning: Could not find P_trans ({e}), using P_guess = {P_trans:.4f} MeV/fm³")

    if verbose:
        n1 = float(nB_crust_of_P(P_trans))
        n2 = float(nB_eos_of_P(P_trans))
        mu_trans = float(muB_crust_of_P(P_trans))
        print(f"  Maxwell transition: P_trans = {P_trans:.4f} MeV/fm³")
        print(f"    n_crust(P_trans) = {n1:.4f} fm⁻³")
        print(f"    n_eos(P_trans) = {n2:.4f} fm⁻³")
        print(f"    μB(P_trans) = {mu_trans:.2f} MeV")
        print(f"    δP = {delta_P:.4f} MeV/fm³")

    # Create unified pressure grid
    # Use crust below P_trans - 3*delta_P, EOS above P_trans + 3*delta_P
    P_low = P_trans - 3 * delta_P if delta_P > 0 else P_trans
    P_high = P_trans + 3 * delta_P if delta_P > 0 else P_trans

    P_crust_use = P_crust_sorted[P_crust_sorted <= P_high]
    P_eos_use = P_eos_sorted[P_eos_sorted >= P_low]

    # Unified pressure grid
    P_unified = np.unique(np.concatenate([P_crust_use, P_eos_use]))
    P_unified = np.sort(P_unified)

    # Blending function in pressure space
    def blend(P):
        if delta_P <= 0:
            # Sharp transition (Maxwell)
            return 0.0 if P < P_trans else 1.0
        else:
            # Smooth crossover
            return 0.5 * (1.0 + np.tanh((P - P_trans) / delta_P))

    # Compute ε(P) and nB(P) using blending
    e_merged = np.zeros_like(P_unified)
    nB_merged = np.zeros_like(P_unified)

    for i, P in enumerate(P_unified):
        f = blend(P)
        # ε(P) = (1-f) * ε_crust(P) + f * ε_eos(P)
        e_merged[i] = (1.0 - f) * float(e_crust_of_P(P)) + f * float(e_eos_of_P(P))
        # nB(P) = (1-f) * nB_crust(P) + f * nB_eos(P)
        nB_merged[i] = (1.0 - f) * float(nB_crust_of_P(P)) + f * float(nB_eos_of_P(P))

    return EOSTable_for_TOV(P=P_unified, epsilon=e_merged, nB=nB_merged)


def _save_eos_table(eos: EOSTable_for_TOV, filepath: str) -> None:
    """Save EOS table to file."""
    header = "# Merged EOS Table\n# Columns: P [MeV/fm³], epsilon [MeV/fm³], nB [fm⁻³]"
    np.savetxt(filepath, np.column_stack([eos.P, eos.epsilon, eos.nB]),
               header=header, fmt='%.10e')
