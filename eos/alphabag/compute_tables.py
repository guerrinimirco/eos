"""
alphabag_compute_tables.py
==========================
User-friendly script for generating AlphaBag quark EOS tables.

Supports two phases:
- 'unpaired': Normal quark matter with perturbative α-corrections
- 'cfl': Color-Flavor Locked phase with pairing gap Δ

Equilibrium types for unpaired phase:
- 'beta_eq': Beta equilibrium with charge neutrality
- 'fixed_yc': Fixed charge fraction Y_C
- 'fixed_yc_ys': Fixed charge fraction Y_C and strangeness fraction Y_S

For CFL phase, flavor-locking constrains n_u = n_d = n_s.

Usage:
    from eos.alphabag.compute_tables import compute_alphabag_table, AlphaBagTableSettings
    
    settings = AlphaBagTableSettings(
        phase='unpaired',
        equilibrium='beta_eq',
        T_values=[10.0, 30.0, 50.0],
        n_B_values=np.linspace(0.1, 10, 100) * 0.16
    )
    results = compute_alphabag_table(settings)
"""
import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Union

from eos.alphabag.parameters import AlphaBagParams, get_alphabag_default, get_alphabag_custom

from eos.alphabag.eos import (
    AlphaBagEOSResult, CFLEOSResult,
    solve_alphabag_beta_eq, solve_alphabag_fixed_yc, solve_alphabag_fixed_yc_ys,
    solve_cfl
)
from eos.alphabag.thermodynamics_quarks import CFLThermo


# =============================================================================
# SETTINGS DATACLASS
# =============================================================================
@dataclass
class AlphaBagTableSettings:
    """
    Configuration for AlphaBag EOS table generation.
    
    Phases:
    - 'unpaired': Normal quark matter (αBag)
    - 'cfl': Color-Flavor Locked quark matter
    
    Equilibrium types (unpaired only):
    - 'beta_eq': Beta equilibrium with charge neutrality
    - 'fixed_yc': Fixed charge fraction Y_C
    - 'fixed_yc_ys': Fixed Y_C and Y_S
    """
    # Model parameters (if None, uses defaults)
    params: Optional[AlphaBagParams] = None
    
    # Easy-access parameters (used if params is None)
    alpha: Optional[float] = None  # QCD coupling α_s
    B4: Optional[float] = None     # Bag constant B^(1/4) in MeV
    m_s: Optional[float] = None    # Strange quark mass in MeV
    
    # Phase selection
    phase: str = 'unpaired'  # 'unpaired' or 'cfl'
    
    # Equilibrium type (for unpaired phase)
    equilibrium: str = 'beta_eq'
    
    # CFL gap (for CFL phase)
    Delta0_values: List[float] = field(default_factory=lambda: [100.0])  # MeV
    
    # Grid definition
    n_B_values: np.ndarray = field(default_factory=lambda: np.linspace(0.1, 10, 100) * 0.16)
    T_values: List[float] = field(default_factory=lambda: [10.0])
    
    # Constraint parameters (for unpaired phase)
    Y_C_values: List[float] = field(default_factory=lambda: [0.0])
    Y_S_values: List[float] = field(default_factory=lambda: [0.0])
    
    # Options
    include_photons: bool = True
    include_gluons: bool = True
    include_electrons: bool = False
    include_thermal_neutrinos: bool = True  
    
    # Output control
    print_results: bool = True
    print_first_n: int = 5
    print_errors: bool = True
    print_timing: bool = True
    
    # File output
    save_to_file: bool = False
    output_filename: Optional[str] = None


# =============================================================================
# TABLE GENERATOR FOR UNPAIRED PHASE
# =============================================================================
def _compute_unpaired_table(settings: AlphaBagTableSettings) -> Dict[Tuple, List[AlphaBagEOSResult]]:
    """Compute table for unpaired αBag quark matter."""
    # Build params from settings or use default
    if settings.params is not None:
        params = settings.params
    elif settings.alpha is not None or settings.B4 is not None or settings.m_s is not None:
        params = get_alphabag_custom(
            alpha=settings.alpha or 0.3,
            B4=settings.B4 or 165.0,
            m_s=settings.m_s or 150.0
        )
    else:
        params = get_alphabag_default()
    
    eq_type = settings.equilibrium.lower()
    
    n_B_arr = np.asarray(settings.n_B_values)
    T_list = list(settings.T_values)
    
    # Build parameter grid
    if eq_type == 'beta_eq':
        grid_params = [(T,) for T in T_list]
        param_names = ['T']
    elif eq_type == 'fixed_yc':
        Y_C_list = list(settings.Y_C_values)
        grid_params = [(T, Y_C) for T in T_list for Y_C in Y_C_list]
        param_names = ['T', 'Y_C']
    elif eq_type == 'fixed_yc_ys':
        Y_C_list = list(settings.Y_C_values)
        Y_S_list = list(settings.Y_S_values)
        grid_params = [(T, Y_C, Y_S) for T in T_list for Y_C in Y_C_list for Y_S in Y_S_list]
        param_names = ['T', 'Y_C', 'Y_S']
    else:
        raise ValueError(f"Unknown equilibrium type: {eq_type}")
    
    n_points = len(n_B_arr)
    n_tables = len(grid_params)
    
    if settings.print_results:
        print(f"\nPhase: UNPAIRED (αBag)")
        print(f"Parameters: B^1/4={params.B4} MeV, α_s={params.alpha}, m_s={params.m_s} MeV")
        print(f"Equilibrium: {eq_type}")
        print(f"Density grid: {n_points} points, n_B = [{n_B_arr[0]:.4e}, {n_B_arr[-1]:.4e}] fm⁻³")
        print(f"Parameter grid: {n_tables} tables\n")
    
    all_results = {}
    total_start = time.time()
    
    for idx, grid_param in enumerate(grid_params):
        T = grid_param[0]
        Y_C = grid_param[1] if len(grid_param) > 1 else None
        Y_S = grid_param[2] if len(grid_param) > 2 else None
        
        if settings.print_results:
            print("-" * 60)
            param_str = f"T={T} MeV"
            if Y_C is not None:
                param_str += f", Y_C={Y_C}"
            if Y_S is not None:
                param_str += f", Y_S={Y_S}"
            print(f"[{idx+1}/{n_tables}] {param_str}")
        
        start_time = time.time()
        results = []
        guess = None
        
        for i, n_B in enumerate(n_B_arr):
            # Call appropriate solver
            if eq_type == 'beta_eq':
                r = solve_alphabag_beta_eq(
                    n_B, T, params,
                    include_photons=settings.include_photons,
                    include_gluons=settings.include_gluons,
                    include_thermal_neutrinos=settings.include_thermal_neutrinos,
                    initial_guess=guess
                )
            elif eq_type == 'fixed_yc':
                r = solve_alphabag_fixed_yc(
                    n_B, Y_C, T, params,
                    include_photons=settings.include_photons,
                    include_gluons=settings.include_gluons,
                    include_electrons=settings.include_electrons,
                    include_thermal_neutrinos=settings.include_thermal_neutrinos,
                    initial_guess=guess
                )
            elif eq_type == 'fixed_yc_ys':
                r = solve_alphabag_fixed_yc_ys(
                    n_B, Y_C, Y_S, T, params,
                    include_photons=settings.include_photons,
                    include_gluons=settings.include_gluons,
                    include_electrons=settings.include_electrons,
                    include_thermal_neutrinos=settings.include_thermal_neutrinos,
                    initial_guess=guess
                )
            
            results.append(r)
            
            # Update guess for next point
            if r.converged:
                guess = np.array([r.mu_u, r.mu_d, r.mu_s, r.mu_e])[:3 if eq_type != 'beta_eq' else 4]
            
            # Print progress
            if settings.print_results:
                should_print = (i < settings.print_first_n or 
                               (settings.print_errors and not r.converged))
                if should_print:
                    status = "OK" if r.converged else "FAILED"
                    print(f"[{i:4d}] n_B={n_B:.4e} [{status}] P={r.P_total:.2f} Y_C={r.Y_C:.4f}")
        
        elapsed = time.time() - start_time
        all_results[grid_param] = results
        
        if settings.print_timing:
            n_converged = sum(1 for r in results if r.converged)
            print(f"\n  {elapsed:.2f}s, Converged: {n_converged}/{n_points} ({100*n_converged/n_points:.1f}%)")
    
    return all_results


# =============================================================================
# TABLE GENERATOR FOR CFL PHASE
# =============================================================================
def _compute_cfl_table(settings: AlphaBagTableSettings) -> Dict[Tuple, List[CFLThermo]]:
    """Compute table for CFL quark matter."""
    # Build params from settings or use default
    if settings.params is not None:
        params = settings.params
    elif settings.alpha is not None or settings.B4 is not None or settings.m_s is not None:
        params = get_alphabag_custom(
            alpha=settings.alpha or 0.3,
            B4=settings.B4 or 165.0,
            m_s=settings.m_s or 150.0
        )
    else:
        params = get_alphabag_default()
    
    n_B_arr = np.asarray(settings.n_B_values)
    T_list = list(settings.T_values)
    Delta0_list = list(settings.Delta0_values)
    
    grid_params = [(T, Delta0) for T in T_list for Delta0 in Delta0_list]
    
    n_points = len(n_B_arr)
    n_tables = len(grid_params)
    
    if settings.print_results:
        print(f"\nPhase: CFL (Color-Flavor Locked)")
        print(f"Parameters: B^1/4={params.B4} MeV, m_s={params.m_s} MeV")
        print(f"Gap values: Δ₀ = {Delta0_list} MeV")
        print(f"Density grid: {n_points} points, n_B = [{n_B_arr[0]:.4e}, {n_B_arr[-1]:.4e}] fm⁻³")
        print(f"Parameter grid: {n_tables} tables\n")
    
    all_results = {}
    total_start = time.time()
    
    for idx, (T, Delta0) in enumerate(grid_params):
        if settings.print_results:
            print("-" * 60)
            print(f"[{idx+1}/{n_tables}] T={T} MeV, Δ₀={Delta0} MeV")
        
        start_time = time.time()
        results = []
        guess = None
        
        for i, n_B in enumerate(n_B_arr):
            # CFL constraint: n_u = n_d = n_s = n_B (charge-neutral by construction)
            r = solve_cfl(
                n_B, T=T, Delta0=Delta0, params=params,
                include_photons=settings.include_photons,
                include_gluons=settings.include_gluons,
                initial_guess=guess
            )
            results.append(r)
            
            # Update guess for next point
            if r.converged:
                guess = np.array([r.mu_u, r.mu_d, r.mu_s])
            
            # Print progress
            if settings.print_results and i < settings.print_first_n:
                mu_avg = (r.mu_u + r.mu_d + r.mu_s) / 3.0
                print(f"[{i:4d}] n_B={n_B:.4e} μ={mu_avg:.2f} Δ={r.Delta:.2f} P={r.P_total:.2f}")
        
        elapsed = time.time() - start_time
        all_results[(T, Delta0)] = results
        
        if settings.print_timing:
            n_converged = sum(1 for r in results if r.converged)
            print(f"\n  {elapsed:.2f}s, Converged: {n_converged}/{n_points} ({100*n_converged/n_points:.1f}%)")
    
    return all_results


# =============================================================================
# MAIN TABLE GENERATOR
# =============================================================================
def compute_alphabag_table(settings: AlphaBagTableSettings) -> Dict[Tuple, List]:
    """
    Compute AlphaBag EOS table.
    
    Dispatches to unpaired or CFL table generator based on settings.phase.
    
    Args:
        settings: AlphaBagTableSettings configuration
        
    Returns:
        Dictionary mapping parameter tuple to list of results
    """
    phase = settings.phase.lower()
    
    if settings.print_results:
        print("=" * 70)
        print("AlphaBag EOS TABLE GENERATOR")
        print("=" * 70)
    
    total_start = time.time()
    
    if phase == 'unpaired':
        all_results = _compute_unpaired_table(settings)
    elif phase == 'cfl':
        all_results = _compute_cfl_table(settings)
    else:
        raise ValueError(f"Unknown phase: {phase}. Use 'unpaired' or 'cfl'.")
    
    total_elapsed = time.time() - total_start
    
    if settings.print_timing:
        n_total = sum(len(v) for v in all_results.values())
        print("\n" + "=" * 70)
        print(f"Total: {total_elapsed:.2f}s, {n_total} points, Avg: {total_elapsed*1000/n_total:.1f}ms/pt")
    
    if settings.save_to_file:
        save_alphabag_results(all_results, settings, phase)
    
    return all_results


# =============================================================================
# SAVE RESULTS
# =============================================================================
def save_alphabag_results(
    all_results: Dict[Tuple, List],
    settings: AlphaBagTableSettings,
    phase: str
):
    """Save results to file."""
    # Match the logic from _compute_unpaired_table for consistency
    if settings.params is not None:
        params = settings.params
    elif settings.alpha is not None or settings.B4 is not None or settings.m_s is not None:
        params = get_alphabag_custom(
            alpha=settings.alpha if settings.alpha is not None else 0.3,
            B4=settings.B4 if settings.B4 is not None else 165.0,
            m_s=settings.m_s if settings.m_s is not None else 150.0
        )
    else:
        params = get_alphabag_default()
    
    if settings.output_filename:
        filename = settings.output_filename
    else:
        from eos import REPO_ROOT
        filename = str(REPO_ROOT / "output" / f"alphabag_{phase}_B{int(params.B4)}_alpha{params.alpha}.dat")
    
    import os
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        f.write(f"# AlphaBag EOS Table ({phase})\n")
        f.write(f"# Parameters: B^1/4={params.B4} MeV, α_s={params.alpha}, m_s={params.m_s} MeV\n")
        f.write(f"# Equilibrium: {settings.equilibrium}\n")
        
        # Write component flags
        components = []
        if settings.include_photons:
            components.append("photons")
        if settings.include_gluons:
            components.append("gluons")
        if settings.include_electrons:
            components.append("electrons")
        if settings.include_thermal_neutrinos:
            components.append("thermal_neutrinos")
        f.write(f"# Components: {', '.join(components) if components else 'quarks only'}\n")
        
        if phase == 'cfl':
            columns = ['n_B', 'T', 'Delta0', 'Delta', 'mu_u', 'mu_d', 'mu_s', 'P', 'e', 's', 'f']
        else:
            eq_type = settings.equilibrium.lower()
            if eq_type == 'beta_eq':
                columns = ['n_B', 'T', 'mu_u', 'mu_d', 'mu_s', 'mu_e', 'Y_u', 'Y_d', 'Y_s', 
                           'P_total', 'e_total', 's_total', 'converged']
            else:
                columns = ['n_B', 'Y_C', 'T', 'mu_u', 'mu_d', 'mu_s', 'Y_u', 'Y_d', 'Y_s',
                           'P_total', 'e_total', 's_total', 'converged']
        
        f.write("# " + " ".join(f"{col:>14}" for col in columns) + "\n")
        
        for params_tuple, results in all_results.items():
            for r in results:
                if phase == 'cfl':
                    row = [r.n_B, r.T, r.Delta0, r.Delta, r.mu_u, r.mu_d, r.mu_s, r.P_total, r.e_total, r.s_total, r.f_total]
                else:
                    if hasattr(r, 'converged') and not r.converged:
                        continue
                    if settings.equilibrium.lower() == 'beta_eq':
                        row = [r.n_B, r.T, r.mu_u, r.mu_d, r.mu_s, r.mu_e,
                               r.Y_u, r.Y_d, r.Y_s, r.P_total, r.e_total, r.s_total, 1]
                    else:
                        row = [r.n_B, r.Y_C, r.T, r.mu_u, r.mu_d, r.mu_s,
                               r.Y_u, r.Y_d, r.Y_s, r.P_total, r.e_total, r.s_total, 1]
                
                f.write(" ".join(f"{v:>14.6e}" if isinstance(v, float) else f"{v:>14}" for v in row) + "\n")
    
    print(f"\nSaved to: {filename}")


# =============================================================================
# HELPER: CONVERT TO ARRAYS
# =============================================================================
def results_to_arrays(results: List, phase: str = 'unpaired') -> Dict[str, np.ndarray]:
    """Convert list of results to dictionary of numpy arrays."""
    if phase == 'cfl':
        attrs = ['n_B', 'T', 'mu', 'Delta', 'Delta0', 'P', 'e', 's', 'f',
                 'n_u', 'n_d', 'n_s', 'Y_u', 'Y_d', 'Y_s']
        arrays = {}
        for attr in attrs:
            try:
                arrays[attr] = np.array([getattr(r, attr) for r in results])
            except AttributeError:
                pass
    else:
        # Filter converged only
        results = [r for r in results if r.converged]
        attrs = ['n_B', 'T', 'Y_C', 'Y_S', 'mu_u', 'mu_d', 'mu_s', 'mu_e',
                 'Y_u', 'Y_d', 'Y_s', 'Y_e', 'P_total', 'e_total', 's_total']
        arrays = {}
        for attr in attrs:
            try:
                arrays[attr] = np.array([getattr(r, attr) for r in results])
            except AttributeError:
                pass
        arrays['converged'] = np.array([r.converged for r in results])

    return arrays


# =============================================================================
# TABLE LOADING AND INTERPOLATION
# =============================================================================

# Column mappings for each equilibrium type
COLUMN_MAPS = {
    'beta_eq': {
        'n_B': 0, 'T': 1, 'mu_u': 2, 'mu_d': 3, 'mu_s': 4, 'mu_e': 5,
        'Y_u': 6, 'Y_d': 7, 'Y_s': 8, 'P_total': 9, 'e_total': 10,
        's_total': 11, 'converged': 12
    },
    'fixed_yc': {
        'n_B': 0, 'Y_C': 1, 'T': 2, 'mu_u': 3, 'mu_d': 4, 'mu_s': 5,
        'Y_u': 6, 'Y_d': 7, 'Y_s': 8, 'P_total': 9, 'e_total': 10,
        's_total': 11, 'converged': 12
    },
    'cfl': {
        'n_B': 0, 'T': 1, 'Delta0': 2, 'Delta': 3, 'mu_u': 4, 'mu_d': 5,
        'mu_s': 6, 'P_total': 7, 'e_total': 8, 's_total': 9, 'f_total': 10
    },
}

# Grid axes for each equilibrium type (order matters for reshaping)
# Note: alpha, B4 are added when loading multiple tables
GRID_AXES = {
    'beta_eq': ['n_B', 'T'],
    'fixed_yc': ['n_B', 'Y_C', 'T'],
    'cfl': ['n_B', 'T', 'Delta0'],
}


@dataclass
class EOSTableData:
    """Container for loaded AlphaBag EOS table with structured grids."""
    eq_type: str
    grids: Dict[str, np.ndarray]      # {'n_B': array, 'T': array, 'alpha': array, 'B4': array, ...}
    data: Dict[str, np.ndarray]       # {'P_total': N-D array, ...}
    filepath: str = ""

    def __repr__(self):
        axes = list(self.grids.keys())
        shapes = [f"{k}={len(v)}" for k, v in self.grids.items()]
        return f"EOSTableData(eq_type='{self.eq_type}', axes={axes}, shape=({', '.join(shapes)}))"


def load_eos_table(filepath: str, eq_type: str,
                   alpha: Optional[float] = None,
                   B4: Optional[float] = None) -> EOSTableData:
    """
    Load an AlphaBag EOS table from file and return structured grids.

    Parameters:
        filepath: Path to the .dat file
        eq_type: Equilibrium type - 'beta_eq', 'fixed_yc', or 'cfl'
        alpha: Override alpha value (if None, read from file header)
        B4: Override B4 value (if None, read from file header)

    Returns:
        EOSTableData with:
        - grids: dict of 1D arrays for each axis (n_B, T, etc.)
        - data: dict of N-dimensional arrays for each quantity

    Example:
        >>> table = load_eos_table('eos_quark_betaeq.dat', 'beta_eq')
        >>> P = table.data['P_total']  # Shape: (n_nB, n_T)
    """
    if eq_type not in COLUMN_MAPS:
        raise ValueError(f"Unknown eq_type: {eq_type}. "
                        f"Valid options: {list(COLUMN_MAPS.keys())}")

    col_map = COLUMN_MAPS[eq_type]
    axes = GRID_AXES[eq_type]

    # Parse header for model parameters if not provided
    B4_file, alpha_file, m_s_file = 165.0, 0.3, 150.0
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('# Parameters:'):
                import re
                match = re.search(r'B\^1/4=(\d+\.?\d*)', line)
                if match:
                    B4_file = float(match.group(1))
                match = re.search(r'α_s=(\d+\.?\d*)', line)
                if match:
                    alpha_file = float(match.group(1))
                match = re.search(r'm_s=(\d+\.?\d*)', line)
                if match:
                    m_s_file = float(match.group(1))
                break

    # Use file values if not overridden
    if alpha is None:
        alpha = alpha_file
    if B4 is None:
        B4 = B4_file

    # Load raw data
    raw_data = np.loadtxt(filepath, comments='#')
    print(f"Loaded {len(raw_data)} points from {filepath}")

    # Extract unique grid values for each axis
    grids = {}
    for axis in axes:
        grids[axis] = np.unique(raw_data[:, col_map[axis]])

    # Determine grid shape
    shape = tuple(len(grids[axis]) for axis in axes)
    n_points_expected = np.prod(shape)

    if len(raw_data) != n_points_expected:
        print(f"  Warning: Expected {n_points_expected} points for complete grid, got {len(raw_data)}")

    # Columns to extract (exclude grid axes and converged flag)
    exclude = set(axes) | {'converged'}
    columns = [c for c in col_map.keys() if c not in exclude]

    # Build structured arrays using vectorized approach
    data = {}

    # Create index mapping
    indices = []
    for axis in axes:
        axis_values = raw_data[:, col_map[axis]]
        grid_values = grids[axis]
        idx = np.searchsorted(grid_values, axis_values)
        indices.append(idx)
    indices = tuple(indices)

    # Fill in data arrays
    for col in columns:
        arr = np.full(shape, np.nan)
        arr[indices] = raw_data[:, col_map[col]]
        data[col] = arr

    # Add derived quantities: f_total = e_total - T * s_total
    if 'e_total' in data and 's_total' in data:
        if eq_type in ['beta_eq', 'fixed_yc']:
            T_idx = axes.index('T')
            T_broadcast_shape = [1] * len(axes)
            T_broadcast_shape[T_idx] = len(grids['T'])
            T_grid = grids['T'].reshape(T_broadcast_shape)
            data['f_total'] = data['e_total'] - T_grid * data['s_total']

    # Print summary
    print(f"  Equilibrium: {eq_type}")
    print(f"  Model: B^1/4={B4} MeV, α_s={alpha}, m_s={m_s_file} MeV")
    for axis in axes:
        print(f"  {axis}: [{grids[axis][0]:.4g}, {grids[axis][-1]:.4g}], {len(grids[axis])} points")

    return EOSTableData(
        eq_type=eq_type,
        grids=grids,
        data=data,
        filepath=filepath
    )


def load_eos_tables_multi(filepaths: List[str], eq_type: str,
                          alpha_values: List[float],
                          B4_values: List[float],
                          Delta0_values: Optional[List[float]] = None) -> EOSTableData:
    """
    Load multiple AlphaBag EOS tables and combine into a single multi-dimensional grid.

    Parameters:
        filepaths: List of file paths (one per parameter combination)
        eq_type: Equilibrium type - 'beta_eq', 'fixed_yc', or 'cfl'
        alpha_values: List of alpha values corresponding to files
        B4_values: List of B4 values corresponding to files
        Delta0_values: List of Delta0 values (required for CFL, optional otherwise)

    Returns:
        EOSTableData with grids including alpha, B4, and Delta0 (for CFL) axes

    Example (beta_eq):
        >>> filepaths = ['eos_B135_a0.1.dat', 'eos_B165_a0.1.dat', ...]
        >>> alpha_vals = [0.1, 0.1, 0.1]
        >>> B4_vals = [135, 165, 180]
        >>> table = load_eos_tables_multi(filepaths, 'beta_eq', alpha_vals, B4_vals)
        >>> P = table.data['P_total']  # Shape: (n_nB, n_T, n_alpha, n_B4)

    Example (CFL):
        >>> filepaths = ['eos_B135_D80.dat', 'eos_B135_D120.dat', ...]
        >>> table = load_eos_tables_multi(filepaths, 'cfl', alpha_vals, B4_vals, Delta0_vals)
        >>> P = table.data['P_total']  # Shape: (n_nB, n_T, n_alpha, n_B4, n_Delta0)
    """
    if len(filepaths) != len(alpha_values) or len(filepaths) != len(B4_values):
        raise ValueError("filepaths, alpha_values, and B4_values must have same length")

    if eq_type == 'cfl' and Delta0_values is not None:
        if len(filepaths) != len(Delta0_values):
            raise ValueError("filepaths and Delta0_values must have same length for CFL")

    # Get unique sorted values
    alpha_arr = np.array(sorted(set(alpha_values)))
    B4_arr = np.array(sorted(set(B4_values)))
    Delta0_arr = np.array(sorted(set(Delta0_values))) if Delta0_values else None

    # Load first table to get base grid structure
    first_table = load_eos_table(filepaths[0], eq_type, alpha_values[0], B4_values[0])
    data_keys = list(first_table.data.keys())

    # For CFL with Delta0_values, use reduced base axes (n_B, T only)
    # since Delta0 becomes an extra axis
    if eq_type == 'cfl' and Delta0_values is not None:
        base_axes = ['n_B', 'T']
        base_grids = {k: first_table.grids[k] for k in base_axes}
    else:
        base_axes = GRID_AXES[eq_type]
        base_grids = first_table.grids

    # Extended grid with alpha, B4, and optionally Delta0
    grids = dict(base_grids)
    grids['alpha'] = alpha_arr
    grids['B4'] = B4_arr
    if Delta0_arr is not None:
        grids['Delta0'] = Delta0_arr

    # Extended shape
    base_shape = tuple(len(base_grids[ax]) for ax in base_axes)
    if Delta0_arr is not None:
        ext_shape = base_shape + (len(alpha_arr), len(B4_arr), len(Delta0_arr))
    else:
        ext_shape = base_shape + (len(alpha_arr), len(B4_arr))

    # Initialize data arrays
    data = {key: np.full(ext_shape, np.nan) for key in data_keys}

    # Fill in data from each file
    if Delta0_values is not None:
        for fpath, alpha, B4, Delta0 in zip(filepaths, alpha_values, B4_values, Delta0_values):
            table = load_eos_table(fpath, eq_type, alpha, B4)

            # Find indices
            i_alpha = np.searchsorted(alpha_arr, alpha)
            i_B4 = np.searchsorted(B4_arr, B4)
            i_Delta0 = np.searchsorted(Delta0_arr, Delta0)

            # Copy data (squeeze out the single Delta0 dimension from file)
            for key in data_keys:
                if key in table.data:
                    # table.data[key] has shape (n_B, T, 1) for CFL files
                    data[key][..., i_alpha, i_B4, i_Delta0] = table.data[key][:, :, 0]
    else:
        for fpath, alpha, B4 in zip(filepaths, alpha_values, B4_values):
            table = load_eos_table(fpath, eq_type, alpha, B4)

            # Find indices
            i_alpha = np.searchsorted(alpha_arr, alpha)
            i_B4 = np.searchsorted(B4_arr, B4)

            # Copy data
            for key in data_keys:
                if key in table.data:
                    data[key][..., i_alpha, i_B4] = table.data[key]

    print(f"\nCombined {len(filepaths)} tables into multi-dimensional grid")
    print(f"  alpha: {alpha_arr}")
    print(f"  B4: {B4_arr}")
    if Delta0_arr is not None:
        print(f"  Delta0: {Delta0_arr}")

    return EOSTableData(
        eq_type=eq_type,
        grids=grids,
        data=data,
        filepath=str(filepaths)
    )


def build_interpolators(table: EOSTableData,
                        method: str = 'linear',
                        bounds_error: bool = False,
                        fill_value: float = np.nan) -> Dict[str, Any]:
    """
    Build interpolation functions from loaded AlphaBag EOS table data.

    Parameters:
        table: EOSTableData from load_eos_table() or load_eos_tables_multi()
        method: Interpolation method ('linear', 'nearest', 'cubic', etc.)
        bounds_error: If True, raise error for out-of-bounds queries
        fill_value: Value to return for out-of-bounds queries

    Returns:
        Dict with:
        - 'interpolators': dict of RegularGridInterpolator for each quantity
        - 'grids': reference to the grid arrays
        - 'axes': list of axis names in order
        - Convenience functions for common quantities

    Example (single table, beta_eq):
        >>> table = load_eos_table('eos_quark.dat', 'beta_eq')
        >>> interp = build_interpolators(table)
        >>> P = interp['P'](0.5, 10.0)  # P(n_B, T)

    Example (multi-table, beta_eq with alpha, B4):
        >>> table = load_eos_tables_multi(files, 'beta_eq', alphas, B4s)
        >>> interp = build_interpolators(table)
        >>> P = interp['P'](0.5, 10.0, 0.3, 165)  # P(n_B, T, alpha, B4)
    """
    from scipy.interpolate import RegularGridInterpolator

    # Determine axes order based on what's in grids
    # For CFL with multi-table loading, Delta0 becomes an extra axis (not base)
    if table.eq_type == 'cfl' and 'alpha' in table.grids and 'Delta0' in table.grids:
        # CFL loaded via load_eos_tables_multi with Delta0_values
        # Axes: (n_B, T, alpha, B4, Delta0)
        base_axes = ['n_B', 'T']
    else:
        base_axes = GRID_AXES[table.eq_type]

    axes = list(base_axes)
    if 'alpha' in table.grids:
        axes.append('alpha')
    if 'B4' in table.grids:
        axes.append('B4')
    if 'Delta0' in table.grids and 'Delta0' not in base_axes:
        axes.append('Delta0')

    grid_tuple = tuple(table.grids[axis] for axis in axes)

    interpolators = {}
    for name, arr in table.data.items():
        interpolators[name] = RegularGridInterpolator(
            grid_tuple, arr,
            method=method,
            bounds_error=bounds_error,
            fill_value=fill_value
        )

    result = {
        'interpolators': interpolators,
        'grids': table.grids,
        'axes': axes,
        'eq_type': table.eq_type,
    }

    # Add convenience functions based on equilibrium type and dimensionality
    has_alpha_B4 = 'alpha' in table.grids and 'B4' in table.grids

    if table.eq_type == 'beta_eq':
        if has_alpha_B4:
            # 4D: f(n_B, T, alpha, B4)
            result['P'] = lambda nB, T, alpha, B4: interpolators['P_total']((nB, T, alpha, B4))
            result['eps'] = lambda nB, T, alpha, B4: interpolators['e_total']((nB, T, alpha, B4))
            result['s'] = lambda nB, T, alpha, B4: interpolators['s_total']((nB, T, alpha, B4))
            result['f'] = lambda nB, T, alpha, B4: interpolators['f_total']((nB, T, alpha, B4))
            result['mu_u'] = lambda nB, T, alpha, B4: interpolators['mu_u']((nB, T, alpha, B4))
            result['mu_d'] = lambda nB, T, alpha, B4: interpolators['mu_d']((nB, T, alpha, B4))
            result['mu_s'] = lambda nB, T, alpha, B4: interpolators['mu_s']((nB, T, alpha, B4))
            result['mu_e'] = lambda nB, T, alpha, B4: interpolators['mu_e']((nB, T, alpha, B4))
            result['Y_u'] = lambda nB, T, alpha, B4: interpolators['Y_u']((nB, T, alpha, B4))
            result['Y_d'] = lambda nB, T, alpha, B4: interpolators['Y_d']((nB, T, alpha, B4))
            result['Y_s'] = lambda nB, T, alpha, B4: interpolators['Y_s']((nB, T, alpha, B4))
        else:
            # 2D: f(n_B, T)
            result['P'] = lambda nB, T: interpolators['P_total']((nB, T))
            result['eps'] = lambda nB, T: interpolators['e_total']((nB, T))
            result['s'] = lambda nB, T: interpolators['s_total']((nB, T))
            result['f'] = lambda nB, T: interpolators['f_total']((nB, T))
            result['mu_u'] = lambda nB, T: interpolators['mu_u']((nB, T))
            result['mu_d'] = lambda nB, T: interpolators['mu_d']((nB, T))
            result['mu_s'] = lambda nB, T: interpolators['mu_s']((nB, T))
            result['mu_e'] = lambda nB, T: interpolators['mu_e']((nB, T))
            result['Y_u'] = lambda nB, T: interpolators['Y_u']((nB, T))
            result['Y_d'] = lambda nB, T: interpolators['Y_d']((nB, T))
            result['Y_s'] = lambda nB, T: interpolators['Y_s']((nB, T))

    elif table.eq_type == 'fixed_yc':
        if has_alpha_B4:
            # 5D: f(n_B, Y_C, T, alpha, B4)
            result['P'] = lambda nB, YC, T, alpha, B4: interpolators['P_total']((nB, YC, T, alpha, B4))
            result['eps'] = lambda nB, YC, T, alpha, B4: interpolators['e_total']((nB, YC, T, alpha, B4))
            result['s'] = lambda nB, YC, T, alpha, B4: interpolators['s_total']((nB, YC, T, alpha, B4))
            result['f'] = lambda nB, YC, T, alpha, B4: interpolators['f_total']((nB, YC, T, alpha, B4))
        else:
            # 3D: f(n_B, Y_C, T)
            result['P'] = lambda nB, YC, T: interpolators['P_total']((nB, YC, T))
            result['eps'] = lambda nB, YC, T: interpolators['e_total']((nB, YC, T))
            result['s'] = lambda nB, YC, T: interpolators['s_total']((nB, YC, T))
            result['f'] = lambda nB, YC, T: interpolators['f_total']((nB, YC, T))

    elif table.eq_type == 'cfl':
        # Check if Delta0 is an extra axis (from load_eos_tables_multi with Delta0_values)
        delta0_is_extra = 'Delta0' in table.grids and 'alpha' in table.grids

        if delta0_is_extra:
            # 5D: f(n_B, T, alpha, B4, Delta0) - from load_eos_tables_multi
            result['P'] = lambda nB, T, alpha, B4, Delta0: interpolators['P_total']((nB, T, alpha, B4, Delta0))
            result['eps'] = lambda nB, T, alpha, B4, Delta0: interpolators['e_total']((nB, T, alpha, B4, Delta0))
            result['s'] = lambda nB, T, alpha, B4, Delta0: interpolators['s_total']((nB, T, alpha, B4, Delta0))
            result['f'] = lambda nB, T, alpha, B4, Delta0: interpolators['f_total']((nB, T, alpha, B4, Delta0))
            result['Delta'] = lambda nB, T, alpha, B4, Delta0: interpolators['Delta']((nB, T, alpha, B4, Delta0))
            result['mu_u'] = lambda nB, T, alpha, B4, Delta0: interpolators['mu_u']((nB, T, alpha, B4, Delta0))
            result['mu_d'] = lambda nB, T, alpha, B4, Delta0: interpolators['mu_d']((nB, T, alpha, B4, Delta0))
            result['mu_s'] = lambda nB, T, alpha, B4, Delta0: interpolators['mu_s']((nB, T, alpha, B4, Delta0))
        elif has_alpha_B4:
            # 5D: f(n_B, T, Delta0, alpha, B4) - old format (single Delta0 per file as base axis)
            result['P'] = lambda nB, T, Delta0, alpha, B4: interpolators['P_total']((nB, T, Delta0, alpha, B4))
            result['eps'] = lambda nB, T, Delta0, alpha, B4: interpolators['e_total']((nB, T, Delta0, alpha, B4))
            result['s'] = lambda nB, T, Delta0, alpha, B4: interpolators['s_total']((nB, T, Delta0, alpha, B4))
            result['f'] = lambda nB, T, Delta0, alpha, B4: interpolators['f_total']((nB, T, Delta0, alpha, B4))
            result['Delta'] = lambda nB, T, Delta0, alpha, B4: interpolators['Delta']((nB, T, Delta0, alpha, B4))
        else:
            # 3D: f(n_B, T, Delta0) - single file
            result['P'] = lambda nB, T, Delta0: interpolators['P_total']((nB, T, Delta0))
            result['eps'] = lambda nB, T, Delta0: interpolators['e_total']((nB, T, Delta0))
            result['s'] = lambda nB, T, Delta0: interpolators['s_total']((nB, T, Delta0))
            result['f'] = lambda nB, T, Delta0: interpolators['f_total']((nB, T, Delta0))
            result['Delta'] = lambda nB, T, Delta0: interpolators['Delta']((nB, T, Delta0))

    return result


# =============================================================================
# CONFIGURATION (EDIT THIS SECTION)
# =============================================================================
settings = AlphaBagTableSettings(
    # Model parameters (easy access - if set, override defaults)
    alpha=0.3*np.pi/2,     # QCD coupling α_s
    B4=165.0,      # Bag constant B^(1/4) in MeV
    m_s=100.0,     # Strange quark mass in MeV
    
    # Phase: 'unpaired' or 'cfl'
    phase='cfl',
    
    # For unpaired phase: 'beta_eq', 'fixed_yc', 'fixed_yc_ys'
    equilibrium='beta',
    
    # For CFL phase: pairing gap values
    Delta0_values=[80.0],
    
    # Grid
    n_B_values=np.linspace(0.1, 12, 300) * 0.16,
    T_values=np.concatenate([[0.1],np.linspace(2.5, 120, 48)]),
    Y_C_values=[0.0,0.1,0.2,0.3,0.4,0.5],
    Y_S_values=[0.0, 0.1,0.2,0.4,0.6,0.8,1],
    
    # Options
    include_photons=True,
    include_gluons=True,
    include_electrons=True,
    include_thermal_neutrinos=True,
    
    # Output
    print_results=True,
    print_first_n=3,
    print_errors=True,
    print_timing=True,
    save_to_file=True,
)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    all_results = compute_alphabag_table(settings)
    print("\nDONE!")
