#!/usr/bin/env python3
"""
zlvmit_hybrid_table_generator.py
================================
Configuration and runner script for ZL+vMIT hybrid EOS table generation.

This script contains configuration parameters and calls the table generation
functions from zlvmit_mixed_phase_eos.py.

Usage:
    python zlvmit_hybrid_table_generator.py
"""

import numpy as np
import os
import time
from datetime import datetime


# =============================================================================
# USER CONFIGURATION - EDIT THESE VALUES
# =============================================================================

# Density grid (fm^-3)
n0 = 0.16  # Nuclear saturation density
n_B_min = 0.1 * n0        # Start at low density
n_B_max = 12.0 * n0       # End at high density  
n_B_steps = 300           # Number of density points

# Temperature grid (MeV)
T_values = np.concatenate([[0.1], np.arange(2.5, 122.5, 2.5)])

# Surface tension parameter η: 0 = Gibbs, 1 = Maxwell, intermediate = hybrid
eta_values = [0., 0.1, 0.3, 0.6, 1.0]

# Equilibrium mode: "beta", "fixed_yc", or "trapped"
EQUILIBRIUM_MODE = "beta"

# Charge fractions (used if EQUILIBRIUM_MODE = "fixed_yc")
Y_C_values = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]

# Lepton fractions (used if EQUILIBRIUM_MODE = "trapped")
Y_L_values = [0.3, 0.4, 0.5]

# vMIT bag model parameters
B4 = 165.0   # Bag constant B^{1/4} in MeV
a = 0.2      # Vector coupling in fm²

# Output directories (absolute paths)
from eos import REPO_ROOT
OUTPUT_DIR = str(REPO_ROOT / "output" / "zlvmit_hybrid_outputs")
BOUNDARY_DIR = str(REPO_ROOT / "output" / "zlvmit_hybrid_outputs")

# Control flags
FORCE_RECOMPUTE_BOUNDARIES = True    # If True, recompute boundaries even if files exist
FORCE_RECOMPUTE_PURE_PHASES = True  # If True, recompute pure H/Q tables even if files exist
VERBOSE = True                       # Print detailed progress


# =============================================================================
# IMPORTS FROM MIXED PHASE MODULE
# =============================================================================

from eos.zlvmit.mixed_phase_eos import (
    # Guess generation
    boundary_result_to_guess, 
    result_to_guess,
    # Table output
    results_to_dict_primary,
    results_to_dict_complete,
    save_table_full,
    # Pure phase caching
    get_pure_table_filename,
    save_pure_table,
    load_pure_table,
    # Core functions (for main script if needed)
    find_boundaries,
    generate_unified_table,
    load_boundaries_from_file,
    save_boundaries_to_file,
)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ZL+vMIT Hybrid EOS Table Generator")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  n_B range: {n_B_min:.4f} - {n_B_max:.4f} fm^-3 ({n_B_steps} points)")
    print(f"  T values: {len(T_values)} temperatures from {T_values[0]:.1f} to {T_values[-1]:.1f} MeV")
    print(f"  eta values: {eta_values}")
    print(f"  Equilibrium mode: {EQUILIBRIUM_MODE}")
    print(f"  vMIT params: B^1/4 = {B4} MeV, a = {a} fm^2")
    print(f"  Output dir: {OUTPUT_DIR}")
    print("=" * 70)
    
    # TODO: Add main table generation workflow here
    # This would call find_all_boundaries, generate_unified_table, etc.
    print("\nNote: Main workflow not implemented in this runner script.")
    print("Import and use functions from zlvmit_mixed_phase_eos.py directly.")
