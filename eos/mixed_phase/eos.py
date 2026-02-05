"""
mixed_phase_eos.py
===================
General framework for mixed hadron-quark phase first-order phase transitions.

This module implements the between local and global charge neutrality η-parameterized framework for first-order phase transitions
where η controls the degree of local vs global charge neutrality:

    η = 0: global charge neutrality only
    η = 1: local charge neutrality in each phase
    0 < η < 1: Intermediate cases

The framework from: 
    - C. Constantinou et al. (2023), Phys.Rev.D 107 (2023) 7, 074013, 
    - C. Constantinou et al. (2025), Phys.Rev.D 112 (2025) 9, 094014   
    - M. Guerrini (2026), PhD Thesis Chapter 3.

Key notation:
    - χ (chi): Volume fraction of quark matter 
    - H subscript: Hadronic phase quantities
    - Q subscript: Quark phase quantities
    - eL: Local electrons (fraction η enforcing local neutrality)
    - eG: Global electrons (fraction 1-η enforcing global neutrality)
    
Units:
    - Energy/mass/chemical potentials: MeV
    - Densities: fm⁻³
    - Pressure/energy density: MeV/fm³
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum, auto


# =============================================================================
# ENUMERATIONS
# =============================================================================
class PhaseState(Enum):
    """Current phase state of matter."""
    HADRON = auto()   # Pure hadronic phase (χ ≤ 0)
    MIXED = auto()    # Mixed phase (0 < χ < 1)
    QUARK = auto()    # Pure quark phase (χ ≥ 1)


class EquilibriumType(Enum):
    """Type of equilibrium/constraint."""
    BETA_EQ = auto()           # Beta equilibrium + charge neutrality
    FIXED_YC = auto()          # Fixed charge fraction Y_C
    FIXED_YC_YS = auto()       # Fixed Y_C and strangeness Y_S
    TRAPPED_NEUTRINOS = auto() # Trapped neutrinos (fixed Y_L)


# =============================================================================
# RESULT DATACLASS
# =============================================================================
@dataclass
class MixedPhaseResult:
    """
    Complete result from mixed phase calculation.
    
    Naming convention following the user's request:
        - n_X_H: density of species X in hadronic phase
        - n_X_Q: density of species X in quark phase
        - n_X: total density of species X
        - Y_X: fraction relative to baryon density
        - μ_X: chemical potential of species X
        
    The total quantities use volume-weighted averages:
        n_i = (1-χ) * n_i_H + χ * n_i_Q    i = all global conserved quantities
        Y_C = n_C / n_B
        Y_C_H = n_C_H / n_B_H
        Y_C_Q = n_C_Q / n_B_Q
        same for Y_S and Y_L
    """
    # Convergence info
    converged: bool = False
    error: float = 1e10
    
    # Phase state
    phase: PhaseState = PhaseState.HADRON
    chi: float = 0.0  # Quark volume fraction
    
    # Input/control parameters
    n_B: float = 0.0      # Total baryon density (fm⁻³)
    T: float = 0.0        # Temperature (MeV)
    eta: float = 0.0      # Local/global neutrality parameter [0, 1]
    
    # Conserved quantities (input constraints)
    Y_C: float = 0.0      # Charge fraction n_C/n_B
    Y_S: float = 0.0      # Strangeness fraction n_S/n_B
    Y_L: float = 0.0      # Lepton fraction (for trapped neutrinos)
    
    # === Hadronic phase quantities (H) ===
    # General B, C, S notation for future hyperonic matter support
    n_B_H: float = 0.0    # Baryon density in H phase (fm⁻³)
    n_C_H: float = 0.0    # Charge density in H phase (fm⁻³)
    n_S_H: float = 0.0    # Strangeness density in H phase (fm⁻³)
    n_L_H: float = 0.0    # Lepton density in H phase (fm⁻³)
    
    # Chemical potentials (conserved charges)
    mu_B_H: float = 0.0   # Baryon chemical potential (MeV)
    mu_C_H: float = 0.0   # Charge chemical potential (MeV)  
    mu_S_H: float = 0.0   # Strangeness chemical potential (MeV)
    
    P_H: float = 0.0      # Pressure of H phase (MeV/fm³)
    e_H: float = 0.0      # Energy density of H phase (MeV/fm³)
    s_H: float = 0.0      # Entropy density of H phase (fm⁻³)
    
    # Local electrons in H phase
    n_eL_H: float = 0.0   # Local electron density
    mu_eL_H: float = 0.0  # Local electron chemical potential
    
    # === Quark phase quantities (Q) ===
    # Using same B, C, S notation
    n_B_Q: float = 0.0    # Baryon density in Q phase (fm⁻³)
    n_C_Q: float = 0.0    # Charge density in Q phase (fm⁻³)
    n_S_Q: float = 0.0    # Strangeness density in Q phase (fm⁻³)
    n_L_Q: float = 0.0    # Lepton density in Q phase (fm⁻³)
    
    # Chemical potentials (conserved charges)
    mu_B_Q: float = 0.0   # Baryon chemical potential (MeV)
    mu_C_Q: float = 0.0   # Charge chemical potential (MeV)
    mu_S_Q: float = 0.0   # Strangeness chemical potential (MeV)
    
    P_Q: float = 0.0      # Pressure of Q phase (MeV/fm³)
    e_Q: float = 0.0      # Energy density of Q phase (MeV/fm³)
    s_Q: float = 0.0      # Entropy density of Q phase (fm⁻³)
    
    # Local electrons in Q phase
    n_eL_Q: float = 0.0   # Local electron density
    mu_eL_Q: float = 0.0  # Local electron chemical potential
    
    # === Global electrons (eG) ===
    n_eG: float = 0.0     # Global electron density
    mu_eG: float = 0.0    # Global electron chemical potential
    
    # === Neutrinos (for trapped case) ===
    n_nu: float = 0.0     # Neutrino density
    mu_nu: float = 0.0    # Neutrino chemical potential
    
    # === Total thermodynamic quantities ===
    P_total: float = 0.0  # Total pressure (MeV/fm³)
    e_total: float = 0.0  # Total energy density (MeV/fm³)
    s_total: float = 0.0  # Total entropy density (fm⁻³)
    f_total: float = 0.0  # Total free energy density (MeV/fm³)
    
    # === Derived total fractions ===
    n_e_total: float = 0.0  # Total electron density
    Y_e: float = 0.0        # Total electron fraction n_e/n_B
    Y_C: float = 0.0        # Total charge fraction n_C/n_B
    Y_S: float = 0.0        # Total strangeness fraction n_S/n_B
    Y_L: float = 0.0        # Total lepton fraction n_L/n_B
    
    def compute_derived(self):
        """Compute derived quantities from primary ones."""
        # Total baryon density (should match input n_B)
        n_B_computed = (1 - self.chi) * self.n_B_H + self.chi * self.n_B_Q
        
        # Total charge density
        n_C_total = (1 - self.chi) * self.n_C_H + self.chi * self.n_C_Q
        
        # Total strangeness density  
        n_S_total = (1 - self.chi) * self.n_S_H + self.chi * self.n_S_Q

        # Total lepton density
        n_L_total = (1 - self.chi) * self.n_L_H + self.chi * self.n_L_Q
        
        # Total electron density
        # n_e = η(1-χ)n_eL_H + η·χ·n_eL_Q + (1-η)n_eG
        self.n_e_total = (self.eta * (1 - self.chi) * self.n_eL_H + 
                          self.eta * self.chi * self.n_eL_Q +
                          (1 - self.eta) * self.n_eG)
        
        # Fractions (relative to total n_B)
        self.Y_e = self.n_e_total / self.n_B
        self.Y_C = n_C_total / self.n_B
        self.Y_S = n_S_total / self.n_B
        
        # Free energy
        self.f_total = self.e_total - self.T * self.s_total
        
        # Determine phase state from chi (but don't modify chi - keep raw for guess propagation)
        if self.chi <= 0.0:
            self.phase = PhaseState.HADRON
        elif self.chi >= 1.0:
            self.phase = PhaseState.QUARK
        else:
            self.phase = PhaseState.MIXED


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
# Note: n_B, n_C, n_S in each phase are computed directly by the specific EOS
# (e.g., ZL for nucleonic, SFHoY for hyperonic, vMIT for quark matter)
# These values are passed to MixedPhaseResult already computed.

# =============================================================================
# CONSERVATION LAW UTILITIES (phase-independent)
# =============================================================================


def compute_total_baryon_density(n_B_H: float, n_B_Q: float, chi: float) -> float:
    """Compute total baryon density: n_B = (1-χ)n_B^H + χ·n_B^Q"""
    return (1 - chi) * n_B_H + chi * n_B_Q


def compute_total_charge_density(n_C_H: float, n_C_Q: float, chi: float) -> float:
    """Compute total charge density: n_C = (1-χ)n_C^H + χ·n_C^Q"""
    return (1 - chi) * n_C_H + chi * n_C_Q

def compute_total_strangeness_density(n_S_H: float, n_S_Q: float, chi: float) -> float:
    """Compute total strangeness density: n_S = (1-χ)n_S^H + χ·n_S^Q"""
    return (1 - chi) * n_S_H + chi * n_S_Q

def compute_total_lepton_density(n_L_H: float, n_L_Q: float, chi: float) -> float:
    """Compute total lepton density: n_L = (1-χ)n_L^H + χ·n_L^Q"""
    return (1 - chi) * n_L_H + chi * n_L_Q


def compute_total_electron_density(
    n_eL_H: float, n_eL_Q: float, n_eG: float, 
    chi: float, eta: float
) -> float:
    """
    Compute total electron density.
    
    n_e = η(1-χ)n_eL^H + η·χ·n_eL^Q + (1-η)n_eG
    """
    return eta * (1 - chi) * n_eL_H + eta * chi * n_eL_Q + (1 - eta) * n_eG


# =============================================================================
# MIXED PHASE PRESSURE WITH η-WEIGHTED ELECTRONS
# =============================================================================
def compute_mixed_P_H(P_hadron: float, P_eL_H: float, P_eG: float, 
                      P_photon: float, eta: float) -> float:
    """
    Compute total pressure in hadronic phase including electrons and photons.
    
    P^H_total = P_hadron + η·P_eL^H + (1-η)·P_eG + P_γ
    """
    return P_hadron + eta * P_eL_H + (1 - eta) * P_eG + P_photon


def compute_mixed_P_Q(P_quark: float, P_eL_Q: float, P_eG: float, 
                      P_photon: float, eta: float) -> float:
    """
    Compute total pressure in quark phase including electrons and photons.
    
    P^Q_total = P_quark + η·P_eL^Q + (1-η)·P_eG + P_γ
    """
    return P_quark + eta * P_eL_Q + (1 - eta) * P_eG + P_photon


def compute_mixed_e_H(e_hadron: float, e_eL_H: float, e_eG: float,
                      e_photon: float, eta: float) -> float:
    """
    Compute total energy density in hadronic phase.
    
    ε^H_total = ε_hadron + η·ε_eL^H + (1-η)·ε_eG + ε_γ
    """
    return e_hadron + eta * e_eL_H + (1 - eta) * e_eG + e_photon


def compute_mixed_e_Q(e_quark: float, e_eL_Q: float, e_eG: float,
                      e_photon: float, eta: float) -> float:
    """
    Compute total energy density in quark phase.
    """
    return e_quark + eta * e_eL_Q + (1 - eta) * e_eG + e_photon


def compute_mixed_s_H(s_hadron: float, s_eL_H: float, s_eG: float,
                      s_photon: float, eta: float) -> float:
    """
    Compute total entropy density in hadronic phase.
    """
    return s_hadron + eta * s_eL_H + (1 - eta) * s_eG + s_photon


def compute_mixed_s_Q(s_quark: float, s_eL_Q: float, s_eG: float,
                      s_photon: float, eta: float) -> float:
    """
    Compute total entropy density in quark phase.
    """
    return s_quark + eta * s_eL_Q + (1 - eta) * s_eG + s_photon


# =============================================================================
# TOTAL MIXED PHASE THERMODYNAMICS
# =============================================================================
def compute_total_pressure(P_H: float, P_Q: float, chi: float) -> float:
    """
    In pressure equilibrium, P^H = P^Q = P_total.
    This function is for verification / pure phases.
    """
    # They should be equal in equilibrium
    return P_H  # or P_Q


def compute_total_energy(e_H: float, e_Q: float, chi: float) -> float:
    """
    Compute total energy density as volume-weighted average.
    
    ε_total = (1-χ)·ε^H + χ·ε^Q
    """
    return (1 - chi) * e_H + chi * e_Q


def compute_total_entropy(s_H: float, s_Q: float, chi: float) -> float:
    """
    Compute total entropy density as volume-weighted average.
    
    s_total = (1-χ)·s^H + χ·s^Q
    """
    return (1 - chi) * s_H + chi * s_Q


# =============================================================================
# CHEMICAL EQUILIBRIUM CONDITIONS (General B, C, S, L)
# =============================================================================

# --- Phase equilibrium (between H and Q phases) ---
def check_baryon_equilibrium(mu_B_H: float, mu_B_Q: float) -> float:
    """Baryon chemical equilibrium: μ_B^H = μ_B^Q"""
    return mu_B_H - mu_B_Q


def check_charge_equilibrium(mu_C_H: float, mu_C_Q: float, 
                              mu_eL_H: float, mu_eL_Q: float, eta: float) -> float:
    """
    Charge chemical equilibrium with η-weighted local electrons.
    
    μ_C^H + η·μ_eL^H = μ_C^Q + η·μ_eL^Q
    """
    return (mu_C_H + eta * mu_eL_H) - (mu_C_Q + eta * mu_eL_Q)


def check_strangeness_equilibrium(mu_S_H: float, mu_S_Q: float) -> float:
    """Strangeness chemical equilibrium: μ_S^H = μ_S^Q"""
    return mu_S_H - mu_S_Q


def check_lepton_equilibrium(mu_L_H: float, mu_L_Q: float) -> float:
    """Lepton chemical equilibrium: μ_L^H = μ_L^Q"""
    return mu_L_H - mu_L_Q


def check_pressure_equilibrium(P_H: float, P_Q: float) -> float:
    """Pressure equilibrium between phases: P^H = P^Q"""
    return P_H - P_Q


# --- Beta equilibrium (within each phase) ---
def check_beta_eq_H(mu_C_H: float, mu_eL_H: float) -> float:
    """
    Hadronic beta equilibrium: μ_C^H + μ_e^H = 0
    (e.g., n ↔ p + e⁻ + ν̄ at equilibrium)
    """
    return mu_C_H + mu_eL_H


def check_beta_eq_Q(mu_C_Q: float, mu_eL_Q: float) -> float:
    """
    Quark beta equilibrium: μ_C^Q + μ_e^Q = 0
    (e.g., d ↔ u + e⁻ + ν̄ at equilibrium)
    """
    return mu_C_Q + mu_eL_Q


# --- Weak equilibrium for strangeness ---
def check_strangeness_weak_eq_Q(mu_S_Q: float) -> float:
    """
    Quark strangeness weak equilibrium: μ_S^Q = 0
    (i.e., μ_d^Q = μ_s^Q, free flavor conversion)
    """
    return mu_S_Q


def check_strangeness_weak_eq_H(mu_S_H: float) -> float:
    """
    Hadronic strangeness weak equilibrium: μ_S^H = 0
    (relevant for hyperonic matter with free Λ ↔ N + K etc.)
    """
    return mu_S_H


# =============================================================================
# CONSERVATION LAW RESIDUALS
# =============================================================================
def baryon_conservation_residual(n_B_target: float, n_B_H: float, n_B_Q: float, 
                                  chi: float) -> float:
    """
    Residual for baryon number conservation.
    
    n_B = (1-χ)n_B^H + χ·n_B^Q
    """
    n_B_computed = (1 - chi) * n_B_H + chi * n_B_Q
    return n_B_computed - n_B_target


def charge_conservation_residual(n_C_target: float, n_C_H: float, n_C_Q: float, 
                                  chi: float) -> float:
    """
    Residual for charge conservation.
    
    n_C = (1-χ)n_C^H + χ·n_C^Q
    """
    n_C_computed = (1 - chi) * n_C_H + chi * n_C_Q
    return n_C_computed - n_C_target


def local_neutrality_H_residual(n_C_H: float, n_eL_H: float) -> float:
    """
    Residual for local charge neutrality in hadronic phase.
    
    n_C^H = n_eL^H
    """
    return n_C_H - n_eL_H


def local_neutrality_Q_residual(n_u_Q: float, n_d_Q: float, n_s_Q: float, 
                                 n_eL_Q: float) -> float:
    """
    Residual for local charge neutrality in quark phase.
    
    (2n_u - n_d - n_s)/3 = n_eL^Q
    """
    n_C_Q = (2 * n_u_Q - n_d_Q - n_s_Q) / 3.0
    return n_C_Q - n_eL_Q


def global_neutrality_residual(n_C_H: float, n_eG_H: float, n_C_Q: float, 
                                n_eG_Q: float, chi: float) -> float:
    """
    Residual for global charge neutrality.
    
    (1-χ)(n_C^H - n_eG^H) + χ(n_C^Q - n_eG^Q) = 0
    
    Note: For simplified case where n_eG_H = n_eG_Q = n_eG (same global electron pool):
    This becomes: (1-χ)n_C^H + χ·n_C^Q = n_eG
    """
    return (1 - chi) * (n_C_H - n_eG_H) + chi * (n_C_Q - n_eG_Q)


# =============================================================================
# GENERAL MIXED PHASE SOLVER (EOS-agnostic)
# =============================================================================
from typing import Callable, Dict, Any
from abc import ABC, abstractmethod
from scipy.optimize import root
import warnings

class MixedPhaseSolverBase(ABC):
    """
    Abstract base class for mixed phase solvers.
    
    This class implements the GENERAL mixed phase equations for all η cases.
    Uses conserved charges (B, C, S) notation for EOS-agnostic interface.
    
    Subclasses must implement EOS-specific thermodynamic functions:
        - hadron_thermo(μ_B, μ_C, μ_S, μ_e, T) -> {n_B_H, n_C_H, n_S_H, n_e_H, P_H, e_H, s_H}
        - quark_thermo(μ_B, μ_C, μ_S, μ_e, T) -> {n_B_Q, n_C_Q, n_S_Q, n_e_Q, P_Q, e_Q, s_Q}
        - electron_thermo(μ_e, T) -> {n, P, e, s}
        - solve_pure_H(n_B, T, eq_type) -> MixedPhaseResult
        - solve_pure_Q(n_B, T, eq_type) -> MixedPhaseResult
    
    Features:
        - Warm start: reuses previous solution as initial guess for next n_B
        - Supports ZL-style (no mean fields) and SFHo-style (with mean fields) EOS
    """
    
    # Solution cache for warm starting
    _last_solution: Optional[np.ndarray] = None
    _last_n_B: Optional[float] = None
    _last_T: Optional[float] = None
    _last_eta: Optional[float] = None
    
    # =========================================================================
    # ABSTRACT METHODS (EOS-specific, must be implemented by subclasses)
    # =========================================================================
    
    @property
    def has_mean_fields(self) -> bool:
        """
        Return True if hadronic EOS has explicit mean fields (SFHo-style).
        Return False for simpler EOS (ZL-style).
        
        Override in subclass if using mean fields.
        """
        return False
    
    @abstractmethod
    def hadron_thermo(self, mu_B: float, mu_C: float, mu_S: float, mu_e: float,
                      T: float, mean_fields: Dict[str, float] = None) -> Dict[str, float]:
        """
        Compute hadronic phase thermodynamics from conserved charge chemical potentials.
        
        Args:
            mu_B: Baryon chemical potential (MeV)
            mu_C: Charge chemical potential (MeV)
            mu_S: Strangeness chemical potential (MeV)
            mu_e: Electron chemical potential (MeV)
            T: Temperature (MeV)
            mean_fields: Optional dict with {sigma, omega, rho, phi} for SFHo-style EOS
                         For ZL-style, this is None and fields are solved internally.
            
        Returns:
            Dict with keys: n_B, n_C, n_S, n_e, P, e, s
            For SFHo-style, also returns: sigma, omega, rho, phi (updated values)
        """
        pass
    
    @abstractmethod
    def quark_thermo(self, mu_B: float, mu_C: float, mu_S: float, mu_e: float,
                     T: float) -> Dict[str, float]:
        """
        Compute quark phase thermodynamics from conserved charge chemical potentials.
        
        Args:
            mu_B: Baryon chemical potential (MeV)
            mu_C: Charge chemical potential (MeV)  
            mu_S: Strangeness chemical potential (MeV)
            mu_e: Electron chemical potential (MeV)
            T: Temperature (MeV)
            
        Returns:
            Dict with keys: n_B, n_C, n_S, n_e, P, e, s
        """
        pass
    
    @abstractmethod
    def electron_thermo(self, mu_e: float, T: float) -> Dict[str, float]:
        """Compute electron thermodynamics. Returns {n, P, e, s}"""
        pass
    
    def get_mean_field_equations(self, mu_B: float, mu_C: float, mu_S: float,
                                  sigma: float, omega: float, rho: float, phi: float,
                                  T: float) -> np.ndarray:
        """
        Return residuals for mean field equations (SFHo-style only).
        
        Override in subclass if has_mean_fields is True.
        Default returns empty array (no mean field equations).
        """
        return np.array([])
    
    @abstractmethod
    def solve_pure_H(self, n_B: float, T: float, 
                     eq_type: EquilibriumType = EquilibriumType.BETA_EQ) -> MixedPhaseResult:
        """Solve pure hadronic phase."""
        pass
    
    @abstractmethod
    def solve_pure_Q(self, n_B: float, T: float,
                     eq_type: EquilibriumType = EquilibriumType.BETA_EQ) -> MixedPhaseResult:
        """Solve pure quark phase."""
        pass
    
    # =========================================================================
    # CONCRETE METHODS (General mixed phase logic)
    # =========================================================================
    
    def solve_mixed(self, n_B: float, T: float, eta: float,
                    eq_type: EquilibriumType = EquilibriumType.BETA_EQ,
                    Y_C: float = None, Y_S: float = None, Y_L: float = None,
                    initial_guess: np.ndarray = None) -> MixedPhaseResult:
        """
        Unified solver for mixed phase.
        
        Args:
            n_B: Baryon density (fm⁻³)
            T: Temperature (MeV)
            eta: Local/global neutrality parameter [0, 1]
            eq_type: Type of equilibrium constraint
            Y_C: Fixed charge fraction (for FIXED_YC)
            Y_S: Fixed strangeness fraction (for FIXED_YC_YS)
            Y_L: Fixed lepton fraction (for TRAPPED_NEUTRINOS)
            initial_guess: Optional initial guess
            
        Note on chemical potentials:
            - μ_B is always equal between phases (baryon conservation)
            - μ_C_H ≠ μ_C_Q in general (local neutrality breaks charge conservation)
            - μ_S follows from strangeness equilibrium condition
        """
        if eta < 0 or eta > 1:
            raise ValueError(f"eta must be in [0, 1], got {eta}")
        
        # Dispatch to appropriate equilibrium type
        if eq_type == EquilibriumType.BETA_EQ:
            return self._solve_beta_eq(n_B, T, eta, initial_guess)
        elif eq_type == EquilibriumType.FIXED_YC:
            if Y_C is None:
                raise ValueError("Y_C required for FIXED_YC equilibrium")
            return self._solve_fixed_YC(n_B, Y_C, T, eta, initial_guess)
        elif eq_type == EquilibriumType.FIXED_YC_YS:
            if Y_C is None or Y_S is None:
                raise ValueError("Y_C and Y_S required for FIXED_YC_YS equilibrium")
            return self._solve_fixed_YC_YS(n_B, Y_C, Y_S, T, eta, initial_guess)
        elif eq_type == EquilibriumType.TRAPPED_NEUTRINOS:
            if Y_L is None:
                raise ValueError("Y_L required for TRAPPED_NEUTRINOS equilibrium")
            return self._solve_trapped_nu(n_B, Y_L, T, eta, initial_guess)
        else:
            raise ValueError(f"Unknown equilibrium type: {eq_type}")
    
    def _solve_beta_eq(self, n_B: float, T: float, eta: float,
                       initial_guess: np.ndarray = None) -> MixedPhaseResult:
        """
        Solve mixed phase for β-equilibrium.
        
        Unknowns: [μ_B, μ_eH, μ_eQ, μ_eG, χ] (5 unknowns for 0 < η < 1)
        
        Constraints:
            - μ_B_H = μ_B_Q = μ_B (baryon number globally conserved)
            - β-eq in Q: μ_d - μ_u = μ_eQ → determines μ_C_Q from μ_eQ
            - β-eq in H: μ_p - μ_n = μ_eH → determines μ_C_H from μ_eH  
            - Strangeness eq: μ_s = μ_d → μ_S = 0
            - Local neutrality: n_C_H = n_eH, n_C_Q = n_eQ
            - Global neutrality: (1-χ)n_C_H + χ*n_C_Q = n_eG
            - Baryon conservation: (1-χ)n_B_H + χ*n_B_Q = n_B
            - Pressure equilibrium: P_H + η*P_eH = P_Q + η*P_eQ
        """
        # Warm start: use previous solution if available and conditions match
        if initial_guess is None:
            if (self._last_solution is not None and 
                self._last_T == T and 
                abs(self._last_eta - eta) < 1e-6):
                # Reuse previous solution as initial guess
                initial_guess = self._last_solution.copy()
            else:
                # Default initial guess: [μ_B, μ_eH, μ_eQ, μ_eG, χ]
                initial_guess = np.array([1000.0, 100.0, -100.0, 100.0, -0.1])
        
        def equations(x):
            mu_B, mu_eH, mu_eQ, mu_eG, chi = x
            
            # For β-equilibrium:
            # μ_C_H = μ_eH (from μ_p - μ_n = μ_e)
            # μ_C_Q = -μ_eQ (from μ_d - μ_u = μ_e, but μ_C = μ_u - μ_d = -μ_e)
            # Actually: μ_d - μ_u = η*μ_eQ + (1-η)*μ_eG
            mu_C_H = mu_eH  # β-eq in hadronic phase
            mu_C_Q = -(eta * mu_eQ + (1 - eta) * mu_eG)  # β-eq in quark phase
            mu_S = 0.0  # Strangeness equilibrium
            
            # Get phase thermodynamics
            H = self.hadron_thermo(mu_B, mu_C_H, mu_S, mu_eH, T)
            Q = self.quark_thermo(mu_B, mu_C_Q, mu_S, mu_eQ, T)
            
            # Electron thermodynamics
            e_H = self.electron_thermo(max(abs(mu_eH), 0.1), T)
            e_Q = self.electron_thermo(max(abs(mu_eQ), 0.1), T)
            e_G = self.electron_thermo(max(abs(mu_eG), 0.1), T)
            
            n_eH = e_H['n'] if mu_eH > 0 else -e_H['n']
            n_eQ = e_Q['n'] if mu_eQ > 0 else -e_Q['n']
            n_eG = e_G['n'] if mu_eG > 0 else -e_G['n']
            
            # Build 5 residual equations using existing utilities
            res = np.zeros(5)
            
            # (1) Baryon conservation
            res[0] = baryon_conservation_residual(n_B, H['n_B'], Q['n_B'], chi) / n_B
            
            # (2) Local neutrality H: n_C_H = n_eH
            res[1] = local_neutrality_H_residual(H['n_C'], n_eH) / max(abs(H['n_C']), 1e-10)
            
            # (3) Local neutrality Q: n_C_Q = n_eQ (using general form)
            res[2] = (Q['n_C'] - n_eQ) / max(abs(Q['n_C']), 1e-10)
            
            # (4) Global neutrality: (1-χ)n_C_H + χ*n_C_Q = n_eG
            res[3] = charge_conservation_residual(n_eG, H['n_C'], Q['n_C'], chi) / max(abs(n_eG), 1e-10)
            
            # (5) Pressure equilibrium
            P_H = H['P'] + eta * e_H['P']
            P_Q = Q['P'] + eta * e_Q['P']
            res[4] = check_pressure_equilibrium(P_H, P_Q) / max(abs(P_H), 1.0)
            
            return res

        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sol = root(equations, initial_guess, method='hybr',
                      options={'maxfev': 5000, 'xtol': 1e-10})
        
        # Extract solution
        mu_B, mu_eH, mu_eQ, mu_eG, chi = sol.x
        mu_C_H = mu_eH
        mu_C_Q = -(eta * mu_eQ + (1 - eta) * mu_eG)
        mu_S = 0.0
        
        # Cache solution for warm start
        self._last_solution = sol.x.copy()
        self._last_n_B = n_B
        self._last_T = T
        self._last_eta = eta
        
        # Build result
        return self._build_result_beta(
            n_B, T, eta, chi, mu_B, mu_C_H, mu_C_Q, mu_S,
            mu_eH, mu_eQ, mu_eG, sol.success, np.max(np.abs(sol.fun))
        )

    
    def _solve_fixed_YC(self, n_B: float, Y_C: float, T: float, eta: float,
                        initial_guess: np.ndarray = None) -> MixedPhaseResult:
        """
        Solve mixed phase with fixed charge fraction Y_C.
        
        Unknowns: [μ_B, μ_C, μ_eH, μ_eQ, μ_eG, χ]
        (μ_C is now a free variable, Y_C is the constraint)
        """
        # TODO: Implement fixed Y_C case
        raise NotImplementedError("Fixed Y_C equilibrium not yet implemented")
    
    def _solve_fixed_YC_YS(self, n_B: float, Y_C: float, Y_S: float, T: float, eta: float,
                           initial_guess: np.ndarray = None) -> MixedPhaseResult:
        """
        Solve mixed phase with fixed Y_C and Y_S.
        """
        # TODO: Implement fixed Y_C + Y_S case
        raise NotImplementedError("Fixed Y_C + Y_S equilibrium not yet implemented")
    
    def _solve_trapped_nu(self, n_B: float, Y_L: float, T: float, eta: float,
                          initial_guess: np.ndarray = None) -> MixedPhaseResult:
        """
        Solve mixed phase with trapped neutrinos (fixed Y_L).
        
        Additional constraint: Y_L = (n_e + n_ν) / n_B = const
        Additional unknown: μ_ν (neutrino chemical potential)
        """
        # TODO: Implement trapped neutrino case
        raise NotImplementedError("Trapped neutrinos equilibrium not yet implemented")
    
    def _build_result_beta(self, n_B, T, eta, chi, mu_B, mu_C_H, mu_C_Q, mu_S,
                           mu_eH, mu_eQ, mu_eG, converged, error) -> MixedPhaseResult:
        """Build result for β-equilibrium case."""
        
        H = self.hadron_thermo(mu_B, mu_C_H, mu_S, mu_eH, T)
        Q = self.quark_thermo(mu_B, mu_C_Q, mu_S, mu_eQ, T)
        e_H = self.electron_thermo(max(abs(mu_eH), 0.1), T)
        e_Q = self.electron_thermo(max(abs(mu_eQ), 0.1), T)
        e_G = self.electron_thermo(max(abs(mu_eG), 0.1), T)
        n_eG = e_G['n'] if mu_eG > 0 else -e_G['n']
        
        n_C_total = (1 - chi) * H['n_C'] + chi * Q['n_C']
        Y_C = n_C_total / n_B if n_B > 0 else 0.0
        
        P_H_phase = H['P'] + eta * e_H['P'] + (1 - eta) * e_G['P']
        P_Q_phase = Q['P'] + eta * e_Q['P'] + (1 - eta) * e_G['P']
        e_H_phase = H['e'] + eta * e_H['e'] + (1 - eta) * e_G['e']
        e_Q_phase = Q['e'] + eta * e_Q['e'] + (1 - eta) * e_G['e']
        s_H_phase = H['s'] + eta * e_H['s'] + (1 - eta) * e_G['s']
        s_Q_phase = Q['s'] + eta * e_Q['s'] + (1 - eta) * e_G['s']
        
        P_total = P_H_phase
        e_total = (1 - chi) * e_H_phase + chi * e_Q_phase
        s_total = (1 - chi) * s_H_phase + chi * s_Q_phase
        
        result = MixedPhaseResult(
            converged=converged, error=error,
            n_B=n_B, T=T, eta=eta, Y_C=Y_C, chi=chi,
            n_B_H=H['n_B'], n_C_H=H['n_C'], n_S_H=H.get('n_S', 0.0),
            mu_B_H=mu_B, mu_C_H=mu_C_H, mu_S_H=mu_S,
            P_H=P_H_phase, e_H=e_H_phase, s_H=s_H_phase,
            n_eL_H=H.get('n_e', 0.0), mu_eL_H=mu_eH,
            n_B_Q=Q['n_B'], n_C_Q=Q['n_C'], n_S_Q=Q.get('n_S', 0.0),
            mu_B_Q=mu_B, mu_C_Q=mu_C_Q, mu_S_Q=mu_S,
            P_Q=P_Q_phase, e_Q=e_Q_phase, s_Q=s_Q_phase,
            n_eL_Q=Q.get('n_e', 0.0), mu_eL_Q=mu_eQ,
            n_eG=n_eG, mu_eG=mu_eG,
            P_total=P_total, e_total=e_total, s_total=s_total,
        )
        result.compute_derived()
        return result
    
    # =========================================================================
    # PHASE BOUNDARY AND FULL EOS METHODS
    # =========================================================================
    
    def find_phase_boundaries(self, T: float, eta: float,
                              eq_type: EquilibriumType = EquilibriumType.BETA_EQ,
                              n_B_range: Tuple[float, float] = (0.1, 2.0),
                              n_B_resolution: int = 50) -> Tuple[float, float]:
        """Find phase boundaries n_B1 (χ=0) and n_B2 (χ=1) by scanning density."""
        from scipy.optimize import brentq
        
        n_B_min, n_B_max = n_B_range
        n_B_values = np.linspace(n_B_min, n_B_max, n_B_resolution)
        
        chi_values = []
        for n_B in n_B_values:
            try:
                result = self.solve_mixed(n_B, T, eta, eq_type)
                chi_values.append(result.chi)
            except:
                chi_values.append(np.nan)
        
        chi_values = np.array(chi_values)
        
        # Find n_B1 where chi crosses 0
        n_B1 = n_B_min
        for i in range(len(chi_values) - 1):
            if not np.isnan(chi_values[i]) and not np.isnan(chi_values[i+1]):
                if chi_values[i] < 0 < chi_values[i+1]:
                    try:
                        def chi_residual(n_B):
                            return self.solve_mixed(n_B, T, eta, eq_type).chi
                        n_B1 = brentq(chi_residual, n_B_values[i], n_B_values[i+1])
                    except:
                        n_B1 = (n_B_values[i] + n_B_values[i+1]) / 2
                    break
        
        # Find n_B2 where chi crosses 1
        n_B2 = n_B_max
        for i in range(len(chi_values) - 1):
            if not np.isnan(chi_values[i]) and not np.isnan(chi_values[i+1]):
                if chi_values[i] < 1 < chi_values[i+1]:
                    try:
                        def chi_residual(n_B):
                            return self.solve_mixed(n_B, T, eta, eq_type).chi - 1.0
                        n_B2 = brentq(chi_residual, n_B_values[i], n_B_values[i+1])
                    except:
                        n_B2 = (n_B_values[i] + n_B_values[i+1]) / 2
                    break
        
        return n_B1, n_B2
    
    def solve_full_eos(self, n_B: float, T: float, eta: float,
                       eq_type: EquilibriumType = EquilibriumType.BETA_EQ,
                       phase_boundaries: Tuple[float, float] = None,
                       **kwargs) -> MixedPhaseResult:
        """Solve complete EOS: pure H + mixed + pure Q."""
        if phase_boundaries is None:
            n_B1, n_B2 = self.find_phase_boundaries(T, eta, eq_type)
        else:
            n_B1, n_B2 = phase_boundaries
        
        if n_B < n_B1:
            return self.solve_pure_H(n_B, T, eq_type)
        elif n_B > n_B2:
            return self.solve_pure_Q(n_B, T, eq_type)
        else:
            return self.solve_mixed(n_B, T, eta, eq_type, **kwargs)



# =============================================================================
if __name__ == "__main__":
    print("Mixed Phase EOS Framework Test")
    print("=" * 50)
    
    # Test volume-weighted average functions with example phase densities
    # (Individual particle -> phase densities are computed by specific EOS modules)
    n_B_H = 0.16   # fm^-3 (example hadronic baryon density)
    n_C_H = 0.05   # fm^-3 (example hadronic charge density)
    n_B_Q = 0.30   # fm^-3 (example quark baryon density)
    n_C_Q = 0.00   # fm^-3 (example quark charge density, neutral)
    
    print(f"\nHadronic phase:")
    print(f"  n_B^H = {n_B_H:.4f} fm⁻³")
    print(f"  n_C^H = {n_C_H:.4f} fm⁻³")
    
    print(f"\nQuark phase:")
    print(f"  n_B^Q = {n_B_Q:.4f} fm⁻³")
    print(f"  n_C^Q = {n_C_Q:.4f} fm⁻³")
    
    chi = 0.5
    n_B_total = compute_total_baryon_density(n_B_H, n_B_Q, chi)
    n_C_total = compute_total_charge_density(n_C_H, n_C_Q, chi)
    
    print(f"\nMixed phase (χ = {chi}):")
    print(f"  n_B_total = {n_B_total:.4f} fm⁻³")
    print(f"  n_C_total = {n_C_total:.4f} fm⁻³")
    print(f"  Y_C = {n_C_total/n_B_total:.4f}")
    
    # Test result dataclass
    result = MixedPhaseResult(n_B=0.5, T=10.0, eta=0.5, chi=0.3)
    result.n_B_H = 0.16
    result.n_B_Q = 0.30
    result.compute_derived()
    
    print(f"\nResult test:")
    print(f"  Phase: {result.phase.name}")
    print(f"  n_B_H = {result.n_B_H:.4f}, n_B_Q = {result.n_B_Q:.4f}")
    
    print("\nOK!")

