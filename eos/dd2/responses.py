"""
responses.py
============
Thermodynamic responses, reference flavor: computed by finite difference along
solved sequences, with an independent cross-check per quantity. This is the
oracle `responses_jac.py` is validated against; it is plain
NumPy/SciPy and is never bypassed (CLAUDE.md section 9).

Two speed-of-sound flavors:
  * equilibrium c_s,eq^2 = dP/deps along the beta-eq / fixed-Y sequence
    (composition follows equilibrium);
  * adiabatic/frozen c_s,ad^2 = (dP/deps) at fixed composition (equals the
    fixed-composition value; at T=0 the two differ only through the composition
    derivative).
Plus the adiabatic index Gamma, the thermal index Gamma_th, and the heat
capacities C_V, C_P.
"""
import numpy as np

from eos.dd2.solver import (
    solve_beta_eq_neutrinoless, solve_composition, solve_snm,
)


def sound_speed_eq(par, n_B, flags, T=0.0, rel_dn=1e-3):
    """
    Equilibrium speed of sound squared c_s^2 = dP/deps along the beta-eq
    sequence at (n_B, T). Central finite difference in n_B.
    """
    lo = solve_beta_eq_neutrinoless(par, n_B * (1 - rel_dn), flags, T=T)
    hi = solve_beta_eq_neutrinoless(par, n_B * (1 + rel_dn), flags, T=T)
    return (hi.P - lo.P) / (hi.eps - lo.eps)


def sound_speed_adiabatic(par, n_B, Y_p, T=0.0, rel_dn=1e-3):
    """
    Adiabatic/frozen c_s^2 = (dP/deps) at FIXED composition, for nucleonic
    matter at proton fraction Y_p. Perturbs n_B holding Y_p (hence n_n, n_p
    fractions) fixed via solve_composition.
    """
    def eval_at(n):
        return solve_composition(par, (1 - Y_p) * n, Y_p * n, T=T)
    lo, hi = eval_at(n_B * (1 - rel_dn)), eval_at(n_B * (1 + rel_dn))
    return (hi.P - lo.P) / (hi.eps - lo.eps)


def adiabatic_index(par, n_B, Y_p, T=0.0, rel_dn=1e-3):
    """Gamma = (eps + P)/P * c_s,ad^2 at fixed composition."""
    p = solve_composition(par, (1 - Y_p) * n_B, Y_p * n_B, T=T)
    cs2 = sound_speed_adiabatic(par, n_B, Y_p, T=T, rel_dn=rel_dn)
    return (p.eps + p.P) / p.P * cs2


def thermal_index(par, n_B, flags, T, rel_dn=None):
    """
    Thermal index Gamma_th = 1 + (P - P_cold)/(eps - eps_cold), with the cold
    (T=0) reference at the SAME n_B and beta-eq composition.
    """
    hot = solve_beta_eq_neutrinoless(par, n_B, flags, T=T)
    cold = solve_beta_eq_neutrinoless(par, n_B, flags, T=0.0)
    dP, de = hot.P - cold.P, hot.eps - cold.eps
    if de <= 0:
        return float("nan")
    return 1.0 + dP / de


def heat_capacity_V(par, n_B, flags, T, dT=1e-2):
    """C_V = T (ds/dT)_{n_B, Y} along the beta-eq sequence (central FD in T)."""
    lo = solve_beta_eq_neutrinoless(par, n_B, flags, T=T - dT)
    hi = solve_beta_eq_neutrinoless(par, n_B, flags, T=T + dT)
    return T * (hi.s - lo.s) / (2.0 * dT)


def snm_sound_speed(par, n_B, T=0.0, rel_dn=1e-3):
    """c_s^2 = dP/deps of symmetric nuclear matter (a clean isoscalar probe)."""
    lo, hi = solve_snm(par, n_B * (1 - rel_dn), T=T), \
        solve_snm(par, n_B * (1 + rel_dn), T=T)
    return (hi.P - lo.P) / (hi.eps - lo.eps)
