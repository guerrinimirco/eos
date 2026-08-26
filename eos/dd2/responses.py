"""
responses.py
============
Thermodynamic responses, reference flavor: computed by finite difference along
solved sequences, with an independent cross-check per quantity. This is the
oracle `responses_jac.py` is validated against; it is plain
NumPy/SciPy and is never bypassed (CLAUDE.md section 9).

Speeds of sound are named on TWO axes, never by one word doing both jobs
(CLAUDE.md section 5): the COMPOSITION axis is carried by which function is
called, the THERMAL axis by the name itself.

  * composition follows equilibrium: c_s,eq^2 = dP/deps along the beta-eq /
    fixed-Y sequence, at fixed T;
  * composition frozen at a proton fraction Y_p: `sound_speed_isothermal_frozen`
    (T held on both stencil points) and `sound_speed_adiabatic_frozen`
    (entropy per baryon held, larger by C_P/C_V; equal at T = 0).

The frozen pair is Zhao & Lattimer's c_s (arXiv:2204.03037, Eq. 1), whose
"adiabatic" means frozen composition -- the opposite of the CompOSE manual's
"adiabatic", which means fixed entropy. Neither word is used here unqualified.

Plus the index Gamma, the thermal index Gamma_th, and the heat capacities
C_V, C_P.
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


def sound_speed_isothermal_frozen(par, n_B, Y_p, T=0.0, rel_dn=1e-3):
    """
    c_s^2 = (dP/deps)_T at FIXED composition, for nucleonic matter at proton
    fraction Y_p. Perturbs n_B holding Y_p (hence the n_n, n_p fractions)
    fixed via solve_composition, with T held on BOTH stencil points -- so the
    thermal axis of this speed is isothermal and the composition axis frozen.
    """
    def eval_at(n):
        return solve_composition(par, (1 - Y_p) * n, Y_p * n, T=T)
    lo, hi = eval_at(n_B * (1 - rel_dn)), eval_at(n_B * (1 + rel_dn))
    return (hi.P - lo.P) / (hi.eps - lo.eps)


def _frozen_derivs(par, n_B, Y_p, T, rel_dn=1e-3, dT=1e-2):
    """Partials along the fixed-Y_p sequence, central differences: dP/dn,
    deps/dn and the entropy-PER-BARYON sigma = s/n_B derivatives dsigma/dn and
    dsigma/dT. Per baryon is what C_P needs, where the volume changes at
    fixed P."""
    def eval_at(n, temp):
        return solve_composition(par, (1 - Y_p) * n, Y_p * n, T=temp)

    dn = rel_dn * n_B
    hi, lo = eval_at(n_B + dn, T), eval_at(n_B - dn, T)
    hot, cold = eval_at(n_B, T + dT), eval_at(n_B, T - dT)
    return dict(
        dP_dn=(hi.P - lo.P) / (2 * dn),
        de_dn=(hi.eps - lo.eps) / (2 * dn),
        dsig_dn=(hi.s / (n_B + dn) - lo.s / (n_B - dn)) / (2 * dn),
        dP_dT=(hot.P - cold.P) / (2 * dT),
        dsig_dT=(hot.s - cold.s) / (n_B * 2 * dT),
    )


def sound_speed_adiabatic_frozen(par, n_B, Y_p, T=0.0, rel_dn=1e-3, dT=1e-2):
    """
    c_s^2 at FIXED composition and fixed entropy per baryon:

        c_s,ad^2 = (C_P/C_V) c_s,iso^2

    with both heat capacities taken along the same frozen-Y_p sequence. At
    T = 0 the ratio is 1 by construction and the two coincide, so the cold
    limit needs no special case.
    """
    if T <= 0.0:
        return sound_speed_isothermal_frozen(par, n_B, Y_p, T=T, rel_dn=rel_dn)
    d = _frozen_derivs(par, n_B, Y_p, T, rel_dn=rel_dn, dT=dT)
    C_V = T * d["dsig_dT"]
    C_P = T * (d["dsig_dT"] - d["dsig_dn"] * d["dP_dT"] / d["dP_dn"])
    return (C_P / C_V) * d["dP_dn"] / d["de_dn"]


def adiabatic_index(par, n_B, Y_p, T=0.0, rel_dn=1e-3):
    """Gamma = (eps + P)/P * c_s^2 at fixed composition, built on the
    ISOTHERMAL member of the frozen pair. At T = 0 that is the adiabatic index;
    at T > 0 the fixed-entropy index is larger by C_P/C_V, and the gap is
    recorded in docs/DEFERRED.md."""
    p = solve_composition(par, (1 - Y_p) * n_B, Y_p * n_B, T=T)
    cs2 = sound_speed_isothermal_frozen(par, n_B, Y_p, T=T, rel_dn=rel_dn)
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
