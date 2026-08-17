"""
responses.py
============
Thermodynamic responses, reference flavour: finite differences along solved
sequences. Plain NumPy/SciPy, re-solving the equilibrium at every perturbed
point, which is the slow and obviously-correct way to take a second derivative
and is what the accelerated flavours are judged against (CLAUDE.md section 9).

Every quantity here holds the same thing fixed -- nothing. The composition
re-equilibrates under the perturbation, which is the `equilibrium` freeze of
CLAUDE.md section 5, and the perturbation is taken at fixed TEMPERATURE. That
matters and is in the names: c_s^2 at fixed T and c_s^2 at fixed entropy per
baryon differ at T > 0 by exactly the heat-capacity ratio, and only agree in
the cold limit.

    (dP/dn_B)_S = (C_P/C_V) (dP/dn_B)_T

is the relation between them (Constantinou, Guerrini et al.,
arXiv:2506.20418 Eq. 76-77; Typel et al., arXiv:2203.03209 section 3.6), and it
is why C_V and C_P are computed here rather than being a separate errand.

References:
- Typel, Oertel, Klahn et al., CompOSE manual, arXiv:2203.03209
- Constantinou, Guerrini, Zhao et al., arXiv:2506.20418
"""
import numpy as np

from eos.sfho.solver import solve_mode


def _solve(par, n_B, flags, spec, T):
    """One converged point of the sequence, or a raise saying where it failed.

    An internal layer may raise (CLAUDE.md section 6); `api.eos_response`
    catches nothing here because a response function asked for at a state the
    model cannot reach is a caller error, not a sampler's bad draw.
    """
    point = solve_mode(par, n_B, flags, spec, T=T)
    if not point.converged:
        raise RuntimeError(
            f"the response stencil needs a converged neighbour and the solve "
            f"at n_B={n_B:g} fm^-3, T={T:g} MeV did not converge "
            f"(residual {point.error:.3e})")
    return point


def sequence_derivs(par, n_B, flags, spec, T, rel_dn=1e-3, dT=0.05):
    """The first derivatives every response below is built from.

    Central differences along the sequence `spec` declares, with a full
    re-solve at each of the four neighbours:

        dP_dn, de_dn, dsig_dn    at fixed T
        dP_dT, dsig_dT           at fixed n_B (zero at T = 0, where there is
                                 no cold side to difference against)

    sigma = s/n_B is the entropy PER BARYON rather than the entropy density,
    because C_P is taken at fixed pressure, where the volume changes.
    """
    dn = rel_dn * n_B
    hi = _solve(par, n_B + dn, flags, spec, T)
    lo = _solve(par, n_B - dn, flags, spec, T)
    d = dict(
        dP_dn=(hi.P - lo.P) / (2.0 * dn),
        de_dn=(hi.eps - lo.eps) / (2.0 * dn),
        dsig_dn=(hi.entropy_per_baryon - lo.entropy_per_baryon) / (2.0 * dn),
        dP_dT=0.0, dsig_dT=0.0,
    )
    if T > 0.0:
        hot = _solve(par, n_B, flags, spec, T + dT)
        cold = _solve(par, n_B, flags, spec, T - dT)
        d["dP_dT"] = (hot.P - cold.P) / (2.0 * dT)
        d["dsig_dT"] = ((hot.entropy_per_baryon - cold.entropy_per_baryon)
                        / (2.0 * dT))
    return d


def sound_speed_isothermal(par, n_B, flags, spec, T=0.0, rel_dn=1e-3):
    """c_s^2 = (dP/dn_B)_T / (deps/dn_B)_T along the sequence [dimensionless].

    The composition re-equilibrates as the density is perturbed, so this is the
    EQUILIBRIUM sound speed, and it is taken at fixed temperature. At T = 0 it
    is the sound speed; at T > 0 the adiabatic one is larger by C_P/C_V.
    """
    d = sequence_derivs(par, n_B, flags, spec, T, rel_dn=rel_dn)
    return d["dP_dn"] / d["de_dn"]


def heat_capacity_V(par, n_B, flags, spec, T, dT=0.05):
    """C_V = T (d(s/n_B)/dT)_{n_B} per baryon [dimensionless]."""
    d = sequence_derivs(par, n_B, flags, spec, T, dT=dT)
    return T * d["dsig_dT"]


def heat_capacity_P(par, n_B, flags, spec, T, dT=0.05, rel_dn=1e-3):
    """C_P = T (d(s/n_B)/dT)_P per baryon [dimensionless].

    Holding the pressure rather than the density,

        (dsig/dT)_P = (dsig/dT)_n - (dsig/dn)_T (dP/dT)_n / (dP/dn)_T

    which is the triple-product rule, so C_P >= C_V comes out rather than
    being imposed.
    """
    d = sequence_derivs(par, n_B, flags, spec, T, rel_dn=rel_dn, dT=dT)
    dsig_dT_P = d["dsig_dT"] - d["dsig_dn"] * d["dP_dT"] / d["dP_dn"]
    return T * dsig_dT_P


def sound_speed_adiabatic(par, n_B, flags, spec, T, dT=0.05, rel_dn=1e-3):
    """c_s^2 at fixed entropy per baryon [dimensionless].

    Two standard steps. The pressure derivative rescales by the heat-capacity
    ratio, (dP/dn)_S = (C_P/C_V)(dP/dn)_T; the energy derivative at fixed s/n_B
    follows from the first law together with the Euler relation of
    CLAUDE.md section 8, which give (deps/dn_B)_S = (eps + P)/n_B. Hence

        c_s,ad^2 = (C_P/C_V) (dP/dn_B)_T n_B / (eps + P).

    At T = 0 the ratio is 1 and this is `sound_speed_isothermal`.
    """
    if T <= 0.0:
        return sound_speed_isothermal(par, n_B, flags, spec, T=T,
                                      rel_dn=rel_dn)
    d = sequence_derivs(par, n_B, flags, spec, T, rel_dn=rel_dn, dT=dT)
    C_V = T * d["dsig_dT"]
    C_P = T * (d["dsig_dT"] - d["dsig_dn"] * d["dP_dT"] / d["dP_dn"])
    here = _solve(par, n_B, flags, spec, T)
    return (C_P / C_V) * d["dP_dn"] * n_B / (here.eps + here.P)


def thermal_index(par, n_B, flags, spec, T):
    """Gamma_th = 1 + (P - P_cold)/(eps - eps_cold) at the same n_B.

    The cold reference is the SAME mode at T = 0, so the difference is the
    thermal part of the pressure and of the energy and nothing else. Returns
    nan where the thermal energy is not positive, which is the honest answer
    rather than a large ratio of two round-off differences.
    """
    hot = _solve(par, n_B, flags, spec, T)
    cold = _solve(par, n_B, flags, spec, 0.0)
    d_eps = hot.eps - cold.eps
    if d_eps <= 0.0:
        return float("nan")
    return 1.0 + (hot.P - cold.P) / d_eps
