"""Second-derivative quantities, reference flavour: finite differences along
re-solved sequences.

Plain NumPy/SciPy, re-solving the equilibrium at every perturbed point. That
is the slow and obviously-correct way to take a second derivative, and it is
what any accelerated flavour would be judged against (CLAUDE.md section 9);
DID ships no accelerated flavour.

Every quantity here holds the same thing fixed -- NOTHING. The composition
re-equilibrates under the perturbation, which is the `equilibrium` freeze of
CLAUDE.md section 5. What differs between the two sound speeds is the THERMAL
variable held: at fixed temperature or at fixed entropy per baryon. They
differ at T > 0 by exactly the heat-capacity ratio,

    (dP/dn_B)_S = (C_P/C_V) (dP/dn_B)_T,

which is why the names say which one is meant and why C_V and C_P are computed
here rather than as a separate errand.

References: Typel, Oertel, Klahn et al., CompOSE manual, arXiv:2203.03209
section 3.6; Constantinou, Guerrini, Zhao et al., arXiv:2506.20418 Eqs. 76-77.
"""
from eos.did.solver import solve_mode
from eos.general.thermal_mesons import condensation_message


def _solve(par, n_B, flags, spec, T):
    """One converged point of the stencil, or a raise saying where it failed.

    An internal layer may raise (CLAUDE.md section 6); a response function
    asked for at a state the model cannot reach is a caller error, not a
    sampler's bad draw, so this is not turned into a status.
    """
    point = solve_mode(par, n_B, flags, spec, T=T)
    if not point.converged:
        if point.condensation >= 1.0:
            raise RuntimeError(
                "the response stencil needs a converged neighbour and "
                + condensation_message(point.condensation, n_B, T))
        raise RuntimeError(
            f"the response stencil needs a converged neighbour and the solve "
            f"at n_B={n_B:g} fm^-3, T={T:g} MeV did not converge "
            f"(residual {point.error:.3e})")
    return point


def sequence_derivs(par, n_B, flags, spec, T, rel_dn=1e-3, dT=0.05):
    """The first derivatives every response below is built from.

    Central differences along the sequence `spec` declares, with a full
    re-solve at each neighbour:

        dP_dn, de_dn, dsig_dn    at fixed T
        dP_dT, dsig_dT           at fixed n_B (zero at T = 0, where there is
                                 no cold side to difference against)

    sigma = s/n_B is the entropy PER BARYON rather than the entropy density,
    because C_P is taken at fixed pressure, where the volume changes.
    """
    dn = rel_dn * n_B
    hi = _solve(par, n_B + dn, flags, spec, T)
    lo = _solve(par, n_B - dn, flags, spec, T)
    out = dict(dP_dn=(hi.P - lo.P) / (2.0 * dn),
               de_dn=(hi.eps - lo.eps) / (2.0 * dn),
               dsig_dn=(hi.entropy_per_baryon - lo.entropy_per_baryon)
               / (2.0 * dn),
               dP_dT=0.0, dsig_dT=0.0)
    if T > 0.0:
        hot = _solve(par, n_B, flags, spec, T + dT)
        cold = _solve(par, n_B, flags, spec, T - dT)
        out["dP_dT"] = (hot.P - cold.P) / (2.0 * dT)
        out["dsig_dT"] = ((hot.entropy_per_baryon - cold.entropy_per_baryon)
                          / (2.0 * dT))
    return out


def sound_speed_isothermal(par, n_B, flags, spec, T=0.0, rel_dn=1e-3):
    """c_s^2 = (dP/dn_B)_T / (deps/dn_B)_T along the sequence.

    At T = 0 this IS the sound speed; at T > 0 it is the isothermal one, which
    is smaller than the adiabatic one by C_V/C_P.
    """
    d = sequence_derivs(par, n_B, flags, spec, T, rel_dn=rel_dn)
    return d["dP_dn"] / d["de_dn"]


def heat_capacity_V(par, n_B, flags, spec, T, dT=0.05):
    """C_V = T (d(s/n_B)/dT) at fixed n_B, per baryon [dimensionless]."""
    return T * sequence_derivs(par, n_B, flags, spec, T, dT=dT)["dsig_dT"]


def heat_capacity_P(par, n_B, flags, spec, T, dT=0.05, rel_dn=1e-3):
    """C_P = T (d(s/n_B)/dT) at fixed PRESSURE, per baryon.

    The pressure is held by letting the density move with the temperature:

        (dsigma/dT)_P = (dsigma/dT)_n - (dP/dT)_n (dsigma/dn)_T / (dP/dn)_T,

    the usual Jacobian rotation, with every partial derivative taken along the
    same re-solved sequence.
    """
    d = sequence_derivs(par, n_B, flags, spec, T, rel_dn=rel_dn, dT=dT)
    dsig_dT_P = d["dsig_dT"] - d["dP_dT"] * d["dsig_dn"] / d["dP_dn"]
    return T * dsig_dT_P


def sound_speed_adiabatic(par, n_B, flags, spec, T=0.0, dT=0.05, rel_dn=1e-3):
    """c_s^2 at fixed entropy per baryon: the one a sound wave travels at.

        c_s,ad^2 = (C_P/C_V) c_s,iso^2

    At T = 0 the ratio is 1 by construction and the two coincide, so the cold
    limit needs no special case.
    """
    if T <= 0.0:
        return sound_speed_isothermal(par, n_B, flags, spec, T, rel_dn=rel_dn)
    d = sequence_derivs(par, n_B, flags, spec, T, rel_dn=rel_dn, dT=dT)
    C_V = T * d["dsig_dT"]
    C_P = T * (d["dsig_dT"] - d["dP_dT"] * d["dsig_dn"] / d["dP_dn"])
    return (C_P / C_V) * d["dP_dn"] / d["de_dn"]


def thermal_index(par, n_B, flags, spec, T):
    """Gamma_th = 1 + (P - P_cold)/(eps - eps_cold) at the same n_B.

    The cold reference is the same mode at T = 0, which is what a simulation's
    thermal-pressure prescription is calibrated against.
    """
    hot = _solve(par, n_B, flags, spec, T)
    cold = _solve(par, n_B, flags, spec, 0.0)
    d_eps = hot.eps - cold.eps
    if d_eps <= 0.0:
        return float("nan")
    return 1.0 + (hot.P - cold.P) / d_eps
