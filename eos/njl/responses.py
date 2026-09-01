"""Second-derivative quantities, reference flavour: finite differences along
re-solved sequences.

Plain NumPy/SciPy, re-solving the equilibrium at every perturbed point. That is
the slow and obviously-correct way to take a second derivative, and it is what
any accelerated flavour would be judged against (CLAUDE.md section 9); NJL
ships no accelerated flavour.

Every quantity here holds the same thing fixed -- NOTHING. The composition
re-equilibrates under the perturbation, and so does the PAIRING PATTERN: each
neighbour of the stencil re-runs the enumeration, so a derivative taken across
a pattern boundary sees the change. That is the honest `equilibrium` freeze
for a model that pairs, and it is also why these derivatives are one-sided in
meaning at such a boundary -- a central difference straddling a first-order
change in pattern returns the chord, not the tangent. A caller wanting the
derivative WITHIN one pattern passes `patterns=(name,)`.

What differs between the two sound speeds is the THERMAL variable held: fixed
temperature, or fixed entropy per baryon. They differ at T > 0 by exactly the
heat-capacity ratio,

    (dP/dn_B)_S = (C_P/C_V) (dP/dn_B)_T,

which is why the names say which one is meant and why C_V and C_P are computed
here rather than as a separate errand.

The asymptotic behaviour these measure is the point of the vector sector: with
a constant G_V, c_s^2 runs to 1, and with the density-dependent forms of
`couplings.py` it settles on max(1 - alpha, 1/3). Pairing does not change it:
c_s^2 -> 1/3 + (2/9) Delta^2/mu^2, which is 2e-3 at mu = 500 MeV and dies as
1/mu^2.

References: Typel, Oertel, Klahn et al., CompOSE manual, arXiv:2203.03209
section 3.6.
"""
from eos.njl.solver import solve


def _solve(par, flags, mode, n_B, T, patterns, fractions,
           backend="reference", _memo=None):
    """One converged point of the stencil, or a raise saying where it failed.

    An internal layer may raise (CLAUDE.md section 6); a response function
    asked for at a state the model cannot reach is a caller error, not a
    sampler's bad draw, so this is not turned into a status.

    `_memo`, when a caller supplies a dict, caches converged points on
    (n_B, T): the six response functions of one `eos_response` call evaluate
    the SAME six stencil states eighteen times between them, and a re-solve of
    identical arguments returns the identical point, so handing back the
    cached one changes no number. The memo is valid only while (par, flags,
    mode, patterns, fractions, backend) are all held fixed, which is why it is
    created per `eos_response` call and never module state (CLAUDE.md
    section 6: no global mutable state).
    """
    if _memo is not None and (float(n_B), float(T)) in _memo:
        return _memo[(float(n_B), float(T))]
    point = solve(par, mode, n_B, T, flags, patterns=patterns,
                  backend=backend, **fractions)
    if not point.converged:
        raise RuntimeError(
            f"the response stencil needs a converged neighbour and the solve "
            f"at n_B={n_B:g} fm^-3, T={T:g} MeV did not converge "
            f"(residual {point.error:.3e})")
    if _memo is not None:
        _memo[(float(n_B), float(T))] = point
    return point


def sequence_derivs(par, flags, mode, n_B, T, rel_dn=1e-3, dT=0.05,
                    patterns=None, backend="reference", _memo=None,
                    **fractions):
    """The first derivatives every response below is built from.

    Central differences along the mode's own sequence, with a full re-solve at
    each neighbour:

        dP_dn, de_dn, dsig_dn    at fixed T
        dP_dT, dsig_dT           at fixed n_B (zero at T = 0, where there is
                                 no cold side to difference against)

    sigma = s/n_B is the entropy PER BARYON rather than the entropy density,
    because C_P is taken at fixed pressure, where the volume changes.
    """
    dn = rel_dn * n_B

    def at(n, temp):
        return _solve(par, flags, mode, n, temp, patterns, fractions,
                      backend=backend, _memo=_memo)

    hi, lo = at(n_B + dn, T), at(n_B - dn, T)
    sigma = (lambda p: p.s / p.n_B if p.n_B else 0.0)
    out = dict(dP_dn=(hi.P - lo.P) / (2.0 * dn),
               de_dn=(hi.eps - lo.eps) / (2.0 * dn),
               dsig_dn=(sigma(hi) - sigma(lo)) / (2.0 * dn),
               dP_dT=0.0, dsig_dT=0.0)
    if T > 0.0:
        hot, cold = at(n_B, T + dT), at(n_B, T - dT)
        out["dP_dT"] = (hot.P - cold.P) / (2.0 * dT)
        out["dsig_dT"] = (sigma(hot) - sigma(cold)) / (2.0 * dT)
    return out


def sound_speed_isothermal(par, flags, mode, n_B, T=0.0, rel_dn=1e-3,
                           dT=0.05, patterns=None, backend="reference",
                           _memo=None, **fractions):
    """c_s^2 = (dP/dn_B)_T / (deps/dn_B)_T along the sequence.

    At T = 0 this IS the sound speed; at T > 0 it is the isothermal one, which
    is smaller than the adiabatic one by C_V/C_P.
    """
    d = sequence_derivs(par, flags, mode, n_B, T, rel_dn=rel_dn, dT=dT,
                        patterns=patterns, backend=backend, _memo=_memo,
                        **fractions)
    return d["dP_dn"] / d["de_dn"]


def heat_capacity_V(par, flags, mode, n_B, T, dT=0.05, rel_dn=1e-3,
                    patterns=None, backend="reference", _memo=None,
                    **fractions):
    """C_V = T (d(s/n_B)/dT) at fixed n_B, per baryon [dimensionless].

    In a fully gapped phase this is exponentially small at low T -- the
    paired entropy is e^(-Delta/T) -- and that suppression is real physics,
    not a numerical failure: it is what makes a colour superconductor cool
    differently from unpaired quark matter.
    """
    return T * sequence_derivs(par, flags, mode, n_B, T, rel_dn=rel_dn,
                               dT=dT, patterns=patterns, backend=backend,
                               _memo=_memo, **fractions)["dsig_dT"]


def heat_capacity_P(par, flags, mode, n_B, T, dT=0.05, rel_dn=1e-3,
                    patterns=None, backend="reference", _memo=None,
                    **fractions):
    """C_P = T (d(s/n_B)/dT) at fixed PRESSURE, per baryon.

    The pressure is held by letting the density move with the temperature:

        (dsigma/dT)_P = (dsigma/dT)_n - (dP/dT)_n (dsigma/dn)_T / (dP/dn)_T,

    the usual Jacobian rotation, with every partial derivative taken along the
    same re-solved sequence.
    """
    d = sequence_derivs(par, flags, mode, n_B, T, rel_dn=rel_dn, dT=dT,
                        patterns=patterns, backend=backend, _memo=_memo,
                        **fractions)
    return T * (d["dsig_dT"] - d["dP_dT"] * d["dsig_dn"] / d["dP_dn"])


def sound_speed_adiabatic(par, flags, mode, n_B, T=0.0, dT=0.05, rel_dn=1e-3,
                          patterns=None, backend="reference", _memo=None,
                          **fractions):
    """c_s^2 at fixed entropy per baryon: the one a sound wave travels at.

        c_s,ad^2 = (C_P/C_V) c_s,iso^2

    At T = 0 the ratio is 1 by construction and the two coincide, so the cold
    limit needs no special case.
    """
    if T <= 0.0:
        return sound_speed_isothermal(par, flags, mode, n_B, T,
                                      rel_dn=rel_dn, dT=dT, patterns=patterns,
                                      backend=backend, _memo=_memo,
                                      **fractions)
    d = sequence_derivs(par, flags, mode, n_B, T, rel_dn=rel_dn, dT=dT,
                        patterns=patterns, backend=backend, _memo=_memo,
                        **fractions)
    C_V = T * d["dsig_dT"]
    C_P = T * (d["dsig_dT"] - d["dP_dT"] * d["dsig_dn"] / d["dP_dn"])
    return (C_P / C_V) * d["dP_dn"] / d["de_dn"]


def thermal_index(par, flags, mode, n_B, T, patterns=None,
                  backend="reference", _memo=None, **fractions):
    """Gamma_th = 1 + (P - P_cold)/(eps - eps_cold) at the same n_B.

    The cold reference is the same mode at T = 0, which is what a simulation's
    thermal-pressure prescription is calibrated against.
    """
    hot = _solve(par, flags, mode, n_B, T, patterns, fractions,
                 backend=backend, _memo=_memo)
    cold = _solve(par, flags, mode, n_B, 0.0, patterns, fractions,
                  backend=backend, _memo=_memo)
    d_eps = hot.eps - cold.eps
    if d_eps <= 0.0:
        return float("nan")
    return 1.0 + (hot.P - cold.P) / d_eps
