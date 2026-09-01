"""Second-derivative quantities, reference flavour: finite differences along
re-solved sequences.

Plain NumPy/SciPy, re-solving the equilibrium at every perturbed point. That is
the slow and obviously-correct way to take a second derivative, and it is what
any accelerated flavour would be judged against (CLAUDE.md section 9); CCDM
ships no accelerated flavour.

Every quantity here holds the same thing fixed -- NOTHING. The composition
re-equilibrates under the perturbation, and so do BOTH enumerations: each
neighbour of the stencil re-runs the branch and pattern comparison, so a
derivative taken across a boundary sees the change.

WHICH IS EXACTLY WHY A CENTRAL DIFFERENCE MUST NOT STRADDLE THE DECONFINEMENT
TRANSITION. It is first order in this model: the fields move discontinuously,
so a stencil with one foot on each side returns the chord across the jump, not
a tangent to either branch. The specification says it in one line -- take
one-sided derivatives on each branch and leave the transition as a gap -- and
the mechanical way to obey it here is to pass `branches=(name,)`, which
restricts the enumeration and makes the derivative a within-branch one. The
same applies to a pairing boundary and `patterns=(name,)`.

`branch_changed` below is the check that says whether a given stencil did
straddle one; a caller taking sound speeds across a table should run it rather
than trust a smooth-looking curve.

What differs between the two sound speeds is the THERMAL variable held: fixed
temperature, or fixed entropy per baryon. They differ at T > 0 by exactly the
heat-capacity ratio,

    (dP/dn_B)_S = (C_P/C_V) (dP/dn_B)_T,

which is why the names say which one is meant and why C_V and C_P are computed
here rather than as a separate errand.

References: Typel, Oertel, Klahn et al., CompOSE manual, arXiv:2203.03209
section 3.6.
"""
from eos.ccdm.solver import solve


def _solve(par, flags, mode, n_B, T, branches, patterns, fractions,
           backend="reference", _memo=None):
    """One converged point of the stencil, or a raise saying where it failed.

    An internal layer may raise (CLAUDE.md section 6); a response function
    asked for at a state the model cannot reach is a caller error, not a
    sampler's bad draw, so this is not turned into a status.

    `_memo`, when a caller supplies a dict, caches converged points on
    (n_B, T): the response functions of one `eos_response` call evaluate the
    SAME six stencil states twenty times between them, and a re-solve of
    identical arguments returns the identical point, so handing back the
    cached one changes no number. The memo is valid only while (par, flags,
    mode, branches, patterns, fractions, backend) are all held fixed, which is
    why it is created per `eos_response` call and never module state
    (CLAUDE.md section 6: no global mutable state).
    """
    if _memo is not None and (float(n_B), float(T)) in _memo:
        return _memo[(float(n_B), float(T))]
    point = solve(par, mode, n_B, T, flags, branches=branches,
                  patterns=patterns, backend=backend, **fractions)
    if not point.converged:
        raise RuntimeError(
            f"the response stencil needs a converged neighbour and the solve "
            f"at n_B={n_B:g} fm^-3, T={T:g} MeV did not converge "
            f"(residual {point.error:.3e}). Below the deconfinement onset "
            f"this model has no deconfined root at fixed density at all, "
            f"which is physics rather than a solver failure")
    if _memo is not None:
        _memo[(float(n_B), float(T))] = point
    return point


def branch_changed(par, flags, mode, n_B, T=0.0, rel_dn=1e-3, branches=None,
                   patterns=None, backend="reference", _memo=None,
                   **fractions):
    """Does the density stencil at this point straddle a branch or pattern
    change?

    True means every derivative below is a chord across a first-order jump
    rather than a tangent, and the answer should be recomputed one-sided --
    by restricting `branches` (and `patterns`) to the side wanted. Cheap
    enough to run before trusting a sound speed, and there is no way to detect
    it from the returned number.
    """
    dn = rel_dn * n_B
    lo = _solve(par, flags, mode, n_B - dn, T, branches, patterns, fractions,
                backend=backend, _memo=_memo)
    hi = _solve(par, flags, mode, n_B + dn, T, branches, patterns, fractions,
                backend=backend, _memo=_memo)
    return (lo.branch, lo.pattern) != (hi.branch, hi.pattern)


def sequence_derivs(par, flags, mode, n_B, T, rel_dn=1e-3, dT=0.05,
                    branches=None, patterns=None, backend="reference",
                    _memo=None, **fractions):
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
        return _solve(par, flags, mode, n, temp, branches, patterns, fractions,
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
                           dT=0.05, branches=None, patterns=None,
                           backend="reference", _memo=None, **fractions):
    """c_s^2 = (dP/dn_B)_T / (deps/dn_B)_T along the sequence.

    At T = 0 this IS the sound speed; at T > 0 it is the isothermal one, which
    is smaller than the adiabatic one by C_V/C_P.

    IT MAY COME OUT NEGATIVE on a raw branch between the deconfinement onset
    and the point where the deconfined branch turns around, because dP/dn_B is
    genuinely negative there -- the mechanically unstable side of a first-order
    transition. That is real physics and CLAUDE.md section 8 admits it in a
    raw branch; what removes it is a construction (Maxwell, Gibbs, or the
    eta-mixed phase of `eos.mixed`), applied before any table reaches a
    structure solver.
    """
    d = sequence_derivs(par, flags, mode, n_B, T, rel_dn=rel_dn, dT=dT,
                        branches=branches, patterns=patterns, backend=backend,
                        _memo=_memo, **fractions)
    return d["dP_dn"] / d["de_dn"]


def heat_capacity_V(par, flags, mode, n_B, T, dT=0.05, rel_dn=1e-3,
                    branches=None, patterns=None, backend="reference",
                    _memo=None, **fractions):
    """C_V = T (d(s/n_B)/dT) at fixed n_B, per baryon [dimensionless].

    In a fully gapped phase this is exponentially small at low T -- the paired
    entropy is e^(-Delta/T) -- and that suppression is real physics, not a
    numerical failure: it is what makes a colour superconductor cool
    differently from unpaired quark matter.
    """
    return T * sequence_derivs(par, flags, mode, n_B, T, rel_dn=rel_dn,
                               dT=dT, branches=branches, patterns=patterns,
                               backend=backend, _memo=_memo,
                               **fractions)["dsig_dT"]


def heat_capacity_P(par, flags, mode, n_B, T, dT=0.05, rel_dn=1e-3,
                    branches=None, patterns=None, backend="reference",
                    _memo=None, **fractions):
    """C_P = T (d(s/n_B)/dT) at fixed PRESSURE, per baryon.

    The pressure is held by letting the density move with the temperature:

        (dsigma/dT)_P = (dsigma/dT)_n - (dP/dT)_n (dsigma/dn)_T / (dP/dn)_T,

    the usual Jacobian rotation, with every partial derivative taken along the
    same re-solved sequence.
    """
    d = sequence_derivs(par, flags, mode, n_B, T, rel_dn=rel_dn, dT=dT,
                        branches=branches, patterns=patterns, backend=backend,
                        _memo=_memo, **fractions)
    return T * (d["dsig_dT"] - d["dP_dT"] * d["dsig_dn"] / d["dP_dn"])


def sound_speed_adiabatic(par, flags, mode, n_B, T=0.0, dT=0.05, rel_dn=1e-3,
                          branches=None, patterns=None, backend="reference",
                          _memo=None, **fractions):
    """c_s^2 at fixed entropy per baryon: the one a sound wave travels at.

        c_s,ad^2 = (C_P/C_V) c_s,iso^2

    At T = 0 the ratio is 1 by construction and the two coincide, so the cold
    limit needs no special case.
    """
    if T <= 0.0:
        return sound_speed_isothermal(par, flags, mode, n_B, T,
                                      rel_dn=rel_dn, dT=dT, branches=branches,
                                      patterns=patterns, backend=backend,
                                      _memo=_memo, **fractions)
    d = sequence_derivs(par, flags, mode, n_B, T, rel_dn=rel_dn, dT=dT,
                        branches=branches, patterns=patterns, backend=backend,
                        _memo=_memo, **fractions)
    C_V = T * d["dsig_dT"]
    C_P = T * (d["dsig_dT"] - d["dP_dT"] * d["dsig_dn"] / d["dP_dn"])
    return (C_P / C_V) * d["dP_dn"] / d["de_dn"]


def thermal_index(par, flags, mode, n_B, T, branches=None, patterns=None,
                  backend="reference", _memo=None, **fractions):
    """Gamma_th = 1 + (P - P_cold)/(eps - eps_cold) at the same n_B.

    The cold reference is the same mode at T = 0, which is what a simulation's
    thermal-pressure prescription is calibrated against. Returns NaN where the
    cold and hot states are on different branches and the difference is not a
    thermal one at all.
    """
    hot = _solve(par, flags, mode, n_B, T, branches, patterns, fractions,
                 backend=backend, _memo=_memo)
    cold = _solve(par, flags, mode, n_B, 0.0, branches, patterns, fractions,
                  backend=backend, _memo=_memo)
    if (hot.branch, hot.pattern) != (cold.branch, cold.pattern):
        return float("nan")
    d_eps = hot.eps - cold.eps
    if d_eps <= 0.0:
        return float("nan")
    return 1.0 + (hot.P - cold.P) / d_eps
