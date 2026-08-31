"""The nine colour-dielectric medium integrals, compiled.

The SAME quadrature `eos.general.fermi_integrals.kinetic_thermo` performs,
written out so numba can compile it: the reference path is ~10 NumPy calls on
~120-element arrays per mode, and at that size the dispatch costs more than the
arithmetic. `eos.ccdm.verify.run_full_check` checks the two against each other.

CLAUDE.md section 5 defines `backends/` by the property that deleting it
changes no number, only the time they take, and section 9 by the reference
flavour being what correctness is judged against. Both hold here: with this
directory gone, `backend='fast'` raises and `eos/ccdm/thermodynamics.py` is the
whole story.

**This is not `eos.njl.backends.kernel_numba`, and it does not import it.**
The two models integrate the same ideal gas but bound it differently, and the
difference is per mode rather than per model:

  * njl is a CUT theory -- one cutoff, `par.Lambda_medium`, shared by all nine
    modes and a parameter of the model. ccdm is UNREGULARISED, so its ceiling
    is the numerical one of `unbounded_k_max`, recomputed for EVERY mode from
    that mode's own potential and effective mass;
  * ccdm additionally carries the `ABSENT_WIDTHS` test, which is the
    confinement mechanism at T > 0: the confined branch drives M* to 1e15 MeV,
    where the occupation is e^-1e13 and integrating is not wrong, merely
    pointless. njl has no such branch and no such test.

A kernel written for one of those is wrong for the other, which is why there
are two.

Units are natural throughout, as in the physics modules this serves: momenta,
masses and potentials in MeV, densities in MeV^3, eps and P in MeV^4.
"""
import numpy as np

try:
    from numba import njit
    NUMBA_OK = True
except ImportError:                       # pragma: no cover - numba is a dep
    NUMBA_OK = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]

        def deco(f):
            return f
        return deco


_PI2 = np.pi ** 2

#: Mirrors `eos.general.fermi_integrals.THERMAL_COLLAR` and `_BREAK_TOL`.
_COLLAR = 25.0
_BREAK_TOL = 1.0e-9

#: Mirrors `eos.ccdm.thermodynamics.ABSENT_WIDTHS`.
_ABSENT_WIDTHS = 60.0

#: Mirrors the `pad` default of `eos.general.fermi_integrals.unbounded_k_max`.
_K_MAX_PAD = 200.0


@njit(cache=True)
def _log1p_exp(z):
    """log(1 + e^z), the overflow-safe branch split. = logaddexp(0, z)."""
    if z > 0.0:
        return z + np.log1p(np.exp(-z))
    return np.log1p(np.exp(z))


@njit(cache=True)
def _occupation(x, T):
    """f = 1/(1 + e^(x/T)), through tanh so it cannot overflow."""
    if T <= 0.0:
        if x < 0.0:
            return 1.0
        if x > 0.0:
            return 0.0
        return 0.5
    return 0.5 * (1.0 - np.tanh(0.5 * x / T))


@njit(cache=True)
def _log_term(x, T):
    """T ln(1 + e^(-x/T)), and its T -> 0 limit max(-x, 0)."""
    if T <= 0.0:
        return -x if x < 0.0 else 0.0
    return T * _log1p_exp(-x / T)


@njit(cache=True)
def _k_max(mu, m, T):
    """The momentum ceiling of an UNREGULARISED integrand [MeV].

        k_max = max(|mu|, m) + 45 T + 12 m + 200

    `eos.general.fermi_integrals.unbounded_k_max`, written out. Per MODE, not
    per model: it depends on that mode's own potential and effective mass, and
    on the confined branch the 12 m term is what covers an antiparticle tail
    that decays on the scale of m rather than of T.
    """
    a = abs(mu)
    b = abs(m)
    top = a if a > b else b
    return top + 45.0 * T + 12.0 * b + _K_MAX_PAD


@njit(cache=True)
def _edges(k_F, T, k_max, out):
    """The panel breakpoints on [0, k_max], sorted and deduplicated.

    Breakpoints at the Fermi momentum and, at T > 0, at +- 25 T around it: the
    integrand kinks at k_F and one panel cannot resolve a kink however many
    nodes it is given. `out` is scratch of length >= 5; returns how many edges
    it holds.
    """
    n = 0
    out[n] = 0.0
    n += 1
    out[n] = k_max
    n += 1
    if k_F > 0.0:
        out[n] = k_F
        n += 1
        if T > 0.0:
            out[n] = k_F - _COLLAR * T
            n += 1
            out[n] = k_F + _COLLAR * T
            n += 1
    # clip into [0, k_max]
    for i in range(n):
        if out[i] < 0.0:
            out[i] = 0.0
        elif out[i] > k_max:
            out[i] = k_max
    # insertion sort: n <= 5
    for i in range(1, n):
        key = out[i]
        j = i - 1
        while j >= 0 and out[j] > key:
            out[j + 1] = out[j]
            j -= 1
        out[j + 1] = key
    # drop edges closer together than the tolerance
    kept = 1
    for i in range(1, n):
        if out[i] - out[kept - 1] > _BREAK_TOL * k_max:
            out[kept] = out[i]
            kept += 1
    return kept


@njit(cache=True)
def mode_thermo(mu, m, T, g, x, wx):
    """One mode as an unregularised ideal gas [MeV]: (n, rho_s, eps, P, s).

        n     = (g/2 pi^2) int dk k^2       (f+ - f-)
        rho_s = (g/2 pi^2) int dk k^2 (m/E) (f+ + f-)
        eps   = (g/2 pi^2) int dk k^2  E    (f+ + f-)
        P     = (g/2 pi^2) int dk k^2 T[ln(1 + e^-(E-mu)/T) + ln(1 + e^-(E+mu)/T)]

    with E = sqrt(k^2 + m^2). P is the LOGARITHM form, the one every assembly
    uses. The upper limit is `_k_max` of this mode's own (mu, m, T), not a
    parameter of the model.

    AT T = 0 WITH |mu| <= |m| EVERY ONE OF THEM IS EXACTLY ZERO and this
    returns zero without integrating. That is not an optimisation: it is the
    confinement mechanism -- as phi_bar -> 1 the dielectric closes, M* diverges
    and the quarks leave the medium -- and smoothing it would turn a first-order
    deconfinement into a crossover.
    """
    n = 0.0
    rho_s = 0.0
    eps = 0.0
    P = 0.0
    s = 0.0
    if T <= 0.0 and abs(mu) <= abs(m):
        return n, rho_s, eps, P, s

    k_max = _k_max(mu, m, T)
    k_F = 0.0
    if abs(mu) > abs(m):
        k_F = np.sqrt(mu * mu - m * m)

    edges = np.empty(5)
    n_edges = _edges(k_F, T, k_max, edges)
    pref = g / (2.0 * _PI2)
    nodes = x.shape[0]

    for p in range(n_edges - 1):
        lo = edges[p]
        hi = edges[p + 1]
        half = 0.5 * (hi - lo)
        mid = 0.5 * (lo + hi)
        for q in range(nodes):
            k = mid + half * x[q]
            w = half * wx[q]
            E = np.sqrt(k * k + m * m)
            f_p = _occupation(E - mu, T)
            f_m = _occupation(E + mu, T)
            weight = w * k * k
            n += weight * (f_p - f_m)
            rho_s += weight * (m / E) * (f_p + f_m)
            eps += weight * E * (f_p + f_m)
            P += weight * (_log_term(E - mu, T) + _log_term(E + mu, T))
            if T > 0.0:
                # sum over particles and antiparticles of
                #   (x/T) f + ln(1 + e^(-x/T))
                zp = (E - mu) / T
                zm = (E + mu) / T
                s += weight * (zp * f_p + _log1p_exp(-zp)
                               + zm * f_m + _log1p_exp(-zm))
    return pref * n, pref * rho_s, pref * eps, pref * P, pref * s


@njit(cache=True)
def modes_thermo(mu, m, T, g, x, wx, absent):
    """Every mode in one pass: (N, 5) of (n, rho_s, eps, P, s).

    Two ways a mode carries no medium, and they are different statements:

      * `absent[j]` -- a DECLARATION removed the flavour from the matter
        (`SpeciesFlags.two_flavour` puts it in the three strange modes);
      * `m - |mu| > 60 T` at T > 0 -- the mode is too heavy to be occupied,
        which is `eos.ccdm.thermodynamics.ABSENT_WIDTHS`. At 60 thermal widths
        the occupation is e^-60 ~ 1e-26, exact well inside double precision.
        At T = 0 the test is exact and lives in `mode_thermo` itself.
    """
    N = mu.shape[0]
    out = np.zeros((N, 5))
    for j in range(N):
        if absent[j]:
            continue
        if T > 0.0 and m[j] - abs(mu[j]) > _ABSENT_WIDTHS * T:
            continue
        a, b, c, d, e = mode_thermo(mu[j], m[j], T, g, x, wx)
        out[j, 0] = a
        out[j, 1] = b
        out[j, 2] = c
        out[j, 3] = d
        out[j, 4] = e
    return out
