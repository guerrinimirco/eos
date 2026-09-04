"""
backends/kernel_numba.py
========================
The nine colour-flavour modes as cut Fermi gases, jitted — the hot path of
every NJL solve.

This is the SAME quadrature as `eos.general.fermi_integrals.kinetic_thermo`
called once per mode, written a second time so it can be compiled: the
panel-split Gauss-Legendre rule of `panel_nodes`, then the six integrals of
`kinetic_thermo`, for all nine modes in one nopython pass. Nothing here is a
different approximation, and `eos/njl/verify/run_full_check.py` checks the two
against each other point by point.

Why this and not the JEL route dd2's kernel had to avoid: NJL's medium
integrals are Gauss-Legendre sums over a sharp cutoff, which are arithmetic
and trace fine, so unlike `eos.dd2.backends.kernel_numba` this kernel is NOT
restricted to T = 0. The reference reaches the same numbers at every
temperature; it reaches them through ~10 NumPy calls on ~120-element arrays
per mode, and at that size the dispatch costs more than the arithmetic.

`backends/` is deletable (CLAUDE.md section 5): with this file gone, or with
numba absent, `eos.njl.thermodynamics` runs its own loop over
`kinetic_thermo` and returns the same numbers more slowly.

The reference Gauss-Legendre rule (x, w on [-1, 1]) is passed IN rather than
built here -- `eos.general.fermi_integrals` memoizes it, and it depends only
on the node count.
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
def _edges(k_F, T, k_max, out):
    """The panel breakpoints on [0, k_max], sorted and deduplicated.

    Breakpoints at each Fermi momentum and, at T > 0, at +- 25 T around it:
    the integrand kinks at k_F and one panel cannot resolve a kink however
    many nodes it is given. `out` is scratch of length >= 5; returns how many
    edges it holds.
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
def mode_thermo(mu, m, T, k_max, g, x, wx):
    """One mode as an ideal gas cut at `k_max` [MeV]: (n, rho_s, eps, P, s).

        n     = (g/2 pi^2) int dk k^2       (f+ - f-)
        rho_s = (g/2 pi^2) int dk k^2 (m/E) (f+ + f-)
        eps   = (g/2 pi^2) int dk k^2  E    (f+ + f-)
        P     = (g/2 pi^2) int dk k^2 T[ln(1 + e^-(E-mu)/T) + ln(1 + e^-(E+mu)/T)]

    with E = sqrt(k^2 + m^2). P is the LOGARITHM form, the one every assembly
    uses. At T = 0 with |mu| <= |m| every one of them is exactly zero and this
    returns zero without integrating -- the statement that a mode too heavy
    for its own potential is not in the medium.
    """
    n = 0.0
    rho_s = 0.0
    eps = 0.0
    P = 0.0
    s = 0.0
    if T <= 0.0 and abs(mu) <= abs(m):
        return n, rho_s, eps, P, s

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
def modes_thermo(mu, m, T, k_max, g, x, wx, absent):
    """Every mode in one pass: (N, 5) of (n, rho_s, eps, P, s).

    `absent` is a per-mode flag: a mode a declaration removes from the matter
    (`SpeciesFlags.two_flavour` puts it in the three strange modes) carries no
    medium at all, which is the same block as a mode below threshold but
    reached by statement rather than by a threshold.
    """
    N = mu.shape[0]
    out = np.zeros((N, 5))
    for j in range(N):
        if absent[j]:
            continue
        a, b, c, d, e = mode_thermo(mu[j], m[j], T, k_max, g, x, wx)
        out[j, 0] = a
        out[j, 1] = b
        out[j, 2] = c
        out[j, 3] = d
        out[j, 4] = e
    return out


@njit(cache=True)
def mode_jacobian(mu, m, T, k_max, g, x, wx):
    """One mode's (dn/dmu, dn/dm, drho_s/dm) [MeV^2], for the Jacobian.

    The same panels and nodes as `mode_thermo`, differentiated under the
    integral. With f = f(E -+ mu) and f' = -f(1 - f)/T,

        dn/dmu     = pref int dk k^2 [f+(1 - f+) + f-(1 - f-)]/T
        dn/dm      = pref int dk k^2 (m/E) (f+' - f-')
        drho_s/dm  = pref int dk k^2 [ (k^2/E^3)(f+ + f-) + (m/E)^2 (f+' + f-') ]

    and drho_s/dmu = -dn/dm, the symmetry of the second derivatives of
    Omega, so it is not returned twice. At T = 0 the f' terms are the step's
    derivative and collapse to the Fermi surface: dn/dmu = pref k_F |mu|,
    dn/dm = -pref k_F m sign(mu), drho_s/dm gets -pref k_F m^2/|mu| beside
    its bulk term. Zero, like the block, for a mode below threshold.
    """
    dn_dmu = 0.0
    dn_dm = 0.0
    drs_dm = 0.0
    if T <= 0.0 and abs(mu) <= abs(m):
        return dn_dmu, dn_dm, drs_dm

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
            drs_dm += weight * (k * k / (E * E * E)) * (f_p + f_m)
            if T > 0.0:
                fp_p = -f_p * (1.0 - f_p) / T
                fp_m = -f_m * (1.0 - f_m) / T
                dn_dmu += weight * (-fp_p - fp_m)
                dn_dm += weight * (m / E) * (fp_p - fp_m)
                drs_dm += weight * (m / E) * (m / E) * (fp_p + fp_m)
    if T <= 0.0:
        sign = 1.0 if mu > 0.0 else -1.0
        dn_dmu += k_F * abs(mu)
        dn_dm += -k_F * m * sign
        drs_dm += -k_F * m * m / abs(mu)
    return pref * dn_dmu, pref * dn_dm, pref * drs_dm


@njit(cache=True)
def modes_jacobian(mu, m, T, k_max, g, x, wx, absent):
    """Every mode's (dn/dmu, dn/dm, drho_s/dm) in one pass: (N, 3).

    `absent` as in `modes_thermo`: a mode a declaration removed carries no
    medium and no derivative of one.
    """
    N = mu.shape[0]
    out = np.zeros((N, 3))
    for j in range(N):
        if absent[j]:
            continue
        a, b, c = mode_jacobian(mu[j], m[j], T, k_max, g, x, wx)
        out[j, 0] = a
        out[j, 1] = b
        out[j, 2] = c
    return out
