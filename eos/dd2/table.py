"""
table.py
====================
Table driver: a TableSpec describing the parametrization, the
equilibrium mode, the axes (n_B and either T or entropy-per-baryon S=s/n_B),
the fixed fractions, and the active species — plus build_table() that solves
the grid with warm-started continuation along n_B.

The temperature axis may be given directly as 'T' or as entropy per baryon
'SnB'; the latter adds an outer 1-D solve for T at each point,
a thin wrapper around the same octet solve.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from eos.dd2.species import SpeciesFlags
from eos.dd2.solver import solve_octet, sweep_octet


#: Equilibrium mode -> the `solve_octet` configuration it means. The fixed
#: fractions are filled in from `TableSpec.fixed` at build time.
#:
#: The names say what the matter is, and they match the modes `eos.mixed`
#: offers, so a purely hadronic table and a hybrid table are requested the
#: same way:
#:
#:   beta_eq_neutrinoless      charge-neutral beta equilibrium, neutrinos escape
#:   beta_eq_neutrino_trapped  ... with neutrinos trapped at fixed Y_L
#:   fixed_YC                  fixed non-leptonic charge fraction, no leptons
#:                             (the CompOSE general-purpose (nB, T, Y_q) slice)
#:   fixed_YC_neutral          ... plus neutralizing leptons, so the total
#:                             system is electrically neutral
#:   fixed_YS                  charge-neutral, strangeness fraction fixed
#:   fixed_YC_YS               both fractions fixed
_MODES = {
    "beta_eq_neutrinoless": dict(charge_mode="neutral"),
    "fixed_YC": dict(charge_mode="fixed"),
    "fixed_YC_neutral": dict(charge_mode="fixed", yc_leptons=True),
    "fixed_YS": dict(charge_mode="neutral", strange_mode="fixed"),
    "fixed_YC_YS": dict(charge_mode="fixed", strange_mode="fixed"),
    "beta_eq_neutrino_trapped": dict(charge_mode="neutral",
                                     lepton_mode="trapped"),
}

#: Which `TableSpec.fixed` keys each mode consumes.
_MODE_FIXED = {
    "beta_eq_neutrinoless": (), "fixed_YC": ("Y_C",),
    "fixed_YC_neutral": ("Y_C",), "fixed_YS": ("Y_S",),
    "fixed_YC_YS": ("Y_C", "Y_S"), "beta_eq_neutrino_trapped": ("Y_L",),
}


def _mode_kwargs(mode, fixed):
    if mode not in _MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {list(_MODES)}")
    kw = dict(_MODES[mode])
    for key in _MODE_FIXED[mode]:
        if key not in fixed:
            raise ValueError(f"mode {mode!r} needs fixed[{key!r}]")
        kw[key] = fixed[key]
    return kw


def solve_octet_at_entropy(par, n_B, S_per_B, flags, x0=None, T_lo=0.2,
                           T_hi=80.0, T_cap=400.0, xtol=1e-5, **mode_kwargs):
    """
    Solve at fixed entropy per baryon S=s/n_B by an outer 1-D root on T
. s(T) is monotone increasing at fixed n_B, so the bracket is
    well-posed; T_hi is doubled up to T_cap if the target is not yet bracketed.
    Returns the converged EoSPoint (at the solved T).
    """
    def point(T):
        return solve_octet(par, n_B, flags, T=T, x0=x0, include_photons=True,
                           **mode_kwargs)

    def f(T):
        return point(T).s / n_B - S_per_B

    f_hi = f(T_hi)
    while f_hi < 0.0 and T_hi < T_cap:
        T_hi = min(2.0 * T_hi, T_cap)
        f_hi = f(T_hi)
    if f_hi < 0.0:
        raise RuntimeError(
            f"entropy target S={S_per_B} unreachable at n_B={n_B} below "
            f"T={T_cap} MeV (s/n_B={f_hi + S_per_B:.3f} there)")
    T = brentq(f, T_lo, T_hi, xtol=xtol)
    return point(T)


@dataclass
class TableSpec:
    """One table request."""
    parametrization: object
    mode: str                             # a key of _MODES, above
    axes: dict                            # {'nB': grid, 'T'|'SnB': grid}
    include: SpeciesFlags = field(default_factory=SpeciesFlags)
    fixed: dict = field(default_factory=dict)   # Y_C / Y_S / Y_L targets
    want_coeffs: bool = False             # attach equilibrium c_s^2 per T-line

    def __post_init__(self):
        if "nB" not in self.axes:
            raise ValueError("TableSpec.axes must contain 'nB'")
        temp_axes = [k for k in self.axes if k in ("T", "SnB")]
        if len(temp_axes) != 1:
            raise ValueError("TableSpec.axes needs exactly one of 'T' / 'SnB'")
        self._temp_key = temp_axes[0]
        _mode_kwargs(self.mode, self.fixed)   # validate early


@dataclass
class TableResult:
    spec: TableSpec
    nB: np.ndarray
    temp_values: np.ndarray               # the T or S grid
    temp_key: str                         # 'T' or 'SnB'
    points: list                          # points[i_temp][i_nB] EoSPoint
    cs2_eq: list = None                   # cs2[i_temp][i_nB] if want_coeffs


def _cs2_along(points):
    """Equilibrium c_s^2 = dP/deps along one n_B line."""
    P = np.array([p.P for p in points])
    eps = np.array([p.eps for p in points])
    return np.gradient(P, eps)


def build_table(spec, skip_errors=False):
    """
    Solve the TableSpec grid. For each temperature-axis value an n_B sweep is
    warm-started along density (the stiff axis); the 'SnB' axis
    replaces each solve with the outer entropy T-solve. Returns a TableResult;
    with want_coeffs, equilibrium c_s^2 is attached per T-line.

    skip_errors: if True, points where the octet solve doesn't converge are
    dropped from their T-line instead of aborting the whole table (the warm
    start resets on a skip). This is expected for constrained modes at low T /
    low density inside the liquid-gas spinodal, where uniform matter has no
    stable solution; the returned lines are then shorter than ``nB``.
    """
    from eos.dd2.solver import octet_warm_start
    nB = np.asarray(spec.axes["nB"], dtype=float)
    temp = np.asarray(spec.axes[spec._temp_key], dtype=float)
    mode_kw = _mode_kwargs(spec.mode, spec.fixed)
    flags = spec.include
    has_phi = flags.phi_field and flags.hyperons
    has_muS = mode_kw.get("strange_mode") == "fixed"
    has_muL = mode_kw.get("lepton_mode") == "trapped"

    def solve_at(n, tv, x0):
        if spec._temp_key == "T":
            return solve_octet(spec.parametrization, float(n), flags,
                               T=float(tv), x0=x0, **mode_kw)
        return solve_octet_at_entropy(spec.parametrization, float(n),
                                      float(tv), flags, x0=x0, **mode_kw)

    points = []
    for tv in temp:
        # Fast path: the whole line in one warm-started sweep (T axis only).
        if spec._temp_key == "T" and not skip_errors:
            points.append(sweep_octet(spec.parametrization, nB, flags,
                                      T=float(tv), **mode_kw))
            continue
        # Tolerant / entropy path: per-point, warm-started, optionally skipping.
        line, x0 = [], None
        for n in nB:
            try:
                p = solve_at(n, tv, x0)
            except RuntimeError:
                if not skip_errors:
                    raise
                x0 = None          # reset the warm start past the gap
                continue
            line.append(p)
            x0 = octet_warm_start(p, has_phi, has_muS, has_muL)
        points.append(line)

    cs2 = [_cs2_along(line) for line in points] if spec.want_coeffs else None
    return TableResult(spec=spec, nB=nB, temp_values=temp,
                       temp_key=spec._temp_key, points=points, cs2_eq=cs2)


if __name__ == "__main__":
    from eos.dd2 import Parametrization
    spec = TableSpec(
        parametrization=Parametrization.from_dd2y_defaults(),
        mode="beta_eq_neutrinoless",
        axes={"nB": np.linspace(0.1, 0.8, 8), "SnB": [1.0, 2.0]},
        include=SpeciesFlags(hyperons=True, phi_field=True),
        want_coeffs=True,
    )
    res = build_table(spec)
    for j, S in enumerate(res.temp_values):
        Ts = [p.T for p in res.points[j]]
        print(f"S={S}: T range {min(Ts):.1f}-{max(Ts):.1f} MeV, "
              f"max cs2={max(res.cs2_eq[j]):.3f}")
