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
from itertools import product

import numpy as np
from scipy.optimize import brentq

from eos.dd2.species import SpeciesFlags, hadronic_charges
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
MODES = {
    "beta_eq_neutrinoless": dict(charge_mode="neutral"),
    "fixed_YC": dict(charge_mode="fixed"),
    "fixed_YC_neutral": dict(charge_mode="fixed", yc_leptons=True),
    "fixed_YS": dict(charge_mode="neutral", strange_mode="fixed"),
    "fixed_YC_YS": dict(charge_mode="fixed", strange_mode="fixed"),
    "beta_eq_neutrino_trapped": dict(charge_mode="neutral",
                                     lepton_mode="trapped"),
}

#: mode name -> the fixed fractions it consumes, either as a `TableSpec.axes`
#: grid or as a scalar in `TableSpec.fixed`. Mirrors
#: `eos.mixed.MODE_FRACTIONS`, which names the four modes both engines share.
MODE_FRACTIONS = {
    "beta_eq_neutrinoless": (), "fixed_YC": ("Y_C",),
    "fixed_YC_neutral": ("Y_C",), "fixed_YS": ("Y_S",),
    "fixed_YC_YS": ("Y_C", "Y_S"), "beta_eq_neutrino_trapped": ("Y_L",),
}


def _mode_kwargs(mode, fixed):
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {list(MODES)}")
    kw = dict(MODES[mode])
    for key in MODE_FRACTIONS[mode]:
        if key not in fixed:
            raise ValueError(f"mode {mode!r} needs fixed[{key!r}]")
        kw[key] = fixed[key]
    return kw


def hadronic_row(p, flags):
    """Flatten one `EoSPoint` into a dict row.

    Keyed exactly the way `eos.mixed.composition_row` keys a mixed point, so a
    pure-hadronic table and a hybrid table concatenate without renaming
    anything. chi = 0 and phase = 'H': no quark matter is present.

    Y_C is the NON-leptonic charge fraction and Y_S counts +1 per s-quark, both
    summed over the active baryons rather than read off the proton alone — with
    hyperons switched on those differ.
    """
    n_B = p.n_B
    _, n_C, n_S = hadronic_charges(flags, p.composition_map)
    row = dict(n_B=n_B, T=p.T, chi=0.0, phase="H", P=p.P, eps=p.eps, s=p.s,
               S_per_B=(p.s / n_B if n_B else 0.0), mu_B=p.mu_n,
               Y_C=n_C / n_B, Y_S=n_S / n_B,
               mu_e=p.mu_e, mu_S=p.mu_S, mu_L=p.mu_L,
               Y_e=p.n_e / n_B, **{"Y_mu-": p.n_mu / n_B})
    for name, n in p.composition_map.items():
        row[f"Y_{name}"] = n / n_B
    if p.n_nu:
        row["Y_nue"] = p.n_nu / n_B
    return row


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
    """One table request.

    axes : {'nB': grid, exactly one of 'T'/'SnB': grid, and optionally any of
           'Y_C'/'Y_S'/'Y_L': grid to sweep that fraction as a further axis}
    fixed: scalar values for the fractions the mode needs that are not swept
           as axes
    """
    parametrization: object
    mode: str                             # a key of MODES, above
    axes: dict
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
        if self.mode not in MODES:
            raise ValueError(f"unknown mode {self.mode!r}; expected one of "
                             f"{list(MODES)}")
        self._frac_keys = [k for k in ("Y_C", "Y_S", "Y_L") if k in self.axes]
        # Validate early that every fraction the mode needs is supplied, by an
        # axis or a scalar; an axis value stands in here only for the check.
        probe = dict(self.fixed)
        probe.update({k: 0.0 for k in self._frac_keys})
        _mode_kwargs(self.mode, probe)


@dataclass
class TableResult:
    spec: TableSpec
    nB: np.ndarray
    temp_values: np.ndarray               # the T or S grid
    temp_key: str                         # 'T' or 'SnB'
    points: list                          # points[i_combo][i_nB] EoSPoint
    cs2_eq: list = None                   # cs2[i_combo][i_nB] if want_coeffs
    #: [(temperature value, {fraction: value}), ...], parallel to `points`.
    #: One entry per line; with no fraction axes it is one entry per
    #: temperature and the dict is empty, which is the historical layout.
    combos: list = None


def _cs2_along(points):
    """Equilibrium c_s^2 = dP/deps along one n_B line."""
    P = np.array([p.P for p in points])
    eps = np.array([p.eps for p in points])
    return np.gradient(P, eps)


def _print_progress(info):
    """The built-in progress printer (verbose=True)."""
    fracs = "".join(f" {k}={v:g}" for k, v in info["fracs"].items())
    print(f"[{info['line']}/{info['n_lines']}] {info['mode']} "
          f"{info['temp_key']}={info['temp']:g}{fracs}: "
          f"{info['n_solved']}/{info['n_requested']} points "
          f"in {info['elapsed_s']:.1f}s")


def build_table(spec, skip_errors=False, rows=False, progress=None,
                verbose=False):
    """
    Solve the TableSpec grid over the product of its temperature and fraction
    axes. Within each combination an n_B sweep is warm-started along density
    (the stiff axis); the 'SnB' axis replaces each solve with the outer entropy
    T-solve.

    rows=False (default) returns a `TableResult`, whose `points` are indexed
    [i_combination][i_nB] — one line per (temperature, fractions) pair, in the
    order `TableResult.combos` records. With want_coeffs, equilibrium c_s^2 is
    attached per line.

    rows=True instead returns `(rows, {})` in the long format
    `eos.mixed.build_mixed_table` returns — one flat dict per converged point,
    ready for `eos.general.table_io`. The empty second element is where that
    function returns its phase windows, which purely hadronic matter has none
    of; it is returned anyway so the two calls unpack the same way.

    skip_errors: if True, points where the octet solve doesn't converge are
    dropped from their line instead of aborting the whole table (the warm
    start resets on a skip). This is expected for constrained modes at low T /
    low density inside the liquid-gas spinodal, where uniform matter has no
    stable solution; the returned lines are then shorter than ``nB``.

    progress: optional callable, invoked once per completed line with a dict
    {mode, line, n_lines, temp_key, temp, fracs, n_solved, n_requested,
    elapsed_s} — the same shape every table builder in this repository uses.
    Default is silent; verbose=True installs the built-in one-line printer.
    Deep solver code never prints.
    """
    import time

    from eos.dd2.solver import octet_warm_start
    if verbose and progress is None:
        progress = _print_progress
    nB = np.asarray(spec.axes["nB"], dtype=float)
    temp = np.asarray(spec.axes[spec._temp_key], dtype=float)
    flags = spec.include
    frac_keys = spec._frac_keys
    frac_grids = [np.atleast_1d(np.asarray(spec.axes[k], float))
                  for k in frac_keys]
    n_lines = len(temp) * max(1, int(np.prod([len(g) for g in frac_grids])))

    points, combos = [], []
    for tv in temp:
        for combo in product(*frac_grids) if frac_grids else [()]:
            fracs = dict(spec.fixed)
            fracs.update(zip(frac_keys, (float(c) for c in combo)))
            mode_kw = _mode_kwargs(spec.mode, fracs)
            has_phi = flags.phi_field and flags.hyperons
            has_muS = mode_kw.get("strange_mode") == "fixed"
            has_muL = mode_kw.get("lepton_mode") == "trapped"

            def solve_at(n, x0):
                if spec._temp_key == "T":
                    return solve_octet(spec.parametrization, float(n), flags,
                                       T=float(tv), x0=x0, **mode_kw)
                return solve_octet_at_entropy(spec.parametrization, float(n),
                                              float(tv), flags, x0=x0,
                                              **mode_kw)

            combos.append((float(tv), dict(zip(frac_keys, map(float, combo)))))
            t_line = time.time()
            # Fast path: the whole line in one warm-started sweep (T axis only).
            if spec._temp_key == "T" and not skip_errors:
                line = sweep_octet(spec.parametrization, nB, flags,
                                   T=float(tv), **mode_kw)
            else:
                # Tolerant / entropy path: per-point, warm-started, may skip.
                line, x0 = [], None
                for n in nB:
                    try:
                        p = solve_at(n, x0)
                    except RuntimeError:
                        if not skip_errors:
                            raise
                        x0 = None      # reset the warm start past the gap
                        continue
                    line.append(p)
                    x0 = octet_warm_start(p, has_phi, has_muS, has_muL)
            points.append(line)
            if progress is not None:
                progress(dict(mode=spec.mode, line=len(points),
                              n_lines=n_lines, temp_key=spec._temp_key,
                              temp=float(tv), fracs=combos[-1][1],
                              n_solved=len(line), n_requested=len(nB),
                              elapsed_s=time.time() - t_line))

    cs2 = [_cs2_along(line) for line in points] if spec.want_coeffs else None
    result = TableResult(spec=spec, nB=nB, temp_values=temp,
                         temp_key=spec._temp_key, points=points, cs2_eq=cs2,
                         combos=combos)
    return (rows_from_result(result), {}) if rows else result


def rows_from_result(result):
    """A solved `TableResult` to the long-format rows `eos.general.table_io`
    writes — the same shape `eos.mixed.build_mixed_table` returns.

    Separate from `build_table` so a table already solved for its `TableResult`
    can be written out without being solved a second time.
    """
    flags = result.spec.include
    out = []
    for (tv, fracs), line in zip(result.combos, result.points):
        for p in line:
            row = hadronic_row(p, flags)
            row.update(fracs)
            if result.temp_key == "SnB":
                row["SnB"] = tv
            out.append(row)
    return out


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
