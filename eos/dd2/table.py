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
from eos.general.state import EOSTable_for_TOV
from eos.dd2.solver import (
    solve, sweep, MODES, MODE_FRACTIONS, _mode_kwargs,
)


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
    lep = p.leptons
    _, n_C, n_S = hadronic_charges(flags, p.matter.densities)
    row = dict(n_B=n_B, T=p.T, chi=0.0, phase="H", P=p.P, eps=p.eps, s=p.s,
               S_per_B=(p.s / n_B if n_B else 0.0), mu_B=p.matter.mu_B,
               Y_C=n_C / n_B, Y_S=n_S / n_B,
               mu_e=lep.mu_e, mu_S=p.matter.mu_S, mu_nue=lep.mu_nue,
               Y_e=lep.densities["e-"] / n_B,
               **{"Y_mu-": lep.densities["mu-"] / n_B})
    for name, n in p.matter.densities.items():
        row[f"Y_{name}"] = n / n_B
    if lep.densities["nu_e"]:
        row["Y_nue"] = lep.densities["nu_e"] / n_B
    return row


def solve_at_entropy(par, n_B, S_per_B, flags, x0=None, T_lo=0.2,
                           T_hi=80.0, T_cap=400.0, xtol=1e-5, **mode_kwargs):
    """
    Solve at fixed entropy per baryon S=s/n_B by an outer 1-D root on T
. s(T) is monotone increasing at fixed n_B, so the bracket is
    well-posed; T_hi is doubled up to T_cap if the target is not yet bracketed.
    Returns the converged EoSPoint (at the solved T).
    """
    def point(T):
        return solve(par, n_B, flags, T=T, x0=x0,
                           include_photons=flags.photons, **mode_kwargs)

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
           'Y_C'/'Y_S'/'Y_Le': grid to sweep that fraction as a further axis}
    fixed: scalar values for the fractions the mode needs that are not swept
           as axes
    leptons: for fixed_YC, whether the neutralizing leptons are present. The
           orthogonal flag of CLAUDE.md section 3, not part of the mode name.
           None leaves it at DD2's leptonless default; on a beta-equilibrium
           mode True is redundant and ignored and False raises.
    """
    parametrization: object
    mode: str                             # a key of MODES, above
    axes: dict
    include: SpeciesFlags = field(default_factory=SpeciesFlags)
    fixed: dict = field(default_factory=dict)   # Y_C / Y_S / Y_Le targets
    leptons: bool = None
    want_coeffs: bool = False             # attach c_s^2 = dP/deps per line

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
        self._frac_keys = [k for k in ("Y_C", "Y_S", "Y_Le") if k in self.axes]
        # Validate early that every fraction the mode needs is supplied, by an
        # axis or a scalar; an axis value stands in here only for the check.
        probe = dict(self.fixed)
        probe.update({k: 0.0 for k in self._frac_keys})
        _mode_kwargs(self.mode, probe, self.leptons)


@dataclass
class TableResult:
    spec: TableSpec
    nB: np.ndarray
    temp_values: np.ndarray               # the T or S grid
    temp_key: str                         # 'T' or 'SnB'
    points: list                          # points[i_combo][i_nB] EoSPoint
    #: c_s^2[i_combo][i_nB] = dP/deps along each line, if want_coeffs. The
    #: composition re-equilibrates along the line, so only the THERMAL axis
    #: needs naming, and the axis the table was built on fixes it: EXACTLY
    #: ONE of the two is populated, the other stays None (CLAUDE.md section 5
    #: — never a bare `cs2` whose meaning depends on the arguments).
    cs2_isothermal: list = None           # populated on a 'T' axis
    cs2_adiabatic: list = None            # populated on an 'SnB' axis
    #: [(temperature value, {fraction: value}), ...], parallel to `points`.
    #: One entry per line; with no fraction axes it is one entry per
    #: temperature and the dict is empty, which is the historical layout.
    combos: list = None


def _cs2_along(points):
    """c_s^2 = dP/deps along one n_B line, composition re-equilibrating. The
    line's thermal condition is whichever axis the spec was built on, so the
    caller stores this under `cs2_isothermal` ('T') or `cs2_adiabatic`
    ('SnB')."""
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
    order `TableResult.combos` records. With want_coeffs, c_s^2 = dP/deps is
    attached per line, under `cs2_isothermal` on a 'T' axis and
    `cs2_adiabatic` on an 'SnB' axis — exactly one of the two, the other None.

    rows=True instead returns `(rows, {})` in the long format
    `eos.mixed.build_table` returns — one flat dict per converged point,
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

    from eos.dd2.solver import warm_start
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
            mode_kw = _mode_kwargs(spec.mode, fracs, spec.leptons)
            has_phi = flags.phi_field and flags.hyperons
            has_muS = mode_kw.get("strange_mode") == "fixed"
            has_muL = mode_kw.get("lepton_mode") == "trapped"

            def solve_at(n, x0):
                if spec._temp_key == "T":
                    return solve(spec.parametrization, float(n), flags,
                                       T=float(tv), x0=x0,
                                       include_photons=flags.photons, **mode_kw)
                return solve_at_entropy(spec.parametrization, float(n),
                                              float(tv), flags, x0=x0,
                                              **mode_kw)

            combos.append((float(tv), dict(zip(frac_keys, map(float, combo)))))
            t_line = time.time()
            # Fast path: the whole line in one warm-started sweep (T axis only).
            if spec._temp_key == "T" and not skip_errors:
                line = sweep(spec.parametrization, nB, flags,
                                   T=float(tv), include_photons=flags.photons,
                                   **mode_kw)
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
                    x0 = warm_start(p, has_phi, has_muS, has_muL)
            points.append(line)
            # `fracs` is the FULL set the line was solved at, swept and
            # fixed alike (CLAUDE.md section 5); `combos` records only the
            # swept keys, because that is the grid axis order it indexes.
            if progress is not None:
                progress(dict(mode=spec.mode, line=len(points),
                              n_lines=n_lines, temp_key=spec._temp_key,
                              temp=float(tv), fracs=dict(fracs),
                              n_solved=len(line), n_requested=len(nB),
                              elapsed_s=time.time() - t_line))

    cs2 = [_cs2_along(line) for line in points] if spec.want_coeffs else None
    result = TableResult(spec=spec, nB=nB, temp_values=temp,
                         temp_key=spec._temp_key, points=points,
                         combos=combos)
    # The thermal axis of the derivative IS the table's temperature axis: a
    # 'T' line holds the temperature, an 'SnB' line the entropy per baryon.
    if spec._temp_key == "T":
        result.cs2_isothermal = cs2
    else:
        result.cs2_adiabatic = cs2
    return (rows_from_result(result), {}) if rows else result


def rows_from_result(result):
    """A solved `TableResult` to the long-format rows `eos.general.table_io`
    writes — the same shape `eos.mixed.build_table` returns.

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


# =============================================================================
# THE CORE TABLE A STRUCTURE SOLVER INTEGRATES
# =============================================================================
# `EOSTable_for_TOV` is the contract between a model and `eos.astro` and lives
# in `eos.general.state`, which both layers may import (CLAUDE.md section 1);
# building one is the model's side of it, so it belongs here rather than in a
# verify suite. Running the sequence over it does not: that is astro's, and
# the callers that want M(R) reach for `eos.astro.tov` themselves.

#: Crust–core transition density [fm^-3] (BPS table tops out at 0.08).
N_TRANSITION = 0.08


def build_core_table(par, flags, n_lo=0.05, n_hi=1.25, n_points=150):
    """
    Cold (T=0) beta-equilibrium core EoS as an EOSTable_for_TOV. Uses a
    geometric density grid and warm-started continuation through the onsets.
    """
    grid = np.geomspace(n_lo, n_hi, n_points)
    # stop_at_boundary: a Δ model may hit scalar collapse (m*->0) before n_hi;
    # take the valid prefix as the core EoS.
    points = sweep(par, grid, flags, T=0.0,
                                 include_photons=flags.photons,
                                 stop_at_boundary=True)
    P = np.array([p.P for p in points])
    eps = np.array([p.eps for p in points])
    nB = np.array([p.n_B for p in points])
    # TOV interpolation needs a monotone-increasing P grid.
    order = np.argsort(P)
    return EOSTable_for_TOV(P=P[order], epsilon=eps[order], nB=nB[order])


if __name__ == "__main__":
    from eos.dd2.parameters import Parameters
    spec = TableSpec(
        parametrization=Parameters.named("DD2Y"),
        mode="beta_eq_neutrinoless",
        axes={"nB": np.linspace(0.1, 0.8, 8), "SnB": [1.0, 2.0]},
        include=SpeciesFlags(hyperons=True, phi_field=True),
        want_coeffs=True,
    )
    res = build_table(spec)
    for j, S in enumerate(res.temp_values):
        Ts = [p.T for p in res.points[j]]
        # an SnB axis, so the populated field is the adiabatic one
        print(f"S={S}: T range {min(Ts):.1f}-{max(Ts):.1f} MeV, "
              f"max cs2_adiabatic={max(res.cs2_adiabatic[j]):.3f}")
