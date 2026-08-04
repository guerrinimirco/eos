"""
mixed/tables/generate.py
========================
Multi-axis EoS table generation for every equilibrium mode.

*Public API* (re-exported from `eos.mixed`): `MixedTableSpec`,
`build_mixed_table`, `solve_mixed_at_entropy`.

One `MixedTableSpec` describes a whole table: the parametrization and species,
the equilibrium mode, and the axes to sweep. The mode decides which fractions
are meaningful axes, so all four modes generate tables through this one path:

    beta_eq_neutrinoless        axes  nB, and T or SnB
    beta_eq_neutrino_trapped    axes  nB, Y_L, and T or SnB
    fixed_YC                    axes  nB, Y_C, and T or SnB
    fixed_YC_YS                 axes  nB, Y_C, Y_S, and T or SnB

The temperature axis may be given directly as 'T', or as entropy per baryon
'SnB', which adds an outer one-dimensional solve for T at each point.

Output is long format — one dict per converged point, carrying the axis values,
the bulk thermodynamics and the volume-weighted composition. Wrap it in pandas
in a notebook; the engine itself stays free of that dependency.

Speed comes from three warm starts, in order of how much they buy: the phase
internal states are cached inside a solve, each density seeds the next along
the stiff n_B axis, and each temperature line seeds the next. On top of that,
the mixed system is solved only inside the located window — outside it the far
cheaper pure-phase solvers answer the same question.
"""
from dataclasses import dataclass, field
from itertools import product

import numpy as np

from eos.mixed.equilibrium.charges import (
    beta_eq_neutrinoless, beta_eq_neutrino_trapped, fixed_YC, fixed_YC_YS,
)
from eos.mixed.equilibrium.residual import mixed_slots
from eos.mixed.solvers.point import solve_mixed
from eos.mixed.solvers.sweep import locate_window, sweep_mixed

#: mode name -> the fixed fractions it consumes (as an axis or in `fixed`).
MODE_FRACTIONS = {
    "beta_eq_neutrinoless": (),
    "beta_eq_neutrino_trapped": ("Y_L",),
    "fixed_YC": ("Y_C",),
    "fixed_YC_YS": ("Y_C", "Y_S"),
}


def make_charge_spec(mode, fracs, leptons=True):
    """`ChargeSpec` for a named mode and its fraction values."""
    if mode == "beta_eq_neutrinoless":
        return beta_eq_neutrinoless()
    if mode == "beta_eq_neutrino_trapped":
        return beta_eq_neutrino_trapped(fracs["Y_L"])
    if mode == "fixed_YC":
        return fixed_YC(fracs["Y_C"], leptons=leptons)
    if mode == "fixed_YC_YS":
        return fixed_YC_YS(fracs["Y_C"], fracs["Y_S"], leptons=leptons)
    raise ValueError(f"unknown mode {mode!r}; expected one of "
                     f"{list(MODE_FRACTIONS)}")


def composition_row(r):
    """Flatten one `MixedResult` into a dict row.

    Carries the bulk thermodynamics and the GLOBAL composition, i.e. each
    species weighted by its phase's volume fraction, Y_i = w n_i / n_B with
    w = 1-chi for hadrons and w = chi for quarks. Y_C and Y_S sum over both
    phases. Any global charge potential the mode carries is included.
    """
    row = dict(n_B=r.n_B, T=r.T, eta=r.eta, chi=r.chi, phase=r.phase,
               P=r.P, eps=r.eps, s=r.s, S_per_B=(r.s / r.n_B if r.n_B else 0.0),
               mu_B=r.mu_B,
               Y_C=((1 - r.chi) * r.th_H.n_C + r.chi * r.th_Q.n_C) / r.n_B,
               Y_S=((1 - r.chi) * r.th_H.n_S + r.chi * r.th_Q.n_S) / r.n_B)
    for key in ("mu_C", "mu_S", "mu_L", "mu_eL_H", "mu_eL_Q", "mu_eG"):
        if key in r.potentials:
            row[key] = r.potentials[key]
    for name, n in r.th_H.densities.items():
        row[f"Y_{name}"] = (1 - r.chi) * n / r.n_B
    for name, n in r.th_Q.densities.items():
        row[f"Y_{name}"] = r.chi * n / r.n_B
    # Leptons, weighted the way the solver populates them: the local species
    # are volume-averaged and carry weight eta, the global one carries 1-eta.
    L_H, L_Q, G = r.extras["L_H"], r.extras["L_Q"], r.extras["G"]
    for label, attr in (("e", "n_e"), ("mu-", "n_mu")):
        n_loc = (1 - r.chi) * getattr(L_H, attr) + r.chi * getattr(L_Q, attr)
        n_tot = r.eta * n_loc + (1 - r.eta) * getattr(G, attr)
        row[f"Y_{label}"] = n_tot / r.n_B
    if r.extras["nu"] is not None:
        row["Y_nue"] = r.extras["nu"].n / r.n_B
    return row


def solve_mixed_at_entropy(par, flags, n_B, SnB, eta, spec, vmit_params=None,
                           x0=None, T_lo=0.5, T_hi=50.0, T_cap=250.0,
                           xtol=1e-4):
    """Mixed solve at fixed entropy per baryon S = s/n_B.

    An outer one-dimensional root on T, mirroring
    `eos/dd2/table.solve_octet_at_entropy`. s(T) increases monotonically at
    fixed n_B so the bracket is well posed; T_hi doubles up to T_cap until the
    target is bracketed.
    """
    from scipy.optimize import brentq

    def point(T):
        return solve_mixed(par, flags, n_B, eta, spec, vmit_params=vmit_params,
                           T=T, x0=x0, check_consistency=False)

    def f(T):
        return point(T).s / n_B - SnB

    f_hi = f(T_hi)
    while f_hi < 0.0 and T_hi < T_cap:
        T_hi = min(2.0 * T_hi, T_cap)
        f_hi = f(T_hi)
    if f_hi < 0.0:
        raise RuntimeError(f"entropy target S={SnB} is unreachable at "
                           f"n_B={n_B} below T={T_cap} MeV")
    return point(brentq(f, T_lo, T_hi, xtol=xtol))


def _sweep_at_entropy(par, flags, n_B_grid, SnB, eta, spec, vmit_params):
    """Warm-started n_B sweep at fixed S = s/n_B (one T solve per point)."""
    slots = mixed_slots(spec, eta, flags)
    out, x0 = [], None
    for n in n_B_grid:
        try:
            r = solve_mixed_at_entropy(par, flags, float(n), SnB, eta, spec,
                                       vmit_params=vmit_params, x0=x0)
        except (RuntimeError, ValueError):
            x0 = None                     # reset the warm start past the gap
            continue
        out.append(r)
        x0 = [r.potentials[s] for s in slots]
    return out


@dataclass
class MixedTableSpec:
    """One multi-axis mixed-EoS table request.

    par, flags : DD2 `Parametrization` and `SpeciesFlags`
    mode       : one of `MODE_FRACTIONS`
    axes       : {'nB': grid, exactly one of 'T'/'SnB': grid, and optionally
                 any of 'Y_C'/'Y_S'/'Y_L': grid to sweep that fraction}
    eta        : local-neutrality fraction (a scalar; loop outside for an
                 eta axis, since eta changes the unknown vector's shape)
    fixed      : scalar values for fractions the mode needs but that are not
                 swept as axes
    leptons    : for the fixed-Y_C modes, whether neutralizing leptons are
                 present (True) or the slice is leptonless (False)
    window_only: solve the mixed system only inside the located transition
                 window (the default and much faster). Set False to solve at
                 every grid point, e.g. when studying chi outside [0, 1].
    """
    par: object
    flags: object
    mode: str
    axes: dict
    eta: float = 0.0
    vmit_params: object = None
    fixed: dict = field(default_factory=dict)
    leptons: bool = True
    window_only: bool = True
    analytic_jac: bool = False

    def __post_init__(self):
        if "nB" not in self.axes:
            raise ValueError("axes must contain 'nB'")
        temp = [k for k in self.axes if k in ("T", "SnB")]
        if len(temp) != 1:
            raise ValueError("axes needs exactly one of 'T' / 'SnB'")
        if self.mode not in MODE_FRACTIONS:
            raise ValueError(f"unknown mode {self.mode!r}; expected one of "
                             f"{list(MODE_FRACTIONS)}")


def _locate_chained(spec, nB, cs, vp, T, hint):
    """`locate_window`, told where the previous temperature found the window.

    A search over the whole density grid spreads its dozen probes across a
    range that is almost all pure phase, and the few that land inside the
    window take steps far larger than the mixed solve can follow. At low
    temperature the refinement recovers from that; by T ~ 80 MeV, where the
    window has moved down to a few tenths of fm^-3 and narrowed, it does not,
    and the boundary search fails outright on scattered temperatures. Since the
    boundaries move smoothly with T, the temperature just solved says where to
    look, and concentrating the probes there both finds the window and costs
    several times less.

    The hint is discarded and the full-grid search repeated when it finds
    nothing, or when a boundary lands on the edge of the hinted range: that
    means the window has moved out from under the hint, and the number would be
    the edge of the search box rather than a phase boundary. So a window that
    genuinely disappears is still reported as gone, and one that jumps is still
    found -- the hint is an accelerator, never the source of the answer.
    """
    kw = dict(vmit_params=vp, T=T, analytic_jac=spec.analytic_jac)
    if hint is not None:
        lo, hi = hint
        pad = max(0.15, hi - lo)          # generous: the docstring's advice
        edge = 0.5 * float(np.min(np.diff(np.sort(nB))))
        window = locate_window(spec.par, spec.flags, nB, spec.eta, cs,
                               hint=(lo - pad, hi + pad), **kw)
        if (window.exists and window.n_onset > lo - pad + edge
                and window.n_offset < hi + pad - edge):
            return window
    return locate_window(spec.par, spec.flags, nB, spec.eta, cs, **kw)


def build_mixed_table(spec, progress=None):
    """Solve a `MixedTableSpec` over the product of its temperature and
    fraction axes, warm-started along n_B within each combination.

    Returns `(rows, windows)`: the long-format rows, and a dict mapping each
    (temperature value, fraction values) key to the `MixedWindow` found there —
    which is what the phase-boundary curves T(n_B) are made of.

    `progress` is an optional callable invoked once per axis combination with a
    dict describing what was just solved, so a notebook can print without this
    module importing anything to print with.

    Non-convergent points are skipped rather than filled: this is a raw table,
    not the smoothed core EoS that `build_mixed_eos_table` produces.
    """
    import time

    nB = np.asarray(spec.axes["nB"], dtype=float)
    temp_key = "SnB" if "SnB" in spec.axes else "T"
    temp_vals = np.atleast_1d(np.asarray(spec.axes[temp_key], dtype=float))
    frac_keys = [k for k in ("Y_C", "Y_S", "Y_L") if k in spec.axes]
    frac_grids = [np.atleast_1d(np.asarray(spec.axes[k], float))
                  for k in frac_keys]
    vp = spec.vmit_params
    if vp is None:
        from eos.vmit.parameters import get_vmit_default
        vp = get_vmit_default()

    rows, windows = [], {}
    hint_of = {}                  # last located window, per fraction combo
    for tv in temp_vals:
        for combo in product(*frac_grids) if frac_grids else [()]:
            t0 = time.time()
            fracs = dict(spec.fixed)
            fracs.update(zip(frac_keys, (float(c) for c in combo)))
            for need in MODE_FRACTIONS[spec.mode]:
                if need not in fracs:
                    raise ValueError(f"mode {spec.mode!r} needs {need!r} "
                                     f"as an axis or in `fixed`")
            cs = make_charge_spec(spec.mode, fracs, leptons=spec.leptons)
            key = (float(tv),) + tuple(round(float(c), 6) for c in combo)

            if temp_key == "SnB":
                results = _sweep_at_entropy(spec.par, spec.flags, nB,
                                            float(tv), spec.eta, cs, vp)
                window = None
            elif spec.window_only:
                window = _locate_chained(spec, nB, cs, vp, float(tv),
                                         hint_of.get(combo))
                if window.exists:
                    hint_of[combo] = (window.n_onset, window.n_offset)
                    inside = nB[(nB >= window.n_onset) & (nB <= window.n_offset)]
                    results = sweep_mixed(spec.par, spec.flags, inside,
                                          spec.eta, cs, vmit_params=vp,
                                          T=float(tv),
                                          analytic_jac=spec.analytic_jac)
                else:
                    results = []
                windows[key] = window
            else:
                results = sweep_mixed(spec.par, spec.flags, nB, spec.eta, cs,
                                      vmit_params=vp, T=float(tv),
                                      analytic_jac=spec.analytic_jac)
                window = None

            for r in results:
                row = composition_row(r)
                for k in MODE_FRACTIONS[spec.mode]:
                    row[k] = fracs[k]
                if temp_key == "SnB":
                    row["SnB"] = float(tv)
                rows.append(row)

            if progress is not None:
                progress(dict(temp_key=temp_key, temp=float(tv),
                              fractions=dict(zip(frac_keys, combo)),
                              eta=spec.eta, n_points=len(results),
                              window=window, seconds=time.time() - t0))
    return rows, windows
