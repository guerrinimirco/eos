"""
mixed/table.py
==============
Full stellar-core EoS across the first-order transition, and its TOV integration
(docs/phase2/SPECIFICATION_AND_PLAN.md §4 milestone P8).

`build_mixed_eos_table` stitches three segments over an n_B grid into one
monotone (P, eps, n_B) table:

  - pure hadronic beta-equilibrium below the mixed onset   (`sweep_beta_eq_octet`)
  - the eta-mixed window                                    (`sweep_mixed`)
  - pure quark beta-equilibrium above the mixed offset      (`solve_vmit_beta_eq`)

The wings are the physical single phases (nucleonic / quark beta-equilibrium,
charge-neutral, with leptons); `spec`/`eta` only shape the WINDOW construction:
eta=0 gives a Gibbs mixed phase (P rises through the window), eta=1 a Maxwell
plateau (constant P, a genuine density jump). Both are handed to `eos/tov/`
unchanged.

Maxwell tidal jump: `eos/tov/solver.compute_tov_sequence` auto-detects the
constant-P plateau (`_detect_maxwell_construction`) and applies the Takatsy &
Kovacs (2020) delta-Y correction across the density discontinuity itself -- so
`mass_radius_mixed` needs no special flag; the correction is on whenever the
table has a Maxwell plateau (eta=1) and off (nothing to correct) for Gibbs.

Scope (ponytail): the wings are beta-equilibrium single phases, i.e. the
physical cold-NS EoS (Mode A). Fixed-Y_C / trapped-nu modes still build a valid
mixed WINDOW, but their non-beta pure wings are not specialised here -- those
modes are snapshot conditions, not cold-NS TOV inputs.
"""
from dataclasses import dataclass, field

import numpy as np

from eos.dd2.solver import sweep_beta_eq_octet
from eos.vmit.eos import solve_vmit_beta_eq
from eos.mixed.solver import solve_mixed
from eos.mixed.continuation import sweep_mixed


@dataclass
class MixedEoSTable:
    """A monotone core EoS. Feed `.to_tov()` to `eos/tov/`."""
    n_B: np.ndarray          # fm^-3, ascending
    P: np.ndarray            # MeV/fm^3
    eps: np.ndarray          # MeV/fm^3
    chi: np.ndarray          # quark volume fraction (0 hadronic .. 1 quark)
    phase: np.ndarray        # 'H' | 'mix' | 'Q' per row
    eta: float
    T: float
    onset: float             # n_B of mixed onset  (nan if no transition)
    offset: float            # n_B of mixed offset (nan if no transition)
    P_trans: float           # plateau pressure at eta=1 (nan for Gibbs/no trans)

    def to_tov(self):
        from eos.tov.solver import EOSTable_for_TOV
        return EOSTable_for_TOV(P=self.P, epsilon=self.eps, nB=self.n_B)


def build_mixed_eos_table(par, flags, n_B_grid, eta, spec,
                          vmit_params=None, T=0.0):
    """Stitch pure-hadronic + mixed-window + pure-quark into one core EoS.

    Runs the eta-mixed sweep to locate the window (points with 0<chi<1), then
    fills the density grid below/above it with the pure beta-equilibrium phases.
    If there is no window (no quark transition for these params) the whole table
    is pure hadronic. Returns a `MixedEoSTable` sorted ascending in n_B.
    """
    n_B_grid = np.asarray(n_B_grid, dtype=float)
    if vmit_params is None:
        from eos.vmit.parameters import get_vmit_default
        vmit_params = get_vmit_default()

    # 1. mixed window (warm-started; skips non-convergent points)
    win = [r for r in sweep_mixed(par, flags, n_B_grid, eta, spec,
                                  vmit_params=vmit_params, T=T)
           if r.in_mixed_phase]
    if win:
        n_lo = min(r.n_B for r in win)
        n_hi = max(r.n_B for r in win)
        P_lo = min(r.P for r in win)             # transition pressure range:
        P_hi = max(r.P for r in win)             # a plateau at eta=1, a band for Gibbs
        onset, offset = n_lo, n_hi
    else:
        n_lo = n_hi = np.inf                      # everything is hadronic
        P_lo = P_hi = np.inf
        onset = offset = np.nan

    rows = []   # (n_B, P, eps, chi, phase)

    # 2. pure hadronic wing: only the branch BELOW the mixed pressure range.
    #    (Above P_lo the pure hadronic phase is metastable -- replaced by the
    #    mixed phase; keeping it would double back in P.)
    had_grid = n_B_grid[n_B_grid < n_lo]
    if had_grid.size:
        for p in sweep_beta_eq_octet(par, had_grid, flags, T=T):
            if p.P <= P_lo + 1e-9:
                rows.append((p.n_B, p.P, p.eps, 0.0, "H"))

    # 3. mixed window
    for r in win:
        rows.append((r.n_B, r.P, r.eps, r.chi, "mix"))

    # 4. pure quark wing: only the branch ABOVE the mixed pressure range. The
    #    mixed sweep can undershoot chi->1 (numeric-Jacobian stall), so guard on
    #    P, not on n_B -- a quark point with P < P_hi is still inside the
    #    transition (the density-jump region) and must not intrude below the
    #    plateau. The n_B gap it leaves is the physical Maxwell discontinuity.
    for n in n_B_grid[n_B_grid > n_hi]:
        q = solve_vmit_beta_eq(float(n), T, params=vmit_params)
        if q.P_total >= P_hi - 1e-9:
            rows.append((q.n_B, q.P_total, q.e_total, 1.0, "Q"))

    # Fill interior grid points the mixed sweep dropped WITHIN the converged
    # window [n_lo, n_hi]. The numeric-Jacobian solve has an intermittent
    # non-convergence band there (hybr stalls; the deferred analytic Jacobian P7
    # removes it). Converged neighbours pin a smooth monotone P(n_B)/eps(n_B)
    # through a mixed phase, so PCHIP fills the gaps below TOV resolution.
    # ponytail: interpolation-fill; drop it once P7 makes every point converge.
    if win:
        got = {round(r[0], 9) for r in rows}
        holes = [g for g in n_B_grid if n_lo < g < n_hi and round(float(g), 9) not in got]
        if holes:
            from scipy.interpolate import PchipInterpolator
            base = sorted(r for r in rows if r[4] == "mix")
            xb = np.array([r[0] for r in base])
            fP = PchipInterpolator(xb, [r[1] for r in base])
            fe = PchipInterpolator(xb, [r[2] for r in base])
            fc = PchipInterpolator(xb, [r[3] for r in base])
            for g in holes:
                rows.append((float(g), float(fP(g)), float(fe(g)), float(fc(g)), "mix"))

    rows.sort(key=lambda row: row[0])
    n_B = np.array([r[0] for r in rows])
    P = np.array([r[1] for r in rows])
    eps = np.array([r[2] for r in rows])
    chi = np.array([r[3] for r in rows])
    phase = np.array([r[4] for r in rows])

    P_trans = (float(np.mean(P[phase == "mix"]))
               if win and eta > 0.999 else np.nan)
    return MixedEoSTable(n_B=n_B, P=P, eps=eps, chi=chi, phase=phase,
                         eta=eta, T=T, onset=onset, offset=offset,
                         P_trans=P_trans)


def _composition_row(r):
    """Flatten one MixedResult into a dict row: bulk thermodynamics + global
    (volume-weighted) composition Y_i = w * n_i / n_B (hadron w=1-chi, quark
    w=chi), Y_C/Y_S summed over both phases, and any global charge potential the
    mode carries (mu_C / mu_S / mu_L)."""
    row = dict(n_B=r.n_B, T=r.T, eta=r.eta, chi=r.chi,
               P=r.P, eps=r.eps, s=r.s, mu_B=r.mu_B,
               Y_C=((1 - r.chi) * r.th_H.n_C + r.chi * r.th_Q.n_C) / r.n_B,
               Y_S=((1 - r.chi) * r.th_H.n_S + r.chi * r.th_Q.n_S) / r.n_B)
    for key in ("mu_C", "mu_S", "mu_L"):        # global potentials, mode-dependent
        if key in r.potentials:
            row[key] = r.potentials[key]
    for name, n in r.th_H.densities.items():
        row[f"Y_{name}"] = (1 - r.chi) * n / r.n_B
    for name, n in r.th_Q.densities.items():
        row[f"Y_{name}"] = r.chi * n / r.n_B
    return row


def composition_table(par, flags, n_B_grid, eta, spec, vmit_params=None, T=0.0):
    """Flatten one eta-mixed n_B sweep (fixed spec, T) into composition rows.

    Convenience one-liner; for multi-axis (n_B x Y_C x T etc.) tables covering
    every mode use `build_mixed_table`. Returns a list of dicts -- wrap in
    pandas.DataFrame(rows) in the notebook (the engine stays dependency-free).
    """
    return [_composition_row(r) for r in
            sweep_mixed(par, flags, n_B_grid, eta, spec,
                        vmit_params=vmit_params, T=T)]


# =============================================================================
# General multi-axis table driver (all modes; T or entropy-per-baryon axis)
# =============================================================================
#: mode name -> the fixed fractions it consumes (mixed analog of dd2 _MODE_FIXED).
_MIXED_MODE_FRACS = {
    "beta": (), "YC": ("Y_C",), "YC+YS": ("Y_C", "Y_S"), "YL": ("Y_L",),
}


def _make_mixed_spec(mode, fracs, yc_leptons):
    from eos.mixed import mode_A, mode_B, mode_C, mode_D
    if mode == "beta":
        return mode_A()
    if mode == "YC":
        return mode_C(fracs["Y_C"], yc_leptons=yc_leptons)
    if mode == "YC+YS":
        return mode_D(fracs["Y_C"], fracs["Y_S"], yc_leptons=yc_leptons)
    if mode == "YL":
        return mode_B(fracs["Y_L"])
    raise ValueError(f"unknown mixed mode {mode!r}; expected {list(_MIXED_MODE_FRACS)}")


def solve_mixed_at_entropy(par, flags, n_B, SnB, eta, spec, vmit_params=None,
                           x0=None, T_lo=0.5, T_hi=50.0, T_cap=250.0, xtol=1e-4):
    """Mixed solve at fixed entropy per baryon S=s/n_B by an outer 1-D root on T
    (mirrors eos/dd2/table.solve_octet_at_entropy). s(T) is monotone increasing
    at fixed n_B so the bracket is well posed; T_hi doubles up to T_cap."""
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
        raise RuntimeError(f"entropy target S={SnB} unreachable at n_B={n_B} "
                           f"below T={T_cap} MeV")
    return point(brentq(f, T_lo, T_hi, xtol=xtol))


def _sweep_at_entropy(par, flags, n_B_grid, SnB, eta, spec, vmit_params):
    """Warm-started n_B sweep at fixed S=s/n_B (per-point T solve)."""
    from eos.mixed.residual import mixed_slots
    slots = mixed_slots(spec, eta)
    out, x0 = [], None
    for n in n_B_grid:
        try:
            r = solve_mixed_at_entropy(par, flags, float(n), SnB, eta, spec,
                                       vmit_params=vmit_params, x0=x0)
        except (RuntimeError, ValueError):
            x0 = None                     # reset warm start past the gap
            continue
        out.append(r)
        x0 = [r.potentials[s] for s in slots]
    return out


@dataclass
class MixedTableSpec:
    """One multi-axis mixed-EoS table request (the mixed analog of
    eos/dd2/table.TableSpec).

    mode  : 'beta' | 'YC' | 'YC+YS' | 'YL'
    axes  : {'nB': grid, one of 'T'/'SnB': grid, and optionally any of
             'Y_C'/'Y_S'/'Y_L': grid to sweep that fraction as an axis}
    eta   : local/global neutrality parameter (scalar; loop it outside for an
            eta axis). fixed : scalar fractions the mode needs but are not swept.
    yc_leptons : §1.6 charge flavor for the YC / YC+YS modes (True = neutralising
            leptons present; False = leptonless CompOSE (n_B,T,Y_q) convention).
    """
    par: object
    flags: object
    mode: str
    axes: dict
    eta: float = 0.0
    vmit_params: object = None
    fixed: dict = field(default_factory=dict)
    yc_leptons: bool = True

    def __post_init__(self):
        if "nB" not in self.axes:
            raise ValueError("axes must contain 'nB'")
        temp = [k for k in self.axes if k in ("T", "SnB")]
        if len(temp) != 1:
            raise ValueError("axes needs exactly one of 'T' / 'SnB'")
        if self.mode not in _MIXED_MODE_FRACS:
            raise ValueError(f"unknown mode {self.mode!r}")


def build_mixed_table(spec):
    """Solve a MixedTableSpec over the Cartesian product of its temperature and
    fraction axes, warm-started along n_B at each combination. Returns long-format
    rows (list of dicts) -- one per converged (n_B, T/SnB, Y_*) point, with the
    axis values and the volume-weighted composition. Wrap in pandas in the
    notebook: `pandas.DataFrame(build_mixed_table(spec))`.

    Every mode generates tables through this one path (spec §1.5): 'beta' ->
    (n_B, T) or (n_B, SnB); 'YC' -> add a Y_C axis; 'YC+YS' -> Y_C and Y_S;
    'YL' -> Y_L (trapped neutrinos). Missing points (non-convergent) are skipped,
    not filled -- this is a raw table, not the TOV core EoS.
    """
    from itertools import product

    nB = np.asarray(spec.axes["nB"], dtype=float)
    temp_key = "SnB" if "SnB" in spec.axes else "T"
    temp_vals = np.atleast_1d(np.asarray(spec.axes[temp_key], dtype=float))
    frac_keys = [k for k in ("Y_C", "Y_S", "Y_L") if k in spec.axes]
    frac_grids = [np.atleast_1d(np.asarray(spec.axes[k], float)) for k in frac_keys]
    vp = spec.vmit_params
    if vp is None:
        from eos.vmit.parameters import get_vmit_default
        vp = get_vmit_default()

    rows = []
    for tv in temp_vals:
        for combo in product(*frac_grids):
            fracs = dict(spec.fixed)
            fracs.update(zip(frac_keys, (float(c) for c in combo)))
            for need in _MIXED_MODE_FRACS[spec.mode]:
                if need not in fracs:
                    raise ValueError(f"mode {spec.mode!r} needs {need!r} "
                                     f"(as an axis or in fixed)")
            cs = _make_mixed_spec(spec.mode, fracs, spec.yc_leptons)
            if temp_key == "T":
                results = sweep_mixed(spec.par, spec.flags, nB, spec.eta, cs,
                                      vmit_params=vp, T=float(tv))
            else:
                results = _sweep_at_entropy(spec.par, spec.flags, nB, float(tv),
                                            spec.eta, cs, vp)
            for r in results:
                row = _composition_row(r)
                for key in _MIXED_MODE_FRACS[spec.mode]:
                    row[key] = fracs[key]
                if temp_key == "SnB":
                    row["SnB"] = float(tv)
                rows.append(row)
    return rows


def mass_radius_mixed(par, flags, n_B_grid, eta, spec, vmit_params=None, T=0.0,
                      crust="BPS", n_transition=0.08, n_ec=160,
                      e_c_min=150.0, e_c_max=3000.0, compute_tidal=True,
                      backend="scipy"):
    """Build the core EoS and run the TOV sequence -> M(R), Lambda(M).

    Mirrors `eos/dd2/verify/tov.py::mass_radius`. The BPS crust is attached below
    `n_transition`; the Maxwell delta-Y tidal correction is applied automatically
    by `compute_tov_sequence` when the eta=1 table carries a plateau. Returns a
    dict: M_max, R_Mmax, R_1p4, e_c_max, the raw `results` array, and the
    `table` used.
    """
    import os
    from eos.tov.solver import (
        compute_tov_sequence, find_mmax_precise, generate_ec_logspace,
        CRUST_PATHS,
    )
    table = build_mixed_eos_table(par, flags, n_B_grid, eta, spec,
                                  vmit_params=vmit_params, T=T)
    if crust == "BPS" and not os.path.isfile(CRUST_PATHS.get("BPS", "")):
        crust = "No"
    e_c_vec = generate_ec_logspace(e_c_min, e_c_max, n_ec)
    results = compute_tov_sequence(
        table.to_tov(), e_c_vec, add_crust_table=crust, add_crust_mode="attach",
        n_transition=(n_transition if crust != "No" else None),
        compute_baryonic_mass=False, compute_tidal=compute_tidal,
        verbose=False, backend=backend,
    )
    idx_max, e_c_max, M_max = find_mmax_precise(results)
    M = results[:idx_max + 1, 4]
    R = results[:idx_max + 1, 3]
    R_14 = float(np.interp(1.4, M, R)) if M[-1] >= 1.4 > M[0] else float("nan")
    return dict(M_max=float(M_max), R_Mmax=float(results[idx_max, 3]),
                R_1p4=R_14, e_c_max=float(e_c_max), results=results, table=table)


if __name__ == "__main__":
    # Self-check: nucleonic DD2 + default vMIT has a mixed window ~0.45-0.80.
    # Gibbs (eta=0) -> strictly rising P through it; Maxwell (eta=1) -> a
    # constant-P plateau that eos/tov detects for the tidal jump.
    from eos.dd2 import Parametrization, SpeciesFlags
    from eos.mixed import mode_A

    par = Parametrization.from_dd2_defaults()
    flags = SpeciesFlags(hyperons=False, muons=False)
    grid = np.arange(0.10, 1.01, 0.02)

    t0 = build_mixed_eos_table(par, flags, grid, 0.0, mode_A())
    assert np.all(np.diff(t0.P) > -1e-9), "Gibbs table not monotone in P"
    assert (t0.chi > 0).any() and (t0.chi < 1).any(), "no mixed window found"

    t1 = build_mixed_eos_table(par, flags, grid, 1.0, mode_A())
    from eos.tov.solver import _detect_maxwell_construction
    assert _detect_maxwell_construction(t1.to_tov()) is not None, \
        "eta=1 table has no detectable Maxwell plateau (tidal jump would be missed)"
    print(f"OK  Gibbs onset={t0.onset:.3f} offset={t0.offset:.3f} fm^-3; "
          f"Maxwell P_trans={t1.P_trans:.2f} MeV/fm^3")
