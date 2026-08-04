"""
mixed/tables/core_eos.py
========================
The stellar-core equation of state across a first-order transition, and its
TOV integration.

*Public API* (re-exported from `eos.mixed`): `MixedEoSTable`,
`build_mixed_eos_table`, `mass_radius_mixed`.

`build_mixed_eos_table` produces one monotone (n_B, P, eps) table made of three
segments:

    n_B < n_onset     pure hadronic beta equilibrium   (eos/dd2)
    in the window     the eta-mixed phase              (eos/mixed)
    n_B > n_offset    pure quark beta equilibrium      (eos/vmit)

The boundaries come from `locate_window`, which reads them off chi — chi = 0 is
where the quark phase first appears, chi = 1 is where the hadronic phase
finally disappears. Because the segments are cut on chi rather than patched
together on pressure, they meet by construction and no interpolation is needed
to close gaps.

eta shapes the window and nothing else: eta = 0 gives a Gibbs mixed phase whose
pressure rises through the window, eta = 1 a Maxwell plateau at constant
pressure with a genuine density jump. Both are handed to `eos/tov/` unchanged;
`compute_tov_sequence` detects the plateau itself and applies the Takatsy &
Kovacs (2020) tidal correction across the discontinuity, so `mass_radius_mixed`
needs no flag for it.

Scope: the pure wings are beta-equilibrium matter, which is the physical
cold-star condition. The fixed-Y_C and trapped-neutrino modes still build a
valid mixed *window*, but their pure wings are not specialised here — those
modes describe snapshot conditions rather than a cold-star core.
"""
from dataclasses import dataclass

import numpy as np

from eos.dd2.solver import sweep_beta_eq_octet
from eos.vmit.eos import solve_vmit_beta_eq
from eos.mixed.solvers.sweep import locate_window, sweep_mixed


@dataclass
class MixedEoSTable:
    """A monotone core equation of state. Feed `.to_tov()` to `eos/tov/`."""
    n_B: np.ndarray          # fm^-3, ascending
    P: np.ndarray            # MeV/fm^3
    eps: np.ndarray          # MeV/fm^3
    chi: np.ndarray          # quark volume fraction, 0 hadronic .. 1 quark
    phase: np.ndarray        # 'H' | 'mix' | 'Q' per row
    eta: float
    T: float
    n_onset: float           # n_B where the quark phase appears (nan if none)
    n_offset: float          # n_B where the hadronic phase vanishes
    P_trans: float           # plateau pressure at eta=1 (nan otherwise)

    @property
    def has_transition(self):
        return np.isfinite(self.n_onset) and np.isfinite(self.n_offset)

    def to_tov(self):
        from eos.tov.solver import EOSTable_for_TOV
        return EOSTable_for_TOV(P=self.P, epsilon=self.eps, nB=self.n_B)


def build_mixed_eos_table(par, flags, n_B_grid, eta, spec, vmit_params=None,
                          T=0.0, analytic_jac=False, window=None):
    """Stitch pure hadronic, eta-mixed and pure quark segments into one core EoS.

    Locates the transition first (or reuses a `MixedWindow` passed as `window`),
    then solves each segment only where it applies. If there is no transition on
    this grid the whole table is pure hadronic.

    Returns a `MixedEoSTable` sorted ascending in n_B.
    """
    grid = np.asarray(n_B_grid, dtype=float)
    if vmit_params is None:
        from eos.vmit.parameters import get_vmit_default
        vmit_params = get_vmit_default()

    if window is None:
        window = locate_window(par, flags, grid, eta, spec,
                               vmit_params=vmit_params, T=T,
                               analytic_jac=analytic_jac)

    rows = []                       # (n_B, P, eps, chi, phase)
    n_lo = window.n_onset if window.exists else np.inf
    n_hi = window.n_offset if window.exists else np.inf

    # 1. pure hadronic wing. Above the onset the hadronic branch is metastable
    #    — the mixed phase has taken over — so it is cut there.
    had_grid = grid[grid < n_lo]
    if had_grid.size:
        for p in sweep_beta_eq_octet(par, had_grid, flags, T=T,
                                     stop_at_boundary=True):
            rows.append((p.n_B, p.P, p.eps, 0.0, "H"))

    # 2. the mixed window, warm-started from the onset.
    if window.exists:
        win_grid = grid[(grid >= n_lo) & (grid <= n_hi)]
        if win_grid.size:
            for r in sweep_mixed(par, flags, win_grid, eta, spec,
                                 vmit_params=vmit_params, T=T,
                                 analytic_jac=analytic_jac):
                # A point that drifted outside (0,1) belongs to a pure wing;
                # the wings below already cover it.
                if r.in_mixed_phase:
                    rows.append((r.n_B, r.P, r.eps, r.chi, "mix"))

    # 3. pure quark wing, above the offset.
    for n in grid[grid > n_hi]:
        try:
            q = solve_vmit_beta_eq(float(n), T, params=vmit_params)
        except Exception:
            continue
        rows.append((q.n_B, q.P_total, q.e_total, 1.0, "Q"))

    rows.sort(key=lambda row: row[0])
    n_B = np.array([r[0] for r in rows])
    P = np.array([r[1] for r in rows])
    eps = np.array([r[2] for r in rows])
    chi = np.array([r[3] for r in rows])
    phase = np.array([r[4] for r in rows])

    mixed_rows = phase == "mix"
    P_trans = (float(np.mean(P[mixed_rows]))
               if window.exists and eta > 0.999 and mixed_rows.any()
               else np.nan)
    return MixedEoSTable(n_B=n_B, P=P, eps=eps, chi=chi, phase=phase,
                         eta=eta, T=T, n_onset=window.n_onset,
                         n_offset=window.n_offset, P_trans=P_trans)


def mass_radius_mixed(par, flags, n_B_grid, eta, spec, vmit_params=None, T=0.0,
                      crust="BPS", n_transition=0.08, n_ec=160,
                      e_c_min=150.0, e_c_max=3000.0, compute_tidal=True,
                      backend="scipy", table=None):
    """Build the core EoS and run the TOV sequence, giving M(R) and Lambda(M).

    A BPS crust is attached below `n_transition`. The Maxwell tidal correction
    across a density discontinuity is applied automatically when the table
    carries a plateau, so it needs no flag here.

    Returns a dict with M_max, R at M_max, R(1.4 Msun), the central energy
    density at M_max, the raw TOV `results` array, and the `table` used. Pass
    an already-built `table` to skip re-solving the EoS.
    """
    import os
    from eos.tov.solver import (
        compute_tov_sequence, find_mmax_precise, generate_ec_logspace,
        CRUST_PATHS,
)
    if table is None:
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
                R_1p4=R_14, e_c_max=float(e_c_max), results=results,
                table=table)
