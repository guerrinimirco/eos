"""
mixed/hybrid.py
===============
The stellar-core equation of state across a first-order transition, and its
TOV integration.

*Public API* (re-exported from `eos.mixed`): `MixedEoSTable`,
`build_mixed_eos_table`, `mass_radius_mixed`.

`build_mixed_eos_table` produces one monotone (n_B, P, eps) table made of three
segments:

    n_B < n_onset     the pure hadronic phase          (eos/dd2)
    in the window     the eta-mixed phase              (eos/mixed)
    n_B > n_offset    the pure quark phase             (eos/vmit)

The whole hybrid is at ONE equilibrium: the mode the `ChargeSpec` declares
holds in the wings and the window alike — if Y_C is fixed, it is fixed in all
three segments; if neutrinos are trapped at Y_Le, both wings trap them too.
The wings are the pure models' own per-mode solves (`eos.dd2.sweep_octet`,
the `eos.vmit` mode solvers), dispatched from the spec's regimes by
`_hadronic_kwargs` and `_quark_wing_solve` below. Only the neutrality locality
— eta, local against global — is specific to the mixed region: a pure phase
has one phase to neutralize, so eta has nothing to distribute. A leptonless
fixed-Y_C hybrid is a charged slice whose window is eta-independent.

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
"""
from dataclasses import dataclass

import numpy as np

from eos.dd2.solver import sweep_octet
from eos.vmit.eos import (
    solve_vmit_beta_eq, solve_vmit_fixed_yc,
    solve_vmit_fixed_yc_ys, solve_vmit_trapped_neutrinos,
)
from eos.mixed.charges import Regime
from eos.mixed.boundaries import locate_window
from eos.mixed.solver import sweep_mixed


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


#: Largest pressure inversion treated as round-off, relative to the local P.
#: A Maxwell plateau is EXACTLY constant, so consecutive solves there differ
#: only in the last bits: on a 440-point grid the observed inversions are
#: 1e-16 to 5e-15 of P. Anything larger is not round-off and is left alone for
#: the caller to see.
_P_ROUNDOFF = 1.0e-12


def _enforce_monotone_pressure(P, n_B):
    """Remove round-off pressure inversions before the table reaches TOV.

    CLAUDE.md section 8: a table DELIVERED to a structure solver has P
    non-decreasing in n_B. A Maxwell plateau makes that hard to satisfy by
    accident -- P is physically constant across the whole mixed window, so the
    sign of the difference between neighbours is pure floating-point noise, and
    a 440-point grid produces a few dozen inversions of order 1e-13 MeV/fm^3.

    They are not harmless. The fast TOV backend returns a garbage tidal
    deformability when it meets one -- for a 0.94 Msun star, Lambda = 14
    against the reference's 2.8e6 -- while the masses and radii still agree, so
    nothing else flags it. Clamping restores what the physics already says the
    plateau is.

    A genuinely non-monotone branch is different physics -- mechanical
    instability inside a first-order transition is real, and section 8 says a
    construction must resolve it before TOV. Inversions too large to be
    round-off are therefore left in place rather than smoothed away.
    """
    P = np.asarray(P, dtype=float)
    drop = np.diff(P)
    scale = np.maximum(np.abs(P[:-1]), 1.0)
    roundoff = (drop < 0.0) & (np.abs(drop) <= _P_ROUNDOFF * scale)
    if not roundoff.any():
        return P
    real = (drop < 0.0) & ~roundoff
    if real.any():
        i = int(np.argmax(real))
        raise ValueError(
            f"pressure decreases by {abs(drop[i]):.3e} MeV/fm^3 at "
            f"n_B = {n_B[i]:.4f} fm^-3, far beyond round-off: the branch is "
            f"mechanically unstable and no construction has resolved it "
            f"(CLAUDE.md section 8)")
    return np.maximum.accumulate(P)


def _hadronic_kwargs(spec, flags):
    """`sweep_octet` mode arguments for the hadronic wing, from the spec.

    The dispatch reads the spec's regimes, not a mode name, so any regime
    combination the window can solve gets a consistent wing. Beta equilibrium
    reduces to exactly the call `sweep_beta_eq_octet` makes.
    """
    if spec.C is Regime.NOT_CONSERVED:                  # beta equilibrium
        kw = dict(charge_mode="neutral")
        if spec.L_e is Regime.GLOBAL:                   # trapped neutrinos
            if not flags.neutrinos:
                raise ValueError(
                    "a trapped-neutrino hybrid needs "
                    "SpeciesFlags(neutrinos=True): the hadronic wing solves "
                    "at fixed Y_Le and must carry the neutrino population")
            kw.update(lepton_mode="trapped", Y_Le=spec.targets["Y_Le"])
        return kw
    kw = dict(charge_mode="fixed", Y_C=spec.targets["Y_C"],
              yc_leptons=spec.yc_leptons)
    if spec.S is Regime.GLOBAL:
        kw.update(strange_mode="fixed", Y_S=spec.targets["Y_S"])
    return kw


def _quark_wing_solve(spec, n_B, T, vmit_params):
    """One pure-quark point at the spec's equilibrium.

    The vmit naming differences are absorbed here and go no further: the
    engine's Y_Le is vmit's Y_L, the engine's `leptons` is vmit's
    `include_electrons`. Each point cold-starts from vmit's own default guess
    — those solves are cheap and robust, and a cold start keeps every wing
    row exactly reproducible by the pure model's own call at the same
    conditions, which is what test/mixed/test_hybrid_modes.py asserts.
    """
    if spec.C is Regime.NOT_CONSERVED:                  # beta equilibrium
        if spec.L_e is Regime.GLOBAL:
            return solve_vmit_trapped_neutrinos(n_B, spec.targets["Y_Le"], T,
                                                params=vmit_params)
        return solve_vmit_beta_eq(n_B, T, params=vmit_params)
    if spec.S is Regime.GLOBAL:
        return solve_vmit_fixed_yc_ys(n_B, spec.targets["Y_C"],
                                      spec.targets["Y_S"], T,
                                      params=vmit_params,
                                      include_electrons=spec.yc_leptons)
    return solve_vmit_fixed_yc(n_B, spec.targets["Y_C"], T,
                               params=vmit_params,
                               include_electrons=spec.yc_leptons)


def build_mixed_eos_table(par, flags, n_B_grid, eta, spec, vmit_params=None,
                          T=0.0, analytic_jac=False, window=None):
    """Stitch pure hadronic, eta-mixed and pure quark segments into one core
    EoS — every segment at the equilibrium the `spec` declares.

    Locates the transition first (or reuses a `MixedWindow` passed as `window`),
    then solves each segment only where it applies. If there is no transition on
    this grid the whole table is pure hadronic.

    Returns a `MixedEoSTable` sorted ascending in n_B.
    """
    grid = np.asarray(n_B_grid, dtype=float)
    if vmit_params is None:
        from eos.vmit.parameters import get_vmit_default
        vmit_params = get_vmit_default()
    had_kwargs = _hadronic_kwargs(spec, flags)   # validates before any solve

    if window is None:
        window = locate_window(par, flags, grid, eta, spec,
                               vmit_params=vmit_params, T=T,
                               analytic_jac=analytic_jac)

    rows = []                       # (n_B, P, eps, chi, phase)
    n_lo = window.n_onset if window.exists else np.inf
    n_hi = window.n_offset if window.exists else np.inf

    # 1. pure hadronic wing, at the spec's equilibrium. Above the onset the
    #    hadronic branch is metastable — the mixed phase has taken over — so
    #    it is cut there.
    had_grid = grid[grid < n_lo]
    if had_grid.size:
        for p in sweep_octet(par, had_grid, flags, T=T,
                             stop_at_boundary=True, **had_kwargs):
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

    # 3. pure quark wing, above the offset, at the spec's equilibrium. A point
    #    that fails or does not converge is a hole, the same semantics as the
    #    window sweep.
    for n in grid[grid > n_hi]:
        try:
            q = _quark_wing_solve(spec, float(n), T, vmit_params)
        except Exception:
            continue
        if not q.converged:
            continue
        rows.append((q.n_B, q.P_total, q.e_total, 1.0, "Q"))

    rows.sort(key=lambda row: row[0])
    n_B = np.array([r[0] for r in rows])
    P = np.array([r[1] for r in rows])
    eps = np.array([r[2] for r in rows])
    chi = np.array([r[3] for r in rows])
    phase = np.array([r[4] for r in rows])

    P = _enforce_monotone_pressure(P, n_B)

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
                      backend="fast", table=None, tov_parallel=True):
    """Build the core EoS and run the TOV sequence, giving M(R) and Lambda(M).

    A BPS crust is attached below `n_transition`. The Maxwell tidal correction
    across a density discontinuity is applied automatically when the table
    carries a plateau, so it needs no flag here.

    This is a convenience wrapper, not a separate TOV implementation: the
    integration is `eos.tov`'s, and the equation of state reaches it through
    `MixedEoSTable.to_tov()`, which is the contract to use directly when you
    want to drive `eos.tov` yourself.

    backend       : 'fast' (default) is the Numba solver; 'scipy' is the
                    readable reference it is validated against. They agree to
                    ~1e-4 Msun on M_max at both eta=0 and eta=1 — see
                    test/mixed/test_tov_backend_parity.py.
    tov_parallel  : fast backend only; set False when this call is already
                    inside a parallel map over equations of state.

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
        verbose=False, backend=backend, tov_parallel=tov_parallel,
)
    idx_max, e_c_max, M_max = find_mmax_precise(results)
    M = results[:idx_max + 1, 4]
    R = results[:idx_max + 1, 3]
    R_14 = float(np.interp(1.4, M, R)) if M[-1] >= 1.4 > M[0] else float("nan")
    return dict(M_max=float(M_max), R_Mmax=float(results[idx_max, 3]),
                R_1p4=R_14, e_c_max=float(e_c_max), results=results,
                table=table)
