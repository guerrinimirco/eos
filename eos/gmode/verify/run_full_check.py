"""
gmode/verify/run_full_check.py
==============================
Physics-invariant verification suite for the g-mode engine.

Run it directly (`python -m eos.gmode.verify.run_full_check`) for a pass/fail
report. Each check returns a structured result with its worst error rather than
printing, so the suite can also be called from a script or a notebook.

These are invariants the engine must satisfy on its own terms, not agreement
with any earlier implementation:

  1. background: the profile-returning TOV integration reproduces the mass and
     radius `eos.tov` gets from the same table, and the metric functions join
     the exterior Schwarzschild solution at the surface;
  2. gravity: g = nu'/2, the identity that lets nu be obtained by quadrature
     from the pressure profile instead of by integrating another ODE;
  3. the null test: an equation of state with a single sound speed has N^2 = 0
     and supports no composition g-mode. This is the sharpest available check
     that the solver finds buoyancy modes and not numerical artefacts;
  4. spectrum ordering: g1 > g2 > ..., the whole g-branch below the f-mode, and
     the f-mode of a 1.4 M_sun star near 2 kHz;
  5. eigenfunction: xi_r regular at the centre and carrying exactly one node
     for the fundamental g-mode;
  6. convective stability of the shipped sound speeds: c_ad > c_eq at every
     density, so N^2 >= 0. This fails loudly if the two sound speeds are
     computed for different fluids (leptons in one and not the other);
  7. finite rates: the dynamical sound speed interpolates between the frozen
     and equilibrium limits, dissipates only positively, and peaks at
     gamma = omega;
  8. reaction rates: the beta-equilibration rate reaches a 1 kHz oscillation
     frequency at a few MeV and is nearly density-independent for DD2, which is
     what the Fermi-surface Urca rates predict.
"""
from dataclasses import dataclass, field

import numpy as np

from eos.dd2 import Parametrization, SpeciesFlags
from eos.dd2.coefficients import sound_speed_eq
from eos.dd2.solver import sweep_beta_eq_octet, solve_beta_eq_octet
from eos.tov.solver import (
    EOSTable_for_TOV, solve_tov_single, _create_interpolators,
)
from eos.gmode.background import build_background, with_crust
from eos.gmode.cowling import mode_spectrum, solve_gmode
from eos.gmode.sound_speeds import cs2_frozen_nucleonic, cs2_dynamical
from eos.gmode.rates import equilibration_rate


@dataclass
class CheckResult:
    name: str
    passed: bool
    max_error: float
    detail: str = ""


@dataclass
class FullCheckReport:
    results: list = field(default_factory=list)

    @property
    def all_passed(self):
        return all(r.passed for r in self.results)

    def __str__(self):
        lines = [f"gmode run_full_check: "
                 f"{'PASS' if self.all_passed else 'FAIL'}"]
        for r in self.results:
            tag = "ok " if r.passed else "FAIL"
            lines.append(f"  [{tag}] {r.name:24s} "
                         f"max_err={r.max_error:.2e}  {r.detail}")
        return "\n".join(lines)


def _polytrope():
    """A Gamma = 2 polytrope: a real star with no composition at all."""
    eps = np.logspace(np.log10(0.5), np.log10(4000.0), 1500)
    P = 2.0e-4 * eps**2
    return EOSTable_for_TOV(P=P, epsilon=eps, nB=eps / 939.0), np.gradient(P, eps)


def _check_background_vs_tov():
    eos, cs2 = _polytrope()
    bg = build_background(eos, cs2, cs2, e_c=900.0, n_points=400)
    P_of_e, e_of_P, n_of_P = _create_interpolators(eos)
    ref = solve_tov_single(900.0, eos, P_of_e, e_of_P, n_of_P,
                           compute_baryonic=False, compute_tidal=False)
    err = max(abs(bg.M_msun - ref.M) / ref.M, abs(bg.R - ref.R) / ref.R)
    return CheckResult("background vs eos.tov", err < 1e-3, err,
                       f"M={bg.M_msun:.4f} vs {ref.M:.4f}, "
                       f"R={bg.R:.3f} vs {ref.R:.3f} km")


def _check_metric():
    eos, cs2 = _polytrope()
    bg = build_background(eos, cs2, cs2, e_c=900.0, n_points=800)
    M, R = bg.M, bg.R
    e_nu_err = abs(bg.e_nu[-1] - (1.0 - 2.0 * M / R))
    e_lam_err = abs(bg.e_lam[-1] - 1.0 / (1.0 - 2.0 * M / R))
    nu = np.log(bg.e_nu)
    sl = slice(5, -5)
    g_err = np.max(np.abs(bg.g[sl] - 0.5 * np.gradient(nu, bg.r)[sl])) \
        / np.max(bg.g)
    err = max(e_nu_err, e_lam_err, g_err)
    return CheckResult("metric and gravity", err < 5e-3, err,
                       f"exterior match {max(e_nu_err, e_lam_err):.1e}, "
                       f"g vs nu'/2 {g_err:.1e}")


def _check_null_test():
    """One sound speed => N^2 == 0 => no g-mode, only the f-mode."""
    eos, cs2 = _polytrope()
    bg = build_background(eos, cs2, cs2, M_target=1.4, n_points=400)
    modes = mode_spectrum(bg, nu_min=50.0, nu_max=3000.0, n_scan=110)
    labels = [m.label for m in modes]
    n2_max = float(np.max(np.abs(bg.N2)))
    f_ok = len(modes) == 1 and labels == ["f"] and 1800 < modes[0].nu_hz < 2700
    return CheckResult("null test (N^2 = 0)", f_ok and n2_max == 0.0, n2_max,
                       f"modes={labels}, "
                       f"f={modes[0].nu_hz:.0f} Hz" if modes else "no modes")


def _dd2_inputs(par, flags, n_lo=0.08, n_hi=1.2, n_points=110):
    grid = np.geomspace(n_lo, n_hi, n_points)
    pts = sweep_beta_eq_octet(par, grid, flags, T=0.0, include_photons=False,
                              stop_at_boundary=True)
    P = np.array([p.P for p in pts])
    eps = np.array([p.eps for p in pts])
    n_B = np.array([p.n_B for p in pts])
    Y_p = np.array([p.Y_p for p in pts])
    c_eq = np.array([sound_speed_eq(par, n, flags, T=0.0) for n in n_B])
    c_ad = np.array([cs2_frozen_nucleonic(par, n, y, muons=flags.muons)
                     for n, y in zip(n_B, Y_p)])
    core = EOSTable_for_TOV(P=P, epsilon=eps, nB=n_B)
    return core, c_eq, c_ad, n_B, Y_p


def _check_convective_stability(c_eq, c_ad):
    """c_ad > c_eq everywhere: N^2 >= 0.

    The most likely way to fail this is to pair sound speeds computed for
    different fluids -- one including the neutralising leptons and one not.
    """
    worst = float(np.min(c_ad - c_eq))
    return CheckResult("convective stability", worst > -1e-12, abs(min(worst, 0)),
                       f"min(c_ad^2 - c_eq^2) = {worst:+.2e}, "
                       f"max = {float(np.max(c_ad - c_eq)):+.2e}")


def _check_spectrum(eos, c_eq, c_ad):
    bg = build_background(eos, c_eq, c_ad, M_target=1.4, n_points=500)
    modes = mode_spectrum(bg, nu_min=40.0, nu_max=3000.0, n_scan=120)
    gmodes = [m for m in modes if m.is_gmode]
    fmode = [m for m in modes if m.label == "f"]
    if not gmodes or not fmode:
        return CheckResult("spectrum ordering", False, 1.0,
                           f"labels={[m.label for m in modes]}")
    by_order = sorted(gmodes, key=lambda m: int(m.label[1:]))
    freqs = [m.nu_hz for m in by_order]
    ordered = freqs == sorted(freqs, reverse=True)
    below = max(freqs) < fmode[0].nu_hz
    f_ok = 1500.0 < fmode[0].nu_hz < 3000.0
    return CheckResult("spectrum ordering", ordered and below and f_ok, 0.0,
                       f"g1={freqs[0]:.1f} Hz, f={fmode[0].nu_hz:.0f} Hz, "
                       f"{len(gmodes)} g-modes")


def _check_eigenfunction(eos, c_eq, c_ad):
    bg = build_background(eos, c_eq, c_ad, M_target=1.4, n_points=500)
    g1 = solve_gmode(bg, nu_min=40.0, nu_max=3000.0, n_scan=120)
    peak = np.max(np.abs(g1.xi_r))
    centre = abs(g1.xi_r[0]) / peak
    interior = g1.xi_r[1:-1]
    big = interior[np.abs(interior) > 1e-6 * peak]
    nodes = int(np.count_nonzero(np.diff(np.sign(big)) != 0))
    ok = np.all(np.isfinite(g1.xi_r)) and centre < 1e-3 and nodes == 1
    return CheckResult("g1 eigenfunction", ok, centre,
                       f"nu={g1.nu_hz:.1f} Hz, nodes={nodes}, "
                       f"xi_r(0)/peak={centre:.1e}")


def _check_dynamical_sound_speed():
    c_eq, c_ad, omega = 0.20, 0.28, 2.0 * np.pi * 300.0
    frozen = cs2_dynamical(c_eq, c_ad, 1e-8 * omega, omega)
    equil = cs2_dynamical(c_eq, c_ad, 1e8 * omega, omega)
    gam = np.logspace(-3, 3, 601) * omega
    dy = cs2_dynamical(c_eq, c_ad, gam, omega)
    peak = gam[int(np.argmax(dy.imag))] / omega
    err = max(abs(frozen.real - c_ad), abs(equil.real - c_eq),
              abs(peak - 1.0), abs(min(float(np.min(dy.imag)), 0.0)))
    return CheckResult("dynamical sound speed", err < 0.06, err,
                       f"limits ok, Im >= 0, resonance at "
                       f"gamma/omega = {peak:.2f}")


def _check_rates(par, flags):
    """gamma reaches a 1 kHz oscillation at a few MeV, flat in density."""
    from scipy.optimize import brentq
    omega = 2.0 * np.pi * 1000.0
    cross = []
    for f in (1.0, 2.0, 3.0, 5.0):
        n_B = f * 0.16
        Y_p = solve_beta_eq_octet(par, n_B, flags, T=1.0).Y_p
        cross.append(brentq(
            lambda T: equilibration_rate(par, n_B, Y_p, T) - omega, 0.5, 20.0))
    spread = max(cross) - min(cross)
    ok = all(3.5 < T < 6.5 for T in cross) and spread < 1.0
    return CheckResult("Urca resonance T", ok, spread,
                       f"gamma=omega(1kHz) at T = "
                       f"{min(cross):.2f}-{max(cross):.2f} MeV")


def run_full_check(par=None, flags=None, include_dd2=True):
    """Run the g-mode verification suite. Returns a FullCheckReport."""
    par = par or Parametrization.from_dd2_defaults()
    flags = flags or SpeciesFlags(muons=True)

    report = FullCheckReport()
    report.results.append(_check_background_vs_tov())
    report.results.append(_check_metric())
    report.results.append(_check_null_test())
    report.results.append(_check_dynamical_sound_speed())
    if include_dd2:
        core, c_eq, c_ad, _n, _y = _dd2_inputs(par, flags)
        eos, ce, ca = with_crust(core, c_eq, c_ad, crust="BPS",
                                 n_transition=0.08)
        report.results.append(_check_convective_stability(c_eq, c_ad))
        report.results.append(_check_spectrum(eos, ce, ca))
        report.results.append(_check_eigenfunction(eos, ce, ca))
        report.results.append(_check_rates(par, flags))
    return report


if __name__ == "__main__":
    print(run_full_check())
