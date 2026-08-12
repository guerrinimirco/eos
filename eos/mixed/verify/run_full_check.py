"""
mixed/verify/run_full_check().py
==============================
Physics-invariant verification suite for the mixed-phase engine.

Run it directly (`python -m eos.mixed.verify.run_full_check()`) for a pass/fail
report. Each check returns a structured result with its worst error rather than
printing, so the suite can also be called from a script or a notebook.

These are invariants the engine must satisfy on its own terms — it is validated
by physics, not by agreement with any earlier implementation:

  1. Euler / Hugenholtz-Van Hove: eps + P = T s + sum_i mu_i n_i for the whole
     mixture, recomputed independently of the solver's own bookkeeping;
  2. mechanical equilibrium: the phase pressures balance, each carrying its own
     local lepton pressure, everywhere in the window;
  3. the two limiting constructions: eta=0 gives a Gibbs mixed phase whose
     pressure rises through the window, eta=1 a flat Maxwell plateau;
  4. cross-mode reproduction: fixing Y_S (or Y_C) at the value beta equilibrium
     produces itself must return the beta-equilibrium state, with the extra
     potentials going to zero. This is the strongest available check that the
     regime machinery is right, since it makes two different unknown vectors
     agree on one physical state;
  5. the analytic Jacobian agrees with a finite difference, and solving with it
     reaches the same root as solving without it;
  6. causality: 0 <= c_s^2 <= 1 and P monotone along the stitched core EoS;
  7. the two sound speeds behave as a first-order transition demands: across a
     Maxwell plateau the equilibrium c_eq collapses to zero while the frozen
     c_ad, which holds the phase fraction fixed, stays finite;
  8. TOV: the mixed core gives a physical maximum mass (optional, slow).
"""
from dataclasses import dataclass, field

import numpy as np

from eos.dd2 import Parametrization, SpeciesFlags
from eos.vmit.parameters import get_vmit_default
from eos.mixed import beta_eq_neutrinoless, fixed_YC, fixed_YC_YS
from eos.mixed.equilibrium.charges import ChargeSpec, Regime
from eos.mixed.solvers.point import solve_mixed
from eos.mixed.solvers.sweep import sweep_mixed
from eos.mixed.equilibrium.residual import build_mixed_ctx, mixed_residual, mixed_slots
from eos.mixed.equilibrium.jacobian import mixed_jacobian
from eos.mixed.tables.core_eos import build_mixed_eos_table


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
        lines = [f"mixed run_full_check: {'PASS' if self.all_passed else 'FAIL'}"]
        for r in self.results:
            tag = "ok " if r.passed else "FAIL"
            lines.append(f"  [{tag}] {r.name:22s} max_err={r.max_error:.2e}  {r.detail}")
        return "\n".join(lines)


def _euler_resid(r):
    """|eps + P - T s - sum_i mu_i n_i| / eps for one solved point.

    Deliberately rebuilds the chemical-potential sum from the solved potentials
    and species densities rather than reading the solver's own bookkeeping, so
    this is an independent oracle and not the same code path twice. Each
    charged-lepton species is counted at its own potential: mu_e for electrons,
    mu_mu = mu_e - mu_nue for muons.
    """
    chi, eta, T = r.chi, r.eta, r.T
    p, e = r.potentials, r.extras
    mu_nue = p.get("mu_nue", 0.0)
    munH = sum(r.th_H.mu_i[n] * dn for n, dn in r.th_H.densities.items())
    munQ = sum(r.th_Q.mu_i[n] * dn for n, dn in r.th_Q.densities.items())
    mdn = (1.0 - chi) * munH + chi * munQ

    def lepton_sum(domain, mu_e):
        return mu_e * domain.n_e + (mu_e - mu_nue) * domain.n_mu

    if "mu_eL_H" in p:
        mdn += eta * ((1.0 - chi) * lepton_sum(e["L_H"], p["mu_eL_H"])
                      + chi * lepton_sum(e["L_Q"], p["mu_eL_Q"]))
    if "mu_eG" in p:
        mdn += (1.0 - eta) * lepton_sum(e["G"], p["mu_eG"])
    if e["nu"] is not None:
        mdn += mu_nue * e["nu"].n
    return abs(r.eps + r.P - T * r.s - mdn) / abs(r.eps)


def _window(par, flags, vp, grid, eta, T=0.0):
    return [r for r in sweep_mixed(par, flags, grid, eta, beta_eq_neutrinoless(),
                                   vmit_params=vp, T=T) if r.in_mixed_phase]


def _check_euler(par, flags, vp, grid):
    worst = 0.0
    for eta in (0.0, 0.5):
        for r in _window(par, flags, vp, grid, eta):
            worst = max(worst, _euler_resid(r))
    return CheckResult("euler/HVH", worst < 1e-8, worst,
                       "eps+P = Ts + sum mu_i n_i over window (eta=0,0.5)")


def _check_mechanical(par, flags, vp, grid):
    worst = 0.0
    for eta in (0.0, 0.5, 1.0):
        for r in _window(par, flags, vp, grid, eta):
            dP = ((r.th_H.P + eta * r.extras["L_H"].P)
                  - (r.th_Q.P + eta * r.extras["L_Q"].P))
            worst = max(worst, abs(dP) / max(abs(r.P), 1.0))
    return CheckResult("mechanical eq", worst < 1e-6, worst,
                       "P_H+eta P_eL_H = P_Q+eta P_eL_Q")


def _check_gibbs_maxwell(par, flags, vp, grid):
    Pg = [r.P for r in _window(par, flags, vp, grid, 0.0)]
    Pm = [r.P for r in _window(par, flags, vp, grid, 1.0)]
    gibbs_spread = (max(Pg) - min(Pg)) if Pg else 0.0
    maxwell_spread = (max(Pm) - min(Pm)) if Pm else 1.0
    # Gibbs: P genuinely varies; Maxwell: flat plateau.
    ok = gibbs_spread > 1.0 and maxwell_spread < 1e-3 * max(Pm + [1.0])
    return CheckResult("Gibbs/Maxwell", ok, maxwell_spread,
                       f"dP_Gibbs={gibbs_spread:.1f}, dP_Maxwell={maxwell_spread:.1e}")


def _check_cross_mode(par, flags, vp, n_B=0.65):
    rA = solve_mixed(par, flags, n_B, 0.0, beta_eq_neutrinoless(), vmit_params=vp)
    Y_C = ((1 - rA.chi) * rA.th_H.n_C + rA.chi * rA.th_Q.n_C) / rA.n_B
    Y_S = ((1 - rA.chi) * rA.th_H.n_S + rA.chi * rA.th_Q.n_S) / rA.n_B
    rC = solve_mixed(par, flags, n_B, 0.0, fixed_YC(Y_C, leptons=True), vmit_params=vp)
    rD = solve_mixed(par, flags, n_B, 0.0,
                     ChargeSpec(S=Regime.GLOBAL, targets={"Y_S": Y_S}), vmit_params=vp)
    worst = max(abs(rC.chi - rA.chi), abs(rD.chi - rA.chi),
                abs(rD.potentials.get("mu_S", 0.0)))
    return CheckResult("cross-mode repro", worst < 1e-6, worst,
                       "fixing Y_C / Y_S at their beta-eq values recovers beta eq")


def _check_jacobian(par, flags, vp, n_B=0.6):
    worst = 0.0
    for eta in (0.0, 0.5):
        r = solve_mixed(par, flags, n_B, eta, beta_eq_neutrinoless(), vmit_params=vp,
                        check_consistency=False)
        slots = mixed_slots(beta_eq_neutrinoless(), eta)
        x = np.array([r.potentials[s] for s in slots])
        ctx = build_mixed_ctx(beta_eq_neutrinoless(), eta, n_B, par, flags, vp)
        Ja = mixed_jacobian(x, ctx)
        Jn = np.zeros_like(Ja)
        for j in range(len(x)):
            h = max(1e-4, 1e-6 * abs(x[j]))
            xp, xm = x.copy(), x.copy()
            xp[j] += h; xm[j] -= h
            Jn[:, j] = (np.array(mixed_residual(xp, ctx))
                        - np.array(mixed_residual(xm, ctx))) / (2.0 * h)
        worst = max(worst, np.abs(Ja - Jn).max())
    return CheckResult("analytic J ~ FD", worst < 1e-6, worst,
                       "mixed_jacobian vs FD (eta=0,0.5)")


def _check_backend_parity(par, flags, vp, n_B=0.6):
    worst = 0.0
    for eta in (0.0, 0.5):
        r = solve_mixed(par, flags, n_B, eta, beta_eq_neutrinoless(), vmit_params=vp,
                        check_consistency=False)
        slots = mixed_slots(beta_eq_neutrinoless(), eta)
        x0 = [r.potentials[s] for s in slots]
        ra = solve_mixed(par, flags, n_B, eta, beta_eq_neutrinoless(), vmit_params=vp,
                         check_consistency=False, analytic_jac=True, x0=x0)
        worst = max(worst, abs(ra.P / r.P - 1.0), abs(ra.chi - r.chi))
    return CheckResult("backend parity", worst < 1e-6, worst,
                       "analytic-J solve reaches numeric-J root")


def _check_causality(par, flags, vp, grid):
    from eos.mixed.responses import sound_speed_eq
    t = build_mixed_eos_table(par, flags, grid, 0.0, beta_eq_neutrinoless(), vmit_params=vp)
    cs2 = sound_speed_eq(t.P, t.eps)
    mono = np.all(np.diff(t.P) > -1e-6)
    ok = mono and np.all(cs2 >= -1e-6) and np.all(cs2 <= 1.0 + 1e-6)
    return CheckResult("causality", ok, float(np.max(cs2)),
                       f"0<=c_s^2<=1, P monotone (max c_s^2={np.max(cs2):.3f})")


def _check_sound_speeds(par, flags, vp, grid):
    """Frozen vs equilibrium through a Maxwell window.

    At eta = 1 the pressure is constant across the mixed phase, so c_eq must
    collapse to zero while c_ad — which holds chi fixed and so cannot slide
    along the plateau — stays finite. Getting this the wrong way round means
    the freeze is not actually freezing anything.
    """
    from eos.mixed.responses import sound_speed_eq, frozen_along
    from eos.mixed.solvers.sweep import locate_window, sweep_mixed

    spec = beta_eq_neutrinoless()
    window = locate_window(par, flags, grid, 1.0, spec, vmit_params=vp)
    if not window.exists:
        return CheckResult("sound speeds", False, float("nan"),
                           "no eta=1 window on this grid")
    inside = grid[(grid >= window.n_onset) & (grid <= window.n_offset)]
    rs = sweep_mixed(par, flags, inside, 1.0, spec, vmit_params=vp)
    P = np.array([r.P for r in rs])
    eps = np.array([r.eps for r in rs])
    c_eq = sound_speed_eq(P, eps)
    c_ad = frozen_along(par, flags, rs, vmit_params=vp)
    good = np.isfinite(c_eq) & np.isfinite(c_ad)
    ok = bool(good.any()
              and np.all(c_ad[good] > c_eq[good])
              and np.all(np.abs(c_eq[good]) < 1e-3)      # plateau: c_eq ~ 0
              and np.all(c_ad[good] > 0.0) and np.all(c_ad[good] <= 1.0))
    return CheckResult("sound speeds", ok, float(np.max(np.abs(c_eq[good]))),
                       f"eta=1 plateau: c_eq={np.mean(c_eq[good]):.2e}, "
                       f"c_ad={np.mean(c_ad[good]):.3f}")


def _check_tov(par, flags, vp, grid):
    from eos.mixed.tables.core_eos import mass_radius_mixed
    r = mass_radius_mixed(par, flags, grid, 1.0, beta_eq_neutrinoless(), vmit_params=vp,
                          n_ec=40, backend="fast")
    ok = 1.9 < r["M_max"] < 2.6 and 10.0 < r["R_Mmax"] < 15.0
    return CheckResult("TOV M_max", ok, max(0.0, 2.0 - r["M_max"]),
                       f"M_max={r['M_max']:.3f} R={r['R_Mmax']:.2f}km")


def run_full_check(par=None, flags=None, vmit_params=None, grid=None,
                   include_tov=True):
    """Run the mixed-phase verification suite. Returns a FullCheckReport()."""
    par = par or Parametrization.from_dd2_defaults()
    flags = flags or SpeciesFlags(hyperons=False, muons=False)
    vp = vmit_params or get_vmit_default()
    # window sweep grid (default vMIT: mixed window ~0.44-1.0); wing for causality.
    grid = np.array(grid) if grid is not None else np.round(np.arange(0.45, 0.86, 0.05), 3)
    full = np.round(np.arange(0.10, 1.25, 0.03), 3)

    report = FullCheckReport()
    report.results.append(_check_euler(par, flags, vp, grid))
    report.results.append(_check_mechanical(par, flags, vp, grid))
    report.results.append(_check_gibbs_maxwell(par, flags, vp, grid))
    report.results.append(_check_cross_mode(par, flags, vp))
    report.results.append(_check_jacobian(par, flags, vp))
    report.results.append(_check_backend_parity(par, flags, vp))
    report.results.append(_check_causality(par, flags, vp, full))
    report.results.append(_check_sound_speeds(par, flags, vp, full))
    if include_tov:
        report.results.append(_check_tov(par, flags, vp, full))
    return report


if __name__ == "__main__":
    print(run_full_check())
