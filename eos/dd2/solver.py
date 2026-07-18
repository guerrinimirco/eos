"""
solver.py
====================
DD2 nucleonic solves (milestones M1–M3): fixed-composition and
beta-equilibrium matter at T = 0 and T > 0.

Fixed composition (n_n, n_p): one scalar sigma gap solve; at T > 0 the
kinetic potentials are recovered per species by inverting the JEL density
(eos.general.fermi_integrals.invert_fermi_density). Beta equilibrium: the
potential-driven charge-vector system of physics/residual.py.

Golden-point convention (dd2_reference_validation.py, the executable spec):
the uniform-matter kernel uses the AVERAGE nucleon mass (m_n + m_p)/2 for
both species; m_n, m_p enter only through that average.

User-facing units: densities fm^-3, fields/potentials MeV, eps/P MeV/fm^3,
entropy density fm^-3. Internally natural units (MeV powers), converted at
the boundary via hc^3. All assembled quantities are evaluated from one
consistent set of densities, so the Hugenholtz–Van Hove identity
eps + P - T s = sum_i mu_i n_i holds to round-off and is asserted.
"""
from dataclasses import dataclass, replace

from scipy.optimize import brentq, root

from eos.general.physics_constants import hc3
from eos.general.particles import Electron, Muon, Neutron, Proton
from eos.general.thermodynamics_leptons import photon_thermo
from eos.dd2.xp import xp
from eos.dd2.physics.thermo import kF_from_n, kinetic_thermo
from eos.dd2.physics.fields import vector_fields, rearrangement, field_eps_P
from eos.dd2.physics.residual import (
    beta_eq_nucleon_nus, beta_eq_residual, make_beta_ctx,
)

#: Hugenholtz–Van Hove residual gate, relative to eps (report §3.x).
HVH_RTOL = 1.0e-8

#: Post-solve gate on the (dimensionless) equilibrium residuals (report §3.x).
RESIDUAL_TOL = 1.0e-10


@dataclass(frozen=True)
class EoSPoint:
    """One solved thermodynamic state (lite version of report §3.3)."""
    n_B: float          # fm^-3
    T: float            # MeV
    n_n: float          # fm^-3
    n_p: float          # fm^-3
    sigma: float        # MeV
    omega0: float       # MeV
    rho0: float         # MeV
    m_eff: float        # MeV (Dirac effective nucleon mass)
    Sigma_R: float      # MeV (rearrangement self-energy)
    nu_n: float         # MeV (kinetic potential mu_n - Sigma0_n)
    nu_p: float         # MeV
    mu_n: float         # MeV
    mu_p: float         # MeV
    eps: float          # MeV/fm^3 (total, incl. leptons/photons when present)
    P: float            # MeV/fm^3 (total, incl. leptons/photons when present)
    s: float            # fm^-3 (total entropy density)
    hvh_rel: float      # (eps + P - T s - sum mu_i n_i)/eps, diagnostics
    n_e: float = 0.0    # fm^-3 (net)
    n_mu: float = 0.0   # fm^-3 (net)
    mu_e: float = 0.0   # MeV

    @property
    def Y_p(self):
        return self.n_p / self.n_B

    @property
    def free_energy_density(self):
        """F = eps - T s [MeV/fm^3]."""
        return self.eps - self.T * self.s


def _nucleon_nus(n_n, n_p, ms, T):
    """Kinetic potentials hitting the target densities [fm^-3] at mass ms."""
    if T == 0.0:
        nu_n = float(xp.sqrt(kF_from_n(n_n * hc3, 2.0) ** 2 + ms ** 2)) \
            if n_n > 0.0 else 0.0
        nu_p = float(xp.sqrt(kF_from_n(n_p * hc3, 2.0) ** 2 + ms ** 2)) \
            if n_p > 0.0 else 0.0
        return nu_n, nu_p
    from eos.general.fermi_integrals import invert_fermi_density
    nu_n = invert_fermi_density(n_n, T, ms, 2.0) if n_n > 0.0 else 0.0
    nu_p = invert_fermi_density(n_p, T, ms, 2.0) if n_p > 0.0 else 0.0
    return nu_n, nu_p


def solve_composition(par, n_n, n_p, T=0.0, check_consistency=True):
    """
    Solve DD2 nucleonic matter for fixed composition (n_n, n_p) [fm^-3] at
    temperature T [MeV].

    Raises ValueError if the Hugenholtz–Van Hove identity fails the HVH_RTOL
    gate (thermodynamic-consistency assertion, report ground rule 4).
    """
    n_B = n_n + n_p
    if n_B <= 0.0:
        raise ValueError("solve_composition requires n_n + n_p > 0")
    mbar = par.m_nucleon
    Gs, _, _, _, _, _ = par.couplings_at(n_B)

    def gap(sig):
        ms = mbar - Gs * sig
        nu_n, nu_p = _nucleon_nus(n_n, n_p, ms, T)
        ns = (kinetic_thermo(nu_n, ms, 2.0, T)[4]
              + kinetic_thermo(nu_p, ms, 2.0, T)[4])
        return sig - Gs * ns / par.m_sigma ** 2

    sigma = brentq(gap, 0.0, 0.999 * mbar / Gs, xtol=1e-12)
    ms = mbar - Gs * sigma
    nu_n, nu_p = _nucleon_nus(n_n, n_p, ms, T)
    tn = kinetic_thermo(nu_n, ms, 2.0, T)
    tp = kinetic_thermo(nu_p, ms, 2.0, T)

    # Assemble everything from ONE consistent density set (the evaluated
    # densities; at T=0 they equal the targets, at T>0 to inversion tol).
    nn_nat, np_nat = tn[0], tp[0]
    nB_nat = nn_nat + np_nat
    n3_nat = Neutron.t3 * nn_nat + Proton.t3 * np_nat
    ns_nat = tn[4] + tp[4]
    Gs, Gw, Gr, dGs, dGw, dGr = par.couplings_at(nB_nat / hc3)

    omega0, rho0 = vector_fields(par, Gw, Gr, nB_nat, n3_nat)
    Sig_R = rearrangement(dGs, dGw, dGr, sigma, omega0, rho0,
                          nB_nat, n3_nat, ns_nat)

    eps_f, P_f = field_eps_P(par, sigma, omega0, rho0)
    eps_nat = tn[2] + tp[2] + eps_f
    P_nat = tn[1] + tp[1] + P_f + nB_nat * Sig_R
    s_nat = tn[3] + tp[3]

    vector_shift = Gw * omega0 + Sig_R
    mu_n = nu_n + vector_shift + Gr * Neutron.t3 * rho0
    mu_p = nu_p + vector_shift + Gr * Proton.t3 * rho0

    hvh_rel = (eps_nat + P_nat - T * s_nat
               - (mu_n * nn_nat + mu_p * np_nat)) / eps_nat
    if check_consistency and abs(hvh_rel) > HVH_RTOL:
        raise ValueError(
            f"Hugenholtz–Van Hove violated at n_B={n_B}, T={T}: "
            f"|{hvh_rel:.2e}| > {HVH_RTOL:.0e} — a Sigma^R term is missing "
            f"or inconsistent")

    return EoSPoint(
        n_B=float(nB_nat / hc3), T=T,
        n_n=float(nn_nat / hc3), n_p=float(np_nat / hc3),
        sigma=float(sigma), omega0=float(omega0), rho0=float(rho0),
        m_eff=float(ms), Sigma_R=float(Sig_R),
        nu_n=float(nu_n), nu_p=float(nu_p),
        mu_n=float(mu_n), mu_p=float(mu_p),
        eps=float(eps_nat / hc3), P=float(P_nat / hc3),
        s=float(s_nat / hc3), hvh_rel=float(hvh_rel),
    )


def solve_composition_t0(par, n_n, n_p, check_consistency=True):
    """T=0 fixed-composition solve (M1 API)."""
    return solve_composition(par, n_n, n_p, T=0.0,
                             check_consistency=check_consistency)


def solve_snm(par, n_B, T=0.0, check_consistency=True):
    """Symmetric nuclear matter: n_n = n_p = n_B/2."""
    return solve_composition(par, 0.5 * n_B, 0.5 * n_B, T=T,
                             check_consistency=check_consistency)


def solve_snm_t0(par, n_B, check_consistency=True):
    """T=0 symmetric nuclear matter (M1 API)."""
    return solve_snm(par, n_B, T=0.0, check_consistency=check_consistency)


def beta_warm_start(point):
    """Warm-start vector [sigma, rho0, nu_n, mu_Q] from a solved EoSPoint."""
    return [point.sigma, point.rho0, point.nu_n, -point.mu_e]


def default_beta_guess(par, n_B, T=0.0, Y_p=0.05):
    """
    Starting vector [sigma, rho0, nu_n, mu_Q] from an exactly solved
    fixed-composition point at Y_p: only the charge closure is off.
    """
    base = solve_composition(par, (1.0 - Y_p) * n_B, Y_p * n_B, T=T)
    return [base.sigma, base.rho0, base.nu_n, -(base.mu_n - base.mu_p)]


def solve_beta_eq(par, n_B, T=0.0, x0=None, include_muons=True,
                  include_photons=True, check_consistency=True):
    """
    Neutrino-transparent beta-equilibrium npemu matter at density n_B
    [fm^-3] and temperature T [MeV] (report §1.7 mode 1: mu_S = mu_L = 0,
    charge neutrality). Photons contribute at T > 0 when include_photons.

    x0: optional warm-start vector [sigma, rho0, nu_n, mu_Q], e.g. from
    beta_warm_start() of a neighbouring solution. Falls back to the default
    guess if the warm start stalls; raises RuntimeError on non-convergence
    — no silent failures.
    """
    ctx = make_beta_ctx(par, n_B, T=T, include_muons=include_muons)
    guesses = [x0] if x0 is not None else []
    guesses.append(default_beta_guess(par, n_B, T=T))
    sol = None
    for guess in guesses:
        sol = root(beta_eq_residual, guess, args=(ctx,), method="hybr",
                   tol=1e-13)
        res_max = max(abs(r) for r in beta_eq_residual(sol.x, ctx))
        if sol.success and res_max <= RESIDUAL_TOL:
            break
    else:
        raise RuntimeError(
            f"beta-equilibrium solve failed at n_B={n_B}, T={T}: "
            f"{sol.message} (max residual {res_max:.2e}, "
            f"tol {RESIDUAL_TOL:.0e})")

    # Converged composition -> assemble the hadronic sector through the same
    # path as the fixed-composition solve (one source of truth).
    nu_n, nu_p, ms = beta_eq_nucleon_nus(sol.x, ctx)
    n_n = kinetic_thermo(nu_n, ms, 2.0, T)[0] / hc3
    n_p = kinetic_thermo(nu_p, ms, 2.0, T)[0] / hc3
    base = solve_composition(par, n_n, n_p, T=T,
                             check_consistency=check_consistency)

    mu_e = -sol.x[3]
    ne_nat, Pe, ee, se, _ = kinetic_thermo(mu_e, Electron.mass, 2.0, T)
    if include_muons:
        nmu_nat, Pmu, emu, smu, _ = kinetic_thermo(mu_e, Muon.mass, 2.0, T)
    else:
        nmu_nat = Pmu = emu = smu = 0.0
    if include_photons and T > 0.0:
        ph = photon_thermo(T)
        Pph, eph, sph = ph.P * hc3, ph.e * hc3, ph.s * hc3
    else:
        Pph = eph = sph = 0.0

    eps_nat = base.eps * hc3 + ee + emu + eph
    P_nat = base.P * hc3 + Pe + Pmu + Pph
    s_nat = base.s * hc3 + se + smu + sph
    rhs = (base.mu_n * base.n_n + base.mu_p * base.n_p) * hc3 \
        + mu_e * (ne_nat + nmu_nat)
    hvh_rel = (eps_nat + P_nat - T * s_nat - rhs) / eps_nat
    beta_res = base.mu_n - base.mu_p - mu_e
    if check_consistency:
        if abs(hvh_rel) > HVH_RTOL:
            raise ValueError(
                f"Hugenholtz–Van Hove violated at n_B={n_B}, T={T} "
                f"(beta-eq): |{hvh_rel:.2e}| > {HVH_RTOL:.0e}")
        if abs(beta_res) > 1e-6:
            raise ValueError(
                f"beta-equilibrium condition violated at n_B={n_B}, T={T}: "
                f"mu_n - mu_p - mu_e = {beta_res:.2e} MeV")

    return replace(
        base,
        eps=float(eps_nat / hc3), P=float(P_nat / hc3),
        s=float(s_nat / hc3), hvh_rel=float(hvh_rel),
        n_e=float(ne_nat / hc3), n_mu=float(nmu_nat / hc3), mu_e=float(mu_e),
    )


def solve_beta_eq_t0(par, n_B, x0=None, include_muons=True,
                     check_consistency=True):
    """T=0 beta-equilibrium solve (M2 API)."""
    return solve_beta_eq(par, n_B, T=0.0, x0=x0, include_muons=include_muons,
                         check_consistency=check_consistency)
