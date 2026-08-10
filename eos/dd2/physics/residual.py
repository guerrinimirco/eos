"""
physics/residual.py
====================
Backend-agnostic residual for the potential-driven DD2 solve.

Unknown vector for beta-equilibrium npemu (the two-charge case):
    x = [sigma, rho0, mu_eff_n, mu_Q]

mu_eff_n is the neutron effective (kinetic) potential. Chemical potentials follow the
conserved-charge decomposition mu_i = B_i mu_B + Q_i mu_Q (so mu_n = mu_B,
mu_p = mu_B + mu_Q, mu_e = -mu_Q; beta equilibrium is automatic), and the
effective potentials are mu_eff_i = mu_i - Sigma0_i. Two consequences of
using mu_eff_n instead of mu_B as the unknown:
  * the species-independent rearrangement term Sigma^R (and the large
    Gamma_omega*omega0 shift) cancel out of the iteration entirely —
    mu_B = mu_eff_n + Sigma0_n is restored at assembly;
  * mu_eff_n varies smoothly with n_B, so warm-starting a density sweep from
    the previous solution is well-conditioned (mu-based unknowns are not:
    the omega shift, evaluated at the target n_B, jumps between sweep
    points and can throw the trial state into the vacuum kF = 0 plateau).
From mu_eff_p - mu_eff_n = mu_Q - Gamma_rho*rho0*(t3_p - t3_n) the proton
potential is mu_eff_p = mu_eff_n + mu_Q - 2*Gamma_rho*rho0.

The isoscalar couplings and omega_0 are evaluated at the TARGET n_B; the
baryon-number residual enforces sum n_i = n_B at the solution, where they
become exactly consistent. rho_0 cannot be eliminated the same way (its
source is the unknown isovector composition), so it is carried as an unknown
with its field equation as residual — the one deviation from the report's
minimal vector [sigma, mu_B, mu_Q].

Residuals are dimensionless: field equations scaled by m_nucleon, density
constraints by n_B. The ctx carries temperature, so the same residual serves
T = 0 (M2) and T > 0 (M3) through kinetic_thermo's two branches.
"""
from typing import NamedTuple

from eos.general.particles import Electron, Muon, Neutron, Proton
from eos.general.physics_constants import hc3
from eos.dd2.physics.thermo import kinetic_thermo


class BetaCtx(NamedTuple):
    nB_nat: float        # target baryon density [MeV^3]
    mbar: float          # average nucleon mass [MeV] (residual scaling)
    m_kn: float          # kernel neutron mass [MeV] (par.kernel_masses)
    m_kp: float          # kernel proton mass [MeV]
    m_sigma2: float      # [MeV^2]
    m_rho2: float        # [MeV^2]
    Gs: float            # Gamma_sigma(n_B target)
    Gw: float
    Gr: float
    m_e: float
    m_mu: float
    T: float             # [MeV]
    include_muons: bool


def make_beta_ctx(par, n_B, T=0.0, include_muons=True):
    Gs, Gw, Gr, _, _, _ = par.couplings_at(n_B)
    m_kn, m_kp = par.kernel_masses()
    return BetaCtx(
        nB_nat=n_B * hc3, mbar=par.m_nucleon, m_kn=m_kn, m_kp=m_kp,
        m_sigma2=par.m_sigma ** 2, m_rho2=par.m_rho ** 2,
        Gs=Gs, Gw=Gw, Gr=Gr,
        m_e=Electron.mass, m_mu=Muon.mass, T=T, include_muons=include_muons,
    )


def beta_eq_nucleon_mu_eff(x, ctx):
    """Effective potentials and effective masses from the unknown vector."""
    sigma, rho0, mu_eff_n, muQ = x
    ms_n = ctx.m_kn - ctx.Gs * sigma
    ms_p = ctx.m_kp - ctx.Gs * sigma
    mu_eff_p = mu_eff_n + muQ - (Proton.t3 - Neutron.t3) * ctx.Gr * rho0
    return mu_eff_n, mu_eff_p, ms_n, ms_p


def beta_eq_residual(x, ctx):
    """
    Dimensionless residuals: [sigma gap, rho0 field eq, baryon number,
    charge neutrality]. Zero exactly at the solved state.
    """
    sigma, rho0, _, muQ = x
    mu_eff_n, mu_eff_p, ms_n, ms_p = beta_eq_nucleon_mu_eff(x, ctx)
    if min(ms_n, ms_p) <= 0.0:
        return [1.0e6, 0.0, 0.0, 0.0]   # outside physical domain

    n_n, _, _, _, ns_n = kinetic_thermo(mu_eff_n, ms_n, 2.0, ctx.T)
    n_p, _, _, _, ns_p = kinetic_thermo(mu_eff_p, ms_p, 2.0, ctx.T)
    mu_e = -muQ
    n_e = kinetic_thermo(mu_e, ctx.m_e, 2.0, ctx.T)[0]
    n_mu = (kinetic_thermo(mu_e, ctx.m_mu, 2.0, ctx.T)[0]
            if ctx.include_muons else 0.0)

    n3 = Neutron.t3 * n_n + Proton.t3 * n_p
    return [
        (sigma - ctx.Gs * (ns_n + ns_p) / ctx.m_sigma2) / ctx.mbar,
        (rho0 - ctx.Gr * n3 / ctx.m_rho2) / ctx.mbar,
        (n_n + n_p) / ctx.nB_nat - 1.0,
        (n_p - n_e - n_mu) / ctx.nB_nat,
    ]
