"""
solver.py
====================
T = 0 nucleonic DD2 solves (milestone M1): given a fixed composition
(n_n, n_p), solve the sigma gap equation and assemble the full thermodynamic
state. Symmetric nuclear matter is the equal-density special case.

Golden-point convention (dd2_reference_validation.py, the executable spec):
the uniform-matter kernel uses the AVERAGE nucleon mass (m_n + m_p)/2 for
both species; m_n, m_p enter only through that average.

User-facing units: densities fm^-3, fields/potentials MeV, eps/P MeV/fm^3.
Internally natural units (MeV powers), converted at the boundary via hc^3.
"""
from dataclasses import dataclass

from scipy.optimize import brentq

from eos.general.physics_constants import hc3
from eos.general.particles import Neutron, Proton
from eos.dd2.xp import xp
from eos.dd2.physics.thermo import (
    kF_from_n, scalar_density_t0, eps_kin_t0, P_kin_t0,
)
from eos.dd2.physics.fields import vector_fields, rearrangement, field_eps_P

#: HugenholtzVan Hove residual gate, relative to eps (report §3.x).
HVH_RTOL = 1.0e-8


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
    mu_n: float         # MeV
    mu_p: float         # MeV
    eps: float          # MeV/fm^3
    P: float            # MeV/fm^3
    s: float            # fm^-3 (entropy density; 0 at T=0)
    hvh_rel: float      # (eps + P - sum mu_i n_i)/eps, diagnostics


def solve_composition_t0(par, n_n, n_p, check_consistency=True):
    """
    Solve DD2 nucleonic matter at T=0 for fixed composition (n_n, n_p) [fm^-3].

    Raises ValueError if the Hugenholtz–Van Hove identity fails the HVH_RTOL
    gate (thermodynamic-consistency assertion, report ground rule 4).
    """
    n_B = n_n + n_p
    if n_B <= 0.0:
        raise ValueError("solve_composition_t0 requires n_n + n_p > 0")
    mbar = par.m_nucleon
    Gs, Gw, Gr, dGs, dGw, dGr = par.couplings_at(n_B)

    nn_nat, np_nat = n_n * hc3, n_p * hc3
    nB_nat = n_B * hc3
    n3_nat = Neutron.t3 * nn_nat + Proton.t3 * np_nat
    kFn = kF_from_n(nn_nat, 2.0) if n_n > 0.0 else 0.0
    kFp = kF_from_n(np_nat, 2.0) if n_p > 0.0 else 0.0

    def gap(sig):
        ms = mbar - Gs * sig
        ns = scalar_density_t0(kFn, ms, 2.0) + scalar_density_t0(kFp, ms, 2.0)
        return sig - Gs * ns / par.m_sigma ** 2

    sigma = brentq(gap, 0.0, 0.999 * mbar / Gs, xtol=1e-12)
    ms = mbar - Gs * sigma
    ns_nat = scalar_density_t0(kFn, ms, 2.0) + scalar_density_t0(kFp, ms, 2.0)

    omega0, rho0 = vector_fields(par, Gw, Gr, nB_nat, n3_nat)
    Sig_R = rearrangement(dGs, dGw, dGr, sigma, omega0, rho0,
                          nB_nat, n3_nat, ns_nat)

    eps_f, P_f = field_eps_P(par, sigma, omega0, rho0)
    eps_nat = eps_kin_t0(kFn, ms, 2.0) + eps_kin_t0(kFp, ms, 2.0) + eps_f
    P_nat = (P_kin_t0(kFn, ms, 2.0) + P_kin_t0(kFp, ms, 2.0) + P_f
             + nB_nat * Sig_R)

    vector_shift = Gw * omega0 + Sig_R
    mu_n = xp.sqrt(kFn ** 2 + ms ** 2) + vector_shift + Gr * Neutron.t3 * rho0
    mu_p = xp.sqrt(kFp ** 2 + ms ** 2) + vector_shift + Gr * Proton.t3 * rho0

    hvh_rel = (eps_nat + P_nat - (mu_n * nn_nat + mu_p * np_nat)) / eps_nat
    if check_consistency and abs(hvh_rel) > HVH_RTOL:
        raise ValueError(
            f"Hugenholtz–Van Hove violated at n_B={n_B}: |{hvh_rel:.2e}| > "
            f"{HVH_RTOL:.0e} — a Sigma^R term is missing or inconsistent")

    return EoSPoint(
        n_B=n_B, T=0.0, n_n=n_n, n_p=n_p,
        sigma=float(sigma), omega0=float(omega0), rho0=float(rho0),
        m_eff=float(ms), Sigma_R=float(Sig_R),
        mu_n=float(mu_n), mu_p=float(mu_p),
        eps=float(eps_nat / hc3), P=float(P_nat / hc3), s=0.0,
        hvh_rel=float(hvh_rel),
    )


def solve_snm_t0(par, n_B, check_consistency=True):
    """Symmetric nuclear matter at T=0: n_n = n_p = n_B/2."""
    return solve_composition_t0(par, 0.5 * n_B, 0.5 * n_B,
                                check_consistency=check_consistency)
