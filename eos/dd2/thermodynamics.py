"""Quantities computed FROM the state: DD2's kernels and its mean fields.

Nothing here knows which equilibrium mode it is in. Given the chemical
potentials, the meson fields and the temperature, this module returns
densities, energy, pressure, entropy and the sums over them; which of those
quantities a mode holds fixed, and what conditions close the system, belong to
`solver.py`. Grep this file for `beta`, `Y_C`, `neutral` or `trapped` and find
nothing -- that is the boundary, stated as a test.

Two layers, both thermodynamics:

    evaluation          at GIVEN fields and potentials. No solve. This is what
                        the residual calls on every iteration.
    self-consistency    at given CHARGE POTENTIALS, solving DD2's OWN meson
                        fields (and, for one phase of a mixture, its own
                        density). A solve -- but of nothing mode-dependent.
                        `thermo_at_potentials` is that layer, and it is what
                        `eos.mixed`'s hadronic adapter calls.

Natural units inside (energies and masses MeV, densities MeV^3, eps and P
MeV^4); the fm-based boundary of CLAUDE.md section 5 is restored where a
`PhaseThermo` is built.

The density-dependent couplings Gamma_i(n_B) are evaluated at the TARGET n_B
carried in the context, not at the running sum of species densities. The
baryon-number condition makes the two equal at the solution, which is where
the thermodynamics is consistent; away from it they differ, and pinning the
couplings is what keeps the residual smooth.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from eos.general.particles import Electron, Muon
from eos.general.physics_constants import hc3
from eos.general.state import PhaseThermo
from eos.general import thermal_mesons as _gas
try:
    from eos.dd2.backends.kernel_numba import meson_sources_t0, _NUMBA_OK
except ImportError:
    # `backends/` is optional: CLAUDE.md section 5 defines it by the property
    # that deleting it changes no number, only the speed. Without it the plain
    # NumPy path below is the whole implementation.
    meson_sources_t0, _NUMBA_OK = None, False
from eos.dd2.species import active_baryons

_PI2 = np.pi ** 2

#: Neutrino degeneracy (one helicity; antineutrinos handled by kinetic_thermo).
G_NU = 1.0

#: Lambda SU(6) omega ratio entering the kaon omega-shift, used when the
#: parametrization carries no explicit Lambda coupling.
_X_OMEGA_LAMBDA = 2.0 / 3.0


# =============================================================================
# ONE SPECIES: the ideal Fermi gas
# =============================================================================

def kF_from_n(n, g):
    """Fermi momentum [MeV] from number density n [MeV^3], degeneracy g."""
    return (6.0 * _PI2 * n / g) ** (1.0 / 3.0)


def number_density_t0(kF, g):
    """n [MeV^3]."""
    return g * kF ** 3 / (6.0 * _PI2)


def scalar_density_t0(kF, ms, g):
    """n_s [MeV^3]."""
    if kF <= 0.0:
        return 0.0
    EF = np.sqrt(kF ** 2 + ms ** 2)
    return (g * ms / (4.0 * _PI2)) * (kF * EF - ms ** 2 * np.log((kF + EF) / ms))


def eps_kin_t0(kF, ms, g):
    """Kinetic energy density [MeV^4]."""
    if kF <= 0.0:
        return 0.0
    EF = np.sqrt(kF ** 2 + ms ** 2)
    return (g / (16.0 * _PI2)) * (kF * EF * (2.0 * kF ** 2 + ms ** 2)
                                  - ms ** 4 * np.log((kF + EF) / ms))


def P_kin_t0(kF, ms, g):
    """Kinetic pressure [MeV^4]."""
    if kF <= 0.0:
        return 0.0
    EF = np.sqrt(kF ** 2 + ms ** 2)
    return (g / (48.0 * _PI2)) * (kF * EF * (2.0 * kF ** 2 - 3.0 * ms ** 2)
                                  + 3.0 * ms ** 4 * np.log((kF + EF) / ms))


def kinetic_thermo(mu_eff, m, g, T=0.0):
    """Full kinetic thermodynamics of one fermion species.

    mu_eff is the effective (kinetic) potential mu - Sigma0 [MeV], m the
    effective mass [MeV], g the degeneracy, T the temperature [MeV].

    Returns (n, P, eps, s, n_s) in natural units. T = 0 uses the exact closed
    forms; T > 0 comes from the Johns-Ellis-Lattimer integrals in
    `eos.general.fermi_integrals`, converted from their fm-based units here so
    that no call site has to.
    """
    if T == 0.0:
        kF2 = mu_eff * mu_eff - m * m
        if kF2 <= 0.0 or mu_eff <= 0.0:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        kF = np.sqrt(kF2)
        if m == 0.0:
            # Massless (neutrinos): the ms^4 log terms are 0*inf, so use the
            # ultra-relativistic closed forms (E = k, eps = g kF^4/8pi^2).
            eps = g * kF ** 4 / (8.0 * _PI2)
            return number_density_t0(kF, g), eps / 3.0, eps, 0.0, 0.0
        return (number_density_t0(kF, g), P_kin_t0(kF, m, g),
                eps_kin_t0(kF, m, g), 0.0, scalar_density_t0(kF, m, g))
    from eos.general.fermi_integrals import solve_fermi_jel
    n, P, e, s, ns = solve_fermi_jel(mu_eff, T, m, g)
    return n * hc3, P * hc3, e * hc3, s * hc3, ns * hc3


# =============================================================================
# THE MEAN FIELDS
# =============================================================================

def vector_fields(par, Gw, Gr, nB_nat, n3_nat):
    """omega_0 and rho_0 [MeV] from the algebraic vector field equations.

    n3_nat = sum_i t3_i n_i is the tau_3 = +/-1 weighted isovector density.
    Only the nucleon-only case can eliminate them this way; with hyperons the
    sources differ per species and the fields are carried as unknowns.
    """
    omega0 = Gw * nB_nat / par.m_omega ** 2
    rho0 = Gr * n3_nat / par.m_rho ** 2
    return omega0, rho0


def rearrangement(dGs, dGw, dGr, sigma, omega0, rho0, nB_nat, n3_nat, ns_nat):
    """Rearrangement self-energy Sigma^R [MeV], identical for all baryons.

    It enters mu and P, never eps -- the thermodynamic-consistency statement of
    CLAUDE.md section 8, and the reason the Euler check is worth running.
    """
    return dGw * omega0 * nB_nat + dGr * rho0 * n3_nat - dGs * sigma * ns_nat


def _field_sources(ctx, kin, phi0):
    """The meson-field sources sum_i x_i Gamma_i (n_i or n_s,i) [MeV^3].

    Returns (src_sigma, src_omega, src_rho, src_phi, n_total): what the field
    equations sigma = src_sigma/m_sigma^2 and so on are solved against.
    """
    src_s = src_w = src_r = src_phi = n_tot = 0.0
    for _name, spec, _mu_eff, _ms, n, ns, _eps, _P, _s in kin:
        _mass, _Q, t3, _g, xs, xw, xr, xphi, _S = spec
        src_s += xs * ctx.Gs_N * ns
        src_w += xw * ctx.Gw_N * n
        src_r += xr * ctx.Gr_N * t3 * n
        src_phi += xphi * ctx.Gw_N * n          # Gamma_phiY = x_phi Gamma_omegaN
        n_tot += n
    return src_s, src_w, src_r, src_phi, n_tot


def field_eps_P(par, sigma, omega0, rho0, phi0=0.0):
    """Meson mean-field contributions (eps_field, P_field) [MeV^4].

    The scalar enters P with a minus sign, the vectors with a plus.
    """
    s2 = par.m_sigma ** 2 * sigma ** 2
    w2 = par.m_omega ** 2 * omega0 ** 2
    r2 = par.m_rho ** 2 * rho0 ** 2
    p2 = par.m_phi ** 2 * phi0 ** 2
    return 0.5 * (s2 + w2 + r2 + p2), 0.5 * (-s2 + w2 + r2 + p2)


# =============================================================================
# THE THERMAL MESON GAS: DD2's couplings for it
# =============================================================================
# The gas itself -- species list, quantum numbers, Bose sums -- is
# `eos.general.thermal_mesons`, one implementation for every model (Lavagno,
# Phys. Rev. C 81, 044909 (2010); see also arXiv:1210.0400). What is DD2's is
# the arithmetic of the three effective potentials, using its DENSITY-DEPENDENT
# couplings in place of constant ones. No rearrangement term enters any mu*_j:
# the gas is a spectator to Sigma^R.

def lambda_omega_ratio(par):
    """x_omega^Lambda entering the kaon omega-shift (SU(6) 2/3 by default)."""
    return (par.hyperon_coupling_map.get("Lambda", (0, 0, _X_OMEGA_LAMBDA))[2]
            if par.hyperon_couplings else _X_OMEGA_LAMBDA)


def meson_potentials(Gw, Gr, x_omega_L, mu_C, mu_S, omega0, rho0):
    """(mu*_pi+, mu*_K+, mu*_K0) [MeV] from DD2's couplings and fields.

        mu*_pi+ = mu_C - Gamma_rhoN rho0
        mu*_K+  = mu_C - mu_S - (Gamma_omegaN - Gamma_omegaL) omega0
                              - 1/2 Gamma_rhoN rho0
        mu*_K0  =      - mu_S - (Gamma_omegaN - Gamma_omegaL) omega0
                              + 1/2 Gamma_rhoN rho0

    The kaon's omega shift is (Gamma_omegaN - Gamma_omegaLambda): under the
    additive quark picture the kaon couples through its one light quark, and
    the Lambda ratio supplies the strange-sector piece.
    """
    dGw_KL = (1.0 - x_omega_L) * Gw
    mu_pi = mu_C - Gr * rho0
    mu_Kp = mu_C - mu_S - dGw_KL * omega0 - 0.5 * Gr * rho0
    mu_K0 = -mu_S - dGw_KL * omega0 + 0.5 * Gr * rho0
    return mu_pi, mu_Kp, mu_K0


def meson_families(Gw, Gr, x_omega_L, mu_C, mu_S, omega0, rho0,
                   include_pseudoscalars=False, include_thermal_vectors=False):
    """(mu_eff, mass, Q, S, g) per thermal meson species at DD2's potentials."""
    mu_pi, mu_Kp, mu_K0 = meson_potentials(Gw, Gr, x_omega_L, mu_C, mu_S,
                                           omega0, rho0)
    return _gas.meson_families(mu_pi, mu_Kp, mu_K0,
                               include_pseudoscalars, include_thermal_vectors)


def thermal_meson_charges(Gw, Gr, x_omega_L, mu_C, mu_S, omega0, rho0, T,
                          include_pseudoscalars=False,
                          include_thermal_vectors=False):
    """(n_C, n_S) of the gas [fm^-3].

    Zero unless a thermal-meson flag is on and T > 0, so every caller can add
    it unconditionally.
    """
    if T <= 0.0 or not (include_pseudoscalars or include_thermal_vectors):
        return 0.0, 0.0
    mu_pi, mu_Kp, mu_K0 = meson_potentials(Gw, Gr, x_omega_L, mu_C, mu_S,
                                           omega0, rho0)
    return _gas.thermal_meson_charges(mu_pi, mu_Kp, mu_K0, T,
                                      include_pseudoscalars,
                                      include_thermal_vectors)


def thermal_meson_thermo(par, n_B, mu_C, mu_S, omega0, rho0, T,
                         include_pseudoscalars=False,
                         include_thermal_vectors=False):
    """Full gas thermodynamics at (n_B [fm^-3], T [MeV]) on DD2's mean field.

    Returns the dict of `eos.general.thermal_mesons.thermal_meson_thermo` in
    fm-based units, with the couplings evaluated at this n_B.
    """
    if T <= 0.0 or not (include_pseudoscalars or include_thermal_vectors):
        return dict(P=0.0, e=0.0, s=0.0, n_C=0.0, n_S=0.0, mu_dot_n=0.0)
    _, Gw, Gr, _, _, _ = par.couplings_at(n_B)
    mu_pi, mu_Kp, mu_K0 = meson_potentials(Gw, Gr, lambda_omega_ratio(par),
                                           mu_C, mu_S, omega0, rho0)
    return _gas.thermal_meson_thermo(mu_pi, mu_Kp, mu_K0, T,
                                     include_pseudoscalars,
                                     include_thermal_vectors)


# =============================================================================
# THE ACTIVE SPECIES AND THEIR COUPLINGS
# =============================================================================
# Each baryon spec is (mass, Q, t3, g, x_sigma, x_omega, x_rho, x_phi, S).

def build_baryon_specs(par, flags):
    """The per-species mass, quantum numbers and coupling ratios.

    x_i = Gamma_i/Gamma_N inherit the nucleon density dependence f_i(x);
    x_phi = g_phiY/g_omegaN also inherits f_omega, so
    Gamma_phiY(n_B) = x_phi Gamma_omegaN(n_B). Hyperon masses are the DD2Y
    values from the coupling map, nucleons use the kernel mass, Delta the PDG
    mass.
    """
    m_kn, m_kp = par.kernel_masses()
    hyp = par.hyperon_coupling_map
    specs = []
    for b in active_baryons(flags):
        if b.name == "n":
            mass, xs, xw, xr, xphi = m_kn, 1.0, 1.0, 1.0, 0.0
        elif b.name == "p":
            mass, xs, xw, xr, xphi = m_kp, 1.0, 1.0, 1.0, 0.0
        elif b.name.startswith("Delta"):
            mass = b.mass
            xs, xw, xr, xphi = (par.x_Delta_sigma, par.x_Delta_omega,
                                par.x_Delta_rho, 0.0)
        else:
            mass, xs, xw, xr, xphi = hyp[b.name]
        if b.t3 is None:
            raise ValueError(f"baryon {b.name} has no t3 set (needed for rho)")
        specs.append((mass, b.charge, b.t3, b.g_degen, xs, xw, xr, xphi,
                      b.strangeness))
    return tuple(specs)


@dataclass
class MatterCtx:
    """The model at one density and temperature, before any mode is chosen.

    Everything here is a property of the matter: which species exist, their
    couplings at the target density, the meson masses, the temperature. No
    equilibrium condition and no held fraction -- those live on the mode
    declaration that `solver.py` carries.

    Mutable rather than frozen because a solve that carries the phase density
    as an unknown re-evaluates the couplings at each trial density; nothing
    else writes to it.
    """
    baryons: tuple                 # active Particle objects, order matches specs
    specs: tuple
    nB_nat: float                  # target baryon density [MeV^3]
    mbar: float                    # residual scaling mass [MeV]
    m_sigma2: float
    m_omega2: float
    m_rho2: float
    m_phi2: float
    Gs_N: float                    # nucleon Gamma_sigma(n_B target)
    Gw_N: float
    Gr_N: float
    dGs_N: float                   # dGamma/dn_B [MeV^-3]
    dGw_N: float
    dGr_N: float
    m_e: float
    m_mu: float
    T: float
    include_muons: bool
    has_phi: bool
    include_pseudoscalars: bool = False
    include_thermal_vectors: bool = False
    x_omega_L: float = _X_OMEGA_LAMBDA


def build_matter_ctx(par, n_B, flags, T=0.0):
    """The context for matter at (n_B [fm^-3], T [MeV]) with these species."""
    Gs, Gw, Gr, dGs, dGw, dGr = par.couplings_at(n_B)
    return MatterCtx(
        baryons=tuple(active_baryons(flags)),
        specs=build_baryon_specs(par, flags),
        nB_nat=n_B * hc3, mbar=par.m_nucleon,
        m_sigma2=par.m_sigma ** 2, m_omega2=par.m_omega ** 2,
        m_rho2=par.m_rho ** 2, m_phi2=par.m_phi ** 2,
        Gs_N=Gs, Gw_N=Gw, Gr_N=Gr, dGs_N=dGs, dGw_N=dGw, dGr_N=dGr,
        m_e=Electron.mass, m_mu=Muon.mass, T=T,
        include_muons=flags.muons,
        has_phi=flags.phi_field and flags.hyperons,
        include_pseudoscalars=flags.include_pseudoscalars,
        include_thermal_vectors=flags.include_thermal_vectors,
        x_omega_L=lambda_omega_ratio(par),
    )


def effective_masses(ctx, sigma):
    """m*_i = m_i - x_sigma,i Gamma_sigmaN sigma, per species [MeV].

    Returns None if any effective mass is non-positive: that is outside the
    physical domain, and a caller stepping there wants to back off rather than
    take the square root of a negative number.
    """
    out = {}
    for spec, b in zip(ctx.specs, ctx.baryons):
        mass, _Q, _t3, _g, xs, _xw, _xr, _xphi, _S = spec
        ms = mass - xs * ctx.Gs_N * sigma
        if ms <= 0.0:
            return None
        out[b.name] = ms
    return out


def effective_potentials(ctx, mu_tilde_B, mu_C, mu_S, omega0, rho0, phi0=0.0):
    """mu_eff_i = mu_i - Sigma0_i, per species [MeV].

    mu_tilde_B = mu_B - Sigma^R is the KINETIC baryon potential: carrying it
    instead of mu_B keeps the rearrangement term and its density circularity
    out of the iteration, and the effective potentials vary smoothly along a
    density sweep, which is what makes warm starts work (CLAUDE.md section 2).
    """
    out = {}
    for spec, b in zip(ctx.specs, ctx.baryons):
        _mass, Q, t3, _g, _xs, xw, xr, xphi, S = spec
        Gw, Gr = xw * ctx.Gw_N, xr * ctx.Gr_N
        Gphi = xphi * ctx.Gw_N          # phi inherits f_omega (DD2Y)
        out[b.name] = (mu_tilde_B + Q * mu_C + S * mu_S
                       - Gw * omega0 - Gr * t3 * rho0 - Gphi * phi0)
    return out


def baryon_kinetics(ctx, sigma, omega0, rho0, phi0, mu_tilde_B, mu_C, mu_S):
    """Per-species (mu_eff, m*, n, n_s, eps, P, s) at the current fields.

    Returns None outside the physical domain (any m* <= 0), which is how a
    trial point that has wandered out of it is reported.
    """
    m_eff_i = effective_masses(ctx, sigma)
    if m_eff_i is None:
        return None
    mu_eff_i = effective_potentials(ctx, mu_tilde_B, mu_C, mu_S,
                                    omega0, rho0, phi0)
    out = []
    for spec, b in zip(ctx.specs, ctx.baryons):
        _mass, _Q, _t3, g, _xs, _xw, _xr, _xphi, _S = spec
        ms, mu_eff = m_eff_i[b.name], mu_eff_i[b.name]
        n, P, eps, s, ns = kinetic_thermo(mu_eff, ms, g, ctx.T)
        if not np.isfinite([n, P, eps, s, ns]).all():
            # The finite-T integrals can return NaN for a wild trial state.
            # Report it as out of domain rather than letting it propagate: a
            # NaN residual gives the solver nothing to back off along.
            return None
        out.append((b.name, spec, mu_eff, ms, n, ns, eps, P, s))
    return out


def meson_charges_nat(ctx, mu_C, mu_S, omega0, rho0):
    """(n_C, n_S) of the thermal meson gas in NATURAL units [MeV^3]."""
    n_C, n_S = thermal_meson_charges(
        ctx.Gw_N, ctx.Gr_N, ctx.x_omega_L, mu_C, mu_S, omega0, rho0, ctx.T,
        include_pseudoscalars=ctx.include_pseudoscalars,
        include_thermal_vectors=ctx.include_thermal_vectors)
    return n_C * hc3, n_S * hc3


def neutralizing_leptons(target_nat, m_e, m_mu, include_muons, T):
    """Leptons at the potential that neutralises a given charge density.

    Populates e (and mu, if enabled) so that n_e + n_mu = `target_nat`, the
    non-leptonic charge in natural units. Muons are in leptonic equilibrium at
    mu_mu = mu_e, so a single potential closes it. Since leptons do not source
    the mean fields this is exact after the fact.

    Model-independent: only masses and the temperature enter, so this belongs
    in `eos.general` and is recorded in docs/DEFERRED.md as owing that move.

    Returns (mu_e, (n,P,eps,s)_e, (n,P,eps,s)_mu), all natural units.
    """
    zero = (0.0, 0.0, 0.0, 0.0)
    if target_nat <= 0.0:                   # no positive charge to neutralize
        return 0.0, zero, zero

    def f(mu):
        ne = kinetic_thermo(mu, m_e, 2.0, T)[0]
        nmu = kinetic_thermo(mu, m_mu, 2.0, T)[0] if include_muons else 0.0
        return ne + nmu - target_nat

    hi = 200.0
    while f(hi) < 0.0:
        hi *= 2.0
    mu_e = brentq(f, 0.0, hi, xtol=1e-10)
    ne, Pe, ee, se, _ = kinetic_thermo(mu_e, m_e, 2.0, T)
    if include_muons:
        nmu, Pmu, emu, smu, _ = kinetic_thermo(mu_e, m_mu, 2.0, T)
        return mu_e, (ne, Pe, ee, se), (nmu, Pmu, emu, smu)
    return mu_e, (ne, Pe, ee, se), zero


# =============================================================================
# ASSEMBLY: the state, matter only
# =============================================================================

def assemble(par, ctx, sigma, omega0, rho0, phi0, mu_tilde_B, mu_C, mu_S):
    """The matter state at these fields and potentials, as a `PhaseThermo`.

    MATTER ONLY -- baryons plus any thermal meson gas, no leptons and no
    photons. Those are shared by the whole system and, in a mixed phase, may
    be distributed differently from the matter, so `solver.py` and
    `eos.mixed` add them.

    The gas is listed in `densities` alongside the baryons, which is what makes
    the returned n_C and n_S the TOTAL non-leptonic charge and strangeness
    that neutrality and the fixed-Y_C / fixed-Y_S conditions are stated in
    terms of (CLAUDE.md section 2). At T = 40 MeV a pion gas carries about 15
    percent of the charge, so counting it is not a refinement.

    Returns None outside the physical domain, matching `baryon_kinetics`.
    """
    kin = baryon_kinetics(ctx, sigma, omega0, rho0, phi0,
                          mu_tilde_B, mu_C, mu_S)
    if kin is None:
        return None

    densities, mu_eff_i, m_eff_i = {}, {}, {}
    eps_b = P_b = s_b = Sig_R = 0.0
    n_tot = charge_had = strangeness_had = 0.0
    for name, spec, mu_eff, ms, n, ns, eps, P, s in kin:
        _mass, Q, t3, _g, xs, xw, xr, xphi, S = spec
        densities[name] = n / hc3
        mu_eff_i[name] = mu_eff
        m_eff_i[name] = ms
        eps_b += eps
        P_b += P
        s_b += s
        n_tot += n
        charge_had += Q * n
        strangeness_had += S * n
        # Rearrangement; phi inherits f_omega, so dGamma_phiY/dn = x_phi dGw_N.
        Sig_R += (xw * ctx.dGw_N * omega0 * n
                  + xr * ctx.dGr_N * rho0 * t3 * n
                  + xphi * ctx.dGw_N * phi0 * n
                  - xs * ctx.dGs_N * sigma * ns)

    eps_fields, P_fields = field_eps_P(par, sigma, omega0, rho0, phi0)

    # The thermal meson gas. It carries charge and strangeness, so it enters
    # n_C and n_S -- through `extra_charges`, because most of its members are
    # not yet in the particle table and so cannot be summed as species.
    gas_C, gas_S = meson_charges_nat(ctx, mu_C, mu_S, omega0, rho0)
    gas = thermal_meson_thermo(
        par, ctx.nB_nat / hc3, mu_C, mu_S, omega0, rho0, ctx.T,
        include_pseudoscalars=ctx.include_pseudoscalars,
        include_thermal_vectors=ctx.include_thermal_vectors)

    mu_B = mu_tilde_B + Sig_R
    # sum_i mu_i n_i over the baryons, plus the gas at its EFFECTIVE
    # potentials: the field shift is already carried by the vector terms.
    mu_dot_n = ((mu_B * n_tot + mu_C * charge_had + mu_S * strangeness_had)
                / hc3 + gas["mu_dot_n"])

    return PhaseThermo.assemble(
        T=ctx.T, mu_B=mu_B, mu_C=mu_C, mu_S=mu_S,
        fields={"sigma": sigma, "omega0": omega0, "rho0": rho0, "phi0": phi0},
        densities=densities, mu_eff_i=mu_eff_i, m_eff_i=m_eff_i,
        P=(P_b + P_fields + ctx.nB_nat * Sig_R) / hc3 + gas["P"],
        eps=(eps_b + eps_fields) / hc3 + gas["e"],
        s=s_b / hc3 + gas["s"],
        mu_dot_n=mu_dot_n,
        Sigma_R=Sig_R,
        extra_charges=(0.0, gas_C / hc3, gas_S / hc3),
    )


# =============================================================================
# SELF-CONSISTENCY: DD2's own fields at given charge potentials
# =============================================================================

#: Residual gate for the phase-internal solve, matching what the equilibrium
#: solvers accept.
RESIDUAL_TOL = 1.0e-10


def self_consistency_residual(x, par, ctx, mu_tilde_B, mu_C, mu_S, fast=True):
    """Meson field gaps plus baryon-density self-consistency.

    Unknowns x = [sigma, omega0, rho0, (phi0), nB_nat]. The density is an
    unknown here -- unlike an ordinary DD2 solve, where it is given -- because
    for one phase of a mixture only the AVERAGE density is prescribed. DD2's
    couplings are density-dependent, so they are re-evaluated at the current
    nB_nat on every iteration.

    There is no charge, strangeness or neutrality row: the potentials are
    inputs. That is what makes this thermodynamics rather than a mode.

    `fast` selects the backend (CLAUDE.md section 9). At T = 0 the per-species
    loop runs in the jitted `meson_sources_t0` kernel, the same closed form the
    NumPy path evaluates and about four times quicker -- which matters because
    a mixed-phase solve calls this once per residual evaluation. `fast=False`
    is the plain NumPy reference, and the two agree to machine precision.
    """
    sigma, omega0, rho0 = x[0], x[1], x[2]
    i = 3
    phi0 = x[i] if ctx.has_phi else 0.0
    i += int(ctx.has_phi)
    nB_nat = x[i]
    if nB_nat <= 0.0:
        return [1.0e6] * len(x)

    Gs, Gw, Gr, dGs, dGw, dGr = par.couplings_at(nB_nat / hc3)
    ctx.Gs_N, ctx.Gw_N, ctx.Gr_N = Gs, Gw, Gr
    ctx.dGs_N, ctx.dGw_N, ctx.dGr_N = dGs, dGw, dGr
    ctx.nB_nat = nB_nat

    if fast and ctx.T == 0.0 and _NUMBA_OK:
        spec_arr = getattr(ctx, "_spec_arr", None)
        if spec_arr is None:                     # built once per solve
            spec_arr = np.asarray(ctx.specs, dtype=np.float64)
            ctx._spec_arr = spec_arr
        src_s, src_w, src_r, src_phi, n_tot = meson_sources_t0(
            spec_arr, sigma, omega0, rho0, phi0, mu_tilde_B, mu_C, mu_S,
            Gs, Gw, Gr)
        if n_tot < 0.0:                          # m* <= 0: outside the domain
            return [1.0e6] * len(x)
    else:
        kin = baryon_kinetics(ctx, sigma, omega0, rho0, phi0,
                              mu_tilde_B, mu_C, mu_S)
        if kin is None:                          # m* <= 0: outside the domain
            return [1.0e6] * len(x)
        src_s, src_w, src_r, src_phi, n_tot = _field_sources(ctx, kin, phi0)
    if not np.isfinite([src_s, src_w, src_r, src_phi, n_tot]).all():
        # A non-finite residual is worse than a large one: the solver cannot
        # back off from NaN, it just stops. Report the same penalty the
        # out-of-domain branch does, so a bad trial point is recoverable.
        return [1.0e6] * len(x)

    res = [
        (sigma - src_s / ctx.m_sigma2) / ctx.mbar,
        (omega0 - src_w / ctx.m_omega2) / ctx.mbar,
        (rho0 - src_r / ctx.m_rho2) / ctx.mbar,
    ]
    if ctx.has_phi:
        res.append((phi0 - src_phi / ctx.m_phi2) / ctx.mbar)
    res.append(n_tot / nB_nat - 1.0)             # density self-consistency
    return res


def thermo_at_potentials(par, flags, mu_tilde_B, mu_C, mu_S=0.0, T=0.0,
                         n_B_guess=0.2, x0=None, x0_fallback=None,
                         return_state=False, fast=True):
    """DD2's matter at fixed conserved-charge potentials, fields solved.

    The self-consistent layer: given (mu_tilde_B, mu_C, mu_S, T) it solves the
    meson fields AND the phase's own baryon density, then assembles. No
    leptons, no neutrality, no held fraction -- those are conditions on a
    system, and this describes matter.

    `mu_tilde_B = mu_B - Sigma^R` is the KINETIC baryon potential: Sigma^R is
    a function of the density, which is itself an unknown here, so carrying
    mu_B would put that circularity into the iteration. The physical mu_B is
    restored at assembly.

    `x0` is the starting configuration [sigma, omega0, rho0, (phi0), nB_nat].
    It must be a DETERMINISTIC function of the caller's inputs: a mixed-phase
    residual is differentiated numerically, so seeding from the previous trial
    point would make this function's output depend on the path taken to reach
    it and corrupt the Jacobian.

    Returns a `PhaseThermo`, or (PhaseThermo, {x_phase, ctx}) with
    `return_state`, so a caller can feed the converged vector back as the next
    `x0` or differentiate the phase without re-solving it.
    """
    from scipy.optimize import root

    ctx = build_matter_ctx(par, n_B_guess, flags, T=T)
    args = (par, ctx, mu_tilde_B, mu_C, mu_S, fast)

    # Try the caller's warm start, then whatever stronger seed it offered, then
    # a seed built here from the field equations.
    # The fallback is what makes a warm-started sweep robust: a previous
    # point's vector is an excellent guess right up until the composition
    # changes under it -- a hyperon threshold, or a meson gas turning on --
    # and then it is a bad one, in a region where the residual answers with a
    # constant penalty and the solver has no gradient to follow. Without the
    # retry those points are simply lost.
    def guesses():
        if x0 is not None:
            yield list(x0)
        if x0_fallback is not None:
            # Callable, so a caller whose fallback is expensive (eos/mixed
            # builds it from a full beta-equilibrium solve) pays for it only
            # when the warm start actually misses.
            yield list(x0_fallback() if callable(x0_fallback) else x0_fallback)
        yield _cold_start(par, ctx, mu_tilde_B, mu_C, mu_S)

    sol = None
    for guess in guesses():
        sol = root(self_consistency_residual, guess, args=args, method="hybr",
                   tol=1e-13)
        res_max = max(abs(r) for r in self_consistency_residual(sol.x, *args))
        if res_max <= RESIDUAL_TOL:
            break
    if res_max > RESIDUAL_TOL:
        raise RuntimeError(
            f"DD2 self-consistency failed at mu_tilde_B={mu_tilde_B}, "
            f"mu_C={mu_C}, mu_S={mu_S}, T={T}: {sol.message} "
            f"(max residual {res_max:.2e}, tol {RESIDUAL_TOL:.0e})")

    sigma, omega0, rho0 = sol.x[0], sol.x[1], sol.x[2]
    i = 3
    phi0 = sol.x[i] if ctx.has_phi else 0.0
    # `_self_consistency_residual` left ctx's couplings and density at the
    # converged values on its last call, so ctx is already the assembly context.
    state = assemble(par, ctx, sigma, omega0, rho0, phi0,
                     mu_tilde_B, mu_C, mu_S)
    if return_state:
        # Shaped for a caller differentiating this phase without re-solving it:
        # `eos.mixed`'s analytic Jacobian re-runs the residual on this same ctx.
        return state, dict(x_phase=list(sol.x), ctx=ctx)
    return state


#: Damping and sweep count for the Picard seed below. Enough to land inside
#: the Newton basin, not enough to be a solver in its own right.
_SEED_SWEEPS = 40
_SEED_MIXING = 0.5


def _sigma_ceiling(ctx):
    """The largest sigma that keeps every effective mass positive.

    m*_i = m_i - x_sigma,i Gamma_sigmaN sigma vanishes at
    sigma_i = m_i / (x_sigma,i Gamma_sigmaN); the domain ends at the smallest
    of those. A seed sweep that overshoots it puts the lightest species at
    m* <= 0 and the iteration never comes back, which is what a linearised
    estimate does at high density with hyperons active. Bracketing below it is
    the same guard `thermo_at_composition` uses on its scalar gap.
    """
    ceilings = []
    for spec in ctx.specs:
        mass, _Q, _t3, _g, xs, _xw, _xr, _xphi, _S = spec
        scale = xs * ctx.Gs_N
        if scale > 0.0:
            ceilings.append(mass / scale)
    return 0.95 * min(ceilings) if ceilings else float("inf")


def _cold_start(par, ctx, mu_tilde_B, mu_C, mu_S):
    """A starting configuration from the field equations themselves.

    Picard iteration on (sigma, omega0, rho0, phi0, n_B): evaluate the species
    at the current fields, rebuild the fields from their own sources, mix, and
    repeat. Starting from zero fields means m* = m > 0, so the first sweep is
    always inside the domain, and the damping keeps it there.

    This deliberately does NOT seed from an equilibrium solve. Doing so would
    make thermodynamics import the solver -- inverting the layer order of
    CLAUDE.md section 5 and putting a mode inside a module that must not know
    about one. It is also slower: a caller in a hot loop passes `x0` instead,
    and `eos.mixed` does exactly that, computing its seed once per solve.
    """
    sigma = omega0 = rho0 = phi0 = 0.0
    nB_nat = ctx.nB_nat
    for _sweep in range(_SEED_SWEEPS):
        sigma = min(sigma, _sigma_ceiling(ctx))
        Gs, Gw, Gr, dGs, dGw, dGr = par.couplings_at(nB_nat / hc3)
        ctx.Gs_N, ctx.Gw_N, ctx.Gr_N = Gs, Gw, Gr
        ctx.dGs_N, ctx.dGw_N, ctx.dGr_N = dGs, dGw, dGr
        ctx.nB_nat = nB_nat
        kin = baryon_kinetics(ctx, sigma, omega0, rho0, phi0,
                              mu_tilde_B, mu_C, mu_S)
        if kin is None:                      # stepped out; back off and stop
            break
        src_s, src_w, src_r, src_phi, n_tot = _field_sources(ctx, kin, phi0)
        new = (src_s / ctx.m_sigma2, src_w / ctx.m_omega2,
               src_r / ctx.m_rho2, src_phi / ctx.m_phi2)
        sigma += _SEED_MIXING * (new[0] - sigma)
        omega0 += _SEED_MIXING * (new[1] - omega0)
        rho0 += _SEED_MIXING * (new[2] - rho0)
        if ctx.has_phi:
            phi0 += _SEED_MIXING * (new[3] - phi0)
        if n_tot > 0.0:
            nB_nat += _SEED_MIXING * (n_tot - nB_nat)

    x = [sigma, omega0, rho0]
    if ctx.has_phi:
        x.append(phi0 if phi0 != 0.0 else -1.0e-3)
    x.append(max(nB_nat, 1.0e-6 * hc3))
    return x
