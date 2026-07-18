"""
physics/octet.py
====================
General potential-driven mean-field residual for the full baryon octet
(report §1.4, §1.7, §3.3). Generalizes the nucleon beta_eq of residual.py to
Λ, Σ, Ξ with the φ field.

Unknown vector (report §1.7 unified scheme):
    x = [sigma, omega0, rho0, (phi0), mu_tilde_B, mu_Q, (mu_S)]

phi0 is present iff phi_field is on AND hyperons are active. mu_S is present
iff the strangeness sector is fixed (fixed-Y_S mode). All vector fields are
carried as unknowns with their field equations as residuals (their sources
are composition-dependent once the couplings differ per species, so they can
no longer be eliminated at the target density as in the nucleon-only case).

Equilibrium modes (the charge/strange sector switches, report §1.7):
    charge = 'neutral' : beta equilibrium, leptons present, charge-neutral;
             'fixed'   : hadronic charge fraction fixed to Y_C, NO leptons
                         (the CompOSE general-purpose (nB,T,Yq) convention).
    strange= 'eq'      : mu_S = 0 (strangeness-changing weak reactions fast);
             'fixed'   : strangeness fraction fixed to Y_S, mu_S unknown.
Beta equilibrium is charge='neutral', strange='eq'; mu_L / neutrinos (fixed
Y_L) are a later milestone.

mu_tilde_B = mu_B - Sigma^R absorbs the species-independent rearrangement term
(and its density circularity) out of the iteration; the baryon kinetic
potential is nu_i = mu_tilde_B*B_i + mu_Q*Q_i - Gamma_omega_i omega0
- Gamma_rho_i t3_i rho0 - Gamma_phi_i phi0. mu_B = mu_tilde_B + Sigma^R is
restored at assembly.

Per-baryon couplings: Gamma_{sigma,omega,rho}_i = x_i * Gamma_iN(n_B) inherit
the nucleon density dependence; Gamma_phi_i = g_phi_i is a CONSTANT (DD2Y).
Residuals dimensionless: field eqs scaled by m_nucleon, densities by n_B.
"""
from dataclasses import dataclass

from eos.general.particles import Electron, Muon
from eos.general.physics_constants import hc3
from eos.dd2.physics.thermo import kinetic_thermo
from eos.dd2.species import active_baryons

# Each baryon spec: (mass, Q, t3, g, x_sigma, x_omega, x_rho, g_phi, S)
_BaryonSpec = tuple


@dataclass
class OctetCtx:
    baryons: tuple         # active Particle objects, order matches specs
    specs: tuple           # tuple of _BaryonSpec
    nB_nat: float          # target baryon density [MeV^3]
    mbar: float            # residual scaling mass [MeV]
    m_sigma2: float
    m_omega2: float
    m_rho2: float
    m_phi2: float
    Gs_N: float            # nucleon Gamma_sigma(n_B target)
    Gw_N: float
    Gr_N: float
    dGs_N: float           # dGamma/dn_B [MeV^-3]
    dGw_N: float
    dGr_N: float
    m_e: float
    m_mu: float
    T: float
    include_muons: bool
    has_phi: bool
    n_unknowns: int
    charge_mode: str = "neutral"   # 'neutral' (beta) | 'fixed' (Y_C)
    Y_C: float = 0.0               # target hadronic charge fraction (fixed mode)
    strange_mode: str = "eq"       # 'eq' (mu_S=0) | 'fixed' (Y_S)
    Y_S: float = 0.0               # target strangeness fraction (fixed mode)

    @property
    def leptons(self):
        return self.charge_mode == "neutral"

    @property
    def has_muS(self):
        return self.strange_mode == "fixed"


def build_baryon_specs(par, flags):
    """(mass, Q, t3, g, x_sigma, x_omega, x_rho, g_phi) per active baryon."""
    m_kn, m_kp = par.kernel_masses()
    hyp = par.hyperon_coupling_map
    specs = []
    for b in active_baryons(flags):
        if b.name == "n":
            mass, (xs, xw, xr, gphi) = m_kn, (1.0, 1.0, 1.0, 0.0)
        elif b.name == "p":
            mass, (xs, xw, xr, gphi) = m_kp, (1.0, 1.0, 1.0, 0.0)
        elif b.name.startswith("Delta"):
            # Δ: S=0 so no φ; ratio couplings from the parametrization.
            mass = b.mass
            xs, xw, xr, gphi = (par.x_Delta_sigma, par.x_Delta_omega,
                                par.x_Delta_rho, 0.0)
        else:
            mass, (xs, xw, xr, gphi) = b.mass, hyp[b.name]
        if b.t3 is None:
            raise ValueError(f"baryon {b.name} has no t3 set (needed for rho)")
        specs.append((mass, b.charge, b.t3, b.g_degen, xs, xw, xr, gphi,
                      b.strangeness))
    return tuple(specs)


def build_octet_ctx(par, n_B, flags, T=0.0, charge_mode="neutral", Y_C=0.0,
                    strange_mode="eq", Y_S=0.0):
    Gs, Gw, Gr, dGs, dGw, dGr = par.couplings_at(n_B)
    has_phi = flags.phi_field and flags.hyperons
    # sigma,omega,rho,(phi),muB~,muQ,(muS)
    n_unknowns = 5 + int(has_phi) + int(strange_mode == "fixed")
    return OctetCtx(
        baryons=tuple(active_baryons(flags)),
        specs=build_baryon_specs(par, flags),
        nB_nat=n_B * hc3, mbar=par.m_nucleon,
        m_sigma2=par.m_sigma ** 2, m_omega2=par.m_omega ** 2,
        m_rho2=par.m_rho ** 2, m_phi2=par.m_phi ** 2,
        Gs_N=Gs, Gw_N=Gw, Gr_N=Gr, dGs_N=dGs, dGw_N=dGw, dGr_N=dGr,
        m_e=Electron.mass, m_mu=Muon.mass, T=T,
        include_muons=flags.muons, has_phi=has_phi, n_unknowns=n_unknowns,
        charge_mode=charge_mode, Y_C=Y_C, strange_mode=strange_mode, Y_S=Y_S,
    )


def _unpack(x, ctx):
    sigma, omega0, rho0 = x[0], x[1], x[2]
    i = 3
    phi0 = x[i] if ctx.has_phi else 0.0
    i += int(ctx.has_phi)
    mu_tilde_B, mu_Q = x[i], x[i + 1]
    mu_S = x[i + 2] if ctx.has_muS else 0.0
    return sigma, omega0, rho0, phi0, mu_tilde_B, mu_Q, mu_S


def _baryon_kinetics(ctx, sigma, omega0, rho0, phi0, mu_tilde_B, mu_Q, mu_S):
    """
    Per-baryon (nu, m*, n, ns, eps, P, s) at the current fields.
    Returns list of (spec, nu, ms, n, ns, eps, P, s) in natural units, or
    None if any effective mass is non-positive (outside the physical domain).
    """
    out = []
    for spec in ctx.specs:
        mass, Q, t3, g, xs, xw, xr, gphi, S = spec
        Gs, Gw, Gr = xs * ctx.Gs_N, xw * ctx.Gw_N, xr * ctx.Gr_N
        ms = mass - Gs * sigma
        if ms <= 0.0:
            return None
        nu = (mu_tilde_B + Q * mu_Q + S * mu_S - Gw * omega0
              - Gr * t3 * rho0 - gphi * phi0)
        n, P, eps, s, ns = kinetic_thermo(nu, ms, g, ctx.T)
        out.append((spec, nu, ms, n, ns, eps, P, s))
    return out


def octet_residual(x, ctx):
    """Dimensionless residual vector (report §1.7 unified scheme)."""
    sigma, omega0, rho0, phi0, mu_tilde_B, mu_Q, mu_S = _unpack(x, ctx)
    kin = _baryon_kinetics(ctx, sigma, omega0, rho0, phi0,
                           mu_tilde_B, mu_Q, mu_S)
    if kin is None:
        return [1.0e6] * ctx.n_unknowns

    src_s = src_w = src_r = src_phi = 0.0
    n_tot = charge = strangeness = 0.0
    for (spec, nu, ms, n, ns, eps, P, s) in kin:
        mass, Q, t3, g, xs, xw, xr, gphi, S = spec
        src_s += xs * ctx.Gs_N * ns
        src_w += xw * ctx.Gw_N * n
        src_r += xr * ctx.Gr_N * t3 * n
        src_phi += gphi * n
        n_tot += n
        charge += Q * n
        strangeness += S * n

    res = [
        (sigma - src_s / ctx.m_sigma2) / ctx.mbar,
        (omega0 - src_w / ctx.m_omega2) / ctx.mbar,
        (rho0 - src_r / ctx.m_rho2) / ctx.mbar,
    ]
    if ctx.has_phi:
        res.append((phi0 - src_phi / ctx.m_phi2) / ctx.mbar)
    res.append(n_tot / ctx.nB_nat - 1.0)

    if ctx.charge_mode == "neutral":
        mu_e = -mu_Q
        n_e = kinetic_thermo(mu_e, ctx.m_e, 2.0, ctx.T)[0]
        n_mu = (kinetic_thermo(mu_e, ctx.m_mu, 2.0, ctx.T)[0]
                if ctx.include_muons else 0.0)
        res.append((charge - n_e - n_mu) / ctx.nB_nat)     # neutrality
    else:
        res.append((charge / ctx.nB_nat) - ctx.Y_C)        # fixed hadronic Y_C
    if ctx.has_muS:
        res.append((strangeness / ctx.nB_nat) - ctx.Y_S)   # fixed Y_S
    return res


def assemble_octet(x, ctx):
    """
    Full thermodynamic state from a converged unknown vector. Returns a dict
    in natural units plus per-species densities (fm^-3) for onset detection.
    """
    sigma, omega0, rho0, phi0, mu_tilde_B, mu_Q, mu_S = _unpack(x, ctx)
    kin = _baryon_kinetics(ctx, sigma, omega0, rho0, phi0,
                           mu_tilde_B, mu_Q, mu_S)

    eps_b = P_b = s_b = 0.0
    Sig_R = 0.0
    n_tot = charge_had = strangeness = 0.0
    m_eff_n = nu_n = nu_p = None
    densities = {}
    for (spec, nu, ms, n, ns, eps, P, s), b in zip(kin, ctx.baryons):
        mass, Q, t3, g, xs, xw, xr, gphi, S = spec
        eps_b += eps
        P_b += P
        s_b += s
        n_tot += n
        charge_had += Q * n
        strangeness += S * n
        densities[b.name] = n / hc3
        # rearrangement (constant phi coupling -> no phi term)
        Sig_R += (xw * ctx.dGw_N * omega0 * n
                  + xr * ctx.dGr_N * rho0 * t3 * n
                  - xs * ctx.dGs_N * sigma * ns)
        if b.name == "n":
            m_eff_n, nu_n = ms, nu
        elif b.name == "p":
            nu_p = nu

    # meson field energies: scalars minus in P, vectors plus
    s2 = ctx.m_sigma2 * sigma ** 2
    w2 = ctx.m_omega2 * omega0 ** 2
    r2 = ctx.m_rho2 * rho0 ** 2
    p2 = ctx.m_phi2 * phi0 ** 2
    eps_fields = 0.5 * (s2 + w2 + r2 + p2)
    P_fields = 0.5 * (-s2 + w2 + r2 + p2)

    mu_e = -mu_Q
    if ctx.leptons:
        ne, Pe, ee, se, _ = kinetic_thermo(mu_e, ctx.m_e, 2.0, ctx.T)
        if ctx.include_muons:
            nmu, Pmu, emu, smu, _ = kinetic_thermo(mu_e, ctx.m_mu, 2.0, ctx.T)
        else:
            nmu = Pmu = emu = smu = 0.0
    else:
        ne = Pe = ee = se = nmu = Pmu = emu = smu = 0.0

    mu_B = mu_tilde_B + Sig_R
    eps = eps_b + eps_fields + ee + emu
    P = P_b + P_fields + ctx.nB_nat * Sig_R + Pe + Pmu
    s = s_b + se + smu

    # baryon chemical-potential sum: mu_i = mu_B + Q_i mu_Q + S_i mu_S
    mu_dot_n = mu_B * n_tot + mu_Q * charge_had + mu_S * strangeness
    lepton_dot_n = mu_e * (ne + nmu)

    return dict(
        sigma=sigma, omega0=omega0, rho0=rho0, phi0=phi0,
        m_eff_n=m_eff_n, nu_n=nu_n, nu_p=nu_p,
        Sigma_R=Sig_R, mu_B=mu_B, mu_Q=mu_Q, mu_S=mu_S, mu_e=mu_e,
        mu_n=mu_B, mu_p=mu_B + mu_Q,
        eps=eps, P=P, s=s, n_tot=n_tot,
        n_e=ne, n_mu=nmu,
        Y_C=charge_had / ctx.nB_nat, Y_S=strangeness / ctx.nB_nat,
        densities=densities,
        mu_dot_n=mu_dot_n + lepton_dot_n,
    )
