"""
physics/octet.py
====================
General potential-driven mean-field residual for the full baryon octet
. Generalizes the nucleon beta_eq of residual.py to
Λ, Σ, Ξ with the φ field.

Unknown vector (the unified charge/strangeness scheme):
    x = [sigma, omega0, rho0, (phi0), mu_tilde_B, mu_C, (mu_S)]

phi0 is present iff phi_field is on AND hyperons are active. mu_S is present
iff the strangeness sector is fixed (fixed-Y_S mode). All vector fields are
carried as unknowns with their field equations as residuals (their sources
are composition-dependent once the couplings differ per species, so they can
no longer be eliminated at the target density as in the nucleon-only case).

Equilibrium modes (the charge/strange sector switches):
    charge = 'neutral' : beta equilibrium, leptons present, charge-neutral;
             'fixed'   : hadronic charge fraction fixed to Y_C, NO leptons
                         (the CompOSE general-purpose (nB,T,Yq) convention).
    strange= 'eq'      : mu_S = 0 (strangeness-changing weak reactions fast);
             'fixed'   : strangeness fraction fixed to Y_S, mu_S unknown.
Beta equilibrium is charge='neutral', strange='eq'; mu_L / neutrinos (fixed
Y_L) are handled by the trapped lepton mode.

mu_tilde_B = mu_B - Sigma^R absorbs the species-independent rearrangement term
(and its density circularity) out of the iteration; the baryon effective
potential is mu_eff_i = mu_tilde_B*B_i + mu_C*Q_i - Gamma_omega_i omega0
- Gamma_rho_i t3_i rho0 - Gamma_phi_i phi0. mu_B = mu_tilde_B + Sigma^R is
restored at assembly.

Per-baryon couplings: Gamma_{sigma,omega,rho}_i = x_i * Gamma_iN(n_B) inherit
the nucleon density dependence; Gamma_phi_i = g_phi_i is a CONSTANT (DD2Y).
Residuals dimensionless: field eqs scaled by m_nucleon, densities by n_B.
"""
from dataclasses import dataclass

from scipy.optimize import brentq

from eos.general.particles import Electron, Muon
from eos.general.physics_constants import hc3
from eos.dd2.thermodynamics import kinetic_thermo
from eos.dd2.thermodynamics import lambda_omega_ratio, thermal_meson_charges
from eos.dd2.species import active_baryons

# Each baryon spec: (mass, Q, t3, g, x_sigma, x_omega, x_rho, x_phi, S)
_BaryonSpec = tuple

#: Neutrino degeneracy (one helicity; antineutrino handled by kinetic_thermo).
_G_NU = 1.0


def _yc_neutralizing_leptons(target_nat, m_e, m_mu, include_muons, T):
    """
    Fixed-Y_C flavor 2b: populate leptons so the total system is
    charge-neutral, n_e(+n_mu) = target (= hadronic charge, natural units).
    Muons (if on) are in leptonic equilibrium mu_mu = mu_e, so a single common
    mu closes n_e(mu) [+ n_mu(mu)] = target. Since leptons don't source the
    mean fields this is exact post-hoc (the hadronic solve is flavor 2a).

    Returns (mu_e, (n,P,eps,s)_e, (n,P,eps,s)_mu), all natural units.
    """
    zero = (0.0, 0.0, 0.0, 0.0)
    if target_nat <= 0.0:                       # no positive charge to neutralize
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
    lepton_mode: str = "transparent"  # 'transparent' (mu_L=0) | 'trapped' (Y_L)
    Y_L: float = 0.0               # target electron lepton fraction (trapped)
    yc_leptons: bool = False       # fixed-Y_C: 2a leptonless (F) | 2b neutralizing (T)
    # Thermal meson gas: contributes n_C and n_S to the charge/strangeness
    # constraints (never to the field sources or to n_B). x_omega_L is the
    # Lambda/nucleon omega ratio entering the kaon shift.
    include_pseudoscalars: bool = False
    include_thermal_vectors: bool = False
    x_omega_L: float = 2.0 / 3.0

    @property
    def leptons(self):
        return self.charge_mode == "neutral"

    @property
    def has_muS(self):
        return self.strange_mode == "fixed"

    @property
    def has_muL(self):
        return self.lepton_mode == "trapped"


def build_baryon_specs(par, flags):
    """
    (mass, Q, t3, g, x_sigma, x_omega, x_rho, x_phi, S) per active baryon.
    x_phi = g_phiY/g_omegaN inherits f_omega, so Γ_phiY = x_phi·Γ_omegaN(n_B)
    (DD2Y density-dependent φ). Hyperon masses are the DD2Y values from the
    coupling map; nucleons use the kernel mass, Δ the PDG mass.
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
            # Δ: S=0 so no φ; ratio couplings from the parametrization.
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


def build_octet_ctx(par, n_B, flags, T=0.0, charge_mode="neutral", Y_C=0.0,
                    strange_mode="eq", Y_S=0.0, lepton_mode="transparent",
                    Y_L=0.0, yc_leptons=False):
    if lepton_mode == "trapped" and charge_mode != "neutral":
        raise ValueError("trapped lepton_mode requires charge_mode='neutral' "
                         "(Y_L trapping implies leptons present, charge-neutral)")
    if yc_leptons and charge_mode != "fixed":
        raise ValueError("yc_leptons (flavor 2b) requires charge_mode='fixed'")
    Gs, Gw, Gr, dGs, dGw, dGr = par.couplings_at(n_B)
    has_phi = flags.phi_field and flags.hyperons
    # sigma,omega,rho,(phi),muB~,muC,(muS),(muL)
    n_unknowns = (5 + int(has_phi) + int(strange_mode == "fixed")
                  + int(lepton_mode == "trapped"))
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
        lepton_mode=lepton_mode, Y_L=Y_L, yc_leptons=yc_leptons,
        include_pseudoscalars=flags.include_pseudoscalars,
        include_thermal_vectors=flags.include_thermal_vectors,
        x_omega_L=lambda_omega_ratio(par),
    )


def meson_charges_nat(ctx, mu_C, mu_S, omega0, rho0):
    """(n_C, n_S) of the thermal meson gas in NATURAL units [MeV^3].

    Zero unless a thermal-meson flag is on and T > 0, so every caller can
    add it unconditionally.
    """
    n_C, n_S = thermal_meson_charges(
        ctx.Gw_N, ctx.Gr_N, ctx.x_omega_L, mu_C, mu_S, omega0, rho0, ctx.T,
        include_pseudoscalars=ctx.include_pseudoscalars,
        include_thermal_vectors=ctx.include_thermal_vectors)
    return n_C * hc3, n_S * hc3


def _unpack(x, ctx):
    sigma, omega0, rho0 = x[0], x[1], x[2]
    i = 3
    phi0 = x[i] if ctx.has_phi else 0.0
    i += int(ctx.has_phi)
    mu_tilde_B, mu_C = x[i], x[i + 1]
    i += 2
    mu_S = x[i] if ctx.has_muS else 0.0
    i += int(ctx.has_muS)
    mu_L = x[i] if ctx.has_muL else 0.0
    return sigma, omega0, rho0, phi0, mu_tilde_B, mu_C, mu_S, mu_L


def _baryon_kinetics(ctx, sigma, omega0, rho0, phi0, mu_tilde_B, mu_C, mu_S):
    """
    Per-baryon (mu_eff, m*, n, ns, eps, P, s) at the current fields.
    Returns list of (spec, mu_eff, ms, n, ns, eps, P, s) in natural units, or
    None if any effective mass is non-positive (outside the physical domain).
    """
    out = []
    for spec in ctx.specs:
        mass, Q, t3, g, xs, xw, xr, xphi, S = spec
        Gs, Gw, Gr = xs * ctx.Gs_N, xw * ctx.Gw_N, xr * ctx.Gr_N
        Gphi = xphi * ctx.Gw_N          # φ inherits f_omega (DD2Y)
        ms = mass - Gs * sigma
        if ms <= 0.0:
            return None
        mu_eff = (mu_tilde_B + Q * mu_C + S * mu_S - Gw * omega0
                  - Gr * t3 * rho0 - Gphi * phi0)
        n, P, eps, s, ns = kinetic_thermo(mu_eff, ms, g, ctx.T)
        out.append((spec, mu_eff, ms, n, ns, eps, P, s))
    return out


def octet_residual(x, ctx):
    """Dimensionless residual vector (the unified charge/strangeness scheme)."""
    sigma, omega0, rho0, phi0, mu_tilde_B, mu_C, mu_S, mu_L = _unpack(x, ctx)
    kin = _baryon_kinetics(ctx, sigma, omega0, rho0, phi0,
                           mu_tilde_B, mu_C, mu_S)
    if kin is None:
        return [1.0e6] * ctx.n_unknowns

    src_s = src_w = src_r = src_phi = 0.0
    n_tot = charge = strangeness = 0.0
    for (spec, mu_eff, ms, n, ns, eps, P, s) in kin:
        mass, Q, t3, g, xs, xw, xr, xphi, S = spec
        src_s += xs * ctx.Gs_N * ns
        src_w += xw * ctx.Gw_N * n
        src_r += xr * ctx.Gr_N * t3 * n
        src_phi += xphi * ctx.Gw_N * n          # Γ_phiY = x_phi·Γ_omegaN
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
    # Thermal mesons carry no baryon number, so n_B above is baryons only, but
    # they do carry charge and strangeness: both constraints below see the sum.
    mC, mS = meson_charges_nat(ctx, mu_C, mu_S, omega0, rho0)
    charge += mC
    strangeness += mS
    res.append(n_tot / ctx.nB_nat - 1.0)

    n_e = 0.0
    if ctx.charge_mode == "neutral":
        # electron carries +mu_L when leptons are trapped; the muon family
        # stays transparent (mu_mu = -mu_C, no mu-neutrino tracked).
        mu_e = mu_L - mu_C
        mu_mu = -mu_C
        n_e = kinetic_thermo(mu_e, ctx.m_e, 2.0, ctx.T)[0]
        n_mu = (kinetic_thermo(mu_mu, ctx.m_mu, 2.0, ctx.T)[0]
                if ctx.include_muons else 0.0)
        res.append((charge - n_e - n_mu) / ctx.nB_nat)     # neutrality
    else:
        res.append((charge / ctx.nB_nat) - ctx.Y_C)        # fixed hadronic Y_C
    if ctx.has_muS:
        res.append((strangeness / ctx.nB_nat) - ctx.Y_S)   # fixed Y_S
    if ctx.has_muL:
        n_nue = kinetic_thermo(mu_L, 0.0, _G_NU, ctx.T)[0]  # mu_nue = mu_L
        res.append((n_e + n_nue) / ctx.nB_nat - ctx.Y_L)   # electron-family Y_L
    return res


def assemble_octet(x, ctx):
    """
    Full thermodynamic state from a converged unknown vector. Returns a dict
    in natural units plus per-species densities (fm^-3) for onset detection.
    """
    sigma, omega0, rho0, phi0, mu_tilde_B, mu_C, mu_S, mu_L = _unpack(x, ctx)
    kin = _baryon_kinetics(ctx, sigma, omega0, rho0, phi0,
                           mu_tilde_B, mu_C, mu_S)

    eps_b = P_b = s_b = 0.0
    Sig_R = 0.0
    n_tot = charge_had = strangeness = 0.0
    m_eff_n = mu_eff_n = mu_eff_p = None
    densities = {}
    for (spec, mu_eff, ms, n, ns, eps, P, s), b in zip(kin, ctx.baryons):
        mass, Q, t3, g, xs, xw, xr, xphi, S = spec
        eps_b += eps
        P_b += P
        s_b += s
        n_tot += n
        charge_had += Q * n
        strangeness += S * n
        densities[b.name] = n / hc3
        # rearrangement; φ inherits f_omega so ∂Γ_phiY/∂n = x_phi·dGw_N
        Sig_R += (xw * ctx.dGw_N * omega0 * n
                  + xr * ctx.dGr_N * rho0 * t3 * n
                  + xphi * ctx.dGw_N * phi0 * n
                  - xs * ctx.dGs_N * sigma * ns)
        if b.name == "n":
            m_eff_n, mu_eff_n = ms, mu_eff
        elif b.name == "p":
            mu_eff_p = mu_eff

    # meson field energies: scalars minus in P, vectors plus
    s2 = ctx.m_sigma2 * sigma ** 2
    w2 = ctx.m_omega2 * omega0 ** 2
    r2 = ctx.m_rho2 * rho0 ** 2
    p2 = ctx.m_phi2 * phi0 ** 2
    eps_fields = 0.5 * (s2 + w2 + r2 + p2)
    P_fields = 0.5 * (-s2 + w2 + r2 + p2)

    # Thermal meson charge/strangeness, matching octet_residual. `charge_tot`
    # is the total NON-leptonic charge (baryons + mesons) and is what
    # neutrality, Y_C and Y_S are stated in terms of; `charge_had` stays
    # baryons-only because mu_dot_n below is the baryon mu_i n_i sum — the
    # meson sum_j mu*_j n_j is a separate term the callers add from
    # thermal_meson_thermo, and folding it in here would double count it.
    mC, mS = meson_charges_nat(ctx, mu_C, mu_S, omega0, rho0)
    charge_tot = charge_had + mC
    strangeness_tot = strangeness + mS

    nnue = Pnue = enue = snue = 0.0
    if ctx.charge_mode == "neutral":
        # beta / trapped: electron carries +mu_L, muon family transparent
        mu_e = mu_L - mu_C
        mu_mu = -mu_C
        ne, Pe, ee, se, _ = kinetic_thermo(mu_e, ctx.m_e, 2.0, ctx.T)
        if ctx.include_muons:
            nmu, Pmu, emu, smu, _ = kinetic_thermo(mu_mu, ctx.m_mu, 2.0, ctx.T)
        else:
            nmu = Pmu = emu = smu = 0.0
        if ctx.has_muL:      # trapped electron-neutrinos (mu_nue = mu_L)
            nnue, Pnue, enue, snue, _ = kinetic_thermo(mu_L, 0.0, _G_NU, ctx.T)
        lepton_dot_n = mu_e * ne + mu_mu * nmu + mu_L * nnue
    elif ctx.yc_leptons:
        # fixed-Y_C flavor 2b: neutralizing leptons (mu_mu = mu_e), n_e+n_mu =
        # hadronic charge so the total is neutral.
        mu_e, (ne, Pe, ee, se), (nmu, Pmu, emu, smu) = _yc_neutralizing_leptons(
            charge_tot, ctx.m_e, ctx.m_mu, ctx.include_muons, ctx.T)
        mu_mu = mu_e
        lepton_dot_n = mu_e * (ne + nmu)
    else:
        # fixed-Y_C flavor 2a: leptonless (CompOSE Y_Q slicing)
        mu_e = -mu_C
        ne = Pe = ee = se = nmu = Pmu = emu = smu = 0.0
        lepton_dot_n = 0.0

    mu_B = mu_tilde_B + Sig_R
    eps = eps_b + eps_fields + ee + emu + enue
    P = P_b + P_fields + ctx.nB_nat * Sig_R + Pe + Pmu + Pnue
    s = s_b + se + smu + snue

    # baryon chemical-potential sum: mu_i = mu_B + Q_i mu_C + S_i mu_S
    mu_dot_n = mu_B * n_tot + mu_C * charge_had + mu_S * strangeness

    return dict(
        sigma=sigma, omega0=omega0, rho0=rho0, phi0=phi0,
        m_eff_n=m_eff_n, mu_eff_n=mu_eff_n, mu_eff_p=mu_eff_p,
        Sigma_R=Sig_R, mu_B=mu_B, mu_C=mu_C, mu_S=mu_S, mu_L=mu_L,
        mu_e=mu_e, mu_nue=mu_L,
        mu_n=mu_B, mu_p=mu_B + mu_C,
        eps=eps, P=P, s=s, n_tot=n_tot,
        n_e=ne, n_mu=nmu, n_nue=nnue,
        Y_C=charge_tot / ctx.nB_nat, Y_S=strangeness_tot / ctx.nB_nat,
        Y_C_baryons=charge_had / ctx.nB_nat,
        Y_S_baryons=strangeness / ctx.nB_nat,
        Y_L=(ne + nnue) / ctx.nB_nat,
        densities=densities,
        mu_dot_n=mu_dot_n + lepton_dot_n,
    )
