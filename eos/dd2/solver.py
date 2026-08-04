"""
solver.py
====================
DD2 nucleonic solves: fixed-composition and beta-equilibrium matter at
T = 0 and T > 0, plus the general octet solve that all equilibrium modes go
through.

Fixed composition (n_n, n_p): one scalar sigma gap solve; at T > 0 the
kinetic potentials are recovered per species by inverting the JEL density
(eos.general.fermi_integrals.invert_fermi_density). Beta equilibrium: the
potential-driven charge-vector system of physics/residual.py.

Nucleon-mass convention: by default the uniform-matter kernel uses the
AVERAGE nucleon mass (m_n + m_p)/2 for both species, so m_n and m_p enter only
through that average. `Parametrization.nucleon_mass_mode` selects the
per-species alternative.

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
from eos.dd2.physics.octet import (
    assemble_octet, build_octet_ctx, octet_residual,
)
from eos.dd2.physics.jacobian import octet_jacobian
from eos.dd2.physics.kernel_numba import (
    residual_t0_jit, jacobian_t0_jit, build_numba_arrays, _NUMBA_OK,
)
from eos.dd2.physics.mesons import thermal_meson_thermo

#: Hugenholtz–Van Hove residual gate, relative to eps.
HVH_RTOL = 1.0e-8

#: Post-solve gate on the (dimensionless) equilibrium residuals.
RESIDUAL_TOL = 1.0e-10


@dataclass(frozen=True)
class EoSPoint:
    """One solved thermodynamic state (lite version of)."""
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
    mu_S: float = 0.0   # MeV (strangeness potential; 0 unless strangeness fixed)
    mu_L: float = 0.0   # MeV (lepton potential; 0 unless neutrino-trapped)
    n_nu: float = 0.0   # fm^-3 (net electron-neutrino density; trapped only)
    phi0: float = 0.0   # MeV (hidden-strange vector; 0 without hyperons)
    composition: tuple = ()  # ((name, n_i [fm^-3]), ...) all active baryons

    @property
    def Y_p(self):
        return self.n_p / self.n_B

    @property
    def Y_e(self):
        """Electron fraction n_e / n_B."""
        return self.n_e / self.n_B

    @property
    def Y_mu(self):
        """Muon fraction n_mu / n_B."""
        return self.n_mu / self.n_B

    @property
    def composition_map(self):
        return dict(self.composition)

    def Y(self, name):
        """Population fraction n_i / n_B of baryon `name` (0 if absent)."""
        return self.composition_map.get(name, 0.0) / self.n_B

    @property
    def free_energy_density(self):
        """F = eps - T s [MeV/fm^3]."""
        return self.eps - self.T * self.s


def _nucleon_nus(n_n, n_p, ms_n, ms_p, T):
    """Kinetic potentials hitting the target densities [fm^-3]."""
    if T == 0.0:
        nu_n = float(xp.sqrt(kF_from_n(n_n * hc3, 2.0) ** 2 + ms_n ** 2)) \
            if n_n > 0.0 else 0.0
        nu_p = float(xp.sqrt(kF_from_n(n_p * hc3, 2.0) ** 2 + ms_p ** 2)) \
            if n_p > 0.0 else 0.0
        return nu_n, nu_p
    from eos.general.fermi_integrals import invert_fermi_density
    nu_n = invert_fermi_density(n_n, T, ms_n, 2.0) if n_n > 0.0 else 0.0
    nu_p = invert_fermi_density(n_p, T, ms_p, 2.0) if n_p > 0.0 else 0.0
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
    m_kn, m_kp = par.kernel_masses()
    Gs, _, _, _, _, _ = par.couplings_at(n_B)

    def gap(sig):
        ms_n, ms_p = m_kn - Gs * sig, m_kp - Gs * sig
        nu_n, nu_p = _nucleon_nus(n_n, n_p, ms_n, ms_p, T)
        ns = (kinetic_thermo(nu_n, ms_n, 2.0, T)[4]
              + kinetic_thermo(nu_p, ms_p, 2.0, T)[4])
        return sig - Gs * ns / par.m_sigma ** 2

    sigma = brentq(gap, 0.0, 0.999 * min(m_kn, m_kp) / Gs, xtol=1e-12)
    ms_n, ms_p = m_kn - Gs * sigma, m_kp - Gs * sigma
    nu_n, nu_p = _nucleon_nus(n_n, n_p, ms_n, ms_p, T)
    tn = kinetic_thermo(nu_n, ms_n, 2.0, T)
    tp = kinetic_thermo(nu_p, ms_p, 2.0, T)

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
        m_eff=float(0.5 * (ms_n + ms_p)), Sigma_R=float(Sig_R),
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
    [fm^-3] and temperature T [MeV] ( mode 1: mu_S = mu_L = 0,
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
        # The residual norm is the real acceptance criterion: hybr can report
        # success=False ("not making good progress") when it is already at the
        # root but cannot improve past round-off.
        res_max = max(abs(r) for r in beta_eq_residual(sol.x, ctx))
        if res_max <= RESIDUAL_TOL:
            break
    else:
        raise RuntimeError(
            f"beta-equilibrium solve failed at n_B={n_B}, T={T}: "
            f"{sol.message} (max residual {res_max:.2e}, "
            f"tol {RESIDUAL_TOL:.0e})")

    # Converged composition -> assemble the hadronic sector through the same
    # path as the fixed-composition solve (one source of truth).
    nu_n, nu_p, ms_n, ms_p = beta_eq_nucleon_nus(sol.x, ctx)
    n_n = kinetic_thermo(nu_n, ms_n, 2.0, T)[0] / hc3
    n_p = kinetic_thermo(nu_p, ms_p, 2.0, T)[0] / hc3
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


# =============================================================================
# OCTET: the general solve over all active baryons
# =============================================================================
def _octet_x0(fields, has_phi, has_muS, has_muL=False):
    """Pack [sigma, omega0, rho0, (phi0), muB~, muQ, (muS), (muL)]."""
    sigma, omega0, rho0, phi0, mutB, muQ, muS, muL = fields
    x = [sigma, omega0, rho0]
    if has_phi:
        x.append(phi0)
    x += [mutB, muQ]
    if has_muS:
        x.append(muS)
    if has_muL:
        x.append(muL)
    return x


def octet_warm_start(point, has_phi, has_muS=False, has_muL=False):
    """Unknown vector from a solved octet EoSPoint (for sweep continuation).

    mu_Q is recovered as mu_p - mu_n (robust in trapped mode, where
    mu_e = mu_L - mu_Q so -mu_e no longer equals mu_Q).
    """
    return _octet_x0((point.sigma, point.omega0, point.rho0, point.phi0,
                      point.mu_n - point.Sigma_R, point.mu_p - point.mu_n,
                      point.mu_S, point.mu_L),
                     has_phi, has_muS, has_muL)


def default_octet_guess(par, n_B, flags, T=0.0, has_muS=False, has_muL=False):
    """
    Seed the octet solve from the nucleon beta-eq solution (hyperons start at
    zero population and switch on as density rises). phi0 seeded slightly
    negative to break the exact-zero symmetry; muS/muL seeded at 0.
    """
    base = solve_beta_eq(par, n_B, T=T, include_muons=flags.muons,
                         include_photons=False, check_consistency=False)
    has_phi = flags.phi_field and flags.hyperons
    return _octet_x0((base.sigma, base.omega0, base.rho0, -1e-3,
                      base.mu_n - base.Sigma_R, base.mu_p - base.mu_n, 0.0, 0.0),
                     has_phi, has_muS, has_muL)


def solve_octet(par, n_B, flags, T=0.0, x0=None, charge_mode="neutral",
                Y_C=0.0, strange_mode="eq", Y_S=0.0, lepton_mode="transparent",
                Y_L=0.0, yc_leptons=False, include_photons=True,
                check_consistency=True, analytic_jac=True):
    """
    General octet solve (the unified charge/strangeness scheme) at (n_B [fm^-3], T [MeV]).

    charge_mode='neutral' is beta equilibrium (leptons, charge-neutral);
    'fixed' fixes the hadronic charge fraction to Y_C with no leptons (the
    CompOSE general-purpose (nB,T,Yq) convention). strange_mode='fixed' adds
    mu_S and fixes the strangeness fraction to Y_S. Raises RuntimeError on
    non-convergence; asserts HVH (and, in beta mode, the beta condition).

    analytic_jac=True (the default) selects the exact-Jacobian backend; it is
    3-11x faster on a warm-started sweep and agrees with the finite-difference
    backend to solver tolerance. Pass False for the finite-difference reference
    path, which stays the correctness oracle when the two disagree. The one
    case where it does not pay is the full octet at T > 0, where it is ~10%
    slower than finite differences; it is left on there for one default rather
    than a speed heuristic.
    """
    ctx = build_octet_ctx(par, n_B, flags, T=T, charge_mode=charge_mode,
                          Y_C=Y_C, strange_mode=strange_mode, Y_S=Y_S,
                          lepton_mode=lepton_mode, Y_L=Y_L, yc_leptons=yc_leptons)
    has_muS, has_muL = ctx.has_muS, ctx.has_muL
    # Lazy guess sequence: try the warm start x0 first and only build the
    # (expensive, un-jitted beta-eq) default guess if x0 is missing or its solve
    # doesn't converge. In a warm-started sweep the fallback is never evaluated,
    # which is the hot-path win — default_octet_guess otherwise dominated the
    # jitted solve (~0.9 -> ~0.3 ms/pt at T=0, fixed-Y_C).
    def _guesses():
        if x0 is not None:
            yield x0
        yield default_octet_guess(par, n_B, flags, T=T, has_muS=has_muS,
                                  has_muL=has_muL)
    # eos_fast backend (analytic_jac): exact analytic Jacobian via MINPACK
    # hybrj. At T=0 the residual AND Jacobian are Numba-jitted (kernel_numba,
    # machine-identical to the NumPy kernel); T>0 uses the NumPy analytic path
    # (the JEL integrals don't jit). eos_ref (analytic_jac=False) uses the
    # forward-difference Jacobian and remains the correctness oracle.
    if analytic_jac and T == 0.0 and _NUMBA_OK:
        _spec, _prm, _flg = build_numba_arrays(ctx)
        res_fn = lambda xx, _c: residual_t0_jit(xx, _spec, _prm, _flg)
        jac_fn = lambda xx, _c: jacobian_t0_jit(xx, _spec, _prm, _flg)
    elif analytic_jac:
        res_fn, jac_fn = octet_residual, octet_jacobian
    else:
        res_fn, jac_fn = octet_residual, None
    sol = None
    for guess in _guesses():
        sol = root(res_fn, guess, args=(ctx,), jac=jac_fn,
                   method="hybr", tol=1e-13)
        res_max = max(abs(r) for r in res_fn(sol.x, ctx))
        if res_max <= RESIDUAL_TOL:
            break
    else:
        raise RuntimeError(
            f"octet solve failed at n_B={n_B}, T={T} "
            f"(charge={charge_mode}, strange={strange_mode}): {sol.message} "
            f"(max residual {res_max:.2e}, tol {RESIDUAL_TOL:.0e})")

    st = assemble_octet(sol.x, ctx)
    if include_photons and T > 0.0:
        ph = photon_thermo(T)
        st["eps"] += ph.e * hc3
        st["P"] += ph.P * hc3
        st["s"] += ph.s * hc3
    # Thermal meson gas: additive Bose gas on top of the mean
    # field, evaluated at the converged charge/strange potentials and vector
    # fields. mu*_j n_j joins the HVH sum (the gas satisfies e+P = Ts+mu* n).
    if (flags.include_pseudoscalars or flags.include_thermal_vectors) and T > 0:
        mg = thermal_meson_thermo(
            par, n_B, st["mu_Q"], st["mu_S"], st["omega0"], st["rho0"], T,
            include_pseudoscalars=flags.include_pseudoscalars,
            include_thermal_vectors=flags.include_thermal_vectors)
        st["eps"] += mg["e"] * hc3
        st["P"] += mg["P"] * hc3
        st["s"] += mg["s"] * hc3
        st["mu_dot_n"] += mg["mu_dot_n"] * hc3

    hvh_rel = (st["eps"] + st["P"] - T * st["s"] - st["mu_dot_n"]) / st["eps"]
    if check_consistency:
        if abs(hvh_rel) > HVH_RTOL:
            raise ValueError(
                f"Hugenholtz–Van Hove violated at n_B={n_B}, T={T} "
                f"(octet {charge_mode}/{strange_mode}): "
                f"|{hvh_rel:.2e}| > {HVH_RTOL:.0e}")
        if charge_mode == "neutral":
            # transparent: mu_n - mu_p = mu_e; trapped: = mu_e - mu_nue.
            beta_res = st["mu_n"] - st["mu_p"] - (st["mu_e"] - st["mu_nue"])
            if abs(beta_res) > 1e-6:
                raise ValueError(
                    f"beta-equilibrium condition violated at n_B={n_B}, "
                    f"T={T}: mu_n - mu_p - (mu_e - mu_nue) = {beta_res:.2e} MeV")

    dmap = st["densities"]
    return EoSPoint(
        n_B=n_B, T=T, n_n=dmap.get("n", 0.0), n_p=dmap.get("p", 0.0),
        sigma=st["sigma"], omega0=st["omega0"], rho0=st["rho0"],
        phi0=st["phi0"], m_eff=st["m_eff_n"], Sigma_R=st["Sigma_R"],
        nu_n=st["nu_n"], nu_p=st["nu_p"], mu_n=st["mu_n"], mu_p=st["mu_p"],
        eps=st["eps"] / hc3, P=st["P"] / hc3, s=st["s"] / hc3,
        hvh_rel=float(hvh_rel), n_e=st["n_e"] / hc3, n_mu=st["n_mu"] / hc3,
        mu_e=st["mu_e"], mu_S=st["mu_S"], mu_L=st["mu_L"],
        n_nu=st["n_nue"] / hc3, composition=tuple(sorted(dmap.items())),
    )


def solve_beta_eq_octet(par, n_B, flags, T=0.0, x0=None,
                        include_photons=True, check_consistency=True,
                        analytic_jac=True):
    """
    Beta-equilibrium matter with the full active baryon set (
    mode 1; mu_S = mu_L = 0, charge neutrality). Thin wrapper over solve_octet.
    Reduces to the nucleon problem when flags.hyperons is False.
    """
    return solve_octet(par, n_B, flags, T=T, x0=x0, charge_mode="neutral",
                       include_photons=include_photons,
                       check_consistency=check_consistency,
                       analytic_jac=analytic_jac)


def solve_fixed_yc_octet(par, n_B, Y_C, flags, T=0.0, x0=None, Y_S=None,
                         leptons=False, include_photons=True,
                         check_consistency=True):
    """
    Fixed hadronic charge fraction Y_C. Two flavors:

    - leptons=False (2a, default): leptonless — the CompOSE general-purpose
      (nB,T,Yq) slicing; mu_Q is the Lagrange multiplier for Y_C.
    - leptons=True (2b): populate electrons (+muons iff flags.muons) so the
      TOTAL is charge-neutral (n_e+n_mu = Y_C n_B, mu_e=mu_mu). The hadronic
      solve is identical to 2a; the leptons are a post-hoc neutraliser. Read
      Y_e / Y_mu off the result.

    Optionally also fix the strangeness fraction Y_S (adds mu_S; §1.7 mode 3),
    composing with either flavor.
    """
    strange_mode = "fixed" if Y_S is not None else "eq"
    return solve_octet(par, n_B, flags, T=T, x0=x0, charge_mode="fixed",
                       Y_C=Y_C, strange_mode=strange_mode, Y_S=(Y_S or 0.0),
                       yc_leptons=leptons, include_photons=include_photons,
                       check_consistency=check_consistency)


def solve_yl_octet(par, n_B, Y_L, flags, T=0.0, x0=None,
                   include_photons=True, check_consistency=True):
    """
    Neutrino-trapped matter at fixed electron lepton fraction
    Y_L = (n_e + n_nue)/n_B. Charge-neutral, mu_L unknown,
    electron-neutrinos included (mu_nue = mu_L, mu_e = mu_L - mu_Q). The muon
    family stays transparent. Requires SpeciesFlags(neutrinos=True).
    """
    if not flags.neutrinos:
        raise ValueError("solve_yl_octet requires SpeciesFlags(neutrinos=True)")
    return solve_octet(par, n_B, flags, T=T, x0=x0, charge_mode="neutral",
                       lepton_mode="trapped", Y_L=Y_L,
                       include_photons=include_photons,
                       check_consistency=check_consistency)


def sweep_beta_eq_octet(par, n_B_grid, flags, T=0.0, include_photons=True,
                        max_bisect=6, stop_at_boundary=False,
                        analytic_jac=True):
    """
    Warm-started density sweep with step-bisection continuation.
    Each point seeds the next; through a sharp onset where the warm start
    would jump branches, the step to the next density is bisected (recursively,
    up to max_bisect levels) so the predictor stays in the corrector's basin.

    A Δ-matter model can hit a scalar-collapse feasibility boundary (m* -> 0)
    at high density. With stop_at_boundary=True the sweep returns
    the valid prefix instead of raising once every sub-step past the boundary
    has failed; otherwise it raises (no silent truncation by default).

    Returns a list of EoSPoint in n_B_grid order.
    """
    return sweep_octet(par, n_B_grid, flags, T=T, include_photons=include_photons,
                       max_bisect=max_bisect, stop_at_boundary=stop_at_boundary,
                       analytic_jac=analytic_jac)


def sweep_octet(par, n_B_grid, flags, T=0.0, charge_mode="neutral", Y_C=0.0,
                strange_mode="eq", Y_S=0.0, lepton_mode="transparent", Y_L=0.0,
                yc_leptons=False, include_photons=True, max_bisect=6,
                stop_at_boundary=False, analytic_jac=True):
    """
    Warm-started density sweep for any octet mode, with the same
    step-bisection continuation and scalar-collapse boundary handling as the
    beta-eq sweep. See solve_octet for the mode arguments. analytic_jac selects
    the eos_fast (exact-Jacobian) backend, on by default; False is the
    finite-difference reference path.
    """
    has_phi = flags.phi_field and flags.hyperons
    has_muS = (strange_mode == "fixed")
    has_muL = (lepton_mode == "trapped")

    def solve_from(n_B, x0):
        return solve_octet(par, n_B, flags, T=T, x0=x0, charge_mode=charge_mode,
                           Y_C=Y_C, strange_mode=strange_mode, Y_S=Y_S,
                           lepton_mode=lepton_mode, Y_L=Y_L, yc_leptons=yc_leptons,
                           include_photons=include_photons,
                           analytic_jac=analytic_jac)

    def step(n_prev, n_target, x0, depth):
        try:
            return solve_from(n_target, x0)
        except RuntimeError:
            if depth >= max_bisect or n_prev is None:
                raise
            n_mid = 0.5 * (n_prev + n_target)
            p_mid = step(n_prev, n_mid, x0, depth + 1)
            return step(n_mid, n_target,
                        octet_warm_start(p_mid, has_phi, has_muS, has_muL),
                        depth + 1)

    points, x0, n_prev = [], None, None
    for n_B in n_B_grid:
        try:
            p = step(n_prev, n_B, x0, 0)
        except RuntimeError:
            if stop_at_boundary and points:
                break
            raise
        points.append(p)
        x0 = octet_warm_start(p, has_phi, has_muS, has_muL)
        n_prev = n_B
    return points
