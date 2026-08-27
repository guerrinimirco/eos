"""
solver.py
=========
DD2's equilibrium conditions and the solves that close them: fixed-composition
and beta-equilibrium nucleonic matter at T = 0 and T > 0, and the general octet
system every mode of CLAUDE.md section 3 goes through.

`thermodynamics.py` computes quantities FROM the state; this module FINDS the
state. What separates them is the mode: the residual here knows which
conserved charges are held and which equilibrate, and it reads that from a
`ModeSpec` rather than branching on strings, so the equations are written once
and the mode selects among them.

Two systems, because they have different unknown vectors:

  nucleons     x = [sigma, rho0, mu_eff_n, mu_C]. omega0 is eliminated at the
               target density, which only works while every species shares the
               nucleon couplings.
  octet        x = [sigma, omega0, rho0, (phi0), mu_tilde_B, mu_C, (mu_S),
               (mu_nue)]. Once the couplings differ per species the vector
               sources are composition-dependent and all the fields become
               unknowns. Reduces to the nucleon problem when hyperons are off,
               and is the path every public mode takes.

Fixed composition (n_n, n_p) is neither: one scalar sigma gap solve, with the
kinetic potentials recovered per species at T > 0 by inverting the JEL density
(eos.general.fermi_integrals.invert_fermi_density).

The unknowns are the EFFECTIVE potentials rather than mu_B (CLAUDE.md section
2): the rearrangement term and the large vector shift cancel out of the
iteration, and mu_eff varies smoothly along a density sweep, which is what
makes the warm starts work.

Nucleon-mass convention: by default the uniform-matter kernel uses the
AVERAGE nucleon mass (m_n + m_p)/2 for both species, so m_n and m_p enter only
through that average. `Parameters.nucleon_mass_mode` selects the
per-species alternative.

User-facing units: densities fm^-3, fields/potentials MeV, eps/P MeV/fm^3,
entropy density fm^-3. Internally natural units (MeV powers), converted at
the boundary via hc^3. All assembled quantities are evaluated from one
consistent set of densities, so the Hugenholtz–Van Hove identity
eps + P - T s = sum_i mu_i n_i holds to round-off and is asserted.
"""
from dataclasses import replace
from typing import NamedTuple

from scipy.optimize import brentq, root

from eos.general.physics_constants import hc3
from eos.general.particles import Electron, Muon, Neutron, Proton
from eos.general.thermodynamics_leptons import photon_thermo
from eos.general.basis import species_potential
from eos.general.state import EoSPoint, LeptonThermo, PhaseThermo
from eos.general.modes import (
    Conservation, ModeSpec, electron_potential, muon_potential,
    resolve_leptons, strangeness_potential,
)
import numpy as np
from eos.dd2.thermodynamics import kF_from_n, kinetic_thermo
from eos.dd2.thermodynamics import vector_fields, rearrangement, field_eps_P
from eos.dd2.thermodynamics import (
    G_NU, baryon_kinetics, build_matter_ctx, meson_charges_nat,
    thermal_meson_thermo,
)
from eos.general.thermodynamics_leptons import neutralizing_leptons
try:
    from eos.dd2.backends.jacobian import residual_jacobian
    from eos.dd2.backends.kernel_numba import (
        residual_t0_jit, jacobian_t0_jit, build_numba_arrays, _NUMBA_OK,
    )
except ImportError:
    # `backends/` is optional: CLAUDE.md section 5 defines it by the property
    # that deleting it changes no number, only the speed. Without it every
    # solve takes the finite-difference reference path below.
    residual_jacobian = build_numba_arrays = None
    residual_t0_jit = jacobian_t0_jit = None
    _NUMBA_OK = False

#: Hugenholtz–Van Hove residual gate, relative to eps.
HVH_RTOL = 1.0e-8

#: Post-solve gate on the (dimensionless) equilibrium residuals.
RESIDUAL_TOL = 1.0e-10


def _nucleon_mu_effs(n_n, n_p, ms_n, ms_p, T):
    """Effective potentials hitting the target densities [fm^-3]."""
    if T == 0.0:
        mu_eff_n = float(np.sqrt(kF_from_n(n_n * hc3, 2.0) ** 2 + ms_n ** 2)) \
            if n_n > 0.0 else 0.0
        mu_eff_p = float(np.sqrt(kF_from_n(n_p * hc3, 2.0) ** 2 + ms_p ** 2)) \
            if n_p > 0.0 else 0.0
        return mu_eff_n, mu_eff_p
    from eos.general.fermi_integrals import invert_fermi_density
    mu_eff_n = invert_fermi_density(n_n, T, ms_n, 2.0) if n_n > 0.0 else 0.0
    mu_eff_p = invert_fermi_density(n_p, T, ms_p, 2.0) if n_p > 0.0 else 0.0
    return mu_eff_n, mu_eff_p


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
        mu_eff_n, mu_eff_p = _nucleon_mu_effs(n_n, n_p, ms_n, ms_p, T)
        ns = (kinetic_thermo(mu_eff_n, ms_n, 2.0, T)[4]
              + kinetic_thermo(mu_eff_p, ms_p, 2.0, T)[4])
        return sig - Gs * ns / par.m_sigma ** 2

    sigma = brentq(gap, 0.0, 0.999 * min(m_kn, m_kp) / Gs, xtol=1e-12)
    ms_n, ms_p = m_kn - Gs * sigma, m_kp - Gs * sigma
    mu_eff_n, mu_eff_p = _nucleon_mu_effs(n_n, n_p, ms_n, ms_p, T)
    tn = kinetic_thermo(mu_eff_n, ms_n, 2.0, T)
    tp = kinetic_thermo(mu_eff_p, ms_p, 2.0, T)

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
    mu_n = mu_eff_n + vector_shift + Gr * Neutron.t3 * rho0
    mu_p = mu_eff_p + vector_shift + Gr * Proton.t3 * rho0

    hvh_rel = (eps_nat + P_nat - T * s_nat
               - (mu_n * nn_nat + mu_p * np_nat)) / eps_nat
    if check_consistency and abs(hvh_rel) > HVH_RTOL:
        raise ValueError(
            f"Hugenholtz–Van Hove violated at n_B={n_B}, T={T}: "
            f"|{hvh_rel:.2e}| > {HVH_RTOL:.0e} — a Sigma^R term is missing "
            f"or inconsistent")

    # Fixing (n_n, n_p) at a given n_B IS fixing the charge fraction, with no
    # leptons: it is `fixed_YC` asked in densities rather than in Y_C.
    # Not `n_B`: that is the TARGET (n_n + n_p), while this is the density the
    # assembly actually evaluated at, which differs at T > 0 by the inversion
    # tolerance. `_solved` is the same distinction `eos.enjl.table` draws with
    # `nB_solved`; `_fm` would say nothing now that a bare name IS fm.
    n_B_solved = float(nB_nat / hc3)
    densities = {"n": float(nn_nat / hc3), "p": float(np_nat / hc3)}
    mu_B, mu_C = float(mu_n), float(mu_p - mu_n)
    matter = PhaseThermo(
        T=T, mu_B=mu_B, mu_C=mu_C, mu_S=0.0,
        fields={"sigma": float(sigma), "omega0": float(omega0),
                "rho0": float(rho0), "phi0": 0.0},
        densities=densities,
        mu_i={name: species_potential(name, mu_B, mu_C, 0.0)
              for name in densities},
        mu_eff_i={"n": float(mu_eff_n), "p": float(mu_eff_p)},
        m_eff_i={"n": float(ms_n), "p": float(ms_p)},
        n_B=n_B_solved, n_C=float(np_nat / hc3), n_S=0.0,
        P=float(P_nat / hc3), eps=float(eps_nat / hc3), s=float(s_nat / hc3),
        mu_dot_n=float((mu_n * nn_nat + mu_p * np_nat) / hc3),
        Sigma_R=float(Sig_R),
    )
    return EoSPoint(
        mode="fixed_YC", n_B=n_B_solved, T=T,
        conditions={"Y_C": float(np_nat / nB_nat)},
        matter=matter, leptons=None,
        P=matter.P, eps=matter.eps, s=matter.s,
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


def nucleon_warm_start(point):
    """Warm-start vector [sigma, rho0, mu_eff_n, mu_C] from a solved EoSPoint."""
    m = point.matter
    return [m.fields["sigma"], m.fields["rho0"], m.mu_eff_i["n"],
            -point.leptons.mu_e]


def default_nucleon_guess(par, n_B, T=0.0, Y_p=0.05):
    """
    Starting vector [sigma, rho0, mu_eff_n, mu_C] from an exactly solved
    fixed-composition point at Y_p: only the charge closure is off.
    """
    base = solve_composition(par, (1.0 - Y_p) * n_B, Y_p * n_B, T=T)
    m = base.matter
    return [m.fields["sigma"], m.fields["rho0"], m.mu_eff_i["n"], -m.mu_C]


class BetaCtx(NamedTuple):
    """The nucleon beta-equilibrium system at one density and temperature.

    Smaller than `MatterCtx` on purpose: this system carries only n and p, so
    omega0 is eliminated at the target density and phi never enters.
    """
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
    sigma, rho0, mu_eff_n, mu_C = x
    ms_n = ctx.m_kn - ctx.Gs * sigma
    ms_p = ctx.m_kp - ctx.Gs * sigma
    mu_eff_p = mu_eff_n + mu_C - (Proton.t3 - Neutron.t3) * ctx.Gr * rho0
    return mu_eff_n, mu_eff_p, ms_n, ms_p


def beta_eq_residual(x, ctx):
    """
    Dimensionless residuals: [sigma gap, rho0 field eq, baryon number,
    charge neutrality]. Zero exactly at the solved state.
    """
    sigma, rho0, _, mu_C = x
    mu_eff_n, mu_eff_p, ms_n, ms_p = beta_eq_nucleon_mu_eff(x, ctx)
    if min(ms_n, ms_p) <= 0.0:
        return [1.0e6, 0.0, 0.0, 0.0]   # outside physical domain

    n_n, _, _, _, ns_n = kinetic_thermo(mu_eff_n, ms_n, 2.0, ctx.T)
    n_p, _, _, _, ns_p = kinetic_thermo(mu_eff_p, ms_p, 2.0, ctx.T)
    mu_e = electron_potential(mu_C)
    n_e = kinetic_thermo(mu_e, ctx.m_e, 2.0, ctx.T)[0]
    n_mu = (kinetic_thermo(muon_potential(mu_e), ctx.m_mu, 2.0, ctx.T)[0]
            if ctx.include_muons else 0.0)

    n3 = Neutron.t3 * n_n + Proton.t3 * n_p
    return [
        (sigma - ctx.Gs * (ns_n + ns_p) / ctx.m_sigma2) / ctx.mbar,
        (rho0 - ctx.Gr * n3 / ctx.m_rho2) / ctx.mbar,
        (n_n + n_p) / ctx.nB_nat - 1.0,
        (n_p - n_e - n_mu) / ctx.nB_nat,
    ]


def solve_beta_eq(par, n_B, T=0.0, x0=None, include_muons=True,
                  include_photons=True, check_consistency=True):
    """
    Neutrino-transparent beta-equilibrium npemu matter at density n_B
    [fm^-3] and temperature T [MeV] ( mode 1: mu_S = mu_nue = 0,
    charge neutrality). Photons contribute at T > 0 when include_photons.

    x0: optional warm-start vector [sigma, rho0, mu_eff_n, mu_C], e.g. from
    nucleon_warm_start() of a neighbouring solution. Falls back to the default
    guess if the warm start stalls; raises RuntimeError on non-convergence
    — no silent failures.
    """
    ctx = make_beta_ctx(par, n_B, T=T, include_muons=include_muons)
    guesses = [x0] if x0 is not None else []
    guesses.append(default_nucleon_guess(par, n_B, T=T))
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
    mu_eff_n, mu_eff_p, ms_n, ms_p = beta_eq_nucleon_mu_eff(sol.x, ctx)
    n_n = kinetic_thermo(mu_eff_n, ms_n, 2.0, T)[0] / hc3
    n_p = kinetic_thermo(mu_eff_p, ms_p, 2.0, T)[0] / hc3
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
    rhs = base.matter.mu_dot_n * hc3 + mu_e * (ne_nat + nmu_nat)
    hvh_rel = (eps_nat + P_nat - T * s_nat - rhs) / eps_nat
    beta_res = -base.matter.mu_C - mu_e
    if check_consistency:
        if abs(hvh_rel) > HVH_RTOL:
            raise ValueError(
                f"Hugenholtz–Van Hove violated at n_B={n_B}, T={T} "
                f"(beta-eq): |{hvh_rel:.2e}| > {HVH_RTOL:.0e}")
        if abs(beta_res) > 1e-6:
            raise ValueError(
                f"beta-equilibrium condition violated at n_B={n_B}, T={T}: "
                f"mu_n - mu_p - mu_e = {beta_res:.2e} MeV")

    leptons = LeptonThermo(
        mu_e=float(mu_e), mu_mu=float(mu_e),
        densities={"e-": float(ne_nat / hc3), "mu-": float(nmu_nat / hc3)},
        P=float((Pe + Pmu) / hc3), eps=float((ee + emu) / hc3),
        s=float((se + smu) / hc3),
        mu_dot_n=float(mu_e * (ne_nat + nmu_nat) / hc3),
    )
    return replace(
        base, mode="beta_eq_neutrinoless", conditions={}, leptons=leptons,
        eps=float(eps_nat / hc3), P=float(P_nat / hc3), s=float(s_nat / hc3),
    )


def solve_beta_eq_t0(par, n_B, x0=None, include_muons=True,
                     check_consistency=True):
    """T=0 beta-equilibrium solve (M2 API)."""
    return solve_beta_eq(par, n_B, T=0.0, x0=x0, include_muons=include_muons,
                         check_consistency=check_consistency)


# =============================================================================
# OCTET: the general solve over all active baryons
# =============================================================================
def _x0(fields, has_phi, has_muS, has_muL=False):
    """Pack [sigma, omega0, rho0, (phi0), muB~, muC, (muS), (muL)]."""
    sigma, omega0, rho0, phi0, mutB, muC, muS, muL = fields
    x = [sigma, omega0, rho0]
    if has_phi:
        x.append(phi0)
    x += [mutB, muC]
    if has_muS:
        x.append(muS)
    if has_muL:
        x.append(muL)
    return x


def warm_start(point, has_phi, has_muS=False, has_muL=False):
    """Unknown vector from a solved octet EoSPoint (for sweep continuation).

    mu_C is recovered as mu_p - mu_n (robust in trapped mode, where
    mu_e = mu_nue - mu_C so -mu_e no longer equals mu_C).
    """
    m = point.matter
    mu_nue = point.leptons.mu_nue if point.leptons is not None else 0.0
    return _x0((m.fields["sigma"], m.fields["omega0"], m.fields["rho0"],
                      m.fields["phi0"], m.mu_B - m.Sigma_R, m.mu_C,
                      m.mu_S, mu_nue),
                     has_phi, has_muS, has_muL)


def default_guess(par, n_B, flags, T=0.0, has_muS=False, has_muL=False):
    """
    Seed the octet solve from the nucleon beta-eq solution (hyperons start at
    zero population and switch on as density rises). phi0 seeded slightly
    negative to break the exact-zero symmetry; muS/muL seeded at 0.
    """
    base = solve_beta_eq(par, n_B, T=T, include_muons=flags.muons,
                         include_photons=False, check_consistency=False)
    has_phi = flags.phi_field and flags.hyperons
    m = base.matter
    return _x0((m.fields["sigma"], m.fields["omega0"], m.fields["rho0"],
                      -1e-3, m.mu_B - m.Sigma_R, m.mu_C, 0.0, 0.0),
                     has_phi, has_muS, has_muL)


def mode_spec(charge_mode="neutral", Y_C=0.0, strange_mode="eq", Y_S=0.0,
              lepton_mode="transparent", Y_Le=0.0, yc_leptons=False):
    """dd2's keyword vocabulary as the shared mode declaration.

    The keywords predate `eos.general.modes` and are what every caller of
    `solve` still writes; this is the one place they turn into a
    `ModeSpec`, so the residual and the assembly branch on the declaration
    rather than on strings.

        charge_mode='neutral'  ->  C equilibrated (beta equilibrium)
        charge_mode='fixed'    ->  C fixed at Y_C, leptons per yc_leptons
        strange_mode='fixed'   ->  S fixed at Y_S; 'eq' leaves mu_S = 0
        lepton_mode='trapped'  ->  L_e fixed at Y_Le, i.e. the spec's Y_Le
    """
    if lepton_mode == "trapped" and charge_mode != "neutral":
        raise ValueError("trapped lepton_mode requires charge_mode='neutral' "
                         "(Y_Le trapping implies leptons present, charge-neutral)")
    if yc_leptons and charge_mode != "fixed":
        raise ValueError("yc_leptons (flavor 2b) requires charge_mode='fixed'")
    fixed = Conservation.FIXED
    eq = Conservation.EQUILIBRATED
    targets = {}
    if charge_mode == "fixed":
        targets["Y_C"] = Y_C
    if strange_mode == "fixed":
        targets["Y_S"] = Y_S
    if lepton_mode == "trapped":
        targets["Y_Le"] = Y_Le
    return ModeSpec(
        C=fixed if charge_mode == "fixed" else eq,
        S=fixed if strange_mode == "fixed" else eq,
        L_e=fixed if lepton_mode == "trapped" else eq,
        targets=targets,
        # `leptons` is only read where C is FIXED; beta equilibrium always has
        # them, and ModeSpec refuses leptons=False there.
        leptons=(yc_leptons or charge_mode != "fixed"),
    )


def n_unknowns(ctx, spec):
    """How many unknowns the octet vector carries in this mode.

        x = [sigma, omega0, rho0, (phi0), mu_tilde_B, mu_C, (mu_S), (mu_nue)]

    Five are always there: the three meson mean fields, the baryon potential
    and mu_C. phi0 joins iff the hidden-strange vector is active, mu_S iff the
    mode holds strangeness, mu_nue iff it traps the electron family. mu_C is an
    unknown in EVERY mode -- what the mode changes is the row that closes it,
    electric neutrality or the held Y_C.

    All the vector fields are unknowns with their field equations as residuals:
    once the couplings differ per species their sources are composition
    dependent, so they can no longer be eliminated at the target density as in
    the nucleon-only system above.
    """
    return (5 + int(ctx.has_phi) + int(spec.is_fixed("S"))
            + int(spec.is_fixed("L_e")))


def _unpack(x, ctx, spec):
    """Read the unknown vector in the order `n_unknowns` documents."""
    sigma, omega0, rho0 = x[0], x[1], x[2]
    i = 3
    phi0 = x[i] if ctx.has_phi else 0.0
    i += int(ctx.has_phi)
    mu_tilde_B, mu_C = x[i], x[i + 1]
    i += 2
    mu_S = x[i] if spec.is_fixed("S") else strangeness_potential(spec)
    i += int(spec.is_fixed("S"))
    mu_nue = x[i] if spec.is_fixed("L_e") else 0.0
    return sigma, omega0, rho0, phi0, mu_tilde_B, mu_C, mu_S, mu_nue


def residual(x, ctx, spec):
    """Dimensionless residual of the octet system in the mode `spec` declares.

    Three families of row, in this order: the meson field equations, always all
    of them because a mean field does not know what is being held fixed; the
    baryon-number row; and one row per conserved charge the mode constrains.
    Field equations are scaled by the nucleon mass and density rows by n_B, so
    every entry is dimensionless.
    """
    sigma, omega0, rho0, phi0, mu_tilde_B, mu_C, mu_S, mu_nue = _unpack(
        x, ctx, spec)
    kin = baryon_kinetics(ctx, sigma, omega0, rho0, phi0,
                          mu_tilde_B, mu_C, mu_S)
    if kin is None:
        return [1.0e6] * n_unknowns(ctx, spec)

    src_s = src_w = src_r = src_phi = 0.0
    n_tot = charge = strangeness = 0.0
    for (_name, bspec, mu_eff, ms, n, ns, eps, P, s) in kin:
        mass, Q, t3, g, xs, xw, xr, xphi, S = bspec
        src_s += xs * ctx.Gs_N * ns
        src_w += xw * ctx.Gw_N * n
        src_r += xr * ctx.Gr_N * t3 * n
        src_phi += xphi * ctx.Gw_N * n          # Gamma_phiY = x_phi*Gamma_omegaN
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
    # Thermal mesons carry no baryon number, so n_tot above is baryons only,
    # but they do carry charge and strangeness: both constraints below see the
    # sum (CLAUDE.md section 2).
    mC, mS = meson_charges_nat(ctx, mu_C, mu_S, omega0, rho0)
    charge += mC
    strangeness += mS
    res.append(n_tot / ctx.nB_nat - 1.0)

    n_e = 0.0
    if not spec.is_fixed("C"):
        # Beta equilibrium closes the charge sector: mu_e follows from mu_C and
        # the electron-neutrino potential, and electric neutrality is the row.
        mu_e = electron_potential(mu_C, mu_nue)
        n_e = kinetic_thermo(mu_e, ctx.m_e, 2.0, ctx.T)[0]
        n_mu = (kinetic_thermo(muon_potential(mu_e, mu_nue), ctx.m_mu, 2.0,
                               ctx.T)[0]
                if ctx.include_muons else 0.0)
        res.append((charge - n_e - n_mu) / ctx.nB_nat)
    else:
        res.append((charge / ctx.nB_nat) - spec.targets["Y_C"])
    if spec.is_fixed("S"):
        res.append((strangeness / ctx.nB_nat) - spec.targets["Y_S"])
    if spec.is_fixed("L_e"):
        n_nue = kinetic_thermo(mu_nue, 0.0, G_NU, ctx.T)[0]
        res.append((n_e + n_nue) / ctx.nB_nat - spec.targets["Y_Le"])
    return res


def assemble(x, ctx, spec):
    """
    Full thermodynamic state from a converged unknown vector. Returns a dict
    in natural units plus per-species densities (fm^-3) for onset detection.
    """
    sigma, omega0, rho0, phi0, mu_tilde_B, mu_C, mu_S, mu_nue = _unpack(
        x, ctx, spec)
    kin = baryon_kinetics(ctx, sigma, omega0, rho0, phi0,
                          mu_tilde_B, mu_C, mu_S)

    eps_b = P_b = s_b = 0.0
    Sig_R = 0.0
    n_tot = charge_had = strangeness = 0.0
    densities = {}
    mu_eff_i = {}
    m_eff_i = {}
    for (_name, bspec, mu_eff, ms, n, ns, eps, P, s), b in zip(kin, ctx.baryons):
        mass, Q, t3, g, xs, xw, xr, xphi, S = bspec
        eps_b += eps
        P_b += P
        s_b += s
        n_tot += n
        charge_had += Q * n
        strangeness += S * n
        # Every active baryon, kept the same way: neither mu_eff_i nor m*_i is
        # recoverable from the densities afterwards, so a Lambda that is
        # dropped here is a Lambda every consumer has to re-solve for.
        densities[b.name] = n / hc3
        mu_eff_i[b.name] = mu_eff
        m_eff_i[b.name] = ms
        # rearrangement; phi inherits f_omega so dGamma_phiY/dn = x_phi*dGw_N
        Sig_R += (xw * ctx.dGw_N * omega0 * n
                  + xr * ctx.dGr_N * rho0 * t3 * n
                  + xphi * ctx.dGw_N * phi0 * n
                  - xs * ctx.dGs_N * sigma * ns)

    # meson field energies: scalars minus in P, vectors plus. Written from the
    # ctx's squared masses rather than through `field_eps_P`, which takes the
    # parameter object this function does not carry.
    s2 = ctx.m_sigma2 * sigma ** 2
    w2 = ctx.m_omega2 * omega0 ** 2
    r2 = ctx.m_rho2 * rho0 ** 2
    p2 = ctx.m_phi2 * phi0 ** 2
    eps_fields = 0.5 * (s2 + w2 + r2 + p2)
    P_fields = 0.5 * (-s2 + w2 + r2 + p2)

    # Thermal meson charge/strangeness, matching residual. `charge_tot`
    # is the total NON-leptonic charge (baryons + mesons) and is what
    # neutrality, Y_C and Y_S are stated in terms of; `charge_had` stays
    # baryons-only because mu_dot_n below is the baryon mu_i n_i sum -- the
    # meson sum_j mu*_j n_j is a separate term the callers add from
    # thermal_meson_thermo, and folding it in here would double count it.
    mC, mS = meson_charges_nat(ctx, mu_C, mu_S, omega0, rho0)
    charge_tot = charge_had + mC
    strangeness_tot = strangeness + mS

    nnue = Pnue = enue = snue = 0.0
    if not spec.is_fixed("C"):
        # beta / trapped: mu_e from beta equilibrium, muon family transparent.
        mu_e = electron_potential(mu_C, mu_nue)
        mu_mu = muon_potential(mu_e, mu_nue)
        ne, Pe, ee, se, _ = kinetic_thermo(mu_e, ctx.m_e, 2.0, ctx.T)
        if ctx.include_muons:
            nmu, Pmu, emu, smu, _ = kinetic_thermo(mu_mu, ctx.m_mu, 2.0, ctx.T)
        else:
            nmu = Pmu = emu = smu = 0.0
        if spec.is_fixed("L_e"):      # trapped electron-neutrinos
            nnue, Pnue, enue, snue, _ = kinetic_thermo(mu_nue, 0.0, G_NU, ctx.T)
        lepton_dot_n = mu_e * ne + mu_mu * nmu + mu_nue * nnue
    elif spec.leptons:
        # fixed-Y_C with neutralizing leptons: mu_mu = mu_e and n_e + n_mu is
        # the non-leptonic charge, so the total is electrically neutral.
        # `neutralizing_leptons` is fm-based, as every shared boundary is;
        # this assembly is in natural units, so the charge goes in divided and
        # the blocks come back multiplied.
        mu_e, e_blk, mu_blk = neutralizing_leptons(
            charge_tot / hc3, ctx.T, include_muons=ctx.include_muons,
            m_e=ctx.m_e, m_mu=ctx.m_mu)
        ne, Pe, ee, se = (e_blk.n * hc3, e_blk.P * hc3,
                          e_blk.e * hc3, e_blk.s * hc3)
        nmu, Pmu, emu, smu = (mu_blk.n * hc3, mu_blk.P * hc3,
                              mu_blk.e * hc3, mu_blk.s * hc3)
        mu_mu = mu_e
        lepton_dot_n = mu_e * (ne + nmu)
    else:
        # fixed-Y_C leptonless: charged matter (the CompOSE Y_q slicing, and
        # what a mixed-phase construction needs per pure phase).
        mu_e = electron_potential(mu_C)
        mu_mu = mu_e
        ne = Pe = ee = se = nmu = Pmu = emu = smu = 0.0
        lepton_dot_n = 0.0

    mu_B = mu_tilde_B + Sig_R
    eps = eps_b + eps_fields + ee + emu + enue
    P = P_b + P_fields + ctx.nB_nat * Sig_R + Pe + Pmu + Pnue
    s = s_b + se + smu + snue

    # baryon chemical-potential sum: mu_i = mu_B + Q_i mu_C + S_i mu_S
    mu_dot_n = mu_B * n_tot + mu_C * charge_had + mu_S * strangeness

    # The same numbers again, split by sector. The totals above keep their
    # summation order exactly as it was, so the split adds quantities rather
    # than recomputing them: `eps`, `P` and `s` are matter plus leptons, and
    # `solve` adds photons (totals only) and the thermal meson gas
    # (which is part of the phase, so it goes to both).
    eps_matter = eps_b + eps_fields
    P_matter = P_b + P_fields + ctx.nB_nat * Sig_R
    s_matter = s_b

    return dict(
        sigma=sigma, omega0=omega0, rho0=rho0, phi0=phi0,
        mu_eff_i=mu_eff_i, m_eff_i=m_eff_i,
        Sigma_R=Sig_R, mu_B=mu_B, mu_C=mu_C, mu_S=mu_S, mu_mu=mu_mu,
        mu_e=mu_e, mu_nue=mu_nue,
        eps=eps, P=P, s=s, n_tot=n_tot,
        n_e=ne, n_mu=nmu, n_nue=nnue,
        n_C=charge_tot, n_S=strangeness_tot,
        Y_C=charge_tot / ctx.nB_nat, Y_S=strangeness_tot / ctx.nB_nat,
        Y_C_baryons=charge_had / ctx.nB_nat,
        Y_S_baryons=strangeness / ctx.nB_nat,
        Y_Le=(ne + nnue) / ctx.nB_nat,
        densities=densities,
        mu_dot_n=mu_dot_n + lepton_dot_n,
        eps_matter=eps_matter, P_matter=P_matter, s_matter=s_matter,
        mu_dot_n_matter=mu_dot_n,
        eps_leptons=ee + emu + enue, P_leptons=Pe + Pmu + Pnue,
        s_leptons=se + smu + snue, mu_dot_n_leptons=lepton_dot_n,
    )


def _residual_and_jacobian(ctx, spec, T, analytic_jac):
    """The (residual, Jacobian) pair the octet solve should use.

    The reference pair is `residual` with no Jacobian, so MINPACK builds
    a forward difference: plain NumPy/SciPy, and the oracle the accelerated
    pair is judged against (CLAUDE.md section 9).

    With `analytic_jac` the exact Jacobian goes to MINPACK's hybrj, which is
    3-11x faster on a warm-started sweep; at T = 0 the residual AND the
    Jacobian come from the jitted kernel instead, machine-identical to the
    NumPy one. Both live in `backends/`, which section 5 makes optional, so a
    missing directory lands on the reference pair rather than raising.
    """
    if not analytic_jac or residual_jacobian is None:
        return residual, None
    if T == 0.0 and _NUMBA_OK:
        arrays = build_numba_arrays(ctx, spec)
        return (lambda xx, *_a: residual_t0_jit(xx, *arrays),
                lambda xx, *_a: jacobian_t0_jit(xx, *arrays))
    # T > 0: the JEL integrals do not jit, so the residual stays NumPy.
    return residual, residual_jacobian


def solve(par, n_B, flags, T=0.0, x0=None, charge_mode="neutral",
                Y_C=0.0, strange_mode="eq", Y_S=0.0, lepton_mode="transparent",
                Y_Le=0.0, yc_leptons=False,
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
    spec = mode_spec(charge_mode, Y_C, strange_mode, Y_S, lepton_mode, Y_Le,
                     yc_leptons)
    ctx = build_matter_ctx(par, n_B, flags, T=T)
    has_muS, has_muL = spec.is_fixed("S"), spec.is_fixed("L_e")
    # Lazy guess sequence: try the warm start x0 first and only build the
    # (expensive, un-jitted beta-eq) default guess if x0 is missing or its solve
    # doesn't converge. In a warm-started sweep the fallback is never evaluated,
    # which is the hot-path win — default_guess otherwise dominated the
    # jitted solve (~0.9 -> ~0.3 ms/pt at T=0, fixed-Y_C).
    def _guesses():
        if x0 is not None:
            yield x0
        yield default_guess(par, n_B, flags, T=T, has_muS=has_muS,
                                  has_muL=has_muL)
    res_fn, jac_fn = _residual_and_jacobian(ctx, spec, T, analytic_jac)
    sol = None
    for guess in _guesses():
        sol = root(res_fn, guess, args=(ctx, spec), jac=jac_fn,
                   method="hybr", tol=1e-13)
        res_max = max(abs(r) for r in res_fn(sol.x, ctx, spec))
        if res_max <= RESIDUAL_TOL:
            break
    else:
        raise RuntimeError(
            f"octet solve failed at n_B={n_B}, T={T} "
            f"(charge={charge_mode}, strange={strange_mode}): {sol.message} "
            f"(max residual {res_max:.2e}, tol {RESIDUAL_TOL:.0e})")

    st = assemble(sol.x, ctx, spec)
    if flags.photons and T > 0.0:
        # Photons carry no conserved charge and belong to no phase: they are
        # added to the totals and nowhere else.
        ph = photon_thermo(T)
        st["eps"] += ph.e * hc3
        st["P"] += ph.P * hc3
        st["s"] += ph.s * hc3
    # Thermal meson gas: additive Bose gas on top of the mean
    # field, evaluated at the converged charge/strange potentials and vector
    # fields. mu*_j n_j joins the HVH sum (the gas satisfies e+P = Ts+mu* n).
    if (flags.thermal_mesons or flags.thermal_vectors) and T > 0:
        mg = thermal_meson_thermo(
            par, n_B, st["mu_C"], st["mu_S"], st["omega0"], st["rho0"], T,
            include_pseudoscalars=flags.thermal_mesons,
            include_thermal_vectors=flags.thermal_vectors)
        st["eps"] += mg["e"] * hc3
        st["P"] += mg["P"] * hc3
        st["s"] += mg["s"] * hc3
        st["mu_dot_n"] += mg["mu_dot_n"] * hc3
        # The gas IS part of the phase (CLAUDE.md section 2), so it joins the
        # matter block as well as the totals.
        st["eps_matter"] += mg["e"] * hc3
        st["P_matter"] += mg["P"] * hc3
        st["s_matter"] += mg["s"] * hc3
        st["mu_dot_n_matter"] += mg["mu_dot_n"] * hc3
        condensation = mg["condensation"]
        # Bose-Einstein condensation is not implemented. Past mu*_j = m_j the
        # excited states saturate -- correctly, that IS the critical density --
        # but the particles beyond it go into the p = 0 state, carrying charge
        # and eps = m n_cond with NO pressure and NO entropy, and n_cond is a
        # new unknown rather than a function of mu*. `solve_bose_jel` caps mu
        # at m, so the gas silently stops absorbing charge and the entropy it
        # reports drifts (it turns NEGATIVE by mu*/m ~ 3). The state is refused
        # rather than returned wrong; `eos.sfho` refuses the same condition,
        # there as a status because its solvers report rather than raise.
        if condensation >= 1.0:
            raise ValueError(
                f"the thermal meson gas Bose-condenses at n_B={n_B}, T={T} "
                f"(max |mu*|/m = {condensation:.3f}); a condensate is not "
                f"implemented, so this state is outside the model")
    else:
        condensation = 0.0

    hvh_rel = (st["eps"] + st["P"] - T * st["s"] - st["mu_dot_n"]) / st["eps"]
    if check_consistency:
        if abs(hvh_rel) > HVH_RTOL:
            raise ValueError(
                f"Hugenholtz–Van Hove violated at n_B={n_B}, T={T} "
                f"(octet {charge_mode}/{strange_mode}): "
                f"|{hvh_rel:.2e}| > {HVH_RTOL:.0e}")
        if not spec.is_fixed("C"):
            # transparent: mu_n - mu_p = mu_e; trapped: = mu_e - mu_nue.
            beta_res = -st["mu_C"] - (st["mu_e"] - st["mu_nue"])
            if abs(beta_res) > 1e-6:
                raise ValueError(
                    f"beta-equilibrium condition violated at n_B={n_B}, "
                    f"T={T}: mu_n - mu_p - (mu_e - mu_nue) = {beta_res:.2e} MeV")

    matter = PhaseThermo(
        T=T, mu_B=st["mu_B"], mu_C=st["mu_C"], mu_S=st["mu_S"],
        fields={"sigma": st["sigma"], "omega0": st["omega0"],
                "rho0": st["rho0"], "phi0": st["phi0"]},
        densities=dict(st["densities"]),
        mu_i={name: species_potential(name, st["mu_B"], st["mu_C"], st["mu_S"])
              for name in st["densities"]},
        mu_eff_i=dict(st["mu_eff_i"]), m_eff_i=dict(st["m_eff_i"]),
        n_B=n_B, n_C=st["n_C"] / hc3, n_S=st["n_S"] / hc3,
        P=st["P_matter"] / hc3, eps=st["eps_matter"] / hc3,
        s=st["s_matter"] / hc3, mu_dot_n=st["mu_dot_n_matter"] / hc3,
        Sigma_R=st["Sigma_R"], condensation=condensation,
    )
    leptons = LeptonThermo(
        mu_e=st["mu_e"], mu_mu=st["mu_mu"], mu_nue=st["mu_nue"],
        densities={"e-": st["n_e"] / hc3, "mu-": st["n_mu"] / hc3,
                   "nu_e": st["n_nue"] / hc3},
        P=st["P_leptons"] / hc3, eps=st["eps_leptons"] / hc3,
        s=st["s_leptons"] / hc3, mu_dot_n=st["mu_dot_n_leptons"] / hc3,
    )
    conditions = {}
    if spec.is_fixed("C"):
        conditions["Y_C"] = st["Y_C"]
    if spec.is_fixed("S"):
        conditions["Y_S"] = st["Y_S"]
    if spec.is_fixed("L_e"):
        conditions["Y_Le"] = st["Y_Le"]
    return EoSPoint(
        mode=spec.name, n_B=n_B, T=T, conditions=conditions,
        matter=matter, leptons=leptons,
        P=st["P"] / hc3, eps=st["eps"] / hc3, s=st["s"] / hc3,
    )


#: Equilibrium mode -> the `solve` configuration it means. The fixed
#: fractions are filled in from `TableSpec.fixed` at build time.
#:
#: The names say what the matter is, and they match the modes `eos.mixed`
#: offers, so a purely hadronic table and a hybrid table are requested the
#: same way:
#:
#:   beta_eq_neutrinoless      charge-neutral beta equilibrium, neutrinos escape
#:   beta_eq_neutrino_trapped  ... with neutrinos trapped at fixed Y_Le
#:   fixed_YC                  fixed non-leptonic charge fraction
#:                             (the CompOSE general-purpose (nB, T, Y_q) slice)
#:   fixed_YC_YS               both fractions fixed
#:
#: Whether the neutralizing leptons are present is the orthogonal `leptons`
#: flag of CLAUDE.md section 3, carried beside the mode rather than folded
#: into its name: with leptons=False the matter is charged, which is what a
#: mixed-phase construction needs per pure phase before imposing global
#: neutrality. `takes_leptons` says which mode the flag applies to.
MODES = {
    "beta_eq_neutrinoless": dict(charge_mode="neutral"),
    "fixed_YC": dict(charge_mode="fixed", takes_leptons=True),
    "fixed_YC_YS": dict(charge_mode="fixed", strange_mode="fixed"),
    "beta_eq_neutrino_trapped": dict(charge_mode="neutral",
                                     lepton_mode="trapped"),
}

#: mode name -> the fixed fractions it consumes, either as a `TableSpec.axes`
#: grid or as a scalar in `TableSpec.fixed`. Mirrors
#: `eos.mixed.MODE_FRACTIONS`, which names the four modes both engines share.
MODE_FRACTIONS = {
    "beta_eq_neutrinoless": (), "fixed_YC": ("Y_C",),
    "fixed_YC_YS": ("Y_C", "Y_S"), "beta_eq_neutrino_trapped": ("Y_Le",),
}


def _mode_kwargs(mode, fixed, leptons=None):
    """The `solve` keywords a mode name, its fractions and the section 3
    `leptons` flag mean.

    `leptons=None` is the caller not naming the flag, which DD2 reads as its
    leptonless default; the beta-equilibrium reading of an explicit value is
    `eos.general.modes.resolve_leptons`'s.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {list(MODES)}")
    kw = dict(MODES[mode])
    leptons = resolve_leptons(mode, leptons, default=False)
    if kw.pop("takes_leptons", False):
        kw["yc_leptons"] = leptons
    elif leptons and not mode.startswith("beta_eq"):
        raise ValueError(
            f"leptons=True does not apply to mode {mode!r}; it selects the "
            f"neutralizing leptons of fixed_YC (fixed_YC_YS with leptons is "
            f"not wired; see docs/DEFERRED.md)")
    for key in MODE_FRACTIONS[mode]:
        if key not in fixed:
            raise ValueError(f"mode {mode!r} needs fixed[{key!r}]")
        kw[key] = fixed[key]
    return kw


def solve_beta_eq_neutrinoless(par, n_B, flags, T=0.0, x0=None,
                        check_consistency=True, analytic_jac=True):
    """
    Beta-equilibrium matter with the full active baryon set (
    mode 1; mu_S = mu_nue = 0, charge neutrality). Thin wrapper over solve.
    Reduces to the nucleon problem when flags.hyperons is False.
    """
    return solve(par, n_B, flags, T=T, x0=x0, charge_mode="neutral",
                       check_consistency=check_consistency,
                       analytic_jac=analytic_jac)


def solve_hadronic(par, flags, n_B, T=0.0, mode="beta_eq_neutrinoless",
                   leptons=None, x0=None, analytic_jac=True,
                   check_consistency=True, **fracs):
    """
    One hadronic point in a NAMED equilibrium mode — the counterpart of
    `eos.mixed.solve`, so both engines are driven the same way.

    mode  : one of `eos.dd2.MODES` — 'beta_eq_neutrinoless',
            'beta_eq_neutrino_trapped', 'fixed_YC', 'fixed_YC_YS'.
    leptons: for fixed_YC, whether the neutralizing leptons are present. The
            orthogonal flag of CLAUDE.md section 3, not part of the mode name.
            None leaves it at DD2's leptonless default; on a beta-equilibrium
            mode True is redundant and ignored and False raises.
    fracs : the fixed fractions the mode consumes, as keywords, e.g.
            Y_C=0.3 for 'fixed_YC' or Y_Le=0.4 for the trapped mode. Which
            keys each mode needs is `eos.dd2.MODE_FRACTIONS`.

    A thin dispatcher over `solve`, which implements every mode; this
    only turns the mode name into its argument set. Returns an `EoSPoint`.
    """
    return solve(par, n_B, flags, T=T, x0=x0,
                       analytic_jac=analytic_jac,
                       check_consistency=check_consistency,
                       **_mode_kwargs(mode, fracs, leptons))


def solve_fixed_yc(par, n_B, Y_C, flags, T=0.0, x0=None, Y_S=None,
                         leptons=False, check_consistency=True):
    """
    Fixed hadronic charge fraction Y_C. Two flavors:

    - leptons=False (2a, default): leptonless — the CompOSE general-purpose
      (nB,T,Yq) slicing; mu_C is the Lagrange multiplier for Y_C.
    - leptons=True (2b): populate electrons (+muons iff flags.muons) so the
      TOTAL is charge-neutral (n_e+n_mu = Y_C n_B, mu_e=mu_mu). The hadronic
      solve is identical to 2a; the leptons are a post-hoc neutraliser. Read
      Y_e / Y_mu off the result.

    Optionally also fix the strangeness fraction Y_S (adds mu_S; §1.7 mode 3),
    composing with either flavor.
    """
    strange_mode = "fixed" if Y_S is not None else "eq"
    return solve(par, n_B, flags, T=T, x0=x0, charge_mode="fixed",
                       Y_C=Y_C, strange_mode=strange_mode, Y_S=(Y_S or 0.0),
                       yc_leptons=leptons,
                       check_consistency=check_consistency)


def solve_beta_eq_neutrino_trapped(par, n_B, Y_Le, flags, T=0.0, x0=None,
                   check_consistency=True):
    """
    Neutrino-trapped matter at fixed electron lepton fraction
    Y_Le = (n_e + n_nue)/n_B. Charge-neutral, mu_nue unknown,
    electron-neutrinos included (mu_nue = mu_nue, mu_e = mu_nue - mu_C). The muon
    family stays transparent. Requires SpeciesFlags(neutrinos=True).
    """
    if not flags.neutrinos:
        raise ValueError("solve_beta_eq_neutrino_trapped requires "
                         "SpeciesFlags(neutrinos=True)")
    return solve(par, n_B, flags, T=T, x0=x0, charge_mode="neutral",
                       lepton_mode="trapped", Y_Le=Y_Le,
                       check_consistency=check_consistency)


def sweep(par, n_B_grid, flags, T=0.0, charge_mode="neutral", Y_C=0.0,
                strange_mode="eq", Y_S=0.0, lepton_mode="transparent", Y_Le=0.0,
                yc_leptons=False, max_bisect=6,
                stop_at_boundary=False, analytic_jac=True, max_skip=3):
    """
    Warm-started density sweep for any octet mode, with the same
    step-bisection continuation and scalar-collapse boundary handling as the
    beta-equilibrium case, which is what the default arguments select. See
    solve for the mode arguments. analytic_jac selects
    the eos_fast (exact-Jacobian) backend, on by default; False is the
    finite-difference reference path.

    `max_skip` is what separates the two reasons a point can fail, which
    otherwise look identical from here. Past the scalar-collapse boundary there
    is nothing left to solve, so EVERY remaining density fails and the sweep
    should end; a single density that misses because the continuation lost the
    basin is a hole, and ending there throws away a branch that is still there
    above it. Up to `max_skip` consecutive misses are therefore skipped — the
    next step is then a longer jump from the last good point, which `step`
    bisects — and only a run of them ends the sweep. Set it to 0 for the old
    stop-at-the-first-miss behaviour.

    A caller that needs to know which happened should read `m_eff` at the last
    returned point: at the boundary it has collapsed to ~0, and a sweep that
    merely gave up ends with a healthy effective mass.
    """
    has_phi = flags.phi_field and flags.hyperons
    has_muS = (strange_mode == "fixed")
    has_muL = (lepton_mode == "trapped")

    def solve_from(n_B, x0):
        return solve(par, n_B, flags, T=T, x0=x0, charge_mode=charge_mode,
                           Y_C=Y_C, strange_mode=strange_mode, Y_S=Y_S,
                           lepton_mode=lepton_mode, Y_Le=Y_Le, yc_leptons=yc_leptons,
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
                        warm_start(p_mid, has_phi, has_muS, has_muL),
                        depth + 1)

    points, x0, n_prev, misses = [], None, None, 0
    for n_B in n_B_grid:
        try:
            p = step(n_prev, n_B, x0, 0)
        except RuntimeError:
            if not (stop_at_boundary and points):
                raise
            misses += 1
            if misses > max_skip:
                break                  # a run of failures: this is the boundary
            continue                   # one miss: a hole, not the end
        misses = 0
        points.append(p)
        x0 = warm_start(p, has_phi, has_muS, has_muL)
        n_prev = n_B
    return points
