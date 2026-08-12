"""
mixed/equilibrium/jacobian.py
=============================
Hand-assembled analytic Jacobian of the mixed-phase residual.

*Internal module.* Supplied to the root finder by `eos.mixed.solve_mixed`
when `analytic_jac=True`; the numeric-Jacobian path remains the correctness
oracle (CLAUDE.md §4), and a finite-difference agreement test guards this file.

The residual couples two phases that are each themselves an implicit solve, so
the Jacobian is built from per-phase derivative blocks and then contracted
through the same regime rules the residual uses — row for row, so the four
named modes share one Jacobian rather than needing four.

**Quark block.** dn_a^Q/dmu_b^Q is analytic. vMIT has a single vector field
V = a*hc*(n_u+n_d+n_s), so every flavor feels the same shift and the flavor
susceptibility is a rank-one update of the free one — Sherman-Morrison:

    chi_flavor = diag(kappa) - g kappa kappa^T / (1 + g K)

with g = a*hc, kappa_f = dn_f/dmu_eff and K = sum_f kappa_f. Rotating into the
(B, C, S) charge basis gives chi_ab = M^T chi_flavor M, and Gibbs-Duhem gives
dP^Q/dmu_a = n_a.

**Hadronic block.** d(n_B, n_C, n_S, P, mu_B)^H/d(mu_tilde_B, mu_C, mu_S) comes
from the implicit function theorem applied to the adapter's *solve-free*
residual: at the converged phase state, dx/dmu = -(dR/dx)^-1 (dR/dmu), then the
chain rule through the (also solve-free) assembly. The two leaf Jacobians are
finite-differenced on plain function evaluations, never on a root find.
# ponytail: IFT with finite-differenced leaf partials. The assembly — IFT plus
# Sherman-Morrison plus the regime chain rule — is what is analytic and what
# the FD-agreement test validates. Differentiating the JEL Fermi core by hand
# buys little and is what the Phase-1 autodiff attempt foundered on; do it only
# if a profile says the leaf evaluations matter.

**Leptons.** dn/dmu_e is finite-differenced on the combined electron+muon
density of a neutrality domain (muons track mu_e one-for-one, so their
susceptibility simply adds), and dP/dmu_e = n by Gibbs-Duhem.

Units: fm-based, so dn/dmu is [fm^-3/MeV], dP/dmu is [fm^-3], dmu/dmu is
dimensionless.
"""
import numpy as np

from eos.general.physics_constants import hc, hc3
from eos.general.thermodynamics_leptons import muon_thermo, neutrino_thermo
from eos.dd2.thermodynamics import (
    assemble as dd2_assemble, self_consistency_residual,
)
from eos.mixed.equilibrium.charges import Regime
from eos.mixed.equilibrium.residual import (
    charged_leptons, has_leptons, _quark_mus_from_charges,
)
from eos.mixed.adapters import hadronic_phase, quark_phase
from eos.vmit.thermodynamics import compute_quark_density

#: mu_flavor = _M @ mu_charge for mu_charge = (mu_B, mu_C, mu_S). Since
#: n_charge = _M^T n_flavor, the charge susceptibility is _M^T chi_flavor _M.
_M = np.array([[1.0 / 3.0,  2.0 / 3.0, 0.0],
               [1.0 / 3.0, -1.0 / 3.0, 0.0],
               [1.0 / 3.0, -1.0 / 3.0, 1.0]])


def _fd_quark_kappa(mu_eff, T, m, h_min=1e-2, rel=1e-4):
    """dn/dmu_eff of one quark flavor, central finite difference."""
    h = max(h_min, rel * abs(mu_eff))
    return (compute_quark_density(mu_eff + h, T, m)
            - compute_quark_density(mu_eff - h, T, m)) / (2.0 * h)


def _lepton_kappa(mu_e, T, muons, mu_nue=0.0, h_min=1e-2, rel=1e-4):
    """d(n_e + n_mu)/dmu_e for one neutrality domain."""
    if mu_e == 0.0:
        return 0.0
    h = max(h_min, rel * abs(mu_e))
    return (charged_leptons(mu_e + h, T, muons, mu_nue).n
            - charged_leptons(mu_e - h, T, muons, mu_nue).n) / (2.0 * h)


def _muon_kappa(mu_e, T, muons, mu_nue=0.0, h_min=1e-2, rel=1e-4):
    """dn_mu/dmu_mu for one neutrality domain (0 when muons are off).

    Needed on its own because mu_mu = mu_e - mu_nue: with trapped neutrinos the
    muon density responds to mu_nue as well as to mu_e, with the opposite sign.
    """
    if not muons:
        return 0.0
    mu_mu = mu_e - mu_nue
    if mu_mu == 0.0:
        return 0.0
    h = max(h_min, rel * abs(mu_mu))
    return (muon_thermo(mu_mu + h, T).n - muon_thermo(mu_mu - h, T).n) / (2.0 * h)


def _neutrino_kappa(mu_nue, T, h_min=1e-2, rel=1e-4):
    if mu_nue == 0.0:
        return 0.0
    h = max(h_min, rel * abs(mu_nue))
    return (neutrino_thermo(mu_nue + h, T).n
            - neutrino_thermo(mu_nue - h, T).n) / (2.0 * h)


def _quark_block(mu_B_Q, mu_C_Q, mu_S, T, params, th_Q):
    """4x3 block: rows [n_B, n_C, n_S, P], columns [mu_B, mu_C, mu_S]."""
    mu_u, mu_d, mu_s = _quark_mus_from_charges(mu_B_Q, mu_C_Q, mu_S)
    N = th_Q.densities["u"] + th_Q.densities["d"] + th_Q.densities["s"]
    g = params.a * hc
    V = g * N
    kap = np.array([
        _fd_quark_kappa(mu_u - V, T, params.m_u),
        _fd_quark_kappa(mu_d - V, T, params.m_d),
        _fd_quark_kappa(mu_s - V, T, params.m_s),
    ])
    K = kap.sum()
    chi_fl = np.diag(kap) - g * np.outer(kap, kap) / (1.0 + g * K)
    chi_charge = _M.T @ chi_fl @ _M                    # rows/cols (B, C, S)
    P_row = np.array([th_Q.n_B, th_Q.n_C, th_Q.n_S])   # Gibbs-Duhem
    return np.vstack([chi_charge, P_row])              # 4x3


def _hadronic_block(par, flags, mu_tB, mu_C, mu_S, T, state):
    """5x3 block: rows [n_B, n_C, n_S, P, mu_B], columns
    [mu_tilde_B, mu_C, mu_S], via the implicit function theorem."""
    ctx = state["ctx"]
    x0 = np.array(state["x_phase"], dtype=float)
    n_f = len(x0) - 1                                   # field count (nB_nat last)

    def R(xp, mu):
        return np.array(self_consistency_residual(list(xp), par, ctx,
                                                  mu[0], mu[1], mu[2]))

    def Out(xp, mu):
        # Re-run the residual first so ctx carries this trial point's
        # density-dependent couplings, then assemble on the same ctx.
        self_consistency_residual(list(xp), par, ctx, mu[0], mu[1], mu[2])
        fields = list(xp[:n_f]) + [0.0] * (4 - n_f)     # sigma, omega0, rho0, phi0
        st = dd2_assemble(par, ctx, fields[0], fields[1], fields[2], fields[3],
                          mu[0], mu[1], mu[2])
        # The density row is the UNKNOWN nB_nat, not the species sum the
        # assembly reports. The two coincide at the solution -- that is what
        # the density row of the residual imposes -- but this is a finite
        # difference taken AWAY from it, and the implicit function theorem
        # wants the outputs as functions of the unknowns. Using the species
        # sum here changes the Jacobian, and shows up downstream as a tidal
        # deformability that disagrees between TOV backends.
        nB = xp[n_f] / hc3
        return np.array([nB, st.Y_C * nB, st.Y_S * nB, st.P, st.mu_B])

    mu0 = [mu_tB, mu_C, mu_S]
    nx = len(x0)
    dR_dx = np.zeros((nx, nx))
    dOut_dx = np.zeros((5, nx))
    for i in range(nx):
        h = max(1e-6, 1e-6 * abs(x0[i]))
        xp, xm = x0.copy(), x0.copy()
        xp[i] += h
        xm[i] -= h
        dR_dx[:, i] = (R(xp, mu0) - R(xm, mu0)) / (2.0 * h)
        dOut_dx[:, i] = (Out(xp, mu0) - Out(xm, mu0)) / (2.0 * h)
    dR_dmu = np.zeros((nx, 3))
    dOut_dmu = np.zeros((5, 3))
    for j in range(3):
        h = max(1e-3, 1e-5 * abs(mu0[j]))
        mp, mm = list(mu0), list(mu0)
        mp[j] += h
        mm[j] -= h
        dR_dmu[:, j] = (R(x0, mp) - R(x0, mm)) / (2.0 * h)
        dOut_dmu[:, j] = (Out(x0, mp) - Out(x0, mm)) / (2.0 * h)
    dx_dmu = np.linalg.solve(dR_dx, -dR_dmu)            # nx x 3
    return dOut_dmu + dOut_dx @ dx_dmu                  # 5x3


def _phase_potentials(x, ctx):
    """The per-phase charge potentials the residual would derive from `x`."""
    spec, eta, slots = ctx.spec, ctx.eta, ctx.slots
    d = dict(zip(slots, x))
    mu_nue = d.get("mu_nue", 0.0)
    mu_S = d.get("mu_S", 0.0)
    if spec.C is Regime.NOT_CONSERVED:
        mu_C_H = mu_nue - (eta * d.get("mu_eL_H", 0.0)
                         + (1.0 - eta) * d.get("mu_eG", 0.0))
        mu_C_Q = mu_nue - (eta * d.get("mu_eL_Q", 0.0)
                         + (1.0 - eta) * d.get("mu_eG", 0.0))
    elif spec.yc_leptons:
        mu_C_H, mu_C_Q = d["mu_C_H"], d["mu_C_Q"]
    else:
        mu_C_H = mu_C_Q = d["mu_C"]
    return d, mu_C_H, mu_C_Q, mu_S


def _sens(slot, name, val=1.0):
    """Kronecker delta for the chain rule: val if this column is `name`."""
    return val if slot == name else 0.0


def mixed_jacobian(x, ctx):
    """d(mixed_residual)/dx, shape (n_residuals x n_slots).

    Regime-driven and row-aligned with `mixed_residual`, so the two stay in
    step by construction rather than by convention.
    """
    spec, eta, slots = ctx.spec, ctx.eta, ctx.slots
    lep = has_leptons(spec)
    muons = bool(ctx.flags.muons)
    ns, mus = ctx.n_scale, ctx.mu_scale

    d, mu_C_H, mu_C_Q, mu_S = _phase_potentials(x, ctx)
    th_H, state_H = hadronic_phase(ctx.par, ctx.flags, d["mu_tilde_B_H"], mu_C_H,
                                   mu_S, T=ctx.T, n_B_guess=ctx.n_B_guess,
                                   x0=ctx.hadronic_seed(), return_state=True)
    mu_u, mu_d, mu_s = _quark_mus_from_charges(d["mu_B_Q"], mu_C_Q, mu_S)
    th_Q = quark_phase(mu_u, mu_d, mu_s, T=ctx.T, params=ctx.vmit_params)

    bH = _hadronic_block(ctx.par, ctx.flags, d["mu_tilde_B_H"], mu_C_H, mu_S,
                         ctx.T, state_H)                 # 5x3
    bQ = _quark_block(d["mu_B_Q"], mu_C_Q, mu_S, ctx.T, ctx.vmit_params, th_Q)

    mu_nue = d.get("mu_nue", 0.0)
    mu_eL_H, mu_eL_Q = d.get("mu_eL_H", 0.0), d.get("mu_eL_Q", 0.0)
    mu_eG = d.get("mu_eG", 0.0)
    on_L = lep and eta > 0.0
    on_G = lep and eta < 1.0
    L_H = charged_leptons(mu_eL_H, ctx.T, muons, mu_nue) if on_L else None
    L_Q = charged_leptons(mu_eL_Q, ctx.T, muons, mu_nue) if on_L else None
    G = charged_leptons(mu_eG, ctx.T, muons, mu_nue) if on_G else None
    k_L_H = _lepton_kappa(mu_eL_H, ctx.T, muons, mu_nue) if on_L else 0.0
    k_L_Q = _lepton_kappa(mu_eL_Q, ctx.T, muons, mu_nue) if on_L else 0.0
    k_G = _lepton_kappa(mu_eG, ctx.T, muons, mu_nue) if on_G else 0.0
    k_nue = (_neutrino_kappa(mu_nue, ctx.T)
             if spec.L_e is Regime.GLOBAL else 0.0)
    # The Y_Le row counts ELECTRONS only (muons carry no electron lepton
    # number), so it needs the electron-only susceptibility.
    ke_L_H = _lepton_kappa(mu_eL_H, ctx.T, False) if on_L else 0.0
    ke_L_Q = _lepton_kappa(mu_eL_Q, ctx.T, False) if on_L else 0.0
    ke_G = _lepton_kappa(mu_eG, ctx.T, False) if on_G else 0.0
    # mu_mu = mu_e - mu_nue, so with trapped neutrinos every muon quantity also
    # responds to mu_nue, with the opposite sign.
    km_L_H = _muon_kappa(mu_eL_H, ctx.T, muons, mu_nue) if on_L else 0.0
    km_L_Q = _muon_kappa(mu_eL_Q, ctx.T, muons, mu_nue) if on_L else 0.0
    km_G = _muon_kappa(mu_eG, ctx.T, muons, mu_nue) if on_G else 0.0

    chi = d["chi"]
    n_L_H, P_L_H = (L_H.n, L_H.P) if on_L else (0.0, 0.0)
    n_L_Q, P_L_Q = (L_Q.n, L_Q.P) if on_L else (0.0, 0.0)
    n_G = G.n if on_G else 0.0
    ne_L_H = L_H.n_e if on_L else 0.0
    ne_L_Q = L_Q.n_e if on_L else 0.0
    nm_L_H = L_H.n_mu if on_L else 0.0
    nm_L_Q = L_Q.n_mu if on_L else 0.0
    P_H_eff = th_H.P + eta * P_L_H
    P_Q_eff = th_Q.P + eta * P_L_Q
    Ps = max(abs(P_H_eff), abs(P_Q_eff), 1.0)

    def col(sj):
        """Derivative of every residual row with respect to slot `sj`."""
        d_mutB = _sens(sj, "mu_tilde_B_H")
        d_muBQ = _sens(sj, "mu_B_Q")
        d_chi = _sens(sj, "chi")
        d_muS = _sens(sj, "mu_S")
        if spec.C is Regime.NOT_CONSERVED:
            d_muCH = (_sens(sj, "mu_nue") - eta * _sens(sj, "mu_eL_H")
                      - (1.0 - eta) * _sens(sj, "mu_eG"))
            d_muCQ = (_sens(sj, "mu_nue") - eta * _sens(sj, "mu_eL_Q")
                      - (1.0 - eta) * _sens(sj, "mu_eG"))
        elif spec.yc_leptons:
            d_muCH, d_muCQ = _sens(sj, "mu_C_H"), _sens(sj, "mu_C_Q")
        else:
            d_muCH = d_muCQ = _sens(sj, "mu_C")

        def hOut(r):   # bH row r contracted with the H input sensitivities
            return bH[r, 0] * d_mutB + bH[r, 1] * d_muCH + bH[r, 2] * d_muS

        def qOut(r):
            return bQ[r, 0] * d_muBQ + bQ[r, 1] * d_muCQ + bQ[r, 2] * d_muS

        dnB_H, dnC_H, dnS_H, dP_H, dmuB_H = (hOut(0), hOut(1), hOut(2),
                                             hOut(3), hOut(4))
        dnB_Q, dnC_Q, dnS_Q, dP_Q = qOut(0), qOut(1), qOut(2), qOut(3)

        # Lepton responses: dP = sum n dmu (Gibbs-Duhem), dn = sum kappa dmu.
        # The muon terms pick up -dmu_L because mu_mu = mu_e - mu_nue.
        d_muL = _sens(sj, "mu_nue")
        dP_L_H = n_L_H * _sens(sj, "mu_eL_H") - nm_L_H * d_muL
        dP_L_Q = n_L_Q * _sens(sj, "mu_eL_Q") - nm_L_Q * d_muL
        dn_L_H = k_L_H * _sens(sj, "mu_eL_H") - km_L_H * d_muL
        dn_L_Q = k_L_Q * _sens(sj, "mu_eL_Q") - km_L_Q * d_muL
        dn_G = k_G * _sens(sj, "mu_eG") - km_G * d_muL

        rows = [
            (dmuB_H - d_muBQ) / mus,
            ((1.0 - chi) * dnB_H + chi * dnB_Q
             + (th_Q.n_B - th_H.n_B) * d_chi) / ns,
            (dP_H + eta * dP_L_H - dP_Q - eta * dP_L_Q) / Ps,
        ]
        if spec.C is Regime.GLOBAL:
            rows.append(((1.0 - chi) * dnC_H + chi * dnC_Q
                         + (th_Q.n_C - th_H.n_C) * d_chi) / ns)
            if spec.yc_leptons:
                rows.append((_sens(sj, "mu_C_H") + eta * _sens(sj, "mu_eL_H")
                             - _sens(sj, "mu_C_Q")
                             - eta * _sens(sj, "mu_eL_Q")) / mus)
        if spec.S is Regime.GLOBAL:
            rows.append(((1.0 - chi) * dnS_H + chi * dnS_Q
                         + (th_Q.n_S - th_H.n_S) * d_chi) / ns)
        if spec.L_e is Regime.GLOBAL:
            dne_avg = (eta * ((1.0 - chi) * ke_L_H * _sens(sj, "mu_eL_H")
                              + chi * ke_L_Q * _sens(sj, "mu_eL_Q")
                              + (ne_L_Q - ne_L_H) * d_chi)
                       + (1.0 - eta) * ke_G * _sens(sj, "mu_eG"))
            rows.append((dne_avg + k_nue * _sens(sj, "mu_nue")) / ns)
        if on_L:
            rows.append((dnC_H - dn_L_H) / ns)
            rows.append((dnC_Q - dn_L_Q) / ns)
        if on_G:
            rows.append(((1.0 - chi) * dnC_H + chi * dnC_Q
                         + (th_Q.n_C - th_H.n_C) * d_chi - dn_G) / ns)
        return rows

    return np.array([col(sj) for sj in slots]).T        # n_res x n_slots
