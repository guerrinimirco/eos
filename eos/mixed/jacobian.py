"""
mixed/jacobian.py
=================
Hand-assembled analytic Jacobian of the eta-mixed-phase residual
(docs/phase2/SPECIFICATION_AND_PLAN.md §4 milestone P7; CLAUDE.md §4). Same
contract as the Phase-1 DD2 Jacobian: an exact Jacobian supplied to the MINPACK
solver, finite-difference-verified, `_ref` (numeric-Jacobian) is the oracle.

The mixed residual couples two phases that are each an implicit solve, so the
Jacobian is assembled from per-phase derivative blocks:

  - **Quark block** dn_a^Q/dmu_b^Q is analytic: vMIT has ONE vector field
    V = a*hc*(n_u+n_d+n_s), so the flavor susceptibility is Sherman-Morrison,
    chi_fl = diag(kappa) - g kappa kappa^T / (1 + g K)  (g=a*hc, K=sum kappa,
    kappa_f = dn_f/dmu_eff from the JEL density), rotated to the (B,C,S) charge
    basis by chi_ab = M^T chi_fl M. dP^Q/dmu_a = n_a (Gibbs-Duhem).

  - **Hadronic block** d(n_B,n_C,n_S,P,mu_B)^H/d(mu_tilde_B,mu_C,mu_S) is the
    implicit-function theorem on the adapter's SOLVE-FREE residual: at the
    converged phase state, dx/dmu = -(dR/dx)^-1 (dR/dmu), then chain through the
    (solve-free) assembly. Both leaf Jacobians are finite-differenced on plain
    function evals (no root-find) -- this handles Sigma^R and the
    density-dependent couplings automatically, which hand-differentiating the
    JEL core does not (the report abandoned that trace, D3).
    # ponytail: IFT with FD leaf-partials of solve-free evals; the assembly
    # (IFT + Sherman-Morrison + the regime chain rule below) is what's analytic
    # and what the FD-agreement test validates. Swap leaf FD for hand JEL
    # derivatives only if a profile says the block eval matters.

  - **Electrons** dn_e/dmu_e (FD on the leptonic thermo), dP_e/dmu_e = n_e.

The outer assembly is regime-driven, mirroring mixed_residual row-for-row so the
four modes are configurations of one Jacobian (spec §1.5), not four.

Units: fm-based (n [fm^-3], P [MeV/fm^3], mu [MeV]); the block is dn/dmu
[fm^-3/MeV], dP/dmu [fm^-3], dmu/dmu [-].
"""
import numpy as np

from eos.general.physics_constants import hc, hc3
from eos.general.thermodynamics_leptons import electron_thermo, neutrino_thermo
from eos.dd2.physics.octet import build_octet_ctx, assemble_octet
from eos.mixed.charges import Regime
from eos.mixed.phases import hadronic_phase, quark_phase, _hadronic_residual
from eos.mixed.residual import has_leptons, _quark_mus_from_charges
from eos.vmit.thermodynamics_quarks import compute_quark_density

#: mu_flavor = _M @ mu_charge, with mu_charge = (mu_B, mu_C, mu_S); so the charge
#: susceptibility is chi_ab = _M^T chi_flavor _M (n_charge = _M^T n_flavor).
_M = np.array([[1.0 / 3.0,  2.0 / 3.0, 0.0],
               [1.0 / 3.0, -1.0 / 3.0, 0.0],
               [1.0 / 3.0, -1.0 / 3.0, 1.0]])


def _dn_dmu(f, x, T, m=None, h_min=1e-2, rel=1e-4):
    """Central FD of a 1-D density f(x[, T, m]) w.r.t. x."""
    h = max(h_min, rel * abs(x))
    if m is None:
        return (f(x + h, T).n - f(x - h, T).n) / (2.0 * h)
    return (f(x + h, T, m) - f(x - h, T, m)) / (2.0 * h)


def _electron_kappa(mu_e, T):
    return _dn_dmu(electron_thermo, mu_e, T) if mu_e != 0.0 else 0.0


def _neutrino_kappa(mu_L, T):
    return _dn_dmu(neutrino_thermo, mu_L, T) if mu_L != 0.0 else 0.0


def _quark_block(mu_B_Q, mu_C_Q, mu_S, T, params, th_Q):
    """4x3 block [n_B,n_C,n_S,P] rows x [mu_B,mu_C,mu_S] cols for the quark phase."""
    mu_u, mu_d, mu_s = _quark_mus_from_charges(mu_B_Q, mu_C_Q, mu_S)
    N = th_Q.densities["u"] + th_Q.densities["d"] + th_Q.densities["s"]
    g = params.a * hc
    V = g * N
    kap = np.array([
        _dn_dmu(compute_quark_density, mu_u - V, T, params.m_u),
        _dn_dmu(compute_quark_density, mu_d - V, T, params.m_d),
        _dn_dmu(compute_quark_density, mu_s - V, T, params.m_s),
    ])
    K = kap.sum()
    chi_fl = np.diag(kap) - g * np.outer(kap, kap) / (1.0 + g * K)
    chi_charge = _M.T @ chi_fl @ _M                    # rows/cols (B,C,S)
    P_row = np.array([th_Q.n_B, th_Q.n_C, th_Q.n_S])   # Gibbs-Duhem dP/dmu_a=n_a
    return np.vstack([chi_charge, P_row])              # 4x3


def _hadronic_block(par, flags, mu_tB, mu_C, mu_S, T, state):
    """5x3 block [n_B,n_C,n_S,P,mu_B] rows x [mu_tilde_B,mu_C,mu_S] cols via IFT."""
    ctx = state["ctx"]
    strange_mode = state["strange_mode"]
    x0 = np.array(state["x_phase"], dtype=float)
    n_f = len(x0) - 1                                   # field count (nB_nat is last)

    def R(xp, mu):
        return np.array(_hadronic_residual(list(xp), ctx, par, flags,
                                           mu[0], mu[1], mu[2]))

    def Out(xp, mu):
        nB_nat = xp[n_f]
        a_ctx = build_octet_ctx(par, nB_nat / hc3, flags, T=T, charge_mode="fixed",
                                strange_mode=strange_mode, Y_C=0.0, Y_S=0.0)
        x_oct = list(xp[:n_f]) + [mu[0], mu[1]]
        if a_ctx.has_muS:
            x_oct.append(mu[2])
        st = assemble_octet(x_oct, a_ctx)
        nB = nB_nat / hc3
        return np.array([nB, st["Y_C"] * nB, st["Y_S"] * nB,
                         st["P"] / hc3, st["mu_B"]])

    mu0 = [mu_tB, mu_C, mu_S]
    nx = len(x0)
    dR_dx = np.zeros((nx, nx))
    dOut_dx = np.zeros((5, nx))
    for i in range(nx):
        h = max(1e-6, 1e-6 * abs(x0[i]))
        xp, xm = x0.copy(), x0.copy()
        xp[i] += h; xm[i] -= h
        dR_dx[:, i] = (R(xp, mu0) - R(xm, mu0)) / (2.0 * h)
        dOut_dx[:, i] = (Out(xp, mu0) - Out(xm, mu0)) / (2.0 * h)
    dR_dmu = np.zeros((nx, 3))
    dOut_dmu = np.zeros((5, 3))
    for j in range(3):
        h = max(1e-3, 1e-5 * abs(mu0[j]))
        mp, mm = list(mu0), list(mu0)
        mp[j] += h; mm[j] -= h
        dR_dmu[:, j] = (R(x0, mp) - R(x0, mm)) / (2.0 * h)
        dOut_dmu[:, j] = (Out(x0, mp) - Out(x0, mm)) / (2.0 * h)
    dx_dmu = np.linalg.solve(dR_dx, -dR_dmu)            # nx x 3
    return dOut_dmu + dOut_dx @ dx_dmu                  # 5x3


def _phase_inputs(x, ctx):
    """Replicate evaluate_phases' input derivation; return the phase potentials
    and the per-slot sensitivities of mu_C_H / mu_C_Q needed by the chain rule."""
    spec, eta, slots = ctx.spec, ctx.eta, ctx.slots
    d = dict(zip(slots, x))
    mu_eL_H = d.get("mu_eL_H", 0.0)
    mu_eL_Q = d.get("mu_eL_Q", 0.0)
    mu_eG = d.get("mu_eG", 0.0)
    mu_L = d.get("mu_L", 0.0)
    mu_S = d.get("mu_S", 0.0)
    if spec.C is Regime.NOT_CONSERVED:
        mu_C_H = mu_L - (eta * mu_eL_H + (1.0 - eta) * mu_eG)
        mu_C_Q = mu_L - (eta * mu_eL_Q + (1.0 - eta) * mu_eG)
    elif spec.yc_leptons:
        mu_C_H, mu_C_Q = d["mu_C_H"], d["mu_C_Q"]
    else:
        mu_C_H = mu_C_Q = d["mu_C"]
    return d, mu_C_H, mu_C_Q, mu_S


def _sens(slot, name, val=1.0):
    """1.0*val if this column IS `name`, else 0.0 (Kronecker for the chain rule)."""
    return val if slot == name else 0.0


def mixed_jacobian(x, ctx):
    """Analytic Jacobian d(mixed_residual)/dx (len(res) x len(slots)), regime-
    driven and row-aligned with mixed_residual. Supplied to hybr as `jac`."""
    spec, eta, slots = ctx.spec, ctx.eta, ctx.slots
    lep = has_leptons(spec)
    ns, mus = ctx.n_scale, ctx.mu_scale

    d, mu_C_H, mu_C_Q, mu_S = _phase_inputs(x, ctx)
    th_H, state_H = hadronic_phase(ctx.par, ctx.flags, d["mu_tilde_B_H"], mu_C_H,
                                   mu_S, T=ctx.T, n_B_guess=ctx.n_B_guess,
                                   return_state=True)
    mu_u, mu_d, mu_s = _quark_mus_from_charges(d["mu_B_Q"], mu_C_Q, mu_S)
    th_Q = quark_phase(mu_u, mu_d, mu_s, T=ctx.T, params=ctx.vmit_params)

    bH = _hadronic_block(ctx.par, ctx.flags, d["mu_tilde_B_H"], mu_C_H, mu_S,
                         ctx.T, state_H)                # 5x3: [nB,nC,nS,P,muB]
    bQ = _quark_block(d["mu_B_Q"], mu_C_Q, mu_S, ctx.T, ctx.vmit_params, th_Q)  # 4x3

    # Electron / neutrino densities + susceptibilities where active.
    mu_eL_H, mu_eL_Q, mu_eG = d.get("mu_eL_H", 0.0), d.get("mu_eL_Q", 0.0), d.get("mu_eG", 0.0)
    on_L = lep and eta > 0.0
    on_G = lep and eta < 1.0
    n_eL_H = electron_thermo(mu_eL_H, ctx.T).n if on_L else 0.0
    n_eL_Q = electron_thermo(mu_eL_Q, ctx.T).n if on_L else 0.0
    k_eL_H = _electron_kappa(mu_eL_H, ctx.T) if on_L else 0.0
    k_eL_Q = _electron_kappa(mu_eL_Q, ctx.T) if on_L else 0.0
    k_eG = _electron_kappa(mu_eG, ctx.T) if on_G else 0.0
    k_nue = _neutrino_kappa(d.get("mu_L", 0.0), ctx.T) if spec.L_e is Regime.GLOBAL else 0.0

    chi = d["chi"]
    P_H_eff = th_H.P + eta * (n_eL_H and electron_thermo(mu_eL_H, ctx.T).P)
    P_Q_eff = th_Q.P + eta * (n_eL_Q and electron_thermo(mu_eL_Q, ctx.T).P)
    Ps = max(abs(P_H_eff), abs(P_Q_eff), 1.0)          # frozen scale (see module doc)

    def col(sj):
        """One Jacobian column: derivative of every residual row w.r.t. slot sj."""
        # --- phase-input sensitivities to this slot ---
        d_mutB = _sens(sj, "mu_tilde_B_H")
        d_muBQ = _sens(sj, "mu_B_Q")
        d_chi = _sens(sj, "chi")
        d_muS = _sens(sj, "mu_S")
        if spec.C is Regime.NOT_CONSERVED:
            d_muCH = (_sens(sj, "mu_L") - eta * _sens(sj, "mu_eL_H")
                      - (1.0 - eta) * _sens(sj, "mu_eG"))
            d_muCQ = (_sens(sj, "mu_L") - eta * _sens(sj, "mu_eL_Q")
                      - (1.0 - eta) * _sens(sj, "mu_eG"))
        elif spec.yc_leptons:
            d_muCH, d_muCQ = _sens(sj, "mu_C_H"), _sens(sj, "mu_C_Q")
        else:
            d_muCH = d_muCQ = _sens(sj, "mu_C")

        # --- phase output derivatives via the blocks ---
        def hOut(r):  # bH row r contracted with (mu_tB, mu_C_H, mu_S) sensitivities
            return bH[r, 0] * d_mutB + bH[r, 1] * d_muCH + bH[r, 2] * d_muS
        def qOut(r):
            return bQ[r, 0] * d_muBQ + bQ[r, 1] * d_muCQ + bQ[r, 2] * d_muS
        dnB_H, dnC_H, dnS_H, dP_H, dmuB_H = (hOut(0), hOut(1), hOut(2), hOut(3), hOut(4))
        dnB_Q, dnC_Q, dnS_Q, dP_Q = (qOut(0), qOut(1), qOut(2), qOut(3))

        # electron responses (dP_e = n_e * dmu_e ; dn_e = kappa * dmu_e)
        dP_eL_H = n_eL_H * _sens(sj, "mu_eL_H")
        dP_eL_Q = n_eL_Q * _sens(sj, "mu_eL_Q")
        dn_eL_H = k_eL_H * _sens(sj, "mu_eL_H")
        dn_eL_Q = k_eL_Q * _sens(sj, "mu_eL_Q")
        dn_eG = k_eG * _sens(sj, "mu_eG")

        rows = [
            (dmuB_H - d_muBQ) / mus,                                        # B (3.24)
            ((1.0 - chi) * dnB_H + chi * dnB_Q
             + (th_Q.n_B - th_H.n_B) * d_chi) / ns,                        # baryon (3.6)
            (dP_H + eta * dP_eL_H - dP_Q - eta * dP_eL_Q) / Ps,            # mechanical
        ]
        if spec.C is Regime.GLOBAL:
            rows.append(((1.0 - chi) * dnC_H + chi * dnC_Q
                         + (th_Q.n_C - th_H.n_C) * d_chi) / ns)            # Y_C cons
            if spec.yc_leptons:
                rows.append((_sens(sj, "mu_C_H") + eta * _sens(sj, "mu_eL_H")
                             - _sens(sj, "mu_C_Q") - eta * _sens(sj, "mu_eL_Q")) / mus)
        if spec.S is Regime.GLOBAL:
            rows.append(((1.0 - chi) * dnS_H + chi * dnS_Q
                         + (th_Q.n_S - th_H.n_S) * d_chi) / ns)            # Y_S cons
        if spec.L_e is Regime.GLOBAL:
            dne_avg = (eta * ((1.0 - chi) * dn_eL_H + chi * dn_eL_Q
                              + (n_eL_Q - n_eL_H) * d_chi)
                       + (1.0 - eta) * dn_eG)
            rows.append((dne_avg + k_nue * _sens(sj, "mu_L")) / ns)       # Y_L cons
        if on_L:
            rows.append((dnC_H - dn_eL_H) / ns)                           # local H (3.8)
            rows.append((dnC_Q - dn_eL_Q) / ns)                           # local Q (3.9)
        if on_G:
            rows.append(((1.0 - chi) * dnC_H + chi * dnC_Q
                         + (th_Q.n_C - th_H.n_C) * d_chi - dn_eG) / ns)   # global (3.10)
        return rows

    return np.array([col(sj) for sj in slots]).T        # (n_res x n_slots)
