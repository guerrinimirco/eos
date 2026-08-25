"""
responses_jac.py
================
Thermodynamic responses wired onto the analytic Jacobian: the fast flavor of
what `responses.py` computes by finite difference, which is the independent
oracle it is validated against.

Two Jacobian products:

- **Susceptibilities** chi_ab = dn_a/dmu_b (a,b in B,C,S) — the CompOSE
  second-derivative block — are read straight off residual_jacobian via the
  field-response identity chi = dn/dmu|_F - (dn/dF)(dG/dF)^-1(dG/dmu).

- **c_s^2, C_V, C_P** come from Jacobian *tangent* sensitivities along the
  sequence: at the converged x0 the neighbour at a perturbed parameter is
  predicted by one Newton step dx = -J^-1 R_perturbed(x0) (the tangent
  predictor) instead of a full re-solve, then assembled. Central differences in
  n_B and T give d(P,eps,s)/d(n_B,T), and the standard thermodynamic
  combinations give the three responses.
"""
import numpy as np

from eos.general.physics_constants import hc3
from eos.general.modes import beta_eq_neutrinoless
from eos.dd2.thermodynamics import build_matter_ctx
from eos.dd2.solver import (
    assemble, mode_spec, residual, warm_start,
    solve_beta_eq_neutrinoless,
)
from eos.dd2.backends.jacobian import residual_jacobian

#: Every quantity here is taken along the beta-equilibrium sequence.
_BETA = beta_eq_neutrinoless()


def _state(par, n_B, flags, T):
    """Converged beta-eq point, its unknown vector x0, ctx and Jacobian."""
    p = solve_beta_eq_neutrinoless(par, n_B, flags, T=T, include_photons=False)
    has_phi = flags.phi_field and flags.hyperons
    x0 = np.array(warm_start(p, has_phi, False, False))
    ctx0 = build_matter_ctx(par, n_B, flags, T=T)
    J = np.array(residual_jacobian(x0, ctx0, _BETA))
    return p, x0, ctx0, J, has_phi


def _tangent(x0, J, ctx_pert):
    """One Newton step from x0 toward the perturbed-ctx root, then assemble.
    Returns (P, eps, s) in fm-based units (MeV/fm^3, MeV/fm^3, fm^-3)."""
    R = np.array(residual(x0, ctx_pert, _BETA))
    dx = np.linalg.solve(J, -R)
    st = assemble(x0 + dx, ctx_pert, _BETA)
    return st["P"] / hc3, st["eps"] / hc3, st["s"] / hc3


def _sequence_derivs(par, n_B, flags, T, rel_dn=1e-3, dT=0.05):
    """Partials along the beta-eq sequence via Jacobian tangent steps (central
    difference in the parameter): dP/dn, deps/dn, and the entropy-PER-BARYON
    sigma = s/n derivatives dsigma/dn, dsigma/dT (per baryon is required for
    C_P, where the volume changes at fixed P)."""
    _, x0, ctx0, J, _ = _state(par, n_B, flags, T)
    dn = rel_dn * n_B
    Pp, ep, sp = _tangent(x0, J, build_matter_ctx(par, n_B + dn, flags, T=T))
    Pm, em, sm = _tangent(x0, J, build_matter_ctx(par, n_B - dn, flags, T=T))
    d = dict(dP_dn=(Pp - Pm) / (2 * dn), de_dn=(ep - em) / (2 * dn),
             dsig_dn=(sp / (n_B + dn) - sm / (n_B - dn)) / (2 * dn),
             dP_dT=0.0, dsig_dT=0.0)
    if T > 0.0:
        Pp, ep, sp = _tangent(x0, J, build_matter_ctx(par, n_B, flags, T=T + dT))
        Pm, em, sm = _tangent(x0, J, build_matter_ctx(par, n_B, flags, T=T - dT))
        d["dP_dT"] = (Pp - Pm) / (2 * dT)
        d["dsig_dT"] = (sp - sm) / (n_B * 2 * dT)
    return d


def sound_speed_eq(par, n_B, flags, T=0.0, rel_dn=1e-3):
    """Equilibrium c_s^2 = (dP/dn)/(deps/dn) along the sequence, from J."""
    d = _sequence_derivs(par, n_B, flags, T, rel_dn=rel_dn)
    return d["dP_dn"] / d["de_dn"]


def heat_capacity_V(par, n_B, flags, T, dT=0.05):
    """C_V = T (d sigma/dT)_{n_B} per baryon [dimensionless], from J."""
    d = _sequence_derivs(par, n_B, flags, T, dT=dT)
    return T * d["dsig_dT"]


def heat_capacity_P(par, n_B, flags, T, dT=0.05):
    """
    C_P = T (d sigma/dT)_P per baryon. Holding P fixed,
    (d sigma/dT)_P = (d sigma/dT)_n - (d sigma/dn)_T (dP/dT)/(dP/dn),
    all from the same tangent sensitivities. C_P >= C_V by construction.
    """
    d = _sequence_derivs(par, n_B, flags, T, dT=dT)
    dsig_dT_P = d["dsig_dT"] - d["dsig_dn"] * d["dP_dT"] / d["dP_dn"]
    return T * dsig_dT_P


#: conserved-charge labels for the susceptibility matrix rows/cols.
SUSCEPT_LABELS = ("B", "C", "S")


def susceptibilities(par, n_B, flags, T=0.0):
    """
    Charge susceptibility matrix chi_ab = dn_a/dmu_b [MeV^2, natural units],
    a,b in (B, C, S), from residual_jacobian's field-response block. Symmetric
    (chi = -d^2 Omega / dmu_a dmu_b). Evaluated at the beta-eq state (mu_S = 0).
    Returns a 3x3 numpy array in SUSCEPT_LABELS order.
    """
    p = solve_beta_eq_neutrinoless(par, n_B, flags, T=T, include_photons=False)
    has_phi = flags.phi_field and flags.hyperons
    # fixed+fixed ctx exposes the mu_S column and the hadronic charge/strange
    # rows; the Y_C/Y_S targets don't enter the (pointwise) Jacobian.
    held = mode_spec(charge_mode="fixed", strange_mode="fixed")
    ctx = build_matter_ctx(par, n_B, flags, T=T)
    m = p.matter
    x = [m.fields["sigma"], m.fields["omega0"], m.fields["rho0"]]
    if has_phi:
        x.append(m.fields["phi0"])
    x += [m.mu_B - m.Sigma_R, m.mu_C, 0.0]
    J = np.array(residual_jacobian(np.array(x), ctx, held))

    n_f = 3 + int(has_phi)
    field_cols = list(range(n_f))
    mu_cols = [n_f, n_f + 1, n_f + 2]          # mutB, muC, muS
    rows = [n_f, n_f + 1, n_f + 2]             # baryon#, charge, strange
    JFF = J[np.ix_(field_cols, field_cols)]
    JFmu = J[np.ix_(field_cols, mu_cols)]
    dF_dmu = np.linalg.solve(JFF, -JFmu)       # (n_f, 3)

    chi = np.empty((3, 3))
    for a, r in enumerate(rows):
        dna_dF = ctx.nB_nat * J[r, field_cols]
        dna_dmu = ctx.nB_nat * J[r, mu_cols]
        chi[a, :] = dna_dmu + dna_dF @ dF_dmu
    return chi


if __name__ == "__main__":
    from eos.dd2 import Parameters, SpeciesFlags
    par = Parameters.named("DD2Y")
    flags = SpeciesFlags(hyperons=True, phi_field=True)
    for n in (0.16, 0.4, 0.8):
        cs2 = sound_speed_eq(par, n, flags)
        chi = susceptibilities(par, n, flags)
        print(f"n_B={n}: cs2_eq={cs2:.4f}  chi_BB={chi[0,0]:.3e}  "
              f"asym={np.max(np.abs(chi - chi.T)):.1e}")
