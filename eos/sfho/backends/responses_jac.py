"""
backends/responses_jac.py
=========================
The response quantities that come off the analytic Jacobian rather than out of
a finite-difference stencil.

One quantity lives here: the charge susceptibility matrix

    chi_ab = dn_a / dmu_b        a, b in (B, C, S)

the second-derivative block of the CompOSE manual (Typel et al.,
arXiv:2203.03209 section 3.6). It is not a stencil quantity at all -- the
solver never varies mu_B, mu_C and mu_S independently, so there is nothing to
difference along -- which is why it is the one response that needs `backends/`.
`responses.py` holds the finite-difference flavour of everything that does have
a sequence to walk along, and stays the reference for those.

The derivation is the field-response identity. Write the solved state as the
mean fields F = (sigma, omega, rho, phi) satisfying G(F, mu) = 0 together with
the densities n_a(F, mu). Then

    dn_a/dmu_b = dn_a/dmu_b |_F  +  (dn_a/dF) (dF/dmu_b)
    dF/dmu     = -(dG/dF)^-1 (dG/dmu)

and every block on the right is a sub-matrix of `jacobian.residual_jacobian`:
rows 0-3 are G, rows 4-6 are n_B, n_C, n_S, columns 0-3 are F and columns 4-6
are the three potentials. The row scaling of the field equations cancels
between (dG/dF) and (dG/dmu), so no unscaling is needed.

Units are fm-based, as at every boundary of this package: chi_ab is in
fm^-3 MeV^-1.
"""
import numpy as np

from eos.general import modes
from eos.sfho.solver import _system, solve
from eos.sfho.backends.jacobian import residual_jacobian

#: Row and column order of the matrix `susceptibilities` returns.
SUSCEPT_LABELS = ("B", "C", "S")


def susceptibilities(par, n_B, flags, T=0.0, spec=None):
    """chi_ab = dn_a/dmu_b [fm^-3 MeV^-1] at the state `spec` selects.

    `spec` defaults to beta equilibrium with free-streaming neutrinos, where
    mu_S = 0 and mu_C is set by neutrality. Whatever the state, the Jacobian is
    built in the fixed-Y_C-and-Y_S declaration, because that is the one that
    carries mu_S as a column and both charge densities as rows -- the targets
    themselves never enter the Jacobian, only which symbols are unknowns.

    The matrix is symmetric, chi_ab = -d^2 Omega / dmu_a dmu_b, and the
    departure from symmetry is a useful check on the whole path.
    """
    spec = spec or modes.beta_eq_neutrinoless()
    state = solve(_system(par, flags, spec, n_B, T=T))
    if not state.converged:
        raise RuntimeError(
            f"no converged state to differentiate at n_B={n_B:g} fm^-3, "
            f"T={T:g} MeV (residual {state.error:.3e})")

    held = modes.fixed_YC_YS(state.Y_C, state.Y_S)
    sys = _system(par, flags, held, n_B, T=T)
    x = np.array([state.sigma, state.omega, state.rho, state.phi,
                  state.mu_B, state.mu_C, state.mu_S])
    J = residual_jacobian(x, sys)

    fields = slice(0, 4)
    pots = slice(4, 7)
    dF_dmu = np.linalg.solve(J[fields, fields], -J[fields, pots])

    chi = np.empty((3, 3))
    for a in range(3):
        row = 4 + a                      # n_B, n_C, n_S rows of the residual
        chi[a, :] = J[row, pots] + J[row, fields] @ dF_dmu
    return chi


if __name__ == "__main__":
    from eos.sfho.parameters import get_sfhoy_fortin
    from eos.sfho.species import SpeciesFlags

    par, flags = get_sfhoy_fortin(), SpeciesFlags(hyperons=True)
    for n in (0.16, 0.4, 0.8):
        chi = susceptibilities(par, n, flags, T=10.0)
        print(f"n_B={n}: chi_BB={chi[0, 0]:.3e}  "
              f"asym={np.max(np.abs(chi - chi.T)):.1e}")
