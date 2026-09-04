"""
backends/jacobian.py
====================
The analytic Jacobian of `eos.njl.solver.residual`: the same equations,
differentiated once by hand.

`residual` is assembled from three blocks of the state -- the nine modes as cut
Fermi gases, the pairing correction, the RG counterterm -- and every row is a
first derivative of Omega or a linear combination of densities. So its Jacobian
is the SECOND derivatives of those blocks, chained to the unknown vector
through the linear map from (mu_B, mu_C, mu_S, mu_3, mu_8, Sigma_V) to the
nine effective mode potentials. The blocks' second derivatives are

    modes       `backends.kernel_numba.modes_jacobian`, the Fermi integrals
                differentiated under the integral (surface terms at T = 0)
    pairing     `eos.general.pairing.pair_hessian`, second-order perturbation
                theory of the quasiparticle spectrum, under the same RG split
                `rg_pair_block` applies to the block itself
    counterterm closed form, below
    Dirac sea   closed form, below

and the leptons are the one column that is differenced: a JEL fit has no
derivative to write down, it is one call per lepton, and it feeds two rows.

Why it pays: MINPACK's forward-difference Jacobian costs one residual per
unknown per Newton step, 11 or 12 of them here, and a residual is three
quadrature passes with a 12 x 12 diagonalisation at every node. The Hessian is
one pass more of the same eigenvectors.

`backends/` is deletable (CLAUDE.md section 5): with this file gone, or with
numba absent, `solve_pattern` hands MINPACK the residual alone and differences
it, and every number is the same. `eos/njl/verify/run_full_check.py` checks
this Jacobian against a central difference of the reference residual.

Parameter order inside: theta = (mu*_0..8, M_u, M_d, M_s, Delta_1, Delta_2,
Delta_3), the 15 quantities the blocks are functions of, in the order
`pair_hessian` uses. Natural units throughout; the density rows are divided by
hc^3 where `residual` divides them.
"""
from functools import lru_cache
import math

import numpy as np

from eos.general.fermi_integrals import (DEGENERACY, NODES_PER_PANEL,
                                         _gauss_legendre, panel_nodes)
from eos.general.modes import electron_potential, muon_potential
from eos.general.pairing import (
    CHARGE, COLOUR_OF_MODE, FLAVOUR_OF_MODE, N_MODES, STRANGENESS, active_gaps,
    gapless_breakpoints, mode_potentials, pair_hessian, pair_nodes,
    pattern_mask,
)
from eos.general.physics_constants import hc3
from eos.general.thermodynamics_leptons import (
    electron_thermo, muon_thermo, neutrino_thermo,
)
from eos.njl.backends.kernel_numba import NUMBA_OK, modes_jacobian
from eos.njl.couplings import vector_self_energy_derivative
from eos.njl.species import DEGENERACY_SEA
from eos.njl.thermodynamics import (
    PAIRS, RG_PANEL_RATIO, _shape_gap_derivative, counterterm_shape,
    has_vector, state_at,
)

_PI2 = math.pi ** 2
_GAUSS_X, _GAUSS_W = _gauss_legendre(NODES_PER_PANEL)

#: theta indices
_MU, _M, _DELTA = slice(0, 9), slice(9, 12), slice(12, 15)


# =============================================================================
# THE CLOSED-FORM PIECES
# =============================================================================
def sea_scalar_density_derivative(m, Lambda, g=DEGENERACY_SEA):
    """d rho_s,vac / d m [MeV^2] of `thermodynamics.sea_scalar_density`.

        (g/2 pi^2) [ Lambda R/2 + Lambda m^2/R - (3/2) m^2 arcsinh(Lambda/m) ]

    with R = sqrt(Lambda^2 + m^2). The condensate is phi = -rho_s,vac + ...,
    so this enters the mass rows with the opposite sign.
    """
    if m <= 0.0:
        return 0.0
    R = math.sqrt(Lambda ** 2 + m ** 2)
    return (g / (2.0 * _PI2)) * (0.5 * Lambda * R + Lambda * m ** 2 / R
                                 - 1.5 * m ** 2 * math.asinh(Lambda / m))


def _shape_gap_second_derivative(Delta, Lambda, Lambda_UV):
    """d^2/dDelta^2 [ Delta^2 g(Delta) ] of `thermodynamics.counterterm_shape`.

    `_shape_gap_derivative` differentiated once more, term by term, with
    d(Lambda/lo)/dDelta = -Lambda Delta/lo^3 and
    d ln(Lambda + lo)/dDelta = Delta/(lo (Lambda + lo)).
    """
    hi = math.hypot(Lambda_UV, Delta)
    lo = math.hypot(Lambda, Delta)
    D = Delta
    return (3.0 * (Lambda / lo - Lambda_UV / hi)
            + 3.0 * D * (-Lambda * D / lo ** 3 + Lambda_UV * D / hi ** 3)
            + 2.0 * math.log((Lambda_UV + hi) / (Lambda + lo))
            + 2.0 * D * (D / (hi * (Lambda_UV + hi)) - D / (lo * (Lambda + lo)))
            + 3.0 * D ** 2 * (Lambda_UV / hi ** 3 - Lambda / lo ** 3)
            + D ** 3 * (-3.0 * Lambda_UV * D / hi ** 5 + 3.0 * Lambda * D / lo ** 5))


def counterterm_jacobian(par, Delta, mu_star):
    """d(n_ct)/d theta (9, 15) and d(gap_ct)/d theta (3, 15) [MeV^2].

    `thermodynamics.counterterm` per pair (eta, i, j) with h = Delta^2 g:

        n_i, n_j  -=  mubar h / pi^2          mubar = (mu*_i + mu*_j)/2
        gap_eta   +=  mubar^2 h' / pi^2

    so d n_i / d mu*_i = d n_i / d mu*_j = -h/(2 pi^2), d n_i / d Delta_eta =
    -mubar h'/pi^2 = -d gap_eta / d mu*_i, and d gap_eta / d Delta_eta =
    mubar^2 h''/pi^2. No mass, no T: nothing else.
    """
    dn = np.zeros((N_MODES, 15))
    dgap = np.zeros((3, 15))
    if not np.any(Delta):
        return dn, dgap
    for eta, i, j in PAIRS:
        D = float(Delta[eta])
        if D == 0.0:
            continue
        mubar = 0.5 * (mu_star[i] + mu_star[j])
        h = D ** 2 * counterterm_shape(D, par.Lambda, par.Lambda_medium)
        h1 = _shape_gap_derivative(D, par.Lambda, par.Lambda_medium)
        h2 = _shape_gap_second_derivative(D, par.Lambda, par.Lambda_medium)
        for row in (i, j):
            dn[row, i] -= 0.5 * h / _PI2
            dn[row, j] -= 0.5 * h / _PI2
            dn[row, 12 + eta] -= mubar * h1 / _PI2
        dgap[eta, i] += mubar * h1 / _PI2
        dgap[eta, j] += mubar * h1 / _PI2
        dgap[eta, 12 + eta] += mubar ** 2 * h2 / _PI2
    return dn, dgap


# =============================================================================
# THE PAIRING HESSIAN UNDER THE RG SPLIT
# =============================================================================
@lru_cache(maxsize=512)
def _vacuum_pair_hessian(M_bytes, Delta_bytes, k_max, nodes_per_panel):
    """`pair_hessian` at mu* = 0, T = 0, memoized on (M, Delta) as
    `thermodynamics._vacuum_pair_block` is.

    The vacuum block is a function of the masses and the gaps alone, so its
    derivatives with respect to the potentials are zero BY CONSTRUCTION, not
    by evaluation: the pass is told to leave them out (`with_mu=False`) and
    returns zero in the mu* columns.
    """
    M = np.frombuffer(M_bytes, dtype=float)
    Delta = np.frombuffer(Delta_bytes, dtype=float)
    zero = np.zeros(N_MODES)
    ph = pair_hessian(M, zero, Delta, 0.0, k_max, nodes_per_panel,
                      quadrature=pair_nodes(M, zero, 0.0, k_max,
                                            nodes_per_panel, RG_PANEL_RATIO),
                      with_mu=False)
    out = tuple(np.ascontiguousarray(block) for block in
                (ph.d_delta_n, ph.d_delta_rho_s, ph.d_gap_kernel))
    for block in out:
        block.flags.writeable = False
    return out


def rg_pair_jacobian(par, M, mu_star, Delta, T, nodes_per_panel=NODES_PER_PANEL):
    """(d delta_n, d delta_rho_s, d gap_kernel) / d theta under the RG split.

    The same three passes as `thermodynamics.rg_pair_block`, differentiated:
    hot(Lambda_UV) - vac(Lambda_UV) + vac(Lambda), on the same quadrature
    rules, so this is the derivative of the block the residual carries. At
    lambda = 1, or with no gap on, it is the hot pass alone.
    """
    if par.lambda_UV == 1.0 or not any(active_gaps(Delta)):
        ph = pair_hessian(M, mu_star, Delta, T, par.Lambda_medium,
                          nodes_per_panel)
        return ph.d_delta_n, ph.d_delta_rho_s, ph.d_gap_kernel
    # the same rule `rg_pair_block` builds, with the crossings found once
    # and handed to the Hessian for its Fermi-surface terms
    M_mode = M[FLAVOUR_OF_MODE]
    inside = np.abs(mu_star) > M_mode
    kF = np.sqrt(np.maximum(mu_star[inside] ** 2 - M_mode[inside] ** 2, 0.0))
    crossings = gapless_breakpoints(M, mu_star, Delta, par.Lambda_medium)
    rule = panel_nodes(np.concatenate([kF, crossings]), T, par.Lambda_medium,
                       nodes_per_panel, RG_PANEL_RATIO)
    hot = pair_hessian(M, mu_star, Delta, T, par.Lambda_medium,
                       nodes_per_panel, quadrature=rule, crossings=crossings)
    M_bytes = np.ascontiguousarray(M, dtype=float).tobytes()
    Delta_bytes = np.ascontiguousarray(Delta, dtype=float).tobytes()
    hi = _vacuum_pair_hessian(M_bytes, Delta_bytes, par.Lambda_medium,
                              nodes_per_panel)
    lo = _vacuum_pair_hessian(M_bytes, Delta_bytes, par.Lambda,
                              nodes_per_panel)
    return tuple(h - a + b for h, a, b in
                 zip((hot.d_delta_n, hot.d_delta_rho_s, hot.d_gap_kernel),
                     hi, lo))


# =============================================================================
# THE LEPTONS
# =============================================================================
#: Relative central-difference step on the lepton potentials.
_LEPTON_STEP = 1.0e-5


def _dn_dmu(thermo, mu, T):
    """dn/dmu [fm^-3 MeV^-1] of one lepton species, by central difference."""
    h = _LEPTON_STEP * max(abs(mu), T, 1.0)
    return (thermo(mu + h, T).n - thermo(mu - h, T).n) / (2.0 * h)


def lepton_derivatives(mu_C, mu_nue, T, flags):
    """d(n_charged, n_Le)/d(mu_C, mu_nue) of `solver.lepton_block`, as a
    2 x 2 array [[dn_ch/dmu_C, dn_ch/dmu_nue], [dn_Le/dmu_C, dn_Le/dmu_nue]].

    The block is n_charged = n_e(mu_e) + n_mu(mu_mu), n_Le = n_e + n_nu(mu_nue)
    with mu_e = mu_nue - mu_C and mu_mu = mu_e - mu_nue = -mu_C, so every
    entry is one species' dn/dmu times a sign.
    """
    mu_e = electron_potential(mu_C, mu_nue)
    de = _dn_dmu(electron_thermo, mu_e, T)
    dm = (_dn_dmu(muon_thermo, muon_potential(mu_e, mu_nue), T)
          if flags.muons else 0.0)
    dnu = _dn_dmu(neutrino_thermo, mu_nue, T) if mu_nue != 0.0 else 0.0
    return np.array([[-de - dm, de],
                     [-de, de + dnu]])


# =============================================================================
# THE JACOBIAN
# =============================================================================
def residual_jacobian(x, names, par, flags, spec, pattern, n_B, T, vac,
                      pair_nodes_per_panel=None, state=None):
    """d(residual)/dx, rows in `solver.residual`'s order, columns in `names`.

    `names` is `solver.unknown_slots(par, spec, pattern)`: the layout arrives
    as data so this module needs nothing from the solver. `state` is the
    `NJLState` at x if the caller has it -- a Newton step has just evaluated
    the residual there, and the state is a third of the Jacobian's cost --
    and is computed here otherwise. Returns the (n_rows, n_unknowns) array;
    the caller divides the rows by their scales.
    """
    if not NUMBA_OK:
        raise NotImplementedError("eos.njl's analytic Jacobian needs numba")
    x = np.asarray(x, dtype=float)
    names = tuple(names)
    col = {name: i for i, name in enumerate(names)}
    n_x = len(names)
    got = dict(zip(names, x))
    M = np.array([got["M_u"], got["M_d"], got["M_s"]])
    Delta = np.array([got.get(f"Delta_{eta + 1}", 0.0) for eta in range(3)])
    mu_3, mu_8 = got.get("mu_3", 0.0), got.get("mu_8", 0.0)
    Sigma_V = got.get("Sigma_V", 0.0)
    mu_B, mu_C = got["mu_B"], got["mu_C"]
    mu_S, mu_nue = got.get("mu_S", 0.0), got.get("mu_nue", 0.0)
    nodes = NODES_PER_PANEL if pair_nodes_per_panel is None else pair_nodes_per_panel

    st = state
    if st is None:
        st = state_at(par, M, Delta, Sigma_V, mu_B, mu_C, mu_S, mu_3, mu_8, T,
                      vac=vac, pattern=pattern, two_flavour=flags.two_flavour,
                      backend="fast",
                      pair_nodes_per_panel=pair_nodes_per_panel)
    mu_star = st.mu_star
    M_mode = M[FLAVOUR_OF_MODE]

    # --- d theta / d x: the 15 block parameters against the unknowns -------
    dtheta = np.zeros((15, n_x))
    for name, args in (("mu_B", (1.0, 0.0, 0.0, 0.0, 0.0)),
                       ("mu_C", (0.0, 1.0, 0.0, 0.0, 0.0)),
                       ("mu_S", (0.0, 0.0, 1.0, 0.0, 0.0)),
                       ("mu_3", (0.0, 0.0, 0.0, 1.0, 0.0)),
                       ("mu_8", (0.0, 0.0, 0.0, 0.0, 1.0))):
        if name in col:
            dtheta[_MU, col[name]] = mode_potentials(*args)
    if "Sigma_V" in col:
        dtheta[_MU, col["Sigma_V"]] = -1.0
    for f, name in enumerate(("M_u", "M_d", "M_s")):
        dtheta[9 + f, col[name]] = 1.0
    for eta in range(3):
        name = f"Delta_{eta + 1}"
        if name in col:
            dtheta[12 + eta, col[name]] = 1.0

    def unit(name):
        e = np.zeros(n_x)
        e[col[name]] = 1.0
        return e

    # --- the blocks' derivatives against theta ----------------------------
    absent = np.array([bool(flags.two_flavour and FLAVOUR_OF_MODE[j] == 2)
                       for j in range(N_MODES)])
    K = modes_jacobian(mu_star, M_mode, T, par.Lambda_medium, DEGENERACY,
                       _GAUSS_X, _GAUSS_W, absent)
    dn = np.zeros((N_MODES, 15))          # d n_j / d theta
    drho = np.zeros((3, 15))              # d rho_s,f / d theta
    for j in range(N_MODES):
        f = FLAVOUR_OF_MODE[j]
        dn[j, j] += K[j, 0]
        dn[j, 9 + f] += K[j, 1]
        drho[f, j] += -K[j, 1]            # d rho_s / d mu = -d n / d M
        drho[f, 9 + f] += K[j, 2]
    pair_n, pair_rho, dkernel = rg_pair_jacobian(par, M, mu_star, Delta, T,
                                                 nodes)
    ct_n, ct_gap = counterterm_jacobian(par, Delta, mu_star)
    dn += pair_n + ct_n
    drho += pair_rho

    # the condensates: phi_f = -rho_s,vac(M_f) + rho_s,f
    dphi = drho.copy()
    for f in range(3):
        dphi[f, 9 + f] -= sea_scalar_density_derivative(M[f], par.Lambda)

    dn_flavour = np.zeros((3, 15))
    for j in range(N_MODES):
        dn_flavour[FLAVOUR_OF_MODE[j]] += dn[j]
    dn_q = dn_flavour.sum(axis=0)
    dn_B = dn_q / 3.0

    # --- the rows -----------------------------------------------------------
    rows = []
    G_S, Kdet = par.G_S, par.K
    phi = st.phi
    cyclic = ((0, 1, 2), (1, 0, 2), (2, 0, 1))
    for f, g, h in cyclic:
        # M_f - [m_f - 4 G_S phi_f + 2 K phi_g phi_h]
        row = 4.0 * G_S * dphi[f] - 2.0 * Kdet * (phi[h] * dphi[g]
                                                  + phi[g] * dphi[h])
        rows.append(row @ dtheta + unit(("M_u", "M_d", "M_s")[f]))
    mask = pattern_mask(pattern)
    for eta in range(3):
        if mask[eta]:
            row = -dkernel[eta] + ct_gap[eta]
            rows.append(row @ dtheta + unit(f"Delta_{eta + 1}") / (2.0 * par.G_D))
    if any(mask):
        colour = COLOUR_OF_MODE
        c3 = np.where(colour == 0, 1.0, np.where(colour == 1, -1.0, 0.0))
        c8 = np.where(colour == 2, -2.0, 1.0)
        rows.append((c3 @ dn) @ dtheta)
        rows.append((c8 @ dn) @ dtheta)
    if has_vector(par):
        rows.append(-vector_self_energy_derivative(par, st.n_q) * (dn_q @ dtheta)
                    + unit("Sigma_V"))
    rows.append((dn_B @ dtheta) / hc3)

    # the charge rows, as `solver._charge_rows` assembles them
    lep = None
    if not spec.is_fixed("C") or spec.is_fixed("L_e"):
        lep = lepton_derivatives(mu_C, mu_nue, T, flags)
    dn_C = CHARGE @ dn_flavour
    dn_S = STRANGENESS @ dn_flavour
    if spec.is_fixed("C"):
        rows.append(((dn_C - spec.targets["Y_C"] * dn_B) @ dtheta) / hc3)
    else:
        row = (dn_C @ dtheta) / hc3 - lep[0, 0] * unit("mu_C")
        if "mu_nue" in col:
            row -= lep[0, 1] * unit("mu_nue")
        rows.append(row)
    if spec.is_fixed("S"):
        if T == 0.0 and spec.targets["Y_S"] == 0.0:
            a = int(np.argmax(mu_star[6:9]))
            rows.append(dtheta[6 + a] - unit("M_s"))
        else:
            rows.append(((dn_S - spec.targets["Y_S"] * dn_B) @ dtheta) / hc3)
    if spec.is_fixed("L_e"):
        row = -spec.targets["Y_Le"] * (dn_B @ dtheta) / hc3
        row += lep[1, 0] * unit("mu_C") + lep[1, 1] * unit("mu_nue")
        rows.append(row)
    return np.array(rows)
