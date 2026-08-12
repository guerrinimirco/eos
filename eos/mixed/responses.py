"""
mixed/responses.py
==================
The two speeds of sound of a hadron-quark mixture, and the adiabatic index.

*Public API* (re-exported from `eos.mixed`): `sound_speed_eq`,
`sound_speed_frozen`, `sound_speed_frozen_hadronic`,
`sound_speed_frozen_quark`, `frozen_along`, `adiabatic_index`.

The frozen speed is defined at every density, not only inside a coexistence
window: `sound_speed_frozen` handles chi = 0 and chi = 1, and the two
pure-phase entry points do the same for a state that was solved on its own
rather than as one phase of a mixture. Outside the window the composition that
gets frozen is the phase's own, so the three agree at the boundaries and the
curve is continuous across them.

A first-order transition has two different sound speeds, and which one is
physical depends on how fast the matter is disturbed relative to the rate at
which one phase converts into the other:

**Equilibrium**, `c_eq^2 = dP/deps` along the solved sequence. The phase
fraction chi is free to readjust, so a compression is answered by converting
hadrons into quarks rather than by raising the pressure. Through a Maxwell
window (eta = 1) the pressure is constant, so `c_eq -> 0`; through a Gibbs
window (eta = 0) it dips but stays finite. This is the sound speed that enters
the TOV equations and the one whose causality bound `0 <= c^2 <= 1` the
verification suite checks.

**Frozen (adiabatic)**, `c_ad^2 = dP/deps` at fixed composition. The mixture is
compressed faster than the phases can convert, so the pressure has to rise and
`c_ad` does *not* collapse in the window. This mirrors what
`eos/dd2/responses.py:sound_speed_adiabatic` means for nucleonic matter,
carried over to a two-phase mixture.

What "frozen" means here — this is a convention, and a different one gives
different numbers
------------------------------------------------------------------------
1. **chi is held fixed.** This is the part that matters: it is what stops the
   mixture sliding along a Maxwell plateau. Freezing only the charge fractions
   would not, because the solve would simply readjust chi and return to the
   plateau.
2. **Each phase is compressed uniformly**: n_H and n_Q both scale by the same
   factor, which at fixed chi is what holding the volume fractions fixed means.
3. **Each phase keeps its own charge and strangeness fraction**, Y_C and Y_S.
   For the quark phase this freezes the composition *exactly*, because the
   flavour densities are simply rescaled. For the hadronic phase with hyperons
   or Deltas active it is weaker than a full freeze: the individual species
   still re-equilibrate among themselves at fixed total Y_C and Y_S. For
   nucleonic matter the two coincide, and the result then reduces exactly to
   `eos.dd2.responses.sound_speed_adiabatic`.
4. **Leptons are re-neutralised** against the frozen total charge, so the
   mixture stays electrically neutral under the perturbation. This is the
   physical choice for stellar matter and it is not a small effect — dropping
   the leptons changes c_ad by several percent. `leptons=False` turns it off,
   which is what makes the chi = 0 limit directly comparable with
   `eos.dd2.responses.sound_speed_adiabatic`, a nucleonic-matter probe that
   carries no leptons.
5. **Photons and trapped neutrinos are omitted.** At fixed T they contribute
   identically at both perturbation points, so they cancel out of dP and deps
   and cannot affect the result.

Units are fm-based on every boundary, as everywhere else in `eos`: densities
fm^-3, P and eps MeV/fm^3, T MeV. Both functions return the dimensionless
c^2 in units of the speed of light.
"""
import numpy as np

from eos.general.physics_constants import hc3
from eos.general.particles import Electron, Muon
from eos.dd2.solver import solve_octet, octet_warm_start
from eos.dd2.thermodynamics import neutralizing_leptons
from eos.vmit.thermodynamics import compute_quark_matter_thermo_from_n
from eos.mixed.equilibrium.charges import quark_charges
from eos.mixed.adapters import hadronic_phase


def sound_speed_eq(P, eps):
    """Equilibrium c_s^2 = dP/deps along a solved sequence.

    `P` and `eps` are arrays in MeV/fm^3, ascending in density — a
    `MixedEoSTable`'s `.P` and `.eps`, or the stitched columns a notebook
    assembles. A central finite difference on the sequence itself, so it needs
    no extra solves; the sequence must be dense enough that the derivative is
    meaningful, which through a transition window means resolving the window.

    Arrays rather than a table object so that the pure phases, the stitched
    core equation of state and a notebook's own columns all go through one
    function.
    """
    P = np.asarray(P, dtype=float)
    eps = np.asarray(eps, dtype=float)
    if P.size < 2:
        return np.full(P.shape, np.nan)
    return np.gradient(P, eps)


def adiabatic_index(P, eps):
    """Gamma = (eps + P)/P * c_eq^2 along the same sequence."""
    P = np.asarray(P, dtype=float)
    eps = np.asarray(eps, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(P > 0.0, (eps + P) / P * sound_speed_eq(P, eps), np.nan)


def _hadronic_block(par, flags, n_B, Y_C, Y_S, T, x0=None):
    """(P, eps, n_C) of hadronic matter at `n_B` with Y_C and Y_S held.

    Matter only: no leptons, no photons. `x0` is the octet unknown vector to
    start from and is what keeps this solvable — see `_octet_seed`.
    """
    # strange_mode='fixed' only when strangeness can actually vary; for
    # nucleonic matter it would add an inert unknown to the solve.
    strange = "fixed" if flags.has_strange_baryons else "eq"
    p = solve_octet(par, n_B, flags, T=T, x0=x0, charge_mode="fixed", Y_C=Y_C,
                    strange_mode=strange, Y_S=Y_S, yc_leptons=False,
                    include_photons=False, check_consistency=False)
    return p.P, p.eps, Y_C * n_B


def _quark_block(n_u, n_d, n_s, T, vmit_params):
    """(P, eps, n_C) of quark matter at the given flavour densities.

    Rescaling the flavour densities freezes the quark composition exactly — no
    re-solve, and no chance of the solve drifting to another root.
    """
    _mu_u, _mu_d, _mu_s, P, eps, _s, _n_B = \
        compute_quark_matter_thermo_from_n(n_u, n_d, n_s, T, vmit_params)
    return P, eps, quark_charges(n_u, n_d, n_s)[1]


def _lepton_block(n_C, flags, T):
    """(P, eps) of the leptons neutralising a charge density `n_C` [fm^-3].

    `neutralizing_leptons` works in natural units and returns
    (mu_e, (n,P,eps,s)_e, (n,P,eps,s)_mu) there.
    """
    _mu_e, e_blk, mu_blk = neutralizing_leptons(
        n_C * hc3, Electron.mass, Muon.mass, flags.muons, T)
    return (e_blk[1] + mu_blk[1]) / hc3, (e_blk[2] + mu_blk[2]) / hc3


def _octet_seed(par, flags, result):
    """Octet unknown vector at the mixture's own hadronic state, or None.

    Both perturbed solves start from here. Without it `solve_octet` falls back
    to `default_octet_guess`, which seeds from *nucleonic beta equilibrium* —
    and the hadronic component of a mixed phase is nowhere near that: at a
    DD2 + vMIT onset with hyperons and Deltas active it carries Y_S ~ 0.5 and a
    slightly negative Y_C, so the solve stalls at a residual of order 1 and the
    point comes back nan. That hit roughly one point in six of a beta-
    equilibrium sweep and left visible holes in c_ad.

    The mixed solve stores its converged potentials but not the meson fields,
    so the fields are recovered by re-solving the hadronic phase at those
    potentials — four or five unknowns with a deterministic seed of its own,
    far cheaper and far more robust than the octet solve it is seeding.
    """
    th = result.th_H
    mu_tB = result.potentials.get("mu_tilde_B_H")
    if mu_tB is None or th.n_B <= 0.0:
        return None
    try:
        _th, state = hadronic_phase(par, flags, mu_tB, th.mu_C, th.mu_S,
                                    T=result.T, n_B_guess=th.n_B,
                                    return_state=True)
    except RuntimeError:
        return None
    # x_phase is [sigma, omega0, rho0, (phi0), n_B]; the octet unknown vector
    # wants the fields, then the potentials.
    x = list(state["x_phase"][:-1]) + [mu_tB, th.mu_C]
    if flags.has_strange_baryons:            # matches strange_mode='fixed'
        x.append(th.mu_S)
    return x


def _frozen_mixture(par, flags, result, vmit_params, scale, leptons=True,
                    x0=None):
    """(P, eps) of the mixture compressed by `scale` at frozen composition.

    Both phases are rescaled by the same factor at fixed chi; see the module
    docstring for exactly what is held fixed.
    """
    chi, T = result.chi, result.T
    th_H, th_Q = result.th_H, result.th_Q
    P_tot = eps_tot = n_C_tot = 0.0

    if chi < 1.0 and th_H.n_B > 0.0:
        P, eps, n_C = _hadronic_block(par, flags, th_H.n_B * scale,
                                      th_H.n_C / th_H.n_B,
                                      th_H.n_S / th_H.n_B, T, x0=x0)
        P_tot += (1.0 - chi) * P
        eps_tot += (1.0 - chi) * eps
        n_C_tot += (1.0 - chi) * n_C

    if chi > 0.0 and th_Q.n_B > 0.0:
        P, eps, n_C = _quark_block(th_Q.densities["u"] * scale,
                                   th_Q.densities["d"] * scale,
                                   th_Q.densities["s"] * scale, T, vmit_params)
        P_tot += chi * P
        eps_tot += chi * eps
        n_C_tot += chi * n_C

    if leptons:
        P_l, eps_l = _lepton_block(n_C_tot, flags, T)
        P_tot += P_l
        eps_tot += eps_l
    return P_tot, eps_tot


def sound_speed_frozen(par, flags, result, vmit_params=None, rel_dn=1e-3,
                       leptons=True):
    """Frozen-composition c_ad^2 = dP/deps at the state `result`.

    par, flags   : the DD2 `Parametrization` and `SpeciesFlags` the state was
                   solved with
    result       : a `MixedResult` from `solve_mixed` — pure phases included,
                   chi = 0 and chi = 1 are handled and give the pure-phase
                   frozen sound speed
    vmit_params  : the `VMITParams` the state was solved with
    rel_dn       : relative density step for the central difference
    leptons      : re-neutralise with leptons (the physical choice for stellar
                   matter, and the default). False gives the matter-only value,
                   which at chi = 0 reproduces
                   `eos.dd2.responses.sound_speed_adiabatic` exactly.

    Read the module docstring before comparing this to a number from elsewhere:
    "frozen" is a convention, and this one freezes chi and each phase's Y_C and
    Y_S. Returns nan if the perturbed states do not bracket a positive deps.
    """
    if vmit_params is None:
        from eos.vmit.parameters import get_vmit_default
        vmit_params = get_vmit_default()
    chi = float(np.clip(result.chi, 0.0, 1.0))
    if chi != result.chi:                       # a drifted point: use the wing
        result = _clipped(result, chi)

    kw = dict(leptons=leptons, x0=_octet_seed(par, flags, result))
    P_lo, e_lo = _frozen_mixture(par, flags, result, vmit_params,
                                 1.0 - rel_dn, **kw)
    P_hi, e_hi = _frozen_mixture(par, flags, result, vmit_params,
                                 1.0 + rel_dn, **kw)
    if not (e_hi > e_lo):
        return float("nan")
    return (P_hi - P_lo) / (e_hi - e_lo)


def _clipped(result, chi):
    """A shallow copy of `result` with chi clamped into [0, 1].

    `solve_mixed` deliberately does not clamp chi — its sign is what classifies
    a density as pure or mixed — but a frozen sound speed at chi < 0 or chi > 1
    is not defined, and the physical answer there is the pure phase.
    """
    from dataclasses import replace
    return replace(result, chi=chi)


def sound_speed_frozen_hadronic(par, flags, point, rel_dn=1e-3, leptons=True):
    """Frozen c_ad^2 of PURE hadronic matter at a solved octet point.

    The chi -> 0 limit of `sound_speed_frozen`, for a state that was solved on
    its own rather than as one phase of a mixture — the wing below the onset.
    Same convention: Y_C and Y_S held, the individual species free to
    re-equilibrate within them, leptons re-neutralised against the frozen
    charge. The two therefore join continuously at the onset, which is the
    point of having them agree — c_ad is a property of the matter, not of the
    machinery that solved it.

    `point` is an `EoSPoint` from any of the `eos.dd2` octet solvers, and
    supplies its own warm start, so no seeding argument is needed.

    This is NOT the equilibrium sound speed of the same wing: there the
    composition follows beta equilibrium (or the fixed-Y_C condition) as the
    density changes, and the gap between the two is exactly what drives a
    composition g-mode in the pure hadronic phase.

    The fractions held are the point's own Y_C and Y_S, which are the TOTAL
    non-leptonic ones -- baryons plus any thermal meson gas. Summing the baryon
    densities instead would freeze a different composition from the one the
    fixed-Y_C solve then imposes, since that condition is stated on the total:
    at T = 40 MeV with a pion gas the two differ by about 16%, and the curve
    would step at the onset instead of joining the mixture's.
    """
    n_B = point.n_B
    if n_B <= 0.0:
        return float("nan")
    Y_C, Y_S, T = point.Y_C, point.Y_S, point.T
    x0 = octet_warm_start(point, flags.phi_field and flags.hyperons,
                          has_muS=flags.has_strange_baryons)

    def at(scale):
        P, eps, n_C_s = _hadronic_block(par, flags, n_B * scale, Y_C, Y_S, T,
                                        x0=x0)
        if leptons:
            P_l, eps_l = _lepton_block(n_C_s, flags, T)
            P, eps = P + P_l, eps + eps_l
        return P, eps

    P_lo, e_lo = at(1.0 - rel_dn)
    P_hi, e_hi = at(1.0 + rel_dn)
    if not (e_hi > e_lo):
        return float("nan")
    return (P_hi - P_lo) / (e_hi - e_lo)


def sound_speed_frozen_quark(n_u, n_d, n_s, T=0.0, vmit_params=None,
                             rel_dn=1e-3, leptons=True, muons=True):
    """Frozen c_ad^2 of PURE quark matter at the given flavour densities.

    The chi -> 1 limit of `sound_speed_frozen` — the wing above the offset.
    The three flavour densities are rescaled together, which freezes the quark
    composition exactly, and the leptons are re-neutralised as in the mixture.
    Nothing is re-solved on the quark side.

    Densities are fm^-3. `muons` selects whether the neutralising lepton gas
    may contain muons, the role `flags.muons` plays elsewhere.
    """
    if vmit_params is None:
        from eos.vmit.parameters import get_vmit_default
        vmit_params = get_vmit_default()
    flags = _LeptonFlags(muons)

    def at(scale):
        P, eps, n_C = _quark_block(n_u * scale, n_d * scale, n_s * scale, T,
                                   vmit_params)
        if leptons:
            P_l, eps_l = _lepton_block(n_C, flags, T)
            P, eps = P + P_l, eps + eps_l
        return P, eps

    P_lo, e_lo = at(1.0 - rel_dn)
    P_hi, e_hi = at(1.0 + rel_dn)
    if not (e_hi > e_lo):
        return float("nan")
    return (P_hi - P_lo) / (e_hi - e_lo)


class _LeptonFlags:
    """The one field `_lepton_block` reads, for the quark-only entry point.

    The quark sector has no `SpeciesFlags` of its own — it is not a DD2 model —
    but the neutralising lepton gas is shared with the hadronic side and is
    built by the same helper.
    """

    def __init__(self, muons):
        self.muons = bool(muons)


def frozen_along(par, flags, results, vmit_params=None, rel_dn=1e-3,
                 leptons=True):
    """`sound_speed_frozen` at every state in a sequence, as an array.

    Non-convergent points come back nan rather than aborting the sequence: a
    gap in a diagnostic curve is a better outcome than losing the curve.
    """
    out = []
    for r in results:
        try:
            out.append(sound_speed_frozen(par, flags, r,
                                          vmit_params=vmit_params,
                                          rel_dn=rel_dn, leptons=leptons))
        except (RuntimeError, ValueError):
            out.append(np.nan)
    return np.asarray(out, dtype=float)
