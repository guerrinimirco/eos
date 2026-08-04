"""
mixed/coefficients.py
=====================
The two speeds of sound of a hadron-quark mixture, and the adiabatic index.

*Public API* (re-exported from `eos.mixed`): `sound_speed_eq`,
`sound_speed_frozen`, `adiabatic_index`.

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
`eos/dd2/coefficients.py:sound_speed_adiabatic` means for nucleonic matter,
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
   `eos.dd2.coefficients.sound_speed_adiabatic`.
4. **Leptons are re-neutralised** against the frozen total charge, so the
   mixture stays electrically neutral under the perturbation. This is the
   physical choice for stellar matter and it is not a small effect — dropping
   the leptons changes c_ad by several percent. `leptons=False` turns it off,
   which is what makes the chi = 0 limit directly comparable with
   `eos.dd2.coefficients.sound_speed_adiabatic`, a nucleonic-matter probe that
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
from eos.dd2.solver import solve_octet
from eos.dd2.physics.octet import _yc_neutralizing_leptons
from eos.vmit.thermodynamics_quarks import compute_quark_matter_thermo_from_n
from eos.mixed.equilibrium.charges import quark_charges


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


def _frozen_mixture(par, flags, result, vmit_params, scale, leptons=True):
    """(P, eps) of the mixture compressed by `scale` at frozen composition.

    Both phases are rescaled by the same factor at fixed chi; see the module
    docstring for exactly what is held fixed.
    """
    chi, T = result.chi, result.T
    th_H, th_Q = result.th_H, result.th_Q
    P_tot = eps_tot = n_C_tot = 0.0

    if chi < 1.0 and th_H.n_B > 0.0:
        n_H = th_H.n_B * scale
        Y_C_H = th_H.n_C / th_H.n_B
        Y_S_H = th_H.n_S / th_H.n_B
        # strange_mode='fixed' only when strangeness can actually vary; for
        # nucleonic matter it would add an inert unknown to the solve.
        strange = "fixed" if flags.has_strange_baryons else "eq"
        p = solve_octet(par, n_H, flags, T=T, charge_mode="fixed", Y_C=Y_C_H,
                        strange_mode=strange, Y_S=Y_S_H, yc_leptons=False,
                        include_photons=False, check_consistency=False)
        P_tot += (1.0 - chi) * p.P
        eps_tot += (1.0 - chi) * p.eps
        n_C_tot += (1.0 - chi) * Y_C_H * n_H

    if chi > 0.0 and th_Q.n_B > 0.0:
        # Rescaling the flavour densities freezes the quark composition exactly
        # — no re-solve, and no chance of the solve drifting to another root.
        n_u = th_Q.densities["u"] * scale
        n_d = th_Q.densities["d"] * scale
        n_s = th_Q.densities["s"] * scale
        _mu_u, _mu_d, _mu_s, P_Q, e_Q, _s_Q, _nB_Q = \
            compute_quark_matter_thermo_from_n(n_u, n_d, n_s, T, vmit_params)
        P_tot += chi * P_Q
        eps_tot += chi * e_Q
        n_C_tot += chi * quark_charges(n_u, n_d, n_s)[1]

    if leptons:
        # Re-neutralise. `_yc_neutralizing_leptons` works in natural units and
        # returns (mu_e, (n,P,eps,s)_e, (n,P,eps,s)_mu) there.
        _mu_e, e_blk, mu_blk = _yc_neutralizing_leptons(
            n_C_tot * hc3, Electron.mass, Muon.mass, flags.muons, T)
        P_tot += (e_blk[1] + mu_blk[1]) / hc3
        eps_tot += (e_blk[2] + mu_blk[2]) / hc3
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
                   `eos.dd2.coefficients.sound_speed_adiabatic` exactly.

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

    kw = dict(leptons=leptons)
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
