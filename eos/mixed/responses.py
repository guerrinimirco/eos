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

from eos.dd2.solver import warm_start
from eos.general.sound_speeds import sound_speed_eq
from eos.general.thermodynamics_leptons import neutralizing_leptons
from eos.mixed.adapters import (
    _dd2_frozen_block, _vmit_frozen_block, default_pair,
)


# `sound_speed_eq(P, eps)` is imported above rather than defined here: it
# differentiates a table and knows no model, so its single home is
# `eos.general.sound_speeds` (CLAUDE.md section 7), beside the g-mode contract
# that also needs it. It stays on this module's public surface.


def adiabatic_index(P, eps):
    """Gamma = (eps + P)/P * c_eq^2 along the same sequence."""
    P = np.asarray(P, dtype=float)
    eps = np.asarray(eps, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(P > 0.0, (eps + P) / P * sound_speed_eq(P, eps), np.nan)


def _lepton_block(n_C, muons, T):
    """(P, eps) of the leptons neutralising a charge density `n_C` [fm^-3]."""
    _mu_e, e_blk, mu_blk = neutralizing_leptons(n_C, T, include_muons=muons)
    return e_blk.P + mu_blk.P, e_blk.e + mu_blk.e


def _frozen_mixture(pair, result, scale, muons, leptons=True):
    """(P, eps) of the mixture compressed by `scale` at frozen composition.

    Both phases are rescaled by the same factor at fixed chi; see the module
    docstring for exactly what is held fixed. Each phase's part is its own
    `frozen_thermo` capability; a phase without one raises naming itself
    rather than silently dropping its contribution.
    """
    chi, T = result.chi, result.T
    P_tot = eps_tot = n_C_tot = 0.0

    for phase, th, w, pos in ((pair[0], result.th_H, 1.0 - chi, "H"),
                              (pair[1], result.th_Q, chi, "Q")):
        if w <= 0.0 or th.n_B <= 0.0:
            continue
        if phase.frozen_thermo is None:
            raise NotImplementedError(
                f"the {phase.name} phase has no frozen_thermo capability, "
                f"so the frozen-composition sound speed is not defined for "
                f"this pairing (see docs/DEFERRED.md)")
        mu_slot = result.potentials.get(phase.slot(pos))
        P, eps, n_C = phase.frozen_thermo(th, scale, T, mu_slot=mu_slot)
        P_tot += w * P
        eps_tot += w * eps
        n_C_tot += w * n_C

    if leptons:
        P_l, eps_l = _lepton_block(n_C_tot, muons, T)
        P_tot += P_l
        eps_tot += eps_l
    return P_tot, eps_tot


def sound_speed_frozen(par, flags, result, vmit_params=None, rel_dn=1e-3,
                       leptons=True, phases=None, muons=None):
    """Frozen-composition c_ad^2 = dP/deps at the state `result`.

    par, flags   : the DD2 `Parameters` and `SpeciesFlags` the state was
                   solved with
    result       : a `Result` from `solve` — pure phases included,
                   chi = 0 and chi = 1 are handled and give the pure-phase
                   frozen sound speed
    vmit_params  : the `Parameters` the state was solved with
    rel_dn       : relative density step for the central difference
    leptons      : re-neutralise with leptons (the physical choice for stellar
                   matter, and the default). False gives the matter-only value,
                   which at chi = 0 reproduces
                   `eos.dd2.responses.sound_speed_adiabatic` exactly.

    Read the module docstring before comparing this to a number from elsewhere:
    "frozen" is a convention, and this one freezes chi and each phase's Y_C and
    Y_S. Returns nan if the perturbed states do not bracket a positive deps.
    """
    if phases is None:
        phases = default_pair(par, flags, vmit_params)
        if muons is None and flags is not None:
            muons = bool(flags.muons)
    muons = bool(muons)
    chi = float(np.clip(result.chi, 0.0, 1.0))
    if chi != result.chi:                       # a drifted point: use the wing
        result = _clipped(result, chi)

    P_lo, e_lo = _frozen_mixture(phases, result, 1.0 - rel_dn, muons,
                                 leptons=leptons)
    P_hi, e_hi = _frozen_mixture(phases, result, 1.0 + rel_dn, muons,
                                 leptons=leptons)
    if not (e_hi > e_lo):
        return float("nan")
    return (P_hi - P_lo) / (e_hi - e_lo)


def _clipped(result, chi):
    """A shallow copy of `result` with chi clamped into [0, 1].

    `solve` deliberately does not clamp chi — its sign is what classifies
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
    Y_C, Y_S, T = point.matter.Y_C, point.matter.Y_S, point.T
    x0 = warm_start(point, flags.phi_field and flags.hyperons,
                          has_muS=flags.has_strange_baryons)

    def at(scale):
        P, eps, n_C_s = _dd2_frozen_block(par, flags, n_B * scale, Y_C, Y_S,
                                          T, x0=x0)
        if leptons:
            P_l, eps_l = _lepton_block(n_C_s, flags.muons, T)
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
        from eos.vmit.parameters import Parameters as VMITParameters
        vmit_params = VMITParameters.default()

    def at(scale):
        P, eps, n_C = _vmit_frozen_block(vmit_params, n_u * scale,
                                         n_d * scale, n_s * scale, T)
        if leptons:
            P_l, eps_l = _lepton_block(n_C, muons, T)
            P, eps = P + P_l, eps + eps_l
        return P, eps

    P_lo, e_lo = at(1.0 - rel_dn)
    P_hi, e_hi = at(1.0 + rel_dn)
    if not (e_hi > e_lo):
        return float("nan")
    return (P_hi - P_lo) / (e_hi - e_lo)


def frozen_along(par, flags, results, vmit_params=None, rel_dn=1e-3,
                 leptons=True, phases=None, muons=None):
    """`sound_speed_frozen` at every state in a sequence, as an array.

    Non-convergent points come back nan rather than aborting the sequence: a
    gap in a diagnostic curve is a better outcome than losing the curve.
    """
    out = []
    for r in results:
        try:
            out.append(sound_speed_frozen(par, flags, r,
                                          vmit_params=vmit_params,
                                          rel_dn=rel_dn, leptons=leptons,
                                          phases=phases, muons=muons))
        except (RuntimeError, ValueError):
            out.append(np.nan)
    return np.asarray(out, dtype=float)
