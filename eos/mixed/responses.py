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

from eos.general.sound_speeds import sound_speed_eq
from eos.general.thermodynamics_leptons import neutralizing_leptons
from eos.mixed.species import mixture_flags


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


def sound_speed_frozen(phases, result, rel_dn=1e-3, leptons=True,
                       species=None):
    """Frozen-composition c_ad^2 = dP/deps at the state `result`.

    phases       : the pairing the state was solved with — two `Phase`
                   objects (`eos.mixed.adapters`), the engine's parameter
                   argument (CLAUDE.md section 5)
    result       : a `Result` from `solve` — pure phases included,
                   chi = 0 and chi = 1 are handled and give the pure-phase
                   frozen sound speed
    rel_dn       : relative density step for the central difference
    leptons      : re-neutralise with leptons (the physical choice for stellar
                   matter, and the default). False gives the matter-only value,
                   which for a DD2 hadronic phase at chi = 0 reproduces
                   `eos.dd2.responses.sound_speed_adiabatic` exactly.
    species      : the ENGINE's own `SpeciesFlags`; only `muons` is read, for
                   the neutralizing lepton gas.

    Read the module docstring before comparing this to a number from elsewhere:
    "frozen" is a convention, and this one freezes chi and each phase's Y_C and
    Y_S. Returns nan if the perturbed states do not bracket a positive deps.
    """
    muons = mixture_flags(species).muons
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


def sound_speed_frozen_pure(phase, th, T=0.0, rel_dn=1e-3, leptons=True,
                            muons=True, mu_slot=None):
    """Frozen c_ad^2 of ONE phase on its own, at the block `th`.

    The chi -> 0 and chi -> 1 limits of `sound_speed_frozen` — the wings
    outside the coexistence window, for a state solved as a pure phase rather
    than as one half of a mixture. Same convention as the mixture: each
    phase's own composition is held by its `frozen_thermo` capability, and the
    leptons are re-neutralised against the frozen charge. The two therefore
    join continuously at a boundary, which is the point of having them agree —
    c_ad is a property of the matter, not of the machinery that solved it.

    `phase` is a `Phase` and `th` its `PhaseThermo` block, so this knows no
    model: whatever a pairing's hadronic side freezes at fixed Y_C and Y_S,
    and whatever its quark side freezes at fixed flavour ratios, is that
    adapter's declaration and not this function's business.

    `mu_slot` is the phase's own baryon-slot potential where it is known; an
    adapter may use it to seed its internal solve (the DD2 one does, and a
    mixed-phase state is nowhere near the nucleonic beta equilibrium its
    fallback seed assumes). `muons` selects whether the neutralising lepton
    gas may contain muons, the role `species.muons` plays in the mixture.

    Returns nan if the perturbed states do not bracket a positive deps.
    """
    if phase.frozen_thermo is None:
        raise NotImplementedError(
            f"the {phase.name} phase has no frozen_thermo capability, so the "
            f"frozen-composition sound speed is not defined for it "
            f"(see docs/DEFERRED.md)")
    if th.n_B <= 0.0:
        return float("nan")

    def at(scale):
        P, eps, n_C = phase.frozen_thermo(th, scale, T, mu_slot=mu_slot)
        if leptons:
            P_l, eps_l = _lepton_block(n_C, muons, T)
            P, eps = P + P_l, eps + eps_l
        return P, eps

    P_lo, e_lo = at(1.0 - rel_dn)
    P_hi, e_hi = at(1.0 + rel_dn)
    if not (e_hi > e_lo):
        return float("nan")
    return (P_hi - P_lo) / (e_hi - e_lo)


def frozen_along(phases, results, rel_dn=1e-3, leptons=True,
                 species=None):
    """`sound_speed_frozen` at every state in a sequence, as an array.

    Non-convergent points come back nan rather than aborting the sequence: a
    gap in a diagnostic curve is a better outcome than losing the curve.
    """
    out = []
    for r in results:
        try:
            out.append(sound_speed_frozen(phases, r, rel_dn=rel_dn,
                                          leptons=leptons, species=species))
        except (RuntimeError, ValueError):
            out.append(np.nan)
    return np.asarray(out, dtype=float)
