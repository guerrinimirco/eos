"""
gmode/sound_speeds.py
=====================
The two sound speeds a composition g-mode needs, and what happens to them when
chemical equilibration proceeds at a finite rate.

A g-mode exists only because a perturbed fluid element responds differently
depending on whether its composition has time to re-equilibrate:

* **equilibrium**, `c_eq^2 = dP/deps` along the equilibrated sequence — the
  composition follows the density, which is what the star does quasi-statically
  and what the TOV equations see;
* **frozen (adiabatic)**, `c_ad^2 = (dP/deps)_x` at fixed composition `x` — the
  element is compressed faster than the reactions that would restore
  equilibrium.

Buoyancy is the difference between them, `N^2 ~ 1/c_eq^2 - 1/c_ad^2`. An
equation of state with a single sound speed supports no composition g-mode.

Which reactions are frozen, and why it is a choice
--------------------------------------------------
The frozen sound speed is only defined once one says *what* is held fixed, and
that follows from comparing reaction timescales with the oscillation period
(a few ms for a g-mode at a few hundred Hz):

* Strong and electromagnetic processes, ~1e-23 s, are always fast. In a
  two-phase mixture this is what keeps the phases in *mechanical* equilibrium,
  i.e. at a common pressure, throughout the oscillation.
* Non-leptonic weak processes in quark matter, u + d <-> u + s, run at ~1e-8 s.
  These are also fast on a g-mode period, so quark-phase strangeness arguably
  re-equilibrates rather than staying frozen.
* Leptonic weak processes (direct and modified Urca) scale as T^-4 and T^-6 and
  are slow for T below roughly 1e10 K. Their slowness is precisely why
  composition g-modes exist in cold neutron stars.
* Converting hadrons into quarks, i.e. changing the quark volume fraction chi,
  needs a high-order weak process and transport across the mixed-phase
  structures. It is slow, so **chi is frozen** — the assumption that makes the
  mixed-phase g-mode large.

Two conventions for the frozen sound speed in a mixed phase
-----------------------------------------------------------
Both are implemented here, because the choice is not settled and it changes the
answer:

1. ``"isobaric"`` (default). The phases share a common pressure perturbation,
   so each responds to the same dP with its own deps:

       1/c_ad,mix^2 = (1 - chi)/c_ad,H^2 + chi/c_ad,Q^2

   the volume-fraction-weighted *reciprocal* combination — pressure adds like a
   voltage across resistors in parallel. This follows from
   eps_mix = (1-chi) eps_H + chi eps_Q at P_H = P_Q, and is the relation used
   in the g-mode literature (Jaikumar, Semposki, Prakash and Constantinou,
   Phys. Rev. D 103, 123009 (2021), Eq. (75)).

2. ``"equal_compression"``. Both phases are scaled by the same density factor
   at fixed chi and the pressures are volume-averaged. This is the convention
   already in `eos.mixed.coefficients.sound_speed_frozen`, and it is delegated
   to unchanged. It does not hold the two phases at a common pressure under the
   perturbation, so it is not the one the mode equations assume; it is offered
   so the two can be compared on the same star.

Both reduce to the same pure-phase value at chi = 0 and chi = 1, so they differ
only inside the mixed-phase window — which is exactly where the g-mode signal
comes from.

Finite reaction rates
---------------------
When the equilibration rate `gamma` is comparable to the mode frequency the
sound speed is neither of the two limits but a complex, frequency-dependent
interpolation between them,

    c_dy^2 = c_eq^2 + (c_ad^2 - c_eq^2) / (1 + gamma / (i omega))

so gamma << omega recovers `c_ad^2` (frozen), gamma >> omega recovers `c_eq^2`
(buoyancy destroyed, the mode is suppressed), and gamma ~ omega gives maximal
dissipation. The imaginary part is bulk viscosity,
zeta = (eps + P) Im[c_dy^2] / omega, and feeding `c_dy^2` into `N^2` turns the
mode frequency complex, its imaginary part being the damping rate. Reference:
Counsell et al., "Suppression of composition g-modes in chemically-equilibrating
warm neutron stars", arXiv:2504.12230.

Units are fm-based on every boundary, as everywhere in `eos`: n_B in fm^-3,
P and eps in MeV/fm^3, T in MeV. Sound speeds are dimensionless (units of c),
`gamma` and `omega` are in s^-1.
"""
import numpy as np
from dataclasses import replace

from eos.general.physics_constants import hc3
from eos.general.particles import Electron, Muon
from eos.mixed.coefficients import sound_speed_eq, sound_speed_frozen

ISOBARIC = "isobaric"
EQUAL_COMPRESSION = "equal_compression"
CONVENTIONS = (ISOBARIC, EQUAL_COMPRESSION)


def cs2_frozen_isobaric(cs2_H, cs2_Q, chi):
    """Combine per-phase frozen sound speeds at a common pressure.

        1/c_ad,mix^2 = (1 - chi)/c_ad,H^2 + chi/c_ad,Q^2

    cs2_H, cs2_Q : frozen c^2 of the hadronic and quark phases (dimensionless),
                   each evaluated at *its own* density in the mixture
    chi          : quark volume fraction in [0, 1]

    Scalars or broadcastable arrays. At chi = 0 this returns cs2_H exactly and
    at chi = 1 it returns cs2_Q, so the pure phases need no special-casing;
    whichever phase carries zero weight may be passed as nan.
    """
    chi = np.clip(np.asarray(chi, dtype=float), 0.0, 1.0)
    cs2_H = np.asarray(cs2_H, dtype=float)
    cs2_Q = np.asarray(cs2_Q, dtype=float)
    # Drop the absent phase before dividing, so a nan there cannot poison a
    # limit that does not depend on it.
    inv_H = np.where(chi < 1.0, (1.0 - chi) / np.where(chi < 1.0, cs2_H, 1.0), 0.0)
    inv_Q = np.where(chi > 0.0, chi / np.where(chi > 0.0, cs2_Q, 1.0), 0.0)
    total = inv_H + inv_Q
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total > 0.0, 1.0 / total, np.nan)


def cs2_frozen_nucleonic(par, n_B, Y_p, T=0.0, muons=True, rel_dn=1e-3,
                         leptons=True):
    """Frozen c_ad^2 of charge-neutral nucleonic matter, leptons included.

    par     : DD2 `Parametrization`
    n_B     : baryon density [fm^-3]
    Y_p     : proton fraction, held fixed under the perturbation
    leptons : include the neutralising lepton gas (the default, and the
              physically right choice for stellar matter)

    Why this exists rather than `eos.dd2.coefficients.sound_speed_adiabatic`:
    that function is deliberately leptonless, being a probe of the nucleonic
    sector alone, whereas `eos.dd2.coefficients.sound_speed_eq` follows the
    beta-equilibrium sequence *with* leptons. Differencing the two would
    compare different fluids, and since the lepton contribution to c_ad is a
    few per cent of c_ad -- comparable to the whole c_ad^2 - c_eq^2 signal that
    drives the g-mode -- the mismatch is a leading-order error in N^2, not a
    refinement. Setting `leptons=False` reproduces
    `eos.dd2.coefficients.sound_speed_adiabatic` exactly.

    Frozen here means fixed Y_p, hence fixed lepton fractions too: the star is
    compressed faster than the Urca processes can convert neutrons to protons.
    """
    from eos.dd2.solver import solve_composition
    from eos.dd2.physics.octet import _yc_neutralizing_leptons

    def state(scale):
        n = n_B * scale
        n_p = Y_p * n
        pt = solve_composition(par, n - n_p, n_p, T=T, check_consistency=False)
        P, eps = pt.P, pt.eps
        if leptons:
            _mu_e, e_blk, mu_blk = _yc_neutralizing_leptons(
                n_p * hc3, Electron.mass, Muon.mass, muons, T)
            P += (e_blk[1] + mu_blk[1]) / hc3
            eps += (e_blk[2] + mu_blk[2]) / hc3
        return P, eps

    P_lo, e_lo = state(1.0 - rel_dn)
    P_hi, e_hi = state(1.0 + rel_dn)
    if not (e_hi > e_lo):
        return float("nan")
    return (P_hi - P_lo) / (e_hi - e_lo)


def cs2_frozen_point(par, flags, result, vmit_params=None,
                     convention=ISOBARIC, rel_dn=1e-3, leptons=True):
    """Frozen c_ad^2 at one solved mixed-phase state, under either convention.

    par, flags  : the DD2 `Parametrization` and `SpeciesFlags` the state used
    result      : a `MixedResult` from `eos.mixed.solve_mixed`; pure phases
                  (chi <= 0 or chi >= 1) are handled and give the pure-phase
                  value under either convention
    vmit_params : the `VMITParams` the state used
    convention  : "isobaric" or "equal_compression" — see the module docstring
    leptons     : re-neutralise each phase with leptons, as
                  `eos.mixed.coefficients.sound_speed_frozen` does

    The per-phase sound speeds the isobaric relation needs are obtained by
    evaluating the existing `sound_speed_frozen` at the same state with chi
    forced to 0 and to 1: that function already compresses one phase at frozen
    composition and re-neutralises it, which is exactly the pure-phase frozen
    sound speed at that phase's own density.
    """
    if convention not in CONVENTIONS:
        raise ValueError(f"convention must be one of {CONVENTIONS}, "
                         f"got {convention!r}")

    kw = dict(vmit_params=vmit_params, rel_dn=rel_dn, leptons=leptons)
    if convention == EQUAL_COMPRESSION:
        return sound_speed_frozen(par, flags, result, **kw)

    chi = float(np.clip(result.chi, 0.0, 1.0))
    cs2_H = (sound_speed_frozen(par, flags, replace(result, chi=0.0), **kw)
             if chi < 1.0 else np.nan)
    cs2_Q = (sound_speed_frozen(par, flags, replace(result, chi=1.0), **kw)
             if chi > 0.0 else np.nan)
    return float(cs2_frozen_isobaric(cs2_H, cs2_Q, chi))


def cs2_frozen_along(par, flags, results, vmit_params=None,
                     convention=ISOBARIC, rel_dn=1e-3, leptons=True):
    """`cs2_frozen_point` at every state of a sequence, as an array.

    Non-convergent points come back nan rather than aborting the sequence, so a
    single bad density leaves a gap in the curve instead of destroying it. This
    mirrors `eos.mixed.coefficients.frozen_along`.
    """
    out = []
    for r in results:
        try:
            out.append(cs2_frozen_point(par, flags, r, vmit_params=vmit_params,
                                        convention=convention, rel_dn=rel_dn,
                                        leptons=leptons))
        except (RuntimeError, ValueError):
            out.append(np.nan)
    return np.asarray(out, dtype=float)


def cs2_dynamical(cs2_eq, cs2_ad, gamma, omega):
    """Complex dynamical sound speed at a finite equilibration rate.

        c_dy^2 = c_eq^2 + (c_ad^2 - c_eq^2) / (1 + gamma/(i omega))

    equivalently, with x = omega^2/(omega^2 + gamma^2),

        Re[c_dy^2] = c_eq^2 + (c_ad^2 - c_eq^2) x
        Im[c_dy^2] = (c_ad^2 - c_eq^2) x gamma/omega   >= 0

    gamma : chemical equilibration rate [s^-1], scalar or array
    omega : angular frequency of the perturbation [s^-1]

    gamma -> 0 gives c_ad^2 (frozen), gamma -> infinity gives c_eq^2, and
    gamma ~ omega maximises Im, i.e. bulk-viscous dissipation. Returns complex
    even in the limits, so downstream code has one dtype to handle.
    """
    cs2_eq = np.asarray(cs2_eq, dtype=float)
    cs2_ad = np.asarray(cs2_ad, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    omega = float(omega)
    if omega <= 0.0:
        raise ValueError(f"omega must be positive, got {omega}")
    # 1/(1 + gamma/(i omega)) = 1/(1 - i gamma/omega) = (omega^2 + i omega gamma)
    #                                                   / (omega^2 + gamma^2)
    ratio = gamma / omega
    weight = 1.0 / (1.0 - 1j * ratio)
    return (cs2_eq + (cs2_ad - cs2_eq) * weight).astype(complex)


def bulk_viscosity(cs2_dy, eps, P, omega):
    """zeta = (eps + P) Im[c_dy^2] / omega, from the dynamical sound speed.

    eps, P in MeV/fm^3 and omega in s^-1, so zeta comes back in MeV s / fm^3.
    A diagnostic: the mode damping it implies is already contained in the
    imaginary part of the eigenfrequency the Cowling solver returns.
    """
    return (np.asarray(eps) + np.asarray(P)) * np.imag(cs2_dy) / float(omega)


__all__ = [
    "sound_speed_eq", "cs2_frozen_isobaric", "cs2_frozen_nucleonic",
    "cs2_frozen_point",
    "cs2_frozen_along", "cs2_dynamical", "bulk_viscosity",
    "ISOBARIC", "EQUAL_COMPRESSION", "CONVENTIONS",
]
