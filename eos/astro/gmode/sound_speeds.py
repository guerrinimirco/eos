"""
gmode/sound_speeds.py
=====================
What happens to the two sound speeds a composition g-mode needs when chemical
equilibration proceeds at a finite rate.

The speeds themselves are not computed here. A g-mode exists only because a
perturbed fluid element responds differently depending on whether its
composition has time to re-equilibrate:

* **equilibrium**, `c_e^2 = dP/deps` along the equilibrated sequence -- the
  composition follows the density, which is what the star does
  quasi-statically and what the TOV equations see;
* **frozen (adiabatic)**, `c_s^2 = (dP/deps)_x` at fixed composition `x` -- the
  element is compressed faster than the reactions that would restore
  equilibrium.

Buoyancy is the difference between them, `N^2 ~ 1/c_e^2 - 1/c_s^2`. An equation
of state with a single sound speed supports no composition g-mode.

Both are produced by the MODEL and reach this package as the two columns of an
`eos.general.sound_speeds.EOSTable_for_gmode` -- the contract that lets
`astro/` consume a model's second derivatives without importing the model
(CLAUDE.md section 1). `sound_speed_eq` and `cs2_frozen_isobaric` are
re-exported from there for callers who assemble a table by hand.

Which reactions are frozen, and why it is a choice
--------------------------------------------------
The frozen sound speed is only defined once one says *what* is held fixed, and
that follows from comparing reaction timescales with the oscillation period
(a few ms for a g-mode at a few hundred Hz):

* Strong and electromagnetic processes, ~1e-23 s, are always fast. In a
  two-phase mixture this is what keeps the phases in *mechanical* equilibrium,
  i.e. at a common pressure, throughout the oscillation -- which is what
  `cs2_frozen_isobaric` assumes when it combines two phases.
* Non-leptonic weak processes in quark matter, u + d <-> u + s, run at ~1e-8 s.
  These are also fast on a g-mode period, so quark-phase strangeness arguably
  re-equilibrates rather than staying frozen.
* Leptonic weak processes (direct and modified Urca) scale as T^-4 and T^-6 and
  are slow for T below roughly 1e10 K. Their slowness is precisely why
  composition g-modes exist in cold neutron stars.
* Converting hadrons into quarks, i.e. changing the quark volume fraction chi,
  needs a high-order weak process and transport across the mixed-phase
  structures. It is slow, so **chi is frozen** -- the assumption that makes the
  mixed-phase g-mode large.

Finite reaction rates
---------------------
When the equilibration rate `gamma` is comparable to the mode frequency the
sound speed is neither of the two limits but a complex, frequency-dependent
interpolation between them,

    c_dy^2 = c_e^2 + (c_s^2 - c_e^2) / (1 + gamma / (i omega))

written for a perturbation going as e^{+i omega t}, the convention
`eos.astro.gmode.cowling` therefore adopts as well. In it a damped mode has
Im(omega) > 0.

So gamma << omega recovers `c_s^2` (frozen), gamma >> omega recovers `c_e^2`
(buoyancy destroyed, the mode is suppressed), and gamma ~ omega gives maximal
dissipation. The imaginary part is bulk viscosity,
zeta = (eps + P) Im[c_dy^2] / omega, and feeding `c_dy^2` into `N^2` turns the
mode frequency complex, its imaginary part being the damping rate. Reference:
Counsell et al., "Suppression of composition g-modes in chemically-equilibrating
warm neutron stars", arXiv:2504.12230.

This is also why the stellar background calls its second slot `cs2_ad` rather
than `cs2_frozen`: once a finite rate is folded in by
`StellarBackground.at_frequency`, what it carries is the dynamical speed, and
"frozen" would be false. The TABLE's column is `cs2_frozen`, because a table
is always the strict limit.

Units are fm-based on every boundary, as everywhere in `eos`: n_B in fm^-3,
P and eps in MeV/fm^3, T in MeV. Sound speeds are dimensionless (units of c),
`gamma` and `omega` are in s^-1.
"""
import numpy as np

from eos.general.sound_speeds import sound_speed_eq, cs2_frozen_isobaric


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
    "sound_speed_eq", "cs2_frozen_isobaric", "cs2_dynamical", "bulk_viscosity",
]
