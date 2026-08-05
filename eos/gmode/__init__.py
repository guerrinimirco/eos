"""
eos.gmode
=========
Composition g-modes of neutron stars: the oscillation whose restoring force is
buoyancy, and which therefore measures the *composition gradient* of dense
matter rather than only its pressure-density relation.

Why this is worth computing
---------------------------
Mass, radius and tidal deformability all depend on the equation of state
through P(eps) alone, and hybrid stars can be built to mimic purely nucleonic
ones in all three. The g-mode cannot be mimicked that way. Its frequency is set
by the Brunt-Vaisala frequency

    N^2 = g^2 (1/c_eq^2 - 1/c_ad^2) e^{nu - lambda}

which vanishes identically unless the equilibrium and frozen sound speeds
differ -- that is, unless the matter's composition changes with density and is
slow to re-equilibrate. Across a quark-hadron mixed phase c_eq drops sharply
while c_ad does not, and the fundamental g-mode frequency roughly doubles, so
detecting one would be evidence for non-nucleonic matter that a mass-radius
measurement cannot supply.

Layout
------
`background`    TOV structure with the radial profiles kept, both metric
                functions, the local gravity, and N^2.
`sound_speeds`  the two sound speeds, the two conventions for combining them
                across a mixed phase, and the complex dynamical sound speed
                that finite reaction rates produce.
`rates`         beta-equilibration rate gamma from Urca processes, which decides
                whether the composition is frozen on the oscillation period.
`cowling`       the eigenvalue problem and its solution.

Quick start
-----------
    from eos.gmode import gmode_frequency, with_crust

    eos, c_eq, c_ad = with_crust(core_table, c_eq_core, c_ad_core)
    mode = gmode_frequency(eos, c_eq, c_ad, M_target=1.4)
    print(mode.label, mode.nu_hz)

The two sound speeds are the entire physics input. Supply them from wherever
the equation of state lives: `eos.gmode.sound_speeds.cs2_frozen_nucleonic` and
`eos.dd2.coefficients.sound_speed_eq` for a nucleonic star,
`eos.gmode.sound_speeds.cs2_frozen_along` and
`eos.mixed.coefficients.sound_speed_eq` for a DD2 + vMIT hybrid.

Everything is in the relativistic Cowling approximation, which gives real
eigenfrequencies accurate to a few per cent for g-modes but no
gravitational-wave damping time. A finite reaction rate does produce a damping
time, the bulk-viscous one; pass `gamma` to `gmode_frequency` for it.
"""
from eos.gmode.background import (
    StellarBackground, build_background, with_crust, brunt_vaisala,
    omega_to_hz, hz_to_omega,
)
from eos.gmode.sound_speeds import (
    sound_speed_eq, cs2_frozen_isobaric, cs2_frozen_nucleonic,
    cs2_frozen_point, cs2_frozen_along, cs2_dynamical, bulk_viscosity,
    ISOBARIC, EQUAL_COMPRESSION, CONVENTIONS,
)
from eos.gmode.rates import (
    equilibration_rate, equilibration_rate_along, susceptibility_A,
    lambda_direct_urca, lambda_modified_urca,
)
from eos.gmode.cowling import (
    Mode, mode_spectrum, solve_gmode, gmode_frequency, integrate_mode,
    surface_discriminant,
)

__all__ = [
    # background
    "StellarBackground", "build_background", "with_crust", "brunt_vaisala",
    "omega_to_hz", "hz_to_omega",
    # sound speeds
    "sound_speed_eq", "cs2_frozen_isobaric", "cs2_frozen_nucleonic",
    "cs2_frozen_point", "cs2_frozen_along", "cs2_dynamical", "bulk_viscosity",
    "ISOBARIC", "EQUAL_COMPRESSION", "CONVENTIONS",
    # rates
    "equilibration_rate", "equilibration_rate_along", "susceptibility_A",
    "lambda_direct_urca", "lambda_modified_urca",
    # modes
    "Mode", "mode_spectrum", "solve_gmode", "gmode_frequency",
    "integrate_mode", "surface_discriminant",
]
