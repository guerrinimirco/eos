"""
eos.astro.gmode
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

What a model owes, and what this package does with it
----------------------------------------------------
The two sound speeds are the entire physics input, and they arrive as an
`eos.general.sound_speeds.EOSTable_for_gmode`: the (P, eps, n_B) table a
structure solver integrates, plus `cs2_equilibrium` and `cs2_frozen` on the
same rows. That table is the CONTRACT (CLAUDE.md section 1). A model or a
composite engine PRODUCES one -- both speeds come from its own `eos_response`,
under `frozen='equilibrium'` and `frozen='composition'` -- and this package
CONSUMES one. Nothing here imports a model, and no model imports `astro/`.

At present `eos.dd2` is the only model whose `eos_response` implements the
composition freeze, so it is the only one that can fill the contract; a model
that cannot raises saying so. That is a per-model, visible gap rather than a
hidden `from eos.dd2.solver import ...`, which is what it replaced.

The tables are T = 0. Zhao and Lattimer's condition is "without varying
chemical composition", not the vanishing temperature: T = 0 collapses the
thermal axis and leaves the composition axis -- the one the g-mode lives on --
entirely intact. Finite T is future work.

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
    from eos.general.sound_speeds import EOSTable_for_gmode
    from eos.astro.gmode import gmode_frequency, with_crust

    table = EOSTable_for_gmode.from_columns(P, eps, n_B, cs2_frozen)
    mode = gmode_frequency(with_crust(table), M_target=1.4)
    print(mode.label, mode.nu_hz)

`eos.astro.gmode.verify.run_full_check.dd2_table` builds one for cold DD2
npemu matter and is the worked example of the producer side.

Everything is in the relativistic Cowling approximation, which gives real
eigenfrequencies accurate to a few per cent for g-modes but no
gravitational-wave damping time. A finite reaction rate does produce a damping
time, the bulk-viscous one; pass `gamma` to `gmode_frequency` for it.
"""
from eos.astro.gmode.background import (
    StellarBackground, build_background, with_crust, brunt_vaisala,
    omega_to_hz, hz_to_omega,
)
from eos.general.sound_speeds import EOSTable_for_gmode
from eos.astro.gmode.sound_speeds import (
    sound_speed_eq, cs2_frozen_isobaric, cs2_dynamical, bulk_viscosity,
)
from eos.astro.gmode.rates import (
    WeakCouplings, WEAK_COUPLINGS,
    equilibration_rate, equilibration_rate_along, susceptibility_A,
    lambda_direct_urca, lambda_modified_urca,
)
from eos.astro.gmode.cowling import (
    Mode, mode_spectrum, solve_gmode, gmode_frequency, integrate_mode,
    surface_discriminant,
)

__all__ = [
    # background
    "StellarBackground", "build_background", "with_crust", "brunt_vaisala",
    "omega_to_hz", "hz_to_omega",
    # the contract, and the sound speeds
    "EOSTable_for_gmode",
    "sound_speed_eq", "cs2_frozen_isobaric", "cs2_dynamical", "bulk_viscosity",
    # rates
    "WeakCouplings", "WEAK_COUPLINGS",
    "equilibration_rate", "equilibration_rate_along", "susceptibility_A",
    "lambda_direct_urca", "lambda_modified_urca",
    # modes
    "Mode", "mode_spectrum", "solve_gmode", "gmode_frequency",
    "integrate_mode", "surface_discriminant",
]
