"""
ABPR: colour-flavour locked quark matter at T = 0, in closed form.

The analytic parametrization of Alford, Braby, Paris and Reddy: a free
three-flavour quark gas carrying the leading perturbative QCD correction in a
single factor a4, the leading cost of the strange quark mass as an expansion
in m_s^2/mu^2, the CFL condensation energy 3 Delta0^2 mu^2/pi^2, and a bag
constant. Pairing locks the three flavour densities together, so the
composition is fixed, the phase is electrically neutral with no leptons, and
the whole model is a polynomial in the common quark chemical potential.

Nothing here iterates: P, n_B and eps are polynomials in mu, and the three
inverse maps mu(n_B), mu(P) and mu(eps) are closed forms as well.

This is the T = 0 analytic limit of the colour-flavour locked phase of
`eos.alphabag`, which carries the strange quark mass exactly through the Fermi
integrals where this model expands it; the two are driven as a matched pair
through alpha_s = pi/2 (1 - a4), and the difference between them is the
O(m_s^4) term measured in `verify/run_full_check.py`.

See `abpr.tex` for the physics.

Reference: M. Alford, M. Braby, M. Paris and S. Reddy, Astrophys. J. 629, 969
(2005).
"""
from eos.abpr.parameters import Parameters
from eos.abpr.species import SpeciesFlags
from eos.abpr.thermodynamics import (
    Thermo, coefficients, pressure, baryon_density, energy_density, entropy,
    sound_speed_squared, thermo_from_mu,
)
from eos.abpr.solver import (
    CFLPoint, MODE_FRACTIONS, MODE_REFUSALS, check_mode, check_temperature,
    mu_from_nB, mu_from_P, mu_from_eps, point_from_mu, response_at_mu,
    solve_cfl,
)
from eos.abpr.api import (
    PointResult, TableResult, RESPONSE_FREEZES, cfl_row,
    eos_point, eos_table, eos_response,
)

__all__ = [
    "Parameters", "SpeciesFlags",
    "Thermo", "coefficients", "pressure", "baryon_density", "energy_density",
    "entropy", "sound_speed_squared", "thermo_from_mu",
    "CFLPoint", "MODE_FRACTIONS", "MODE_REFUSALS", "check_mode",
    "check_temperature", "mu_from_nB", "mu_from_P", "mu_from_eps",
    "point_from_mu", "response_at_mu", "solve_cfl",
    "PointResult", "TableResult", "RESPONSE_FREEZES", "cfl_row",
    "eos_point", "eos_table", "eos_response",
]
