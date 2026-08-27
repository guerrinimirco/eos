"""Thermodynamic kernels for vMIT quark matter.

Three quark flavours in a bag of constant energy density B, interacting
through a flavour-blind isoscalar-vector field. The quark masses are the
current masses and are parameters, so there is no scalar condensate and no gap
equation: the only mean field is the vector one, and its equation of motion is
algebraic,

    V = (g_V^2 / m_V^2) sum_q n_q = a hbar c (n_u + n_d + n_s)      [MeV]

carried through the single identifiable combination a = g_V^2/m_V^2 in fm^2.
Being flavour blind it shifts all three potentials equally, so what enters the
Fermi integrals is the effective (kinetic) potential mu_eff_q = mu_q - V.

The kinetic sector is a free Fermi gas per flavour with degeneracy
g = 2 (spin) x 3 (colour) = 6, evaluated with antiparticles through the JEL
integrals of `eos.general.fermi_integrals`. The vector field and the bag add

    P_V = eps_V = (1/2) a hbar c (sum_q n_q)^2
    P_B = -B/(hbar c)^3            eps_B = +B/(hbar c)^3

A vector field enters P and eps with the SAME sign; the bag with opposite
signs, which is what makes the pressure negative below deconfinement. The
Euler relation eps + P = T s + sum_q mu_q n_q then holds identically: the bag
terms cancel and 2 P_V is exactly the shift sum_q (mu_q - mu_eff_q) n_q.

Reference: Chodos et al., Phys. Rev. D 9, 3471 (1974) for the bag; the vector
term in the form used by Gomes et al., Astrophys. J. 877, 139 (2019) and
Constantinou et al., Phys. Rev. D 104, 123032 (2021) and 107, 074013 (2023).
See `vmit.tex` for the full description.

Units at every boundary: densities in fm^-3, masses/potentials/temperature in
MeV, P and eps in MeV/fm^3.
"""
from dataclasses import dataclass
from typing import Tuple

from eos.general.physics_constants import hc, hc3
from eos.vmit.parameters import Parameters
from eos.general.fermi_integrals import solve_fermi_jel, invert_fermi_density
from eos.general import particles
from eos.general.basis import quark_charges, charge_potentials_from_quarks


#: Spin (2) x colour (3). Taken from the particle table, not written as 6.
G_QUARK = particles.get_particle("quark").g_degen


@dataclass
class QuarkThermo:
    """Kinetic thermodynamics of a single quark flavour."""
    n: float = 0.0      # Number density (fm^-3)
    P: float = 0.0      # Pressure (MeV/fm^3)
    e: float = 0.0      # Energy density (MeV/fm^3)
    s: float = 0.0      # Entropy density (fm^-3)
    f: float = 0.0      # Free energy density (MeV/fm^3)


@dataclass
class MatterThermo:
    """A full vMIT quark-matter state (no leptons)."""
    # Inputs
    n_u: float = 0.0       # Up quark density (fm^-3)
    n_d: float = 0.0       # Down quark density (fm^-3)
    n_s: float = 0.0       # Strange quark density (fm^-3)
    n_B: float = 0.0       # Baryon density (fm^-3)
    n_C: float = 0.0       # Non-leptonic charge density (fm^-3)
    n_S: float = 0.0       # Strangeness density, +1 per s quark (fm^-3)
    T: float = 0.0         # Temperature (MeV)
    mu_u: float = 0.0      # Up quark physical chemical potential (MeV)
    mu_d: float = 0.0      # Down quark physical chemical potential (MeV)
    mu_s: float = 0.0      # Strange quark physical chemical potential (MeV)
    # Outputs
    P: float = 0.0         # Total pressure (MeV/fm^3)
    e: float = 0.0         # Total energy density (MeV/fm^3)
    s: float = 0.0         # Total entropy density (fm^-3)
    f: float = 0.0         # Free energy density f = e - s*T (MeV/fm^3)
    Y_C: float = 0.0       # Charge fraction n_C/n_B
    Y_S: float = 0.0       # Strangeness fraction n_S/n_B
    mu_B: float = 0.0      # Baryon chemical potential (MeV)
    mu_C: float = 0.0      # Charge chemical potential (MeV)
    mu_S: float = 0.0      # Strangeness chemical potential (MeV)


# =============================================================================
# SINGLE QUARK THERMODYNAMICS
# =============================================================================
def kinetic_thermo(mu_eff: float, T: float, m: float,
                   include_antiparticles: bool = True) -> QuarkThermo:
    """Kinetic n, P, e, s of one flavour at effective potential `mu_eff`."""
    result = solve_fermi_jel(mu_eff, T, m, G_QUARK,
                             include_antiparticles=include_antiparticles)

    return QuarkThermo(
        n=result[0],
        P=result[1],
        e=result[2],
        s=result[3]
    )


#: A flavour that is not a degree of freedom of the matter: no population, no
#: pressure, no energy, no entropy. What `SpeciesFlags.two_flavour` puts in
#: the strange slot, so the sector is off because the flag says so and not
#: because its potential happened to sit below a threshold.
_EMPTY_FLAVOUR = QuarkThermo()


def quark_density(mu_eff: float, T: float, m: float,
                  include_antiparticles: bool = True) -> float:
    """Number density of one flavour at effective potential `mu_eff`."""
    result = solve_fermi_jel(mu_eff, T, m, G_QUARK,
                             include_antiparticles=include_antiparticles)
    return result[0]


# =============================================================================
# VECTOR FIELD
# =============================================================================
def vector_field(n_u: float, n_d: float, n_s: float,
                 params: Parameters) -> float:
    """The vector mean field V = a hbar c (n_u + n_d + n_s), in MeV.

    a is in fm^2 and the densities in fm^-3, so hbar c (MeV fm) closes the
    units. Flavour blind: the same V shifts all three potentials.
    """
    n_total = n_u + n_d + n_s
    return params.a * hc * n_total


def vector_pressure(n_u: float, n_d: float, n_s: float,
                    params: Parameters) -> float:
    """P_V = (1/2) a hbar c (n_u + n_d + n_s)^2, in MeV/fm^3."""
    n_total = n_u + n_d + n_s
    return 0.5 * params.a * hc * n_total**2


def vector_energy(n_u: float, n_d: float, n_s: float,
                  params: Parameters) -> float:
    """eps_V = P_V: a vector field contributes equally to P and eps."""
    return vector_pressure(n_u, n_d, n_s, params)


# =============================================================================
# BAG CONSTANT
# =============================================================================
def bag_pressure(params: Parameters) -> float:
    """P_B = -B/(hbar c)^3, in MeV/fm^3 — negative, the confining pressure."""
    return -params.B / hc3


def bag_energy(params: Parameters) -> float:
    """eps_B = +B/(hbar c)^3, in MeV/fm^3 — the bag's energy density."""
    return params.B / hc3


# =============================================================================
# EFFECTIVE CHEMICAL POTENTIAL
# =============================================================================
def effective_potential(mu: float, n_u: float, n_d: float, n_s: float,
                        params: Parameters) -> float:
    """mu_eff = mu - V, the potential the Fermi integrals are evaluated at."""
    V = vector_field(n_u, n_d, n_s, params)
    return mu - V


def effective_potentials(
    mu_u: float, mu_d: float, mu_s: float,
    n_u: float, n_d: float, n_s: float,
    params: Parameters
) -> Tuple[float, float, float]:
    """(mu_eff_u, mu_eff_d, mu_eff_s) = (mu_u, mu_d, mu_s) - V.

    One V for all three flavours: the vector coupling is flavour blind, so the
    isospin and strangeness structure of the state lives entirely in the
    physical potentials.
    """
    V = vector_field(n_u, n_d, n_s, params)
    return mu_u - V, mu_d - V, mu_s - V


@dataclass
class QuarkMuDensity:
    """Effective potentials and the densities they imply, for the solvers.

    Carries both the mean-field densities the caller supplied (`n_u`, ...) and
    the densities the effective potentials produce (`n_u_calc`, ...). The
    solver's job is to make the two agree; the conserved charges are exposed
    for each set, so a residual can be written against either.
    """
    # Physical chemical potentials (MeV)
    mu_u: float = 0.0
    mu_d: float = 0.0
    mu_s: float = 0.0
    # Input number densities, i.e. the mean-field values (fm^-3)
    n_u: float = 0.0
    n_d: float = 0.0
    n_s: float = 0.0
    # Effective chemical potentials (MeV)
    mu_eff_u: float = 0.0
    mu_eff_d: float = 0.0
    mu_eff_s: float = 0.0
    # Densities implied by the effective potentials (fm^-3)
    n_u_calc: float = 0.0
    n_d_calc: float = 0.0
    n_s_calc: float = 0.0

    # -- conserved charges of the computed densities ------------------------
    @property
    def n_B_calc(self) -> float:
        """Baryon density of the computed densities."""
        return quark_charges(self.n_u_calc, self.n_d_calc, self.n_s_calc)[0]

    @property
    def n_C_calc(self) -> float:
        """Non-leptonic charge density of the computed densities."""
        return quark_charges(self.n_u_calc, self.n_d_calc, self.n_s_calc)[1]

    @property
    def n_S_calc(self) -> float:
        """Strangeness density of the computed densities, +1 per s quark."""
        return quark_charges(self.n_u_calc, self.n_d_calc, self.n_s_calc)[2]

    # -- conserved charges of the mean-field densities ----------------------
    @property
    def n_B(self) -> float:
        """Baryon density of the mean-field densities."""
        return quark_charges(self.n_u, self.n_d, self.n_s)[0]

    @property
    def n_C(self) -> float:
        """Non-leptonic charge density of the mean-field densities."""
        return quark_charges(self.n_u, self.n_d, self.n_s)[1]

    @property
    def n_S(self) -> float:
        """Strangeness density of the mean-field densities, +1 per s quark."""
        return quark_charges(self.n_u, self.n_d, self.n_s)[2]

    # -- conserved-charge potentials ----------------------------------------
    @property
    def mu_B(self) -> float:
        """mu_B = mu_u + 2 mu_d."""
        return charge_potentials_from_quarks(self.mu_u, self.mu_d, self.mu_s)[0]

    @property
    def mu_C(self) -> float:
        """mu_C = mu_u - mu_d."""
        return charge_potentials_from_quarks(self.mu_u, self.mu_d, self.mu_s)[1]

    @property
    def mu_S(self) -> float:
        """mu_S = mu_s - mu_d, with S = +1 per s quark."""
        return charge_potentials_from_quarks(self.mu_u, self.mu_d, self.mu_s)[2]


def effective_state(
    mu_u: float, mu_d: float, mu_s: float,
    n_u: float, n_d: float, n_s: float,
    T: float, params: Parameters, two_flavour: bool = False
) -> QuarkMuDensity:
    """Effective potentials at the given mean field, and the densities they
    imply — the inner step of every solver in `eos.vmit.solver`.

    `n_u, n_d, n_s` enter only through the vector field V; the returned
    `n_*_calc` are what the resulting effective potentials produce. The solvers
    require `n_*_calc == n_*`, which is the self-consistency of V.

    `two_flavour` is `SpeciesFlags.two_flavour`: with it on the strange
    flavour is not a degree of freedom of the matter, so `n_s_calc` is zero
    whatever mu_eff_s happens to be. The solvers' own `n_s_calc == n_s` row
    then pins n_s to zero, which is what removes the sector rather than
    letting it be emptied by a fraction that happens to vanish (CLAUDE.md
    section 4).
    """
    m_u, m_d, m_s = params.m_u, params.m_d, params.m_s

    mu_eff_u, mu_eff_d, mu_eff_s = effective_potentials(
        mu_u, mu_d, mu_s, n_u, n_d, n_s, params)

    n_u_calc = quark_density(mu_eff_u, T, m_u)
    n_d_calc = quark_density(mu_eff_d, T, m_d)
    n_s_calc = 0.0 if two_flavour else quark_density(mu_eff_s, T, m_s)

    return QuarkMuDensity(
        mu_u=mu_u, mu_d=mu_d, mu_s=mu_s,
        mu_eff_u=mu_eff_u, mu_eff_d=mu_eff_d, mu_eff_s=mu_eff_s,
        n_u_calc=n_u_calc, n_d_calc=n_d_calc, n_s_calc=n_s_calc,
        n_u=n_u, n_d=n_d, n_s=n_s
    )


def physical_potentials(mu_eff: float, n_u: float, n_d: float, n_s: float,
                        params: Parameters) -> float:
    """mu = mu_eff + V, the inverse of `effective_potential`."""
    V = vector_field(n_u, n_d, n_s, params)
    return mu_eff + V


# =============================================================================
# FULL QUARK MATTER THERMODYNAMICS (without leptons)
# =============================================================================
def thermo_from_mu_n(
    mu_u: float, mu_d: float, mu_s: float,
    n_u: float, n_d: float, n_s: float,
    T: float, params: Parameters = None, two_flavour: bool = False
) -> MatterThermo:
    """Assemble a full quark-matter state from potentials and the mean field.

    Both the potentials and the densities are inputs because the densities
    play the role of the mean field, exactly as the meson fields do in an RMF
    model: they set V, and V sets the effective potentials that produce the
    densities. A converged state satisfies n_q(mu_eff_q, T) = n_q, which is
    what the solvers in `eos.vmit.solver` impose; away from a solution the
    returned densities are the computed ones, not the ones passed in.

    `two_flavour` is `SpeciesFlags.two_flavour`: the strange flavour carries
    no population, no pressure, no energy and no entropy, and mu_S is zero
    because no species left in the state carries strangeness. mu_B and mu_C
    are untouched -- the basis map has mu_B = mu_u + 2 mu_d and
    mu_C = mu_u - mu_d, neither of which reads mu_s -- so the two-flavour
    surface satisfies the same E/A = mu_B + Y_S mu_S with both strange terms
    identically zero.
    """
    if params is None:
        params = Parameters.default()

    m_u, m_d, m_s = params.m_u, params.m_d, params.m_s

    mu_eff_u, mu_eff_d, mu_eff_s = effective_potentials(
        mu_u, mu_d, mu_s, n_u, n_d, n_s, params
    )

    # Kinetic contributions from the Fermi integrals
    thermo_u = kinetic_thermo(mu_eff_u, T, m_u)
    thermo_d = kinetic_thermo(mu_eff_d, T, m_d)
    thermo_s = (_EMPTY_FLAVOUR if two_flavour
                else kinetic_thermo(mu_eff_s, T, m_s))

    n_u_calc = thermo_u.n
    n_d_calc = thermo_d.n
    n_s_calc = thermo_s.n

    P_kin = thermo_u.P + thermo_d.P + thermo_s.P
    e_kin = thermo_u.e + thermo_d.e + thermo_s.e
    s_kin = thermo_u.s + thermo_d.s + thermo_s.s

    # Vector and bag contributions
    P_V = vector_pressure(n_u_calc, n_d_calc, n_s_calc, params)
    P_B = bag_pressure(params)
    e_B = bag_energy(params)

    # eps_V = P_V for a vector field; the bag enters with opposite signs
    P_total = P_kin + P_V + P_B
    e_total = e_kin + P_V + e_B
    s_total = s_kin

    f_total = e_total - s_total * T

    n_B, n_C, n_S = quark_charges(n_u_calc, n_d_calc, n_s_calc)
    Y_C = n_C / n_B
    Y_S = n_S / n_B

    mu_B, mu_C, mu_S = charge_potentials_from_quarks(mu_u, mu_d, mu_s)
    if two_flavour:
        # No populated species carries strangeness, so S has no potential
        # conjugate to it. mu_s stays tied to mu_d by the solvers' own
        # strangeness-equilibrium row -- which is what keeps its slot
        # DETERMINED rather than a null Jacobian column -- and mu_S = mu_s -
        # mu_d is zero as a consequence rather than by assignment. It is set
        # here so that round-off in that row cannot leak into a reported
        # potential of a charge the state does not carry.
        mu_S = 0.0

    return MatterThermo(
        n_u=n_u_calc, n_d=n_d_calc, n_s=n_s_calc, n_B=n_B, n_C=n_C, n_S=n_S,
        Y_C=Y_C, Y_S=Y_S,
        T=T,
        mu_u=mu_u, mu_d=mu_d, mu_s=mu_s, mu_B=mu_B, mu_C=mu_C, mu_S=mu_S,
        P=P_total, e=e_total, s=s_total, f=f_total,
    )


def thermo_from_n(
    n_u: float, n_d: float, n_s: float, T: float,
    params: Parameters = None
) -> Tuple[float, float, float, float, float, float, float]:
    """Quark-matter thermodynamics at given flavour densities.

    Densities fix the mean field directly, so this needs no root find: invert
    the Fermi integrals for the effective potentials, add V back to get the
    physical ones, and assemble.

    Returns (mu_u, mu_d, mu_s, P, eps, s, n_B) — potentials in MeV, P and eps
    in MeV/fm^3, s and n_B in fm^-3.
    """
    if params is None:
        params = Parameters.default()

    m_u, m_d, m_s = params.m_u, params.m_d, params.m_s

    n_B = quark_charges(n_u, n_d, n_s)[0]

    # Invert the Fermi integrals for the effective potentials
    mu_eff_u = invert_fermi_density(n_u, T, m_u, G_QUARK)
    mu_eff_d = invert_fermi_density(n_d, T, m_d, G_QUARK)
    mu_eff_s = invert_fermi_density(n_s, T, m_s, G_QUARK)

    # Physical chemical potentials
    mu_u = physical_potentials(mu_eff_u, n_u, n_d, n_s, params)
    mu_d = physical_potentials(mu_eff_d, n_u, n_d, n_s, params)
    mu_s = physical_potentials(mu_eff_s, n_u, n_d, n_s, params)

    thermo_u = kinetic_thermo(mu_eff_u, T, m_u)
    thermo_d = kinetic_thermo(mu_eff_d, T, m_d)
    thermo_s = kinetic_thermo(mu_eff_s, T, m_s)

    P_kin = thermo_u.P + thermo_d.P + thermo_s.P
    e_kin = thermo_u.e + thermo_d.e + thermo_s.e
    s_kin = thermo_u.s + thermo_d.s + thermo_s.s

    P_V = vector_pressure(n_u, n_d, n_s, params)
    P_B = bag_pressure(params)
    e_B = bag_energy(params)

    P_quarks = P_kin + P_V + P_B
    e_quarks = e_kin + P_V + e_B      # eps_V = P_V
    s_quarks = s_kin

    return (mu_u, mu_d, mu_s, P_quarks, e_quarks, s_quarks, n_B)


def thermo_from_mu(
    mu_u: float, mu_d: float, mu_s: float, T: float,
    params: Parameters = None
) -> Tuple[float, float, float, float, float, float, float]:
    """Quark-matter thermodynamics at given physical chemical potentials.

    Potentials do NOT fix the mean field directly — mu_eff = mu - V(n) and
    n = n(mu_eff) — so this is a 3-dimensional root find on the effective
    potentials, with the free-gas limit V = 0 as the starting guess.

    Returns (n_u, n_d, n_s, P, eps, s, n_B).
    """
    from scipy.optimize import root

    if params is None:
        params = Parameters.default()

    m_u, m_d, m_s = params.m_u, params.m_d, params.m_s

    def equations(mu_eff_vec):
        mu_eff_u, mu_eff_d, mu_eff_s = mu_eff_vec
        n_u = kinetic_thermo(mu_eff_u, T, m_u).n
        n_d = kinetic_thermo(mu_eff_d, T, m_d).n
        n_s = kinetic_thermo(mu_eff_s, T, m_s).n
        V = vector_field(n_u, n_d, n_s, params)
        return [mu_eff_u + V - mu_u,
                mu_eff_d + V - mu_d,
                mu_eff_s + V - mu_s]

    # Initial guess: V = 0 (free gas limit)
    x0 = [mu_u, mu_d, mu_s]
    sol = root(equations, x0, method='hybr')

    mu_eff_u, mu_eff_d, mu_eff_s = sol.x

    thermo_u = kinetic_thermo(mu_eff_u, T, m_u)
    thermo_d = kinetic_thermo(mu_eff_d, T, m_d)
    thermo_s = kinetic_thermo(mu_eff_s, T, m_s)

    n_u, n_d, n_s = thermo_u.n, thermo_d.n, thermo_s.n
    n_B = quark_charges(n_u, n_d, n_s)[0]

    P_kin = thermo_u.P + thermo_d.P + thermo_s.P
    e_kin = thermo_u.e + thermo_d.e + thermo_s.e
    s_kin = thermo_u.s + thermo_d.s + thermo_s.s

    P_V = vector_pressure(n_u, n_d, n_s, params)
    P_B = bag_pressure(params)
    e_B = bag_energy(params)

    P_quarks = P_kin + P_V + P_B
    e_quarks = e_kin + P_V + e_B
    s_quarks = s_kin

    return (n_u, n_d, n_s, P_quarks, e_quarks, s_quarks, n_B)
