"""The vector coupling as a FUNCTION of the state, and the rearrangement
self-energy it owes.

A parameter takes no arguments; a coupling is a function of the state
(CLAUDE.md section 5). `gbar_omega` and `n_c` are parameters and live in
`parameters.py`; g_omega(n_B) is evaluated at every density and never stored,
so its functional form lives here.

    g_omega(n_B) = gbar_omega / [1 + (n_B/n_c)^2]

The form is a repulsion that dies off at high density, which is what keeps the
sound speed away from the causal limit without a hand-placed ceiling: the
vector energy grows as g^2 n_q^2, so a coupling falling as n^-2 turns it into
a term that stops growing at all.

THE REARRANGEMENT TERM IS MANDATORY. Once g_omega depends on the density, the
vector self-energy is the derivative of the interaction energy with respect to
the quark density, not just g_omega omega_0:

    W(n_B)   = (1/2) m_omega^2 omega_0^2 ,   omega_0 = g_omega(n_B) n_q/m_omega^2
    dW/dn_q  = g_omega omega_0 + Sigma_R ,   Sigma_R = (dg_omega/dn_B) omega_0 n_B

(the chain rule through n_B = n_q/3 is what turns the naive dg/dn_q n_q into
the n_B written above). Omitting Sigma_R breaks n = -dOmega/dmu at the first
digit and shifts P by percents.

SIGMA_R ENTERS mu AND P, NEVER eps -- CLAUDE.md section 8, and the same
statement as in every density-dependent relativistic mean-field model in this
repository. The specification's section 4.1 omits the compensating
-Sigma_R n_q from Omega while carrying Sigma_R in mu*; its own Euler audit
(section 9.6) is what catches that, and `eos.ccdm.thermodynamics` carries the
term. Section 2 of `ccdm.tex` writes the corrected assembly out.

Units: n_B and n_q in MeV^3, n_c in fm^-3 (converted here), g_omega
dimensionless, omega_0 in MeV, Sigma_R in MeV.
"""
from eos.general.physics_constants import hc3


def vector_coupling(par, n_B):
    """g_omega(n_B), dimensionless.

        g_omega = gbar_omega / [1 + (n_B/n_c)^2]

    with n_B in MeV^3 and the parameter n_c in fm^-3, converted here so that
    the caller never has to remember which side of the boundary it is on.
    """
    if par.gbar_omega == 0.0:
        return 0.0
    u = n_B / (par.n_c * hc3)
    return par.gbar_omega / (1.0 + u * u)


def vector_coupling_derivative(par, n_B):
    """dg_omega/dn_B [MeV^-3].

        dg/dn_B = -2 gbar_omega (n_B/n_c^2) / [1 + (n_B/n_c)^2]^2

    Negative everywhere: the repulsion weakens as the medium fills up, which
    is also why the innermost fixed point on omega_0 converges monotonically.
    """
    if par.gbar_omega == 0.0:
        return 0.0
    n_c = par.n_c * hc3
    u = n_B / n_c
    return -2.0 * par.gbar_omega * (n_B / n_c ** 2) / (1.0 + u * u) ** 2


def vector_field(par, n_B):
    """omega_0 = g_omega(n_B) n_q / m_omega^2 [MeV], at n_q = 3 n_B.

    THE SOURCE IS THE QUARK NUMBER DENSITY n_q = 3 n_B, not n_B: the coupling
    in the Lagrangian is to qbar gamma^mu q. Using n_B understates omega_0 by
    a factor 3 and the repulsive energy by 9.

    This is the right-hand side of the R_4 row rather than a substitute for
    it: the field is an unknown of the same Newton solve as the scalars,
    because n_B depends on omega_0 through the shifted potentials.
    """
    return vector_coupling(par, n_B) * 3.0 * n_B / par.m_omega ** 2


def rearrangement(par, n_B, omega_0):
    """Sigma_R = (dg_omega/dn_B) omega_0 n_B [MeV].

    The shift that makes mu*_(f,a) = mu_(f,a) - g_omega omega_0 - Sigma_R the
    potential the momentum integrals see. Identically zero for a constant
    coupling, and identically zero when gbar_omega is.
    """
    return vector_coupling_derivative(par, n_B) * omega_0 * n_B


def diquark_coupling(par, chi):
    """G_D(chi) [MeV^-2]: the diquark coupling, dielectric-dressed at q = 1.

        G_D -> G_D / chi^q ,    q in {0, 1}

    q = 0 leaves it bare. q = 1 is the exponent a gluon-exchange origin gives
    and the largest that leaves the pairing channel confined, the criterion
    being q <= p because the critical coupling for vacuum diquark condensation
    grows only linearly in M* ~ chi^-p. The de Carvalho contact coupling
    carries q = 4 at p = 1, which violates it and is therefore NOT used here
    (section 10 of docs/ccdm_implementation.md); what is taken from that work
    is the argument that the pairing term is a legitimate leading term at all.
    """
    if par.q == 0:
        return par.G_D
    return par.G_D / chi ** par.q


def vector_self_energy(par, n_B):
    """Sigma_V = g_omega omega_0 + Sigma_R [MeV], the total shift of mu.

        mu*_(f,a) = mu_(f,a) - Sigma_V

    Written as one quantity because that is what the momentum integrals see
    and what the solver carries as an unknown (CLAUDE.md section 2: solver
    unknown vectors use the EFFECTIVE potentials). Carrying Sigma_V rather
    than omega_0 is also what removes the circularity -- omega_0 depends on
    n_B, which depends on omega_0 through mu* -- since at a given Sigma_V the
    densities follow explicitly, omega_0 follows from them explicitly, and the
    residual Sigma_V - vector_self_energy(par, n_B) is exactly the statement
    that the returned field is the one the returned densities source.

    Equal to dW/dn_q with W = (1/2) m_omega^2 omega_0^2, so it is the
    derivative of the interaction energy density with respect to the quark
    density, which is what makes n = -dOmega/dmu hold.
    """
    omega_0 = vector_field(par, n_B)
    return vector_coupling(par, n_B) * omega_0 + rearrangement(par, n_B, omega_0)


def has_vector(par):
    """Is there a vector coupling at all? A zero one is not carried.

    Public because `solver.py` asks the same question when it lays out the
    mode's unknown vector: whether Sigma_V is an unknown is one fact, and one
    fact has one home.
    """
    return par.gbar_omega != 0.0
