"""The vector coupling as a FUNCTION of the state, and the rearrangement
self-energy it owes.

A parameter takes no arguments; a coupling is a function of the state
(CLAUDE.md section 5). `gbar_omega` and `n_c` are parameters and live in
`parameters.py`; g_omega(n_B) is evaluated at every density and never stored,
so its functional form lives here.

    g_omega(n_B) = gbar_omega / [1 + (n_B/n_c)^k] ,    k = par.k_omega

The form is a repulsion that dies off at high density, which is what keeps the
vector energy from growing without bound: that energy goes as g^2 n_q^2, so a
coupling falling fast enough turns it into a term that stops growing at all.

WHAT THAT COSTS, AND WHERE. Saturating the vector energy is the same statement
as making the vector sector's PRESSURE turn over, because the rearrangement
below is mandatory. Collecting the two vector terms of P (the field energy and
Sigma_R n_q, derived below),

    P_vec = (n_q^2/m_omega^2) g [ g/2 + n_B dg/dn_B ]
          = (n_q^2/m_omega^2) g^2 [ 1/2 - k u^k/(1 + u^k) ] ,   u = n_B/n_c

so the bracket -- and with it the vector contribution to the pressure -- turns
NEGATIVE once the coupling's logarithmic slope passes -1/2:

    dln g/dln n_B < -1/2      <=>      P_vec < 0

For the shipped k = 2 that is u > 1/sqrt(3) = 0.577, and P_vec is already
FALLING from u = 0.363 (the roots of 3u^4 - 8u^2 + 1 = 0 are 0.363 and 1.592,
its maximum and its minimum). A parameter point whose density axis reaches past
n_B = 0.363 n_c is therefore one where the vector sector removes pressure as
the density rises, and if it removes it faster than the kinetic term adds it
the total P is non-monotonic -- a mechanically unstable branch, not a soft one.
At gbar_omega = 4 and n_c = 3 fm^-3 that is exactly what happens: P falls from
the onset.

THE TRADE IS STRUCTURAL, not a bad choice of numbers. k <= 1/2 keeps the
logarithmic slope above -1/2 at every density and P_vec positive everywhere --
but a coupling that gentle does not saturate the vector energy either, which
was the point of making it density dependent. `n_c = inf` is the other end: no
density dependence, no rearrangement, and a vector energy that grows forever
(the `constvector` set). What a parameter point actually chooses is WHERE the
turnover sits, and n_c is the lever for it: keep 0.363 n_c above the top of the
density axis and the vector sector is monotonic over it.

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

Units: n_B and n_q in MeV^3, n_c in fm^-3 (converted here), k_omega and
g_omega dimensionless, omega_0 in MeV, Sigma_R in MeV.
"""
from eos.general.physics_constants import hc3


def _decay(u, k):
    """u^k, with the shipped k = 2 evaluated as u*u.

    Not an optimisation and not a style choice. For PYTHON FLOATS `u ** 2.0`
    goes through libm's pow and `u * u` is a single multiply, and the two
    differ in the last place; half an ulp in g_omega walks the
    colour-superconducting solve at n_B = 1.5 fm^-3 to a different last digit
    and moves a frozen regression baseline. Making the default path
    bit-for-bit what it was before the exponent became a parameter is worth
    one branch. (NumPy special-cases the square and shows no such difference,
    which is exactly how this is easy to miss.)
    """
    return u * u if k == 2.0 else u ** k


def vector_coupling(par, n_B):
    """g_omega(n_B), dimensionless.

        g_omega = gbar_omega / [1 + (n_B/n_c)^k] ,   k = par.k_omega

    with n_B in MeV^3 and the parameter n_c in fm^-3, converted here so that
    the caller never has to remember which side of the boundary it is on.
    n_c = inf returns gbar_omega exactly, for every k.
    """
    if par.gbar_omega == 0.0:
        return 0.0
    u = n_B / (par.n_c * hc3)
    return par.gbar_omega / (1.0 + _decay(u, par.k_omega))


def vector_coupling_derivative(par, n_B):
    """dg_omega/dn_B [MeV^-3].

        dg/dn_B = -k gbar_omega (n_B^(k-1)/n_c^k) / [1 + u^k]^2 ,  u = n_B/n_c

    Negative everywhere: the repulsion weakens as the medium fills up, which
    is also why the innermost fixed point on omega_0 converges monotonically.
    Zero at u = 0, which is the value the limit takes for k > 1 and the value
    Sigma_R = (dg/dn_B) omega_0 n_B takes for every k > 0 -- so returning it
    there is what keeps k < 1 (where dg/dn_B alone diverges) finite in the one
    quantity that is used.

    GROUPED AS (n_B^(k-1)/n_c^k) rather than the tidier (k/n_c) u^(k-1), which
    is the same number in exact arithmetic and NOT the same double: at k = 2
    this grouping reproduces the fixed-exponent expression it replaces bit for
    bit, and the other differs in the last place. Half an ulp in Sigma_R walks
    a pairing solve to a different last digit and moves the frozen regression
    baselines, so the grouping is load-bearing.
    """
    if par.gbar_omega == 0.0:
        return 0.0
    n_c = par.n_c * hc3
    u = n_B / n_c
    if u == 0.0:
        return 0.0
    k = par.k_omega
    return -k * par.gbar_omega * (n_B ** (k - 1.0) / n_c ** k) / (
        1.0 + _decay(u, k)) ** 2


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
