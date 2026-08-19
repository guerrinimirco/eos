"""The DID coupling functionals: how g_Mi depends on the state.

Pure mathematics -- the functional FORM of the couplings and nothing else.
The numbers that pin the form down are `parameters.py`, and the mean fields
they multiply are `thermodynamics.py` (CLAUDE.md section 5: a parameter takes
no arguments, a coupling is a function of the state).

What makes this model DID (density- and isospin-dependent) is that a coupling
depends on TWO state variables rather than one -- the baryon density n_B and
the isospin asymmetry

    beta = sum_i tau_3i n_i / n_B,        tau_3 = +/-1 for the nucleons,

so beta = 0 is isospin-symmetric matter (ISM) and beta = -1 is pure neutron
matter (NM). The coupling interpolates between one branch fitted in each
(Frohaug, Maslov, Dexheimer et al., arXiv:2511.15646, Eq. 4):

    g_Mi(n_B, beta) = [1 - w] g^S_Mi(n_B) + w g^N_Mi(n_B),
    w(x, beta)      = beta^2 tanh(x / e),      x = n_B/n_0,  e = 1/3,

and each branch carries the same shape in x (their Eq. 5),

    g^{S,N}_Mi(n_B) = g^{S,N(0)}_Mi F_M(x),
    F_M(x) = E_M(x) [1 - t_M(x)]/2 + b_M [1 + t_M(x)]/2,
    E_M(x) = exp[1 - ((x+1)/2)^{2 a_M}],   t_M(x) = tanh[(x - c_M)/d_M],

so that the shape F_M is a property of the MESON and the two numbers
g^{S(0)}_Mi, g^{N(0)}_Mi are properties of the meson-baryon vertex. That
factorisation is the paper's own ("this can be implemented by adjusting only
the saturation values") and is why one shape function serves every baryon.

Both derivatives are analytic and both are needed: the model carries TWO
rearrangement self-energies, Sigma^r from dg/dn_B and Sigma^t from dg/dbeta
(their Eqs. 10-11), and a finite difference of a coupling inside a Newton
residual is a good way to lose the thermodynamic consistency the rearrangement
terms exist to provide.

The tanh(x/e) factor in w is not decoration: it makes the couplings
isospin-INDEPENDENT at zero density, which is what keeps Sigma^t finite as
n_B -> 0 (Sigma^t carries a 1/n_B prefactor).
"""
import math

#: The isospin-blend scale e of Eq. (4), fixed at 1/3 by the paper.
E_ISOSPIN = 1.0 / 3.0

#: The ideal omega-phi mixing angle, tan(theta) = 1/sqrt(2): theta = 35.26 deg
#: makes the phi a pure s-sbar state. The vector mixing angle is fixed by the
#: meson masses to essentially this value, so the paper uses ideal mixing.
TAN_THETA_IDEAL = 1.0 / math.sqrt(2.0)

#: F/(D+F) at the three-octet-hadron vertex. The paper fixes alpha = 1 and
#: varies only z, so this is the default everywhere.
ALPHA_IDEAL = 1.0

#: z = g_1/g_8 at SU(6) symmetry, where the ratios below collapse to
#: g_omegaLambda/g_omegaN = 2/3, g_omegaXi/g_omegaN = 1/3, g_phiN = 0.
Z_SU6 = 1.0 / math.sqrt(6.0)


# =============================================================================
# THE DENSITY SHAPE  F_M(x)  AND ITS DERIVATIVE
# =============================================================================

def shape(x, a, b, c, d):
    """F_M(x) of Eq. (5): the density dependence shared by both branches.

    A low-density exponential decay switched over, around x = c and with
    width d, to a constant plateau b -- the "detachment of high-density from
    low-density behaviour" the model is built on.

    c = infinity is a supported and USED value: the paper sets c_sigma = inf so
    the sigma coupling never flattens (flattening it drives c_s^2 negative),
    which makes d_sigma and b_sigma irrelevant. It is taken as a branch here
    rather than fed to tanh, where inf - inf would be nan.
    """
    exp_branch = math.exp(1.0 - ((x + 1.0) / 2.0) ** (2.0 * a))
    if math.isinf(c):
        return exp_branch
    t = math.tanh((x - c) / d)
    return exp_branch * (1.0 - t) / 2.0 + b * (1.0 + t) / 2.0


def dshape_dx(x, a, b, c, d):
    """dF_M/dx, in closed form.

    With u = (x+1)/2, E = exp[1 - u^(2a)] and t = tanh[(x-c)/d]:

        dE/dx = -a u^(2a-1) E,      dt/dx = (1 - t^2)/d,
        dF/dx = (dE/dx)(1-t)/2 + (b - E)(dt/dx)/2.
    """
    u = (x + 1.0) / 2.0
    exp_branch = math.exp(1.0 - u ** (2.0 * a))
    dexp = -a * u ** (2.0 * a - 1.0) * exp_branch
    if math.isinf(c):
        return dexp
    t = math.tanh((x - c) / d)
    dt = (1.0 - t * t) / d
    return dexp * (1.0 - t) / 2.0 + (b - exp_branch) * dt / 2.0


# =============================================================================
# THE ISOSPIN BLEND  w(x, beta)
# =============================================================================

def blend(x, beta, e=E_ISOSPIN):
    """w = beta^2 tanh(x/e): the weight of the neutron-matter branch.

    w = 0 in ISM and w -> 1 in NM at any density above a fraction of n_0.
    |beta| > 1 is reachable in Sigma-rich matter and simply extrapolates.
    """
    return beta * beta * math.tanh(x / e)


def dblend_dx(x, beta, e=E_ISOSPIN):
    """dw/dx."""
    t = math.tanh(x / e)
    return beta * beta * (1.0 - t * t) / e


def dblend_dbeta(x, beta, e=E_ISOSPIN):
    """dw/dbeta."""
    return 2.0 * beta * math.tanh(x / e)


def coupling(g_S, g_N, x, beta, a, b, c, d, e=E_ISOSPIN):
    """g_Mi(n_B, beta) and its derivatives (g, dg/dx, dg/dbeta).

    dg/dx rather than dg/dn_B: the caller divides by n_0 once, where it also
    knows the units it wants (see `parameters.Parameters.couplings_at`).

        g       = [(1-w) g_S + w g_N] F
        dg/dx   = [(1-w) g_S + w g_N] F' + (g_N - g_S) (dw/dx) F
        dg/dbeta= (g_N - g_S) (dw/dbeta) F
    """
    F = shape(x, a, b, c, d)
    dF = dshape_dx(x, a, b, c, d)
    w = blend(x, beta, e)
    mixed = (1.0 - w) * g_S + w * g_N
    branch_gap = g_N - g_S
    return (mixed * F,
            mixed * dF + branch_gap * dblend_dx(x, beta, e) * F,
            branch_gap * dblend_dbeta(x, beta, e) * F)


# =============================================================================
# SU(3)_f FOR THE ISOSCALAR VECTORS
# =============================================================================
# The omega and phi couplings of the whole octet follow from three numbers --
# the mixing angle theta, alpha = F/(D+F), and z = g_1/g_8 -- times the octet
# coupling g_8 (paper Eq. 6). The scalar and isovector sectors are NOT related
# by SU(3) in this model: g_sigma i and g_rho i are free fit parameters.

def su3_vector_ratios(z, alpha=ALPHA_IDEAL, tan_theta=TAN_THETA_IDEAL):
    """{multiplet: (g_omega/g_8, g_phi/g_8)} from (z, alpha, theta).

    The paper's Eq. (6), with one correction. Its g_phiXi line prints
    -(2z/sqrt3)(1+2 alpha) where every other line pairs the phi coefficient
    with the coefficient multiplying tan(theta) in the same baryon's omega
    line -- here (z/sqrt3)(1+2 alpha). The paired form is the one used, on the
    evidence of the SU(6) limit: at z = 1/sqrt6, alpha = 1, ideal mixing it
    gives EXACTLY the textbook ratios

        g_omegaLambda = g_omegaSigma = (2/3) g_omegaN,  g_omegaXi = (1/3) g_omegaN,
        g_phiN = 0,  g_phiLambda = g_phiSigma = -(sqrt2/3) g_omegaN,
        g_phiXi = -(2 sqrt2/3) g_omegaN,

    all four of which the printed form breaks (it gives g_phiXi = -sqrt2 g_omegaN).
    `verify/run_full_check.py` runs that limit as a standing check.
    """
    r3 = math.sqrt(3.0)
    #: The coefficient c_i for which g_omega,i = g_8 (1 - c_i tan theta) and
    #: g_phi,i = g_8 (-tan theta - c_i).
    c = {
        "N": (z / r3) * (1.0 - 4.0 * alpha),
        "Lambda": (2.0 * z / r3) * (1.0 - alpha),
        "Sigma": -(2.0 * z / r3) * (1.0 - alpha),
        "Xi": (z / r3) * (1.0 + 2.0 * alpha),
    }
    return {name: (1.0 - c_i * tan_theta, -tan_theta - c_i)
            for name, c_i in c.items()}


def g8_from_aggregate(g_tilde_omega_N, z, m_omega, m_phi,
                      alpha=ALPHA_IDEAL, tan_theta=TAN_THETA_IDEAL):
    """The octet coupling g_8 behind an aggregated omega-phi strength.

    The Bayesian analysis varies the single combination (paper Eq. 52)

        g~_omegaN = g_omegaN sqrt(1 + [(g_phiN/m_phi)/(g_omegaN/m_omega)]^2),

    which is what nucleonic matter actually feels, since both vectors enter the
    energy as g^2/m^2 times the density. Writing g_omegaN = g_8 A_omega and
    g_phiN = g_8 A_phi from `su3_vector_ratios` inverts it in closed form:

        g_8 = g~_omegaN / sqrt(A_omega^2 + (m_omega/m_phi)^2 A_phi^2).

    Unpacking it here is what lets the two fields be carried separately, which
    they must be: they have different masses, and with hyperons they have
    different sources.
    """
    A_omega, A_phi = su3_vector_ratios(z, alpha, tan_theta)["N"]
    scale = math.sqrt(A_omega ** 2 + (m_omega / m_phi) ** 2 * A_phi ** 2)
    return g_tilde_omega_N / scale
