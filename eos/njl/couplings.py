"""The vector coupling as a FUNCTION of the state, and the self-energy it owes.

A parameter takes no arguments; a coupling is a function of the state
(CLAUDE.md section 5). G_V is a parameter in the constant variant and lives in
`parameters.py`; in the two density-dependent variants it is evaluated at
every density and never stored, so its functional form lives here.

Why the model has these at all
------------------------------
With chiral symmetry restored the scalar channel dies (M -> m -> 0) and the
high-density behaviour is set entirely by the vector term. At CONSTANT G_V the
self-consistency mu = mu' + (2 G_V N_c N_f/3 pi^2) mu'^3 is cubic, so the
shifted potential grows only as mu^(1/3), the interaction energy W = G_V n^2
grows like n^2 against the kinetic n^(4/3), and the sound speed runs away to 1
-- Zel'dovich behaviour, and a violation of the conformal bound that has
nothing to do with the physics one wants from the vector channel.

Writing eps = sum_i C_i n^(p_i), each term contributes P_i = C_i (p_i - 1)
n^(p_i), so

    c_s^2(n -> infinity) = max(1 - alpha, 1/3)      for G_V ~ n^(-alpha)

because the vector term then has p_V = 2 - alpha against the free-quark 4/3.
alpha = 2/3 is the marginal exponent, and it is marginal EXACTLY rather than
asymptotically: there the interaction pressure is one third of the interaction
energy density at every density, so the vector term is conformal on its own.

THE REARRANGEMENT TERM IS MANDATORY. Once G_V depends on the density, the
vector self-energy is the derivative of the energy density,

    W(n) = G_V(n) n^2 ,   Sigma_V = dW/dn = (2 - alpha) G_V(n) n

and not 2 G_V n. Omitting it is a 5% error in P and breaks n = dP/dmu at the
first digit. (A coupling that depends on a mean FIELD needs no such term,
because it enters through that field's own equation of motion. A
density-dependent one does.)

Section 9 of docs/njl_csc_implementation.md is the reference for all of the
above; `njl.tex` writes the forms out.

Units: n_q in MeV^3, G_V in MeV^-2, W in MeV^4, Sigma_V in MeV.
"""
import math

from eos.general.physics_constants import hc3

_PI2 = math.pi ** 2


def vector_coupling(par, n_q):
    """G_V(n_q) [MeV^-2] for the parameter set's declared form.

    constant        G_V = eta_V G_S.
    power_law       G_V = G_V0 (n_ref/n_q)^alpha, with G_V0 = eta_V G_S the
                    coupling AT n_ref, so the constant variant is the
                    alpha = 0 member of the same family.
    gluon_exchange  G_V = G_V0/[1 + 8 k_F^2/(9 M_g^2)], k_F = (pi^2 n_q/2)^(1/3),
                    with a non-perturbative gluon mass M_g. This is Eq. (23) of
                    Song, Baym, Hatsuda and Kojo, Phys. Rev. D 100, 034018
                    (2019) [arXiv:1905.01005], written G_V = (4 pi alpha_s/3) /
                    (9 M_g^2 + 8 k_F^2) there, so G_V0 = 4 pi alpha_s/(27 M_g^2)
                    is their k_F -> 0 limit. They obtain it by matching the NJL
                    vector energy G_V n_q^2 to the single-gluon-exchange (Fock)
                    energy of QCD in Landau gauge with a massive gluon
                    propagator D(p) = 1/(p^2 - M_g^2) and a frozen alpha_s; it
                    is a LOCAL NJL result, not a nonlocal one, and their gluon
                    mass estimate is 500 +/- 200 MeV, which is where this
                    model's default M_g = 500 MeV comes from. Their vacuum
                    Fierz ratio G_V0/G_S = 1/2 is the default G_V0_over_GS.

                    Its effective exponent -d ln G_V/d ln n_q runs 0.062, 0.460,
                    0.608, 0.653 at n_q = 1e6, 1e8, 1e9, 1e10 MeV^3. It arrives
                    at the conformal 2/3 because that exponent is the pQCD limit
                    built into the interpolation and not an emergent accident:
                    the other limit of the same expression is
                    G_V(k_F >> M_g) -> pi alpha_s/(6 k_F^2) (their Eq. 22),
                    which is dimensional analysis -- G_V has dimension mass^-2
                    and at asymptotic density k_F is the only scale -- so
                    G_V ~ k_F^-2 ~ n_q^(-2/3) exactly. The gluon mass regulates
                    the other end: G_V(0) is finite, which is why this form needs
                    no guard at n_q = 0 and `power_law` does. Both properties are
                    why it is the recommended choice rather than a fitted
                    interpolation.
    """
    form = par.vector_form
    if form == "constant":
        return par.eta_V * par.G_S
    if form == "power_law":
        if n_q <= 0.0:
            return 0.0
        n_ref = par.n_ref * hc3
        return par.eta_V * par.G_S * (n_ref / n_q) ** par.alpha
    if form == "gluon_exchange":
        return par.G_V0_over_GS * par.G_S / (1.0 + _gluon_ratio(par, n_q))
    raise ValueError(f"unknown vector_form {form!r}; eos.njl implements "
                     f"'constant', 'power_law', 'gluon_exchange'")


def _gluon_ratio(par, n_q):
    """8 k_F^2/(9 M_g^2), the dimensionless argument of the gluon form."""
    if n_q <= 0.0:
        return 0.0
    kF2 = (0.5 * _PI2 * n_q) ** (2.0 / 3.0)
    return 8.0 * kF2 / (9.0 * par.M_g ** 2)


def effective_exponent(par, n_q):
    """alpha_eff = -d ln G_V / d ln n_q, dimensionless.

    The number that decides the asymptotic sound speed: c_s^2 -> max(1 -
    alpha_eff, 1/3). Constant coupling gives 0 (and c_s^2 -> 1); the
    power-law form gives alpha by construction; the gluon-exchange form gives
    (2/3) u/(1 + u) with u = 8 k_F^2/(9 M_g^2), which tends to 2/3.
    """
    form = par.vector_form
    if form == "constant":
        return 0.0
    if form == "power_law":
        return par.alpha
    if form == "gluon_exchange":
        u = _gluon_ratio(par, n_q)
        return (2.0 / 3.0) * u / (1.0 + u)
    raise ValueError(f"unknown vector_form {form!r}")


def vector_energy(par, n_q):
    """W(n_q) = G_V(n_q) n_q^2 [MeV^4], the vector interaction energy density.

    It enters eps with a plus sign and Omega through the combination
    -(Sigma_V n_q - W), which for constant G_V is the familiar -G_V n_q^2.
    """
    return vector_coupling(par, n_q) * n_q ** 2


def vector_self_energy(par, n_q):
    """Sigma_V = dW/dn_q [MeV], the shift mu*_j = mu_j - Sigma_V.

    Equal to (2 - alpha_eff) G_V n_q, so it is the naive 2 G_V n_q only at
    alpha_eff = 0. At alpha = 1/3 the ratio is 0.833 and at 2/3 it is 0.667:
    using the naive form instead shifts P by about 5%, and breaks
    n = dP/dmu, which holds to 1e-8 with the correct one.
    """
    return (2.0 - effective_exponent(par, n_q)) * vector_coupling(par, n_q) * n_q


def vector_self_energy_derivative(par, n_q):
    """d Sigma_V / d n_q = d^2 W / d n_q^2 [MeV^-2], for the Jacobian.

    constant        2 G_V
    power_law       (2 - alpha)(1 - alpha) G_V(n_q)
    gluon_exchange  G_V0 [ 2/D - (22/9) u/D^2 + (8/9) u^2/D^3 ],  D = 1 + u

    the last from W = G_V0 n^2/(1 + u) with u = 8 k_F^2/(9 M_g^2) ~ n^(2/3),
    so u' = (2/3) u/n and u'' = -(2/9) u/n^2.
    """
    form = par.vector_form
    if form == "constant":
        return 2.0 * par.eta_V * par.G_S
    if form == "power_law":
        return (2.0 - par.alpha) * (1.0 - par.alpha) * vector_coupling(par, n_q)
    if form == "gluon_exchange":
        u = _gluon_ratio(par, n_q)
        D = 1.0 + u
        return par.G_V0_over_GS * par.G_S * (
            2.0 / D - (22.0 / 9.0) * u / D ** 2 + (8.0 / 9.0) * u ** 2 / D ** 3)
    raise ValueError(f"unknown vector_form {form!r}")
