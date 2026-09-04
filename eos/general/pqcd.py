"""
general/pqcd.py
===============
The perturbative-QCD pressure of cold quark matter, and the band that reports
its truncation error.

At asymptotically high density QCD is weakly coupled and the pressure of
beta-equilibrated, electrically neutral quark matter is known as a series in
alpha_s. It is the one first-principles statement about the same quantity the
models in this repository parametrize, so it belongs beside the observational
constraints rather than inside any one model: nothing here is fitted and
nothing here is a free parameter of a model.

WHAT IS IMPLEMENTED
-------------------
Three massless flavours in beta equilibrium, so mu_u = mu_d = mu_s = mu_B/3
and the free gas is

    p_free(mu_B) = mu_B^4 / (108 pi^2)                                     (1)

The unpaired pressure is carried to N3LO [1] as

    p_N3LO / p_free = 1 - 2a - 3 a^2 [ L + 3 lnX + 5.0021 ]
                        + 9 a^3 [ (11/12) L^2 + (-6.5968 - 3 lnX) L
                                  + 5.1342 + (2/3) c0
                                  - 18.284 lnX - 4.5 ln^2 X ]              (2)

with a = alpha_s/pi and L = ln(3 alpha_s/pi). A colour-flavour-locked
condensate adds its condensation energy, known to NLO [2],

    p_CFL = p_free gamma_1 Delta_bar^2 ,
    gamma_1 = 4 + 40.9 alpha_s ,      Delta_bar = Delta / (mu_B/3)         (3)

so `pressure(..., Delta=0.0)` is the unpaired result and a non-zero `Delta`
is the CFL one -- at this order the gap enters as one additive term and no
gap equation is solved: Delta is GIVEN, exactly as it is to the CFL modes of
`eos.abpr` and `eos.alphabag`.

The strange-quark mass is dropped. It enters (2)-(3) as -(4/3) m_s^2/mu_q^2 in
gamma_1 and as -m_s^4/(4 mu_q^4) in the pressure; with m_s(2 GeV) = 93.8 MeV
both are below a percent of the terms kept anywhere this expression is quoted,
and carrying them would mean running m_s as well as alpha_s.

THE SCALE, AND WHY THE RESULT IS A BAND
---------------------------------------
The renormalization scale enters through

    X = 3 Lambda_bar / (2 mu_B) ,   i.e.  Lambda_bar = 2 X mu_B / 3        (4)

A summed series would not depend on it, so the spread of (2) over
X in [1/2, 2] IS the reported size of the missing orders: `band()` sweeps X
and returns the envelope. This is the same convention, written in different
units, as the older X = 3 Lambda_bar/mu_B in [1, 4].

`c0` is the IR-finite remainder of the unresummed four-loop diagrams and is
still unknown. It is NOT a detail that can be set to zero: 9 a^3 (2/3) c0 is
the term that decides the low-X edge, and at c0 = 0 the band reaches
p/p_free = 1.7 at mu_B = 2.6 GeV, which is the series failing rather than a
prediction. The default here is the posterior median of [3], which is also
what that paper's own comparison figures use.

WHERE IT MAY BE EVALUATED
-------------------------
`MU_B_MIN` is the scale below which the papers above stop quoting the result:
at mu_B = 2.6 GeV the X = 1/2 edge already sits at alpha_s = 0.64. `band()`
refuses below it rather than returning numbers that look like a prediction;
`pressure()` does not, so a caller who wants to watch the series break down
can. Below Lambda_bar = Lambda_MSbar the coupling has no real value at all and
`alpha_s` raises.

REFERENCES
----------
[1] T. Gorda, A. Kurkela, R. Paatelainen, S. Saeppi and A. Vuorinen,
    "Soft Interactions in Cold Quark Matter", PRL 127, 162003 (2021).
[2] A. Geissel, T. Gorda and J. Braun, "Color Superconductivity under
    Neutron-Star Conditions at Next-to-Leading Order", PRL (2025);
    "Pressure and speed of sound in two-flavor color-superconducting quark
    matter at next-to-leading order", PRD 110, 014034 (2024).
[3] S.-P. Tang, Y.-J. Huang and Y.-Z. Fan, "Neutron Star Observations
    Challenge a Large Colour-Superconducting Gap in Dense Quark Matter",
    arXiv:2606.03707. Eqs. (11)-(13) there are (2)-(3) above.
"""
import numpy as np

from eos.general.physics_constants import hc3

#: MeV. Lambda_MSbar at N_f = 3. Two-loop running from it gives
#: alpha_s(2 GeV) = 0.2994, the value [1]-[3] start their four-loop running
#: from, so the two agree to better than 1e-4 at the reference scale.
LAMBDA_MS = 378.0

#: The unknown four-loop constant of Eq. (2), at the posterior median of [3].
C0_DEFAULT = -21.2

#: The renormalization-scale sweep that makes the band, in the X of Eq. (4).
X_RANGE = (0.5, 2.0)

#: MeV. The lowest mu_B at which the series is quoted; see the module docstring.
MU_B_MIN = 2600.0


def alpha_s(mu_bar):
    """Two-loop MSbar running coupling at N_f = 3.

    `mu_bar` is the renormalization scale in MeV -- Lambda_bar of Eq. (4), not
    a chemical potential. Raises below Lambda_MSbar, where the two-loop form
    has no real value.
    """
    mu_bar = np.asarray(mu_bar, dtype=float)
    if np.any(mu_bar <= LAMBDA_MS):
        raise ValueError(
            f"alpha_s is not defined at or below Lambda_MSbar = {LAMBDA_MS} "
            f"MeV; asked for {np.min(mu_bar)} MeV")

    b0 = (33.0 - 2.0 * 3.0) / (12.0 * np.pi)
    b1 = (153.0 - 19.0 * 3.0) / (24.0 * np.pi ** 2)
    t = np.log(mu_bar ** 2 / LAMBDA_MS ** 2)
    return (1.0 / (b0 * t)) * (1.0 - (b1 / b0 ** 2) * np.log(t) / t)


def free_pressure(mu_B):
    """Eq. (1): the free massless three-flavour gas at mu_u = mu_d = mu_s.

    `mu_B` in MeV, out in MeV/fm^3.
    """
    mu_B = np.asarray(mu_B, dtype=float)
    return mu_B ** 4 / (108.0 * np.pi ** 2) / hc3


def pressure(mu_B, X=1.0, Delta=0.0, c0=C0_DEFAULT):
    """Eqs. (2)-(3): the N3LO pressure, plus the NLO gap term if `Delta` > 0.

    Parameters
    ----------
    mu_B : float or array
        Baryon chemical potential in MeV.
    X : float
        Renormalization scale in units of the natural one, Eq. (4). X = 1 is
        Lambda_bar = 2 mu_B/3; `X_RANGE` is the sweep that makes the band.
    Delta : float
        The CFL gap in MeV, GIVEN rather than solved. Zero is unpaired.
    c0 : float
        The unknown four-loop constant; see the module docstring before
        changing it.

    Returns
    -------
    float or array, MeV/fm^3.
    """
    mu_B = np.asarray(mu_B, dtype=float)
    coupling = alpha_s(2.0 * X * mu_B / 3.0)
    a = coupling / np.pi
    L = np.log(3.0 * a)
    lnX = np.log(X)
    p_free = free_pressure(mu_B)

    unpaired = p_free * (
        1.0
        - 2.0 * a
        - 3.0 * a ** 2 * (L + 3.0 * lnX + 5.0021)
        + 9.0 * a ** 3 * (11.0 / 12.0 * L ** 2
                          + (-6.5968 - 3.0 * lnX) * L
                          + (5.1342 + 2.0 / 3.0 * c0
                             - 18.284 * lnX - 4.5 * lnX ** 2)))

    if Delta == 0.0:
        return unpaired

    gamma_1 = 4.0 + 40.9 * coupling
    return unpaired + p_free * gamma_1 * (3.0 * Delta / mu_B) ** 2


def band(mu_B, Delta=0.0, c0=C0_DEFAULT, x_range=X_RANGE, n_x=25):
    """The envelope of `pressure` over the scale sweep: (lower, upper).

    Two arrays shaped like `mu_B`, in MeV/fm^3. The width is the reported size
    of the missing orders, not a statistical uncertainty.

    Raises below `MU_B_MIN`, where the series is not quoted: a band drawn there
    is a picture of the expansion failing, and returning it silently would let
    it be read as a prediction. A caller who wants exactly that picture asks
    `pressure` for it, one X at a time.
    """
    mu_B = np.asarray(mu_B, dtype=float)
    if np.any(mu_B < MU_B_MIN):
        raise ValueError(
            f"the pQCD series is not quoted below mu_B = {MU_B_MIN} MeV "
            f"(at X = {x_range[0]} that is already alpha_s = "
            f"{float(alpha_s(2.0 * x_range[0] * MU_B_MIN / 3.0)):.2f}); "
            f"asked for {np.min(mu_B)} MeV")

    scanned = np.array([pressure(mu_B, X, Delta, c0)
                        for X in np.linspace(x_range[0], x_range[1], n_x)])
    return scanned.min(axis=0), scanned.max(axis=0)
