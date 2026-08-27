"""Thermodynamics of alphaBag quark matter, at given chemical potentials.

Two phases live here, and they are two potentials rather than one potential
and a correction:

    UNPAIRED   a gas of u, d, s quarks and their antiquarks inside a bag,
               with the leading perturbative QCD correction carried as one
               constant coupling alpha_s multiplying the free-gas terms
               (T. Fischer et al., Astrophys. J. Suppl. Ser. 194, 39 (2011)).
    CFL        the colour-flavour locked condensate, the same quark gas plus
               the Delta^2 mu^2 pairing term of M. Alford, M. Braby, M. Paris
               and S. Reddy, Astrophys. J. 629, 969 (2005).

Nothing here knows which equilibrium mode it is in: this module takes chemical
potentials and returns densities, P, eps and s; `solver.py` finds the
potentials a mode's conditions ask for.

There is no field to solve. The potential is explicit in mu -- no vector
field, no gap equation, the quark masses are parameters -- which is what makes
this model cheap enough to scan over.

Reading order: the massless closed forms, one flavour as an ideal gas, the
bag, the gluons, the unpaired sums; then the CFL gap, its corrections and the
paired sums.

Units: potentials and masses in MeV, densities in fm^-3, P and eps in
MeV/fm^3, s in fm^-3. See `alphabag.tex` for the equations.
"""
import numpy as np
from dataclasses import dataclass
from typing import Tuple

from eos.general.physics_constants import hc3, PI, PI2
from eos.general.fermi_integrals import solve_fermi_jel
from eos.general import particles
from eos.general.basis import quark_charges, charge_potentials_from_quarks
from eos.alphabag.parameters import Parameters, TC_COEFF

#: Spin (2) x colour (3) for one quark flavour. Antiquarks are carried by the
#: integrals themselves, not by a further factor of two.
G_QUARK = particles.get_particle("quark").g_degen

#: Below this mass the massless branch is taken: its closed forms are the
#: exact m -> 0 limit, and evaluating a Fermi integral at m = 0 through the
#: general machinery would only be slower.
M_MASSLESS = 1e-5

#: Below this the square root in dDelta/dT is treated as zero rather than
#: divided by, which bounds the entropy correction at the last grid point
#: before T_c instead of letting it diverge.
GAP_SLOPE_FLOOR = 1e-10


# =============================================================================
# RESULT RECORDS
# =============================================================================
@dataclass
class QuarkThermo:
    """One quark flavour as an ideal gas, with its antiquarks."""
    n: float = 0.0      # net number density (fm^-3)
    P: float = 0.0      # pressure (MeV/fm^3)
    e: float = 0.0      # energy density (MeV/fm^3)
    s: float = 0.0      # entropy density (fm^-3)
    f: float = 0.0      # free energy density f = e - T s (MeV/fm^3)


#: A flavour that is not a degree of freedom of the matter: no population, no
#: pressure, no energy, no entropy. What `SpeciesFlags.two_flavour` puts in
#: the strange slot, so the sector is off because the flag says so and not
#: because its potential happened to sit below a threshold.
_EMPTY_FLAVOUR = QuarkThermo()


@dataclass
class MatterThermo:
    """The unpaired quark sector at given potentials: bag included, leptons not.

    The gluons are NOT in here either: they carry no conserved charge and are
    a flag at the solver level, so the phase's own potential is the quarks and
    the bag.
    """
    # Inputs
    n_u: float = 0.0       # up quark density (fm^-3)
    n_d: float = 0.0       # down quark density (fm^-3)
    n_s: float = 0.0       # strange quark density (fm^-3)
    n_B: float = 0.0       # baryon density (fm^-3)
    n_C: float = 0.0       # non-leptonic charge density (fm^-3)
    n_S: float = 0.0       # strangeness density, S = +1 per s quark (fm^-3)
    T: float = 0.0         # temperature (MeV)
    mu_u: float = 0.0      # up quark chemical potential (MeV)
    mu_d: float = 0.0      # down quark chemical potential (MeV)
    mu_s: float = 0.0      # strange quark chemical potential (MeV)
    # Outputs
    P: float = 0.0         # pressure (MeV/fm^3)
    e: float = 0.0         # energy density (MeV/fm^3)
    s: float = 0.0         # entropy density (fm^-3)
    f: float = 0.0         # free energy density f = e - T s (MeV/fm^3)
    Y_C: float = 0.0       # charge fraction n_C/n_B
    Y_S: float = 0.0       # strangeness fraction n_S/n_B
    mu_B: float = 0.0      # baryon chemical potential (MeV)
    mu_C: float = 0.0      # charge chemical potential (MeV)
    mu_S: float = 0.0      # strangeness chemical potential (MeV)


@dataclass
class CFLThermo:
    """The colour-flavour locked sector at given potentials, bag included.

    Same shape as `MatterThermo`, with the gap it was evaluated at. The phase
    is electrically neutral by construction, so n_C comes back as round-off
    rather than as a solved quantity.
    """
    # Inputs
    n_B: float = 0.0        # baryon density (fm^-3)
    T: float = 0.0          # temperature (MeV)
    mu: float = 0.0         # mean quark potential (mu_u + mu_d + mu_s)/3 (MeV)
    Delta: float = 0.0      # gap at this temperature (MeV)
    Delta0: float = 0.0     # zero-temperature gap (MeV)
    # Outputs
    P: float = 0.0          # pressure (MeV/fm^3)
    e: float = 0.0          # energy density (MeV/fm^3)
    s: float = 0.0          # entropy density (fm^-3)
    f: float = 0.0          # free energy density (MeV/fm^3)
    # Quark densities and fractions
    n_u: float = 0.0        # up quark density (fm^-3)
    n_d: float = 0.0        # down quark density (fm^-3)
    n_s: float = 0.0        # strange quark density (fm^-3)
    n_C: float = 0.0        # non-leptonic charge density (fm^-3)
    Y_u: float = 1.0/3.0    # up quark fraction per baryon
    Y_d: float = 1.0/3.0    # down quark fraction per baryon
    Y_s: float = 1.0/3.0    # strange quark fraction per baryon
    Y_C: float = 0.0        # charge fraction
    Y_S: float = 0.0        # strangeness fraction
    # Conserved-charge chemical potentials
    mu_u: float = 0.0       # up quark chemical potential (MeV)
    mu_d: float = 0.0       # down quark chemical potential (MeV)
    mu_s: float = 0.0       # strange quark chemical potential (MeV)
    mu_B: float = 0.0       # baryon: mu_u + 2 mu_d (MeV)
    mu_C: float = 0.0       # charge: mu_u - mu_d (MeV)
    mu_S: float = 0.0       # strangeness: mu_s - mu_d (MeV)


# =============================================================================
# THE MASSLESS FLAVOUR IN CLOSED FORM
# =============================================================================
def P_massless(mu: float, T: float, alpha: float) -> float:
    """Pressure of a massless flavour with the alpha_s correction.

        P = [ (7/60) pi^2 T^4 (1 - 50 a/(21 pi))
              + (T^2 mu^2/2 + mu^4/(4 pi^2)) (1 - 2 a/pi) ] / (hbar c)^3

    At a = 0 this is the textbook massless Fermi gas at degeneracy g = 6 with
    antiparticles. The two correction factors are the O(alpha_s) result of
    Freedman and McLerran, Phys. Rev. D 16, 1169 (1977), in the arrangement of
    Fischer et al., Astrophys. J. Suppl. Ser. 194, 39 (2011).

    Args:
        mu: chemical potential (MeV)
        T: temperature (MeV)
        alpha: QCD coupling alpha_s

    Returns:
        Pressure (MeV/fm^3)
    """
    alpha_T_factor = 1.0 - 50.0 * alpha / (21.0 * PI)
    alpha_n_factor = 1.0 - 2.0 * alpha / PI

    T4 = T**4
    T2_mu2 = T**2 * mu**2
    mu4 = mu**4

    P_thermal = (7.0 / 60.0) * PI2 * T4 * alpha_T_factor
    P_fermi = (0.5 * T2_mu2 + mu4 / (4.0 * PI2)) * alpha_n_factor

    return (P_thermal + P_fermi) / hc3


def e_massless(mu: float, T: float, alpha: float) -> float:
    """Energy density of a massless flavour: eps = 3P, exact for any alpha_s
    since every term of `P_massless` scales as T^4, T^2 mu^2 or mu^4.
    """
    return 3.0 * P_massless(mu, T, alpha)


def n_massless(mu: float, T: float, alpha: float) -> float:
    """Net density of a massless flavour, dP/dmu of `P_massless`:

        n = ( mu T^2 + mu^3/pi^2 ) (1 - 2 a/pi) / (hbar c)^3
    """
    alpha_factor = 1.0 - 2.0 * alpha / PI

    n_value = (mu * T**2 + mu**3 / PI2) * alpha_factor

    return n_value / hc3


def s_massless(mu: float, T: float, alpha: float) -> float:
    """Entropy density of a massless flavour, dP/dT of `P_massless`:

        s = [ (7/15) pi^2 T^3 (1 - 50 a/(21 pi))
              + T mu^2 (1 - 2 a/pi) ] / (hbar c)^3
    """
    alpha_T_factor = 1.0 - 50.0 * alpha / (21.0 * PI)
    alpha_n_factor = 1.0 - 2.0 * alpha / PI

    T3 = T**3
    T_mu2 = T * mu**2

    s_thermal = (7.0 / 15.0) * PI2 * T3 * alpha_T_factor
    s_fermi = T_mu2 * alpha_n_factor

    return (s_thermal + s_fermi) / hc3


# =============================================================================
# ONE FLAVOUR AS AN IDEAL GAS
# =============================================================================
def fermi_thermo(mu: float, T: float, m: float) -> Tuple[float, float, float,
                                                         float]:
    """(n, P, eps, s) of an UNCORRECTED Fermi gas of mass m, degeneracy 6.

    From `eos.general.fermi_integrals`, which evaluates the integrals through
    the Johns-Ellis-Lattimer analytic approximation -- uniformly valid from
    the degenerate to the non-degenerate limit, exact at T = 0, and handling
    the m = 0 limit internally. Antiparticles are included, so n is the NET
    density.
    """
    result = solve_fermi_jel(mu, max(T, 0.0), m, G_QUARK,
                            include_antiparticles=True)
    return (result[0], result[1], result[2], result[3])


def kinetic_thermo(mu: float, T: float, m: float,
                   alpha: float) -> QuarkThermo:
    """One quark flavour as an ideal gas, with the alpha_s correction.

    Below `M_MASSLESS` the closed forms above are used directly. Above it, the
    correction added to the exact Fermi gas is the MASSLESS one evaluated at
    the same mu,

        X(mu,T,m,a) = X_Fermi(mu,T,m) + [ X_0(mu,T,a) - X_0(mu,T,0) ]

    for X in (n, P, eps, s). This is a prescription rather than an expansion
    of the massive result -- the true O(alpha_s) term differs at relative
    order m^2/mu^2 -- with two properties that make it safe: it reduces
    exactly to the massless forms as m -> 0, and the correction is itself a
    consistent set (dn = d(dP)/dmu, ds = d(dP)/dT, deps = 3 dP), so a free
    Fermi gas satisfying eps + P = T s + mu n still satisfies it afterwards.

    Args:
        mu: chemical potential (MeV)
        T: temperature (MeV)
        m: quark mass (MeV)
        alpha: QCD coupling alpha_s

    Returns:
        QuarkThermo with n, P, e, s, f
    """
    if m < M_MASSLESS:
        n = n_massless(mu, T, alpha)
        P = P_massless(mu, T, alpha)
        e = e_massless(mu, T, alpha)
        s = s_massless(mu, T, alpha)
        return QuarkThermo(n=n, P=P, e=e, s=s, f=e - T * s)

    n_fermi, P_fermi, e_fermi, s_fermi = fermi_thermo(mu, T, m)

    P_corr = P_massless(mu, T, alpha) - P_massless(mu, T, 0.0)
    e_corr = e_massless(mu, T, alpha) - e_massless(mu, T, 0.0)
    n_corr = n_massless(mu, T, alpha) - n_massless(mu, T, 0.0)
    s_corr = s_massless(mu, T, alpha) - s_massless(mu, T, 0.0)

    n = n_fermi + n_corr
    P = P_fermi + P_corr
    e = e_fermi + e_corr
    s = s_fermi + s_corr

    return QuarkThermo(n=n, P=P, e=e, s=s, f=e - T * s)


def quark_density(mu: float, T: float, m: float, alpha: float) -> float:
    """The density alone of one flavour -- what a residual row needs.

    Same value `kinetic_thermo(...).n` carries; separate because the
    equilibrium conditions are density conditions and evaluating P, eps and s
    on every solver iteration would be wasted work.
    """
    if m < M_MASSLESS:
        return n_massless(mu, T, alpha)
    n_fermi, _, _, _ = fermi_thermo(mu, T, m)
    n_correction = n_massless(mu, T, alpha) - n_massless(mu, T, 0.0)
    return n_fermi + n_correction


# =============================================================================
# THE BAG AND THE GLUONS
# =============================================================================
def bag_pressure(params: Parameters) -> float:
    """P_B = -B/(hbar c)^3 (MeV/fm^3), negative: the bag confines."""
    return -params.B / hc3


def bag_energy(params: Parameters) -> float:
    """eps_B = +B/(hbar c)^3 (MeV/fm^3).

    Equal and opposite to `bag_pressure`, which is why the bag cancels out of
    eps + P and leaves no term in the Euler relation.
    """
    return params.B / hc3


def gluon_thermo(T: float, alpha: float) -> QuarkThermo:
    """A thermal gluon gas: 16 massless bosons at mu = 0, alpha_s corrected.

        P_g = (8 pi^2/45) T^4 (1 - 15 a/(4 pi)) / (hbar c)^3
        e_g = 3 P_g
        s_g = (32 pi^2/45) T^3 (1 - 15 a/(4 pi)) / (hbar c)^3

    n = 0 and no conserved charge is carried, so switching the sector on
    shifts P, eps and s and nothing else. It vanishes identically at T = 0.

    Args:
        T: temperature (MeV)
        alpha: QCD coupling alpha_s

    Returns:
        QuarkThermo with n = 0, P, e, s, f
    """
    if T <= 0:
        return QuarkThermo(n=0.0, P=0.0, e=0.0, s=0.0, f=0.0)

    alpha_factor = 1.0 - 15.0 * alpha / (4.0 * PI)

    T3 = T**3
    T4 = T * T3

    P = 8.0 * PI2 / 45.0 * T4 * alpha_factor / hc3
    e = 3.0 * P
    s = 32.0 * PI2 / 45.0 * T3 * alpha_factor / hc3
    f = e - T * s

    return QuarkThermo(n=0.0, P=P, e=e, s=s, f=f)


# =============================================================================
# THE UNPAIRED SUMS
# =============================================================================
def thermo_from_mu(mu_u: float, mu_d: float, mu_s: float, T: float,
                   params: Parameters,
                   two_flavour: bool = False) -> MatterThermo:
    """The unpaired quark sector at given flavour potentials.

    Sums the three flavours of `kinetic_thermo` and subtracts the bag:

        P   = sum_q P_q - B/(hbar c)^3
        eps = sum_q eps_q + B/(hbar c)^3
        s   = sum_q s_q

    so that eps + P = T s + sum_q mu_q n_q holds with no bag term in it. The
    gluon gas is NOT added here: it carries no conserved charge and is a flag
    at the solver level.

    The conserved charges come from `eos.general.basis`, not from a local
    copy, so this cannot drift away from the hadronic bookkeeping that uses
    the same particle table: n_B = (n_u+n_d+n_s)/3, n_C = (2n_u-n_d-n_s)/3,
    n_S = n_s with S = +1 per s quark, and mu_B = mu_u + 2 mu_d,
    mu_C = mu_u - mu_d, mu_S = mu_s - mu_d.

    `two_flavour` is `SpeciesFlags.two_flavour`: the strange flavour is not a
    degree of freedom of the matter, so it contributes nothing to any of the
    three sums and n_S is zero. mu_S is set to zero with it -- no species left
    in the state carries strangeness, so S has no potential conjugate to it --
    while mu_B = mu_u + 2 mu_d and mu_C = mu_u - mu_d are untouched, neither
    reading mu_s. The bag is unchanged: it is the cost of the deconfined
    region, not of a flavour.

    Args:
        mu_u, mu_d, mu_s: quark chemical potentials (MeV)
        T: temperature (MeV)
        params: the parameter set
        two_flavour: u and d only; the s flavour leaves the matter

    Returns:
        MatterThermo
    """
    alpha = params.alpha
    m_u, m_d, m_s = params.m_u, params.m_d, params.m_s

    thermo_u = kinetic_thermo(mu_u, T, m_u, alpha)
    thermo_d = kinetic_thermo(mu_d, T, m_d, alpha)
    thermo_s = (_EMPTY_FLAVOUR if two_flavour
                else kinetic_thermo(mu_s, T, m_s, alpha))

    n_u = thermo_u.n
    n_d = thermo_d.n
    n_s = thermo_s.n

    P_kin = thermo_u.P + thermo_d.P + thermo_s.P
    e_kin = thermo_u.e + thermo_d.e + thermo_s.e
    s_kin = thermo_u.s + thermo_d.s + thermo_s.s

    P_total = P_kin + bag_pressure(params)
    e_total = e_kin + bag_energy(params)
    s_total = s_kin
    f_total = e_total - s_total * T

    n_B, n_C, n_S = quark_charges(n_u, n_d, n_s)
    mu_B, mu_C, mu_S = charge_potentials_from_quarks(mu_u, mu_d, mu_s)
    if two_flavour:
        mu_S = 0.0

    return MatterThermo(
        n_u=n_u, n_d=n_d, n_s=n_s, n_B=n_B, n_C=n_C, n_S=n_S,
        Y_C=n_C / n_B, Y_S=n_S / n_B,
        T=T,
        mu_u=mu_u, mu_d=mu_d, mu_s=mu_s, mu_B=mu_B, mu_C=mu_C, mu_S=mu_S,
        P=P_total, e=e_total, s=s_total, f=f_total
    )


# =============================================================================
# THE CFL GAP
# =============================================================================
def T_critical(Delta0: float, tc_coeff: float = TC_COEFF) -> float:
    """The temperature at which the CFL gap closes, T_c = tc_coeff * Delta0.

    `tc_coeff` is `Parameters.tc_coeff`; the default is the shipped set's
    0.57 * 2^(1/3). It is an argument rather than a module constant so that
    an inference run over CFL pairing can vary it (CLAUDE.md section 6).
    """
    return tc_coeff * Delta0


def cfl_gap(T: float, Delta0: float, tc_coeff: float = TC_COEFF) -> float:
    """The CFL pairing gap at temperature T, BCS-shaped and imposed.

        Delta(T) = Delta0 sqrt(1 - T^2/T_c^2)   for T < T_c,   0 above.

    The gap is not solved for: this model has no gap equation, and Delta0 is
    a phase selector passed per call rather than a parameter of the set;
    `tc_coeff` fixes T_c and IS a parameter.
    """
    if Delta0 <= 0 or T < 0:
        return 0.0
    T_c = T_critical(Delta0, tc_coeff)
    if T >= T_c:
        return 0.0
    return Delta0 * np.sqrt(1.0 - (T/T_c)**2)


def cfl_dgap_dT(T: float, Delta0: float,
                tc_coeff: float = TC_COEFF) -> float:
    """dDelta/dT of `cfl_gap`, which the entropy correction needs:

        dDelta/dT = -Delta0 T / ( T_c^2 sqrt(1 - T^2/T_c^2) )

    zero at T = 0 and at or above T_c. The expression diverges as T -> T_c
    from below; once the square root falls under `GAP_SLOPE_FLOOR` zero is
    returned instead, which bounds the entropy correction at the last grid
    point before T_c rather than letting it blow up.
    """
    if Delta0 <= 0 or T <= 0:
        return 0.0
    T_c = T_critical(Delta0, tc_coeff)
    if T >= T_c:
        return 0.0
    sqrt_term = np.sqrt(1.0 - (T/T_c)**2)
    if sqrt_term < GAP_SLOPE_FLOOR:
        return 0.0
    return -Delta0 * T / (T_c**2 * sqrt_term)


# =============================================================================
# THE CFL CORRECTIONS AND SUMS
# =============================================================================
def cfl_P_correction(mu_u: float, mu_d: float, mu_s: float,
                     T: float, Delta0: float,
                     tc_coeff: float = TC_COEFF) -> float:
    """The pairing term of the CFL pressure,

        dP = (mu_u^2 + mu_d^2 + mu_s^2) Delta(T)^2 / ( pi^2 (hbar c)^3 ) ,

    written per flavour so the three potentials need not be equal. At a common
    mu it is 3 mu^2 Delta^2/pi^2, the term of Alford, Braby, Paris and Reddy,
    Astrophys. J. 629, 969 (2005).

    The -3 m_s^2 mu^2/(4 pi^2) term that reference carries alongside it is NOT
    added: it is the leading expansion of the massive strange Fermi gas, which
    `kinetic_thermo` already carries exactly, and adding both would count it
    twice.
    """
    Delta = cfl_gap(T, Delta0, tc_coeff)
    mu_sum_sq = mu_u**2 + mu_d**2 + mu_s**2
    return mu_sum_sq * Delta**2 / (PI2 * hc3)


def cfl_n_correction(mu: float, T: float, Delta0: float,
                     tc_coeff: float = TC_COEFF) -> float:
    """The pairing term of one flavour's density, d(dP)/dmu_q:

        dn_q = 2 mu_q Delta(T)^2 / ( pi^2 (hbar c)^3 )
    """
    Delta = cfl_gap(T, Delta0, tc_coeff)
    return 2.0 * mu * Delta**2 / (PI2 * hc3)


def cfl_s_correction(mu_u: float, mu_d: float, mu_s: float,
                     T: float, Delta0: float,
                     tc_coeff: float = TC_COEFF) -> float:
    """The pairing term of the entropy, d(dP)/dT:

        ds = 2 (mu_u^2 + mu_d^2 + mu_s^2) Delta(T) (dDelta/dT)
             / ( pi^2 (hbar c)^3 )

    negative wherever the gap is falling: pairing removes states from the
    Fermi surface, so the condensate carries less entropy than the gas it
    replaces.
    """
    Delta = cfl_gap(T, Delta0, tc_coeff)
    dDelta_dT = cfl_dgap_dT(T, Delta0, tc_coeff)
    mu_sum_sq = mu_u**2 + mu_d**2 + mu_s**2
    return 2.0 * mu_sum_sq * Delta * dDelta_dT / (PI2 * hc3)


def cfl_thermo_from_mu(mu_u: float, mu_d: float, mu_s: float, T: float,
                       Delta0: float, params: Parameters) -> CFLThermo:
    """The colour-flavour locked sector at given flavour potentials.

    The same quark gas as `thermo_from_mu`, with the pairing term added to the
    pressure and everything else taken as a derivative of it:

        P     = sum_q P_q + dP - B/(hbar c)^3
        n_q   = n_q + 2 mu_q Delta^2/(pi^2 (hbar c)^3)
        s     = sum_q s_q + ds
        f     = -P + sum_q mu_q n_q
        eps   = f + T s

    eps is DEFINED by the last line rather than summed from the flavours, so
    the Euler relation eps + P = T s + sum_q mu_q n_q holds in the paired
    phase by construction; `verify/` checks it all the same.

    Gluons are not part of this potential: in the CFL phase they are
    Meissner-massive and their thermal population is suppressed. The sector
    stays available as a flag at the solver level.

    Args:
        mu_u, mu_d, mu_s: quark chemical potentials (MeV)
        T: temperature (MeV)
        Delta0: zero-temperature pairing gap (MeV)
        params: the parameter set

    Returns:
        CFLThermo
    """
    alpha = params.alpha
    m_u, m_d, m_s = params.m_u, params.m_d, params.m_s

    Delta = cfl_gap(T, Delta0, params.tc_coeff)

    thermo_u = kinetic_thermo(mu_u, T, m_u, alpha)
    thermo_d = kinetic_thermo(mu_d, T, m_d, alpha)
    thermo_s = kinetic_thermo(mu_s, T, m_s, alpha)

    P_unpaired = thermo_u.P + thermo_d.P + thermo_s.P
    n_u_unpaired = thermo_u.n
    n_d_unpaired = thermo_d.n
    n_s_unpaired = thermo_s.n
    s_unpaired = thermo_u.s + thermo_d.s + thermo_s.s

    tc = params.tc_coeff
    P_corr = cfl_P_correction(mu_u, mu_d, mu_s, T, Delta0, tc)
    n_u_corr = cfl_n_correction(mu_u, T, Delta0, tc)
    n_d_corr = cfl_n_correction(mu_d, T, Delta0, tc)
    n_s_corr = cfl_n_correction(mu_s, T, Delta0, tc)
    s_corr = cfl_s_correction(mu_u, mu_d, mu_s, T, Delta0, tc)

    B = params.B / hc3
    P = P_unpaired + P_corr - B
    n_u = n_u_unpaired + n_u_corr
    n_d = n_d_unpaired + n_d_corr
    n_s = n_s_unpaired + n_s_corr
    s = s_unpaired + s_corr

    f = -P + mu_u * n_u + mu_d * n_d + mu_s * n_s
    e = f + T * s

    n_B, n_C, n_S = quark_charges(n_u, n_d, n_s)
    mu_B, mu_C, mu_S = charge_potentials_from_quarks(mu_u, mu_d, mu_s)

    Y_u = n_u / n_B if n_B > 0 else 1.0/3.0
    Y_d = n_d / n_B if n_B > 0 else 1.0/3.0
    Y_s = n_s / n_B if n_B > 0 else 1.0/3.0
    Y_C = n_C / n_B if n_B > 0 else 0.0
    Y_S = n_S / n_B if n_B > 0 else 0.0

    mu = (mu_u + mu_d + mu_s) / 3.0

    return CFLThermo(
        n_B=n_B, T=T, mu=mu, Delta=Delta, Delta0=Delta0,
        P=P, e=e, s=s, f=f,
        n_u=n_u, n_d=n_d, n_s=n_s, n_C=n_C,
        Y_u=Y_u, Y_d=Y_d, Y_s=Y_s, Y_C=Y_C, Y_S=Y_S,
        mu_u=mu_u, mu_d=mu_d, mu_s=mu_s,
        mu_B=mu_B, mu_C=mu_C, mu_S=mu_S,
    )
