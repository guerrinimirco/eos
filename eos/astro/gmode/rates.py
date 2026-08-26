"""
gmode/rates.py
==============
How fast matter re-establishes beta equilibrium, and therefore whether a
composition g-mode survives.

The question a g-mode asks of the microphysics is a single comparison: is the
chemical equilibration rate `gamma` large or small compared with the mode's
angular frequency `omega`?

    gamma << omega   composition frozen during the oscillation. The element
                     responds with c_ad, buoyancy is maximal, the g-mode exists.
    gamma >> omega   composition equilibrates instantaneously. The element
                     responds with c_eq, buoyancy vanishes, the mode is gone.
    gamma ~  omega   maximal dissipation: the reaction lags the compression by
                     a quarter cycle, which is bulk viscosity, and the mode is
                     damped hardest.

Since a g-mode sits at a few hundred Hz, `omega` is ~1e3 s^-1, and the Urca
rates that set `gamma` are steep functions of temperature. Cold stars
(T << 1 MeV) are firmly in the frozen limit; merger remnants and protoneutron
stars at a few MeV are not.

Definitions
-----------
The chemical imbalance driving the reactions is

    mu_Delta = mu_n - mu_p - mu_e

and in the subthermal regime `mu_Delta << T` the net conversion rate is linear
in it, `Gamma_{n->p} - Gamma_{p->n} = lambda mu_Delta`. The relaxation rate of
the composition is then

    gamma = lambda * A ,    A = (d mu_Delta / d n_n)_{n_B}

`A` is a purely thermodynamic susceptibility, taken here by finite difference on
the equation of state; `lambda` is the weak-interaction physics. This is the
decomposition of Alford, Harutyunyan and Sedrakian, "Bulk viscosity of baryonic
matter with trapped neutrinos", Particles 3, 34 (2020), Eqs. (10), (16), (21).

Rates implemented
-----------------
**Direct Urca**, n -> p + e + nubar and p + e -> n + nu, exact expression:

    lambda_dU = (17 / 240 pi) Gtilde^2 m*_n m*_p p_Fe T^4
                * theta(p_Fp + p_Fe - p_Fn)

with Gtilde^2 = G_F^2 cos^2(theta_c) (1 + 3 g_A^2). The step function is the
triangle inequality on the Fermi momenta: below the threshold density the
proton and electron Fermi momenta cannot add up to the neutron's, and the
process is forbidden. Same reference, Eq. (36).

**Modified Urca**, n + N -> p + N + e + nubar, which has no threshold and
therefore dominates everywhere below the direct Urca onset. Its equilibrium
rate (neutron spectator, the dominant channel) is

    Gamma_mU = A_mU G^2 f^4 g_A^2 (m*_n^3 m*_p / m_pi^4)
               * p_Fn^4 p_Fp / (p_Fn^2 + m_pi^2)^2 * vartheta_n * T^7

from Alford and Harris, "Beta equilibrium in neutron star mergers",
Phys. Rev. C 98, 065806 (2018), Eqs. (12)-(13), with A_mU = 7 * 2300 / (64 pi^9).
The proton-spectator channel is omitted: it is only allowed below a proton
fraction of 1/65, far below any density of interest here.

Accuracy, and the knob that is deliberately left exposed
--------------------------------------------------------
These are Fermi-surface-approximation rates, valid for T well below the Fermi
energies; above ~1 MeV the approximation degrades and below the direct Urca
threshold it underestimates the true rate, because thermally excited particles
away from the Fermi surface can satisfy the kinematics that particles on it
cannot. Nucleon superfluidity, not modelled here, suppresses both rates by
orders of magnitude once T drops below the gap. `gamma` therefore carries real
model uncertainty, and every entry point accepts a user-supplied replacement:
pass your own callable wherever `gamma` or `rate_model` is taken, and the weak
couplings themselves as a `WeakCouplings` (CLAUDE.md section 6 -- they are
parameters, so they are arguments, and the published values are a named
default rather than module constants nobody can override).

What the model supplies, and what this module supplies
------------------------------------------------------
Of everything the rates need, only three numbers per point come from the
equation of state: the two Dirac effective masses and the isospin
susceptibility `A`. The Fermi momenta follow from n_B and Y_p by kinematics
alone and the electron potential from `eos.general.thermodynamics_leptons`, so
this module takes those three as ARGUMENTS and imports no model -- the same
division the sound-speed contract makes (`eos.general.sound_speeds`), for the
same reason (CLAUDE.md section 1). `susceptibility_A` takes the chemical
imbalance as a callable because it is a derivative of it, and only the model
can evaluate one at a perturbed composition.

Units: n_B in fm^-3 and T in MeV on the boundary, as everywhere in `eos`;
`gamma` is returned in s^-1 so it can be compared directly with a mode
frequency. Internally `lambda` is in MeV^3 and `A` in MeV^-2.
"""
from dataclasses import dataclass

import numpy as np
from scipy.special import zeta

from eos.general.physics_constants import hc, hc3
from eos.general.particles import Electron, Neutron, Proton, PiP
from eos.general.thermodynamics_leptons import neutralizing_leptons


@dataclass(frozen=True)
class WeakCouplings:
    """The weak-sector constants the Urca rates are built from.

    G2_fermi : Fermi coupling squared times the Cabibbo factor,
               G_F^2 cos^2(theta_c) [MeV^-4]. Alford and Harris (2018), text
               below their Eq. (8).
    g_A      : axial-vector coupling of the nucleon.
    f_pi_NN  : pion-nucleon p-wave coupling entering the modified-Urca matrix
               element.

    A parameter takes no arguments, so these are stored numbers and therefore
    arguments (CLAUDE.md section 6): `WeakCouplings()` is the published set,
    and a rate computed with a different g_A is a keyword away rather than a
    source edit. The charged-pion mass is NOT here -- it is a particle
    property and comes from `eos.general.particles` (section 7).
    """
    G2_fermi: float = 1.1e-22
    g_A: float = 1.26
    f_pi_NN: float = 1.0


#: The published values, used wherever a caller supplies no couplings.
WEAK_COUPLINGS = WeakCouplings()

#: Charged pion mass [MeV], from the single home for particle properties.
M_PI = PiP.mass

#: hbar [MeV s], for turning a rate in MeV into s^-1.
HBAR_MEV_S = 6.582119569e-22

#: Direct Urca equilibrium-rate coefficient, 3(pi^2 zeta(3) + 15 zeta(5))/(16 pi^5).
A_DU = 3.0 * (np.pi**2 * zeta(3) + 15.0 * zeta(5)) / (16.0 * np.pi**5)

#: Modified Urca equilibrium-rate coefficient, 7 * 2300 / (64 pi^9).
A_MU = 7.0 * 2300.0 / (64.0 * np.pi**9)

#: Ratio of the subthermal response to the equilibrium rate over T, fixed by
#: the two independent direct-Urca results quoted in the module docstring:
#: lambda_dU / (Gamma_dU / T) = (17/240 pi) / A_DU.
SUBTHERMAL_FACTOR = (17.0 / (240.0 * np.pi)) / A_DU


def _fermi_momentum(n):
    """p_F = hbar c (3 pi^2 n)^{1/3}, n in fm^-3, result in MeV."""
    return hc * (3.0 * np.pi**2 * np.maximum(n, 0.0))**(1.0 / 3.0)


def lambda_direct_urca(p_Fn, p_Fp, p_Fe, m_n_eff, m_p_eff, T,
                       couplings=WEAK_COUPLINGS):
    """Subthermal direct-Urca response coefficient [MeV^3].

    All momenta and masses in MeV, T in MeV. Returns 0 below the threshold set
    by the triangle inequality p_Fn <= p_Fp + p_Fe.
    """
    allowed = np.where(p_Fn <= p_Fp + p_Fe, 1.0, 0.0)
    g_tilde2 = couplings.G2_fermi * (1.0 + 3.0 * couplings.g_A**2)
    return (17.0 / (240.0 * np.pi)) * g_tilde2 * m_n_eff * m_p_eff \
        * p_Fe * T**4 * allowed


def lambda_modified_urca(p_Fn, p_Fp, p_Fe, m_n, m_p, T,
                         couplings=WEAK_COUPLINGS):
    """Subthermal modified-Urca response coefficient [MeV^3].

    Neutron-spectator channel, no density threshold. The phase-space factor
    `vartheta_n` is unity once p_Fn exceeds p_Fp + p_Fe and is reduced below
    that, per Alford and Harris (2018) Eq. (12).

    `m_n` and `m_p` are the **vacuum** nucleon masses, not the effective ones:
    the source fixes the nucleon dispersion as E = U + m + p^2/(2m) with the
    vacuum rest mass, and puts all the medium dependence into the mean field U.
    Using an effective mass here instead would introduce a spurious m*^4
    suppression at high density, since this prefactor enters to the fourth
    power. The direct-Urca expression, taken from a different source, does use
    the effective mass, as that reference specifies.
    """
    excess = p_Fp + p_Fe - p_Fn
    vartheta = np.where(
        excess <= 0.0, 1.0,
        1.0 - (3.0 / 8.0) * excess**2 / np.maximum(p_Fp * p_Fe, 1e-30))
    vartheta = np.clip(vartheta, 0.0, 1.0)

    gamma_eq = (A_MU * couplings.G2_fermi * couplings.f_pi_NN**4
                * couplings.g_A**2
                * (m_n**3 * m_p / M_PI**4)
                * p_Fn**4 * p_Fp / (p_Fn**2 + M_PI**2)**2
                * vartheta * T**7)
    # ponytail: the subthermal factor is calibrated on direct Urca, where both
    # lambda and Gamma are known in closed form, and reused for the T^7 process.
    # The exact modified-Urca phase-space coefficient would replace
    # SUBTHERMAL_FACTOR here; it is an O(1) correction, well inside the
    # uncertainty the Fermi-surface approximation already carries.
    return SUBTHERMAL_FACTOR * gamma_eq / T


def susceptibility_A(mu_Delta, n_B, Y_p, rel_dn=1e-3):
    """A = (d mu_Delta / d n_n) at fixed n_B, in MeV^-2.

    mu_Delta : callable mapping the neutron density n_n [fm^-3] to the chemical
               imbalance mu_n - mu_p - mu_e [MeV] at the SAME total baryon
               density n_B, with the leptons re-neutralised against the new
               proton fraction -- that is what supplies the mu_e part. Only the
               model can evaluate it, which is why it arrives as a callable and
               this module imports no model (CLAUDE.md section 1).
    n_B      : baryon density [fm^-3]
    Y_p      : proton fraction of the equilibrium state

    A central finite difference trading neutrons for protons at constant baryon
    density. `A > 0` for any stable equation of state -- it is the curvature of
    the energy against isospin, i.e. essentially the symmetry energy.
    """
    dn = rel_dn * n_B
    n_n0 = (1.0 - Y_p) * n_B
    A_fm = (mu_Delta(n_n0 + dn) - mu_Delta(n_n0 - dn)) / (2.0 * dn)
    return A_fm / hc3                          # MeV fm^3 -> MeV^-2


def equilibration_rate(n_B, Y_p, T, m_eff_n, m_eff_p, A, muons=True,
                       processes="both", m_n=Neutron.mass, m_p=Proton.mass,
                       couplings=WEAK_COUPLINGS):
    """Beta-equilibration rate gamma = lambda * A, in s^-1.

    n_B       : baryon density [fm^-3]
    Y_p       : proton fraction of the equilibrium state
    T         : temperature [MeV]
    m_eff_n,
    m_eff_p   : Dirac effective masses [MeV] at that state -- the model's
                contribution, and the reason direct Urca is medium-dependent
    A         : isospin susceptibility [MeV^-2] from `susceptibility_A`
    processes : "both", "direct", or "modified"
    m_n, m_p  : VACUUM nucleon masses [MeV]; a model that carries its own in its
                parameter dataclass passes them, otherwise the values of
                `eos.general.particles` are used
    couplings : `WeakCouplings`

    The Fermi momenta come from n_B and Y_p by kinematics and the electron
    potential from the neutralising lepton gas, so nothing else is needed from
    the equation of state. Compare the result with the mode's angular frequency
    omega = 2 pi nu: the composition is frozen when gamma << omega and
    equilibrated when gamma >> omega. Returns 0.0 at T = 0, where every rate
    vanishes and the frozen limit is exact.
    """
    if processes not in ("both", "direct", "modified"):
        raise ValueError("processes must be 'both', 'direct' or 'modified', "
                         f"got {processes!r}")
    if T <= 0.0:
        return 0.0

    n_p = Y_p * n_B
    n_n = n_B - n_p
    mu_e, _e, _m = neutralizing_leptons(n_p, T, include_muons=muons)

    p_Fn, p_Fp = _fermi_momentum(n_n), _fermi_momentum(n_p)
    p_Fe = np.sqrt(max(mu_e**2 - Electron.mass**2, 0.0))

    lam = 0.0
    # Each rate gets the mass its own source prescribes: the Dirac effective
    # masses for direct Urca, the vacuum masses for modified Urca.
    if processes in ("both", "direct"):
        lam += lambda_direct_urca(p_Fn, p_Fp, p_Fe, m_eff_n, m_eff_p, T,
                                  couplings=couplings)
    if processes in ("both", "modified"):
        lam += lambda_modified_urca(p_Fn, p_Fp, p_Fe, m_n, m_p, T,
                                    couplings=couplings)

    return float(lam * A / HBAR_MEV_S)         # MeV -> s^-1


def equilibration_rate_along(n_B_grid, Y_p_grid, T, m_eff_n_grid,
                             m_eff_p_grid, A_grid, **kw):
    """`equilibration_rate` along a density profile, as an array [s^-1].

    The three model-supplied columns are parallel to `n_B_grid`. Points that
    fail come back nan rather than aborting the sweep, matching the behaviour
    of the sound-speed sequence helpers.
    """
    out = []
    for n_B, Y_p, mn, mp, A in zip(np.atleast_1d(n_B_grid),
                                   np.atleast_1d(Y_p_grid),
                                   np.atleast_1d(m_eff_n_grid),
                                   np.atleast_1d(m_eff_p_grid),
                                   np.atleast_1d(A_grid)):
        try:
            out.append(equilibration_rate(float(n_B), float(Y_p), T, float(mn),
                                          float(mp), float(A), **kw))
        except (RuntimeError, ValueError):
            out.append(np.nan)
    return np.asarray(out, dtype=float)


__all__ = [
    "WeakCouplings", "WEAK_COUPLINGS",
    "equilibration_rate", "equilibration_rate_along", "susceptibility_A",
    "lambda_direct_urca", "lambda_modified_urca",
]
