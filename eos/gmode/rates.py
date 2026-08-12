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
pass your own callable wherever `gamma` or `rate_model` is taken.

Units: n_B in fm^-3 and T in MeV on the boundary, as everywhere in `eos`;
`gamma` is returned in s^-1 so it can be compared directly with a mode
frequency. Internally `lambda` is in MeV^3 and `A` in MeV^-2.
"""
import numpy as np
from scipy.special import zeta

from eos.general.physics_constants import hc, hc3
from eos.general.particles import Electron, Muon
from eos.dd2.solver import solve_composition
from eos.dd2.thermodynamics import neutralizing_leptons

#: Fermi coupling squared times the Cabibbo factor, G_F^2 cos^2(theta_c)
#: [MeV^-4]. Alford and Harris (2018), text below their Eq. (8).
G2_FERMI = 1.1e-22

#: Axial-vector coupling and the pion-nucleon p-wave coupling.
G_A = 1.26
F_PI_NN = 1.0

#: Charged pion mass [MeV].
M_PI = 139.57039

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


def lambda_direct_urca(p_Fn, p_Fp, p_Fe, m_n_eff, m_p_eff, T):
    """Subthermal direct-Urca response coefficient [MeV^3].

    All momenta and masses in MeV, T in MeV. Returns 0 below the threshold set
    by the triangle inequality p_Fn <= p_Fp + p_Fe.
    """
    allowed = np.where(p_Fn <= p_Fp + p_Fe, 1.0, 0.0)
    g_tilde2 = G2_FERMI * (1.0 + 3.0 * G_A**2)
    return (17.0 / (240.0 * np.pi)) * g_tilde2 * m_n_eff * m_p_eff \
        * p_Fe * T**4 * allowed


def lambda_modified_urca(p_Fn, p_Fp, p_Fe, m_n, m_p, T):
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

    gamma_eq = (A_MU * G2_FERMI * F_PI_NN**4 * G_A**2
                * (m_n**3 * m_p / M_PI**4)
                * p_Fn**4 * p_Fp / (p_Fn**2 + M_PI**2)**2
                * vartheta * T**7)
    # ponytail: the subthermal factor is calibrated on direct Urca, where both
    # lambda and Gamma are known in closed form, and reused for the T^7 process.
    # The exact modified-Urca phase-space coefficient would replace
    # SUBTHERMAL_FACTOR here; it is an O(1) correction, well inside the
    # uncertainty the Fermi-surface approximation already carries.
    return SUBTHERMAL_FACTOR * gamma_eq / T


def susceptibility_A(par, n_B, Y_p, T=0.0, muons=True, rel_dn=1e-3):
    """A = (d mu_Delta / d n_n) at fixed n_B, in MeV^-2.

    Central finite difference: neutrons are traded for protons at constant
    baryon density, and the leptons are re-neutralised against the new proton
    fraction at each step, which is what supplies the mu_e part of mu_Delta.
    `A > 0` for any stable equation of state — it is the curvature of the
    energy against isospin, i.e. essentially the symmetry energy.
    """
    dn = rel_dn * n_B

    def mu_delta(n_n):
        n_p = n_B - n_n
        pt = solve_composition(par, n_n, n_p, T=T, check_consistency=False)
        mu_e, _e, _m = neutralizing_leptons(
            n_p * hc3, Electron.mass, Muon.mass, muons, T)
        return -pt.mu_C - mu_e

    n_n0 = (1.0 - Y_p) * n_B
    A_fm = (mu_delta(n_n0 + dn) - mu_delta(n_n0 - dn)) / (2.0 * dn)
    return A_fm / hc3                          # MeV fm^3 -> MeV^-2


def equilibration_rate(par, n_B, Y_p, T, muons=True, processes="both",
                       rel_dn=1e-3):
    """Beta-equilibration rate gamma = lambda * A, in s^-1.

    par       : DD2 `Parametrization`
    n_B       : baryon density [fm^-3]
    Y_p       : proton fraction of the equilibrium state
    T         : temperature [MeV]
    processes : "both", "direct", or "modified"

    Compare the result with the mode's angular frequency omega = 2 pi nu: the
    composition is frozen when gamma << omega and equilibrated when
    gamma >> omega. Returns 0.0 at T = 0, where every rate vanishes and the
    frozen limit is exact.
    """
    if T <= 0.0:
        return 0.0

    n_p = Y_p * n_B
    n_n = n_B - n_p
    pt = solve_composition(par, n_n, n_p, T=T, check_consistency=False)
    mu_e, _e, _m = neutralizing_leptons(
        n_p * hc3, Electron.mass, Muon.mass, muons, T)

    p_Fn, p_Fp = _fermi_momentum(n_n), _fermi_momentum(n_p)
    p_Fe = np.sqrt(max(mu_e**2 - Electron.mass**2, 0.0))
    # Each rate gets the mass its own source prescribes: the Dirac effective
    # masses for direct Urca, the vacuum masses for modified Urca. The two
    # nucleons carry their own m*_i; they coincide under DD2's default
    # averaged kernel mass and differ when nucleon_mass_mode splits them.
    m_eff_n, m_eff_p = pt.m_eff("n"), pt.m_eff("p")

    lam = 0.0
    if processes in ("both", "direct"):
        lam += lambda_direct_urca(p_Fn, p_Fp, p_Fe, m_eff_n, m_eff_p, T)
    if processes in ("both", "modified"):
        lam += lambda_modified_urca(p_Fn, p_Fp, p_Fe, par.m_n, par.m_p, T)
    if processes not in ("both", "direct", "modified"):
        raise ValueError("processes must be 'both', 'direct' or 'modified', "
                         f"got {processes!r}")

    A = susceptibility_A(par, n_B, Y_p, T=T, muons=muons, rel_dn=rel_dn)
    return float(lam * A / HBAR_MEV_S)         # MeV -> s^-1


def equilibration_rate_along(par, n_B_grid, Y_p_grid, T, **kw):
    """`equilibration_rate` along a density profile, as an array [s^-1].

    Points that fail to converge come back nan rather than aborting the sweep,
    matching the behaviour of the sound-speed sequence helpers.
    """
    out = []
    for n_B, Y_p in zip(np.atleast_1d(n_B_grid), np.atleast_1d(Y_p_grid)):
        try:
            out.append(equilibration_rate(par, float(n_B), float(Y_p), T, **kw))
        except (RuntimeError, ValueError):
            out.append(np.nan)
    return np.asarray(out, dtype=float)


__all__ = [
    "equilibration_rate", "equilibration_rate_along", "susceptibility_A",
    "lambda_direct_urca", "lambda_modified_urca",
]
