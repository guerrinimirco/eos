"""Thermal meson gas: the model-independent machinery.

At T > 0 the pseudoscalar nonet (pi, K, eta, eta') and optionally the vector
nonet (rho, omega, K*, phi) ride on a mean field as one-body ideal Bose
gases, following Mueller, Nucl. Phys. A 618 (1997) 349 and Lavagno, Phys. Rev.
C 81 (2010) 044909 (see also arXiv:1210.0400 and arXiv:2301.06909). The gas
carries electric charge and strangeness — which enter neutrality and
fixed-Y_C / fixed-Y_S conditions — but no baryon number, and it does not
source the meson field equations.

The split between this module and a model:

* HERE: the species list with quantum numbers, degeneracies and masses; the
  conjugation rules (mu*_{j-} = -mu*_{j+}, neutral strangeless mesons at
  mu* = 0); the summed charges and thermodynamics via the Bose integrals of
  :mod:`eos.general.bose_integrals`.
* THE MODEL: the three independent effective potentials

      mu*_{pi+},   mu*_{K+},   mu*_{K0}

  computed from its own couplings and mean fields. In an RMF they are the
  charge/strangeness potentials shifted by the vector fields the meson's quark
  content couples to; a model with constant couplings and one with
  density-dependent ones differ only in that arithmetic, which is why it stays
  with the model.

The masses are physical (PDG) and live here, once, for every model — a meson
mass is not a model parameter, and two models disagreeing about m_pi is a
difference with no physics in it. The isospin partners are SEPARATE:
m_pi0 = 134.977 against m_pi+- = 139.570, and m_K0 = 497.611 against
m_K+- = 493.677. The splittings are a few MeV on a gas whose population goes
like exp(-m/T), so the 4.6 MeV pion splitting alone changes the pi0 density by
about 17 percent at T = 30 MeV.

Strangeness follows this repository's sign convention, S = +1 per s quark:
K+ (u sbar) carries S = -1 and K- (ubar s) carries S = +1. No rearrangement
term enters any mu*_j — the gas is a spectator to Sigma^R.

BOSE-EINSTEIN CONDENSATION IS NOT IMPLEMENTED. Where |mu*_j| reaches m_j the
species condenses and this ideal-gas expression stops describing it;
`solve_bose_jel` caps mu at m rather than diverging, so a caller that does not
look would silently receive the saturated value instead of a condensate. Every
entry point therefore reports `condensation`, the largest |mu*_j| / m_j over
the active species, so the model can refuse the point (CLAUDE.md section 6:
what the state says about itself is a return value). It is not a corner case —
in SFHo at n_B = 1.2 fm^-3 the ratio reaches 3.5.

Units: potentials, masses and T in MeV; densities fm^-3, P and e in MeV/fm^3
(the fm-based units the Bose integrals return).
"""
from eos.general.bose_integrals import solve_bose_jel

#: Physical meson masses [MeV], Particle Data Group. The isospin partners are
#: separate constants; see the module docstring for why that matters.
M_PI_PM, M_PI_0 = 139.57039, 134.9768
M_K_PM, M_K_0 = 493.677, 497.611
M_ETA, M_ETAP = 547.862, 957.78
M_RHO, M_OMEGA, M_KSTAR, M_PHI = 775.26, 782.66, 891.67, 1019.461


def _bose(mu_eff, T, m, g):
    """(n, P, e, s) fm-based; antiparticles are separate species here."""
    n, P, e, s, _ = solve_bose_jel(mu_eff, T, m, g, include_antiparticles=False)
    return n, P, e, s


def meson_families(mu_pi_plus, mu_K_plus, mu_K0,
                   include_pseudoscalars=False, include_thermal_vectors=False):
    """(name, mu_eff, mass, Q, S, g) per thermal meson species.

    Antiparticles are listed explicitly with conjugated potentials; the
    strangeless neutral mesons sit at mu* = 0. The vector nonet reuses the
    same three potentials — rho+ with the pion's, K*+ / K*0 with the kaons' —
    because the shift depends on the quark content, which vector and
    pseudoscalar partners share.
    """
    families = []
    if include_pseudoscalars:
        families += [
            ("pi+", mu_pi_plus, M_PI_PM, +1, 0, 1.0),
            ("pi-", -mu_pi_plus, M_PI_PM, -1, 0, 1.0),
            ("pi0", 0.0, M_PI_0, 0, 0, 1.0),
            ("K+", mu_K_plus, M_K_PM, +1, -1, 1.0),
            ("K-", -mu_K_plus, M_K_PM, -1, +1, 1.0),
            ("K0", mu_K0, M_K_0, 0, -1, 1.0),
            ("K0_bar", -mu_K0, M_K_0, 0, +1, 1.0),
            ("eta", 0.0, M_ETA, 0, 0, 1.0),
            ("eta_prime", 0.0, M_ETAP, 0, 0, 1.0),
        ]
    if include_thermal_vectors:
        families += [
            ("rho+", mu_pi_plus, M_RHO, +1, 0, 3.0),
            ("rho-", -mu_pi_plus, M_RHO, -1, 0, 3.0),
            ("rho0", 0.0, M_RHO, 0, 0, 3.0),
            ("omega", 0.0, M_OMEGA, 0, 0, 3.0),
            ("K*+", mu_K_plus, M_KSTAR, +1, -1, 3.0),
            ("K*-", -mu_K_plus, M_KSTAR, -1, +1, 3.0),
            ("K*0", mu_K0, M_KSTAR, 0, -1, 3.0),
            ("K*0_bar", -mu_K0, M_KSTAR, 0, +1, 3.0),
            ("phi", 0.0, M_PHI, 0, 0, 3.0),
        ]
    return families


def condensation_ratio(mu_pi_plus, mu_K_plus, mu_K0,
                       include_pseudoscalars=False,
                       include_thermal_vectors=False):
    """max_j |mu*_j| / m_j over the active species.

    Below 1 the ideal Bose gas below describes the species; at 1 it condenses
    and this module's expressions stop applying. See the module docstring.
    """
    worst = 0.0
    for _name, mu_eff, m, _Q, _S, _g in meson_families(
            mu_pi_plus, mu_K_plus, mu_K0,
            include_pseudoscalars, include_thermal_vectors):
        worst = max(worst, abs(mu_eff) / m)
    return worst


def thermal_meson_charges(mu_pi_plus, mu_K_plus, mu_K0, T,
                          include_pseudoscalars=False,
                          include_thermal_vectors=False):
    """(n_C, n_S) of the gas [fm^-3] — the piece a solver's charge and
    strangeness constraints need, skipping the P/e/s work.

    Zero at T <= 0 or with both nonets off, so callers can add it blindly.
    """
    if T <= 0.0 or not (include_pseudoscalars or include_thermal_vectors):
        return 0.0, 0.0
    n_C = n_S = 0.0
    for _name, mu_eff, m, Q, S, g in meson_families(
            mu_pi_plus, mu_K_plus, mu_K0,
            include_pseudoscalars, include_thermal_vectors):
        n = _bose(mu_eff, T, m, g)[0]
        n_C += Q * n
        n_S += S * n
    return n_C, n_S


def thermal_meson_thermo(mu_pi_plus, mu_K_plus, mu_K0, T,
                         include_pseudoscalars=False,
                         include_thermal_vectors=False):
    """Full gas thermodynamics at the supplied effective potentials.

    Returns a dict (fm-based units): P, e, s, the net charge and strangeness
    densities n_C and n_S, mu_dot_n = sum_j mu*_j n_j for the
    Hugenholtz–Van Hove bookkeeping — the Bose gas obeys e + P = T s + mu* n
    per species — the per-species `densities`, and `condensation`, the largest
    |mu*_j| / m_j, which a caller MUST test before trusting the rest (see the
    module docstring).
    """
    if T <= 0.0 or not (include_pseudoscalars or include_thermal_vectors):
        return dict(P=0.0, e=0.0, s=0.0, n_C=0.0, n_S=0.0, mu_dot_n=0.0,
                    densities={}, condensation=0.0)
    P = e = s = n_C = n_S = mudotn = 0.0
    densities, worst = {}, 0.0
    for name, mu_eff, m, Q, S, g in meson_families(
            mu_pi_plus, mu_K_plus, mu_K0,
            include_pseudoscalars, include_thermal_vectors):
        n, P_j, e_j, s_j = _bose(mu_eff, T, m, g)
        densities[name] = n
        P += P_j
        e += e_j
        s += s_j
        n_C += Q * n
        n_S += S * n
        mudotn += mu_eff * n
        worst = max(worst, abs(mu_eff) / m)
    return dict(P=P, e=e, s=s, n_C=n_C, n_S=n_S, mu_dot_n=mudotn,
                densities=densities, condensation=worst)
