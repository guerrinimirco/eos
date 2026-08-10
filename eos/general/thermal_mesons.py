"""Thermal meson gas: the model-independent machinery.

At T > 0 the pseudoscalar nonet (pi, K, eta, eta') and optionally the vector
nonet (rho, omega, K*, phi) ride on a mean field as one-body ideal Bose
gases, following Lavagno, Phys. Rev. C 81, 044909 (2010) (see also
arXiv:1210.0400). The gas carries electric charge and strangeness — which
enter neutrality and fixed-Y_C / fixed-Y_S conditions — but no baryon number,
and it does not source the meson field equations.

The split between this module and a model:

* HERE: the species list with quantum numbers, degeneracies and PDG masses;
  the conjugation rules (mu*_{j-} = -mu*_{j+}, neutral strangeless mesons at
  mu* = 0); the summed charges and thermodynamics via the Bose integrals of
  :mod:`eos.general.bose_integrals`.
* THE MODEL: the three independent effective potentials

      mu*_{pi+},   mu*_{K+},   mu*_{K0}

  computed from its own couplings and mean fields. In an RMF they are the
  charge/strangeness potentials shifted by the vector fields the meson's
  quark content couples to; a model with constant couplings and one with
  density-dependent ones differ only in that arithmetic, which is why it
  stays with the model.

Strangeness follows this repository's sign convention, S = +1 per s quark:
K+ (u sbar) carries S = -1 and K- (ubar s) carries S = +1. No rearrangement
term enters any mu*_j — the gas is a spectator to Sigma^R.

Units: potentials, masses and T in MeV; densities fm^-3, P and e in MeV/fm^3
(the fm-based units the Bose integrals return).
"""
from eos.general.bose_integrals import solve_bose_jel

#: PDG masses [MeV].
M_PI, M_K, M_ETA, M_ETAP = 140.0, 494.0, 547.0, 958.0
M_KSTAR, M_RHO, M_OMEGA, M_PHI = 892.0, 771.0, 782.0, 1020.0


def _bose(mu_eff, T, m, g):
    """(n, P, e, s) fm-based; antiparticles are separate species here."""
    n, P, e, s, _ = solve_bose_jel(mu_eff, T, m, g, include_antiparticles=False)
    return n, P, e, s


def meson_families(mu_pi_plus, mu_K_plus, mu_K0,
                   include_pseudoscalars=False, include_thermal_vectors=False):
    """(mu_eff, mass, Q, S, g) per thermal meson species.

    Antiparticles are listed explicitly with conjugated potentials; the
    strangeless neutral mesons sit at mu* = 0. The vector nonet reuses the
    same three potentials — rho+ with the pion's, K*+ / K*0 with the kaons' —
    because the shift depends on the quark content, which vector and
    pseudoscalar partners share.
    """
    families = []
    if include_pseudoscalars:
        families += [
            (mu_pi_plus, M_PI, +1, 0, 1.0), (-mu_pi_plus, M_PI, -1, 0, 1.0),
            (0.0, M_PI, 0, 0, 1.0),                          # pi0
            (mu_K_plus, M_K, +1, -1, 1.0), (-mu_K_plus, M_K, -1, +1, 1.0),
            (mu_K0, M_K, 0, -1, 1.0), (-mu_K0, M_K, 0, +1, 1.0),
            (0.0, M_ETA, 0, 0, 1.0), (0.0, M_ETAP, 0, 0, 1.0),
        ]
    if include_thermal_vectors:
        families += [
            (mu_pi_plus, M_RHO, +1, 0, 3.0), (-mu_pi_plus, M_RHO, -1, 0, 3.0),
            (0.0, M_RHO, 0, 0, 3.0), (0.0, M_OMEGA, 0, 0, 3.0),
            (mu_K_plus, M_KSTAR, +1, -1, 3.0), (-mu_K_plus, M_KSTAR, -1, +1, 3.0),
            (mu_K0, M_KSTAR, 0, -1, 3.0), (-mu_K0, M_KSTAR, 0, +1, 3.0),
            (0.0, M_PHI, 0, 0, 3.0),
        ]
    return families


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
    for mu_eff, m, Q, S, g in meson_families(
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
    densities n_C and n_S, and mu_dot_n = sum_j mu*_j n_j for the
    Hugenholtz–Van Hove bookkeeping — the Bose gas obeys
    e + P = T s + mu* n per species.
    """
    if T <= 0.0 or not (include_pseudoscalars or include_thermal_vectors):
        return dict(P=0.0, e=0.0, s=0.0, n_C=0.0, n_S=0.0, mu_dot_n=0.0)
    P = e = s = n_C = n_S = mudotn = 0.0
    for mu_eff, m, Q, S, g in meson_families(
            mu_pi_plus, mu_K_plus, mu_K0,
            include_pseudoscalars, include_thermal_vectors):
        n, P_j, e_j, s_j = _bose(mu_eff, T, m, g)
        P += P_j
        e += e_j
        s += s_j
        n_C += Q * n
        n_S += S * n
        mudotn += mu_eff * n
    return dict(P=P, e=e, s=s, n_C=n_C, n_S=n_S, mu_dot_n=mudotn)
