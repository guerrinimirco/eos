"""Quantities computed FROM the state: DID's kernels, fields and self-energies.

Nothing here knows which equilibrium mode it is in. Given the mean fields, the
isospin asymmetry, the chemical potentials and the temperature, this module
returns densities, energy, pressure, entropy and the sums over them; which of
those a mode holds fixed, and what closes the system, is `solver.py`. Grep
this file for `Y_C`, `neutral` or `trapped` and find nothing -- that is the
boundary, stated as a test.

WHAT DEFINES A DID STATE is larger than in an ordinary DD-RMF, and this is the
one thing to understand before reading further. Its couplings depend on the
isospin asymmetry as well as the density,

    g_Mi = g_Mi(n_B, beta),      beta = sum_i tau_3i n_i / n_B,

so beta enters the couplings, the couplings set the effective masses and
potentials, and those set the densities beta is made of. The same is true of
the second rearrangement self-energy

    Sigma^t = (1/n_B) sum_i [-dg_sigma i/dbeta sigma n^s_i
                             + dg_omega i/dbeta omega n_i
                             + dg_phi i/dbeta phi n_i
                             + dg_rho i/dbeta tau_3i rho n_i],

which shifts every effective potential by (tau_3i - beta) Sigma^t. Neither can
be evaluated from the fields alone. So the state carries SEVEN numbers,

    sigma, omega, rho, phi, beta, Sigma^t, and the coupling density n_B,

and beta and Sigma^t are solved for exactly as the mean fields are, each with
its defining equation as a residual row. Treating either as a quantity to be
recomputed inside an evaluation would make the residual implicit in itself.

The rearrangement terms enter mu and P, never eps (CLAUDE.md section 8). The
Sigma^t term has one further property worth stating because it is a strong
check: it cancels identically out of every SUM, since
sum_i (tau_3i - beta) n_i = n_B beta - beta n_B = 0. It shifts the individual
mu_i -- which is the whole point, it is what splits the Sigma and Xi
single-particle potentials in neutron matter -- and leaves P, eps and
sum_i mu_i n_i untouched.

Reference: Frohaug, Maslov, Dexheimer et al., arXiv:2511.15646, Eqs. (3),
(7)-(17) and Appendix A.

Units are fm-based throughout (densities fm^-3, P and eps MeV/fm^3, T and
potentials MeV), as on every public boundary in this repository. The single
exception is the meson field equations, where a density must be expressed in
MeV^3 to divide by a squared meson mass; hc3 appears there and nowhere else.
"""
from typing import NamedTuple

import numpy as np

from eos.general.fermi_integrals import solve_fermi_jel
from eos.general.physics_constants import hc3
from eos.general.state import PhaseThermo
from eos.general import thermal_mesons as _gas
from eos.general.physics_constants import hc as HBARC
from eos.did.parameters import MULTIPLET_OF, tau3
from eos.did.species import active_baryons

#: Degeneracy of one neutrino flavour (one helicity state); antineutrinos come
#: from the antiparticle branch of the integral.
G_NU = 1.0

#: Number of neutrino flavours carried as a mu = 0 thermal gas when the
#: composition tracks none of them (`SpeciesFlags.thermal_neutrinos`).
N_NEUTRINO_FLAVOURS = 3.0


# =============================================================================
# ONE SPECIES: the ideal Fermi gas
# =============================================================================

def kinetic_thermo(mu_eff, m_eff, g, T=0.0):
    """(n, P, eps, s, n_s) of one fermion species, antiparticles included.

    `mu_eff` is the effective (kinetic) potential nu_i [MeV], `m_eff` the
    Dirac effective mass m*_i [MeV], `g` the spin degeneracy, `T` the
    temperature [MeV]. Everything is fm-based: n and n_s in fm^-3, P and eps
    in MeV/fm^3, s in fm^-3.

    The integrals themselves are `eos.general.fermi_integrals` (CLAUDE.md
    section 7): the JEL approximation at T > 0 and the exact closed forms
    below T = 1e-4 MeV, chosen there rather than here. The scalar density
    comes back through the trace identity n_s = (eps - 3P)/m*.
    """
    if m_eff <= 0.0:
        raise ValueError(f"effective mass {m_eff} MeV is not positive: the "
                         f"state is outside the model's domain")
    n, P, eps, s, n_s = solve_fermi_jel(mu_eff, T, m_eff, g)
    return float(n), float(P), float(eps), float(s), float(n_s)


# =============================================================================
# THE ACTIVE SPECIES
# =============================================================================

class Species(NamedTuple):
    """The static data of one baryon: what does not depend on the state."""
    name: str
    mass: float          # vacuum mass [MeV] (PDG, via eos.general.particles)
    charge: float        # electric charge Q_i
    strangeness: float   # S_i, +1 per s quark (CLAUDE.md section 2)
    tau3: float          # tau_3 = 2 I_3, +/-1 for the nucleons
    g_degen: float
    multiplet: str       # which coupling multiplet it belongs to


def species_table(flags):
    """The active baryons as `Species` rows, in a fixed order."""
    return tuple(Species(name=b.name, mass=b.mass, charge=b.charge,
                         strangeness=b.strangeness, tau3=tau3(b),
                         g_degen=b.g_degen, multiplet=MULTIPLET_OF[b.name])
                 for b in active_baryons(flags))


class Fields(NamedTuple):
    """The model's own self-consistent variables (see the module docstring).

    `n_B` is the density the couplings are evaluated at. In a mode it is the
    density that was asked for and is not iterated on; for one phase of a
    mixture only the average density is prescribed, so there it is an unknown
    like the rest.
    """
    sigma: float         # MeV
    omega: float         # MeV
    rho: float           # MeV
    phi: float           # MeV
    beta: float          # dimensionless
    Sigma_t: float       # MeV
    n_B: float           # fm^-3


# =============================================================================
# THE STATE EVALUATED: one pass over the species
# =============================================================================

class Matter(NamedTuple):
    """Everything one evaluation of the baryon sector produces.

    Both the residual and the assembly read this, so the equations and the
    reported state cannot drift apart: they are the same numbers.
    """
    densities: dict      # {name: n_i} [fm^-3]
    scalar_densities: dict
    mu_eff_i: dict       # nu_i [MeV]
    m_eff_i: dict        # m*_i [MeV]
    n_B: float           # fm^-3, summed (not the coupling density)
    n_C: float           # non-leptonic charge, baryons only
    n_S: float           # strangeness, +1 per s quark, baryons only
    n_3: float           # sum_i tau_3i n_i [fm^-3]
    P: float             # MeV/fm^3, kinetic only
    eps: float
    s: float             # fm^-3
    src_sigma: float     # sum_i g_sigma i n^s_i [fm^-3]
    src_omega: float     # sum_i g_omega i n_i
    src_rho: float       # sum_i g_rho i tau_3i n_i
    src_phi: float       # sum_i g_phi i n_i
    Sigma_r: float       # MeV, the density rearrangement self-energy
    Sigma_t: float       # MeV, the isospin rearrangement self-energy


def effective_masses(specs, couplings, sigma):
    """m*_i = m_i - g_sigma i sigma, per species [MeV] (paper Eq. 8)."""
    return {sp.name: sp.mass - couplings[("sigma", sp.multiplet)][0] * sigma
            for sp in specs}


def effective_potentials(specs, couplings, fields, mu_tilde_B, mu_C, mu_S):
    """nu_i = mu_i - Sigma^v_i, per species [MeV] (paper Eq. 9, rearranged).

        nu_i = mu_tilde_B + Q_i mu_C + S_i mu_S
               - g_omega i omega - g_phi i phi - g_rho i tau_3i rho
               - (tau_3i - beta) Sigma^t

    `mu_tilde_B = mu_B - Sigma^r` is the KINETIC baryon potential: the density
    rearrangement term is common to every species, so carrying it outside the
    iteration keeps its density circularity out of the solve and makes the
    unknowns vary smoothly along a density sweep (CLAUDE.md section 2). The
    isospin rearrangement cannot be absorbed the same way -- it is weighted by
    (tau_3i - beta) and is therefore species-dependent -- so it stays here,
    with Sigma^t itself an unknown of the state.
    """
    out = {}
    for sp in specs:
        g_omega = couplings[("omega", sp.multiplet)][0]
        g_phi = couplings[("phi", sp.multiplet)][0]
        g_rho = couplings[("rho", sp.multiplet)][0]
        out[sp.name] = (mu_tilde_B + sp.charge * mu_C + sp.strangeness * mu_S
                        - g_omega * fields.omega
                        - g_phi * fields.phi
                        - g_rho * sp.tau3 * fields.rho
                        - (sp.tau3 - fields.beta) * fields.Sigma_t)
    return out


def mean_fields(par, sources):
    """(sigma, omega, rho, phi) implied by their sources (paper Eq. 3).

        sigma = sum_i g_sigma i n^s_i / m_sigma^2,   and likewise for the
        vectors with n_i (and tau_3i n_i for the rho).

    `sources` are the four sums in fm^-3; hc3 converts them to MeV^3 so the
    division by a squared meson mass gives MeV. This is the only place in the
    package where a natural-unit density appears.
    """
    src_sigma, src_omega, src_rho, src_phi = sources
    return (src_sigma * hc3 / par.m_sigma ** 2,
            src_omega * hc3 / par.m_omega ** 2,
            src_rho * hc3 / par.m_rho ** 2,
            src_phi * hc3 / par.m_phi ** 2)


def baryon_kinetics(par, specs, fields, mu_tilde_B, mu_C, mu_S, T=0.0):
    """One pass over the baryons at the given state -> `Matter`.

    The couplings are evaluated at the state's (n_B, beta) -- the density the
    caller pinned and the asymmetry the solve carries -- and every sum the
    model needs is accumulated in the same loop, including both rearrangement
    self-energies (paper Eqs. 10 and 11).

    Raises ValueError where an effective mass has collapsed, which is the
    domain boundary rather than a failure of the solve; the callers turn it
    into a residual penalty or a non-convergence status.
    """
    couplings = par.couplings_at(fields.n_B, fields.beta)
    m_eff_i = effective_masses(specs, couplings, fields.sigma)
    mu_eff_i = effective_potentials(specs, couplings, fields, mu_tilde_B,
                                    mu_C, mu_S)

    densities, scalar_densities = {}, {}
    n_B = n_C = n_S = n_3 = 0.0
    P = eps = s = 0.0
    src_sigma = src_omega = src_rho = src_phi = 0.0
    Sigma_r = Sigma_t = 0.0

    for sp in specs:
        n, P_i, eps_i, s_i, n_s = kinetic_thermo(
            mu_eff_i[sp.name], m_eff_i[sp.name], sp.g_degen, T)
        densities[sp.name] = n
        scalar_densities[sp.name] = n_s
        n_B += n
        n_C += sp.charge * n
        n_S += sp.strangeness * n
        n_3 += sp.tau3 * n
        P += P_i
        eps += eps_i
        s += s_i

        g_sigma, dsigma_dn, dsigma_dbeta = couplings[("sigma", sp.multiplet)]
        g_omega, domega_dn, domega_dbeta = couplings[("omega", sp.multiplet)]
        g_rho, drho_dn, drho_dbeta = couplings[("rho", sp.multiplet)]
        g_phi, dphi_dn, dphi_dbeta = couplings[("phi", sp.multiplet)]

        src_sigma += g_sigma * n_s
        src_omega += g_omega * n
        src_rho += g_rho * sp.tau3 * n
        src_phi += g_phi * n

        Sigma_r += (-dsigma_dn * fields.sigma * n_s
                    + domega_dn * fields.omega * n
                    + dphi_dn * fields.phi * n
                    + drho_dn * sp.tau3 * fields.rho * n)
        Sigma_t += (-dsigma_dbeta * fields.sigma * n_s
                    + domega_dbeta * fields.omega * n
                    + dphi_dbeta * fields.phi * n
                    + drho_dbeta * sp.tau3 * fields.rho * n)

    # Sigma^t carries a 1/n_B prefactor. The tanh(x/e) factor in the isospin
    # blend makes every dg/dbeta vanish as n_B -> 0 faster than n_B does, so
    # the limit is zero; the guard is for a trial state at exactly zero
    # density, not for the physics.
    Sigma_t = Sigma_t / fields.n_B if fields.n_B > 0.0 else 0.0

    return Matter(
        densities=densities, scalar_densities=scalar_densities,
        mu_eff_i=mu_eff_i, m_eff_i=m_eff_i,
        n_B=n_B, n_C=n_C, n_S=n_S, n_3=n_3,
        P=P, eps=eps, s=s,
        src_sigma=src_sigma, src_omega=src_omega, src_rho=src_rho,
        src_phi=src_phi, Sigma_r=Sigma_r, Sigma_t=Sigma_t)


# =============================================================================
# THE THERMAL MESON GAS: DID's effective potentials for it
# =============================================================================
# The gas itself -- species, quantum numbers, Bose sums -- is
# `eos.general.thermal_mesons`, one implementation for every model. What is
# DID's is the arithmetic of the three effective potentials, with its
# density- and isospin-dependent couplings in place of constant ones. No
# rearrangement term enters any mu*_j: the gas is a spectator to Sigma^r and
# Sigma^t, carrying no baryon number and no tau_3-weighted source.

def meson_potentials(couplings, fields, mu_C, mu_S):
    """(mu*_pi+, mu*_K+, mu*_K0) [MeV] on DID's mean fields.

        mu*_pi+ = mu_C - g_rhoN rho
        mu*_K+  = mu_C - mu_S - (g_omegaN - g_omegaLambda) omega
                              - 1/2 g_rhoN rho
        mu*_K0  =      - mu_S - (g_omegaN - g_omegaLambda) omega
                              + 1/2 g_rhoN rho

    The kaon's omega shift is (g_omegaN - g_omegaLambda): under the additive
    quark picture the kaon couples through its one light quark, and the
    Lambda coupling supplies the strange-sector piece. This is the same
    arithmetic `eos.dd2` uses, so the two models' gases differ only through
    their couplings. The phi field is NOT included in the kaon shift -- see
    docs/DEFERRED.md, where DID's nonzero g_phiN makes that a real omission
    rather than an inherited zero.
    """
    g_rho_N = couplings[("rho", "N")][0]
    g_omega_N = couplings[("omega", "N")][0]
    g_omega_L = couplings[("omega", "Lambda")][0]
    d_omega = (g_omega_N - g_omega_L) * fields.omega
    return (mu_C - g_rho_N * fields.rho,
            mu_C - mu_S - d_omega - 0.5 * g_rho_N * fields.rho,
            -mu_S - d_omega + 0.5 * g_rho_N * fields.rho)


def thermal_meson_thermo(par, fields, mu_C, mu_S, T, thermal_mesons=False):
    """The gas at DID's potentials, or an all-zero block when it is off.

    Returns the dictionary of `eos.general.thermal_mesons.thermal_meson_thermo`
    (P, e, s, n_C, n_S, mu_dot_n, densities, condensation) in fm-based units,
    every key present either way so no caller has to ask whether the gas is on.

    The vector nonet is not wired: DID's `SpeciesFlags` carries one
    `thermal_mesons` flag, which is the pseudoscalar gas.
    """
    if T <= 0.0 or not thermal_mesons:
        return dict(P=0.0, e=0.0, s=0.0, n_C=0.0, n_S=0.0, mu_dot_n=0.0,
                    densities={}, condensation=0.0)
    couplings = par.couplings_at(fields.n_B, fields.beta)
    mu_pi, mu_Kp, mu_K0 = meson_potentials(couplings, fields, mu_C, mu_S)
    return _gas.thermal_meson_thermo(mu_pi, mu_Kp, mu_K0, T,
                                     include_pseudoscalars=True,
                                     include_thermal_vectors=False)


# =============================================================================
# ASSEMBLY: the state, matter only
# =============================================================================

def field_eps_P(par, fields):
    """(eps, P) of the meson mean fields [MeV/fm^3] (paper Eqs. 7 and 15).

    The scalar enters P with a minus sign, the vectors with a plus; both enter
    eps with a plus. Written in MeV^4 and converted once.
    """
    s2 = par.m_sigma ** 2 * fields.sigma ** 2
    w2 = par.m_omega ** 2 * fields.omega ** 2
    r2 = par.m_rho ** 2 * fields.rho ** 2
    p2 = par.m_phi ** 2 * fields.phi ** 2
    return 0.5 * (s2 + w2 + r2 + p2) / hc3, 0.5 * (-s2 + w2 + r2 + p2) / hc3


def thermo_from_fields(par, flags, fields, mu_tilde_B, mu_C, mu_S=0.0, T=0.0,
                       matter=None):
    """The matter block at given potentials and fields, as a `PhaseThermo`.

    MATTER ONLY: baryons plus any thermal meson gas, no leptons and no
    photons. Those are shared by the whole system and, in a mixed phase, may
    be distributed differently from the matter, so `solver.py` and
    `eos.mixed` add them.

    The gas is listed with the baryons in `densities`, which is what makes the
    returned n_C and n_S the TOTAL non-leptonic charge and strangeness the
    fixed-Y_C, fixed-Y_S and neutrality conditions are stated in terms of
    (CLAUDE.md section 2).

    Pass `matter` to reuse an evaluation the caller already has; otherwise one
    is made here. The pressure carries n_B Sigma^r and the energy does not
    (section 8); the Sigma^t term of the thermodynamic potential,
    -sum_i (tau_3i - beta) n_i Sigma^t, is identically zero at the solved beta
    and is therefore not written -- `verify/` checks that it is.
    """
    specs = species_table(flags)
    if matter is None:
        matter = baryon_kinetics(par, specs, fields, mu_tilde_B, mu_C, mu_S, T)
    eps_fields, P_fields = field_eps_P(par, fields)
    gas = thermal_meson_thermo(par, fields, mu_C, mu_S, T,
                               thermal_mesons=flags.thermal_mesons)

    mu_B = mu_tilde_B + matter.Sigma_r
    mu_dot_n = (mu_B * matter.n_B + mu_C * matter.n_C + mu_S * matter.n_S
                + gas["mu_dot_n"])
    return PhaseThermo.assemble(
        T=T, mu_B=mu_B, mu_C=mu_C, mu_S=mu_S,
        fields={"sigma": fields.sigma, "omega": fields.omega,
                "rho": fields.rho, "phi": fields.phi,
                "beta": fields.beta, "Sigma_t": fields.Sigma_t},
        densities=dict(matter.densities), mu_eff_i=dict(matter.mu_eff_i),
        m_eff_i=dict(matter.m_eff_i),
        P=matter.P + P_fields + matter.n_B * matter.Sigma_r + gas["P"],
        eps=matter.eps + eps_fields + gas["e"],
        s=matter.s + gas["s"],
        mu_dot_n=mu_dot_n,
        Sigma_R=matter.Sigma_r,
        extra_charges=(0.0, gas["n_C"], gas["n_S"]),
        condensation=gas["condensation"])


def single_particle_potential(couplings, fields, multiplet, tau_3, Sigma_r):
    """U_i = Sigma^v_i - Sigma^s_i, the energy gained by adding a baryon at
    zero three-momentum to the medium (paper Eq. 12):

        U_i = -g_sigma i sigma + g_omega i omega + g_phi i phi
              + g_rho i tau_3i rho + Sigma^r + (tau_3i - beta) Sigma^t.

    `couplings` is what `Parameters.couplings_at(n_B, beta)` returned for the
    MEDIUM, and `multiplet` and `tau_3` identify the baryon being added. It is
    what the hyperon couplings were fitted to, in isospin-symmetric and in
    neutron matter, so it is the model's own calibration observable rather
    than a derived diagnostic. The species need not be present in the medium:
    U_i is evaluated for a test particle at the medium's fields, which is what
    makes U_Y at n_0 meaningful before any hyperon has appeared.
    """
    g_sigma = couplings[("sigma", multiplet)][0]
    g_omega = couplings[("omega", multiplet)][0]
    g_phi = couplings[("phi", multiplet)][0]
    g_rho = couplings[("rho", multiplet)][0]
    return (-g_sigma * fields.sigma + g_omega * fields.omega
            + g_phi * fields.phi + g_rho * tau_3 * fields.rho
            + Sigma_r + (tau_3 - fields.beta) * fields.Sigma_t)


# =============================================================================
# SELF-CONSISTENCY: DID's own state at given charge potentials
# =============================================================================
# The phase-adapter surface of CLAUDE.md section 5, seen from inside the model.
# `solver.py` closes a MODE -- a density and a set of conserved-charge
# conditions; this closes the model's own self-consistency at given POTENTIALS,
# with the density itself an unknown, which is what one phase of a mixture
# needs (only the average density is prescribed there, never each phase's).
# Nothing here knows about neutrality, a held fraction or a lepton.

#: Post-solve gate for the phase-internal solve, matching what the equilibrium
#: solvers accept.
RESIDUAL_TOL = 1.0e-10


def _pack(fields):
    return [fields.sigma, fields.omega, fields.rho, fields.phi,
            fields.beta, fields.Sigma_t, fields.n_B]


def _unpack_fields(x):
    return Fields(sigma=x[0], omega=x[1], rho=x[2], phi=x[3], beta=x[4],
                  Sigma_t=x[5], n_B=x[6])


def self_consistency_residual(x, par, specs, mu_tilde_B, mu_C, mu_S, T):
    """The model's own equations at fixed potentials, density included.

    Unknowns x = [sigma, omega, rho, phi, beta, Sigma^t, n_B]; rows are the
    four field equations, the definition of beta, the definition of Sigma^t
    and the density self-consistency n_B = sum_i n_i. There is no charge,
    strangeness or neutrality row: the potentials are inputs. That is what
    makes this thermodynamics rather than a mode.
    """
    fields = _unpack_fields(x)
    if fields.n_B <= 0.0:
        return [1.0e6] * len(x)
    try:
        matter = baryon_kinetics(par, specs, fields, mu_tilde_B, mu_C, mu_S, T)
    except (ValueError, FloatingPointError):
        return [1.0e6] * len(x)
    implied = mean_fields(par, (matter.src_sigma, matter.src_omega,
                                matter.src_rho, matter.src_phi))
    rows = [(fields.sigma - implied[0]) / 30.0,
            (fields.omega - implied[1]) / 30.0,
            (fields.rho - implied[2]) / 30.0,
            (fields.phi - implied[3]) / 30.0,
            fields.beta - matter.n_3 / fields.n_B,
            (fields.Sigma_t - matter.Sigma_t) / 30.0,
            matter.n_B / fields.n_B - 1.0]
    return rows if np.isfinite(rows).all() else [1.0e6] * len(x)


def field_estimate(par, flags, n_B, beta=0.0):
    """The mean fields of a free nucleon gas at (n_B, beta), as a `Fields`.

    Sources evaluated on a degenerate free gas of the given composition, with
    the scalar field iterated three times against its own effective mass. It
    is right to a few MeV over the whole density range, which is what a Newton
    step needs, and -- unlike zero fields -- it puts the vector potentials at
    the right SIZE. That matters: at fixed potentials the kinetic potential is
    mu - g_omega omega, so starting the vectors at zero leaves nu ~ 1500 MeV
    and a density of 14 fm^-3 on the first sweep, from which nothing recovers.

    Sigma^t is estimated as zero: it is a few MeV, and it vanishes with beta.
    """
    couplings = par.couplings_at(n_B, beta)
    g_sigma = couplings[("sigma", "N")][0]
    baryons = active_baryons(flags)
    m_N = 0.5 * (baryons[0].mass + baryons[1].mass)
    n_p, n_n = 0.5 * (1.0 + beta) * n_B, 0.5 * (1.0 - beta) * n_B
    kF = [(3.0 * np.pi ** 2 * max(n, 1.0e-12)) ** (1.0 / 3.0) * HBARC
          for n in (n_p, n_n)]

    m_eff, n_s = m_N, n_B
    for _sweep in range(3):
        n_s = sum(n * m_eff / np.sqrt(k ** 2 + m_eff ** 2)
                  for n, k in zip((n_p, n_n), kF))
        sigma = mean_fields(par, (g_sigma * n_s, 0.0, 0.0, 0.0))[0]
        m_eff = max(0.1 * m_N, m_N - g_sigma * sigma)
    sigma, omega, rho, phi = mean_fields(
        par, (g_sigma * n_s,
              couplings[("omega", "N")][0] * n_B,
              couplings[("rho", "N")][0] * (n_p - n_n),
              couplings[("phi", "N")][0] * n_B))
    return Fields(sigma=sigma, omega=omega, rho=rho, phi=phi, beta=beta,
                  Sigma_t=0.0, n_B=n_B)


def cold_start(par, flags, n_B_guess=0.2, beta=0.0):
    """The starting state a solve at fixed potentials begins from.

    `field_estimate` at the guessed density, and nothing more. A damped Picard
    sweep on top of it was tried and removed: at fixed potentials the density
    is the runaway direction of the fixed-point map -- nu_i rises when the
    vector fields fall, which raises the density, which raises the fields --
    and the iteration diverges above about 3 n_0 from a seed Newton solves
    from immediately.

    A PURE FUNCTION of its arguments, and it has to be: `eos.mixed`
    differentiates its residual by finite differences, so a seed that
    remembered the previous trial point would make the phase block depend on
    the path taken to reach it and corrupt the Jacobian.
    """
    return _pack(field_estimate(par, flags, n_B_guess, beta))


#: The asymmetries a cold start is tried at, in order. Symmetric first because
#: it is right for a mixed phase's hadronic component at high density; the
#: neutron-rich one catches beta-equilibrium states, where the composition the
#: seed assumes is far from symmetric.
COLD_START_BETAS = (0.0, -0.9)


def thermo_from_mu(par, flags, mu_tilde_B, mu_C, mu_S=0.0, T=0.0,
                   n_B_guess=0.2, x0=None, x0_fallback=None,
                   return_state=False):
    """DID's matter at fixed conserved-charge potentials, state solved.

    `mu_tilde_B = mu_B - Sigma^r` is the KINETIC baryon potential: Sigma^r is
    a function of the density, which is itself an unknown here, so carrying
    mu_B would put that circularity into the iteration. The physical mu_B is
    restored at assembly.

    `x0` is a starting vector in the layout of `self_consistency_residual`,
    and must be a DETERMINISTIC function of the caller's inputs for the reason
    `cold_start` gives. `x0_fallback` is a second guess (or a callable
    producing one) tried when the first misses -- what makes a warm-started
    mixed-phase sweep robust across a composition change.

    Returns a `PhaseThermo`, or (PhaseThermo, {x_phase}) with `return_state`,
    so a caller can feed the converged vector back as the next `x0`.
    """
    from scipy.optimize import root

    specs = species_table(flags)
    args = (par, specs, mu_tilde_B, mu_C, mu_S, T)

    def guesses():
        if x0 is not None:
            yield list(x0)
        if x0_fallback is not None:
            yield list(x0_fallback() if callable(x0_fallback) else x0_fallback)
        for beta in COLD_START_BETAS:
            yield cold_start(par, flags, n_B_guess, beta)

    best, best_err = None, np.inf
    for guess in guesses():
        sol = root(self_consistency_residual, guess, args=args, method="hybr",
                   tol=1.0e-13)
        err = max(abs(r) for r in self_consistency_residual(sol.x, *args))
        if err < best_err:
            best, best_err = sol.x, err
        if best_err <= RESIDUAL_TOL:
            break
    if best_err > RESIDUAL_TOL:
        raise RuntimeError(
            f"DID self-consistency failed at mu_tilde_B={mu_tilde_B}, "
            f"mu_C={mu_C}, mu_S={mu_S}, T={T} (max residual {best_err:.2e}, "
            f"tol {RESIDUAL_TOL:.0e})")

    fields = _unpack_fields(best)
    state = thermo_from_fields(par, flags, fields, mu_tilde_B, mu_C, mu_S, T)
    return (state, dict(x_phase=list(best))) if return_state else state
