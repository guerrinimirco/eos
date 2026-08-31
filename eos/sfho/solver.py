"""
solver.py
=========
The equilibrium conditions of the SFHo model, and the solve that closes them.

`thermodynamics.py` computes quantities FROM the state; this module FINDS the
state. It is the only file in `eos/sfho` that knows what a mode is.

There is ONE system here, not one per mode. A solve needs three families of
equation and only the last two depend on the mode:

    field equations      the four meson mean fields, ALWAYS all of them -- a
                         mean field does not know what is being held fixed
    charge conditions    n_C = Y_C n_B, n_S = Y_S n_B, (n_e + n_nue) = Y_Le n_B
    equilibrium relations  mu_S = 0, mu_C + mu_e = mu_nue

and those last two are paired: for each conserved charge, either its fraction
is imposed and its potential is an unknown, or its potential is set by a
relation and the fraction comes out. That single binary choice per charge is
the whole content of a mode, and it is declared once for the repository in
`eos.general.modes`. `residual()` below reads a `ModeSpec` and assembles the
rows; the named modes at the bottom of the file are configurations of it, not
separate solvers.

The unknown vector is

    x = [sigma, omega, rho, phi, mu_B, mu_C, (mu_S), (mu_nue), (T)]

with mu_S present iff the mode holds strangeness, mu_nue iff it traps the
electron family, and T iff an entropy per baryon is imposed in its place
(CLAUDE.md section 3: wherever a temperature axis is accepted, S/A is accepted
instead). mu_C is an unknown in EVERY mode -- what the mode changes is the row
that closes it, electric neutrality or the held Y_C.

Sign conventions are the repository's (CLAUDE.md section 2): C is the charge of
strongly-interacting matter only, S = +1 per s quark, and mu_C = mu_p - mu_n,
so beta equilibrium reads mu_C + mu_e = mu_nue.

Usage:
    from eos.sfho.solver import solve_beta_eq_neutrinoless, BARYONS_N
    from eos.sfho.parameters import Parameters

    params = Parameters.named("SFHo_2fam_phi")
    result = solve_beta_eq_neutrinoless(n_B=0.16, T=10.0, params=params,
                                        particles=BARYONS_N)
    print(f"P = {result.P} MeV/fm^3")

References:
- Fortin, Oertel, Providencia, PASA 35 (2018) e044
- Steiner, Hempel, Fischer, ApJ 774 (2013) 17

see also:
- Guerrini, PhD Thesis (2026)
"""
import numpy as np
from dataclasses import replace
from typing import Optional, List, NamedTuple
from scipy.optimize import root

from eos.general.particles import (
    Particle, Proton, Neutron,
    Lambda, SigmaP, Sigma0, SigmaM, Xi0, XiM,
    DeltaPP, DeltaP, Delta0, DeltaM
)
from eos.general import modes
from eos.general.state import EoSPoint, LeptonThermo
from eos.general.modes import (
    ModeSpec, electron_potential, strangeness_potential
)
from eos.sfho.parameters import Parameters
from eos.sfho.species import SpeciesFlags, active_baryons, check_couplings
from eos.sfho.thermodynamics import (
    baryon_thermo, field_residuals, species_constants, thermal_meson_thermo,
    thermo_from_fields,
)
from eos.general.thermodynamics_leptons import (
    electron_thermo, photon_thermo, neutrino_thermo, electron_thermo_from_density
)

try:
    from eos.sfho.backends.jacobian import residual_jacobian
except ImportError:
    # `backends/` is optional: CLAUDE.md section 5 defines it by the property
    # that deleting it changes no number, only the speed. Without it MINPACK
    # builds a forward-difference Jacobian instead.
    residual_jacobian = None


# =============================================================================
# PARTICLE LISTS
# =============================================================================
NUCLEONS = [Proton, Neutron]
HYPERONS = [Lambda, SigmaP, Sigma0, SigmaM, Xi0, XiM]
DELTAS = [DeltaPP, DeltaP, Delta0, DeltaM]

BARYONS_N = NUCLEONS
BARYONS_NY = NUCLEONS + HYPERONS
BARYONS_NYD = NUCLEONS + HYPERONS + DELTAS

#: Field scale used to make the four field-equation residuals dimensionless.
#: A field equation is m^2 phi = source in MeV^3, so dividing by m^2 times a
#: typical field puts every row near unity; sigma at saturation is about 30 MeV.
FIELD_SCALE = 30.0

#: Neutrino flavours in nature. `thermal_neutrinos` carries the ones the
#: composition does NOT track, as mu = 0 gases (CLAUDE.md section 4): all
#: three where the electron neutrino is free-streaming, two where a trapped
#: mode already counts it at its own potential.
N_NEUTRINO_FLAVOURS = 3.0


# =============================================================================
# THE RECORD ONE SOLVE RETURNS
# =============================================================================
# =============================================================================
# INITIAL GUESSES
# =============================================================================
def _bulk_guess(n_B: float):
    """(sigma, omega, mu_B) at n_B, the part every mode's guess shares.

    Both fields grow roughly linearly with density and saturate; mu_B sits
    near the nucleon mass and rises. Fitted to converged solutions, not
    derived.
    """
    n_sat = 0.158
    ratio = n_B / n_sat

    sigma = min(30.0 * ratio, 100.0)
    omega = min(19.0 * ratio, 80.0)

    if ratio < 0.5:
        mu_B = 940.0 + 5.0 * ratio
    elif ratio < 2.0:
        mu_B = 942.0 + 40.0 * (ratio - 0.5)
    else:
        mu_B = 1002.0 + 80.0 * (ratio - 2.0)

    return sigma, omega, mu_B


def get_default_guess_beta_eq(
    n_B: float, T: float, params: Parameters
) -> np.ndarray:
    """
    Generate initial guess for beta equilibrium: [σ, ω, ρ, φ, μ_B, μ_C].

    μ_C is negative and grows in magnitude with density: about -20 MeV at low
    n_B and -50 MeV at saturation, values read off converged tables.
    """
    sigma, omega, mu_B = _bulk_guess(n_B)
    ratio = n_B / 0.158

    if ratio < 0.1:
        mu_C = -20.0 - 100.0 * ratio  # -20 to -30 MeV
    elif ratio < 0.5:
        mu_C = -30.0 - 40.0 * ratio   # -30 to -50 MeV
    elif ratio < 1.5:
        mu_C = -50.0 - 40.0 * (ratio - 0.5)  # -50 to -90 MeV
    else:
        mu_C = -90.0 - 30.0 * (ratio - 1.5)  # -90 and beyond

    return np.array([sigma, omega, 0.0, 0.0, mu_B, mu_C])


def get_default_guess_fixed_yc(
    n_B: float, Y_C: float, T: float, params: Parameters
) -> np.ndarray:
    """
    Generate initial guess for fixed Y_C: [σ, ω, ρ, φ, μ_B, μ_C].

    ρ and μ_C both scale with the isospin asymmetry: at Y_C = 0.5 matter is
    symmetric and both vanish, at Y_C = 0 it is pure neutron matter.
    """
    sigma, omega, mu_B = _bulk_guess(n_B)
    ratio = n_B / 0.158

    asymmetry = 0.5 - Y_C  # Positive for neutron-rich
    rho = -5.0 * min(ratio, 2.0) * asymmetry / 0.5
    phi = 0.0              # φ = 0 for nucleons only

    base_mu_C = -150.0 * asymmetry  # -75 to 0 for Y_C=0 to 0.5
    density_factor = min(ratio, 2.0) / 1.0
    mu_C = base_mu_C * (0.5 + 0.5 * density_factor)

    return np.array([sigma, omega, rho, phi, mu_B, mu_C])


def get_default_guess_fixed_yc_ys(
    n_B: float, Y_C: float, Y_S: float, T: float, params: Parameters
) -> np.ndarray:
    """
    Generate initial guess for fixed Y_C and Y_S: [σ, ω, ρ, φ, μ_B, μ_C, μ_S].

    Based on analysis of converged solutions at n_B=0.01583:
    - Y_C=0.5, Y_S=0.5-0.9: mu_B~910-920, mu_C~15-35, mu_S~195-210 MeV
    - μ_S scales weakly with Y_S (~2 MeV per 0.1 Y_S)
    """
    guess_yc = get_default_guess_fixed_yc(n_B, Y_C, T, params)

    n_sat = 0.158
    ratio = n_B / n_sat

    # For symmetric matter with strangeness (Y_C ~0.5), use observed pattern
    if Y_C > 0.4 and Y_S > 0.3:
        phi = -0.5 * Y_S

        # At low density: mu_S ~ 195 + 15*Y_S (from converged solutions)
        # mu_C ~ 15 + 25*Y_S (increases with strangeness)
        if ratio < 0.5:
            mu_S = 195.0 + 20.0 * Y_S
            mu_C = 15.0 + 30.0 * Y_S
            mu_B = 920.0 - 10.0 * Y_S  # slight decrease with Y_S
        else:
            # Higher density: adjust
            mu_S = (195.0 + 20.0 * Y_S) * (1.0 - 0.3 * (ratio - 0.5))
            mu_C = (15.0 + 30.0 * Y_S) * (1.0 + 0.5 * ratio)
            mu_B = 920.0 + 50.0 * ratio

        return np.array([guess_yc[0], guess_yc[1], guess_yc[2], phi,
                         mu_B, mu_C, mu_S])

    # Default case for low Y_C or low Y_S
    phi = -0.5 * min(ratio, 5.0) * Y_S

    # Base mu_S depends mainly on density
    if ratio < 0.5:
        mu_S_base = 180.0
    elif ratio < 1.0:
        mu_S_base = 130.0 - 30.0 * (ratio - 0.5)
    elif ratio < 2.0:
        mu_S_base = 105.0 - 80.0 * (ratio - 1.0)
    elif ratio < 4.0:
        mu_S_base = 25.0 - 40.0 * (ratio - 2.0)
    else:
        mu_S_base = -55.0 - 20.0 * (ratio - 4.0)

    # Weak Y_S scaling
    if Y_S > 0:
        mu_S = mu_S_base * (1.0 + 0.1 * np.log10(Y_S / 0.1 + 1))
    else:
        mu_S = 0.0

    return np.array([guess_yc[0], guess_yc[1], guess_yc[2], phi,
                     guess_yc[4], guess_yc[5], mu_S])


def get_default_guess_trapped(
    n_B: float, Y_Le: float, T: float, params: Parameters
) -> np.ndarray:
    """
    Generate initial guess for trapped neutrinos: [σ, ω, ρ, φ, μ_B, μ_C, μ_nue].

    A higher trapped lepton fraction needs a higher neutrino potential.
    """
    guess_beta = get_default_guess_beta_eq(n_B, T, params)
    mu_nue_est = 10.0 + 50.0 * Y_Le
    return np.append(guess_beta, mu_nue_est)


def default_guess(spec: ModeSpec, n_B: float, T: float,
                  params: Parameters, SnB: Optional[float] = None):
    """The cold start for the mode `spec` declares (CLAUDE.md section 13).

    Dispatches to the fitted heuristics above -- they differ because the modes
    differ, not by accident -- and appends the entropy axis's T unknown last,
    in the order `unknown_names` documents.
    """
    if spec.is_fixed("S"):
        x0 = get_default_guess_fixed_yc_ys(
            n_B, spec.targets["Y_C"], spec.targets["Y_S"], T, params)
    elif spec.is_fixed("C"):
        x0 = get_default_guess_fixed_yc(n_B, spec.targets["Y_C"], T, params)
    elif spec.is_fixed("L_e"):
        x0 = get_default_guess_trapped(n_B, spec.targets["Y_Le"], T, params)
    else:
        x0 = get_default_guess_beta_eq(n_B, T, params)
    if SnB is not None:
        x0 = np.append(x0, T)
    return x0


# =============================================================================
# THE SYSTEM
# =============================================================================
class System(NamedTuple):
    """Everything the residual needs beyond the unknown vector.

    `T` is the imposed temperature and is None exactly when `SnB` imposes an
    entropy per baryon instead, in which case T joins the unknowns.
    """
    params: Parameters
    particles: List[Particle]
    spec: ModeSpec
    n_B: float
    T: Optional[float] = None
    SnB: Optional[float] = None
    thermal_mesons: bool = False       # the pi, K, eta Bose gas
    photons: bool = True
    thermal_neutrinos: bool = False    # flavours the composition does not track
    #: Select the analytic-Jacobian backend (CLAUDE.md section 9). It is OFF by
    #: default because it is not faster here: it cuts the residual evaluations
    #: of a warm-started sweep from ~18 to ~11.5 per point, but a T > 0 kinetic
    #: derivative costs four JEL evaluations per species, so the wall clock
    #: moves the wrong way -- 0.49 -> 0.53 ms/pt for nucleons at T = 10 MeV,
    #: 1.50 -> 2.01 ms/pt for the octet with a meson gas at T = 30 MeV. What it
    #: is for is the second-derivative quantities, which need dR/dx itself
    #: rather than a faster root. The finite-difference path stays the
    #: reference, and the two agree to solver tolerance.
    analytic_jac: bool = False
    #: The `species_constants` table of (particles, params). It is part of the
    #: system rather than looked up inside the residual because it does not
    #: move while the system is being solved. `_system` fills it in; None
    #: leaves `baryon_thermo` to build its own, which is what a System
    #: assembled by hand gets.
    constants: Optional[tuple] = None

    @property
    def isentropic(self):
        return self.SnB is not None


def unknown_names(sys: System):
    """The unknown vector, in order. See the module docstring."""
    names = ["sigma", "omega", "rho", "phi", "mu_B", "mu_C"]
    if sys.spec.is_fixed("S"):
        names.append("mu_S")
    if sys.spec.is_fixed("L_e"):
        names.append("mu_nue")
    if sys.isentropic:
        names.append("T")
    return tuple(names)


def _unpack(x, sys: System):
    """Read the unknown vector, filling in what the mode determines instead."""
    sigma, omega, rho, phi, mu_B, mu_C = x[0], x[1], x[2], x[3], x[4], x[5]
    i = 6
    if sys.spec.is_fixed("S"):
        mu_S = x[i]
        i += 1
    else:
        mu_S = strangeness_potential(sys.spec)
    if sys.spec.is_fixed("L_e"):
        mu_nue = x[i]
        i += 1
    else:
        mu_nue = 0.0
    if sys.isentropic:
        # A negative temperature is meaningless and the Fermi integrals do not
        # accept one; the floor keeps a wide step from leaving the domain.
        T = max(x[i], 0.1)
    else:
        T = sys.T
    return sigma, omega, rho, phi, mu_B, mu_C, mu_S, mu_nue, T


class _Leptons(NamedTuple):
    """The leptons that enter the EQUATIONS, at the current unknowns.

    The two blocks stay separate rather than pre-summed: every total in this
    file accumulates matter, then electrons, then neutrinos, in that order, and
    floating-point addition is not associative.
    """
    mu_e: float
    electrons: object            # what electron_thermo returned
    neutrinos: Optional[object]  # trapped nu_e, None where they free-stream

    @property
    def n_nue(self):
        return self.neutrinos.n if self.neutrinos is not None else 0.0

    @property
    def P(self):
        return self.electrons.P + (self.neutrinos.P
                                   if self.neutrinos is not None else 0.0)


def _lepton_sector(spec: ModeSpec, mu_C, mu_nue, T):
    """Leptons at the current potentials, or None when the mode has no row.

    Where C is EQUILIBRATED the beta relation gives mu_e from mu_C and mu_nue,
    the electrons are part of the residual (electric neutrality is the row
    that closes mu_C), and the neutrinos join them once the electron family is
    trapped. Where Y_C is held instead, n_C is already pinned by the charge row
    and the neutralizing electrons follow from it AFTER the solve -- they enter
    no equation, so there is nothing to compute here.
    """
    if spec.is_fixed("C"):
        return None
    mu_e = electron_potential(mu_C, mu_nue)
    e = electron_thermo(mu_e, T, include_antiparticles=True)
    nu = (neutrino_thermo(mu_nue, T, include_antiparticles=True)
          if spec.is_fixed("L_e") else None)
    return _Leptons(mu_e, e, nu)


def residual(x, sys: System):
    """The equations that must vanish, assembled from the mode declaration.

    Rows, in this order:

        4  meson field equations, scaled by m^2 times a typical field so each
           is dimensionless
        1  baryon number, n_B = n_B_target
        1  the charge row: electric neutrality where C is EQUILIBRATED,
           n_C = Y_C n_B where it is FIXED
        1  strangeness, n_S = Y_S n_B                    iff S is FIXED
        1  lepton number, (n_e + n_nue)/n_B = Y_Le       iff L_e is FIXED
        1  entropy per baryon, s/n_B = SnB               iff the entropy axis

    The four field equations are always present: a mean field does not know
    what is being held fixed.
    """
    par, spec = sys.params, sys.spec
    sigma, omega, rho, phi, mu_B, mu_C, mu_S, mu_nue, T = _unpack(x, sys)

    # The baryon kinetics and the gas, but NOT the mean-field pressure and
    # energy: the residual needs the field sources and the conserved charges,
    # and assembling the full state on every iteration would be work thrown
    # away. `assemble` below builds it once, from the converged vector.
    hadron = baryon_thermo(
        T, mu_B, mu_C, mu_S, sigma, omega, rho, phi, sys.particles, par,
        sys.constants)
    n_C = hadron.n_C
    n_S = hadron.n_S
    s_matter = hadron.s_hadrons
    if sys.thermal_mesons:
        # The gas carries charge and strangeness, so it enters the charge rows
        # and not only the entropy one (CLAUDE.md section 2).
        gas = thermal_meson_thermo(T, mu_C, mu_S, omega, rho, par)
        n_C += gas["n_C"]
        n_S += gas["n_S"]
        s_matter += gas["s"]

    res_sigma, res_omega, res_rho, res_phi = field_residuals(
        sigma, omega, rho, phi,
        hadron.src_sigma, hadron.src_omega, hadron.src_rho, hadron.src_phi,
        par)

    rows = [
        res_sigma / (par.m_sigma**2 * FIELD_SCALE),
        res_omega / (par.m_omega**2 * FIELD_SCALE),
        res_rho / (par.m_rho**2 * FIELD_SCALE),
        res_phi / (par.m_phi**2 * FIELD_SCALE),
        hadron.n_B - sys.n_B,
    ]

    lep = _lepton_sector(spec, mu_C, mu_nue, T)
    if spec.is_fixed("C"):
        # n_C is the TOTAL non-leptonic charge, thermal meson gas included.
        rows.append(n_C - spec.targets["Y_C"] * sys.n_B)
    else:
        rows.append(n_C - lep.electrons.n)
    if spec.is_fixed("S"):
        rows.append(n_S - spec.targets["Y_S"] * sys.n_B)
    if spec.is_fixed("L_e"):
        rows.append((lep.electrons.n + lep.n_nue) / sys.n_B
                    - spec.targets["Y_Le"])
    if sys.isentropic:
        s_total = s_matter
        if lep is not None:
            s_total += lep.electrons.s
            if lep.neutrinos is not None:
                s_total += lep.neutrinos.s
        if sys.photons:
            s_total += photon_thermo(T).s
        rows.append(s_total / sys.n_B - sys.SnB)
    return rows


def assemble(x, sys: System, x0=None) -> EoSPoint:
    """The full thermodynamic state from a converged unknown vector.

    Everything a mode can report is filled in here once. Sectors the mode does
    not carry stay at their zero defaults, which is what they physically are:
    no trapped neutrinos means n_nu = 0, and leptonless fixed-Y_C matter is
    genuinely charged, with no electrons and no lepton pressure.
    """
    par, spec = sys.params, sys.spec
    sigma, omega, rho, phi, mu_B, mu_C, mu_S, mu_nue, T = _unpack(x, sys)

    matter = thermo_from_fields(
        mu_B, mu_C, mu_S, sigma, omega, rho, phi, T, sys.particles, par,
        include_pseudoscalar_mesons=sys.thermal_mesons)

    # Totals accumulate matter, then leptons, then radiation, in that order.
    P_total, e_total, s_total = matter.P, matter.eps, matter.s
    mu_dot_n = matter.mu_dot_n
    mu_e = n_e = n_nu = P_leptons = 0.0
    eps_leptons = s_leptons = 0.0

    lep = _lepton_sector(spec, mu_C, mu_nue, T)
    if lep is not None:
        # Beta equilibrium: mu_e came from the beta relation and the electrons
        # (plus the trapped neutrinos) were part of the residual.
        mu_e = lep.mu_e
        n_e, n_nu = lep.electrons.n, lep.n_nue
        P_leptons = lep.P
        P_total += lep.electrons.P
        e_total += lep.electrons.e
        s_total += lep.electrons.s
        eps_leptons = lep.electrons.e
        s_leptons = lep.electrons.s
        if lep.neutrinos is not None:
            P_total += lep.neutrinos.P
            e_total += lep.neutrinos.e
            s_total += lep.neutrinos.s
            eps_leptons += lep.neutrinos.e
            s_leptons += lep.neutrinos.s
    elif spec.leptons:
        # Fixed Y_C with neutralizing leptons: n_C is already determined, so
        # mu_e is whatever gives n_e = n_C. A caller may seed it by appending
        # a guess past the end of the unknown vector.
        n_unknowns = len(unknown_names(sys))
        mu_e_guess = (x0[n_unknowns] if x0 is not None and len(x0) > n_unknowns
                      else None)
        e = electron_thermo_from_density(matter.n_C, T, mu_e_guess=mu_e_guess)
        mu_e, n_e = e.mu, e.n
        P_leptons = e.P
        P_total += e.P
        e_total += e.e
        s_total += e.s
        eps_leptons, s_leptons = e.e, e.s
    lepton_dot_n = mu_e * n_e + mu_nue * n_nu
    mu_dot_n += lepton_dot_n

    if sys.photons:
        gamma = photon_thermo(T)
        P_total += gamma.P
        e_total += gamma.e
        s_total += gamma.s

    if sys.thermal_neutrinos and T > 0:
        nu = neutrino_thermo(0.0, T, include_antiparticles=True)
        n_flavours = N_NEUTRINO_FLAVOURS - (1.0 if spec.is_fixed("L_e") else 0.0)
        P_total += n_flavours * nu.P
        e_total += n_flavours * nu.e
        s_total += n_flavours * nu.s

    # Y_C and Y_S are the ACHIEVED fractions, read off the solved state rather
    # than echoed from the targets, so a point that did not converge does not
    # claim the fraction it was asked for. Y_Le has no such reading: the
    # lepton content is not part of the matter block.
    leptons = LeptonThermo(
        mu_e=mu_e, mu_mu=mu_e, mu_nue=mu_nue,
        densities={"e-": n_e, "mu-": 0.0, "nu_e": n_nu},
        P=P_leptons, eps=eps_leptons, s=s_leptons, mu_dot_n=lepton_dot_n,
    )
    conditions = {}
    if spec.is_fixed("C"):
        conditions["Y_C"] = matter.Y_C
    if spec.is_fixed("S"):
        conditions["Y_S"] = matter.Y_S
    if spec.is_fixed("L_e"):
        conditions["Y_Le"] = spec.targets["Y_Le"]
    return EoSPoint(
        mode=spec.name, n_B=sys.n_B, T=T, conditions=conditions,
        matter=matter, leptons=leptons,
        eps=e_total, P=P_total, s=s_total,
    )


def _jacobian(sys: System):
    """The Jacobian MINPACK should use, or None for a forward difference.

    The reference path is `residual` with no Jacobian: plain NumPy/SciPy, and
    the oracle the analytic one is judged against (CLAUDE.md section 9). The
    analytic one lives in `backends/`, which section 5 makes optional, so a
    missing directory lands here rather than raising.

    An isentropic solve always takes the reference path: the entropy row and
    the temperature unknown are not differentiated analytically.
    """
    if residual_jacobian is None or not sys.analytic_jac or sys.isentropic:
        return None
    return lambda x: residual_jacobian(x, sys)


def solve(sys: System, x0=None) -> EoSPoint:
    """Solve one point of the mode `sys.spec` declares.

    Non-convergence is a return value, never an exception (CLAUDE.md section
    6): `converged` is judged on the largest residual and `error` carries it,
    so a sampler can score a bad point and move on.
    """
    if sys.isentropic and sys.spec.is_fixed("C") and sys.spec.leptons:
        raise NotImplementedError(
            "an isentropic fixed-Y_C solve with neutralizing leptons is not "
            "wired: the electrons follow from n_C only after the solve, so "
            "they would be missing from the entropy row that fixes T.")

    T_guess = sys.T if sys.T is not None else 10.0
    if x0 is None:
        x0 = default_guess(sys.spec, sys.n_B, T_guess, sys.params, sys.SnB)

    # The entropy row and the outer T unknown make the system stiffer than a
    # fixed-T solve, so it gets more evaluations and a looser gate. Both
    # numbers are the model's originals rather than a re-derivation.
    maxit = 3000 if sys.isentropic else 2000
    tight, loose = (1e-4, 1e-3) if sys.isentropic else (1e-6, 1e-4)

    def equations(x):
        return residual(x, sys)

    jac = _jacobian(sys)
    sol = root(equations, x0, jac=jac, method='hybr', options={'maxfev': maxit})
    if not sol.success:
        sol = root(equations, x0, jac=jac, method='lm',
                   options={'maxiter': maxit})

    error = max(abs(r) for r in equations(sol.x))
    converged = (error < tight) or (sol.success and error < loose)
    point = replace(assemble(sol.x, sys, x0=x0),
                    error=error, converged=converged)

    # Bose-Einstein condensation is not implemented. Where a meson's effective
    # potential reaches its mass the ideal-gas expressions stop describing it,
    # and `solve_bose_jel` caps mu at m rather than diverging -- so the solve
    # converges happily on a saturated gas that is not the physical state. The
    # point is refused rather than returned: a wrong number that looks
    # converged is worse than no number, and CLAUDE.md section 6 wants that
    # said as a return value. Only the CONVERGED state is tested; an iterate
    # that passes through condensation on its way somewhere legitimate is not
    # the solver's problem.
    if point.converged and point.matter.condensation >= 1.0:
        point = replace(point, converged=False)
    return point


# =============================================================================
# THE NAMED MODES  (CLAUDE.md section 3)
# =============================================================================
# Each is one line of declaration plus the sectors `flags` carries. They are
# configurations of `solve` above, not separate solvers.
#
# Signatures follow the repository order: the parameters first and never
# optional, then n_B, then the fractions the mode fixes, then the species
# flags, then the temperature axis. `eos.dd2` reads the same way.

def solve_beta_eq_neutrinoless(
    par: Parameters, n_B: float, flags: SpeciesFlags,
    T: Optional[float] = 0.0, x0: Optional[np.ndarray] = None,
    SnB: Optional[float] = None
) -> EoSPoint:
    """
    Beta equilibrium with free-streaming neutrinos. Variables (n_B, T).

    Nothing but baryon number is conserved: mu_S = 0, mu_nue = 0, and mu_C is
    closed by electric neutrality together with mu_C + mu_e = 0.

    6 unknowns [σ, ω, ρ, φ, μ_B, μ_C]; rows: four field equations, n_B, and
    charge neutrality n_C = n_e.

    Giving `SnB` instead of `T` puts the isentrope in place of the isotherm
    (CLAUDE.md section 3): T becomes a seventh unknown and s/n_B = SnB the
    seventh row, and the solved temperature comes back in `result.T`. The
    entropy counted is the total, photons included where they are. SFHo
    carries the axis as a row of its own residual, so T is solved for together
    with the fields and potentials rather than by the outer 1-D bracket of
    `eos.general.tabulate.temperature_at_entropy`; the two agree to about
    5e-8 in T, the coupled solve being the tighter.

    Args:
        par: SFHo parameters
        n_B: Baryon density (fm⁻³)
        flags: active degrees of freedom
        T: Temperature (MeV); None exactly when SnB is given
        x0: warm start [σ, ω, ρ, φ, μ_B, μ_C], with T appended under SnB
        SnB: Entropy per baryon, in place of T
    """
    if SnB is not None:
        T = None
    return solve(_system(par, flags, modes.beta_eq_neutrinoless(), n_B,
                         T=T, SnB=SnB), x0=x0)


def solve_fixed_yc(
    par: Parameters, n_B: float, Y_C: float, flags: SpeciesFlags,
    T: float = 0.0, x0: Optional[np.ndarray] = None, leptons: bool = False
) -> EoSPoint:
    """
    Fixed non-leptonic charge fraction Y_C. Variables (n_B, Y_C, T).

    6 unknowns [σ, ω, ρ, φ, μ_B, μ_C]; the charge row is n_C = Y_C n_B, and
    n_C is the TOTAL, thermal meson gas included (CLAUDE.md section 2).

    `leptons=False` leaves the matter electrically charged, which is what a
    mixed-phase construction needs per pure phase; True adds the neutralizing
    electrons, which enter no equation because n_C is already pinned -- mu_e is
    whatever gives n_e = n_C.

    Args:
        par: SFHo parameters
        n_B: Baryon density (fm⁻³)
        Y_C: Charge fraction n_C/n_B
        flags: active degrees of freedom
        T: Temperature (MeV)
        x0: warm start [σ, ω, ρ, φ, μ_B, μ_C], optionally with a seed for μ_e
            appended past the end
        leptons: add neutralizing electrons with n_e = n_C
    """
    return solve(_system(par, flags, modes.fixed_YC(Y_C, leptons=leptons),
                         n_B, T=T), x0=x0)


def solve_fixed_yc_ys(
    par: Parameters, n_B: float, Y_C: float, Y_S: float, flags: SpeciesFlags,
    T: float = 0.0, x0: Optional[np.ndarray] = None, leptons: bool = False
) -> EoSPoint:
    """
    Fixed charge AND strangeness fractions. Variables (n_B, Y_C, Y_S, T).

    7 unknowns [σ, ω, ρ, φ, μ_B, μ_C, μ_S]: holding Y_S makes μ_S an unknown
    rather than the zero of strangeness equilibrium. Y_C = 0.5, Y_S = 0 is
    symmetric nuclear matter, for heavy-ion comparisons.

    Args:
        par: SFHo parameters
        n_B: Baryon density (fm⁻³)
        Y_C: Charge fraction n_C/n_B
        Y_S: Strangeness fraction n_S/n_B, S = +1 per s quark
        flags: active degrees of freedom (hyperons wanted, or Y_S is
            unreachable and mu_S is unconstrained)
        T: Temperature (MeV)
        x0: warm start [σ, ω, ρ, φ, μ_B, μ_C, μ_S]
        leptons: add neutralizing electrons with n_e = n_C
    """
    return solve(_system(par, flags,
                         modes.fixed_YC_YS(Y_C, Y_S, leptons=leptons),
                         n_B, T=T), x0=x0)


def solve_beta_eq_neutrino_trapped(
    par: Parameters, n_B: float, Y_Le: float, flags: SpeciesFlags,
    T: Optional[float] = 0.0, x0: Optional[np.ndarray] = None,
    SnB: Optional[float] = None
) -> EoSPoint:
    """
    Beta equilibrium with trapped neutrinos. Variables (n_B, Y_Le, T).

    7 unknowns [σ, ω, ρ, φ, μ_B, μ_C, μ_nue]: the electron lepton family is
    conserved at Y_Le = (n_e + n_nue)/n_B, so μ_nue is an unknown rather than
    zero and beta equilibrium reads μ_C + μ_e = μ_nue.

    Giving `SnB` instead of `T` puts the isentrope in place of the isotherm
    (CLAUDE.md section 3): T joins as an eighth unknown and s/n_B = SnB as an
    eighth row, and the solved temperature comes back in `result.T`.

    Args:
        par: SFHo parameters
        n_B: Baryon density (fm⁻³)
        Y_Le: Electron lepton fraction (n_e + n_nue)/n_B
        flags: active degrees of freedom
        T: Temperature (MeV); None exactly when SnB is given
        x0: warm start [σ, ω, ρ, φ, μ_B, μ_C, μ_nue], T appended under SnB
        SnB: Entropy per baryon, in place of T
    """
    if SnB is not None:
        T = None
    return solve(_system(par, flags, modes.beta_eq_neutrino_trapped(Y_Le),
                         n_B, T=T, SnB=SnB), x0=x0)


def solve_mode(
    par: Parameters, n_B: float, flags: SpeciesFlags, spec: ModeSpec,
    T: Optional[float] = None, SnB: Optional[float] = None,
    x0: Optional[np.ndarray] = None
) -> EoSPoint:
    """One point of an arbitrary mode, given as a `ModeSpec` rather than a name.

    The named modes above are the readable vocabulary and are what a notebook
    calls; this is the same solve with the declaration passed in, which is what
    a driver sweeping a grid of modes wants (`table.build_table`). Exactly one
    of T and SnB is given, per CLAUDE.md section 3's rule that an entropy per
    baryon may stand in for a temperature anywhere.
    """
    if (T is None) == (SnB is None):
        raise ValueError("exactly one of T / SnB must be given")
    return solve(_system(par, flags, spec, n_B, T=T, SnB=SnB), x0=x0)


def _system(par, flags, spec, n_B, T=None, SnB=None):
    """The `System` a named mode hands to `solve`.

    One place where species flags become the sector booleans the residual
    reads, and the one place that checks the parametrization actually couples
    the sectors the flags switched on (CLAUDE.md §4; see species.py).
    """
    check_couplings(par, flags)
    particles = active_baryons(flags)
    return System(
        params=par, particles=particles, spec=spec,
        n_B=n_B, T=T, SnB=SnB,
        thermal_mesons=flags.thermal_mesons, photons=flags.photons,
        thermal_neutrinos=flags.thermal_neutrinos,
        constants=species_constants(particles, par),
    )


# =============================================================================
# WARM START
# =============================================================================
def warm_start(result: EoSPoint, spec: ModeSpec,
               isentropic: bool = False) -> np.ndarray:
    """The seed the next point of a sweep starts from: this point's unknowns.

    Read off the mode declaration, in the order `unknown_names` documents, so
    there is no second list of layouts to drift out of step with the residual.

    Args:
        result: the previous converged point
        spec: the mode being swept — it decides which potentials are unknowns
        isentropic: True when T is an unknown rather than an input
    """
    m = result.matter
    x = [m.fields["sigma"], m.fields["omega"], m.fields["rho"],
         m.fields["phi"], m.mu_B, m.mu_C]
    if spec.is_fixed("S"):
        x.append(m.mu_S)
    if spec.is_fixed("L_e"):
        x.append(result.leptons.mu_nue)
    if isentropic:
        x.append(result.T)
    return np.array(x)


# =============================================================================
# SELF-TEST
# =============================================================================
if __name__ == "__main__":

    par, n_B, T = Parameters.named("SFHo_2fam_phi"), 0.16, 10.0
    nucleons = SpeciesFlags()
    hyperons = SpeciesFlags(hyperons=True)

    print("SFHo EOS Solvers Test")
    print("=" * 60)
    print(f"\nTest at n_B = {n_B} fm^-3, T = {T} MeV")

    r = solve_beta_eq_neutrinoless(par, n_B, nucleons, T=T)
    print("\nbeta eq, nucleons        "
          f"converged={r.converged}  Y_C={r.matter.Y_C:.4f}  P={r.P:.2f}")
    r = solve_beta_eq_neutrinoless(par, 0.32, hyperons, T=T)
    print("beta eq, +hyperons       "
          f"converged={r.converged}  Y_C={r.matter.Y_C:.4f}  Y_S={r.matter.Y_S:.4f}")
    r = solve_fixed_yc(par, n_B, 0.3, nucleons, T=T)
    print("fixed Y_C = 0.3          "
          f"converged={r.converged}  Y_C={r.matter.Y_C:.4f}  P={r.P:.2f}")
    r = solve_fixed_yc_ys(Parameters.named("SFHoY_Fortin"), 0.32, 0.4, 0.1,
                          hyperons, T=T)
    print("fixed Y_C, Y_S           "
          f"converged={r.converged}  Y_S={r.matter.Y_S:.4f}  mu_S={r.matter.mu_S:.2f}")
    r = solve_beta_eq_neutrino_trapped(par, n_B, 0.4, nucleons, T=50.0)
    print("trapped, Y_Le = 0.4      "
          f"converged={r.converged}  mu_nue={r.leptons.mu_nue:.2f}  "
          f"n_nu={r.leptons.densities['nu_e']:.3e}")
    r = solve_beta_eq_neutrinoless(par, n_B, nucleons, SnB=1.0)
    print("isentropic, S/A = 1      "
          f"converged={r.converged}  T={r.T:.3f} MeV  P={r.P:.2f}")
    print("\n" + "=" * 60)
