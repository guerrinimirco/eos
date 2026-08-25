"""DID's equilibrium conditions, and the solves that close them.

`thermodynamics.py` computes quantities FROM the state; this module FINDS the
state. What separates them is the mode: the residual here knows which
conserved charges are held and which equilibrate, and it reads that from a
`ModeSpec` (`eos.general.modes`) rather than branching on strings, so the
equations are written once and the mode selects among them.

THE UNKNOWN VECTOR

    x = [sigma, omega, rho, phi, beta, Sigma^t, mu~_B, mu_C,
         (mu_S), (mu_nue), (T)]

Six of the entries are the model's own self-consistency and are present in
every mode: the four meson mean fields, the isospin asymmetry beta and the
isospin rearrangement self-energy Sigma^t. The last two are what makes DID
different from an ordinary DD-RMF -- its couplings depend on beta, and
Sigma^t shifts every effective potential by (tau_3i - beta) Sigma^t, so
neither can be evaluated from the fields alone (see `thermodynamics.py`).
Carrying them as unknowns with their defining equations as rows is what keeps
the residual an explicit function of x.

mu~_B = mu_B - Sigma^r is the KINETIC baryon potential (CLAUDE.md section 2):
the density rearrangement term is common to every species, and iterating on
the kinetic potential keeps its density circularity out of the solve. mu_C is
an unknown in every mode; what the mode changes is the row that closes it,
electric neutrality or the held Y_C. mu_S and mu_nue join it where the mode
holds strangeness or traps the electron family, and T joins where an entropy
per baryon is imposed instead of a temperature.

THE ROWS, in the order the residual assembles them:

    4  meson field equations, F_M = source_M / m_M^2
    1  isospin asymmetry,     beta = sum_i tau_3i n_i / n_B
    1  isospin rearrangement, Sigma^t = its defining sum (paper Eq. 11)
    1  baryon number,         sum_i n_i = n_B
    1  the charge row:        n_C = n_e (+ n_mu) where C equilibrates,
                              n_C = Y_C n_B where it is held
    1  strangeness,           n_S = Y_S n_B              iff S is FIXED
    1  lepton number,         (n_e + n_nue)/n_B = Y_Le   iff L_e is FIXED
    1  entropy per baryon,    s/n_B = SnB                iff the entropy axis

Every row is divided by the scale of the quantity it balances, so one
tolerance means the same thing for all of them and the shared gate of
`eos.general.solve` applies unchanged.

NON-CONVERGENCE IS A RETURN VALUE (CLAUDE.md section 6): every solve returns an
`EoSPoint` carrying `converged` and the residual it was judged on, so a
sampler can score a point in unphysical parameter space and move on.

Reference: Frohaug, Maslov, Dexheimer et al., arXiv:2511.15646.
"""
from dataclasses import dataclass, replace
from typing import NamedTuple, Optional

import numpy as np

from eos.general.basis import species_potential
from eos.general.modes import (
    ModeSpec, beta_eq_neutrinoless, beta_eq_neutrino_trapped, electron_potential,
    fixed_YC, fixed_YC_YS, muon_potential, strangeness_potential,
)
from eos.general.solve import solve_system
from eos.general.thermodynamics_leptons import (
    electron_thermo, muon_thermo, neutralizing_leptons, neutrino_thermo,
    photon_thermo,
)
from eos.did.parameters import Parameters
from eos.did.species import SpeciesFlags, active_baryons
from eos.did.thermodynamics import (
    Fields, G_NU, N_NEUTRINO_FLAVOURS, baryon_kinetics, mean_fields, species_table,
    thermal_meson_thermo, thermo_from_fields,
)

#: A typical mean field [MeV]. A field equation is field = source/m^2, so
#: dividing the gap by this puts the row near unity.
FIELD_SCALE = 30.0

#: mode name -> the fractions it consumes, for the table driver and the API.
MODE_FRACTIONS = {
    "beta_eq_neutrinoless": (),
    "beta_eq_neutrino_trapped": ("Y_Le",),
    "fixed_YC": ("Y_C",),
    "fixed_YC_YS": ("Y_C", "Y_S"),
}


# =============================================================================
# THE RECORD ONE SOLVE RETURNS
# =============================================================================
@dataclass(frozen=True)
class EoSPoint:
    """One solved thermodynamic state.

    The same record `eos.sfho` and `eos.dd2` return, with DID's two
    rearrangement self-energies and its isospin asymmetry alongside the four
    mean fields. Every baryon is carried the same way -- `composition`,
    `mu_eff_i` and `m_eff_i` run over every active species -- so a Lambda is
    read exactly as a neutron is, and neither mu_eff_i nor m_eff_i has to be
    re-derived by a consumer that has only the densities.

    The potentials are the conserved-charge basis of CLAUDE.md section 2:
    mu_i = B_i mu_B + C_i mu_C + S_i mu_S, so `mu_B` is the neutron potential
    and `mu_C` is mu_p - mu_n.
    """
    n_B: float          # fm^-3
    T: float            # MeV
    sigma: float        # MeV
    omega: float        # MeV
    rho: float          # MeV
    phi: float          # MeV
    beta: float         # sum_i tau_3i n_i / n_B, -1 in pure neutron matter
    Sigma_r: float      # MeV, density rearrangement self-energy
    Sigma_t: float      # MeV, isospin rearrangement self-energy
    mu_B: float         # MeV
    mu_C: float         # MeV (mu_p - mu_n; beta equilibrium is mu_C + mu_e = 0)
    eps: float          # MeV/fm^3 (total, incl. leptons/photons when present)
    P: float            # MeV/fm^3 (total)
    s: float            # fm^-3 (total)
    hvh_rel: float      # (eps + P - T s - sum_i mu_i n_i)/eps, diagnostics
    mu_S: float = 0.0   # MeV (0 unless strangeness is held)
    mu_e: float = 0.0   # MeV
    mu_nue: float = 0.0  # MeV (0 unless the electron family is trapped)
    n_e: float = 0.0    # fm^-3 (net)
    n_mu: float = 0.0   # fm^-3 (net)
    n_nu: float = 0.0   # fm^-3 (net electron-neutrino density; trapped only)
    n_C: float = 0.0    # fm^-3, TOTAL non-leptonic charge, meson gas included
    n_S: float = 0.0    # fm^-3, TOTAL strangeness, +1 per s quark
    composition: tuple = ()   # ((name, n_i [fm^-3]), ...)
    mu_eff_i: tuple = ()      # ((name, nu_i [MeV]), ...)
    m_eff_i: tuple = ()       # ((name, m*_i [MeV]), ...)
    Y_C: float = 0.0
    Y_S: float = 0.0
    Y_Le: float = 0.0
    #: max_j |mu*_j|/m_j over the thermal meson gas, 0 without one. At 1 the
    #: gas Bose-condenses and the ideal-gas expressions stop describing it.
    condensation: float = 0.0
    P_hadrons: float = 0.0
    P_leptons: float = 0.0
    P_photons: float = 0.0
    converged: bool = False
    error: float = 0.0

    @property
    def composition_map(self):
        return dict(self.composition)

    def n(self, name):
        """Density n_i [fm^-3] of baryon `name` (0 if not active)."""
        return self.composition_map.get(name, 0.0)

    def Y(self, name):
        """Population fraction n_i/n_B of baryon `name` (0 if absent)."""
        return self.n(name) / self.n_B if self.n_B else 0.0

    def mu_eff(self, name):
        """Effective (kinetic) potential nu_i [MeV]."""
        return dict(self.mu_eff_i)[name]

    def m_eff(self, name):
        """Effective (Dirac) mass m*_i [MeV]."""
        return dict(self.m_eff_i)[name]

    def mu(self, name):
        """Full chemical potential mu_i [MeV], derived from the charge basis."""
        return species_potential(name, self.mu_B, self.mu_C, self.mu_S)

    @property
    def Y_e(self):
        return self.n_e / self.n_B if self.n_B else 0.0

    @property
    def entropy_per_baryon(self):
        """S/A = s/n_B, the axis an isentrope is drawn along."""
        return self.s / self.n_B if self.n_B else 0.0

    @property
    def free_energy_density(self):
        """f = eps - T s [MeV/fm^3]."""
        return self.eps - self.T * self.s

    def fields(self):
        """The `thermodynamics.Fields` record of this state."""
        return Fields(sigma=self.sigma, omega=self.omega, rho=self.rho,
                      phi=self.phi, beta=self.beta, Sigma_t=self.Sigma_t,
                      n_B=self.n_B)


# =============================================================================
# THE SYSTEM
# =============================================================================
class System(NamedTuple):
    """Everything the residual needs beyond the unknown vector.

    `T` is the imposed temperature and is None exactly when `SnB` imposes an
    entropy per baryon instead, in which case T joins the unknowns.
    """
    par: Parameters
    flags: SpeciesFlags
    specs: tuple
    spec: ModeSpec
    n_B: float
    T: Optional[float] = None
    SnB: Optional[float] = None

    @property
    def isentropic(self):
        return self.SnB is not None


def _system(par, flags, spec, n_B, T=None, SnB=None):
    """The `System` a named mode hands to `solve`."""
    if flags.thermal_neutrinos and spec.is_fixed("L_e"):
        raise ValueError(
            "thermal_neutrinos are the flavours the composition does NOT "
            "track; with the electron family trapped they would double-count "
            "nu_e. Wire the remaining two flavours before enabling it.")
    return System(par=par, flags=flags, specs=species_table(flags), spec=spec,
                  n_B=n_B, T=T, SnB=SnB)


def unknown_names(sys: System):
    """The unknown vector, in order. See the module docstring."""
    names = ["sigma", "omega", "rho", "phi", "beta", "Sigma_t",
             "mu_tilde_B", "mu_C"]
    if sys.spec.is_fixed("S"):
        names.append("mu_S")
    if sys.spec.is_fixed("L_e"):
        names.append("mu_nue")
    if sys.isentropic:
        names.append("T")
    return tuple(names)


def _unpack(x, sys: System):
    """Read the unknown vector, filling in what the mode determines instead."""
    fields = Fields(sigma=x[0], omega=x[1], rho=x[2], phi=x[3],
                    beta=x[4], Sigma_t=x[5], n_B=sys.n_B)
    mu_tilde_B, mu_C = x[6], x[7]
    i = 8
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
    # A negative temperature is meaningless and the Fermi integrals do not
    # accept one; the floor keeps a wide step from leaving the domain.
    T = max(x[i], 0.1) if sys.isentropic else sys.T
    return fields, mu_tilde_B, mu_C, mu_S, mu_nue, T


# =============================================================================
# GUESSES
# =============================================================================
def default_guess(spec: ModeSpec, n_B, T, par: Parameters, flags,
                  Y_C_guess=None):
    """The cold start: a state estimated from the field equations themselves.

    The fields are their own sources evaluated on a free nucleon gas of the
    guessed composition, which is right to a few MeV over the whole density
    range and is what the Newton step needs. beta follows from the guessed
    charge fraction, Sigma^t starts at zero (it is a few MeV), and the baryon
    potential is the neutron's kinetic energy plus its vector shift.

    `Y_C_guess` defaults to the fraction the mode holds, or to 0.05 in beta
    equilibrium -- the paper's own answer there is X_p = 0.034 at saturation.
    """
    if Y_C_guess is None:
        Y_C_guess = spec.targets.get("Y_C", 0.05)
    beta = 2.0 * Y_C_guess - 1.0
    couplings = par.couplings_at(n_B, beta)
    g_sigma = couplings[("sigma", "N")][0]
    m_N = 0.5 * sum(b.mass for b in active_baryons(flags)[:2])

    # One Picard sweep on a free gas: densities from the guessed composition,
    # fields from their sources, effective mass from the scalar field.
    n_p, n_n = Y_C_guess * n_B, (1.0 - Y_C_guess) * n_B
    kF = [(3.0 * np.pi ** 2 * max(n, 1e-12)) ** (1.0 / 3.0) * 197.327
          for n in (n_p, n_n)]
    m_eff = m_N
    for _sweep in range(3):
        n_s = sum(n * m_eff / np.sqrt(k ** 2 + m_eff ** 2)
                  for n, k in zip((n_p, n_n), kF))
        sigma = mean_fields(par, (g_sigma * n_s, 0.0, 0.0, 0.0))[0]
        m_eff = max(0.1 * m_N, m_N - g_sigma * sigma)
    _sigma, omega, rho, phi = mean_fields(
        par, (g_sigma * n_s,
              couplings[("omega", "N")][0] * n_B,
              couplings[("rho", "N")][0] * (n_p - n_n),
              couplings[("phi", "N")][0] * n_B))

    nu_n = np.sqrt(kF[1] ** 2 + m_eff ** 2)
    mu_tilde_B = nu_n + (couplings[("omega", "N")][0] * omega
                         + couplings[("phi", "N")][0] * phi
                         - couplings[("rho", "N")][0] * rho)
    if spec.is_fixed("C"):
        # The Lagrange multiplier of a held charge fraction, from the parabolic
        # symmetry energy: mu_C = mu_p - mu_n = 4 S_2 beta, with S_2 ~ 32 MeV.
        mu_C = 4.0 * 32.0 * beta
    else:
        # Beta equilibrium: mu_C = -mu_e, with the electrons degenerate.
        mu_C = -np.sqrt((3.0 * np.pi ** 2 * max(Y_C_guess * n_B, 1e-12))
                        ** (2.0 / 3.0) * 197.327 ** 2 + 0.511 ** 2)

    x = [sigma, omega, rho, phi, beta, 0.0, mu_tilde_B, mu_C]
    if spec.is_fixed("S"):
        x.append(0.0)
    if spec.is_fixed("L_e"):
        x.append(100.0)
    if T is None:
        x.append(10.0)
    return np.array(x, dtype=float)


def warm_start(point: EoSPoint, spec: ModeSpec, isentropic=False):
    """The seed the next point of a sweep starts from: this point's unknowns.

    Read off the mode declaration in the order `unknown_names` documents, so
    there is no second layout to drift out of step with the residual.
    """
    x = [point.sigma, point.omega, point.rho, point.phi, point.beta,
         point.Sigma_t, point.mu_B - point.Sigma_r, point.mu_C]
    if spec.is_fixed("S"):
        x.append(point.mu_S)
    if spec.is_fixed("L_e"):
        x.append(point.mu_nue)
    if isentropic:
        x.append(point.T)
    return np.array(x, dtype=float)


# =============================================================================
# THE LEPTONS THAT ENTER THE EQUATIONS
# =============================================================================
class _Leptons(NamedTuple):
    """The leptons the residual sees, at the current unknowns.

    Only a mode where C EQUILIBRATES has any: there mu_e follows from mu_C by
    the beta relation and electric neutrality is the row that closes mu_C.
    Where Y_C is held instead, n_C is already pinned and the neutralizing
    leptons follow from it AFTER the solve, entering no equation.
    """
    mu_e: float
    mu_mu: float
    electrons: object
    muons: Optional[object]
    neutrinos: Optional[object]

    @property
    def n_charged(self):
        return self.electrons.n + (self.muons.n if self.muons else 0.0)

    @property
    def n_nue(self):
        return self.neutrinos.n if self.neutrinos else 0.0


def _lepton_sector(sys: System, mu_C, mu_nue, T):
    """Leptons at the current potentials, or None where the mode has no row."""
    if sys.spec.is_fixed("C"):
        return None
    mu_e = electron_potential(mu_C, mu_nue)
    mu_mu = muon_potential(mu_e, mu_nue)
    return _Leptons(
        mu_e=mu_e, mu_mu=mu_mu,
        electrons=electron_thermo(mu_e, T),
        muons=muon_thermo(mu_mu, T) if sys.flags.muons else None,
        neutrinos=(neutrino_thermo(mu_nue, T)
                   if sys.spec.is_fixed("L_e") else None))


def _radiation(sys: System, T):
    """(P, eps, s) of the photon gas and any untracked neutrino flavours.

    Both carry no conserved charge and enter eps, P and s alone; the thermal
    neutrinos are mu = 0 gases of the flavours the composition does not track
    (CLAUDE.md section 4).
    """
    P = eps = s = 0.0
    if sys.flags.photons:
        ph = photon_thermo(T)
        P, eps, s = ph.P, ph.e, ph.s
    if sys.flags.thermal_neutrinos and T > 0.0:
        nu = neutrino_thermo(0.0, T)
        P += N_NEUTRINO_FLAVOURS * nu.P * G_NU
        eps += N_NEUTRINO_FLAVOURS * nu.e * G_NU
        s += N_NEUTRINO_FLAVOURS * nu.s * G_NU
    return P, eps, s


# =============================================================================
# THE RESIDUAL
# =============================================================================
def residual(x, sys: System):
    """The equations that must vanish, assembled from the mode declaration.

    See the module docstring for the row order. Out-of-domain trial states
    (a collapsed effective mass, a non-finite integral) come back as a large
    finite penalty rather than an exception: the solver can back away from a
    number, not from a traceback.
    """
    fields, mu_tilde_B, mu_C, mu_S, mu_nue, T = _unpack(x, sys)
    try:
        matter = baryon_kinetics(sys.par, sys.specs, fields, mu_tilde_B, mu_C, mu_S, T)
    except (ValueError, FloatingPointError):
        return [1.0e6] * len(x)

    n_C, n_S, s_matter = matter.n_C, matter.n_S, matter.s
    if sys.flags.thermal_mesons:
        # The gas carries charge and strangeness, so it enters the charge rows
        # and not only the entropy one (CLAUDE.md section 2).
        gas = thermal_meson_thermo(sys.par, fields, mu_C, mu_S, T,
                                   thermal_mesons=True)
        n_C += gas["n_C"]
        n_S += gas["n_S"]
        s_matter += gas["s"]

    implied = mean_fields(sys.par, (matter.src_sigma, matter.src_omega,
                                    matter.src_rho, matter.src_phi))
    rows = [(fields.sigma - implied[0]) / FIELD_SCALE,
            (fields.omega - implied[1]) / FIELD_SCALE,
            (fields.rho - implied[2]) / FIELD_SCALE,
            (fields.phi - implied[3]) / FIELD_SCALE,
            fields.beta - matter.n_3 / sys.n_B,
            (fields.Sigma_t - matter.Sigma_t) / FIELD_SCALE,
            matter.n_B / sys.n_B - 1.0]

    lep = _lepton_sector(sys, mu_C, mu_nue, T)
    if sys.spec.is_fixed("C"):
        rows.append(n_C / sys.n_B - sys.spec.targets["Y_C"])
    else:
        rows.append((n_C - lep.n_charged) / sys.n_B)
    if sys.spec.is_fixed("S"):
        rows.append(n_S / sys.n_B - sys.spec.targets["Y_S"])
    if sys.spec.is_fixed("L_e"):
        rows.append((lep.electrons.n + lep.n_nue) / sys.n_B
                    - sys.spec.targets["Y_Le"])
    if sys.isentropic:
        s_total = s_matter
        if lep is not None:
            s_total += lep.electrons.s + (lep.muons.s if lep.muons else 0.0)
            s_total += lep.neutrinos.s if lep.neutrinos else 0.0
        elif sys.spec.leptons:
            # Neutralizing leptons enter no field equation, but they do carry
            # entropy, so the row that fixes T has to see them -- the same
            # population `assemble` builds after the solve. Without this the
            # solved T lands where the MATTER alone carries the requested
            # entropy, and the state comes back at the wrong one.
            _mu_e, electrons, muons = neutralizing_leptons(
                n_C, T, include_muons=sys.flags.muons)
            s_total += electrons.s + muons.s
        s_total += _radiation(sys, T)[2]
        rows.append(s_total / sys.n_B - sys.SnB)
    if not np.isfinite(rows).all():
        return [1.0e6] * len(x)
    return rows


def _scales(x, _sys=None):
    """The scale of the quantity each row balances -- all unity here.

    Every row above is already divided by its own scale: a field gap by a
    typical field, a density row by n_B, a fraction row by nothing because it
    is one. Stating that here rather than scaling twice keeps the rows
    readable and lets the shared gate of `eos.general.solve` apply unchanged.
    The system is square, so there are as many scales as unknowns.
    """
    return [1.0] * len(x)


# =============================================================================
# ASSEMBLY
# =============================================================================
def assemble(x, sys: System) -> EoSPoint:
    """The full thermodynamic state from a converged unknown vector.

    Totals accumulate matter, then leptons, then radiation, in that order:
    floating-point addition is not associative, and the regression baselines
    are stated against this order.
    """
    fields, mu_tilde_B, mu_C, mu_S, mu_nue, T = _unpack(x, sys)
    matter = baryon_kinetics(sys.par, sys.specs, fields, mu_tilde_B, mu_C, mu_S, T)
    block = thermo_from_fields(sys.par, sys.flags, fields, mu_tilde_B, mu_C,
                               mu_S, T, matter=matter)

    P, eps, s = block.P, block.eps, block.s
    mu_dot_n = block.mu_dot_n
    mu_e = n_e = n_mu = n_nu = P_leptons = 0.0

    lep = _lepton_sector(sys, mu_C, mu_nue, T)
    if lep is not None:
        mu_e, n_e = lep.mu_e, lep.electrons.n
        P_leptons = lep.electrons.P
        P += lep.electrons.P
        eps += lep.electrons.e
        s += lep.electrons.s
        mu_dot_n += mu_e * n_e
        if lep.muons is not None:
            n_mu = lep.muons.n
            P_leptons += lep.muons.P
            P += lep.muons.P
            eps += lep.muons.e
            s += lep.muons.s
            mu_dot_n += lep.mu_mu * n_mu
        if lep.neutrinos is not None:
            n_nu = lep.neutrinos.n
            P_leptons += lep.neutrinos.P
            P += lep.neutrinos.P
            eps += lep.neutrinos.e
            s += lep.neutrinos.s
            mu_dot_n += mu_nue * n_nu
    elif sys.spec.leptons:
        # fixed-Y_C with neutralizing leptons: they enter no equation, so they
        # are populated here, at the one potential that makes the total
        # electrically neutral.
        mu_e, electrons, muons = neutralizing_leptons(
            block.n_C, T, include_muons=sys.flags.muons)
        n_e, n_mu = electrons.n, muons.n
        P_leptons = electrons.P + muons.P
        P += P_leptons
        eps += electrons.e + muons.e
        s += electrons.s + muons.s
        mu_dot_n += mu_e * (n_e + n_mu)
    else:
        # Leptonless fixed-Y_C: genuinely charged matter, no lepton sector at
        # all. mu_e is reported from the beta relation as a diagnostic only.
        mu_e = electron_potential(mu_C)

    P_rad, eps_rad, s_rad = _radiation(sys, T)
    P += P_rad
    eps += eps_rad
    s += s_rad

    hvh = (eps + P - T * s - mu_dot_n) / eps if eps else 0.0
    return EoSPoint(
        n_B=sys.n_B, T=T,
        sigma=fields.sigma, omega=fields.omega, rho=fields.rho,
        phi=fields.phi, beta=fields.beta,
        Sigma_r=block.Sigma_R, Sigma_t=fields.Sigma_t,
        mu_B=block.mu_B, mu_C=mu_C, mu_S=mu_S, mu_e=mu_e, mu_nue=mu_nue,
        eps=eps, P=P, s=s, hvh_rel=hvh,
        n_e=n_e, n_mu=n_mu, n_nu=n_nu, n_C=block.n_C, n_S=block.n_S,
        composition=tuple(sorted(block.densities.items())),
        mu_eff_i=tuple(sorted(block.mu_eff_i.items())),
        m_eff_i=tuple(sorted(block.m_eff_i.items())),
        Y_C=block.Y_C, Y_S=block.Y_S,
        Y_Le=(n_e + n_nu) / sys.n_B if sys.n_B else 0.0,
        condensation=block.condensation,
        P_hadrons=block.P, P_leptons=P_leptons, P_photons=P_rad)


# =============================================================================
# THE SOLVE
# =============================================================================
def solve(sys: System, x0=None) -> EoSPoint:
    """Solve one point of the mode `sys.spec` declares.

    Non-convergence is a return value (CLAUDE.md section 6): `converged` is
    judged on the largest scaled residual and `error` carries it. A
    Bose-condensed thermal meson gas is refused the same way -- the ideal-gas
    expressions stop describing it there, so a converged-looking point is
    worse than none.
    """
    cold = default_guess(sys.spec, sys.n_B, sys.T, sys.par, sys.flags)
    x, error, converged = solve_system(
        lambda v: residual(v, sys), cold if x0 is None else np.asarray(x0),
        _scales,
        x0_fallback=None if x0 is None else cold, tol=1.0e-13)

    point = replace(assemble(x, sys), error=error, converged=converged)
    if point.converged and point.condensation >= 1.0:
        point = replace(point, converged=False)
    return point


# =============================================================================
# THE NAMED MODES  (CLAUDE.md section 3)
# =============================================================================
# Each is one line of declaration plus the sectors `flags` carries; they are
# configurations of `solve`, not separate solvers. Signatures follow the
# repository order: the parameters first and never optional, then n_B, then
# the fractions the mode fixes, then the species flags, then the temperature
# axis -- the same reading order as `eos.sfho`.

def solve_beta_eq_neutrinoless(par, n_B, flags=None, T=0.0, x0=None):
    """Beta equilibrium, free-streaming neutrinos. Variables (n_B, T).

    mu_S = 0, mu_nue = 0; mu_C is closed by electric neutrality together with
    mu_C + mu_e = 0. This is the mode the neutron-star EoS is drawn in.
    """
    flags = flags if flags is not None else SpeciesFlags()
    return solve(_system(par, flags, beta_eq_neutrinoless(), n_B, T=T), x0=x0)


def solve_beta_eq_neutrino_trapped(par, n_B, Y_Le, flags=None, T=0.0, x0=None):
    """Beta equilibrium with a trapped electron family. (n_B, Y_Le, T).

    mu_nue becomes an unknown, closed by (n_e + n_nue)/n_B = Y_Le. The muon
    family stays transparent (mu_numu = 0); a trapped muon family is not
    wired, and `api.eos_point` raises rather than ignoring a Y_Lmu.
    """
    flags = flags if flags is not None else SpeciesFlags()
    return solve(_system(par, flags, beta_eq_neutrino_trapped(Y_Le), n_B, T=T),
                 x0=x0)


def solve_fixed_yc(par, n_B, Y_C, flags=None, T=0.0, leptons=False, x0=None):
    """Fixed non-leptonic charge fraction. Variables (n_B, Y_C, T).

    The simulation-table mode. `leptons=True` adds the electrons (and muons,
    if the flags enable them) that make the total electrically neutral;
    `leptons=False` leaves the matter charged, which is what a mixed-phase
    construction needs per pure phase. Strangeness still equilibrates
    (mu_S = 0).
    """
    flags = flags if flags is not None else SpeciesFlags()
    return solve(_system(par, flags, fixed_YC(Y_C, leptons=leptons), n_B, T=T),
                 x0=x0)


def solve_fixed_yc_ys(par, n_B, Y_C, Y_S, flags=None, T=0.0, leptons=False,
                      x0=None):
    """Fixed charge and strangeness. Variables (n_B, Y_C, Y_S, T).

    mu_S becomes an unknown, determined by how much strangeness was demanded.
    Y_C = 0.5, Y_S = 0 is symmetric nuclear matter for heavy-ion comparisons.

    Without a strange species there is nothing for mu_S to act on: the row
    n_S = Y_S n_B is then satisfied at Y_S = 0 for any mu_S and unsatisfiable
    otherwise, so the mode raises rather than returning a meaningless
    potential (the failure mode docs/DEFERRED.md records for other models).
    """
    flags = flags if flags is not None else SpeciesFlags()
    if not flags.has_strange_baryons:
        raise ValueError(
            "fixed_YC_YS needs a strange degree of freedom: with nucleons "
            "only, n_S = 0 identically and mu_S is not determined by any "
            "equation. Enable SpeciesFlags(hyperons=True)")
    return solve(_system(par, flags, fixed_YC_YS(Y_C, Y_S, leptons=leptons),
                         n_B, T=T), x0=x0)


def solve_mode(par, n_B, flags, spec: ModeSpec, T=None, SnB=None, x0=None):
    """One point of an arbitrary mode, given as a `ModeSpec` rather than a name.

    The named modes above are the readable vocabulary; this is the same solve
    with the declaration passed in, which is what the table driver wants.
    Exactly one of T and SnB is given -- CLAUDE.md section 3 allows an entropy
    per baryon to stand in for a temperature anywhere.
    """
    if (T is None) == (SnB is None):
        raise ValueError("exactly one of T / SnB must be given")
    return solve(_system(par, flags, spec, n_B, T=T, SnB=SnB), x0=x0)


def mode_spec(mode, conditions):
    """The `ModeSpec` a mode name and its fractions mean.

    One place where the names of CLAUDE.md section 3 turn into a declaration,
    so nothing else in the package branches on a mode string.
    """
    leptons = conditions.get("leptons", False)
    if mode == "beta_eq_neutrinoless":
        return beta_eq_neutrinoless()
    if mode == "beta_eq_neutrino_trapped":
        return beta_eq_neutrino_trapped(conditions["Y_Le"])
    if mode == "fixed_YC":
        return fixed_YC(conditions["Y_C"], leptons=leptons)
    if mode == "fixed_YC_YS":
        return fixed_YC_YS(conditions["Y_C"], conditions["Y_S"],
                           leptons=leptons)
    raise ValueError(f"unknown mode {mode!r}; expected one of "
                     f"{list(MODE_FRACTIONS)}")
