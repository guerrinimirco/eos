"""
mixed/thermodynamics.py
=======================
Quantities of the mixture computed FROM the state.

*Internal module.* Driven by `eos.mixed.solver`.

Everything here takes potentials, phase blocks, the volume fraction chi, the
locality parameter eta and T, and returns thermodynamic numbers. Which
equilibrium produced that state — which charges are conserved, what closes the
system — is not represented in this file at all: that is the boundary CLAUDE.md
section 5 draws between a model's thermodynamics and its solver, seen from
inside the composite engine.

The weighting convention, used identically for P, eps, s and sum mu_i n_i:
the two matter phases are volume-averaged with weights (1-chi, chi); the local
lepton populations carry weight eta and are themselves volume-averaged; the
global lepton population carries weight 1-eta and is uniform; photons (when
`SpeciesFlags.photons` is set) and any neutrino population are uniform across
the whole mixture and are counted once.
The one exception is the pressure, which is uniform in equilibrium and is
therefore read off the hadronic phase (the phases are equal by the
mechanical-equilibrium row) plus the phase-common parts.

Units are fm-based throughout: densities in fm^-3, potentials in MeV, P and
eps in MeV/fm^3.
"""
from dataclasses import dataclass

from eos.general.thermodynamics_leptons import (
    electron_thermo, muon_thermo, photon_thermo,
)


@dataclass(frozen=True)
class LeptonDomain:
    """The negatively-charged leptons sharing one domain of the eta split.

    `n`, `P`, `e`, `s` are the electron plus (if enabled) muon totals;
    `mu_dot_n` is mu_e n_e + mu_mu n_mu, and `kappa` = dn/dmu_e, which is what
    the Jacobian needs (mu_mu tracks mu_e one-for-one, so the muon
    susceptibility simply adds).
    """
    n: float = 0.0
    P: float = 0.0
    e: float = 0.0
    s: float = 0.0
    mu_dot_n: float = 0.0
    n_e: float = 0.0
    n_mu: float = 0.0


def charged_leptons(mu_e, T, muons, mu_nue=0.0):
    """Electron (+muon) thermodynamics in one domain of the eta split.

    Muons are in equilibrium with electrons at mu_mu = mu_e - mu_nue: mu_nue
    is zero for free-streaming neutrinos, and nonzero when the electron
    neutrinos carry a chemical potential while the muon family streams —
    matching eos/dd2/solver.py.
    """
    e = electron_thermo(mu_e, T)
    if not muons:
        return LeptonDomain(n=e.n, P=e.P, e=e.e, s=e.s,
                            mu_dot_n=mu_e * e.n, n_e=e.n)
    mu_mu = mu_e - mu_nue
    m = muon_thermo(mu_mu, T)
    return LeptonDomain(
        n=e.n + m.n, P=e.P + m.P, e=e.e + m.e, s=e.s + m.s,
        mu_dot_n=mu_e * e.n + mu_mu * m.n, n_e=e.n, n_mu=m.n)


def assemble(chi, eta, th_H, th_Q, L_H, L_Q, G, nu, mu_nue=0.0, T=0.0,
             photons=False):
    """The totals of the mixture: (P, eps, s, sum_i mu_i n_i).

    chi, eta     : quark volume fraction and locality parameter
    th_H, th_Q   : the two matter `PhaseThermo` blocks
    L_H, L_Q, G  : the eta-split `LeptonDomain` populations (zero-valued
                   domains when a population is absent)
    nu           : the neutrino block, or None when none is tracked
    mu_nue       : the neutrino potential entering sum mu_i n_i

    photons      : the phase-common radiation gas (CLAUDE.md section 4).
                   No adapter's `thermo` — the surface this assembles from —
                   adds a photon gas (`eos.mixed.adapters`), so the single
                   term here is the whole of it. A phase's `wing_sweep` is
                   the other path and DOES carry the caller's photons; see
                   `eos.mixed.species` for why the two rules differ.

    Weighted as the module docstring states; photons enter at T > 0 with
    mu = 0, so they contribute to P, eps and s but not to sum mu_i n_i.
    """
    ph = photon_thermo(T) if (photons and T > 0.0) else None
    P_g, e_g, s_g = (ph.P, ph.e, ph.s) if ph else (0.0, 0.0, 0.0)
    P_nu, e_nu, s_nu = (nu.P, nu.e, nu.s) if nu else (0.0, 0.0, 0.0)

    def avg_local(attr):
        return (1.0 - chi) * getattr(L_H, attr) + chi * getattr(L_Q, attr)

    # Pressure is uniform, so it is read off either phase (they are equal by
    # the mechanical-equilibrium row) plus the phase-common parts.
    P_total = (th_H.P + eta * L_H.P + (1.0 - eta) * G.P + P_nu + P_g)
    eps_total = ((1.0 - chi) * th_H.eps + chi * th_Q.eps
                 + eta * avg_local("e") + (1.0 - eta) * G.e + e_nu + e_g)
    s_total = ((1.0 - chi) * th_H.s + chi * th_Q.s
               + eta * avg_local("s") + (1.0 - eta) * G.s + s_nu + s_g)

    # sum_i mu_i n_i over both phases and every lepton species, weighted
    # exactly as eps and s are. An absent lepton population is a zero-valued
    # domain, so its term adds exactly 0.0.
    mu_dot_n = (1.0 - chi) * th_H.mu_dot_n + chi * th_Q.mu_dot_n
    mu_dot_n += eta * avg_local("mu_dot_n") + (1.0 - eta) * G.mu_dot_n
    if nu is not None:
        mu_dot_n += mu_nue * nu.n
    return P_total, eps_total, s_total, mu_dot_n


def euler_residual(P, eps, s, mu_dot_n, T):
    """Relative defect of the Euler relation eps + P = T s + sum_i mu_i n_i.

    For the mixture this is not an algebraic identity — every weighting error
    shows up in it — so `eos.mixed.solver` asserts it on every solved point
    (CLAUDE.md section 8).
    """
    return (eps + P - T * s - mu_dot_n) / eps
