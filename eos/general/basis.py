"""Changes of basis between conserved charges and species.

Matter in this library is bookkept in the conserved-charge basis (B, C, S) and
solved in a species basis -- quark flavours for a quark phase, baryons for a
hadronic one. The algebra connecting the two is trivial and therefore easy to
restate; restating it is how a sign convention drifts between two models that
are supposed to agree. It is written once, here, and derived from the quantum
numbers in :mod:`eos.general.particles` rather than from literal fractions, so
there is exactly one place in the repository where a charge is declared.

Conventions (CLAUDE.md section 2):

* **C** is the electric charge of strongly-interacting matter only -- baryons,
  quarks and charged mesons. Leptons are excluded. Total electric neutrality,
  n_C = n_e + n_mu, is a separate condition a mode may or may not impose.
* **S = +1 per s quark**, so the s quark has S = +1, Lambda has S = +1 and Xi
  has S = +2. This is the OPPOSITE of the PDG convention and is used
  throughout.
* Species potentials are *derived*, never independent unknowns:

      mu_i = B_i mu_B + C_i mu_C + S_i mu_S

  which fixes the sign of mu_C through mu_C = mu_p - mu_n, so beta equilibrium
  reads mu_C + mu_e = 0.

Units are whatever the caller uses: every map here is linear and carries the
input units straight through. Densities in fm^-3 give charge densities in
fm^-3; potentials in MeV give potentials in MeV.
"""
from __future__ import annotations

from eos.general.particles import Up, Down, Strange, get_particle

#: The three light flavours, in the fixed order used by every quark map here.
QUARK_FLAVOURS = (Up, Down, Strange)


def charges_of(particle):
    """(B, C, S) of one species, in this repository's sign convention.

    Accepts a `Particle` or its name, so a caller holding either can ask.
    """
    if isinstance(particle, str):
        found = get_particle(particle)
        if found is None:
            raise KeyError(f"unknown species {particle!r}")
        particle = found
    return particle.baryon_no, particle.charge, particle.strangeness


# --------------------------------------------------------------------------
# Densities: species -> conserved charges
# --------------------------------------------------------------------------

def charges_from_densities(densities):
    """(n_B, n_C, n_S) summed over any set of species.

    `densities` maps species name to number density. Leptons are ignored: C is
    the charge of strongly-interacting matter, and a lepton carries no B or S
    either, so including them would corrupt all three sums rather than just
    one. Impose neutrality separately.

        n_B = sum_i B_i n_i,   n_C = sum_i C_i n_i,   n_S = sum_i S_i n_i
    """
    n_B = 0.0
    n_C = 0.0
    n_S = 0.0
    for name, n in densities.items():
        particle = get_particle(name)
        if particle is None:
            raise KeyError(f"unknown species {name!r}")
        if particle.is_lepton:
            continue
        B, C, S = charges_of(particle)
        n_B += B * n
        n_C += C * n
        n_S += S * n
    return n_B, n_C, n_S


def quark_charges(n_u, n_d, n_s):
    """(n_B, n_C, n_S) of a quark phase from its flavour densities.

        n_B = (n_u + n_d + n_s) / 3
        n_C = (2 n_u - n_d - n_s) / 3
        n_S = n_s

    The coefficients come from the quark quantum numbers, so this cannot drift
    away from the hadronic bookkeeping that uses the same table.
    """
    n_B = 0.0
    n_C = 0.0
    n_S = 0.0
    for n, flavour in zip((n_u, n_d, n_s), QUARK_FLAVOURS):
        B, C, S = charges_of(flavour)
        n_B += B * n
        n_C += C * n
        n_S += S * n
    return n_B, n_C, n_S


# --------------------------------------------------------------------------
# Potentials: conserved charges <-> species
# --------------------------------------------------------------------------

def species_potential(particle, mu_B, mu_C=0.0, mu_S=0.0):
    """mu_i = B_i mu_B + C_i mu_C + S_i mu_S for one species.

    This is the whole content of "species potentials are derived": a species
    has no chemical potential of its own, only the projection of the conserved
    ones onto its quantum numbers. Applied to the nucleons it gives
    mu_p = mu_B + mu_C and mu_n = mu_B, hence mu_C = mu_p - mu_n.
    """
    B, C, S = charges_of(particle)
    return B * mu_B + C * mu_C + S * mu_S


def quark_potentials(mu_B, mu_C=0.0, mu_S=0.0):
    """(mu_u, mu_d, mu_s) from the conserved-charge potentials.

        mu_u = mu_B/3 + 2 mu_C/3
        mu_d = mu_B/3 -   mu_C/3
        mu_s = mu_B/3 -   mu_C/3 + mu_S

    With S = +1 on the s quark, mu_S enters mu_s with a PLUS sign; under the
    PDG convention it would enter with a minus, which is the single most
    likely place for a sign to be lost.
    """
    return tuple(species_potential(flavour, mu_B, mu_C, mu_S)
                 for flavour in QUARK_FLAVOURS)


def charge_potentials_from_quarks(mu_u, mu_d, mu_s):
    """(mu_B, mu_C, mu_S) from the quark flavour potentials -- the inverse of
    :func:`quark_potentials`.

        mu_C = mu_u - mu_d
        mu_S = mu_s - mu_d
        mu_B = mu_u + 2 mu_d

    The three flavours span the three charges, so the map is exactly
    invertible; a solver may work in whichever basis conditions better and
    report the other.
    """
    mu_C = mu_u - mu_d
    mu_S = mu_s - mu_d
    mu_B = mu_u + 2.0 * mu_d
    return mu_B, mu_C, mu_S


def baryon_potentials(mu_B, mu_C=0.0, mu_S=0.0, species=None):
    """{name: mu_i} for a set of baryons, from the conserved potentials.

    `species` is a sequence of `Particle` or names; with none given the full
    octet plus the Delta quartet is returned. Useful for setting up a hadronic
    solve, where every species potential follows from three numbers.
    """
    from eos.general.particles import BARYONS_ALL

    if species is None:
        species = BARYONS_ALL
    out = {}
    for entry in species:
        name = entry if isinstance(entry, str) else entry.name
        out[name] = species_potential(entry, mu_B, mu_C, mu_S)
    return out
