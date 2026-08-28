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

from eos.general.particles import (Up, Down, Strange, get_particle,
                                   NUCLEONS, HYPERONS_OCTET, DELTAS)

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


def active_baryons(flags):
    """Ordered list of the baryon `Particle` objects active under `flags`.

    Nucleons are always present; `flags.hyperons` adds the Lambda-Sigma-Xi
    octet and `flags.deltas` the Delta quartet (CLAUDE.md section 4). Any
    object carrying those two booleans serves, so this reads a model's flag
    set without knowing which model it belongs to.
    """
    baryons = list(NUCLEONS)
    if flags.hyperons:
        baryons += list(HYPERONS_OCTET)
    if flags.deltas:
        baryons += list(DELTAS)
    return baryons


def hadronic_qn(flags):
    """(name, B, C, S) for each baryon active under `flags`.

    Strangeness is S = +1 per s-quark, so Lambda has S = +1 and Xi has S = +2.
    That is the opposite of the PDG sign and is used consistently throughout
    this repository.
    """
    return tuple((b.name, b.baryon_no, b.charge, b.strangeness)
                 for b in active_baryons(flags))


def hadronic_charges(flags, densities):
    """(n_B, n_C, n_S) of the ACTIVE BARYONS in a {name: n} map.

    The baryon counterpart of `quark_charges`, and narrower than
    `charges_from_densities`: it sums the baryons `flags` declares active and
    ignores everything else in the map, so a thermal meson gas travelling in
    the same dictionary does not enter. Use `charges_from_densities` for the
    sum over every strongly-interacting species present.

    n_C is the NON-leptonic electric charge density; total electric neutrality
    is a separate condition that also counts the leptons. `densities` and the
    result share whatever units the caller passes in, and baryons absent from
    the map contribute zero.
    """
    n_B = n_C = n_S = 0.0
    for name, B, C, S in hadronic_qn(flags):
        n = densities.get(name, 0.0)
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


def lepton_charges(n_e=0.0, n_nue=0.0, n_mu=0.0, n_numu=0.0):
    """(n_Le, n_Lmu), the lepton family number densities.

        n_Le  = n_e  + n_nue
        n_Lmu = n_mu + n_numu

    L_e and L_mu are conserved charges of the reduced basis (B, C, S, L_e,
    L_mu), so Y_Le = n_Le/n_B and Y_Lmu = n_Lmu/n_B are fractions OF THE
    SOLVED STATE, defined in every mode -- exactly as Y_C and Y_S are. A mode
    that holds one of them (`beta_eq_neutrino_trapped` holds Y_Le) constrains
    the value the state reaches; it does not create the quantity, any more
    than `fixed_YC` makes Y_C meaningless everywhere else.

    The densities are net, with antiparticles already subtracted, and the
    result carries whatever units the caller passes in. A family the model
    does not track contributes zero, and that zero is the right answer rather
    than a placeholder: with no muons there is no muon-family number to
    report, and with the lepton sector off there are no leptons at all.

    Leptons carry no strong charge and so are absent from the `n_C` the other
    maps here return (section 2). Total electric neutrality, n_C = n_e + n_mu,
    is a separate condition a mode may or may not impose.
    """
    return n_e + n_nue, n_mu + n_numu


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


# --------------------------------------------------------------------------
# Reading a state against the projection
# --------------------------------------------------------------------------
# Two screens, and they answer different questions. The first asks whether one
# solved state OBEYS the projection above; the second asks, of two runs of the
# same case, whether what moved between them is one conserved-charge potential
# that nothing determined. A potential nothing determines obeys the projection
# at every point and still lands somewhere else on the next run, so the first
# screen cannot see it -- which is why both are here.

def projection_residual(mu_i, mu_B, mu_C=0.0, mu_S=0.0):
    """How far a state's species potentials are from their own projection.

    `mu_i` maps species name to the potential the state reports. Every entry
    must satisfy mu_i = B_i mu_B + C_i mu_C + S_i mu_S, since a species has no
    potential of its own; a state that violates it has either a wrong quantum
    number or a species potential carried as an independent unknown, both of
    which this catches at one point with no second run.

    Returns (worst absolute residual, the species carrying it), in the units
    of the potentials handed in; an empty `mu_i` gives (0.0, None). Species
    absent from `eos.general.particles` raise, rather than being skipped: a
    name the table does not know is exactly the case where the projection was
    never applied.
    """
    worst, carrier = 0.0, None
    for name, mu in mu_i.items():
        expected = species_potential(name, mu_B, mu_C, mu_S)
        error = abs(float(mu) - expected)
        if error > worst:
            worst, carrier = error, name
    return worst, carrier


def undetermined_potential(shifts, charge, rtol=1.0e-6):
    """Is a set of per-species shifts one undetermined conserved potential?

    `shifts` maps species name to how much its potential moved between two
    runs of the same case; `charge` is 'B', 'C' or 'S'. Because the species
    potentials are the projection, an undetermined mu_charge moves every
    species by its OWN coefficient times one common number:

        delta mu_i = X_i * delta mu_charge,     X in (B_i, C_i, S_i)

    so dividing each shift by its coefficient must give the same delta for
    every species that carries the charge, while every species with X_i = 0
    must not have moved at all. That exact-ratio structure is what separates a
    potential the equations never pinned -- legitimate, and not a regression --
    from a physics change, which moves species in no such proportion.

    Returns (delta, None) where the pattern holds -- `delta` being the shift
    of the potential itself -- and (None, reason) where it does not.

    Scope. The algebra is the same for all three charges, but only mu_C and
    mu_S are ever undetermined in practice: mu_B is conjugate to n_B, and no
    mode of CLAUDE.md section 3 leaves n_B free.
    """
    if charge not in ("B", "C", "S"):
        raise ValueError(f"charge must be one of B, C, S; got {charge!r}")
    index = ("B", "C", "S").index(charge)

    deltas = []
    for name, shift in shifts.items():
        coefficient = charges_of(name)[index]
        if coefficient == 0.0:
            if shift != 0.0:
                return None, (f"{name} carries no {charge} and still moved "
                              f"by {shift:.3e}")
            continue
        deltas.append((name, float(shift) / coefficient))
    if not deltas:
        return None, f"no species in the set carries {charge}"

    scale = max(abs(d) for _, d in deltas)
    if scale == 0.0:
        return 0.0, None
    for name, delta in deltas:
        if abs(delta - deltas[0][1]) > rtol * scale:
            return None, (f"{name} implies d(mu_{charge}) = {delta:.6e}, "
                          f"{deltas[0][0]} implies {deltas[0][1]:.6e}")
    return deltas[0][1], None
