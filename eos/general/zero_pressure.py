"""The self-bound surface: the density where P = 0, and E/A there.

A self-bound phase ends at finite density with no crust. That endpoint is
where the pressure crosses zero, and the energy per baryon there is the number
the Bodmer-Witten hypothesis is read against: three-flavour quark matter below
the 930.4 MeV of iron is absolutely stable, two-flavour quark matter above it
leaves ordinary nuclei stable. A parameter set has to clear BOTH to be
admissible, so an inference run wants both numbers per sample.

    E/A = eps/n_B  at  P(n_B) = 0,  T = 0

is a one-dimensional root find and one read, and the Euler relation makes the
read self-checking. At T = 0, section 8's

    eps + P = sum_i mu_i n_i

evaluated at P = 0 gives eps/n_B as the Gibbs energy per baryon exactly. In
the conserved-charge basis of section 2, with beta equilibrium (mu_C + mu_e =
0) and total electric neutrality (n_C = n_e), the lepton terms cancel against
the charge term and what is left is

    E/A = mu_B + Y_S mu_S                                            (*)

so a located root that does not satisfy (*) is a root of something other than
P. `ZeroPressurePoint.identity_error` reports that residual at every point.

THE FULL IDENTITY IS THE ONE TO READ, IN EVERY MODEL. `E/A = mu_B` alone is
the special case Y_S mu_S = 0, and it is not general: on a colour-flavour
locked surface the condensate pairs equal densities at unequal masses, so mu_S
is nonzero -- 40.68 MeV for `eos.alphabag`'s default set, where mu_B alone
would miss E/A by 41 MeV. It reads correctly there only because a beta
equilibrium mode has mu_S = 0 by construction (section 3, strangeness
self-equilibrates) and a single-mu parametrization like `eos.abpr` has mu_S
vanishing identically. So there are no two conventions to reconcile, only one
identity and the cases where a term of it drops out.

This module knows no model. The locator takes the state as a CALLABLE, so it
imports nothing from a model package and nothing downstream of `general/`
(CLAUDE.md section 1), and a model whose pressure has a closed-form inverse is
free to keep using it -- `eos.abpr.mu_from_P` is not generalisable and is not
generalised here.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from eos.general.physics_constants import E_per_A_iron

#: Where the scan for a sign change of P(n_B) runs, in fm^-3. Wide enough to
#: hold the surface of every quark parametrization in this repository and to
#: stop below the density where a cutoff-regularized model leaves its domain.
N_LO_DEFAULT = 0.02
N_HI_DEFAULT = 1.5

#: How many densities the bracket scan samples before the root find, giving a
#: spacing of about 0.025 fm^-3 over the default window. The scan exists
#: because P(n_B) of a self-bound phase can cross zero more than once -- a
#: model with a first-order internal transition has two -- and the SURFACE is
#: the lowest crossing from negative to positive, which a bare two-point
#: bracket cannot distinguish from any other. It is what finds the `eos.njl`
#: surface at 0.382 fm^-3 that a plain bracket over the same interval misses.
#:
#: It is also the whole cost of the locator, one full solve per sample, and a
#: model whose every point diagonalises an 18x18 matrix feels it. Such a
#: caller passes its own window rather than paying for the wide one; the
#: `verify/` suites of `eos.njl` and `eos.ccdm` do exactly that.
N_SCAN_DEFAULT = 60


class _SolveGap(Exception):
    """The model failed to converge at a density INSIDE the bracket.

    The scan proves a sign change between two densities the model reached; it
    does not prove the model reaches every density between them, and a solve
    with a branch or pattern enumeration inside it genuinely does fail there.
    Carrying that out of the root find as an exception and turning it into a
    status at the boundary is CLAUDE.md section 6's rule applied where it is
    easy to get wrong -- the alternative, feeding the root finder a sentinel
    pressure, would fabricate a crossing that the model never produced.
    """

    def __init__(self, n_B):
        super().__init__(f"the solve did not converge at n_B = {n_B:.6f} "
                         f"fm^-3, inside a bracket whose ends both solved")
        self.n_B = n_B


@dataclass(frozen=True)
class ZeroPressurePoint:
    """The located surface, or the reason there is none.

    Non-convergence is a return value here as everywhere else (CLAUDE.md
    section 6): a parameter set with no self-bound surface is an ordinary
    outcome of a sampler's walk, not an exception. Test `ok` first; every
    other field is zero when it is False.

    n_B              the surface density [fm^-3]
    E_per_A          eps/n_B there [MeV] -- the number
    mu_B, Y_S, mu_S  the right-hand side of the identity (*), as solved
    identity_error   |E/A - (mu_B + Y_S mu_S)| / E/A, dimensionless
    P                the pressure actually reached at n_B [MeV/fm^3]
    two_flavour      the flavour content REQUESTED of the solve
    below_iron       whether E/A < E_per_A_iron

    `two_flavour` and `Y_S` are both here on purpose, because they answer
    different questions: the first is what was asked for, the second is what
    was found. A three-flavour REQUEST returns whatever strangeness the
    equilibrium populated, which is zero for a set whose surface sits below
    the s quark's threshold: read the content off Y_S, not off `two_flavour`.
    Y_S is measured by the model, through `eos.general.basis` on its own
    solved flavour densities -- never off a cached field, which is how this
    paragraph came to carry a worked example that was not true. It named
    `eos.vmit`'s default set as the case that comes apart, on a Y_S = 0 read
    from a point field three of that model's four solvers never assigned. Its
    surface is at Y_S = 0.8379 and is not two-flavour.

    `below_iron` is a FACT REPORTED, never an invariant asserted. Whether a
    set sits in the Bodmer-Witten window is a property of the set: a
    three-flavour surface below iron says strange quark matter is absolutely
    stable, a two-flavour surface below iron says ordinary nuclei would
    already have decayed and EXCLUDES the set. The same True means opposite
    things on the two arms, which is why the judgement is left to the caller
    and no `verify/` suite asserts it.
    """
    ok: bool
    message: str
    n_B: float = 0.0
    E_per_A: float = 0.0
    mu_B: float = 0.0
    Y_S: float = 0.0
    mu_S: float = 0.0
    identity_error: float = 0.0
    P: float = 0.0
    two_flavour: bool = False
    below_iron: bool = False


def locate_zero_pressure(point_at, two_flavour=False,
                         n_lo=N_LO_DEFAULT, n_hi=N_HI_DEFAULT,
                         n_scan=N_SCAN_DEFAULT):
    """Find P(n_B) = 0 and report E/A there.

    `point_at(n_B)` is the model's own solve at one density, returning

        (P, E_per_A, mu_B, Y_S, mu_S)

    in the fm-based units of CLAUDE.md section 5 -- P in MeV/fm^3, the
    energies and potentials in MeV -- or None where the solve did not
    converge, so a density the model cannot reach thins the scan instead of
    aborting it. P comes first because P is the only one the root find reads;
    the rest are read once, at the located root.

    IT IS E/A THAT COMES BACK, NOT eps, and the division is the model's on
    purpose. A solve closes n_B to its own residual, so the density that goes
    with the eps just computed is the one the solved flavour densities sum to,
    not the one that was asked for. Dividing by the requested density here
    instead put a solver residual straight into the identity below -- 1.3e-12
    for `eos.alphabag`'s three-flavour surface, against 4e-14 once the model
    divides by its own.

    The surface is taken as the LOWEST density at which P crosses from
    negative to positive. Crossings the other way are the top of a
    mechanically unstable region, not a surface, and a higher rising crossing
    belongs to a branch the phase reaches only above one.

    Returns a `ZeroPressurePoint`; test `.ok`.
    """
    grid = np.linspace(n_lo, n_hi, n_scan)

    def pressure(n_B):
        state = point_at(float(n_B))
        return None if state is None else float(state[0])

    sampled = [(float(n), pressure(n)) for n in grid]
    usable = [(n, P) for n, P in sampled if P is not None]
    if len(usable) < 2:
        return ZeroPressurePoint(
            False, f"the model converged at {len(usable)} of {n_scan} scan "
                   f"densities in [{n_lo}, {n_hi}] fm^-3; no bracket",
            two_flavour=two_flavour)

    bracket = None
    for (n_a, P_a), (n_b, P_b) in zip(usable, usable[1:]):
        if P_a < 0.0 <= P_b:
            bracket = (n_a, n_b)
            break
    if bracket is None:
        return ZeroPressurePoint(
            False, f"P(n_B) does not rise through zero anywhere in "
                   f"[{n_lo}, {n_hi}] fm^-3 (P spans "
                   f"[{min(P for _, P in usable):.4g}, "
                   f"{max(P for _, P in usable):.4g}] MeV/fm^3); this "
                   f"parameter set has no self-bound surface there",
            two_flavour=two_flavour)

    def bracketed_pressure(n_B):
        P = pressure(n_B)
        if P is None:
            raise _SolveGap(float(n_B))
        return P

    try:
        n_surface = brentq(bracketed_pressure, bracket[0], bracket[1],
                           xtol=1.0e-14, rtol=8.9e-16, maxiter=200)
    except _SolveGap as gap:
        return ZeroPressurePoint(
            False, f"P changes sign between {bracket[0]:.4f} and "
                   f"{bracket[1]:.4f} fm^-3, but {gap}",
            two_flavour=two_flavour)

    state = point_at(float(n_surface))
    if state is None:
        return ZeroPressurePoint(
            False, f"the solve failed at the located root n_B = "
                   f"{n_surface:.6f} fm^-3 after succeeding on both sides",
            two_flavour=two_flavour)

    P, E_per_A, mu_B, Y_S, mu_S = (float(v) for v in state)
    identity_error = abs(E_per_A - (mu_B + Y_S * mu_S)) / abs(E_per_A)
    return ZeroPressurePoint(
        True, "located", n_B=n_surface, E_per_A=E_per_A, mu_B=mu_B, Y_S=Y_S,
        mu_S=mu_S, identity_error=identity_error, P=P,
        two_flavour=two_flavour, below_iron=E_per_A < E_per_A_iron)
