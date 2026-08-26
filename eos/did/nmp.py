"""The nuclear-matter-parameter map, forward.

Everything here needs symmetric or asymmetric nuclear matter SOLVED at a
density, which is why this sits above `solver.py` in the layer order of
CLAUDE.md section 5 and why the Delta-coupling inversion below is a free
function here rather than a constructor on `Parameters`.

Only the FORWARD map is implemented: couplings -> nuclear-matter parameters.
DID's couplings are the output of a Bayesian analysis over 18 observables,
not of an inversion of a fixed list of saturation properties, and the paper
publishes its nuclear-matter parameters as PREDICTIONS (their Table VI)
rather than as imposed constraints. The inverse map is recorded in
docs/DEFERRED.md.

THE TWO SYMMETRY ENERGIES. DID needs both, and they differ by more than
rounding here (the paper's Table VI: S - S_2 = -2.72 MeV, against +1.00 for
DD2), which is a direct consequence of the isospin dependence of the
couplings -- the binding energy is no longer close to quadratic in beta:

    S_2 = (1/2) d^2 B/dbeta^2 at beta = 0,   the quadratic coefficient
    S   = B(beta = -1) - B(beta = 0),        the full ISM-to-NM difference

with their slopes L_2, L and curvatures K_sym2, K_sym defined from each. All
density derivatives are taken with respect to x = n_B/n_0, the paper's
convention (their Table VI caption): L = 3 dS/dx, K = 9 d^2B/dx^2,
Q = 27 d^3B/dx^3, K_sym = 9 d^2S/dx^2.

Reference: Frohaug, Maslov, Dexheimer et al., arXiv:2511.15646, Tables III
and VI.
"""
from eos.general.particles import Neutron, Proton
from eos.did.species import SpeciesFlags
from eos.did.solver import solve_fixed_yc

#: The rest mass each nucleon carries out of the energy density. The binding
#: energy is B = (eps - sum_i m_i n_i)/n_B, so the mass subtracted follows the
#: COMPOSITION rather than being one average: with the physical m_n - m_p =
#: 1.29 MeV, subtracting an average instead leaves a term linear in beta,
#: -0.65 beta MeV, which is larger than the whole quadratic symmetry term
#: below beta ~ 0.04 and makes S_2 unrecoverable. In symmetric matter the two
#: definitions coincide, which is why B = -15.40 MeV is the same either way.
NUCLEON_MASS = {"p": Proton.mass, "n": Neutron.mass}

#: Nucleons only, no leptons, no gas: nuclear matter is the isolated strong
#: sector, and every parameter below is a property of that sector alone.
NUCLEAR_FLAGS = SpeciesFlags(muons=False, photons=False)


def nuclear_matter(par, n_B, beta, T=0.0, x0=None):
    """Nuclear matter at density n_B and asymmetry beta, leptonless.

    beta = 0 is ISM and beta = -1 is pure neutron matter; the charge fraction
    that means is Y_C = (1 + beta)/2, since for nucleons beta = 2 Y_C - 1.
    """
    return solve_fixed_yc(par, n_B, 0.5 * (1.0 + beta), NUCLEAR_FLAGS, T=T,
                          leptons=False, x0=x0)


def energy_per_baryon(par, n_B, beta=0.0, T=0.0):
    """B = eps/n_B - m_N [MeV], the binding energy per baryon.

    Non-convergence is a NaN rather than an exception: this feeds finite
    differences, and a caller sweeping parameters wants the derivative to come
    back not-a-number rather than to stop.
    """
    point = nuclear_matter(par, n_B, beta, T=T)
    if not point.converged:
        return float("nan")
    rest = sum(mass * point.n(name) for name, mass in NUCLEON_MASS.items())
    return (point.eps - rest) / point.n_B


def pressure(par, n_B, beta=0.0, T=0.0):
    """P [MeV/fm^3] of nuclear matter at (n_B, beta)."""
    point = nuclear_matter(par, n_B, beta, T=T)
    return point.P if point.converged else float("nan")


def symmetry_energy_quadratic(par, n_B, T=0.0, h=1.0):
    """S_2, the coefficient of beta^2 in B(beta) = B(0) + S_2 beta^2 + S_4 beta^4.

    With f(beta) = (B(beta) - B(0))/beta^2 = S_2 + S_4 beta^2, one Richardson
    step removes the quartic term: S_2 = (4 f(h/2) - f(h))/3 + O(h^4).

    Two choices here are deliberate. The step is taken on the NEUTRON-RICH
    side (beta < 0): proton-rich matter at |beta| -> 1 empties the neutron
    sector, and a chemical potential conjugate to a vanishing density is not
    pinned by any equation (the failure docs/DEFERRED.md records across
    models). And h = 1 is a full step to pure neutron matter rather than a
    small one, because B(beta) - B(0) at beta = 0.05 is 0.08 MeV against a
    binding energy of 900 MeV -- eleven digits in, where the difference is
    numerical noise. The published S_2 = 32.44 MeV comes back to 0.06 MeV
    this way; a small-step estimator does not come back at all.
    """
    B0 = energy_per_baryon(par, n_B, 0.0, T)
    f_h = (energy_per_baryon(par, n_B, -h, T) - B0) / h ** 2
    f_half = (energy_per_baryon(par, n_B, -0.5 * h, T) - B0) / (0.5 * h) ** 2
    return (4.0 * f_half - f_h) / 3.0


def symmetry_energy_full(par, n_B, T=0.0):
    """S = B(beta = -1) - B(beta = 0) [MeV], the full ISM-to-NM difference."""
    return (energy_per_baryon(par, n_B, -1.0, T)
            - energy_per_baryon(par, n_B, 0.0, T))


def _derivatives_in_x(f, par, n_0, order, h=0.02):
    """d^k f/dx^k at x = 1, with x = n_B/n_0, by central differences.

    `f(par, n_B)` is any quantity of nuclear matter. The step is in x, so h is
    a fraction of saturation density; 0.02 keeps the fourth-order truncation
    below the third digit of Q while staying far above the solver's residual.
    """
    def at(k):
        return f(par, n_0 * (1.0 + k * h))

    if order == 1:
        return (at(1) - at(-1)) / (2.0 * h)
    if order == 2:
        return (at(1) - 2.0 * at(0) + at(-1)) / h ** 2
    if order == 3:
        return (at(2) - 2.0 * at(1) + 2.0 * at(-1) - at(-2)) / (2.0 * h ** 3)
    raise ValueError(f"derivative order {order} is not implemented")


def crossover_M(par, n_B=0.11, h=0.01):
    """M(n_B) [MeV], the finite-nucleus crossover derivative (paper Eq. 53):

        M = 3 n_B d/dn_B [ 9 n^2 d^2B/dn^2 + 18 P/n ].

    Evaluated at 0.11 fm^-3 ~ 0.7 n_0, the mean density in the outer region of
    a heavy nucleus, where equations of state fitted to finite nuclei agree
    most closely with one another. The bracket is built from ISM alone.
    """
    def bracket(n):
        d2B = (energy_per_baryon(par, n * (1.0 + h))
               - 2.0 * energy_per_baryon(par, n)
               + energy_per_baryon(par, n * (1.0 - h))) / (n * h) ** 2
        return 9.0 * n ** 2 * d2B + 18.0 * pressure(par, n) / n

    return 3.0 * n_B * (bracket(n_B * (1.0 + h))
                        - bracket(n_B * (1.0 - h))) / (2.0 * n_B * h)


def compute_nmp(par, T=0.0):
    """The nuclear-matter parameters of a DID parameter set.

    Returns a dict with, all in MeV except n_0 [fm^-3] and X_p (dimensionless):

        n_0, B, K, Q            the isoscalar sector at saturation
        M                       the crossover derivative at 0.11 fm^-3
        S_2, L_2, K_sym2        the quadratic symmetry energy and its
                                density derivatives
        S, L, K_sym             the same from the full ISM-to-NM difference
        X_p_eq                  the proton fraction of beta-equilibrated
                                matter at saturation

    The paper's Table VI is exactly this list, and `verify/run_full_check.py`
    compares against it. Every number is a PREDICTION of the couplings: only
    n_0 is imposed, by the requirement P(n_0) = 0 in symmetric matter.
    """
    from eos.did.solver import solve_beta_eq_neutrinoless

    n_0 = par.n_0
    B = energy_per_baryon(par, n_0, 0.0, T)
    K = 9.0 * _derivatives_in_x(
        lambda p, n: energy_per_baryon(p, n, 0.0, T), par, n_0, 2)
    Q = 27.0 * _derivatives_in_x(
        lambda p, n: energy_per_baryon(p, n, 0.0, T), par, n_0, 3)

    S_2 = symmetry_energy_quadratic(par, n_0, T)
    L_2 = 3.0 * _derivatives_in_x(
        lambda p, n: symmetry_energy_quadratic(p, n, T), par, n_0, 1)
    K_sym2 = 9.0 * _derivatives_in_x(
        lambda p, n: symmetry_energy_quadratic(p, n, T), par, n_0, 2)

    S = symmetry_energy_full(par, n_0, T)
    L = 3.0 * _derivatives_in_x(
        lambda p, n: symmetry_energy_full(p, n, T), par, n_0, 1)
    K_sym = 9.0 * _derivatives_in_x(
        lambda p, n: symmetry_energy_full(p, n, T), par, n_0, 2)

    beta_eq = solve_beta_eq_neutrinoless(par, n_0, SpeciesFlags(muons=False),
                                         T=T)
    return dict(n_0=n_0, B=B, K=K, Q=Q, M=crossover_M(par),
                S_2=S_2, L_2=L_2, K_sym2=K_sym2,
                S=S, L=L, K_sym=K_sym,
                X_p_eq=beta_eq.Y("p") if beta_eq.converged else float("nan"))


def invert_nmp(*args, **kwargs):
    """Not implemented: the DID inversion has no published closure.

    The forward map is complete -- `compute_nmp` above -- and the paper
    publishes its nuclear-matter parameters as PREDICTIONS (arXiv:2511.15646,
    Table VI), not as constraints the couplings were fitted to. The couplings
    are the maximum-likelihood point of a Bayesian analysis over 18
    observables: hyperon potentials in two media, saturation properties,
    chiral-EFT and heavy-ion pressures. Inverting means choosing which of
    those to impose on 15 sampled numbers, and the paper makes no such choice.

    DID also has a second ambiguity the other models do not. Its symmetry
    energy comes in two inequivalent forms -- the quadratic coefficient S_2
    and the full ISM-to-neutron-matter difference S, which differ by 2.72 MeV
    at saturation here -- so even the LIST of data to impose is undetermined:
    {n_0, B, K, S, L} and {n_0, B, K, S_2, L_2} are different inversions with
    different answers, and nothing in the paper singles out one.

    Two ways to close it, either of which makes this implementable:

      - declare the target list explicitly, including WHICH symmetry energy,
        and hold the 10 or so unsampled couplings (the vector transition-zone
        shapes, the hyperon sector) at their published values;
      - invert the isoscalar sector alone against {n_0, B, K} for
        (g_sigma_N_S, g_tilde_omega_N_S, a_sigma), which is square, and report
        the isovector parameters as predictions.

    Until one is chosen this raises rather than returning a member of a family
    the caller cannot see, which is CLAUDE.md section 3's rule for a gap
    applied to a map rather than a mode: it says which, and it is never a
    silent no-op. Recorded in docs/DEFERRED.md.
    """
    raise NotImplementedError(
        "eos.did has no NMP inversion: its couplings are the maximum-"
        "likelihood point of a Bayesian analysis over 18 observables, not the "
        "solution of a fixed list of saturation data, and the model carries "
        "two inequivalent symmetry energies (S and S_2) so the list to impose "
        "is itself undetermined. Declare a target list explicitly, or invert "
        "the isoscalar sector alone, and this can be written. compute_nmp "
        "(the forward map) is available.")


def from_nmp(*args, **kwargs):
    """Not implemented; see `invert_nmp`, of which this is the constructor form."""
    return invert_nmp(*args, **kwargs)


# =============================================================================
# THE DELTA SECTOR: a coupling ratio from a chosen potential
# =============================================================================

def delta_ratios_from_potential(par, U_Delta=-50.0, x_omega=1.0, x_rho=1.0):
    """Delta coupling ratios from the Delta potential in ISM at saturation.

    arXiv:2511.15646 has no Delta isobars, so there is no published DID Delta
    coupling table; this implementation adds the quartet with the ratio scheme
    `eos.dd2` uses, and this is the constructor that fixes the scalar ratio
    from a chosen single-particle potential. In symmetric matter at saturation
    rho = 0 and Sigma^t = 0, so Eq. (12) collapses to

        U_Delta = -x_sigma g_sigmaN sigma + x_omega g_omegaN omega + Sigma^r,

    one linear equation for x_sigma. The literature range for U_Delta is
    [-100, -50] MeV; a value outside it is refused rather than extrapolated.

    Returns a new `Parameters` carrying the three ratios.
    """
    if not -100.0 <= U_Delta <= -50.0:
        raise ValueError(
            f"U_Delta = {U_Delta} MeV is outside the literature range "
            f"[-100, -50]; pass a value in range or widen it deliberately")
    ism = nuclear_matter(par, par.n_0, 0.0)
    if not ism.converged:
        raise RuntimeError("symmetric matter did not converge at n_0; the "
                           "Delta inversion has nothing to invert against")
    couplings = par.couplings_at(ism.n_B, ism.beta)
    g_sigma_N = couplings[("sigma", "N")][0]
    g_omega_N = couplings[("omega", "N")][0]
    x_sigma = (x_omega * g_omega_N * ism.omega + ism.Sigma_r - U_Delta) \
        / (g_sigma_N * ism.sigma)
    return par.with_deltas(x_sigma=x_sigma, x_omega=x_omega, x_rho=x_rho)
