"""
nmp.py
====================
The nuclear-matter parameters of a ZL parameter set: the forward map only.

ZL is a nucleonic energy-density functional whose six parameters exist to set
these numbers, so the map from couplings to (n_sat, E_sat, K_sat, E_sym, L_sym)
is the natural statement of what a parameter set *is*. Everything here is a
PREDICTION of the couplings: ZL imposes no saturation condition internally, so
even n_sat is found rather than declared.

The inverse map is deliberately absent and `invert_nmp` raises saying why --
see its docstring. That is CLAUDE.md section 3's rule applied to a map rather
than a mode: a gap that raises, never a silent no-op.

Nucleons only, no leptons: nuclear matter is the isolated strong sector, and
every quantity below is a property of that sector alone. `thermo_from_n` is
already leptonless, so no flag is needed to say so.
"""
from scipy.optimize import brentq

from eos.zl.parameters import Parameters
from eos.zl.thermodynamics import thermo_from_n

#: Bracket for the saturation root, in fm^-3. Wide enough for any parameter
#: set a sampler is likely to propose, narrow enough that the bracket does not
#: reach the spinodal region where P = 0 has a second root.
N_SAT_BRACKET = (0.10, 0.25)


def nuclear_matter(par, n_B, beta=0.0, T=0.0):
    """Nuclear matter at density n_B and asymmetry beta, leptonless.

    beta = 0 is symmetric matter and beta = -1 pure neutron matter; the charge
    fraction that means is Y_C = (1 - beta)/2, since for nucleons
    beta = (n_n - n_p)/n_B = 1 - 2 Y_C.
    """
    return thermo_from_n(n_B, 0.5 * (1.0 - beta), T, par)


def energy_per_baryon(par, n_B, beta=0.0, T=0.0):
    """E/A = eps/n_B - m [MeV], the binding energy per baryon.

    The rest mass subtracted follows the COMPOSITION, m_p n_p + m_n n_n, not an
    average over one nucleon mass: with the physical m_n - m_p = 1.29 MeV an
    average leaves a term linear in beta that swamps the quadratic symmetry
    term at small asymmetry. In symmetric matter the two coincide.
    """
    st = nuclear_matter(par, n_B, beta, T)
    rest = par.m_p * st.n_p + par.m_n * st.n_n
    return (st.e - rest) / st.n_B


def pressure(par, n_B, beta=0.0, T=0.0):
    """P [MeV/fm^3] of nuclear matter at (n_B, beta)."""
    return nuclear_matter(par, n_B, beta, T).P


def saturation_density(par, T=0.0, bracket=N_SAT_BRACKET):
    """n_sat [fm^-3]: the density where symmetric matter has P = 0.

    Solved as a root of P rather than as a minimum of E/A. The two are the same
    condition -- P = n^2 d(E/A)/dn -- but P is returned exactly by the
    thermodynamics, so rooting it avoids differencing E/A and costs one
    derivative less of accuracy in everything computed at n_sat.
    """
    return brentq(lambda n: pressure(par, n, 0.0, T), *bracket, xtol=1e-13)


def symmetry_energy(par, n_B, T=0.0, d=2e-3):
    """S(n) [MeV], the coefficient of beta^2 in E/A(beta) = E/A(0) + S beta^2.

    Taken as the curvature at beta = 0,

        S(n) = (1/2) d^2 (E/A) / d beta^2 |_{beta=0} ,

    by a symmetric second difference. This is the standard definition and it is
    the one Constantinou et al. quote: it returns E_sym = 30.848 MeV against
    their 30.85 and L_sym = 41.27 against their 41.26.

    A note for anyone comparing with `eos.did.nmp`, which uses a full step to
    pure neutron matter with a Richardson correction instead. That estimator
    measures the same coefficient only when the quartic term is negligible;
    here it returns 30.776 and 41.124, which is a real difference rather than
    numerical noise -- it carries beta^4 contamination that the published
    numbers do not include. DID needs the full step because its E/A difference
    at small asymmetry sits in numerical noise; ZL's does not, so ZL can take
    the definition directly. The step is stable to the fifth digit across
    d = 1e-3 to 1e-2.
    """
    e = lambda beta: energy_per_baryon(par, n_B, beta, T)
    return 0.5 * (e(d) - 2.0 * e(0.0) + e(-d)) / d ** 2


def _derivative_in_x(f, par, n_sat, order, h=0.02):
    """d^k f/dx^k at x = 1, with x = n_B/n_sat, by central differences.

    `f(par, n_B)` is any quantity of nuclear matter. The step is in x, so h is
    a fraction of saturation density; 0.02 keeps the truncation error below the
    third digit of K_sat while staying far above the solver's residual.
    """
    def at(k):
        return f(par, n_sat * (1.0 + k * h))

    if order == 1:
        return (at(1) - at(-1)) / (2.0 * h)
    if order == 2:
        return (at(1) - 2.0 * at(0) + at(-1)) / h ** 2
    if order == 3:
        return (at(2) - 2.0 * at(1) + 2.0 * at(-1) - at(-2)) / (2.0 * h ** 3)
    raise ValueError(f"derivative order {order} is not implemented")


def compute_nmp(par, T=0.0):
    """The nuclear-matter parameters of a ZL parameter set.

    Returns a dict, all in MeV except n_sat [fm^-3]:

        n_sat       saturation density, where P = 0 in symmetric matter
        E_sat       binding energy per baryon there
        K_sat       incompressibility,  9 n^2 d^2(E/A)/dn^2
        Q_sat       skewness,          27 n^3 d^3(E/A)/dn^3
        E_sym       symmetry energy at saturation
        L_sym       its slope,          3 n dS/dn
        K_sym       its curvature,      9 n^2 d^2S/dn^2

    Every one is a prediction: ZL imposes none of them. The published set of
    Constantinou et al. is pinned in `verify/run_full_check.py`.

    Derivatives in x = n_B/n_sat rather than in n_B directly, so that
    K_sat = 9 d^2(E/A)/dx^2 and L_sym = 3 dS/dx with no density factors left
    to get wrong.
    """
    n_sat = saturation_density(par, T)
    e_of_n = lambda p, n: energy_per_baryon(p, n, 0.0, T)
    s_of_n = lambda p, n: symmetry_energy(p, n, T)

    return {
        "n_sat": n_sat,
        "E_sat": energy_per_baryon(par, n_sat, 0.0, T),
        "K_sat": 9.0 * _derivative_in_x(e_of_n, par, n_sat, 2),
        "Q_sat": 27.0 * _derivative_in_x(e_of_n, par, n_sat, 3),
        "E_sym": symmetry_energy(par, n_sat, T),
        "L_sym": 3.0 * _derivative_in_x(s_of_n, par, n_sat, 1),
        "K_sym": 9.0 * _derivative_in_x(s_of_n, par, n_sat, 2),
    }


def invert_nmp(*args, **kwargs):
    """Not implemented: the ZL inversion is underdetermined as published.

    ZL carries six parameters -- a0, b0, gamma, a1, b1, gamma1 -- against the
    five nuclear-matter parameters {n_sat, E_sat, K_sat, E_sym, L_sym} that
    CLAUDE.md section 5 names for the inverse map. Five equations in six
    unknowns has a one-parameter family of solutions, so an inversion needs a
    sixth condition to close it.

    DD2 closes its isoscalar sector with a structural cross-constraint,
    f''_sigma(1) = f''_omega(1), plus one shape coefficient held at its
    published value. ZL has no counterpart in the literature: nothing in
    Constantinou et al. singles out a member of the family, and inventing one
    here would put a choice with no physical warrant inside a function whose
    callers would reasonably assume it had one.

    Two ways to close it, either of which makes this implementable:

      - impose a sixth datum, e.g. Q_sat or the effective mass, and say so in
        the signature;
      - fix one coupling (gamma1 is the natural candidate, being the least
        constrained by the isoscalar sector) and invert for the other five.

    Until one is chosen this raises rather than returning an arbitrary member
    of the family, which is CLAUDE.md section 3's rule for a gap: it says which,
    and it is never a silent no-op. The forward map, `compute_nmp`, is complete.
    """
    raise NotImplementedError(
        "eos.zl has no NMP inversion: six parameters against the five NMPs of "
        "the standard list leaves a one-parameter family, and ZL has no "
        "published closure condition. Impose a sixth datum (Q_sat or m*/m) or "
        "hold one coupling fixed, then this can be written. compute_nmp (the "
        "forward map) is available.")


def from_nmp(*args, **kwargs):
    """Not implemented; see `invert_nmp`, of which this is the constructor form."""
    return invert_nmp(*args, **kwargs)


if __name__ == "__main__":
    par = Parameters.default()
    for name, value in compute_nmp(par).items():
        print(f"  {name:<7} {value:12.5f}")
